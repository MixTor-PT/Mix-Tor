//! Wire encryption for MixTor (closes the plaintext real/dummy + magic leak).
//!
//! Before this layer, every frame went out as `[MAGIC][kind][len][payload]` in
//! the clear — an on-path observer read the `kind` byte (0x01 real / 0x02 dummy)
//! and separated real cells from cover without decrypting anything, defeating
//! the entire shaping defense, and matched the static `MXT1` magic as a DPI
//! signature.
//!
//! This module provides:
//!
//!  1. An **ntor-style X25519 handshake** (one-way authenticated): the client
//!     verifies it is talking to the bridge whose static public key it was
//!     configured with; the server authenticates *no one* (any client may
//!     connect, as a bridge should). It is forward-secret — each side
//!     contributes a fresh ephemeral key, so a later compromise of the server's
//!     static key does not decrypt past captures.
//!
//!  2. A **counter-nonce AEAD frame cipher** (ChaCha20-Poly1305, per direction).
//!     Each frame is fixed size, carries the real/dummy `kind` marker *inside*
//!     the ciphertext, and uses a monotonic per-direction counter as the nonce
//!     (never reused, never sent on the wire). On the wire there is no magic, no
//!     length, no kind — a real cell and a dummy cell are byte-indistinguishable.
//!
//! Primitives are all from vetted crates (`x25519-dalek`, `curve25519-elligator2`,
//! `chacha20poly1305`, `blake3`); this module only *composes* them per the ntor
//! recipe — it does not implement any primitive.
//!
//! **Handshake obfuscation (Elligator2).** The two handshake messages do NOT
//! carry raw X25519 public keys (which an observer can recognise as curve
//! points — e.g. bit 255 is always 0). Instead each side sends the **Elligator2
//! `Randomized` representative** of its ephemeral public key: a 32-byte string
//! that is indistinguishable from uniform random (the `Randomized` variant is
//! specifically hardened against the computational distinguisher that the plain
//! RFC 9380 map leaves). The peer decodes the representative back to the curve
//! point. Because ~50% of keys are representable, ephemeral generation retries
//! until it finds one. The `Randomized` map adds a low-order point to the public
//! key so the representative is uniform; the peer's *clamped* DH scalar
//! annihilates that low-order component, so key agreement is the standard X25519
//! shared secret. The handshake transcript is bound to the representatives (the
//! actual wire bytes), so both sides derive identical keys and auth.
//!
//! **Active-probing resistance (obfs4-style MARK + MAC).** Each handshake message
//! is `<repr>[|auth] | random_pad | MARK | MAC`, where MARK and MAC are keyed by
//! the bridge's identity key (derived from `B`, which only a client configured
//! with the bridge key can compute). The server validates the client's MARK/MAC
//! BEFORE doing any DH work or sending a single byte; if they don't validate it
//! drops the connection silently. So an active prober that does not know `B`
//! cannot elicit any MixTor-identifying response — the server is a black hole to
//! it. The MAC binds the epoch hour (±1 for skew) to bound replay, and the
//! random padding makes the handshake length vary. Ported from
//! `obfs4/handshake_ntor.go` (Yawning Angel, BSD-licensed).
//!
//! REMAINING REFINEMENT (not load-bearing): on probe rejection the server closes
//! immediately rather than draining into a void with a randomised timeout the
//! way obfs4 does, so the *close timing* is a weak side channel. The server
//! still emits zero identifying bytes; masking the close is a polish follow-up.

use blake3;
use chacha20poly1305::aead::Aead;
use chacha20poly1305::{ChaCha20Poly1305, Key, KeyInit, Nonce};
use curve25519_elligator2::{MapToPointVariant, MontgomeryPoint, Randomized};
use rand::rngs::OsRng;
use rand::RngCore;
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};
use subtle::ConstantTimeEq;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use x25519_dalek::{PublicKey, StaticSecret};
use zeroize::{Zeroize, Zeroizing};

// ---------------------------------------------------------------------------
// Protocol constants
// ---------------------------------------------------------------------------

/// Protocol/version tag mixed into every derivation so keys from a different
/// version (or a different protocol that happens to reuse the primitives) never
/// collide.
const PROTOID: &[u8] = b"mixtor-ntor-v1";
const CTX_KEY_EXTRACT: &str = "mixtor-ntor-v1:key_extract";
const CTX_C2S: &str = "mixtor-ntor-v1:c2s"; // client→server direction key
const CTX_S2C: &str = "mixtor-ntor-v1:s2c"; // server→client direction key
const AUTH_LABEL: &[u8] = b"mixtor-ntor-v1:server-auth";

/// Length of an Elligator2 representative (each handshake message's key portion).
pub const PUBKEY_LEN: usize = 32;
const AUTH_LEN: usize = 32;
/// Retry budget for sampling a representable ephemeral key (~50% per try, so
/// 64 tries fail with probability 2^-64).
const ELLIGATOR_RETRY_LIMIT: usize = 64;

// --- Active-probing resistance (obfs4-style mark + MAC) ---------------------
// Each handshake message is `<repr>[|auth] | random_pad | MARK | MAC`, where
// MARK and MAC are keyed by the server's identity key (which only a client that
// knows the bridge's static public key can compute). The server reads until it
// finds the expected MARK and then verifies the MAC; if neither appears it
// sends NOTHING and drops the connection, so an active prober that does not know
// the bridge key cannot elicit any MixTor-identifying response. The random pad
// makes the handshake's byte length vary; the MAC binds the current epoch hour
// (±1 for clock skew) to bound replay. Mirrors `obfs4/handshake_ntor.go`.
const MARK_LEN: usize = 16; // truncated keyed-hash MARK
const MAC_LEN: usize = 16; // truncated keyed-hash MAC
/// Hard cap on bytes read while looking for the MARK — a probe/garbage cutoff.
const MAX_HANDSHAKE_LEN: usize = 8192;
/// Max random padding added to each handshake message (length obfuscation).
const HS_MAX_PAD: usize = 2048;
const KIND_LEN: usize = 1;
const TAG_LEN: usize = 16; // Poly1305 tag

/// Plaintext `kind` markers, carried *inside* the ciphertext.
pub const KIND_REAL: u8 = 1;
pub const KIND_DUMMY: u8 = 2;

/// On-wire size of one encrypted frame whose plaintext cell is `cell_bytes`.
/// Fixed for a given deployment, so the receiver reads fixed-size chunks and
/// needs no length prefix.
pub const fn enc_frame_len(cell_bytes: usize) -> usize {
    KIND_LEN + cell_bytes + TAG_LEN
}

// ---------------------------------------------------------------------------
// Static identity keypair (the server/bridge identity)
// ---------------------------------------------------------------------------

/// The server's long-term identity keypair. Its public half is distributed to
/// clients out-of-band (analogous to an obfs4 bridge line's `cert=`).
pub struct StaticKeypair {
    secret: StaticSecret,
    public: PublicKey,
}

impl StaticKeypair {
    /// Generate a fresh random identity keypair.
    pub fn generate() -> Self {
        let secret = StaticSecret::random_from_rng(OsRng);
        let public = PublicKey::from(&secret);
        Self { secret, public }
    }

    /// Reconstruct from the 32-byte secret scalar (e.g. from a config file).
    pub fn from_secret_bytes(bytes: [u8; 32]) -> Self {
        let secret = StaticSecret::from(bytes);
        let public = PublicKey::from(&secret);
        Self { secret, public }
    }

    /// The public key clients must be configured with.
    pub fn public_bytes(&self) -> [u8; 32] {
        self.public.to_bytes()
    }

    /// The secret scalar (for persisting the identity). Handle with care.
    pub fn secret_bytes(&self) -> [u8; 32] {
        self.secret.to_bytes()
    }
}

/// Hex-encode a 32-byte key (lower-case, 64 chars) for config/CLI.
pub fn key_to_hex(bytes: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Parse a 64-char hex string into a 32-byte key. Returns `None` on bad input.
pub fn key_from_hex(s: &str) -> Option<[u8; 32]> {
    let s = s.trim();
    if s.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// Handshake errors
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum HandshakeError {
    /// Underlying transport I/O failure.
    Io(std::io::Error),
    /// Peer closed before completing the handshake.
    Closed,
    /// A Diffie-Hellman output was non-contributory (low-order point) — the
    /// peer sent a degenerate key; reject rather than proceed with a weak key.
    NonContributory,
    /// The server's authentication tag did not verify: either a man-in-the-
    /// middle without the bridge's static key, or a wrong configured key.
    BadServerAuth,
    /// The peer's handshake MARK/MAC did not validate within the byte budget:
    /// the peer could not prove knowledge of the bridge's identity key. On the
    /// server this means a likely active probe — we respond with nothing.
    ProbeRejected,
}

impl fmt::Display for HandshakeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "handshake I/O error: {e}"),
            Self::Closed => write!(f, "peer closed during handshake"),
            Self::NonContributory => write!(f, "handshake produced a non-contributory shared secret"),
            Self::BadServerAuth => write!(f, "server authentication failed (wrong key or MITM)"),
            Self::ProbeRejected => write!(f, "handshake MARK/MAC did not validate (likely active probe)"),
        }
    }
}

impl std::error::Error for HandshakeError {}

impl From<std::io::Error> for HandshakeError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

// ---------------------------------------------------------------------------
// Frame cipher
// ---------------------------------------------------------------------------

/// A one-direction AEAD frame cipher with a monotonic counter nonce.
///
/// Sealing and opening each advance the counter by one; the two endpoints stay
/// in lockstep because frames travel in order over TCP. The session keys are
/// ephemeral (per connection), so a counter never repeats under a given key.
pub struct FrameCipher {
    aead: ChaCha20Poly1305,
    counter: u64,
}

impl FrameCipher {
    fn new(mut key: [u8; 32]) -> Self {
        let aead = ChaCha20Poly1305::new(Key::from_slice(&key));
        key.zeroize();
        Self { aead, counter: 0 }
    }

    fn next_nonce(&mut self) -> Nonce {
        // 12-byte nonce = 4 zero bytes || 8-byte little-endian counter.
        let mut n = [0u8; 12];
        n[4..].copy_from_slice(&self.counter.to_le_bytes());
        self.counter = self.counter.wrapping_add(1);
        *Nonce::from_slice(&n)
    }

    /// Encrypt `(kind, cell)` into one fixed-size on-wire frame. The `kind`
    /// marker is inside the ciphertext, so real and dummy frames are
    /// indistinguishable on the wire.
    pub fn seal(&mut self, kind: u8, cell: &[u8]) -> Vec<u8> {
        let mut pt = Vec::with_capacity(KIND_LEN + cell.len());
        pt.push(kind);
        pt.extend_from_slice(cell);
        let nonce = self.next_nonce();
        let ct = self
            .aead
            .encrypt(&nonce, pt.as_ref())
            .expect("ChaCha20-Poly1305 encryption is infallible for in-range input");
        pt.zeroize();
        ct
    }

    /// Decrypt and authenticate one on-wire frame, returning `(kind, cell)`.
    /// Returns `BadServerAuth` on any tag/format failure (tamper, desync).
    pub fn open(&mut self, frame: &[u8]) -> Result<(u8, Vec<u8>), HandshakeError> {
        let nonce = self.next_nonce();
        let mut pt = self
            .aead
            .decrypt(&nonce, frame)
            .map_err(|_| HandshakeError::BadServerAuth)?;
        if pt.len() < KIND_LEN {
            pt.zeroize();
            return Err(HandshakeError::BadServerAuth);
        }
        let kind = pt[0];
        let cell = pt[KIND_LEN..].to_vec();
        pt.zeroize();
        Ok((kind, cell))
    }
}

// ---------------------------------------------------------------------------
// Key derivation (shared by both handshake roles — must be byte-identical)
// ---------------------------------------------------------------------------

/// Derive `(k_c2s, k_s2c, server_auth)` from the two DH outputs and the
/// handshake transcript. Both roles feed identical inputs (the DH outputs match
/// by the X25519 symmetry, and `B`/`x_repr`/`y_repr` are the same bytes on both
/// ends — the representatives are exactly what travelled on the wire), so both
/// compute identical keys and the same expected auth tag.
fn derive_session(
    dh1: &[u8; 32],
    dh2: &[u8; 32],
    b_pub: &[u8; 32],
    x_repr: &[u8; 32],
    y_repr: &[u8; 32],
) -> ([u8; 32], [u8; 32], blake3::Hash) {
    let mut secret_input = Vec::with_capacity(32 * 5 + PROTOID.len());
    secret_input.extend_from_slice(dh1);
    secret_input.extend_from_slice(dh2);
    secret_input.extend_from_slice(b_pub);
    secret_input.extend_from_slice(x_repr);
    secret_input.extend_from_slice(y_repr);
    secret_input.extend_from_slice(PROTOID);

    // Wrap in Zeroizing so the seed (and any of these transient secrets) is wiped
    // when it drops rather than lingering on the stack.
    let key_seed = Zeroizing::new(blake3::derive_key(CTX_KEY_EXTRACT, &secret_input));
    secret_input.zeroize();

    let k_c2s = blake3::derive_key(CTX_C2S, &key_seed[..]);
    let k_s2c = blake3::derive_key(CTX_S2C, &key_seed[..]);

    // Server-auth tag: a keyed hash over the transcript. Only a party that
    // computed the correct key_seed (i.e. that knows the bridge static secret,
    // via dh2) can produce it — this is what authenticates the server.
    let mut auth_in = Vec::with_capacity(AUTH_LABEL.len() + 96);
    auth_in.extend_from_slice(AUTH_LABEL);
    auth_in.extend_from_slice(b_pub);
    auth_in.extend_from_slice(x_repr);
    auth_in.extend_from_slice(y_repr);
    let auth = blake3::keyed_hash(&key_seed, &auth_in);

    (k_c2s, k_s2c, auth)
}

/// Sample an ephemeral X25519 private scalar whose Elligator2 `Randomized`
/// public representative exists, returning `(private_bytes, representative)`.
/// The representative is what goes on the wire (uniform-random-looking); the
/// private bytes are used locally for DH via `MontgomeryPoint::mul_clamped`.
fn gen_representable_ephemeral() -> (Zeroizing<[u8; 32]>, [u8; 32]) {
    let mut rng = OsRng;
    // The tweak randomises the representative's high bits; it does not affect
    // representability, so it is drawn once.
    let tweak = (rng.next_u32() & 0xff) as u8;
    let mut priv_bytes = Zeroizing::new([0u8; 32]);
    for _ in 0..ELLIGATOR_RETRY_LIMIT {
        rng.fill_bytes(&mut priv_bytes[..]);
        let repr: Option<[u8; 32]> = Randomized::to_representative(&priv_bytes, tweak).into();
        if let Some(repr) = repr {
            return (priv_bytes, repr);
        }
    }
    panic!("failed to sample a representable ephemeral key in {ELLIGATOR_RETRY_LIMIT} tries (bad RNG?)");
}

/// Decode a peer's Elligator2 representative back to the X25519 public key
/// (Montgomery u-coordinate bytes), or `None` if it doesn't decode.
fn decode_representative(repr: &[u8; 32]) -> Option<[u8; 32]> {
    MontgomeryPoint::from_representative::<Randomized>(repr).map(|m| m.0)
}

/// X25519 DH via the clamped Montgomery ladder: `clamp(scalar) · point`.
/// Returns the shared `u`-coordinate. The clamping annihilates the low-order
/// component the `Randomized` Elligator2 map adds to public keys.
fn dh(point: &[u8; 32], scalar: &[u8; 32]) -> Zeroizing<[u8; 32]> {
    Zeroizing::new(MontgomeryPoint(*point).mul_clamped(*scalar).0)
}

/// True if a DH output is the all-zero point (non-contributory / low-order peer
/// key) — reject rather than proceed with a degenerate shared secret.
fn dh_is_zero(out: &[u8; 32]) -> bool {
    out.iter().all(|&b| b == 0)
}

// ---------------------------------------------------------------------------
// Active-probing resistance: obfs4-style MARK + MAC framing
// ---------------------------------------------------------------------------

/// Derive the identity MAC key from the bridge's static public key. Both sides
/// can compute this (the client is configured with `B`, the server holds it),
/// but an active prober who does not know `B` cannot — which is what lets the
/// server stay silent to probes.
fn identity_mac_key(server_pub: &[u8; 32]) -> [u8; 32] {
    blake3::derive_key("mixtor-probe-v1:identity-mac", server_pub)
}

/// Keyed hash truncated to 16 bytes (used for both MARK and MAC, like obfs4's
/// HMAC-SHA256-128). blake3 keyed_hash is a PRF/MAC.
fn keyed16(key: &[u8; 32], data: &[u8]) -> [u8; 16] {
    let h = blake3::keyed_hash(key, data);
    let mut out = [0u8; 16];
    out.copy_from_slice(&h.as_bytes()[..16]);
    out
}

/// The MARK is keyed over the ephemeral representative only; it lets the server
/// locate the end of the variable-length padding by scanning for it.
fn mark_value(key: &[u8; 32], repr: &[u8]) -> [u8; 16] {
    keyed16(key, repr)
}

/// Current epoch (unix hour) as ASCII bytes, with `offset` to tolerate skew.
fn epoch_hour_bytes(offset: i64) -> Vec<u8> {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    ((secs / 3600) + offset).to_string().into_bytes()
}

/// The MAC is keyed over `data | epoch`, binding the message to the hour.
fn mac_with_epoch(key: &[u8; 32], data: &[u8], offset: i64) -> [u8; 16] {
    let mut buf = Vec::with_capacity(data.len() + 12);
    buf.extend_from_slice(data);
    buf.extend_from_slice(&epoch_hour_bytes(offset));
    keyed16(key, &buf)
}

/// Constant-time MAC check over the current hour and its neighbours (clock skew).
/// Always evaluates all three epochs so the timing is independent of the result.
fn verify_mac_epochs(key: &[u8; 32], data: &[u8], received: &[u8]) -> bool {
    let mut ok = 0u8;
    for offset in [0i64, -1, 1] {
        let expected = mac_with_epoch(key, data, offset);
        ok |= expected[..].ct_eq(received).unwrap_u8();
    }
    ok == 1
}

/// Random padding for length obfuscation: 0..=`max` uniform-random bytes.
fn random_pad(max: usize) -> Vec<u8> {
    let mut rng = OsRng;
    let len = (rng.next_u32() as usize) % (max + 1);
    let mut p = vec![0u8; len];
    rng.fill_bytes(&mut p);
    p
}

/// First index of `needle` in `haystack`, or `None`. Used to locate the MARK;
/// the MARK is not secret (the server computes its own and looks for it), so a
/// plain scan is fine here — only the MAC compare is constant-time.
fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Build a handshake message: `prefix | random_pad | MARK | MAC`. `prefix` is
/// `repr` (client) or `repr | auth` (server); the MARK is keyed over the
/// leading representative, the MAC over everything-so-far plus the epoch.
fn build_marked_message(mac_key: &[u8; 32], prefix: &[u8]) -> Vec<u8> {
    let pad = random_pad(HS_MAX_PAD);
    let mark = mark_value(mac_key, &prefix[..PUBKEY_LEN]);
    let mut msg = Vec::with_capacity(prefix.len() + pad.len() + MARK_LEN + MAC_LEN);
    msg.extend_from_slice(prefix);
    msg.extend_from_slice(&pad);
    msg.extend_from_slice(&mark);
    let mac = mac_with_epoch(mac_key, &msg, 0);
    msg.extend_from_slice(&mac);
    msg
}

/// Read a `prefix | pad | MARK | MAC` message, validating the MARK then the MAC,
/// and return the `prefix` bytes (length `prefix_len`). On any failure — MARK
/// never appears within `MAX_HANDSHAKE_LEN`, or the MAC doesn't verify — returns
/// an error so the caller can drop the connection WITHOUT replying.
async fn read_marked_message<S>(
    stream: &mut S,
    mac_key: &[u8; 32],
    prefix_len: usize,
) -> Result<Vec<u8>, HandshakeError>
where
    S: AsyncRead + Unpin,
{
    let mut buf: Vec<u8> = Vec::with_capacity(prefix_len + HS_MAX_PAD + MARK_LEN + MAC_LEN);
    let mut tmp = [0u8; 1024];
    loop {
        // Once the leading representative is present, compute the expected MARK
        // and search the post-prefix region (the padding then the MARK) for it.
        if buf.len() >= prefix_len {
            let mark = mark_value(mac_key, &buf[..PUBKEY_LEN]);
            if let Some(rel) = find_subslice(&buf[prefix_len..], &mark) {
                let mark_pos = prefix_len + rel;
                let mac_end = mark_pos + MARK_LEN + MAC_LEN;
                if buf.len() >= mac_end {
                    let received = &buf[mark_pos + MARK_LEN..mac_end];
                    if verify_mac_epochs(mac_key, &buf[..mark_pos + MARK_LEN], received) {
                        buf.truncate(prefix_len);
                        return Ok(buf);
                    }
                    return Err(HandshakeError::ProbeRejected);
                }
                // MARK found but MAC not fully read yet — fall through and read.
            }
        }
        if buf.len() >= MAX_HANDSHAKE_LEN {
            return Err(HandshakeError::ProbeRejected);
        }
        let n = stream.read(&mut tmp).await?;
        if n == 0 {
            return Err(HandshakeError::Closed);
        }
        buf.extend_from_slice(&tmp[..n]);
    }
}

// ---------------------------------------------------------------------------
// Handshake roles
// ---------------------------------------------------------------------------

/// Run the client side of the handshake over `stream`, verifying the server
/// against `server_static_pub`. Returns `(send, recv)` ciphers: `send` encrypts
/// the client→server (uplink) direction, `recv` decrypts server→client.
pub async fn client_handshake<S>(
    stream: &mut S,
    server_static_pub: &[u8; 32],
) -> Result<(FrameCipher, FrameCipher), HandshakeError>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let mac_key = identity_mac_key(server_static_pub);

    // Ephemeral client key x; the wire carries its Elligator2 representative,
    // which is indistinguishable from uniform random bytes.
    let (x_priv, x_repr) = gen_representable_ephemeral();

    // → x_repr | pad | MARK | MAC. The MARK/MAC prove we know the bridge key, so
    //   the server only answers a client that already knows `B` (anti-probing).
    let msg = build_marked_message(&mac_key, &x_repr);
    stream.write_all(&msg).await?;
    stream.flush().await?;

    // ← y_repr | auth | pad | MARK | MAC
    let prefix = read_marked_message(stream, &mac_key, PUBKEY_LEN + AUTH_LEN).await?;
    let mut y_repr = [0u8; 32];
    y_repr.copy_from_slice(&prefix[..PUBKEY_LEN]);
    let mut recv_auth = [0u8; 32];
    recv_auth.copy_from_slice(&prefix[PUBKEY_LEN..PUBKEY_LEN + AUTH_LEN]);

    // Decode the server's ephemeral point and run both DHs.
    let y_point = decode_representative(&y_repr).ok_or(HandshakeError::NonContributory)?;
    let dh1 = dh(&y_point, &x_priv); // = clamp(x)·Y  (ephemeral–ephemeral)
    let dh2 = dh(server_static_pub, &x_priv); // = clamp(x)·B  (authenticates server)
    if dh_is_zero(&dh1) || dh_is_zero(&dh2) {
        return Err(HandshakeError::NonContributory);
    }

    let (k_c2s, k_s2c, auth) =
        derive_session(&dh1, &dh2, server_static_pub, &x_repr, &y_repr);

    // Constant-time verify (blake3::Hash equality is constant-time).
    if auth != blake3::Hash::from_bytes(recv_auth) {
        return Err(HandshakeError::BadServerAuth);
    }

    Ok((FrameCipher::new(k_c2s), FrameCipher::new(k_s2c)))
}

/// Run the server side of the handshake over `stream` using our identity
/// `static_kp`. Returns `(send, recv)` ciphers: `send` encrypts the
/// server→client (downlink) direction, `recv` decrypts client→server.
///
/// The client's MARK/MAC are validated BEFORE any reply or DH work. If they
/// don't validate (an active probe, or a client without the bridge key) this
/// returns an error and the caller drops the connection having written NOTHING —
/// so a prober cannot elicit any MixTor-identifying response.
pub async fn server_handshake<S>(
    stream: &mut S,
    static_kp: &StaticKeypair,
) -> Result<(FrameCipher, FrameCipher), HandshakeError>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let b_pub_bytes = static_kp.public_bytes();
    let mac_key = identity_mac_key(&b_pub_bytes);

    // ← x_repr | pad | MARK | MAC — reject (silently) if the proof fails.
    let prefix = read_marked_message(stream, &mac_key, PUBKEY_LEN).await?;
    let mut x_repr = [0u8; 32];
    x_repr.copy_from_slice(&prefix[..PUBKEY_LEN]);
    let x_point = decode_representative(&x_repr).ok_or(HandshakeError::NonContributory)?;

    // Ephemeral server key y; the wire carries its representative.
    let (y_priv, y_repr) = gen_representable_ephemeral();
    let mut b_secret_bytes = static_kp.secret_bytes();

    let dh1 = dh(&x_point, &y_priv); // = clamp(y)·X  (ephemeral–ephemeral)
    let dh2 = dh(&x_point, &b_secret_bytes); // = clamp(b)·X  (proves we hold the static secret)
    b_secret_bytes.zeroize();
    if dh_is_zero(&dh1) || dh_is_zero(&dh2) {
        return Err(HandshakeError::NonContributory);
    }

    let (k_c2s, k_s2c, auth) =
        derive_session(&dh1, &dh2, &b_pub_bytes, &x_repr, &y_repr);

    // → y_repr | auth | pad | MARK | MAC
    let mut prefix_out = Vec::with_capacity(PUBKEY_LEN + AUTH_LEN);
    prefix_out.extend_from_slice(&y_repr);
    prefix_out.extend_from_slice(auth.as_bytes());
    let msg = build_marked_message(&mac_key, &prefix_out);
    stream.write_all(&msg).await?;
    stream.flush().await?;

    Ok((FrameCipher::new(k_s2c), FrameCipher::new(k_c2s)))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const CELL: usize = 514;

    #[tokio::test]
    async fn handshake_agrees_and_frames_round_trip_both_directions() {
        let kp = StaticKeypair::generate();
        let server_pub = kp.public_bytes();
        let (mut c, mut s) = tokio::io::duplex(4096);

        let client = tokio::spawn(async move {
            client_handshake(&mut c, &server_pub).await.map(|keys| (keys, c))
        });
        let (mut s_send, mut s_recv) = server_handshake(&mut s, &kp).await.unwrap();
        let ((mut c_send, mut c_recv), _c) = client.await.unwrap().unwrap();

        // Uplink: client seals, server opens.
        let cell: Vec<u8> = (0..CELL).map(|i| (i % 251) as u8).collect();
        let frame = c_send.seal(KIND_REAL, &cell);
        assert_eq!(frame.len(), enc_frame_len(CELL));
        let (kind, got) = s_recv.open(&frame).unwrap();
        assert_eq!(kind, KIND_REAL);
        assert_eq!(got, cell);

        // Downlink: server seals, client opens.
        let dummy = vec![0xABu8; CELL];
        let frame = s_send.seal(KIND_DUMMY, &dummy);
        let (kind, got) = c_recv.open(&frame).unwrap();
        assert_eq!(kind, KIND_DUMMY);
        assert_eq!(got, dummy);
    }

    #[tokio::test]
    async fn wrong_server_key_is_rejected() {
        let kp = StaticKeypair::generate();
        let wrong_pub = StaticKeypair::generate().public_bytes();
        let (mut c, mut s) = tokio::io::duplex(16384);
        let dur = std::time::Duration::from_secs(1);

        // A client with the wrong bridge key produces a MARK/MAC the server
        // can't match, so the server never replies — both sides fail (time out)
        // rather than the server authenticating an unknown client.
        let client = tokio::spawn(async move {
            tokio::time::timeout(dur, client_handshake(&mut c, &wrong_pub)).await
        });
        let server = tokio::time::timeout(dur, server_handshake(&mut s, &kp)).await;
        assert!(
            server.is_err() || server.unwrap().is_err(),
            "server must not accept a client using the wrong bridge key"
        );
        let client_res = client.await.unwrap();
        assert!(
            client_res.is_err() || client_res.unwrap().is_err(),
            "client using the wrong key must not complete the handshake"
        );
    }

    #[tokio::test]
    async fn active_probe_gets_no_response() {
        // The core anti-probing property: a peer that does NOT know the bridge
        // key cannot elicit any MixTor-identifying response. Send more than
        // MAX_HANDSHAKE_LEN of garbage (no valid MARK can appear) and confirm the
        // server rejects it AND wrote zero bytes back.
        let kp = StaticKeypair::generate();
        let (mut prober, mut s) = tokio::io::duplex(MAX_HANDSHAKE_LEN * 2);

        let garbage = vec![0u8; MAX_HANDSHAKE_LEN + 64];
        prober.write_all(&garbage).await.unwrap();
        prober.flush().await.unwrap();

        let res = server_handshake(&mut s, &kp).await;
        assert!(
            matches!(res, Err(HandshakeError::ProbeRejected)),
            "server must reject a probe with ProbeRejected"
        );

        // Server must have sent NOTHING — close its half and confirm EOF/empty.
        drop(s);
        let mut resp = Vec::new();
        prober.read_to_end(&mut resp).await.unwrap();
        assert!(
            resp.is_empty(),
            "server must send zero bytes to an active probe, got {}",
            resp.len()
        );
    }

    #[tokio::test]
    async fn tampered_frame_fails_to_open() {
        let kp = StaticKeypair::generate();
        let server_pub = kp.public_bytes();
        let (mut c, mut s) = tokio::io::duplex(4096);
        let client = tokio::spawn(async move {
            client_handshake(&mut c, &server_pub).await.map(|k| (k, c))
        });
        let (_s_send, mut s_recv) = server_handshake(&mut s, &kp).await.unwrap();
        let ((mut c_send, _c_recv), _c) = client.await.unwrap().unwrap();

        let cell = vec![7u8; CELL];
        let mut frame = c_send.seal(KIND_REAL, &cell);
        frame[10] ^= 0x01; // flip one ciphertext bit
        assert!(s_recv.open(&frame).is_err(), "tampered frame must not open");
    }

    #[test]
    fn same_plaintext_seals_differently_each_frame() {
        // Counter nonce → identical (kind, cell) encrypts to distinct frames,
        // so repeated cover cells don't reveal repetition on the wire.
        let mut cipher = FrameCipher::new([42u8; 32]);
        let cell = vec![0u8; CELL];
        let a = cipher.seal(KIND_DUMMY, &cell);
        let b = cipher.seal(KIND_DUMMY, &cell);
        assert_eq!(a.len(), b.len());
        assert_ne!(a, b, "counter nonce must make repeated frames differ");
    }

    #[test]
    fn real_and_dummy_frames_are_same_size() {
        // Indistinguishability by size: a real and a dummy frame are identical
        // length, and neither exposes the kind byte in the clear.
        let mut cipher = FrameCipher::new([9u8; 32]);
        let real = cipher.seal(KIND_REAL, &vec![1u8; CELL]);
        let dummy = cipher.seal(KIND_DUMMY, &vec![2u8; CELL]);
        assert_eq!(real.len(), dummy.len());
        assert_eq!(real.len(), enc_frame_len(CELL));
    }

    #[test]
    fn keypair_round_trips_through_secret_bytes() {
        let kp = StaticKeypair::generate();
        let restored = StaticKeypair::from_secret_bytes(kp.secret_bytes());
        assert_eq!(kp.public_bytes(), restored.public_bytes());
    }

    #[test]
    fn ephemeral_representatives_randomize_high_bits() {
        // A raw X25519 public key always has bit 255 = 0 (field < 2^255 − 19) —
        // a standing distinguisher. The Elligator2 `Randomized` representative
        // randomises the top two bits, so across samples the high bit must take
        // BOTH values. (This is the property that makes the handshake bytes look
        // uniform rather than like curve points.)
        let mut saw_set = false;
        let mut saw_clear = false;
        for _ in 0..64 {
            let (_priv, repr) = gen_representable_ephemeral();
            if repr[31] & 0x80 != 0 {
                saw_set = true;
            } else {
                saw_clear = true;
            }
            if saw_set && saw_clear {
                break;
            }
        }
        assert!(
            saw_set && saw_clear,
            "Elligator2 representative high bit must vary across samples"
        );
    }

    #[test]
    fn representative_decodes_back_to_a_point() {
        // Every representative we emit must decode to a curve point (round-trip),
        // otherwise the peer could not recover our key.
        for _ in 0..16 {
            let (_priv, repr) = gen_representable_ephemeral();
            assert!(
                decode_representative(&repr).is_some(),
                "a generated representative must decode to a point"
            );
        }
    }

    #[test]
    fn key_hex_round_trips() {
        let kp = StaticKeypair::generate();
        let pub_bytes = kp.public_bytes();
        let hex = key_to_hex(&pub_bytes);
        assert_eq!(hex.len(), 64);
        assert_eq!(key_from_hex(&hex), Some(pub_bytes));
        assert_eq!(key_from_hex("not-hex"), None);
        assert_eq!(key_from_hex(""), None);
    }
}
