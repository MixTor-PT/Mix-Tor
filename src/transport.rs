use crate::clumping::Packet;
use crate::composition::{CompositionConfig, MixedFrame, ShapedSession};
use crate::lab::LabLogger;
use crate::crypto::{
    client_handshake, enc_frame_len, server_handshake, FrameCipher, StaticKeypair, KIND_DUMMY,
    KIND_REAL,
};
use crate::seeds::SessionSeeds;
use crate::session_bounder::FuzzedClassThresholds;
#[cfg(test)]
use crate::session_bounder::SessionClass;
use rand::{rngs::StdRng, Rng, RngCore, SeedableRng};
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::atomic::{AtomicU16, AtomicUsize, Ordering};
use std::error::Error;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

// ---------------------------------------------------------------------------
// Uniform-cell framing with stream multiplexing.
//
// Every outbound frame on the client→server link — real or dummy — is exactly
// `CELL_BYTES` bytes, so a wire observer cannot separate real frames from
// dummies by size. A real cell carries a 4-byte header so the receiver can
// demultiplex multiple Tor circuits over one shaped wire and strip padding:
//
//   real cell:  [u16 BE stream_id][u16 BE len][data][random padding] == CELL_BYTES
//   dummy cell: [CELL_BYTES random bytes]
//
// `stream_id = STREAM_ID_SINGLE` (0) is used for legacy single-circuit mode.
// `len = 0` signals stream teardown for that stream_id (the close sentinel).
// ---------------------------------------------------------------------------
const CELL_BYTES: usize = 514;   // Tor cell size
const CELL_HEADER: usize = 4;    // u16 stream_id + u16 length prefix
const CELL_PAYLOAD: usize = CELL_BYTES - CELL_HEADER; // 510 real bytes per cell

/// Upper bound on bytes read from a source per read (the `max_read` clamp).
const DEFAULT_MAX_FRAME_LEN: usize = 64 * 1024;

/// Stream ID for single-circuit (non-mux) connections. A real cell with `len = 0`
/// is the stream-close sentinel (see `encode_close_cell`).
const STREAM_ID_SINGLE: u16 = 0x0000;

/// Encode up to `CELL_PAYLOAD` real bytes into exactly `CELL_BYTES`.
/// `stream_id` tags the cell so the receiver can demultiplex circuits.
fn encode_real_cell(stream_id: u16, data: &[u8], pad_rng: &mut impl RngCore) -> Vec<u8> {
    let n = data.len().min(CELL_PAYLOAD);
    let mut cell = vec![0u8; CELL_BYTES];
    cell[0..2].copy_from_slice(&stream_id.to_be_bytes());
    cell[2..4].copy_from_slice(&(n as u16).to_be_bytes());
    cell[4..4 + n].copy_from_slice(&data[..n]);
    pad_rng.fill_bytes(&mut cell[4 + n..]);
    cell
}

/// Encode a stream-close cell for `stream_id` (`len = 0` is the close sentinel).
fn encode_close_cell(stream_id: u16, pad_rng: &mut impl RngCore) -> Vec<u8> {
    let mut cell = vec![0u8; CELL_BYTES];
    cell[0..2].copy_from_slice(&stream_id.to_be_bytes());
    // len bytes [2..4] are 0x0000 (the close sentinel) from zero-initialisation.
    pad_rng.fill_bytes(&mut cell[4..]);
    cell
}

/// Decode a real-cell payload, returning `(stream_id, data)`.
/// An empty `data` slice means `len == 0`, i.e. a stream-close sentinel.
fn decode_real_cell(cell: &[u8]) -> (u16, &[u8]) {
    if cell.len() < CELL_HEADER {
        return (0, &[]);
    }
    let stream_id = u16::from_be_bytes([cell[0], cell[1]]);
    let n = u16::from_be_bytes([cell[2], cell[3]]) as usize;
    let end = (CELL_HEADER + n).min(cell.len());
    (stream_id, &cell[CELL_HEADER..end])
}

// ---------------------------------------------------------------------------
// No warmup — both directions are shaped from cell 0. A previous design let real
// cells bypass the clock during the Tor link handshake, but that created a
// correlated uncovered burst at session start that an adversary watching BOTH
// the ISP link and the bridge could use to link client→bridge (the classic
// ISP+end-node startup correlation). Cover dummies now flow before any real data
// arrives, so there is no unshaped window; the ~3.5 ms per-cell interval is
// negligible versus the multi-RTT Tor handshake.
// ---------------------------------------------------------------------------

/// Max time allowed for the ntor handshake before the connection is dropped.
/// The handshake is ~2 KB each way + a few curve ops, so this is generous for
/// any real link; its real job is to free server tasks held by active probes
/// that send partial data and then stall (the anti-probing read otherwise blocks
/// waiting for a MARK that never arrives).
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(15);

/// After a failed handshake the server blackholes the peer (read+discard) for a
/// random interval in this range before closing, so the close timing leaks no
/// signal about the rejection (see `blackhole`).
const BLACKHOLE_MIN_SECS: u64 = 2;
const BLACKHOLE_MAX_SECS: u64 = 30;

/// Process-wide cap on simultaneously-established server connections (each is a
/// FlowSlot on the shared emitter). A legit client uses ONE persistent wire, so
/// this is generous; it exists to bound emitter load under connection flooding.
const MAX_TOTAL_CONNECTIONS: usize = 4096;

/// Lazily-initialised global connection limiter (see `MAX_TOTAL_CONNECTIONS`).
fn global_connection_sem() -> &'static tokio::sync::Semaphore {
    static SEM: std::sync::OnceLock<tokio::sync::Semaphore> = std::sync::OnceLock::new();
    SEM.get_or_init(|| tokio::sync::Semaphore::new(MAX_TOTAL_CONNECTIONS))
}

pub async fn handle_client_connection(
    local: TcpStream,
    server: SocketAddr,
    max_read: usize,
    server_pub: [u8; 32],
) -> Result<(), Box<dyn Error + Send + Sync>> {
    // No shared emitter supplied (single-connection convenience / tests): spin up
    // a private one and DON'T pad the tail (short-lived connections shouldn't
    // block on padding). Real multi-flow deployments share an emitter and choose
    // a tail policy via the `_with_lab` entry point.
    handle_client_connection_with_lab(local, server, max_read, None, EmitterHandle::new(), TailPolicy::Off, server_pub)
        .await
}

pub async fn handle_client_connection_with_lab(
    local: TcpStream,
    server: SocketAddr,
    max_read: usize,
    lab: Option<Arc<LabLogger>>,
    emitter: EmitterHandle,
    tail: TailPolicy,
    server_pub: [u8; 32],
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let conn_id = lab.as_ref().map(|l| l.next_conn_id()).unwrap_or(0);
    let mut remote = TcpStream::connect(server).await?;
    // Disable Nagle so the departure clock's per-cell spacing reaches the wire
    // intact instead of being coalesced by the kernel (esp. the `remote` wire
    // side, which carries the shaped cells). Best-effort.
    let _ = local.set_nodelay(true);
    let _ = remote.set_nodelay(true);
    // ntor handshake on the raw stream before splitting: the client verifies the
    // bridge's static identity and derives the per-direction frame ciphers. A
    // timeout here means the server never answered (wrong key, or it silently
    // rejected us as a probe) — surface it as an error to the caller.
    let (send_cipher, recv_cipher) =
        match tokio::time::timeout(HANDSHAKE_TIMEOUT, client_handshake(&mut remote, &server_pub)).await {
            Ok(r) => r?,
            Err(_) => return Err("mixtor handshake timed out (wrong server key or no server?)".into()),
        };
    let (local_read,  local_write)  = local.into_split();
    let (remote_read, remote_write) = remote.into_split();
    let max_read = max_read.clamp(1, DEFAULT_MAX_FRAME_LEN);

    let outbound_lab = lab.clone();
    let outbound = tokio::spawn(async move {
        send_mixed_outbound(local_read, remote_write, max_read, outbound_lab, conn_id, emitter, tail, send_cipher).await
    });
    let inbound = tokio::spawn(async move {
        receive_real_inbound(remote_read, local_write, recv_cipher, lab, conn_id).await
    });

    tokio::select! {
        result = outbound => result??,
        result = inbound  => result??,
    }
    Ok(())
}

pub async fn handle_server_connection(
    client: TcpStream,
    upstream: SocketAddr,
    max_read: usize,
    static_kp: Arc<StaticKeypair>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    // No shared emitter supplied (single-connection convenience / tests): spin up
    // a private one and DON'T pad the tail (short-lived connections shouldn't
    // block on padding). Real multi-flow deployments share an emitter and choose
    // a tail policy via the `_with_lab` entry point.
    handle_server_connection_with_lab(client, upstream, max_read, None, EmitterHandle::new(), TailPolicy::Off, static_kp, None)
        .await
}

/// Read and discard everything a peer sends until it closes or a randomised
/// deadline elapses, then drop the connection. Used after a failed handshake so
/// the *close timing* doesn't reveal that the peer was rejected — to an active
/// prober the bridge looks like a generic server that simply never replies
/// (obfs4-style). The caller holds the handshake permit across this, so the
/// number of probes that can blackhole us at once is bounded by the semaphore.
async fn blackhole(mut client: TcpStream) {
    let secs = StdRng::from_entropy().gen_range(BLACKHOLE_MIN_SECS..=BLACKHOLE_MAX_SECS);
    let deadline = tokio::time::sleep(Duration::from_secs(secs));
    tokio::pin!(deadline);
    let mut buf = [0u8; 2048];
    loop {
        tokio::select! {
            _ = &mut deadline => break,
            r = client.read(&mut buf) => match r {
                Ok(0) | Err(_) => break, // peer closed
                Ok(_) => {}              // discard and keep waiting
            }
        }
    }
}

pub async fn handle_server_connection_with_lab(
    mut client: TcpStream,
    upstream: SocketAddr,
    max_read: usize,
    lab: Option<Arc<LabLogger>>,
    emitter: EmitterHandle,
    tail: TailPolicy,
    static_kp: Arc<StaticKeypair>,
    handshake_permit: Option<tokio::sync::OwnedSemaphorePermit>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let conn_id = lab.as_ref().map(|l| l.next_conn_id()).unwrap_or(0);
    let _ = client.set_nodelay(true);
    // ntor handshake on the raw stream before splitting: derive the downlink
    // (send) and uplink (recv) frame ciphers for this connection.
    let (send_cipher, recv_cipher) =
        match tokio::time::timeout(HANDSHAKE_TIMEOUT, server_handshake(&mut client, &static_kp)).await {
            Ok(Ok(ciphers)) => ciphers,
            // Failed MARK/MAC (an active probe) or timeout: the server has
            // written nothing, so a prober learns nothing from the bytes — but
            // an instant close is itself a weak tell, so blackhole for a
            // randomised interval first (holding the permit so probes stay
            // bounded), then drop.
            Ok(Err(_)) | Err(_) => {
                blackhole(client).await;
                return Ok(());
            }
        };
    // Handshake done — release the gate so established connections don't count
    // against the in-progress-handshake limit.
    drop(handshake_permit);

    // Cap TOTAL established connections (each registers a FlowSlot serviced by
    // the single shared emitter). The handshake gate bounds the rate; this bounds
    // the steady-state count, so an authenticated client can't flood connections
    // and starve the cover clock for everyone. Held for the connection's life.
    let _conn_permit = match global_connection_sem().try_acquire() {
        Ok(p) => p,
        Err(_) => return Ok(()), // at capacity — shed this connection
    };
    let (client_read, client_write) = client.into_split();

    // One shared shaped downlink back to the client. Bridge relay tasks push
    // tagged cells here; the single FlowSlot shapes them all on one departure
    // clock. The channel closes when all relay tasks (and this function) drop
    // their sender clones, at which point the FlowSlot starts the tail and
    // then exits.
    let downlink_tx = register_shaped_flow(
        client_write, send_cipher, DOWNLINK_LABELS.wire_event, lab.clone(), conn_id, &emitter, tail.clone(),
    )?;

    // Demux the shaped uplink: strip dummies, route real cells to per-stream
    // bridge connections, echo replies back through downlink_tx.
    mux_demux_to_bridges(client_read, recv_cipher, upstream, max_read, downlink_tx, lab, conn_id, &tail).await
}

// ---------------------------------------------------------------------------
// Shaped forwarding — the unified departure clock, used in BOTH directions.
//
// All real and dummy cells leave on a single near-constant-rate Poisson clock
// (`ShapedSession`). Real cells occupy slots as they queue; idle slots are
// filled with dummy cells. Because the clock's rate is (almost) independent of
// the real-traffic timing, the wire's byte/packet envelope does not track real
// activity at any lag — defeating the volume/timing correlation attack — while
// the uniform cell size defeats size-based dummy filtering.
//
// The same loop drives the client→server uplink (request traffic) and the
// server→client downlink (the bulk reply/download traffic). Previously the
// downlink was an unshaped raw passthrough — the dominant, most-fingerprintable
// flow with zero cover. Routing it through the same clock closes that hole.
// ---------------------------------------------------------------------------

/// Per-direction parameters for the shaped send path.
#[derive(Clone, Copy)]
struct ShapeLabels {
    /// Logged once per plaintext read from the source (real ground truth).
    input_event: &'static str,
    /// Logged once per frame written to the wire (mixed real + dummy).
    wire_event: &'static str,
}

const UPLINK_LABELS: ShapeLabels = ShapeLabels {
    input_event: "client_input",
    wire_event: "client_wire_output",
};
const DOWNLINK_LABELS: ShapeLabels = ShapeLabels {
    input_event: "server_bridge_reply",
    wire_event: "server_wire_return",
};

// Tail cover: at end-of-stream the connection keeps emitting cover so that its
// *lifetime* and *total wire volume* are bounded below by a coarse grid rather
// than ending exactly when real traffic stops. Without this the per-window
// envelope can be flat yet the connection's coarse-scale envelope and integral
// still encode its duration — which tracks real activity — so flows leak at
// multi-second timescales and rank by total volume.
//
// The minimum lifetime is a *floor* read from `MIXTOR_TAIL_FLOOR_MS` (a floor,
// not a round-up: a round-up would re-pad a connection that EOFs just past a
// boundary and could cascade across the two directions). All sessions shorter
// than the floor pad to the same lifetime, so their wire spans — and hence total
// wire volume at the fixed cover rate — converge, collapsing the duration and
// integral leaks.
//
// The policy is an explicit `TailPolicy` parameter (below). The production
// binaries default it to `ByClass` (the duration fix is ON for a real Tor PT,
// whose OR-connections are long-lived anyway); the no-lab/test handlers and
// measurement tools default it to `Off` so short-lived connections don't block
// padding. `MIXTOR_TAIL_FLOOR_MS` overrides the default in any caller:
//   unset       → caller's default (ByClass in binaries, Off in tools/tests).
//   "off"/"0"   → OFF — `pad_tail` is a no-op; the connection closes on drain.
//   <N> (ms)    → fixed N-ms floor for every connection (used by tests/demos).
//   "class"     → volume-based `SessionClass` floor: the session's real byte
//                 count picks Short/Medium/Long (30 s / 5 min / 30 min and the
//                 matching volume target). All sessions in a class converge to
//                 the same lifetime and total volume, so neither ranks flows.
const TAIL_FLOOR_ENV: &str = "MIXTOR_TAIL_FLOOR_MS";
const TAIL_VOLUME_QUANTUM_CELLS: u64 = 256; // ~128 KB of wire (fixed-floor mode)

// ---------------------------------------------------------------------------
// Epoch clock — shared alignment for batch mixing (Fix C → epoch model)
//
// Time is divided into fixed-size epochs.  Every connection using the same
// `EpochClock` is aligned to the same boundaries:
//
//   epoch N:  [T·N ─────────────────── T·(N+1))
//              ↑ connection A starts   ↑ A & B end together
//              ↑ connection B starts   (regardless of when real traffic ended)
//
// From the bridge-side adversary's perspective, k connections that registered
// during the same epoch all start forwarding real traffic at T·N and all
// terminate at some later epoch boundary — they are indistinguishable by
// duration or volume.  This is the k-anonymity / batch-mixing property: the
// adversary must guess which of k ISP-side clients maps to which of k bridge
// connections, rather than correlating a single matched-duration pair.
//
// The ISP-side wire still starts when the TCP connection is established (the
// cover clock starts immediately), but real data is held until the epoch
// boundary, so the bridge-side timing is epoch-aligned while the client-side
// ISP timing is not — a deliberate mismatch that further degrades cross-side
// duration matching.
// ---------------------------------------------------------------------------

/// Default epoch period used by production binaries.
///
/// Since the tail length is now decoupled from the epoch (the exit tail pads to
/// the session-class lifetime, not to one epoch — see `tail_targets`), the epoch
/// period only governs (a) the entry-gate start latency and (b) the boundary
/// granularity that batch-aligns starts/exits. A SHORT period is therefore
/// correct: ~5 s start latency instead of ~30 s, and the tail no longer over-pads
/// (a long period would round the class floor up by up to a full period). Flows
/// whose real traffic begins/ends within the same 5 s window are batched together
/// for k-anonymity; operators can widen it for a larger anonymity set via
/// `MIXTOR_TAIL_FLOOR_MS=epoch:<ms>`.
pub const DEFAULT_EPOCH_DURATION: Duration = Duration::from_secs(5);

/// A shared wall-clock epoch timer.  All connections clocked by the same
/// `EpochClock` instance share the same epoch boundaries and therefore the
/// same apparent start/end times from the outside.
pub struct EpochClock {
    origin: std::time::Instant,
    period: Duration,
}

impl EpochClock {
    /// Create a new clock starting now with `period` per epoch.
    pub fn new(period: Duration) -> Arc<Self> {
        Arc::new(Self { origin: std::time::Instant::now(), period })
    }

    /// The nearest epoch boundary strictly after `now`.
    pub fn next_boundary(&self, now: std::time::Instant) -> std::time::Instant {
        let elapsed_ns = now.duration_since(self.origin).as_nanos();
        let period_ns  = self.period.as_nanos();
        if period_ns == 0 { return now; }
        let next_ns = (elapsed_ns / period_ns + 1) * period_ns;
        self.origin + Duration::from_nanos(next_ns as u64)
    }
}

impl std::fmt::Debug for EpochClock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EpochClock").field("period", &self.period).finish()
    }
}

/// Tail-cover policy. Closes the session-duration / total-volume leak: without
/// it a connection's wire lifetime tracks the session's, which a learned attack
/// links (the uplink wire stops ~one cover interval after real traffic does).
/// Production deployments run `Epoch`; tests and tools use `Off`.
#[derive(Clone, Debug)]
pub enum TailPolicy {
    /// No tail cover — the connection closes on drain (leaks duration).
    Off,
    /// Pad to a fixed lifetime floor (tests/demos; use `MIXTOR_TAIL_FLOOR_MS=<N>`).
    Fixed(Duration),
    /// Volume-based `SessionClass` floor with per-session fuzzy boundaries.
    /// Kept for backward compatibility; `Epoch` is preferred.
    ByClass,
    /// Epoch-aligned tail — the production policy.
    ///
    /// Real data is held until the next epoch boundary after connection setup
    /// (all connections sharing the same clock appear to start together), then
    /// the connection pads until the next epoch boundary after real EOF (all
    /// connections in the same epoch appear to end together).  This gives a
    /// k-anonymity batch-mixing property: an adversary matching ISP-side and
    /// bridge-side wires by duration sees k connections starting and ending
    /// simultaneously and cannot tell which is which.
    Epoch(Arc<EpochClock>),
}

impl TailPolicy {
    /// Parse `MIXTOR_TAIL_FLOOR_MS`:
    /// `epoch` / `epoch:<ms>` → Epoch, `class` → ByClass, `off`/`0` → Off,
    /// `<N>` → Fixed(N ms), unset → None (caller picks default).
    pub fn from_env() -> Option<Self> {
        match std::env::var(TAIL_FLOOR_ENV) {
            Err(_) => None,
            Ok(v) if v.eq_ignore_ascii_case("epoch") => {
                Some(Self::Epoch(EpochClock::new(DEFAULT_EPOCH_DURATION)))
            }
            Ok(v) if v.len() > 6 && v[..6].eq_ignore_ascii_case("epoch:") => {
                let ms: u64 = v[6..].parse().ok()?;
                Some(Self::Epoch(EpochClock::new(Duration::from_millis(ms))))
            }
            Ok(v) if v.eq_ignore_ascii_case("class") => Some(Self::ByClass),
            Ok(v) if v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("none") => {
                Some(Self::Off)
            }
            Ok(v) => match v.parse::<u64>() {
                Ok(0) => Some(Self::Off),
                Ok(ms) => Some(Self::Fixed(Duration::from_millis(ms))),
                Err(_) => None,
            },
        }
    }

    /// `MIXTOR_TAIL_FLOOR_MS` if set, otherwise `default`.
    pub fn from_env_or(default: Self) -> Self {
        Self::from_env().unwrap_or(default)
    }

    /// Default for production binaries: `Epoch` with `DEFAULT_EPOCH_DURATION`,
    /// overridable by `MIXTOR_TAIL_FLOOR_MS`.
    pub fn default_for_production() -> Self {
        Self::from_env().unwrap_or_else(|| Self::Epoch(EpochClock::new(DEFAULT_EPOCH_DURATION)))
    }
}

/// Returns `(absolute deadline, wire-volume target in cells)` the tail must
/// reach before the flow closes, or `None` when tail cover is disabled.
///
/// For `Epoch`, the deadline is the next epoch boundary after EOF and the
/// volume target is 0 (the time check is sufficient — the clock fills the
/// volume automatically at the fixed cover rate).  For other policies the
/// deadline is `session_start + floor` and the volume target ensures the wire
/// doesn't close early if the clock ran slow.
fn tail_targets(
    real_bytes_sent: u64,
    wire_seq: u64,
    policy: &TailPolicy,
    thresholds: &FuzzedClassThresholds,
    session_start: std::time::Instant,
) -> Option<(std::time::Instant, u64)> {
    match policy {
        TailPolicy::Off => None,
        TailPolicy::Fixed(d) => {
            let q = TAIL_VOLUME_QUANTUM_CELLS;
            let vol = wire_seq.saturating_add(q - 1) / q * q;
            Some((session_start + *d, vol))
        }
        TailPolicy::ByClass => {
            let (floor, vol) = class_targets_fuzzy(real_bytes_sent, thresholds);
            Some((session_start + floor, vol))
        }
        TailPolicy::Epoch(clock) => {
            // Decouple duration-hiding from start-alignment. The epoch period is
            // chosen SHORT (low start-gate latency + batch-mixing alignment), but
            // a short period must NOT force a short tail: if the wire went dark
            // only ~one short epoch after real EOF, the wire→0 position would
            // track when real stopped — a per-flow duration tell (measured: it
            // was the uplink residual under a 2 s epoch). So pad to at least the
            // session-class lifetime (Short/Medium/Long → 30 s/5 min/30 min),
            // THEN round up to the next epoch boundary so flows still EXIT on
            // aligned boundaries. Long tail hides duration; boundary keeps the
            // batch-mixing k-anonymity.
            let (floor, _vol) = class_targets_fuzzy(real_bytes_sent, thresholds);
            let now = std::time::Instant::now();
            let min_deadline = session_start + floor;
            let anchor = if min_deadline > now { min_deadline } else { now };
            Some((clock.next_boundary(anchor), 0))
        }
    }
}

/// Map a session's real byte volume to its `SessionClass` lifetime floor and
/// total-wire-volume target (in cells) using the hard-coded default boundaries.
/// Used only in tests; live code uses `class_targets_fuzzy`.
#[cfg(test)]
fn class_targets(real_bytes_sent: u64) -> (Duration, u64) {
    let class = SessionClass::for_volume(real_bytes_sent as usize);
    let vol_cells = (class.target_volume_bytes() / CELL_BYTES) as u64;
    (class.target_duration(), vol_cells)
}

/// Map a session's real byte volume to its lifetime floor and wire-volume target
/// (in cells) using per-session fuzzy class boundaries.
fn class_targets_fuzzy(real_bytes_sent: u64, thresholds: &FuzzedClassThresholds) -> (Duration, u64) {
    let class = thresholds.classify(real_bytes_sent as usize);
    let vol_cells = (class.target_volume_bytes() / CELL_BYTES) as u64;
    (class.target_duration(), vol_cells)
}

/// IAT range for per-session cover-rate jitter (Fix A).
///
/// Each session draws `initial_iat_ms` uniformly from this band, so every
/// connection runs at a slightly different cover rate. A cross-point adversary
/// watching both sides cannot subtract a known fixed rate to extract a residual
/// signal; they would need to estimate the per-session rate from the wire itself,
/// which means the wire IS the signal they're trying to remove — an impossibility
/// unless they already know which flows correspond. The range [2.5, 5.0] ms keeps
/// the cover rate between ~200 cells/s (5 ms, ~103 KB/s) and ~400 cells/s
/// (2.5 ms, ~206 KB/s), staying well above typical browsing download bandwidth.
const JITTER_IAT_MIN_MS: f64 = 2.5;
const JITTER_IAT_MAX_MS: f64 = 5.0;

/// Build the fixed-rate cover-clock config for one session.
///
/// `initial_iat_ms` — the per-session mean inter-cell time drawn uniformly from
/// `[JITTER_IAT_MIN_MS, JITTER_IAT_MAX_MS]`. Every session gets a fresh random
/// rate so an observer cannot assume a single known cover rate when comparing the
/// two sides of a connection.
fn shaped_composition_config(initial_iat_ms: f64) -> CompositionConfig {
    CompositionConfig {
        cell_bytes: Some(CELL_BYTES),
        // Fixed-rate departure clock: a homogeneous Poisson process at a constant
        // mean inter-cell time (chosen per-session from [2.5, 5.0] ms) that is
        // *independent of real-traffic timing*. Real cells occupy slots as they
        // queue and idle slots are filled with dummies, so the wire's packet/byte
        // envelope is flat whether or not real traffic is flowing. Exponential (not
        // fixed) intervals keep the spectrum flat (no periodic keepalive peak). The
        // wide [min,max] band avoids truncating the exponential. Adaptation is OFF
        // because bursty per-cell injection would otherwise drag the rate up in
        // exactly the active windows, re-coupling wire rate to real volume.
        clock_adapt_rate:     Some(false),
        clock_ema_alpha:      Some(0.05),
        clock_min_iat_ms:     Some(1.0),
        clock_initial_iat_ms: Some(initial_iat_ms),
        clock_max_iat_ms:     Some(60.0),
        // Reserve ~12% of slots for dummies even under load, so a sustained
        // real rate above the clock rate cannot drive the cover to zero and
        // re-expose the real envelope in the busy windows.
        clock_min_dummy_fraction: Some(0.12),
        ..CompositionConfig::default()
    }
}

// Buffer between the steady-state reader and emitter. Bounds memory and gives
// back-pressure: a full channel blocks the reader's `send().await`, which stops
// it reading the source — the correct upstream flow-control signal.
//
// Sized to ABSORB a transient startup burst. A small buffer fills during the
// front-loaded burst that opens many sessions, so back-pressure propagates to
// the source TCP link and rate-limits it to the cover clock — for that window
// the source link and the wire are both driven by the same clock, a learnable
// startup correlation (measured: CNN AUC ~0.72 with a 256-cell buffer). A larger
// buffer absorbs the burst so the source link runs free (decoupled from cover)
// and only the wire is clock-paced. 1024 cells ≈ 520 KB and bounds the worst-
// case added latency (a sustained over-rate source) to ~3.6 s at the cover rate;
// transient bursts smaller than the buffer add far less.
const STEADY_CHANNEL_CAP: usize = 1024;

/// Forward `src` → `wire`, shaping the plaintext into uniform cells on the
/// unified departure clock. Direction-agnostic: `labels` selects the lab event
/// names so the same path serves the uplink and the downlink.
///
/// After warmup the flow is handed to the process-wide shared `emitter` (one
/// always-runnable, timer-driven task that clocks the cover for ALL flows); this
/// function then becomes just the *reader*, splitting source bytes into cells and
/// feeding them to the emitter over a bounded channel. The emitter — not a
/// per-flow task — owns the wire and decides emission timing from the invariant
/// clock, so emission is independent of this flow's scheduling. That removes the
/// many-flow contention leak (idle per-flow emitters used to starve unevenly) and
/// any machine (scheduler/CPU-frequency) fingerprint.
///
/// Warmup (uplink only) stays single-task: real cells bypass the clock for low
/// handshake latency, with the cover credit so the window matches the steady rate.
async fn shaped_forward(
    mut src: tokio::net::tcp::OwnedReadHalf,
    wire: tokio::net::tcp::OwnedWriteHalf,
    max_read: usize,
    lab: Option<Arc<LabLogger>>,
    conn_id: u64,
    labels: ShapeLabels,
    emitter: EmitterHandle,
    tail: TailPolicy,
    cipher: FrameCipher,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    // Per-session cover rate jitter: draw the mean IAT once from the [2.5, 5.0] ms
    // band so each session runs at a different rate. A cross-point observer cannot
    // subtract a single known cover rate to extract a residual signal; they must
    // estimate the rate from the wire they're already trying to analyse.
    let iat_ms: f64 = StdRng::from_entropy().gen_range(JITTER_IAT_MIN_MS..=JITTER_IAT_MAX_MS);
    // Per-session fuzzy class boundaries: drawn at session START so the commitment
    // happens before any real bytes are seen. The two sides of the same connection
    // draw independently — their class assignments (and wire durations) may differ,
    // which degrades cross-side duration matching even when the tail floor fires.
    let thresholds = FuzzedClassThresholds::sample(&mut StdRng::from_entropy());
    let seeds = SessionSeeds::generate()?;
    let session = ShapedSession::new(seeds.burst_composition(), shaped_composition_config(iat_ms))?;
    let mut pad_rng = StdRng::from_entropy();
    let mut buffer = vec![0u8; max_read.max(1)];
    let mut input_seq = 0u64;
    let session_start = std::time::Instant::now();

    // Epoch entry gate: in Epoch mode, compute the next epoch boundary from now
    // so the emitter holds real data until then. Cover dummies flow immediately
    // from connection time (the wire is never dark), but the bridge only starts
    // seeing real traffic at the epoch boundary — the entry-side batch-mixing
    // property. For non-Epoch policies, set the gate to now (open immediately).
    let real_hold_until = match &tail {
        TailPolicy::Epoch(clock) => clock.next_boundary(session_start),
        _ => session_start,
    };

    let (cell_tx, cell_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(STEADY_CHANNEL_CAP);
    emitter.register(FlowSlot {
        wire,
        session,
        cell_rx,
        lab: lab.clone(),
        conn_id,
        wire_event: labels.wire_event,
        wire_seq: 0,
        session_start,
        pending: Vec::new(),
        reader_done: false,
        policy: tail,
        tail: None,
        finish_by: None,
        thresholds,
        real_hold_until,
        real_bytes: 0,
        cipher,
        clock_started: false,
    });

    loop {
        let n = match src.read(&mut buffer).await {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) => return Err(e.into()),
        };
        log_event(&lab, labels.input_event, conn_id, input_seq, n, "real");
        input_seq += 1;
        for chunk in buffer[..n].chunks(CELL_PAYLOAD) {
            let cell = encode_real_cell(STREAM_ID_SINGLE, chunk, &mut pad_rng);
            if cell_tx.send(cell).await.is_err() {
                return Ok(());
            }
        }
    }
    drop(cell_tx);
    Ok(())
}

// ---------------------------------------------------------------------------
// Unified emitter — one timer-driven task clocks the cover for ALL flows.
//
// Each post-warmup flow registers a `FlowSlot` (its wire + `ShapedSession` +
// reader channel). The single `run_emitter` task wakes on its own fixed
// `EMIT_TICK` timer — never on any flow's read activity — and synchronously
// services every flow: it drains queued real cells into the session, ticks the
// per-flow Poisson clock, and flushes the result to the wire with a
// non-blocking `try_write`. No flow gets its own task, so none can be starved
// relative to another; and because the wake cadence is a wall-clock timer
// (invariant across CPU frequency and load), there is no scheduler/CPU-speed
// machine fingerprint. Per-flow emission timing is still that flow's own
// absolute Poisson schedule (`ShapedSession::next_departure`), merely quantised
// to `EMIT_TICK`.
// ---------------------------------------------------------------------------

/// How often the shared emitter wakes to service all flows. This sets only the
/// timing *granularity*, not the rate: each flow still emits on its own
/// absolute Poisson schedule, quantised to this tick. 1 ms ≪ the ~3.5 ms mean
/// inter-cell time, so quantisation adds negligible jitter and no periodic peak
/// (the underlying per-flow schedule is aperiodic/exponential).
const EMIT_TICK: Duration = Duration::from_millis(1);

/// Cap on a flow's encoded-but-unwritten wire bytes. Bounds memory and provides
/// back-pressure: at the cap the emitter stops ticking new frames and pulling
/// cells, so the channel fills and the reader's `send().await` blocks — the
/// correct upstream flow-control signal.
const PENDING_CAP: usize = 64 * 1024;


/// One flow owned by the shared emitter: its wire, its shaped session/clock, the
/// reader→emitter cell channel, and the per-flow emit bookkeeping.
struct FlowSlot {
    wire: tokio::net::tcp::OwnedWriteHalf,
    session: ShapedSession,
    cell_rx: tokio::sync::mpsc::Receiver<Vec<u8>>,
    lab: Option<Arc<LabLogger>>,
    conn_id: u64,
    wire_event: &'static str,
    wire_seq: u64,
    session_start: std::time::Instant,
    /// Encoded-but-unwritten wire bytes; `try_write` drains this each tick.
    pending: Vec<u8>,
    /// Set once the reader channel closes (EOF): drain the queue, then pad the tail.
    reader_done: bool,
    /// Tail-cover policy for this flow (selects the floor at EOF).
    policy: TailPolicy,
    /// `(absolute deadline, wire-volume target in cells)` to reach before closing,
    /// or `None` when tail cover is disabled. Computed once at reader EOF.
    tail: Option<(std::time::Instant, u64)>,
    /// Hard backstop computed at reader EOF: force-close past this instant even
    /// if `pending` cannot flush (peer stopped reading), so a flow is never kept
    /// alive indefinitely.
    finish_by: Option<std::time::Instant>,
    /// Per-session fuzzy class boundaries drawn at session start. Used at EOF
    /// to assign the `SessionClass` (and hence tail floor) without exposing the
    /// hard-coded default cutoffs to an adversary who reads the source.
    thresholds: FuzzedClassThresholds,
    /// True real bytes injected so far — the sum of the decoded `len` fields of
    /// the real cells, NOT `cells × CELL_PAYLOAD`. Counting padded cells would
    /// let many small writes (each consuming a whole cell) inflate the apparent
    /// volume and bump the `ByClass` tail into a longer-lived class — a write-
    /// fragmentation fingerprint. Close sentinels (`len == 0`) add nothing.
    real_bytes: u64,
    /// Epoch-mode entry gate: real cells are not forwarded to the session until
    /// this instant (the next epoch boundary from connection time). Cover dummies
    /// flow the whole time, so the wire starts immediately, but the bridge only
    /// sees real traffic starting at the epoch boundary — the entry side of the
    /// batch-mixing property. For non-Epoch policies this is set to connection
    /// time so the gate is open from the first tick.
    real_hold_until: std::time::Instant,
    /// Per-direction AEAD frame cipher for THIS flow's wire (the send side).
    /// Each emitted cell is sealed with it before hitting the wire, so the
    /// real/dummy kind and the cell bytes are ciphertext — indistinguishable on
    /// the wire. Derived by the connection's ntor handshake.
    cipher: FrameCipher,
    /// Cleared until the emitter's first service of this flow, when the
    /// departure clock is restarted from now. Without this the slots that came
    /// due between session creation and first service drain as a startup burst
    /// whose size tracks the queued real data — a per-flow startup timing leak.
    clock_started: bool,
}

/// Handle to the process-wide shared emitter. Clone it once per accept loop and
/// move a clone into each connection task; every clone keeps the emitter task
/// alive and lets a connection `register` its flow once warmup is done.
#[derive(Clone)]
pub struct EmitterHandle {
    reg: tokio::sync::mpsc::UnboundedSender<FlowSlot>,
}

impl EmitterHandle {
    /// Spawn the shared emitter and return a handle to it.
    ///
    /// The emitter runs on its **own dedicated OS thread** with a single-thread
    /// runtime — NOT as a task on the caller's runtime. This is deliberate: as a
    /// task it competes with the (many) per-flow reader tasks for workers, so
    /// under busy multi-flow bursts it is scheduled late, its tick rate dips, and
    /// the cover rate drops in exactly the busy windows — re-coupling cover to
    /// activity (a machine-scheduling fingerprint). On a dedicated thread the
    /// clock ticks at a fixed wall-clock cadence regardless of how loaded the
    /// main runtime is. `try_write` is readiness-based and works on the wire
    /// fds (registered with the caller's runtime) from this thread.
    pub fn new() -> Self {
        let (reg, rx) = tokio::sync::mpsc::unbounded_channel();
        std::thread::Builder::new()
            .name("mixtor-emitter".into())
            .spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("build emitter runtime");
                rt.block_on(run_emitter(rx));
            })
            .expect("spawn emitter thread");
        Self { reg }
    }

    /// Hand a post-warmup flow to the emitter. If the emitter has already
    /// stopped (process shutdown) the send fails and the slot is dropped, which
    /// closes its wire — acceptable, as this only happens at teardown.
    fn register(&self, slot: FlowSlot) {
        let _ = self.reg.send(slot);
    }
}

impl Default for EmitterHandle {
    fn default() -> Self {
        Self::new()
    }
}

/// The shared emitter loop. Admits newly-registered flows, then every
/// `EMIT_TICK` services each flow once, dropping those that have finished.
async fn run_emitter(mut reg: tokio::sync::mpsc::UnboundedReceiver<FlowSlot>) {
    use tokio::sync::mpsc::error::TryRecvError;
    let mut flows: Vec<FlowSlot> = Vec::new();
    let mut reg_open = true;
    loop {
        // Admit any newly-registered flows (non-blocking).
        if reg_open {
            loop {
                match reg.try_recv() {
                    Ok(slot) => flows.push(slot),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        reg_open = false;
                        break;
                    }
                }
            }
        }
        // All handles dropped and every flow finished: nothing more can arrive.
        if !reg_open && flows.is_empty() {
            return;
        }
        tokio::time::sleep(EMIT_TICK).await;
        // Service every flow once; drop (swap-remove) those that are done. Order
        // is irrelevant — each flow drives its own independent wire and clock.
        let mut i = 0;
        while i < flows.len() {
            if service_flow(&mut flows[i]) {
                i += 1;
            } else {
                flows.swap_remove(i);
            }
        }
    }
}

/// Service one flow for this tick. Returns `true` to keep the flow, `false` to
/// drop it (drained and tail satisfied, or the wire errored). Fully synchronous
/// and non-blocking: it never `.await`s, so emission timing is decoupled from
/// read arrivals and from this function's own scheduling.
fn service_flow(flow: &mut FlowSlot) -> bool {
    use tokio::sync::mpsc::error::TryRecvError;

    // First service: restart the departure clock from now so the slots that
    // came due between session creation and this moment don't drain as a startup
    // burst (which, via pull-per-slot, would be sized by the queued real data —
    // a per-flow startup timing leak). After this the flow is serviced every
    // tick, so no backlog accumulates and steady-state catch-up is unaffected.
    if !flow.clock_started {
        flow.session.restart_clock();
        flow.clock_started = true;
    }

    // Tick the departure clock one slot at a time; for each slot that fires,
    // attempt exactly one pull from the reader channel before ticking.
    //
    // Hard throughput cap: this couples the real-ingestion rate directly to
    // the cover-clock rate — real cells can never be consumed from the channel
    // faster than the clock emits slots. A source faster than the clock rate
    // fills the channel and backs pressure upstream smoothly at the cover rate,
    // rather than bursting into the correlator queue and then stalling
    // irregularly (the old pull-many-then-tick pattern allowed real cells to
    // race ~384 cells ahead of the wire, creating a stall pattern an ISP+exit
    // observer could correlate). With pull-per-slot the correlator queue depth
    // stays at 0–1; the channel provides burst absorption at the reader side.
    if flow.pending.len() < PENDING_CAP {
        // Epoch entry gate: in Epoch mode real cells are held until the first
        // epoch boundary after connection time (real_hold_until). Cover dummies
        // still tick every slot regardless — the wire is never dark.
        let real_gate_open = std::time::Instant::now() >= flow.real_hold_until;

        while flow.session.slot_ready() && flow.pending.len() < PENDING_CAP {
            // One pull attempt per slot, subject to the epoch gate.
            if !flow.reader_done && flow.session.has_queue_space() && real_gate_open {
                match flow.cell_rx.try_recv() {
                    Ok(cell) => {
                        // Accumulate the cell's TRUE payload length (from its
                        // header) before injecting, so `ByClass` classifies on
                        // real bytes, not padded cell count. Once try_recv
                        // returns Disconnected every sent cell has been pulled,
                        // so this sum is the flow's complete real-byte total.
                        let true_len = decode_real_cell(&cell).1.len() as u64;
                        if let Ok(packet) = Packet::new(cell) {
                            if flow.session.inject_real(packet).is_ok() {
                                flow.real_bytes = flow.real_bytes.saturating_add(true_len);
                            }
                        }
                    }
                    Err(TryRecvError::Empty) => {}
                    Err(TryRecvError::Disconnected) => {
                        flow.reader_done = true;
                        flow.tail = tail_targets(
                            flow.real_bytes,
                            flow.wire_seq,
                            &flow.policy,
                            &flow.thresholds,
                            flow.session_start,
                        );
                        // Backstop: force-close 5 s past the deadline so a
                        // stuck peer (stopped reading) never leaks a flow.
                        let backstop = flow
                            .tail
                            .map(|(deadline, _)| deadline + Duration::from_secs(5))
                            .unwrap_or_else(|| std::time::Instant::now() + Duration::from_secs(5));
                        flow.finish_by = Some(backstop);
                    }
                }
            }
            if let Some(frame) = flow.session.tick() {
                let (bytes, kind, plen) = encode_frame(frame, &mut flow.cipher);
                log_event(&flow.lab, flow.wire_event, flow.conn_id, flow.wire_seq, plen, kind);
                flow.wire_seq += 1;
                flow.pending.extend_from_slice(&bytes);
            }
        }
    }

    // 3) Flush pending to the wire, non-blocking. A partial write keeps the rest
    //    for the next tick; WouldBlock just means the socket buffer is full.
    if !flow.pending.is_empty() {
        match flow.wire.try_write(&flow.pending) {
            Ok(0) => return false, // wire closed
            Ok(n) => {
                flow.pending.drain(0..n);
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(_) => return false, // wire error → drop the flow
        }
    }

    // 4) Termination: only once the reader has EOF'd. The flow ends when the
    //    real queue is drained and everything is written AND the tail target is
    //    met — or unconditionally past the safety backstop.
    if flow.reader_done {
        let drained = flow.session.queue_depth() == 0 && flow.pending.is_empty();
        let tail_met = match flow.tail {
            None => true, // tail cover disabled: close as soon as drained
            Some((deadline, vol)) => std::time::Instant::now() >= deadline && flow.wire_seq >= vol,
        };
        let past_deadline = flow
            .finish_by
            .map(|d| std::time::Instant::now() >= d)
            .unwrap_or(false);
        if (drained && tail_met) || past_deadline {
            // Dropping the FlowSlot drops `wire`, which closes the connection.
            return false;
        }
    }
    true
}

/// Seal a mixed frame into its encrypted on-wire bytes, returning
/// `(ciphertext, kind-label, plaintext-cell-len)` for the wire and the lab log.
/// The real/dummy `kind` marker is sealed INSIDE the ciphertext, so the wire
/// carries only a fixed-size opaque blob — real and dummy are indistinguishable.
fn encode_frame(frame: MixedFrame, cipher: &mut FrameCipher) -> (Vec<u8>, &'static str, usize) {
    let (kind_byte, kind_label, cell) = match frame {
        MixedFrame::Real(packet) => (KIND_REAL, "real", packet.into_bytes()),
        MixedFrame::Dummy(bytes) => (KIND_DUMMY, "dummy", bytes),
    };
    let plen = cell.len();
    (cipher.seal(kind_byte, &cell), kind_label, plen)
}

/// Outbound path (client → server): shape the request/uplink traffic.
async fn send_mixed_outbound(
    local: tokio::net::tcp::OwnedReadHalf,
    remote: tokio::net::tcp::OwnedWriteHalf,
    max_read: usize,
    lab: Option<Arc<LabLogger>>,
    conn_id: u64,
    emitter: EmitterHandle,
    tail: TailPolicy,
    cipher: FrameCipher,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    shaped_forward(local, remote, max_read, lab, conn_id, UPLINK_LABELS, emitter, tail, cipher).await
}


// ---------------------------------------------------------------------------
// Inbound path (server → client)
// ---------------------------------------------------------------------------

/// Per-direction lab event labels for the strip/decode path.
#[derive(Clone, Copy)]
struct StripLabels {
    /// Logged once per frame seen on the wire (mixed real + dummy).
    wire_event: &'static str,
    /// Logged once per decoded real payload forwarded downstream (ground truth).
    real_event: &'static str,
}

const UPLINK_STRIP: StripLabels = StripLabels {
    wire_event: "server_wire_input",
    real_event: "server_real_output",
};
const DOWNLINK_STRIP: StripLabels = StripLabels {
    wire_event: "client_wire_return",
    real_event: "client_inbound_real",
};

/// Read one fixed-size encrypted frame from `wire` and open it with `cipher`.
/// `Ok(None)` = clean EOF; `Ok(Some((kind, cell)))` = a decrypted frame; an
/// auth/format failure is an error (tamper or desync → the caller aborts).
/// `buf` must be `enc_frame_len(CELL_BYTES)` long.
async fn read_sealed_frame(
    wire:   &mut tokio::net::tcp::OwnedReadHalf,
    cipher: &mut FrameCipher,
    buf:    &mut [u8],
) -> Result<Option<(u8, Vec<u8>)>, Box<dyn Error + Send + Sync>> {
    match wire.read_exact(buf).await {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e.into()),
    }
    let (kind, cell) = cipher.open(buf)?;
    Ok(Some((kind, cell)))
}

/// Read fixed-size encrypted frames from `wire`, decrypt them, drop dummies,
/// unpad real cells back to their true bytes, and forward them in order to
/// `out`. Direction-agnostic.
async fn strip_and_forward(
    mut wire:   tokio::net::tcp::OwnedReadHalf,
    mut out:    tokio::net::tcp::OwnedWriteHalf,
    mut cipher: FrameCipher,
    lab:        Option<Arc<LabLogger>>,
    conn_id:    u64,
    labels:     StripLabels,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut wire_seq   = 0u64;
    let mut output_seq = 0u64;
    let mut buf = vec![0u8; enc_frame_len(CELL_BYTES)];

    while let Some((kind, cell)) = read_sealed_frame(&mut wire, &mut cipher, &mut buf).await? {
        let kind_label = if kind == KIND_REAL { "real" } else { "dummy" };
        log_event(&lab, labels.wire_event, conn_id, wire_seq, cell.len(), kind_label);
        wire_seq += 1;

        // Dummies are dropped; real cells are unpadded back to their true bytes
        // and forwarded in order.
        if kind == KIND_REAL {
            let (_stream_id, data) = decode_real_cell(&cell);
            if !data.is_empty() {
                log_event(&lab, labels.real_event, conn_id, output_seq, data.len(), "real");
                output_seq += 1;
                out.write_all(data).await?;
            }
        }
    }
    let _ = out.shutdown().await;
    Ok(())
}

/// Inbound on the client (server → app): strip the shaped downlink to plaintext.
async fn receive_real_inbound(
    remote:  tokio::net::tcp::OwnedReadHalf,
    local:   tokio::net::tcp::OwnedWriteHalf,
    cipher:  FrameCipher,
    lab:     Option<Arc<LabLogger>>,
    conn_id: u64,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    strip_and_forward(remote, local, cipher, lab, conn_id, DOWNLINK_STRIP).await
}

// ---------------------------------------------------------------------------
// Server-side mux: demultiplex one shaped client wire into N bridge connections,
// and re-multiplex N bridge reply streams into one shaped downlink.
//
// Stream IDs come from the client's cell headers. When a new stream_id appears
// a fresh bridge TCP connection is opened.  When a close sentinel (len = 0)
// arrives the bridge connection is torn down. The shapes downlink FlowSlot is
// shared across all active bridge streams via clones of `downlink_tx`.
// ---------------------------------------------------------------------------

/// Create and register a shaped FlowSlot on `emitter`.
/// Returns the `Sender` callers use to inject pre-encoded cells.
/// `wire_event` selects the lab event name logged per wire write.
fn register_shaped_flow(
    wire: tokio::net::tcp::OwnedWriteHalf,
    cipher: FrameCipher,
    wire_event: &'static str,
    lab: Option<Arc<LabLogger>>,
    conn_id: u64,
    emitter: &EmitterHandle,
    tail: TailPolicy,
) -> Result<tokio::sync::mpsc::Sender<Vec<u8>>, Box<dyn Error + Send + Sync>> {
    let iat_ms: f64 = StdRng::from_entropy().gen_range(JITTER_IAT_MIN_MS..=JITTER_IAT_MAX_MS);
    let thresholds = FuzzedClassThresholds::sample(&mut StdRng::from_entropy());
    let seeds = SessionSeeds::generate()?;
    let session = ShapedSession::new(seeds.burst_composition(), shaped_composition_config(iat_ms))?;
    let session_start = std::time::Instant::now();
    // Mux wires carry no entry gate. Individual circuit starts are already
    // invisible inside a long-lived shaped wire (the cover is always flat), so
    // epoch entry alignment provides no additional protection here. Only the
    // exit tail (TailPolicy) aligns wire teardown to epoch boundaries.
    // (Compare: shaped_forward for single-stream connections does gate entry.)
    let real_hold_until = session_start;
    let (cell_tx, cell_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(STEADY_CHANNEL_CAP);
    emitter.register(FlowSlot {
        wire,
        session,
        cell_rx,
        lab,
        conn_id,
        wire_event,
        wire_seq: 0,
        session_start,
        pending: Vec::new(),
        reader_done: false,
        policy: tail,
        tail: None,
        finish_by: None,
        thresholds,
        real_hold_until,
        real_bytes: 0,
        cipher,
        clock_started: false,
    });
    Ok(cell_tx)
}

/// Max concurrent mux streams (≈Tor circuits) a single client wire may open. A
/// client that knows the cert could otherwise spam fresh stream_ids, each
/// spawning a task + an upstream ORPort connection; past this the wire is an
/// abuser and is dropped.
const MAX_STREAMS_PER_WIRE: usize = 512;
/// Process-wide cap on live bridge streams (bounds total upstream ORPort
/// connections across ALL client wires). New streams past this are refused.
const MAX_GLOBAL_BRIDGE_STREAMS: usize = 4096;
static GLOBAL_BRIDGE_STREAMS: AtomicUsize = AtomicUsize::new(0);
/// Cap on bytes a single stream may buffer during the epoch hold window before
/// back-pressure kicks in (bounds server memory per held stream).
const EPOCH_HOLD_MAX_BYTES: usize = 256 * 1024;

/// RAII decrement of the global live-stream counter — held by each bridge relay
/// task so the count falls when the stream's relay ends, however it ends.
struct BridgeStreamGuard;
impl Drop for BridgeStreamGuard {
    fn drop(&mut self) {
        GLOBAL_BRIDGE_STREAMS.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Read shaped cells from `client`, strip dummies, and route real cells by
/// stream_id to per-stream bridge connections. Each new stream_id triggers a
/// new bridge TCP connection whose replies are shaped back to the client via
/// clones of `downlink_tx`.
async fn mux_demux_to_bridges(
    mut client: tokio::net::tcp::OwnedReadHalf,
    mut cipher: FrameCipher,
    upstream: SocketAddr,
    max_read: usize,
    downlink_tx: tokio::sync::mpsc::Sender<Vec<u8>>,
    lab: Option<Arc<LabLogger>>,
    conn_id: u64,
    tail: &TailPolicy,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut stream_map: HashMap<u16, tokio::sync::mpsc::Sender<Vec<u8>>> = HashMap::new();
    let mut wire_seq = 0u64;
    let mut pad_rng = StdRng::from_entropy();
    let mut buf = vec![0u8; enc_frame_len(CELL_BYTES)];

    while let Some((kind, cell)) = read_sealed_frame(&mut client, &mut cipher, &mut buf).await? {
        let kind_str = if kind == KIND_REAL { "real" } else { "dummy" };
        log_event(&lab, UPLINK_STRIP.wire_event, conn_id, wire_seq, cell.len(), kind_str);
        wire_seq += 1;
        if kind != KIND_REAL { continue; }

        let (stream_id, data) = decode_real_cell(&cell);

        // len = 0 is the close sentinel: tear down the stream.
        if data.is_empty() {
            stream_map.remove(&stream_id);
            let close = encode_close_cell(stream_id, &mut pad_rng);
            let _ = downlink_tx.send(close).await;
            continue;
        }

        // Get-or-create the bridge channel for this stream.
        let active_streams = stream_map.len();
        let bridge_tx = match stream_map.entry(stream_id) {
            Entry::Occupied(o) => o.into_mut(),
            Entry::Vacant(v) => {
                // Per-wire cap: a client spamming fresh stream_ids is abusive —
                // drop the whole mux connection.
                if active_streams >= MAX_STREAMS_PER_WIRE {
                    return Err("mux client exceeded per-wire stream cap".into());
                }
                // Global cap: bound total upstream ORPort connections. Past it,
                // refuse this stream (don't open a bridge) but keep the wire.
                let prev = GLOBAL_BRIDGE_STREAMS.fetch_add(1, Ordering::Relaxed);
                if prev >= MAX_GLOBAL_BRIDGE_STREAMS {
                    GLOBAL_BRIDGE_STREAMS.fetch_sub(1, Ordering::Relaxed);
                    continue;
                }
                let guard = BridgeStreamGuard; // decrements the global count on relay end
                // Epoch gate: compute the next epoch boundary from *now* so all
                // streams whose first cell arrives in the same epoch window appear
                // to start forwarding at the same instant from the bridge's view.
                // Non-epoch policies pass Instant::now() → gate_until is already
                // in the past by the time the task checks it, so it's a no-op.
                let gate_until = match tail {
                    TailPolicy::Epoch(clock) => clock.next_boundary(std::time::Instant::now()),
                    _ => std::time::Instant::now(),
                };
                let (data_tx, data_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(256);
                let dl_tx = downlink_tx.clone();
                let bridge_lab = lab.clone();
                tokio::spawn(async move {
                    let _guard = guard; // dropped (count--) when the relay finishes
                    if let Err(e) = bridge_stream_relay(
                        upstream, stream_id, data_rx, dl_tx, max_read, bridge_lab, conn_id,
                        gate_until,
                    ).await {
                        eprintln!("stream {stream_id} bridge error: {e}");
                    }
                });
                v.insert(data_tx)
            }
        };

        // Non-blocking route: a `.await` here would head-of-line-block the whole
        // wire on a single stalled stream (full bridge channel = ~one slow
        // circuit). Instead `try_send`: on Closed the relay is gone, so drop the
        // map entry; on Full the stream's bridge isn't draining — drop this cell
        // rather than stall every other circuit on the wire. The resulting gap
        // makes Tor tear that connection down, which cleans the stream up. (The
        // 256-cell channel + Tor's own flow control mean Full only happens on a
        // genuine sustained stall, not transient jitter.)
        match bridge_tx.try_send(data.to_vec()) {
            Ok(()) => {}
            Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                stream_map.remove(&stream_id);
            }
            Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => { /* drop cell */ }
        }
    }
    Ok(())
}

/// Full-duplex relay for one mux stream to its upstream bridge connection.
///
/// `gate_until` is the epoch boundary this stream must wait behind before the
/// bridge connection is opened and data begins flowing. The hold phase buffers
/// incoming real cells; when the boundary arrives the bridge TCP connection is
/// opened and the buffered bytes are flushed atomically. From the bridge's
/// perspective the TCP handshake and the first data byte arrive at the same
/// epoch boundary — matching all other streams in the same epoch window and
/// providing k-anonymity for circuit-start events.
async fn bridge_stream_relay(
    upstream: SocketAddr,
    stream_id: u16,
    mut data_rx: tokio::sync::mpsc::Receiver<Vec<u8>>,
    cell_tx: tokio::sync::mpsc::Sender<Vec<u8>>,
    max_read: usize,
    lab: Option<Arc<LabLogger>>,
    conn_id: u64,
    gate_until: std::time::Instant,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    // Hold phase: buffer real bytes until the epoch gate opens.
    // Delaying the bridge connect (not just the writes) means the bridge sees
    // the TCP connection appear at the same instant it sees the first data.
    //
    // The buffer is BOUNDED: once `EPOCH_HOLD_MAX_BYTES` are held we stop pulling
    // from `data_rx`, so the channel fills and back-pressure reaches the client
    // (rather than letting a client fill server memory during the up-to-one-epoch
    // hold). We still wait out the gate before connecting.
    let mut held: Vec<Vec<u8>> = Vec::new();
    let mut held_bytes = 0usize;
    if std::time::Instant::now() < gate_until {
        let deadline = tokio::time::sleep_until(tokio::time::Instant::from_std(gate_until));
        tokio::pin!(deadline);
        'hold: loop {
            if held_bytes >= EPOCH_HOLD_MAX_BYTES {
                // Cap hit: quit draining (channel back-pressures the client) and
                // just wait for the gate to open.
                (&mut deadline).await;
                break 'hold;
            }
            tokio::select! {
                biased; // deadline takes priority when both branches are ready
                _ = &mut deadline => break 'hold,
                msg = data_rx.recv() => match msg {
                    Some(data) => { held_bytes += data.len(); held.push(data); }
                    None => return Ok(()), // stream closed before gate; nothing to forward
                },
            }
        }
    }

    let bridge = TcpStream::connect(upstream).await?;
    let _ = bridge.set_nodelay(true);
    let (bridge_read, bridge_write) = bridge.into_split();

    let writer_lab = lab.clone();
    let writer = tokio::spawn(async move {
        bridge_stream_writer(data_rx, bridge_write, held, writer_lab, conn_id).await
    });
    let reader = tokio::spawn(async move {
        bridge_stream_reader(bridge_read, cell_tx, stream_id, max_read, lab, conn_id).await
    });

    tokio::select! {
        r = writer => r??,
        r = reader => r??,
    }
    Ok(())
}

/// Uplink leg: flush `held` (buffered during hold phase) then forward live
/// bytes from `data_rx` to the bridge.
async fn bridge_stream_writer(
    mut data_rx: tokio::sync::mpsc::Receiver<Vec<u8>>,
    mut bridge_write: tokio::net::tcp::OwnedWriteHalf,
    held: Vec<Vec<u8>>,
    lab: Option<Arc<LabLogger>>,
    conn_id: u64,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut seq = 0u64;
    for data in held {
        log_event(&lab, UPLINK_STRIP.real_event, conn_id, seq, data.len(), "real");
        seq += 1;
        bridge_write.write_all(&data).await?;
    }
    while let Some(data) = data_rx.recv().await {
        log_event(&lab, UPLINK_STRIP.real_event, conn_id, seq, data.len(), "real");
        seq += 1;
        bridge_write.write_all(&data).await?;
    }
    let _ = bridge_write.shutdown().await;
    Ok(())
}

/// Downlink leg: read bridge replies, encode as tagged cells into `cell_tx`.
/// Sends a close sentinel when the bridge closes.
async fn bridge_stream_reader(
    mut bridge_read: tokio::net::tcp::OwnedReadHalf,
    cell_tx: tokio::sync::mpsc::Sender<Vec<u8>>,
    stream_id: u16,
    max_read: usize,
    lab: Option<Arc<LabLogger>>,
    conn_id: u64,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let max_read = max_read.clamp(1, DEFAULT_MAX_FRAME_LEN);
    let mut pad_rng = StdRng::from_entropy();
    let mut buf = vec![0u8; max_read];
    let mut seq = 0u64;
    loop {
        let n = match bridge_read.read(&mut buf).await {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) => return Err(e.into()),
        };
        log_event(&lab, DOWNLINK_LABELS.input_event, conn_id, seq, n, "real");
        seq += 1;
        for chunk in buf[..n].chunks(CELL_PAYLOAD) {
            let cell = encode_real_cell(stream_id, chunk, &mut pad_rng);
            if cell_tx.send(cell).await.is_err() {
                return Ok(());
            }
        }
    }
    let close = encode_close_cell(stream_id, &mut pad_rng);
    let _ = cell_tx.send(close).await;
    Ok(())
}

// ---------------------------------------------------------------------------
// Client-side mux — multiple Tor circuits over one shaped wire (Fix D).
//
// `MuxWire` manages one TCP connection to the server. Its shaped uplink
// FlowSlot is shared across all active Tor circuits (stream_ids 1..=0xFFFE).
// Each circuit gets a private clone of the shared `cell_tx` (for sending
// tagged uplink cells) and a private `Receiver` (for receiving tagged
// downlink bytes after the receive task demuxes them by stream_id).
//
// `MuxClientHandle` keeps a map of `server addr → MuxWire` so a single accept
// loop reuses one long-lived wire per destination across every circuit. The
// wire's uplink FlowSlot keeps clocking cover even while no circuit is active
// (it closes only on TCP error, never on circuit churn), so the wire's
// lifetime and total cover volume reflect *client online time*, not any single
// circuit's duration — this is what decouples wire duration from flow duration
// (closing the epoch-tail volume/duration leak at its root). Raw mode pre-warms
// the wire at startup; managed mode keys by the SOCKS-supplied bridge address
// (normally one entry) so the same persistent-wire path is active there too.
// ---------------------------------------------------------------------------

/// The shared mux wire: one shaped TCP connection to the server, shared by
/// N Tor circuits. Cheaply clonable via `Arc`.
struct MuxWire {
    /// Channel into the uplink FlowSlot. Clone one copy per Tor circuit.
    cell_tx: tokio::sync::mpsc::Sender<Vec<u8>>,
    /// Map stream_id → per-circuit downlink sender, populated by `add_stream`.
    stream_map: Arc<tokio::sync::Mutex<HashMap<u16, tokio::sync::mpsc::Sender<Vec<u8>>>>>,
    /// Monotonically-increasing stream-ID allocator. Skip 0 (STREAM_ID_SINGLE).
    next_id: Arc<AtomicU16>,
}

impl MuxWire {
    async fn connect(
        server: SocketAddr,
        server_pub: [u8; 32],
        lab: Option<Arc<LabLogger>>,
        emitter: EmitterHandle,
        tail: TailPolicy,
        conn_id: u64,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let mut remote = TcpStream::connect(server).await?;
        let _ = remote.set_nodelay(true);
        // ntor handshake before splitting: verify the bridge and derive the
        // uplink (send) and downlink (recv) frame ciphers for this wire. A
        // timeout means the server never answered (wrong key / silent probe
        // rejection / server down).
        let (send_cipher, recv_cipher) =
            match tokio::time::timeout(HANDSHAKE_TIMEOUT, client_handshake(&mut remote, &server_pub)).await {
                Ok(r) => r?,
                Err(_) => return Err("mixtor handshake timed out (wrong server key or no server?)".into()),
            };
        let (remote_read, remote_write) = remote.into_split();

        let cell_tx = register_shaped_flow(
            remote_write, send_cipher, UPLINK_LABELS.wire_event, lab.clone(), conn_id, &emitter, tail,
        )?;

        let stream_map: Arc<tokio::sync::Mutex<HashMap<u16, tokio::sync::mpsc::Sender<Vec<u8>>>>> =
            Arc::new(tokio::sync::Mutex::new(HashMap::new()));
        let map_for_task = stream_map.clone();
        tokio::spawn(async move {
            if let Err(e) = receive_mux_inbound(remote_read, recv_cipher, map_for_task, lab, conn_id).await {
                eprintln!("mux inbound conn={conn_id}: {e}");
            }
        });

        Ok(Self {
            cell_tx,
            stream_map,
            next_id: Arc::new(AtomicU16::new(1)),
        })
    }

    fn is_alive(&self) -> bool { !self.cell_tx.is_closed() }

    /// Allocate a new stream, or `None` if the wire is saturated (≈65 534 live
    /// circuits — caller should rotate to a fresh wire).
    ///
    /// The `next_id` counter is u16 and wraps; on a long-lived wire it WILL
    /// eventually cycle back to an id whose circuit is still active. We must not
    /// reuse a live id (that would cross-route two circuits), so we scan forward
    /// from the counter under the `stream_map` lock and skip both `0`
    /// (`STREAM_ID_SINGLE`) and any id currently in the map. Streams are removed
    /// from the map when their circuit closes (the close sentinel / a failed
    /// send), so ids free up as circuits end.
    async fn add_stream(
        &self,
    ) -> Option<(u16, tokio::sync::mpsc::Sender<Vec<u8>>, tokio::sync::mpsc::Receiver<Vec<u8>>)> {
        let mut map = self.stream_map.lock().await;
        // All ids but 0 are taken → genuinely exhausted.
        if map.len() >= (u16::MAX as usize) - 1 {
            return None;
        }
        // Guaranteed to terminate: at least one id in 1..=0xFFFE is free.
        let id = loop {
            let candidate = self.next_id.fetch_add(1, Ordering::Relaxed);
            if candidate != STREAM_ID_SINGLE && !map.contains_key(&candidate) {
                break candidate;
            }
        };
        let (down_tx, down_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(256);
        map.insert(id, down_tx);
        Some((id, self.cell_tx.clone(), down_rx))
    }
}

/// Public handle for mux client connections. Clone freely; all clones share
/// the same underlying wire map. One long-lived wire is kept per destination
/// server address and reused across Tor circuits as long as it stays alive.
#[derive(Clone)]
pub struct MuxClientHandle {
    max_read:   usize,
    lab:        Option<Arc<LabLogger>>,
    emitter:    EmitterHandle,
    tail:       TailPolicy,
    /// `server addr → persistent shaped wire`. Normally a single entry (one
    /// bridge); managed mode may add more if circuits target different servers.
    wires:      Arc<tokio::sync::Mutex<HashMap<SocketAddr, Arc<MuxWire>>>>,
}

impl MuxClientHandle {
    pub fn new(
        max_read: usize,
        lab:      Option<Arc<LabLogger>>,
        emitter:  EmitterHandle,
        tail:     TailPolicy,
    ) -> Self {
        Self { max_read, lab, emitter, tail, wires: Arc::new(tokio::sync::Mutex::new(HashMap::new())) }
    }

    /// Eagerly establish the persistent wire to `server` (verified against
    /// `server_pub`) so cover begins flowing before the first circuit arrives
    /// (raw mode, where the bridge address + key are known at startup).
    /// Best-effort: on failure the wire is created lazily on the first circuit.
    pub async fn prewarm(
        &self,
        server:     SocketAddr,
        server_pub: [u8; 32],
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        self.get_or_connect_wire(server, server_pub).await.map(|_| ())
    }

    /// Accept one Tor circuit to `server` (whose identity is `server_pub`, e.g.
    /// from the bridge line's `cert=`), assign it a stream_id on that server's
    /// persistent wire, and drive it to completion.
    pub async fn connect_circuit(
        &self,
        local:      TcpStream,
        server:     SocketAddr,
        server_pub: [u8; 32],
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let wire = self.get_or_connect_wire(server, server_pub).await?;
        let conn_id = self.lab.as_ref().map(|l| l.next_conn_id()).unwrap_or(0);
        let (stream_id, cell_tx, down_rx) = match wire.add_stream().await {
            Some(s) => s,
            // Wire saturated with live circuits: drop this one rather than risk
            // reusing an in-use stream id. (A new wire is created on the next
            // call once some circuits close.)
            None => return Err("mux wire stream-id space exhausted".into()),
        };
        handle_mux_tor_connection(local, stream_id, cell_tx, down_rx, self.max_read, self.lab.clone(), conn_id).await
    }

    async fn get_or_connect_wire(
        &self,
        server:     SocketAddr,
        server_pub: [u8; 32],
    ) -> Result<Arc<MuxWire>, Box<dyn Error + Send + Sync>> {
        let mut guard = self.wires.lock().await;
        if let Some(w) = guard.get(&server) {
            if w.is_alive() { return Ok(w.clone()); }
        }
        let conn_id = self.lab.as_ref().map(|l| l.next_conn_id()).unwrap_or(0);
        let w = Arc::new(MuxWire::connect(
            server, server_pub, self.lab.clone(),
            self.emitter.clone(), self.tail.clone(), conn_id,
        ).await?);
        guard.insert(server, w.clone());
        Ok(w)
    }
}

/// Full-duplex relay between one Tor circuit and its mux stream.
async fn handle_mux_tor_connection(
    local: TcpStream,
    stream_id: u16,
    cell_tx: tokio::sync::mpsc::Sender<Vec<u8>>,
    down_rx: tokio::sync::mpsc::Receiver<Vec<u8>>,
    max_read: usize,
    lab: Option<Arc<LabLogger>>,
    conn_id: u64,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let _ = local.set_nodelay(true);
    let (local_read, local_write) = local.into_split();

    let writer_lab = lab.clone();
    let writer = tokio::spawn(async move {
        mux_tor_writer(down_rx, local_write, writer_lab, conn_id).await
    });
    let reader = tokio::spawn(async move {
        mux_tor_reader(local_read, stream_id, cell_tx, max_read, lab, conn_id).await
    });

    tokio::select! {
        r = writer => r??,
        r = reader => r??,
    }
    Ok(())
}

/// Write decoded downlink bytes from `down_rx` to the Tor circuit.
async fn mux_tor_writer(
    mut down_rx: tokio::sync::mpsc::Receiver<Vec<u8>>,
    mut local_write: tokio::net::tcp::OwnedWriteHalf,
    lab: Option<Arc<LabLogger>>,
    conn_id: u64,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut seq = 0u64;
    while let Some(data) = down_rx.recv().await {
        log_event(&lab, DOWNLINK_STRIP.real_event, conn_id, seq, data.len(), "real");
        seq += 1;
        local_write.write_all(&data).await?;
    }
    let _ = local_write.shutdown().await;
    Ok(())
}

/// Read from the Tor circuit, encode as tagged cells, send into the shared uplink.
/// Sends a close sentinel when the Tor circuit closes.
async fn mux_tor_reader(
    mut local_read: tokio::net::tcp::OwnedReadHalf,
    stream_id: u16,
    cell_tx: tokio::sync::mpsc::Sender<Vec<u8>>,
    max_read: usize,
    lab: Option<Arc<LabLogger>>,
    conn_id: u64,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let max_read = max_read.clamp(1, DEFAULT_MAX_FRAME_LEN);
    let mut pad_rng = StdRng::from_entropy();
    let mut buf = vec![0u8; max_read];
    let mut seq = 0u64;
    loop {
        let n = match local_read.read(&mut buf).await {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) => return Err(e.into()),
        };
        log_event(&lab, UPLINK_LABELS.input_event, conn_id, seq, n, "real");
        seq += 1;
        for chunk in buf[..n].chunks(CELL_PAYLOAD) {
            let cell = encode_real_cell(stream_id, chunk, &mut pad_rng);
            if cell_tx.send(cell).await.is_err() { return Ok(()); }
        }
    }
    let close = encode_close_cell(stream_id, &mut pad_rng);
    let _ = cell_tx.send(close).await;
    Ok(())
}

/// Receive shaped cells from the server, strip dummies, and route real payloads
/// by stream_id to the corresponding per-circuit downlink channel.
async fn receive_mux_inbound(
    mut remote: tokio::net::tcp::OwnedReadHalf,
    mut cipher: FrameCipher,
    stream_map: Arc<tokio::sync::Mutex<HashMap<u16, tokio::sync::mpsc::Sender<Vec<u8>>>>>,
    lab: Option<Arc<LabLogger>>,
    conn_id: u64,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut wire_seq = 0u64;
    let mut buf = vec![0u8; enc_frame_len(CELL_BYTES)];
    while let Some((kind, cell)) = read_sealed_frame(&mut remote, &mut cipher, &mut buf).await? {
        let kind_str = if kind == KIND_REAL { "real" } else { "dummy" };
        log_event(&lab, DOWNLINK_STRIP.wire_event, conn_id, wire_seq, cell.len(), kind_str);
        wire_seq += 1;
        if kind != KIND_REAL { continue; }

        let (stream_id, data) = decode_real_cell(&cell);

        if data.is_empty() {
            // Close sentinel: drop the per-circuit sender; the writer task will
            // see the channel close and shut down the Tor-side write half.
            stream_map.lock().await.remove(&stream_id);
            continue;
        }

        // Clone the sender UNDER the lock, then release it before sending — so a
        // slow circuit can't hold the map lock and block new stream allocation.
        let tx = {
            let map = stream_map.lock().await;
            match map.get(&stream_id) {
                Some(tx) => tx.clone(),
                None => continue,
            }
        };
        // Non-blocking route (same rationale as the server uplink): on Closed the
        // circuit is gone; on Full it's stalled, so drop this cell rather than
        // head-of-line-block every other circuit on the wire.
        match tx.try_send(data.to_vec()) {
            Ok(()) => {}
            Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                stream_map.lock().await.remove(&stream_id);
            }
            Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => { /* drop cell */ }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn log_event(
    lab:     &Option<Arc<LabLogger>>,
    event:   &'static str,
    conn_id: u64,
    seq:     u64,
    bytes:   usize,
    kind:    &'static str,
) {
    if let Some(logger) = lab {
        logger.log(event, conn_id, seq, bytes, kind);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[tokio::test]
    async fn loopback_proxy_round_trip() {
        let payload: Vec<u8> = (0..4096).map(|i| (i % 251) as u8).collect();
        let expected = payload.clone();
        let bridge = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let bridge_addr = bridge.local_addr().unwrap();
        let bridge_task = tokio::spawn(async move {
            let (mut s, _) = bridge.accept().await.unwrap();
            let mut buf = vec![0u8; expected.len()];
            s.read_exact(&mut buf).await.unwrap();
            assert_eq!(buf, expected);
            s.write_all(&buf).await.unwrap();
        });

        let identity = Arc::new(StaticKeypair::generate());
        let server_pub = identity.public_bytes();

        let server = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let server_addr = server.local_addr().unwrap();
        let server_task = tokio::spawn(async move {
            let (s, _) = server.accept().await.unwrap();
            handle_server_connection(s, bridge_addr, 1200, identity).await.unwrap();
        });

        let client_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let client_addr = client_listener.local_addr().unwrap();
        let client_task = tokio::spawn(async move {
            let (s, _) = client_listener.accept().await.unwrap();
            handle_client_connection(s, server_addr, 1200, server_pub).await.unwrap();
        });

        let mut tor = TcpStream::connect(client_addr).await.unwrap();
        tor.write_all(&payload).await.unwrap();
        let mut out = vec![0u8; payload.len()];
        tor.read_exact(&mut out).await.unwrap();
        assert_eq!(out, payload);
        drop(tor);

        client_task.await.unwrap();
        server_task.await.unwrap();
        bridge_task.await.unwrap();
    }

    #[test]
    fn cell_codec_round_trips_all_lengths() {
        let mut rng = StdRng::from_entropy();
        for len in [1usize, 2, 100, 509, 510] {
            let data: Vec<u8> = (0..len).map(|i| (i % 256) as u8).collect();
            let cell = encode_real_cell(0x0042, &data, &mut rng);
            assert_eq!(cell.len(), CELL_BYTES, "every cell must be uniform size");
            let (sid, decoded) = decode_real_cell(&cell);
            assert_eq!(sid, 0x0042, "stream_id must survive the round trip");
            assert_eq!(decoded, &data[..], "len {len} must round-trip");
        }
    }

    #[test]
    fn cell_codec_truncates_oversize_chunk() {
        // The caller splits on CELL_PAYLOAD; a chunk larger than that is clamped.
        let mut rng = StdRng::from_entropy();
        let data = vec![7u8; CELL_PAYLOAD + 50];
        let cell = encode_real_cell(0x0001, &data, &mut rng);
        assert_eq!(cell.len(), CELL_BYTES);
        let (sid, decoded) = decode_real_cell(&cell);
        assert_eq!(sid, 0x0001);
        assert_eq!(decoded.len(), CELL_PAYLOAD);
    }

    #[test]
    fn close_cell_signals_stream_teardown() {
        let mut rng = StdRng::from_entropy();
        let cell = encode_close_cell(0x0007, &mut rng);
        assert_eq!(cell.len(), CELL_BYTES);
        let (sid, data) = decode_real_cell(&cell);
        assert_eq!(sid, 0x0007, "stream_id must be preserved in close cell");
        assert!(data.is_empty(), "close cell must have zero-length payload");
    }

    /// End-to-end round trip across the steady-state departure clock: a 32 KB
    /// payload forces traffic through the unified clock + the inject queue + EOF
    /// drain, and must still arrive byte-for-byte intact.
    #[tokio::test]
    async fn steady_state_round_trip() {
        let payload: Vec<u8> = (0..32 * 1024).map(|i| (i % 251) as u8).collect();
        let expected = payload.clone();
        let bridge_expected = payload.clone();
        let total = payload.len();

        let bridge = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let bridge_addr = bridge.local_addr().unwrap();
        let bridge_task = tokio::spawn(async move {
            let (mut s, _) = bridge.accept().await.unwrap();
            let mut buf = vec![0u8; bridge_expected.len()];
            s.read_exact(&mut buf).await.unwrap();
            assert_eq!(buf, bridge_expected, "bridge must receive the exact real stream");
            s.write_all(&buf).await.unwrap();
        });

        let identity = Arc::new(StaticKeypair::generate());
        let server_pub = identity.public_bytes();

        let server = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let server_addr = server.local_addr().unwrap();
        tokio::spawn(async move {
            let (s, _) = server.accept().await.unwrap();
            handle_server_connection(s, bridge_addr, 1200, identity).await.unwrap();
        });

        let client_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let client_addr = client_listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (s, _) = client_listener.accept().await.unwrap();
            handle_client_connection(s, server_addr, 1200, server_pub).await.unwrap();
        });

        let mut tor = TcpStream::connect(client_addr).await.unwrap();
        let (mut rd, mut wr) = tor.split();
        let writer = async move {
            wr.write_all(&payload).await.unwrap();
            wr.shutdown().await.unwrap();
        };
        let reader = async move {
            let mut out = vec![0u8; total];
            rd.read_exact(&mut out).await.unwrap();
            out
        };
        let (_, out) = tokio::join!(writer, reader);
        assert_eq!(out, expected, "round trip across the clock must be lossless");

        bridge_task.await.unwrap();
    }

    /// Both warmup constants must be 0: the warmup bypass is intentionally
    /// disabled so both directions are shaped from cell 0.  An ISP+end-node
    /// observer would otherwise see a correlated uncovered burst at session
    /// startup on both sides simultaneously and use it to link the client to
    /// the bridge.  The ~3.5 ms per-cell clock interval adds negligible
    /// latency versus the multi-RTT Tor handshake.
    #[test]
    fn class_targets_select_floor_by_volume() {
        // Tail floor in `ByClass` mode is chosen from the session's real volume.
        let (dur_short, cells_short) = class_targets(10 * 1024);
        assert_eq!(dur_short, SessionClass::Short.target_duration());
        assert_eq!(
            cells_short,
            (SessionClass::Short.target_volume_bytes() / CELL_BYTES) as u64
        );

        assert_eq!(
            class_targets(512 * 1024).0,
            SessionClass::Medium.target_duration(),
            "~512 KB real should select the Medium class"
        );
        assert_eq!(
            class_targets(8 * 1024 * 1024).0,
            SessionClass::Long.target_duration(),
            "8 MB real should select the Long class"
        );
        assert_eq!(
            class_targets(20 * 1024 * 1024).0,
            SessionClass::VeryLong.target_duration(),
            "20 MB real should select the VeryLong class"
        );

        // The class lifetime floor is always reachable at the cover rate well
        // within the duration (volume target never dominates).
        let (dur_long, cells_long) = class_targets(8 * 1024 * 1024);
        let cover_cells_per_sec = 1000.0 / 7.0; // ~7 ms mean inter-cell
        let secs_to_volume = cells_long as f64 / cover_cells_per_sec;
        assert!(
            secs_to_volume < dur_long.as_secs_f64(),
            "volume target {cells_long} cells (~{secs_to_volume:.0}s) should be \
             reachable within the {}s lifetime floor",
            dur_long.as_secs()
        );
    }

    /// Two concurrent circuits multiplexed over ONE persistent mux wire must
    /// each round-trip their own bytes through the server's per-stream bridge
    /// connections — and must NOT receive each other's data. This exercises the
    /// production client path end to end: `MuxClientHandle` wire creation +
    /// reuse, stream-ID tagging, server demux to per-stream bridges, reply
    /// re-tagging, and client-side demux back to the right circuit.
    #[tokio::test]
    async fn mux_two_concurrent_circuits_roundtrip_isolated() {
        use std::time::Duration as Dur;

        // Bridge: a simple echo server, one task per accepted connection.
        let bridge = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let bridge_addr = bridge.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut s, _)) = bridge.accept().await {
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 8192];
                    loop {
                        match s.read(&mut buf).await {
                            Ok(0) | Err(_) => break,
                            Ok(n) => {
                                if s.write_all(&buf[..n]).await.is_err() {
                                    break;
                                }
                            }
                        }
                    }
                });
            }
        });

        // Shared bridge identity: server proves it, client verifies it.
        let identity = Arc::new(StaticKeypair::generate());
        let server_pub = identity.public_bytes();

        // mixtor-server accept loop (the mux demux path).
        let server = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let server_addr = server.local_addr().unwrap();
        let server_emitter = EmitterHandle::new();
        tokio::spawn(async move {
            while let Ok((client, _)) = server.accept().await {
                let emitter = server_emitter.clone();
                let identity = identity.clone();
                tokio::spawn(async move {
                    let _ = handle_server_connection_with_lab(
                        client, bridge_addr, 1200, None, emitter, TailPolicy::Off, identity, None,
                    )
                    .await;
                });
            }
        });

        // mixtor-client persistent mux handle + its app-side accept loop.
        let mux = MuxClientHandle::new(1200, None, EmitterHandle::new(), TailPolicy::Off);
        let app = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let app_addr = app.local_addr().unwrap();
        let mux_loop = mux.clone();
        tokio::spawn(async move {
            while let Ok((local, _)) = app.accept().await {
                let mux = mux_loop.clone();
                tokio::spawn(async move {
                    let _ = mux.connect_circuit(local, server_addr, server_pub).await;
                });
            }
        });

        // Drive one circuit: connect to the app listener, send a tagged payload,
        // and read back exactly that many bytes — asserting byte-equality so any
        // cross-circuit leakage (wrong stream demux) fails the test.
        async fn run_circuit(app_addr: SocketAddr, tag: u8) {
            let payload: Vec<u8> = (0..4096).map(|i| tag ^ (i % 251) as u8).collect();
            let expected = payload.clone();
            let tor = TcpStream::connect(app_addr).await.unwrap();
            tor.set_nodelay(true).unwrap();
            let (mut rd, mut wr) = tor.into_split();
            let writer = tokio::spawn(async move {
                wr.write_all(&payload).await.unwrap();
                wr // hold the write half open until the echo is fully read
            });
            let mut out = vec![0u8; expected.len()];
            rd.read_exact(&mut out).await.unwrap();
            assert_eq!(out, expected, "circuit {tag}: echo mismatch — demux leak?");
            drop(writer.await.unwrap());
        }

        // Run both circuits concurrently over the single shared wire. The second
        // `connect_circuit` reuses the wire the first created (map hit).
        let both = async {
            tokio::join!(run_circuit(app_addr, 0x00), run_circuit(app_addr, 0xFF));
        };
        tokio::time::timeout(Dur::from_secs(20), both)
            .await
            .expect("two concurrent mux circuits should round-trip within 20s");
    }
}
