# MixTor

MixTor is a pluggable transport for Tor, written in Rust. It hides the timing,
size, and volume of a Tor bridge connection behind a constant-rate stream of
fixed-size encrypted cells, so that an observer watching the connection cannot
correlate it with the traffic it carries. It fills the same role as obfs4, but
its focus is resistance to traffic-analysis and flow-correlation attacks, not
only making the protocol bytes look random.

## Threat model

MixTor defends against a passive observer on the network path between a Tor user
and a MixTor bridge. That observer sees every byte on the wire: when each packet
is sent, how large it is, and how much data flows in total. The same observer, or
a colluding one, may also watch traffic elsewhere, for example near a site the
user visits, and try to match the two sides by timing or volume. This is the
flow-correlation attack that defeats an ordinary encrypted tunnel, whose packet
timing and sizes still track the traffic inside it.

MixTor also assumes an active prober that connects to the bridge and sends
arbitrary bytes to learn whether it is a MixTor bridge. The transport stays silent
to any peer that cannot prove it already knows the bridge's public key.

The design does not defend against an adversary who controls the bridge itself,
nor against long-term end-to-end confirmation by an adversary who already watches
both the user's entry link and their final destination at the packet level. Tor's
own circuit design addresses parts of that space. MixTor's job is the first hop.

## How it works

### The data path

MixTor runs as two managed pluggable transports that Tor launches itself:

```
Tor client  ->  mixtor-client  ->  shaped encrypted wire  ->  mixtor-server  ->  Tor bridge ORPort  ->  Tor network
```

On the user's machine, Tor starts `mixtor-client` through a `ClientTransportPlugin`
line and hands it traffic over SOCKS. On the bridge, Tor starts `mixtor-server`
through a `ServerTransportPlugin` line. The two speak MixTor's wire protocol to
each other, and on the bridge side `mixtor-server` forwards the recovered traffic
to the local Tor ORPort. Neither binary is started by hand in normal use. Tor
manages both.

### The shaped wire

The central idea is that the wire never reflects the real traffic. A dedicated
emitter thread sends one fixed-size cell at a fixed interval, around 285 cells per
second, for the whole life of the connection. When real data is waiting, the next
cell carries it. When no real data is waiting, the next cell carries random
padding instead. Either way a cell goes out on schedule and at the same size, so
the rate and size of the wire stay constant and independent of what the user is
doing.

Each cell is 514 bytes, the size of a Tor cell. A real cell holds a short header
(a stream identifier and a length) followed by payload and then random padding out
to the fixed size. A padding cell cannot be told apart from a real one once
encrypted, because the field that marks a cell as real or padding sits inside the
encrypted region described below.

This is what defeats correlation. An observer sees a steady stream of identical
encrypted cells whose timing and size carry no information about the traffic
underneath.

### Multiplexing

Several Tor circuits share one MixTor wire. Each cell header carries a stream
identifier, so `mixtor-server` can split the recovered traffic back into separate
connections to the bridge. The wire is opened once and kept alive, so individual
circuit setup and teardown do not show up as separate connections that an observer
could count or time. If one circuit stalls, its cell is dropped rather than allowed
to block the shared wire, and Tor's own integrity checks then tear that circuit
down.

### Encryption and the handshake

Before any traffic flows, the two sides run a handshake that authenticates the
bridge and sets up the frame keys.

The handshake is an ntor-style X25519 exchange. The bridge has a long-term key
pair, and its public key is the `cert` value that operators give to clients on the
Bridge line. The client knows that public key in advance, so it can confirm it is
talking to the right bridge, while the bridge accepts any client, which is the
correct behavior for a bridge that anyone may use. The shared secret is derived
with BLAKE3 and verified in constant time.

The handshake is also disguised so it does not look like a key exchange. Each
side's ephemeral public key is sent as an Elligator2 representative, a string of
bytes that is computationally indistinguishable from random, rather than as a
recognizable curve point. To stop an active prober from confirming the bridge,
each handshake message carries a keyed marker and authentication tag derived from
the bridge's public key, following the obfs4 design. The bridge checks the marker
and tag before it answers. A peer that does not already know the bridge's key
cannot produce them, so the bridge reads the bytes, fails the check, and stays
silent for a random delay instead of replying. There is no response for a prober
to fingerprint.

Once the handshake completes, every frame on the wire is sealed with
ChaCha20-Poly1305 under a per-direction counter used as the nonce. The byte that
marks a cell as real or padding is sealed inside the ciphertext, and every frame
is the same size, so on the wire there is no magic number, no length prefix, and
no visible type. Each frame is one opaque fixed-size blob. Any tampering or
desynchronization makes the authentication fail and closes the connection.

### Epoch alignment and the tail

A constant rate hides traffic within a connection, but the moments a connection
starts and stops can still leak. MixTor divides time into fixed epochs. A new
connection holds its first real cell until the next epoch boundary, so connections
that start at slightly different times begin carrying real traffic together. At
the end, a connection keeps emitting cover past the real data until it has reached
a duration drawn from its session class, then stops on an epoch boundary, so
connections also tend to end together rather than at the exact moment their real
traffic stops. Start alignment and tail length are configured separately through
the `MIXTOR_TAIL_FLOOR_MS` setting.

## Build

```
cargo build --release --bins
```

This produces `mixtor-server`, `mixtor-client`, and `mixtor-correlate` in
`target/release`.

## Running it

MixTor is meant to run as a managed transport under Tor, configured through torrc.
The bridge generates or pins an identity key and advertises the matching `cert`,
and clients put that `cert` on their Bridge line. The full setup, including the
bridge and client torrc files and the key workflow, is in
[DEPLOYMENT.md](DEPLOYMENT.md).

## Testing and validation

Run the unit and integration tests with:

```
cargo test
```

MixTor includes a correlation test harness that measures how much the shaped wire
leaks. The `mixtor-correlate` binary reads lab-log captures and scores a set of
attacks: telling real cells from padding by size, correlating the wire's volume
against the real traffic at several time scales, linking separate wire flows to
their real flows, and ranking flows by total volume or duration. The scripts under
`localtest` drive this end to end over a real local Tor bridge, and a learned CNN
attack under `localtest/dl` provides a stronger check than the built-in linear
scores. Lab logging writes traffic timing to disk and weakens privacy by design,
so it is for local testing only, never for real traffic.

## Repository layout

- `src/transport.rs`: the shaped wire, the emitter, multiplexing, and the epoch model.
- `src/crypto.rs`: the handshake, Elligator2 obfuscation, anti-probing, and the frame cipher.
- `src/composition.rs`, `src/timing.rs`, `src/session_bounder.rs`: traffic shaping and session-class policy.
- `src/socks.rs`: the SOCKS interface to Tor, including the `cert` handoff.
- `src/correlation_attack.rs`: the scored attacks used by the analyzer.
- `src/bin`: the `mixtor-server`, `mixtor-client`, and `mixtor-correlate` programs.
- `DEPLOYMENT.md`: how to run a bridge and a client under Tor.

## Security status

MixTor is a research prototype. The cryptography is built from standard,
well-reviewed primitives, but the way they are composed into the handshake has not
had an independent review, and the correlation results come from a local test
setup rather than a deployed network. Do not rely on it to protect real users yet.
It is published for study and review.

## License

MIT OR Apache-2.0.
