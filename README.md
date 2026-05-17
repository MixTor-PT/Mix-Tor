# MixTor

MixTor is a Rust pluggable transport prototype for adding entry-point mixing in
front of Tor bridge connections.

Current implementation status:

- Step 1: dual 256-bit ephemeral CSPRNG seed foundation with daily rotation.
- Step 2: Seed A-driven packet clumping pool with microclump defaults.
- Step 3: Seed A chaotic timing meta-scheduler.
- Step 4: Seed B non-stationary dummy ratio composition.
- Step 5: verification smoke-test binary.
- Phase 2 start: raw MixTor client/server TCP proxy pair.

## Seed Foundation

`SessionSeeds::generate()` creates two independent process-local seeds:

- Seed A: timing and clumping-pool scheduling.
- Seed B: burst composition and dummy-ratio scheduling.

The seed types are intentionally not clonable, not copyable, not serializable,
and do not expose printable seed material. Seed bytes are zeroized when dropped.

Root seed material rotates at least daily by default. Later engine components
should still change derived timing, pool, and burst-composition behavior on
internal traffic-derived thresholds during the day. Operators should add jitter
around daily rotation rather than rekeying at a crisp wall-clock boundary.

```rust
use mixtor::seeds::SessionSeeds;

let seeds = SessionSeeds::generate()?;

if seeds.should_rotate() {
    seeds.rotate()?;
}
```

Run checks with:

```powershell
cargo test
cargo clippy --all-targets -- -D warnings
cargo run --bin verify
```

## Packet Clumping Pool

`ClumpingPool` buffers incoming packets and flushes them into bursts. Each pool
cycle gets two randomized thresholds derived from Seed A:

- packet threshold: default 2-7 packets
- timeout threshold: default 12-85 ms

These defaults are intentionally microclumps. They destroy direct packet-level
correspondence without introducing the long waits that would make interactive
traffic painful.

```rust
use mixtor::clumping::{ClumpingPool, Packet};
use mixtor::seeds::SessionSeeds;
use std::time::Instant;

let seeds = SessionSeeds::generate()?;
let mut pool = ClumpingPool::new(seeds.timing_and_pool())?;
let now = Instant::now();

let bursts = pool.push(Packet::new(b"payload".to_vec())?, now);

if let Some(timeout_burst) = pool.tick(now) {
    // forward timeout_burst to the next engine stage
}
```

Handled edge cases:

- single packet before timeout: flushed when the randomized timeout expires
- bursty overflow: split across bursts with no data loss
- adversarial probing: size and timeout thresholds regenerate every cycle

## Seed A Timing Scheduler

`SeedATimingScheduler` emits inter-burst timing decisions from six distribution
families:

- Poisson
- Pareto
- bimodal
- burst-and-silence
- uniform
- exponential

Regime switches are driven by randomized byte and packet thresholds, not
wall-clock time. Each timing decision also carries fresh clumping pool
thresholds for callers that want one Seed A source of truth for pool parameters.

## Seed B Burst Composition

`SeedBComposer` converts real bursts into mixed bursts containing real and dummy
frames. The dummy ratio distribution is itself non-stationary, switching across
sparse, balanced, heavy-cover, and spiky regimes based on burst, byte, and
packet counters.

Dummy packet sizes are sampled from real packet sizes in the burst, and frames
are shuffled before output so the real packet order is not exposed as a simple
prefix followed by dummies.

## Verification

Before PT integration, run:

```powershell
cargo run --bin verify
```

The verifier simulates independent sessions and reports:

- cross-session timing correlation
- cross-session dummy-ratio correlation
- real burst structure versus mixed burst structure correlation
- dummy-ratio window drift across one session
- packet-order prefix leakage rate after real/dummy frame shuffling

This is a statistical smoke test, not a formal anonymity proof. Treat failures
as a sign that the current parameters are too stationary, too correlated, or too
close to one-packet-in/one-packet-out behavior.

## Raw PT Proxy Pair

This is the first runnable bridge-side slice. It supports raw proxy mode and a
managed Tor client transport mode, giving us the network path:

```text
Tor client -> mixtor-client -> mixtor-server -> local Tor bridge daemon
```

Start the server next to a local Tor bridge daemon:

```powershell
cargo run --bin mixtor-server -- --listen 0.0.0.0:9001 --upstream 127.0.0.1:9002
```

Start the client where Tor will connect locally:

```powershell
cargo run --bin mixtor-client -- --listen 127.0.0.1:9000 --server BRIDGE_IP:9001
```

Current behavior:

- client accepts local TCP traffic
- client clumps packets, applies Seed A timing, adds Seed B dummy frames
- server strips dummy frames
- server forwards only real payloads to the upstream bridge daemon
- bridge replies are sent back as real frames
- Tor can launch the client through `ClientTransportPlugin`

## Local Tor Bridge Test

The client binary can now run as a Tor managed client transport. Tor launches it
through `ClientTransportPlugin`, receives a SOCKS5 listener, and connects to the
MixTor server address from the `Bridge mixtor ...` line.

Build the binaries:

```powershell
cargo build --bins
```

Start a local Tor bridge daemon with:

```powershell
tor -f examples/local-bridge.torrc
```

In another terminal, start MixTor server in front of that bridge daemon:

```powershell
cargo run --bin mixtor-server -- --listen 127.0.0.1:9001 --upstream 127.0.0.1:9002
```

In another terminal, start a Tor client using MixTor:

```powershell
tor -f examples/local-client.torrc
```

The local client config contains:

```text
ClientTransportPlugin mixtor exec C:\Users\Thanthal\Documents\Mixtor\target\debug\mixtor-client.exe
Bridge mixtor 127.0.0.1:9001
```

If `cargo` is still not visible in your PowerShell session, use:

```powershell
& "$env:USERPROFILE\.cargo\bin\cargo.exe" build --bins
```

## Lab Traffic Logs

For local attack-simulation only, both binaries support temporary CSV logging:

```powershell
cargo run --bin mixtor-server -- --listen 127.0.0.1:9001 --upstream 127.0.0.1:9002 --lab-log .\target\lab-logs
cargo run --bin mixtor-client -- --listen 127.0.0.1:9000 --server 127.0.0.1:9001 --lab-log .\target\lab-logs
```

This writes:

- `target/lab-logs/client.csv`
- `target/lab-logs/server.csv`

Columns:

```text
unix_nanos,role,event,conn_id,seq,bytes,kind
```

Useful events:

- `client_input`: bytes entering MixTor before clumping/mixing
- `client_wire_output`: real and dummy frames leaving the client
- `server_wire_input`: real and dummy frames arriving at the server
- `server_real_output`: real bytes after dummy stripping

These logs intentionally weaken privacy. Use them only with synthetic/local test
traffic and delete them immediately after analysis:

```powershell
Remove-Item -Recurse -Force .\target\lab-logs
```
