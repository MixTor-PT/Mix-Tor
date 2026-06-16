use crate::clumping::Packet;
use crate::composition::{CompositionConfig, MixedFrame, ShapedSession};
use crate::lab::LabLogger;
use crate::protocol::{read_frame, write_frame, FrameKind, WireFrame, DEFAULT_MAX_FRAME_LEN};
use crate::seeds::SessionSeeds;
use crate::session_bounder::SessionClass;
use rand::{rngs::StdRng, RngCore, SeedableRng};
use std::error::Error;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::time::{sleep_until, Instant as TokioInstant};

// ---------------------------------------------------------------------------
// Uniform-cell framing.
//
// Every outbound frame on the client→server link — real or dummy — is exactly
// `CELL_BYTES` bytes, so a wire observer cannot separate real frames from
// dummies by size. A real cell carries its true length in a 2-byte prefix
// (hidden under the deployment's wire encryption) so the server can strip the
// random padding before forwarding to the bridge:
//
//   real cell:  [u16 BE len][len bytes of real data][random padding]  == CELL_BYTES
//   dummy cell: [CELL_BYTES random bytes]
// ---------------------------------------------------------------------------
const CELL_BYTES: usize = 514; // Tor cell size
const CELL_HEADER: usize = 2; // u16 length prefix
const CELL_PAYLOAD: usize = CELL_BYTES - CELL_HEADER; // 512 real bytes per cell

/// Encode up to `CELL_PAYLOAD` real bytes into exactly `CELL_BYTES`.
fn encode_real_cell(data: &[u8], pad_rng: &mut impl RngCore) -> Vec<u8> {
    let n = data.len().min(CELL_PAYLOAD);
    let mut cell = vec![0u8; CELL_BYTES];
    cell[..CELL_HEADER].copy_from_slice(&(n as u16).to_be_bytes());
    cell[CELL_HEADER..CELL_HEADER + n].copy_from_slice(&data[..n]);
    pad_rng.fill_bytes(&mut cell[CELL_HEADER + n..]);
    cell
}

/// Recover the real bytes from a `CELL_BYTES` real-frame payload.
fn decode_real_cell(cell: &[u8]) -> &[u8] {
    if cell.len() < CELL_HEADER {
        return &[];
    }
    let n = u16::from_be_bytes([cell[0], cell[1]]) as usize;
    let end = (CELL_HEADER + n).min(cell.len());
    &cell[CELL_HEADER..end]
}

// ---------------------------------------------------------------------------
// Warmup — protects Tor's connection setup from the departure clock.
//
// During warmup, real cells bypass the clock and go out immediately (still as
// uniform cells; the cover clock runs concurrently and the warmup credit keeps
// it from piling on top, so the window matches the steady cover rate). This
// avoids adding the clock's per-cell latency to Tor's multi-round-trip link
// handshake (VERSIONS/CERTS/AUTH/NETINFO) and first circuit build.
//
// Warmup is measured by real bytes sent and is **per direction**:
// - Uplink (`WARMUP_BYTES`, 64 KB): the client *initiates* — its setup sends are
//   latency-critical and small/sub-cover-rate, so the credit fully masks them; a
//   generous window keeps the client handshake off the clock.
// - Downlink (`DOWNLINK_WARMUP_BYTES`, 0): the server→client stream only ever
//   *responds*; it has no latency-critical first-send to protect. Any nonzero
//   downlink warmup is consumed by the first *web response* (which arrives faster
//   than the cover rate, so the credit can't mask it) and leaks the download
//   envelope as a start-of-session spike. Shaping the downlink from the first
//   cell queues/back-pressures the first response so the wire is flat from t=0,
//   at the cost of ~one clock interval (~7 ms) of latency on each relay response
//   cell during setup — a one-time cost, never a stall.
// ---------------------------------------------------------------------------
const WARMUP_BYTES: u64 = 64 * 1024;
const DOWNLINK_WARMUP_BYTES: u64 = 0;

pub async fn handle_client_connection(
    local: TcpStream,
    server: SocketAddr,
    max_read: usize,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    // No shared emitter supplied (single-connection convenience / tests): spin
    // up a private one. Real multi-flow deployments share one across the accept
    // loop via the `_with_lab` entry point.
    handle_client_connection_with_lab(local, server, max_read, None, EmitterHandle::new()).await
}

pub async fn handle_client_connection_with_lab(
    local: TcpStream,
    server: SocketAddr,
    max_read: usize,
    lab: Option<Arc<LabLogger>>,
    emitter: EmitterHandle,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let conn_id = lab.as_ref().map(|l| l.next_conn_id()).unwrap_or(0);
    let remote  = TcpStream::connect(server).await?;
    // Disable Nagle so the departure clock's per-cell spacing reaches the wire
    // intact instead of being coalesced by the kernel (esp. the `remote` wire
    // side, which carries the shaped cells). Best-effort.
    let _ = local.set_nodelay(true);
    let _ = remote.set_nodelay(true);
    let (local_read,  local_write)  = local.into_split();
    let (remote_read, remote_write) = remote.into_split();
    let max_read = max_read.clamp(1, DEFAULT_MAX_FRAME_LEN);

    let outbound_lab = lab.clone();
    let outbound = tokio::spawn(async move {
        send_mixed_outbound(local_read, remote_write, max_read, outbound_lab, conn_id, emitter).await
    });
    let inbound = tokio::spawn(async move {
        receive_real_inbound(remote_read, local_write, lab, conn_id).await
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
) -> Result<(), Box<dyn Error + Send + Sync>> {
    // No shared emitter supplied (single-connection convenience / tests): spin
    // up a private one. Real multi-flow deployments share one across the accept
    // loop via the `_with_lab` entry point.
    handle_server_connection_with_lab(client, upstream, max_read, None, EmitterHandle::new()).await
}

pub async fn handle_server_connection_with_lab(
    client: TcpStream,
    upstream: SocketAddr,
    max_read: usize,
    lab: Option<Arc<LabLogger>>,
    emitter: EmitterHandle,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let conn_id = lab.as_ref().map(|l| l.next_conn_id()).unwrap_or(0);
    let bridge  = TcpStream::connect(upstream).await?;
    // Disable Nagle so the departure clock's per-cell spacing reaches the wire
    // intact (esp. the `client` wire side, which carries the shaped downlink).
    let _ = client.set_nodelay(true);
    let _ = bridge.set_nodelay(true);
    let (client_read, client_write) = client.into_split();
    let (bridge_read, bridge_write) = bridge.into_split();

    let inbound_lab = lab.clone();
    let inbound = tokio::spawn(async move {
        strip_dummies_to_bridge(client_read, bridge_write, inbound_lab, conn_id).await
    });
    let outbound = tokio::spawn(async move {
        send_bridge_replies(bridge_read, client_write, max_read, lab, conn_id, emitter).await
    });

    tokio::select! {
        result = inbound  => result??,
        result = outbound => result??,
    }
    Ok(())
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
    /// Real bytes that bypass the clock at session start (per-direction warmup).
    warmup_bytes: u64,
}

const UPLINK_LABELS: ShapeLabels = ShapeLabels {
    input_event: "client_input",
    wire_event: "client_wire_output",
    warmup_bytes: WARMUP_BYTES,
};
const DOWNLINK_LABELS: ShapeLabels = ShapeLabels {
    input_event: "server_bridge_reply",
    wire_event: "server_wire_return",
    warmup_bytes: DOWNLINK_WARMUP_BYTES,
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
// It is OFF by default: a minimum lifetime is real bandwidth+latency overhead and
// a session-length policy the operator must opt into. `MIXTOR_TAIL_FLOOR_MS`
// selects the policy:
//   unset / 0   → OFF — `pad_tail` is a no-op; the connection closes on drain.
//   <N> (ms)    → fixed N-ms floor for every connection (used by tests/demos).
//   "class"     → volume-based `SessionClass` floor: the session's real byte
//                 count picks Short/Medium/Long (30 s / 5 min / 30 min and the
//                 matching volume target). This is the principled production
//                 policy — all sessions in a class converge to the same lifetime
//                 and total volume, so neither ranks flows. See `session_bounder`.
const TAIL_FLOOR_ENV: &str = "MIXTOR_TAIL_FLOOR_MS";
const TAIL_VOLUME_QUANTUM_CELLS: u64 = 256; // ~128 KB of wire (fixed-floor mode)

/// Tail-cover policy parsed from `MIXTOR_TAIL_FLOOR_MS`.
enum TailPolicy {
    Off,
    Fixed(Duration),
    ByClass,
}

fn tail_policy() -> TailPolicy {
    match std::env::var(TAIL_FLOOR_ENV) {
        Ok(v) if v.eq_ignore_ascii_case("class") => TailPolicy::ByClass,
        Ok(v) => match v.parse::<u64>() {
            Ok(ms) if ms > 0 => TailPolicy::Fixed(Duration::from_millis(ms)),
            _ => TailPolicy::Off,
        },
        Err(_) => TailPolicy::Off,
    }
}

/// The `(lifetime floor, wire-volume target in cells)` the tail should reach,
/// or `None` when tail cover is disabled. `real_bytes_sent` selects the
/// `SessionClass` in `ByClass` mode; `wire_seq` rounds up the volume target in
/// `Fixed` mode.
fn tail_targets(real_bytes_sent: u64, wire_seq: u64) -> Option<(Duration, u64)> {
    match tail_policy() {
        TailPolicy::Off => None,
        TailPolicy::Fixed(d) => {
            let q = TAIL_VOLUME_QUANTUM_CELLS;
            Some((d, wire_seq.saturating_add(q - 1) / q * q))
        }
        TailPolicy::ByClass => Some(class_targets(real_bytes_sent)),
    }
}

/// Map a session's real byte volume to its `SessionClass` lifetime floor and
/// total-wire-volume target (in cells). Pure — unit-tested without env.
fn class_targets(real_bytes_sent: u64) -> (Duration, u64) {
    let class = SessionClass::for_volume(real_bytes_sent as usize);
    let vol_cells = (class.target_volume_bytes() / CELL_BYTES) as u64;
    (class.target_duration(), vol_cells)
}

/// The fixed-rate cover-clock configuration shared by both directions.
fn shaped_composition_config() -> CompositionConfig {
    CompositionConfig {
        cell_bytes: Some(CELL_BYTES),
        // Fixed-rate departure clock: a homogeneous Poisson process at a
        // constant ~3.5 ms mean inter-cell time (~285 cells/s, ~147 KB/s of
        // cover per direction) that is *independent of real-traffic timing*.
        // Real cells occupy slots as they queue and idle slots are filled with
        // dummies, so the wire's packet/byte envelope is flat whether or not
        // real traffic is flowing — there is no rate signal for a correlator to
        // lock onto at any lag. Exponential (not fixed) intervals keep the
        // spectrum flat (no periodic keepalive peak). The wide [min,max] band
        // avoids truncating the exponential. Adaptation is OFF because bursty
        // per-cell injection would otherwise drag the rate up in exactly the
        // active windows, re-coupling wire rate to real volume.
        //
        // Rate sized for headroom over real download bandwidth: a multi-flow
        // real-Tor test showed that when sustained download approached the cover
        // rate, the downlink wire envelope weakly tracked real volume at coarse
        // (>=1 s) timescales (the queue backs up in busy windows, so catch-up
        // draining bursts correlate with load). Keeping cover well above typical
        // browsing download rate keeps the wire cover-dominated and flat.
        clock_adapt_rate:     Some(false),
        clock_ema_alpha:      Some(0.05),
        clock_min_iat_ms:     Some(1.0),
        clock_initial_iat_ms: Some(3.5),
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
const STEADY_CHANNEL_CAP: usize = 256;

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
    mut wire: tokio::net::tcp::OwnedWriteHalf,
    max_read: usize,
    lab: Option<Arc<LabLogger>>,
    conn_id: u64,
    labels: ShapeLabels,
    emitter: EmitterHandle,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let seeds = SessionSeeds::generate()?;
    let mut session = ShapedSession::new(seeds.burst_composition(), shaped_composition_config())?;

    let mut pad_rng = StdRng::from_entropy();
    let mut buffer = vec![0u8; max_read.max(1)];
    let mut input_seq = 0u64;
    let mut wire_seq = 0u64;
    let mut real_bytes_sent: u64 = 0;
    let session_start = std::time::Instant::now();
    let mut warmup_credit: u64 = 0;

    // --- Phase 1: warmup (single task; latency-critical, short, uplink-only). ---
    while real_bytes_sent < labels.warmup_bytes {
        let next = TokioInstant::from_std(session.next_departure());
        tokio::select! {
            _ = sleep_until(next) => {
                while let Some(frame) = session.tick() {
                    // A warmup real cell already filled this cover slot: swallow
                    // the dummy rather than pile it on top, so warmup doesn't run hot.
                    if warmup_credit > 0 && frame.is_dummy() {
                        warmup_credit -= 1;
                        continue;
                    }
                    emit_frame(&mut wire, frame, &lab, conn_id, &mut wire_seq, labels.wire_event).await?;
                }
            }
            r = src.read(&mut buffer) => {
                match r {
                    Ok(0) => {
                        drain_session(&mut session, &mut wire, &lab, conn_id, &mut wire_seq, labels.wire_event).await?;
                        pad_tail(&mut session, &mut wire, &lab, conn_id, &mut wire_seq, labels.wire_event, session_start, real_bytes_sent).await;
                        let _ = wire.shutdown().await;
                        return Ok(());
                    }
                    Ok(n) => {
                        log_event(&lab, labels.input_event, conn_id, input_seq, n, "real");
                        input_seq += 1;
                        real_bytes_sent = real_bytes_sent.saturating_add(n as u64);
                        for chunk in buffer[..n].chunks(CELL_PAYLOAD) {
                            let cell = encode_real_cell(chunk, &mut pad_rng);
                            let frame = WireFrame::real(cell)?;
                            log_event(&lab, labels.wire_event, conn_id, wire_seq, frame.payload().len(), "real");
                            wire_seq += 1;
                            write_frame(&mut wire, &frame).await?;
                            warmup_credit = warmup_credit.saturating_add(1);
                        }
                    }
                    Err(e) => return Err(e.into()),
                }
            }
        }
    }

    // --- Phase 2: steady state. Hand the wire + shaped session to the shared
    // emitter and become a pure reader: split source bytes into cells and feed
    // them over a bounded channel. The emitter clocks emission for ALL flows
    // from its own timer, so this flow's wire timing no longer depends on how
    // this task (or any sibling) happens to be scheduled — closing the
    // many-sparse-flows contention leak and any machine-timing fingerprint. ---
    let (cell_tx, cell_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(STEADY_CHANNEL_CAP);
    emitter.register(FlowSlot {
        wire,
        session,
        cell_rx,
        lab: lab.clone(),
        conn_id,
        wire_event: labels.wire_event,
        wire_seq,
        session_start,
        pending: Vec::new(),
        reader_done: false,
        tail: None,
        finish_by: None,
    });

    // From here this task is a pure reader; the emitter owns emission and
    // derives the session's real volume from its emission counts for the tail
    // floor, so `real_bytes_sent` is no longer advanced past warmup.
    loop {
        let n = match src.read(&mut buffer).await {
            Ok(0) => break, // EOF: dropping cell_tx tells the emitter to drain + pad + close.
            Ok(n) => n,
            Err(e) => return Err(e.into()), // cell_tx drops → the emitter still finishes the tail.
        };
        log_event(&lab, labels.input_event, conn_id, input_seq, n, "real");
        input_seq += 1;
        for chunk in buffer[..n].chunks(CELL_PAYLOAD) {
            let cell = encode_real_cell(chunk, &mut pad_rng);
            if cell_tx.send(cell).await.is_err() {
                // Emitter dropped this flow (peer closed); stop reading.
                return Ok(());
            }
        }
    }
    drop(cell_tx); // signal EOF; the emitter drains the queue, pads the tail, closes the wire.
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

/// Max real cells a single flow may ingest from its channel per emitter tick.
/// The clock drains only ~1 cell per few ticks, so this is far above what is
/// needed to keep the queue fed; its purpose is to bound the *work* one
/// `service_flow` call can do, so a large arriving burst is spread over several
/// ticks instead of stalling the shared emitter for one long synchronous pull.
const MAX_PULL_PER_TICK: usize = 32;

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
    /// `(lifetime floor, wire-volume target in cells)` to reach before closing,
    /// or `None` when tail cover is disabled. Computed once at reader EOF.
    tail: Option<(Duration, u64)>,
    /// Hard backstop computed at reader EOF: force-close past this instant even
    /// if `pending` cannot flush (peer stopped reading), so a flow is never kept
    /// alive indefinitely.
    finish_by: Option<std::time::Instant>,
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

    // 1) Pull queued real cells into the session while it has room and we are
    //    not already buffer-bound, up to MAX_PULL_PER_TICK so one busy flow
    //    can't stall the shared emitter with a long synchronous pull.
    //    Non-blocking — reads never gate emission.
    if !flow.reader_done && flow.pending.len() < PENDING_CAP {
        let mut pulled = 0;
        while pulled < MAX_PULL_PER_TICK && flow.session.has_queue_space() {
            match flow.cell_rx.try_recv() {
                // Packet::new only fails on empty input; cells are always
                // CELL_BYTES, and has_queue_space() guards inject_real — so both
                // error arms are unreachable in practice.
                Ok(cell) => {
                    if let Ok(packet) = Packet::new(cell) {
                        let _ = flow.session.inject_real(packet);
                    }
                    pulled += 1;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    // Reader EOF: fix the tail target from the real volume emitted.
                    flow.reader_done = true;
                    let real_bytes = flow
                        .session
                        .emission_counts()
                        .0
                        .saturating_mul(CELL_PAYLOAD as u64);
                    flow.tail = tail_targets(real_bytes, flow.wire_seq);
                    let floor = flow.tail.map(|(f, _)| f).unwrap_or_default();
                    flow.finish_by =
                        Some(std::time::Instant::now() + floor + Duration::from_secs(5));
                    break;
                }
            }
        }
    }

    // 2) Tick the per-flow departure clock; encode each due frame into `pending`.
    if flow.pending.len() < PENDING_CAP {
        while let Some(frame) = flow.session.tick() {
            let (bytes, kind, plen) = encode_frame(frame);
            log_event(&flow.lab, flow.wire_event, flow.conn_id, flow.wire_seq, plen, kind);
            flow.wire_seq += 1;
            flow.pending.extend_from_slice(&bytes);
            if flow.pending.len() >= PENDING_CAP {
                break;
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
            Some((floor, vol)) => flow.session_start.elapsed() >= floor && flow.wire_seq >= vol,
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

/// Encode a mixed frame to its on-wire bytes, returning
/// `(bytes, kind-label, payload-len)` for the wire and the lab log.
fn encode_frame(frame: MixedFrame) -> (Vec<u8>, &'static str, usize) {
    let (wire_frame, kind) = match frame {
        // Real/dummy cells are exactly CELL_BYTES, so WireFrame::{real,dummy}
        // (which only reject empty / >64 KB payloads) never fail here.
        MixedFrame::Real(packet) => (
            WireFrame::real(packet.into_bytes()).expect("real cell is a valid frame payload"),
            "real",
        ),
        MixedFrame::Dummy(bytes) => (
            WireFrame::dummy(bytes).expect("dummy cell is a valid frame payload"),
            "dummy",
        ),
    };
    let plen = wire_frame.payload().len();
    (wire_frame.encode(), kind, plen)
}

/// Outbound path (client → server): shape the request/uplink traffic.
async fn send_mixed_outbound(
    local: tokio::net::tcp::OwnedReadHalf,
    remote: tokio::net::tcp::OwnedWriteHalf,
    max_read: usize,
    lab: Option<Arc<LabLogger>>,
    conn_id: u64,
    emitter: EmitterHandle,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    shaped_forward(local, remote, max_read, lab, conn_id, UPLINK_LABELS, emitter).await
}

/// Emit one shaped frame onto the wire, logging and tagging real vs dummy.
async fn emit_frame(
    wire_out:   &mut tokio::net::tcp::OwnedWriteHalf,
    frame:      MixedFrame,
    lab:        &Option<Arc<LabLogger>>,
    conn_id:    u64,
    wire_seq:   &mut u64,
    wire_event: &'static str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let (frame, kind) = match frame {
        MixedFrame::Real(packet) => (WireFrame::real(packet.into_bytes())?, "real"),
        MixedFrame::Dummy(bytes) => (WireFrame::dummy(bytes)?, "dummy"),
    };
    log_event(lab, wire_event, conn_id, *wire_seq, frame.payload().len(), kind);
    *wire_seq += 1;
    write_frame(wire_out, &frame).await?;
    Ok(())
}

/// Drain all queued real cells on the departure clock before shutdown so no
/// real data is lost. Paces on the clock to avoid an end-of-session burst.
async fn drain_session(
    session:    &mut ShapedSession,
    wire_out:   &mut tokio::net::tcp::OwnedWriteHalf,
    lab:        &Option<Arc<LabLogger>>,
    conn_id:    u64,
    wire_seq:   &mut u64,
    wire_event: &'static str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    while session.queue_depth() > 0 {
        let d = TokioInstant::from_std(session.next_departure());
        sleep_until(d).await;
        if let Some(frame) = session.tick() {
            emit_frame(wire_out, frame, lab, conn_id, wire_seq, wire_event).await?;
        }
    }
    Ok(())
}

/// Pad the end of a connection with cover so its lifetime and total wire volume
/// reach the targets from `tail_targets` (a fixed floor, or a `SessionClass`
/// floor chosen by `real_bytes_sent`), rather than ending exactly when real
/// traffic stops. Best-effort: a write error (the peer has already closed) ends
/// padding cleanly without propagating.
async fn pad_tail(
    session:       &mut ShapedSession,
    wire_out:      &mut tokio::net::tcp::OwnedWriteHalf,
    lab:           &Option<Arc<LabLogger>>,
    conn_id:       u64,
    wire_seq:        &mut u64,
    wire_event:      &'static str,
    session_start:   std::time::Instant,
    real_bytes_sent: u64,
) {
    let (floor, vol_target) = match tail_targets(real_bytes_sent, *wire_seq) {
        Some(t) => t,
        None => return, // tail cover disabled (default): close on drain
    };
    // Safety deadline: never pad more than the floor plus a small slack, so the
    // loop always terminates even if a slot somehow never fires.
    let pad_deadline = session_start + floor + Duration::from_secs(5);
    loop {
        let dur_ok = session_start.elapsed() >= floor;
        let vol_ok = *wire_seq >= vol_target;
        if (dur_ok && vol_ok) || std::time::Instant::now() >= pad_deadline {
            return;
        }
        let d = TokioInstant::from_std(session.next_departure());
        sleep_until(d).await;
        while let Some(frame) = session.tick() {
            if emit_frame(wire_out, frame, lab, conn_id, wire_seq, wire_event)
                .await
                .is_err()
            {
                return;
            }
        }
    }
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

/// Read uniform cells from `wire`, drop dummies, unpad real cells back to their
/// true bytes, and forward them in order to `out`. Direction-agnostic.
async fn strip_and_forward(
    mut wire: tokio::net::tcp::OwnedReadHalf,
    mut out:  tokio::net::tcp::OwnedWriteHalf,
    lab:      Option<Arc<LabLogger>>,
    conn_id:  u64,
    labels:   StripLabels,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut wire_seq   = 0u64;
    let mut output_seq = 0u64;

    while let Some(frame) = read_frame(&mut wire).await? {
        let kind = match frame.kind() {
            FrameKind::Real  => "real",
            FrameKind::Dummy => "dummy",
        };
        log_event(&lab, labels.wire_event, conn_id, wire_seq, frame.payload().len(), kind);
        wire_seq += 1;

        // Dummies are dropped; real cells are unpadded back to their true bytes
        // and forwarded in order.
        if frame.kind() == FrameKind::Real {
            let data = decode_real_cell(frame.payload());
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
    lab:     Option<Arc<LabLogger>>,
    conn_id: u64,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    strip_and_forward(remote, local, lab, conn_id, DOWNLINK_STRIP).await
}

/// Inbound on the server (client → bridge): strip the shaped uplink to plaintext.
async fn strip_dummies_to_bridge(
    client:  tokio::net::tcp::OwnedReadHalf,
    bridge:  tokio::net::tcp::OwnedWriteHalf,
    lab:     Option<Arc<LabLogger>>,
    conn_id: u64,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    strip_and_forward(client, bridge, lab, conn_id, UPLINK_STRIP).await
}

/// Downlink path (server → client): shape the bulk reply/download traffic on the
/// same fixed-rate clock as the uplink — previously an unshaped raw passthrough.
async fn send_bridge_replies(
    bridge:   tokio::net::tcp::OwnedReadHalf,
    client:   tokio::net::tcp::OwnedWriteHalf,
    max_read: usize,
    lab:      Option<Arc<LabLogger>>,
    conn_id:  u64,
    emitter:  EmitterHandle,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let max_read = max_read.clamp(1, DEFAULT_MAX_FRAME_LEN);
    shaped_forward(bridge, client, max_read, lab, conn_id, DOWNLINK_LABELS, emitter).await
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

        let server = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let server_addr = server.local_addr().unwrap();
        let server_task = tokio::spawn(async move {
            let (s, _) = server.accept().await.unwrap();
            handle_server_connection(s, bridge_addr, 1200).await.unwrap();
        });

        let client_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let client_addr = client_listener.local_addr().unwrap();
        let client_task = tokio::spawn(async move {
            let (s, _) = client_listener.accept().await.unwrap();
            handle_client_connection(s, server_addr, 1200).await.unwrap();
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
        for len in [1usize, 2, 100, 511, 512] {
            let data: Vec<u8> = (0..len).map(|i| (i % 256) as u8).collect();
            let cell = encode_real_cell(&data, &mut rng);
            assert_eq!(cell.len(), CELL_BYTES, "every cell must be uniform size");
            assert_eq!(decode_real_cell(&cell), &data[..], "len {len} must round-trip");
        }
    }

    #[test]
    fn cell_codec_truncates_oversize_chunk() {
        // The caller splits on CELL_PAYLOAD; a chunk larger than that is clamped.
        let mut rng = StdRng::from_entropy();
        let data = vec![7u8; CELL_PAYLOAD + 50];
        let cell = encode_real_cell(&data, &mut rng);
        assert_eq!(cell.len(), CELL_BYTES);
        assert_eq!(decode_real_cell(&cell).len(), CELL_PAYLOAD);
    }

    /// End-to-end round trip across the steady-state departure clock: a payload
    /// larger than WARMUP_BYTES forces traffic through the unified clock + the
    /// inject queue + EOF drain, and must still arrive byte-for-byte intact.
    #[tokio::test]
    async fn steady_state_round_trip_past_warmup() {
        let payload: Vec<u8> = (0..(WARMUP_BYTES as usize + 32 * 1024))
            .map(|i| (i % 251) as u8)
            .collect();
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

        let server = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let server_addr = server.local_addr().unwrap();
        tokio::spawn(async move {
            let (s, _) = server.accept().await.unwrap();
            handle_server_connection(s, bridge_addr, 1200).await.unwrap();
        });

        let client_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let client_addr = client_listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (s, _) = client_listener.accept().await.unwrap();
            handle_client_connection(s, server_addr, 1200).await.unwrap();
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

    /// WARMUP_BYTES must be > 0 so the Tor handshake's first cells bypass the
    /// departure clock and are not held by its initial inter-cell delay.
    #[test]
    fn warmup_protects_first_burst() {
        assert!(
            WARMUP_BYTES > 0,
            "WARMUP_BYTES=0 would route the TLS Client Hello through the \
             departure clock's initial inter-cell delay, stalling handshake setup"
        );
    }

    /// Warmup must outlast the whole Tor connection setup, not just the TLS
    /// handshake.  A byte budget far larger than a handful of cells ensures the
    /// link handshake (VERSIONS/CERTS/AUTH/NETINFO) and first circuit build
    /// (CREATE/CREATED) all complete before the pool/scheduler engage —
    /// otherwise the per-round-trip latency stalls the bridge connection.
    #[test]
    fn warmup_covers_full_connection_setup() {
        assert!(
            WARMUP_BYTES >= 16 * 1024,
            "WARMUP_BYTES={WARMUP_BYTES} is too small to cover the Tor link \
             handshake and initial circuit setup; shaping would engage \
             mid-handshake and stall the bridge connection"
        );
    }

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
}
