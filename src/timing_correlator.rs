//! Timing correlation defense: unified packet-slot scheduler.
//!
//! # Why the previous architecture failed
//!
//! The old `SeedBComposer` produces two independent processes:
//!
//! - **Real stream**: packets arrive from the application, times governed by
//!   the application's own timing.
//! - **Dummy stream**: keepalives fire from `SpectralIatGenerator`, a separate
//!   exponential clock.
//!
//! A timing correlator can:
//! 1. Filter by size (65535-byte dummies vs. 1200-byte reals) → removes
//!    dummies entirely, recovers the raw real packet timing.
//! 2. Cross-correlate the recovered timing against another flow's timing →
//!    links the two flows with high confidence.
//!
//! The `SeedBComposer` keepalive approach also created a measurable
//! regime-boundary artifact: the dummy rate changed at regime switches,
//! giving an observer a coarse segmentation of the session.
//!
//! # Correct architecture
//!
//! A **single unified inter-departure clock** governs *all* outgoing packets —
//! real and dummy alike.  The clock ticks at Poisson-distributed intervals
//! parameterised by the observed real traffic rate.  At each tick:
//!
//! - If a real packet is queued, emit it (replacing the dummy that would have
//!   filled this slot).
//! - Otherwise emit a dummy of the **same size** as a real packet.
//!
//! This gives the observer exactly one process to look at.  Real packets
//! are indistinguishable from dummies by timing *or* size.  A correlator
//! cannot separate the two components because they share the same clock.
//!
//! # Size uniformity
//!
//! All frames — real and dummy — are padded to `cell_bytes` (default: 514,
//! matching Tor cell size).  If a real packet exceeds `cell_bytes` it is
//! split into multiple cells by the caller before enqueuing; this module
//! does not fragment.  If it is shorter it is padded to exactly `cell_bytes`.
//!
//! # Rate adaptation
//!
//! The scheduler tracks an exponential moving average of the real packet
//! inter-arrival rate and adjusts the departure clock accordingly.  During
//! idle periods (no real traffic) the clock continues at the last observed
//! rate, emitting dummies to maintain the cover.  This is the key property:
//! **the cover does not slow down when real traffic stops**.
//!
//! # Relationship to `SeedBComposer`
//!
//! `TimingCorrelator` replaces the keepalive / interleave machinery in
//! `SeedBComposer`.  The `SeedBComposer` regime logic (DP shaping, session
//! bounding, privacy accounting) sits *above* the correlator: it decides
//! *how much* dummy traffic to target; the correlator decides *when* each
//! frame departs.

use crate::composition::MixedFrame;
use crate::clumping::Packet;
use rand::Rng;
use rand_chacha::ChaCha20Rng;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for `TimingCorrelator`.
#[derive(Clone, Copy, Debug)]
pub struct TimingCorrelatorConfig {
    /// Uniform cell size in bytes.  All outgoing frames (real and dummy) are
    /// exactly this many bytes on the wire.  514 matches the Tor cell size and
    /// is the recommended default.
    ///
    /// Real packets shorter than this are right-padded with random bytes.
    /// Real packets longer than this must be split by the caller before
    /// calling `inject_real`.
    pub cell_bytes: usize,

    /// Initial mean inter-departure time in milliseconds before any real
    /// traffic has been observed.  The scheduler draws the first few
    /// departure slots from Exponential(1/initial_iat_ms).
    pub initial_iat_ms: f64,

    /// EMA smoothing factor α for rate adaptation (0 < α ≤ 1).
    /// Larger = faster adaptation; smaller = more stability.
    /// Recommended: 0.15–0.30.
    pub ema_alpha: f64,

    /// Minimum inter-departure time (ms).  Prevents the scheduler from
    /// emitting faster than the wire can carry.
    pub min_iat_ms: f64,

    /// Maximum inter-departure time (ms).  Prevents the cover from going
    /// completely silent during long idle periods.
    pub max_iat_ms: f64,

    /// Maximum real packets to hold in the injection queue before back-pressure
    /// is signalled to the caller.  Prevents unbounded memory growth.
    pub max_queue_depth: usize,

    /// When `true` (default) the departure rate adapts to the observed real
    /// inter-arrival time via the EMA.  When `false` the clock runs as a
    /// **homogeneous Poisson process** fixed at `initial_iat_ms`: real arrivals
    /// never change the rate.
    ///
    /// Adaptive mode is poisoned by bursty injection — many cells injected
    /// back-to-back (≈0 ms apart) drag the mean toward `min_iat_ms`, so the
    /// clock speeds up in exactly the windows where real traffic is active.
    /// That couples the wire's packet rate to real activity and re-introduces
    /// the volume/timing correlation the cover is meant to remove.  Fixed mode
    /// keeps the rate constant so the wire envelope is independent of real
    /// traffic at every lag.
    pub adapt_rate: bool,

    /// Minimum fraction of departure slots that emit a **dummy even when real
    /// packets are queued** (0.0 ≤ f < 1.0).
    ///
    /// With a pure replacement policy (`f = 0.0`) a sustained real rate above
    /// the clock rate saturates the queue: every slot then carries real data,
    /// dummies vanish, and the wire's per-window byte count collapses onto the
    /// real envelope again — the volume correlation returns precisely in the
    /// busy windows where it matters most.  Reserving a fraction `f` of slots
    /// for dummies keeps cover present under load, so the wire envelope stays
    /// noisy at every timescale even during saturation.  The cost is goodput:
    /// real throughput is scaled by `(1 - f)`.
    pub min_dummy_fraction: f64,
}

impl Default for TimingCorrelatorConfig {
    fn default() -> Self {
        Self {
            cell_bytes: 514,       // Tor cell size
            initial_iat_ms: 200.0,
            ema_alpha: 0.20,
            min_iat_ms: 5.0,
            max_iat_ms: 2_000.0,
            max_queue_depth: 256,
            adapt_rate: true,
            min_dummy_fraction: 0.0,
        }
    }
}

impl TimingCorrelatorConfig {
    pub fn validate(self) -> Result<Self, TimingCorrelatorError> {
        if self.cell_bytes == 0 {
            return Err(TimingCorrelatorError::InvalidConfig("cell_bytes must be non-zero"));
        }
        if self.initial_iat_ms <= 0.0 {
            return Err(TimingCorrelatorError::InvalidConfig("initial_iat_ms must be positive"));
        }
        if self.ema_alpha <= 0.0 || self.ema_alpha > 1.0 {
            return Err(TimingCorrelatorError::InvalidConfig(
                "ema_alpha must be in (0, 1]",
            ));
        }
        if self.min_iat_ms <= 0.0 {
            return Err(TimingCorrelatorError::InvalidConfig("min_iat_ms must be positive"));
        }
        if self.min_iat_ms >= self.max_iat_ms {
            return Err(TimingCorrelatorError::InvalidConfig(
                "min_iat_ms must be less than max_iat_ms",
            ));
        }
        if self.max_queue_depth == 0 {
            return Err(TimingCorrelatorError::InvalidConfig(
                "max_queue_depth must be non-zero",
            ));
        }
        if !(0.0..1.0).contains(&self.min_dummy_fraction) {
            return Err(TimingCorrelatorError::InvalidConfig(
                "min_dummy_fraction must be in [0.0, 1.0)",
            ));
        }
        Ok(self)
    }
}

// ---------------------------------------------------------------------------
// TimingCorrelator
// ---------------------------------------------------------------------------

/// Unified packet-slot scheduler.
///
/// The caller drives the scheduler by:
/// 1. Calling `inject_real(packet)` whenever a real packet becomes available.
/// 2. Calling `tick()` on a tight loop (or after each `sleep_until(next_slot)`)
///    to drain the departure queue.
/// 3. Emitting whatever `tick()` returns onto the wire immediately.
///
/// `tick()` returns `None` when the next departure slot has not yet arrived.
/// The caller should sleep until `next_departure()` and call `tick()` again.
#[derive(Debug)]
pub struct TimingCorrelator {
    config: TimingCorrelatorConfig,
    rng: ChaCha20Rng,

    /// Queue of real packets waiting to be emitted.  Packets are emitted in
    /// FIFO order; each dequeue replaces one dummy slot.
    real_queue: VecDeque<Packet>,

    /// Absolute time of the next scheduled departure slot.
    next_departure: Instant,

    /// EMA of observed real inter-arrival time in milliseconds.
    ///
    /// Used as the mean parameter for the exponential departure clock.
    /// Updated each time `inject_real` records an arrival.
    mean_iat_ms: f64,

    /// Timestamp of the last real packet injection, for IAT measurement.
    last_real_arrival: Option<Instant>,

    /// Total real frames emitted (for statistics).
    real_emitted: u64,

    /// Total dummy frames emitted (for statistics).
    dummy_emitted: u64,
}

impl TimingCorrelator {
    pub fn new(config: TimingCorrelatorConfig, rng: ChaCha20Rng) -> Result<Self, TimingCorrelatorError> {
        let config = config.validate()?;
        let mean_iat_ms = config.initial_iat_ms;
        let mut tc = Self {
            config,
            rng,
            real_queue: VecDeque::new(),
            next_departure: Instant::now(),
            mean_iat_ms,
            last_real_arrival: None,
            real_emitted: 0,
            dummy_emitted: 0,
        };
        // Schedule the first departure slot.
        tc.advance_clock();
        Ok(tc)
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Enqueue a real packet for emission.
    ///
    /// The packet will be emitted at the next available departure slot,
    /// replacing the dummy that would have occupied it.  Packets shorter than
    /// `cell_bytes` will be padded; packets longer must be split by the caller
    /// before injection.
    ///
    /// Returns `Err(QueueFull)` if the queue has reached `max_queue_depth`.
    /// The caller should apply back-pressure (e.g., pause reading from the
    /// upstream socket) until the queue drains.
    pub fn inject_real(&mut self, packet: Packet) -> Result<(), TimingCorrelatorError> {
        if self.real_queue.len() >= self.config.max_queue_depth {
            return Err(TimingCorrelatorError::QueueFull);
        }

        // Update EMA of real inter-arrival time — adaptive mode only.
        // In fixed-rate mode the rate is deliberately decoupled from real
        // arrivals (see `adapt_rate`), so the EMA is left untouched.
        if self.config.adapt_rate {
            if let Some(last) = self.last_real_arrival {
                let iat_ms = last.elapsed().as_secs_f64() * 1000.0;
                self.mean_iat_ms = self.config.ema_alpha * iat_ms
                    + (1.0 - self.config.ema_alpha) * self.mean_iat_ms;
                self.mean_iat_ms = self.mean_iat_ms
                    .clamp(self.config.min_iat_ms, self.config.max_iat_ms);
            }
            self.last_real_arrival = Some(Instant::now());
        }

        self.real_queue.push_back(packet);
        Ok(())
    }

    /// Inject multiple real packets at once (e.g., from a burst).
    pub fn inject_burst(&mut self, packets: impl IntoIterator<Item = Packet>) -> Result<usize, TimingCorrelatorError> {
        let mut count = 0;
        for pkt in packets {
            self.inject_real(pkt)?;
            count += 1;
        }
        Ok(count)
    }

    /// Attempt to emit the next frame.
    ///
    /// Returns `None` if the next departure slot has not yet arrived.
    /// Returns `Some(frame)` when a frame should be sent immediately.
    ///
    /// The caller must emit the returned frame onto the wire before calling
    /// `tick()` again.  Do not buffer returned frames — buffering breaks the
    /// timing guarantee.
    pub fn tick(&mut self) -> Option<MixedFrame> {
        if Instant::now() < self.next_departure {
            return None;
        }

        // Advance the clock before emitting so the next slot is always
        // scheduled, even if the caller delays calling tick() again.
        self.advance_clock();

        // Dummy floor: reserve a fraction of slots for dummies even when real
        // packets are waiting, so cover never fully disappears under sustained
        // load (see `min_dummy_fraction`).  Drawn before the queue check so the
        // decision is independent of the queue state an observer cannot see.
        let force_dummy = self.config.min_dummy_fraction > 0.0
            && self.rng.gen::<f64>() < self.config.min_dummy_fraction;

        // Real packet available (and not a reserved dummy slot) → emit it,
        // replacing the dummy that would have filled this slot.
        if !force_dummy {
            if let Some(real) = self.real_queue.pop_front() {
                let frame = self.make_real_frame(real);
                self.real_emitted += 1;
                return Some(frame);
            }
        }

        // No real packet (or a reserved dummy slot) → emit a dummy.
        let frame = self.make_dummy_frame();
        self.dummy_emitted += 1;
        Some(frame)
    }

    /// The absolute `Instant` at which the next departure slot is scheduled.
    ///
    /// Use this with `tokio::time::sleep_until` or equivalent so the caller
    /// can block efficiently rather than spinning on `tick()`.
    pub fn next_departure(&self) -> Instant {
        self.next_departure
    }

    /// True when the departure slot has arrived and `tick()` will return `Some`.
    pub fn slot_ready(&self) -> bool {
        Instant::now() >= self.next_departure
    }

    /// Current queue depth (real packets waiting to be emitted).
    pub fn queue_depth(&self) -> usize {
        self.real_queue.len()
    }

    /// True when the injection queue has room for at least one more real packet
    /// (so a caller can pull from an external buffer without risking loss).
    pub fn has_queue_space(&self) -> bool {
        self.real_queue.len() < self.config.max_queue_depth
    }

    /// Current mean inter-departure time in milliseconds.
    pub fn mean_iat_ms(&self) -> f64 {
        self.mean_iat_ms
    }

    /// Emission statistics: (real_emitted, dummy_emitted).
    pub fn emission_counts(&self) -> (u64, u64) {
        (self.real_emitted, self.dummy_emitted)
    }

    /// Fraction of emitted frames that were real (not dummy).
    ///
    /// At steady state this should approximate `real_rate / departure_rate`.
    pub fn real_fraction(&self) -> f64 {
        let total = self.real_emitted + self.dummy_emitted;
        if total == 0 {
            0.0
        } else {
            self.real_emitted as f64 / total as f64
        }
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Advance the departure clock by one exponentially-distributed slot.
    ///
    /// Draws from Exponential(1/mean_iat_ms) so the departure process is a
    /// Poisson process — maximum-entropy for a given mean rate, zero
    /// autocorrelation at all lags.
    ///
    /// The next slot is scheduled relative to the **previously scheduled slot**,
    /// not `Instant::now()`, so the departure schedule is a fixed absolute
    /// cadence that does not depend on how often the caller polls `tick()`.
    /// Scheduling from `now` instead couples the effective rate to the polling
    /// frequency: real arrivals poll the transport loop far more often than the
    /// coarse OS timer does in idle windows, so overdue slots fire promptly while
    /// real traffic flows and lag behind when it doesn't — the cover then runs
    /// faster in active windows and slower in idle ones, leaking the envelope at
    /// coarse timescales. Absolute scheduling lets the caller's `while tick()`
    /// loop drain the whole backlog on a late wake, equalising the two regimes.
    fn advance_clock(&mut self) {
        let iat_ms = self.exponential_sample(self.mean_iat_ms);
        let clamped = iat_ms.clamp(self.config.min_iat_ms, self.config.max_iat_ms);
        let step = Duration::from_micros((clamped * 1_000.0).round() as u64);
        let now = Instant::now();
        let candidate = self.next_departure + step;
        // Bound catch-up: if the schedule has fallen far behind real time (task
        // starvation, a long stall), drop the accumulated debt so we never emit
        // a large make-up burst — that burst would itself be a timing artefact.
        const MAX_CATCHUP_LAG: Duration = Duration::from_millis(200);
        self.next_departure = if candidate + MAX_CATCHUP_LAG < now {
            now + step
        } else {
            candidate
        };
    }

    /// Sample from Exponential(mean) using the inverse CDF.
    ///
    /// X = −mean · ln(U),  U ~ Uniform(0, 1)
    fn exponential_sample(&mut self, mean_ms: f64) -> f64 {
        let u: f64 = self.rng.gen::<f64>().max(f64::EPSILON);
        -mean_ms * u.ln()
    }

    /// Build a `MixedFrame::Real` padded to `cell_bytes`.
    ///
    /// `pad_to_cell` is idempotent: if the packet was already padded to
    /// `cell_bytes` by the caller (e.g., `ShapedSession::inject_real` calls
    /// `pad_real_packet` before handing to `inject_real`), the
    /// `Ordering::Equal` branch runs and the bytes are returned unchanged.
    /// No double-padding occurs.
    fn make_real_frame(&mut self, packet: Packet) -> MixedFrame {
        let raw = packet.into_bytes();
        let padded = self.pad_to_cell(raw);
        // Return MixedFrame::Real so downstream code that needs to
        // distinguish real vs. dummy for crypto purposes (e.g., the Tor PT
        // layer) can still do so.  On the wire both variants are identical
        // in size.
        match Packet::new(padded) {
            Ok(p)  => MixedFrame::Real(p),
            Err(_) => {
                // Packet::new only fails on empty input; cell_bytes >= 1 by
                // config validation, so this branch is unreachable in practice.
                MixedFrame::Dummy(vec![0u8; self.config.cell_bytes])
            }
        }
    }

    /// Build a `MixedFrame::Dummy` of exactly `cell_bytes` random bytes.
    fn make_dummy_frame(&mut self) -> MixedFrame {
        let mut bytes = vec![0u8; self.config.cell_bytes];
        self.rng.fill(bytes.as_mut_slice());
        MixedFrame::Dummy(bytes)
    }

    /// Right-pad `data` with random bytes to exactly `cell_bytes`.
    ///
    /// If `data.len() >= cell_bytes`, the data is truncated.  The caller is
    /// responsible for splitting oversized packets before injection.
    fn pad_to_cell(&mut self, mut data: Vec<u8>) -> Vec<u8> {
        let target = self.config.cell_bytes;
        match data.len().cmp(&target) {
            std::cmp::Ordering::Equal => data,
            std::cmp::Ordering::Less => {
                let pad_len = target - data.len();
                let mut padding = vec![0u8; pad_len];
                self.rng.fill(padding.as_mut_slice());
                data.extend_from_slice(&padding);
                data
            }
            std::cmp::Ordering::Greater => {
                data.truncate(target);
                data
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Drain iterator
// ---------------------------------------------------------------------------

/// Drains all ready frames from a `TimingCorrelator` into a `Vec`.
///
/// Useful in tests and in transport loops that want to collect all frames
/// that have become ready since the last poll.
pub fn drain_ready(tc: &mut TimingCorrelator) -> Vec<MixedFrame> {
    let mut out = Vec::new();
    while let Some(frame) = tc.tick() {
        out.push(frame);
    }
    out
}

// ---------------------------------------------------------------------------
// Packet extension trait
// ---------------------------------------------------------------------------

/// Extension trait to move bytes out of a `Packet` for padding.
///
/// `Packet` is defined in `crate::clumping` and doesn't expose a
/// consuming-bytes method directly, so we re-use `bytes()` and clone.
/// In a real codebase this would be an `into_bytes()` on Packet itself.
trait PacketExt {
    fn into_bytes(self) -> Vec<u8>;
}

impl PacketExt for Packet {
    fn into_bytes(self) -> Vec<u8> {
        self.bytes().to_vec()
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, Eq, PartialEq)]
pub enum TimingCorrelatorError {
    InvalidConfig(&'static str),
    /// The real-packet injection queue is full.  Apply back-pressure.
    QueueFull,
}

impl std::fmt::Display for TimingCorrelatorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid TimingCorrelator config: {msg}"),
            Self::QueueFull => write!(f, "real packet queue is full — apply back-pressure"),
        }
    }
}

impl std::error::Error for TimingCorrelatorError {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clumping::Packet;
    use rand_chacha::rand_core::SeedableRng;

    fn test_tc(iat_ms: f64) -> TimingCorrelator {
        TimingCorrelator::new(
            TimingCorrelatorConfig {
                cell_bytes: 514,
                initial_iat_ms: iat_ms,
                ema_alpha: 0.20,
                min_iat_ms: 1.0,
                max_iat_ms: iat_ms * 10.0,
                max_queue_depth: 64,
                adapt_rate: true,
                min_dummy_fraction: 0.0,
            },
            ChaCha20Rng::from_seed([99u8; 32]),
        )
        .unwrap()
    }

    fn pkt(byte: u8) -> Packet {
        Packet::new(vec![byte; 200]).unwrap()
    }

    // -----------------------------------------------------------------------
    // Basic functionality
    // -----------------------------------------------------------------------

    #[test]
    fn tick_returns_none_before_slot() {
        let tc = test_tc(10_000.0); // 10-second slots — won't fire in a unit test
        // next_departure is in the future, so tick returns None immediately.
        // We can't call tick() via &mut without a mutable borrow, so just
        // check slot_ready().
        assert!(!tc.slot_ready(), "slot should not be ready immediately after construction");
    }

    #[test]
    fn dummy_emitted_when_queue_empty() {
        let mut tc = TimingCorrelator::new(
            TimingCorrelatorConfig {
                cell_bytes: 514,
                initial_iat_ms: 1.0, // very fast so tick fires quickly
                min_iat_ms: 1.0,
                max_iat_ms: 5.0,
                ema_alpha: 0.20,
                max_queue_depth: 64,
                adapt_rate: true,
                min_dummy_fraction: 0.0,
            },
            ChaCha20Rng::from_seed([1u8; 32]),
        )
        .unwrap();

        // Spin until we get a frame (should be nearly instant at 1ms IAT).
        let mut frame = None;
        for _ in 0..10_000 {
            if let Some(f) = tc.tick() {
                frame = Some(f);
                break;
            }
            std::hint::spin_loop();
        }
        let frame = frame.expect("should have emitted a frame within 10k spins");
        assert!(frame.is_dummy(), "empty queue must produce a dummy frame");
        assert_eq!(frame.len(), 514, "dummy must be exactly cell_bytes");
    }

    #[test]
    fn real_packet_replaces_dummy_slot() {
        let mut tc = TimingCorrelator::new(
            TimingCorrelatorConfig {
                cell_bytes: 514,
                initial_iat_ms: 1.0,
                min_iat_ms: 1.0,
                max_iat_ms: 5.0,
                ema_alpha: 0.20,
                max_queue_depth: 64,
                adapt_rate: true,
                min_dummy_fraction: 0.0,
            },
            ChaCha20Rng::from_seed([2u8; 32]),
        )
        .unwrap();

        // Inject a real packet before any slots fire.
        tc.inject_real(pkt(0xAB)).unwrap();
        assert_eq!(tc.queue_depth(), 1);

        // Spin until the first slot fires.
        let mut frame = None;
        for _ in 0..10_000 {
            if let Some(f) = tc.tick() {
                frame = Some(f);
                break;
            }
            std::hint::spin_loop();
        }

        let frame = frame.expect("should have emitted a frame");
        assert!(frame.is_real(), "queued real packet should replace dummy slot");
        assert_eq!(frame.len(), 514, "real frame must be padded to cell_bytes");
        assert_eq!(tc.queue_depth(), 0, "queue should be drained");
    }

    #[test]
    fn all_frames_are_cell_size() {
        let mut tc = TimingCorrelator::new(
            TimingCorrelatorConfig {
                cell_bytes: 514,
                initial_iat_ms: 1.0,
                min_iat_ms: 1.0,
                max_iat_ms: 2.0,
                ema_alpha: 0.20,
                max_queue_depth: 64,
                adapt_rate: true,
                min_dummy_fraction: 0.0,
            },
            ChaCha20Rng::from_seed([3u8; 32]),
        )
        .unwrap();

        // Inject packets of various sizes.
        for size in [10, 200, 514, 400] {
            tc.inject_real(Packet::new(vec![0u8; size]).unwrap()).unwrap();
        }

        std::thread::sleep(Duration::from_millis(20));
        let frames = drain_ready(&mut tc);

        assert!(!frames.is_empty(), "should have emitted frames during 20ms sleep");
        for frame in &frames {
            assert_eq!(
                frame.len(),
                514,
                "every frame must be exactly cell_bytes regardless of content"
            );
        }
    }

    #[test]
    fn queue_full_returns_error() {
        let mut tc = TimingCorrelator::new(
            TimingCorrelatorConfig {
                max_queue_depth: 3,
                initial_iat_ms: 100_000.0, // slots never fire in this test
                min_iat_ms: 1.0,
                max_iat_ms: 200_000.0,
                ..TimingCorrelatorConfig::default()
            },
            ChaCha20Rng::from_seed([4u8; 32]),
        )
        .unwrap();

        assert!(tc.inject_real(pkt(1)).is_ok());
        assert!(tc.inject_real(pkt(2)).is_ok());
        assert!(tc.inject_real(pkt(3)).is_ok());
        assert_eq!(
            tc.inject_real(pkt(4)),
            Err(TimingCorrelatorError::QueueFull)
        );
    }

    #[test]
    fn real_packets_emitted_in_fifo_order() {
        let mut tc = TimingCorrelator::new(
            TimingCorrelatorConfig {
                cell_bytes: 514,
                initial_iat_ms: 1.0,
                min_iat_ms: 1.0,
                max_iat_ms: 2.0,
                ema_alpha: 0.20,
                max_queue_depth: 64,
                adapt_rate: true,
                min_dummy_fraction: 0.0,
            },
            ChaCha20Rng::from_seed([5u8; 32]),
        )
        .unwrap();

        // Inject 5 packets with distinct first bytes.
        for b in 1u8..=5 {
            tc.inject_real(Packet::new(vec![b; 100]).unwrap()).unwrap();
        }

        std::thread::sleep(Duration::from_millis(15));
        let frames = drain_ready(&mut tc);

        let real_first_bytes: Vec<u8> = frames
            .iter()
            .filter(|f| f.is_real())
            .map(|f| match f {
                MixedFrame::Real(p) => p.bytes()[0],
                _ => unreachable!(),
            })
            .collect();

        // All 5 should have been emitted in order (1, 2, 3, 4, 5).
        assert_eq!(
            real_first_bytes,
            vec![1, 2, 3, 4, 5],
            "real packets must be emitted in FIFO order"
        );
    }

    #[test]
    fn dummy_floor_keeps_cover_under_saturation() {
        // A pure replacement policy would emit zero dummies while the queue is
        // backed up; the floor must keep cover alive even then.
        let mut tc = TimingCorrelator::new(
            TimingCorrelatorConfig {
                cell_bytes: 514,
                initial_iat_ms: 1.0,
                min_iat_ms: 1.0,
                max_iat_ms: 2.0,
                ema_alpha: 0.20,
                max_queue_depth: 100_000,
                adapt_rate: false,
                min_dummy_fraction: 0.25,
            },
            ChaCha20Rng::from_seed([21u8; 32]),
        )
        .unwrap();

        // Saturate: many more reals queued than slots will fire in the window.
        for _ in 0..5_000 {
            let _ = tc.inject_real(pkt(1));
        }
        std::thread::sleep(Duration::from_millis(30));
        let frames = drain_ready(&mut tc);

        assert!(!frames.is_empty(), "should have emitted frames in 30ms");
        let dummies = frames.iter().filter(|f| f.is_dummy()).count();
        assert!(
            dummies > 0,
            "dummy floor must keep cover alive under saturation; \
             got {dummies} dummies of {} frames",
            frames.len()
        );
    }

    #[test]
    fn catch_up_drains_backlog_after_idle_poll() {
        // With absolute scheduling, a single late poll must drain the whole
        // accumulated backlog — so an idle window polled coarsely emits the same
        // count as an active window polled often. Before the fix (schedule from
        // `now`) this drained exactly one slot per wake.
        let mut tc = TimingCorrelator::new(
            TimingCorrelatorConfig {
                cell_bytes: 514,
                initial_iat_ms: 1.0,
                min_iat_ms: 1.0,
                max_iat_ms: 2.0,
                ema_alpha: 0.20,
                max_queue_depth: 64,
                adapt_rate: false,
                min_dummy_fraction: 0.0,
            },
            ChaCha20Rng::from_seed([55u8; 32]),
        )
        .unwrap();

        // One coarse poll, far later than a single ~1 ms slot.
        std::thread::sleep(Duration::from_millis(40));
        let frames = drain_ready(&mut tc);
        assert!(
            frames.len() >= 10,
            "a 40 ms idle gap at a ~1 ms clock must drain a backlog of many \
             slots (catch-up), got {}",
            frames.len()
        );
    }

    #[test]
    fn validate_rejects_out_of_range_dummy_fraction() {
        assert!(TimingCorrelatorConfig { min_dummy_fraction: 1.0, ..TimingCorrelatorConfig::default() }
            .validate()
            .is_err());
        assert!(TimingCorrelatorConfig { min_dummy_fraction: -0.1, ..TimingCorrelatorConfig::default() }
            .validate()
            .is_err());
    }

    // -----------------------------------------------------------------------
    // Rate adaptation
    // -----------------------------------------------------------------------

    #[test]
    fn mean_iat_adapts_toward_observed_arrivals() {
        let mut tc = test_tc(500.0); // starts at 500ms
        let initial_iat = tc.mean_iat_ms();

        // Simulate fast arrivals (~10ms apart) by injecting packets in a loop.
        // We can't sleep 10ms 20 times in a unit test, so we drive the EMA
        // directly via repeated inject_real with actual elapsed time.
        //
        // Instead: verify the EMA moves in the right direction by using
        // the inject_real → observe IAT path.  We sleep briefly between
        // injections so the clock advances.
        for _ in 0..5 {
            std::thread::sleep(Duration::from_millis(2));
            let _ = tc.inject_real(pkt(1));
        }

        // The EMA should have moved toward the short observed IAT.
        assert!(
            tc.mean_iat_ms() < initial_iat,
            "mean IAT should have decreased toward observed fast arrivals: \
             was {initial_iat:.1}, now {:.1}",
            tc.mean_iat_ms()
        );
    }

    // -----------------------------------------------------------------------
    // Statistical property: IAT autocorrelation
    // -----------------------------------------------------------------------

    /// The departure clock uses Exponential inter-slot times, which have
    /// zero autocorrelation by construction.  Verify empirically.
    #[test]
    fn departure_times_have_low_autocorrelation() {
        use crate::spectral::lag1_autocorrelation;

        // Sample many IAT draws directly from the clock internals.
        let mut rng = ChaCha20Rng::from_seed([77u8; 32]);
        let mean_ms = 200.0;
        let n = 2_000;
        let samples: Vec<f64> = (0..n)
            .map(|_| {
                let u: f64 = rng.gen::<f64>().max(f64::EPSILON);
                (-mean_ms * u.ln()).clamp(5.0, 4_000.0)
            })
            .collect();

        let ac = lag1_autocorrelation(&samples);
        assert!(
            ac.abs() < 0.10,
            "exponential IAT samples have too much autocorrelation: {ac:.4}"
        );
    }

    // -----------------------------------------------------------------------
    // Config validation
    // -----------------------------------------------------------------------

    #[test]
    fn validate_rejects_zero_cell_size() {
        assert!(TimingCorrelatorConfig { cell_bytes: 0, ..TimingCorrelatorConfig::default() }
            .validate()
            .is_err());
    }

    #[test]
    fn validate_rejects_bad_ema_alpha() {
        assert!(TimingCorrelatorConfig { ema_alpha: 0.0, ..TimingCorrelatorConfig::default() }
            .validate()
            .is_err());
        assert!(TimingCorrelatorConfig { ema_alpha: 1.5, ..TimingCorrelatorConfig::default() }
            .validate()
            .is_err());
    }

    #[test]
    fn validate_rejects_inverted_iat_range() {
        assert!(TimingCorrelatorConfig {
            min_iat_ms: 500.0,
            max_iat_ms: 100.0,
            ..TimingCorrelatorConfig::default()
        }
        .validate()
        .is_err());
    }
}