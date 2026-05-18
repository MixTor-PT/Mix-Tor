use crate::clumping::{Burst, Packet};
use crate::seeds::{EphemeralSeed, SeedRole};
use rand::seq::IteratorRandom;
use rand::Rng;
use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};
use std::fmt;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MixedBurst {
    frames: Vec<MixedFrame>,
    ratio_regime: RatioRegime,
    real_packets: usize,
    dummy_packets: usize,
}

impl MixedBurst {
    pub fn frames(&self) -> &[MixedFrame] {
        &self.frames
    }

    pub fn into_frames(self) -> Vec<MixedFrame> {
        self.frames
    }

    pub fn total_packets(&self) -> usize {
        self.frames.len()
    }

    pub fn real_packets(&self) -> usize {
        self.real_packets
    }

    pub fn dummy_packets(&self) -> usize {
        self.dummy_packets
    }

    pub fn ratio_regime(&self) -> RatioRegime {
        self.ratio_regime
    }

    pub fn total_bytes(&self) -> usize {
        self.frames.iter().map(MixedFrame::len).sum()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MixedFrame {
    Real(Packet),
    Dummy(Vec<u8>),
}

impl MixedFrame {
    pub fn len(&self) -> usize {
        match self {
            Self::Real(packet) => packet.len(),
            Self::Dummy(bytes) => bytes.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_real(&self) -> bool {
        matches!(self, Self::Real(_))
    }

    pub fn is_dummy(&self) -> bool {
        matches!(self, Self::Dummy(_))
    }
}

// ---------------------------------------------------------------------------
// Regime / distribution descriptors
// ---------------------------------------------------------------------------

/// The four traffic-cover personalities.
///
/// Each variant carries **absolute** dummy-count statistics (μ, σ) that are
/// independent of how many real packets appear in a burst.  The correlation
/// between real volume and dummy volume is thereby broken by construction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RatioDistributionKind {
    /// Low, steady trickle of cover (~2 dummies/burst on average).
    Sparse,
    /// Moderate, balanced cover (~6 dummies/burst).
    Balanced,
    /// Heavy, sustained cover (~12 dummies/burst).
    HeavyCover,
    /// Bimodal: mostly very quiet, occasionally floods (~1 or ~14).
    Spiky,
}

impl RatioDistributionKind {
    /// Return (μ, σ) for the absolute dummy count distribution.
    ///
    /// These values are intentionally far apart so that regime transitions
    /// produce a clearly visible step-change in observed dummy rate rather
    /// than blending into a near-uniform average.
    pub fn dummy_mu_sigma(self) -> (f64, f64) {
        match self {
            // Spreads are intentionally wide and far apart.  Sparse sits near
            // 0-2, HeavyCover near 18, giving a ~18-unit separation that
            // forces std > 0.15 when measured across regimes.
            Self::Sparse     => (1.0,  0.5),
            Self::Balanced   => (7.0,  1.0),
            Self::HeavyCover => (18.0, 2.0),
            // Spiky is bimodal; these values are for introspection only —
            // dummy_count_absolute handles it directly.
            Self::Spiky      => (10.0, 4.0),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RatioSwitchThreshold {
    pub bursts: u64,
    pub real_bytes: u64,
    pub real_packets: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RatioRegime {
    pub kind: RatioDistributionKind,
    pub switch_after: RatioSwitchThreshold,
    /// Dummy count pre-committed for every burst in this regime.
    ///
    /// Drawn once at regime creation so `dummy_count_absolute` never reads
    /// `real_packets` at all — breaking dummy-real coupling structurally.
    /// Every burst in a Sparse regime emits exactly this many dummies whether
    /// it carries 1 real packet or 20.
    pub fixed_dummy_count: usize,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompositionConfig {
    pub max_dummy_packets_per_burst: usize,

    // --- Regime duration knobs ---
    //
    // Raised substantially from the old defaults (24/256 bursts) so that each
    // regime runs long enough to be clearly observable as a distinct block
    // (~30-90 s at typical burst rates) before switching.
    pub min_switch_bursts: u64,
    pub max_switch_bursts: u64,
    pub min_switch_real_bytes: u64,
    pub max_switch_real_bytes: u64,
    pub min_switch_real_packets: u64,
    pub max_switch_real_packets: u64,

    // --- Keepalive / idle-gap ceiling ---
    //
    // If no burst has been emitted within this window the composer signals
    // that a dummy-only keepalive burst should be injected.  Keeps the
    // "continuous traffic envelope" claim honest by ensuring at most one
    // `max_idle_gap_ms` dead zone between any two bursts.
    pub max_idle_gap_ms: u64,

    /// Minimum packet size used when synthesising keepalive payloads and when
    /// no real-packet length history is available.
    pub keepalive_min_bytes: usize,
    /// Maximum packet size for the same.
    pub keepalive_max_bytes: usize,
}

impl Default for CompositionConfig {
    fn default() -> Self {
        Self {
            // Raised to 32 so HeavyCover (μ=18) and Spiky's flood mode (μ=22)
            // can reach their target means without being clipped, which was the
            // primary cause of the narrow 0.59-0.94 dummy-ratio band.
            max_dummy_packets_per_burst: 32,

            // Long enough that an observer sees ~30-90 s blocks at a typical
            // burst rate of 3-6 bursts/s.  The old default (24–256) produced
            // blocks of only 4-85 s at the high end and as little as 4 s at
            // the low end, causing regimes to blend together.
            min_switch_bursts:       180,
            max_switch_bursts:       800,

            // Real-byte and real-packet trip-wires are kept as secondary
            // circuit-breakers for very bursty sessions; primary switching is
            // expected to be burst-count-driven.
            min_switch_real_bytes:   64  * 1024,
            max_switch_real_bytes:   2   * 1024 * 1024,
            min_switch_real_packets: 256,
            max_switch_real_packets: 8192,

            // Cap inter-burst gap at 2.5 s  (well under the 3 s target).
            max_idle_gap_ms: 2_500,

            keepalive_min_bytes: 64,
            keepalive_max_bytes: 1_200,
        }
    }
}

impl CompositionConfig {
    pub fn validate(self) -> Result<Self, CompositionError> {
        if self.max_dummy_packets_per_burst == 0 {
            return Err(CompositionError::InvalidConfig(
                "maximum dummy packets must be non-zero",
            ));
        }
        if self.min_switch_bursts == 0
            || self.min_switch_real_bytes == 0
            || self.min_switch_real_packets == 0
        {
            return Err(CompositionError::InvalidConfig(
                "ratio switch thresholds must be non-zero",
            ));
        }
        if self.min_switch_bursts > self.max_switch_bursts {
            return Err(CompositionError::InvalidConfig(
                "minimum switch bursts cannot exceed maximum switch bursts",
            ));
        }
        if self.min_switch_real_bytes > self.max_switch_real_bytes {
            return Err(CompositionError::InvalidConfig(
                "minimum switch bytes cannot exceed maximum switch bytes",
            ));
        }
        if self.min_switch_real_packets > self.max_switch_real_packets {
            return Err(CompositionError::InvalidConfig(
                "minimum switch packets cannot exceed maximum switch packets",
            ));
        }
        if self.max_idle_gap_ms == 0 {
            return Err(CompositionError::InvalidConfig(
                "max idle gap must be non-zero",
            ));
        }
        if self.keepalive_min_bytes == 0 || self.keepalive_min_bytes > self.keepalive_max_bytes {
            return Err(CompositionError::InvalidConfig(
                "keepalive byte range is invalid",
            ));
        }
        Ok(self)
    }
}

// ---------------------------------------------------------------------------
// Composer
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct SeedBComposer {
    config: CompositionConfig,
    rng: ChaCha20Rng,
    regime: RatioRegime,
    bursts_in_regime: u64,
    real_bytes_in_regime: u64,
    real_packets_in_regime: u64,
    /// Tracks the last time any burst (real or keepalive) was emitted so the
    /// caller can poll `needs_keepalive` on a ticker.
    last_burst_at: Instant,
}

impl SeedBComposer {
    pub fn new(seed_b: &EphemeralSeed) -> Result<Self, CompositionError> {
        Self::with_config(seed_b, CompositionConfig::default())
    }

    pub fn with_config(
        seed_b: &EphemeralSeed,
        config: CompositionConfig,
    ) -> Result<Self, CompositionError> {
        if seed_b.role() != SeedRole::BurstComposition {
            return Err(CompositionError::WrongSeedRole {
                expected: SeedRole::BurstComposition,
                actual: seed_b.role(),
            });
        }

        let config = config.validate()?;
        let mut rng = ChaCha20Rng::from_seed(seed_b.derive_rng_seed(b"mixtor.composition.v1"));
        let regime = Self::next_regime(config, &mut rng);

        Ok(Self {
            config,
            rng,
            regime,
            bursts_in_regime: 0,
            real_bytes_in_regime: 0,
            real_packets_in_regime: 0,
            last_burst_at: Instant::now(),
        })
    }

    // -----------------------------------------------------------------------
    // Primary API
    // -----------------------------------------------------------------------

    /// Mix a real burst with cover traffic.
    pub fn compose(&mut self, burst: Burst) -> MixedBurst {
        let real_packets = burst.packet_count();
        let real_bytes   = burst.total_bytes();
        let dummy_count  = self.dummy_count_absolute();
        let real_lengths: Vec<usize> = burst.packets().iter().map(Packet::len).collect();

        let frames = self.interleave_preserving_real_order(
            burst.into_packets(),
            dummy_count,
            &real_lengths,
        );

        let mixed = MixedBurst {
            frames,
            ratio_regime: self.regime,
            real_packets,
            dummy_packets: dummy_count,
        };

        self.observe(real_bytes as u64, real_packets as u64);
        self.last_burst_at = Instant::now();
        mixed
    }

    /// Build a dummy-only burst to keep the traffic envelope continuous.
    ///
    /// The returned `MixedBurst` has `real_packets == 0`.  The caller should
    /// emit it whenever `needs_keepalive()` returns `true`.
    ///
    /// Keepalive bursts **do not** advance regime counters (they carry no real
    /// traffic), but they do reset `last_burst_at`.
    pub fn compose_keepalive(&mut self) -> MixedBurst {
        // Use a modest fixed count (1–4 packets) so keepalives themselves
        // don't look like a uniform signal.
        let dummy_count = self.rng.gen_range(1usize..=4);
        let frames = (0..dummy_count)
            .map(|_| {
                let len = self.rng.gen_range(
                    self.config.keepalive_min_bytes..=self.config.keepalive_max_bytes,
                );
                let mut bytes = vec![0u8; len];
                self.rng.fill(bytes.as_mut_slice());
                MixedFrame::Dummy(bytes)
            })
            .collect();

        self.last_burst_at = Instant::now();

        MixedBurst {
            frames,
            ratio_regime: self.regime,
            real_packets: 0,
            dummy_packets: dummy_count,
        }
    }

    /// Returns `true` when the idle gap since the last burst exceeds the
    /// configured ceiling and a keepalive should be injected.
    ///
    /// Intended to be polled on a short-interval timer (e.g. every 500 ms).
    pub fn needs_keepalive(&self) -> bool {
        self.last_burst_at.elapsed()
            >= Duration::from_millis(self.config.max_idle_gap_ms)
    }

    pub fn current_regime(&self) -> RatioRegime {
        self.regime
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn observe(&mut self, real_bytes: u64, real_packets: u64) {
        self.bursts_in_regime        = self.bursts_in_regime.saturating_add(1);
        self.real_bytes_in_regime    = self.real_bytes_in_regime.saturating_add(real_bytes);
        self.real_packets_in_regime  = self.real_packets_in_regime.saturating_add(real_packets);

        if self.bursts_in_regime       >= self.regime.switch_after.bursts
            || self.real_bytes_in_regime   >= self.regime.switch_after.real_bytes
            || self.real_packets_in_regime >= self.regime.switch_after.real_packets
        {
            self.regime = Self::next_regime(self.config, &mut self.rng);
            self.bursts_in_regime       = 0;
            self.real_bytes_in_regime   = 0;
            self.real_packets_in_regime = 0;
        }
    }

    /// Return the dummy count for this burst.
    ///
    /// The value is the regime's `fixed_dummy_count`, which was drawn once
    /// from the regime's Gaussian when the regime was created.  This function
    /// does **not** read `real_packets` and contains no RNG call — coupling
    /// is zero by construction.
    ///
    /// Removing the `.max(1)` floor is intentional: Sparse regimes must be
    /// able to emit 0 dummies on some bursts to pull their observed mean down
    /// to ~1.  The old floor was silently dragging Sparse upward and
    /// compressing the inter-regime spread.
    fn dummy_count_absolute(&self) -> usize {
        self.regime.fixed_dummy_count
    }


    fn dummy_payload_from_lengths(&mut self, real_lengths: &[usize]) -> Vec<u8> {
        let len = real_lengths
            .iter()
            .choose(&mut self.rng)
            .copied()
            .unwrap_or_else(|| {
                self.rng.gen_range(
                    self.config.keepalive_min_bytes..=self.config.keepalive_max_bytes,
                )
            });
        let mut bytes = vec![0u8; len];
        self.rng.fill(bytes.as_mut_slice());
        bytes
    }

    fn interleave_preserving_real_order(
        &mut self,
        real_packets: Vec<Packet>,
        dummy_packets: usize,
        real_lengths: &[usize],
    ) -> Vec<MixedFrame> {
        // Distribute dummy slots uniformly across the gaps between real packets
        // (including before the first and after the last).
        let mut dummy_gaps = vec![0usize; real_packets.len() + 1];
        for _ in 0..dummy_packets {
            let gap = self.rng.gen_range(0..dummy_gaps.len());
            dummy_gaps[gap] += 1;
        }

        let mut frames = Vec::with_capacity(real_packets.len() + dummy_packets);

        for _ in 0..dummy_gaps[0] {
            frames.push(MixedFrame::Dummy(
                self.dummy_payload_from_lengths(real_lengths),
            ));
        }

        for (index, packet) in real_packets.into_iter().enumerate() {
            frames.push(MixedFrame::Real(packet));
            for _ in 0..dummy_gaps[index + 1] {
                frames.push(MixedFrame::Dummy(
                    self.dummy_payload_from_lengths(real_lengths),
                ));
            }
        }

        frames
    }

    fn next_regime(config: CompositionConfig, rng: &mut ChaCha20Rng) -> RatioRegime {
        let kind = match rng.gen_range(0..4) {
            0 => RatioDistributionKind::Sparse,
            1 => RatioDistributionKind::Balanced,
            2 => RatioDistributionKind::HeavyCover,
            _ => RatioDistributionKind::Spiky,
        };

        // Pre-commit the dummy count for the entire regime.  Drawing it here
        // rather than per-burst means every burst in this regime emits the
        // same number of dummies regardless of its real packet count.
        let fixed_dummy_count = {
            let raw = match kind {
                RatioDistributionKind::Spiky => {
                    // Bimodal: 74 % quiet (≈0-1), 26 % flood (≈22).
                    if rng.gen_bool(0.74) {
                        Self::gaussian_rng(rng, 0.5, 0.3)
                    } else {
                        Self::gaussian_rng(rng, 22.0, 3.0)
                    }
                }
                other => {
                    let (mu, sigma) = other.dummy_mu_sigma();
                    Self::gaussian_rng(rng, mu, sigma)
                }
            };
            // No .max(1) — Sparse must be able to emit 0 dummies.
            raw.min(config.max_dummy_packets_per_burst)
        };

        RatioRegime {
            kind,
            fixed_dummy_count,
            switch_after: RatioSwitchThreshold {
                bursts: rng
                    .gen_range(config.min_switch_bursts..=config.max_switch_bursts),
                real_bytes: rng
                    .gen_range(config.min_switch_real_bytes..=config.max_switch_real_bytes),
                real_packets: rng
                    .gen_range(config.min_switch_real_packets..=config.max_switch_real_packets),
            },
        }
    }

    /// Box-Muller normal sample from a bare RNG reference (used at regime
    /// creation time before `self` exists).
    fn gaussian_rng(rng: &mut ChaCha20Rng, mu: f64, sigma: f64) -> usize {
        let u1: f64 = rng.gen_range(f64::EPSILON..1.0);
        let u2: f64 = rng.gen();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let sample = mu + sigma * z;
        (sample.round() as isize).max(0) as usize
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, Eq, PartialEq)]
pub enum CompositionError {
    InvalidConfig(&'static str),
    WrongSeedRole {
        expected: SeedRole,
        actual: SeedRole,
    },
}

impl fmt::Display for CompositionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => {
                write!(formatter, "invalid composition config: {message}")
            }
            Self::WrongSeedRole { expected, actual } => {
                write!(
                    formatter,
                    "wrong seed role: expected {expected:?}, got {actual:?}"
                )
            }
        }
    }
}

impl std::error::Error for CompositionError {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clumping::{CycleThresholds, FlushReason};
    use crate::seeds::SessionSeeds;
    use std::time::Duration;

    fn packet(byte: u8, len: usize) -> Packet {
        Packet::new(vec![byte; len]).expect("packet should be valid")
    }

    fn burst(packets: Vec<Packet>) -> Burst {
        Burst::from_parts_for_tests(
            packets,
            CycleThresholds {
                max_packets: 4,
                timeout: Duration::from_millis(20),
            },
            FlushReason::Manual,
        )
    }

    /// A config whose regime changes on every single burst so the full
    /// switching machinery is exercised quickly in unit tests.
    fn fast_switch_config() -> CompositionConfig {
        CompositionConfig {
            max_dummy_packets_per_burst: 32,
            min_switch_bursts: 1,
            max_switch_bursts: 2,
            min_switch_real_bytes: 1,
            max_switch_real_bytes: 2,
            min_switch_real_packets: 1,
            max_switch_real_packets: 2,
            max_idle_gap_ms: 2_500,
            keepalive_min_bytes: 64,
            keepalive_max_bytes: 1_200,
        }
    }

    // ------------------------------------------------------------------
    // Existing / ported tests
    // ------------------------------------------------------------------

    #[test]
    fn rejects_seed_a() {
        let seeds = SessionSeeds::generate().expect("seed generation should succeed");
        let error =
            SeedBComposer::new(seeds.timing_and_pool()).expect_err("Seed A should not compose");

        assert_eq!(
            error,
            CompositionError::WrongSeedRole {
                expected: SeedRole::BurstComposition,
                actual: SeedRole::TimingAndPool,
            }
        );
    }

    #[test]
    fn adds_dummy_packets_and_preserves_real_packets() {
        let seeds = SessionSeeds::generate().expect("seed generation should succeed");
        let mut composer =
            SeedBComposer::new(seeds.burst_composition()).expect("composer should initialize");
        let mixed = composer.compose(burst(vec![packet(1, 40), packet(2, 80)]));

        assert_eq!(mixed.real_packets(), 2);
        // Sparse regime may legitimately emit 0 dummies; just check the
        // accounting identity holds.
        assert_eq!(
            mixed.total_packets(),
            mixed.real_packets() + mixed.dummy_packets()
        );
        assert_eq!(
            mixed.frames().iter().filter(|f| f.is_real()).count(),
            2
        );
        assert_eq!(
            mixed.frames().iter().filter(|f| f.is_dummy()).count(),
            mixed.dummy_packets()
        );
    }

    #[test]
    fn dummy_sizes_match_real_packet_sizes() {
        let seeds = SessionSeeds::generate().expect("seed generation should succeed");
        let mut composer =
            SeedBComposer::new(seeds.burst_composition()).expect("composer should initialize");
        let mixed = composer.compose(burst(vec![packet(1, 41), packet(2, 97)]));

        for frame in mixed.frames().iter().filter(|f| f.is_dummy()) {
            assert!(frame.len() == 41 || frame.len() == 97);
        }
    }

    #[test]
    fn real_packet_order_is_preserved() {
        let seeds = SessionSeeds::generate().expect("seed generation should succeed");
        let mut composer =
            SeedBComposer::new(seeds.burst_composition()).expect("composer should initialize");
        let mixed =
            composer.compose(burst(vec![packet(1, 41), packet(2, 97), packet(3, 123)]));

        let real_first_bytes: Vec<u8> = mixed
            .frames()
            .iter()
            .filter_map(|frame| match frame {
                MixedFrame::Real(p) => Some(p.bytes()[0]),
                MixedFrame::Dummy(_) => None,
            })
            .collect();

        assert_eq!(real_first_bytes, vec![1, 2, 3]);
    }

    #[test]
    fn ratio_regime_switches_from_traffic_counters() {
        let seeds = SessionSeeds::generate().expect("seed generation should succeed");
        let mut composer =
            SeedBComposer::with_config(seeds.burst_composition(), fast_switch_config())
                .expect("composer should initialize");

        for _ in 0..8 {
            let _ = composer.compose(burst(vec![packet(9, 16)]));
        }

        assert_eq!(composer.bursts_in_regime, 0);
    }

    #[test]
    fn emits_all_ratio_distribution_families_over_many_regimes() {
        let seeds = SessionSeeds::generate().expect("seed generation should succeed");
        let mut composer =
            SeedBComposer::with_config(seeds.burst_composition(), fast_switch_config())
                .expect("composer should initialize");
        let mut kinds = Vec::new();

        for _ in 0..256 {
            kinds.push(composer.current_regime().kind);
            let _ = composer.compose(burst(vec![packet(7, 32)]));
        }

        kinds.sort_by_key(|k| *k as u8);
        kinds.dedup();
        assert_eq!(kinds.len(), 4);
    }

    // ------------------------------------------------------------------
    // New tests for the three fixes
    // ------------------------------------------------------------------

    /// Fix 1: dummy count must be completely independent of real packet count.
    ///
    /// Because `fixed_dummy_count` is pre-committed at regime creation,
    /// every burst in the same regime emits the exact same dummy count
    /// regardless of real burst size.  The mean difference between a
    /// 1-real-packet batch and an 8-real-packet batch must be < 1.0.
    #[test]
    fn dummy_count_is_independent_of_real_packet_count() {
        // Use a slow-switching config so both batches run inside the same
        // regime, making the independence check as tight as possible.
        let config = CompositionConfig {
            min_switch_bursts: 500,
            max_switch_bursts: 500,
            min_switch_real_bytes: u64::MAX / 2,
            max_switch_real_bytes: u64::MAX / 2,
            min_switch_real_packets: u64::MAX / 2,
            max_switch_real_packets: u64::MAX / 2,
            ..CompositionConfig::default()
        };
        let seeds = SessionSeeds::generate().expect("seed generation should succeed");
        let mut composer =
            SeedBComposer::with_config(seeds.burst_composition(), config)
                .expect("composer should initialize");

        let n = 100usize;
        let small_dummies: Vec<usize> = (0..n)
            .map(|_| composer.compose(burst(vec![packet(1, 64)])).dummy_packets())
            .collect();

        let large_dummies: Vec<usize> = (0..n)
            .map(|_| {
                composer
                    .compose(burst(vec![
                        packet(1, 64),
                        packet(2, 64),
                        packet(3, 64),
                        packet(4, 64),
                        packet(5, 64),
                        packet(6, 64),
                        packet(7, 64),
                        packet(8, 64),
                    ]))
                    .dummy_packets()
            })
            .collect();

        let mean_small = small_dummies.iter().sum::<usize>() as f64 / n as f64;
        let mean_large = large_dummies.iter().sum::<usize>() as f64 / n as f64;

        // Within a single regime every burst gets the same fixed_dummy_count,
        // so both means should be identical (or differ only due to a regime
        // boundary crossing during the test).
        assert!(
            (mean_large - mean_small).abs() < 1.0,
            "dummy counts appear coupled to real packet count: \
             mean_small={mean_small:.2}, mean_large={mean_large:.2}"
        );
    }

    /// Fix 2: each regime block must have a characteristic dummy rate that
    /// differs visibly from neighbouring blocks.
    ///
    /// We collect (regime_kind, dummy_count) pairs over many bursts with a
    /// medium switch cadence, group by kind, and assert that the per-kind
    /// averages span at least 4 dummies from lowest to highest.
    #[test]
    fn regime_blocks_have_distinct_dummy_rates() {
        // Moderate switch speed: each regime lasts ~6-12 bursts.
        let config = CompositionConfig {
            min_switch_bursts: 6,
            max_switch_bursts: 12,
            ..fast_switch_config()
        };
        let seeds = SessionSeeds::generate().expect("seed generation should succeed");
        let mut composer =
            SeedBComposer::with_config(seeds.burst_composition(), config)
                .expect("composer should initialize");

        let mut per_kind: std::collections::HashMap<u8, Vec<usize>> = Default::default();

        for _ in 0..1_000 {
            let kind = composer.current_regime().kind as u8;
            let mixed = composer.compose(burst(vec![packet(1, 64)]));
            per_kind.entry(kind).or_default().push(mixed.dummy_packets());
        }

        let means: Vec<f64> = per_kind
            .values()
            .map(|v| v.iter().sum::<usize>() as f64 / v.len() as f64)
            .collect();

        let min_mean = means.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_mean = means.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        assert!(
            max_mean - min_mean >= 10.0,
            "regime dummy-rate spread is too narrow: min={min_mean:.2}, max={max_mean:.2}"
        );
    }

    /// Fix 3: `needs_keepalive` must fire after the configured idle gap.
    #[test]
    fn needs_keepalive_fires_after_idle_gap() {
        let config = CompositionConfig {
            // Very short gap so the test doesn't have to sleep.
            max_idle_gap_ms: 1,
            ..fast_switch_config()
        };
        let seeds = SessionSeeds::generate().expect("seed generation should succeed");
        let mut composer =
            SeedBComposer::with_config(seeds.burst_composition(), config)
                .expect("composer should initialize");

        // Should not need keepalive immediately.
        assert!(!composer.needs_keepalive());

        std::thread::sleep(Duration::from_millis(5));
        assert!(composer.needs_keepalive());

        // After emitting a keepalive the flag should clear.
        let _ = composer.compose_keepalive();
        assert!(!composer.needs_keepalive());
    }

    /// Keepalive bursts must have zero real packets and at least one dummy.
    #[test]
    fn keepalive_burst_is_dummy_only() {
        let seeds = SessionSeeds::generate().expect("seed generation should succeed");
        let mut composer =
            SeedBComposer::new(seeds.burst_composition()).expect("composer should initialize");
        let ka = composer.compose_keepalive();

        assert_eq!(ka.real_packets(), 0);
        assert!(ka.dummy_packets() >= 1);
        assert!(ka.frames().iter().all(|f| f.is_dummy()));
    }

    /// Config validation must reject a zero idle gap.
    #[test]
    fn rejects_zero_idle_gap() {
        let config = CompositionConfig {
            max_idle_gap_ms: 0,
            ..CompositionConfig::default()
        };
        assert!(config.validate().is_err());
    }
}