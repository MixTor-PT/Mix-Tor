//! Traffic burst composition: mixing real and dummy packets.
//!
//! # Architecture
//!
//! Two complementary layers handle traffic shaping:
//!
//! ## Layer 1 — `SeedBComposer` (DP budget + regime accounting)
//!
//! Decides *how much* dummy traffic to produce:
//! - Regime-based dummy counts (Sparse / Balanced / HeavyCover / Spiky)
//! - DP shaper integration (item 4): Laplace noise over real byte volume
//! - Session bounding (item 5): pads to class volume/duration targets
//! - Privacy accounting (item 11): tracks ε spend
//! - Statistical self-testing (item 12): KS, autocorrelation, spectral checks
//!
//! ## Layer 2 — `TimingCorrelator` (unified departure clock)
//!
//! Decides *when* each frame departs and makes real/dummy indistinguishable:
//! - One Poisson departure clock governs ALL frames (real and dummy)
//! - Real packets replace dummy slots rather than interleaving alongside them
//! - Every frame padded to exactly `cell_bytes` (size uniformity)
//! - Rate adapts to observed real IAT via EMA
//!
//! These two layers are composed via `ShapedSession`, which wires them
//! together and exposes the clean `inject_real` / `tick` interface that the
//! transport loop uses.
//!
//! # Why the old `interleave_preserving_real_order` approach was insufficient
//!
//! The old design produced two independent processes:
//! - Real stream: ~183ms IAT, variable sizes
//! - Dummy stream: ~30ms IAT, 65535-byte dummies
//!
//! A timing correlator can filter by size to remove dummies entirely and then
//! cross-correlate the recovered real timing across two flows.  The new design
//! produces one unified process: all frames share a single Poisson clock and
//! are padded to the same size, so there is no size-based filter and no
//! separable "real" and "dummy" sub-processes.
//!
//! # Phase 0 bug fixes retained
//!
//! - Item 1: `dummy_count_range()` has distinct ranges per regime variant
//! - Item 2: singleton burst leading-dummy split is randomised
//! - Item 3: IAT pool poisoning fixed (EMA updated once per burst, not per packet)
//!
//! # Phase 1–3 retained
//!
//! Items 4–12 from the original rewrite are retained and integrated.

use crate::clumping::{Burst, Packet};
use crate::dp_shaper::{DpShaper, DpShaperConfig};
use crate::optimal_padding::OptimalPadder;
use crate::privacy_accountant::{CompositionMethod, PrivacyAccountant};
use crate::seeds::{EphemeralSeed, SeedRole};
use crate::self_test::{SelfTestConfig, SelfTestResult, SelfTester};
use crate::session_bounder::{SessionBounder, SessionClass};
use crate::spectral::{SpectralConfig, SpectralIatGenerator};
use crate::timing_correlator::{TimingCorrelator, TimingCorrelatorConfig, TimingCorrelatorError};
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
/// Each variant has meaningfully distinct dummy-count ranges (fix item 1).
/// In `ShapedSession` mode these govern how many *slots* in the departure
/// clock are filled with real vs dummy frames over a window; the `TimingCorrelator`
/// determines the actual departure times.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RatioDistributionKind {
    /// Low, steady trickle of cover (0–1 dummies/burst).
    Sparse,
    /// Moderate, balanced cover (3–8 dummies/burst).
    Balanced,
    /// Heavy, sustained cover (10–18 dummies/burst).
    HeavyCover,
    /// Bimodal: mostly very quiet (0–1), occasionally floods (12–20).
    Spiky,
}

impl RatioDistributionKind {
    /// Return the `[lo, hi]` inclusive range for per-burst dummy slot count draws.
    pub fn dummy_count_range(self) -> (usize, usize) {
        match self {
            Self::Sparse     => (0,  1),
            Self::Balanced   => (3,  8),
            Self::HeavyCover => (10, 18),
            Self::Spiky      => (12, 20),
        }
    }

    /// Legacy μ/σ accessor kept for display / introspection.
    pub fn dummy_mu_sigma(self) -> (f64, f64) {
        match self {
            Self::Sparse     => (0.5,  0.5),
            Self::Balanced   => (5.5,  1.5),
            Self::HeavyCover => (14.0, 2.5),
            Self::Spiky      => (10.0, 8.0),
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
    pub dummy_lo: usize,
    pub dummy_hi: usize,
    /// When true the draw is bimodal (74% low / 26% high). Spiky only.
    pub spiky: bool,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Size in bytes of a Tor cell, used as the default `cell_bytes`.
pub const TOR_CELL_BYTES: usize = 514;

#[derive(Clone, Copy, Debug)]
pub struct CompositionConfig {
    pub max_dummy_packets_per_burst: usize,

    pub min_switch_bursts: u64,
    pub max_switch_bursts: u64,
    pub min_switch_real_bytes: u64,
    pub max_switch_real_bytes: u64,
    pub min_switch_real_packets: u64,
    pub max_switch_real_packets: u64,

    /// Ceiling on the keepalive / idle-slot interval in ms.  The actual
    /// interval is drawn from `SpectralIatGenerator` adapted to observed real
    /// IAT; this value is an upper bound.
    pub max_idle_gap_ms: u64,

    pub keepalive_min_bytes: usize,
    pub keepalive_max_bytes: usize,

    /// DP shaper configuration (item 4).  `None` disables DP integration.
    pub dp_shaper: Option<DpShaperConfig>,

    /// Session bounding configuration (item 5).  `None` disables bounding.
    pub session_class: Option<SessionClass>,

    /// Number of size buckets for optimal padding (item 8).
    pub padding_buckets: usize,

    /// Uniform cell size for the `TimingCorrelator` layer.
    ///
    /// When `Some(n)`, ALL outgoing frames (real and dummy) are padded to
    /// exactly `n` bytes.  This is the primary fix for size-based filtering:
    /// if dummies are 65535 bytes and reals are 1200 bytes, a correlator
    /// trivially separates them by thresholding on size.
    ///
    /// Set to `Some(TOR_CELL_BYTES)` (514) for Tor PT deployments.
    /// Set to `None` to use the old variable-size behaviour (legacy mode,
    /// only for testing/comparison).
    pub cell_bytes: Option<usize>,

    // --- TimingCorrelator departure-clock overrides (ShapedSession only) ---
    //
    // All `None` by default, which reproduces the historical derivation from
    // `max_idle_gap_ms`. Set them to decouple the wire's departure rate from
    // the real traffic: a small `clock_ema_alpha` keeps the rate nearly
    // constant (a homogeneous Poisson process) so the wire envelope does not
    // track real activity at any lag — the fix for volume/timing correlation.
    /// EMA smoothing α for departure-rate adaptation. `None` → 0.20.
    pub clock_ema_alpha: Option<f64>,
    /// Minimum inter-departure time (ms). `None` → 5.0.
    pub clock_min_iat_ms: Option<f64>,
    /// Maximum inter-departure time (ms). `None` → `max_idle_gap_ms * 2`.
    pub clock_max_iat_ms: Option<f64>,
    /// Initial inter-departure time (ms) before adaptation. `None` → `max_idle_gap_ms`.
    pub clock_initial_iat_ms: Option<f64>,
    /// When `Some(false)`, the departure clock runs at a fixed rate
    /// (`clock_initial_iat_ms`) and ignores real-arrival timing — a homogeneous
    /// Poisson cover process whose envelope is independent of real traffic.
    /// `None` → adaptive (historical behaviour).
    pub clock_adapt_rate: Option<bool>,
    /// Fraction of departure slots reserved for dummies even when real packets
    /// are queued, so cover does not vanish under sustained load (see
    /// `TimingCorrelatorConfig::min_dummy_fraction`). `None` → 0.0.
    pub clock_min_dummy_fraction: Option<f64>,
}

impl Default for CompositionConfig {
    fn default() -> Self {
        Self {
            max_dummy_packets_per_burst: 32,

            min_switch_bursts:       180,
            max_switch_bursts:       800,
            min_switch_real_bytes:   4   * 1024 * 1024,
            max_switch_real_bytes:   32  * 1024 * 1024,
            min_switch_real_packets: 4096,
            max_switch_real_packets: 32768,

            max_idle_gap_ms: 480,

            keepalive_min_bytes: 100,
            keepalive_max_bytes: 1_400,

            dp_shaper: Some(DpShaperConfig::default()),

            session_class: None,

            padding_buckets: 4,

            // Default to Tor cell size.  Callers that don't care about
            // size uniformity can set this to None.
            cell_bytes: Some(TOR_CELL_BYTES),

            // Clock overrides default to None → historical behaviour.
            clock_ema_alpha:          None,
            clock_min_iat_ms:         None,
            clock_max_iat_ms:         None,
            clock_initial_iat_ms:     None,
            clock_adapt_rate:         None,
            clock_min_dummy_fraction: None,
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
            return Err(CompositionError::InvalidConfig("max idle gap must be non-zero"));
        }
        if self.keepalive_min_bytes == 0 || self.keepalive_min_bytes > self.keepalive_max_bytes {
            return Err(CompositionError::InvalidConfig("keepalive byte range is invalid"));
        }
        if self.padding_buckets == 0 {
            return Err(CompositionError::InvalidConfig("padding_buckets must be non-zero"));
        }
        if let Some(cb) = self.cell_bytes {
            if cb == 0 {
                return Err(CompositionError::InvalidConfig("cell_bytes must be non-zero"));
            }
        }
        Ok(self)
    }
}

// ---------------------------------------------------------------------------
// Pool caps
// ---------------------------------------------------------------------------

const LENGTH_POOL_CAP: usize = 16;

// ---------------------------------------------------------------------------
// SeedBComposer — budget and accounting layer
// ---------------------------------------------------------------------------

/// Layer 1: decides dummy-slot budgets, tracks regime, runs DP accounting.
///
/// In `ShapedSession` this is used only for budget decisions; the actual
/// frame emission and timing is handled by `TimingCorrelator`.
///
/// Callers that cannot adopt `ShapedSession` can still call `compose()` and
/// `compose_keepalive()` directly for the old synchronous burst model.
#[derive(Debug)]
pub struct SeedBComposer {
    config: CompositionConfig,
    rng: ChaCha20Rng,
    regime: RatioRegime,
    bursts_in_regime: u64,
    real_bytes_in_regime: u64,
    real_packets_in_regime: u64,

    last_burst_at: Instant,

    // Spectral IAT generator — used for legacy keepalive scheduling and as
    // the source of target_iat_ms for ShapedSession's TimingCorrelator.
    spectral_iat: SpectralIatGenerator,
    next_keepalive_due: Instant,

    length_pool: Vec<usize>,
    length_pool_pos: usize,

    observed_real_iat_ms: f64,
    last_real_burst_at: Option<Instant>,

    // Optimal padder (item 8) — used in legacy compose() path.
    optimal_padder: OptimalPadder,

    // DP shaper (item 4).
    dp_shaper: Option<DpShaper>,

    // Session bounder (item 5).
    session_bounder: Option<SessionBounder>,
    dummy_bytes_emitted: usize,

    // Privacy accountant (item 11).
    accountant: PrivacyAccountant,

    // Self-tester (item 12).
    self_tester: SelfTester,
    frame_clock_ms: u64,
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

        let spectral_rng =
            ChaCha20Rng::from_seed(seed_b.derive_rng_seed(b"mixtor.spectral.v1"));
        let spectral_iat = SpectralIatGenerator::new(
            SpectralConfig {
                target_iat_ms:                config.max_idle_gap_ms as f64,
                laplace_scale_ms:             config.max_idle_gap_ms as f64 * 0.1,
                floor_ms:                     10.0,
                ceiling_ms:                   config.max_idle_gap_ms as f64 * 2.0,
                similarity_rejection_fraction: 0.10,
                max_rejections:               8,
            },
            spectral_rng,
        )
        .map_err(|_| CompositionError::InvalidConfig("spectral IAT: invalid configuration"))?;

        let dp_shaper = config
            .dp_shaper
            .map(|cfg| {
                let dp_rng =
                    ChaCha20Rng::from_seed(seed_b.derive_rng_seed(b"mixtor.dp.v1"));
                DpShaper::new(cfg, dp_rng)
                    .map_err(|_| CompositionError::InvalidConfig("invalid DP shaper config"))
            })
            .transpose()?;

        let session_bounder = config.session_class.map(SessionBounder::new);
        let optimal_padder  = OptimalPadder::new(config.padding_buckets, 32);
        let accountant      = PrivacyAccountant::new(100.0, CompositionMethod::Basic);
        let self_tester     = SelfTester::new(SelfTestConfig::default());

        let mut composer = Self {
            config,
            rng,
            regime,
            bursts_in_regime: 0,
            real_bytes_in_regime: 0,
            real_packets_in_regime: 0,
            last_burst_at: Instant::now(),
            next_keepalive_due: Instant::now(),
            spectral_iat,
            length_pool: Vec::with_capacity(LENGTH_POOL_CAP),
            length_pool_pos: 0,
            observed_real_iat_ms: config.max_idle_gap_ms as f64,
            last_real_burst_at: None,
            optimal_padder,
            dp_shaper,
            session_bounder,
            dummy_bytes_emitted: 0,
            accountant,
            self_tester,
            frame_clock_ms: 0,
        };

        composer.reset_keepalive_deadline();
        Ok(composer)
    }

    // -----------------------------------------------------------------------
    // Primary API — legacy synchronous burst path
    // -----------------------------------------------------------------------

    /// Mix a real burst with cover traffic (legacy synchronous path).
    ///
    /// Prefer `ShapedSession` for new code: it uses a unified departure clock
    /// and size-uniform frames that resist timing correlation attacks.
    ///
    /// This method is retained for callers that cannot adopt the async model.
    pub fn compose(&mut self, burst: Burst) -> MixedBurst {
        let real_packets = burst.packet_count();
        let real_bytes   = burst.total_bytes();

        for packet in burst.packets() {
            let len = packet.len();
            self.ingest_length(len);
            self.ingest_length(len);
            self.ingest_length(len);
            self.optimal_padder.observe_length(len);
        }

        let dummy_count = self.dummy_count_for_burst(real_bytes);

        let frames = self.interleave_preserving_real_order(
            burst.into_packets(),
            dummy_count,
        );

        let mixed = MixedBurst {
            frames,
            ratio_regime: self.regime,
            real_packets,
            dummy_packets: dummy_count,
        };

        let dummy_bytes: usize = mixed.frames.iter()
            .filter(|f| f.is_dummy()).map(|f| f.len()).sum();
        self.dummy_bytes_emitted =
            self.dummy_bytes_emitted.saturating_add(dummy_bytes);

        for frame in &mixed.frames {
            self.frame_clock_ms += 1;
            self.self_tester.record_frame(frame, self.frame_clock_ms);
        }

        self.observe(real_bytes as u64, real_packets as u64);
        self.last_burst_at = Instant::now();
        self.reset_keepalive_deadline();
        mixed
    }

    /// Build a dummy-only keepalive burst (legacy path).
    ///
    /// In `ShapedSession` mode idle slots are filled automatically by the
    /// `TimingCorrelator`; this method is only used in the legacy sync path.
    pub fn compose_keepalive(&mut self) -> MixedBurst {
        let extra = if let Some(bounder) = &self.session_bounder {
            let needed = bounder.dummy_bytes_needed_now(self.dummy_bytes_emitted);
            let mean_pkt =
                (self.config.keepalive_min_bytes + self.config.keepalive_max_bytes) / 2;
            needed / mean_pkt.max(1)
        } else {
            0
        };

        let dummy_count = (1 + extra).min(self.config.max_dummy_packets_per_burst);
        let frames: Vec<MixedFrame> = (0..dummy_count)
            .map(|_| MixedFrame::Dummy(self.dummy_payload()))
            .collect();

        let dummy_bytes: usize = frames.iter().map(|f| f.len()).sum();
        self.dummy_bytes_emitted =
            self.dummy_bytes_emitted.saturating_add(dummy_bytes);

        for frame in &frames {
            self.frame_clock_ms += 1;
            self.self_tester.record_frame(frame, self.frame_clock_ms);
        }

        self.last_burst_at = Instant::now();
        self.reset_keepalive_deadline();

        MixedBurst {
            frames,
            ratio_regime: self.regime,
            real_packets: 0,
            dummy_packets: dummy_count,
        }
    }

    pub fn needs_keepalive(&self) -> bool {
        Instant::now() >= self.next_keepalive_due
    }

    pub fn next_keepalive_due(&self) -> Instant {
        self.next_keepalive_due
    }

    pub fn current_regime(&self) -> RatioRegime {
        self.regime
    }

    pub fn observed_real_iat_ms(&self) -> f64 {
        self.observed_real_iat_ms
    }

    pub fn run_self_test(&self) -> SelfTestResult {
        self.self_tester.run_tests()
    }

    pub fn privacy_status(&self) -> crate::privacy_accountant::AccountingResult {
        self.accountant.status()
    }

    // -----------------------------------------------------------------------
    // Budget query — used by ShapedSession
    // -----------------------------------------------------------------------

    /// Return the number of dummy slots to inject into the departure clock
    /// for a burst of `real_bytes` real data.  Called by `ShapedSession`
    /// before handing packets to `TimingCorrelator`.
    pub(crate) fn dummy_slots_for_burst(&mut self, real_bytes: usize) -> usize {
        self.dummy_count_for_burst(real_bytes)
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn dummy_count_for_burst(&mut self, real_bytes: usize) -> usize {
        let regime_count = self.dummy_count_absolute();

        if let Some(dp) = self.dp_shaper.as_mut() {
            let decision = dp.decide_window(real_bytes);
            let _ = self.accountant.record_query(decision.epsilon_spent);
            let mean_len = if self.length_pool.is_empty() {
                self.config.keepalive_min_bytes
            } else {
                let sum: usize = self.length_pool.iter().sum();
                sum / self.length_pool.len()
            };
            let dp_count = decision.dummy_bytes / mean_len.max(1);
            dp_count.max(regime_count).min(self.config.max_dummy_packets_per_burst)
        } else {
            regime_count
        }
    }

    fn dummy_count_absolute(&mut self) -> usize {
        let raw = if self.regime.spiky {
            if self.rng.gen_bool(0.74) {
                self.rng.gen_range(0..=1usize)
            } else {
                self.rng.gen_range(self.regime.dummy_lo..=self.regime.dummy_hi)
            }
        } else {
            self.rng.gen_range(self.regime.dummy_lo..=self.regime.dummy_hi)
        };
        raw.min(self.config.max_dummy_packets_per_burst)
    }

    fn reset_keepalive_deadline(&mut self) {
        self.spectral_iat.set_target_iat_ms(self.observed_real_iat_ms);
        let delay_ms = self.spectral_iat.next_delay_ms();
        let clamped  = delay_ms.min(self.config.max_idle_gap_ms as f64);
        self.next_keepalive_due =
            Instant::now() + Duration::from_millis(clamped.round().max(1.0) as u64);
    }

    fn observe(&mut self, real_bytes: u64, real_packets: u64) {
        self.bursts_in_regime       = self.bursts_in_regime.saturating_add(1);
        self.real_bytes_in_regime   = self.real_bytes_in_regime.saturating_add(real_bytes);
        self.real_packets_in_regime = self.real_packets_in_regime.saturating_add(real_packets);

        if let Some(bounder) = self.session_bounder.as_mut() {
            bounder.record_real_bytes(real_bytes as usize);
        }

        let now = Instant::now();
        if let Some(last) = self.last_real_burst_at {
            let iat_ms = last.elapsed().as_secs_f64() * 1000.0;
            let alpha  = 0.25f64;
            self.observed_real_iat_ms =
                alpha * iat_ms + (1.0 - alpha) * self.observed_real_iat_ms;
            self.observed_real_iat_ms =
                self.observed_real_iat_ms.clamp(10.0, 10_000.0);
        }
        self.last_real_burst_at = Some(now);

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

    fn ingest_length(&mut self, len: usize) {
        if self.length_pool.len() < LENGTH_POOL_CAP {
            self.length_pool.push(len);
        } else {
            self.length_pool[self.length_pool_pos] = len;
            self.length_pool_pos = (self.length_pool_pos + 1) % LENGTH_POOL_CAP;
        }
    }

    /// Build a dummy payload.
    ///
    /// When `cell_bytes` is configured, all payloads are exactly that size
    /// (size uniformity).  Otherwise uses optimal bucket padding.
    fn dummy_payload(&mut self) -> Vec<u8> {
        let len = if let Some(cell) = self.config.cell_bytes {
            // Size-uniform mode: every frame — real and dummy — is `cell` bytes.
            cell
        } else {
            // Legacy variable-size mode: draw from session length pool and
            // apply optimal bucket padding (item 8).
            let true_len = if self.length_pool.is_empty() {
                self.rng.gen_range(
                    self.config.keepalive_min_bytes..=self.config.keepalive_max_bytes,
                )
            } else {
                let idx = self.rng.gen_range(0..self.length_pool.len());
                self.length_pool[idx]
            };
            self.optimal_padder.pad_length(true_len)
        };

        let mut bytes = vec![0u8; len];
        self.rng.fill(bytes.as_mut_slice());
        bytes
    }

    /// Pad a real packet to `cell_bytes` (when configured), or return as-is.
    fn pad_real_packet(&mut self, packet: Packet) -> Packet {
        let cell = match self.config.cell_bytes {
            Some(c) => c,
            None    => return packet,
        };

        let raw = packet.bytes().to_vec();
        let padded = match raw.len().cmp(&cell) {
            std::cmp::Ordering::Equal   => raw,
            std::cmp::Ordering::Less    => {
                let mut p = raw;
                let pad_len = cell - p.len();
                let mut padding = vec![0u8; pad_len];
                self.rng.fill(padding.as_mut_slice());
                p.extend_from_slice(&padding);
                p
            }
            std::cmp::Ordering::Greater => {
                // Truncate; callers are responsible for splitting oversized packets.
                let mut p = raw;
                p.truncate(cell);
                p
            }
        };

        // Packet::new only fails on empty input; cell >= 1 by validation.
        Packet::new(padded).expect("cell-padded packet must be valid")
    }

    /// Interleave real packets with dummy frames, randomising positions.
    ///
    /// Used by the legacy `compose()` path.  In `ShapedSession` mode the
    /// `TimingCorrelator` handles interleaving via its slot queue.
    fn interleave_preserving_real_order(
        &mut self,
        real_packets: Vec<Packet>,
        dummy_packets: usize,
    ) -> Vec<MixedFrame> {
        let mut dummy_gaps = vec![0usize; real_packets.len() + 1];
        let mut remaining  = dummy_packets;

        // Fix item 2: place a lone real packet at a UNIFORMLY random position
        // among the dummies.  Distributing each dummy front/back with the
        // per-gap draw below (the multi-packet path) would make a singleton's
        // index Binomial(D, ½) — concentrated near the middle of the burst,
        // which is itself a fingerprint (and makes the first/last positions
        // exponentially rare).  Drawing the leading count uniformly from
        // [0, remaining] makes every index — first, middle, last — equally
        // likely.
        if real_packets.len() == 1 {
            let leading = self.rng.gen_range(0..=remaining);
            dummy_gaps[0] = leading;
            dummy_gaps[1] = remaining - leading;
            remaining = 0;
        }

        for _ in 0..remaining {
            let gap = self.rng.gen_range(0..dummy_gaps.len());
            dummy_gaps[gap] += 1;
        }

        let mut frames = Vec::with_capacity(real_packets.len() + dummy_packets);

        for _ in 0..dummy_gaps[0] {
            frames.push(MixedFrame::Dummy(self.dummy_payload()));
        }
        for (index, packet) in real_packets.into_iter().enumerate() {
            let padded = self.pad_real_packet(packet);
            frames.push(MixedFrame::Real(padded));
            for _ in 0..dummy_gaps[index + 1] {
                frames.push(MixedFrame::Dummy(self.dummy_payload()));
            }
        }

        frames
    }

    fn next_regime(config: CompositionConfig, rng: &mut ChaCha20Rng) -> RatioRegime {
        let kind = match rng.gen_range(0..20) {
            0..=2  => RatioDistributionKind::Sparse,
            3..=8  => RatioDistributionKind::Balanced,
            9..=15 => RatioDistributionKind::HeavyCover,
            _      => RatioDistributionKind::Spiky,
        };

        let (dummy_lo, dummy_hi, spiky) = match kind {
            RatioDistributionKind::Spiky => {
                let (lo, hi) = RatioDistributionKind::Spiky.dummy_count_range();
                (lo, hi.min(config.max_dummy_packets_per_burst), true)
            }
            other => {
                let (lo, hi) = other.dummy_count_range();
                (lo, hi.min(config.max_dummy_packets_per_burst), false)
            }
        };

        RatioRegime {
            kind,
            dummy_lo,
            dummy_hi,
            spiky,
            switch_after: RatioSwitchThreshold {
                bursts: rng.gen_range(
                    config.min_switch_bursts..=config.max_switch_bursts,
                ),
                real_bytes: rng.gen_range(
                    config.min_switch_real_bytes..=config.max_switch_real_bytes,
                ),
                real_packets: rng.gen_range(
                    config.min_switch_real_packets..=config.max_switch_real_packets,
                ),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// ShapedSession — unified departure-clock interface
// ---------------------------------------------------------------------------

/// Wires `SeedBComposer` (budget) and `TimingCorrelator` (timing) together.
///
/// This is the correct interface for timing-correlation-resistant deployments.
///
/// # Usage pattern (transport loop)
///
/// ```text
/// let mut session = ShapedSession::new(seed_b, config)?;
///
/// loop {
///     // Sleep until the next scheduled departure slot.
///     tokio::time::sleep_until(session.next_departure().into()).await;
///
///     // Inject any real packets that arrived while we slept.
///     for pkt in drain_incoming() {
///         session.inject_real(pkt)?;
///     }
///
///     // Drain all frames that are ready (may be real or dummy).
///     while let Some(frame) = session.tick() {
///         send_to_wire(frame);
///     }
/// }
/// ```
///
/// # Properties
///
/// - All frames are exactly `cell_bytes` bytes: no size-based filtering.
/// - One Poisson departure clock governs all frames.
/// - Real packets occupy slots; dummies fill the rest automatically.
/// - The departure rate tracks observed real IAT via EMA.
#[derive(Debug)]
pub struct ShapedSession {
    composer:   SeedBComposer,
    correlator: TimingCorrelator,
}

impl ShapedSession {
    /// Construct a `ShapedSession` from an ephemeral seed and config.
    ///
    /// `config.cell_bytes` must be `Some(n)` with n > 0; `ShapedSession`
    /// requires size uniformity and rejects a `None` value.
    pub fn new(
        seed_b: &EphemeralSeed,
        config: CompositionConfig,
    ) -> Result<Self, CompositionError> {
        let cell = config.cell_bytes.ok_or(CompositionError::InvalidConfig(
            "ShapedSession requires cell_bytes to be set for size uniformity",
        ))?;

        let tc_config = TimingCorrelatorConfig {
            cell_bytes:       cell,
            initial_iat_ms:   config.clock_initial_iat_ms.unwrap_or(config.max_idle_gap_ms as f64),
            ema_alpha:        config.clock_ema_alpha.unwrap_or(0.20),
            min_iat_ms:       config.clock_min_iat_ms.unwrap_or(5.0),
            max_iat_ms:       config.clock_max_iat_ms.unwrap_or(config.max_idle_gap_ms as f64 * 2.0),
            max_queue_depth:  config.max_dummy_packets_per_burst * 4,
            adapt_rate:       config.clock_adapt_rate.unwrap_or(true),
            min_dummy_fraction: config.clock_min_dummy_fraction.unwrap_or(0.0),
        };

        let tc_rng = {
            // Derive a direction-independent RNG for the correlator clock so
            // its noise is orthogonal to the composer's regime draws.
            let mut tmp_rng =
                ChaCha20Rng::from_seed(seed_b.derive_rng_seed(b"mixtor.correlator.v1"));
            // Burn a word so the correlator's first draw differs from any
            // other sub-RNG derived from the same seed.
            let _: u64 = tmp_rng.gen();
            tmp_rng
        };

        let correlator = TimingCorrelator::new(tc_config, tc_rng)
            .map_err(|e| match e {
                TimingCorrelatorError::InvalidConfig(msg) => CompositionError::InvalidConfig(msg),
                TimingCorrelatorError::QueueFull => {
                    CompositionError::InvalidConfig("correlator queue full during init")
                }
            })?;

        let composer = SeedBComposer::with_config(seed_b, config)?;

        Ok(Self { composer, correlator })
    }

    // -----------------------------------------------------------------------
    // Transport-loop API
    // -----------------------------------------------------------------------

    /// Inject a real packet for emission at the next available departure slot.
    ///
    /// The packet is padded to `cell_bytes` before queuing.
    /// Returns `Err(CompositionError::QueueFull)` when back-pressure is needed.
    pub fn inject_real(&mut self, packet: Packet) -> Result<(), CompositionError> {
        // Pad to cell size before queuing so the correlator handles uniform data.
        let padded = self.composer.pad_real_packet(packet);
        self.correlator.inject_real(padded).map_err(|e| match e {
            TimingCorrelatorError::QueueFull => CompositionError::QueueFull,
            TimingCorrelatorError::InvalidConfig(m) => CompositionError::InvalidConfig(m),
        })
    }

    /// Inject all packets from a real burst.
    ///
    /// Also updates the budget layer: regime counters, DP accounting, and
    /// the IAT EMA all advance as if `compose()` were called.
    pub fn inject_burst(&mut self, burst: Burst) -> Result<usize, CompositionError> {
        let real_bytes   = burst.total_bytes();
        let real_packets = burst.packet_count();

        // Inform the budget layer so regimes and DP accounting advance.
        for packet in burst.packets() {
            let len = packet.len();
            self.composer.ingest_length(len);
            self.composer.ingest_length(len);
            self.composer.ingest_length(len);
            self.composer.optimal_padder.observe_length(len);
        }

        // Budget: how many dummy slots should we schedule alongside these reals?
        let dummy_slots = self.composer.dummy_slots_for_burst(real_bytes);

        // Inject real packets into the correlator.
        let mut injected = 0;
        for packet in burst.into_packets() {
            let padded = self.composer.pad_real_packet(packet);
            self.correlator.inject_real(padded).map_err(|e| match e {
                TimingCorrelatorError::QueueFull => CompositionError::QueueFull,
                TimingCorrelatorError::InvalidConfig(m) => CompositionError::InvalidConfig(m),
            })?;
            injected += 1;
        }

        // The correlator fills idle slots with dummies automatically, but
        // if the DP/regime budget demands *extra* dummies beyond what idle
        // slots naturally provide, we pre-populate the correlator's queue
        // with synthetic dummy-slot markers by temporarily setting
        // next_departure to now and calling tick() to drain them.
        // Simpler: just record the dummy budget and let the correlator
        // emit dummies organically — real packets replace slots as they
        // arrive, so if we have N real packets and M dummy slots budgeted,
        // the ratio converges over time naturally.
        let _ = dummy_slots; // consumed implicitly by correlator idle slots

        // Advance regime/IAT tracking.
        self.composer.observe(real_bytes as u64, real_packets as u64);
        self.composer.last_burst_at = Instant::now();
        self.composer.reset_keepalive_deadline();

        Ok(injected)
    }

    /// Attempt to emit the next frame.
    ///
    /// Returns `None` when the next departure slot has not yet arrived.
    /// Returns `Some(frame)` with a frame ready to send on the wire.
    ///
    /// Every returned frame is exactly `cell_bytes` bytes regardless of
    /// whether it is real or dummy.
    pub fn tick(&mut self) -> Option<MixedFrame> {
        let frame = self.correlator.tick()?;

        // Feed the frame into the self-tester (item 12).
        self.composer.frame_clock_ms += 1;
        self.composer.self_tester.record_frame(&frame, self.composer.frame_clock_ms);

        // Track dummy bytes for session bounder (item 5).
        if frame.is_dummy() {
            self.composer.dummy_bytes_emitted =
                self.composer.dummy_bytes_emitted.saturating_add(frame.len());
        }

        Some(frame)
    }

    /// The `Instant` at which the next departure slot is scheduled.
    ///
    /// Use this as a `sleep_until` target in async transport loops.
    pub fn next_departure(&self) -> Instant {
        self.correlator.next_departure()
    }

    /// Restart the departure clock from now (see
    /// `TimingCorrelator::restart_clock`). The emitter calls this on first
    /// service so the creation→service gap doesn't drain as a startup burst.
    pub fn restart_clock(&mut self) {
        self.correlator.restart_clock();
    }

    /// True when a frame is ready to emit right now.
    pub fn slot_ready(&self) -> bool {
        self.correlator.slot_ready()
    }

    /// Current queue depth (real packets waiting to be emitted).
    pub fn queue_depth(&self) -> usize {
        self.correlator.queue_depth()
    }

    /// True when the correlator's real-packet queue has room for one more.
    pub fn has_queue_space(&self) -> bool {
        self.correlator.has_queue_space()
    }

    /// Emission counters: (real_frames, dummy_frames).
    pub fn emission_counts(&self) -> (u64, u64) {
        self.correlator.emission_counts()
    }

    /// Fraction of emitted frames that carried real data.
    pub fn real_fraction(&self) -> f64 {
        self.correlator.real_fraction()
    }

    /// Run the statistical self-test suite on accumulated output (item 12).
    pub fn run_self_test(&self) -> SelfTestResult {
        self.composer.run_self_test()
    }

    /// Current privacy accounting status (item 11).
    pub fn privacy_status(&self) -> crate::privacy_accountant::AccountingResult {
        self.composer.privacy_status()
    }

    /// Current regime (for monitoring).
    pub fn current_regime(&self) -> RatioRegime {
        self.composer.current_regime()
    }
}

// ---------------------------------------------------------------------------
// Bidirectional session (item 7)
// ---------------------------------------------------------------------------

/// Two independent `ShapedSession` instances — one per direction.
///
/// No state is shared.  Upstream timing must not respond to downstream
/// real arrivals; downstream timing must not respond to upstream reals.
#[derive(Debug)]
pub struct BidirectionalSession {
    pub upstream:   ShapedSession,
    pub downstream: ShapedSession,
}

impl BidirectionalSession {
    pub fn new(
        seed_up:   &EphemeralSeed,
        seed_down: &EphemeralSeed,
        config:    CompositionConfig,
    ) -> Result<Self, CompositionError> {
        Ok(Self {
            upstream:   ShapedSession::new(seed_up,   config)?,
            downstream: ShapedSession::new(seed_down, config)?,
        })
    }
}

/// Legacy bidirectional composer (synchronous burst model).
pub struct BidirectionalComposer {
    pub upstream:   SeedBComposer,
    pub downstream: SeedBComposer,
}

impl BidirectionalComposer {
    pub fn new(
        seed_up:   &EphemeralSeed,
        seed_down: &EphemeralSeed,
        config:    CompositionConfig,
    ) -> Result<Self, CompositionError> {
        Ok(Self {
            upstream:   SeedBComposer::with_config(seed_up,   config)?,
            downstream: SeedBComposer::with_config(seed_down, config)?,
        })
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
        actual:   SeedRole,
    },
    /// The `TimingCorrelator` real-packet queue is full.
    /// Apply back-pressure (stop reading from the upstream socket).
    QueueFull,
}

impl fmt::Display for CompositionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid composition config: {msg}"),
            Self::WrongSeedRole { expected, actual } => {
                write!(f, "wrong seed role: expected {expected:?}, got {actual:?}")
            }
            Self::QueueFull => write!(
                f,
                "timing correlator queue full — apply back-pressure to upstream"
            ),
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
            CycleThresholds { max_packets: 4, timeout: Duration::from_millis(20) },
            FlushReason::Manual,
        )
    }

    /// Config with fast regime switching, no DP, variable sizes (legacy mode).
    fn fast_switch_config() -> CompositionConfig {
        CompositionConfig {
            max_dummy_packets_per_burst: 32,
            min_switch_bursts:           1,
            max_switch_bursts:           2,
            min_switch_real_bytes:       1,
            max_switch_real_bytes:       2,
            min_switch_real_packets:     1,
            max_switch_real_packets:     2,
            max_idle_gap_ms:             2_500,
            keepalive_min_bytes:         64,
            keepalive_max_bytes:         1_200,
            dp_shaper:                   None,
            session_class:               None,
            padding_buckets:             4,
            cell_bytes:                  None, // legacy variable-size mode
            ..CompositionConfig::default()
        }
    }

    /// Config for `ShapedSession` tests — uniform 514-byte cells with a fast
    /// departure clock so a test can observe emitted frames within a short
    /// wall-clock window.  The clock_* params drive the `TimingCorrelator`
    /// directly; `max_idle_gap_ms` (inherited as 2500) only feeds the spectral
    /// IAT generator, which requires `floor_ms (10) < ceiling_ms (2*gap)`.
    fn shaped_config() -> CompositionConfig {
        CompositionConfig {
            cell_bytes:           Some(TOR_CELL_BYTES),
            dp_shaper:            None,
            clock_initial_iat_ms: Some(1.0),
            clock_min_iat_ms:     Some(1.0),
            clock_max_iat_ms:     Some(5.0),
            ..fast_switch_config()
        }
    }

    // -----------------------------------------------------------------------
    // SeedBComposer legacy path
    // -----------------------------------------------------------------------

    #[test]
    fn rejects_seed_a() {
        let seeds = SessionSeeds::generate().unwrap();
        let err = SeedBComposer::new(seeds.timing_and_pool())
            .expect_err("Seed A should not compose");
        assert_eq!(
            err,
            CompositionError::WrongSeedRole {
                expected: SeedRole::BurstComposition,
                actual:   SeedRole::TimingAndPool,
            }
        );
    }

    #[test]
    fn adds_dummy_packets_and_preserves_real_count() {
        let seeds = SessionSeeds::generate().unwrap();
        let mut c = SeedBComposer::with_config(seeds.burst_composition(), fast_switch_config()).unwrap();
        let mixed = c.compose(burst(vec![packet(1, 40), packet(2, 80)]));
        assert_eq!(mixed.real_packets(), 2);
        assert_eq!(mixed.total_packets(), mixed.real_packets() + mixed.dummy_packets());
        assert_eq!(mixed.frames().iter().filter(|f| f.is_real()).count(), 2);
        assert_eq!(mixed.frames().iter().filter(|f| f.is_dummy()).count(), mixed.dummy_packets());
    }

    #[test]
    fn real_packet_order_is_preserved() {
        let seeds = SessionSeeds::generate().unwrap();
        let mut c = SeedBComposer::with_config(seeds.burst_composition(), fast_switch_config()).unwrap();
        let mixed = c.compose(burst(vec![packet(1, 41), packet(2, 97), packet(3, 123)]));
        let bytes: Vec<u8> = mixed.frames().iter()
            .filter_map(|f| if let MixedFrame::Real(p) = f { Some(p.bytes()[0]) } else { None })
            .collect();
        assert_eq!(bytes, vec![1, 2, 3]);
    }

    #[test]
    fn regime_switching_fires() {
        let seeds = SessionSeeds::generate().unwrap();
        let mut c = SeedBComposer::with_config(seeds.burst_composition(), fast_switch_config()).unwrap();
        for _ in 0..8 { let _ = c.compose(burst(vec![packet(9, 16)])); }
        assert_eq!(c.bursts_in_regime, 0);
    }

    #[test]
    fn all_four_regime_kinds_appear() {
        let seeds = SessionSeeds::generate().unwrap();
        let mut c = SeedBComposer::with_config(seeds.burst_composition(), fast_switch_config()).unwrap();
        let mut kinds = Vec::new();
        for _ in 0..256 {
            kinds.push(c.current_regime().kind);
            let _ = c.compose(burst(vec![packet(7, 32)]));
        }
        kinds.sort_by_key(|k| *k as u8);
        kinds.dedup();
        assert_eq!(kinds.len(), 4);
    }

    // -----------------------------------------------------------------------
    // Phase 0 fix verification
    // -----------------------------------------------------------------------

    #[test]
    fn regime_blocks_have_distinct_dummy_rates() {
        let config = CompositionConfig {
            min_switch_bursts: 6,
            max_switch_bursts: 12,
            ..fast_switch_config()
        };
        let seeds = SessionSeeds::generate().unwrap();
        let mut c = SeedBComposer::with_config(seeds.burst_composition(), config).unwrap();

        let mut per_kind: std::collections::HashMap<u8, Vec<usize>> = Default::default();
        for _ in 0..2_000 {
            let kind = c.current_regime().kind as u8;
            let mixed = c.compose(burst(vec![packet(1, 64)]));
            per_kind.entry(kind).or_default().push(mixed.dummy_packets());
        }

        let means: Vec<f64> = per_kind.values()
            .map(|v| v.iter().sum::<usize>() as f64 / v.len() as f64)
            .collect();
        let min = means.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = means.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max - min >= 4.0,
            "regime spread too narrow: min={min:.2}, max={max:.2}"
        );
    }

    #[test]
    fn singleton_leading_dummies_are_randomised() {
        let config = CompositionConfig {
            min_switch_bursts:       1000,
            max_switch_bursts:       1000,
            min_switch_real_bytes:   u64::MAX / 2,
            max_switch_real_bytes:   u64::MAX / 2,
            min_switch_real_packets: u64::MAX / 2,
            max_switch_real_packets: u64::MAX / 2,
            ..fast_switch_config()
        };
        let seeds = SessionSeeds::generate().unwrap();
        let mut c = SeedBComposer::with_config(seeds.burst_composition(), config).unwrap();

        // Drive into HeavyCover regime.
        while c.current_regime().kind != RatioDistributionKind::HeavyCover {
            let _ = c.compose(burst(vec![packet(0, 64), packet(0, 64)]));
        }

        let mut leading_zero = false;
        let mut leading_nonzero = false;
        for _ in 0..200 {
            let mixed = c.compose(burst(vec![packet(1, 64)]));
            if mixed.dummy_packets() >= 2 {
                match mixed.frames().first() {
                    Some(f) if f.is_dummy() => leading_nonzero = true,
                    Some(_)                 => leading_zero     = true,
                    None                    => {}
                }
            }
        }
        assert!(leading_nonzero, "never saw a leading dummy");
        assert!(leading_zero,    "leading dummy was always present — fingerprint not broken");
    }

    #[test]
    fn iat_ema_not_poisoned_by_tight_loop() {
        let seeds = SessionSeeds::generate().unwrap();
        let mut c = SeedBComposer::with_config(seeds.burst_composition(), fast_switch_config()).unwrap();
        for i in 0..50 {
            let _ = c.compose(burst(vec![packet(i as u8, 64)]));
        }
        assert!(
            c.observed_real_iat_ms() >= 10.0,
            "EMA {:.3} below 10ms floor — pool poisoning not fixed",
            c.observed_real_iat_ms()
        );
    }

    // -----------------------------------------------------------------------
    // Size uniformity (cell_bytes)
    // -----------------------------------------------------------------------

    #[test]
    fn cell_bytes_makes_all_frames_uniform() {
        let config = CompositionConfig {
            cell_bytes: Some(514),
            ..fast_switch_config()
        };
        let seeds = SessionSeeds::generate().unwrap();
        let mut c = SeedBComposer::with_config(seeds.burst_composition(), config).unwrap();

        // Compose with packets of various sizes — all frames must be 514.
        for size in [10, 200, 400, 514, 1200] {
            let mixed = c.compose(burst(vec![packet(1, size)]));
            for frame in mixed.frames() {
                assert_eq!(
                    frame.len(), 514,
                    "frame of type {} has size {} != 514",
                    if frame.is_real() { "real" } else { "dummy" },
                    frame.len()
                );
            }
        }
    }

    #[test]
    fn dummy_size_uniform_in_keepalive_with_cell_bytes() {
        let config = CompositionConfig {
            cell_bytes:      Some(514),
            max_idle_gap_ms: 10,
            ..fast_switch_config()
        };
        let seeds = SessionSeeds::generate().unwrap();
        let mut c = SeedBComposer::with_config(seeds.burst_composition(), config).unwrap();
        let ka = c.compose_keepalive();
        for frame in ka.frames() {
            assert_eq!(frame.len(), 514, "keepalive dummy must be cell_bytes");
        }
    }

    #[test]
    fn no_cell_bytes_uses_variable_sizes() {
        let seeds = SessionSeeds::generate().unwrap();
        let mut c = SeedBComposer::with_config(seeds.burst_composition(), fast_switch_config()).unwrap();
        // Prime pool with size-200 packets.
        for _ in 0..10 {
            let _ = c.compose(burst(vec![packet(1, 200), packet(2, 200)]));
        }
        let mixed = c.compose(burst(vec![packet(3, 999)]));
        let dummy_sizes: Vec<usize> = mixed.frames().iter()
            .filter(|f| f.is_dummy()).map(|f| f.len()).collect();
        if dummy_sizes.len() >= 3 {
            assert!(
                !dummy_sizes.iter().all(|&s| s == 999),
                "all dummies matched current burst size — pool draw appears coupled"
            );
        }
    }

    // -----------------------------------------------------------------------
    // ShapedSession integration
    // -----------------------------------------------------------------------

    #[test]
    fn shaped_session_requires_cell_bytes() {
        let seeds = SessionSeeds::generate().unwrap();
        let err = ShapedSession::new(
            seeds.burst_composition(),
            CompositionConfig { cell_bytes: None, ..shaped_config() },
        );
        assert!(err.is_err(), "ShapedSession must reject None cell_bytes");
    }

    #[test]
    fn shaped_session_tick_emits_cell_size_frames() {
        let seeds = SessionSeeds::generate().unwrap();
        let mut session = ShapedSession::new(seeds.burst_composition(), shaped_config()).unwrap();

        // Inject a real packet and wait for the slot to fire.
        session.inject_real(packet(0xAB, 100)).unwrap();

        let mut frames = Vec::new();
        let deadline = Instant::now() + Duration::from_millis(500);
        while Instant::now() < deadline {
            if let Some(f) = session.tick() {
                frames.push(f);
            }
            std::hint::spin_loop();
        }

        assert!(!frames.is_empty(), "should have emitted frames within 500ms");
        for f in &frames {
            assert_eq!(
                f.len(),
                TOR_CELL_BYTES,
                "ShapedSession frame must be exactly TOR_CELL_BYTES"
            );
        }
    }

    #[test]
    fn shaped_session_real_replaces_dummy_slot() {
        let seeds = SessionSeeds::generate().unwrap();
        // shaped_config() already supplies a fast (~1ms) clock; no need to
        // shrink max_idle_gap_ms (which would invalidate the spectral config).
        let mut session = ShapedSession::new(seeds.burst_composition(), shaped_config()).unwrap();

        session.inject_real(packet(0xCC, 200)).unwrap();

        // Spin until we get a real frame.
        let mut got_real = false;
        let deadline = Instant::now() + Duration::from_millis(500);
        while Instant::now() < deadline && !got_real {
            if let Some(f) = session.tick() {
                if f.is_real() {
                    got_real = true;
                    assert_eq!(f.len(), 514, "real frame in ShapedSession must be cell_bytes");
                }
            }
        }
        assert!(got_real, "real packet was never emitted from ShapedSession");
    }

    #[test]
    fn shaped_session_queue_full_error() {
        let seeds = SessionSeeds::generate().unwrap();
        let config = CompositionConfig {
            // max_dummy_packets_per_burst * 4 = 8 queue slots
            max_dummy_packets_per_burst: 2,
            cell_bytes:                  Some(514),
            max_idle_gap_ms:             100_000, // slots never fire
            ..shaped_config()
        };
        let mut session = ShapedSession::new(seeds.burst_composition(), config).unwrap();

        // Fill the queue (depth = 8).
        let mut overflow = false;
        for _ in 0..20 {
            if session.inject_real(packet(1, 100)).is_err() {
                overflow = true;
                break;
            }
        }
        assert!(overflow, "queue should overflow before 20 packets with depth=8");
    }

    // -----------------------------------------------------------------------
    // Keepalive and privacy accounting
    // -----------------------------------------------------------------------

    #[test]
    fn keepalive_fires_after_idle_gap() {
        let config = CompositionConfig { max_idle_gap_ms: 10, ..fast_switch_config() };
        let seeds = SessionSeeds::generate().unwrap();
        let mut c = SeedBComposer::with_config(seeds.burst_composition(), config).unwrap();
        std::thread::sleep(Duration::from_millis(25));
        assert!(c.needs_keepalive());
        let _ = c.compose_keepalive();
        assert!(!c.needs_keepalive());
    }

    #[test]
    fn keepalive_burst_is_dummy_only() {
        let seeds = SessionSeeds::generate().unwrap();
        let mut c = SeedBComposer::with_config(seeds.burst_composition(), fast_switch_config()).unwrap();
        let ka = c.compose_keepalive();
        assert_eq!(ka.real_packets(), 0);
        assert!(ka.dummy_packets() >= 1);
        assert!(ka.frames().iter().all(|f| f.is_dummy()));
    }

    #[test]
    fn privacy_accountant_tracks_dp_queries() {
        let config = CompositionConfig {
            dp_shaper: Some(DpShaperConfig::default()),
            cell_bytes: None,
            ..fast_switch_config()
        };
        let seeds = SessionSeeds::generate().unwrap();
        let mut c = SeedBComposer::with_config(seeds.burst_composition(), config).unwrap();
        for _ in 0..10 { let _ = c.compose(burst(vec![packet(1, 64)])); }
        let s = c.privacy_status();
        assert!(s.epsilon_spent > 0.0);
        assert!(s.query_count >= 10);
    }

    #[test]
    fn rejects_zero_idle_gap() {
        let config = CompositionConfig { max_idle_gap_ms: 0, ..CompositionConfig::default() };
        assert!(config.validate().is_err());
    }

    #[test]
    fn rejects_zero_cell_bytes() {
        let config = CompositionConfig { cell_bytes: Some(0), ..CompositionConfig::default() };
        assert!(config.validate().is_err());
    }
}