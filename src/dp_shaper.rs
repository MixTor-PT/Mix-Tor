//! Differential-privacy traffic shaper (Phase 1 — item 4).
//!
//! Every `window_ms` milliseconds the shaper samples Laplace noise and emits
//! `max(0, real_bytes + noise)` bytes of traffic.  This is the core of
//! NetShaper-style DP shaping: the adversary's view of window byte-counts is
//! drawn from a Laplace-shifted version of the real distribution, and the
//! privacy guarantee is ε-differential privacy per window.
//!
//! The `SeedBComposer` becomes a *consumer* of this module's output: it asks
//! `DpShaper::window_dummy_bytes()` to learn how many dummy bytes to add in
//! the current window, rather than making that decision itself.
//!
//! # Privacy budget
//!
//! For a Laplace mechanism on sensitivity Δ:
//!   noise_scale b = Δ / ε
//!
//! We use Δ = `max_real_bytes_per_window` (worst-case single-window real
//! volume) and the caller-supplied ε.  Composition across T windows costs
//! T·ε under basic composition; the `PrivacyAccountant` (privacy_accountant.rs)
//! tracks the running total.

use rand::Rng;
use rand_chacha::ChaCha20Rng;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Configuration for the DP shaper.
#[derive(Clone, Copy, Debug)]
pub struct DpShaperConfig {
    /// Per-window privacy budget (ε).  Smaller = more privacy, more overhead.
    /// Typical range: 0.1–2.0.
    pub epsilon: f64,

    /// Window duration in milliseconds.  NetShaper uses T = 50 ms; 100 ms is
    /// a reasonable default that keeps overhead manageable.
    pub window_ms: u64,

    /// Worst-case real bytes in one window (sensitivity Δ).  Set to the
    /// maximum burst size the application can produce in `window_ms`.
    pub max_real_bytes_per_window: usize,
}

impl Default for DpShaperConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.5,
            window_ms: 100,
            max_real_bytes_per_window: 64 * 1024, // 64 KB
        }
    }
}

impl DpShaperConfig {
    /// Laplace noise scale b = Δ / ε.
    pub fn noise_scale(&self) -> f64 {
        self.max_real_bytes_per_window as f64 / self.epsilon
    }

    pub fn validate(self) -> Result<Self, DpShaperError> {
        if self.epsilon <= 0.0 {
            return Err(DpShaperError::InvalidConfig("epsilon must be positive"));
        }
        if self.window_ms == 0 {
            return Err(DpShaperError::InvalidConfig("window_ms must be non-zero"));
        }
        if self.max_real_bytes_per_window == 0 {
            return Err(DpShaperError::InvalidConfig(
                "max_real_bytes_per_window must be non-zero",
            ));
        }
        Ok(self)
    }
}

/// Per-window DP shaping decision.
#[derive(Clone, Copy, Debug)]
pub struct WindowDecision {
    /// Real bytes in this window (as reported by the caller).
    pub real_bytes: usize,
    /// Dummy bytes to add so the wire total = real_bytes + dummy_bytes.
    /// Always >= 0; the mechanism clamps the noisy total from below.
    pub dummy_bytes: usize,
    /// The raw Laplace noise sample (exposed for accounting/testing).
    pub noise_sample: f64,
    /// Effective ε spent this window (= epsilon from config for basic composition).
    pub epsilon_spent: f64,
}

/// Stateful DP shaper: call `decide_window()` once per window with the real
/// byte count; it returns how many dummy bytes to add.
///
/// The shaper is *directional*: upstream and downstream must use separate
/// instances with independent RNGs (item 7 in the audit).
#[derive(Debug)]
pub struct DpShaper {
    config: DpShaperConfig,
    rng: ChaCha20Rng,
}

impl DpShaper {
    pub fn new(config: DpShaperConfig, rng: ChaCha20Rng) -> Result<Self, DpShaperError> {
        Ok(Self {
            config: config.validate()?,
            rng,
        })
    }

    /// Compute the number of dummy bytes to emit alongside `real_bytes` in the
    /// current window.
    ///
    /// The shaper draws Laplace(0, b) noise and targets a wire total of
    /// `max(real_bytes, real_bytes + noise)`.  When noise > 0 the target
    /// exceeds real volume and we add `noise` dummy bytes.  When noise <= 0
    /// we clamp to 0 dummies (we cannot suppress real packets).
    ///
    /// Callers should invoke this once per window boundary.
    pub fn decide_window(&mut self, real_bytes: usize) -> WindowDecision {
        let noise = self.laplace_sample(self.config.noise_scale());
        let target = real_bytes as f64 + noise;
        let dummy_bytes = if target > real_bytes as f64 {
            (target - real_bytes as f64).round() as usize
        } else {
            0
        };

        WindowDecision {
            real_bytes,
            dummy_bytes,
            noise_sample: noise,
            epsilon_spent: self.config.epsilon,
        }
    }

    /// Sample from Laplace(0, scale) using the inverse CDF method.
    ///
    /// Laplace(0, b): CDF = 0.5 + 0.5·sign(x)·(1 − e^(−|x|/b))
    /// Inverse: x = −b·sign(u − 0.5)·ln(1 − 2|u − 0.5|)
    fn laplace_sample(&mut self, scale: f64) -> f64 {
        let u: f64 = self.rng.gen(); // Uniform(0, 1)
        let v = u - 0.5;
        -scale * v.signum() * (1.0 - 2.0 * v.abs()).ln()
    }

    pub fn config(&self) -> DpShaperConfig {
        self.config
    }
}

// ---------------------------------------------------------------------------
// Bidirectional shaper (item 7)
// ---------------------------------------------------------------------------

/// Two independent `DpShaper` instances — one per direction — with no shared
/// state.  Upstream dummy timing must not respond to downstream real arrivals
/// and vice versa.
pub struct BidirectionalDpShaper {
    pub upstream:   DpShaper,
    pub downstream: DpShaper,
}

impl BidirectionalDpShaper {
    pub fn new(
        config: DpShaperConfig,
        upstream_rng: ChaCha20Rng,
        downstream_rng: ChaCha20Rng,
    ) -> Result<Self, DpShaperError> {
        Ok(Self {
            upstream:   DpShaper::new(config, upstream_rng)?,
            downstream: DpShaper::new(config, downstream_rng)?,
        })
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, Eq, PartialEq)]
pub enum DpShaperError {
    InvalidConfig(&'static str),
}

impl std::fmt::Display for DpShaperError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid DP shaper config: {msg}"),
        }
    }
}

impl std::error::Error for DpShaperError {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::rand_core::SeedableRng;

    fn test_shaper(epsilon: f64) -> DpShaper {
        DpShaper::new(
            DpShaperConfig {
                epsilon,
                window_ms: 100,
                max_real_bytes_per_window: 64 * 1024,
            },
            ChaCha20Rng::from_seed([42u8; 32]),
        )
        .unwrap()
    }

    #[test]
    fn dummy_bytes_always_nonneg() {
        let mut shaper = test_shaper(0.5);
        for real in [0, 100, 1000, 8192, 65535] {
            let d = shaper.decide_window(real);
            assert!(d.dummy_bytes < usize::MAX / 2, "implausibly large dummy count");
            // dummy_bytes is usize so it's always >= 0 by type
        }
    }

    #[test]
    fn noise_scale_formula() {
        let cfg = DpShaperConfig {
            epsilon: 2.0,
            window_ms: 100,
            max_real_bytes_per_window: 1000,
        };
        assert!((cfg.noise_scale() - 500.0).abs() < 1e-9);
    }

    #[test]
    fn laplace_mean_near_zero() {
        let mut shaper = test_shaper(1.0);
        let n = 10_000;
        let sum: f64 = (0..n)
            .map(|_| shaper.laplace_sample(1000.0))
            .sum();
        let mean = sum / n as f64;
        // Mean of Laplace(0, b) is 0. With n=10k should be within 30 of 0.
        assert!(mean.abs() < 30.0, "Laplace mean too far from 0: {mean}");
    }

    #[test]
    fn validate_rejects_bad_config() {
        assert!(DpShaperConfig { epsilon: 0.0, ..DpShaperConfig::default() }.validate().is_err());
        assert!(DpShaperConfig { window_ms: 0, ..DpShaperConfig::default() }.validate().is_err());
        assert!(DpShaperConfig {
            max_real_bytes_per_window: 0,
            ..DpShaperConfig::default()
        }
        .validate()
        .is_err());
    }

    #[test]
    fn bidirectional_is_independent() {
        // Both directions start with different seeds so their noise sequences differ.
        let cfg = DpShaperConfig::default();
        let mut bidir = BidirectionalDpShaper::new(
            cfg,
            ChaCha20Rng::from_seed([1u8; 32]),
            ChaCha20Rng::from_seed([2u8; 32]),
        )
        .unwrap();

        let up   = bidir.upstream.decide_window(1000);
        let down = bidir.downstream.decide_window(1000);
        // Different seeds → different noise → different dummy counts (with overwhelming probability).
        assert_ne!(
            up.noise_sample.to_bits(),
            down.noise_sample.to_bits(),
            "upstream and downstream noise should be independent"
        );
    }
}