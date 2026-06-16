//! Spectrally-white inter-arrival time generation (Phase 1 — items 6 & 9).
//!
//! # Problem
//!
//! Sampling keepalive delays directly from an empirical IAT pool reproduces
//! the autocorrelation structure of the real traffic, which leaks information.
//! Even with jitter, if successive real IATs are correlated (e.g., because
//! the real application sends in bursts with regular inter-burst gaps) the
//! dummy IATs will be too.
//!
//! # Solution
//!
//! Generate synthetic IATs from a process with *controlled* autocorrelation —
//! ideally zero autocorrelation at all lags > 0 (white noise).
//!
//! The `SpectralIatGenerator` targets:
//!   E[d_i · d_{i+1}] = E[d_i] · E[d_{i+1}]  (zero covariance)
//!
//! by drawing each delay from an independent Laplace-perturbed exponential
//! distribution, then *rejecting* samples that are too close to the previous
//! value (anti-autocorrelation jitter, item 9).
//!
//! Exponential inter-arrival times correspond to a Poisson process — the
//! maximum-entropy distribution for a given mean rate — which ensures we leak
//! nothing about the real traffic pattern beyond the mean rate.
//!
//! # Spectral analysis note
//!
//! A fixed keepalive period T produces a spectral peak at 1/T Hz.  By drawing
//! from Exponential(λ) with λ = 1/target_iat_ms, power is spread uniformly
//! across all frequencies (flat spectrum) and the spectral peak disappears.

use rand::Rng;
use rand_chacha::ChaCha20Rng;

// ---------------------------------------------------------------------------
// SpectralIatGenerator
// ---------------------------------------------------------------------------

/// Configuration for the spectrally-white IAT generator.
#[derive(Clone, Copy, Debug)]
pub struct SpectralConfig {
    /// Target mean inter-arrival time in milliseconds.
    ///
    /// This is the λ parameter of the underlying exponential: each draw has
    /// E[IAT] = target_iat_ms.  Set this to the observed real traffic IAT
    /// mean so dummy and real cadences match in expectation.
    pub target_iat_ms: f64,

    /// Laplace noise scale in milliseconds (additive perturbation on top of
    /// the exponential draw).  Larger = more jitter, more spectral whitening.
    /// Typical: 0.1 × target_iat_ms.
    pub laplace_scale_ms: f64,

    /// Hard floor on generated IAT (ms).  No delay shorter than this.
    pub floor_ms: f64,

    /// Hard ceiling on generated IAT (ms).
    pub ceiling_ms: f64,

    /// Anti-autocorrelation window: if the new sample is within this fraction
    /// of the previous sample, reject and redraw (up to `max_rejections` times).
    /// Set to 0.0 to disable.
    pub similarity_rejection_fraction: f64,

    /// Maximum rejection attempts before accepting anyway.
    pub max_rejections: usize,
}

impl Default for SpectralConfig {
    fn default() -> Self {
        Self {
            target_iat_ms: 200.0,
            laplace_scale_ms: 20.0,
            floor_ms: 10.0,
            ceiling_ms: 5_000.0,
            similarity_rejection_fraction: 0.10,
            max_rejections: 8,
        }
    }
}

impl SpectralConfig {
    pub fn validate(self) -> Result<Self, SpectralError> {
        if self.target_iat_ms <= 0.0 {
            return Err(SpectralError::InvalidConfig("target_iat_ms must be positive"));
        }
        if self.floor_ms <= 0.0 {
            return Err(SpectralError::InvalidConfig("floor_ms must be positive"));
        }
        if self.floor_ms >= self.ceiling_ms {
            return Err(SpectralError::InvalidConfig("floor_ms must be less than ceiling_ms"));
        }
        Ok(self)
    }
}

/// Generates spectrally-white inter-arrival times.
///
/// Each call to `next_delay_ms()` returns an IAT sample that:
///  - is drawn from Exponential(1/target_iat_ms) + Laplace(0, laplace_scale)
///  - is clamped to [floor_ms, ceiling_ms]
///  - has low autocorrelation with the previous sample (rejection sampling)
#[derive(Debug)]
pub struct SpectralIatGenerator {
    config: SpectralConfig,
    rng: ChaCha20Rng,
    /// Previous sample for anti-autocorrelation rejection.
    last_sample_ms: Option<f64>,
}

impl SpectralIatGenerator {
    pub fn new(config: SpectralConfig, rng: ChaCha20Rng) -> Result<Self, SpectralError> {
        Ok(Self {
            config: config.validate()?,
            rng,
            last_sample_ms: None,
        })
    }

    /// Update the target mean IAT (e.g., from observed real traffic rate).
    pub fn set_target_iat_ms(&mut self, ms: f64) {
        if ms > 0.0 {
            self.config.target_iat_ms = ms.clamp(self.config.floor_ms, self.config.ceiling_ms);
        }
    }

    /// Generate the next IAT sample in milliseconds.
    ///
    /// Algorithm:
    ///  1. Draw X ~ Exponential(1/target_iat_ms)
    ///  2. Draw N ~ Laplace(0, laplace_scale_ms)
    ///  3. d = clamp(X + N, floor_ms, ceiling_ms)
    ///  4. If |d - last| / last < similarity_rejection_fraction, redraw (up to max_rejections)
    ///  5. Record d as last_sample_ms and return it.
    pub fn next_delay_ms(&mut self) -> f64 {
        let mut attempts = 0;
        loop {
            let candidate = self.raw_sample();
            let clamped = candidate.clamp(self.config.floor_ms, self.config.ceiling_ms);

            let accept = match self.last_sample_ms {
                None => true,
                Some(last) if last <= 0.0 => true,
                Some(last) => {
                    let similarity = (clamped - last).abs() / last;
                    similarity >= self.config.similarity_rejection_fraction
                        || attempts >= self.config.max_rejections
                }
            };

            if accept {
                self.last_sample_ms = Some(clamped);
                return clamped;
            }
            attempts += 1;
        }
    }

    /// Next delay as a `std::time::Duration`.
    pub fn next_delay(&mut self) -> std::time::Duration {
        let ms = self.next_delay_ms();
        std::time::Duration::from_millis(ms.round() as u64)
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Draw from Exponential(1/target_iat_ms) + Laplace(0, laplace_scale_ms).
    fn raw_sample(&mut self) -> f64 {
        let exp_sample = self.exponential_sample(self.config.target_iat_ms);
        let lap_sample = self.laplace_sample(self.config.laplace_scale_ms);
        exp_sample + lap_sample
    }

    /// Sample from Exponential with mean `mean_ms` using the inverse CDF.
    ///
    /// X = −mean · ln(U),  U ~ Uniform(0, 1)
    fn exponential_sample(&mut self, mean_ms: f64) -> f64 {
        let u: f64 = self.rng.gen::<f64>().max(f64::EPSILON); // avoid ln(0)
        -mean_ms * u.ln()
    }

    /// Sample from Laplace(0, scale) using the inverse CDF.
    fn laplace_sample(&mut self, scale: f64) -> f64 {
        if scale <= 0.0 {
            return 0.0;
        }
        let u: f64 = self.rng.gen::<f64>() - 0.5; // Uniform(-0.5, 0.5)
        -scale * u.signum() * (1.0 - 2.0 * u.abs()).ln()
    }

    pub fn config(&self) -> SpectralConfig {
        self.config
    }
}

// ---------------------------------------------------------------------------
// Autocorrelation checker (for self-testing)
// ---------------------------------------------------------------------------

/// Compute the lag-1 autocorrelation of a slice.
///
/// Returns a value in [-1, 1].  Values near 0 indicate white noise;
/// values far from 0 indicate exploitable temporal structure.
pub fn lag1_autocorrelation(samples: &[f64]) -> f64 {
    if samples.len() < 2 {
        return 0.0;
    }
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let variance: f64 = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    if variance < f64::EPSILON {
        return 0.0;
    }
    let covariance: f64 = samples
        .windows(2)
        .map(|w| (w[0] - mean) * (w[1] - mean))
        .sum::<f64>()
        / (n - 1.0);
    covariance / variance
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Eq, PartialEq)]
pub enum SpectralError {
    InvalidConfig(&'static str),
}

impl std::fmt::Display for SpectralError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid spectral config: {msg}"),
        }
    }
}

impl std::error::Error for SpectralError {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::rand_core::SeedableRng;

    fn test_gen(target_ms: f64) -> SpectralIatGenerator {
        SpectralIatGenerator::new(
            SpectralConfig {
                target_iat_ms: target_ms,
                laplace_scale_ms: target_ms * 0.1,
                floor_ms: 5.0,
                ceiling_ms: target_ms * 10.0,
                similarity_rejection_fraction: 0.10,
                max_rejections: 8,
            },
            ChaCha20Rng::from_seed([7u8; 32]),
        )
        .unwrap()
    }

    #[test]
    fn samples_within_bounds() {
        let mut gen = test_gen(200.0);
        for _ in 0..1_000 {
            let d = gen.next_delay_ms();
            assert!(d >= 5.0, "below floor: {d}");
            assert!(d <= 2_000.0, "above ceiling: {d}");
        }
    }

    #[test]
    fn mean_near_target() {
        let mut gen = test_gen(200.0);
        let n = 5_000;
        let sum: f64 = (0..n).map(|_| gen.next_delay_ms()).sum();
        let mean = sum / n as f64;
        // Exponential mean = target; clamping shifts it slightly.
        // Allow ±40% tolerance due to clamping effects.
        assert!(
            (mean - 200.0).abs() < 80.0,
            "mean {mean:.1} too far from target 200ms"
        );
    }

    #[test]
    fn lag1_autocorrelation_near_zero() {
        let mut gen = test_gen(200.0);
        let samples: Vec<f64> = (0..2_000).map(|_| gen.next_delay_ms()).collect();
        let ac = lag1_autocorrelation(&samples);
        // Exponential distribution has 0 autocorrelation by construction.
        // Rejection sampling adds slight negative correlation; allow ±0.15.
        assert!(
            ac.abs() < 0.15,
            "lag-1 autocorrelation {ac:.3} exceeds threshold — temporal leakage detected"
        );
    }

    #[test]
    fn set_target_updates_generator() {
        let mut gen = test_gen(200.0);
        gen.set_target_iat_ms(50.0);
        assert!((gen.config().target_iat_ms - 50.0).abs() < 1e-9);
    }

    #[test]
    fn validate_rejects_bad_config() {
        assert!(SpectralConfig { target_iat_ms: 0.0, ..SpectralConfig::default() }
            .validate()
            .is_err());
        assert!(SpectralConfig { floor_ms: 0.0, ..SpectralConfig::default() }
            .validate()
            .is_err());
        assert!(SpectralConfig { floor_ms: 500.0, ceiling_ms: 100.0, ..SpectralConfig::default() }
            .validate()
            .is_err());
    }
}