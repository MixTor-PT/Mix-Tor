//! Continuous statistical self-testing (Phase 3 — item 12).
//!
//! Runs KS tests, spectral analysis, and autocorrelation checks on the
//! shaper's own output in real time.  If the output fails to match target
//! distributions, the caller can degrade gracefully or panic.
//!
//! # Tests performed
//!
//! 1. **KS test** on dummy packet size distribution vs. real size distribution.
//!    A high KS statistic means dummy sizes are distinguishable from real sizes.
//!
//! 2. **Lag-1 autocorrelation** on IAT stream.
//!    Values far from 0 indicate exploitable temporal patterns.
//!
//! 3. **Spectral peak detection** on IAT stream.
//!    A sharp peak at any frequency indicates a periodic signal (keepalive
//!    leakage or other regular pattern).
//!
//! 4. **Dummy ratio check**: dummy packets / total packets should stay within
//!    configured bounds.

use crate::composition::MixedFrame;

// ---------------------------------------------------------------------------
// Test configuration
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct SelfTestConfig {
    /// KS statistic threshold — fail if KS > this value.
    /// 0.05–0.15 is typical depending on sample size.
    pub ks_threshold: f64,

    /// Lag-1 autocorrelation magnitude threshold.
    /// Fail if |ac| > this value.
    pub autocorr_threshold: f64,

    /// Minimum acceptable dummy ratio (dummy_packets / total_packets).
    pub dummy_ratio_min: f64,

    /// Maximum acceptable dummy ratio.
    pub dummy_ratio_max: f64,

    /// Spectral peak threshold: fail if max PSD peak / mean PSD > this.
    /// Values > 10 indicate a suspicious periodic spike.
    pub spectral_peak_ratio_threshold: f64,
}

impl Default for SelfTestConfig {
    fn default() -> Self {
        Self {
            ks_threshold:                0.15,
            autocorr_threshold:          0.20,
            dummy_ratio_min:             0.30,
            dummy_ratio_max:             0.98,
            spectral_peak_ratio_threshold: 8.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct SelfTestResult {
    pub passed: bool,
    pub failures: Vec<SelfTestFailure>,
    pub ks_statistic: Option<f64>,
    pub lag1_autocorr: Option<f64>,
    pub dummy_ratio: Option<f64>,
    pub spectral_peak_ratio: Option<f64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SelfTestFailure {
    KsTestFailed { statistic: u32 /* f64 * 1000 as integer for Eq */ },
    AutocorrelationTooHigh { lag1_times_1000: i32 },
    DummyRatioOutOfBounds { ratio_times_1000: u32 },
    SpectralPeakDetected { ratio_times_100: u32 },
    InsufficientData,
}

impl SelfTestFailure {
    pub fn description(&self) -> String {
        match self {
            Self::KsTestFailed { statistic } => {
                format!("KS test failed: statistic = {:.3}", *statistic as f64 / 1000.0)
            }
            Self::AutocorrelationTooHigh { lag1_times_1000 } => {
                format!(
                    "lag-1 autocorrelation too high: {:.3}",
                    *lag1_times_1000 as f64 / 1000.0
                )
            }
            Self::DummyRatioOutOfBounds { ratio_times_1000 } => {
                format!(
                    "dummy ratio out of bounds: {:.3}",
                    *ratio_times_1000 as f64 / 1000.0
                )
            }
            Self::SpectralPeakDetected { ratio_times_100 } => {
                format!("spectral peak detected: ratio = {:.2}", *ratio_times_100 as f64 / 100.0)
            }
            Self::InsufficientData => "insufficient data for self-test".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// ShaperError
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum ShaperError {
    SelfTestFailed(Vec<SelfTestFailure>),
}

impl std::fmt::Display for ShaperError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SelfTestFailed(failures) => {
                let descs: Vec<String> = failures.iter().map(|e| e.description()).collect();
                write!(f, "shaper self-test failed: {}", descs.join("; "))
            }
        }
    }
}

impl std::error::Error for ShaperError {}

// ---------------------------------------------------------------------------
// Self-tester
// ---------------------------------------------------------------------------

/// Accumulates output frames and runs statistical checks on demand.
#[derive(Debug)]
pub struct SelfTester {
    config: SelfTestConfig,
    /// IAT samples (ms) of successive frames (real or dummy).
    iat_samples: Vec<f64>,
    /// Sizes of dummy frames.
    dummy_sizes: Vec<usize>,
    /// Sizes of real frames.
    real_sizes: Vec<usize>,
    /// Total real frames seen.
    real_count: usize,
    /// Total dummy frames seen.
    dummy_count: usize,
    /// Timestamp (ms since epoch) of the last frame, for IAT measurement.
    last_frame_at_ms: Option<u64>,
}

impl SelfTester {
    pub fn new(config: SelfTestConfig) -> Self {
        Self {
            config,
            iat_samples: Vec::new(),
            dummy_sizes: Vec::new(),
            real_sizes: Vec::new(),
            real_count: 0,
            dummy_count: 0,
            last_frame_at_ms: None,
        }
    }

    /// Record an emitted frame at `now_ms` (milliseconds since an arbitrary epoch).
    pub fn record_frame(&mut self, frame: &MixedFrame, now_ms: u64) {
        match frame {
            MixedFrame::Real(p) => {
                self.real_count += 1;
                self.real_sizes.push(p.len());
            }
            MixedFrame::Dummy(b) => {
                self.dummy_count += 1;
                self.dummy_sizes.push(b.len());
            }
        }

        if let Some(last) = self.last_frame_at_ms {
            let iat = now_ms.saturating_sub(last) as f64;
            self.iat_samples.push(iat);
        }
        self.last_frame_at_ms = Some(now_ms);
    }

    /// Run all statistical self-tests on accumulated data.
    ///
    /// Returns `Ok(SelfTestResult)` if all tests pass, or the result with
    /// `passed = false` listing all failures.
    pub fn run_tests(&self) -> SelfTestResult {
        let mut failures = Vec::new();
        let total = self.real_count + self.dummy_count;

        if total < 20 {
            return SelfTestResult {
                passed: false,
                failures: vec![SelfTestFailure::InsufficientData],
                ks_statistic: None,
                lag1_autocorr: None,
                dummy_ratio: None,
                spectral_peak_ratio: None,
            };
        }

        // --- KS test: dummy sizes vs real sizes ---
        let ks = self.ks_statistic(&self.dummy_sizes, &self.real_sizes);
        if ks > self.config.ks_threshold {
            failures.push(SelfTestFailure::KsTestFailed {
                statistic: (ks * 1000.0).round() as u32,
            });
        }

        // --- Lag-1 autocorrelation on IATs ---
        let ac = crate::spectral::lag1_autocorrelation(&self.iat_samples);
        if ac.abs() > self.config.autocorr_threshold {
            failures.push(SelfTestFailure::AutocorrelationTooHigh {
                lag1_times_1000: (ac * 1000.0).round() as i32,
            });
        }

        // --- Dummy ratio check ---
        let dummy_ratio = self.dummy_count as f64 / total as f64;
        if dummy_ratio < self.config.dummy_ratio_min || dummy_ratio > self.config.dummy_ratio_max {
            failures.push(SelfTestFailure::DummyRatioOutOfBounds {
                ratio_times_1000: (dummy_ratio * 1000.0).round() as u32,
            });
        }

        // --- Spectral peak detection on IATs (simplified DFT-based) ---
        let peak_ratio = self.spectral_peak_ratio(&self.iat_samples);
        if let Some(ratio) = peak_ratio {
            if ratio > self.config.spectral_peak_ratio_threshold {
                failures.push(SelfTestFailure::SpectralPeakDetected {
                    ratio_times_100: (ratio * 100.0).round() as u32,
                });
            }
        }

        SelfTestResult {
            passed: failures.is_empty(),
            failures,
            ks_statistic: Some(ks),
            lag1_autocorr: Some(ac),
            dummy_ratio: Some(dummy_ratio),
            spectral_peak_ratio: peak_ratio,
        }
    }

    /// Convenience: return `Err(ShaperError::SelfTestFailed)` if tests fail.
    pub fn assert_passing(&self) -> Result<SelfTestResult, ShaperError> {
        let result = self.run_tests();
        if result.passed {
            Ok(result)
        } else {
            Err(ShaperError::SelfTestFailed(result.failures))
        }
    }

    pub fn total_frames(&self) -> usize {
        self.real_count + self.dummy_count
    }

    // -----------------------------------------------------------------------
    // Statistical helpers
    // -----------------------------------------------------------------------

    /// Two-sample Kolmogorov-Smirnov statistic.
    ///
    /// Returns the maximum absolute difference between the empirical CDFs.
    /// 0 = identical distributions; 1 = maximally different.
    fn ks_statistic(&self, a: &[usize], b: &[usize]) -> f64 {
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }
        let mut a_sorted: Vec<f64> = a.iter().map(|&x| x as f64).collect();
        let mut b_sorted: Vec<f64> = b.iter().map(|&x| x as f64).collect();
        a_sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
        b_sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());

        // Merge and walk both CDFs.
        let na = a_sorted.len() as f64;
        let nb = b_sorted.len() as f64;
        let mut ia = 0usize;
        let mut ib = 0usize;
        let mut max_diff = 0.0f64;

        while ia < a_sorted.len() || ib < b_sorted.len() {
            let next_a = a_sorted.get(ia).copied().unwrap_or(f64::INFINITY);
            let next_b = b_sorted.get(ib).copied().unwrap_or(f64::INFINITY);
            let x = next_a.min(next_b);

            while ia < a_sorted.len() && a_sorted[ia] <= x {
                ia += 1;
            }
            while ib < b_sorted.len() && b_sorted[ib] <= x {
                ib += 1;
            }

            let cdf_a = ia as f64 / na;
            let cdf_b = ib as f64 / nb;
            max_diff = max_diff.max((cdf_a - cdf_b).abs());
        }

        max_diff
    }

    /// Simplified spectral peak ratio: max DFT magnitude / mean DFT magnitude
    /// over the IAT samples.  Uses a direct DFT (O(n²)) — fine for n ≤ 256.
    fn spectral_peak_ratio(&self, samples: &[f64]) -> Option<f64> {
        let n = samples.len();
        if n < 16 {
            return None;
        }
        // Use at most 256 samples for performance.
        let s: &[f64] = if n > 256 { &samples[n - 256..] } else { samples };
        let n = s.len();
        let mean = s.iter().sum::<f64>() / n as f64;
        let centered: Vec<f64> = s.iter().map(|x| x - mean).collect();

        // Compute DFT magnitudes for frequencies 1..n/2.
        let mut magnitudes = Vec::with_capacity(n / 2);
        for k in 1..n / 2 {
            let re: f64 = centered.iter().enumerate()
                .map(|(j, x)| x * (2.0 * std::f64::consts::PI * k as f64 * j as f64 / n as f64).cos())
                .sum();
            let im: f64 = centered.iter().enumerate()
                .map(|(j, x)| -x * (2.0 * std::f64::consts::PI * k as f64 * j as f64 / n as f64).sin())
                .sum();
            magnitudes.push((re * re + im * im).sqrt());
        }

        let max = magnitudes.iter().cloned().fold(0.0f64, f64::max);
        let mean_mag = magnitudes.iter().sum::<f64>() / magnitudes.len() as f64;
        if mean_mag < f64::EPSILON {
            return None;
        }
        Some(max / mean_mag)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clumping::Packet;

    fn dummy_frame(len: usize) -> MixedFrame {
        MixedFrame::Dummy(vec![0u8; len])
    }

    fn real_frame(len: usize) -> MixedFrame {
        MixedFrame::Real(Packet::new(vec![1u8; len]).unwrap())
    }

    #[test]
    fn insufficient_data_fails_gracefully() {
        let tester = SelfTester::new(SelfTestConfig::default());
        let result = tester.run_tests();
        assert!(!result.passed);
        assert!(result.failures.contains(&SelfTestFailure::InsufficientData));
    }

    #[test]
    fn good_output_passes_tests() {
        let mut tester = SelfTester::new(SelfTestConfig::default());
        let mut now = 0u64;
        // Emit frames at genuinely aperiodic intervals (a deterministic LCG
        // gives white-ish, low-autocorrelation IATs — like the shaper's Poisson
        // departure clock).  The previous version advanced `now` by exactly 1ms
        // between each real/dummy pair, which created a strict period-2 IAT
        // oscillation [~120, 1, ~120, 1, …]: lag-1 autocorrelation ≈ −1 and a
        // sharp Nyquist spectral peak — both legitimately flagged.
        let mut state = 0x1234_5678_9abc_def0u64;
        for i in 0usize..200 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let iat = 50 + (state >> 40) % 150; // aperiodic IAT in [50, 200) ms
            now += iat;
            if i % 2 == 0 {
                tester.record_frame(&real_frame(800 + (i * 13 % 200)), now);
            } else {
                tester.record_frame(&dummy_frame(900 + (i * 7 % 200)), now);
            }
        }
        let result = tester.run_tests();
        // Should not fail autocorrelation or spectral tests for irregular frames.
        let non_ks_failures: Vec<_> = result
            .failures
            .iter()
            .filter(|f| !matches!(f, SelfTestFailure::KsTestFailed { .. }))
            .collect();
        assert!(
            non_ks_failures.is_empty(),
            "unexpected non-KS failures: {:?}",
            non_ks_failures
        );
    }

    #[test]
    fn periodic_iats_trigger_spectral_failure() {
        let mut tester = SelfTester::new(SelfTestConfig {
            spectral_peak_ratio_threshold: 3.0, // very sensitive
            dummy_ratio_min: 0.0,               // don't fail on ratio
            ..SelfTestConfig::default()
        });
        // A periodic-but-varying IAT pattern (period 10) concentrates spectral
        // power in a single DFT bin, which the detector must flag.  Note: a
        // *constant* IAT will NOT trip this test — after mean removal it is
        // identically zero, so the mean-removed DFT has no power anywhere; that
        // degenerate case has zero AC energy and is a different failure mode.
        let mut now = 0u64;
        for i in 0..200u64 {
            let phase = (i % 10) as f64 / 10.0 * 2.0 * std::f64::consts::PI;
            let iat = (100.0 + 60.0 * phase.sin()).round() as u64;
            now += iat;
            tester.record_frame(&dummy_frame(500), now);
        }
        let result = tester.run_tests();
        let has_spectral = result
            .failures
            .iter()
            .any(|f| matches!(f, SelfTestFailure::SpectralPeakDetected { .. }));
        assert!(has_spectral, "periodic IATs should trigger spectral failure");
    }

    #[test]
    fn ks_statistic_identical_distributions() {
        let tester = SelfTester::new(SelfTestConfig::default());
        let a = vec![100, 200, 300, 400];
        let b = vec![100, 200, 300, 400];
        let ks = tester.ks_statistic(&a, &b);
        assert!(ks < 1e-9, "identical distributions should have KS = 0, got {ks}");
    }

    #[test]
    fn ks_statistic_disjoint_distributions() {
        let tester = SelfTester::new(SelfTestConfig::default());
        let a = vec![100, 150, 200];
        let b = vec![500, 600, 700];
        let ks = tester.ks_statistic(&a, &b);
        assert!(ks > 0.9, "disjoint distributions should have KS near 1, got {ks}");
    }
}
