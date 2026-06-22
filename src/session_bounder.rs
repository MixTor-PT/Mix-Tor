//! Session bounding: volume and duration classes (Phase 1 — item 5).
//!
//! All sessions in the same class transmit *exactly* the same total bytes and
//! last *exactly* the same duration.  If real traffic finishes early, pad with
//! dummies to the target.  If real traffic exceeds the target, split into
//! additional sessions (the caller's responsibility) or clip.
//!
//! This provides information-theoretic protection of session length: an
//! observer cannot distinguish a 100 KB session from a 900 KB session if both
//! are bucketed to `Medium` (1 MB target).

use rand::Rng;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Session classes
// ---------------------------------------------------------------------------

/// Coarse session size/duration buckets.
///
/// Each class defines a *target* total wire volume and a *target* duration.
/// The actual real traffic may be shorter/smaller; padding fills the gap.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum SessionClass {
    /// ≤ 64 KB real traffic, up to 30 seconds.
    Short,
    /// ≤ 1 MB real traffic, up to 5 minutes.
    Medium,
    /// ≤ 16 MB real traffic, up to 30 minutes.
    Long,
    /// ≤ 256 MB real traffic, up to 2 hours.  Sessions that outrun Long
    /// (sustained large downloads, long-lived circuits) are bucketed here
    /// rather than being cut off at 30 minutes with real traffic still flowing
    /// — an abrupt wire→0 that is itself a tell.
    VeryLong,
}

impl SessionClass {
    /// Total wire bytes to emit for this session class (real + dummy).
    pub fn target_volume_bytes(self) -> usize {
        match self {
            Self::Short   => 64  * 1_024,
            Self::Medium  => 1   * 1_024 * 1_024,
            Self::Long    => 16  * 1_024 * 1_024,
            Self::VeryLong => 256 * 1_024 * 1_024,
        }
    }

    /// Total wall-clock duration this session should appear to last.
    pub fn target_duration(self) -> Duration {
        match self {
            Self::Short   => Duration::from_secs(30),
            Self::Medium  => Duration::from_secs(5 * 60),
            Self::Long    => Duration::from_secs(30 * 60),
            Self::VeryLong => Duration::from_secs(2 * 60 * 60),
        }
    }

    /// Select the smallest class whose volume fits `real_bytes`.
    pub fn for_volume(real_bytes: usize) -> Self {
        if real_bytes <= Self::Short.target_volume_bytes() {
            Self::Short
        } else if real_bytes <= Self::Medium.target_volume_bytes() {
            Self::Medium
        } else if real_bytes <= Self::Long.target_volume_bytes() {
            Self::Long
        } else {
            Self::VeryLong
        }
    }

    /// How many bytes of padding are needed given `real_bytes_sent` so far.
    ///
    /// Returns 0 if real volume has met or exceeded the target (e.g., due to
    /// splitting a real session across class boundaries — the caller handles
    /// that case).
    pub fn padding_remaining(self, real_bytes_sent: usize) -> usize {
        self.target_volume_bytes().saturating_sub(real_bytes_sent)
    }
}

// ---------------------------------------------------------------------------
// Per-session fuzzy class boundaries
// ---------------------------------------------------------------------------

/// Jitter fraction applied to each class boundary: ±30 % multiplicative noise.
///
/// The boundaries are far enough apart (64 KB → 1 MB is a 16× jump) that ±30 %
/// never violates the short_max < medium_max < long_max ordering invariant.
const BOUNDARY_JITTER: f64 = 0.30;

/// Per-session class-boundary thresholds drawn with multiplicative noise.
///
/// Hard-coded boundaries (64 KB / 1 MB / 16 MB) are in the source code.  An
/// adversary who reads this code knows those values and can infer a session's
/// real-volume range from its wire duration.  Sampling fresh boundaries each
/// session removes that anchor: the adversary cannot distinguish "a 30 s session
/// that sent 55 KB" from "a 30 s session that sent 75 KB" because either could
/// be Short depending on that session's threshold draw.
///
/// The two sides of the same connection sample *independently*, so their class
/// assignments (and hence wire durations) may differ — further degrading cross-
/// side duration matching.
pub struct FuzzedClassThresholds {
    short_max:  usize,
    medium_max: usize,
    long_max:   usize,
}

impl FuzzedClassThresholds {
    /// Sample fresh thresholds from `rng`.
    pub fn sample(rng: &mut impl Rng) -> Self {
        let t = Self {
            short_max:  Self::fuzz(rng, SessionClass::Short.target_volume_bytes()),
            medium_max: Self::fuzz(rng, SessionClass::Medium.target_volume_bytes()),
            long_max:   Self::fuzz(rng, SessionClass::Long.target_volume_bytes()),
        };
        // Ordering invariant: the 16× gap between classes means ±30% can never
        // flip the order, but assert in debug builds for safety.
        debug_assert!(t.short_max < t.medium_max && t.medium_max < t.long_max);
        t
    }

    fn fuzz(rng: &mut impl Rng, base: usize) -> usize {
        let lo = 1.0 - BOUNDARY_JITTER;
        let hi = 1.0 + BOUNDARY_JITTER;
        let factor: f64 = rng.gen_range(lo..=hi);
        ((base as f64) * factor) as usize
    }

    /// Assign a `SessionClass` to a session that sent `real_bytes`.
    pub fn classify(&self, real_bytes: usize) -> SessionClass {
        if real_bytes <= self.short_max {
            SessionClass::Short
        } else if real_bytes <= self.medium_max {
            SessionClass::Medium
        } else if real_bytes <= self.long_max {
            SessionClass::Long
        } else {
            SessionClass::VeryLong
        }
    }
}

// ---------------------------------------------------------------------------
// Session bounder
// ---------------------------------------------------------------------------

/// Tracks real-byte and elapsed-time progress for a single session and
/// determines how much dummy padding is still required.
///
/// Typical usage:
/// ```text
/// let mut bounder = SessionBounder::new(SessionClass::for_volume(expected_bytes));
/// while session_active {
///     let mixed = composer.compose(burst);
///     bounder.record_real_bytes(burst_real_bytes);
///     let pad = bounder.dummy_bytes_needed_now();
///     // emit `pad` bytes of dummy traffic
/// }
/// // At session end emit bounder.final_padding_bytes() to hit exact volume target.
/// ```
#[derive(Debug)]
pub struct SessionBounder {
    class: SessionClass,
    real_bytes_sent: usize,
    session_start: std::time::Instant,
}

impl SessionBounder {
    pub fn new(class: SessionClass) -> Self {
        Self {
            class,
            real_bytes_sent: 0,
            session_start: std::time::Instant::now(),
        }
    }

    pub fn class(&self) -> SessionClass {
        self.class
    }

    /// Record that `bytes` of real traffic have been sent.
    pub fn record_real_bytes(&mut self, bytes: usize) {
        self.real_bytes_sent = self.real_bytes_sent.saturating_add(bytes);
    }

    /// Dummy bytes still needed to reach the volume target.
    pub fn volume_padding_remaining(&self) -> usize {
        self.class.padding_remaining(self.real_bytes_sent)
    }

    /// How long until the session should nominally end (may be negative if
    /// we've run over; callers should clamp to 0).
    pub fn time_remaining(&self) -> Duration {
        let elapsed = self.session_start.elapsed();
        self.class
            .target_duration()
            .checked_sub(elapsed)
            .unwrap_or(Duration::ZERO)
    }

    /// True when both the volume target has been met (real traffic has consumed
    /// or exceeded it) and the time target has elapsed.
    pub fn is_complete(&self) -> bool {
        self.real_bytes_sent >= self.class.target_volume_bytes()
            && self.time_remaining() == Duration::ZERO
    }

    /// Dummy bytes the caller must emit *right now* to stay on track for the
    /// volume target.  This is a simple linear schedule:
    ///   expected_dummy_bytes_by_now = total_padding * (elapsed / target_duration)
    ///   overdue = expected_dummy_bytes_by_now - dummy_bytes_emitted_so_far
    ///
    /// Callers should call this frequently (e.g., each keepalive tick) and
    /// emit the returned number of dummy bytes.
    pub fn dummy_bytes_needed_now(&self, dummy_bytes_emitted: usize) -> usize {
        let total_padding = self.class.target_volume_bytes().saturating_sub(self.real_bytes_sent);
        if total_padding == 0 {
            return 0;
        }
        let elapsed   = self.session_start.elapsed();
        let target    = self.class.target_duration();
        let progress  = elapsed.as_secs_f64() / target.as_secs_f64();
        let progress  = progress.min(1.0);
        let expected_so_far = (total_padding as f64 * progress).round() as usize;
        expected_so_far.saturating_sub(dummy_bytes_emitted)
    }

    /// Total dummy bytes to emit at session teardown to hit the exact volume target.
    pub fn final_padding_bytes(&self, dummy_bytes_emitted: usize) -> usize {
        let total_padding = self.class.target_volume_bytes().saturating_sub(self.real_bytes_sent);
        total_padding.saturating_sub(dummy_bytes_emitted)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn class_selection() {
        assert_eq!(SessionClass::for_volume(0), SessionClass::Short);
        assert_eq!(SessionClass::for_volume(64 * 1024), SessionClass::Short);
        assert_eq!(SessionClass::for_volume(64 * 1024 + 1), SessionClass::Medium);
        assert_eq!(SessionClass::for_volume(1024 * 1024), SessionClass::Medium);
        assert_eq!(SessionClass::for_volume(1024 * 1024 + 1), SessionClass::Long);
        assert_eq!(SessionClass::for_volume(16 * 1024 * 1024), SessionClass::Long);
        assert_eq!(SessionClass::for_volume(16 * 1024 * 1024 + 1), SessionClass::VeryLong);
        assert_eq!(SessionClass::for_volume(256 * 1024 * 1024), SessionClass::VeryLong);
    }

    #[test]
    fn padding_remaining_never_negative() {
        let class = SessionClass::Short;
        // Over-sending real traffic should clamp to 0 padding, not underflow.
        assert_eq!(class.padding_remaining(class.target_volume_bytes() + 9999), 0);
    }

    #[test]
    fn volume_padding_tracks_real_bytes() {
        let mut bounder = SessionBounder::new(SessionClass::Short);
        let target = SessionClass::Short.target_volume_bytes();
        assert_eq!(bounder.volume_padding_remaining(), target);

        bounder.record_real_bytes(1024);
        assert_eq!(bounder.volume_padding_remaining(), target - 1024);

        bounder.record_real_bytes(target); // overshoot
        assert_eq!(bounder.volume_padding_remaining(), 0);
    }

    #[test]
    fn final_padding_adds_up_to_target() {
        let mut bounder = SessionBounder::new(SessionClass::Short);
        let target = SessionClass::Short.target_volume_bytes();

        // Real sends 10 KB, dummy sends 5 KB so far.
        bounder.record_real_bytes(10 * 1024);
        let dummy_so_far = 5 * 1024usize;
        let final_pad = bounder.final_padding_bytes(dummy_so_far);

        assert_eq!(
            10 * 1024 + dummy_so_far + final_pad,
            target,
            "real + dummy + final_pad should equal target volume"
        );
    }

    #[test]
    fn targets_are_sensible() {
        // Verify the class ordering makes sense.
        assert!(SessionClass::Short.target_volume_bytes() < SessionClass::Medium.target_volume_bytes());
        assert!(SessionClass::Medium.target_volume_bytes() < SessionClass::Long.target_volume_bytes());
        assert!(SessionClass::Long.target_volume_bytes() < SessionClass::VeryLong.target_volume_bytes());
        assert!(SessionClass::Short.target_duration() < SessionClass::Medium.target_duration());
        assert!(SessionClass::Medium.target_duration() < SessionClass::Long.target_duration());
        assert!(SessionClass::Long.target_duration() < SessionClass::VeryLong.target_duration());
    }

    #[test]
    fn fuzzy_thresholds_preserve_ordering() {
        use rand::{rngs::StdRng, SeedableRng};
        // Run many samples; ±30% jitter should never flip short < medium < long.
        for seed in 0u64..200 {
            let mut rng = StdRng::seed_from_u64(seed);
            let t = FuzzedClassThresholds::sample(&mut rng);
            assert!(t.short_max < t.medium_max, "seed {seed}: short_max >= medium_max");
            assert!(t.medium_max < t.long_max, "seed {seed}: medium_max >= long_max");
        }
    }

    #[test]
    fn fuzzy_thresholds_classify_correctly() {
        use rand::{rngs::StdRng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(42);
        let t = FuzzedClassThresholds::sample(&mut rng);
        // Zero bytes always lands in the smallest class.
        assert_eq!(t.classify(0), SessionClass::Short);
        // Very large bytes always land in VeryLong.
        assert_eq!(t.classify(512 * 1024 * 1024), SessionClass::VeryLong);
        // Just above long_max lands in VeryLong.
        assert_eq!(t.classify(t.long_max + 1), SessionClass::VeryLong);
        // Just at short_max lands in Short.
        assert_eq!(t.classify(t.short_max), SessionClass::Short);
    }
}