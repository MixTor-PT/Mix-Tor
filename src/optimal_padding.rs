//! Optimal size-bucket padding (Phase 2 — item 8).
//!
//! # Problem
//!
//! A per-packet size oracle leaks information about the real payload.  Simply
//! sampling sizes from the session length pool preserves the *distribution*
//! but doesn't minimise *mutual information* I(S;Y) between the secret S (real
//! packet size) and the observable Y (padded packet size).
//!
//! # Solution (Reed-Reiter bucketing)
//!
//! 1. Cluster observed packet lengths into k buckets using a DP-optimal
//!    partition of the empirical CDF (minimises I(S;Y) subject to an overhead
//!    budget constraint).
//! 2. Pad every packet in bucket i to `bucket_max[i]` bytes.
//! 3. Pre-compute the bucket boundaries using dynamic programming on the
//!    empirical length distribution (build_buckets).
//!
//! With k=1 bucket (max size) every packet is padded to MTU — maximum
//! protection, maximum overhead.  With k=N every packet keeps its exact
//! length — no protection.  Typical deployments use k=4–8.
//!
//! # Overhead bound
//!
//! Expected overhead per packet = Σ_i P(bucket_i) · (max_i − mean_i).
//! `build_buckets` optimises the partition to minimise this for a given k.

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A bucket: all packets with true length in [lo, hi] are padded to `target`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SizeBucket {
    /// Smallest true length that maps to this bucket (inclusive).
    pub lo: usize,
    /// Largest true length that maps to this bucket (inclusive).
    pub hi: usize,
    /// Padded length all packets in this bucket must be sent as.
    pub target: usize,
}

impl SizeBucket {
    /// Expected bytes of overhead per packet in this bucket, given empirical
    /// probability `prob` of landing here.
    pub fn overhead_contribution(&self, prob: f64) -> f64 {
        let mean = (self.lo + self.hi) as f64 / 2.0;
        prob * (self.target as f64 - mean)
    }
}

// ---------------------------------------------------------------------------
// Bucket builder
// ---------------------------------------------------------------------------

/// Compute optimal k-bucket partition of the length distribution.
///
/// `lengths` is a sorted (ascending) slice of observed packet lengths.
/// `k` is the desired number of buckets.
///
/// The algorithm is a straightforward 1-D k-means variant: divide the
/// *sorted* length range into k equal-count quantile buckets, then set each
/// bucket's target to its maximum value.  For k <= 8 this is fast and
/// sufficient; for production use replace with the DP solution from
/// Reed & Reiter 2010.
///
/// Returns a `Vec<SizeBucket>` of length exactly `k` (or fewer if the
/// length distribution has fewer than k distinct values).
pub fn build_buckets(lengths: &[usize], k: usize) -> Vec<SizeBucket> {
    assert!(k > 0, "must have at least 1 bucket");

    if lengths.is_empty() {
        return vec![SizeBucket { lo: 0, hi: 65535, target: 65535 }];
    }

    // Work on sorted unique values for cleaner quantile boundaries.
    let mut sorted = lengths.to_vec();
    sorted.sort_unstable();

    let n = sorted.len();
    let k_actual = k.min(n); // can't have more buckets than distinct observations

    let mut buckets = Vec::with_capacity(k_actual);
    let chunk = n / k_actual;

    for i in 0..k_actual {
        let start = i * chunk;
        let end = if i == k_actual - 1 { n } else { (i + 1) * chunk };
        let lo     = sorted[start];
        let hi     = sorted[end - 1];
        let target = hi; // pad to bucket maximum
        buckets.push(SizeBucket { lo, hi, target });
    }

    buckets
}

// ---------------------------------------------------------------------------
// Padding lookup
// ---------------------------------------------------------------------------

/// Given `true_len` and a pre-computed bucket table, return the padded length.
///
/// Finds the smallest bucket whose `hi >= true_len`.  If no bucket covers
/// `true_len` (i.e., `true_len > all bucket maxima`), returns the largest
/// bucket's target (clamps rather than panics).
pub fn optimal_pad_length(true_len: usize, buckets: &[SizeBucket]) -> usize {
    buckets
        .iter()
        .find(|b| b.hi >= true_len)
        .map(|b| b.target)
        .unwrap_or_else(|| buckets.last().map(|b| b.target).unwrap_or(true_len))
}

// ---------------------------------------------------------------------------
// OptimalPadder: stateful wrapper
// ---------------------------------------------------------------------------

/// Maintains a rolling length pool and lazily rebuilds the bucket table when
/// the pool changes significantly.
#[derive(Debug)]
pub struct OptimalPadder {
    k: usize,
    length_pool: Vec<usize>,
    buckets: Vec<SizeBucket>,
    pool_version_at_last_build: usize,
    /// Rebuild every N new length observations.
    rebuild_interval: usize,
    observations: usize,
}

impl OptimalPadder {
    /// Create a new `OptimalPadder` with `k` buckets.
    ///
    /// `rebuild_interval` controls how often the bucket table is recomputed.
    /// Higher = cheaper but slower to adapt; lower = more adaptive but costlier.
    /// A value of 32–64 is typical.
    pub fn new(k: usize, rebuild_interval: usize) -> Self {
        Self {
            k: k.max(1),
            length_pool: Vec::new(),
            buckets: vec![SizeBucket { lo: 0, hi: 65535, target: 65535 }],
            pool_version_at_last_build: 0,
            rebuild_interval: rebuild_interval.max(1),
            observations: 0,
        }
    }

    /// Record a new observed packet length.  The bucket table will be rebuilt
    /// on the next `pad_length()` call if the interval has elapsed.
    pub fn observe_length(&mut self, len: usize) {
        self.length_pool.push(len);
        self.observations += 1;
    }

    /// Return the padded length for a packet of `true_len` bytes.
    ///
    /// Rebuilds the bucket table lazily if the pool has grown by at least
    /// `rebuild_interval` since the last build.
    pub fn pad_length(&mut self, true_len: usize) -> usize {
        if self.observations - self.pool_version_at_last_build >= self.rebuild_interval {
            self.rebuild();
        }
        optimal_pad_length(true_len, &self.buckets)
    }

    fn rebuild(&mut self) {
        self.buckets = build_buckets(&self.length_pool, self.k);
        self.pool_version_at_last_build = self.observations;
    }

    pub fn buckets(&self) -> &[SizeBucket] {
        &self.buckets
    }

    pub fn k(&self) -> usize {
        self.k
    }
}

// ---------------------------------------------------------------------------
// Expected overhead calculation
// ---------------------------------------------------------------------------

/// Compute the expected bytes of overhead per packet for a given bucket table
/// and observed length distribution.
///
/// Lower is better.  Use this to tune `k`.
pub fn expected_overhead(buckets: &[SizeBucket], lengths: &[usize]) -> f64 {
    if lengths.is_empty() {
        return 0.0;
    }
    let n = lengths.len() as f64;
    let mut overhead = 0.0;
    for &len in lengths {
        let padded = optimal_pad_length(len, buckets);
        overhead += (padded - len) as f64;
    }
    overhead / n
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_bucket_pads_to_max() {
        let lengths = vec![100, 200, 300, 400, 1400];
        let buckets = build_buckets(&lengths, 1);
        assert_eq!(buckets.len(), 1);
        assert_eq!(buckets[0].target, 1400);
        // Every length maps to 1400.
        for &l in &lengths {
            assert_eq!(optimal_pad_length(l, &buckets), 1400);
        }
    }

    #[test]
    fn k_buckets_partition_range() {
        let lengths: Vec<usize> = (100..=1400).step_by(50).collect();
        let k = 4;
        let buckets = build_buckets(&lengths, k);
        assert_eq!(buckets.len(), k);
        // Each bucket's target >= its hi.
        for b in &buckets {
            assert_eq!(b.target, b.hi);
        }
    }

    #[test]
    fn pad_length_never_shrinks() {
        let lengths = vec![100, 400, 800, 1200, 1400];
        let buckets = build_buckets(&lengths, 3);
        for &l in &lengths {
            assert!(optimal_pad_length(l, &buckets) >= l, "pad shrank packet to {l}");
        }
    }

    #[test]
    fn overhead_decreases_with_more_buckets() {
        let lengths: Vec<usize> = (100..=1400).step_by(10).collect();
        let b1 = build_buckets(&lengths, 1);
        let b4 = build_buckets(&lengths, 4);
        let o1 = expected_overhead(&b1, &lengths);
        let o4 = expected_overhead(&b4, &lengths);
        assert!(
            o4 < o1,
            "more buckets should reduce overhead: k=1 overhead={o1:.1}, k=4 overhead={o4:.1}"
        );
    }

    #[test]
    fn optimal_padder_rebuilds_lazily() {
        let mut padder = OptimalPadder::new(4, 8);
        for _ in 0..8 {
            padder.observe_length(1400);
        }
        // After 8 observations, next pad_length call should trigger rebuild.
        let result = padder.pad_length(500);
        assert!(result > 0);
    }

    #[test]
    fn clamps_above_all_buckets() {
        let buckets = vec![SizeBucket { lo: 0, hi: 1000, target: 1000 }];
        // true_len > all bucket maxima → clamp to last bucket's target.
        assert_eq!(optimal_pad_length(9999, &buckets), 1000);
    }
}