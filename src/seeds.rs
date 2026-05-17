use getrandom::getrandom;
use std::fmt;
use std::time::{Duration, Instant};
use zeroize::Zeroize;

const SEED_LEN: usize = 32;
const DEFAULT_ROTATION_INTERVAL: Duration = Duration::from_secs(24 * 60 * 60);

/// A 256-bit seed used by one dimension of the MixTor engine.
///
/// This type is intentionally not `Clone`, `Copy`, or serializable. It owns the
/// seed bytes and zeroizes them when dropped.
pub struct EphemeralSeed {
    role: SeedRole,
    bytes: [u8; SEED_LEN],
}

impl EphemeralSeed {
    fn generate(role: SeedRole) -> Result<Self, SeedError> {
        let mut bytes = [0u8; SEED_LEN];
        getrandom(&mut bytes).map_err(SeedError::Random)?;

        if bytes.iter().all(|byte| *byte == 0) {
            return Err(SeedError::AllZeroSeed { role });
        }

        Ok(Self { role, bytes })
    }

    /// Provides temporary read-only access to seed material for in-crate engine
    /// initialization without making the seed clonable or printable.
    pub(crate) fn with_bytes<R>(&self, read: impl FnOnce(&[u8; SEED_LEN]) -> R) -> R {
        read(&self.bytes)
    }

    pub(crate) fn derive_rng_seed(&self, domain: &[u8]) -> [u8; SEED_LEN] {
        blake3::keyed_hash(&self.bytes, domain).into()
    }

    pub fn role(&self) -> SeedRole {
        self.role
    }
}

impl Drop for EphemeralSeed {
    fn drop(&mut self) {
        self.bytes.zeroize();
    }
}

impl fmt::Debug for EphemeralSeed {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EphemeralSeed")
            .field("role", &self.role)
            .field("bytes", &"[redacted; zeroized on drop]")
            .finish()
    }
}

/// The two independent seed roles in the dual-seed engine.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SeedRole {
    /// Seed A: inter-burst timing and clumping pool parameters.
    TimingAndPool,
    /// Seed B: burst composition and dummy-ratio distribution.
    BurstComposition,
}

/// Process-local seed pair generated at startup.
///
/// Construct this once during process initialization and keep it in memory only
/// for the lifetime of the running process. Dropping this value zeroizes both
/// seeds.
pub struct SessionSeeds {
    timing_and_pool: EphemeralSeed,
    burst_composition: EphemeralSeed,
    generated_at: Instant,
    rotation_interval: Duration,
}

impl SessionSeeds {
    /// Generates two independent 256-bit seeds from the operating system CSPRNG.
    pub fn generate() -> Result<Self, SeedError> {
        Self::generate_with_rotation_interval(DEFAULT_ROTATION_INTERVAL)
    }

    /// Generates seed material with an explicit maximum lifetime.
    ///
    /// Daily rotation is the default project policy, but tests and future
    /// operators can use a shorter interval. Callers should add jitter around
    /// the configured interval before invoking rotation logic so rekeys do not
    /// create a crisp wall-clock fingerprint.
    pub fn generate_with_rotation_interval(rotation_interval: Duration) -> Result<Self, SeedError> {
        if rotation_interval.is_zero() {
            return Err(SeedError::InvalidRotationInterval);
        }

        let timing_and_pool = EphemeralSeed::generate(SeedRole::TimingAndPool)?;
        let burst_composition = EphemeralSeed::generate(SeedRole::BurstComposition)?;

        if timing_and_pool.with_bytes(|a| burst_composition.with_bytes(|b| a == b)) {
            return Err(SeedError::DuplicateSeeds);
        }

        Ok(Self {
            timing_and_pool,
            burst_composition,
            generated_at: Instant::now(),
            rotation_interval,
        })
    }

    pub fn timing_and_pool(&self) -> &EphemeralSeed {
        &self.timing_and_pool
    }

    pub fn burst_composition(&self) -> &EphemeralSeed {
        &self.burst_composition
    }

    /// Returns true once this seed pair has reached its maximum configured age.
    pub fn should_rotate(&self) -> bool {
        self.generated_at.elapsed() >= self.rotation_interval
    }

    /// Replaces both seeds with a fresh independent pair and zeroizes the old
    /// material as it is dropped.
    pub fn rotate(&mut self) -> Result<(), SeedError> {
        let next = Self::generate_with_rotation_interval(self.rotation_interval)?;
        *self = next;
        Ok(())
    }

    pub fn rotation_interval(&self) -> Duration {
        self.rotation_interval
    }
}

impl fmt::Debug for SessionSeeds {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SessionSeeds")
            .field("timing_and_pool", &self.timing_and_pool)
            .field("burst_composition", &self.burst_composition)
            .field("rotation_interval", &self.rotation_interval)
            .finish()
    }
}

#[derive(Debug)]
pub enum SeedError {
    Random(getrandom::Error),
    AllZeroSeed { role: SeedRole },
    DuplicateSeeds,
    InvalidRotationInterval,
}

impl fmt::Display for SeedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Random(error) => write!(formatter, "failed to read OS CSPRNG: {error}"),
            Self::AllZeroSeed { role } => write!(formatter, "generated all-zero seed for {role:?}"),
            Self::DuplicateSeeds => write!(formatter, "generated duplicate dual-seed material"),
            Self::InvalidRotationInterval => {
                write!(formatter, "seed rotation interval must be non-zero")
            }
        }
    }
}

impl std::error::Error for SeedError {}

#[cfg(test)]
mod tests {
    use super::*;
    use static_assertions::assert_not_impl_any;

    assert_not_impl_any!(EphemeralSeed: Clone, Copy);
    assert_not_impl_any!(SessionSeeds: Clone, Copy);

    #[test]
    fn generates_two_256_bit_seeds() {
        let seeds = SessionSeeds::generate().expect("seed generation should succeed");

        seeds
            .timing_and_pool()
            .with_bytes(|seed| assert_eq!(seed.len(), SEED_LEN));
        seeds
            .burst_composition()
            .with_bytes(|seed| assert_eq!(seed.len(), SEED_LEN));
    }

    #[test]
    fn seeds_have_distinct_roles() {
        let seeds = SessionSeeds::generate().expect("seed generation should succeed");

        assert_eq!(seeds.timing_and_pool().role(), SeedRole::TimingAndPool);
        assert_eq!(seeds.burst_composition().role(), SeedRole::BurstComposition);
    }

    #[test]
    fn seeds_are_independent_within_a_session() {
        let seeds = SessionSeeds::generate().expect("seed generation should succeed");

        let are_equal = seeds
            .timing_and_pool()
            .with_bytes(|a| seeds.burst_composition().with_bytes(|b| a == b));

        assert!(!are_equal);
    }

    #[test]
    fn sessions_do_not_reuse_seed_material() {
        let first = SessionSeeds::generate().expect("first seed generation should succeed");
        let second = SessionSeeds::generate().expect("second seed generation should succeed");

        let timing_reused = first
            .timing_and_pool()
            .with_bytes(|a| second.timing_and_pool().with_bytes(|b| a == b));
        let burst_reused = first
            .burst_composition()
            .with_bytes(|a| second.burst_composition().with_bytes(|b| a == b));

        assert!(!timing_reused);
        assert!(!burst_reused);
    }

    #[test]
    fn debug_output_redacts_seed_bytes() {
        let seeds = SessionSeeds::generate().expect("seed generation should succeed");
        let output = format!("{seeds:?}");

        assert!(output.contains("[redacted; zeroized on drop]"));
        assert!(!output.contains("bytes: ["));
    }

    #[test]
    fn default_rotation_interval_is_daily() {
        let seeds = SessionSeeds::generate().expect("seed generation should succeed");

        assert_eq!(seeds.rotation_interval(), DEFAULT_ROTATION_INTERVAL);
    }

    #[test]
    fn rotation_replaces_both_seeds() {
        let mut seeds = SessionSeeds::generate().expect("seed generation should succeed");

        let old_timing = seeds.timing_and_pool().with_bytes(|seed| *seed);
        let old_burst = seeds.burst_composition().with_bytes(|seed| *seed);

        seeds.rotate().expect("seed rotation should succeed");

        let timing_reused = seeds
            .timing_and_pool()
            .with_bytes(|seed| *seed == old_timing);
        let burst_reused = seeds
            .burst_composition()
            .with_bytes(|seed| *seed == old_burst);

        assert!(!timing_reused);
        assert!(!burst_reused);
    }

    #[test]
    fn zero_rotation_interval_is_rejected() {
        let error = SessionSeeds::generate_with_rotation_interval(Duration::ZERO)
            .expect_err("zero rotation interval should fail");

        assert!(matches!(error, SeedError::InvalidRotationInterval));
    }
}
