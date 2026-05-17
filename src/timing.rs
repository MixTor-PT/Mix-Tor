use crate::clumping::{ClumpingConfig, CycleThresholds};
use crate::seeds::{EphemeralSeed, SeedRole};
use rand::Rng;
use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};
use rand_distr::{Distribution, Exp, Pareto, Poisson};
use std::fmt;
use std::time::Duration;

const MIN_DELAY_MS: u64 = 4;
const MAX_DELAY_MS: u64 = 240;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TimingDistributionKind {
    Poisson,
    Pareto,
    Bimodal,
    BurstAndSilence,
    Uniform,
    Exponential,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegimeSwitchThreshold {
    pub bytes: u64,
    pub packets: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TimingRegime {
    pub kind: TimingDistributionKind,
    pub switch_after: RegimeSwitchThreshold,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TimingDecision {
    pub delay: Duration,
    pub regime: TimingRegime,
    pub pool_thresholds: CycleThresholds,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TimingConfig {
    pub min_delay: Duration,
    pub max_delay: Duration,
    pub min_regime_bytes: u64,
    pub max_regime_bytes: u64,
    pub min_regime_packets: u64,
    pub max_regime_packets: u64,
    pub clumping: ClumpingConfig,
}

impl Default for TimingConfig {
    fn default() -> Self {
        Self {
            min_delay: Duration::from_millis(MIN_DELAY_MS),
            max_delay: Duration::from_millis(MAX_DELAY_MS),
            min_regime_bytes: 24 * 1024,
            max_regime_bytes: 512 * 1024,
            min_regime_packets: 48,
            max_regime_packets: 1024,
            clumping: ClumpingConfig::default(),
        }
    }
}

impl TimingConfig {
    pub fn validate(self) -> Result<Self, TimingError> {
        if self.min_delay.is_zero() {
            return Err(TimingError::InvalidConfig("minimum delay must be non-zero"));
        }

        if self.min_delay > self.max_delay {
            return Err(TimingError::InvalidConfig(
                "minimum delay cannot exceed maximum delay",
            ));
        }

        if self.min_regime_bytes == 0 || self.min_regime_packets == 0 {
            return Err(TimingError::InvalidConfig(
                "regime switch thresholds must be non-zero",
            ));
        }

        if self.min_regime_bytes > self.max_regime_bytes {
            return Err(TimingError::InvalidConfig(
                "minimum regime bytes cannot exceed maximum regime bytes",
            ));
        }

        if self.min_regime_packets > self.max_regime_packets {
            return Err(TimingError::InvalidConfig(
                "minimum regime packets cannot exceed maximum regime packets",
            ));
        }

        self.clumping.validate()?;
        Ok(self)
    }
}

#[derive(Debug)]
pub struct SeedATimingScheduler {
    config: TimingConfig,
    rng: ChaCha20Rng,
    regime: TimingRegime,
    bytes_in_regime: u64,
    packets_in_regime: u64,
}

impl SeedATimingScheduler {
    pub fn new(seed_a: &EphemeralSeed) -> Result<Self, TimingError> {
        Self::with_config(seed_a, TimingConfig::default())
    }

    pub fn with_config(seed_a: &EphemeralSeed, config: TimingConfig) -> Result<Self, TimingError> {
        if seed_a.role() != SeedRole::TimingAndPool {
            return Err(TimingError::WrongSeedRole {
                expected: SeedRole::TimingAndPool,
                actual: seed_a.role(),
            });
        }

        let config = config.validate()?;
        let mut rng = ChaCha20Rng::from_seed(seed_a.derive_rng_seed(b"mixtor.timing.v1"));
        let regime = Self::next_regime(config, &mut rng);

        Ok(Self {
            config,
            rng,
            regime,
            bytes_in_regime: 0,
            packets_in_regime: 0,
        })
    }

    pub fn observe_burst(&mut self, bytes: u64, packets: u64) {
        self.bytes_in_regime = self.bytes_in_regime.saturating_add(bytes);
        self.packets_in_regime = self.packets_in_regime.saturating_add(packets);

        if self.bytes_in_regime >= self.regime.switch_after.bytes
            || self.packets_in_regime >= self.regime.switch_after.packets
        {
            self.regime = Self::next_regime(self.config, &mut self.rng);
            self.bytes_in_regime = 0;
            self.packets_in_regime = 0;
        }
    }

    pub fn next_decision(&mut self) -> TimingDecision {
        TimingDecision {
            delay: self.next_delay(),
            regime: self.regime,
            pool_thresholds: self.next_pool_thresholds(),
        }
    }

    pub fn current_regime(&self) -> TimingRegime {
        self.regime
    }

    fn next_delay(&mut self) -> Duration {
        let min_ms = self.config.min_delay.as_millis() as f64;
        let max_ms = self.config.max_delay.as_millis() as f64;
        let sampled_ms = match self.regime.kind {
            TimingDistributionKind::Poisson => {
                let lambda = self.rng.gen_range(min_ms..=max_ms).max(1.0);
                Poisson::new(lambda)
                    .expect("lambda is positive")
                    .sample(&mut self.rng)
            }
            TimingDistributionKind::Pareto => {
                let scale = self.rng.gen_range(min_ms..=(max_ms / 3.0).max(min_ms));
                let shape = self.rng.gen_range(1.2..=2.8);
                Pareto::new(scale, shape)
                    .expect("pareto parameters are positive")
                    .sample(&mut self.rng)
            }
            TimingDistributionKind::Bimodal => {
                if self.rng.gen_bool(0.68) {
                    self.rng.gen_range(min_ms..=(max_ms * 0.30).max(min_ms))
                } else {
                    self.rng.gen_range((max_ms * 0.55).max(min_ms)..=max_ms)
                }
            }
            TimingDistributionKind::BurstAndSilence => {
                if self.rng.gen_bool(0.82) {
                    self.rng.gen_range(min_ms..=(max_ms * 0.18).max(min_ms))
                } else {
                    self.rng.gen_range((max_ms * 0.60).max(min_ms)..=max_ms)
                }
            }
            TimingDistributionKind::Uniform => self.rng.gen_range(min_ms..=max_ms),
            TimingDistributionKind::Exponential => {
                let mean = self.rng.gen_range(min_ms..=max_ms).max(1.0);
                Exp::new(1.0 / mean)
                    .expect("exponential rate is positive")
                    .sample(&mut self.rng)
            }
        };

        Duration::from_millis(sampled_ms.round().clamp(min_ms, max_ms) as u64)
    }

    fn next_pool_thresholds(&mut self) -> CycleThresholds {
        let config = self.config.clumping;
        let max_packets = self
            .rng
            .gen_range(config.min_packets_per_burst..=config.max_packets_per_burst);
        let min_micros = config.min_timeout.as_micros() as u64;
        let max_micros = config.max_timeout.as_micros() as u64;
        let timeout = Duration::from_micros(self.rng.gen_range(min_micros..=max_micros));

        CycleThresholds {
            max_packets,
            timeout,
        }
    }

    fn next_regime(config: TimingConfig, rng: &mut ChaCha20Rng) -> TimingRegime {
        let kind = match rng.gen_range(0..6) {
            0 => TimingDistributionKind::Poisson,
            1 => TimingDistributionKind::Pareto,
            2 => TimingDistributionKind::Bimodal,
            3 => TimingDistributionKind::BurstAndSilence,
            4 => TimingDistributionKind::Uniform,
            _ => TimingDistributionKind::Exponential,
        };

        TimingRegime {
            kind,
            switch_after: RegimeSwitchThreshold {
                bytes: rng.gen_range(config.min_regime_bytes..=config.max_regime_bytes),
                packets: rng.gen_range(config.min_regime_packets..=config.max_regime_packets),
            },
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum TimingError {
    InvalidConfig(&'static str),
    InvalidClumpingConfig(crate::clumping::ClumpingError),
    WrongSeedRole {
        expected: SeedRole,
        actual: SeedRole,
    },
}

impl From<crate::clumping::ClumpingError> for TimingError {
    fn from(error: crate::clumping::ClumpingError) -> Self {
        Self::InvalidClumpingConfig(error)
    }
}

impl fmt::Display for TimingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(formatter, "invalid timing config: {message}"),
            Self::InvalidClumpingConfig(error) => write!(formatter, "{error}"),
            Self::WrongSeedRole { expected, actual } => {
                write!(
                    formatter,
                    "wrong seed role: expected {expected:?}, got {actual:?}"
                )
            }
        }
    }
}

impl std::error::Error for TimingError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seeds::SessionSeeds;

    fn fast_switch_config() -> TimingConfig {
        TimingConfig {
            min_delay: Duration::from_millis(2),
            max_delay: Duration::from_millis(20),
            min_regime_bytes: 1,
            max_regime_bytes: 2,
            min_regime_packets: 1,
            max_regime_packets: 2,
            clumping: ClumpingConfig::default(),
        }
    }

    #[test]
    fn rejects_seed_b() {
        let seeds = SessionSeeds::generate().expect("seed generation should succeed");
        let error = SeedATimingScheduler::new(seeds.burst_composition())
            .expect_err("Seed B should not initialize timing");

        assert_eq!(
            error,
            TimingError::WrongSeedRole {
                expected: SeedRole::TimingAndPool,
                actual: SeedRole::BurstComposition,
            }
        );
    }

    #[test]
    fn delays_stay_inside_configured_bounds() {
        let seeds = SessionSeeds::generate().expect("seed generation should succeed");
        let mut scheduler = SeedATimingScheduler::with_config(
            seeds.timing_and_pool(),
            TimingConfig {
                min_delay: Duration::from_millis(5),
                max_delay: Duration::from_millis(40),
                ..TimingConfig::default()
            },
        )
        .expect("scheduler should initialize");

        for _ in 0..128 {
            let decision = scheduler.next_decision();
            assert!(decision.delay >= Duration::from_millis(5));
            assert!(decision.delay <= Duration::from_millis(40));
        }
    }

    #[test]
    fn switches_regime_from_traffic_counters_not_time() {
        let seeds = SessionSeeds::generate().expect("seed generation should succeed");
        let mut scheduler =
            SeedATimingScheduler::with_config(seeds.timing_and_pool(), fast_switch_config())
                .expect("scheduler should initialize");

        scheduler.observe_burst(2, 2);

        assert_eq!(scheduler.bytes_in_regime, 0);
        assert_eq!(scheduler.packets_in_regime, 0);
    }

    #[test]
    fn emits_all_distribution_families_over_many_regimes() {
        let seeds = SessionSeeds::generate().expect("seed generation should succeed");
        let mut scheduler =
            SeedATimingScheduler::with_config(seeds.timing_and_pool(), fast_switch_config())
                .expect("scheduler should initialize");
        let mut kinds = Vec::new();

        for _ in 0..256 {
            kinds.push(scheduler.current_regime().kind);
            scheduler.observe_burst(2, 2);
        }

        kinds.sort_by_key(|kind| *kind as u8);
        kinds.dedup();
        assert_eq!(kinds.len(), 6);
    }

    #[test]
    fn pool_thresholds_vary_per_cycle() {
        let seeds = SessionSeeds::generate().expect("seed generation should succeed");
        let mut scheduler = SeedATimingScheduler::new(seeds.timing_and_pool())
            .expect("scheduler should initialize");
        let mut thresholds = Vec::new();

        for _ in 0..64 {
            thresholds.push(scheduler.next_decision().pool_thresholds);
        }

        thresholds.dedup();
        assert!(thresholds.len() > 1);
    }
}
