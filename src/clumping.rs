use crate::seeds::{EphemeralSeed, SeedRole};
use rand::Rng;
use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};
use std::collections::VecDeque;
use std::fmt;
use std::time::{Duration, Instant};

/// Low-latency default bounds for packet clumping.
///
/// These values intentionally create microclumps: enough aggregation to destroy
/// one-packet-in/one-packet-out correspondence, short enough to avoid the long
/// stalls that make interactive traffic miserable.
pub const DEFAULT_MIN_PACKETS_PER_BURST: usize = 2;
pub const DEFAULT_MAX_PACKETS_PER_BURST: usize = 7;
pub const DEFAULT_MIN_TIMEOUT: Duration = Duration::from_millis(12);
pub const DEFAULT_MAX_TIMEOUT: Duration = Duration::from_millis(85);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Packet {
    bytes: Vec<u8>,
}

impl Packet {
    pub fn new(bytes: impl Into<Vec<u8>>) -> Result<Self, ClumpingError> {
        let bytes = bytes.into();
        if bytes.is_empty() {
            return Err(ClumpingError::EmptyPacket);
        }

        Ok(Self { bytes })
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Burst {
    packets: Vec<Packet>,
    threshold: CycleThresholds,
    reason: FlushReason,
}

impl Burst {
    #[cfg(test)]
    pub(crate) fn from_parts_for_tests(
        packets: Vec<Packet>,
        threshold: CycleThresholds,
        reason: FlushReason,
    ) -> Self {
        Self {
            packets,
            threshold,
            reason,
        }
    }

    pub fn packets(&self) -> &[Packet] {
        &self.packets
    }

    pub fn into_packets(self) -> Vec<Packet> {
        self.packets
    }

    pub fn packet_count(&self) -> usize {
        self.packets.len()
    }

    pub fn total_bytes(&self) -> usize {
        self.packets.iter().map(Packet::len).sum()
    }

    pub fn threshold(&self) -> CycleThresholds {
        self.threshold
    }

    pub fn reason(&self) -> FlushReason {
        self.reason
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CycleThresholds {
    pub max_packets: usize,
    pub timeout: Duration,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FlushReason {
    SizeThreshold,
    Timeout,
    Manual,
}

/// Bounds for randomized per-cycle pool thresholds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ClumpingConfig {
    pub min_packets_per_burst: usize,
    pub max_packets_per_burst: usize,
    pub min_timeout: Duration,
    pub max_timeout: Duration,
}

impl Default for ClumpingConfig {
    fn default() -> Self {
        Self {
            min_packets_per_burst: DEFAULT_MIN_PACKETS_PER_BURST,
            max_packets_per_burst: DEFAULT_MAX_PACKETS_PER_BURST,
            min_timeout: DEFAULT_MIN_TIMEOUT,
            max_timeout: DEFAULT_MAX_TIMEOUT,
        }
    }
}

impl ClumpingConfig {
    pub fn validate(self) -> Result<Self, ClumpingError> {
        if self.min_packets_per_burst == 0 {
            return Err(ClumpingError::InvalidConfig(
                "minimum packet threshold must be at least one",
            ));
        }

        if self.min_packets_per_burst > self.max_packets_per_burst {
            return Err(ClumpingError::InvalidConfig(
                "minimum packet threshold cannot exceed maximum packet threshold",
            ));
        }

        if self.min_timeout.is_zero() {
            return Err(ClumpingError::InvalidConfig(
                "minimum timeout must be non-zero",
            ));
        }

        if self.min_timeout > self.max_timeout {
            return Err(ClumpingError::InvalidConfig(
                "minimum timeout cannot exceed maximum timeout",
            ));
        }

        Ok(self)
    }
}

/// Seed A-driven packet clumping pool.
///
/// Each cycle has fresh randomized thresholds. A cycle ends when either the
/// packet threshold is reached or the randomized timeout expires after the
/// first packet entered the pool.
#[derive(Debug)]
pub struct ClumpingPool {
    config: ClumpingConfig,
    rng: ChaCha20Rng,
    buffered: VecDeque<Packet>,
    thresholds: CycleThresholds,
    deadline: Option<Instant>,
}

impl ClumpingPool {
    pub fn new(seed_a: &EphemeralSeed) -> Result<Self, ClumpingError> {
        Self::with_config(seed_a, ClumpingConfig::default())
    }

    pub fn with_config(
        seed_a: &EphemeralSeed,
        config: ClumpingConfig,
    ) -> Result<Self, ClumpingError> {
        if seed_a.role() != SeedRole::TimingAndPool {
            return Err(ClumpingError::WrongSeedRole {
                expected: SeedRole::TimingAndPool,
                actual: seed_a.role(),
            });
        }

        let config = config.validate()?;
        let mut rng = ChaCha20Rng::from_seed(seed_a.derive_rng_seed(b"mixtor.clumping.v1"));
        let thresholds = Self::next_thresholds(config, &mut rng);

        Ok(Self {
            config,
            rng,
            buffered: VecDeque::new(),
            thresholds,
            deadline: None,
        })
    }

    /// Adds one packet and returns every burst flushed by this operation.
    ///
    /// Bursty overflow is split cleanly across cycles. Because thresholds are
    /// regenerated after each flush, crafted probes cannot assume that the next
    /// cycle behaves like the previous one.
    pub fn push(&mut self, packet: Packet, now: Instant) -> Vec<Burst> {
        if self.buffered.is_empty() {
            self.deadline = Some(now + self.thresholds.timeout);
        }

        self.buffered.push_back(packet);

        let mut bursts = Vec::new();
        while self.buffered.len() >= self.thresholds.max_packets {
            bursts.push(self.flush_ready(FlushReason::SizeThreshold, now));

            if !self.buffered.is_empty() {
                self.deadline = Some(now + self.thresholds.timeout);
            }
        }

        bursts
    }

    /// Flushes a timed-out pool, including the single-packet timeout edge case.
    pub fn tick(&mut self, now: Instant) -> Option<Burst> {
        match self.deadline {
            Some(deadline) if now >= deadline && !self.buffered.is_empty() => {
                Some(self.flush_ready(FlushReason::Timeout, now))
            }
            _ => None,
        }
    }

    /// Drains currently buffered packets if any exist.
    pub fn flush(&mut self, now: Instant) -> Option<Burst> {
        if self.buffered.is_empty() {
            None
        } else {
            Some(self.flush_ready(FlushReason::Manual, now))
        }
    }

    pub fn pending_packets(&self) -> usize {
        self.buffered.len()
    }

    pub fn current_thresholds(&self) -> CycleThresholds {
        self.thresholds
    }

    pub fn deadline(&self) -> Option<Instant> {
        self.deadline
    }

    fn flush_ready(&mut self, reason: FlushReason, now: Instant) -> Burst {
        let threshold = self.thresholds;
        let packet_count = self.buffered.len().min(threshold.max_packets);
        let packets = self.buffered.drain(..packet_count).collect();

        self.thresholds = Self::next_thresholds(self.config, &mut self.rng);
        self.deadline = if self.buffered.is_empty() {
            None
        } else {
            Some(now + self.thresholds.timeout)
        };

        Burst {
            packets,
            threshold,
            reason,
        }
    }

    fn next_thresholds(config: ClumpingConfig, rng: &mut ChaCha20Rng) -> CycleThresholds {
        let max_packets =
            rng.gen_range(config.min_packets_per_burst..=config.max_packets_per_burst);
        let min_micros = config.min_timeout.as_micros() as u64;
        let max_micros = config.max_timeout.as_micros() as u64;
        let timeout = Duration::from_micros(rng.gen_range(min_micros..=max_micros));

        CycleThresholds {
            max_packets,
            timeout,
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum ClumpingError {
    EmptyPacket,
    InvalidConfig(&'static str),
    WrongSeedRole {
        expected: SeedRole,
        actual: SeedRole,
    },
}

impl fmt::Display for ClumpingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPacket => write!(formatter, "packet cannot be empty"),
            Self::InvalidConfig(message) => write!(formatter, "invalid clumping config: {message}"),
            Self::WrongSeedRole { expected, actual } => {
                write!(
                    formatter,
                    "wrong seed role: expected {expected:?}, got {actual:?}"
                )
            }
        }
    }
}

impl std::error::Error for ClumpingError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seeds::SessionSeeds;

    fn packet(byte: u8) -> Packet {
        Packet::new(vec![byte]).expect("packet should be valid")
    }

    fn fixed_config(max_packets: usize, timeout: Duration) -> ClumpingConfig {
        ClumpingConfig {
            min_packets_per_burst: max_packets,
            max_packets_per_burst: max_packets,
            min_timeout: timeout,
            max_timeout: timeout,
        }
    }

    #[test]
    fn creates_microclump_defaults() {
        let config = ClumpingConfig::default();

        assert_eq!(config.min_packets_per_burst, 2);
        assert_eq!(config.max_packets_per_burst, 7);
        assert!(config.max_timeout <= Duration::from_millis(100));
    }

    #[test]
    fn rejects_empty_packets() {
        let error = Packet::new(Vec::<u8>::new()).expect_err("empty packet should fail");

        assert_eq!(error, ClumpingError::EmptyPacket);
    }

    #[test]
    fn rejects_seed_b_for_clumping() {
        let seeds = SessionSeeds::generate().expect("seed generation should succeed");
        let error = ClumpingPool::new(seeds.burst_composition())
            .expect_err("Seed B should not initialize clumping");

        assert_eq!(
            error,
            ClumpingError::WrongSeedRole {
                expected: SeedRole::TimingAndPool,
                actual: SeedRole::BurstComposition,
            }
        );
    }

    #[test]
    fn flushes_when_packet_threshold_is_reached() {
        let seeds = SessionSeeds::generate().expect("seed generation should succeed");
        let mut pool = ClumpingPool::with_config(
            seeds.timing_and_pool(),
            fixed_config(3, Duration::from_secs(1)),
        )
        .expect("pool should initialize");
        let now = Instant::now();

        assert!(pool.push(packet(1), now).is_empty());
        assert!(pool.push(packet(2), now).is_empty());

        let bursts = pool.push(packet(3), now);

        assert_eq!(bursts.len(), 1);
        assert_eq!(bursts[0].packet_count(), 3);
        assert_eq!(bursts[0].reason(), FlushReason::SizeThreshold);
        assert_eq!(pool.pending_packets(), 0);
    }

    #[test]
    fn single_packet_flushes_on_timeout() {
        let seeds = SessionSeeds::generate().expect("seed generation should succeed");
        let mut pool = ClumpingPool::with_config(
            seeds.timing_and_pool(),
            fixed_config(4, Duration::from_millis(25)),
        )
        .expect("pool should initialize");
        let now = Instant::now();

        assert!(pool.push(packet(1), now).is_empty());
        assert!(pool.tick(now + Duration::from_millis(24)).is_none());

        let burst = pool
            .tick(now + Duration::from_millis(25))
            .expect("packet should flush at timeout");

        assert_eq!(burst.packet_count(), 1);
        assert_eq!(burst.reason(), FlushReason::Timeout);
        assert_eq!(pool.pending_packets(), 0);
    }

    #[test]
    fn bursty_overflow_splits_across_bursts_without_loss() {
        let seeds = SessionSeeds::generate().expect("seed generation should succeed");
        let mut pool = ClumpingPool::with_config(
            seeds.timing_and_pool(),
            fixed_config(2, Duration::from_secs(1)),
        )
        .expect("pool should initialize");
        let now = Instant::now();
        let mut bursts = Vec::new();

        for byte in 0..5 {
            bursts.extend(pool.push(packet(byte), now));
        }

        assert_eq!(bursts.len(), 2);
        assert_eq!(bursts[0].packet_count(), 2);
        assert_eq!(bursts[1].packet_count(), 2);
        assert_eq!(pool.pending_packets(), 1);

        let tail = pool.flush(now).expect("tail packet should flush manually");
        assert_eq!(tail.packet_count(), 1);
    }

    #[test]
    fn thresholds_are_regenerated_each_cycle() {
        let seeds = SessionSeeds::generate().expect("seed generation should succeed");
        let mut pool = ClumpingPool::new(seeds.timing_and_pool()).expect("pool should initialize");
        let now = Instant::now();
        let mut observed = Vec::new();

        for i in 0..64 {
            observed.push(pool.current_thresholds());
            let threshold = pool.current_thresholds().max_packets;
            for j in 0..threshold {
                let _ = pool.push(packet((i + j) as u8), now);
            }
        }

        observed.dedup();
        assert!(
            observed.len() > 1,
            "thresholds should vary across cycles to resist probing"
        );
    }

    #[test]
    fn invalid_configs_are_rejected() {
        let error = ClumpingConfig {
            min_packets_per_burst: 8,
            max_packets_per_burst: 2,
            min_timeout: Duration::from_millis(1),
            max_timeout: Duration::from_millis(2),
        }
        .validate()
        .expect_err("inverted packet range should fail");

        assert!(matches!(error, ClumpingError::InvalidConfig(_)));
    }
}
