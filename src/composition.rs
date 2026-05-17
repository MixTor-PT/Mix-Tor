use crate::clumping::{Burst, Packet};
use crate::seeds::{EphemeralSeed, SeedRole};
use rand::seq::IteratorRandom;
use rand::Rng;
use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};
use std::fmt;

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RatioDistributionKind {
    Sparse,
    Balanced,
    HeavyCover,
    Spiky,
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
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompositionConfig {
    pub max_dummy_packets_per_burst: usize,
    pub min_switch_bursts: u64,
    pub max_switch_bursts: u64,
    pub min_switch_real_bytes: u64,
    pub max_switch_real_bytes: u64,
    pub min_switch_real_packets: u64,
    pub max_switch_real_packets: u64,
}

impl Default for CompositionConfig {
    fn default() -> Self {
        Self {
            max_dummy_packets_per_burst: 18,
            min_switch_bursts: 24,
            max_switch_bursts: 256,
            min_switch_real_bytes: 32 * 1024,
            max_switch_real_bytes: 768 * 1024,
            min_switch_real_packets: 64,
            max_switch_real_packets: 2048,
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

        Ok(self)
    }
}

#[derive(Debug)]
pub struct SeedBComposer {
    config: CompositionConfig,
    rng: ChaCha20Rng,
    regime: RatioRegime,
    bursts_in_regime: u64,
    real_bytes_in_regime: u64,
    real_packets_in_regime: u64,
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
        })
    }

    pub fn compose(&mut self, burst: Burst) -> MixedBurst {
        let real_packets = burst.packet_count();
        let real_bytes = burst.total_bytes();
        let dummy_packets = self.dummy_count(real_packets);
        let real_lengths: Vec<usize> = burst.packets().iter().map(Packet::len).collect();
        let frames = self.interleave_preserving_real_order(
            burst.into_packets(),
            dummy_packets,
            &real_lengths,
        );

        let mixed = MixedBurst {
            frames,
            ratio_regime: self.regime,
            real_packets,
            dummy_packets,
        };

        self.observe(real_bytes as u64, real_packets as u64);
        mixed
    }

    pub fn current_regime(&self) -> RatioRegime {
        self.regime
    }

    fn observe(&mut self, real_bytes: u64, real_packets: u64) {
        self.bursts_in_regime = self.bursts_in_regime.saturating_add(1);
        self.real_bytes_in_regime = self.real_bytes_in_regime.saturating_add(real_bytes);
        self.real_packets_in_regime = self.real_packets_in_regime.saturating_add(real_packets);

        if self.bursts_in_regime >= self.regime.switch_after.bursts
            || self.real_bytes_in_regime >= self.regime.switch_after.real_bytes
            || self.real_packets_in_regime >= self.regime.switch_after.real_packets
        {
            self.regime = Self::next_regime(self.config, &mut self.rng);
            self.bursts_in_regime = 0;
            self.real_bytes_in_regime = 0;
            self.real_packets_in_regime = 0;
        }
    }

    fn dummy_count(&mut self, real_packets: usize) -> usize {
        let ratio = match self.regime.kind {
            RatioDistributionKind::Sparse => self.rng.gen_range(0.15..=0.75),
            RatioDistributionKind::Balanced => self.rng.gen_range(0.65..=1.60),
            RatioDistributionKind::HeavyCover => self.rng.gen_range(1.25..=3.00),
            RatioDistributionKind::Spiky => {
                if self.rng.gen_bool(0.74) {
                    self.rng.gen_range(0.05..=0.70)
                } else {
                    self.rng.gen_range(2.00..=4.25)
                }
            }
        };

        ((real_packets as f64 * ratio).round() as usize)
            .max(1)
            .min(self.config.max_dummy_packets_per_burst)
    }

    fn dummy_payload_from_lengths(&mut self, real_lengths: &[usize]) -> Vec<u8> {
        let len = real_lengths
            .iter()
            .choose(&mut self.rng)
            .copied()
            .unwrap_or_else(|| self.rng.gen_range(64..=1200));
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

        RatioRegime {
            kind,
            switch_after: RatioSwitchThreshold {
                bursts: rng.gen_range(config.min_switch_bursts..=config.max_switch_bursts),
                real_bytes: rng
                    .gen_range(config.min_switch_real_bytes..=config.max_switch_real_bytes),
                real_packets: rng
                    .gen_range(config.min_switch_real_packets..=config.max_switch_real_packets),
            },
        }
    }
}

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

    fn fast_switch_config() -> CompositionConfig {
        CompositionConfig {
            max_dummy_packets_per_burst: 12,
            min_switch_bursts: 1,
            max_switch_bursts: 2,
            min_switch_real_bytes: 1,
            max_switch_real_bytes: 2,
            min_switch_real_packets: 1,
            max_switch_real_packets: 2,
        }
    }

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
        assert!(mixed.dummy_packets() >= 1);
        assert_eq!(
            mixed.total_packets(),
            mixed.real_packets() + mixed.dummy_packets()
        );
        assert_eq!(
            mixed
                .frames()
                .iter()
                .filter(|frame| frame.is_real())
                .count(),
            2
        );
        assert_eq!(
            mixed
                .frames()
                .iter()
                .filter(|frame| frame.is_dummy())
                .count(),
            mixed.dummy_packets()
        );
    }

    #[test]
    fn dummy_sizes_match_real_packet_sizes() {
        let seeds = SessionSeeds::generate().expect("seed generation should succeed");
        let mut composer =
            SeedBComposer::new(seeds.burst_composition()).expect("composer should initialize");
        let mixed = composer.compose(burst(vec![packet(1, 41), packet(2, 97)]));

        for frame in mixed.frames().iter().filter(|frame| frame.is_dummy()) {
            assert!(frame.len() == 41 || frame.len() == 97);
        }
    }

    #[test]
    fn real_packet_order_is_preserved() {
        let seeds = SessionSeeds::generate().expect("seed generation should succeed");
        let mut composer =
            SeedBComposer::new(seeds.burst_composition()).expect("composer should initialize");
        let mixed = composer.compose(burst(vec![packet(1, 41), packet(2, 97), packet(3, 123)]));
        let real_first_bytes: Vec<u8> = mixed
            .frames()
            .iter()
            .filter_map(|frame| match frame {
                MixedFrame::Real(packet) => Some(packet.bytes()[0]),
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

        kinds.sort_by_key(|kind| *kind as u8);
        kinds.dedup();
        assert_eq!(kinds.len(), 4);
    }
}
