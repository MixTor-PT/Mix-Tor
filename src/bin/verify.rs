use mixtor::clumping::{ClumpingConfig, ClumpingPool, Packet};
use mixtor::composition::{MixedFrame, SeedBComposer};
use mixtor::seeds::SessionSeeds;
use mixtor::timing::{SeedATimingScheduler, TimingConfig};
use std::error::Error;
use std::time::{Duration, Instant};

const PACKETS_PER_SESSION: usize = 6_000;

fn main() -> Result<(), Box<dyn Error>> {
    let first = simulate_session(PACKETS_PER_SESSION)?;
    let second = simulate_session(PACKETS_PER_SESSION)?;

    let delay_corr = pearson(&first.delays_ms, &second.delays_ms);
    let ratio_corr = pearson(&first.dummy_ratios, &second.dummy_ratios);
    let real_to_mixed_packet_corr = pearson(&first.real_packet_counts, &first.mixed_packet_counts);
    let real_to_mixed_byte_corr = pearson(&first.real_byte_counts, &first.mixed_byte_counts);
    let ratio_window_range = window_mean_range(&first.dummy_ratios, 64);

    let results = [
        Check {
            name: "non-stationarity across sessions: delay correlation",
            value: delay_corr.abs(),
            limit: 0.35,
            passes_when_below: true,
        },
        Check {
            name: "non-stationarity across sessions: dummy-ratio correlation",
            value: ratio_corr.abs(),
            limit: 0.35,
            passes_when_below: true,
        },
        Check {
            name: "burst packet-count decoupling",
            value: real_to_mixed_packet_corr.abs(),
            limit: 0.92,
            passes_when_below: true,
        },
        Check {
            name: "burst byte-volume decoupling",
            value: real_to_mixed_byte_corr.abs(),
            limit: 0.96,
            passes_when_below: true,
        },
        Check {
            name: "dummy-ratio window drift",
            value: ratio_window_range,
            limit: 0.35,
            passes_when_below: false,
        },
        Check {
            name: "packet-order prefix leakage rate",
            value: first.real_prefix_rate,
            limit: 0.35,
            passes_when_below: true,
        },
    ];

    println!("MixTor verification smoke test");
    println!("sessions simulated: 2");
    println!("packets per session: {PACKETS_PER_SESSION}");
    println!("bursts in first session: {}", first.bursts);
    println!();

    let mut failed = false;
    for check in results {
        let passed = check.passed();
        failed |= !passed;
        println!(
            "{:<55} {:>8.3}  limit {:>5.2}  {}",
            check.name,
            check.value,
            check.limit,
            if passed { "PASS" } else { "FAIL" }
        );
    }

    println!();
    println!("Notes:");
    println!("- These are statistical smoke tests, not a proof of anonymity.");
    println!("- Re-run after changing scheduler, clumping, or composition parameters.");
    println!("- Expect small variation between runs; persistent failures mean the engine is too stationary or too correlated.");

    if failed {
        Err("verification failed".into())
    } else {
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct SessionSample {
    delays_ms: Vec<f64>,
    dummy_ratios: Vec<f64>,
    real_packet_counts: Vec<f64>,
    mixed_packet_counts: Vec<f64>,
    real_byte_counts: Vec<f64>,
    mixed_byte_counts: Vec<f64>,
    real_prefix_rate: f64,
    bursts: usize,
}

#[derive(Clone, Copy, Debug)]
struct Check {
    name: &'static str,
    value: f64,
    limit: f64,
    passes_when_below: bool,
}

impl Check {
    fn passed(self) -> bool {
        if self.passes_when_below {
            self.value < self.limit
        } else {
            self.value > self.limit
        }
    }
}

fn simulate_session(packet_count: usize) -> Result<SessionSample, Box<dyn Error>> {
    let seeds = SessionSeeds::generate()?;
    let clumping = ClumpingConfig {
        min_packets_per_burst: 2,
        max_packets_per_burst: 7,
        min_timeout: Duration::from_millis(12),
        max_timeout: Duration::from_millis(85),
    };
    let timing_config = TimingConfig {
        min_delay: Duration::from_millis(4),
        max_delay: Duration::from_millis(180),
        min_regime_bytes: 4 * 1024,
        max_regime_bytes: 48 * 1024,
        min_regime_packets: 16,
        max_regime_packets: 128,
        clumping,
    };

    let mut pool = ClumpingPool::with_config(seeds.timing_and_pool(), clumping)?;
    let mut timing = SeedATimingScheduler::with_config(seeds.timing_and_pool(), timing_config)?;
    let mut composer = SeedBComposer::new(seeds.burst_composition())?;
    let now = Instant::now();

    let mut delays_ms = Vec::new();
    let mut dummy_ratios = Vec::new();
    let mut real_packet_counts = Vec::new();
    let mut mixed_packet_counts = Vec::new();
    let mut real_byte_counts = Vec::new();
    let mut mixed_byte_counts = Vec::new();
    let mut prefix_leaks = 0usize;
    let mut bursts = 0usize;

    for index in 0..packet_count {
        for burst in pool.push(synthetic_packet(index)?, now) {
            let decision = timing.next_decision();
            timing.observe_burst(burst.total_bytes() as u64, burst.packet_count() as u64);

            let mixed = composer.compose(burst);
            delays_ms.push(decision.delay.as_millis() as f64);
            dummy_ratios.push(mixed.dummy_packets() as f64 / mixed.real_packets() as f64);
            real_packet_counts.push(mixed.real_packets() as f64);
            mixed_packet_counts.push(mixed.total_packets() as f64);

            let real_bytes: usize = mixed
                .frames()
                .iter()
                .filter_map(|frame| match frame {
                    MixedFrame::Real(packet) => Some(packet.len()),
                    MixedFrame::Dummy(_) => None,
                })
                .sum();
            real_byte_counts.push(real_bytes as f64);
            mixed_byte_counts.push(mixed.total_bytes() as f64);

            if real_frames_are_prefix(mixed.frames()) {
                prefix_leaks += 1;
            }
            bursts += 1;
        }
    }

    if let Some(burst) = pool.flush(now) {
        let decision = timing.next_decision();
        timing.observe_burst(burst.total_bytes() as u64, burst.packet_count() as u64);

        let mixed = composer.compose(burst);
        delays_ms.push(decision.delay.as_millis() as f64);
        dummy_ratios.push(mixed.dummy_packets() as f64 / mixed.real_packets() as f64);
        real_packet_counts.push(mixed.real_packets() as f64);
        mixed_packet_counts.push(mixed.total_packets() as f64);

        let real_bytes: usize = mixed
            .frames()
            .iter()
            .filter_map(|frame| match frame {
                MixedFrame::Real(packet) => Some(packet.len()),
                MixedFrame::Dummy(_) => None,
            })
            .sum();
        real_byte_counts.push(real_bytes as f64);
        mixed_byte_counts.push(mixed.total_bytes() as f64);

        if real_frames_are_prefix(mixed.frames()) {
            prefix_leaks += 1;
        }
        bursts += 1;
    }

    Ok(SessionSample {
        delays_ms,
        dummy_ratios,
        real_packet_counts,
        mixed_packet_counts,
        real_byte_counts,
        mixed_byte_counts,
        real_prefix_rate: prefix_leaks as f64 / bursts.max(1) as f64,
        bursts,
    })
}

fn synthetic_packet(index: usize) -> Result<Packet, Box<dyn Error>> {
    let patterned = [72, 96, 128, 256, 384, 640, 900, 1180];
    let jitter = index.wrapping_mul(37) % 61;
    let len = patterned[index % patterned.len()] + jitter;
    let byte = (index % 251) as u8;
    Ok(Packet::new(vec![byte; len])?)
}

fn real_frames_are_prefix(frames: &[MixedFrame]) -> bool {
    let mut saw_dummy = false;
    let mut real_count = 0usize;

    for frame in frames {
        match frame {
            MixedFrame::Real(_) if saw_dummy => return false,
            MixedFrame::Real(_) => real_count += 1,
            MixedFrame::Dummy(_) => saw_dummy = true,
        }
    }

    real_count > 0 && saw_dummy
}

fn pearson(left: &[f64], right: &[f64]) -> f64 {
    let len = left.len().min(right.len());
    if len < 2 {
        return 0.0;
    }

    let left = &left[..len];
    let right = &right[..len];
    let left_mean = mean(left);
    let right_mean = mean(right);

    let mut numerator = 0.0;
    let mut left_square_sum = 0.0;
    let mut right_square_sum = 0.0;

    for (a, b) in left.iter().zip(right) {
        let da = a - left_mean;
        let db = b - right_mean;
        numerator += da * db;
        left_square_sum += da * da;
        right_square_sum += db * db;
    }

    let denominator = left_square_sum.sqrt() * right_square_sum.sqrt();
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

fn window_mean_range(values: &[f64], window: usize) -> f64 {
    if values.len() < window || window == 0 {
        return 0.0;
    }

    let mut min_mean = f64::INFINITY;
    let mut max_mean = f64::NEG_INFINITY;

    for chunk in values.chunks(window) {
        if chunk.len() == window {
            let chunk_mean = mean(chunk);
            min_mean = min_mean.min(chunk_mean);
            max_mean = max_mean.max(chunk_mean);
        }
    }

    max_mean - min_mean
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len().max(1) as f64
}
