//! Adversarial traffic-correlation analyzer for MixTor lab logs.
//!
//! Plays a passive on-path adversary against a recorded session: it reads the
//! lab CSV and measures how much the shaped wire traffic leaks about the real
//! traffic underneath. See `mixtor::correlation_attack` for the threat model.
//!
//! Usage:
//!   mixtor-correlate analyze <server.csv> [--bin-ms 100] [--side server|client]
//!                                         [--direction up|down|both] [--skip-ms N]
//!   mixtor-correlate gen-sample <dir>     # produce server.csv + client.csv over loopback
//!
//! `--direction` selects which traffic direction to attack:
//!   up   — client→server request/uplink   (server: server_wire_input)
//!   down — server→client reply/download    (server: server_wire_return)
//!   both — run the attack on each in turn (default); fails if either leaks.

use mixtor::correlation_attack::{
    parse_log, run_adversary, AdversaryReport, AttackConfig, EventNames,
};
use std::error::Error;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let cmd = args.next().unwrap_or_default();

    match cmd.as_str() {
        "analyze" => {
            let mut path: Option<PathBuf> = None;
            let mut bin_ms = 100.0f64;
            let mut side = "server".to_string();
            let mut direction = "both".to_string();
            let mut skip_ms = 0.0f64;
            while let Some(a) = args.next() {
                match a.as_str() {
                    "--bin-ms" => bin_ms = args.next().ok_or("--bin-ms needs a value")?.parse()?,
                    "--side" => side = args.next().ok_or("--side needs a value")?,
                    "--direction" => {
                        direction = args.next().ok_or("--direction needs a value")?
                    }
                    // Drop the first N ms (the connection-setup / handshake phase,
                    // which is intentionally unshaped). The steady-state browsing
                    // traffic is the real correlation attack surface.
                    "--skip-ms" => skip_ms = args.next().ok_or("--skip-ms needs a value")?.parse()?,
                    other => path = Some(PathBuf::from(other)),
                }
            }
            let path = path.ok_or("analyze needs a CSV path")?;

            let directions = resolve_directions(&side, &direction)?;

            let mut records = parse_log(&path)?;
            eprintln!("parsed {} records from {}", records.len(), path.display());
            if skip_ms > 0.0 {
                let before = records.len();
                if let Some(t0) = records.iter().map(|r| r.unix_nanos).min() {
                    let cutoff = t0 + (skip_ms * 1_000_000.0) as u128;
                    records.retain(|r| r.unix_nanos >= cutoff);
                }
                eprintln!(
                    "skipped first {:.0} ms (handshake): {} -> {} records",
                    skip_ms,
                    before,
                    records.len()
                );
            }

            let config = AttackConfig { bin_ms, ..AttackConfig::default() };
            let mut any_won = false;
            for (label, events) in &directions {
                println!("\n############################################################");
                println!("# DIRECTION: {label}");
                println!("############################################################");
                let report = run_adversary(&records, events, config);
                print_report(&report);
                any_won |= report.adversary_won();
            }
            if any_won {
                std::process::exit(1);
            }
            Ok(())
        }
        "gen-sample" => {
            let dir = args.next().ok_or("gen-sample needs an output dir")?;
            // Optional: number of concurrent flows (default 4) and bursts per
            // flow (default 64 ≈ ~20s; raise for long traces / DL datasets).
            let flows = args.next().and_then(|s| s.parse().ok()).unwrap_or(4usize);
            let bursts = args.next().and_then(|s| s.parse().ok()).unwrap_or(64usize);
            gen::generate(PathBuf::from(dir), flows, bursts)
        }
        _ => {
            eprintln!(
                "usage:\n  \
                 mixtor-correlate analyze <csv> [--bin-ms N] [--side server|client] \
                 [--direction up|down|both] [--skip-ms N]\n  \
                 mixtor-correlate gen-sample <dir>"
            );
            Err("unknown or missing subcommand".into())
        }
    }
}

/// Map `--side` + `--direction` to the labelled `EventNames` to attack.
fn resolve_directions(
    side: &str,
    direction: &str,
) -> Result<Vec<(String, EventNames)>, Box<dyn Error>> {
    let up = match side {
        "server" => ("uplink (client→server)".to_string(), EventNames::server()),
        "client" => ("uplink (client→server)".to_string(), EventNames::client()),
        other => return Err(format!("--side must be server or client, got {other}").into()),
    };
    let down = match side {
        "server" => (
            "downlink (server→client)".to_string(),
            EventNames::server_return(),
        ),
        "client" => (
            "downlink (server→client)".to_string(),
            EventNames::client_return(),
        ),
        _ => unreachable!(),
    };
    match direction {
        "up" => Ok(vec![up]),
        "down" => Ok(vec![down]),
        "both" => Ok(vec![up, down]),
        other => Err(format!("--direction must be up, down or both, got {other}").into()),
    }
}

fn print_report(report: &AdversaryReport) {
    println!("\n=== MixTor adversarial correlation report ===");
    println!(
        "base bin width: {:.0} ms   flows in log: {}   max lag searched: {} bins",
        report.config.bin_ms,
        report.conn_ids.len(),
        report.config.max_lag_bins
    );

    println!("\n[1] Size-classification attack (real vs dummy by frame size)");
    match &report.size_attack {
        None => println!("    skipped — log has no real+dummy wire frames to compare"),
        Some(a) => {
            println!(
                "    real frames: {}   dummy frames: {}",
                a.real_frames, a.dummy_frames
            );
            println!(
                "    KS statistic:            {:.3}  (threshold {:.2})",
                a.ks_statistic, report.config.size_ks_threshold
            );
            println!(
                "    best size-classifier acc: {:.3}  (0.50 = chance)",
                a.best_threshold_accuracy
            );
            verdict(
                a.distinguishable,
                "sizes distinguishable — adversary can strip dummies",
                "sizes indistinguishable — dummies blend with real frames",
            );
        }
    }

    println!("\n[2] Volume-correlation attack (wire envelope vs real envelope, per flow)");
    match &report.volume_attack {
        None => println!("    skipped — not enough binned data"),
        Some(a) => {
            println!("    worst-case flow: {}   bins: {}", a.flow, a.bins);
            println!("    Pearson(wire, real):      {:+.3}", a.pearson);
            println!(
                "    peak cross-corr:          {:.3}  at lag {} bins ({:+.0} ms)",
                a.peak_cross_corr,
                a.peak_lag_bins,
                a.peak_lag_bins as f64 * report.config.bin_ms
            );
            verdict(
                a.leaks,
                "wire volume tracks real activity — cover leaks the envelope",
                "wire volume decoupled from real activity — cover holds",
            );
        }
    }

    println!("\n[2b] Multi-scale volume attack (same attack at coarser bins)");
    if report.volume_coarse.is_empty() {
        println!("    skipped — no coarser scales applicable");
    } else {
        for (bin_ms, a) in &report.volume_coarse {
            println!(
                "    {:>6.0} ms bins: Pearson {:+.3}  peak {:.3}  (flow {})  -> {}",
                bin_ms,
                a.pearson,
                a.peak_cross_corr,
                a.flow,
                if a.leaks { "LEAKS" } else { "ok" }
            );
        }
        let worst = report
            .volume_coarse
            .iter()
            .any(|(_, a)| a.leaks);
        verdict(
            worst,
            "a slow envelope leak shows at coarse timescales",
            "no slow leak at coarse timescales",
        );
    }

    println!("\n[3] Flow-linking attack (match each wire flow to its real flow)");
    match &report.linking_attack {
        None => println!("    skipped — need >= 2 flows in the log"),
        Some(a) => {
            println!(
                "    linking accuracy: {:.3}   chance baseline: {:.3}",
                a.linking_accuracy, a.chance
            );
            for (wire, guess, ok) in &a.matches {
                println!(
                    "    wire flow {wire} -> real flow {guess}   [{}]",
                    if *ok { "CORRECT" } else { "wrong" }
                );
            }
            verdict(
                a.beats_chance,
                "flows linkable above chance — connections de-anonymisable",
                "flows not linkable above chance",
            );
        }
    }

    println!("\n[4] Total-volume (integral) attack (per-flow total wire vs total real)");
    match &report.total_volume_attack {
        None => println!("    skipped — need >= 3 flows with real traffic"),
        Some(a) => {
            println!(
                "    flows: {}   Pearson(total): {:+.3}   Spearman(total): {:+.3}  (threshold {:.2})",
                a.flows, a.pearson_total, a.spearman_total, report.config.volume_corr_threshold
            );
            println!(
                "    wire-total spread (CV): {:.3}  ({})",
                a.wire_cv,
                if a.wire_cv < 0.05 {
                    "totals ~uniform — not rankable"
                } else {
                    "totals distinguishable"
                }
            );
            for (id, wire, real) in &a.totals {
                println!(
                    "    flow {id}: wire {:>9.0} B   real {:>9.0} B   ratio {:.2}",
                    wire,
                    real,
                    if *real > 0.0 { wire / real } else { 0.0 }
                );
            }
            verdict(
                a.leaks,
                "total wire volume ranks flows by real volume — integral leaks",
                "total wire volume decoupled from real volume — integral holds",
            );
        }
    }

    println!("\n[5] Duration leak (per-flow wire active-span vs real active-span) — informational");
    match &report.duration_leak {
        None => println!("    skipped — not enough timestamped events"),
        Some(d) => {
            for (id, wire_ms, real_ms) in &d.spans {
                println!(
                    "    flow {id}: wire span {:>8.0} ms   real span {:>8.0} ms",
                    wire_ms, real_ms
                );
            }
            println!("    median |wire-real|/real span gap: {:.3}", d.median_rel_gap);
            if d.leaks {
                println!(
                    "    >> NOTE: wire and real spans track closely — session DURATION is \
                     essentially unpadded (a known residual; not scored)."
                );
            } else {
                println!("    >> duration spans diverge — lifetime not a clean tell");
            }
        }
    }

    println!("\n=== Overall (scored attacks) ===");
    if report.adversary_won() {
        println!("ADVERSARY GAINED AN ADVANTAGE — shaping leaked exploitable correlation.");
    } else {
        println!("No exploitable correlation found at this configuration.");
    }
    println!();
}

fn verdict(adversary_succeeded: bool, on_success: &str, on_fail: &str) {
    if adversary_succeeded {
        println!("    >> ADVERSARY WINS: {on_success}");
    } else {
        println!("    >> defended:       {on_fail}");
    }
}

// ---------------------------------------------------------------------------
// Sample-log generation over loopback
// ---------------------------------------------------------------------------
//
// Drives four concurrent client flows through the real shaping path against a
// *web-like* bridge that amplifies each request into a larger reply, so the
// server→client download path — the bulk traffic and the dominant correlation
// surface — is actually exercised. Each flow gets a distinct amplification
// factor so the download volumes differ, making the linking/integral attacks
// meaningful. Produces a server.csv + client.csv the analyzer can chew on
// without a live Tor bridge.

mod gen {
    use mixtor::lab::LabLogger;
    use mixtor::transport::{
        handle_client_connection_with_lab, handle_server_connection_with_lab, EmitterHandle,
    };
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use std::error::Error;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};

    pub fn generate(dir: PathBuf, flows: usize, bursts: usize) -> Result<(), Box<dyn Error>> {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        rt.block_on(run(dir, flows.max(1), bursts.max(1)))
    }

    async fn run(dir: PathBuf, flow_count: usize, bursts: usize) -> Result<(), Box<dyn Error>> {
        let server_lab = LabLogger::create(&dir, "server")?;
        let client_lab = LabLogger::create(&dir, "client")?;

        // Web-like "bridge": each request is amplified into a larger reply so
        // the download path carries the bulk. A per-connection factor gives each
        // flow a distinct download volume.
        let bridge = TcpListener::bind("127.0.0.1:0").await?;
        let bridge_addr = bridge.local_addr()?;
        let conn_counter = Arc::new(AtomicU64::new(0));
        tokio::spawn(async move {
            loop {
                let Ok((mut s, _)) = bridge.accept().await else {
                    break;
                };
                let idx = conn_counter.fetch_add(1, Ordering::Relaxed);
                let factor = 2 + (idx % 4) as usize; // 2..=5×
                tokio::spawn(async move {
                    const PER_WRITE_CAP: usize = 32 * 1024;
                    const TOTAL_CAP: usize = 400 * 1024;
                    let mut buf = vec![0u8; 64 * 1024];
                    let mut sent: usize = 0;
                    loop {
                        match s.read(&mut buf).await {
                            Ok(0) | Err(_) => break,
                            Ok(n) => {
                                let want = (n * factor)
                                    .min(PER_WRITE_CAP)
                                    .min(TOTAL_CAP.saturating_sub(sent));
                                if want == 0 {
                                    continue; // total cap reached; keep draining
                                }
                                let resp = vec![0xE5u8; want];
                                if s.write_all(&resp).await.is_err() {
                                    break;
                                }
                                sent += want;
                            }
                        }
                    }
                });
            }
        });

        // MixTor server proxy.
        let server = TcpListener::bind("127.0.0.1:0").await?;
        let server_addr = server.local_addr()?;
        {
            let server_lab = server_lab.clone();
            tokio::spawn(async move {
                // One shared emitter clocks every server-side flow — this is the
                // many-flows-per-process case the unified emitter fixes.
                let emitter = EmitterHandle::new();
                loop {
                    let Ok((client, _)) = server.accept().await else {
                        break;
                    };
                    let lab = Some(server_lab.clone());
                    let emitter = emitter.clone();
                    tokio::spawn(async move {
                        let _ = handle_server_connection_with_lab(
                            client, bridge_addr, 1200, lab, emitter,
                        )
                        .await;
                    });
                }
            });
        }

        // MixTor client proxy.
        let client_listener = TcpListener::bind("127.0.0.1:0").await?;
        let client_addr = client_listener.local_addr()?;
        {
            let client_lab = client_lab.clone();
            tokio::spawn(async move {
                // One shared emitter clocks every client-side flow.
                let emitter = EmitterHandle::new();
                loop {
                    let Ok((app, _)) = client_listener.accept().await else {
                        break;
                    };
                    let lab = Some(client_lab.clone());
                    let emitter = emitter.clone();
                    tokio::spawn(async move {
                        let _ = handle_client_connection_with_lab(
                            app, server_addr, 1200, lab, emitter,
                        )
                        .await;
                    });
                }
            });
        }

        // `flow_count` concurrent flows with distinct bursty patterns so the
        // linking/integral attacks have heterogeneous flows to tell apart. Burst
        // sizes model client→server *uplink* requests (small — the bulk travels
        // on the amplified downlink reply path), each a handful of cells.
        // MIXTOR_WM=1 makes flow 0 inject a square-wave timing watermark (active
        // attack probe); the rest are normal cover traffic.
        let wm = std::env::var("MIXTOR_WM").is_ok();
        let mut flows = Vec::with_capacity(flow_count);
        for i in 0..flow_count {
            let seed = 7 + (i as u64) * 1009;
            let burst_bytes = 500 + (i % 6) * 220;
            let min_idle = 70 + (i % 5) as u64 * 30;
            let max_idle = min_idle + 150 + (i % 4) as u64 * 90;
            flows.push(tokio::spawn(app_flow(
                client_addr, seed, burst_bytes, min_idle, max_idle, bursts, wm && i == 0,
            )));
        }
        for f in flows {
            f.await?.map_err(|e| e.to_string())?;
        }

        // Give the proxies a moment to flush trailing cover/logs.
        tokio::time::sleep(Duration::from_millis(500)).await;
        let (fired, dropped) = mixtor::timing_correlator::catchup_stats();
        eprintln!(
            "emitter slots: fired {fired}  dropped(catch-up) {dropped}  ({:.2}% dropped)",
            if fired > 0 { 100.0 * dropped as f64 / fired as f64 } else { 0.0 }
        );
        eprintln!("wrote server.csv and client.csv to {}", dir.display());
        eprintln!(
            "now run:  mixtor-correlate analyze {}/server.csv --direction both --skip-ms 1500",
            dir.display()
        );
        eprintln!(
            "tip: set MIXTOR_TAIL_FLOOR_MS=25000 before gen-sample to pad session \
             lifetime and close the duration/integral leak."
        );
        Ok(())
    }

    /// One application flow: warm up past the dummy threshold, then alternate
    /// real bursts with idle gaps (during which the cover clock emits dummies).
    /// A background reader drains the (amplified) download so the socket never
    /// back-pressures.
    async fn app_flow(
        addr: std::net::SocketAddr,
        seed: u64,
        burst_bytes: usize,
        min_idle_ms: u64,
        max_idle_ms: u64,
        bursts: usize,
        watermark: bool,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let mut rng = StdRng::seed_from_u64(seed);
        let stream = TcpStream::connect(addr).await?;
        let (mut rd, mut wr) = stream.into_split();

        // Drain the downloaded bytes so the socket never back-pressures.
        let reader = tokio::spawn(async move {
            let mut buf = vec![0u8; 64 * 1024];
            while let Ok(n) = rd.read(&mut buf).await {
                if n == 0 {
                    break;
                }
            }
        });

        // Warmup: trickle ~70 KB in small RTT-paced chunks (not an instant dump)
        // so we clear WARMUP_BYTES (64 KB) while modelling a real handshake's
        // round-trip pacing. Real cells go out immediately for low setup latency;
        // the transport's warmup credit keeps the cover from piling on top, so
        // this window now sits at the steady cover rate rather than spiking.
        let warm = vec![0xABu8; 2560];
        for _ in 0..28 {
            wr.write_all(&warm).await?;
            tokio::time::sleep(Duration::from_millis(35)).await;
        }

        if watermark {
            // ACTIVE-ATTACK probe: this flow's traffic carries a deterministic
            // square-wave watermark — HIGH phase (send every 100 ms) then SILENT
            // phase, period ~3 s (~0.33 Hz). An active attacker who controls a
            // flow's timing injects this and checks whether it survives onto the
            // shaped wire. If the cover erases timing, the 0.33 Hz line is absent.
            const HALF_MS: u64 = 1500;
            let cycles = (bursts / 30).max(4);
            for _ in 0..cycles {
                let mut elapsed = 0u64;
                while elapsed < HALF_MS {
                    wr.write_all(&vec![0xCDu8; burst_bytes]).await?;
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    elapsed += 100;
                }
                tokio::time::sleep(Duration::from_millis(HALF_MS)).await; // SILENT
            }
        } else {
            // Steady state: bursty on/off browsing with random idle gaps.
            for _ in 0..bursts {
                let n = rng.gen_range(burst_bytes / 2..=burst_bytes);
                wr.write_all(&vec![0xCDu8; n]).await?;
                let idle = rng.gen_range(min_idle_ms..=max_idle_ms);
                tokio::time::sleep(Duration::from_millis(idle)).await;
            }
        }

        wr.shutdown().await?;
        let _ = reader.await;
        Ok(())
    }
}
