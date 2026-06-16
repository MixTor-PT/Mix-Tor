//! Adversarial correlation analysis over lab traffic logs.
//!
//! # Threat model
//!
//! This module plays the role of a *passive on-path adversary* who has
//! recorded the lab CSV produced by `LabLogger` and now tries to defeat the
//! transport's traffic shaping using only the information a real observer
//! would have: the timing and size of every frame on the wire.
//!
//! The adversary's goal is to recover, from the shaped wire traffic, the
//! pattern of the *real* underlying traffic — the thing the shaping is meant
//! to hide.  The lab CSV conveniently labels each wire frame `real`/`dummy`
//! (the `kind` column) and separately logs the real bytes forwarded to the
//! bridge (`server_real_output`).  A real adversary would NOT have those
//! labels; we use them only as **ground truth to score the attack**.  Every
//! attack input that an adversary actually gets to see is derived purely from
//! the wire stream (`server_wire_input` / `client_wire_output`) — timing and
//! size, never the `kind` label.
//!
//! # Attacks implemented
//!
//! 1. **Size-classification attack** (`SizeAttack`).  Can the adversary tell a
//!    real frame from a dummy by its size alone?  We compute the two-sample KS
//!    statistic between the real-frame and dummy-frame size distributions, plus
//!    the accuracy of the best single-threshold size classifier.  If sizes are
//!    distinguishable the adversary simply discards dummies and recovers the
//!    real timing exactly — the worst case for the defence.
//!
//! 2. **Volume-correlation attack** (`VolumeAttack`).  Bin time into fixed
//!    windows and build two series: the *observable* wire volume (every frame)
//!    and the *secret* real volume (real frames only).  We report the Pearson
//!    correlation and the peak normalised cross-correlation over a lag window.
//!    A high value means the wire volume tracks real activity despite the
//!    cover — the dummies are not actually masking the real envelope.
//!
//! 3. **Flow-linking attack** (`LinkingAttack`).  With two or more connections
//!    in the log, the adversary tries to match each flow's observable wire
//!    series to the correct real series by maximum correlation.  We report the
//!    fraction of flows linked correctly; 1.0 means every flow was
//!    de-anonymised, `1/n` is the chance baseline.
//!
//! All three are pure functions over parsed records and are unit-tested with
//! synthetic logs at the bottom of the file.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

// ---------------------------------------------------------------------------
// Log records
// ---------------------------------------------------------------------------

/// One parsed line of a lab CSV.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogRecord {
    pub unix_nanos: u128,
    pub role: String,
    pub event: String,
    pub conn_id: u64,
    pub seq: u64,
    pub bytes: usize,
    /// "real" or "dummy" — ground-truth label, used only for scoring.
    pub kind: String,
}

/// Parse a lab CSV file written by `LabLogger`.
///
/// The header line is skipped.  Malformed lines are skipped rather than
/// aborting the whole analysis, so a truncated log (e.g. a session still in
/// flight) can still be examined.
pub fn parse_log(path: impl AsRef<Path>) -> io::Result<Vec<LogRecord>> {
    let file = File::open(path)?;
    Ok(parse_reader(BufReader::new(file)))
}

/// Parse lab-CSV records from any buffered reader (used by tests).
pub fn parse_reader<R: BufRead>(reader: R) -> Vec<LogRecord> {
    let mut out = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };
        // Skip the header row and blank lines.
        if i == 0 && line.starts_with("unix_nanos") {
            continue;
        }
        if line.trim().is_empty() {
            continue;
        }
        if let Some(rec) = parse_line(&line) {
            out.push(rec);
        }
    }
    out
}

fn parse_line(line: &str) -> Option<LogRecord> {
    // unix_nanos,role,event,conn_id,seq,bytes,kind
    let mut f = line.split(',');
    let unix_nanos = f.next()?.trim().parse().ok()?;
    let role = f.next()?.trim().to_string();
    let event = f.next()?.trim().to_string();
    let conn_id = f.next()?.trim().parse().ok()?;
    let seq = f.next()?.trim().parse().ok()?;
    let bytes = f.next()?.trim().parse().ok()?;
    let kind = f.next()?.trim().to_string();
    Some(LogRecord {
        unix_nanos,
        role,
        event,
        conn_id,
        seq,
        bytes,
        kind,
    })
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Which event names hold the data each attack consumes.
///
/// Defaults match the server-side log.  For a client-side log pass
/// `EventNames::client()`.
#[derive(Clone, Debug)]
pub struct EventNames {
    /// Frames the adversary observes on the wire (mixed real + dummy).
    pub wire: String,
    /// Real traffic ground truth (dummies already stripped).
    pub real: String,
}

impl EventNames {
    pub fn server() -> Self {
        Self {
            wire: "server_wire_input".to_string(),
            real: "server_real_output".to_string(),
        }
    }

    pub fn client() -> Self {
        Self {
            wire: "client_wire_output".to_string(),
            real: "client_input".to_string(),
        }
    }

    /// Server-side view of the **download** direction (server → client). The
    /// shaped reply traffic on the wire vs. the real bytes read from the bridge.
    /// This direction was historically an unshaped raw passthrough; it carries
    /// the bulk of browsing traffic and is the dominant correlation surface.
    pub fn server_return() -> Self {
        Self {
            wire: "server_wire_return".to_string(),
            real: "server_bridge_reply".to_string(),
        }
    }

    /// Client-side view of the **download** direction (server → client): the
    /// shaped frames arriving on the wire vs. the real bytes delivered to the app.
    pub fn client_return() -> Self {
        Self {
            wire: "client_wire_return".to_string(),
            real: "client_inbound_real".to_string(),
        }
    }
}

/// Tunables for the correlation attacks.
#[derive(Clone, Copy, Debug)]
pub struct AttackConfig {
    /// Width of each time bin in milliseconds for the volume/linking series.
    pub bin_ms: f64,

    /// Maximum lag (in bins) searched in the cross-correlation. The adversary
    /// allows for buffering/scheduling delay between real and wire traffic.
    pub max_lag_bins: usize,

    /// Pearson/cross-correlation magnitude above which the volume attack is
    /// judged a success (the wire leaks the real envelope).
    pub volume_corr_threshold: f64,

    /// KS statistic above which real and dummy sizes are judged distinguishable.
    pub size_ks_threshold: f64,

    /// Minimum number of bins a flow's series must have for the volume attack to
    /// trust it. Below this a cross-correlation over a handful of points is
    /// statistically meaningless (3 near-collinear points give |r|≈1), so coarse
    /// bin widths on short sessions would otherwise produce false positives.
    pub min_bins: usize,
}

impl Default for AttackConfig {
    fn default() -> Self {
        Self {
            bin_ms: 100.0,
            max_lag_bins: 20,
            volume_corr_threshold: 0.5,
            size_ks_threshold: 0.15,
            min_bins: 3,
        }
    }
}

// ---------------------------------------------------------------------------
// Report types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct AdversaryReport {
    pub size_attack: Option<SizeAttack>,
    pub volume_attack: Option<VolumeAttack>,
    /// Volume attack repeated at coarser bin widths. A constant-rate cover can
    /// flatten the fine-grained (100 ms) envelope while a slow leak — e.g. a
    /// queue-backlog plateau under load — still shows at multi-second bins.
    /// Each entry is `(bin_ms, worst-flow attack at that scale)`.
    pub volume_coarse: Vec<(f64, VolumeAttack)>,
    /// Cross-flow total-volume (integral) attack.
    pub total_volume_attack: Option<TotalVolumeAttack>,
    /// Per-connection active-span (duration) leak — informational.
    pub duration_leak: Option<DurationLeak>,
    pub linking_attack: Option<LinkingAttack>,
    pub conn_ids: Vec<u64>,
    pub config: AttackConfig,
}

impl AdversaryReport {
    /// True if any *scored* attack succeeded — i.e. the adversary gained an
    /// advantage the shaping was supposed to deny. The duration leak is reported
    /// but not scored here: session-length quantization is a known residual that
    /// the per-window shaping does not fully remove, surfaced for visibility.
    pub fn adversary_won(&self) -> bool {
        self.size_attack.as_ref().is_some_and(|a| a.distinguishable)
            || self.volume_attack.as_ref().is_some_and(|a| a.leaks)
            || self.volume_coarse.iter().any(|(_, a)| a.leaks)
            || self.total_volume_attack.as_ref().is_some_and(|a| a.leaks)
            || self.linking_attack.as_ref().is_some_and(|a| a.beats_chance)
    }
}

/// Result of the size-based real-vs-dummy classification attack.
#[derive(Clone, Debug)]
pub struct SizeAttack {
    pub real_frames: usize,
    pub dummy_frames: usize,
    /// Two-sample KS statistic between real and dummy size distributions.
    pub ks_statistic: f64,
    /// Accuracy of the best single size-threshold classifier (0.5 = chance).
    pub best_threshold_accuracy: f64,
    /// True when sizes are distinguishable beyond the configured KS threshold.
    pub distinguishable: bool,
}

/// Result of the wire-vs-real volume correlation attack.
///
/// Measured per flow (an adversary attacks one connection at a time); the
/// reported figures are for the worst-case — most leaky — flow.
#[derive(Clone, Debug)]
pub struct VolumeAttack {
    /// The connection this (worst-case) result is for.
    pub flow: u64,
    pub bins: usize,
    /// Pearson correlation of observable wire volume vs secret real volume.
    pub pearson: f64,
    /// Peak |normalised cross-correlation| over the lag window.
    pub peak_cross_corr: f64,
    /// Lag (in bins) at which the peak occurred.
    pub peak_lag_bins: i64,
    /// True when correlation exceeds the configured threshold.
    pub leaks: bool,
}

/// Result of the cross-flow **total-volume (integral)** attack.
///
/// The per-window envelope can be perfectly flat yet the *total* bytes a flow
/// puts on the wire over its lifetime can still rank flows by their real volume
/// — e.g. when a sustained real rate above the cover rate makes the queue back
/// up, so total wire ≈ total real. This attack correlates each flow's total
/// wire bytes against its total real bytes across all flows.
#[derive(Clone, Debug)]
pub struct TotalVolumeAttack {
    pub flows: usize,
    /// Pearson correlation of per-flow (total wire bytes, total real bytes).
    pub pearson_total: f64,
    /// Spearman rank correlation — catches a monotone-but-nonlinear integral
    /// leak (volume quantization makes the relationship a step function).
    pub spearman_total: f64,
    /// Coefficient of variation (stddev/mean) of the per-flow total wire bytes.
    /// When the wire totals are ~uniform this is near zero and the integral
    /// carries no usable signal regardless of how the rank noise falls.
    pub wire_cv: f64,
    /// Per-flow `(conn_id, total_wire_bytes, total_real_bytes)`.
    pub totals: Vec<(u64, f64, f64)>,
    /// True when the wire totals vary enough to be distinguishable AND a
    /// correlation exceeds the configured threshold.
    pub leaks: bool,
}

/// Per-connection active-span leak (informational, not scored).
///
/// If a connection's wire activity spans the same wall-clock window as its real
/// activity, the session *duration* leaks even when the within-session envelope
/// is flat. Reported so residual duration correlation is visible.
#[derive(Clone, Debug)]
pub struct DurationLeak {
    /// Per-flow `(conn_id, wire_span_ms, real_span_ms)`.
    pub spans: Vec<(u64, f64, f64)>,
    /// Median |wire_span - real_span| / real_span across flows.
    pub median_rel_gap: f64,
    /// True when wire and real spans track closely (median rel. gap < 0.20):
    /// the duration is essentially unpadded.
    pub leaks: bool,
}

/// Result of the cross-flow linking attack.
#[derive(Clone, Debug)]
pub struct LinkingAttack {
    pub flows: usize,
    /// Fraction of flows whose real series was correctly matched by max corr.
    pub linking_accuracy: f64,
    /// Chance baseline (1 / flows).
    pub chance: f64,
    /// (wire_conn, guessed_real_conn, correct?) for each flow.
    pub matches: Vec<(u64, u64, bool)>,
    /// True when accuracy beats the chance baseline.
    pub beats_chance: bool,
}

// ---------------------------------------------------------------------------
// Top-level driver
// ---------------------------------------------------------------------------

/// Run every applicable attack over `records` and return a scored report.
pub fn run_adversary(
    records: &[LogRecord],
    events: &EventNames,
    config: AttackConfig,
) -> AdversaryReport {
    let conn_ids = distinct_conn_ids(records, events);

    // Repeat the volume attack at coarser bins than the base scale. A constant
    // cover rate flattens the fine envelope, but a slow integral leak (queue
    // backlog under load) only resolves at multi-second bins.
    let mut volume_coarse = Vec::new();
    for &scale in &[1_000.0_f64, 5_000.0] {
        if scale <= config.bin_ms * 1.5 {
            continue; // not meaningfully coarser than the base scale
        }
        // Require >= 30 bins at this scale before scoring it. A coarse-bin
        // Pearson over a handful of points - and reported as the worst of N
        // flows - is dominated by small-sample noise: a real multi-flow Tor
        // capture of the same (defended) traffic read 0.77 at the 5 s scale over
        // ~17 bins but 0.30 over ~38 bins. 30 bins keeps the null CI comfortably
        // under the 0.5 threshold. Judge at lag 0 (max_lag_bins = 0): a real
        // envelope leak from cover that buffers by at most a few hundred ms shows
        // within the same >=1 s bin, so searching second-wide lags only maximises
        // noise (upward-biased peak).
        let cfg = AttackConfig {
            bin_ms: scale,
            min_bins: 30,
            max_lag_bins: 0,
            ..config
        };
        if let Some(a) = volume_attack(records, events, cfg, &conn_ids) {
            volume_coarse.push((scale, a));
        }
    }

    AdversaryReport {
        size_attack: size_attack(records, events, config),
        volume_attack: volume_attack(records, events, config, &conn_ids),
        volume_coarse,
        total_volume_attack: total_volume_attack(records, events, config, &conn_ids),
        duration_leak: duration_leak(records, events, &conn_ids),
        linking_attack: linking_attack(records, events, config, &conn_ids),
        conn_ids,
        config,
    }
}

fn distinct_conn_ids(records: &[LogRecord], events: &EventNames) -> Vec<u64> {
    let mut ids: Vec<u64> = records
        .iter()
        .filter(|r| r.event == events.wire || r.event == events.real)
        .map(|r| r.conn_id)
        .collect();
    ids.sort_unstable();
    ids.dedup();
    ids
}

// ---------------------------------------------------------------------------
// Attack 1: size-based real/dummy classification
// ---------------------------------------------------------------------------

fn size_attack(
    records: &[LogRecord],
    events: &EventNames,
    config: AttackConfig,
) -> Option<SizeAttack> {
    let mut real_sizes = Vec::new();
    let mut dummy_sizes = Vec::new();
    for r in records.iter().filter(|r| r.event == events.wire) {
        match r.kind.as_str() {
            "real" => real_sizes.push(r.bytes as f64),
            "dummy" => dummy_sizes.push(r.bytes as f64),
            _ => {}
        }
    }
    if real_sizes.is_empty() || dummy_sizes.is_empty() {
        return None;
    }

    let ks = ks_statistic(&real_sizes, &dummy_sizes);
    let accuracy = best_threshold_accuracy(&real_sizes, &dummy_sizes);

    Some(SizeAttack {
        real_frames: real_sizes.len(),
        dummy_frames: dummy_sizes.len(),
        ks_statistic: ks,
        best_threshold_accuracy: accuracy,
        distinguishable: ks > config.size_ks_threshold,
    })
}

// ---------------------------------------------------------------------------
// Attack 2: wire-vs-real volume correlation
// ---------------------------------------------------------------------------

fn volume_attack(
    records: &[LogRecord],
    events: &EventNames,
    config: AttackConfig,
    conn_ids: &[u64],
) -> Option<VolumeAttack> {
    // An adversary attacks one connection at a time: for each flow, correlate
    // the observable wire envelope (every frame) against the secret real
    // envelope (real frames only). Report the worst-case — most leaky — flow.
    // (Aggregating independent flows into one series is not a sound model: it
    // conflates the connections' start/stop boundaries, which the adversary
    // sees at the TCP layer anyway, with the within-connection envelope the
    // cover is meant to hide.)
    let mut worst: Option<VolumeAttack> = None;
    for &id in conn_ids {
        let (wire_series, real_series) =
            match aligned_series(records, events, config.bin_ms, Some(id)) {
                Some(s) => s,
                None => continue,
            };
        if wire_series.len() < config.min_bins.max(3) {
            continue;
        }
        let pearson = pearson(&wire_series, &real_series);
        let (peak_lag, peak) =
            peak_cross_correlation(&wire_series, &real_series, config.max_lag_bins);
        let cand = VolumeAttack {
            flow: id,
            bins: wire_series.len(),
            pearson,
            peak_cross_corr: peak,
            peak_lag_bins: peak_lag,
            leaks: peak.abs() > config.volume_corr_threshold
                || pearson.abs() > config.volume_corr_threshold,
        };
        if worst
            .as_ref()
            .map_or(true, |w| cand.peak_cross_corr > w.peak_cross_corr)
        {
            worst = Some(cand);
        }
    }
    worst
}

/// Build wire and real byte series on a single shared time epoch/bin grid so
/// they are directly comparable index-for-index. When `conn` is `Some(id)`,
/// only that connection's records are included.
fn aligned_series(
    records: &[LogRecord],
    events: &EventNames,
    bin_ms: f64,
    conn: Option<u64>,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let bin_ns = (bin_ms * 1_000_000.0) as u128;
    if bin_ns == 0 {
        return None;
    }
    let relevant: Vec<&LogRecord> = records
        .iter()
        .filter(|r| r.event == events.wire || r.event == events.real)
        .filter(|r| conn.map_or(true, |id| r.conn_id == id))
        .collect();
    let t0 = relevant.iter().map(|r| r.unix_nanos).min()?;
    let t_end = relevant.iter().map(|r| r.unix_nanos).max()?;
    let n_bins = ((t_end - t0) / bin_ns) as usize + 1;

    let mut wire = vec![0.0f64; n_bins];
    let mut real = vec![0.0f64; n_bins];
    for r in relevant {
        let idx = ((r.unix_nanos - t0) / bin_ns) as usize;
        if r.event == events.wire {
            wire[idx] += r.bytes as f64;
        } else {
            real[idx] += r.bytes as f64;
        }
    }
    Some((wire, real))
}

// ---------------------------------------------------------------------------
// Attack 3: cross-flow linking
// ---------------------------------------------------------------------------

fn linking_attack(
    records: &[LogRecord],
    events: &EventNames,
    config: AttackConfig,
    conn_ids: &[u64],
) -> Option<LinkingAttack> {
    if conn_ids.len() < 2 {
        return None;
    }
    let bin_ns = (config.bin_ms * 1_000_000.0) as u128;
    if bin_ns == 0 {
        return None;
    }
    // Shared epoch across all flows so series are comparable.
    let t0 = records
        .iter()
        .filter(|r| r.event == events.wire || r.event == events.real)
        .map(|r| r.unix_nanos)
        .min()?;
    let t_end = records
        .iter()
        .filter(|r| r.event == events.wire || r.event == events.real)
        .map(|r| r.unix_nanos)
        .max()?;
    let n_bins = ((t_end - t0) / bin_ns) as usize + 1;

    let mut wire_by_flow: BTreeMap<u64, Vec<f64>> = BTreeMap::new();
    let mut real_by_flow: BTreeMap<u64, Vec<f64>> = BTreeMap::new();
    for &id in conn_ids {
        wire_by_flow.insert(id, vec![0.0; n_bins]);
        real_by_flow.insert(id, vec![0.0; n_bins]);
    }
    for r in records {
        let idx = if r.unix_nanos >= t0 {
            (((r.unix_nanos - t0) / bin_ns) as usize).min(n_bins - 1)
        } else {
            continue;
        };
        if r.event == events.wire {
            if let Some(v) = wire_by_flow.get_mut(&r.conn_id) {
                v[idx] += r.bytes as f64;
            }
        } else if r.event == events.real {
            if let Some(v) = real_by_flow.get_mut(&r.conn_id) {
                v[idx] += r.bytes as f64;
            }
        }
    }

    // For each wire flow, guess the real flow with the highest peak cross-corr.
    let mut matches = Vec::new();
    let mut correct = 0usize;
    for &wire_id in conn_ids {
        let wire_series = &wire_by_flow[&wire_id];
        let mut best_id = conn_ids[0];
        let mut best_score = f64::NEG_INFINITY;
        for &real_id in conn_ids {
            let real_series = &real_by_flow[&real_id];
            let (_, score) =
                peak_cross_correlation(wire_series, real_series, config.max_lag_bins);
            if score > best_score {
                best_score = score;
                best_id = real_id;
            }
        }
        let is_correct = best_id == wire_id;
        if is_correct {
            correct += 1;
        }
        matches.push((wire_id, best_id, is_correct));
    }

    let flows = conn_ids.len();
    let accuracy = correct as f64 / flows as f64;
    let chance = 1.0 / flows as f64;
    Some(LinkingAttack {
        flows,
        linking_accuracy: accuracy,
        chance,
        matches,
        // Beat chance by a margin so a single lucky match on 2 flows
        // (accuracy 0.5 == chance) does not register as a win.
        beats_chance: accuracy > chance + 1e-9 && accuracy > 0.5,
    })
}

// ---------------------------------------------------------------------------
// Attack 4: cross-flow total-volume (integral)
// ---------------------------------------------------------------------------

fn total_volume_attack(
    records: &[LogRecord],
    events: &EventNames,
    config: AttackConfig,
    conn_ids: &[u64],
) -> Option<TotalVolumeAttack> {
    // A cross-flow correlation needs several flows to be meaningful.
    if conn_ids.len() < 3 {
        return None;
    }
    let mut totals = Vec::new();
    let mut wire_v = Vec::new();
    let mut real_v = Vec::new();
    for &id in conn_ids {
        let mut wire = 0.0;
        let mut real = 0.0;
        for r in records.iter().filter(|r| r.conn_id == id) {
            if r.event == events.wire {
                wire += r.bytes as f64;
            } else if r.event == events.real {
                real += r.bytes as f64;
            }
        }
        // Only flows that actually carried real data are informative.
        if real > 0.0 {
            totals.push((id, wire, real));
            wire_v.push(wire);
            real_v.push(real);
        }
    }
    if totals.len() < 3 {
        return None;
    }
    let pearson_total = pearson(&wire_v, &real_v);
    let spearman_total = spearman(&wire_v, &real_v);
    // Spread of the wire totals. If they are ~uniform (e.g. all sessions padded
    // to the same volume), an adversary cannot rank flows by total wire volume
    // no matter how the rank noise on indistinguishable values happens to fall,
    // so the integral does not leak however high |Spearman| reads on few flows.
    let wire_mean = mean(&wire_v);
    let wire_cv = if wire_mean > 0.0 {
        (variance(&wire_v).sqrt()) / wire_mean
    } else {
        0.0
    };
    const MIN_DISTINGUISHABLE_CV: f64 = 0.05; // 5% — below this, totals are uniform
    let correlated = pearson_total.abs() > config.volume_corr_threshold
        || spearman_total.abs() > config.volume_corr_threshold;
    let leaks = wire_cv >= MIN_DISTINGUISHABLE_CV && correlated;
    Some(TotalVolumeAttack {
        flows: totals.len(),
        pearson_total,
        spearman_total,
        wire_cv,
        totals,
        leaks,
    })
}

// ---------------------------------------------------------------------------
// Duration leak (informational)
// ---------------------------------------------------------------------------

fn duration_leak(
    records: &[LogRecord],
    events: &EventNames,
    conn_ids: &[u64],
) -> Option<DurationLeak> {
    let mut spans = Vec::new();
    let mut rel_gaps = Vec::new();
    for &id in conn_ids {
        let wire_ts: Vec<u128> = records
            .iter()
            .filter(|r| r.conn_id == id && r.event == events.wire)
            .map(|r| r.unix_nanos)
            .collect();
        let real_ts: Vec<u128> = records
            .iter()
            .filter(|r| r.conn_id == id && r.event == events.real)
            .map(|r| r.unix_nanos)
            .collect();
        if wire_ts.len() < 2 || real_ts.len() < 2 {
            continue;
        }
        let wire_span =
            (wire_ts.iter().max().unwrap() - wire_ts.iter().min().unwrap()) as f64 / 1e6;
        let real_span =
            (real_ts.iter().max().unwrap() - real_ts.iter().min().unwrap()) as f64 / 1e6;
        spans.push((id, wire_span, real_span));
        if real_span > 0.0 {
            rel_gaps.push((wire_span - real_span).abs() / real_span);
        }
    }
    if spans.is_empty() {
        return None;
    }
    let had_gaps = !rel_gaps.is_empty();
    let median_rel_gap = median(&mut rel_gaps);
    Some(DurationLeak {
        spans,
        median_rel_gap,
        // Treat a near-zero gap as a leak only when we actually had gaps to
        // measure; if every flow's real span was zero, leave it unflagged.
        leaks: had_gaps && median_rel_gap < 0.20,
    })
}

// ---------------------------------------------------------------------------
// Statistics helpers
// ---------------------------------------------------------------------------

/// Pearson correlation coefficient of two equal-length series.
pub fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return 0.0;
    }
    let a = &a[..n];
    let b = &b[..n];
    let ma = mean(a);
    let mb = mean(b);
    let mut num = 0.0;
    let mut da2 = 0.0;
    let mut db2 = 0.0;
    for (x, y) in a.iter().zip(b) {
        let da = x - ma;
        let db = y - mb;
        num += da * db;
        da2 += da * da;
        db2 += db * db;
    }
    let den = (da2.sqrt()) * (db2.sqrt());
    if den == 0.0 {
        0.0
    } else {
        num / den
    }
}

/// Peak |normalised cross-correlation| of `a` against `b` over lags
/// `-max_lag..=max_lag`.  Returns `(lag_at_peak, peak_value)`.
///
/// A positive lag `k` means `a` is compared against `b` shifted later by `k`
/// bins — i.e. wire traffic trailing the real traffic by `k` bins, which is
/// what buffering/scheduling delay produces.
pub fn peak_cross_correlation(a: &[f64], b: &[f64], max_lag: usize) -> (i64, f64) {
    let n = a.len().min(b.len());
    // Require a substantial overlap so extreme lags — where only a handful of
    // bins overlap — cannot produce a spurious ±1 correlation that swamps the
    // true low-lag match. At least half the series, and never fewer than 3 bins.
    let min_overlap = (n / 2).max(3);
    let mut best_lag = 0i64;
    let mut best = 0.0f64;
    let max_lag = max_lag.min(n.saturating_sub(min_overlap));
    for lag in -(max_lag as i64)..=(max_lag as i64) {
        let overlap = n - lag.unsigned_abs() as usize;
        if overlap < min_overlap {
            continue;
        }
        let c = shifted_pearson(a, b, lag).abs();
        if c > best {
            best = c;
            best_lag = lag;
        }
    }
    (best_lag, best)
}

/// Pearson of `a[i]` against `b[i - lag]` over the overlapping region.
fn shifted_pearson(a: &[f64], b: &[f64], lag: i64) -> f64 {
    let n = a.len().min(b.len()) as i64;
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for i in 0..n {
        let j = i - lag;
        if j >= 0 && j < n {
            xs.push(a[i as usize]);
            ys.push(b[j as usize]);
        }
    }
    pearson(&xs, &ys)
}

/// Two-sample Kolmogorov–Smirnov statistic: the maximum gap between the two
/// empirical CDFs.  0 = identical distributions, 1 = disjoint.
pub fn ks_statistic(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let mut a = a.to_vec();
    let mut b = b.to_vec();
    a.sort_by(|x, y| x.partial_cmp(y).unwrap());
    b.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let (na, nb) = (a.len() as f64, b.len() as f64);
    let (mut ia, mut ib) = (0usize, 0usize);
    let mut max_diff = 0.0f64;
    while ia < a.len() || ib < b.len() {
        let next_a = a.get(ia).copied().unwrap_or(f64::INFINITY);
        let next_b = b.get(ib).copied().unwrap_or(f64::INFINITY);
        let x = next_a.min(next_b);
        while ia < a.len() && a[ia] <= x {
            ia += 1;
        }
        while ib < b.len() && b[ib] <= x {
            ib += 1;
        }
        max_diff = max_diff.max((ia as f64 / na - ib as f64 / nb).abs());
    }
    max_diff
}

/// Accuracy of the best single-threshold size classifier separating two
/// labelled samples.  0.5 means the classifier does no better than a coin.
///
/// The classifier predicts "real if size <= t else dummy" (or the reverse);
/// we sweep every candidate threshold and take the best balanced accuracy.
fn best_threshold_accuracy(real: &[f64], dummy: &[f64]) -> f64 {
    let mut candidates: Vec<f64> = real.iter().chain(dummy.iter()).copied().collect();
    candidates.sort_by(|x, y| x.partial_cmp(y).unwrap());
    candidates.dedup();

    let mut best = 0.5f64;
    for &t in &candidates {
        // Orientation A: real <= t, dummy > t.
        let real_correct = real.iter().filter(|&&x| x <= t).count() as f64 / real.len() as f64;
        let dummy_correct = dummy.iter().filter(|&&x| x > t).count() as f64 / dummy.len() as f64;
        let acc_a = 0.5 * (real_correct + dummy_correct);
        // Orientation B (real > t, dummy <= t) is the mirror: its balanced
        // accuracy is exactly 1 - acc_a.
        best = best.max(acc_a).max(1.0 - acc_a);
    }
    best
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

/// Population variance.
fn variance(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let m = mean(xs);
    xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / xs.len() as f64
}

/// Spearman rank correlation: Pearson on the rank-transformed series. Detects a
/// monotone integral leak that volume quantization turns into a step function
/// (which dents Pearson but not the ranks).
pub fn spearman(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return 0.0;
    }
    let ra = ranks(&a[..n]);
    let rb = ranks(&b[..n]);
    pearson(&ra, &rb)
}

/// Fractional ranks, averaging ranks within ties. Absolute scale is irrelevant
/// to the subsequent Pearson.
fn ranks(xs: &[f64]) -> Vec<f64> {
    let n = xs.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| xs[i].partial_cmp(&xs[j]).unwrap_or(std::cmp::Ordering::Equal));
    let mut r = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && xs[idx[j + 1]] == xs[idx[i]] {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0; // mean rank for the tie group [i, j]
        for &k in &idx[i..=j] {
            r[k] = avg;
        }
        i = j + 1;
    }
    r
}

/// Median of a slice (sorts in place). Returns 0.0 for an empty slice.
fn median(xs: &mut [f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = xs.len();
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        0.5 * (xs[n / 2 - 1] + xs[n / 2])
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn rec(nanos: u128, event: &str, conn: u64, bytes: usize, kind: &str) -> LogRecord {
        LogRecord {
            unix_nanos: nanos,
            role: "server".into(),
            event: event.into(),
            conn_id: conn,
            seq: 0,
            bytes,
            kind: kind.into(),
        }
    }

    #[test]
    fn parses_header_and_rows() {
        let csv = "unix_nanos,role,event,conn_id,seq,bytes,kind\n\
                   1000,server,server_wire_input,1,0,514,real\n\
                   2000,server,server_wire_input,1,1,514,dummy\n";
        let recs = parse_reader(Cursor::new(csv));
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].bytes, 514);
        assert_eq!(recs[1].kind, "dummy");
    }

    #[test]
    fn malformed_lines_are_skipped() {
        let csv = "unix_nanos,role,event,conn_id,seq,bytes,kind\n\
                   not,a,valid,row\n\
                   3000,server,server_wire_input,1,0,514,real\n";
        let recs = parse_reader(Cursor::new(csv));
        assert_eq!(recs.len(), 1);
    }

    #[test]
    fn ks_zero_for_identical_sizes() {
        // Real and dummy frames padded to the same size -> indistinguishable.
        let mut recs = Vec::new();
        for i in 0..50 {
            recs.push(rec(i * 1000, "server_wire_input", 1, 514, "real"));
            recs.push(rec(i * 1000 + 500, "server_wire_input", 1, 514, "dummy"));
        }
        let a = size_attack(&recs, &EventNames::server(), AttackConfig::default()).unwrap();
        assert!(a.ks_statistic < 1e-9, "ks should be 0, got {}", a.ks_statistic);
        assert!(!a.distinguishable);
        assert!((a.best_threshold_accuracy - 0.5).abs() < 1e-9);
    }

    #[test]
    fn ks_detects_size_difference() {
        // Classic failure: 1200-byte reals, 65535-byte dummies.
        let mut recs = Vec::new();
        for i in 0..50 {
            recs.push(rec(i * 1000, "server_wire_input", 1, 1200, "real"));
            recs.push(rec(i * 1000 + 500, "server_wire_input", 1, 65535, "dummy"));
        }
        let a = size_attack(&recs, &EventNames::server(), AttackConfig::default()).unwrap();
        assert!(a.ks_statistic > 0.9, "ks {}", a.ks_statistic);
        assert!(a.distinguishable);
        assert!(a.best_threshold_accuracy > 0.99);
    }

    #[test]
    fn volume_attack_flags_correlated_cover() {
        // Wire volume that exactly mirrors real volume: an on/off pattern with
        // no dummy fill during idle -> the wire leaks the real envelope.
        let mut recs = Vec::new();
        let bin = 100_000_000u128; // 100ms in ns
        for b in 0..40u128 {
            let on = (b / 5) % 2 == 0; // 5 bins on, 5 off
            if on {
                for k in 0..4 {
                    let t = b * bin + k * 1000;
                    recs.push(rec(t, "server_wire_input", 1, 514, "real"));
                    recs.push(rec(t + 10, "server_real_output", 1, 500, "real"));
                }
            }
        }
        let ids = distinct_conn_ids(&recs, &EventNames::server());
        let a = volume_attack(&recs, &EventNames::server(), AttackConfig::default(), &ids).unwrap();
        assert!(a.leaks, "pearson {} peak {}", a.pearson, a.peak_cross_corr);
    }

    #[test]
    fn volume_attack_clears_constant_cover() {
        // Constant wire rate regardless of real activity: real is bursty but the
        // wire emits a steady stream -> low correlation, defence holds.
        let mut recs = Vec::new();
        let bin = 100_000_000u128;
        for b in 0..40u128 {
            // Steady wire: 4 frames every bin, always.
            for k in 0..4 {
                recs.push(rec(b * bin + k * 1000, "server_wire_input", 1, 514, "real"));
            }
            // Bursty real: only during "on" bins.
            if (b / 5) % 2 == 0 {
                recs.push(rec(b * bin + 50, "server_real_output", 1, 500, "real"));
            }
        }
        let ids = distinct_conn_ids(&recs, &EventNames::server());
        let a = volume_attack(&recs, &EventNames::server(), AttackConfig::default(), &ids).unwrap();
        assert!(!a.leaks, "pearson {} peak {}", a.pearson, a.peak_cross_corr);
    }

    #[test]
    fn linking_attack_links_distinct_flows() {
        // Two flows active over the SAME window but with independent burst
        // patterns. The wire mirrors each flow's real timing, so each wire flow
        // correlates with its own real flow and not the other's.
        let mut recs = Vec::new();
        let bin = 100_000_000u128;
        // Distinct pseudo-random on/off patterns for the two flows.
        let pat1 = [1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1];
        let pat2 = [1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0];
        for (b, (&a, &c)) in pat1.iter().zip(pat2.iter()).enumerate() {
            let t = b as u128 * bin;
            if a == 1 {
                recs.push(rec(t, "server_wire_input", 1, 514, "real"));
                recs.push(rec(t + 10, "server_real_output", 1, 500, "real"));
            }
            if c == 1 {
                recs.push(rec(t, "server_wire_input", 2, 514, "real"));
                recs.push(rec(t + 10, "server_real_output", 2, 500, "real"));
            }
        }
        let conn_ids = distinct_conn_ids(&recs, &EventNames::server());
        let a =
            linking_attack(&recs, &EventNames::server(), AttackConfig::default(), &conn_ids).unwrap();
        assert_eq!(a.flows, 2);
        assert!(a.linking_accuracy > 0.99, "acc {}", a.linking_accuracy);
        assert!(a.beats_chance);
    }

    #[test]
    fn pearson_basic() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [2.0, 4.0, 6.0, 8.0];
        assert!((pearson(&a, &b) - 1.0).abs() < 1e-9);
        let c = [4.0, 3.0, 2.0, 1.0];
        assert!((pearson(&a, &c) + 1.0).abs() < 1e-9);
    }

    #[test]
    fn cross_correlation_finds_lag() {
        // b is a shifted by +2 bins; peak should be at lag 2.
        let a = [0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0];
        let b = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let (lag, score) = peak_cross_correlation(&a, &b, 4);
        assert_eq!(lag, 2);
        assert!(score > 0.9, "score {score}");
    }

    #[test]
    fn total_volume_attack_flags_proportional_totals() {
        // Each flow's total wire volume scales with its real volume — the
        // classic integral leak (e.g. queue backlog under sustained load).
        let mut recs = Vec::new();
        for id in 1..=4u64 {
            for k in 0..(id * 5) {
                let t = id as u128 * 1_000 + k as u128;
                recs.push(rec(t, "server_wire_input", id, 514, "real"));
                recs.push(rec(t, "server_real_output", id, 500, "real"));
            }
        }
        let ids = distinct_conn_ids(&recs, &EventNames::server());
        let a = total_volume_attack(&recs, &EventNames::server(), AttackConfig::default(), &ids)
            .unwrap();
        assert!(a.leaks, "pearson {} spearman {}", a.pearson_total, a.spearman_total);
    }

    #[test]
    fn total_volume_attack_clears_constant_wire_totals() {
        // Constant total wire volume regardless of real volume — the integral
        // is decoupled, so the cross-flow correlation must not register.
        let mut recs = Vec::new();
        for id in 1..=4u64 {
            for k in 0..20 {
                recs.push(rec(1_000 + k, "server_wire_input", id, 514, "real"));
            }
            for k in 0..(id * 3) {
                recs.push(rec(1_001 + k as u128, "server_real_output", id, 500, "real"));
            }
        }
        let ids = distinct_conn_ids(&recs, &EventNames::server());
        let a = total_volume_attack(&recs, &EventNames::server(), AttackConfig::default(), &ids)
            .unwrap();
        assert!(!a.leaks, "pearson {} spearman {}", a.pearson_total, a.spearman_total);
    }

    #[test]
    fn spearman_detects_monotone_nonlinear() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [1.0, 4.0, 9.0, 16.0, 25.0]; // monotone but nonlinear
        assert!((spearman(&a, &b) - 1.0).abs() < 1e-9);
        // Pearson is < 1 on the same data (nonlinearity), so spearman adds signal.
        assert!(pearson(&a, &b) < 1.0);
    }

    #[test]
    fn duration_leak_flags_matching_spans() {
        // Wire activity spans essentially the same window as real activity.
        let mut recs = Vec::new();
        for id in 1..=2u64 {
            recs.push(rec(0, "server_wire_input", id, 514, "real"));
            recs.push(rec(1_000_000_000, "server_wire_input", id, 514, "dummy"));
            recs.push(rec(10_000_000, "server_real_output", id, 500, "real"));
            recs.push(rec(990_000_000, "server_real_output", id, 500, "real"));
        }
        let ids = distinct_conn_ids(&recs, &EventNames::server());
        let d = duration_leak(&recs, &EventNames::server(), &ids).unwrap();
        assert!(d.leaks, "median rel gap {}", d.median_rel_gap);
    }
}
