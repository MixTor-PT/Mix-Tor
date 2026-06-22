use clap::Parser;
use mixtor::crypto;
use mixtor::lab::LabLogger;
use mixtor::socks::accept_socks5_connect;
use mixtor::transport::{EmitterHandle, MuxClientHandle, TailPolicy};
use std::error::Error;
use std::io::{self, Write};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::net::TcpStream;

// Windows timer resolution fix (Fix B): the default OS timer granularity on
// Windows is ~15 ms. The emitter thread wakes on EMIT_TICK = 1 ms, so without
// this the cover clock fires at ~15 ms rather than ~1 ms — coarse enough that
// in busy windows the emitter simply can't keep up, cover events cluster, and
// the wire rate drops in active windows. This re-couples the wire envelope to
// real activity at exactly the coarse timescale an adversary watches.
// `timeBeginPeriod(1)` raises the system-wide timer resolution for this process
// to 1 ms. It persists for the process lifetime (Windows reverts at exit).
// This call is at the process level and takes effect before the emitter thread
// is spawned, so the emitter's `tokio::time::sleep(EMIT_TICK)` wakes at 1 ms.
#[cfg(windows)]
#[link(name = "winmm")]
extern "system" {
    fn timeBeginPeriod(uperiod: u32) -> u32;
}

#[cfg(windows)]
fn raise_timer_resolution() {
    unsafe { timeBeginPeriod(1); }
}

#[cfg(not(windows))]
fn raise_timer_resolution() {}

#[derive(Debug, Parser)]
#[command(about = "MixTor client transport proxy")]
struct Args {
    /// Local address Tor should connect to.
    #[arg(long, default_value = "127.0.0.1:9000")]
    listen: SocketAddr,

    /// Remote MixTor server address.
    #[arg(long)]
    server: Option<SocketAddr>,

    /// Maximum bytes read from local Tor per packet.
    #[arg(long, default_value_t = 1200)]
    max_read: usize,

    /// Run as a Tor managed client transport.
    #[arg(long)]
    managed: bool,

    /// Hex-encoded 32-byte server identity PUBLIC key (the bridge `cert=`).
    /// Required in raw mode. In managed mode it normally comes from the Bridge
    /// line's `cert=` (via SOCKS); this is only a fallback. Env: MIXTOR_SERVER_KEY.
    #[arg(long, env = "MIXTOR_SERVER_KEY")]
    server_key: Option<String>,

    /// LAB ONLY: write temporary traffic timing/size CSV logs to this directory.
    #[arg(long)]
    lab_log: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    raise_timer_resolution();
    let args = Args::parse();

    // Optional in managed mode: the bridge's identity normally arrives per
    // connection from the `cert=` on the Bridge line (via SOCKS). `--server-key`
    // / MIXTOR_SERVER_KEY is a fallback (and required in raw mode).
    let server_pub = match args.server_key.as_deref() {
        Some(hex) => Some(crypto::key_from_hex(hex).ok_or("--server-key must be 64 hex chars (32 bytes)")?),
        None => None,
    };

    if args.managed || std::env::var_os("TOR_PT_MANAGED_TRANSPORT_VER").is_some() {
        run_managed_client(args.max_read, args.lab_log, server_pub).await
    } else {
        let server = args
            .server
            .ok_or("--server is required unless running as a managed transport")?;
        let server_pub = server_pub.ok_or("--server-key is required in raw mode")?;
        run_raw_client(args.listen, server, args.max_read, args.lab_log, server_pub).await
    }
}

async fn run_raw_client(
    listen: SocketAddr,
    server: SocketAddr,
    max_read: usize,
    lab_log: Option<PathBuf>,
    server_pub: [u8; 32],
) -> Result<(), Box<dyn Error>> {
    let lab = create_lab_logger(lab_log, "client")?;
    let listener = TcpListener::bind(listen).await?;
    eprintln!(
        "mixtor-client listening on {} and forwarding to {}",
        listener.local_addr()?,
        server
    );

    // One shared emitter and one persistent mux wire for every Tor circuit this
    // process handles. Circuits are multiplexed over a single long-lived shaped
    // TCP connection to the server, hiding individual circuit start/stop events
    // — and the wire's duration/volume from the single flow's.
    let emitter = EmitterHandle::new();
    let tail = TailPolicy::default_for_production();
    let mux = MuxClientHandle::new(max_read, lab.clone(), emitter, tail);

    // Pre-warm the wire so cover begins flowing from process start, before any
    // circuit arrives — the wire's lifetime then reflects client online time,
    // not the first circuit's. Best-effort: lazily retried on the first circuit.
    if let Err(error) = mux.prewarm(server, server_pub).await {
        eprintln!("mixtor-client: initial wire to {server} not up yet ({error}); will retry on first circuit");
    }

    loop {
        let (local, peer) = listener.accept().await?;
        let mux = mux.clone();

        tokio::spawn(async move {
            if let Err(error) = mux.connect_circuit(local, server, server_pub).await {
                eprintln!("client connection from {peer} closed: {error}");
            }
        });
    }
}

async fn run_managed_client(
    max_read: usize,
    lab_log: Option<PathBuf>,
    fallback_pub: Option<[u8; 32]>,
) -> Result<(), Box<dyn Error>> {
    let lab = create_lab_logger(lab_log, "client")?;
    let transports = std::env::var("TOR_PT_CLIENT_TRANSPORTS").unwrap_or_else(|_| "*".to_string());
    if transports != "*" && !transports.split(',').any(|transport| transport == "mixtor") {
        println!("VERSION 1");
        println!("CMETHODS DONE");
        io::stdout().flush()?;
        return Ok(());
    }

    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;

    println!("VERSION 1");
    println!("CMETHOD mixtor socks5 {addr}");
    println!("CMETHODS DONE");
    io::stdout().flush()?;

    eprintln!("mixtor-client managed SOCKS5 listener on {addr}");

    // Managed PT is the real deployment path: route it through the same
    // persistent mux as raw mode so circuit start/stop hiding is active here
    // too. The bridge address comes from each SOCKS request; the handle keeps
    // one long-lived wire per distinct address (normally a single bridge).
    let emitter = EmitterHandle::new();
    let tail = TailPolicy::default_for_production();
    let mux = MuxClientHandle::new(max_read, lab.clone(), emitter, tail);
    loop {
        let (local, peer) = listener.accept().await?;
        let mux = mux.clone();

        tokio::spawn(async move {
            if let Err(error) = handle_managed_socks_connection(local, mux, fallback_pub).await {
                eprintln!("managed client connection from {peer} closed: {error}");
            }
        });
    }
}

async fn handle_managed_socks_connection(
    mut local: TcpStream,
    mux: MuxClientHandle,
    fallback_pub: Option<[u8; 32]>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    // The bridge's identity key arrives as `cert=<hex>` on the Bridge line,
    // which Tor hands us through the SOCKS auth fields. Fall back to
    // --server-key / MIXTOR_SERVER_KEY only if the line carried no cert.
    let (server, socks_args) = accept_socks5_connect(&mut local).await?;
    let server_pub = socks_args.cert.or(fallback_pub).ok_or(
        "no bridge identity: add `cert=<hex>` to the Bridge line (or set MIXTOR_SERVER_KEY)",
    )?;
    mux.connect_circuit(local, server, server_pub).await
}

fn create_lab_logger(
    lab_log: Option<PathBuf>,
    role: &'static str,
) -> Result<Option<Arc<LabLogger>>, Box<dyn Error>> {
    match lab_log {
        Some(dir) => {
            eprintln!("WARNING: LAB LOGGING ENABLED. Do not use this for real privacy traffic.");
            Ok(Some(LabLogger::create(dir, role)?))
        }
        None => Ok(None),
    }
}
