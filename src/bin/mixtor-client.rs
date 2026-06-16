use clap::Parser;
use mixtor::lab::LabLogger;
use mixtor::socks::accept_socks5_connect;
use mixtor::transport::{handle_client_connection_with_lab, EmitterHandle, TailPolicy};
use std::error::Error;
use std::io::{self, Write};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::net::TcpStream;

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

    /// LAB ONLY: write temporary traffic timing/size CSV logs to this directory.
    #[arg(long)]
    lab_log: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    if args.managed || std::env::var_os("TOR_PT_MANAGED_TRANSPORT_VER").is_some() {
        run_managed_client(args.max_read, args.lab_log).await
    } else {
        let server = args
            .server
            .ok_or("--server is required unless running as a managed transport")?;
        run_raw_client(args.listen, server, args.max_read, args.lab_log).await
    }
}

async fn run_raw_client(
    listen: SocketAddr,
    server: SocketAddr,
    max_read: usize,
    lab_log: Option<PathBuf>,
) -> Result<(), Box<dyn Error>> {
    let lab = create_lab_logger(lab_log, "client")?;
    let listener = TcpListener::bind(listen).await?;
    eprintln!(
        "mixtor-client listening on {} and forwarding to {}",
        listener.local_addr()?,
        server
    );

    // One shared emitter for every flow this process clocks (see EmitterHandle).
    let emitter = EmitterHandle::new();
    // Duration fix: production defaults to the SessionClass tail floor so a
    // connection's lifetime/volume don't track the session. Override (incl. off)
    // with MIXTOR_TAIL_FLOOR_MS.
    let tail = TailPolicy::from_env_or(TailPolicy::ByClass);
    loop {
        let (local, peer) = listener.accept().await?;
        let lab = lab.clone();
        let emitter = emitter.clone();

        tokio::spawn(async move {
            if let Err(error) =
                handle_client_connection_with_lab(local, server, max_read, lab, emitter, tail).await
            {
                eprintln!("client connection from {peer} closed: {error}");
            }
        });
    }
}

async fn run_managed_client(
    max_read: usize,
    lab_log: Option<PathBuf>,
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

    // One shared emitter for every flow this process clocks (see EmitterHandle).
    let emitter = EmitterHandle::new();
    let tail = TailPolicy::from_env_or(TailPolicy::ByClass);
    loop {
        let (local, peer) = listener.accept().await?;
        let lab = lab.clone();
        let emitter = emitter.clone();

        tokio::spawn(async move {
            if let Err(error) = handle_managed_socks_connection(local, max_read, lab, emitter, tail).await {
                eprintln!("managed client connection from {peer} closed: {error}");
            }
        });
    }
}

async fn handle_managed_socks_connection(
    mut local: TcpStream,
    max_read: usize,
    lab: Option<Arc<LabLogger>>,
    emitter: EmitterHandle,
    tail: TailPolicy,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let server = accept_socks5_connect(&mut local).await?;
    handle_client_connection_with_lab(local, server, max_read, lab, emitter, tail).await
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
