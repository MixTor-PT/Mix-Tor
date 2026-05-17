use clap::Parser;
use mixtor::lab::LabLogger;
use mixtor::transport::handle_server_connection_with_lab;
use std::error::Error;
use std::io::{self, Write};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::TcpListener;

#[derive(Debug, Parser)]
#[command(about = "MixTor server transport proxy")]
struct Args {
    /// Public address MixTor clients connect to.
    #[arg(long, default_value = "0.0.0.0:9001")]
    listen: SocketAddr,

    /// Local Tor bridge daemon address.
    #[arg(long, default_value = "127.0.0.1:9002")]
    upstream: SocketAddr,

    /// Maximum bytes read from the upstream bridge per real frame.
    #[arg(long, default_value_t = 1200)]
    max_read: usize,

    /// Run as a Tor managed server transport.
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
        run_managed_server(args.max_read, args.lab_log).await
    } else {
        run_raw_server(args.listen, args.upstream, args.max_read, args.lab_log).await
    }
}

async fn run_raw_server(
    listen: SocketAddr,
    upstream: SocketAddr,
    max_read: usize,
    lab_log: Option<PathBuf>,
) -> Result<(), Box<dyn Error>> {
    let lab = create_lab_logger(lab_log, "server")?;
    let listener = TcpListener::bind(listen).await?;
    eprintln!(
        "mixtor-server listening on {} and forwarding real traffic to {}",
        listener.local_addr()?,
        upstream
    );

    loop {
        let (client, peer) = listener.accept().await?;
        let lab = lab.clone();

        tokio::spawn(async move {
            if let Err(error) =
                handle_server_connection_with_lab(client, upstream, max_read, lab).await
            {
                eprintln!("server connection from {peer} closed: {error}");
            }
        });
    }
}

async fn run_managed_server(
    max_read: usize,
    lab_log: Option<PathBuf>,
) -> Result<(), Box<dyn Error>> {
    let lab = create_lab_logger(lab_log, "server")?;
    let transports = std::env::var("TOR_PT_SERVER_TRANSPORTS").unwrap_or_else(|_| "*".to_string());
    if transports != "*" && !transports.split(',').any(|transport| transport == "mixtor") {
        println!("VERSION 1");
        println!("SMETHODS DONE");
        io::stdout().flush()?;
        return Ok(());
    }

    let upstream: SocketAddr = std::env::var("TOR_PT_ORPORT")?.parse()?;
    let bind = managed_bind_addr().unwrap_or_else(|| "127.0.0.1:0".parse().expect("valid bind"));
    let listener = TcpListener::bind(bind).await?;
    let addr = listener.local_addr()?;

    println!("VERSION 1");
    println!("SMETHOD mixtor {addr}");
    println!("SMETHODS DONE");
    io::stdout().flush()?;

    eprintln!("mixtor-server managed listener on {addr}, upstream {upstream}");

    loop {
        let (client, peer) = listener.accept().await?;
        let lab = lab.clone();

        tokio::spawn(async move {
            if let Err(error) =
                handle_server_connection_with_lab(client, upstream, max_read, lab).await
            {
                eprintln!("managed server connection from {peer} closed: {error}");
            }
        });
    }
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

fn managed_bind_addr() -> Option<SocketAddr> {
    let bind_addrs = std::env::var("TOR_PT_SERVER_BINDADDR").ok()?;
    bind_addrs.split(',').find_map(|entry| {
        let (transport, addr) = entry.split_once('-')?;
        if transport == "mixtor" {
            addr.parse().ok()
        } else {
            None
        }
    })
}
