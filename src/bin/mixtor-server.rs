use clap::Parser;
use mixtor::crypto::{self, StaticKeypair};
use mixtor::lab::LabLogger;
use mixtor::transport::{handle_server_connection_with_lab, EmitterHandle, TailPolicy};
use std::error::Error;
use std::io::{self, Write};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::TcpListener;

// See the identical block in mixtor-client.rs for the full explanation.
// tl;dr: the default Windows timer granularity is ~15 ms; this raises it to
// 1 ms so the emitter thread's 1 ms sleep wakes on time and the cover clock
// stays flat under load on Windows.
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

    /// Path to a file containing the hex-encoded 32-byte identity SECRET (the
    /// PREFERRED way to pin an identity — keeps the secret off the command line
    /// and out of torrc). Created with owner-only perms by --keygen-file.
    #[arg(long)]
    static_key_file: Option<PathBuf>,

    /// Hex-encoded 32-byte static identity SECRET key. DISCOURAGED: command lines
    /// and torrc exec lines are easy to expose — prefer --static-key-file or the
    /// auto-persisted state-dir identity. Env: MIXTOR_STATIC_KEY.
    #[arg(long, env = "MIXTOR_STATIC_KEY")]
    static_key: Option<String>,

    /// Generate a fresh identity, print its secret + cert, and exit. Use the
    /// secret with --static-key-file on the bridge and the cert on the client's
    /// `Bridge mixtor <addr> <fp> cert=<cert>` line.
    #[arg(long)]
    keygen: bool,

    /// Generate a fresh identity, write the secret to this path with owner-only
    /// permissions, print the cert, and exit. Pass the same path to the bridge
    /// via --static-key-file.
    #[arg(long, value_name = "PATH")]
    keygen_file: Option<PathBuf>,

    /// LAB ONLY: write temporary traffic timing/size CSV logs to this directory.
    #[arg(long)]
    lab_log: Option<PathBuf>,
}

/// Resolve the server's identity keypair, in priority order:
///   1. `--static-key-file <path>` (preferred pin — secret stays off the CLI),
///   2. `--static-key <hex secret>` (discouraged), else
///   3. a persisted `mixtor_identity` in `state_dir` (the PT state location), so
///      the bridge's `cert=` is STABLE across restarts — generated and saved on
///      first run, else
///   4. a fresh ephemeral identity (raw mode with no state dir / no flag).
/// Prints the public key (the `cert=`) so the operator can build the Bridge line.
fn load_identity(
    static_key_file: Option<PathBuf>,
    static_key: Option<String>,
    state_dir: Option<PathBuf>,
) -> Result<Arc<StaticKeypair>, Box<dyn Error>> {
    let kp = if let Some(path) = static_key_file {
        warn_if_world_readable(&path);
        let s = std::fs::read_to_string(&path)
            .map_err(|e| format!("--static-key-file {}: {e}", path.display()))?;
        let secret = crypto::key_from_hex(s.trim())
            .ok_or("--static-key-file must contain 64 hex chars (32 bytes)")?;
        StaticKeypair::from_secret_bytes(secret)
    } else if let Some(hex) = static_key {
        eprintln!("mixtor-server: WARNING --static-key exposes the secret on the command line / torrc; prefer --static-key-file");
        let secret =
            crypto::key_from_hex(&hex).ok_or("--static-key must be 64 hex chars (32 bytes)")?;
        StaticKeypair::from_secret_bytes(secret)
    } else if let Some(dir) = state_dir.as_ref() {
        let path = dir.join("mixtor_identity");
        match std::fs::read_to_string(&path) {
            Ok(s) => {
                warn_if_world_readable(&path);
                let secret = crypto::key_from_hex(s.trim())
                    .ok_or("corrupt mixtor_identity (expected 64 hex chars) in PT state dir")?;
                StaticKeypair::from_secret_bytes(secret)
            }
            Err(_) => {
                let kp = StaticKeypair::generate();
                let _ = std::fs::create_dir_all(dir);
                let hex = crypto::key_to_hex(&kp.secret_bytes());
                match write_secret_file(&path, &hex) {
                    Ok(()) => {}
                    // Lost a create race with another instance: adopt theirs.
                    Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                        let s = std::fs::read_to_string(&path)?;
                        let secret = crypto::key_from_hex(s.trim())
                            .ok_or("corrupt mixtor_identity in PT state dir")?;
                        return finish(StaticKeypair::from_secret_bytes(secret));
                    }
                    Err(e) => eprintln!(
                        "mixtor-server: WARNING could not persist identity to {} ({e}); cert will change on restart",
                        path.display()
                    ),
                }
                kp
            }
        }
    } else {
        StaticKeypair::generate()
    };

    finish(kp)
}

/// Print the cert and wrap the identity. Shared so the create-race path can
/// return early with the same reporting.
fn finish(kp: StaticKeypair) -> Result<Arc<StaticKeypair>, Box<dyn Error>> {
    let cert = crypto::key_to_hex(&kp.public_bytes());
    eprintln!("mixtor-server identity public key (cert): {cert}");
    eprintln!("  → give clients a Bridge line: Bridge mixtor <ip:port> <fingerprint> cert={cert}");
    Ok(Arc::new(kp))
}

/// Write the secret key with owner-only permissions FROM CREATION — the secret
/// is never on disk under broad permissions:
///   * unix: `create_new` (atomic) + mode 0600, then write.
///   * windows: `create_new` an EMPTY file, lock its ACL to the current user
///     (`icacls`), and only THEN write the secret. FAIL CLOSED — if the ACL
///     lock fails, delete the empty file and return an error rather than persist
///     a secret under inherited ACLs.
#[cfg(unix)]
fn write_secret_file(path: &std::path::Path, hex: &str) -> std::io::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)?;
    f.write_all(hex.as_bytes())?;
    f.flush()
}

#[cfg(windows)]
fn write_secret_file(path: &std::path::Path, hex: &str) -> std::io::Result<()> {
    use std::io::Write;
    // 1. Create empty (atomic; fails if it already exists).
    drop(std::fs::OpenOptions::new().write(true).create_new(true).open(path)?);
    // 2. Lock the ACL while the file is still empty.
    if !restrict_windows_acl(path) {
        let _ = std::fs::remove_file(path);
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "could not restrict identity file ACL to the current user",
        ));
    }
    // 3. Now write the secret, under the restricted ACL (write preserves it).
    let mut f = std::fs::OpenOptions::new().write(true).truncate(true).open(path)?;
    f.write_all(hex.as_bytes())?;
    f.flush()
}

/// Strip inherited ACEs and grant the file only to the current user via
/// `icacls`. Returns whether it succeeded (exit status 0).
#[cfg(windows)]
fn restrict_windows_acl(path: &std::path::Path) -> bool {
    let user = std::env::var("USERNAME").unwrap_or_default();
    if user.is_empty() {
        return false;
    }
    std::process::Command::new("icacls")
        .arg(path)
        .arg("/inheritance:r")
        .arg("/grant:r")
        .arg(format!("{user}:F"))
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Warn if the persisted secret is readable by group/other (unix only).
#[cfg(unix)]
fn warn_if_world_readable(path: &std::path::Path) {
    use std::os::unix::fs::PermissionsExt;
    if let Ok(meta) = std::fs::metadata(path) {
        if meta.permissions().mode() & 0o077 != 0 {
            eprintln!(
                "mixtor-server: WARNING {} is accessible beyond its owner; run: chmod 600 {0}",
                path.display()
            );
        }
    }
}
#[cfg(not(unix))]
fn warn_if_world_readable(_path: &std::path::Path) {}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    raise_timer_resolution();
    let args = Args::parse();

    if args.keygen {
        let kp = StaticKeypair::generate();
        println!("secret={}", crypto::key_to_hex(&kp.secret_bytes()));
        println!("cert={}", crypto::key_to_hex(&kp.public_bytes()));
        return Ok(());
    }
    if let Some(path) = args.keygen_file {
        let kp = StaticKeypair::generate();
        // Write the secret with owner-only perms FROM CREATION.
        write_secret_file(&path, &crypto::key_to_hex(&kp.secret_bytes()))
            .map_err(|e| format!("could not write key file {}: {e}", path.display()))?;
        eprintln!("wrote secret (owner-only) to {}", path.display());
        println!("cert={}", crypto::key_to_hex(&kp.public_bytes()));
        return Ok(());
    }

    // Tor sets TOR_PT_STATE_LOCATION in managed mode; absent in raw mode.
    let state_dir = std::env::var_os("TOR_PT_STATE_LOCATION").map(PathBuf::from);
    let identity = load_identity(args.static_key_file, args.static_key, state_dir)?;

    if args.managed || std::env::var_os("TOR_PT_MANAGED_TRANSPORT_VER").is_some() {
        run_managed_server(args.max_read, args.lab_log, identity).await
    } else {
        run_raw_server(args.listen, args.upstream, args.max_read, args.lab_log, identity).await
    }
}

async fn run_raw_server(
    listen: SocketAddr,
    upstream: SocketAddr,
    max_read: usize,
    lab_log: Option<PathBuf>,
    identity: Arc<StaticKeypair>,
) -> Result<(), Box<dyn Error>> {
    let lab = create_lab_logger(lab_log, "server")?;
    let listener = TcpListener::bind(listen).await?;
    eprintln!(
        "mixtor-server listening on {} and forwarding real traffic to {}",
        listener.local_addr()?,
        upstream
    );

    // Epoch tail (in serve_accept_loop): all connections sharing the same clock
    // enter and exit at epoch boundaries — batch mixing hides which ISP-side
    // client maps to which bridge connection. Override with MIXTOR_TAIL_FLOOR_MS.
    serve_accept_loop(listener, upstream, max_read, lab, identity, "server").await
}

/// Max number of in-progress (not-yet-completed) handshakes. A flood of half-
/// open or bad handshakes — each held up to HANDSHAKE_TIMEOUT plus a blackhole —
/// cannot exhaust the process: beyond this, new connections are shed at accept.
const MAX_CONCURRENT_HANDSHAKES: usize = 512;

/// Shared accept loop: bound concurrent handshakes with a semaphore, then hand
/// each accepted connection to the transport (which releases the permit once the
/// handshake completes, so established connections don't count against the cap).
async fn serve_accept_loop(
    listener: TcpListener,
    upstream: SocketAddr,
    max_read: usize,
    lab: Option<Arc<LabLogger>>,
    identity: Arc<StaticKeypair>,
    label: &'static str,
) -> Result<(), Box<dyn Error>> {
    let emitter = EmitterHandle::new();
    let tail = TailPolicy::default_for_production();
    let handshakes = Arc::new(tokio::sync::Semaphore::new(MAX_CONCURRENT_HANDSHAKES));
    loop {
        let (client, peer) = listener.accept().await?;
        // Shed load if too many handshakes are already in flight.
        let permit = match handshakes.clone().try_acquire_owned() {
            Ok(p) => p,
            Err(_) => continue, // drops `client` → connection closed
        };
        let lab = lab.clone();
        let emitter = emitter.clone();
        let tail = tail.clone();
        let identity = identity.clone();

        tokio::spawn(async move {
            if let Err(error) = handle_server_connection_with_lab(
                client, upstream, max_read, lab, emitter, tail, identity, Some(permit),
            )
            .await
            {
                eprintln!("{label} connection from {peer} closed: {error}");
            }
        });
    }
}

async fn run_managed_server(
    max_read: usize,
    lab_log: Option<PathBuf>,
    identity: Arc<StaticKeypair>,
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

    // Advertise the identity as the transport ARG `cert=<hex>`. Tor records it
    // in the bridge's descriptor so it propagates to clients as the `cert=` on
    // the Bridge line (and the operator can read it from the startup log).
    let cert = crypto::key_to_hex(&identity.public_bytes());
    println!("VERSION 1");
    println!("SMETHOD mixtor {addr} ARGS:cert={cert}");
    println!("SMETHODS DONE");
    io::stdout().flush()?;

    eprintln!("mixtor-server managed listener on {addr}, upstream {upstream} (cert={cert})");

    serve_accept_loop(listener, upstream, max_read, lab, identity, "managed server").await
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
