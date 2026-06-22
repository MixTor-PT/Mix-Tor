use std::fmt;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

const SOCKS_VERSION: u8 = 0x05;
const NO_AUTH: u8 = 0x00;
const USERPASS_AUTH: u8 = 0x02; // RFC 1929 — Tor passes PT bridge args here
const USERPASS_VERSION: u8 = 0x01;
const CONNECT: u8 = 0x01;
const IPV4: u8 = 0x01;
const DOMAIN: u8 = 0x03;
const IPV6: u8 = 0x04;

/// Per-connection transport arguments Tor passes to the client PT through the
/// SOCKS5 auth fields (the `cert=`/etc. from a `Bridge mixtor … cert=…` line).
#[derive(Clone, Debug, Default)]
pub struct ClientArgs {
    /// The bridge's static identity public key, parsed from `cert=<64 hex>`.
    pub cert: Option<[u8; 32]>,
}

/// Accept a SOCKS5 CONNECT from the local Tor, returning the requested target
/// **and** any per-connection PT args Tor supplied via the auth fields.
///
/// Tor carries per-bridge transport parameters (for MixTor, `cert=<server
/// identity hex>`) in the SOCKS5 username/password auth (RFC 1929): the username
/// and password are concatenated to reconstruct the parameter string. We offer
/// username/password auth when the client lists it (so the cert reaches us) and
/// fall back to no-auth otherwise (the caller then needs the key from
/// `--server-key`/env).
pub async fn accept_socks5_connect(
    stream: &mut TcpStream,
) -> Result<(SocketAddr, ClientArgs), SocksError> {
    let args = negotiate(stream).await?;
    let target = read_connect_target(stream).await?;
    stream
        .write_all(&[SOCKS_VERSION, 0x00, 0x00, IPV4, 0, 0, 0, 0, 0, 0])
        .await?;
    Ok((target, args))
}

async fn negotiate(stream: &mut TcpStream) -> Result<ClientArgs, SocksError> {
    let mut header = [0u8; 2];
    stream.read_exact(&mut header).await?;
    if header[0] != SOCKS_VERSION {
        return Err(SocksError::UnsupportedVersion(header[0]));
    }

    let method_count = header[1] as usize;
    if method_count == 0 {
        return Err(SocksError::NoSupportedAuthMethod);
    }

    let mut methods = vec![0u8; method_count];
    stream.read_exact(&mut methods).await?;

    // Prefer username/password — that's how Tor hands us the bridge args (cert).
    if methods.contains(&USERPASS_AUTH) {
        stream.write_all(&[SOCKS_VERSION, USERPASS_AUTH]).await?;
        return read_userpass_args(stream).await;
    }
    if methods.contains(&NO_AUTH) {
        stream.write_all(&[SOCKS_VERSION, NO_AUTH]).await?;
        return Ok(ClientArgs::default());
    }

    stream.write_all(&[SOCKS_VERSION, 0xff]).await?;
    Err(SocksError::NoSupportedAuthMethod)
}

/// RFC 1929 username/password sub-negotiation. Tor packs the PT parameter string
/// into the username (and overflow into the password); we concatenate them and
/// pull out `cert=<hex>`. We always answer "success" — the parameters are not a
/// password to verify, just a carrier for the bridge args.
async fn read_userpass_args(stream: &mut TcpStream) -> Result<ClientArgs, SocksError> {
    let mut ver = [0u8; 1];
    stream.read_exact(&mut ver).await?;
    if ver[0] != USERPASS_VERSION {
        return Err(SocksError::UnsupportedVersion(ver[0]));
    }
    let mut ulen = [0u8; 1];
    stream.read_exact(&mut ulen).await?;
    let mut uname = vec![0u8; ulen[0] as usize];
    stream.read_exact(&mut uname).await?;
    let mut plen = [0u8; 1];
    stream.read_exact(&mut plen).await?;
    let mut passwd = vec![0u8; plen[0] as usize];
    stream.read_exact(&mut passwd).await?;

    // Auth "success".
    stream.write_all(&[USERPASS_VERSION, 0x00]).await?;

    let mut params = uname;
    params.extend_from_slice(&passwd);
    Ok(ClientArgs {
        cert: extract_cert(&params),
    })
}

/// Pull the 32-byte identity key out of a `cert=<64 hex>` parameter. Robust to
/// the surrounding separators (`;`, `,`, space, NUL) Tor may use: it locates the
/// `cert=` key and reads the run of hex digits that follows.
fn extract_cert(params: &[u8]) -> Option<[u8; 32]> {
    let s = String::from_utf8_lossy(params);
    let start = s.find("cert=")? + 5;
    let hex: String = s[start..]
        .chars()
        .take_while(|c| c.is_ascii_hexdigit())
        .take(64)
        .collect();
    crate::crypto::key_from_hex(&hex)
}

async fn read_connect_target(stream: &mut TcpStream) -> Result<SocketAddr, SocksError> {
    let mut header = [0u8; 4];
    stream.read_exact(&mut header).await?;
    if header[0] != SOCKS_VERSION {
        return Err(SocksError::UnsupportedVersion(header[0]));
    }

    if header[1] != CONNECT {
        return Err(SocksError::UnsupportedCommand(header[1]));
    }

    // Parse the address directly to an `IpAddr` — never resolve here. For a
    // bridge the target IS the bridge, and a DOMAIN target would have to be
    // resolved via the local DNS, leaking the bridge hostname. Tor must address
    // the bridge by IP (the norm for `Bridge` lines), so reject DOMAIN.
    let ip = match header[3] {
        IPV4 => {
            let mut octets = [0u8; 4];
            stream.read_exact(&mut octets).await?;
            IpAddr::V4(Ipv4Addr::from(octets))
        }
        IPV6 => {
            let mut octets = [0u8; 16];
            stream.read_exact(&mut octets).await?;
            IpAddr::V6(Ipv6Addr::from(octets))
        }
        DOMAIN => {
            // Drain the (length-prefixed) host + port so the framing is consumed,
            // then refuse rather than resolve it locally.
            let mut len = [0u8; 1];
            stream.read_exact(&mut len).await?;
            let mut rest = vec![0u8; len[0] as usize + 2];
            stream.read_exact(&mut rest).await?;
            return Err(SocksError::DomainTargetRejected);
        }
        kind => return Err(SocksError::UnsupportedAddressKind(kind)),
    };

    let mut port = [0u8; 2];
    stream.read_exact(&mut port).await?;
    Ok(SocketAddr::new(ip, u16::from_be_bytes(port)))
}

#[derive(Debug)]
pub enum SocksError {
    Io(std::io::Error),
    DomainTargetRejected,
    NoSupportedAuthMethod,
    UnsupportedAddressKind(u8),
    UnsupportedCommand(u8),
    UnsupportedVersion(u8),
}

impl fmt::Display for SocksError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "SOCKS I/O error: {error}"),
            Self::DomainTargetRejected => write!(
                formatter,
                "SOCKS domain target rejected (bridge must be addressed by IP to avoid a DNS leak)"
            ),
            Self::NoSupportedAuthMethod => write!(formatter, "SOCKS client offered no supported auth method"),
            Self::UnsupportedAddressKind(kind) => {
                write!(formatter, "unsupported SOCKS address kind {kind}")
            }
            Self::UnsupportedCommand(command) => {
                write!(formatter, "unsupported SOCKS command {command}")
            }
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported SOCKS version {version}")
            }
        }
    }
}

impl std::error::Error for SocksError {}

impl From<std::io::Error> for SocksError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::TcpListener;

    #[tokio::test]
    async fn accepts_ipv4_connect() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let addr = listener.local_addr().expect("addr should exist");

        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.expect("accept should succeed");
            accept_socks5_connect(&mut stream)
                .await
                .expect("SOCKS handshake should succeed")
        });

        let mut client = TcpStream::connect(addr)
            .await
            .expect("client should connect");
        client
            .write_all(&[0x05, 0x01, 0x00])
            .await
            .expect("greeting should write");
        let mut response = [0u8; 2];
        client
            .read_exact(&mut response)
            .await
            .expect("greeting response should read");
        assert_eq!(response, [0x05, 0x00]);

        client
            .write_all(&[0x05, 0x01, 0x00, 0x01, 127, 0, 0, 1, 0x23, 0x29])
            .await
            .expect("connect should write");
        let mut response = [0u8; 10];
        client
            .read_exact(&mut response)
            .await
            .expect("connect response should read");
        assert_eq!(response[1], 0x00);

        let (target, args) = server.await.expect("server should join");
        assert_eq!(target, SocketAddr::from(([127, 0, 0, 1], 9001)));
        assert!(args.cert.is_none(), "no-auth path carries no cert");
    }

    #[tokio::test]
    async fn reads_cert_from_userpass_args() {
        // Simulate Tor passing `cert=<hex>` via SOCKS5 username/password auth.
        let cert = [0x5au8; 32];
        let cert_hex = crate::crypto::key_to_hex(&cert);
        let params = format!("cert={cert_hex};iat-mode=0");

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut s, _) = listener.accept().await.unwrap();
            accept_socks5_connect(&mut s).await.unwrap()
        });

        let mut client = TcpStream::connect(addr).await.unwrap();
        // Greeting: offer no-auth AND username/password.
        client.write_all(&[0x05, 0x02, 0x00, 0x02]).await.unwrap();
        let mut resp = [0u8; 2];
        client.read_exact(&mut resp).await.unwrap();
        assert_eq!(resp, [0x05, USERPASS_AUTH], "server must pick username/password");

        // RFC 1929: VER ULEN UNAME PLEN PASSWD. Put all params in the username.
        let pb = params.as_bytes();
        let mut auth = vec![USERPASS_VERSION, pb.len() as u8];
        auth.extend_from_slice(pb);
        auth.push(0); // empty password
        client.write_all(&auth).await.unwrap();
        let mut authresp = [0u8; 2];
        client.read_exact(&mut authresp).await.unwrap();
        assert_eq!(authresp, [USERPASS_VERSION, 0x00], "auth must succeed");

        // CONNECT to 127.0.0.1:9001.
        client.write_all(&[0x05, 0x01, 0x00, 0x01, 127, 0, 0, 1, 0x23, 0x29]).await.unwrap();
        let mut creply = [0u8; 10];
        client.read_exact(&mut creply).await.unwrap();
        assert_eq!(creply[1], 0x00);

        let (target, args) = server.await.unwrap();
        assert_eq!(target, SocketAddr::from(([127, 0, 0, 1], 9001)));
        assert_eq!(args.cert, Some(cert), "cert must be parsed from the SOCKS args");
    }

    #[tokio::test]
    async fn rejects_domain_target() {
        // A DOMAIN target must be refused (resolving it would leak the bridge
        // hostname to local DNS) — Tor must address the bridge by IP.
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server =
            tokio::spawn(async move {
                let (mut s, _) = listener.accept().await.unwrap();
                accept_socks5_connect(&mut s).await
            });

        let mut client = TcpStream::connect(addr).await.unwrap();
        client.write_all(&[0x05, 0x01, 0x00]).await.unwrap(); // no-auth
        let mut resp = [0u8; 2];
        client.read_exact(&mut resp).await.unwrap();

        // CONNECT to example.com:443 via DOMAIN address type.
        let host = b"example.com";
        let mut req = vec![0x05, 0x01, 0x00, DOMAIN, host.len() as u8];
        req.extend_from_slice(host);
        req.extend_from_slice(&443u16.to_be_bytes());
        client.write_all(&req).await.unwrap();

        let result = server.await.unwrap();
        assert!(
            matches!(result, Err(SocksError::DomainTargetRejected)),
            "domain targets must be rejected, not resolved"
        );
    }
}
