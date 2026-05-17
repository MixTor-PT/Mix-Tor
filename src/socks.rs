use std::fmt;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, ToSocketAddrs};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

const SOCKS_VERSION: u8 = 0x05;
const NO_AUTH: u8 = 0x00;
const CONNECT: u8 = 0x01;
const IPV4: u8 = 0x01;
const DOMAIN: u8 = 0x03;
const IPV6: u8 = 0x04;

pub async fn accept_socks5_connect(stream: &mut TcpStream) -> Result<SocketAddr, SocksError> {
    negotiate_no_auth(stream).await?;
    let target = read_connect_target(stream).await?;
    stream
        .write_all(&[SOCKS_VERSION, 0x00, 0x00, IPV4, 0, 0, 0, 0, 0, 0])
        .await?;
    Ok(target)
}

async fn negotiate_no_auth(stream: &mut TcpStream) -> Result<(), SocksError> {
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
    if !methods.contains(&NO_AUTH) {
        stream.write_all(&[SOCKS_VERSION, 0xff]).await?;
        return Err(SocksError::NoSupportedAuthMethod);
    }

    stream.write_all(&[SOCKS_VERSION, NO_AUTH]).await?;
    Ok(())
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

    let host = match header[3] {
        IPV4 => {
            let mut octets = [0u8; 4];
            stream.read_exact(&mut octets).await?;
            IpAddr::V4(Ipv4Addr::from(octets)).to_string()
        }
        IPV6 => {
            let mut octets = [0u8; 16];
            stream.read_exact(&mut octets).await?;
            IpAddr::V6(Ipv6Addr::from(octets)).to_string()
        }
        DOMAIN => {
            let mut len = [0u8; 1];
            stream.read_exact(&mut len).await?;
            let mut domain = vec![0u8; len[0] as usize];
            stream.read_exact(&mut domain).await?;
            String::from_utf8(domain).map_err(|_| SocksError::InvalidDomain)?
        }
        kind => return Err(SocksError::UnsupportedAddressKind(kind)),
    };

    let mut port = [0u8; 2];
    stream.read_exact(&mut port).await?;
    let port = u16::from_be_bytes(port);
    let mut addrs = (host.as_str(), port)
        .to_socket_addrs()
        .map_err(SocksError::Resolve)?;
    addrs
        .next()
        .ok_or(SocksError::UnresolvableTarget(host, port))
}

#[derive(Debug)]
pub enum SocksError {
    Io(std::io::Error),
    InvalidDomain,
    NoSupportedAuthMethod,
    Resolve(std::io::Error),
    UnresolvableTarget(String, u16),
    UnsupportedAddressKind(u8),
    UnsupportedCommand(u8),
    UnsupportedVersion(u8),
}

impl fmt::Display for SocksError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "SOCKS I/O error: {error}"),
            Self::InvalidDomain => write!(formatter, "SOCKS target domain is not valid UTF-8"),
            Self::NoSupportedAuthMethod => write!(formatter, "SOCKS client did not offer no-auth"),
            Self::Resolve(error) => write!(formatter, "failed to resolve SOCKS target: {error}"),
            Self::UnresolvableTarget(host, port) => {
                write!(formatter, "SOCKS target {host}:{port} did not resolve")
            }
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

        let target = server.await.expect("server should join");
        assert_eq!(target, SocketAddr::from(([127, 0, 0, 1], 9001)));
    }
}
