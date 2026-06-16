use std::fmt;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

const MAGIC: &[u8; 4] = b"MXT1";
const HEADER_LEN: usize = 9;
pub const DEFAULT_MAX_FRAME_LEN: usize = 64 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrameKind {
    Real = 1,
    Dummy = 2,
}

impl FrameKind {
    fn from_byte(byte: u8) -> Result<Self, ProtocolError> {
        match byte {
            1 => Ok(Self::Real),
            2 => Ok(Self::Dummy),
            _ => Err(ProtocolError::UnknownFrameKind(byte)),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WireFrame {
    kind: FrameKind,
    payload: Vec<u8>,
}

impl WireFrame {
    pub fn real(payload: Vec<u8>) -> Result<Self, ProtocolError> {
        Self::new(FrameKind::Real, payload)
    }

    pub fn dummy(payload: Vec<u8>) -> Result<Self, ProtocolError> {
        Self::new(FrameKind::Dummy, payload)
    }

    pub fn new(kind: FrameKind, payload: Vec<u8>) -> Result<Self, ProtocolError> {
        if payload.is_empty() {
            return Err(ProtocolError::EmptyPayload);
        }

        if payload.len() > DEFAULT_MAX_FRAME_LEN {
            return Err(ProtocolError::FrameTooLarge {
                len: payload.len(),
                max: DEFAULT_MAX_FRAME_LEN,
            });
        }

        Ok(Self { kind, payload })
    }

    pub fn kind(&self) -> FrameKind {
        self.kind
    }

    pub fn payload(&self) -> &[u8] {
        &self.payload
    }

    pub fn into_payload(self) -> Vec<u8> {
        self.payload
    }

    /// Serialize to the on-wire bytes (header + payload) — the same bytes
    /// `write_frame` emits. Lets a caller batch frames and write them with a
    /// single non-blocking `try_write` (used by the unified emitter).
    pub fn encode(&self) -> Vec<u8> {
        let len = self.payload.len() as u32; // new() bounds payload <= DEFAULT_MAX_FRAME_LEN
        let mut out = Vec::with_capacity(HEADER_LEN + self.payload.len());
        out.extend_from_slice(MAGIC);
        out.push(self.kind as u8);
        out.extend_from_slice(&len.to_be_bytes());
        out.extend_from_slice(&self.payload);
        out
    }
}

pub async fn write_frame<W>(writer: &mut W, frame: &WireFrame) -> Result<(), ProtocolError>
where
    W: AsyncWrite + Unpin,
{
    let len = u32::try_from(frame.payload.len()).map_err(|_| ProtocolError::FrameTooLarge {
        len: frame.payload.len(),
        max: DEFAULT_MAX_FRAME_LEN,
    })?;
    let mut header = [0u8; HEADER_LEN];
    header[..4].copy_from_slice(MAGIC);
    header[4] = frame.kind as u8;
    header[5..].copy_from_slice(&len.to_be_bytes());

    writer.write_all(&header).await?;
    writer.write_all(&frame.payload).await?;
    Ok(())
}

pub async fn read_frame<R>(reader: &mut R) -> Result<Option<WireFrame>, ProtocolError>
where
    R: AsyncRead + Unpin,
{
    let mut header = [0u8; HEADER_LEN];
    match reader.read_exact(&mut header).await {
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(error) => return Err(error.into()),
    }

    if &header[..4] != MAGIC {
        return Err(ProtocolError::BadMagic);
    }

    let kind = FrameKind::from_byte(header[4])?;
    let len = u32::from_be_bytes([header[5], header[6], header[7], header[8]]) as usize;

    if len == 0 {
        return Err(ProtocolError::EmptyPayload);
    }

    if len > DEFAULT_MAX_FRAME_LEN {
        return Err(ProtocolError::FrameTooLarge {
            len,
            max: DEFAULT_MAX_FRAME_LEN,
        });
    }

    let mut payload = vec![0u8; len];
    reader.read_exact(&mut payload).await?;

    Ok(Some(WireFrame { kind, payload }))
}

#[derive(Debug)]
pub enum ProtocolError {
    Io(std::io::Error),
    BadMagic,
    EmptyPayload,
    FrameTooLarge { len: usize, max: usize },
    UnknownFrameKind(u8),
}

impl fmt::Display for ProtocolError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "protocol I/O error: {error}"),
            Self::BadMagic => write!(formatter, "invalid MixTor frame magic"),
            Self::EmptyPayload => write!(formatter, "frame payload cannot be empty"),
            Self::FrameTooLarge { len, max } => {
                write!(
                    formatter,
                    "frame payload length {len} exceeds maximum {max}"
                )
            }
            Self::UnknownFrameKind(kind) => write!(formatter, "unknown frame kind {kind}"),
        }
    }
}

impl std::error::Error for ProtocolError {}

impl From<std::io::Error> for ProtocolError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::duplex;

    #[tokio::test]
    async fn round_trips_real_frame() {
        let (mut client, mut server) = duplex(128);
        let frame = WireFrame::real(b"hello".to_vec()).expect("frame should be valid");

        write_frame(&mut client, &frame)
            .await
            .expect("write should succeed");
        let decoded = read_frame(&mut server)
            .await
            .expect("read should succeed")
            .expect("frame should exist");

        assert_eq!(decoded, frame);
    }

    #[tokio::test]
    async fn rejects_bad_magic() {
        let (mut client, mut server) = duplex(128);
        client
            .write_all(b"NOPE\x01\0\0\0\x01x")
            .await
            .expect("write should succeed");

        let error = read_frame(&mut server)
            .await
            .expect_err("bad magic should fail");

        assert!(matches!(error, ProtocolError::BadMagic));
    }
}
