use crate::clumping::{ClumpingPool, Packet};
use crate::composition::{MixedFrame, SeedBComposer};
use crate::lab::LabLogger;
use crate::protocol::{read_frame, write_frame, FrameKind, WireFrame, DEFAULT_MAX_FRAME_LEN};
use crate::seeds::SessionSeeds;
use crate::timing::SeedATimingScheduler;
use std::error::Error;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::time::{sleep, timeout};

pub async fn handle_client_connection(
    local: TcpStream,
    server: SocketAddr,
    max_read: usize,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    handle_client_connection_with_lab(local, server, max_read, None).await
}

pub async fn handle_client_connection_with_lab(
    local: TcpStream,
    server: SocketAddr,
    max_read: usize,
    lab: Option<Arc<LabLogger>>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let conn_id = lab
        .as_ref()
        .map(|logger| logger.next_conn_id())
        .unwrap_or(0);
    let remote = TcpStream::connect(server).await?;
    let (local_read, local_write) = local.into_split();
    let (remote_read, remote_write) = remote.into_split();
    let max_read = max_read.clamp(1, DEFAULT_MAX_FRAME_LEN);

    let outbound_lab = lab.clone();
    let outbound = tokio::spawn(async move {
        send_mixed_outbound(local_read, remote_write, max_read, outbound_lab, conn_id).await
    });
    let inbound =
        tokio::spawn(
            async move { receive_real_inbound(remote_read, local_write, lab, conn_id).await },
        );

    tokio::select! {
        result = outbound => result??,
        result = inbound => result??,
    }

    Ok(())
}

pub async fn handle_server_connection(
    client: TcpStream,
    upstream: SocketAddr,
    max_read: usize,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    handle_server_connection_with_lab(client, upstream, max_read, None).await
}

pub async fn handle_server_connection_with_lab(
    client: TcpStream,
    upstream: SocketAddr,
    max_read: usize,
    lab: Option<Arc<LabLogger>>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let conn_id = lab
        .as_ref()
        .map(|logger| logger.next_conn_id())
        .unwrap_or(0);
    let bridge = TcpStream::connect(upstream).await?;
    let (client_read, client_write) = client.into_split();
    let (bridge_read, bridge_write) = bridge.into_split();

    let inbound_lab = lab.clone();
    let inbound = tokio::spawn(async move {
        strip_dummies_to_bridge(client_read, bridge_write, inbound_lab, conn_id).await
    });
    let outbound = tokio::spawn(async move {
        send_bridge_replies(bridge_read, client_write, max_read, lab, conn_id).await
    });

    tokio::select! {
        result = inbound => result??,
        result = outbound => result??,
    }

    Ok(())
}

async fn send_mixed_outbound(
    mut local: tokio::net::tcp::OwnedReadHalf,
    mut remote: tokio::net::tcp::OwnedWriteHalf,
    max_read: usize,
    lab: Option<Arc<LabLogger>>,
    conn_id: u64,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let seeds = SessionSeeds::generate()?;
    let mut pool = ClumpingPool::new(seeds.timing_and_pool())?;
    let mut timing = SeedATimingScheduler::new(seeds.timing_and_pool())?;
    let mut composer = SeedBComposer::new(seeds.burst_composition())?;
    let mut buffer = vec![0u8; max_read.max(1)];
    let mut input_seq = 0u64;
    let mut wire_seq = 0u64;

    loop {
        if let Some(deadline) = pool.deadline() {
            let wait = deadline.saturating_duration_since(Instant::now());
            match timeout(wait, local.read(&mut buffer)).await {
                Ok(Ok(0)) => {
                    flush_pool(
                        &mut pool,
                        &mut timing,
                        &mut composer,
                        &mut remote,
                        &lab,
                        conn_id,
                        &mut wire_seq,
                    )
                    .await?;
                    remote.shutdown().await?;
                    return Ok(());
                }
                Ok(Ok(n)) => {
                    log_event(&lab, "client_input", conn_id, input_seq, n, "real");
                    input_seq += 1;
                    push_and_send(
                        &mut pool,
                        &mut timing,
                        &mut composer,
                        &mut remote,
                        &buffer[..n],
                        &lab,
                        conn_id,
                        &mut wire_seq,
                    )
                    .await?;
                }
                Ok(Err(error)) => return Err(error.into()),
                Err(_) => {
                    if let Some(burst) = pool.tick(Instant::now()) {
                        send_burst(
                            &mut timing,
                            &mut composer,
                            &mut remote,
                            burst,
                            &lab,
                            conn_id,
                            &mut wire_seq,
                        )
                        .await?;
                    }
                }
            }
        } else {
            let n = local.read(&mut buffer).await?;
            if n == 0 {
                remote.shutdown().await?;
                return Ok(());
            }

            log_event(&lab, "client_input", conn_id, input_seq, n, "real");
            input_seq += 1;
            push_and_send(
                &mut pool,
                &mut timing,
                &mut composer,
                &mut remote,
                &buffer[..n],
                &lab,
                conn_id,
                &mut wire_seq,
            )
            .await?;
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn push_and_send(
    pool: &mut ClumpingPool,
    timing: &mut SeedATimingScheduler,
    composer: &mut SeedBComposer,
    remote: &mut tokio::net::tcp::OwnedWriteHalf,
    bytes: &[u8],
    lab: &Option<Arc<LabLogger>>,
    conn_id: u64,
    wire_seq: &mut u64,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let packet = Packet::new(bytes.to_vec())?;
    for burst in pool.push(packet, Instant::now()) {
        send_burst(timing, composer, remote, burst, lab, conn_id, wire_seq).await?;
    }
    Ok(())
}

async fn flush_pool(
    pool: &mut ClumpingPool,
    timing: &mut SeedATimingScheduler,
    composer: &mut SeedBComposer,
    remote: &mut tokio::net::tcp::OwnedWriteHalf,
    lab: &Option<Arc<LabLogger>>,
    conn_id: u64,
    wire_seq: &mut u64,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    if let Some(burst) = pool.flush(Instant::now()) {
        send_burst(timing, composer, remote, burst, lab, conn_id, wire_seq).await?;
    }
    Ok(())
}

async fn send_burst(
    timing: &mut SeedATimingScheduler,
    composer: &mut SeedBComposer,
    remote: &mut tokio::net::tcp::OwnedWriteHalf,
    burst: crate::clumping::Burst,
    lab: &Option<Arc<LabLogger>>,
    conn_id: u64,
    wire_seq: &mut u64,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let decision = timing.next_decision();
    timing.observe_burst(burst.total_bytes() as u64, burst.packet_count() as u64);
    sleep(decision.delay).await;

    let mixed = composer.compose(burst);
    for frame in mixed.into_frames() {
        let wire = match frame {
            MixedFrame::Real(packet) => WireFrame::real(packet.into_bytes())?,
            MixedFrame::Dummy(bytes) => WireFrame::dummy(bytes)?,
        };
        let kind = match wire.kind() {
            FrameKind::Real => "real",
            FrameKind::Dummy => "dummy",
        };
        log_event(
            lab,
            "client_wire_output",
            conn_id,
            *wire_seq,
            wire.payload().len(),
            kind,
        );
        *wire_seq += 1;
        write_frame(remote, &wire).await?;
    }
    Ok(())
}

async fn receive_real_inbound(
    mut remote: tokio::net::tcp::OwnedReadHalf,
    mut local: tokio::net::tcp::OwnedWriteHalf,
    lab: Option<Arc<LabLogger>>,
    conn_id: u64,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut seq = 0u64;
    while let Some(frame) = read_frame(&mut remote).await? {
        if frame.kind() == FrameKind::Real {
            log_event(
                &lab,
                "client_inbound_real",
                conn_id,
                seq,
                frame.payload().len(),
                "real",
            );
            seq += 1;
            local.write_all(frame.payload()).await?;
        }
    }
    local.shutdown().await?;
    Ok(())
}

async fn strip_dummies_to_bridge(
    mut client: tokio::net::tcp::OwnedReadHalf,
    mut bridge: tokio::net::tcp::OwnedWriteHalf,
    lab: Option<Arc<LabLogger>>,
    conn_id: u64,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut wire_seq = 0u64;
    let mut output_seq = 0u64;
    while let Some(frame) = read_frame(&mut client).await? {
        let kind = match frame.kind() {
            FrameKind::Real => "real",
            FrameKind::Dummy => "dummy",
        };
        log_event(
            &lab,
            "server_wire_input",
            conn_id,
            wire_seq,
            frame.payload().len(),
            kind,
        );
        wire_seq += 1;
        match frame.kind() {
            FrameKind::Real => {
                log_event(
                    &lab,
                    "server_real_output",
                    conn_id,
                    output_seq,
                    frame.payload().len(),
                    "real",
                );
                output_seq += 1;
                bridge.write_all(frame.payload()).await?;
            }
            FrameKind::Dummy => {}
        }
    }
    bridge.shutdown().await?;
    Ok(())
}

async fn send_bridge_replies(
    mut bridge: tokio::net::tcp::OwnedReadHalf,
    mut client: tokio::net::tcp::OwnedWriteHalf,
    max_read: usize,
    lab: Option<Arc<LabLogger>>,
    conn_id: u64,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut buffer = vec![0u8; max_read.clamp(1, DEFAULT_MAX_FRAME_LEN)];
    let mut seq = 0u64;
    loop {
        let n = bridge.read(&mut buffer).await?;
        if n == 0 {
            client.shutdown().await?;
            return Ok(());
        }
        log_event(&lab, "server_bridge_reply", conn_id, seq, n, "real");
        seq += 1;
        let frame = WireFrame::real(buffer[..n].to_vec())?;
        write_frame(&mut client, &frame).await?;
    }
}

fn log_event(
    lab: &Option<Arc<LabLogger>>,
    event: &'static str,
    conn_id: u64,
    seq: u64,
    bytes: usize,
    kind: &'static str,
) {
    if let Some(logger) = lab {
        logger.log(event, conn_id, seq, bytes, kind);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[tokio::test]
    async fn loopback_proxy_round_trip() {
        let bridge = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bridge listener should bind");
        let bridge_addr = bridge.local_addr().expect("bridge addr should exist");
        let bridge_task = tokio::spawn(async move {
            let (mut socket, _) = bridge.accept().await.expect("bridge should accept");
            let mut buffer = vec![0u8; 1024];
            let n = socket.read(&mut buffer).await.expect("bridge should read");
            socket
                .write_all(&buffer[..n])
                .await
                .expect("bridge should echo");
        });

        let server = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("server listener should bind");
        let server_addr = server.local_addr().expect("server addr should exist");
        let server_task = tokio::spawn(async move {
            let (socket, _) = server.accept().await.expect("server should accept");
            handle_server_connection(socket, bridge_addr, 1200)
                .await
                .expect("server transport should complete");
        });

        let client = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("client listener should bind");
        let client_addr = client.local_addr().expect("client addr should exist");
        let client_task = tokio::spawn(async move {
            let (socket, _) = client.accept().await.expect("client should accept");
            handle_client_connection(socket, server_addr, 1200)
                .await
                .expect("client transport should complete");
        });

        let mut tor_side = TcpStream::connect(client_addr)
            .await
            .expect("tor side should connect");
        tor_side
            .write_all(b"mixtor-loopback-smoke")
            .await
            .expect("tor side should write");
        let mut output = vec![0u8; "mixtor-loopback-smoke".len()];
        tor_side
            .read_exact(&mut output)
            .await
            .expect("tor side should read echo");

        assert_eq!(output, b"mixtor-loopback-smoke");
        drop(tor_side);

        client_task.await.expect("client task should join");
        server_task.await.expect("server task should join");
        bridge_task.await.expect("bridge task should join");
    }
}
