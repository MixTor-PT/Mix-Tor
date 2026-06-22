# Deploying MixTor as a Tor pluggable transport

MixTor runs as a managed Tor pluggable transport (like obfs4): a **bridge** side
(`mixtor-server`) and a **client** side (`mixtor-client`), wired up entirely
through `torrc`. The client authenticates the bridge with the bridge's identity
key, distributed on the `Bridge` line as `cert=<hex>` — exactly the obfs4 model.

This guide covers the software setup. It does **not** cover network plumbing
(public IP, firewall, port-forwarding the bridge's listen port) — do that
separately for the `ServerTransportListenAddr` port below.

## 1. Build

```
cargo build --release --bins
```

Binaries land in `target/release/`: `mixtor-server`, `mixtor-client`.

## 2. The bridge identity (cert)

The client must know the bridge's identity public key (the `cert`). Two ways:

**A. Let the bridge generate + persist it (recommended).** In managed mode the
bridge auto-generates a keypair on first run and stores it in Tor's PT state dir
(`TOR_PT_STATE_LOCATION`), so the cert is stable across restarts. It logs the
cert at startup:

```
mixtor-server identity public key (cert): 6c16f6fe…198524
  → give clients a Bridge line: Bridge mixtor <ip:port> <fingerprint> cert=6c16f6fe…198524
```

Read that line from the bridge's Tor notice log.

**B. Pin it yourself.** Mint a pair, writing the secret to an **owner-only key
file** (keeps it off the command line and out of torrc):

```
$ mixtor-server --keygen-file /etc/mixtor/identity.key
cert=6c16f6fe…198524
```

Pass `--static-key-file /etc/mixtor/identity.key` to the bridge (below) and hand
clients the `cert`.

> Avoid `--static-key <hex>` / `MIXTOR_STATIC_KEY` in production: command lines
> and torrc exec lines are easy to expose (process listings, logs, backups).
> Prefer a key file or the auto-persisted state-dir identity (option A).

## 3. Bridge `torrc` (`mixtor-server`)

```
BridgeRelay 1
ORPort 9002
ServerTransportPlugin mixtor exec /path/to/mixtor-server --managed
ServerTransportListenAddr mixtor 0.0.0.0:9001
ExtORPort auto
# ...standard bridge settings (DataDirectory, GeoIP, etc.)
```

- Clients connect to the bridge on the **`ServerTransportListenAddr` port (9001)** —
  that's the port to expose/forward, not the ORPort.
- To pin a fixed identity, append `--static-key-file <path>` to the
  `ServerTransportPlugin` exec line (don't put the raw secret on the line).
- `mixtor-server` forwards decrypted traffic to the local ORPort
  (`TOR_PT_ORPORT`); no `--upstream` needed in managed mode.

## 4. Client `torrc` (`mixtor-client`)

```
UseBridges 1
ClientTransportPlugin mixtor exec /path/to/mixtor-client --managed
Bridge mixtor <bridge-ip>:9001 <bridge-fingerprint> cert=<hex>
```

Tor hands the `cert=` to `mixtor-client` over SOCKS; the client authenticates the
bridge with it and refuses to talk to a bridge whose key doesn't match. The
`<bridge-fingerprint>` is the bridge's Tor relay fingerprint (from its
`DataDirectory/fingerprint`).

## 5. Tuning (optional, env vars on either side)

- `MIXTOR_TAIL_FLOOR_MS` — shaping/tail policy. Default `epoch` (production):
  `epoch:<ms>` sets the start-alignment + boundary granularity (default 5 s);
  the exit tail is the session-class duration. `class` = class floor only;
  `<N>` = fixed N-ms floor; `off` = no tail (testing only).
- The cover rate, cell size, and channel buffer are compile-time constants in
  `src/transport.rs` (see the cover-clock docs); raise the cover rate if your
  users' sustained download bandwidth approaches it.

## Notes / current limitations

- The wire is uniform-random end to end (Elligator2 handshake + ChaCha20-Poly1305
  fixed-size frames) and the bridge stays silent to active probes that don't know
  the `cert` (obfs4-style MARK/MAC gate).
- A local lab harness for multi-flow correlation testing against a real Tor bridge
  is in `localtest/run-multiflow-test.ps1` (Windows; needs a local Tor).
