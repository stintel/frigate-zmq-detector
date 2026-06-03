# Frigate ZMQ TFLite Sidecar

Standalone Rust detector sidecar for Frigate using ZMQ REQ/REP protocol, the
TFLite C runtime, and the Mesa Teflon delegate.

## Features

- **ZMQ REQ/REP protocol** — matches Frigate's `zmq_ipc` plugin exactly
- **Model pre-load** — load the TFLite model at startup; Frigate model
  transfers are acknowledged but ignored once the model is ready
- **Mesa Teflon delegate** — hardware-accelerated inference on Rockchip NPU via Rocket
- **CPU-only mode** — falls back to CPU via `--no-delegate`
- **SSD post-processing** — 4 TFLite SSD outputs → `(20,6)` float32 detections
- **Zero-panic runtime** — all errors handled gracefully; returns zero detections
  on failure instead of crashing
- **Worker supervision** — keeps the ZMQ endpoint alive by restarting the worker
  if the native TFLite/delegate stack times out or exits

## Quick Start

### Container

```bash
podman build -t frigate-sidecar .

# With Teflon delegate
podman run --rm \
  --network host \
  --device /dev/accel/accel0 \
  frigate-sidecar \
    --endpoint tcp://0.0.0.0:5555 \
    --delegate /usr/lib/teflon/libteflon.so \
    --threads 1

# CPU-only (no delegate)
podman run --rm --network host frigate-sidecar \
  --no-delegate --threads 1
```

### Local build

```bash
# Requires: Rust toolchain and a TFLite C runtime shared library.
# The ZMQ transport is pure Rust; libzmq is not used.
cargo build --release
./target/release/frigate-sidecar \
  --endpoint tcp://0.0.0.0:5555 \
  --delegate /usr/lib/teflon/libteflon.so \
  --threads 1
```

### With pre-mounted model

```bash
./target/release/frigate-sidecar \
  --endpoint tcp://0.0.0.0:5555 \
  --model /models/ssd_mobilenet_v2_fpnlite_320x320_coco_2021_07_28_fp16.tflite \
  --threads 1
```

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--endpoint` | `tcp://0.0.0.0:5555` | ZMQ REP socket to bind |
| `--model` | *(none)* | Pre-load a `.tflite` model from disk |
| `--delegate` | `/usr/lib/teflon/libteflon.so` | Path to Teflon delegate `.so` |
| `--threads` | `1` | TFLite CPU threads |
| `--no-delegate` | `false` | Disable Teflon delegate (CPU-only) |
| `--warmup-runs` | `3` | Warmup invocations at startup |
| `--inference-timeout-ms` | `150` | Abort and restart the worker if one inference exceeds this |
| `--tflite-lib` | `/usr/lib/aarch64-linux-gnu/libtensorflow-lite.so.2.14.1` | TFLite C library path |
| `--debug` | `false` | Enable debug logging |
| `--send-timeout-secs` | `5` | Timeout for a single ZMQ reply send; set to `0` to disable |

By default the binary starts a small supervisor process which spawns the actual
TFLite/ZMQ worker. Set `FRIGATE_SIDECAR_SUPERVISE=0` to disable this and run the
worker directly.

## Frigate Configuration

Add this detector to your Frigate `config.yml`:

```yaml
detectors:
  teflon_sidecar:
    type: zmq
    endpoint: tcp://192.168.1.50:5555
    request_timeout_ms: 300
```

Replace `endpoint` with your sidecar's ZMQ endpoint.
Keep Frigate's `request_timeout_ms` higher than the sidecar's
`--inference-timeout-ms`, so the worker abort/restart happens before Frigate's
detector request times out.

## How It Works

1. **Startup** — binds ZMQ REP socket, optionally pre-loads model file, runs warmup
2. **Model availability** — Frigate sends `{"model_request": true}` → sidecar replies
   `{"model_available": true, "model_loaded": true}`
3. **Model transfer** — Frigate sends 2-frame message (JSON header + `.tflite` bytes).
   If a pre-loaded model is already ready, sidecar acknowledges the transfer and
   keeps the existing interpreter. Otherwise, it validates the flatbuffer, builds
   the interpreter, and keeps it hot
4. **Inference** — Frigate sends 2-frame message (JSON header + uint8 tensor bytes).
   Sidecar invokes the cached TFLite interpreter, post-processes SSD output, returns
   2-frame response (JSON header + 480 float32 LE bytes)

### Protocol Details

| Message | Frame 1 | Frame 2 |
|---|---|---|
| Model availability | `{"model_request": true}` | *(none)* |
| Model transfer | `{"model_data": true, "model_name": "..."}` | `.tflite` bytes |
| Inference request | `{"shape":[...], "dtype":"uint8"}` | uint8 tensor bytes |
| Inference reply | `{"shape":[20,6], "dtype":"float32"}` | 480 float32 LE bytes |

## Troubleshooting

### Model not loading

Check logs for model validation errors. The TFLite model must match the expected
input tensor shape (usually `[1,320,320,3]` uint8 for Frigate's default model).

### Delegate load failure

Ensure the Teflon delegate `.so` is present and the GPU driver is loaded:
```bash
ls -la /usr/lib/teflon/libteflon.so
ls -la /dev/accel/accel0
```

### ZMQ connection refused

Verify the sidecar is listening on the expected endpoint:
```bash
ss -tlnp | grep 5555
```

### Slow first inference

The model transfer or pre-load step builds the interpreter. Warmup then invokes
the cached interpreter so delegate graph compilation happens before the first
Frigate detection request.

## License

MIT
