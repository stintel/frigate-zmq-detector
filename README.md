# Frigate ZMQ TFLite Sidecar

Standalone Rust detector sidecar for Frigate using ZMQ REQ/REP protocol, the
TFLite C runtime, and the Mesa Teflon delegate.

## Features

- **ZMQ REQ/REP protocol** — matches Frigate's `zmq_ipc` plugin exactly
- **Model transfer over ZMQ** — Frigate sends the TFLite model at runtime
- **Mesa Teflon delegate** — hardware-accelerated inference on Intel Arc GPUs
- **CPU-only mode** — falls back to CPU via `--no-delegate`
- **SSD post-processing** — 4 TFLite SSD outputs → `(20,6)` float32 detections
- **Zero-panic runtime** — all errors handled gracefully; returns zero detections
  on failure instead of crashing

## Quick Start

### Docker

```bash
docker build -t frigate-sidecar .

# With Teflon delegate
docker run --rm \
  --network host \
  --device /dev/dri/renderD128 \
  -v /run/udev:/run/udev:ro \
  -v /models:/models \
  frigate-sidecar \
    --endpoint tcp://0.0.0.0:5555 \
    --delegate /usr/lib/teflon/libteflon.so \
    --threads 1

# CPU-only (no delegate)
docker run --rm --network host frigate-sidecar \
  --no-delegate --threads 1
```

### Local build

```bash
# Requires: libzmq3-dev, pkg-config, Rust toolchain
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
| `--model-dir` | `/models` | Model cache directory |
| `--tflite-lib` | `/usr/lib/aarch64-linux-gnu/libtensorflow-lite.so` | TFLite C library path |
| `--debug` | `false` | Enable debug logging |

## Frigate Configuration

Add this detector to your Frigate `config.yml`:

```yaml
detectors:
  teeflon_sidecar:
    type: zmq_ipc
    api_url: http://192.168.1.50:5555
    model: ssd_mobilenet_v2
```

Replace `api_url` with your sidecar's endpoint address.

## How It Works

1. **Startup** — binds ZMQ REP socket, optionally pre-loads model file, runs warmup
2. **Model availability** — Frigate sends `{"model_request": true}` → sidecar replies
   `{"model_available": true, "model_loaded": true}`
3. **Model transfer** — Frigate sends 2-frame message (JSON header + `.tflite` bytes).
   Sidecar validates flatbuffer, caches in memory
4. **Inference** — Frigate sends 2-frame message (JSON header + uint8 tensor bytes).
   Sidecar builds TFLite interpreter, invokes, post-processes SSD output, returns
   2-frame response (JSON header + 480 float32 LE bytes)

### Protocol Details

| Message | Frame 1 | Frame 2 |
|---|---|---|
| Model availability | `{"model_request": true}` | *(none)* |
| Model transfer | `{"shape":[...], "dtype":"uint8"}` | `.tflite` bytes |
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
lsof | grep -i tefflon  # check for conflicts
```

### ZMQ connection refused

Verify the sidecar is listening on the expected endpoint:
```bash
ss -tlnp | grep 5555
```

### Slow first inference

The first invocation triggers delegate graph compilation. Warmup at startup
pre-compiles the graph. If warmup is disabled or fails, expect a ~500 ms cold start
on the first detection.

## License

MIT
