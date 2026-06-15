<!-- SPDX-License-Identifier: Apache-2.0 OR MIT -->

# Frigate ZMQ Detector

Standalone Frigate `type: zmq` detector sidecar for running TFLite inference
outside the main Frigate process. The first supported backend is Mesa Teflon for
Rockchip NPU acceleration through Rocket.

## Why This Exists

Mesa Rocket/Teflon builds are affected by an `mmap()` leak in repeated buffer
map/unmap cycles. The Mesa fix is tracked in:

https://gitlab.freedesktop.org/mesa/mesa/-/merge_requests/41887

The local Mesa branch used while developing this project is
`fixes/teflon_rocket`; its fix commit is titled
`rocket: fix mmap leak in buffer map/unmap`. The commit message describes the
issue as `rkt_buffer_map()` creating an `mmap()` mapping for every buffer map,
while `rkt_buffer_unmap()` did not call `munmap()`. Repeated inference can
eventually exhaust the process address space and make `mmap()` fail.

That upstream fix may take time to reach distro releases. Running Rocket/Teflon
inside the main Frigate process can therefore make Frigate unstable over time.
The community workaround of injecting Teflon libraries into an existing Frigate
container is also fragile because the Mesa, Teflon, TFLite, and distro ABI set
must all line up at runtime.

Moving Frigate to a newer base OS is not a simple short-term workaround either:
some Python dependencies used by Frigate do not publish wheels for recent Python
versions yet, which makes a base image update risky even if the newer distro
ships Mesa with Rocket support.

This project keeps Frigate talking to a normal ZMQ detector while the native
TFLite/delegate stack runs in a separate process or container. If the native
stack wedges, times out, or exits, the detector process can be restarted without
taking Frigate down with it.

## Status

Experimental `0.1.x` project.

- Tested target: Mesa Teflon / Rocket on Rockchip NPU
- Frigate integration: Frigate `type: zmq` detector protocol
- Inference runtime: TFLite C runtime via `edgefirst-tflite`
- ZMQ transport: pure Rust `zeromq`; `libzmq` is not used
- Default model handling: Frigate can transfer the model over ZMQ, or a model can
  be pre-mounted with `--model`
- Hardware stability caveat: spontaneous reboots have been observed on Rock 5B+
  systems while running this stack. Kernel-side Rocket fixes are being tested in
  <https://github.com/stintel/linux/tree/fixes/rocket>.

## Features

- ZMQ REQ/REP protocol compatible with Frigate's ZMQ detector
- Model pre-load support for avoiding repeated model transfer work
- Mesa Teflon delegate support for Rockchip NPU acceleration
- CPU-only fallback with `--no-delegate`
- SSD post-processing from 4 TFLite SSD outputs to Frigate's `(20, 6)` float32
  detection format
- Error handling that returns zero detections on inference failure
- Worker supervision and inference timeouts to recover from native runtime hangs

## Quick Start

### Container

```bash
podman build -t frigate-zmq-detector .

podman run --rm \
  --network host \
  --device /dev/accel/accel0 \
  frigate-zmq-detector \
    --endpoint tcp://0.0.0.0:5555 \
    --delegate /usr/lib/teflon/libteflon.so \
    --threads 1
```

CPU-only mode:

```bash
podman run --rm --network host frigate-zmq-detector \
  --no-delegate \
  --threads 1
```

### Local Build

```bash
# Requires: Rust toolchain and a TFLite C runtime shared library.
cargo build --release

./target/release/frigate-zmq-detector \
  --endpoint tcp://0.0.0.0:5555 \
  --delegate /usr/lib/teflon/libteflon.so \
  --threads 1
```

### With Pre-Mounted Model

```bash
./target/release/frigate-zmq-detector \
  --endpoint tcp://0.0.0.0:5555 \
  --model /models/ssd_mobilenet_v2_fpnlite_320x320_coco_2021_07_28_fp16.tflite \
  --threads 1
```

## Frigate Configuration

Add this detector to your Frigate `config.yml`:

```yaml
detectors:
  teflon:
    type: zmq
    endpoint: tcp://192.168.1.50:5555
    request_timeout_ms: 300
```

Replace `endpoint` with the IP address and port where the detector is listening.
Keep Frigate's `request_timeout_ms` higher than this project's
`--inference-timeout-ms`, so the worker abort/restart path happens before
Frigate's detector request timeout.

See [frigate-example.yml](frigate-example.yml) for a larger configuration
fragment.

## CLI Reference

| Flag | Default | Description |
|---|---:|---|
| `--backend` | `teflon` | Detector backend to use |
| `--endpoint` | `tcp://0.0.0.0:5555` | ZMQ REP socket to bind |
| `--model` | none | Pre-load a `.tflite` model from disk |
| `--delegate` | `/usr/lib/teflon/libteflon.so` | Path to Teflon delegate `.so` |
| `--threads` | `1` | TFLite CPU threads |
| `--no-delegate` | `false` | Disable Teflon delegate and run CPU-only |
| `--warmup-runs` | `3` | Warmup invocations after model load |
| `--inference-timeout-ms` | `150` | Abort and restart the worker if one inference exceeds this |
| `--tflite-lib` | `/usr/lib/aarch64-linux-gnu/libtensorflow-lite.so.2.14.1` | TFLite C library path |
| `--debug` | `false` | Enable debug logging |
| `--recv-timeout-secs` | `30` | Timeout for a single ZMQ receive; set to `0` to disable |
| `--send-timeout-secs` | `5` | Timeout for a single ZMQ reply send; set to `0` to disable |
| `--max-no-progress-secs` | `60` | Exit if no successful response completes in this many seconds |
| `--max-requests` | `0` | Exit after this many successful inference requests; `0` disables recycling |
| `--max-lifetime-secs` | `0` | Exit after this many seconds of uptime; `0` disables recycling |

By default the binary starts a small supervisor process which spawns the actual
TFLite/ZMQ worker. Set `FRIGATE_ZMQ_DETECTOR_SUPERVISE=0` to disable this and
run the worker directly.

## Environment Variables

Most CLI flags can also be configured with environment variables:

| Variable | CLI flag |
|---|---|
| `BACKEND` | `--backend` |
| `ZMQ_ENDPOINT` | `--endpoint` |
| `MODEL_PATH` | `--model` |
| `TEFLON_LIB` | `--delegate` |
| `TFLITE_LIB` | `--tflite-lib` |
| `TFLITE_THREADS` | `--threads` |
| `WARMUP_RUNS` | `--warmup-runs` |
| `INFERENCE_TIMEOUT_MS` | `--inference-timeout-ms` |
| `NO_DELEGATE` | `--no-delegate` |
| `DEBUG` | `--debug` |
| `FRIGATE_ZMQ_DETECTOR_RECV_TIMEOUT_SECS` | `--recv-timeout-secs` |
| `FRIGATE_ZMQ_DETECTOR_SEND_TIMEOUT_SECS` | `--send-timeout-secs` |
| `FRIGATE_ZMQ_DETECTOR_MAX_NO_PROGRESS_SECS` | `--max-no-progress-secs` |
| `FRIGATE_ZMQ_DETECTOR_MAX_REQUESTS` | `--max-requests` |
| `FRIGATE_ZMQ_DETECTOR_MAX_LIFETIME_SECS` | `--max-lifetime-secs` |

## How It Works

1. Startup binds the ZMQ REP socket, optionally pre-loads a model file, and runs
   warmup inference.
2. Frigate sends `{"model_request": true}` and the detector reports whether the
   requested model is loaded.
3. Frigate can send a 2-frame model transfer message with JSON metadata plus
   `.tflite` bytes. If a pre-loaded model is already ready, the transfer is
   acknowledged and ignored.
4. Frigate sends inference requests as JSON metadata plus uint8 tensor bytes.
5. The detector invokes the cached TFLite interpreter and returns a 2-frame
   response with JSON metadata plus 480 bytes of little-endian float32 detection
   data.

### Protocol Details

| Message | Frame 1 | Frame 2 |
|---|---|---|
| Model availability | `{"model_request": true}` | none |
| Model transfer | `{"model_data": true, "model_name": "..."}` | `.tflite` bytes |
| Inference request | `{"shape":[...], "dtype":"uint8"}` | uint8 tensor bytes |
| Inference reply | `{"shape":[20,6], "dtype":"float32"}` | 480 float32 LE bytes |

## Troubleshooting

### Model Not Loading

Check logs for model validation errors. The TFLite model must match the expected
input tensor shape, usually `[1, 320, 320, 3]` uint8 for Frigate's default SSD
model.

### Delegate Load Failure

Ensure the Teflon delegate and Rocket accel node are present:

```bash
ls -la /usr/lib/teflon/libteflon.so
ls -la /dev/accel/accel0
```

Also verify that the TFLite runtime and delegate packages come from a compatible
distribution or image. Mixing libraries copied from another Frigate image or
host install can fail because of ABI mismatches.

### Rock 5B+ Reboots

Spontaneous Rock 5B+ reboots have been observed while running the Rocket/Teflon
stack. This appears to be below the detector process rather than a normal
application crash path. Kernel-side Rocket fixes are being tested separately in
<https://github.com/stintel/linux/tree/fixes/rocket>.

With those fixes applied, all 3 NPU cores are being used, compared to only 2
cores before. The same systems have also reached more than 8 days of uptime;
before those fixes, they would often reboot several times within 24 hours.

Those kernel patches are not being submitted upstream at this time. They touch
subsystems used by Rocket that I am not familiar enough with to review and
submit under the kernel project's AI contribution policy.

### ZMQ Connection Refused

Verify the detector is listening on the expected endpoint:

```bash
ss -tlnp | grep 5555
```

### Slow First Inference

The model transfer or pre-load step builds the interpreter. Warmup then invokes
the cached interpreter so delegate graph compilation happens before the first
Frigate detection request.

## Publishing Checklist

Before tagging a release:

```bash
cargo fmt --check
cargo test
cargo clippy --all-targets -- -D warnings
podman build -t frigate-zmq-detector .
```

Then smoke-test the container with Frigate configured to use the ZMQ detector.

## Security

This project is experimental and does not currently provide a dedicated security
support process. Please avoid exposing the ZMQ endpoint to untrusted networks.

## Credits

This project was built with the assistance of AI coding agents:

- Codex (OpenAI)
- Qwen Code with Qwen3.6-27B

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
