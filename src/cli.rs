//! Command-line argument definitions.

use std::path::PathBuf;

use clap::Parser;

/// Frigate ZMQ detector sidecar with `TFLite` + Mesa Teflon delegate.
#[derive(Parser, Debug, Clone)]
#[command(name = "frigate-zmq-detector", version)]
pub struct Cli {
    /// ZMQ REP endpoint to bind (`tcp://host:port` or `ipc://path`).
    #[arg(long, env = "ZMQ_ENDPOINT", default_value = "tcp://0.0.0.0:5555")]
    pub endpoint: String,

    /// Path to `TFLite` model file. If set, ZMQ model transfers are acknowledged but ignored.
    #[arg(long, env = "MODEL_PATH")]
    pub model: Option<PathBuf>,

    /// Path to Teflon delegate shared library.
    #[arg(
        long,
        env = "TEFLON_LIB",
        default_value = "/usr/lib/teflon/libteflon.so"
    )]
    pub delegate: PathBuf,

    /// Number of CPU threads for `TFLite` interpreter.
    #[arg(long, env = "TFLITE_THREADS", default_value_t = 1)]
    pub threads: i32,

    /// Number of warmup runs at startup.
    #[arg(long, env = "WARMUP_RUNS", default_value_t = 3)]
    pub warmup_runs: u32,

    /// Abort and restart the worker if one inference takes longer than this.
    #[arg(long, env = "INFERENCE_TIMEOUT_MS", default_value_t = 150)]
    pub inference_timeout_ms: u64,

    /// Disable Teflon delegate (CPU-only mode).
    #[arg(long, env = "NO_DELEGATE", default_value_t = false)]
    pub no_delegate: bool,

    /// `TFLite` shared library path.
    #[arg(
        long,
        env = "TFLITE_LIB",
        default_value = "/usr/lib/aarch64-linux-gnu/libtensorflow-lite.so.2.14.1"
    )]
    pub tflite_lib: PathBuf,

    /// Enable verbose debug logging.
    #[arg(long, short, env = "DEBUG", default_value_t = false)]
    pub debug: bool,

    /// Exit if no successful response completes within this many seconds.
    /// Set to 0 to disable. Protects against wedged ZMQ recv or hung inference.
    #[arg(
        long,
        env = "FRIGATE_ZMQ_DETECTOR_MAX_NO_PROGRESS_SECS",
        default_value_t = 60
    )]
    pub max_no_progress_secs: u64,

    /// Exit cleanly after this many successful inference requests (recycling).
    /// Set to 0 to disable. Mitigates gradual resource leaks.
    #[arg(long, env = "FRIGATE_ZMQ_DETECTOR_MAX_REQUESTS", default_value_t = 0)]
    pub max_requests: u64,

    /// Exit cleanly after this many seconds of uptime (recycling).
    /// Set to 0 to disable.
    #[arg(
        long,
        env = "FRIGATE_ZMQ_DETECTOR_MAX_LIFETIME_SECS",
        default_value_t = 0
    )]
    pub max_lifetime_secs: u64,

    /// Timeout for a single ZMQ recv call (seconds). Prevents infinite blocking
    /// on a hung recv. Set to 0 to disable.
    #[arg(
        long,
        env = "FRIGATE_ZMQ_DETECTOR_RECV_TIMEOUT_SECS",
        default_value_t = 30
    )]
    pub recv_timeout_secs: u64,

    /// Timeout for a single ZMQ send call (seconds). Prevents blocking forever
    /// while replying to a disconnected Frigate request. Set to 0 to disable.
    #[arg(
        long,
        env = "FRIGATE_ZMQ_DETECTOR_SEND_TIMEOUT_SECS",
        default_value_t = 5
    )]
    pub send_timeout_secs: u64,
}
