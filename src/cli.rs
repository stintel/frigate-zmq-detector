//! Command-line argument definitions.

use std::path::PathBuf;

use clap::Parser;

/// Frigate ZMQ detector sidecar with `TFLite` + Mesa Teflon delegate.
#[derive(Parser, Debug, Clone)]
#[command(name = "frigate-sidecar", version)]
pub struct Cli {
    /// ZMQ REP endpoint to bind (`tcp://host:port` or `ipc://path`).
    #[arg(long, env = "ZMQ_ENDPOINT", default_value = "tcp://0.0.0.0:5555")]
    pub endpoint: String,

    /// Path to `TFLite` model file. Ignored if model is transferred via ZMQ.
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

    /// Disable Teflon delegate (CPU-only mode).
    #[arg(long, env = "NO_DELEGATE", default_value_t = false)]
    pub no_delegate: bool,

    /// `TFLite` shared library path.
    #[arg(
        long,
        env = "TFLITE_LIB",
        default_value = "/usr/lib/aarch64-linux-gnu/libtensorflow-lite.so"
    )]
    pub tflite_lib: PathBuf,

    /// Enable verbose debug logging.
    #[arg(long, short, env = "DEBUG", default_value_t = false)]
    pub debug: bool,
}
