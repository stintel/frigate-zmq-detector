//! Frigate ZMQ detector sidecar with `TFLite` + Mesa Teflon delegate.
//!
//! This sidecar listens on a ZMQ REP socket, receives inference requests from
//! Frigate's `zmq_ipc` plugin, and returns detection results.

mod cli;
mod error;
mod protocol;
mod tflite;

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use clap::Parser;
use zeromq::{RepSocket, Socket, SocketRecv, SocketSend, ZmqError};

use crate::cli::Cli;
use crate::error::{Result, SidecarError};
use crate::tflite::TfliteManager;

static REQUEST_COUNT: AtomicU32 = AtomicU32::new(0);

/// Periodically log request count every 60 s (approximated).
const LOG_INTERVAL: u32 = 60_000;

// ---------------------------------------------------------------------------
// Entrypoint
// ---------------------------------------------------------------------------

fn main() {
    if let Err(e) = run() {
        eprintln!("Fatal: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging.
    env_logger::init_from_env(env_logger::Env::new().default_filter_or(if cli.debug {
        "debug"
    } else {
        "info"
    }));

    log::info!(
        "Starting frigate-sidecar v{} (git {})",
        env!("CARGO_PKG_VERSION"),
        env!("GIT_HASH")
    );
    log::info!("ZMQ endpoint: {}", cli.endpoint);
    log::info!("Threads: {}", cli.threads);
    log::info!("Teflon delegate: {}", !cli.no_delegate);

    // Load TFLite C library.
    let library = edgefirst_tflite::Library::from_path(&cli.tflite_lib).map_err(|e| {
        SidecarError::Tflite(format!(
            "Failed to load TFLite library {}: {e:#?}",
            cli.tflite_lib.display()
        ))
    })?;
    log::info!("Loaded TFLite from {}", cli.tflite_lib.display());
    let library = Box::leak(Box::new(library));

    // Build manager.
    let mut manager = TfliteManager::new(library, cli.threads);

    if cli.no_delegate {
        manager.set_delegate("", false);
    } else {
        manager.set_delegate(&cli.delegate.to_string_lossy(), true);
    }

    // If a model file is pre-mounted, load it.
    if let Some(ref model_path) = cli.model {
        log::info!("Loading model from {}", model_path.display());
        let data = std::fs::read(model_path)
            .map_err(|e| SidecarError::Io(format!("read {}: {e:#?}", model_path.display())))?;
        manager.cache_model(data)?;
    } else {
        log::info!("No model pre-loaded; awaiting ZMQ transfer from Frigate");
    }

    // Warmup if model is available.
    if manager.is_ready() && cli.warmup_runs > 0 {
        log::info!("Running {} warmup inference(s)", cli.warmup_runs);
        for run in 1..=cli.warmup_runs {
            let warmup_start = std::time::Instant::now();
            if let Ok(()) = manager.warmup() {
                let ms = warmup_start.elapsed().as_secs_f64() * 1000.0;
                log::info!(
                    "Warmup inference {run}/{} completed in {ms:.1} ms",
                    cli.warmup_runs
                );
            } else {
                log::warn!("Warmup inference {run}/{} failed", cli.warmup_runs);
                break;
            }
        }
    }

    // Wrap manager in Arc for shared access (single-threaded REP loop).
    let manager = Arc::new(std::sync::Mutex::new(manager));

    // Run ZMQ REP server.
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|e| SidecarError::Io(format!("tokio runtime init: {e:#?}")))?;
    runtime
        .block_on(async { zmq_rep_loop(cli.endpoint.clone(), Arc::clone(&manager)).await })
        .map_err(|e| SidecarError::Io(format!("zmq_rep_loop failed: {e:#?}")))?;

    Ok(())
}

/// Main ZMQ REP server loop.
async fn zmq_rep_loop(
    endpoint: String,
    manager: Arc<std::sync::Mutex<TfliteManager>>,
) -> Result<()> {
    let mut socket = RepSocket::new();

    socket
        .bind(&endpoint)
        .await
        .map_err(|e| SidecarError::Zmq(format!("bind {endpoint}: {e:#?}")))?;

    log::info!("Listening on {endpoint}");

    loop {
        // Wait for a request.
        let msg = match socket.recv().await {
            Ok(m) => m,
            Err(ZmqError::NoMessage) => continue,
            Err(e) => {
                log::error!("recv error: {e:#?}");
                continue;
            }
        };

        let count = REQUEST_COUNT.fetch_add(1, Ordering::Relaxed) + 1;

        // Periodic stats log.
        if count.is_multiple_of(LOG_INTERVAL) {
            log::info!("Processed {count} requests total");
        }

        // Dispatch: model management or inference.
        let is_mgmt = protocol::classify_message(&msg);
        let reply = if is_mgmt {
            let mut mgr = manager
                .lock()
                .map_err(|e| SidecarError::Tflite(format!("manager lock: {e:#?}")))?;
            match protocol::handle_model_request(msg, &mut mgr) {
                Ok(r) => r,
                Err(e) => {
                    log::error!("Model request error: {e}");
                    protocol::model_error_reply()
                }
            }
        } else {
            let mut mgr = manager
                .lock()
                .map_err(|e| SidecarError::Tflite(format!("manager lock: {e:#?}")))?;
            match protocol::handle_inference(msg, &mut mgr) {
                Ok(r) => r,
                Err(e) => {
                    log::error!("Inference dispatch error: {e}");
                    protocol::zero_inference_reply()
                }
            }
        };

        // Send reply.
        if let Err(e) = socket.send(reply).await {
            log::error!("send reply: {e:#?}");
        }
    }
}
