//! Frigate ZMQ detector sidecar with `TFLite` + Mesa Teflon delegate.
//!
//! This sidecar listens on a ZMQ REP socket, receives inference requests from
//! Frigate's `zmq_ipc` plugin, and returns detection results.

mod cli;
mod error;
mod protocol;
mod tflite;
mod watchdog;

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;
use std::{env, process::Command};

use clap::Parser;
use zeromq::{RepSocket, Socket, SocketRecv, SocketSend, ZmqError};

use crate::cli::Cli;
use crate::error::{Result, SidecarError};
use crate::tflite::TfliteManager;

static REQUEST_COUNT: AtomicU32 = AtomicU32::new(0);

/// Periodically log request count every 60 s (approximated).
const LOG_INTERVAL: u32 = 60_000;
const WORKER_ENV: &str = "FRIGATE_SIDECAR_WORKER";
const SUPERVISE_ENV: &str = "FRIGATE_SIDECAR_SUPERVISE";

// ---------------------------------------------------------------------------
// Entrypoint
// ---------------------------------------------------------------------------

fn main() {
    install_panic_logger();

    if let Err(e) = run() {
        eprintln!("Fatal: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging.
    let _ = env_logger::Builder::from_env(env_logger::Env::new().default_filter_or(if cli.debug {
        "debug"
    } else {
        "info"
    }))
    .try_init();

    if supervisor_enabled() && env::var_os(WORKER_ENV).is_none() {
        return supervise();
    }

    log::info!(
        "Starting frigate-sidecar v{} (git {})",
        env!("CARGO_PKG_VERSION"),
        env!("GIT_HASH")
    );
    log::info!("ZMQ endpoint: {}", cli.endpoint);
    log::info!("Threads: {}", cli.threads);
    log::info!("Teflon delegate: {}", !cli.no_delegate);
    log::info!("Inference timeout: {} ms", cli.inference_timeout_ms);

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
            let warmup_result = watchdog::run_with_process_watchdog(
                "warmup inference",
                Duration::from_millis(cli.inference_timeout_ms),
                || manager.warmup(),
            );
            if let Ok(()) = warmup_result {
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
        .block_on(async {
            zmq_rep_loop(
                cli.endpoint.clone(),
                Arc::clone(&manager),
                Duration::from_millis(cli.inference_timeout_ms),
            )
            .await
        })
        .map_err(|e| SidecarError::Io(format!("zmq_rep_loop failed: {e:#?}")))?;

    Ok(())
}

fn supervisor_enabled() -> bool {
    env::var(SUPERVISE_ENV)
        .map(|value| {
            !matches!(
                value.as_str(),
                "0" | "false" | "False" | "FALSE" | "no" | "No"
            )
        })
        .unwrap_or(true)
}

fn supervise() -> Result<()> {
    let exe =
        env::current_exe().map_err(|e| SidecarError::Io(format!("current_exe failed: {e:#?}")))?;
    let args = env::args_os().skip(1).collect::<Vec<_>>();
    let mut backoff = Duration::from_secs(1);

    log::info!(
        "Supervisor enabled; set {SUPERVISE_ENV}=0 to run without worker restart supervision"
    );

    loop {
        let started = std::time::Instant::now();
        let mut child = Command::new(&exe)
            .args(&args)
            .env(WORKER_ENV, "1")
            .spawn()
            .map_err(|e| SidecarError::Io(format!("spawn worker failed: {e:#?}")))?;

        log::info!("Started sidecar worker pid={}", child.id());

        let status = child
            .wait()
            .map_err(|e| SidecarError::Io(format!("wait worker failed: {e:#?}")))?;
        let runtime = started.elapsed();

        log::error!("Sidecar worker exited after {runtime:.1?}: {status}");

        if runtime >= Duration::from_secs(60) {
            backoff = Duration::from_secs(1);
        } else {
            log::warn!("Worker exited quickly; backing off for {backoff:.1?}");
            std::thread::sleep(backoff);
            backoff = (backoff * 2).min(Duration::from_secs(30));
            continue;
        }

        std::thread::sleep(backoff);
    }
}

fn install_panic_logger() {
    std::panic::set_hook(Box::new(|info| {
        log::error!("panic: {info}");
        eprintln!("panic: {info}");
    }));
}

/// Main ZMQ REP server loop.
async fn zmq_rep_loop(
    endpoint: String,
    manager: Arc<std::sync::Mutex<TfliteManager>>,
    inference_timeout: Duration,
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
        log::debug!("Received ZMQ request #{count} with {} frame(s)", msg.len());

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
            match protocol::handle_inference(msg, &mut mgr, inference_timeout) {
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
