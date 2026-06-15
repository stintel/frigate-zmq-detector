// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Frigate ZMQ detector sidecar with `TFLite` + Mesa Teflon delegate.
//!
//! This sidecar listens on a ZMQ REP socket, receives inference requests from
//! Frigate's `zmq_ipc` plugin, and returns detection results.

mod backend;
mod cli;
mod error;
mod protocol;
mod tflite;
mod watchdog;

use std::sync::Arc;
use std::time::{Duration, Instant};
use std::{env, process::Command};

use clap::Parser;
use zeromq::{RepSocket, Socket, SocketRecv, SocketSend, ZmqError};

use crate::backend::DetectorBackend;
use crate::cli::{BackendKind, Cli};
use crate::error::{Result, SidecarError};
use crate::tflite::TfliteManager;
use crate::watchdog::{EXIT_CODE_NO_PROGRESS, ProgressWatchdog};

const HEALTH_LOG_INTERVAL: Duration = Duration::from_secs(60);
const WATCHDOG_CHECK_INTERVAL: Duration = Duration::from_secs(5);
const WORKER_ENV: &str = "FRIGATE_ZMQ_DETECTOR_WORKER";
const SUPERVISE_ENV: &str = "FRIGATE_ZMQ_DETECTOR_SUPERVISE";

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
        "Starting frigate-zmq-detector v{} (git {})",
        env!("CARGO_PKG_VERSION"),
        env!("GIT_HASH")
    );
    log::info!("ZMQ endpoint: {}", cli.endpoint);
    log::info!("Threads: {}", cli.threads);
    log::info!("Teflon delegate: {}", !cli.no_delegate);
    log::info!("Inference timeout: {} ms", cli.inference_timeout_ms);
    log::info!(
        "Progress watchdog: no_progress={}s, recv_timeout={}s, max_requests={}, max_lifetime={}s",
        cli.max_no_progress_secs,
        cli.recv_timeout_secs,
        cli.max_requests,
        cli.max_lifetime_secs
    );
    log::info!("ZMQ send timeout: {}s", cli.send_timeout_secs);

    // Select detector backend.
    let mut manager = match cli.backend {
        BackendKind::Teflon => {
            log::info!("Detector backend: teflon");

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

            manager
        }
    };

    // If a model file is pre-mounted, load it.
    if let Some(ref model_path) = cli.model {
        log::info!("Loading model from {}", model_path.display());
        let data = std::fs::read(model_path)
            .map_err(|e| SidecarError::Io(format!("read {}: {e:#?}", model_path.display())))?;
        let model_name = model_path
            .file_name()
            .and_then(|name| name.to_str())
            .map(str::to_string);
        manager.cache_model(data, model_name)?;
    } else {
        log::info!("No model pre-loaded; awaiting ZMQ transfer from Frigate");
    }

    // Warmup if model is available.
    if DetectorBackend::is_ready(&manager) && cli.warmup_runs > 0 {
        log::info!("Running {} warmup inference(s)", cli.warmup_runs);
        for run in 1..=cli.warmup_runs {
            let warmup_start = std::time::Instant::now();
            let warmup_result = watchdog::run_with_process_watchdog(
                "warmup inference",
                Duration::from_millis(cli.inference_timeout_ms),
                || DetectorBackend::warmup(&mut manager),
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

    // Build progress watchdog.
    let watchdog = ProgressWatchdog::new(
        cli.max_no_progress_secs,
        cli.max_requests,
        cli.max_lifetime_secs,
    );

    // Run ZMQ REP server.
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|e| SidecarError::Io(format!("tokio runtime init: {e:#?}")))?;
    runtime
        .block_on(async {
            zmq_rep_loop(
                cli.endpoint.clone(),
                Arc::clone(&manager),
                Duration::from_millis(cli.inference_timeout_ms),
                watchdog,
                Duration::from_secs(cli.recv_timeout_secs),
                Duration::from_secs(cli.send_timeout_secs),
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

/// Main ZMQ REP server loop — hardened with recv timeout, error backoff,
/// and progress tracking.
async fn zmq_rep_loop(
    endpoint: String,
    manager: Arc<std::sync::Mutex<TfliteManager>>,
    inference_timeout: Duration,
    health: Arc<ProgressWatchdog>,
    recv_timeout: Duration,
    send_timeout: Duration,
) -> Result<()> {
    let mut socket = RepSocket::new();

    socket
        .bind(&endpoint)
        .await
        .map_err(|e| SidecarError::Zmq(format!("bind {endpoint}: {e:#?}")))?;

    log::info!("Listening on {endpoint}");

    let mut stats = LoopStats::new();
    let mut consecutive_recv_errors: u32 = 0;

    // Spawn watchdog monitor in the background.
    let monitor_handle = tokio::spawn(watchdog_monitor(Arc::clone(&health)));

    loop {
        // Receive request — with optional timeout to prevent infinite blocking.
        // Result variants: Ok = message, Err(RecvStatus::Timeout), Err(RecvStatus::Error)
        let msg = if recv_timeout.is_zero() {
            match socket.recv().await {
                Ok(m) => {
                    consecutive_recv_errors = 0;
                    Some(m)
                }
                Err(e) => {
                    handle_recv_error(&mut consecutive_recv_errors, &e, &health, &monitor_handle)
                        .await;
                    None
                }
            }
        } else {
            match tokio::time::timeout(recv_timeout, socket.recv()).await {
                Ok(Ok(m)) => {
                    consecutive_recv_errors = 0;
                    Some(m)
                }
                Ok(Err(e)) => {
                    handle_recv_error(&mut consecutive_recv_errors, &e, &health, &monitor_handle)
                        .await;
                    None
                }
                Err(_) => {
                    // Timeout — recv took too long. Check progress watchdog.
                    if let Some(reason) = health.check() {
                        log::error!("recv watchdog: {reason}; exiting");
                        monitor_handle.abort();
                        std::process::exit(EXIT_CODE_NO_PROGRESS);
                    }
                    None
                }
            }
        };

        // If no message received, loop back to recv.
        let msg = match msg {
            Some(m) => m,
            None => continue,
        };

        let request_started = Instant::now();

        // Dispatch: model management or inference.
        let is_mgmt = protocol::classify_message(&msg);

        if !is_mgmt {
            // Track inference requests in progress watchdog.
            health.request_start();
            log::debug!(
                "Inference request #{}, {} frame(s)",
                health.snapshot().total_requests,
                msg.len()
            );
        } else {
            log::debug!("Model mgmt request, {} frame(s)", msg.len());
        }

        let dispatch_start = Instant::now();
        let reply = if is_mgmt {
            let mut mgr = manager
                .lock()
                .map_err(|e| SidecarError::Tflite(format!("manager lock: {e:#?}")))?;
            match protocol::handle_model_request(msg, &mut *mgr) {
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
            match protocol::handle_inference(msg, &mut *mgr, inference_timeout) {
                Ok(r) => r,
                Err(e) => {
                    log::error!("Inference dispatch error: {e}");
                    protocol::zero_inference_reply()
                }
            }
        };
        let dispatch_ms = dispatch_start.elapsed().as_secs_f64() * 1000.0;

        // Send reply — measure send duration.
        let send_start = Instant::now();
        let send_result = if send_timeout.is_zero() {
            socket.send(reply).await.map_err(SendFailure::Zmq)
        } else {
            match tokio::time::timeout(send_timeout, socket.send(reply)).await {
                Ok(result) => result.map_err(SendFailure::Zmq),
                Err(_) => Err(SendFailure::Timeout),
            }
        };
        let send_ok = match send_result {
            Ok(()) => {
                log::debug!(
                    "Reply sent in {:.1} ms",
                    send_start.elapsed().as_secs_f64() * 1000.0
                );
                true
            }
            Err(SendFailure::Zmq(e)) => {
                log::error!("send reply: {e:#?}");
                false
            }
            Err(SendFailure::Timeout) => {
                log::error!("send reply timed out after {send_timeout:.1?}");
                false
            }
        };

        // Update progress tracking.
        if !is_mgmt {
            if send_ok {
                health.response_ok();
            } else {
                health.response_err();
            }

            let total_ms = request_started.elapsed().as_secs_f64() * 1000.0;
            log::debug!(
                "Request complete in {:.1} ms (dispatch={:.1} ms, send={:.1} ms)",
                total_ms,
                dispatch_ms,
                send_start.elapsed().as_secs_f64() * 1000.0
            );
        }

        stats.record(is_mgmt, request_started.elapsed());
    }
}

enum SendFailure {
    Zmq(ZmqError),
    Timeout,
}

/// Handle a ZMQ recv error: log, back off on repeated errors, and check watchdog.
async fn handle_recv_error(
    consecutive: &mut u32,
    error: &ZmqError,
    health: &ProgressWatchdog,
    _monitor: &tokio::task::JoinHandle<()>,
) {
    *consecutive += 1;
    if *consecutive <= 3 || (*consecutive).is_multiple_of(10) {
        log::error!("recv error (consecutive={}): {error:#?}", *consecutive);
    }

    // Back off on repeated errors to avoid CPU spin.
    if *consecutive > 2 {
        let backoff_ms = (*consecutive as u64).min(2000);
        log::warn!("recv backoff: {} ms", backoff_ms);
        tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
    }

    // Check if we're stuck with no progress.
    if let Some(reason) = health.check() {
        log::error!("recv error watchdog: {reason}; exiting");
        // Can't abort monitor from here without handle — rely on process exit.
        std::process::exit(EXIT_CODE_NO_PROGRESS);
    }
}

/// Background task: periodically log health state and check if we should exit.
async fn watchdog_monitor(health: Arc<ProgressWatchdog>) {
    let mut health_timer = tokio::time::interval(HEALTH_LOG_INTERVAL);
    let mut check_timer = tokio::time::interval(WATCHDOG_CHECK_INTERVAL);

    // Skip first immediate tick from both intervals.
    health_timer.tick().await;
    check_timer.tick().await;

    loop {
        tokio::select! {
            _ = health_timer.tick() => {
                let snap = health.snapshot();
                if snap.total_requests > 0 {
                    log::info!(
                        "health: requests={}, successes={}, failures={}, last_success_age={:.1}s",
                        snap.total_requests,
                        snap.total_successes,
                        snap.total_failures,
                        snap.last_success_age_secs(),
                    );
                } else {
                    log::info!(
                        "health: idle (uptime={:.0}s)",
                        (snap.now_ms as f64) / 1000.0,
                    );
                }
            }
            _ = check_timer.tick() => {
                if health.is_enabled()
                    && let Some(reason) = health.check() {
                        log::error!("watchdog monitor fired: {reason}; exiting");
                        std::process::exit(EXIT_CODE_NO_PROGRESS);
                    }
            }
        }
    }
}

struct LoopStats {
    interval_started: Instant,
    requests: u64,
    mgmt_requests: u64,
    inference_requests: u64,
    total_response_time: Duration,
    max_response_time: Duration,
}

impl LoopStats {
    fn new() -> Self {
        Self {
            interval_started: Instant::now(),
            requests: 0,
            mgmt_requests: 0,
            inference_requests: 0,
            total_response_time: Duration::ZERO,
            max_response_time: Duration::ZERO,
        }
    }

    fn record(&mut self, is_mgmt: bool, response_time: Duration) {
        self.requests += 1;
        if is_mgmt {
            self.mgmt_requests += 1;
        } else {
            self.inference_requests += 1;
        }
        self.total_response_time += response_time;
        self.max_response_time = self.max_response_time.max(response_time);

        let elapsed = self.interval_started.elapsed();
        if elapsed < HEALTH_LOG_INTERVAL {
            return;
        }

        let avg_ms = if self.requests == 0 {
            0.0
        } else {
            self.total_response_time.as_secs_f64() * 1000.0 / self.requests as f64
        };
        let req_per_sec = self.requests as f64 / elapsed.as_secs_f64();
        let infer_per_sec = self.inference_requests as f64 / elapsed.as_secs_f64();
        let max_ms = self.max_response_time.as_secs_f64() * 1000.0;

        log::info!(
            "ZMQ stats: {:.1} req/s, {:.1} infer/s, {} mgmt, avg {:.1} ms, max {:.1} ms",
            req_per_sec,
            infer_per_sec,
            self.mgmt_requests,
            avg_ms,
            max_ms
        );

        *self = Self::new();
    }
}
