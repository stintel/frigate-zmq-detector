//! Process-level watchdogs for native calls that can hang inside TFLite/delegates.
//!
//! Contains two watchdogs:
//! 1. **ProcessWatchdog** — fires `std::process::abort()` when a single native call
//!    (e.g., inference) exceeds its deadline.
//! 2. **ProgressWatchdog** — tracks request/response progress and triggers
//!    a self-exit when no successful response completes within a timeout,
//!    or when lifetime / request-count limits are exceeded.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

static WATCHDOG: OnceLock<Arc<ProcessWatchdog>> = OnceLock::new();

struct ProcessWatchdog {
    state: Mutex<WatchdogState>,
    changed: Condvar,
}

#[derive(Default)]
struct WatchdogState {
    armed: Option<ArmedCall>,
    generation: u64,
}

struct ArmedCall {
    label: &'static str,
    deadline: Instant,
    timeout: Duration,
    generation: u64,
}

pub(crate) fn run_with_process_watchdog<T, F>(label: &'static str, timeout: Duration, f: F) -> T
where
    F: FnOnce() -> T,
{
    if timeout.is_zero() {
        return f();
    }

    let watchdog = WATCHDOG.get_or_init(ProcessWatchdog::spawn);
    let generation = watchdog.arm(label, timeout);
    let _guard = WatchdogGuard {
        watchdog: Arc::clone(watchdog),
        generation,
    };

    f()
}

impl ProcessWatchdog {
    fn spawn() -> Arc<Self> {
        let watchdog = Arc::new(Self {
            state: Mutex::new(WatchdogState::default()),
            changed: Condvar::new(),
        });
        Self::start_thread(&watchdog);
        watchdog
    }

    fn start_thread(watchdog: &Arc<Self>) {
        let watchdog = Arc::clone(watchdog);
        std::thread::Builder::new()
            .name("sidecar-watchdog".to_string())
            .spawn(move || watchdog.run())
            .expect("spawn process watchdog");
    }

    fn arm(&self, label: &'static str, timeout: Duration) -> u64 {
        let mut state = self.state.lock().expect("watchdog mutex poisoned");
        state.generation = state.generation.wrapping_add(1);
        let generation = state.generation;
        state.armed = Some(ArmedCall {
            label,
            deadline: Instant::now() + timeout,
            timeout,
            generation,
        });
        self.changed.notify_one();
        generation
    }

    fn disarm(&self, generation: u64) {
        let mut state = self.state.lock().expect("watchdog mutex poisoned");
        if state
            .armed
            .as_ref()
            .is_some_and(|armed| armed.generation == generation)
        {
            state.armed = None;
            self.changed.notify_one();
        }
    }

    fn run(&self) {
        let mut state = self.state.lock().expect("watchdog mutex poisoned");
        loop {
            let Some(armed) = state.armed.as_ref() else {
                state = self.changed.wait(state).expect("watchdog mutex poisoned");
                continue;
            };

            let now = Instant::now();
            if now >= armed.deadline {
                let elapsed = armed.timeout + now.duration_since(armed.deadline);
                log::error!(
                    "{} exceeded {:.1?} after {:.1?}; aborting worker",
                    armed.label,
                    armed.timeout,
                    elapsed
                );
                eprintln!(
                    "{} exceeded {:.1?} after {:.1?}; aborting worker",
                    armed.label, armed.timeout, elapsed
                );
                std::process::abort();
            }

            let wait = armed.deadline.duration_since(now);
            let (new_state, _timeout) = self
                .changed
                .wait_timeout(state, wait)
                .expect("watchdog mutex poisoned");
            state = new_state;
        }
    }
}

struct WatchdogGuard {
    watchdog: Arc<ProcessWatchdog>,
    generation: u64,
}

impl Drop for WatchdogGuard {
    fn drop(&mut self) {
        self.watchdog.disarm(self.generation);
    }
}

// ---------------------------------------------------------------------------
// Progress watchdog — track request/response progress and self-terminate
// on no-progress for too long.
// ---------------------------------------------------------------------------

/// Exit code when the progress watchdog fires (no successful response for
/// too long). Chosen to be distinct and not overlap with common codes.
pub const EXIT_CODE_NO_PROGRESS: i32 = 70;

/// Milliseconds since process start (epoch = `PROGRESS_EPOCH`).
fn elapsed_ms() -> u64 {
    let epoch = PROGRESS_EPOCH.get().expect("epoch not initialized");
    epoch.elapsed().as_millis() as u64
}

static PROGRESS_EPOCH: OnceLock<Instant> = OnceLock::new();

/// Monitors that request/response progress is being made.
///
/// All state is atomic so it can be accessed from both the async recv loop
/// and a background monitor task without holding locks across await points.
pub struct ProgressWatchdog {
    /// Total inference requests received (excluding model-management messages).
    total_requests: AtomicU64,
    /// Total successful responses sent.
    total_successes: AtomicU64,
    /// Total failed/error responses sent.
    total_failures: AtomicU64,
    /// Timestamp (ms since epoch) when the last successful response was sent.
    /// 0 means no success yet.
    last_success_ms: AtomicU64,
    /// Timestamp (ms since epoch) when the last failed response completed.
    /// 0 means no failure yet.
    last_failure_ms: AtomicU64,
    /// Timestamp (ms since epoch) when the first request started processing.
    /// 0 means no inference request has been observed yet.
    first_request_ms: AtomicU64,
    /// Timestamp (ms since epoch) when the current request started processing.
    /// 0 means no in-flight request.
    in_flight_start_ms: AtomicU64,
    /// Max elapsed seconds since last success before firing (0 = disabled).
    max_no_progress_secs: u64,
    /// Max total successful requests before clean exit (0 = disabled).
    max_requests: u64,
    /// Max runtime seconds before clean exit (0 = disabled).
    max_lifetime_secs: u64,
    /// Whether the watchdog has already fired (idempotent).
    fired: AtomicBool,
}

impl ProgressWatchdog {
    /// Create a new progress watchdog with the given configuration.
    pub fn new(max_no_progress_secs: u64, max_requests: u64, max_lifetime_secs: u64) -> Arc<Self> {
        PROGRESS_EPOCH.get_or_init(Instant::now);
        Arc::new(Self {
            total_requests: AtomicU64::new(0),
            total_successes: AtomicU64::new(0),
            total_failures: AtomicU64::new(0),
            last_success_ms: AtomicU64::new(0),
            last_failure_ms: AtomicU64::new(0),
            first_request_ms: AtomicU64::new(0),
            in_flight_start_ms: AtomicU64::new(0),
            max_no_progress_secs,
            max_requests,
            max_lifetime_secs,
            fired: AtomicBool::new(false),
        })
    }

    // -- Mutations called from the request loop --

    /// Mark the start of a new inference request.
    pub fn request_start(&self) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        let started_ms = elapsed_ms() + 1;
        let _ = self.first_request_ms.compare_exchange(
            0,
            started_ms,
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
        // Store +1 so that 0 is always a valid "not set" sentinel.
        self.in_flight_start_ms.store(started_ms, Ordering::Relaxed);
    }

    /// Mark a successful response completion.
    pub fn response_ok(&self) {
        // Store +1 so that 0 is always a valid "not set" sentinel even when
        // elapsed_ms() returns exactly 0 in the first millisecond.
        self.last_success_ms
            .store(elapsed_ms() + 1, Ordering::Relaxed);
        self.in_flight_start_ms.store(0, Ordering::Relaxed);
        let count = self.total_successes.fetch_add(1, Ordering::Relaxed) + 1;

        if self.max_requests > 0 && count >= self.max_requests {
            log::info!(
                "Reached max_requests={}, exiting for recycling",
                self.max_requests
            );
            std::process::exit(0);
        }
    }

    /// Mark a failed response completion.
    pub fn response_err(&self) {
        self.last_failure_ms
            .store(elapsed_ms() + 1, Ordering::Relaxed);
        self.total_failures.fetch_add(1, Ordering::Relaxed);
        self.in_flight_start_ms.store(0, Ordering::Relaxed);
    }

    // -- Inspection --

    /// Current snapshot of health state.
    pub fn snapshot(&self) -> HealthSnapshot {
        HealthSnapshot {
            total_requests: self.total_requests.load(Ordering::Relaxed),
            total_successes: self.total_successes.load(Ordering::Relaxed),
            total_failures: self.total_failures.load(Ordering::Relaxed),
            last_success_ms: self.last_success_ms.load(Ordering::Relaxed),
            last_failure_ms: self.last_failure_ms.load(Ordering::Relaxed),
            first_request_ms: self.first_request_ms.load(Ordering::Relaxed),
            in_flight_start_ms: self.in_flight_start_ms.load(Ordering::Relaxed),
            now_ms: elapsed_ms(),
        }
    }

    /// Check if we should fire the watchdog and exit.
    ///
    /// Returns `Some(reason)` if we should exit; `None` if we are healthy.
    /// Idempotent — returns `None` after the first fire to avoid double-exit.
    pub fn check(&self) -> Option<&'static str> {
        if self.fired.swap(true, Ordering::Relaxed) {
            return None; // already fired
        }

        let snap = self.snapshot();

        // Lifetime limit
        if self.max_lifetime_secs > 0 {
            let lifetime_secs = snap.now_ms / 1000;
            if lifetime_secs >= self.max_lifetime_secs {
                log::error!(
                    "max_lifetime_secs={} reached (lifetime={lifetime_secs}s); exiting for recycling",
                    self.max_lifetime_secs
                );
                return Some("max_lifetime exceeded");
            }
        }

        // No-progress timeout — only fires if we have seen at least one request
        if self.max_no_progress_secs > 0 && snap.total_requests > 0 {
            if snap.last_success_ms == 0 {
                let secs_since_first_request =
                    snap.now_ms.saturating_sub(snap.first_request_ms) / 1000;
                if secs_since_first_request >= self.max_no_progress_secs {
                    log::error!(
                        "progress watchdog fired: no success for {}s since first request \
                         (requests={}, failures={}, in_flight_age={:.1}s); exiting with code {}",
                        secs_since_first_request,
                        snap.total_requests,
                        snap.total_failures,
                        snap.in_flight_age_secs(),
                        EXIT_CODE_NO_PROGRESS
                    );
                    return Some("no success timeout");
                }
            }

            if snap.last_failure_ms > snap.last_success_ms {
                let secs_since_failure = snap.now_ms.saturating_sub(snap.last_failure_ms) / 1000;
                if secs_since_failure >= self.max_no_progress_secs {
                    log::error!(
                        "progress watchdog fired: no success for {}s since response failure \
                         (requests={}, successes={}, failures={}, last_success_age={:.1}s, \
                         in_flight_age={:.1}s); exiting with code {}",
                        secs_since_failure,
                        snap.total_requests,
                        snap.total_successes,
                        snap.total_failures,
                        snap.last_success_age_secs(),
                        snap.in_flight_age_secs(),
                        EXIT_CODE_NO_PROGRESS
                    );
                    return Some("failure timeout");
                }
            }
        }

        // In-flight request stuck
        if self.max_no_progress_secs > 0 && snap.in_flight_start_ms > 0 {
            let in_flight_secs = snap.now_ms.saturating_sub(snap.in_flight_start_ms) / 1000;
            if in_flight_secs >= self.max_no_progress_secs {
                log::error!(
                    "progress watchdog fired: in-flight request stuck for {}s \
                     (requests={}, successes={}, failures={}); exiting with code {}",
                    in_flight_secs,
                    snap.total_requests,
                    snap.total_successes,
                    snap.total_failures,
                    EXIT_CODE_NO_PROGRESS
                );
                return Some("in-flight stuck");
            }
        }

        // Reset the fired flag so we can check again later
        self.fired.store(false, Ordering::Relaxed);
        None
    }

    /// Whether the watchdog is configured with any active limits.
    pub fn is_enabled(&self) -> bool {
        self.max_no_progress_secs > 0 || self.max_requests > 0 || self.max_lifetime_secs > 0
    }
}

/// Snapshot of current health state at a point in time.
pub struct HealthSnapshot {
    pub total_requests: u64,
    pub total_successes: u64,
    pub total_failures: u64,
    pub last_success_ms: u64,
    pub last_failure_ms: u64,
    pub first_request_ms: u64,
    pub in_flight_start_ms: u64,
    pub now_ms: u64,
}

impl HealthSnapshot {
    /// Seconds since last successful response (0.0 if no success yet).
    pub fn last_success_age_secs(&self) -> f64 {
        if self.last_success_ms == 0 {
            0.0
        } else {
            // Saturating sub protects against underflow if snapshot's now_ms < stored ms
            // (possible within same ms tick or due to the +1 offset).
            (self.now_ms.saturating_sub(self.last_success_ms) as f64) / 1000.0
        }
    }

    /// Seconds since current in-flight request started (0.0 if no in-flight).
    pub fn in_flight_age_secs(&self) -> f64 {
        if self.in_flight_start_ms == 0 {
            0.0
        } else {
            (self.now_ms.saturating_sub(self.in_flight_start_ms) as f64) / 1000.0
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn new_watchdog(
        max_no_progress_secs: u64,
        max_requests: u64,
        max_lifetime_secs: u64,
    ) -> Arc<ProgressWatchdog> {
        ProgressWatchdog::new(max_no_progress_secs, max_requests, max_lifetime_secs)
    }

    #[test]
    fn test_initial_state_is_clean() {
        let wd = new_watchdog(60, 0, 0);
        let snap = wd.snapshot();

        assert_eq!(snap.total_requests, 0);
        assert_eq!(snap.total_successes, 0);
        assert_eq!(snap.total_failures, 0);
        assert_eq!(snap.last_success_ms, 0);
        assert_eq!(snap.last_failure_ms, 0);
        assert_eq!(snap.first_request_ms, 0);
        assert_eq!(snap.in_flight_start_ms, 0);
    }

    #[test]
    fn test_idle_no_traffic_does_not_fire() {
        let wd = new_watchdog(5, 0, 0);

        // Even after waiting, if no requests came, watchdog should not fire.
        std::thread::sleep(Duration::from_millis(100));
        let reason = wd.check();
        assert!(
            reason.is_none(),
            "watchdog should not fire when there is no traffic; got {reason:?}"
        );
    }

    #[test]
    fn test_request_start_sets_in_flight() {
        let wd = new_watchdog(60, 0, 0);
        wd.request_start();

        let snap = wd.snapshot();
        assert_eq!(snap.total_requests, 1);
        assert!(
            snap.in_flight_start_ms > 0,
            "in_flight_start_ms should be > 0 after request_start"
        );
        assert!(
            snap.first_request_ms > 0,
            "first_request_ms should be > 0 after request_start"
        );
    }

    #[test]
    fn test_response_ok_clears_in_flight() {
        let wd = new_watchdog(60, 0, 0);
        wd.request_start();
        wd.response_ok();

        let snap = wd.snapshot();
        assert_eq!(snap.total_requests, 1);
        assert_eq!(snap.total_successes, 1);
        assert_eq!(snap.in_flight_start_ms, 0, "in_flight should be cleared");
        assert!(snap.last_success_ms > 0, "last_success should be set");
    }

    #[test]
    fn test_response_err_clears_in_flight() {
        let wd = new_watchdog(60, 0, 0);
        wd.request_start();
        wd.response_err();

        let snap = wd.snapshot();
        assert_eq!(snap.total_requests, 1);
        assert_eq!(snap.total_successes, 0);
        assert_eq!(snap.total_failures, 1);
        assert!(snap.last_failure_ms > 0, "last_failure should be set");
        assert_eq!(snap.in_flight_start_ms, 0, "in_flight should be cleared");
    }

    #[test]
    fn test_multiple_requests_and_responses() {
        let wd = new_watchdog(60, 0, 0);

        for i in 0..5 {
            wd.request_start();
            if i % 2 == 0 {
                wd.response_ok();
            } else {
                wd.response_err();
            }
        }

        let snap = wd.snapshot();
        assert_eq!(snap.total_requests, 5);
        assert_eq!(snap.total_successes, 3); // indices 0, 2, 4
        assert_eq!(snap.total_failures, 2); // indices 1, 3
    }

    #[test]
    fn test_no_progress_timeout_fires_on_stuck_in_flight() {
        let _wd = new_watchdog(0, 0, 0); // 0-second timeout for fast test

        // Simulate: use the public API. The timeout is 0 seconds, meaning
        // any in-flight request is immediately "stuck" from a check perspective.
        // In reality the check has >= comparison, so 0 seconds since start
        // may or may not trigger. Let's test with a very low value instead.

        let wd = new_watchdog(1, 0, 0); // 1 second timeout
        wd.request_start();
        // Don't complete the request.

        std::thread::sleep(Duration::from_secs(2));
        let reason = wd.check();
        assert!(
            reason.is_some(),
            "watchdog should fire on stuck in-flight request; got {reason:?}"
        );
    }

    #[test]
    fn test_no_progress_timeout_fires_on_no_success() {
        // With a 1-second timeout, if requests come in but no success for 2s, fire.
        let wd = new_watchdog(1, 0, 0);

        // Simulate: request + failure, then wait past timeout.
        wd.request_start();
        wd.response_err();

        std::thread::sleep(Duration::from_secs(2));
        let reason = wd.check();
        assert!(
            reason.is_some(),
            "watchdog should fire after no success for longer than timeout; got {reason:?}"
        );
    }

    #[test]
    fn test_no_success_waits_for_timeout() {
        let wd = new_watchdog(60, 0, 0);

        wd.request_start();
        wd.response_err();

        let reason = wd.check();
        assert!(
            reason.is_none(),
            "watchdog should wait for timeout before firing with no successes; got {reason:?}"
        );
    }

    #[test]
    fn test_no_fire_when_success_recent() {
        let wd = new_watchdog(5, 0, 0);

        wd.request_start();
        wd.response_ok();

        // Within the timeout window, check should return None.
        let reason = wd.check();
        assert!(
            reason.is_none(),
            "watchdog should not fire when last success is recent; got {reason:?}"
        );
    }

    #[test]
    fn test_idle_after_success_does_not_fire() {
        let wd = new_watchdog(1, 0, 0);

        wd.request_start();
        wd.response_ok();

        std::thread::sleep(Duration::from_secs(2));
        let reason = wd.check();
        assert!(
            reason.is_none(),
            "watchdog should not fire while idle after a success; got {reason:?}"
        );
    }

    #[test]
    fn test_max_requests_recycling() {
        // max_requests=1 — should return Some from response_ok path via exit.
        // Since we can't easily test std::process::exit in a unit test,
        // we verify that the count is correct.
        let wd = new_watchdog(0, 100, 0);

        for _ in 0..50 {
            wd.request_start();
            wd.response_ok();
        }

        let snap = wd.snapshot();
        assert_eq!(snap.total_successes, 50);
        // max_requests=100, we've done 50 — no exit yet.
        let reason = wd.check();
        assert!(
            reason.is_none(),
            "should not fire when under max_requests; got {reason:?}"
        );
    }

    #[test]
    fn test_health_snapshot_age_calculations() {
        let wd = new_watchdog(60, 0, 0);
        wd.request_start();
        std::thread::sleep(Duration::from_millis(100));
        wd.response_ok();

        let snap = wd.snapshot();
        // in_flight should be 0 since we completed.
        assert_eq!(snap.in_flight_start_ms, 0);
        assert!(snap.in_flight_age_secs() == 0.0);

        // last_success_age should be small (recent).
        assert!(
            snap.last_success_age_secs() < 0.5,
            "last success age should be < 500ms but got {:.2}",
            snap.last_success_age_secs()
        );
    }

    #[test]
    fn test_watchdog_check_is_idempotent() {
        // After one fire, subsequent checks should return None.
        let wd = new_watchdog(1, 0, 0);
        wd.request_start();
        std::thread::sleep(Duration::from_secs(2));

        let reason1 = wd.check();
        let reason2 = wd.check();

        assert!(
            reason1.is_some(),
            "first check should fire; got {reason1:?}"
        );
        assert!(
            reason2.is_none(),
            "second check should return None (idempotent); got {reason2:?}"
        );
    }

    #[test]
    fn test_is_enabled() {
        let wd1 = new_watchdog(0, 0, 0);
        assert!(!wd1.is_enabled(), "all zeros should be disabled");

        let wd2 = new_watchdog(60, 0, 0);
        assert!(wd2.is_enabled(), "max_no_progress>0 should be enabled");

        let wd3 = new_watchdog(0, 100, 0);
        assert!(wd3.is_enabled(), "max_requests>0 should be enabled");

        let wd4 = new_watchdog(0, 0, 3600);
        assert!(wd4.is_enabled(), "max_lifetime>0 should be enabled");
    }
}
