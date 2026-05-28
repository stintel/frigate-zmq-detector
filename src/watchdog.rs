//! Process-level watchdogs for native calls that can hang inside TFLite/delegates.

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
