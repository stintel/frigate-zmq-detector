//! Process-level watchdogs for native calls that can hang inside TFLite/delegates.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

pub(crate) fn run_with_process_watchdog<T, F>(label: &'static str, timeout: Duration, f: F) -> T
where
    F: FnOnce() -> T,
{
    if timeout.is_zero() {
        return f();
    }

    let completed = Arc::new(AtomicBool::new(false));
    let watchdog_completed = Arc::clone(&completed);
    let started = Instant::now();

    std::thread::spawn(move || {
        std::thread::sleep(timeout);
        if !watchdog_completed.load(Ordering::Acquire) {
            let elapsed = started.elapsed();
            log::error!("{label} exceeded {timeout:.1?} after {elapsed:.1?}; aborting worker");
            eprintln!("{label} exceeded {timeout:.1?} after {elapsed:.1?}; aborting worker");
            std::process::abort();
        }
    });

    let result = f();
    completed.store(true, Ordering::Release);
    result
}
