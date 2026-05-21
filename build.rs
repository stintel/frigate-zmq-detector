//! Build script to embed git commit hash at compile time.
//!
//! The version is already injected via `CARGO_PKG_VERSION` by Cargo.

use std::process::Command;

fn main() {
    // Capture short git commit hash.
    let hash = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .unwrap_or_else(|| "unknown".to_string())
        .trim()
        .to_string();

    println!("cargo:rustc-env=GIT_HASH={hash}");

    // Rebuild if .git/HEAD changes.
    println!("cargo:rerun-if-changed=.git/HEAD");
}
