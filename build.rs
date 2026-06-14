// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Build script to embed git commit hash at compile time.
//!
//! The version is already injected via `CARGO_PKG_VERSION` by Cargo.

use std::fs;
use std::path::Path;
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

    emit_git_rerun_paths();
}

fn emit_git_rerun_paths() {
    let git_dir = Path::new(".git");
    let head_path = git_dir.join("HEAD");

    println!("cargo:rerun-if-changed={}", head_path.display());

    let Ok(head) = fs::read_to_string(&head_path) else {
        return;
    };

    let Some(ref_name) = head.trim().strip_prefix("ref: ") else {
        return;
    };

    println!(
        "cargo:rerun-if-changed={}",
        git_dir.join(ref_name).display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        git_dir.join("packed-refs").display()
    );
}
