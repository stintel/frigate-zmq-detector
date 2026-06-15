// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Unified error types for the sidecar.

use std::fmt;

/// Sidecar-specific error type.
#[derive(Debug)]
pub enum SidecarError {
    /// I/O error (model save, file access, etc.).
    Io(String),
    /// Invalid CLI configuration.
    InvalidConfiguration(String),
    /// JSON parse error.
    Json(String),
    /// Model was not loaded before inference was requested.
    ModelNotLoaded,
    /// `TFLite` inference / delegate / model error.
    Tflite(String),
    /// ZMQ socket or protocol error.
    Zmq(String),
}

impl fmt::Display for SidecarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SidecarError::Io(msg) => write!(f, "I/O error: {msg}"),
            SidecarError::InvalidConfiguration(msg) => write!(f, "invalid configuration: {msg}"),
            SidecarError::Json(msg) => write!(f, "JSON error: {msg}"),
            SidecarError::ModelNotLoaded => write!(f, "model not loaded"),
            SidecarError::Tflite(msg) => write!(f, "TFLite error: {msg}"),
            SidecarError::Zmq(msg) => write!(f, "ZMQ error: {msg}"),
        }
    }
}

impl std::error::Error for SidecarError {}

impl From<serde_json::Error> for SidecarError {
    fn from(e: serde_json::Error) -> Self {
        SidecarError::Json(format!("{e:#?}"))
    }
}

/// Alias for sidecar-specific Result.
pub type Result<T> = std::result::Result<T, SidecarError>;
