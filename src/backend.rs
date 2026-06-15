// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Detector backend abstraction.
//!
//! Defines the trait that all inference backends must implement. The current
//! TFLite backend delegates to `TfliteManager`; additional backends (e.g.,
//! Coral / Edge TPU) can be added later without changing the protocol layer.

use crate::error::Result;

/// Capability that any inference backend must provide.
pub trait DetectorBackend {
    /// Returns `true` if a model is loaded and ready for inference.
    fn is_ready(&self) -> bool;

    /// Returns the name of the currently loaded model, if known.
    fn model_name(&self) -> Option<&str>;

    /// Returns `true` if the backend is ready and the requested model name
    /// matches the currently loaded model.
    fn is_model_ready(&self, model_name: &str) -> bool;

    /// Load and cache a model from raw `.tflite` bytes.
    ///
    /// The `model_name` is optional metadata used for identification and
    /// logging; it does not affect the binary model data.
    fn cache_model(&mut self, data: Vec<u8>, model_name: Option<String>) -> Result<()>;

    /// Run a single inference on pre-processed input bytes and return the
    /// raw output bytes (expected to be a 20×6 float32 detection buffer).
    fn run(&mut self, input_bytes: &[u8]) -> Result<Vec<u8>>;

    /// Run a warmup inference (typically with zeroed input) to trigger any
    /// one-time initialization such as delegate graph compilation.
    fn warmup(&mut self) -> Result<()>;
}
