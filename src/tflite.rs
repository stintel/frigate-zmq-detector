// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `TFLite` inference engine wrapping edgefirst-tflite.

use std::path::Path;

use edgefirst_tflite::{Delegate, Interpreter, Library, Model};

use crate::error::{Result, SidecarError};

/// Maximum detection slots in the output buffer (20).
const MAX_DETECTIONS: usize = 20;

/// Minimum detection score threshold.
const MIN_SCORE: f32 = 0.4;

// ---------------------------------------------------------------------------
// TfliteManager
// ---------------------------------------------------------------------------

/// Manages the `TFLite` model lifecycle and runs inference.
///
/// The model, delegate, and interpreter are built once when the model is
/// loaded, then reused for all inference requests. Rebuilding the interpreter
/// repeatedly forces delegate graph compilation repeatedly, which is both slow
/// and rough on experimental delegates.
pub struct TfliteManager {
    library: &'static Library,
    /// Raw .tflite bytes received from Frigate or read from disk.
    model_bytes: Option<Vec<u8>>,
    model_name: Option<String>,
    model: Option<&'static Model<'static>>,
    interpreter: Option<Interpreter<'static>>,
    delegate_path: String,
    use_delegate: bool,
    threads: i32,
}

impl TfliteManager {
    /// Create a new manager using the default system `TFLite` library.
    pub fn new(library: &'static Library, threads: i32) -> Self {
        Self {
            library,
            model_bytes: None,
            model_name: None,
            model: None,
            interpreter: None,
            delegate_path: String::new(),
            use_delegate: false,
            threads,
        }
    }

    /// Configure delegate path and enable / disable the Teflon delegate.
    pub fn set_delegate(&mut self, path: &str, enabled: bool) {
        self.delegate_path = path.to_string();
        self.use_delegate = enabled;
    }

    /// Returns true if model bytes are cached.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.interpreter.is_some()
    }

    /// Returns the name associated with the loaded model, if known.
    #[must_use]
    pub fn model_name(&self) -> Option<&str> {
        self.model_name.as_deref()
    }

    /// Returns true if the requested model is the active loaded model.
    #[must_use]
    pub fn is_model_ready(&self, model_name: &str) -> bool {
        self.is_ready() && self.model_name() == Some(model_name)
    }

    /// Cache model bytes from a ZMQ transfer or file load.
    pub fn cache_model(&mut self, data: Vec<u8>, model_name: Option<String>) -> Result<()> {
        let model = Model::from_bytes(self.library, data.clone())
            .map_err(|e| SidecarError::Tflite(format!("Model validation failed: {e:#?}")))?;
        let model = Box::new(model);
        let model_ptr = Box::into_raw(model);
        // The cached interpreter borrows the model for the process lifetime.
        // If interpreter construction fails, reconstruct the Box below so the
        // model allocation is not leaked.
        let model_ref: &'static Model<'static> = unsafe { &*model_ptr };
        let interpreter = match build_interpreter(
            model_ref,
            self.library,
            &self.delegate_path,
            self.use_delegate,
            self.threads,
        ) {
            Ok(interpreter) => interpreter,
            Err(e) => {
                // SAFETY: model_ptr was created by Box::into_raw above and no
                // interpreter exists that can hold this reference on error.
                unsafe {
                    drop(Box::from_raw(model_ptr));
                }
                return Err(e);
            }
        };

        self.model_bytes = Some(data);
        self.model_name = model_name;
        self.interpreter = Some(interpreter);
        self.model = Some(model_ref);

        Ok(())
    }

    /// Run inference on pre-processed uint8 pixel bytes and return a
    /// (20, 6) float32 detection buffer as raw little-endian bytes (480).
    pub fn run(&mut self, input_bytes: &[u8]) -> Result<Vec<u8>> {
        let interpreter = self
            .interpreter
            .as_mut()
            .ok_or(SidecarError::ModelNotLoaded)?;
        let output = run_inference(interpreter, input_bytes)?;
        Ok(output)
    }

    /// Warmup: run inference once with zeroed input to trigger delegate
    /// graph compilation.
    pub fn warmup(&mut self) -> Result<()> {
        let interpreter = self
            .interpreter
            .as_mut()
            .ok_or(SidecarError::ModelNotLoaded)?;
        // Zero out input for warmup.
        let input_size = {
            let inputs = interpreter
                .inputs_mut()
                .map_err(|e| SidecarError::Tflite(format!("inputs_mut() failed: {e:#?}")))?;
            inputs[0].byte_size()
        };
        let zeros = vec![0u8; input_size];
        {
            let mut inputs = interpreter.inputs_mut().map_err(|e| {
                SidecarError::Tflite(format!("inputs_mut() (warmup set) failed: {e:#?}"))
            })?;
            inputs[0].copy_from_slice::<u8>(&zeros).map_err(|e| {
                SidecarError::Tflite(format!("copy_from_slice on warmup input failed: {e:#?}"))
            })?;
        }

        interpreter
            .invoke()
            .map_err(|e| SidecarError::Tflite(format!("warmup invoke failed: {e:#?}")))?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a Model + Interpreter with the given config.
fn build_interpreter<'a>(
    model: &'a Model,
    library: &'a Library,
    delegate_path: &str,
    use_delegate: bool,
    threads: i32,
) -> Result<Interpreter<'a>> {
    let mut builder = Interpreter::builder(library)
        .map_err(|e| SidecarError::Tflite(format!("Interpreter::builder failed: {e:#?}")))?;
    builder = builder.num_threads(threads);

    if use_delegate && !delegate_path.is_empty() {
        let delegate = Delegate::load(Path::new(delegate_path)).map_err(|e| {
            SidecarError::Tflite(format!("Delegate::load({delegate_path}) failed: {e:#?}"))
        })?;
        builder = builder.delegate(delegate);
    }

    builder
        .build(model)
        .map_err(|e| SidecarError::Tflite(format!("Interpreter build failed: {e:#?}")))
}

/// Set uint8 input tensor, invoke, and return post-processed (20, 6) bytes.
fn run_inference(interpreter: &mut Interpreter, input_bytes: &[u8]) -> Result<Vec<u8>> {
    // Set input.
    {
        let mut inputs = interpreter
            .inputs_mut()
            .map_err(|e| SidecarError::Tflite(format!("inputs_mut() failed: {e:#?}")))?;

        let input = &mut inputs[0];
        if input.byte_size() != input_bytes.len() {
            return Err(SidecarError::Tflite(format!(
                "Input size mismatch: tensor expects {} bytes, got {}",
                input.byte_size(),
                input_bytes.len()
            )));
        }

        input.copy_from_slice::<u8>(input_bytes).map_err(|e| {
            SidecarError::Tflite(format!("copy_from_slice on input tensor failed: {e:#?}"))
        })?;
    }

    // Invoke.
    interpreter
        .invoke()
        .map_err(|e| SidecarError::Tflite(format!("interpreter.invoke() failed: {e:#?}")))?;

    // Post-process SSD outputs.
    post_process_ssd(interpreter)
}

/// Post-process 4 SSD output tensors into (20, 6) float32 LE byte buffer.
///
/// Output tensors:
///   [0] boxes:     (1, max, 4) f32  — [ymin, xmin, ymax, xmax]
///   [1] `class_ids`: (1, max)    f32
///   [2] scores:    (1, max)    f32
///   [3] count:     (1)         f32
fn post_process_ssd(interp: &Interpreter) -> Result<Vec<u8>> {
    let outputs = interp
        .outputs()
        .map_err(|e| SidecarError::Tflite(format!("outputs() failed: {e:#?}")))?;

    if outputs.len() < 4 {
        return Err(SidecarError::Tflite(format!(
            "Expected 4 SSD output tensors, got {}",
            outputs.len()
        )));
    }

    let boxes = outputs[0]
        .as_slice::<f32>()
        .map_err(|e| SidecarError::Tflite(format!("boxes tensor read failed: {e:#?}")))?;
    let class_ids = outputs[1]
        .as_slice::<f32>()
        .map_err(|e| SidecarError::Tflite(format!("class_ids tensor read failed: {e:#?}")))?;
    let scores = outputs[2]
        .as_slice::<f32>()
        .map_err(|e| SidecarError::Tflite(format!("scores tensor read failed: {e:#?}")))?;
    let count = outputs[3]
        .as_slice::<f32>()
        .map_err(|e| SidecarError::Tflite(format!("count tensor read failed: {e:#?}")))?;

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let det_count = count
        .first()
        .map_or(0, |c| (*c as usize).min(MAX_DETECTIONS));

    // (20, 6) buffer: [class_id, score, y_min, x_min, y_max, x_max]
    let mut detections = vec![0.0f32; MAX_DETECTIONS * 6];

    for i in 0..det_count {
        let score = scores.get(i).copied().unwrap_or(0.0);
        if score < MIN_SCORE || i >= MAX_DETECTIONS {
            break;
        }

        let bo = i * 4;
        let y_min = boxes.get(bo).copied().unwrap_or(0.0);
        let x_min = boxes.get(bo.saturating_add(1)).copied().unwrap_or(0.0);
        let y_max = boxes.get(bo.saturating_add(2)).copied().unwrap_or(0.0);
        let x_max = boxes.get(bo.saturating_add(3)).copied().unwrap_or(0.0);
        let cls_id = class_ids.get(i).copied().unwrap_or(0.0);

        let oo = i * 6;
        detections[oo] = cls_id;
        detections[oo.saturating_add(1)] = score;
        detections[oo.saturating_add(2)] = y_min;
        detections[oo.saturating_add(3)] = x_min;
        detections[oo.saturating_add(4)] = y_max;
        detections[oo.saturating_add(5)] = x_max;
    }

    Ok(f32_slice_to_le_bytes(&detections))
}

/// Convert a float32 slice to little-endian bytes.
fn f32_slice_to_le_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}
