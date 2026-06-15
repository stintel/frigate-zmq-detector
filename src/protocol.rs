// SPDX-License-Identifier: Apache-2.0 OR MIT

//! ZMQ REQ/REP protocol implementation matching Frigate `zmq_ipc` plugin.

use serde_json::json;
use std::time::Duration;
use zeromq::ZmqMessage;

use crate::backend::DetectorBackend;
use crate::error::{Result, SidecarError};
use crate::watchdog::run_with_process_watchdog;

// (20, 6) float32 output = 480 bytes.
const EXPECTED_OUTPUT_BYTES: usize = 20 * 6 * 4;

// ---------------------------------------------------------------------------
// Message dispatch
// ---------------------------------------------------------------------------

/// Determine whether an incoming `ZmqMessage` is a model-management or
/// inference request by inspecting the first frame.
pub(crate) fn classify_message(msg: &ZmqMessage) -> bool {
    // If first frame parses as JSON with model_request or model_data, it's
    // a model-management message.
    if let Some(frame) = msg.get(0)
        && let Ok(val) = serde_json::from_slice::<serde_json::Value>(frame)
    {
        return val.get("model_request").is_some() || val.get("model_data").is_some();
    }
    false
}

// ---------------------------------------------------------------------------
// Model-management messages (single-frame JSON reply)
// ---------------------------------------------------------------------------

/// Handle a model availability query or model data transfer.
pub(crate) fn handle_model_request(
    msg: ZmqMessage,
    backend: &mut dyn DetectorBackend,
) -> Result<ZmqMessage> {
    let frames = msg.into_vec();
    if frames.is_empty() {
        return Err(SidecarError::Zmq("empty model request".to_string()));
    }

    let header: serde_json::Value =
        serde_json::from_slice(&frames[0]).map_err(|e| SidecarError::Json(format!("{e:#?}")))?;
    let name = header
        .get("model_name")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    // Single frame: availability query.
    if frames.len() == 1 {
        let loaded = backend.is_model_ready(name);
        log::info!("Model availability request for {name}: loaded={loaded}");
        return Ok(model_availability_reply(name, loaded, backend.model_name()));
    }

    // Two frames: model data transfer (header + .tflite bytes).
    let data: &[u8] = frames
        .get(1)
        .ok_or_else(|| SidecarError::Zmq("model transfer missing data frame".to_string()))?;

    if backend.is_ready() {
        let current_name = backend.model_name().unwrap_or("unknown");
        if backend.is_model_ready(name) {
            log::info!("Ignoring model transfer for {name}; model is already ready");
            return Ok(model_loaded_reply(
                name,
                true,
                true,
                Some("model already loaded"),
                None,
            ));
        }
        log::warn!(
            "Ignoring model transfer for {name} ({size} bytes); model {current_name} is already ready",
            size = data.len()
        );
        return Ok(model_loaded_reply(
            name,
            false,
            false,
            Some("different model already loaded"),
            Some(current_name),
        ));
    }

    let data_bytes: Vec<u8> = data.to_vec();
    log::info!(
        "Caching model {name} ({size} bytes)",
        size = data_bytes.len()
    );
    backend.cache_model(data_bytes, Some(name.to_string()))?;
    log::info!("Model {name} loaded");

    Ok(model_loaded_reply(
        name,
        true,
        true,
        Some("model loaded"),
        None,
    ))
}

fn model_availability_reply(
    model_name: &str,
    loaded: bool,
    current_model: Option<&str>,
) -> ZmqMessage {
    let mut resp = json!({
        "model_available": loaded,
        "model_loaded": loaded,
        "model_name": model_name,
        "message": if loaded { "model loaded" } else { "model not loaded" },
    });
    if !loaded && let Some(current_model) = current_model {
        resp["current_model"] = json!(current_model);
    }
    ZmqMessage::from(resp.to_string().into_bytes())
}

fn model_loaded_reply(
    model_name: &str,
    saved: bool,
    loaded: bool,
    message: Option<&str>,
    current_model: Option<&str>,
) -> ZmqMessage {
    let mut resp = json!({
        "model_saved": saved,
        "model_loaded": loaded,
        "model_name": model_name,
    });
    if let Some(message) = message {
        resp["message"] = json!(message);
    }
    if let Some(current_model) = current_model {
        resp["current_model"] = json!(current_model);
    }
    ZmqMessage::from(resp.to_string().into_bytes())
}

// ---------------------------------------------------------------------------
// Inference messages (2-frame: header_json + tensor_bytes)
// ---------------------------------------------------------------------------

/// Return a model error JSON reply (model not loaded).
pub(crate) fn model_error_reply() -> ZmqMessage {
    let resp = json!({"model_available": false, "model_loaded": false});
    ZmqMessage::from(resp.to_string().into_bytes())
}

/// Return a zero inference result (20, 6) float32.
pub(crate) fn zero_inference_reply() -> ZmqMessage {
    let header = json!({"shape": [20, 6], "dtype": "float32"});
    let header_bytes = header.to_string().into_bytes();
    let zeros = vec![0u8; EXPECTED_OUTPUT_BYTES];

    let mut reply = ZmqMessage::from(header_bytes);
    reply.push_back(zeros.into());

    reply
}

/// Handle an inference request and return a 2-frame reply.
pub(crate) fn handle_inference(
    msg: ZmqMessage,
    backend: &mut dyn DetectorBackend,
    inference_timeout: Duration,
) -> Result<ZmqMessage> {
    let frames = msg.into_vec();
    if frames.len() < 2 {
        return Err(SidecarError::Zmq(format!(
            "inference request needs 2 frames, got {}",
            frames.len()
        )));
    }

    let input_data: &[u8] = &frames[1];
    let start = log::log_enabled!(log::Level::Debug).then(std::time::Instant::now);

    let output = if backend.is_ready() {
        match run_with_process_watchdog("inference", inference_timeout, || backend.run(input_data))
        {
            Ok(buf) => buf,
            Err(e) => {
                log::error!("Inference error: {e} — returning zero detections");
                vec![0u8; EXPECTED_OUTPUT_BYTES]
            }
        }
    } else {
        let header = serde_json::from_slice::<serde_json::Value>(&frames[0]).ok();
        let shape = header
            .as_ref()
            .and_then(|h| h.get("shape").and_then(|s| s.as_array()))
            .map(|a| {
                a.iter()
                    .filter_map(serde_json::Value::as_i64)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        log::warn!("Inference requested but model not loaded (shape={shape:?}) — zero result");
        vec![0u8; EXPECTED_OUTPUT_BYTES]
    };

    if let Some(start) = start {
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        log::debug!("Inference in {ms:.1} ms");
    }

    // Build 2-frame response.
    let resp_header = json!({"shape": [20, 6], "dtype": "float32"});
    let header_bytes = resp_header.to_string().into_bytes();

    let mut reply = ZmqMessage::from(header_bytes);
    reply.push_back(output.into());

    Ok(reply)
}
