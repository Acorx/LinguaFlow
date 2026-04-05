/// Speech-to-text module supporting OpenAI Whisper API and local whisper.cpp.
///
/// - OpenAI: multipart/form-data upload to /v1/audio/transcriptions
/// - Local: whisper.cpp binary detection and subprocess invocation
/// - Auto-fallback: if local not available, uses API (with warning)

use log::{debug, error, info, warn};
use reqwest::Client;
use serde::Deserialize;
use std::env;
use std::path::PathBuf;

/// STT provider selection.
#[derive(Debug, Clone)]
pub enum STTProvider {
    OpenAI { api_key: String },
    WhisperCpp { model_path: String },
}

/// OpenAI transcription response.
#[derive(Debug, Deserialize)]
struct OpenAITranscription {
    text: String,
}

/// Transcribes a WAV file to text using the detected provider.
///
/// # Arguments
/// * `wav_path` — path to a 16kHz mono WAV file
/// * `provider` — the STT provider to use
///
/// # Returns
/// The transcribed text, or an error message on failure.
pub async fn transcribe(wav_path: &str, provider: &STTProvider) -> String {
    match provider {
        STTProvider::OpenAI { api_key } => {
            transcribe_openai(wav_path, api_key).await
        }
        STTProvider::WhisperCpp { model_path } => {
            transcribe_whisper_cpp(wav_path, model_path).await
        }
    }
}

/// Transcribes using the OpenAI Whisper API.
async fn transcribe_openai(wav_path: &str, api_key: &str) -> String {
    let url = "https://api.openai.com/v1/audio/transcriptions";

    debug!("OpenAI STT request for: {}", wav_path);

    // Read the WAV file
    let wav_data = match std::fs::read(wav_path) {
        Ok(data) => data,
        Err(e) => {
            error!("Failed to read WAV file {}: {}", wav_path, e);
            return format!("Error: unable to read audio file ({})", e);
        }
    };

    // Extract just the filename for the form field
    let filename = wav_path
        .rsplit(std::path::MAIN_SEPARATOR)
        .next()
        .unwrap_or("audio.wav");

    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .expect("Failed to create HTTP client");

    let form = reqwest::multipart::Form::new()
        .text("model", "whisper-1")
        .part(
            "file",
            reqwest::multipart::Part::bytes(wav_data)
                .file_name(filename.to_string())
                .mime_str("audio/wav")
                .expect("Invalid MIME type"),
        )
        .text("language", "en")
        .text("response_format", "json");

    match client
        .post(url)
        .header("Authorization", format!("Bearer {}", api_key))
        .multipart(form)
        .send()
        .await
    {
        Ok(response) => {
            if response.status().is_success() {
                match response.json::<OpenAITranscription>().await {
                    Ok(resp) => {
                        info!("OpenAI STT transcription: {} chars", resp.text.len());
                        resp.text.trim().to_string()
                    }
                    Err(e) => {
                        error!("Failed to parse OpenAI transcription response: {}", e);
                        "There was an issue with the transcription response.".to_string()
                    }
                }
            } else {
                let status = response.status();
                let text = response.text().await.unwrap_or_default();
                error!("OpenAI STT error {}: {}", status, text);
                format!("STT error: {} ({})", status.as_u16(), status.canonical_reason().unwrap_or("unknown"))
            }
        }
        Err(e) => {
            error!("OpenAI STT request failed: {}", e);
            format!("Failed to connect to OpenAI STT: {}", e)
        }
    }
}

/// Transcribes using local whisper.cpp (via subprocess).
async fn transcribe_whisper_cpp(wav_path: &str, model_path: &str) -> String {
    // Check if whisper.cpp (whisper-main executable) is available
    let whisper_bin = match find_in_path("main")
        .or_else(|| find_in_path("whisper-main"))
    {
        Some(p) => p,
        None => {
            error!("whisper.cpp binary not found (neither 'main' nor 'whisper-main' in PATH)");
            warn!("Local STT unavailable, please install whisper.cpp or use OpenAI API");
            return "Error: whisper.cpp not found. Please install it or configure OpenAI STT."
                .to_string();
        }
    };

    info!(
        "Using whisper.cpp at: {} with model: {}",
        whisper_bin.display(),
        model_path
    );

    // Run whisper.cpp as subprocess
    let output = async move {
        tokio::process::Command::new(&whisper_bin)
            .arg("-m")
            .arg(model_path)
            .arg("-f")
            .arg(wav_path)
            .arg("-otxt")
            .output()
            .await
    }
    .await;

    match output {
        Ok(output) => {
            if output.status.success() {
                // whisper.cpp outputs to a .txt file with same base name
                let txt_path = format!("{}.txt", wav_path.trim_end_matches(".wav"));
                match std::fs::read_to_string(&txt_path) {
                    Ok(text) => {
                        let result = text.trim().to_string();
                        info!("whisper.cpp transcription: {} chars", result.len());
                        // Clean up the .txt file
                        let _ = std::fs::remove_file(&txt_path);
                        result
                    }
                    Err(e) => {
                        // Try alternative: parse stdout
                        let stdout = String::from_utf8_lossy(&output.stdout);
                        let result = stdout.trim().to_string();
                        if result.is_empty() {
                            error!("whisper.cpp produced no output and no txt file: {}", e);
                            "No transcription result.".to_string()
                        } else {
                            debug!("Using whisper.cpp stdout as transcription");
                            result
                        }
                    }
                }
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                error!(
                    "whisper.cpp failed with exit code {:?}: {}",
                    output.status.code(),
                    stderr
                );
                format!("whisper.cpp error: exit {:?}", output.status.code())
            }
        }
        Err(e) => {
            error!("Failed to execute whisper.cpp: {}", e);
            format!("Failed to run whisper.cpp: {}", e)
        }
    }
}

/// Auto-detects the best available STT provider.
///
/// Detection order:
/// 1. If OPENAI_API_KEY env var is set → OpenAI Whisper API
/// 2. If whisper.cpp binary is available in PATH → local
/// 3. Returns None (caller should handle gracefully)
pub fn auto_detect_provider() -> Option<STTProvider> {
    if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        info!("Auto-detected OpenAI API key for STT");
        return Some(STTProvider::OpenAI { api_key });
    }

    if find_in_path("main").is_some() || find_in_path("whisper-main").is_some() {
        let model_path = std::env::var("LINGUAFLOW_WHISPER_MODEL")
            .unwrap_or_else(|_| "models/ggml-base.en.bin".to_string());
        info!("Auto-detected whisper.cpp for STT");
        return Some(STTProvider::WhisperCpp { model_path });
    }

    None
}

/// Simple manual replacement for `which::which` — searches PATH for a binary.
fn find_in_path(name: &str) -> Option<PathBuf> {
    let path_var = env::var("PATH").ok()?;
    let sep = if cfg!(windows) { ";" } else { ":" };
    for dir in path_var.split(sep) {
        let candidate = PathBuf::from(dir).join(name);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}
