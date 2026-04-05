/// Text-to-speech module supporting OpenAI TTS API and local Piper TTS.
///
/// - OpenAI: POST to /v1/audio/speech with tts-1-hd model
/// - Local: Piper TTS subprocess invocation
///
/// Generated audio is cached in ~/.linguaflow/cache/.

use log::{debug, error, info, warn};
use reqwest::Client;
use std::env;
use std::path::PathBuf;
use std::process::Command;

/// TTS provider selection.
#[derive(Debug, Clone)]
pub enum TTSProvider {
    OpenAI { api_key: String },
    Piper { model_path: String },
}

/// Text-to-speech client.
pub struct TTSClient {
    pub provider: TTSProvider,
    http_client: Client,
    cache_dir: PathBuf,
}

impl Default for TTSClient {
    fn default() -> Self {
        Self::new()
    }
}

impl TTSClient {
    /// Creates a new TTSClient with auto-detected provider.
    pub fn new() -> Self {
        let cache_dir = Self::ensure_cache_dir();

        let provider = if let Ok(api_key) = env::var("OPENAI_API_KEY") {
            info!("TTS: Using OpenAI TTS API");
            TTSProvider::OpenAI { api_key }
        } else {
            warn!("TTS: No API key found. Set OPENAI_API_KEY for OpenAI TTS or configure Piper.");
            let model_path = env::var("LINGUAFLOW_PIPER_MODEL")
                .unwrap_or_else(|_| "piper-voices/en_US-lessac-medium.onnx".to_string());
            TTSProvider::Piper { model_path }
        };

        Self {
            provider,
            http_client: Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .expect("Failed to create HTTP client"),
            cache_dir,
        }
    }

    /// Creates a TTSClient with a specific provider.
    pub fn with_provider(provider: TTSProvider) -> Self {
        let cache_dir = Self::ensure_cache_dir();
        Self {
            provider,
            http_client: Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .expect("Failed to create HTTP client"),
            cache_dir,
        }
    }

    /// Synthesizes speech from text, returning PCM f32 samples.
    ///
    /// # Arguments
    /// * `text` — the text to synthesize
    /// * `voice` — the voice/voice code (e.g., "alloy", "nova", or language code for Piper)
    ///
    /// # Returns
    /// Vec<f32> samples at 22050Hz mono, or empty vec on failure.
    pub async fn synthesize(&self, text: &str, voice: &str) -> Vec<f32> {
        match &self.provider {
            TTSProvider::OpenAI { api_key } => {
                self.synthesize_openai(text, voice, api_key).await
            }
            TTSProvider::Piper { model_path } => {
                self.synthesize_piper(text, voice, model_path).await
            }
        }
    }

    /// Synthesizes speech using OpenAI TTS API.
    async fn synthesize_openai(&self, text: &str, voice: &str, api_key: &str) -> Vec<f32> {
        // Check cache first
        if let Some(cached) = self.load_from_cache(text, voice, "openai") {
            debug!("OpenAI TTS: using cached audio");
            return cached;
        }

        let url = "https://api.openai.com/v1/audio/speech";

        let body = serde_json::json!({
            "model": "tts-1-hd",
            "input": text,
            "voice": voice,
            "response_format": "wav",
        });

        debug!(
            "OpenAI TTS: requesting {} chars with voice '{}'",
            text.len(),
            voice
        );

        match self
            .http_client
            .post(url)
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&body)
            .send()
            .await
        {
            Ok(response) => {
                if response.status().is_success() {
                    match response.bytes().await {
                        Ok(wav_bytes) => {
                            // Parse WAV to PCM f32
                            let samples = Self::wav_to_pcm_f32(&wav_bytes);
                            if !samples.is_empty() {
                                // Cache the result
                                self.save_to_cache(text, voice, "openai", &wav_bytes);
                            }
                            info!("OpenAI TTS: {} samples generated", samples.len());
                            samples
                        }
                        Err(e) => {
                            error!("Failed to read OpenAI TTS response body: {}", e);
                            vec![]
                        }
                    }
                } else {
                    let status = response.status();
                    let text_err = response.text().await.unwrap_or_default();
                    error!("OpenAI TTS error {}: {}", status, text_err);
                    vec![]
                }
            }
            Err(e) => {
                error!("OpenAI TTS request failed: {}", e);
                vec![]
            }
        }
    }

    /// Synthesizes speech using Piper TTS (local).
    async fn synthesize_piper(&self, text: &str, voice: &str, model_path: &str) -> Vec<f32> {
        // Check cache first
        if let Some(cached) = self.load_from_cache(text, voice, "piper") {
            debug!("Piper TTS: using cached audio");
            return cached;
        }

        // Find piper binary
        let piper_bin = match find_binary_in_path("piper") {
            Some(p) => p,
            None => {
                error!("Piper binary not found in PATH");
                warn!("Local TTS unavailable. Please install Piper or use OpenAI TTS.");
                return vec![];
            }
        };

        let wav_path = self.cache_dir.join(format!("piper_{}.wav", hash_str(text)));

        info!(
            "Piper TTS: synthesizing {} chars with model {}",
            text.len(),
            model_path
        );

        let output = tokio::task::spawn_blocking({
            let model_path = model_path.to_string();
            let text = text.to_string();
            let wav_path = wav_path.clone();
            move || {
                Command::new(&piper_bin)
                    .arg("--model")
                    .arg(&model_path)
                    .arg("--output_file")
                    .arg(&wav_path)
                    .arg("--json-input")
                    .stdin(std::process::Stdio::piped())
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped())
                    .spawn()
                    .and_then(|mut child| {
                        if let Some(mut stdin) = child.stdin.take() {
                            use std::io::Write;
                            let _ = stdin.write_all(
                                serde_json::json!({
                                    "text": text
                                })
                                .to_string()
                                .as_bytes(),
                            );
                        }
                        child.wait_with_output()
                    })
            }
        })
        .await
        .ok()
        .and_then(|r| match r {
            Ok(output) => Some(output),
            Err(e) => {
                error!("Piper subprocess error: {}", e);
                None
            }
        });

        match output {
            Some(output) => {
                if output.status.success() {
                    // Read WAV file
                    match std::fs::read(&wav_path) {
                        Ok(wav_bytes) => {
                            let samples = Self::wav_to_pcm_f32(&wav_bytes);
                            if !samples.is_empty() {
                                self.save_to_cache(text, voice, "piper", &wav_bytes);
                            }
                            info!("Piper TTS: {} samples generated", samples.len());
                            samples
                        }
                        Err(e) => {
                            error!("Failed to read Piper WAV output: {}", e);
                            vec![]
                        }
                    }
                } else {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    error!("Piper TTS failed: {}", stderr);
                    vec![]
                }
            }
            None => {
                error!("Piper TTS execution failed");
                vec![]
            }
        }
    }

    /// Converts WAV bytes to PCM f32 samples.
    fn wav_to_pcm_f32(wav_bytes: &[u8]) -> Vec<f32> {
        use std::io::Cursor;
        let cursor = Cursor::new(wav_bytes);
        let reader = match hound::WavReader::new(cursor) {
            Ok(r) => r,
            Err(e) => {
                error!("Failed to parse WAV: {}", e);
                return vec![];
            }
        };

        let spec = reader.spec();
        let samples_len = reader.len() as usize;
        let mut samples = Vec::with_capacity(samples_len);

        match spec.sample_format {
            hound::SampleFormat::Int => {
                match spec.bits_per_sample {
                    16 => {
                        for sample in reader.into_samples::<i16>() {
                            if let Ok(s) = sample {
                                samples.push(s as f32 / i16::MAX as f32);
                            }
                        }
                    }
                    32 => {
                        for sample in reader.into_samples::<i32>() {
                            if let Ok(s) = sample {
                                samples.push(s as f32 / i32::MAX as f32);
                            }
                        }
                    }
                    _ => {
                        error!("Unsupported WAV bits per sample: {}", spec.bits_per_sample);
                    }
                }
            }
            hound::SampleFormat::Float => {
                for sample in reader.into_samples::<f32>() {
                    if let Ok(s) = sample {
                        samples.push(s);
                    }
                }
            }
        }

        samples
    }

    /// Loads synthesized audio from cache if available.
    fn load_from_cache(&self, text: &str, voice: &str, provider: &str) -> Option<Vec<f32>> {
        let cache_key = format!("{}_{}_{}", provider, voice, hash_str(text));
        let cache_file = self.cache_dir.join(format!("{}.wav", cache_key));

        if cache_file.exists() {
            if let Ok(wav_bytes) = std::fs::read(&cache_file) {
                let samples = Self::wav_to_pcm_f32(&wav_bytes);
                if !samples.is_empty() {
                    debug!("TTS cache hit for key: {}", cache_key);
                    return Some(samples);
                }
            }
        }
        None
    }

    /// Saves synthesized audio to cache.
    fn save_to_cache(&self, text: &str, voice: &str, provider: &str, wav_bytes: &[u8]) {
        let cache_key = format!("{}_{}_{}", provider, voice, hash_str(text));
        let cache_file = self.cache_dir.join(format!("{}.wav", cache_key));

        if let Err(e) = std::fs::write(&cache_file, wav_bytes) {
            debug!("Failed to cache TTS audio: {}", e);
        } else {
            debug!("TTS audio cached: {}", cache_key);
        }
    }

    /// Ensures the cache directory exists.
    fn ensure_cache_dir() -> PathBuf {
        let mut path = Self::linguaflow_dir();
        path.push("cache");
        let _ = std::fs::create_dir_all(&path);
        path
    }

    /// Returns the path to the ~/.linguaflow/ data directory.
    fn linguaflow_dir() -> PathBuf {
        dirs::data_dir()
            .map(|d| d.join("LinguaFlow"))
            .unwrap_or_else(|| {
                let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
                let mut path = PathBuf::from(home);
                path.push(".linguaflow");
                path
            })
    }

    /// Clears the TTS cache directory.
    pub fn clear_cache(&self) {
        if let Ok(entries) = std::fs::read_dir(&self.cache_dir) {
            let mut count = 0;
            for entry in entries.filter_map(Result::ok) {
                if entry.path().extension().map_or(false, |ext| ext == "wav") {
                    let _ = std::fs::remove_file(entry.path());
                    count += 1;
                }
            }
            info!("Cleared {} cached TTS files", count);
        }
    }
}

/// Simple hash function for cache keys.
fn hash_str(s: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Simple replacement for `which::which` — searches PATH for a binary.
fn find_binary_in_path(name: &str) -> Option<std::path::PathBuf> {
    let path_var = env::var("PATH").ok()?;
    let sep = if cfg!(windows) { ";" } else { ":" };
    for dir in path_var.split(sep) {
        let candidate = std::path::PathBuf::from(dir).join(name);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}
