/// Native audio capture and playback using cpal and hound.
///
/// Supports microphone recording with voice activity detection (RMS energy
/// threshold), WAV file creation, and audio playback.
///
/// Sample rate: 16kHz mono (suitable for STT processing).

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound::{SampleFormat, WavSpec, WavWriter};
use log::{debug, error, info};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU16, Ordering};
use std::sync::{Arc, Mutex};

/// Audio provider mode selection.
#[derive(Debug, Clone, PartialEq)]
pub enum AudioProvider {
    OpenAI,
    Local,
}

/// Target audio configuration.
pub const SAMPLE_RATE: u32 = 16_000;
pub const CHANNELS: u16 = 1;
/// RMS energy threshold for voice activity detection.
pub const VAD_THRESHOLD: f64 = 0.008;
/// Number of silent samples after voice before stopping.
pub const SILENCE_SAMPLES: usize = 8000;

/// Holds state shared between record thread and audio callback.
struct RecordShared {
    is_recording: Arc<AtomicBool>,
    buffer: Arc<Mutex<Vec<f32>>>,
    rms_level: Arc<AtomicU16>,
    silence_count: Arc<Mutex<usize>>,
    has_voice: Arc<AtomicBool>,
}

/// Thread-safe audio recorder state.
pub struct AudioRecorder {
    pub is_recording: Arc<AtomicBool>,
    pub buffer: Arc<Mutex<Vec<f32>>>,
    pub rms_level: Arc<AtomicU16>,
}

impl Default for AudioRecorder {
    fn default() -> Self {
        Self {
            is_recording: Arc::new(AtomicBool::new(false)),
            buffer: Arc::new(Mutex::new(Vec::new())),
            rms_level: Arc::new(AtomicU16::new(0)),
        }
    }
}

impl AudioRecorder {
    /// Creates a new AudioRecorder instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Starts recording from the default microphone.
    ///
    /// Blocks until silence is detected (VAD), then returns the captured PCM
    /// f32 samples at 16kHz mono along with the path to the saved WAV file.
    pub fn record_until_silence(&self) -> Option<(Vec<f32>, PathBuf)> {
        let host = cpal::default_host();
        let device = host.default_input_device()?;
        debug!(
            "Using input device: {}",
            device.name().unwrap_or_else(|_| "unknown".into())
        );

        let config = self.find_matching_config(&device)?;
        let sample_rate = config.sample_rate.0;

        // Shared state for the closure
        let is_rec = Arc::clone(&self.is_recording);
        let buf = Arc::clone(&self.buffer);
        let rms = Arc::clone(&self.rms_level);
        let silence_count: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
        let has_voice: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));

        {
            let mut buffer = buf.lock().unwrap();
            buffer.clear();
        }

        is_rec.store(true, Ordering::SeqCst);

        // Create clones for the closure
        let is_rec_cb = Arc::clone(&is_rec);
        let buf_cb = Arc::clone(&buf);
        let rms_cb = Arc::clone(&rms);
        let silence_cb = Arc::clone(&silence_count);
        let voice_cb = Arc::clone(&has_voice);
        let channels = config.channels;

        let error_callback = move |err| {
            debug!("Audio stream error (non-fatal): {}", err);
        };

        let stream = device.build_input_stream(
            &config.into(),
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                if !is_rec_cb.load(Ordering::SeqCst) {
                    return;
                }
                let frame_rms = Self::compute_rms(data);
                rms_cb.store((frame_rms * 65535.0) as u16, Ordering::SeqCst);

                if frame_rms > VAD_THRESHOLD {
                    voice_cb.store(true, Ordering::SeqCst);
                    *silence_cb.lock().unwrap() = 0;
                    buf_cb.lock().unwrap().extend_from_slice(data);
                } else {
                    if voice_cb.load(Ordering::SeqCst) {
                        let s = (data.len() / channels as usize).saturating_sub(0);
                        let mut silent = silence_cb.lock().unwrap();
                        *silent += s;
                        if *silent >= SILENCE_SAMPLES {
                            // Enough silence — stop
                            is_rec_cb.store(false, Ordering::SeqCst);
                        }
                    }
                }
            },
            error_callback,
            None,
        );

        let stream = match stream {
            Ok(s) => s,
            Err(e) => {
                error!("Failed to build input stream: {}", e);
                self.is_recording.store(false, Ordering::SeqCst);
                return None;
            }
        };

        if let Err(e) = stream.play() {
            error!("Failed to play input stream: {}", e);
            self.is_recording.store(false, Ordering::SeqCst);
            return None;
        }

        info!("Recording started (RMS threshold: {})", VAD_THRESHOLD);

        // Wait for recording to complete
        let start = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs(120);

        while is_rec.load(Ordering::SeqCst) {
            std::thread::sleep(std::time::Duration::from_millis(50));
            if start.elapsed() > timeout {
                info!("Recording timeout reached (120s)");
                is_rec.store(false, Ordering::SeqCst);
                break;
            }
        }

        std::thread::sleep(std::time::Duration::from_millis(100));

        let samples = {
            let mut b = buf.lock().unwrap();
            std::mem::take(&mut *b)
        };

        if samples.is_empty() {
            info!("No voice detected during recording");
            return None;
        }

        info!(
            "Recording complete: {} samples ({:.2}s)",
            samples.len(),
            samples.len() as f64 / sample_rate as f64
        );

        let wav_path = self.save_wav(&samples, sample_rate);
        Some((samples, wav_path))
    }

    /// Attempts to find a matching input stream config preferring 16kHz.
    fn find_matching_config(&self, device: &cpal::Device) -> Option<cpal::StreamConfig> {
        if let Ok(configs) = device.supported_input_configs() {
            let configs: Vec<_> = configs.collect();

            for config in &configs {
                if config.min_sample_rate().0 <= SAMPLE_RATE
                    && config.max_sample_rate().0 >= SAMPLE_RATE
                    && config.channels() <= CHANNELS
                {
                    return Some(cpal::StreamConfig {
                        channels: CHANNELS,
                        sample_rate: cpal::SampleRate(SAMPLE_RATE),
                        buffer_size: cpal::BufferSize::Default,
                    });
                }
            }

            if let Some(cfg) = configs.into_iter().next() {
                return Some(cpal::StreamConfig {
                    channels: CHANNELS,
                    sample_rate: cpal::SampleRate(SAMPLE_RATE),
                    buffer_size: cpal::BufferSize::Default,
                });
            }
        }

        device.default_input_config().ok().map(|c| c.into())
    }

    /// Computes the RMS (root mean square) energy of audio samples.
    fn compute_rms(samples: &[f32]) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum: f64 = samples.iter().map(|s| (*s as f64).powi(2)).sum();
        (sum / samples.len() as f64).sqrt()
    }

    /// Saves PCM f32 samples to a WAV file.
    fn save_wav(&self, samples: &[f32], sample_rate: u32) -> PathBuf {
        let cache_dir = self.ensure_cache_dir();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let wav_path = cache_dir.join(format!("recording_{}.wav", timestamp));

        let spec = WavSpec {
            channels: CHANNELS,
            sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };

        let mut writer = match WavWriter::create(&wav_path, spec) {
            Ok(w) => w,
            Err(e) => {
                error!("Failed to create WAV writer: {}", e);
                return wav_path;
            }
        };

        for &sample in samples {
            let sample_i16 = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            let _ = writer.write_sample(sample_i16);
        }

        if let Err(e) = writer.finalize() {
            error!("Failed to finalize WAV file: {}", e);
        }

        debug!("WAV saved to: {:?}", wav_path);
        wav_path
    }

    /// Ensures the cache directory exists for temporary audio files.
    fn ensure_cache_dir(&self) -> PathBuf {
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
                let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
                let mut path = PathBuf::from(home);
                path.push(".linguaflow");
                path
            })
    }
}

/// Thread-safe audio playback engine.
pub struct AudioPlayer {
    pub is_playing: Arc<AtomicBool>,
}

impl Default for AudioPlayer {
    fn default() -> Self {
        Self {
            is_playing: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl AudioPlayer {
    /// Creates a new AudioPlayer instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Plays PCM f32 samples at the given sample rate.
    pub fn play(&self, samples: &[f32], sample_rate: u32) {
        let host = cpal::default_host();
        let device = match host.default_output_device() {
            Some(d) => d,
            None => {
                error!("No default output device found");
                return;
            }
        };

        let config = cpal::StreamConfig {
            channels: CHANNELS,
            sample_rate: cpal::SampleRate(sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        let sample_data: Arc<Mutex<Box<[f32]>>> =
            Arc::new(Mutex::new(samples.to_vec().into_boxed_slice()));
        let position: Arc<Mutex<usize>> = Arc::new(Mutex::new(0usize));
        let is_playing = Arc::clone(&self.is_playing);
        is_playing.store(true, Ordering::SeqCst);

        let sample_data_cb = Arc::clone(&sample_data);
        let position_cb = Arc::clone(&position);
        let is_playing_cb = Arc::clone(&is_playing);

        let error_callback = move |err| {
            debug!("Audio output error (non-fatal): {}", err);
        };

        let stream = device.build_output_stream(
            &config.into(),
            move |output: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let mut pos = position_cb.lock().unwrap();
                let data = sample_data_cb.lock().unwrap();
                for frame in output.iter_mut() {
                    if *pos < data.len() {
                        *frame = data[*pos];
                        *pos += 1;
                    } else {
                        *frame = 0.0;
                    }
                }
                if *pos >= data.len() {
                    is_playing_cb.store(false, Ordering::SeqCst);
                }
            },
            error_callback,
            None,
        );

        let stream = match stream {
            Ok(s) => s,
            Err(e) => {
                error!("Failed to build output stream: {}", e);
                is_playing.store(false, Ordering::SeqCst);
                return;
            }
        };

        if let Err(e) = stream.play() {
            error!("Failed to start output stream: {}", e);
            is_playing.store(false, Ordering::SeqCst);
            return;
        }

        // Wait for playback to finish
        while self.is_playing.load(Ordering::SeqCst) {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }

        debug!("Audio playback complete");
    }

    /// Stops playback immediately.
    pub fn stop(&self) {
        self.is_playing.store(false, Ordering::SeqCst);
    }
}

/// Combined audio state for the application UI.
pub struct AudioState {
    pub recorder: Arc<AudioRecorder>,
    pub player: Arc<AudioPlayer>,
}

impl Default for AudioState {
    fn default() -> Self {
        Self {
            recorder: Arc::new(AudioRecorder::new()),
            player: Arc::new(AudioPlayer::new()),
        }
    }
}

impl AudioState {
    /// Creates a new AudioState instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the current RMS level as a float in [0, 1] range.
    pub fn rms_level(&self) -> f32 {
        let raw = self.recorder.rms_level.load(Ordering::SeqCst);
        raw as f32 / 65535.0
    }

    /// Checks if the recorder is currently running.
    pub fn is_recording(&self) -> bool {
        self.recorder.is_recording.load(Ordering::SeqCst)
    }

    /// Checks if the player is currently outputting audio.
    pub fn is_playing(&self) -> bool {
        self.player.is_playing.load(Ordering::SeqCst)
    }
}
