/// Application module — main egui app state and UI.
///
/// Contains the LinguaFlowApp struct which implements eframe::App,
/// managing the full conversation flow, settings panel, and UI rendering.

use crate::audio::AudioState;
use crate::llm::{LLMClient, LLMProvider, Message as LLMMessage};
use crate::memory::UserMemory;
use crate::stt;
use crate::stt::STTProvider;
use crate::tts::TTSClient;
use crate::tts::TTSProvider as TTSProviderEnum;
use eframe::egui;
use egui::{Align, Color32, Layout, RichText, ScrollArea, TextEdit, Ui, Vec2};
use log::{debug, error, info};
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::Instant;

/// A single chat message displayed in the UI.
#[derive(Clone)]
pub struct ChatMessage {
    /// "user" or "assistant"
    pub role: String,
    /// The message text content
    pub content: String,
    /// Optional corrections detected in the message
    pub corrections: Vec<Correction>,
    /// Optional vocabulary words highlighted
    pub vocabulary: Vec<String>,
    /// When the message was sent
    pub timestamp: std::time::SystemTime,
    /// Whether the TTS audio has been generated
    pub tts_ready: bool,
}

/// A grammar/vocabulary correction.
#[derive(Clone, Debug)]
pub struct Correction {
    pub original: String,
    pub corrected: String,
    pub explanation: String,
}

/// Topic suggestion for conversation starters.
#[derive(Clone, Debug)]
pub struct TopicSuggestion {
    pub label: String,
    pub prompt: String,
}

/// Pending LLM response shared between background thread and UI.
struct PendingResponse {
    rx: Receiver<Result<(String, Vec<f32>), String>>,
    start: Instant,
}

/// Shared tokio runtime — created once, reused for all async calls.
/// Lazily initialized on first use to avoid blocking UI at startup.
static TOKIO_RUNTIME: std::sync::OnceLock<Arc<tokio::runtime::Runtime>> = std::sync::OnceLock::new();

fn get_runtime() -> Arc<tokio::runtime::Runtime> {
    TOKIO_RUNTIME
        .get_or_init(|| {
            Arc::new(
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("Failed to create tokio runtime"),
            )
        })
        .clone()
}

/// Default topic suggestions to show when conversation stalls.
fn default_suggestions(target_lang: &str) -> Vec<TopicSuggestion> {
    vec![
        TopicSuggestion {
            label: "🌍 Travel".to_string(),
            prompt: format!(
                "Let's talk about traveling to a country where {} is spoken.",
                target_lang
            ),
        },
        TopicSuggestion {
            label: "🍕 Food".to_string(),
            prompt: format!(
                "Describe your favorite meal in {}.",
                target_lang
            ),
        },
        TopicSuggestion {
            label: "🎬 Movies".to_string(),
            prompt: format!(
                "What's a good movie in {}? Let's discuss it!",
                target_lang
            ),
        },
        TopicSuggestion {
            label: "💼 Work".to_string(),
            prompt: format!(
                "Let's practice professional {}.",
                target_lang
            ),
        },
        TopicSuggestion {
            label: "📚 Stories".to_string(),
            prompt: format!(
                "Tell me a short story in {}.",
                target_lang
            ),
        },
    ]
}

/// Wrapper around TTSClient that supports Clone via Arc.
#[derive(Clone)]
struct SharedTTSClient {
    inner: Arc<TTSClient>,
}

impl std::ops::Deref for SharedTTSClient {
    type Target = TTSClient;
    fn deref(&self) -> &TTSClient {
        &self.inner
    }
}

/// The main LinguaFlow application state.
pub struct LinguaFlowApp {
    /// Conversation messages displayed in the chat
    pub messages: Vec<ChatMessage>,
    /// Current user text input
    pub current_input: String,
    /// Whether we're actively recording audio
    pub is_recording: bool,
    /// Whether we're currently processing a message
    pub is_processing: bool,
    /// Target language being learned (e.g., "french", "spanish")
    pub target_lang: String,
    /// Provider mode: "auto", "api", or "local"
    pub provider_mode: String,
    /// Whether the settings panel is open
    pub show_settings: bool,
    /// Whether the side panel (stats, vocabulary) is visible
    pub show_panel: bool,
    /// OpenAI / Whisper API key
    pub openai_api_key: String,
    /// OpenRouter API key
    pub openrouter_api_key: String,
    /// Audio state (recorder, player, levels)
    pub audio: AudioState,
    /// SQLite memory store
    pub memory: UserMemory,
    /// LLM client for chat completions
    pub llm: LLMClient,
    /// TTS client for voice output (wrapped in Arc for sharing)
    tts: SharedTTSClient,
    /// STT provider configuration
    pub stt_provider: STTProvider,
    /// Topic suggestions to display
    pub suggestions: Vec<TopicSuggestion>,
    /// Current user profile name
    pub user_name: String,
    /// Learner level
    pub user_level: String,
    /// User interests (comma-separated)
    pub user_interests: String,
    /// Whether to scroll the chat to bottom on next frame
    pub scroll_to_bottom: bool,
    /// Current LLM model name
    pub llm_model: String,
    /// Last error displayed
    pub last_error: Option<String>,
    /// Timer for recording feedback
    pub recording_start: Option<Instant>,
    /// Pending LLM response from background thread
    pending_response: Option<PendingResponse>,
}

impl Default for LinguaFlowApp {
    fn default() -> Self {
        Self::new()
    }
}

impl LinguaFlowApp {
    /// Creates a new LinguaFlowApp with default state.
    pub fn new() -> Self {
        let memory = UserMemory::new();
        let llm = LLMClient::new();
        let tts_client = TTSClient::new();

        // Detect STT provider
        let stt_provider = stt::auto_detect_provider().unwrap_or_else(|| {
            let openai_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
            if openai_key.is_empty() {
                STTProvider::WhisperCpp {
                    model_path: "models/ggml-base.en.bin".to_string(),
                }
            } else {
                STTProvider::OpenAI {
                    api_key: openai_key,
                }
            }
        });

        // Load saved profile if exists
        let profile = memory.load_profile();
        let target_lang = profile
            .as_ref()
            .map(|p| p.target_lang.clone())
            .unwrap_or_else(|| "french".to_string());
        let user_name = profile
            .as_ref()
            .map(|p| p.name.clone())
            .unwrap_or_default();
        let user_level = profile
            .as_ref()
            .map(|p| p.level.clone())
            .unwrap_or_else(|| "beginner".to_string());
        let user_interests = profile
            .as_ref()
            .map(|p| p.interests.clone())
            .unwrap_or_default();

        let llm_model = llm.current_model().to_string();

        // Determine provider mode
        let provider_mode = match &llm.provider {
            LLMProvider::OpenRouter { .. } => "api".to_string(),
            LLMProvider::Ollama { .. } => "local".to_string(),
        };

        let openai_api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
        let openrouter_api_key = std::env::var("OPENROUTER_API_KEY").unwrap_or_default();

        let welcome_msg = ChatMessage {
            role: "assistant".to_string(),
            content: format!(
                "Welcome to LinguaFlow! 🎓 I'm your AI language tutor for {}.\n\n\
                 I'll help you practice through natural conversation. Feel free to:\
                 \n\n• Type messages to practice writing\
                 \n• Click the microphone button 🎤 to speak\
                 \n• I'll correct your mistakes gently and help you learn\
                 \n\nLet's start! How are you today in {}?",
                target_lang, target_lang
            ),
            corrections: vec![],
            vocabulary: vec![],
            timestamp: std::time::SystemTime::now(),
            tts_ready: false,
        };

        memory.save_conversation("assistant", &welcome_msg.content, &target_lang);

        Self {
            messages: vec![welcome_msg],
            current_input: String::new(),
            is_recording: false,
            is_processing: false,
            target_lang,
            provider_mode,
            show_settings: false,
            show_panel: false,
            openai_api_key,
            openrouter_api_key,
            audio: AudioState::new(),
            memory,
            llm,
            tts: SharedTTSClient {
                inner: Arc::new(tts_client),
            },
            stt_provider,
            suggestions: vec![],
            user_name,
            user_level,
            user_interests,
            scroll_to_bottom: true,
            llm_model,
            last_error: None,
            recording_start: None,
            pending_response: None,
        }
    }

    /// Sends a user message through the LLM pipeline.
    fn send_message(&mut self) {
        let input = self.current_input.trim().to_string();
        if input.is_empty() || self.is_processing {
            return;
        }

        self.current_input.clear();
        self.is_processing = true;
        self.last_error = None;

        // Add user message to UI
        let user_msg = ChatMessage {
            role: "user".to_string(),
            content: input.clone(),
            corrections: vec![],
            vocabulary: vec![],
            timestamp: std::time::SystemTime::now(),
            tts_ready: false,
        };
        self.messages.push(user_msg);

        // Save to memory
        self.memory
            .save_conversation("user", &input, &self.target_lang);

        // Build LLM messages from conversation history
        let mut llm_messages: Vec<LLMMessage> = Vec::new();
        for msg in &self.messages {
            llm_messages.push(LLMMessage {
                role: msg.role.clone(),
                content: msg.content.clone(),
            });
        }

        let target_lang = self.target_lang.clone();
        let memory = self.memory.clone();
        let llm = self.llm.clone();
        let tts = self.tts.clone();

        let (tx, rx): (Sender<Result<(String, Vec<f32>), String>>, _) = mpsc::channel();

        // Spawn background thread for async LLM + TTS
        std::thread::spawn(move || {
            let rt = get_runtime();

            let response = rt.block_on(async {
                llm.chat(llm_messages).await
            });

            // Generate TTS for the response
            let tts_samples = rt.block_on(async {
                tts.inner.as_ref().synthesize(&response, "alloy").await
            });

            // Save assistant message to memory
            memory.save_conversation("assistant", &response, &target_lang);

            let _ = tx.send(Ok((response, tts_samples)));
        });

        // Store the receiver for polling in update()
        self.pending_response = Some(PendingResponse {
            rx,
            start: Instant::now(),
        });
    }

    /// Polls the pending LLM response and updates the UI when ready.
    fn poll_pending_response(&mut self) {
        if let Some(pending) = &self.pending_response {
            match pending.rx.try_recv() {
                Ok(Ok((response, tts_samples))) => {
                    // Check if response is an error
                    if response.starts_with("error:") || response.starts_with("Failed") {
                        self.last_error = Some(response.clone());
                    }

                    let assistant_msg = ChatMessage {
                        role: "assistant".to_string(),
                        content: response.clone(),
                        corrections: vec![],
                        vocabulary: vec![],
                        timestamp: std::time::SystemTime::now(),
                        tts_ready: !tts_samples.is_empty(),
                    };
                    self.messages.push(assistant_msg);
                    self.scroll_to_bottom = true;
                    self.is_processing = false;
                    self.pending_response = None;

                    // Play TTS if we got audio
                    if !tts_samples.is_empty() && !self.audio.is_playing() {
                        info!("Playing TTS response: {} samples", tts_samples.len());
                        let player = self.audio.player.clone();
                        let samples = tts_samples.clone();
                        std::thread::spawn(move || {
                            player.play(&samples, 22050);
                        });
                    }
                }
                Ok(Err(e)) => {
                    // Background thread reported an error
                    self.last_error = Some(e);
                    self.is_processing = false;
                    self.pending_response = None;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    // Still waiting...
                    if pending.start.elapsed().as_secs() > 60 {
                        self.last_error =
                            Some("Response timed out after 60 seconds.".to_string());
                        self.is_processing = false;
                        self.pending_response = None;
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.last_error =
                        Some("Connection to AI service was lost. Please try again.".to_string());
                    self.is_processing = false;
                    self.pending_response = None;
                }
            }
        }
    }

    /// Records audio, transcribes it with STT, and adds the result as a message.
    fn handle_voice_input(&mut self) {
        if self.is_recording {
            // Stop recording
            self.audio
                .recorder
                .is_recording
                .store(false, std::sync::atomic::Ordering::SeqCst);
            self.is_recording = false;
            self.recording_start = None;
            return;
        } else if self.is_processing {
            return;
        }

        self.recording_start = Some(Instant::now());
        self.is_recording = true;
    }

    /// Saves the current user profile.
    fn save_profile(&mut self) {
        self.memory.update_profile(
            &self.user_name,
            &self.target_lang,
            &self.user_level,
            &self.user_interests,
        );
        self.llm
            .set_system_prompt(&self.user_level, &self.target_lang);
        self.suggestions = default_suggestions(&self.target_lang);
        info!("Profile saved and LLM prompt updated");
    }

    /// Clears all conversation history from UI and storage.
    fn clear_history(&mut self) {
        self.messages.clear();
        self.memory.clear_conversations();

        self.messages.push(ChatMessage {
            role: "assistant".to_string(),
            content: format!(
                "Conversation history cleared. Let's start fresh! What would you like to practice in {}?",
                self.target_lang
            ),
            corrections: vec![],
            vocabulary: vec![],
            timestamp: std::time::SystemTime::now(),
            tts_ready: false,
        });
    }

    /// Uses a topic suggestion as a conversation starter.
    fn use_suggestion(&mut self, prompt: &str) {
        self.current_input = prompt.to_string();
        self.send_message();
    }

    /// Renders the top bar with app title and controls.
    fn render_top_bar(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.heading(
                RichText::new("LinguaFlow")
                    .color(Color32::from_rgb(130, 200, 255))
                    .size(20.0),
            );

            ui.separator();

            // Language selector
            ui.label("Learning:");
            egui::ComboBox::from_id_salt("target_lang")
                .selected_text(&self.target_lang)
                .width(120.0)
                .show_ui(ui, |ui| {
                    for lang in &[
                        "french", "spanish", "german", "italian", "portuguese",
                        "english", "japanese", "chinese", "korean", "russian", "arabic",
                    ] {
                        ui.selectable_value(
                            &mut self.target_lang,
                            lang.to_string(),
                            lang.to_string(),
                        );
                    }
                });

            ui.separator();
            ui.label(format!("📊 {}", self.user_level));

            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if ui.button("⚙️").clicked() {
                    self.show_settings = !self.show_settings;
                }
                if ui.button("📋").clicked() {
                    self.show_panel = !self.show_panel;
                }
            });
        });
        ui.separator();
    }

    /// Renders the chat messages area.
    fn render_chat_area(&mut self, ui: &mut Ui) {
        ScrollArea::vertical()
            .auto_shrink([false; 2])
            .stick_to_bottom(self.scroll_to_bottom)
            .show(ui, |ui| {
                for (i, msg) in self.messages.iter().enumerate() {
                    self.render_message(ui, i, msg);
                    ui.add_space(4.0);
                }

                // Typing indicator
                if self.is_processing {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("💬").color(Color32::GRAY).size(16.0));
                        ui.label(
                            RichText::new("AI is thinking...")
                                .color(Color32::GRAY)
                                .italics()
                                .size(14.0),
                        );
                    });
                }
            });
        self.scroll_to_bottom = false;
    }

    /// Renders a single chat message.
    fn render_message(&self, ui: &mut Ui, _index: usize, msg: &ChatMessage) {
        let is_user = msg.role == "user";

        if is_user {
            ui.with_layout(Layout::right_to_left(Align::TOP), |ui| {
                ui.label(RichText::new("🗣️ You").size(11.0).color(Color32::GRAY));
                egui::Frame::none()
                    .fill(Color32::from_rgb(40, 60, 100))
                    .rounding(10.0)
                    .inner_margin(egui::Margin::same(10.0))
                    .show(ui, |ui| {
                        ui.set_max_width(ui.available_width() * 0.8);
                        ui.colored_label(
                            Color32::from_rgb(200, 220, 255),
                            RichText::new(&msg.content).size(14.0),
                        );
                    });
            });
        } else {
            ui.with_layout(Layout::left_to_right(Align::TOP), |ui| {
                ui.label(
                    RichText::new("🤖 AI")
                        .size(11.0)
                        .color(Color32::from_rgb(130, 200, 255)),
                );
                egui::Frame::none()
                    .fill(Color32::from_rgb(30, 35, 45))
                    .rounding(10.0)
                    .inner_margin(egui::Margin::same(10.0))
                    .show(ui, |ui| {
                        ui.set_max_width(ui.available_width() * 0.8);
                        ui.colored_label(Color32::LIGHT_GRAY, RichText::new(&msg.content).size(14.0));
                    });
            });

            // TTS audio indicator
            if msg.tts_ready && ui
                .small_button("🔊 Play")
                .clicked()
            {
                // Re-synthesize TTS for playback
                let tts = self.tts.clone();
                let content = msg.content.clone();
                let player = self.audio.player.clone();
                std::thread::spawn(move || {
                    let rt = match tokio::runtime::Runtime::new() {
                        Ok(r) => r,
                        Err(_) => return,
                    };
                    let samples = rt.block_on(async {
                        tts.inner.as_ref().synthesize(&content, "alloy").await
                    });
                    if !samples.is_empty() {
                        player.play(&samples, 22050);
                    }
                });
            }
        }

        // Show corrections if any
        if !msg.corrections.is_empty() {
            ui.indent("", |ui| {
                for correction in &msg.corrections {
                    ui.horizontal(|ui| {
                        ui.label(
                            RichText::new(format!("❌ {}", correction.original))
                                .color(Color32::from_rgb(255, 120, 120))
                                .size(12.0),
                        );
                        ui.label(
                            RichText::new(format!("✅ {}", correction.corrected))
                                .color(Color32::from_rgb(120, 255, 120))
                                .size(12.0),
                        );
                    });
                    if !correction.explanation.is_empty() {
                        ui.label(
                            RichText::new(&correction.explanation)
                                .color(Color32::GRAY)
                                .size(11.0)
                                .italics(),
                        );
                    }
                }
            });
        }
    }

    /// Renders the input area (text input + buttons).
    fn render_input_area(&mut self, ui: &mut Ui) {
        ui.separator();

        ui.horizontal(|ui| {
            // Microphone button
            let mic_color = if self.is_recording {
                Color32::RED
            } else {
                Color32::GRAY
            };

            let mic_btn = ui
                .add(
                    egui::Button::new(if self.is_recording { "⏹️" } else { "🎤" })
                        .fill(mic_color)
                        .min_size(Vec2::new(40.0, 36.0)),
                )
                .clicked();

            if mic_btn {
                self.handle_voice_input();
            }

            // Recording time indicator
            if self.is_recording {
                if let Some(start) = self.recording_start {
                    let elapsed = start.elapsed().as_secs();
                    ui.label(
                        RichText::new(format!("🔴 Recording ({:.1}s)", elapsed as f64))
                            .color(Color32::RED)
                            .size(13.0),
                    );
                }
            }

            // Audio level meter when recording
            if self.is_recording {
                let rms = self.audio.rms_level();
                let meter_width = 80.0;
                let filled = rms * meter_width;
                let rect = ui.allocate_exact_size(
                    Vec2::new(meter_width, 14.0),
                    egui::Sense::hover(),
                ).0;
                ui.painter().rect_filled(rect, 3.0, Color32::DARK_GRAY);
                if filled > 0.0 {
                    let filled_rect = egui::Rect::from_min_size(
                        rect.min,
                        Vec2::new(filled, rect.height()),
                    );
                    ui.painter()
                        .rect_filled(filled_rect, 3.0, Color32::GREEN);
                }
            }

            ui.add_space(8.0);

            // Text input
            let text_edit_response = ui.add_sized(
                ui.available_size(),
                TextEdit::multiline(&mut self.current_input)
                    .desired_width(f32::INFINITY)
                    .desired_rows(1)
                    .hint_text(format!(
                        "Type in {} or press Enter to send...",
                        self.target_lang
                    )),
            );

            // Send button
            let send_btn = ui
                .add_enabled(
                    !self.is_processing && !self.current_input.trim().is_empty(),
                    egui::Button::new("➤")
                        .fill(if self.is_processing {
                            Color32::DARK_GRAY
                        } else {
                            Color32::from_rgb(40, 120, 200)
                        })
                        .min_size(Vec2::new(40.0, 36.0)),
                )
                .clicked();

            if send_btn
                || (text_edit_response.has_focus()
                    && ui.input(|i| i.key_pressed(egui::Key::Enter))
                    && !ui.input(|i| i.modifiers.shift))
            {
                self.send_message();
            }
        });

        // Show last error if present
        if let Some(err) = self.last_error.take() {
            ui.horizontal(|ui| {
                ui.colored_label(
                    Color32::from_rgb(255, 100, 100),
                    RichText::new(format!("⚠️ {}", err)).size(12.0),
                );
                if !ui.button("Dismiss").clicked() {
                    self.last_error = Some(err);
                }
            });
            if self.last_error.is_none() {
                // Error was dismissed
            }
        }
    }

    /// Renders the settings panel (side panel on the right).
    fn render_settings(&mut self, ctx: &egui::Context) {
        egui::SidePanel::right("settings_panel")
            .default_width(350.0)
            .resizable(true)
            .show(ctx, |ui| {
                ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("⚙️ Settings");
                    ui.separator();

                    // User Profile
                    ui.collapsing("👤 User Profile", |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Name:");
                            ui.text_edit_singleline(&mut self.user_name);
                        });

                        ui.label("Level:");
                        egui::ComboBox::from_label("level")
                            .selected_text(&self.user_level)
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    &mut self.user_level,
                                    "beginner".to_string(),
                                    "A1-A2 Beginner",
                                );
                                ui.selectable_value(
                                    &mut self.user_level,
                                    "intermediate".to_string(),
                                    "B1-B2 Intermediate",
                                );
                                ui.selectable_value(
                                    &mut self.user_level,
                                    "advanced".to_string(),
                                    "C1-C2 Advanced",
                                );
                            });

                        ui.horizontal(|ui| {
                            ui.label("Interests:");
                            ui.text_edit_singleline(&mut self.user_interests);
                        });

                        ui.horizontal(|ui| {
                            ui.label("Language:");
                            egui::ComboBox::from_label("settings_lang")
                                .selected_text(&self.target_lang)
                                .width(120.0)
                                .show_ui(ui, |ui| {
                                    for lang in &[
                                        "french", "spanish", "german", "italian",
                                        "portuguese", "english", "japanese", "chinese",
                                        "korean", "russian", "arabic",
                                    ] {
                                        ui.selectable_value(
                                            &mut self.target_lang,
                                            lang.to_string(),
                                            lang.to_string(),
                                        );
                                    }
                                });
                        });

                        if ui.button("💾 Save Profile").clicked() {
                            self.save_profile();
                        }
                    });

                    // AI Provider
                    ui.collapsing("🔌 AI Provider", |ui| {
                        ui.label(format!(
                            "Current: {} ({})",
                            self.llm.provider_name(),
                            self.llm.current_model()
                        ));

                        ui.label("Provider Mode:");
                        egui::ComboBox::from_label("prov_mode")
                            .selected_text(&self.provider_mode)
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    &mut self.provider_mode,
                                    "auto".to_string(),
                                    "Auto-detect",
                                );
                                ui.selectable_value(
                                    &mut self.provider_mode,
                                    "api".to_string(),
                                    "API (OpenRouter)",
                                );
                                ui.selectable_value(
                                    &mut self.provider_mode,
                                    "local".to_string(),
                                    "Local (Ollama)",
                                );
                            });

                        ui.horizontal(|ui| {
                            ui.label("OpenRouter Key:");
                            ui.text_edit_singleline(&mut self.openrouter_api_key);
                        });

                        ui.horizontal(|ui| {
                            ui.label("OpenAI Key:");
                            ui.text_edit_singleline(&mut self.openai_api_key);
                        });

                        // Model selector
                        ui.horizontal(|ui| {
                            ui.label("Model:");
                            egui::ComboBox::from_label("model_select")
                                .selected_text(self.llm.current_model())
                                .show_ui(ui, |ui| {
                                    for model in self.llm.supported_models() {
                                        ui.selectable_value(
                                            &mut self.llm_model,
                                            model.to_string(),
                                            model.to_string(),
                                        );
                                    }
                                });
                        });

                        if ui.button("🔄 Apply Settings").clicked() {
                            if !self.openrouter_api_key.trim().is_empty() {
                                self.llm.switch_to_openrouter(
                                    self.openrouter_api_key.trim().to_string(),
                                );
                            }
                            if self.llm.current_model() != self.llm_model {
                                self.llm.switch_model(&self.llm_model);
                            }
                            self.save_profile();
                        }
                    });

                    // STT Settings
                    ui.collapsing("🎙️ Speech-to-Text", |ui| {
                        ui.label(format!("Provider: {:?}", self.stt_provider));
                    });

                    // TTS Settings
                    ui.collapsing("🔊 Text-to-Speech", |ui| {
                        ui.label(format!("Provider: {:?}", self.tts.provider));
                        if ui.button("🗑️ Clear TTS Cache").clicked() {
                            self.tts.clear_cache();
                        }
                    });

                    // Data Management
                    ui.collapsing("💾 Data", |ui| {
                        ui.label(format!("Database: {}", self.memory.db_path));

                        let vocab_count =
                            self.memory.vocabulary_count(&self.target_lang);
                        let error_count = self.memory.error_count();
                        ui.label(format!(
                            "Vocabulary entries ({}): {}",
                            self.target_lang, vocab_count
                        ));
                        ui.label(format!("Tracked errors: {}", error_count));

                        if ui.button("🗑️ Clear History").clicked() {
                            self.clear_history();
                        }
                    });

                    // Keyboard shortcuts
                    ui.collapsing("⌨️ Shortcuts", |ui| {
                        ui.label("Enter: Send message");
                        ui.label("Shift+Enter: New line");
                        ui.label("🎤: Toggle voice recording");
                    });
                });
            });
    }

    /// Renders the side panel (left) with stats and vocabulary.
    fn render_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("stats_panel")
            .default_width(250.0)
            .resizable(true)
            .show(ctx, |ui| {
                ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("📊 Progress");
                    ui.separator();

                    ui.horizontal(|ui| {
                        ui.label("Messages:");
                        ui.label(format!("{}", self.messages.len() - 1));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Vocabulary:");
                        ui.label(format!(
                            "{} words",
                            self.memory.vocabulary_count(&self.target_lang)
                        ));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Tracked errors:");
                        ui.label(format!("{}", self.memory.error_count()));
                    });

                    ui.separator();

                    // Frequent errors
                    ui.heading("🔴 Frequent Errors");
                    let errors = self.memory.get_frequent_errors(10);
                    if errors.is_empty() {
                        ui.label(
                            RichText::new("No errors tracked yet.")
                                .italics()
                                .size(12.0),
                        );
                    } else {
                        for err in &errors {
                            ui.vertical(|ui| {
                                ui.label(
                                    RichText::new(format!("❌ {}", err.error))
                                        .color(Color32::from_rgb(255, 120, 120))
                                        .size(12.0),
                                );
                                ui.label(
                                    RichText::new(format!("✅ {}", err.correction))
                                        .color(Color32::from_rgb(120, 255, 120))
                                        .size(12.0),
                                );
                                ui.label(
                                    RichText::new(format!("×{} times", err.count))
                                        .color(Color32::GRAY)
                                        .size(10.0),
                                );
                                ui.separator();
                            });
                        }
                    }

                    ui.separator();

                    // Topic suggestions
                    ui.heading("💡 Try These Topics");
                    let lang = self.target_lang.clone();
                    let suggestions = if self.suggestions.is_empty() {
                        default_suggestions(&lang)
                    } else {
                        self.suggestions.clone()
                    };

                    for suggestion in &suggestions {
                        if ui
                            .button(RichText::new(&suggestion.label).size(13.0))
                            .clicked()
                        {
                            self.use_suggestion(&suggestion.prompt);
                        }
                    }
                });
            });
    }
}

impl eframe::App for LinguaFlowApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Dark theme
        ctx.set_visuals(egui::Visuals::dark());

        // Poll for pending LLM response (non-blocking)
        self.poll_pending_response();

        // Show panels if enabled
        if self.show_settings {
            self.render_settings(ctx);
        }
        if self.show_panel {
            self.render_panel(ctx);
        }

        // Main layout
        egui::CentralPanel::default().show(ctx, |ui| {
            // Top bar
            self.render_top_bar(ui);

            ui.vertical(|ui| {
                // Chat area (takes remaining space)
                let available = ui.available_size();
                ui.allocate_ui_with_layout(
                    Vec2::new(available.x, available.y - 90.0),
                    Layout::top_down(Align::Min),
                    |ui| {
                        self.render_chat_area(ui);
                    },
                );

                // Input area (fixed height)
                ui.allocate_ui_with_layout(
                    Vec2::new(available.x, 90.0),
                    Layout::top_down(Align::Min),
                    |ui| {
                        // Suggestions bar if conversation just started
                        if self.messages.len() <= 1 {
                            ui.horizontal_wrapped(|ui| {
                                ui.label(
                                    RichText::new("Quick start:")
                                        .size(12.0)
                                        .color(Color32::GRAY),
                                );
                                let lang = self.target_lang.clone();
                                let suggestions = if self.suggestions.is_empty() {
                                    default_suggestions(&lang)
                                } else {
                                    self.suggestions.clone()
                                };
                                for suggestion in suggestions {
                                    if ui.small_button(&suggestion.label).clicked() {
                                        self.current_input = suggestion.prompt;
                                        self.send_message();
                                    }
                                }
                            });
                            ui.add_space(4.0);
                        }

                        self.render_input_area(ui);
                    },
                );
            });
        });

        // Request repaint while processing or recording
        if self.is_processing || self.is_recording {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
    }
}
