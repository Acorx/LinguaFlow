/// LLM abstraction supporting OpenRouter and Ollama backends.
///
/// - OpenRouter: cloud API with access to Claude, GPT, DeepSeek
/// - Ollama: local models via localhost:11434
///
/// Auto-detects provider based on environment (OPENROUTER_API_KEY -> OpenRouter,
/// localhost Ollama -> local fallback, otherwise basic mode).

use log::{debug, error, info, warn};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;
use std::sync::Arc;

/// LLM provider selection.
#[derive(Debug, Clone)]
pub enum LLMProvider {
    OpenRouter { api_key: String, model: String },
    Ollama { host: String, model: String },
}

/// A chat message with role and content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// Response wrapper for OpenRouter API.
#[derive(Debug, Deserialize)]
struct OpenRouterResponse {
    choices: Vec<OpenRouterChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterChoice {
    message: OpenRouterMessage,
}

#[derive(Debug, Deserialize)]
struct OpenRouterMessage {
    content: String,
}

/// Response wrapper for Ollama API.
#[derive(Debug, Deserialize)]
struct OllamaResponse {
    message: OllamaMessage,
}

#[derive(Debug, Deserialize)]
struct OllamaMessage {
    content: String,
}

/// Ollama chat request body.
#[derive(Debug, Serialize)]
struct OllamaRequest {
    model: String,
    messages: Vec<OllamaMessageInner>,
    stream: bool,
}

#[derive(Debug, Serialize)]
struct OllamaMessageInner {
    role: String,
    content: String,
}

/// Client for communicating with LLM providers.
#[derive(Clone)]
pub struct LLMClient {
    pub provider: LLMProvider,
    pub system_prompt: String,
    client: Arc<Client>,
}

impl Default for LLMClient {
    fn default() -> Self {
        Self::new()
    }
}

impl LLMClient {
    /// Creates a new LLMClient with auto-detected provider.
    pub fn new() -> Self {
        let system_prompt = Self::default_system_prompt("beginner", "the target language");

        let provider = if let Ok(api_key) = env::var("OPENROUTER_API_KEY") {
            let model = env::var("LINGUAFLOW_MODEL")
                .unwrap_or_else(|_| "anthropic/claude-sonnet-4-20250514".to_string());
            info!("LLM: Using OpenRouter with model: {}", model);
            LLMProvider::OpenRouter { api_key, model }
        } else {
            let ollama_available = check_ollama_sync();

            if ollama_available {
                let model = env::var("LINGUAFLOW_MODEL")
                    .unwrap_or_else(|_| "llama3.2:3b".to_string());
                info!("LLM: Using Ollama with model: {}", model);
                LLMProvider::Ollama {
                    host: "http://localhost:11434".to_string(),
                    model,
                }
            } else {
                warn!(
                    "LLM: No provider auto-detected. Set OPENROUTER_API_KEY or start Ollama."
                );
                LLMProvider::Ollama {
                    host: "http://localhost:11434".to_string(),
                    model: "qwen2.5:3b".to_string(),
                }
            }
        };

        let client = Arc::new(
            Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .build()
                .expect("Failed to create HTTP client"),
        );

        Self {
            provider,
            system_prompt,
            client,
        }
    }

    /// Creates a new LLMClient with a specific provider.
    pub fn with_provider(provider: LLMProvider) -> Self {
        let system_prompt = Self::default_system_prompt("beginner", "the target language");
        Self {
            provider,
            system_prompt,
            client: Arc::new(
                Client::builder()
                    .timeout(std::time::Duration::from_secs(120))
                    .build()
                    .expect("Failed to create HTTP client"),
            ),
        }
    }

    /// Creates a client with a custom system prompt.
    pub fn with_system_prompt(provider: LLMProvider, prompt: String) -> Self {
        Self {
            provider,
            system_prompt: prompt,
            client: Arc::new(
                Client::builder()
                    .timeout(std::time::Duration::from_secs(120))
                    .build()
                    .expect("Failed to create HTTP client"),
            ),
        }
    }

    /// Updates the system prompt for a specific user level and target language.
    pub fn set_system_prompt(&mut self, user_level: &str, target_lang: &str) {
        self.system_prompt = Self::default_system_prompt(user_level, target_lang);
        debug!(
            "System prompt updated for level: {}, language: {}",
            user_level, target_lang
        );
    }

    /// Generates the default system prompt for language learning.
    fn default_system_prompt(user_level: &str, target_lang: &str) -> String {
        format!(
            r#"You are LinguaFlow, a friendly and expert language tutor AI.

Your mission: help the user practice and improve their {target_lang} skills through natural conversation.

Rules:
1. ALWAYS respond primarily in {target_lang}, unless the user explicitly asks for help in another language.
2. CORRECT errors naturally: if the user makes a grammatical or vocabulary mistake, gently rephrase it correctly.
   Format corrections as: [Correction: "original" → "corrected" — brief explanation]
3. VARY topics: discuss culture, daily life, science, travel, hobbies, food, history — keep conversations engaging.
4. ADAPT to the user's level ({user_level}):
   - Beginner (A1-A2): short sentences, simple vocabulary, frequent encouragement, translations
   - Intermediate (B1-B2): complex grammar, idiomatic expressions, natural flow
   - Advanced (C1-C2): nuanced discussions, subtle grammar points, sophisticated vocabulary
5. TEACH vocabulary in context and provide translations when helpful.
6. HIGHLIGHT corrections with brief explanations, then continue the conversation.
7. ENCOURAGE: always be positive, motivating, and celebrate progress.
8. If asked about a word/phrase, explain clearly with examples in context.
9. Periodically suggest new topics based on conversation flow and interests.

Current level: {user_level}
Target language: {target_lang}
Be conversational, warm, patient, and effective."#,
        )
    }

    /// Sends a chat request and returns the assistant's response.
    pub async fn chat(&self, messages: Vec<Message>) -> String {
        match &self.provider {
            LLMProvider::OpenRouter { api_key, model } => {
                self.chat_openrouter(api_key, model, messages).await
            }
            LLMProvider::Ollama { host, model } => {
                self.chat_ollama(host, model, messages).await
            }
        }
    }

    /// Sends a chat request via OpenRouter API.
    async fn chat_openrouter(
        &self,
        api_key: &str,
        model: &str,
        messages: Vec<Message>,
    ) -> String {
        let url = "https://openrouter.ai/api/v1/chat/completions";

        let mut all_messages = Vec::new();
        all_messages.push(Message {
            role: "system".to_string(),
            content: self.system_prompt.clone(),
        });
        all_messages.extend(messages);

        let body = serde_json::json!({
            "model": model,
            "messages": all_messages
                .iter()
                .map(|m| serde_json::json!({
                    "role": m.role,
                    "content": m.content
                }))
                .collect::<Vec<_>>(),
            "max_tokens": 2048,
            "temperature": 0.7,
        });

        debug!("OpenRouter: POST {} with model {}", url, model);

        match self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("HTTP-Referer", "https://github.com/LinguaFlow")
            .header("X-Title", "LinguaFlow")
            .json(&body)
            .send()
            .await
        {
            Ok(response) => {
                let status = response.status();
                if status.is_success() {
                    match response.json::<OpenRouterResponse>().await {
                        Ok(resp) => {
                            if let Some(choice) = resp.choices.into_iter().next() {
                                let content = choice.message.content;
                                debug!("OpenRouter response: {} chars", content.len());
                                return content;
                            } else {
                                error!("OpenRouter: no choices in response");
                            }
                        }
                        Err(e) => {
                            error!("Failed to parse OpenRouter response: {}", e);
                        }
                    }
                    "Sorry, I couldn't process the response. Please try again.".to_string()
                } else {
                    let text = response.text().await.unwrap_or_default();
                    error!(
                        "OpenRouter error {}: {}",
                        status,
                        text.chars().take(200).collect::<String>()
                    );
                    format!("OpenRouter error ({}): Check your API key and model.", status)
                }
            }
            Err(e) => {
                error!("OpenRouter request failed: {}", e);
                format!("Failed to connect to OpenRouter: {}", e)
            }
        }
    }

    /// Sends a chat request via Ollama API.
async fn chat_ollama(&self, host: &str, model: &str, messages: Vec<Message>) -> String {
        let url = format!("{}/api/chat", host);

        let mut all_messages = Vec::new();
        all_messages.push(OllamaMessageInner {
            role: "system".to_string(),
            content: self.system_prompt.clone(),
        });
        for m in messages {
            all_messages.push(OllamaMessageInner {
                role: m.role,
                content: m.content,
            });
        }

        let body = OllamaRequest {
            model: model.to_string(),
            messages: all_messages,
            stream: false,
        };

        debug!("Ollama: POST {} with model {}", url, model);

        match self.client.post(&url).json(&body).send().await {
            Ok(response) => {
                let status = response.status();
                if status.is_success() {
                    match response.json::<OllamaResponse>().await {
                        Ok(resp) => {
                            let content = resp.message.content;
                            debug!("Ollama response: {} chars", content.len());
                            return content;
                        }
                        Err(e) => {
                            error!("Failed to parse Ollama response: {}", e);
                        }
                    }
                    "Sorry, I couldn't process the response. Please try again.".to_string()
                } else {
                    let text = response.text().await.unwrap_or_default();
                    error!("Ollama error {}: {}", status, text.chars().take(200).collect::<String>());
                    format!("Ollama error ({}): Is the model '{}' available?", status, model)
                }
            }
            Err(e) => {
                error!("Ollama request failed: {}", e);
                format!("Failed to connect to Ollama at {}: {}", host, e)
            }
        }
    }

    /// Returns a list of supported models for the current provider.
    pub fn supported_models(&self) -> Vec<&str> {
        match &self.provider {
            LLMProvider::OpenRouter { .. } => {
                vec![
                    "anthropic/claude-sonnet-4-20250514",
                    "openai/gpt-4o-mini",
                    "deepseek/deepseek-chat",
                ]
            }
            LLMProvider::Ollama { .. } => {
                vec!["llama3.2:3b", "qwen2.5:3b"]
            }
        }
    }

    /// Returns the current model name.
    pub fn current_model(&self) -> &str {
        match &self.provider {
            LLMProvider::OpenRouter { model, .. } => model,
            LLMProvider::Ollama { model, .. } => model,
        }
    }

    /// Returns the provider name as a string for display.
    pub fn provider_name(&self) -> &str {
        match &self.provider {
            LLMProvider::OpenRouter { .. } => "OpenRouter",
            LLMProvider::Ollama { .. } => "Ollama",
        }
    }

    /// Switches to a different model within the current provider.
    pub fn switch_model(&mut self, model: &str) {
        match &mut self.provider {
            LLMProvider::OpenRouter { model: ref mut m, .. } => {
                *m = model.to_string();
                info!("Switched to OpenRouter model: {}", model);
            }
            LLMProvider::Ollama { model: ref mut m, .. } => {
                *m = model.to_string();
                info!("Switched to Ollama model: {}", model);
            }
        }
    }

    /// Switches to OpenRouter provider with the given API key.
    pub fn switch_to_openrouter(&mut self, api_key: String) {
        self.provider = LLMProvider::OpenRouter {
            api_key,
            model: "anthropic/claude-sonnet-4-20250514".to_string(),
        };
        info!("Switched to OpenRouter provider");
    }

    /// Switches to Ollama provider.
    pub fn switch_to_ollama(&mut self, host: String) {
        let model = if self.current_model().contains('/') {
            "llama3.2:3b".to_string()
        } else {
            self.current_model().to_string()
        };
        self.provider = LLMProvider::Ollama { host, model };
        info!("Switched to Ollama provider");
    }
}

/// Synchronous check for Ollama availability.
fn check_ollama_sync() -> bool {
    let rt = match tokio::runtime::Runtime::new() {
        Ok(r) => r,
        Err(_) => return false,
    };
    rt.block_on(async {
        let client = match Client::builder()
            .timeout(std::time::Duration::from_secs(2))
            .build()
        {
            Ok(c) => c,
            Err(_) => return false,
        };
        match client.get("http://localhost:11434/api/tags").send().await {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
    })
}
