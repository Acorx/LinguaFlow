/// LinguaFlow — AI-powered language learning through voice conversations.
///
/// A cross-platform desktop application built with egui/eframe for the GUI,
/// cpal for audio, reqwest for HTTP, and rusqlite for persistent memory.
///
/// Usage: Set OPENROUTER_API_KEY or OPENAI_API_KEY environment variables,
/// then run with `cargo run --release`.

// Module declarations
mod app;
mod audio;
mod llm;
mod memory;
mod stt;
mod tts;

use app::LinguaFlowApp;
use eframe::egui;
use egui::ViewportBuilder;
use log::info;

fn main() -> eframe::Result<()> {
    // Initialize logger
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_secs()
        .init();

    info!("LinguaFlow starting...");

    // Window configuration
    let viewport = ViewportBuilder::default()
        .with_title("LinguaFlow")
        .with_inner_size([1100.0, 750.0])
        .with_min_inner_size([800.0, 600.0])
        .with_resizable(true);

    // Native options
    let native_options = eframe::NativeOptions {
        viewport,
        centered: true,
        ..Default::default()
    };

    // Run the eframe application
    eframe::run_native(
        "LinguaFlow",
        native_options,
        Box::new(|cc| {
            // Setup dark visual theme
            cc.egui_ctx.set_visuals(egui::Visuals::dark());

            // Configure fonts
            configure_fonts(&cc.egui_ctx);

            // Create the application
            let app = LinguaFlowApp::new();
            info!("LinguaFlow initialized successfully");
            Ok(Box::new(app))
        }),
    )
}

/// Configures custom fonts for the egui application.
fn configure_fonts(_ctx: &egui::Context) {
    // Font configuration can be added here for custom emoji or CJK support.
    // Using default egui fonts for now.
}
