/// SQLite memory module — persists user profile, error tracking,
/// vocabulary learning progress, and conversation history.
///
/// Database location: ~/.linguaflow/memory.db

use log::{error, info};
use rusqlite::{params, Connection, Result as RusqliteResult};
use std::path::PathBuf;

/// Represents the user's learning profile stored in memory.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct UserProfile {
    pub name: String,
    pub target_lang: String,
    pub level: String,
    pub interests: String,
}

/// Tracks errors the user makes so the system can focus on weak areas.
#[derive(Debug, Clone)]
pub struct ErrorRecord {
    pub error: String,
    pub correction: String,
    pub context: String,
    pub lang: String,
    pub count: i64,
    pub last_seen: String,
}

/// A vocabulary word the user has studied.
#[derive(Debug, Clone)]
pub struct VocabularyEntry {
    pub word: String,
    pub translation: String,
    pub context: String,
    pub lang: String,
    pub known: bool,
    pub learned_at: String,
}

/// A single conversation message persisted to history.
#[derive(Debug, Clone)]
pub struct ConversationEntry {
    pub id: i64,
    pub role: String,
    pub content: String,
    pub lang: String,
    pub created_at: String,
}

/// Core memory handler wrapping an SQLite connection.
#[derive(Clone)]
pub struct UserMemory {
    pub db_path: String,
}

impl UserMemory {
    /// Creates a new UserMemory instance and initializes the database schema.
    ///
    /// The database is stored at `~/.linguaflow/memory.db` by default.
    pub fn new() -> Self {
        let base_dir = Self::linguaflow_dir();
        std::fs::create_dir_all(&base_dir).unwrap_or_else(|e| {
            log::error!("Failed to create data directory {:?}: {}", base_dir, e);
        });
        let db_path = base_dir.join("memory.db");
        let db_path_str = db_path.to_string_lossy().to_string();

        let instance = Self {
            db_path: db_path_str.clone(),
        };
        instance.init_tables();
        info!("UserMemory initialized at {}", db_path_str);
        instance
    }

    /// Returns the path to the ~/.linguaflow/ data directory.
    fn linguaflow_dir() -> PathBuf {
        dirs::data_dir()
            .map(|d| d.join("LinguaFlow"))
            .unwrap_or_else(|| {
                let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
                let mut p = PathBuf::from(home);
                p.push(".linguaflow");
                p
            })
    }

    /// Opens a connection to the SQLite database.
    fn open_conn(&self) -> RusqliteResult<Connection> {
        Connection::open(&self.db_path)
    }

    /// Creates the database schema if tables don't exist yet.
    fn init_tables(&self) {
        let conn = match self.open_conn() {
            Ok(c) => c,
            Err(e) => {
                error!("Failed to open database {}: {}", self.db_path, e);
                return;
            }
        };

        let queries = [
            // User profile
            "CREATE TABLE IF NOT EXISTS user_profile (
                name TEXT,
                target_lang TEXT,
                level TEXT,
                interests TEXT
            )",
            // Error tracking
            "CREATE TABLE IF NOT EXISTS errors (
                error TEXT,
                correction TEXT,
                context TEXT,
                lang TEXT,
                count INTEGER DEFAULT 1,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )",
            // Vocabulary
            "CREATE TABLE IF NOT EXISTS vocabulary (
                word TEXT,
                translation TEXT,
                context TEXT,
                lang TEXT,
                known INTEGER DEFAULT 0,
                learned_at TIMESTAMP
            )",
            // Conversation history
            "CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT,
                content TEXT,
                lang TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )",
        ];

        for q in &queries {
            if let Err(e) = conn.execute(q, []) {
                error!("Failed to create table: {}", e);
            }
        }

        info!("Database tables initialized successfully");
    }

    /// Saves or updates the user's learning profile.
    pub fn update_profile(&self, name: &str, target_lang: &str, level: &str, interests: &str) {
        let conn = match self.open_conn() {
            Ok(c) => c,
            Err(e) => {
                error!("Cannot update profile: {}", e);
                return;
            }
        };
        // Delete existing profile, then insert new one
        let _ = conn.execute("DELETE FROM user_profile", []);
        if let Err(e) = conn.execute(
            "INSERT INTO user_profile (name, target_lang, level, interests) VALUES (?1, ?2, ?3, ?4)",
            params![name, target_lang, level, interests],
        ) {
            error!("Failed to insert profile: {}", e);
        }
        info!("Profile updated: {} ({} - {}) - {}", name, target_lang, level, interests);
    }

    /// Loads the current user profile, if any.
    pub fn load_profile(&self) -> Option<UserProfile> {
        let conn = self.open_conn().ok()?;
        conn.query_row(
            "SELECT name, target_lang, level, interests FROM user_profile LIMIT 1",
            [],
            |row| {
                Ok(UserProfile {
                    name: row.get(0).unwrap_or_default(),
                    target_lang: row.get(1).unwrap_or_default(),
                    level: row.get(2).unwrap_or_default(),
                    interests: row.get(3).unwrap_or_default(),
                })
            },
        )
        .ok()
    }

    /// Records an error the user made, incrementing the count if it already exists.
    pub fn add_error(&self, error_text: &str, correction: &str, context: &str, lang: &str) {
        let conn = match self.open_conn() {
            Ok(c) => c,
            Err(e) => {
                error!("Cannot add error: {}", e);
                return;
            }
        };

        // Try to increment if error already known
        let updated = conn.execute(
            "UPDATE errors SET count = count + 1, last_seen = CURRENT_TIMESTAMP WHERE error = ?1",
            params![error_text],
        );

        if let Ok(0) = updated {
            // New error, insert it
            if let Err(e) = conn.execute(
                "INSERT INTO errors (error, correction, context, lang) VALUES (?1, ?2, ?3, ?4)",
                params![error_text, correction, context, lang],
            ) {
                error!("Failed to insert new error: {}", e);
            }
        }

        info!("Error tracked: '{}' → '{}'", error_text, correction);
    }

    /// Returns the most frequent errors in descending order of frequency.
    pub fn get_frequent_errors(&self, limit: usize) -> Vec<ErrorRecord> {
        let conn = match self.open_conn() {
            Ok(c) => c,
            Err(_) => return vec![],
        };

        let mut stmt = match conn.prepare(
            "SELECT error, correction, context, lang, count, last_seen
             FROM errors ORDER BY count DESC LIMIT ?1"
        ) {
            Ok(s) => s,
            Err(e) => {
                error!("Failed to prepare frequent errors query: {}", e);
                return vec![];
            }
        };

        let rows = match stmt.query_map(params![limit as i64], |row| {
            Ok(ErrorRecord {
                error: row.get(0).unwrap_or_default(),
                correction: row.get(1).unwrap_or_default(),
                context: row.get(2).unwrap_or_default(),
                lang: row.get(3).unwrap_or_default(),
                count: row.get(4).unwrap_or(0),
                last_seen: row.get(5).unwrap_or_default(),
            })
        }) {
            Ok(r) => r,
            Err(e) => {
                error!("Failed to query frequent errors: {}", e);
                return vec![];
            }
        };

        rows.filter_map(Result::ok).collect()
    }

    /// Adds or updates a vocabulary word.
    pub fn add_vocabulary(&self, word: &str, translation: &str, context: &str, lang: &str) {
        let conn = match self.open_conn() {
            Ok(c) => c,
            Err(e) => {
                error!("Cannot add vocabulary: {}", e);
                return;
            }
        };

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs().to_string())
            .unwrap_or_else(|_| "0".to_string());

        // Update or insert
        let updated = conn.execute(
            "UPDATE vocabulary SET translation = ?2, context = ?3, known = known, learned_at = ?4 WHERE word = ?1",
            params![word, translation, context, now],
        );

        if let Ok(0) = updated {
            if let Err(e) = conn.execute(
                "INSERT INTO vocabulary (word, translation, context, lang, known, learned_at)
                 VALUES (?1, ?2, ?3, ?4, 0, ?5)",
                params![word, translation, context, lang, now],
            ) {
                error!("Failed to insert vocabulary: {}", e);
            }
        }

        info!("Vocabulary added: {} → {} ({})", word, translation, lang);
    }

    /// Retrieves vocabulary entries. When `known_only` is true, only returns mastered words.
    pub fn get_vocabulary(&self, lang: &str, known_only: bool) -> Vec<VocabularyEntry> {
        let conn = match self.open_conn() {
            Ok(c) => c,
            Err(_) => return vec![],
        };

        let query = if known_only {
            "SELECT word, translation, context, lang, known, learned_at FROM vocabulary
             WHERE lang = ?1 AND known = 1 ORDER BY learned_at DESC"
        } else {
            "SELECT word, translation, context, lang, known, learned_at FROM vocabulary
             WHERE lang = ?1 ORDER BY learned_at DESC"
        };

        let mut stmt = match conn.prepare(query) {
            Ok(s) => s,
            Err(e) => {
                error!("Failed to prepare vocabulary query: {}", e);
                return vec![];
            }
        };

        let rows = match stmt.query_map(params![lang], |row| {
            Ok(VocabularyEntry {
                word: row.get(0).unwrap_or_default(),
                translation: row.get(1).unwrap_or_default(),
                context: row.get(2).unwrap_or_default(),
                lang: row.get(3).unwrap_or_default(),
                known: row.get::<_, i64>(4).unwrap_or(0) != 0,
                learned_at: row.get(5).unwrap_or_default(),
            })
        }) {
            Ok(r) => r,
            Err(e) => {
                error!("Failed to query vocabulary: {}", e);
                return vec![];
            }
        };

        rows.filter_map(Result::ok).collect()
    }

    /// Marks a vocabulary entry as known (mastered).
    pub fn mark_known(&self, word: &str) {
        let conn = match self.open_conn() {
            Ok(c) => c,
            Err(e) => {
                error!("Cannot mark known: {}", e);
                return;
            }
        };
        let _ = conn.execute(
            "UPDATE vocabulary SET known = 1 WHERE word = ?1",
            params![word],
        );
    }

    /// Saves a conversation message to history.
    pub fn save_conversation(&self, role: &str, content: &str, lang: &str) {
        let conn = match self.open_conn() {
            Ok(c) => c,
            Err(e) => {
                error!("Cannot save conversation: {}", e);
                return;
            }
        };

        if let Err(e) = conn.execute(
            "INSERT INTO conversations (role, content, lang) VALUES (?1, ?2, ?3)",
            params![role, content, lang],
        ) {
            error!("Failed to save conversation: {}", e);
        }
    }

    /// Returns the most recent conversation messages as a formatted string
    /// suitable for injecting into the LLM system prompt as context.
    pub fn get_recent_context(&self, limit: usize) -> String {
        let conn = match self.open_conn() {
            Ok(c) => c,
            Err(_) => return String::new(),
        };

        let mut stmt = match conn.prepare(
            "SELECT role, content, lang FROM conversations
             ORDER BY id DESC LIMIT ?1"
        ) {
            Ok(s) => s,
            Err(e) => {
                error!("Failed to prepare conversation query: {}", e);
                return String::new();
            }
        };

        let rows = match stmt.query_map(params![limit as i64], |row| {
            Ok((
                row.get::<_, String>(0).unwrap_or_default(),
                row.get::<_, String>(1).unwrap_or_default(),
                row.get::<_, String>(2).unwrap_or_default(),
            ))
        }) {
            Ok(r) => r,
            Err(e) => {
                error!("Failed to query recent context: {}", e);
                return String::new();
            }
        };

        let mut entries: Vec<(String, String, String)> = rows.filter_map(Result::ok).collect();
        // Reverse to get chronological order
        entries.reverse();

        let context: String = entries
            .iter()
            .map(|(role, content, _lang)| format!("[{}] {}", role, content))
            .collect::<Vec<_>>()
            .join("\n");

        if context.is_empty() {
            String::from("No previous conversation context.")
        } else {
            format!("Recent conversation history:\n{}", context)
        }
    }

    /// Clears all conversation history.
    pub fn clear_conversations(&self) {
        let conn = match self.open_conn() {
            Ok(c) => c,
            Err(_) => return,
        };
        let _ = conn.execute("DELETE FROM conversations", []);
        info!("Conversation history cleared");
    }

    /// Returns the number of vocabulary words known for a language.
    pub fn vocabulary_count(&self, lang: &str) -> usize {
        let conn = match self.open_conn() {
            Ok(c) => c,
            Err(_) => return 0,
        };
        conn.query_row(
            "SELECT COUNT(*) FROM vocabulary WHERE lang = ?1",
            params![lang],
            |row| row.get(0),
        )
        .unwrap_or(0)
    }

    /// Returns the total number of errors tracked.
    pub fn error_count(&self) -> usize {
        let conn = match self.open_conn() {
            Ok(c) => c,
            Err(_) => return 0,
        };
        conn.query_row("SELECT COUNT(*) FROM errors", [], |row| row.get(0))
            .unwrap_or(0)
    }
}
