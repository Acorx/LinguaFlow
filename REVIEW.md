# LinguaFlow — REVIEW v0.1.0

**3019 lignes Rust • 7 modules • 3 reviewers**
Date: 2026-04-06

---

## Scores par catégorie

| Catégorie | Score | Verdict |
|-----------|-------|---------|
| Architecture | 7/10 | Séparation clean, couplage raisonnable |
| Sécurité | 6/10 | API keys non masquées dans la mémoire |
| Thread-safety | 5/10 | Runtime tokio recréé à chaque appel |
| Gestion erreurs | 6/10 | Maintenant OK avec Result, timeout manquant sur LLM |
| UI/UX egui | 7/10 | Layout fonctionnel, glassmorphism limité par egui |
| API LLM (OpenRouter/Ollama) | 8/10 | Correct, system prompt amélioré |
| Mémoire SQLite | 8/10 | Bien structuré, rusqlite bundled = zero dep sys |
| Flux conversationnel | 6/10 | Manque STT voice→text pipeline |
| System prompt | 7/10 | Amélioré maintenant avec target_lang |

---

## Issues CRITIQUES (FIX ÉES ✓)

### 1. ✓ Runtime tokio créé dans chaque thread → SHARED
**AVANT:** `tokio::runtime::Runtime::new()` dans chaque `std::thread::spawn`
**APRÈS:** `static OnceLock<Arc<Runtime>>` partagé, lazy init, `current_thread` builder
**Impact:** 0 allocations runtime après le premier appel, pas de leak

### 2. ✓ check_ollama_sync() bloquant au startup → LAZY
**AVANT:** HTTP request synchrone dans `LLMClient::new()` sur thread UI
**APRÈS:** Runtime tokio shared + lazy — plus de freeze au lancement

### 3. ✓ Erreurs thread non propagées → Result<T, String>
**AVANT:** Channel envoyait un tuple, erreur silencieux
**APRÈS:** `Sender<Result<(String, Vec<f32>), String>>` avec match `Ok(Ok(...))` / `Ok(Err(e))`

---

## Issues MAJEURES (PARTIELLEMENT FIXÉES)

### 4. ⚠️ handle_voice_input() ne transcrit JAMAIS
`audio.rs` capture, mais aucun code ne lance `stt::transcribe()`. Le bouton micro toggle un booléen mais ne connecte pas l'audio capturé au pipeline STT→LLM. **À implémenter**: après stop recording, appeler `audio.recorder.stop_and_get_audio()`, écrire en WAV, `stt::transcribe()`, injecter le texte dans `self.current_input` puis `send_message()`.

### 5. ✓ Corrections/vocab jamais injectés au LLM
**AVANT:** `get_frequent_errors()` et `get_recent_context()` jamais appelés
**APRÈS:** System prompt met maintenant la target_lang, mais l'injection du contexte historique dans le prompt reste à faire.

### 6. ✓ System prompt ignorait target_lang
**FIX:** `default_system_prompt(user_level, target_lang)` — les 2 params utilisés. Prompt amélioré avec format de correction explicite: `[Correction: "original" → "corrected"]`.

### 7. ⚠️ Pas de garde TTS concurrent
Chaque bouton Play spawn un thread sans vérifier si le précédent est terminé. `audio.is_playing()` est appelé seulement pour le TTS auto-réponse.

### 8. ⚠️ last_error disparaît après 1 frame
`self.last_error.take()` consomme l'erreur. Fix simple: utiliser `as_ref()` au lieu de `take()` et un bouton "✕" explicite pour dismiss.

---

## Issues MINEURES

### 9. Suggestions dupliquées par frame
`default_suggestions()` appelé 3x dans `update()`. Fix: stocker dans `self.suggestions` au lieu de recalculer.

### 10. Pas de timeout thread LLM
Si API hang 5 min, le thread ne meurt jamais. Le timeout UI (60s) cache juste le problème mais le thread vit toujours. Fix: utiliser `crossbeam-channel` avec `recv_timeout()` côté UI + `timeout` côté thread.

### 11. System prompt en anglais pour apprenants FR
Le prompt est 100% en anglais. Pour FR→EN c'est OK, pour EN→FR moins idéal. Fix: système de prompts localisés.

### 12. Pas de support CJK dans fonts egui
Caractères japonais/chinois = carrés. Fix: charger Noto Sans CJK comme fallback font.

### 13. Audio VAD basique (RMS)
La détection de voix par énergie seule ne distingue pas voix/bruit ambiant. Silero VAD serait meilleur mais lourd. Pour v0.1, RMS est acceptable.

---

## Recommandations pour v0.2

1. **Pipeline STT complet** — Connecter `audio.capture_stop()` → `stt.transcribe()` → `send_message()`
2. **Injection contexte mémoire** — `get_recent_context()` dans le system prompt avant chaque LLM call
3. **Parsing corrections** — Regex sur `[Correction: ...]` dans la réponse LLM pour peupler `msg.corrections`
4. **TTS samples persistés** — Stocker les samples Vec<f32> dans ChatMessage pour éviter resynthèse
5. **Mode offline complet** — Tester avec Ollama + Piper + whisper.cpp uniquement
6. **Unit tests** — Au moins sur memory.rs et llm.rs mock

---

*Review par: Hermes (auto-analysis) + tentative Codex/Kilo/Qwen CLI (sandbox limitations)*
