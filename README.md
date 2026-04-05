# LinguaFlow

```
 __     __           _       _                  _
 \ \   / /__  _ __  | |   __| | __ _  ___ _ __ (_)___
  \ \ / / _ \| '__| | |  / _` |/ _` |/ _ \ '_ \| / __|
   \ V / (_) | |    | | | (_| | (_| |  __/ | | | \__ \
    \_/ \___/|_|    |_|  \__,_|\__, |\___|_| |_|_|___/
                               |___/
```

**Apprentissage des langues par conversation IA — en natif, avec Rust.**

LinguaFlow est une application desktop native écrite en Rust qui permet d'apprendre les langues étrangères grâce à la conversation vocale assistée par intelligence artificielle. Communiquez naturellement en parlant, recevez des corrections en temps réel et progressez grâce à une mémoire adaptative qui suit vos erreurs et vos progrès.

---

## Fonctionnalités

- **Conversation vocale native** — Parlez et écutez votre IA en temps réel grâce à `cpal` et `rodio`, sans intermédiaire web
- **Corrections IA en temps réel** — Transcription automatique de votre voix via Whisper, analyse grammaticale et prononciation par l'IA, feedback instantané
- **Mémoire adaptative** — Base de données SQLite (`rusqlite`) qui suit vos erreurs récurrentes, vos points forts et vos progrès pour personnaliser les sessions
- **Multi-providers (API + local)** — Choisissez entre les API cloud (OpenRouter, OpenAI) pour une précision maximale ou des modèles locaux (Ollama) pour fonctionner hors-ligne
- **Glassmorphism natif** — Interface élégante avec transparences et effets visuels via `eframe`/`egui`
- **Binaire léger** — Profil release optimisé avec LTO et strip pour un exécutable compact

---

## Compilation

Assurez-vous d'avoir Rust installé (rustup recommandé), puis :

```bash
cargo build --release
```

Une fois compilé, le binaire se trouve dans `target/release/linguaflow`.

## Exécution

```bash
cargo run --release
```

---

## Configuration API

Créez un fichier `.env` à la racine du projet (ou dans le répertoire de lancement) avec vos clés API :

```env
OPENROUTER_API_KEY=votre_cle_openrouter
OPENAI_API_KEY=votre_cle_openai
```

Seules les clés dont vous avez besoin pour votre mode d'utilisation sont nécessaires. LinguaFlow détecte automatiquement les clés disponibles.

---

## Mode local (hors-ligne)

Pour utiliser LinguaFlow sans dépendre d'API cloud, installez les outils suivants :

### 1. Ollama (modèle de langage)

```bash
# Installer Ollama (https://ollama.com)
ollama pull llama3.2:3b
```

Le modèle `llama3.2:3b` est un bon compromis entre performance et ressources. Pour plus de précision, envisagez `llama3.1:8b` ou `mistral:7b`.

### 2. Piper TTS (synthèse vocale)

```bash
# Télécharger depuis https://github.com/rhasspy/piper
piper --model fr_FR-mls-medium.onnx
```

Installez un modèle vocal pour la langue cible (français, anglais, espagnol, etc.). Piper offre une synthèse locale rapide et de qualité.

### 3. whisper.cpp (transcription)

```bash
# Installer depuis https://github.com/ggerganov/whisper.cpp
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp && make
# Télécharger un modèle (medium ou large recommandé)
./models/download-ggml-model.sh medium
```

whisper.cpp transcrit votre voix en texte localement, sans envoyer vos données à un serveur.

---

## Comparaison API vs Local

| Critère              | Mode API (OpenRouter/OpenAI) | Mode Local (Ollama) |
|----------------------|------------------------------|---------------------|
| **Précision**        | Excellente                   | Bonne à très bonne  |
| **Rapidité**         | Très rapide                  | Dépend du matériel  |
| **Privacy**          | Données envoyées au cloud    | 100% local          |
| **Hors-ligne**       | Non                          | Oui                 |
| **Installation**     | Clé API uniquement           | Ollama + Piper + whisper.cpp |
| **Coût**             | Payant à l'usage             | Gratuit             |
| **Recommandé pour**  | Précision max, débutants     | Privacy, usage offline |

---

## Modèles recommandés par cas d'usage

| Cas d'usage                        | Modèle recommandé          | Provider  |
|------------------------------------|----------------------------|-----------|
| Conversation courante              | `openai/gpt-4o-mini`       | API       |
| Corrections grammaticales          | `anthropic/claude-3.5-haiku` | API     |
| Usage quotidien léger              | `meta-llama/llama-3.2-3b`  | Ollama    |
| Conversations avancées (hors-ligne) | `meta-llama/llama-3.1-8b`  | Ollama    |
| Synthèse vocale (local)            | Piper `fr_FR-mls-medium`   | Local     |
| Transcription vocale (local)       | Whisper `medium` (769M)    | whisper.cpp |
| Équilibre coût/performance         | `google/gemini-2.0-flash`  | API       |

---

## Structure du code

```
LinguaFlow/
├── Cargo.toml              # Dépendances et configuration du projet
├── README.md               # Cette documentation
├── .env                    # Clés API (non versionné)
└── src/
    ├── main.rs             # Point d'entrée, initialisation eframe
    ├── ui/
    │   ├── mod.rs          # Module UI principal
    │   └── app.rs          # Application egui principale (glassmorphism, layout)
    ├── conversation/
    │   ├── mod.rs          # Module de conversation
    │   ├── engine.rs       # Moteur de gestion des conversations
    │   └── session.rs      # Gestion des sessions
    ├── ai/
    │   ├── mod.rs          # Module IA
    │   ├── provider.rs     # Trait et sélection de provider
    │   ├── openrouter.rs   # Client OpenRouter
    │   ├── openai.rs       # Client OpenAI
    │   └── ollama.rs       # Client Ollama (local)
    ├── voice/
    │   ├── mod.rs          # Module voix
    │   ├── mic.rs          # Capture microphone (cpal)
    │   ├── tts.rs          # Synthèse vocale (rodio + Piper)
    │   └── stt.rs          # Reconnaissance vocale (whisper.cpp)
    ├── memory/
    │   ├── mod.rs          # Module mémoire
    │   └── db.rs           # Base SQLite (rusqlite) — erreurs, progrès, historique
    ├── config/
    │   ├── mod.rs          # Module configuration
    │   └── settings.rs     # Chargement des paramètres (.env, fichiers)
    └── utils/
        ├── mod.rs          # Module utilitaires
        └── helpers.rs      # Fonctions utilitaires diverses
```

---

## Roadmap

- [x] Architecture de base avec eframe/egui
- [x] Intégration API OpenRouter et OpenAI
- [x] Support Ollama pour le mode local
- [x] Système de mémoire adaptative avec SQLite
- [ ] Correction grammaticale avancée avec suggestions contextuelles
- [ ] Support multi-langues dans l'interface
- [ ] Statistiques et graphiques de progression
- [ ] Sauvegarde et import de sessions
- [ ] Mode examen et quiz générés par IA
- [ ] Support pour plus de providers (Claude, Gemini local, Groq)
- [ ] Application mobile (egui sur Android)

---

## Licence

Ce projet est distribué sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

*Fait avec ❤️ en Rust — Rapide, sûr, natif.*
