# EchoVault

A local voice assistant that chains Whisper (speech-to-text), Claude (LLM), and Piper (text-to-speech) into a real-time conversational loop. Supports push-to-talk and always-listening (wake word) modes.

## Architecture

```
Microphone → AudioRecorder → Whisper STT → Claude LLM → Piper TTS → Speaker
                                ↑
                        OpenWakeWord (always-listening mode)
```

All speech processing runs locally except the LLM call to the Anthropic API.

## Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org/) (installed automatically by `setup.sh`)
- System dependencies:
  - **macOS**: `portaudio`, `ffmpeg` (via Homebrew)
  - **Linux**: `portaudio19-dev`, `ffmpeg`, `python3-dev` (via apt)
- An [Anthropic API key](https://console.anthropic.com/)

## Quick Start

```bash
# 1. Clone and set up
git clone https://github.com/your-user/echovault.git
cd echovault
bash setup.sh

# 2. Add your API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

# 3. Run
bash run.sh                          # push-to-talk (default)
bash run.sh --mode always-listening  # wake word mode
```

## Modes

### Push-to-Talk (default)

Press Enter to start recording, press Enter again to stop. Your speech is transcribed, sent to Claude, and the response is spoken aloud.

```bash
bash run.sh
# or explicitly:
bash run.sh --mode push-to-talk
```

In-session commands:
- `quit` — exit the assistant
- `reset` — clear conversation history

### Always-Listening

The assistant listens continuously for a wake word ("hey jarvis" by default). Once detected, it records your command using voice activity detection (VAD), then responds.

```bash
bash run.sh --mode always-listening
```

Press `Ctrl+C` to exit.

## Configuration

### `.env`

```
ANTHROPIC_API_KEY=sk-ant-xxxxx
```

### `config.yaml`

Copied from `config.example.yaml` during setup. Key sections:

| Section    | Options                                                         |
|------------|-----------------------------------------------------------------|
| `audio`    | `sample_rate`, `channels`, `chunk_size`, `format`               |
| `whisper`  | `model` (e.g. `base.en`, `small.en`), `device`, `language`     |
| `claude`   | `model`, `max_tokens`, `system_prompt`, `max_history_pairs`     |
| `piper`    | `model_path`, `config_path`                                     |
| `vad`      | `aggressiveness` (0-3), `silence_timeout`, `frame_duration_ms`  |
| `wakeword` | `model_name`, `threshold`, `chunk_size`                         |

### CLI Options

```
usage: python -m src.main [-h] [--mode {push-to-talk,always-listening}] [--config CONFIG]

  --mode     Operating mode (default: push-to-talk)
  --config   Path to config file (default: config.yaml)
```

## Project Structure

```
echovault/
├── src/
│   ├── main.py              # Entry point and mode runners
│   ├── config.py            # YAML + env config loader
│   ├── audio/
│   │   ├── recorder.py      # Mic input (push-to-talk & VAD)
│   │   └── player.py        # Audio playback
│   ├── stt/
│   │   ├── base.py          # STT interface
│   │   └── whisper_stt.py   # OpenAI Whisper
│   ├── llm/
│   │   ├── base.py          # LLM interface
│   │   └── claude_llm.py    # Anthropic Claude
│   ├── tts/
│   │   ├── base.py          # TTS interface
│   │   └── piper_tts.py     # Piper TTS
│   ├── wakeword/
│   │   ├── base.py          # Wake word interface
│   │   └── oww_wakeword.py  # OpenWakeWord
│   └── utils/
│       └── logger.py        # Logging setup
├── models/                  # Piper voice models (downloaded by setup.sh)
├── config.example.yaml      # Default configuration
├── .env.example             # API key template
├── pyproject.toml           # Poetry dependencies
├── setup.sh                 # One-step setup script
└── run.sh                   # Run the assistant
```
