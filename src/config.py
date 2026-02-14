from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    format: str = "int16"


@dataclass
class WhisperConfig:
    backend: str = "local"  # "local" or "api" (OpenAI Whisper API)
    model: str = "turbo"
    device: str = "auto"
    language: str = "en"
    api_key: str = ""


@dataclass
class ClaudeConfig:
    model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 1024
    system_prompt: str = (
        "You are a helpful voice assistant called Jarvis. Keep your responses "
        "concise and conversational — ideally 1-3 sentences unless the user "
        "asks for detail. You are speaking out loud, so avoid markdown, "
        "bullet points, or formatting that doesn't translate to speech."
    )
    max_history_pairs: int = 10
    api_key: str = ""


@dataclass
class PiperConfig:
    model_path: str = "models/en_US-lessac-medium.onnx"
    config_path: str = "models/en_US-lessac-medium.onnx.json"


@dataclass
class VADConfig:
    aggressiveness: int = 2
    silence_timeout: float = 1.0
    frame_duration_ms: int = 30


@dataclass
class OpenWakeWordConfig:
    model_name: str = "hey_jarvis"
    threshold: float = 0.5
    chunk_size: int = 1280


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    claude: ClaudeConfig = field(default_factory=ClaudeConfig)
    piper: PiperConfig = field(default_factory=PiperConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    wakeword: OpenWakeWordConfig = field(default_factory=OpenWakeWordConfig)

    @classmethod
    def load(cls, path: str = "config.yaml") -> Config:
        load_dotenv()

        data: dict = {}
        config_path = Path(path)
        if config_path.exists():
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}

        config = cls(
            audio=AudioConfig(**data.get("audio", {})),
            whisper=WhisperConfig(**data.get("whisper", {})),
            claude=ClaudeConfig(**data.get("claude", {})),
            piper=PiperConfig(**data.get("piper", {})),
            vad=VADConfig(**data.get("vad", {})),
            wakeword=OpenWakeWordConfig(**data.get("wakeword", {})),
        )

        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if api_key:
            config.claude.api_key = api_key

        openai_key = os.getenv("OPENAI_API_KEY", "")
        if openai_key:
            config.whisper.api_key = openai_key

        config._validate()
        return config

    def _validate(self):
        if not self.claude.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Add it to .env or set the environment variable."
            )
        if self.whisper.backend == "api" and not self.whisper.api_key:
            raise ValueError(
                "OPENAI_API_KEY not set. Required when whisper.backend is 'api'. "
                "Add it to .env or set the environment variable."
            )
        if not Path(self.piper.model_path).exists():
            raise FileNotFoundError(
                f"Piper model not found at {self.piper.model_path}. Run setup.sh first."
            )
