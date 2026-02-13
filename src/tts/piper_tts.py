from __future__ import annotations

import io
import wave
from typing import Iterator

import numpy as np
from piper import PiperVoice

from src.tts.base import BaseTTS
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class PiperTTS(BaseTTS):
    def __init__(self, model_path: str, config_path: str | None = None):
        logger.info(f"Loading Piper voice model from {model_path}...")
        self._voice = PiperVoice.load(model_path, config_path=config_path)
        self._sample_rate = self._voice.config.sample_rate
        logger.info(f"Piper loaded (sample_rate={self._sample_rate}).")

    def synthesize(self, text: str) -> np.ndarray:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            self._voice.synthesize(text, wf)
        buf.seek(0)
        with wave.open(buf, "rb") as wf:
            raw = wf.readframes(wf.getnframes())
        return np.frombuffer(raw, dtype=np.int16)

    def synthesize_stream(self, text: str) -> Iterator[bytes]:
        for audio_chunk in self._voice.synthesize(text):
            yield audio_chunk.audio_int16_bytes

    def get_sample_rate(self) -> int:
        return self._sample_rate
