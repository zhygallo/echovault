from __future__ import annotations

import io
import wave

import numpy as np
from openai import OpenAI

from src.stt.base import BaseSTT
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class WhisperAPISTT(BaseSTT):
    """STT backend that uses the OpenAI Whisper API instead of running locally."""

    def __init__(self, api_key: str):
        self._client = OpenAI(api_key=api_key)
        logger.info("Using OpenAI Whisper API for speech-to-text.")

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if audio.dtype != np.int16:
            audio = (audio * 32768.0).clip(-32768, 32767).astype(np.int16)

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit = 2 bytes
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())

        wav_buffer.seek(0)
        wav_buffer.name = "audio.wav"

        result = self._client.audio.transcriptions.create(
            model="whisper-1",
            file=wav_buffer,
        )
        text = result.text.strip()
        logger.info(f"Transcription: {text}")
        return text
