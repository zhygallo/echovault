from __future__ import annotations

import numpy as np
import torch
import whisper

from src.stt.base import BaseSTT
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def _resolve_device(device: str) -> tuple[str, bool]:
    """Resolve device string to (device, fp16) tuple."""
    if device != "auto":
        fp16 = device == "cuda"
        return device, fp16

    if torch.backends.mps.is_available():
        return "mps", False
    if torch.cuda.is_available():
        return "cuda", True
    return "cpu", False


class WhisperSTT(BaseSTT):
    def __init__(self, model_name: str = "turbo", device: str = "auto"):
        resolved_device, self._fp16 = _resolve_device(device)
        logger.info(f"Loading Whisper model '{model_name}' on {resolved_device}...")
        self._model = whisper.load_model(model_name, device=resolved_device)
        logger.info("Whisper model loaded.")

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        # Whisper expects float32 in [-1, 1]
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0

        # Resample to 16kHz if needed (Whisper requires 16kHz)
        if sample_rate != 16000:
            duration = len(audio) / sample_rate
            target_len = int(duration * 16000)
            indices = np.linspace(0, len(audio) - 1, target_len).astype(int)
            audio = audio[indices]

        result = self._model.transcribe(audio, fp16=self._fp16, language="en")
        text = result["text"].strip()
        logger.info(f"Transcription: {text}")
        return text
