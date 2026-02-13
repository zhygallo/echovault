from __future__ import annotations

from typing import Iterator

import numpy as np
import sounddevice as sd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class AudioPlayer:
    def play(self, audio: np.ndarray, sample_rate: int):
        """Play an audio array (int16 or float32) and block until done."""
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        sd.play(audio, samplerate=sample_rate)
        sd.wait()

    def play_stream(self, chunks: Iterator[bytes], sample_rate: int):
        """Play streaming int16 audio chunks."""
        buffer: list[bytes] = []
        for chunk in chunks:
            buffer.append(chunk)

        if not buffer:
            return

        audio = np.frombuffer(b"".join(buffer), dtype=np.int16)
        self.play(audio, sample_rate)
