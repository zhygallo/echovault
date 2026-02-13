from __future__ import annotations

import threading
from typing import Callable

import numpy as np
import pyaudio
from openwakeword.model import Model
from openwakeword.utils import download_models

from src.wakeword.base import BaseWakeWord
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class OpenWakeWordDetector(BaseWakeWord):
    def __init__(self, model_name: str = "hey_jarvis", threshold: float = 0.5,
                 chunk_size: int = 1280, sample_rate: int = 16000):
        self._model_name = model_name
        self._threshold = threshold
        self._chunk_size = chunk_size
        self._sample_rate = sample_rate
        self._stop_event = threading.Event()
        self._pa = pyaudio.PyAudio()

        logger.info(f"Loading OpenWakeWord model '{model_name}'...")
        download_models(model_names=[model_name])
        self._model = Model(wakeword_models=[model_name], inference_framework="onnx")
        logger.info("OpenWakeWord model loaded.")

    def listen(self, callback: Callable[[], None]):
        """Listen for wake word in a blocking loop. Calls callback on detection."""
        self._stop_event.clear()

        stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._sample_rate,
            input=True,
            frames_per_buffer=self._chunk_size,
        )

        logger.info(f"Listening for wake word '{self._model_name}'...")

        try:
            while not self._stop_event.is_set():
                data = stream.read(self._chunk_size, exception_on_overflow=False)
                audio = np.frombuffer(data, dtype=np.int16)

                prediction = self._model.predict(audio)

                for model_name, score in prediction.items():
                    if score > self._threshold:
                        logger.info(f"Wake word detected! (score={score:.2f})")
                        self._model.reset()
                        callback()
                        break
        except KeyboardInterrupt:
            pass
        finally:
            stream.stop_stream()
            stream.close()

    def stop(self):
        self._stop_event.set()

    def close(self):
        self.stop()
        self._pa.terminate()
