from __future__ import annotations

import threading

import numpy as np
import pyaudio
import webrtcvad

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class AudioRecorder:
    def __init__(self, sample_rate: int = 16000, channels: int = 1,
                 chunk_size: int = 1024):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self._pa = pyaudio.PyAudio()

    def record_until_enter(self) -> np.ndarray:
        """Record audio until the user presses Enter. Returns int16 ndarray."""
        stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )

        frames: list[bytes] = []
        stop_event = threading.Event()

        def _capture():
            while not stop_event.is_set():
                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    frames.append(data)
                except OSError:
                    break

        thread = threading.Thread(target=_capture, daemon=True)
        thread.start()

        logger.info("Recording... press Enter to stop.")
        input()
        stop_event.set()
        thread.join(timeout=1.0)

        stream.stop_stream()
        stream.close()

        if not frames:
            return np.array([], dtype=np.int16)

        audio = np.frombuffer(b"".join(frames), dtype=np.int16)
        duration = len(audio) / self.sample_rate
        logger.info(f"Recorded {duration:.1f}s of audio.")
        return audio

    def record_with_vad(self, aggressiveness: int = 2,
                        silence_timeout: float = 1.0,
                        frame_duration_ms: int = 30,
                        listen_timeout: float = 0) -> np.ndarray:
        """Record audio using VAD, stopping after silence_timeout seconds of silence.

        Args:
            listen_timeout: Max seconds to wait for speech to start. 0 = no limit.
        """
        vad = webrtcvad.Vad(aggressiveness)
        frame_size = int(self.sample_rate * frame_duration_ms / 1000)  # samples per frame
        frame_bytes = frame_size * 2  # int16 = 2 bytes per sample

        stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=frame_size,
        )

        frames: list[bytes] = []
        speech_started = False
        silent_frames = 0
        waiting_frames = 0
        max_silent_frames = int(silence_timeout * 1000 / frame_duration_ms)
        max_waiting_frames = int(listen_timeout * 1000 / frame_duration_ms) if listen_timeout > 0 else 0

        logger.info("Listening for speech...")

        try:
            while True:
                data = stream.read(frame_size, exception_on_overflow=False)
                if len(data) < frame_bytes:
                    continue

                is_speech = vad.is_speech(data[:frame_bytes], self.sample_rate)

                if is_speech:
                    if not speech_started:
                        speech_started = True
                        logger.info("Speech detected, recording...")
                    silent_frames = 0
                    frames.append(data)
                elif speech_started:
                    frames.append(data)
                    silent_frames += 1
                    if silent_frames >= max_silent_frames:
                        logger.info("Silence detected, stopping recording.")
                        break
                else:
                    waiting_frames += 1
                    if max_waiting_frames > 0 and waiting_frames >= max_waiting_frames:
                        logger.info("No speech detected within listen timeout.")
                        break
        except KeyboardInterrupt:
            pass
        finally:
            stream.stop_stream()
            stream.close()

        if not frames:
            return np.array([], dtype=np.int16)

        audio = np.frombuffer(b"".join(frames), dtype=np.int16)
        duration = len(audio) / self.sample_rate
        logger.info(f"Recorded {duration:.1f}s of audio.")
        return audio

    def close(self):
        self._pa.terminate()
