from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

import numpy as np


class BaseTTS(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text to audio array (int16)."""
        ...

    @abstractmethod
    def synthesize_stream(self, text: str) -> Iterator[bytes]:
        """Synthesize text and yield raw audio chunks (int16 bytes)."""
        ...

    @abstractmethod
    def get_sample_rate(self) -> int:
        """Return the output sample rate."""
        ...
