from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable


class BaseWakeWord(ABC):
    @abstractmethod
    def listen(self, callback: Callable[[], None]):
        """Start listening for wake word. Call callback when detected."""
        ...

    @abstractmethod
    def stop(self):
        """Stop listening."""
        ...
