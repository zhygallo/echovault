from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any, Callable


class EventBus:
    """Simple thread-safe publish/subscribe event bus."""

    def __init__(self):
        self._listeners: dict[str, list[Callable]] = defaultdict(list)
        self._lock = threading.Lock()

    def on(self, event: str, callback: Callable) -> None:
        with self._lock:
            self._listeners[event].append(callback)

    def emit(self, event: str, data: Any = None) -> None:
        with self._lock:
            callbacks = list(self._listeners.get(event, []))
        for cb in callbacks:
            cb(data)
