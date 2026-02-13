from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    @abstractmethod
    def respond(self, user_message: str, conversation_history: list[dict] | None = None) -> str:
        """Generate a response to the user message."""
        ...

    @abstractmethod
    def reset_conversation(self):
        """Clear conversation history."""
        ...
