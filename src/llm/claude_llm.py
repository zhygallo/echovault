from __future__ import annotations

import anthropic

from src.llm.base import BaseLLM
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ClaudeLLM(BaseLLM):
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5-20250929",
                 max_tokens: int = 1024, system_prompt: str = "",
                 max_history_pairs: int = 10):
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt
        self._max_history_pairs = max_history_pairs
        self._history: list[dict] = []
        logger.info(f"Claude LLM initialized (model={model}).")

    def respond(self, user_message: str, conversation_history: list[dict] | None = None) -> str:
        self._history.append({"role": "user", "content": user_message})

        # Trim history to max pairs (user + assistant = 2 messages per pair)
        max_messages = self._max_history_pairs * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]

        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=self._system_prompt,
            messages=self._history,
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
        )

        # Extract text from response (skip search result blocks)
        assistant_text = "".join(
            block.text for block in response.content if block.type == "text"
        )
        self._history.append({"role": "assistant", "content": response.content})
        logger.info(f"Claude response: {assistant_text[:80]}...")
        return assistant_text

    def reset_conversation(self):
        self._history.clear()
        logger.info("Conversation history cleared.")
