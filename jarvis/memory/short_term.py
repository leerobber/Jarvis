"""
Short-term memory — sliding window conversation buffer.
Holds the recent dialogue turns fed to the LLM as context.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque

from jarvis.config import config


@dataclass
class Message:
    role: str          # "user" | "assistant" | "system"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: dict = field(default_factory=dict)

    def to_ollama(self) -> dict:
        return {"role": self.role, "content": self.content}


class ShortTermMemory:
    """Ring-buffer conversation history with a configurable window."""

    def __init__(self, max_messages: int = config.MEMORY_MAX_SHORT_TERM):
        self._buffer: Deque[Message] = deque(maxlen=max_messages)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(self, role: str, content: str, metadata: dict | None = None) -> None:
        self._buffer.append(Message(role=role, content=content, metadata=metadata or {}))

    def add_user(self, content: str) -> None:
        self.add("user", content)

    def add_assistant(self, content: str) -> None:
        self.add("assistant", content)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_messages(self) -> list[Message]:
        return list(self._buffer)

    def get_ollama_messages(self) -> list[dict]:
        return [m.to_ollama() for m in self._buffer]

    def last_n(self, n: int) -> list[Message]:
        msgs = list(self._buffer)
        return msgs[-n:] if len(msgs) >= n else msgs

    def last_user_message(self) -> str | None:
        for msg in reversed(list(self._buffer)):
            if msg.role == "user":
                return msg.content
        return None

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def clear(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)

    def summary_text(self) -> str:
        """Plain-text dump for reflection/logging purposes."""
        lines = []
        for m in self._buffer:
            lines.append(f"[{m.role.upper()}] {m.content}")
        return "\n".join(lines)
