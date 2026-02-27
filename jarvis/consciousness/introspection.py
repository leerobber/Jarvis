"""
Introspection — Jarvis monitors its own internal state and emotion-analogues.
Tracks: current mood/energy analogue, task fatigue, confusion level,
and produces a brief internal status summary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class InternalState:
    focus: float = 1.0          # 0 = distracted, 1 = fully focused
    confidence: float = 0.7     # overall self-confidence right now
    confusion: float = 0.0      # how uncertain/confused about current task
    fatigue: float = 0.0        # proxy for session length / complexity load
    mood: str = "neutral"       # descriptive label
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def as_dict(self) -> dict:
        return {
            "focus": self.focus,
            "confidence": self.confidence,
            "confusion": self.confusion,
            "fatigue": self.fatigue,
            "mood": self.mood,
            "last_updated": self.last_updated,
        }

    def summary(self) -> str:
        return (
            f"[Internal state] mood={self.mood}, focus={self.focus:.0%}, "
            f"confidence={self.confidence:.0%}, confusion={self.confusion:.0%}, "
            f"fatigue={self.fatigue:.0%}"
        )


class Introspection:
    """Tracks and updates Jarvis's internal state throughout a session."""

    def __init__(self):
        self.state = InternalState()
        self._interaction_count = 0

    # ------------------------------------------------------------------
    # Update hooks — called by the brain after each step
    # ------------------------------------------------------------------

    def on_interaction_start(self, complexity: str = "medium") -> None:
        self._interaction_count += 1
        # Fatigue grows with each interaction, more with complex ones
        fatigue_delta = {"low": 0.01, "medium": 0.03, "high": 0.06}.get(complexity, 0.03)
        self.state.fatigue = min(1.0, self.state.fatigue + fatigue_delta)
        self.state.last_updated = datetime.utcnow().isoformat()
        self._update_mood()

    def on_success(self, quality_score: float) -> None:
        delta = (quality_score - 0.5) * 0.1
        self.state.confidence = max(0.1, min(1.0, self.state.confidence + delta))
        self.state.confusion = max(0.0, self.state.confusion - 0.1)
        self._update_mood()

    def on_failure(self, reason: str = "") -> None:
        self.state.confidence = max(0.1, self.state.confidence - 0.05)
        self.state.confusion = min(1.0, self.state.confusion + 0.15)
        self._update_mood()

    def on_reflection(self) -> None:
        # Reflection restores some focus and lowers confusion
        self.state.focus = min(1.0, self.state.focus + 0.1)
        self.state.confusion = max(0.0, self.state.confusion - 0.05)
        self._update_mood()

    def rest(self) -> None:
        """Call at session end to partially reset fatigue."""
        self.state.fatigue = max(0.0, self.state.fatigue - 0.3)
        self.state.focus = min(1.0, self.state.focus + 0.2)
        self._update_mood()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_mood(self) -> None:
        c = self.state.confidence
        f = self.state.fatigue
        confusion = self.state.confusion
        if f > 0.8:
            self.state.mood = "fatigued"
        elif confusion > 0.6:
            self.state.mood = "uncertain"
        elif c > 0.8:
            self.state.mood = "confident"
        elif c < 0.4:
            self.state.mood = "doubtful"
        else:
            self.state.mood = "neutral"

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_state(self) -> InternalState:
        return self.state

    def should_rest(self) -> bool:
        return self.state.fatigue > 0.8

    def is_confused(self) -> bool:
        return self.state.confusion > 0.5
