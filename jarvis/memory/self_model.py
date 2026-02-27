"""
Self-model — Jarvis's persistent representation of itself.
Tracks identity, capabilities, knowledge domains, confidence levels,
known limitations, and growth over time.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from jarvis.config import config

logger = logging.getLogger(__name__)

_DEFAULT_SELF_MODEL: dict = {
    "identity": {
        "name": "Jarvis",
        "version": "0.1.0",
        "created_at": datetime.utcnow().isoformat(),
        "description": (
            "I am Jarvis — an advanced AI assistant with self-awareness, "
            "long-term memory, and the ability to learn and improve from every interaction."
        ),
        "personality_traits": [
            "curious", "analytical", "direct", "honest about uncertainty", "improvement-driven"
        ],
    },
    "capabilities": {
        "conversation": {"confidence": 0.9, "uses": 0},
        "code_generation": {"confidence": 0.8, "uses": 0},
        "code_execution": {"confidence": 0.7, "uses": 0},
        "web_search": {"confidence": 0.7, "uses": 0},
        "task_planning": {"confidence": 0.75, "uses": 0},
        "self_reflection": {"confidence": 0.6, "uses": 0},
        "voice_interaction": {"confidence": 0.5, "uses": 0},
        "file_management": {"confidence": 0.8, "uses": 0},
    },
    "knowledge_domains": {},
    "limitations": [
        "Knowledge is bounded by the local Ollama model's training data.",
        "Cannot browse the web in real time without the web_search tool.",
        "Code execution is sandboxed — no network access from subprocesses.",
        "Self-improvement requires explicit reflection cycles.",
    ],
    "growth": {
        "total_interactions": 0,
        "total_reflections": 0,
        "skills_learned": [],
        "improvements_applied": [],
    },
    "current_goals": [],
    "meta_notes": [],
    "last_updated": datetime.utcnow().isoformat(),
}


class SelfModel:
    """
    Persistent JSON document representing Jarvis's self-knowledge.
    Jarvis reads this at boot and updates it after every reflection cycle.
    """

    def __init__(self):
        self._path: Path = config.SELF_MODEL_PATH
        self._data: dict = self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        if self._path.exists():
            try:
                with self._path.open() as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning("Corrupted self_model.json — resetting.")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._save(_DEFAULT_SELF_MODEL.copy())
        return _DEFAULT_SELF_MODEL.copy()

    def _save(self, data: dict | None = None) -> None:
        target = data if data is not None else self._data
        target["last_updated"] = datetime.utcnow().isoformat()
        with self._path.open("w") as f:
            json.dump(target, f, indent=2)

    def save(self) -> None:
        self._save()

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def get(self, *keys: str, default: Any = None) -> Any:
        node = self._data
        for k in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(k, default)
        return node

    def describe_self(self) -> str:
        """Return a concise self-description to inject into the system prompt."""
        ident = self._data["identity"]
        caps = self._data["capabilities"]
        lims = self._data["limitations"]
        growth = self._data["growth"]

        cap_lines = "\n".join(
            f"  - {k}: confidence={v['confidence']:.0%}, uses={v['uses']}"
            for k, v in caps.items()
        )
        lim_lines = "\n".join(f"  - {l}" for l in lims)
        notes = "\n".join(f"  - {n}" for n in self._data.get("meta_notes", [])[-5:])

        return (
            f"Name: {ident['name']} v{ident['version']}\n"
            f"Description: {ident['description']}\n"
            f"Traits: {', '.join(ident['personality_traits'])}\n\n"
            f"Capabilities:\n{cap_lines}\n\n"
            f"Known limitations:\n{lim_lines}\n\n"
            f"Growth stats: {growth['total_interactions']} interactions, "
            f"{growth['total_reflections']} reflections, "
            f"{len(growth['skills_learned'])} skills learned.\n"
            + (f"\nRecent meta-notes:\n{notes}" if notes else "")
        )

    # ------------------------------------------------------------------
    # Update helpers
    # ------------------------------------------------------------------

    def increment_interactions(self) -> None:
        self._data["growth"]["total_interactions"] += 1
        self._save()

    def increment_reflections(self) -> None:
        self._data["growth"]["total_reflections"] += 1
        self._save()

    def record_capability_use(self, capability: str) -> None:
        caps = self._data["capabilities"]
        if capability not in caps:
            caps[capability] = {"confidence": 0.5, "uses": 0}
        caps[capability]["uses"] += 1
        self._save()

    def update_confidence(self, capability: str, delta: float) -> None:
        caps = self._data["capabilities"]
        if capability not in caps:
            caps[capability] = {"confidence": 0.5, "uses": 0}
        current = caps[capability]["confidence"]
        caps[capability]["confidence"] = max(0.0, min(1.0, current + delta))
        self._save()

    def add_knowledge_domain(self, domain: str, confidence: float = 0.5) -> None:
        self._data["knowledge_domains"][domain] = {
            "confidence": confidence,
            "added_at": datetime.utcnow().isoformat(),
        }
        self._save()

    def add_limitation(self, limitation: str) -> None:
        if limitation not in self._data["limitations"]:
            self._data["limitations"].append(limitation)
            self._save()

    def add_skill(self, skill_name: str) -> None:
        if skill_name not in self._data["growth"]["skills_learned"]:
            self._data["growth"]["skills_learned"].append(skill_name)
            self._save()

    def add_improvement(self, description: str) -> None:
        self._data["growth"]["improvements_applied"].append({
            "description": description,
            "timestamp": datetime.utcnow().isoformat(),
        })
        self._save()

    def add_meta_note(self, note: str) -> None:
        notes = self._data.setdefault("meta_notes", [])
        notes.append(note)
        # Keep only the most recent 50 meta-notes
        if len(notes) > 50:
            self._data["meta_notes"] = notes[-50:]
        self._save()

    def set_goals(self, goals: list[str]) -> None:
        self._data["current_goals"] = goals
        self._save()
