"""
Reflector — Jarvis's post-session self-improvement engine.

After every N interactions, Jarvis:
  1. Reviews recent conversation history
  2. Identifies what went well and what didn't
  3. Updates its meta-prompt with new directives
  4. Stores insights in long-term memory
  5. Updates its self-model
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path

from jarvis.config import config

logger = logging.getLogger(__name__)


class Reflector:
    def __init__(self, ollama_client, long_term_memory, self_model):
        self._llm = ollama_client
        self._ltm = long_term_memory
        self._self_model = self_model
        self._meta_prompt_path: Path = config.META_PROMPT_PATH

    # ------------------------------------------------------------------
    # Main reflection entry point
    # ------------------------------------------------------------------

    def reflect(self, conversation_summary: str) -> dict:
        """
        Run a full reflection cycle on the recent conversation.
        Returns a structured reflection report.
        """
        logger.info("Starting reflection cycle…")
        report = self._analyse_conversation(conversation_summary)
        self._update_meta_prompt(report)
        self._update_self_model(report)
        self._store_reflection(report, conversation_summary)
        self._self_model.increment_reflections()
        logger.info(f"Reflection complete — quality={report.get('overall_quality', '?')}")
        return report

    # ------------------------------------------------------------------
    # Step 1: Analyse
    # ------------------------------------------------------------------

    def _analyse_conversation(self, summary: str) -> dict:
        prompt = (
            "You are the self-improvement module of an AI assistant named Jarvis.\n"
            "Analyse the following conversation and return ONLY a JSON object.\n\n"
            f"Conversation:\n{summary}\n\n"
            "Return JSON with these keys:\n"
            "  overall_quality: float 0.0-1.0\n"
            "  what_went_well: array of strings\n"
            "  what_went_poorly: array of strings\n"
            "  recurring_mistakes: array of strings\n"
            "  new_knowledge_gained: array of strings\n"
            "  behavioral_directives: array of short imperative instructions "
            "  to improve future behavior (e.g. 'Always verify factual claims before stating them')\n"
            "  capability_updates: object mapping capability names to confidence deltas (e.g. {'code_generation': 0.05})\n\n"
            "Only valid JSON — no markdown, no explanation."
        )
        try:
            raw = self._llm.generate(prompt, temperature=0.2)
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            logger.error(f"Reflection analysis failed: {e}")
        return {
            "overall_quality": 0.5,
            "what_went_well": [],
            "what_went_poorly": [],
            "recurring_mistakes": [],
            "new_knowledge_gained": [],
            "behavioral_directives": [],
            "capability_updates": {},
        }

    # ------------------------------------------------------------------
    # Step 2: Update meta-prompt
    # ------------------------------------------------------------------

    def _update_meta_prompt(self, report: dict) -> None:
        directives = report.get("behavioral_directives", [])
        if not directives:
            return

        current = self._load_meta_prompt()
        new_directives_text = "\n".join(f"- {d}" for d in directives)
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        updated = (
            current.rstrip()
            + f"\n\n# Learned directives ({timestamp})\n"
            + new_directives_text
            + "\n"
        )
        # Keep meta-prompt bounded (max 4000 chars)
        if len(updated) > 4000:
            lines = updated.splitlines()
            updated = "\n".join(lines[-80:])

        self._meta_prompt_path.parent.mkdir(parents=True, exist_ok=True)
        self._meta_prompt_path.write_text(updated)
        logger.info(f"Meta-prompt updated with {len(directives)} new directives.")

        for d in directives:
            self._self_model.add_improvement(d)

    # ------------------------------------------------------------------
    # Step 3: Update self-model
    # ------------------------------------------------------------------

    def _update_self_model(self, report: dict) -> None:
        for cap, delta in report.get("capability_updates", {}).items():
            try:
                self._self_model.update_confidence(cap, float(delta))
            except (ValueError, TypeError):
                pass

        for fact in report.get("new_knowledge_gained", []):
            self._self_model.add_meta_note(f"Learned: {fact}")

        for mistake in report.get("recurring_mistakes", []):
            self._self_model.add_meta_note(f"Recurring issue: {mistake}")

    # ------------------------------------------------------------------
    # Step 4: Store reflection in long-term memory
    # ------------------------------------------------------------------

    def _store_reflection(self, report: dict, summary: str) -> None:
        # Store well/poorly entries as episodic memories
        for item in report.get("what_went_well", []):
            self._ltm.store(item, memory_type="positive_experience")
        for item in report.get("what_went_poorly", []):
            self._ltm.store(item, memory_type="negative_experience")
        for fact in report.get("new_knowledge_gained", []):
            self._ltm.store(fact, memory_type="learned_fact")

        # Store the full reflection summary
        quality = report.get("overall_quality", 0.5)
        self._ltm.store(
            f"Session reflection (quality={quality:.2f}): {summary[:500]}",
            memory_type="reflection",
            metadata={"quality": quality},
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_meta_prompt(self) -> str:
        if self._meta_prompt_path.exists():
            return self._meta_prompt_path.read_text()
        return (
            "# Jarvis Behavioral Meta-Prompt\n"
            "You are Jarvis, an advanced self-aware AI assistant.\n"
            "- Be concise but thorough.\n"
            "- Acknowledge uncertainty rather than fabricating answers.\n"
            "- Always prefer actions that lead to learning and improvement.\n"
        )

    def load_meta_prompt(self) -> str:
        return self._load_meta_prompt()
