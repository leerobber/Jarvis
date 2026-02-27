"""
Metacognition — Jarvis thinks about its own thinking.

Before responding: pre-flight reasoning check.
After responding: post-flight quality assessment.
These are internal monologue steps hidden from the user but logged and
used by the learning system.
"""
from __future__ import annotations

import logging

from jarvis.config import config

logger = logging.getLogger(__name__)


class Metacognition:
    """
    Uses the LLM to reason about the quality and nature of its own
    reasoning before and after each user interaction.
    """

    def __init__(self, ollama_client, self_model):
        self._llm = ollama_client
        self._self_model = self_model

    # ------------------------------------------------------------------
    # Pre-response: "What kind of task is this, and how should I approach it?"
    # ------------------------------------------------------------------

    def pre_flight(self, user_input: str, context: str = "") -> dict:
        """
        Analyse the incoming request and return a strategy dict:
          - task_type: e.g. "code", "factual", "creative", "planning"
          - complexity: "low" | "medium" | "high"
          - required_capabilities: list of capability names
          - confidence_estimate: 0.0–1.0
          - approach: short description of how to tackle it
        """
        prompt = (
            f"You are the internal reasoning module of {self._self_model.get('identity', 'name')}.\n"
            f"Analyse the following user request and return ONLY a JSON object.\n\n"
            f"User request: {user_input}\n\n"
            f"Context available:\n{context or 'None'}\n\n"
            "Return JSON with keys: task_type, complexity, required_capabilities (array), "
            "confidence_estimate (0.0-1.0), approach (one sentence).\n"
            "No explanation — only valid JSON."
        )
        try:
            raw = self._llm.generate(prompt, temperature=0.1)
            import json, re
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            logger.debug(f"pre_flight parse error: {e}")
        # Fallback
        return {
            "task_type": "general",
            "complexity": "medium",
            "required_capabilities": ["conversation"],
            "confidence_estimate": 0.6,
            "approach": "Respond thoughtfully using available context.",
        }

    # ------------------------------------------------------------------
    # Post-response: "How well did I do?"
    # ------------------------------------------------------------------

    def post_flight(self, user_input: str, response: str) -> dict:
        """
        Evaluate the quality of a response and return:
          - quality_score: 0.0–1.0
          - strengths: list of strings
          - weaknesses: list of strings
          - improvement_suggestions: list of strings
          - should_remember: bool — worth storing in long-term memory?
        """
        prompt = (
            "You are the self-evaluation module of an AI assistant.\n"
            "Evaluate the following interaction and return ONLY a JSON object.\n\n"
            f"User asked: {user_input}\n\n"
            f"Assistant responded: {response}\n\n"
            "Return JSON with keys: quality_score (0.0-1.0), strengths (array), "
            "weaknesses (array), improvement_suggestions (array), should_remember (bool).\n"
            "No explanation — only valid JSON."
        )
        try:
            raw = self._llm.generate(prompt, temperature=0.1)
            import json, re
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            logger.debug(f"post_flight parse error: {e}")
        return {
            "quality_score": 0.5,
            "strengths": [],
            "weaknesses": [],
            "improvement_suggestions": [],
            "should_remember": False,
        }

    # ------------------------------------------------------------------
    # Uncertainty check
    # ------------------------------------------------------------------

    def estimate_confidence(self, statement: str) -> float:
        """
        Return a 0–1 confidence score for whether the statement is correct.
        Used before including factual claims in responses.
        """
        prompt = (
            "Rate your confidence that the following statement is accurate. "
            "Reply with ONLY a decimal between 0.0 and 1.0.\n\n"
            f"Statement: {statement}"
        )
        try:
            raw = self._llm.generate(prompt, temperature=0.0).strip()
            import re
            match = re.search(r"[01]?\.\d+", raw)
            if match:
                return float(match.group())
        except Exception:
            pass
        return 0.5
