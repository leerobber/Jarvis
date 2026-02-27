"""
Critic — evaluates Jarvis's own plans and responses before finalising them.
Implements a "think twice" loop to catch errors and improve output quality.
"""
from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)


class Critic:
    def __init__(self, ollama_client):
        self._llm = ollama_client

    # ------------------------------------------------------------------
    # Critique a draft response
    # ------------------------------------------------------------------

    def critique_response(self, user_input: str, draft: str) -> dict:
        """
        Review a draft response and return:
          { approved: bool, issues: list[str], revised: str | None }
        """
        prompt = (
            "You are a quality-control module reviewing an AI assistant's draft response.\n\n"
            f"User asked: {user_input}\n\n"
            f"Draft response:\n{draft}\n\n"
            "Check for: factual errors, missing information, unclear logic, "
            "unnecessarily verbose wording, unhelpful tone.\n\n"
            "Return ONLY JSON with keys:\n"
            "  approved: bool\n"
            "  issues: array of strings (empty if none)\n"
            "  revised: improved version (string) or null if approved as-is\n"
            "Only valid JSON — no markdown."
        )
        try:
            raw = self._llm.generate(prompt, temperature=0.1)
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            logger.debug(f"Critic parse error: {e}")
        return {"approved": True, "issues": [], "revised": None}

    # ------------------------------------------------------------------
    # Critique a plan before execution
    # ------------------------------------------------------------------

    def critique_plan(self, goal: str, plan_summary: str) -> dict:
        """
        Review a plan and return:
          { viable: bool, risks: list[str], suggestions: list[str] }
        """
        prompt = (
            "You are a plan reviewer. Assess whether the following plan is sensible, "
            "complete, and safe to execute.\n\n"
            f"Goal: {goal}\n\n"
            f"Plan:\n{plan_summary}\n\n"
            "Return ONLY JSON with keys:\n"
            "  viable: bool\n"
            "  risks: array of strings\n"
            "  suggestions: array of strings\n"
            "Only valid JSON — no markdown."
        )
        try:
            raw = self._llm.generate(prompt, temperature=0.1)
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            logger.debug(f"Plan critique error: {e}")
        return {"viable": True, "risks": [], "suggestions": []}

    # ------------------------------------------------------------------
    # Fact-check a statement
    # ------------------------------------------------------------------

    def fact_check(self, statement: str) -> dict:
        """
        Return { likely_accurate: bool, confidence: float, caveat: str }
        """
        prompt = (
            "Fact-check the following statement based on your training knowledge.\n\n"
            f"Statement: {statement}\n\n"
            "Return ONLY JSON with keys:\n"
            "  likely_accurate: bool\n"
            "  confidence: float 0.0-1.0\n"
            "  caveat: string (empty if none)\n"
            "Only valid JSON."
        )
        try:
            raw = self._llm.generate(prompt, temperature=0.0)
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            logger.debug(f"Fact-check error: {e}")
        return {"likely_accurate": True, "confidence": 0.5, "caveat": ""}
