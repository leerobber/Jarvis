"""
Planner — decomposes complex tasks into an ordered list of steps
and identifies which capability/tool each step requires.

Uses fast_generate() for the decomposition step.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Step:
    index: int
    description: str
    tool: str            # "conversation" | "code" | "web_search" | "file" | "skill"
    args: dict = field(default_factory=dict)
    result: Optional[str] = None
    status: str = "pending"   # pending | running | done | failed


@dataclass
class Plan:
    goal: str
    steps: list[Step]
    status: str = "pending"   # pending | running | done | failed

    def next_step(self) -> Optional[Step]:
        for s in self.steps:
            if s.status == "pending":
                return s
        return None

    def all_done(self) -> bool:
        return all(s.status in ("done", "failed") for s in self.steps)

    def summary(self) -> str:
        lines = [f"Goal: {self.goal}"]
        for s in self.steps:
            icon = {"pending": "○", "running": "→", "done": "✓", "failed": "✗"}.get(s.status, "?")
            result_preview = f"  → {s.result[:80]}" if s.result else ""
            lines.append(f"  {icon} [{s.tool}] {s.description}{result_preview}")
        return "\n".join(lines)


class Planner:
    def __init__(self, ollama_client):
        self._llm = ollama_client

    def plan(self, goal: str, available_tools: list[str]) -> Plan:
        """
        Decompose goal into steps.  Returns a Plan object.
        """
        tools_str = ", ".join(available_tools)
        prompt = (
            "You are a task planning module. Decompose the following goal into an "
            "ordered list of concrete steps. For each step specify which tool to use.\n\n"
            f"Goal: {goal}\n"
            f"Available tools: {tools_str}\n\n"
            "Return ONLY a JSON object with keys:\n"
            "  goal: string\n"
            "  steps: array of objects, each with: description (string), tool (string from available tools)\n"
            "No explanation — only valid JSON."
        )
        try:
            raw = self._llm.fast_generate(prompt, temperature=0.2)
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
                steps = [
                    Step(index=i, description=s["description"], tool=s.get("tool", "conversation"))
                    for i, s in enumerate(data.get("steps", []))
                ]
                return Plan(goal=goal, steps=steps)
        except Exception as e:
            logger.error(f"Planning failed: {e}")

        # Fallback: single step
        return Plan(goal=goal, steps=[Step(index=0, description=goal, tool="conversation")])
