"""
Skill Manager — Jarvis can create, store, and invoke learned Python skills.

A "skill" is a Python function stored as a .py file under data/skills/.
Jarvis can generate new skills during a session and call them later.
This gives it a growing toolkit that persists across sessions.
"""
from __future__ import annotations

import importlib.util
import logging
import re
import traceback
from pathlib import Path
from typing import Any, Callable

from jarvis.config import config

logger = logging.getLogger(__name__)


class SkillManager:
    def __init__(self, ollama_client, self_model):
        self._llm = ollama_client
        self._self_model = self_model
        self._skills_dir: Path = config.SKILLS_PATH
        self._skills_dir.mkdir(parents=True, exist_ok=True)
        self._loaded: dict[str, Callable] = {}
        self._load_existing_skills()

    # ------------------------------------------------------------------
    # Load existing skills from disk
    # ------------------------------------------------------------------

    def _load_existing_skills(self) -> None:
        for skill_file in self._skills_dir.glob("*.py"):
            self._load_skill_file(skill_file)

    def _load_skill_file(self, path: Path) -> None:
        try:
            spec = importlib.util.spec_from_file_location(path.stem, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            fn = getattr(module, "run", None)
            if callable(fn):
                self._loaded[path.stem] = fn
                logger.debug(f"Loaded skill: {path.stem}")
        except Exception as e:
            logger.warning(f"Failed to load skill '{path.stem}': {e}")

    # ------------------------------------------------------------------
    # Generate a new skill via LLM
    # ------------------------------------------------------------------

    def create_skill(self, task_description: str) -> str | None:
        """
        Ask the LLM to write a Python function that solves the task.
        Returns the skill name if successful, None otherwise.
        """
        prompt = (
            "Write a Python module with a single function called `run(*args, **kwargs)` "
            "that accomplishes the following task:\n\n"
            f"{task_description}\n\n"
            "Rules:\n"
            "- Only standard library imports are allowed (no third-party packages).\n"
            "- The function must return a string result.\n"
            "- Include a module-level docstring with a one-line description.\n"
            "- Name the file appropriately (snake_case, no spaces).\n\n"
            "Return ONLY a JSON object with keys:\n"
            "  name: the snake_case skill name (no .py)\n"
            "  code: the full Python source code as a string\n"
            "No explanation — only valid JSON."
        )
        try:
            raw = self._llm.generate(prompt, temperature=0.2)
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                return None
            import json
            data = json.loads(match.group())
            name = re.sub(r"[^a-z0-9_]", "", data["name"].lower())
            code = data["code"]

            skill_path = self._skills_dir / f"{name}.py"
            skill_path.write_text(code)
            self._load_skill_file(skill_path)
            self._self_model.add_skill(name)
            logger.info(f"Created new skill: {name}")
            return name
        except Exception as e:
            logger.error(f"Skill creation failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Invoke a skill
    # ------------------------------------------------------------------

    def invoke(self, skill_name: str, *args, **kwargs) -> Any:
        fn = self._loaded.get(skill_name)
        if not fn:
            raise ValueError(f"Skill '{skill_name}' not found.")
        try:
            return fn(*args, **kwargs)
        except Exception:
            logger.error(f"Skill '{skill_name}' raised:\n{traceback.format_exc()}")
            raise

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    def list_skills(self) -> list[str]:
        return list(self._loaded.keys())

    def describe_skills(self) -> str:
        if not self._loaded:
            return "No custom skills loaded."
        lines = []
        for name in self._loaded:
            path = self._skills_dir / f"{name}.py"
            doc = ""
            try:
                first_lines = path.read_text().splitlines()[:3]
                for line in first_lines:
                    line = line.strip().strip('"""').strip("'").strip()
                    if line:
                        doc = line
                        break
            except Exception:
                pass
            lines.append(f"  - {name}: {doc}")
        return "\n".join(lines)
