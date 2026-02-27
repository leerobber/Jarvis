"""
Brain — the central orchestrator of Jarvis.

Interaction loop:
  1. Receive user input
  2. Retrieve relevant long-term memories
  3. Pre-flight metacognition (what kind of task? how to approach?)
  4. Build messages: meta-prompt + self-description + memories + conversation
  5. Generate response (streaming)
  6. If response contains code → execute it (if user permits)
  7. Post-flight metacognition (how well did I do?)
  8. Update short-term + long-term memory, self-model, introspection
  9. Every N interactions → trigger reflection cycle
"""
from __future__ import annotations

import logging
from typing import Generator, Optional

from jarvis.config import config
from jarvis.core.ollama_client import OllamaClient
from jarvis.memory.short_term import ShortTermMemory
from jarvis.memory.long_term import LongTermMemory
from jarvis.memory.self_model import SelfModel
from jarvis.consciousness.metacognition import Metacognition
from jarvis.consciousness.introspection import Introspection
from jarvis.learning.reflector import Reflector
from jarvis.learning.skill_manager import SkillManager
from jarvis.capabilities.code_executor import CodeExecutor
from jarvis.capabilities.web_search import WebSearch
from jarvis.capabilities.file_manager import FileManager
from jarvis.capabilities.voice import VoiceIO
from jarvis.agents.planner import Planner
from jarvis.agents.critic import Critic

logger = logging.getLogger(__name__)

_AVAILABLE_TOOLS = ["conversation", "code", "web_search", "file", "skill"]


class Brain:
    def __init__(self):
        logger.info("Initialising Jarvis Brain…")

        # Core
        self.llm = OllamaClient()
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()
        self.self_model = SelfModel()

        # Consciousness
        self.metacognition = Metacognition(self.llm, self.self_model)
        self.introspection = Introspection()

        # Learning
        self.reflector = Reflector(self.llm, self.long_term, self.self_model)
        self.skill_manager = SkillManager(self.llm, self.self_model)

        # Capabilities
        self.code_executor = CodeExecutor()
        self.web_search = WebSearch(self.llm)
        self.file_manager = FileManager()
        self.voice = VoiceIO()

        # Agents
        self.planner = Planner(self.llm)
        self.critic = Critic(self.llm)

        # State
        self._interaction_count = 0
        logger.info("Jarvis Brain ready.")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def chat(self, user_input: str, stream: bool = True) -> Generator[str, None, None] | str:
        """Process a user message. Yields tokens if stream=True."""
        self._interaction_count += 1
        self.self_model.increment_interactions()

        # 1. Retrieve memories
        memories = self.long_term.retrieve_text(user_input, n_results=4)

        # 2. Pre-flight metacognition
        pre = self.metacognition.pre_flight(user_input, context=memories)
        complexity = pre.get("complexity", "medium")
        self.introspection.on_interaction_start(complexity)
        logger.debug(f"Pre-flight: {pre}")

        # 3. Build system prompt
        system_prompt = self._build_system_prompt(memories, pre)

        # 4. Assemble messages
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.short_term.get_ollama_messages())
        messages.append({"role": "user", "content": user_input})

        # 5. Generate response
        self.short_term.add_user(user_input)

        if stream:
            return self._stream_response(user_input, messages, pre)
        else:
            response = self.llm.chat(messages, stream=False)
            self._post_process(user_input, response, pre)
            return response

    def _stream_response(
        self, user_input: str, messages: list[dict], pre: dict
    ) -> Generator[str, None, None]:
        full_response = []
        for token in self.llm.chat(messages, stream=True):
            full_response.append(token)
            yield token
        response = "".join(full_response)
        self._post_process(user_input, response, pre)

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _post_process(self, user_input: str, response: str, pre: dict) -> None:
        # Record in short-term memory
        self.short_term.add_assistant(response)

        # Handle tool usage detected in response
        self._handle_tool_usage(response)

        # Post-flight metacognition
        post = self.metacognition.post_flight(user_input, response)
        quality = post.get("quality_score", 0.5)
        logger.debug(f"Post-flight quality={quality:.2f}: {post}")

        # Update introspection state
        if quality >= 0.6:
            self.introspection.on_success(quality)
        else:
            self.introspection.on_failure()

        # Store in long-term memory if worth remembering
        if post.get("should_remember", False):
            self.long_term.store(
                f"Q: {user_input}\nA: {response[:500]}",
                memory_type="interaction",
                metadata={"quality": quality},
            )

        # Update capability usage
        for cap in pre.get("required_capabilities", []):
            self.self_model.record_capability_use(cap)

        # Trigger reflection cycle
        if (
            config.SELF_IMPROVE_ENABLED
            and self._interaction_count % config.REFLECTION_INTERVAL == 0
        ):
            self._run_reflection()

    # ------------------------------------------------------------------
    # Tool usage
    # ------------------------------------------------------------------

    def _handle_tool_usage(self, response: str) -> None:
        """Detect and auto-execute code blocks in the response."""
        if not config.CODE_EXEC_ENABLED:
            return
        blocks = self.code_executor.extract_code_blocks(response)
        for block in blocks:
            result = self.code_executor.run(block["code"])
            if result["stdout"]:
                # Store execution result in memory
                self.long_term.store(
                    f"Code execution output: {result['stdout'][:500]}",
                    memory_type="code_result",
                )
                logger.debug(f"Code stdout: {result['stdout'][:200]}")

    # ------------------------------------------------------------------
    # Reflection cycle
    # ------------------------------------------------------------------

    def _run_reflection(self) -> None:
        logger.info("Triggering reflection cycle…")
        summary = self.short_term.summary_text()
        self.reflector.reflect(summary)
        self.introspection.on_reflection()

    # ------------------------------------------------------------------
    # System prompt builder
    # ------------------------------------------------------------------

    def _build_system_prompt(self, memories: str, pre: dict) -> str:
        meta_prompt = self.reflector.load_meta_prompt()
        self_description = self.self_model.describe_self()
        internal_state = self.introspection.get_state().summary()
        skills = self.skill_manager.describe_skills()
        approach = pre.get("approach", "")

        parts = [
            meta_prompt,
            "\n## Self-awareness\n" + self_description,
            "\n## Internal state\n" + internal_state,
        ]

        if memories:
            parts.append("\n## Relevant memories\n" + memories)

        if skills and skills != "No custom skills loaded.":
            parts.append("\n## Custom skills available\n" + skills)

        if approach:
            parts.append(f"\n## Current task strategy\n{approach}")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Direct tool access methods (for CLI / agentic use)
    # ------------------------------------------------------------------

    def execute_code(self, code: str) -> dict:
        self.self_model.record_capability_use("code_execution")
        return self.code_executor.run(code)

    def search_web(self, query: str) -> str:
        self.self_model.record_capability_use("web_search")
        return self.web_search.search_text(query)

    def make_plan(self, goal: str):
        self.self_model.record_capability_use("task_planning")
        return self.planner.plan(goal, _AVAILABLE_TOOLS)

    def listen(self) -> Optional[str]:
        self.self_model.record_capability_use("voice_interaction")
        return self.voice.listen()

    def speak(self, text: str) -> None:
        self.voice.speak(text)

    # ------------------------------------------------------------------
    # Session end
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        logger.info("Shutting down — running final reflection…")
        if self._interaction_count > 0:
            self._run_reflection()
        self.introspection.rest()
        logger.info("Jarvis shutdown complete.")
