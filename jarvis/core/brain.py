"""
Brain — the central orchestrator of Jarvis.

Interaction loop:
  1. Receive user input
  2. Retrieve relevant long-term memories
  3. Pre-flight metacognition (what kind of task? how to approach?)
  4. Build messages: meta-prompt + self-description + memories + conversation
  5. Generate response (streaming)
  6. Parse tool calls embedded in response ([[SEARCH:]], [[CODE:]], [[FILE:]], [[SKILL:]])
  7. Auto-execute code with retry-on-error debug loop
  8. Run critic on final response (think-twice loop)
  9. Post-flight metacognition (how well did I do?)
  10. Update short-term + long-term memory, self-model, introspection
  11. Every N interactions → reflection cycle
  12. Every M interactions → memory consolidation

Tool-call syntax for the LLM:
  [[SEARCH: <query>]]
  [[CODE: <python code>]]
  [[FILE: read|<path>]]
  [[FILE: write|<path>|<content>]]
  [[FILE: list|<directory>]]
  [[SKILL: <name>|<arg1>|<arg2>...]]
  [[CREATE_SKILL: <one-line description>]]
  [[PLAN: <goal>]]
  [[MEMO: <text to remember>]]
"""
from __future__ import annotations

import logging
import re
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
from jarvis.agents.planner import Planner, Plan, Step
from jarvis.agents.critic import Critic

logger = logging.getLogger(__name__)

_AVAILABLE_TOOLS = ["conversation", "code", "web_search", "file", "skill"]

# Goals injected into the system prompt are capped to prevent unbounded token growth.
_GOALS_MAX_COUNT = 10
_GOALS_MAX_GOAL_LENGTH = 200

# Regex patterns for inline tool calls
_RE_SEARCH = re.compile(r"\[\[SEARCH:\s*(.*?)\]\]", re.DOTALL)
_RE_CODE   = re.compile(r"\[\[CODE:\s*(.*?)\]\]", re.DOTALL)
_RE_FILE   = re.compile(r"\[\[FILE:\s*(.*?)\]\]", re.DOTALL)
_RE_SKILL  = re.compile(r"\[\[SKILL:\s*(.*?)\]\]", re.DOTALL)
_RE_CREATE = re.compile(r"\[\[CREATE_SKILL:\s*(.*?)\]\]", re.DOTALL)
_RE_PLAN   = re.compile(r"\[\[PLAN:\s*(.*?)\]\]", re.DOTALL)
_RE_MEMO   = re.compile(r"\[\[MEMO:\s*(.*?)\]\]", re.DOTALL)


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

        # 0. Compress context window before it silently loses old turns
        self._maybe_compress_context()

        # 1. Retrieve memories — only keep sufficiently relevant ones (≥ 0.4 similarity)
        memories = self.long_term.retrieve_text(user_input, n_results=4, min_relevance=0.4)

        # 2. Pre-flight metacognition
        pre = self.metacognition.pre_flight(user_input, context=memories)
        complexity = pre.get("complexity", "medium")
        self.introspection.on_interaction_start(complexity)
        logger.debug(f"Pre-flight: {pre}")

        # 2b. Proactive web search when confidence is low on factual queries
        if (
            config.PROACTIVE_SEARCH_ENABLED
            and pre.get("confidence_estimate", 1.0) < config.PROACTIVE_SEARCH_CONFIDENCE_THRESHOLD
            and pre.get("task_type", "") in ("factual", "general", "knowledge")
        ):
            try:
                extra = self.web_search.search_text(user_input, max_results=3)
                if extra:
                    memories = (memories + "\n\n[Proactive search]\n" + extra).strip()
                    logger.info("Proactive search triggered (low pre-flight confidence).")
            except Exception as e:
                logger.debug(f"Proactive search failed: {e}")

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

        # Run tool calls — results appended as a follow-up yield
        tool_output = self._handle_tool_usage(response)
        if tool_output:
            yield tool_output

        self._post_process(user_input, response + tool_output, pre)

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _post_process(self, user_input: str, response: str, pre: dict) -> None:
        # Think-twice: critic reviews the response and may flag corrections
        if config.CRITIC_ENABLED and pre.get("complexity") != "low":
            critique = self.critic.critique_response(user_input, response)
            if not critique.get("approved", True) and critique.get("issues"):
                issues = "; ".join(critique["issues"][:3])
                logger.info(f"Critic flagged issues: {issues}")
                # Store the critique as a learning note
                self.long_term.store(
                    f"Critique of response to '{user_input[:80]}': {issues}",
                    memory_type="self_critique",
                )
                # Store the critic's improved version so future retrievals benefit
                revised = (critique.get("revised") or "").strip()
                if revised:
                    self.long_term.store(
                        f"Improved response to '{user_input[:80]}': {revised[:600]}",
                        memory_type="corrected_response",
                    )

        # Record in short-term memory
        self.short_term.add_assistant(response)

        # Post-flight metacognition
        post = self.metacognition.post_flight(user_input, response)
        quality = post.get("quality_score", 0.5)
        logger.debug(f"Post-flight quality={quality:.2f}: {post}")

        # Persist improvement suggestions so they accumulate in the self-model
        for suggestion in post.get("improvement_suggestions", [])[:3]:
            self.self_model.add_meta_note(f"Improvement: {suggestion}")

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

        # Periodic reflection
        if (
            config.SELF_IMPROVE_ENABLED
            and self._interaction_count % config.REFLECTION_INTERVAL == 0
        ):
            self._run_reflection()

        # Periodic memory consolidation
        if (
            config.MEMORY_CONSOLIDATION_INTERVAL > 0
            and self._interaction_count % config.MEMORY_CONSOLIDATION_INTERVAL == 0
        ):
            removed = self.long_term.consolidate(self.llm)
            if removed:
                logger.info(f"Memory consolidation removed {removed} redundant entries.")

    # ------------------------------------------------------------------
    # Tool-call parser — handles inline [[TOOL: ...]] syntax
    # ------------------------------------------------------------------

    def _handle_tool_usage(self, response: str) -> str:
        """
        Parse [[TOOL: ...]] calls embedded in the LLM response.
        Returns a formatted string of tool results (empty string if none triggered).
        """
        results: list[str] = []

        # Web search
        for m in _RE_SEARCH.finditer(response):
            query = m.group(1).strip()
            logger.info(f"Tool: SEARCH '{query}'")
            try:
                text = self.web_search.search_text(query)
                self.self_model.record_capability_use("web_search")
                results.append(f"\n**[Search: {query}]**\n{text}")
                self.long_term.store(f"Web search '{query}': {text[:400]}", memory_type="web_search")
            except Exception as e:
                results.append(f"\n**[Search: {query}]** Error: {e}")

        # Code execution (with auto-debug retry)
        for m in _RE_CODE.finditer(response):
            code = m.group(1).strip()
            logger.info("Tool: CODE execution")
            result = self._execute_with_debug(code)
            self.self_model.record_capability_use("code_execution")
            if result["success"]:
                results.append(f"\n**[Code output]**\n```\n{result['stdout']}\n```")
                self.long_term.store(f"Code executed OK: {result['stdout'][:300]}", memory_type="code_result")
            else:
                err = result.get("error") or result.get("stderr", "")
                results.append(f"\n**[Code error]** {err}")

        # File operations
        for m in _RE_FILE.finditer(response):
            args = [p.strip() for p in m.group(1).split("|")]
            op = args[0].lower() if args else ""
            logger.info(f"Tool: FILE {op}")
            try:
                if op == "read" and len(args) >= 2:
                    content = self.file_manager.read(args[1])
                    results.append(f"\n**[File: {args[1]}]**\n```\n{content[:1000]}\n```")
                elif op == "write" and len(args) >= 3:
                    self.file_manager.write(args[1], args[2])
                    results.append(f"\n**[File written: {args[1]}]**")
                elif op == "list":
                    directory = args[1] if len(args) >= 2 else "."
                    items = self.file_manager.list(directory)
                    results.append(f"\n**[Files in {directory}]**\n" + "\n".join(items))
                elif op == "delete" and len(args) >= 2:
                    self.file_manager.delete(args[1])
                    results.append(f"\n**[Deleted: {args[1]}]**")
                else:
                    results.append(f"\n**[File]** Unknown op or missing args: {m.group(1)}")
                self.self_model.record_capability_use("file_management")
            except Exception as e:
                results.append(f"\n**[File error]** {e}")

        # Skill invocation
        for m in _RE_SKILL.finditer(response):
            parts = [p.strip() for p in m.group(1).split("|")]
            skill_name = parts[0]
            skill_args = parts[1:]
            logger.info(f"Tool: SKILL '{skill_name}'")
            try:
                output = self.skill_manager.invoke(skill_name, *skill_args)
                results.append(f"\n**[Skill: {skill_name}]**\n{output}")
                self.self_model.record_capability_use("skill_use")
            except Exception as e:
                results.append(f"\n**[Skill: {skill_name}]** Error: {e}")

        # Skill creation
        for m in _RE_CREATE.finditer(response):
            description = m.group(1).strip()
            logger.info(f"Tool: CREATE_SKILL '{description}'")
            name = self._create_skill_with_validation(description)
            if name:
                results.append(f"\n**[New skill created: `{name}`]**")
            else:
                results.append(f"\n**[Skill creation failed]** Could not generate a working skill.")

        # Inline planning
        for m in _RE_PLAN.finditer(response):
            goal = m.group(1).strip()
            logger.info(f"Tool: PLAN '{goal}'")
            try:
                plan = self.planner.plan(goal, _AVAILABLE_TOOLS)
                self.self_model.record_capability_use("task_planning")
                results.append(f"\n**[Plan for: {goal}]**\n{plan.summary()}")
                self.long_term.store(
                    f"Plan created for goal '{goal[:80]}': {plan.summary()[:400]}",
                    memory_type="plan",
                )
            except Exception as e:
                results.append(f"\n**[Plan: {goal}]** Error: {e}")

        # Explicit memory storage
        for m in _RE_MEMO.finditer(response):
            text = m.group(1).strip()[:2000]  # bound to prevent oversized entries
            logger.info(f"Tool: MEMO '{text[:60]}'")
            try:
                self.long_term.store(text, memory_type="explicit_memory")
                self.self_model.record_capability_use("memory_management")
                results.append(f"\n**[Memo saved]** {text[:100]}")
            except Exception as e:
                results.append(f"\n**[Memo error]** {e}")

        return "".join(results)

    # ------------------------------------------------------------------
    # Auto-debug code execution — retries with LLM-assisted fixes
    # ------------------------------------------------------------------

    def _execute_with_debug(self, code: str) -> dict:
        """
        Run code; if it fails, ask the LLM to fix it and retry up to
        CODE_DEBUG_RETRIES times.
        """
        result = self.code_executor.run(code)
        if result["success"] or not config.CODE_EXEC_ENABLED:
            return result

        max_retries = config.CODE_DEBUG_RETRIES
        for attempt in range(1, max_retries + 1):
            error = result.get("stderr") or result.get("error", "unknown error")
            logger.info(f"Code failed (attempt {attempt}): {error[:120]} — asking LLM to fix…")

            fix_prompt = (
                f"The following Python code raised an error:\n\n"
                f"```python\n{code}\n```\n\n"
                f"Error:\n{error}\n\n"
                "Fix the code. Return ONLY the corrected Python code inside a ```python block. "
                "No explanation."
            )
            raw = self.llm.fast_generate(fix_prompt, temperature=0.1)
            blocks = self.code_executor.extract_code_blocks(raw)
            if blocks:
                code = blocks[0]["code"]
            else:
                # Strip markdown fences if present
                code = re.sub(r"^```[a-z]*\n?", "", raw.strip(), flags=re.MULTILINE)
                code = re.sub(r"\n?```$", "", code.strip(), flags=re.MULTILINE)

            result = self.code_executor.run(code)
            if result["success"]:
                logger.info(f"Code fixed on attempt {attempt}.")
                self.long_term.store(
                    f"Auto-fixed code after {attempt} attempt(s).",
                    memory_type="code_result",
                )
                return result

        logger.warning(f"Code auto-debug exhausted {max_retries} retries.")
        return result

    # ------------------------------------------------------------------
    # Skill creation with validation
    # ------------------------------------------------------------------

    def _create_skill_with_validation(self, description: str, max_attempts: int = 3) -> Optional[str]:
        """
        Generate a new skill and validate it runs without crashing.
        Retries up to max_attempts times.
        """
        for attempt in range(1, max_attempts + 1):
            name = self.skill_manager.create_skill(description)
            if not name:
                continue
            # Validate: try invoking with no args
            try:
                self.skill_manager.invoke(name)
                logger.info(f"Skill '{name}' validated on attempt {attempt}.")
                return name
            except Exception as e:
                logger.warning(f"Skill '{name}' failed validation: {e} — regenerating…")
                # Remove the broken skill file so it's not re-loaded
                skill_path = self.skill_manager._skills_dir / f"{name}.py"
                skill_path.unlink(missing_ok=True)
                if name in self.skill_manager._loaded:
                    del self.skill_manager._loaded[name]
        return None

    # ------------------------------------------------------------------
    # Plan execution loop
    # ------------------------------------------------------------------

    def execute_plan(self, goal: str, on_step=None) -> Plan:
        """
        Create a plan for goal and execute each step in order.
        on_step(step, result) is called after each step completes (optional callback).
        Returns the completed Plan.
        """
        plan = self.planner.plan(goal, _AVAILABLE_TOOLS)
        plan.status = "running"
        self.self_model.record_capability_use("task_planning")

        # Critic reviews the plan before execution
        review = self.critic.critique_plan(goal, plan.summary())
        if not review.get("viable", True):
            risks = "; ".join(review.get("risks", []))
            logger.warning(f"Plan flagged as unviable: {risks}")

        context_so_far: list[str] = []

        while not plan.all_done():
            step = plan.next_step()
            if step is None:
                break

            step.status = "running"
            logger.info(f"Executing step {step.index}: [{step.tool}] {step.description}")

            result = self._execute_step(step, context="\n".join(context_so_far))
            step.result = result
            step.status = "done" if result and "Error" not in result[:20] else "failed"

            context_so_far.append(f"Step {step.index} ({step.tool}): {result[:300]}")

            if on_step:
                on_step(step, result)

            # Store step result in memory
            self.long_term.store(
                f"Plan step '{step.description}' → {result[:400]}",
                memory_type="plan_step",
            )

        plan.status = "done" if all(s.status == "done" for s in plan.steps) else "failed"
        return plan

    def _execute_step(self, step: Step, context: str = "") -> str:
        """Dispatch a single plan step to the right tool."""
        tool = step.tool
        desc = step.description

        if tool == "web_search":
            try:
                return self.web_search.search_text(desc)
            except Exception as e:
                return f"Error: {e}"

        elif tool == "code":
            # Ask LLM to write code for this step given context
            code_prompt = (
                f"Write Python code to accomplish this task:\n{desc}\n\n"
                f"Context from previous steps:\n{context or 'None'}\n\n"
                "Return ONLY a ```python block. No explanation."
            )
            raw = self.llm.generate(code_prompt, temperature=0.2)
            blocks = self.code_executor.extract_code_blocks(raw)
            if not blocks:
                return "Error: LLM did not return a code block."
            result = self._execute_with_debug(blocks[0]["code"])
            if result["success"]:
                return result["stdout"] or "(no output)"
            return f"Error: {result.get('error') or result.get('stderr')}"

        elif tool == "file":
            args = step.args
            op = args.get("op", "read")
            path = args.get("path", "")
            try:
                if op == "read":
                    return self.file_manager.read(path)
                elif op == "list":
                    return "\n".join(self.file_manager.list(path or "."))
                else:
                    return f"Unsupported file op in plan step: {op}"
            except Exception as e:
                return f"Error: {e}"

        elif tool == "skill":
            skill_name = step.args.get("skill_name", desc.split()[0])
            try:
                return str(self.skill_manager.invoke(skill_name))
            except Exception as e:
                return f"Error: {e}"

        else:  # "conversation" — ask the LLM to handle this step
            prompt = (
                f"Complete the following task step:\n{desc}\n\n"
                f"Context from previous steps:\n{context or 'None'}\n\n"
                "Give a concise, actionable answer."
            )
            return self.llm.generate(prompt, temperature=0.4)

    # ------------------------------------------------------------------
    # Context window compression — prevents silent context loss
    # ------------------------------------------------------------------

    def _maybe_compress_context(self) -> None:
        """
        If short-term memory is within 4 messages of its window limit, summarise
        the oldest half of turns into a compact system message and evict them.
        This prevents the ring-buffer from silently dropping important early context.
        """
        if not config.CONTEXT_COMPRESS_ENABLED:
            return
        if len(self.short_term) < config.MEMORY_MAX_SHORT_TERM - config.CONTEXT_COMPRESS_THRESHOLD_OFFSET:
            return

        msgs = self.short_term.get_messages()
        half = max(1, len(msgs) // 2)
        old_msgs = msgs[:half]
        recent_msgs = msgs[half:]

        transcript = "\n".join(
            f"[{m.role.upper()}] {m.content[:300]}" for m in old_msgs
        )
        prompt = (
            "Summarise the following conversation excerpt into 2-4 concise bullet points "
            "capturing the most important context, facts, and decisions. "
            "Reply with ONLY the bullet points, no preamble.\n\n"
            f"{transcript}"
        )
        try:
            summary = self.llm.fast_generate(prompt, temperature=0.1).strip()
            self.short_term.clear()
            self.short_term.add("system", f"[Earlier context summary]\n{summary}")
            for msg in recent_msgs:
                self.short_term.add(msg.role, msg.content, msg.metadata)
            logger.info(
                f"Context compressed: {half} turns summarised. "
                f"Buffer: {len(msgs)} → {1 + len(recent_msgs)} messages."
            )
        except Exception as e:
            logger.warning(f"Context compression failed: {e}")

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

        goals = self.self_model.get("current_goals") or []
        if goals:
            capped = [g[:_GOALS_MAX_GOAL_LENGTH] for g in goals[:_GOALS_MAX_COUNT]]
            parts.append("\n## Current goals\n" + "\n".join(f"- {g}" for g in capped))

        parts.append(
            "\n## Inline tool calls\n"
            "You may embed tool calls directly in your response using these tags:\n"
            "  [[SEARCH: <query>]]          — search the web\n"
            "  [[CODE: <python code>]]      — execute Python (auto-debugged on error)\n"
            "  [[FILE: read|<path>]]        — read a file from workspace\n"
            "  [[FILE: write|<path>|<text>]] — write a file to workspace\n"
            "  [[FILE: list|<dir>]]         — list files in workspace directory\n"
            "  [[SKILL: <name>|<arg>...]]   — invoke a loaded skill\n"
            "  [[CREATE_SKILL: <desc>]]     — generate and save a new skill\n"
            "  [[PLAN: <goal>]]             — decompose a goal into an ordered plan\n"
            "  [[MEMO: <text>]]             — explicitly save a fact to long-term memory\n"
            "Results will be appended automatically after your response."
        )

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Direct tool access methods (for CLI / agentic use)
    # ------------------------------------------------------------------

    def execute_code(self, code: str) -> dict:
        self.self_model.record_capability_use("code_execution")
        return self._execute_with_debug(code)

    def search_web(self, query: str) -> str:
        self.self_model.record_capability_use("web_search")
        return self.web_search.search_text(query)

    def make_plan(self, goal: str) -> Plan:
        self.self_model.record_capability_use("task_planning")
        return self.planner.plan(goal, _AVAILABLE_TOOLS)

    def manage_file(self, op: str, path: str, content: str = "") -> str:
        """Direct file manager access for CLI commands."""
        self.self_model.record_capability_use("file_management")
        op = op.lower()
        if op == "read":
            return self.file_manager.read(path)
        elif op == "write":
            self.file_manager.write(path, content)
            return f"Written: {path}"
        elif op == "append":
            self.file_manager.append(path, content)
            return f"Appended: {path}"
        elif op == "list":
            items = self.file_manager.list(path or ".")
            return "\n".join(items) if items else "(empty)"
        elif op == "delete":
            self.file_manager.delete(path)
            return f"Deleted: {path}"
        else:
            return f"Unknown file op: {op}"

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
