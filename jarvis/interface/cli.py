"""
CLI Interface — Rich terminal UI for Jarvis.
Supports text chat, voice mode, direct tool commands, and status display.
"""
from __future__ import annotations

import logging
import signal
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint

console = Console()
logger = logging.getLogger(__name__)

_COMMANDS = {
    "/help":    "Show this help message",
    "/status":  "Show Jarvis internal state and self-model stats",
    "/memory":  "Show recent long-term memory entries",
    "/skills":  "List all learned skills",
    "/search":  "Search the web: /search <query>",
    "/run":     "Run Python code: /run <code>",
    "/plan":    "Plan a task: /plan <goal>",
    "/reflect": "Trigger a manual reflection cycle now",
    "/voice":   "Toggle voice input mode",
    "/clear":   "Clear short-term conversation memory",
    "/save":    "Save a note to memory: /save <text>",
    "/quit":    "Exit Jarvis",
}


class CLI:
    def __init__(self, brain):
        self._brain = brain
        self._voice_mode = False
        self._running = True
        signal.signal(signal.SIGINT, self._handle_interrupt)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._print_banner()
        while self._running:
            try:
                user_input = self._get_input()
                if not user_input:
                    continue
                if user_input.startswith("/"):
                    self._handle_command(user_input)
                else:
                    self._chat(user_input)
            except EOFError:
                self._quit()

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def _get_input(self) -> str:
        if self._voice_mode and self._brain.voice.available:
            console.print("[bold cyan]🎤 Listening…[/bold cyan]")
            text = self._brain.listen()
            if text:
                console.print(f"[dim]You said:[/dim] {text}")
                return text
            return ""
        try:
            return console.input("[bold green]You>[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            return "/quit"

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def _chat(self, user_input: str) -> None:
        console.print()
        console.print("[bold blue]Jarvis>[/bold blue] ", end="")
        full_response = []
        try:
            for token in self._brain.chat(user_input, stream=True):
                console.print(token, end="", highlight=False)
                full_response.append(token)
            console.print()

            if self._voice_mode and self._brain.voice.available:
                self._brain.speak("".join(full_response))

        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            logger.error(f"Chat error: {e}", exc_info=True)
        console.print()

    # ------------------------------------------------------------------
    # Command dispatcher
    # ------------------------------------------------------------------

    def _handle_command(self, raw: str) -> None:
        parts = raw.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        handlers = {
            "/help":    lambda: self._cmd_help(),
            "/status":  lambda: self._cmd_status(),
            "/memory":  lambda: self._cmd_memory(),
            "/skills":  lambda: self._cmd_skills(),
            "/search":  lambda: self._cmd_search(args),
            "/run":     lambda: self._cmd_run(args),
            "/plan":    lambda: self._cmd_plan(args),
            "/reflect": lambda: self._cmd_reflect(),
            "/voice":   lambda: self._cmd_toggle_voice(),
            "/clear":   lambda: self._cmd_clear(),
            "/save":    lambda: self._cmd_save(args),
            "/quit":    lambda: self._quit(),
            "/exit":    lambda: self._quit(),
        }
        handler = handlers.get(cmd)
        if handler:
            handler()
        else:
            console.print(f"[yellow]Unknown command: {cmd}. Type /help for a list.[/yellow]")

    # ------------------------------------------------------------------
    # Command implementations
    # ------------------------------------------------------------------

    def _cmd_help(self) -> None:
        lines = ["[bold]Available commands:[/bold]"]
        for cmd, desc in _COMMANDS.items():
            lines.append(f"  [cyan]{cmd:<12}[/cyan] {desc}")
        console.print("\n".join(lines))
        console.print()

    def _cmd_status(self) -> None:
        brain = self._brain
        state = brain.introspection.get_state()
        sm = brain.self_model

        panel = Panel(
            Text.assemble(
                ("Self-Model\n", "bold white"),
                (sm.describe_self(), "dim"),
                ("\n\nInternal State\n", "bold white"),
                (state.summary(), "cyan"),
                (f"\n\nLong-term memories stored: ", "white"),
                (str(brain.long_term.count()), "green"),
                (f"\n  Skills loaded: ", "white"),
                (str(len(brain.skill_manager.list_skills())), "green"),
                (f"\n  Voice mode: ", "white"),
                ("ON" if self._voice_mode else "OFF", "green" if self._voice_mode else "red"),
            ),
            title="[bold blue]Jarvis Status[/bold blue]",
            border_style="blue",
        )
        console.print(panel)

    def _cmd_memory(self) -> None:
        memories = self._brain.long_term.retrieve("recent interactions", n_results=8)
        if not memories:
            console.print("[dim]No memories yet.[/dim]")
            return
        for m in memories:
            tag = m["metadata"].get("type", "?")
            score = m["relevance"]
            console.print(f"[cyan][{tag}][/cyan] ({score:.2f}) {m['text'][:120]}")
        console.print()

    def _cmd_skills(self) -> None:
        desc = self._brain.skill_manager.describe_skills()
        console.print(f"[bold]Skills:[/bold]\n{desc}\n")

    def _cmd_search(self, query: str) -> None:
        if not query:
            console.print("[yellow]Usage: /search <query>[/yellow]")
            return
        console.print(f"[dim]Searching: {query}…[/dim]")
        result = self._brain.search_web(query)
        console.print(Markdown(f"**Web search results:**\n\n{result}"))
        console.print()

    def _cmd_run(self, code: str) -> None:
        if not code:
            console.print("[yellow]Usage: /run <python code>[/yellow]")
            return
        result = self._brain.execute_code(code)
        if result["success"]:
            console.print(f"[green]✓ Output:[/green]\n{result['stdout']}")
        else:
            console.print(f"[red]✗ Error:[/red] {result['error'] or result['stderr']}")
        console.print()

    def _cmd_plan(self, goal: str) -> None:
        if not goal:
            console.print("[yellow]Usage: /plan <goal>[/yellow]")
            return
        plan = self._brain.make_plan(goal)
        review = self._brain.critic.critique_plan(goal, plan.summary())
        console.print(Panel(plan.summary(), title="[bold blue]Plan[/bold blue]", border_style="blue"))
        if review.get("risks"):
            console.print(f"[yellow]Risks:[/yellow] {'; '.join(review['risks'])}")
        console.print()

    def _cmd_reflect(self) -> None:
        console.print("[dim]Running reflection cycle…[/dim]")
        summary = self._brain.short_term.summary_text()
        if not summary:
            console.print("[yellow]Nothing to reflect on yet.[/yellow]")
            return
        report = self._brain.reflector.reflect(summary)
        quality = report.get("overall_quality", 0)
        well = report.get("what_went_well", [])
        poor = report.get("what_went_poorly", [])
        directives = report.get("behavioral_directives", [])
        console.print(f"[bold]Reflection Report[/bold] (quality: {quality:.0%})")
        if well:
            console.print(f"[green]+ {'; '.join(well[:3])}[/green]")
        if poor:
            console.print(f"[red]- {'; '.join(poor[:3])}[/red]")
        if directives:
            console.print(f"[cyan]New directives: {len(directives)} added to meta-prompt[/cyan]")
        console.print()

    def _cmd_toggle_voice(self) -> None:
        if not self._brain.voice.available:
            console.print("[red]Voice not available — check VOICE_ENABLED and audio hardware.[/red]")
            return
        self._voice_mode = not self._voice_mode
        state = "ON" if self._voice_mode else "OFF"
        console.print(f"[cyan]Voice mode: {state}[/cyan]")

    def _cmd_clear(self) -> None:
        self._brain.short_term.clear()
        console.print("[dim]Short-term memory cleared.[/dim]")

    def _cmd_save(self, text: str) -> None:
        if not text:
            console.print("[yellow]Usage: /save <note text>[/yellow]")
            return
        self._brain.long_term.store(text, memory_type="user_note")
        console.print("[green]Saved to memory.[/green]")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _quit(self) -> None:
        console.print("\n[dim]Running shutdown reflection…[/dim]")
        self._brain.shutdown()
        console.print("[bold blue]Jarvis offline. Goodbye.[/bold blue]")
        self._running = False
        sys.exit(0)

    def _handle_interrupt(self, sig, frame) -> None:
        console.print("\n[yellow]Use /quit to exit properly.[/yellow]")

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------

    def _print_banner(self) -> None:
        banner = """
     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
     ██║███████║██████╔╝██║   ██║██║███████╗
██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║
 ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝
        """
        console.print(f"[bold blue]{banner}[/bold blue]")
        sm = self._brain.self_model
        name = sm.get("identity", "name")
        version = sm.get("identity", "version")
        interactions = sm.get("growth", "total_interactions")
        memories = self._brain.long_term.count()
        skills = len(self._brain.skill_manager.list_skills())
        console.print(
            Panel(
                f"[bold]{name}[/bold] v{version}  |  "
                f"Memories: [cyan]{memories}[/cyan]  |  "
                f"Skills: [cyan]{skills}[/cyan]  |  "
                f"Interactions: [cyan]{interactions}[/cyan]\n"
                f"[dim]Type [cyan]/help[/cyan] for commands. Type naturally to chat.[/dim]",
                border_style="blue",
            )
        )
        console.print()
