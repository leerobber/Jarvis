"""
jarvis/__main__.py — package entry point.

Supports:
  jarvis                  # interactive CLI chat
  jarvis doctor           # verify environment (Ollama, model, data dirs)
  jarvis --model mistral  # override Ollama model
  jarvis --no-stream      # disable streaming output
  jarvis --voice          # enable voice mode at startup
  jarvis --log-level DEBUG
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from rich.console import Console

console = Console()


# ---------------------------------------------------------------------------
# Helpers shared with main.py
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    from jarvis.config import config
    log_path = Path(config.LOG_PATH)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stderr) if level == "DEBUG" else logging.NullHandler(),
        ],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Jarvis — Self-learning AI assistant")
    parser.add_argument("--model", help="Ollama model to use (overrides .env / OLLAMA_MODEL)")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    parser.add_argument("--voice", action="store_true", help="Enable voice mode at startup")

    sub = parser.add_subparsers(dest="command")
    sub.add_parser("doctor", help="Check environment: Ollama reachability, model presence, data dirs")

    return parser


# ---------------------------------------------------------------------------
# Doctor command
# ---------------------------------------------------------------------------

def run_doctor() -> int:
    """Verify the runtime environment.  Returns 0 on success, 1 on any failure."""
    from jarvis.config import config

    ok = True

    # 1. Ollama reachability
    console.print("[bold]Checking Ollama…[/bold]")
    try:
        from jarvis.core.ollama_client import OllamaClient
        client = OllamaClient()
        if client.is_available():
            console.print(f"  [green]✓[/green] Ollama reachable at {config.OLLAMA_HOST}")
        else:
            console.print(
                f"  [red]✗[/red] Cannot reach Ollama at {config.OLLAMA_HOST}\n"
                "    [yellow]Fix:[/yellow] run  ollama serve"
            )
            ok = False
            # Skip model check — Ollama is down
            return 1
    except Exception as exc:
        console.print(f"  [red]✗[/red] Ollama client error: {exc}")
        return 1

    # 2. Model presence
    console.print("[bold]Checking model…[/bold]")
    try:
        import ollama as _ollama
        raw_client = _ollama.Client(host=config.OLLAMA_HOST)
        available = [m.model for m in raw_client.list().models]
        if any(config.OLLAMA_MODEL in m for m in available):
            console.print(f"  [green]✓[/green] Model '{config.OLLAMA_MODEL}' is present")
        else:
            console.print(
                f"  [red]✗[/red] Model '{config.OLLAMA_MODEL}' not found\n"
                f"    [yellow]Fix:[/yellow] run  ollama pull {config.OLLAMA_MODEL}"
            )
            ok = False
    except Exception as exc:
        console.print(f"  [red]✗[/red] Could not list models: {exc}")
        ok = False

    # 3. Data directories / log directory
    console.print("[bold]Checking data directories…[/bold]")
    dirs_to_check = {
        "log dir": Path(config.LOG_PATH).parent,
        "memory dir": Path(config.MEMORY_CHROMA_PATH),
        "skills dir": Path(config.SKILLS_PATH),
    }
    for label, path in dirs_to_check.items():
        try:
            path.mkdir(parents=True, exist_ok=True)
            # Verify we can actually write there
            test_file = path / ".jarvis_write_test"
            test_file.touch()
            test_file.unlink()
            console.print(f"  [green]✓[/green] {label}: {path}")
        except Exception as exc:
            console.print(
                f"  [red]✗[/red] {label} not writable ({path}): {exc}\n"
                f"    [yellow]Fix:[/yellow] ensure the process has write access to {path}"
            )
            ok = False

    if ok:
        console.print("\n[bold green]All checks passed — Jarvis is ready to run.[/bold green]")
        return 0
    else:
        console.print("\n[bold red]Some checks failed — see above for guidance.[/bold red]")
        return 1


# ---------------------------------------------------------------------------
# Interactive chat
# ---------------------------------------------------------------------------

def run_chat(args: argparse.Namespace) -> None:
    setup_logging(args.log_level)

    from jarvis.config import config
    from jarvis.core.brain import Brain
    from jarvis.interface.cli import CLI

    console.print("[dim]Connecting to Ollama…[/dim]")
    from jarvis.core.ollama_client import OllamaClient
    client = OllamaClient()
    if not client.is_available():
        console.print(
            f"[bold red]✗ Cannot reach Ollama at {config.OLLAMA_HOST}[/bold red]\n"
            "[yellow]Make sure Ollama is running:  ollama serve[/yellow]\n"
            f"[yellow]And the model is pulled:      ollama pull {config.OLLAMA_MODEL}[/yellow]\n"
            "[dim]Tip: run  jarvis doctor  for a full environment check[/dim]"
        )
        sys.exit(1)

    console.print(f"[green]✓ Ollama connected — model: {config.OLLAMA_MODEL}[/green]")
    client.ensure_model()

    brain = Brain()
    cli = CLI(brain)
    if args.voice:
        cli._voice_mode = True
    cli.run()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Apply env-var overrides before any config-dependent imports
    if args.model:
        os.environ["OLLAMA_MODEL"] = args.model
    if args.voice:
        os.environ["VOICE_ENABLED"] = "true"

    if args.command == "doctor":
        sys.exit(run_doctor())
    else:
        run_chat(args)


if __name__ == "__main__":
    main()
