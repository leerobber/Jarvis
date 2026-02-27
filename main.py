#!/usr/bin/env python3
"""
Jarvis — Advanced Intelligent Self-Learning AI Assistant
Entry point.

Usage:
  python main.py                  # Interactive CLI chat
  python main.py --model mistral  # Override Ollama model
  python main.py --no-stream      # Disable streaming output
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console

console = Console()


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jarvis — Self-learning AI assistant")
    parser.add_argument("--model", help="Ollama model to use (overrides .env)")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--voice", action="store_true", help="Enable voice mode at startup")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Apply overrides before importing config-dependent modules
    if args.model:
        os.environ["OLLAMA_MODEL"] = args.model
    if args.voice:
        os.environ["VOICE_ENABLED"] = "true"

    setup_logging(args.log_level)

    from jarvis.config import config
    from jarvis.core.brain import Brain
    from jarvis.interface.cli import CLI

    # Verify Ollama is available
    console.print("[dim]Connecting to Ollama…[/dim]")
    from jarvis.core.ollama_client import OllamaClient
    client = OllamaClient()
    if not client.is_available():
        console.print(
            f"[bold red]✗ Cannot reach Ollama at {config.OLLAMA_HOST}[/bold red]\n"
            "[yellow]Make sure Ollama is running:  ollama serve[/yellow]\n"
            f"[yellow]And the model is pulled:      ollama pull {config.OLLAMA_MODEL}[/yellow]"
        )
        sys.exit(1)

    console.print(f"[green]✓ Ollama connected — model: {config.OLLAMA_MODEL}[/green]")
    client.ensure_model()

    # Boot the brain
    brain = Brain()

    # Launch CLI
    cli = CLI(brain)
    if args.voice:
        cli._voice_mode = True
    cli.run()


if __name__ == "__main__":
    main()
