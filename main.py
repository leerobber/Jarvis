#!/usr/bin/env python3
"""
Jarvis — backwards-compatible entry point.

Usage:
  python main.py                  # Interactive CLI chat
  python main.py --model mistral  # Override Ollama model
  python main.py --no-stream      # Disable streaming output
  python main.py --voice          # Enable voice mode

For the full CLI (including jarvis doctor), prefer:
  pip install -e .
  jarvis --help
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).parent))

from jarvis.__main__ import main

if __name__ == "__main__":
    main()
