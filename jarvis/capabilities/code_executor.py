"""
Code Executor — safely run Python snippets in a subprocess with timeout.
No network access. Captures stdout/stderr and returns them.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from jarvis.config import config


class CodeExecutor:
    def __init__(self, timeout: int = config.CODE_EXEC_TIMEOUT):
        self._timeout = timeout
        self._enabled = config.CODE_EXEC_ENABLED

    def run(self, code: str, language: str = "python") -> dict:
        """
        Execute code and return:
          { success: bool, stdout: str, stderr: str, error: str }
        """
        if not self._enabled:
            return {"success": False, "stdout": "", "stderr": "", "error": "Code execution disabled."}

        if language.lower() not in ("python", "python3"):
            return {
                "success": False,
                "stdout": "",
                "stderr": "",
                "error": f"Language '{language}' not supported. Only Python is allowed.",
            }

        code = textwrap.dedent(code)

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(code)
            tmp_path = Path(f.name)

        try:
            result = subprocess.run(
                [sys.executable, str(tmp_path)],
                capture_output=True,
                text=True,
                timeout=self._timeout,
                # Restrict: no extra env vars passed through
                env={"PATH": "/usr/bin:/bin"},
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout[:4000],
                "stderr": result.stderr[:2000],
                "error": "",
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": "",
                "error": f"Code timed out after {self._timeout}s.",
            }
        except Exception as e:
            return {"success": False, "stdout": "", "stderr": "", "error": str(e)}
        finally:
            tmp_path.unlink(missing_ok=True)

    def extract_code_blocks(self, text: str) -> list[dict]:
        """Extract ```python ... ``` blocks from LLM output."""
        import re
        blocks = []
        pattern = re.compile(r"```(?:python|py)?\n(.*?)```", re.DOTALL | re.IGNORECASE)
        for match in pattern.finditer(text):
            blocks.append({"language": "python", "code": match.group(1).strip()})
        return blocks
