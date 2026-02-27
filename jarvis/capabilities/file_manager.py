"""
File Manager — safe, sandboxed file operations for Jarvis.
All paths are resolved and validated against an allowed root.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_ROOT = Path.home() / "jarvis_workspace"


class FileManager:
    def __init__(self, root: Path = _DEFAULT_ROOT):
        self._root = root.resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def _safe_path(self, relative: str) -> Path:
        resolved = (self._root / relative).resolve()
        if not str(resolved).startswith(str(self._root)):
            raise PermissionError(f"Path escape attempt blocked: {relative}")
        return resolved

    # ------------------------------------------------------------------
    # Read / Write / Append
    # ------------------------------------------------------------------

    def read(self, path: str) -> str:
        p = self._safe_path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return p.read_text()

    def write(self, path: str, content: str) -> None:
        p = self._safe_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        logger.info(f"Wrote file: {p}")

    def append(self, path: str, content: str) -> None:
        p = self._safe_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            f.write(content)

    # ------------------------------------------------------------------
    # List / Delete / Move
    # ------------------------------------------------------------------

    def list(self, directory: str = ".") -> list[str]:
        p = self._safe_path(directory)
        if not p.is_dir():
            return []
        return [str(item.relative_to(self._root)) for item in p.iterdir()]

    def delete(self, path: str) -> None:
        p = self._safe_path(path)
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink(missing_ok=True)
        logger.info(f"Deleted: {p}")

    def move(self, src: str, dst: str) -> None:
        s, d = self._safe_path(src), self._safe_path(dst)
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(s), str(d))

    def exists(self, path: str) -> bool:
        return self._safe_path(path).exists()

    @property
    def workspace(self) -> str:
        return str(self._root)
