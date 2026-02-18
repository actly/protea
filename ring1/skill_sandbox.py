"""VenvManager — per-skill virtual environment management.

Creates isolated venv environments for capability skills that require
pip dependencies.  Reuses existing envs when deps haven't changed.
Pure stdlib — uses venv, subprocess, hashlib, shutil.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
import sys
import venv
from pathlib import Path

log = logging.getLogger("protea.skill_sandbox")


class VenvManager:
    """Manage per-skill virtual environments."""

    def __init__(self, base_dir: Path, max_envs: int = 10) -> None:
        self.base_dir = Path(base_dir)
        self.max_envs = max_envs
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def ensure_env(self, skill_name: str, dependencies: list[str]) -> Path:
        """Create venv (if needed) and install dependencies.

        Returns path to the venv's Python executable.
        Reuses existing venv if deps haven't changed (hash check).
        """
        env_dir = self.base_dir / skill_name
        marker = env_dir / ".deps_hash"
        deps_hash = hashlib.md5(",".join(sorted(dependencies)).encode()).hexdigest()

        if env_dir.exists() and marker.exists() and marker.read_text().strip() == deps_hash:
            return self._python_path(env_dir)

        # Create fresh venv.
        if env_dir.exists():
            shutil.rmtree(env_dir)
        venv.create(str(env_dir), with_pip=True)

        # Install dependencies.
        python = self._python_path(env_dir)
        subprocess.run(
            [str(python), "-m", "pip", "install", "--quiet"] + dependencies,
            check=True,
            timeout=120,
            capture_output=True,
        )
        marker.write_text(deps_hash)
        log.info("Created venv for '%s' with deps: %s", skill_name, dependencies)
        return python

    def remove_env(self, skill_name: str) -> bool:
        """Remove a skill's venv. Returns True if removed."""
        env_dir = self.base_dir / skill_name
        if env_dir.exists():
            shutil.rmtree(env_dir)
            log.info("Removed venv for '%s'", skill_name)
            return True
        return False

    def list_envs(self) -> list[dict]:
        """List active envs with name and disk size."""
        envs = []
        if not self.base_dir.exists():
            return envs
        for child in sorted(self.base_dir.iterdir()):
            if child.is_dir() and (child / ".deps_hash").exists():
                size = sum(f.stat().st_size for f in child.rglob("*") if f.is_file())
                envs.append({"name": child.name, "size_bytes": size})
        return envs

    def _python_path(self, env_dir: Path) -> Path:
        """Return path to Python executable (platform-aware)."""
        if sys.platform == "win32":
            return env_dir / "Scripts" / "python.exe"
        return env_dir / "bin" / "python"
