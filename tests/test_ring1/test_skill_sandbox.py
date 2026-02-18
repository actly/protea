"""Tests for ring1.skill_sandbox — VenvManager."""

from __future__ import annotations

import sys
from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest

from ring1.skill_sandbox import VenvManager


class TestEnsureEnv:
    """ensure_env() should create venvs and install dependencies."""

    def test_creates_env_dir(self, tmp_path):
        """Verify the env directory is created (mocking venv.create and pip)."""
        mgr = VenvManager(tmp_path / "envs")
        env_dir = tmp_path / "envs" / "test_skill"

        def fake_create(path, **kwargs):
            Path(path).mkdir(parents=True, exist_ok=True)

        with patch("ring1.skill_sandbox.venv.create", side_effect=fake_create) as mock_create, \
             patch("ring1.skill_sandbox.subprocess.run") as mock_run:
            python = mgr.ensure_env("test_skill", ["requests"])
            assert mock_create.called
            mock_create.assert_called_once_with(str(env_dir), with_pip=True)
            # pip install should be called.
            mock_run.assert_called_once()
            args = mock_run.call_args
            assert "pip" in str(args)
            assert "requests" in args[0][0]

    def test_reuses_env_when_deps_unchanged(self, tmp_path):
        """If deps hash matches, should NOT recreate."""
        mgr = VenvManager(tmp_path / "envs")
        env_dir = tmp_path / "envs" / "test_skill"
        env_dir.mkdir(parents=True)

        import hashlib
        deps_hash = hashlib.md5(",".join(sorted(["requests"])).encode()).hexdigest()
        (env_dir / ".deps_hash").write_text(deps_hash)

        # Create the expected python path so it can be returned.
        if sys.platform == "win32":
            python_path = env_dir / "Scripts" / "python.exe"
        else:
            python_path = env_dir / "bin" / "python"
        python_path.parent.mkdir(parents=True, exist_ok=True)
        python_path.touch()

        with patch("ring1.skill_sandbox.venv.create") as mock_create:
            result = mgr.ensure_env("test_skill", ["requests"])
            # Should NOT recreate.
            mock_create.assert_not_called()
            assert result == python_path

    def test_recreates_env_when_deps_change(self, tmp_path):
        """If deps change, should recreate from scratch."""
        mgr = VenvManager(tmp_path / "envs")
        env_dir = tmp_path / "envs" / "test_skill"
        env_dir.mkdir(parents=True)
        (env_dir / ".deps_hash").write_text("old_hash")

        def fake_create(path, **kwargs):
            Path(path).mkdir(parents=True, exist_ok=True)

        with patch("ring1.skill_sandbox.venv.create", side_effect=fake_create) as mock_create, \
             patch("ring1.skill_sandbox.subprocess.run"):
            mgr.ensure_env("test_skill", ["requests", "pandas"])
            mock_create.assert_called_once()


class TestRemoveEnv:
    """remove_env() should clean up a skill's venv."""

    def test_remove_existing(self, tmp_path):
        mgr = VenvManager(tmp_path / "envs")
        env_dir = tmp_path / "envs" / "test_skill"
        env_dir.mkdir(parents=True)
        (env_dir / "dummy").touch()
        assert mgr.remove_env("test_skill") is True
        assert not env_dir.exists()

    def test_remove_nonexistent(self, tmp_path):
        mgr = VenvManager(tmp_path / "envs")
        assert mgr.remove_env("nonexistent") is False


class TestListEnvs:
    """list_envs() should list active environments."""

    def test_empty(self, tmp_path):
        mgr = VenvManager(tmp_path / "envs")
        assert mgr.list_envs() == []

    def test_lists_envs_with_marker(self, tmp_path):
        mgr = VenvManager(tmp_path / "envs")
        env_dir = tmp_path / "envs" / "my_skill"
        env_dir.mkdir(parents=True)
        (env_dir / ".deps_hash").write_text("abc")
        (env_dir / "file.txt").write_text("data")

        envs = mgr.list_envs()
        assert len(envs) == 1
        assert envs[0]["name"] == "my_skill"
        assert envs[0]["size_bytes"] > 0

    def test_ignores_dirs_without_marker(self, tmp_path):
        mgr = VenvManager(tmp_path / "envs")
        (tmp_path / "envs" / "orphan").mkdir(parents=True)
        assert mgr.list_envs() == []


class TestPythonPath:
    """_python_path should return platform-appropriate path."""

    def test_unix_path(self, tmp_path):
        mgr = VenvManager(tmp_path / "envs")
        env_dir = tmp_path / "envs" / "skill"
        with patch("ring1.skill_sandbox.sys") as mock_sys:
            mock_sys.platform = "darwin"
            path = mgr._python_path(env_dir)
            assert path == env_dir / "bin" / "python"

    def test_windows_path(self, tmp_path):
        mgr = VenvManager(tmp_path / "envs")
        env_dir = tmp_path / "envs" / "skill"
        with patch("ring1.skill_sandbox.sys") as mock_sys:
            mock_sys.platform = "win32"
            path = mgr._python_path(env_dir)
            assert path == env_dir / "Scripts" / "python.exe"
