"""Tests for ring1.tools.filesystem."""

from __future__ import annotations

import os

import pytest

from ring1.tools.filesystem import _resolve_safe, make_filesystem_tools


@pytest.fixture
def workspace(tmp_path):
    """Create a workspace directory with sample files."""
    (tmp_path / "hello.txt").write_text("line1\nline2\nline3\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "nested.txt").write_text("nested content\n")
    return tmp_path


@pytest.fixture
def tools(workspace):
    """Create filesystem tools bound to the workspace."""
    tool_list = make_filesystem_tools(str(workspace))
    return {t.name: t for t in tool_list}


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------

class TestResolveSafe:
    def test_normal_path(self, workspace):
        result = _resolve_safe(workspace, "hello.txt")
        assert result == workspace / "hello.txt"

    def test_subdir_path(self, workspace):
        result = _resolve_safe(workspace, "sub/nested.txt")
        assert result == workspace / "sub" / "nested.txt"

    def test_dotdot_escape_blocked(self, workspace):
        with pytest.raises(ValueError, match="escapes workspace"):
            _resolve_safe(workspace, "../../../etc/passwd")

    def test_absolute_path_outside_blocked(self, workspace):
        with pytest.raises(ValueError, match="escapes workspace"):
            _resolve_safe(workspace, "/etc/passwd")

    def test_dot_path(self, workspace):
        result = _resolve_safe(workspace, ".")
        assert result == workspace.resolve()


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

class TestReadFile:
    def test_read_basic(self, tools):
        result = tools["read_file"].execute({"path": "hello.txt"})
        assert "line1" in result
        assert "line2" in result
        assert "line3" in result

    def test_read_with_line_numbers(self, tools):
        result = tools["read_file"].execute({"path": "hello.txt"})
        assert "\t" in result  # line number tab separator

    def test_read_with_offset_limit(self, tools):
        result = tools["read_file"].execute({"path": "hello.txt", "offset": 1, "limit": 1})
        assert "line2" in result
        assert "line1" not in result
        assert "line3" not in result

    def test_read_nonexistent(self, tools):
        result = tools["read_file"].execute({"path": "nope.txt"})
        assert "Error" in result

    def test_read_escape_blocked(self, tools):
        result = tools["read_file"].execute({"path": "../../../etc/passwd"})
        assert "Error" in result
        assert "escapes" in result

    def test_read_nested(self, tools):
        result = tools["read_file"].execute({"path": "sub/nested.txt"})
        assert "nested content" in result


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------

class TestWriteFile:
    def test_write_new_file(self, tools, workspace):
        result = tools["write_file"].execute({"path": "new.txt", "content": "hello"})
        assert "Written" in result
        assert (workspace / "new.txt").read_text() == "hello"

    def test_write_creates_parent_dirs(self, tools, workspace):
        result = tools["write_file"].execute(
            {"path": "deep/nested/dir/file.txt", "content": "data"}
        )
        assert "Written" in result
        assert (workspace / "deep" / "nested" / "dir" / "file.txt").read_text() == "data"

    def test_write_overwrites(self, tools, workspace):
        tools["write_file"].execute({"path": "hello.txt", "content": "new content"})
        assert (workspace / "hello.txt").read_text() == "new content"

    def test_write_escape_blocked(self, tools):
        result = tools["write_file"].execute(
            {"path": "../../../tmp/evil.txt", "content": "bad"}
        )
        assert "Error" in result
        assert "escapes" in result


# ---------------------------------------------------------------------------
# edit_file
# ---------------------------------------------------------------------------

class TestEditFile:
    def test_edit_basic(self, tools, workspace):
        result = tools["edit_file"].execute({
            "path": "hello.txt",
            "old_string": "line2",
            "new_string": "LINE_TWO",
        })
        assert "successfully" in result
        content = (workspace / "hello.txt").read_text()
        assert "LINE_TWO" in content
        assert "line2" not in content

    def test_edit_not_found(self, tools):
        result = tools["edit_file"].execute({
            "path": "hello.txt",
            "old_string": "nonexistent text",
            "new_string": "replacement",
        })
        assert "not found" in result

    def test_edit_multiple_matches(self, tools, workspace):
        (workspace / "dup.txt").write_text("aaa\naaa\n")
        result = tools["edit_file"].execute({
            "path": "dup.txt",
            "old_string": "aaa",
            "new_string": "bbb",
        })
        assert "matches 2 times" in result

    def test_edit_nonexistent_file(self, tools):
        result = tools["edit_file"].execute({
            "path": "nope.txt",
            "old_string": "x",
            "new_string": "y",
        })
        assert "Error" in result

    def test_edit_escape_blocked(self, tools):
        result = tools["edit_file"].execute({
            "path": "../../../etc/hosts",
            "old_string": "localhost",
            "new_string": "evil",
        })
        assert "Error" in result
        assert "escapes" in result


# ---------------------------------------------------------------------------
# list_dir
# ---------------------------------------------------------------------------

class TestListDir:
    def test_list_root(self, tools, workspace):
        result = tools["list_dir"].execute({})
        assert "hello.txt" in result
        assert "sub/" in result

    def test_list_subdir(self, tools):
        result = tools["list_dir"].execute({"path": "sub"})
        assert "nested.txt" in result

    def test_list_nonexistent(self, tools):
        result = tools["list_dir"].execute({"path": "nope"})
        assert "Error" in result

    def test_list_escape_blocked(self, tools):
        result = tools["list_dir"].execute({"path": "../../../"})
        assert "Error" in result

    def test_list_empty_dir(self, tools, workspace):
        (workspace / "empty").mkdir()
        result = tools["list_dir"].execute({"path": "empty"})
        assert "empty" in result.lower()

    def test_dirs_sorted_first(self, tools, workspace):
        """Directories should appear before files."""
        result = tools["list_dir"].execute({})
        lines = result.strip().splitlines()
        dir_indices = [i for i, l in enumerate(lines) if l.endswith("/")]
        file_indices = [i for i, l in enumerate(lines) if not l.endswith("/")]
        if dir_indices and file_indices:
            assert max(dir_indices) < min(file_indices)
