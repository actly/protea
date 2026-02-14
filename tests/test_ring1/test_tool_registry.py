"""Tests for ring1.tool_registry."""

from __future__ import annotations

import pytest

from ring1.tool_registry import Tool, ToolRegistry


def _make_tool(name: str = "echo", result: str = "ok") -> Tool:
    return Tool(
        name=name,
        description=f"Test tool: {name}",
        input_schema={"type": "object", "properties": {}, "required": []},
        execute=lambda inp, _r=result: _r,
    )


class TestToolRegistry:
    def test_register_and_execute(self):
        reg = ToolRegistry()
        reg.register(_make_tool("hello", "world"))
        assert reg.execute("hello", {}) == "world"

    def test_unknown_tool(self):
        reg = ToolRegistry()
        result = reg.execute("nonexistent", {})
        assert "Unknown tool" in result

    def test_get_schemas(self):
        reg = ToolRegistry()
        reg.register(_make_tool("a"))
        reg.register(_make_tool("b"))
        schemas = reg.get_schemas()
        assert len(schemas) == 2
        names = {s["name"] for s in schemas}
        assert names == {"a", "b"}
        for s in schemas:
            assert "description" in s
            assert "input_schema" in s

    def test_unregister(self):
        reg = ToolRegistry()
        reg.register(_make_tool("x"))
        assert len(reg) == 1
        reg.unregister("x")
        assert len(reg) == 0
        assert reg.execute("x", {}) == "Unknown tool: x"

    def test_unregister_nonexistent(self):
        reg = ToolRegistry()
        reg.unregister("nope")  # should not raise

    def test_clone_without(self):
        reg = ToolRegistry()
        reg.register(_make_tool("a", "A"))
        reg.register(_make_tool("b", "B"))
        reg.register(_make_tool("c", "C"))

        cloned = reg.clone_without("b")
        assert len(cloned) == 2
        assert "b" not in cloned.tool_names()
        assert cloned.execute("a", {}) == "A"
        assert cloned.execute("c", {}) == "C"
        assert "Unknown" in cloned.execute("b", {})

    def test_clone_without_multiple(self):
        reg = ToolRegistry()
        reg.register(_make_tool("a"))
        reg.register(_make_tool("b"))
        reg.register(_make_tool("c"))

        cloned = reg.clone_without("a", "c")
        assert cloned.tool_names() == ["b"]

    def test_clone_preserves_original(self):
        reg = ToolRegistry()
        reg.register(_make_tool("a"))
        reg.register(_make_tool("b"))

        cloned = reg.clone_without("a")
        assert len(reg) == 2  # original unchanged
        assert len(cloned) == 1

    def test_overwrite_on_register(self):
        reg = ToolRegistry()
        reg.register(_make_tool("x", "first"))
        reg.register(_make_tool("x", "second"))
        assert reg.execute("x", {}) == "second"
        assert len(reg) == 1

    def test_get_returns_tool(self):
        reg = ToolRegistry()
        tool = _make_tool("t")
        reg.register(tool)
        assert reg.get("t") is tool
        assert reg.get("nope") is None

    def test_tool_names(self):
        reg = ToolRegistry()
        reg.register(_make_tool("alpha"))
        reg.register(_make_tool("beta"))
        assert set(reg.tool_names()) == {"alpha", "beta"}

    def test_execute_passes_input(self):
        captured = {}

        def handler(inp):
            captured.update(inp)
            return "done"

        tool = Tool(
            name="capture",
            description="captures input",
            input_schema={"type": "object", "properties": {}, "required": []},
            execute=handler,
        )
        reg = ToolRegistry()
        reg.register(tool)
        reg.execute("capture", {"key": "value"})
        assert captured == {"key": "value"}
