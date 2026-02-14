"""Tests for ring1.tools.message."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ring1.tools.message import make_message_tool


class TestMessageTool:
    def test_send_success(self):
        reply_fn = MagicMock()
        tool = make_message_tool(reply_fn)
        result = tool.execute({"text": "Hello user"})
        assert result == "Message sent"
        reply_fn.assert_called_once_with("Hello user")

    def test_send_failure(self):
        reply_fn = MagicMock(side_effect=RuntimeError("network error"))
        tool = make_message_tool(reply_fn)
        result = tool.execute({"text": "test"})
        assert "Error" in result
        assert "network error" in result

    def test_schema(self):
        tool = make_message_tool(MagicMock())
        assert tool.name == "message"
        assert "text" in tool.input_schema["properties"]
        assert "text" in tool.input_schema["required"]
