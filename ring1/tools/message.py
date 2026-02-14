"""Message tool — lets the LLM proactively send Telegram messages.

Pure stdlib.
"""

from __future__ import annotations

import logging

from ring1.tool_registry import Tool

log = logging.getLogger("protea.tools.message")


def make_message_tool(reply_fn) -> Tool:
    """Create a Tool that sends a Telegram message via *reply_fn*.

    Args:
        reply_fn: Callable(text: str) -> None to send a message.
    """

    def _exec_message(inp: dict) -> str:
        text = inp["text"]
        try:
            reply_fn(text)
        except Exception as exc:
            log.warning("Message tool send failed: %s", exc)
            return f"Error sending message: {exc}"
        return "Message sent"

    return Tool(
        name="message",
        description=(
            "Send a message to the user via Telegram.  Use this during "
            "multi-step tasks to report progress or intermediate results."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The message text to send.",
                },
            },
            "required": ["text"],
        },
        execute=_exec_message,
    )
