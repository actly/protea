"""Spawn tool — creates background subagent tasks.

Pure stdlib.
"""

from __future__ import annotations

import logging

from ring1.tool_registry import Tool

log = logging.getLogger("protea.tools.spawn")


def make_spawn_tool(subagent_manager) -> Tool:
    """Create a Tool that spawns a background subagent task.

    Args:
        subagent_manager: SubagentManager instance.
    """

    def _exec_spawn(inp: dict) -> str:
        task_description = inp["task"]
        context = inp.get("context", "")
        return subagent_manager.spawn(task_description, context=context)

    return Tool(
        name="spawn",
        description=(
            "Start a background task that runs autonomously.  Use for "
            "long-running analysis or tasks that don't need an immediate "
            "response.  The result will be sent via Telegram when complete."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Description of the background task to run.",
                },
                "context": {
                    "type": "string",
                    "description": "Optional context or data for the task.",
                },
            },
            "required": ["task"],
        },
        execute=_exec_spawn,
    )
