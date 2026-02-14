"""Tool package — factory for creating the default ToolRegistry.

Pure stdlib.
"""

from __future__ import annotations

from ring1.tool_registry import ToolRegistry


def create_default_registry(
    *,
    workspace_path: str = ".",
    shell_timeout: int = 30,
    reply_fn=None,
    subagent_manager=None,
) -> ToolRegistry:
    """Build a ToolRegistry with all standard tools.

    Args:
        workspace_path: Root directory for file/shell tools.
        shell_timeout: Default timeout in seconds for shell exec.
        reply_fn: Callable(text) for the message tool.  None disables it.
        subagent_manager: SubagentManager for the spawn tool.  None disables it.

    Returns:
        A fully populated ToolRegistry.
    """
    from ring1.tools.filesystem import make_filesystem_tools
    from ring1.tools.message import make_message_tool
    from ring1.tools.shell import make_shell_tool
    from ring1.tools.web import make_web_tools

    registry = ToolRegistry()

    for tool in make_web_tools():
        registry.register(tool)

    for tool in make_filesystem_tools(workspace_path):
        registry.register(tool)

    registry.register(make_shell_tool(workspace_path, timeout=shell_timeout))

    if reply_fn is not None:
        registry.register(make_message_tool(reply_fn))

    if subagent_manager is not None:
        from ring1.tools.spawn import make_spawn_tool
        registry.register(make_spawn_tool(subagent_manager))

    return registry
