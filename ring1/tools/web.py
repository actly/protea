"""Web tools — wraps existing web_search/web_fetch as Tool instances.

Pure stdlib.
"""

from __future__ import annotations

import json

from ring1.tool_registry import Tool
from ring1.web_tools import web_fetch, web_search


def make_web_tools() -> list[Tool]:
    """Create Tool instances for web_search and web_fetch."""

    def _exec_search(inp: dict) -> str:
        results = web_search(
            query=inp["query"],
            max_results=inp.get("max_results", 5),
        )
        return json.dumps(results, ensure_ascii=False)

    def _exec_fetch(inp: dict) -> str:
        return web_fetch(
            url=inp["url"],
            max_chars=inp.get("max_chars", 5000),
        )

    search_tool = Tool(
        name="web_search",
        description=(
            "Search the web via DuckDuckGo.  Returns a list of results, each "
            "with title, url, and snippet.  Use for research or lookup tasks."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default 5).",
                },
            },
            "required": ["query"],
        },
        execute=_exec_search,
    )

    fetch_tool = Tool(
        name="web_fetch",
        description=(
            "Fetch a URL and extract its text content.  Use to read a specific "
            "web page for detailed information."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch.",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return (default 5000).",
                },
            },
            "required": ["url"],
        },
        execute=_exec_fetch,
    )

    return [search_tool, fetch_tool]
