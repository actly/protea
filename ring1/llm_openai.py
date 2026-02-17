"""OpenAI-compatible LLM client — covers OpenAI, DeepSeek, and similar APIs.

Pure stdlib (urllib.request + json).  Same retry pattern as the Anthropic client.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Callable

from ring1.llm_base import LLMClient, LLMError

log = logging.getLogger("protea.llm_openai")

_RETRYABLE_CODES = {429, 500, 502, 503}
_MAX_RETRIES = 3
_BASE_DELAY = 2.0  # seconds


def _convert_tool_schema(tool: dict) -> dict:
    """Convert an Anthropic-style tool definition to OpenAI function-calling format.

    Anthropic uses ``input_schema``; OpenAI uses ``parameters`` inside a
    ``function`` wrapper.
    """
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {}),
        },
    }


class OpenAIClient(LLMClient):
    """OpenAI-compatible chat completions client (no third-party deps)."""

    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int = 4096,
        api_url: str = "https://api.openai.com/v1/chat/completions",
    ) -> None:
        if not api_key:
            raise LLMError("API key is not set")
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.api_url = api_url

    # ------------------------------------------------------------------
    # Internal: HTTP + retry
    # ------------------------------------------------------------------

    def _call_api(self, payload: dict) -> dict:
        """POST *payload* to the chat completions endpoint with retry."""
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                req = urllib.request.Request(
                    self.api_url, data=data, headers=headers, method="POST",
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last_error = exc
                code = exc.code
                if code in _RETRYABLE_CODES and attempt < _MAX_RETRIES - 1:
                    delay = _BASE_DELAY * (2 ** attempt)
                    log.warning(
                        "OpenAI-compat API %d — retry %d/%d in %.1fs",
                        code, attempt + 1, _MAX_RETRIES, delay,
                    )
                    time.sleep(delay)
                    continue
                raise LLMError(
                    f"OpenAI-compat API HTTP {code}: "
                    f"{exc.read().decode('utf-8', errors='replace')}"
                ) from exc
            except urllib.error.URLError as exc:
                last_error = exc
                if attempt < _MAX_RETRIES - 1:
                    delay = _BASE_DELAY * (2 ** attempt)
                    log.warning(
                        "OpenAI-compat API network error — retry %d/%d in %.1fs",
                        attempt + 1, _MAX_RETRIES, delay,
                    )
                    time.sleep(delay)
                    continue
                raise LLMError(f"OpenAI-compat API network error: {exc}") from exc
            except (TimeoutError, OSError) as exc:
                last_error = exc
                if attempt < _MAX_RETRIES - 1:
                    delay = _BASE_DELAY * (2 ** attempt)
                    log.warning(
                        "OpenAI-compat API timeout — retry %d/%d in %.1fs",
                        attempt + 1, _MAX_RETRIES, delay,
                    )
                    time.sleep(delay)
                    continue
                raise LLMError(f"OpenAI-compat API timeout: {exc}") from exc

        raise LLMError(
            f"OpenAI-compat API failed after {_MAX_RETRIES} retries"
        ) from last_error

    # ------------------------------------------------------------------
    # Public: simple message (no tools)
    # ------------------------------------------------------------------

    def send_message(self, system_prompt: str, user_message: str) -> str:
        """Send a message and return the assistant's text response."""
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }
        body = self._call_api(payload)
        return self._extract_text(body)

    # ------------------------------------------------------------------
    # Public: message with tool-call loop
    # ------------------------------------------------------------------

    def send_message_with_tools(
        self,
        system_prompt: str,
        user_message: str,
        tools: list[dict],
        tool_executor: Callable[[str, dict], str],
        max_rounds: int = 5,
    ) -> str:
        """Send a message and handle tool_calls rounds until a final text reply."""
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        openai_tools = [_convert_tool_schema(t) for t in tools]
        last_text: str = ""

        for _round_idx in range(max_rounds):
            payload = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": messages,
                "tools": openai_tools,
            }
            body = self._call_api(payload)
            choice = body.get("choices", [{}])[0]
            message = choice.get("message", {})
            finish_reason = choice.get("finish_reason", "stop")

            # Capture any text content.
            content = message.get("content") or ""
            tool_calls = message.get("tool_calls") or []

            if not tool_calls or finish_reason != "tool_calls":
                # No more tool calls — return text.
                if content:
                    return content
                if last_text:
                    return last_text
                raise LLMError("No text content in API response")

            # Remember text from this round as fallback.
            if content:
                last_text = content

            # Append assistant message (must include tool_calls for the API).
            messages.append(message)

            # Execute each tool call and append results.
            for tc in tool_calls:
                fn = tc.get("function", {})
                tool_name = fn.get("name", "")
                try:
                    tool_input = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    tool_input = {}

                try:
                    result_str = tool_executor(tool_name, tool_input)
                except Exception as exc:
                    log.warning("Tool %s execution failed: %s", tool_name, exc)
                    result_str = f"Error: {exc}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_str,
                })

        # max_rounds exhausted.
        log.warning("Tool use loop exhausted after %d rounds", max_rounds)
        if last_text:
            return last_text
        return (
            "I ran out of tool-call budget before finishing. "
            "The task may be partially complete — please check and retry if needed."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(body: dict) -> str:
        """Extract text from a chat completions response."""
        choices = body.get("choices", [])
        if not choices:
            raise LLMError("No choices in API response")
        content = choices[0].get("message", {}).get("content")
        if not content:
            raise LLMError("No text content in API response")
        return content
