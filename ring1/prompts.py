"""Evolution and crystallization prompt templates for Ring 1.

Builds system + user prompts for Claude to mutate Ring 2 code.
Extracts Python code blocks from LLM responses.
Crystallization: analyse surviving Ring 2 code and extract reusable skills.
"""

from __future__ import annotations

import json
import re

SYSTEM_PROMPT = """\
You are the evolution engine for Protea, a self-evolving artificial life system.
Your task is to mutate the Ring 2 code to create a new generation.

## Absolute Constraints (MUST follow — violation = immediate death)
1. The code MUST maintain a heartbeat protocol:
   - Read the heartbeat file path from PROTEA_HEARTBEAT environment variable
   - Write the file every 2 seconds with format: "{pid}\\n{timestamp}\\n"
   - The heartbeat keeps the program alive — without it, Ring 0 will kill the process
2. The code MUST have a `main()` function as entry point
3. The code MUST be a single valid Python file (pure stdlib only, no pip packages)
4. The code MUST handle KeyboardInterrupt gracefully and clean up the heartbeat file

## Evolution Strategy
Beyond the heartbeat constraint, evolve the code to be USEFUL to the user.
Prioritize capabilities that align with the user's recent tasks and needs.
If no task history is available, explore interesting computational abilities.
The code can:
- Compute things (math, fractals, cellular automata, simulations)
- Generate data (sequences, patterns, art in text)
- Explore algorithms (sorting, searching, optimization)
- Build data structures
- Implement games or puzzles
- Do file I/O (within the ring2 directory)
- Anything else that's interesting and runs with pure stdlib

Refer to user task history (if provided) to guide evolution direction.
Avoid duplicating existing skills — develop complementary capabilities.

## Fitness (scored 0.0–1.0)
Survival is necessary but NOT sufficient — a program that only heartbeats scores 0.50.
- Base survival: 0.50 (survived max_runtime)
- Output volume: up to +0.15 (meaningful non-empty lines, saturates at 50 lines)
- Output diversity: up to +0.15 (unique lines / total lines)
- Structured output: up to +0.10 (JSON blocks, tables, key:value reports)
- Error penalty: up to −0.10 (traceback/error/exception lines reduce score)
Produce diverse, structured, error-free output to maximise your score.

## Response Format
Start with a SHORT reflection (1-2 sentences max), then the complete code.
Keep the reflection brief — the code is what matters.

## Reflection
[1-2 sentences: what pattern you noticed and your mutation strategy]

```python
# your complete mutated code here
```
"""


def build_evolution_prompt(
    current_source: str,
    fitness_history: list[dict],
    best_performers: list[dict],
    params: dict,
    generation: int,
    survived: bool,
    directive: str = "",
    memories: list[dict] | None = None,
    task_history: list[dict] | None = None,
    skills: list[dict] | None = None,
    crash_logs: list[dict] | None = None,
) -> tuple[str, str]:
    """Build (system_prompt, user_message) for the evolution LLM call."""
    parts: list[str] = []

    parts.append(f"## Generation {generation}")
    parts.append(f"Previous generation {'SURVIVED' if survived else 'DIED'}.")
    parts.append(f"Mutation rate: {params.get('mutation_rate', 0.1)}")
    parts.append(f"Max runtime: {params.get('max_runtime_sec', 60)}s")
    parts.append("")

    # Current source code
    parts.append("## Current Ring 2 Code")
    parts.append("```python")
    parts.append(current_source.rstrip())
    parts.append("```")
    parts.append("")

    # Fitness history
    if fitness_history:
        parts.append("## Recent Fitness History")
        for entry in fitness_history[:10]:
            status = "SURVIVED" if entry.get("survived") else "DIED"
            parts.append(
                f"- Gen {entry.get('generation', '?')}: "
                f"score={entry.get('score', 0):.2f}, "
                f"runtime={entry.get('runtime_sec', 0):.1f}s, "
                f"{status}"
            )
        parts.append("")

    # Best performers
    if best_performers:
        parts.append("## Best Performers (by score)")
        for entry in best_performers[:5]:
            parts.append(
                f"- Gen {entry.get('generation', '?')}: "
                f"score={entry.get('score', 0):.2f}, "
                f"hash={entry.get('commit_hash', '?')[:8]}"
            )
        parts.append("")

    # Learned patterns from memory
    if memories:
        parts.append("## Learned Patterns (from memory)")
        for mem in memories:
            gen = mem.get("generation", "?")
            mtype = mem.get("entry_type", "?")
            content = mem.get("content", "")
            parts.append(f"- [Gen {gen}, {mtype}] {content}")
        parts.append("")

    # Recent user tasks — guide evolution direction
    if task_history:
        parts.append("## Recent User Tasks")
        for task in task_history[:10]:
            content = task.get("content", "")
            parts.append(f"- {content}")
        parts.append("Consider evolving capabilities useful for these tasks.")
        parts.append("")

    # Existing skills — avoid duplication
    if skills:
        parts.append("## Existing Skills (avoid duplication)")
        for skill in skills[:20]:
            name = skill.get("name", "?")
            desc = skill.get("description", "")
            parts.append(f"- {name}: {desc}")
        parts.append("Develop complementary capabilities instead of duplicating these.")
        parts.append("")

    # Recent crash logs — help diagnose failures
    if crash_logs:
        parts.append("## Recent Crash Logs")
        for log_entry in crash_logs[:3]:
            gen = log_entry.get("generation", "?")
            content = log_entry.get("content", "")
            parts.append(f"### Gen {gen} crash:")
            parts.append(content[:2000])
        parts.append("")

    # Instructions based on outcome
    if survived:
        parts.append("## Instructions")
        parts.append(
            "The previous code survived! Evolve it further — make it do something "
            "more interesting or complex while keeping the heartbeat alive. "
            "Be creative and try something NEW."
        )
    else:
        parts.append("## Instructions")
        parts.append(
            "The previous code DIED (heartbeat lost). Fix the issue and make it "
            "more robust. Ensure the heartbeat loop runs reliably. "
            "Then add interesting behavior on top."
        )

    if directive:
        parts.append("")
        parts.append("## User Directive")
        parts.append(
            f"The user has requested a specific direction for evolution: {directive}\n"
            "Prioritize this directive while still following all constraints above."
        )

    return SYSTEM_PROMPT, "\n".join(parts)


def extract_python_code(response: str) -> str | None:
    """Extract the first ```python code block from an LLM response.

    Returns None if no valid code block is found.
    """
    # Match ```python ... ``` blocks (non-greedy).
    pattern = r"```python\s*\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if code:
            return code
    return None


def extract_reflection(response: str) -> str | None:
    """Extract reflection text from an LLM response.

    Looks for text between ``## Reflection`` and the first
    ````` ```python ````` code fence.  Returns ``None`` if no reflection found.
    """
    pattern = r"## Reflection\s*\n(.*?)```python"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        text = match.group(1).strip()
        if text:
            return text
    return None


# ---------------------------------------------------------------------------
# Skill Crystallization
# ---------------------------------------------------------------------------

CRYSTALLIZE_SYSTEM_PROMPT = """\
You are the skill crystallization engine for Protea, a self-evolving artificial life system.

Your task: analyse Ring 2 source code that has successfully survived, and decide \
whether it represents a reusable *skill* worth preserving.

## What to ignore
- Heartbeat boilerplate (PROTEA_HEARTBEAT, write_heartbeat, heartbeat loop)
- Generic setup code (import os, pathlib, signal handling)
- Trivial programs that only maintain the heartbeat and do nothing else

## What to extract
Focus on the **core capability** — the interesting algorithm, interaction pattern, \
data processing, game logic, web server, visualisation, or other useful behaviour \
beyond the heartbeat.

## Decision
Compare the code's capability against the list of existing skills provided.
- **create**: The code demonstrates a genuinely new capability not covered by any \
existing skill.
- **update**: The code is an improved or extended version of an existing skill.
- **skip**: The existing skills already cover this capability, or the code is too \
trivial to crystallize.

## Response format
Respond with a single JSON object (no markdown fences, no extra text):

For create:
{"action": "create", "name": "skill_name_snake_case", "description": "One-sentence description", "prompt_template": "Core pattern description with key code snippets and algorithms", "tags": ["tag1", "tag2"]}

For update:
{"action": "update", "existing_name": "skill_name", "description": "Updated description", "prompt_template": "Updated core pattern", "tags": ["tag1", "tag2"]}

For skip:
{"action": "skip", "reason": "Brief explanation of why this was skipped"}
"""


def build_crystallize_prompt(
    source_code: str,
    output: str,
    generation: int,
    existing_skills: list[dict],
    skill_cap: int = 100,
) -> tuple[str, str]:
    """Build (system_prompt, user_message) for the crystallization LLM call."""
    parts: list[str] = []

    parts.append(f"## Ring 2 Source (Generation {generation})")
    parts.append("```python")
    parts.append(source_code.rstrip())
    parts.append("```")
    parts.append("")

    if output:
        parts.append("## Program Output (last lines)")
        parts.append(output[-2000:])
        parts.append("")

    if existing_skills:
        parts.append("## Existing Skills")
        for skill in existing_skills:
            name = skill.get("name", "?")
            desc = skill.get("description", "")
            tags = skill.get("tags", [])
            parts.append(f"- {name}: {desc} (tags: {', '.join(tags) if tags else 'none'})")
        parts.append("")

    active_count = len(existing_skills)
    parts.append(f"## Capacity: {active_count}/{skill_cap} skills")
    if active_count >= skill_cap:
        parts.append("The skill store is FULL. Only create if this is clearly better than the least-used existing skill.")
    parts.append("")

    parts.append("Respond with a single JSON object.")

    return CRYSTALLIZE_SYSTEM_PROMPT, "\n".join(parts)


_VALID_ACTIONS = {"create", "update", "skip"}


def parse_crystallize_response(response: str) -> dict | None:
    """Parse the JSON response from the crystallization LLM call.

    Handles optional markdown code-block wrappers. Returns None on
    parse failure or invalid action.
    """
    text = response.strip()
    # Strip markdown code fences if present.
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("action") not in _VALID_ACTIONS:
        return None
    return data
