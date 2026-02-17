# Protea Long-Term Memory: Storage & Retrieval Analysis

> Date: 2026-02-18

## 1. Storage Layer — `MemoryStore`

`ring0/memory.py` uses a single SQLite table:

```sql
CREATE TABLE memory (
    id          INTEGER PRIMARY KEY,
    generation  INTEGER  NOT NULL,
    entry_type  TEXT     NOT NULL,    -- 'observation' | 'crash_log' | 'reflection' | 'directive' | 'task'
    content     TEXT     NOT NULL,    -- free-form text
    metadata    TEXT     DEFAULT '{}', -- JSON extension field (currently unused)
    timestamp   TEXT     DEFAULT CURRENT_TIMESTAMP
)
```

All memory entries share the same flat table. No indexing beyond the primary key.

---

## 2. Memory Types (5 types)

| Type | Written When | Written At | Content |
|------|-------------|------------|---------|
| `observation` | After each survived generation | `sentinel.py:589-595` | "Gen N survived Xs, Code: N bytes, Output: ..." |
| `crash_log` | After each failed generation | `sentinel.py:714-720` | "Gen N died after Xs, Reason: ..., Last output: ..." |
| `reflection` | After LLM evolution | `evolver.py:141-146` | LLM self-reflection text (1-2 sentences) |
| `directive` | When user sets evolution direction | `sentinel.py:636-637` | User's directive text |
| `task` | After user task completion | `task_executor.py:388-401` | "User asked: ..., Result: ..." |

---

## 3. Retrieval Patterns (Only 2 Query Methods)

`MemoryStore` provides only two retrieval methods:

```python
# Method 1: Most recent N entries (all types, ordered by id DESC)
memory_store.get_recent(limit=3)

# Method 2: Most recent N entries of a specific type
memory_store.get_by_type("crash_log", limit=3)
memory_store.get_by_type("task", limit=5)
```

**No semantic search, no keyword matching, no relevance ranking.**

---

## 4. How Memory Is Consumed

### During Evolution (`_try_evolve` in sentinel.py:148-154)

| Query | Injected Into Prompt Section |
|-------|------------------------------|
| `get_recent(3)` | `## Learned Patterns` — last 3 memories (any type) |
| `get_by_type("task", 5)` | `## Recent User Tasks` — last 5 task records |
| `get_by_type("crash_log", 3)` | `## Recent Crashes` — last 3 crash logs |

These are concatenated into the evolution prompt (`ring1/prompts.py:142-206`) and sent to the LLM.

### During User Task Execution (task_executor.py:326-340)

| Query | Injected Into Prompt Section |
|-------|------------------------------|
| `get_recent(3)` | `## Recent Learnings` — last 3 memories |

### During Autonomous Task Generation (task_executor.py:418-427)

| Query | Purpose |
|-------|---------|
| `get_by_type("task", 10)` | Infer user needs from recent task history |

---

## 5. Short-Term Conversation Memory (Separate System)

`task_executor.py:235-254` maintains a purely in-memory conversation history:

```python
self._chat_history: list[tuple[float, str, str]] = []
self._chat_history_max = 5       # max 5 rounds
self._chat_history_ttl = 600     # 10-minute TTL
```

**This is NOT persisted** — lost on process restart. It provides conversational continuity within a single session only.

---

## 6. Data Flow Diagram

```
User Task ──────► TaskExecutor ──► memory_store.add("task", ...)
                       │
                       ▼
                  _chat_history (in-memory, 5 rounds, 10min TTL)

Ring 2 Survived ──► sentinel ──► memory_store.add("observation", ...)
                       │
                       ▼
                  gene_pool.add(generation, score, source)

Ring 2 Died ────► sentinel ──► memory_store.add("crash_log", ...)

Evolution LLM ──► evolver ──► memory_store.add("reflection", ...)

User Directive ─► sentinel ──► memory_store.add("directive", ...)

                    ┌──────────────────────────────┐
                    │        RETRIEVAL              │
                    │                               │
Evolution ──────►   │  get_recent(3)                │──► LLM prompt
                    │  get_by_type("task", 5)       │
                    │  get_by_type("crash_log", 3)  │
                    │                               │
Task Exec ──────►   │  get_recent(3)                │──► LLM prompt
                    │                               │
P1 Auto Task ───►   │  get_by_type("task", 10)      │──► LLM prompt
                    └──────────────────────────────┘
```

---

## 7. Core Problems

| Problem | Impact | Severity |
|---------|--------|----------|
| **No semantic retrieval** | Only retrieves "most recent N". Valuable early experiences get buried as memories accumulate to hundreds | High |
| **No summarization/compression** | Each memory stored verbatim with raw output. Only 3 fit in LLM prompt, rest wasted | High |
| **No importance weighting** | `observation`, `reflection`, `crash_log` treated equally. `get_recent(3)` may return 3 unimportant observations | Medium |
| **Conversation history not persisted** | `_chat_history` is pure in-memory, lost on restart. User context continuity breaks | Medium |
| **No cross-generation linking** | No way to track "Gen 50's crash and Gen 55's reflection solved the same problem" | Medium |
| **No forgetting/eviction** | Table only grows, never shrinks. No expiry or cleanup mechanism | Low |
| **metadata field unused** | Schema has metadata JSON field but no writer actually populates it | Low |

---

## 8. Improvement Proposals

### P0 — Tiered Retrieval (No External Dependencies)

Replace flat `get_recent()` with importance-aware retrieval:

```python
def get_context_for_evolution(self, limit=10):
    """Retrieve memories ranked by relevance, not just recency."""
    results = []
    # 1. Most recent directive (user's latest intent) — highest priority
    results += self.get_by_type("directive", limit=1)
    # 2. Recent crash_logs (current problems to fix)
    results += self.get_by_type("crash_log", limit=2)
    # 3. Recent reflections (LLM insights)
    results += self.get_by_type("reflection", limit=3)
    # 4. High-score generation observations (successful patterns)
    results += self._get_high_score_observations(limit=2)
    # 5. Recent tasks (user needs)
    results += self.get_by_type("task", limit=2)
    return results[:limit]
```

### P1 — Memory Summarization

Periodically compress old memories via LLM:

```python
# Every 50 generations, summarize old memories into a condensed entry:
# "Gen 1-50: Learned heartbeat mechanism, JSON output, HTTP server.
#  Recurring issues: import errors, heartbeat loss.
#  User preferences: web scraping, data analysis tasks."
```

This keeps the memory table manageable while preserving long-term knowledge.

### P1 — Persist Conversation History

Move `_chat_history` from in-memory list to SQLite:

```python
# New table in memory.py or separate store
CREATE TABLE chat_history (
    id          INTEGER PRIMARY KEY,
    user_text   TEXT NOT NULL,
    response    TEXT NOT NULL,
    timestamp   TEXT DEFAULT CURRENT_TIMESTAMP
)
```

With TTL-based cleanup on retrieval (same 10-minute window, but survives restart).

### P2 — Keyword-Based Search (Pure Stdlib)

Add basic full-text search without external dependencies:

```python
def search(self, keywords: list[str], limit: int = 10) -> list[dict]:
    """Search memories by keyword matching."""
    placeholders = " AND ".join(["content LIKE ?" for _ in keywords])
    params = [f"%{kw}%" for kw in keywords]
    with self._connect() as con:
        rows = con.execute(
            f"SELECT * FROM memory WHERE {placeholders} "
            "ORDER BY id DESC LIMIT ?",
            params + [limit],
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]
```

Or use SQLite FTS5 for better full-text search performance (still pure stdlib).

### P2 — Forgetting & Decay

Implement a retention policy:

```python
def compact(self, keep_recent: int = 200, keep_types: dict[str, int] | None = None):
    """Remove old low-value memories, keeping the most recent N per type."""
    # Keep last 200 entries overall
    # Keep all directives and tasks (user intent is always valuable)
    # Summarize and remove old observations and crash_logs
```

### P3 — Semantic Search (Requires External Dependencies)

For Ring 1 only (Ring 0 must stay pure stdlib):

- Embed memories using the LLM (or a lightweight embedding model)
- Store embeddings as BLOB in SQLite or a vector store
- Retrieve by cosine similarity to the current context

---

## 9. Comparison: Current vs Ideal

| Aspect | Current | Ideal |
|--------|---------|-------|
| Storage | Single flat table, unbounded growth | Tiered: hot (recent), warm (summarized), cold (archived) |
| Retrieval | `ORDER BY id DESC LIMIT N` | Relevance-ranked: type priority + recency + keyword match |
| Context window | Fixed 3 memories per prompt | Adaptive: more memories when context budget allows |
| Compression | None — raw text stored | Periodic LLM summarization of old memories |
| Conversation | In-memory, lost on restart | SQLite-backed, TTL-managed |
| Cross-reference | None | Link related memories (e.g., crash → fix → reflection) |
| Eviction | None | TTL + importance decay + capacity limits |
