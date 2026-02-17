"""Fitness tracker backed by SQLite.

Records and queries fitness scores for every generation in the
self-evolving lifecycle.  Pure stdlib — no external dependencies.
"""

from __future__ import annotations

import json
import pathlib
import re
import sqlite3

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS fitness_log (
    id          INTEGER PRIMARY KEY,
    generation  INTEGER  NOT NULL,
    commit_hash TEXT     NOT NULL,
    score       REAL     NOT NULL,
    runtime_sec REAL     NOT NULL,
    survived    BOOLEAN  NOT NULL,
    timestamp   TEXT     DEFAULT CURRENT_TIMESTAMP
)
"""


_STRUCTURED_PATTERNS = [
    re.compile(r"^\s*[\[{]"),           # JSON array/object start
    re.compile(r"^\s*\|.*\|"),          # Markdown/ASCII table row
    re.compile(r"^[+-]{3,}"),           # table separator
    re.compile(r"^={3,}"),              # section separator
    re.compile(r"^\s*\w+\s*:\s+\S"),   # key: value pairs
]


def evaluate_output(
    output_lines: list[str],
    survived: bool,
    elapsed: float,
    max_runtime: float,
) -> tuple[float, dict]:
    """Score a Ring 2 run based on output quality.

    Returns (score, detail_dict) where score is 0.0–1.0 and detail_dict
    contains the scoring breakdown.
    """
    if not survived:
        ratio = min(elapsed / max_runtime, 0.99) if max_runtime > 0 else 0.0
        score = ratio * 0.49
        return score, {"basis": "failure", "elapsed_ratio": round(ratio, 4)}

    # --- Survivor scoring (0.50 – 1.0) ---
    base = 0.50

    # Filter meaningful lines (non-empty, non-whitespace-only).
    meaningful = [ln for ln in output_lines if ln.strip()]
    total = len(meaningful)

    # Volume: up to 0.15.  Ramp linearly to 50 lines, then saturate.
    volume = min(total / 50, 1.0) * 0.15

    # Diversity: unique content ratio.  Up to 0.15.
    if total > 0:
        unique = len(set(meaningful))
        diversity = (unique / total) * 0.15
    else:
        diversity = 0.0

    # Structured output: proportion of lines matching structured patterns.
    structured_count = 0
    for ln in meaningful:
        if any(pat.match(ln) for pat in _STRUCTURED_PATTERNS):
            structured_count += 1
    structure = min(structured_count / max(total, 1) * 2, 1.0) * 0.10

    # Error penalty: traceback/error lines reduce score.
    error_count = 0
    for ln in output_lines:
        low = ln.lower()
        if "traceback" in low or "error" in low or "exception" in low:
            error_count += 1
    error_penalty = min(error_count / max(total, 1), 1.0) * 0.10

    score = base + volume + diversity + structure - error_penalty
    score = max(0.50, min(score, 1.0))

    detail = {
        "basis": "survived",
        "base": base,
        "volume": round(volume, 4),
        "diversity": round(diversity, 4),
        "structure": round(structure, 4),
        "error_penalty": round(error_penalty, 4),
        "meaningful_lines": total,
        "error_lines": error_count,
    }
    return round(score, 4), detail


class FitnessTracker:
    """Evaluate and record fitness scores in a local SQLite database."""

    def __init__(self, db_path: pathlib.Path) -> None:
        self.db_path = db_path
        with self._connect() as con:
            con.execute(_CREATE_TABLE)
            # Migrate: add detail column if missing.
            try:
                con.execute("ALTER TABLE fitness_log ADD COLUMN detail TEXT")
            except sqlite3.OperationalError:
                pass  # column already exists

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.db_path))
        con.row_factory = sqlite3.Row
        return con

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        return dict(row)

    def record(
        self,
        generation: int,
        commit_hash: str,
        score: float,
        runtime_sec: float,
        survived: bool,
        detail: dict | None = None,
    ) -> int:
        """Insert a fitness entry and return its *rowid*."""
        detail_json = json.dumps(detail) if detail else None
        with self._connect() as con:
            cur = con.execute(
                "INSERT INTO fitness_log "
                "(generation, commit_hash, score, runtime_sec, survived, detail) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (generation, commit_hash, score, runtime_sec, survived, detail_json),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def get_best(self, n: int = 5) -> list[dict]:
        """Return the top *n* entries ordered by score descending."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM fitness_log ORDER BY score DESC LIMIT ?",
                (n,),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_generation_stats(self, generation: int) -> dict | None:
        """Return aggregate stats for a single generation.

        Returns a dict with keys *avg_score*, *max_score*, *min_score*,
        and *count*, or ``None`` if the generation has no entries.
        """
        with self._connect() as con:
            row = con.execute(
                "SELECT AVG(score) AS avg_score, MAX(score) AS max_score, "
                "MIN(score) AS min_score, COUNT(*) AS count "
                "FROM fitness_log WHERE generation = ?",
                (generation,),
            ).fetchone()
            if row is None or row["count"] == 0:
                return None
            return self._row_to_dict(row)

    def get_history(self, limit: int = 50) -> list[dict]:
        """Return the most recent entries ordered by *id* descending."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM fitness_log ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]
