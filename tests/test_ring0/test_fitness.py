"""Tests for ring0.fitness — FitnessTracker and evaluate_output."""

from __future__ import annotations

import json

from ring0.fitness import FitnessTracker, evaluate_output


class TestRecord:
    """record() should insert rows and return their rowid."""

    def test_insert_returns_rowid(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        rid = tracker.record(
            generation=1,
            commit_hash="abc123",
            score=0.85,
            runtime_sec=1.2,
            survived=True,
        )
        assert rid == 1

    def test_successive_inserts_increment_rowid(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        r1 = tracker.record(1, "aaa", 0.5, 1.0, True)
        r2 = tracker.record(1, "bbb", 0.6, 1.1, False)
        assert r2 == r1 + 1


class TestGetBest:
    """get_best() should return entries sorted by score descending."""

    def test_returns_sorted_by_score(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        tracker.record(1, "low", 0.1, 1.0, True)
        tracker.record(1, "high", 0.9, 1.0, True)
        tracker.record(1, "mid", 0.5, 1.0, True)

        best = tracker.get_best(n=3)
        scores = [entry["score"] for entry in best]
        assert scores == [0.9, 0.5, 0.1]

    def test_limits_to_n(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        for i in range(10):
            tracker.record(1, f"hash{i}", float(i), 1.0, True)

        best = tracker.get_best(n=3)
        assert len(best) == 3
        assert best[0]["score"] == 9.0

    def test_empty_database_returns_empty_list(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        assert tracker.get_best() == []


class TestGetGenerationStats:
    """get_generation_stats() should compute correct aggregates."""

    def test_computes_correct_stats(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        tracker.record(1, "a", 0.2, 1.0, True)
        tracker.record(1, "b", 0.8, 1.0, True)
        tracker.record(1, "c", 0.5, 1.0, False)

        stats = tracker.get_generation_stats(generation=1)
        assert stats is not None
        assert stats["count"] == 3
        assert stats["max_score"] == 0.8
        assert stats["min_score"] == 0.2
        assert abs(stats["avg_score"] - 0.5) < 1e-9

    def test_returns_none_for_missing_generation(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        assert tracker.get_generation_stats(generation=99) is None

    def test_ignores_other_generations(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        tracker.record(1, "a", 1.0, 1.0, True)
        tracker.record(2, "b", 0.0, 1.0, True)

        stats = tracker.get_generation_stats(generation=1)
        assert stats is not None
        assert stats["count"] == 1
        assert stats["avg_score"] == 1.0


class TestGetHistory:
    """get_history() should return entries in reverse chronological order."""

    def test_returns_reverse_order(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        tracker.record(1, "first", 0.1, 1.0, True)
        tracker.record(2, "second", 0.2, 1.0, True)
        tracker.record(3, "third", 0.3, 1.0, True)

        history = tracker.get_history(limit=10)
        hashes = [entry["commit_hash"] for entry in history]
        assert hashes == ["third", "second", "first"]

    def test_respects_limit(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        for i in range(10):
            tracker.record(i, f"hash{i}", float(i), 1.0, True)

        history = tracker.get_history(limit=3)
        assert len(history) == 3
        # Most recent first.
        assert history[0]["commit_hash"] == "hash9"

    def test_empty_database_returns_empty_list(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        assert tracker.get_history() == []


class TestEvaluateOutput:
    """evaluate_output() multi-factor scoring."""

    def test_empty_output_survivor_gets_base(self):
        score, detail = evaluate_output([], survived=True, elapsed=60, max_runtime=60)
        assert score == 0.50
        assert detail["basis"] == "survived"
        assert detail["volume"] == 0.0

    def test_rich_diverse_output_scores_high(self):
        lines = [f"result_{i}: {i * 3.14:.4f}" for i in range(60)]
        score, detail = evaluate_output(lines, survived=True, elapsed=60, max_runtime=60)
        assert score > 0.80
        assert detail["volume"] == 0.15  # saturated at 50+
        assert detail["diversity"] > 0.10

    def test_duplicate_output_low_diversity(self):
        lines = ["same line"] * 50
        score, detail = evaluate_output(lines, survived=True, elapsed=60, max_runtime=60)
        # 1 unique / 50 total → diversity near 0
        assert detail["diversity"] < 0.01

    def test_structured_output_bonus(self):
        lines = [
            '{"key": "value"}',
            '| col1 | col2 |',
            'status: running',
            'normal line',
        ]
        score, detail = evaluate_output(lines, survived=True, elapsed=60, max_runtime=60)
        assert detail["structure"] > 0.0

    def test_error_penalty_applied(self):
        lines = [
            "Traceback (most recent call last):",
            '  File "main.py", line 1',
            "ZeroDivisionError: division by zero",
            "Another error occurred",
        ]
        score, detail = evaluate_output(lines, survived=True, elapsed=60, max_runtime=60)
        assert detail["error_penalty"] > 0.0
        # Score should still be >= 0.50 (floor for survivors)
        assert score >= 0.50

    def test_failure_capped_below_survivor(self):
        # Best possible failure: elapsed == max_runtime → ratio 0.99
        score, detail = evaluate_output(
            ["some output"], survived=False, elapsed=59.4, max_runtime=60,
        )
        assert detail["basis"] == "failure"
        assert score < 0.50

    def test_failure_zero_runtime(self):
        score, detail = evaluate_output([], survived=False, elapsed=0, max_runtime=60)
        assert score == 0.0

    def test_failure_zero_max_runtime(self):
        score, detail = evaluate_output([], survived=False, elapsed=10, max_runtime=0)
        assert score == 0.0

    def test_whitespace_lines_ignored(self):
        lines = ["  ", "\t", "", "actual output"]
        score, detail = evaluate_output(lines, survived=True, elapsed=60, max_runtime=60)
        assert detail["meaningful_lines"] == 1


class TestDetailColumn:
    """Schema migration adds detail column, record() stores it."""

    def test_detail_stored_and_retrievable(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        detail = {"basis": "survived", "volume": 0.15}
        tracker.record(1, "abc", 0.75, 60.0, True, detail=detail)

        rows = tracker.get_history(limit=1)
        assert len(rows) == 1
        stored = json.loads(rows[0]["detail"])
        assert stored["volume"] == 0.15

    def test_detail_none_by_default(self, tmp_path):
        tracker = FitnessTracker(tmp_path / "fitness.db")
        tracker.record(1, "abc", 0.5, 60.0, True)

        rows = tracker.get_history(limit=1)
        assert rows[0]["detail"] is None

    def test_migration_idempotent(self, tmp_path):
        """Creating FitnessTracker twice should not error."""
        db = tmp_path / "fitness.db"
        FitnessTracker(db)
        FitnessTracker(db)  # second init — ALTER TABLE should be no-op
