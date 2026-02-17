"""Tests for ring0.gene_pool."""

import pathlib
import subprocess

import pytest

from ring0.fitness import FitnessTracker
from ring0.gene_pool import GenePool


# --- Sample Ring 2 source code for testing ---

SAMPLE_SOURCE = '''\
import os, pathlib, time, threading, json

def write_heartbeat(path, pid):
    """Write heartbeat file."""
    path.write_text(f"{pid}\\n{time.time()}\\n")

def heartbeat_loop(hb_path, pid):
    while True:
        write_heartbeat(hb_path, pid)
        time.sleep(2)

class StreamAnalyzer:
    """Real-time anomaly detection in data streams."""

    def __init__(self, window_size=100):
        self.window_size = window_size
        self.buffer = []

    def analyze(self, value):
        """Detect anomalies using z-score method."""
        self.buffer.append(value)

class PackageScanner:
    """PyPI dependency graph builder."""

    def scan(self, package_name):
        """Scan a package and its dependencies."""
        pass

def compute_fibonacci(n):
    """Calculate fibonacci sequence."""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

def main():
    hb = pathlib.Path(os.environ.get("PROTEA_HEARTBEAT", ".heartbeat"))
    pid = os.getpid()
    t = threading.Thread(target=heartbeat_loop, args=(hb, pid), daemon=True)
    t.start()
    analyzer = StreamAnalyzer()
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
'''

TRIVIAL_SOURCE = '''\
import os, pathlib, time

def write_heartbeat(path, pid):
    path.write_text(f"{pid}\\n{time.time()}\\n")

def main():
    hb = pathlib.Path(os.environ.get("PROTEA_HEARTBEAT", ".heartbeat"))
    pid = os.getpid()
    while True:
        write_heartbeat(hb, pid)
        time.sleep(2)

if __name__ == "__main__":
    main()
'''


class TestExtractSummary:
    def test_extracts_class_signatures(self):
        summary = GenePool.extract_summary(SAMPLE_SOURCE)
        assert "class StreamAnalyzer" in summary
        assert "class PackageScanner" in summary

    def test_extracts_function_signatures(self):
        summary = GenePool.extract_summary(SAMPLE_SOURCE)
        assert "def compute_fibonacci" in summary

    def test_extracts_method_signatures(self):
        summary = GenePool.extract_summary(SAMPLE_SOURCE)
        # AST can extract methods inside classes.
        assert "def analyze" in summary
        assert "def scan" in summary

    def test_extracts_docstrings(self):
        summary = GenePool.extract_summary(SAMPLE_SOURCE)
        assert "Real-time anomaly detection" in summary
        assert "PyPI dependency graph" in summary

    def test_skips_heartbeat_boilerplate(self):
        summary = GenePool.extract_summary(SAMPLE_SOURCE)
        assert "write_heartbeat" not in summary
        assert "heartbeat_loop" not in summary
        assert "def main" not in summary

    def test_caps_at_500_chars(self):
        # Create a source with many classes to exceed 500 chars.
        lines = []
        for i in range(50):
            lines.append(f"class Widget{i}:")
            lines.append(f'    """Widget number {i} with a description."""')
            lines.append(f"    pass")
        big_source = "\n".join(lines)
        summary = GenePool.extract_summary(big_source)
        assert len(summary) <= 500
        assert summary.endswith("...")

    def test_empty_source(self):
        summary = GenePool.extract_summary("")
        assert summary == ""

    def test_trivial_heartbeat_only(self):
        summary = GenePool.extract_summary(TRIVIAL_SOURCE)
        # Should be empty — only heartbeat boilerplate.
        assert summary.strip() == ""

    def test_regex_fallback_on_broken_code(self):
        """Code with syntax errors falls back to regex extraction."""
        broken = (
            "class GoodClass:\n"
            '    """A useful class."""\n'
            "    pass\n\n"
            "def broken_syntax(\n"  # unclosed paren — SyntaxError
        )
        summary = GenePool.extract_summary(broken)
        assert "GoodClass" in summary
        assert "A useful class" in summary

    def test_extracts_method_docstrings(self):
        summary = GenePool.extract_summary(SAMPLE_SOURCE)
        assert "Detect anomalies" in summary
        assert "Scan a package" in summary

    def test_skips_init_methods(self):
        summary = GenePool.extract_summary(SAMPLE_SOURCE)
        assert "__init__" not in summary


class TestGenePoolAdd:
    def test_add_basic(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=5)
        added = gp.add(1, 0.85, SAMPLE_SOURCE)
        assert added is True
        assert gp.count() == 1

    def test_add_dedup(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=5)
        gp.add(1, 0.85, SAMPLE_SOURCE)
        added = gp.add(2, 0.90, SAMPLE_SOURCE)  # same source
        assert added is False
        assert gp.count() == 1

    def test_add_trivial_rejected(self, tmp_path):
        """Trivial code with empty summary should not be added."""
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=5)
        added = gp.add(1, 0.85, TRIVIAL_SOURCE)
        assert added is False
        assert gp.count() == 0

    def test_evict_lowest_when_full(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=3)

        # Fill the pool with 3 entries.
        for i in range(3):
            source = SAMPLE_SOURCE + f"\n# unique_{i}\n"
            gp.add(i, 0.50 + i * 0.10, source)
        assert gp.count() == 3

        # Add a higher score — should evict the lowest (0.50).
        new_source = SAMPLE_SOURCE + "\n# unique_high\n"
        added = gp.add(99, 0.95, new_source)
        assert added is True
        assert gp.count() == 3

        # The lowest should now be 0.60 (not 0.50).
        top = gp.get_top(10)
        scores = [g["score"] for g in top]
        assert 0.50 not in scores
        assert 0.95 in scores

    def test_reject_when_full_and_low_score(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=2)

        gp.add(1, 0.80, SAMPLE_SOURCE + "\n# a\n")
        gp.add(2, 0.90, SAMPLE_SOURCE + "\n# b\n")
        assert gp.count() == 2

        # Score too low to enter.
        added = gp.add(3, 0.70, SAMPLE_SOURCE + "\n# c\n")
        assert added is False
        assert gp.count() == 2


class TestGenePoolGetTop:
    def test_get_top_sorted(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)

        gp.add(1, 0.60, SAMPLE_SOURCE + "\n# v1\n")
        gp.add(2, 0.90, SAMPLE_SOURCE + "\n# v2\n")
        gp.add(3, 0.75, SAMPLE_SOURCE + "\n# v3\n")

        top = gp.get_top(2)
        assert len(top) == 2
        assert top[0]["score"] == 0.90
        assert top[1]["score"] == 0.75

    def test_get_top_with_fewer(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        gp.add(1, 0.80, SAMPLE_SOURCE + "\n# only\n")

        top = gp.get_top(5)
        assert len(top) == 1

    def test_get_top_empty(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)

        top = gp.get_top(3)
        assert top == []

    def test_get_top_contains_summary(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        gp.add(1, 0.85, SAMPLE_SOURCE + "\n# x\n")

        top = gp.get_top(1)
        assert len(top) == 1
        assert "gene_summary" in top[0]
        assert "StreamAnalyzer" in top[0]["gene_summary"]


class TestGenePoolBackfill:
    def test_backfill_from_skills(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)

        # Mock skill store.
        class MockSkillStore:
            def get_active(self, limit):
                return [
                    {"id": 1, "source_code": SAMPLE_SOURCE, "usage_count": 5},
                    {"id": 2, "source_code": SAMPLE_SOURCE + "\n# v2\n", "usage_count": 0},
                ]

        added = gp.backfill(MockSkillStore())
        assert added >= 1

    def test_backfill_skips_if_nonempty(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)
        gp.add(1, 0.85, SAMPLE_SOURCE + "\n# pre\n")

        class MockSkillStore:
            def get_active(self, limit):
                return [{"id": 1, "source_code": SAMPLE_SOURCE + "\n# new\n", "usage_count": 5}]

        added = gp.backfill(MockSkillStore())
        assert added == 0

    def test_backfill_skips_short_source(self, tmp_path):
        db = tmp_path / "test.db"
        gp = GenePool(db, max_size=10)

        class MockSkillStore:
            def get_active(self, limit):
                return [
                    {"id": 1, "source_code": "x=1", "usage_count": 5},
                    {"id": 2, "source_code": "", "usage_count": 0},
                ]

        added = gp.backfill(MockSkillStore())
        assert added == 0


def _init_git_repo(ring2_path: pathlib.Path, commits: list[tuple[str, str]]):
    """Create a git repo with commits. Each commit is (source_code, label).

    Returns list of actual commit hashes.
    """
    subprocess.run(["git", "init"], cwd=str(ring2_path), capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(ring2_path), capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(ring2_path), capture_output=True)

    hashes = []
    for source, _ in commits:
        (ring2_path / "main.py").write_text(source)
        subprocess.run(["git", "add", "main.py"], cwd=str(ring2_path), capture_output=True, check=True)
        subprocess.run(["git", "commit", "-m", "gen"], cwd=str(ring2_path), capture_output=True, check=True)
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ring2_path), capture_output=True, text=True, check=True,
        )
        hashes.append(result.stdout.strip())
    return hashes


class TestBackfillFromGit:
    def test_backfills_from_fitness_log(self, tmp_path):
        db = tmp_path / "test.db"
        ring2 = tmp_path / "ring2"
        ring2.mkdir()

        source_v1 = SAMPLE_SOURCE + "\n# gen1\n"
        source_v2 = SAMPLE_SOURCE + "\n# gen2\n"
        hashes = _init_git_repo(ring2, [(source_v1, ""), (source_v2, "")])

        gp = GenePool(db, max_size=10)
        ft = FitnessTracker(db)
        ft.record(1, hashes[0], 0.85, 1.0, True)
        ft.record(2, hashes[1], 0.90, 1.0, True)

        added = gp.backfill_from_git(ring2, ft)
        assert added == 2
        assert gp.count() == 2

    def test_skips_when_pool_full(self, tmp_path):
        db = tmp_path / "test.db"
        ring2 = tmp_path / "ring2"
        ring2.mkdir()

        hashes = _init_git_repo(ring2, [(SAMPLE_SOURCE + "\n# x\n", "")])

        gp = GenePool(db, max_size=1)
        ft = FitnessTracker(db)
        ft.record(1, hashes[0], 0.90, 1.0, True)

        # Fill the pool first.
        gp.add(99, 0.95, SAMPLE_SOURCE + "\n# existing\n")
        assert gp.count() == 1

        added = gp.backfill_from_git(ring2, ft)
        assert added == 0

    def test_dedup_same_source(self, tmp_path):
        db = tmp_path / "test.db"
        ring2 = tmp_path / "ring2"
        ring2.mkdir()

        # One commit, but two fitness_log entries pointing to it.
        source = SAMPLE_SOURCE + "\n# same\n"
        hashes = _init_git_repo(ring2, [(source, "")])

        gp = GenePool(db, max_size=10)
        ft = FitnessTracker(db)
        ft.record(1, hashes[0], 0.85, 1.0, True)
        ft.record(2, hashes[0], 0.90, 1.0, True)

        added = gp.backfill_from_git(ring2, ft)
        assert added == 1  # dedup by source_hash
        assert gp.count() == 1

    def test_skips_low_score(self, tmp_path):
        db = tmp_path / "test.db"
        ring2 = tmp_path / "ring2"
        ring2.mkdir()

        hashes = _init_git_repo(ring2, [(SAMPLE_SOURCE + "\n# low\n", "")])

        gp = GenePool(db, max_size=10)
        ft = FitnessTracker(db)
        ft.record(1, hashes[0], 0.50, 1.0, True)  # below 0.75 threshold

        added = gp.backfill_from_git(ring2, ft)
        assert added == 0

    def test_skips_without_git_dir(self, tmp_path):
        db = tmp_path / "test.db"
        ring2 = tmp_path / "ring2"
        ring2.mkdir()  # No .git

        gp = GenePool(db, max_size=10)
        ft = FitnessTracker(db)

        added = gp.backfill_from_git(ring2, ft)
        assert added == 0
