"""Sentinel — Ring 0 main loop (pure stdlib).

Launches and supervises Ring 2.  On success (survived max_runtime_sec),
triggers Ring 1 evolution to mutate the code.  On failure, rolls back
to the last known-good commit, evolves from that base, and restarts.
"""

from __future__ import annotations

import logging
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time
import tomllib
from typing import NamedTuple

from ring0.commit_watcher import CommitWatcher
from ring0.fitness import FitnessTracker, evaluate_output
from ring0.fitness_v2 import evaluate_output_v2, get_task_level_for_generation
from ring0.git_manager import GitManager
from ring0.heartbeat import HeartbeatMonitor
from ring0.gene_pool import GenePool
from ring0.memory import MemoryStore
from ring0.parameter_seed import generate_params, params_to_dict
from ring0.resource_monitor import check_resources
from ring0.skill_store import SkillStore
from ring0.task_store import TaskStore

log = logging.getLogger("protea.sentinel")


class Ring0Config(NamedTuple):
    """Typed configuration for the Ring 0 sentinel loop."""

    ring2_path: pathlib.Path
    heartbeat_path: pathlib.Path
    db_path: pathlib.Path
    heartbeat_interval_sec: int
    heartbeat_timeout_sec: int
    seed: int
    cooldown_sec: int
    plateau_window: int
    plateau_epsilon: float
    max_cpu_percent: float
    max_memory_percent: float
    max_disk_percent: float
    skill_cap: int


def _load_config(project_root: pathlib.Path) -> dict:
    cfg_path = project_root / "config" / "config.toml"
    with open(cfg_path, "rb") as f:
        return tomllib.load(f)


def _load_ring0_config(project_root: pathlib.Path, cfg: dict) -> Ring0Config:
    """Parse the [ring0] section of config into a typed Ring0Config."""
    r0 = cfg["ring0"]
    ring2_path = project_root / r0["git"]["ring2_path"]
    return Ring0Config(
        ring2_path=ring2_path,
        heartbeat_path=ring2_path / ".heartbeat",
        db_path=project_root / r0["fitness"]["db_path"],
        heartbeat_interval_sec=r0["heartbeat_interval_sec"],
        heartbeat_timeout_sec=r0["heartbeat_timeout_sec"],
        seed=r0["evolution"]["seed"],
        cooldown_sec=r0["evolution"].get("cooldown_sec", 900),
        plateau_window=r0["evolution"].get("plateau_window", 5),
        plateau_epsilon=r0["evolution"].get("plateau_epsilon", 0.03),
        max_cpu_percent=r0["max_cpu_percent"],
        max_memory_percent=r0["max_memory_percent"],
        max_disk_percent=r0["max_disk_percent"],
        skill_cap=r0["evolution"].get("skill_max_count", 100),
    )


def _start_ring2(ring2_path: pathlib.Path, heartbeat_path: pathlib.Path) -> subprocess.Popen:
    """Launch the Ring 2 process and return its Popen handle."""
    log_file = ring2_path / ".output.log"
    fh = open(log_file, "a")
    env = {**os.environ, "PROTEA_HEARTBEAT": str(heartbeat_path)}
    proc = subprocess.Popen(
        [sys.executable, str(ring2_path / "main.py")],
        cwd=str(ring2_path),
        env=env,
        stdout=fh,
        stderr=subprocess.STDOUT,
    )
    proc._log_fh = fh          # keep reference for later close
    proc._log_path = log_file  # keep path for later read
    log.info("Ring 2 started  pid=%d", proc.pid)
    return proc


def _stop_ring2(proc: subprocess.Popen | None) -> None:
    """Terminate the Ring 2 process if it is still running."""
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log.info("Ring 2 stopped  pid=%d", proc.pid)
    fh = getattr(proc, "_log_fh", None)
    if fh:
        fh.close()


def _read_ring2_output(proc, max_lines: int = 100) -> str:
    """Read the last *max_lines* from Ring 2's captured output log."""
    log_path = getattr(proc, "_log_path", None)
    if not log_path or not log_path.exists():
        return ""
    lines = log_path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def _classify_failure(proc, output: str) -> str:
    """Determine why Ring 2 failed based on return code and output."""
    rc = proc.returncode
    if rc is None:
        return "heartbeat timeout (process still running)"
    if rc < 0:
        import signal as _signal
        sig = _signal.Signals(-rc).name if -rc in _signal.Signals._value2member_map_ else str(-rc)
        return f"killed by signal {sig}"
    if rc != 0:
        # Extract the last Traceback from output.
        lines = output.splitlines()
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith("Traceback"):
                return "\n".join(lines[i:])
        return f"exit code {rc}"
    return "clean exit but heartbeat lost"


def _should_evolve(state, cooldown_sec: int, fitness=None, plateau_window: int = 5, plateau_epsilon: float = 0.03, has_directive: bool = False) -> tuple[bool, bool]:
    """Check whether evolution should proceed.

    Returns (should_evolve, is_plateaued):
    - should_evolve: True if evolution should run.
    - is_plateaued: True if scores are stagnant (signals to LLM to try
      something fundamentally different).

    Adaptive evolution: when scores are plateaued AND no user directive
    is pending, skip the LLM call to save tokens.  A directive always
    forces evolution.
    """
    if state.p0_active.is_set():
        return False, False
    if state.p1_active.is_set():
        return False, False
    if time.time() - state.last_evolution_time < cooldown_sec:
        return False, False

    # Detect plateau.
    plateaued = False
    if fitness:
        try:
            plateaued = fitness.is_plateaued(window=plateau_window, epsilon=plateau_epsilon)
        except Exception:
            pass

    # Adaptive: skip evolution when plateaued unless a directive is pending.
    if plateaued and not has_directive:
        log.info("Scores plateaued — skipping evolution to save tokens (set a directive to force)")
        return False, True

    return True, plateaued


def _try_evolve(project_root, fitness, ring2_path, generation, params, survived, notifier, directive="", memory_store=None, skill_store=None, crash_logs=None, is_plateaued=False, gene_pool=None):
    """Best-effort evolution.  Returns True if new code was written."""
    try:
        from ring1.config import load_ring1_config
        from ring1.evolver import Evolver

        r1_config = load_ring1_config(project_root)
        if not r1_config.has_llm_config():
            log.warning("LLM API key not configured — skipping evolution")
            return False

        # Compact context to save tokens: fewer memories.
        memories = memory_store.get_recent(3) if memory_store else []

        # Gather task history and skills for directed evolution.
        task_history = []
        if memory_store:
            try:
                task_history = memory_store.get_by_type("task", limit=5)
            except Exception:
                pass

        skills = []
        if skill_store:
            try:
                skills = skill_store.get_active(15)
            except Exception:
                pass

        # Get persistent error signatures from recent fitness history.
        persistent_errors = []
        try:
            persistent_errors = fitness.get_recent_error_signatures(limit=5)
        except Exception:
            pass

        # Get top genes for inheritance.
        genes = []
        if gene_pool:
            try:
                genes = gene_pool.get_top(3)
            except Exception:
                pass

        evolver = Evolver(r1_config, fitness, memory_store=memory_store)
        result = evolver.evolve(
            ring2_path=ring2_path,
            generation=generation,
            params=params_to_dict(params),
            survived=survived,
            directive=directive,
            memories=memories,
            task_history=task_history,
            skills=skills,
            crash_logs=crash_logs,
            persistent_errors=persistent_errors,
            is_plateaued=is_plateaued,
            gene_pool=genes,
        )
        if result.success:
            log.info("Evolution succeeded: %s", result.reason)
            return True
        else:
            log.warning("Evolution failed: %s", result.reason)
            if notifier:
                notifier.notify_error(generation, result.reason)
            return False
    except Exception as exc:
        log.error("Evolution error (non-fatal): %s", exc)
        if notifier:
            notifier.notify_error(generation, str(exc))
        return False


def _try_crystallize(project_root, skill_store, source_code, output, generation, skill_cap=100, registry_client=None):
    """Best-effort crystallization.  Returns action string or None."""
    try:
        from ring1.config import load_ring1_config
        from ring1.crystallizer import Crystallizer

        r1_config = load_ring1_config(project_root)
        if not r1_config.has_llm_config():
            log.warning("LLM API key not configured — skipping crystallization")
            return None

        crystallizer = Crystallizer(r1_config, skill_store)
        result = crystallizer.crystallize(
            source_code=source_code,
            output=output,
            generation=generation,
            skill_cap=skill_cap,
        )
        log.info("Crystallization result: action=%s skill=%s reason=%s",
                 result.action, result.skill_name, result.reason)

        # Auto-publish to registry on successful crystallization.
        if result.action in ("create", "update") and result.skill_name and registry_client:
            try:
                skill_data = skill_store.get_by_name(result.skill_name)
                if skill_data:
                    registry_client.publish(
                        name=skill_data["name"],
                        description=skill_data.get("description", ""),
                        prompt_template=skill_data.get("prompt_template", ""),
                        parameters=skill_data.get("parameters"),
                        tags=skill_data.get("tags"),
                        source_code=skill_data.get("source_code", ""),
                    )
                    log.info("Published skill %r to registry", result.skill_name)
            except Exception as pub_exc:
                log.debug("Registry publish failed (non-fatal): %s", pub_exc)

        return result.action
    except Exception as exc:
        log.error("Crystallization error (non-fatal): %s", exc)
        return None


def _best_effort(label, factory):
    """Run *factory*; return its result or None on any error."""
    try:
        return factory()
    except Exception as exc:
        log.debug("%s not available: %s", label, exc)
        return None


def _create_notifier(project_root):
    """Best-effort Telegram notifier creation.  Returns None on any error."""
    def _factory():
        from ring1.config import load_ring1_config
        from ring1.telegram import create_notifier
        return create_notifier(load_ring1_config(project_root))
    return _best_effort("Telegram notifier", _factory)


def _create_bot(project_root, state, fitness, ring2_path):
    """Best-effort Telegram bot creation.  Returns None on any error."""
    def _factory():
        from ring1.config import load_ring1_config
        from ring1.telegram_bot import create_bot, start_bot_thread
        bot = create_bot(load_ring1_config(project_root), state, fitness, ring2_path)
        if bot:
            start_bot_thread(bot)
            log.info("Telegram bot started")
        return bot
    return _best_effort("Telegram bot", _factory)


def _create_registry_client(project_root, cfg):
    """Best-effort RegistryClient creation.  Returns None on any error."""
    def _factory():
        from ring1.registry_client import RegistryClient
        reg_cfg = cfg.get("registry", {})
        if not reg_cfg.get("enabled", False):
            return None
        url = reg_cfg.get("url", "https://protea-hub-production.up.railway.app")
        import socket
        node_id = reg_cfg.get("node_id", "default")
        if node_id == "default":
            node_id = socket.gethostname()
        client = RegistryClient(url, node_id)
        log.info("RegistryClient created (url=%s, node_id=%s)", url, node_id)
        return client
    return _best_effort("RegistryClient", _factory)


def _create_portal(project_root, cfg, skill_store, skill_runner):
    """Best-effort Skill Portal creation.  Returns None on any error."""
    def _factory():
        from ring1.skill_portal import create_portal, start_portal_thread
        portal = create_portal(skill_store, skill_runner, project_root, cfg)
        if portal:
            start_portal_thread(portal)
            log.info("Skill Portal started")
        return portal
    return _best_effort("Skill Portal", _factory)


def _create_executor(project_root, state, ring2_path, reply_fn, memory_store=None, skill_store=None, skill_runner=None, task_store=None, registry_client=None):
    """Best-effort task executor creation.  Returns None on any error."""
    def _factory():
        from ring1.config import load_ring1_config
        from ring1.task_executor import create_executor, start_executor_thread
        r1_config = load_ring1_config(project_root)
        executor = create_executor(r1_config, state, ring2_path, reply_fn, memory_store=memory_store, skill_store=skill_store, skill_runner=skill_runner, task_store=task_store, registry_client=registry_client)
        if executor:
            thread = start_executor_thread(executor)
            state.executor_thread = thread
            log.info("Task executor started")
        return executor
    return _best_effort("Task executor", _factory)


class Sentinel:
    """Orchestrates the Ring 0 main loop — launch, supervise, evolve."""

    def __init__(self, project_root: pathlib.Path) -> None:
        self._project_root = project_root

        # -- Config --
        self._cfg = _load_config(project_root)
        self._r0cfg = _load_ring0_config(project_root, self._cfg)
        self._r0cfg.db_path.parent.mkdir(parents=True, exist_ok=True)

        # -- Stores --
        self._git = GitManager(self._r0cfg.ring2_path)
        self._git.init_repo()
        self._fitness = FitnessTracker(self._r0cfg.db_path)
        self._memory_store = _best_effort("MemoryStore", lambda: MemoryStore(self._r0cfg.db_path))
        self._skill_store = _best_effort("SkillStore", lambda: SkillStore(self._r0cfg.db_path))
        self._gene_pool = _best_effort("GenePool", lambda: GenePool(self._r0cfg.db_path))
        self._task_store = _best_effort("TaskStore", lambda: TaskStore(self._r0cfg.db_path))

        # -- Services (lazy imports for optional ring1 deps) --
        from ring1.task_generator import TaskGenerator
        self._task_generator = TaskGenerator(
            base_level=1, adjustment_window=10,
            upgrade_threshold=0.90, downgrade_threshold=0.65,
        )
        from ring1.auto_crystallizer import AutoCrystallizer
        self._auto_crystallizer = AutoCrystallizer(
            skills_dir=project_root / "skills",
            min_stability=0.80, min_score=0.85, min_occurrences=5,
        )
        self._hb = HeartbeatMonitor(
            self._r0cfg.heartbeat_path,
            timeout_sec=self._r0cfg.heartbeat_timeout_sec,
        )
        self._notifier = _create_notifier(project_root)

        # -- Shared state for Telegram bot --
        from ring1.telegram_bot import SentinelState
        self._state = SentinelState()
        self._state.notifier = self._notifier

        def _make_skill_runner():
            from ring1.skill_runner import SkillRunner
            return SkillRunner()
        self._skill_runner = _best_effort("SkillRunner", _make_skill_runner)

        self._state.memory_store = self._memory_store
        self._state.skill_store = self._skill_store
        self._state.skill_runner = self._skill_runner
        self._state.task_store = self._task_store

        self._bot = _create_bot(
            project_root, self._state, self._fitness, self._r0cfg.ring2_path,
        )

        # -- Registry client --
        self._registry_client = _create_registry_client(project_root, self._cfg)
        self._state.registry_client = self._registry_client

        # Evict stale hub skills on startup.
        if self._skill_store:
            evicted = self._skill_store.evict_stale()
            if evicted:
                log.info("Evicted %d stale hub skills", evicted)

        # Backfill gene pool from existing skills (one-time).
        if self._gene_pool and self._skill_store:
            try:
                backfilled = self._gene_pool.backfill(self._skill_store)
                if backfilled:
                    log.info("Gene pool backfilled %d genes from skills", backfilled)
            except Exception as exc:
                log.debug("Gene pool backfill failed (non-fatal): %s", exc)

        # -- Task executor --
        reply_fn = self._bot._send_reply if self._bot else (lambda text: None)
        self._executor = _create_executor(
            project_root, self._state, self._r0cfg.ring2_path, reply_fn,
            memory_store=self._memory_store, skill_store=self._skill_store,
            skill_runner=self._skill_runner, task_store=self._task_store,
            registry_client=self._registry_client,
        )
        self._state.subagent_manager = (
            getattr(self._executor, "subagent_manager", None)
            if self._executor else None
        )

        # -- Skill Portal --
        self._portal = _create_portal(
            project_root, self._cfg, self._skill_store, self._skill_runner,
        )

        # -- Commit watcher --
        self._commit_watcher = CommitWatcher(project_root, self._state.restart_event)
        threading.Thread(
            target=self._commit_watcher.run, name="commit-watcher", daemon=True,
        ).start()

        # -- Runtime state --
        self._generation = 0
        self._last_good_hash: str | None = None
        self._last_crystallized_hash: str | None = None
        self._proc: subprocess.Popen | None = None
        self._params = None
        self._start_time = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Sentinel main loop — run until interrupted."""
        def _sigterm_handler(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, _sigterm_handler)

        # Initial snapshot of seed code.
        try:
            self._last_good_hash = self._git.snapshot(
                f"gen-{self._generation} seed",
            )
        except subprocess.CalledProcessError:
            pass

        r0cfg = self._r0cfg
        log.info(
            "Sentinel online — heartbeat every %ds, timeout %ds, cooldown %ds",
            r0cfg.heartbeat_interval_sec, r0cfg.heartbeat_timeout_sec,
            r0cfg.cooldown_sec,
        )

        try:
            self._params = generate_params(self._generation, r0cfg.seed)
            self._proc = _start_ring2(r0cfg.ring2_path, r0cfg.heartbeat_path)
            self._start_time = time.time()
            self._hb.wait_for_heartbeat(startup_timeout=r0cfg.heartbeat_timeout_sec)

            while True:
                self._state.p0_event.wait(timeout=r0cfg.heartbeat_interval_sec)
                self._state.p0_event.clear()

                # --- resource check ---
                ok, msg = check_resources(
                    r0cfg.max_cpu_percent,
                    r0cfg.max_memory_percent,
                    r0cfg.max_disk_percent,
                )
                if not ok:
                    log.warning("Resource alert: %s", msg)

                elapsed = time.time() - self._start_time

                # --- update shared state for bot ---
                with self._state.lock:
                    self._state.generation = self._generation
                    self._state.start_time = self._start_time
                    self._state.alive = self._hb.is_alive()
                    self._state.mutation_rate = self._params.mutation_rate
                    self._state.max_runtime_sec = self._params.max_runtime_sec

                # --- pause check (bot can set this) ---
                if self._state.pause_event.is_set():
                    continue

                # --- kill check (bot can set this) ---
                if self._state.kill_event.is_set():
                    self._state.kill_event.clear()
                    log.info("Kill signal received — restarting Ring 2 (gen-%d)", self._generation)
                    _stop_ring2(self._proc)
                    self._proc = _start_ring2(r0cfg.ring2_path, r0cfg.heartbeat_path)
                    self._start_time = time.time()
                    self._hb.wait_for_heartbeat(startup_timeout=r0cfg.heartbeat_timeout_sec)
                    continue

                # --- restart check (commit watcher sets this) ---
                if self._state.restart_event.is_set():
                    log.info("New commit detected — restarting Protea")
                    break

                # --- success check: survived max_runtime_sec ---
                if elapsed >= self._params.max_runtime_sec and self._hb.is_alive():
                    self._handle_success(elapsed)
                    continue

                # --- heartbeat check ---
                if self._hb.is_alive():
                    continue

                # Ring 2 is dead — failure path.
                self._handle_failure(elapsed)

        except KeyboardInterrupt:
            log.info("Sentinel shutting down (KeyboardInterrupt)")
        finally:
            self._shutdown()

        # Restart the entire process if triggered by CommitWatcher.
        if self._state.restart_event.is_set():
            log.info("Restarting via os.execv()")
            os.execv(sys.executable, [sys.executable] + sys.argv)

    # ------------------------------------------------------------------
    # Internal: success / failure / generation lifecycle
    # ------------------------------------------------------------------

    def _handle_success(self, elapsed: float) -> None:
        """Handle a generation that survived its full runtime."""
        log.info(
            "Ring 2 survived gen-%d (%.1fs >= %ds)",
            self._generation, elapsed, self._params.max_runtime_sec,
        )
        _stop_ring2(self._proc)

        # Read output and score (with novelty from recent fingerprints).
        output = _read_ring2_output(self._proc, max_lines=200)
        output_lines = output.splitlines() if output else []
        recent_fps = []
        try:
            recent_fps = self._fitness.get_recent_fingerprints(limit=10)
        except Exception:
            pass
        score, detail = evaluate_output(
            output_lines, survived=True,
            elapsed=elapsed, max_runtime=self._params.max_runtime_sec,
            recent_fingerprints=recent_fps,
        )

        # Record success.
        commit_hash = self._last_good_hash or "unknown"
        self._fitness.record(
            generation=self._generation,
            commit_hash=commit_hash,
            score=score,
            runtime_sec=elapsed,
            survived=True,
            detail=detail,
        )
        log.info("Fitness score gen-%d: %.4f  detail=%s", self._generation, score, detail)

        with self._state.lock:
            self._state.last_score = score
            self._state.last_survived = True

        # Snapshot the surviving code.
        try:
            self._last_good_hash = self._git.snapshot(
                f"gen-{self._generation} survived",
            )
        except subprocess.CalledProcessError:
            pass
        source = (self._r0cfg.ring2_path / "main.py").read_text()
        if self._memory_store:
            self._memory_store.add(
                self._generation, "observation",
                f"Gen {self._generation} survived {elapsed:.0f}s "
                f"(max {self._params.max_runtime_sec}s). "
                f"Code: {len(source)} bytes.\n"
                f"Output (last 50 lines):\n"
                f"{output[-1000:] if output else '(no output)'}",
            )

        # Store gene in pool (best-effort).
        if self._gene_pool:
            try:
                self._gene_pool.add(self._generation, score, source)
            except Exception as exc:
                log.debug("Gene pool add failed (non-fatal): %s", exc)

        # Crystallize skill (best-effort) — skip if source unchanged.
        if self._skill_store:
            import hashlib
            source_hash = hashlib.sha256(source.encode()).hexdigest()
            if source_hash != self._last_crystallized_hash:
                log.info(
                    "Crystallizing gen-%d (hash=%s…)",
                    self._generation, source_hash[:12],
                )
                _try_crystallize(
                    self._project_root, self._skill_store, source, output,
                    self._generation, skill_cap=self._r0cfg.skill_cap,
                    registry_client=self._registry_client,
                )
                self._last_crystallized_hash = source_hash
            else:
                log.debug(
                    "Skipping crystallization — source unchanged (hash=%s…)",
                    source_hash[:12],
                )

        # Evolve (best-effort) — skip if busy, cooling down, or plateaued.
        evolved = self._try_evolve_step(survived=True)
        if evolved:
            try:
                self._git.snapshot(f"gen-{self._generation} evolved")
            except subprocess.CalledProcessError:
                pass

        # Notify.
        if self._notifier:
            self._notifier.notify_generation_complete(
                self._generation, score, True,
                self._last_good_hash or "unknown",
            )

        # Next generation.
        self._next_generation("Starting")

    def _handle_failure(self, elapsed: float) -> None:
        """Handle a generation that lost its heartbeat."""
        log.warning(
            "Ring 2 lost heartbeat after %.1fs (gen-%d)",
            elapsed, self._generation,
        )
        output = _read_ring2_output(self._proc, max_lines=200)
        _stop_ring2(self._proc)

        failure_reason = _classify_failure(self._proc, output)
        log.warning("Failure reason: %s", failure_reason)

        output_lines = output.splitlines() if output else []
        score, detail = evaluate_output(
            output_lines, survived=False,
            elapsed=elapsed, max_runtime=self._params.max_runtime_sec,
        )
        log.info("Fitness score gen-%d: %.4f  detail=%s", self._generation, score, detail)
        commit_hash = self._last_good_hash or "unknown"
        self._fitness.record(
            generation=self._generation,
            commit_hash=commit_hash,
            score=score,
            runtime_sec=elapsed,
            survived=False,
            detail=detail,
        )

        with self._state.lock:
            self._state.last_score = score
            self._state.last_survived = False

        # Rollback to last known-good code.
        if self._last_good_hash:
            log.info("Rolling back to %s", self._last_good_hash[:12])
            self._git.rollback(self._last_good_hash)

        # Record crash log in memory.
        if self._memory_store:
            self._memory_store.add(
                self._generation, "crash_log",
                f"Gen {self._generation} died after {elapsed:.0f}s.\n"
                f"Reason: {failure_reason}\n\n"
                f"--- Last output ---\n"
                f"{output[-2000:] if output else '(no output)'}",
            )

        # Evolve from the good base (best-effort).
        # Failures always trigger evolution (no plateau skip) to fix the issue.
        evolved = self._try_evolve_step(survived=False)
        if evolved:
            try:
                self._git.snapshot(
                    f"gen-{self._generation} evolved-from-rollback",
                )
            except subprocess.CalledProcessError:
                pass

        # Notify.
        if self._notifier:
            self._notifier.notify_generation_complete(
                self._generation, score, False, commit_hash,
            )

        # Next generation.
        self._next_generation("Restarting Ring 2 —")

    def _try_evolve_step(self, survived: bool) -> bool:
        """Check cooldown/plateau and evolve if appropriate.  Returns True if evolved."""
        with self._state.lock:
            pending_directive = self._state.evolution_directive
        should_evo, plateaued = _should_evolve(
            self._state, self._r0cfg.cooldown_sec, fitness=self._fitness,
            plateau_window=self._r0cfg.plateau_window,
            plateau_epsilon=self._r0cfg.plateau_epsilon,
            has_directive=bool(pending_directive) if survived else True,
        )
        if not should_evo:
            if not (survived and plateaued):
                log.info("Skipping evolution (busy or cooldown)")
            return False

        with self._state.lock:
            directive = self._state.evolution_directive
            self._state.evolution_directive = ""
        if directive and self._memory_store:
            self._memory_store.add(self._generation, "directive", directive)

        crash_logs = []
        if self._memory_store:
            try:
                crash_logs = self._memory_store.get_by_type("crash_log", limit=3)
            except Exception:
                pass

        evolved = _try_evolve(
            self._project_root, self._fitness, self._r0cfg.ring2_path,
            self._generation, self._params, survived, self._notifier,
            directive=directive,
            memory_store=self._memory_store,
            skill_store=self._skill_store,
            crash_logs=crash_logs,
            is_plateaued=plateaued if survived else False,
            gene_pool=self._gene_pool,
        )
        if evolved:
            self._state.last_evolution_time = time.time()
        return evolved

    def _next_generation(self, label: str) -> None:
        """Advance to the next generation and start Ring 2."""
        self._generation += 1
        self._params = generate_params(self._generation, self._r0cfg.seed)
        log.info("%s generation %d (params: %s)", label, self._generation, self._params)
        self._proc = _start_ring2(self._r0cfg.ring2_path, self._r0cfg.heartbeat_path)
        self._start_time = time.time()
        self._hb.wait_for_heartbeat(startup_timeout=self._r0cfg.heartbeat_timeout_sec)

    def _shutdown(self) -> None:
        """Clean shutdown of all services."""
        self._commit_watcher.stop()
        if self._portal:
            self._portal.stop()
        if self._skill_runner and self._skill_runner.is_running():
            self._skill_runner.stop()
        if self._executor:
            self._executor.stop()
        if self._bot:
            self._bot.stop()
        _stop_ring2(self._proc)
        log.info("Sentinel offline")


def run(project_root: pathlib.Path) -> None:
    """Backward-compatible entry point — creates a Sentinel and runs it."""
    sentinel = Sentinel(project_root)
    sentinel.run()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    project_root = pathlib.Path(__file__).resolve().parent.parent
    run(project_root)
