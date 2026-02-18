"""Progress Monitor — Automatic progress reporting for long-running operations.

Wraps tool execution to track duration and send periodic updates for operations
that take longer than expected.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

log = logging.getLogger("protea.progress_monitor")

# Configuration
PROGRESS_THRESHOLD_SEC = 10  # Send update if operation takes longer than this
PROGRESS_INTERVAL_SEC = 30   # Send periodic updates at this interval


class ProgressMonitor:
    """Monitors tool execution and sends automatic progress updates."""

    def __init__(self, reply_fn: Callable[[str], None]) -> None:
        """
        Args:
            reply_fn: Function to send progress messages (e.g., Telegram).
        """
        self.reply_fn = reply_fn
        self._active_operations: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._monitor_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start the background monitoring thread."""
        if self._monitor_thread is not None:
            return
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="progress-monitor",
            daemon=True,
        )
        self._monitor_thread.start()
        log.info("Progress monitor started")

    def stop(self) -> None:
        """Stop the background monitoring thread."""
        if self._monitor_thread is None:
            return
        self._stop_event.set()
        self._monitor_thread.join(timeout=2)
        self._monitor_thread = None
        log.info("Progress monitor stopped")

    def wrap_tool(
        self,
        tool_name: str,
        tool_fn: Callable,
        description: str = "",
    ) -> Callable:
        """Wrap a tool function with progress monitoring.

        Args:
            tool_name: Name of the tool (for progress messages).
            tool_fn: The actual tool function to wrap.
            description: Optional description for progress messages.

        Returns:
            Wrapped function that reports progress automatically.
        """
        def wrapped(*args, **kwargs) -> Any:
            op_id = f"{tool_name}-{time.time()}"
            start_time = time.time()

            # Track this operation
            with self._lock:
                self._active_operations[op_id] = {
                    "tool": tool_name,
                    "description": description or tool_name,
                    "start": start_time,
                    "last_update": start_time,
                }

            try:
                result = tool_fn(*args, **kwargs)
                duration = time.time() - start_time

                # Send completion notice if it took a while
                if duration > PROGRESS_THRESHOLD_SEC:
                    self._send_update(
                        f"✅ Completed: {tool_name} ({duration:.1f}s)"
                    )

                return result
            finally:
                # Remove from tracking
                with self._lock:
                    self._active_operations.pop(op_id, None)

        return wrapped

    def _monitor_loop(self) -> None:
        """Background loop that checks for long-running operations."""
        while not self._stop_event.wait(5):  # Check every 5 seconds
            try:
                self._check_operations()
            except Exception:
                log.error("Progress monitor error", exc_info=True)

    def _check_operations(self) -> None:
        """Check all active operations and send updates for slow ones."""
        now = time.time()
        updates_to_send = []

        with self._lock:
            for op_id, op in list(self._active_operations.items()):
                elapsed = now - op["start"]
                since_update = now - op["last_update"]

                # Send initial "still working" message if past threshold
                if elapsed > PROGRESS_THRESHOLD_SEC and since_update >= PROGRESS_THRESHOLD_SEC:
                    op["last_update"] = now
                    updates_to_send.append(
                        f"🔄 Still working: {op['description']} ({elapsed:.0f}s elapsed)"
                    )
                # Send periodic updates for very long operations
                elif since_update >= PROGRESS_INTERVAL_SEC:
                    op["last_update"] = now
                    updates_to_send.append(
                        f"🔄 In progress: {op['description']} ({elapsed:.0f}s elapsed)"
                    )

        # Send updates outside the lock
        for msg in updates_to_send:
            self._send_update(msg)

    def _send_update(self, message: str) -> None:
        """Send a progress update message."""
        try:
            self.reply_fn(f"[Progress] {message}")
        except Exception:
            log.error("Failed to send progress update", exc_info=True)


def create_monitored_tool(
    tool_name: str,
    tool_fn: Callable,
    monitor: ProgressMonitor,
    description: str = "",
) -> Callable:
    """Helper to create a progress-monitored tool.

    Args:
        tool_name: Name of the tool.
        tool_fn: The tool function.
        monitor: ProgressMonitor instance.
        description: Optional description.

    Returns:
        Wrapped tool function.
    """
    return monitor.wrap_tool(tool_name, tool_fn, description)
