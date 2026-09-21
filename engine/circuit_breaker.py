"""
Circuit breaker: halts signal generation after consecutive scan failures.
Prevents runaway alerts from a broken pipeline.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path

logger = logging.getLogger("circuit_breaker")


class CircuitBreaker:
    """
    Tracks consecutive scan failures. When threshold is exceeded, halts all
    alerts for `halt_duration_minutes`. Auto-resets on first successful scan.
    """

    def __init__(
        self,
        state_file: Path,
        max_consecutive_failures: int = 5,
        halt_duration_minutes: float = 30.0,
    ) -> None:
        if max_consecutive_failures < 1:
            raise ValueError("max_consecutive_failures must be >= 1")
        if halt_duration_minutes <= 0.0:
            raise ValueError("halt_duration_minutes must be positive")
        self.state_file = state_file
        self.max_failures = max_consecutive_failures
        self.halt_ms = int(halt_duration_minutes * 60_000)
        self.state = self._load()

    def _load(self) -> dict:
        if not self.state_file.exists():
            return {"consecutive_failures": 0, "halted_until_ms": 0, "total_failures": 0}
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return {"consecutive_failures": 0, "halted_until_ms": 0, "total_failures": 0}
            return {
                "consecutive_failures": int(data.get("consecutive_failures", 0)),
                "halted_until_ms": int(data.get("halted_until_ms", 0)),
                "total_failures": int(data.get("total_failures", 0)),
            }
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("circuit_breaker_load_failed error=%s", exc)
            return {"consecutive_failures": 0, "halted_until_ms": 0, "total_failures": 0}

    def _save(self) -> None:
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            fd, temp_name = tempfile.mkstemp(
                dir=str(self.state_file.parent), prefix=f".{self.state_file.name}.", suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(self.state, fh, separators=(",", ":"))
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(temp_name, self.state_file)
                dir_fd = os.open(self.state_file.parent, os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            finally:
                try:
                    os.unlink(temp_name)
                except FileNotFoundError:
                    pass
        except OSError as exc:
            logger.error("circuit_breaker_save_failed error=%s", exc)

    def is_halted(self) -> bool:
        return int(time.time() * 1000) < self.state.get("halted_until_ms", 0)

    def halt_remaining_minutes(self) -> float:
        remaining_ms = self.state.get("halted_until_ms", 0) - int(time.time() * 1000)
        return max(0.0, remaining_ms / 60_000.0)

    def record_success(self) -> None:
        if self.state.get("consecutive_failures", 0) > 0:
            self.state["consecutive_failures"] = 0
            self.state["halted_until_ms"] = 0
            self._save()

    def record_failure(self) -> None:
        self.state["consecutive_failures"] = self.state.get("consecutive_failures", 0) + 1
        self.state["total_failures"] = self.state.get("total_failures", 0) + 1
        if self.state["consecutive_failures"] >= self.max_failures:
            self.state["halted_until_ms"] = int(time.time() * 1000) + self.halt_ms
            logger.critical(
                "circuit_breaker_tripped consecutive=%d halt_minutes=%.1f",
                self.state["consecutive_failures"], self.halt_ms / 60_000.0,
            )
        self._save()
