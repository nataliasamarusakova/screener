"""Fail-closed portfolio risk and emergency kill-switch guard."""
from __future__ import annotations

import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Optional


class RiskGuard:
    """Blocks new risk when an explicit kill switch or equity guard is triggered.

    The screener is signal/alert-only, so realized equity must be supplied by an
    execution/reconciliation process through ``data/equity_state.json`` or the
    ``CURRENT_EQUITY_USDT`` environment variable. Missing reconciliation data is
    tolerated only in paper mode; live mode fails closed.
    """

    def __init__(
        self,
        state_file: Path,
        account_equity: float,
        max_drawdown_pct: float = 0.20,
        daily_loss_limit_pct: float = 0.08,
        paper_trading: bool = False,
        kill_switch_file: Optional[Path] = None,
    ) -> None:
        values = (account_equity, max_drawdown_pct, daily_loss_limit_pct)
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError("Risk guard configuration must be finite")
        if account_equity <= 0.0:
            raise ValueError("account_equity must be positive")
        if not (0.0 < max_drawdown_pct < 1.0):
            raise ValueError("max_drawdown_pct must be in (0, 1)")
        if not (0.0 < daily_loss_limit_pct < 1.0):
            raise ValueError("daily_loss_limit_pct must be in (0, 1)")
        self.state_file = state_file
        self.account_equity = float(account_equity)
        self.max_drawdown_pct = float(max_drawdown_pct)
        self.daily_loss_limit_pct = float(daily_loss_limit_pct)
        self.paper_trading = paper_trading
        self.kill_switch_file = kill_switch_file or Path("data/KILL_SWITCH")

    def _load_equity_state(self) -> Optional[dict]:
        current_from_env = os.getenv("CURRENT_EQUITY_USDT")
        if current_from_env:
            try:
                equity = float(current_from_env)
            except ValueError:
                return {"error": "CURRENT_EQUITY_USDT_INVALID"}
            return {
                "equity_usdt": equity,
                "day_start_equity_usdt": self.account_equity,
                "peak_equity_usdt": self.account_equity,
            }
        if not self.state_file.exists():
            return None
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {"error": "EQUITY_STATE_INVALID"}
        except (OSError, ValueError, TypeError):
            return {"error": "EQUITY_STATE_UNREADABLE"}

    def evaluate(self) -> tuple[bool, str]:
        if self.kill_switch_file.exists() or os.getenv("KILL_SWITCH", "0") == "1":
            return False, "KILL_SWITCH_ACTIVE"

        state = self._load_equity_state()
        if state is None:
            if self.paper_trading:
                return True, "PAPER_NO_EQUITY_RECONCILIATION"
            return False, "EQUITY_RECONCILIATION_REQUIRED"

        if "error" in state:
            return False, str(state["error"])

        try:
            equity = float(state["equity_usdt"])
            day_start = float(state.get("day_start_equity_usdt", self.account_equity))
            peak = float(state.get("peak_equity_usdt", max(self.account_equity, equity)))
        except (KeyError, TypeError, ValueError):
            return False, "EQUITY_STATE_INVALID"

        if not all(math.isfinite(v) for v in (equity, day_start, peak)) or equity <= 0.0 or day_start <= 0.0 or peak <= 0.0:
            return False, "EQUITY_STATE_INVALID"

        drawdown = 1.0 - equity / max(peak, self.account_equity)
        daily_loss = 1.0 - equity / day_start
        if drawdown >= self.max_drawdown_pct:
            return False, f"MAX_DRAWDOWN_REACHED:{drawdown:.4f}"
        if daily_loss >= self.daily_loss_limit_pct:
            return False, f"DAILY_LOSS_LIMIT_REACHED:{daily_loss:.4f}"
        return True, "PASSED"

    def write_equity_state(
        self,
        equity_usdt: float,
        *,
        day_start_equity_usdt: Optional[float] = None,
        peak_equity_usdt: Optional[float] = None,
    ) -> None:
        values = [equity_usdt]
        if day_start_equity_usdt is not None:
            values.append(day_start_equity_usdt)
        if peak_equity_usdt is not None:
            values.append(peak_equity_usdt)
        if not all(math.isfinite(float(v)) and float(v) > 0.0 for v in values):
            raise ValueError("equity state values must be finite and positive")

        previous = self._load_equity_state() or {}
        day_start = (
            float(day_start_equity_usdt)
            if day_start_equity_usdt is not None
            else float(previous.get("day_start_equity_usdt", self.account_equity))
        )
        peak = (
            float(peak_equity_usdt)
            if peak_equity_usdt is not None
            else max(float(previous.get("peak_equity_usdt", self.account_equity)), float(equity_usdt))
        )
        payload = {
            "timestamp_ms": int(time.time() * 1000),
            "equity_usdt": float(equity_usdt),
            "day_start_equity_usdt": day_start,
            "peak_equity_usdt": peak,
        }
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(self.state_file.parent), prefix=f".{self.state_file.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, separators=(",", ":"))
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, self.state_file)
        finally:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
