"""Durable record of final RiskGuard/dispatch status for strong candidates."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from contracts import SignalEvent
from engine.provenance import provenance


def append_dispatch_events(
    path: Path,
    signals: Iterable[SignalEvent],
    *,
    allowed: bool,
    block_reason: str | None = None,
    dispatched_at_ms: int | None = None,
    dispatched_signal_ids: set[str] | None = None,
) -> int:
    rows = []
    sent_ids = dispatched_signal_ids or set()
    now_ms = int(dispatched_at_ms or 0)
    for sig in signals:
        if sig.signal_type not in ("STRONG_LONG", "STRONG_SHORT"):
            continue
        signal_id = f"{sig.symbol}:{sig.timestamp_ms}:{sig.signal_type}"
        rows.append({
            "signal_id": signal_id,
            "symbol": sig.symbol,
            "timestamp_ms": int(sig.timestamp_ms),
            "dispatch_allowed": bool(allowed),
            "dispatch_block_reason": None if allowed else block_reason,
            "dispatched": signal_id in sent_ids,
            "dispatched_at_ms": now_ms if signal_id in sent_ids else None,
            **provenance(),
        })
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return len(rows)
