"""Append-only signal ledger for forward-outcome validation."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

INTERVAL_MS = 5 * 60 * 1000

from contracts import SignalEvent
from engine.serialization import signal_to_dict
from engine.provenance import provenance


def append_signal_events(path: Path, signals: Iterable[SignalEvent]) -> int:
    existing_ids: set[str] = set()
    if path.exists():
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    signal_id = str(json.loads(line).get("signal_id", ""))
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                if signal_id:
                    existing_ids.add(signal_id)
        except OSError:
            existing_ids = set()

    rows = []
    for sig in signals:
        if sig.signal_type not in ("STRONG_LONG", "STRONG_SHORT"):
            continue
        signal_id = f"{sig.symbol}:{sig.timestamp_ms}:{sig.signal_type}"
        if signal_id in existing_ids:
            continue
        candle_open_ms = (int(sig.timestamp_ms) // INTERVAL_MS) * INTERVAL_MS
        payload = signal_to_dict(sig)
        payload.update({
            "signal_id": signal_id,
            "candle_open_ms": candle_open_ms,
            "candle_close_ms": candle_open_ms + INTERVAL_MS - 1,
            "applied_friction_rt_pct": sig.applied_friction_rt_pct,
            **provenance(),
        })
        rows.append(payload)

    if not rows:
        return 0

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return len(rows)
