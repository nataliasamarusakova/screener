"""Durable point-in-time feature recorder for future research datasets."""
from __future__ import annotations

import json
import sqlite3
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA = """
CREATE TABLE IF NOT EXISTS feature_rows (
    symbol TEXT NOT NULL,
    timestamp_ms INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY(symbol, timestamp_ms)
);
CREATE INDEX IF NOT EXISTS idx_feature_rows_ts ON feature_rows(timestamp_ms);
"""


class ResearchRecorder:
    """Idempotent SQLite append-only recorder. Existing keys are never overwritten."""

    def __init__(
        self,
        path: Path = Path("data/research/features.sqlite3"),
        shard_dir: Path | None = None,
    ) -> None:
        self.path = path
        self.shard_dir = shard_dir if shard_dir is not None else path.parent / "shards"

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.path)
        con.executescript(SCHEMA)
        return con

    def append_rows(self, rows: Iterable[dict[str, Any]]) -> int:
        materialized = []
        for row in rows:
            symbol = str(row["symbol"]).upper()
            timestamp_ms = int(row["timestamp_ms"])
            payload = dict(row)
            materialized.append((symbol, timestamp_ms, json.dumps(payload, separators=(",", ":"), sort_keys=True)))
        if not materialized:
            return 0
        with self._connect() as con:
            cur = con.executemany(
                "INSERT OR IGNORE INTO feature_rows(symbol,timestamp_ms,payload_json) VALUES(?,?,?)",
                materialized,
            )
            con.commit()
            inserted = int(cur.rowcount if cur.rowcount is not None and cur.rowcount >= 0 else 0)
        self._append_persistent_shards(materialized)
        return inserted

    def _append_persistent_shards(self, materialized: list[tuple[str, int, str]]) -> None:
        """Persist compact hourly JSONL shards so history survives ephemeral runners."""
        grouped: dict[Path, list[tuple[str, int, str]]] = {}
        for symbol, timestamp_ms, payload_json in materialized:
            dt = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
            shard = self.shard_dir / dt.strftime("%Y-%m-%d") / f"{dt:%H}.jsonl"
            grouped.setdefault(shard, []).append((symbol, timestamp_ms, payload_json))

        for shard, rows in grouped.items():
            shard.parent.mkdir(parents=True, exist_ok=True)
            existing: set[tuple[str, int]] = set()
            if shard.exists():
                for line in shard.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                        existing.add((str(payload["symbol"]).upper(), int(payload["timestamp_ms"])))
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                        continue
            with shard.open("a", encoding="utf-8") as fh:
                for symbol, timestamp_ms, payload_json in sorted(rows, key=lambda x: (x[1], x[0])):
                    key = (symbol, timestamp_ms)
                    if key in existing:
                        continue
                    fh.write(payload_json)
                    fh.write("\n")
                    existing.add(key)
                fh.flush()
                os.fsync(fh.fileno())


    def load_recent_histories(
        self,
        *,
        before_timestamp_ms: int,
        symbols: Iterable[str],
        limit: int,
        interval_ms: int = 5 * 60 * 1000,
        expected_provenance: Mapping[str, Any] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Load recent persisted point-in-time rows for state recovery.

        Only a small rolling window of hourly shards is scanned. Rows are strictly
        older than ``before_timestamp_ms`` and are returned in chronological order.
        The caller decides whether the resulting tail is sufficient for a factor.
        """
        wanted = {str(symbol).upper() for symbol in symbols}
        result: dict[str, list[dict[str, Any]]] = {symbol: [] for symbol in wanted}
        if not wanted or limit <= 0 or not self.shard_dir.exists():
            return result

        horizon_ms = max(limit * interval_ms * 2, 6 * 60 * 60 * 1000)
        expected = {str(k): v for k, v in (expected_provenance or {}).items()}

        def provenance_matches(payload: dict[str, Any]) -> bool:
            return all(payload.get(key) == value for key, value in expected.items())
        start_ms = int(before_timestamp_ms) - horizon_ms

        shard_paths = sorted(self.shard_dir.glob("**/*.jsonl"))
        candidates: dict[str, dict[int, dict[str, Any]]] = {symbol: {} for symbol in wanted}
        for shard in shard_paths:
            try:
                year, month, day = (int(part) for part in shard.parent.name.split("-"))
                hour = int(shard.stem)
            except (ValueError, TypeError):
                continue
            shard_dt = datetime(year, month, day, hour, tzinfo=timezone.utc)
            shard_start_ms = int(shard_dt.timestamp() * 1000)
            shard_end_ms = shard_start_ms + 60 * 60 * 1000 - 1
            if shard_end_ms < start_ms or shard_start_ms >= before_timestamp_ms:
                continue
            try:
                lines = shard.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    symbol = str(payload["symbol"]).upper()
                    timestamp_ms = int(payload["timestamp_ms"])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
                if symbol not in wanted or timestamp_ms < start_ms or timestamp_ms >= before_timestamp_ms:
                    continue
                if not provenance_matches(payload):
                    continue
                current = candidates[symbol].get(timestamp_ms)
                if current is None:
                    candidates[symbol][timestamp_ms] = payload

        for symbol, by_ts in candidates.items():
            rows = [by_ts[ts] for ts in sorted(by_ts)]
            if rows:
                contiguous = [rows[-1]]
                previous_ts = int(rows[-1]["timestamp_ms"])
                for row in reversed(rows[:-1]):
                    ts = int(row["timestamp_ms"])
                    if previous_ts - ts != interval_ms:
                        break
                    contiguous.append(row)
                    previous_ts = ts
                rows = list(reversed(contiguous))
            result[symbol] = rows[-limit:]
        return result

    def export_jsonl(self, output: Path, start_ms: int | None = None, end_ms: int | None = None) -> int:
        """Export persisted shards plus any local SQLite rows, deduplicated by symbol/timestamp."""
        records: dict[tuple[str, int], str] = {}

        if self.shard_dir.exists():
            for shard in sorted(self.shard_dir.glob("**/*.jsonl")):
                for line in shard.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                        symbol = str(payload["symbol"]).upper()
                        timestamp_ms = int(payload["timestamp_ms"])
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                        continue
                    if start_ms is not None and timestamp_ms < int(start_ms):
                        continue
                    if end_ms is not None and timestamp_ms > int(end_ms):
                        continue
                    records[(symbol, timestamp_ms)] = json.dumps(payload, separators=(",", ":"), sort_keys=True)

        if self.path.exists():
            with sqlite3.connect(self.path) as con:
                con.executescript(SCHEMA)
                clauses = []
                params: list[Any] = []
                if start_ms is not None:
                    clauses.append("timestamp_ms >= ?")
                    params.append(int(start_ms))
                if end_ms is not None:
                    clauses.append("timestamp_ms <= ?")
                    params.append(int(end_ms))
                where = " WHERE " + " AND ".join(clauses) if clauses else ""
                for payload_json, in con.execute(
                    f"SELECT payload_json FROM feature_rows{where} ORDER BY timestamp_ms,symbol", params
                ):
                    try:
                        payload = json.loads(payload_json)
                        key = (str(payload["symbol"]).upper(), int(payload["timestamp_ms"]))
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                        continue
                    # Persistent hourly shards are authoritative. The local SQLite
                    # cache is only a fallback for keys absent from the shards.
                    records.setdefault(key, payload_json)

        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as fh:
            for key in sorted(records, key=lambda item: (item[1], item[0])):
                fh.write(records[key])
                fh.write("\n")
        return len(records)
