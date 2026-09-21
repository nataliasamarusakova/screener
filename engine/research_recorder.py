"""Durable point-in-time feature recorder for future research datasets."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable


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

    def __init__(self, path: Path = Path("data/research/features.sqlite3")) -> None:
        self.path = path

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
            return int(cur.rowcount if cur.rowcount is not None and cur.rowcount >= 0 else 0)

    def export_jsonl(self, output: Path, start_ms: int | None = None, end_ms: int | None = None) -> int:
        clauses = []
        params: list[Any] = []
        if start_ms is not None:
            clauses.append("timestamp_ms >= ?")
            params.append(int(start_ms))
        if end_ms is not None:
            clauses.append("timestamp_ms <= ?")
            params.append(int(end_ms))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        query = f"SELECT payload_json FROM feature_rows{where} ORDER BY timestamp_ms,symbol"
        output.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        if not self.path.exists():
            output.write_text("", encoding="utf-8")
            return 0
        with sqlite3.connect(self.path) as con, output.open("w", encoding="utf-8") as fh:
            con.executescript(SCHEMA)
            for (payload,) in con.execute(query, params):
                fh.write(payload)
                fh.write("\n")
                count += 1
        return count
