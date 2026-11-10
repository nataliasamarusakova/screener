from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from .models import MarketRow

log = logging.getLogger(__name__)


class Store:
    """Append-only JSONL storage for snapshots, market rows, notifications and runs.

    A snapshot manifest is appended only after all of its market-row records have
    been written. The manifest therefore acts as the commit marker: incomplete
    row batches left by an interrupted process are ignored by history queries.
    GitHub Actions serializes scanner runs and commits data/ JSONL files to the
    workflow's current branch, so no dedicated state branch is required.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_path = self.data_dir / "snapshots.jsonl"
        self.market_rows_path = self.data_dir / "market_rows.jsonl"
        self.notifications_path = self.data_dir / "notifications.jsonl"
        self.runs_path = self.data_dir / "runs.jsonl"
        for path in self.paths:
            path.touch(exist_ok=True)
        self._manifests_cache: list[dict[str, Any]] | None = None
        self._history_cache: dict[str, list[dict[str, Any]]] | None = None

    @property
    def paths(self) -> tuple[Path, Path, Path, Path]:
        return (
            self.snapshots_path,
            self.market_rows_path,
            self.notifications_path,
            self.runs_path,
        )

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {str(key): Store._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [Store._json_safe(item) for item in value]
        return value

    @staticmethod
    def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("Skipping malformed JSONL record: %s:%d", path.name, line_number)
                    continue
                if isinstance(item, dict):
                    yield item

    @staticmethod
    def _repair_trailing_line(path: Path) -> None:
        """Repair a truncated final record left by a process killed mid-append."""
        if not path.exists() or path.stat().st_size == 0:
            return
        with path.open("r+b") as stream:
            stream.seek(-1, os.SEEK_END)
            if stream.read(1) == b"\n":
                return
            stream.seek(0)
            content = stream.read()
            boundary = content.rfind(b"\n")
            tail = content[boundary + 1:]
            # A valid final JSON object without a newline is preserved and
            # terminated; malformed trailing bytes are discarded only to the
            # previous complete line boundary.
            try:
                json.loads(tail.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                stream.truncate(boundary + 1 if boundary >= 0 else 0)
            else:
                stream.seek(0, os.SEEK_END)
                stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _append_jsonl_many(path: Path, payloads: Iterable[dict[str, Any]]) -> None:
        lines = []
        for payload in payloads:
            lines.append(json.dumps(
                Store._json_safe(payload), ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False,
            ))
        if not lines:
            return
        data = ("\n".join(lines) + "\n").encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        Store._repair_trailing_line(path)
        # O_APPEND keeps a batch contiguous; one fsync per snapshot avoids a
        # filesystem sync for every coin in a typical 80–300 row universe.
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            view = memoryview(data)
            while view:
                written = os.write(fd, view)
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
        Store._append_jsonl_many(path, [payload])

    @staticmethod
    def _rewrite_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                for record in records:
                    stream.write(json.dumps(
                        Store._json_safe(record), ensure_ascii=False, sort_keys=True,
                        separators=(",", ":"), allow_nan=False,
                    ))
                    stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, path)
        except Exception:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

    @staticmethod
    def _utc_iso(value: datetime) -> str:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()

    @staticmethod
    def _row_payload(row: MarketRow | dict[str, Any]) -> dict[str, Any]:
        payload = row.to_dict() if isinstance(row, MarketRow) else dict(row)
        payload.pop("raw", None)  # DOM diagnostics are not historical market data.
        return Store._json_safe(payload)

    @classmethod
    def _canonical_snapshot(cls, contract_type: str, rows: list[MarketRow] | list[dict[str, Any]]) -> str:
        normalized = [cls._row_payload(row) for row in rows]
        normalized.sort(key=lambda item: (str(item.get("symbol", "")), json.dumps(item, sort_keys=True, ensure_ascii=False)))
        canonical = json.dumps(
            {"contract_type": contract_type, "rows": normalized},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _snapshot_manifests(self) -> list[dict[str, Any]]:
        if self._manifests_cache is None:
            self._manifests_cache = [
                r for r in self._read_jsonl(self.snapshots_path)
                if r.get("record_type") == "snapshot"
            ]
        return self._manifests_cache

    def _history_index(self) -> dict[str, list[dict[str, Any]]]:
        if self._history_cache is None:
            committed = {
                item["snapshot_id"]: item.get("fetched_at")
                for item in self._snapshot_manifests() if item.get("snapshot_id")
            }
            index: dict[str, list[dict[str, Any]]] = {}
            for item in self._read_jsonl(self.market_rows_path):
                snapshot_id = item.get("snapshot_id")
                symbol = item.get("symbol")
                data = item.get("data")
                if snapshot_id not in committed or not symbol or not isinstance(data, dict):
                    continue
                index.setdefault(str(symbol), []).append({
                    **data,
                    "snapshot_at": committed[snapshot_id],
                })
            for records in index.values():
                records.sort(key=lambda row: str(row.get("snapshot_at", "")))
            self._history_cache = index
        return self._history_cache

    def snapshot_unchanged(self, contract_type: str, rows: list[MarketRow]) -> bool:
        manifests = self._snapshot_manifests()
        if not manifests:
            return False
        previous = manifests[-1]
        if previous.get("contract_type") != contract_type:
            return False
        return previous.get("snapshot_hash") == self._canonical_snapshot(contract_type, rows)

    def add_snapshot(self, fetched_at: datetime, contract_type: str, rows: list[MarketRow]) -> str:
        snapshot_id = uuid.uuid4().hex
        timestamp = self._utc_iso(fetched_at)
        normalized_rows = [self._row_payload(row) for row in rows]

        # Write row records first. The manifest below is the commit marker; if the
        # process stops mid-write, uncommitted records are ignored and later pruned.
        row_records = [{
            "record_type": "market_row",
            "snapshot_id": snapshot_id,
            "symbol": row.get("symbol"),
            "data": row,
        } for row in normalized_rows]
        self._append_jsonl_many(self.market_rows_path, row_records)

        self._append_jsonl(self.snapshots_path, {
            "record_type": "snapshot",
            "snapshot_id": snapshot_id,
            "fetched_at": timestamp,
            "contract_type": contract_type,
            "row_count": len(normalized_rows),
            "snapshot_hash": self._canonical_snapshot(contract_type, normalized_rows),
        })
        self._manifests_cache = None
        self._history_cache = None
        return snapshot_id

    def recent_symbol_history(self, symbol: str, limit: int = 24) -> list[dict[str, Any]]:
        records = self._history_index().get(symbol, [])
        return records[-max(1, limit):]

    def spot_sequence_status(
        self, symbol: str, required: int = 3, interval_hours: float = 4.0, tolerance_minutes: int = 10
    ) -> bool | None:
        """Find positive Spot readings spaced by distinct, committed 4H snapshots."""
        history = self.recent_symbol_history(symbol, limit=max(100, required * 12))
        if len(history) < required:
            return None
        target = timedelta(hours=interval_hours)
        tolerance = timedelta(minutes=tolerance_minutes)
        times: list[tuple[datetime, float | None]] = []
        for item in history:
            try:
                ts = datetime.fromisoformat(item["snapshot_at"])
            except (KeyError, TypeError, ValueError):
                continue
            times.append((ts, item.get("spot_imbalance_4h")))
        if len(times) < required:
            return None
        for end in range(len(times) - 1, required - 2, -1):
            selected = [times[end]]
            cursor = end
            while len(selected) < required:
                target_time = selected[-1][0] - target
                options = [
                    (idx, item) for idx, item in enumerate(times[:cursor])
                    if abs(item[0] - target_time) <= tolerance
                ]
                if not options:
                    break
                idx, item = min(options, key=lambda pair: abs(pair[1][0] - target_time))
                selected.append(item)
                cursor = idx
            if len(selected) == required:
                return all(value is not None and value > 0 for _, value in selected)
        return None

    def _notification_records(self, event_key: str, chat_id: str | None = None) -> list[dict[str, Any]]:
        records = [
            item for item in self._read_jsonl(self.notifications_path)
            if item.get("event_key") == event_key
        ]
        if chat_id is None:
            return records
        recipient = str(chat_id)
        # Records written before recipient-scoped tracking represented a fully
        # successful broadcast, so keep treating them as success for every chat.
        return [
            item for item in records
            if item.get("chat_id") is None or str(item.get("chat_id")) == recipient
        ]

    def notification_exists(self, event_key: str, chat_id: str | None = None) -> bool:
        return bool(self._notification_records(event_key, chat_id))

    def notification_due(
        self, event_key: str, cooldown_hours: float, chat_id: str | None = None
    ) -> bool:
        matching = [
            item for item in self._notification_records(event_key, chat_id)
            if item.get("sent_at")
        ]
        if not matching:
            return True
        last_sent = datetime.fromisoformat(str(matching[-1]["sent_at"]))
        if last_sent.tzinfo is None:
            last_sent = last_sent.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) - last_sent >= timedelta(hours=cooldown_hours)

    def record_notification(
        self, event_key: str, message_id: str | int | None = None,
        chat_id: str | None = None,
    ) -> None:
        record = {
            "record_type": "notification",
            "event_key": event_key,
            "sent_at": datetime.now(timezone.utc).isoformat(),
            "telegram_message_id": str(message_id) if message_id is not None else None,
        }
        if chat_id is not None:
            record["chat_id"] = str(chat_id)
        self._append_jsonl(self.notifications_path, record)

    def start_run(self) -> str:
        run_id = uuid.uuid4().hex
        self._append_jsonl(self.runs_path, {
            "record_type": "run_started",
            "run_id": run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
        })
        return run_id

    def finish_run(
        self, run_id: str, row_count: int, result_count: int,
        status: str = "OK", error: str | None = None,
    ) -> None:
        self._append_jsonl(self.runs_path, {
            "record_type": "run_finished",
            "run_id": run_id,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "row_count": int(row_count),
            "result_count": int(result_count),
            "status": status,
            "error": error,
        })

    def prune(
        self,
        *,
        snapshot_retention_days: int = 30,
        run_retention_days: int = 90,
        notification_retention_days: int = 180,
    ) -> dict[str, int]:
        """Compact expired JSONL history without changing retained observations."""
        now = datetime.now(timezone.utc)
        snapshot_cutoff = now - timedelta(days=max(9, snapshot_retention_days))
        run_cutoff = now - timedelta(days=max(1, run_retention_days))
        notification_cutoff = now - timedelta(days=max(1, notification_retention_days))

        manifests = self._snapshot_manifests()
        kept_manifests: list[dict[str, Any]] = []
        for item in manifests:
            try:
                timestamp = datetime.fromisoformat(str(item["fetched_at"]))
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
            except (KeyError, TypeError, ValueError):
                continue
            if timestamp >= snapshot_cutoff:
                kept_manifests.append(item)

        deleted_snapshots = len(manifests) - len(kept_manifests)
        keep_ids = {str(item.get("snapshot_id")) for item in kept_manifests if item.get("snapshot_id")}
        all_rows = list(self._read_jsonl(self.market_rows_path))
        kept_rows = [item for item in all_rows if str(item.get("snapshot_id")) in keep_ids]
        # Also discard orphan rows from a process interrupted before its snapshot
        # manifest was committed, even when no retention deadline has elapsed.
        if len(kept_rows) != len(all_rows):
            self._rewrite_jsonl(self.market_rows_path, kept_rows)
            self._history_cache = None
        if deleted_snapshots:
            self._rewrite_jsonl(self.snapshots_path, kept_manifests)
            self._manifests_cache = kept_manifests
            self._history_cache = None

        runs = list(self._read_jsonl(self.runs_path))
        kept_runs: list[dict[str, Any]] = []
        for item in runs:
            timestamp_value = item.get("completed_at") or item.get("started_at")
            try:
                timestamp = datetime.fromisoformat(str(timestamp_value))
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
            except (TypeError, ValueError):
                continue
            if timestamp >= run_cutoff:
                kept_runs.append(item)
        deleted_runs = len(runs) - len(kept_runs)
        if deleted_runs:
            self._rewrite_jsonl(self.runs_path, kept_runs)

        notifications = list(self._read_jsonl(self.notifications_path))
        kept_notifications: list[dict[str, Any]] = []
        for item in notifications:
            try:
                timestamp = datetime.fromisoformat(str(item["sent_at"]))
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
            except (KeyError, TypeError, ValueError):
                continue
            if timestamp >= notification_cutoff:
                kept_notifications.append(item)
        deleted_notifications = len(notifications) - len(kept_notifications)
        if deleted_notifications:
            self._rewrite_jsonl(self.notifications_path, kept_notifications)

        return {
            "snapshots_deleted": deleted_snapshots,
            "runs_deleted": deleted_runs,
            "notifications_deleted": deleted_notifications,
        }
