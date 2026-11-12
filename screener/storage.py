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
        """Return only snapshots whose manifest, row count and canonical hash agree.

        A manifest is a commit marker, not proof that the committed rows remain
        intact forever. Validate the complete snapshot before exposing any of its
        rows to historical scenario rules.
        """
        if self._history_cache is None:
            rows_by_snapshot: dict[str, list[dict[str, Any]]] = {}
            invalid_snapshots: set[str] = set()
            for item in self._read_jsonl(self.market_rows_path):
                if item.get("record_type") != "market_row":
                    continue
                snapshot_id = str(item.get("snapshot_id") or "")
                symbol = str(item.get("symbol") or "")
                data = item.get("data")
                if not snapshot_id:
                    continue
                if not symbol or not isinstance(data, dict) or str(data.get("symbol") or "") != symbol:
                    invalid_snapshots.add(snapshot_id)
                    continue
                rows_by_snapshot.setdefault(snapshot_id, []).append(item)

            manifest_counts: dict[str, int] = {}
            for manifest in self._snapshot_manifests():
                snapshot_id = str(manifest.get("snapshot_id") or "")
                if snapshot_id:
                    manifest_counts[snapshot_id] = manifest_counts.get(snapshot_id, 0) + 1

            index: dict[str, list[dict[str, Any]]] = {}
            for manifest in self._snapshot_manifests():
                snapshot_id = str(manifest.get("snapshot_id") or "")
                if not snapshot_id:
                    continue
                if manifest_counts.get(snapshot_id) != 1 or snapshot_id in invalid_snapshots:
                    log.error("Ignoring snapshot %s: missing/duplicate manifest or malformed row", snapshot_id)
                    continue
                contract_type = str(manifest.get("contract_type") or "").upper()
                fetched_at = manifest.get("fetched_at")
                expected_count = manifest.get("row_count")
                expected_hash = str(manifest.get("snapshot_hash") or "")
                try:
                    datetime.fromisoformat(str(fetched_at))
                    expected_count = int(expected_count)
                except (TypeError, ValueError):
                    expected_count = -1
                snapshot_rows = rows_by_snapshot.get(snapshot_id, [])
                symbols = [str(item.get("symbol") or "") for item in snapshot_rows]
                row_data = [item["data"] for item in snapshot_rows]
                valid = (
                    bool(contract_type)
                    and expected_count >= 0
                    and len(snapshot_rows) == expected_count
                    and len(set(symbols)) == len(symbols)
                    and bool(expected_hash)
                    and self._canonical_snapshot(contract_type, row_data) == expected_hash
                )
                if not valid:
                    log.error(
                        "Ignoring snapshot %s: integrity check failed (rows=%d expected=%s)",
                        snapshot_id, len(snapshot_rows), expected_count,
                    )
                    continue
                for item in snapshot_rows:
                    symbol = str(item["symbol"])
                    index.setdefault(symbol, []).append({
                        **item["data"],
                        "snapshot_at": fetched_at,
                        "contract_type": contract_type,
                    })

            for records in index.values():
                records.sort(key=lambda row: str(row.get("snapshot_at", "")))
            self._history_cache = index
        return self._history_cache

    def snapshot_unchanged(self, contract_type: str, rows: list[MarketRow]) -> bool:
        # Compare against the latest observation of this same contract universe.
        # A more recent ALL_CONTRACTS snapshot must not make unchanged PERPETUAL
        # data look new (or vice versa) and thereby bypass the entry freshness gate.
        expected_contract = str(contract_type).upper()
        manifests = [
            item for item in self._snapshot_manifests()
            if str(item.get("contract_type") or "").upper() == expected_contract
        ]
        if not manifests:
            return False
        previous = manifests[-1]
        return previous.get("snapshot_hash") == self._canonical_snapshot(contract_type, rows)

    def snapshot_due(
        self,
        fetched_at: datetime,
        *,
        interval_hours: float = 4.0,
        tolerance_minutes: int = 10,
        contract_type: str | None = None,
    ) -> bool:
        """Return true only after a complete interval since the last snapshot.

        Coinalyze 4H values are rolling windows. UTC-slot changes alone do not
        make two observations independent: two polls on opposite sides of a
        slot boundary can still be seconds apart. ``tolerance_minutes`` is kept
        for call compatibility; it is used by sequence validation as a late
        polling allowance, never as an early-save allowance.
        """
        manifests = self._snapshot_manifests()
        normalized_contract = str(contract_type).upper() if contract_type else None
        if normalized_contract:
            manifests = [
                item for item in manifests
                if str(item.get("contract_type") or "").upper() == normalized_contract
            ]
        if not manifests:
            return True
        try:
            interval_seconds = float(interval_hours) * 3600.0
            if not math.isfinite(interval_seconds) or interval_seconds <= 0:
                raise ValueError("interval_hours must be positive and finite")
            previous_at = datetime.fromisoformat(str(manifests[-1].get("fetched_at")))
        except (TypeError, ValueError):
            log.warning("Latest snapshot has invalid timestamp or interval; allowing a fresh snapshot")
            return True

        current_at = fetched_at if fetched_at.tzinfo else fetched_at.replace(tzinfo=timezone.utc)
        previous_at = previous_at if previous_at.tzinfo else previous_at.replace(tzinfo=timezone.utc)
        current_ts = current_at.astimezone(timezone.utc).timestamp()
        previous_ts = previous_at.astimezone(timezone.utc).timestamp()
        if current_ts < previous_ts:
            log.warning("Coinalyze fetched_at moved backwards; refusing to create an out-of-order snapshot")
            return False
        elapsed_seconds = current_ts - previous_ts
        return elapsed_seconds >= interval_seconds

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

    def recent_symbol_history(
        self, symbol: str, limit: int = 24, *, contract_type: str | None = None
    ) -> list[dict[str, Any]]:
        records = self._history_index().get(symbol, [])
        if contract_type:
            expected = str(contract_type).upper()
            records = [row for row in records if str(row.get("contract_type") or "").upper() == expected]
        return records[-max(1, limit):]

    def spot_sequence_status(
        self,
        symbol: str,
        required: int = 3,
        interval_hours: float = 4.0,
        tolerance_minutes: int = 10,
        *,
        as_of: datetime | None = None,
        contract_type: str | None = None,
    ) -> bool | None:
        """Evaluate the latest consecutive positive Spot observations only.

        An older positive subsequence cannot survive a newer negative reading.
        A missing, stale, out-of-order or poorly-spaced latest observation is
        UNKNOWN (None), not evidence of accumulation.
        """
        if required < 1:
            return None
        history = self.recent_symbol_history(
            symbol, limit=max(100, required * 12), contract_type=contract_type
        )
        if len(history) < required:
            return None

        target_seconds = float(interval_hours) * 3600.0
        tolerance_seconds = max(0, int(tolerance_minutes)) * 60.0
        if not math.isfinite(target_seconds) or target_seconds <= 0:
            return None

        selected = history[-required:]
        parsed: list[tuple[datetime, float | None]] = []
        for item in selected:
            try:
                timestamp = datetime.fromisoformat(str(item["snapshot_at"]))
            except (KeyError, TypeError, ValueError):
                return None
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            value = item.get("spot_imbalance_4h")
            try:
                number = float(value) if value is not None else None
            except (TypeError, ValueError):
                number = None
            if number is not None and not math.isfinite(number):
                number = None
            parsed.append((timestamp.astimezone(timezone.utc), number))

        now = as_of or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        latest_at, latest_value = parsed[-1]
        age_seconds = (now.astimezone(timezone.utc) - latest_at).total_seconds()
        if age_seconds < -300 or age_seconds > target_seconds + tolerance_seconds:
            return None
        # A fresh, known non-positive latest reading terminates the sequence,
        # regardless of whether older samples were positive.
        if latest_value is not None and latest_value <= 0:
            return False
        if latest_value is None:
            return None

        for (earlier_at, _), (later_at, _) in zip(parsed, parsed[1:]):
            gap = (later_at - earlier_at).total_seconds()
            if not target_seconds <= gap <= target_seconds + tolerance_seconds:
                return None
        if any(value is None for _, value in parsed):
            return None
        return all(value is not None and value > 0 for _, value in parsed)

    def migrate_notification_recipient_ids(self, key_function: Any) -> int:
        """Replace legacy raw Telegram chat IDs with one-way recipient keys."""
        records = list(self._read_jsonl(self.notifications_path))
        migrated = 0
        for item in records:
            raw_chat = item.get("chat_id")
            if raw_chat is None or item.get("recipient_key"):
                continue
            item["recipient_key"] = str(key_function(str(raw_chat)))
            item.pop("chat_id", None)
            migrated += 1
        if migrated:
            self._rewrite_jsonl(self.notifications_path, records)
        return migrated

    def migrate_redacted_broadcast_notifications(self, recipient_keys: list[str]) -> int:
        """Assign explicitly redacted legacy broadcast rows to current recipients.

        The packaged legacy state may have its original recipient removed for
        privacy. On the first configured run, materialize one recipient-scoped
        copy per current recipient, then remove the broadcast marker.
        """
        keys = list(dict.fromkeys(str(value) for value in recipient_keys if str(value)))
        if not keys:
            return 0
        if len(keys) != 1:
            log.warning(
                "Cannot safely assign redacted legacy notification state to %d recipients; "
                "configure the original recipient alone for the migration run",
                len(keys),
            )
            return 0
        records = list(self._read_jsonl(self.notifications_path))
        migrated: list[dict[str, Any]] = []
        expanded = 0
        changed = False
        for item in records:
            if item.get("recipient_scope") != "LEGACY_REDACTED_BROADCAST":
                migrated.append(item)
                continue
            changed = True
            base = dict(item)
            base.pop("recipient_scope", None)
            base.pop("chat_id", None)
            # A Telegram message ID is scoped to one private chat. The redacted
            # legacy row cannot tell us which recipient owns that ID.
            base.pop("telegram_message_id", None)
            for key in keys:
                copy = dict(base)
                copy["recipient_key"] = key
                migrated.append(copy)
                expanded += 1
        if changed:
            self._rewrite_jsonl(self.notifications_path, migrated)
        return expanded

    def _notification_records(
        self, event_key: str, chat_id: str | None = None, *, recipient_key: str | None = None
    ) -> list[dict[str, Any]]:
        records = [
            item for item in self._read_jsonl(self.notifications_path)
            if item.get("event_key") == event_key
        ]
        recipient = recipient_key if recipient_key is not None else chat_id
        if recipient is None:
            return records
        recipient = str(recipient)
        # Rows without a recipient key are legacy broadcast records.
        return [
            item for item in records
            if (item.get("recipient_key") is None and item.get("chat_id") is None)
            or str(item.get("recipient_key") or item.get("chat_id")) == recipient
        ]

    def notification_exists(
        self, event_key: str, chat_id: str | None = None, *, recipient_key: str | None = None
    ) -> bool:
        return bool(self._notification_records(event_key, chat_id, recipient_key=recipient_key))

    def latest_notification_state(
        self, event_group: str, chat_id: str | None = None, *, recipient_key: str | None = None
    ) -> dict[str, Any] | None:
        """Return the latest state for one recipient key (or legacy broadcast)."""
        recipient = recipient_key if recipient_key is not None else chat_id
        matching = [
            item for item in self._read_jsonl(self.notifications_path)
            if item.get("event_group") == event_group
            and item.get("fingerprint")
            and (
                recipient is None
                or (item.get("recipient_key") is None and item.get("chat_id") is None)
                or str(item.get("recipient_key") or item.get("chat_id")) == str(recipient)
            )
        ]
        return matching[-1] if matching else None

    def notification_due(
        self, event_key: str, cooldown_hours: float, chat_id: str | None = None,
        *, recipient_key: str | None = None,
    ) -> bool:
        matching = [
            item for item in self._notification_records(event_key, chat_id, recipient_key=recipient_key)
            if item.get("sent_at")
        ]
        if not matching:
            return True
        last_sent: datetime | None = None
        for item in matching:
            try:
                candidate = datetime.fromisoformat(str(item["sent_at"]))
            except (KeyError, TypeError, ValueError):
                continue
            if candidate.tzinfo is None:
                candidate = candidate.replace(tzinfo=timezone.utc)
            if last_sent is None or candidate > last_sent:
                last_sent = candidate
        if last_sent is None:
            return True
        return datetime.now(timezone.utc) - last_sent >= timedelta(hours=cooldown_hours)

    def record_notification(
        self,
        event_key: str,
        message_id: str | int | None = None,
        chat_id: str | None = None,
        *,
        recipient_key: str | None = None,
        event_group: str | None = None,
        fingerprint: str | None = None,
        event_state: str | None = None,
        direction: str | None = None,
        invalidation_level: float | None = None,
        pair: str | None = None,
        scenario_code: str | None = None,
        contract_type: str | None = None,
        delivery_status: str = "SENT",
        error: str | None = None,
    ) -> None:
        """Append a delivery and optional lifecycle-state record.

        Production callers pass ``recipient_key`` (an HMAC pseudonym). ``chat_id``
        remains only as a backwards-compatible input for old tests/migrations.
        """
        record: dict[str, Any] = {
            "record_type": "notification",
            "event_key": event_key,
            "sent_at": datetime.now(timezone.utc).isoformat(),
            "telegram_message_id": str(message_id) if message_id is not None else None,
            "delivery_status": str(delivery_status).upper(),
        }
        if recipient_key is not None:
            record["recipient_key"] = str(recipient_key)
        elif chat_id is not None:
            record["chat_id"] = str(chat_id)
        if event_group is not None:
            record["event_group"] = str(event_group)
        if fingerprint is not None:
            record["fingerprint"] = str(fingerprint)
        if event_state is not None:
            record["event_state"] = str(event_state).upper()
        if direction is not None:
            record["direction"] = str(direction).upper()
        if invalidation_level is not None and math.isfinite(float(invalidation_level)):
            record["invalidation_level"] = float(invalidation_level)
        if pair is not None:
            record["pair"] = str(pair).upper()
        if scenario_code is not None:
            record["scenario_code"] = str(scenario_code).upper()
        if contract_type is not None:
            record["contract_type"] = str(contract_type).upper()
        if error:
            record["error"] = str(error)[:500]
        self._append_jsonl(self.notifications_path, record)

    def active_notification_states(
        self,
        *,
        chat_ids: list[str] | None = None,
        recipient_keys: list[str] | None = None,
        contract_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the latest active alert per scenario group and recipient key.

        ``chat_ids`` is retained as a compatibility alias for stored recipient
        keys. Legacy broadcast rows are expanded into the requested recipient set.
        """
        allowed = recipient_keys if recipient_keys is not None else chat_ids
        allowed_recipients = {str(value) for value in allowed} if allowed is not None else None
        expected_contract = str(contract_type).upper() if contract_type else None
        latest_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for item in self._read_jsonl(self.notifications_path):
            group = str(item.get("event_group") or "")
            if not group or not item.get("fingerprint"):
                continue
            stored_recipient = item.get("recipient_key") or item.get("chat_id")
            if item.get("recipient_scope") == "LEGACY_REDACTED_BROADCAST" and stored_recipient is None:
                # The original recipient was intentionally removed from the
                # public archive. Never broadcast its lifecycle to unrelated
                # recipients; the migration method assigns it only when one
                # unambiguous configured recipient is available.
                log.warning(
                    "Skipping legacy notification %s: recipient is redacted and cannot be mapped safely",
                    item.get("event_key"),
                )
                continue
            target_recipients = [str(stored_recipient or "")]
            if allowed_recipients is not None and stored_recipient is not None and str(stored_recipient) not in allowed_recipients:
                continue
            stored_contract = item.get("contract_type")
            if expected_contract and stored_contract and str(stored_contract).upper() != expected_contract:
                continue
            for recipient in target_recipients:
                latest_by_key[(group, recipient)] = {
                    **item,
                    "recipient_key": recipient,
                    # Kept in the returned in-memory shape for lifecycle compatibility;
                    # this field is never written back by production code.
                    "chat_id": recipient,
                }

        active: list[dict[str, Any]] = []
        for item in latest_by_key.values():
            state = str(item.get("event_state") or "ACTIVE").upper()
            delivery = str(item.get("delivery_status") or "SENT").upper()
            if state == "ACTIVE" and delivery in {"SENT", "UNKNOWN"}:
                active.append(dict(item))
        return active

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
        # Keep the latest ACTIVE/UNKNOWN-or-SENT lifecycle row for every
        # scenario-recipient pair even after the ordinary notification retention
        # period. Otherwise pruning would silently forget a still-active signal.
        latest_state: dict[tuple[str, str], dict[str, Any]] = {}
        for item in notifications:
            group = str(item.get("event_group") or "")
            if not group or not item.get("fingerprint"):
                continue
            recipient = str(item.get("recipient_key") or item.get("chat_id") or "")
            latest_state[(group, recipient)] = item
        protected_ids = {
            id(item) for item in latest_state.values()
            if str(item.get("event_state") or "ACTIVE").upper() == "ACTIVE"
            and str(item.get("delivery_status") or "SENT").upper() in {"SENT", "UNKNOWN"}
        }
        kept_notifications: list[dict[str, Any]] = []
        for item in notifications:
            try:
                timestamp = datetime.fromisoformat(str(item["sent_at"]))
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
            except (KeyError, TypeError, ValueError):
                continue
            if timestamp >= notification_cutoff or id(item) in protected_ids:
                kept_notifications.append(item)
        deleted_notifications = len(notifications) - len(kept_notifications)
        if deleted_notifications:
            self._rewrite_jsonl(self.notifications_path, kept_notifications)

        return {
            "snapshots_deleted": deleted_snapshots,
            "runs_deleted": deleted_runs,
            "notifications_deleted": deleted_notifications,
        }
