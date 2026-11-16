#!/usr/bin/env python3
"""Safely merge append-only scanner state from a runner with branch state.

Run with ``python -m screener.merge_state --help``.

The scanner's JSONL files are append-oriented. Git's line-based merge can
conflict when two independent runs append to the same file, so this helper
merges records by their domain identity and validates snapshot integrity before
writing the combined state. It never discards conflicting records silently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

JSONL_FILES = ("snapshots.jsonl", "market_rows.jsonl", "notifications.jsonl")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines = raw.splitlines()
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            # Storage appends a JSONL record in one batch but a process can be
            # killed during the final write. Match Store._repair_trailing_line:
            # discard only a malformed, unterminated tail; corruption in the
            # middle of a file must still stop persistence.
            is_unterminated_tail = index == len(lines) and not raw.endswith(b"\n")
            if is_unterminated_tail:
                break
            raise ValueError(f"Invalid JSONL record {path}:{index}: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"JSONL record must be an object: {path}:{index}")
        records.append(item)
    return records


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _record_time(record: dict[str, Any], field: str, source: str) -> datetime:
    raw = record.get(field)
    try:
        value = datetime.fromisoformat(str(raw))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field!r} timestamp in {source}: {raw!r}") from exc
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _merge_snapshots(remote: list[dict[str, Any]], local: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for record in [*remote, *local]:
        snapshot_id = str(record.get("snapshot_id") or "")
        if not snapshot_id:
            raise ValueError("Snapshot manifest is missing snapshot_id")
        previous = by_id.get(snapshot_id)
        if previous is not None and _canonical(previous) != _canonical(record):
            raise ValueError(f"Conflicting snapshot manifests for snapshot_id={snapshot_id}")
        by_id[snapshot_id] = record
    return sorted(
        by_id.values(),
        key=lambda row: (_record_time(row, "fetched_at", "snapshots.jsonl"), str(row["snapshot_id"])),
    )


def _snapshot_hash(contract_type: str, rows: list[dict[str, Any]]) -> str:
    normalized = []
    for row in rows:
        item = dict(row)
        item.pop("raw", None)
        normalized.append(item)
    normalized.sort(key=lambda item: (str(item.get("symbol", "")), _canonical(item)))
    payload = {"contract_type": contract_type, "rows": normalized}
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _merge_market_rows(
    remote: list[dict[str, Any]], local: list[dict[str, Any]], manifests: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_snapshot_symbol: dict[tuple[str, str], dict[str, Any]] = {}
    order_by_record: dict[tuple[str, str], int] = {}
    for source_index, record in enumerate([*remote, *local]):
        snapshot_id = str(record.get("snapshot_id") or "")
        symbol = str(record.get("symbol") or "")
        data = record.get("data")
        if not snapshot_id or not symbol or not isinstance(data, dict) or str(data.get("symbol") or "") != symbol:
            raise ValueError("Invalid market_rows.jsonl record; refusing to merge partial or malformed state")
        key = (snapshot_id, symbol)
        previous = by_snapshot_symbol.get(key)
        if previous is not None and _canonical(previous) != _canonical(record):
            raise ValueError(f"Conflicting market rows for snapshot={snapshot_id}, symbol={symbol}")
        if previous is None:
            by_snapshot_symbol[key] = record
            order_by_record[key] = source_index

    manifest_ids = {str(item["snapshot_id"]) for item in manifests}
    orphaned_snapshots = {snapshot_id for snapshot_id, _symbol in by_snapshot_symbol} - manifest_ids
    if orphaned_snapshots:
        # Store writes all market rows first and the manifest last. A killed
        # process can leave a complete-but-uncommitted row batch; discard it,
        # as Store._history_index() does, rather than blocking notifications.
        for key in [key for key in by_snapshot_symbol if key[0] in orphaned_snapshots]:
            by_snapshot_symbol.pop(key, None)
            order_by_record.pop(key, None)

    manifest_order = {str(item["snapshot_id"]): idx for idx, item in enumerate(manifests)}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for (snapshot_id, _symbol), record in by_snapshot_symbol.items():
        grouped.setdefault(snapshot_id, []).append(record)

    # Preserve source order inside a snapshot; order distinct snapshots by their
    # manifest timestamps so per-symbol history remains chronological.
    merged: list[dict[str, Any]] = []
    ordered_snapshot_ids = sorted(
        grouped,
        key=lambda sid: (manifest_order.get(sid, len(manifest_order)), sid),
    )
    for snapshot_id in ordered_snapshot_ids:
        merged.extend(sorted(
            grouped[snapshot_id],
            key=lambda record: order_by_record[(snapshot_id, str(record["symbol"]))],
        ))

    rows_by_id: dict[str, list[dict[str, Any]]] = {}
    for record in merged:
        rows_by_id.setdefault(str(record["snapshot_id"]), []).append(record["data"])
    for manifest in manifests:
        snapshot_id = str(manifest["snapshot_id"])
        rows = rows_by_id.get(snapshot_id, [])
        expected_count = int(manifest.get("row_count", -1))
        contract_type = str(manifest.get("contract_type") or "")
        expected_hash = str(manifest.get("snapshot_hash") or "")
        if len(rows) != expected_count:
            raise ValueError(
                f"Snapshot {snapshot_id} has {len(rows)} merged rows; manifest expects {expected_count}"
            )
        if not contract_type or not expected_hash or _snapshot_hash(contract_type, rows) != expected_hash:
            raise ValueError(f"Merged rows fail snapshot hash validation for snapshot_id={snapshot_id}")
    return merged


def _merge_notifications(remote: list[dict[str, Any]], local: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, tuple[int, dict[str, Any]]] = {}
    for index, record in enumerate([*remote, *local]):
        identity = _canonical(record)
        unique.setdefault(identity, (index, record))
    records = list(unique.values())
    records.sort(key=lambda pair: (_record_time(pair[1], "sent_at", "notifications.jsonl"), pair[0]))
    return [record for _, record in records]


def _read_poll_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "contracts": {}}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable poll state file: {path}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("contracts"), dict):
        raise ValueError(f"Invalid poll state schema: {path}")
    return value


def _merge_poll_state(remote_path: Path, local_path: Path) -> str:
    remote = _read_poll_state(remote_path)
    local = _read_poll_state(local_path)
    contracts: dict[str, dict[str, Any]] = {}
    for source in (remote.get("contracts", {}), local.get("contracts", {})):
        for contract, state in source.items():
            if not isinstance(state, dict):
                raise ValueError(f"Invalid poll state for contract {contract!r}")
            previous = contracts.get(str(contract))
            if previous is None:
                contracts[str(contract)] = dict(state)
                continue
            new_at = _record_time({"polled_at": state.get("polled_at")}, "polled_at", "poll_state.json")
            old_at = _record_time({"polled_at": previous.get("polled_at")}, "polled_at", "poll_state.json")
            if new_at == old_at and _canonical(state) != _canonical(previous):
                raise ValueError(
                    f"Conflicting poll states for contract {contract!r} at the same polled_at timestamp"
                )
            if new_at > old_at:
                contracts[str(contract)] = dict(state)
    return _canonical({"schema_version": 1, "contracts": contracts}) + "\n"


def merge_state(source_dir: Path, target_dir: Path) -> None:
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    source_snapshots = _read_jsonl(source_dir / "snapshots.jsonl")
    target_snapshots = _read_jsonl(target_dir / "snapshots.jsonl")
    manifests = _merge_snapshots(target_snapshots, source_snapshots)

    source_rows = _read_jsonl(source_dir / "market_rows.jsonl")
    target_rows = _read_jsonl(target_dir / "market_rows.jsonl")
    market_rows = _merge_market_rows(target_rows, source_rows, manifests)

    source_notifications = _read_jsonl(source_dir / "notifications.jsonl")
    target_notifications = _read_jsonl(target_dir / "notifications.jsonl")
    notifications = _merge_notifications(target_notifications, source_notifications)

    source_poll = source_dir / "poll_state.json"
    target_poll = target_dir / "poll_state.json"
    poll_state_content = None
    if source_poll.exists() or target_poll.exists():
        # Validate and prepare every state file before replacing any target.
        poll_state_content = _merge_poll_state(target_poll, source_poll)

    _write_atomic(target_dir / "snapshots.jsonl", "".join(_canonical(row) + "\n" for row in manifests))
    _write_atomic(target_dir / "market_rows.jsonl", "".join(_canonical(row) + "\n" for row in market_rows))
    _write_atomic(target_dir / "notifications.jsonl", "".join(_canonical(row) + "\n" for row in notifications))
    if poll_state_content is not None:
        _write_atomic(target_poll, poll_state_content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Runner state captured before git reset")
    parser.add_argument("--target", required=True, type=Path, help="Latest branch data directory")
    args = parser.parse_args()
    merge_state(args.source, args.target)


if __name__ == "__main__":
    main()
