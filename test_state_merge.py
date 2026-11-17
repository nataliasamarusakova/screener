from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from screener.merge_state import merge_state
from screener.models import MarketRow
from screener.storage import Store


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n" for item in records))


def _snapshot(snapshot_id: str, fetched_at: str, symbol: str, name: str) -> tuple[dict, dict]:
    row = MarketRow(symbol=symbol, name=name, price=1.0, chg24_pct=1.0).to_dict()
    row.pop("raw", None)
    manifest = {
        "record_type": "snapshot",
        "snapshot_id": snapshot_id,
        "fetched_at": fetched_at,
        "contract_type": "PERPETUAL",
        "row_count": 1,
        "snapshot_hash": Store._canonical_snapshot("PERPETUAL", [row]),
    }
    market_row = {
        "record_type": "market_row", "snapshot_id": snapshot_id,
        "symbol": symbol, "data": row,
    }
    return manifest, market_row


def test_merge_state_preserves_concurrent_snapshots_notifications_and_latest_poll(tmp_path):
    remote, local, target = tmp_path / "remote", tmp_path / "local", tmp_path / "target"
    remote_manifest, remote_row = _snapshot("remote-1", "2026-11-16T10:00:00+00:00", "BTCUSDT_PERP", "Bitcoin")
    local_manifest, local_row = _snapshot("local-1", "2026-11-16T10:05:00+00:00", "BTCUSDT_PERP", "Bitcoin")
    remote_notification = {
        "record_type": "notification", "event_key": "event-1", "sent_at": "2026-11-16T10:01:00+00:00",
        "delivery_status": "SENT", "recipient_key": "recipient", "contract_type": "PERPETUAL",
    }
    local_notification = {
        "record_type": "notification", "event_key": "event-2", "sent_at": "2026-11-16T10:06:00+00:00",
        "delivery_status": "SENT", "recipient_key": "recipient", "contract_type": "PERPETUAL",
    }
    _write_jsonl(remote / "snapshots.jsonl", [remote_manifest])
    _write_jsonl(remote / "market_rows.jsonl", [remote_row])
    _write_jsonl(remote / "notifications.jsonl", [remote_notification])
    (remote / "poll_state.json").write_text(json.dumps({
        "schema_version": 1,
        "contracts": {"PERPETUAL": {"snapshot_hash": "old", "polled_at": "2026-11-16T10:00:00+00:00"}},
    }))

    _write_jsonl(local / "snapshots.jsonl", [local_manifest])
    _write_jsonl(local / "market_rows.jsonl", [local_row])
    _write_jsonl(local / "notifications.jsonl", [local_notification])
    (local / "poll_state.json").write_text(json.dumps({
        "schema_version": 1,
        "contracts": {"PERPETUAL": {"snapshot_hash": "new", "polled_at": "2026-11-16T10:05:00+00:00"}},
    }))
    target.mkdir()
    for file in ("snapshots.jsonl", "market_rows.jsonl", "notifications.jsonl", "poll_state.json"):
        (target / file).write_bytes((remote / file).read_bytes())

    merge_state(local, target)

    store = Store(target, create_files=False)
    history = store.recent_symbol_history("BTCUSDT_PERP", limit=5, contract_type="PERPETUAL")
    assert len(history) == 2
    assert [item["_snapshot_id"] for item in history] == ["remote-1", "local-1"]
    notifications = [json.loads(line) for line in (target / "notifications.jsonl").read_text().splitlines()]
    assert [item["event_key"] for item in notifications] == ["event-1", "event-2"]
    poll_state = json.loads((target / "poll_state.json").read_text())
    assert poll_state["contracts"]["PERPETUAL"]["snapshot_hash"] == "new"

    # Retrying persistence after an uncertain git-push result must be idempotent.
    first_snapshot_bytes = (target / "snapshots.jsonl").read_bytes()
    first_row_bytes = (target / "market_rows.jsonl").read_bytes()
    first_notification_bytes = (target / "notifications.jsonl").read_bytes()
    merge_state(local, target)
    assert (target / "snapshots.jsonl").read_bytes() == first_snapshot_bytes
    assert (target / "market_rows.jsonl").read_bytes() == first_row_bytes
    assert (target / "notifications.jsonl").read_bytes() == first_notification_bytes


def test_merge_state_keeps_latest_binance_cooldown_from_either_runner(tmp_path):
    remote, local, target = tmp_path / "remote", tmp_path / "local", tmp_path / "target"
    remote.mkdir()
    local.mkdir()
    target.mkdir()
    for folder, cooldown in ((remote, "2026-11-17T09:10:00+00:00"), (local, "2026-11-17T09:15:00+00:00")):
        (folder / "poll_state.json").write_text(json.dumps({
            "schema_version": 1,
            "contracts": {},
            "binance_cooldown_until": cooldown,
        }))
    (target / "poll_state.json").write_bytes((remote / "poll_state.json").read_bytes())

    merge_state(local, target)

    merged = json.loads((target / "poll_state.json").read_text())
    assert merged["binance_cooldown_until"] == "2026-11-17T09:15:00+00:00"


def test_merge_state_rejects_conflicting_duplicate_snapshot_manifest(tmp_path):
    remote, local, target = tmp_path / "remote", tmp_path / "local", tmp_path / "target"
    manifest, row = _snapshot("same-id", "2026-11-16T10:00:00+00:00", "BTCUSDT_PERP", "Bitcoin")
    conflicting = {**manifest, "fetched_at": "2026-11-16T10:05:00+00:00"}
    _write_jsonl(remote / "snapshots.jsonl", [manifest])
    _write_jsonl(remote / "market_rows.jsonl", [row])
    _write_jsonl(local / "snapshots.jsonl", [conflicting])
    _write_jsonl(local / "market_rows.jsonl", [row])
    _write_jsonl(remote / "notifications.jsonl", [])
    _write_jsonl(local / "notifications.jsonl", [])
    target.mkdir()
    for file in ("snapshots.jsonl", "market_rows.jsonl", "notifications.jsonl"):
        (target / file).write_bytes((remote / file).read_bytes())

    with pytest.raises(ValueError, match="Conflicting snapshot manifests"):
        merge_state(local, target)


def test_merge_state_discards_uncommitted_market_rows_without_snapshot_manifest(tmp_path):
    remote, local, target = tmp_path / "remote", tmp_path / "local", tmp_path / "target"
    manifest, row = _snapshot("known-id", "2026-11-16T10:00:00+00:00", "BTCUSDT_PERP", "Bitcoin")
    _write_jsonl(remote / "snapshots.jsonl", [manifest])
    _write_jsonl(remote / "market_rows.jsonl", [row])
    _write_jsonl(remote / "notifications.jsonl", [])
    _write_jsonl(local / "snapshots.jsonl", [])
    orphan = {**row, "snapshot_id": "orphan-id"}
    _write_jsonl(local / "market_rows.jsonl", [orphan])
    _write_jsonl(local / "notifications.jsonl", [])
    target.mkdir()
    for file in ("snapshots.jsonl", "market_rows.jsonl", "notifications.jsonl"):
        (target / file).write_bytes((remote / file).read_bytes())

    merge_state(local, target)
    merged_rows = [json.loads(line) for line in (target / "market_rows.jsonl").read_text().splitlines()]
    assert [item["snapshot_id"] for item in merged_rows] == ["known-id"]


def test_merge_state_rejects_conflicting_poll_states_with_same_timestamp(tmp_path):
    remote, local, target = tmp_path / "remote", tmp_path / "local", tmp_path / "target"
    remote.mkdir()
    local.mkdir()
    target.mkdir()
    timestamp = "2026-11-16T10:00:00+00:00"
    (remote / "poll_state.json").write_text(json.dumps({
        "schema_version": 1,
        "contracts": {"PERPETUAL": {"snapshot_hash": "hash-a", "polled_at": timestamp}},
    }))
    (local / "poll_state.json").write_text(json.dumps({
        "schema_version": 1,
        "contracts": {"PERPETUAL": {"snapshot_hash": "hash-b", "polled_at": timestamp}},
    }))
    (target / "poll_state.json").write_bytes((remote / "poll_state.json").read_bytes())

    with pytest.raises(ValueError, match="same polled_at timestamp"):
        merge_state(local, target)


def test_merge_state_fails_closed_on_unterminated_notification_tail(tmp_path):
    remote, local, target = tmp_path / "remote", tmp_path / "local", tmp_path / "target"
    manifest, row = _snapshot("committed", "2026-11-16T10:00:00+00:00", "BTCUSDT_PERP", "Bitcoin")
    _write_jsonl(remote / "snapshots.jsonl", [manifest])
    _write_jsonl(remote / "market_rows.jsonl", [row])
    _write_jsonl(remote / "notifications.jsonl", [])
    (local / "snapshots.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (local / "snapshots.jsonl").write_bytes(b"{\"partial\":")
    (local / "market_rows.jsonl").write_bytes(b"")
    (local / "notifications.jsonl").write_bytes(b"{\"record_type\":\"notification\"")
    target.mkdir()
    for file in ("snapshots.jsonl", "market_rows.jsonl", "notifications.jsonl"):
        (target / file).write_bytes((remote / file).read_bytes())

    with pytest.raises(ValueError, match="Invalid JSONL record"):
        merge_state(local, target)

    # Snapshot/row readers can still discard an unterminated tail; notification
    # markers deliberately use strict_tail=True because they guard external sends.
    from screener.merge_state import _read_jsonl
    assert _read_jsonl(local / "snapshots.jsonl") == []
    assert _read_jsonl(local / "market_rows.jsonl") == []
    assert (target / "notifications.jsonl").read_text() == ""


def test_merge_state_fails_on_malformed_jsonl_in_middle_of_file(tmp_path):
    remote, local, target = tmp_path / "remote", tmp_path / "local", tmp_path / "target"
    for folder in (remote, local, target):
        folder.mkdir()
    valid, _row = _snapshot("valid", "2026-11-16T10:00:00+00:00", "BTCUSDT_PERP", "Bitcoin")
    (remote / "snapshots.jsonl").write_text(json.dumps(valid) + "\nnot-json\n")
    for folder in (remote, local):
        (folder / "market_rows.jsonl").write_text("")
        (folder / "notifications.jsonl").write_text("")
    (local / "snapshots.jsonl").write_text("")
    for file in ("snapshots.jsonl", "market_rows.jsonl", "notifications.jsonl"):
        (target / file).write_bytes((remote / file).read_bytes())

    with pytest.raises(ValueError, match="Invalid JSONL record"):
        merge_state(local, target)


def test_pruning_after_state_merge_removes_expired_remote_snapshots(tmp_path):
    from datetime import datetime, timedelta, timezone

    remote, local, target = tmp_path / "remote-old", tmp_path / "local-new", tmp_path / "target-pruned"
    old_at = (datetime.now(timezone.utc) - timedelta(days=45)).isoformat()
    new_at = datetime.now(timezone.utc).isoformat()
    old_manifest, old_row = _snapshot("old-remote", old_at, "BTCUSDT_PERP", "Bitcoin")
    new_manifest, new_row = _snapshot("new-local", new_at, "BTCUSDT_PERP", "Bitcoin")
    _write_jsonl(remote / "snapshots.jsonl", [old_manifest])
    _write_jsonl(remote / "market_rows.jsonl", [old_row])
    _write_jsonl(remote / "notifications.jsonl", [])
    _write_jsonl(local / "snapshots.jsonl", [new_manifest])
    _write_jsonl(local / "market_rows.jsonl", [new_row])
    _write_jsonl(local / "notifications.jsonl", [])
    target.mkdir()
    for name in ("snapshots.jsonl", "market_rows.jsonl", "notifications.jsonl"):
        (target / name).write_bytes((remote / name).read_bytes())

    merge_state(local, target)
    assert len(Store(target, create_files=False).recent_symbol_history(
        "BTCUSDT_PERP", limit=10, contract_type="PERPETUAL"
    )) == 2

    Store(target, create_files=False).prune(snapshot_retention_days=30)
    history = Store(target, create_files=False).recent_symbol_history(
        "BTCUSDT_PERP", limit=10, contract_type="PERPETUAL"
    )
    assert [item["_snapshot_id"] for item in history] == ["new-local"]
