from __future__ import annotations

import json

import pytest

from screener.rearm_failed_notification import rearm_failed_notification


def _record(event_key: str, status: str = "FAILED") -> dict:
    return {
        "record_type": "notification",
        "event_key": event_key,
        "recipient_key": "hmac-sha256:recipient",
        "contract_type": "PERPETUAL",
        "event_state": status,
        "delivery_status": status,
        "sent_at": "2026-11-16T10:00:00+00:00",
    }


def test_rearm_failed_notification_is_read_only_by_default(tmp_path):
    path = tmp_path / "notifications.jsonl"
    path.write_text(json.dumps(_record("FAILED|event-1")) + "\n")

    result = rearm_failed_notification(
        tmp_path, "FAILED|event-1", "hmac-sha256:recipient", "PERPETUAL",
    )

    assert result["applied"] is False
    assert result["matching_records"] == 1
    assert len(path.read_text().splitlines()) == 1
    assert not list(tmp_path.glob("*.bak"))


def test_rearm_failed_notification_removes_only_exact_failed_record_and_backs_up(tmp_path):
    path = tmp_path / "notifications.jsonl"
    rows = [_record("FAILED|event-1"), _record("FAILED|event-2")]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    result = rearm_failed_notification(
        tmp_path, "FAILED|event-1", "hmac-sha256:recipient", "PERPETUAL", apply=True,
    )

    remaining = [json.loads(line) for line in path.read_text().splitlines()]
    backups = list(tmp_path.glob("notifications.jsonl.before-rearm-*.bak"))
    assert result["applied"] is True
    assert [row["event_key"] for row in remaining] == ["FAILED|event-2"]
    assert len(backups) == 1
    assert len(backups[0].read_text().splitlines()) == 2


def test_rearm_failed_notification_refuses_sent_or_unknown_events(tmp_path):
    path = tmp_path / "notifications.jsonl"
    rows = [_record("event-1"), _record("event-1", status="SENT")]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(ValueError, match="not exclusively terminal FAILED"):
        rearm_failed_notification(
            tmp_path, "event-1", "hmac-sha256:recipient", "PERPETUAL", apply=True,
        )
    assert len(path.read_text().splitlines()) == 2
    assert not list(tmp_path.glob("*.bak"))
