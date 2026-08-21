from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.runtime_maintenance import (
    MAINTENANCE_HOLD_TOKEN_ENV,
    engage_maintenance_hold,
    maintenance_hold_snapshot,
    maintenance_hold_token_authorized,
    release_maintenance_hold,
)


def test_maintenance_hold_engage_expire_and_release(tmp_path: Path, monkeypatch) -> None:
    hold_path = tmp_path / "runtime_maintenance.flag"
    monkeypatch.setenv("RUNTIME_MAINTENANCE_HOLD_PATH", str(hold_path))

    engaged = engage_maintenance_hold(
        tmp_path,
        reason="sqlite_local_failover",
        owner="test",
        ttl_seconds=600,
    )

    assert engaged["active"] is True
    assert engaged["reason"] == "sqlite_local_failover"
    assert engaged["token"]
    assert maintenance_hold_token_authorized(engaged, token=engaged["token"]) is True
    assert maintenance_hold_token_authorized(engaged, token="wrong-token") is False
    monkeypatch.setenv(MAINTENANCE_HOLD_TOKEN_ENV, engaged["token"])
    assert maintenance_hold_token_authorized(engaged) is True
    expires_at = datetime.fromisoformat(engaged["expires_at_utc"])
    expired = maintenance_hold_snapshot(tmp_path, now_utc=expires_at + timedelta(seconds=1))
    assert expired["active"] is False
    assert expired["expired"] is True

    mismatch = release_maintenance_hold(tmp_path, expected_token="wrong-token")
    assert mismatch["released"] is False
    assert hold_path.exists()
    released = release_maintenance_hold(tmp_path, expected_token=engaged["token"])
    assert released["released"] is True
    assert released["active"] is False
    assert not hold_path.exists()


def test_unreadable_maintenance_hold_fails_closed(tmp_path: Path, monkeypatch) -> None:
    hold_path = tmp_path / "runtime_maintenance.flag"
    hold_path.write_text("not-json", encoding="utf-8")
    monkeypatch.setenv("RUNTIME_MAINTENANCE_HOLD_PATH", str(hold_path))

    snapshot = maintenance_hold_snapshot(tmp_path, now_utc=datetime.now(timezone.utc))

    assert snapshot["active"] is True
    assert snapshot["valid"] is False
    assert snapshot["reason"] == "unreadable_maintenance_hold_fail_closed"
