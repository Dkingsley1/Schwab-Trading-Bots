from datetime import datetime, timezone

from core.fx_twelve_data_guard import (
    classify_twelve_data_failure,
    mark_twelve_data_cooldown,
    twelve_data_cooldown_status,
)


def test_classify_twelve_data_failure_detects_daily_quota() -> None:
    assert (
        classify_twelve_data_failure(
            code="429",
            message="You have run out of API credits for the day. Wait for the next day.",
        )
        == "daily_quota"
    )
    assert classify_twelve_data_failure(code="429", message="Too many requests") == "rate_limit"


def test_mark_twelve_data_cooldown_uses_next_reset(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("FX_TWELVE_DATA_DAILY_QUOTA_RESET_HOUR_UTC", "0")
    monkeypatch.setenv("FX_TWELVE_DATA_DAILY_QUOTA_RESET_MINUTE_UTC", "0")
    monkeypatch.setenv("FX_TWELVE_DATA_DAILY_QUOTA_RESET_GRACE_SECONDS", "300")
    now_dt = datetime(2026, 4, 20, 20, 15, tzinfo=timezone.utc)
    state = mark_twelve_data_cooldown(
        project_root=tmp_path,
        kind="daily_quota",
        code="429",
        message="run out of API credits for the day",
        symbol="EURUSD",
        source="test",
        now_ts=now_dt.timestamp(),
    )
    assert state["active"] is True
    assert state["kind"] == "daily_quota"
    assert datetime.fromisoformat(state["cooldown_until_utc"]).replace(tzinfo=timezone.utc).hour == 0
    assert state["remaining_seconds"] > 0
    persisted = twelve_data_cooldown_status(tmp_path, now_ts=now_dt.timestamp())
    assert persisted["active"] is True

