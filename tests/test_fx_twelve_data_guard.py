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
    assert classify_twelve_data_failure(code="401", message="HTTP Error 401: Unauthorized") == "auth"
    assert classify_twelve_data_failure(code="403", message="Forbidden") == "auth"


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


def test_mark_twelve_data_auth_failure_uses_long_bounded_cooldown(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("FX_TWELVE_DATA_AUTH_COOLDOWN_SECONDS", "7200")
    now_dt = datetime(2026, 8, 4, 23, 0, tzinfo=timezone.utc)

    state = mark_twelve_data_cooldown(
        project_root=tmp_path,
        kind="auth",
        code="401",
        message="HTTP Error 401: Unauthorized",
        symbol="EURUSD",
        source="test",
        now_ts=now_dt.timestamp(),
    )

    assert state["active"] is True
    assert state["kind"] == "auth"
    assert state["credential_action_required"] is True
    assert state["retry_policy"] == "credential_fix_or_bounded_cooldown"
    assert state["remaining_seconds"] == 7200.0
