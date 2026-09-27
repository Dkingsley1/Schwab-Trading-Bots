from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import provider_access_guard as guard
import scripts.run_shadow_training_loop as loop


class _Response:
    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict:
        return dict(self._payload)


class _DeniedClient:
    def __init__(self) -> None:
        self.quote_calls = 0

    def get_quote(self, symbol: str) -> _Response:
        self.quote_calls += 1
        return _Response(403)


def test_provider_http_status_code_parses_runtime_and_broker_errors() -> None:
    assert (
        guard.provider_http_status_code("schwab_quote_http_error symbol=SPY status=403")
        == 403
    )
    assert guard.provider_http_status_code("RuntimeError:http_status_429") == 429
    assert guard.provider_http_status_code({"status_code": 401}) == 401


def test_provider_cooldown_is_shared_and_persisted(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SCHWAB_HTTP_403_COOLDOWN_SECONDS", "60")

    activated = guard.activate_provider_cooldown(
        tmp_path,
        "schwab",
        status_code=403,
        reason="access_denied",
        symbol="SPY",
        profile="baseline",
        domain="equities",
    )
    loaded = guard.provider_access_status(tmp_path, "schwab")

    assert activated["active"] is True
    assert loaded["active"] is True
    assert loaded["status_code"] == 403
    assert loaded["remaining_seconds"] >= 58
    assert loaded["denial_count"] == 1
    assert Path(loaded["path"]).exists()


def test_verified_provider_request_can_force_recovery_from_active_cooldown(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("SCHWAB_HTTP_403_COOLDOWN_SECONDS", "60")
    guard.activate_provider_cooldown(
        tmp_path,
        "schwab",
        status_code=403,
        reason="historical_access_denied",
    )

    recovered = guard.mark_provider_recovered(
        tmp_path,
        "schwab",
        evidence="verified_post_auth_account_snapshot_and_broker_truth",
        force=True,
    )

    assert recovered["state"] == "ready"
    assert recovered["active"] is False
    assert recovered["forced_recovery_from_verified_request"] is True


def test_runtime_cooldown_deadline_clears_when_shared_provider_state_recovers() -> None:
    active = loop._reconcile_provider_cooldown_deadline(
        0.0,
        {"active": True, "state": "cooldown", "cooldown_until_epoch": 200.0},
        now_ts=100.0,
    )
    recovered = loop._reconcile_provider_cooldown_deadline(
        active,
        {"active": False, "state": "ready"},
        now_ts=110.0,
    )

    assert active == 200.0
    assert recovered == 0.0


def test_auth_success_does_not_clear_rate_limit_even_after_expiry(
    tmp_path, monkeypatch
):
    clock = [1000.0]
    monkeypatch.setattr(guard.time, "time", lambda: clock[0])
    denied = guard.activate_provider_cooldown(
        tmp_path, "schwab", status_code=429, reason="rate_limited", cooldown_seconds=300
    )
    path = Path(denied["path"])
    original = path.read_bytes()
    for now in (1001.0, 1301.0):
        clock[0] = now
        result = guard.mark_provider_recovered(
            tmp_path,
            "schwab",
            evidence="verified_post_auth_account_snapshot_and_broker_truth",
            force=True,
        )
        assert result["state"] == "cooldown"
        assert path.read_bytes() == original
    recovered = guard.mark_provider_recovered(
        tmp_path, "schwab", evidence="successful_market_data_request"
    )
    assert recovered["state"] == "ready"
    assert recovered["denial_count"] == denied["denial_count"]


def test_new_rate_limit_under_recovery_lock_is_preserved(tmp_path, monkeypatch):
    guard.activate_provider_cooldown(tmp_path, "schwab", status_code=403, reason="auth")
    original_status = guard.provider_access_status
    calls = [0]

    def racing_status(*args, **kwargs):
        calls[0] += 1
        value = original_status(*args, **kwargs)
        if calls[0] == 2:
            value["status_code"] = 429
        return value

    monkeypatch.setattr(guard, "provider_access_status", racing_status)
    result = guard.mark_provider_recovered(
        tmp_path, "schwab", evidence="auth_ok", force=True
    )
    assert result["state"] == "cooldown"
    assert result["status_code"] == 429
    assert calls[0] == 2


def test_free_request_slot_cannot_bypass_shared_cooldown(tmp_path):
    guard.activate_provider_cooldown(
        tmp_path, "schwab", status_code=429, reason="rate_limit"
    )
    with pytest.raises(RuntimeError, match="provider_cooldown_active"):
        with guard.provider_request_slot(tmp_path, "schwab", "SCHD"):
            pytest.fail("provider request admitted during cooldown")


def test_rejected_request_slot_releases_lock(tmp_path, monkeypatch):
    active = [True]
    monkeypatch.setattr(
        guard, "provider_access_status", lambda *a, **kw: {"active": active[0]}
    )
    with pytest.raises(RuntimeError, match="provider_cooldown_active"):
        with guard.provider_request_slot(tmp_path, "schwab", "SCHD"):
            pass
    active[0] = False
    with guard.provider_request_slot(tmp_path, "schwab", "SCHD", wait_seconds=0.1):
        pass


def test_shared_market_snapshot_cache_is_bounded_by_freshness(tmp_path: Path) -> None:
    assert guard.write_shared_market_snapshot(
        tmp_path, "schwab", "SPY", {"last_price": 601.25}
    )

    fresh = guard.load_shared_market_snapshot(
        tmp_path,
        "schwab",
        "SPY",
        max_age_seconds=5,
    )
    assert fresh is not None
    assert fresh["last_price"] == pytest.approx(601.25)
    assert fresh["shared_provider_cache_hit"] == 1.0

    cache_path = next(
        (
            tmp_path
            / "governance"
            / "health"
            / "provider_market_snapshot_cache"
            / "schwab"
        ).glob("*.json")
    )
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    payload["timestamp_epoch"] = 1.0
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    assert (
        guard.load_shared_market_snapshot(tmp_path, "schwab", "SPY", max_age_seconds=5)
        is None
    )


def test_first_schwab_403_stops_followup_client_calls(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(loop, "PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("SCHWAB_HTTP_403_COOLDOWN_SECONDS", "60")
    first = _DeniedClient()

    with pytest.raises(RuntimeError, match="status=403"):
        loop._market_snapshot_from_schwab_guarded(first, "SPY")
    assert first.quote_calls == 1
    assert guard.provider_access_status(tmp_path, "schwab")["active"] is True

    second = _DeniedClient()
    with pytest.raises(RuntimeError, match="provider_cooldown_active"):
        loop._market_snapshot_from_schwab_guarded(second, "QQQ")
    assert second.quote_calls == 0
