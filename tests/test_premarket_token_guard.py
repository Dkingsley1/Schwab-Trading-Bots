import json
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

from scripts.ops import premarket_token_guard as ptg


def test_token_status_reads_nested_epoch_expiry() -> None:
    with tempfile.TemporaryDirectory() as td:
        token_path = Path(td) / "token.json"
        token_path.write_text(
            json.dumps(
                {
                    "creation_timestamp": 1773069288,
                    "token": {
                        "refresh_token": "refresh-token",
                        "access_token": "access-token",
                        "expires_at": 4102444800,
                    },
                }
            ),
            encoding="utf-8",
        )

        status = ptg._token_status(token_path)

    assert status["exists"] is True
    assert status["expires_at"] == "4102444800"
    assert float(status["expires_in_seconds"]) > 0.0


def test_guard_fails_when_auth_reports_success_but_token_stays_stale() -> None:
    captured: list[dict] = []

    def _capture_payload(_path: Path, _fallback: Path, payload: dict) -> str:
        captured.append(payload)
        return "/tmp/premarket_token_guard_latest.json"

    with mock.patch.object(ptg, "_token_status", side_effect=[{"exists": True, "size_bytes": 808}, {"exists": True, "size_bytes": 808}]):
        with mock.patch.object(
            ptg,
            "_token_needs_refresh",
            side_effect=[(True, "token_age_high:1.0"), (True, "token_age_high:1.1"), (True, "token_age_high:1.1")],
        ):
            with mock.patch.object(ptg, "_auth_attempt", return_value={"attempted": True, "ok": True, "reason": "auth_success"}):
                with mock.patch.object(ptg, "_write_json", side_effect=_capture_payload):
                    with mock.patch.object(ptg, "_append_jsonl", return_value="/tmp/premarket_token_guard_events.jsonl"):
                        with mock.patch.object(ptg, "_alert", return_value={"attempted": False}):
                            with mock.patch.object(sys, "argv", ["premarket_token_guard.py"]):
                                rc = ptg.main()

    primary_payload = next(row for row in captured if "ok" in row)
    assert rc == 2
    assert primary_payload["ok"] is False
    assert primary_payload["refresh_needed_after"] is True


def test_guard_warns_but_stays_ready_when_early_refresh_fails_above_ready_floor() -> None:
    captured: list[dict] = []
    status = {
        "exists": True,
        "size_bytes": 808,
        "age_seconds": 100.0,
        "expires_in_seconds": 1200.0,
    }

    def _capture_payload(_path: Path, _fallback: Path, payload: dict) -> str:
        captured.append(payload)
        return "/tmp/premarket_token_guard_latest.json"

    with mock.patch.object(ptg, "_token_status", side_effect=[status, status]):
        with mock.patch.object(ptg, "_probe_network", return_value={"hostport": "api.schwabapi.com:443", "ok": True}):
            with mock.patch.object(
                ptg,
                "_direct_refresh_token_grant",
                return_value={"attempted": True, "ok": False, "reason": "refresh_token_grant_unavailable"},
            ):
                with mock.patch.object(
                    ptg,
                    "_auth_attempt",
                    return_value={"attempted": True, "ok": False, "reason": "account_probe_failed:403"},
                ):
                    with mock.patch.object(ptg, "_write_json", side_effect=_capture_payload):
                        with mock.patch.object(ptg, "_append_jsonl", return_value="/tmp/premarket_token_guard_events.jsonl"):
                            with mock.patch.object(ptg, "_alert", return_value={"attempted": False}):
                                with mock.patch.object(
                                    sys,
                                    "argv",
                                    [
                                        "premarket_token_guard.py",
                                        "--min-expires-seconds",
                                        "1500",
                                        "--ready-min-expires-seconds",
                                        "900",
                                    ],
                                ):
                                    rc = ptg.main()

    primary_payload = next(row for row in captured if "ok" in row)
    broker_readiness = next(row for row in captured if "ready_for_open" in row)
    assert rc == 0
    assert primary_payload["ok"] is True
    assert primary_payload["refresh_needed_after"] is True
    assert primary_payload["token_ready_after"] is True
    assert broker_readiness["ready_for_open"] is True
    assert broker_readiness["preflight_checks"]["token_ready_for_open"] is True


def test_guard_clears_pre_refresh_warning_after_successful_token_renewal() -> None:
    captured: list[dict] = []
    before = {
        "exists": True,
        "size_bytes": 808,
        "age_seconds": 300.0,
        "expires_in_seconds": 1400.0,
    }
    after = {
        "exists": True,
        "size_bytes": 808,
        "age_seconds": 0.0,
        "expires_in_seconds": 1800.0,
    }

    def _capture_payload(_path: Path, _fallback: Path, payload: dict) -> str:
        captured.append(payload)
        return "/tmp/premarket_token_guard_latest.json"

    with mock.patch.object(ptg, "_token_status", side_effect=[before, after]):
        with mock.patch.object(ptg, "_probe_network", return_value={"hostport": "api.schwabapi.com:443", "ok": True}):
            with mock.patch.object(
                ptg,
                "_direct_refresh_token_grant",
                return_value={"attempted": True, "ok": True, "reason": "refresh_token_grant_success"},
            ):
                with mock.patch.object(ptg, "_write_json", side_effect=_capture_payload):
                    with mock.patch.object(ptg, "_append_jsonl", return_value="/tmp/premarket_token_guard_events.jsonl"):
                        with mock.patch.object(ptg, "_alert", return_value={"attempted": False}):
                            with mock.patch.object(
                                sys,
                                "argv",
                                [
                                    "premarket_token_guard.py",
                                    "--min-expires-seconds",
                                    "1500",
                                    "--ready-min-expires-seconds",
                                    "900",
                                ],
                            ):
                                rc = ptg.main()

    primary_payload = next(row for row in captured if "ok" in row)
    broker_readiness = next(row for row in captured if "ready_for_open" in row)
    assert rc == 0
    assert primary_payload["refresh_needed_before"] is True
    assert primary_payload["refresh_needed_after"] is False
    assert broker_readiness["ready_for_open"] is True
    assert broker_readiness["token_expires_in_seconds"] == 1800.0
    assert broker_readiness["warnings"] == []


def test_browser_disabled_skips_premarket_client_auth_fallback(monkeypatch) -> None:
    captured: list[dict] = []
    status = {
        "exists": True,
        "size_bytes": 808,
        "age_seconds": 50000.0,
        "expires_in_seconds": 100.0,
    }

    def _capture_payload(_path: Path, _fallback: Path, payload: dict) -> str:
        captured.append(payload)
        return "/tmp/premarket_token_guard_latest.json"

    monkeypatch.setenv("PREMARKET_TOKEN_BROWSER_AUTH_DISABLED", "1")
    monkeypatch.setenv("SCHWAB_AUTH_BROWSER_DISABLED", "1")
    monkeypatch.setenv("SCHWAB_AUTH_ALLOW_BROWSER_OPEN", "0")

    with mock.patch.object(ptg, "_token_status", side_effect=[status, status]):
        with mock.patch.object(ptg, "_probe_network", return_value={"hostport": "api.schwabapi.com:443", "ok": True}):
            with mock.patch.object(
                ptg,
                "_token_needs_refresh",
                side_effect=[(True, "token_expiring_soon:100.0"), (True, "token_expiring_soon:100.0"), (True, "token_expiring_soon:100.0")],
            ):
                with mock.patch.object(
                    ptg,
                    "_direct_refresh_token_grant",
                    return_value={"attempted": True, "ok": False, "reason": "refresh_grant_failed"},
                ):
                    with mock.patch.object(ptg, "_auth_attempt", side_effect=AssertionError("browser fallback should not run")):
                        with mock.patch.object(ptg, "_write_json", side_effect=_capture_payload):
                            with mock.patch.object(ptg, "_append_jsonl", return_value="/tmp/premarket_token_guard_events.jsonl"):
                                with mock.patch.object(ptg, "_alert", return_value={"attempted": False}):
                                    with mock.patch.object(sys, "argv", ["premarket_token_guard.py"]):
                                        rc = ptg.main()

    primary_payload = next(row for row in captured if "ok" in row)
    assert rc == 2
    assert primary_payload["auth"]["reason"] == "browser_auth_disabled"
    assert primary_payload["auth"]["details"]["browser_disabled"] is True
    assert primary_payload["auth"]["refresh_grant"]["reason"] == "refresh_grant_failed"


def test_token_needs_refresh_uses_configurable_expiry_floor() -> None:
    status = {
        "exists": True,
        "size_bytes": 808,
        "age_seconds": 30.0,
        "expires_in_seconds": 500.0,
    }

    needs_refresh, reason = ptg._token_needs_refresh(status, max_age_seconds=3600.0, min_expires_seconds=300.0)
    assert needs_refresh is False
    assert reason == "token_fresh"

    needs_refresh, reason = ptg._token_needs_refresh(status, max_age_seconds=3600.0, min_expires_seconds=600.0)
    assert needs_refresh is True
    assert reason.startswith("token_expiring_soon:")


def test_direct_refresh_token_grant_extends_and_writes_atomically(monkeypatch, tmp_path: Path) -> None:
    token_path = tmp_path / "token.json"
    token_path.write_text(
        json.dumps(
            {
                "creation_timestamp": int(time.time()) - 100,
                "token": {
                    "access_token": "old-access",
                    "refresh_token": "refresh-token",
                    "expires_at": time.time() + 120,
                    "expires_in": 1800,
                    "scope": "api",
                    "token_type": "Bearer",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SCHWAB_API_KEY", "real_key")
    monkeypatch.setenv("SCHWAB_SECRET", "real_secret")

    class FakeOAuth2Client:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

        def refresh_token(self, *_args, **_kwargs):
            return {
                "access_token": "new-access",
                "refresh_token": "refresh-token",
                "expires_at": time.time() + 1800,
                "expires_in": 1800,
                "scope": "api",
                "token_type": "Bearer",
            }

    import authlib.integrations.httpx_client as httpx_client

    monkeypatch.setattr(httpx_client, "OAuth2Client", FakeOAuth2Client)

    result = ptg._direct_refresh_token_grant(token_path, min_extension_seconds=300.0)

    assert result["ok"] is True
    assert result["reason"] == "refresh_token_grant_success"
    refreshed = json.loads(token_path.read_text(encoding="utf-8"))
    assert refreshed["token"]["access_token"] == "new-access"
    assert refreshed["token"]["refresh_token"] == "refresh-token"


def test_token_warning_level_scales_with_age() -> None:
    assert ptg._token_warning_level(100.0, max_age_seconds=1000.0) == "fresh"
    assert ptg._token_warning_level(600.0, max_age_seconds=1000.0) == "watch"
    assert ptg._token_warning_level(900.0, max_age_seconds=1000.0) == "warn"
    assert ptg._token_warning_level(1200.0, max_age_seconds=1000.0) == "critical"
