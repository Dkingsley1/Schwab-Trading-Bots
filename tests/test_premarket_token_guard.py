import json
import sys
import tempfile
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
    captured: dict = {}

    def _capture_payload(_path: Path, _fallback: Path, payload: dict) -> str:
        captured["payload"] = payload
        return "/tmp/premarket_token_guard_latest.json"

    with mock.patch.object(ptg, "_token_status", side_effect=[{"exists": True, "size_bytes": 808}, {"exists": True, "size_bytes": 808}]):
        with mock.patch.object(
            ptg,
            "_token_needs_refresh",
            side_effect=[(True, "token_age_high:1.0"), (True, "token_age_high:1.1")],
        ):
            with mock.patch.object(ptg, "_auth_attempt", return_value={"attempted": True, "ok": True, "reason": "auth_success"}):
                with mock.patch.object(ptg, "_write_json", side_effect=_capture_payload):
                    with mock.patch.object(ptg, "_append_jsonl", return_value="/tmp/premarket_token_guard_events.jsonl"):
                        with mock.patch.object(ptg, "_alert", return_value={"attempted": False}):
                            with mock.patch.object(sys, "argv", ["premarket_token_guard.py"]):
                                rc = ptg.main()

    assert rc == 2
    assert captured["payload"]["ok"] is False
    assert captured["payload"]["refresh_needed_after"] is True

