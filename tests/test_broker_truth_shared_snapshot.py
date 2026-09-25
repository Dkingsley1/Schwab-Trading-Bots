import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
LIVE_PROJECT_ROOT = Path("/Users/dankingsley/PycharmProjects/schwab_trading_bot")
if not (PROJECT_ROOT / "core").exists() and str(LIVE_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(LIVE_PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import scripts.run_shadow_training_loop as loop
from core.base_trader import BaseTrader


class _FailingTrader:
    client = object()

    def _live_fetch_accounts_payload(self):
        raise AssertionError("shared broker truth snapshot should have been reused")


class _SnapshotTrader:
    client = object()

    def __init__(self, fetched: dict):
        self._fetched = dict(fetched)

    def _live_fetch_accounts_payload(self):
        return dict(self._fetched)

    def _extract_all_positions_from_payload(self, payload):
        rows = []
        if isinstance(payload, dict):
            sec = payload.get("securitiesAccount") if isinstance(payload.get("securitiesAccount"), dict) else payload
            for row in sec.get("positions", []) if isinstance(sec, dict) else []:
                inst = row.get("instrument") if isinstance(row, dict) and isinstance(row.get("instrument"), dict) else {}
                symbol = str(inst.get("symbol") or "").strip().upper()
                if symbol:
                    rows.append({"symbol": symbol, "quantity": float(row.get("longQuantity", row.get("quantity", 0.0)) or 0.0)})
        return rows

    def _extract_open_order_ids_from_payload(self, payload):
        return []


def test_shared_snapshot_preserves_last_good_across_failed_refresh(tmp_path) -> None:
    success = {
        "ok": True,
        "account_count": 1,
        "payload": {"accounts": [{"securitiesAccount": {"positions": []}}]},
    }
    failure = {
        "ok": False,
        "error": "account_discovery_provider_unavailable",
        "status_code": 500,
        "provider_failure": True,
        "provider_failure_class": "provider_unavailable",
    }

    assert loop._write_broker_truth_shared_snapshot(
        project_root=str(tmp_path), broker="schwab", fetched=success
    )
    assert loop._write_broker_truth_shared_snapshot(
        project_root=str(tmp_path), broker="schwab", fetched=failure
    )

    latest = json.loads(
        loop._broker_truth_shared_snapshot_cache_path(str(tmp_path), "schwab").read_text(
            encoding="utf-8"
        )
    )
    last_good = json.loads(
        loop._broker_truth_shared_snapshot_last_good_path(str(tmp_path), "schwab").read_text(
            encoding="utf-8"
        )
    )
    assert latest["fetched"]["ok"] is False
    assert last_good["fetched"]["ok"] is True
    assert last_good["fetched"]["account_count"] == 1


def test_shared_snapshot_redacts_account_credentials_and_is_owner_only(
    tmp_path,
) -> None:
    fetched = {
        "ok": True,
        "accessToken": "secret-access-token",
        "payload": {
            "accounts": [
                {
                    "_broker_account": {
                        "account_number_tail": "6789",
                        "account_reference_present": True,
                        "account_reference": "secret-account-hash",
                    },
                    "securitiesAccount": {
                        "accountNumber": "123456789",
                        "hashValue": "secret-account-hash",
                        "positions": [],
                    },
                }
            ]
        },
    }

    assert loop._write_broker_truth_shared_snapshot(
        project_root=str(tmp_path), broker="schwab", fetched=fetched
    )
    cache_path = loop._broker_truth_shared_snapshot_cache_path(
        str(tmp_path), "schwab"
    )
    raw = cache_path.read_text(encoding="utf-8")
    payload = json.loads(raw)

    assert "123456789" not in raw
    assert "secret-account-hash" not in raw
    assert "secret-access-token" not in raw
    assert payload["redaction"]["raw_account_number_emitted"] is False
    assert (
        payload["fetched"]["payload"]["accounts"][0]["_broker_account"]
        ["account_number_tail"]
        == "6789"
    )
    assert cache_path.stat().st_mode & 0o077 == 0


def test_shared_snapshot_serves_bounded_last_good_only_in_collection_mode(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(loop, "PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("MARKET_DATA_ONLY", "1")
    monkeypatch.setenv("ALLOW_ORDER_EXECUTION", "0")
    success = {
        "ok": True,
        "account_count": 1,
        "payload": {"accounts": [{"securitiesAccount": {"positions": []}}]},
    }
    failure = {
        "ok": False,
        "error": "account_discovery_provider_unavailable",
        "status_code": 500,
        "provider_failure": True,
        "provider_failure_class": "provider_unavailable",
    }
    loop._write_broker_truth_shared_snapshot(
        project_root=str(tmp_path), broker="schwab", fetched=success
    )
    loop._write_broker_truth_shared_snapshot(
        project_root=str(tmp_path), broker="schwab", fetched=failure
    )

    fetched = loop._shared_broker_truth_accounts_payload(
        trader=_FailingTrader(), broker="schwab"
    )

    assert fetched["ok"] is True
    assert fetched["_shared_snapshot_stale_fallback"] is True
    assert fetched["_shared_snapshot_current_failure"] == "account_discovery_provider_unavailable"
    assert fetched["provider_failure"] is True
    assert fetched["soft_failure"] is True


def test_shared_snapshot_never_serves_last_good_to_live_execution(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(loop, "PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("MARKET_DATA_ONLY", "0")
    monkeypatch.setenv("ALLOW_ORDER_EXECUTION", "1")
    loop._write_broker_truth_shared_snapshot(
        project_root=str(tmp_path),
        broker="schwab",
        fetched={"ok": True, "account_count": 1, "payload": {"accounts": []}},
    )
    loop._write_broker_truth_shared_snapshot(
        project_root=str(tmp_path),
        broker="schwab",
        fetched={
            "ok": False,
            "error": "account_discovery_provider_unavailable",
            "status_code": 500,
            "provider_failure": True,
        },
    )

    fetched = loop._shared_broker_truth_accounts_payload(
        trader=_FailingTrader(), broker="schwab"
    )

    assert fetched["ok"] is False
    assert fetched.get("_shared_snapshot_stale_fallback") is not True


def test_shared_broker_truth_snapshot_reuses_recent_cached_payload(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(loop, "PROJECT_ROOT", str(tmp_path))
    cache_path = loop._broker_truth_shared_snapshot_cache_path(str(tmp_path), "schwab")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "broker": "schwab",
                "owner_pid": os.getpid() + 1000,
                "fetched": {
                    "ok": False,
                    "error": "RuntimeError:http_status_403",
                    "soft_failure": True,
                    "soft_fail_streak": 1,
                    "soft_fail_grace": 3,
                },
            }
        ),
        encoding="utf-8",
    )

    fetched = loop._shared_broker_truth_accounts_payload(trader=_FailingTrader(), broker="schwab")

    assert fetched["error"] == "RuntimeError:http_status_403"
    assert fetched["_shared_snapshot_cache_hit"] is True
    assert fetched["_shared_snapshot_cache_owner_pid"] != os.getpid()


def test_fetch_broker_truth_snapshot_suppresses_duplicate_soft_fail_alert_from_shared_cache(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(loop, "PROJECT_ROOT", str(tmp_path))
    cache_path = loop._broker_truth_shared_snapshot_cache_path(str(tmp_path), "schwab")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "broker": "schwab",
                "owner_pid": os.getpid() + 1000,
                "fetched": {
                    "ok": False,
                    "error": "RuntimeError:http_status_403",
                    "soft_failure": True,
                    "soft_fail_streak": 1,
                    "soft_fail_grace": 3,
                },
            }
        ),
        encoding="utf-8",
    )

    snapshot = loop._fetch_broker_truth_snapshot(
        trader=_FailingTrader(),
        broker="schwab",
        simulate=False,
        iter_count=7,
        manual_payload={},
        manual_tolerance=1.0,
        previous_state={},
    )

    assert snapshot["ok"] is False
    assert snapshot["status"] == "transient_error"
    assert snapshot["soft_failure"] is True
    assert snapshot["shared_snapshot_cache_hit"] is True
    assert snapshot["alert_suppressed"] is True


def test_fetch_broker_truth_snapshot_rejects_empty_ok_snapshot(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(loop, "PROJECT_ROOT", str(tmp_path))

    snapshot = loop._fetch_broker_truth_snapshot(
        trader=_SnapshotTrader({"ok": True, "payload": {}}),
        broker="schwab",
        simulate=False,
        iter_count=3,
        manual_payload={},
        manual_tolerance=1.0,
        previous_state={},
    )

    assert snapshot["ok"] is False
    assert snapshot["status"] == "empty_snapshot"
    assert snapshot["error"] == "empty_or_unrecognized_accounts_snapshot"
    assert snapshot["account_snapshot_proof"]["account_snapshot_proof_ok"] is False


def test_fetch_broker_truth_snapshot_accepts_verified_empty_account(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(loop, "PROJECT_ROOT", str(tmp_path))

    snapshot = loop._fetch_broker_truth_snapshot(
        trader=_SnapshotTrader({"ok": True, "payload": {"securitiesAccount": {"positions": []}}}),
        broker="schwab",
        simulate=False,
        iter_count=3,
        manual_payload={},
        manual_tolerance=1.0,
        previous_state={},
    )

    assert snapshot["ok"] is True
    assert snapshot["status"] == "ok"
    assert snapshot["account_count"] == 1
    assert snapshot["position_count"] == 0
    assert snapshot["account_snapshot_proof"]["account_snapshot_proof_ok"] is True
    assert snapshot["broker_truth_reconcile_v2"]["truth_grade"] in {"A", "B"}
    assert snapshot["broker_truth_reconcile_v2"]["account_identity"]["account_count"] == 1


def test_fetch_broker_truth_snapshot_v2_tracks_balance_orders_and_deltas(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(loop, "PROJECT_ROOT", str(tmp_path))
    payload = {
        "securitiesAccount": {
            "accountNumber": "123456789",
            "currentBalances": {
                "cashBalance": 1000.0,
                "buyingPower": 2500.0,
                "liquidationValue": 5000.0,
            },
            "positions": [{"instrument": {"symbol": "AAPL"}, "longQuantity": 3}],
            "orderStrategies": [
                {"orderId": 1, "status": "FILLED"},
                {"orderId": 2, "status": "WORKING"},
            ],
        }
    }

    snapshot = loop._fetch_broker_truth_snapshot(
        trader=_SnapshotTrader({"ok": True, "payload": payload}),
        broker="schwab",
        simulate=False,
        iter_count=4,
        manual_payload={"symbols": {"AAPL": {"position_qty": 1}}},
        manual_tolerance=0.1,
        previous_state={},
    )

    v2 = snapshot["broker_truth_reconcile_v2"]
    assert snapshot["status"] == "mismatch"
    assert v2["truth_grade"] in {"A", "C"}
    assert v2["balance_truth"]["has_balance_truth"] is True
    assert v2["order_truth"]["filled_order_count"] == 1
    assert v2["order_truth"]["pending_order_count"] == 1
    assert v2["paper_ledger_delta"]["delta_symbol_count"] == 1
    assert v2["account_identity"]["redacted_account_markers"] == ["****6789"]
    assert "123456789" not in json.dumps(v2)


def test_clear_critical_alert_latest_removes_matching_broker_truth_alert(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(loop, "PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("SHADOW_PROFILE", "aggressive")
    monkeypatch.setenv("SHADOW_DOMAIN", "equities")
    alert_path = Path(loop._critical_alert_latest_path(str(tmp_path), "schwab"))
    alert_path.parent.mkdir(parents=True, exist_ok=True)
    alert_path.write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "event": "broker_truth_reconcile",
                "severity": "critical",
                "message": "error",
                "broker": "schwab",
                "profile": "aggressive",
                "domain": "equities",
            }
        ),
        encoding="utf-8",
    )

    result = loop._clear_critical_alert_latest(
        project_root=str(tmp_path),
        broker="schwab",
        event="broker_truth_reconcile",
    )

    assert result["cleared"] is True
    assert not alert_path.exists()


def test_clear_critical_alert_latest_keeps_unrelated_alert(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(loop, "PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("SHADOW_PROFILE", "aggressive")
    monkeypatch.setenv("SHADOW_DOMAIN", "equities")
    alert_path = Path(loop._critical_alert_latest_path(str(tmp_path), "schwab"))
    alert_path.parent.mkdir(parents=True, exist_ok=True)
    alert_path.write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "event": "options_margin_guard",
                "severity": "critical",
                "message": "blocked",
                "broker": "schwab",
                "profile": "aggressive",
                "domain": "equities",
            }
        ),
        encoding="utf-8",
    )

    result = loop._clear_critical_alert_latest(
        project_root=str(tmp_path),
        broker="schwab",
        event="broker_truth_reconcile",
    )

    assert result["cleared"] is False
    assert result["reason"] == "event_mismatch"
    assert alert_path.exists()


def test_broker_truth_403_without_mismatch_routes_to_access_degradation() -> None:
    route = loop._broker_truth_alert_route(
        {
            "ok": False,
            "status": "transient_error",
            "error": "RuntimeError:http_status_403",
            "status_code": 403,
            "mismatch_count": 0,
            "soft_failure": True,
            "soft_fail_streak": 1,
            "soft_fail_grace": 3,
        }
    )

    assert route["event"] == "broker_access_degraded"
    assert route["severity"] == "warn"
    assert route["failure_class"] == "broker_access_denied"


def test_broker_truth_real_mismatch_remains_critical_reconcile() -> None:
    route = loop._broker_truth_alert_route(
        {
            "ok": False,
            "status": "mismatch",
            "mismatch_count": 1,
        }
    )

    assert route["event"] == "broker_truth_reconcile"
    assert route["severity"] == "critical"
    assert route["failure_class"] == "broker_truth_mismatch"


def test_persistent_broker_access_denial_escalates_without_claiming_mismatch() -> None:
    route = loop._broker_truth_alert_route(
        {
            "ok": False,
            "status": "error",
            "error": "RuntimeError:http_status_403",
            "status_code": 403,
            "mismatch_count": 0,
            "soft_failure": False,
            "soft_fail_streak": 4,
            "soft_fail_grace": 3,
        }
    )

    assert route["event"] == "broker_access_degraded"
    assert route["severity"] == "critical"
    assert route["event"] != "broker_truth_reconcile"


def test_connected_account_aggregate_preserves_soft_failure_metadata() -> None:
    trader = BaseTrader.__new__(BaseTrader)
    trader.fetch_connected_accounts = lambda: [
        SimpleNamespace(account_reference="account-hash", account_number="123456789")
    ]
    trader.broker_adapter = SimpleNamespace(accounts_snapshot_candidates=lambda **kwargs: [])
    trader._invoke_client_candidates = lambda **kwargs: {
        "ok": False,
        "error": "RuntimeError:http_status_403",
        "status_code": 403,
        "soft_failure": True,
        "soft_fail_streak": 1,
        "soft_fail_grace": 3,
    }

    result = trader._live_fetch_connected_accounts_payload()

    assert result["ok"] is False
    assert result["status_code"] == 403
    assert result["soft_failure"] is True
    assert result["soft_fail_streak"] == 1
    assert result["soft_fail_grace"] == 3
