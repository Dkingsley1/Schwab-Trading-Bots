#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.live_order_ledger import LiveOrderLedger
    from core.live_transition_safety import evaluate_release_interlock
    from scripts.ops.long_runtime_common import iso_now, write_payload
else:
    from core.live_order_ledger import LiveOrderLedger
    from core.live_transition_safety import evaluate_release_interlock
    from .long_runtime_common import PROJECT_ROOT, iso_now, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "live_transition_chaos_harness_latest.json"


def _signals(**overrides: Any) -> dict[str, Any]:
    return {
        "restart_reconciled": True,
        "auth_ready": True,
        "auth_generation_stable": True,
        "quote_fresh": True,
        "sources_ready": True,
        "reconciliation_clean": True,
        "drawdown_within_limit": True,
        "storage_ready": True,
        "production_ready": True,
        "operator_release_present": True,
        "exit_route_ready": True,
        "broker_reachable": True,
        **overrides,
    }


def _scenario(name: str, fn: Callable[[], tuple[bool, dict[str, Any]]]) -> dict[str, Any]:
    try:
        passed, evidence = fn()
        return {"scenario": name, "passed": bool(passed), "evidence": evidence, "error": ""}
    except Exception as exc:
        return {
            "scenario": name,
            "passed": False,
            "evidence": {},
            "error": f"{type(exc).__name__}:{exc}",
        }


def _partial_fill(path: Path) -> tuple[bool, dict[str, Any]]:
    ledger = LiveOrderLedger(path)
    ledger.reserve(intent_id="partial-fill", payload={"symbol": "SPY"}, requested_quantity=2.0)
    ledger.mark_submitting("partial-fill")
    ledger.mark_submit_result(intent_id="partial-fill", acknowledged=True, broker_order_id="broker-partial")
    row = ledger.record_broker_update(
        broker_order_id="broker-partial",
        broker_status="PARTIALLY_FILLED",
        filled_quantity=1.0,
        average_fill_price=100.0,
    )
    return bool(row.get("state") == "partially_filled" and float(row.get("filled_quantity", 0.0)) == 1.0), row


def _delayed_response(path: Path) -> tuple[bool, dict[str, Any]]:
    ledger = LiveOrderLedger(path)
    ledger.reserve(intent_id="delayed", payload={"symbol": "QQQ"}, requested_quantity=1.0)
    ledger.mark_submitting("delayed")
    unknown = ledger.mark_submit_result(intent_id="delayed", acknowledged=False, error="timeout")
    duplicate = ledger.reserve(intent_id="delayed", payload={"symbol": "QQQ"}, requested_quantity=1.0)
    passed = bool(unknown.get("state") == "submit_unknown" and duplicate.get("duplicate") and not duplicate.get("reserved"))
    return passed, {"unknown": unknown, "duplicate": duplicate}


def _token_expiry() -> tuple[bool, dict[str, Any]]:
    result = evaluate_release_interlock(_signals(auth_ready=False))
    return bool(result.get("auto_relocked") and "auth_not_ready" in result.get("entry_lock_reasons", [])), result


def _network_loss() -> tuple[bool, dict[str, Any]]:
    entry = evaluate_release_interlock(_signals(auth_ready=False, broker_reachable=False))
    exit_result = evaluate_release_interlock(
        _signals(auth_ready=False, broker_reachable=False),
        risk_reducing_exit=True,
    )
    return bool(not entry.get("entry_allowed") and not exit_result.get("risk_reducing_exit_allowed")), {
        "entry": entry,
        "risk_reducing_exit": exit_result,
    }


def _cancel_replace_race(path: Path) -> tuple[bool, dict[str, Any]]:
    ledger = LiveOrderLedger(path)
    ledger.reserve(intent_id="cancel-race", payload={"symbol": "IWM"}, requested_quantity=1.0)
    ledger.mark_submitting("cancel-race")
    ledger.mark_submit_result(intent_id="cancel-race", acknowledged=True, broker_order_id="broker-cancel")
    ledger.record_broker_update(broker_order_id="broker-cancel", broker_status="WORKING")
    ledger.mark_cancel_pending("broker-cancel")
    unknown = ledger.mark_cancel_unknown("broker-cancel", error="replace_response_race")
    resolved = ledger.reconcile_ambiguous(
        intent_id="cancel-race",
        resolution="open",
        evidence="broker order query proves replacement remains open",
        broker_order_id="broker-cancel",
    )
    return bool(unknown.get("state") == "cancel_unknown" and resolved.get("state") == "open"), {
        "unknown": unknown,
        "resolved": resolved,
    }


def _restart_with_open_order(path: Path) -> tuple[bool, dict[str, Any]]:
    ledger = LiveOrderLedger(path)
    ledger.reserve(intent_id="restart-open", payload={"symbol": "DIA"}, requested_quantity=1.0)
    ledger.mark_submitting("restart-open")
    ledger.mark_submit_result(intent_id="restart-open", acknowledged=True, broker_order_id="broker-open")
    ledger.record_broker_update(broker_order_id="broker-open", broker_status="WORKING")
    reopened = LiveOrderLedger(path)
    rows = reopened.unresolved()
    passed = any(row.get("intent_id") == "restart-open" and row.get("state") == "open" for row in rows)
    return bool(passed and reopened.verify_event_chain().get("ok")), {
        "unresolved": rows,
        "event_chain": reopened.verify_event_chain(),
    }


def _storage_loss() -> tuple[bool, dict[str, Any]]:
    result = evaluate_release_interlock(_signals(storage_ready=False))
    return bool(result.get("auto_relocked") and "durable_storage_unavailable" in result.get("entry_lock_reasons", [])), result


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    del project_root
    with tempfile.TemporaryDirectory(prefix="live-transition-chaos-") as raw_dir:
        root = Path(raw_dir)
        rows = [
            _scenario("partial_fills", lambda: _partial_fill(root / "partial.sqlite3")),
            _scenario("delayed_broker_response", lambda: _delayed_response(root / "delayed.sqlite3")),
            _scenario("token_expiry", _token_expiry),
            _scenario("network_loss", _network_loss),
            _scenario("cancel_replace_race", lambda: _cancel_replace_race(root / "cancel.sqlite3")),
            _scenario("restart_with_open_orders", lambda: _restart_with_open_order(root / "restart.sqlite3")),
            _scenario("durable_storage_loss", _storage_loss),
        ]
    passed = sum(1 for row in rows if row.get("passed", False))
    all_passed = passed == len(rows)
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": all_passed,
        "overall_status": "ready" if all_passed else "blocked",
        "grade": "A+" if all_passed else "F",
        "passed_scenario_count": passed,
        "scenario_count": len(rows),
        "scenarios": rows,
        "simulation_only": True,
        "live_execution_authority": False,
        "operational_drill_substitute": False,
        "policy": "deterministic non-network simulations prove fail-closed code behavior; production operational drills remain separately required",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic fail-closed live-transition chaos simulations.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--out-file", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    out_path = args.out_file if args.out_file.is_absolute() else project_root / args.out_file
    payload = build_payload(project_root)
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "live_transition_chaos_harness "
            f"status={payload['overall_status']} passed={payload['passed_scenario_count']}/{payload['scenario_count']}"
        )
    return 0 if payload.get("ok", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
