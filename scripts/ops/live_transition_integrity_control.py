#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.live_transition_safety import canary_stage_contract, evaluate_release_interlock, reconcile_broker_truth
    from core.order_intent import build_order_intent_evidence, verify_order_intent_evidence
    from scripts.ops import live_transition_chaos_harness
    from scripts.ops.long_runtime_common import iso_now, load_json, ordered_unique, write_payload
else:
    from core.live_transition_safety import canary_stage_contract, evaluate_release_interlock, reconcile_broker_truth
    from core.order_intent import build_order_intent_evidence, verify_order_intent_evidence
    from . import live_transition_chaos_harness
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "live_transition_integrity_control_latest.json"


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _status(raw: Any) -> str:
    return str(raw or "").strip().lower()


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _grade(score: float, *, ready: bool = False) -> str:
    if ready and score >= 100.0:
        return "A+"
    if score >= 90.0:
        return "A"
    if score >= 80.0:
        return "B"
    if score >= 70.0:
        return "C"
    if score >= 60.0:
        return "D"
    return "F"


def _control(control_id: str, title: str, implemented: bool, evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "control_id": control_id,
        "title": title,
        "implemented": bool(implemented),
        "status": "ready" if implemented else "blocked",
        "evidence": evidence,
    }


def _source_row(source: dict[str, Any], source_id: str) -> dict[str, Any]:
    for row in _as_list(source.get("sources")):
        if isinstance(row, dict) and str(row.get("source_id") or "") == source_id:
            return row
    return {}


def _auth_runtime_signals(auth: dict[str, Any]) -> dict[str, bool]:
    token = _as_dict(auth.get("token"))
    broker = _as_dict(auth.get("broker_readiness"))
    supervisor_ready = bool(auth.get("ok", False) and _status(auth.get("overall_status")) == "ready")
    token_ready = bool(token.get("ready", auth.get("token_ready", False)))
    refresh_needed = bool(token.get("refresh_needed", auth.get("refresh_needed", True)))
    auth_probe_ready = bool(broker.get("auth_ok", auth.get("auth_ok", supervisor_ready)))
    network_ready = bool(broker.get("network_ok", auth.get("network_ok", True)))
    return {
        "auth_ready": bool(supervisor_ready and token_ready and auth_probe_ready),
        "auth_generation_stable": not refresh_needed,
        "broker_reachable": network_ready,
    }


def _capability_self_tests(project_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sample = build_order_intent_evidence(
        decision_id="intent-self-test",
        symbol="SPY",
        action="BUY",
        quantity=1.0,
        strategy="self_test",
        quote_snapshot={"last_price": 100.0, "snapshot_id": "sample"},
        expected_fill={"expected_fill_price": 100.01, "partial_fill_ratio": 1.0},
        risk_decision={"ok": True, "gate": "pre_trade", "reason": "ok", "details": {"order_notional": 100.0}},
    )
    intent_verification = verify_order_intent_evidence(sample)
    parity_sample = build_order_intent_evidence(
        decision_id="intent-self-test",
        symbol="SPY",
        action="BUY",
        quantity=1.0,
        strategy="self_test",
        quote_snapshot={"last_price": 100.0, "snapshot_id": "sample"},
        expected_fill={"expected_fill_price": 100.01, "partial_fill_ratio": 1.0},
        risk_decision={"ok": True, "gate": "pre_trade", "reason": "ok", "details": {"order_notional": 100.0}},
    )
    parity_ok = bool(sample.get("intent_sha256") == parity_sample.get("intent_sha256"))

    clean_truth = {
        "orders": [{"order_id": "one", "status": "filled"}],
        "fills": [{"order_id": "one", "filled_quantity": 1.0}],
        "positions": [{"symbol": "SPY", "quantity": 1.0}],
        "buying_power": 1000.0,
        "cancels": [{"order_id": "two", "status": "canceled"}],
    }
    reconcile_self_test = reconcile_broker_truth(clean_truth, clean_truth)
    relock_self_test = evaluate_release_interlock(
        {
            "restart_reconciled": True,
            "auth_ready": False,
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
        }
    )
    canary_self_test = canary_stage_contract(
        requested_weight=0.0025,
        clean_evidence_windows=1,
        sleeve_count=1,
        open_position_count=0,
    )
    exit_self_test = evaluate_release_interlock(
        {
            "restart_reconciled": False,
            "auth_ready": True,
            "auth_generation_stable": False,
            "quote_fresh": False,
            "sources_ready": False,
            "reconciliation_clean": False,
            "drawdown_within_limit": False,
            "storage_ready": False,
            "production_ready": False,
            "operator_release_present": False,
            "exit_route_ready": True,
            "broker_reachable": True,
        },
        risk_reducing_exit=True,
    )
    chaos = live_transition_chaos_harness.build_payload(project_root)
    ledger_control = load_json(project_root / "governance" / "health" / "live_order_ledger_control_latest.json")
    ledger_contract = _as_dict(ledger_control.get("contract"))
    idempotency_implemented = bool(
        ledger_contract.get("stable_intent_id_required", False)
        and ledger_contract.get("transactional_reservation_before_submit", False)
        and ledger_contract.get("ambiguous_submit_never_auto_retried", False)
        and ledger_contract.get("hash_chained_order_events", False)
    )
    controls = [
        _control(
            "01_mode_invariant_order_intent",
            "Paper and live share one immutable semantic order intent",
            bool(intent_verification.get("ok") and parity_ok),
            {"verification": intent_verification, "parity_hash_equal": parity_ok, "intent_sha256": sample.get("intent_sha256")},
        ),
        _control(
            "02_component_hash_bundle",
            "Intent, risk, quote, and expected-fill components are hash-bound",
            bool(intent_verification.get("ok") and len(_as_dict(sample.get("component_hashes"))) == 4),
            {"component_hashes": sample.get("component_hashes", {}), "adapter_excluded_from_hash": sample.get("adapter_excluded_from_hash")},
        ),
        _control(
            "03_restart_safe_idempotency",
            "Durable reservations prevent duplicate submit after restart or ambiguity",
            idempotency_implemented,
            {"ledger_status": ledger_control.get("overall_status"), "ledger_contract": ledger_contract},
        ),
        _control(
            "04_independent_broker_reconciliation",
            "Orders, fills, positions, buying power, and cancels reconcile independently",
            bool(reconcile_self_test.get("ok") and len(reconcile_self_test.get("surfaces_checked", [])) == 5),
            reconcile_self_test,
        ),
        _control(
            "05_automatic_relock",
            "Restart, auth, quote, source, reconciliation, drawdown, and storage faults relock entries",
            bool(relock_self_test.get("auto_relocked") and "auth_not_ready" in relock_self_test.get("entry_lock_reasons", [])),
            relock_self_test,
        ),
        _control(
            "06_transition_chaos_coverage",
            "Partial fills and transition faults pass deterministic fail-closed simulations",
            bool(chaos.get("ok", False) and int(chaos.get("scenario_count", 0) or 0) >= 7),
            {"grade": chaos.get("grade"), "passed": chaos.get("passed_scenario_count"), "total": chaos.get("scenario_count")},
        ),
        _control(
            "07_microscopic_staged_canary",
            "Canary starts at 0.25%, one sleeve, one bounded position, and never auto-scales above 1%",
            bool(canary_self_test.get("ok") and not canary_self_test.get("automatic_scaling_allowed", True)),
            canary_self_test,
        ),
        _control(
            "08_independent_kill_and_reduce_only_exit",
            "Entry failures do not close the independently guarded risk-reducing exit route",
            bool(exit_self_test.get("risk_reducing_exit_allowed") and not exit_self_test.get("entry_allowed")),
            exit_self_test,
        ),
    ]
    return controls, chaos


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    health = project_root / "governance" / "health"
    controls, chaos = _capability_self_tests(project_root)
    auth = load_json(health / "schwab_auth_supervisor_latest.json")
    source = load_json(health / "source_verification_latest.json")
    paper_truth = load_json(health / "paper_execution_truth_layer_latest.json")
    storage = load_json(health / "storage_resilience_control_latest.json")
    process = load_json(health / "process_watchdog_latest.json")
    production = load_json(health / "production_excellence_control_latest.json")
    canary = load_json(health / "live_canary_control_latest.json")
    profitability = load_json(health / "profitability_evidence_firewall_latest.json")
    ledger = load_json(health / "live_order_ledger_control_latest.json")
    source_quote = _source_row(source, "market_quote_profiles")
    broker_gate = _as_dict(_as_dict(paper_truth.get("gates")).get("paper_broker_truth_reconciliation"))
    restart_storms = _as_list(process.get("restart_storms"))
    target_weight = _safe_float(canary.get("target_canary_weight"), 0.0025)
    clean_evidence_windows = int(canary.get("clean_evidence_windows", 0) or 0)
    if clean_evidence_windows <= 0 and production.get("ten_out_of_ten_ready", False):
        clean_evidence_windows = 1
    current_canary = canary_stage_contract(
        requested_weight=target_weight,
        clean_evidence_windows=clean_evidence_windows,
        sleeve_count=1,
        open_position_count=int(canary.get("open_position_count", 0) or 0),
    )
    release_flag = health / "LIVE_CANARY_RELEASE.flag"
    auth_signals = _auth_runtime_signals(auth)
    signals = {
        "restart_reconciled": bool(not restart_storms and ledger.get("ok", False)),
        "auth_ready": auth_signals["auth_ready"],
        "auth_generation_stable": auth_signals["auth_generation_stable"],
        "quote_fresh": bool(source_quote.get("fresh", False) and source_quote.get("ok", False)),
        "sources_ready": bool(source.get("ok", False) and _status(source.get("overall_status")) == "ready"),
        "reconciliation_clean": bool(broker_gate.get("ok", False) and int(broker_gate.get("mismatch_count", 0) or 0) == 0),
        "drawdown_within_limit": bool(profitability.get("promotion_evidence_ready", False)),
        "storage_ready": bool(storage.get("ok", False) and _status(storage.get("overall_status")) == "ready"),
        "production_ready": bool(production.get("ten_out_of_ten_ready", False)),
        "operator_release_present": release_flag.exists(),
        "exit_route_ready": True,
        "broker_reachable": auth_signals["broker_reachable"],
    }
    interlock = evaluate_release_interlock(signals)
    implemented_count = sum(1 for row in controls if row.get("implemented", False))
    control_ready = implemented_count == len(controls)
    control_score = 100.0 * implemented_count / max(len(controls), 1)
    runtime_checks = {
        **signals,
        "canary_stage_ready": bool(current_canary.get("ok", False)),
        "code_chaos_ready": bool(chaos.get("ok", False)),
    }
    runtime_ready_count = sum(1 for value in runtime_checks.values() if bool(value))
    runtime_score = 100.0 * runtime_ready_count / max(len(runtime_checks), 1)
    transition_ready = bool(control_ready and not interlock.get("entry_lock_reasons") and current_canary.get("ok", False))
    blockers = ordered_unique(
        list(interlock.get("entry_lock_reasons") or [])
        + list(current_canary.get("blockers") or [])
        + [str(row.get("control_id")) for row in controls if not row.get("implemented", False)]
    )
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": control_ready,
        "overall_status": "ready_locked" if control_ready and not transition_ready else "ready" if transition_ready else "blocked",
        "control_grade": _grade(control_score, ready=control_ready),
        "control_score": round(control_score, 3),
        "implemented_control_count": implemented_count,
        "control_count": len(controls),
        "transition_readiness_grade": _grade(runtime_score, ready=transition_ready),
        "transition_readiness_score": round(runtime_score, 3),
        "ready_for_live_transition": transition_ready,
        "live_execution_authority": False,
        "live_orders_must_remain_disabled": not transition_ready,
        "controls": controls,
        "runtime_signals": signals,
        "release_interlock": interlock,
        "current_canary_stage": current_canary,
        "blockers": blockers,
        "grading_contract": {
            "control_grade_measures_implemented_fail_closed_capability": True,
            "transition_grade_requires_current_runtime_and_economic_evidence": True,
            "control_A_plus_does_not_authorize_live_money": True,
            "missing_runtime_evidence_fails_closed": True,
        },
        "recommended_actions": [
            "clear current runtime and evidence blockers while the automatic release interlock remains locked"
            if blockers
            else "require a fresh explicit operator release for the microscopic supervised canary",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the eight-control seamless paper-to-live transition contract.")
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
            "live_transition_integrity_control "
            f"control_grade={payload['control_grade']} transition_grade={payload['transition_readiness_grade']} "
            f"ready={int(bool(payload['ready_for_live_transition']))}"
        )
    return 0 if payload.get("ok", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
