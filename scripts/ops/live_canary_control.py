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
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "live_canary_control_latest.json"
DEFAULT_MAX_CANARY_WEIGHT = 0.12
RECOVERABLE_RUNTIME_CLEARANCE_STATES = {
    "awaiting_cold_lane",
    "awaiting_coverage_cycles",
    "managed_cold_lane_deferred",
    "managed_coverage_stage_deferred",
    "staged_preclearance",
    "coverage_cycles_ready",
    "off_hours_cold_lane_launch_ready",
    "scheduled_off_hours_launch",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def build_payload(project_root: Path = PROJECT_ROOT, *, max_canary_weight: float = DEFAULT_MAX_CANARY_WEIGHT) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    broker = load_json(health_root / "broker_readiness_latest.json")
    session = load_json(health_root / "session_ready_latest.json")
    storage = load_json(health_root / "storage_route_status_latest.json")
    live_lane = load_json(health_root / "execution_lane_live_latest.json")
    live_readiness = load_json(health_root / "live_readiness_smoke_latest.json")
    runtime = load_json(health_root / "live_runtime_separation_control_latest.json")
    promotion_autopilot = load_json(project_root / "governance" / "champion_challenger" / "promotion_autopilot_packet_latest.json")
    canary_auto_tuner = load_json(health_root / "canary_auto_tuner_latest.json")
    canary_rollout = load_json(health_root / "canary_rollout_latest.json")
    live_money_contract = load_json(health_root / "live_money_readiness_contract_latest.json")

    broker_ready = bool(broker.get("ready_for_open", False))
    session_ready = bool(session.get("ready", session.get("ok", False)))
    storage_ok = bool(storage.get("ok", True))
    storage_mode = str(storage.get("mode") or "").strip()
    live_lane_running = bool(
        (bool(live_lane) and not bool(live_lane.get("stale", False)))
        or live_readiness.get("live_lane_running", False)
    )
    clearance_state = str(((runtime.get("clearance_plan") or {}).get("clearance_state") or "")).strip().lower()
    clearance_ready = clearance_state in {"", "ready", "cleared", "released"}
    runtime_status = str(runtime.get("overall_status") or "").strip().lower()
    runtime_clearance_recoverable = bool(
        clearance_state in RECOVERABLE_RUNTIME_CLEARANCE_STATES
        and runtime_status in {"ready", "degraded", "needs_attention"}
    )
    release_contract = runtime.get("release_contract") if isinstance(runtime.get("release_contract"), dict) else {}
    live_lane_should_be_read_only = bool(release_contract.get("live_lane_should_be_read_only", False))
    coverage_seed_contract = (
        promotion_autopilot.get("coverage_seed_contract")
        if isinstance(promotion_autopilot.get("coverage_seed_contract"), dict)
        else {}
    )
    canary_seed_ready = bool(
        promotion_autopilot.get("canary_packet_ready", False)
        or coverage_seed_contract.get("canary_seed_ready", False)
    )

    packet_ready = bool(
        promotion_autopilot
        and str(promotion_autopilot.get("autopilot_state") or "").strip()
        in {"awaiting_approval", "ready_for_canary", "ready_for_supervised_canary"}
        and bool(
            promotion_autopilot.get("promotion_ready", False)
            or promotion_autopilot.get("canary_packet_ready", False)
        )
    )
    packet_preclearance_ready = bool(
        packet_ready
        or (
            promotion_autopilot
            and str(promotion_autopilot.get("overall_status") or "").strip().lower() in {"degraded", "needs_attention", "ready"}
            and bool(
                promotion_autopilot.get("committee_packet_seed_ready", False)
                or ((promotion_autopilot.get("signability_contract") or {}).get("committee_packet_seed_ready", False))
                or ((promotion_autopilot.get("approval_record") or {}).get("committee_packet_seed_ready", False))
                or canary_seed_ready
            )
            and int(((promotion_autopilot.get("readiness_repair_contract") or {}).get("critical_repair_gate_count", 0) or 0)) <= 4
        )
    )
    target_canary_weight = _safe_float(canary_auto_tuner.get("target_canary_max_weight"), 0.0)
    applied_weight = _safe_float(canary_rollout.get("applied_weight"), target_canary_weight)
    canary_weight = applied_weight if applied_weight > 0.0 else target_canary_weight
    weight_ok = canary_weight > 0.0 and canary_weight <= float(max_canary_weight)
    canary_signal_seed_ready = bool(target_canary_weight > 0.0 and weight_ok and canary_auto_tuner)
    canary_signal_ready = bool(
        (canary_rollout.get("eligible", False) and canary_rollout.get("promote_canary", False))
        or canary_signal_seed_ready
    )
    live_money_contract_enforced = bool(live_money_contract.get("policy_id"))
    live_money_contract_ready = bool(
        not live_money_contract_enforced
        or live_money_contract.get("faithful_live_money_ready", live_money_contract.get("ok", False))
    )
    prereq_ready = bool(broker_ready and session_ready and storage_ok and storage_mode == "external")

    blocking_reasons = ordered_unique(
        [
            "faithful_live_money_contract_not_ready" if live_money_contract_enforced and not live_money_contract_ready else "",
            "broker_not_ready" if not broker_ready else "",
            "session_not_ready" if not session_ready else "",
            "storage_not_ready" if not storage_ok else "",
            "storage_not_external" if storage_ok and storage_mode and storage_mode != "external" else "",
            "live_lane_not_running" if not live_lane_running else "",
            "runtime_clearance_not_ready" if not clearance_ready else "",
            "live_lane_read_only" if live_lane_should_be_read_only else "",
            ("promotion_packet_not_ready" if not packet_preclearance_ready else ("promotion_packet_preclearance_only" if not packet_ready else "")),
            "canary_rollout_not_ready" if not canary_signal_ready else "",
            "canary_weight_not_ready" if not weight_ok else "",
        ]
    )
    supervised_canary_ready = not blocking_reasons
    staged_preclearance_ready = bool(
        prereq_ready
        and not supervised_canary_ready
        and bool(blocking_reasons)
        and all(
            reason
            in {
                "live_lane_not_running",
                "runtime_clearance_not_ready",
                "live_lane_read_only",
                "promotion_packet_not_ready",
                "promotion_packet_preclearance_only",
                "canary_rollout_not_ready",
                "canary_weight_not_ready",
            }
            for reason in blocking_reasons
        )
    )
    preapproved_supervised_ready = bool(
        staged_preclearance_ready
        and packet_preclearance_ready
        and (runtime_clearance_recoverable or clearance_ready)
        and all(
            reason
            in {
                "live_lane_not_running",
                "runtime_clearance_not_ready",
                "live_lane_read_only",
                "promotion_packet_preclearance_only",
                "canary_rollout_not_ready",
                "canary_weight_not_ready",
            }
            for reason in blocking_reasons
        )
    )
    runnable_after_release_window = bool(
        preapproved_supervised_ready
        and canary_signal_ready
        and all(
            reason
            in {
                "live_lane_not_running",
                "runtime_clearance_not_ready",
                "live_lane_read_only",
            }
            for reason in blocking_reasons
        )
    )
    bounded_blocker_count = sum(
        1
        for reason in blocking_reasons
        if reason
        in {
            "live_lane_not_running",
            "runtime_clearance_not_ready",
            "live_lane_read_only",
            "promotion_packet_preclearance_only",
            "canary_rollout_not_ready",
            "canary_weight_not_ready",
        }
    )
    preclearance_score = 0.0
    if prereq_ready:
        preclearance_score += 35.0
    if packet_preclearance_ready:
        preclearance_score += 20.0
    if runtime_clearance_recoverable or clearance_ready:
        preclearance_score += 15.0
    if live_lane_running:
        preclearance_score += 10.0
    if canary_signal_ready:
        preclearance_score += 10.0
    if weight_ok:
        preclearance_score += 10.0
    if preapproved_supervised_ready:
        preclearance_score += 5.0
    if runnable_after_release_window:
        preclearance_score += 5.0
    preclearance_score = min(round(preclearance_score, 2), 100.0)
    recommended_mode = (
        "supervised_canary"
        if supervised_canary_ready
        else "runnable_pending_release_window"
        if runnable_after_release_window
        else "preapproved_supervised"
        if preapproved_supervised_ready
        else "staged_preclearance"
        if staged_preclearance_ready
        else "validate_only"
    )
    contract_hard_block = bool(live_money_contract_enforced and not live_money_contract_ready)
    overall_status = (
        "ready"
        if supervised_canary_ready
        else "blocked"
        if contract_hard_block
        else "degraded"
        if staged_preclearance_ready or (broker_ready and session_ready and storage_ok)
        else "blocked"
    )

    recommended_actions = ordered_unique(
        [
            "run the live canary only after the promotion packet reaches a signed awaiting-approval or canary-ready state" if not packet_ready else "",
            "treat the current promotion packet as canary-seeded only; finish the remaining promotion repairs before supervised submit" if packet_preclearance_ready and not packet_ready else "",
            "clear runtime separation blockers before allowing even a supervised canary write path" if not clearance_ready or live_lane_should_be_read_only else "",
            "keep live money locked until the A/A+ faithful-live contract clears" if live_money_contract_enforced and not live_money_contract_ready else "",
            "refresh the canary rollout guard so supervised canary permission is backed by recent paper evidence" if not canary_signal_ready else "",
            "the canary package is runnable once the runtime release window opens; the remaining blockers are only the cold-lane/runtime release protections" if runnable_after_release_window else "",
            f"keep the canary weight at or under {float(max_canary_weight):.2f} before supervised live submit" if not weight_ok else "",
            "finish the cold-lane/runtime clearance steps to convert staged preclearance into supervised canary permission" if staged_preclearance_ready else "",
            "treat the canary as preapproved and waiting only on the cold-lane/runtime clearance window before supervised submit" if preapproved_supervised_ready else "",
            "keep the system in validate-only mode until broker, session, storage, and live-lane readiness are all green" if blocking_reasons else "",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": supervised_canary_ready,
        "overall_status": overall_status,
        "recommended_mode": recommended_mode,
        "supervised_canary_ready": supervised_canary_ready,
        "staged_preclearance_ready": staged_preclearance_ready,
        "preapproved_supervised_ready": preapproved_supervised_ready,
        "runnable_after_release_window": runnable_after_release_window,
        "blocking_reasons": blocking_reasons,
        "broker_ready": broker_ready,
        "session_ready": session_ready,
        "storage_ok": storage_ok,
        "storage_mode": storage_mode,
        "live_lane_running": live_lane_running,
        "runtime_clearance_state": clearance_state or "unknown",
        "runtime_clearance_recoverable": runtime_clearance_recoverable,
        "live_lane_should_be_read_only": live_lane_should_be_read_only,
        "promotion_packet_ready": packet_ready,
        "promotion_packet_preclearance_ready": packet_preclearance_ready,
        "canary_signal_ready": canary_signal_ready,
        "canary_signal_seed_ready": canary_signal_seed_ready,
        "live_money_contract_enforced": live_money_contract_enforced,
        "live_money_contract_ready": live_money_contract_ready,
        "live_money_contract_hard_block": contract_hard_block,
        "live_money_contract_target_date": live_money_contract.get("target_date", ""),
        "live_money_contract_days_remaining": live_money_contract.get("days_remaining"),
        "target_canary_weight": round(target_canary_weight, 6),
        "applied_canary_weight": round(canary_weight, 6),
        "max_canary_weight": round(float(max_canary_weight), 6),
        "canary_weight_ok": weight_ok,
        "bounded_blocker_count": bounded_blocker_count,
        "preclearance_score": preclearance_score,
        "recommended_actions": recommended_actions,
        "source_artifacts": {
            "broker_readiness": str(health_root / "broker_readiness_latest.json"),
            "session_ready": str(health_root / "session_ready_latest.json"),
            "storage_route_status": str(health_root / "storage_route_status_latest.json"),
            "execution_lane_live": str(health_root / "execution_lane_live_latest.json"),
            "live_runtime_separation_control": str(health_root / "live_runtime_separation_control_latest.json"),
            "promotion_autopilot_packet": str(project_root / "governance" / "champion_challenger" / "promotion_autopilot_packet_latest.json"),
            "canary_auto_tuner": str(health_root / "canary_auto_tuner_latest.json"),
            "canary_rollout_guard": str(health_root / "canary_rollout_latest.json"),
            "live_money_readiness_contract": str(health_root / "live_money_readiness_contract_latest.json"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish a supervised live-canary contract for broker submit gating.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--max-canary-weight", type=float, default=DEFAULT_MAX_CANARY_WEIGHT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve(), max_canary_weight=float(args.max_canary_weight))
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "live_canary_control "
            f"overall_status={payload.get('overall_status', '')} "
            f"supervised_canary_ready={int(bool(payload.get('supervised_canary_ready', False)))} "
            f"mode={payload.get('recommended_mode', '')}"
        )
    return 0 if bool(payload.get("supervised_canary_ready", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
