#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "service_control_plane_latest.json"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in items:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _status_rank(status: str) -> int:
    text = str(status or "").strip().lower()
    if text in {"blocked", "error", "failed"}:
        return 3
    if text in {"degraded", "running", "busy", "missing", "needs_work", "needs_coverage"}:
        return 2
    if text in {
        "advisory",
        "applied_with_followups",
        "managed_paper_soak",
        "ok",
        "prep_only",
        "ready",
        "ready_with_advisories",
        "waiting_for_off_hours",
        "watch",
    }:
        return 1
    return 2


def _rollup_status(statuses: list[str]) -> str:
    rank = max((_status_rank(status) for status in statuses), default=1)
    if rank >= 3:
        return "blocked"
    if rank >= 2:
        return "degraded"
    return "ready"


def _lane(status: str, summary: str, *, raw_status: str = "", details: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "status": str(status or "missing"),
        "summary": str(summary or ""),
    }
    if raw_status:
        payload["raw_status"] = str(raw_status)
    if details:
        payload["details"] = details
    return payload


def _payload_fresh(payload: dict[str, Any], *, max_age_minutes: float = 30.0) -> bool:
    raw = str(payload.get("timestamp_utc") or "").strip().replace("Z", "+00:00")
    if not raw:
        return False
    try:
        parsed = datetime.fromisoformat(raw)
    except Exception:
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds() / 60.0
    return bool(age <= max(float(max_age_minutes), 0.0))


def _provider_lane_status(provider_mesh: dict[str, Any]) -> str:
    raw = str(provider_mesh.get("overall_status") or ("missing" if not provider_mesh else "degraded"))
    if not provider_mesh:
        return raw
    summary = provider_mesh.get("summary") if isinstance(provider_mesh.get("summary"), dict) else {}
    required_collectors = int(summary.get("required_collectors", 0) or 0)
    required_contract_ok = int(summary.get("required_contract_ok", 0) or 0)
    required_snapshot_ready = int(summary.get("required_snapshot_ready", 0) or 0)
    required_failures = int(summary.get("required_failure_count", 0) or 0)
    cooldowns = provider_mesh.get("cooldowns") if isinstance(provider_mesh.get("cooldowns"), list) else []
    required_ready = bool(
        required_collectors > 0
        and required_contract_ok >= required_collectors
        and required_snapshot_ready >= required_collectors
        and required_failures == 0
        and not cooldowns
    )
    if raw in {"degraded", "missing"} and required_ready:
        return "ready"
    return raw


def _runtime_lane_status(runtime_separation: dict[str, Any]) -> str:
    raw = str(runtime_separation.get("overall_status") or ("missing" if not runtime_separation else "degraded"))
    if raw != "degraded" or not runtime_separation:
        return raw
    live_plane = runtime_separation.get("live_plane") if isinstance(runtime_separation.get("live_plane"), dict) else {}
    clearance = runtime_separation.get("clearance_plan") if isinstance(runtime_separation.get("clearance_plan"), dict) else {}
    pressure = runtime_separation.get("shared_host_pressure") if isinstance(runtime_separation.get("shared_host_pressure"), dict) else {}
    signals = pressure.get("signals") if isinstance(pressure.get("signals"), dict) else {}
    clearance_state = str(clearance.get("clearance_state") or "").strip()
    if (
        bool(live_plane.get("ready", False))
        and bool(live_plane.get("broker_ready", True))
        and bool(live_plane.get("session_ready", True))
        and int(pressure.get("contention_score", 0) or 0) <= 3
        and clearance_state
        in {"awaiting_coverage_cycles", "awaiting_cold_lane", "managed_cold_lane_deferred", "managed_coverage_stage_deferred", "protect_live", "ready"}
        and not bool(signals.get("restart_storm_present", False))
        and not bool(signals.get("swap_pressure_elevated", False))
    ):
        return "advisory"
    return raw


def _cockpit_lane_status(operator_cockpit: dict[str, Any]) -> str:
    raw = str(operator_cockpit.get("overall_status") or ("degraded" if operator_cockpit else "missing"))
    if raw != "degraded" or not operator_cockpit:
        return raw
    posture = operator_cockpit.get("adaptive_posture") if isinstance(operator_cockpit.get("adaptive_posture"), dict) else {}
    hard_blockers = posture.get("hard_blockers") if isinstance(posture.get("hard_blockers"), list) else []
    if bool(posture.get("live_collection_ready", False)) and not hard_blockers:
        return "advisory"
    return raw


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    allocator_root = project_root / "governance" / "allocator"
    risk_root = project_root / "governance" / "risk"

    ops_coordinator = _load_json(health_root / "ops_coordinator_latest.json")
    process_watchdog = _load_json(health_root / "process_watchdog_latest.json")
    platform_control = _load_json(health_root / "platform_control_plane_latest.json")
    provider_mesh = _load_json(health_root / "provider_mesh_latest.json")
    allocator = _load_json(allocator_root / "portfolio_allocator_service_latest.json")
    risk_boundary = _load_json(risk_root / "risk_service_boundary_latest.json")
    paper_lane = _load_json(health_root / "execution_lane_paper_latest.json")
    live_lane = _load_json(health_root / "execution_lane_live_latest.json")
    retrain_orchestrator = _load_json(health_root / "retrain_orchestrator_latest.json")
    retrain_launch = _load_json(health_root / "retrain_launch_latest.json")
    retrain_scorecard = _load_json(health_root / "retrain_scorecard_latest.json")
    training_success = _load_json(health_root / "training_success_latest.json")
    event_store = _load_json(health_root / "point_in_time_event_store_latest.json")
    runtime_separation = _load_json(health_root / "live_runtime_separation_control_latest.json")
    operator_cockpit = _load_json(health_root / "operator_cockpit_latest.json")

    restart_storms = len(process_watchdog.get("restart_storms") or [])
    platform_status = str((platform_control.get("institutional_readiness") or {}).get("overall_status") or "")
    process_watchdog_status = str(process_watchdog.get("overall_status") or process_watchdog.get("status") or "")
    control_status = "ready"
    if restart_storms > 0:
        control_status = "blocked"
    elif not ops_coordinator or not process_watchdog:
        control_status = "degraded"
    elif not bool(ops_coordinator.get("ok", False)):
        control_status = "advisory" if process_watchdog_status == "ready" else "degraded"
    elif platform_status and platform_status not in {"ready", "advancing"}:
        control_status = "advisory"
    control_summary = (
        f"ops_ok={int(bool(ops_coordinator.get('ok', False)))} "
        f"restart_storms={restart_storms} "
        f"platform_status={platform_status or 'missing'}"
    )

    provider_status = _provider_lane_status(provider_mesh)
    provider_raw_status = str(provider_mesh.get("overall_status") or "")
    provider_summary = (
        f"required_contract_ok={int(((provider_mesh.get('summary') or {}).get('required_contract_ok', 0) or 0))}/"
        f"{int(((provider_mesh.get('summary') or {}).get('required_collectors', 0) or 0))} "
        f"cooldowns={len(provider_mesh.get('cooldowns') or [])}"
        if provider_mesh
        else "provider_mesh_missing"
    )

    approved_intents = allocator.get("approved_intents") if isinstance(allocator.get("approved_intents"), list) else []
    pre_trade_rows = risk_boundary.get("pre_trade_decisions") if isinstance(risk_boundary.get("pre_trade_decisions"), list) else []
    paper_stale = bool(paper_lane.get("stale", False))
    live_stale = bool(live_lane.get("stale", False))
    execution_status = "ready"
    if not allocator or not risk_boundary:
        execution_status = "degraded"
    elif not bool(allocator.get("ok", False)) or not bool(risk_boundary.get("ok", False)):
        execution_status = "blocked"
    elif paper_stale or live_stale:
        execution_status = "degraded"
    elif len(pre_trade_rows) <= 0 and len(approved_intents) <= 0:
        execution_status = "advisory"
    execution_summary = (
        f"approved_intents={len(approved_intents)} "
        f"pre_trade_orders={len(pre_trade_rows)} "
        f"paper_stale={int(paper_stale)} live_stale={int(live_stale)}"
    )

    launch_state = str(retrain_launch.get("state") or "")
    launch_final_status = str(retrain_launch.get("final_status") or "")
    training_reason = str(training_success.get("reason") or retrain_scorecard.get("training_reason") or "")
    training_reason_lower = training_reason.lower()
    failure_count = int(
        training_success.get("failure_count", retrain_scorecard.get("failure_count", 0)) or 0
    )
    retrain_status = "degraded"
    retrain_raw_status = launch_state or launch_final_status or str(retrain_orchestrator.get("reason") or "")
    if launch_state == "running":
        retrain_status = "running"
    elif bool(training_success.get("confirmed_training_success", False)):
        retrain_status = "ready"
    elif failure_count > 0 or "failure" in training_reason or "blocked" in str(retrain_orchestrator.get("reason") or ""):
        retrain_status = "blocked"
    elif (
        launch_state == "completed"
        and failure_count == 0
        and (
            training_reason in {"", "no_trained_targets", "skipped_by_flag"}
            or "skipped_by_flag" in training_reason_lower
            or "trained_ok_but_not_promotable" in training_reason_lower
        )
    ):
        retrain_status = "managed_paper_soak"
    elif retrain_orchestrator or retrain_launch or retrain_scorecard:
        retrain_status = "degraded"
    retrain_summary = (
        f"launch_state={launch_state or 'none'} "
        f"failure_count={failure_count} "
        f"training_reason={training_reason or 'unknown'}"
    )

    event_count = int(event_store.get("event_count", 0) or 0)
    category_counts = event_store.get("category_counts") if isinstance(event_store.get("category_counts"), dict) else {}
    event_status = "ready" if bool(event_store.get("ok", False)) and event_count > 0 else ("degraded" if event_store else "missing")
    event_summary = f"event_count={event_count} categories={len(category_counts)}"

    runtime_status = _runtime_lane_status(runtime_separation)
    runtime_raw_status = str(runtime_separation.get("overall_status") or "")
    runtime_summary = (
        f"contention_score={int(((runtime_separation.get('shared_host_pressure') or {}).get('contention_score', 0) or 0))}"
        if runtime_separation
        else "runtime_separation_missing"
    )

    cockpit_status = _cockpit_lane_status(operator_cockpit)
    cockpit_raw_status = str(operator_cockpit.get("overall_status") or "")
    cockpit_summary = (
        f"recommended_actions={len(operator_cockpit.get('recommended_actions') or [])}"
        if operator_cockpit
        else "operator_cockpit_latest_missing"
    )

    upgrade_lanes = {
        "control_plane": _lane(control_status, control_summary, details={"restart_storms": restart_storms}),
        "provider_mesh": _lane(provider_status, provider_summary, raw_status=provider_raw_status),
        "execution_gateway": _lane(
            execution_status,
            execution_summary,
            details={
                "approved_intents": len(approved_intents),
                "pre_trade_orders": len(pre_trade_rows),
                "paper_lane_stale": paper_stale,
                "live_lane_stale": live_stale,
            },
        ),
        "retrain_pipeline": _lane(retrain_status, retrain_summary, raw_status=retrain_raw_status),
        "event_history": _lane(event_status, event_summary, details={"event_count": event_count}),
        "runtime_separation": _lane(runtime_status, runtime_summary, raw_status=runtime_raw_status),
        "operator_cockpit_contract": _lane(cockpit_status, cockpit_summary, raw_status=cockpit_raw_status),
    }

    overall_status = _rollup_status([str(row.get("status") or "") for row in upgrade_lanes.values()])
    ready_count = sum(1 for row in upgrade_lanes.values() if str(row.get("status") or "") == "ready")
    recommended_actions = _ordered_unique(
        list(provider_mesh.get("recommended_actions") or [])
        + list(runtime_separation.get("recommended_actions") or [])
        + list(operator_cockpit.get("recommended_actions") or [])
        + [
            "finish wiring the service control plane through ops refresh paths before growing the runtime surface" if control_status != "ready" else "",
            "refresh the allocator and risk boundary before treating live execution as production-ready" if execution_status != "ready" else "",
            "split retrain into resumable stages and gate promotions on stage outputs rather than one monolithic run" if retrain_status in {"blocked", "degraded"} else "",
            "keep the point-in-time event store current so incident review is driven by a single timeline surface" if event_status != "ready" else "",
        ]
    )[:14]

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "summary": {
            "upgrade_count": 7,
            "ready_count": ready_count,
            "completion_score": round((ready_count / 7.0) * 100.0, 2),
        },
        "upgrade_lanes": upgrade_lanes,
        "recommended_actions": recommended_actions,
        "surfaces": {
            "ops_coordinator": str(health_root / "ops_coordinator_latest.json"),
            "process_watchdog": str(health_root / "process_watchdog_latest.json"),
            "platform_control_plane": str(health_root / "platform_control_plane_latest.json"),
            "provider_mesh": str(health_root / "provider_mesh_latest.json"),
            "portfolio_allocator_service": str(allocator_root / "portfolio_allocator_service_latest.json"),
            "risk_service_boundary": str(risk_root / "risk_service_boundary_latest.json"),
            "execution_lane_paper": str(health_root / "execution_lane_paper_latest.json"),
            "execution_lane_live": str(health_root / "execution_lane_live_latest.json"),
            "retrain_orchestrator": str(health_root / "retrain_orchestrator_latest.json"),
            "retrain_launch": str(health_root / "retrain_launch_latest.json"),
            "retrain_scorecard": str(health_root / "retrain_scorecard_latest.json"),
            "point_in_time_event_store": str(health_root / "point_in_time_event_store_latest.json"),
            "live_runtime_separation_control": str(health_root / "live_runtime_separation_control_latest.json"),
            "operator_cockpit": str(health_root / "operator_cockpit_latest.json"),
        },
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish one service-level control plane across orchestration, providers, execution, retrain, event history, and runtime separation.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve())
    out_path = Path(args.out_file).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "service_control_plane "
            f"overall_status={payload.get('overall_status', '')} "
            f"completion_score={float(((payload.get('summary') or {}).get('completion_score', 0.0) or 0.0)):.2f}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
