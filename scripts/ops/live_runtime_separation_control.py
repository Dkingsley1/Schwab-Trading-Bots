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
    from scripts.ops.long_runtime_common import PROJECT_ROOT, eastern_off_hours_window, iso_now, load_json, ordered_unique, payload_age_minutes, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, eastern_off_hours_window, iso_now, load_json, ordered_unique, payload_age_minutes, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "live_runtime_separation_control_latest.json"
OVERLAY_RAW_LIVE_MAX_CORE_LINES = 10_000
OVERLAY_RAW_LIVE_MAX_TOTAL_LINES = 15_000
OVERLAY_RAW_LIVE_MAX_AGE_SECONDS = 15 * 60


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _overlay_only_storage_relief(storage_control: dict[str, Any], runtime_throttle: dict[str, Any]) -> dict[str, Any]:
    backpressure = storage_control.get("backpressure") if isinstance(storage_control.get("backpressure"), dict) else {}
    effective_raw_live = backpressure.get("effective_raw_live") if isinstance(backpressure.get("effective_raw_live"), dict) else {}
    effective_source = str(backpressure.get("effective_raw_live_source") or effective_raw_live.get("source") or "").strip()
    use_effective = bool(
        backpressure.get("overlay_adjusted", False)
        and (backpressure.get("overlay_pressure_clear", False) or effective_source == "fresh_empty_sql_ingestion_overlay")
        and effective_raw_live
    )
    raw_live = effective_raw_live if use_effective else backpressure.get("raw_live") if isinstance(backpressure.get("raw_live"), dict) else {}
    raw_core = _safe_int(raw_live.get("core_pending_lines"), 0)
    raw_total = _safe_int(raw_live.get("total_pending_lines"), 0)
    raw_oldest = _safe_float(raw_live.get("oldest_pending_age_seconds"), 0.0)
    raw_live_clear = bool(
        raw_live
        and raw_core <= OVERLAY_RAW_LIVE_MAX_CORE_LINES
        and raw_total <= OVERLAY_RAW_LIVE_MAX_TOTAL_LINES
        and raw_oldest <= OVERLAY_RAW_LIVE_MAX_AGE_SECONDS
    )
    overlay_adjusted = bool(backpressure.get("overlay_adjusted", False))

    runtime_snapshot = runtime_throttle.get("runtime_snapshot") if isinstance(runtime_throttle.get("runtime_snapshot"), dict) else {}
    runtime_storage = runtime_snapshot.get("storage_pressure") if isinstance(runtime_snapshot.get("storage_pressure"), dict) else {}
    paper_capacity = (
        runtime_throttle.get("paper_capacity_contract")
        if isinstance(runtime_throttle.get("paper_capacity_contract"), dict)
        else {}
    )
    capacity_relief = (
        paper_capacity.get("storage_overlay_capacity_relief")
        if isinstance(paper_capacity.get("storage_overlay_capacity_relief"), dict)
        else {}
    )
    runtime_relief_active = bool(runtime_storage.get("overlay_capacity_relief", False) or capacity_relief.get("active", False))
    active = bool((overlay_adjusted and raw_live_clear) or (runtime_relief_active and raw_live_clear))
    return {
        "active": active,
        "overlay_adjusted": overlay_adjusted,
        "runtime_overlay_capacity_relief": runtime_relief_active,
        "effective_raw_live_used": use_effective,
        "effective_raw_live_source": effective_source,
        "raw_live_clear": raw_live_clear,
        "raw_live": {
            "core_pending_lines": raw_core,
            "total_pending_lines": raw_total,
            "oldest_pending_age_seconds": round(raw_oldest, 3),
            "max_core_pending_lines": OVERLAY_RAW_LIVE_MAX_CORE_LINES,
            "max_total_pending_lines": OVERLAY_RAW_LIVE_MAX_TOTAL_LINES,
            "max_oldest_pending_age_seconds": OVERLAY_RAW_LIVE_MAX_AGE_SECONDS,
        },
        "policy": "treat SQL-overlay-only pressure as bounded for live separation when raw live backlog is cool",
    }


def _near_steady_state_storage_ready(storage_control: dict[str, Any], storage_overlay_relief: dict[str, Any]) -> bool:
    if str(storage_control.get("overall_status") or "").strip().lower() != "ready":
        return False
    if str(storage_control.get("severity") or "").strip().lower() != "stable":
        return False
    external_route = storage_control.get("external_route_verification") if isinstance(storage_control.get("external_route_verification"), dict) else {}
    if str(external_route.get("verification_state") or "").strip().lower() not in {
        "ready",
        "verified",
        "curated_ready",
        "active_passthrough",
        "active_local_ready",
    }:
        return False
    steady_state = storage_control.get("steady_state") if isinstance(storage_control.get("steady_state"), dict) else {}
    target_status = steady_state.get("target_status") if isinstance(steady_state.get("target_status"), dict) else {}
    target_breaches = {str(item) for item in target_status.get("target_breaches") or []}
    raw_live_clear = bool((storage_overlay_relief.get("raw_live") or {}) and storage_overlay_relief.get("raw_live_clear", False))
    return bool(
        raw_live_clear
        and target_breaches.issubset({"pressure_index"})
        and _safe_int(target_status.get("target_breach_count"), len(target_breaches)) <= 1
        and _safe_float(storage_control.get("backpressure_quality_score"), 0.0) >= 92.0
        and _safe_float(storage_control.get("recovery_quality_score"), 0.0) >= 88.0
        and _safe_float(storage_control.get("pressure_index"), 0.0) <= 0.50
    )


def _cold_lane_contract(
    project_root: Path,
    *,
    training_blocked: bool,
    storage_blocked: bool,
    contention_score: int,
) -> dict[str, Any]:
    payload = load_json(project_root / "governance" / "health" / "cold_lane_refresh_latest.json")
    refresh_required = bool(training_blocked or storage_blocked or contention_score > 0)
    reason = str(payload.get("reason") or "").strip().lower()
    overall_status = "ready"
    next_action = "keep reusing the existing cold-lane research artifact while the live plane stays on frozen release bundles"

    if refresh_required and not payload:
        overall_status = "blocked"
        next_action = "run the cold-lane refresh before allowing heavy research work back onto the shared host"
    elif refresh_required and reason == "resource_guard_blocked":
        overall_status = "blocked"
        next_action = "wait for the resource guard to clear before retrying the cold-lane refresh"
    elif refresh_required and reason == "already_running":
        overall_status = "degraded"
        next_action = "let the in-flight cold-lane refresh finish and keep the live plane read-only"
    elif refresh_required and (bool(payload.get("ok", False)) or reason == "fresh_strategy_research_reused"):
        overall_status = "ready"
    elif refresh_required:
        overall_status = "degraded"
        next_action = "refresh the cold lane again before lifting the shared-host protection gates"

    return {
        "overall_status": overall_status,
        "refresh_required": refresh_required,
        "refresh_state": reason or ("not_required" if not refresh_required else "missing"),
        "refresh_ok": bool(payload.get("ok", False)),
        "ran": bool(payload.get("ran", False)),
        "skipped": bool(payload.get("skipped", False)),
        "strategy_age_minutes_after": payload.get("strategy_age_minutes_after"),
        "next_action": next_action,
        "recommended_command": [
            "./scripts/ops/opsctl.sh",
            "cold-lane-refresh",
            "--json",
        ],
    }


def _coverage_clearance_contract(project_root: Path, *, coverage_shortfall_bots: int) -> dict[str, Any]:
    payload = load_json(project_root / "governance" / "walk_forward" / "coverage_gap_closer_latest.json")
    autopilot = payload.get("autopilot_contract") if isinstance(payload.get("autopilot_contract"), dict) else {}
    launch_contract = autopilot.get("launch_contract") if isinstance(autopilot.get("launch_contract"), dict) else {}
    stage_count = int(payload.get("staged_candidate_count", len(payload.get("active_stage_candidates") or [])) or 0)
    overall_status = "ready"
    next_action = "coverage debt is clear enough for promotion gating and shared-host training to proceed"
    launch_state = str(autopilot.get("launch_state") or "").strip().lower()
    auto_launch_pending = bool(launch_contract.get("auto_launch_pending", False) or autopilot.get("auto_launch_pending", False))

    if coverage_shortfall_bots > 0 and not payload:
        overall_status = "blocked"
        next_action = "refresh the coverage gap closer and stage the next candidates before resuming promotion work"
    elif coverage_shortfall_bots > 0 and autopilot:
        overall_status = str(autopilot.get("overall_status") or "degraded")
        next_action = str(autopilot.get("next_action") or "keep staged candidates moving until the walk-forward shortfall is gone")
    elif coverage_shortfall_bots > 0:
        overall_status = "degraded" if stage_count > 0 else "blocked"
        next_action = "stage coverage candidates and keep cycling the light coverage canary until the shortfall is cleared"

    return {
        "overall_status": overall_status,
        "shortfall_bots": int(coverage_shortfall_bots),
        "stage_candidate_count": stage_count,
        "launch_state": launch_state or ("cleared" if coverage_shortfall_bots <= 0 else "unknown"),
        "can_launch_now": bool(autopilot.get("can_launch_now", False)),
        "can_auto_launch_off_hours": bool(autopilot.get("can_auto_launch_off_hours", False)),
        "auto_launch_pending": auto_launch_pending,
        "launch_mode": str(autopilot.get("launch_mode") or ""),
        "off_hours_preferred": bool(autopilot.get("off_hours_preferred", False)),
        "off_hours_window": autopilot.get("off_hours_window") if isinstance(autopilot.get("off_hours_window"), dict) else {},
        "launch_contract": launch_contract,
        "next_action": next_action,
        "recommended_command": [
            "./scripts/ops/opsctl.sh",
            "coverage-gap-closer",
            "--apply-stage",
            ("--auto-launch-off-hours" if bool(autopilot.get("can_auto_launch_off_hours", False) or auto_launch_pending) else "--launch"),
            "--json",
        ],
    }


def build_payload(project_root: Path = PROJECT_ROOT, *, live_fresh_minutes: float = 240.0) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    walk_root = project_root / "governance" / "walk_forward"
    live_readiness_path = health_root / "live_readiness_smoke_latest.json"
    live_readiness = load_json(live_readiness_path)
    training_runtime = load_json(health_root / "training_runtime_control_latest.json")
    storage_tier = load_json(health_root / "storage_tier_policy_latest.json")
    storage_control = load_json(health_root / "ingestion_storage_control_latest.json")
    runtime_throttle = load_json(health_root / "runtime_throttle_control_latest.json")
    resource_guard = load_json(health_root / "resource_guard_latest.json")
    coverage_seed = load_json(walk_root / "coverage_seed_latest.json")
    process_watchdog = load_json(health_root / "process_watchdog_latest.json")
    off_hours = eastern_off_hours_window()

    live_age_minutes = payload_age_minutes(live_readiness, live_readiness_path)
    live_ready = bool(live_readiness.get("ok", False)) and (
        live_age_minutes is None or float(live_age_minutes) <= max(float(live_fresh_minutes), 1.0)
    )
    training_blocked = str(training_runtime.get("overall_status") or "") == "blocked"
    storage_blocked_raw = str(storage_tier.get("overall_status") or "") == "blocked"
    hot_path_over_budget_bytes = int(((storage_tier.get("pressure") or {}).get("hot_path_over_budget_bytes", 0)) or 0)
    storage_target_status = storage_control.get("steady_state", {}).get("target_status", {}) if isinstance(storage_control.get("steady_state"), dict) else {}
    external_route = storage_control.get("external_route_verification") if isinstance(storage_control.get("external_route_verification"), dict) else {}
    storage_steady_state_ready_strict = bool(
        str(storage_control.get("overall_status") or "").strip().lower() == "ready"
        and str(storage_control.get("severity") or "").strip().lower() == "stable"
        and bool(storage_target_status.get("steady_state_ready", False))
        and _safe_float(storage_control.get("backpressure_quality_score"), 0.0) >= 95.0
        and _safe_float(storage_control.get("recovery_quality_score"), 0.0) >= 88.0
        and str(external_route.get("verification_state") or "").strip().lower()
        in {"ready", "verified", "curated_ready", "active_passthrough", "active_local_ready"}
    )
    storage_overlay_relief = _overlay_only_storage_relief(storage_control, runtime_throttle)
    storage_near_steady_state_ready = _near_steady_state_storage_ready(storage_control, storage_overlay_relief)
    storage_steady_state_ready = bool(storage_steady_state_ready_strict or storage_near_steady_state_ready)
    storage_bounded_by_control = bool(storage_steady_state_ready or storage_overlay_relief.get("active", False))
    storage_blocked = bool(storage_blocked_raw and not storage_bounded_by_control)
    coverage_shortfall_bots = int(coverage_seed.get("coverage_shortfall_bots", 0) or 0)
    swap_used_gb = float(resource_guard.get("swap_used_gb", 0.0) or 0.0)
    restart_storms = len(process_watchdog.get("restart_storms") or [])

    contention_signals = {
        "training_runtime_blocked": training_blocked,
        "storage_hot_path_blocked": storage_blocked,
        "storage_hot_path_bounded_by_control": bool(storage_blocked_raw and storage_bounded_by_control),
        "coverage_shortfall_present": coverage_shortfall_bots > 0,
        "swap_pressure_elevated": swap_used_gb >= 8.0,
        "restart_storm_present": restart_storms > 0,
    }
    contention_score = sum(1 for key, value in contention_signals.items() if key != "storage_hot_path_bounded_by_control" and value)

    overall_status = "ready"
    if contention_score >= 3:
        overall_status = "blocked"
    elif contention_score > 0 or not live_ready:
        overall_status = "degraded"

    cold_lane_contract = _cold_lane_contract(
        project_root,
        training_blocked=training_blocked,
        storage_blocked=storage_blocked,
        contention_score=contention_score,
    )
    coverage_clearance = _coverage_clearance_contract(
        project_root,
        coverage_shortfall_bots=coverage_shortfall_bots,
    )
    clearance_state = "ready"
    if contention_score > 0:
        clearance_state = "protect_live"
    if bool(cold_lane_contract.get("refresh_required", False)) and str(cold_lane_contract.get("overall_status") or "") != "ready":
        clearance_state = "awaiting_cold_lane"
    elif coverage_shortfall_bots > 0 and bool(coverage_clearance.get("can_auto_launch_off_hours", False)):
        clearance_state = "off_hours_cold_lane_launch_ready"
    elif coverage_shortfall_bots > 0 and bool(coverage_clearance.get("auto_launch_pending", False)):
        clearance_state = "scheduled_off_hours_launch"
    elif coverage_shortfall_bots > 0 and not bool(coverage_clearance.get("can_launch_now", False)):
        clearance_state = "awaiting_coverage_cycles"
    elif coverage_shortfall_bots > 0:
        clearance_state = "coverage_cycles_ready"

    managed_cold_lane_deferred = bool(
        live_ready
        and coverage_shortfall_bots > 0
        and not training_blocked
        and not storage_blocked
        and swap_used_gb < 8.0
        and restart_storms <= 0
        and str(coverage_clearance.get("overall_status") or "") in {"ready", "degraded"}
        and int(coverage_clearance.get("stage_candidate_count") or 0) >= min(coverage_shortfall_bots, 1)
        and str(cold_lane_contract.get("refresh_state") or "") in {"resource_guard_blocked", "already_running"}
        and bool(cold_lane_contract.get("refresh_required", False))
    )
    coverage_launch_contract = (
        coverage_clearance.get("launch_contract") if isinstance(coverage_clearance.get("launch_contract"), dict) else {}
    )
    managed_coverage_stage_deferred = bool(
        live_ready
        and coverage_shortfall_bots > 0
        and not training_blocked
        and not storage_blocked
        and swap_used_gb < 8.0
        and restart_storms <= 0
        and str(coverage_clearance.get("overall_status") or "") in {"ready", "degraded", "needs_cycles"}
        and int(coverage_clearance.get("stage_candidate_count") or 0) >= min(coverage_shortfall_bots, 1)
        and str(cold_lane_contract.get("overall_status") or "") == "ready"
        and str(cold_lane_contract.get("refresh_state") or "") in {"fresh_strategy_research_reused", "not_required", "auth_success", "ready"}
        and str(coverage_clearance.get("launch_state") or "") in {"stage_only_training_blocked", "stage_only_off_hours", "stage_only", "manual"}
        and (
            bool(coverage_launch_contract.get("training_launch_blocked", False))
            or bool(coverage_clearance.get("off_hours_preferred", False))
            or str(coverage_launch_contract.get("launch_guard") or "") == "off_hours_only"
        )
    )
    if managed_cold_lane_deferred or managed_coverage_stage_deferred:
        clearance_state = "managed_cold_lane_deferred" if managed_cold_lane_deferred else "managed_coverage_stage_deferred"
        overall_status = "ready"

    guarded_live_read_only = bool(
        clearance_state == "protect_live"
        and live_ready
        and training_blocked
        and not storage_blocked
        and coverage_shortfall_bots <= 0
        and swap_used_gb < 8.0
        and restart_storms <= 0
        and str(cold_lane_contract.get("overall_status") or "") == "ready"
    )
    if guarded_live_read_only:
        clearance_state = "guarded_live_read_only"
        overall_status = "ready"

    isolation_grade = "shared_host"
    if contention_score <= 0 and coverage_shortfall_bots <= 0 and not training_blocked and not storage_blocked:
        isolation_grade = "clear"
    elif managed_cold_lane_deferred or managed_coverage_stage_deferred:
        isolation_grade = "managed_cold_lane"
    elif bool(cold_lane_contract.get("refresh_required", False)) and str(cold_lane_contract.get("overall_status") or "") == "ready":
        isolation_grade = "soft_cold_lane"

    clearance_actions = ordered_unique(
        [
            str(cold_lane_contract.get("next_action") or ""),
            str(coverage_clearance.get("next_action") or ""),
            "launch the staged coverage pass in the cold lane during the current off-hours window" if clearance_state == "off_hours_cold_lane_launch_ready" else "",
            "keep the staged coverage pass armed for the next off-hours window so the cold lane can fire without reopening the live lane" if clearance_state == "scheduled_off_hours_launch" else "",
            "keep the live plane on frozen release bundles until both cold-lane refresh and coverage debt clearance stop signaling contention" if contention_score > 0 else "",
        ]
    )

    recommended_actions = ordered_unique(
        [
            "keep live runtime on frozen release bundles while training and coverage jobs run in a cold lane" if contention_score > 0 else "",
            "defer targeted retrains and coverage seeding away from the live host until snapshot cache freshness recovers" if training_blocked else "",
            "offload explanation and shard maintenance to the async lane to protect live decision latency" if storage_blocked else "",
            "trim noncritical workers or recycle training-side services before swap pressure bleeds into live sleeves" if swap_used_gb >= 8.0 else "",
            "treat the current runtime as a live-only lane until walk-forward coverage debt drops" if coverage_shortfall_bots > 0 else "",
            "use the off-hours cold-lane launch window to clear coverage debt without reopening the live lane" if clearance_state == "off_hours_cold_lane_launch_ready" else "",
            *clearance_actions,
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "live_plane": {
            "ready": bool(live_ready),
            "broker_ready": bool(live_readiness.get("broker_ready", False)),
            "session_ready": bool(live_readiness.get("session_ready", False)),
            "live_lane_running": bool(live_readiness.get("live_lane_running", False)),
            "live_readiness_age_minutes": round(float(live_age_minutes), 4) if live_age_minutes is not None else None,
        },
        "training_plane": {
            "overall_status": str(training_runtime.get("overall_status") or ""),
            "snapshot_ready": bool(training_runtime.get("snapshot_ready", False)),
            "precompute_target_count": len(training_runtime.get("precompute_targets") or []),
            "coverage_shortfall_bots": coverage_shortfall_bots,
        },
        "shared_host_pressure": {
            "contention_score": int(contention_score),
            "hot_path_over_budget_bytes": hot_path_over_budget_bytes,
            "swap_used_gb": round(float(swap_used_gb), 3),
            "restart_storms": restart_storms,
            "storage_steady_state_ready": bool(storage_steady_state_ready),
            "storage_steady_state_strict_ready": bool(storage_steady_state_ready_strict),
            "storage_near_steady_state_ready": bool(storage_near_steady_state_ready),
            "storage_overlay_relief": storage_overlay_relief,
            "guarded_live_read_only": guarded_live_read_only,
            "managed_cold_lane_deferred": managed_cold_lane_deferred,
            "managed_coverage_stage_deferred": managed_coverage_stage_deferred,
            "signals": contention_signals,
        },
        "release_contract": {
            "live_lane_should_be_read_only": bool(
                (
                    live_ready
                    or (
                        bool(live_readiness.get("broker_ready", False))
                        and bool(live_readiness.get("session_ready", False))
                    )
                )
                and contention_score > 0
            ),
            "promotions_should_wait_for_cold_lane": bool(training_blocked or coverage_shortfall_bots > 0),
            "shared_host_training_resume_allowed": bool(
                contention_score <= 0
                and coverage_shortfall_bots <= 0
                and not training_blocked
                and not storage_blocked
            ),
            "heavy_research_must_stay_cold_lane": bool(training_blocked or storage_blocked or contention_score > 0),
            "infra_bots": [
                "live_runtime_separation_control",
                "training_runtime_control",
                "storage_tier_policy",
                "walk_forward_coverage_seed",
            ],
        },
        "enclave_contract": {
            "overall_status": overall_status,
            "isolation_grade": isolation_grade,
            "off_hours_window": off_hours,
            "live_host_role": "live_read_only" if contention_score > 0 else "multi_mode_ready",
            "research_host_role": ("cold_lane_only" if training_blocked or storage_blocked or coverage_shortfall_bots > 0 else "shared_host_ok"),
            "cold_lane_refresh_state": str(cold_lane_contract.get("refresh_state") or ""),
            "coverage_launch_state": str(coverage_clearance.get("launch_state") or ""),
            "coverage_launch_window_open": bool(coverage_clearance.get("can_auto_launch_off_hours", False)),
            "coverage_auto_launch_pending": bool(coverage_clearance.get("auto_launch_pending", False)),
        },
        "clearance_plan": {
            "overall_status": overall_status,
            "clearance_state": clearance_state,
            "cold_lane_refresh_required": bool(cold_lane_contract.get("refresh_required", False)),
            "coverage_gap_closer_required": coverage_shortfall_bots > 0,
            "can_resume_shared_host_training": bool(
                contention_score <= 0
                and coverage_shortfall_bots <= 0
                and not training_blocked
                and not storage_blocked
            ),
            "cold_lane_refresh": cold_lane_contract,
            "coverage_gap_closer": coverage_clearance,
            "launch_commitment": {
                "auto_launch_pending": bool(coverage_clearance.get("auto_launch_pending", False)),
                "can_auto_launch_off_hours": bool(coverage_clearance.get("can_auto_launch_off_hours", False)),
                "window_label": str((((coverage_clearance.get("launch_contract") or {}).get("window_label")) or "")),
                "window_start_local": str((((coverage_clearance.get("launch_contract") or {}).get("window_start_local")) or "")),
                "recommended_command": list(coverage_clearance.get("recommended_command") or []),
            },
            "clearance_actions": clearance_actions,
        },
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Track whether live trading is sufficiently separated from training and research pressure.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--live-fresh-minutes", type=float, default=240.0)
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve(), live_fresh_minutes=float(args.live_fresh_minutes))
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "live_runtime_separation_control "
            f"overall_status={payload.get('overall_status', '')} "
            f"contention_score={int(((payload.get('shared_host_pressure') or {}).get('contention_score', 0) or 0))}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
