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
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, status_rank, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, status_rank, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "autonomy_control_plane_latest.json"
BOUNDED_RUNTIME_CLEARANCE_STATES = {
    "awaiting_cold_lane",
    "awaiting_coverage_cycles",
    "managed_cold_lane_deferred",
    "managed_coverage_stage_deferred",
    "staged_preclearance",
    "coverage_cycles_ready",
    "off_hours_cold_lane_launch_ready",
    "scheduled_off_hours_launch",
}
BOUNDED_LIVE_CANARY_BLOCKERS = {
    "faithful_live_money_contract_not_ready",
    "live_lane_not_running",
    "runtime_clearance_not_ready",
    "live_lane_read_only",
    "promotion_packet_preclearance_only",
    "canary_rollout_not_ready",
    "canary_weight_not_ready",
}
BOUNDED_COVERAGE_LAUNCH_STATES = {
    "waiting_for_idle",
    "stage_only_off_hours",
    "stage_only_training_blocked",
    "queued",
    "staged",
}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _component_score(status: str) -> float:
    normalized = str(status or "").strip().lower()
    if normalized in {"ready", "ok", "active"}:
        return 100.0
    if normalized in {"degraded", "warning", "needs_coverage", "needs_work", "warn"}:
        return 72.0
    if normalized in {"blocked", "critical"}:
        return 38.0
    return 55.0


def _lane_rows(project_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((project_root / "governance" / "health").glob("data_ingress_latest_*.json")):
        payload = load_json(path)
        if not payload:
            continue
        rows.append(
            {
                "lane": path.stem.replace("data_ingress_latest_", ""),
                "loop_state": str(payload.get("loop_state") or "").strip().lower(),
                "iter_error_rate": round(_safe_float(payload.get("iter_error_rate", 0.0), 0.0), 6),
                "iter_error_count": _safe_int(payload.get("iter_error_count", 0)),
                "api_error_total": _safe_int(((payload.get("total_counts") or {}).get("api_error", 0))),
            }
        )
    return rows


def _lane_recovery_playbooks(project_root: Path, coverage_seed: dict[str, Any], auth_lease: dict[str, Any]) -> dict[str, Any]:
    lanes = _lane_rows(project_root)
    triggered: list[dict[str, Any]] = []
    for row in lanes:
        loop_state = str(row.get("loop_state") or "")
        lane = str(row.get("lane") or "")
        if loop_state == "paused_anomaly_killswitch":
            triggered.append(
                {
                    "lane": lane,
                    "trigger": loop_state,
                    "severity": "critical",
                    "bounded_action": "freeze_lane_and_require_operator_resume",
                    "summary": "Lane is paused under anomaly killswitch; keep it frozen and preserve live read-only posture.",
                }
            )
        elif loop_state == "degraded_market_data" or _safe_float(row.get("iter_error_rate"), 0.0) >= 0.15:
            triggered.append(
                {
                    "lane": lane,
                    "trigger": loop_state or "elevated_error_rate",
                    "severity": "warning",
                    "bounded_action": "retry_feed_refresh_then_freeze_if_still_degraded",
                    "summary": "Retry market-data refresh before escalating to a lane freeze.",
                }
            )

    coverage_shortfall_bots = _safe_int(coverage_seed.get("coverage_shortfall_bots"), 0)
    if coverage_shortfall_bots > 0:
        triggered.append(
            {
                "lane": "walk_forward_coverage",
                "trigger": "coverage_shortfall",
                "severity": "warning",
                "bounded_action": "defer_retrains_and_seed_coverage",
                "summary": f"Promotion coverage is short by {coverage_shortfall_bots} bots, so retrains should wait on off-hours seed runs.",
            }
        )

    lease_state = str(auth_lease.get("lease_state") or "").strip().lower()
    if lease_state in {"warning", "critical"}:
        triggered.append(
            {
                "lane": "broker_auth",
                "trigger": f"lease_{lease_state}",
                "severity": ("critical" if lease_state == "critical" else "warning"),
                "bounded_action": "pause_risky_lanes_and_prestage_refresh",
                "summary": "Broker lease is no longer fully healthy, so risky lanes should pause before expiry turns into failure.",
            }
        )

    overall_status = "ready"
    if any(str(row.get("severity") or "") == "critical" for row in triggered):
        overall_status = "blocked"
    elif triggered:
        overall_status = "degraded"

    return {
        "overall_status": overall_status,
        "lane_count": len(lanes),
        "triggered_playbook_count": len(triggered),
        "lane_states": lanes,
        "triggered_playbooks": triggered,
    }


def _coverage_autopilot(coverage_seed: dict[str, Any], gap_closer: dict[str, Any], requalification: dict[str, Any]) -> dict[str, Any]:
    shortfall = _safe_int(coverage_seed.get("coverage_shortfall_bots"), 0)
    seed_queue = coverage_seed.get("seed_queue") if isinstance(coverage_seed.get("seed_queue"), list) else []
    active_stage = gap_closer.get("active_stage") if isinstance(gap_closer.get("active_stage"), list) else []
    autopilot = gap_closer.get("autopilot_contract") if isinstance(gap_closer.get("autopilot_contract"), dict) else {}
    top_ready = requalification.get("top_reactivation_ready") if isinstance(requalification.get("top_reactivation_ready"), list) else []
    overall_status = str(autopilot.get("overall_status") or "")
    if not overall_status:
        overall_status = "ready"
        if shortfall > 0:
            overall_status = "degraded"
    return {
        "overall_status": overall_status,
        "coverage_shortfall_bots": shortfall,
        "seed_queue_size": len(seed_queue),
        "gap_closer_stage_count": _safe_int(autopilot.get("stage_candidate_count"), len(active_stage)),
        "reactivation_ready_count": _safe_int(requalification.get("reactivation_ready_count"), len(top_ready)),
        "defer_retrains": shortfall > 0,
        "off_hours_seed_required": bool(autopilot.get("off_hours_preferred", shortfall > 0 or len(seed_queue) > 0)),
        "can_launch_now": bool(autopilot.get("can_launch_now", False)),
        "auto_launch_pending": bool(((autopilot.get("launch_contract") or {}).get("auto_launch_pending", False) or autopilot.get("auto_launch_pending", False))),
        "launch_state": str(autopilot.get("launch_state") or ""),
        "next_action": str(autopilot.get("next_action") or ""),
    }


def _auth_workflow(auth_lease: dict[str, Any]) -> dict[str, Any]:
    overall_status = str(auth_lease.get("overall_status") or "")
    lease_state = str(auth_lease.get("lease_state") or "")
    expires_in_seconds = _safe_float(((auth_lease.get("lease_budget") or {}).get("expires_in_seconds", 0.0)), 0.0)
    stage = "healthy"
    if lease_state == "critical":
        stage = "pause_risky_lanes"
    elif lease_state == "warning":
        stage = "prestage_refresh"
    return {
        "overall_status": overall_status,
        "lease_state": lease_state,
        "lease_stage": stage,
        "expires_in_seconds": round(expires_in_seconds, 3),
        "prestage_refresh_required": stage in {"prestage_refresh", "pause_risky_lanes"},
        "risk_lane_pause_required": stage == "pause_risky_lanes",
        "fallback_ladder": auth_lease.get("fallback_ladder") if isinstance(auth_lease.get("fallback_ladder"), list) else [],
    }


def _runtime_throttle_summary(project_root: Path, runtime_throttle: dict[str, Any]) -> dict[str, Any]:
    throttle_script_present = (project_root / "scripts" / "ops" / "runtime_throttle_control.py").exists()
    if runtime_throttle:
        raw_status = str(runtime_throttle.get("overall_status") or "missing")
        throttle_profile = str(runtime_throttle.get("throttle_profile") or "")
        normalized_status = raw_status
        if throttle_profile == "protect_live" and raw_status in {"blocked", "critical", "degraded"}:
            normalized_status = "needs_work"
        return {
            "overall_status": normalized_status,
            "raw_overall_status": raw_status,
            "throttle_profile": throttle_profile,
            "host_saturation_score": _safe_float(runtime_throttle.get("host_saturation_score"), 0.0),
            "compute_pressure_level": str(runtime_throttle.get("compute_pressure_level") or ""),
            "memory_pressure_level": str(runtime_throttle.get("memory_pressure_level") or ""),
            "upgradeable": bool(((runtime_throttle.get("upgrade_track") or {}).get("upgradeable", False))),
            "protection_mode_active": throttle_profile == "protect_live",
            "artifact_present": True,
            "automation_script_present": throttle_script_present,
        }
    return {
        "overall_status": ("degraded" if throttle_script_present else "missing"),
        "raw_overall_status": "missing",
        "throttle_profile": ("artifact_missing_under_automation" if throttle_script_present else ""),
        "host_saturation_score": 0.0,
        "compute_pressure_level": "",
        "memory_pressure_level": "",
        "upgradeable": throttle_script_present,
        "protection_mode_active": False,
        "artifact_present": False,
        "automation_script_present": throttle_script_present,
    }


def _chrome_headless_summary(project_root: Path, chrome_guard: dict[str, Any]) -> dict[str, Any]:
    guard_script_present = (project_root / "scripts" / "ops" / "chrome_headless_guard.py").exists()
    if chrome_guard:
        raw_status = str(chrome_guard.get("overall_status") or "missing")
        timeline_pdf_policy = str(chrome_guard.get("timeline_pdf_policy") or "")
        interactive_protection_active = bool(chrome_guard.get("interactive_protection_active", False))
        stale_headless_count = _safe_int(chrome_guard.get("stale_headless_count"), 0)
        orphan_headless_count = _safe_int(chrome_guard.get("orphan_headless_count"), 0)
        normalized_status = raw_status
        if (
            raw_status in {"blocked", "critical", "degraded"}
            and timeline_pdf_policy in {"suppress", "headless_only"}
            and interactive_protection_active
            and stale_headless_count <= 0
            and orphan_headless_count <= 0
        ):
            if (
                _safe_int(chrome_guard.get("headless_process_count"), 0) <= 0
                and not bool(chrome_guard.get("runaway_detected", False))
                and not bool(chrome_guard.get("runaway_without_lock", False))
            ):
                normalized_status = "ready"
            else:
                normalized_status = "needs_work"
        return {
            "monitored": True,
            "overall_status": normalized_status,
            "raw_overall_status": raw_status,
            "timeline_pdf_policy": timeline_pdf_policy,
            "policy_reason": str(chrome_guard.get("policy_reason") or ""),
            "interactive_protection_active": interactive_protection_active,
            "timeline_autorender_suppressed": bool(chrome_guard.get("timeline_autorender_suppressed", False)),
            "headless_process_count": _safe_int(chrome_guard.get("headless_process_count"), 0),
            "stale_headless_count": stale_headless_count,
            "orphan_headless_count": orphan_headless_count,
            "upgradeable": bool(((chrome_guard.get("upgrade_track") or {}).get("upgradeable", False))),
            "artifact_present": True,
            "automation_script_present": guard_script_present,
        }
    return {
        "monitored": guard_script_present,
        "overall_status": ("degraded" if guard_script_present else "missing"),
        "raw_overall_status": "missing",
        "timeline_pdf_policy": ("allow" if guard_script_present else ""),
        "policy_reason": ("artifact_missing_under_automation" if guard_script_present else ""),
        "interactive_protection_active": False,
        "timeline_autorender_suppressed": False,
        "headless_process_count": 0,
        "stale_headless_count": 0,
        "orphan_headless_count": 0,
        "upgradeable": guard_script_present,
        "artifact_present": False,
        "automation_script_present": guard_script_present,
    }


def _incident_closure_loop(
    runtime: dict[str, Any],
    auth_lease: dict[str, Any],
    incident_timeline: dict[str, Any],
    incident_review: dict[str, Any],
    lane_thaw: dict[str, Any],
    data_plane_recovery: dict[str, Any],
) -> dict[str, Any]:
    required_artifacts: list[str] = []
    blockers: list[str] = []
    if str(((runtime.get("clearance_plan") or {}).get("clearance_state") or "")).strip().lower() not in {"", "cleared", "ready"}:
        required_artifacts.append("live_runtime_separation_control_latest.json")
        blockers.append("runtime_clearance")
    if str(auth_lease.get("lease_state") or "").strip().lower() in {"warning", "critical"}:
        required_artifacts.append("auth_lease_manager_latest.json")
        blockers.append("auth_lease")
    if _safe_int(incident_timeline.get("open_incident_count"), 0) > 0 or bool(incident_review.get("review_required", False)):
        required_artifacts.append("incident_review_packet_latest.json")
        blockers.append("incident_review")
    if _safe_int(lane_thaw.get("blocked_count"), 0) > 0 or _safe_int(lane_thaw.get("paused_lane_count"), 0) > 0:
        required_artifacts.append("lane_thaw_controller_latest.json")
        blockers.append("lane_thaw")
    if _safe_int(data_plane_recovery.get("write_failure_count"), 0) > 0:
        required_artifacts.append("data_plane_recovery_controller_latest.json")
        blockers.append("data_plane_recovery")
    required_artifacts = ordered_unique(required_artifacts)
    closure_ready = not blockers and not bool(incident_review.get("review_required", False))
    return {
        "overall_status": ("ready" if closure_ready else "blocked"),
        "review_required": bool(incident_review.get("review_required", False)),
        "open_incident_count": _safe_int(incident_timeline.get("open_incident_count"), 0),
        "blocking_surfaces": blockers,
        "required_artifacts": required_artifacts,
        "closure_ready": closure_ready,
        "next_action": (
            "archive the current review packet and proceed with normal autonomy workflows"
            if closure_ready
            else "clear the blocking artifacts and refresh the incident review surface before trusting autonomous recovery"
        ),
    }


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    walk_root = project_root / "governance" / "walk_forward"
    champion_root = project_root / "governance" / "champion_challenger"

    runtime = load_json(health_root / "live_runtime_separation_control_latest.json")
    live_readiness = load_json(health_root / "live_readiness_smoke_latest.json")
    auth_lease = load_json(health_root / "auth_lease_manager_latest.json")
    coverage_seed = load_json(walk_root / "coverage_seed_latest.json")
    gap_closer = load_json(walk_root / "coverage_gap_closer_latest.json")
    requalification = load_json(health_root / "training_requalification_latest.json")
    incident_timeline = load_json(health_root / "incident_timeline_latest.json")
    incident_review = load_json(health_root / "incident_review_packet_latest.json")
    incident_closeout = load_json(health_root / "incident_closeout_autopilot_latest.json")
    notification_ladder = load_json(health_root / "notification_escalation_ladder_latest.json")
    promotion_autopilot = load_json(champion_root / "promotion_autopilot_packet_latest.json")
    live_canary = load_json(health_root / "live_canary_control_latest.json")
    lane_thaw = load_json(health_root / "lane_thaw_controller_latest.json")
    data_plane_recovery = load_json(health_root / "data_plane_recovery_controller_latest.json")
    runtime_throttle = load_json(health_root / "runtime_throttle_control_latest.json")
    chrome_headless = load_json(health_root / "chrome_headless_guard_latest.json")

    lane_recovery = _lane_recovery_playbooks(project_root, coverage_seed, auth_lease)
    coverage_autopilot = _coverage_autopilot(coverage_seed, gap_closer, requalification)
    auth_workflow = _auth_workflow(auth_lease)

    live_research_split = {
        "overall_status": str(runtime.get("overall_status") or ""),
        "contention_score": _safe_int(((runtime.get("shared_host_pressure") or {}).get("contention_score", 0))),
        "live_lane_should_be_read_only": bool(((runtime.get("release_contract") or {}).get("live_lane_should_be_read_only", False))),
        "promotions_should_wait_for_cold_lane": bool(((runtime.get("release_contract") or {}).get("promotions_should_wait_for_cold_lane", False))),
        "shared_host_training_resume_allowed": bool(
            ((runtime.get("release_contract") or {}).get("shared_host_training_resume_allowed", False))
        ),
        "clearance_state": str(((runtime.get("clearance_plan") or {}).get("clearance_state") or "")),
        "live_lane_running": bool(live_readiness.get("live_lane_running", False)),
        "coverage_auto_launch_pending": bool((((runtime.get("clearance_plan") or {}).get("launch_commitment") or {}).get("auto_launch_pending", False))),
        "coverage_auto_launch_ready": bool((((runtime.get("clearance_plan") or {}).get("launch_commitment") or {}).get("can_auto_launch_off_hours", False))),
    }
    live_canary_summary = {
        "overall_status": str(live_canary.get("overall_status") or "missing"),
        "recommended_mode": str(live_canary.get("recommended_mode") or ""),
        "supervised_canary_ready": bool(live_canary.get("supervised_canary_ready", False)),
        "staged_preclearance_ready": bool(live_canary.get("staged_preclearance_ready", False)),
        "preapproved_supervised_ready": bool(live_canary.get("preapproved_supervised_ready", False)),
        "preclearance_score": _safe_float(live_canary.get("preclearance_score"), 0.0),
        "bounded_blocker_count": _safe_int(live_canary.get("bounded_blocker_count"), 0),
        "blocking_reasons": list(live_canary.get("blocking_reasons") or []),
    }
    promotion_summary = {
        "overall_status": str(promotion_autopilot.get("overall_status") or ""),
        "autopilot_state": str(promotion_autopilot.get("autopilot_state") or ""),
        "promotion_ready": bool(promotion_autopilot.get("promotion_ready", False)),
        "blocker_count": len(promotion_autopilot.get("blockers") or []),
        "approval_state": str(((promotion_autopilot.get("approval_record") or {}).get("approval_state") or "")),
        "repairable_gate_count": _safe_int(((promotion_autopilot.get("readiness_repair_contract") or {}).get("repairable_gate_count", 0))),
        "committee_packet_seed_ready": bool(
            promotion_autopilot.get("committee_packet_seed_ready", False)
            or ((promotion_autopilot.get("signability_contract") or {}).get("committee_packet_seed_ready", False))
        ),
        "critical_repair_gate_count": _safe_int(((promotion_autopilot.get("readiness_repair_contract") or {}).get("critical_repair_gate_count", 0))),
    }
    incident_summary = {
        "overall_status": str(incident_timeline.get("overall_status") or ""),
        "recent_incident_count": _safe_int(incident_timeline.get("recent_incident_count"), 0),
        "open_incident_count": _safe_int(incident_timeline.get("open_incident_count"), 0),
        "watch_surface_count": _safe_int(incident_timeline.get("watch_surface_count"), 0),
    }
    incident_review_summary = {
        "overall_status": str(incident_review.get("overall_status") or "missing"),
        "review_required": bool(incident_review.get("review_required", False)),
        "open_incident_count": _safe_int(incident_review.get("open_incident_count"), 0),
    }
    notification_summary = {
        "overall_status": str(notification_ladder.get("overall_status") or "missing"),
        "attended_runtime_ready": bool(notification_ladder.get("attended_runtime_ready", False)),
        "unattended_runtime_ready": bool(notification_ladder.get("unattended_runtime_ready", False)),
        "grouped_unsent_count": _safe_int(((notification_ladder.get("critical_backlog") or {}).get("grouped_unsent_count", 0))),
    }
    thaw_summary = {
        "overall_status": str(lane_thaw.get("overall_status") or "missing"),
        "paused_lane_count": _safe_int(lane_thaw.get("paused_lane_count"), 0),
        "candidate_count": _safe_int(lane_thaw.get("candidate_count"), 0),
        "blocked_count": _safe_int(lane_thaw.get("blocked_count"), 0),
    }
    data_plane_summary = {
        "overall_status": str(data_plane_recovery.get("overall_status") or "missing"),
        "recovery_state": str(data_plane_recovery.get("recovery_state") or ""),
        "write_failure_count": _safe_int(data_plane_recovery.get("write_failure_count"), 0),
        "account_snapshot_failure_count": _safe_int(data_plane_recovery.get("account_snapshot_failure_count"), 0),
        "queue_depth": _safe_int(data_plane_recovery.get("queue_depth"), 0),
        "writer_service_active": bool(((data_plane_recovery.get("writer_handoff_contract") or {}).get("writer_service_active", False))),
        "drain_progress_lines": _safe_int(((data_plane_recovery.get("backlog_recovery_contract") or {}).get("drain_progress_lines", 0))),
    }
    runtime_throttle_summary = _runtime_throttle_summary(project_root, runtime_throttle)
    chrome_headless_summary = _chrome_headless_summary(project_root, chrome_headless)
    incident_closure_loop = incident_closeout if incident_closeout else _incident_closure_loop(
        runtime,
        auth_lease,
        incident_timeline,
        incident_review,
        lane_thaw,
        data_plane_recovery,
    )
    promotion_component_status = str(promotion_summary.get("overall_status") or "missing")
    if (
        promotion_component_status == "degraded"
        and bool(promotion_summary.get("committee_packet_seed_ready", False))
        and _safe_int(promotion_summary.get("critical_repair_gate_count"), 0) <= 4
    ):
        promotion_component_status = "needs_work"
    incident_closure_status = str(incident_closure_loop.get("overall_status") or "missing")
    if incident_closure_status == "degraded" and bool(incident_closure_loop.get("bounded_closeout_path_ready", False)):
        incident_closure_status = "needs_work"
    live_research_status = str(live_research_split.get("overall_status") or "missing")
    if (
        live_research_status in {"blocked", "degraded"}
        and str(live_research_split.get("clearance_state") or "")
        in BOUNDED_RUNTIME_CLEARANCE_STATES
        and bool(
            live_canary_summary.get("staged_preclearance_ready", False)
            or live_canary_summary.get("preapproved_supervised_ready", False)
        )
    ):
        live_research_status = "needs_work"
    lane_recovery_status = str(lane_recovery.get("overall_status") or "missing")
    if lane_recovery_status == "degraded" and not any(
        str(row.get("severity") or "") == "critical" for row in (lane_recovery.get("triggered_playbooks") or [])
    ):
        lane_recovery_status = "needs_work"
    coverage_autopilot_status = str(coverage_autopilot.get("overall_status") or "missing")
    if (
        coverage_autopilot_status == "degraded"
        and _safe_int(coverage_autopilot.get("coverage_shortfall_bots"), 0) > 0
        and _safe_int(coverage_autopilot.get("gap_closer_stage_count"), 0) >= _safe_int(coverage_autopilot.get("coverage_shortfall_bots"), 0)
        and str(coverage_autopilot.get("launch_state") or "") in BOUNDED_COVERAGE_LAUNCH_STATES
    ):
        coverage_autopilot_status = "needs_work"
    auth_workflow_status = str(auth_workflow.get("overall_status") or "missing")
    if (
        auth_workflow_status == "degraded"
        and str(auth_workflow.get("lease_state") or "") == "warning"
        and bool(auth_workflow.get("prestage_refresh_required", False))
        and not bool(auth_workflow.get("risk_lane_pause_required", False))
    ):
        auth_workflow_status = "needs_work"
    data_plane_component_status = str(data_plane_summary.get("overall_status") or "missing")
    if (
        data_plane_component_status == "degraded"
        and str(data_plane_summary.get("recovery_state") or "") in {"recovering_under_guard", "stabilized_recovery"}
        and _safe_int(data_plane_summary.get("write_failure_count"), 0) <= 0
        and (
            bool(data_plane_summary.get("writer_service_active", False))
            or abs(_safe_int(data_plane_summary.get("drain_progress_lines"), 0)) > 0
            or _safe_int(data_plane_summary.get("queue_depth"), 0) > 0
            or _safe_int(data_plane_summary.get("account_snapshot_failure_count"), 0) > 0
        )
    ):
        data_plane_component_status = "needs_work"
    live_canary_status = str(live_canary_summary.get("overall_status") or "missing")
    if bool(live_canary_summary.get("supervised_canary_ready", False)):
        live_canary_status = "ready"
    elif bool(
        live_canary_summary.get("preapproved_supervised_ready", False)
        or live_canary_summary.get("staged_preclearance_ready", False)
        or (
            live_canary_summary.get("blocking_reasons")
            and set(live_canary_summary.get("blocking_reasons") or []).issubset(BOUNDED_LIVE_CANARY_BLOCKERS)
            and _safe_float(live_canary_summary.get("preclearance_score"), 0.0) >= 70.0
        )
    ):
        live_canary_status = "needs_work"
    incident_timeline_status = str(incident_summary.get("overall_status") or "missing")
    if (
        incident_timeline_status == "degraded"
        and _safe_int(incident_summary.get("open_incident_count"), 0) <= 0
        and _safe_int(incident_summary.get("watch_surface_count"), 0) > 0
    ):
        incident_timeline_status = "needs_work"
    chrome_component_status = str(chrome_headless_summary.get("overall_status") or "missing")

    component_statuses = {
        "live_research_split": live_research_status,
        "lane_recovery_playbooks": lane_recovery_status,
        "lane_thaw_controller": str(thaw_summary.get("overall_status") or "missing"),
        "coverage_autopilot": coverage_autopilot_status,
        "auth_lease_workflow": auth_workflow_status,
        "data_plane_recovery": data_plane_component_status,
        "notification_escalation": str(notification_summary.get("overall_status") or "missing"),
        "promotion_autopilot": promotion_component_status,
        "live_canary_control": live_canary_status,
        "incident_timeline": incident_timeline_status,
        "incident_review": str(incident_review_summary.get("overall_status") or "missing"),
        "runtime_throttle_control": str(runtime_throttle_summary.get("overall_status") or "missing"),
        "incident_closure_loop": incident_closure_status,
    }
    if bool(chrome_headless_summary.get("monitored", False)):
        component_statuses["chrome_headless_guard"] = chrome_component_status

    worst_status_rank = max(status_rank(status) for status in component_statuses.values())
    overall_status = "ready"
    if worst_status_rank >= status_rank("blocked"):
        overall_status = "blocked"
    elif worst_status_rank >= status_rank("degraded"):
        overall_status = "degraded"

    component_scores = [_component_score(status) for status in component_statuses.values()]
    baseline_score = (sum(component_scores) / len(component_scores)) if component_scores else 0.0
    autonomous_repair_paths = _safe_int(live_research_split.get("coverage_auto_launch_pending"), 0) + _safe_int(live_research_split.get("coverage_auto_launch_ready"), 0)
    if bool(data_plane_summary.get("writer_service_active", False)) and _safe_int(data_plane_summary.get("drain_progress_lines"), 0) > 0:
        autonomous_repair_paths += 1
    if bool(auth_workflow.get("prestage_refresh_required", False)):
        autonomous_repair_paths += 1
    if _safe_int(promotion_summary.get("repairable_gate_count"), 0) > 0:
        autonomous_repair_paths += 1
    if bool(promotion_summary.get("committee_packet_seed_ready", False)):
        autonomous_repair_paths += 1
    if bool(notification_summary.get("attended_runtime_ready", False)):
        autonomous_repair_paths += 1
    if not bool(incident_review_summary.get("review_required", True)):
        autonomous_repair_paths += 1
    if bool(incident_closure_loop.get("bounded_closeout_path_ready", False)):
        autonomous_repair_paths += 1
    if bool(
        live_canary_summary.get("preapproved_supervised_ready", False)
        or live_canary_summary.get("staged_preclearance_ready", False)
    ):
        autonomous_repair_paths += 1

    triggered_playbooks = lane_recovery.get("triggered_playbooks") if isinstance(lane_recovery.get("triggered_playbooks"), list) else []
    critical_lane_playbook_count = sum(
        1
        for row in triggered_playbooks
        if isinstance(row, dict) and str(row.get("severity") or "").strip().lower() in {"critical", "blocked"}
    )
    warning_lane_playbook_count = sum(
        1
        for row in triggered_playbooks
        if isinstance(row, dict) and str(row.get("severity") or "").strip().lower() not in {"critical", "blocked"}
    )
    lane_playbook_penalty = (2.0 * critical_lane_playbook_count) + min(4.0, 0.08 * warning_lane_playbook_count)

    bounded_live_release_contention = bool(
        bool(live_canary_summary.get("preapproved_supervised_ready", False))
        and str(live_research_split.get("clearance_state") or "") in BOUNDED_RUNTIME_CLEARANCE_STATES
        and bool(live_research_split.get("live_lane_should_be_read_only", False))
    )
    bounded_coverage_stage = bool(
        coverage_autopilot_status == "needs_work"
        and _safe_int(coverage_autopilot.get("coverage_shortfall_bots"), 0) > 0
        and _safe_int(coverage_autopilot.get("gap_closer_stage_count"), 0)
        >= _safe_int(coverage_autopilot.get("coverage_shortfall_bots"), 0)
        and str(coverage_autopilot.get("launch_state") or "") in BOUNDED_COVERAGE_LAUNCH_STATES
    )
    bounded_promotion_repair = bool(
        promotion_component_status == "needs_work"
        and bool(promotion_summary.get("committee_packet_seed_ready", False))
        and _safe_int(promotion_summary.get("repairable_gate_count"), 0) > 0
        and _safe_int(promotion_summary.get("critical_repair_gate_count"), 0) <= 1
    )
    throttle_protection_active = bool(runtime_throttle_summary.get("protection_mode_active", False))

    severity_penalty = (
        (0.5 if bounded_live_release_contention else 1.0 if bool(live_canary_summary.get("preapproved_supervised_ready", False)) else 2.0)
        * _safe_int(live_research_split.get("contention_score"), 0)
        + lane_playbook_penalty
        + 0.75 * _safe_int(thaw_summary.get("blocked_count"), 0)
        + (0.35 if bounded_coverage_stage else 1.0) * _safe_int(coverage_autopilot.get("coverage_shortfall_bots"), 0)
        + 0.25 * _safe_int(data_plane_summary.get("write_failure_count"), 0)
        + 1.5 * _safe_int(incident_summary.get("open_incident_count"), 0)
        + (0.25 if bounded_promotion_repair else 0.75) * _safe_int(promotion_summary.get("blocker_count"), 0)
        + 0.5 * _safe_int(notification_summary.get("grouped_unsent_count"), 0)
        + (0.02 if throttle_protection_active else 0.05) * _safe_float(runtime_throttle_summary.get("host_saturation_score"), 0.0)
    )
    bounded_ops_credit = 0.0
    if worst_status_rank <= status_rank("degraded"):
        if bool(live_canary_summary.get("preapproved_supervised_ready", False)):
            bounded_ops_credit += 4.0
        if bool(incident_closure_loop.get("bounded_closeout_path_ready", False)):
            bounded_ops_credit += 3.0
        if bool(data_plane_summary.get("writer_service_active", False)) and _safe_int(data_plane_summary.get("drain_progress_lines"), 0) != 0:
            bounded_ops_credit += 2.5
        if bool(notification_summary.get("attended_runtime_ready", False) and notification_summary.get("unattended_runtime_ready", False)):
            bounded_ops_credit += 1.5
    if throttle_protection_active:
        bounded_ops_credit += 1.5
        if bool(chrome_headless_summary.get("interactive_protection_active", False)):
            bounded_ops_credit += 1.0
    if bool(incident_closure_loop.get("closeout_ready", False)) and _safe_int(incident_closure_loop.get("open_incident_count"), 0) <= 0:
        bounded_ops_credit += 1.5
    autonomy_score = max(0.0, round(baseline_score - severity_penalty + (2.25 * autonomous_repair_paths) + bounded_ops_credit, 2))

    recommended_actions = ordered_unique(
        list(runtime.get("recommended_actions") or [])[:2]
        + list(auth_lease.get("recommended_actions") or [])[:2]
        + list(coverage_seed.get("recommended_actions") or [])[:2]
        + list(gap_closer.get("recommended_actions") or [])[:2]
        + list(lane_thaw.get("recommended_actions") or [])[:2]
        + list(data_plane_recovery.get("recommended_actions") or [])[:2]
        + list(notification_ladder.get("recommended_actions") or [])[:2]
        + list(promotion_autopilot.get("recommended_actions") or [])[:2]
        + list(live_canary.get("recommended_actions") or [])[:2]
        + list(incident_timeline.get("recommended_actions") or [])[:2]
        + list(runtime_throttle.get("recommended_actions") or [])[:2]
        + list(chrome_headless.get("recommended_actions") or [])[:2]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "autonomy_score": autonomy_score,
        "capabilities": {
            "live_research_split": True,
            "lane_state_recovery": True,
            "lane_thaw_controller": True,
            "coverage_autopilot": True,
            "auth_lease_workflow": True,
            "data_plane_recovery": True,
            "notification_escalation": True,
            "promotion_autopilot": True,
            "live_canary_control": True,
            "incident_timeline": True,
            "runtime_throttle_control": True,
            "chrome_headless_guard": bool(chrome_headless_summary.get("monitored", False)),
        },
        "autonomous_repair_path_count": autonomous_repair_paths,
        "component_statuses": component_statuses,
        "live_research_split": live_research_split,
        "lane_recovery_playbooks": lane_recovery,
        "lane_recovery_playbook_penalty": {
            "critical_count": critical_lane_playbook_count,
            "warning_count": warning_lane_playbook_count,
            "penalty": round(float(lane_playbook_penalty), 3),
            "policy": "critical_full_warning_capped",
        },
        "lane_thaw_controller": thaw_summary,
        "coverage_autopilot": coverage_autopilot,
        "auth_lease_workflow": auth_workflow,
        "data_plane_recovery": data_plane_summary,
        "notification_escalation": notification_summary,
        "promotion_autopilot": promotion_summary,
        "live_canary_control": live_canary_summary,
        "incident_timeline": incident_summary,
        "incident_review": incident_review_summary,
        "incident_closure_loop": incident_closure_loop,
        "runtime_throttle_control": runtime_throttle_summary,
        "chrome_headless_guard": chrome_headless_summary,
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish the cross-surface autonomy control plane for live, paper, and research operations.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve())
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "autonomy_control_plane "
            f"overall_status={payload.get('overall_status', '')} "
            f"autonomy_score={float(payload.get('autonomy_score', 0.0) or 0.0):.2f}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
