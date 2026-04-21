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
    if normalized in {"degraded", "warning", "needs_coverage"}:
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
    notification_ladder = load_json(health_root / "notification_escalation_ladder_latest.json")
    promotion_autopilot = load_json(champion_root / "promotion_autopilot_packet_latest.json")
    lane_thaw = load_json(health_root / "lane_thaw_controller_latest.json")
    data_plane_recovery = load_json(health_root / "data_plane_recovery_controller_latest.json")

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
    promotion_summary = {
        "overall_status": str(promotion_autopilot.get("overall_status") or ""),
        "autopilot_state": str(promotion_autopilot.get("autopilot_state") or ""),
        "promotion_ready": bool(promotion_autopilot.get("promotion_ready", False)),
        "blocker_count": len(promotion_autopilot.get("blockers") or []),
        "approval_state": str(((promotion_autopilot.get("approval_record") or {}).get("approval_state") or "")),
        "repairable_gate_count": _safe_int(((promotion_autopilot.get("readiness_repair_contract") or {}).get("repairable_gate_count", 0))),
    }
    incident_summary = {
        "overall_status": str(incident_timeline.get("overall_status") or ""),
        "recent_incident_count": _safe_int(incident_timeline.get("recent_incident_count"), 0),
        "open_incident_count": _safe_int(incident_timeline.get("open_incident_count"), 0),
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

    component_statuses = {
        "live_research_split": str(live_research_split.get("overall_status") or "missing"),
        "lane_recovery_playbooks": str(lane_recovery.get("overall_status") or "missing"),
        "lane_thaw_controller": str(thaw_summary.get("overall_status") or "missing"),
        "coverage_autopilot": str(coverage_autopilot.get("overall_status") or "missing"),
        "auth_lease_workflow": str(auth_workflow.get("overall_status") or "missing"),
        "data_plane_recovery": str(data_plane_summary.get("overall_status") or "missing"),
        "notification_escalation": str(notification_summary.get("overall_status") or "missing"),
        "promotion_autopilot": str(promotion_summary.get("overall_status") or "missing"),
        "incident_timeline": str(incident_summary.get("overall_status") or "missing"),
        "incident_review": str(incident_review_summary.get("overall_status") or "missing"),
    }

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
    if bool(notification_summary.get("attended_runtime_ready", False)):
        autonomous_repair_paths += 1
    if not bool(incident_review_summary.get("review_required", True)):
        autonomous_repair_paths += 1

    severity_penalty = (
        2.0 * _safe_int(live_research_split.get("contention_score"), 0)
        + 0.75 * _safe_int(lane_recovery.get("triggered_playbook_count"), 0)
        + 0.75 * _safe_int(thaw_summary.get("blocked_count"), 0)
        + 1.0 * _safe_int(coverage_autopilot.get("coverage_shortfall_bots"), 0)
        + 0.25 * _safe_int(data_plane_summary.get("write_failure_count"), 0)
        + 1.5 * _safe_int(incident_summary.get("open_incident_count"), 0)
        + 0.75 * _safe_int(promotion_summary.get("blocker_count"), 0)
        + 0.5 * _safe_int(notification_summary.get("grouped_unsent_count"), 0)
    )
    autonomy_score = max(0.0, round(baseline_score - severity_penalty + (2.25 * autonomous_repair_paths), 2))

    recommended_actions = ordered_unique(
        list(runtime.get("recommended_actions") or [])[:2]
        + list(auth_lease.get("recommended_actions") or [])[:2]
        + list(coverage_seed.get("recommended_actions") or [])[:2]
        + list(gap_closer.get("recommended_actions") or [])[:2]
        + list(lane_thaw.get("recommended_actions") or [])[:2]
        + list(data_plane_recovery.get("recommended_actions") or [])[:2]
        + list(notification_ladder.get("recommended_actions") or [])[:2]
        + list(promotion_autopilot.get("recommended_actions") or [])[:2]
        + list(incident_timeline.get("recommended_actions") or [])[:2]
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
            "incident_timeline": True,
        },
        "autonomous_repair_path_count": autonomous_repair_paths,
        "component_statuses": component_statuses,
        "live_research_split": live_research_split,
        "lane_recovery_playbooks": lane_recovery,
        "lane_thaw_controller": thaw_summary,
        "coverage_autopilot": coverage_autopilot,
        "auth_lease_workflow": auth_workflow,
        "data_plane_recovery": data_plane_summary,
        "notification_escalation": notification_summary,
        "promotion_autopilot": promotion_summary,
        "incident_timeline": incident_summary,
        "incident_review": incident_review_summary,
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
