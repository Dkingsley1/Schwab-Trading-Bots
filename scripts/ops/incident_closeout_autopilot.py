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


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "incident_closeout_autopilot_latest.json"
RECOVERABLE_RUNTIME_CLEARANCE_STATES = {
    "awaiting_coverage_cycles",
    "awaiting_cold_lane",
    "staged_preclearance",
    "coverage_cycles_ready",
    "off_hours_cold_lane_launch_ready",
    "scheduled_off_hours_launch",
    "managed_coverage_stage_deferred",
}
GUARDED_READ_ONLY_RUNTIME_STATES = {
    "guarded_live_read_only",
}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _surface_block(
    *,
    surface: str,
    severity: str,
    summary: str,
    artifact_name: str,
    command: list[str],
) -> dict[str, Any]:
    return {
        "surface": surface,
        "severity": severity,
        "summary": summary,
        "artifact_name": artifact_name,
        "recommended_command": list(command),
    }


def _recoverable_runtime_clearance(runtime: dict[str, Any]) -> bool:
    clearance_state = str(((runtime.get("clearance_plan") or {}).get("clearance_state") or "")).strip().lower()
    runtime_status = str(runtime.get("overall_status") or "").strip().lower()
    return bool(
        (clearance_state in RECOVERABLE_RUNTIME_CLEARANCE_STATES or _guarded_read_only_runtime(runtime))
        and runtime_status in {"ready", "degraded", "needs_attention"}
    )


def _guarded_read_only_runtime(runtime: dict[str, Any]) -> bool:
    clearance_state = str(((runtime.get("clearance_plan") or {}).get("clearance_state") or "")).strip().lower()
    return clearance_state in GUARDED_READ_ONLY_RUNTIME_STATES


def _managed_coverage_stage_deferred_runtime(runtime: dict[str, Any]) -> bool:
    clearance_state = str(((runtime.get("clearance_plan") or {}).get("clearance_state") or "")).strip().lower()
    return bool(clearance_state == "managed_coverage_stage_deferred" and str(runtime.get("overall_status") or "").strip().lower() == "ready")


def _recoverable_review_gate(review: dict[str, Any]) -> bool:
    review_status = str(review.get("overall_status") or "").strip().lower()
    closure_contract = review.get("closure_contract") if isinstance(review.get("closure_contract"), dict) else {}
    closure_reason = str(closure_contract.get("closure_reason") or "").strip().lower()
    open_surfaces = review.get("open_surfaces") if isinstance(review.get("open_surfaces"), list) else []
    open_incident_count = _safe_int(review.get("open_incident_count"), 0)
    return bool(
        bool(review.get("review_required", False))
        and closure_reason in {
            "open_surfaces_present",
            "open_surfaces_present_or_no_recent_incidents",
            "watchdog_alerts_present",
        }
        and (
            review_status in {"degraded", "needs_attention"}
            or (
                review_status == "blocked"
                and not open_surfaces
                and 0 < open_incident_count <= 3
                and closure_reason == "open_surfaces_present"
            )
        )
    )


def _bounded_data_plane_recovery(data_plane: dict[str, Any]) -> bool:
    recovery_state = str(data_plane.get("recovery_state") or "").strip().lower()
    overall_status = str(data_plane.get("overall_status") or "").strip().lower()
    recovery_contract = data_plane.get("recovery_contract") if isinstance(data_plane.get("recovery_contract"), dict) else {}
    backlog_contract = data_plane.get("backlog_recovery_contract") if isinstance(data_plane.get("backlog_recovery_contract"), dict) else {}
    writer_handoff = data_plane.get("writer_handoff_contract") if isinstance(data_plane.get("writer_handoff_contract"), dict) else {}
    writer_service_active = bool(
        writer_handoff.get("writer_service_active", False)
        or recovery_contract.get("writer_service_active", False)
    )
    follow_through_status = str(backlog_contract.get("follow_through_status") or "").strip().lower()
    external_backlog_status = str(data_plane.get("external_backlog_status") or "").strip().lower()
    drain_progress_lines = abs(_safe_int(backlog_contract.get("drain_progress_lines"), 0))
    progress_observed = bool(backlog_contract.get("progress_observed", False))
    return bool(
        recovery_state == "recovering_under_guard"
        and overall_status in {"degraded", "needs_work"}
        and writer_service_active
        and (
            progress_observed
            or drain_progress_lines > 0
            or follow_through_status in {"handoff_requested", "drain_active", "writer_handoff_active"}
            or external_backlog_status in {"drain_active", "handoff_requested"}
        )
    )


def _bounded_auth_lease(auth: dict[str, Any]) -> bool:
    lease_state = str(auth.get("lease_state") or "").strip().lower()
    lease_budget = auth.get("lease_budget") if isinstance(auth.get("lease_budget"), dict) else {}
    broker_state = auth.get("broker_state") if isinstance(auth.get("broker_state"), dict) else {}
    expires_in_seconds = _safe_int(lease_budget.get("expires_in_seconds"), 0)
    critical_lease_seconds = _safe_int(lease_budget.get("critical_lease_seconds"), 0)
    return bool(
        lease_state == "warning"
        and bool(broker_state.get("broker_ready", False))
        and bool(broker_state.get("auth_ok", False))
        and bool(broker_state.get("configured_for_refresh", False))
        and expires_in_seconds > max(critical_lease_seconds, 0)
    )


def _isolated_read_only_watchdog_debt(watchdog: dict[str, Any]) -> bool:
    isolation = watchdog.get("restart_storm_isolation") if isinstance(watchdog.get("restart_storm_isolation"), dict) else {}
    if not isolation:
        intelligence = watchdog.get("watchdog_intelligence") if isinstance(watchdog.get("watchdog_intelligence"), dict) else {}
        isolation = (
            intelligence.get("restart_storm_isolation")
            if isinstance(intelligence.get("restart_storm_isolation"), dict)
            else {}
        )
    isolated_count = _safe_int(isolation.get("isolated_count"), 0)
    execution_blocking_count = _safe_int(isolation.get("execution_blocking_count"), 0)
    if isolated_count > 0 and execution_blocking_count <= 0 and bool(isolation.get("all_active_storms_isolated", False)):
        return True

    storms = watchdog.get("restart_storms") if isinstance(watchdog.get("restart_storms"), list) else []
    active_storms = [row for row in storms if isinstance(row, dict) and not bool(row.get("resolved", False))]
    if not active_storms:
        return False
    return all(
        bool(row.get("quarantinable", False))
        and not bool(row.get("blocks_execution_clear", True))
        and str(row.get("impact") or "").strip().lower() == "read_only_collection"
        for row in active_storms
    )


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    timeline = load_json(health_root / "incident_timeline_latest.json")
    review = load_json(health_root / "incident_review_packet_latest.json")
    runtime = load_json(health_root / "live_runtime_separation_control_latest.json")
    auth = load_json(health_root / "auth_lease_manager_latest.json")
    alerts = load_json(health_root / "remote_alert_control_latest.json")
    thaw = load_json(health_root / "lane_thaw_controller_latest.json")
    data_plane = load_json(health_root / "data_plane_recovery_controller_latest.json")
    watchdog = load_json(health_root / "process_watchdog_latest.json")

    clearance_state = str(((runtime.get("clearance_plan") or {}).get("clearance_state") or "")).strip().lower()
    lease_state = str(auth.get("lease_state") or "").strip().lower()
    open_incident_count = _safe_int(timeline.get("open_incident_count"), 0)
    review_required = bool(review.get("review_required", False))
    watch_surfaces = timeline.get("watch_surfaces") if isinstance(timeline.get("watch_surfaces"), list) else []
    paused_lane_count = _safe_int(thaw.get("paused_lane_count"), 0)
    blocked_lane_count = _safe_int(thaw.get("blocked_count"), 0)
    write_failure_count = _safe_int(data_plane.get("write_failure_count"), 0)
    account_snapshot_failure_count = _safe_int(data_plane.get("account_snapshot_failure_count"), 0)
    critical_backlog = alerts.get("critical_backlog") if isinstance(alerts.get("critical_backlog"), dict) else {}
    unacked_count = _safe_int(critical_backlog.get("unacked_count"), 0)
    unsent_count = _safe_int(critical_backlog.get("unsent_count"), 0)
    restart_storm_count = len(watchdog.get("restart_storms")) if isinstance(watchdog.get("restart_storms"), list) else 0
    watchdog_alert_count = len(watchdog.get("alerts")) if isinstance(watchdog.get("alerts"), list) else 0
    bounded_data_plane_recovery = _bounded_data_plane_recovery(data_plane)
    bounded_auth_lease = _bounded_auth_lease(auth)
    derived_storage_watchdog = any(
        isinstance(row, dict)
        and str(row.get("surface") or "").strip().lower() == "process_watchdog"
        and str(row.get("watch_reason") or "").strip().lower() == "derived_storage_backpressure"
        for row in watch_surfaces
    )
    isolated_read_only_watchdog = _isolated_read_only_watchdog_debt(watchdog)

    blockers: list[dict[str, Any]] = []
    guarded_read_only_runtime = _guarded_read_only_runtime(runtime)
    managed_coverage_stage_deferred_runtime = _managed_coverage_stage_deferred_runtime(runtime)
    recoverable_runtime_clearance = _recoverable_runtime_clearance(runtime)
    recoverable_review_gate = _recoverable_review_gate(review)

    if clearance_state not in {"", "ready", "cleared", "released"}:
        blockers.append(
            _surface_block(
                surface="runtime_clearance",
                severity=("warning" if recoverable_runtime_clearance else "critical"),
                summary=(
                    "runtime is intentionally parked in guarded live read-only for paper soak"
                    if guarded_read_only_runtime
                    else "runtime coverage repair is deferred while guarded paper soak remains green"
                    if managed_coverage_stage_deferred_runtime
                    else f"runtime clearance remains {clearance_state or 'unknown'}"
                ),
                artifact_name="live_runtime_separation_control_latest.json",
                command=["./scripts/ops/opsctl.sh", "live-runtime-separation", "--json"],
            )
        )
    if lease_state in {"warning", "critical", "expired", "stale"}:
        blockers.append(
            _surface_block(
                surface="auth_lease",
                severity=("critical" if lease_state in {"critical", "expired"} else "warning"),
                summary=f"broker auth lease remains {lease_state}",
                artifact_name="auth_lease_manager_latest.json",
                command=["./scripts/ops/opsctl.sh", "auth-lease", "--json"],
            )
        )
    if review_required:
        blockers.append(
            _surface_block(
                surface="incident_review",
                severity=("warning" if recoverable_review_gate else "critical"),
                summary="incident review packet still requires remediation or approval",
                artifact_name="incident_review_packet_latest.json",
                command=["./scripts/ops/opsctl.sh", "incident-report", "--json"],
            )
        )
    if open_incident_count > 0:
        blockers.append(
            _surface_block(
                surface="incident_timeline",
                severity="warning",
                summary=f"{open_incident_count} open incidents remain in the current window",
                artifact_name="incident_timeline_latest.json",
                command=["./scripts/ops/opsctl.sh", "incident-timeline", "--json"],
            )
        )
    if paused_lane_count > 0 or blocked_lane_count > 0:
        blockers.append(
            _surface_block(
                surface="lane_thaw",
                severity="warning",
                summary=f"paused_lane_count={paused_lane_count} blocked_count={blocked_lane_count}",
                artifact_name="lane_thaw_controller_latest.json",
                command=["./scripts/ops/opsctl.sh", "sleeve-isolation", "--json"],
            )
        )
    if write_failure_count > 0 or account_snapshot_failure_count > 0:
        blockers.append(
            _surface_block(
                surface="data_plane_recovery",
                severity=("warning" if bounded_data_plane_recovery else "critical"),
                summary=(
                    f"bounded recovery remains active write_failures={write_failure_count} account_snapshot_failures={account_snapshot_failure_count}"
                    if bounded_data_plane_recovery
                    else f"write_failures={write_failure_count} account_snapshot_failures={account_snapshot_failure_count}"
                ),
                artifact_name="data_plane_recovery_controller_latest.json",
                command=["./scripts/ops/opsctl.sh", "ops-coordinator", "--json"],
            )
        )
    if unacked_count > 0 or unsent_count > 0:
        blockers.append(
            _surface_block(
                surface="remote_alert_backlog",
                severity="warning",
                summary=f"critical alert backlog remains unacked={unacked_count} unsent={unsent_count}",
                artifact_name="remote_alert_control_latest.json",
                command=["./scripts/ops/opsctl.sh", "remote-alert-control", "--json"],
            )
        )
    if restart_storm_count > 0 or watchdog_alert_count > 0:
        blockers.append(
            _surface_block(
                surface="process_watchdog",
                severity=(
                    "warning"
                    if derived_storage_watchdog or isolated_read_only_watchdog
                    else "critical"
                    if restart_storm_count > 0
                    else "warning"
                ),
                summary=(
                    "paper-lane watchdog pressure is being absorbed inside bounded storage recovery"
                    if derived_storage_watchdog
                    else "read-only collector restart debt is isolated and does not block execution clearance"
                    if isolated_read_only_watchdog
                    else f"restart_storms={restart_storm_count} alerts={watchdog_alert_count}"
                ),
                artifact_name="process_watchdog_latest.json",
                command=["./scripts/ops/opsctl.sh", "status"],
            )
        )

    timeline_closeout = timeline.get("auto_close_contract") if isinstance(timeline.get("auto_close_contract"), dict) else {}
    review_closeout = review.get("closure_contract") if isinstance(review.get("closure_contract"), dict) else {}
    explicit_closeout_ready = bool(
        timeline_closeout.get("closure_ready", True)
        and review_closeout.get("closure_ready", True)
    )
    warning_only_blockers = bool(blockers) and all(str(row.get("severity") or "") == "warning" for row in blockers)
    bounded_incident_backlog = bool(open_incident_count > 0 and open_incident_count <= 3 and warning_only_blockers)
    bounded_warning_closeout_ready = bool(
        explicit_closeout_ready
        and review_required is False
        and open_incident_count <= 0
        and warning_only_blockers
        and all(
            (
                (row.get("surface") == "runtime_clearance" and recoverable_runtime_clearance)
                or (row.get("surface") == "auth_lease" and bounded_auth_lease)
                or (row.get("surface") == "data_plane_recovery" and bounded_data_plane_recovery)
                or (row.get("surface") == "remote_alert_backlog" and unacked_count <= 0 and unsent_count <= 0)
                or (row.get("surface") == "incident_review" and recoverable_review_gate)
                or (row.get("surface") == "process_watchdog" and (derived_storage_watchdog or isolated_read_only_watchdog))
            )
            for row in blockers
        )
    )
    closeout_ready = bool((not blockers and explicit_closeout_ready) or bounded_warning_closeout_ready)
    bounded_closeout_path_ready = bool(
        not closeout_ready
        and warning_only_blockers
        and (
            recoverable_runtime_clearance
            or recoverable_review_gate
            or bounded_data_plane_recovery
            or bounded_incident_backlog
        )
    )
    required_artifacts = ordered_unique(str(row.get("artifact_name") or "") for row in blockers)
    closeout_commands = [list(row.get("recommended_command") or []) for row in blockers if list(row.get("recommended_command") or [])]

    overall_status = "ready" if closeout_ready else ("blocked" if any(str(row.get("severity") or "") == "critical" for row in blockers) else "degraded")
    closeout_score = 100.0
    critical_blocker_count = sum(1 for row in blockers if str(row.get("severity") or "") == "critical")
    warning_blocker_count = sum(1 for row in blockers if str(row.get("severity") or "") == "warning")
    warning_penalty = 2.0 if bounded_warning_closeout_ready else 5.0
    closeout_score -= 12.0 * critical_blocker_count
    closeout_score -= warning_penalty * warning_blocker_count
    closeout_score -= min(float(open_incident_count) * 3.0, 12.0)
    if bounded_closeout_path_ready:
        closeout_score += 8.0
    closeout_score = max(0.0, min(round(closeout_score, 2), 100.0))
    recommended_actions = ordered_unique(
        [str(row.get("summary") or "") for row in blockers]
        + [
            "refresh the incident report after the blocking surfaces clear so the packet hash and closeout contract stay in sync" if blockers else "",
            "treat the current incident set as bounded closeout-in-progress; keep refreshing the review packet until the warning-only surfaces clear" if bounded_closeout_path_ready else "",
            "archive the current review packet and mark the incident closeout contract satisfied" if closeout_ready else "",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": closeout_ready,
        "overall_status": overall_status,
        "review_required": review_required,
        "open_incident_count": int(open_incident_count),
        "closeout_ready": closeout_ready,
        "explicit_closeout_ready": explicit_closeout_ready,
        "bounded_incident_backlog": bounded_incident_backlog,
        "bounded_closeout_path_ready": bounded_closeout_path_ready,
        "closeout_score": closeout_score,
        "bounded_data_plane_recovery": bounded_data_plane_recovery,
        "bounded_auth_lease": bounded_auth_lease,
        "guarded_read_only_runtime": guarded_read_only_runtime,
        "managed_coverage_stage_deferred_runtime": managed_coverage_stage_deferred_runtime,
        "isolated_read_only_watchdog": isolated_read_only_watchdog,
        "recoverable_runtime_clearance": recoverable_runtime_clearance,
        "recoverable_review_gate": recoverable_review_gate,
        "bounded_warning_closeout_ready": bounded_warning_closeout_ready,
        "blocking_surfaces": blockers,
        "required_artifacts": required_artifacts,
        "closeout_commands": closeout_commands,
        "next_action": (
            "archive the review packet and proceed with normal operations"
            if closeout_ready
            else "clear the blocking surfaces and refresh the incident report before considering the incident closed"
        ),
        "recommended_actions": recommended_actions,
        "source_artifacts": {
            "incident_timeline": str(health_root / "incident_timeline_latest.json"),
            "incident_review_packet": str(health_root / "incident_review_packet_latest.json"),
            "live_runtime_separation_control": str(health_root / "live_runtime_separation_control_latest.json"),
            "auth_lease_manager": str(health_root / "auth_lease_manager_latest.json"),
            "remote_alert_control": str(health_root / "remote_alert_control_latest.json"),
            "lane_thaw_controller": str(health_root / "lane_thaw_controller_latest.json"),
            "data_plane_recovery_controller": str(health_root / "data_plane_recovery_controller_latest.json"),
            "process_watchdog": str(health_root / "process_watchdog_latest.json"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish a closeout contract for incident remediation and archival readiness.")
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
            "incident_closeout_autopilot "
            f"overall_status={payload.get('overall_status', '')} "
            f"closeout_ready={int(bool(payload.get('closeout_ready', False)))} "
            f"blockers={len(payload.get('blocking_surfaces') or [])}"
        )
    return 0 if bool(payload.get("closeout_ready", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
