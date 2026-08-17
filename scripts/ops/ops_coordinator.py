#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_python import resolve_runtime_python


PY = resolve_runtime_python(PROJECT_ROOT)
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "ops_coordinator_latest.json"
DEFAULT_LOCK_PATH = PROJECT_ROOT / "governance" / "locks" / "ops_coordinator.lock"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _parse_json_output(text: str) -> dict[str, Any]:
    for line in reversed([raw.strip() for raw in str(text or "").splitlines() if raw.strip()]):
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _run_json_command(
    cmd: list[str],
    *,
    cwd: Path,
    payload_path: Path | None = None,
    timeout_seconds: float = 180.0,
) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    stdout = ""
    stderr = ""
    rc = 1
    timed_out = False
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=max(float(timeout_seconds), 1.0),
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        rc = int(proc.returncode)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        rc = 124
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        stderr = "\n".join([stderr, f"timed_out_after_seconds={float(timeout_seconds):.1f}"]).strip()
    payload = _parse_json_output(stdout)
    if not payload and payload_path is not None:
        payload = _load_json(payload_path)
    duration_ms = round((datetime.now(timezone.utc) - started).total_seconds() * 1000.0, 3)
    return {
        "cmd": list(cmd),
        "rc": rc,
        "duration_ms": duration_ms,
        "timed_out": timed_out,
        "payload": payload,
        "stdout_tail": "\n".join(stdout.splitlines()[-12:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-12:]),
    }


def _step_status(result: dict[str, Any]) -> str:
    payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
    if int(result.get("rc", 1)) != 0 and not payload:
        return "error"
    if bool(result.get("timed_out", False)) and bool(payload.get("ok", False)):
        return "degraded"
    if bool(payload.get("busy", False)):
        return "busy"
    if bool(payload.get("skipped", False)):
        return "skipped"
    if payload.get("ok") is False:
        overall_status = str(payload.get("overall_status") or "").strip().lower()
        return overall_status or "error"
    if int(result.get("rc", 1)) != 0:
        overall_status = str(payload.get("overall_status") or "").strip().lower()
        return overall_status or "error"
    return "ok"


def _step_record(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": _step_status(result),
        "rc": int(result.get("rc", 1)),
        "duration_ms": float(result.get("duration_ms", 0.0) or 0.0),
        "cmd": list(result.get("cmd") or []),
        "stdout_tail": str(result.get("stdout_tail") or ""),
        "stderr_tail": str(result.get("stderr_tail") or ""),
    }


def build_ops_coordinator_payload(
    project_root: Path,
    *,
    day: str,
    max_rows: int,
    strategy_max_age_minutes: float,
    sandbox_max_age_minutes: float,
    watchdog_refresh_max_age_seconds: int,
    resource_profile: str,
) -> dict[str, Any]:
    resource_guard = _run_json_command(
        [str(PY), str(project_root / "scripts" / "resource_guard.py"), "--profile", str(resource_profile or "refresh"), "--json"],
        cwd=project_root,
    )
    watchdog = _run_json_command(
        [
            str(PY),
            str(project_root / "scripts" / "ops" / "process_watchdog.py"),
            "--refresh-max-age-seconds",
            str(max(int(watchdog_refresh_max_age_seconds), 1)),
            "--json",
        ],
        cwd=project_root,
        payload_path=project_root / "governance" / "health" / "process_watchdog_latest.json",
    )
    live_readiness = _run_json_command(
        [
            str(PY),
            str(project_root / "scripts" / "live_readiness_smoke.py"),
            "--json",
        ],
        cwd=project_root,
        payload_path=project_root / "governance" / "health" / "live_readiness_smoke_latest.json",
    )
    runtime_separation = _run_json_command(
        [
            str(PY),
            str(project_root / "scripts" / "ops" / "live_runtime_separation_control.py"),
            "--json",
        ],
        cwd=project_root,
        payload_path=project_root / "governance" / "health" / "live_runtime_separation_control_latest.json",
    )
    incident_timeline = _run_json_command(
        [
            str(PY),
            str(project_root / "scripts" / "ops" / "incident_timeline.py"),
            "--json",
        ],
        cwd=project_root,
        payload_path=project_root / "governance" / "health" / "incident_timeline_latest.json",
    )
    derived_state = _run_json_command(
        [str(PY), str(project_root / "scripts" / "derived_state_snapshot.py"), "--json"],
        cwd=project_root,
        payload_path=project_root / "governance" / "health" / "derived_state_latest.json",
    )
    strategy_research = _run_json_command(
        [
            str(PY),
            str(project_root / "scripts" / "strategy_research_lane.py"),
            "--day",
            str(day),
            "--max-rows",
            str(max(int(max_rows), 1)),
            "--skip-sandbox",
            "--max-age-minutes",
            str(max(float(strategy_max_age_minutes), 0.0)),
            "--sandbox-max-age-minutes",
            str(max(float(sandbox_max_age_minutes), 0.0)),
            "--json",
        ],
        cwd=project_root,
        payload_path=project_root / "governance" / "health" / "strategy_research_latest.json",
    )
    training_registry_audit = _run_json_command(
        [
            str(PY),
            str(project_root / "scripts" / "training_registry_audit.py"),
            "--json",
        ],
        cwd=project_root,
        payload_path=project_root / "governance" / "health" / "training_registry_audit_latest.json",
    )
    training_label_audit = _run_json_command(
        [
            str(PY),
            str(project_root / "scripts" / "training_label_audit.py"),
            "--json",
        ],
        cwd=project_root,
        payload_path=project_root / "governance" / "health" / "training_label_audit_latest.json",
    )
    provider_mesh = _run_json_command(
        [
            str(PY),
            str(project_root / "scripts" / "provider_mesh_control.py"),
            "--json",
        ],
        cwd=project_root,
        payload_path=project_root / "governance" / "health" / "provider_mesh_latest.json",
    )
    control_plane = _run_json_command(
        [
            str(PY),
            str(project_root / "scripts" / "platform_control_plane_report.py"),
            "--max-rows",
            str(max(int(max_rows), 1)),
            "--json",
        ],
        cwd=project_root,
        payload_path=project_root / "governance" / "health" / "platform_control_plane_latest.json",
    )
    service_control_plane = _run_json_command(
        [
            str(PY),
            str(project_root / "scripts" / "service_control_plane.py"),
            "--json",
        ],
        cwd=project_root,
        payload_path=project_root / "governance" / "health" / "service_control_plane_latest.json",
    )
    promotion_autopilot = _run_json_command(
        [
            str(PY),
            str(project_root / "scripts" / "ops" / "promotion_autopilot_packet.py"),
            "--json",
        ],
        cwd=project_root,
        payload_path=project_root / "governance" / "champion_challenger" / "promotion_autopilot_packet_latest.json",
    )
    notification_ladder = _run_json_command(
        [
            str(PY),
            str(project_root / "scripts" / "ops" / "notification_escalation_ladder.py"),
            "--json",
        ],
        cwd=project_root,
        payload_path=project_root / "governance" / "health" / "notification_escalation_ladder_latest.json",
    )
    incident_review = _run_json_command(
        [
            str(PY),
            str(project_root / "scripts" / "ops" / "incident_review_packet.py"),
            "--json",
        ],
        cwd=project_root,
        payload_path=project_root / "governance" / "health" / "incident_review_packet_latest.json",
    )
    lane_thaw = _run_json_command(
        [
            str(PY),
            str(project_root / "scripts" / "ops" / "lane_thaw_controller.py"),
            "--json",
        ],
        cwd=project_root,
        payload_path=project_root / "governance" / "health" / "lane_thaw_controller_latest.json",
    )
    data_plane_recovery = _run_json_command(
        [
            str(PY),
            str(project_root / "scripts" / "ops" / "data_plane_recovery_controller.py"),
            "--json",
        ],
        cwd=project_root,
        payload_path=project_root / "governance" / "health" / "data_plane_recovery_controller_latest.json",
    )
    sql_access_runtime_audit = _run_json_command(
        [
            str(PY),
            str(project_root / "scripts" / "ops" / "sql_access_runtime_audit.py"),
            "--json",
        ],
        cwd=project_root,
        payload_path=project_root / "governance" / "health" / "sql_access_runtime_audit_latest.json",
    )
    runtime_dependency_profiles = _run_json_command(
        [
            str(PY),
            str(project_root / "scripts" / "ops" / "runtime_dependency_profiles.py"),
            "--json",
        ],
        cwd=project_root,
        payload_path=project_root / "governance" / "health" / "runtime_dependency_profiles_latest.json",
    )
    sql_analytics_mirror = _run_json_command(
        [
            str(PY),
            str(project_root / "scripts" / "ops" / "sql_analytics_mirror.py"),
            "--lookback-days",
            "1",
            "--json",
        ],
        cwd=project_root,
        payload_path=project_root / "governance" / "health" / "sql_analytics_mirror_latest.json",
        timeout_seconds=45.0,
    )
    macro_intelligence = _run_json_command(
        [
            str(PY),
            str(project_root / "scripts" / "ops" / "macro_event_intelligence.py"),
            "--json",
        ],
        cwd=project_root,
        payload_path=project_root / "governance" / "health" / "macro_event_intelligence_latest.json",
    )
    autonomy_control = _run_json_command(
        [
            str(PY),
            str(project_root / "scripts" / "ops" / "autonomy_control_plane.py"),
            "--json",
        ],
        cwd=project_root,
        payload_path=project_root / "governance" / "health" / "autonomy_control_plane_latest.json",
    )

    steps = {
        "resource_guard": _step_record(resource_guard),
        "process_watchdog": _step_record(watchdog),
        "live_readiness_smoke": _step_record(live_readiness),
        "live_runtime_separation_control": _step_record(runtime_separation),
        "incident_timeline": _step_record(incident_timeline),
        "derived_state": _step_record(derived_state),
        "strategy_research_fast": _step_record(strategy_research),
        "training_registry_audit": _step_record(training_registry_audit),
        "training_label_audit": _step_record(training_label_audit),
        "provider_mesh": _step_record(provider_mesh),
        "platform_control_plane": _step_record(control_plane),
        "service_control_plane": _step_record(service_control_plane),
        "promotion_autopilot_packet": _step_record(promotion_autopilot),
        "notification_escalation_ladder": _step_record(notification_ladder),
        "incident_review_packet": _step_record(incident_review),
        "lane_thaw_controller": _step_record(lane_thaw),
        "data_plane_recovery_controller": _step_record(data_plane_recovery),
        "sql_access_runtime_audit": _step_record(sql_access_runtime_audit),
        "runtime_dependency_profiles": _step_record(runtime_dependency_profiles),
        "sql_analytics_mirror": _step_record(sql_analytics_mirror),
        "macro_event_intelligence": _step_record(macro_intelligence),
        "autonomy_control_plane": _step_record(autonomy_control),
    }
    resource_payload = resource_guard.get("payload") if isinstance(resource_guard.get("payload"), dict) else {}
    watchdog_payload = watchdog.get("payload") if isinstance(watchdog.get("payload"), dict) else {}
    live_readiness_payload = live_readiness.get("payload") if isinstance(live_readiness.get("payload"), dict) else {}
    runtime_separation_payload = runtime_separation.get("payload") if isinstance(runtime_separation.get("payload"), dict) else {}
    incident_timeline_payload = incident_timeline.get("payload") if isinstance(incident_timeline.get("payload"), dict) else {}
    derived_payload = derived_state.get("payload") if isinstance(derived_state.get("payload"), dict) else {}
    strategy_payload = strategy_research.get("payload") if isinstance(strategy_research.get("payload"), dict) else {}
    registry_audit_payload = training_registry_audit.get("payload") if isinstance(training_registry_audit.get("payload"), dict) else {}
    label_audit_payload = training_label_audit.get("payload") if isinstance(training_label_audit.get("payload"), dict) else {}
    provider_mesh_payload = provider_mesh.get("payload") if isinstance(provider_mesh.get("payload"), dict) else {}
    control_payload = control_plane.get("payload") if isinstance(control_plane.get("payload"), dict) else {}
    service_control_payload = service_control_plane.get("payload") if isinstance(service_control_plane.get("payload"), dict) else {}
    promotion_autopilot_payload = promotion_autopilot.get("payload") if isinstance(promotion_autopilot.get("payload"), dict) else {}
    notification_ladder_payload = notification_ladder.get("payload") if isinstance(notification_ladder.get("payload"), dict) else {}
    incident_review_payload = incident_review.get("payload") if isinstance(incident_review.get("payload"), dict) else {}
    lane_thaw_payload = lane_thaw.get("payload") if isinstance(lane_thaw.get("payload"), dict) else {}
    data_plane_payload = data_plane_recovery.get("payload") if isinstance(data_plane_recovery.get("payload"), dict) else {}
    sql_audit_payload = sql_access_runtime_audit.get("payload") if isinstance(sql_access_runtime_audit.get("payload"), dict) else {}
    dependency_profiles_payload = runtime_dependency_profiles.get("payload") if isinstance(runtime_dependency_profiles.get("payload"), dict) else {}
    sql_analytics_payload = sql_analytics_mirror.get("payload") if isinstance(sql_analytics_mirror.get("payload"), dict) else {}
    macro_intelligence_payload = macro_intelligence.get("payload") if isinstance(macro_intelligence.get("payload"), dict) else {}
    autonomy_control_payload = autonomy_control.get("payload") if isinstance(autonomy_control.get("payload"), dict) else {}
    watchdog_network = watchdog_payload.get("network") if isinstance(watchdog_payload.get("network"), dict) else {}
    watchdog_storage_guard = watchdog_payload.get("storage_mount_guard") if isinstance(watchdog_payload.get("storage_mount_guard"), dict) else {}
    backlog = control_payload.get("storage_sql_backlog_shaping") if isinstance(control_payload.get("storage_sql_backlog_shaping"), dict) else {}
    rollout = control_payload.get("model_registry_and_rollout") if isinstance(control_payload.get("model_registry_and_rollout"), dict) else {}
    institutional_readiness = control_payload.get("institutional_readiness") if isinstance(control_payload.get("institutional_readiness"), dict) else {}
    strategy_summary = strategy_payload.get("summary") if isinstance(strategy_payload.get("summary"), dict) else {}

    ok = True
    for result in (
        resource_guard,
        watchdog,
        live_readiness,
        runtime_separation,
        incident_timeline,
        derived_state,
        strategy_research,
        training_registry_audit,
        training_label_audit,
        provider_mesh,
        control_plane,
        service_control_plane,
        promotion_autopilot,
        notification_ladder,
        incident_review,
        lane_thaw,
        data_plane_recovery,
        sql_access_runtime_audit,
        runtime_dependency_profiles,
        sql_analytics_mirror,
        macro_intelligence,
        autonomy_control,
    ):
        payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
        status = _step_status(result)
        if status == "error":
            ok = False
            break
        if status in {"blocked", "critical"}:
            ok = False
            break

    reason = "ok"
    if not ok:
        for name, record in steps.items():
            status = str(record.get("status") or "")
            if status == "error":
                reason = f"{name}_failed"
                break
            if status in {"blocked", "critical"}:
                reason = f"{name}_blocked"
                break

    overall_status = "ready"
    if not ok or str(runtime_separation_payload.get("overall_status") or "") == "blocked":
        overall_status = "blocked"
    elif any(
        status in {"degraded", "advancing", "held_out", "blocked"}
        for status in (
            str(live_readiness_payload.get("overall_status") or ""),
            str(runtime_separation_payload.get("overall_status") or ""),
            str(provider_mesh_payload.get("overall_status") or ""),
            str(service_control_payload.get("overall_status") or ""),
            str(institutional_readiness.get("overall_status") or ""),
            str(rollout.get("promotion_status") or ""),
        )
    ):
        overall_status = "degraded"

    payload = {
        "timestamp_utc": _utc_now(),
        "project_root": str(project_root),
        "day": str(day),
        "ok": bool(ok),
        "overall_status": overall_status,
        "reason": reason,
        "steps": steps,
        "artifacts": {
            "process_watchdog": str(project_root / "governance" / "health" / "process_watchdog_latest.json"),
            "live_readiness_smoke": str(project_root / "governance" / "health" / "live_readiness_smoke_latest.json"),
            "live_runtime_separation_control": str(project_root / "governance" / "health" / "live_runtime_separation_control_latest.json"),
            "incident_timeline": str(project_root / "governance" / "health" / "incident_timeline_latest.json"),
            "derived_state": str(project_root / "governance" / "health" / "derived_state_latest.json"),
            "strategy_research": str(project_root / "governance" / "health" / "strategy_research_latest.json"),
            "training_registry_audit": str(project_root / "governance" / "health" / "training_registry_audit_latest.json"),
            "training_label_audit": str(project_root / "governance" / "health" / "training_label_audit_latest.json"),
            "provider_mesh": str(project_root / "governance" / "health" / "provider_mesh_latest.json"),
            "platform_control_plane": str(project_root / "governance" / "health" / "platform_control_plane_latest.json"),
            "service_control_plane": str(project_root / "governance" / "health" / "service_control_plane_latest.json"),
            "promotion_autopilot_packet": str(project_root / "governance" / "champion_challenger" / "promotion_autopilot_packet_latest.json"),
            "notification_escalation_ladder": str(project_root / "governance" / "health" / "notification_escalation_ladder_latest.json"),
            "incident_review_packet": str(project_root / "governance" / "health" / "incident_review_packet_latest.json"),
            "lane_thaw_controller": str(project_root / "governance" / "health" / "lane_thaw_controller_latest.json"),
            "data_plane_recovery_controller": str(project_root / "governance" / "health" / "data_plane_recovery_controller_latest.json"),
            "sql_access_runtime_audit": str(project_root / "governance" / "health" / "sql_access_runtime_audit_latest.json"),
            "runtime_dependency_profiles": str(project_root / "governance" / "health" / "runtime_dependency_profiles_latest.json"),
            "sql_analytics_mirror": str(project_root / "governance" / "health" / "sql_analytics_mirror_latest.json"),
            "macro_event_intelligence": str(project_root / "governance" / "health" / "macro_event_intelligence_latest.json"),
            "autonomy_control_plane": str(project_root / "governance" / "health" / "autonomy_control_plane_latest.json"),
        },
        "resource_guard": resource_payload,
        "process_watchdog": {
            "storage_mode": str(watchdog_payload.get("storage_mode") or watchdog_storage_guard.get("storage_mode") or ""),
            "storage_ok": bool(watchdog_payload.get("storage_ok", watchdog_storage_guard.get("external_available", False))),
            "network_ok": bool(watchdog_payload.get("network_ok", watchdog_network.get("any_ok", False))),
            "restart_storm_count": int(len(watchdog_payload.get("restart_storms", []) or [])),
        },
        "live_readiness": {
            "overall_status": str(live_readiness_payload.get("overall_status") or ""),
            "readiness_score": float(live_readiness_payload.get("readiness_score", 0.0) or 0.0),
            "broker_ready": bool(live_readiness_payload.get("broker_ready", False)),
            "session_ready": bool(live_readiness_payload.get("session_ready", False)),
            "paper_lane_fresh": bool(live_readiness_payload.get("paper_lane_fresh", False)),
            "live_lane_running": bool(live_readiness_payload.get("live_lane_running", False)),
            "watchdog_healthy": bool(((live_readiness_payload.get("process_watchdog") or {}).get("healthy", False))),
        },
        "runtime_separation": {
            "overall_status": str(runtime_separation_payload.get("overall_status") or ""),
            "contention_score": int((((runtime_separation_payload.get("shared_host_pressure") or {}).get("contention_score", 0)) or 0)),
            "live_lane_should_be_read_only": bool(((runtime_separation_payload.get("release_contract") or {}).get("live_lane_should_be_read_only", False))),
            "promotions_should_wait_for_cold_lane": bool(((runtime_separation_payload.get("release_contract") or {}).get("promotions_should_wait_for_cold_lane", False))),
            "shared_host_training_resume_allowed": bool(
                ((runtime_separation_payload.get("release_contract") or {}).get("shared_host_training_resume_allowed", False))
            ),
            "clearance_state": str(((runtime_separation_payload.get("clearance_plan") or {}).get("clearance_state") or "")),
            "coverage_gap_launch_state": str(
                (((runtime_separation_payload.get("clearance_plan") or {}).get("coverage_gap_closer") or {}).get("launch_state") or "")
            ),
        },
        "incident_timeline": {
            "overall_status": str(incident_timeline_payload.get("overall_status") or ""),
            "recent_incident_count": int(incident_timeline_payload.get("recent_incident_count", 0) or 0),
            "open_incident_count": int(incident_timeline_payload.get("open_incident_count", 0) or 0),
        },
        "derived_state": {
            "risk_level": str(derived_payload.get("risk_level") or ""),
            "gross_risk_budget": derived_payload.get("gross_risk_budget"),
            "max_total_actions_per_hour": derived_payload.get("max_total_actions_per_hour"),
            "max_total_open_orders": derived_payload.get("max_total_open_orders"),
        },
        "strategy_research": {
            "promotable": bool(strategy_payload.get("promotable", False)),
            "recommended_action": str(strategy_summary.get("recommended_action") or ""),
            "research_sandbox_ok": bool(strategy_payload.get("research_sandbox_ok", False)),
        },
        "training_registry": {
            "active_sample_starved": int(len(registry_audit_payload.get("active_sample_starved", []) or [])),
            "active_quality_failed": int(len(registry_audit_payload.get("active_quality_failed", []) or [])),
            "active_stale_diagnostics": int(len(registry_audit_payload.get("active_stale_diagnostics", []) or [])),
            "tier_counts": registry_audit_payload.get("tier_counts", {}),
        },
        "training_label_quality": {
            "top_actions": label_audit_payload.get("top_actions", []),
            "recommendation_counts": label_audit_payload.get("recommendation_counts", {}),
        },
        "provider_mesh": {
            "overall_status": str(provider_mesh_payload.get("overall_status") or ""),
            "required_failure_count": int(((provider_mesh_payload.get("summary") or {}).get("required_failure_count", 0) or 0)),
            "soft_failure_count": int(((provider_mesh_payload.get("summary") or {}).get("soft_failure_count", 0) or 0)),
        },
        "control_plane": {
            "promotion_status": str(rollout.get("promotion_status") or ""),
            "training_reason": str(rollout.get("training_reason") or ""),
            "institutional_readiness_status": str(institutional_readiness.get("overall_status") or ""),
            "institutional_readiness_score": float(institutional_readiness.get("overall_score", 0.0) or 0.0),
            "pending_lines": int(backlog.get("pending_lines", 0) or 0),
            "pending_lines_cold": int(backlog.get("pending_lines_cold", 0) or 0),
            "cold_lane_recommendation": str(backlog.get("cold_lane_recommendation") or ""),
        },
        "service_control_plane": {
            "overall_status": str(service_control_payload.get("overall_status") or ""),
            "completion_score": float(((service_control_payload.get("summary") or {}).get("completion_score", 0.0) or 0.0)),
        },
        "promotion_autopilot": {
            "overall_status": str(promotion_autopilot_payload.get("overall_status") or ""),
            "autopilot_state": str(promotion_autopilot_payload.get("autopilot_state") or ""),
            "promotion_ready": bool(promotion_autopilot_payload.get("promotion_ready", False)),
            "blocker_count": int(len(promotion_autopilot_payload.get("blockers", []) or [])),
        },
        "notification_escalation_ladder": {
            "overall_status": str(notification_ladder_payload.get("overall_status") or ""),
            "attended_runtime_ready": bool(notification_ladder_payload.get("attended_runtime_ready", False)),
            "unattended_runtime_ready": bool(notification_ladder_payload.get("unattended_runtime_ready", False)),
            "grouped_unsent_count": int((((notification_ladder_payload.get("critical_backlog") or {}).get("grouped_unsent_count", 0)) or 0)),
        },
        "incident_review_packet": {
            "overall_status": str(incident_review_payload.get("overall_status") or ""),
            "review_required": bool(incident_review_payload.get("review_required", False)),
            "open_incident_count": int(incident_review_payload.get("open_incident_count", 0) or 0),
        },
        "lane_thaw_controller": {
            "overall_status": str(lane_thaw_payload.get("overall_status") or ""),
            "paused_lane_count": int(lane_thaw_payload.get("paused_lane_count", 0) or 0),
            "candidate_count": int(lane_thaw_payload.get("candidate_count", 0) or 0),
            "chronic_offender_count": int((((lane_thaw_payload.get("cooldown_history") or {}).get("chronic_offender_count", 0)) or 0)),
            "watchlist_count": int((((lane_thaw_payload.get("cooldown_history") or {}).get("watchlist_count", 0)) or 0)),
        },
        "data_plane_recovery": {
            "overall_status": str(data_plane_payload.get("overall_status") or ""),
            "write_failure_count": int(data_plane_payload.get("write_failure_count", 0) or 0),
            "account_snapshot_failure_count": int(data_plane_payload.get("account_snapshot_failure_count", 0) or 0),
        },
        "sql_access_runtime_audit": {
            "critical_packages_ok": bool(sql_audit_payload.get("critical_packages_ok", False)),
            "profile_files_present": sql_audit_payload.get("profile_files_present", {}),
            "recommendations": sql_audit_payload.get("recommendations", []),
            "data_library_roles": sql_audit_payload.get("data_library_roles", {}),
        },
        "runtime_dependency_profiles": {
            "ok": bool(dependency_profiles_payload.get("ok", False)),
            "profile_counts": dependency_profiles_payload.get("profile_counts", {}),
            "overlap_package_count": int(dependency_profiles_payload.get("overlap_package_count", 0) or 0),
        },
        "sql_analytics_mirror": {
            "ok": bool(sql_analytics_payload.get("ok", False)),
            "summary_refresh_ok": bool(sql_analytics_payload.get("summary_refresh_ok", False)),
            "duckdb_mirror_ready": bool(((sql_analytics_payload.get("duckdb_mirror") or {}).get("mirror_ready", False))),
            "source_record_count": int((((sql_analytics_payload.get("materialized_summaries") or {}).get("source_record_count", 0)) or 0)),
        },
        "macro_event_intelligence": {
            "overall_status": str(macro_intelligence_payload.get("overall_status") or ""),
            "market_relevance": str(macro_intelligence_payload.get("market_relevance") or ""),
            "transcript_quality": str(macro_intelligence_payload.get("transcript_quality") or ""),
        },
        "autonomy_control_plane": {
            "overall_status": str(autonomy_control_payload.get("overall_status") or ""),
            "autonomy_score": float(autonomy_control_payload.get("autonomy_score", 0.0) or 0.0),
            "playbook_count": int((((autonomy_control_payload.get("lane_recovery_playbooks") or {}).get("triggered_playbook_count", 0)) or 0)),
        },
        "summary": {
            "safe_operating_envelope": bool(ok),
            "overall_status": overall_status,
            "memory_pressure_kind": str(resource_payload.get("memory_pressure_kind") or ""),
            "storage_mode": str(watchdog_payload.get("storage_mode") or watchdog_storage_guard.get("storage_mode") or ""),
            "live_readiness_status": str(live_readiness_payload.get("overall_status") or ""),
            "live_readiness_score": float(live_readiness_payload.get("readiness_score", 0.0) or 0.0),
            "runtime_separation_status": str(runtime_separation_payload.get("overall_status") or ""),
            "runtime_contention_score": int((((runtime_separation_payload.get("shared_host_pressure") or {}).get("contention_score", 0)) or 0)),
            "runtime_clearance_state": str(((runtime_separation_payload.get("clearance_plan") or {}).get("clearance_state") or "")),
            "shared_host_training_resume_allowed": bool(
                ((runtime_separation_payload.get("release_contract") or {}).get("shared_host_training_resume_allowed", False))
            ),
            "risk_level": str(derived_payload.get("risk_level") or ""),
            "promotion_status": str(rollout.get("promotion_status") or ""),
            "institutional_readiness_status": str(institutional_readiness.get("overall_status") or ""),
            "institutional_readiness_score": float(institutional_readiness.get("overall_score", 0.0) or 0.0),
            "recommended_action": str(strategy_summary.get("recommended_action") or ""),
            "active_sample_starved": int(len(registry_audit_payload.get("active_sample_starved", []) or [])),
            "active_quality_failed": int(len(registry_audit_payload.get("active_quality_failed", []) or [])),
            "active_stale_diagnostics": int(len(registry_audit_payload.get("active_stale_diagnostics", []) or [])),
            "pending_lines": int(backlog.get("pending_lines", 0) or 0),
            "pending_lines_cold": int(backlog.get("pending_lines_cold", 0) or 0),
            "provider_mesh_status": str(provider_mesh_payload.get("overall_status") or ""),
            "service_control_status": str(service_control_payload.get("overall_status") or ""),
            "incident_timeline_status": str(incident_timeline_payload.get("overall_status") or ""),
            "open_incident_count": int(incident_timeline_payload.get("open_incident_count", 0) or 0),
            "promotion_autopilot_state": str(promotion_autopilot_payload.get("autopilot_state") or ""),
            "notification_ladder_status": str(notification_ladder_payload.get("overall_status") or ""),
            "incident_review_status": str(incident_review_payload.get("overall_status") or ""),
            "lane_thaw_candidates": int(lane_thaw_payload.get("candidate_count", 0) or 0),
            "lane_thaw_chronic_offenders": int((((lane_thaw_payload.get("cooldown_history") or {}).get("chronic_offender_count", 0)) or 0)),
            "data_plane_write_failures": int(data_plane_payload.get("write_failure_count", 0) or 0),
            "sql_access_runtime_ready": bool(sql_audit_payload.get("critical_packages_ok", False)),
            "runtime_dependency_profiles_ready": bool(dependency_profiles_payload.get("ok", False)),
            "analytics_mirror_ready": bool(((sql_analytics_payload.get("duckdb_mirror") or {}).get("mirror_ready", False))),
            "macro_event_relevance": str(macro_intelligence_payload.get("market_relevance") or ""),
            "autonomy_status": str(autonomy_control_payload.get("overall_status") or ""),
            "autonomy_score": float(autonomy_control_payload.get("autonomy_score", 0.0) or 0.0),
        },
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Coordinate lightweight ops freshness and health refreshes outside the trading registry.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--day", default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    parser.add_argument("--max-rows", type=int, default=4000)
    parser.add_argument("--strategy-max-age-minutes", type=float, default=90.0)
    parser.add_argument("--sandbox-max-age-minutes", type=float, default=720.0)
    parser.add_argument("--watchdog-refresh-max-age-seconds", type=int, default=7200)
    parser.add_argument("--resource-profile", default="refresh")
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--lock-file", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_file = Path(args.out_file).expanduser()
    lock_file = Path(args.lock_file).expanduser()
    lock_file.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "timestamp_utc": _utc_now(),
        "project_root": str(project_root),
        "day": str(args.day),
        "ok": True,
        "skipped": False,
        "reason": "pending",
    }

    with lock_file.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            payload.update({"skipped": True, "reason": "already_running"})
            _write_json(out_file, payload)
            if args.json:
                print(json.dumps(payload, ensure_ascii=True))
            else:
                print("ops_coordinator skipped=1 reason=already_running")
            return 0

        payload = build_ops_coordinator_payload(
            project_root,
            day=str(args.day),
            max_rows=int(args.max_rows),
            strategy_max_age_minutes=float(args.strategy_max_age_minutes),
            sandbox_max_age_minutes=float(args.sandbox_max_age_minutes),
            watchdog_refresh_max_age_seconds=int(args.watchdog_refresh_max_age_seconds),
            resource_profile=str(args.resource_profile or "refresh"),
        )
        _write_json(out_file, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "ops_coordinator "
            f"ok={int(bool(payload.get('ok', False)))} "
            f"pending_lines={int(((payload.get('summary') or {}).get('pending_lines', 0) or 0))} "
            f"recommended_action={((payload.get('summary') or {}).get('recommended_action', '') or '')}"
        )
    return 0 if bool(payload.get("ok", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
