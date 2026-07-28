#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.runtime_python import resolve_runtime_python
    from scripts.ops.long_runtime_common import iso_now, ordered_unique, write_payload
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from core.runtime_python import resolve_runtime_python
    from .long_runtime_common import iso_now, ordered_unique, write_payload


PY = resolve_runtime_python(PROJECT_ROOT)
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "runtime_artifact_refresh_latest.json"
PAPER_SOAK_MANAGED_STEPS = {
    "training_lineage_manifest",
    "training_quality_control",
    "architecture_upgrade_scoreboard",
    "system_architecture_contract_graph",
    "system_architecture_autopilot",
    "portfolio_capacity_curve_report",
    "canary_rollout_guard",
    "promotion_autopilot_packet",
    "source_verification",
    "paper_execution_truth",
    "retrain_schema_compatibility",
    "promotion_packet_builder",
    "promotion_quality_gate",
    "training_report",
    "runtime_snapshot_cache_control",
    "covered_call_roll_watch",
    "roster_resilience_planner",
    "chaos_drill_coordinator",
    "incident_timeline",
    "incident_closeout_autopilot",
    "live_canary_control",
    "live_money_readiness_contract",
    "regime_control_plane",
    "market_cycle_extraction_engine",
    "coordination_state_control",
    "multiple_testing_guard",
    "decay_monitor",
    "rolling_restart_controller",
    "operator_cockpit",
    "service_control_plane",
}
PAPER_SOAK_MANAGED_STATUSES = {
    "blocked",
    "critical",
    "degraded",
    "needs_attention",
    "needs_coverage",
    "needs_cycles",
    "needs_review",
    "needs_work",
    "thin",
    "warn",
}
RAW_LIVE_SOAK_MAX_CORE_LINES = 10000
RAW_LIVE_SOAK_MAX_TOTAL_LINES = 15000
RAW_LIVE_SOAK_MAX_AGE_SECONDS = 900.0
STATEFUL_SQL_SOFT_QUOTA_MAX_HARD_RATIO = 0.92


RefreshRunner = Callable[[dict[str, Any], Path], dict[str, Any]]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


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


def _string_set(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {str(item or "").strip() for item in value if str(item or "").strip()}


def _parse_json_output(text: str) -> dict[str, Any]:
    for raw in reversed([line.strip() for line in str(text or "").splitlines() if line.strip()]):
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _artifact_present(path: Path) -> bool:
    return path.exists() and bool(_load_json(path))


def _step_specs(project_root: Path) -> list[dict[str, Any]]:
    ops_root = project_root / "scripts" / "ops"
    health_root = project_root / "governance" / "health"
    return [
        {
            "name": "runtime_access_mode",
            "payload_path": health_root / "runtime_access_mode_latest.json",
            "cmd": [str(PY), str(ops_root / "runtime_access_mode.py"), "status", "--json"],
        },
        {
            "name": "apple_silicon_profile",
            "payload_path": health_root / "apple_silicon_profile_latest.json",
            "cmd": [str(PY), str(ops_root / "apple_silicon_profile.py"), "status", "--json"],
        },
        {
            "name": "memory_efficiency_control",
            "payload_path": health_root / "memory_efficiency_control_latest.json",
            "cmd": [str(PY), str(ops_root / "memory_efficiency_control.py"), "status", "--json"],
        },
        {
            "name": "training_lineage_manifest",
            "payload_path": health_root / "training_lineage_manifest_latest.json",
            "cmd": [str(PY), str(ops_root / "training_lineage_manifest.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "training_quality_control",
            "payload_path": health_root / "training_quality_control_latest.json",
            "cmd": [str(PY), str(ops_root / "training_quality_control.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "architecture_upgrade_scoreboard",
            "payload_path": health_root / "architecture_upgrade_scoreboard_latest.json",
            "cmd": [str(PY), str(ops_root / "architecture_upgrade_scoreboard.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "system_architecture_contract_graph",
            "payload_path": health_root / "system_architecture_contract_graph_latest.json",
            "cmd": [str(PY), str(ops_root / "system_architecture_contract_graph.py"), "--apply", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "system_architecture_autopilot",
            "payload_path": health_root / "system_architecture_autopilot_latest.json",
            "cmd": [str(PY), str(ops_root / "system_architecture_autopilot.py"), "--apply", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "portfolio_capacity_curve_report",
            "payload_path": project_root / "governance" / "allocator" / "portfolio_capacity_curve_latest.json",
            "cmd": [str(PY), str(project_root / "scripts" / "portfolio_capacity_curve_report.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "cross_host_parity_report",
            "payload_path": health_root / "cross_host_parity_report_latest.json",
            "cmd": [str(PY), str(ops_root / "cross_host_parity_report.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "cost_telemetry",
            "payload_path": health_root / "cost_telemetry_latest.json",
            "cmd": [str(PY), str(ops_root / "cost_telemetry.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "broker_readiness",
            "payload_path": health_root / "broker_readiness_latest.json",
            "cmd": [str(PY), str(ops_root / "premarket_token_guard.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "session_ready",
            "payload_path": health_root / "session_ready_latest.json",
            "cmd": [str(PY), str(project_root / "scripts" / "session_ready_check.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "storage_failback_sync",
            "payload_path": health_root / "storage_failback_sync_latest.json",
            "cmd": [str(PY), str(ops_root / "storage_failback_sync.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "canary_auto_tuner",
            "payload_path": health_root / "canary_auto_tuner_latest.json",
            "cmd": [str(PY), str(ops_root / "canary_auto_tuner.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "canary_rollout_guard",
            "payload_path": health_root / "canary_rollout_latest.json",
            "cmd": [str(PY), str(project_root / "scripts" / "canary_rollout_guard.py")],
            "timeout_sec": 45,
            "optional": True,
        },
        {
            "name": "promotion_autopilot_packet",
            "payload_path": project_root / "governance" / "champion_challenger" / "promotion_autopilot_packet_latest.json",
            "cmd": [str(PY), str(ops_root / "promotion_autopilot_packet.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "source_verification",
            "payload_path": health_root / "source_verification_latest.json",
            "cmd": [str(PY), str(ops_root / "source_verification_report.py"), "--json"],
            "timeout_sec": 120,
        },
        {
            "name": "paper_profitability_control",
            "payload_path": health_root / "paper_profitability_control_latest.json",
            "cmd": [str(PY), str(ops_root / "paper_profitability_control.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "paper_replay_drill",
            "payload_path": health_root / "paper_replay_drill_latest.json",
            "cmd": [str(PY), str(project_root / "scripts" / "paper_replay_drill.py"), "--hours", "24", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "paper_execution_truth",
            "payload_path": health_root / "paper_execution_truth_layer_latest.json",
            "cmd": [str(PY), str(ops_root / "paper_execution_truth_layer.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "retrain_schema_compatibility",
            "payload_path": health_root / "retrain_schema_compatibility_latest.json",
            "cmd": [str(PY), str(project_root / "scripts" / "retrain_schema_compatibility_guard.py"), "--json"],
            "timeout_sec": 120,
        },
        {
            "name": "promotion_packet_builder",
            "payload_path": project_root / "governance" / "champion_challenger" / "promotion_packet_latest.json",
            "cmd": [str(PY), str(project_root / "scripts" / "promotion_packet_builder.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "promotion_quality_gate",
            "payload_path": health_root / "promotion_quality_gate_latest.json",
            "cmd": [str(PY), str(project_root / "scripts" / "promotion_quality_gate.py"), "--json"],
            "timeout_sec": 120,
        },
        {
            "name": "training_report",
            "payload_path": health_root / "training_report_latest.json",
            "cmd": [str(PY), str(ops_root / "training_report.py"), "--no-render-pdf", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "platform_control_plane",
            "payload_path": health_root / "platform_control_plane_latest.json",
            "cmd": [str(PY), str(project_root / "scripts" / "platform_control_plane_report.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "security_evidence_autofix",
            "payload_path": health_root / "security_evidence_autofix_latest.json",
            "cmd": [str(PY), str(ops_root / "security_evidence_autofix.py"), "--json"],
            "timeout_sec": 900,
        },
        {
            "name": "security_audit",
            "payload_path": health_root / "security_audit_latest.json",
            "cmd": [str(PY), str(project_root / "scripts" / "security_hardening_audit.py")],
            "timeout_sec": 180,
        },
        {
            "name": "ingestion_storage_control",
            "payload_path": health_root / "ingestion_storage_control_latest.json",
            "cmd": [str(PY), str(ops_root / "ingestion_storage_control.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "storage_pressure_clearance",
            "payload_path": health_root / "storage_pressure_clearance_latest.json",
            "cmd": [str(PY), str(ops_root / "storage_pressure_clearance_bot.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "storage_resilience_control",
            "payload_path": health_root / "storage_resilience_control_latest.json",
            "cmd": [str(PY), str(ops_root / "storage_resilience_control.py"), "--fast", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "notification_escalation_ladder",
            "payload_path": health_root / "notification_escalation_ladder_latest.json",
            "cmd": [str(PY), str(ops_root / "notification_escalation_ladder.py"), "--json"],
            "timeout_sec": 60,
        },
        {
            "name": "unattended_soak_readiness",
            "payload_path": health_root / "unattended_soak_readiness_latest.json",
            "cmd": [str(PY), str(ops_root / "unattended_soak_readiness.py"), "--json"],
            "timeout_sec": 60,
        },
        {
            "name": "live_runtime_separation_control",
            "payload_path": health_root / "live_runtime_separation_control_latest.json",
            "cmd": [str(PY), str(ops_root / "live_runtime_separation_control.py"), "--json"],
        },
        {
            "name": "auth_lease_manager",
            "payload_path": health_root / "auth_lease_manager_latest.json",
            "cmd": [str(PY), str(ops_root / "auth_lease_manager.py"), "--json"],
        },
        {
            "name": "schwab_auth_supervisor",
            "payload_path": health_root / "schwab_auth_supervisor_latest.json",
            "cmd": [str(PY), str(ops_root / "schwab_auth_supervisor.py"), "--json"],
        },
        {
            "name": "blackstart_recovery",
            "payload_path": health_root / "blackstart_recovery_latest.json",
            "cmd": [str(PY), str(ops_root / "blackstart_recovery.py"), "--json"],
        },
        {
            "name": "sleeve_isolation_guard",
            "payload_path": health_root / "sleeve_isolation_guard_latest.json",
            "cmd": [str(PY), str(ops_root / "sleeve_isolation_guard.py"), "--json"],
        },
        {
            "name": "artifact_freshness_slo",
            "payload_path": health_root / "artifact_freshness_slo_latest.json",
            "cmd": [str(PY), str(ops_root / "artifact_freshness_slo.py"), "--json"],
        },
        {
            "name": "runtime_snapshot_cache_control",
            "payload_path": health_root / "runtime_snapshot_cache_control_latest.json",
            "cmd": [str(PY), str(ops_root / "runtime_snapshot_cache_control.py"), "--json"],
        },
        {
            "name": "remote_alert_control",
            "payload_path": health_root / "remote_alert_control_latest.json",
            "cmd": [str(PY), str(ops_root / "remote_alert_control.py"), "--json"],
        },
        {
            "name": "schwab_account_snapshot_refresh",
            "payload_path": health_root / "schwab_account_snapshot_refresh_latest.json",
            "cmd": [str(PY), str(ops_root / "schwab_account_snapshot_refresh.py"), "--json", "--skip-derived"],
            "timeout_sec": 120,
        },
        {
            "name": "covered_call_roll_watch",
            "payload_path": health_root / "covered_call_roll_watch_latest.json",
            "cmd": [str(PY), str(ops_root / "covered_call_roll_watch.py"), "--json"],
        },
        {
            "name": "account_position_study",
            "payload_path": health_root / "account_position_study_latest.json",
            "cmd": [str(PY), str(ops_root / "account_position_study.py"), "--json"],
        },
        {
            "name": "storage_quota_guard",
            "payload_path": health_root / "storage_quota_guard_latest.json",
            "cmd": [str(PY), str(ops_root / "storage_quota_guard.py"), "--json"],
        },
        {
            "name": "release_freeze_guard",
            "payload_path": health_root / "release_freeze_guard_latest.json",
            "cmd": [str(PY), str(ops_root / "release_freeze_guard.py"), "--json"],
        },
        {
            "name": "roster_resilience_planner",
            "payload_path": health_root / "roster_resilience_planner_latest.json",
            "cmd": [str(PY), str(ops_root / "roster_resilience_planner.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "chaos_drill_coordinator",
            "payload_path": health_root / "chaos_drill_coordinator_latest.json",
            "cmd": [str(PY), str(ops_root / "chaos_drill_coordinator.py"), "--json"],
        },
        {
            "name": "rolling_restart_controller",
            "payload_path": health_root / "rolling_restart_controller_latest.json",
            "cmd": [str(PY), str(ops_root / "rolling_restart_controller.py"), "--json"],
        },
        {
            "name": "incident_timeline",
            "payload_path": health_root / "incident_timeline_latest.json",
            "cmd": [str(PY), str(ops_root / "incident_timeline.py"), "--json"],
        },
        {
            "name": "incident_review_packet",
            "payload_path": health_root / "incident_review_packet_latest.json",
            "cmd": [str(PY), str(ops_root / "incident_review_packet.py"), "--json"],
        },
        {
            "name": "incident_closeout_autopilot",
            "payload_path": health_root / "incident_closeout_autopilot_latest.json",
            "cmd": [str(PY), str(ops_root / "incident_closeout_autopilot.py"), "--json"],
        },
        {
            "name": "live_canary_control",
            "payload_path": health_root / "live_canary_control_latest.json",
            "cmd": [str(PY), str(ops_root / "live_canary_control.py"), "--json"],
        },
        {
            "name": "live_readiness_smoke",
            "payload_path": health_root / "live_readiness_smoke_latest.json",
            "cmd": [str(PY), str(project_root / "scripts" / "live_readiness_smoke.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "live_money_readiness_contract",
            "payload_path": health_root / "live_money_readiness_contract_latest.json",
            "cmd": [str(PY), str(ops_root / "live_money_readiness_contract.py"), "--json"],
            "timeout_sec": 120,
        },
        {
            "name": "runtime_throttle_control",
            "payload_path": health_root / "runtime_throttle_control_latest.json",
            "cmd": [str(PY), str(ops_root / "runtime_throttle_control.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "regime_control_plane",
            "payload_path": health_root / "regime_control_plane_latest.json",
            "cmd": [str(PY), str(ops_root / "regime_control_plane.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "market_cycle_extraction_engine",
            "payload_path": health_root / "market_cycle_state_latest.json",
            "cmd": [str(PY), str(ops_root / "market_cycle_extraction_engine.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "coordination_state_control",
            "payload_path": health_root / "coordination_state_latest.json",
            "cmd": [str(PY), str(ops_root / "coordination_state_control.py"), "--json"],
            "timeout_sec": 60,
        },
        {
            "name": "chrome_headless_guard",
            "payload_path": health_root / "chrome_headless_guard_latest.json",
            "cmd": [str(PY), str(ops_root / "chrome_headless_guard.py"), "--apply", "--json"],
            "timeout_sec": 90,
        },
        {
            "name": "multiple_testing_guard",
            "payload_path": project_root / "governance" / "research" / "multiple_testing_guard_latest.json",
            "cmd": [str(PY), str(project_root / "scripts" / "multiple_testing_guard.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "decay_monitor",
            "payload_path": project_root / "governance" / "research" / "decay_monitor_latest.json",
            "cmd": [str(PY), str(project_root / "scripts" / "decay_monitor.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "service_control_plane",
            "payload_path": health_root / "service_control_plane_latest.json",
            "cmd": [str(PY), str(project_root / "scripts" / "service_control_plane.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "operator_cockpit",
            "payload_path": health_root / "operator_cockpit_latest.json",
            "cmd": [str(PY), str(ops_root / "operator_cockpit.py"), "--json"],
            "timeout_sec": 180,
        },
    ]


def _run_spec(spec: dict[str, Any], project_root: Path) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    payload_path = Path(spec["payload_path"]).expanduser()
    try:
        proc = subprocess.run(
            list(spec["cmd"]),
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=max(int(spec.get("timeout_sec", 120) or 120), 1),
        )
        payload = _parse_json_output(proc.stdout or "")
        if not payload:
            payload = _load_json(payload_path)
        rc = int(proc.returncode)
        stdout_tail = "\n".join((proc.stdout or "").splitlines()[-12:])
        stderr_tail = "\n".join((proc.stderr or "").splitlines()[-12:])
    except subprocess.TimeoutExpired as exc:
        rc = 124
        payload = _load_json(payload_path)
        stdout_tail = "\n".join((exc.stdout or "").splitlines()[-12:]) if isinstance(exc.stdout, str) else ""
        stderr_tail = "\n".join((exc.stderr or "").splitlines()[-12:]) if isinstance(exc.stderr, str) else "timeout"
    duration_ms = round((datetime.now(timezone.utc) - started).total_seconds() * 1000.0, 3)
    return {
        "cmd": list(spec["cmd"]),
        "rc": rc,
        "payload": payload,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "duration_ms": duration_ms,
    }


def _paper_soak_contract_ready(project_root: Path) -> bool:
    health_root = project_root / "governance" / "health"
    soak = _load_json(health_root / "unattended_soak_readiness_latest.json")
    paper_guard = _load_json(health_root / "runtime_paper_regression_guard_latest.json")
    return bool(
        str(soak.get("overall_status") or "").strip().lower() == "ready"
        and bool(soak.get("ok", False))
        and bool(soak.get("safe_to_leave_unattended", False))
        and str(paper_guard.get("overall_status") or "").strip().lower() == "ready"
        and bool(paper_guard.get("ok", False))
    )


def _core_storage_ready(project_root: Path) -> bool:
    health_root = project_root / "governance" / "health"
    ingestion = _load_json(health_root / "ingestion_storage_control_latest.json")
    quota = _load_json(health_root / "storage_quota_guard_latest.json")
    return bool(
        str(ingestion.get("overall_status") or ingestion.get("status") or "").strip().lower() == "ready"
        and str(quota.get("overall_status") or quota.get("status") or "").strip().lower() == "ready"
    )


def _raw_live_backlog_clear(ingestion: dict[str, Any]) -> bool:
    backpressure = _as_dict(ingestion.get("backpressure"))
    raw_live = _as_dict(backpressure.get("effective_raw_live")) or _as_dict(backpressure.get("raw_live"))
    return bool(
        _safe_int(raw_live.get("core_pending_lines"), 0) <= RAW_LIVE_SOAK_MAX_CORE_LINES
        and _safe_int(raw_live.get("total_pending_lines"), 0) <= RAW_LIVE_SOAK_MAX_TOTAL_LINES
        and _safe_float(raw_live.get("oldest_pending_age_seconds"), 0.0) <= RAW_LIVE_SOAK_MAX_AGE_SECONDS
    )


def _stateful_sql_soft_quota_managed_for_paper_soak(project_root: Path, payload: dict[str, Any]) -> bool:
    quota_summary = _as_dict(payload.get("quota_summary"))
    hard_breaches = _safe_int(quota_summary.get("hard_breaches"), 0)
    soft_breaches = _safe_int(quota_summary.get("soft_breaches"), 0)
    if hard_breaches > 0 or soft_breaches != 1:
        return False
    if _string_set(quota_summary.get("blocked_families")):
        return False
    degraded_families = _string_set(quota_summary.get("degraded_families"))
    lanes = payload.get("lanes") if isinstance(payload.get("lanes"), list) else []
    if not degraded_families:
        degraded_families = {
            str(row.get("family") or "").strip()
            for row in lanes
            if isinstance(row, dict) and str(row.get("status") or "").strip().lower() in {"blocked", "degraded"}
        }
        degraded_families.discard("")
    if degraded_families != {"sql_link_shards"}:
        return False
    sql_lane = next((row for row in lanes if isinstance(row, dict) and str(row.get("family") or "") == "sql_link_shards"), {})
    over_hard_gb = _safe_float(sql_lane.get("over_hard_gb"), _safe_float(quota_summary.get("worst_over_hard_gb"), 0.0))
    hard_ratio = _safe_float(sql_lane.get("hard_ratio"), _safe_float(quota_summary.get("worst_hard_ratio"), 0.0))
    if over_hard_gb > 0.0 or hard_ratio > STATEFUL_SQL_SOFT_QUOTA_MAX_HARD_RATIO:
        return False

    health_root = project_root / "governance" / "health"
    ingestion = _load_json(health_root / "ingestion_storage_control_latest.json")
    ingestion_status = str(ingestion.get("overall_status") or ingestion.get("status") or "").strip().lower()
    severity = str(ingestion.get("severity") or "").strip().lower()
    if ingestion_status not in {"ready", "ok", "advisory"} or severity not in {"", "stable", "ready", "low", "normal"}:
        return False
    if not _raw_live_backlog_clear(ingestion):
        return False

    unison = _load_json(health_root / "storage_retention_unison_latest.json")
    continuous = _as_dict(unison.get("continuous_run_contract"))
    controls = _as_dict(continuous.get("storage_controls"))
    forecast = _as_dict(unison.get("storage_growth_forecast"))
    forecast_status = str(forecast.get("status") or "").strip()
    days_until_pressure = forecast.get("days_until_pressure_free")
    forecast_ready = bool(
        forecast_status in {"stable_or_improving", "forecast_ready", "ready"}
        and (days_until_pressure is None or _safe_float(days_until_pressure, 0.0) >= 30.0)
    )
    continuous_ready = bool(continuous.get("ready", False) or continuous.get("status") == "ready" or forecast_ready)
    quota_ready = bool(controls.get("quota_ready", False)) or not bool(quota_summary.get("external_free_below_target", False))
    tier = _load_json(health_root / "storage_tier_policy_latest.json")
    manifest_contract = _as_dict(tier.get("manifest_backed_offload_contract"))
    integration = _as_dict(unison.get("integration_contract"))
    stateful_policy = str(manifest_contract.get("stateful_sql_policy") or "").lower()
    compaction_only = bool(integration.get("stateful_sql_compaction_only", False)) or (
        "never source-delete" in stateful_policy and "checkpoint" in stateful_policy
    )
    return bool(continuous_ready and quota_ready and compaction_only)


def _paper_soak_managed_step(name: str, payload: dict[str, Any], *, project_root: Path, paper_soak_ready: bool) -> bool:
    if not paper_soak_ready:
        return False
    status = str(payload.get("overall_status") or payload.get("status") or "").strip().lower()
    if name == "storage_pressure_clearance":
        return bool(status in PAPER_SOAK_MANAGED_STATUSES and _core_storage_ready(project_root))
    if name == "storage_quota_guard":
        return bool(status in PAPER_SOAK_MANAGED_STATUSES and _stateful_sql_soft_quota_managed_for_paper_soak(project_root, payload))
    if name == "rolling_restart_controller":
        signals = _as_dict(payload.get("due_signals"))
        scope = str(payload.get("recommended_scope") or "").strip().lower()
        checkpoint_only = bool(signals.get("checkpoint_missing_or_stale", False)) and not any(
            bool(signals.get(key, False))
            for key in (
                "session_stale",
                "shadow_heartbeat_stale",
                "swap_pressure_high",
                "restart_storm_present",
            )
        )
        return bool(status in PAPER_SOAK_MANAGED_STATUSES and checkpoint_only and scope in {"", "none"})
    if name not in PAPER_SOAK_MANAGED_STEPS:
        return False
    if status in PAPER_SOAK_MANAGED_STATUSES:
        return True
    return bool(status == "" and "ok" in payload and not bool(payload.get("ok", False)))


def _step_status(result: dict[str, Any], *, name: str = "", project_root: Path = PROJECT_ROOT, paper_soak_ready: bool = False) -> str:
    payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
    if paper_soak_ready and name in PAPER_SOAK_MANAGED_STEPS and int(result.get("rc", 1)) != 0 and not payload:
        return "managed_paper_soak"
    if int(result.get("rc", 1)) != 0 and not payload:
        return "error"
    if bool(payload.get("busy", False)):
        return "busy"
    if bool(payload.get("skipped", False)):
        return "skipped"
    if _live_money_ready_locked(payload):
        return "ready_locked"
    if _idle_promotion_packet_seed_ready(payload):
        return "ready_seeded"
    if _retrain_schema_seed_ready(payload):
        return "ready_seeded"
    if _paper_soak_managed_step(name, payload, project_root=project_root, paper_soak_ready=paper_soak_ready):
        return "managed_paper_soak"
    status = str(payload.get("overall_status") or "").strip().lower()
    if (
        name == "paper_profitability_control"
        and status == "protective_tightening"
        and bool(payload.get("ok", False))
        and str(payload.get("controlled_profitability_grade") or payload.get("controlled_financial_grade") or "").strip().upper() == "A+"
    ):
        return "ready_protective"
    if status:
        return status
    if "ok" in payload:
        return "ready" if bool(payload.get("ok", False)) else "blocked"
    return "ok" if int(result.get("rc", 1)) == 0 else "error"


def _live_money_ready_locked(payload: dict[str, Any]) -> bool:
    blocking = {
        str(item or "").strip()
        for item in payload.get("blocking_reasons", [])
        if str(item or "").strip()
    }
    allowed_locks = {"target_window_not_complete", "live_execution_operator_release_required"}
    summary = payload.get("grade_summary") if isinstance(payload.get("grade_summary"), dict) else {}
    return bool(
        payload.get("live_money_locked", False)
        and blocking
        and blocking.issubset(allowed_locks)
        and not summary.get("below_floor_sections")
        and not summary.get("not_ready_sections")
    )


def _idle_promotion_packet_seed_ready(payload: dict[str, Any]) -> bool:
    scope = payload.get("promotion_scope") if isinstance(payload.get("promotion_scope"), dict) else {}
    gates = payload.get("gate_results") if isinstance(payload.get("gate_results"), dict) else {}
    replayability = (
        payload.get("replayability_contract")
        if isinstance(payload.get("replayability_contract"), dict)
        else {}
    )
    return bool(
        not bool(payload.get("ok", False))
        and not bool(scope.get("target_count", 0) or scope.get("trained_bot_ids") or scope.get("failure_count", 0))
        and bool(payload.get("committee_packet_seed_ready", False))
        and bool(replayability.get("hash_bundle_complete", False))
        and bool(replayability.get("exact_replay_ready", False))
        and gates
        and all(bool(value) for value in gates.values())
    )


def _retrain_schema_seed_ready(payload: dict[str, Any]) -> bool:
    return bool(
        bool(payload.get("ok", False))
        and bool(payload.get("compatibility_seed_ready", False))
        and not payload.get("failed_checks")
        and not payload.get("drifted_fields")
    )


def _payload_summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in (
        "overall_status",
        "ok",
        "timestamp_utc",
        "mode",
        "lease_state",
        "resilience_score",
        "account_snapshot_mode",
        "account_count",
        "discovered_account_count",
        "failed_account_count",
        "position_rows",
        "broker_truth_mismatch_count",
        "blocking_reasons",
        "grade_summary",
        "profitability_display_grade",
        "raw_profitability_grade",
        "rows",
        "failed_checks",
        "committee_packet_seed_ready",
        "packet_complete",
        "signing_material_ready",
        "compatibility_score",
        "compatibility_seed_ready",
    ):
        if key in payload:
            summary[key] = payload.get(key)
    source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
    if source:
        for key in ("execution_result_rows", "execution_result_stale_skip_rows", "execution_intent_rows", "source_mode"):
            if key in source:
                summary[key] = source.get(key)
    return summary


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    specs: list[dict[str, Any]] | None = None,
    runner: RefreshRunner | None = None,
) -> dict[str, Any]:
    refresh_specs = list(specs or _step_specs(project_root))
    run_step = runner or _run_spec
    missing_before = [str(spec["name"]) for spec in refresh_specs if not _artifact_present(Path(spec["payload_path"]))]

    steps: list[dict[str, Any]] = []
    statuses: list[str] = []
    missing_after: list[str] = []
    recovered = 0
    paper_soak_ready_before_refresh = _paper_soak_contract_ready(project_root)
    for spec in refresh_specs:
        payload_path = Path(spec["payload_path"])
        result = run_step(spec, project_root)
        payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
        present_after = _artifact_present(payload_path)
        if str(spec["name"]) in missing_before and present_after:
            recovered += 1
        if not present_after:
            missing_after.append(str(spec["name"]))
        steps.append(
            {
                "name": str(spec["name"]),
                "result": result,
                "payload": payload,
                "payload_path": payload_path,
                "present_after": present_after,
                "optional": bool(spec.get("optional", False)),
            }
        )

    paper_soak_ready_after_refresh = _paper_soak_contract_ready(project_root)
    paper_soak_ready = bool(paper_soak_ready_before_refresh or paper_soak_ready_after_refresh)
    rendered_steps: list[dict[str, Any]] = []
    for row in steps:
        payload_path = Path(row["payload_path"])
        result = row["result"] if isinstance(row.get("result"), dict) else {}
        payload = row["payload"] if isinstance(row.get("payload"), dict) else {}
        optional = bool(row.get("optional", False))
        status = _step_status(
            result,
            name=str(row["name"]),
            project_root=project_root,
            paper_soak_ready=paper_soak_ready,
        )
        if status == "error" and optional:
            status = "optional_advisory" if paper_soak_ready else "degraded"
        statuses.append(status)
        rendered_steps.append(
            {
                "name": str(row["name"]),
                "status": status,
                "rc": int(result.get("rc", 1)),
                "duration_ms": float(result.get("duration_ms", 0.0) or 0.0),
                "payload_path": str(payload_path),
                "optional": optional,
                "artifact_present": bool(row.get("present_after", False)),
                "payload_summary": _payload_summary(payload),
                "cmd": list(result.get("cmd") or []),
                "stdout_tail": str(result.get("stdout_tail") or ""),
                "stderr_tail": str(result.get("stderr_tail") or ""),
            }
        )

    optional_names = {str(spec["name"]) for spec in refresh_specs if bool(spec.get("optional", False))}
    required_missing_after = [name for name in missing_after if name not in optional_names]
    error_statuses = {"error"}
    degraded_statuses = {"warn", "thin", "degraded", "needs_work", "needs_review", "blocked", "busy", "skipped"}
    error_step_count = sum(1 for status in statuses if status in error_statuses)
    degraded_step_count = sum(1 for status in statuses if status in degraded_statuses)
    blocked_step_count = sum(1 for status in statuses if status == "blocked")
    managed_paper_soak_step_count = sum(1 for status in statuses if status == "managed_paper_soak")
    optional_advisory_step_count = sum(1 for status in statuses if status == "optional_advisory")
    overall_status = "ready"
    if error_step_count > 0 or required_missing_after:
        overall_status = "blocked"
    elif degraded_step_count > 0:
        overall_status = "degraded"

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "project_root": str(project_root),
        "ok": overall_status != "blocked",
        "overall_status": overall_status,
        "target_artifact_count": len(refresh_specs),
        "artifact_present_count_after": len(refresh_specs) - len(missing_after),
        "artifacts_recovered_count": recovered,
        "missing_before": missing_before,
        "missing_after": missing_after,
        "required_missing_after": required_missing_after,
        "blocked_step_count": blocked_step_count,
        "degraded_step_count": degraded_step_count,
        "error_step_count": error_step_count,
        "managed_paper_soak_step_count": managed_paper_soak_step_count,
        "optional_advisory_step_count": optional_advisory_step_count,
        "paper_soak_ready_before_refresh": paper_soak_ready_before_refresh,
        "paper_soak_ready_after_refresh": paper_soak_ready_after_refresh,
        "recommended_actions": ordered_unique(
            [
                "./scripts/ops/opsctl.sh dashboard" if not missing_after and error_step_count == 0 else "",
                "inspect the step stderr tails for the artifacts that are still missing" if required_missing_after else "",
                "treat optional proof steps like canary rollout diagnostics as advisory when they time out under live load" if any(name in optional_names for name in missing_after) else "",
                "treat blocked refresh outputs as real runtime issues instead of silent dashboard omissions" if blocked_step_count else "",
                "paper soak is green; proof, promotion, and research debts are tracked as managed_paper_soak without blocking collection" if managed_paper_soak_step_count else "",
            ]
        ),
        "steps": rendered_steps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh the runtime dashboard's prerequisite artifacts before grading the live system.")
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
            "runtime_artifact_refresh "
            f"overall_status={payload.get('overall_status', '')} "
            f"recovered={int(payload.get('artifacts_recovered_count', 0) or 0)} "
            f"missing_after={len(payload.get('missing_after') or [])}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
