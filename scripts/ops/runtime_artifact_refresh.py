#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.runtime_python import resolve_runtime_python
    from scripts.ops.long_runtime_common import iso_now, ordered_unique, run_bounded_process_group, write_payload
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from core.runtime_python import resolve_runtime_python
    from .long_runtime_common import iso_now, ordered_unique, run_bounded_process_group, write_payload


PY = resolve_runtime_python(PROJECT_ROOT)
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "runtime_artifact_refresh_latest.json"
REFRESH_ACTIVE_ENV = "RUNTIME_ARTIFACT_REFRESH_ACTIVE"
PAPER_SOAK_MANAGED_STEPS = {
    "training_lineage_manifest",
    "training_quality_control",
    "architecture_upgrade_scoreboard",
    "system_architecture_contract_graph",
    "system_architecture_autopilot",
    "system_drift_guard_pre_architecture",
    "portfolio_capacity_curve_report",
    "canary_rollout_guard",
    "promotion_autopilot_packet",
    "source_verification",
    "source_verification_autorefresh",
    "source_verification_verified",
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
    "profitability_evidence_firewall",
    "production_excellence_control",
    "continuous_soak_integrity_control",
    "live_transition_integrity_control",
    "live_money_readiness_contract_verified",
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


def _tail_text(text: str, *, max_lines: int = 12, max_chars: int = 1000) -> str:
    tail = "\n".join(str(text or "").splitlines()[-max(int(max_lines), 1) :])
    limit = max(int(max_chars), 1)
    if len(tail) <= limit:
        return tail
    return f"...[truncated {len(tail) - limit} chars]...\n{tail[-limit:]}"


def _artifact_present(path: Path) -> bool:
    return path.exists() and bool(_load_json(path))


def _artifact_signature(path: Path) -> tuple[int, int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (int(stat.st_mtime_ns), int(stat.st_size), int(stat.st_ino))


def _artifact_refreshed_since(
    path: Path,
    started: datetime,
    *,
    previous_signature: tuple[int, int, int] | None = None,
) -> bool:
    if not _artifact_present(path):
        return False
    current_signature = _artifact_signature(path)
    if current_signature is None:
        return False
    modified_during_cycle = current_signature[0] >= int(started.timestamp() * 1_000_000_000) - 1_000_000_000
    if previous_signature is None:
        return modified_during_cycle
    return modified_during_cycle and current_signature != previous_signature


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
            "name": "bot_needs_intelligence",
            "payload_path": health_root / "bot_needs_intelligence_latest.json",
            "cmd": [str(PY), str(ops_root / "bot_needs_intelligence.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "training_runtime_control",
            "payload_path": health_root / "training_runtime_control_latest.json",
            "cmd": [str(PY), str(ops_root / "training_runtime_control.py"), "--json"],
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
            "name": "source_verification_autorefresh",
            "payload_path": health_root / "source_verification_autorefresh_latest.json",
            "cmd": [
                str(PY),
                str(ops_root / "source_verification_autorefresh.py"),
                "--apply",
                "--max-commands",
                "1",
                "--max-heavy-commands",
                "1",
                "--json",
            ],
            "timeout_sec": 360,
            "optional": True,
        },
        {
            "name": "source_verification_verified",
            "payload_path": health_root / "source_verification_latest.json",
            "cmd": [str(PY), str(ops_root / "source_verification_report.py"), "--json"],
            "timeout_sec": 120,
        },
        {
            "name": "paper_performance",
            "payload_path": health_root / "paper_performance_latest.json",
            "cmd": [str(PY), str(project_root / "scripts" / "paper_performance_report.py"), "--week-days", "7", "--json-only", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "paper_profitability_control",
            "payload_path": health_root / "paper_profitability_control_latest.json",
            "additional_payload_paths": [health_root / "paper_runtime_profitability_controls_latest.json"],
            "cmd": [str(PY), str(ops_root / "paper_profitability_control.py"), "--apply", "--json"],
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
            "name": "new_bot_graduation_gate",
            "payload_path": project_root / "governance" / "walk_forward" / "new_bot_graduation_latest.json",
            "cmd": [str(PY), str(project_root / "scripts" / "new_bot_graduation_gate.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "new_bot_admission_guard",
            "payload_path": health_root / "new_bot_admission_guard_latest.json",
            "cmd": [str(PY), str(project_root / "scripts" / "new_bot_admission_guard.py"), "--json"],
            "timeout_sec": 180,
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
            "name": "storage_quota_guard",
            "payload_path": health_root / "storage_quota_guard_latest.json",
            "cmd": [str(PY), str(ops_root / "storage_quota_guard.py"), "--json"],
        },
        {
            "name": "state_snapshot_restore_drill",
            "payload_path": project_root / "exports" / "state_snapshot_drills" / "latest.json",
            "cmd": [str(PY), str(project_root / "scripts" / "daily_state_snapshot_drill.py"), "--json"],
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
            "name": "storage_retention_unison",
            "payload_path": health_root / "storage_retention_unison_latest.json",
            "cmd": [str(PY), str(ops_root / "storage_retention_unison.py"), "--json"],
            "timeout_sec": 240,
        },
        {
            "name": "storage_resilience_control_terminal",
            "payload_path": health_root / "storage_resilience_control_latest.json",
            "cmd": [str(PY), str(ops_root / "storage_resilience_control.py"), "--fast", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "storage_disaster_recovery",
            "payload_path": health_root / "storage_disaster_recovery_latest.json",
            "cmd": [str(PY), str(ops_root / "storage_disaster_recovery.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "notification_escalation_ladder",
            "payload_path": health_root / "notification_escalation_ladder_latest.json",
            "cmd": [str(PY), str(ops_root / "notification_escalation_ladder.py"), "--json"],
            "timeout_sec": 60,
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
            "cmd": [
                str(ops_root / "opsctl.sh"),
                "schwab-account-snapshot-refresh",
                "--json",
                "--skip-derived",
            ],
            "timeout_sec": 120,
        },
        {
            "name": "tax_regulation_update",
            "payload_path": health_root / "tax_regulation_update_latest.json",
            "cmd": [str(PY), str(ops_root / "tax_regulation_update.py"), "--auto", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "schwab_tax_ledger_refresh",
            "payload_path": health_root / "schwab_tax_ledger_refresh_latest.json",
            "cmd": [str(ops_root / "opsctl.sh"), "schwab-tax-ledger-refresh", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "trading_tax_estimate",
            "payload_path": health_root / "trading_tax_estimate_latest.json",
            "cmd": [str(PY), str(ops_root / "trading_tax_estimator.py"), "--json"],
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
            "name": "position_opportunity_watch",
            "payload_path": health_root / "position_opportunity_watch_latest.json",
            "cmd": [str(PY), str(ops_root / "position_opportunity_watch.py"), "--json"],
            "timeout_sec": 120,
        },
        {
            "name": "one_numbers_portfolio_prerequisite",
            "payload_path": health_root / "one_numbers_regression_guard_latest.json",
            "cmd": [
                str(ops_root / "opsctl.sh"),
                "one-numbers-regression-guard",
                "--apply",
                "--json",
            ],
            "timeout_sec": 180,
        },
        {
            "name": "sleeve_allocator",
            "payload_path": project_root / "governance" / "allocator" / "sleeve_allocator_latest.json",
            "cmd": [str(PY), str(project_root / "scripts" / "sleeve_allocator.py"), "--json"],
            "timeout_sec": 120,
        },
        {
            "name": "portfolio_risk_ledger",
            "payload_path": project_root / "governance" / "risk" / "portfolio_risk_latest.json",
            "cmd": [str(PY), str(project_root / "scripts" / "portfolio_risk_ledger.py"), "--json"],
            "timeout_sec": 120,
        },
        {
            "name": "execution_budget",
            "payload_path": project_root / "governance" / "risk" / "execution_budget_latest.json",
            "cmd": [str(PY), str(project_root / "scripts" / "execution_budgeter.py"), "--json"],
            "timeout_sec": 120,
        },
        {
            "name": "risk_service_boundary",
            "payload_path": project_root / "governance" / "risk" / "risk_service_boundary_latest.json",
            "cmd": [str(PY), str(project_root / "scripts" / "risk_service_boundary.py"), "--json"],
            "timeout_sec": 120,
        },
        {
            "name": "position_round_trip_watch",
            "payload_path": health_root / "position_round_trip_watch_latest.json",
            "cmd": [
                str(ops_root / "opsctl.sh"),
                "position-round-trip-watch",
                "--refresh-market-data",
                "--json",
            ],
            "timeout_sec": 240,
        },
        {
            "name": "portfolio_allocator_service",
            "payload_path": project_root / "governance" / "allocator" / "portfolio_allocator_service_latest.json",
            "cmd": [str(PY), str(project_root / "scripts" / "portfolio_allocator_service.py"), "--json"],
            "timeout_sec": 120,
        },
        {
            "name": "account_buildout_plan",
            "payload_path": health_root / "account_buildout_plan_latest.json",
            "cmd": [str(PY), str(ops_root / "account_buildout_planner.py"), "--json"],
            "timeout_sec": 120,
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
            "additional_payload_paths": [health_root / "production_recovery_drill_harness_latest.json"],
            "cmd": [str(PY), str(ops_root / "chaos_drill_coordinator.py"), "--run-isolated", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "live_order_ledger_control",
            "payload_path": health_root / "live_order_ledger_control_latest.json",
            "cmd": [str(PY), str(ops_root / "live_order_ledger_control.py"), "--json"],
            "timeout_sec": 60,
        },
        {
            "name": "rolling_restart_controller",
            "payload_path": health_root / "rolling_restart_controller_latest.json",
            "cmd": [str(PY), str(ops_root / "rolling_restart_controller.py"), "--json"],
        },
        {
            "name": "runtime_throttle_control",
            "payload_path": health_root / "runtime_throttle_control_latest.json",
            "cmd": [str(PY), str(ops_root / "runtime_throttle_control.py"), "--apply", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "paper_400_ramp",
            "payload_path": health_root / "paper_400_ramp_latest.json",
            "cmd": [str(PY), str(ops_root / "paper_400_ramp_control.py"), "--apply", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "runtime_paper_regression_guard",
            "payload_path": health_root / "runtime_paper_regression_guard_latest.json",
            "cmd": [str(PY), str(ops_root / "runtime_paper_regression_guard.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "live_runtime_separation_control",
            "payload_path": health_root / "live_runtime_separation_control_latest.json",
            "cmd": [str(PY), str(ops_root / "live_runtime_separation_control.py"), "--json"],
        },
        {
            "name": "paper_reconciliation_slo",
            "payload_path": health_root / "paper_reconciliation_slo_latest.json",
            "cmd": [str(PY), str(project_root / "scripts" / "paper_reconciliation_slo_guard.py"), "--json"],
            "timeout_sec": 60,
        },
        {
            "name": "live_reconciliation_slo",
            "payload_path": health_root / "live_reconciliation_slo_latest.json",
            "cmd": [str(PY), str(project_root / "scripts" / "live_reconciliation_slo_guard.py"), "--json"],
            "timeout_sec": 60,
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
            "name": "blackstart_recovery",
            "payload_path": health_root / "blackstart_recovery_latest.json",
            "cmd": [str(PY), str(ops_root / "blackstart_recovery.py"), "--json"],
        },
        {
            "name": "live_money_readiness_contract",
            "payload_path": health_root / "live_money_readiness_contract_latest.json",
            "cmd": [str(PY), str(ops_root / "live_money_readiness_contract.py"), "--json"],
            "timeout_sec": 120,
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
            "name": "ingestion_backpressure_final",
            "payload_path": health_root / "ingestion_backpressure_latest.json",
            "cmd": [str(PY), str(project_root / "scripts" / "ingestion_backpressure_guard.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "ingestion_storage_control_final",
            "payload_path": health_root / "ingestion_storage_control_latest.json",
            "cmd": [str(PY), str(ops_root / "ingestion_storage_control.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "ingestion_storage_governor_final",
            "payload_path": health_root / "ingestion_storage_governor_latest.json",
            "cmd": [str(PY), str(ops_root / "ingestion_storage_governor.py"), "apply", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "ingestion_storage_control_post_governor",
            "payload_path": health_root / "ingestion_storage_control_latest.json",
            "cmd": [str(PY), str(ops_root / "ingestion_storage_control.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "ingestion_storage_governor_verify",
            "payload_path": health_root / "ingestion_storage_governor_latest.json",
            "cmd": [str(PY), str(ops_root / "ingestion_storage_governor.py"), "apply", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "ingestion_storage_control_verified",
            "payload_path": health_root / "ingestion_storage_control_latest.json",
            "cmd": [str(PY), str(ops_root / "ingestion_storage_control.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "health_fast",
            "payload_path": health_root / "health_fast_latest.json",
            "cmd": [str(PY), str(ops_root / "health_fast.py"), "--project-root", str(project_root), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "unattended_soak_readiness",
            "payload_path": health_root / "unattended_soak_readiness_latest.json",
            "cmd": [str(PY), str(ops_root / "unattended_soak_readiness.py"), "--json"],
            "timeout_sec": 60,
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
            "name": "health_gates",
            "payload_path": health_root / "health_gates_latest.json",
            "cmd": [
                str(PY),
                str(project_root / "scripts" / "health_gates.py"),
                "--project-root",
                str(project_root),
                "--json",
            ],
            "timeout_sec": 180,
        },
        {
            "name": "service_control_plane",
            "payload_path": health_root / "service_control_plane_latest.json",
            "cmd": [str(PY), str(project_root / "scripts" / "service_control_plane.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "runtime_throttle_control_verified",
            "payload_path": health_root / "runtime_throttle_control_latest.json",
            "cmd": [str(PY), str(ops_root / "runtime_throttle_control.py"), "--apply", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "paper_400_ramp_verified",
            "payload_path": health_root / "paper_400_ramp_latest.json",
            "cmd": [str(PY), str(ops_root / "paper_400_ramp_control.py"), "--apply", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "runtime_paper_regression_guard_verified",
            "payload_path": health_root / "runtime_paper_regression_guard_latest.json",
            "cmd": [str(PY), str(ops_root / "runtime_paper_regression_guard.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "halt_trigger_control_plane_verified",
            "payload_path": health_root / "halt_trigger_control_plane_latest.json",
            "cmd": [str(PY), str(ops_root / "halt_trigger_control_plane.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "coordination_state_control_verified",
            "payload_path": health_root / "coordination_state_latest.json",
            "cmd": [str(PY), str(ops_root / "coordination_state_control.py"), "--json"],
            "timeout_sec": 60,
        },
        {
            "name": "health_fast_verified",
            "payload_path": health_root / "health_fast_latest.json",
            "cmd": [str(PY), str(ops_root / "health_fast.py"), "--project-root", str(project_root), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "unattended_soak_readiness_verified",
            "payload_path": health_root / "unattended_soak_readiness_latest.json",
            "cmd": [str(PY), str(ops_root / "unattended_soak_readiness.py"), "--json"],
            "timeout_sec": 60,
        },
        {
            "name": "replay_hash_registry_final",
            "payload_path": health_root / "replay_hash_registry_guard_latest.json",
            "cmd": [str(PY), str(project_root / "scripts" / "replay_hash_registry_guard.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "golden_replay_regression_final",
            "payload_path": health_root / "golden_replay_regression_latest.json",
            "cmd": [str(PY), str(project_root / "scripts" / "golden_replay_regression_guard.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "paper_execution_truth_verified",
            "payload_path": health_root / "paper_execution_truth_layer_latest.json",
            "cmd": [str(PY), str(ops_root / "paper_execution_truth_layer.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "stateful_storage_regression_guard_verified",
            "payload_path": health_root / "stateful_storage_regression_guard_latest.json",
            "cmd": [
                str(ops_root / "opsctl.sh"),
                "stateful-storage-regression-guard",
                "--apply",
                "--json",
            ],
            "timeout_sec": 180,
        },
        {
            "name": "one_numbers_regression_guard_verified",
            "payload_path": health_root / "one_numbers_regression_guard_latest.json",
            "cmd": [
                str(ops_root / "opsctl.sh"),
                "one-numbers-regression-guard",
                "--apply",
                "--json",
            ],
            "timeout_sec": 180,
        },
        {
            "name": "grade_regression_guard_verified",
            "payload_path": health_root / "grade_regression_guard_latest.json",
            "cmd": [str(PY), str(ops_root / "grade_regression_guard.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "section_grade_guard_verified",
            "payload_path": health_root / "section_grade_guard_latest.json",
            "cmd": [str(PY), str(ops_root / "section_grade_guard.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "system_drift_registry_verified",
            "payload_path": health_root / "system_drift_registry_latest.json",
            "cmd": [str(PY), str(ops_root / "system_drift_registry.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "codex_project_guard_verified",
            "payload_path": health_root / "codex_project_guard_latest.json",
            "cmd": [str(ops_root / "opsctl.sh"), "codex-project-guard", "--staged", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "coinbase_api_health_verified",
            "payload_path": health_root / "coinbase_api_health_latest.json",
            "cmd": [str(ops_root / "opsctl.sh"), "coinbase-api-health", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "infrastructure_autofix_verified",
            "payload_path": health_root / "infrastructure_autofix_bot_latest.json",
            "cmd": [str(PY), str(ops_root / "infrastructure_autofix_bot.py"), "--apply", "--json"],
            "timeout_sec": 300,
        },
        {
            "name": "master_infrastructure_supervisor_verified",
            "payload_path": health_root / "master_infrastructure_supervisor_latest.json",
            "cmd": [str(PY), str(ops_root / "master_infrastructure_supervisor.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "process_watchdog_verified",
            "payload_path": health_root / "process_watchdog_latest.json",
            "cmd": [str(ops_root / "opsctl.sh"), "process-watchdog", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "livefeed_refresh_guard_verified",
            "payload_path": health_root / "livefeed_refresh_guard_latest.json",
            "cmd": [str(PY), str(ops_root / "livefeed_refresh_guard.py"), "--apply", "--json"],
            "timeout_sec": 90,
        },
        {
            "name": "runtime_throttle_control_final",
            "payload_path": health_root / "runtime_throttle_control_latest.json",
            "cmd": [str(PY), str(ops_root / "runtime_throttle_control.py"), "--apply", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "paper_400_ramp_final",
            "payload_path": health_root / "paper_400_ramp_latest.json",
            "cmd": [str(PY), str(ops_root / "paper_400_ramp_control.py"), "--apply", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "runtime_paper_regression_guard_final",
            "payload_path": health_root / "runtime_paper_regression_guard_latest.json",
            "cmd": [str(PY), str(ops_root / "runtime_paper_regression_guard.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "halt_trigger_control_plane_final",
            "payload_path": health_root / "halt_trigger_control_plane_latest.json",
            "cmd": [str(PY), str(ops_root / "halt_trigger_control_plane.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "coordination_state_control_final",
            "payload_path": health_root / "coordination_state_latest.json",
            "cmd": [str(PY), str(ops_root / "coordination_state_control.py"), "--json"],
            "timeout_sec": 60,
        },
        {
            "name": "health_fast_final",
            "payload_path": health_root / "health_fast_latest.json",
            "cmd": [str(PY), str(ops_root / "health_fast.py"), "--project-root", str(project_root), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "adaptive_regression_guard_final",
            "payload_path": health_root / "adaptive_regression_guard_latest.json",
            "cmd": [str(PY), str(ops_root / "adaptive_regression_guard.py"), "--apply", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "system_drift_guard_pre_architecture",
            "payload_path": health_root / "system_drift_guard_latest.json",
            "cmd": [str(PY), str(ops_root / "system_drift_guard.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "schwab_indicator_intelligence_verified",
            "payload_path": health_root / "schwab_indicator_intelligence_latest.json",
            "cmd": [str(ops_root / "opsctl.sh"), "schwab-indicator-intelligence", "--offline", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "system_expansion_execution_verified",
            "payload_path": health_root / "system_expansion_execution_layer_latest.json",
            "cmd": [str(ops_root / "opsctl.sh"), "system-expansion-execution", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "distributed_cell_architecture_verified",
            "payload_path": health_root / "distributed_cell_architecture_latest.json",
            "cmd": [str(ops_root / "opsctl.sh"), "distributed-cell-architecture", "--apply", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "system_architecture_hardening_verified",
            "payload_path": health_root / "system_architecture_hardening_latest.json",
            "cmd": [str(ops_root / "opsctl.sh"), "system-architecture-hardening", "--apply", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "system_self_model_pre_architecture",
            "payload_path": health_root / "system_self_model_latest.json",
            "cmd": [str(ops_root / "opsctl.sh"), "big-platform-brain", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "system_architecture_contract_graph_final",
            "payload_path": health_root / "system_architecture_contract_graph_latest.json",
            "cmd": [str(PY), str(ops_root / "system_architecture_contract_graph.py"), "--apply", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "system_architecture_autopilot_final",
            "payload_path": health_root / "system_architecture_autopilot_latest.json",
            "cmd": [str(PY), str(ops_root / "system_architecture_autopilot.py"), "--apply", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "system_drift_guard_verified",
            "payload_path": health_root / "system_drift_guard_latest.json",
            "cmd": [str(PY), str(ops_root / "system_drift_guard.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "system_drift_autopilot_verified",
            "payload_path": health_root / "system_drift_autopilot_latest.json",
            "cmd": [str(PY), str(ops_root / "system_drift_autopilot.py"), "--apply", "--json"],
            "timeout_sec": 300,
        },
        {
            "name": "master_infrastructure_supervisor_final",
            "payload_path": health_root / "master_infrastructure_supervisor_latest.json",
            "cmd": [str(PY), str(ops_root / "master_infrastructure_supervisor.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "system_self_model_verified",
            "payload_path": health_root / "system_self_model_latest.json",
            "cmd": [str(ops_root / "opsctl.sh"), "big-platform-brain", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "system_architecture_contract_graph_verified",
            "payload_path": health_root / "system_architecture_contract_graph_latest.json",
            "cmd": [str(PY), str(ops_root / "system_architecture_contract_graph.py"), "--apply", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "system_architecture_autopilot_verified",
            "payload_path": health_root / "system_architecture_autopilot_latest.json",
            "cmd": [str(PY), str(ops_root / "system_architecture_autopilot.py"), "--apply", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "broker_readiness_terminal",
            "payload_path": health_root / "broker_readiness_latest.json",
            "cmd": [str(PY), str(ops_root / "premarket_token_guard.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "auth_lease_manager_terminal",
            "payload_path": health_root / "auth_lease_manager_latest.json",
            "cmd": [str(PY), str(ops_root / "auth_lease_manager.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "schwab_auth_supervisor_terminal",
            "payload_path": health_root / "schwab_auth_supervisor_latest.json",
            "cmd": [str(PY), str(ops_root / "schwab_auth_supervisor.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "backlog_pcore_accelerator_terminal",
            "payload_path": health_root / "backlog_pcore_accelerator_latest.json",
            "cmd": [
                str(ops_root / "opsctl.sh"),
                "backlog-pcore-accelerator",
                "--apply",
                "--json",
            ],
            "timeout_sec": 180,
        },
        {
            "name": "backpressure_drainer_fleet_terminal",
            "payload_path": health_root / "backpressure_drainer_fleet_latest.json",
            "cmd": [str(PY), str(ops_root / "backpressure_drainer_fleet.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "ingestion_storage_control_terminal",
            "payload_path": health_root / "ingestion_storage_control_latest.json",
            "cmd": [str(PY), str(ops_root / "ingestion_storage_control.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "runtime_throttle_control_terminal",
            "payload_path": health_root / "runtime_throttle_control_latest.json",
            "cmd": [str(PY), str(ops_root / "runtime_throttle_control.py"), "--apply", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "paper_400_ramp_terminal",
            "payload_path": health_root / "paper_400_ramp_latest.json",
            "cmd": [str(PY), str(ops_root / "paper_400_ramp_control.py"), "--apply", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "runtime_paper_regression_guard_terminal",
            "payload_path": health_root / "runtime_paper_regression_guard_latest.json",
            "cmd": [str(PY), str(ops_root / "runtime_paper_regression_guard.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "halt_trigger_control_plane_terminal",
            "payload_path": health_root / "halt_trigger_control_plane_latest.json",
            "cmd": [str(PY), str(ops_root / "halt_trigger_control_plane.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "coordination_state_control_terminal",
            "payload_path": health_root / "coordination_state_latest.json",
            "cmd": [str(PY), str(ops_root / "coordination_state_control.py"), "--json"],
            "timeout_sec": 60,
        },
        {
            "name": "health_fast_terminal",
            "payload_path": health_root / "health_fast_latest.json",
            "cmd": [str(PY), str(ops_root / "health_fast.py"), "--project-root", str(project_root), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "unattended_soak_readiness_terminal",
            "payload_path": health_root / "unattended_soak_readiness_latest.json",
            "cmd": [str(PY), str(ops_root / "unattended_soak_readiness.py"), "--json"],
            "timeout_sec": 60,
        },
        {
            "name": "one_numbers_regression_guard_terminal",
            "payload_path": health_root / "one_numbers_regression_guard_latest.json",
            "cmd": [
                str(ops_root / "opsctl.sh"),
                "one-numbers-regression-guard",
                "--apply",
                "--json",
            ],
            "timeout_sec": 180,
        },
        {
            "name": "grade_regression_guard_terminal",
            "payload_path": health_root / "grade_regression_guard_latest.json",
            "cmd": [str(PY), str(ops_root / "grade_regression_guard.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "section_grade_guard_terminal",
            "payload_path": health_root / "section_grade_guard_latest.json",
            "cmd": [str(PY), str(ops_root / "section_grade_guard.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "livefeed_refresh_guard_terminal",
            "payload_path": health_root / "livefeed_refresh_guard_latest.json",
            "cmd": [str(PY), str(ops_root / "livefeed_refresh_guard.py"), "--apply", "--json"],
            "timeout_sec": 90,
        },
        {
            "name": "adaptive_regression_guard_terminal",
            "payload_path": health_root / "adaptive_regression_guard_latest.json",
            "cmd": [str(PY), str(ops_root / "adaptive_regression_guard.py"), "--apply", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "system_self_model_convergence",
            "payload_path": health_root / "system_self_model_latest.json",
            "cmd": [str(ops_root / "opsctl.sh"), "big-platform-brain", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "system_architecture_contract_graph_convergence",
            "payload_path": health_root / "system_architecture_contract_graph_latest.json",
            "cmd": [str(PY), str(ops_root / "system_architecture_contract_graph.py"), "--apply", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "system_architecture_autopilot_convergence",
            "payload_path": health_root / "system_architecture_autopilot_latest.json",
            "cmd": [str(PY), str(ops_root / "system_architecture_autopilot.py"), "--apply", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "system_drift_guard_terminal",
            "payload_path": health_root / "system_drift_guard_latest.json",
            "cmd": [str(PY), str(ops_root / "system_drift_guard.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "system_drift_autopilot_terminal",
            "payload_path": health_root / "system_drift_autopilot_latest.json",
            "cmd": [str(PY), str(ops_root / "system_drift_autopilot.py"), "--apply", "--json"],
            "timeout_sec": 300,
        },
        {
            "name": "system_drift_guard_post_autopilot_terminal",
            "payload_path": health_root / "system_drift_guard_latest.json",
            "cmd": [str(PY), str(ops_root / "system_drift_guard.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "master_infrastructure_supervisor_terminal",
            "payload_path": health_root / "master_infrastructure_supervisor_latest.json",
            "cmd": [str(PY), str(ops_root / "master_infrastructure_supervisor.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "infrastructure_autofix_terminal",
            "payload_path": health_root / "infrastructure_autofix_bot_latest.json",
            "cmd": [str(PY), str(ops_root / "infrastructure_autofix_bot.py"), "--apply", "--json"],
            "timeout_sec": 300,
        },
        {
            "name": "system_self_model_final",
            "payload_path": health_root / "system_self_model_latest.json",
            "cmd": [str(ops_root / "opsctl.sh"), "big-platform-brain", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "system_architecture_contract_graph_terminal",
            "payload_path": health_root / "system_architecture_contract_graph_latest.json",
            "cmd": [str(PY), str(ops_root / "system_architecture_contract_graph.py"), "--apply", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "system_architecture_autopilot_terminal",
            "payload_path": health_root / "system_architecture_autopilot_latest.json",
            "cmd": [str(PY), str(ops_root / "system_architecture_autopilot.py"), "--apply", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "system_drift_guard_settled",
            "payload_path": health_root / "system_drift_guard_latest.json",
            "cmd": [str(PY), str(ops_root / "system_drift_guard.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "system_drift_autopilot_settled",
            "payload_path": health_root / "system_drift_autopilot_latest.json",
            "cmd": [str(PY), str(ops_root / "system_drift_autopilot.py"), "--apply", "--json"],
            "timeout_sec": 300,
        },
        {
            "name": "system_drift_guard_post_settled",
            "payload_path": health_root / "system_drift_guard_latest.json",
            "cmd": [str(PY), str(ops_root / "system_drift_guard.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "master_infrastructure_supervisor_settled",
            "payload_path": health_root / "master_infrastructure_supervisor_latest.json",
            "cmd": [str(PY), str(ops_root / "master_infrastructure_supervisor.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "infrastructure_autofix_settled",
            "payload_path": health_root / "infrastructure_autofix_bot_latest.json",
            "cmd": [str(PY), str(ops_root / "infrastructure_autofix_bot.py"), "--apply", "--json"],
            "timeout_sec": 300,
        },
        {
            "name": "system_self_model_settled",
            "payload_path": health_root / "system_self_model_latest.json",
            "cmd": [str(ops_root / "opsctl.sh"), "big-platform-brain", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "system_architecture_contract_graph_settled",
            "payload_path": health_root / "system_architecture_contract_graph_latest.json",
            "cmd": [str(PY), str(ops_root / "system_architecture_contract_graph.py"), "--apply", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "system_architecture_autopilot_settled",
            "payload_path": health_root / "system_architecture_autopilot_latest.json",
            "cmd": [str(PY), str(ops_root / "system_architecture_autopilot.py"), "--apply", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "system_drift_guard_final",
            "payload_path": health_root / "system_drift_guard_latest.json",
            "cmd": [str(PY), str(ops_root / "system_drift_guard.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "master_infrastructure_supervisor_final_settled",
            "payload_path": health_root / "master_infrastructure_supervisor_latest.json",
            "cmd": [str(PY), str(ops_root / "master_infrastructure_supervisor.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "broker_readiness_post_settlement",
            "payload_path": health_root / "broker_readiness_latest.json",
            "cmd": [str(PY), str(ops_root / "premarket_token_guard.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "auth_lease_manager_post_settlement",
            "payload_path": health_root / "auth_lease_manager_latest.json",
            "cmd": [str(PY), str(ops_root / "auth_lease_manager.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "schwab_auth_supervisor_post_settlement",
            "payload_path": health_root / "schwab_auth_supervisor_latest.json",
            "cmd": [str(PY), str(ops_root / "schwab_auth_supervisor.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "backpressure_drainer_fleet_post_settlement",
            "payload_path": health_root / "backpressure_drainer_fleet_latest.json",
            "cmd": [str(PY), str(ops_root / "backpressure_drainer_fleet.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "ingestion_storage_control_post_settlement",
            "payload_path": health_root / "ingestion_storage_control_latest.json",
            "cmd": [str(PY), str(ops_root / "ingestion_storage_control.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "runtime_throttle_control_post_settlement",
            "payload_path": health_root / "runtime_throttle_control_latest.json",
            "cmd": [str(PY), str(ops_root / "runtime_throttle_control.py"), "--apply", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "paper_400_ramp_post_settlement",
            "payload_path": health_root / "paper_400_ramp_latest.json",
            "cmd": [str(PY), str(ops_root / "paper_400_ramp_control.py"), "--apply", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "runtime_throttle_control_post_settlement_verified",
            "payload_path": health_root / "runtime_throttle_control_latest.json",
            "cmd": [str(PY), str(ops_root / "runtime_throttle_control.py"), "--apply", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "paper_400_ramp_post_settlement_verified",
            "payload_path": health_root / "paper_400_ramp_latest.json",
            "cmd": [str(PY), str(ops_root / "paper_400_ramp_control.py"), "--apply", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "runtime_paper_regression_guard_post_settlement",
            "payload_path": health_root / "runtime_paper_regression_guard_latest.json",
            "cmd": [str(PY), str(ops_root / "runtime_paper_regression_guard.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "halt_trigger_control_plane_post_settlement",
            "payload_path": health_root / "halt_trigger_control_plane_latest.json",
            "cmd": [str(PY), str(ops_root / "halt_trigger_control_plane.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "coordination_state_control_post_settlement",
            "payload_path": health_root / "coordination_state_latest.json",
            "cmd": [str(PY), str(ops_root / "coordination_state_control.py"), "--json"],
            "timeout_sec": 60,
        },
        {
            "name": "health_fast_post_settlement",
            "payload_path": health_root / "health_fast_latest.json",
            "cmd": [str(PY), str(ops_root / "health_fast.py"), "--project-root", str(project_root), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "live_runtime_separation_post_settlement",
            "payload_path": health_root / "live_runtime_separation_control_latest.json",
            "cmd": [str(PY), str(ops_root / "live_runtime_separation_control.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "incident_closeout_autopilot_post_settlement",
            "payload_path": health_root / "incident_closeout_autopilot_latest.json",
            "cmd": [str(PY), str(ops_root / "incident_closeout_autopilot.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "sleeve_isolation_guard_post_settlement",
            "payload_path": health_root / "sleeve_isolation_guard_latest.json",
            "cmd": [str(PY), str(ops_root / "sleeve_isolation_guard.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "section_grade_guard_post_settlement",
            "payload_path": health_root / "section_grade_guard_latest.json",
            "cmd": [str(PY), str(ops_root / "section_grade_guard.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "unattended_soak_readiness_post_settlement",
            "payload_path": health_root / "unattended_soak_readiness_latest.json",
            "cmd": [str(PY), str(ops_root / "unattended_soak_readiness.py"), "--json"],
            "timeout_sec": 60,
        },
        {
            "name": "livefeed_refresh_guard_post_settlement",
            "payload_path": health_root / "livefeed_refresh_guard_latest.json",
            "cmd": [str(PY), str(ops_root / "livefeed_refresh_guard.py"), "--apply", "--json"],
            "timeout_sec": 90,
        },
        {
            "name": "adaptive_regression_guard_post_settlement",
            "payload_path": health_root / "adaptive_regression_guard_latest.json",
            "cmd": [str(PY), str(ops_root / "adaptive_regression_guard.py"), "--apply", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "system_drift_guard_post_evidence_probe",
            "payload_path": health_root / "system_drift_guard_latest.json",
            "cmd": [str(PY), str(ops_root / "system_drift_guard.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "system_architecture_contract_graph_post_evidence_probe",
            "payload_path": health_root / "system_architecture_contract_graph_latest.json",
            "cmd": [str(PY), str(ops_root / "system_architecture_contract_graph.py"), "--apply", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "system_drift_guard_post_evidence_reconciled",
            "payload_path": health_root / "system_drift_guard_latest.json",
            "cmd": [str(PY), str(ops_root / "system_drift_guard.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "system_architecture_contract_graph_post_evidence_verified",
            "payload_path": health_root / "system_architecture_contract_graph_latest.json",
            "cmd": [str(PY), str(ops_root / "system_architecture_contract_graph.py"), "--apply", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "system_architecture_autopilot_post_evidence_verified",
            "payload_path": health_root / "system_architecture_autopilot_latest.json",
            "cmd": [str(PY), str(ops_root / "system_architecture_autopilot.py"), "--apply", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "system_drift_guard_post_architecture_verified",
            "payload_path": health_root / "system_drift_guard_latest.json",
            "cmd": [str(PY), str(ops_root / "system_drift_guard.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "system_drift_autopilot_post_evidence_verified",
            "payload_path": health_root / "system_drift_autopilot_latest.json",
            "cmd": [str(PY), str(ops_root / "system_drift_autopilot.py"), "--apply", "--json"],
            "timeout_sec": 300,
        },
        {
            "name": "profitability_evidence_firewall",
            "payload_path": health_root / "profitability_evidence_firewall_latest.json",
            "cmd": [str(PY), str(ops_root / "profitability_evidence_firewall.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "content_addressed_artifact_store",
            "payload_path": project_root / "governance" / "content_store" / "latest.json",
            "cmd": [str(PY), str(ops_root / "content_addressed_artifact_store.py"), "--json"],
            "timeout_sec": 300,
        },
        {
            "name": "storage_disaster_recovery_verified",
            "payload_path": health_root / "storage_disaster_recovery_latest.json",
            "cmd": [str(PY), str(ops_root / "storage_disaster_recovery.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "production_readiness_control",
            "payload_path": health_root / "production_readiness_control_latest.json",
            "cmd": [str(PY), str(ops_root / "production_readiness_control.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "production_excellence_control",
            "payload_path": health_root / "production_excellence_control_latest.json",
            "cmd": [str(PY), str(ops_root / "production_excellence_control.py"), "--apply", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "continuous_soak_integrity_control",
            "payload_path": health_root / "continuous_soak_integrity_control_latest.json",
            "cmd": [str(PY), str(ops_root / "continuous_soak_integrity_control.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "live_transition_integrity_control",
            "payload_path": health_root / "live_transition_integrity_control_latest.json",
            "cmd": [str(PY), str(ops_root / "live_transition_integrity_control.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "live_money_readiness_contract_verified",
            "payload_path": health_root / "live_money_readiness_contract_latest.json",
            "cmd": [str(PY), str(ops_root / "live_money_readiness_contract.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "runtime_gate_dashboard_pre_master",
            "payload_path": health_root / "runtime_gate_dashboard_latest.json",
            "cmd": [str(PY), str(ops_root / "runtime_gate_dashboard.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "master_infrastructure_supervisor_post_evidence_probe",
            "payload_path": health_root / "master_infrastructure_supervisor_latest.json",
            "cmd": [str(PY), str(ops_root / "master_infrastructure_supervisor.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "infrastructure_autofix_post_evidence_verified",
            "payload_path": health_root / "infrastructure_autofix_bot_latest.json",
            "cmd": [str(PY), str(ops_root / "infrastructure_autofix_bot.py"), "--apply", "--json"],
            "timeout_sec": 300,
        },
        {
            "name": "system_self_model_post_evidence_verified",
            "payload_path": health_root / "system_self_model_latest.json",
            "cmd": [str(ops_root / "opsctl.sh"), "big-platform-brain", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "master_infrastructure_supervisor_post_evidence_verified",
            "payload_path": health_root / "master_infrastructure_supervisor_latest.json",
            "cmd": [str(PY), str(ops_root / "master_infrastructure_supervisor.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "runtime_gate_dashboard_pre_operator",
            "payload_path": health_root / "runtime_gate_dashboard_latest.json",
            "cmd": [str(PY), str(ops_root / "runtime_gate_dashboard.py"), "--json"],
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
    child_env = os.environ.copy()
    child_env[REFRESH_ACTIVE_ENV] = "1"
    result = run_bounded_process_group(
        list(spec["cmd"]),
        cwd=project_root,
        timeout_seconds=max(int(spec.get("timeout_sec", 120) or 120), 1),
        env=child_env,
    )
    stdout = str(result.get("stdout") or "")
    stderr = str(result.get("stderr") or "")
    payload = _parse_json_output(stdout)
    payload_source = "stdout" if payload else "artifact_fallback"
    if not payload:
        payload = _load_json(payload_path)
    rc = int(result.get("rc", 1) or 0)
    stdout_tail = _tail_text(stdout)
    stderr_tail = _tail_text(stderr or ("timeout" if result.get("timed_out") else ""))
    duration_ms = round((datetime.now(timezone.utc) - started).total_seconds() * 1000.0, 3)
    return {
        "cmd": list(spec["cmd"]),
        "rc": rc,
        "payload": payload,
        "payload_source": payload_source,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "duration_ms": duration_ms,
        "timed_out": bool(result.get("timed_out", False)),
        "timeout_cleanup": result.get("timeout_cleanup") if isinstance(result.get("timeout_cleanup"), dict) else {},
    }


def _run_spec_with_freshness(
    spec: dict[str, Any],
    project_root: Path,
    run_step: RefreshRunner,
) -> dict[str, Any]:
    payload_path = Path(spec["payload_path"]).expanduser()
    additional_payload_paths = [
        Path(path).expanduser()
        for path in spec.get("additional_payload_paths", [])
        if str(path or "").strip()
    ]
    tracked_paths = [payload_path, *additional_payload_paths]
    attempt_rows: list[dict[str, Any]] = []
    published_from_stdout = False
    result: dict[str, Any] = {}
    refreshed_this_cycle = False
    path_freshness: dict[Path, bool] = {path: False for path in tracked_paths}

    for attempt in range(1, 3):
        attempt_started = datetime.now(timezone.utc)
        previous_signatures = {path: _artifact_signature(path) for path in tracked_paths}
        result = run_step(spec, project_root)
        payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
        path_freshness = {
            path: _artifact_refreshed_since(
                path,
                attempt_started,
                previous_signature=previous_signatures[path],
            )
            for path in tracked_paths
        }
        published_this_attempt = False
        if not path_freshness[payload_path] and result.get("payload_source") == "stdout" and payload:
            write_payload(payload_path, payload)
            published_from_stdout = True
            published_this_attempt = True
            path_freshness[payload_path] = _artifact_refreshed_since(
                payload_path,
                attempt_started,
                previous_signature=previous_signatures[payload_path],
            )
        refreshed_this_cycle = all(path_freshness.values())
        attempt_rows.append(
            {
                "attempt": attempt,
                "rc": int(result.get("rc", 1)),
                "payload_source": str(result.get("payload_source") or "runner"),
                "artifact_refreshed": refreshed_this_cycle,
                "artifact_paths_refreshed": {
                    str(path): bool(path_freshness[path]) for path in tracked_paths
                },
                "published_from_stdout": published_this_attempt,
            }
        )
        if refreshed_this_cycle:
            break

    result = dict(result)
    failure_envelope_published = False
    failure_envelope_paths: list[str] = []
    if not refreshed_this_cycle:
        for stale_path in (path for path, fresh in path_freshness.items() if not fresh):
            failure_envelope = {
                "timestamp_utc": iso_now(),
                "schema_version": 1,
                "ok": False,
                "overall_status": "degraded" if bool(spec.get("optional", False)) else "blocked",
                "artifact_refresh_failed": True,
                "stale_source_rejected": True,
                "producer": str(spec.get("name") or ""),
                "producer_rc": int(result.get("rc", 1)),
                "artifact_path": str(stale_path),
                "refresh_attempt_count": len(attempt_rows),
                "recommended_actions": [
                    "inspect the producer stderr and restore current-cycle publication before trusting this artifact"
                ],
            }
            write_payload(stale_path, failure_envelope)
            if _artifact_present(stale_path):
                failure_envelope_paths.append(str(stale_path))
            if stale_path == payload_path:
                result["payload"] = failure_envelope
                result["payload_source"] = "refresh_failure_envelope"
        failure_envelope_published = len(failure_envelope_paths) == sum(
            1 for fresh in path_freshness.values() if not fresh
        )
    result["refresh_attempt_count"] = len(attempt_rows)
    result["refresh_attempts"] = attempt_rows
    result["artifact_refreshed_this_cycle"] = refreshed_this_cycle
    result["artifact_path_freshness"] = {
        str(path): bool(path_freshness[path]) for path in tracked_paths
    }
    result["published_from_stdout"] = published_from_stdout
    result["failure_envelope_published"] = failure_envelope_published
    result["failure_envelope_paths"] = failure_envelope_paths
    return result


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
    if name.startswith("halt_trigger_control_plane"):
        execution_policy = _as_dict(payload.get("execution_policy"))
        manual_flags = _as_dict(payload.get("manual_flags"))
        operator_stop = _as_dict(manual_flags.get("operator_stop"))
        global_halt = _as_dict(manual_flags.get("global_halt"))
        issue_names = {
            str(row.get("name") or "").strip()
            for row in payload.get("issues", [])
            if isinstance(row, dict) and str(row.get("name") or "").strip()
        }
        expected_lock_issues = {
            "paper_trade_lock_active",
            "runtime_release_live_read_only",
            "runtime_clearance_not_thaw_safe",
            "live_runtime_release_read_only",
            "heavy_research_must_stay_cold_lane",
        }
        return bool(
            status in PAPER_SOAK_MANAGED_STATUSES
            and str(payload.get("effective_state") or "").strip().lower() == "live_read_only"
            and bool(execution_policy.get("paper_trade_lock_active", False))
            and not bool(execution_policy.get("effective_live_order_execution_allowed", False))
            and not bool(operator_stop.get("active", False))
            and not bool(global_halt.get("active", False))
            and issue_names
            and issue_names.issubset(expected_lock_issues)
        )
    if name.startswith("coordination_state_control") and status in PAPER_SOAK_MANAGED_STATUSES:
        return True
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
        "artifact_refresh_failed",
        "stale_source_rejected",
        "producer",
        "producer_rc",
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
    cycle_started = datetime.now(timezone.utc)
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
        result = _run_spec_with_freshness(spec, project_root, run_step)
        payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
        present_after = _artifact_present(payload_path)
        refreshed_this_cycle = bool(result.get("artifact_refreshed_this_cycle", False))
        failure_envelope_published = bool(result.get("failure_envelope_published", False))
        producer_artifact_present = bool(present_after and not failure_envelope_published)
        if str(spec["name"]) in missing_before and refreshed_this_cycle:
            recovered += 1
        if not producer_artifact_present:
            missing_after.append(str(spec["name"]))
        steps.append(
            {
                "name": str(spec["name"]),
                "result": result,
                "payload": payload,
                "payload_path": payload_path,
                "present_after": present_after,
                "producer_artifact_present": producer_artifact_present,
                "refreshed_this_cycle": refreshed_this_cycle,
                "optional": bool(spec.get("optional", False)),
            }
        )

    paper_soak_ready_after_refresh = _paper_soak_contract_ready(project_root)
    paper_soak_ready = bool(paper_soak_ready_before_refresh or paper_soak_ready_after_refresh)
    rendered_steps: list[dict[str, Any]] = []
    stale_after_refresh: list[str] = []
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
        refreshed_this_cycle = bool(row.get("refreshed_this_cycle", False))
        if bool(row.get("present_after", False)) and not refreshed_this_cycle:
            stale_after_refresh.append(str(row["name"]))
            if not (optional and status == "managed_paper_soak"):
                status = "optional_advisory" if optional else "stale"
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
                "producer_artifact_present": bool(row.get("producer_artifact_present", False)),
                "artifact_refreshed_this_cycle": refreshed_this_cycle,
                "artifact_path_freshness": dict(result.get("artifact_path_freshness") or {}),
                "refresh_attempt_count": int(result.get("refresh_attempt_count", 1) or 1),
                "published_from_stdout": bool(result.get("published_from_stdout", False)),
                "failure_envelope_published": bool(result.get("failure_envelope_published", False)),
                "payload_summary": _payload_summary(payload),
                "cmd": list(result.get("cmd") or []),
                "stdout_tail": str(result.get("stdout_tail") or ""),
                "stderr_tail": str(result.get("stderr_tail") or ""),
            }
        )

    last_step_index_by_artifact: dict[str, int] = {}
    for index, row in enumerate(rendered_steps):
        last_step_index_by_artifact[str(row["payload_path"])] = index
    for index, row in enumerate(rendered_steps):
        terminal = last_step_index_by_artifact[str(row["payload_path"])] == index
        row["counts_toward_overall"] = terminal
        row["superseded_by_later_verifier"] = not terminal

    effective_steps = [row for row in rendered_steps if bool(row.get("counts_toward_overall", False))]
    statuses = [str(row.get("status") or "") for row in effective_steps]
    missing_after = [str(row["name"]) for row in effective_steps if not bool(row.get("producer_artifact_present", False))]
    stale_after_refresh = [
        str(row["name"])
        for row in effective_steps
        if bool(row.get("artifact_present", False)) and not bool(row.get("artifact_refreshed_this_cycle", False))
    ]

    optional_names = {str(spec["name"]) for spec in refresh_specs if bool(spec.get("optional", False))}
    required_missing_after = [name for name in missing_after if name not in optional_names]
    required_stale_after = [name for name in stale_after_refresh if name not in optional_names]
    error_statuses = {"error", "stale"}
    degraded_statuses = {"warn", "thin", "degraded", "needs_work", "needs_review", "blocked", "busy", "skipped"}
    error_step_count = sum(1 for status in statuses if status in error_statuses)
    degraded_step_count = sum(1 for status in statuses if status in degraded_statuses)
    blocked_step_count = sum(1 for status in statuses if status == "blocked")
    managed_paper_soak_step_count = sum(1 for status in statuses if status == "managed_paper_soak")
    optional_advisory_step_count = sum(1 for status in statuses if status == "optional_advisory")
    overall_status = "ready"
    if error_step_count > 0 or required_missing_after or required_stale_after:
        overall_status = "blocked"
    elif degraded_step_count > 0:
        overall_status = "degraded"

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "refresh_cycle_started_utc": cycle_started.isoformat(),
        "project_root": str(project_root),
        "ok": overall_status != "blocked",
        "overall_status": overall_status,
        "target_refresh_step_count": len(refresh_specs),
        "target_artifact_count": len(effective_steps),
        "artifact_present_count_after": len(effective_steps) - len(missing_after),
        "superseded_step_count": len(refresh_specs) - len(effective_steps),
        "artifacts_recovered_count": recovered,
        "missing_before": missing_before,
        "missing_after": missing_after,
        "required_missing_after": required_missing_after,
        "stale_after_refresh": stale_after_refresh,
        "required_stale_after": required_stale_after,
        "all_required_artifacts_fresh": not required_missing_after and not required_stale_after,
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
                "required stale inputs were retried and cannot be trusted until their producers publish current-cycle evidence" if required_stale_after else "",
                "treat optional proof steps like canary rollout diagnostics as advisory when they time out under live load" if any(name in optional_names for name in missing_after) else "",
                "treat blocked refresh outputs as real runtime issues instead of silent dashboard omissions" if blocked_step_count else "",
                "paper soak is green; proof, promotion, and research debts are tracked as managed_paper_soak without blocking collection" if managed_paper_soak_step_count else "",
            ]
        ),
        "steps": rendered_steps,
    }


def _publish_dashboard(project_root: Path) -> dict[str, Any]:
    dashboard_path = project_root / "governance" / "health" / "runtime_gate_dashboard_latest.json"
    spec = {
        "name": "runtime_gate_dashboard",
        "payload_path": dashboard_path,
        "cmd": [str(PY), str(project_root / "scripts" / "ops" / "runtime_gate_dashboard.py"), "--json"],
        "timeout_sec": 180,
    }
    result = _run_spec_with_freshness(spec, project_root, _run_spec)
    dashboard_payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
    overall = dashboard_payload.get("overall") if isinstance(dashboard_payload.get("overall"), dict) else {}
    return {
        "ok": bool(result.get("artifact_refreshed_this_cycle", False)),
        "status": str(overall.get("status") or dashboard_payload.get("overall_status") or "unknown"),
        "producer_rc": int(result.get("rc", 1)),
        "artifact_path": str(dashboard_path),
        "artifact_present": _artifact_present(dashboard_path),
        "artifact_refreshed_this_cycle": bool(result.get("artifact_refreshed_this_cycle", False)),
        "refresh_attempt_count": int(result.get("refresh_attempt_count", 1) or 1),
        "published_from_stdout": bool(result.get("published_from_stdout", False)),
        "stderr_tail": str(result.get("stderr_tail") or ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh the runtime dashboard's prerequisite artifacts before grading the live system.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if str(os.getenv(REFRESH_ACTIVE_ENV, "")).strip().lower() in {"1", "true", "yes", "on"}:
        payload = {
            "timestamp_utc": iso_now(),
            "schema_version": 1,
            "ok": True,
            "overall_status": "nested_refresh_skipped",
            "nested_refresh_skipped": True,
            "reason": "runtime_artifact_refresh_already_active",
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=True))
        else:
            print("runtime_artifact_refresh overall_status=nested_refresh_skipped")
        return 0

    payload = build_payload(Path(args.project_root).resolve())
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    dashboard_publish = _publish_dashboard(Path(args.project_root).resolve())
    payload["dashboard_publish"] = dashboard_publish
    payload["all_required_artifacts_fresh"] = bool(
        payload.get("all_required_artifacts_fresh", False)
        and dashboard_publish.get("artifact_refreshed_this_cycle", False)
    )
    if not dashboard_publish.get("artifact_refreshed_this_cycle", False):
        payload["ok"] = False
        payload["overall_status"] = "blocked"
        payload["recommended_actions"] = ordered_unique(
            [
                *(payload.get("recommended_actions") or []),
                "runtime dashboard publication remained stale after a bounded retry; inspect its producer stderr before trusting the dashboard",
            ]
        )
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
