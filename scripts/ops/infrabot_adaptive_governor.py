#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, load_json, load_recent_jsonl, ordered_unique, write_payload
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from .long_runtime_common import iso_now, load_json, load_recent_jsonl, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "infrabot_adaptive_governor_latest.json"
SELF_HEALING_STATE_PATH = PROJECT_ROOT / "governance" / "health" / "infrabot_adaptive_self_healing_state.json"
SCHEMA_VERSION = 1
SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
GOOD_STATUSES = {
    "",
    "ready",
    "ok",
    "active",
    "running",
    "watch",
    "watching",
    "advisory",
    "applied",
    "applied_with_followups",
    "guarded",
    "complete",
    "completed",
    "pass",
    "passed",
}
BAD_STATUSES = {
    "blocked",
    "critical",
    "degraded",
    "failed",
    "missing",
    "needs_work",
    "protective_tightening",
    "stale",
    "thin",
    "warn",
    "warning",
    "waiting_for_writer",
    "writer_active",
}
SUCCESSFUL_NON_READY_STATUSES = {
    "protective_tightening",
}
SAFE_EXEC_ENV = {
    "MARKET_DATA_ONLY": "1",
    "ALLOW_ORDER_EXECUTION": "0",
    "TOP_BOT_ENABLE_LIVE_EXECUTION": "0",
    "EXECUTION_LANE_LIVE_ENABLED": "0",
    "RUN_ALL_SLEEVES_WITH_LIVE_EXECUTOR": "0",
    "TOP_BOT_PAPER_TRADING_ENABLED": "1",
    "BOT_PROTECTED_VOLUME_DENYLIST": "/Volumes/VIDEO",
    "INFRA_BOT_ADAPTIVE_SAFE_REPAIR_EXECUTOR": "1",
}


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return raw != 0
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on", "ready", "ok", "active", "armed"}


def _status(raw: Any) -> str:
    return str(raw or "").strip().lower()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_utc(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso_from_dt(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _health(project_root: Path, name: str) -> dict[str, Any]:
    return _as_dict(load_json(project_root / "governance" / "health" / name))


def _opsctl(*args: str) -> list[str]:
    return ["./scripts/ops/opsctl.sh", *args]


def _capability(
    *,
    capability_id: str,
    title: str,
    owns: list[str],
    command: list[str],
    risk_level: str = "low",
    cost_class: str = "low",
    apply_safe: bool = False,
    advisory_only: bool = False,
    safe_under_pressure: bool = False,
    requires_single_writer_idle: bool = False,
    blocks_live_execution: bool = True,
    success_artifact: str = "",
    cadence: str = "on_degradation",
) -> dict[str, Any]:
    return {
        "id": capability_id,
        "title": title,
        "owns": list(owns),
        "command": list(command),
        "risk_level": risk_level,
        "cost_class": cost_class,
        "apply_safe": bool(apply_safe),
        "advisory_only": bool(advisory_only),
        "safe_under_pressure": bool(safe_under_pressure),
        "requires_single_writer_idle": bool(requires_single_writer_idle),
        "blocks_live_execution": bool(blocks_live_execution),
        "success_artifact": success_artifact,
        "cadence": cadence,
        "authority_boundary": "advisory_and_safe_repair_only_no_live_execution_authority",
    }


def _capability_registry() -> list[dict[str, Any]]:
    return [
        _capability(
            capability_id="system_needs_intelligence",
            title="System Needs Contract",
            owns=["system_needs_contract", "low_grade_layer_audit", "operator_next_steps"],
            command=_opsctl("system-needs", "--json"),
            advisory_only=True,
            success_artifact="governance/health/system_needs_intelligence_latest.json",
        ),
        _capability(
            capability_id="pressure_relief_control",
            title="Pressure Relief Control",
            owns=["host_saturation", "support_job_cooldown", "heavy_feed_ttl"],
            command=_opsctl("pressure-relief", "--apply", "--json"),
            apply_safe=True,
            safe_under_pressure=True,
            success_artifact="governance/health/pressure_relief_control_latest.json",
        ),
        _capability(
            capability_id="runtime_throttle_control",
            title="Runtime Throttle Control",
            owns=["process_fanout", "renice", "runtime_smoothing"],
            command=_opsctl("runtime-throttle", "--apply", "--json"),
            apply_safe=True,
            safe_under_pressure=True,
            success_artifact="governance/health/runtime_throttle_control_latest.json",
        ),
        _capability(
            capability_id="sql_writer_fluidity_governor",
            title="SQL Writer Fluidity Governor",
            owns=["sql_writer_fluidity", "single_writer_cpu", "mac_fluidity"],
            command=_opsctl("runtime-throttle", "--apply", "--max-renice-processes", "8", "--json"),
            apply_safe=True,
            safe_under_pressure=True,
            success_artifact="governance/health/runtime_throttle_control_latest.json",
        ),
        _capability(
            capability_id="memory_pressure_intelligence",
            title="Memory Pressure Intelligence",
            owns=["memory_headroom", "swap_pressure", "memory_clear_samples"],
            command=_opsctl("memory-pressure-intelligence", "--apply", "--json"),
            apply_safe=True,
            safe_under_pressure=True,
            success_artifact="governance/health/memory_pressure_intelligence_latest.json",
        ),
        _capability(
            capability_id="broker_auth_supervisor",
            title="Broker Auth Supervisor",
            owns=["broker_auth_contract", "auth_lease", "schwab_token_refresh"],
            command=_opsctl("schwab-auth-supervisor", "--apply", "--json"),
            apply_safe=True,
            safe_under_pressure=True,
            success_artifact="governance/health/schwab_auth_supervisor_latest.json",
        ),
        _capability(
            capability_id="global_halt_refresh",
            title="Global Halt Refresh",
            owns=["global_clear_blocker", "halt_clear_readiness", "paper_ramp_clear_state"],
            command=_opsctl("global-halt-refresh", "--json"),
            apply_safe=True,
            safe_under_pressure=True,
            success_artifact="governance/health/global_killswitch_latest.json",
        ),
        _capability(
            capability_id="paper_ramp_guard",
            title="Paper Ramp Guard",
            owns=["guarded_paper_ramp", "paper_trade_eligibility", "paper_lane_rearm"],
            command=_opsctl("paper-400-ramp", "--apply", "--json"),
            apply_safe=True,
            safe_under_pressure=True,
            success_artifact="governance/health/paper_400_ramp_latest.json",
        ),
        _capability(
            capability_id="live_canary_readiness_contract",
            title="Live Canary Readiness Contract",
            owns=["live_canary_readiness_bar", "production_hardening_soak", "live_money_blockers"],
            command=_opsctl("live-canary-readiness", "--apply", "--json"),
            advisory_only=True,
            safe_under_pressure=True,
            success_artifact="governance/health/live_canary_readiness_contract_latest.json",
        ),
        _capability(
            capability_id="production_quality_control",
            title="Production Quality Control",
            owns=["production_quality_bar", "safe_repair_ordering", "live_canary_blocker_remediation"],
            command=_opsctl("production-quality", "--apply", "--refresh-contract", "--json"),
            apply_safe=True,
            safe_under_pressure=True,
            success_artifact="governance/health/production_quality_control_latest.json",
        ),
        _capability(
            capability_id="production_quality_slo_guard",
            title="Production Quality SLO Guard",
            owns=["production_quality_slo", "recurring_degradation_memory", "bounded_repair_escalation"],
            command=_opsctl("production-quality-slo", "--apply", "--refresh-quality", "--json"),
            apply_safe=True,
            safe_under_pressure=True,
            success_artifact="governance/health/production_quality_slo_guard_latest.json",
        ),
        _capability(
            capability_id="source_mutation_guard",
            title="Source Mutation Guard",
            owns=["runtime_source_mutation_guard", "protected_source_dirty", "canonical_source_write_contract"],
            command=_opsctl("source-mutation-guard", "--check-clean", "--json"),
            advisory_only=True,
            safe_under_pressure=True,
            success_artifact="governance/health/live_canary_readiness_contract_latest.json",
        ),
        _capability(
            capability_id="production_flow_smoke",
            title="Production Flow Smoke",
            owns=["ci_production_guardrails", "production_flow_contract", "showcase_artifact_flow"],
            command=_opsctl("production-flow-smoke", "--json"),
            advisory_only=True,
            safe_under_pressure=True,
            success_artifact="governance/health/live_canary_readiness_contract_latest.json",
        ),
        _capability(
            capability_id="writer_cycle_coordinator",
            title="Writer Cycle Coordinator",
            owns=["writer_handoff", "single_writer_lock", "drainer_progress"],
            command=_opsctl("writer-cycle-coordinator", "--apply", "--handoff-only", "--json"),
            apply_safe=True,
            safe_under_pressure=True,
            success_artifact="governance/health/writer_cycle_coordinator_latest.json",
        ),
        _capability(
            capability_id="storage_backpressure_autopilot",
            title="Storage Backpressure Autopilot",
            owns=["storage_backpressure", "bounded_backlog_draining", "retention_debt"],
            command=_opsctl("storage-backpressure-autopilot", "--apply", "--quick-bounded", "--json"),
            risk_level="medium",
            cost_class="medium",
            apply_safe=True,
            requires_single_writer_idle=True,
            success_artifact="governance/health/storage_backpressure_autopilot_latest.json",
        ),
        _capability(
            capability_id="external_backlog_drain_handoff",
            title="External Backlog Drain Handoff",
            owns=["cleanup_handoff_ingestion", "external_backlog_drain", "writer_safe_drain_request"],
            command=_opsctl(
                "external-backlog-drain",
                "--apply",
                "--follow-through",
                "--poll-seconds",
                "5",
                "--wait-timeout-seconds",
                "45",
                "--json",
            ),
            risk_level="medium",
            cost_class="medium",
            apply_safe=True,
            safe_under_pressure=True,
            success_artifact="governance/health/external_backlog_drain_latest.json",
        ),
        _capability(
            capability_id="raw_training_cleanup_handoff",
            title="Raw Training Cleanup Handoff",
            owns=["cleanup_handoff_ingestion", "raw_training_compaction", "training_source_manifest"],
            command=_opsctl("raw-training-compaction", "--apply", "--max-files", "12", "--max-gb", "8.0", "--json"),
            risk_level="medium",
            cost_class="medium",
            apply_safe=True,
            requires_single_writer_idle=True,
            success_artifact="governance/health/raw_training_compaction_intelligence_latest.json",
        ),
        _capability(
            capability_id="storage_retention_unison_handoff",
            title="Storage Retention Unison Handoff",
            owns=["cleanup_handoff_ingestion", "hot_plane_compaction", "retention_unison", "storage_quota_guard", "stateful_sql_compaction"],
            command=_opsctl(
                "storage-retention-unison",
                "--apply",
                "--raw-max-files",
                "8",
                "--raw-max-gb",
                "8.0",
                "--telemetry-max-gb",
                "16.0",
                "--lifecycle-max-gb",
                "4.0",
                "--decision-max-gb",
                "8.0",
                "--cleanup-max-delete-gb",
                "16.0",
                "--target-free-gb",
                "125.0",
                "--soak-days",
                "30",
                "--json",
            ),
            risk_level="medium",
            cost_class="medium",
            apply_safe=True,
            requires_single_writer_idle=True,
            success_artifact="governance/health/storage_retention_unison_latest.json",
        ),
        _capability(
            capability_id="deep_cold_second_cold_handoff",
            title="Deep Cold Second-Cold Handoff",
            owns=["deep_cold_archive", "second_cold_archive", "stale_stage_pressure_relief", "storage_quota_guard"],
            command=_opsctl(
                "deep-cold-storage-layer",
                "--apply",
                "--move-to-second-cold",
                "--adaptive",
                "--second-cold-root",
                "/Volumes/VIDEO/schwab_trading_bot_cold",
                "--max-move-gb",
                "96",
                "--max-move-files",
                "500",
                "--json",
            ),
            risk_level="medium",
            cost_class="high",
            apply_safe=True,
            safe_under_pressure=True,
            success_artifact="governance/health/deep_cold_storage_layer_latest.json",
        ),
        _capability(
            capability_id="stateful_sql_quota_relief",
            title="Stateful SQL Quota Relief",
            owns=["storage_quota_guard", "stateful_sql_hot_retention", "sql_link_shards", "sqlite_compaction"],
            command=_opsctl("storage-maintenance", "--force", "--json"),
            risk_level="medium",
            cost_class="high",
            apply_safe=True,
            requires_single_writer_idle=True,
            success_artifact="governance/health/storage_maintenance_latest.json",
        ),
        _capability(
            capability_id="livefeed_refresh_guard",
            title="Livefeed Refresh Guard",
            owns=["livefeed_continuity", "terminal_feed_health", "heavy_feed_visibility"],
            command=_opsctl("livefeed-refresh-guard", "--apply", "--json"),
            apply_safe=True,
            safe_under_pressure=True,
            success_artifact="governance/health/livefeed_refresh_guard_latest.json",
        ),
        _capability(
            capability_id="commands_hygiene",
            title="Commands Hygiene",
            owns=["commands_md", "operator_runbook", "command_contract"],
            command=_opsctl("commands-hygiene", "--apply", "--json"),
            apply_safe=True,
            success_artifact="governance/health/commands_hygiene_latest.json",
        ),
        _capability(
            capability_id="command_validity",
            title="Command Validity",
            owns=["opsctl_routes", "command_parseability", "operator_surface_verification"],
            command=_opsctl("command-validity", "--json"),
            advisory_only=True,
            success_artifact="governance/health/command_validity_latest.json",
        ),
        _capability(
            capability_id="daily_verify_auto_remediation",
            title="Daily Verify Auto Remediation",
            owns=["daily_verify_auto_remediation", "failed_check_repair", "verification_refresh"],
            command=_opsctl("daily-verify-remediation", "--apply", "--json"),
            risk_level="medium",
            cost_class="medium",
            apply_safe=True,
            success_artifact="governance/health/daily_verify_auto_remediation_latest.json",
        ),
        _capability(
            capability_id="stateful_storage_regression_guard",
            title="Stateful Storage Regression Guard",
            owns=["stateful_storage_regression", "local_stateful_cleanup", "storage_route_contract"],
            command=_opsctl("stateful-storage-regression-guard", "--apply", "--json"),
            risk_level="medium",
            cost_class="medium",
            apply_safe=True,
            requires_single_writer_idle=True,
            success_artifact="governance/health/stateful_storage_regression_guard_latest.json",
        ),
        _capability(
            capability_id="system_drift_autopilot",
            title="System Drift Autopilot",
            owns=["system_drift", "registry_drift", "safe_drift_repairs"],
            command=_opsctl("system-drift-autopilot", "--apply", "--max-steps", "3", "--json"),
            risk_level="medium",
            cost_class="medium",
            apply_safe=True,
            success_artifact="governance/health/system_drift_autopilot_latest.json",
        ),
        _capability(
            capability_id="runtime_snapshot_refresh",
            title="Runtime Snapshot Refresh",
            owns=["runtime_snapshot_cache", "training_prep_snapshot", "prep_only_training_inputs"],
            command=_opsctl(
                "runtime-training-snapshot",
                "--reuse-if-fresh-minutes",
                "360",
                "--light-refresh-existing",
                "--json",
            ),
            risk_level="medium",
            cost_class="medium",
            apply_safe=True,
            success_artifact="governance/health/runtime_snapshot_cache_control_latest.json",
        ),
        _capability(
            capability_id="restart_blackstart_refresh",
            title="Restart And Blackstart Refresh",
            owns=["restart_sanity", "blackstart_recovery", "read_only_recovery_evidence"],
            command=_opsctl("restart-sanity", "--json"),
            cost_class="low",
            apply_safe=True,
            safe_under_pressure=True,
            success_artifact="governance/health/restart_sanity_bundle_latest.json",
        ),
        _capability(
            capability_id="blackstart_recovery_refresh",
            title="Blackstart Recovery Refresh",
            owns=["blackstart_recovery", "reboot_resilience", "read_only_recovery_evidence"],
            command=_opsctl("blackstart-recovery", "--json"),
            cost_class="low",
            apply_safe=True,
            safe_under_pressure=True,
            success_artifact="governance/health/blackstart_recovery_latest.json",
        ),
        _capability(
            capability_id="infrabot_gap_roster",
            title="Infrabot Gap Roster",
            owns=["unassigned_degradation", "missing_infrabot_coverage", "delegation_map"],
            command=_opsctl("infrabot-gap-roster", "--json"),
            advisory_only=True,
            success_artifact="governance/health/infrabot_gap_roster_latest.json",
        ),
        _capability(
            capability_id="infrabot_gap_safe_delegation",
            title="Infrabot Gap Safe Delegation",
            owns=["low_risk_gap_delegation", "assigned_infrabot_repairs", "gap_roster_safe_apply"],
            command=_opsctl("infrabot-gap-roster", "--apply", "--safe-apply", "--timeout-sec", "120", "--json"),
            risk_level="medium",
            cost_class="medium",
            apply_safe=True,
            success_artifact="governance/health/infrabot_gap_roster_latest.json",
        ),
        _capability(
            capability_id="infrastructure_autofix",
            title="Infrastructure Autofix",
            owns=["safe_infra_repairs", "repair_plan_consolidation", "cross_surface_remediation"],
            command=_opsctl("infrastructure-autofix", "--apply", "--timeout-sec", "180", "--json"),
            risk_level="medium",
            cost_class="medium",
            apply_safe=True,
            success_artifact="governance/health/infrastructure_autofix_bot_latest.json",
        ),
        _capability(
            capability_id="master_infra_supervisor",
            title="Master Infrastructure Supervisor",
            owns=["infra_dependency_graph", "child_bot_outcomes", "system_wide_infra_state"],
            command=_opsctl("master-infra-supervisor", "--apply", "--timeout-sec", "180", "--json"),
            risk_level="medium",
            cost_class="medium",
            apply_safe=True,
            success_artifact="governance/health/master_infrastructure_supervisor_latest.json",
        ),
        _capability(
            capability_id="training_runtime_control",
            title="Training Runtime Control",
            owns=["training_gate", "batch_size_contract", "runtime_training_headroom"],
            command=_opsctl("training-runtime-control", "--json"),
            advisory_only=True,
            safe_under_pressure=True,
            success_artifact="governance/health/training_runtime_control_latest.json",
        ),
        _capability(
            capability_id="training_prep_autopilot",
            title="Training Prep Autopilot",
            owns=["training_gate", "prep_only_remediation", "runtime_snapshot", "training_backlog_readiness"],
            command=_opsctl(
                "training-drain-autopilot",
                "--apply",
                "--prep-only",
                "--limit",
                "4",
                "--max-cycles",
                "1",
                "--command-timeout-seconds",
                "420",
                "--json",
            ),
            risk_level="medium",
            cost_class="medium",
            apply_safe=True,
            requires_single_writer_idle=True,
            success_artifact="governance/health/training_drain_autopilot_latest.json",
        ),
        _capability(
            capability_id="training_data_intake_labeling",
            title="Training Data Intake Labeling",
            owns=["bot_data_labeling", "collect_first_queue", "paper_loss_hard_negative_ingress"],
            command=_opsctl("training-data-intake", "--apply", "--focus-limit", "160", "--json"),
            risk_level="medium",
            cost_class="medium",
            apply_safe=True,
            success_artifact="governance/health/training_data_intake_expansion_latest.json",
        ),
        _capability(
            capability_id="training_labeling_intelligence",
            title="Training Labeling Intelligence",
            owns=["label_contracts", "hard_negative_labels", "point_in_time_labeling"],
            command=_opsctl("training-labeling-intelligence", "--apply", "--json"),
            risk_level="medium",
            cost_class="medium",
            apply_safe=True,
            success_artifact="governance/health/training_labeling_intelligence_latest.json",
        ),
        _capability(
            capability_id="bot_quality_autopilot",
            title="Bot Quality Autopilot",
            owns=["bot_quality", "duplicate_alpha", "mentor_repair_plan"],
            command=_opsctl("bot-quality-autopilot", "--apply", "--json"),
            risk_level="medium",
            cost_class="medium",
            apply_safe=True,
            success_artifact="governance/health/bot_quality_autopilot_latest.json",
        ),
        _capability(
            capability_id="paper_execution_truth_layer",
            title="Paper Execution Truth Layer",
            owns=["paper_truth_watch", "counterfactual_attribution", "broker_truth_reconciliation"],
            command=_opsctl("paper-truth", "--json"),
            advisory_only=True,
            safe_under_pressure=True,
            success_artifact="governance/health/paper_execution_truth_layer_latest.json",
        ),
        _capability(
            capability_id="paper_profitability_control",
            title="Paper Profitability Control",
            owns=["paper_trade_feedback", "profitability_grade", "profile_quarantine"],
            command=_opsctl("paper-profitability-control", "--apply", "--json"),
            risk_level="medium",
            cost_class="medium",
            apply_safe=True,
            safe_under_pressure=True,
            success_artifact="governance/health/paper_profitability_control_latest.json",
        ),
        _capability(
            capability_id="runtime_paper_regression_guard",
            title="Runtime Paper Regression Guard",
            owns=["runtime_paper_contract", "paper_soak_continuity", "paper_runtime_artifact_freshness"],
            command=_opsctl("runtime-paper-regression-guard", "--json"),
            advisory_only=True,
            safe_under_pressure=True,
            success_artifact="governance/health/runtime_paper_regression_guard_latest.json",
        ),
        _capability(
            capability_id="promotion_quality_gate",
            title="Promotion Quality Gate",
            owns=["promotion_gate_freshness", "promotion_quality_contract", "unknown_blocks_promotion"],
            command=_opsctl("promotion-quality-gate", "--json"),
            advisory_only=True,
            safe_under_pressure=True,
            success_artifact="governance/health/promotion_quality_gate_latest.json",
        ),
        _capability(
            capability_id="paper_performance_refresh",
            title="Paper Performance Refresh",
            owns=["paper_performance", "paper_feedback_freshness", "sleeve_profitability_inputs"],
            command=_opsctl("paper-performance", "--json"),
            cost_class="low",
            apply_safe=True,
            safe_under_pressure=True,
            success_artifact="governance/health/paper_performance_latest.json",
        ),
        _capability(
            capability_id="master_grandmaster_profitability_trainer",
            title="Master Grandmaster Profitability Trainer",
            owns=["raw_profitability_training_feedback", "master_profit_calibration", "paper_loss_hard_negative_labels"],
            command=_opsctl("master-grandmaster-train", "--apply", "--json"),
            risk_level="medium",
            cost_class="medium",
            apply_safe=True,
            success_artifact="governance/health/master_grandmaster_profitability_training_latest.json",
        ),
        _capability(
            capability_id="source_verification",
            title="Source Verification",
            owns=["unverified_sources", "stale_sources", "provider_mesh_quality"],
            command=_opsctl("source-verification", "--json"),
            advisory_only=True,
            success_artifact="governance/health/source_verification_latest.json",
        ),
        _capability(
            capability_id="source_verification_autorefresh",
            title="Source Verification Autorefresh",
            owns=["stale_sources", "provider_mesh_refresh", "source_quality_repair", "collector_contract_reconciliation", "health_gate_reconciliation"],
            command=_opsctl("source-verification-refresh", "--apply", "--json"),
            risk_level="medium",
            cost_class="medium",
            apply_safe=True,
            success_artifact="governance/health/source_verification_autorefresh_latest.json",
        ),
        _capability(
            capability_id="health_gates_recheck",
            title="Health Gates Recheck",
            owns=["hard_gate_truth", "dashboard_gate_freshness", "post_repair_reconciliation"],
            command=_opsctl("health-gates", "--json"),
            advisory_only=True,
            safe_under_pressure=True,
            success_artifact="governance/health/health_gates_latest.json",
        ),
        _capability(
            capability_id="provider_mesh_refresh",
            title="Provider Mesh Refresh",
            owns=["provider_mesh_quality", "provider_cooldowns", "source_cross_verification"],
            command=_opsctl("provider-mesh", "--json"),
            cost_class="low",
            apply_safe=True,
            safe_under_pressure=True,
            success_artifact="governance/health/provider_mesh_latest.json",
        ),
        _capability(
            capability_id="market_explanation_evidence",
            title="Market Explanation Evidence",
            owns=["market_move_explainer", "symbol_evidence_backfill", "explanation_auditability"],
            command=_opsctl("decision-intelligence", "--json"),
            cost_class="low",
            apply_safe=True,
            safe_under_pressure=True,
            success_artifact="governance/health/market_move_explainer_latest.json",
        ),
    ]


def _need(
    *,
    need_id: str,
    title: str,
    category: str,
    severity: str,
    evidence: list[str],
    target_capabilities: list[str],
    stop_when: str,
    expected_impact: str,
) -> dict[str, Any]:
    return {
        "id": need_id,
        "title": title,
        "category": category,
        "severity": severity,
        "evidence": ordered_unique(evidence),
        "target_capabilities": ordered_unique(target_capabilities),
        "stop_when": stop_when,
        "expected_impact": expected_impact,
    }


def _profitability_grade_below_a(value: Any) -> bool:
    grade = str(value or "").strip().upper()
    return bool(grade and grade not in {"A", "A+"})


def _nested_dict(payload: dict[str, Any], *path: str) -> dict[str, Any]:
    current: Any = payload
    for key in path:
        current = _as_dict(current).get(key)
    return _as_dict(current)


def _first_nested_dict(*values: dict[str, Any]) -> dict[str, Any]:
    for value in values:
        if value:
            return value
    return {}


def _loss_cause_names(*contracts: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for contract in contracts:
        for row in _as_list(contract.get("top_loss_causes")):
            if isinstance(row, dict):
                cause = str(row.get("cause") or "").strip()
            else:
                cause = str(row or "").strip()
            if cause and cause not in names:
                names.append(cause)
    return names


def _raw_profitability_recovery_context(
    *,
    paper_profitability: dict[str, Any],
    paper_runtime_profitability: dict[str, Any],
    live_canary_readiness: dict[str, Any],
) -> dict[str, Any]:
    source = paper_profitability if paper_profitability else paper_runtime_profitability
    runtime_source = paper_runtime_profitability if paper_runtime_profitability else paper_profitability
    raw_grade = str(
        source.get("raw_profitability_grade")
        or runtime_source.get("raw_profitability_grade")
        or ""
    ).strip().upper()
    controlled_grade = str(
        source.get("controlled_profitability_grade")
        or runtime_source.get("controlled_profitability_grade")
        or source.get("profitability_grade")
        or runtime_source.get("profitability_grade")
        or ""
    ).strip().upper()
    financial_grade = str(
        source.get("financial_profitability_grade")
        or runtime_source.get("financial_profitability_grade")
        or source.get("financial_display_grade")
        or runtime_source.get("financial_display_grade")
        or ""
    ).strip().upper()
    a_plus = _first_nested_dict(
        _nested_dict(paper_runtime_profitability, "a_plus_target_contract"),
        _nested_dict(paper_profitability, "a_plus_target_contract"),
    )
    current = _as_dict(a_plus.get("current"))
    thresholds = _as_dict(a_plus.get("thresholds"))
    raw_improvement = _first_nested_dict(
        _nested_dict(paper_runtime_profitability, "raw_profitability_improvement_contract"),
        _nested_dict(paper_profitability, "raw_profitability_improvement_contract"),
    )
    raw_a_recovery = _first_nested_dict(
        _nested_dict(paper_runtime_profitability, "raw_profitability_a_recovery_contract"),
        _nested_dict(paper_profitability, "raw_profitability_a_recovery_contract"),
    )
    raw_six = _first_nested_dict(
        _nested_dict(paper_runtime_profitability, "raw_profitability_six_point_recovery_contract"),
        _nested_dict(paper_profitability, "raw_profitability_six_point_recovery_contract"),
    )
    burn_down = _first_nested_dict(
        _nested_dict(raw_improvement, "burn_down_contract"),
        _nested_dict(raw_a_recovery, "burn_down_contract"),
        _nested_dict(raw_six, "burn_down_contract"),
    )
    loss_feedback = _first_nested_dict(
        _nested_dict(raw_improvement, "loss_cause_training_feedback_contract"),
        _nested_dict(raw_six, "loss_cause_filter_contract"),
        raw_a_recovery,
    )
    requirements = _as_list(raw_improvement.get("requirements"))
    live_blockers = [
        str(item or "").strip()
        for item in _as_list(live_canary_readiness.get("blockers"))
        if str(item or "").strip()
    ]
    raw_live_blockers = [item for item in live_blockers if "raw_profitability" in item]
    net_pnl = _safe_float(current.get("net_pnl"), _safe_float(burn_down.get("current_net_pnl"), 0.0))
    combined_ready = bool(a_plus.get("combined_a_plus_ready", False)) if a_plus else False
    ready_requirement_count = sum(1 for row in requirements if isinstance(row, dict) and bool(row.get("ready", False)))
    active = bool(
        raw_grade
        and (
            _profitability_grade_below_a(raw_grade)
            or net_pnl < 0.0
            or _profitability_grade_below_a(financial_grade)
            or raw_live_blockers
            or (bool(a_plus) and not combined_ready)
        )
    )
    return {
        "active": bool(active),
        "raw_profitability_grade": raw_grade,
        "controlled_profitability_grade": controlled_grade,
        "financial_profitability_grade": financial_grade,
        "net_pnl": net_pnl,
        "realized_pnl": _safe_float(current.get("realized_pnl"), 0.0),
        "unrealized_pnl": _safe_float(current.get("unrealized_pnl"), 0.0),
        "change_vs_previous_day": _safe_float(current.get("change_vs_previous_day"), 0.0),
        "executions": _safe_int(current.get("executions"), 0),
        "weak_profile_count": _safe_int(current.get("weak_profile_count"), 0),
        "strategy_control_count": _safe_int(current.get("strategy_control_count"), 0),
        "unprotected_weak_profile_count": _safe_int(current.get("unprotected_weak_profile_count"), 0),
        "unprotected_strategy_control_count": _safe_int(current.get("unprotected_strategy_control_count"), 0),
        "min_net_pnl": _safe_float(thresholds.get("min_net_pnl"), 0.0),
        "combined_a_plus_ready": combined_ready,
        "daily_net_improvement_target": max(
            _safe_float(burn_down.get("required_average_daily_net_improvement"), 0.0),
            _safe_float(_as_dict(raw_improvement.get("runtime_enforcement")).get("raw_d_daily_net_improvement_target"), 0.0),
        ),
        "top_loss_causes": _loss_cause_names(loss_feedback, raw_a_recovery),
        "requirement_count": len(requirements),
        "ready_requirement_count": ready_requirement_count,
        "top_drag_profiles": _as_list(burn_down.get("top_drag_profiles"))[:5],
        "live_canary_raw_blockers": raw_live_blockers,
    }


def _needs_contract(project_root: Path, *, refresh_needs: bool = False) -> dict[str, Any]:
    health = project_root / "governance" / "health"
    if refresh_needs:
        try:
            from scripts.ops import system_needs_intelligence

            refreshed = system_needs_intelligence.build_payload(project_root)
            if isinstance(refreshed, dict) and refreshed:
                write_payload(health / "system_needs_intelligence_latest.json", refreshed)
        except Exception:
            pass

    system_needs = _health(project_root, "system_needs_intelligence_latest.json")
    health_fast = _health(project_root, "health_fast_latest.json")
    schwab_auth = _health(project_root, "schwab_auth_supervisor_latest.json")
    auth_lease = _health(project_root, "auth_lease_manager_latest.json")
    premarket_guard = _health(project_root, "premarket_token_guard_latest.json")
    broker_readiness = _health(project_root, "broker_readiness_latest.json")
    global_halt = _health(project_root, "global_killswitch_latest.json")
    paper_ramp = _health(project_root, "paper_400_ramp_latest.json")
    pressure = _health(project_root, "pressure_relief_control_latest.json")
    runtime_throttle = _health(project_root, "runtime_throttle_control_latest.json")
    memory = _health(project_root, "memory_pressure_intelligence_latest.json")
    storage = _health(project_root, "ingestion_storage_control_latest.json")
    storage_auto = _health(project_root, "storage_backpressure_autopilot_latest.json")
    external_backlog = _health(project_root, "external_backlog_drain_latest.json")
    raw_training = _health(project_root, "raw_training_compaction_intelligence_latest.json")
    storage_retention = _health(project_root, "storage_retention_unison_latest.json")
    storage_quota = _health(project_root, "storage_quota_guard_latest.json")
    writer = _health(project_root, "writer_cycle_coordinator_latest.json")
    training = _health(project_root, "training_runtime_control_latest.json")
    livefeed = _health(project_root, "livefeed_local_latest.json")
    livefeed_refresh_guard = _health(project_root, "livefeed_refresh_guard_latest.json")
    heavy_view = _health(project_root, "live_feed_heavy_view_latest.json")
    commands = _health(project_root, "commands_hygiene_latest.json")
    command_validity = _health(project_root, "command_validity_latest.json")
    infra = _health(project_root, "infrastructure_autofix_bot_latest.json")
    gap_roster = _health(project_root, "infrabot_gap_roster_latest.json")
    master = _health(project_root, "master_infrastructure_supervisor_latest.json")
    bot_quality = _health(project_root, "bot_quality_autopilot_latest.json")
    paper_profitability = _health(project_root, "paper_profitability_control_latest.json")
    paper_runtime_profitability = _health(project_root, "paper_runtime_profitability_controls_latest.json")
    paper_truth = _health(project_root, "paper_execution_truth_layer_latest.json")
    paper_runtime = _health(project_root, "runtime_paper_regression_guard_latest.json")
    paper_backlog = _health(project_root, "paper_execution_backlog_relief_latest.json")
    live_canary_readiness = _health(project_root, "live_canary_readiness_contract_latest.json")
    production_quality = _health(project_root, "production_quality_control_latest.json")
    production_quality_slo = _health(project_root, "production_quality_slo_guard_latest.json")
    source_verification = _health(project_root, "source_verification_latest.json")
    provider_mesh = _health(project_root, "provider_mesh_latest.json")
    market_explainer = _health(project_root, "market_move_explainer_latest.json")
    training_labeling = _health(project_root, "training_labeling_intelligence_latest.json")

    needs: list[dict[str, Any]] = []

    lease_budget = _as_dict(auth_lease.get("lease_budget"))
    auth_status = _status(schwab_auth.get("overall_status"))
    lease_status = _status(auth_lease.get("overall_status"))
    lease_state = _status(auth_lease.get("lease_state"))
    token = _as_dict(schwab_auth.get("token"))
    expires_in = max(
        _safe_float(auth_lease.get("expires_in_seconds"), 0.0),
        _safe_float(lease_budget.get("expires_in_seconds"), 0.0),
        _safe_float(token.get("expires_in_seconds"), 0.0),
        _safe_float(broker_readiness.get("token_expires_in_seconds"), 0.0),
    )
    broker_ready = bool(broker_readiness.get("ready_for_open", premarket_guard.get("ok", False)))
    broker_auth_ok = bool(broker_readiness.get("auth_ok", broker_ready))
    broker_network_ok = bool(broker_readiness.get("network_ok", premarket_guard.get("network_ok", True)))
    halt_clear = _as_dict(global_halt.get("clear_state"))
    clear_blockers = ordered_unique(
        [str(item or "").strip() for item in _as_list(halt_clear.get("blockers") or global_halt.get("clear_blockers"))]
    )
    ramp_status = _status(paper_ramp.get("overall_status") or paper_ramp.get("status") or paper_ramp.get("stage"))
    ramp_blockers = ordered_unique(
        [str(item or "").strip() for item in _as_list(paper_ramp.get("blockers")) if str(item or "").strip()]
    )
    auth_issue = bool(
        auth_status in BAD_STATUSES
        or lease_status in BAD_STATUSES
        or lease_state in {"critical", "blocked", "expired", "warning"}
        or (expires_in > 0 and expires_in < 1500.0)
        or not broker_ready
        or not broker_auth_ok
        or not broker_network_ok
        or "auth_lease_critical" in set(clear_blockers)
        or "global_clear_blocker=auth_lease_critical" in set(ramp_blockers)
    )
    if auth_issue:
        needs.append(
            _need(
                need_id="broker_auth_continuity",
                title="Broker auth lease needs preemptive self-healing",
                category="broker_auth",
                severity=(
                    "critical"
                    if (
                        auth_status == "blocked"
                        or lease_status == "blocked"
                        or lease_state == "critical"
                        or not broker_ready
                        or not broker_auth_ok
                        or (expires_in > 0 and expires_in < 600.0)
                    )
                    else "high"
                ),
                evidence=[
                    f"schwab_auth_status={auth_status or 'unknown'}",
                    f"auth_lease_status={lease_status or 'unknown'}",
                    f"auth_lease_state={lease_state or 'unknown'}",
                    f"expires_in_seconds={expires_in:.1f}",
                    f"broker_ready={broker_ready}",
                    f"broker_auth_ok={broker_auth_ok}",
                    f"broker_network_ok={broker_network_ok}",
                    f"global_clear_blockers={','.join(clear_blockers[:6]) or 'none'}",
                    f"paper_ramp_status={ramp_status or 'unknown'}",
                    f"paper_ramp_blockers={','.join(ramp_blockers[:6]) or 'none'}",
                ],
                target_capabilities=[
                    "broker_auth_supervisor",
                    "global_halt_refresh",
                    "paper_ramp_guard",
                    "runtime_paper_regression_guard",
                ],
                stop_when="auth lease is healthy above the proactive floor, broker readiness is true, global clear blockers are empty, and paper ramp is armed.",
                expected_impact="Refreshes Schwab auth and re-arms guarded paper before token drift pauses paper trading during the soak.",
            )
        )

    live_canary_status = _status(live_canary_readiness.get("overall_status"))
    live_canary_ready = bool(live_canary_readiness.get("live_canary_money_ready", False))
    live_canary_blockers = [
        str(item)
        for item in _as_list(live_canary_readiness.get("blockers"))
        if str(item or "").strip()
    ]
    if live_canary_readiness and (live_canary_status in BAD_STATUSES or not live_canary_ready or live_canary_blockers):
        needs.append(
            _need(
                need_id="live_canary_readiness_bar",
                title="Live canary money remains blocked by production-hardening bar",
                category="live_canary",
                severity="critical" if live_canary_status == "blocked" else "high",
                evidence=[
                    f"live_canary_readiness_status={live_canary_status or 'unknown'}",
                    f"live_canary_money_ready={live_canary_ready}",
                    f"ready_gate_count={live_canary_readiness.get('ready_gate_count', 0)}",
                    f"gate_count={live_canary_readiness.get('gate_count', 0)}",
                    f"blockers={','.join(live_canary_blockers[:8]) or 'none'}",
                    str(live_canary_readiness.get("infrastructure_message") or ""),
                ],
                target_capabilities=[
                    "live_canary_readiness_contract",
                    "production_quality_control",
                    "production_quality_slo_guard",
                    "paper_profitability_control",
                    "paper_performance_refresh",
                    "paper_execution_truth_layer",
                    "runtime_paper_regression_guard",
                    "paper_ramp_guard",
                    "broker_auth_supervisor",
                    "daily_verify_auto_remediation",
                    "storage_backpressure_autopilot",
                    "source_mutation_guard",
                    "production_flow_smoke",
                    "promotion_quality_gate",
                ],
                stop_when="live_canary_readiness_contract reports live_canary_money_ready=true after the sustained production-hardening window.",
                expected_impact="Keeps infrastructure bots focused on hardening paper/auth/storage/source/CI/gate freshness before any live-money canary.",
            )
        )

    production_slo_status = _status(production_quality_slo.get("overall_status"))
    production_slo_breach_count = _safe_int(production_quality_slo.get("breach_count"), 0)
    production_slo_warning_count = _safe_int(production_quality_slo.get("warning_count"), 0)
    if production_quality_slo and (
        production_slo_status in BAD_STATUSES or production_slo_breach_count > 0 or production_slo_warning_count > 0
    ):
        breached_lanes = [
            str(_as_dict(row).get("lane_id") or "")
            for row in _as_list(production_quality_slo.get("breached_lanes"))
            if str(_as_dict(row).get("lane_id") or "").strip()
        ]
        warning_lanes = [
            str(_as_dict(row).get("lane_id") or "")
            for row in _as_list(production_quality_slo.get("warning_lanes"))
            if str(_as_dict(row).get("lane_id") or "").strip()
        ]
        needs.append(
            _need(
                need_id="production_quality_slo_breach",
                title="Production quality lane SLO needs bounded repair escalation",
                category="live_canary",
                severity="critical" if production_slo_breach_count > 0 else "high",
                evidence=[
                    f"production_quality_slo_status={production_slo_status or 'unknown'}",
                    f"breach_count={production_slo_breach_count}",
                    f"warning_count={production_slo_warning_count}",
                    f"breached_lanes={','.join(breached_lanes[:8]) or 'none'}",
                    f"warning_lanes={','.join(warning_lanes[:8]) or 'none'}",
                ],
                target_capabilities=[
                    "production_quality_slo_guard",
                    "production_quality_control",
                    "paper_profitability_control",
                    "broker_auth_supervisor",
                    "paper_ramp_guard",
                    "storage_backpressure_autopilot",
                    "daily_verify_auto_remediation",
                ],
                stop_when="production_quality_slo_guard reports no warning or breached lanes after production-quality lanes clear or remain under SLO thresholds.",
                expected_impact="Prevents repeated production-quality degradation from becoming an unbounded repair loop before live canary money.",
            )
        )

    pressure_score = max(
        _safe_float(pressure.get("host_saturation_score"), 0.0),
        _safe_float(runtime_throttle.get("host_saturation_score"), 0.0),
        _safe_float(memory.get("host_saturation_score"), 0.0),
    )
    compute_level = _status(pressure.get("compute_pressure_level") or runtime_throttle.get("compute_pressure_level"))
    memory_level = _status(pressure.get("memory_pressure_level") or memory.get("memory_pressure_level"))
    pressure_status = _status(pressure.get("overall_status"))
    if pressure_score >= 55.0 or compute_level in {"high", "critical"} or memory_level in {"high", "critical"} or pressure_status in BAD_STATUSES:
        severity = "critical" if pressure_score >= 80.0 or "critical" in {compute_level, memory_level, pressure_status} else "high"
        needs.append(
            _need(
                need_id="host_pressure",
                title="Host saturation and memory pressure need active relief",
                category="runtime",
                severity=severity,
                evidence=[
                    f"host_saturation_score={pressure_score:.1f}",
                    f"compute_pressure_level={compute_level or 'unknown'}",
                    f"memory_pressure_level={memory_level or 'unknown'}",
                    f"pressure_status={pressure_status or 'unknown'}",
                ],
                target_capabilities=["pressure_relief_control", "runtime_throttle_control", "memory_pressure_intelligence"],
                stop_when="host_saturation_score stays below 55 and compute/memory pressure return to normal.",
                expected_impact="Keeps support jobs, feed refreshes, and training decisions from fighting the live runtime for CPU and memory.",
            )
        )

    host_attribution = _as_dict(runtime_throttle.get("host_pressure_attribution"))
    sql_writer_fluidity = _as_dict(runtime_throttle.get("sql_writer_fluidity_contract"))
    mac_fluidity = _as_dict(runtime_throttle.get("mac_fluidity_contract"))
    sql_writer_cpu = _safe_float(
        _as_dict(sql_writer_fluidity.get("measurements")).get(
            "storage_writer_cpu_percent",
            host_attribution.get("storage_writer_cpu_percent"),
        ),
        0.0,
    )
    sql_fluidity_active = bool(sql_writer_fluidity.get("active", False))
    mac_fluidity_status = _status(mac_fluidity.get("overall_status"))
    mac_fluidity_band = _status(mac_fluidity.get("fluidity_band"))
    if (
        sql_fluidity_active
        or (
            sql_writer_cpu >= 85.0
            and (
                pressure_score >= 55.0
                or compute_level in {"high", "critical"}
                or mac_fluidity_status in BAD_STATUSES
                or mac_fluidity_band in {"strained", "protect"}
            )
        )
    ):
        needs.append(
            _need(
                need_id="sql_writer_fluidity",
                title="SQL writer heat needs automatic fluidity caps",
                category="runtime",
                severity="high" if sql_writer_cpu >= 150.0 or mac_fluidity_band == "protect" else "medium",
                evidence=[
                    f"sql_writer_fluidity_active={sql_fluidity_active}",
                    f"sql_writer_cpu_percent={sql_writer_cpu:.1f}",
                    f"mac_fluidity_status={mac_fluidity_status or 'unknown'}",
                    f"mac_fluidity_band={mac_fluidity_band or 'unknown'}",
                    f"host_saturation_score={pressure_score:.1f}",
                    f"compute_pressure_level={compute_level or 'unknown'}",
                ],
                target_capabilities=["sql_writer_fluidity_governor", "runtime_throttle_control", "writer_cycle_coordinator"],
                stop_when="SQL writer CPU is below 85%, Mac fluidity is watch/ready, and host saturation is below the guarded band.",
                expected_impact="Caps storage writer fanout before it degrades foreground/runtime fluidity, while preserving the single SQLite writer contract.",
            )
        )

    storage_status = _status(storage.get("overall_status"))
    storage_backpressure = _as_dict(storage.get("backpressure"))
    pending_lines = _safe_int(storage_backpressure.get("total_pending_lines"), 0)
    core_pending_lines = _safe_int(storage_backpressure.get("core_pending_lines"), pending_lines)
    pending_threshold = max(_safe_int(storage_backpressure.get("pending_lines_threshold"), 15000), 1)
    oldest_pending = _safe_float(storage_backpressure.get("oldest_pending_age_seconds"), 0.0)
    oldest_threshold = max(_safe_float(storage_backpressure.get("oldest_age_threshold_seconds"), 240.0), 1.0)
    overlay_pressure_clear = _bool(storage_backpressure.get("overlay_pressure_clear", False))
    storage_severity = _status(storage.get("severity"))
    pressure_index = _safe_float(storage.get("pressure_index"), 0.0)
    health_fast_storage = _as_dict(health_fast.get("storage"))
    health_fast_backpressure = _as_dict(health_fast_storage.get("backpressure"))
    health_fast_storage_clear = bool(
        health_fast
        and _status(health_fast_storage.get("severity")) in {"stable", "ready", "ok"}
        and _safe_float(health_fast_storage.get("pressure_index"), 1.0) < 0.5
        and _safe_int(health_fast_backpressure.get("core_pending_lines"), 0) <= pending_threshold
        and _safe_float(health_fast_backpressure.get("oldest_pending_age_seconds"), 0.0) <= oldest_threshold
        and _bool(health_fast_backpressure.get("overlay_pressure_clear", False))
    )
    storage_operationally_clear = bool(
        storage_status in GOOD_STATUSES
        and storage_severity not in {"blocked", "critical", "high"}
        and pressure_index < 0.5
        and core_pending_lines <= pending_threshold
        and oldest_pending <= oldest_threshold
        and (overlay_pressure_clear or pending_lines == 0 or health_fast_storage_clear)
    )
    retention_debt_gb = _safe_float(_as_dict(storage.get("storage")).get("retention_debt_gb"), 0.0)
    storage_auto_status = _status(storage_auto.get("overall_status"))
    material_pending = pending_lines > 0 and not storage_operationally_clear
    if storage_status in BAD_STATUSES or storage_auto_status in BAD_STATUSES or material_pending or retention_debt_gb > 0.25:
        needs.append(
            _need(
                need_id="storage_backpressure",
                title="Storage and ingestion backpressure need bounded draining",
                category="storage",
                severity="high" if storage_status in {"blocked", "critical"} or pending_lines > 10000 else "medium",
                evidence=[
                    f"ingestion_storage_status={storage_status or 'unknown'}",
                    f"storage_autopilot_status={storage_auto_status or 'unknown'}",
                    f"pending_lines={pending_lines}",
                    f"core_pending_lines={core_pending_lines}",
                    f"overlay_pressure_clear={overlay_pressure_clear}",
                    f"storage_operationally_clear={storage_operationally_clear}",
                    f"retention_debt_gb={retention_debt_gb:.3f}",
                ],
                target_capabilities=["writer_cycle_coordinator", "storage_backpressure_autopilot"],
                stop_when="pending backlog is drained or bounded, retention debt is controlled, and single-writer state is idle/healthy.",
                expected_impact="Keeps ingestion cleaner so the system does not run into space pressure or writer contention as quickly.",
            )
        )

    storage_plane = _as_dict(storage.get("storage_plane_contract"))
    storage_allowed_work = _as_dict(storage.get("allowed_work"))
    storage_allowed_work.update(_as_dict(storage_plane.get("allowed_work")))
    storage_details = _as_dict(storage.get("storage"))
    raw_status = _status(raw_training.get("overall_status"))
    raw_summary = _as_dict(raw_training.get("raw_summary"))
    raw_candidate_count = max(
        _safe_int(raw_summary.get("compression_candidate_count"), 0),
        _safe_int(raw_summary.get("training_candidate_count"), 0),
    )
    raw_candidate_gb = max(
        _safe_float(raw_summary.get("compression_candidate_gb"), 0.0),
        _safe_float(raw_summary.get("training_candidate_gb"), 0.0),
    )
    external_status = _status(external_backlog.get("overall_status") or external_backlog.get("status"))
    external_follow = _as_dict(external_backlog.get("follow_through"))
    external_follow_status = _status(external_follow.get("status"))
    external_recommended = bool(
        external_backlog.get("recommended_now")
        or external_backlog.get("material_drain_recommended")
        or storage_details.get("backlog_drain_recommended_now")
    )
    retention_status = _status(storage_retention.get("overall_status"))
    retention_next_action = str(storage_retention.get("next_action") or "")
    retention_recommended = retention_status in BAD_STATUSES or "rerun storage-retention-unison" in retention_next_action
    quota_status = _status(storage_quota.get("overall_status"))
    quota_summary = _as_dict(storage_quota.get("quota_summary"))
    quota_degraded_families = {
        str(item or "").strip()
        for item in quota_summary.get("degraded_families", [])
        if str(item or "").strip()
    } if isinstance(quota_summary.get("degraded_families"), list) else set()
    quota_hard_breaches = _safe_int(quota_summary.get("hard_breaches"), 0)
    quota_soft_breaches = _safe_int(quota_summary.get("soft_breaches"), 0)
    stateful_sql_quota_pressure = bool(
        quota_status in BAD_STATUSES
        or quota_hard_breaches > 0
        or quota_soft_breaches > 0
        or "sql_link_shards" in quota_degraded_families
    )
    cleanup_handoff_needed = (
        external_recommended
        or external_status in {"drain_active", "handoff_requested", "ready_to_drain", "waiting_for_writer"}
        or external_follow_status in {"handoff_requested", "requested_live_writer", "drain_active"}
        or raw_status in BAD_STATUSES
        or raw_candidate_count > 0
        or raw_candidate_gb > 0.05
        or retention_recommended
        or stateful_sql_quota_pressure
        or bool(storage_allowed_work.get("raw_training_compaction_apply"))
    )
    if cleanup_handoff_needed:
        needs.append(
            _need(
                need_id="cleanup_handoff_ingestion",
                title="Cleanup, handoff, and ingestion drainers need specialized coordination",
                category="storage",
                severity="high" if pressure_index >= 0.55 or pending_lines > 5000 or external_status == "drain_active" else "medium",
                evidence=[
                    f"pressure_index={pressure_index:.3f}",
                    f"pending_lines={pending_lines}",
                    f"external_backlog_status={external_status or 'unknown'}",
                    f"external_follow_status={external_follow_status or 'unknown'}",
                    f"external_recommended={external_recommended}",
                    f"raw_training_status={raw_status or 'unknown'}",
                    f"raw_training_candidate_count={raw_candidate_count}",
                    f"raw_training_candidate_gb={raw_candidate_gb:.3f}",
                    f"retention_status={retention_status or 'unknown'}",
                    f"storage_quota_status={quota_status or 'unknown'}",
                    f"storage_quota_soft_breaches={quota_soft_breaches}",
                    f"storage_quota_hard_breaches={quota_hard_breaches}",
                    f"storage_quota_degraded_families={','.join(sorted(quota_degraded_families)) or 'none'}",
                    f"raw_training_compaction_apply_allowed={bool(storage_allowed_work.get('raw_training_compaction_apply'))}",
                ],
                target_capabilities=[
                    "external_backlog_drain_handoff",
                    "raw_training_cleanup_handoff",
                    "deep_cold_second_cold_handoff",
                    "storage_retention_unison_handoff",
                    "stateful_sql_quota_relief",
                    "storage_backpressure_autopilot",
                    "writer_cycle_coordinator",
                ],
                stop_when="external backlog drain is complete, raw training compaction has no selected/eligible cleanup work, retention unison reports ready, storage quota guard is ready, and ingestion backpressure is below pressure thresholds.",
                expected_impact="Adds focused cleanup, cold-archive, and stateful SQL owners so ingestion drains faster without starting competing SQLite writers or broad cleanup fanout.",
            )
        )

    writer_status = _status(writer.get("overall_status") or writer.get("state"))
    writer_before = _as_dict(writer.get("writer_state_before"))
    writer_after = _as_dict(writer.get("writer_state_after_wait"))
    child_writer_active = bool(writer_before.get("child_writer_active") or writer_after.get("child_writer_active"))
    handoff_needed = bool(
        writer_before.get("complete_lock_handoff_needed")
        or writer_after.get("complete_lock_handoff_needed")
        or _as_dict(writer.get("summary")).get("completed_writer_lock_handoff_needed")
    )
    if writer_status in BAD_STATUSES or child_writer_active or handoff_needed:
        needs.append(
            _need(
                need_id="writer_handoff",
                title="Single-writer handoff state needs coordination",
                category="storage",
                severity="medium",
                evidence=[
                    f"writer_status={writer_status or 'unknown'}",
                    f"child_writer_active={child_writer_active}",
                    f"handoff_needed={handoff_needed}",
                ],
                target_capabilities=["writer_cycle_coordinator"],
                stop_when="writer-cycle-coordinator reports no child writer and no completed writer lock handoff is needed.",
                expected_impact="Prevents competing SQLite writers and makes drainer handoffs predictable.",
            )
        )

    launch_contract = _as_dict(training.get("training_launch_contract"))
    launch_allowed = bool(launch_contract.get("launch_allowed", training.get("launch_allowed", False)))
    launch_blockers = _as_list(launch_contract.get("launch_blockers")) + _as_list(launch_contract.get("prep_blockers"))
    training_status = _status(training.get("overall_status"))
    if training_status in BAD_STATUSES or not launch_allowed or launch_blockers:
        needs.append(
            _need(
                need_id="training_gate",
                title="Training gate needs current runtime headroom truth",
                category="training",
                severity="medium",
                evidence=[
                    f"training_status={training_status or 'unknown'}",
                    f"launch_allowed={launch_allowed}",
                    f"launch_blockers={','.join(str(item) for item in launch_blockers[:6]) or 'none'}",
                ],
                target_capabilities=["training_prep_autopilot", "runtime_snapshot_refresh", "training_runtime_control"],
                stop_when="training-runtime-control reports launch_allowed=true with a current recommended command, or explicitly says to wait.",
                expected_impact="Keeps retraining in sync with real host pressure instead of launching into a saturated box.",
            )
        )

    livefeed_status = _status(livefeed.get("status") or livefeed.get("overall_status"))
    livefeed_alive = bool(livefeed.get("alive", livefeed.get("following", False)))
    idle_heartbeat = _safe_float(livefeed.get("idle_heartbeat_seconds"), 0.0)
    heavy_mode = _status(heavy_view.get("mode") or heavy_view.get("status"))
    skipped_files = _safe_int(livefeed.get("skipped_files") or livefeed.get("skipped_unreadable_files"), 0)
    refresh_guard_status = _status(livefeed_refresh_guard.get("overall_status"))
    refresh_guard_stamp = _parse_utc(livefeed_refresh_guard.get("timestamp_utc"))
    refresh_guard_age_seconds = (
        max(0.0, (_utc_now() - refresh_guard_stamp).total_seconds())
        if refresh_guard_stamp is not None
        else None
    )
    refresh_guard_due = bool(
        not livefeed_refresh_guard
        or refresh_guard_status not in GOOD_STATUSES
        or refresh_guard_age_seconds is None
        or refresh_guard_age_seconds > 15 * 60
    )
    if (
        (livefeed_status and livefeed_status not in GOOD_STATUSES)
        or not livefeed_alive
        or idle_heartbeat > 90.0
        or heavy_mode in {"expired_or_closed", "closed", "expired", "missing"}
        or skipped_files > 0
        or refresh_guard_due
    ):
        needs.append(
            _need(
                need_id="livefeed_continuity",
                title="Heavy livefeed continuity needs refresh guard coverage",
                category="operator_visibility",
                severity="high" if not livefeed_alive else "medium",
                evidence=[
                    f"livefeed_status={livefeed_status or 'unknown'}",
                    f"alive={livefeed_alive}",
                    f"idle_heartbeat_seconds={idle_heartbeat:.1f}",
                    f"heavy_view_mode={heavy_mode or 'unknown'}",
                    f"skipped_files={skipped_files}",
                    f"refresh_guard_status={refresh_guard_status or 'missing'}",
                    f"refresh_guard_age_seconds={refresh_guard_age_seconds:.1f}" if refresh_guard_age_seconds is not None else "refresh_guard_age_seconds=unknown",
                ],
                target_capabilities=["livefeed_refresh_guard"],
                stop_when="livefeed health reports alive/running, route validation is under 15 minutes old, and the heavy view has no unreadable-file churn.",
                expected_impact="Restores the operator feed without restarting trading sleeves or creating extra writer load.",
            )
        )

    commands_status = _status(commands.get("overall_status"))
    command_validity_status = _status(command_validity.get("overall_status"))
    commands_changed = bool(commands.get("commands_changed") or _as_dict(commands.get("apply_results")).get("commands_md_written"))
    command_validity_metrics = _as_dict(command_validity.get("metrics"))
    command_validity_failure_count = sum(
        _safe_int(command_validity_metrics.get(key), 0)
        for key in (
            "blocked_entry_count",
            "degraded_entry_count",
            "smoke_failure_count",
            "runtime_smoke_failure_count",
            "base_runtime_smoke_failure_count",
            "contract_dispatch_smoke_failure_count",
            "commands_hygiene_failure_count",
            "contract_hash_mismatch_count",
        )
    )
    command_validity_effective_bad = command_validity_status in BAD_STATUSES and (
        not bool(command_validity.get("ok", False)) or command_validity_failure_count > 0
    )
    command_issue_count = (
        len(_as_list(commands.get("issues")))
        + len(_as_list(command_validity.get("issues")))
        + command_validity_failure_count
    )
    if commands_status in BAD_STATUSES or command_validity_effective_bad or commands_changed or command_issue_count > 0:
        needs.append(
            _need(
                need_id="command_surface_hygiene",
                title="Operator command surface needs contract refresh",
                category="operator_surface",
                severity="low",
                evidence=[
                    f"commands_status={commands_status or 'unknown'}",
                    f"command_validity_status={command_validity_status or 'unknown'}",
                    f"commands_changed={commands_changed}",
                    f"command_validity_failure_count={command_validity_failure_count}",
                    f"issue_count={command_issue_count}",
                ],
                target_capabilities=["commands_hygiene", "command_validity"],
                stop_when="COMMANDS.md, runbook, commands contract, and opsctl validity agree.",
                expected_impact="Keeps the user-facing commands aligned with the actual ops routes.",
            )
        )

    infra_status = _status(infra.get("overall_status"))
    repair_plan_count = len(_as_list(infra.get("repair_plan")))
    advisory_repair_count = len(_as_list(infra.get("advisory_repair_plan")))
    master_status = _status(master.get("overall_status"))
    if infra_status in BAD_STATUSES or repair_plan_count > 0 or master_status in BAD_STATUSES:
        needs.append(
            _need(
                need_id="infrastructure_repair_plan",
                title="Infrastructure repair plan needs parent supervision",
                category="infrastructure",
                severity="medium" if infra_status != "blocked" else "high",
                evidence=[
                    f"infrastructure_autofix_status={infra_status or 'unknown'}",
                    f"repair_plan_count={repair_plan_count}",
                    f"advisory_repair_count={advisory_repair_count}",
                    f"master_infra_status={master_status or 'unknown'}",
                ],
                target_capabilities=[
                    "daily_verify_auto_remediation",
                    "stateful_storage_regression_guard",
                    "system_drift_autopilot",
                    "runtime_snapshot_refresh",
                    "restart_blackstart_refresh",
                    "blackstart_recovery_refresh",
                    "bot_quality_autopilot",
                    "infrastructure_autofix",
                    "master_infra_supervisor",
                ],
                stop_when="master supervisor and infrastructure autofix agree that no repair plan remains active.",
                expected_impact="Keeps broad repair work sequenced behind the pressure and writer guards.",
            )
        )

    active_gap_count = _safe_int(gap_roster.get("active_count"), 0)
    if active_gap_count > 0 or _status(gap_roster.get("overall_status")) in BAD_STATUSES:
        needs.append(
            _need(
                need_id="infrabot_coverage_gaps",
                title="Infrabot coverage gaps need delegation",
                category="infrastructure",
                severity="medium",
                evidence=[
                    f"gap_roster_status={_status(gap_roster.get('overall_status')) or 'unknown'}",
                    f"active_gap_count={active_gap_count}",
                ],
                target_capabilities=["infrabot_gap_roster", "infrabot_gap_safe_delegation", "master_infra_supervisor"],
                stop_when="infrabot-gap-roster reports active_count=0 or all active gaps have assigned owner bots.",
                expected_impact="Ensures new system needs get explicit owners instead of becoming silent degradation.",
            )
        )

    bot_quality_status = _status(bot_quality.get("overall_status"))
    paper_status = _status(paper_profitability.get("overall_status"))
    paper_truth_status = _status(paper_truth.get("overall_status") or paper_truth.get("status"))
    paper_truth_failed = [str(item) for item in _as_list(paper_truth.get("failed_checks")) if str(item or "").strip()]
    paper_truth_warnings = [str(item) for item in _as_list(paper_truth.get("warnings")) if str(item or "").strip()]
    paper_truth_watch = bool(paper_truth and paper_truth_status == "watch" and not paper_truth_failed and paper_truth.get("ok") is not False)
    training_labeling_status = _status(training_labeling.get("overall_status"))
    missing_label_contracts = _safe_int(training_labeling.get("missing_label_contract_count"), 0)
    incomplete_label_contracts = _safe_int(training_labeling.get("incomplete_label_contract_count"), 0)
    paper_grade = str(paper_profitability.get("profitability_grade") or paper_profitability.get("grade") or "").upper()
    low_grade_summary = _as_dict(paper_profitability.get("low_grade_layer_summary"))
    active_low_grade_blockers = _safe_int(low_grade_summary.get("active_blocker_count"), 0)
    paper_runtime_status = _status(paper_runtime.get("overall_status"))
    paper_backlog_ok = bool(paper_backlog.get("ok", True))
    raw_recovery = _raw_profitability_recovery_context(
        paper_profitability=paper_profitability,
        paper_runtime_profitability=paper_runtime_profitability,
        live_canary_readiness=live_canary_readiness,
    )
    if bool(raw_recovery.get("active", False)):
        top_drags = [
            str(_as_dict(row).get("profile") or "")
            for row in _as_list(raw_recovery.get("top_drag_profiles"))
            if str(_as_dict(row).get("profile") or "").strip()
        ]
        needs.append(
            _need(
                need_id="raw_profitability_burn_down",
                title="Raw profitability recovery needs burn-down evidence",
                category="paper_trading",
                severity="critical" if str(raw_recovery.get("raw_profitability_grade") or "") in {"D", "F"} else "high",
                evidence=[
                    f"raw_profitability_grade={raw_recovery.get('raw_profitability_grade') or 'unknown'}",
                    f"controlled_profitability_grade={raw_recovery.get('controlled_profitability_grade') or 'unknown'}",
                    f"financial_profitability_grade={raw_recovery.get('financial_profitability_grade') or 'unknown'}",
                    f"raw_net_pnl={_safe_float(raw_recovery.get('net_pnl'), 0.0):.6f}",
                    f"realized_pnl={_safe_float(raw_recovery.get('realized_pnl'), 0.0):.6f}",
                    f"unrealized_pnl={_safe_float(raw_recovery.get('unrealized_pnl'), 0.0):.6f}",
                    f"change_vs_previous_day={_safe_float(raw_recovery.get('change_vs_previous_day'), 0.0):.6f}",
                    f"weak_profile_count={_safe_int(raw_recovery.get('weak_profile_count'), 0)}",
                    f"strategy_control_count={_safe_int(raw_recovery.get('strategy_control_count'), 0)}",
                    f"daily_net_improvement_target={_safe_float(raw_recovery.get('daily_net_improvement_target'), 0.0):.6f}",
                    f"top_loss_causes={','.join(_as_list(raw_recovery.get('top_loss_causes'))[:6]) or 'none'}",
                    f"top_drag_profiles={','.join(top_drags[:5]) or 'none'}",
                    f"live_canary_raw_blockers={','.join(_as_list(raw_recovery.get('live_canary_raw_blockers'))[:6]) or 'none'}",
                    f"raw_recovery_requirements={_safe_int(raw_recovery.get('ready_requirement_count'), 0)}/{_safe_int(raw_recovery.get('requirement_count'), 0)}",
                ],
                target_capabilities=[
                    "paper_profitability_control",
                    "paper_performance_refresh",
                    "runtime_paper_regression_guard",
                    "paper_execution_truth_layer",
                    "training_data_intake_labeling",
                    "training_labeling_intelligence",
                    "master_grandmaster_profitability_trainer",
                    "live_canary_readiness_contract",
                    "promotion_quality_gate",
                ],
                stop_when=(
                    "raw_profitability_grade is A or better, raw net PnL is non-negative, weak profiles and losing "
                    "strategy pairs have three profitable refreshes or remain quarantined, and live-canary raw blockers are empty."
                ),
                expected_impact=(
                    "Keeps raw D/F paper outcomes routed to zero-entry controls, reduce-only drag burn-down, strict clean-sleeve "
                    "admission, loss-cause training feedback, and promotion/live-canary gates without granting cosmetic grade credit."
                ),
            )
        )
    if (
        bot_quality_status in BAD_STATUSES
        or paper_status in BAD_STATUSES
        or (paper_truth_status in BAD_STATUSES and not paper_truth_watch)
        or bool(paper_truth_failed)
        or paper_runtime_status in BAD_STATUSES
        or training_labeling_status in BAD_STATUSES
        or missing_label_contracts > 0
        or incomplete_label_contracts > 0
        or (paper_grade in {"D", "F"} and active_low_grade_blockers > 0)
        or not paper_backlog_ok
    ):
        needs.append(
            _need(
                need_id="paper_feedback_quality",
                title="Paper trading feedback quality needs repair loop",
                category="paper_trading",
                severity="medium",
                evidence=[
                    f"bot_quality_status={bot_quality_status or 'unknown'}",
                    f"paper_profitability_status={paper_status or 'unknown'}",
                    f"paper_profitability_grade={paper_grade or 'unknown'}",
                    f"paper_truth_status={paper_truth_status or 'unknown'}",
                    f"paper_truth_failed_checks={','.join(paper_truth_failed[:6]) or 'none'}",
                    f"active_low_grade_blockers={active_low_grade_blockers}",
                    f"paper_runtime_status={paper_runtime_status or 'unknown'}",
                    f"paper_backlog_ok={paper_backlog_ok}",
                    f"training_labeling_status={training_labeling_status or 'unknown'}",
                    f"missing_label_contracts={missing_label_contracts}",
                    f"incomplete_label_contracts={incomplete_label_contracts}",
                ],
                target_capabilities=[
                    "broker_auth_supervisor",
                    "runtime_paper_regression_guard",
                    "paper_performance_refresh",
                    "paper_execution_truth_layer",
                    "paper_profitability_control",
                    "paper_ramp_guard",
                    "bot_quality_autopilot",
                    "training_data_intake_labeling",
                    "training_labeling_intelligence",
                ],
                stop_when="paper runtime regression guard passes and weak profiles are repaired, quarantined, or deweighted.",
                expected_impact="Feeds harder real-world paper outcomes back into bot promotion and abstention behavior.",
            )
        )
    elif paper_truth_watch:
        needs.append(
            _need(
                need_id="paper_truth_watch_reconciliation",
                title="Paper execution truth watch needs attribution follow-through",
                category="paper_trading",
                severity="low",
                evidence=[
                    f"paper_truth_status={paper_truth_status}",
                    f"paper_truth_failed_checks={','.join(paper_truth_failed) or 'none'}",
                    f"paper_truth_warnings={','.join(paper_truth_warnings[:6]) or 'none'}",
                ],
                target_capabilities=["paper_execution_truth_layer", "paper_performance_refresh"],
                stop_when="paper-execution-truth reports ready or watch with no failed checks and fresh attribution warnings only.",
                expected_impact="Keeps counterfactual attribution and broker-truth reconciliation owned without treating a clean watch state as a paper blocker.",
            )
        )

    source_status = _status(source_verification.get("overall_status"))
    provider_status = _status(provider_mesh.get("overall_status"))
    overall_sources = _as_dict(source_verification.get("overall"))
    unverified_count = len(_as_list(overall_sources.get("unverified_sources")))
    stale_count = len(_as_list(overall_sources.get("stale_sources"))) or _safe_int(overall_sources.get("stale_source_count"), 0)
    if source_status in BAD_STATUSES or provider_status in BAD_STATUSES or unverified_count > 0 or stale_count > 0:
        needs.append(
            _need(
                need_id="source_quality",
                title="Source verification and provider mesh need fresh coverage",
                category="data_quality",
                severity="medium",
                evidence=[
                    f"source_verification_status={source_status or 'unknown'}",
                    f"provider_mesh_status={provider_status or 'unknown'}",
                    f"unverified_sources={unverified_count}",
                    f"stale_sources={stale_count}",
                ],
                target_capabilities=[
                    "source_verification",
                    "source_verification_autorefresh",
                    "health_gates_recheck",
                    "provider_mesh_refresh",
                    "market_explanation_evidence",
                    "infrastructure_autofix",
                ],
                stop_when="source verification reports zero unverified/stale sources and provider mesh is ready.",
                expected_impact="Improves data trust before training, paper evaluation, and market sentiment layers consume the inputs.",
            )
        )

    explainer_status = _status(market_explainer.get("overall_status"))
    symbol_evidence_count = _safe_int(market_explainer.get("symbol_evidence_count"), 0)
    primary_confidence = _safe_float(market_explainer.get("primary_confidence"), 0.0)
    source_coverage = _as_dict(market_explainer.get("source_coverage"))
    missing_source_coverage = [
        str(name)
        for name, covered in source_coverage.items()
        if not bool(covered) and str(name).strip()
    ]
    if (
        explainer_status in BAD_STATUSES
        or symbol_evidence_count <= 0
        or primary_confidence < 0.70
        or bool(missing_source_coverage)
    ):
        needs.append(
            _need(
                need_id="market_explanation_evidence",
                title="Market explanation evidence needs symbol-backed refresh",
                category="data_quality",
                severity="medium",
                evidence=[
                    f"market_explainer_status={explainer_status or 'unknown'}",
                    f"symbol_evidence_count={symbol_evidence_count}",
                    f"primary_confidence={primary_confidence:.2f}",
                    f"missing_source_coverage={','.join(missing_source_coverage[:6]) or 'none'}",
                ],
                target_capabilities=["market_explanation_evidence", "source_verification_autorefresh", "provider_mesh_refresh"],
                stop_when="market_move_explainer reports ready with confidence >= 0.70 and symbol-specific evidence present.",
                expected_impact="Keeps market-move explanations audit-backed before they influence paper feedback or training labels.",
            )
        )

    low_grade_audit = _as_dict(system_needs.get("low_grade_layer_audit"))
    low_grade_blockers = _safe_int(low_grade_audit.get("active_blocker_count"), 0)
    low_grade = str(low_grade_audit.get("control_posture_grade") or "").upper()
    if low_grade_blockers > 0:
        needs.append(
            _need(
                need_id="low_grade_layer_audit",
                title="Low-grade system layers need targeted owner repair",
                category="quality",
                severity="medium" if low_grade_blockers <= 2 else "high",
                evidence=[
                    f"active_low_grade_blockers={low_grade_blockers}",
                    f"control_posture_grade={low_grade or 'unknown'}",
                ],
                target_capabilities=["system_needs_intelligence", "infrabot_gap_roster", "infrastructure_autofix"],
                stop_when="system-needs low_grade_layer_audit active_blocker_count is zero.",
                expected_impact="Routes real D/F evidence to owning repair bots while keeping raw evidence visible.",
            )
        )

    needs = sorted(
        needs,
        key=lambda row: (SEVERITY_RANK.get(str(row.get("severity") or "info"), 4), str(row.get("id") or "")),
    )
    severity_counts: dict[str, int] = {}
    for row in needs:
        severity = str(row.get("severity") or "info")
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": iso_now(),
        "source": "infrabot_adaptive_governor",
        "refresh_needs": bool(refresh_needs),
        "need_count": len(needs),
        "severity_counts": severity_counts,
        "needs": needs,
        "artifact_inputs": {
            "system_needs": bool(system_needs),
            "schwab_auth_supervisor": bool(schwab_auth),
            "auth_lease_manager": bool(auth_lease),
            "broker_readiness": bool(broker_readiness),
            "global_killswitch": bool(global_halt),
            "paper_400_ramp": bool(paper_ramp),
            "pressure_relief_control": bool(pressure),
            "runtime_throttle_control": bool(runtime_throttle),
            "memory_pressure_intelligence": bool(memory),
            "ingestion_storage_control": bool(storage),
            "writer_cycle_coordinator": bool(writer),
            "external_backlog_drain": bool(external_backlog),
            "raw_training_compaction": bool(raw_training),
            "storage_retention_unison": bool(storage_retention),
            "storage_quota_guard": bool(storage_quota),
            "training_runtime_control": bool(training),
            "livefeed_local": bool(livefeed),
            "commands_hygiene": bool(commands),
            "infrastructure_autofix": bool(infra),
            "infrabot_gap_roster": bool(gap_roster),
            "paper_profitability_control": bool(paper_profitability),
            "paper_runtime_profitability_controls": bool(paper_runtime_profitability),
            "paper_execution_truth_layer": bool(paper_truth),
            "runtime_paper_regression_guard": bool(paper_runtime),
            "live_canary_readiness_contract": bool(live_canary_readiness),
            "production_quality_control": bool(production_quality),
            "production_quality_slo_guard": bool(production_quality_slo),
            "source_verification": bool(source_verification),
            "provider_mesh": bool(provider_mesh),
            "market_move_explainer": bool(market_explainer),
            "training_labeling_intelligence": bool(training_labeling),
        },
    }


def _build_safety_guard(contract: dict[str, Any], registry: list[dict[str, Any]], project_root: Path) -> dict[str, Any]:
    pressure = _health(project_root, "pressure_relief_control_latest.json")
    runtime_throttle = _health(project_root, "runtime_throttle_control_latest.json")
    memory = _health(project_root, "memory_pressure_intelligence_latest.json")
    training = _health(project_root, "training_runtime_control_latest.json")
    writer = _health(project_root, "writer_cycle_coordinator_latest.json")

    pressure_score = max(
        _safe_float(pressure.get("host_saturation_score"), 0.0),
        _safe_float(runtime_throttle.get("host_saturation_score"), 0.0),
        _safe_float(memory.get("host_saturation_score"), 0.0),
    )
    compute_level = _status(pressure.get("compute_pressure_level") or runtime_throttle.get("compute_pressure_level"))
    memory_level = _status(pressure.get("memory_pressure_level") or memory.get("memory_pressure_level"))
    pressure_block = pressure_score >= 75.0 or compute_level in {"high", "critical"} or memory_level in {"high", "critical"}

    launch_contract = _as_dict(training.get("training_launch_contract"))
    training_launch_allowed = bool(launch_contract.get("launch_allowed", training.get("launch_allowed", False)))
    recommended_retrain_command = _as_list(launch_contract.get("recommended_retrain_command")) or _as_list(
        training.get("recommended_retrain_command")
    )
    writer_before = _as_dict(writer.get("writer_state_before"))
    writer_after = _as_dict(writer.get("writer_state_after_wait"))
    single_writer_busy = bool(
        writer_before.get("child_writer_active")
        or writer_after.get("child_writer_active")
        or _status(writer.get("overall_status")) in {"waiting_for_writer", "writer_active"}
    )

    allowed_apply: list[str] = []
    blocked: list[str] = []
    guardrails = [
        "no_live_execution_authority",
        "do_not_bypass_training_runtime_control",
        "do_not_start_competing_sqlite_writers",
        "prefer_contract_publication_over_broad_repair_fanout_under_pressure",
    ]
    for cap in registry:
        cap_id = str(cap.get("id") or "")
        if not bool(cap.get("apply_safe")):
            continue
        if pressure_block and not bool(cap.get("safe_under_pressure")):
            blocked.append(cap_id)
            continue
        if bool(cap.get("requires_single_writer_idle")) and single_writer_busy:
            blocked.append(cap_id)
            continue
        allowed_apply.append(cap_id)

    if not training_launch_allowed:
        guardrails.append("training_launch_blocked_until_training_runtime_control_allows")
    healing_contract = _self_awareness_healing_contract(project_root)
    if bool(healing_contract.get("enabled")):
        guardrails.append("self_healing_playbooks_define_retry_verify_and_hold_policy")

    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": iso_now(),
        "overall_status": "guarded" if pressure_block or single_writer_busy or not training_launch_allowed else "ready",
        "host_pressure_score": round(pressure_score, 3),
        "host_pressure_block_active": bool(pressure_block),
        "compute_pressure_level": compute_level or "unknown",
        "memory_pressure_level": memory_level or "unknown",
        "single_writer_busy": bool(single_writer_busy),
        "training_launch_allowed": bool(training_launch_allowed),
        "recommended_retrain_command": recommended_retrain_command,
        "live_execution_authority": False,
        "allowed_apply_capability_ids": ordered_unique(allowed_apply),
        "blocked_capability_ids": ordered_unique(blocked),
        "guardrails": guardrails,
        "need_count": int(contract.get("need_count") or 0),
        "self_healing_playbook_contract": healing_contract,
    }


def _self_awareness_healing_contract(project_root: Path) -> dict[str, Any]:
    payload = _health(project_root, "infrabot_library_self_awareness_control_latest.json")
    healing = _as_dict(payload.get("self_healing_playbooks"))
    playbooks = [row for row in _as_list(healing.get("playbooks")) if isinstance(row, dict)]
    playbooks_by_lane = {str(row.get("lane") or ""): row for row in playbooks if str(row.get("lane") or "").strip()}
    return {
        "enabled": bool(healing.get("enabled", False)),
        "grade": str(healing.get("grade") or ""),
        "playbook_count": _safe_int(healing.get("playbook_count"), len(playbooks)),
        "complete_playbook_count": _safe_int(healing.get("complete_playbook_count"), 0),
        "all_playbooks_complete": bool(healing.get("all_playbooks_complete", False)),
        "all_lanes_have_playbooks": bool(healing.get("all_lanes_have_playbooks", False)),
        "all_needs_have_playbooks": bool(healing.get("all_needs_have_playbooks", False)),
        "authority_safe": bool(healing.get("authority_safe", False))
        and not bool(healing.get("live_execution_authority", True))
        and not bool(healing.get("dependency_mutation_authority", True)),
        "playbooks_by_lane": playbooks_by_lane,
        "source_artifact": "governance/health/infrabot_library_self_awareness_control_latest.json",
    }


def _capability_healing_lane(capability: dict[str, Any]) -> str:
    cap_id = str(capability.get("id") or "").lower()
    if cap_id in {
        "broker_auth_supervisor",
        "global_halt_refresh",
        "paper_ramp_guard",
        "live_canary_readiness_contract",
        "production_quality_control",
        "production_quality_slo_guard",
        "source_mutation_guard",
        "production_flow_smoke",
    }:
        return "auth_live_lock"
    if cap_id in {
        "paper_profitability_control",
        "paper_performance_refresh",
        "paper_execution_truth_layer",
        "runtime_paper_regression_guard",
        "promotion_quality_gate",
        "training_data_intake_labeling",
        "training_labeling_intelligence",
        "master_grandmaster_profitability_trainer",
    }:
        return "raw_profitability_recovery"
    text = " ".join([cap_id, str(capability.get("title") or ""), " ".join(str(item) for item in _as_list(capability.get("owns")))]).lower()
    if any(item in text for item in ("storage", "writer", "sql", "backpressure", "retention", "drain")):
        return "storage_writer"
    if any(item in text for item in ("profitability", "paper", "raw", "training_label", "grandmaster", "loss")):
        return "raw_profitability_recovery"
    if any(item in text for item in ("source", "provider", "collector", "market_explanation")):
        return "source_truth"
    if any(item in text for item in ("pressure", "runtime", "memory", "mlx", "library", "throttle", "fluidity")):
        return "runtime_memory"
    if any(item in text for item in ("auth", "broker", "halt", "canary", "execution", "ramp")):
        return "auth_live_lock"
    return "governance_regression"


def _route_healing_context(capability: dict[str, Any], safety: dict[str, Any]) -> dict[str, Any]:
    lane = _capability_healing_lane(capability)
    contract = _as_dict(safety.get("self_healing_playbook_contract"))
    playbook = _as_dict(_as_dict(contract.get("playbooks_by_lane")).get(lane))
    return {
        "lane": lane,
        "playbook_id": playbook.get("playbook_id"),
        "primary_capability": playbook.get("primary_capability"),
        "max_attempts_per_incident": playbook.get("max_attempts_per_incident"),
        "cooldown_seconds": playbook.get("cooldown_seconds"),
        "verify_command": playbook.get("verify_command") or [],
        "proof_artifacts": playbook.get("proof_artifacts") or [],
        "hold_condition": playbook.get("hold_condition") or "",
        "authority_boundary": playbook.get("authority_boundary") or "advisory_and_safe_repair_only_no_live_execution_authority",
        "contract_ready": bool(contract.get("enabled")) and bool(contract.get("authority_safe")) and bool(playbook),
    }


def _route_policy(
    contract: dict[str, Any],
    registry: list[dict[str, Any]],
    safety: dict[str, Any],
    *,
    max_actions: int,
) -> dict[str, Any]:
    needs = [row for row in _as_list(contract.get("needs")) if isinstance(row, dict)]
    needs_by_capability: dict[str, list[dict[str, Any]]] = {}
    for need in needs:
        for cap_id in _as_list(need.get("target_capabilities")):
            needs_by_capability.setdefault(str(cap_id), []).append(need)

    pressure_block = bool(safety.get("host_pressure_block_active"))
    single_writer_busy = bool(safety.get("single_writer_busy"))
    blocked_ids = set(str(item) for item in _as_list(safety.get("blocked_capability_ids")))
    allowed_apply = set(str(item) for item in _as_list(safety.get("allowed_apply_capability_ids")))
    route_rows: list[dict[str, Any]] = []

    for cap in registry:
        cap_id = str(cap.get("id") or "")
        healing_context = _route_healing_context(cap, safety)
        matching_needs = needs_by_capability.get(cap_id, [])
        matching_need_ids = [str(need.get("id") or "") for need in matching_needs]
        if not matching_needs:
            route_rows.append(
                {
                    "capability_id": cap_id,
                    "title": cap.get("title"),
                    "action": "standby",
                    "command": cap.get("command"),
                    "needs": [],
                    "reason": "No active need currently targets this capability.",
                    "blocked_by": [],
                    "stop_when": "",
                    "self_healing": healing_context,
                }
            )
            continue

        blocked_by: list[str] = []
        action = "run_now"
        reason = "Active need targets this capability."
        if bool(cap.get("advisory_only")):
            action = "advisory_only"
            reason = "Capability is read-only/advisory in this governor."
        if cap_id in blocked_ids:
            blocked_by.append("safety_guard")
            action = "blocked_by_safety"
            reason = "Safety guard blocked apply for this capability."
        elif pressure_block and not bool(cap.get("safe_under_pressure")) and not bool(cap.get("advisory_only")):
            blocked_by.append("host_pressure")
            action = "queue_until_pressure_eases"
            reason = "Host pressure is active, so non-pressure repair fanout is queued."
        elif bool(cap.get("requires_single_writer_idle")) and single_writer_busy:
            blocked_by.append("single_writer_busy")
            action = "queue_until_writer_idle"
            reason = "Single-writer coordination is busy."
        elif bool(cap.get("apply_safe")) and cap_id not in allowed_apply:
            blocked_by.append("not_in_allowed_apply_set")
            action = "blocked_by_safety"
            reason = "Capability is not in the current allowed apply set."

        stop_when = " | ".join(ordered_unique([str(need.get("stop_when") or "") for need in matching_needs]))
        route_rows.append(
            {
                "capability_id": cap_id,
                "title": cap.get("title"),
                "action": action,
                "command": cap.get("command"),
                "needs": matching_need_ids,
                "need_severity": min(
                    (SEVERITY_RANK.get(str(need.get("severity") or "info"), 4) for need in matching_needs),
                    default=4,
                ),
                "reason": reason,
                "blocked_by": ordered_unique(blocked_by),
                "stop_when": stop_when,
                "self_healing": healing_context,
            }
        )

    route_rows = sorted(
        route_rows,
        key=lambda row: (
            {"run_now": 0, "advisory_only": 1, "queue_until_pressure_eases": 2, "queue_until_writer_idle": 2, "blocked_by_safety": 3, "standby": 4}.get(
                str(row.get("action") or ""),
                5,
            ),
            int(row.get("need_severity") or 4),
            str(row.get("capability_id") or ""),
        ),
    )

    run_now = [row for row in route_rows if row.get("action") == "run_now"]
    recommended = [row.get("command") for row in run_now[: max(int(max_actions), 1)] if isinstance(row.get("command"), list)]
    advisory = [row.get("command") for row in route_rows if row.get("action") == "advisory_only" and isinstance(row.get("command"), list)]
    action_counts: dict[str, int] = {}
    for row in route_rows:
        action = str(row.get("action") or "unknown")
        action_counts[action] = action_counts.get(action, 0) + 1

    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": iso_now(),
        "max_actions": max(int(max_actions), 1),
        "action_counts": action_counts,
        "routes": route_rows,
        "recommended_commands": recommended,
        "advisory_commands": advisory[: max(int(max_actions), 1)],
        "integration_contract": {
            "live_execution_authority": False,
            "training_launch_authority": False,
            "sqlite_writer_authority": "single_writer_guarded_only",
            "contracts_published_for": [
                "system_needs_intelligence",
                "infrabot_gap_roster",
                "master_infrastructure_supervisor",
                "infrastructure_autofix",
                "live_canary_readiness_contract",
                "production_quality_control",
                "production_quality_slo_guard",
                "operator_cockpit",
            ],
            "uses_self_healing_playbooks": bool(_as_dict(safety.get("self_healing_playbook_contract")).get("enabled", False)),
        },
    }


def _feedback(project_root: Path) -> dict[str, Any]:
    path = project_root / "governance" / "health" / "infrabot_adaptive_feedback.jsonl"
    rows = load_recent_jsonl(path, limit=100)
    action_counts: dict[str, int] = {}
    for row in rows:
        action = str(row.get("event") or "unknown")
        action_counts[action] = action_counts.get(action, 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "path": str(path),
        "event_count": len(rows),
        "event_counts": action_counts,
        "recent_events": rows[-10:],
    }


def _append_feedback(project_root: Path, payload: dict[str, Any]) -> bool:
    path = project_root / "governance" / "health" / "infrabot_adaptive_feedback.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp_utc": iso_now(),
        "event": "adaptive_governor_publish",
        "need_count": _safe_int(_as_dict(payload.get("system_needs_contract")).get("need_count"), 0),
        "route_action_counts": _as_dict(_as_dict(payload.get("adaptive_policy_router")).get("action_counts")),
        "safety_status": _as_dict(payload.get("safety_guard")).get("overall_status"),
        "recommended_commands": _as_list(_as_dict(payload.get("adaptive_policy_router")).get("recommended_commands")),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return True


def _append_execution_feedback(project_root: Path, execution: dict[str, Any]) -> bool:
    path = project_root / "governance" / "health" / "infrabot_adaptive_feedback.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp_utc": iso_now(),
        "event": "adaptive_governor_safe_repair_execution",
        "executed_count": _safe_int(execution.get("executed_count"), 0),
        "skipped_count": _safe_int(execution.get("skipped_count"), 0),
        "failed_count": _safe_int(execution.get("failed_count"), 0),
        "timed_out_count": _safe_int(execution.get("timed_out_count"), 0),
        "max_execute_actions": _safe_int(execution.get("max_execute_actions"), 0),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return True


def _allowed_execution_commands(registry_rows: list[dict[str, Any]], safety: dict[str, Any]) -> dict[tuple[str, ...], str]:
    allowed_ids = set(str(item) for item in _as_list(safety.get("allowed_apply_capability_ids")))
    allowed: dict[tuple[str, ...], str] = {}
    for cap in registry_rows:
        cap_id = str(cap.get("id") or "")
        command = [str(item) for item in _as_list(cap.get("command"))]
        if not command:
            continue
        if command[0] != "./scripts/ops/opsctl.sh":
            continue
        if not bool(cap.get("apply_safe")) or bool(cap.get("advisory_only")):
            continue
        if cap_id not in allowed_ids:
            continue
        allowed[tuple(command)] = cap_id
    return allowed


def _parse_json_stdout(stdout: str) -> dict[str, Any]:
    text = str(stdout or "").strip()
    if not text:
        return {}
    candidates = [text]
    candidates.extend(reversed([line.strip() for line in text.splitlines() if line.strip()]))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _command_result_summary(parsed: dict[str, Any]) -> dict[str, Any]:
    apply_result = _as_dict(parsed.get("apply_result"))
    write_result = _as_dict(parsed.get("write_result"))
    return {
        "timestamp_utc": str(parsed.get("timestamp_utc") or parsed.get("timestamp") or ""),
        "overall_status": str(parsed.get("overall_status") or parsed.get("status") or ""),
        "ok": parsed.get("ok"),
        "applied": apply_result.get("applied", write_result.get("applied", parsed.get("apply"))),
        "recommended_actions": _as_list(parsed.get("recommended_actions"))[:5],
    }


def _load_self_healing_state(project_root: Path) -> dict[str, Any]:
    state = _as_dict(load_json(project_root / "governance" / "health" / "infrabot_adaptive_self_healing_state.json"))
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": str(state.get("timestamp_utc") or ""),
        "capabilities": _as_dict(state.get("capabilities")),
    }


def _self_healing_capability_state(state: dict[str, Any], cap_id: str) -> dict[str, Any]:
    capabilities = _as_dict(state.get("capabilities"))
    return _as_dict(capabilities.get(cap_id))


def _active_cooldown(state: dict[str, Any], cap_id: str, now: datetime) -> dict[str, Any]:
    cap_state = _self_healing_capability_state(state, cap_id)
    until = _parse_utc(cap_state.get("cooldown_until_utc"))
    if until is None or until <= now:
        return {"active": False}
    return {
        "active": True,
        "cooldown_until_utc": _iso_from_dt(until),
        "reason": str(cap_state.get("cooldown_reason") or ""),
        "last_outcome": str(cap_state.get("last_outcome") or ""),
        "failure_count": _safe_int(cap_state.get("failure_count"), 0),
    }


def _routes_by_capability(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    policy = _as_dict(payload.get("adaptive_policy_router"))
    return {
        str(row.get("capability_id") or ""): row
        for row in _as_list(policy.get("routes"))
        if isinstance(row, dict) and str(row.get("capability_id") or "").strip()
    }


def _self_healing_execution_gate(
    state: dict[str, Any],
    cap_id: str,
    now: datetime,
    healing_context: dict[str, Any],
) -> dict[str, Any]:
    cooldown = _active_cooldown(state, cap_id, now)
    if bool(cooldown.get("active")):
        cooldown["gate"] = "cooldown"
        return cooldown

    cap_state = _self_healing_capability_state(state, cap_id)
    failure_count = _safe_int(cap_state.get("failure_count"), 0)
    max_attempts = _safe_int(healing_context.get("max_attempts_per_incident"), 0)
    if max_attempts <= 0 or failure_count < max_attempts:
        return {"active": False}

    last_seen = _parse_utc(cap_state.get("last_seen_utc"))
    cooldown_seconds = max(_safe_int(healing_context.get("cooldown_seconds"), 0), 60)
    incident_window_seconds = max(cooldown_seconds * max(max_attempts, 1), 3600)
    if last_seen is not None and (now - last_seen).total_seconds() > incident_window_seconds:
        return {"active": False, "reason": "self_healing_incident_window_elapsed"}

    return {
        "active": True,
        "gate": "retry_budget",
        "reason": "self_healing_retry_budget_exhausted",
        "lane": healing_context.get("lane"),
        "playbook_id": healing_context.get("playbook_id"),
        "failure_count": failure_count,
        "max_attempts_per_incident": max_attempts,
        "hold_condition": healing_context.get("hold_condition") or "hold visible and escalate when retry budget is exhausted",
        "requires_operator_attention": True,
    }


def _payload_applied(parsed: dict[str, Any]) -> bool:
    apply_result = _as_dict(parsed.get("apply_result"))
    write_result = _as_dict(parsed.get("write_result"))
    return bool(apply_result.get("applied", write_result.get("applied", parsed.get("apply", False))))


def _classify_command_outcome(returncode: int | None, parsed: dict[str, Any], *, timed_out: bool = False) -> dict[str, Any]:
    summary = _command_result_summary(parsed)
    status = _status(summary.get("overall_status"))
    applied = bool(summary.get("applied")) or _payload_applied(parsed)
    ok = parsed.get("ok")
    if timed_out:
        outcome = "timeout"
        retryable = True
        budget_consuming = True
        success_like = False
    elif applied and status in BAD_STATUSES:
        outcome = "partial_success_blocked"
        retryable = True
        budget_consuming = True
        success_like = True
    elif applied:
        outcome = "success"
        retryable = False
        budget_consuming = True
        success_like = True
    elif returncode == 0 and ok is True and status in SUCCESSFUL_NON_READY_STATUSES:
        outcome = "success"
        retryable = False
        budget_consuming = True
        success_like = True
    elif returncode == 0 and ok is True:
        outcome = "partial_success_blocked" if status in BAD_STATUSES else "success"
        retryable = status in BAD_STATUSES
        budget_consuming = True
        success_like = True
    elif returncode == 0 and status not in BAD_STATUSES and ok is not False:
        outcome = "success"
        retryable = False
        budget_consuming = True
        success_like = True
    elif status == "blocked":
        outcome = "blocked_no_apply"
        retryable = True
        budget_consuming = False
        success_like = False
    elif returncode not in {0, None} or status in BAD_STATUSES:
        outcome = "failed"
        retryable = True
        budget_consuming = returncode is not None
        success_like = False
    else:
        outcome = "advisory_no_apply"
        retryable = False
        budget_consuming = False
        success_like = True
    return {
        "outcome": outcome,
        "retryable": retryable,
        "budget_consuming": budget_consuming,
        "success_like": success_like,
        "summary": summary,
    }


def _cooldown_seconds_for(cap_id: str, outcome: str, failure_count: int, healing_context: dict[str, Any] | None = None) -> int:
    healing_cooldown = _safe_int(_as_dict(healing_context).get("cooldown_seconds"), 0)
    if outcome == "partial_success_blocked":
        return max(180, healing_cooldown)
    if outcome == "timeout":
        return min(3600, max(healing_cooldown, 900 * max(1, min(failure_count, 4))))
    if outcome == "blocked_no_apply":
        base = 1800 if cap_id == "external_backlog_drain_handoff" else 900
        return min(7200, max(healing_cooldown, base * max(1, min(failure_count, 4))))
    if outcome == "failed":
        return min(3600, max(healing_cooldown, 600 * max(1, min(failure_count, 4))))
    return 0


def _update_self_healing_state(
    project_root: Path,
    state: dict[str, Any],
    *,
    cap_id: str,
    command: list[str],
    classification: dict[str, Any],
    returncode: int | None,
    healing_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    healing_context = _as_dict(healing_context)
    capabilities = _as_dict(state.setdefault("capabilities", {}))
    cap_state = _as_dict(capabilities.get(cap_id))
    outcome = str(classification.get("outcome") or "")
    success_like = bool(classification.get("success_like"))
    previous_failures = _safe_int(cap_state.get("failure_count"), 0)
    budget_consumed = bool(classification.get("budget_consuming"))
    consumes_retry_budget = bool(classification.get("retryable")) and budget_consumed
    failure_count = 0 if success_like and outcome != "partial_success_blocked" else previous_failures + (1 if consumes_retry_budget else 0)
    max_attempts = _safe_int(healing_context.get("max_attempts_per_incident"), 0)
    retry_budget_exhausted = bool(max_attempts > 0 and failure_count >= max_attempts and consumes_retry_budget)
    cooldown_seconds = _cooldown_seconds_for(cap_id, outcome, failure_count, healing_context)
    now = _utc_now()
    next_retry = now + timedelta(seconds=cooldown_seconds) if cooldown_seconds > 0 else None
    cap_state.update(
        {
            "capability_id": cap_id,
            "lane": healing_context.get("lane"),
            "playbook_id": healing_context.get("playbook_id"),
            "last_seen_utc": _iso_from_dt(now),
            "last_command": command,
            "last_returncode": returncode,
            "last_outcome": outcome,
            "last_summary": _as_dict(classification.get("summary")),
            "failure_count": failure_count,
            "max_attempts_per_incident": max_attempts,
            "retry_budget_exhausted": retry_budget_exhausted,
            "hold_condition": healing_context.get("hold_condition") or "",
            "cooldown_seconds": cooldown_seconds,
            "cooldown_until_utc": _iso_from_dt(next_retry) if next_retry else "",
            "cooldown_reason": "self_healing_backoff_after_retryable_blocker" if cooldown_seconds else "",
        }
    )
    capabilities[cap_id] = cap_state
    state["capabilities"] = capabilities
    state["timestamp_utc"] = iso_now()
    state["policy"] = {
        "mode": "adaptive_self_healing",
        "cooldown_blocks_repeat_failures": True,
        "blocked_no_apply_does_not_consume_repair_budget": True,
        "playbook_retry_budget_enforced": True,
        "partial_success_blocks_get_short_backoff": True,
        "live_execution_authority": False,
    }
    write_payload(project_root / "governance" / "health" / "infrabot_adaptive_self_healing_state.json", state)
    return cap_state


def _execute_safe_recommended(
    project_root: Path,
    payload: dict[str, Any],
    *,
    max_execute_actions: int,
    command_timeout_seconds: int,
    self_healing: bool = True,
) -> dict[str, Any]:
    registry_rows = [
        row
        for row in _as_list(_as_dict(payload.get("capability_registry")).get("capabilities"))
        if isinstance(row, dict)
    ]
    policy = _as_dict(payload.get("adaptive_policy_router"))
    safety = _as_dict(payload.get("safety_guard"))
    recommended = [row for row in _as_list(policy.get("recommended_commands")) if isinstance(row, list)]
    allowed = _allowed_execution_commands(registry_rows, safety)
    route_contexts = _routes_by_capability(payload)
    max_execute = max(int(max_execute_actions), 0)
    timeout = max(int(command_timeout_seconds), 30)
    env = os.environ.copy()
    env.update(SAFE_EXEC_ENV)
    env.setdefault("BOT_RUNTIME_PROFILE", "live")
    env.setdefault("PYTHONUNBUFFERED", "1")
    self_healing_state = _load_self_healing_state(project_root) if self_healing else {"capabilities": {}}
    now = _utc_now()

    results: list[dict[str, Any]] = []
    attempted_count = 0
    budget_consumed_count = 0
    failed_count = 0
    timed_out_count = 0
    skipped_count = 0
    cooldown_skipped_count = 0
    retry_budget_skipped_count = 0
    for raw_command in recommended:
        command = [str(item) for item in raw_command]
        cap_id = allowed.get(tuple(command), "")
        if not cap_id:
            skipped_count += 1
            results.append(
                {
                    "command": command,
                    "executed": False,
                    "reason": "command_not_in_exact_safe_apply_allowlist",
                }
            )
            continue
        route_context = _as_dict(route_contexts.get(cap_id))
        healing_context = _as_dict(route_context.get("self_healing"))
        healing_gate = _self_healing_execution_gate(self_healing_state, cap_id, now, healing_context) if self_healing else {"active": False}
        if bool(healing_gate.get("active")):
            skipped_count += 1
            if str(healing_gate.get("gate") or "") == "retry_budget":
                retry_budget_skipped_count += 1
            else:
                cooldown_skipped_count += 1
            results.append(
                {
                    "capability_id": cap_id,
                    "command": command,
                    "executed": False,
                    "reason": str(healing_gate.get("reason") or "self_healing_gate_active"),
                    "self_healing": healing_gate,
                }
            )
            continue
        if budget_consumed_count >= max_execute:
            skipped_count += 1
            results.append(
                {
                    "capability_id": cap_id,
                    "command": command,
                    "executed": False,
                    "reason": "max_execute_actions_reached",
                }
            )
            continue
        try:
            completed = subprocess.run(
                command,
                cwd=project_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            parsed = _parse_json_stdout(completed.stdout)
            classification = _classify_command_outcome(completed.returncode, parsed)
            failed = not bool(classification.get("success_like"))
            failed_count += 1 if failed else 0
            attempted_count += 1
            budget_consumed_count += 1 if bool(classification.get("budget_consuming")) else 0
            cap_state = (
                _update_self_healing_state(
                    project_root,
                    self_healing_state,
                    cap_id=cap_id,
                    command=command,
                    classification=classification,
                    returncode=completed.returncode,
                    healing_context=healing_context,
                )
                if self_healing
                else {}
            )
            results.append(
                {
                    "capability_id": cap_id,
                    "command": command,
                    "executed": True,
                    "returncode": completed.returncode,
                    "failed": failed,
                    "classification": classification,
                    "summary": _as_dict(classification.get("summary")),
                    "self_healing": {
                        "enabled": bool(self_healing),
                        "playbook": healing_context,
                        "state": cap_state,
                    },
                    "stdout_tail": completed.stdout[-1200:],
                    "stderr_tail": completed.stderr[-1200:],
                }
            )
        except subprocess.TimeoutExpired as exc:
            attempted_count += 1
            budget_consumed_count += 1
            failed_count += 1
            timed_out_count += 1
            classification = _classify_command_outcome(None, {}, timed_out=True)
            cap_state = (
                _update_self_healing_state(
                    project_root,
                    self_healing_state,
                    cap_id=cap_id,
                    command=command,
                    classification=classification,
                    returncode=None,
                    healing_context=healing_context,
                )
                if self_healing
                else {}
            )
            results.append(
                {
                    "capability_id": cap_id,
                    "command": command,
                    "executed": True,
                    "returncode": None,
                    "failed": True,
                    "timed_out": True,
                    "timeout_seconds": timeout,
                    "classification": classification,
                    "self_healing": {
                        "enabled": bool(self_healing),
                        "playbook": healing_context,
                        "state": cap_state,
                    },
                    "stdout_tail": str(exc.stdout or "")[-1200:],
                    "stderr_tail": str(exc.stderr or "")[-1200:],
                }
            )

    execution = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": iso_now(),
        "enabled": True,
        "self_healing_enabled": bool(self_healing),
        "max_execute_actions": max_execute,
        "command_timeout_seconds": timeout,
        "recommended_count": len(recommended),
        "executed_count": attempted_count,
        "budget_consumed_count": budget_consumed_count,
        "skipped_count": skipped_count,
        "cooldown_skipped_count": cooldown_skipped_count,
        "retry_budget_skipped_count": retry_budget_skipped_count,
        "failed_count": failed_count,
        "timed_out_count": timed_out_count,
        "live_execution_authority": False,
        "exact_allowlist_enforced": True,
        "self_healing_state_path": str(project_root / "governance" / "health" / "infrabot_adaptive_self_healing_state.json"),
        "env_safety_overrides": SAFE_EXEC_ENV,
        "commands": results,
    }
    _append_execution_feedback(project_root, execution)
    return execution


def _apply_contracts(project_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    health = project_root / "governance" / "health"
    paths = {
        "governor": health / "infrabot_adaptive_governor_latest.json",
        "system_needs_contract": health / "infrabot_system_needs_contract_latest.json",
        "capability_registry": health / "infrabot_capability_registry_latest.json",
        "adaptive_policy_router": health / "infrabot_adaptive_policy_latest.json",
        "safety_guard": health / "infrabot_safety_guard_latest.json",
    }
    write_payload(paths["system_needs_contract"], _as_dict(payload.get("system_needs_contract")))
    write_payload(paths["capability_registry"], _as_dict(payload.get("capability_registry")))
    write_payload(paths["adaptive_policy_router"], _as_dict(payload.get("adaptive_policy_router")))
    write_payload(paths["safety_guard"], _as_dict(payload.get("safety_guard")))
    feedback_appended = _append_feedback(project_root, payload)
    updated_feedback = _feedback(project_root)
    payload["learning_feedback"] = updated_feedback
    payload["apply_result"] = {
        "applied": True,
        "contracts_written": {name: str(path) for name, path in paths.items() if name != "governor"},
        "governor_artifact": str(paths["governor"]),
        "feedback_appended": feedback_appended,
        "executed_commands": [],
        "note": "Apply publishes shared infrabot contracts and feedback only; it does not launch repair fanout, training, live execution, or competing SQLite writers.",
    }
    write_payload(paths["governor"], payload)
    return payload["apply_result"]


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool = False,
    refresh_needs: bool = False,
    max_actions: int = 8,
    execute_safe_repairs: bool = False,
    max_execute_actions: int | None = None,
    command_timeout_seconds: int = 300,
    self_healing: bool = True,
) -> dict[str, Any]:
    contract = _needs_contract(project_root, refresh_needs=refresh_needs)
    registry_rows = _capability_registry()
    registry = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": iso_now(),
        "capability_count": len(registry_rows),
        "capabilities": registry_rows,
        "integration_contract": {
            "live_execution_authority": False,
            "safe_apply_only": True,
            "broad_repair_fanout_requires_green_safety_guard": True,
        },
    }
    safety = _build_safety_guard(contract, registry_rows, project_root)
    policy = _route_policy(contract, registry_rows, safety, max_actions=max_actions)
    feedback = _feedback(project_root)

    blocked_count = _safe_int(_as_dict(policy.get("action_counts")).get("blocked_by_safety"), 0)
    queued_count = _safe_int(_as_dict(policy.get("action_counts")).get("queue_until_pressure_eases"), 0) + _safe_int(
        _as_dict(policy.get("action_counts")).get("queue_until_writer_idle"),
        0,
    )
    need_count = _safe_int(contract.get("need_count"), 0)
    overall_status = "ready"
    if need_count > 0:
        overall_status = "coordinating"
    if blocked_count > 0 or queued_count > 0 or _status(safety.get("overall_status")) == "guarded":
        overall_status = "guarded"

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": iso_now(),
        "overall_status": overall_status,
        "system_needs_contract": contract,
        "capability_registry": registry,
        "adaptive_policy_router": policy,
        "safety_guard": safety,
        "learning_feedback": feedback,
        "apply_result": {
            "applied": False,
            "contracts_written": {},
            "feedback_appended": False,
            "executed_commands": [],
            "note": "Dry run only; use --apply to publish shared contracts.",
        },
    }
    if apply:
        _apply_contracts(project_root, payload)
    else:
        write_payload(project_root / "governance" / "health" / "infrabot_adaptive_governor_latest.json", payload)
    if execute_safe_repairs:
        execution = _execute_safe_recommended(
            project_root,
            payload,
            max_execute_actions=max_execute_actions if max_execute_actions is not None else max_actions,
            command_timeout_seconds=command_timeout_seconds,
            self_healing=self_healing,
        )
        apply_result = _as_dict(payload.get("apply_result"))
        apply_result["executed_commands"] = _as_list(execution.get("commands"))
        apply_result["safe_repair_execution"] = execution
        payload["apply_result"] = apply_result
        payload["learning_feedback"] = _feedback(project_root)
        write_payload(project_root / "governance" / "health" / "infrabot_adaptive_governor_latest.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish the adaptive infrabot governor contracts.")
    parser.add_argument("--apply", action="store_true", help="Publish shared contracts and append a learning-feedback event.")
    parser.add_argument("--refresh-needs", action="store_true", help="Refresh system-needs before routing infrabots.")
    parser.add_argument("--max-actions", type=int, default=8, help="Maximum run-now commands to expose in recommended_commands.")
    parser.add_argument(
        "--execute-safe-repairs",
        action="store_true",
        help="Execute exact allowlisted apply-safe recommended commands after publishing contracts.",
    )
    parser.add_argument(
        "--max-execute-actions",
        type=int,
        default=None,
        help="Maximum safe recommended commands to execute; defaults to --max-actions.",
    )
    parser.add_argument("--command-timeout-seconds", type=int, default=300, help="Per-command timeout for safe repair execution.")
    parser.add_argument("--no-self-healing", action="store_true", help="Disable adaptive cooldown/backoff for repeated safe repair blockers.")
    parser.add_argument("--json", action="store_true", help="Print JSON payload.")
    args = parser.parse_args(argv)
    if args.execute_safe_repairs and not args.apply:
        parser.error("--execute-safe-repairs requires --apply")

    payload = build_payload(
        PROJECT_ROOT,
        apply=args.apply,
        refresh_needs=args.refresh_needs,
        max_actions=args.max_actions,
        execute_safe_repairs=args.execute_safe_repairs,
        max_execute_actions=args.max_execute_actions,
        command_timeout_seconds=args.command_timeout_seconds,
        self_healing=not args.no_self_healing,
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        policy = _as_dict(payload.get("adaptive_policy_router"))
        counts = _as_dict(policy.get("action_counts"))
        print(
            "infrabot_adaptive_governor "
            f"status={payload.get('overall_status')} "
            f"needs={_as_dict(payload.get('system_needs_contract')).get('need_count')} "
            f"run_now={counts.get('run_now', 0)} "
            f"queued={_safe_int(counts.get('queue_until_pressure_eases'), 0) + _safe_int(counts.get('queue_until_writer_idle'), 0)} "
            f"blocked={counts.get('blocked_by_safety', 0)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
