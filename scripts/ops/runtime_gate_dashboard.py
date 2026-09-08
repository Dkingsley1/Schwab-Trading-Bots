import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_PATH = (
    PROJECT_ROOT / "governance" / "health" / "runtime_gate_dashboard_latest.json"
)
STATEFUL_SQL_SOFT_QUOTA_MAX_HARD_RATIO = 0.92
OVERLAY_RAW_LIVE_MAX_CORE_LINES = 10_000
OVERLAY_RAW_LIVE_MAX_TOTAL_LINES = 15_000
OVERLAY_RAW_LIVE_MAX_AGE_SECONDS = 15 * 60
BOUNDED_TRANSIENT_STORAGE_PRESSURE_MAX = 1.0


def _hours_to_minutes(hours: float) -> float:
    return max(float(hours), 0.0) * 60.0


def _days_to_minutes(days: float) -> float:
    return _hours_to_minutes(float(days) * 24.0)


def _artifact_config(project_root: Path) -> Dict[str, Dict[str, Any]]:
    return {
        "session_ready": {
            "paths": [
                project_root / "governance" / "health" / "session_ready_latest.json"
            ],
            "max_age_minutes": 15.0,
            "required": True,
        },
        "daily_auto_verify": {
            "paths": [
                project_root / "governance" / "health" / "daily_auto_verify_latest.json"
            ],
            "max_age_minutes": _hours_to_minutes(36.0),
            "required": False,
        },
        "data_source_divergence": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "data_source_divergence_latest.json"
            ],
            "max_age_minutes": _hours_to_minutes(6.0),
            "required": False,
        },
        "source_verification": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "source_verification_latest.json"
            ],
            "max_age_minutes": _hours_to_minutes(6.0),
            "required": False,
        },
        "execution_queue_stress": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "execution_queue_stress_latest.json"
            ],
            "max_age_minutes": _hours_to_minutes(6.0),
            "required": False,
        },
        "sqlite_maintenance": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "sqlite_maintenance_latest.json"
            ],
            "max_age_minutes": _hours_to_minutes(6.0),
            "required": False,
        },
        "snapshot_coverage": {
            "paths": [
                project_root / "governance" / "health" / "snapshot_coverage_latest.json"
            ],
            "max_age_minutes": _hours_to_minutes(12.0),
            "required": False,
        },
        "health_gates": {
            "paths": [
                project_root / "governance" / "health" / "health_gates_latest.json"
            ],
            "max_age_minutes": 240.0,
            "required": True,
        },
        "all_sleeves_launcher": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "all_sleeves_launcher_latest.json"
            ],
            "max_age_minutes": 2.0,
            "required": False,
        },
        "paper_400_ramp": {
            "paths": [
                project_root / "governance" / "health" / "paper_400_ramp_latest.json"
            ],
            "max_age_minutes": 15.0,
            "required": False,
        },
        "unattended_soak_readiness": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "unattended_soak_readiness_latest.json"
            ],
            "max_age_minutes": 15.0,
            "required": False,
        },
        "data_collection_observation_rollup": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "data_collection_observation_rollup_latest.json"
            ],
            "max_age_minutes": 60.0,
            "required": False,
        },
        "global_killswitch": {
            "paths": [
                project_root / "governance" / "health" / "global_killswitch_latest.json"
            ],
            "max_age_minutes": 15.0,
            "required": False,
        },
        "runtime_access_mode": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "runtime_access_mode_latest.json"
            ],
            "max_age_minutes": _hours_to_minutes(24.0),
            "required": False,
        },
        "apple_silicon_profile": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "apple_silicon_profile_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(7.0),
            "required": False,
        },
        "memory_efficiency_control": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "memory_efficiency_control_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "bot_organization_control": {
            "paths": [
                project_root / "governance" / "health" / "bot_organization_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(1.0),
            "required": True,
        },
        "system_role_contract": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "system_role_contract_latest.json"
            ],
            "max_age_minutes": 30.0,
            "required": True,
        },
        "collector_capability_control": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "collector_capability_control_latest.json"
            ],
            "max_age_minutes": 30.0,
            "required": False,
        },
        "capability_materialization": {
            "paths": [
                project_root
                / "governance"
                / "collector_capabilities"
                / "materialized_capabilities_latest.json"
            ],
            "max_age_minutes": 30.0,
            "required": True,
        },
        "bot_profitability_scalability_control": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "bot_profitability_scalability_latest.json"
            ],
            "max_age_minutes": 120.0,
            "required": True,
        },
        "sleeve_scalability_selector": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "sleeve_scalability_selector_latest.json"
            ],
            "max_age_minutes": 120.0,
            "required": True,
        },
        "master_grandmaster_evidence_v2": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "master_grandmaster_evidence_v2_latest.json"
            ],
            "max_age_minutes": 120.0,
            "required": True,
        },
        "training_report": {
            "paths": [
                project_root / "governance" / "health" / "training_report_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(3.0),
            "required": False,
        },
        "nightly_resilience": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "nightly_resilience_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "training_quality_control": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "training_quality_control_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "canary_auto_tuner": {
            "paths": [
                project_root / "governance" / "health" / "canary_auto_tuner_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "ingestion_storage_control": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "ingestion_storage_control_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "ingestion_storage_governor": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "ingestion_storage_governor_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "storage_tier_policy": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "storage_tier_policy_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "runtime_training_snapshot": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "runtime_training_snapshot_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "hdf5_training_cache": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "hdf5_training_cache_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "training_runtime_control": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "training_runtime_control_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "external_backlog_drain": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "external_backlog_drain_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(1.0),
            "required": False,
        },
        "external_backlog_retry_bot": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "external_backlog_retry_bot_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(1.0),
            "required": False,
        },
        "platform_control_plane": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "platform_control_plane_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "platform_intelligence_expansion": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "platform_intelligence_expansion_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "bot_founder_dna_lineage": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "bot_founder_dna_lineage_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "security_evidence_autofix": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "security_evidence_autofix_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "ingestion_priority_queue": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "ingestion_priority_queue_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "storage_split_brain_reconciler": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "storage_split_brain_reconciler_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "storage_resilience_control": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "storage_resilience_control_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "daily_verify_auto_remediation_bot": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "daily_verify_auto_remediation_bot_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "operator_cockpit": {
            "paths": [
                project_root / "governance" / "health" / "operator_cockpit_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(1.0),
            "required": False,
        },
        "incident_closeout_autopilot": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "incident_closeout_autopilot_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "live_canary_control": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "live_canary_control_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "regime_control_plane": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "regime_control_plane_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "supportability_control": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "supportability_control_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "teacher_quality_guard": {
            "paths": [
                project_root
                / "governance"
                / "distillation"
                / "teacher_quality_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "strategy_generation_control": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "strategy_generation_control_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "bot_quality_autopilot": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "bot_quality_autopilot_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "infrastructure_autofix_bot": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "infrastructure_autofix_bot_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "live_runtime_separation_control": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "live_runtime_separation_control_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "rolling_restart_controller": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "rolling_restart_controller_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "auth_lease_manager": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "auth_lease_manager_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(1.0),
            "required": False,
        },
        "schwab_auth_supervisor": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "schwab_auth_supervisor_latest.json"
            ],
            "max_age_minutes": 30.0,
            "required": False,
        },
        "blackstart_recovery": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "blackstart_recovery_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "sleeve_isolation_guard": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "sleeve_isolation_guard_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "artifact_freshness_slo": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "artifact_freshness_slo_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "runtime_snapshot_cache_control": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "runtime_snapshot_cache_control_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "remote_alert_control": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "remote_alert_control_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(1.0),
            "required": False,
        },
        "coordination_state_control": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "coordination_state_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(1.0),
            "required": False,
        },
        "storage_quota_guard": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "storage_quota_guard_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "storage_retention_unison": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "storage_retention_unison_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "release_freeze_guard": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "release_freeze_guard_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "roster_resilience_planner": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "roster_resilience_planner_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "chaos_drill_coordinator": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "chaos_drill_coordinator_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(7.0),
            "required": False,
        },
        "paper_execution_calibration": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "paper_execution_calibration_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "stale_artifact_sweeper_bot": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "stale_artifact_sweeper_bot_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "stale_artifact_reaper_bot": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "stale_artifact_reaper_bot_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "sql_link_service": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "sql_link_service_progress_latest.json",
                project_root / "governance" / "health" / "sql_link_service_latest.json",
            ],
            "max_age_minutes": 30.0,
            "required": True,
        },
        "sql_ingestion": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "jsonl_sql_ingestion_health_trading_latest.json",
                project_root
                / "governance"
                / "health"
                / "jsonl_sql_ingestion_health_latest.json",
                project_root
                / "governance"
                / "health"
                / "jsonl_sql_ingestion_health_data_latest.json",
                project_root
                / "governance"
                / "health"
                / "jsonl_sql_ingestion_health_governance_latest.json",
            ],
            "max_age_minutes": 30.0,
            "required": True,
        },
        "promotion_readiness": {
            "paths": [
                project_root
                / "governance"
                / "walk_forward"
                / "promotion_readiness_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "new_bot_graduation": {
            "paths": [
                project_root
                / "governance"
                / "walk_forward"
                / "new_bot_graduation_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "new_bot_admission_guard": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "new_bot_admission_guard_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "bot_support_owner_guard": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "bot_support_owner_guard_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "replay_hash_registry_guard": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "replay_hash_registry_guard_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "schema_migration_guard": {
            "paths": [project_root / "governance" / "migrations" / "latest.json"],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "retrain_schema_compatibility_guard": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "retrain_schema_compatibility_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "golden_replay_regression_guard": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "golden_replay_regression_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "cohort_drift_baseline_guard": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "cohort_drift_baseline_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "champion_challenger_probation_guard": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "champion_challenger_probation_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "champion_challenger_probation_action": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "champion_challenger_probation_action_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "retrain_lane_scheduler": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "retrain_lane_scheduler_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "promotion_packet": {
            "paths": [
                project_root
                / "governance"
                / "champion_challenger"
                / "promotion_packet_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "promotion_quality_gate": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "promotion_quality_gate_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(2.0),
            "required": False,
        },
        "retrain_artifact_freshness": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "retrain_artifact_freshness_latest.json"
            ],
            "max_age_minutes": _hours_to_minutes(24.0),
            "required": False,
        },
        "retrain_scorecard": {
            "paths": [
                project_root / "governance" / "health" / "retrain_scorecard_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(3.0),
            "required": False,
        },
        "official_macro_context_sync": {
            "paths": [
                project_root
                / "governance"
                / "health"
                / "official_macro_context_sync_latest.json"
            ],
            "max_age_minutes": _days_to_minutes(3.0),
            "required": False,
        },
        "live_macro_media": {
            "paths": [
                project_root / "governance" / "health" / "live_macro_media_status.json"
            ],
            "max_age_minutes": _days_to_minutes(3.0),
            "required": False,
        },
    }


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _parse_iso_utc(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _safety_artifact_state(name: str, artifact: Dict[str, Any]) -> bool | None:
    if not artifact.get("exists"):
        return None
    summary = (
        artifact.get("summary") if isinstance(artifact.get("summary"), dict) else {}
    )
    if name == "health_gates":
        return bool(
            artifact.get("ok") is not False
            and not bool(summary.get("hard_gate_triggered", False))
        )
    if name == "paper_400_ramp":
        return bool(
            artifact.get("ok") is not False
            and bool(summary.get("armed", False))
            and not list(summary.get("blocked_reasons") or [])
        )
    if name == "unattended_soak_readiness":
        return bool(artifact.get("ok") is True and summary.get("ok") is True)
    if name == "all_sleeves_launcher":
        return bool(
            artifact.get("ok") is not False
            and bool(summary.get("collection_fanout_ready", False))
            and bool(summary.get("paper_execution_ready", False))
        )
    return artifact.get("ok") if isinstance(artifact.get("ok"), bool) else None


def _safety_artifact_guarded_execution_reason(
    name: str, artifact: Dict[str, Any]
) -> str:
    if not artifact.get("exists"):
        return ""
    summary = (
        artifact.get("summary") if isinstance(artifact.get("summary"), dict) else {}
    )
    if name == "all_sleeves_launcher":
        readiness_status = str(summary.get("readiness_status") or "").strip().lower()
        collection_ready = bool(summary.get("collection_fanout_ready", False))
        paper_ready = bool(summary.get("paper_execution_ready", False))
        guarded_status = readiness_status in {
            "resident_execution_safety_hold",
            "guarded_collection",
            "guarded_ready",
            "paper_execution_safety_hold",
        }
        if (
            artifact.get("ok") is not False
            and collection_ready
            and not paper_ready
            and guarded_status
        ):
            return "collection_fanout_ready_with_paper_execution_guarded"
    return ""


def _safety_snapshot_contract(
    artifacts: Dict[str, Any], *, maximum_skew_seconds: float = 900.0
) -> Dict[str, Any]:
    names = (
        "health_gates",
        "paper_400_ramp",
        "unattended_soak_readiness",
        "all_sleeves_launcher",
    )
    rows: List[Dict[str, Any]] = []
    observed_times: List[datetime] = []
    current_times: List[datetime] = []
    current_states: List[bool] = []
    stale_sources: List[str] = []
    current_guarded_execution_sources: List[str] = []
    current_conflict_sources: List[str] = []
    for name in names:
        artifact = artifacts.get(name) if isinstance(artifacts.get(name), dict) else {}
        timestamp = _parse_iso_utc(artifact.get("timestamp_utc"))
        exists = bool(artifact.get("exists"))
        stale = bool(artifact.get("stale"))
        state = _safety_artifact_state(name, artifact)
        guarded_execution_reason = _safety_artifact_guarded_execution_reason(
            name, artifact
        )
        if exists and timestamp is not None:
            observed_times.append(timestamp)
            if not stale:
                current_times.append(timestamp)
                if state is not None:
                    current_states.append(state)
                    if state is False:
                        if guarded_execution_reason:
                            current_guarded_execution_sources.append(name)
                        else:
                            current_conflict_sources.append(name)
            else:
                stale_sources.append(name)
        rows.append(
            {
                "source": name,
                "exists": exists,
                "stale": stale,
                "timestamp_utc": timestamp.isoformat() if timestamp is not None else "",
                "age_minutes": artifact.get("age_minutes"),
                "safe_state": state,
                "included_in_current_agreement": bool(
                    exists and timestamp is not None and not stale and state is not None
                ),
                "guarded_execution_reason": guarded_execution_reason,
            }
        )

    observed_skew = (
        max((max(observed_times) - min(observed_times)).total_seconds(), 0.0)
        if len(observed_times) >= 2
        else 0.0
    )
    current_skew = (
        max((max(current_times) - min(current_times)).total_seconds(), 0.0)
        if len(current_times) >= 2
        else 0.0
    )
    raw_consistent = bool(
        len(observed_times) < 2 or observed_skew <= maximum_skew_seconds
    )
    current_sources_agree_safe = bool(len(current_states) >= 2 and all(current_states))
    current_sources_safe_or_execution_guarded = bool(
        len(current_states) >= 2
        and not current_conflict_sources
        and (current_sources_agree_safe or current_guarded_execution_sources)
    )
    managed_cadence_skew = bool(
        not raw_consistent and current_sources_safe_or_execution_guarded
    )
    consistent = bool(raw_consistent or managed_cadence_skew)
    return {
        "consistent": consistent,
        "raw_consistent": raw_consistent,
        "managed_cadence_skew": managed_cadence_skew,
        "managed_reason": (
            "nonstale_safety_sources_agree_while_independent_refresh_cadences_skew_timestamps"
            if managed_cadence_skew
            else ""
        ),
        "maximum_skew_seconds": float(maximum_skew_seconds),
        "observed_skew_seconds": round(float(observed_skew), 3),
        "current_source_skew_seconds": round(float(current_skew), 3),
        "source_count": len(observed_times),
        "current_source_count": len(current_times),
        "current_state_count": len(current_states),
        "current_sources_agree_safe": current_sources_agree_safe,
        "current_sources_safe_or_execution_guarded": current_sources_safe_or_execution_guarded,
        "current_guarded_execution_sources": current_guarded_execution_sources,
        "current_conflict_sources": current_conflict_sources,
        "stale_sources": stale_sources,
        "sources": list(names),
        "source_evidence": rows,
        "policy": "timestamp skew is managed when at least two nonstale safety sources independently agree safe, or when collection safety is current and the only unsafe source is an explicit paper-execution guard; contradictory current safety sources remain degraded",
    }


def _payload_timestamp(payload: Dict[str, Any], path: Path) -> datetime | None:
    for key in (
        "timestamp_utc",
        "updated_at_utc",
        "updated_at",
        "created_at",
        "ended_utc",
        "started_utc",
    ):
        ts = _parse_iso_utc(payload.get(key))
        if ts is not None:
            return ts
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except Exception:
        return None


def _pick_latest_artifact(paths: Iterable[Path]) -> Tuple[Path, Dict[str, Any]]:
    candidates: List[Tuple[float, Path, Dict[str, Any]]] = []
    fallback: Path | None = None
    for path in paths:
        fallback = fallback or path
        payload = _load_json(path)
        if not payload:
            continue
        ts = _payload_timestamp(payload, path)
        score = float(ts.timestamp()) if ts is not None else 0.0
        candidates.append((score, path, payload))
    if not candidates:
        return fallback or Path(""), {}
    candidates.sort(key=lambda row: row[0])
    _, path, payload = candidates[-1]
    return path, payload


def _infer_ok(payload: Dict[str, Any]) -> bool | None:
    if not isinstance(payload, dict) or not payload:
        return None
    raw_ok = payload.get("ok")
    if isinstance(raw_ok, bool):
        return raw_ok
    if "halt" in payload:
        return not bool(payload.get("halt"))
    if "hard_gate_triggered" in payload:
        return not bool(payload.get("hard_gate_triggered"))
    if "promote_ok" in payload:
        return bool(payload.get("promote_ok"))
    if "learning_ready" in payload:
        return bool(payload.get("learning_ready"))
    if "overall_status" in payload:
        status = str(payload.get("overall_status", "") or "").strip().lower()
        if status in {"ready", "ok"}:
            return True
        if status in {"blocked", "regressed"}:
            return False
    status = str(payload.get("status", "") or "").strip().lower()
    if status in {"ok", "healthy", "ready", "success", "pass"}:
        return True
    if status in {"error", "failed", "fail", "degraded", "stale"}:
        return False
    return None


def _infer_status(payload: Dict[str, Any], ok_value: bool | None) -> str:
    status = str(payload.get("status", "") or "").strip().lower()
    if status:
        return status
    if ok_value is True:
        return "ok"
    if ok_value is False:
        return "error"
    return "unknown"


def _artifact_summary(name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if name == "session_ready":
        checks = (
            payload.get("checks") if isinstance(payload.get("checks"), list) else []
        )
        return {
            "expected_profiles": (
                payload.get("expected_profiles")
                if isinstance(payload.get("expected_profiles"), list)
                else []
            ),
            "check_count": len(checks),
        }
    if name == "daily_auto_verify":
        return {
            "failed_checks": (
                payload.get("failed_checks")
                if isinstance(payload.get("failed_checks"), list)
                else []
            ),
            "operational_ok": bool(
                payload.get("operational_ok", payload.get("ok", False))
            ),
            "operational_failed_checks": (
                payload.get("operational_failed_checks")
                if isinstance(payload.get("operational_failed_checks"), list)
                else []
            ),
            "evidence_debt_checks": (
                payload.get("evidence_debt_checks")
                if isinstance(payload.get("evidence_debt_checks"), list)
                else []
            ),
            "completed_checks": int(payload.get("completed_checks", 0) or 0),
        }
    if name == "data_source_divergence":
        return {
            "ok": bool(payload.get("ok", False)),
            "worst_relative_spread": float(
                payload.get("worst_relative_spread", 0.0) or 0.0
            ),
        }
    if name == "source_verification":
        overall = (
            payload.get("overall") if isinstance(payload.get("overall"), dict) else {}
        )
        runtime = (
            payload.get("source_runtime_contract")
            if isinstance(payload.get("source_runtime_contract"), dict)
            else {}
        )
        confidence = (
            payload.get("source_confidence_summary")
            if isinstance(payload.get("source_confidence_summary"), dict)
            else {}
        )
        return {
            "overall_status": str(payload.get("overall_status", "") or ""),
            "source_evidence_grade": str(
                payload.get("source_evidence_grade", "") or ""
            ),
            "source_control_grade": str(payload.get("source_control_grade", "") or ""),
            "decision_critical_sources_ready": bool(
                runtime.get("decision_critical_sources_ready", False)
            ),
            "decision_critical_blocker_count": int(
                runtime.get("decision_critical_blocker_count", 0) or 0
            ),
            "decision_critical_blockers": (
                runtime.get("decision_critical_blockers")
                if isinstance(runtime.get("decision_critical_blockers"), list)
                else []
            ),
            "decision_context_debt": (
                runtime.get("decision_context_debt")
                if isinstance(runtime.get("decision_context_debt"), list)
                else []
            ),
            "optional_enrichment_debt": (
                runtime.get("optional_enrichment_debt")
                if isinstance(runtime.get("optional_enrichment_debt"), list)
                else []
            ),
            "unverified_sources": (
                overall.get("unverified_sources")
                if isinstance(overall.get("unverified_sources"), list)
                else []
            ),
            "low_confidence_sources": (
                confidence.get("low_confidence_sources")
                if isinstance(confidence.get("low_confidence_sources"), list)
                else []
            ),
        }
    if name == "execution_queue_stress":
        return {
            "ok": bool(payload.get("ok", False)),
            "samples": int(payload.get("samples", 0) or 0),
            "queue_breach_rate": float(payload.get("queue_breach_rate", 0.0) or 0.0),
            "max_queue_depth_seen": int(payload.get("max_queue_depth_seen", 0) or 0),
        }
    if name == "sqlite_maintenance":
        return {
            "ok": bool(payload.get("ok", False)),
            "timed_out": bool(payload.get("timed_out", False)),
            "checkpoint_only": bool(payload.get("checkpoint_only", False)),
            "running": bool(payload.get("running", False)),
            "current_step": str(payload.get("current_step", "") or ""),
        }
    if name == "snapshot_coverage":
        return {
            "ok": bool(payload.get("ok", False)),
            "operational_ok": bool(
                payload.get("operational_ok", payload.get("ok", False))
            ),
            "evidence_ready": bool(
                payload.get("evidence_ready", payload.get("ok", False))
            ),
            "coverage_ratio": float(payload.get("coverage_ratio", 0.0) or 0.0),
            "missing_file_count": int(payload.get("missing_file_count", 0) or 0),
        }
    if name == "health_gates":
        return {
            "data_quality_score": float(payload.get("data_quality_score", 0.0) or 0.0),
            "hard_gate_triggered": bool(payload.get("hard_gate_triggered", False)),
            "inputs": (
                payload.get("inputs") if isinstance(payload.get("inputs"), dict) else {}
            ),
        }
    if name == "all_sleeves_launcher":
        readiness = (
            payload.get("launcher_readiness_contract")
            if isinstance(payload.get("launcher_readiness_contract"), dict)
            else {}
        )
        return {
            "overall_status": str(payload.get("overall_status") or ""),
            "expected_job_count": int(payload.get("expected_job_count", 0) or 0),
            "running_job_count": int(payload.get("running_job_count", 0) or 0),
            "collection_fanout_ready": bool(
                readiness.get("collection_fanout_ready", False)
            ),
            "paper_execution_ready": bool(
                readiness.get("paper_execution_ready", False)
            ),
            "readiness_status": str(readiness.get("readiness_status") or ""),
            "execution_attention": (
                readiness.get("execution_attention")
                if isinstance(readiness.get("execution_attention"), list)
                else []
            ),
            "broker_auth_epoch": (
                payload.get("broker_auth_epoch")
                if isinstance(payload.get("broker_auth_epoch"), dict)
                else {}
            ),
        }
    if name == "paper_400_ramp":
        return {
            "overall_status": str(
                payload.get("overall_status") or payload.get("status") or ""
            ),
            "armed": bool(payload.get("armed", False)),
            "blocked_reasons": (
                payload.get("blocked_reasons")
                if isinstance(payload.get("blocked_reasons"), list)
                else []
            ),
        }
    if name == "unattended_soak_readiness":
        return {
            "overall_status": str(
                payload.get("overall_status") or payload.get("status") or ""
            ),
            "ok": bool(payload.get("ok", False)),
            "grade": str(payload.get("grade") or ""),
        }
    if name == "data_collection_observation_rollup":
        return {
            "overall_status": str(payload.get("overall_status") or ""),
            "collector_count": int(payload.get("collector_count", 0) or 0),
            "total_observations": int(payload.get("total_observations", 0) or 0),
            "collection_coverage_score": float(
                payload.get("collection_coverage_score", 0.0) or 0.0
            ),
            "data_quality_score": float(payload.get("data_quality_score", 0.0) or 0.0),
        }
    if name == "global_killswitch":
        reasons = (
            payload.get("reasons") if isinstance(payload.get("reasons"), list) else []
        )
        return {
            "halt": bool(payload.get("halt", False)),
            "action": str(payload.get("action", "") or ""),
            "reason_count": len(reasons),
            "reasons": reasons,
        }
    if name == "runtime_access_mode":
        return {
            "mode": str(payload.get("mode", "") or ""),
            "ml_backend": str(payload.get("ml_backend", "") or ""),
            "portable_enabled": bool(payload.get("portable_enabled", False)),
            "backend_contract": (
                payload.get("backend_contract")
                if isinstance(payload.get("backend_contract"), dict)
                else {}
            ),
            "detected_backends": (
                payload.get("detected_backends")
                if isinstance(payload.get("detected_backends"), dict)
                else {}
            ),
        }
    if name == "apple_silicon_profile":
        hardware = (
            payload.get("hardware") if isinstance(payload.get("hardware"), dict) else {}
        )
        return {
            "applied_tier": str(payload.get("applied_tier", "") or ""),
            "detected_tier": str(payload.get("detected_tier", "") or ""),
            "chip": str(hardware.get("chip", "") or ""),
            "memory_gb": float(hardware.get("memory_gb", 0.0) or 0.0),
            "override_exists": bool(payload.get("override_exists", False)),
        }
    if name == "memory_efficiency_control":
        memory_snapshot = (
            payload.get("memory_snapshot")
            if isinstance(payload.get("memory_snapshot"), dict)
            else {}
        )
        return {
            "overall_status": str(payload.get("overall_status", "") or ""),
            "recommended_profile": str(payload.get("recommended_profile", "") or ""),
            "memory_pressure_state": str(
                memory_snapshot.get("memory_pressure_state", "") or ""
            ),
            "memory_pressure_kind": str(
                memory_snapshot.get("memory_pressure_kind", "") or ""
            ),
            "swap_used_gb": float(memory_snapshot.get("swap_used_gb", 0.0) or 0.0),
        }
    if name == "training_report":
        summary = (
            payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        )
        return {
            "overall_status": str(payload.get("overall_status", "") or ""),
            "confirmed_training_success": bool(
                summary.get("confirmed_training_success", False)
            ),
            "target_count": int(summary.get("target_count", 0) or 0),
            "trained_count": int(summary.get("trained_count", 0) or 0),
            "blocking_reasons": (
                payload.get("blocking_reasons")
                if isinstance(payload.get("blocking_reasons"), list)
                else []
            ),
        }
    if name == "nightly_resilience":
        metrics = (
            payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        )
        return {
            "ok": bool(payload.get("ok", False)),
            "failed_checks": (
                payload.get("failed_checks")
                if isinstance(payload.get("failed_checks"), list)
                else []
            ),
            "watchdog_process_count": int(
                metrics.get("watchdog_process_count", 0) or 0
            ),
            "shadow_loop_process_count": int(
                metrics.get("shadow_loop_process_count", 0) or 0
            ),
            "watchdog_log_age_minutes": float(
                metrics.get("watchdog_log_age_minutes", 0.0) or 0.0
            ),
        }
    if name == "training_quality_control":
        supportability = (
            payload.get("supportability")
            if isinstance(payload.get("supportability"), dict)
            else {}
        )
        return {
            "overall_status": str(payload.get("overall_status", "") or ""),
            "training_quality_score": float(
                payload.get("training_quality_score", 0.0) or 0.0
            ),
            "top_priorities": (
                payload.get("top_priorities")
                if isinstance(payload.get("top_priorities"), list)
                else []
            ),
            "active_supportability_score": float(
                supportability.get("active_supportability_score", 0.0) or 0.0
            ),
            "implemented_improvement_count": int(
                payload.get("implemented_improvement_count", 0) or 0
            ),
        }
    if name == "hdf5_training_cache":
        cache = payload.get("cache") if isinstance(payload.get("cache"), dict) else {}
        freshness = (
            payload.get("freshness_gate")
            if isinstance(payload.get("freshness_gate"), dict)
            else {}
        )
        schema = (
            payload.get("schema_validation")
            if isinstance(payload.get("schema_validation"), dict)
            else {}
        )
        benchmark = (
            payload.get("performance_benchmark")
            if isinstance(payload.get("performance_benchmark"), dict)
            else {}
        )
        return {
            "overall_status": str(payload.get("overall_status", "") or ""),
            "fresh": bool(freshness.get("fresh", False)),
            "schema_ok": bool(schema.get("ok", False)),
            "row_count": int(cache.get("row_count", 0) or 0),
            "feature_count": int(cache.get("feature_count", 0) or 0),
            "sequence_count": int(cache.get("sequence_count", 0) or 0),
            "h5_size_bytes": int(cache.get("h5_size_bytes", 0) or 0),
            "benchmark_status": str(benchmark.get("status", "") or ""),
            "speedup_ratio": float(benchmark.get("speedup_ratio", 0.0) or 0.0),
        }
    if name == "ingestion_storage_control":
        backpressure = (
            payload.get("backpressure")
            if isinstance(payload.get("backpressure"), dict)
            else {}
        )
        storage = (
            payload.get("storage") if isinstance(payload.get("storage"), dict) else {}
        )
        queue_watermarks = (
            payload.get("queue_watermarks")
            if isinstance(payload.get("queue_watermarks"), dict)
            else {}
        )
        writer_shedding = (
            payload.get("writer_shedding")
            if isinstance(payload.get("writer_shedding"), dict)
            else {}
        )
        route_verification = (
            payload.get("external_route_verification")
            if isinstance(payload.get("external_route_verification"), dict)
            else {}
        )
        return {
            "overall_status": str(payload.get("overall_status", "") or ""),
            "severity": str(payload.get("severity", "") or ""),
            "pressure_index": float(payload.get("pressure_index", 0.0) or 0.0),
            "recommended_operating_mode": str(
                payload.get("recommended_operating_mode", "") or ""
            ),
            "core_pending_lines": int(backpressure.get("core_pending_lines", 0) or 0),
            "total_pending_lines": int(backpressure.get("total_pending_lines", 0) or 0),
            "estimated_core_drain_minutes": backpressure.get(
                "estimated_core_drain_minutes"
            ),
            "estimated_total_drain_minutes": backpressure.get(
                "estimated_total_drain_minutes"
            ),
            "retention_debt_gb": float(storage.get("retention_debt_gb", 0.0) or 0.0),
            "backlog_quarantine_status": str(
                storage.get("backlog_quarantine_status", "") or ""
            ),
            "backlog_quarantine_candidate_files": int(
                storage.get("backlog_quarantine_candidate_files", 0) or 0
            ),
            "backlog_quarantine_moved_files": int(
                storage.get("backlog_quarantine_moved_files", 0) or 0
            ),
            "queue_watermarks_overall_status": str(
                queue_watermarks.get("overall_status", "") or ""
            ),
            "writer_shedding_level": str(writer_shedding.get("level", "") or ""),
            "writer_shedding_active": bool(writer_shedding.get("active", False)),
            "external_route_verification_state": str(
                route_verification.get("verification_state", "") or ""
            ),
        }
    if name == "ingestion_storage_governor":
        sql_primary_db = (
            payload.get("sql_primary_db")
            if isinstance(payload.get("sql_primary_db"), dict)
            else {}
        )
        throttles = (
            payload.get("throttle_controls")
            if isinstance(payload.get("throttle_controls"), dict)
            else {}
        )
        queue_watermarks = (
            payload.get("queue_watermarks")
            if isinstance(payload.get("queue_watermarks"), dict)
            else {}
        )
        writer_shedding = (
            payload.get("writer_shedding")
            if isinstance(payload.get("writer_shedding"), dict)
            else {}
        )
        return {
            "profile": str(payload.get("profile", "") or ""),
            "route_drift": bool(sql_primary_db.get("route_drift", False)),
            "deferred_files_budget": int(
                throttles.get("deferred_files_budget", 0) or 0
            ),
            "cold_files_budget": int(throttles.get("cold_files_budget", 0) or 0),
            "queue_watermarks_overall_status": str(
                queue_watermarks.get("overall_status", "") or ""
            ),
            "writer_shedding_level": str(writer_shedding.get("level", "") or ""),
            "writer_shedding_active": bool(writer_shedding.get("active", False)),
        }
    if name == "external_backlog_drain":
        drain_overrides = (
            payload.get("drain_overrides")
            if isinstance(payload.get("drain_overrides"), dict)
            else {}
        )
        off_hours = (
            payload.get("off_hours_window")
            if isinstance(payload.get("off_hours_window"), dict)
            else {}
        )
        follow_through = (
            payload.get("follow_through")
            if isinstance(payload.get("follow_through"), dict)
            else {}
        )
        return {
            "overall_status": str(payload.get("overall_status", "") or ""),
            "recommended_now": bool(payload.get("recommended_now", False)),
            "material_drain_recommended": bool(
                payload.get("material_drain_recommended", False)
            ),
            "apply_executed": bool(payload.get("apply_executed", False)),
            "writer_busy": bool(payload.get("writer_busy", False)),
            "off_hours_active": bool(off_hours.get("active", False)),
            "aged_candidate_files": int(payload.get("aged_candidate_files", 0) or 0),
            "deferred_files_budget": int(
                drain_overrides.get("deferred_files_budget", 0) or 0
            ),
            "cold_files_budget": int(drain_overrides.get("cold_files_budget", 0) or 0),
            "follow_through_status": str(follow_through.get("status", "") or ""),
            "follow_through_progress_state": str(
                follow_through.get("progress_state", "") or ""
            ),
            "follow_through_progress_observed": bool(
                follow_through.get("progress_observed", False)
            ),
        }
    if name == "external_backlog_retry_bot":
        drain_result = (
            payload.get("drain_result")
            if isinstance(payload.get("drain_result"), dict)
            else {}
        )
        return {
            "overall_status": str(payload.get("overall_status", "") or ""),
            "actionable": bool(payload.get("actionable", False)),
            "backlog_needed": bool(payload.get("backlog_needed", False)),
            "drain_follow_through_status": str(
                drain_result.get("follow_through_status", "") or ""
            ),
            "drain_follow_through_progress_state": str(
                drain_result.get("follow_through_progress_state", "") or ""
            ),
            "follow_through_attempts": int(
                drain_result.get("follow_through_attempts", 0) or 0
            ),
        }
    if name == "platform_control_plane":
        readiness = (
            payload.get("institutional_readiness")
            if isinstance(payload.get("institutional_readiness"), dict)
            else {}
        )
        weakest = (
            readiness.get("weakest_domains")
            if isinstance(readiness.get("weakest_domains"), list)
            else []
        )
        return {
            "overall_status": str(readiness.get("overall_status", "") or ""),
            "overall_score": float(readiness.get("overall_score", 0.0) or 0.0),
            "top_priorities": (
                readiness.get("top_priorities")
                if isinstance(readiness.get("top_priorities"), list)
                else []
            ),
            "weakest_domains": [
                str((row or {}).get("slug") or "")
                for row in weakest
                if isinstance(row, dict) and str((row or {}).get("slug") or "").strip()
            ],
            "domain_count": int(readiness.get("domain_count", 0) or 0),
        }
    if name == "platform_intelligence_expansion":
        sections = (
            payload.get("sections") if isinstance(payload.get("sections"), dict) else {}
        )
        dashboard = (
            sections.get("professional_system_dashboard")
            if isinstance(sections.get("professional_system_dashboard"), dict)
            else {}
        )
        pressure = (
            payload.get("pressure_snapshot")
            if isinstance(payload.get("pressure_snapshot"), dict)
            else {}
        )
        return {
            "overall_status": str(payload.get("overall_status", "") or ""),
            "bot_count": int(payload.get("bot_count", 0) or 0),
            "sleeve_count": int(payload.get("sleeve_count", 0) or 0),
            "expansion_count": int(payload.get("expansion_count", 0) or 0),
            "section_count": int(dashboard.get("section_count", 0) or 0),
            "swap_tier": str(pressure.get("swap_tier", "") or ""),
            "host_saturation_score": float(
                pressure.get("host_saturation_score", 0.0) or 0.0
            ),
            "top_actions": (
                payload.get("top_actions")
                if isinstance(payload.get("top_actions"), list)
                else []
            ),
        }
    if name == "bot_founder_dna_lineage":
        summary = (
            payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        )
        apply_result = (
            payload.get("apply_result")
            if isinstance(payload.get("apply_result"), dict)
            else {}
        )
        return {
            "overall_status": str(payload.get("overall_status", "") or ""),
            "founder_bot_id": str(summary.get("founder_bot_id", "") or ""),
            "founder_dna_version": str(summary.get("founder_dna_version", "") or ""),
            "total_bots": int(summary.get("total_bots", 0) or 0),
            "explicit_founder_dna_count": int(
                summary.get("explicit_founder_dna_count", 0) or 0
            ),
            "missing_founder_dna_count": int(
                summary.get("missing_founder_dna_count", 0) or 0
            ),
            "coverage_ratio": float(summary.get("coverage_ratio", 0.0) or 0.0),
            "all_have_founder_dna": bool(summary.get("all_have_founder_dna", False)),
            "changed_rows": int(apply_result.get("changed_rows", 0) or 0),
            "top_actions": (
                payload.get("top_actions")
                if isinstance(payload.get("top_actions"), list)
                else []
            ),
        }
    if name == "ingestion_priority_queue":
        lane_counts = (
            payload.get("lane_counts")
            if isinstance(payload.get("lane_counts"), dict)
            else {}
        )
        return {
            "queue_depth": int(payload.get("queue_depth", 0) or 0),
            "items_synced": int(payload.get("items_synced", 0) or 0),
            "core_pending_lines": int(
                ((lane_counts.get("core") or {}).get("pending_lines", 0)) or 0
            ),
            "event_count": int(payload.get("event_count", 0) or 0),
        }
    if name == "storage_split_brain_reconciler":
        summary = (
            payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        )
        return {
            "conflict_files": int(summary.get("conflict_files", 0) or 0),
            "unresolved_conflicts": int(summary.get("unresolved_conflicts", 0) or 0),
            "force_failback_eligible": bool(
                summary.get("force_failback_eligible", False)
            ),
        }
    if name == "storage_resilience_control":
        return {
            "overall_status": str(payload.get("overall_status", "") or ""),
            "resilience_score": int(payload.get("resilience_score", 0) or 0),
            "restore_drill_fresh": bool(payload.get("restore_drill_fresh", False)),
            "unresolved_split_brain_conflicts": int(
                payload.get("unresolved_split_brain_conflicts", 0) or 0
            ),
        }
    if name == "daily_verify_auto_remediation_bot":
        return {
            "overall_status": str(payload.get("overall_status", "") or ""),
            "resolved_checks": (
                payload.get("resolved_checks")
                if isinstance(payload.get("resolved_checks"), list)
                else []
            ),
            "unresolved_checks": (
                payload.get("unresolved_checks")
                if isinstance(payload.get("unresolved_checks"), list)
                else []
            ),
        }
    if name == "operator_cockpit":
        return {
            "overall_status": str(payload.get("overall_status", "") or ""),
            "recommended_actions": (
                payload.get("recommended_actions")
                if isinstance(payload.get("recommended_actions"), list)
                else []
            ),
        }
    if name == "stale_artifact_sweeper_bot":
        summary = (
            payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        )
        return {
            "busy": bool(payload.get("busy", False)),
            "reason": str(payload.get("reason", "") or ""),
            "candidate_files": int(summary.get("candidate_files", 0) or 0),
            "staged_files": int(summary.get("staged_files", 0) or 0),
            "staged_bytes": int(summary.get("staged_bytes", 0) or 0),
            "delete_errors": int(summary.get("delete_errors", 0) or 0),
        }
    if name == "stale_artifact_reaper_bot":
        summary = (
            payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        )
        return {
            "busy": bool(payload.get("busy", False)),
            "reason": str(payload.get("reason", "") or ""),
            "candidate_files": int(summary.get("candidate_files", 0) or 0),
            "deleted_files": int(summary.get("deleted_files", 0) or 0),
            "deleted_bytes": int(summary.get("deleted_bytes", 0) or 0),
            "delete_errors": int(summary.get("delete_errors", 0) or 0),
        }
    if name == "sql_ingestion":
        sqlite = (
            payload.get("sqlite") if isinstance(payload.get("sqlite"), dict) else {}
        )
        return {
            "pending_lines": int(sqlite.get("pending_lines", 0) or 0),
            "oldest_uningested_age_seconds": float(
                sqlite.get("oldest_uningested_age_seconds", 0.0) or 0.0
            ),
            "invalid_lines": int(sqlite.get("invalid", 0) or 0),
            "files_discovered": int(payload.get("files_discovered", 0) or 0),
        }
    if name == "promotion_readiness":
        return {
            "promote_ok": bool(payload.get("promote_ok", False)),
            "considered_bots": int(payload.get("considered_bots", 0) or 0),
            "failed_bots": int(payload.get("failed_bots", 0) or 0),
            "fail_share": float(payload.get("fail_share", 0.0) or 0.0),
        }
    if name == "new_bot_graduation":
        return {
            "ok": bool(payload.get("ok", False)),
            "mature_bots": int(
                ((payload.get("maturity") or {}).get("mature_bots", 0)) or 0
            ),
            "immature_active_count": int(payload.get("immature_active_count", 0) or 0),
        }
    if name == "new_bot_admission_guard":
        global_prereqs = (
            payload.get("global_prerequisites")
            if isinstance(payload.get("global_prerequisites"), dict)
            else {}
        )
        return {
            "ok": bool(payload.get("ok", False)),
            "candidate_bot_count": int(payload.get("candidate_bot_count", 0) or 0),
            "blocking_candidate_count": int(
                payload.get("blocking_candidate_count", 0) or 0
            ),
            "feature_store_manifest_ready": bool(
                global_prereqs.get("feature_store_manifest_ready", False)
            ),
            "replay_hash_registry_ready": bool(
                global_prereqs.get("replay_hash_registry_ready", False)
            ),
        }
    if name == "bot_support_owner_guard":
        summary = (
            payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        )
        return {
            "ok": bool(payload.get("ok", False)),
            "in_scope_bot_count": int(summary.get("in_scope_bot_count", 0) or 0),
            "blocking_bot_count": int(payload.get("blocking_bot_count", 0) or 0),
            "covered_bot_count": int(summary.get("covered_bot_count", 0) or 0),
        }
    if name == "replay_hash_registry_guard":
        return {
            "ok": bool(payload.get("ok", False)),
            "failed_checks": (
                payload.get("failed_checks")
                if isinstance(payload.get("failed_checks"), list)
                else []
            ),
        }
    if name == "schema_migration_guard":
        summary = (
            payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        )
        return {
            "ok": bool(payload.get("ok", False)),
            "overall_status": str(payload.get("overall_status", "") or ""),
            "missing_contracts": int(summary.get("missing_contracts", 0) or 0),
            "legacy_unversioned_contracts": int(
                summary.get("legacy_unversioned_contracts", 0) or 0
            ),
        }
    if name == "retrain_schema_compatibility_guard":
        return {
            "ok": bool(payload.get("ok", False)),
            "overall_status": str(payload.get("overall_status", "") or ""),
            "baseline_ready": bool(payload.get("baseline_ready", False)),
            "drifted_fields": (
                payload.get("drifted_fields")
                if isinstance(payload.get("drifted_fields"), list)
                else []
            ),
        }
    if name == "golden_replay_regression_guard":
        return {
            "ok": bool(payload.get("ok", False)),
            "case_count": int(payload.get("case_count", 0) or 0),
            "failed_case_count": int(payload.get("failed_case_count", 0) or 0),
            "failed_cases": (
                payload.get("failed_cases")
                if isinstance(payload.get("failed_cases"), list)
                else []
            ),
        }
    if name == "cohort_drift_baseline_guard":
        summary = (
            payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        )
        return {
            "ok": bool(payload.get("ok", False)),
            "overall_status": str(payload.get("overall_status", "") or ""),
            "cohort_count": int(summary.get("cohort_count", 0) or 0),
            "severe_cohort_count": int(summary.get("severe_cohort_count", 0) or 0),
        }
    if name == "champion_challenger_probation_guard":
        return {
            "ok": bool(payload.get("ok", False)),
            "rollback_required": bool(payload.get("rollback_required", False)),
            "probation_cohort_count": int(
                payload.get("probation_cohort_count", 0) or 0
            ),
            "failed_checks": (
                payload.get("failed_checks")
                if isinstance(payload.get("failed_checks"), list)
                else []
            ),
        }
    if name == "champion_challenger_probation_action":
        return {
            "ok": bool(payload.get("ok", False)),
            "action_required": bool(payload.get("action_required", False)),
            "action": str(payload.get("action", "") or ""),
            "promotion_frozen": bool(payload.get("promotion_frozen", False)),
        }
    if name == "retrain_lane_scheduler":
        summary = (
            payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        )
        return {
            "ok": bool(payload.get("ok", False)),
            "candidate_count": int(summary.get("candidate_count", 0) or 0),
            "selected_count": int(summary.get("selected_count", 0) or 0),
            "lane_count": int(summary.get("lane_count", 0) or 0),
        }
    if name == "promotion_packet":
        scope = (
            payload.get("promotion_scope")
            if isinstance(payload.get("promotion_scope"), dict)
            else {}
        )
        signature = (
            payload.get("signature")
            if isinstance(payload.get("signature"), dict)
            else {}
        )
        return {
            "ok": bool(payload.get("ok", False)),
            "ready_for_committee": bool(payload.get("ready_for_committee", False)),
            "trained_target_count": len(
                scope.get("trained_bot_ids")
                if isinstance(scope.get("trained_bot_ids"), list)
                else []
            ),
            "master_update_status": str(scope.get("master_update_status", "") or ""),
            "signature_verified": bool(signature.get("verified", False)),
        }
    if name == "promotion_quality_gate":
        return {
            "ok": bool(payload.get("ok", False)),
            "failed_checks": (
                payload.get("failed_checks")
                if isinstance(payload.get("failed_checks"), list)
                else []
            ),
        }
    if name == "sql_link_service":
        return {
            "running": bool(payload.get("running", False)),
            "current_step": str(payload.get("current_step", "") or ""),
            "completed_shard_count": int(payload.get("completed_shard_count", 0) or 0),
            "completed_merge_count": int(payload.get("completed_merge_count", 0) or 0),
            "merged_rows_this_cycle": int(
                payload.get("merged_rows_this_cycle", 0) or 0
            ),
        }
    if name == "retrain_artifact_freshness":
        return {
            "failed_checks": (
                payload.get("failed_checks")
                if isinstance(payload.get("failed_checks"), list)
                else []
            ),
            "failure_categories": (
                payload.get("failure_categories")
                if isinstance(payload.get("failure_categories"), dict)
                else {}
            ),
            "availability_failed_checks": (
                payload.get("availability_failed_checks")
                if isinstance(payload.get("availability_failed_checks"), list)
                else []
            ),
            "freshness_failed_checks": (
                payload.get("freshness_failed_checks")
                if isinstance(payload.get("freshness_failed_checks"), list)
                else []
            ),
            "sample_sufficiency_failed_checks": (
                payload.get("sample_sufficiency_failed_checks")
                if isinstance(payload.get("sample_sufficiency_failed_checks"), list)
                else []
            ),
            "artifact_health_failed_checks": (
                payload.get("artifact_health_failed_checks")
                if isinstance(payload.get("artifact_health_failed_checks"), list)
                else []
            ),
            "max_age_minutes": float(payload.get("max_age_minutes", 0.0) or 0.0),
        }
    if name == "retrain_scorecard":
        outcomes = (
            payload.get("target_outcomes")
            if isinstance(payload.get("target_outcomes"), list)
            else []
        )
        status_counts = (
            payload.get("status_counts")
            if isinstance(payload.get("status_counts"), dict)
            else {}
        )
        trained_count = int(status_counts.get("trained", 0) or 0)
        return {
            "target_count": int(payload.get("target_count", 0) or 0),
            "trained_count": trained_count,
            "failure_count": int(payload.get("failure_count", 0) or 0),
            "master_update_status": str(payload.get("master_update_status", "") or ""),
            "outcome_count": len(outcomes),
        }
    if name == "official_macro_context_sync":
        sources = (
            payload.get("sources") if isinstance(payload.get("sources"), dict) else {}
        )
        return {
            "source_count": len(sources),
            "ok_sources": sum(
                1
                for row in sources.values()
                if isinstance(row, dict) and bool(row.get("ok", False))
            ),
        }
    if name == "live_macro_media":
        return {
            "source": str(payload.get("source", "") or ""),
            "speaker": str(payload.get("speaker", "") or ""),
            "learning_ready": bool(payload.get("learning_ready", False)),
            "training_feature_count": int(
                payload.get("training_feature_count", 0) or 0
            ),
        }
    if name == "teacher_quality_guard":
        summary = (
            payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        )
        return {
            "overall_status": str(payload.get("overall_status", "") or ""),
            "qualified_teacher_count": int(
                summary.get("qualified_teacher_count", 0) or 0
            ),
            "elite_teacher_count": int(summary.get("elite_teacher_count", 0) or 0),
        }
    if name == "strategy_generation_control":
        summary = (
            payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        )
        resource = (
            payload.get("resource_envelope")
            if isinstance(payload.get("resource_envelope"), dict)
            else {}
        )
        hardening = (
            payload.get("hardening")
            if isinstance(payload.get("hardening"), dict)
            else {}
        )
        return {
            "overall_status": str(payload.get("overall_status", "") or ""),
            "strategy_generation": int(payload.get("strategy_generation", 0) or 0),
            "eligible_parent_count": int(summary.get("eligible_parent_count", 0) or 0),
            "active_offspring_count": int(
                summary.get("active_offspring_count", 0) or 0
            ),
            "retained_offspring_count": int(
                summary.get("retained_offspring_count", 0) or 0
            ),
            "stale_training_count": int(summary.get("stale_training_count", 0) or 0),
            "execution_authority_count": int(
                summary.get("execution_authority_count", 0) or 0
            ),
            "candidate_integrity_violation_count": int(
                summary.get("candidate_integrity_violation_count", 0) or 0
            ),
            "event_chain_ok": bool((payload.get("event_chain") or {}).get("ok", False)),
            "signed_event_chain": bool(hardening.get("signed_event_chain", False)),
            "hardening_grade": str(hardening.get("grade", "") or ""),
            "max_active_offspring": int(resource.get("max_active_offspring", 0) or 0),
            "max_retained_offspring": int(
                resource.get("max_retained_offspring", 0) or 0
            ),
            "max_lineage_depth": int(resource.get("max_lineage_depth", 0) or 0),
            "max_concurrent_training_jobs": int(
                resource.get("max_concurrent_training_jobs", 0) or 0
            ),
            "candidate_artifact_bytes": int(
                resource.get("candidate_artifact_bytes", 0) or 0
            ),
        }
    if name == "bot_quality_autopilot":
        teacher_summary = (
            payload.get("teacher_summary")
            if isinstance(payload.get("teacher_summary"), dict)
            else {}
        )
        blockers = (
            payload.get("quality_blockers")
            if isinstance(payload.get("quality_blockers"), dict)
            else {}
        )
        training_guard = (
            payload.get("training_candidate_guard")
            if isinstance(payload.get("training_candidate_guard"), dict)
            else {}
        )
        return {
            "overall_status": str(payload.get("overall_status", "") or ""),
            "quality_queue": len(payload.get("quality_upgrade_queue") or []),
            "qualified_teacher_count": int(
                teacher_summary.get("qualified_teacher_count", 0) or 0
            ),
            "teacher_quality_status": str(
                teacher_summary.get("teacher_quality_status", "") or ""
            ),
            "coverage_shortfall_bots": int(
                blockers.get("coverage_shortfall_bots", 0) or 0
            ),
            "rerouted_targeted_retrain_count": int(
                training_guard.get("rerouted_count", 0) or 0
            ),
            "minimum_sample_count": int(
                training_guard.get("minimum_sample_count", 0) or 0
            ),
        }
    if name == "infrastructure_autofix_bot":
        return {
            "overall_status": str(payload.get("overall_status", "") or ""),
            "applyable_repair_count": int(
                payload.get("applyable_repair_count", 0) or 0
            ),
            "operator_followups": len(payload.get("operator_followups") or []),
        }
    if name == "live_runtime_separation_control":
        pressure = (
            payload.get("shared_host_pressure")
            if isinstance(payload.get("shared_host_pressure"), dict)
            else {}
        )
        live_plane = (
            payload.get("live_plane")
            if isinstance(payload.get("live_plane"), dict)
            else {}
        )
        return {
            "overall_status": str(payload.get("overall_status", "") or ""),
            "contention_score": int(pressure.get("contention_score", 0) or 0),
            "live_ready": bool(live_plane.get("ready", False)),
        }
    if name == "rolling_restart_controller":
        return {
            "overall_status": str(payload.get("overall_status", "") or ""),
            "restart_due": bool(payload.get("restart_due", False)),
            "recommended_scope": str(payload.get("recommended_scope", "") or ""),
        }
    if name == "auth_lease_manager":
        return {
            "overall_status": str(payload.get("overall_status", "") or ""),
            "lease_state": str(payload.get("lease_state", "") or ""),
            "expires_in_seconds": float(
                (
                    (payload.get("lease_budget") or {}).get("expires_in_seconds", 0.0)
                    or 0.0
                )
            ),
        }
    if name == "blackstart_recovery":
        return {
            "overall_status": str(payload.get("overall_status", "") or ""),
            "stage_count": len(payload.get("stages") or []),
        }
    if name == "sleeve_isolation_guard":
        return {
            "overall_status": str(payload.get("overall_status", "") or ""),
            "isolated_lane_count": int(
                (
                    (payload.get("sleeve_matrix") or {}).get("isolated_lane_count", 0)
                    or 0
                )
            ),
            "quarantine_events": int(
                ((payload.get("quarantine_pressure") or {}).get("events", 0) or 0)
            ),
        }
    if name == "artifact_freshness_slo":
        summary = (
            payload.get("sla_summary")
            if isinstance(payload.get("sla_summary"), dict)
            else {}
        )
        return {
            "overall_status": str(payload.get("overall_status", "") or ""),
            "stale_required": int(summary.get("stale_required", 0) or 0),
            "stale_optional": int(summary.get("stale_optional", 0) or 0),
        }
    if name == "system_role_contract":
        summary = (
            payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        )
        return {
            "overall_status": str(payload.get("overall_status", "") or ""),
            "grade": str(payload.get("grade", "") or ""),
            "operating_mode": str(payload.get("operating_mode", "") or ""),
            "role_count": int(summary.get("role_count", 0) or 0),
            "component_count": int(summary.get("component_count", 0) or 0),
            "state_domain_count": int(summary.get("state_domain_count", 0) or 0),
            "exclusive_action_count": int(
                summary.get("exclusive_action_count", 0) or 0
            ),
            "operating_plane_count": int(summary.get("operating_plane_count", 0) or 0),
            "classified_action_count": int(
                summary.get("classified_action_count", 0) or 0
            ),
            "action_lease_count": int(summary.get("action_lease_count", 0) or 0),
            "escalation_route_count": int(
                summary.get("escalation_route_count", 0) or 0
            ),
            "registry_role_coverage_ratio": float(
                summary.get("registry_role_coverage_ratio", 0.0) or 0.0
            ),
            "authority_conflict_count": int(
                summary.get("authority_conflict_count", 0) or 0
            ),
            "blockers": (
                payload.get("blockers")
                if isinstance(payload.get("blockers"), list)
                else []
            ),
        }
    if name == "bot_organization_control":
        setup = (
            payload.get("bot_setup_summary")
            if isinstance(payload.get("bot_setup_summary"), dict)
            else {}
        )
        setup_hardening = (
            setup.get("hardening") if isinstance(setup.get("hardening"), dict) else {}
        )
        tripwire = (
            payload.get("tripwire_summary")
            if isinstance(payload.get("tripwire_summary"), dict)
            else {}
        )
        tripwire_hardening = (
            tripwire.get("hardening")
            if isinstance(tripwire.get("hardening"), dict)
            else {}
        )
        return {
            "overall_status": str(payload.get("overall_status", "") or ""),
            "grade": str(payload.get("grade", "") or ""),
            "structural_grade": str(payload.get("structural_grade", "") or ""),
            "registry_bot_count": int(payload.get("registry_bot_count", 0) or 0),
            "organized_bot_count": int(payload.get("organized_bot_count", 0) or 0),
            "organization_coverage_ratio": float(
                payload.get("organization_coverage_ratio", 0.0) or 0.0
            ),
            "high_confidence_ratio": float(
                payload.get("high_confidence_ratio", 0.0) or 0.0
            ),
            "bot_setup_contract_id": str(setup.get("contract_id", "") or ""),
            "setup_hardening_status": str(
                setup_hardening.get("overall_status", "") or ""
            ),
            "setup_coverage_ratio": float(
                setup.get("setup_coverage_ratio", 0.0) or 0.0
            ),
            "setup_tier_counts": (
                setup.get("tier_counts")
                if isinstance(setup.get("tier_counts"), dict)
                else {}
            ),
            "setup_role_group_counts": (
                setup.get("role_group_counts")
                if isinstance(setup.get("role_group_counts"), dict)
                else {}
            ),
            "setup_lifecycle_state_counts": (
                setup.get("lifecycle_state_counts")
                if isinstance(setup.get("lifecycle_state_counts"), dict)
                else {}
            ),
            "missing_setup_metadata_count": int(
                setup.get("missing_setup_metadata_count", 0) or 0
            ),
            "tripwire_status": str(tripwire.get("overall_status", "") or ""),
            "tripwire_hardening_status": str(
                tripwire_hardening.get("overall_status", "") or ""
            ),
            "active_tripwire_count": int(tripwire.get("active_tripwire_count", 0) or 0),
            "blocking_tripwire_count": int(
                tripwire.get("blocking_tripwire_count", 0) or 0
            ),
            "tripwire_severity_counts": (
                tripwire.get("severity_counts")
                if isinstance(tripwire.get("severity_counts"), dict)
                else {}
            ),
            "tripwire_category_counts": (
                tripwire.get("category_counts")
                if isinstance(tripwire.get("category_counts"), dict)
                else {}
            ),
            "regime_quality_grade": str(payload.get("regime_quality_grade", "") or ""),
            "regime_axis_coverage_ratio": float(
                payload.get("regime_axis_coverage_ratio", 0.0) or 0.0
            ),
            "regime_axis_specificity_ratio": float(
                payload.get("regime_axis_specificity_ratio", 0.0) or 0.0
            ),
            "regime_review_count": int(payload.get("regime_review_count", 0) or 0),
            "regime_scenario_profile_count": int(
                payload.get("regime_scenario_profile_count", 0) or 0
            ),
            "regime_scenario_count": int(payload.get("regime_scenario_count", 0) or 0),
            "regime_scenario_review_count": int(
                payload.get("regime_scenario_review_count", 0) or 0
            ),
            "invalid_regime_scenario_profile_count": int(
                payload.get("invalid_regime_scenario_profile_count", 0) or 0
            ),
            "overbroad_regime_profile_count": int(
                payload.get("overbroad_regime_profile_count", 0) or 0
            ),
            "unknown_regime_profile_count": int(
                payload.get("unknown_regime_profile_count", 0) or 0
            ),
            "regime_metadata_access_grade": str(
                payload.get("regime_metadata_access_grade", "") or ""
            ),
            "regime_metadata_access_ready_count": int(
                payload.get("regime_metadata_access_ready_count", 0) or 0
            ),
            "regime_metadata_access_ratio": float(
                payload.get("regime_metadata_access_ratio", 0.0) or 0.0
            ),
            "regime_metadata_context_required_count": int(
                payload.get("regime_metadata_context_required_count", 0) or 0
            ),
            "regime_metadata_access_error_count": int(
                payload.get("regime_metadata_access_error_count", 0) or 0
            ),
            "review_queue_count": int(payload.get("review_queue_count", 0) or 0),
            "hard_limit_shadow_cell_count": len(
                payload.get("hard_limit_shadow_cells") or []
            ),
        }
    if name == "capability_materialization":
        capabilities = [
            row for row in payload.get("capabilities", []) if isinstance(row, dict)
        ]
        calendar = (
            payload.get("calendar_materialization")
            if isinstance(payload.get("calendar_materialization"), dict)
            else {}
        )
        derivatives = (
            payload.get("derivative_contract_materialization")
            if isinstance(payload.get("derivative_contract_materialization"), dict)
            else {}
        )
        stress = (
            payload.get("stress_scenario_materialization")
            if isinstance(payload.get("stress_scenario_materialization"), dict)
            else {}
        )
        return {
            "overall_status": str(payload.get("overall_status", "") or ""),
            "live_promotion_ready": bool(payload.get("live_promotion_ready", False)),
            "ready_capability_count": sum(
                1
                for row in capabilities
                if row.get("usable") is True
                and str(row.get("proof_semantics") or "") == "direct"
                and bool(str(row.get("proof_receipt_sha256") or ""))
            ),
            "capability_count": len(capabilities),
            "contract_count": int(derivatives.get("contract_count", 0) or 0),
            "stress_scenario_count": int(stress.get("scenario_count", 0) or 0),
            "exchange_calendars_version": str(
                calendar.get("library_version", "") or ""
            ),
        }
    if name == "collector_capability_control":
        summary = (
            payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        )
        return {
            "overall_status": str(payload.get("overall_status", "") or ""),
            "paper_soak_ready": bool(payload.get("paper_soak_ready", False)),
            "live_promotion_ready": bool(payload.get("live_promotion_ready", False)),
            "plane_count": int(summary.get("plane_count", 0) or 0),
            "capability_count": int(summary.get("capability_count", 0) or 0),
            "bot_binding_count": int(summary.get("bot_binding_count", 0) or 0),
            "assignment_count": int(summary.get("assignment_count", 0) or 0),
            "subscription_profile_count": int(
                summary.get("subscription_profile_count", 0) or 0
            ),
            "unmapped_current_collector_count": int(
                summary.get("unmapped_current_collector_count", 0) or 0
            ),
            "coverage_gap_count": int(
                ((payload.get("coverage_debt") or {}).get("gap_count", 0) or 0)
            ),
            "candidate_blocking_gap_count": int(
                (
                    (payload.get("coverage_debt") or {}).get(
                        "candidate_blocking_gap_count", 0
                    )
                    or 0
                )
            ),
            "optional_gap_count": int(
                ((payload.get("coverage_debt") or {}).get("optional_gap_count", 0) or 0)
            ),
            "required_capability_usable_ratio": float(
                summary.get("required_capability_usable_ratio", 0.0) or 0.0
            ),
            "required_capability_redundancy_ratio": float(
                summary.get("required_capability_redundancy_ratio", 0.0) or 0.0
            ),
            "full_catalog_coverage_ready": bool(
                summary.get("full_catalog_coverage_ready", False)
            ),
        }
    if name == "bot_profitability_scalability_control":
        diagnosis = (
            payload.get("profitability_diagnosis")
            if isinstance(payload.get("profitability_diagnosis"), dict)
            else {}
        )
        recovery = (
            diagnosis.get("paper_balance_recovery_plan")
            if isinstance(diagnosis.get("paper_balance_recovery_plan"), dict)
            else {}
        )
        recovery_attribution = (
            recovery.get("candidate_attribution")
            if isinstance(recovery.get("candidate_attribution"), dict)
            else {}
        )
        recovery_execution = (
            recovery.get("execution_evidence")
            if isinstance(recovery.get("execution_evidence"), dict)
            else {}
        )
        recovery_preflight = (
            recovery.get("preflight")
            if isinstance(recovery.get("preflight"), dict)
            else {}
        )
        return {
            "overall_status": str(payload.get("overall_status", "") or ""),
            "control_grade": str(payload.get("control_grade", "") or ""),
            "evidence_grade": str(
                payload.get("economic_and_scale_evidence_grade", "") or ""
            ),
            "implemented_control_count": int(
                payload.get("implemented_control_count", 0) or 0
            ),
            "evidence_ready_control_count": int(
                payload.get("evidence_ready_control_count", 0) or 0
            ),
            "catalog_bot_count": int(payload.get("catalog_bot_count", 0) or 0),
            "candidate_observed_bot_count": int(
                payload.get("candidate_observed_bot_count", 0) or 0
            ),
            "learned_preference_bot_count": int(
                payload.get("learned_preference_bot_count", 0) or 0
            ),
            "ranked_bot_count": int(payload.get("ranked_bot_count", 0) or 0),
            "capacity_ready_bot_count": int(
                payload.get("capacity_ready_bot_count", 0) or 0
            ),
            "planned_active_bot_count": int(
                payload.get("planned_active_bot_count", 0) or 0
            ),
            "live_allocation_ready": bool(payload.get("live_allocation_ready", False)),
            "evidence_debt_count": len(payload.get("evidence_debt") or []),
            "profitability_claim_ready": bool(
                diagnosis.get("profitability_claim_ready", False)
            ),
            "profitability_primary_reason_code": str(
                diagnosis.get("primary_reason_code") or ""
            ),
            "profitability_direct_answer": str(diagnosis.get("direct_answer") or ""),
            "paper_recovery_active": bool(recovery.get("active", False)),
            "paper_recovery_next_objective": str(recovery.get("next_objective") or ""),
            "paper_recovery_remaining_debt_amount": float(
                recovery.get("remaining_debt_amount", 0.0) or 0.0
            ),
            "paper_recovery_baseline_source": str(
                recovery.get("baseline_source") or ""
            ),
            "paper_recovery_fresh_forward_balance_active": bool(
                recovery.get("fresh_forward_balance_active", False)
            ),
            "paper_recovery_lifetime_observed_book_remaining_debt": float(
                recovery.get("lifetime_observed_book_remaining_debt", 0.0) or 0.0
            ),
            "paper_recovery_candidate_sample_count": int(
                recovery_attribution.get("sample_count", 0) or 0
            ),
            "paper_recovery_independent_fill_sample_count": int(
                recovery_execution.get("independent_fill_sample_count", 0) or 0
            ),
            "paper_recovery_preflight_status": str(
                recovery_preflight.get("status") or ""
            ),
            "paper_recovery_review_required_count": int(
                recovery_preflight.get("review_required_count", 0) or 0
            ),
            "paper_recovery_safe_auto_apply_count": int(
                recovery_preflight.get("safe_auto_apply_count", 0) or 0
            ),
            "paper_recovery_safe_maintenance_pending_binding_count": int(
                recovery_preflight.get("safe_maintenance_pending_binding_count", 0) or 0
            ),
            "paper_recovery_rejected_action_count": int(
                recovery_preflight.get("rejected_action_count", 0) or 0
            ),
        }
    if name == "sleeve_scalability_selector":
        summary = (
            payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        )
        plan = (
            payload.get("selected_advisory_plan")
            if isinstance(payload.get("selected_advisory_plan"), dict)
            else {}
        )
        selected = (
            plan.get("sleeve_ids") if isinstance(plan.get("sleeve_ids"), list) else []
        )
        growth = (
            payload.get("capital_growth_plan")
            if isinstance(payload.get("capital_growth_plan"), dict)
            else {}
        )
        growth_plan = (
            growth.get("target_sleeve_plan")
            if isinstance(growth.get("target_sleeve_plan"), dict)
            else {}
        )
        growth_selected = (
            growth_plan.get("sleeve_ids")
            if isinstance(growth_plan.get("sleeve_ids"), list)
            else []
        )
        account_scope = (
            payload.get("account_scaling_scope")
            if isinstance(payload.get("account_scaling_scope"), dict)
            else {}
        )
        portability = (
            payload.get("host_portability")
            if isinstance(payload.get("host_portability"), dict)
            else {}
        )
        return {
            "overall_status": str(payload.get("overall_status", "") or ""),
            "control_ready": bool(payload.get("control_ready", False)),
            "recommendation_ready": bool(payload.get("recommendation_ready", False)),
            "candidate_id": str(
                (payload.get("identity") or {}).get("candidate_id", "") or ""
            ),
            "active_capital_tier": str(
                (payload.get("identity") or {}).get("active_capital_tier", "") or ""
            ),
            "evaluated_sleeve_count": int(
                summary.get("evaluated_sleeve_count", 0) or 0
            ),
            "research_eligible_sleeve_count": int(
                summary.get("research_eligible_sleeve_count", 0) or 0
            ),
            "application_eligible_sleeve_count": int(
                summary.get("application_eligible_sleeve_count", 0) or 0
            ),
            "earned_scalability_goal_count": int(
                summary.get("earned_scalability_goal_count", 0) or 0
            ),
            "scalability_goal_count": int(
                summary.get("scalability_goal_count", 0) or 0
            ),
            "selected_sleeve_ids": [str(item) for item in selected if str(item)],
            "organic_capital_usd": float(growth.get("organic_capital_usd", 0.0) or 0.0),
            "next_organic_capital_target_usd": float(
                growth.get("next_target_capital_usd", 0.0) or 0.0
            ),
            "long_range_organic_target_usd": float(
                growth.get("long_range_target_capital_usd", 0.0) or 0.0
            ),
            "next_target_operating_class": str(
                growth.get("next_target_operating_class", "") or ""
            ),
            "organic_growth_progress_percent": float(
                growth.get("progress_to_next_target_percent", 0.0) or 0.0
            ),
            "organic_growth_action": str(growth.get("recommended_action", "") or ""),
            "growth_target_sleeve_ids": [
                str(item) for item in growth_selected if str(item)
            ],
            "growth_operator_review_ready": bool(
                growth.get("operator_review_ready", False)
            ),
            "classified_account_policy_count": int(
                account_scope.get("classified_account_policy_count", 0) or 0
            ),
            "account_evidence_isolated": bool(
                account_scope.get(
                    "organic_progress_isolated_by_account_policy_key", False
                )
            ),
            "host_portability_status": str(portability.get("status", "") or ""),
            "automatic_live_reactivation_on_new_host": bool(
                portability.get("automatic_live_reactivation_on_new_host", False)
            ),
            "automatic_reinvestment": bool(
                (growth.get("authority_contract") or {}).get(
                    "automatic_reinvestment", False
                )
            ),
            "evidence_debt_count": len(payload.get("evidence_debt") or []),
            "paper_execution_authority": bool(
                (payload.get("authority_contract") or {}).get(
                    "paper_execution_authority", False
                )
            ),
            "live_execution_authority": bool(
                (payload.get("authority_contract") or {}).get(
                    "live_execution_authority", False
                )
            ),
        }
    if name == "master_grandmaster_evidence_v2":
        grand = (
            payload.get("grand_master")
            if isinstance(payload.get("grand_master"), dict)
            else {}
        )
        summary = (
            payload.get("sleeve_master_summary")
            if isinstance(payload.get("sleeve_master_summary"), dict)
            else {}
        )
        return {
            "overall_status": str(payload.get("overall_status", "") or ""),
            "evidence_grade": str(payload.get("grade", "") or ""),
            "structural_grade": str(payload.get("structural_grade", "") or ""),
            "paper_coordination_ready": bool(
                payload.get("paper_coordination_ready", False)
            ),
            "human_live_review_evidence_ready": bool(
                payload.get("human_live_review_evidence_ready", False)
            ),
            "automatic_live_promotion_allowed": bool(
                grand.get("automatic_live_promotion_allowed", False)
            ),
            "sleeve_master_count": int(payload.get("sleeve_master_count", 0) or 0),
            "organized_bot_count": int(payload.get("organized_bot_count", 0) or 0),
            "master_status_counts": summary.get("status_counts") or {},
            "master_grade_counts": summary.get("grade_counts") or {},
            "operational_holds": payload.get("operational_holds") or [],
            "promotion_blockers": payload.get("promotion_blockers") or [],
            "recommended_posture": str(grand.get("recommended_posture", "") or ""),
        }
    if name == "runtime_snapshot_cache_control":
        cache = (
            payload.get("cache_health")
            if isinstance(payload.get("cache_health"), dict)
            else {}
        )
        return {
            "overall_status": str(payload.get("overall_status", "") or ""),
            "snapshot_ready": bool(cache.get("snapshot_ready", False)),
            "snapshot_age_minutes": cache.get("snapshot_age_minutes"),
        }
    if name == "remote_alert_control":
        critical = (
            payload.get("critical_backlog")
            if isinstance(payload.get("critical_backlog"), dict)
            else {}
        )
        channels = (
            payload.get("channels") if isinstance(payload.get("channels"), dict) else {}
        )
        return {
            "overall_status": str(payload.get("overall_status", "") or ""),
            "any_configured": bool(channels.get("any_configured", False)),
            "unacked_critical": int(critical.get("unacked_count", 0) or 0),
            "unsent_critical": int(critical.get("unsent_count", 0) or 0),
        }
    if name == "coordination_state_control":
        policies = (
            payload.get("policies") if isinstance(payload.get("policies"), dict) else {}
        )
        live = (
            policies.get("live_orders")
            if isinstance(policies.get("live_orders"), dict)
            else {}
        )
        paper = (
            policies.get("paper_execution")
            if isinstance(policies.get("paper_execution"), dict)
            else {}
        )
        heavy = (
            policies.get("heavy_viewer")
            if isinstance(policies.get("heavy_viewer"), dict)
            else {}
        )
        training = (
            policies.get("training_launch")
            if isinstance(policies.get("training_launch"), dict)
            else {}
        )
        terminal = (
            policies.get("terminal_restart")
            if isinstance(policies.get("terminal_restart"), dict)
            else {}
        )
        return {
            "overall_status": str(payload.get("overall_status", "") or ""),
            "coordination_mode": str(payload.get("coordination_mode", "") or ""),
            "live_orders_allowed": bool(live.get("allowed", False)),
            "paper_execution_allowed": bool(paper.get("allowed", False)),
            "heavy_viewer_allowed": bool(heavy.get("allowed", False)),
            "training_launch_allowed": bool(training.get("allowed", False)),
            "terminal_restart_safe": bool(terminal.get("safe", False)),
        }
    if name == "storage_quota_guard":
        quota = (
            payload.get("quota_summary")
            if isinstance(payload.get("quota_summary"), dict)
            else {}
        )
        return {
            "overall_status": str(payload.get("overall_status", "") or ""),
            "hard_breaches": int(quota.get("hard_breaches", 0) or 0),
            "soft_breaches": int(quota.get("soft_breaches", 0) or 0),
        }
    if name == "storage_retention_unison":
        continuous = (
            payload.get("continuous_run_contract")
            if isinstance(payload.get("continuous_run_contract"), dict)
            else {}
        )
        controls = (
            continuous.get("storage_controls")
            if isinstance(continuous.get("storage_controls"), dict)
            else {}
        )
        integration = (
            payload.get("integration_contract")
            if isinstance(payload.get("integration_contract"), dict)
            else {}
        )
        return {
            "overall_status": str(payload.get("overall_status", "") or ""),
            "continuous_run_ready": bool(continuous.get("ready", False)),
            "quota_ready": bool(controls.get("quota_ready", False)),
            "quota_status": str(controls.get("quota_status") or ""),
            "stateful_sql_compaction_only": bool(
                integration.get("stateful_sql_compaction_only", False)
            ),
        }
    if name == "release_freeze_guard":
        window = (
            payload.get("window") if isinstance(payload.get("window"), dict) else {}
        )
        return {
            "overall_status": str(payload.get("overall_status", "") or ""),
            "active": bool(window.get("active", False)),
        }
    if name == "roster_resilience_planner":
        bench = payload.get("bench") if isinstance(payload.get("bench"), dict) else {}
        return {
            "overall_status": str(payload.get("overall_status", "") or ""),
            "operational_ready": bool(
                payload.get("operational_ready", payload.get("ok", False))
            ),
            "bench_depth": int(bench.get("bench_depth", 0) or 0),
            "active_supportable_bots": int(
                bench.get("active_supportable_bots", 0) or 0
            ),
        }
    if name == "chaos_drill_coordinator":
        return {
            "overall_status": str(payload.get("overall_status", "") or ""),
            "overdue_drills": len(payload.get("overdue_drills") or []),
        }
    return {}


def _latest_shadow_loop_timestamp(project_root: Path) -> datetime | None:
    latest: datetime | None = None
    for path in (project_root / "governance" / "health").glob("shadow_loop_*.json"):
        payload = _load_json(path)
        ts = _payload_timestamp(payload, path)
        if ts is None:
            continue
        if latest is None or ts > latest:
            latest = ts
    return latest


def _lock_owner_pid(lock_path_text: str) -> int | None:
    lock_path = Path(str(lock_path_text or "").strip()).expanduser()
    if not lock_path.exists():
        return None
    try:
        raw = lock_path.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    for token in raw.split():
        if not token.startswith("pid="):
            continue
        try:
            return int(token.split("=", 1)[1])
        except Exception:
            return None
    return None


def _live_sql_writer_pid(artifact: Dict[str, Any]) -> int | None:
    if not bool(artifact.get("exists")):
        return None
    summary = (
        artifact.get("summary") if isinstance(artifact.get("summary"), dict) else {}
    )
    if not bool(summary.get("running")):
        return None
    age_minutes = artifact.get("age_minutes")
    max_age_minutes = float(artifact.get("max_age_minutes", 30.0) or 30.0)
    if age_minutes is not None and float(age_minutes) > max_age_minutes * 8.0:
        return None
    lock_path_text = str(artifact.get("path") or "")
    payload_path = Path(lock_path_text) if lock_path_text else Path("")
    payload = _load_json(payload_path) if payload_path.exists() else {}
    pid = _lock_owner_pid(str(payload.get("lock_path") or ""))
    if pid is None:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        return pid
    return pid


def _artifact_contract(
    artifacts: Dict[str, Dict[str, Any]], name: str
) -> Dict[str, Any]:
    artifact = artifacts.get(name, {})
    exists = bool(artifact.get("exists"))
    stale = bool(artifact.get("stale"))
    summary = (
        artifact.get("summary") if isinstance(artifact.get("summary"), dict) else {}
    )
    status = str(
        summary.get("overall_status")
        or summary.get("status")
        or artifact.get("status")
        or "unknown"
    )
    artifact_status = "fresh"
    artifact_reason = "ok"
    if not exists:
        artifact_status = "missing"
        artifact_reason = "artifact_missing"
        status = "unknown"
    elif stale:
        artifact_status = "stale"
        artifact_reason = "artifact_stale"
        if status in {"", "ok"}:
            status = "unknown"
    elif not summary:
        artifact_status = "summary_empty"
        artifact_reason = "artifact_summary_empty"
        status = "unknown"
    return {
        "artifact_status": artifact_status,
        "artifact_reason": artifact_reason,
        "source_path": str(artifact.get("path") or ""),
        "required": bool(artifact.get("required", False)),
        "status": status or "unknown",
    }


def _stale_artifact_bot_contention_nonblocking(
    artifact: Dict[str, Any], *, action_key: str
) -> bool:
    if not artifact or artifact.get("ok") is not False or artifact.get("stale"):
        return False
    summary = (
        artifact.get("summary") if isinstance(artifact.get("summary"), dict) else {}
    )
    reason = str(summary.get("reason", "") or "").strip().lower()
    if reason not in {"already_running", "lock_busy"}:
        return False
    if not bool(summary.get("busy", False)):
        return False
    if int(summary.get("delete_errors", 0) or 0) > 0:
        return False
    if int(summary.get("candidate_files", 0) or 0) > 0:
        return False
    if int(summary.get(action_key, 0) or 0) > 0:
        return False
    summary["maintenance_contention_nonblocking"] = True
    return True


def _external_backlog_tiny_drain_tail_nonblocking(
    storage_summary: Dict[str, Any], drain_summary: Dict[str, Any]
) -> bool:
    storage_ready = bool(
        str(storage_summary.get("overall_status", "") or "") == "ready"
        and str(storage_summary.get("severity", "") or "") in {"stable", ""}
        and _safe_float(storage_summary.get("pressure_index"), 1.0) <= 0.25
    )
    tiny_drain_tail = bool(
        (
            _safe_float(storage_summary.get("estimated_total_drain_minutes"), 999.0)
            <= 1.0
            or (
                _safe_int(storage_summary.get("core_pending_lines"), 999_999) <= 5_000
                and _safe_int(storage_summary.get("total_pending_lines"), 999_999)
                <= 5_000
            )
        )
        and int(drain_summary.get("aged_candidate_files", 0) or 0) <= 0
    )
    handoff_in_progress = str(drain_summary.get("follow_through_status", "") or "") in {
        "handoff_requested",
        "completed",
        "",
    }
    if not (storage_ready and tiny_drain_tail and handoff_in_progress):
        return False
    drain_summary["tiny_drain_tail_nonblocking"] = True
    return True


_EVIDENCE_DEBT_NON_DEGRADED_ARTIFACTS = {
    "bot_profitability_scalability_control",
    "sleeve_scalability_selector",
    "master_grandmaster_evidence_v2",
}
_EVIDENCE_DEBT_NON_DEGRADED_STATUSES = {
    "ready_with_evidence_debt",
    "ready_with_review_debt",
}


def _artifact_waiting_on_evidence_not_degraded(
    name: str, status: str, summary: Dict[str, Any]
) -> bool:
    if name not in _EVIDENCE_DEBT_NON_DEGRADED_ARTIFACTS:
        return False
    if str(status or "").strip().lower() not in _EVIDENCE_DEBT_NON_DEGRADED_STATUSES:
        return False
    if name == "bot_profitability_scalability_control":
        return bool(
            str(summary.get("control_grade", "") or "").upper() in {"A", "A+"}
            and not bool(summary.get("profitability_claim_ready", False))
            and not bool(summary.get("live_allocation_ready", False))
        )
    if name == "sleeve_scalability_selector":
        return bool(
            summary.get("control_ready") is True
            and not bool(summary.get("recommendation_ready", False))
            and not bool(summary.get("paper_execution_authority", False))
            and not bool(summary.get("live_execution_authority", False))
        )
    if name == "master_grandmaster_evidence_v2":
        return bool(
            str(summary.get("structural_grade", "") or "").upper() in {"A", "A+"}
            and summary.get("paper_coordination_ready") is True
            and not bool(summary.get("automatic_live_promotion_allowed", False))
        )
    return False


_CONTAINED_ATTENTION = {
    "paper_execution_safety_guard_active",
    "external_backlog_drain_recommended",
    "external_backlog_drain_writer_busy",
    "external_backlog_retry_bot_followups",
    "retrain_artifact_freshness_not_ok",
    "promotion_not_ready",
    "training_quality_control_blocked",
    "bot_quality_autopilot_evidence_pending",
    "infrastructure_autofix_bot_needs_work",
    "roster_resilience_planner_needs_work",
    "source_verification_context_debt",
    "coordination_state_control_needs_work",
}

_ADVISORY_ATTENTION = {
    "memory_efficiency_control_needs_work",
    "paper_execution_safety_guard_active",
    "external_backlog_drain_recommended",
    "external_backlog_drain_writer_busy",
    "external_backlog_retry_bot_followups",
    "retrain_artifact_freshness_not_ok",
    "infrastructure_autofix_bot_blocked",
    "live_runtime_separation_control_needs_work",
    "rolling_restart_controller_blocked",
    "blackstart_recovery_needs_work",
    "coordination_state_control_blocked",
    "coordination_state_control_needs_work",
    "promotion_not_ready",
    "training_quality_control_blocked",
    "platform_control_plane_upgrade_required",
    "daily_verify_auto_remediation_pending",
    "bot_quality_autopilot_evidence_pending",
    "infrastructure_autofix_bot_needs_work",
    "artifact_freshness_slo_blocked",
    "runtime_snapshot_cache_control_needs_work",
    "chaos_drill_coordinator_needs_work",
    "roster_resilience_planner_needs_work",
    "source_verification_context_debt",
}

_CRITICAL_ATTENTION = {
    "health_gates_hard_gate_triggered",
    "global_killswitch_not_ok",
}

_DEGRADED_ATTENTION = {
    "daily_auto_verify_not_ok",
    "session_ready_not_ok",
    "health_gates_not_ok",
    "sql_link_service_not_ok",
    "ingestion_storage_control_blocked",
    "ingestion_storage_governor_critical",
    "sql_primary_route_drift",
    "storage_split_brain_needs_review",
    "master_grandmaster_evidence_v2_not_ok",
    "bot_profitability_scalability_control_not_ok",
    "sleeve_scalability_selector_not_ok",
    "source_verification_decision_critical_blocked",
    "safety_evidence_snapshot_skew",
}

_ATTENTION_OWNER_ACTIONS: dict[str, dict[str, Any]] = {
    "master_grandmaster_evidence_v2_not_ok": {
        "owner": "master_grandmaster_evidence_v2",
        "command": [
            "./scripts/ops/opsctl.sh",
            "master-grandmaster-evidence",
            "--json",
        ],
        "timeout_seconds": 120,
        "success_condition": "master_grandmaster_evidence_v2_latest.ok is true and integrity_blockers is empty",
    },
    "bot_profitability_scalability_control_not_ok": {
        "owner": "bot_profitability_scalability_control",
        "command": [
            "./scripts/ops/opsctl.sh",
            "bot-profitability-scalability",
            "--json",
        ],
        "timeout_seconds": 180,
        "success_condition": "all 16 controls are implemented and structural blockers are empty; economic evidence debt remains separately visible",
    },
    "sleeve_scalability_selector_not_ok": {
        "owner": "sleeve_scalability_selector",
        "command": [
            "./scripts/ops/opsctl.sh",
            "sleeve-scalability-selector",
            "--json",
        ],
        "timeout_seconds": 120,
        "success_condition": "selector control is structurally ready; earned sleeve and scalability evidence remains separately visible",
    },
    "sql_link_service_not_ok": {
        "owner": "sql_link_service",
        "command": [
            "./scripts/ops/opsctl.sh",
            "sql-link-service",
            "--once",
            "--json",
        ],
        "timeout_seconds": 240,
        "success_condition": "sql_link_service_latest is ready/running and writer progress is current without duplicate writer processes",
    },
    "paper_execution_safety_guard_active": {
        "owner": "paper_execution_safety_guard",
        "command": [
            "./scripts/ops/opsctl.sh",
            "paper-profitability-control",
            "--json",
        ],
        "timeout_seconds": 120,
        "success_condition": "paper execution stays guarded until post-cost fills, persistence, and execution evidence are gradeable",
    },
    "source_verification_context_debt": {
        "owner": "source_verification_report",
        "command": [
            "./scripts/ops/opsctl.sh",
            "source-verification-refresh",
            "--apply",
            "--json",
        ],
        "timeout_seconds": 240,
        "success_condition": "decision-critical sources remain ready and context debt clears after dependency-ordered refresh",
    },
    "source_verification_decision_critical_blocked": {
        "owner": "source_verification_report",
        "command": [
            "./scripts/ops/opsctl.sh",
            "source-verification-refresh",
            "--apply",
            "--json",
        ],
        "timeout_seconds": 240,
        "success_condition": "source_runtime_contract.decision_critical_sources_ready is true before paper/live decisions rely on market context",
    },
    "external_backlog_drain_recommended": {
        "owner": "external_backlog_drain",
        "command": [
            "./scripts/ops/opsctl.sh",
            "external-backlog-drain",
            "--apply",
            "--follow-through",
            "--json",
        ],
        "timeout_seconds": 240,
        "success_condition": "external backlog drain has no aged candidates and storage pressure remains stable",
    },
    "external_backlog_drain_writer_busy": {
        "owner": "writer_cycle_coordinator",
        "command": [
            "./scripts/ops/opsctl.sh",
            "writer-cycle-coordinator",
            "--apply",
            "--fast-handoff",
            "--json",
        ],
        "timeout_seconds": 240,
        "success_condition": "writer handoff progresses or storage backlog remains inside stable queue watermarks",
    },
    "external_backlog_retry_bot_followups": {
        "owner": "external_backlog_retry_bot",
        "command": [
            "./scripts/ops/opsctl.sh",
            "external-backlog-retry-bot",
            "--apply",
            "--json",
        ],
        "timeout_seconds": 240,
        "success_condition": "retry bot either clears followups or records active progress while storage remains stable",
    },
    "daily_auto_verify_not_ok": {
        "owner": "daily_verify_auto_remediation_bot",
        "command": [
            "./scripts/ops/opsctl.sh",
            "daily-verify-remediation",
            "--apply",
            "--json",
        ],
        "timeout_seconds": 180,
        "success_condition": "daily_auto_verify_latest.ok is true or failed checks resolve through mapped owner artifacts",
    },
    "retrain_artifact_freshness_not_ok": {
        "owner": "retrain_artifact_freshness_guard",
        "command": ["python", "scripts/retrain_artifact_freshness_guard.py", "--json"],
        "timeout_seconds": 180,
        "success_condition": "retrain_artifact_freshness_latest.ok is true or sample sufficiency is advisory-only while daily verify is clean",
    },
    "promotion_not_ready": {
        "owner": "promotion_quality_gate",
        "command": [
            "./scripts/ops/opsctl.sh",
            "promotion-quality-gate",
            "--json",
        ],
        "timeout_seconds": 120,
        "success_condition": "promotion_quality_gate_latest.overall_status is ready or failed_checks explicitly name contained evidence debt",
    },
    "training_quality_control_blocked": {
        "owner": "training_quality_control",
        "command": [
            "./scripts/ops/opsctl.sh",
            "training-quality",
            "--json",
        ],
        "timeout_seconds": 120,
        "success_condition": "training_quality_control_latest.overall_status is ready or operating_contract lists named evidence-only blockers",
    },
    "bot_quality_autopilot_evidence_pending": {
        "owner": "bot_quality_autopilot",
        "command": [
            "./scripts/ops/opsctl.sh",
            "bot-quality-autopilot",
            "--json",
        ],
        "timeout_seconds": 240,
        "success_condition": "bot_quality_autopilot_latest either clears or records evidence-only queue debt while training budget remains closed",
    },
    "memory_efficiency_control_needs_work": {
        "owner": "memory_efficiency_control",
        "command": ["./scripts/ops/opsctl.sh", "memory-efficiency", "apply", "--json"],
        "timeout_seconds": 90,
        "success_condition": "memory pressure remains green; compressed-store-only findings stay advisory",
    },
    "ingestion_storage_control_blocked": {
        "owner": "storage_backpressure_autopilot",
        "command": [
            "./scripts/ops/opsctl.sh",
            "storage-backpressure-autopilot",
            "--apply",
            "--quick-bounded",
            "--json",
        ],
        "timeout_seconds": 240,
        "success_condition": "ingestion_storage_control_latest.overall_status is ready and pressure_index is below critical",
    },
    "infrastructure_autofix_bot_blocked": {
        "owner": "infrastructure_autofix_bot",
        "command": [
            "./scripts/ops/opsctl.sh",
            "infrastructure-autofix",
            "--apply",
            "--json",
        ],
        "timeout_seconds": 180,
        "success_condition": "infrastructure_autofix_bot_latest.overall_status is ready or no critical runtime gate depends on it",
    },
    "infrastructure_autofix_bot_needs_work": {
        "owner": "infrastructure_autofix_bot",
        "command": [
            "./scripts/ops/opsctl.sh",
            "infrastructure-autofix",
            "--apply",
            "--json",
        ],
        "timeout_seconds": 180,
        "success_condition": "infrastructure_autofix_bot findings remain advisory while hot-path readiness is green",
    },
    "live_runtime_separation_control_needs_work": {
        "owner": "live_runtime_separation_control",
        "command": ["./scripts/ops/opsctl.sh", "live-runtime-separation", "--json"],
        "timeout_seconds": 90,
        "success_condition": "live runtime separation is ready or live execution is locked/read-only",
    },
    "rolling_restart_controller_blocked": {
        "owner": "rolling_restart_controller",
        "command": ["./scripts/ops/opsctl.sh", "rolling-restart", "--json"],
        "timeout_seconds": 90,
        "success_condition": "restart is not due, or due restart is advisory while critical lanes are healthy",
    },
    "blackstart_recovery_needs_work": {
        "owner": "blackstart_recovery",
        "command": ["./scripts/ops/opsctl.sh", "blackstart-recovery", "--json"],
        "timeout_seconds": 90,
        "success_condition": "blackstart recovery is ready or degraded only by non-critical drill freshness",
    },
    "coordination_state_control_blocked": {
        "owner": "coordination_state_control",
        "command": ["./scripts/ops/opsctl.sh", "coordination-status", "--json"],
        "timeout_seconds": 90,
        "success_condition": "coordination_state_latest is ready or blocks are advisory while global/live execution gates are clear",
    },
    "coordination_state_control_needs_work": {
        "owner": "coordination_state_control",
        "command": ["./scripts/ops/opsctl.sh", "coordination-status", "--json"],
        "timeout_seconds": 90,
        "success_condition": "coordination_state_latest is ready or blocks are advisory while global/live execution gates are clear",
    },
    "artifact_freshness_slo_blocked": {
        "owner": "artifact_freshness_slo",
        "command": ["./scripts/ops/opsctl.sh", "artifact-freshness-slo", "--json"],
        "timeout_seconds": 90,
        "success_condition": "required stale artifacts are not hot-path blockers while health-fast, watchdog, and storage remain green",
    },
    "runtime_snapshot_cache_control_needs_work": {
        "owner": "runtime_snapshot_cache_control",
        "command": ["./scripts/ops/opsctl.sh", "runtime-snapshot-cache", "--json"],
        "timeout_seconds": 90,
        "success_condition": "runtime snapshot cache is ready or remains advisory while collection/paper runtime is healthy",
    },
    "remote_alert_control_needs_work": {
        "owner": "remote_alert_control",
        "command": ["./scripts/ops/opsctl.sh", "remote-alert-control", "--json"],
        "timeout_seconds": 90,
        "success_condition": "remote_alert_control_latest has no critical unsent/unacked backlog, or remote paging remains attended/local-only by policy",
    },
    "chaos_drill_coordinator_needs_work": {
        "owner": "chaos_drill_coordinator",
        "command": ["./scripts/ops/opsctl.sh", "chaos-drills", "--json"],
        "timeout_seconds": 90,
        "success_condition": "overdue drills stay advisory and do not block the current guarded paper/data path",
    },
    "roster_resilience_planner_needs_work": {
        "owner": "roster_resilience_planner",
        "command": ["./scripts/ops/opsctl.sh", "roster-resilience", "--json"],
        "timeout_seconds": 90,
        "success_condition": "roster_resilience_planner_latest.operational_ready is true and any advisory bench or teacher debt remains non-runtime-blocking",
    },
}


_GREEN_SOAK_MANAGED_ATTENTION_REASONS = {
    "daily_auto_verify_not_ok": "daily_verify_training_promotion_checks_deferred_while_paper_soak_is_green",
    "promotion_not_ready": "promotion_deferred_while_paper_soak_is_green",
    "training_quality_control_blocked": "training_quality_recovery_deferred_while_paper_execution_is_clean",
    "teacher_quality_guard_blocked": "teacher_qualification_deferred_while_unqualified_teachers_are_fail_closed",
    "bot_quality_autopilot_blocked": "bot_quality_retrain_queue_deferred_while_training_budget_is_closed",
    "external_backlog_drain_recommended": "external_backlog_handoff_managed_while_ingestion_soak_is_green",
    "external_backlog_drain_writer_busy": "external_backlog_writer_busy_managed_while_ingestion_soak_is_green",
    "external_backlog_retry_bot_followups": "external_backlog_retry_followup_deferred_while_ingestion_soak_is_green",
    "auth_lease_manager_needs_work": "schwab_auth_warning_managed_while_token_above_paper_readiness_floor",
    "ingestion_storage_governor_critical": "deferred_backlog_governor_profile_managed_by_storage_soak_contract",
    "infrastructure_autofix_bot_blocked": "safe_infrastructure_repair_timer_gap_deferred_while_hot_path_is_green",
    "infrastructure_autofix_bot_needs_work": "safe_infrastructure_repair_timer_gap_deferred_while_hot_path_is_green",
    "live_runtime_separation_control_needs_work": "live_money_cold_lane_clearance_deferred_while_paper_soak_is_green",
    "runtime_snapshot_cache_control_needs_work": "snapshot_cache_upstream_training_freshness_deferred_while_snapshot_is_ready",
    "coordination_state_control_blocked": "protective_live_order_lock_allows_paper_collection_soak",
    "coordination_state_control_needs_work": "protective_live_order_lock_allows_paper_collection_soak",
    "storage_quota_guard_needs_work": "soft_storage_quota_pressure_managed_by_ingestion_soak_contract",
    "roster_resilience_planner_needs_work": "roster_coverage_topoff_deferred_while_paper_soak_is_green",
    "chaos_drill_coordinator_blocked": "disruptive_recovery_drills_deferred_while_paper_soak_is_green",
    "chaos_drill_coordinator_needs_work": "disruptive_recovery_drills_deferred_while_paper_soak_is_green",
}

_PAPER_SOAK_ALLOWED_BLOCKED_TEACHER_STATUSES = {
    "registry_only_guarded",
    "insufficient_evidence",
    "high_accuracy_guarded",
}

_GREEN_SOAK_MANAGED_DAILY_VERIFY_CHECKS = {
    "snapshot_coverage_sentinel",
    "feature_store_manifest",
    "retrain_schema_compatibility_guard",
    "promotion_packet_builder",
    "promotion_quality_gate",
}


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _dashboard_soak_context(project_root: Path) -> dict[str, Any]:
    health = project_root / "governance" / "health"
    soak = _load_json(health / "unattended_soak_readiness_latest.json")
    paper_guard = _load_json(health / "runtime_paper_regression_guard_latest.json")
    health_fast = _load_json(health / "health_fast_latest.json")
    soak_status = (
        str(soak.get("overall_status") or soak.get("status") or "").strip().lower()
    )
    soak_grade = (
        str(soak.get("overall_grade") or soak.get("grade") or "").strip().upper()
    )
    soak_ready = bool(soak.get("safe_to_leave_unattended", False)) and soak_status in {
        "ready",
        "ok",
        "healthy",
    }
    if soak_grade and soak_grade not in {"A", "A+"}:
        soak_ready = False
    paper_status = (
        str(paper_guard.get("overall_status") or paper_guard.get("status") or "")
        .strip()
        .lower()
    )
    paper_clean = (
        bool(paper_guard.get("ok", False))
        and paper_status in {"ready", "ok", "healthy"}
        and _safe_int(paper_guard.get("failed_guard_count"), 0) <= 0
        and not (
            paper_guard.get("failed_guards")
            if isinstance(paper_guard.get("failed_guards"), list)
            else []
        )
        and bool(paper_guard.get("paper_armed", False))
        and not bool(paper_guard.get("paper_blocked", False))
    )
    health_status = (
        str(health_fast.get("overall_status") or health_fast.get("status") or "")
        .strip()
        .lower()
    )
    operational = (
        health_fast.get("operational_readiness")
        if isinstance(health_fast.get("operational_readiness"), dict)
        else {}
    )
    guarded_paper = (
        operational.get("guarded_paper")
        if isinstance(operational.get("guarded_paper"), dict)
        else {}
    )
    guarded_health_ready = bool(
        health_status in {"ready", "ok", "healthy", "guarded_ready"}
        and guarded_paper.get("ok", False)
        and str(guarded_paper.get("status") or "").strip().lower() == "ready"
    )
    return {
        "enabled": bool(
            soak_ready
            and paper_clean
            and (health_status in {"ready", "ok", "healthy"} or guarded_health_ready)
        ),
        "soak_ready": bool(soak_ready),
        "soak_status": soak_status,
        "soak_grade": soak_grade,
        "paper_guard_clean": bool(paper_clean),
        "paper_guard_status": paper_status,
        "paper_stage": str(paper_guard.get("paper_stage") or ""),
        "paper_armed": bool(paper_guard.get("paper_armed", False)),
        "paper_blocked": bool(paper_guard.get("paper_blocked", False)),
        "health_fast_status": health_status,
        "guarded_health_ready": guarded_health_ready,
    }


def _snapshot_cache_ready_for_soak(artifacts: Dict[str, Dict[str, Any]]) -> bool:
    summary = artifacts.get("runtime_snapshot_cache_control", {}).get("summary", {})
    cache = (
        summary.get("cache_health")
        if isinstance(summary.get("cache_health"), dict)
        else {}
    )
    if bool(summary.get("snapshot_ready", False)):
        return True
    return bool(cache.get("snapshot_ready", False)) and bool(
        cache.get("snapshot_exists", True)
    )


def _roster_resilience_ready_for_soak(artifacts: Dict[str, Dict[str, Any]]) -> bool:
    summary = artifacts.get("roster_resilience_planner", {}).get("summary", {})
    contract = (
        summary.get("a_plus_contract")
        if isinstance(summary.get("a_plus_contract"), dict)
        else {}
    )
    bench = summary.get("bench") if isinstance(summary.get("bench"), dict) else {}
    active_supportable = _safe_int(
        contract.get("active_supportable_bots"),
        _safe_int(bench.get("active_supportable_bots"), 0),
    )
    active_target = _safe_int(contract.get("active_supportable_target"), 0)
    bench_depth = _safe_int(
        contract.get("bench_depth"), _safe_int(bench.get("bench_depth"), 0)
    )
    bench_target = _safe_int(contract.get("bench_depth_target"), 0)
    return bool(active_supportable >= active_target and bench_depth >= bench_target)


def _ingestion_soak_ready_for_dashboard(artifacts: Dict[str, Dict[str, Any]]) -> bool:
    artifact = artifacts.get("ingestion_storage_control", {})
    if bool(artifact.get("stale", False)):
        return False
    summary = artifact.get("summary", {})
    if str(summary.get("overall_status", "") or "") not in {"ready", "ok"}:
        return False
    severity = str(summary.get("severity", "") or "")
    if severity not in {"stable", "low", "normal", "elevated", ""}:
        return False
    pressure_index = float(summary.get("pressure_index", 0.0) or 0.0)
    if pressure_index >= BOUNDED_TRANSIENT_STORAGE_PRESSURE_MAX:
        return False
    payload = _load_json(
        Path(str(artifacts.get("ingestion_storage_control", {}).get("path", "") or ""))
    )
    contract = (
        payload.get("continuous_run_soak_contract")
        if isinstance(payload.get("continuous_run_soak_contract"), dict)
        else {}
    )
    if severity != "elevated" and bool(
        contract.get("ready", False) or contract.get("soak_ready", False)
    ):
        return True
    bounded = (
        payload.get("bounded_recovery_contract")
        if isinstance(payload.get("bounded_recovery_contract"), dict)
        else {}
    )
    integrity = (
        payload.get("data_integrity")
        if isinstance(payload.get("data_integrity"), dict)
        else {}
    )
    writer = (
        payload.get("writer_shedding")
        if isinstance(payload.get("writer_shedding"), dict)
        else {}
    )
    blockers = {
        str(item or "").strip()
        for item in (
            contract.get("blockers")
            if isinstance(contract.get("blockers"), list)
            else []
        )
        if str(item or "").strip()
    }
    return bool(
        pressure_index < BOUNDED_TRANSIENT_STORAGE_PRESSURE_MAX
        and blockers.issubset({"steady_state_targets_not_clear"})
        and bool(bounded.get("route_verified", False))
        and bool(
            bounded.get("active_drain_progress", False)
            or bounded.get("drain_delta_signal_observed", False)
        )
        and not bool(bounded.get("hard_gate_active", False))
        and not bool(bounded.get("effective_hard_gate_active", False))
        and not writer.get("hard_breaches")
        and not writer.get("elevated_breaches")
        and all(
            _safe_int(integrity.get(key), 0) == 0
            for key in (
                "sql_invalid_lines",
                "sql_overlay_invalid_lines",
                "sql_overlay_oversize_payloads",
                "sql_overlay_ops_write_failures",
            )
        )
        and _raw_live_backlog_clear_for_storage_soak(payload)
    )


def _overlay_raw_live_candidate(
    backpressure: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    effective = (
        backpressure.get("effective_raw_live")
        if isinstance(backpressure.get("effective_raw_live"), dict)
        else {}
    )
    effective_source = str(
        backpressure.get("effective_raw_live_source") or effective.get("source") or ""
    )
    estimate = (
        effective.get("raw_live_estimate")
        if isinstance(effective.get("raw_live_estimate"), dict)
        else {}
    )
    if backpressure.get("effective_pressure_clear", False) and effective:
        return effective, effective_source or "effective_pressure_contract"
    if estimate and effective_source == "sql_ingestion_overlay_pressure":
        return estimate, "effective_raw_live.raw_live_estimate"
    raw_live = (
        backpressure.get("raw_live")
        if isinstance(backpressure.get("raw_live"), dict)
        else {}
    )
    if raw_live:
        return raw_live, "raw_live"
    return backpressure, "backpressure"


def _raw_live_backlog_clear_for_storage_soak(ingestion_storage: dict[str, Any]) -> bool:
    backpressure = (
        ingestion_storage.get("backpressure")
        if isinstance(ingestion_storage.get("backpressure"), dict)
        else {}
    )
    raw_live, _source = _overlay_raw_live_candidate(backpressure)
    raw_core = _safe_int(raw_live.get("core_pending_lines"), 0)
    raw_total = _safe_int(raw_live.get("total_pending_lines"), raw_core)
    raw_oldest = _safe_float(raw_live.get("oldest_pending_age_seconds"), 0.0)
    return bool(
        raw_core <= OVERLAY_RAW_LIVE_MAX_CORE_LINES
        and raw_total <= OVERLAY_RAW_LIVE_MAX_TOTAL_LINES
        and raw_oldest <= OVERLAY_RAW_LIVE_MAX_AGE_SECONDS
    )


def _stateful_sql_soft_quota_deferred_for_paper_soak(
    artifacts: Dict[str, Dict[str, Any]],
) -> bool:
    quota_artifact = artifacts.get("storage_quota_guard", {})
    quota_payload = _load_json(Path(str(quota_artifact.get("path") or "")))
    if not quota_payload:
        return False
    quota_summary = (
        quota_payload.get("quota_summary")
        if isinstance(quota_payload.get("quota_summary"), dict)
        else {}
    )
    lanes = (
        quota_payload.get("lanes")
        if isinstance(quota_payload.get("lanes"), list)
        else []
    )
    hard_breaches = _safe_int(quota_summary.get("hard_breaches"), 0)
    soft_breaches = _safe_int(quota_summary.get("soft_breaches"), 0)
    if hard_breaches > 0 or soft_breaches > 1:
        return False
    blocked_families = {
        str(item or "").strip()
        for item in (
            quota_summary.get("blocked_families")
            if isinstance(quota_summary.get("blocked_families"), list)
            else []
        )
        if str(item or "").strip()
    }
    if blocked_families:
        return False
    degraded_families = {
        str(item or "").strip()
        for item in (
            quota_summary.get("degraded_families")
            if isinstance(quota_summary.get("degraded_families"), list)
            else []
        )
        if str(item or "").strip()
    }
    sql_lane: dict[str, Any] = {}
    for row in lanes:
        if isinstance(row, dict) and str(row.get("family") or "") == "sql_link_shards":
            sql_lane = row
            break
    if not degraded_families:
        degraded_families = {
            str(row.get("family") or "")
            for row in lanes
            if isinstance(row, dict)
            and str(row.get("status") or "") in {"degraded", "blocked"}
        }
        degraded_families.discard("")
    if not sql_lane or not degraded_families.issubset({"sql_link_shards"}):
        return False
    if _safe_float(sql_lane.get("over_hard_gb"), 0.0) > 0.0:
        return False
    if (
        _safe_float(sql_lane.get("hard_ratio"), 0.0)
        > STATEFUL_SQL_SOFT_QUOTA_MAX_HARD_RATIO
    ):
        return False
    ingestion_payload = _load_json(
        Path(str(artifacts.get("ingestion_storage_control", {}).get("path") or ""))
    )
    severity = str(ingestion_payload.get("severity") or "").strip().lower()
    storage_status = str(ingestion_payload.get("overall_status") or "").strip().lower()
    if severity not in {
        "",
        "ready",
        "stable",
        "low",
        "normal",
    } or storage_status not in {"", "ready", "ok", "advisory"}:
        return False
    if not _raw_live_backlog_clear_for_storage_soak(ingestion_payload):
        return False
    unison_payload = _load_json(
        Path(str(artifacts.get("storage_retention_unison", {}).get("path") or ""))
    )
    continuous = (
        unison_payload.get("continuous_run_contract")
        if isinstance(unison_payload.get("continuous_run_contract"), dict)
        else {}
    )
    controls = (
        continuous.get("storage_controls")
        if isinstance(continuous.get("storage_controls"), dict)
        else {}
    )
    forecast = (
        unison_payload.get("storage_growth_forecast")
        if isinstance(unison_payload.get("storage_growth_forecast"), dict)
        else {}
    )
    forecast_status = str(forecast.get("status") or "").strip()
    days_until_pressure = forecast.get("days_until_pressure_free")
    forecast_ready = bool(
        forecast_status in {"stable_or_improving", "forecast_ready", "ready"}
        and (
            days_until_pressure is None or _safe_float(days_until_pressure, 0.0) >= 30.0
        )
    )
    continuous_ready = bool(
        continuous.get("ready", False)
        or continuous.get("status") == "ready"
        or forecast_ready
    )
    quota_ready = bool(controls.get("quota_ready", False)) or (
        hard_breaches == 0
        and not bool(quota_summary.get("external_free_below_target", False))
    )
    integration = (
        unison_payload.get("integration_contract")
        if isinstance(unison_payload.get("integration_contract"), dict)
        else {}
    )
    tier_payload = _load_json(
        Path(str(artifacts.get("storage_tier_policy", {}).get("path") or ""))
    )
    manifest_contract = (
        tier_payload.get("manifest_backed_offload_contract")
        if isinstance(tier_payload.get("manifest_backed_offload_contract"), dict)
        else {}
    )
    stateful_policy = str(manifest_contract.get("stateful_sql_policy") or "").lower()
    stateful_sql_compaction_only = bool(
        integration.get("stateful_sql_compaction_only", False)
    ) or ("never source-delete" in stateful_policy and "checkpoint" in stateful_policy)
    return bool(continuous_ready and quota_ready and stateful_sql_compaction_only)


def _daily_verify_deferred_for_paper_soak(artifacts: Dict[str, Dict[str, Any]]) -> bool:
    daily_summary = artifacts.get("daily_auto_verify", {}).get("summary", {})
    failed = daily_summary.get("effective_failed_checks")
    if not isinstance(failed, list):
        failed = (
            daily_summary.get("failed_checks")
            if isinstance(daily_summary.get("failed_checks"), list)
            else []
        )
    failed_set = {str(item or "").strip() for item in failed if str(item or "").strip()}
    if not failed_set or not failed_set.issubset(
        _GREEN_SOAK_MANAGED_DAILY_VERIFY_CHECKS
    ):
        return False
    completed = _safe_int(daily_summary.get("completed_checks"), 0)
    if completed <= 0:
        return False
    health_summary = artifacts.get("health_gates", {}).get("summary", {})
    if bool(health_summary.get("hard_gate_triggered", False)):
        return False
    return bool(
        _snapshot_cache_ready_for_soak(artifacts)
        or _ingestion_soak_ready_for_dashboard(artifacts)
    )


def _training_quality_deferred_for_paper_soak(
    artifacts: Dict[str, Dict[str, Any]],
) -> bool:
    summary = artifacts.get("training_quality_control", {}).get("summary", {})
    if str(summary.get("overall_status", "") or "") != "blocked":
        return False
    score = float(summary.get("training_quality_score", 0.0) or 0.0)
    if score < 70.0 and not _ingestion_soak_ready_for_dashboard(artifacts):
        return False
    priorities = {
        str(item or "").strip()
        for item in (
            summary.get("top_priorities")
            if isinstance(summary.get("top_priorities"), list)
            else []
        )
        if str(item or "").strip()
    }
    cold_lane_priorities = {
        "active_probation_isolation",
        "experiment_replayability",
        "feature_store_lineage",
        "multiple_testing_control",
        "promotion_coverage",
    }
    return bool(not priorities or priorities.issubset(cold_lane_priorities))


def _teacher_quality_deferred_for_paper_soak(
    artifacts: Dict[str, Dict[str, Any]],
) -> bool:
    artifact = artifacts.get("teacher_quality_guard", {})
    if artifact.get("stale") is True:
        return False
    payload = _load_json(Path(str(artifact.get("path") or "")))
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    awareness = (
        payload.get("overfitting_awareness")
        if isinstance(payload.get("overfitting_awareness"), dict)
        else {}
    )
    status_counts = (
        awareness.get("blocked_status_counts")
        if isinstance(awareness.get("blocked_status_counts"), dict)
        else {}
    )
    blocked_statuses = {
        str(key or "").strip() for key in status_counts if str(key or "").strip()
    }
    blocked_count = _safe_int(awareness.get("blocked_teacher_count"), 0)
    counted_blocked = sum(
        max(_safe_int(value, 0), 0) for value in status_counts.values()
    )
    return bool(
        str(payload.get("overall_status") or "").strip().lower() == "blocked"
        and "teaching_enabled" in summary
        and not bool(summary.get("teaching_enabled"))
        and _safe_int(summary.get("qualified_teacher_count"), 0) == 0
        and _safe_int(summary.get("elite_teacher_count"), 0) == 0
        and str(awareness.get("overall_status") or "").strip().lower() == "guarded"
        and _safe_int(awareness.get("risk_bot_count"), 0) == 0
        and _safe_int(awareness.get("hard_risk_bot_count"), 0) == 0
        and blocked_count > 0
        and counted_blocked == blocked_count
        and bool(blocked_statuses)
        and blocked_statuses.issubset(_PAPER_SOAK_ALLOWED_BLOCKED_TEACHER_STATUSES)
    )


def _daily_verify_has_only_evidence_debt(
    artifacts: Dict[str, Dict[str, Any]],
) -> bool:
    daily_summary = artifacts.get("daily_auto_verify", {}).get("summary", {})
    failed = daily_summary.get("effective_failed_checks")
    if not isinstance(failed, list):
        failed = (
            daily_summary.get("failed_checks")
            if isinstance(daily_summary.get("failed_checks"), list)
            else []
        )
    failed_set = {str(item or "").strip() for item in failed if str(item or "").strip()}
    if not failed_set:
        return False
    return bool(
        bool(daily_summary.get("operational_ok", False))
        and failed_set.issubset(_GREEN_SOAK_MANAGED_DAILY_VERIFY_CHECKS)
    )


def _retrain_waiting_on_paper_sample_evidence(
    artifacts: Dict[str, Dict[str, Any]],
) -> bool:
    summary = artifacts.get("retrain_artifact_freshness", {}).get("summary", {})
    failed = summary.get("sample_sufficiency_failed_checks")
    if not isinstance(failed, list):
        categories = (
            summary.get("failure_categories")
            if isinstance(summary.get("failure_categories"), dict)
            else {}
        )
        failed = (
            categories.get("sample_sufficiency")
            if isinstance(categories.get("sample_sufficiency"), list)
            else []
        )
    failed_set = {str(item or "").strip() for item in failed if str(item or "").strip()}
    return "paper_replay" in failed_set


def _bot_quality_autopilot_deferred_for_evidence_collection(
    artifacts: Dict[str, Dict[str, Any]],
) -> bool:
    artifact = artifacts.get("bot_quality_autopilot", {})
    if bool(artifact.get("stale", False)):
        return False
    summary = (
        artifact.get("summary", {}) if isinstance(artifact.get("summary"), dict) else {}
    )
    if str(summary.get("overall_status", "") or "") != "blocked":
        return False
    if _safe_int(summary.get("qualified_teacher_count"), 0) > 0:
        return False
    if _safe_int(summary.get("quality_queue"), 0) <= 0:
        return False

    teacher_status = (
        str(summary.get("teacher_quality_status", "") or "").strip().lower()
    )
    teacher_collecting = teacher_status in {
        "collecting_evidence",
        "evidence_pending",
        "insufficient_evidence",
    }
    teacher_fail_closed = bool(
        teacher_status == "blocked"
        and _teacher_quality_deferred_for_paper_soak(artifacts)
    )
    if not (teacher_collecting or teacher_fail_closed):
        return False

    return bool(
        _safe_int(summary.get("rerouted_targeted_retrain_count"), 0) > 0
        or _safe_int(summary.get("coverage_shortfall_bots"), 0) > 0
        or _safe_int(summary.get("minimum_sample_count"), 0) > 0
        or _retrain_waiting_on_paper_sample_evidence(artifacts)
        or _daily_verify_has_only_evidence_debt(artifacts)
    )


def _infrastructure_autofix_deferred_for_paper_soak(
    artifacts: Dict[str, Dict[str, Any]],
) -> bool:
    summary = artifacts.get("infrastructure_autofix_bot", {}).get("summary", {})
    status = str(summary.get("overall_status", "") or "")
    if status not in {"blocked", "degraded", "inactive"}:
        return False
    if _safe_int(summary.get("operator_followups"), 0) > 0:
        return False
    return _safe_int(summary.get("applyable_repair_count"), 0) <= 10


def _live_runtime_separation_deferred_for_paper_soak(
    artifacts: Dict[str, Dict[str, Any]],
) -> bool:
    payload = _load_json(
        Path(
            str(
                artifacts.get("live_runtime_separation_control", {}).get("path", "")
                or ""
            )
        )
    )
    if str(payload.get("overall_status", "") or "") not in {"degraded", "ready"}:
        return False
    live_plane = (
        payload.get("live_plane") if isinstance(payload.get("live_plane"), dict) else {}
    )
    release_contract = (
        payload.get("release_contract")
        if isinstance(payload.get("release_contract"), dict)
        else {}
    )
    pressure = (
        payload.get("shared_host_pressure")
        if isinstance(payload.get("shared_host_pressure"), dict)
        else {}
    )
    overlay = (
        pressure.get("storage_overlay_relief")
        if isinstance(pressure.get("storage_overlay_relief"), dict)
        else {}
    )
    return bool(
        live_plane.get("ready", False)
        and release_contract.get("live_lane_should_be_read_only", False)
        and release_contract.get("promotions_should_wait_for_cold_lane", False)
        and overlay.get("raw_live_clear", False)
        and _safe_int(pressure.get("restart_storms"), 0) == 0
        and _safe_int(pressure.get("restart_storm_contention_count"), 0) == 0
    )


def _coordination_state_deferred_for_paper_soak(
    artifacts: Dict[str, Dict[str, Any]],
) -> bool:
    payload = _load_json(
        Path(str(artifacts.get("coordination_state_control", {}).get("path", "") or ""))
    )
    if str(payload.get("overall_status", "") or "") != "blocked":
        return False
    policies = (
        payload.get("policies") if isinstance(payload.get("policies"), dict) else {}
    )
    live = (
        policies.get("live_orders")
        if isinstance(policies.get("live_orders"), dict)
        else {}
    )
    paper = (
        policies.get("paper_execution")
        if isinstance(policies.get("paper_execution"), dict)
        else {}
    )
    terminal = (
        policies.get("terminal_restart")
        if isinstance(policies.get("terminal_restart"), dict)
        else {}
    )
    light_livefeed = (
        policies.get("light_livefeed")
        if isinstance(policies.get("light_livefeed"), dict)
        else {}
    )
    live_blockers = {
        str(item or "").strip()
        for item in (
            live.get("blockers") if isinstance(live.get("blockers"), list) else []
        )
        if str(item or "").strip()
    }
    protective_lock = bool(
        not bool(live.get("allowed", True))
        and {
            "paper_trade_lock_active",
            "runtime_release_live_read_only",
            "live_runtime_release_read_only",
        }
        & live_blockers
    )
    return bool(
        protective_lock
        and bool(paper.get("allowed", False))
        and bool(paper.get("paper_trade_lock_active", False))
        and bool(terminal.get("safe", False))
        and bool(light_livefeed.get("allowed", True))
    )


def _storage_quota_deferred_for_paper_soak(
    artifacts: Dict[str, Dict[str, Any]],
) -> bool:
    summary = artifacts.get("storage_quota_guard", {}).get("summary", {})
    if str(summary.get("overall_status", "") or "") not in {"degraded", "ready"}:
        return False
    if _safe_int(summary.get("hard_breaches"), 0) > 0:
        return False
    if bool(
        _safe_int(summary.get("soft_breaches"), 0) <= 1
        and _ingestion_soak_ready_for_dashboard(artifacts)
    ):
        return True
    return _stateful_sql_soft_quota_deferred_for_paper_soak(artifacts)


def _ingestion_storage_governor_deferred_for_paper_soak(
    artifacts: Dict[str, Dict[str, Any]],
) -> bool:
    governor_summary = artifacts.get("ingestion_storage_governor", {}).get(
        "summary", {}
    )
    if str(governor_summary.get("profile", "") or "") != "critical_backpressure":
        return False
    if bool(governor_summary.get("route_drift", False)):
        return False
    storage_payload = _load_json(
        Path(str(artifacts.get("ingestion_storage_control", {}).get("path", "") or ""))
    )
    contract = (
        storage_payload.get("continuous_run_soak_contract")
        if isinstance(storage_payload.get("continuous_run_soak_contract"), dict)
        else {}
    )
    storage = (
        storage_payload.get("storage")
        if isinstance(storage_payload.get("storage"), dict)
        else {}
    )
    forecast = (
        contract.get("forecast") if isinstance(contract.get("forecast"), dict) else {}
    )
    contract_grade = str(contract.get("grade") or "").strip().upper()
    contract_blockers = (
        contract.get("blockers") if isinstance(contract.get("blockers"), list) else []
    )
    return bool(
        str(storage_payload.get("overall_status") or "") in {"ready", "ok"}
        and str(storage_payload.get("severity") or "")
        in {"stable", "low", "normal", ""}
        and bool(contract.get("soak_ready", False) or contract.get("ready", False))
        and not contract_blockers
        and (not contract_grade or contract_grade in {"A", "A+"})
        and str(forecast.get("continuous_run_status") or "ready")
        in {"ready", "stable", "stable_or_improving"}
        and bool(storage.get("sql_primary_route_drift", False)) is False
    )


def _external_backlog_deferred_for_paper_soak(
    item: str,
    artifacts: Dict[str, Dict[str, Any]],
) -> bool:
    if not _ingestion_soak_ready_for_dashboard(artifacts):
        return False
    drain = artifacts.get("external_backlog_drain", {}).get("summary", {})
    retry = artifacts.get("external_backlog_retry_bot", {}).get("summary", {})
    raw_aged_candidate_files = _safe_int(drain.get("aged_candidate_files"), 0)
    effective_aged_candidate_files = raw_aged_candidate_files
    ingestion_payload = _load_json(
        Path(str(artifacts.get("ingestion_storage_control", {}).get("path", "") or ""))
    )
    ingestion_storage = (
        ingestion_payload.get("storage")
        if isinstance(ingestion_payload.get("storage"), dict)
        else {}
    )
    ingestion_backpressure = (
        ingestion_payload.get("backpressure")
        if isinstance(ingestion_payload.get("backpressure"), dict)
        else {}
    )
    normalized_aged_candidate_files = _safe_int(
        ingestion_storage.get("aged_backlog_candidate_files"),
        raw_aged_candidate_files,
    )
    storage_raw_aged_candidate_files = _safe_int(
        ingestion_storage.get("raw_aged_backlog_candidate_files"),
        raw_aged_candidate_files,
    )
    if (
        bool(
            ingestion_storage.get(
                "aged_backlog_candidate_files_suppressed_by_effective_pressure",
                False,
            )
        )
        and bool(ingestion_backpressure.get("effective_pressure_clear", False))
        and storage_raw_aged_candidate_files == raw_aged_candidate_files
        and normalized_aged_candidate_files == 0
    ):
        effective_aged_candidate_files = 0
    if item == "external_backlog_retry_bot_followups":
        if (
            effective_aged_candidate_files == 0
            and _safe_int(drain.get("candidate_files"), 0) == 0
        ):
            return True
        return bool(
            str(retry.get("overall_status", "") or "")
            in {"idle", "ready", "applied_with_followups", "blocked"}
            and not bool(retry.get("actionable", False))
            and not bool(retry.get("backlog_needed", False))
        )
    follow_status = str(drain.get("follow_through_status", "") or "")
    follow_progress = str(drain.get("follow_through_progress_state", "") or "")
    drain_in_progress = follow_status in {
        "handoff_requested",
        "drain_active",
    } or follow_progress in {
        "requested_live_writer",
        "progressing",
    }
    return bool(
        str(drain.get("overall_status", "") or "")
        in {"drain_active", "blocked", "ready"}
        and effective_aged_candidate_files == 0
        and (drain_in_progress or not bool(drain.get("writer_busy", False)))
    )


def _auth_lease_deferred_for_paper_soak(artifacts: Dict[str, Dict[str, Any]]) -> bool:
    auth_artifact = artifacts.get("auth_lease_manager", {})
    supervisor_artifact = artifacts.get("schwab_auth_supervisor", {})
    auth_payload = _load_json(Path(str(auth_artifact.get("path") or "")))
    supervisor_payload = _load_json(Path(str(supervisor_artifact.get("path") or "")))
    lease_budget = (
        auth_payload.get("lease_budget")
        if isinstance(auth_payload.get("lease_budget"), dict)
        else {}
    )
    broker_state = (
        auth_payload.get("broker_state")
        if isinstance(auth_payload.get("broker_state"), dict)
        else {}
    )
    expires_in = _safe_float(lease_budget.get("expires_in_seconds"), 0.0)
    critical_seconds = max(
        _safe_float(lease_budget.get("critical_lease_seconds"), 600.0), 600.0
    )
    readiness_floor = max(critical_seconds, 900.0)
    status = str(auth_payload.get("overall_status") or "").strip().lower()
    lease_state = str(auth_payload.get("lease_state") or "").strip().lower()
    return bool(
        status in {"ready", "degraded"}
        and lease_state in {"healthy", "ready", "ok", "warning"}
        and expires_in >= readiness_floor
        and bool(
            broker_state.get("broker_operable", broker_state.get("broker_ready", False))
        )
        and bool(broker_state.get("network_ok", True) is not False)
        and (
            bool(supervisor_payload.get("paper_soak_auth_operable", False))
            or bool(lease_budget.get("token_lease_grace", False))
        )
    )


def _attention_managed_by_green_soak(
    item: str,
    artifacts: Dict[str, Dict[str, Any]],
    context: dict[str, Any],
) -> str:
    if not bool(context.get("enabled", False)):
        return ""
    reason = _GREEN_SOAK_MANAGED_ATTENTION_REASONS.get(str(item or "").strip())
    if not reason:
        return ""
    if (
        item == "runtime_snapshot_cache_control_needs_work"
        and not _snapshot_cache_ready_for_soak(artifacts)
    ):
        return ""
    if (
        item == "roster_resilience_planner_needs_work"
        and not _roster_resilience_ready_for_soak(artifacts)
    ):
        return ""
    if (
        item == "daily_auto_verify_not_ok"
        and not _daily_verify_deferred_for_paper_soak(artifacts)
    ):
        return ""
    if (
        item == "training_quality_control_blocked"
        and not _training_quality_deferred_for_paper_soak(artifacts)
    ):
        return ""
    if (
        item == "teacher_quality_guard_blocked"
        and not _teacher_quality_deferred_for_paper_soak(artifacts)
    ):
        return ""
    if item in {
        "infrastructure_autofix_bot_blocked",
        "infrastructure_autofix_bot_needs_work",
    } and not _infrastructure_autofix_deferred_for_paper_soak(artifacts):
        return ""
    if (
        item == "auth_lease_manager_needs_work"
        and not _auth_lease_deferred_for_paper_soak(artifacts)
    ):
        return ""
    if (
        item == "live_runtime_separation_control_needs_work"
        and not _live_runtime_separation_deferred_for_paper_soak(artifacts)
    ):
        return ""
    if (
        item == "coordination_state_control_blocked"
        and not _coordination_state_deferred_for_paper_soak(artifacts)
    ):
        return ""
    if (
        item == "storage_quota_guard_needs_work"
        and not _storage_quota_deferred_for_paper_soak(artifacts)
    ):
        return ""
    if (
        item == "ingestion_storage_governor_critical"
        and not _ingestion_storage_governor_deferred_for_paper_soak(artifacts)
    ):
        return ""
    if item.startswith(
        "external_backlog_"
    ) and not _external_backlog_deferred_for_paper_soak(item, artifacts):
        return ""
    return reason


def _split_green_soak_managed_attention(
    attention: list[str],
    artifacts: Dict[str, Dict[str, Any]],
    context: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    active: list[str] = []
    managed: list[dict[str, Any]] = []
    for item in attention:
        reason = _attention_managed_by_green_soak(item, artifacts, context)
        if not reason:
            active.append(item)
            continue
        managed.append(
            {
                "attention": item,
                "managed_control_state": reason,
                "managed_by": "unattended_soak_readiness",
                "soak_ready": bool(context.get("soak_ready", False)),
                "paper_guard_clean": bool(context.get("paper_guard_clean", False)),
                "paper_stage": str(context.get("paper_stage") or ""),
                "paper_armed": bool(context.get("paper_armed", False)),
                "action_policy": "keep_visible_but_do_not_degrade_dashboard_until_soak_or_paper_guard_fails",
                "when_to_unmanage": (
                    "surface as dashboard attention if unattended soak is no longer ready, paper regression guards fail, "
                    "health-fast is not ready, or the item becomes a safety/storage/auth/live-paper blocker."
                ),
            }
        )
    return active, managed


def _attention_tier(item: str) -> str:
    text = str(item or "").strip()
    if not text:
        return "advisory"
    if text in _ADVISORY_ATTENTION:
        return "advisory"
    if text in _CRITICAL_ATTENTION or text.endswith("_missing"):
        return "critical"
    if text in _DEGRADED_ATTENTION:
        return "degraded"
    if text.endswith("_stale") or text.endswith("_not_ok"):
        return "degraded"
    if text.endswith("_blocked"):
        return "degraded"
    return "watch"


def _attention_tiers(attention: list[str]) -> dict[str, list[str]]:
    tiers: dict[str, list[str]] = {
        "critical": [],
        "degraded": [],
        "watch": [],
        "advisory": [],
    }
    for item in attention:
        tiers.setdefault(_attention_tier(item), []).append(item)
    return tiers


def _severity_from_attention(attention: list[str]) -> int:
    tiers = _attention_tiers(attention)
    if tiers["critical"]:
        return 3
    if tiers["degraded"]:
        return 2
    if tiers["watch"]:
        return 1
    return 0


def _remediation_actions(attention: list[str]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in attention:
        key = str(item or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        action = dict(_ATTENTION_OWNER_ACTIONS.get(key) or {})
        if not action:
            action = {
                "owner": "operator_review",
                "command": ["./scripts/ops/opsctl.sh", "dashboard", "--json"],
                "timeout_seconds": 60,
                "success_condition": "owner mapping required before this item can block operational status",
            }
        action["attention"] = key
        action["tier"] = _attention_tier(key)
        actions.append(action)
    return actions


def _ordered_unique(items: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in items:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _containment_rule_for_attention(item: str) -> Dict[str, Any]:
    key = str(item or "").strip()
    exact: Dict[str, Dict[str, Any]] = {
        "paper_execution_safety_guard_active": {
            "domain": "execution_safety",
            "blast_radius": "paper_execution_only",
            "trading_impact": "paper_order_submission_blocked",
            "data_collection_impact": "none",
            "containment_action": "keep_collection_fanout_running_and_require_gradeable_post_cost_fill_evidence_before_paper_execution",
            "release_condition": "paper_execution_safety_guard_clears_after_profitability_and_execution_evidence_are_gradeable",
        },
        "promotion_not_ready": {
            "domain": "training_promotion",
            "blast_radius": "promotion_only",
            "trading_impact": "live_promotion_blocked",
            "data_collection_impact": "none",
            "containment_action": "freeze_promotion_path_without_stopping_market_observation",
            "release_condition": "promotion_quality_gate_promote_ok_true",
        },
        "retrain_artifact_freshness_not_ok": {
            "domain": "training_promotion",
            "blast_radius": "training_refresh_only",
            "trading_impact": "new_model_promotion_blocked",
            "data_collection_impact": "none",
            "containment_action": "defer_training_refresh_until_storage_and_sample_inputs_are_clean",
            "release_condition": "retrain_artifact_freshness_latest_ok_true",
        },
        "training_quality_control_blocked": {
            "domain": "training_promotion",
            "blast_radius": "training_quality_only",
            "trading_impact": "promotion_blocked",
            "data_collection_impact": "none",
            "containment_action": "fail_closed_on_training_quality_and_continue_evidence_collection",
            "release_condition": "training_quality_control_overall_status_ready",
        },
        "source_verification_context_debt": {
            "domain": "market_context",
            "blast_radius": "context_confidence_only",
            "trading_impact": "context_claims_and_promotion_blocked",
            "data_collection_impact": "none",
            "containment_action": "keep_decision_critical_sources_usable_and_downweight_unverified_context_layers",
            "release_condition": "source_verification_all_sources_verified_or_context_debt_empty",
        },
        "source_verification_decision_critical_blocked": {
            "domain": "market_context",
            "blast_radius": "decision_runtime",
            "trading_impact": "paper_and_live_decisions_blocked",
            "data_collection_impact": "context_collection_only",
            "containment_action": "block_market_context_dependent_decisions_until_decision_critical_sources_recover",
            "release_condition": "source_runtime_contract_decision_critical_sources_ready_true",
        },
        "external_backlog_drain_recommended": {
            "domain": "storage_backlog",
            "blast_radius": "cold_or_deferred_backlog",
            "trading_impact": "none_while_ingestion_storage_control_is_stable",
            "data_collection_impact": "bounded_background_drain",
            "containment_action": "drain_aged_candidates_without_stopping_hot_path_collection",
            "release_condition": "external_backlog_drain_recommended_now_false_and_aged_candidates_zero",
        },
        "external_backlog_drain_writer_busy": {
            "domain": "storage_backlog",
            "blast_radius": "writer_handoff_lane",
            "trading_impact": "none_while_queue_watermarks_are_clear",
            "data_collection_impact": "handoff_waiting_or_background_drain",
            "containment_action": "coordinate_writer_handoff_and_keep_storage_watermarks_authoritative",
            "release_condition": "writer_busy_false_or_follow_through_progress_observed",
        },
        "external_backlog_retry_bot_followups": {
            "domain": "storage_backlog",
            "blast_radius": "retry_lane",
            "trading_impact": "none_while_storage_pressure_is_stable",
            "data_collection_impact": "background_retry_only",
            "containment_action": "retry_backlog_handoff_without_escalating_to_hot_path_failure",
            "release_condition": "external_backlog_retry_bot_ready_or_no_actionable_followups",
        },
        "infrastructure_autofix_bot_needs_work": {
            "domain": "ops_self_healing",
            "blast_radius": "automatic_repair_lane",
            "trading_impact": "none_unless_owner_command_fails",
            "data_collection_impact": "none",
            "containment_action": "leave_runtime_hot_path_untouched_and_route_repairs_through_bounded_owner_commands",
            "release_condition": "infrastructure_autofix_bot_ready_or_no_runtime_gate_depends_on_it",
        },
        "roster_resilience_planner_needs_work": {
            "domain": "bot_roster",
            "blast_radius": "bench_and_replacement_planning",
            "trading_impact": "scaling_blocked_not_current_collection",
            "data_collection_impact": "none",
            "containment_action": "hold_roster_expansion_until_supportability_and_bench_depth_recover",
            "release_condition": "roster_resilience_planner_operational_ready_and_status_ready",
        },
        "ingestion_storage_control_blocked": {
            "domain": "storage_hot_path",
            "blast_radius": "data_ingestion",
            "trading_impact": "paper_and_training_blocked",
            "data_collection_impact": "write_path_blocked",
            "containment_action": "throttle_or_pause_writers_and_run_storage_backpressure_autopilot",
            "release_condition": "ingestion_storage_control_ready_with_pressure_below_elevated",
        },
        "ingestion_storage_control_elevated": {
            "domain": "storage_hot_path",
            "blast_radius": "data_ingestion",
            "trading_impact": "training_and_promotion_delayed",
            "data_collection_impact": "throttled",
            "containment_action": "keep_writers_bounded_and_drain_until_pressure_returns_to_stable",
            "release_condition": "storage_pressure_index_below_0_25_and_no_hard_breaches",
        },
        "ingestion_storage_governor_critical": {
            "domain": "storage_hot_path",
            "blast_radius": "data_ingestion",
            "trading_impact": "paper_and_training_blocked",
            "data_collection_impact": "write_path_throttled",
            "containment_action": "apply_governor_profile_and_drain_backlog_before_expansion",
            "release_condition": "ingestion_storage_governor_profile_not_critical_backpressure",
        },
        "storage_split_brain_needs_review": {
            "domain": "storage_integrity",
            "blast_radius": "stateful_storage_routes",
            "trading_impact": "execution_and_training_blocked_until_reconciled",
            "data_collection_impact": "route_specific",
            "containment_action": "isolate_conflicting_storage_routes_and_require_hash_reconciliation",
            "release_condition": "unresolved_split_brain_conflicts_zero",
        },
        "global_killswitch_not_ok": {
            "domain": "runtime_safety",
            "blast_radius": "whole_platform",
            "trading_impact": "all_execution_blocked",
            "data_collection_impact": "possibly_blocked",
            "containment_action": "honor_global_halt_until_clear_all_halts_reports_ready",
            "release_condition": "global_killswitch_halt_false",
        },
        "health_gates_hard_gate_triggered": {
            "domain": "runtime_safety",
            "blast_radius": "whole_platform",
            "trading_impact": "all_execution_blocked",
            "data_collection_impact": "possibly_blocked",
            "containment_action": "treat_hard_gate_as_authoritative_until_health_gates_clear",
            "release_condition": "health_gates_hard_gate_triggered_false",
        },
    }
    if key in exact:
        return dict(exact[key])
    if key.endswith("_missing"):
        return {
            "domain": "artifact_freshness",
            "blast_radius": "unknown_until_artifact_exists",
            "trading_impact": "blocked_if_required",
            "data_collection_impact": "unknown",
            "containment_action": "rebuild_missing_artifact_before_trusting_downstream_status",
            "release_condition": "required_artifact_exists_and_is_fresh",
        }
    if key.endswith("_stale"):
        return {
            "domain": "artifact_freshness",
            "blast_radius": "status_visibility",
            "trading_impact": "promotion_blocked_until_fresh",
            "data_collection_impact": "none_if_runtime_heartbeat_is_fresh",
            "containment_action": "refresh_owner_artifact_or_accept_authoritative_live_heartbeat",
            "release_condition": "artifact_age_inside_slo",
        }
    if key.endswith("_not_ok") or key.endswith("_blocked"):
        return {
            "domain": "owner_control",
            "blast_radius": "owner_surface",
            "trading_impact": "blocked_for_that_surface",
            "data_collection_impact": "surface_specific",
            "containment_action": "route_to_owner_command_and_keep_unrelated_lanes_running",
            "release_condition": "owner_surface_status_ready",
        }
    return {
        "domain": "watch",
        "blast_radius": "unknown",
        "trading_impact": "review_required",
        "data_collection_impact": "unknown",
        "containment_action": "classify_owner_before_allowing_escalation",
        "release_condition": "owner_mapping_added_or_attention_clears",
    }


def _degradation_containment_contract(
    attention: list[str],
    attention_tiers: Dict[str, list[str]],
    artifacts: Dict[str, Dict[str, Any]],
    managed_controls: list[dict[str, Any]],
) -> Dict[str, Any]:
    rows: list[Dict[str, Any]] = []
    for item in attention:
        key = str(item or "").strip()
        if not key:
            continue
        rule = _containment_rule_for_attention(key)
        tier = _attention_tier(key)
        action = dict(_ATTENTION_OWNER_ACTIONS.get(key) or {})
        contained = bool(key in _CONTAINED_ATTENTION and tier in {"advisory", "watch"})
        if rule.get("domain") in {
            "runtime_safety",
            "storage_hot_path",
            "storage_integrity",
        }:
            contained = False
        rows.append(
            {
                "attention": key,
                "tier": tier,
                "contained": contained,
                "domain": str(rule.get("domain") or "unknown"),
                "blast_radius": str(rule.get("blast_radius") or "unknown"),
                "trading_impact": str(rule.get("trading_impact") or "review_required"),
                "data_collection_impact": str(
                    rule.get("data_collection_impact") or "unknown"
                ),
                "containment_action": str(rule.get("containment_action") or ""),
                "release_condition": str(rule.get("release_condition") or ""),
                "owner": str(action.get("owner") or "operator_review"),
                "next_command": (
                    action.get("command")
                    if isinstance(action.get("command"), list)
                    else []
                ),
            }
        )
    for managed in managed_controls:
        key = str(managed.get("attention") or "").strip()
        if not key:
            continue
        rule = _containment_rule_for_attention(key)
        rows.append(
            {
                "attention": key,
                "tier": "managed",
                "contained": True,
                "domain": str(rule.get("domain") or "paper_soak_managed"),
                "blast_radius": str(rule.get("blast_radius") or "managed_attention"),
                "trading_impact": str(rule.get("trading_impact") or "none"),
                "data_collection_impact": "none",
                "containment_action": str(managed.get("managed_control_state") or ""),
                "release_condition": str(managed.get("when_to_unmanage") or ""),
                "owner": str(managed.get("managed_by") or "unattended_soak_readiness"),
                "next_command": [],
            }
        )
    uncontained = [row for row in rows if not bool(row.get("contained", False))]
    uncontained_domains = sorted(
        {
            str(row.get("domain") or "")
            for row in uncontained
            if str(row.get("domain") or "")
        }
    )
    hot_path_domains = {"runtime_safety", "storage_hot_path", "storage_integrity"}
    return {
        "status": (
            "contained"
            if rows and not uncontained
            else ("clear" if not rows else "uncontained")
        ),
        "contained_count": sum(1 for row in rows if bool(row.get("contained", False))),
        "uncontained_count": len(uncontained),
        "uncontained_domains": uncontained_domains,
        "critical_count": len(attention_tiers.get("critical", [])),
        "degraded_count": len(attention_tiers.get("degraded", [])),
        "watch_count": len(attention_tiers.get("watch", [])),
        "advisory_count": len(attention_tiers.get("advisory", [])),
        "hot_path_blocked": any(
            domain in hot_path_domains for domain in uncontained_domains
        ),
        "containment_rows": rows,
        "policy": "each degraded signal declares owner_domain_blast_radius_and_release_condition_so_unrelated_lanes_keep_running",
    }


def _ops_smoothing_contract(
    artifacts: Dict[str, Dict[str, Any]],
    attention: list[str],
    attention_tiers: Dict[str, list[str]],
    managed_controls: list[dict[str, Any]],
) -> Dict[str, Any]:
    launcher = artifacts.get("all_sleeves_launcher", {}).get("summary", {})
    storage = artifacts.get("ingestion_storage_control", {}).get("summary", {})
    source = artifacts.get("source_verification", {}).get("summary", {})
    bot_org = artifacts.get("bot_organization_control", {}).get("summary", {})
    profitability = artifacts.get("bot_profitability_scalability_control", {}).get(
        "summary", {}
    )
    sleeve = artifacts.get("sleeve_scalability_selector", {}).get("summary", {})
    master = artifacts.get("master_grandmaster_evidence_v2", {}).get("summary", {})
    training = artifacts.get("training_report", {}).get("summary", {})
    training_quality = artifacts.get("training_quality_control", {}).get("summary", {})
    health = artifacts.get("health_gates", {}).get("summary", {})
    killswitch = artifacts.get("global_killswitch", {}).get("summary", {})
    cockpit = artifacts.get("operator_cockpit", {}).get("summary", {})

    storage_ready = bool(
        str(storage.get("overall_status") or "").strip().lower() == "ready"
        and str(storage.get("severity") or "").strip().lower()
        in {"", "stable", "low", "normal", "ready"}
    )
    collection_fanout_ready = bool(launcher.get("collection_fanout_ready", False))
    health_clear = not bool(health.get("hard_gate_triggered", False))
    halt_clear = not bool(killswitch.get("halt", False))
    critical_or_degraded_hot_path = [
        item
        for item in attention_tiers.get("critical", [])
        + attention_tiers.get("degraded", [])
        if _containment_rule_for_attention(item).get("domain")
        in {"runtime_safety", "storage_hot_path", "storage_integrity"}
    ]
    safe_to_collect_evidence = bool(
        collection_fanout_ready and storage_ready and health_clear and halt_clear
    )
    source_context_debt = _ordered_unique(
        [
            *[
                str(item)
                for item in source.get("decision_context_debt", [])
                if str(item).strip()
            ],
            *[
                str(item)
                for item in source.get("optional_enrichment_debt", [])
                if str(item).strip()
            ],
        ]
    )
    evidence_debt_surfaces = _ordered_unique(
        [
            name
            for name, summary in {
                "bot_organization_control": bot_org,
                "bot_profitability_scalability_control": profitability,
                "sleeve_scalability_selector": sleeve,
                "master_grandmaster_evidence_v2": master,
            }.items()
            if str(summary.get("overall_status") or "").strip().lower()
            in {"ready_with_evidence_debt", "ready_with_review_debt"}
        ]
    )
    promotion_blockers = _ordered_unique(
        [
            *[
                str(item)
                for item in master.get("promotion_blockers", [])
                if str(item).strip()
            ],
            *(["promotion_not_ready"] if "promotion_not_ready" in attention else []),
        ]
    )
    safe_to_promote_or_live_trade = bool(
        safe_to_collect_evidence
        and not evidence_debt_surfaces
        and not source_context_debt
        and not promotion_blockers
        and bool(launcher.get("paper_execution_ready", False))
        and bool(profitability.get("live_allocation_ready", False))
        and bool(sleeve.get("recommendation_ready", False))
        and bool(master.get("automatic_live_promotion_allowed", False))
    )
    lanes = {
        "dashboard_warning": {
            "status": "clear" if not attention else "contained_attention",
            "attention_count": len(attention),
            "critical_or_degraded_hot_path": critical_or_degraded_hot_path,
        },
        "operator_cockpit": {
            "status": str(cockpit.get("overall_status") or "missing"),
            "contained_by": "runtime_dashboard_degradation_containment",
        },
        "storage_reserve": {
            "status": (
                "ready"
                if storage_ready
                else str(storage.get("overall_status") or "missing")
            ),
            "pressure_index": float(storage.get("pressure_index", 0.0) or 0.0),
            "severity": str(storage.get("severity") or ""),
        },
        "soak_unattended": {
            "status": (
                "blocked"
                if "paper_execution_safety_guard_active" in attention
                else "unknown"
            ),
            "paper_execution_ready": bool(launcher.get("paper_execution_ready", False)),
        },
        "bot_organization": {
            "status": str(bot_org.get("overall_status") or "missing"),
            "review_queue_count": int(bot_org.get("review_queue_count", 0) or 0),
            "blocking_tripwire_count": int(
                bot_org.get("blocking_tripwire_count", 0) or 0
            ),
        },
        "profitability_evidence": {
            "status": str(profitability.get("overall_status") or "missing"),
            "evidence_debt_count": int(
                profitability.get("evidence_debt_count", 0) or 0
            ),
            "profitability_claim_ready": bool(
                profitability.get("profitability_claim_ready", False)
            ),
        },
        "sleeve_selection": {
            "status": str(sleeve.get("overall_status") or "missing"),
            "application_eligible_sleeve_count": int(
                sleeve.get("application_eligible_sleeve_count", 0) or 0
            ),
            "recommendation_ready": bool(sleeve.get("recommendation_ready", False)),
        },
        "master_grandmaster": {
            "status": str(master.get("overall_status") or "missing"),
            "structural_grade": str(master.get("structural_grade") or ""),
            "evidence_grade": str(master.get("evidence_grade") or ""),
            "promotion_blocker_count": len(promotion_blockers),
        },
        "source_verification": {
            "status": str(source.get("overall_status") or "missing"),
            "decision_critical_sources_ready": bool(
                source.get("decision_critical_sources_ready", False)
            ),
            "context_debt": source_context_debt,
        },
        "training_promotion": {
            "status": str(
                training_quality.get("overall_status")
                or training.get("overall_status")
                or "missing"
            ),
            "promotion_blockers": promotion_blockers,
        },
    }
    return {
        "status": (
            "ready_to_collect_evidence"
            if safe_to_collect_evidence and not safe_to_promote_or_live_trade
            else (
                "ready_for_promotion"
                if safe_to_promote_or_live_trade
                else "needs_runtime_attention"
            )
        ),
        "safe_to_collect_evidence": safe_to_collect_evidence,
        "safe_to_promote_or_live_trade": safe_to_promote_or_live_trade,
        "paper_execution_ready": bool(launcher.get("paper_execution_ready", False)),
        "collection_fanout_ready": collection_fanout_ready,
        "evidence_debt_surfaces": evidence_debt_surfaces,
        "source_context_debt": source_context_debt,
        "promotion_blockers": promotion_blockers,
        "managed_attention_count": len(managed_controls),
        "lanes": lanes,
        "policy": "runtime_collection_promotion_live_trading_and_ops_repair_degrade_independently",
    }


def _registry_summary(project_root: Path) -> Dict[str, Any]:
    registry = _load_json(project_root / "master_bot_registry.json")
    summary = (
        registry.get("summary") if isinstance(registry.get("summary"), dict) else {}
    )
    sub_bots = (
        registry.get("sub_bots") if isinstance(registry.get("sub_bots"), list) else []
    )
    active_rows = [
        row for row in sub_bots if isinstance(row, dict) and bool(row.get("active"))
    ]
    deleted_rows = [
        row
        for row in sub_bots
        if isinstance(row, dict) and bool(row.get("deleted_from_rotation"))
    ]
    return {
        "updated_at_utc": str(registry.get("updated_at_utc", "") or ""),
        "total_bots": int(summary.get("total_bots", len(sub_bots)) or len(sub_bots)),
        "active_bots": int(
            summary.get("active_bots", len(active_rows)) or len(active_rows)
        ),
        "deleted_from_rotation": int(
            summary.get("deleted_from_rotation", len(deleted_rows)) or len(deleted_rows)
        ),
        "top_active": (
            summary.get("top_active")
            if isinstance(summary.get("top_active"), list)
            else []
        ),
        "deletion_guard_ok": bool(summary.get("deletion_guard_ok", False)),
        "deletion_guard_reason": str(summary.get("deletion_guard_reason", "") or ""),
    }


def _resolved_daily_auto_verify_failures(
    daily_verify_payload: Dict[str, Any],
    artifacts: Dict[str, Dict[str, Any]],
) -> tuple[list[str], list[str]]:
    failed = (
        daily_verify_payload.get("failed_checks")
        if isinstance(daily_verify_payload.get("failed_checks"), list)
        else []
    )
    checks = (
        daily_verify_payload.get("checks")
        if isinstance(daily_verify_payload.get("checks"), dict)
        else {}
    )
    unresolved: list[str] = []
    resolved: list[str] = []
    for name in failed:
        key = str(name or "").strip()
        resolver = _DAILY_AUTO_VERIFY_RESOLVERS.get(key)
        if resolver is not None and resolver(daily_verify_payload, artifacts, checks):
            resolved.append(key)
            continue
        unresolved.append(key)
    return unresolved, resolved


def _artifact_freshness_recovered(payload: Dict[str, Any]) -> bool:
    if not isinstance(payload, dict) or not payload:
        return False
    max_age_minutes = float(payload.get("max_age_minutes", 0.0) or 0.0)
    fresh_if_newer_than = _parse_iso_utc(payload.get("fresh_if_newer_than_utc"))
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    paths: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        path_text = str(row.get("path", "") or "").strip()
        if path_text:
            paths.append(path_text)
    if not paths:
        for item in (
            payload.get("stale_files")
            if isinstance(payload.get("stale_files"), list)
            else []
        ):
            path_text = str(item or "").strip()
            if path_text:
                paths.append(path_text)
        for item in (
            payload.get("missing_files")
            if isinstance(payload.get("missing_files"), list)
            else []
        ):
            path_text = str(item or "").strip()
            if path_text:
                paths.append(path_text)
    if not paths:
        return False

    now = datetime.now(timezone.utc)
    for path_text in paths:
        path = Path(path_text)
        if not path.exists():
            return False
        current_payload = _load_json(path)
        ts = _payload_timestamp(current_payload, path)
        if ts is None:
            return False
        if fresh_if_newer_than is not None and ts >= fresh_if_newer_than:
            continue
        age_minutes = max((now - ts).total_seconds() / 60.0, 0.0)
        if max_age_minutes <= 0.0 or age_minutes > max_age_minutes:
            return False
    return True


def _sql_plane_freshness_from_storage_control(
    artifacts: Dict[str, Dict[str, Any]],
) -> dict[str, Any]:
    storage_artifact = artifacts.get("ingestion_storage_control", {})
    if not bool(storage_artifact.get("exists")) or bool(storage_artifact.get("stale")):
        return {}
    storage_path = Path(str(storage_artifact.get("path") or ""))
    storage_payload = _load_json(storage_path)
    if not storage_payload:
        return {}

    overlay = (
        storage_payload.get("sql_ingestion_pending_overlay")
        if isinstance(storage_payload.get("sql_ingestion_pending_overlay"), dict)
        else {}
    )
    continuous = (
        storage_payload.get("continuous_run_soak_contract")
        if isinstance(storage_payload.get("continuous_run_soak_contract"), dict)
        else {}
    )
    backpressure = (
        storage_payload.get("backpressure")
        if isinstance(storage_payload.get("backpressure"), dict)
        else {}
    )
    raw_live, raw_live_source = _overlay_raw_live_candidate(backpressure)
    raw_live_evidence = bool(backpressure and raw_live)
    raw_core = _safe_int(raw_live.get("core_pending_lines"), 0)
    raw_total = _safe_int(raw_live.get("total_pending_lines"), raw_core)
    raw_oldest = _safe_float(raw_live.get("oldest_pending_age_seconds"), 0.0)
    service_artifact = artifacts.get("sql_link_service", {})
    service_summary = (
        service_artifact.get("summary")
        if isinstance(service_artifact.get("summary"), dict)
        else {}
    )
    service_idle_complete = bool(
        service_artifact.get("ok") is True
        and not bool(service_summary.get("running", False))
        and str(service_summary.get("current_step") or "").strip().lower()
        in {"complete", "idle", "done"}
    )
    sql_ingestion_artifact = artifacts.get("sql_ingestion", {})
    sql_ingestion_summary = (
        sql_ingestion_artifact.get("summary")
        if isinstance(sql_ingestion_artifact.get("summary"), dict)
        else {}
    )
    sql_ingestion_summary_clear = bool(
        bool(sql_ingestion_artifact.get("exists"))
        and _safe_int(sql_ingestion_summary.get("pending_lines"), 0) == 0
        and _safe_int(sql_ingestion_summary.get("invalid_lines"), 0) == 0
    )
    overlay_clear = bool(
        bool(overlay.get("active", False))
        and _safe_int(overlay.get("fresh_source_count"), 0) > 0
        and _safe_int(overlay.get("fresh_pending_unknown_source_count"), 0) == 0
        and _safe_int(overlay.get("total_pending_lines"), 0) == 0
        and _safe_int(overlay.get("files_with_pending"), 0) == 0
        and _safe_int(overlay.get("invalid_lines"), 0) == 0
        and _safe_int(overlay.get("stale_pending_lines"), 0) == 0
        and _safe_int(overlay.get("ops_write_failures"), 0) == 0
    )
    storage_ready = bool(
        str(storage_payload.get("overall_status") or "").strip().lower() == "ready"
        and str(storage_payload.get("severity") or "").strip().lower()
        in {"stable", "low", "normal", ""}
    )
    raw_live_clear = bool(
        raw_live_evidence
        and raw_core <= OVERLAY_RAW_LIVE_MAX_CORE_LINES
        and raw_total <= OVERLAY_RAW_LIVE_MAX_TOTAL_LINES
        and raw_oldest <= OVERLAY_RAW_LIVE_MAX_AGE_SECONDS
    )
    data_integrity = (
        storage_payload.get("data_integrity")
        if isinstance(storage_payload.get("data_integrity"), dict)
        else {}
    )
    data_integrity_clear = all(
        _safe_int(data_integrity.get(key), 0) == 0
        for key in (
            "sql_invalid_lines",
            "sql_overlay_invalid_lines",
            "sql_overlay_ops_write_failures",
        )
    )
    writer_shedding = (
        storage_payload.get("writer_shedding")
        if isinstance(storage_payload.get("writer_shedding"), dict)
        else {}
    )
    writer_shedding_clear = bool(
        not (
            writer_shedding.get("hard_breaches")
            if isinstance(writer_shedding.get("hard_breaches"), list)
            else []
        )
        and not (
            writer_shedding.get("elevated_breaches")
            if isinstance(writer_shedding.get("elevated_breaches"), list)
            else []
        )
    )
    continuous_blockers = sorted(
        {
            str(item or "").strip()
            for item in (
                continuous.get("blockers")
                if isinstance(continuous.get("blockers"), list)
                else []
            )
            if str(item or "").strip()
        }
    )
    managed_continuous_blockers = {
        "steady_state_targets_not_clear",
        "backlog_relief_contract_active",
        "drain_time_above_target",
    }
    continuous_ready = not continuous or bool(
        continuous.get("ready") is True and continuous.get("soak_ready") is not False
    )
    continuous_managed_for_sql = bool(
        raw_live_clear
        and continuous_blockers
        and set(continuous_blockers).issubset(managed_continuous_blockers)
    )
    continuous_ok_for_sql_freshness = bool(
        continuous_ready or continuous_managed_for_sql
    )
    backpressure_clear = bool(
        raw_live_clear
        and sql_ingestion_summary_clear
        and data_integrity_clear
        and writer_shedding_clear
    )
    if not (
        service_idle_complete
        and storage_ready
        and continuous_ok_for_sql_freshness
        and (overlay_clear or backpressure_clear)
    ):
        return {}
    source = (
        "ingestion_storage_control_sql_plane_overlay"
        if overlay_clear
        else "ingestion_storage_control_reconciled_backpressure"
    )
    return {
        "source": source,
        "storage_control_path": str(storage_path),
        "storage_control_status": str(storage_payload.get("overall_status") or ""),
        "storage_control_severity": str(storage_payload.get("severity") or ""),
        "overlay_fresh_source_count": _safe_int(overlay.get("fresh_source_count"), 0),
        "overlay_total_pending_lines": _safe_int(overlay.get("total_pending_lines"), 0),
        "overlay_files_with_pending": _safe_int(overlay.get("files_with_pending"), 0),
        "raw_live_source": raw_live_source,
        "raw_live_core_pending_lines": raw_core,
        "raw_live_total_pending_lines": raw_total,
        "raw_live_oldest_pending_age_seconds": raw_oldest,
        "sql_ingestion_summary_clear": sql_ingestion_summary_clear,
        "data_integrity_clear": data_integrity_clear,
        "writer_shedding_clear": writer_shedding_clear,
        "continuous_soak_ready": continuous_ready,
        "continuous_soak_blockers_managed_for_sql_freshness": continuous_managed_for_sql,
        "managed_continuous_soak_blockers": (
            continuous_blockers if continuous_managed_for_sql else []
        ),
        "service_current_step": str(service_summary.get("current_step") or ""),
        "service_idle_complete": True,
    }


DailyVerifyResolver = Callable[
    [Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, Any]], bool
]


def _artifact_ok_resolver(artifact_name: str) -> DailyVerifyResolver:
    def _resolver(
        _daily_verify_payload: Dict[str, Any],
        artifacts: Dict[str, Dict[str, Any]],
        _checks: Dict[str, Any],
    ) -> bool:
        return artifacts.get(artifact_name, {}).get("ok") is True

    return _resolver


def _promotion_packet_builder_resolver(
    _daily_verify_payload: Dict[str, Any],
    artifacts: Dict[str, Dict[str, Any]],
    _checks: Dict[str, Any],
) -> bool:
    artifact = artifacts.get("promotion_packet", {})
    if artifact.get("ok") is True:
        return True
    path = Path(str(artifact.get("path") or ""))
    packet = _load_json(path) if str(path) else {}
    if not packet:
        return False
    gate_results = (
        packet.get("gate_results")
        if isinstance(packet.get("gate_results"), dict)
        else {}
    )
    failed_gates = {str(key) for key, value in gate_results.items() if not bool(value)}
    replayability = (
        packet.get("replayability_contract")
        if isinstance(packet.get("replayability_contract"), dict)
        else {}
    )
    signature = (
        packet.get("signature") if isinstance(packet.get("signature"), dict) else {}
    )
    return bool(
        packet.get("committee_packet_seed_ready", False)
        and bool(packet.get("signing_material_ready", False))
        and bool(signature.get("verified", False))
        and bool(packet.get("trained_models_complete", False))
        and bool(replayability.get("hash_bundle_complete", False))
        and bool(replayability.get("exact_replay_ready", False))
        and failed_gates
        and failed_gates.issubset({"training_success_confirmed"})
    )


def _nightly_resilience_resolver(
    _daily_verify_payload: Dict[str, Any],
    artifacts: Dict[str, Dict[str, Any]],
    _checks: Dict[str, Any],
) -> bool:
    nightly = artifacts.get("nightly_resilience", {})
    return nightly.get("ok") is True and not bool(nightly.get("stale", False))


def _artifact_freshness_resolver(
    _daily_verify_payload: Dict[str, Any],
    artifacts: Dict[str, Dict[str, Any]],
    checks: Dict[str, Any],
) -> bool:
    slo = artifacts.get("artifact_freshness_slo", {})
    slo_summary = slo.get("summary") if isinstance(slo.get("summary"), dict) else {}
    sla_summary = (
        slo_summary.get("sla_summary")
        if isinstance(slo_summary.get("sla_summary"), dict)
        else {}
    )
    if (
        slo.get("ok") is True
        and not bool(slo.get("stale", False))
        and str(slo_summary.get("overall_status", "") or "") in {"ready", "ok", ""}
        and _safe_int(
            slo_summary.get(
                "stale_required", _safe_int(sla_summary.get("stale_required"), 0)
            ),
            0,
        )
        <= 0
    ):
        return True
    freshness = (
        checks.get("artifact_freshness")
        if isinstance(checks.get("artifact_freshness"), dict)
        else {}
    )
    return _artifact_freshness_recovered(freshness)


def _db_integrity_resolver(
    _daily_verify_payload: Dict[str, Any],
    artifacts: Dict[str, Dict[str, Any]],
    _checks: Dict[str, Any],
) -> bool:
    maintenance = artifacts.get("sqlite_maintenance", {})
    summary = (
        maintenance.get("summary")
        if isinstance(maintenance.get("summary"), dict)
        else {}
    )
    return bool(
        maintenance.get("ok") is True
        and not bool(maintenance.get("stale", False))
        and summary.get("ok", True) is not False
        and not bool(summary.get("timed_out", False))
        and str(summary.get("current_step", "complete") or "complete") == "complete"
    )


def _incomplete_run_recovered_resolver(
    daily_verify_payload: Dict[str, Any],
    _artifacts: Dict[str, Dict[str, Any]],
    _checks: Dict[str, Any],
) -> bool:
    note = str(daily_verify_payload.get("note", "") or "").lower()
    if "recovered_stale_progress" in note:
        return True
    return bool(
        daily_verify_payload.get("running") is False
        and int(daily_verify_payload.get("completed_checks", 0) or 0) > 0
    )


_DAILY_AUTO_VERIFY_RESOLVERS: Dict[str, DailyVerifyResolver] = {
    "new_bot_graduation_gate": _artifact_ok_resolver("new_bot_graduation"),
    "bot_support_owner_guard": _artifact_ok_resolver("bot_support_owner_guard"),
    "new_bot_admission_guard": _artifact_ok_resolver("new_bot_admission_guard"),
    "data_source_divergence_bot": _artifact_ok_resolver("data_source_divergence"),
    "execution_queue_stress_bot": _artifact_ok_resolver("execution_queue_stress"),
    "snapshot_coverage_sentinel": _artifact_ok_resolver("snapshot_coverage"),
    "retrain_schema_compatibility_guard": _artifact_ok_resolver(
        "retrain_schema_compatibility_guard"
    ),
    "golden_replay_regression_guard": _artifact_ok_resolver(
        "golden_replay_regression_guard"
    ),
    "cohort_drift_baseline_guard": _artifact_ok_resolver("cohort_drift_baseline_guard"),
    "replay_hash_registry_guard": _artifact_ok_resolver("replay_hash_registry_guard"),
    "champion_challenger_probation_guard": _artifact_ok_resolver(
        "champion_challenger_probation_guard"
    ),
    "champion_challenger_probation_action": _artifact_ok_resolver(
        "champion_challenger_probation_action"
    ),
    "retrain_lane_scheduler": _artifact_ok_resolver("retrain_lane_scheduler"),
    "promotion_packet_builder": _promotion_packet_builder_resolver,
    "promotion_quality_gate": _artifact_ok_resolver("promotion_quality_gate"),
    "db_integrity": _db_integrity_resolver,
    "nightly_resilience_check": _nightly_resilience_resolver,
    "artifact_freshness": _artifact_freshness_resolver,
    "incomplete_run_recovered": _incomplete_run_recovered_resolver,
}


def build_dashboard(project_root: Path = PROJECT_ROOT) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    artifacts: Dict[str, Dict[str, Any]] = {}
    attention: List[str] = []
    severity = 0

    for name, cfg in _artifact_config(project_root).items():
        path, payload = _pick_latest_artifact(cfg["paths"])
        exists = bool(path and path.exists() and payload)
        ts = _payload_timestamp(payload, path) if exists else None
        age_minutes = (
            max((now - ts).total_seconds() / 60.0, 0.0) if ts is not None else None
        )
        ok_value = _infer_ok(payload) if exists else None
        status = _infer_status(payload, ok_value) if exists else "missing"
        stale = bool(
            age_minutes is not None and age_minutes > float(cfg["max_age_minutes"])
        )
        summary = _artifact_summary(name, payload) if exists else {}
        if name == "sql_link_service" and stale:
            live_pid = _live_sql_writer_pid(
                {
                    "path": str(path) if path else "",
                    "exists": exists,
                    "age_minutes": age_minutes,
                    "max_age_minutes": float(cfg["max_age_minutes"]),
                    "summary": summary,
                }
            )
            if live_pid is not None:
                stale = False
                summary = {
                    **summary,
                    "freshness_inferred_from_live_lock": True,
                    "lock_owner_pid": live_pid,
                }
        if name == "sql_ingestion" and stale:
            sql_service = artifacts.get("sql_link_service", {})
            sql_service_summary = (
                sql_service.get("summary")
                if isinstance(sql_service.get("summary"), dict)
                else {}
            )
            sql_service_fresh = bool(sql_service.get("exists")) and (
                not bool(sql_service.get("stale"))
            )
            sql_service_live = bool(sql_service_summary.get("running")) or (
                sql_service.get("ok") is True
            )
            pending_lines = int(summary.get("pending_lines", 0) or 0)
            invalid_lines = int(summary.get("invalid_lines", 0) or 0)
            if (
                sql_service_fresh
                and sql_service_live
                and pending_lines == 0
                and invalid_lines == 0
            ):
                stale = False
                summary = {
                    **summary,
                    "freshness_via_service_heartbeat": True,
                    "service_heartbeat_age_minutes": sql_service.get("age_minutes"),
                }
        artifacts[name] = {
            "path": str(path) if path else "",
            "exists": exists,
            "timestamp_utc": ts.isoformat() if ts is not None else "",
            "age_minutes": round(age_minutes, 4) if age_minutes is not None else None,
            "max_age_minutes": float(cfg["max_age_minutes"]),
            "required": bool(cfg["required"]),
            "ok": ok_value,
            "stale": stale,
            "status": status,
            "summary": summary,
        }
        if bool(cfg["required"]) and not exists:
            attention.append(f"{name}_missing")
            severity = max(severity, 3)
        elif bool(cfg["required"]) and stale:
            attention.append(f"{name}_stale")
            severity = max(severity, 2)
        elif ok_value is False and name in {
            "session_ready",
            "health_gates",
            "retrain_artifact_freshness",
            "sql_link_service",
            "master_grandmaster_evidence_v2",
            "bot_profitability_scalability_control",
            "sleeve_scalability_selector",
            "system_role_contract",
        }:
            if _artifact_waiting_on_evidence_not_degraded(name, status, summary):
                summary["evidence_debt_non_degraded"] = True
            else:
                attention.append(f"{name}_not_ok")
                severity = max(severity, 2)

    hdf5_artifact = (
        artifacts.get("hdf5_training_cache")
        if isinstance(artifacts.get("hdf5_training_cache"), dict)
        else {}
    )
    hdf5_summary = (
        hdf5_artifact.get("summary")
        if isinstance(hdf5_artifact.get("summary"), dict)
        else {}
    )
    if hdf5_artifact.get("exists") and (
        str(hdf5_summary.get("overall_status") or "") != "ready"
        or not bool(hdf5_summary.get("fresh", False))
        or not bool(hdf5_summary.get("schema_ok", False))
    ):
        attention.append("hdf5_training_cache_not_fresh")
        severity = max(severity, 1)

    health_inputs = (
        artifacts.get("health_gates", {}).get("summary", {}).get("inputs", {})
    )
    storage_backpressure_override = (
        health_inputs.get("backpressure_storage_control_override")
        if isinstance(health_inputs.get("backpressure_storage_control_override"), dict)
        else {}
    )
    queue_override_clear = bool(
        storage_backpressure_override.get("active", False)
        and not bool(storage_backpressure_override.get("overload", False))
        and not bool(storage_backpressure_override.get("line_pressure", False))
        and not bool(storage_backpressure_override.get("file_pressure", False))
        and not bool(storage_backpressure_override.get("age_pressure", False))
    )
    if queue_override_clear and isinstance(
        artifacts.get("ingestion_priority_queue"), dict
    ):
        queue_artifact = artifacts["ingestion_priority_queue"]
        queue_summary = (
            queue_artifact.get("summary")
            if isinstance(queue_artifact.get("summary"), dict)
            else {}
        )
        queue_artifact["summary"] = {
            **queue_summary,
            "raw_queue_depth": int(queue_summary.get("queue_depth", 0) or 0),
            "raw_core_pending_lines": int(
                queue_summary.get("core_pending_lines", 0) or 0
            ),
            "queue_depth": 0,
            "core_pending_lines": int(
                storage_backpressure_override.get("pending_lines", 0) or 0
            ),
            "effective_source": str(
                storage_backpressure_override.get("source")
                or "storage_backpressure_override"
            ),
            "raw_queue_suppressed_by_storage_overlay": True,
        }
        artifacts["ingestion_priority_queue"] = queue_artifact
    if artifacts.get("health_gates", {}).get("summary", {}).get("hard_gate_triggered"):
        attention.append("health_gates_hard_gate_triggered")
        severity = max(severity, 2)
    launcher_summary = artifacts.get("all_sleeves_launcher", {}).get("summary", {})
    if (
        artifacts.get("all_sleeves_launcher", {}).get("exists")
        and not artifacts.get("all_sleeves_launcher", {}).get("stale")
        and bool(launcher_summary.get("collection_fanout_ready", False))
        and not bool(launcher_summary.get("paper_execution_ready", False))
    ):
        attention.append("paper_execution_safety_guard_active")
        severity = max(severity, 1)

    safety_snapshot = _safety_snapshot_contract(artifacts)
    if not bool(safety_snapshot.get("consistent", False)):
        attention.append("safety_evidence_snapshot_skew")
        severity = max(severity, 2)
    daily_verify_payload = _load_json(
        Path(str(artifacts.get("daily_auto_verify", {}).get("path", "") or ""))
    )
    unresolved_daily_verify, resolved_daily_verify = (
        _resolved_daily_auto_verify_failures(daily_verify_payload, artifacts)
    )
    if artifacts.get("daily_auto_verify", {}).get("exists"):
        artifacts["daily_auto_verify"]["summary"][
            "resolved_failed_checks"
        ] = resolved_daily_verify
        artifacts["daily_auto_verify"]["summary"][
            "effective_failed_checks"
        ] = unresolved_daily_verify
    daily_summary = artifacts.get("daily_auto_verify", {}).get("summary", {})
    daily_operational_ok = bool(daily_summary.get("operational_ok", False))
    if (
        artifacts.get("daily_auto_verify", {}).get("ok") is False
        and unresolved_daily_verify
        and not daily_operational_ok
    ):
        if not (
            set(unresolved_daily_verify) == {"promotion_quality_gate"}
            and artifacts.get("promotion_readiness", {})
            .get("summary", {})
            .get("promote_ok")
            is False
        ):
            attention.append("daily_auto_verify_not_ok")
            severity = max(severity, 2)
    if (
        artifacts.get("promotion_readiness", {}).get("summary", {}).get("promote_ok")
        is False
    ):
        attention.append("promotion_not_ready")
        severity = max(severity, 1)
    if (
        artifacts.get("training_quality_control", {})
        .get("summary", {})
        .get("overall_status")
        == "blocked"
    ):
        attention.append("training_quality_control_blocked")
        severity = max(severity, 1)
    storage_summary = artifacts.get("ingestion_storage_control", {}).get("summary", {})
    storage_status = str(storage_summary.get("overall_status", "") or "")
    storage_severity = str(storage_summary.get("severity", "") or "")
    if storage_status == "blocked":
        attention.append("ingestion_storage_control_blocked")
        severity = max(severity, 2)
    elif storage_severity in {"high", "critical"}:
        attention.append("ingestion_storage_control_elevated")
        severity = max(severity, 1)
    governor_summary = artifacts.get("ingestion_storage_governor", {}).get(
        "summary", {}
    )
    if bool(governor_summary.get("route_drift", False)):
        attention.append("sql_primary_route_drift")
        severity = max(severity, 2)
    if str(governor_summary.get("profile", "") or "") == "critical_backpressure":
        attention.append("ingestion_storage_governor_critical")
        severity = max(severity, 1)
    drain_summary = artifacts.get("external_backlog_drain", {}).get("summary", {})
    drain_material = bool(
        drain_summary.get("recommended_now", False)
        or drain_summary.get("material_drain_recommended", False)
        or int(drain_summary.get("aged_candidate_files", 0) or 0) > 0
    )
    tiny_drain_tail_nonblocking = _external_backlog_tiny_drain_tail_nonblocking(
        storage_summary,
        drain_summary,
    )
    if bool(drain_summary.get("recommended_now", False)) and not (
        tiny_drain_tail_nonblocking
    ):
        attention.append("external_backlog_drain_recommended")
        severity = max(severity, 1)
    if (
        bool(drain_summary.get("writer_busy", False))
        and drain_material
        and not tiny_drain_tail_nonblocking
    ):
        attention.append("external_backlog_drain_writer_busy")
        severity = max(severity, 1)
    follow_through_status = str(drain_summary.get("follow_through_status", "") or "")
    follow_through_progress_state = str(
        drain_summary.get("follow_through_progress_state", "") or ""
    )
    if (
        follow_through_status == "timed_out"
        and follow_through_progress_state != "progressing"
    ):
        attention.append("external_backlog_drain_follow_through_stalled")
        severity = max(severity, 1)
    retry_summary = artifacts.get("external_backlog_retry_bot", {}).get("summary", {})
    retry_status = str(retry_summary.get("overall_status", "") or "")
    retry_has_material_followup = (
        bool(retry_summary.get("backlog_needed", False)) and drain_material
    )
    if retry_status in {"blocked", "apply_failed"} or (
        retry_status == "applied_with_followups" and retry_has_material_followup
    ):
        attention.append("external_backlog_retry_bot_followups")
        severity = max(severity, 1)
    memory_summary = artifacts.get("memory_efficiency_control", {}).get("summary", {})
    memory_status = str(memory_summary.get("overall_status", "") or "")
    if memory_status == "blocked":
        attention.append("memory_efficiency_control_blocked")
        severity = max(severity, 2)
    elif memory_status == "needs_work":
        attention.append("memory_efficiency_control_needs_work")
        severity = max(severity, 1)
    platform_summary = artifacts.get("platform_control_plane", {}).get("summary", {})
    system_role_summary = artifacts.get("system_role_contract", {}).get("summary", {})
    platform_status = str(platform_summary.get("overall_status", "") or "")
    if platform_status in {"upgrade_required", "gap_heavy"}:
        attention.append("platform_control_plane_upgrade_required")
        severity = max(severity, 1)
    queue_summary = artifacts.get("ingestion_priority_queue", {}).get("summary", {})
    if (
        int(queue_summary.get("queue_depth", 0) or 0) > 0
        and int(queue_summary.get("core_pending_lines", 0) or 0) > 50000
    ):
        attention.append("ingestion_priority_queue_core_heavy")
        severity = max(severity, 1)
    split_brain_summary = artifacts.get("storage_split_brain_reconciler", {}).get(
        "summary", {}
    )
    if int(split_brain_summary.get("unresolved_conflicts", 0) or 0) > 0:
        attention.append("storage_split_brain_needs_review")
        severity = max(severity, 2)
    resilience_summary = artifacts.get("storage_resilience_control", {}).get(
        "summary", {}
    )
    if str(resilience_summary.get("overall_status", "") or "") == "needs_work":
        attention.append("storage_resilience_control_needs_work")
        severity = max(severity, 1)
    remediation_summary = artifacts.get("daily_verify_auto_remediation_bot", {}).get(
        "summary", {}
    )
    if str(remediation_summary.get("overall_status", "") or "") == "pending":
        attention.append("daily_verify_auto_remediation_pending")
        severity = max(severity, 1)
    stale_sweeper_artifact = artifacts.get("stale_artifact_sweeper_bot", {})
    if stale_sweeper_artifact.get("ok") is False and not (
        _stale_artifact_bot_contention_nonblocking(
            stale_sweeper_artifact, action_key="staged_files"
        )
    ):
        attention.append("stale_artifact_sweeper_bot_not_ok")
        severity = max(severity, 1)
    stale_reaper_artifact = artifacts.get("stale_artifact_reaper_bot", {})
    if stale_reaper_artifact.get("ok") is False and not (
        _stale_artifact_bot_contention_nonblocking(
            stale_reaper_artifact, action_key="deleted_files"
        )
    ):
        attention.append("stale_artifact_reaper_bot_not_ok")
        severity = max(severity, 1)
    if float(health_inputs.get("blocked_rate", 0.0) or 0.0) >= 0.50:
        attention.append("blocked_rate_elevated")
        severity = max(severity, 1)
    for name in (
        "teacher_quality_guard",
        "bot_quality_autopilot",
        "infrastructure_autofix_bot",
        "live_runtime_separation_control",
        "rolling_restart_controller",
        "auth_lease_manager",
        "blackstart_recovery",
        "sleeve_isolation_guard",
        "artifact_freshness_slo",
        "runtime_snapshot_cache_control",
        "remote_alert_control",
        "coordination_state_control",
        "storage_quota_guard",
        "release_freeze_guard",
        "roster_resilience_planner",
        "chaos_drill_coordinator",
    ):
        summary = artifacts.get(name, {}).get("summary", {})
        status = str((summary or {}).get("overall_status", "") or "")
        if name == "roster_resilience_planner" and bool(
            summary.get("operational_ready", False)
        ):
            if status not in {"ready", "ok"}:
                attention.append("roster_resilience_planner_needs_work")
                severity = max(severity, 1)
            continue
        if status == "blocked":
            if (
                name == "bot_quality_autopilot"
                and _bot_quality_autopilot_deferred_for_evidence_collection(artifacts)
            ):
                attention.append("bot_quality_autopilot_evidence_pending")
                severity = max(severity, 1)
                continue
            attention.append(f"{name}_blocked")
            severity = max(severity, 2)
        elif status in {"degraded", "inactive"}:
            attention.append(f"{name}_needs_work")
            severity = max(severity, 1)

    sql_ingestion_artifact = artifacts.get("sql_ingestion", {})
    sql_service_artifact = artifacts.get("sql_link_service", {})
    sql_ingestion_summary = (
        sql_ingestion_artifact.get("summary")
        if isinstance(sql_ingestion_artifact.get("summary"), dict)
        else {}
    )
    if (
        "sql_link_service_stale" in attention
        and bool(sql_ingestion_artifact.get("exists"))
        and not bool(sql_ingestion_artifact.get("stale"))
        and int(sql_ingestion_summary.get("pending_lines", 0) or 0) == 0
        and int(sql_ingestion_summary.get("invalid_lines", 0) or 0) == 0
    ):
        attention = [item for item in attention if item != "sql_link_service_stale"]
        sql_service_summary = (
            sql_service_artifact.get("summary")
            if isinstance(sql_service_artifact.get("summary"), dict)
            else {}
        )
        sql_service_artifact["summary"] = {
            **sql_service_summary,
            "freshness_inferred_from_sql_ingestion": True,
        }
        artifacts["sql_link_service"] = sql_service_artifact

    sql_plane_freshness = _sql_plane_freshness_from_storage_control(artifacts)
    if sql_plane_freshness:
        if "sql_link_service_stale" in attention:
            attention = [item for item in attention if item != "sql_link_service_stale"]
            sql_service_artifact = artifacts.get("sql_link_service", {})
            sql_service_summary = (
                sql_service_artifact.get("summary")
                if isinstance(sql_service_artifact.get("summary"), dict)
                else {}
            )
            sql_service_artifact["stale"] = False
            sql_service_artifact["summary"] = {
                **sql_service_summary,
                "freshness_inferred_from_ingestion_storage_control": True,
                "freshness_inference": sql_plane_freshness,
            }
            artifacts["sql_link_service"] = sql_service_artifact
        if "sql_ingestion_stale" in attention:
            attention = [item for item in attention if item != "sql_ingestion_stale"]
            sql_ingestion_artifact = artifacts.get("sql_ingestion", {})
            sql_ingestion_summary = (
                sql_ingestion_artifact.get("summary")
                if isinstance(sql_ingestion_artifact.get("summary"), dict)
                else {}
            )
            sql_ingestion_artifact["stale"] = False
            sql_ingestion_artifact["summary"] = {
                **sql_ingestion_summary,
                "freshness_inferred_from_ingestion_storage_control": True,
                "freshness_inference": sql_plane_freshness,
            }
            artifacts["sql_ingestion"] = sql_ingestion_artifact

    if "session_ready_stale" in attention:
        latest_shadow_ts = _latest_shadow_loop_timestamp(project_root)
        if latest_shadow_ts is not None:
            shadow_age_minutes = max(
                (now - latest_shadow_ts).total_seconds() / 60.0, 0.0
            )
            if shadow_age_minutes <= float(
                artifacts.get("session_ready", {}).get("max_age_minutes", 15.0) or 15.0
            ):
                attention = [
                    item for item in attention if item != "session_ready_stale"
                ]
                session_ready_artifact = artifacts.get("session_ready", {})
                session_ready_summary = (
                    session_ready_artifact.get("summary")
                    if isinstance(session_ready_artifact.get("summary"), dict)
                    else {}
                )
                session_ready_artifact["stale"] = False
                session_ready_artifact["summary"] = {
                    **session_ready_summary,
                    "freshness_inferred_from_shadow_loop": True,
                    "latest_shadow_activity_age_minutes": round(shadow_age_minutes, 4),
                }
                artifacts["session_ready"] = session_ready_artifact

    source_summary = artifacts.get("source_verification", {}).get("summary", {})
    if artifacts.get("source_verification", {}).get("exists") and not bool(
        artifacts.get("source_verification", {}).get("stale")
    ):
        source_decision_critical_ready = bool(
            source_summary.get("decision_critical_sources_ready", False)
        )
        source_decision_critical_blockers = (
            source_summary.get("decision_critical_blockers")
            if isinstance(source_summary.get("decision_critical_blockers"), list)
            else []
        )
        source_context_debt = _ordered_unique(
            [
                *[
                    str(item)
                    for item in source_summary.get("decision_context_debt", [])
                    if str(item).strip()
                ],
                *[
                    str(item)
                    for item in source_summary.get("optional_enrichment_debt", [])
                    if str(item).strip()
                ],
            ]
        )
        if (not source_decision_critical_ready) and source_decision_critical_blockers:
            attention.append("source_verification_decision_critical_blocked")
            severity = max(severity, 2)
        elif source_context_debt:
            attention.append("source_verification_context_debt")
            severity = max(severity, 1)

    forensic_attention = list(attention)
    soak_management_context = _dashboard_soak_context(project_root)
    attention, managed_controls = _split_green_soak_managed_attention(
        attention,
        artifacts,
        soak_management_context,
    )
    raw_attention = list(attention)
    managed_attention = [
        row["attention"]
        for row in managed_controls
        if isinstance(row.get("attention"), str)
    ]
    attention_tiers = _attention_tiers(attention)
    remediation_actions = _remediation_actions(attention)
    severity = _severity_from_attention(attention)
    degradation_containment = _degradation_containment_contract(
        attention,
        attention_tiers,
        artifacts,
        managed_controls,
    )
    ops_smoothing = _ops_smoothing_contract(
        artifacts,
        attention,
        attention_tiers,
        managed_controls,
    )

    status_map = {
        0: "ok",
        1: "warn",
        2: "degraded",
        3: "critical",
    }
    health_summary = artifacts.get("health_gates", {}).get("summary", {})
    collection_summary = artifacts.get("data_collection_observation_rollup", {}).get(
        "summary", {}
    )
    runtime_summary = artifacts.get("runtime_access_mode", {}).get("summary", {})
    apple_summary = artifacts.get("apple_silicon_profile", {}).get("summary", {})
    memory_summary = artifacts.get("memory_efficiency_control", {}).get("summary", {})
    killswitch_summary = artifacts.get("global_killswitch", {}).get("summary", {})
    training_summary = artifacts.get("training_report", {}).get("summary", {})
    training_quality_summary = artifacts.get("training_quality_control", {}).get(
        "summary", {}
    )
    hdf5_training_summary = artifacts.get("hdf5_training_cache", {}).get("summary", {})
    storage_summary = artifacts.get("ingestion_storage_control", {}).get("summary", {})
    governor_summary = artifacts.get("ingestion_storage_governor", {}).get(
        "summary", {}
    )
    drain_summary = artifacts.get("external_backlog_drain", {}).get("summary", {})
    retry_summary = artifacts.get("external_backlog_retry_bot", {}).get("summary", {})
    platform_summary = artifacts.get("platform_control_plane", {}).get("summary", {})
    queue_summary = artifacts.get("ingestion_priority_queue", {}).get("summary", {})
    resilience_summary = artifacts.get("storage_resilience_control", {}).get(
        "summary", {}
    )
    split_brain_summary = artifacts.get("storage_split_brain_reconciler", {}).get(
        "summary", {}
    )
    remediation_summary = artifacts.get("daily_verify_auto_remediation_bot", {}).get(
        "summary", {}
    )
    cockpit_summary = artifacts.get("operator_cockpit", {}).get("summary", {})
    long_runtime_summary = {
        "live_runtime_separation_control": artifacts.get(
            "live_runtime_separation_control", {}
        ).get("summary", {}),
        "rolling_restart_controller": artifacts.get(
            "rolling_restart_controller", {}
        ).get("summary", {}),
        "auth_lease_manager": artifacts.get("auth_lease_manager", {}).get(
            "summary", {}
        ),
        "blackstart_recovery": artifacts.get("blackstart_recovery", {}).get(
            "summary", {}
        ),
        "sleeve_isolation_guard": artifacts.get("sleeve_isolation_guard", {}).get(
            "summary", {}
        ),
        "artifact_freshness_slo": artifacts.get("artifact_freshness_slo", {}).get(
            "summary", {}
        ),
        "bot_profitability_scalability_control": artifacts.get(
            "bot_profitability_scalability_control", {}
        ).get("summary", {}),
        "sleeve_scalability_selector": artifacts.get(
            "sleeve_scalability_selector", {}
        ).get("summary", {}),
        "master_grandmaster_evidence_v2": artifacts.get(
            "master_grandmaster_evidence_v2", {}
        ).get("summary", {}),
        "runtime_snapshot_cache_control": artifacts.get(
            "runtime_snapshot_cache_control", {}
        ).get("summary", {}),
        "remote_alert_control": artifacts.get("remote_alert_control", {}).get(
            "summary", {}
        ),
        "coordination_state_control": artifacts.get(
            "coordination_state_control", {}
        ).get("summary", {}),
        "storage_quota_guard": artifacts.get("storage_quota_guard", {}).get(
            "summary", {}
        ),
        "release_freeze_guard": artifacts.get("release_freeze_guard", {}).get(
            "summary", {}
        ),
        "roster_resilience_planner": artifacts.get("roster_resilience_planner", {}).get(
            "summary", {}
        ),
        "chaos_drill_coordinator": artifacts.get("chaos_drill_coordinator", {}).get(
            "summary", {}
        ),
    }
    runtime_contract = _artifact_contract(artifacts, "runtime_access_mode")
    apple_contract = _artifact_contract(artifacts, "apple_silicon_profile")
    memory_contract = _artifact_contract(artifacts, "memory_efficiency_control")
    training_contract = _artifact_contract(artifacts, "training_report")
    storage_contract = _artifact_contract(artifacts, "ingestion_storage_control")
    resilience_contract = _artifact_contract(artifacts, "storage_resilience_control")
    platform_contract = _artifact_contract(artifacts, "platform_control_plane")
    overall_status = status_map.get(severity, "unknown")
    overall_ok = severity == 0
    payload = {
        "timestamp_utc": now.isoformat(),
        "overall_status": overall_status,
        "ok": overall_ok,
        "overall": {
            "status": overall_status,
            "ok": overall_ok,
            "attention": attention,
            "raw_attention": raw_attention,
            "forensic_attention": forensic_attention,
            "attention_tiers": attention_tiers,
            "remediation_actions": remediation_actions,
            "managed_attention": managed_attention,
            "managed_controls": managed_controls,
            "soak_management_context": soak_management_context,
            "degradation_containment": degradation_containment,
            "ops_smoothing": ops_smoothing,
        },
        "data_quality_score": float(
            health_summary.get("data_quality_score", 0.0) or 0.0
        ),
        "data_quality_dimensions": {
            "decision_freshness_quality_score": float(
                health_summary.get("data_quality_score", 0.0) or 0.0
            ),
            "collector_coverage_quality_score": float(
                collection_summary.get("collection_coverage_score", 0.0) or 0.0
            ),
            "collector_observation_quality_score": float(
                collection_summary.get("data_quality_score", 0.0) or 0.0
            ),
            "collector_count": int(collection_summary.get("collector_count", 0) or 0),
            "total_observations": int(
                collection_summary.get("total_observations", 0) or 0
            ),
            "policy": "decision freshness and historical collector coverage are separate metrics and may not overwrite each other",
        },
        "safety_evidence_snapshot": safety_snapshot,
        "degradation_containment": degradation_containment,
        "ops_smoothing": ops_smoothing,
        "execution_runtime": {
            "collection_fanout_ready": bool(
                launcher_summary.get("collection_fanout_ready", False)
            ),
            "paper_execution_ready": bool(
                launcher_summary.get("paper_execution_ready", False)
            ),
            "launcher_status": str(launcher_summary.get("overall_status") or "unknown"),
            "launcher_readiness_status": str(
                launcher_summary.get("readiness_status") or "unknown"
            ),
            "execution_attention": (
                launcher_summary.get("execution_attention")
                if isinstance(launcher_summary.get("execution_attention"), list)
                else []
            ),
            "broker_auth_epoch": (
                launcher_summary.get("broker_auth_epoch")
                if isinstance(launcher_summary.get("broker_auth_epoch"), dict)
                else {}
            ),
        },
        "health_gate_triggered": bool(health_summary.get("hard_gate_triggered", False)),
        "global_kill_triggered": bool(killswitch_summary.get("halt", False)),
        "gates": {
            "health_gate_triggered": bool(
                health_summary.get("hard_gate_triggered", False)
            ),
            "global_kill_triggered": bool(killswitch_summary.get("halt", False)),
            "promotion_not_ready": "promotion_not_ready" in attention,
            "daily_auto_verify_not_ok": "daily_auto_verify_not_ok" in attention,
            "system_role_contract_not_ok": "system_role_contract_not_ok" in attention,
        },
        "system_responsibility": {
            **_artifact_contract(artifacts, "system_role_contract"),
            "grade": str(system_role_summary.get("grade", "") or "unknown"),
            "operating_mode": str(
                system_role_summary.get("operating_mode", "") or "unknown"
            ),
            "role_count": int(system_role_summary.get("role_count", 0) or 0),
            "component_count": int(system_role_summary.get("component_count", 0) or 0),
            "state_domain_count": int(
                system_role_summary.get("state_domain_count", 0) or 0
            ),
            "exclusive_action_count": int(
                system_role_summary.get("exclusive_action_count", 0) or 0
            ),
            "operating_plane_count": int(
                system_role_summary.get("operating_plane_count", 0) or 0
            ),
            "classified_action_count": int(
                system_role_summary.get("classified_action_count", 0) or 0
            ),
            "action_lease_count": int(
                system_role_summary.get("action_lease_count", 0) or 0
            ),
            "escalation_route_count": int(
                system_role_summary.get("escalation_route_count", 0) or 0
            ),
            "registry_role_coverage_ratio": float(
                system_role_summary.get("registry_role_coverage_ratio", 0.0) or 0.0
            ),
            "authority_conflict_count": int(
                system_role_summary.get("authority_conflict_count", 0) or 0
            ),
            "blockers": (
                system_role_summary.get("blockers")
                if isinstance(system_role_summary.get("blockers"), list)
                else []
            ),
        },
        "runtime": {
            **runtime_contract,
            "mode": str(runtime_summary.get("mode", "") or "unknown"),
            "ml_backend": str(runtime_summary.get("ml_backend", "") or "unknown"),
            "portable_enabled": bool(runtime_summary.get("portable_enabled", False)),
            "backend_contract": (
                runtime_summary.get("backend_contract")
                if isinstance(runtime_summary.get("backend_contract"), dict)
                else {}
            ),
        },
        "apple_silicon": {
            **apple_contract,
            "applied_tier": str(apple_summary.get("applied_tier", "") or "unknown"),
            "detected_tier": str(apple_summary.get("detected_tier", "") or "unknown"),
            "memory_gb": float(apple_summary.get("memory_gb", 0.0) or 0.0),
            "chip": str(apple_summary.get("chip", "") or "unknown"),
        },
        "memory": {
            **memory_contract,
            "overall_status": str(
                memory_summary.get("overall_status", "") or memory_contract["status"]
            ),
            "recommended_profile": str(
                memory_summary.get("recommended_profile", "") or "unknown"
            ),
            "memory_pressure_state": str(
                memory_summary.get("memory_pressure_state", "") or "unknown"
            ),
            "memory_pressure_kind": str(
                memory_summary.get("memory_pressure_kind", "") or "unknown"
            ),
            "swap_used_gb": float(memory_summary.get("swap_used_gb", 0.0) or 0.0),
        },
        "training": {
            **training_contract,
            "overall_status": str(
                training_summary.get("overall_status", "")
                or training_contract["status"]
            ),
            "blocking_reasons": (
                training_summary.get("blocking_reasons")
                if isinstance(training_summary.get("blocking_reasons"), list)
                else []
            ),
            "quality_score": float(
                training_quality_summary.get("training_quality_score", 0.0) or 0.0
            ),
            "top_priorities": (
                training_quality_summary.get("top_priorities")
                if isinstance(training_quality_summary.get("top_priorities"), list)
                else []
            ),
            "active_supportability_score": float(
                training_quality_summary.get("active_supportability_score", 0.0) or 0.0
            ),
            "hdf5_cache": (
                hdf5_training_summary if isinstance(hdf5_training_summary, dict) else {}
            ),
        },
        "storage": {
            **storage_contract,
            "overall_status": str(
                storage_summary.get("overall_status", "") or storage_contract["status"]
            ),
            "severity": str(storage_summary.get("severity", "") or "unknown"),
            "pressure_index": float(storage_summary.get("pressure_index", 0.0) or 0.0),
            "recommended_operating_mode": str(
                storage_summary.get("recommended_operating_mode", "") or "unknown"
            ),
            "pressure_profile": str(governor_summary.get("profile", "") or "unknown"),
            "sql_primary_route_drift": bool(governor_summary.get("route_drift", False)),
            "core_pending_lines": int(
                storage_summary.get("core_pending_lines", 0) or 0
            ),
            "total_pending_lines": int(
                storage_summary.get("total_pending_lines", 0) or 0
            ),
            "deferred_files_budget": int(
                governor_summary.get("deferred_files_budget", 0) or 0
            ),
            "cold_files_budget": int(governor_summary.get("cold_files_budget", 0) or 0),
            "backlog_drain_tiny_tail_nonblocking": bool(tiny_drain_tail_nonblocking),
            "backlog_drain_recommended_raw": bool(
                drain_summary.get("recommended_now", False)
            ),
            "backlog_drain_writer_busy_raw": bool(
                drain_summary.get("writer_busy", False)
            ),
            "backlog_drain_recommended": bool(
                drain_summary.get("recommended_now", False)
                and not tiny_drain_tail_nonblocking
            ),
            "backlog_drain_writer_busy": bool(
                drain_summary.get("writer_busy", False)
                and not tiny_drain_tail_nonblocking
            ),
            "backlog_drain_aged_candidate_files": int(
                drain_summary.get("aged_candidate_files", 0) or 0
            ),
            "backlog_drain_deferred_budget": int(
                drain_summary.get("deferred_files_budget", 0) or 0
            ),
            "backlog_drain_cold_budget": int(
                drain_summary.get("cold_files_budget", 0) or 0
            ),
            "backlog_drain_follow_through_status": str(
                drain_summary.get("follow_through_status", "") or ""
            ),
            "backlog_drain_follow_through_progress_state": str(
                drain_summary.get("follow_through_progress_state", "") or ""
            ),
            "backlog_drain_follow_through_progress_observed": bool(
                drain_summary.get("follow_through_progress_observed", False)
            ),
            "backlog_retry_bot_status": retry_status,
            "backlog_retry_bot_actionable": bool(
                retry_summary.get("actionable", False)
            ),
            "backlog_quarantine_status": str(
                storage_summary.get("backlog_quarantine_status", "") or ""
            ),
            "backlog_quarantine_candidate_files": int(
                storage_summary.get("backlog_quarantine_candidate_files", 0) or 0
            ),
            "backlog_quarantine_moved_files": int(
                storage_summary.get("backlog_quarantine_moved_files", 0) or 0
            ),
            "estimated_core_drain_minutes": storage_summary.get(
                "estimated_core_drain_minutes"
            ),
            "estimated_total_drain_minutes": storage_summary.get(
                "estimated_total_drain_minutes"
            ),
            "retention_debt_gb": float(
                storage_summary.get("retention_debt_gb", 0.0) or 0.0
            ),
            "queue_watermarks_status": str(
                storage_summary.get("queue_watermarks_overall_status", "") or "unknown"
            ),
            "writer_shedding_level": str(
                storage_summary.get("writer_shedding_level", "")
                or str(governor_summary.get("writer_shedding_level", "") or "unknown")
            ),
            "writer_shedding_active": bool(
                storage_summary.get("writer_shedding_active", False)
                or governor_summary.get("writer_shedding_active", False)
            ),
            "external_route_verification_state": str(
                storage_summary.get("external_route_verification_state", "")
                or "unknown"
            ),
        },
        "ingestion_queue": {
            "queue_depth": int(queue_summary.get("queue_depth", 0) or 0),
            "items_synced": int(queue_summary.get("items_synced", 0) or 0),
            "core_pending_lines": int(queue_summary.get("core_pending_lines", 0) or 0),
            "event_count": int(queue_summary.get("event_count", 0) or 0),
        },
        "storage_resilience": {
            **resilience_contract,
            "overall_status": str(
                resilience_summary.get("overall_status", "")
                or resilience_contract["status"]
            ),
            "resilience_score": int(resilience_summary.get("resilience_score", 0) or 0),
            "restore_drill_fresh": bool(
                resilience_summary.get("restore_drill_fresh", False)
            ),
            "unresolved_split_brain_conflicts": int(
                resilience_summary.get("unresolved_split_brain_conflicts", 0) or 0
            ),
        },
        "split_brain": {
            "conflict_files": int(split_brain_summary.get("conflict_files", 0) or 0),
            "unresolved_conflicts": int(
                split_brain_summary.get("unresolved_conflicts", 0) or 0
            ),
            "force_failback_eligible": bool(
                split_brain_summary.get("force_failback_eligible", False)
            ),
        },
        "automation": {
            "daily_verify_auto_remediation_status": str(
                remediation_summary.get("overall_status", "") or ""
            ),
            "resolved_checks": (
                remediation_summary.get("resolved_checks")
                if isinstance(remediation_summary.get("resolved_checks"), list)
                else []
            ),
            "operator_cockpit_status": str(
                cockpit_summary.get("overall_status", "") or ""
            ),
        },
        "long_runtime": {
            "live_runtime_separation_status": str(
                (long_runtime_summary.get("live_runtime_separation_control") or {}).get(
                    "overall_status", ""
                )
                or ""
            ),
            "rolling_restart_due": bool(
                (long_runtime_summary.get("rolling_restart_controller") or {}).get(
                    "restart_due", False
                )
            ),
            "auth_lease_state": str(
                (long_runtime_summary.get("auth_lease_manager") or {}).get(
                    "lease_state", ""
                )
                or ""
            ),
            "isolated_lanes": int(
                (long_runtime_summary.get("sleeve_isolation_guard") or {}).get(
                    "isolated_lane_count", 0
                )
                or 0
            ),
            "artifact_sla_required_breaches": int(
                (long_runtime_summary.get("artifact_freshness_slo") or {}).get(
                    "stale_required", 0
                )
                or 0
            ),
            "snapshot_cache_ready": bool(
                (long_runtime_summary.get("runtime_snapshot_cache_control") or {}).get(
                    "snapshot_ready", False
                )
            ),
            "remote_alert_unacked_critical": int(
                (long_runtime_summary.get("remote_alert_control") or {}).get(
                    "unacked_critical", 0
                )
                or 0
            ),
            "storage_quota_hard_breaches": int(
                (long_runtime_summary.get("storage_quota_guard") or {}).get(
                    "hard_breaches", 0
                )
                or 0
            ),
            "release_freeze_active": bool(
                (long_runtime_summary.get("release_freeze_guard") or {}).get(
                    "active", False
                )
            ),
            "bench_depth": int(
                (long_runtime_summary.get("roster_resilience_planner") or {}).get(
                    "bench_depth", 0
                )
                or 0
            ),
            "overdue_chaos_drills": int(
                (long_runtime_summary.get("chaos_drill_coordinator") or {}).get(
                    "overdue_drills", 0
                )
                or 0
            ),
            "artifact_contracts": {
                name: _artifact_contract(artifacts, name)
                for name in long_runtime_summary
            },
        },
        "platform": {
            **platform_contract,
            "overall_status": str(
                platform_summary.get("overall_status", "")
                or platform_contract["status"]
            ),
            "overall_score": float(platform_summary.get("overall_score", 0.0) or 0.0),
            "top_priorities": (
                platform_summary.get("top_priorities")
                if isinstance(platform_summary.get("top_priorities"), list)
                else []
            ),
            "weakest_domains": (
                platform_summary.get("weakest_domains")
                if isinstance(platform_summary.get("weakest_domains"), list)
                else []
            ),
        },
        "artifacts": artifacts,
        "registry": _registry_summary(project_root),
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a normalized runtime monitoring dashboard snapshot."
    )
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_dashboard(PROJECT_ROOT)
    out_path = Path(args.out_file).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
    )

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            f"runtime_gate_dashboard status={payload['overall']['status']} "
            f"attention={','.join(payload['overall']['attention']) if payload['overall']['attention'] else 'none'}"
        )
    return 0 if payload["overall"]["status"] in {"ok", "warn"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
