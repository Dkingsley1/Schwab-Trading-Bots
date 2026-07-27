import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import ingestion_storage_control as src


def test_collector_intake_audit_accepts_stricter_a_plus_plus_target(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / ".env.storage_pressure_override").write_text(
        "\n".join(
            [
                "BOT_COLLECTION_DUTY_CYCLE_ENABLED=1",
                "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO=0.16",
                "BOT_COLLECTION_DUTY_CYCLE_A_PLUS_PLUS_TARGET=1",
                "TRAINING_RUNTIME_PAUSED_FOR_BACKLOG=0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    contract = {
        "control_env_recommendations": {
            "BOT_COLLECTION_DUTY_CYCLE_ENABLED": "1",
            "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO": "0.35",
            "BOT_COLLECTION_DUTY_CYCLE_A_PLUS_PLUS_TARGET": "0",
            "TRAINING_RUNTIME_PAUSED_FOR_BACKLOG": "0",
        }
    }

    audit = src._collector_intake_enforcement_audit(tmp_path, contract)

    assert audit["status"] == "enforced"
    assert audit["mismatch_count"] == 0
    assert audit["mismatches"] == []


def test_continuous_ingestion_soak_contract_blocks_on_forecast_and_route() -> None:
    payload = src._continuous_ingestion_soak_contract(
        horizon_days=28.0,
        overall_status="ready",
        severity="stable",
        steady_state={"target_status": {"steady_state_ready": True}},
        recovery_scorecard={"score": 96.0},
        backlog_relief_contract={"active": False, "overall_grade": "A+"},
        collector_intake_audit={"status": "enforced"},
        storage_efficiency_contract={"overall_status": "ready", "grade": "A+"},
        storage_growth_forecast={"status": "near_pressure", "days_until_pressure_free": 12.0},
        storage_retention_unison={"continuous_run_contract": {"status": "blocked", "ready": False, "available_margin_gb": -4.0}},
        route_verified=False,
        resilience_status="ready",
        unresolved_split_brain_conflicts=0,
        retention_debt_gb=0.0,
        drain_minutes_total=4.0,
        data_integrity={
            "sql_invalid_lines": 0,
            "sql_overlay_invalid_lines": 0,
            "sql_overlay_ops_write_failures": 0,
            "sql_overlay_oversize_payloads": 0,
        },
    )

    assert payload["status"] == "blocked"
    assert payload["ready"] is False
    assert "external_route_not_verified" in payload["blockers"]
    assert "storage_growth_forecast_not_28_day_ready" in payload["blockers"]
    assert payload["control_env"]["TRAINING_RUNTIME_PAUSED_FOR_BACKLOG"] == "1"


def test_continuous_ingestion_soak_contract_ready_when_all_gates_clear() -> None:
    payload = src._continuous_ingestion_soak_contract(
        horizon_days=28.0,
        overall_status="ready",
        severity="stable",
        steady_state={"target_status": {"steady_state_ready": True}},
        recovery_scorecard={"score": 100.0},
        backlog_relief_contract={"active": False, "overall_grade": "A+"},
        collector_intake_audit={"status": "enforced"},
        storage_efficiency_contract={"overall_status": "ready", "grade": "A+"},
        storage_growth_forecast={"status": "forecast_ready", "days_until_pressure_free": 90.0},
        storage_retention_unison={"continuous_run_contract": {"status": "ready", "ready": True, "available_margin_gb": 80.0}},
        route_verified=True,
        resilience_status="ready",
        unresolved_split_brain_conflicts=0,
        retention_debt_gb=0.0,
        drain_minutes_total=3.0,
        data_integrity={
            "sql_invalid_lines": 0,
            "sql_overlay_invalid_lines": 0,
            "sql_overlay_ops_write_failures": 0,
            "sql_overlay_oversize_payloads": 0,
        },
    )

    assert payload["status"] == "ready"
    assert payload["ready"] is True
    assert payload["blockers"] == []
    assert payload["control_env"]["TRAINING_RUNTIME_PAUSED_FOR_BACKLOG"] == "0"


def test_continuous_ingestion_soak_contract_allows_unknown_drain_after_steady_state_guard() -> None:
    payload = src._continuous_ingestion_soak_contract(
        horizon_days=28.0,
        overall_status="ready",
        severity="stable",
        steady_state={
            "target_status": {
                "steady_state_ready": True,
                "estimated_total_drain_minutes_ok": True,
            }
        },
        recovery_scorecard={"score": 96.0},
        backlog_relief_contract={"active": False, "overall_grade": "A+"},
        collector_intake_audit={"status": "enforced"},
        storage_efficiency_contract={"overall_status": "ready", "grade": "A+"},
        storage_growth_forecast={"status": "forecast_ready", "days_until_pressure_free": 90.0},
        storage_retention_unison={"continuous_run_contract": {"status": "ready", "ready": True, "available_margin_gb": 80.0}},
        route_verified=True,
        resilience_status="ready",
        unresolved_split_brain_conflicts=0,
        retention_debt_gb=0.0,
        drain_minutes_total=None,
        data_integrity={
            "sql_invalid_lines": 0,
            "sql_overlay_invalid_lines": 0,
            "sql_overlay_ops_write_failures": 0,
            "sql_overlay_oversize_payloads": 0,
        },
    )

    assert payload["status"] == "ready"
    assert "drain_time_unknown" not in payload["warnings"]
    assert "bounded_queue_drain_time_unknown_allowed" in payload["non_blocking_conditions"]


def test_continuous_ingestion_soak_contract_allows_bounded_drain_time_only_watch() -> None:
    payload = src._continuous_ingestion_soak_contract(
        horizon_days=30.0,
        overall_status="ready",
        severity="stable",
        steady_state={
            "target_status": {
                "steady_state_ready": False,
                "target_breaches": ["estimated_total_drain_minutes"],
                "estimated_total_drain_minutes_ok": False,
            },
            "ratios": {
                "pressure_index": 0.7,
                "core_pending_lines": 0.52,
                "estimated_total_drain_minutes": 114.7,
            },
        },
        recovery_scorecard={"score": 86.0},
        backlog_relief_contract={
            "active": True,
            "overall_grade": "B",
            "active_issue_ids": ["raw_live_expansion_headroom"],
            "raw_live_expansion_headroom": {
                "hard_block": False,
                "raw_live": {
                    "core_pending_lines": 2598,
                    "total_pending_lines": 3745,
                    "oldest_pending_age_seconds": 3.4,
                },
                "targets": {
                    "absolute_total_threshold_lines": 15000,
                    "absolute_age_threshold_seconds": 240,
                },
            },
        },
        collector_intake_audit={"status": "enforced"},
        storage_efficiency_contract={"overall_status": "ready", "grade": "A+"},
        storage_growth_forecast={"status": "forecast_ready", "days_until_pressure_free": 60.0},
        storage_retention_unison={"continuous_run_contract": {"status": "watch", "ready": True, "available_margin_gb": 180.0}},
        route_verified=True,
        resilience_status="ready",
        unresolved_split_brain_conflicts=0,
        retention_debt_gb=0.0,
        drain_minutes_total=207.0,
        data_integrity={
            "sql_invalid_lines": 0,
            "sql_overlay_invalid_lines": 0,
            "sql_overlay_ops_write_failures": 0,
            "sql_overlay_oversize_payloads": 0,
        },
    )

    assert payload["status"] == "watch"
    assert payload["ready"] is False
    assert payload["soak_ready"] is True
    assert "steady_state_targets_not_clear" not in payload["blockers"]
    assert "drain_time_above_target" not in payload["blockers"]
    assert "steady_state_drain_time_in_bounded_soak_watch" in payload["warnings"]
    assert "bounded_drain_time_backlog_allowed_for_soak" in payload["non_blocking_conditions"]


def test_continuous_ingestion_soak_contract_marks_a_plus_raw_live_drain_time_clear_ready() -> None:
    payload = src._continuous_ingestion_soak_contract(
        horizon_days=30.0,
        overall_status="ready",
        severity="stable",
        steady_state={
            "target_status": {
                "steady_state_ready": False,
                "target_breaches": ["estimated_total_drain_minutes"],
                "estimated_total_drain_minutes_ok": False,
            },
            "ratios": {
                "pressure_index": 0.604,
                "core_pending_lines": 0.452,
                "estimated_total_drain_minutes": 506.827,
            },
        },
        recovery_scorecard={"score": 75.0},
        backlog_relief_contract={
            "active": False,
            "overall_grade": "A+",
            "active_issue_ids": [],
            "raw_live_expansion_headroom": {
                "active": False,
                "grade": "A+",
                "expansion_ready": True,
                "hard_block": False,
                "raw_live": {
                    "core_pending_lines": 2261,
                    "total_pending_lines": 4098,
                    "oldest_pending_age_seconds": 0.0,
                },
                "targets": {
                    "absolute_core_target_lines": 5000,
                    "absolute_total_threshold_lines": 15000,
                    "absolute_age_threshold_seconds": 240.0,
                },
            },
        },
        collector_intake_audit={"status": "enforced"},
        storage_efficiency_contract={"overall_status": "ready", "grade": "A+"},
        storage_growth_forecast={"status": "forecast_ready", "days_until_pressure_free": 676.24},
        storage_retention_unison={"continuous_run_contract": {"status": "ready", "ready": True, "available_margin_gb": 108.808}},
        route_verified=True,
        resilience_status="ready",
        unresolved_split_brain_conflicts=0,
        retention_debt_gb=0.0,
        drain_minutes_total=7602.404,
        data_integrity={
            "sql_invalid_lines": 0,
            "sql_overlay_invalid_lines": 0,
            "sql_overlay_ops_write_failures": 0,
            "sql_overlay_oversize_payloads": 0,
        },
    )

    assert payload["status"] == "ready"
    assert payload["ready"] is True
    assert payload["soak_ready"] is True
    assert payload["grade"] == "A+"
    assert payload["blockers"] == []
    assert payload["warnings"] == []
    assert "a_plus_raw_live_drain_time_estimate_clear_for_soak" in payload["non_blocking_conditions"]
    assert "a_plus_total_drain_time_estimate_above_target_allowed_for_soak" in payload["non_blocking_conditions"]
    assert payload["inputs"]["a_plus_drain_time_only_soak_clear"] is True
    assert payload["inputs"]["a_plus_drain_time_horizon_ok"] is True
    assert payload["control_env"]["BOT_CONTINUOUS_COLLECTION_READY"] == "1"
    assert payload["control_env"]["TRAINING_RUNTIME_PAUSED_FOR_BACKLOG"] == "0"


def test_continuous_ingestion_soak_contract_allows_pressure_only_reserve_headroom_with_training_pause_mismatch() -> None:
    payload = src._continuous_ingestion_soak_contract(
        horizon_days=30.0,
        overall_status="ready",
        severity="elevated",
        steady_state={
            "target_status": {
                "steady_state_ready": False,
                "target_breaches": ["pressure_index"],
            },
            "ratios": {
                "pressure_index": 3.096,
                "core_pending_lines": 0.767,
                "estimated_total_drain_minutes": 1.0,
            },
        },
        recovery_scorecard={"score": 96.0},
        backlog_relief_contract={
            "active": True,
            "overall_grade": "B",
            "active_issue_ids": ["raw_live_expansion_headroom"],
            "raw_live_expansion_headroom": {
                "active": True,
                "grade": "A",
                "expansion_tier": "limited_expansion_only",
                "hard_block": False,
                "raw_live": {
                    "core_pending_lines": 3835,
                    "total_pending_lines": 4375,
                    "oldest_pending_age_seconds": 185.726,
                },
                "targets": {
                    "absolute_total_threshold_lines": 15000,
                    "absolute_age_threshold_seconds": 240,
                },
            },
        },
        collector_intake_audit={
            "status": "partial",
            "mismatches": [
                {
                    "key": "TRAINING_RUNTIME_PAUSED_FOR_BACKLOG",
                    "expected": "1",
                    "observed": "0",
                }
            ],
        },
        storage_efficiency_contract={"overall_status": "ready", "grade": "A+"},
        storage_growth_forecast={"status": "target_floor_breach", "days_until_pressure_free": 0.57},
        storage_retention_unison={
            "continuous_run_contract": {"status": "watch", "ready": True, "available_margin_gb": 11.88}
        },
        route_verified=True,
        resilience_status="ready",
        unresolved_split_brain_conflicts=0,
        retention_debt_gb=0.0,
        drain_minutes_total=15.0,
        data_integrity={
            "sql_invalid_lines": 0,
            "sql_overlay_invalid_lines": 0,
            "sql_overlay_ops_write_failures": 0,
            "sql_overlay_oversize_payloads": 0,
        },
    )

    assert payload["status"] == "watch"
    assert payload["ready"] is False
    assert payload["soak_ready"] is True
    assert payload["blockers"] == []
    assert "steady_state_pressure_index_in_bounded_soak_watch" in payload["warnings"]
    assert "training_pause_mismatch_allowed_for_pressure_index_soak" in payload["non_blocking_conditions"]
    assert payload["inputs"]["collector_intake_soak_safe"] is True
    assert payload["inputs"]["collector_partial_reserve_pressure_soak_safe"] is True


def test_continuous_ingestion_soak_contract_allows_pressure_only_watch_when_backlog_relief_clear() -> None:
    payload = src._continuous_ingestion_soak_contract(
        horizon_days=30.0,
        overall_status="ready",
        severity="stable",
        steady_state={
            "target_status": {
                "steady_state_ready": False,
                "target_breaches": ["pressure_index"],
            },
            "ratios": {
                "pressure_index": 1.788,
                "core_pending_lines": 0.27,
                "estimated_total_drain_minutes": 0.04,
            },
        },
        recovery_scorecard={"score": 82.0},
        backlog_relief_contract={
            "active": False,
            "overall_grade": "A+",
            "active_issue_ids": [],
            "raw_live_expansion_headroom": {
                "active": False,
                "grade": "A+",
                "hard_block": False,
                "raw_live": {
                    "core_pending_lines": 1348,
                    "total_pending_lines": 1348,
                    "oldest_pending_age_seconds": 107.396,
                },
                "targets": {
                    "absolute_total_threshold_lines": 15000,
                    "absolute_age_threshold_seconds": 240,
                },
            },
        },
        collector_intake_audit={"status": "enforced"},
        storage_efficiency_contract={"overall_status": "ready", "grade": "A+"},
        storage_growth_forecast={"status": "stable_or_improving", "days_until_pressure_free": None},
        storage_retention_unison={
            "continuous_run_contract": {"status": "watch", "ready": True, "available_margin_gb": 19.809}
        },
        route_verified=True,
        resilience_status="ready",
        unresolved_split_brain_conflicts=0,
        retention_debt_gb=0.0,
        drain_minutes_total=0.602,
        data_integrity={
            "sql_invalid_lines": 0,
            "sql_overlay_invalid_lines": 0,
            "sql_overlay_ops_write_failures": 0,
            "sql_overlay_oversize_payloads": 0,
        },
    )

    assert payload["status"] == "watch"
    assert payload["ready"] is False
    assert payload["soak_ready"] is True
    assert payload["blockers"] == []
    assert "steady_state_pressure_index_in_bounded_soak_watch" in payload["warnings"]
    assert "pressure_index_only_clear_backlog_under_soak_controls" in payload["non_blocking_conditions"]
    assert payload["inputs"]["pressure_only_clear_backlog_soak_watch"] is True


def test_continuous_ingestion_soak_contract_tolerates_reserve_only_headroom() -> None:
    payload = src._continuous_ingestion_soak_contract(
        horizon_days=28.0,
        overall_status="ready",
        severity="stable",
        steady_state={"target_status": {"steady_state_ready": True}},
        recovery_scorecard={"score": 100.0},
        backlog_relief_contract={
            "active": True,
            "overall_grade": "C",
            "active_issue_ids": ["raw_live_expansion_headroom"],
            "raw_live_expansion_headroom": {
                "active": True,
                "grade": "C",
                "expansion_tier": "limited_expansion_only",
                "hard_block": False,
            },
        },
        collector_intake_audit={
            "status": "partial",
            "mismatches": [
                {
                    "key": "TRAINING_RUNTIME_PAUSED_FOR_BACKLOG",
                    "expected": "1",
                    "observed": "0",
                }
            ],
        },
        storage_efficiency_contract={"overall_status": "ready", "grade": "A+"},
        storage_growth_forecast={"status": "stable_or_improving", "days_until_pressure_free": 90.0},
        storage_retention_unison={"continuous_run_contract": {"status": "ready", "ready": True, "available_margin_gb": 80.0}},
        route_verified=True,
        resilience_status="ready",
        unresolved_split_brain_conflicts=0,
        retention_debt_gb=0.0,
        drain_minutes_total=3.0,
        data_integrity={
            "sql_invalid_lines": 0,
            "sql_overlay_invalid_lines": 0,
            "sql_overlay_ops_write_failures": 0,
            "sql_overlay_oversize_payloads": 0,
        },
    )

    assert payload["status"] == "ready"
    assert payload["ready"] is True
    assert payload["blockers"] == []
    assert "raw_live_expansion_headroom_limited_to_existing_collection" in payload["non_blocking_conditions"]
    assert "training_pause_mismatch_allowed_for_reserve_only_soak" in payload["non_blocking_conditions"]
    assert payload["inputs"]["collector_intake_soak_safe"] is True


def test_continuous_ingestion_soak_contract_tolerates_managed_sparse_jsonl_relief() -> None:
    payload = src._continuous_ingestion_soak_contract(
        horizon_days=28.0,
        overall_status="ready",
        severity="stable",
        steady_state={"target_status": {"steady_state_ready": True}},
        recovery_scorecard={"score": 96.0},
        backlog_relief_contract={
            "active": True,
            "overall_grade": "F",
            "active_issue_ids": ["sparse_huge_jsonl_files"],
        },
        collector_intake_audit={"status": "enforced"},
        storage_efficiency_contract={"overall_status": "ready", "grade": "A+"},
        storage_growth_forecast={"status": "stable_or_improving", "days_until_pressure_free": 90.0},
        storage_retention_unison={"continuous_run_contract": {"status": "ready", "ready": True, "available_margin_gb": 80.0}},
        route_verified=True,
        resilience_status="ready",
        unresolved_split_brain_conflicts=0,
        retention_debt_gb=0.0,
        drain_minutes_total=3.0,
        data_integrity={
            "sql_invalid_lines": 0,
            "sql_overlay_invalid_lines": 0,
            "sql_overlay_ops_write_failures": 0,
            "sql_overlay_oversize_payloads": 0,
        },
    )

    assert payload["status"] == "ready"
    assert payload["ready"] is True
    assert payload["blockers"] == []
    assert "managed_sparse_jsonl_backlog_under_storage_efficiency_contract" in payload["non_blocking_conditions"]
    assert payload["inputs"]["managed_sparse_jsonl_relief_soak_safe"] is True


def test_continuous_ingestion_soak_contract_marks_bounded_backlog_as_soak_ready_watch() -> None:
    payload = src._continuous_ingestion_soak_contract(
        horizon_days=30.0,
        overall_status="ready",
        severity="stable",
        steady_state={
            "target_status": {
                "steady_state_ready": False,
                "estimated_total_drain_minutes_ok": False,
            },
            "ratios": {
                "pressure_index": 1.8,
                "core_pending_lines": 1.35,
                "estimated_total_drain_minutes": 2.0,
            },
        },
        recovery_scorecard={"score": 72.0},
        backlog_relief_contract={
            "active": True,
            "overall_grade": "C",
            "active_issue_ids": ["intake_outpaces_drain", "raw_live_expansion_headroom"],
            "raw_live_expansion_headroom": {
                "hard_block": True,
                "raw_live": {
                    "core_pending_lines": 6800,
                    "total_pending_lines": 9800,
                    "oldest_pending_age_seconds": 12.0,
                },
                "targets": {
                    "absolute_total_threshold_lines": 15000,
                    "absolute_age_threshold_seconds": 240.0,
                },
            },
        },
        collector_intake_audit={"status": "enforced"},
        storage_efficiency_contract={"overall_status": "ready", "grade": "A+"},
        storage_growth_forecast={"status": "forecast_ready", "days_until_pressure_free": 90.0},
        storage_retention_unison={"continuous_run_contract": {"status": "ready", "ready": True, "available_margin_gb": 64.0}},
        route_verified=True,
        resilience_status="ready",
        unresolved_split_brain_conflicts=0,
        retention_debt_gb=0.0,
        drain_minutes_total=None,
        data_integrity={
            "sql_invalid_lines": 0,
            "sql_overlay_invalid_lines": 0,
            "sql_overlay_ops_write_failures": 0,
            "sql_overlay_oversize_payloads": 0,
        },
    )

    assert payload["status"] == "watch"
    assert payload["ready"] is False
    assert payload["soak_ready"] is True
    assert payload["grade"] == "A"
    assert payload["blockers"] == []
    assert "steady_state_targets_in_bounded_soak_watch" in payload["warnings"]
    assert "drain_time_unknown" in payload["warnings"]
    assert "bounded_steady_state_backlog_allowed_for_soak" in payload["non_blocking_conditions"]
    assert "bounded_intake_and_expansion_backlog_relief_under_soak_controls" in payload["non_blocking_conditions"]
    assert payload["inputs"]["bounded_soak_backlog_relief"] is True
    assert payload["control_env"]["BOT_CONTINUOUS_COLLECTION_READY"] == "1"
    assert payload["control_env"]["TRAINING_RUNTIME_PAUSED_FOR_BACKLOG"] == "1"


def test_continuous_ingestion_soak_contract_allows_pressure_only_writer_lag_watch() -> None:
    payload = src._continuous_ingestion_soak_contract(
        horizon_days=30.0,
        overall_status="ready",
        severity="stable",
        steady_state={
            "target_status": {
                "steady_state_ready": False,
                "target_breaches": ["pressure_index"],
                "core_pending_lines_ok": True,
                "estimated_total_drain_minutes_ok": True,
                "stale_stage_pending_lines_ok": True,
                "retention_debt_gb_ok": True,
            },
            "ratios": {
                "pressure_index": 2.9,
                "core_pending_lines": 0.5,
                "estimated_total_drain_minutes": 0.6,
            },
        },
        recovery_scorecard={"score": 96.0},
        backlog_relief_contract={
            "active": False,
            "overall_grade": "A+",
            "active_issue_ids": [],
            "raw_live_expansion_headroom": {
                "raw_live": {
                    "core_pending_lines": 2400,
                    "total_pending_lines": 2600,
                    "oldest_pending_age_seconds": 180.0,
                },
                "targets": {
                    "absolute_total_threshold_lines": 15000,
                    "absolute_age_threshold_seconds": 240.0,
                },
            },
        },
        collector_intake_audit={"status": "enforced"},
        storage_efficiency_contract={"overall_status": "ready", "grade": "A+"},
        storage_growth_forecast={"status": "stable_or_improving", "days_until_pressure_free": None},
        storage_retention_unison={"continuous_run_contract": {"status": "ready", "ready": True, "available_margin_gb": 34.0}},
        route_verified=True,
        resilience_status="ready",
        unresolved_split_brain_conflicts=0,
        retention_debt_gb=0.0,
        drain_minutes_total=8.0,
        data_integrity={
            "sql_invalid_lines": 0,
            "sql_overlay_invalid_lines": 0,
            "sql_overlay_ops_write_failures": 0,
            "sql_overlay_oversize_payloads": 0,
        },
    )

    assert payload["status"] == "watch"
    assert payload["soak_ready"] is True
    assert payload["blockers"] == []
    assert "steady_state_pressure_index_in_bounded_soak_watch" in payload["warnings"]
    assert "pressure_index_only_writer_lag_under_soak_controls" in payload["non_blocking_conditions"]
    assert payload["inputs"]["pressure_only_writer_lag_soak_watch"] is True
    assert payload["control_env"]["BOT_CONTINUOUS_COLLECTION_READY"] == "1"


def test_continuous_ingestion_soak_contract_allows_sparse_reserve_watch_under_deep_cold_controls() -> None:
    payload = src._continuous_ingestion_soak_contract(
        horizon_days=30.0,
        overall_status="ready",
        severity="stable",
        steady_state={
            "target_status": {
                "steady_state_ready": False,
                "target_breaches": ["pressure_index", "estimated_total_drain_minutes"],
            },
            "ratios": {
                "pressure_index": 1.172,
                "core_pending_lines": 0.88,
                "estimated_total_drain_minutes": 133.327,
            },
        },
        recovery_scorecard={"score": 72.0},
        backlog_relief_contract={
            "active": True,
            "overall_grade": "F",
            "active_issue_ids": ["sparse_huge_jsonl_files", "raw_live_expansion_headroom"],
            "raw_live_expansion_headroom": {
                "hard_block": False,
                "raw_live": {
                    "core_pending_lines": 4400,
                    "total_pending_lines": 11669,
                    "oldest_pending_age_seconds": 0.0,
                },
                "targets": {
                    "absolute_total_threshold_lines": 15000,
                    "absolute_age_threshold_seconds": 240.0,
                },
            },
        },
        collector_intake_audit={"status": "enforced"},
        storage_efficiency_contract={
            "overall_status": "ready",
            "grade": "A+",
            "deep_cold_managed_relief": True,
            "deep_cold_layer": {"ready": True},
        },
        storage_growth_forecast={"status": "stable_or_improving", "days_until_pressure_free": None},
        storage_retention_unison={"continuous_run_contract": {"status": "watch", "ready": True, "available_margin_gb": 52.6}},
        route_verified=True,
        resilience_status="ready",
        unresolved_split_brain_conflicts=0,
        retention_debt_gb=0.0,
        drain_minutes_total=1999.911,
        data_integrity={
            "sql_invalid_lines": 0,
            "sql_overlay_invalid_lines": 0,
            "sql_overlay_ops_write_failures": 0,
            "sql_overlay_oversize_payloads": 0,
        },
    )

    assert payload["status"] == "watch"
    assert payload["soak_ready"] is True
    assert payload["blockers"] == []
    assert "steady_state_sparse_reserve_in_bounded_soak_watch" in payload["warnings"]
    assert "backlog_relief_contract_active" not in payload["blockers"]
    assert "sparse_jsonl_and_raw_live_reserve_under_soak_controls" in payload["non_blocking_conditions"]
    assert payload["inputs"]["bounded_sparse_reserve_soak_watch"] is True
    assert payload["control_env"]["BOT_CONTINUOUS_COLLECTION_READY"] == "1"


def test_continuous_ingestion_soak_contract_tolerates_partial_sparse_relief_with_bounded_duty_cycle() -> None:
    payload = src._continuous_ingestion_soak_contract(
        horizon_days=28.0,
        overall_status="ready",
        severity="stable",
        steady_state={"target_status": {"steady_state_ready": True}},
        recovery_scorecard={"score": 96.0},
        backlog_relief_contract={
            "active": True,
            "overall_grade": "F",
            "active_issue_ids": ["sparse_huge_jsonl_files"],
        },
        collector_intake_audit={
            "status": "partial",
            "observed_env": {
                "BOT_COLLECTION_DUTY_CYCLE_ENABLED": {
                    "storage_pressure_override": "1",
                    "process_env": "1",
                },
                "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO": {
                    "storage_pressure_override": "0.35",
                    "process_env": "0.16",
                },
                "BOT_COLLECTION_DUTY_CYCLE_A_PLUS_PLUS_TARGET": {
                    "storage_pressure_override": "0",
                    "process_env": "0",
                },
                "TRAINING_RUNTIME_PAUSED_FOR_BACKLOG": {
                    "storage_pressure_override": "0",
                    "process_env": "0",
                },
            },
            "mismatches": [
                {
                    "key": "BOT_COLLECTION_DUTY_CYCLE_A_PLUS_PLUS_TARGET",
                    "expected": "1",
                    "observed": "0,0",
                },
                {
                    "key": "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO",
                    "expected": "0.20",
                    "observed": "0.35,0.16",
                },
                {
                    "key": "TRAINING_RUNTIME_PAUSED_FOR_BACKLOG",
                    "expected": "1",
                    "observed": "0,0",
                },
            ],
        },
        storage_efficiency_contract={"overall_status": "ready", "grade": "A+"},
        storage_growth_forecast={"status": "target_floor_breach", "days_until_pressure_free": 0.05},
        storage_retention_unison={"continuous_run_contract": {"status": "blocked", "ready": False, "available_margin_gb": -7000.0}},
        route_verified=True,
        resilience_status="ready",
        unresolved_split_brain_conflicts=0,
        retention_debt_gb=0.0,
        drain_minutes_total=15.0,
        data_integrity={
            "sql_invalid_lines": 0,
            "sql_overlay_invalid_lines": 0,
            "sql_overlay_ops_write_failures": 0,
            "sql_overlay_oversize_payloads": 0,
        },
    )

    assert payload["status"] == "ready"
    assert payload["ready"] is True
    assert payload["blockers"] == []
    assert "collector_partial_sparse_relief_bounded_by_visible_duty_cycle" in payload["non_blocking_conditions"]
    assert "managed_sparse_effective_queue_overrides_sparse_growth_forecast" in payload["non_blocking_conditions"]
    assert payload["inputs"]["collector_intake_soak_safe"] is True
    assert payload["inputs"]["managed_sparse_jsonl_relief_soak_safe"] is True


def test_continuous_ingestion_soak_contract_blocks_non_training_collector_mismatch() -> None:
    payload = src._continuous_ingestion_soak_contract(
        horizon_days=28.0,
        overall_status="ready",
        severity="stable",
        steady_state={"target_status": {"steady_state_ready": True}},
        recovery_scorecard={"score": 100.0},
        backlog_relief_contract={
            "active": True,
            "overall_grade": "C",
            "active_issue_ids": ["raw_live_expansion_headroom"],
            "raw_live_expansion_headroom": {"hard_block": False},
        },
        collector_intake_audit={
            "status": "partial",
            "mismatches": [
                {
                    "key": "BOT_COLLECTION_DUTY_CYCLE_ENABLED",
                    "expected": "1",
                    "observed": "0",
                }
            ],
        },
        storage_efficiency_contract={"overall_status": "ready", "grade": "A+"},
        storage_growth_forecast={"status": "forecast_ready", "days_until_pressure_free": 90.0},
        storage_retention_unison={"continuous_run_contract": {"status": "ready", "ready": True, "available_margin_gb": 80.0}},
        route_verified=True,
        resilience_status="ready",
        unresolved_split_brain_conflicts=0,
        retention_debt_gb=0.0,
        drain_minutes_total=3.0,
        data_integrity={
            "sql_invalid_lines": 0,
            "sql_overlay_invalid_lines": 0,
            "sql_overlay_ops_write_failures": 0,
            "sql_overlay_oversize_payloads": 0,
        },
    )

    assert payload["status"] == "blocked"
    assert "collector_intake_controls_not_enforced" in payload["blockers"]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    if path.name.startswith("jsonl_sql_ingestion_health") and path.name.endswith("_latest.json"):
        project_root = path.parents[2]
        sqlite = payload.get("sqlite") if isinstance(payload.get("sqlite"), dict) else {}
        for row in sqlite.get("top_pending_files") if isinstance(sqlite.get("top_pending_files"), list) else []:
            if not isinstance(row, dict):
                continue
            source_rel = str(row.get("source_rel") or "").strip()
            if not source_rel or source_rel.startswith("/") or ".." in Path(source_rel).parts:
                continue
            source_path = project_root / source_rel
            source_path.parent.mkdir(parents=True, exist_ok=True)
            if not source_path.exists():
                source_path.write_text('{"test": "sql_overlay_source"}\n', encoding="utf-8")


def test_ingestion_storage_control_keeps_support_training_tail_out_of_critical_path(tmp_path: Path) -> None:
    now = datetime(2026, 7, 23, 13, 45, tzinfo=timezone.utc)
    health = tmp_path / "governance" / "health"
    config = tmp_path / "config"
    config.mkdir()
    (config / ".env.storage_pressure_override").write_text(
        "\n".join(
            [
                "BOT_COLLECTION_DUTY_CYCLE_ENABLED=1",
                "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO=0.16",
                "BOT_COLLECTION_DUTY_CYCLE_A_PLUS_PLUS_TARGET=0",
                "TRAINING_RUNTIME_PAUSED_FOR_BACKLOG=0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "pending_lines": 7,
            "pending_lines_total": 3925,
            "pending_lines_deferred": 3918,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 2,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 0.0,
            "oldest_age_threshold_seconds": 240.0,
            "overload": False,
        },
    )
    _write_json(
        health / "jsonl_sql_ingestion_health_governance_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "sqlite": {
                "pending_lines": 3,
                "oldest_uningested_age_seconds": 1742.557,
                "files_with_pending": 1,
                "inserted": 0,
                "invalid": 1,
                "oversize_payloads": 0,
                "ops_write_failures": 0,
                "top_pending_files": [
                    {
                        "source_rel": "governance/training/raw_training_eligible_source_queue_latest.jsonl",
                        "stream": "governance",
                        "pending_lines": 3,
                        "oldest_pending_age_seconds": 1742.557,
                        "total_lines": 223,
                        "last_line": 220,
                    }
                ],
            },
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "cycle_started_utc": (now - timedelta(minutes=1)).isoformat(),
            "merged_rows_this_cycle": 120000,
        },
    )
    _write_json(health / "sql_link_service_latest.json", {"sqlite_wal_size_gb": 0.0})
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": False,
            "hard_gates": {},
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": False},
            "ingestion_pressure": {"severe_backpressure_overload": False},
        },
    )
    _write_json(
        health / "storage_failback_sync_latest.json",
        {"route_verification": {"verification_state": "curated_ready", "mismatches": []}},
    )
    _write_json(
        health / "storage_resilience_control_latest.json",
        {
            "overall_status": "ready",
            "resilience_score": 100,
            "restore_drill_fresh": True,
            "dual_root_ready": True,
            "warm_standby_ready": True,
            "unresolved_split_brain_conflicts": 0,
        },
    )
    _write_json(
        health / "storage_growth_forecast_latest.json",
        {"status": "stable_or_improving", "days_until_pressure_free": 90.0},
    )
    _write_json(
        health / "storage_retention_unison_latest.json",
        {"continuous_run_contract": {"status": "ready", "ready": True, "available_margin_gb": 64.0}},
    )
    _write_json(health / "data_collection_storage_guard_latest.json", {"duplicate_cleanup": {}, "safe_space_recovery": {}})
    _write_json(
        health / "raw_training_compaction_intelligence_latest.json",
        {
            "raw_summary": {
                "raw_jsonl_count": 2,
                "eligible_training_source_count": 1,
                "compression_candidate_count": 0,
                "compression_candidate_gb": 0.0,
                "local_fallback_reconciliation_count": 0,
                "current_day_protected_count": 0,
            }
        },
    )
    _write_json(health / "storage_quota_guard_latest.json", {"quota_summary": {"hard_breaches": 0, "soft_breaches": 0}, "lanes": []})
    _write_json(health / "storage_mount_guard_latest.json", {"external_available": True, "storage_mode": "external"})

    payload = src.build_payload(tmp_path, now_utc=now)

    assert payload["overall_status"] == "ready"
    assert payload["severity"] == "stable"
    assert payload["pressure_index"] < 0.75
    assert payload["sql_ingestion_pending_overlay"]["managed_support_training_tail_under_hot_path_limits"] is True
    assert payload["sql_ingestion_pending_overlay"]["managed_training_queue_invalid_quarantine"] is True
    assert payload["sql_ingestion_pending_overlay"]["raw_invalid_lines"] == 1
    assert payload["data_integrity"]["sql_overlay_invalid_lines"] == 0
    assert payload["storage_efficiency_contract"]["overall_status"] == "ready"
    assert payload["storage_efficiency_contract"]["grade"] == "A+"
    assert payload["continuous_run_soak_contract"]["grade"] == "A+"
    assert payload["continuous_run_soak_contract"]["soak_ready"] is True
    assert "stale_old_pending_work" not in payload["backlog_relief_contract"]["active_issue_ids"]


def test_ingestion_storage_control_estimates_drain_time_and_retention_pressure(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 18000,
            "pending_lines_total": 900000,
            "pending_lines_deferred": 882000,
            "pending_lines_cold": 600000,
            "pending_lines_support_telemetry": 220000,
            "pending_lines_stale_stage": 580000,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 600.0,
            "oldest_age_threshold_seconds": 240.0,
            "overload": True,
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "cycle_started_utc": (now - timedelta(minutes=10)).isoformat(),
            "merged_rows_this_cycle": 120000,
        },
    )
    _write_json(health / "sql_link_service_latest.json", {"sqlite_wal_size_gb": 19.4})
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": True,
            "recommended_operating_mode": "maintenance_only",
            "storage_pressure": {"retention_debt_gb": 4.2, "severe_backpressure_overload": True},
        },
    )
    _write_json(health / "storage_maintenance_latest.json", {"reason": "resource_guard_blocked"})
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "critical_backpressure",
            "sql_primary_db": {"route_drift": True},
            "queue_watermarks": {"overall_status": "blocked"},
            "writer_shedding": {"active": True, "level": "protect_core", "freeze_cold_lanes": True},
            "throttle_controls": {
                "deferred_files_budget": 0,
                "cold_files_budget": 0,
                "queue_prune_orphans": "1",
                "queue_orphan_days": 7,
                "queue_max_db_gb": 8,
                "stale_purge_low_value_days": 3,
                "stale_purge_medium_value_days": 14,
                "stale_purge_high_value_days": 30,
                "stale_purge_critical_value_days": 90,
                "stale_purge_max_gb": 20,
                "log_api_calls": "0",
                "log_loop_state": "0",
                "log_data_ingress": "0",
                "log_shadow_pnl_attribution": "0",
            },
        },
    )
    _write_json(
        health / "external_backlog_drain_latest.json",
        {
            "overall_status": "ready",
            "recommended_now": True,
            "aged_candidate_files": 4,
            "off_hours_window": {"active": True},
            "drain_overrides": {"deferred_files_budget": 6, "cold_files_budget": 2},
        },
    )
    _write_json(
        health / "storage_failback_sync_latest.json",
        {"route_verification": {"verification_state": "warning", "ready_count": 1, "tracked_count": 3, "coverage_ratio": 0.333333, "mismatches": ["data/jsonl_link.sqlite3"]}},
    )
    _write_json(health / "stale_artifact_sweeper_bot_latest.json", {"summary": {"candidate_files": 12, "candidate_bytes": 4096, "staged_files": 9, "staged_bytes": 3072}})
    _write_json(
        health / "stale_artifact_reaper_bot_latest.json",
        {
            "summary": {
                "candidate_files": 7,
                "candidate_bytes": 2048,
                "candidate_files_raw": 9,
                "candidate_bytes_raw": 4096,
                "deleted_files": 3,
                "deleted_bytes": 1024,
                "delete_errors": 1,
                "budget_limited": True,
                "skipped_by_budget_files": 2,
                "skipped_by_tier_files": 4,
                "manifest_lines_after": 12,
                "purge_policy": {"low_value_days": 3, "medium_value_days": 14},
            }
        },
    )
    _write_json(health / "data_retention_latest.json", {"deleted": 22})
    _write_json(
        health / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "files_discovered": 8, "sqlite": {"invalid": 0}},
    )

    payload = src.build_payload(tmp_path, now_utc=datetime(2026, 4, 6, 21, 0, tzinfo=timezone.utc))

    assert payload["overall_status"] == "blocked"
    assert payload["severity"] == "critical"
    assert payload["backpressure"]["estimated_core_drain_minutes"] is not None
    assert payload["storage"]["retention_debt_gb"] == 4.2
    assert payload["storage"]["governor_profile"] == "critical_backpressure"
    assert payload["storage"]["sql_primary_route_drift"] is True
    assert payload["storage"]["backlog_drain_recommended_now"] is True
    assert payload["storage"]["aged_backlog_candidate_files"] == 4
    assert payload["throttling"]["deferred_files_budget"] == 0
    assert payload["throttling"]["backlog_drain_deferred_budget"] == 6
    assert payload["throttling"]["queue_prune_orphans"] == "1"
    assert payload["throttling"]["stale_purge_low_value_days"] == 3
    assert payload["backpressure"]["support_pending_lines"] == 220000
    assert payload["backpressure"]["stale_stage_pending_lines"] == 580000
    assert payload["storage"]["stale_stage_deleted_bytes"] == 1024
    assert payload["storage"]["stale_stage_budget_limited"] is True
    assert payload["storage"]["stale_stage_delete_errors"] == 1
    assert payload["storage"]["stale_stage_purge_policy"]["low_value_days"] == 3
    assert payload["backpressure_quality_score"] < 50.0
    assert payload["steady_state"]["target_status"]["target_breach_count"] >= 4
    assert payload["recommended_operating_mode"] == "maintenance_drain_window"
    assert payload["top_actions"][0].startswith("normalize the SQL linker")
    assert any("support shard" in action for action in payload["top_actions"])
    assert any("stale-stage" in action for action in payload["top_actions"])
    assert payload["queue_watermarks"]["overall_status"] == "blocked"
    assert payload["writer_shedding"]["level"] == "protect_core"
    assert payload["external_route_verification"]["verification_state"] == "warning"


def test_ingestion_storage_control_uses_market_hours_protection_when_only_quarantine_is_available(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 9000,
            "pending_lines_total": 450000,
            "pending_lines_deferred": 220000,
            "pending_lines_cold": 180000,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 1800.0,
            "oldest_age_threshold_seconds": 240.0,
            "overload": True,
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "cycle_started_utc": (now - timedelta(minutes=8)).isoformat(),
            "merged_rows_this_cycle": 90000,
        },
    )
    _write_json(health / "sql_link_service_latest.json", {"sqlite_wal_size_gb": 2.0})
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": False,
            "recommended_operating_mode": "maintenance_only",
            "storage_pressure": {"retention_debt_gb": 3.5, "severe_backpressure_overload": True},
        },
    )
    _write_json(health / "storage_maintenance_latest.json", {"reason": "ok"})
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "critical_backpressure",
            "sql_primary_db": {"route_drift": False},
            "throttle_controls": {"deferred_files_budget": 0, "cold_files_budget": 0},
        },
    )
    _write_json(
        health / "external_backlog_drain_latest.json",
        {
            "overall_status": "blocked",
            "recommended_now": True,
            "aged_candidate_files": 2,
            "off_hours_window": {"active": False},
            "drain_overrides": {"deferred_files_budget": 6, "cold_files_budget": 2},
        },
    )
    _write_json(
        health / "backlog_quarantine_bot_latest.json",
        {
            "overall_status": "ready",
            "candidate_files": 2,
            "moved_files": 0,
            "moved_pending_lines": 0,
        },
    )
    _write_json(health / "stale_artifact_sweeper_bot_latest.json", {"summary": {"candidate_files": 0, "staged_files": 0}})
    _write_json(health / "stale_artifact_reaper_bot_latest.json", {"summary": {"deleted_files": 0}})
    _write_json(health / "data_retention_latest.json", {"deleted": 0})
    _write_json(
        health / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "files_discovered": 8, "sqlite": {"invalid": 0}},
    )

    payload = src.build_payload(tmp_path, now_utc=datetime(2026, 4, 7, 15, 0, tzinfo=timezone.utc))

    assert payload["recommended_operating_mode"] == "market_hours_backlog_protection"
    assert payload["storage"]["backlog_quarantine_candidate_files"] == 2


def test_ingestion_storage_control_reports_green_steady_state_targets(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 300,
            "pending_lines_total": 1200,
            "pending_lines_deferred": 900,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 0,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 15.0,
            "oldest_age_threshold_seconds": 240.0,
            "overload": False,
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "cycle_started_utc": (now - timedelta(minutes=5)).isoformat(),
            "merged_rows_this_cycle": 90000,
        },
    )
    _write_json(health / "sql_link_service_latest.json", {"sqlite_wal_size_gb": 0.1})
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": False,
            "recommended_operating_mode": "live_cautious",
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": False},
            "ingestion_pressure": {"severe_backpressure_overload": False},
        },
    )
    _write_json(health / "storage_maintenance_latest.json", {"reason": "ok"})
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "steady_state",
            "sql_primary_db": {"route_drift": False},
            "queue_watermarks": {"overall_status": "ready"},
            "writer_shedding": {"active": False, "level": "normal"},
            "throttle_controls": {"deferred_files_budget": 2, "cold_files_budget": 1},
        },
    )
    _write_json(health / "external_backlog_drain_latest.json", {"overall_status": "ready", "recommended_now": False, "aged_candidate_files": 0})
    _write_json(health / "stale_artifact_sweeper_bot_latest.json", {"summary": {"candidate_files": 0, "staged_files": 0}})
    _write_json(health / "stale_artifact_reaper_bot_latest.json", {"summary": {"deleted_files": 0}})
    _write_json(health / "data_retention_latest.json", {"deleted": 0})
    _write_json(
        health / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "files_discovered": 4, "sqlite": {"invalid": 0}},
    )

    payload = src.build_payload(tmp_path, now_utc=datetime(2026, 4, 7, 18, 0, tzinfo=timezone.utc))

    assert payload["overall_status"] == "ready"
    assert payload["severity"] == "stable"
    assert payload["backpressure_quality_score"] >= 95.0
    assert payload["steady_state"]["target_status"]["steady_state_ready"] is True
    assert payload["steady_state"]["target_status"]["target_breaches"] == []
    assert payload["queue_watermarks"]["overall_status"] == "ready"
    assert payload["writer_shedding"]["active"] is False
    assert payload["storage_efficiency_contract"]["overall_status"] == "ready"
    assert payload["storage_efficiency_contract"]["grade"] == "A+"


def test_ingestion_storage_control_decays_stale_overload_gate_when_measured_backpressure_is_clear(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    config = tmp_path / "config"
    config.mkdir()
    (config / ".env.storage_pressure_override").write_text(
        "\n".join(
            [
                "BOT_COLLECTION_DUTY_CYCLE_ENABLED=1",
                "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO=0.16",
                "BOT_COLLECTION_DUTY_CYCLE_A_PLUS_PLUS_TARGET=0",
                "TRAINING_RUNTIME_PAUSED_FOR_BACKLOG=1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "timestamp_utc": (now - timedelta(minutes=4)).isoformat(),
            "pending_lines": 2315,
            "pending_lines_total": 9168,
            "pending_lines_deferred": 6853,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 23,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 0.0,
            "oldest_age_threshold_seconds": 240.0,
            "overload": True,
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "cycle_started_utc": (now - timedelta(minutes=5)).isoformat(),
            "merged_rows_this_cycle": 90000,
        },
    )
    _write_json(health / "sql_link_service_latest.json", {"sqlite_wal_size_gb": 0.0})
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": True,
            "recommended_operating_mode": "shadow_only",
            "hard_gates": {"ingestion_backpressure_overload": True},
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": False},
            "ingestion_pressure": {"severe_backpressure_overload": True},
        },
    )
    _write_json(health / "storage_maintenance_latest.json", {"reason": "ok"})
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "critical_backpressure",
            "sql_primary_db": {"route_drift": False},
            "queue_watermarks": {"overall_status": "ready"},
            "writer_shedding": {"active": True, "level": "protect_core"},
            "throttle_controls": {"deferred_files_budget": 2, "cold_files_budget": 0},
        },
    )
    _write_json(health / "external_backlog_drain_latest.json", {"overall_status": "blocked", "recommended_now": False, "aged_candidate_files": 0})
    _write_json(
        health / "storage_failback_sync_latest.json",
        {"route_verification": {"verification_state": "ready", "ready_count": 3, "tracked_count": 3, "coverage_ratio": 1.0, "mismatches": []}},
    )
    _write_json(
        health / "storage_resilience_control_latest.json",
        {
            "overall_status": "ready",
            "resilience_score": 100,
            "restore_drill_fresh": True,
            "dual_root_ready": True,
            "warm_standby_ready": True,
            "unresolved_split_brain_conflicts": 0,
        },
    )
    _write_json(
        health / "data_collection_storage_guard_latest.json",
        {
            "disk": {"available_gb": 220.0, "used_percent": 77.0},
            "safe_space_recovery": {"candidate_count": 0, "candidate_gb": 0.0, "target_free_gb": 64.0, "target_free_deficit_gb": 0.0},
            "duplicate_cleanup": {"candidate_count": 0, "candidate_gb": 0.0},
        },
    )
    _write_json(
        health / "raw_training_compaction_intelligence_latest.json",
        {
            "raw_summary": {
                "raw_jsonl_count": 736,
                "eligible_training_source_count": 597,
                "compression_candidate_count": 0,
                "compression_candidate_gb": 0.0,
                "local_fallback_reconciliation_count": 0,
                "current_day_protected_count": 617,
            }
        },
    )
    _write_json(health / "storage_quota_guard_latest.json", {"quota_summary": {"hard_breaches": 0, "soft_breaches": 0}, "lanes": []})
    _write_json(health / "storage_growth_forecast_latest.json", {"status": "forecast_ready", "days_until_pressure_free": 90.0})
    _write_json(
        health / "storage_retention_unison_latest.json",
        {"continuous_run_contract": {"status": "ready", "ready": True, "available_margin_gb": 120.0}},
    )
    _write_json(
        health / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "files_discovered": 4, "sqlite": {"invalid": 0}, "pending_lines": 0, "files_with_pending": 0},
    )

    payload = src.build_payload(tmp_path, now_utc=now)

    assert payload["overall_status"] == "ready"
    assert payload["severity"] == "stable"
    assert payload["steady_state"]["target_status"]["steady_state_ready"] is True
    assert payload["bounded_recovery_contract"]["effective_hard_gate_active"] is False
    assert "ingestion_backpressure_overload" in payload["bounded_recovery_contract"]["stale_hard_gate_suppressed"]
    assert "ingestion_backpressure_latest.overload" in payload["stabilization_contract"]["stale_backpressure_overload_suppressed"]
    assert payload["storage_efficiency_contract"]["overall_status"] == "ready"
    assert payload["continuous_run_soak_contract"]["soak_ready"] is True


def test_ingestion_storage_control_builds_manifest_first_storage_efficiency_contract(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 21000,
            "pending_lines_total": 180000,
            "pending_lines_deferred": 110000,
            "pending_lines_cold": 42000,
            "pending_lines_support_telemetry": 7000,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 900.0,
            "oldest_age_threshold_seconds": 240.0,
            "overload": True,
            "line_estimation": {
                "sparse_large_line_active": True,
                "sparse_large_line_pending_bytes": 188_743_680,
                "sparse_large_line_pending_lines": 92,
            },
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "cycle_started_utc": (now - timedelta(minutes=12)).isoformat(),
            "merged_rows_this_cycle": 12000,
        },
    )
    _write_json(health / "sql_link_service_latest.json", {"sqlite_wal_size_gb": 0.3})
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": False,
            "recommended_operating_mode": "maintenance_only",
            "storage_pressure": {"retention_debt_gb": 1.4, "severe_backpressure_overload": True},
            "ingestion_pressure": {"severe_backpressure_overload": True},
        },
    )
    _write_json(health / "storage_maintenance_latest.json", {"reason": "ok"})
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "critical_backpressure",
            "sql_primary_db": {"route_drift": True},
            "queue_watermarks": {"overall_status": "blocked"},
            "writer_shedding": {"active": True, "level": "protect_core", "freeze_cold_lanes": True},
            "throttle_controls": {"deferred_files_budget": 0, "cold_files_budget": 0},
        },
    )
    _write_json(health / "external_backlog_drain_latest.json", {"overall_status": "ready", "recommended_now": True, "aged_candidate_files": 3})
    _write_json(
        health / "storage_failback_sync_latest.json",
        {
            "route_verification": {
                "verification_state": "warning",
                "ready_count": 2,
                "tracked_count": 4,
                "coverage_ratio": 0.5,
                "mismatches": ["data/jsonl_link.sqlite3"],
            }
        },
    )
    _write_json(health / "storage_mount_guard_latest.json", {"external_available": True, "storage_mode": "external"})
    _write_json(
        health / "data_collection_storage_guard_latest.json",
        {
            "overall_status": "ready",
            "duplicate_cleanup": {"enabled": True, "candidate_count": 7, "candidate_gb": 3.25},
        },
    )
    _write_json(
        health / "raw_training_compaction_intelligence_latest.json",
        {
            "raw_summary": {
                "raw_jsonl_count": 90,
                "eligible_training_source_count": 76,
                "compression_candidate_count": 32,
                "compression_candidate_gb": 18.5,
                "local_fallback_reconciliation_count": 4,
                "current_day_protected_count": 6,
            }
        },
    )
    _write_json(
        health / "storage_quota_guard_latest.json",
        {
            "overall_status": "degraded",
            "quota_summary": {"hard_breaches": 0, "soft_breaches": 1},
            "lanes": [{"family": "decisions", "status": "degraded"}],
        },
    )
    _write_json(
        health / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "files_discovered": 8, "sqlite": {"invalid": 0}},
    )

    payload = src.build_payload(tmp_path, now_utc=datetime(2026, 4, 7, 21, 0, tzinfo=timezone.utc))
    contract = payload["storage_efficiency_contract"]

    assert contract["overall_status"] == "needs_work"
    assert contract["write_intake_mode"] == "manifest_only_hot_path"
    assert contract["raw_payload_policy"] == "manifest_first_compress_old_sources"
    assert contract["dedupe_required"] is True
    assert contract["raw_compaction_required"] is True
    assert contract["fallback_reconciliation_required"] is True
    assert contract["quota_relief_required"] is True
    assert contract["adaptive_raw_training_wave"]["manifest_refresh_required"] is True
    assert contract["adaptive_raw_training_wave"]["compaction_apply_allowed_now"] is False
    assert contract["storage_plane_phase_contract"]["phase"] == "manifest_only_recovery"
    assert contract["storage_plane_phase_contract"]["allowed_work"]["raw_training_manifest_refresh"] is True
    assert contract["storage_plane_phase_contract"]["allowed_work"]["raw_training_compaction_apply"] is False
    assert contract["recommended_commands"]["raw_training_manifest_refresh"]["active"] is True
    assert contract["recommended_commands"]["raw_training_compaction_wave"]["active"] is False
    assert contract["recommended_commands"]["dedupe_fallback_artifacts"]["active"] is True
    assert contract["control_env_recommendations"]["BOT_RAW_PAYLOAD_STORAGE_MODE"] == "manifest_first"
    assert contract["control_env_recommendations"]["BOT_RAW_TRAINING_WAVE_MAX_FILES"] == "4"
    assert contract["control_env_recommendations"]["BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO"] == "0.15"
    assert payload["storage"]["write_intake_mode"] == "manifest_only_hot_path"
    assert payload["storage_plane_contract"]["phase"] == "manifest_only_recovery"


def test_ingestion_storage_control_does_not_label_sparse_tail_as_raw_or_fallback_debt(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 509,
            "pending_lines_total": 948,
            "pending_lines_deferred": 439,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 11,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 21.812,
            "oldest_age_threshold_seconds": 240.0,
            "overload": False,
            "line_estimation": {
                "sparse_large_line_active": True,
                "sparse_large_line_pending_lines": 478,
                "sparse_large_line_pending_bytes": 507_583_898,
            },
        },
    )
    _write_json(health / "sql_link_service_progress_latest.json", {"cycle_started_utc": (now - timedelta(minutes=2)).isoformat(), "merged_rows_this_cycle": 1545})
    _write_json(health / "sql_link_service_latest.json", {"sqlite_wal_size_gb": 0.0})
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": False,
            "recommended_operating_mode": "live_cautious",
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": False},
            "ingestion_pressure": {"severe_backpressure_overload": False},
        },
    )
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "steady_state",
            "sql_primary_db": {"route_drift": False},
            "queue_watermarks": {"overall_status": "ready"},
            "writer_shedding": {"active": False, "level": "normal"},
        },
    )
    _write_json(health / "external_backlog_drain_latest.json", {"overall_status": "ready", "recommended_now": False, "aged_candidate_files": 0})
    _write_json(health / "storage_maintenance_latest.json", {"reason": "ok"})
    _write_json(
        health / "storage_failback_sync_latest.json",
        {"route_verification": {"verification_state": "ready", "ready_count": 3, "tracked_count": 3, "coverage_ratio": 1.0, "mismatches": []}},
    )
    _write_json(
        health / "storage_resilience_control_latest.json",
        {
            "overall_status": "ready",
            "resilience_score": 100,
            "restore_drill_fresh": True,
            "dual_root_ready": True,
            "warm_standby_ready": True,
            "unresolved_split_brain_conflicts": 0,
        },
    )
    _write_json(
        health / "data_collection_storage_guard_latest.json",
        {
            "disk": {"available_gb": 220.0, "used_percent": 77.0},
            "safe_space_recovery": {
                "candidate_count": 0,
                "candidate_gb": 0.0,
                "target_free_gb": 220.0,
                "target_free_deficit_gb": 0.0,
                "scan": {"unbacked_duplicate_count": 7, "unbacked_duplicate_gb": 0.0},
            },
            "duplicate_cleanup": {"enabled": True, "candidate_count": 7, "candidate_gb": 0.0},
        },
    )
    _write_json(
        health / "raw_training_compaction_intelligence_latest.json",
        {
            "raw_summary": {
                "raw_jsonl_count": 281,
                "eligible_training_source_count": 256,
                "compression_candidate_count": 0,
                "compression_candidate_gb": 0.0,
                "local_fallback_reconciliation_count": 0,
                "current_day_protected_count": 247,
            }
        },
    )
    _write_json(
        health / "storage_quota_guard_latest.json",
        {"quota_summary": {"hard_breaches": 0, "soft_breaches": 0}, "lanes": []},
    )
    _write_json(
        health / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "files_discovered": 4, "sqlite": {"invalid": 0}},
    )

    payload = src.build_payload(tmp_path, now_utc=now)
    contract = payload["storage_efficiency_contract"]

    assert contract["overall_status"] == "ready"
    assert contract["grade"] == "A+"
    assert contract["active_blockers"] == []
    assert "raw_training_compaction_debt" not in contract["active_blockers"]
    assert "fallback_route_reconciliation" not in contract["active_blockers"]
    assert "duplicate_fallback_artifacts" not in contract["active_blockers"]
    assert contract["dedupe_required"] is False
    assert contract["raw_candidate_compaction_required"] is False
    assert contract["sparse_byte_window_required"] is True
    assert contract["fallback_reconciliation_required"] is False
    assert contract["metrics"]["raw_compression_candidate_gb"] == 0.0
    assert contract["metrics"]["local_fallback_reconciliation_count"] == 0
    assert contract["metrics"]["backlog_relief_sparse_watch_only"] is True
    assert payload["recovery_quality_score"] == 100.0
    assert payload["recovery_contract"]["full_steady_state_recovery_credit"] is True


def test_storage_efficiency_treats_tiny_raw_compaction_tail_as_manifest_watch(tmp_path: Path) -> None:
    contract = src._ingestion_storage_efficiency_contract(
        project_root=tmp_path,
        severity="stable",
        queue_watermarks={"overall_status": "ready"},
        backlog_relief_contract={"active": False, "overall_grade": "A+"},
        data_collection_storage_guard={
            "disk": {"available_gb": 510.0, "used_percent": 45.0},
            "safe_space_recovery": {
                "candidate_count": 0,
                "candidate_gb": 0.0,
                "selected_gb": 0.0,
                "target_free_gb": 125.0,
                "target_free_deficit_gb": 0.0,
                "scan": {"unbacked_duplicate_count": 0, "unbacked_duplicate_gb": 0.0},
            },
            "duplicate_cleanup": {"candidate_count": 0, "candidate_gb": 0.0},
        },
        raw_training_compaction={
            "raw_summary": {
                "raw_jsonl_count": 204,
                "eligible_training_source_count": 96,
                "compression_candidate_count": 9,
                "compression_candidate_gb": 0.021,
                "local_fallback_reconciliation_count": 0,
                "current_day_protected_count": 88,
            }
        },
        storage_quota={"quota_summary": {"hard_breaches": 0, "soft_breaches": 0}, "lanes": []},
        storage_mount={},
        route_drift=False,
        route_verified=True,
        route_verification_state="ready",
        route_verification={"mismatches": []},
        unresolved_split_brain_conflicts=0,
        line_estimation={},
        total_pending_lines=0,
        core_pending_lines=0,
        retention_debt_gb=0.0,
        overlay_pressure_clear=True,
    )

    assert contract["overall_status"] == "ready"
    assert contract["grade"] == "A+"
    assert contract["raw_candidate_compaction_required"] is False
    assert contract["raw_candidate_count_pressure"] is False
    assert contract["recommended_commands"]["raw_training_manifest_refresh"]["active"] is True
    assert "raw_training_compaction_debt" not in contract["active_blockers"]


def test_ingestion_storage_control_enters_emergency_disk_guard_when_external_free_space_is_tiny(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health = tmp_path / "governance" / "health"
    external_root = tmp_path / "missing_bot_logs"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 2000,
            "pending_lines_total": 4000,
            "pending_lines_deferred": 2000,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 0,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 20.0,
            "oldest_age_threshold_seconds": 240.0,
            "overload": False,
        },
    )
    _write_json(health / "sql_link_service_progress_latest.json", {"cycle_started_utc": (now - timedelta(minutes=5)).isoformat(), "merged_rows_this_cycle": 5000})
    _write_json(health / "sql_link_service_latest.json", {"sqlite_wal_size_gb": 0.0})
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": False,
            "recommended_operating_mode": "live_cautious",
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": False},
            "ingestion_pressure": {"severe_backpressure_overload": False},
        },
    )
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {"profile": "steady_state", "sql_primary_db": {"route_drift": False}, "writer_shedding": {"active": False}},
    )
    _write_json(health / "external_backlog_drain_latest.json", {"overall_status": "ready", "recommended_now": False, "aged_candidate_files": 0})
    _write_json(
        health / "storage_failback_sync_latest.json",
        {"route_verification": {"verification_state": "ready", "ready_count": 3, "tracked_count": 3, "coverage_ratio": 1.0, "mismatches": []}},
    )
    _write_json(health / "storage_mount_guard_latest.json", {"external_available": True, "storage_mode": "external"})
    _write_json(
        health / "data_collection_storage_guard_latest.json",
        {
            "external_root": str(external_root),
            "disk": {"available_gb": 1.5, "used_percent": 99.8},
            "duplicate_cleanup": {"enabled": True, "candidate_count": 0, "candidate_gb": 0.0},
        },
    )
    _write_json(
        health / "raw_training_compaction_intelligence_latest.json",
        {
            "scan_roots": [{"path": str(external_root), "exists": False, "protected": False}],
            "raw_summary": {
                "raw_jsonl_count": 22,
                "eligible_training_source_count": 20,
                "compression_candidate_count": 12,
                "compression_candidate_gb": 9.0,
                "local_fallback_reconciliation_count": 0,
            },
        },
    )
    _write_json(
        health / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "files_discovered": 4, "sqlite": {"invalid": 0}},
    )

    payload = src.build_payload(tmp_path, now_utc=now)
    contract = payload["storage_efficiency_contract"]

    assert contract["storage_plane_phase_contract"]["phase"] == "emergency_disk_guard"
    assert contract["storage_plane_phase_contract"]["disk_contract"]["external_available_gb"] == 1.5
    assert contract["adaptive_raw_training_wave"]["max_files"] == 0
    assert contract["adaptive_raw_training_wave"]["max_gb"] == 0.0
    assert contract["control_env_recommendations"]["BOT_STORAGE_EMERGENCY_DISK_GUARD"] == "1"
    assert contract["control_env_recommendations"]["BOT_STORAGE_ALLOW_RAW_COMPACTION_APPLY"] == "0"
    assert contract["recommended_commands"]["raw_training_manifest_refresh"]["active"] is True


def test_ingestion_storage_control_credits_a_plus_relief_when_queue_is_small_but_not_freshest(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 342,
            "pending_lines_total": 422,
            "pending_lines_deferred": 80,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 77,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 128.98,
            "oldest_age_threshold_seconds": 240.0,
            "overload": False,
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "cycle_started_utc": (now - timedelta(minutes=2)).isoformat(),
            "merged_rows_this_cycle": 0,
        },
    )
    _write_json(health / "sql_link_service_latest.json", {"sqlite_wal_size_gb": 0.0})
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": False,
            "recommended_operating_mode": "live_cautious",
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": False},
            "ingestion_pressure": {"severe_backpressure_overload": False},
        },
    )
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "steady_state",
            "sql_primary_db": {"route_drift": False},
            "queue_watermarks": {"overall_status": "ready"},
            "writer_shedding": {"active": False, "level": "normal"},
        },
    )
    _write_json(health / "external_backlog_drain_latest.json", {"overall_status": "ready", "recommended_now": False, "aged_candidate_files": 0})
    _write_json(health / "storage_maintenance_latest.json", {"reason": "ok"})
    _write_json(
        health / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "files_discovered": 4, "sqlite": {"invalid": 0}},
    )

    payload = src.build_payload(tmp_path, now_utc=now)

    assert payload["backlog_relief_contract"]["overall_grade"] == "A+"
    assert payload["backlog_relief_contract"]["active_issue_count"] == 0
    assert payload["backpressure_quality_score"] >= 97.0
    assert payload["steady_state"]["target_status"]["backlog_relief_a_plus_ready"] is True


def test_ingestion_storage_control_does_not_penalize_tiny_idle_queue_for_missing_drain_estimate(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 125,
            "pending_lines_total": 352,
            "pending_lines_deferred": 227,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 6,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 0.0,
            "oldest_age_threshold_seconds": 240.0,
            "overload": False,
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "cycle_started_utc": (now - timedelta(minutes=10)).isoformat(),
            "merged_rows_this_cycle": 0,
        },
    )
    _write_json(health / "sql_link_service_latest.json", {"sqlite_wal_size_gb": 0.0})
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": False,
            "recommended_operating_mode": "live_cautious",
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": False},
            "ingestion_pressure": {"severe_backpressure_overload": False},
        },
    )
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "steady_state",
            "sql_primary_db": {"route_drift": False},
            "queue_watermarks": {"overall_status": "ready"},
            "writer_shedding": {"active": False, "level": "normal"},
        },
    )
    _write_json(health / "external_backlog_drain_latest.json", {"overall_status": "ready", "recommended_now": False, "aged_candidate_files": 0})
    _write_json(health / "storage_maintenance_latest.json", {"reason": "ok"})
    _write_json(
        health / "storage_failback_sync_latest.json",
        {"route_verification": {"verification_state": "ready", "ready_count": 3, "tracked_count": 3, "coverage_ratio": 1.0, "mismatches": []}},
    )
    _write_json(
        health / "storage_resilience_control_latest.json",
        {
            "overall_status": "ready",
            "resilience_score": 78,
            "restore_drill_fresh": False,
            "dual_root_ready": True,
            "warm_standby_ready": True,
            "unresolved_split_brain_conflicts": 0,
        },
    )
    _write_json(
        health / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "files_discovered": 4, "sqlite": {"invalid": 0}},
    )

    payload = src.build_payload(tmp_path, now_utc=now)

    assert payload["overall_status"] == "ready"
    assert payload["severity"] == "stable"
    assert payload["backpressure_quality_score"] >= 95.0
    assert payload["recovery_quality_score"] >= 88.0
    assert payload["recovery_contract"]["steady_state_recovery_ready"] is True
    assert "estimated_total_drain_minutes" not in payload["steady_state"]["target_status"]["target_breaches"]


def test_ingestion_storage_control_tolerates_bounded_hot_queue_with_unknown_drain_estimate(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 3169,
            "pending_lines_total": 7201,
            "pending_lines_deferred": 4032,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 4,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 24.23,
            "oldest_age_threshold_seconds": 240.0,
            "overload": False,
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "cycle_started_utc": (now - timedelta(minutes=10)).isoformat(),
            "merged_rows_this_cycle": 0,
        },
    )
    _write_json(health / "sql_link_service_latest.json", {"sqlite_wal_size_gb": 0.0})
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": False,
            "recommended_operating_mode": "live_cautious",
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": False},
            "ingestion_pressure": {"severe_backpressure_overload": False},
        },
    )
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "steady_state",
            "sql_primary_db": {"route_drift": False},
            "queue_watermarks": {"overall_status": "ready"},
            "writer_shedding": {"active": False, "level": "normal"},
        },
    )
    _write_json(health / "external_backlog_drain_latest.json", {"overall_status": "ready", "recommended_now": False, "aged_candidate_files": 0})
    _write_json(health / "storage_maintenance_latest.json", {"reason": "ok"})
    _write_json(
        health / "storage_failback_sync_latest.json",
        {"route_verification": {"verification_state": "ready", "ready_count": 3, "tracked_count": 3, "coverage_ratio": 1.0, "mismatches": []}},
    )
    _write_json(
        health / "storage_resilience_control_latest.json",
        {
            "overall_status": "ready",
            "resilience_score": 78,
            "restore_drill_fresh": False,
            "dual_root_ready": True,
            "warm_standby_ready": True,
            "unresolved_split_brain_conflicts": 0,
        },
    )
    _write_json(
        health / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "files_discovered": 4, "sqlite": {"invalid": 0}},
    )

    payload = src.build_payload(tmp_path, now_utc=now)

    assert payload["backpressure"]["estimated_total_drain_minutes"] is None
    assert payload["steady_state"]["target_status"]["estimated_total_drain_minutes_ok"] is True
    assert "estimated_total_drain_minutes" not in payload["steady_state"]["target_status"]["target_breaches"]
    assert payload["backpressure_quality_score"] >= 95.0


def test_ingestion_storage_control_bounds_isolated_support_overlay_pressure(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 125,
            "pending_lines_total": 352,
            "pending_lines_deferred": 227,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 6,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 0.0,
            "oldest_age_threshold_seconds": 240.0,
            "overload": False,
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "status": "running",
            "cycle_started_utc": (now - timedelta(minutes=3)).isoformat(),
            "merged_rows_this_cycle": 1200,
        },
    )
    _write_json(
        health / "jsonl_sql_ingestion_health_support_watchdog_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "sqlite": {
                "pending_lines": 55000,
                "files_with_pending": 1,
                "invalid": 0,
                "oldest_uningested_age_seconds": 600.0,
                "top_pending_files": [
                    {
                        "source_rel": "governance/channels/risk/conservative_equities_schwab/risk_20260521.jsonl",
                        "pending_lines": 55000,
                        "oldest_pending_age_seconds": 600.0,
                    }
                ],
            },
        },
    )
    _write_json(health / "sql_link_service_latest.json", {"sqlite_wal_size_gb": 0.0})
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": False,
            "recommended_operating_mode": "live_cautious",
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": False},
            "ingestion_pressure": {"severe_backpressure_overload": False},
        },
    )
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "steady_state",
            "sql_primary_db": {"route_drift": False},
            "queue_watermarks": {"overall_status": "ready"},
            "writer_shedding": {"active": False, "level": "normal"},
        },
    )
    _write_json(health / "external_backlog_drain_latest.json", {"overall_status": "ready", "recommended_now": False, "aged_candidate_files": 0})
    _write_json(health / "storage_maintenance_latest.json", {"reason": "ok"})
    _write_json(
        health / "storage_failback_sync_latest.json",
        {"route_verification": {"verification_state": "ready", "ready_count": 3, "tracked_count": 3, "coverage_ratio": 1.0, "mismatches": []}},
    )
    _write_json(
        health / "storage_resilience_control_latest.json",
        {
            "overall_status": "ready",
            "resilience_score": 100,
            "restore_drill_fresh": True,
            "dual_root_ready": True,
            "warm_standby_ready": True,
            "unresolved_split_brain_conflicts": 0,
        },
    )

    payload = src.build_payload(tmp_path, now_utc=now)

    assert payload["backpressure"]["raw_live"]["total_pending_lines"] == 352
    assert payload["backpressure"]["support_pending_lines"] == 55000
    assert payload["steady_state"]["support_overlay_isolated"] is True
    assert payload["backpressure_quality_score"] >= 96.0
    assert payload["recovery_quality_score"] >= 88.0


def test_ingestion_storage_control_decays_fresh_overlay_when_raw_backpressure_cleared(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 516,
            "pending_lines_total": 1149,
            "pending_lines_deferred": 633,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 10,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 0.898,
            "oldest_age_threshold_seconds": 240.0,
            "overload": False,
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "cycle_started_utc": (now - timedelta(minutes=5)).isoformat(),
            "merged_rows_this_cycle": 140000,
        },
    )
    _write_json(health / "sql_link_service_latest.json", {"sqlite_wal_size_gb": 0.0})
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": False,
            "recommended_operating_mode": "live_cautious",
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": False},
            "ingestion_pressure": {"severe_backpressure_overload": False},
        },
    )
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "critical_backpressure",
            "sql_primary_db": {"route_drift": False},
            "queue_watermarks": {"overall_status": "ready"},
            "writer_shedding": {"active": True, "level": "protect_core"},
        },
    )
    _write_json(
        health / "external_backlog_drain_latest.json",
        {"overall_status": "drain_active", "recommended_now": True, "aged_candidate_files": 0},
    )
    _write_json(
        health / "jsonl_sql_ingestion_health_governance_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "sqlite": {
                "pending_lines": 122221,
                "files_with_pending": 1,
                "oldest_uningested_age_seconds": 4.279,
                "top_pending_files": [
                    {
                        "source_rel": "governance/events/signal_generation_20260524.jsonl",
                        "stream": "governance_events",
                        "storage_temperature": "warm",
                        "ingestion_lane": "nearline_lane",
                        "pending_lines": 122221,
                        "oldest_pending_age_seconds": 4.279,
                    }
                ],
            },
        },
    )
    _write_json(
        health / "jsonl_sql_ingestion_health_crypto_trading_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "sqlite": {
                "pending_lines": 106398,
                "files_with_pending": 2,
                "oldest_uningested_age_seconds": 17.308,
                "top_pending_files": [
                    {
                        "source_rel": "decisions/shadow_crypto/trade_decisions_20260524.jsonl",
                        "stream": "decisions",
                        "storage_temperature": "hot",
                        "ingestion_lane": "hot_lane",
                        "pending_lines": 60441,
                        "oldest_pending_age_seconds": 13.844,
                    },
                    {
                        "source_rel": "decisions/shadow_crypto_futures_crypto/trade_decisions_20260524.jsonl",
                        "stream": "decisions",
                        "storage_temperature": "hot",
                        "ingestion_lane": "hot_lane",
                        "pending_lines": 45957,
                        "oldest_pending_age_seconds": 17.308,
                    },
                ],
            },
        },
    )

    payload = src.build_payload(tmp_path, now_utc=now)

    assert payload["backpressure"]["overlay_adjusted"] is False
    assert payload["backpressure"]["core_pending_lines"] == 516
    assert payload["backpressure"]["estimated_total_drain_minutes"] <= 15.0
    assert payload["stabilization_contract"]["small_hot_queue_stable"] is True
    assert payload["backlog_truth"]["authoritative_mode"] == "raw_live_overlay_decayed"
    assert payload["overlay_decay"]["should_decay"] is True
    assert payload["overlay_decay"]["reason"] == "raw_live_clear_overlay_fresh_overstates_after_drain"


def test_ingestion_storage_control_prefers_live_backpressure_over_stale_governor_watermarks(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 315,
            "pending_lines_total": 389,
            "pending_lines_deferred": 74,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 1,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 13.0,
            "oldest_age_threshold_seconds": 240.0,
            "overload": False,
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "cycle_started_utc": (now - timedelta(minutes=5)).isoformat(),
            "merged_rows_this_cycle": 1000,
        },
    )
    _write_json(health / "sql_link_service_latest.json", {"sqlite_wal_size_gb": 0.0})
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": True,
            "hard_gates": {"ingestion_backpressure_overload": True},
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": False},
            "ingestion_pressure": {"severe_backpressure_overload": False},
        },
    )
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "critical_backpressure",
            "sql_primary_db": {"route_drift": False},
            "queue_watermarks": {
                "overall_status": "blocked",
                "lanes": {"core": {"pending_lines": 64458}},
                "breaches": {"hard": ["core"]},
            },
            "writer_shedding": {"active": True, "level": "protect_core"},
            "throttle_controls": {"deferred_files_budget": 0, "cold_files_budget": 0},
        },
    )
    _write_json(health / "external_backlog_drain_latest.json", {"overall_status": "ready", "recommended_now": False})
    _write_json(
        health / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "files_discovered": 4, "sqlite": {"invalid": 0}},
    )

    payload = src.build_payload(tmp_path, now_utc=datetime(2026, 5, 1, 2, 15, tzinfo=timezone.utc))

    assert payload["queue_watermarks_source"] == "live_backpressure"
    assert payload["queue_watermarks"]["overall_status"] == "ready"
    assert payload["queue_watermarks"]["lanes"]["core"]["pending_lines"] == 315
    assert payload["bounded_recovery_contract"]["stale_hard_gate_suppressed"] == ["ingestion_backpressure_overload"]
    assert payload["bounded_recovery_contract"]["hard_gate_keys"] == []
    assert payload["overall_status"] == "ready"


def test_ingestion_storage_control_stabilizes_tiny_hot_queue_under_active_drain(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 586,
            "pending_lines_total": 586,
            "pending_lines_deferred": 0,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 0,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 19.4,
            "oldest_age_threshold_seconds": 240.0,
            "overload": False,
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "cycle_started_utc": (now - timedelta(seconds=25)).isoformat(),
            "merged_rows_this_cycle": 1,
        },
    )
    _write_json(health / "sql_link_service_latest.json", {"sqlite_wal_size_gb": 0.0})
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": True,
            "recommended_operating_mode": "maintenance_only",
            "hard_gates": {"ingestion_backpressure_overload": True},
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": True},
            "ingestion_pressure": {"severe_backpressure_overload": True},
        },
    )
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "critical_backpressure",
            "sql_primary_db": {"route_drift": False},
            "queue_watermarks": {"overall_status": "blocked"},
            "writer_shedding": {"active": True, "level": "protect_core", "freeze_cold_lanes": True},
        },
    )
    _write_json(
        health / "external_backlog_drain_latest.json",
        {
            "overall_status": "drain_active",
            "recommended_now": True,
            "follow_through": {"progress_observed": False, "status": "handoff_requested"},
            "drain_delta": {"core_pending_lines": 0, "total_pending_lines": 0},
        },
    )
    _write_json(
        health / "storage_failback_sync_latest.json",
        {"route_verification": {"verification_state": "curated_ready", "ready_count": 3, "tracked_count": 3, "coverage_ratio": 1.0, "mismatches": []}},
    )
    _write_json(
        health / "storage_resilience_control_latest.json",
        {
            "overall_status": "ready",
            "resilience_score": 100,
            "restore_drill_fresh": True,
            "dual_root_ready": True,
            "warm_standby_ready": True,
            "unresolved_split_brain_conflicts": 0,
        },
    )
    _write_json(
        health / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "files_discovered": 2, "sqlite": {"invalid": 0}},
    )

    payload = src.build_payload(tmp_path, now_utc=now)

    assert payload["overall_status"] == "ready"
    assert payload["severity"] == "stable"
    assert payload["stabilization_contract"]["small_hot_queue_stable"] is True
    assert payload["stabilization_contract"]["drain_minutes_total_bounded"] is True
    assert payload["bounded_recovery_contract"]["stale_hard_gate_suppressed"] == ["ingestion_backpressure_overload"]
    assert payload["bounded_recovery_contract"]["stale_severe_backpressure_suppressed"] == ["severe_backpressure_overload"]
    assert payload["backpressure"]["estimated_total_drain_minutes"] == 15.0


def test_ingestion_storage_control_suppresses_stale_severe_gate_after_drain_clears(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 74,
            "pending_lines_total": 1980,
            "pending_lines_deferred": 1906,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 0,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 0.0,
            "oldest_age_threshold_seconds": 240.0,
            "overload": False,
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "cycle_started_utc": (now - timedelta(minutes=5)).isoformat(),
            "merged_rows_this_cycle": 0,
        },
    )
    _write_json(health / "sql_link_service_latest.json", {"sqlite_wal_size_gb": 0.0})
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": True,
            "recommended_operating_mode": "shadow_only",
            "hard_gates": {"ingestion_backpressure_overload": True},
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": False},
            "ingestion_pressure": {"severe_backpressure_overload": True},
        },
    )
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "critical_backpressure",
            "sql_primary_db": {"route_drift": False},
            "queue_watermarks": {"overall_status": "blocked"},
            "writer_shedding": {"active": True, "level": "protect_core", "freeze_cold_lanes": True},
        },
    )
    _write_json(
        health / "external_backlog_drain_latest.json",
        {
            "overall_status": "ready",
            "recommended_now": False,
            "follow_through": {"progress_observed": False, "status": "not_needed"},
            "drain_delta": {"core_pending_lines": 0, "total_pending_lines": 0},
        },
    )
    _write_json(
        health / "storage_failback_sync_latest.json",
        {"route_verification": {"verification_state": "curated_ready", "ready_count": 3, "tracked_count": 3, "coverage_ratio": 1.0, "mismatches": []}},
    )
    _write_json(
        health / "storage_resilience_control_latest.json",
        {
            "overall_status": "ready",
            "resilience_score": 100,
            "restore_drill_fresh": True,
            "dual_root_ready": True,
            "warm_standby_ready": True,
            "unresolved_split_brain_conflicts": 0,
        },
    )
    _write_json(
        health / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "files_discovered": 3, "sqlite": {"invalid": 0}},
    )

    payload = src.build_payload(tmp_path, now_utc=now)

    assert payload["severity"] == "stable"
    assert payload["overall_status"] == "ready"
    assert payload["queue_watermarks"]["overall_status"] == "ready"
    assert payload["stabilization_contract"]["small_hot_queue_stable"] is False
    assert payload["bounded_recovery_contract"]["stale_hard_gate_suppressed"] == ["ingestion_backpressure_overload"]
    assert payload["bounded_recovery_contract"]["stale_severe_backpressure_suppressed"] == [
        "severe_backpressure_overload",
        "measured_backpressure_clear_after_drain",
    ]
    assert payload["storage_efficiency_contract"]["overall_status"] == "ready"


def test_ingestion_storage_control_tolerates_incidental_side_lane_trickle(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 1058,
            "pending_lines_total": 1061,
            "pending_lines_deferred": 3,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 3,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 2.9,
            "oldest_age_threshold_seconds": 240.0,
            "overload": False,
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "cycle_started_utc": (now - timedelta(seconds=18)).isoformat(),
            "merged_rows_this_cycle": 0,
        },
    )
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": True,
            "hard_gates": {"ingestion_backpressure_overload": True},
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": True},
            "ingestion_pressure": {"severe_backpressure_overload": True},
        },
    )
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "critical_backpressure",
            "sql_primary_db": {"route_drift": False},
            "writer_shedding": {"active": True, "level": "protect_core", "freeze_cold_lanes": True},
        },
    )
    _write_json(
        health / "external_backlog_drain_latest.json",
        {
            "overall_status": "drain_active",
            "recommended_now": True,
            "follow_through": {"progress_observed": False, "status": "handoff_requested"},
        },
    )
    _write_json(
        health / "storage_failback_sync_latest.json",
        {"route_verification": {"verification_state": "curated_ready", "ready_count": 3, "tracked_count": 3, "coverage_ratio": 1.0, "mismatches": []}},
    )
    _write_json(
        health / "storage_resilience_control_latest.json",
        {
            "overall_status": "ready",
            "resilience_score": 100,
            "restore_drill_fresh": True,
            "dual_root_ready": True,
            "warm_standby_ready": True,
            "unresolved_split_brain_conflicts": 0,
        },
    )
    _write_json(
        health / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "files_discovered": 3, "sqlite": {"invalid": 0}},
    )

    payload = src.build_payload(tmp_path, now_utc=now)

    assert payload["overall_status"] == "ready"
    assert payload["severity"] == "stable"
    assert payload["stabilization_contract"]["small_hot_queue_stable"] is True
    assert payload["backpressure"]["estimated_total_drain_minutes"] == 15.0


def test_ingestion_storage_control_degrades_ready_state_when_storage_resilience_needs_work(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 300,
            "pending_lines_total": 1200,
            "pending_lines_deferred": 900,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 0,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 15.0,
            "oldest_age_threshold_seconds": 240.0,
            "overload": False,
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "cycle_started_utc": (now - timedelta(minutes=5)).isoformat(),
            "merged_rows_this_cycle": 90000,
        },
    )
    _write_json(health / "sql_link_service_latest.json", {"sqlite_wal_size_gb": 0.1})
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": False,
            "recommended_operating_mode": "live_cautious",
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": False},
            "ingestion_pressure": {"severe_backpressure_overload": False},
        },
    )
    _write_json(health / "storage_maintenance_latest.json", {"reason": "ok"})
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "steady_state",
            "sql_primary_db": {"route_drift": False},
            "queue_watermarks": {"overall_status": "ready"},
            "writer_shedding": {"active": False, "level": "normal"},
            "throttle_controls": {"deferred_files_budget": 2, "cold_files_budget": 1},
        },
    )
    _write_json(health / "external_backlog_drain_latest.json", {"overall_status": "ready", "recommended_now": False, "aged_candidate_files": 0})
    _write_json(health / "stale_artifact_sweeper_bot_latest.json", {"summary": {"candidate_files": 0, "staged_files": 0}})
    _write_json(health / "stale_artifact_reaper_bot_latest.json", {"summary": {"deleted_files": 0}})
    _write_json(health / "data_retention_latest.json", {"deleted": 0})
    _write_json(
        health / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "files_discovered": 4, "sqlite": {"invalid": 0}},
    )
    _write_json(
        health / "storage_resilience_control_latest.json",
        {
            "overall_status": "needs_work",
            "resilience_score": 62,
            "restore_drill_fresh": False,
            "dual_root_ready": True,
            "warm_standby_ready": True,
            "unresolved_split_brain_conflicts": 1,
        },
    )

    payload = src.build_payload(tmp_path, now_utc=datetime(2026, 4, 7, 18, 0, tzinfo=timezone.utc))

    assert payload["overall_status"] == "needs_work"
    assert payload["ok"] is False
    assert payload["storage_resilience"]["overall_status"] == "needs_work"
    assert payload["storage_resilience"]["unresolved_split_brain_conflicts"] == 1
    assert any("restore drill" in action for action in payload["top_actions"])


def test_ingestion_storage_control_ignores_non_storage_hard_gates(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 5000,
            "pending_lines_total": 30000,
            "pending_lines_deferred": 12000,
            "pending_lines_cold": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 30.0,
            "oldest_age_threshold_seconds": 240.0,
            "overload": False,
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "cycle_started_utc": (now - timedelta(minutes=3)).isoformat(),
            "merged_rows_this_cycle": 60000,
        },
    )
    _write_json(health / "sql_link_service_latest.json", {"sqlite_wal_size_gb": 0.2})
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": True,
            "hard_gates": {
                "collector_contracts": True,
                "ingestion_backpressure_overload": False,
                "priority_shard_storage": False,
                "sql_progress_stall": False,
                "sql_wal_pressure": False,
            },
            "recommended_operating_mode": "live_cautious",
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": False},
            "ingestion_pressure": {"severe_backpressure_overload": False},
        },
    )
    _write_json(health / "storage_maintenance_latest.json", {"reason": "ok"})
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "steady_state",
            "sql_primary_db": {"route_drift": False},
            "throttle_controls": {"deferred_files_budget": 4, "cold_files_budget": 1},
        },
    )
    _write_json(health / "external_backlog_drain_latest.json", {"overall_status": "idle", "recommended_now": False, "aged_candidate_files": 0})
    _write_json(health / "backlog_quarantine_bot_latest.json", {"overall_status": "idle", "candidate_files": 0, "moved_files": 0, "moved_pending_lines": 0})
    _write_json(health / "stale_artifact_sweeper_bot_latest.json", {"summary": {"candidate_files": 0, "staged_files": 0}})
    _write_json(health / "stale_artifact_reaper_bot_latest.json", {"summary": {"deleted_files": 0}})
    _write_json(health / "data_retention_latest.json", {"deleted": 0})
    _write_json(
        health / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "files_discovered": 6, "sqlite": {"invalid": 0}},
    )

    payload = src.build_payload(tmp_path, now_utc=now)

    assert payload["severity"] in {"stable", "elevated"}
    assert payload["overall_status"] == "ready"
    assert payload["ok"] is True


def test_ingestion_storage_control_marks_bounded_critical_pressure_as_recovering(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 24000,
            "pending_lines_total": 600000,
            "pending_lines_deferred": 480000,
            "pending_lines_cold": 120000,
            "pending_lines_support_telemetry": 0,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 5000,
            "oldest_pending_age_seconds": 60.0,
            "oldest_age_threshold_seconds": 240.0,
            "overload": False,
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "cycle_started_utc": (now - timedelta(minutes=10)).isoformat(),
            "merged_rows_this_cycle": 400000,
        },
    )
    _write_json(health / "sql_link_service_latest.json", {"sqlite_wal_size_gb": 1.1})
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": False,
            "recommended_operating_mode": "maintenance_only",
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": False},
            "ingestion_pressure": {"severe_backpressure_overload": False},
        },
    )
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "critical_backpressure",
            "sql_primary_db": {"route_drift": False},
            "writer_shedding": {"active": True, "level": "protect_core", "freeze_cold_lanes": True},
        },
    )
    _write_json(health / "storage_maintenance_latest.json", {"reason": "ok"})
    _write_json(health / "external_backlog_drain_latest.json", {"overall_status": "ready", "recommended_now": True, "aged_candidate_files": 0})
    _write_json(
        health / "storage_failback_sync_latest.json",
        {"route_verification": {"verification_state": "curated_ready", "ready_count": 3, "tracked_count": 3, "coverage_ratio": 1.0, "mismatches": []}},
    )
    _write_json(
        health / "storage_resilience_control_latest.json",
        {
            "overall_status": "ready",
            "resilience_score": 82,
            "restore_drill_fresh": True,
            "dual_root_ready": True,
            "warm_standby_ready": True,
            "unresolved_split_brain_conflicts": 0,
        },
    )
    _write_json(
        health / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "files_discovered": 8, "sqlite": {"invalid": 0}},
    )

    payload = src.build_payload(tmp_path, now_utc=datetime(2026, 4, 7, 21, 0, tzinfo=timezone.utc))

    assert payload["severity"] == "critical"
    assert payload["overall_status"] == "degraded"
    assert payload["recovery_state"] == "recovering_under_guard"
    assert payload["bounded_recovery_contract"]["active"] is True


def test_ingestion_storage_control_accepts_recoverable_hard_gates_when_drain_is_active(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 20995,
            "pending_lines_total": 21311,
            "pending_lines_deferred": 316,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 0,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 291.366,
            "oldest_age_threshold_seconds": 240.0,
            "overload": True,
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "cycle_started_utc": (now - timedelta(minutes=2)).isoformat(),
            "merged_rows_this_cycle": 0,
        },
    )
    _write_json(health / "sql_link_service_latest.json", {"sqlite_wal_size_gb": 0.0})
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": True,
            "recommended_operating_mode": "shadow_only",
            "hard_gates": {
                "ingestion_backpressure_overload": True,
                "collector_contracts": True,
                "priority_shard_storage": False,
                "sql_progress_stall": False,
                "sql_wal_pressure": False,
            },
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": True},
            "ingestion_pressure": {"severe_backpressure_overload": True},
        },
    )
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "critical_backpressure",
            "sql_primary_db": {"route_drift": False},
            "queue_watermarks": {"overall_status": "degraded"},
            "writer_shedding": {"active": True, "level": "protect_core", "freeze_cold_lanes": True, "throttle_deferred_lanes": True},
        },
    )
    _write_json(
        health / "external_backlog_drain_latest.json",
        {
            "overall_status": "drain_active",
            "recommended_now": True,
            "apply_requested": True,
            "aged_candidate_files": 1,
            "follow_through": {"progress_observed": False, "status": "handoff_requested"},
            "drain_delta": {"core_pending_lines": 4791, "total_pending_lines": 4781},
        },
    )
    _write_json(health / "storage_maintenance_latest.json", {"reason": "ok"})
    _write_json(
        health / "storage_failback_sync_latest.json",
        {"route_verification": {"verification_state": "curated_ready", "ready_count": 3, "tracked_count": 3, "coverage_ratio": 1.0, "mismatches": []}},
    )
    _write_json(
        health / "storage_resilience_control_latest.json",
        {
            "overall_status": "ready",
            "resilience_score": 80,
            "restore_drill_fresh": False,
            "dual_root_ready": True,
            "warm_standby_ready": True,
            "unresolved_split_brain_conflicts": 0,
        },
    )
    _write_json(
        health / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "files_discovered": 8, "sqlite": {"invalid": 0}},
    )

    payload = src.build_payload(tmp_path, now_utc=now)

    assert payload["severity"] == "critical"
    assert payload["overall_status"] == "degraded"
    assert payload["recovery_state"] == "recovering_under_guard"
    assert payload["bounded_recovery_contract"]["active"] is True
    assert payload["bounded_recovery_contract"]["recoverable_hard_gate_only"] is True
    assert payload["bounded_recovery_contract"]["drain_delta_core_lines"] == 4791


def test_ingestion_storage_control_accepts_guarded_blocked_queue_with_negative_drain_deltas(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 90197,
            "pending_lines_total": 1708861,
            "pending_lines_deferred": 1618666,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 0,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 312.0,
            "oldest_age_threshold_seconds": 240.0,
            "overload": True,
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "cycle_started_utc": (now - timedelta(minutes=5)).isoformat(),
            "merged_rows_this_cycle": 0,
        },
    )
    _write_json(health / "sql_link_service_latest.json", {"sqlite_wal_size_gb": 0.0})
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": True,
            "recommended_operating_mode": "shadow_only",
            "hard_gates": {
                "ingestion_backpressure_overload": True,
                "collector_contracts": True,
                "priority_shard_storage": False,
                "sql_progress_stall": False,
                "sql_wal_pressure": False,
            },
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": True},
            "ingestion_pressure": {"severe_backpressure_overload": True},
        },
    )
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "critical_backpressure",
            "sql_primary_db": {"route_drift": False},
            "queue_watermarks": {"overall_status": "blocked"},
            "writer_shedding": {"active": True, "level": "protect_core", "freeze_cold_lanes": True, "throttle_deferred_lanes": True},
        },
    )
    _write_json(
        health / "external_backlog_drain_latest.json",
        {
            "overall_status": "drain_active",
            "recommended_now": True,
            "apply_requested": True,
            "aged_candidate_files": 1,
            "follow_through": {"progress_observed": False, "status": "handoff_requested"},
            "drain_delta": {"core_pending_lines": -78, "total_pending_lines": -102},
        },
    )
    _write_json(health / "storage_maintenance_latest.json", {"reason": "ok"})
    _write_json(
        health / "storage_failback_sync_latest.json",
        {"route_verification": {"verification_state": "curated_ready", "ready_count": 3, "tracked_count": 3, "coverage_ratio": 1.0, "mismatches": []}},
    )
    _write_json(
        health / "storage_resilience_control_latest.json",
        {
            "overall_status": "ready",
            "resilience_score": 80,
            "restore_drill_fresh": True,
            "dual_root_ready": True,
            "warm_standby_ready": True,
            "unresolved_split_brain_conflicts": 0,
        },
    )
    _write_json(
        health / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "files_discovered": 8, "sqlite": {"invalid": 0}},
    )

    payload = src.build_payload(tmp_path, now_utc=now)

    assert payload["severity"] == "critical"
    assert payload["overall_status"] == "degraded"
    assert payload["recovery_state"] == "stabilized_recovery"
    assert payload["bounded_recovery_contract"]["active"] is True
    assert payload["bounded_recovery_contract"]["guarded_blocked_queue"] is True
    assert payload["bounded_recovery_contract"]["quality_ready"] is True
    assert payload["bounded_recovery_contract"]["drain_delta_signal_observed"] is True
    assert payload["bounded_recovery_contract"]["drain_delta_core_lines"] == -78
    assert payload["backpressure_quality_score"] >= 96.0
    assert payload["recovery_quality_score"] >= 96.0


def test_ingestion_storage_control_accepts_sql_progress_stall_as_recoverable_under_active_drain(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 107061,
            "pending_lines_total": 2100000,
            "pending_lines_deferred": 1992939,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 0,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 330.0,
            "oldest_age_threshold_seconds": 240.0,
            "overload": True,
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "cycle_started_utc": (now - timedelta(minutes=5)).isoformat(),
            "merged_rows_this_cycle": 0,
        },
    )
    _write_json(health / "sql_link_service_latest.json", {"sqlite_wal_size_gb": 0.0})
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": True,
            "recommended_operating_mode": "shadow_only",
            "hard_gates": {
                "ingestion_backpressure_overload": True,
                "sql_progress_stall": True,
                "sql_wal_pressure": True,
            },
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": True},
            "ingestion_pressure": {"severe_backpressure_overload": True},
        },
    )
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "critical_backpressure",
            "sql_primary_db": {"route_drift": False},
            "queue_watermarks": {"overall_status": "blocked"},
            "writer_shedding": {"active": True, "level": "protect_core", "freeze_cold_lanes": True, "throttle_deferred_lanes": True},
        },
    )
    _write_json(
        health / "external_backlog_drain_latest.json",
        {
            "overall_status": "drain_active",
            "recommended_now": True,
            "apply_requested": True,
            "aged_candidate_files": 1,
            "follow_through": {"progress_observed": False, "status": "handoff_requested"},
            "drain_delta": {"core_pending_lines": -7743, "total_pending_lines": -8930},
        },
    )
    _write_json(health / "storage_maintenance_latest.json", {"reason": "ok"})
    _write_json(
        health / "storage_failback_sync_latest.json",
        {"route_verification": {"verification_state": "curated_ready", "ready_count": 3, "tracked_count": 3, "coverage_ratio": 1.0, "mismatches": []}},
    )
    _write_json(
        health / "storage_resilience_control_latest.json",
        {
            "overall_status": "ready",
            "resilience_score": 100,
            "restore_drill_fresh": True,
            "dual_root_ready": True,
            "warm_standby_ready": True,
            "unresolved_split_brain_conflicts": 0,
        },
    )
    _write_json(
        health / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "files_discovered": 8, "sqlite": {"invalid": 0}},
    )

    payload = src.build_payload(tmp_path, now_utc=now)

    assert payload["overall_status"] == "degraded"
    assert payload["recovery_state"] == "stabilized_recovery"
    assert payload["bounded_recovery_contract"]["active"] is True
    assert payload["bounded_recovery_contract"]["recoverable_hard_gate_only"] is True
    assert payload["bounded_recovery_contract"]["quality_ready"] is True
    assert payload["backpressure_quality_score"] >= 96.0
    assert payload["recovery_quality_score"] >= 96.0


def test_ingestion_storage_control_uses_fresh_sql_ingestion_overlay_when_summary_undercounts(tmp_path: Path) -> None:
    now = datetime(2026, 5, 19, 18, 20, tzinfo=timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 5805,
            "pending_lines_total": 6806,
            "pending_lines_deferred": 1001,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 0,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 45.0,
            "oldest_age_threshold_seconds": 240.0,
            "overload": False,
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "cycle_started_utc": (now - timedelta(minutes=10)).isoformat(),
            "merged_rows_this_cycle": 60000,
        },
    )
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": False,
            "recommended_operating_mode": "live_cautious",
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": False},
            "ingestion_pressure": {"severe_backpressure_overload": False},
        },
    )
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "steady_state",
            "sql_primary_db": {"route_drift": False},
            "writer_shedding": {"active": False, "level": "normal"},
        },
    )
    _write_json(health / "external_backlog_drain_latest.json", {"overall_status": "idle", "recommended_now": False})
    _write_json(
        health / "jsonl_sql_ingestion_health_trading_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "files_discovered": 2,
            "sqlite": {
                "inserted": 8810,
                "invalid": 0,
                "oversize_payloads": 0,
                "ops_write_failures": 0,
                "pending_lines": 134199,
                "oldest_uningested_age_seconds": 558.97,
                "files_with_pending": 2,
                "top_pending_files": [
                    {
                        "source_rel": "decisions/paper/trade_decisions_20260519.jsonl",
                        "stream": "decisions",
                        "storage_temperature": "hot",
                        "ingestion_lane": "hot_lane",
                        "pending_lines": 92520,
                        "oldest_pending_age_seconds": 535.049,
                        "total_lines": 154027,
                        "last_line": 61507,
                    },
                    {
                        "source_rel": "decisions/shadow_conservative_equities/trade_decisions_20260519.jsonl",
                        "stream": "decisions",
                        "storage_temperature": "hot",
                        "ingestion_lane": "hot_lane",
                        "pending_lines": 41679,
                        "oldest_pending_age_seconds": 452.991,
                        "total_lines": 50000,
                        "last_line": 8321,
                    },
                ],
            },
        },
    )

    payload = src.build_payload(tmp_path, now_utc=now)

    assert payload["backpressure"]["overlay_adjusted"] is True
    assert payload["backpressure"]["raw_live"]["total_pending_lines"] == 6806
    assert payload["backpressure"]["core_pending_lines"] == 134199
    assert payload["backpressure"]["total_pending_lines"] >= 134199
    assert payload["queue_watermarks_source"] == "live_backpressure+sql_ingestion_overlay"
    assert payload["sql_ingestion_pending_overlay"]["used_for_pressure"] is True
    assert payload["sql_ingestion_pending_overlay"]["fresh_source_count"] == 1
    assert payload["sql_ingestion_pending_overlay"]["top_pending_files"][0]["source_rel"] == "decisions/paper/trade_decisions_20260519.jsonl"
    assert payload["backlog_truth"]["authoritative_mode"] == "overlay_source_attributed"
    assert payload["backlog_truth"]["raw_live"]["total_pending_lines"] == 6806
    assert payload["backlog_truth"]["sql_overlay"]["used_for_pressure"] is True
    assert payload["stale_pending_locator"]["status"] == "attributed"
    assert payload["stale_pending_locator"]["oldest_sources"][0]["source_rel"] == "decisions/paper/trade_decisions_20260519.jsonl"
    assert payload["data_integrity"]["sql_overlay_pending_lines"] == 134199
    assert any("SQL ingestion overlay" in action for action in payload["top_actions"])


def test_ingestion_storage_control_ignores_stale_sql_ingestion_overlay(tmp_path: Path) -> None:
    now = datetime(2026, 5, 19, 18, 20, tzinfo=timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 300,
            "pending_lines_total": 1200,
            "pending_lines_deferred": 900,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 0,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 15.0,
            "oldest_age_threshold_seconds": 240.0,
            "overload": False,
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "cycle_started_utc": (now - timedelta(minutes=5)).isoformat(),
            "merged_rows_this_cycle": 90000,
        },
    )
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": False,
            "recommended_operating_mode": "live_cautious",
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": False},
            "ingestion_pressure": {"severe_backpressure_overload": False},
        },
    )
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "steady_state",
            "sql_primary_db": {"route_drift": False},
            "queue_watermarks": {"overall_status": "blocked"},
            "writer_shedding": {"active": False, "level": "normal"},
        },
    )
    _write_json(health / "external_backlog_drain_latest.json", {"overall_status": "idle", "recommended_now": False})
    _write_json(
        health / "jsonl_sql_ingestion_health_trading_latest.json",
        {
            "timestamp_utc": (now - timedelta(hours=2)).isoformat(),
            "files_discovered": 1,
            "sqlite": {
                "invalid": 0,
                "pending_lines": 200000,
                "oldest_uningested_age_seconds": 7200.0,
                "files_with_pending": 1,
                "top_pending_files": [
                    {
                        "source_rel": "decisions/paper/trade_decisions_20260519.jsonl",
                        "stream": "decisions",
                        "storage_temperature": "hot",
                        "ingestion_lane": "hot_lane",
                        "pending_lines": 200000,
                    }
                ],
            },
        },
    )

    payload = src.build_payload(tmp_path, now_utc=now)

    assert payload["backpressure"]["overlay_adjusted"] is False
    assert payload["backpressure"]["core_pending_lines"] == 300
    assert payload["backpressure"]["total_pending_lines"] == 1200
    assert payload["queue_watermarks_source"] == "live_backpressure"
    assert payload["sql_ingestion_pending_overlay"]["active"] is False
    assert payload["sql_ingestion_pending_overlay"]["fresh_source_count"] == 0
    assert payload["sql_ingestion_pending_overlay"]["stale_source_count"] == 1
    assert payload["backlog_truth"]["authoritative_mode"] == "raw_live"
    assert payload["backlog_truth"]["sql_overlay"]["used_for_pressure"] is False


def test_ingestion_storage_control_uses_fresh_overlay_to_reconcile_stale_raw_pressure_downward(tmp_path: Path) -> None:
    now = datetime(2026, 5, 22, 22, 40, tzinfo=timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 3863,
            "pending_lines_total": 251486,
            "pending_lines_deferred": 247623,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 6,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 0.013,
            "oldest_age_threshold_seconds": 240.0,
            "overload": False,
        },
    )
    _write_json(
        health / "jsonl_sql_ingestion_health_trading_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "sqlite": {
                "pending_lines": 1290,
                "oldest_uningested_age_seconds": 73.346,
                "files_with_pending": 2,
                "top_pending_files": [
                    {
                        "source_rel": "governance/channels/decision/intraday_aggressive_equities_schwab/decision_20260522.jsonl",
                        "stream": "decisions",
                        "storage_temperature": "hot",
                        "ingestion_lane": "nearline_lane",
                        "pending_lines": 716,
                        "oldest_pending_age_seconds": 73.267,
                    },
                    {
                        "source_rel": "governance/channels/decision/conservative_equities_schwab/decision_20260522.jsonl",
                        "stream": "decisions",
                        "storage_temperature": "hot",
                        "ingestion_lane": "nearline_lane",
                        "pending_lines": 574,
                        "oldest_pending_age_seconds": 38.329,
                    },
                ],
            },
        },
    )

    payload = src.build_payload(tmp_path, now_utc=now)

    assert payload["backpressure"]["overlay_adjusted"] is True
    assert payload["backpressure"]["raw_live"]["total_pending_lines"] == 251486
    assert payload["backpressure"]["total_pending_lines"] == 1290
    assert payload["backpressure"]["deferred_pending_lines"] == 0
    assert payload["sql_ingestion_pending_overlay"]["used_for_pressure"] is True
    assert payload["sql_ingestion_pending_overlay"]["reconciled_downward_for_pressure"] is True
    assert payload["backlog_truth"]["authoritative_mode"] == "overlay_fresh_shard_level"


def test_ingestion_storage_control_reconciles_stale_raw_core_when_focused_empty_overlay_covers_top_files(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 6, 30, 3, 25, tzinfo=timezone.utc)
    health = tmp_path / "governance" / "health"
    focused_paths = [
        "governance/events/auth_events_20260630.jsonl",
        "governance/events/auth_events_20260629.jsonl",
        "governance/events/write_failures_20260630.jsonl",
        "governance/events/write_failures_20260629.jsonl",
        "governance/events/premarket_token_guard_20260630.jsonl",
        "governance/events/premarket_token_guard_20260629.jsonl",
    ]
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "timestamp_utc": (now - timedelta(minutes=20)).isoformat(),
            "pending_lines": 6503,
            "pending_lines_total": 10558,
            "pending_lines_deferred": 4055,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 0,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 11117.908,
            "oldest_age_threshold_seconds": 240.0,
            "overload": True,
            "top_pending_files": [
                {"source_rel": focused_paths[0], "pending_lines": 2613, "oldest_pending_age_seconds": 385.796},
                {"source_rel": focused_paths[1], "pending_lines": 1755, "oldest_pending_age_seconds": 11117.908},
                {"source_rel": focused_paths[2], "pending_lines": 1072, "oldest_pending_age_seconds": 402.35},
                {"source_rel": focused_paths[3], "pending_lines": 818, "oldest_pending_age_seconds": 11117.899},
                {"source_rel": focused_paths[4], "pending_lines": 111, "oldest_pending_age_seconds": 63.586},
                {"source_rel": focused_paths[5], "pending_lines": 56, "oldest_pending_age_seconds": 11196.201},
                {
                    "source_rel": "governance/events/paper_execution_guard_20260629.jsonl",
                    "pending_lines": 28,
                    "oldest_pending_age_seconds": 16871.689,
                },
            ],
        },
    )
    _write_json(
        health / "jsonl_sql_ingestion_health_governance_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "filters": {"path_contains": focused_paths},
            "sqlite": {
                "pending_lines": 0,
                "oldest_uningested_age_seconds": 0.0,
                "files_with_pending": 0,
                "top_pending_files": [],
            },
        },
    )

    payload = src.build_payload(tmp_path, now_utc=now)

    assert payload["backpressure"]["overlay_adjusted"] is True
    assert payload["backpressure"]["raw_live"]["core_pending_lines"] == 6503
    assert payload["backpressure"]["core_pending_lines"] == 0
    assert payload["backpressure"]["deferred_pending_lines"] == 4055
    assert payload["backpressure"]["total_pending_lines"] == 4055
    assert payload["backpressure"]["oldest_pending_age_seconds"] == 0.0
    assert payload["sql_ingestion_pending_overlay"]["reconciled_downward_for_pressure"] is True
    assert payload["sql_ingestion_pending_overlay"]["reconciled_focused_raw_pressure"] is True
    assert payload["sql_ingestion_pending_overlay"]["fresh_overlay_raw_top_coverage"]["covers_raw_pressure"] is True
    assert payload["sql_ingestion_pending_overlay"]["fresh_overlay_raw_top_coverage"]["uncovered_raw_top_pending_lines"] == 28


def test_ingestion_storage_control_reconciles_stale_raw_age_when_locator_is_clear(tmp_path: Path) -> None:
    now = datetime(2026, 6, 5, 6, 15, tzinfo=timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 6256,
            "pending_lines_total": 6256,
            "pending_lines_deferred": 0,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 0,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 525.367,
            "oldest_age_threshold_seconds": 240.0,
            "overload": True,
        },
    )
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": False,
            "recommended_operating_mode": "live_cautious",
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": False},
            "ingestion_pressure": {"severe_backpressure_overload": False},
        },
    )
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "steady_state",
            "sql_primary_db": {"route_drift": False},
            "queue_watermarks": {"overall_status": "ready"},
            "writer_shedding": {"active": False, "level": "normal"},
        },
    )
    _write_json(health / "external_backlog_drain_latest.json", {"overall_status": "idle", "recommended_now": False, "aged_candidate_files": 0})
    _write_json(
        health / "jsonl_sql_ingestion_health_crypto_trading_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "sqlite": {
                "pending_lines": 4131,
                "oldest_uningested_age_seconds": 15.958,
                "files_with_pending": 1,
                "top_pending_files": [
                    {
                        "source_rel": "governance/channels/decision/default_crypto_schwab/decision_20260605.jsonl",
                        "stream": "decisions",
                        "storage_temperature": "hot",
                        "ingestion_lane": "hot_lane",
                        "pending_lines": 4131,
                        "oldest_pending_age_seconds": 15.958,
                        "total_lines": 12414,
                        "last_line": 8283,
                    }
                ],
            },
        },
    )

    payload = src.build_payload(tmp_path, now_utc=now)

    assert payload["stale_pending_locator"]["status"] == "clear"
    assert payload["sql_ingestion_pending_overlay"]["used_for_pressure"] is False
    assert payload["sql_ingestion_pending_overlay"]["reconciled_stale_age_for_pressure"] is True
    assert payload["backpressure"]["oldest_pending_age_seconds"] == 15.958
    assert payload["backpressure"]["raw_live"]["oldest_pending_age_seconds"] == 15.958
    assert payload["backpressure"]["raw_live"]["raw_oldest_pending_age_seconds"] == 525.367
    assert payload["backpressure"]["raw_live"]["age_reconciled_from_stale_locator"] is True
    assert payload["backlog_truth"]["raw_live"]["oldest_pending_age_seconds"] == 15.958
    assert payload["bounded_recovery_contract"]["stale_backpressure_overload_suppressed"] == [
        "ingestion_backpressure_latest.overload"
    ]
    assert payload["pressure_index"] < 0.75
    assert payload["severity"] == "stable"
    assert payload["overall_status"] == "ready"
    assert payload["raw_live_expansion_contract"]["ratios"]["oldest_age"] < 1.0


def test_ingestion_storage_control_suppresses_tiny_aged_candidates_when_clear_overlay_has_no_sources(tmp_path: Path) -> None:
    now = datetime(2026, 7, 14, 13, 50, tzinfo=timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "pending_lines": 235,
            "pending_lines_total": 383,
            "pending_lines_deferred": 148,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 0,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 1099749.837,
            "oldest_age_threshold_seconds": 240.0,
            "overload": True,
            "top_pending_files": [
                {
                    "source_rel": "governance/events/write_failures_20260701.jsonl",
                    "pending_lines": 155,
                    "oldest_pending_age_seconds": 1099749.837,
                },
                {
                    "source_rel": "governance/events/live_execution_guard_20260701.jsonl",
                    "pending_lines": 57,
                    "oldest_pending_age_seconds": 1086940.826,
                },
            ],
        },
    )
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": False,
            "recommended_operating_mode": "live_cautious",
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": False},
            "ingestion_pressure": {"severe_backpressure_overload": False},
        },
    )
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "critical_backpressure",
            "sql_primary_db": {"route_drift": False},
            "queue_watermarks": {"overall_status": "ready"},
            "writer_shedding": {"active": False, "level": "normal"},
        },
    )
    _write_json(
        health / "external_backlog_drain_latest.json",
        {
            "overall_status": "drain_active",
            "recommended_now": True,
            "aged_candidate_files": 7,
        },
    )
    _write_json(
        health / "jsonl_sql_ingestion_health_governance_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "sqlite": {
                "pending_lines": 0,
                "oldest_uningested_age_seconds": 0.0,
                "files_with_pending": 0,
                "top_pending_files": [],
                "inserted": 84,
                "invalid": 0,
                "oversize_payloads": 0,
                "ops_write_failures": 0,
            },
        },
    )

    payload = src.build_payload(tmp_path, now_utc=now)

    assert payload["stale_pending_locator"]["status"] == "clear"
    assert payload["sql_ingestion_pending_overlay"]["clear_overlay_reconciled_stale_raw_age"] is True
    assert payload["backpressure"]["oldest_pending_age_seconds"] == 0.0
    assert payload["backpressure"]["raw_live"]["raw_oldest_pending_age_seconds"] == 1099749.837
    assert payload["backpressure"]["raw_live"]["age_reconciliation_source"] == "fresh_clear_sql_overlay"
    assert payload["storage"]["raw_aged_backlog_candidate_files"] == 7
    assert payload["storage"]["aged_backlog_candidate_files"] == 0
    assert payload["storage"]["aged_backlog_candidate_files_suppressed_by_clear_overlay"] is True
    assert "stale_old_pending_work" not in payload["backlog_relief_contract"]["active_issue_ids"]
    assert payload["raw_live_expansion_contract"]["active"] is False
    assert payload["pressure_index"] < 0.75
    assert payload["severity"] == "stable"
    assert payload["overall_status"] == "ready"


def test_ingestion_storage_control_reconciles_stale_raw_age_when_fresh_overlay_is_empty(tmp_path: Path) -> None:
    now = datetime(2026, 6, 11, 3, 30, tzinfo=timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "timestamp_utc": (now - timedelta(days=5)).isoformat(),
            "pending_lines": 1470,
            "pending_lines_total": 1470,
            "pending_lines_deferred": 0,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 0,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 115725.922,
            "oldest_age_threshold_seconds": 240.0,
            "overload": True,
            "line_estimation": {
                "sparse_large_line_active": True,
                "sparse_large_line_pending_lines": 1470,
                "sparse_large_line_pending_bytes": 84591899,
            },
        },
    )
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": False,
            "recommended_operating_mode": "live_cautious",
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": False},
            "ingestion_pressure": {"severe_backpressure_overload": False},
        },
    )
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "steady_state",
            "sql_primary_db": {"route_drift": False},
            "queue_watermarks": {"overall_status": "ready"},
            "writer_shedding": {"active": False, "level": "normal"},
        },
    )
    _write_json(health / "external_backlog_drain_latest.json", {"overall_status": "idle", "recommended_now": False, "aged_candidate_files": 0})
    _write_json(
        health / "jsonl_sql_ingestion_health_trading_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "sqlite": {
                "pending_lines": 0,
                "oldest_uningested_age_seconds": 0.0,
                "files_with_pending": 0,
                "top_pending_files": [],
            },
        },
    )

    payload = src.build_payload(tmp_path, now_utc=now)

    assert payload["stale_pending_locator"]["status"] == "clear"
    assert payload["sql_ingestion_pending_overlay"]["fresh_source_count"] == 1
    assert payload["sql_ingestion_pending_overlay"]["total_pending_lines"] == 0
    assert payload["sql_ingestion_pending_overlay"]["reconciled_stale_age_for_pressure"] is True
    assert payload["sql_ingestion_pending_overlay"]["empty_overlay_reconciled_stale_raw_age"] is True
    assert payload["backpressure"]["oldest_pending_age_seconds"] == 0.0
    assert payload["backpressure"]["raw_live"]["raw_oldest_pending_age_seconds"] == 115725.922
    assert payload["backpressure"]["raw_live"]["age_reconciliation_source"] == "fresh_empty_sql_overlay"
    assert payload["backpressure"]["raw_live"]["age_reconciled_from_stale_locator"] is True
    assert payload["backpressure"]["raw_live"]["artifact_stale_for_overlay_reconciliation"] is True
    assert payload["backpressure"]["total_pending_lines"] == 1470
    assert payload["pressure_index"] < 0.75
    assert payload["severity"] == "stable"
    assert payload["overall_status"] == "ready"


def test_ingestion_storage_control_decays_unattributed_overlay_pressure(tmp_path: Path) -> None:
    now = datetime(2026, 5, 20, 16, 0, tzinfo=timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 200,
            "pending_lines_total": 400,
            "pending_lines_deferred": 200,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 0,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 60.0,
            "oldest_age_threshold_seconds": 240.0,
            "overload": False,
        },
    )
    _write_json(health / "sql_link_service_progress_latest.json", {"cycle_started_utc": (now - timedelta(minutes=2)).isoformat(), "merged_rows_this_cycle": 1000})
    _write_json(health / "ingestion_storage_governor_latest.json", {"writer_shedding": {"active": False}, "sql_primary_db": {"route_drift": False}})
    _write_json(health / "health_gates_latest.json", {"hard_gate_triggered": False})
    _write_json(health / "external_backlog_drain_latest.json", {"recommended_now": False})
    _write_json(
        health / "jsonl_sql_ingestion_health_trading_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "sqlite": {
                "pending_lines": 120000,
                "oldest_uningested_age_seconds": 900.0,
                "files_with_pending": 4,
                "top_pending_files": [],
            },
        },
    )

    payload = src.build_payload(tmp_path, now_utc=now)

    assert payload["overlay_decay"]["should_decay"] is True
    assert payload["backpressure"]["overlay_adjusted"] is False
    assert payload["backlog_truth"]["authoritative_mode"] == "raw_live_overlay_decayed"
    assert payload["backpressure"]["total_pending_lines"] == 400


def test_ingestion_storage_control_surfaces_sparse_large_line_action(tmp_path: Path) -> None:
    now = datetime(2026, 5, 19, 21, 45, tzinfo=timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 42000,
            "pending_lines_total": 42000,
            "pending_lines_deferred": 0,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 0,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 900.0,
            "oldest_age_threshold_seconds": 240.0,
            "overload": True,
            "line_estimation": {
                "sparse_large_line_active": True,
                "sparse_large_line_files": 3,
                "sparse_large_line_pending_lines": 12000,
                "sparse_large_line_bytes": 5_000_000_000,
            },
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "cycle_started_utc": (now - timedelta(minutes=5)).isoformat(),
            "merged_rows_this_cycle": 10000,
        },
    )
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": False,
            "recommended_operating_mode": "live_cautious",
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": False},
            "ingestion_pressure": {"severe_backpressure_overload": True},
        },
    )
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "critical_backpressure",
            "sql_primary_db": {"route_drift": False},
            "writer_shedding": {"active": True, "level": "protect_core"},
        },
    )
    _write_json(health / "external_backlog_drain_latest.json", {"overall_status": "idle", "recommended_now": False})

    payload = src.build_payload(tmp_path, now_utc=now)

    assert payload["backpressure"]["raw_live"]["line_estimation"]["sparse_large_line_active"] is True
    assert any("sparse-large-line decision drainer profile" in action for action in payload["top_actions"])


def test_ingestion_storage_control_builds_expansion_ready_backlog_relief_contract(tmp_path: Path) -> None:
    now = datetime(2026, 5, 20, 13, 0, tzinfo=timezone.utc)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 42000,
            "pending_lines_total": 53000,
            "pending_lines_deferred": 9000,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 2000,
            "pending_lines_stale_stage": 0,
            "pending_lines_threshold": 15000,
            "oldest_pending_age_seconds": 3600.0,
            "oldest_age_threshold_seconds": 240.0,
            "overload": True,
            "line_estimation": {
                "sparse_large_line_active": True,
                "sparse_large_line_files": 2,
                "sparse_large_line_pending_lines": 8000,
                "sparse_large_line_pending_bytes": 1_200_000_000,
            },
        },
    )
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "cycle_started_utc": (now - timedelta(minutes=30)).isoformat(),
            "merged_rows_this_cycle": 12000,
        },
    )
    _write_json(health / "sql_link_service_latest.json", {"sqlite_wal_size_gb": 0.75})
    _write_json(
        health / "ingestion_storage_governor_latest.json",
        {
            "profile": "critical_backpressure",
            "sql_primary_db": {"route_drift": False},
            "writer_shedding": {"active": True, "level": "protect_core"},
        },
    )
    _write_json(
        health / "external_backlog_drain_latest.json",
        {
            "overall_status": "ready",
            "recommended_now": True,
            "aged_candidate_files": 3,
            "off_hours_window": {"active": True},
        },
    )
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": True,
            "storage_pressure": {"retention_debt_gb": 0.0, "severe_backpressure_overload": True},
            "ingestion_pressure": {"severe_backpressure_overload": True},
        },
    )

    payload = src.build_payload(tmp_path, now_utc=now)

    contract = payload["backlog_relief_contract"]
    assert contract["active"] is True
    assert set(contract["active_issue_ids"]) == {
        "single_writer_merge_speed",
        "storage_write_latency",
        "sparse_huge_jsonl_files",
        "intake_outpaces_drain",
        "raw_live_expansion_headroom",
        "stale_old_pending_work",
    }
    assert payload["raw_live_expansion_contract"]["active"] is True
    assert contract["raw_live_expansion_headroom"]["active"] is True
    assert contract["control_env_recommendations"]["SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE"] == "150"
    assert contract["control_env_recommendations"]["RAW_LIVE_EXPANSION_GUARD_ACTIVE"] == "1"
    assert contract["control_env_recommendations"]["SQL_LINK_SERVICE_COLD_STAGE_YIELDS_TO_RAW_LIVE"] == "1"
    assert contract["control_env_recommendations"]["INGEST_MAX_BYTES_PER_FILE"] == str(128 * 1024 * 1024)
    assert contract["control_env_recommendations"]["BACKLOG_PCORE_ALLOCATION_ACTIVE"] == "1"
    assert contract["control_env_recommendations"]["BACKLOG_DRAIN_SINGLE_WRITER_ONLY"] == "1"
    assert contract["control_env_recommendations"]["TRAINING_PCORE_ALLOWED_WHEN_BACKLOG_GREEN"] == "1"
    assert float(contract["control_env_recommendations"]["BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO"]) <= 0.30
    p_core = contract["p_core_backlog_allocation_contract"]
    assert p_core["policy"] == "p_core_preprocess_single_sql_writer"
    assert p_core["sqlite_writer_count"] == 1
    assert p_core["training_pcore_gate"]["small_targeted_training_allowed_now"] is False
    assert p_core["catch_up_wave_controller"]["max_waves"] == 6
    assert any("bounded catch-up waves" in action for action in payload["top_actions"])


def test_raw_live_expansion_headroom_contract_marks_warm_raw_live_as_limited() -> None:
    contract = src._raw_live_expansion_headroom_contract(
        raw_live_backpressure={
            "core_pending_lines": 6500,
            "total_pending_lines": 7200,
            "oldest_pending_age_seconds": 120.0,
        },
        pending_threshold=15000,
        age_threshold_seconds=240.0,
        core_target=5000,
    )

    assert contract["active"] is True
    assert contract["expansion_ready"] is False
    assert contract["grade"] in {"A", "B"}
    assert contract["control_env"]["RAW_LIVE_EXPANSION_GUARD_ACTIVE"] == "1"
    assert contract["control_env"]["BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO"] == "0.16"


def test_raw_live_expansion_headroom_contract_allows_bigger_expansion_when_cool() -> None:
    contract = src._raw_live_expansion_headroom_contract(
        raw_live_backpressure={
            "core_pending_lines": 1500,
            "total_pending_lines": 2400,
            "oldest_pending_age_seconds": 24.0,
        },
        pending_threshold=15000,
        age_threshold_seconds=240.0,
        core_target=5000,
    )

    assert contract["active"] is False
    assert contract["expansion_ready"] is True
    assert contract["grade"] == "A+"
    assert contract["expansion_tier"] == "ready_for_bigger_expansion"
    assert contract["control_env"]["RAW_LIVE_EXPANSION_READY"] == "1"


def test_backlog_relief_ignores_tiny_sparse_tail_for_training_gate() -> None:
    contract = src._backlog_relief_contract(
        core_pending_lines=125,
        total_pending_lines=352,
        deferred_pending_lines=227,
        cold_pending_lines=0,
        support_pending_lines=6,
        stale_stage_pending_lines=0,
        oldest_age_seconds=0.0,
        age_threshold_seconds=240.0,
        pending_threshold=15000,
        drain_minutes_total=0.12,
        target_total_drain_minutes=30.0,
        throughput_rows_per_second=50.0,
        merged_rows_this_cycle=14027,
        line_estimation={
            "sparse_large_line_active": True,
            "sparse_large_line_files": 1,
            "sparse_large_line_pending_lines": 125,
            "sparse_large_line_pending_bytes": 1_856_299,
        },
        sql_pending_overlay={},
        sql_service={},
        route_drift=False,
        writer_shedding_active=False,
        aged_candidate_files=0,
    )

    sparse_issue = next(row for row in contract["issues"] if row["id"] == "sparse_huge_jsonl_files")
    assert sparse_issue["active"] is False
    assert sparse_issue["evidence"]["sparse_large_line_detected"] is True
    assert contract["active"] is False
    assert contract["overall_grade"] == "A+"
    assert contract["p_core_backlog_allocation_contract"]["training_pcore_gate"]["small_targeted_training_allowed_now"] is True


def test_p_core_burst_intelligence_uses_seven_when_host_is_deep_green(monkeypatch) -> None:
    monkeypatch.setenv("BACKLOG_PCORE_TARGET", "8")
    monkeypatch.delenv("BACKLOG_PCORE_FOREGROUND_RESERVE", raising=False)
    monkeypatch.delenv("BACKLOG_PCORE_PREPROCESS_WORKERS_OVERRIDE", raising=False)
    contract = src._p_core_backlog_allocation_contract(
        active_issue_ids=["single_writer_merge_speed", "sparse_huge_jsonl_files"],
        core_pending_lines=26000,
        total_pending_lines=42000,
        core_target=5000,
        total_target=15000,
        oldest_age_seconds=1200.0,
        age_threshold_seconds=240.0,
        sparse_active=True,
        sparse_pending_bytes=1_000_000_000,
        host_context={
            "off_hours_active": True,
            "runtime_throttle": {
                "host_saturation_score": 35.0,
                "compute_pressure_level": "normal",
                "memory_pressure_level": "normal",
                "throttle_profile": "soft_cap",
            },
            "resource_guard": {"creative_session_level": "idle"},
        },
    )

    assert contract["preprocess_worker_budget"] == 7
    assert contract["p_core_burst_intelligence"]["mode"] == "full_p_core_budget_7_plus_primary_writer"
    assert contract["p_core_burst_intelligence"]["seventh_core_burst"]["allowed"] is True
    assert contract["control_env"]["SQL_LINK_SERVICE_PREPROCESS_WORKERS"] == "7"


def test_p_core_burst_intelligence_caps_after_recent_storage_eject(tmp_path: Path, monkeypatch) -> None:
    log_path = tmp_path / "storage_eject_guard.log"
    now = datetime.now(timezone.utc)
    log_path.write_text(
        f"[{now.isoformat().replace('+00:00', 'Z')}] disk disappeared mountRoot=/Volumes/BOT_LOGS volumeBSD=disk5s1 wholeBSD=disk5 mode=external\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("STORAGE_EJECT_GUARD_LOG", str(log_path))
    monkeypatch.setenv("BACKLOG_PCORE_TARGET", "8")
    monkeypatch.delenv("BACKLOG_PCORE_FOREGROUND_RESERVE", raising=False)
    monkeypatch.delenv("BACKLOG_PCORE_PREPROCESS_WORKERS_OVERRIDE", raising=False)

    contract = src._p_core_backlog_allocation_contract(
        active_issue_ids=["single_writer_merge_speed", "sparse_huge_jsonl_files"],
        core_pending_lines=26000,
        total_pending_lines=42000,
        core_target=5000,
        total_target=15000,
        oldest_age_seconds=1200.0,
        age_threshold_seconds=240.0,
        sparse_active=True,
        sparse_pending_bytes=1_000_000_000,
        host_context={
            "off_hours_active": True,
            "runtime_throttle": {
                "host_saturation_score": 35.0,
                "compute_pressure_level": "normal",
                "memory_pressure_level": "normal",
                "throttle_profile": "soft_cap",
            },
            "resource_guard": {"creative_session_level": "idle"},
        },
    )

    burst = contract["p_core_burst_intelligence"]
    assert contract["preprocess_worker_budget"] == 3
    assert burst["mode"] == "storage_eject_cooldown_3"
    assert burst["storage_eject_cooldown"]["active"] is True
    assert burst["storage_eject_cooldown"]["previous_selected_workers"] == 7


def test_p_core_burst_intelligence_uses_four_worker_protect_live_probe_for_extreme_backlog(monkeypatch) -> None:
    monkeypatch.setenv("BACKLOG_PCORE_TARGET", "8")
    monkeypatch.delenv("BACKLOG_PCORE_FOREGROUND_RESERVE", raising=False)
    monkeypatch.delenv("BACKLOG_PCORE_PREPROCESS_WORKERS_OVERRIDE", raising=False)
    contract = src._p_core_backlog_allocation_contract(
        active_issue_ids=["single_writer_merge_speed", "sparse_huge_jsonl_files"],
        core_pending_lines=3_000_000,
        total_pending_lines=5_000_000,
        core_target=5000,
        total_target=15000,
        oldest_age_seconds=14_000.0,
        age_threshold_seconds=240.0,
        sparse_active=True,
        sparse_pending_bytes=1_000_000_000,
        host_context={
            "runtime_throttle": {
                "host_saturation_score": 55.0,
                "compute_pressure_level": "elevated",
                "memory_pressure_level": "normal",
                "memory_pressure_kind": "none",
                "throttle_profile": "protect_live",
                "swap_used_gb": 1.8,
            },
            "resource_guard": {"creative_session_level": "idle", "compressed_store_gb": 9.5},
            "computer_task": {"primary_task": "backlog_drain"},
        },
    )

    assert contract["preprocess_worker_budget"] == 4
    assert contract["p_core_burst_intelligence"]["mode"] == "protect_live_backlog_probe_4"
    assert contract["p_core_burst_intelligence"]["protected_live_backlog_probe"]["wide_allowed"] is True
    assert contract["control_env"]["SQL_LINK_SERVICE_PREPROCESS_WORKERS"] == "4"


def test_p_core_burst_intelligence_keeps_three_worker_probe_under_guarded_host_saturation(monkeypatch) -> None:
    monkeypatch.setenv("BACKLOG_PCORE_TARGET", "8")
    monkeypatch.delenv("BACKLOG_PCORE_FOREGROUND_RESERVE", raising=False)
    monkeypatch.delenv("BACKLOG_PCORE_PREPROCESS_WORKERS_OVERRIDE", raising=False)
    contract = src._p_core_backlog_allocation_contract(
        active_issue_ids=["single_writer_merge_speed", "sparse_huge_jsonl_files"],
        core_pending_lines=3_000_000,
        total_pending_lines=5_000_000,
        core_target=5000,
        total_target=15000,
        oldest_age_seconds=14_000.0,
        age_threshold_seconds=240.0,
        sparse_active=True,
        sparse_pending_bytes=1_000_000_000,
        host_context={
            "runtime_throttle": {
                "host_saturation_score": 63.0,
                "compute_pressure_level": "elevated",
                "memory_pressure_level": "normal",
                "memory_pressure_kind": "none",
                "throttle_profile": "protect_live",
                "swap_used_gb": 1.8,
            },
            "resource_guard": {"creative_session_level": "idle", "compressed_store_gb": 9.5},
            "computer_task": {"primary_task": "backlog_drain"},
        },
    )

    assert contract["preprocess_worker_budget"] == 3
    assert contract["p_core_burst_intelligence"]["mode"] == "protect_live_backlog_probe_3"
    assert contract["p_core_burst_intelligence"]["protected_live_backlog_probe"]["allowed"] is True
    assert contract["p_core_burst_intelligence"]["protected_live_backlog_probe"]["wide_allowed"] is False


def test_p_core_burst_intelligence_holds_three_workers_when_compute_is_high_but_memory_clear(monkeypatch) -> None:
    monkeypatch.setenv("BACKLOG_PCORE_TARGET", "8")
    monkeypatch.setenv("BACKLOG_PCORE_USER_APP_RESERVE_TARGET", "5")
    monkeypatch.delenv("BACKLOG_PCORE_FOREGROUND_RESERVE", raising=False)
    monkeypatch.delenv("BACKLOG_PCORE_PREPROCESS_WORKERS_OVERRIDE", raising=False)
    contract = src._p_core_backlog_allocation_contract(
        active_issue_ids=["single_writer_merge_speed", "sparse_huge_jsonl_files", "intake_outpaces_drain"],
        core_pending_lines=2_100_000,
        total_pending_lines=4_400_000,
        core_target=5000,
        total_target=15000,
        oldest_age_seconds=29_000.0,
        age_threshold_seconds=240.0,
        sparse_active=True,
        sparse_pending_bytes=1_300_000_000,
        host_context={
            "runtime_throttle": {
                "host_saturation_score": 59.52,
                "compute_pressure_level": "high",
                "memory_pressure_level": "normal",
                "memory_pressure_kind": "none",
                "throttle_profile": "protect_live",
                "swap_used_gb": 1.8,
            },
            "resource_guard": {"creative_session_level": "idle", "compressed_store_gb": 8.0},
            "computer_task": {"primary_task": "backlog_drain"},
        },
    )

    assert contract["preprocess_worker_budget"] == 3
    assert contract["p_core_burst_intelligence"]["mode"] == "protect_live_backlog_probe_3"
    assert contract["p_core_burst_intelligence"]["protected_live_backlog_probe"]["allowed"] is True
    assert contract["p_core_burst_intelligence"]["protected_live_backlog_probe"]["wide_allowed"] is False
    assert contract["control_env"]["SQL_LINK_SERVICE_SHARD_WRITER_LANES"] == "3"


def test_p_core_burst_intelligence_keeps_guarded_three_worker_pump_when_host_is_warm(monkeypatch) -> None:
    monkeypatch.setenv("BACKLOG_PCORE_TARGET", "8")
    monkeypatch.setenv("BACKLOG_PCORE_USER_APP_RESERVE_TARGET", "5")
    monkeypatch.delenv("BACKLOG_PCORE_FOREGROUND_RESERVE", raising=False)
    monkeypatch.delenv("BACKLOG_PCORE_PREPROCESS_WORKERS_OVERRIDE", raising=False)
    contract = src._p_core_backlog_allocation_contract(
        active_issue_ids=["single_writer_merge_speed", "sparse_huge_jsonl_files", "intake_outpaces_drain"],
        core_pending_lines=2_100_000,
        total_pending_lines=4_400_000,
        core_target=5000,
        total_target=15000,
        oldest_age_seconds=29_000.0,
        age_threshold_seconds=240.0,
        sparse_active=True,
        sparse_pending_bytes=1_300_000_000,
        host_context={
            "runtime_throttle": {
                "host_saturation_score": 72.0,
                "compute_pressure_level": "elevated",
                "memory_pressure_level": "normal",
                "memory_pressure_kind": "none",
                "throttle_profile": "protect_live",
                "swap_used_gb": 1.8,
            },
            "resource_guard": {"creative_session_level": "idle", "compressed_store_gb": 12.5},
            "computer_task": {"primary_task": "backlog_drain"},
        },
    )

    assert contract["preprocess_worker_budget"] == 3
    assert contract["p_core_burst_intelligence"]["mode"] == "guarded_backlog_probe_3"
    assert contract["p_core_burst_intelligence"]["guarded_backlog_probe"]["allowed"] is True
    assert contract["control_env"]["SQL_LINK_SERVICE_PREPROCESS_WORKERS"] == "3"


def test_p_core_burst_intelligence_loans_fourth_worker_when_compression_is_allocation_only(monkeypatch) -> None:
    monkeypatch.setenv("BACKLOG_PCORE_TARGET", "8")
    monkeypatch.setenv("BACKLOG_PCORE_USER_APP_RESERVE_TARGET", "5")
    monkeypatch.delenv("BACKLOG_PCORE_FOREGROUND_RESERVE", raising=False)
    monkeypatch.delenv("BACKLOG_PCORE_PREPROCESS_WORKERS_OVERRIDE", raising=False)
    contract = src._p_core_backlog_allocation_contract(
        active_issue_ids=["single_writer_merge_speed", "sparse_huge_jsonl_files", "intake_outpaces_drain"],
        core_pending_lines=337_223,
        total_pending_lines=595_964,
        core_target=5000,
        total_target=15000,
        oldest_age_seconds=60.0,
        age_threshold_seconds=240.0,
        sparse_active=True,
        sparse_pending_bytes=1_300_000_000,
        host_context={
            "runtime_throttle": {
                "host_saturation_score": 71.25,
                "compute_pressure_level": "elevated",
                "memory_pressure_level": "normal",
                "memory_pressure_kind": "none",
                "throttle_profile": "protect_live",
                "swap_used_gb": 1.4,
            },
            "resource_guard": {
                "creative_session_level": "idle",
                "compressed_store_gb": 17.7,
                "compressor_gb": 5.8,
                "pages_throttled": 0,
            },
            "computer_task": {"primary_task": "backlog_drain"},
        },
    )

    burst = contract["p_core_burst_intelligence"]
    assert contract["preprocess_worker_budget"] == 4
    assert burst["mode"] == "guarded_backlog_probe_4"
    assert burst["guarded_backlog_probe"]["wide_allowed"] is True
    assert burst["user_app_reserve"]["elastic_loan_allowed"] is True
    assert burst["inputs"]["compressed_pressure_gb"] == 5.8
    assert contract["control_env"]["SQL_LINK_SERVICE_PREPROCESS_WORKERS"] == "4"
    assert contract["accelerator_contract"]["mode"] == "p_core_sparse_catchup_wave_6"
    assert contract["accelerator_contract"]["catch_up_wave_controller"]["max_waves"] == 6
    assert contract["control_env"]["WRITER_CYCLE_MAX_CATCH_UP_WAVES"] == "6"
    assert contract["control_env"]["SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE"] == "150"


def test_p_core_burst_intelligence_honors_six_p_core_user_reserve_target(monkeypatch) -> None:
    monkeypatch.setenv("BACKLOG_PCORE_TARGET", "8")
    monkeypatch.setenv("BACKLOG_PCORE_USER_APP_RESERVE_TARGET", "6")
    monkeypatch.delenv("BACKLOG_PCORE_FOREGROUND_RESERVE", raising=False)
    monkeypatch.delenv("BACKLOG_PCORE_PREPROCESS_WORKERS_OVERRIDE", raising=False)
    contract = src._p_core_backlog_allocation_contract(
        active_issue_ids=["single_writer_merge_speed", "sparse_huge_jsonl_files"],
        core_pending_lines=3_000_000,
        total_pending_lines=5_000_000,
        core_target=5000,
        total_target=15000,
        oldest_age_seconds=14_000.0,
        age_threshold_seconds=240.0,
        sparse_active=True,
        sparse_pending_bytes=1_000_000_000,
        host_context={
            "runtime_throttle": {
                "host_saturation_score": 35.0,
                "compute_pressure_level": "normal",
                "memory_pressure_level": "normal",
                "memory_pressure_kind": "none",
                "throttle_profile": "soft_cap",
                "swap_used_gb": 1.0,
            },
            "resource_guard": {"creative_session_level": "idle", "compressed_store_gb": 4.0},
            "computer_task": {"primary_task": "backlog_drain"},
        },
    )

    assert contract["preprocess_worker_budget"] == 2
    assert contract["p_core_burst_intelligence"]["user_app_reserve"]["target_p_cores"] == 6
    assert contract["p_core_burst_intelligence"]["user_app_reserve"]["worker_cap"] == 2
    assert contract["control_env"]["BACKLOG_PCORE_USER_APP_RESERVE_TARGET"] == "6"


def test_p_core_burst_intelligence_protects_creative_work(monkeypatch) -> None:
    monkeypatch.setenv("BACKLOG_PCORE_TARGET", "8")
    contract = src._p_core_backlog_allocation_contract(
        active_issue_ids=["single_writer_merge_speed"],
        core_pending_lines=26000,
        total_pending_lines=42000,
        core_target=5000,
        total_target=15000,
        oldest_age_seconds=1200.0,
        age_threshold_seconds=240.0,
        sparse_active=True,
        sparse_pending_bytes=1_000_000_000,
        host_context={
            "off_hours_active": True,
            "runtime_throttle": {
                "host_saturation_score": 30.0,
                "compute_pressure_level": "normal",
                "memory_pressure_level": "normal",
            },
            "resource_guard": {"creative_session_level": "hot", "creative_session_kind": "video_editing"},
            "computer_task": {"primary_task": "video_editing"},
        },
    )

    assert contract["preprocess_worker_budget"] == 3
    assert contract["p_core_burst_intelligence"]["mode"] == "creative_foreground_protect_3"


def test_p_core_burst_intelligence_narrows_host_pressure_before_runtime_degrades(monkeypatch) -> None:
    monkeypatch.setenv("BACKLOG_PCORE_TARGET", "8")
    contract = src._p_core_backlog_allocation_contract(
        active_issue_ids=["single_writer_merge_speed"],
        core_pending_lines=26000,
        total_pending_lines=42000,
        core_target=5000,
        total_target=15000,
        oldest_age_seconds=1200.0,
        age_threshold_seconds=240.0,
        sparse_active=True,
        sparse_pending_bytes=1_000_000_000,
        host_context={
            "off_hours_active": True,
            "runtime_throttle": {
                "host_saturation_score": 79.0,
                "compute_pressure_level": "high",
                "memory_pressure_level": "normal",
                "throttle_profile": "sustain",
            },
            "resource_guard": {"creative_session_level": "idle"},
        },
    )

    assert contract["preprocess_worker_budget"] == 3
    assert contract["p_core_burst_intelligence"]["mode"] == "host_pressure_relief_3"


def test_p_core_burst_intelligence_keeps_daily_driver_at_five(monkeypatch) -> None:
    monkeypatch.setenv("BACKLOG_PCORE_TARGET", "8")
    contract = src._p_core_backlog_allocation_contract(
        active_issue_ids=["single_writer_merge_speed"],
        core_pending_lines=11000,
        total_pending_lines=17000,
        core_target=5000,
        total_target=15000,
        oldest_age_seconds=300.0,
        age_threshold_seconds=240.0,
        sparse_active=False,
        sparse_pending_bytes=0,
        host_context={
            "off_hours_active": False,
            "runtime_throttle": {
                "host_saturation_score": 45.0,
                "compute_pressure_level": "elevated",
                "memory_pressure_level": "normal",
            },
            "resource_guard": {"creative_session_level": "active", "creative_session_kind": "music_playback"},
            "computer_task": {"primary_task": "music_playback"},
        },
    )

    assert contract["preprocess_worker_budget"] == 5
    assert contract["p_core_burst_intelligence"]["mode"] == "daily_driver_5"


def test_p_core_burst_intelligence_narrows_for_memory_pressure(monkeypatch) -> None:
    monkeypatch.setenv("BACKLOG_PCORE_TARGET", "8")
    contract = src._p_core_backlog_allocation_contract(
        active_issue_ids=["single_writer_merge_speed", "sparse_huge_jsonl_files"],
        core_pending_lines=26000,
        total_pending_lines=42000,
        core_target=5000,
        total_target=15000,
        oldest_age_seconds=1200.0,
        age_threshold_seconds=240.0,
        sparse_active=True,
        sparse_pending_bytes=1_000_000_000,
        host_context={
            "off_hours_active": True,
            "runtime_throttle": {
                "host_saturation_score": 35.0,
                "compute_pressure_level": "normal",
                "memory_pressure_level": "yellow",
                "throttle_profile": "sustain",
            },
            "resource_guard": {"memory_pressure_kind": "swap_only", "swap_used_gb": 13.0},
        },
    )

    assert contract["preprocess_worker_budget"] == 3
    assert contract["p_core_burst_intelligence"]["mode"] == "memory_relief_3"
    assert contract["control_env"]["BACKLOG_MEMORY_PRESSURE_CORE_OPTIMIZER"] == "1"
