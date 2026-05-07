import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import backpressure_drainer_fleet as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_backpressure_drainer_fleet_routes_concentrated_decisions_to_trading(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 64455,
            "pending_lines_total": 64472,
            "top_pending_files": [
                {
                    "source_rel": "governance/channels/decision/conservative_equities_schwab/decision_20260430.jsonl",
                    "pending_lines": 30675,
                    "oldest_pending_age_seconds": 7200.0,
                },
                {
                    "source_rel": "governance/channels/decision/aggressive_equities_schwab/decision_20260430.jsonl",
                    "pending_lines": 29664,
                    "oldest_pending_age_seconds": 7200.0,
                },
            ],
        },
    )
    _write_json(health / "ingestion_storage_control_latest.json", {"severity": "critical"})

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 5, 1, 2, 0, tzinfo=timezone.utc),
    )

    assert payload["overall_status"] == "ready"
    assert payload["active_drainer"]["name"] == "core_decision_drainer"
    assert payload["active_drainer"]["shards"][:2] == ["trading", "aggressive_trading"]
    assert payload["active_drainer"]["concentration"]["concentrated"] is True
    assert payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARDS"].startswith("trading,")
    assert payload["active_env_overrides"]["SQL_LINK_SERVICE_CONCENTRATED_CORE_DRAIN"] == "1"
    assert payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARD_LINK_TIMEOUT_SECONDS"] == "420"
    assert payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_MAX_LINES_PER_FILE"] == "12000"
    assert payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARD_TRADING_MAX_LINES_PER_FILE"] == "24000"
    assert "conservative_equities_schwab" in payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARD_TRADING_PATH_CONTAINS"]


def test_backpressure_drainer_fleet_routes_crypto_decisions_to_crypto_shard(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 1200,
            "pending_lines_total": 1200,
            "top_pending_files": [
                {
                    "source_rel": "governance/channels/decision/default_crypto_schwab/decision_20260501.jsonl",
                    "pending_lines": 800,
                    "oldest_pending_age_seconds": 900.0,
                },
                {
                    "source_rel": "decisions/shadow_crypto_futures_crypto/trade_decisions_20260501.jsonl",
                    "pending_lines": 400,
                    "oldest_pending_age_seconds": 600.0,
                },
            ],
        },
    )

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 5, 1, 2, 0, tzinfo=timezone.utc),
    )

    assert payload["overall_status"] == "ready"
    assert payload["active_drainer"]["name"] == "core_decision_drainer"
    assert payload["active_drainer"]["shards"][:1] == ["crypto_trading"]
    assert payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARDS"].startswith("crypto_trading,")
    assert "SQL_LINK_SERVICE_SHARD_TRADING_PATH_CONTAINS" not in payload["active_env_overrides"]
    assert "default_crypto_schwab" in payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_PATH_CONTAINS"]


def test_backpressure_drainer_fleet_writes_single_writer_handoff(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 1000,
            "pending_lines_total": 1000,
            "pending_lines_support_telemetry": 1000,
            "top_support_telemetry_pending_files": [
                {
                    "source_rel": "governance/watchdog/failover_events.jsonl",
                    "pending_lines": 1000,
                    "oldest_pending_age_seconds": 300.0,
                }
            ],
        },
    )

    payload = src.build_payload(
        project_root,
        apply=True,
        now_utc=datetime(2026, 5, 1, 15, 0, tzinfo=timezone.utc),
    )

    assert payload["overall_status"] == "handoff_requested"
    assert payload["service_request"]["request_kind"] == "backpressure_drainer_fleet"
    assert payload["service_request"]["reason"].endswith(":support_watchdog_drainer")
    assert payload["service_request"]["env_overrides"]["SQL_LINK_SERVICE_SHARDS"] == "support_watchdog,health_fast"
    assert (health / "sql_link_service_request_latest.json").exists()


def test_backpressure_drainer_fleet_guards_cold_stage_during_market_hours(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 10,
            "pending_lines_total": 120000,
            "pending_lines_cold": 120000,
            "top_cold_pending_files": [
                {
                    "source_rel": "data/stale_stage/decision_explanations/project/decision_explanations/a.jsonl",
                    "pending_lines": 120000,
                    "oldest_pending_age_seconds": 86400.0,
                }
            ],
        },
    )

    payload = src.build_payload(
        project_root,
        apply=True,
        now_utc=datetime(2026, 5, 1, 15, 0, tzinfo=timezone.utc),
    )

    assert payload["overall_status"] == "blocked"
    assert payload["blocked_reasons"] == ["market_hours_guard"]
    assert not (health / "sql_link_service_request_latest.json").exists()


def test_backpressure_drainer_fleet_can_force_cold_stage_handoff(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 10,
            "pending_lines_total": 120000,
            "pending_lines_cold": 120000,
            "top_cold_pending_files": [
                {
                    "source_rel": "data/stale_stage/decision_explanations/project/decision_explanations/a.jsonl",
                    "pending_lines": 120000,
                    "oldest_pending_age_seconds": 86400.0,
                }
            ],
        },
    )

    payload = src.build_payload(
        project_root,
        apply=True,
        force_live_window=True,
        now_utc=datetime(2026, 5, 1, 15, 0, tzinfo=timezone.utc),
    )

    assert payload["overall_status"] == "handoff_requested"
    assert payload["active_drainer"]["name"] == "cold_stage_drainer"
    assert payload["service_request"]["env_overrides"]["SQL_LINK_SERVICE_SHARDS"].startswith("data,")
    assert payload["service_request"]["env_overrides"]["SQL_LINK_SERVICE_SHARD_EXPLANATIONS_PATH_CONTAINS"].endswith("a.jsonl")
    assert payload["service_request"]["env_overrides"]["SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_MAX_LINES_PER_FILE"] == "64000"



def test_backpressure_drainer_fleet_splits_api_ingress_from_runtime(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines_total": 2700,
            "top_pending_files": [
                {
                    "source_rel": "governance/channels/api/default_crypto_schwab/api_20260504.jsonl",
                    "pending_lines": 900,
                    "oldest_pending_age_seconds": 600.0,
                },
                {
                    "source_rel": "governance/channels/ingress/default_equities_schwab/ingress_20260504.jsonl",
                    "pending_lines": 700,
                    "oldest_pending_age_seconds": 600.0,
                },
                {
                    "source_rel": "governance/channels/runtime/default_equities_schwab/runtime_20260504.jsonl",
                    "pending_lines": 1100,
                    "oldest_pending_age_seconds": 600.0,
                },
            ],
        },
    )

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 5, 1, 15, 0, tzinfo=timezone.utc),
    )

    assert payload["overall_status"] == "ready"
    assert payload["active_drainer"]["name"] == "api_ingress_drainer"
    assert payload["active_drainer"]["shards"][:2] == ["crypto_api_ingress", "governance"]
    assert "default_crypto_schwab" in payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARD_CRYPTO_API_INGRESS_PATH_CONTAINS"]
    assert "default_equities_schwab" in payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARD_GOVERNANCE_PATH_CONTAINS"]
    runtime = next(row for row in payload["candidate_drainers"] if row["name"] == "runtime_channel_drainer")
    assert runtime["pending_lines"] == 1100


def test_backpressure_drainer_fleet_routes_schema_violations_to_isolated_shard(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines_total": 600,
            "top_pending_files": [
                {
                    "source_rel": "governance/events/channel_schema_violations_20260504.jsonl",
                    "pending_lines": 600,
                    "oldest_pending_age_seconds": 1200.0,
                }
            ],
        },
    )

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 5, 1, 15, 0, tzinfo=timezone.utc),
    )

    assert payload["active_drainer"]["name"] == "schema_violation_drainer"
    assert payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARDS"] == "schema_violations,health_fast"
    assert payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARD_SCHEMA_VIOLATIONS_MAX_LINES_PER_FILE"] == "16000"


def test_backpressure_drainer_fleet_queues_secondary_live_safe_drainers(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines_total": 1200,
            "top_pending_files": [
                {
                    "source_rel": "governance/channels/paper_broker_bridge/default_equities_schwab/bridge_20260504.jsonl",
                    "pending_lines": 700,
                    "oldest_pending_age_seconds": 300.0,
                },
                {
                    "source_rel": "governance/events/shadow_pnl_attribution_default_equities_20260504.jsonl",
                    "pending_lines": 500,
                    "oldest_pending_age_seconds": 300.0,
                },
            ],
        },
    )

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 5, 1, 15, 0, tzinfo=timezone.utc),
    )

    assert payload["active_drainer"]["name"] == "fast_trade_bridge_drainer"
    assert payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARDS"] == "trading_fast,health_fast"
    assert payload["next_drainer_queue"][0]["name"] == "attribution_drainer"
    assert payload["metrics"]["expanded_lane_count"] >= 9


def test_backpressure_drainer_fleet_prioritizes_old_live_safe_age_pressure(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 4993,
            "pending_lines_total": 5022,
            "oldest_age_threshold_seconds": 240,
            "top_pending_files": [
                {
                    "source_rel": "decisions/shadow_cdo_squared_equities/trade_decisions_20260506.jsonl",
                    "pending_lines": 382,
                    "oldest_pending_age_seconds": 215.0,
                },
                {
                    "source_rel": "exports/paper_broker_bridge/paper/paper_bridge_orders_20260506.jsonl",
                    "pending_lines": 189,
                    "oldest_pending_age_seconds": 1088.0,
                },
                {
                    "source_rel": "paper_trades_paper.jsonl",
                    "pending_lines": 189,
                    "oldest_pending_age_seconds": 1088.0,
                },
            ],
        },
    )

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 5, 6, 23, 0, tzinfo=timezone.utc),
    )

    assert payload["active_drainer"]["name"] == "fast_trade_bridge_drainer"
    assert payload["active_drainer"]["age_pressure_priority_bonus"] > 0
    assert payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARDS"] == "trading_fast,health_fast"
    assert "paper_trades_paper.jsonl" in payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARD_TRADING_FAST_PATH_CONTAINS"]


def test_backpressure_drainer_fleet_routes_stale_provider_tails_even_when_tiny(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 245,
            "pending_lines_total": 245,
            "top_pending_files": [
                {
                    "source_rel": "decisions/shadow_conservative_equities/trade_decisions_20260505.jsonl",
                    "pending_lines": 237,
                    "oldest_pending_age_seconds": 9289.538,
                },
                {
                    "source_rel": "data/tradingeconomics/tradingeconomics_guest_rows_20260502.jsonl",
                    "pending_lines": 4,
                    "oldest_pending_age_seconds": 226918.124,
                },
                {
                    "source_rel": "data/tradingeconomics/tradingeconomics_guest_rows_20260503.jsonl",
                    "pending_lines": 4,
                    "oldest_pending_age_seconds": 140032.872,
                },
            ],
        },
    )
    _write_json(health / "ingestion_storage_control_latest.json", {"severity": "critical"})

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 5, 5, 15, 0, tzinfo=timezone.utc),
    )

    assert payload["active_drainer"]["name"] == "market_data_provider_drainer"
    assert payload["active_drainer"]["readiness_reason"] == "stale_tail"
    assert payload["active_drainer"]["pending_lines"] == 8
    assert payload["active_drainer"]["shards"] == ["data", "governance", "health_fast"]
    assert "tradingeconomics_guest_rows_20260502" in payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARD_DATA_PATH_CONTAINS"]
    assert payload["next_drainer_queue"][0]["name"] == "core_decision_drainer"
    assert payload["metrics"]["stale_tail_ready_count"] == 1
    assert payload["metrics"]["expanded_lane_count"] >= 15


def test_backpressure_drainer_fleet_routes_derivatives_to_focused_trading_shards(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines_total": 1700,
            "top_pending_files": [
                {
                    "source_rel": "decisions/shadow_options_on_futures_aggressive/trade_decisions_20260505.jsonl",
                    "pending_lines": 900,
                    "oldest_pending_age_seconds": 600.0,
                },
                {
                    "source_rel": "governance/quant_models/model_risk_validation/retrain_surface_20260505.jsonl",
                    "pending_lines": 800,
                    "oldest_pending_age_seconds": 900.0,
                },
            ],
        },
    )

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 5, 5, 15, 0, tzinfo=timezone.utc),
    )

    assert payload["active_drainer"]["name"] == "derivatives_surface_drainer"
    assert "trading" in payload["active_drainer"]["shards"]
    assert "options_on_futures" in payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARD_TRADING_PATH_CONTAINS"]
    model = next(row for row in payload["candidate_drainers"] if row["name"] == "model_research_drainer")
    assert model["status"] == "ready"
    assert model["assigned_pressure_lane"] == "model_retrain_research_backpressure"


def test_backpressure_drainer_fleet_keeps_report_cockpit_drainer_protected(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines_total": 900,
            "top_deferred_pending_files": [
                {
                    "source_rel": "governance/reports/operator_cockpit/report_ready_packets_20260505.jsonl",
                    "pending_lines": 900,
                    "oldest_pending_age_seconds": 12000.0,
                }
            ],
        },
    )

    payload = src.build_payload(
        project_root,
        apply=True,
        now_utc=datetime(2026, 5, 5, 15, 0, tzinfo=timezone.utc),
    )

    assert payload["active_drainer"]["name"] == "report_cockpit_drainer"
    assert payload["active_drainer"]["live_window_safe"] is False
    assert payload["overall_status"] == "blocked"
    assert payload["blocked_reasons"] == ["market_hours_guard"]

    forced = src.build_payload(
        project_root,
        apply=True,
        force_live_window=True,
        now_utc=datetime(2026, 5, 5, 15, 0, tzinfo=timezone.utc),
    )
    assert forced["overall_status"] == "handoff_requested"
    assert forced["service_request"]["assigned_pressure_lane"] == "report_cockpit_backpressure"


def test_backpressure_drainer_fleet_routes_settlement_reconciliation_to_dedicated_lane(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines_total": 1600,
            "top_pending_files": [
                {
                    "source_rel": "governance/reconciliation/positions/fills_20260506.jsonl",
                    "pending_lines": 1600,
                    "oldest_pending_age_seconds": 1800.0,
                }
            ],
        },
    )

    payload = src.build_payload(
        project_root,
        apply=True,
        now_utc=datetime(2026, 5, 6, 15, 0, tzinfo=timezone.utc),
    )

    assert payload["overall_status"] == "handoff_requested"
    assert payload["active_drainer"]["name"] == "settlement_reconciliation_drainer"
    assert payload["active_drainer"]["assigned_pressure_lane"] == "settlement_reconciliation_backpressure"
    assert payload["active_drainer"]["shards"] == ["governance", "trading_fast", "health_fast"]
    assert payload["active_drainer"]["self_accommodation"]["allowed_parallel_writers"] == 1
    assert payload["active_drainer"]["self_accommodation"]["starts_parallel_sql_writers"] is False
    assert payload["service_request"]["self_accommodation"]["coordination_model"] == "single_sql_writer_focused_handoff"
    assert payload["self_accommodation"]["next_safe_action"] == "single_writer_handoff_requested"
    assert "positions/fills_20260506" in payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARD_GOVERNANCE_PATH_CONTAINS"]


def test_backpressure_drainer_fleet_exposes_self_accommodation_contracts_for_new_lanes(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines_total": 1800,
            "top_pending_files": [
                {
                    "source_rel": "governance/health/runtime_pressure/memory_efficiency_20260506.jsonl",
                    "pending_lines": 1000,
                    "oldest_pending_age_seconds": 2400.0,
                },
                {
                    "source_rel": "governance/data_quality/source_verification/provider_adapter_20260506.jsonl",
                    "pending_lines": 800,
                    "oldest_pending_age_seconds": 2400.0,
                },
            ],
        },
    )

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 5, 6, 15, 0, tzinfo=timezone.utc),
    )

    assert payload["overall_status"] == "ready"
    assert payload["metrics"]["expanded_lane_count"] >= 19
    assert payload["metrics"]["self_accommodating_lane_count"] == payload["metrics"]["expanded_lane_count"]
    assert payload["self_accommodation"]["allowed_parallel_writers"] == 1
    assert payload["self_accommodation"]["next_safe_action"] == "run_backpressure_drainer_fleet_apply_or_bounded_super_drainer_wave"
    memory_lane = next(row for row in payload["candidate_drainers"] if row["name"] == "memory_runtime_artifact_drainer")
    data_quality_lane = next(row for row in payload["candidate_drainers"] if row["name"] == "data_quality_contract_drainer")
    assert memory_lane["status"] == "ready"
    assert data_quality_lane["status"] == "ready"
    assert data_quality_lane["self_accommodation"]["safe_expansion_rule"].startswith("sequence_bounded_handoffs")


def test_backpressure_drainer_fleet_routes_predictive_and_self_healing_lanes(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines_total": 2700,
            "top_pending_files": [
                {
                    "source_rel": "governance/predictive_stability/pressure_trajectory_20260506.jsonl",
                    "pending_lines": 1400,
                    "oldest_pending_age_seconds": 2200.0,
                },
                {
                    "source_rel": "governance/self_healing/blocked_surface_recovery_plan_20260506.jsonl",
                    "pending_lines": 1300,
                    "oldest_pending_age_seconds": 2100.0,
                },
            ],
        },
    )

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 5, 6, 15, 0, tzinfo=timezone.utc),
    )

    assert payload["overall_status"] == "ready"
    assert payload["active_drainer"]["name"] == "predictive_stability_drainer"
    assert payload["active_drainer"]["assigned_pressure_lane"] == "predictive_stability_backpressure"
    assert payload["next_drainer_queue"][0]["name"] == "self_healing_recovery_drainer"
    assert payload["metrics"]["expanded_lane_count"] >= 25


def test_backpressure_drainer_fleet_routes_admission_and_writer_recovery_lanes(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines_total": 2000,
            "top_pending_files": [
                {
                    "source_rel": "governance/admission_evidence/new_bot_admission/sample_depth_20260506.jsonl",
                    "pending_lines": 1200,
                    "oldest_pending_age_seconds": 1900.0,
                },
                {
                    "source_rel": "governance/writer_progress/jsonl_sql_writer/merge_progress_20260506.jsonl",
                    "pending_lines": 800,
                    "oldest_pending_age_seconds": 950.0,
                },
            ],
        },
    )

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 5, 6, 15, 0, tzinfo=timezone.utc),
    )

    assert payload["active_drainer"]["name"] == "admission_evidence_drainer"
    writer_lane = next(row for row in payload["candidate_drainers"] if row["name"] == "writer_progress_recovery_drainer")
    assert writer_lane["status"] == "ready"
    assert writer_lane["assigned_pressure_lane"] == "writer_progress_recovery_backpressure"


def test_backpressure_drainer_fleet_routes_training_collection_storage_and_ingestion_lanes(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines_total": 6200,
            "top_pending_files": [
                {
                    "source_rel": "governance/training_lineage/lineage_manifest_20260506.jsonl",
                    "pending_lines": 1600,
                    "oldest_pending_age_seconds": 2200.0,
                },
                {
                    "source_rel": "governance/training_labeling_intelligence/label_coverage_20260506.jsonl",
                    "pending_lines": 1300,
                    "oldest_pending_age_seconds": 2100.0,
                },
                {
                    "source_rel": "governance/collector_telemetry/observation_rollup_20260506.jsonl",
                    "pending_lines": 1200,
                    "oldest_pending_age_seconds": 1900.0,
                },
                {
                    "source_rel": "governance/storage_route/split_brain_reconcile_20260506.jsonl",
                    "pending_lines": 1100,
                    "oldest_pending_age_seconds": 1800.0,
                },
                {
                    "source_rel": "governance/ingestion_priority/backlog_quarantine_20260506.jsonl",
                    "pending_lines": 1000,
                    "oldest_pending_age_seconds": 1700.0,
                },
            ],
        },
    )

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 5, 6, 15, 0, tzinfo=timezone.utc),
    )

    assert payload["active_drainer"]["name"] == "training_lineage_drainer"
    assert payload["active_drainer"]["assigned_pressure_lane"] == "training_lineage_backpressure"
    lanes = {row["name"]: row for row in payload["candidate_drainers"]}
    assert lanes["label_contract_drainer"]["status"] == "ready"
    assert lanes["collector_telemetry_rollup_drainer"]["status"] == "ready"
    assert lanes["storage_route_reconcile_drainer"]["status"] == "ready"
    assert lanes["ingestion_priority_drainer"]["status"] == "ready"
    assert lanes["collector_telemetry_rollup_drainer"]["self_accommodation"]["allowed_parallel_writers"] == 1
    assert payload["metrics"]["expanded_lane_count"] >= 30
