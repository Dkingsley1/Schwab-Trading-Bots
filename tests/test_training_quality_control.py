import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import training_quality_control as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_build_payload_surfaces_blockers_and_targeted_actions(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    walk_root = tmp_path / "governance" / "walk_forward"

    _write_json(
        health_root / "training_registry_audit_latest.json",
        {
            "registry_active_bots": 6,
            "supportability_counts": {"unsupported_runtime_inputs": 2, "unsupported_stale_diagnostics": 1},
            "tier_counts": {"active_repair": 2, "active_probation": 1, "active_stale": 1, "research_candidate": 6},
            "active_sample_starved": [
                {
                    "bot_id": "brain_refinery_v4_simple",
                    "supportability_status": "unsupported_runtime_inputs",
                    "inferred_cause": "shared_runtime_input_gap",
                },
                {
                    "bot_id": "brain_refinery_v13_choppy",
                    "supportability_status": "unsupported_runtime_inputs",
                    "inferred_cause": "sequence_depth_gap",
                },
            ],
            "active_quality_failed": [{"bot_id": "brain_refinery_v43_intraday_ultrafast_proxy"}],
            "active_stale_diagnostics": [{"bot_id": "brain_refinery_v35_dmi_state_machine"}],
        },
    )
    _write_json(
        health_root / "training_label_audit_latest.json",
        {
            "top_actions": ["fix_shared_runtime_input", "tighten_abstention_thresholds"],
            "recommendation_counts": {"fix_shared_runtime_input": 2, "tighten_abstention_thresholds": 1},
        },
    )
    _write_json(
        health_root / "runtime_training_snapshot_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "row_count": 1000,
            "sequence_count": 12,
            "coverage": {
                "top_modes": [
                    {"mode": "shadow_intraday_aggressive_equities", "row_count": 420},
                    {"mode": "shadow_crypto", "row_count": 280},
                    {"mode": "shadow_dividend_equities", "row_count": 70},
                ],
                "top_symbols": [
                    {"symbol": "BTC-USD", "row_count": 90},
                    {"symbol": "ETH-USD", "row_count": 80},
                    {"symbol": "SPY", "row_count": 70},
                ],
            },
        },
    )
    _write_json(
        walk_root / "promotion_readiness_latest.json",
        {
            "promote_ok": False,
            "considered_bots": 1,
            "thresholds": {"min_considered_bots": 4},
        },
    )
    _write_json(health_root / "promotion_quality_gate_latest.json", {"ok": False})
    _write_json(
        tmp_path / "governance" / "champion_challenger" / "promotion_packet_latest.json",
        {
            "packet_sha256": "seeded-packet",
            "dataset": {"rows_sha256": "rows-hash"},
        },
    )
    _write_json(
        tmp_path / "governance" / "feature_store" / "latest.json",
        {
            "ok": False,
            "lineage_schema_version": 1,
            "dataset_contract": {"rows_sha256": ""},
            "point_in_time_contract": {"dataset_join_keys": []},
            "lane_partitions": [],
        },
    )
    _write_json(tmp_path / "governance" / "research" / "multiple_testing_guard_latest.json", {"ok": False, "overall_status": "blocked", "family_size": 0})
    _write_json(tmp_path / "governance" / "research" / "decay_monitor_latest.json", {"overall_status": "needs_work", "weak_sleeve_count": 1})
    _write_json(health_root / "replay_hash_registry_guard_latest.json", {"ok": False})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "severity": "critical",
            "backpressure": {"estimated_core_drain_minutes": 55.0, "estimated_total_drain_minutes": 260.0},
            "storage": {"retention_debt_gb": 3.5},
            "top_actions": ["force retention"],
        },
    )
    experiment_path = tmp_path / "governance" / "experiments" / "experiment_registry.jsonl"
    experiment_path.parent.mkdir(parents=True, exist_ok=True)
    experiment_path.write_text(
        json.dumps({"experiment_id": "exp_bad", "replayability": {"exact_replay_ready": False}}) + "\n",
        encoding="utf-8",
    )
    _write_json(
        health_root / "training_report_latest.json",
        {
            "overall_status": "blocked",
            "summary": {"confirmed_training_success": False},
        },
    )
    _write_json(health_root / "health_gates_latest.json", {"hard_gate_triggered": True})
    _write_json(
        health_root / "paper_performance_latest.json",
        {
            "sleeve_latest": [
                {"profile": "intraday_aggressive", "ending_net_pnl_total": -5.0, "win_rate": 0.31},
                {"profile": "dividend", "ending_net_pnl_total": 1.0, "win_rate": 0.60},
            ]
        },
    )
    _write_json(
        health_root / "roster_resilience_planner_latest.json",
        {"bench": {"bench_depth": 1}, "a_plus_contract": {"a_plus_ready": False}},
    )

    payload = src.build_payload(tmp_path)

    assert payload["implemented_improvement_count"] == 26
    assert payload["overall_status"] == "blocked"
    assert "runtime_input_coverage" in payload["top_priorities"]
    assert "active_supportability" in payload["top_priorities"]
    assert "promotion_coverage" in payload["top_priorities"]
    assert payload["targeted_actions"]["repair_runtime_input_bot_ids"] == [
        "brain_refinery_v4_simple",
    ]
    assert payload["targeted_actions"]["runtime_input_depth_debt_bot_ids"] == ["brain_refinery_v13_choppy"]
    assert payload["targeted_actions"]["quality_probation_bot_ids"] == [
        "brain_refinery_v43_intraday_ultrafast_proxy",
    ]
    assert payload["dataset_shape"]["lane_lookback_days"]["intraday_aggressive"] == 23
    assert payload["a_plus_contract"]["roster_a_plus_ready"] is False
    assert payload["rollout"]["promotion_confidence_ready"] is False
    assert payload["data_ops"]["health_gate_triggered"] is True
    assert payload["data_ops"]["ingestion_storage_status"] == "blocked"
    assert payload["research"]["multiple_testing_status"] == "blocked"
    assert payload["immutable_lineage"]["lineage_status"] == "blocked"
    assert "storage_backpressure" in payload["failure_taxonomy"]["failure_buckets"]
    assert "runtime_input_depth_debt" in payload["failure_taxonomy"]["failure_buckets"]


def test_build_payload_marks_ready_when_training_surface_is_healthy(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    walk_root = tmp_path / "governance" / "walk_forward"

    _write_json(
        health_root / "training_registry_audit_latest.json",
        {
            "registry_active_bots": 4,
            "supportability_counts": {"supportable_active": 4},
            "tier_counts": {"research_candidate": 1},
            "active_sample_starved": [],
            "active_quality_failed": [],
            "active_stale_diagnostics": [],
        },
    )
    _write_json(health_root / "training_label_audit_latest.json", {"top_actions": [], "recommendation_counts": {}})
    _write_json(
        health_root / "runtime_training_snapshot_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "row_count": 2000,
            "sequence_count": 30,
            "coverage": {
                "top_modes": [
                    {"mode": "shadow_dividend_equities", "row_count": 350},
                    {"mode": "shadow_bond_equities", "row_count": 300},
                    {"mode": "shadow_fx_equities", "row_count": 250},
                ],
                "top_symbols": [
                    {"symbol": "SPY", "row_count": 40},
                    {"symbol": "AAPL", "row_count": 35},
                    {"symbol": "QQQ", "row_count": 30},
                ],
            },
        },
    )
    _write_json(
        walk_root / "promotion_readiness_latest.json",
        {
            "promote_ok": True,
            "considered_bots": 5,
            "thresholds": {"min_considered_bots": 4},
        },
    )
    _write_json(health_root / "promotion_quality_gate_latest.json", {"ok": True})
    _write_json(
        tmp_path / "governance" / "feature_store" / "latest.json",
        {
            "ok": True,
            "lineage_schema_version": 1,
            "dataset_contract": {"rows_sha256": "rows-hash"},
            "point_in_time_contract": {"dataset_join_keys": ["snapshot_id", "symbol"]},
            "lane_partitions": [{"lane": "dividend", "row_count": 350}],
        },
    )
    _write_json(
        tmp_path / "governance" / "research" / "multiple_testing_guard_latest.json",
        {"ok": True, "overall_status": "ready", "family_size": 8, "correction_method": "bonferroni", "regime_segments": ["dividend"]},
    )
    _write_json(
        tmp_path / "governance" / "research" / "decay_monitor_latest.json",
        {"ok": True, "overall_status": "ready", "weak_sleeve_count": 0, "history_days_available": 2, "pnl_slope": 1.5},
    )
    _write_json(health_root / "replay_hash_registry_guard_latest.json", {"ok": True})
    _write_json(
        health_root / "training_lineage_manifest_latest.json",
        {
            "lineage_contract_ready": True,
            "promotion_bundle_ready": True,
            "hash_bundle_complete": True,
            "feature_store_lineage_ok": True,
            "feature_store_schema_version": 1,
            "exact_replay_ready": True,
            "replay_hash_registry_ok": True,
            "lineage_score": 100.0,
        },
    )
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "backpressure": {"estimated_core_drain_minutes": 4.0, "estimated_total_drain_minutes": 20.0},
            "storage": {"retention_debt_gb": 0.0},
            "top_actions": [],
        },
    )
    experiment_path = tmp_path / "governance" / "experiments" / "experiment_registry.jsonl"
    experiment_path.parent.mkdir(parents=True, exist_ok=True)
    experiment_path.write_text(
        json.dumps({"experiment_id": "exp_ok", "replayability": {"exact_replay_ready": True}}) + "\n",
        encoding="utf-8",
    )
    _write_json(
        health_root / "training_report_latest.json",
        {
            "overall_status": "ready",
            "summary": {"confirmed_training_success": True},
        },
    )
    _write_json(health_root / "health_gates_latest.json", {"hard_gate_triggered": False})
    _write_json(health_root / "paper_performance_latest.json", {"sleeve_latest": []})
    _write_json(
        health_root / "roster_resilience_planner_latest.json",
        {"bench": {"bench_depth": 5}, "a_plus_contract": {"a_plus_ready": True}},
    )
    _write_json(health_root / "calibration_abstention_control_latest.json", {"overall_status": "ready"})

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "ready"
    assert payload["ok"] is True
    assert payload["rollout"]["promotion_confidence_ready"] is True
    assert payload["a_plus_contract"]["roster_a_plus_ready"] is True
    assert payload["improvement_status_counts"]["blocked"] == 0


def test_build_payload_softens_seeded_coverage_and_recovering_ingestion(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    walk_root = tmp_path / "governance" / "walk_forward"

    _write_json(
        health_root / "training_registry_audit_latest.json",
        {
            "registry_active_bots": 24,
            "supportability_counts": {"supportable_active": 18},
            "tier_counts": {"active_stale": 0, "research_candidate": 2},
            "active_sample_starved": [],
            "active_quality_failed": [],
            "active_stale_diagnostics": [],
        },
    )
    _write_json(health_root / "training_label_audit_latest.json", {"top_actions": [], "recommendation_counts": {}})
    _write_json(
        health_root / "runtime_training_snapshot_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "row_count": 3200,
            "sequence_count": 48,
            "coverage": {
                "top_modes": [
                    {"mode": "shadow_aggressive_equities", "row_count": 900},
                    {"mode": "shadow_conservative_equities", "row_count": 750},
                ],
                "top_symbols": [{"symbol": "SPY", "row_count": 50}],
            },
        },
    )
    _write_json(
        walk_root / "promotion_readiness_latest.json",
        {
            "promote_ok": False,
            "considered_bots": 0,
            "thresholds": {"min_considered_bots": 4},
        },
    )
    _write_json(
        walk_root / "coverage_seed_latest.json",
        {
            "seed_queue": [{"bot_id": "brain_refinery_v4_simple"}],
            "coverage_shortfall_bots": 4,
        },
    )
    _write_json(health_root / "promotion_quality_gate_latest.json", {"ok": False})
    _write_json(
        tmp_path / "governance" / "champion_challenger" / "promotion_packet_latest.json",
        {
            "packet_sha256": "seeded-packet",
            "dataset": {"rows_sha256": "rows-hash"},
        },
    )
    _write_json(
        tmp_path / "governance" / "feature_store" / "latest.json",
        {
            "ok": True,
            "lineage_schema_version": 1,
            "dataset_contract": {"rows_sha256": "rows-hash"},
            "point_in_time_contract": {"dataset_join_keys": ["snapshot_id", "symbol"]},
            "lane_partitions": [{"lane": "aggressive", "row_count": 900}],
        },
    )
    _write_json(
        tmp_path / "governance" / "research" / "multiple_testing_guard_latest.json",
        {
            "ok": False,
            "overall_status": "blocked",
            "family_size": 270,
            "correction_method": "benjamini_hochberg_fdr",
            "failed_checks": [],
        },
    )
    _write_json(tmp_path / "governance" / "research" / "decay_monitor_latest.json", {"overall_status": "ready", "weak_sleeve_count": 0})
    _write_json(health_root / "replay_hash_registry_guard_latest.json", {"ok": False})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "severity": "critical",
            "backpressure": {"estimated_core_drain_minutes": None, "estimated_total_drain_minutes": None},
            "storage": {"retention_debt_gb": 0.0, "backlog_drain_status": "drain_active"},
            "queue_watermarks": {"breaches": {"hard": []}},
            "ingestion_pressure": {
                "critical_priority_failures": [],
                "critical_priority_shard_storage_failures": [],
            },
            "top_actions": ["keep draining backlog"],
        },
    )
    _write_json(
        health_root / "training_report_latest.json",
        {
            "overall_status": "needs_attention",
            "summary": {
                "confirmed_training_success": False,
                "target_count": 0,
                "trained_count": 0,
                "failure_count": 0,
            },
        },
    )
    _write_json(
        health_root / "health_gates_latest.json",
        {
            "hard_gate_triggered": True,
            "hard_gates": {
                "ingestion_backpressure_overload": True,
                "collector_contracts": True,
                "stale_windows": False,
                "ingestion_pending_lines": False,
                "ingestion_oldest_age": False,
                "ingestion_invalid_lines": False,
                "sql_progress_stall": False,
                "sql_wal_pressure": False,
            },
        },
    )
    _write_json(health_root / "paper_performance_latest.json", {"sleeve_latest": []})
    _write_json(
        health_root / "roster_resilience_planner_latest.json",
        {"bench": {"bench_depth": 5}, "a_plus_contract": {"a_plus_ready": False}},
    )
    _write_json(health_root / "calibration_abstention_control_latest.json", {"overall_status": "ready"})

    payload = src.build_payload(tmp_path)
    improvement_by_key = {row["key"]: row for row in payload["improvements"]}

    assert payload["overall_status"] == "needs_attention"
    assert improvement_by_key["promotion_coverage"]["status"] == "needs_work"
    assert improvement_by_key["ingestion_health_guard"]["status"] == "ready"
    assert improvement_by_key["ingestion_drain_time_guard"]["status"] == "needs_work"
    assert improvement_by_key["multiple_testing_control"]["status"] == "ready"
    assert payload["research"]["multiple_testing_ready"] is True
    assert "training_not_confirmed" not in payload["failure_taxonomy"]["failure_buckets"]


def test_build_payload_credits_provisional_registry_backed_active_bots(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    walk_root = tmp_path / "governance" / "walk_forward"
    champion_root = tmp_path / "governance" / "champion_challenger"

    _write_json(
        health_root / "training_registry_audit_latest.json",
        {
            "registry_active_bots": 4,
            "supportability_counts": {},
            "tier_counts": {"active_stale": 4},
            "active_sample_starved": [],
            "active_quality_failed": [],
            "active_stale_diagnostics": [
                {"bot_id": "bot_a", "registry_quality_score": 0.45, "registry_test_accuracy": 0.76},
                {"bot_id": "bot_b", "registry_quality_score": 0.31, "registry_test_accuracy": 0.54},
                {"bot_id": "bot_c", "registry_quality_score": 0.20, "registry_test_accuracy": 0.51},
                {"bot_id": "bot_d", "registry_quality_score": 0.05, "registry_test_accuracy": 0.49},
            ],
        },
    )
    _write_json(health_root / "training_label_audit_latest.json", {"top_actions": [], "recommendation_counts": {}})
    _write_json(
        health_root / "runtime_training_snapshot_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "row_count": 1600,
            "sequence_count": 18,
            "coverage": {
                "top_modes": [{"mode": "shadow_aggressive_equities", "row_count": 400}],
                "top_symbols": [{"symbol": "SPY", "row_count": 40}],
            },
        },
    )
    _write_json(
        walk_root / "promotion_readiness_latest.json",
        {
            "promote_ok": True,
            "considered_bots": 4,
            "thresholds": {"min_considered_bots": 4},
        },
    )
    _write_json(health_root / "promotion_quality_gate_latest.json", {"ok": True})
    _write_json(
        tmp_path / "governance" / "feature_store" / "latest.json",
        {
            "ok": True,
            "lineage_schema_version": 1,
            "dataset_contract": {"rows_sha256": "rows-hash"},
            "point_in_time_contract": {"dataset_join_keys": ["snapshot_id", "symbol"]},
            "lane_partitions": [{"lane": "aggressive", "row_count": 400}],
        },
    )
    _write_json(
        champion_root / "promotion_packet_latest.json",
        {
            "packet_sha256": "packet-sha",
            "dataset": {"rows_sha256": "rows-hash"},
        },
    )
    _write_json(
        tmp_path / "governance" / "research" / "multiple_testing_guard_latest.json",
        {
            "ok": False,
            "overall_status": "blocked",
            "family_size": 12,
            "correction_method": "benjamini_hochberg_fdr",
            "regime_segments": ["aggressive"],
        },
    )
    _write_json(
        tmp_path / "governance" / "research" / "decay_monitor_latest.json",
        {"ok": True, "overall_status": "ready", "weak_sleeve_count": 0},
    )
    _write_json(health_root / "replay_hash_registry_guard_latest.json", {"ok": False})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "backpressure": {"estimated_core_drain_minutes": 8.0, "estimated_total_drain_minutes": 22.0},
            "storage": {"retention_debt_gb": 0.0},
            "top_actions": [],
        },
    )
    _write_json(
        health_root / "training_report_latest.json",
        {
            "overall_status": "ready",
            "summary": {"confirmed_training_success": True, "target_count": 0, "failure_count": 0},
        },
    )
    _write_json(health_root / "health_gates_latest.json", {"hard_gate_triggered": False})
    _write_json(health_root / "paper_performance_latest.json", {"sleeve_latest": []})
    _write_json(
        health_root / "roster_resilience_planner_latest.json",
        {"bench": {"bench_depth": 4}, "a_plus_contract": {"a_plus_ready": False}},
    )
    _write_json(health_root / "calibration_abstention_control_latest.json", {"overall_status": "ready"})

    payload = src.build_payload(tmp_path)

    assert payload["supportability"]["active_supportable_bots"] == 1
    assert payload["supportability"]["active_supportability_score"] == 25.0
    assert payload["targeted_actions"]["provisional_registry_backed_bot_ids"] == ["bot_a"]
    assert payload["targeted_actions"]["unsupported_stale_bot_ids"] == ["bot_b", "bot_c", "bot_d"]
    assert payload["research"]["multiple_testing_contract_present"] is True
    assert payload["immutable_lineage"]["provisional_lineage_ready"] is True
    assert payload["improvement_status_counts"]["blocked"] == 1
    assert payload["targeted_actions"]["targeted_retrain_bot_ids"] == []
    assert payload["rollout"]["exact_replay_ready"] is False


def test_build_payload_uses_supportability_active_denominator_when_collection_bots_are_isolated(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    walk_root = tmp_path / "governance" / "walk_forward"

    _write_json(
        health_root / "training_registry_audit_latest.json",
        {
            "registry_active_bots": 104,
            "registry_supportability_active_bots": 4,
            "active_collection_only_bots": 100,
            "supportability_counts": {"supportable_active": 4, "collection_only_active": 100},
            "tier_counts": {"active_collection_only": 100, "active_stale": 0},
            "active_sample_starved": [],
            "active_quality_failed": [],
            "active_stale_diagnostics": [],
        },
    )
    _write_json(health_root / "training_label_audit_latest.json", {"top_actions": [], "recommendation_counts": {}})
    _write_json(
        health_root / "runtime_training_snapshot_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "row_count": 2000,
            "sequence_count": 30,
            "coverage": {
                "top_modes": [{"mode": "shadow_dividend_equities", "row_count": 300}],
                "top_symbols": [{"symbol": "SPY", "row_count": 30}],
            },
        },
    )
    _write_json(
        walk_root / "promotion_readiness_latest.json",
        {"promote_ok": True, "considered_bots": 4, "thresholds": {"min_considered_bots": 4}},
    )
    _write_json(health_root / "promotion_quality_gate_latest.json", {"ok": True})
    _write_json(
        tmp_path / "governance" / "feature_store" / "latest.json",
        {
            "ok": True,
            "lineage_schema_version": 1,
            "dataset_contract": {"rows_sha256": "rows-hash"},
            "point_in_time_contract": {"dataset_join_keys": ["snapshot_id", "symbol"]},
            "lane_partitions": [{"lane": "dividend", "row_count": 300}],
        },
    )
    _write_json(tmp_path / "governance" / "research" / "multiple_testing_guard_latest.json", {"ok": True, "overall_status": "ready", "family_size": 8, "correction_method": "bonferroni"})
    _write_json(tmp_path / "governance" / "research" / "decay_monitor_latest.json", {"ok": True, "overall_status": "ready", "weak_sleeve_count": 0})
    _write_json(health_root / "replay_hash_registry_guard_latest.json", {"ok": True})
    _write_json(
        health_root / "training_lineage_manifest_latest.json",
        {
            "lineage_contract_ready": True,
            "promotion_bundle_ready": True,
            "hash_bundle_complete": True,
            "feature_store_lineage_ok": True,
            "exact_replay_ready": True,
            "replay_hash_registry_ok": True,
            "lineage_score": 100.0,
        },
    )
    _write_json(health_root / "ingestion_storage_control_latest.json", {"overall_status": "ready", "storage": {"retention_debt_gb": 0.0}})
    _write_json(health_root / "training_report_latest.json", {"overall_status": "ready", "summary": {"confirmed_training_success": True}})
    _write_json(health_root / "health_gates_latest.json", {"hard_gate_triggered": False})
    _write_json(health_root / "paper_performance_latest.json", {"sleeve_latest": []})
    _write_json(health_root / "roster_resilience_planner_latest.json", {"bench": {"bench_depth": 4}, "a_plus_contract": {"a_plus_ready": True}})
    _write_json(health_root / "calibration_abstention_control_latest.json", {"overall_status": "ready"})

    payload = src.build_payload(tmp_path)

    assert payload["supportability"]["active_bots"] == 4
    assert payload["supportability"]["raw_registry_active_bots"] == 104
    assert payload["supportability"]["active_collection_only_bots"] == 100
    assert payload["supportability"]["active_supportability_score"] == 100.0


def test_build_payload_rewards_launch_ready_coverage_and_stronger_provisional_lineage(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    walk_root = tmp_path / "governance" / "walk_forward"
    research_root = tmp_path / "governance" / "research"
    champion_root = tmp_path / "governance" / "champion_challenger"
    experiments_root = tmp_path / "governance" / "experiments"

    _write_json(
        health_root / "training_registry_audit_latest.json",
        {
            "registry_active_bots": 18,
            "supportability_counts": {"supportable_active": 12, "registry_seeded_active": 3},
            "tier_counts": {"active_stale": 3, "research_candidate": 2},
            "active_sample_starved": [],
            "active_quality_failed": [],
            "active_stale_diagnostics": [
                {"bot_id": "bot_a", "registry_quality_score": 0.4, "registry_test_accuracy": 0.78},
                {"bot_id": "bot_b", "registry_quality_score": 0.35, "registry_test_accuracy": 0.75},
                {"bot_id": "bot_c", "registry_quality_score": 0.28, "registry_test_accuracy": 0.53},
            ],
            "active_registry_seeded": [
                {"bot_id": "bot_a"},
                {"bot_id": "bot_b"},
                {"bot_id": "bot_c"},
            ],
        },
    )
    _write_json(
        health_root / "training_label_audit_latest.json",
        {"top_actions": ["refresh_training_diagnostics"], "recommendation_counts": {"refresh_training_diagnostics": 1}},
    )
    _write_json(
        health_root / "runtime_training_snapshot_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "row_count": 2800,
            "sequence_count": 44,
            "coverage": {
                "top_modes": [
                    {"mode": "shadow_aggressive_equities", "row_count": 780},
                    {"mode": "shadow_conservative_equities", "row_count": 740},
                ],
                "top_symbols": [{"symbol": "SPY", "row_count": 38}],
            },
        },
    )
    _write_json(
        walk_root / "promotion_readiness_latest.json",
        {
            "promote_ok": False,
            "considered_bots": 0,
            "thresholds": {"min_considered_bots": 4},
        },
    )
    _write_json(
        walk_root / "coverage_gap_closer_latest.json",
        {
            "staged_candidate_count": 3,
            "autopilot_contract": {
                "can_apply_stage": True,
                "can_launch_now": True,
                "gating_signals": {"staged_candidates_present": True},
            },
        },
    )
    _write_json(health_root / "promotion_quality_gate_latest.json", {"ok": False})
    _write_json(
        champion_root / "promotion_packet_latest.json",
        {
            "packet_sha256": "packet-sha",
            "dataset": {"rows_sha256": "rows-hash"},
        },
    )
    _write_json(
        tmp_path / "governance" / "feature_store" / "latest.json",
        {
            "ok": True,
            "lineage_schema_version": 1,
            "dataset_contract": {"rows_sha256": "rows-hash"},
            "point_in_time_contract": {"dataset_join_keys": ["snapshot_id", "symbol"]},
            "lane_partitions": [{"lane": "aggressive", "row_count": 780}],
        },
    )
    _write_json(
        research_root / "multiple_testing_guard_latest.json",
        {
            "ok": False,
            "overall_status": "blocked",
            "family_size": 24,
            "correction_method": "benjamini_hochberg_fdr",
            "failed_checks": [],
        },
    )
    _write_json(research_root / "decay_monitor_latest.json", {"overall_status": "ready", "weak_sleeve_count": 0})
    _write_json(health_root / "replay_hash_registry_guard_latest.json", {"ok": True})
    _write_json(
        health_root / "training_lineage_manifest_latest.json",
        {
            "lineage_contract_ready": False,
            "promotion_bundle_ready": False,
            "hash_bundle_complete": False,
            "feature_store_lineage_ok": True,
            "feature_store_schema_version": 1,
            "exact_replay_ready": False,
            "replay_hash_registry_ok": True,
            "multiple_testing_ready": True,
            "decay_monitor_ready": True,
            "lineage_score": 57.5,
        },
    )
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "backpressure": {"estimated_core_drain_minutes": 7.0, "estimated_total_drain_minutes": 18.0},
            "storage": {"retention_debt_gb": 0.0},
            "top_actions": [],
        },
    )
    _write_json(
        health_root / "training_report_latest.json",
        {
            "overall_status": "needs_attention",
            "summary": {"confirmed_training_success": False, "target_count": 0, "trained_count": 0, "failure_count": 0},
        },
    )
    _write_json(health_root / "health_gates_latest.json", {"hard_gate_triggered": False})
    _write_json(health_root / "paper_performance_latest.json", {"sleeve_latest": []})
    _write_json(
        health_root / "roster_resilience_planner_latest.json",
        {"bench": {"bench_depth": 5}, "a_plus_contract": {"a_plus_ready": False}},
    )
    _write_json(health_root / "calibration_abstention_control_latest.json", {"overall_status": "ready"})
    experiments_root.mkdir(parents=True, exist_ok=True)
    (experiments_root / "experiment_registry.jsonl").write_text(
        json.dumps(
            {
                "experiment_id": "exp_seeded",
                "replayability": {
                    "bundle_hash": "bundle-hash",
                    "dataset_hash": "dataset-hash",
                    "model_hash": "",
                    "replay_hash": "",
                    "exact_replay_ready": False,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = src.build_payload(tmp_path)
    improvement_by_key = {row["key"]: row for row in payload["improvements"]}

    assert improvement_by_key["promotion_coverage"]["status"] == "needs_work"
    assert improvement_by_key["promotion_coverage"]["priority"] == 1
    assert improvement_by_key["promotion_coverage"]["metric"]["coverage_launch_ready"] is True
    assert improvement_by_key["label_and_abstention_calibration"]["status"] == "ready"
    assert payload["immutable_lineage"]["stronger_provisional_lineage_ready"] is True
    assert payload["immutable_lineage"]["experiment_bundle_seeded"] is True
    assert payload["rollout"]["lineage_contract_ready"] is False
    assert payload["rollout"]["promotion_bundle_ready"] is False
    assert payload["data_ops"]["retention_debt_gb"] == 0.0
    assert payload["immutable_lineage"]["lineage_status"] == "blocked"
    assert payload["immutable_lineage"]["hash_bundle_complete"] is False


def test_build_payload_treats_guarded_ingestion_and_signed_replay_bundle_as_recoverable(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    walk_root = tmp_path / "governance" / "walk_forward"
    champion_root = tmp_path / "governance" / "champion_challenger"

    _write_json(
        health_root / "training_registry_audit_latest.json",
        {
            "registry_active_bots": 20,
            "supportability_counts": {"registry_seeded_active": 12},
            "tier_counts": {"active_stale": 12},
            "active_sample_starved": [],
            "active_quality_failed": [],
            "active_stale_diagnostics": [{"bot_id": f"bot_{idx}"} for idx in range(12)],
            "active_registry_seeded": [{"bot_id": f"bot_{idx}"} for idx in range(12)],
        },
    )
    _write_json(health_root / "training_label_audit_latest.json", {"top_actions": [], "recommendation_counts": {}})
    _write_json(
        health_root / "runtime_training_snapshot_latest.json",
        {"timestamp_utc": now.isoformat(), "row_count": 2200, "sequence_count": 24, "coverage": {"top_modes": [], "top_symbols": []}},
    )
    _write_json(
        walk_root / "promotion_readiness_latest.json",
        {"promote_ok": False, "considered_bots": 0, "thresholds": {"min_considered_bots": 4}},
    )
    _write_json(
        walk_root / "coverage_gap_closer_latest.json",
        {
            "staged_candidate_count": 4,
            "autopilot_contract": {
                "can_apply_stage": True,
                "can_launch_now": False,
                "launch_state": "waiting_for_idle",
                "gating_signals": {"staged_candidates_present": True},
            },
        },
    )
    _write_json(health_root / "promotion_quality_gate_latest.json", {"ok": True})
    _write_json(
        champion_root / "promotion_packet_latest.json",
        {
            "packet_sha256": "packet-sha",
            "dataset": {"rows_sha256": "rows-hash"},
            "replayability_contract": {
                "dataset_hash": "dataset-hash",
                "model_hash": "model-hash",
                "replay_hash": "replay-hash",
                "bundle_hash": "bundle-hash",
            },
        },
    )
    _write_json(
        tmp_path / "governance" / "feature_store" / "latest.json",
        {
            "ok": True,
            "lineage_schema_version": 1,
            "dataset_contract": {"rows_sha256": "rows-hash"},
            "point_in_time_contract": {"dataset_join_keys": ["snapshot_id", "symbol"]},
        },
    )
    _write_json(tmp_path / "governance" / "research" / "multiple_testing_guard_latest.json", {"ok": True, "overall_status": "ready"})
    _write_json(tmp_path / "governance" / "research" / "decay_monitor_latest.json", {"overall_status": "ready", "weak_sleeve_count": 0})
    _write_json(health_root / "replay_hash_registry_guard_latest.json", {"ok": True})
    _write_json(
        health_root / "training_lineage_manifest_latest.json",
        {
            "lineage_contract_ready": True,
            "promotion_bundle_ready": False,
            "hash_bundle_complete": True,
            "feature_store_lineage_ok": True,
            "feature_store_schema_version": 1,
            "exact_replay_ready": True,
            "replay_hash_registry_ok": True,
            "lineage_score": 97.5,
        },
    )
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "recovery_state": "recovering_under_guard",
            "recovery_quality_score": 96.0,
            "backpressure": {"estimated_core_drain_minutes": None, "estimated_total_drain_minutes": None},
            "storage": {"retention_debt_gb": 0.0, "backlog_drain_status": "drain_active"},
            "bounded_recovery_contract": {"active": True},
            "top_actions": [],
        },
    )
    _write_json(
        health_root / "training_report_latest.json",
        {"overall_status": "blocked", "summary": {"confirmed_training_success": False, "target_count": 0, "trained_count": 0, "failure_count": 0}},
    )
    _write_json(health_root / "health_gates_latest.json", {"hard_gate_triggered": True})
    _write_json(health_root / "paper_performance_latest.json", {"sleeve_latest": []})
    _write_json(
        health_root / "roster_resilience_planner_latest.json",
        {"bench": {"bench_depth": 8}, "a_plus_contract": {"a_plus_ready": True}},
    )
    _write_json(health_root / "calibration_abstention_control_latest.json", {"overall_status": "ready"})

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] in {"needs_attention", "ready"}
    assert payload["immutable_lineage"]["lineage_contract_ready"] is True
    assert payload["rollout"]["exact_replay_ready"] is True
    assert payload["training_quality_index"] >= payload["training_quality_score"]
    assert payload["training_quality_base_score"] <= 100.0
    assert payload["training_quality_bonus_score"] >= 0.0
    assert payload["training_quality_score"] >= 80.0


def test_build_payload_counts_artifact_backed_active_supportability(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    walk_root = tmp_path / "governance" / "walk_forward"

    _write_json(
        health_root / "training_registry_audit_latest.json",
        {
            "registry_active_bots": 4,
            "supportability_counts": {"artifact_backed_active": 3, "supportable_active": 1},
            "tier_counts": {"active_stale": 3},
            "active_sample_starved": [],
            "active_quality_failed": [],
            "active_stale_diagnostics": [{"bot_id": "brain_refinery_v10_seasonal"}],
        },
    )
    _write_json(health_root / "training_label_audit_latest.json", {"top_actions": [], "recommendation_counts": {}})
    _write_json(
        health_root / "runtime_training_snapshot_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "row_count": 1000,
            "sequence_count": 12,
            "coverage": {"top_modes": [], "top_symbols": []},
        },
    )
    _write_json(
        walk_root / "promotion_readiness_latest.json",
        {"promote_ok": True, "considered_bots": 4, "thresholds": {"min_considered_bots": 4}},
    )
    _write_json(health_root / "promotion_quality_gate_latest.json", {"ok": True})
    _write_json(
        tmp_path / "governance" / "feature_store" / "latest.json",
        {
            "ok": True,
            "lineage_schema_version": 1,
            "dataset_contract": {"rows_sha256": "rows-hash"},
            "point_in_time_contract": {"dataset_join_keys": ["snapshot_id", "symbol"]},
        },
    )
    _write_json(tmp_path / "governance" / "research" / "multiple_testing_guard_latest.json", {"ok": True, "overall_status": "ready"})
    _write_json(tmp_path / "governance" / "research" / "decay_monitor_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "replay_hash_registry_guard_latest.json", {"ok": True})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {"overall_status": "ready", "backpressure": {}, "storage": {"retention_debt_gb": 0.0}},
    )
    experiment_path = tmp_path / "governance" / "experiments" / "experiment_registry.jsonl"
    experiment_path.parent.mkdir(parents=True, exist_ok=True)
    experiment_path.write_text(json.dumps({"experiment_id": "exp_ok", "replayability": {"exact_replay_ready": True}}) + "\n", encoding="utf-8")
    _write_json(health_root / "training_report_latest.json", {"overall_status": "ready", "summary": {"confirmed_training_success": True}})
    _write_json(health_root / "health_gates_latest.json", {"hard_gate_triggered": False})
    _write_json(health_root / "paper_performance_latest.json", {"sleeve_latest": []})
    _write_json(health_root / "roster_resilience_planner_latest.json", {"bench": {"bench_depth": 5}, "a_plus_contract": {"a_plus_ready": True}})
    _write_json(health_root / "calibration_abstention_control_latest.json", {"overall_status": "ready"})

    payload = src.build_payload(tmp_path)

    assert payload["supportability"]["active_supportable_bots"] == 4
    assert payload["supportability"]["active_supportability_score"] == 100.0


def test_build_payload_softens_guarded_storage_recovery_and_provisional_coverage_seed(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    walk_root = tmp_path / "governance" / "walk_forward"

    _write_json(
        health_root / "training_registry_audit_latest.json",
        {
            "registry_active_bots": 18,
            "supportability_counts": {"supportable_active": 16},
            "tier_counts": {"active_stale": 0, "research_candidate": 1},
            "active_sample_starved": [],
            "active_quality_failed": [],
            "active_stale_diagnostics": [],
        },
    )
    _write_json(health_root / "training_label_audit_latest.json", {"top_actions": [], "recommendation_counts": {}})
    _write_json(
        health_root / "runtime_training_snapshot_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "row_count": 2400,
            "sequence_count": 24,
            "coverage": {
                "top_modes": [
                    {"mode": "shadow_aggressive_equities", "row_count": 700},
                    {"mode": "shadow_conservative_equities", "row_count": 600},
                ],
                "top_symbols": [{"symbol": "SPY", "row_count": 40}],
            },
        },
    )
    _write_json(
        walk_root / "promotion_readiness_latest.json",
        {
            "promote_ok": False,
            "considered_bots": 0,
            "thresholds": {"min_considered_bots": 4},
        },
    )
    _write_json(health_root / "promotion_quality_gate_latest.json", {"ok": False})
    _write_json(
        tmp_path / "governance" / "feature_store" / "latest.json",
        {
            "ok": True,
            "lineage_schema_version": 1,
            "dataset_contract": {"rows_sha256": "rows-hash"},
            "point_in_time_contract": {"dataset_join_keys": ["snapshot_id", "symbol"]},
            "lane_partitions": [{"lane": "aggressive", "row_count": 700}],
        },
    )
    _write_json(
        tmp_path / "governance" / "research" / "multiple_testing_guard_latest.json",
        {
            "ok": True,
            "overall_status": "ready",
            "family_size": 18,
            "correction_method": "benjamini_hochberg_fdr",
        },
    )
    _write_json(tmp_path / "governance" / "research" / "decay_monitor_latest.json", {"overall_status": "ready", "weak_sleeve_count": 0})
    _write_json(health_root / "replay_hash_registry_guard_latest.json", {"ok": True})
    _write_json(
        health_root / "training_lineage_manifest_latest.json",
        {
            "lineage_contract_ready": True,
            "promotion_bundle_ready": False,
            "hash_bundle_complete": True,
            "feature_store_lineage_ok": True,
            "feature_store_schema_version": 1,
            "exact_replay_ready": True,
            "replay_hash_registry_ok": True,
            "multiple_testing_ready": True,
            "decay_monitor_ready": True,
            "lineage_score": 90.0,
        },
    )
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "degraded",
            "severity": "critical",
            "recovery_state": "recovering_under_guard",
            "backpressure": {"estimated_core_drain_minutes": None, "estimated_total_drain_minutes": None},
            "storage": {"retention_debt_gb": 0.0},
            "bounded_recovery_contract": {"active": True},
            "top_actions": ["keep draining backlog"],
        },
    )
    _write_json(
        health_root / "training_report_latest.json",
        {
            "overall_status": "needs_attention",
            "summary": {"confirmed_training_success": False, "target_count": 0, "trained_count": 0, "failure_count": 0},
        },
    )
    _write_json(health_root / "health_gates_latest.json", {"hard_gate_triggered": False})
    _write_json(health_root / "paper_performance_latest.json", {"sleeve_latest": []})
    _write_json(
        health_root / "roster_resilience_planner_latest.json",
        {"bench": {"bench_depth": 5}, "a_plus_contract": {"a_plus_ready": True}},
    )
    _write_json(health_root / "calibration_abstention_control_latest.json", {"overall_status": "ready"})
    experiment_path = tmp_path / "governance" / "experiments" / "experiment_registry.jsonl"
    experiment_path.parent.mkdir(parents=True, exist_ok=True)
    experiment_path.write_text(
        json.dumps({"experiment_id": "exp_ok", "replayability": {"exact_replay_ready": True}}) + "\n",
        encoding="utf-8",
    )

    payload = src.build_payload(tmp_path)
    improvement_by_key = {row["key"]: row for row in payload["improvements"]}

    assert payload["overall_status"] == "needs_attention"
    assert improvement_by_key["ingestion_drain_time_guard"]["status"] == "needs_work"
    assert improvement_by_key["continuous_coverage_seed"]["status"] == "needs_work"
    assert payload["improvement_status_counts"]["effective_blocked"] == 0
    assert payload["recoverable_blocked_keys"] == []


def test_build_payload_promotes_staged_supportability_and_strong_coverage_queue(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    walk_root = tmp_path / "governance" / "walk_forward"
    champion_root = tmp_path / "governance" / "champion_challenger"

    _write_json(
        health_root / "training_registry_audit_latest.json",
        {
            "registry_active_bots": 30,
            "supportability_counts": {
                "registry_seeded_active": 4,
                "staged_support_recovery": 21,
            },
            "tier_counts": {"active_stale": 25},
            "active_sample_starved": [],
            "active_quality_failed": [],
            "active_stale_diagnostics": [{"bot_id": f"bot_{idx}", "supportability_status": "staged_support_recovery"} for idx in range(21)]
            + [{"bot_id": f"strong_{idx}", "registry_quality_score": 0.6, "registry_test_accuracy": 0.76} for idx in range(4)],
            "active_registry_seeded": [{"bot_id": f"strong_{idx}"} for idx in range(4)],
            "active_staged_support_recovery": [{"bot_id": f"bot_{idx}"} for idx in range(21)],
        },
    )
    _write_json(health_root / "training_label_audit_latest.json", {"top_actions": [], "recommendation_counts": {}})
    _write_json(
        health_root / "runtime_training_snapshot_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "row_count": 2200,
            "sequence_count": 64,
            "coverage": {
                "top_modes": [
                    {"mode": "shadow_intraday_aggressive_equities", "row_count": 720},
                    {"mode": "shadow_conservative_equities", "row_count": 510},
                    {"mode": "shadow_dividend_equities", "row_count": 240},
                    {"mode": "shadow_fx_equities", "row_count": 180},
                    {"mode": "shadow_crypto_futures", "row_count": 160},
                    {"mode": "shadow_bond_equities", "row_count": 140},
                ],
                "top_symbols": [{"symbol": "SPY", "row_count": 40}],
            },
        },
    )
    _write_json(
        walk_root / "promotion_readiness_latest.json",
        {"promote_ok": False, "considered_bots": 0, "thresholds": {"min_considered_bots": 4}},
    )
    _write_json(
        walk_root / "coverage_seed_latest.json",
        {
            "seed_queue": [
                {"bot_id": "a", "test_accuracy": 0.82, "quality_score": 0.91, "strong_seed_candidate": True},
                {"bot_id": "b", "test_accuracy": 0.79, "quality_score": 0.74, "strong_seed_candidate": True},
                {"bot_id": "c", "test_accuracy": 0.76, "quality_score": 0.65, "strong_seed_candidate": True},
                {"bot_id": "d", "test_accuracy": 0.75, "quality_score": 0.59, "strong_seed_candidate": True},
            ],
            "coverage_shortfall_bots": 4,
        },
    )
    _write_json(
        walk_root / "coverage_gap_closer_latest.json",
        {
            "staged_candidate_count": 4,
            "autopilot_contract": {
                "can_apply_stage": True,
                "can_launch_now": False,
                "gating_signals": {"staged_candidates_present": True},
            },
        },
    )
    _write_json(health_root / "promotion_quality_gate_latest.json", {"ok": True})
    _write_json(
        champion_root / "promotion_packet_latest.json",
        {
            "packet_sha256": "packet-sha",
            "dataset": {"rows_sha256": "rows-hash"},
            "replayability_contract": {
                "dataset_hash": "dataset-hash",
                "model_hash": "model-hash",
                "replay_hash": "replay-hash",
                "bundle_hash": "bundle-hash",
            },
        },
    )
    _write_json(
        tmp_path / "governance" / "feature_store" / "latest.json",
        {
            "ok": True,
            "lineage_schema_version": 1,
            "dataset_contract": {"rows_sha256": "rows-hash"},
            "point_in_time_contract": {"dataset_join_keys": ["snapshot_id", "symbol"]},
            "lane_partitions": [{"lane": "aggressive", "row_count": 720}],
        },
    )
    _write_json(tmp_path / "governance" / "research" / "multiple_testing_guard_latest.json", {"ok": True, "overall_status": "ready"})
    _write_json(tmp_path / "governance" / "research" / "decay_monitor_latest.json", {"overall_status": "ready", "weak_sleeve_count": 0})
    _write_json(health_root / "replay_hash_registry_guard_latest.json", {"ok": True})
    _write_json(
        health_root / "training_lineage_manifest_latest.json",
        {
            "lineage_contract_ready": True,
            "promotion_bundle_ready": True,
            "hash_bundle_complete": True,
            "feature_store_lineage_ok": True,
            "exact_replay_ready": True,
            "replay_hash_registry_ok": True,
            "multiple_testing_ready": True,
            "decay_monitor_ready": True,
            "lineage_score": 100.0,
            "promotion_packet_ready": True,
        },
    )
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "severity": "critical",
            "recovery_state": "stabilized_recovery",
            "recovery_quality_score": 92.0,
            "backpressure": {"estimated_core_drain_minutes": None, "estimated_total_drain_minutes": None},
            "storage": {"retention_debt_gb": 0.0},
            "queue_watermarks": {"breaches": {"hard": []}},
            "ingestion_pressure": {
                "critical_priority_failures": [],
                "critical_priority_shard_storage_failures": [],
            },
            "top_actions": [],
        },
    )
    _write_json(
        health_root / "training_report_latest.json",
        {
            "overall_status": "needs_attention",
            "summary": {"confirmed_training_success": False, "target_count": 0, "trained_count": 0, "failure_count": 0},
        },
    )
    _write_json(
        health_root / "health_gates_latest.json",
        {
            "hard_gate_triggered": True,
            "hard_gates": {
                "ingestion_backpressure_overload": True,
                "collector_contracts": True,
                "sql_wal_pressure": True,
                "stale_windows": False,
                "ingestion_pending_lines": False,
                "ingestion_oldest_age": False,
                "ingestion_invalid_lines": False,
                "sql_progress_stall": False,
            },
        },
    )
    _write_json(health_root / "paper_performance_latest.json", {"sleeve_latest": []})
    _write_json(
        health_root / "roster_resilience_planner_latest.json",
        {"bench": {"bench_depth": 5}, "a_plus_contract": {"a_plus_ready": True}},
    )
    _write_json(health_root / "calibration_abstention_control_latest.json", {"overall_status": "ready"})

    payload = src.build_payload(tmp_path)
    improvement_by_key = {row["key"]: row for row in payload["improvements"]}

    assert payload["supportability"]["active_supportability_score"] >= 80.0
    assert payload["supportability"]["staged_support_recovery_bot_count"] == 21
    assert improvement_by_key["active_supportability"]["status"] == "ready"
    assert improvement_by_key["active_diagnostic_sla"]["status"] == "ready"
    assert improvement_by_key["stale_active_diagnostics"]["status"] == "ready"
    assert improvement_by_key["promotion_coverage"]["status"] == "ready"
    assert improvement_by_key["lane_dominance_cap"]["status"] == "ready"
    assert improvement_by_key["ingestion_health_guard"]["status"] == "ready"
    assert payload["rollout"]["coverage_quality_ready_count"] == 4


def test_build_payload_uses_stronger_provisional_accuracy_floor(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    walk_root = tmp_path / "governance" / "walk_forward"
    champion_root = tmp_path / "governance" / "champion_challenger"

    _write_json(
        health_root / "training_registry_audit_latest.json",
        {
            "registry_active_bots": 4,
            "supportability_counts": {},
            "tier_counts": {"active_stale": 4},
            "active_sample_starved": [],
            "active_quality_failed": [],
            "active_stale_diagnostics": [
                {"bot_id": "bot_a", "registry_quality_score": 0.31, "registry_test_accuracy": 0.74},
                {"bot_id": "bot_b", "registry_quality_score": 0.31, "registry_test_accuracy": 0.75},
                {"bot_id": "bot_c", "candidate_quality_score": 0.59, "candidate_test_accuracy": 0.76},
                {"bot_id": "bot_d", "registry_quality_score": 0.18, "registry_test_accuracy": 0.62},
            ],
        },
    )
    _write_json(health_root / "training_label_audit_latest.json", {"top_actions": [], "recommendation_counts": {}})
    _write_json(
        health_root / "runtime_training_snapshot_latest.json",
        {"timestamp_utc": now.isoformat(), "row_count": 1600, "sequence_count": 18, "coverage": {"top_modes": [], "top_symbols": []}},
    )
    _write_json(walk_root / "promotion_readiness_latest.json", {"promote_ok": True, "considered_bots": 4, "thresholds": {"min_considered_bots": 4}})
    _write_json(health_root / "promotion_quality_gate_latest.json", {"ok": True})
    _write_json(champion_root / "promotion_packet_latest.json", {"packet_sha256": "packet", "dataset": {"rows_sha256": "rows"}})
    _write_json(
        tmp_path / "governance" / "feature_store" / "latest.json",
        {"ok": True, "dataset_contract": {"rows_sha256": "rows"}, "point_in_time_contract": {"dataset_join_keys": ["snapshot_id"]}},
    )
    _write_json(tmp_path / "governance" / "research" / "multiple_testing_guard_latest.json", {"ok": True, "overall_status": "ready"})
    _write_json(tmp_path / "governance" / "research" / "decay_monitor_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "replay_hash_registry_guard_latest.json", {"ok": True})
    _write_json(health_root / "training_lineage_manifest_latest.json", {"lineage_contract_ready": True, "exact_replay_ready": True, "feature_store_lineage_ok": True, "lineage_score": 100.0})
    _write_json(health_root / "ingestion_storage_control_latest.json", {"overall_status": "ready", "storage": {"retention_debt_gb": 0.0}})
    _write_json(health_root / "training_report_latest.json", {"overall_status": "ready", "summary": {"confirmed_training_success": True}})
    _write_json(health_root / "health_gates_latest.json", {"hard_gate_triggered": False})
    _write_json(health_root / "paper_performance_latest.json", {"sleeve_latest": []})
    _write_json(health_root / "roster_resilience_planner_latest.json", {"bench": {"bench_depth": 5}, "a_plus_contract": {"a_plus_ready": True}})
    _write_json(health_root / "calibration_abstention_control_latest.json", {"overall_status": "ready"})

    payload = src.build_payload(tmp_path)

    assert payload["targeted_actions"]["provisional_registry_backed_bot_ids"] == ["bot_b", "bot_c"]
