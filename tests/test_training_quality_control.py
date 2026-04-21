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
            "active_sample_starved": [{"bot_id": "brain_refinery_v4_simple"}, {"bot_id": "brain_refinery_v13_choppy"}],
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
    assert "stale_active_diagnostics" in payload["top_priorities"]
    assert payload["targeted_actions"]["repair_runtime_input_bot_ids"] == [
        "brain_refinery_v4_simple",
        "brain_refinery_v13_choppy",
    ]
    assert payload["targeted_actions"]["quality_probation_bot_ids"] == [
        "brain_refinery_v43_intraday_ultrafast_proxy",
    ]
    assert payload["dataset_shape"]["lane_lookback_days"]["intraday_aggressive"] == 23
    assert payload["a_plus_contract"]["roster_a_plus_ready"] is False
    assert payload["rollout"]["promotion_confidence_ready"] is False
    assert payload["data_ops"]["health_gate_triggered"] is True
    assert payload["data_ops"]["ingestion_storage_status"] == "blocked"
    assert payload["research"]["multiple_testing_status"] == "blocked"


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
    assert payload["targeted_actions"]["targeted_retrain_bot_ids"] == []
    assert payload["rollout"]["exact_replay_ready"] is True
    assert payload["data_ops"]["retention_debt_gb"] == 0.0


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
