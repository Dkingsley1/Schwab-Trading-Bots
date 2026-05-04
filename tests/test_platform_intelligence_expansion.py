import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.ops import platform_intelligence_expansion as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _seed_project(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    _write_json(
        tmp_path / "master_bot_registry.json",
        {
            "sub_bots": [
                {
                    "bot_id": "brain_refinery_intraday_aggressive_breakout_bot",
                    "bot_role": "signal_sub_bot",
                    "active": True,
                    "lifecycle_state": "data_collection_only",
                    "quality_score": 0.62,
                    "test_accuracy": 0.59,
                    "data_collection_active": True,
                    "data_collection_observations": 1200,
                    "minimum_training_observations": 1000,
                    "data_collection_started_utc": (now - timedelta(days=10)).isoformat(),
                    "minimum_data_collection_days": 7,
                    "training_candidate_after_threshold": True,
                    "exclude_from_training": True,
                    "paper_trade_lock_required": True,
                    "resource_throttle_aware": True,
                    "global_halt_aware": True,
                    "direct_execution_allowed": False,
                    "no_improvement_streak": 4,
                    "correlation_dependencies": ["cross_sleeve_correlation_matrix"],
                    "correlation_peer_sleeves": ["aggressive", "futures"],
                },
                {
                    "bot_id": "brain_refinery_dividend_defensive_quality_bot",
                    "bot_role": "signal_sub_bot",
                    "active": True,
                    "lifecycle_state": "active",
                    "quality_score": 0.81,
                    "test_accuracy": 0.64,
                    "paper_trade_lock_required": True,
                    "resource_throttle_aware": True,
                    "global_halt_aware": True,
                    "direct_execution_allowed": False,
                    "correlation_dependencies": ["cross_sleeve_correlation_matrix"],
                    "correlation_peer_sleeves": ["conservative"],
                },
                {
                    "bot_id": "brain_refinery_platform_infrastructure_guard_bot",
                    "bot_role": "infrastructure_sub_bot",
                    "active": True,
                    "lifecycle_state": "data_collection_only",
                    "quality_score": 0.25,
                    "candidate_test_accuracy": 0.0,
                    "data_collection_active": True,
                    "data_collection_observations": 50,
                    "minimum_training_observations": 1000,
                    "data_collection_started_utc": (now - timedelta(days=1)).isoformat(),
                    "minimum_data_collection_days": 7,
                    "paper_trade_lock_required": True,
                    "resource_throttle_aware": True,
                    "global_halt_aware": True,
                    "correlation_dependencies": ["resource_profile"],
                    "correlation_peer_sleeves": ["system_governor_expansion"],
                },
            ]
        },
    )
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "swap_pressure_governor_latest.json",
        {"swap_pressure": {"tier": "pause_research", "swap_used_gb": 19.1}},
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {"overall_status": "blocked", "host_saturation_score": 88.0},
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {"overall_status": "ready", "pressure_index": 0.01},
    )
    _write_json(
        health / "memory_efficiency_control_latest.json",
        {"overall_status": "blocked", "recommended_profile": "constrained"},
    )
    _write_json(
        health / "regime_control_plane_latest.json",
        {"overall_status": "ready", "regime_state": "risk_off_shock", "stance_label": "bearish", "stance_score": -0.4},
    )
    _write_json(
        health / "paper_execution_calibration_latest.json",
        {"metrics": {"mae_bps": 8.0, "poor_or_fair_fill_count": 1}},
    )
    _write_json(
        health / "execution_lab_latest.json",
        {"top_worst_case_scenarios": [{"slippage_bps": 18.0}]},
    )
    _write_json(
        tmp_path / "governance" / "allocator" / "portfolio_capacity_curve_latest.json",
        {"summary": {"constrained_curve_count": 0}},
    )
    _write_json(
        tmp_path / "governance" / "research" / "decay_monitor_latest.json",
        {"overall_status": "needs_work", "weak_sleeves": [{"profile": "intraday_aggressive"}]},
    )


def test_platform_intelligence_expansion_builds_all_ten_sections(tmp_path: Path) -> None:
    _seed_project(tmp_path)

    payload = src.build_payload(tmp_path, max_rows=10)

    assert payload["expansion_count"] == 10
    assert set(payload["sections"]) == {
        "bot_admission_controller",
        "per_sleeve_master_bots",
        "bot_quality_score_system",
        "execution_realism_engine",
        "market_regime_router",
        "swap_cpu_capacity_planner",
        "research_to_strategy_pipeline",
        "cross_sleeve_correlation_governor",
        "model_decay_detector",
        "professional_system_dashboard",
    }
    assert payload["sections"]["bot_admission_controller"]["overall_status"] == "protect_live"
    assert payload["sections"]["swap_cpu_capacity_planner"]["training_policy"] == "paused"
    assert payload["sections"]["swap_cpu_capacity_planner"]["max_new_collectors_now"] == 0


def test_admission_downshifts_collection_and_blocks_training_under_swap(tmp_path: Path) -> None:
    _seed_project(tmp_path)

    payload = src.build_payload(tmp_path, max_rows=10)
    admissions = payload["sections"]["bot_admission_controller"]["sampled_admissions"]
    row = next(item for item in admissions if item["bot_id"] == "brain_refinery_intraday_aggressive_breakout_bot")

    assert row["collect_allowed"] is True
    assert row["collection_mode"] == "thin_sample"
    assert row["train_allowed"] is False
    assert row["admission_state"] == "defer_training_until_resource_pressure_clears"


def test_sleeve_masters_research_pipeline_and_decay_are_written(tmp_path: Path) -> None:
    _seed_project(tmp_path)

    payload = src.build_payload(tmp_path, max_rows=10)
    written = src.write_section_artifacts(tmp_path, payload)

    assert len(written) == 10
    for path in written.values():
        assert Path(path).exists()
    masters = payload["sections"]["per_sleeve_master_bots"]["sleeve_masters"]
    assert any(row["sleeve"] == "intraday_aggressive" for row in masters)
    assert payload["sections"]["research_to_strategy_pipeline"]["stage_counts"]["paper_only_collecting"] >= 1
    assert payload["sections"]["model_decay_detector"]["decaying_bot_count"] >= 1
    assert payload["sections"]["cross_sleeve_correlation_governor"]["overall_status"] == "needs_work"
