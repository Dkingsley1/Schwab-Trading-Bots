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
                    "target_functions": ["breakout_execution", "market_regime_router"],
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
                    "target_functions": ["dividend_quality", "market_regime_router"],
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
                    "target_functions": ["runtime_capacity", "backpressure_prediction"],
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
        {
            "overall_status": "ready",
            "pressure_index": 0.01,
            "backpressure": {
                "total_pending_lines": 2500,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 120,
                "oldest_age_threshold_seconds": 240,
                "estimated_total_drain_minutes": 10,
            },
        },
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
    _write_json(
        health / "data_ingress_latest_schwab_futures_equities_schwab.json",
        {
            "loop_state": "paused_market_data_provider_cooldown",
            "pause_gate": "market_data_provider_cooldown",
            "pause_reason": "provider_http_403_429",
            "total_counts": {"api_ok": 0, "api_error": 4},
        },
    )
    _write_json(health / "global_halt_auto_clear_latest.json", {"halt": False, "clear_ready": True})
    _write_json(health / "process_watchdog_latest.json", {"alerts": []})
    _write_json(health / "auth_lease_manager_latest.json", {"overall_status": "ready"})
    _write_json(health / "paper_400_ramp_latest.json", {"overall_status": "planned"})


def test_platform_intelligence_expansion_builds_all_twelve_primary_sections(tmp_path: Path) -> None:
    _seed_project(tmp_path)

    payload = src.build_payload(tmp_path, max_rows=10)

    assert payload["expansion_count"] == 12
    assert set(payload["primary_section_keys"]) == set(src.PRIMARY_SECTION_KEYS)
    assert set(payload["primary_sections"]) == set(src.PRIMARY_SECTION_KEYS)
    assert payload["sections"]["bot_admission_controller"]["overall_status"] == "protect_live"
    assert payload["sections"]["swap_cpu_capacity_planner"]["training_policy"] == "paused"
    assert payload["sections"]["swap_cpu_capacity_planner"]["max_new_collectors_now"] == 0
    assert payload["sections"]["provider_rotation_failover_mesh"]["degraded_provider_count"] == 1
    assert payload["sections"]["paper_trade_capacity_governor"]["live_execution_allowed"] is False
    assert payload["recommended_env_overrides"]["PLATFORM_INTELLIGENCE_LAYER_VERSION"] == "2"


def test_admission_downshifts_collection_and_blocks_training_under_swap(tmp_path: Path) -> None:
    _seed_project(tmp_path)

    payload = src.build_payload(tmp_path, max_rows=10)
    admissions = payload["sections"]["bot_admission_controller"]["sampled_admissions"]
    row = next(item for item in admissions if item["bot_id"] == "brain_refinery_intraday_aggressive_breakout_bot")

    assert row["collect_allowed"] is True
    assert row["collection_mode"] == "thin_sample"
    assert row["train_allowed"] is False
    assert row["admission_state"] == "defer_training_until_resource_pressure_clears"
    lifecycle = payload["sections"]["bot_lifecycle_manager"]["sampled_lifecycle"]
    lifecycle_row = next(item for item in lifecycle if item["bot_id"] == "brain_refinery_intraday_aggressive_breakout_bot")
    assert lifecycle_row["lifecycle_stage"] == "paper_ready_train_review"


def test_sleeve_masters_research_pipeline_decay_and_black_box_are_written(tmp_path: Path) -> None:
    _seed_project(tmp_path)

    payload = src.build_payload(tmp_path, max_rows=10)
    written = src.write_section_artifacts(tmp_path, payload)

    assert len(written) >= 20
    for path in written.values():
        assert Path(path).exists()
    masters = payload["sections"]["per_sleeve_master_bots"]["sleeve_masters"]
    assert any(row["sleeve"] == "intraday_aggressive" for row in masters)
    assert payload["sections"]["research_to_strategy_pipeline"]["stage_counts"]["paper_only_collecting"] >= 1
    assert payload["sections"]["model_decay_detector"]["decaying_bot_count"] >= 1
    assert payload["sections"]["cross_sleeve_correlation_governor"]["overall_status"] == "needs_work"
    assert payload["sections"]["system_black_box_recorder"]["captured_file_count"] > 0
