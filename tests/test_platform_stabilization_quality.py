import json
from pathlib import Path

from scripts.ops import platform_stabilization_quality as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _seed_project(project_root: Path) -> None:
    _write_json(
        project_root / "master_bot_registry.json",
        {
            "sub_bots": [
                {"bot_id": "ready_intraday_bot", "active": True, "data_collection_active": True, "data_collection_training_ready": True},
                {"bot_id": "cold_macro_bot", "active": True, "data_collection_active": True, "training_excluded": True},
                {"bot_id": "inactive_old_bot", "active": False},
            ]
        },
    )
    _write_json(
        project_root / "governance" / "health" / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 22000,
            "pending_lines_total": 22300,
            "pending_lines_deferred": 300,
            "pending_lines_cold": 0,
            "pending_lines_support_telemetry": 10,
            "oldest_pending_age_seconds": 33.0,
        },
    )
    _write_json(
        project_root / "governance" / "health" / "ingestion_storage_control_latest.json",
        {
            "severity": "high",
            "pressure_index": 1.49,
            "backpressure": {
                "core_pending_lines": 22100,
                "deferred_pending_lines": 300,
                "cold_pending_lines": 0,
                "support_pending_lines": 10,
                "total_pending_lines": 22410,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 33.0,
            },
        },
    )
    _write_json(
        project_root / "governance" / "health" / "backpressure_drainer_fleet_latest.json",
        {
            "overall_status": "handoff_requested",
            "ready_drainer_count": 2,
            "active_drainer": {"name": "core_decision_drainer", "pending_lines": 22000},
        },
    )
    _write_json(
        project_root / "governance" / "health" / "platform_intelligence_expansion_latest.json",
        {
            "overall_status": "degraded",
            "sections": {
                "bot_data_quality_scores": {"overall_status": "needs_work", "average_quality_score": 42.0, "label_counts": {"cold_start": 2}},
                "duplicate_alpha_overlap_detector": {"overall_status": "needs_work", "overlap_cluster_count": 3},
                "execution_paper_trade_realism_layer": {"overall_status": "ready", "mae_bps": 12.0},
                "provider_rotation_failover_mesh": {"overall_status": "needs_work", "degraded_provider_count": 2},
            },
        },
    )
    _write_json(project_root / "governance" / "health" / "bot_quality_autopilot_latest.json", {"overall_status": "blocked"})
    _write_json(project_root / "governance" / "health" / "training_quality_control_latest.json", {"overall_status": "needs_work", "training_quality_score": 38.0})
    _write_json(project_root / "governance" / "health" / "data_collection_observation_rollup_latest.json", {"collector_count": 2, "bots_with_observations": 2, "zero_observation_count": 0})
    _write_json(project_root / "governance" / "health" / "paper_execution_calibration_latest.json", {"overall_status": "ready", "metrics": {"mae_bps": 12.0, "p95_bps": 55.0}})
    _write_json(project_root / "governance" / "health" / "execution_lab_latest.json", {"top_worst_case_scenarios": [{"slippage_bps": 42.0}]})
    _write_json(
        project_root / "governance" / "health" / "provider_mesh_latest.json",
        {"overall_status": "degraded", "summary": {"required_failure_count": 1, "soft_failure_count": 2}, "cooldowns": []},
    )
    _write_json(project_root / "governance" / "health" / "source_verification_latest.json", {"overall_status": "degraded"})
    _write_json(project_root / "governance" / "health" / "training_runtime_control_latest.json", {"overall_status": "degraded", "snapshot_ready": True})
    _write_json(
        project_root / "governance" / "health" / "platform_brain_v4_latest.json",
        {
            "overall_status": "needs_work",
            "sections": {
                "training_scheduler_brain": {"training_policy": "off_hours_micro_batches", "train_allowed_count": 2, "sample_debt_count": 10},
                "bot_portfolio_economist": {"trainable_bots": 2},
            },
        },
    )
    _write_json(
        project_root / "governance" / "health" / "platform_brain_v5_latest.json",
        {
            "overall_status": "needs_work",
            "sections": {
                "strategic_roadmap_synthesizer": {"expansion_allowed_now": False},
                "scenario_rehearsal_lab": {"scenario_count": 5, "scenarios": [{"scenario": "add_25_bots_now", "recommendation": "defer"}]},
            },
        },
    )


def test_platform_stabilization_quality_builds_seven_controls(tmp_path: Path) -> None:
    _seed_project(tmp_path)

    payload = src.build_payload(tmp_path)

    assert payload["section_count"] == 7
    assert set(payload["section_keys"]) == set(src.SECTION_KEYS)
    assert payload["control_count"] == 7
    assert payload["sections"]["backlog_drain_stabilizer"]["queue_backpressure_active"] is True
    assert payload["sections"]["provider_cooldown_failover_v2"]["degraded_provider_count"] == 3
    assert payload["sections"]["expansion_rehearsal_gate"]["expansion_allowed_now"] is False
    assert "queue_backpressure_active" in payload["sections"]["expansion_rehearsal_gate"]["gate_closed_reasons"]
    assert "storage_or_queue_not_settled" in payload["sections"]["expansion_rehearsal_gate"]["gate_closed_reasons"]
    assert payload["next_best_command"].startswith("./scripts/ops/opsctl.sh backpressure-drainers")
    assert payload["recommended_env_overrides"]["PRIMARY_ML_RUNTIME_BACKEND"] == "mlx"
    assert payload["recommended_env_overrides"]["EXPANSION_APPLY_ALLOWED"] == "0"
    assert payload["recommended_env_overrides"]["EXPANSION_CALM_GATE_ENABLED"] == "1"
    assert payload["recommended_env_overrides"]["EXPANSION_REQUIRE_RUNTIME_CALM"] == "1"
    assert payload["recommended_env_overrides"]["STABILITY_HARDENING_V2_ENABLED"] == "1"
    assert payload["recommended_env_overrides"]["HEALTH_ARTIFACT_COALESCE_ENABLED"] == "1"
    assert payload["recommended_env_overrides"]["BOT_COLLECTION_DUTY_CYCLE_ENABLED"] == "1"


def test_platform_stabilization_quality_writes_artifacts(tmp_path: Path) -> None:
    _seed_project(tmp_path)

    payload = src.build_payload(tmp_path)
    written = src.write_section_artifacts(tmp_path, payload)

    assert len(written) == 8
    assert all(Path(path).exists() for path in written.values())
    assert "infrabot_assignments" in written


def test_platform_stabilization_quality_keeps_training_ready_only(tmp_path: Path) -> None:
    _seed_project(tmp_path)

    payload = src.build_payload(tmp_path)
    training = payload["sections"]["ready_only_microtraining"]

    assert training["train_allowed_count"] == 2
    assert training["sample_debt_count"] == 10
    assert "train_ready_bots_only" in training["training_contract"]
    assert payload["recommended_env_overrides"]["TRAINING_READY_ONLY_MICROBATCH_ENABLED"] == "1"


def test_platform_stabilization_quality_requires_true_calm_before_expansion(tmp_path: Path) -> None:
    _seed_project(tmp_path)
    _write_json(
        tmp_path / "governance" / "health" / "runtime_throttle_control_latest.json",
        {"overall_status": "blocked", "host_saturation_score": 88.0, "compute_pressure_level": "elevated", "memory_pressure_level": "elevated"},
    )
    _write_json(
        tmp_path / "governance" / "health" / "global_halt_auto_clear_latest.json",
        {"halt": False, "halt_state": "clear_blocked", "clear_blockers": ["queue_backpressure_active"]},
    )

    payload = src.build_payload(tmp_path)
    gate = payload["sections"]["expansion_rehearsal_gate"]

    assert gate["overall_status"] == "blocked"
    assert gate["expansion_allowed_now"] is False
    assert "runtime_not_calm" in gate["gate_closed_reasons"]
    assert "global_clear_blockers_present" in gate["gate_closed_reasons"]
    assert gate["pre_expansion_snapshot"]["host_saturation_score"] == 88.0
    assert payload["recommended_env_overrides"]["EXPANSION_CALM_BLOCKERS"] != "none"
    assert payload["recommended_env_overrides"]["ROSTER_EXPANSION_ALLOWED"] == "0"
