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


def test_platform_stabilization_marks_duplicate_alpha_as_watch_when_novelty_contract_controls_overlap(tmp_path: Path) -> None:
    _seed_project(tmp_path)
    _write_json(
        tmp_path / "governance" / "health" / "platform_intelligence_expansion_latest.json",
        {
            "overall_status": "needs_work",
            "sections": {
                "bot_data_quality_scores": {"overall_status": "ready", "average_quality_score": 88.0},
                "duplicate_alpha_overlap_detector": {
                    "overall_status": "needs_work",
                    "overlap_cluster_count": 225,
                    "high_overlap_cluster_count": 26,
                    "novelty_contract": {"active": True, "review_required": True},
                },
                "execution_paper_trade_realism_layer": {"overall_status": "ready", "mae_bps": 12.0},
                "provider_rotation_failover_mesh": {"overall_status": "ready", "degraded_provider_count": 0},
            },
        },
    )

    payload = src.build_payload(tmp_path)
    duplicate = payload["sections"]["duplicate_alpha_compression"]

    assert duplicate["overall_status"] == "watch"
    assert duplicate["controlled_by_novelty_contract"] is True
    assert duplicate["high_overlap_cluster_count"] == 26


def test_platform_stabilization_marks_optional_provider_failures_as_watch(tmp_path: Path) -> None:
    _seed_project(tmp_path)
    _write_json(
        tmp_path / "governance" / "health" / "provider_mesh_latest.json",
        {
            "overall_status": "degraded",
            "summary": {
                "required_failure_count": 0,
                "soft_failure_count": 3,
            },
            "cooldowns": [],
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "source_verification_latest.json",
        {"overall_status": "degraded"},
    )

    payload = src.build_payload(tmp_path)
    provider = payload["sections"]["provider_cooldown_failover_v2"]

    assert provider["overall_status"] == "watch"
    assert provider["required_failure_count"] == 0
    assert provider["soft_failure_count"] == 3


def test_platform_stabilization_marks_quality_repair_queue_as_watch_when_training_quality_is_strong(tmp_path: Path) -> None:
    _seed_project(tmp_path)
    _write_json(
        tmp_path / "governance" / "health" / "platform_intelligence_expansion_latest.json",
        {
            "overall_status": "needs_work",
            "sections": {
                "bot_data_quality_scores": {"overall_status": "needs_work", "average_quality_score": 42.0, "label_counts": {"cold_start": 10}},
                "duplicate_alpha_overlap_detector": {"overall_status": "ready", "overlap_cluster_count": 0},
                "execution_paper_trade_realism_layer": {"overall_status": "ready", "mae_bps": 12.0},
                "provider_rotation_failover_mesh": {"overall_status": "ready", "degraded_provider_count": 0},
            },
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "bot_quality_autopilot_latest.json",
        {
            "overall_status": "needs_work",
            "quality_blockers": {
                "quality_probation_bot_ids": ["bot_a"],
                "targeted_retrain_bot_ids": ["bot_a"],
            },
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "training_quality_control_latest.json",
        {
            "overall_status": "needs_attention",
            "training_quality_score": 85.0,
            "recoverable_blocked_keys": [],
        },
    )

    payload = src.build_payload(tmp_path)
    quality = payload["sections"]["bot_data_quality_governor"]

    assert quality["overall_status"] == "watch"
    assert quality["training_quality_score"] == 85.0
    assert quality["quality_probation_count"] == 1
    assert quality["targeted_retrain_count"] == 1


def test_platform_stabilization_rolls_all_watch_sections_to_watch(tmp_path: Path) -> None:
    rows = [
        {"section": "backlog_drain_stabilizer", "overall_status": "ready"},
        {"section": "bot_data_quality_governor", "overall_status": "watch"},
        {"section": "duplicate_alpha_compression", "overall_status": "watch"},
        {"section": "paper_trade_realism_v2", "overall_status": "watch"},
        {"section": "provider_cooldown_failover_v2", "overall_status": "watch"},
        {"section": "ready_only_microtraining", "overall_status": "ready"},
        {"section": "expansion_rehearsal_gate", "overall_status": "ready"},
    ]

    assert src._worst_status(rows) == "watch"


def test_expansion_gate_clear_blocker_only_is_watch_not_repair_failure(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {"overall_status": "advisory", "host_saturation_score": 28.0, "compute_pressure_level": "normal", "memory_pressure_level": "normal"},
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.025,
            "backpressure": {"total_pending_lines": 100, "pending_lines_threshold": 15000},
        },
    )
    _write_json(
        health / "platform_settlement_stabilization_latest.json",
        {"overall_status": "watch", "sections": {"queue_decay_meter": {"queue_backpressure_active": False}}},
    )
    _write_json(health / "data_collection_observation_rollup_latest.json", {"collector_count": 10, "bots_with_observations": 10, "zero_observation_count": 0})
    _write_json(health / "global_halt_auto_clear_latest.json", {"halt": False, "clear_blockers": ["write_path_recovery_pending"]})
    _write_json(health / "swap_pressure_governor_latest.json", {"swap_pressure": {"tier": "normal", "swap_used_gb": 1.0}})
    brain = {
        "sections": {
            "strategic_roadmap_synthesizer": {"expansion_allowed_now": True},
            "scenario_rehearsal_lab": {"scenario_count": 5, "scenarios": []},
        }
    }

    gate = src._expansion_gate(tmp_path, brain, {"queue_backpressure_active": False})

    assert gate["overall_status"] == "watch"
    assert gate["repair_required_reasons"] == []
    assert gate["expansion_allowed_now"] is False


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


def test_platform_stabilization_uses_ready_storage_truth_over_stale_backpressure(tmp_path: Path) -> None:
    _seed_project(tmp_path)
    _write_json(
        tmp_path / "governance" / "health" / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 251486,
            "pending_lines_total": 251486,
            "oldest_pending_age_seconds": 4928.503,
        },
    )
    _write_json(
        tmp_path / "governance" / "health" / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.04,
            "backpressure": {
                "core_pending_lines": 65,
                "deferred_pending_lines": 11941,
                "cold_pending_lines": 0,
                "support_pending_lines": 1,
                "total_pending_lines": 12006,
                "pending_lines_threshold": 15000,
                "oldest_pending_age_seconds": 0.0,
            },
        },
    )

    payload = src.build_payload(tmp_path)
    metrics = payload["sections"]["backlog_drain_stabilizer"]["metrics"]

    assert metrics["storage_live_authoritative"] is True
    assert metrics["total_pending_lines"] == 12006
    assert metrics["oldest_pending_age_seconds"] == 0.0
    assert payload["sections"]["backlog_drain_stabilizer"]["queue_backpressure_active"] is False
