import json
from pathlib import Path

from scripts.ops import platform_brain_v4 as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _seed_project(project_root: Path) -> None:
    _write_json(
        project_root / "master_bot_registry.json",
        {
            "sub_bots": [
                {"bot_id": "alpha_unique_bot", "active": True, "data_collection_active": True, "lifecycle_state": "data_collection_only"},
                {"bot_id": "quality_ready_bot", "active": True, "data_collection_active": True, "lifecycle_state": "active"},
                {"bot_id": "inactive_legacy_bot", "active": False, "lifecycle_state": "inactive"},
            ]
        },
    )
    sections = {
        "bot_lifecycle_manager": {"overall_status": "ready", "lifecycle_counts": {"collecting": 2, "trainable": 1}},
        "bot_data_quality_scores": {"overall_status": "needs_work", "average_quality_score": 42.0, "label_counts": {"cold_start": 2, "watch": 1}},
        "provider_rotation_failover_mesh": {"overall_status": "needs_work", "degraded_provider_count": 1},
        "backpressure_prediction_engine": {"overall_status": "ready", "pending_ratio": 0.2},
        "duplicate_alpha_overlap_detector": {"overall_status": "needs_work", "overlap_cluster_count": 2},
        "paper_trade_capacity_governor": {"overall_status": "ready", "recommended_max_paper_bots_now": 50},
        "self_healing_incident_playbooks": {"overall_status": "needs_work", "triggered_count": 1},
        "per_sleeve_master_bots": {"overall_status": "ready", "sleeve_master_count": 2},
        "training_readiness_board": {"overall_status": "ready", "train_allowed_count": 1, "sample_debt_count": 2},
        "market_regime_router": {"overall_status": "ready", "regime_state": "mixed_transition"},
        "execution_paper_trade_realism_layer": {"overall_status": "needs_work", "mae_bps": 38},
        "system_black_box_recorder": {"overall_status": "ready", "captured_file_count": 8},
        "cross_sleeve_correlation_governor": {"overall_status": "needs_work"},
        "model_decay_detector": {"overall_status": "needs_work"},
    }
    _write_json(
        project_root / "governance" / "health" / "platform_intelligence_expansion_latest.json",
        {
            "overall_status": "degraded",
            "bot_count": 3,
            "top_actions": ["work the data-quality queue", "refresh execution realism"],
            "pressure_snapshot": {
                "overall_status": "degraded",
                "compute_policy": "sustain",
                "host_saturation_score": 64.0,
                "storage_pressure_index": 0.3,
            },
            "sections": sections,
        },
    )
    _write_json(project_root / "governance" / "health" / "memory_efficiency_control_latest.json", {"overall_status": "ready"})
    _write_json(project_root / "governance" / "health" / "creative_cotenant_guard_latest.json", {"overall_status": "ready"})


def test_platform_brain_v4_builds_all_twelve_sections(tmp_path: Path) -> None:
    _seed_project(tmp_path)

    payload = src.build_payload(tmp_path)

    assert payload["section_count"] == 12
    assert set(payload["section_keys"]) == set(src.SECTION_KEYS)
    assert payload["control_count"] == 12
    assert payload["sections"]["executive_meta_orchestrator"]["ranked_priority_count"] >= 1
    assert payload["sections"]["causal_world_model"]["causal_edge_count"] >= 3
    assert payload["sections"]["training_scheduler_brain"]["training_policy"] == "off_hours_micro_batches"
    assert payload["recommended_env_overrides"]["PLATFORM_BRAIN_V4_ENABLED"] == "1"
    assert payload["recommended_env_overrides"]["PRIMARY_ML_RUNTIME_BACKEND"] == "mlx"


def test_platform_brain_v4_writes_section_artifacts_and_memory(tmp_path: Path) -> None:
    _seed_project(tmp_path)

    payload = src.build_payload(tmp_path)
    written = src.write_section_artifacts(tmp_path, payload)
    memory_event = payload["sections"]["experience_memory_core_v2"]["latest_memory_event"]
    memory_path = tmp_path / "governance" / "platform_brain_v4" / "experience_memory" / "events.jsonl"

    assert len(written) == 12
    assert all(Path(path).exists() for path in written.values())
    assert src._append_memory_event(memory_path, memory_event) is True
    assert memory_path.read_text(encoding="utf-8").count("\n") == 1


def test_platform_brain_v4_simulates_expansion_and_prioritizes_actions(tmp_path: Path) -> None:
    _seed_project(tmp_path)

    payload = src.build_payload(tmp_path)
    simulator = payload["sections"]["predictive_expansion_simulator"]
    priority = payload["sections"]["autonomous_priority_ranker"]
    critics = payload["sections"]["critic_council"]

    assert [row["additional_bots"] for row in simulator["simulations"]] == [25, 100, 250]
    assert priority["priority_count"] >= 4
    assert critics["caution_count"] >= 3


def test_platform_brain_v4_rolls_caution_only_debt_to_watch(tmp_path: Path) -> None:
    _seed_project(tmp_path)

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "watch"
    assert payload["ok"] is True
    assert payload["sections"]["bot_portfolio_economist"]["overall_status"] == "watch"
    assert payload["sections"]["critic_council"]["overall_status"] == "watch"
    assert payload["sections"]["data_value_engine"]["overall_status"] == "watch"
    assert (
        payload["sections"]["critic_council"]["severity_policy"]
        == "caution_votes_hold_expansion_without_blocking_guarded_collection_or_paper"
    )


def test_platform_brain_v4_keeps_hard_provider_data_value_as_needs_work() -> None:
    section = src._data_value_engine(
        {
            "bot_data_quality_scores": {"average_quality_score": 20.0},
            "provider_rotation_failover_mesh": {"overall_status": "critical", "degraded_provider_count": 4},
            "execution_paper_trade_realism_layer": {"overall_status": "watch"},
        }
    )

    assert section["overall_status"] == "needs_work"
    assert section["severity_policy"] == "low_data_value_with_hard_provider_failure_requires_repair"


def test_platform_brain_v4_keeps_blocked_pressure_critic_hard() -> None:
    council = src._critic_council(
        {"bot_data_quality_scores": {"overall_status": "ready"}, "execution_paper_trade_realism_layer": {"overall_status": "ready"}},
        {"overall_status": "blocked"},
    )

    assert council["overall_status"] == "needs_work"
    assert council["severity_policy"] == "blocked_or_critical_pressure_keeps_critic_council_hard"
