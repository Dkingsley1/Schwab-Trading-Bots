import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import expansion_capacity_planner_bot as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _seed_registry(project_root: Path) -> None:
    _write_json(
        project_root / "master_bot_registry.json",
        {
            "sub_bots": [
                {
                    "bot_id": "brain_refinery_v672_dcc_garch_correlation_bot",
                    "bot_role": "signal_sub_bot",
                    "active": True,
                    "lifecycle_state": "data_collection_only",
                    "training_excluded": True,
                    "sleeve_profile": "tail_dependency_risk",
                },
                {
                    "bot_id": "brain_refinery_v673_evt_peaks_over_threshold_tail_bot",
                    "bot_role": "signal_sub_bot",
                    "active": True,
                    "lifecycle_state": "data_collection_only",
                    "exclude_from_training": True,
                    "sleeve_profile": "tail_dependency_risk",
                },
            ]
        },
    )


def _seed_health(
    project_root: Path,
    *,
    halt: bool = False,
    swap_tier: str = "normal",
    swap_gb: float = 2.0,
    runtime_status: str = "ready",
    memory_status: str = "ready",
    storage_status: str = "ready",
    admission_blocking: int = 0,
) -> None:
    health = project_root / "governance" / "health"
    _write_json(health / "global_killswitch_latest.json", {"global_halt_active": halt})
    _write_json(health / "swap_pressure_governor_latest.json", {"swap_pressure": {"tier": swap_tier, "swap_used_gb": swap_gb}})
    _write_json(health / "runtime_throttle_control_latest.json", {"overall_status": runtime_status, "host_saturation_score": 22.0})
    _write_json(health / "memory_efficiency_control_latest.json", {"overall_status": memory_status})
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": storage_status,
            "pressure_index": 0.02,
            "raw_live_expansion_contract": {
                "active": False,
                "expansion_ready": True,
                "grade": "A+",
                "expansion_tier": "ready_for_bigger_expansion",
                "pressure_ratio": 0.20,
                "estimated_expansion_headroom": {"estimated_new_bot_headroom": 100},
                "raw_live": {"core_pending_lines": 500, "total_pending_lines": 900},
            },
            "backlog_truth": {
                "raw_live": {"grade": "A+", "pressure_ratio": 0.08},
                "sql_overlay": {"grade": "A+", "pressure_ratio": 0.10, "total_pending_lines": 1000, "core_pending_lines": 500},
            },
        },
    )
    _write_json(health / "data_collection_storage_guard_latest.json", {"overall_status": storage_status})
    _write_json(health / "provider_mesh_latest.json", {"overall_status": "ready", "summary": {"required_failure_count": 0, "soft_failure_count": 0}})
    _write_json(
        health / "source_verification_latest.json",
        {"overall_status": "ready", "overall": {"all_verified": True, "counts": {"single_source_unverified": 0}, "unverified_sources": []}},
    )
    _write_json(health / "data_collection_observation_rollup_latest.json", {"data_quality_score": 96.0})
    _write_json(health / "new_bot_admission_guard_latest.json", {"candidate_bot_count": 3, "blocking_candidate_count": admission_blocking})
    _write_json(health / "runtime_gate_dashboard_latest.json", {"overall_status": "ready"})


def test_expansion_capacity_allows_collection_only_wave_when_pressure_is_ready(tmp_path: Path) -> None:
    _seed_registry(tmp_path)
    _seed_health(tmp_path)

    payload = src.build_payload(tmp_path, requested_wave_size=20)

    assert payload["overall_status"] == "ready"
    assert payload["capacity_contract"]["recommended_wave_size_now"] == 20
    assert payload["capacity_contract"]["rollout_mode"] == "collection_only_wave_allowed"
    assert payload["clean_scaling_contract"]["overall_status"] == "ready"
    assert payload["clean_scaling_contract"]["mode"] == "clean_collection_wave"
    assert payload["capacity_contract"]["next_bot_id_range"]["start"] == "brain_refinery_v674"
    assert "brain_refinery_v674_expansion_capacity_planner_bot" in payload["support_infrabots"]
    assert payload["growth_invariants"][0] == "new bots enter as data_collection_only"


def test_expansion_capacity_blocks_new_runtime_when_halt_or_swap_pressure_active(tmp_path: Path) -> None:
    _seed_registry(tmp_path)
    _seed_health(tmp_path, halt=True, swap_tier="pause_research", swap_gb=21.0, admission_blocking=4)

    payload = src.build_payload(tmp_path, requested_wave_size=20)

    assert payload["overall_status"] == "blocked"
    assert payload["capacity_contract"]["max_new_collectors_now"] == 0
    assert payload["capacity_contract"]["recommended_wave_size_now"] == 0
    assert payload["capacity_contract"]["rollout_mode"] == "blocked_clean_scaling_no_new_runtime_loops"
    assert "admission_evidence" in payload["clean_scaling_contract"]["blocked_dimensions"]
    assert "global_halt_active" in payload["pressure_snapshot"]["blocking_reasons"]
    assert "clear new-bot admission contracts before allowing any of the expanded roster into training" in payload["recommended_actions"]


def test_expansion_capacity_blocks_when_queue_or_green_room_gate_is_closed(tmp_path: Path) -> None:
    _seed_registry(tmp_path)
    _seed_health(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "high",
            "pressure_index": 1.4,
            "backpressure": {"total_pending_lines": 22000, "pending_lines_threshold": 15000},
            "raw_live_expansion_contract": {
                "active": False,
                "expansion_ready": True,
                "grade": "A+",
                "expansion_tier": "ready_for_bigger_expansion",
                "pressure_ratio": 0.2,
                "estimated_expansion_headroom": {"estimated_new_bot_headroom": 100},
            },
            "backlog_truth": {
                "raw_live": {"grade": "A+", "pressure_ratio": 0.1},
                "sql_overlay": {"grade": "A+", "pressure_ratio": 0.2, "total_pending_lines": 1000, "core_pending_lines": 500},
            },
        },
    )
    _write_json(health / "ingestion_backpressure_latest.json", {"pending_lines_total": 22000})
    _write_json(
        health / "platform_stabilization_quality_latest.json",
        {"sections": {"expansion_rehearsal_gate": {"expansion_allowed_now": False}}},
    )
    _write_json(
        health / "platform_settlement_stabilization_latest.json",
        {"sections": {"queue_decay_meter": {"queue_backpressure_active": True}}},
    )

    payload = src.build_payload(tmp_path, requested_wave_size=20)

    assert payload["overall_status"] == "blocked"
    assert payload["capacity_contract"]["max_new_collectors_now"] == 0
    assert "queue_backpressure_active" in payload["pressure_snapshot"]["blocking_reasons"]
    assert "pre_expansion_stabilization_gate_closed" in payload["pressure_snapshot"]["blocking_reasons"]
    assert payload["pressure_snapshot"]["pending_ratio"] > 1.0


def test_expansion_capacity_blocks_broad_wave_when_clean_scaling_contract_fails(tmp_path: Path) -> None:
    _seed_registry(tmp_path)
    _seed_health(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "pressure_index": 0.02,
            "raw_live_expansion_contract": {
                "active": True,
                "expansion_ready": False,
                "grade": "B",
                "expansion_tier": "blocked_until_raw_live_cools",
                "pressure_ratio": 1.4,
                "estimated_expansion_headroom": {"estimated_new_bot_headroom": 0},
                "raw_live": {"core_pending_lines": 6500, "total_pending_lines": 7200},
            },
            "backlog_truth": {
                "raw_live": {"grade": "A+", "pressure_ratio": 0.48},
                "sql_overlay": {"grade": "F", "pressure_ratio": 9.0, "total_pending_lines": 140000, "core_pending_lines": 12000},
            },
            "storage_efficiency_contract": {
                "active": True,
                "control_env_recommendations": {"BOT_STORAGE_ALLOW_EXPANSION": "0"},
            },
        },
    )

    payload = src.build_payload(tmp_path, requested_wave_size=20)

    assert payload["overall_status"] == "blocked"
    assert payload["clean_scaling_contract"]["overall_status"] == "blocked"
    assert "raw_live_headroom" in payload["clean_scaling_contract"]["blocked_dimensions"]
    assert "sql_overlay_tail_debt" in payload["clean_scaling_contract"]["blocked_dimensions"]
    assert "storage_efficiency" in payload["clean_scaling_contract"]["blocked_dimensions"]
    assert payload["capacity_contract"]["recommended_wave_size_now"] == 0
    assert payload["capacity_contract"]["rollout_mode"] == "blocked_clean_scaling_no_new_runtime_loops"


def test_expansion_capacity_grades_controlled_recovery_as_c(tmp_path: Path) -> None:
    _seed_registry(tmp_path)
    _seed_health(tmp_path)
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "overall_status": "blocked",
            "host_saturation_score": 60.0,
            "compute_pressure_level": "elevated",
            "memory_pressure_level": "normal",
        },
    )
    _write_json(
        health / "provider_mesh_latest.json",
        {"overall_status": "degraded", "summary": {"required_failure_count": 0, "soft_failure_count": 1}},
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "degraded",
            "pressure_index": 0.8,
            "raw_live_expansion_contract": {
                "active": True,
                "expansion_ready": False,
                "grade": "B",
                "expansion_tier": "blocked_until_raw_live_cools",
                "pressure_ratio": 1.4,
                "estimated_expansion_headroom": {"estimated_new_bot_headroom": 400},
                "raw_live": {"core_pending_lines": 2200, "total_pending_lines": 2700},
            },
            "backlog_truth": {
                "raw_live": {"grade": "B", "pressure_ratio": 1.4},
                "sql_overlay": {"grade": "F", "pressure_ratio": 200.0, "total_pending_lines": 90000, "core_pending_lines": 2400},
            },
            "storage_efficiency_contract": {
                "active": True,
                "control_env_recommendations": {
                    "BOT_STORAGE_PLANE_PHASE": "manifest_only_recovery",
                    "BOT_STORAGE_ALLOW_EXPANSION": "0",
                    "BOT_STORAGE_ALLOW_TRAINING": "0",
                    "BOT_STORAGE_EXTERNAL_FREE_GB": "66.0",
                    "BOT_STORAGE_EXTERNAL_MIN_FREE_GB": "32.0",
                    "BOT_STORAGE_SPACE_RECOVERY_DEFICIT_GB": "0.0",
                },
            },
        },
    )

    payload = src.build_payload(tmp_path, requested_wave_size=20)

    assert payload["overall_status"] == "blocked"
    assert payload["clean_scaling_contract"]["grade"] == "C"
    assert payload["clean_scaling_contract"]["score"] == 76.0
    assert payload["clean_scaling_contract"]["blocked_dimensions"] == []
    assert set(payload["clean_scaling_contract"]["watch_dimensions"]) == {
        "raw_live_headroom",
        "sql_overlay_tail_debt",
        "storage_efficiency",
        "runtime_headroom",
    }
    assert payload["capacity_contract"]["recommended_wave_size_now"] == 0
    assert payload["capacity_contract"]["rollout_mode"] == "protect_live_no_new_runtime_loops"
