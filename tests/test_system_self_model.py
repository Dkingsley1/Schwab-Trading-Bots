from __future__ import annotations

import json
from pathlib import Path

from scripts.ops import system_self_model as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_system_self_model_builds_awareness_domains_and_optimizations(tmp_path: Path) -> None:
    rows = [
        {
            "bot_id": f"brain_refinery_v{i}",
            "active": True,
            "data_collection_active": i >= 2,
            "training_excluded": i >= 2,
            "lifecycle_state": "data_collection_only" if i >= 2 else "active",
            "sleeve_profile": f"sleeve_{i % 3}",
            "capability_pack_slug": "test_pack" if i >= 2 else "",
        }
        for i in range(1, 8)
    ]
    _write_json(tmp_path / "master_bot_registry.json", {"summary": {}, "sub_bots": rows})
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "memory_efficiency_control_latest.json",
        {
            "overall_status": "ready",
            "recommended_profile": "pro_balanced",
            "cotenant_awareness": {
                "mode": "managed_cotenant",
                "open_apps": ["PyCharm", "Chrome"],
                "memory_pressure_clear": True,
                "storage_pressure_clear": True,
            },
            "memory_snapshot": {"memory_pressure_state": "green", "memory_pressure_kind": "none", "swap_used_gb": 0.2},
            "storage_snapshot": {"pressure_index": 0.01},
            "expansion_session": {"pressure_level": "normal", "sleeve_profile_count": 3},
        },
    )
    _write_json(health / "runtime_throttle_control_latest.json", {"overall_status": "ready", "memory_pressure_level": "normal", "throttle_profile": "observe"})
    _write_json(health / "ingestion_storage_control_latest.json", {"overall_status": "ready", "severity": "stable", "pressure_index": 0.01})
    _write_json(
        health / "mlx_intelligence_router_latest.json",
        {
            "overall_status": "ready",
            "library_coverage": {"coverage_ratio": 1.0, "missing_count": 0},
            "route_coverage": {"route_coverage_ratio": 1.0, "blocked_lane_count": 0},
            "library_utilization_matrix": {"mapped_library_ratio": 1.0},
            "runtime_caps": {"profile": "foreground_safe", "max_concurrent_mlx_jobs": 2, "compile_mode": "canary_first", "heavy_vlm_enabled": True},
            "control_contract": {"safe_utilization_goal": "100_percent_library_coverage_with_memory_aware_caps"},
        },
    )
    _write_json(
        health / "library_utilization_router_latest.json",
        {
            "overall_status": "ready",
            "coverage": {
                "managed_non_mlx_package_count": 80,
                "locked_non_mlx_package_count": 75,
                "coverage_ratio": 1.0,
                "locked_runtime_ok_ratio": 1.0,
                "missing_runtime_count": 0,
                "version_mismatch_count": 0,
            },
            "library_utilization_matrix": {"mapped_package_ratio": 1.0},
            "runtime_caps": {"profile": "foreground_safe"},
            "control_contract": {
                "safe_utilization_goal": "100_percent_non_mlx_library_lane_coverage_with_runtime_caps",
                "default_ml_backend": "mlx",
                "portable_ml_policy": "pytorch_onnx_transformers_stay_canary_or_off_hours_when_live_collection_is_active",
            },
        },
    )
    _write_json(health / "global_killswitch_latest.json", {"halt": False, "action": "none", "reasons": []})
    _write_json(
        health / "operator_cockpit_latest.json",
        {
            "overall_status": "ready",
            "adaptive_posture": {"hard_blockers": [], "pressure_level": "normal"},
            "hardening_scorecard": {"process_ownership_canonical": True},
            "recommended_actions": ["watch queue"],
        },
    )
    _write_json(health / "core_bot_materialization_guard_latest.json", {"overall_status": "ready", "summary": {"missing_core_module_count": 0, "duplicate_core_version_count": 0}})
    _write_json(tmp_path / "governance" / "alerts" / "incident_auto_halt_latest.json", {"overall_status": "ready", "event": "none"})
    scripts_ops = tmp_path / "scripts" / "ops"
    scripts_ops.mkdir(parents=True, exist_ok=True)
    (scripts_ops / "mlx_intelligence_router.py").write_text(
        "LANE_SPECS = []\nlibrary_utilization_matrix = {}\nrecommended_runtime_env = {}\n",
        encoding="utf-8",
    )
    (scripts_ops / "library_utilization_router.py").write_text(
        "LANE_SPECS = []\nlibrary_utilization_matrix = {}\nLIBRARY_DEFAULT_ML_BACKEND = 'mlx'\n",
        encoding="utf-8",
    )
    (scripts_ops / "opsctl.sh").write_text("mlx-intelligence-router\nlibrary-utilization-router\n", encoding="utf-8")

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "ready"
    assert payload["identity"]["total_bots"] == 7
    assert payload["identity"]["data_collection_active_bots"] == 6
    assert payload["awareness_domains"]["resource_awareness"]["status"] == "advisory"
    assert payload["awareness_domains"]["mlx_intelligence_awareness"]["status"] == "ready"
    assert payload["awareness_domains"]["mlx_intelligence_awareness"]["library_coverage_ratio"] == 1.0
    assert payload["awareness_domains"]["library_utilization_awareness"]["status"] == "ready"
    assert payload["awareness_domains"]["library_utilization_awareness"]["mapped_package_ratio"] == 1.0
    assert payload["awareness_domains"]["library_utilization_awareness"]["default_ml_backend"] == "mlx"
    assert payload["awareness_domains"]["bot_awareness"]["status"] == "ready"
    assert payload["awareness_domains"]["failure_memory"]["status"] == "ready"
    assert payload["awareness_domains"]["dependency_awareness"]["edge_count"] >= 5
    assert payload["dependency_memory"]["edge_count"] >= 5
    assert payload["failure_memory_index"]["current_event_count"] >= 1
    assert payload["registry_diff_memory"]["diff_status"] == "baseline"
    assert payload["upgrade_optimizer"]["next_generation_backlog"]
    assert "mlx_compute_brain" in payload["upgrade_optimizer"]["implemented_lanes"]
    assert "library_utilization_brain" in payload["upgrade_optimizer"]["implemented_lanes"]
    assert payload["control_contract"]["platform_brain_mode"] == "big_platform_brain_operational_control_plane"
    assert len(payload["upgrades_and_optimizations"]) >= 6
    assert payload["control_contract"]["consciousness_claim"] == "none_operational_self_model_only"


def test_system_self_model_writes_json_and_markdown(tmp_path: Path) -> None:
    _write_json(tmp_path / "master_bot_registry.json", {"summary": {"total_bots": 1, "active_bots": 1}, "sub_bots": []})
    _write_json(tmp_path / "governance" / "health" / "memory_efficiency_control_latest.json", {"overall_status": "ready", "memory_snapshot": {"memory_pressure_state": "green", "memory_pressure_kind": "none"}})
    _write_json(tmp_path / "governance" / "health" / "ingestion_storage_control_latest.json", {"overall_status": "ready", "severity": "stable", "pressure_index": 0.0})

    payload = src.build_payload(tmp_path)
    out_path = tmp_path / "governance" / "health" / "system_self_model_latest.json"
    md_path = tmp_path / "exports" / "reports" / "operator" / "system_self_model_latest.md"
    brief_path = tmp_path / "exports" / "reports" / "operator" / "system_self_brief_latest.md"
    dependency_path = tmp_path / "governance" / "health" / "system_dependency_memory_latest.json"
    failure_path = tmp_path / "governance" / "health" / "system_failure_memory_latest.json"
    registry_diff_path = tmp_path / "governance" / "health" / "system_registry_diff_latest.json"
    upgrade_path = tmp_path / "governance" / "health" / "system_upgrade_optimizer_latest.json"
    src.write_outputs(
        payload,
        out_path=out_path,
        markdown_path=md_path,
        brief_path=brief_path,
        dependency_memory_path=dependency_path,
        failure_memory_path=failure_path,
        registry_diff_path=registry_diff_path,
        upgrade_plan_path=upgrade_path,
    )

    assert out_path.exists()
    assert md_path.exists()
    assert brief_path.exists()
    assert dependency_path.exists()
    assert failure_path.exists()
    assert registry_diff_path.exists()
    assert upgrade_path.exists()
    assert "# System Self Model" in md_path.read_text(encoding="utf-8")
    assert "# System Self Brief" in brief_path.read_text(encoding="utf-8")
