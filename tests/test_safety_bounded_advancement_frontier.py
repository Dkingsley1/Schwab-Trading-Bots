import json
from pathlib import Path

from scripts.ops.safety_bounded_advancement_frontier import build_payload


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_safety_bounded_frontier_pushes_control_plane_then_pauses(tmp_path: Path):
    health_root = tmp_path / "governance" / "health"
    _write_json(
        health_root / "quant_strategy_lane_upgrades_latest.json",
        {
            "overall_status": "collection_runtime_active_paper_activation_blocked",
            "collection_runtime_active": True,
            "gate_state": {
                "runtime_green": True,
                "storage_green": True,
                "paper_400_ready": True,
                "promotion_quality_ready": False,
                "promotion_quality_failed_checks": ["promotion_gate_blocked"],
                "promotion_readiness_blockers": ["insufficient_walk_forward_coverage"],
            },
        },
    )
    _write_json(
        health_root / "library_efficiency_deepening_latest.json",
        {
            "overall_status": "library_efficiency_layers_installed_dual_mode_activation_blocked",
            "efficiency_score": 1.0,
            "gate_state": {
                "runtime_green": True,
                "storage_green": True,
                "paper_400_ready": True,
                "promotion_quality_ready": False,
                "blockers": ["promotion_quality:promotion_gate_blocked"],
            },
        },
    )
    _write_json(
        health_root / "deep_quant_layer_upgrade_latest.json",
        {
            "overall_status": "deep_quant_layers_installed_collection_only_activation_blocked",
            "coverage_ratio": 1.0,
            "activation_blockers": ["promotion_quality_ready=false"],
        },
    )
    _write_json(health_root / "retrain_launch_latest.json", {"state": "running", "pid": 123})

    payload = build_payload(tmp_path)

    assert payload["ok"] is True
    assert payload["overall_status"] == "frontier_control_plane_applied_pause_for_soak"
    assert payload["stage_count"] == 10
    assert payload["control_plane_stage_count"] == 10
    assert payload["safety_stop_active"] is True
    assert payload["pause_kind"] == "soak_until_training_batch_and_promotion_evidence_clear"
    assert payload["paper_execution_authority_enabled"] is False
    assert payload["live_execution_authority_enabled"] is False
    assert payload["allocation_authority_enabled"] is False
    assert payload["training_intake_authority_enabled"] is False
    assert "large_training_batch_running_control_plane_only" in payload["safety_stop_reason"]
    assert all(stage["state"] == "applied_control_plane" for stage in payload["stages"])
    assert all(stage["paper_execution_authority_enabled"] is False for stage in payload["stages"])
    assert all(stage["live_execution_authority_enabled"] is False for stage in payload["stages"])
