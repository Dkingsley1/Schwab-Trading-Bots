import json
from pathlib import Path

from scripts.ops.whole_system_safety_frontier_push import build_payload


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_whole_system_frontier_applies_12_domains_then_pauses(tmp_path: Path):
    health_root = tmp_path / "governance" / "health"
    _write_json(
        health_root / "safety_bounded_advancement_frontier_latest.json",
        {
            "overall_status": "frontier_control_plane_applied_pause_for_soak",
            "safety_stop_active": True,
            "safety_stop_reason": ["promotion_gate_blocked", "large_training_batch_running_control_plane_only"],
        },
    )
    _write_json(
        health_root / "quant_strategy_lane_upgrades_latest.json",
        {
            "overall_status": "collection_runtime_active_paper_activation_blocked",
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
        health_root / "health_fast_latest.json",
        {
            "storage": {"severity": "stable"},
            "runtime_pressure": {"overall_status": "ready"},
        },
    )
    _write_json(health_root / "retrain_launch_latest.json", {"state": "running", "pid": 123})

    payload = build_payload(tmp_path)

    assert payload["ok"] is True
    assert payload["overall_status"] == "whole_system_frontier_control_plane_applied_pause_for_soak"
    assert payload["domain_count"] == 12
    assert payload["control_plane_domain_count"] == 12
    assert payload["advisory_domain_count"] == 12
    assert payload["paper_rehearsal_domain_count"] == 12
    assert payload["live_advisory_domain_count"] == 12
    assert payload["safety_stop_active"] is True
    assert payload["paper_execution_authority_enabled"] is False
    assert payload["live_execution_authority_enabled"] is False
    assert payload["allocation_authority_enabled"] is False
    assert payload["training_intake_authority_enabled"] is False
    assert payload["new_collector_authority_enabled"] is False
    assert payload["heavy_replay_authority_enabled"] is False
    assert "heavy_replay_or_large_training" in payload["do_not_push_until_guard_clears"]
    assert all(domain["state"] == "applied_control_plane" for domain in payload["domains"])
    assert all(domain["paper_execution_authority_enabled"] is False for domain in payload["domains"])
    assert all(domain["live_execution_authority_enabled"] is False for domain in payload["domains"])
    assert all(domain["heavy_replay_authority_enabled"] is False for domain in payload["domains"])
