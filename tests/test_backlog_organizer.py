from __future__ import annotations

import json
from pathlib import Path

from scripts.ops import backlog_organizer as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_backlog_organizer_allocates_blocking_lanes(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "master_bot_registry.json",
        {
            "sub_bots": [
                {
                    "bot_id": "brain_refinery_v100_example",
                    "active": True,
                    "lifecycle_state": "data_collection_only",
                    "data_collection_active": True,
                    "training_excluded": True,
                }
            ]
        },
    )
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {"overall_status": "degraded", "host_saturation_score": 80.0, "compute_pressure_level": "high"},
    )
    _write_json(
        health / "expansion_capacity_planner_latest.json",
        {
            "pressure_snapshot": {"admission_blocking_candidate_count": 12},
            "capacity_contract": {"rollout_mode": "protect_live_no_new_runtime_loops"},
        },
    )
    _write_json(
        health / "new_bot_admission_guard_latest.json",
        {
            "candidate_bot_count": 12,
            "blocking_candidate_count": 12,
            "top_actions": ["refresh replay hashes"],
        },
    )
    _write_json(
        health / "data_collection_observation_rollup_latest.json",
        {
            "overall_status": "ready",
            "collector_count": 1,
            "bots_with_observations": 1,
            "total_observations": 50,
            "training_ready_count": 0,
        },
    )
    _write_json(
        health / "runtime_gate_dashboard_latest.json",
        {"overall": {"attention": ["session_ready_missing", "promotion_not_ready"]}},
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "pressure_index": 0.333,
            "backpressure": {"total_pending_lines": 5129, "estimated_total_drain_minutes": 1000},
        },
    )

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "blocked"
    assert payload["summary"]["total_bots"] == 1
    assert payload["summary"]["blocking_lane_count"] >= 1
    lane_ids = [row["lane_id"] for row in payload["lanes"]]
    assert "runtime_pressure" in lane_ids
    assert "admission_contracts" in lane_ids
    assert "storage_backlog" in lane_ids
    assert payload["allocated_organizers"][0]["priority"] >= payload["allocated_organizers"][-1]["priority"]


def test_backlog_organizer_ready_when_gates_are_clear(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "master_bot_registry.json",
        {"sub_bots": [{"bot_id": "brain_refinery_v100_example", "active": True}]},
    )
    health = tmp_path / "governance" / "health"
    _write_json(health / "runtime_throttle_control_latest.json", {"overall_status": "ready"})
    _write_json(
        health / "expansion_capacity_planner_latest.json",
        {"capacity_contract": {"rollout_mode": "collection_only_wave_allowed"}},
    )
    _write_json(health / "new_bot_admission_guard_latest.json", {"candidate_bot_count": 0, "blocking_candidate_count": 0})
    _write_json(
        health / "data_collection_observation_rollup_latest.json",
        {"overall_status": "ready", "collector_count": 0, "training_ready_count": 0},
    )
    _write_json(health / "runtime_gate_dashboard_latest.json", {"overall_status": "ready", "overall": {"attention": []}})
    _write_json(health / "ingestion_storage_control_latest.json", {"overall_status": "ready", "backpressure": {"total_pending_lines": 0}})
    _write_json(health / "training_quality_control_latest.json", {"overall_status": "ready"})
    _write_json(health / "bot_quality_autopilot_latest.json", {"overall_status": "ready"})
    _write_json(health / "live_runtime_separation_control_latest.json", {"overall_status": "ready"})
    _write_json(health / "auth_lease_manager_latest.json", {"overall_status": "ready"})

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "ready"
    assert payload["ok"] is True


def test_backlog_organizer_allocates_drainer_self_accommodation_lane(tmp_path: Path) -> None:
    _write_json(tmp_path / "master_bot_registry.json", {"sub_bots": []})
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "pressure_index": 0.41,
            "backpressure": {"total_pending_lines": 24000, "estimated_total_drain_minutes": 180.0},
        },
    )
    _write_json(
        health / "backpressure_drainer_fleet_latest.json",
        {
            "overall_status": "ready",
            "ready_drainer_count": 3,
            "active_drainer": {"name": "settlement_reconciliation_drainer"},
            "self_accommodation": {
                "mode": "preview_ready",
                "next_safe_action": "run_backpressure_drainer_fleet_apply_or_bounded_super_drainer_wave",
            },
        },
    )
    _write_json(
        health / "backpressure_super_drainer_latest.json",
        {
            "overall_status": "applied_with_followups",
            "summary": {"waves_run": 1, "progress_waves": 1, "stop_reason": "max_waves_reached"},
            "grandmaster_context_packet": {"safe_next_action": "run_next_bounded_wave"},
        },
    )

    payload = src.build_payload(tmp_path)

    lanes = {row["lane_id"]: row for row in payload["lanes"]}
    assert "drainer_self_accommodation" in lanes
    accommodation = lanes["drainer_self_accommodation"]
    assert accommodation["status"] == "needs_work"
    assert any("fleet_active_drainer=settlement_reconciliation_drainer" == item for item in accommodation["evidence"])
    assert any("super_stop_reason=max_waves_reached" == item for item in accommodation["evidence"])
    storage = lanes["storage_backlog"]
    assert any("active_drainer=settlement_reconciliation_drainer" == item for item in storage["evidence"])
    assert any(command[1] == "backpressure-super-drainer" and "--apply" in command for command in storage["next_commands"])
