import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import live_canary_control as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_live_canary_control_reports_ready_when_supervised_canary_is_fully_clear(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    champion = tmp_path / "governance" / "champion_challenger"
    _write_json(health / "broker_readiness_latest.json", {"ready_for_open": True})
    _write_json(health / "session_ready_latest.json", {"ready": True})
    _write_json(health / "storage_route_status_latest.json", {"ok": True, "mode": "external"})
    _write_json(health / "execution_lane_live_latest.json", {"stale": False})
    _write_json(
        health / "live_runtime_separation_control_latest.json",
        {"overall_status": "ready", "clearance_plan": {"clearance_state": "cleared"}, "release_contract": {"live_lane_should_be_read_only": False}},
    )
    _write_json(
        champion / "promotion_autopilot_packet_latest.json",
        {"overall_status": "ready", "autopilot_state": "awaiting_approval", "promotion_ready": True},
    )
    _write_json(health / "canary_auto_tuner_latest.json", {"target_canary_max_weight": 0.01})
    _write_json(health / "canary_rollout_latest.json", {"eligible": True, "promote_canary": True, "applied_weight": 0.01})

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "ready"
    assert payload["supervised_canary_ready"] is True
    assert payload["recommended_mode"] == "supervised_canary"
    assert payload["promotion_packet_preclearance_ready"] is True


def test_live_canary_control_blocks_when_faithful_live_money_contract_is_not_ready(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    champion = tmp_path / "governance" / "champion_challenger"
    _write_json(health / "broker_readiness_latest.json", {"ready_for_open": True})
    _write_json(health / "session_ready_latest.json", {"ready": True})
    _write_json(health / "storage_route_status_latest.json", {"ok": True, "mode": "external"})
    _write_json(health / "execution_lane_live_latest.json", {"stale": False})
    _write_json(
        health / "live_runtime_separation_control_latest.json",
        {"overall_status": "ready", "clearance_plan": {"clearance_state": "cleared"}, "release_contract": {"live_lane_should_be_read_only": False}},
    )
    _write_json(
        champion / "promotion_autopilot_packet_latest.json",
        {"overall_status": "ready", "autopilot_state": "awaiting_approval", "promotion_ready": True},
    )
    _write_json(health / "canary_auto_tuner_latest.json", {"target_canary_max_weight": 0.01})
    _write_json(health / "canary_rollout_latest.json", {"eligible": True, "promote_canary": True, "applied_weight": 0.01})
    _write_json(
        health / "live_money_readiness_contract_latest.json",
        {
            "policy_id": "faithful_live_money_a_grade_20260826",
            "faithful_live_money_ready": False,
            "target_date": "2026-08-26",
            "days_remaining": 56,
            "blocking_reasons": ["target_window_not_complete", "decision_replay_harness_below_A"],
        },
    )

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "blocked"
    assert payload["supervised_canary_ready"] is False
    assert payload["live_money_contract_enforced"] is True
    assert payload["live_money_contract_ready"] is False
    assert "faithful_live_money_contract_not_ready" in payload["blocking_reasons"]


def test_live_canary_control_surfaces_packet_preclearance_when_only_seeded_committee_packet_exists(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    champion = tmp_path / "governance" / "champion_challenger"
    _write_json(health / "broker_readiness_latest.json", {"ready_for_open": True})
    _write_json(health / "session_ready_latest.json", {"ready": True})
    _write_json(health / "storage_route_status_latest.json", {"ok": True, "mode": "external"})
    _write_json(health / "execution_lane_live_latest.json", {"stale": True})
    _write_json(
        health / "live_runtime_separation_control_latest.json",
        {
            "overall_status": "degraded",
            "clearance_plan": {"clearance_state": "awaiting_cold_lane"},
            "release_contract": {"live_lane_should_be_read_only": True},
        },
    )
    _write_json(
        champion / "promotion_autopilot_packet_latest.json",
        {
            "overall_status": "degraded",
            "autopilot_state": "assembling_packet",
            "promotion_ready": False,
            "committee_packet_seed_ready": True,
            "signability_contract": {"committee_packet_seed_ready": True},
            "readiness_repair_contract": {"critical_repair_gate_count": 4},
        },
    )
    _write_json(health / "canary_auto_tuner_latest.json", {"target_canary_max_weight": 0.01})
    _write_json(health / "canary_rollout_latest.json", {"eligible": False, "promote_canary": False, "applied_weight": 0.01})

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "degraded"
    assert payload["staged_preclearance_ready"] is True
    assert payload["preapproved_supervised_ready"] is True
    assert payload["promotion_packet_ready"] is False
    assert payload["promotion_packet_preclearance_ready"] is True
    assert payload["runtime_clearance_recoverable"] is True
    assert payload["recommended_mode"] == "preapproved_supervised"
    assert "promotion_packet_preclearance_only" in payload["blocking_reasons"]
    assert payload["preclearance_score"] >= 80.0


def test_live_canary_control_treats_managed_coverage_stage_as_recoverable(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    champion = tmp_path / "governance" / "champion_challenger"
    _write_json(health / "broker_readiness_latest.json", {"ready_for_open": True})
    _write_json(health / "session_ready_latest.json", {"ready": True})
    _write_json(health / "storage_route_status_latest.json", {"ok": True, "mode": "external"})
    _write_json(health / "execution_lane_live_latest.json", {"stale": True})
    _write_json(
        health / "live_runtime_separation_control_latest.json",
        {
            "overall_status": "ready",
            "clearance_plan": {"clearance_state": "managed_coverage_stage_deferred"},
            "release_contract": {"live_lane_should_be_read_only": True},
        },
    )
    _write_json(
        champion / "promotion_autopilot_packet_latest.json",
        {
            "overall_status": "degraded",
            "autopilot_state": "assembling_packet",
            "promotion_ready": False,
            "committee_packet_seed_ready": True,
            "readiness_repair_contract": {"critical_repair_gate_count": 2},
        },
    )
    _write_json(health / "canary_auto_tuner_latest.json", {"target_canary_max_weight": 0.01})
    _write_json(health / "canary_rollout_latest.json", {"eligible": False, "promote_canary": False, "applied_weight": 0.01})

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "degraded"
    assert payload["runtime_clearance_state"] == "managed_coverage_stage_deferred"
    assert payload["runtime_clearance_recoverable"] is True
    assert payload["preapproved_supervised_ready"] is True
    assert payload["recommended_mode"] == "preapproved_supervised"


def test_live_canary_control_preapproves_seeded_packet_when_runtime_is_already_clear(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    champion = tmp_path / "governance" / "champion_challenger"
    _write_json(health / "broker_readiness_latest.json", {"ready_for_open": True})
    _write_json(health / "session_ready_latest.json", {"ready": True})
    _write_json(health / "storage_route_status_latest.json", {"ok": True, "mode": "external"})
    _write_json(health / "execution_lane_live_latest.json", {"stale": False})
    _write_json(
        health / "live_runtime_separation_control_latest.json",
        {
            "overall_status": "ready",
            "clearance_plan": {"clearance_state": "ready"},
            "release_contract": {"live_lane_should_be_read_only": False},
        },
    )
    _write_json(
        champion / "promotion_autopilot_packet_latest.json",
        {
            "overall_status": "degraded",
            "autopilot_state": "assembling_packet",
            "promotion_ready": False,
            "committee_packet_seed_ready": True,
            "readiness_repair_contract": {"critical_repair_gate_count": 1},
        },
    )
    _write_json(health / "canary_auto_tuner_latest.json", {"target_canary_max_weight": 0.01})
    _write_json(health / "canary_rollout_latest.json", {"eligible": True, "promote_canary": True, "applied_weight": 0.01})

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "degraded"
    assert payload["supervised_canary_ready"] is False
    assert payload["staged_preclearance_ready"] is True
    assert payload["preapproved_supervised_ready"] is True
    assert payload["recommended_mode"] == "preapproved_supervised"
    assert payload["blocking_reasons"] == ["promotion_packet_preclearance_only"]


def test_live_canary_control_blocks_when_core_prerequisites_are_missing(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    champion = tmp_path / "governance" / "champion_challenger"
    _write_json(health / "broker_readiness_latest.json", {"ready_for_open": False})
    _write_json(health / "session_ready_latest.json", {"ready": False})
    _write_json(health / "storage_route_status_latest.json", {"ok": False, "mode": "local"})
    _write_json(health / "execution_lane_live_latest.json", {"stale": True})
    _write_json(health / "live_runtime_separation_control_latest.json", {"overall_status": "blocked", "clearance_plan": {"clearance_state": "awaiting_cold_lane"}})
    _write_json(champion / "promotion_autopilot_packet_latest.json", {})
    _write_json(health / "canary_auto_tuner_latest.json", {"target_canary_max_weight": 0.2})
    _write_json(health / "canary_rollout_latest.json", {"eligible": False, "promote_canary": False})

    payload = src.build_payload(tmp_path)

    assert payload["overall_status"] == "blocked"
    assert payload["staged_preclearance_ready"] is False
    assert "broker_not_ready" in payload["blocking_reasons"]
    assert "session_not_ready" in payload["blocking_reasons"]


def test_live_canary_control_marks_coverage_cycles_ready_as_runnable_release_window(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    champion = tmp_path / "governance" / "champion_challenger"
    _write_json(health / "broker_readiness_latest.json", {"ready_for_open": True})
    _write_json(health / "session_ready_latest.json", {"ready": True})
    _write_json(health / "storage_route_status_latest.json", {"ok": True, "mode": "external"})
    _write_json(health / "execution_lane_live_latest.json", {"stale": True})
    _write_json(
        health / "live_runtime_separation_control_latest.json",
        {
            "overall_status": "degraded",
            "clearance_plan": {"clearance_state": "coverage_cycles_ready"},
            "release_contract": {"live_lane_should_be_read_only": True},
        },
    )
    _write_json(
        champion / "promotion_autopilot_packet_latest.json",
        {
            "overall_status": "degraded",
            "autopilot_state": "ready_for_supervised_canary",
            "promotion_ready": True,
            "canary_packet_ready": True,
        },
    )
    _write_json(health / "canary_auto_tuner_latest.json", {"target_canary_max_weight": 0.01})
    _write_json(health / "canary_rollout_latest.json", {"eligible": False, "promote_canary": False, "applied_weight": 0.01})

    payload = src.build_payload(tmp_path)

    assert payload["runtime_clearance_recoverable"] is True
    assert payload["preapproved_supervised_ready"] is True
    assert payload["runnable_after_release_window"] is True
    assert payload["recommended_mode"] == "runnable_pending_release_window"
