import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import promotion_autopilot_packet


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_promotion_autopilot_packet_surfaces_ready_signed_bundle(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "governance" / "champion_challenger" / "promotion_packet_latest.json",
        {
            "packet_complete": True,
            "packet_sha256": "abc123",
            "signature": {"verified": True, "status": "verified"},
            "gate_results": {"training_success_confirmed": True, "golden_replay_regression_ok": True},
            "rollback_bundle": {"rollback_reference": "deadbeef", "rollback_command": "./scripts/release_ops.sh rollback deadbeef"},
            "code": {"git_commit": "deadbeef"},
            "sources": {"training_success": "governance/health/training_success_latest.json"},
        },
    )
    _write_json(
        project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json",
        {"promote_ok": True, "coverage_shortfall_bots": 0, "blocking_reasons": []},
    )
    _write_json(project_root / "governance" / "health" / "promotion_quality_gate_latest.json", {"ok": True})
    _write_json(project_root / "governance" / "walk_forward" / "promotion_pipeline_latest.json", {"ok": True})

    payload = promotion_autopilot_packet.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["autopilot_state"] == "awaiting_approval"
    assert payload["promotion_ready"] is True
    assert payload["approval_state"] == "awaiting_operator_signoff"
    assert payload["repairable_gate_count"] == 0
    assert payload["blocker_count"] == 0
    assert payload["packet_completeness_score"] == 100.0
    assert payload["approval_record"]["approval_state"] == "awaiting_operator_signoff"
    assert payload["committee_packet_seed_ready"] is True
    assert payload["signed_bundle_contract"]["signature_verified"] is True
    assert payload["signed_bundle_contract"]["rollback_ready"] is True
    assert payload["signability_contract"]["committee_packet_ready"] is True
    assert payload["signability_contract"]["committee_packet_seed_ready"] is True
    assert payload["readiness_repair_contract"]["repairable_gate_count"] == 0
    assert payload["blockers"] == []


def test_promotion_autopilot_packet_degrades_on_unsigned_or_unready_bundle(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "governance" / "champion_challenger" / "promotion_packet_latest.json",
        {
            "packet_complete": False,
            "packet_sha256": "abc123",
            "signature": {"verified": False, "status": "missing_signing_key"},
            "gate_results": {"training_success_confirmed": False},
        },
    )
    _write_json(
        project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json",
        {"promote_ok": False, "coverage_shortfall_bots": 3, "blocking_reasons": ["insufficient_walk_forward_coverage"]},
    )
    _write_json(project_root / "governance" / "health" / "promotion_quality_gate_latest.json", {"ok": False})
    _write_json(project_root / "governance" / "walk_forward" / "promotion_pipeline_latest.json", {"ok": False})

    payload = promotion_autopilot_packet.build_payload(project_root)

    assert payload["overall_status"] == "degraded"
    assert payload["autopilot_state"] == "assembling_packet"
    assert payload["approval_state"] == "not_ready"
    assert payload["repairable_gate_count"] >= 3
    assert payload["blocker_count"] >= 4
    assert payload["repairable_packet_state"] is True
    assert payload["committee_packet_seed_ready"] is False
    assert payload["signed_bundle_contract"]["signature_verified"] is False
    assert payload["signability_contract"]["committee_packet_ready"] is False
    assert payload["readiness_repair_contract"]["repairable_gate_count"] >= 3
    assert "coverage_shortfall_bots=3" in payload["blockers"]
    assert "promotion_quality_gate_failed" in payload["blockers"]


def test_promotion_autopilot_packet_accepts_env_signing_material_for_ready_packet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "governance" / "champion_challenger" / "promotion_packet_latest.json",
        {
            "packet_complete": True,
            "packet_sha256": "abc123",
            "signature": {"verified": False, "status": "missing_signing_key"},
            "gate_results": {"training_success_confirmed": True, "golden_replay_regression_ok": True},
            "rollback_bundle": {"rollback_reference": "deadbeef", "rollback_command": "./scripts/release_ops.sh rollback deadbeef"},
            "code": {"git_commit": "deadbeef"},
            "sources": {"training_success": "governance/health/training_success_latest.json"},
        },
    )
    _write_json(
        project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json",
        {"promote_ok": True, "coverage_shortfall_bots": 0, "blocking_reasons": []},
    )
    _write_json(project_root / "governance" / "health" / "promotion_quality_gate_latest.json", {"ok": True})
    _write_json(project_root / "governance" / "walk_forward" / "promotion_pipeline_latest.json", {"ok": True})
    monkeypatch.setenv("PROMOTION_PACKET_SIGNING_KEY", "topsecret")

    payload = promotion_autopilot_packet.build_payload(project_root)

    assert payload["overall_status"] == "degraded"
    assert payload["autopilot_state"] == "awaiting_signature"
    assert payload["signability_contract"]["env_key_present"] is True
    assert payload["signability_contract"]["can_sign_now"] is True
    assert payload["signability_contract"]["committee_packet_seed_ready"] is True
    assert payload["approval_record"]["approval_record_seed_ready"] is True


def test_promotion_autopilot_packet_focuses_on_readiness_repairs_once_bundle_is_signed(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "governance" / "champion_challenger" / "promotion_packet_latest.json",
        {
            "packet_complete": False,
            "packet_sha256": "abc123",
            "signature": {"verified": True, "status": "verified"},
            "gate_results": {
                "training_success_confirmed": False,
                "feature_store_manifest_strict_ok": False,
                "retrain_schema_compatibility_ok": False,
            },
            "rollback_bundle": {"rollback_reference": "deadbeef", "rollback_command": "./scripts/release_ops.sh rollback deadbeef"},
            "code": {"git_commit": "deadbeef"},
            "sources": {"training_success": "governance/health/training_success_latest.json"},
        },
    )
    _write_json(
        project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json",
        {"promote_ok": False, "coverage_shortfall_bots": 2, "blocking_reasons": ["insufficient_walk_forward_coverage"]},
    )
    _write_json(project_root / "governance" / "health" / "promotion_quality_gate_latest.json", {"ok": False})
    _write_json(project_root / "governance" / "walk_forward" / "promotion_pipeline_latest.json", {"ok": False})

    payload = promotion_autopilot_packet.build_payload(project_root)

    assert payload["overall_status"] == "degraded"
    assert payload["autopilot_state"] == "repairing_readiness"
    assert payload["approval_state"] == "not_ready"
    assert payload["repairable_gate_count"] >= 5
    assert payload["blocker_count"] >= 6
    assert payload["readiness_repair_contract"]["repairable_gate_count"] >= 5
    assert any(row["gate"] == "training_success_confirmed" for row in payload["readiness_repair_contract"]["repair_rows"])
    assert any(row["gate"] == "walk_forward_coverage" for row in payload["readiness_repair_contract"]["repair_rows"])


def test_promotion_autopilot_packet_blocks_when_packet_is_missing(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json",
        {"promote_ok": False, "coverage_shortfall_bots": 0, "blocking_reasons": []},
    )
    _write_json(project_root / "governance" / "health" / "promotion_quality_gate_latest.json", {"ok": False})
    _write_json(project_root / "governance" / "walk_forward" / "promotion_pipeline_latest.json", {"ok": False})

    payload = promotion_autopilot_packet.build_payload(project_root)

    assert payload["overall_status"] == "blocked"
    assert payload["autopilot_state"] == "missing_packet"
    assert payload["repairable_packet_state"] is False


def test_promotion_autopilot_packet_uses_ready_for_committee_fallback_for_complete_packets(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "governance" / "champion_challenger" / "promotion_packet_latest.json",
        {
            "ready_for_committee": True,
            "committee_packet_seed_ready": True,
            "packet_sha256": "abc123",
            "signature": {"verified": True, "status": "verified"},
            "gate_results": {"training_success_confirmed": True, "golden_replay_regression_ok": True},
            "rollback_bundle": {"rollback_reference": "deadbeef", "rollback_command": "./scripts/release_ops.sh rollback deadbeef"},
            "code": {"git_commit": "deadbeef"},
            "dataset": {"rows_sha256": "rows-hash"},
            "sources": {"training_success": "governance/health/training_success_latest.json"},
        },
    )
    _write_json(
        project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json",
        {"promote_ok": True, "coverage_shortfall_bots": 0, "blocking_reasons": []},
    )
    _write_json(project_root / "governance" / "health" / "promotion_quality_gate_latest.json", {"ok": True})
    _write_json(project_root / "governance" / "walk_forward" / "promotion_pipeline_latest.json", {"ok": True})

    payload = promotion_autopilot_packet.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["signability_contract"]["committee_packet_ready"] is True
    assert payload["committee_packet_seed_ready"] is True


def test_promotion_autopilot_packet_degrades_when_canary_ready_but_not_yet_awaiting_approval(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "governance" / "champion_challenger" / "promotion_packet_latest.json",
        {
            "packet_complete": True,
            "packet_sha256": "abc123",
            "signature": {"verified": True, "status": "verified"},
            "gate_results": {"training_success_confirmed": True, "golden_replay_regression_ok": True},
            "rollback_bundle": {"rollback_reference": "deadbeef", "rollback_command": "./scripts/release_ops.sh rollback deadbeef"},
            "code": {"git_commit": "deadbeef"},
            "sources": {"training_success": "governance/health/training_success_latest.json"},
        },
    )
    _write_json(
        project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json",
        {"promote_ok": False, "coverage_shortfall_bots": 4, "blocking_reasons": ["insufficient_walk_forward_coverage"]},
    )
    _write_json(
        project_root / "governance" / "walk_forward" / "coverage_seed_latest.json",
        {
            "standing_queue": {"seed_queue_size": 8},
            "seed_queue": [{"bot_id": "brain_refinery_v4_simple"}],
        },
    )
    _write_json(
        project_root / "governance" / "walk_forward" / "coverage_gap_closer_latest.json",
        {
            "staged_candidate_count": 4,
            "autopilot_contract": {"overall_status": "ready", "launch_state": "ready_to_launch"},
        },
    )
    _write_json(project_root / "governance" / "health" / "promotion_quality_gate_latest.json", {"ok": True})
    _write_json(project_root / "governance" / "walk_forward" / "promotion_pipeline_latest.json", {"ok": True})

    payload = promotion_autopilot_packet.build_payload(project_root)

    assert payload["autopilot_state"] == "ready_for_supervised_canary"
    assert payload["overall_status"] == "degraded"
    assert payload["canary_packet_ready"] is True
    assert payload["blockers"] == []


def test_promotion_autopilot_packet_accepts_waiting_for_idle_coverage_stage_as_seed_ready(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "governance" / "champion_challenger" / "promotion_packet_latest.json",
        {
            "packet_complete": True,
            "packet_sha256": "abc123",
            "signature": {"verified": True, "status": "verified"},
            "gate_results": {"training_success_confirmed": True, "golden_replay_regression_ok": True},
            "rollback_bundle": {"rollback_reference": "deadbeef", "rollback_command": "./scripts/release_ops.sh rollback deadbeef"},
            "code": {"git_commit": "deadbeef"},
            "sources": {"training_success": "governance/health/training_success_latest.json"},
        },
    )
    _write_json(
        project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json",
        {"promote_ok": False, "coverage_shortfall_bots": 4, "blocking_reasons": ["insufficient_walk_forward_coverage"]},
    )
    _write_json(
        project_root / "governance" / "walk_forward" / "coverage_seed_latest.json",
        {
            "standing_queue": {"seed_queue_size": 8},
            "seed_queue": [{"bot_id": "brain_refinery_v4_simple"}],
        },
    )
    _write_json(
        project_root / "governance" / "walk_forward" / "coverage_gap_closer_latest.json",
        {
            "staged_candidate_count": 4,
            "autopilot_contract": {"overall_status": "degraded", "launch_state": "waiting_for_idle", "can_apply_stage": True},
        },
    )
    _write_json(project_root / "governance" / "health" / "promotion_quality_gate_latest.json", {"ok": True})
    _write_json(project_root / "governance" / "walk_forward" / "promotion_pipeline_latest.json", {"ok": True})

    payload = promotion_autopilot_packet.build_payload(project_root)

    assert payload["canary_packet_ready"] is True
    assert payload["blockers"] == []
