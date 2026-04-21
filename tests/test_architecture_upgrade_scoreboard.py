import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import architecture_upgrade_scoreboard as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_architecture_upgrade_scoreboard_tracks_twelve_surfaces(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    walk = project_root / "governance" / "walk_forward"
    champion = project_root / "governance" / "champion_challenger"

    _write_json(health / "live_runtime_separation_control_latest.json", {"overall_status": "degraded", "shared_host_pressure": {"contention_score": 2}, "clearance_plan": {"clearance_state": "awaiting_coverage_cycles"}})
    _write_json(walk / "coverage_gap_closer_latest.json", {"autopilot_contract": {"overall_status": "degraded", "launch_state": "stage_only_off_hours", "stage_candidate_count": 3}})
    _write_json(champion / "promotion_autopilot_packet_latest.json", {"overall_status": "ready", "signed_bundle_contract": {"signature_verified": True, "rollback_ready": True}})
    _write_json(
        health / "portable_brain_contract_latest.json",
        {
            "host_contract": {"host_profile": "portable_throughput", "chip": "AMD Ryzen"},
            "adaptation_contract": {"recommended_runtime_access_mode": "portable"},
            "cross_platform_proof_node": {"status": "ready", "effective_backend": "onnx"},
        },
    )
    _write_json(health / "apple_silicon_profile_latest.json", {"overall_status": "ready", "applied_tier": "max_throughput"})
    _write_json(health / "mode_switchboard_mission_control_latest.json", {"overall_status": "ready", "mode_counts": {"active": 3, "ready": 3}, "control_surface": {"clearance_state": "ready"}})
    _write_json(health / "autonomy_control_plane_latest.json", {"overall_status": "degraded", "autonomy_score": 77.5, "lane_recovery_playbooks": {"triggered_playbook_count": 2}})
    _write_json(health / "decision_provenance_cards_latest.json", {"overall_status": "ready", "card_count": 6, "mode_count": 4})
    _write_json(health / "notification_escalation_ladder_latest.json", {"overall_status": "ready", "remote_pager_ready": True, "critical_backlog": {"unacked_count": 0}})
    _write_json(health / "chaos_drill_coordinator_latest.json", {"overall_status": "degraded", "overdue_drills": [{"drill": "storage_failover"}], "drill_program": {"program_score": 82.0}})
    _write_json(health / "incident_review_packet_latest.json", {"overall_status": "degraded", "review_required": True, "packet_sha256": "abc123"})
    _write_json(health / "macro_auto_watch_status.json", {"live_detected": True})

    payload = src.build_payload(project_root)

    assert payload["upgrade_count"] == 12
    assert "portable_brain_contract" in payload["special_features_map"]
    assert any(row["slug"] == "cross_platform_proof_node" for row in payload["rows"])
    assert payload["overall_status"] == "degraded"
