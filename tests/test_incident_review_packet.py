import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import incident_review_packet as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_incident_review_packet_hashes_open_incident_snapshot(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "incident_timeline_latest.json",
        {
            "overall_status": "blocked",
            "recent_incident_count": 3,
            "open_incident_count": 2,
            "auto_close_contract": {"closure_ready": False, "candidate_count": 0},
            "open_surfaces": [{"surface": "runtime_separation", "status": "blocked"}],
            "recent_incidents": [{"summary": "auth lease warning"}],
            "recommended_actions": ["pause risky lanes"],
        },
    )
    _write_json(health / "live_runtime_separation_control_latest.json", {"overall_status": "blocked", "clearance_plan": {"clearance_state": "awaiting_coverage_cycles"}})
    _write_json(health / "auth_lease_manager_latest.json", {"overall_status": "degraded", "lease_state": "warning"})
    _write_json(health / "remote_alert_control_latest.json", {"overall_status": "degraded", "critical_backlog": {"unacked_count": 1, "unsent_count": 0}})
    _write_json(health / "lane_thaw_controller_latest.json", {"overall_status": "degraded", "paused_lane_count": 2, "candidate_count": 1})
    _write_json(health / "data_plane_recovery_controller_latest.json", {"overall_status": "degraded", "write_failure_count": 3, "account_snapshot_failure_count": 1})

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "blocked"
    assert payload["review_required"] is True
    assert payload["review_state"] == "awaiting_remediation"
    assert payload["open_incident_count"] == 2
    assert payload["recent_categories"] == []
    assert payload["closure_contract"]["closure_ready"] is False
    assert len(payload["packet_sha256"]) == 64
    assert payload["immutability_contract"]["hash_algorithm"] == "sha256"
