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


def test_incident_review_packet_archives_when_timeline_is_watch_only(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "incident_timeline_latest.json",
        {
            "overall_status": "degraded",
            "recent_incident_count": 3,
            "open_incident_count": 0,
            "watch_surface_count": 2,
            "review_required": False,
            "auto_close_contract": {"closure_ready": True, "candidate_count": 1, "review_required": False},
            "open_surfaces": [],
            "watch_surfaces": [{"surface": "runtime_separation", "status": "degraded"}],
            "stitched_threads": [{"thread_id": "runtime_preclearance"}],
            "recent_incidents": [{"summary": "auth lease warning"}],
            "recommended_actions": ["refresh the cold lane before live writes"],
        },
    )
    _write_json(health / "live_runtime_separation_control_latest.json", {"overall_status": "degraded", "clearance_plan": {"clearance_state": "awaiting_cold_lane"}})
    _write_json(health / "auth_lease_manager_latest.json", {"overall_status": "degraded", "lease_state": "warning"})
    _write_json(health / "remote_alert_control_latest.json", {"overall_status": "ready", "critical_backlog": {"unacked_count": 0, "unsent_count": 0}})
    _write_json(health / "lane_thaw_controller_latest.json", {"overall_status": "ready", "paused_lane_count": 0, "candidate_count": 0})
    _write_json(health / "data_plane_recovery_controller_latest.json", {"overall_status": "degraded", "write_failure_count": 1, "account_snapshot_failure_count": 0})

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["review_required"] is False
    assert payload["review_state"] == "ready_to_archive"
    assert payload["watch_surface_count"] == 2
    assert payload["closure_contract"]["closure_ready"] is True


def test_incident_review_packet_main_writes_pdf(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "incident_timeline_latest.json",
        {
            "overall_status": "degraded",
            "recent_incident_count": 1,
            "open_incident_count": 0,
            "review_required": False,
            "auto_close_contract": {"closure_ready": True, "candidate_count": 1, "review_required": False},
            "recent_incidents": [{"category": "operations", "summary": "watch-only incident"}],
            "recommended_actions": ["archive the packet"],
        },
    )
    _write_json(health / "live_runtime_separation_control_latest.json", {"overall_status": "ready", "clearance_plan": {"clearance_state": "cleared"}})
    _write_json(health / "auth_lease_manager_latest.json", {"overall_status": "ready", "lease_state": "healthy"})
    _write_json(health / "remote_alert_control_latest.json", {"overall_status": "ready", "critical_backlog": {"unacked_count": 0, "unsent_count": 0}})
    _write_json(health / "lane_thaw_controller_latest.json", {"overall_status": "ready", "paused_lane_count": 0, "candidate_count": 0})
    _write_json(health / "data_plane_recovery_controller_latest.json", {"overall_status": "ready", "write_failure_count": 0, "account_snapshot_failure_count": 0})
    out_file = health / "incident_review_packet_latest.json"
    pdf_file = project_root / "exports" / "reports" / "incident_review_packet_latest.pdf"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "incident_review_packet.py",
            "--project-root",
            str(project_root),
            "--out-file",
            str(out_file),
            "--pdf-out-file",
            str(pdf_file),
            "--json",
        ],
    )

    rc = src.main()
    payload = json.loads(out_file.read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["artifacts"]["pdf_available"] is True
    assert payload["artifacts"]["pdf"] == str(pdf_file)
    assert pdf_file.read_bytes().startswith(b"%PDF-1.4")
