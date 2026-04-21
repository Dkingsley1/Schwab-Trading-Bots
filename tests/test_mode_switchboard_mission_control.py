import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import mode_switchboard_mission_control as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_mode_switchboard_mission_control_tracks_shadow_paper_and_live(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "live_readiness_smoke_latest.json",
        {"broker_ready": True, "session_ready": True, "paper_lane_fresh": True, "live_lane_running": False},
    )
    _write_json(
        health / "live_runtime_separation_control_latest.json",
        {"overall_status": "ready", "clearance_plan": {"clearance_state": "ready"}, "release_contract": {"shared_host_training_resume_allowed": True}},
    )
    _write_json(health / "runtime_access_mode_latest.json", {"mode": "native"})
    _write_json(health / "portable_brain_contract_latest.json", {"host_contract": {"host_profile": "max_throughput"}})
    _write_json(
        health / "process_watchdog_latest.json",
        {"status": [{"name": "shadow_watchdog", "running": 1}, {"name": "paper_execution_lane", "running": 1}]},
    )

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["mode_counts"]["active"] == 2
    assert payload["control_surface"]["host_profile"] == "max_throughput"
    assert any(row["mode"] == "shadow" and row["active"] for row in payload["modes"])
    assert any(row["mode"] == "paper" and row["active"] for row in payload["modes"])
    assert any(row["mode"] == "live" and row["ready"] for row in payload["modes"])
