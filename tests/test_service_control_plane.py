import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.service_control_plane as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_service_control_plane_rolls_up_upgrade_lanes(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    allocator_root = project_root / "governance" / "allocator"
    risk_root = project_root / "governance" / "risk"

    _write_json(health / "ops_coordinator_latest.json", {"ok": True})
    _write_json(health / "process_watchdog_latest.json", {"ok": True, "restart_storms": []})
    _write_json(health / "platform_control_plane_latest.json", {"institutional_readiness": {"overall_status": "ready"}})
    _write_json(health / "provider_mesh_latest.json", {"overall_status": "ready", "summary": {"required_contract_ok": 2, "required_collectors": 2}, "cooldowns": []})
    _write_json(
        allocator_root / "portfolio_allocator_service_latest.json",
        {"ok": True, "approved_intents": [{"symbol": "AAPL", "side": "BUY", "approved_qty": 2}]},
    )
    _write_json(
        risk_root / "risk_service_boundary_latest.json",
        {"ok": True, "pre_trade_decisions": [{"symbol": "AAPL", "requested_action": "BUY", "approved_action": "BUY", "risk_limit_ok": True}]},
    )
    _write_json(health / "execution_lane_paper_latest.json", {"stale": False})
    _write_json(health / "execution_lane_live_latest.json", {"stale": False})
    _write_json(health / "retrain_launch_latest.json", {"state": "running"})
    _write_json(health / "retrain_orchestrator_latest.json", {"ok": True})
    _write_json(health / "retrain_scorecard_latest.json", {"failure_count": 0})
    _write_json(health / "training_success_latest.json", {"confirmed_training_success": False, "failure_count": 0})
    _write_json(health / "point_in_time_event_store_latest.json", {"ok": True, "event_count": 12, "category_counts": {"control_plane": 2}})
    _write_json(health / "live_runtime_separation_control_latest.json", {"overall_status": "ready", "shared_host_pressure": {"contention_score": 0}})
    _write_json(health / "operator_cockpit_latest.json", {"overall_status": "ready", "recommended_actions": []})

    payload = src.build_payload(project_root)

    assert payload["overall_status"] == "degraded"
    assert payload["upgrade_lanes"]["control_plane"]["status"] == "ready"
    assert payload["upgrade_lanes"]["provider_mesh"]["status"] == "ready"
    assert payload["upgrade_lanes"]["execution_gateway"]["status"] == "ready"
    assert payload["upgrade_lanes"]["retrain_pipeline"]["status"] == "running"
    assert payload["upgrade_lanes"]["event_history"]["status"] == "ready"
    assert payload["upgrade_lanes"]["runtime_separation"]["status"] == "ready"
    assert payload["summary"]["completion_score"] > 50.0
