import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import incident_timeline as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_incident_timeline_downgrades_paper_lane_watchdog_under_storage_recovery(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    walk = tmp_path / "governance" / "walk_forward"

    _write_json(health / "live_readiness_smoke_latest.json", {"overall_status": "blocked"})
    _write_json(
        health / "live_runtime_separation_control_latest.json",
        {"overall_status": "degraded", "clearance_plan": {"clearance_state": "awaiting_coverage_cycles"}},
    )
    _write_json(
        health / "auth_lease_manager_latest.json",
        {
            "overall_status": "degraded",
            "lease_state": "warning",
            "lease_budget": {"expires_in_seconds": 1800, "critical_lease_seconds": 900},
        },
    )
    _write_json(walk / "coverage_seed_latest.json", {"overall_status": "needs_coverage", "coverage_shortfall_bots": 4, "seed_queue": ["a"]})
    _write_json(walk / "coverage_gap_closer_latest.json", {"overall_status": "degraded"})
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "recovery_state": "blocked_backpressure",
            "pressure_index": 5.22,
        },
    )
    _write_json(
        health / "process_watchdog_latest.json",
        {
            "restart_storms": [{"name": "execution_lane_paper"}],
            "alerts": [{"name": "execution_lane_paper"}],
        },
    )

    payload = src.build_payload(tmp_path)

    assert payload["open_incident_count"] == 1
    assert [row["surface"] for row in payload["open_surfaces"]] == ["live_readiness"]
    assert any(
        row["surface"] == "process_watchdog" and row.get("watch_reason") == "derived_storage_backpressure"
        for row in payload["watch_surfaces"]
    )


def test_incident_timeline_treats_bounded_live_release_window_as_watch_only(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    walk = tmp_path / "governance" / "walk_forward"

    _write_json(
        health / "live_readiness_smoke_latest.json",
        {
            "overall_status": "blocked",
            "canary_control": {"bounded_runtime_preclearance": True},
            "process_watchdog": {"bounded_paper_lane_watchdog": True},
        },
    )
    _write_json(
        health / "live_runtime_separation_control_latest.json",
        {"overall_status": "degraded", "clearance_plan": {"clearance_state": "awaiting_coverage_cycles"}},
    )
    _write_json(health / "auth_lease_manager_latest.json", {"overall_status": "ready", "lease_state": "healthy"})
    _write_json(walk / "coverage_seed_latest.json", {"overall_status": "ready", "coverage_shortfall_bots": 0, "seed_queue": []})
    _write_json(walk / "coverage_gap_closer_latest.json", {"overall_status": "ready"})
    _write_json(health / "ingestion_storage_control_latest.json", {"overall_status": "ready", "recovery_state": "steady_state", "pressure_index": 0.2})
    _write_json(health / "process_watchdog_latest.json", {"restart_storms": [], "alerts": []})

    payload = src.build_payload(tmp_path)

    assert payload["open_incident_count"] == 0
    assert any(
        row["surface"] == "live_readiness" and row.get("watch_reason") == "bounded_release_window"
        for row in payload["watch_surfaces"]
    )


def test_incident_timeline_downgrades_paper_lane_watchdog_when_drain_contract_is_active(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    walk = tmp_path / "governance" / "walk_forward"

    _write_json(health / "live_readiness_smoke_latest.json", {"overall_status": "ready"})
    _write_json(
        health / "live_runtime_separation_control_latest.json",
        {"overall_status": "degraded", "clearance_plan": {"clearance_state": "awaiting_coverage_cycles"}},
    )
    _write_json(health / "auth_lease_manager_latest.json", {"overall_status": "ready", "lease_state": "healthy"})
    _write_json(walk / "coverage_seed_latest.json", {"overall_status": "ready", "coverage_shortfall_bots": 0, "seed_queue": []})
    _write_json(walk / "coverage_gap_closer_latest.json", {"overall_status": "ready"})
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "recovery_state": "blocked_backpressure",
            "pressure_index": 7.13,
            "bounded_recovery_contract": {
                "active": False,
                "quality_ready": False,
                "active_drain_progress": True,
                "drain_follow_through_status": "handoff_requested",
            },
        },
    )
    _write_json(
        health / "process_watchdog_latest.json",
        {
            "restart_storms": [{"name": "execution_lane_paper"}],
            "alerts": [{"name": "execution_lane_paper"}],
        },
    )

    payload = src.build_payload(tmp_path)

    assert payload["open_incident_count"] == 0
    assert any(
        row["surface"] == "process_watchdog" and row.get("watch_reason") == "derived_storage_backpressure"
        for row in payload["watch_surfaces"]
    )
