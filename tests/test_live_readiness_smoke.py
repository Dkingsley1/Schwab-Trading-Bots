import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import live_readiness_smoke as smoke


def test_live_readiness_smoke_includes_preopen_dashboard_and_memory_hygiene(tmp_path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)

    (health / "broker_readiness_latest.json").write_text(
        json.dumps(
            {
                "ready_for_open": True,
                "network_ok": True,
                "auth_ok": True,
                "token_warning_level": "watch",
                "token_age_seconds": 123.0,
                "account_probe_status_code": 200,
            }
        ),
        encoding="utf-8",
    )
    (health / "premarket_token_guard_latest.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (health / "session_ready_latest.json").write_text(json.dumps({"ready": True}), encoding="utf-8")
    (health / "execution_lane_paper_latest.json").write_text(json.dumps({"stale": False}), encoding="utf-8")
    (health / "execution_lane_live_latest.json").write_text(json.dumps({}), encoding="utf-8")
    (health / "storage_route_status_latest.json").write_text(json.dumps({"ok": True, "mode": "external"}), encoding="utf-8")
    (health / "resource_guard_latest.json").write_text(
        json.dumps({"resource_guard_ok": True, "memory_pressure_state": "red", "memory_pressure_kind": "swap_only", "swap_used_gb": 22.4}),
        encoding="utf-8",
    )
    (health / "process_watchdog_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-04-21T00:00:00+00:00",
                "status": [{"name": "paper_lane", "running": 1, "heartbeat_ok": True}],
                "restart_storms": [],
                "alerts": [],
            }
        ),
        encoding="utf-8",
    )

    out = tmp_path / "live_readiness.json"
    old_root = smoke.PROJECT_ROOT
    try:
        smoke.PROJECT_ROOT = project_root
        sys.argv = [
            "live_readiness_smoke.py",
            "--project-root",
            str(project_root),
            "--out-file",
            str(out),
            "--json",
        ]
        rc = smoke.main()
    finally:
        smoke.PROJECT_ROOT = old_root

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["overall_status"] == "degraded"
    assert payload["readiness_score"] > 80.0
    assert payload["preopen_dashboard"]["token_warning_level"] == "watch"
    assert payload["process_watchdog"]["healthy"] is False
    assert "token_watch_window" in payload["warnings"]
    assert payload["memory_hygiene"]["memory_pressure_kind"] == "swap_only"
    assert "schedule_worker_recycle" in payload["memory_hygiene"]["recommended_actions"]


def test_live_readiness_smoke_blocks_on_restart_storm(tmp_path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)

    (health / "broker_readiness_latest.json").write_text(json.dumps({"ready_for_open": True, "network_ok": True, "auth_ok": True}), encoding="utf-8")
    (health / "premarket_token_guard_latest.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (health / "session_ready_latest.json").write_text(json.dumps({"ready": True}), encoding="utf-8")
    (health / "execution_lane_paper_latest.json").write_text(json.dumps({"stale": False}), encoding="utf-8")
    (health / "execution_lane_live_latest.json").write_text(json.dumps({"stale": False}), encoding="utf-8")
    (health / "storage_route_status_latest.json").write_text(json.dumps({"ok": True, "mode": "external"}), encoding="utf-8")
    (health / "resource_guard_latest.json").write_text(json.dumps({"resource_guard_ok": True}), encoding="utf-8")
    (health / "process_watchdog_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-04-21T00:00:00+00:00",
                "status": [{"name": "paper_lane", "running": 1, "heartbeat_ok": True}],
                "restart_storms": [{"name": "paper_lane"}],
                "alerts": [],
            }
        ),
        encoding="utf-8",
    )

    out = tmp_path / "live_readiness.json"
    old_root = smoke.PROJECT_ROOT
    try:
        smoke.PROJECT_ROOT = project_root
        sys.argv = [
            "live_readiness_smoke.py",
            "--project-root",
            str(project_root),
            "--out-file",
            str(out),
            "--json",
        ]
        rc = smoke.main()
    finally:
        smoke.PROJECT_ROOT = old_root

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 2
    assert payload["ok"] is False
    assert payload["overall_status"] == "blocked"
    assert "watchdog_restart_storm" in payload["hard_blocks"]
    assert "clear_restart_storm_before_live_submit" in payload["recommended_actions"]
