import json
import sys
from datetime import datetime, timezone
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


def test_live_readiness_smoke_treats_ready_resource_guard_with_green_memory_as_ready(tmp_path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)

    (health / "broker_readiness_latest.json").write_text(
        json.dumps({"ready_for_open": True, "network_ok": True, "auth_ok": True}),
        encoding="utf-8",
    )
    (health / "premarket_token_guard_latest.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (health / "session_ready_latest.json").write_text(json.dumps({"ready": True}), encoding="utf-8")
    (health / "execution_lane_paper_latest.json").write_text(json.dumps({"stale": False}), encoding="utf-8")
    (health / "execution_lane_live_latest.json").write_text(json.dumps({"stale": False}), encoding="utf-8")
    (health / "storage_route_status_latest.json").write_text(json.dumps({"ok": True, "mode": "external"}), encoding="utf-8")
    (health / "resource_guard_latest.json").write_text(
        json.dumps(
            {
                "ok": True,
                "overall_status": "ready",
                "resource_guard_ok": False,
                "memory_pressure_state": "green",
                "memory_pressure_kind": "normal",
                "swap_used_gb": 3.1,
            }
        ),
        encoding="utf-8",
    )
    (health / "process_watchdog_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "status": [{"name": "live_lane", "running": 1, "heartbeat_ok": True}],
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
    assert payload["overall_status"] == "ready"
    assert payload["readiness_score"] == 100.0
    assert "resource_guard_not_ok" not in payload["warnings"]
    assert payload["memory_hygiene"]["resource_guard_ok"] is True
    assert payload["memory_hygiene"]["resource_guard_raw_ok"] is False


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


def test_live_readiness_smoke_supports_supervised_canary_mode(tmp_path) -> None:
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
    (health / "live_canary_control_latest.json").write_text(
        json.dumps(
            {
                "overall_status": "ready",
                "supervised_canary_ready": True,
                "recommended_mode": "supervised_canary",
                "blocking_reasons": [],
                "target_canary_weight": 0.08,
                "applied_canary_weight": 0.08,
                "canary_weight_ok": True,
            }
        ),
        encoding="utf-8",
    )
    (health / "process_watchdog_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "status": [{"name": "live_lane", "running": 1, "heartbeat_ok": True}],
                "restart_storms": [],
                "alerts": [],
            }
        ),
        encoding="utf-8",
    )

    out = tmp_path / "live_readiness_canary.json"
    old_root = smoke.PROJECT_ROOT
    try:
        smoke.PROJECT_ROOT = project_root
        sys.argv = [
            "live_readiness_smoke.py",
            "--project-root",
            str(project_root),
            "--out-file",
            str(out),
            "--allow-live-canary-submit",
            "--json",
        ]
        rc = smoke.main()
    finally:
        smoke.PROJECT_ROOT = old_root

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["mode"] == "supervised_canary"
    assert payload["submit_path_enabled"] is True
    assert payload["canary_control"]["supervised_canary_ready"] is True
    assert "live_canary_not_ready" not in payload["hard_blocks"]


def test_live_readiness_smoke_reports_canary_preclearance_when_submit_is_not_ready(tmp_path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)

    (health / "broker_readiness_latest.json").write_text(json.dumps({"ready_for_open": True, "network_ok": True, "auth_ok": True}), encoding="utf-8")
    (health / "premarket_token_guard_latest.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (health / "session_ready_latest.json").write_text(json.dumps({"ready": True}), encoding="utf-8")
    (health / "execution_lane_paper_latest.json").write_text(json.dumps({"stale": False}), encoding="utf-8")
    (health / "execution_lane_live_latest.json").write_text(json.dumps({}), encoding="utf-8")
    (health / "storage_route_status_latest.json").write_text(json.dumps({"ok": True, "mode": "external"}), encoding="utf-8")
    (health / "resource_guard_latest.json").write_text(json.dumps({"resource_guard_ok": True}), encoding="utf-8")
    (health / "live_canary_control_latest.json").write_text(
        json.dumps(
            {
                "overall_status": "degraded",
                "supervised_canary_ready": False,
                "staged_preclearance_ready": True,
                "recommended_mode": "staged_preclearance",
                "blocking_reasons": ["runtime_clearance_not_ready", "live_lane_read_only", "promotion_packet_not_ready"],
                "target_canary_weight": 0.08,
                "applied_canary_weight": 0.0,
                "canary_weight_ok": False,
            }
        ),
        encoding="utf-8",
    )
    (health / "process_watchdog_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "status": [{"name": "paper_lane", "running": 1, "heartbeat_ok": True}],
                "restart_storms": [],
                "alerts": [],
            }
        ),
        encoding="utf-8",
    )

    out = tmp_path / "live_readiness_canary_preclearance.json"
    old_root = smoke.PROJECT_ROOT
    try:
        smoke.PROJECT_ROOT = project_root
        sys.argv = [
            "live_readiness_smoke.py",
            "--project-root",
            str(project_root),
            "--out-file",
            str(out),
            "--allow-live-canary-submit",
            "--json",
        ]
        rc = smoke.main()
    finally:
        smoke.PROJECT_ROOT = old_root

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 2
    assert payload["canary_control"]["staged_preclearance_ready"] is True
    assert "live_canary_preclearance_only" in payload["warnings"]
    assert "live_canary_not_ready" in payload["hard_blocks"]


def test_live_readiness_smoke_softens_paper_lane_watchdog_under_bounded_storage_recovery(tmp_path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)

    (health / "broker_readiness_latest.json").write_text(json.dumps({"ready_for_open": True, "network_ok": True, "auth_ok": True}), encoding="utf-8")
    (health / "premarket_token_guard_latest.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (health / "session_ready_latest.json").write_text(json.dumps({"ready": True}), encoding="utf-8")
    (health / "execution_lane_paper_latest.json").write_text(json.dumps({"stale": False}), encoding="utf-8")
    (health / "execution_lane_live_latest.json").write_text(json.dumps({}), encoding="utf-8")
    (health / "storage_route_status_latest.json").write_text(json.dumps({"ok": True, "mode": "external"}), encoding="utf-8")
    (health / "resource_guard_latest.json").write_text(json.dumps({"resource_guard_ok": True}), encoding="utf-8")
    (health / "live_runtime_separation_control_latest.json").write_text(
        json.dumps({"overall_status": "degraded", "clearance_plan": {"clearance_state": "awaiting_coverage_cycles"}}),
        encoding="utf-8",
    )
    (health / "live_canary_control_latest.json").write_text(
        json.dumps(
            {
                "overall_status": "degraded",
                "recommended_mode": "preapproved_supervised",
                "preclearance_score": 95.0,
                "preapproved_supervised_ready": True,
                "staged_preclearance_ready": True,
                "supervised_canary_ready": False,
            }
        ),
        encoding="utf-8",
    )
    (health / "ingestion_storage_control_latest.json").write_text(
        json.dumps(
            {
                "overall_status": "blocked",
                "recovery_state": "recovering_under_guard",
                "bounded_recovery_contract": {
                    "active": True,
                    "active_drain_progress": True,
                    "drain_follow_through_status": "handoff_requested",
                },
            }
        ),
        encoding="utf-8",
    )
    (health / "process_watchdog_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "status": [{"name": "execution_lane_paper", "running": 1, "heartbeat_ok": False}],
                "restart_storms": [{"name": "execution_lane_paper"}],
                "alerts": [{"name": "execution_lane_paper"}],
            }
        ),
        encoding="utf-8",
    )

    out = tmp_path / "live_readiness_bounded.json"
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
    assert "watchdog_restart_storm" not in payload["hard_blocks"]
    assert "watchdog_targets_missing" not in payload["hard_blocks"]
    assert "bounded_paper_lane_watchdog_pressure" in payload["warnings"]
    assert payload["process_watchdog"]["bounded_paper_lane_watchdog"] is True


def test_live_readiness_smoke_treats_all_sleeves_watchdog_as_live_lane(tmp_path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)

    (health / "broker_readiness_latest.json").write_text(json.dumps({"ready_for_open": True, "network_ok": True, "auth_ok": True}), encoding="utf-8")
    (health / "premarket_token_guard_latest.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (health / "session_ready_latest.json").write_text(json.dumps({"ready": True}), encoding="utf-8")
    (health / "execution_lane_paper_latest.json").write_text(json.dumps({"stale": False}), encoding="utf-8")
    (health / "execution_lane_live_latest.json").write_text(json.dumps({"stale": True}), encoding="utf-8")
    (health / "storage_route_status_latest.json").write_text(json.dumps({"ok": True, "mode": "external"}), encoding="utf-8")
    (health / "resource_guard_latest.json").write_text(json.dumps({"resource_guard_ok": True}), encoding="utf-8")
    (health / "process_watchdog_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "status": [{"name": "all_sleeves", "running": 1, "heartbeat_ok": True}],
                "restart_storms": [],
                "alerts": [],
            }
        ),
        encoding="utf-8",
    )

    out = tmp_path / "live_readiness_watchdog_lane.json"
    old_root = smoke.PROJECT_ROOT
    try:
        smoke.PROJECT_ROOT = project_root
        sys.argv = ["live_readiness_smoke.py", "--project-root", str(project_root), "--out-file", str(out), "--json"]
        rc = smoke.main()
    finally:
        smoke.PROJECT_ROOT = old_root

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["live_lane_running"] is True
    assert payload["preopen_dashboard"]["live_lane_running"] is True


def test_live_readiness_smoke_accepts_virtual_idle_sql_writer(tmp_path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)

    (health / "broker_readiness_latest.json").write_text(json.dumps({"ready_for_open": True, "network_ok": True, "auth_ok": True}), encoding="utf-8")
    (health / "premarket_token_guard_latest.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (health / "session_ready_latest.json").write_text(json.dumps({"ready": True}), encoding="utf-8")
    (health / "execution_lane_paper_latest.json").write_text(json.dumps({"stale": False}), encoding="utf-8")
    (health / "execution_lane_live_latest.json").write_text(json.dumps({"stale": True}), encoding="utf-8")
    (health / "storage_route_status_latest.json").write_text(json.dumps({"ok": True, "mode": "external"}), encoding="utf-8")
    (health / "resource_guard_latest.json").write_text(json.dumps({"resource_guard_ok": True}), encoding="utf-8")
    (health / "process_watchdog_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "status": [
                    {"name": "all_sleeves", "running": 1, "heartbeat_ok": True},
                    {
                        "name": "sql_link_writer",
                        "running": 0,
                        "heartbeat_ok": True,
                        "virtual_process_live": True,
                        "writer_idle_ok": True,
                        "process_live_reason": "sql_writer_on_demand_idle_complete",
                    },
                ],
                "restart_storms": [],
                "alerts": [],
            }
        ),
        encoding="utf-8",
    )

    out = tmp_path / "live_readiness_virtual_writer.json"
    old_root = smoke.PROJECT_ROOT
    try:
        smoke.PROJECT_ROOT = project_root
        sys.argv = ["live_readiness_smoke.py", "--project-root", str(project_root), "--out-file", str(out), "--json"]
        rc = smoke.main()
    finally:
        smoke.PROJECT_ROOT = old_root

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["process_watchdog"]["healthy"] is True
    assert payload["process_watchdog"]["healthy_target_count"] == 2
    assert payload["process_watchdog"]["unhealthy_target_count"] == 0


def test_live_readiness_smoke_forgives_creative_cotenant_paused_targets_in_validate_only(tmp_path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)

    (health / "broker_readiness_latest.json").write_text(json.dumps({"ready_for_open": True, "network_ok": True, "auth_ok": True}), encoding="utf-8")
    (health / "premarket_token_guard_latest.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (health / "session_ready_latest.json").write_text(json.dumps({"ready": True}), encoding="utf-8")
    (health / "execution_lane_paper_latest.json").write_text(json.dumps({"stale": False}), encoding="utf-8")
    (health / "execution_lane_live_latest.json").write_text(json.dumps({"stale": True}), encoding="utf-8")
    (health / "storage_route_status_latest.json").write_text(json.dumps({"ok": True, "mode": "external"}), encoding="utf-8")
    (health / "resource_guard_latest.json").write_text(json.dumps({"resource_guard_ok": True}), encoding="utf-8")
    (health / "process_watchdog_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "status": [
                    {"name": "all_sleeves", "running": 1, "heartbeat_ok": True},
                    {
                        "name": "coinbase_loop",
                        "running": 0,
                        "heartbeat_ok": False,
                        "paused_by_creative_cotenant_guard": True,
                        "restart_skipped": "creative_cotenant_pause_active",
                    },
                    {
                        "name": "coinbase_futures_loop",
                        "running": 0,
                        "heartbeat_ok": False,
                        "paused_by_creative_cotenant_guard": True,
                        "restart_skipped": "creative_cotenant_pause_active",
                    },
                ],
                "restart_storms": [],
                "alerts": [],
            }
        ),
        encoding="utf-8",
    )

    out = tmp_path / "live_readiness_creative_pause.json"
    old_root = smoke.PROJECT_ROOT
    try:
        smoke.PROJECT_ROOT = project_root
        sys.argv = ["live_readiness_smoke.py", "--project-root", str(project_root), "--out-file", str(out), "--json"]
        rc = smoke.main()
    finally:
        smoke.PROJECT_ROOT = old_root

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["overall_status"] == "ready"
    assert payload["process_watchdog"]["creative_paused_target_count"] == 2
    assert payload["process_watchdog"]["unhealthy_target_count"] == 0
    assert "watchdog_targets_missing" not in payload["hard_blocks"]
