import json
import sys
from pathlib import Path

from scripts.ops import halt_trigger_control_plane as src


FRESH_TS = "2099-01-01T00:00:00+00:00"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_healthy_artifacts(project_root: Path) -> None:
    health = project_root / "governance" / "health"
    alerts = project_root / "governance" / "alerts"
    _write_json(
        health / "global_killswitch_latest.json",
        {
            "timestamp_utc": FRESH_TS,
            "halt": False,
            "halt_state": "clear_ready",
            "clear_ready": True,
            "clear_blockers": [],
            "reasons": [],
        },
    )
    _write_json(
        health / "health_gates_latest.json",
        {
            "timestamp_utc": FRESH_TS,
            "hard_gate_triggered": False,
            "hard_gates": {"sql_wal_pressure": False},
            "recommended_operating_mode": "live_full",
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "timestamp_utc": FRESH_TS,
            "overall_status": "ready",
            "throttle_profile": "soft_cap",
            "storage_stabilization": {"recommended_operating_mode": "live_full", "backlog_drain_status": "clear"},
            "mac_fluidity_contract": {"fluidity_band": "guarded_smooth"},
            "release_contract": {"live_lane_should_be_read_only": False, "paper_trade_lock_active": False},
        },
    )
    _write_json(
        health / "live_runtime_separation_control_latest.json",
        {
            "timestamp_utc": FRESH_TS,
            "overall_status": "ready",
            "clearance_plan": {"clearance_state": "ready"},
            "live_plane": {"ready": True, "live_lane_running": True},
            "release_contract": {"live_lane_should_be_read_only": False, "heavy_research_must_stay_cold_lane": False},
        },
    )
    _write_json(
        alerts / "incident_auto_halt_latest.json",
        {
            "timestamp_utc": FRESH_TS,
            "ok": True,
            "halt": False,
            "event": "state_update",
            "failed_checks": [],
            "detail": {"enforcement_suppressed": False},
        },
    )
    _write_json(
        health / "paper_execution_truth_layer_latest.json",
        {"timestamp_utc": FRESH_TS, "pause_paper_execution": False, "blocked": False},
    )
    _write_json(
        health / "paper_400_ramp_latest.json",
        {"timestamp_utc": FRESH_TS, "pause_paper_execution": False, "blocked": False},
    )


def test_halt_trigger_control_plane_fails_closed_on_invalid_active_halt_flag(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    health.mkdir(parents=True)
    _write_healthy_artifacts(project_root)
    (health / "GLOBAL_TRADING_HALT.flag").write_text("halted", encoding="utf-8")
    (health / "OPERATOR_STOP.flag").write_text(json.dumps({"reason": "operator_test"}), encoding="utf-8")

    monkeypatch.setattr(src, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(sys, "argv", ["halt_trigger_control_plane.py", "--json", "--assert-clear"])

    rc = src.main()
    payload = json.loads((health / "halt_trigger_control_plane_latest.json").read_text(encoding="utf-8"))

    assert rc == 2
    assert payload["effective_state"] == "operator_stop"
    assert "manual_operator_stop_active" in payload["blockers"]["halt_clear"]
    assert "manual_global_halt_active" in payload["blockers"]["halt_clear"]
    assert "invalid_global_halt_payload" in payload["blockers"]["halt_clear"]
    assert payload["manual_flags"]["global_halt"]["valid"] is False
    assert payload["execution_policy"]["control_plane_allows_live_orders"] is False
    assert payload["viewer_policy"]["heavy_livefeed_allowed"] is True


def test_halt_trigger_control_plane_reports_clear_when_all_safety_artifacts_are_clean(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    _write_healthy_artifacts(project_root)
    monkeypatch.setenv("ALLOW_ORDER_EXECUTION", "1")
    monkeypatch.setenv("MARKET_DATA_ONLY", "0")
    monkeypatch.setattr(src, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(sys, "argv", ["halt_trigger_control_plane.py", "--json", "--assert-clear"])

    rc = src.main()
    payload = json.loads((project_root / "governance" / "health" / "halt_trigger_control_plane_latest.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["overall_status"] == "ready"
    assert payload["effective_state"] == "clear"
    assert payload["blockers"]["halt_clear"] == []
    assert payload["blockers"]["live_execution"] == []
    assert payload["execution_policy"]["control_plane_allows_live_orders"] is True
    assert payload["execution_policy"]["effective_live_order_execution_allowed"] is True
    assert payload["viewer_policy"]["heavy_livefeed_allowed"] is True


def test_halt_trigger_control_plane_blocks_heavy_viewer_on_runtime_protect_and_hard_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_healthy_artifacts(project_root)
    _write_json(
        health / "health_gates_latest.json",
        {
            "timestamp_utc": FRESH_TS,
            "hard_gate_triggered": True,
            "hard_gates": {"sql_wal_pressure": True},
            "ingestion_pressure": {"critical_priority_failures": ["equities_minute"]},
        },
    )
    _write_json(
        health / "runtime_throttle_control_latest.json",
        {
            "timestamp_utc": FRESH_TS,
            "overall_status": "blocked",
            "throttle_profile": "protect_live",
            "storage_stabilization": {"recommended_operating_mode": "protect_live", "backlog_drain_status": "drain_active"},
            "mac_fluidity_contract": {"fluidity_band": "protect"},
            "release_contract": {"live_lane_should_be_read_only": True, "effective_live_read_only_reason": "protect_live"},
        },
    )
    monkeypatch.setattr(src, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(sys, "argv", ["halt_trigger_control_plane.py", "--json", "--assert-clear"])

    rc = src.main()
    payload = json.loads((health / "halt_trigger_control_plane_latest.json").read_text(encoding="utf-8"))

    assert rc == 2
    assert "health_hard_gates_active" in payload["blockers"]["live_execution"]
    assert "runtime_throttle_protective_profile" in payload["blockers"]["live_execution"]
    assert "runtime_throttle_protective_profile" in payload["viewer_policy"]["heavy_livefeed_wait_reasons"]
    assert "health_hard_gates_active" in payload["viewer_policy"]["heavy_livefeed_wait_reasons"]
    assert payload["viewer_policy"]["heavy_livefeed_allowed"] is False


def test_halt_trigger_control_plane_fails_closed_when_required_artifact_is_stale(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_healthy_artifacts(project_root)
    _write_json(
        health / "health_gates_latest.json",
        {
            "timestamp_utc": "2000-01-01T00:00:00+00:00",
            "hard_gate_triggered": False,
            "hard_gates": {"sql_wal_pressure": False},
        },
    )
    monkeypatch.setattr(src, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(sys, "argv", ["halt_trigger_control_plane.py", "--json", "--assert-clear"])

    rc = src.main()
    payload = json.loads((health / "halt_trigger_control_plane_latest.json").read_text(encoding="utf-8"))

    assert rc == 2
    assert "critical_artifact_stale:health_gates" in payload["blockers"]["halt_clear"]
    assert payload["effective_state"] == "safety_artifact_uncertain"
