import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import global_risk_killswitch as kill_src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_global_risk_killswitch_blocks_auto_clear_when_runtime_is_stressed(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    halt_flag = health / "GLOBAL_TRADING_HALT.flag"
    halt_flag.parent.mkdir(parents=True, exist_ok=True)
    halt_flag.write_text("halted", encoding="utf-8")

    _write_json(health / "one_numbers_latest.json", {"combined_blocked_rate": 0.0, "combined_pnl_proxy": 0.0, "decision_stale_windows_4h": 0, "watchdog_restarts": 0})
    _write_json(health / "health_gates_latest.json", {"hard_gate_triggered": False})
    _write_json(health / "auth_lease_manager_latest.json", {"lease_state": "healthy"})
    _write_json(health / "data_plane_recovery_controller_latest.json", {"write_failure_count": 1, "account_snapshot_failure_count": 0, "queue_depth": 500})
    _write_json(health / "process_watchdog_latest.json", {"restart_storms": []})
    _write_json(health / "live_runtime_separation_control_latest.json", {"clearance_plan": {"clearance_state": "coverage_cycles_ready"}})

    monkeypatch.setattr(kill_src, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(sys, "argv", ["global_risk_killswitch.py", "--auto-clear"])

    rc = kill_src.main()
    payload = json.loads((health / "global_killswitch_latest.json").read_text(encoding="utf-8"))

    assert rc == 1
    assert payload["action"] == "clear_blocked"
    assert payload["clear_ready"] is False
    assert "write_path_recovery_pending" in payload["clear_blockers"]
    assert halt_flag.exists()


def test_global_risk_killswitch_reports_operator_stop_as_clear_blocker(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    halt_flag = health / "GLOBAL_TRADING_HALT.flag"
    halt_flag.parent.mkdir(parents=True, exist_ok=True)
    halt_flag.write_text("halted", encoding="utf-8")
    (health / "OPERATOR_STOP.flag").write_text("stopped", encoding="utf-8")

    _write_json(health / "one_numbers_latest.json", {"combined_blocked_rate": 0.0, "combined_pnl_proxy": 0.0, "decision_stale_windows_4h": 0, "watchdog_restarts": 0})
    _write_json(health / "health_gates_latest.json", {"hard_gate_triggered": False})
    _write_json(health / "auth_lease_manager_latest.json", {"lease_state": "healthy"})
    _write_json(health / "data_plane_recovery_controller_latest.json", {"write_failure_count": 0, "account_snapshot_failure_count": 0, "queue_depth": 0})
    _write_json(health / "process_watchdog_latest.json", {"restart_storms": []})
    _write_json(health / "live_runtime_separation_control_latest.json", {"clearance_plan": {"clearance_state": "ready"}})

    monkeypatch.setattr(kill_src, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(sys, "argv", ["global_risk_killswitch.py", "--auto-clear"])

    rc = kill_src.main()
    payload = json.loads((health / "global_killswitch_latest.json").read_text(encoding="utf-8"))

    assert rc == 1
    assert "operator_stop_active" in payload["clear_blockers"]
    assert payload["metrics"]["operator_stop_active"] is True
    assert payload["operator_stop"] is True
    assert payload["control_commands"]["safe_auto_clear"] == ["./scripts/ops/opsctl.sh", "global-halt-auto-clear", "--json"]
    assert payload["control_commands"]["manual_clear_all_halts"] == ["./scripts/ops/opsctl.sh", "clear-all-halts", "--json"]
    assert ["./scripts/ops/opsctl.sh", "operator-release", "--json"] in payload["recommended_commands"]


def test_global_risk_killswitch_accepts_guarded_live_read_only_runtime_state(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    halt_flag = health / "GLOBAL_TRADING_HALT.flag"
    halt_flag.parent.mkdir(parents=True, exist_ok=True)
    halt_flag.write_text(json.dumps({"reason": "prior_storage_recovery"}), encoding="utf-8")

    _write_json(health / "one_numbers_latest.json", {"combined_blocked_rate": 0.0, "combined_pnl_proxy": 0.0, "decision_stale_windows_4h": 0, "watchdog_restarts": 0})
    _write_json(health / "health_gates_latest.json", {"hard_gate_triggered": False})
    _write_json(health / "auth_lease_manager_latest.json", {"lease_state": "healthy"})
    _write_json(health / "data_plane_recovery_controller_latest.json", {"write_failure_count": 0, "account_snapshot_failure_count": 0, "queue_depth": 0})
    _write_json(health / "process_watchdog_latest.json", {"restart_storms": []})
    _write_json(
        health / "live_runtime_separation_control_latest.json",
        {"clearance_plan": {"clearance_state": "guarded_live_read_only"}, "live_plane": {"live_lane_running": True}},
    )

    monkeypatch.setattr(kill_src, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(sys, "argv", ["global_risk_killswitch.py", "--auto-clear"])

    rc = kill_src.main()
    payload = json.loads((health / "global_killswitch_latest.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["action"] == "halt_cleared"
    assert payload["clear_blockers"] == []
    assert payload["metrics"]["runtime_clearance_state"] == "guarded_live_read_only"
    assert not halt_flag.exists()


def test_global_risk_killswitch_exposes_active_hard_gates_and_exit_zero(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    halt_flag = health / "GLOBAL_TRADING_HALT.flag"
    halt_flag.parent.mkdir(parents=True, exist_ok=True)
    halt_flag.write_text(json.dumps({"reason": "test_halt"}), encoding="utf-8")

    _write_json(health / "one_numbers_latest.json", {"combined_blocked_rate": 0.0, "combined_pnl_proxy": 0.0, "decision_stale_windows_4h": 0, "watchdog_restarts": 0})
    _write_json(health / "health_gates_latest.json", {"hard_gate_triggered": True, "hard_gates": {"sql_wal_pressure": True, "blocked_rate": False}})
    _write_json(health / "auth_lease_manager_latest.json", {"lease_state": "healthy"})
    _write_json(health / "data_plane_recovery_controller_latest.json", {"write_failure_count": 0, "account_snapshot_failure_count": 0, "queue_depth": 0})
    _write_json(health / "process_watchdog_latest.json", {"restart_storms": []})
    _write_json(health / "live_runtime_separation_control_latest.json", {"clearance_plan": {"clearance_state": "ready"}})

    monkeypatch.setattr(kill_src, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(sys, "argv", ["global_risk_killswitch.py", "--status-only", "--exit-zero"])

    rc = kill_src.main()
    payload = json.loads((health / "global_killswitch_latest.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["action"] == "halt_would_set"
    assert payload["hard_gate_names"] == ["sql_wal_pressure"]
    assert payload["global_halt_payload"] == {"reason": "test_halt"}
    assert ["./scripts/ops/opsctl.sh", "ingestion-storage-control", "--json"] in payload["recommended_commands"]


def test_global_risk_killswitch_auto_clear_does_not_rewrite_active_halt_when_gates_still_fail(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    halt_flag = health / "GLOBAL_TRADING_HALT.flag"
    original_payload = {"reason": "softguard_api_circuit_opened", "timestamp_utc": "2026-04-30T21:01:17+00:00"}
    halt_flag.parent.mkdir(parents=True, exist_ok=True)
    halt_flag.write_text(json.dumps(original_payload), encoding="utf-8")

    _write_json(health / "one_numbers_latest.json", {"combined_blocked_rate": 0.0, "combined_pnl_proxy": 0.0, "decision_stale_windows_4h": 0, "watchdog_restarts": 0})
    _write_json(health / "health_gates_latest.json", {"hard_gate_triggered": True, "hard_gates": {"sql_wal_pressure": True}})
    _write_json(health / "auth_lease_manager_latest.json", {"lease_state": "healthy"})
    _write_json(health / "data_plane_recovery_controller_latest.json", {"write_failure_count": 0, "account_snapshot_failure_count": 0, "queue_depth": 0})
    _write_json(health / "process_watchdog_latest.json", {"restart_storms": []})
    _write_json(health / "live_runtime_separation_control_latest.json", {"clearance_plan": {"clearance_state": "ready"}})

    monkeypatch.setattr(kill_src, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(sys, "argv", ["global_risk_killswitch.py", "--auto-clear"])

    rc = kill_src.main()
    payload = json.loads((health / "global_killswitch_latest.json").read_text(encoding="utf-8"))

    assert rc == 2
    assert payload["action"] == "clear_blocked"
    assert json.loads(halt_flag.read_text(encoding="utf-8")) == original_payload


def test_global_risk_killswitch_auto_clear_is_clear_only_when_halt_is_unlatched(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    halt_flag = health / "GLOBAL_TRADING_HALT.flag"

    _write_json(health / "one_numbers_latest.json", {"combined_blocked_rate": 0.0, "combined_pnl_proxy": 0.0, "decision_stale_windows_4h": 0, "watchdog_restarts": 0})
    _write_json(health / "health_gates_latest.json", {"hard_gate_triggered": True, "hard_gates": {"sql_wal_pressure": True}})
    _write_json(health / "auth_lease_manager_latest.json", {"lease_state": "healthy"})
    _write_json(health / "data_plane_recovery_controller_latest.json", {"write_failure_count": 0, "account_snapshot_failure_count": 0, "queue_depth": 0})
    _write_json(health / "process_watchdog_latest.json", {"restart_storms": []})
    _write_json(health / "live_runtime_separation_control_latest.json", {"clearance_plan": {"clearance_state": "ready"}})

    monkeypatch.setattr(kill_src, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(sys, "argv", ["global_risk_killswitch.py", "--auto-clear"])

    rc = kill_src.main()
    payload = json.loads((health / "global_killswitch_latest.json").read_text(encoding="utf-8"))

    assert rc == 2
    assert payload["action"] == "halt_required_unlatched"
    assert payload["halt"] is False
    assert payload["halt_latched"] is False
    assert payload["halt_latched_before"] is False
    assert payload["halt_required"] is True
    assert payload["would_rehalt"] is True
    assert payload["halt_posture"] == "unlatched_halt_required"
    assert payload["halt_pressure"]["reasons"] == ["health_hard_gate_triggered"]
    assert not halt_flag.exists()


def test_global_risk_killswitch_downgrades_recovered_expansion_pressure(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    halt_flag = health / "GLOBAL_TRADING_HALT.flag"
    halt_flag.parent.mkdir(parents=True, exist_ok=True)
    halt_flag.write_text(json.dumps({"reason": "prior_expansion_pressure"}), encoding="utf-8")

    _write_json(health / "one_numbers_latest.json", {"combined_blocked_rate": 0.0, "combined_pnl_proxy": 0.0, "decision_stale_windows_4h": 0, "watchdog_restarts": 0})
    _write_json(health / "health_gates_latest.json", {"hard_gate_triggered": True, "hard_gates": {"collector_contracts": True, "ingestion_backpressure_overload": True}})
    _write_json(health / "ingestion_backpressure_latest.json", {"overload": False, "pending_lines_total": 77, "pending_lines_threshold": 15000, "line_pressure": False, "file_pressure": False, "age_pressure": False})
    _write_json(health / "auth_lease_manager_latest.json", {"lease_state": "healthy"})
    _write_json(health / "data_plane_recovery_controller_latest.json", {"write_failure_count": 0, "account_snapshot_failure_count": 0, "queue_depth": 77})
    _write_json(health / "process_watchdog_latest.json", {"restart_storms": []})
    _write_json(health / "live_runtime_separation_control_latest.json", {"clearance_plan": {"clearance_state": "ready"}, "live_plane": {"live_lane_running": False}})

    monkeypatch.setattr(kill_src, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(sys, "argv", ["global_risk_killswitch.py", "--auto-clear"])

    rc = kill_src.main()
    payload = json.loads((health / "global_killswitch_latest.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["action"] == "halt_cleared"
    assert payload["critical_hard_gate_names"] == []
    assert payload["degraded_hard_gate_names"] == ["collector_contracts"]
    assert payload["stale_hard_gate_names"] == ["ingestion_backpressure_overload"]
    assert payload["operating_mode"] == "degraded_collection"
    assert payload["sleeve_throttle_recommended"] is True
    assert not halt_flag.exists()


def test_global_risk_killswitch_softens_stale_queue_depth_when_backpressure_recovered(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    halt_flag = health / "GLOBAL_TRADING_HALT.flag"
    halt_flag.parent.mkdir(parents=True, exist_ok=True)
    halt_flag.write_text(json.dumps({"reason": "prior_queue_backpressure"}), encoding="utf-8")

    _write_json(health / "one_numbers_latest.json", {"combined_blocked_rate": 0.0, "combined_pnl_proxy": 0.0, "decision_stale_windows_4h": 0, "watchdog_restarts": 0})
    _write_json(health / "health_gates_latest.json", {"hard_gate_triggered": False})
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "overload": False,
            "pending_lines_total": 10082,
            "pending_lines_threshold": 15000,
            "line_pressure": False,
            "file_pressure": False,
            "age_pressure": False,
        },
    )
    _write_json(health / "auth_lease_manager_latest.json", {"lease_state": "healthy"})
    _write_json(health / "data_plane_recovery_controller_latest.json", {"write_failure_count": 0, "account_snapshot_failure_count": 0, "queue_depth": 13322})
    _write_json(health / "process_watchdog_latest.json", {"restart_storms": []})
    _write_json(health / "live_runtime_separation_control_latest.json", {"clearance_plan": {"clearance_state": "ready"}, "live_plane": {"live_lane_running": True}})

    monkeypatch.setenv("ALLOW_ORDER_EXECUTION", "0")
    monkeypatch.setenv("MARKET_DATA_ONLY", "1")
    monkeypatch.setattr(kill_src, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(sys, "argv", ["global_risk_killswitch.py", "--auto-clear"])

    rc = kill_src.main()
    payload = json.loads((health / "global_killswitch_latest.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["action"] == "halt_cleared"
    assert payload["clear_blockers"] == []
    assert payload["degraded_clear_blockers"] == ["queue_depth_recovered_waiting_backlog_drain"]
    assert payload["metrics"]["current_backpressure_clear"] is True
    assert not halt_flag.exists()


def test_global_risk_killswitch_softens_snapshot_and_runtime_when_not_executing(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    halt_flag = health / "GLOBAL_TRADING_HALT.flag"
    halt_flag.parent.mkdir(parents=True, exist_ok=True)
    halt_flag.write_text(json.dumps({"reason": "snapshot_probe"}), encoding="utf-8")

    _write_json(health / "one_numbers_latest.json", {"combined_blocked_rate": 0.0, "combined_pnl_proxy": 0.0, "decision_stale_windows_4h": 0, "watchdog_restarts": 0})
    _write_json(health / "health_gates_latest.json", {"hard_gate_triggered": False})
    _write_json(health / "auth_lease_manager_latest.json", {"lease_state": "healthy"})
    _write_json(health / "data_plane_recovery_controller_latest.json", {"write_failure_count": 0, "account_snapshot_failure_count": 3, "queue_depth": 100})
    _write_json(health / "process_watchdog_latest.json", {"restart_storms": []})
    _write_json(health / "live_runtime_separation_control_latest.json", {"clearance_plan": {"clearance_state": "awaiting_cold_lane"}, "live_plane": {"live_lane_running": False}})

    monkeypatch.setenv("ALLOW_ORDER_EXECUTION", "0")
    monkeypatch.setenv("MARKET_DATA_ONLY", "1")
    monkeypatch.setattr(kill_src, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(sys, "argv", ["global_risk_killswitch.py", "--auto-clear"])

    rc = kill_src.main()
    payload = json.loads((health / "global_killswitch_latest.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["action"] == "halt_cleared"
    assert payload["clear_blockers"] == []
    assert payload["degraded_clear_blockers"] == [
        "account_snapshot_recovery_pending",
        "runtime_clearance=awaiting_cold_lane",
    ]
    assert payload["operating_mode"] == "degraded_collection"



def test_global_risk_killswitch_softens_runtime_coverage_debt_for_live_data_only_lane(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    halt_flag = health / "GLOBAL_TRADING_HALT.flag"
    halt_flag.parent.mkdir(parents=True, exist_ok=True)
    halt_flag.write_text(json.dumps({"reason": "coverage_debt"}), encoding="utf-8")

    _write_json(health / "one_numbers_latest.json", {"combined_blocked_rate": 0.0, "combined_pnl_proxy": 0.0, "decision_stale_windows_4h": 0, "watchdog_restarts": 0})
    _write_json(health / "health_gates_latest.json", {"hard_gate_triggered": False})
    _write_json(health / "auth_lease_manager_latest.json", {"lease_state": "healthy"})
    _write_json(health / "data_plane_recovery_controller_latest.json", {"write_failure_count": 0, "account_snapshot_failure_count": 0, "queue_depth": 245})
    _write_json(health / "process_watchdog_latest.json", {"restart_storms": []})
    _write_json(health / "live_runtime_separation_control_latest.json", {"clearance_plan": {"clearance_state": "awaiting_coverage_cycles"}, "live_plane": {"live_lane_running": True}})

    monkeypatch.setenv("ALLOW_ORDER_EXECUTION", "0")
    monkeypatch.setenv("MARKET_DATA_ONLY", "1")
    monkeypatch.setattr(kill_src, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(sys, "argv", ["global_risk_killswitch.py", "--auto-clear"])

    rc = kill_src.main()
    payload = json.loads((health / "global_killswitch_latest.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["action"] == "halt_cleared"
    assert payload["clear_blockers"] == []
    assert payload["degraded_clear_blockers"] == ["runtime_clearance=awaiting_coverage_cycles"]
    assert payload["metrics"]["live_lane_running"] is True
    assert payload["metrics"]["execution_expected"] is False


def test_global_risk_killswitch_softens_recovered_restart_storm(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    halt_flag = health / "GLOBAL_TRADING_HALT.flag"
    halt_flag.parent.mkdir(parents=True, exist_ok=True)
    halt_flag.write_text(json.dumps({"reason": "planned_livefeed_refresh"}), encoding="utf-8")

    _write_json(health / "one_numbers_latest.json", {"combined_blocked_rate": 0.0, "combined_pnl_proxy": 0.0, "decision_stale_windows_4h": 0, "watchdog_restarts": 0})
    _write_json(health / "health_gates_latest.json", {"hard_gate_triggered": False})
    _write_json(health / "auth_lease_manager_latest.json", {"lease_state": "healthy"})
    _write_json(health / "data_plane_recovery_controller_latest.json", {"write_failure_count": 0, "account_snapshot_failure_count": 0, "queue_depth": 0})
    _write_json(
        health / "process_watchdog_latest.json",
        {
            "restart_storms": [{"name": "all_sleeves", "count": 4, "resolved": False}],
            "status": [{"name": "all_sleeves", "running": 1, "alt_running": 3, "heartbeat_ok": True, "process_live": True}],
        },
    )
    _write_json(health / "live_runtime_separation_control_latest.json", {"clearance_plan": {"clearance_state": "ready"}, "live_plane": {"live_lane_running": True}})

    monkeypatch.setenv("ALLOW_ORDER_EXECUTION", "0")
    monkeypatch.setenv("MARKET_DATA_ONLY", "1")
    monkeypatch.setattr(kill_src, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(sys, "argv", ["global_risk_killswitch.py", "--auto-clear"])

    rc = kill_src.main()
    payload = json.loads((health / "global_killswitch_latest.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["action"] == "halt_cleared"
    assert payload["clear_blockers"] == []
    assert payload["degraded_clear_blockers"] == ["restart_storm_recovered_waiting_settle"]
    assert payload["metrics"]["restart_storm_recovered"] is True
    assert not halt_flag.exists()


def test_global_risk_killswitch_softens_isolated_read_only_restart_storm(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    halt_flag = health / "GLOBAL_TRADING_HALT.flag"
    halt_flag.parent.mkdir(parents=True, exist_ok=True)
    halt_flag.write_text(json.dumps({"reason": "collector_restart_storm"}), encoding="utf-8")

    _write_json(health / "one_numbers_latest.json", {"combined_blocked_rate": 0.0, "combined_pnl_proxy": 0.0, "decision_stale_windows_4h": 0, "watchdog_restarts": 0})
    _write_json(health / "health_gates_latest.json", {"hard_gate_triggered": False})
    _write_json(health / "ingestion_backpressure_latest.json", {"overload": False, "pending_lines_total": 0, "pending_lines_threshold": 15000})
    _write_json(health / "auth_lease_manager_latest.json", {"lease_state": "healthy"})
    _write_json(health / "data_plane_recovery_controller_latest.json", {"write_failure_count": 0, "account_snapshot_failure_count": 0, "queue_depth": 0})
    _write_json(
        health / "process_watchdog_latest.json",
        {
            "restart_storms": [
                {
                    "name": "all_sleeves",
                    "count": 4,
                    "resolved": False,
                    "impact": "read_only_collection",
                    "quarantinable": True,
                    "blocks_execution_clear": False,
                }
            ],
            "status": [{"name": "all_sleeves", "running": 0, "heartbeat_ok": False, "process_live": False}],
        },
    )
    _write_json(health / "live_runtime_separation_control_latest.json", {"clearance_plan": {"clearance_state": "ready"}, "live_plane": {"live_lane_running": True}})

    monkeypatch.setenv("ALLOW_ORDER_EXECUTION", "0")
    monkeypatch.setenv("MARKET_DATA_ONLY", "1")
    monkeypatch.setattr(kill_src, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(sys, "argv", ["global_risk_killswitch.py", "--auto-clear"])

    rc = kill_src.main()
    payload = json.loads((health / "global_killswitch_latest.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["action"] == "halt_cleared"
    assert payload["clear_blockers"] == []
    assert payload["degraded_clear_blockers"] == ["restart_storm_isolated_read_only_collection"]
    assert payload["metrics"]["restart_storm_isolation"]["execution_blocking_count"] == 0
    assert payload["metrics"]["restart_storm_isolation"]["isolated_targets"] == ["all_sleeves"]
    assert not halt_flag.exists()


def test_global_risk_killswitch_keeps_execution_restart_storm_hard_blocked(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    halt_flag = health / "GLOBAL_TRADING_HALT.flag"
    halt_flag.parent.mkdir(parents=True, exist_ok=True)
    halt_flag.write_text(json.dumps({"reason": "execution_restart_storm"}), encoding="utf-8")

    _write_json(health / "one_numbers_latest.json", {"combined_blocked_rate": 0.0, "combined_pnl_proxy": 0.0, "decision_stale_windows_4h": 0, "watchdog_restarts": 0})
    _write_json(health / "health_gates_latest.json", {"hard_gate_triggered": False})
    _write_json(health / "ingestion_backpressure_latest.json", {"overload": False, "pending_lines_total": 0, "pending_lines_threshold": 15000})
    _write_json(health / "auth_lease_manager_latest.json", {"lease_state": "healthy"})
    _write_json(health / "data_plane_recovery_controller_latest.json", {"write_failure_count": 0, "account_snapshot_failure_count": 0, "queue_depth": 0})
    _write_json(
        health / "process_watchdog_latest.json",
        {
            "restart_storms": [
                {
                    "name": "execution_lane_live",
                    "count": 4,
                    "resolved": False,
                    "impact": "execution_lane",
                    "quarantinable": False,
                    "blocks_execution_clear": True,
                }
            ],
            "status": [{"name": "execution_lane_live", "running": 0, "heartbeat_ok": False, "process_live": False}],
        },
    )
    _write_json(health / "live_runtime_separation_control_latest.json", {"clearance_plan": {"clearance_state": "ready"}, "live_plane": {"live_lane_running": True}})

    monkeypatch.setenv("ALLOW_ORDER_EXECUTION", "0")
    monkeypatch.setenv("MARKET_DATA_ONLY", "1")
    monkeypatch.setattr(kill_src, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(sys, "argv", ["global_risk_killswitch.py", "--auto-clear"])

    rc = kill_src.main()
    payload = json.loads((health / "global_killswitch_latest.json").read_text(encoding="utf-8"))

    assert rc == 1
    assert payload["action"] == "clear_blocked"
    assert payload["clear_blockers"] == ["restart_storm_active"]
    assert payload["metrics"]["restart_storm_isolation"]["execution_blocking_targets"] == ["execution_lane_live"]
    assert halt_flag.exists()


def test_global_risk_killswitch_escalates_recoverable_gates_when_live_execution_expected(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"

    _write_json(health / "one_numbers_latest.json", {"combined_blocked_rate": 0.0, "combined_pnl_proxy": 0.0, "decision_stale_windows_4h": 0, "watchdog_restarts": 0})
    _write_json(health / "health_gates_latest.json", {"hard_gate_triggered": True, "hard_gates": {"collector_contracts": True}})
    _write_json(health / "ingestion_backpressure_latest.json", {"overload": False, "pending_lines_total": 0, "pending_lines_threshold": 15000})
    _write_json(health / "auth_lease_manager_latest.json", {"lease_state": "healthy"})
    _write_json(health / "data_plane_recovery_controller_latest.json", {"write_failure_count": 0, "account_snapshot_failure_count": 0, "queue_depth": 0})
    _write_json(health / "process_watchdog_latest.json", {"restart_storms": []})
    _write_json(health / "live_runtime_separation_control_latest.json", {"clearance_plan": {"clearance_state": "ready"}, "live_plane": {"live_lane_running": True}})

    monkeypatch.setenv("ALLOW_ORDER_EXECUTION", "1")
    monkeypatch.setenv("MARKET_DATA_ONLY", "0")
    monkeypatch.setattr(kill_src, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(sys, "argv", ["global_risk_killswitch.py", "--status-only"])

    rc = kill_src.main()
    payload = json.loads((health / "global_killswitch_latest.json").read_text(encoding="utf-8"))

    assert rc == 2
    assert payload["action"] == "halt_would_set"
    assert payload["operating_mode"] == "global_halt_required"


def test_global_risk_killswitch_surfaces_quant_expansion_pressure(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    halt_flag = health / "GLOBAL_TRADING_HALT.flag"
    halt_flag.parent.mkdir(parents=True, exist_ok=True)
    halt_flag.write_text(json.dumps({"reason": "quant_pressure"}), encoding="utf-8")

    _write_json(health / "one_numbers_latest.json", {"combined_blocked_rate": 0.0, "combined_pnl_proxy": 0.0, "decision_stale_windows_4h": 0, "watchdog_restarts": 0})
    _write_json(health / "health_gates_latest.json", {"hard_gate_triggered": False})
    _write_json(health / "ingestion_backpressure_latest.json", {"overload": False, "pending_lines_total": 0, "pending_lines_threshold": 15000})
    _write_json(health / "auth_lease_manager_latest.json", {"lease_state": "healthy"})
    _write_json(health / "data_plane_recovery_controller_latest.json", {"write_failure_count": 0, "account_snapshot_failure_count": 0, "queue_depth": 0})
    _write_json(health / "process_watchdog_latest.json", {"restart_storms": []})
    _write_json(health / "live_runtime_separation_control_latest.json", {"clearance_plan": {"clearance_state": "ready"}, "live_plane": {"live_lane_running": False}})
    _write_json(health / "quant_model_control_latest.json", {"overall_status": "degraded", "features": {"quant_model_resource_pressure_norm": 0.86}})

    monkeypatch.setattr(kill_src, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(sys, "argv", ["global_risk_killswitch.py", "--auto-clear"])

    rc = kill_src.main()
    payload = json.loads((health / "global_killswitch_latest.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["action"] == "halt_cleared"
    assert "quant_model_resource_pressure" in payload["degraded_clear_blockers"]
    assert payload["metrics"]["quant_model_status"] == "degraded"
    assert payload["metrics"]["quant_model_resource_pressure"] == 0.86
    assert ["./scripts/ops/opsctl.sh", "quant-model-control", "--json"] in payload["recommended_commands"]
    assert ["./scripts/ops/opsctl.sh", "memory-efficiency", "--apply", "--json"] in payload["recommended_commands"]
    assert payload["critical_hard_gate_names"] == []
    assert payload["degraded_hard_gate_names"] == []


def test_global_risk_killswitch_escalates_severe_backpressure_even_in_collection_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"

    _write_json(health / "one_numbers_latest.json", {"combined_blocked_rate": 0.0, "combined_pnl_proxy": 0.0, "decision_stale_windows_4h": 0, "watchdog_restarts": 0})
    _write_json(health / "health_gates_latest.json", {"hard_gate_triggered": True, "hard_gates": {"ingestion_backpressure_overload": True}})
    _write_json(health / "ingestion_backpressure_latest.json", {"overload": True, "pending_lines_total": 40000, "pending_lines_threshold": 15000, "line_pressure": True})
    _write_json(health / "auth_lease_manager_latest.json", {"lease_state": "healthy"})
    _write_json(health / "data_plane_recovery_controller_latest.json", {"write_failure_count": 0, "account_snapshot_failure_count": 0, "queue_depth": 40000})
    _write_json(health / "process_watchdog_latest.json", {"restart_storms": []})
    _write_json(health / "live_runtime_separation_control_latest.json", {"clearance_plan": {"clearance_state": "ready"}, "live_plane": {"live_lane_running": False}})

    monkeypatch.setattr(kill_src, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(sys, "argv", ["global_risk_killswitch.py", "--status-only"])

    rc = kill_src.main()
    payload = json.loads((health / "global_killswitch_latest.json").read_text(encoding="utf-8"))

    assert rc == 2
    assert payload["action"] == "halt_would_set"
    assert payload["critical_hard_gate_names"] == ["ingestion_backpressure_overload"]
    assert payload["metrics"]["backpressure_pressure_ratio"] > 2.0


def test_global_risk_killswitch_bounds_clear_blocker_refresh(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(kill_src, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(kill_src, "_clear_blocker_steps", lambda: [("slow_step", ["slow-command"])])

    def slow_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout", 1), output="partial", stderr="")

    monkeypatch.setattr(kill_src.subprocess, "run", slow_run)

    attempts = kill_src._attempt_clear_blockers(timeout_sec=1)

    assert attempts == [
        {
            "name": "slow_step",
            "rc": 124,
            "ok": False,
            "timed_out": True,
            "payload": {},
            "stdout_tail": "partial",
            "stderr_tail": "timeout",
        }
    ]
