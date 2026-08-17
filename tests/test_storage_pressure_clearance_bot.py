from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPS_DIR = PROJECT_ROOT / "scripts" / "ops"
if str(OPS_DIR) not in sys.path:
    sys.path.insert(0, str(OPS_DIR))

import storage_pressure_clearance_bot as clearance_src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _active_pressure_fixture(project_root: Path) -> None:
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "severity": "critical",
            "pressure_index": 2.6,
            "backpressure": {
                "core_pending_lines": 39529,
                "total_pending_lines": 39971,
                "estimated_total_drain_minutes": None,
            },
            "storage": {"sqlite_wal_size_gb": 154.849, "backlog_drain_status": "blocked"},
            "bounded_recovery_contract": {
                "route_verified": True,
                "hard_gate_keys": ["ingestion_backpressure_overload", "sql_wal_pressure"],
            },
            "steady_state": {
                "targets": {"pressure_index": 0.25, "core_pending_lines": 5000},
                "target_status": {"steady_state_ready": False},
            },
        },
    )
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": True,
            "hard_gates": {"ingestion_backpressure_overload": True, "sql_wal_pressure": True},
            "thresholds": {"sql_wal_size_gb_limit": 24.0, "ingestion_pending_lines_limit": 20000},
            "inputs": {
                "backpressure_overload_severe": True,
                "backpressure_pending_lines": 39971,
                "sql_wal_size_gb_live": 154.849,
            },
        },
    )
    _write_json(
        health / "storage_backpressure_autopilot_latest.json",
        {"timestamp_utc": "2099-04-24T14:00:00+00:00", "overall_status": "already_running", "ok": True, "busy": True},
    )


def test_storage_pressure_clearance_refuses_to_fake_clear_active_pressure(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    _active_pressure_fixture(project_root)
    seen: list[str] = []

    def _fake_run_json(cmd: list[str], *, cwd: Path, timeout_sec: int) -> dict:
        joined = " ".join(cmd)
        seen.append(joined)
        return {"cmd": list(cmd), "rc": 0, "timed_out": False, "stdout_tail": "{}", "stderr_tail": "", "payload": {}}

    monkeypatch.setattr(clearance_src, "_run_json", _fake_run_json)

    payload = clearance_src.build_payload(
        project_root,
        apply=True,
        force_clear_stale_gate=True,
        command_timeout_seconds=1,
    )

    assert payload["overall_status"] == "degraded"
    assert payload["force_clear_refused_reason"] == "active_storage_pressure"
    assert payload["metrics"]["active_storage_pressure"] is True
    assert any("sqlite_performance_maintenance.py" in cmd for cmd in seen)
    assert not any("storage_backpressure_autopilot.py" in cmd for cmd in seen)
    assert not any("global_risk_killswitch.py" in cmd for cmd in seen)


def test_storage_pressure_clearance_clears_only_stale_storage_gate(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.1,
            "backpressure": {"core_pending_lines": 0, "total_pending_lines": 0, "estimated_total_drain_minutes": 0.0},
            "storage": {"sqlite_wal_size_gb": 1.0, "backlog_drain_status": "idle"},
            "bounded_recovery_contract": {"route_verified": True},
            "steady_state": {
                "targets": {"pressure_index": 0.25, "core_pending_lines": 5000},
                "target_status": {"steady_state_ready": True},
            },
        },
    )
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gate_triggered": True,
            "hard_gates": {"sql_wal_pressure": True},
            "thresholds": {"sql_wal_size_gb_limit": 24.0, "ingestion_pending_lines_limit": 20000},
            "inputs": {"backpressure_overload_severe": False, "backpressure_pending_lines": 0, "sql_wal_size_gb_live": 1.0},
        },
    )
    _write_json(health / "storage_backpressure_autopilot_latest.json", {"overall_status": "ready", "ok": True})
    seen: list[str] = []

    def _fake_run_json(cmd: list[str], *, cwd: Path, timeout_sec: int) -> dict:
        joined = " ".join(cmd)
        seen.append(joined)
        if "health_gates.py" in joined:
            _write_json(
                health / "health_gates_latest.json",
                {
                    "hard_gate_triggered": False,
                    "hard_gates": {"sql_wal_pressure": False},
                    "thresholds": {"sql_wal_size_gb_limit": 24.0, "ingestion_pending_lines_limit": 20000},
                    "inputs": {"backpressure_overload_severe": False, "backpressure_pending_lines": 0, "sql_wal_size_gb_live": 1.0},
                },
            )
        return {"cmd": list(cmd), "rc": 0, "timed_out": False, "stdout_tail": "{}", "stderr_tail": "", "payload": {}}

    monkeypatch.setattr(clearance_src, "_run_json", _fake_run_json)

    payload = clearance_src.build_payload(project_root, apply=True, force_clear_stale_gate=True)

    assert payload["overall_status"] == "ready"
    assert payload["force_clear_refused_reason"] == ""
    assert payload["metrics"]["active_storage_pressure"] is False
    assert any("health_gates.py" in cmd for cmd in seen)
    assert any("global_risk_killswitch.py" in cmd for cmd in seen)


def test_storage_pressure_clearance_keeps_bounded_draining_soft_target_paper_safe(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "elevated",
            "pressure_index": 0.91,
            "backpressure": {
                "core_pending_lines": 991,
                "total_pending_lines": 2237,
                "oldest_pending_age_seconds": 219.0,
            },
            "storage": {"sqlite_wal_size_gb": 0.0, "backlog_drain_status": "drain_active"},
            "data_integrity": {
                "sql_overlay_invalid_lines": 0,
                "sql_overlay_oversize_payloads": 0,
                "sql_overlay_ops_write_failures": 0,
            },
            "bounded_recovery_contract": {"route_verified": True, "active_drain_progress": True},
            "steady_state": {
                "targets": {"pressure_index": 0.25, "core_pending_lines": 5000},
                "target_status": {"steady_state_ready": False},
            },
        },
    )
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gates": {},
            "thresholds": {"sql_wal_size_gb_limit": 24.0, "ingestion_pending_lines_limit": 20000},
            "inputs": {
                "backpressure_overload_severe": False,
                "backpressure_pending_lines": 2237,
                "sql_wal_size_gb_live": 0.0,
            },
        },
    )

    payload = clearance_src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    after = payload["storage_pressure"]["after"]
    assert after["active_storage_pressure"] is False
    assert after["bounded_soft_target_pressure"] is True
    assert after["soft_pressure_advisory_reasons"] == ["pressure_index_above_target"]


def test_storage_pressure_clearance_accepts_small_verified_queue_waiting_for_off_hours(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "elevated",
            "pressure_index": 0.797,
            "backpressure": {"core_pending_lines": 233, "total_pending_lines": 235},
            "storage": {"sqlite_wal_size_gb": 0.0, "backlog_drain_status": "waiting_for_off_hours"},
            "data_integrity": {
                "sql_overlay_invalid_lines": 0,
                "sql_overlay_oversize_payloads": 0,
                "sql_overlay_ops_write_failures": 0,
            },
            "bounded_recovery_contract": {"route_verified": True},
            "steady_state": {
                "targets": {"pressure_index": 0.25, "core_pending_lines": 5000},
                "target_status": {"steady_state_ready": False},
            },
        },
    )
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gates": {},
            "thresholds": {"sql_wal_size_gb_limit": 24.0, "ingestion_pending_lines_limit": 20000},
            "inputs": {},
        },
    )

    payload = clearance_src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    after = payload["storage_pressure"]["after"]
    assert after["active_storage_pressure"] is False
    assert after["bounded_soft_target_pressure"] is True
    assert after["backlog_drain_status"] == "waiting_for_off_hours"


def test_storage_pressure_clearance_accepts_verified_intentional_local_hot_route(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "pressure_index": 0.01,
            "backpressure": {"core_pending_lines": 100, "total_pending_lines": 200},
            "storage": {"sqlite_wal_size_gb": 0.0, "backlog_drain_status": "drain_active"},
            "bounded_recovery_contract": {"route_verified": True, "active_drain_progress": True},
            "steady_state": {
                "targets": {"pressure_index": 0.25, "core_pending_lines": 5000},
                "target_status": {"steady_state_ready": True},
            },
        },
    )
    _write_json(health / "health_gates_latest.json", {"hard_gates": {}, "thresholds": {}})
    _write_json(
        health / "storage_mount_guard_latest.json",
        {
            "storage_mode": "local_fallback",
            "external_available": False,
            "external_unavailable_reason": "cold_archive_only_local_hot_storage_policy",
            "external_required_for_hot_path": False,
            "hot_storage_available": True,
        },
    )

    payload = clearance_src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    after = payload["storage_pressure"]["after"]
    assert after["intentional_local_hot_route"] is True
    assert after["route_blocked"] is False


def test_storage_pressure_clearance_still_blocks_at_hard_pressure_envelope(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "elevated",
            "pressure_index": 1.0,
            "backpressure": {"core_pending_lines": 991, "total_pending_lines": 2237},
            "storage": {"sqlite_wal_size_gb": 0.0, "backlog_drain_status": "drain_active"},
            "data_integrity": {
                "sql_overlay_invalid_lines": 0,
                "sql_overlay_oversize_payloads": 0,
                "sql_overlay_ops_write_failures": 0,
            },
            "bounded_recovery_contract": {"route_verified": True, "active_drain_progress": True},
            "steady_state": {
                "targets": {"pressure_index": 0.25, "core_pending_lines": 5000},
                "target_status": {"steady_state_ready": False},
            },
        },
    )
    _write_json(
        health / "health_gates_latest.json",
        {
            "hard_gates": {},
            "thresholds": {"sql_wal_size_gb_limit": 24.0, "ingestion_pending_lines_limit": 20000},
            "inputs": {},
        },
    )

    payload = clearance_src.build_payload(project_root)

    assert payload["overall_status"] == "blocked"
    assert payload["storage_pressure"]["after"]["active_storage_pressure"] is True
