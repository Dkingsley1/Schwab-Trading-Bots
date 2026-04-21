from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPS_DIR = PROJECT_ROOT / "scripts" / "ops"
if str(OPS_DIR) not in sys.path:
    sys.path.insert(0, str(OPS_DIR))

import storage_transition_coordinator as coordinator_src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_storage_transition_coordinator_assigns_local_fallback_bots(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"

    _write_json(
        health / "storage_mount_guard_latest.json",
        {
            "external_available": True,
            "mount_present": True,
            "storage_mode": "local_fallback",
            "external_root": "/Volumes/BOT_LOGS/schwab_trading_bot",
        },
    )
    _write_json(health / "storage_failback_sync_latest.json", {"mode": "local_fallback"})
    _write_json(health / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 0}})
    _write_json(
        health / "storage_resilience_control_latest.json",
        {"overall_status": "ready", "ok": True, "top_actions": ["keep fallback warm"]},
    )
    _write_json(
        health / "storage_quota_guard_latest.json",
        {"overall_status": "degraded", "ok": False, "recommended_actions": ["trim local telemetry sooner"]},
    )
    _write_json(
        health / "storage_backpressure_autopilot_latest.json",
        {"overall_status": "ready", "ok": True, "recommended_actions": ["hold deferred quota-limited"]},
    )

    payload = coordinator_src.build_payload(project_root, transition_mode="local", apply=False)

    assert payload["transition_mode"] == "local"
    assert payload["current_storage_mode"] == "local_fallback"
    names = [row["name"] for row in payload["assigned_bots"]]
    assert names == [
        "storage_split_brain_reconciler",
        "storage_resilience_control",
        "storage_quota_guard",
        "storage_backpressure_autopilot",
    ]
    assert payload["overall_status"] == "degraded"
    assert "trim local telemetry sooner" in payload["recommended_actions"]
    assert "hold deferred quota-limited" in payload["recommended_actions"]


def test_storage_transition_coordinator_apply_refreshes_expected_steps(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(health / "storage_mount_guard_latest.json", {"storage_mode": "external", "external_available": True, "mount_present": True})
    _write_json(health / "storage_failback_sync_latest.json", {"mode": "external"})
    _write_json(health / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 0}})
    _write_json(health / "storage_resilience_control_latest.json", {"overall_status": "ready", "ok": True})
    _write_json(health / "ops_coordinator_latest.json", {"ok": True, "overall_status": "ready"})

    seen: list[str] = []

    def _fake_run(cmd: list[str], *, cwd: Path, payload_path: Path | None = None, timeout_sec: int = 120) -> dict:
        script_name = Path(cmd[1]).name
        seen.append(script_name)
        if payload_path is not None:
            payload_path.parent.mkdir(parents=True, exist_ok=True)
            if script_name == "process_watchdog.py":
                payload_path.write_text(json.dumps({"storage_mode": "external", "ok": True}), encoding="utf-8")
            elif script_name == "storage_split_brain_reconciler.py":
                payload_path.write_text(json.dumps({"summary": {"unresolved_conflicts": 0}}), encoding="utf-8")
            elif script_name == "storage_resilience_control.py":
                payload_path.write_text(json.dumps({"overall_status": "ready", "ok": True}), encoding="utf-8")
            elif script_name == "ops_coordinator.py":
                payload_path.write_text(json.dumps({"overall_status": "ready", "ok": True}), encoding="utf-8")
        return {
            "cmd": list(cmd),
            "rc": 0,
            "duration_ms": 5.0,
            "payload": {"ok": True, "overall_status": "ready"},
            "stdout_tail": "",
            "stderr_tail": "",
        }

    monkeypatch.setattr(coordinator_src, "_run_json_command", _fake_run)

    attempts = coordinator_src._apply_refresh(project_root, transition_mode="external")

    assert [row["name"] for row in attempts] == [
        "process_watchdog",
        "storage_split_brain_reconciler",
        "storage_resilience_control",
        "ops_coordinator",
    ]
    assert seen == [
        "process_watchdog.py",
        "storage_split_brain_reconciler.py",
        "storage_resilience_control.py",
        "ops_coordinator.py",
    ]
