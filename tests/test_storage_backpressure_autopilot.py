from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPS_DIR = PROJECT_ROOT / "scripts" / "ops"
if str(OPS_DIR) not in sys.path:
    sys.path.insert(0, str(OPS_DIR))

import storage_backpressure_autopilot as autopilot_src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_storage_backpressure_autopilot_builds_coordinated_plan(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "governance" / "health" / "ingestion_storage_control_latest.json",
        {
            "overall_status": "blocked",
            "severity": "critical",
            "backpressure": {
                "pending_lines_threshold": 15000,
                "total_pending_lines": 75000,
                "estimated_total_drain_minutes": 95.0,
            },
            "storage": {
                "retention_debt_gb": 5.8,
                "backlog_drain_recommended_now": True,
                "backlog_quarantine_candidate_files": 2,
            },
            "data_integrity": {"sql_invalid_lines": 1},
        },
    )

    monkeypatch.setattr(
        autopilot_src.backpressure_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "ready",
            "actionable": True,
            "recommended_profile": "critical_backpressure",
            "recommended_actions": ["apply the governor"],
        },
    )
    monkeypatch.setattr(
        autopilot_src.coordinator_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "ready",
            "actionable": True,
            "drain_ready": True,
            "maintenance_ready": True,
            "recommended_actions": ["run the drain window"],
        },
    )
    monkeypatch.setattr(
        autopilot_src.sheriff_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "ready",
            "actionable": True,
            "focus": {
                "focus_shards": ["explanations"],
                "targeted_retention_debt_gb": 3.1,
                "severe_focus": True,
            },
            "recommended_actions": ["drain explanation debt"],
        },
    )

    payload = autopilot_src.build_payload(project_root, apply=False)

    names = [row["name"] for row in payload["repair_plan"]]
    assert payload["overall_status"] == "ready"
    assert names == [
        "backpressure_slo_bot",
        "writer_cycle_coordinator",
        "retention_debt_sheriff",
    ]
    coordinator_cmd = next(row["cmd"] for row in payload["repair_plan"] if row["name"] == "writer_cycle_coordinator")
    sheriff_cmd = next(row["cmd"] for row in payload["repair_plan"] if row["name"] == "retention_debt_sheriff")
    assert "--maintenance-force" in coordinator_cmd
    assert "--force" in sheriff_cmd
    assert payload["metrics"]["repair_step_count"] == 3
    assert payload["previews"]["retention_debt_sheriff"]["severe_focus"] is True


def test_storage_backpressure_autopilot_applies_only_needed_lanes(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    storage_path = project_root / "governance" / "health" / "ingestion_storage_control_latest.json"
    _write_json(
        storage_path,
        {
            "overall_status": "needs_work",
            "severity": "high",
            "backpressure": {
                "pending_lines_threshold": 15000,
                "total_pending_lines": 42000,
                "estimated_total_drain_minutes": 70.0,
            },
            "storage": {
                "retention_debt_gb": 0.2,
                "backlog_drain_recommended_now": True,
                "backlog_quarantine_candidate_files": 0,
            },
            "data_integrity": {"sql_invalid_lines": 0},
        },
    )

    monkeypatch.setattr(
        autopilot_src.backpressure_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "ready",
            "actionable": True,
            "recommended_profile": "elevated_backpressure",
            "recommended_actions": [],
        },
    )
    monkeypatch.setattr(
        autopilot_src.coordinator_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "ready",
            "actionable": True,
            "drain_ready": True,
            "maintenance_ready": True,
            "recommended_actions": [],
        },
    )
    monkeypatch.setattr(
        autopilot_src.sheriff_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "idle",
            "actionable": False,
            "focus": {
                "focus_shards": [],
                "targeted_retention_debt_gb": 0.0,
                "severe_focus": False,
            },
            "recommended_actions": [],
        },
    )

    seen: list[str] = []

    def _fake_run_json(cmd: list[str], *, cwd: Path, timeout_sec: int) -> dict:
        script = Path(cmd[1]).name
        seen.append(script)
        if script == "writer_cycle_coordinator.py":
            _write_json(
                storage_path,
                {
                    "overall_status": "ready",
                    "severity": "elevated",
                    "backpressure": {
                        "pending_lines_threshold": 15000,
                        "total_pending_lines": 18000,
                        "estimated_total_drain_minutes": 12.0,
                    },
                    "storage": {
                        "retention_debt_gb": 0.0,
                        "backlog_drain_recommended_now": False,
                        "backlog_quarantine_candidate_files": 0,
                    },
                    "data_integrity": {"sql_invalid_lines": 0},
                },
            )
        return {
            "cmd": list(cmd),
            "rc": 0,
            "timed_out": False,
            "stdout_tail": "",
            "stderr_tail": "",
            "payload": {
                "bot": script.replace(".py", ""),
                "overall_status": "applied",
                "ok": True,
            },
        }

    monkeypatch.setattr(autopilot_src, "_run_json", _fake_run_json)

    payload = autopilot_src.build_payload(
        project_root,
        apply=True,
        poll_seconds=5.0,
        wait_timeout_seconds=30.0,
        command_timeout_seconds=60,
        backpressure_command_timeout_seconds=20,
    )

    assert payload["overall_status"] == "applied"
    assert payload["metrics"]["attempted_step_count"] == 2
    assert seen == [
        "backpressure_slo_bot.py",
        "writer_cycle_coordinator.py",
    ]
    assert payload["metrics"]["cycle_count"] == 1


def test_storage_backpressure_autopilot_repeats_cycles_until_targets_clear(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)
    storage_path = health / "ingestion_storage_control_latest.json"

    def _write_storage(total_pending_lines: int, retention_debt_gb: float, *, overall_status: str, severity: str) -> None:
        _write_json(
            storage_path,
            {
                "overall_status": overall_status,
                "severity": severity,
                "backpressure": {
                    "pending_lines_threshold": 15000,
                    "total_pending_lines": total_pending_lines,
                    "estimated_total_drain_minutes": 95.0,
                },
                "storage": {
                    "retention_debt_gb": retention_debt_gb,
                    "backlog_drain_recommended_now": total_pending_lines > 0,
                    "backlog_quarantine_candidate_files": 0,
                },
                "data_integrity": {"sql_invalid_lines": 0},
            },
        )

    _write_storage(76000, 6.4, overall_status="blocked", severity="critical")

    monkeypatch.setattr(
        autopilot_src.backpressure_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "ready",
            "actionable": True,
            "recommended_profile": "critical_backpressure",
            "recommended_actions": ["apply the governor"],
        },
    )

    def _coordinator_preview(*_args, **_kwargs) -> dict:
        current = json.loads(storage_path.read_text(encoding="utf-8"))
        total_pending_lines = current["backpressure"]["total_pending_lines"]
        return {
            "overall_status": "ready",
            "actionable": total_pending_lines > 0,
            "drain_ready": total_pending_lines > 0,
            "maintenance_ready": True,
            "recommended_actions": ["run the drain window"],
        }

    monkeypatch.setattr(autopilot_src.coordinator_src, "build_payload", _coordinator_preview)

    def _sheriff_preview(*_args, **_kwargs) -> dict:
        current = json.loads(storage_path.read_text(encoding="utf-8"))
        retention_debt_gb = float(current["storage"]["retention_debt_gb"])
        return {
            "overall_status": "ready" if retention_debt_gb > 0.0 else "idle",
            "actionable": retention_debt_gb > 0.0,
            "focus": {
                "focus_shards": ["explanations"] if retention_debt_gb > 0.0 else [],
                "targeted_retention_debt_gb": retention_debt_gb,
                "severe_focus": retention_debt_gb >= 5.0,
            },
            "recommended_actions": ["drain explanation debt"] if retention_debt_gb > 0.0 else [],
        }

    monkeypatch.setattr(autopilot_src.sheriff_src, "build_payload", _sheriff_preview)

    seen: list[list[str]] = []
    cycle_counter = {"count": 0}

    def _fake_run_json(cmd: list[str], *, cwd: Path, timeout_sec: int) -> dict:
        seen.append(list(cmd))
        script = Path(cmd[1]).name
        if script == "backpressure_slo_bot.py":
            payload = {"bot": "backpressure_slo_bot", "overall_status": "applied", "ok": True}
        elif script == "writer_cycle_coordinator.py":
            cycle_counter["count"] += 1
            if cycle_counter["count"] == 1:
                _write_storage(26000, 1.2, overall_status="needs_work", severity="high")
            else:
                _write_storage(15000, 0.1, overall_status="ready", severity="elevated")
            payload = {"bot": "writer_cycle_coordinator", "overall_status": "applied", "ok": True}
        elif script == "retention_debt_sheriff.py":
            payload = {"bot": "retention_debt_sheriff", "overall_status": "applied", "ok": True}
        else:
            raise AssertionError(f"unexpected command: {cmd}")
        return {
            "cmd": list(cmd),
            "rc": 0,
            "timed_out": False,
            "stdout_tail": "",
            "stderr_tail": "",
            "payload": payload,
        }

    monkeypatch.setattr(autopilot_src, "_run_json", _fake_run_json)

    payload = autopilot_src.build_payload(
        project_root,
        apply=True,
        poll_seconds=5.0,
        wait_timeout_seconds=30.0,
        command_timeout_seconds=60,
        backpressure_command_timeout_seconds=20,
        max_cycles=3,
        target_pending_lines=20000,
        target_retention_debt_gb=0.25,
    )

    assert payload["overall_status"] == "applied"
    assert payload["clearance_state"]["cleared"] is True
    assert payload["metrics"]["cycle_count"] == 2
    assert payload["metrics"]["attempted_step_count"] == 6
    assert payload["cycle_records"][0]["progress"]["progress_observed"] is True
    assert payload["cycle_records"][1]["clearance_after"]["cleared"] is True
    writer_cmds = [cmd for cmd in seen if Path(cmd[1]).name == "writer_cycle_coordinator.py"]
    assert len(writer_cmds) == 2
    assert "--maintenance-force" in writer_cmds[0]
    sheriff_cmds = [cmd for cmd in seen if Path(cmd[1]).name == "retention_debt_sheriff.py"]
    assert sheriff_cmds
    assert "--force" in sheriff_cmds[0]
