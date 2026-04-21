import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import writer_cycle_coordinator as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_writer_cycle_coordinator_waits_for_writer_and_runs_drain_then_maintenance(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    (project_root / "governance" / "health").mkdir(parents=True, exist_ok=True)

    writer_states = [
        {"active": False, "current_step": "complete"},
    ]

    def _fake_writer_state(*args, **kwargs) -> dict:
        if writer_states:
            return writer_states.pop(0)
        return {"active": False, "current_step": "complete"}

    monkeypatch.setattr(src, "writer_state_snapshot", _fake_writer_state)
    monkeypatch.setattr(src.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        src.drain_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "ready",
            "recommended_now": True,
            "blocked_reasons": [],
            "top_actions": ["drain deferred backlog"],
            "off_hours_window": {"active": True},
            "aged_candidate_files": 2,
            "writer_busy": False,
        },
    )
    monkeypatch.setattr(
        src.maintenance_src,
        "_priority_retention_focus",
        lambda *args, **kwargs: {
            "enabled": True,
            "focus_shards": ["explanations"],
            "targeted_retention_debt_gb": 51.055,
            "top_actions": ["focus explanation shards"],
        },
    )

    def _fake_run(cmd: list[str], *, cwd: Path, payload_path: Path | None = None, timeout_sec: int) -> dict:
        joined = " ".join(cmd)
        if "external_backlog_drain.py" in joined:
            payload = {"ok": True, "overall_status": "drain_active", "writer_busy": False, "follow_through": {"status": "completed"}}
        elif "storage_maintenance_lane.py" in joined:
            payload = {"ok": True, "reason": "ok"}
        elif "ingestion_storage_control.py" in joined:
            payload = {"overall_status": "blocked"}
        elif "runtime_gate_dashboard.py" in joined:
            payload = {"overall": {"status": "degraded"}}
        elif "operator_cockpit.py" in joined:
            payload = {"overall_status": "degraded"}
        else:
            raise AssertionError(f"unexpected command: {cmd}")
        if payload_path is not None:
            _write_json(payload_path, payload)
        return {"cmd": cmd, "rc": 0, "duration_ms": 8.0, "payload": payload, "stdout_tail": "", "stderr_tail": "", "timed_out": False}

    monkeypatch.setattr(src, "_run_json_command", _fake_run)

    payload = src.build_payload(project_root, apply=True, poll_seconds=0.0, wait_timeout_seconds=30.0)

    assert payload["overall_status"] == "applied"
    assert payload["wait_for_writer"]["completed"] is True
    assert payload["summary"]["wait_timed_out"] is False
    assert payload["summary"]["drain_applied"] is True
    assert payload["summary"]["maintenance_applied"] is True
    assert payload["steps"]["external_backlog_drain"]["status"] == "ok"
    assert payload["steps"]["storage_maintenance_lane"]["status"] == "ok"
    assert payload["refresh_steps"]["runtime_gate_dashboard"]["status"] == "ok"


def test_writer_cycle_coordinator_marks_progressing_writer_wait_as_healthy(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    (project_root / "governance" / "health").mkdir(parents=True, exist_ok=True)

    writer_states = [
        {
            "active": True,
            "running": True,
            "current_step": "merge_primary",
            "completed_shard_count": 13,
            "completed_merge_count": 9,
            "merged_rows_this_cycle": 1000,
        },
        {
            "active": True,
            "running": True,
            "current_step": "merge_primary",
            "completed_shard_count": 13,
            "completed_merge_count": 10,
            "merged_rows_this_cycle": 2500,
        },
    ]

    def _fake_writer_state(*args, **kwargs) -> dict:
        if writer_states:
            return writer_states.pop(0)
        return {
            "active": True,
            "running": True,
            "current_step": "merge_primary",
            "completed_shard_count": 13,
            "completed_merge_count": 10,
            "merged_rows_this_cycle": 2500,
        }

    monkeypatch.setattr(src, "writer_state_snapshot", _fake_writer_state)
    monkeypatch.setattr(src, "_wait_for_writer_idle", lambda *args, **kwargs: {"requested": True, "completed": False, "timed_out": True, "waited_seconds": 30.0, "attempts": 2, "final_state": _fake_writer_state()})
    monkeypatch.setattr(
        src.drain_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "idle",
            "recommended_now": False,
            "blocked_reasons": ["market_hours_guard"],
            "top_actions": [],
            "off_hours_window": {"active": False},
            "aged_candidate_files": 0,
            "writer_busy": False,
        },
    )
    monkeypatch.setattr(
        src.maintenance_src,
        "_priority_retention_focus",
        lambda *args, **kwargs: {"enabled": True, "focus_shards": ["explanations"], "top_actions": ["focus explanation shards"]},
    )

    payload = src.build_payload(project_root, apply=True, poll_seconds=0.0, wait_timeout_seconds=30.0)

    assert payload["overall_status"] == "progressing_waiting_for_writer"
    assert payload["ok"] is True
    assert payload["writer_progress"]["progress_observed"] is True
    assert payload["summary"]["writer_merged_rows_delta"] == 1500


def test_writer_cycle_coordinator_applies_drain_handoff_while_writer_active(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    (project_root / "governance" / "health").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        src,
        "writer_state_snapshot",
        lambda *args, **kwargs: {"active": True, "running": True, "current_step": "merge_primary", "merged_rows_this_cycle": 1800},
    )
    monkeypatch.setattr(
        src.drain_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "ready",
            "recommended_now": True,
            "blocked_reasons": [],
            "top_actions": ["drain deferred backlog"],
            "off_hours_window": {"active": True},
            "aged_candidate_files": 2,
            "writer_busy": True,
            "follow_through": {"status": "handoff_requested"},
        },
    )
    monkeypatch.setattr(
        src.maintenance_src,
        "_priority_retention_focus",
        lambda *args, **kwargs: {"enabled": True, "focus_shards": ["explanations"], "top_actions": ["focus explanation shards"]},
    )

    def _fake_run(cmd: list[str], *, cwd: Path, payload_path: Path | None = None, timeout_sec: int) -> dict:
        joined = " ".join(cmd)
        if "external_backlog_drain.py" in joined:
            payload = {
                "ok": True,
                "overall_status": "drain_active",
                "writer_busy": True,
                "follow_through": {"status": "handoff_requested"},
            }
        elif "ingestion_storage_control.py" in joined:
            payload = {"overall_status": "blocked"}
        elif "runtime_gate_dashboard.py" in joined:
            payload = {"overall": {"status": "degraded"}}
        elif "operator_cockpit.py" in joined:
            payload = {"overall_status": "degraded"}
        else:
            raise AssertionError(f"unexpected command: {cmd}")
        if payload_path is not None:
            _write_json(payload_path, payload)
        return {"cmd": cmd, "rc": 0, "duration_ms": 5.0, "payload": payload, "stdout_tail": "", "stderr_tail": "", "timed_out": False}

    monkeypatch.setattr(src, "_run_json_command", _fake_run)

    payload = src.build_payload(project_root, apply=True, poll_seconds=0.0, wait_timeout_seconds=30.0)

    assert payload["overall_status"] == "applied_with_followups"
    assert payload["summary"]["drain_applied"] is True
    assert payload["summary"]["maintenance_applied"] is False
    assert payload["steps"]["external_backlog_drain"]["status"] == "ok"
    assert "storage_maintenance_lane" not in payload["steps"]
