import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import external_backlog_retry_bot as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_external_backlog_retry_bot_waits_for_off_hours_when_backlog_exists(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    monkeypatch.setattr(
        src.drain_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "blocked",
            "blocked_reasons": ["market_hours_guard"],
            "off_hours_window": {"active": False},
            "backpressure_before": {"deferred_pending_lines": 1200, "cold_pending_lines": 50},
            "aged_candidate_files": 1,
            "recommended_now": False,
            "storage_mode": "external",
            "writer_busy": False,
        },
    )

    payload = src.build_payload(project_root, apply=True)

    assert payload["overall_status"] == "waiting_for_off_hours"
    assert payload["actionable"] is False
    assert payload["backlog_needed"] is True


def test_external_backlog_retry_bot_runs_drain_and_refreshes_surfaces(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    monkeypatch.setattr(
        src.drain_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "ready",
            "blocked_reasons": [],
            "off_hours_window": {"active": True},
            "backpressure_before": {"deferred_pending_lines": 8000, "cold_pending_lines": 0},
            "aged_candidate_files": 2,
            "recommended_now": True,
            "storage_mode": "external",
            "writer_busy": True,
        },
    )

    def _fake_run(cmd: list[str], *, cwd: Path, payload_path: Path | None = None, timeout_sec: int) -> dict:
        joined = " ".join(cmd)
        if "external_backlog_drain.py" in joined:
            payload = {
                "overall_status": "drain_active",
                "writer_busy": True,
                "follow_through": {"status": "timed_out", "progress_state": "stalled", "attempts": 4, "waited_seconds": 1.0},
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
        return {"cmd": cmd, "rc": 0, "duration_ms": 8.0, "payload": payload, "stdout_tail": "", "stderr_tail": "", "timed_out": False}

    monkeypatch.setattr(src, "_run_json_command", _fake_run)

    payload = src.build_payload(project_root, apply=True, wait_timeout_seconds=30.0)

    assert payload["overall_status"] == "applied_with_followups"
    assert payload["actionable"] is True
    assert payload["drain_result"]["follow_through_status"] == "timed_out"
    assert payload["steps"]["external_backlog_drain"]["status"] == "ok"
    assert payload["refresh_steps"]["runtime_gate_dashboard"]["status"] == "ok"


def test_external_backlog_retry_bot_marks_progressing_follow_through_as_healthy(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    monkeypatch.setattr(
        src.drain_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "ready",
            "blocked_reasons": [],
            "off_hours_window": {"active": True},
            "backpressure_before": {"deferred_pending_lines": 8000, "cold_pending_lines": 0},
            "aged_candidate_files": 2,
            "recommended_now": True,
            "storage_mode": "external",
            "writer_busy": True,
        },
    )

    def _fake_run(cmd: list[str], *, cwd: Path, payload_path: Path | None = None, timeout_sec: int) -> dict:
        joined = " ".join(cmd)
        if "external_backlog_drain.py" in joined:
            payload = {
                "overall_status": "drain_active",
                "writer_busy": True,
                "follow_through": {
                    "status": "timed_out",
                    "progress_state": "progressing",
                    "progress_observed": True,
                    "progress_events": 3,
                    "attempts": 4,
                    "waited_seconds": 1.0,
                },
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
        return {"cmd": cmd, "rc": 0, "duration_ms": 8.0, "payload": payload, "stdout_tail": "", "stderr_tail": "", "timed_out": False}

    monkeypatch.setattr(src, "_run_json_command", _fake_run)

    payload = src.build_payload(project_root, apply=True, wait_timeout_seconds=30.0)

    assert payload["overall_status"] == "applied_progressing"
    assert payload["ok"] is True
    assert payload["drain_result"]["follow_through_status"] == "timed_out"
    assert payload["drain_result"]["follow_through_progress_state"] == "progressing"


def test_external_backlog_retry_bot_accepts_completed_busy_handoff(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    monkeypatch.setattr(
        src.drain_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "ready",
            "blocked_reasons": [],
            "off_hours_window": {"active": True},
            "backpressure_before": {"deferred_pending_lines": 8000, "cold_pending_lines": 0},
            "aged_candidate_files": 0,
            "recommended_now": True,
            "storage_mode": "external",
            "writer_busy": True,
        },
    )

    def _fake_run(cmd: list[str], *, cwd: Path, payload_path: Path | None = None, timeout_sec: int) -> dict:
        joined = " ".join(cmd)
        if "external_backlog_drain.py" in joined:
            payload = {
                "overall_status": "drain_active",
                "writer_busy": True,
                "follow_through": {
                    "completed": True,
                    "status": "handoff_requested",
                    "progress_state": "requested_live_writer",
                    "attempts": 1,
                    "waited_seconds": 0.0,
                },
            }
        elif "ingestion_storage_control.py" in joined:
            payload = {"overall_status": "ready"}
        elif "runtime_gate_dashboard.py" in joined:
            payload = {"overall": {"status": "ok"}}
        elif "operator_cockpit.py" in joined:
            payload = {"overall_status": "ready"}
        else:
            raise AssertionError(f"unexpected command: {cmd}")
        if payload_path is not None:
            _write_json(payload_path, payload)
        return {"cmd": cmd, "rc": 0, "duration_ms": 8.0, "payload": payload, "stdout_tail": "", "stderr_tail": "", "timed_out": False}

    monkeypatch.setattr(src, "_run_json_command", _fake_run)

    payload = src.build_payload(project_root, apply=True, wait_timeout_seconds=30.0)

    assert payload["overall_status"] == "applied_progressing"
    assert payload["ok"] is True
    assert payload["drain_result"]["follow_through_completed"] is True
    assert payload["drain_result"]["follow_through_status"] == "handoff_requested"


def test_external_backlog_retry_bot_accepts_not_needed_handoff_as_applied(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    monkeypatch.setattr(
        src.drain_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "ready",
            "blocked_reasons": [],
            "off_hours_window": {"active": True},
            "backpressure_before": {"deferred_pending_lines": 8000, "cold_pending_lines": 0},
            "aged_candidate_files": 0,
            "recommended_now": True,
            "storage_mode": "external",
            "writer_busy": False,
        },
    )

    def _fake_run(cmd: list[str], *, cwd: Path, payload_path: Path | None = None, timeout_sec: int) -> dict:
        joined = " ".join(cmd)
        if "external_backlog_drain.py" in joined:
            payload = {
                "overall_status": "ready",
                "writer_busy": False,
                "follow_through": {"status": "not_needed", "progress_state": "not_needed"},
            }
        elif "ingestion_storage_control.py" in joined:
            payload = {"overall_status": "ready"}
        elif "runtime_gate_dashboard.py" in joined:
            payload = {"overall": {"status": "ok"}}
        elif "operator_cockpit.py" in joined:
            payload = {"overall_status": "ready"}
        else:
            raise AssertionError(f"unexpected command: {cmd}")
        if payload_path is not None:
            _write_json(payload_path, payload)
        return {"cmd": cmd, "rc": 0, "duration_ms": 8.0, "payload": payload, "stdout_tail": "", "stderr_tail": "", "timed_out": False}

    monkeypatch.setattr(src, "_run_json_command", _fake_run)

    payload = src.build_payload(project_root, apply=True, wait_timeout_seconds=30.0)

    assert payload["overall_status"] == "applied"
    assert payload["ok"] is True
    assert payload["drain_result"]["follow_through_status"] == "not_needed"


def test_external_backlog_retry_bot_runs_quarantine_during_market_hours(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    monkeypatch.setattr(
        src.drain_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "blocked",
            "blocked_reasons": ["market_hours_guard"],
            "off_hours_window": {"active": False},
            "backpressure_before": {"deferred_pending_lines": 8000, "cold_pending_lines": 180000},
            "aged_candidate_files": 1,
            "recommended_now": False,
            "storage_mode": "external",
            "writer_busy": False,
        },
    )
    monkeypatch.setattr(
        src.quarantine_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "ready",
            "candidate_files": 2,
            "candidate_pending_lines": 240000,
        },
    )

    def _fake_run(cmd: list[str], *, cwd: Path, payload_path: Path | None = None, timeout_sec: int) -> dict:
        joined = " ".join(cmd)
        if "backlog_quarantine_bot.py" in joined:
            payload = {
                "overall_status": "applied",
                "moved_files": 2,
                "moved_pending_lines": 240000,
            }
        elif "ingestion_storage_control.py" in joined:
            payload = {"overall_status": "needs_work"}
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

    payload = src.build_payload(project_root, apply=True)

    assert payload["overall_status"] == "quarantine_applied_waiting_for_off_hours"
    assert payload["quarantine_actionable"] is True
    assert payload["quarantine_result"]["moved_files"] == 2
    assert payload["steps"]["backlog_quarantine_bot"]["status"] == "ok"


def test_external_backlog_retry_bot_main_writes_artifact_before_post_refresh(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    out_file = project_root / "governance" / "health" / "external_backlog_retry_bot_latest.json"
    lock_file = project_root / "governance" / "locks" / "external_backlog_retry_bot.lock"

    monkeypatch.setattr(
        src,
        "build_payload",
        lambda *args, **kwargs: {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "schema_version": 1,
            "ok": False,
            "overall_status": "applied_with_followups",
            "apply": True,
            "actionable": True,
            "backlog_needed": True,
        },
    )
    monkeypatch.setattr(
        src,
        "_refresh_surface_artifacts",
        lambda *args, **kwargs: {
            "runtime_gate_dashboard": {"status": "ok"},
            "operator_cockpit": {"status": "ok"},
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "external_backlog_retry_bot.py",
            "--project-root",
            str(project_root),
            "--out-file",
            str(out_file),
            "--lock-file",
            str(lock_file),
            "--apply",
        ],
    )

    rc = src.main()
    payload = json.loads(out_file.read_text(encoding="utf-8"))

    assert rc == 2
    assert payload["overall_status"] == "applied_with_followups"
    assert payload["post_write_refresh_steps"]["runtime_gate_dashboard"]["status"] == "ok"
