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


def test_writer_cycle_coordinator_rechecks_stale_writer_after_timeout(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    (project_root / "governance" / "health").mkdir(parents=True, exist_ok=True)

    initial_state = {
        "active": True,
        "running": True,
        "writer_lock_held": True,
        "writer_lock_owner": "pid=123 started=old cmd=sql_link_shard_manager",
        "current_step": "merge_primary",
        "progress_age_minutes": 2.0,
        "cycle_age_minutes": 10.0,
        "completed_shard_count": 18,
        "planned_shard_count": 18,
        "completed_merge_count": 0,
        "merged_rows_this_cycle": 0,
    }
    stale_after_wait = dict(initial_state, progress_age_minutes=45.0, cycle_age_minutes=55.0)
    recovered_state = {
        "active": False,
        "running": False,
        "writer_lock_held": False,
        "writer_lock_owner": "",
        "current_step": "complete",
        "progress_age_minutes": 0.0,
        "completed_merge_count": 0,
        "merged_rows_this_cycle": 0,
    }
    states = [initial_state, recovered_state]

    def _fake_writer_state(*args, **kwargs) -> dict:
        if states:
            return states.pop(0)
        return recovered_state

    monkeypatch.setattr(src, "writer_state_snapshot", _fake_writer_state)
    monkeypatch.setattr(
        src,
        "_wait_for_writer_idle",
        lambda *args, **kwargs: {
            "requested": True,
            "completed": False,
            "timed_out": True,
            "waited_seconds": 900.0,
            "attempts": 45,
            "final_state": stale_after_wait,
        },
    )
    monkeypatch.setattr(
        src,
        "_terminate_stale_writer",
        lambda *args, **kwargs: {
            "attempted": True,
            "needed": True,
            "pid": 123,
            "terminated": True,
            "lock_released": True,
            "reason": "terminated",
        },
    )
    monkeypatch.setattr(
        src.drain_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "idle",
            "recommended_now": False,
            "blocked_reasons": [],
            "top_actions": [],
            "off_hours_window": {"active": False},
            "aged_candidate_files": 0,
            "writer_busy": False,
        },
    )
    monkeypatch.setattr(
        src.drainer_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "idle",
            "ready_drainer_count": 0,
            "blocked_reasons": [],
            "recommended_actions": [],
            "active_drainer": {},
        },
    )
    monkeypatch.setattr(
        src.maintenance_src,
        "_priority_retention_focus",
        lambda *args, **kwargs: {"enabled": False, "focus_shards": [], "top_actions": []},
    )

    payload = src.build_payload(project_root, apply=True, poll_seconds=0.0, wait_timeout_seconds=900.0)

    assert payload["post_wait_stale_writer_remediation"]["attempted"] is True
    assert payload["summary"]["post_wait_stale_writer_detected"] is True
    assert payload["summary"]["post_wait_stale_writer_terminated"] is True
    assert payload["writer_state_after_wait"]["active"] is False


def test_writer_drain_effectiveness_suppresses_no_progress_when_backlog_is_calm() -> None:
    before = {
        "pressure_index": 0.03,
        "backpressure": {
            "total_pending_lines": 1002,
            "core_pending_lines": 397,
            "oldest_pending_age_seconds": 0.0,
            "pending_lines_threshold": 15000,
            "oldest_age_threshold_seconds": 240.0,
            "raw_live": {"line_estimation": {"sparse_large_line_pending_bytes": 100}},
        },
    }
    after = {
        "pressure_index": 0.03,
        "backpressure": {
            "total_pending_lines": 1002,
            "core_pending_lines": 397,
            "oldest_pending_age_seconds": 0.0,
            "pending_lines_threshold": 15000,
            "oldest_age_threshold_seconds": 240.0,
            "raw_live": {"line_estimation": {"sparse_large_line_pending_bytes": 100}},
        },
    }

    result = src._drain_effectiveness_score(before, after, merged_rows=0, waves_run=0)

    assert result["status"] == "settled_no_action_needed"
    assert result["false_alarm_guard"]["suppressed_no_progress_alarm"] is True
    assert "already inside target" in result["next_action"]


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


def test_writer_state_snapshot_ignores_unheld_stale_lock_file(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    locks = project_root / "governance" / "locks"
    health.mkdir(parents=True, exist_ok=True)
    locks.mkdir(parents=True, exist_ok=True)
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "timestamp_utc": "2026-05-02T10:00:00+00:00",
            "status": "running",
            "running": True,
            "current_step": "merge_primary",
        },
    )
    (locks / "jsonl_sql_writer.lock").write_text("pid=123 started=old cmd=sql_link_shard_manager", encoding="utf-8")

    state = src.writer_state_snapshot(project_root, now_utc=src.datetime.fromisoformat("2026-05-02T12:00:00+00:00"))

    assert state["writer_lock_held"] is False
    assert state["active"] is False


def test_writer_state_snapshot_marks_unowned_running_progress_orphaned_after_grace(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    locks = project_root / "governance" / "locks"
    health.mkdir(parents=True, exist_ok=True)
    locks.mkdir(parents=True, exist_ok=True)
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "timestamp_utc": "2026-05-02T11:55:00+00:00",
            "status": "running",
            "running": True,
            "current_step": "shard_linking",
        },
    )
    (locks / "jsonl_sql_writer.lock").write_text("", encoding="utf-8")

    state = src.writer_state_snapshot(project_root, now_utc=src.datetime.fromisoformat("2026-05-02T12:00:00+00:00"))

    assert state["writer_lock_held"] is False
    assert state["active"] is False
    assert state["progress_orphaned"] is True
    assert state["active_source"] == "orphaned_progress"


def test_writer_state_snapshot_marks_dead_owner_pid_progress_orphaned_after_grace(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    locks = project_root / "governance" / "locks"
    health.mkdir(parents=True, exist_ok=True)
    locks.mkdir(parents=True, exist_ok=True)
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "timestamp_utc": "2026-05-02T11:55:00+00:00",
            "status": "running",
            "running": True,
            "current_step": "shard_linking",
        },
    )
    (locks / "jsonl_sql_writer.lock").write_text("pid=999999 started=old cmd=sql_link_shard_manager", encoding="utf-8")

    state = src.writer_state_snapshot(project_root, now_utc=src.datetime.fromisoformat("2026-05-02T12:00:00+00:00"))

    assert state["writer_lock_held"] is False
    assert state["writer_owner_pid_live"] is False
    assert state["active"] is False
    assert state["progress_orphaned"] is True


def test_writer_state_snapshot_allows_short_unowned_running_progress_grace(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    locks = project_root / "governance" / "locks"
    health.mkdir(parents=True, exist_ok=True)
    locks.mkdir(parents=True, exist_ok=True)
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "timestamp_utc": "2026-05-02T11:59:00+00:00",
            "status": "running",
            "running": True,
            "current_step": "shard_linking",
        },
    )
    (locks / "jsonl_sql_writer.lock").write_text("", encoding="utf-8")

    state = src.writer_state_snapshot(project_root, now_utc=src.datetime.fromisoformat("2026-05-02T12:00:00+00:00"))

    assert state["writer_lock_held"] is False
    assert state["active"] is True
    assert state["progress_orphaned"] is False
    assert state["active_source"] == "recent_progress"


def test_writer_state_snapshot_surfaces_active_child_after_reported_complete(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    locks = project_root / "governance" / "locks"
    health.mkdir(parents=True, exist_ok=True)
    locks.mkdir(parents=True, exist_ok=True)
    _write_json(
        health / "sql_link_service_progress_latest.json",
        {
            "timestamp_utc": "2026-05-02T11:59:00+00:00",
            "status": "ok",
            "running": False,
            "current_step": "complete",
            "completed_shard_count": 25,
            "completed_merge_count": 25,
        },
    )
    (locks / "jsonl_sql_writer.lock").write_text("pid=123 started=old cmd=sql_link_shard_manager", encoding="utf-8")
    monkeypatch.setattr(src, "_lock_snapshot", lambda *_args, **_kwargs: {"held": True, "owner": "pid=123 started=old cmd=sql_link_shard_manager"})
    monkeypatch.setattr(src, "_pid_exists", lambda pid: int(pid) == 123)
    monkeypatch.setattr(src, "_child_writer_processes", lambda pid: [{"pid": 456, "command": "python scripts/link_jsonl_to_sql.py --mode sqlite"}])

    state = src.writer_state_snapshot(project_root, now_utc=src.datetime.fromisoformat("2026-05-02T12:00:00+00:00"))

    assert state["active"] is True
    assert state["child_writer_active"] is True
    assert state["active_child_writer_count"] == 1
    assert state["effective_current_step"] == "shard_worker_active_after_reported_complete"


def test_stale_writer_detection_uses_semantic_cycle_progress_not_heartbeat_only() -> None:
    state = {
        "active": True,
        "writer_lock_held": True,
        "current_step": "shard_linking",
        "progress_age_minutes": 0.25,
        "cycle_age_minutes": 310.0,
        "completed_shard_count": 12,
        "planned_shard_count": 18,
        "timed_out_shard_count": 4,
        "completed_merge_count": 0,
        "merged_rows_this_cycle": 0,
    }

    assert src._stale_writer_detected(state, stale_progress_minutes=30.0) is True


def test_stale_writer_detection_allows_long_cycle_with_recent_semantic_progress() -> None:
    state = {
        "active": True,
        "writer_lock_held": True,
        "current_step": "merge_primary",
        "progress_age_minutes": 0.25,
        "cycle_age_minutes": 45.0,
        "completed_shard_count": 18,
        "planned_shard_count": 18,
        "timed_out_shard_count": 0,
        "completed_merge_count": 4,
        "merged_rows_this_cycle": 24000,
    }

    assert src._stale_writer_detected(state, stale_progress_minutes=30.0) is False


def test_writer_cycle_coordinator_recovers_stale_writer_before_drain(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    (project_root / "governance" / "health").mkdir(parents=True, exist_ok=True)

    states = [
        {
            "active": True,
            "running": True,
            "writer_lock_held": True,
            "writer_lock_owner": "pid=123 started=old cmd=sql_link_shard_manager",
            "current_step": "merge_primary",
            "progress_age_minutes": 45.0,
            "merged_rows_this_cycle": 16,
        },
        {
            "active": False,
            "running": False,
            "writer_lock_held": False,
            "writer_lock_owner": "",
            "current_step": "complete",
            "progress_age_minutes": 0.0,
            "merged_rows_this_cycle": 16,
        },
    ]

    def _fake_writer_state(*args, **kwargs) -> dict:
        if states:
            return states.pop(0)
        return {
            "active": False,
            "running": False,
            "writer_lock_held": False,
            "writer_lock_owner": "",
            "current_step": "complete",
            "progress_age_minutes": 0.0,
            "merged_rows_this_cycle": 16,
        }

    monkeypatch.setattr(src, "writer_state_snapshot", _fake_writer_state)
    monkeypatch.setattr(
        src,
        "_terminate_stale_writer",
        lambda *args, **kwargs: {
            "attempted": True,
            "needed": True,
            "pid": 123,
            "terminated": True,
            "lock_released": True,
            "reason": "terminated",
        },
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
        lambda *args, **kwargs: {"enabled": False, "focus_shards": [], "top_actions": []},
    )

    def _fake_run(cmd: list[str], *, cwd: Path, payload_path: Path | None = None, timeout_sec: int) -> dict:
        joined = " ".join(cmd)
        if "external_backlog_drain.py" in joined:
            payload = {"ok": True, "overall_status": "drain_active", "writer_busy": False, "follow_through": {"status": "completed"}}
        elif "ingestion_storage_control.py" in joined:
            payload = {"overall_status": "blocked"}
        elif "runtime_gate_dashboard.py" in joined:
            payload = {"overall": {"status": "degraded"}}
        elif "operator_cockpit.py" in joined:
            payload = {"overall_status": "degraded"}
        else:
            raise AssertionError(f"unexpected command: {cmd}")
        return {"cmd": cmd, "rc": 0, "duration_ms": 5.0, "payload": payload, "stdout_tail": "", "stderr_tail": "", "timed_out": False}

    monkeypatch.setattr(src, "_run_json_command", _fake_run)

    payload = src.build_payload(project_root, apply=True, poll_seconds=0.0, wait_timeout_seconds=30.0)

    assert payload["summary"]["stale_writer_detected"] is True
    assert payload["summary"]["stale_writer_terminated"] is True
    assert payload["steps"]["external_backlog_drain"]["status"] == "ok"



def test_writer_cycle_coordinator_runs_live_safe_drainer_handoff(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    (project_root / "governance" / "health").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        src,
        "writer_state_snapshot",
        lambda *args, **kwargs: {"active": False, "running": False, "current_step": "complete"},
    )
    monkeypatch.setattr(
        src.drain_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "blocked",
            "recommended_now": False,
            "blocked_reasons": ["market_hours_guard"],
            "top_actions": [],
            "writer_busy": False,
        },
    )
    monkeypatch.setattr(
        src.drainer_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "ready",
            "ready_drainer_count": 1,
            "blocked_reasons": [],
            "recommended_actions": ["keep one SQL writer active"],
            "active_drainer": {
                "name": "core_decision_drainer",
                "status": "ready",
                "live_window_safe": True,
            },
        },
    )
    monkeypatch.setattr(
        src.maintenance_src,
        "_priority_retention_focus",
        lambda *args, **kwargs: {"enabled": False, "focus_shards": [], "top_actions": []},
    )

    def _fake_run(cmd: list[str], *, cwd: Path, payload_path: Path | None = None, timeout_sec: int) -> dict:
        joined = " ".join(cmd)
        if "backpressure_drainer_fleet.py" in joined:
            payload = {
                "ok": True,
                "overall_status": "handoff_requested",
                "service_request": {
                    "active": True,
                    "env_overrides": {"SQL_LINK_SERVICE_SHARDS": "trading,health_fast"},
                },
            }
        elif "sql_link_shard_manager.py" in joined:
            payload = {"ok": True, "rc": 0, "merged_rows_this_cycle": 1200}
        elif "ingestion_storage_control.py" in joined:
            payload = {"overall_status": "needs_work"}
        elif "runtime_gate_dashboard.py" in joined:
            payload = {"overall": {"status": "ready"}}
        elif "operator_cockpit.py" in joined:
            payload = {"overall_status": "ready"}
        else:
            raise AssertionError(f"unexpected command: {cmd}")
        if payload_path is not None:
            _write_json(payload_path, payload)
        return {"cmd": cmd, "rc": 0, "duration_ms": 5.0, "payload": payload, "stdout_tail": "", "stderr_tail": "", "timed_out": False}

    monkeypatch.setattr(src, "_run_json_command", _fake_run)

    payload = src.build_payload(project_root, apply=True, poll_seconds=0.0, wait_timeout_seconds=30.0)

    assert payload["overall_status"] == "applied"
    assert payload["live_drainer_ready"] is True
    assert payload["summary"]["live_drainer_applied"] is True
    assert payload["summary"]["active_drainer"] == "core_decision_drainer"
    assert "backpressure_drainer_fleet" in payload["steps"]
    assert "sql_link_shard_manager" in payload["steps"]
    assert "external_backlog_drain" not in payload["steps"]


def test_writer_cycle_coordinator_force_live_window_handoffs_protected_drainer(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    (project_root / "governance" / "health").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(src, "writer_state_snapshot", lambda *args, **kwargs: {"active": False, "running": False, "current_step": "complete"})
    monkeypatch.setattr(
        src.drain_src,
        "build_payload",
        lambda *args, **kwargs: {"overall_status": "blocked", "recommended_now": False, "blocked_reasons": ["market_hours_guard"], "top_actions": []},
    )
    seen_drainer_kwargs: list[dict] = []

    def _drainer_payload(*args, **kwargs):
        seen_drainer_kwargs.append(kwargs)
        return {
            "overall_status": "handoff_requested" if kwargs.get("apply") else "ready",
            "ready_drainer_count": 1,
            "blocked_reasons": [] if kwargs.get("force_live_window") else ["market_hours_guard"],
            "recommended_actions": [],
            "active_drainer": {"name": "cold_stage_drainer", "status": "ready", "live_window_safe": False},
            "service_request": {
                "active": True,
                "env_overrides": {"SQL_LINK_SERVICE_SHARDS": "data,explanations,crypto_explanations,health_fast"},
            } if kwargs.get("apply") else {},
        }

    monkeypatch.setattr(src.drainer_src, "build_payload", _drainer_payload)
    monkeypatch.setattr(src.maintenance_src, "_priority_retention_focus", lambda *args, **kwargs: {"enabled": False, "focus_shards": [], "top_actions": []})

    commands: list[list[str]] = []

    def _fake_run(cmd: list[str], *, cwd: Path, payload_path: Path | None = None, timeout_sec: int) -> dict:
        commands.append(cmd)
        joined = " ".join(cmd)
        if "backpressure_drainer_fleet.py" in joined:
            payload = {
                "ok": True,
                "overall_status": "handoff_requested",
                "service_request": {
                    "active": True,
                    "env_overrides": {"SQL_LINK_SERVICE_SHARDS": "data,explanations,crypto_explanations,health_fast"},
                },
            }
        elif "sql_link_shard_manager.py" in joined:
            payload = {"ok": True, "merged_rows_this_cycle": 44000}
        elif "ingestion_storage_control.py" in joined:
            payload = {"overall_status": "needs_work"}
        elif "runtime_gate_dashboard.py" in joined:
            payload = {"overall": {"status": "ready"}}
        elif "operator_cockpit.py" in joined:
            payload = {"overall_status": "ready"}
        else:
            raise AssertionError(f"unexpected command: {cmd}")
        if payload_path is not None:
            _write_json(payload_path, payload)
        return {"cmd": cmd, "rc": 0, "duration_ms": 5.0, "payload": payload, "stdout_tail": "", "stderr_tail": "", "timed_out": False}

    monkeypatch.setattr(src, "_run_json_command", _fake_run)

    payload = src.build_payload(project_root, apply=True, poll_seconds=0.0, wait_timeout_seconds=30.0, force_live_window=True)

    assert payload["live_drainer_ready"] is True
    assert payload["summary"]["active_drainer"] == "cold_stage_drainer"
    assert payload["settings"]["force_live_window"] is True
    assert any("--force-live-window" in command for command in commands if "backpressure_drainer_fleet.py" in " ".join(command))
    assert any(kwargs.get("force_live_window") is True for kwargs in seen_drainer_kwargs)


def test_writer_cycle_coordinator_surfaces_catch_up_followup_from_writer(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    (project_root / "governance" / "health").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(src, "writer_state_snapshot", lambda *args, **kwargs: {"active": False, "running": False, "current_step": "complete"})
    monkeypatch.setattr(
        src.drain_src,
        "build_payload",
        lambda *args, **kwargs: {"overall_status": "blocked", "recommended_now": False, "blocked_reasons": ["market_hours_guard"], "top_actions": []},
    )
    monkeypatch.setattr(
        src.drainer_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "ready",
            "ready_drainer_count": 1,
            "blocked_reasons": [],
            "recommended_actions": [],
            "active_drainer": {"name": "core_decision_drainer", "status": "ready", "live_window_safe": True},
        },
    )
    monkeypatch.setattr(src.maintenance_src, "_priority_retention_focus", lambda *args, **kwargs: {"enabled": False, "focus_shards": [], "top_actions": []})

    shard_runs = {"count": 0}

    def _fake_run(cmd: list[str], *, cwd: Path, payload_path: Path | None = None, timeout_sec: int) -> dict:
        joined = " ".join(cmd)
        if "backpressure_drainer_fleet.py" in joined:
            payload = {
                "ok": True,
                "overall_status": "handoff_requested",
                "service_request": {
                    "active": True,
                    "env_overrides": {
                        "SQL_LINK_SERVICE_SHARDS": "trading",
                        "WRITER_CYCLE_MAX_CATCH_UP_WAVES": "2",
                    },
                },
            }
        elif "sql_link_shard_manager.py" in joined:
            shard_runs["count"] += 1
            followup_needed = shard_runs["count"] == 1
            payload = {
                "ok": True,
                "rc": 0,
                "merged_rows_this_cycle": 12000,
                "merge_followup": {
                    "followup_needed": followup_needed,
                    "catch_up_recommended": followup_needed,
                    "followup_reasons": ["merge_row_cap_remaining"] if followup_needed else [],
                },
            }
        elif "ingestion_storage_control.py" in joined:
            payload = {"overall_status": "needs_work"}
        elif "runtime_gate_dashboard.py" in joined:
            payload = {"overall": {"status": "degraded"}}
        elif "operator_cockpit.py" in joined:
            payload = {"overall_status": "degraded"}
        else:
            raise AssertionError(f"unexpected command: {cmd}")
        return {"cmd": cmd, "rc": 0, "duration_ms": 5.0, "payload": payload, "stdout_tail": "", "stderr_tail": "", "timed_out": False}

    monkeypatch.setattr(src, "_run_json_command", _fake_run)

    payload = src.build_payload(project_root, apply=True, poll_seconds=0.0, wait_timeout_seconds=30.0)

    assert payload["overall_status"] == "applied"
    assert payload["summary"]["catch_up_followup_needed"] is False
    assert payload["summary"]["catch_up_waves_run"] == 2
    assert payload["catch_up_wave_controller"]["waves_run"] == 2
    assert payload["catch_up_wave_controller"]["followup_remaining"] is False
    assert payload["steps"]["sql_link_shard_manager_wave_2"]["status"] == "ok"


def test_writer_cycle_coordinator_chains_storage_contract_catch_up_waves(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WRITER_CYCLE_MAX_CATCH_UP_WAVES", "3")
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "backpressure": {
                "core_pending_lines": 50000,
                "total_pending_lines": 60000,
                "oldest_pending_age_seconds": 7200,
                "raw_live": {"line_estimation": {"sparse_large_line_pending_bytes": 1000}},
            },
            "sql_ingestion_pending_overlay": {"total_pending_lines": 60000},
            "backlog_relief_contract": {
                "active": True,
                "active_issue_ids": ["single_writer_merge_speed", "stale_old_pending_work"],
                "p_core_backlog_allocation_contract": {
                    "catch_up_wave_controller": {"enabled": True, "max_waves": 3}
                },
            },
        },
    )

    monkeypatch.setattr(src, "writer_state_snapshot", lambda *args, **kwargs: {"active": False, "running": False, "current_step": "complete"})
    monkeypatch.setattr(
        src.drain_src,
        "build_payload",
        lambda *args, **kwargs: {"overall_status": "blocked", "recommended_now": False, "blocked_reasons": ["market_hours_guard"], "top_actions": []},
    )
    monkeypatch.setattr(
        src.drainer_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "ready",
            "ready_drainer_count": 1,
            "blocked_reasons": [],
            "recommended_actions": [],
            "active_drainer": {"name": "core_decision_drainer", "status": "ready", "live_window_safe": True},
        },
    )
    monkeypatch.setattr(src.maintenance_src, "_priority_retention_focus", lambda *args, **kwargs: {"enabled": False, "focus_shards": [], "top_actions": []})
    shard_runs = {"count": 0}

    def _fake_run(cmd: list[str], *, cwd: Path, payload_path: Path | None = None, timeout_sec: int) -> dict:
        joined = " ".join(cmd)
        if "backpressure_drainer_fleet.py" in joined:
            payload = {"ok": True, "overall_status": "handoff_requested", "service_request": {"active": True, "env_overrides": {"SQL_LINK_SERVICE_SHARDS": "trading"}}}
        elif "sql_link_shard_manager.py" in joined:
            shard_runs["count"] += 1
            payload = {"ok": True, "rc": 0, "merged_rows_this_cycle": 5000, "merge_followup": {"followup_needed": False}}
        elif "ingestion_storage_control.py" in joined:
            payload = {
                "backpressure": {
                    "core_pending_lines": 35000,
                    "total_pending_lines": 40000,
                    "oldest_pending_age_seconds": 3600,
                    "raw_live": {"line_estimation": {"sparse_large_line_pending_bytes": 500}},
                },
                "sql_ingestion_pending_overlay": {"total_pending_lines": 40000},
            }
            if payload_path is not None:
                _write_json(payload_path, payload)
        elif "runtime_gate_dashboard.py" in joined:
            payload = {"overall": {"status": "degraded"}}
        elif "operator_cockpit.py" in joined:
            payload = {"overall_status": "degraded"}
        else:
            raise AssertionError(f"unexpected command: {cmd}")
        return {"cmd": cmd, "rc": 0, "duration_ms": 5.0, "payload": payload, "stdout_tail": "", "stderr_tail": "", "timed_out": False}

    monkeypatch.setattr(src, "_run_json_command", _fake_run)

    payload = src.build_payload(project_root, apply=True, poll_seconds=0.0, wait_timeout_seconds=30.0)

    assert shard_runs["count"] == 3
    assert payload["catch_up_wave_controller"]["wave_limit"] == 3
    assert payload["summary"]["catch_up_waves_run"] == 3
    assert payload["drain_effectiveness"]["status"] in {"progress", "strong_progress"}
    assert payload["drain_effectiveness"]["total_pending_delta"] == 20000


def test_writer_cycle_coordinator_honors_caller_wave_cap() -> None:
    assert src._bounded_catch_up_wave_limit(5, 1) == 1
    assert src._bounded_catch_up_wave_limit(3, 2) == 2
    assert src._bounded_catch_up_wave_limit(3, 0) == 3


def test_writer_cycle_coordinator_keeps_sparse_jsonl_catch_up_waves_alive(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "backlog_relief_contract": {
                "active": True,
                "active_issue_ids": ["sparse_huge_jsonl_files"],
                "p_core_backlog_allocation_contract": {
                    "catch_up_wave_controller": {"enabled": True, "max_waves": 5}
                },
            }
        },
    )

    assert src._should_run_next_catch_up_wave(
        project_root,
        {"merged_rows_this_cycle": 250, "merge_followup": {"followup_needed": False}},
        wave_index=1,
        wave_limit=5,
    )


def test_writer_cycle_coordinator_uses_storage_accelerator_wave_limit(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "backlog_relief_contract": {
                "active": True,
                "active_issue_ids": ["single_writer_merge_speed", "sparse_huge_jsonl_files"],
                "p_core_backlog_allocation_contract": {
                    "catch_up_wave_controller": {"enabled": True, "max_waves": 5}
                },
                "accelerator_contract": {
                    "enabled": True,
                    "catch_up_wave_controller": {"enabled": True, "max_waves": 6},
                },
            }
        },
    )

    assert src._storage_catch_up_wave_limit(project_root) == 6


def test_writer_cycle_coordinator_treats_bounded_shard_timeout_as_partial_progress(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    (project_root / "governance" / "health").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(src, "writer_state_snapshot", lambda *args, **kwargs: {"active": False, "running": False, "current_step": "complete"})
    monkeypatch.setattr(
        src.drain_src,
        "build_payload",
        lambda *args, **kwargs: {"overall_status": "blocked", "recommended_now": False, "blocked_reasons": ["market_hours_guard"], "top_actions": []},
    )
    monkeypatch.setattr(
        src.drainer_src,
        "build_payload",
        lambda *args, **kwargs: {
            "overall_status": "ready",
            "ready_drainer_count": 1,
            "blocked_reasons": [],
            "active_drainer": {"name": "core_decision_drainer", "status": "ready", "live_window_safe": True},
        },
    )
    monkeypatch.setattr(src.maintenance_src, "_priority_retention_focus", lambda *args, **kwargs: {"enabled": False, "focus_shards": [], "top_actions": []})

    def _fake_run(cmd: list[str], *, cwd: Path, payload_path: Path | None = None, timeout_sec: int) -> dict:
        joined = " ".join(cmd)
        if "backpressure_drainer_fleet.py" in joined:
            payload = {"ok": True, "overall_status": "handoff_requested", "service_request": {"active": True, "env_overrides": {"SQL_LINK_SERVICE_SHARDS": "trading"}}}
            rc = 0
        elif "sql_link_shard_manager.py" in joined:
            payload = {"ok": False, "rc": 1, "merged_rows_this_cycle": 40106}
            rc = 1
        elif "ingestion_storage_control.py" in joined:
            payload = {"overall_status": "needs_work"}
            rc = 0
        elif "runtime_gate_dashboard.py" in joined:
            payload = {"overall": {"status": "degraded"}}
            rc = 0
        elif "operator_cockpit.py" in joined:
            payload = {"overall_status": "degraded"}
            rc = 0
        else:
            raise AssertionError(f"unexpected command: {cmd}")
        return {"cmd": cmd, "rc": rc, "duration_ms": 5.0, "payload": payload, "stdout_tail": "", "stderr_tail": "", "timed_out": False}

    monkeypatch.setattr(src, "_run_json_command", _fake_run)

    payload = src.build_payload(project_root, apply=True, poll_seconds=0.0, wait_timeout_seconds=30.0)

    assert payload["overall_status"] == "applied_with_followups"
    assert payload["ok"] is True
    assert payload["steps"]["sql_link_shard_manager"]["status"] == "partial_progress"
    assert payload["summary"]["partial_progress"] is True
    assert payload["summary"]["drain_applied"] is True


def test_sql_link_manager_timeout_scales_with_service_request_shards() -> None:
    payload = {
        "service_request": {
            "env_overrides": {
                "SQL_LINK_SERVICE_SHARDS": "trading,aggressive_trading,crypto_trading,health_fast,support_watchdog",
                "SQL_LINK_SERVICE_SHARD_LINK_TIMEOUT_SECONDS": "420",
                "SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE": "60",
            }
        }
    }

    timeout_seconds = src._sql_link_manager_timeout_seconds(
        command_timeout_seconds=420,
        wait_timeout_seconds=120.0,
        drainer_payload=payload,
    )

    assert timeout_seconds == 2280


def test_sql_link_manager_timeout_honors_micro_drain_cap() -> None:
    payload = {
        "service_request": {
            "env_overrides": {
                "SQL_LINK_SERVICE_SHARDS": "trading,aggressive_trading,crypto_trading,health_fast,support_watchdog",
                "SQL_LINK_SERVICE_SHARD_LINK_TIMEOUT_SECONDS": "420",
                "SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE": "60",
            }
        }
    }

    timeout_seconds = src._sql_link_manager_timeout_seconds(
        command_timeout_seconds=420,
        wait_timeout_seconds=120.0,
        drainer_payload=payload,
        timeout_cap_seconds=600,
    )

    assert timeout_seconds == 600


def test_writer_cycle_coordinator_already_running_payload_includes_writer_snapshot(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)
    previous_path = health / "writer_cycle_coordinator_latest.json"
    _write_json(
        previous_path,
        {
            "timestamp_utc": "2026-05-20T01:00:00+00:00",
            "overall_status": "waiting_for_writer",
            "summary": {"active_drainer": "core_decision_drainer"},
        },
    )

    monkeypatch.setattr(
        src,
        "writer_state_snapshot",
        lambda *args, **kwargs: {
            "active": True,
            "running": True,
            "current_step": "merge_primary",
            "merged_rows_this_cycle": 1150,
            "completed_merge_count": 8,
        },
    )
    monkeypatch.setattr(
        src.writer_intelligence_src,
        "build_payload",
        lambda *args, **kwargs: {
            "decision_packet": {
                "action": "wait_for_active_writer_progress",
                "expanded_writer_lane_count": 25,
            }
        },
    )

    payload = src._already_running_payload(project_root, previous_path=previous_path)

    assert payload["overall_status"] == "already_running"
    assert payload["busy"] is True
    assert payload["writer_state_before"]["current_step"] == "merge_primary"
    assert payload["summary"]["writer_active_initial"] is True
    assert payload["summary"]["writer_process_action"] == "wait_for_active_writer_progress"
    assert payload["summary"]["expanded_writer_lane_count"] == 25
    assert payload["previous_coordinator"]["summary"]["active_drainer"] == "core_decision_drainer"


def test_writer_cycle_coordinator_handoff_only_releases_completed_lock_without_drain(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    (project_root / "governance" / "health").mkdir(parents=True, exist_ok=True)
    writer_states = [
        {
            "active": True,
            "active_source": "completed_lock_handoff_needed",
            "running": False,
            "status": "ok",
            "current_step": "complete",
            "writer_lock_held": True,
            "child_writer_active": False,
            "completed_shard_count": 4,
            "planned_shard_count": 4,
        },
        {
            "active": False,
            "running": False,
            "status": "ok",
            "current_step": "complete",
            "writer_lock_held": False,
            "child_writer_active": False,
            "completed_shard_count": 4,
            "planned_shard_count": 4,
        },
    ]

    def _fake_writer_state(*args, **kwargs) -> dict:
        if writer_states:
            return writer_states.pop(0)
        return {
            "active": False,
            "running": False,
            "status": "ok",
            "current_step": "complete",
            "writer_lock_held": False,
        }

    def _fake_release(project_root_arg: Path, state: dict, *, grace_seconds: float = 3.0) -> dict:
        assert project_root_arg == project_root
        assert state["current_step"] == "complete"
        assert grace_seconds == 0.25
        return {
            "attempted": True,
            "needed": True,
            "pid": 1234,
            "owner": "pid=1234",
            "terminated": True,
            "lock_released": True,
            "reason": "completed_writer_handoff_released",
        }

    monkeypatch.setattr(src, "writer_state_snapshot", _fake_writer_state)
    monkeypatch.setattr(src, "_release_completed_writer_lock", _fake_release)
    monkeypatch.setattr(
        src,
        "_run_json_command",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("handoff-only must not run drain commands")),
    )

    payload = src.build_handoff_payload(project_root, apply=True, grace_seconds=0.25)

    assert payload["overall_status"] == "handoff_released"
    assert payload["handoff_only"] is True
    assert payload["summary"]["completed_writer_lock_handoff_initial_needed"] is True
    assert payload["summary"]["completed_writer_lock_handoff_needed"] is False
    assert payload["summary"]["completed_writer_lock_handoff_released"] is True
    assert payload["summary"]["writer_active_after_wait"] is False
    assert payload["steps"] == {}
    assert payload["refresh_steps"] == {}


def test_writer_cycle_coordinator_handoff_only_does_not_release_active_child_writer(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    (project_root / "governance" / "health").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        src,
        "writer_state_snapshot",
        lambda *args, **kwargs: {
            "active": True,
            "running": True,
            "status": "ok",
            "current_step": "complete",
            "writer_lock_held": True,
            "child_writer_active": True,
            "completed_shard_count": 4,
            "planned_shard_count": 4,
        },
    )
    monkeypatch.setattr(
        src,
        "_release_completed_writer_lock",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("active child writer must not be terminated")),
    )

    payload = src.build_handoff_payload(project_root, apply=True)

    assert payload["overall_status"] == "writer_active"
    assert payload["completed_writer_lock_handoff"]["needed"] is False
    assert payload["summary"]["completed_writer_lock_handoff_attempted"] is False
