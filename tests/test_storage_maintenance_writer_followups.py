import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import storage_backpressure_autopilot as autopilot_src
from scripts.ops import storage_maintenance_lane as maintenance_src
from scripts.ops import writer_cycle_coordinator as writer_src


SUPPORT_FREEZE = "support_maintenance_frozen_for_mac_fluidity"


def _result(cmd: list[str], payload: dict, rc: int = 0) -> dict:
    return {
        "cmd": list(cmd),
        "rc": rc,
        "duration_ms": 7.0,
        "payload": payload,
        "stdout_tail": "",
        "stderr_tail": "",
        "timed_out": False,
    }


def test_storage_maintenance_support_freeze_is_guarded_hold(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True)
    monkeypatch.setattr(maintenance_src, "_storage_roots", lambda _: (project_root, project_root))

    def _fake_run(cmd: list[str], **_: object) -> dict:
        joined = " ".join(cmd)
        if "ingestion_storage_governor.py" in joined:
            return _result(cmd, {"ok": True, "profile": "steady_state"})
        if "maintenance_strategy_reloader.py" in joined:
            return _result(cmd, {"ok": True, "changed": False})
        if "resource_guard.py" in joined:
            return _result(
                cmd,
                {
                    "ok": True,
                    "skipped_reason": SUPPORT_FREEZE,
                    "support_maintenance_frozen": True,
                    "resource_guard_reasons": [SUPPORT_FREEZE],
                },
                rc=2,
            )
        if "storage_failback_sync.py" in joined:
            return _result(cmd, {"ok": True, "mode": "external", "active_root": str(project_root)})
        raise AssertionError(f"support freeze should skip heavy maintenance command: {cmd}")

    monkeypatch.setattr(maintenance_src, "_run_json_command", _fake_run)

    payload = maintenance_src.build_storage_maintenance_payload(
        project_root,
        resource_profile="optional",
        force=False,
        vacuum=False,
    )

    assert payload["ok"] is True
    assert payload["overall_status"] == "guarded_hold"
    assert payload["reason"] == SUPPORT_FREEZE
    assert payload["heavy_steps_skipped"] is True
    assert payload["steps"]["resource_guard"]["status"] == "busy"
    assert payload["steps"]["sql_link_shard_manager"]["reason"] == SUPPORT_FREEZE


def test_writer_coordinator_treats_storage_maintenance_guarded_hold_as_followup(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    (project_root / "governance" / "health").mkdir(parents=True)
    idle_writer = {
        "active": False,
        "current_step": "complete",
        "effective_current_step": "complete",
        "writer_lock_held": False,
        "completed_shard_count": 0,
        "completed_merge_count": 0,
        "merged_rows_this_cycle": 0,
    }

    monkeypatch.setattr(writer_src, "writer_state_snapshot", lambda *_args, **_kwargs: dict(idle_writer))
    monkeypatch.setattr(
        writer_src.drain_src,
        "build_payload",
        lambda *_args, **_kwargs: {"overall_status": "blocked", "recommended_now": False, "blocked_reasons": ["market_hours_guard"]},
    )
    monkeypatch.setattr(
        writer_src.drainer_src,
        "build_payload",
        lambda *_args, **_kwargs: {
            "overall_status": "ready",
            "blocked_reasons": [],
            "ready_drainer_count": 1,
            "active_drainer": {"name": "core_decision_drainer", "status": "ready", "live_window_safe": True},
        },
    )
    monkeypatch.setattr(
        writer_src.maintenance_src,
        "_priority_retention_focus",
        lambda *_args, **_kwargs: {"enabled": True, "top_actions": ["pause noncritical maintenance until Mac fluidity clears"]},
    )
    monkeypatch.setattr(writer_src, "_refresh_surface_artifacts", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        writer_src.writer_intelligence_src,
        "build_payload",
        lambda *_args, **_kwargs: {
            "overall_status": "ready",
            "decision_packet": {"action": "run_focused_writer_cycle", "expanded_writer_lane_count": 1},
            "writer_health": {"state": "idle"},
        },
    )

    def _fake_run(cmd: list[str], **_: object) -> dict:
        joined = " ".join(cmd)
        if "backpressure_drainer_fleet.py" in joined:
            return _result(
                cmd,
                {
                    "overall_status": "handoff_requested",
                    "service_request": {
                        "active": True,
                        "env_overrides": {
                            "SQL_LINK_SERVICE_SHARDS": "health_fast",
                            "WRITER_CYCLE_MAX_CATCH_UP_WAVES": "1",
                        },
                    },
                },
            )
        if "sql_link_shard_manager.py" in joined:
            return _result(cmd, {"ok": True, "merged_rows_this_cycle": 25, "merge_followup": {"followup_needed": False}})
        if "storage_maintenance_lane.py" in joined:
            return _result(
                cmd,
                {
                    "ok": True,
                    "overall_status": "guarded_hold",
                    "reason": SUPPORT_FREEZE,
                    "resource_guard": {
                        "skipped_reason": SUPPORT_FREEZE,
                        "support_maintenance_frozen": True,
                        "resource_guard_reasons": [SUPPORT_FREEZE],
                    },
                },
            )
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(writer_src, "_run_json_command", _fake_run)

    payload = writer_src.build_payload(
        project_root,
        apply=True,
        wait_timeout_seconds=0.1,
        command_timeout_seconds=10,
        sql_manager_timeout_cap_seconds=10,
    )

    assert payload["ok"] is True
    assert payload["overall_status"] == "applied_with_followups"
    assert payload["steps"]["storage_maintenance_lane"]["status"] == "busy"
    assert payload["summary"]["maintenance_applied"] is False
    assert payload["summary"]["drain_applied"] is True


def test_storage_autopilot_manages_timeout_when_writer_is_progressing() -> None:
    payload = {
        "bot": "writer_cycle_coordinator",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": "waiting_for_writer",
        "summary": {
            "writer_active_after_wait": True,
            "wait_timed_out": False,
            "post_wait_stale_writer_detected": False,
        },
        "writer_state_after_wait": {"active": True, "progress_orphaned": False, "progress_age_minutes": 0.5},
        "writer_process_intelligence": {"decision_packet": {"action": "wait_for_active_writer_progress"}},
    }

    record = autopilot_src._attempt_record({"payload": payload, "rc": 124, "timed_out": True})

    assert record["status"] == "followup"
    assert record["timeout_managed"] is True


def test_storage_autopilot_keeps_stale_writer_timeout_failed() -> None:
    payload = {
        "bot": "writer_cycle_coordinator",
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "overall_status": "waiting_for_writer",
        "summary": {"writer_active_after_wait": True},
    }

    record = autopilot_src._attempt_record({"payload": payload, "rc": 124, "timed_out": True})

    assert record["status"] == "timed_out"
    assert record["timeout_managed"] is False
