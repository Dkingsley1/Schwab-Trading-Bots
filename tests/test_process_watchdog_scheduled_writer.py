import json
from datetime import datetime, timedelta, timezone

import pytest

from scripts.ops import process_watchdog as src


@pytest.fixture
def scheduled_writer(tmp_path, monkeypatch):
    monkeypatch.setattr(src, "HEALTH_DIR", tmp_path)
    now = datetime.now(timezone.utc)
    progress = {
        "timestamp_utc": (now - timedelta(seconds=20)).isoformat(),
        "ok": True,
        "status": "ok",
        "running": False,
        "current_step": "complete",
        "planned_shard_count": 5,
        "completed_shard_count": 5,
        "pending_shard_count": 0,
        "timed_out_shard_count": 0,
    }
    lifecycle = {
        "source": "scheduled_lifecycle_runner",
        "job_id": "sql_link_writer",
        "scheduled": True,
        "completed": True,
        "failed": False,
        "timed_out": False,
        "rc": 0,
        "terminal_status": "completed",
        "schedule_interval_seconds": 60,
        "started_utc": (now - timedelta(seconds=30)).isoformat(),
        "completed_utc": (now - timedelta(seconds=10)).isoformat(),
        "command": [
            "/bin/zsh",
            str(src.PROJECT_ROOT / "scripts/ops/run_sql_link_writer_launchd.sh"),
        ],
    }

    def assess():
        (tmp_path / "sql_link_service_latest.json").write_text(
            json.dumps({"job_lifecycle": lifecycle})
        )
        (tmp_path / "sql_link_service_progress_latest.json").write_text(
            json.dumps(progress)
        )
        (tmp_path / "ingestion_backpressure_latest.json").write_text(
            json.dumps(
                {
                    "pending_lines": 4000,
                    "pending_lines_total": 20000,
                    "oldest_pending_age_seconds": 15,
                }
            )
        )
        return src._sql_link_writer_idle_health()

    return now, progress, lifecycle, assess


def test_scheduled_completion_gets_grace_without_claiming_queue_clear(scheduled_writer):
    _, _, _, assess = scheduled_writer
    result = assess()
    assert result["ok"] is True
    assert result["reason"] == "sql_writer_between_scheduled_cycles"
    assert result["queue_idle_clear"] is False
    assert result["scheduled_wait"]["grace_seconds"] == 150


@pytest.mark.parametrize(
    "case",
    [
        "stale",
        "future",
        "naive",
        "running",
        "error",
        "partial",
        "timeout",
        "missing_timestamp",
    ],
)
def test_scheduler_cannot_mask_bad_or_stale_progress(scheduled_writer, case):
    now, progress, _, assess = scheduled_writer
    changes = {
        "stale": {"timestamp_utc": (now - timedelta(seconds=151)).isoformat()},
        "future": {"timestamp_utc": (now + timedelta(seconds=1)).isoformat()},
        "naive": {"timestamp_utc": now.replace(tzinfo=None).isoformat()},
        "running": {"running": True},
        "error": {"ok": False},
        "partial": {"completed_shard_count": 4, "pending_shard_count": 1},
        "timeout": {"timed_out_shard_count": 1},
        "missing_timestamp": {"timestamp_utc": None},
    }
    progress.update(changes[case])
    assert assess()["ok"] is False


@pytest.mark.parametrize(
    "change",
    [
        {"scheduled": False},
        {"failed": True},
        {"timed_out": True},
        {"rc": 2},
        {"rc": False},
        {"job_id": "unrelated"},
        {"source": "unknown"},
        {"command": ["wrong"]},
        {"schedule_interval_seconds": True},
        {"schedule_interval_seconds": 900},
        {"schedule_interval_seconds": "60"},
        {"terminal_status": "failed"},
    ],
)
def test_scheduler_grace_requires_valid_native_receipt(scheduled_writer, change):
    _, _, lifecycle, assess = scheduled_writer
    lifecycle.update(change)
    assert assess()["ok"] is False


def test_repeated_deferrals_cannot_extend_progress_grace(scheduled_writer):
    now, progress, lifecycle, assess = scheduled_writer
    lifecycle.update(
        terminal_status="deferred", deferred=True, completed_utc=now.isoformat()
    )
    assert assess()["ok"] is True
    progress["timestamp_utc"] = (now - timedelta(seconds=151)).isoformat()
    assert assess()["ok"] is False


@pytest.mark.parametrize("age", [-1, 151])
def test_future_or_stale_lifecycle_receipt_cannot_extend_grace(scheduled_writer, age):
    now, _, lifecycle, assess = scheduled_writer
    lifecycle["completed_utc"] = (now - timedelta(seconds=age)).isoformat()
    assert assess()["ok"] is False


def test_storm_resolution_names_scheduled_wait_without_erasing_history():
    now = datetime.now(timezone.utc).timestamp()
    events = [
        {"name": "sql_link_writer", "event": "restart", "ts_epoch": now - i}
        for i in range(5)
    ]
    active, recent = src._resolved_restart_storms(
        events=events,
        status_rows=[
            {
                "name": "sql_link_writer",
                "running": 0,
                "writer_idle_ok": True,
                "heartbeat_ok": True,
                "process_live_reason": "sql_writer_between_scheduled_cycles",
            }
        ],
        restart_window_seconds=3600,
        restart_storm_threshold=5,
        settle_seconds=900,
        now_epoch=now,
    )
    assert active == []
    assert len(events) == 5
    assert recent[0]["count"] == 5
    assert recent[0]["resolution_reason"] == "sql_writer_between_scheduled_cycles"
