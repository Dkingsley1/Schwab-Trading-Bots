import json
from datetime import datetime, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.ops.restart_sanity_bundle as bundle


def test_restart_sanity_bundle_watcher_summary_marks_recent_file_fresh(tmp_path) -> None:
    status_path = tmp_path / "macro_auto_watch_status.json"
    status_path.write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "stream_state": "live",
                "resolved_video_url": "https://example.com/live",
                "youtube_channel_url": "https://example.com/channel",
                "media_ingest_triggered": True,
            }
        ),
        encoding="utf-8",
    )

    payload = bundle._watcher_summary(status_path, max_age_hours=6.0)

    assert payload["exists"] is True
    assert payload["fresh"] is True
    assert payload["stream_state"] == "live"
    assert payload["media_ingest_triggered"] is True


def test_sql_sync_guard_skips_when_progress_artifact_is_active(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(bundle, "HEALTH_ROOT", tmp_path)
    (tmp_path / "sql_link_service_progress_latest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "ok": True,
                "running": True,
                "current_step": "shard_linking",
                "completed_shard_count": 9,
                "planned_shard_count": 17,
            }
        ),
        encoding="utf-8",
    )

    def fail_runner(cmd, timeout_seconds):  # pragma: no cover - only called on regression
        raise AssertionError("fresh active SQL writer artifact should avoid launching sql-sync")

    payload = bundle._run_sql_sync_with_guard(
        timeout_seconds=1,
        max_age_minutes=30,
        runner=fail_runner,
    )

    assert payload["ok"] is True
    assert payload["skipped"] is True
    assert payload["payload"]["status"] == "active_progressing"


def test_sql_sync_guard_accepts_timeout_when_artifact_turns_healthy(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(bundle, "HEALTH_ROOT", tmp_path)

    def timeout_runner(cmd, timeout_seconds):
        (tmp_path / "sql_link_service_progress_latest.json").write_text(
            json.dumps(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "ok": True,
                    "running": True,
                    "current_step": "shard_linking",
                    "completed_shard_count": 2,
                    "planned_shard_count": 17,
                }
            ),
            encoding="utf-8",
        )
        return {
            "ok": False,
            "rc": 124,
            "timed_out": True,
            "command": cmd,
            "payload": {},
            "stdout_tail": "",
            "stderr_tail": "",
        }

    payload = bundle._run_sql_sync_with_guard(
        timeout_seconds=1,
        max_age_minutes=30,
        runner=timeout_runner,
    )

    assert payload["ok"] is True
    assert payload["timed_out"] is True
    assert payload["sanity_observed_ok"] is True
    assert payload["payload"]["sanity_source"] == "artifact_snapshot_after_probe"
