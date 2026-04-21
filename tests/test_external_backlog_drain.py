import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import external_backlog_drain as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_external_backlog_drain_builds_offhours_plan(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 12000,
            "pending_lines_total": 510000,
            "pending_lines_deferred": 410000,
            "pending_lines_cold": 180000,
            "top_deferred_pending_files": [
                {
                    "source_rel": "governance/events/api_calls_20260406.jsonl",
                    "pending_lines": 210000,
                    "oldest_pending_age_seconds": 14400.0,
                }
            ],
            "top_cold_pending_files": [
                {
                    "source_rel": "governance/shadow_pnl_attribution_20260406.jsonl",
                    "pending_lines": 180000,
                    "oldest_pending_age_seconds": 25200.0,
                }
            ],
        },
    )
    _write_json(health / "ingestion_priority_queue_latest.json", {"queue_depth": 14})
    _write_json(health / "ingestion_storage_control_latest.json", {"overall_status": "blocked"})
    _write_json(health / "storage_mount_guard_latest.json", {"external_available": True, "storage_mode": "external"})
    _write_json(health / "storage_failback_sync_latest.json", {"mode": "external", "split_brain_conflicts": 0})
    _write_json(health / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 0}})
    _write_json(health / "sql_link_service_latest.json", {"primary_db": str(project_root / "data" / "jsonl_link.sqlite3")})
    _write_json(health / "sql_link_service_progress_latest.json", {})
    _write_json(health / "health_gates_latest.json", {"hard_gate_triggered": False, "storage_pressure": {"retention_debt_gb": 2.0}})

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 4, 6, 21, 0, tzinfo=timezone.utc),
    )

    assert payload["recommended_now"] is True
    assert payload["drain_profile"] == "offhours_external_backlog_drain"
    assert payload["aged_candidate_files"] == 2
    assert payload["drain_overrides"]["deferred_files_budget"] == 6
    assert payload["drain_overrides"]["sql_interval_seconds"] == 12
    assert payload["drain_overrides"]["hot_batch_size"] == 240000
    assert any("compact or archive" in item for item in payload["top_actions"])


def test_external_backlog_drain_calls_out_stale_stage_archive_candidates(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 200,
            "pending_lines_total": 120200,
            "pending_lines_deferred": 120000,
            "pending_lines_cold": 119900,
            "top_deferred_pending_files": [
                {
                    "source_rel": "data/stale_stage/decision_explanations/project/decision_explanations/shadow_crypto_futures_crypto/decision_explanations_20260413.jsonl",
                    "pending_lines": 119900,
                    "oldest_pending_age_seconds": 572800.0,
                }
            ],
            "top_cold_pending_files": [
                {
                    "source_rel": "data/stale_stage/decision_explanations/project/decision_explanations/shadow_crypto_futures_crypto/decision_explanations_20260413.jsonl",
                    "pending_lines": 119900,
                    "oldest_pending_age_seconds": 572800.0,
                }
            ],
        },
    )
    _write_json(health / "ingestion_priority_queue_latest.json", {"queue_depth": 3})
    _write_json(health / "ingestion_storage_control_latest.json", {"overall_status": "ready"})
    _write_json(health / "storage_mount_guard_latest.json", {"external_available": True, "storage_mode": "external"})
    _write_json(health / "storage_failback_sync_latest.json", {"mode": "external", "split_brain_conflicts": 0})
    _write_json(health / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 0}})
    _write_json(health / "sql_link_service_latest.json", {"primary_db": str(project_root / "data" / "jsonl_link.sqlite3")})
    _write_json(health / "sql_link_service_progress_latest.json", {})
    _write_json(health / "health_gates_latest.json", {"hard_gate_triggered": False, "storage_pressure": {"retention_debt_gb": 0.0}})

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 4, 6, 21, 0, tzinfo=timezone.utc),
    )

    assert payload["stale_stage_candidate_files"] == 1
    assert payload["stale_stage_candidate_pending_lines"] == 119900
    assert any(str(row.get("candidate_action")) == "reap_or_archive_stale_stage" for row in payload["hotspots"])
    assert any("staged stale artifacts" in item for item in payload["top_actions"])


def test_external_backlog_drain_calls_out_watchdog_support_candidates(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 400,
            "pending_lines_total": 182400,
            "pending_lines_deferred": 182000,
            "pending_lines_support_telemetry": 180000,
            "pending_lines_cold": 0,
            "top_support_telemetry_pending_files": [
                {
                    "source_rel": "governance/watchdog/failover_events.jsonl",
                    "pending_lines": 180000,
                    "oldest_pending_age_seconds": 90.0,
                }
            ],
        },
    )
    _write_json(health / "ingestion_priority_queue_latest.json", {"queue_depth": 2})
    _write_json(health / "ingestion_storage_control_latest.json", {"overall_status": "ready"})
    _write_json(health / "storage_mount_guard_latest.json", {"external_available": True, "storage_mode": "external"})
    _write_json(health / "storage_failback_sync_latest.json", {"mode": "external", "split_brain_conflicts": 0})
    _write_json(health / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 0}})
    _write_json(health / "sql_link_service_latest.json", {"primary_db": str(project_root / "data" / "jsonl_link.sqlite3")})
    _write_json(health / "sql_link_service_progress_latest.json", {})
    _write_json(health / "health_gates_latest.json", {"hard_gate_triggered": False, "storage_pressure": {"retention_debt_gb": 0.0}})

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 4, 6, 21, 0, tzinfo=timezone.utc),
    )

    assert payload["support_watchdog_candidate_files"] == 1
    assert payload["support_watchdog_candidate_pending_lines"] == 180000
    assert any(str(row.get("candidate_action")) == "drain_support_watchdog" for row in payload["hotspots"])
    assert any("support shard" in item for item in payload["top_actions"])


def test_external_backlog_drain_apply_executes_and_refreshes_backlog(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 10000,
            "pending_lines_total": 300000,
            "pending_lines_deferred": 220000,
            "pending_lines_cold": 80000,
        },
    )
    _write_json(health / "ingestion_priority_queue_latest.json", {"queue_depth": 12})
    _write_json(health / "storage_mount_guard_latest.json", {"external_available": True, "storage_mode": "external"})
    _write_json(health / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 0}})

    monkeypatch.setattr(
        src.governor_src,
        "build_payload",
        lambda *args, **kwargs: {
            "profile": "critical_backpressure",
            "env_overrides": {
                "SQL_LINK_SERVICE_PRIMARY_DB": str(project_root / "data" / "jsonl_link.sqlite3"),
                "BOT_CHANNEL_QUEUE_DB": str(project_root / "data" / "bot_channel_queue.sqlite3"),
                "SQL_LINK_SERVICE_QUEUE_DB": str(project_root / "data" / "bot_channel_queue.sqlite3"),
                "INGEST_MAX_DEFERRED_FILES": "0",
                "JSONL_SQL_MAX_COLD_LANE_FILES": "0",
            },
        },
    )

    seen: list[str] = []

    def _fake_run(
        cmd: list[str],
        *,
        cwd: Path,
        payload_path: Path | None = None,
        env_overrides: dict[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> dict:
        joined = " ".join(cmd)
        seen.append(joined)
        if "ingestion_backpressure_guard.py" in joined and "before" not in joined:
            if len([row for row in seen if "ingestion_backpressure_guard.py" in row]) == 1:
                payload = {
                    "pending_lines": 10000,
                    "pending_lines_total": 300000,
                    "pending_lines_deferred": 220000,
                    "pending_lines_cold": 80000,
                }
            else:
                payload = {
                    "pending_lines": 4000,
                    "pending_lines_total": 180000,
                    "pending_lines_deferred": 120000,
                    "pending_lines_cold": 56000,
                }
        elif "ingestion_priority_queue.py" in joined:
            if len([row for row in seen if "ingestion_priority_queue.py" in row]) == 1:
                payload = {"queue_depth": 12}
            else:
                payload = {"queue_depth": 7}
        elif "resource_guard.py" in joined:
            payload = {"ok": True}
        elif "sql_link_shard_manager.py" in joined:
            assert env_overrides is not None
            assert env_overrides["INGEST_MAX_DEFERRED_FILES"] == "6"
            assert env_overrides["JSONL_SQL_MAX_COLD_LANE_FILES"] == "2"
            assert env_overrides["SQL_LINK_SERVICE_WAL_CHECKPOINT_THRESHOLD_GB"] == "0.25"
            assert env_overrides["SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE"] == "25"
            assert env_overrides["SQL_LINK_SERVICE_SHARD_RUNTIME_STATE_CHECKPOINT_LINES"] == "1500"
            payload = {"ok": True, "reason": "ok"}
        elif "sqlite_performance_maintenance.py" in joined:
            assert timeout_seconds == 20.0
            payload = {"ok": True}
        elif "stale_artifact_sweeper_bot.py" in joined:
            payload = {"ok": True, "summary": {"candidate_files": 3, "staged_files": 2}}
        elif "stale_artifact_reaper_bot.py" in joined:
            payload = {"ok": True, "summary": {"candidate_files": 1, "deleted_files": 1}}
        elif "data_retention_policy.py" in joined:
            payload = {"ok": True, "deleted": 9}
        else:
            raise AssertionError(f"unexpected command: {cmd}")
        if payload_path is not None:
            _write_json(payload_path, payload)
        return {"cmd": cmd, "rc": 0, "duration_ms": 5.0, "payload": payload, "stdout_tail": "", "stderr_tail": ""}

    monkeypatch.setattr(src, "_run_json_command", _fake_run)

    payload = src.build_payload(
        project_root,
        apply=True,
        now_utc=datetime(2026, 4, 6, 21, 0, tzinfo=timezone.utc),
    )

    assert payload["apply_executed"] is True
    assert payload["drain_delta"]["deferred_pending_lines"] == 100000
    assert payload["drain_delta"]["cold_pending_lines"] == 24000
    assert payload["queue_depth_after"] == 7
    assert payload["steps"]["sql_link_shard_manager"]["status"] == "ok"


def test_external_backlog_drain_follow_through_retries_busy_writer(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(health / "ingestion_backpressure_latest.json", {"pending_lines": 50, "pending_lines_total": 100, "pending_lines_deferred": 40, "pending_lines_cold": 10})
    _write_json(health / "ingestion_priority_queue_latest.json", {"queue_depth": 2})
    _write_json(health / "storage_mount_guard_latest.json", {"external_available": True, "storage_mode": "external"})
    _write_json(health / "storage_split_brain_reconciler_latest.json", {"summary": {"unresolved_conflicts": 0}})
    lock_path = project_root / "governance" / "locks" / "jsonl_sql_writer.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("pid=4321 started=2026-04-06T20:00:00+00:00 cmd=sql_link_shard_manager", encoding="utf-8")

    monkeypatch.setattr(src, "SQL_WRITER_LOCK_PATH", lock_path)
    monkeypatch.setattr(
        src.governor_src,
        "build_payload",
        lambda *args, **kwargs: {"profile": "critical_backpressure", "env_overrides": {}},
    )
    monkeypatch.setattr(src.time_mod, "sleep", lambda seconds: None)

    def _fake_run(
        cmd: list[str],
        *,
        cwd: Path,
        payload_path: Path | None = None,
        env_overrides: dict[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> dict:
        joined = " ".join(cmd)
        if "ingestion_backpressure_guard.py" in joined:
            payload = {"pending_lines": 50, "pending_lines_total": 100, "pending_lines_deferred": 40, "pending_lines_cold": 10}
        elif "ingestion_priority_queue.py" in joined:
            payload = {"queue_depth": 2}
        elif "resource_guard.py" in joined:
            payload = {"ok": True}
        elif "sql_link_shard_manager.py" in joined:
            payload = {"ok": False, "reason": "writer_lock_busy", "busy": True}
        elif "sqlite_performance_maintenance.py" in joined:
            payload = {"ok": True}
        elif "stale_artifact_sweeper_bot.py" in joined:
            payload = {"ok": True, "summary": {"candidate_files": 0, "staged_files": 0}}
        elif "stale_artifact_reaper_bot.py" in joined:
            payload = {"ok": True, "summary": {"candidate_files": 0, "deleted_files": 0}}
        elif "data_retention_policy.py" in joined:
            payload = {"ok": True, "deleted": 0}
        else:
            raise AssertionError(f"unexpected command: {cmd}")
        if payload_path is not None:
            _write_json(payload_path, payload)
        return {"cmd": cmd, "rc": 0, "duration_ms": 5.0, "payload": payload, "stdout_tail": "", "stderr_tail": "", "timed_out": False}

    monkeypatch.setattr(src, "_run_json_command", _fake_run)

    payload = src.build_payload(
        project_root,
        apply=True,
        follow_through=True,
        poll_seconds=0.1,
        wait_timeout_seconds=1.0,
        now_utc=datetime(2026, 4, 6, 21, 0, tzinfo=timezone.utc),
    )

    assert payload["follow_through"]["requested"] is True
    assert payload["follow_through"]["completed"] is True
    assert payload["follow_through"]["attempts"] == 1
    assert payload["follow_through"]["status"] == "handoff_requested"
    assert payload["follow_through"]["progress_state"] == "requested_live_writer"
    assert payload["steps"]["sql_link_shard_manager_initial"]["status"] == "busy"
    assert payload["steps"]["sql_link_service_request"]["status"] == "ok"
    assert payload["service_request"]["request_kind"] == "external_backlog_drain"
    assert (project_root / "governance" / "health" / "sql_link_service_request_latest.json").exists()


def test_follow_through_retry_marks_progressing_timeout(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)
    lock_path = project_root / "governance" / "locks" / "jsonl_sql_writer.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("pid=4321 started=2026-04-06T20:00:00+00:00 cmd=sql_link_shard_manager", encoding="utf-8")

    base_now = datetime(2026, 4, 6, 21, 0, tzinfo=timezone.utc)

    class _FakeDatetime:
        current = base_now

        @classmethod
        def now(cls, tz=None):
            value = cls.current
            return value if tz is None else value.astimezone(tz)

    attempts = {"count": 0}

    def _fake_run(
        cmd: list[str],
        *,
        cwd: Path,
        payload_path: Path | None = None,
        env_overrides: dict[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> dict:
        attempts["count"] += 1
        _FakeDatetime.current += timedelta(seconds=0.6)
        payload = {
            "ok": False,
            "reason": "writer_lock_busy",
            "busy": True,
            "current_step": "merge_primary",
            "completed_shard_count": attempts["count"],
            "completed_merge_count": 0,
            "merged_rows_this_cycle": attempts["count"] * 100,
        }
        if payload_path is not None:
            _write_json(payload_path, payload)
        return {"cmd": cmd, "rc": 0, "duration_ms": 5.0, "payload": payload, "stdout_tail": "", "stderr_tail": "", "timed_out": False}

    monkeypatch.setattr(src, "SQL_WRITER_LOCK_PATH", lock_path)
    monkeypatch.setattr(src, "datetime", _FakeDatetime)
    monkeypatch.setattr(src, "_run_json_command", _fake_run)
    monkeypatch.setattr(src.time_mod, "sleep", lambda seconds: None)

    result = src._follow_through_retry(
        project_root=project_root,
        health_root=health,
        drain_env={},
        poll_seconds=0.1,
        wait_timeout_seconds=1.0,
    )

    assert result["completed"] is False
    assert result["status"] == "timed_out"
    assert result["progress_state"] == "progressing"
    assert result["progress_observed"] is True
    assert result["progress_events"] >= 1
