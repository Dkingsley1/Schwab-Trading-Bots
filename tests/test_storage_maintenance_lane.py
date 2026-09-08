import sys
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import storage_maintenance_lane


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_crypto_attribution_retention_focus_uses_intraday_hot_window(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "governance" / "health" / "health_gates_latest.json",
        {
            "priority_shards": [
                {
                    "shard": "crypto_shadow_attribution",
                    "retention_debt_gb": 0.289,
                    "latency_limit_multiplier": 0.3,
                    "storage_breached": True,
                    "latency_breached": False,
                    "recommended_action": "force_retention",
                }
            ]
        },
    )

    focus = storage_maintenance_lane._priority_retention_focus(project_root, {})

    assert focus["severe_focus"] is True
    assert focus["env_overrides"][
        "SQL_LINK_SERVICE_SHARD_CRYPTO_SHADOW_ATTRIBUTION_HOT_RETENTION_HOT_HOURS"
    ] == "6"


def test_build_storage_maintenance_payload_runs_all_steps(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    (project_root / "governance" / "health").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(storage_maintenance_lane, "PY", Path("/usr/bin/python3"))
    monkeypatch.setattr(
        storage_maintenance_lane,
        "_usage_snapshot",
        lambda path: {"path": str(path), "exists": True, "free_gb": 120.0, "used_gb": 80.0, "total_gb": 200.0},
    )
    primary_db = project_root / "data" / "jsonl_link.sqlite3"
    wal_path = project_root / "data" / "jsonl_link.sqlite3-wal"
    primary_db.parent.mkdir(parents=True, exist_ok=True)
    primary_db.write_bytes(b"db")
    wal_path.write_bytes(b"wal")
    (project_root / "governance" / "health" / "sql_link_service_progress_latest.json").write_text(
        (
            '{"timestamp_utc":"2026-04-04T22:34:48+00:00",'
            '"status":"running","current_step":"merge_primary","primary_db":"%s"}'
        )
        % str(primary_db),
        encoding="utf-8",
    )
    (project_root / "governance" / "health" / "sql_link_service_latest.json").write_text("{}", encoding="utf-8")
    _write_json(
        project_root / "governance" / "health" / "health_gates_latest.json",
        {
            "priority_shards": [
                {
                    "shard": "crypto_explanations",
                    "retention_debt_gb": 12.71,
                    "latency_limit_multiplier": 1.17,
                    "storage_breached": True,
                    "latency_breached": True,
                    "recommended_action": "force_retention_and_throttle",
                },
                {
                    "shard": "explanations",
                    "retention_debt_gb": 51.055,
                    "latency_limit_multiplier": 1.727,
                    "storage_breached": True,
                    "latency_breached": True,
                    "recommended_action": "force_retention_and_throttle",
                },
            ]
        },
    )

    env_by_cmd: dict[str, dict[str, str]] = {}

    def _fake_run(cmd: list[str], *, cwd: Path, payload_path: Path | None = None, env_overrides: dict[str, str] | None = None) -> dict:
        joined = " ".join(cmd)
        env_by_cmd[joined] = dict(env_overrides or {})
        if "ingestion_storage_governor.py" in joined:
            payload = {
                "ok": True,
                "profile": "critical_backpressure",
                "sql_primary_db": {"route_drift": True},
                "env_overrides": {
                    "SQL_LINK_SERVICE_PRIMARY_DB": str(project_root / "data" / "jsonl_link.sqlite3"),
                    "BOT_CHANNEL_QUEUE_DB": str(project_root / "data" / "bot_channel_queue.sqlite3"),
                    "SQL_LINK_SERVICE_QUEUE_DB": str(project_root / "data" / "bot_channel_queue.sqlite3"),
                },
            }
        elif "maintenance_strategy_reloader.py" in joined:
            payload = {"changed": True, "deferred": False}
        elif "resource_guard.py" in joined:
            payload = {"ok": True, "memory_pressure_kind": "none"}
        elif "storage_failback_sync.py" in joined:
            payload = {
                "ok": True,
                "mode": "external",
                "active_root": "/Volumes/BOT_LOGS/schwab_trading_bot",
                "autosync": {"copied_files": 3},
                "low_space_autoprune": {"deleted_count": 4},
            }
        elif "sql_link_shard_manager.py" in joined:
            payload = {"ok": True, "reason": "ok"}
        elif "sqlite_performance_maintenance.py" in joined:
            payload = {"ok": True, "wal_size_gb_before": 9.2, "wal_size_gb_after": 1.1}
        elif "stale_artifact_sweeper_bot.py" in joined:
            payload = {"ok": True, "summary": {"candidate_files": 8, "staged_files": 6, "staged_bytes": 8192, "delete_errors": 0}}
        elif "stale_artifact_reaper_bot.py" in joined:
            payload = {"ok": True, "summary": {"candidate_files": 2, "deleted_files": 2, "deleted_bytes": 2048, "delete_errors": 0}}
        elif "data_retention_policy.py" in joined:
            payload = {"deleted": 14, "delete_errors": 0}
        elif "content_addressed_artifact_store.py" in joined:
            payload = {
                "ok": True,
                "skipped_blob_count": 1,
                "metadata_only_blob_count": 1,
                "unsafe_skipped_blob_count": 0,
                "gc": {"deleted_blob_count": 3, "deleted_bytes": 4096},
            }
        else:
            raise AssertionError(f"unexpected command: {cmd}")
        return {"cmd": cmd, "rc": 0, "duration_ms": 7.0, "payload": payload, "stdout_tail": "", "stderr_tail": ""}

    monkeypatch.setattr(storage_maintenance_lane, "_run_json_command", _fake_run)

    payload = storage_maintenance_lane.build_storage_maintenance_payload(
        project_root,
        resource_profile="optional",
        force=False,
        vacuum=False,
    )

    assert payload["ok"] is True
    assert payload["heavy_steps_skipped"] is False
    assert payload["summary"]["ingestion_storage_profile"] == "critical_backpressure"
    assert payload["summary"]["governor_route_drift"] is True
    assert payload["summary"]["priority_retention_focus_enabled"] is True
    assert payload["summary"]["priority_retention_focus_shards"] == ["explanations", "crypto_explanations"]
    assert payload["summary"]["priority_retention_handoff_ready"] is True
    assert payload["summary"]["priority_retention_hold_released"] is True
    assert payload["summary"]["maintenance_reloader_changed"] is True
    assert payload["summary"]["storage_mode"] == "external"
    assert payload["summary"]["autoprune_deleted_count"] == 4
    assert payload["summary"]["stale_stage_candidate_files"] == 8
    assert payload["summary"]["stale_stage_staged_files"] == 6
    assert payload["summary"]["stale_stage_reaped_files"] == 2
    assert payload["summary"]["retention_deleted"] == 14
    assert payload["summary"]["content_store_deleted_blobs"] == 3
    assert payload["summary"]["content_store_deleted_bytes"] == 4096
    assert payload["summary"]["content_store_skipped_blobs"] == 1
    assert payload["summary"]["content_store_metadata_only_blobs"] == 1
    assert payload["summary"]["content_store_unsafe_skipped_blobs"] == 0
    assert payload["summary"]["sql_sync_status"] == "running"
    assert payload["summary"]["sql_sync_step"] == "merge_primary"
    assert payload["summary"]["primary_db_size_gb_live"] >= 0.0
    assert payload["summary"]["primary_wal_size_gb_live"] >= 0.0
    assert payload["steps"]["stale_artifact_sweeper_bot"]["status"] == "ok"
    assert payload["steps"]["stale_artifact_reaper_bot"]["status"] == "ok"
    assert payload["steps"]["data_retention_policy"]["status"] == "ok"
    assert payload["steps"]["content_addressed_artifact_store"]["status"] == "ok"
    shard_env = next(env for cmd, env in env_by_cmd.items() if "sql_link_shard_manager.py" in cmd)
    assert shard_env["SQL_LINK_SERVICE_PRIMARY_DB"] == str(project_root / "data" / "jsonl_link.sqlite3")
    assert shard_env["BOT_CHANNEL_QUEUE_DB"] == str(project_root / "data" / "bot_channel_queue.sqlite3")
    assert shard_env["SQL_LINK_SERVICE_SHARDS"] == "health_fast,crypto_explanations,explanations,crypto_shadow_attribution,shadow_attribution"
    assert shard_env["SQL_LINK_SERVICE_SHARD_EXPLANATIONS_MAX_FILES"] == "8"
    assert shard_env["SQL_LINK_SERVICE_MAINTENANCE_HOLD_TOKEN"]


def test_storage_maintenance_treats_missing_sqlite_primary_as_skipped(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    (project_root / "governance" / "health").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(storage_maintenance_lane, "PY", Path("/usr/bin/python3"))
    monkeypatch.setattr(
        storage_maintenance_lane,
        "_usage_snapshot",
        lambda path: {"path": str(path), "exists": True, "free_gb": 120.0, "used_gb": 80.0, "total_gb": 200.0},
    )

    def _fake_run(cmd: list[str], *, cwd: Path, payload_path: Path | None = None, env_overrides: dict[str, str] | None = None) -> dict:
        joined = " ".join(cmd)
        if "ingestion_storage_governor.py" in joined:
            payload = {"ok": True, "profile": "critical_backpressure", "sql_primary_db": {"route_drift": False}, "env_overrides": {}}
            rc = 0
        elif "maintenance_strategy_reloader.py" in joined:
            payload = {"changed": False, "deferred": False}
            rc = 0
        elif "resource_guard.py" in joined:
            payload = {"ok": True, "memory_pressure_kind": "none"}
            rc = 0
        elif "storage_failback_sync.py" in joined:
            payload = {"ok": True, "mode": "local_fallback", "autosync": {"copied_files": 0}, "low_space_autoprune": {"deleted_count": 0}}
            rc = 0
        elif "sql_link_shard_manager.py" in joined:
            payload = {"ok": True, "reason": "ok"}
            rc = 0
        elif "sqlite_performance_maintenance.py" in joined:
            payload = {
                "ok": False,
                "error": f"db_missing:{project_root / 'data' / 'jsonl_link.sqlite3'}",
                "vacuum_ran": False,
            }
            rc = 2
        elif "stale_artifact_sweeper_bot.py" in joined:
            payload = {"ok": True, "summary": {"candidate_files": 0, "staged_files": 0, "staged_bytes": 0, "delete_errors": 0}}
            rc = 0
        elif "stale_artifact_reaper_bot.py" in joined:
            payload = {"ok": True, "summary": {"candidate_files": 0, "deleted_files": 0, "deleted_bytes": 0, "delete_errors": 0}}
            rc = 0
        elif "data_retention_policy.py" in joined:
            payload = {"deleted": 0, "delete_errors": 0}
            rc = 0
        elif "content_addressed_artifact_store.py" in joined:
            payload = {"ok": True, "skipped_blob_count": 0, "gc": {"deleted_blob_count": 0, "deleted_bytes": 0}}
            rc = 0
        else:
            raise AssertionError(f"unexpected command: {cmd}")
        return {"cmd": cmd, "rc": rc, "duration_ms": 5.0, "payload": payload, "stdout_tail": "", "stderr_tail": ""}

    monkeypatch.setattr(storage_maintenance_lane, "_run_json_command", _fake_run)

    payload = storage_maintenance_lane.build_storage_maintenance_payload(
        project_root,
        resource_profile="optional",
        force=False,
        vacuum=False,
    )

    assert payload["ok"] is True
    assert payload["reason"] == "ok"
    assert payload["steps"]["sqlite_maintenance"]["status"] == "skipped"


def test_build_storage_maintenance_payload_skips_heavy_steps_when_guard_blocks(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    (project_root / "governance" / "health").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(storage_maintenance_lane, "PY", Path("/usr/bin/python3"))
    monkeypatch.setattr(
        storage_maintenance_lane,
        "_usage_snapshot",
        lambda path: {"path": str(path), "exists": True, "free_gb": 120.0, "used_gb": 80.0, "total_gb": 200.0},
    )

    calls: list[list[str]] = []

    def _fake_run(cmd: list[str], *, cwd: Path, payload_path: Path | None = None, env_overrides: dict[str, str] | None = None) -> dict:
        calls.append(cmd)
        joined = " ".join(cmd)
        if "ingestion_storage_governor.py" in joined:
            payload = {"ok": True, "profile": "critical_backpressure", "sql_primary_db": {"route_drift": False}, "env_overrides": {}}
        elif "maintenance_strategy_reloader.py" in joined:
            payload = {"changed": False, "deferred": False}
        elif "resource_guard.py" in joined:
            payload = {"ok": False, "memory_pressure_kind": "swap_only"}
        elif "storage_failback_sync.py" in joined:
            payload = {"ok": True, "mode": "local_fallback", "autosync": {"copied_files": 0}, "low_space_autoprune": {"deleted_count": 0}}
        else:
            raise AssertionError(f"heavy step should have been skipped: {cmd}")
        return {"cmd": cmd, "rc": 0, "duration_ms": 4.0, "payload": payload, "stdout_tail": "", "stderr_tail": ""}

    monkeypatch.setattr(storage_maintenance_lane, "_run_json_command", _fake_run)

    payload = storage_maintenance_lane.build_storage_maintenance_payload(
        project_root,
        resource_profile="optional",
        force=False,
        vacuum=False,
    )

    assert payload["ok"] is True
    assert payload["heavy_steps_skipped"] is True
    assert payload["steps"]["sql_link_shard_manager"]["status"] == "skipped"
    assert payload["steps"]["content_addressed_artifact_store"]["status"] == "skipped"
    assert payload["steps"]["stale_artifact_sweeper_bot"]["status"] == "skipped"
    assert payload["steps"]["stale_artifact_reaper_bot"]["status"] == "skipped"
    assert payload["steps"]["ingestion_storage_governor"]["status"] == "ok"
    assert payload["steps"]["maintenance_strategy_reloader"]["status"] == "ok"
    assert len(calls) == 4


def test_build_storage_maintenance_payload_retries_shard_manager_when_priority_focus_is_severe(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    health.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(storage_maintenance_lane, "PY", Path("/usr/bin/python3"))
    monkeypatch.setattr(
        storage_maintenance_lane,
        "_usage_snapshot",
        lambda path: {"path": str(path), "exists": True, "free_gb": 120.0, "used_gb": 80.0, "total_gb": 200.0},
    )
    monkeypatch.setattr(storage_maintenance_lane, "DEFAULT_SQL_LOCK_PATH", project_root / "governance" / "locks" / "jsonl_sql_writer.lock")
    monkeypatch.setenv("SQL_LINK_SERVICE_MAINTENANCE_LOCK_WAIT_SECONDS", "0.05")
    monkeypatch.setenv("SQL_LINK_SERVICE_MAINTENANCE_LOCK_POLL_SECONDS", "0.01")

    primary_db = project_root / "data" / "jsonl_link.sqlite3"
    primary_db.parent.mkdir(parents=True, exist_ok=True)
    primary_db.write_bytes(b"db")
    (health / "sql_link_service_progress_latest.json").write_text(
        (
            '{"timestamp_utc":"2026-04-04T22:34:48+00:00",'
            '"status":"running","current_step":"merge_primary","primary_db":"%s"}'
        )
        % str(primary_db),
        encoding="utf-8",
    )
    (health / "sql_link_service_latest.json").write_text("{}", encoding="utf-8")
    _write_json(
        health / "health_gates_latest.json",
        {
            "priority_shards": [
                {
                    "shard": "explanations",
                    "retention_debt_gb": 66.6,
                    "latency_limit_multiplier": 9.7,
                    "storage_breached": True,
                    "latency_breached": True,
                    "recommended_action": "force_retention_and_throttle",
                }
            ]
        },
    )

    shard_calls = {"count": 0}

    def _fake_run(cmd: list[str], *, cwd: Path, payload_path: Path | None = None, env_overrides: dict[str, str] | None = None) -> dict:
        joined = " ".join(cmd)
        if "ingestion_storage_governor.py" in joined:
            payload = {"ok": True, "profile": "critical_backpressure", "sql_primary_db": {"route_drift": False}, "env_overrides": {}}
        elif "maintenance_strategy_reloader.py" in joined:
            payload = {"changed": False, "deferred": False}
        elif "resource_guard.py" in joined:
            payload = {"ok": True, "memory_pressure_kind": "none"}
        elif "storage_failback_sync.py" in joined:
            payload = {"ok": True, "mode": "external", "active_root": "/Volumes/BOT_LOGS/schwab_trading_bot", "autosync": {"copied_files": 0}, "low_space_autoprune": {"deleted_count": 0}}
        elif "sql_link_shard_manager.py" in joined:
            shard_calls["count"] += 1
            if shard_calls["count"] == 1:
                payload = {"ok": False, "reason": "writer_lock_busy", "owner": "pid=123"}
            else:
                payload = {"ok": True, "reason": "ok"}
        elif "sqlite_performance_maintenance.py" in joined:
            payload = {"ok": True, "wal_size_gb_before": 3.5, "wal_size_gb_after": 0.8}
        elif "stale_artifact_sweeper_bot.py" in joined:
            payload = {"ok": True, "summary": {"candidate_files": 0, "staged_files": 0, "staged_bytes": 0, "delete_errors": 0}}
        elif "stale_artifact_reaper_bot.py" in joined:
            payload = {"ok": True, "summary": {"candidate_files": 0, "deleted_files": 0, "deleted_bytes": 0, "delete_errors": 0}}
        elif "data_retention_policy.py" in joined:
            payload = {"deleted": 0, "delete_errors": 0}
        elif "content_addressed_artifact_store.py" in joined:
            payload = {"ok": True, "skipped_blob_count": 0, "gc": {"deleted_blob_count": 0, "deleted_bytes": 0}}
        else:
            raise AssertionError(f"unexpected command: {cmd}")
        return {"cmd": cmd, "rc": 0, "duration_ms": 5.0, "payload": payload, "stdout_tail": "", "stderr_tail": ""}

    monkeypatch.setattr(storage_maintenance_lane, "_run_json_command", _fake_run)

    payload = storage_maintenance_lane.build_storage_maintenance_payload(
        project_root,
        resource_profile="optional",
        force=False,
        vacuum=False,
    )

    assert payload["ok"] is True
    assert shard_calls["count"] == 2
    assert payload["summary"]["shard_follow_through_completed"] is True
    assert payload["steps"]["sql_link_shard_manager_follow_through"]["status"] == "ok"
