import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.daily_auto_verify as daily_auto_verify
import scripts.health_gates as health_gates
import scripts.ops.runtime_gate_dashboard as runtime_gate_dashboard


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_health_gates_prefers_freshest_ingestion_payload(tmp_path: Path) -> None:
    older = tmp_path / "jsonl_sql_ingestion_health_latest.json"
    newer = tmp_path / "jsonl_sql_ingestion_health_trading_latest.json"
    now = datetime.now(timezone.utc)

    _write_json(older, {"timestamp_utc": (now - timedelta(hours=3)).isoformat(), "sqlite": {"pending_lines": 90}})
    _write_json(newer, {"timestamp_utc": now.isoformat(), "sqlite": {"pending_lines": 5}})

    payload, source = health_gates._freshest_non_empty_json([older, newer])

    assert source == str(newer)
    assert payload["sqlite"]["pending_lines"] == 5


def test_daily_auto_verify_resolves_best_freshness_artifact(tmp_path: Path) -> None:
    original_root = daily_auto_verify.PROJECT_ROOT
    original_groups = daily_auto_verify.DEFAULT_FRESHNESS_FILE_GROUPS
    try:
        daily_auto_verify.PROJECT_ROOT = tmp_path
        old_file = tmp_path / "governance" / "health" / "jsonl_sql_ingestion_health_latest.json"
        new_file = tmp_path / "governance" / "health" / "jsonl_sql_ingestion_health_trading_latest.json"
        now = datetime.now(timezone.utc)
        _write_json(old_file, {"timestamp_utc": (now - timedelta(hours=2)).isoformat()})
        _write_json(new_file, {"timestamp_utc": now.isoformat()})
        daily_auto_verify.DEFAULT_FRESHNESS_FILE_GROUPS = [[old_file, new_file]]

        resolved = daily_auto_verify._resolve_freshness_files("")

        assert resolved == [new_file]
    finally:
        daily_auto_verify.PROJECT_ROOT = original_root
        daily_auto_verify.DEFAULT_FRESHNESS_FILE_GROUPS = original_groups


def test_runtime_gate_dashboard_ignores_stale_daily_verify_failures_when_fresh_gates_are_green(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    walk_root = tmp_path / "governance" / "walk_forward"

    _write_json(
        health_root / "session_ready_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": True, "expected_profiles": ["default"], "checks": []},
    )
    _write_json(
        health_root / "health_gates_latest.json",
        {"timestamp_utc": now.isoformat(), "data_quality_score": 99.9, "hard_gate_triggered": False, "inputs": {"blocked_rate": 0.01}},
    )
    _write_json(
        health_root / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "sqlite": {"pending_lines": 0, "oldest_uningested_age_seconds": 0.0, "invalid": 0}},
    )
    _write_json(
        health_root / "sql_link_service_progress_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": True, "running": True, "status": "running", "current_step": "merge_primary"},
    )
    _write_json(
        health_root / "daily_auto_verify_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "ok": False,
            "failed_checks": ["new_bot_graduation_gate", "replay_hash_registry_guard"],
            "completed_checks": 39,
        },
    )
    _write_json(health_root / "replay_hash_registry_guard_latest.json", {"timestamp_utc": now.isoformat(), "ok": True, "failed_checks": []})
    _write_json(health_root / "promotion_quality_gate_latest.json", {"timestamp_utc": now.isoformat(), "ok": True, "failed_checks": []})
    _write_json(
        walk_root / "new_bot_graduation_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": True, "maturity": {"mature_bots": 8}, "immature_active_count": 0},
    )
    _write_json(
        walk_root / "promotion_readiness_latest.json",
        {"timestamp_utc": now.isoformat(), "promote_ok": True, "considered_bots": 5, "failed_bots": 0, "fail_share": 0.0},
    )
    _write_json(tmp_path / "master_bot_registry.json", {"summary": {"total_bots": 1, "active_bots": 1, "deleted_from_rotation": 0}, "sub_bots": []})

    payload = runtime_gate_dashboard.build_dashboard(tmp_path)

    assert "daily_auto_verify_not_ok" not in payload["overall"]["attention"]
    assert payload["artifacts"]["daily_auto_verify"]["summary"]["effective_failed_checks"] == []
    assert sorted(payload["artifacts"]["daily_auto_verify"]["summary"]["resolved_failed_checks"]) == [
        "new_bot_graduation_gate",
        "replay_hash_registry_guard",
    ]


def test_runtime_gate_dashboard_dedupes_daily_verify_when_only_promotion_gate_remains(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    walk_root = tmp_path / "governance" / "walk_forward"

    _write_json(
        health_root / "session_ready_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": True, "expected_profiles": ["default"], "checks": []},
    )
    _write_json(
        health_root / "health_gates_latest.json",
        {"timestamp_utc": now.isoformat(), "data_quality_score": 99.9, "hard_gate_triggered": False, "inputs": {"blocked_rate": 0.01}},
    )
    _write_json(
        health_root / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "sqlite": {"pending_lines": 0, "oldest_uningested_age_seconds": 0.0, "invalid": 0}},
    )
    _write_json(
        health_root / "sql_link_service_progress_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": True, "running": True, "status": "running", "current_step": "merge_primary"},
    )
    _write_json(
        health_root / "daily_auto_verify_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "ok": False,
            "failed_checks": ["promotion_quality_gate"],
            "completed_checks": 39,
        },
    )
    _write_json(health_root / "replay_hash_registry_guard_latest.json", {"timestamp_utc": now.isoformat(), "ok": True, "failed_checks": []})
    _write_json(health_root / "promotion_quality_gate_latest.json", {"timestamp_utc": now.isoformat(), "ok": False, "failed_checks": ["promotion_gate_blocked"]})
    _write_json(
        walk_root / "new_bot_graduation_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": True, "maturity": {"mature_bots": 8}, "immature_active_count": 0},
    )
    _write_json(
        walk_root / "promotion_readiness_latest.json",
        {"timestamp_utc": now.isoformat(), "promote_ok": False, "considered_bots": 5, "failed_bots": 3, "fail_share": 0.6},
    )
    _write_json(tmp_path / "master_bot_registry.json", {"summary": {"total_bots": 1, "active_bots": 1, "deleted_from_rotation": 0}, "sub_bots": []})

    payload = runtime_gate_dashboard.build_dashboard(tmp_path)

    assert "promotion_not_ready" in payload["overall"]["attention"]
    assert "daily_auto_verify_not_ok" not in payload["overall"]["attention"]
    assert payload["artifacts"]["daily_auto_verify"]["summary"]["effective_failed_checks"] == ["promotion_quality_gate"]


def test_daily_auto_verify_artifact_freshness_accepts_artifacts_written_during_run(tmp_path: Path) -> None:
    artifact = tmp_path / "governance" / "health" / "session_ready_latest.json"
    now = datetime.now(timezone.utc)
    _write_json(artifact, {"timestamp_utc": (now - timedelta(minutes=90)).isoformat()})

    status = daily_auto_verify._artifact_freshness_status(
        [artifact],
        max_age_minutes=20.0,
        fresh_if_newer_than=now - timedelta(hours=2),
    )

    assert status["ok"] is True
    assert status["rows"][0]["refreshed_in_run"] is True


def test_runtime_gate_dashboard_uses_current_registry_and_trading_ingestion(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    walk_root = tmp_path / "governance" / "walk_forward"

    _write_json(
        health_root / "session_ready_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": True, "expected_profiles": ["default", "fx"], "checks": []},
    )
    _write_json(
        health_root / "daily_auto_verify_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": True, "failed_checks": [], "completed_checks": 5},
    )
    _write_json(
        health_root / "health_gates_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "data_quality_score": 88.2,
            "hard_gate_triggered": False,
            "inputs": {"blocked_rate": 0.22},
        },
    )
    _write_json(
        health_root / "jsonl_sql_ingestion_health_latest.json",
        {
            "timestamp_utc": (now - timedelta(hours=4)).isoformat(),
            "sqlite": {"pending_lines": 777, "oldest_uningested_age_seconds": 999.0, "invalid": 3},
        },
    )
    _write_json(
        health_root / "jsonl_sql_ingestion_health_trading_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "files_discovered": 12,
            "sqlite": {"pending_lines": 5, "oldest_uningested_age_seconds": 12.0, "invalid": 0},
        },
    )
    _write_json(
        health_root / "sql_link_service_latest.json",
        {
            "timestamp_utc": (now - timedelta(hours=3)).isoformat(),
            "ok": True,
            "current_step": "complete",
            "merged_rows_this_cycle": 99,
        },
    )
    _write_json(
        health_root / "sql_link_service_progress_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "ok": True,
            "running": True,
            "status": "running",
            "current_step": "merge_primary",
            "completed_shard_count": 2,
            "completed_merge_count": 1,
            "merged_rows_this_cycle": 42,
        },
    )
    _write_json(
        walk_root / "promotion_readiness_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "promote_ok": False,
            "considered_bots": 7,
            "failed_bots": 2,
            "fail_share": 0.285714,
        },
    )
    _write_json(
        tmp_path / "master_bot_registry.json",
        {
            "updated_at_utc": now.isoformat(),
            "summary": {
                "total_bots": 96,
                "active_bots": 15,
                "deleted_from_rotation": 69,
                "deletion_guard_ok": False,
                "deletion_guard_reason": "training_success_not_confirmed",
                "top_active": [{"bot_id": "brain_refinery_v4_simple"}],
            },
            "sub_bots": [{"bot_id": "brain_refinery_v4_simple", "active": True}],
        },
    )

    payload = runtime_gate_dashboard.build_dashboard(tmp_path)

    assert payload["artifacts"]["sql_ingestion"]["path"].endswith("jsonl_sql_ingestion_health_trading_latest.json")
    assert payload["artifacts"]["sql_ingestion"]["summary"]["pending_lines"] == 5
    assert payload["artifacts"]["sql_link_service"]["path"].endswith("sql_link_service_progress_latest.json")
    assert payload["artifacts"]["sql_link_service"]["summary"]["current_step"] == "merge_primary"
    assert payload["registry"]["active_bots"] == 15
    assert payload["registry"]["total_bots"] == 96
    assert payload["overall"]["status"] == "warn"
    assert "promotion_not_ready" in payload["overall"]["attention"]


def test_runtime_gate_dashboard_uses_service_heartbeat_for_sql_ingestion_freshness(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    walk_root = tmp_path / "governance" / "walk_forward"

    _write_json(
        health_root / "session_ready_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": True, "expected_profiles": ["default"], "checks": []},
    )
    _write_json(
        health_root / "health_gates_latest.json",
        {"timestamp_utc": now.isoformat(), "data_quality_score": 99.9, "hard_gate_triggered": False, "inputs": {"blocked_rate": 0.01}},
    )
    _write_json(
        health_root / "sql_link_service_progress_latest.json",
        {"timestamp_utc": (now - timedelta(minutes=5)).isoformat(), "ok": True, "running": True, "status": "running", "current_step": "shard_linking"},
    )
    _write_json(
        health_root / "jsonl_sql_ingestion_health_trading_latest.json",
        {
            "timestamp_utc": (now - timedelta(minutes=45)).isoformat(),
            "files_discovered": 12,
            "sqlite": {"pending_lines": 0, "oldest_uningested_age_seconds": 0.0, "invalid": 0},
        },
    )
    _write_json(
        walk_root / "promotion_readiness_latest.json",
        {"timestamp_utc": now.isoformat(), "promote_ok": True, "considered_bots": 5, "failed_bots": 0, "fail_share": 0.0},
    )
    _write_json(tmp_path / "master_bot_registry.json", {"summary": {"total_bots": 1, "active_bots": 1, "deleted_from_rotation": 0}, "sub_bots": []})

    payload = runtime_gate_dashboard.build_dashboard(tmp_path)

    assert payload["artifacts"]["sql_ingestion"]["stale"] is False
    assert payload["artifacts"]["sql_ingestion"]["summary"]["freshness_via_service_heartbeat"] is True
    assert "sql_ingestion_stale" not in payload["overall"]["attention"]


def test_runtime_gate_dashboard_suppresses_sql_service_stale_when_ingestion_is_fresh(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    walk_root = tmp_path / "governance" / "walk_forward"

    _write_json(
        health_root / "session_ready_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": True, "expected_profiles": ["default"], "checks": []},
    )
    _write_json(
        health_root / "health_gates_latest.json",
        {"timestamp_utc": now.isoformat(), "data_quality_score": 99.9, "hard_gate_triggered": False, "inputs": {"blocked_rate": 0.01}},
    )
    _write_json(
        health_root / "sql_link_service_progress_latest.json",
        {"timestamp_utc": (now - timedelta(minutes=45)).isoformat(), "ok": True, "running": True, "status": "running", "current_step": "shard_linking"},
    )
    _write_json(
        health_root / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "sqlite": {"pending_lines": 0, "oldest_uningested_age_seconds": 0.0, "invalid": 0}},
    )
    _write_json(
        walk_root / "promotion_readiness_latest.json",
        {"timestamp_utc": now.isoformat(), "promote_ok": True, "considered_bots": 5, "failed_bots": 0, "fail_share": 0.0},
    )
    _write_json(tmp_path / "master_bot_registry.json", {"summary": {"total_bots": 1, "active_bots": 1, "deleted_from_rotation": 0}, "sub_bots": []})

    payload = runtime_gate_dashboard.build_dashboard(tmp_path)

    assert "sql_link_service_stale" not in payload["overall"]["attention"]
    assert payload["artifacts"]["sql_link_service"]["summary"]["freshness_inferred_from_sql_ingestion"] is True


def test_runtime_gate_dashboard_suppresses_sql_stale_when_live_writer_lock_exists(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    walk_root = tmp_path / "governance" / "walk_forward"
    lock_path = tmp_path / "governance" / "locks" / "jsonl_sql_writer.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(f"pid={os.getpid()} started={now.isoformat()} cmd=sql_link_shard_manager", encoding="utf-8")

    _write_json(
        health_root / "session_ready_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": True, "expected_profiles": ["default"], "checks": []},
    )
    _write_json(
        health_root / "health_gates_latest.json",
        {"timestamp_utc": now.isoformat(), "data_quality_score": 99.9, "hard_gate_triggered": False, "inputs": {"blocked_rate": 0.01}},
    )
    _write_json(
        health_root / "sql_link_service_progress_latest.json",
        {
            "timestamp_utc": (now - timedelta(minutes=45)).isoformat(),
            "ok": True,
            "running": True,
            "status": "running",
            "current_step": "merge_primary",
            "lock_path": str(lock_path),
        },
    )
    _write_json(
        health_root / "jsonl_sql_ingestion_health_governance_latest.json",
        {
            "timestamp_utc": (now - timedelta(minutes=45)).isoformat(),
            "files_discovered": 20,
            "sqlite": {"pending_lines": 0, "oldest_uningested_age_seconds": 0.0, "invalid": 0},
        },
    )
    _write_json(
        walk_root / "promotion_readiness_latest.json",
        {"timestamp_utc": now.isoformat(), "promote_ok": True, "considered_bots": 5, "failed_bots": 0, "fail_share": 0.0},
    )
    _write_json(tmp_path / "master_bot_registry.json", {"summary": {"total_bots": 1, "active_bots": 1, "deleted_from_rotation": 0}, "sub_bots": []})

    payload = runtime_gate_dashboard.build_dashboard(tmp_path)

    assert "sql_link_service_stale" not in payload["overall"]["attention"]
    assert "sql_ingestion_stale" not in payload["overall"]["attention"]
    assert payload["artifacts"]["sql_link_service"]["summary"]["freshness_inferred_from_live_lock"] is True
    assert payload["artifacts"]["sql_link_service"]["summary"]["lock_owner_pid"] == os.getpid()
    assert payload["artifacts"]["sql_ingestion"]["summary"]["freshness_via_service_heartbeat"] is True


def test_runtime_gate_dashboard_suppresses_session_ready_stale_when_shadow_loop_is_fresh(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    walk_root = tmp_path / "governance" / "walk_forward"

    _write_json(
        health_root / "session_ready_latest.json",
        {"timestamp_utc": (now - timedelta(minutes=20)).isoformat(), "ok": True, "expected_profiles": ["default"], "checks": []},
    )
    _write_json(
        health_root / "shadow_loop_default_equities_schwab_1234.json",
        {"timestamp_utc": now.isoformat(), "ok": True},
    )
    _write_json(
        health_root / "health_gates_latest.json",
        {"timestamp_utc": now.isoformat(), "data_quality_score": 99.9, "hard_gate_triggered": False, "inputs": {"blocked_rate": 0.01}},
    )
    _write_json(
        health_root / "sql_link_service_progress_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": True, "running": True, "status": "running", "current_step": "shard_linking"},
    )
    _write_json(
        health_root / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "sqlite": {"pending_lines": 0, "oldest_uningested_age_seconds": 0.0, "invalid": 0}},
    )
    _write_json(
        walk_root / "promotion_readiness_latest.json",
        {"timestamp_utc": now.isoformat(), "promote_ok": True, "considered_bots": 5, "failed_bots": 0, "fail_share": 0.0},
    )
    _write_json(tmp_path / "master_bot_registry.json", {"summary": {"total_bots": 1, "active_bots": 1, "deleted_from_rotation": 0}, "sub_bots": []})

    payload = runtime_gate_dashboard.build_dashboard(tmp_path)

    assert "session_ready_stale" not in payload["overall"]["attention"]
    assert payload["artifacts"]["session_ready"]["summary"]["freshness_inferred_from_shadow_loop"] is True


def test_runtime_gate_dashboard_uses_day_based_units_for_optional_artifacts(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    walk_root = tmp_path / "governance" / "walk_forward"

    _write_json(
        health_root / "session_ready_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": True, "expected_profiles": ["default"], "checks": []},
    )
    _write_json(
        health_root / "health_gates_latest.json",
        {"timestamp_utc": now.isoformat(), "data_quality_score": 99.9, "hard_gate_triggered": False, "inputs": {"blocked_rate": 0.01}},
    )
    _write_json(
        health_root / "sql_link_service_progress_latest.json",
        {"timestamp_utc": now.isoformat(), "ok": True, "running": True, "status": "running", "current_step": "shard_linking"},
    )
    _write_json(
        health_root / "jsonl_sql_ingestion_health_trading_latest.json",
        {"timestamp_utc": now.isoformat(), "sqlite": {"pending_lines": 0, "oldest_uningested_age_seconds": 0.0, "invalid": 0}},
    )
    _write_json(
        health_root / "official_macro_context_sync_latest.json",
        {"timestamp_utc": (now - timedelta(hours=10)).isoformat(), "ok": True, "sources": {"fed": {"ok": True}}},
    )
    _write_json(
        health_root / "live_macro_media_status.json",
        {"timestamp_utc": (now - timedelta(hours=12)).isoformat(), "ok": True, "learning_ready": True, "training_feature_count": 10},
    )
    _write_json(
        walk_root / "promotion_readiness_latest.json",
        {"timestamp_utc": now.isoformat(), "promote_ok": True, "considered_bots": 5, "failed_bots": 0, "fail_share": 0.0},
    )
    _write_json(tmp_path / "master_bot_registry.json", {"summary": {"total_bots": 1, "active_bots": 1, "deleted_from_rotation": 0}, "sub_bots": []})

    payload = runtime_gate_dashboard.build_dashboard(tmp_path)

    assert payload["artifacts"]["official_macro_context_sync"]["stale"] is False
    assert payload["artifacts"]["live_macro_media"]["stale"] is False


def test_daily_auto_verify_uses_slow_timeout_for_heavy_checks() -> None:
    slow_timeout = 300

    assert daily_auto_verify._timeout_for_check("daily_runtime_summary", slow_timeout) == slow_timeout
    assert daily_auto_verify._timeout_for_check("data_source_divergence_bot", slow_timeout) == slow_timeout
    assert daily_auto_verify._timeout_for_check("replay_preopen_sanity", slow_timeout) == slow_timeout
    assert daily_auto_verify._timeout_for_check("resource_guard", slow_timeout) == daily_auto_verify.DEFAULT_CMD_TIMEOUT_SEC


def test_daily_auto_verify_active_progress_pid_requires_recent_live_pid(tmp_path: Path) -> None:
    progress_path = tmp_path / "daily_auto_verify_progress_latest.json"
    now = datetime.now(timezone.utc)
    _write_json(
        progress_path,
        {
            "timestamp_utc": now.isoformat(),
            "running": True,
            "pid": os.getpid(),
        },
    )

    active_pid = daily_auto_verify._active_progress_pid(progress_path, max_age_seconds=300)

    assert active_pid == os.getpid()


def test_daily_auto_verify_active_progress_pid_ignores_stale_progress(tmp_path: Path) -> None:
    progress_path = tmp_path / "daily_auto_verify_progress_latest.json"
    _write_json(
        progress_path,
        {
            "timestamp_utc": (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat(),
            "running": True,
            "pid": os.getpid(),
        },
    )

    active_pid = daily_auto_verify._active_progress_pid(progress_path, max_age_seconds=300)

    assert active_pid is None


def test_daily_auto_verify_main_skips_when_recent_progress_pid_is_alive(tmp_path: Path, monkeypatch, capsys) -> None:
    original_progress = daily_auto_verify.PROGRESS_PATH
    original_lock = daily_auto_verify.LOCK_PATH
    try:
        progress_path = tmp_path / "governance" / "health" / "daily_auto_verify_progress_latest.json"
        lock_path = tmp_path / "governance" / "locks" / "daily_auto_verify.lock"
        _write_json(
            progress_path,
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "running": True,
                "pid": os.getpid(),
            },
        )
        daily_auto_verify.PROGRESS_PATH = progress_path
        daily_auto_verify.LOCK_PATH = lock_path
        monkeypatch.setattr(sys, "argv", ["daily_auto_verify.py", "--day", "20260327", "--json"])

        rc = daily_auto_verify.main()
        payload = json.loads(capsys.readouterr().out)

        assert rc == 0
        assert payload["note"] == f"already_running_progress pid={os.getpid()}"
        assert payload["lock_path"] == str(lock_path)
        assert not lock_path.exists()
    finally:
        daily_auto_verify.PROGRESS_PATH = original_progress
        daily_auto_verify.LOCK_PATH = original_lock


def test_daily_auto_verify_recovers_stale_progress_to_latest(tmp_path: Path) -> None:
    progress_path = tmp_path / "governance" / "health" / "daily_auto_verify_progress_latest.json"
    latest_path = tmp_path / "governance" / "health" / "daily_auto_verify_latest.json"
    _write_json(
        progress_path,
        {
            "timestamp_utc": (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat(),
            "running": True,
            "pid": 999999,
            "current_check": "health_gates",
            "completed_checks": 36,
            "ok": True,
            "failed_checks": [],
            "checks": {"health_gates": {"ok": True}},
        },
    )
    _write_json(
        latest_path,
        {
            "timestamp_utc": (datetime.now(timezone.utc) - timedelta(hours=4)).isoformat(),
            "running": False,
            "ok": True,
            "failed_checks": [],
            "checks": {},
        },
    )

    note = daily_auto_verify._recover_stale_progress(progress_path, latest_path, max_age_seconds=300)
    recovered = json.loads(latest_path.read_text(encoding="utf-8"))

    assert "recovered_stale_progress" in note
    assert progress_path.exists() is False
    assert recovered["running"] is False
    assert recovered["ok"] is False
    assert "incomplete_run_recovered" in recovered["failed_checks"]
    assert recovered["current_check"] == "health_gates"


def test_health_gates_weights_data_blocked_more_than_risk_blocked(tmp_path: Path, monkeypatch) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    sql_root = tmp_path / "exports" / "sql_reports"

    _write_json(
        health_root / "one_numbers_latest.json",
        {
            "generated_utc": now.isoformat(),
            "combined_blocked_rate": "0.560000",
            "data_blocked_rate": "0.100000",
            "risk_blocked_rate": "0.460000",
            "decision_stale_windows_4h": "0",
            "watchdog_restarts": "0",
        },
    )
    _write_json(
        sql_root / "daily_runtime_summary_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "watchdog": {"restarts": 0},
        },
    )
    _write_json(
        health_root / "jsonl_sql_ingestion_health_trading_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "sqlite": {"pending_lines": 0, "oldest_uningested_age_seconds": 0.0, "invalid": 0},
            "latency_slo": {"sqlite": {"all": {"p95_seconds": 5.0}}},
        },
    )
    _write_json(
        health_root / "ingestion_backpressure_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "pending_lines": 0,
            "pending_files": 0,
            "oldest_pending_age_seconds": 0.0,
            "overload": False,
        },
    )

    monkeypatch.setattr(sys, "argv", ["health_gates.py", "--project-root", str(tmp_path)])
    rc = health_gates.main()

    payload = json.loads((health_root / "health_gates_latest.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["hard_gate_triggered"] is False
    assert payload["inputs"]["combined_blocked_rate"] == 0.56
    assert abs(float(payload["inputs"]["blocked_rate"]) - 0.215) < 1e-9


def test_health_gates_falls_back_to_legacy_combined_blocked_rate(tmp_path: Path, monkeypatch) -> None:
    now = datetime.now(timezone.utc)
    health_root = tmp_path / "governance" / "health"
    sql_root = tmp_path / "exports" / "sql_reports"

    _write_json(
        health_root / "one_numbers_latest.json",
        {
            "generated_utc": now.isoformat(),
            "combined_blocked_rate": "0.410000",
            "decision_stale_windows_4h": "0",
            "watchdog_restarts": "0",
        },
    )
    _write_json(health_root / "ingestion_backpressure_latest.json", {"timestamp_utc": now.isoformat(), "overload": False})
    _write_json(
        health_root / "jsonl_sql_ingestion_health_trading_latest.json",
        {
            "timestamp_utc": now.isoformat(),
            "sqlite": {"pending_lines": 0, "oldest_uningested_age_seconds": 0.0, "invalid": 0},
        },
    )
    _write_json(sql_root / "daily_runtime_summary_latest.json", {"timestamp_utc": now.isoformat(), "watchdog": {"restarts": 0}})

    monkeypatch.setattr(sys, "argv", ["health_gates.py", "--project-root", str(tmp_path)])
    health_gates.main()

    payload = json.loads((health_root / "health_gates_latest.json").read_text(encoding="utf-8"))

    assert payload["inputs"]["data_blocked_rate"] == 0.41
    assert payload["inputs"]["risk_blocked_rate"] == 0.0
    assert payload["inputs"]["blocked_rate"] == 0.41
