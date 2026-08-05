import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import backpressure_drainer_fleet as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_backpressure_drainer_fleet_routes_concentrated_decisions_to_trading(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 64455,
            "pending_lines_total": 64472,
            "top_pending_files": [
                {
                    "source_rel": "governance/channels/decision/conservative_equities_schwab/decision_20260430.jsonl",
                    "pending_lines": 30675,
                    "oldest_pending_age_seconds": 7200.0,
                },
                {
                    "source_rel": "governance/channels/decision/aggressive_equities_schwab/decision_20260430.jsonl",
                    "pending_lines": 29664,
                    "oldest_pending_age_seconds": 7200.0,
                },
            ],
        },
    )
    _write_json(health / "ingestion_storage_control_latest.json", {"severity": "critical"})

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 5, 1, 2, 0, tzinfo=timezone.utc),
    )

    assert payload["overall_status"] == "ready"
    assert payload["active_drainer"]["name"] == "core_decision_drainer"
    assert payload["active_drainer"]["shards"][:2] == ["trading", "aggressive_trading"]
    assert payload["active_drainer"]["concentration"]["concentrated"] is True
    assert payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARDS"].startswith("trading,")
    assert payload["active_env_overrides"]["SQL_LINK_SERVICE_CONCENTRATED_CORE_DRAIN"] == "1"
    assert payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARD_LINK_TIMEOUT_SECONDS"] == "420"
    assert payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_MAX_LINES_PER_FILE"] == "12000"
    assert payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARD_TRADING_MAX_LINES_PER_FILE"] == "24000"
    assert "conservative_equities_schwab" in payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARD_TRADING_PATH_CONTAINS"]


def test_backpressure_drainer_fleet_byte_bounds_sparse_large_decision_rows(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 12000,
            "pending_lines_total": 12000,
            "top_pending_files": [
                {
                    "source_rel": "governance/channels/decision/dividend_capture_equities_schwab/decision_20260519.jsonl",
                    "pending_lines": 9000,
                    "oldest_pending_age_seconds": 900.0,
                    "file_size_bytes": 2_200_000_000,
                    "estimated_avg_bytes_per_line": 250_000.0,
                    "estimated_pending_bytes": 2_200_000_000,
                    "sparse_large_line": True,
                },
                {
                    "source_rel": "decisions/paper/trade_decisions_20260519.jsonl",
                    "pending_lines": 3000,
                    "oldest_pending_age_seconds": 300.0,
                },
            ],
        },
    )
    _write_json(health / "ingestion_storage_control_latest.json", {"severity": "critical"})

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 5, 19, 21, 45, tzinfo=timezone.utc),
    )

    env = payload["active_env_overrides"]
    assert payload["active_drainer"]["name"] == "core_decision_drainer"
    assert payload["active_drainer"]["sparse_large_line_pressure"]["active"] is True
    assert payload["active_drainer"]["sparse_large_line_pressure"]["estimated_pending_bytes"] == 2_200_000_000
    assert env["SQL_LINK_SERVICE_SPARSE_LARGE_DECISION_DRAIN"] == "1"
    assert env["INGEST_MAX_BYTES_PER_FILE"] == str(128 * 1024 * 1024)
    assert env["SQLITE_BATCH_MAX_BYTES"] == str(32 * 1024 * 1024)
    assert env["SQL_LINK_SERVICE_SHARD_LINK_TIMEOUT_SECONDS"] == "420"
    assert env["BACKLOG_PCORE_ALLOCATION_ACTIVE"] == "1"
    assert env["BACKLOG_DRAIN_SINGLE_WRITER_ONLY"] == "1"
    assert env["SQL_LINK_SERVICE_SINGLE_WRITER_ONLY"] == "1"
    assert int(env["BACKLOG_PCORE_PREPROCESS_WORKERS"]) >= 1
    assert env["BOT_COLLECTION_DUTY_CYCLE_ENABLED"] == "1"
    assert env["TRAINING_RUNTIME_PAUSED_FOR_BACKLOG"] == "1"
    p_core = payload["active_drainer"]["p_core_backlog_allocation_contract"]
    assert p_core["policy"] == "p_core_preprocess_single_sql_writer"
    assert p_core["single_writer_only"] is True
    assert p_core["training_pcore_gate"]["allowed_when_backlog_green"] is True
    assert env["SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE"] == "180"
    assert env["SQL_LINK_SERVICE_SHARD_TRADING_MAX_FILES"] == "8"
    assert env["SQL_LINK_SERVICE_SHARD_TRADING_MAX_LINES_PER_FILE"] == "12000"
    assert env["SQL_LINK_SERVICE_SHARD_TRADING_STATE_CHECKPOINT_LINES"] == "2000"
    assert env["SQL_LINK_SERVICE_SHARD_TRADING_MERGE_MAX_JSONL_ROWS"] == "2000"
    assert env["SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_MERGE_MAX_JSONL_ROWS"] == "24000"
    assert env["SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_MERGE_MAX_JSONL_ROWS"] == "32000"


def test_backpressure_drainer_fleet_suppresses_stale_raw_risk_when_overlay_is_fresh_empty(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    risk_sources = [
        "governance/channels/risk/default_crypto_schwab/risk_20260701.jsonl",
        "governance/channels/risk/fx_equities_schwab/risk_20260701.jsonl",
        "governance/channels/risk/conservative_equities_schwab/risk_20260701.jsonl",
        "governance/channels/risk/aggressive_equities_schwab/risk_20260701.jsonl",
        "governance/channels/risk/dividend_equities_schwab/risk_20260701.jsonl",
        "governance/channels/risk/bond_equities_schwab/risk_20260701.jsonl",
    ]
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 31999,
            "pending_lines_total": 261614,
            "pending_lines_deferred": 229615,
            "oldest_pending_age_seconds": 148.891,
            "top_pending_files": [
                {
                    "source_rel": "governance/events/signal_generation_20260701.jsonl",
                    "pending_lines": 25006,
                    "oldest_pending_age_seconds": 14.266,
                },
                {
                    "source_rel": "decisions/shadow_crypto/trade_decisions_20260701.jsonl",
                    "pending_lines": 3835,
                    "oldest_pending_age_seconds": 81.115,
                },
            ],
            "top_deferred_pending_files": [
                {
                    "source_rel": source,
                    "pending_lines": pending,
                    "oldest_pending_age_seconds": 82.529,
                }
                for source, pending in zip(risk_sources, [72400, 44000, 33044, 30000, 22000, 14546])
            ],
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "severity": "high",
            "backlog_truth": {
                "authoritative_mode": "overlay_fresh_shard_level",
                "overlay_decay": {"should_decay": False, "attribution_ratio": 1.0},
            },
            "sql_ingestion_pending_overlay": {
                "active": True,
                "used_for_pressure": True,
                "source_count": 19,
                "fresh_source_count": 19,
                "stale_source_count": 0,
                "explicit_empty_source_count": 15,
                "total_pending_lines": 32490,
                "core_pending_lines": 32490,
                "fresh_path_contains": [
                    *risk_sources,
                    "governance/events/signal_generation_",
                    "shadow_crypto/",
                ],
                "top_pending_files": [
                    {
                        "source_rel": "governance/events/signal_generation_20260701.jsonl",
                        "shard": "governance",
                        "pending_lines": 25006,
                        "oldest_pending_age_seconds": 14.266,
                    },
                    {
                        "source_rel": "decisions/shadow_crypto/trade_decisions_20260701.jsonl",
                        "shard": "crypto_trading",
                        "pending_lines": 3835,
                        "oldest_pending_age_seconds": 81.115,
                    },
                ],
            },
        },
    )

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 7, 1, 14, 0, tzinfo=timezone.utc),
    )

    risk = next(row for row in payload["candidate_drainers"] if row["name"] == "risk_support_drainer")
    assert risk["status"] == "idle"
    assert risk["pending_lines"] == 0
    assert payload["active_drainer"]["name"] == "core_decision_drainer"
    assert payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARDS"].startswith("crypto_trading,governance,")
    assert "governance/events/signal_generation_20260701.jsonl" in payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARD_GOVERNANCE_PATH_CONTAINS"]


def test_backpressure_drainer_fleet_bursts_p_core_seven_when_host_is_deep_green(monkeypatch) -> None:
    monkeypatch.setenv("BACKLOG_PCORE_TARGET", "8")
    workers, intelligence = src._p_core_preprocess_workers(
        critical=True,
        backpressure={
            "pending_lines": 30000,
            "pending_lines_total": 40000,
            "oldest_pending_age_seconds": 1200.0,
            "pending_lines_threshold": 15000,
            "oldest_age_threshold_seconds": 240.0,
            "line_estimation": {"sparse_large_line_active": True},
        },
        host_context={
            "off_hours_active": True,
            "runtime_throttle": {
                "host_saturation_score": 35.0,
                "compute_pressure_level": "normal",
                "memory_pressure_level": "normal",
                "throttle_profile": "soft_cap",
            },
            "resource_guard": {"creative_session_level": "idle"},
        },
    )

    assert workers == 7
    assert intelligence["mode"] == "full_p_core_budget_7_plus_primary_writer"
    assert intelligence["seventh_core_burst"]["allowed"] is True


def test_backpressure_drainer_fleet_uses_four_worker_protect_live_probe_for_extreme_backlog(monkeypatch) -> None:
    monkeypatch.setenv("BACKLOG_PCORE_TARGET", "8")
    monkeypatch.delenv("BACKLOG_PCORE_PREPROCESS_WORKERS_OVERRIDE", raising=False)
    workers, intelligence = src._p_core_preprocess_workers(
        critical=True,
        backpressure={
            "pending_lines": 3_000_000,
            "pending_lines_total": 5_000_000,
            "oldest_pending_age_seconds": 14_000.0,
            "pending_lines_threshold": 15000,
            "oldest_age_threshold_seconds": 240.0,
            "line_estimation": {"sparse_large_line_active": True},
        },
        host_context={
            "runtime_throttle": {
                "host_saturation_score": 55.0,
                "compute_pressure_level": "elevated",
                "memory_pressure_level": "normal",
                "memory_pressure_kind": "none",
                "throttle_profile": "protect_live",
                "swap_used_gb": 1.8,
            },
            "resource_guard": {"creative_session_level": "idle", "compressed_store_gb": 9.5},
            "computer_task": {"primary_task": "backlog_drain"},
        },
    )

    assert workers == 4
    assert intelligence["mode"] == "protect_live_backlog_probe_4"
    assert intelligence["protected_live_backlog_probe"]["wide_allowed"] is True


def test_backpressure_drainer_fleet_caps_after_recent_storage_eject(tmp_path: Path, monkeypatch) -> None:
    log_path = tmp_path / "storage_eject_guard.log"
    now = datetime.now(timezone.utc)
    log_path.write_text(
        f"[{now.isoformat().replace('+00:00', 'Z')}] disk disappeared mountRoot=/Volumes/BOT_LOGS volumeBSD=disk5s1 wholeBSD=disk5 mode=external\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("STORAGE_EJECT_GUARD_LOG", str(log_path))
    monkeypatch.setenv("BACKLOG_PCORE_TARGET", "8")
    monkeypatch.delenv("BACKLOG_PCORE_PREPROCESS_WORKERS_OVERRIDE", raising=False)

    workers, intelligence = src._p_core_preprocess_workers(
        critical=True,
        backpressure={
            "pending_lines": 26000,
            "pending_lines_total": 42000,
            "oldest_pending_age_seconds": 1200.0,
            "pending_lines_threshold": 15000,
            "oldest_age_threshold_seconds": 240.0,
            "line_estimation": {"sparse_large_line_active": True, "sparse_large_line_pending_bytes": 1_000_000_000},
        },
        host_context={
            "off_hours_active": True,
            "runtime_throttle": {
                "host_saturation_score": 35.0,
                "compute_pressure_level": "normal",
                "memory_pressure_level": "normal",
                "throttle_profile": "soft_cap",
            },
            "resource_guard": {"creative_session_level": "idle"},
        },
    )

    assert workers == 3
    assert intelligence["mode"] == "storage_eject_cooldown_3"
    assert intelligence["storage_eject_cooldown"]["active"] is True
    assert intelligence["storage_eject_cooldown"]["previous_selected_workers"] == 7


def test_backpressure_drainer_fleet_keeps_three_worker_probe_under_guarded_host_saturation(monkeypatch) -> None:
    monkeypatch.setenv("BACKLOG_PCORE_TARGET", "8")
    monkeypatch.delenv("BACKLOG_PCORE_PREPROCESS_WORKERS_OVERRIDE", raising=False)
    workers, intelligence = src._p_core_preprocess_workers(
        critical=True,
        backpressure={
            "pending_lines": 3_000_000,
            "pending_lines_total": 5_000_000,
            "oldest_pending_age_seconds": 14_000.0,
            "pending_lines_threshold": 15000,
            "oldest_age_threshold_seconds": 240.0,
            "line_estimation": {"sparse_large_line_active": True},
        },
        host_context={
            "runtime_throttle": {
                "host_saturation_score": 63.0,
                "compute_pressure_level": "elevated",
                "memory_pressure_level": "normal",
                "memory_pressure_kind": "none",
                "throttle_profile": "protect_live",
                "swap_used_gb": 1.8,
            },
            "resource_guard": {"creative_session_level": "idle", "compressed_store_gb": 9.5},
            "computer_task": {"primary_task": "backlog_drain"},
        },
    )

    assert workers == 3
    assert intelligence["mode"] == "protect_live_backlog_probe_3"
    assert intelligence["protected_live_backlog_probe"]["allowed"] is True
    assert intelligence["protected_live_backlog_probe"]["wide_allowed"] is False


def test_backpressure_drainer_fleet_holds_three_workers_when_compute_is_high_but_memory_clear(monkeypatch) -> None:
    monkeypatch.setenv("BACKLOG_PCORE_TARGET", "8")
    monkeypatch.setenv("BACKLOG_PCORE_USER_APP_RESERVE_TARGET", "5")
    monkeypatch.delenv("BACKLOG_PCORE_PREPROCESS_WORKERS_OVERRIDE", raising=False)
    workers, intelligence = src._p_core_preprocess_workers(
        critical=True,
        backpressure={
            "pending_lines": 2_100_000,
            "pending_lines_total": 4_400_000,
            "oldest_pending_age_seconds": 29_000.0,
            "pending_lines_threshold": 15000,
            "oldest_age_threshold_seconds": 240.0,
            "line_estimation": {"sparse_large_line_active": True},
        },
        host_context={
            "runtime_throttle": {
                "host_saturation_score": 59.52,
                "compute_pressure_level": "high",
                "memory_pressure_level": "normal",
                "memory_pressure_kind": "none",
                "throttle_profile": "protect_live",
                "swap_used_gb": 1.8,
            },
            "resource_guard": {"creative_session_level": "idle", "compressed_store_gb": 8.0},
            "computer_task": {"primary_task": "backlog_drain"},
        },
    )

    assert workers == 3
    assert intelligence["mode"] == "protect_live_backlog_probe_3"
    assert intelligence["protected_live_backlog_probe"]["allowed"] is True
    assert intelligence["protected_live_backlog_probe"]["wide_allowed"] is False


def test_backpressure_drainer_fleet_keeps_guarded_three_worker_pump_when_host_is_warm(monkeypatch) -> None:
    monkeypatch.setenv("BACKLOG_PCORE_TARGET", "8")
    monkeypatch.setenv("BACKLOG_PCORE_USER_APP_RESERVE_TARGET", "5")
    monkeypatch.delenv("BACKLOG_PCORE_PREPROCESS_WORKERS_OVERRIDE", raising=False)
    workers, intelligence = src._p_core_preprocess_workers(
        critical=True,
        backpressure={
            "pending_lines": 2_100_000,
            "pending_lines_total": 4_400_000,
            "oldest_pending_age_seconds": 29_000.0,
            "pending_lines_threshold": 15000,
            "oldest_age_threshold_seconds": 240.0,
            "line_estimation": {"sparse_large_line_active": True},
        },
        host_context={
            "runtime_throttle": {
                "host_saturation_score": 72.0,
                "compute_pressure_level": "elevated",
                "memory_pressure_level": "normal",
                "memory_pressure_kind": "none",
                "throttle_profile": "protect_live",
                "swap_used_gb": 1.8,
            },
            "resource_guard": {"creative_session_level": "idle", "compressed_store_gb": 12.5},
            "computer_task": {"primary_task": "backlog_drain"},
        },
    )

    assert workers == 3
    assert intelligence["mode"] == "guarded_backlog_probe_3"
    assert intelligence["guarded_backlog_probe"]["allowed"] is True


def test_backpressure_drainer_fleet_honors_six_p_core_user_reserve_target(monkeypatch) -> None:
    monkeypatch.setenv("BACKLOG_PCORE_TARGET", "8")
    monkeypatch.setenv("BACKLOG_PCORE_USER_APP_RESERVE_TARGET", "6")
    monkeypatch.delenv("BACKLOG_PCORE_PREPROCESS_WORKERS_OVERRIDE", raising=False)
    workers, intelligence = src._p_core_preprocess_workers(
        critical=True,
        backpressure={
            "pending_lines": 3_000_000,
            "pending_lines_total": 5_000_000,
            "oldest_pending_age_seconds": 14_000.0,
            "pending_lines_threshold": 15000,
            "oldest_age_threshold_seconds": 240.0,
            "line_estimation": {"sparse_large_line_active": True},
        },
        host_context={
            "runtime_throttle": {
                "host_saturation_score": 35.0,
                "compute_pressure_level": "normal",
                "memory_pressure_level": "normal",
                "memory_pressure_kind": "none",
                "throttle_profile": "soft_cap",
                "swap_used_gb": 1.0,
            },
            "resource_guard": {"creative_session_level": "idle", "compressed_store_gb": 4.0},
            "computer_task": {"primary_task": "backlog_drain"},
        },
    )

    assert workers == 2
    assert intelligence["user_app_reserve"]["target_p_cores"] == 6
    assert intelligence["user_app_reserve"]["worker_cap"] == 2


def test_backpressure_drainer_fleet_protects_creative_p_core_headroom(monkeypatch) -> None:
    monkeypatch.setenv("BACKLOG_PCORE_TARGET", "8")
    workers, intelligence = src._p_core_preprocess_workers(
        critical=True,
        backpressure={
            "pending_lines": 30000,
            "pending_lines_total": 40000,
            "oldest_pending_age_seconds": 1200.0,
            "pending_lines_threshold": 15000,
            "oldest_age_threshold_seconds": 240.0,
            "line_estimation": {"sparse_large_line_active": True},
        },
        host_context={
            "off_hours_active": True,
            "runtime_throttle": {
                "host_saturation_score": 30.0,
                "compute_pressure_level": "normal",
                "memory_pressure_level": "normal",
            },
            "resource_guard": {"creative_session_level": "hot", "creative_session_kind": "audio_production"},
            "computer_task": {"primary_task": "audio_production"},
        },
    )

    assert workers == 3
    assert intelligence["mode"] == "creative_foreground_protect_3"


def test_backpressure_drainer_fleet_narrows_p_core_workers_for_memory_pressure(monkeypatch) -> None:
    monkeypatch.setenv("BACKLOG_PCORE_TARGET", "8")
    workers, intelligence = src._p_core_preprocess_workers(
        critical=True,
        backpressure={
            "pending_lines": 30000,
            "pending_lines_total": 40000,
            "oldest_pending_age_seconds": 1200.0,
            "pending_lines_threshold": 15000,
            "oldest_age_threshold_seconds": 240.0,
            "line_estimation": {"sparse_large_line_active": True},
        },
        host_context={
            "off_hours_active": True,
            "runtime_throttle": {
                "host_saturation_score": 35.0,
                "compute_pressure_level": "normal",
                "memory_pressure_level": "red",
                "throttle_profile": "sustain",
            },
            "resource_guard": {"memory_pressure_kind": "throttled", "swap_used_gb": 19.0, "pages_throttled": 1},
        },
    )

    assert workers == 2
    assert intelligence["mode"] == "memory_relief_2"


def test_backpressure_drainer_fleet_routes_crypto_decisions_to_crypto_shard(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 1200,
            "pending_lines_total": 1200,
            "top_pending_files": [
                {
                    "source_rel": "governance/channels/decision/default_crypto_schwab/decision_20260501.jsonl",
                    "pending_lines": 800,
                    "oldest_pending_age_seconds": 900.0,
                },
                {
                    "source_rel": "decisions/shadow_crypto_futures_crypto/trade_decisions_20260501.jsonl",
                    "pending_lines": 400,
                    "oldest_pending_age_seconds": 600.0,
                },
            ],
        },
    )

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 5, 1, 2, 0, tzinfo=timezone.utc),
    )

    assert payload["overall_status"] == "ready"
    assert payload["active_drainer"]["name"] == "core_decision_drainer"
    assert payload["active_drainer"]["shards"][:1] == ["crypto_trading"]
    assert payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARDS"].startswith("crypto_trading,")
    assert "SQL_LINK_SERVICE_SHARD_TRADING_PATH_CONTAINS" not in payload["active_env_overrides"]
    assert "default_crypto_schwab" in payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_PATH_CONTAINS"]


def test_backpressure_drainer_fleet_writes_single_writer_handoff(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 1000,
            "pending_lines_total": 1000,
            "pending_lines_support_telemetry": 1000,
            "top_support_telemetry_pending_files": [
                {
                    "source_rel": "governance/watchdog/failover_events.jsonl",
                    "pending_lines": 1000,
                    "oldest_pending_age_seconds": 300.0,
                }
            ],
        },
    )

    payload = src.build_payload(
        project_root,
        apply=True,
        now_utc=datetime(2026, 5, 1, 15, 0, tzinfo=timezone.utc),
    )

    assert payload["overall_status"] == "handoff_requested"
    assert payload["service_request"]["request_kind"] == "backpressure_drainer_fleet"
    assert payload["service_request"]["reason"].endswith(":support_watchdog_drainer")
    assert payload["service_request"]["env_overrides"]["SQL_LINK_SERVICE_SHARDS"] == "support_watchdog,health_fast"
    assert payload["service_request"]["env_overrides"]["BACKLOG_PCORE_ALLOCATION_ACTIVE"] == "1"
    assert payload["service_request"]["p_core_backlog_allocation_contract"]["sqlite_writer_count"] == 1
    assert (health / "sql_link_service_request_latest.json").exists()


def test_backpressure_drainer_fleet_routes_operations_guard_feedback_tails(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 420,
            "pending_lines_total": 3923,
            "oldest_age_threshold_seconds": 240.0,
            "top_pending_files": [
                {
                    "source_rel": "governance/events/paper_execution_guard_20260630.jsonl",
                    "pending_lines": 241,
                    "oldest_pending_age_seconds": 554.0,
                },
                {
                    "source_rel": "governance/distillation/teacher_student_events_20260630.jsonl",
                    "pending_lines": 117,
                    "oldest_pending_age_seconds": 315.0,
                },
            ],
            "top_deferred_pending_files": [
                {
                    "source_rel": "governance/health/infrabot_adaptive_feedback.jsonl",
                    "pending_lines": 1460,
                    "oldest_pending_age_seconds": 56.0,
                },
                {
                    "source_rel": "governance/health/adaptive_regression_guard_feedback.jsonl",
                    "pending_lines": 222,
                    "oldest_pending_age_seconds": 712.0,
                },
            ],
        },
    )

    payload = src.build_payload(
        project_root,
        apply=True,
        now_utc=datetime(2026, 5, 1, 15, 0, tzinfo=timezone.utc),
    )

    assert payload["overall_status"] == "handoff_requested"
    assert payload["active_drainer"]["name"] == "operations_guard_drainer"
    assert payload["active_drainer"]["assigned_pressure_lane"] == "operations_guard_feedback_backpressure"
    assert "adaptive_regression_guard" in payload["active_drainer"]["ops_infrabots"]
    assert "infrabot_adaptive_governor" in payload["active_drainer"]["ops_infrabots"]
    env = payload["service_request"]["env_overrides"]
    assert env["SQL_LINK_SERVICE_SHARDS"].startswith("governance,")
    assert "paper_execution_guard_20260630" in env["SQL_LINK_SERVICE_SHARD_GOVERNANCE_PATH_CONTAINS"]
    assert "infrabot_adaptive_feedback" in env["SQL_LINK_SERVICE_SHARD_GOVERNANCE_PATH_CONTAINS"]


def test_backpressure_drainer_fleet_handoffs_overlay_risk_support_backlog(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 125,
            "pending_lines_total": 352,
            "pending_lines_support_telemetry": 6,
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "severity": "elevated",
            "backpressure": {
                "core_pending_lines": 292,
                "support_pending_lines": 346679,
                "total_pending_lines": 347198,
            },
            "sql_ingestion_pending_overlay": {
                "active": True,
                "total_pending_lines": 346971,
                "support_pending_lines": 346679,
                "top_pending_files": [
                    {
                        "source_rel": "governance/channels/risk/default_crypto_schwab/risk_20260521.jsonl",
                        "shard": "risk_support",
                        "pending_lines": 269938,
                        "oldest_pending_age_seconds": 7.198,
                    },
                    {
                        "source_rel": "governance/channels/risk/crypto_futures_crypto_schwab/risk_20260521.jsonl",
                        "shard": "risk_support",
                        "pending_lines": 76741,
                        "oldest_pending_age_seconds": 6.827,
                    },
                ],
            },
        },
    )

    payload = src.build_payload(
        project_root,
        apply=True,
        now_utc=datetime(2026, 5, 21, 15, 0, tzinfo=timezone.utc),
    )

    assert payload["overall_status"] == "handoff_requested"
    assert payload["active_drainer"]["name"] == "risk_support_drainer"
    assert payload["active_drainer"]["pending_lines"] == 346679
    assert payload["service_request"]["assigned_pressure_lane"] == "risk_support_backpressure"
    env = payload["service_request"]["env_overrides"]
    assert env["SQL_LINK_SERVICE_SHARDS"] == "risk_support,health_fast"
    assert "default_crypto_schwab" in env["SQL_LINK_SERVICE_SHARD_RISK_SUPPORT_PATH_CONTAINS"]
    assert env["SQL_LINK_SERVICE_SHARD_RISK_SUPPORT_MAX_LINES_PER_FILE"] == "160000"
    assert env["SQL_LINK_SERVICE_SHARD_RISK_SUPPORT_STATE_CHECKPOINT_LINES"] == "8000"
    assert env["BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO"] == "0.20"


def test_backpressure_drainer_fleet_handoffs_raw_live_deferred_risk_backlog(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 3078,
            "pending_lines_total": 8200,
            "oldest_pending_age_seconds": 231.0,
            "top_pending_files": [
                {
                    "source_rel": "decisions/shadow_crypto/trade_decisions_20260729.jsonl",
                    "pending_lines": 1798,
                    "oldest_pending_age_seconds": 231.0,
                }
            ],
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "severity": "high",
            "backpressure": {
                "core_pending_lines": 3078,
                "deferred_pending_lines": 534169,
                "total_pending_lines": 540247,
                "raw_live": {
                    "core_pending_lines": 3078,
                    "total_pending_lines": 540247,
                    "oldest_pending_age_seconds": 231.553,
                    "top_pending_files": [
                        {
                            "source_rel": "decisions/shadow_crypto/trade_decisions_20260729.jsonl",
                            "pending_lines": 1798,
                            "oldest_pending_age_seconds": 231.553,
                        }
                    ],
                    "top_deferred_pending_files": [
                        {
                            "source_rel": "governance/channels/risk/default_crypto_schwab/risk_20260729.jsonl",
                            "shard": "risk_support",
                            "pending_lines": 505819,
                            "oldest_pending_age_seconds": 231.415,
                        }
                    ],
                },
            },
            "sql_ingestion_pending_overlay": {
                "active": True,
                "used_for_pressure": True,
                "total_pending_lines": 3114,
                "top_pending_files": [],
                "fresh_path_contains": ["governance/channels/risk/"],
            },
            "backlog_truth": {"authoritative_mode": "overlay_source_attributed"},
        },
    )

    payload = src.build_payload(
        project_root,
        apply=True,
        now_utc=datetime(2026, 7, 29, 23, 40, tzinfo=timezone.utc),
    )

    assert payload["overall_status"] == "handoff_requested"
    assert payload["active_drainer"]["name"] == "risk_support_drainer"
    assert payload["active_drainer"]["pending_lines"] == 505819
    assert payload["service_request"]["assigned_pressure_lane"] == "risk_support_backpressure"
    env = payload["service_request"]["env_overrides"]
    assert env["SQL_LINK_SERVICE_SHARDS"] == "risk_support,health_fast"
    assert "risk_20260729" in env["SQL_LINK_SERVICE_SHARD_RISK_SUPPORT_PATH_CONTAINS"]


def test_backpressure_drainer_fleet_prioritizes_risk_channel_when_it_is_raw_live_pressure(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 192,
            "pending_lines_total": 563828,
            "pending_lines_deferred": 563636,
            "oldest_pending_age_seconds": 408.391,
            "pending_lines_threshold": 15000,
            "oldest_age_threshold_seconds": 240.0,
            "top_pending_files": [
                {
                    "source_rel": "decisions/shadow_crypto/trade_decisions_20260730.jsonl",
                    "pending_lines": 138,
                    "oldest_pending_age_seconds": 408.391,
                }
            ],
            "top_deferred_pending_files": [
                {
                    "source_rel": "governance/channels/risk/swing_aggressive_equities_schwab/risk_20260730.jsonl",
                    "pending_lines": 412479,
                    "oldest_pending_age_seconds": 410.232,
                },
                {
                    "source_rel": "governance/channels/risk/aggressive_equities_schwab/risk_20260730.jsonl",
                    "pending_lines": 62402,
                    "oldest_pending_age_seconds": 408.315,
                },
                {
                    "source_rel": "governance/channels/risk/crypto_futures_crypto_schwab/risk_20260730.jsonl",
                    "pending_lines": 49191,
                    "oldest_pending_age_seconds": 408.909,
                },
                {
                    "source_rel": "governance/channels/ingress/crypto_futures_crypto_schwab/ingress_20260730.jsonl",
                    "pending_lines": 3735,
                    "oldest_pending_age_seconds": 408.604,
                },
                {
                    "source_rel": "governance/channels/ingress/aggressive_equities_schwab/ingress_20260730.jsonl",
                    "pending_lines": 3184,
                    "oldest_pending_age_seconds": 407.794,
                },
            ],
        },
    )
    _write_json(health / "ingestion_storage_control_latest.json", {"severity": "high"})

    payload = src.build_payload(
        project_root,
        apply=True,
        now_utc=datetime(2026, 7, 30, 13, 5, tzinfo=timezone.utc),
    )

    risk = next(row for row in payload["candidate_drainers"] if row["name"] == "risk_support_drainer")
    api = next(row for row in payload["candidate_drainers"] if row["name"] == "api_ingress_drainer")
    assert risk["raw_live_expansion_risk_channel_pressure"] is True
    assert risk["raw_live_expansion_preemption_tier"] == api["raw_live_expansion_preemption_tier"] == 3
    assert risk["effective_priority_score"] > api["effective_priority_score"]
    assert payload["active_drainer"]["name"] == "risk_support_drainer"
    assert payload["service_request"]["assigned_pressure_lane"] == "risk_support_backpressure"
    env = payload["service_request"]["env_overrides"]
    assert env["SQL_LINK_SERVICE_SHARDS"] == "risk_support,health_fast"
    assert "swing_aggressive_equities_schwab" in env["SQL_LINK_SERVICE_SHARD_RISK_SUPPORT_PATH_CONTAINS"]


def test_backpressure_drainer_fleet_does_not_use_deferred_support_age_as_core_expansion_age(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 102,
            "pending_lines_total": 8510,
            "pending_lines_deferred": 8408,
            "pending_lines_support_telemetry": 101,
            "oldest_pending_age_seconds": 0.0,
            "pending_lines_threshold": 15000,
            "oldest_age_threshold_seconds": 240.0,
            "top_pending_files": [
                {
                    "source_rel": "decisions/shadow_crypto/trade_decisions_20260730.jsonl",
                    "pending_lines": 54,
                    "oldest_pending_age_seconds": 574.798,
                },
                {
                    "source_rel": "governance/events/auth_events_20260730.jsonl",
                    "pending_lines": 24,
                    "oldest_pending_age_seconds": 2.99,
                },
            ],
            "top_deferred_pending_files": [
                {
                    "source_rel": "governance/channels/ingress/aggressive_equities_schwab/ingress_20260730.jsonl",
                    "pending_lines": 3184,
                    "oldest_pending_age_seconds": 1341.99,
                },
                {
                    "source_rel": "governance/channels/ingress/crypto_futures_crypto_schwab/ingress_20260730.jsonl",
                    "pending_lines": 3745,
                    "oldest_pending_age_seconds": 573.854,
                },
            ],
            "top_support_telemetry_pending_files": [
                {
                    "source_rel": "governance/watchdog/incident_auto_halt_events.jsonl",
                    "pending_lines": 1,
                    "oldest_pending_age_seconds": 28986.759,
                }
            ],
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "steady_state": {"target_status": {"steady_state_ready": True}},
            "stale_pending_locator": {
                "status": "clear",
                "oldest_pending_age_seconds": 0.0,
                "stale_source_count": 0,
            },
            "backpressure": {
                "core_pending_lines": 94,
                "total_pending_lines": 8407,
                "oldest_pending_age_seconds": 0.0,
            },
            "raw_live_expansion_contract": {
                "active": True,
                "targets": {
                    "core_reserve_lines": 4000,
                    "total_reserve_lines": 5500,
                    "oldest_age_reserve_seconds": 180.0,
                },
            },
        },
    )

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 7, 30, 13, 20, tzinfo=timezone.utc),
    )

    guard = payload["metrics"]["raw_live_expansion_guard"]
    assert guard["active"] is True
    assert guard["ratios"]["total"] == 1.274
    assert guard["ratios"]["oldest_age"] == 0.0
    assert guard["raw_live"]["age_guard_source_pending_lines"] == 78
    assert guard["raw_live"]["guard_oldest_pending_age_seconds"] == 0.0
    assert guard["raw_live"]["deferred_or_support_hot_source_oldest_pending_age_seconds"] == 28986.759


def test_backpressure_drainer_fleet_prioritizes_core_when_risk_overlay_is_louder(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 109000,
            "pending_lines_total": 2540000,
            "oldest_pending_age_seconds": 70500.0,
            "pending_lines_threshold": 15000,
            "oldest_age_threshold_seconds": 240.0,
            "top_pending_files": [
                {
                    "source_rel": "governance/events/signal_generation_20260729.jsonl",
                    "shard": "governance",
                    "pending_lines": 104000,
                    "oldest_pending_age_seconds": 480.0,
                }
            ],
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "severity": "critical",
            "backpressure": {
                "core_pending_lines": 109000,
                "support_pending_lines": 612000,
                "total_pending_lines": 3150000,
            },
            "sql_ingestion_pending_overlay": {
                "active": True,
                "used_for_pressure": True,
                "source_count": 1,
                "fresh_source_count": 1,
                "stale_source_count": 0,
                "total_pending_lines": 612000,
                "support_pending_lines": 612000,
                "top_pending_files": [
                    {
                        "source_rel": "governance/channels/risk/aggressive_equities_schwab/risk_20260729.jsonl",
                        "shard": "risk_support",
                        "pending_lines": 612000,
                        "oldest_pending_age_seconds": 960.0,
                    }
                ],
            },
            "backlog_truth": {"authoritative_mode": "overlay_sql_ingestion"},
            "overlay_decay": {"should_decay": False, "attribution_ratio": 1.0},
        },
    )

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 7, 29, 16, 0, tzinfo=timezone.utc),
    )

    risk = next(row for row in payload["candidate_drainers"] if row["name"] == "risk_support_drainer")
    core = next(row for row in payload["candidate_drainers"] if row["name"] == "core_decision_drainer")
    assert risk["status"] == "ready"
    assert core["status"] == "ready"
    assert risk["raw_live_expansion_preemption_tier"] < core["raw_live_expansion_preemption_tier"]
    assert risk["raw_live_expansion_priority_bonus"] == 0
    assert payload["active_drainer"]["name"] == "core_decision_drainer"
    assert payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARDS"].startswith("governance,")


def test_backpressure_drainer_fleet_prioritizes_core_reserve_before_support_overlay(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 13644,
            "pending_lines_total": 22611,
            "oldest_pending_age_seconds": 559.0,
            "pending_lines_threshold": 15000,
            "oldest_age_threshold_seconds": 240.0,
            "top_pending_files": [
                {
                    "source_rel": "decisions/shadow_crypto/trade_decisions_20260730.jsonl",
                    "shard": "crypto_trading",
                    "pending_lines": 7270,
                    "oldest_pending_age_seconds": 344.0,
                },
                {
                    "source_rel": "decisions/shadow_crypto_futures_crypto/trade_decisions_20260730.jsonl",
                    "shard": "crypto_trading",
                    "pending_lines": 2912,
                    "oldest_pending_age_seconds": 374.0,
                },
                {
                    "source_rel": "governance/events/signal_generation_20260730.jsonl",
                    "shard": "governance",
                    "pending_lines": 2458,
                    "oldest_pending_age_seconds": 346.0,
                },
            ],
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "severity": "high",
            "backpressure": {
                "core_pending_lines": 13644,
                "support_pending_lines": 332707,
                "total_pending_lines": 350318,
                "raw_live": {
                    "core_pending_lines": 13644,
                    "total_pending_lines": 22611,
                    "oldest_pending_age_seconds": 559.0,
                },
            },
            "raw_live_expansion_contract": {
                "active": True,
                "targets": {
                    "core_reserve_lines": 4000,
                    "total_reserve_lines": 5500,
                    "oldest_age_reserve_seconds": 180.0,
                },
            },
            "sql_ingestion_pending_overlay": {
                "active": True,
                "used_for_pressure": True,
                "source_count": 1,
                "fresh_source_count": 1,
                "stale_source_count": 0,
                "total_pending_lines": 333193,
                "support_pending_lines": 332707,
                "top_pending_files": [
                    {
                        "source_rel": "governance/channels/risk/default_crypto_schwab/risk_20260730.jsonl",
                        "shard": "risk_support",
                        "pending_lines": 332707,
                        "oldest_pending_age_seconds": 423.0,
                    }
                ],
            },
            "backlog_truth": {"authoritative_mode": "overlay_sql_ingestion"},
            "overlay_decay": {"should_decay": False, "attribution_ratio": 1.0},
        },
    )

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 7, 30, 1, 15, tzinfo=timezone.utc),
    )

    risk = next(row for row in payload["candidate_drainers"] if row["name"] == "risk_support_drainer")
    core = next(row for row in payload["candidate_drainers"] if row["name"] == "core_decision_drainer")
    assert risk["status"] == "ready"
    assert core["status"] == "ready"
    assert risk["raw_live_expansion_core_handoff_required"] is True
    assert risk["raw_live_expansion_preemption_tier"] < core["raw_live_expansion_preemption_tier"]
    assert risk["raw_live_expansion_priority_bonus"] == 0
    assert payload["active_drainer"]["name"] == "core_decision_drainer"
    assert payload["active_env_overrides"]["SQL_LINK_SERVICE_RAW_LIVE_PRIORITY_BOOST"] == "1"


def test_backpressure_drainer_fleet_promotes_overwhelming_risk_support_when_core_is_mild(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 6430,
            "pending_lines_total": 14862908,
            "oldest_pending_age_seconds": 434.0,
            "pending_lines_threshold": 15000,
            "oldest_age_threshold_seconds": 240.0,
            "top_pending_files": [
                {
                    "source_rel": "governance/channels/ingress/aggressive_equities_schwab/ingress_20260730.jsonl",
                    "shard": "governance",
                    "pending_lines": 5191,
                    "oldest_pending_age_seconds": 0.2,
                }
            ],
            "top_deferred_pending_files": [
                {
                    "source_rel": "governance/channels/risk/default_crypto_schwab/risk_20260730.jsonl",
                    "shard": "risk_support",
                    "pending_lines": 7_788_913,
                    "oldest_pending_age_seconds": 0.0,
                },
                {
                    "source_rel": "governance/channels/risk/crypto_futures_crypto_schwab/risk_20260730.jsonl",
                    "shard": "risk_support",
                    "pending_lines": 4_975_524,
                    "oldest_pending_age_seconds": 0.0,
                },
                {
                    "source_rel": "governance/channels/ingress/intraday_aggressive_equities_schwab/ingress_20260730.jsonl",
                    "shard": "governance",
                    "pending_lines": 2990,
                    "oldest_pending_age_seconds": 0.5,
                },
            ],
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "severity": "critical",
            "backpressure": {
                "core_pending_lines": 6430,
                "deferred_pending_lines": 12_777_652,
                "support_pending_lines": 2_078_826,
                "total_pending_lines": 14_862_908,
                "raw_live": {
                    "core_pending_lines": 6430,
                    "total_pending_lines": 12_784_082,
                    "oldest_pending_age_seconds": 434.0,
                },
            },
            "raw_live_expansion_contract": {
                "active": True,
                "targets": {
                    "core_reserve_lines": 4000,
                    "total_reserve_lines": 5500,
                    "oldest_age_reserve_seconds": 180.0,
                },
            },
        },
    )

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 7, 30, 20, 30, tzinfo=timezone.utc),
    )

    risk = next(row for row in payload["candidate_drainers"] if row["name"] == "risk_support_drainer")
    api = next(row for row in payload["candidate_drainers"] if row["name"] == "api_ingress_drainer")
    assert risk["raw_live_expansion_dominant_risk_support_pressure"] is True
    assert risk["raw_live_expansion_preemption_tier"] == api["raw_live_expansion_preemption_tier"] == 3
    assert risk["effective_priority_score"] > api["effective_priority_score"]
    assert payload["active_drainer"]["name"] == "risk_support_drainer"
    assert payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARDS"] == "risk_support,health_fast"


def test_backpressure_drainer_fleet_scores_sql_overlay_signal_generation_before_tiny_runtime(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 220,
            "pending_lines_total": 220,
            "top_pending_files": [
                {
                    "source_rel": "governance/channels/runtime/default_equities_schwab/runtime_20260527.jsonl",
                    "pending_lines": 220,
                    "oldest_pending_age_seconds": 900.0,
                },
            ],
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "severity": "critical",
            "sql_ingestion_pending_overlay": {
                "active": True,
                "total_pending_lines": 1_450_688,
                "core_pending_lines": 1_450_688,
                "top_pending_files": [
                    {
                        "source_rel": "governance/events/signal_generation_20260527.jsonl",
                        "shard": "governance",
                        "pending_lines": 1_450_688,
                        "oldest_pending_age_seconds": 5.0,
                    },
                ],
            },
        },
    )

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 5, 27, 22, 0, tzinfo=timezone.utc),
    )

    assert payload["overall_status"] == "ready"
    assert payload["active_drainer"]["name"] == "core_decision_drainer"
    assert payload["active_drainer"]["pending_lines"] == 1_450_688
    assert payload["active_drainer"]["path_focus"] == ["governance/events/signal_generation_20260527.jsonl"]
    env = payload["active_env_overrides"]
    assert "governance" in payload["active_drainer"]["shards"]
    assert "signal_generation_20260527" in env["SQL_LINK_SERVICE_SHARD_GOVERNANCE_PATH_CONTAINS"]
    assert env["SQL_LINK_SERVICE_SHARD_GOVERNANCE_MAX_LINES_PER_FILE"] == "512000"
    assert env["SQL_LINK_SERVICE_SHARD_GOVERNANCE_MERGE_MAX_JSONL_ROWS"] == "256000"
    assert env["INGEST_MAX_BYTES_PER_FILE"] == str(1024 * 1024 * 1024)
    assert env["SQLITE_BATCH_MAX_BYTES"] == str(256 * 1024 * 1024)


def test_backpressure_drainer_fleet_handoffs_sql_overlay_explanation_tails_off_hours(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 12,
            "pending_lines_total": 12,
            "top_pending_files": [],
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "severity": "critical",
            "sql_ingestion_pending_overlay": {
                "active": True,
                "total_pending_lines": 126_386,
                "cold_pending_lines": 126_386,
                "top_pending_files": [
                    {
                        "source_rel": "decision_explanations/shadow_neural_operator_surrogates_equities/decision_explanations_20260527.jsonl",
                        "shard": "crypto_explanations",
                        "pressure_lane": "cold",
                        "pending_lines": 126_386,
                        "oldest_pending_age_seconds": 35_804.0,
                    },
                ],
            },
        },
    )

    payload = src.build_payload(
        project_root,
        apply=True,
        now_utc=datetime(2026, 5, 27, 22, 0, tzinfo=timezone.utc),
    )

    assert payload["overall_status"] == "handoff_requested"
    assert payload["metrics"]["raw_live_expansion_guard"]["active"] is False
    assert payload["active_drainer"]["name"] == "cold_stage_drainer"
    assert payload["active_drainer"]["pending_lines"] == 126_386
    env = payload["service_request"]["env_overrides"]
    assert env["SQL_LINK_SERVICE_SHARDS"] == "data,explanations,crypto_explanations,health_fast"
    assert "shadow_neural_operator_surrogates" in env["SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_PATH_CONTAINS"]


def test_backpressure_drainer_fleet_reserves_raw_live_handoff_before_cold_overlay(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 6411,
            "pending_lines_total": 6990,
            "oldest_pending_age_seconds": 120.0,
            "oldest_age_threshold_seconds": 240.0,
            "top_pending_files": [
                {
                    "source_rel": "governance/channels/decision/default_equities_schwab/decision_20260527.jsonl",
                    "pending_lines": 6411,
                    "oldest_pending_age_seconds": 120.0,
                }
            ],
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "severity": "critical",
            "sql_ingestion_pending_overlay": {
                "active": True,
                "total_pending_lines": 126_386,
                "cold_pending_lines": 126_386,
                "top_pending_files": [
                    {
                        "source_rel": "decision_explanations/shadow_neural_operator_surrogates_equities/decision_explanations_20260527.jsonl",
                        "shard": "crypto_explanations",
                        "pressure_lane": "cold",
                        "pending_lines": 126_386,
                        "oldest_pending_age_seconds": 35_804.0,
                    },
                ],
            },
        },
    )

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 5, 27, 22, 0, tzinfo=timezone.utc),
    )

    assert payload["overall_status"] == "ready"
    assert payload["metrics"]["raw_live_expansion_guard"]["active"] is True
    assert payload["active_drainer"]["name"] == "core_decision_drainer"
    assert payload["active_drainer"]["raw_live_expansion_priority_bonus"] > 0
    assert payload["active_env_overrides"]["RAW_LIVE_EXPANSION_GUARD_ACTIVE"] == "1"
    cold = next(row for row in payload["candidate_drainers"] if row["name"] == "cold_stage_drainer")
    assert cold["raw_live_expansion_cold_penalty"] > 0
    assert cold["effective_priority_score"] < cold["priority_score"]


def test_backpressure_drainer_fleet_keeps_hot_core_ahead_of_small_stale_governance_tail(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 6599,
            "pending_lines_total": 6604,
            "oldest_pending_age_seconds": 3897.893,
            "oldest_age_threshold_seconds": 240.0,
            "top_pending_files": [
                {
                    "source_rel": "governance/events/signal_generation_20260731.jsonl",
                    "pending_lines": 2948,
                    "oldest_pending_age_seconds": 123.016,
                },
                {
                    "source_rel": "decisions/shadow_crypto/trade_decisions_20260731.jsonl",
                    "pending_lines": 1763,
                    "oldest_pending_age_seconds": 473.212,
                },
                {
                    "source_rel": "governance/events/auth_events_20260730.jsonl",
                    "pending_lines": 82,
                    "oldest_pending_age_seconds": 3897.893,
                },
                {
                    "source_rel": "governance/events/premarket_token_guard_20260730.jsonl",
                    "pending_lines": 75,
                    "oldest_pending_age_seconds": 3753.807,
                },
            ],
        },
    )
    _write_json(health / "ingestion_storage_control_latest.json", {"severity": "elevated"})

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 7, 31, 1, 5, tzinfo=timezone.utc),
    )

    governance = next(row for row in payload["candidate_drainers"] if row["name"] == "governance_execution_drainer")
    core = next(row for row in payload["candidate_drainers"] if row["name"] == "core_decision_drainer")
    assert governance["age_pressure_priority_bonus"] > core["age_pressure_priority_bonus"]
    assert core["raw_live_expansion_core_handoff_required"] is True
    assert core["raw_live_expansion_core_first_preemption"] is True
    assert core["raw_live_expansion_preemption_tier"] == 4
    assert governance["raw_live_expansion_preemption_tier"] == 3
    assert payload["active_drainer"]["name"] == "core_decision_drainer"


def test_backpressure_drainer_fleet_routes_old_governance_event_tails_before_deferred_explanations(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 597,
            "pending_lines_total": 131932,
            "pending_lines_deferred": 131335,
            "oldest_pending_age_seconds": 52833.682,
            "oldest_age_threshold_seconds": 240.0,
            "top_pending_files": [
                {
                    "source_rel": "governance/events/auth_events_20260624.jsonl",
                    "pending_lines": 246,
                    "oldest_pending_age_seconds": 51539.739,
                },
                {
                    "source_rel": "governance/events/execution_lane_stale_skips_20260624.jsonl",
                    "pending_lines": 890,
                    "oldest_pending_age_seconds": 75140.601,
                },
                {
                    "source_rel": "governance/events/live_execution_guard_20260624.jsonl",
                    "pending_lines": 192,
                    "oldest_pending_age_seconds": 52833.682,
                },
                {
                    "source_rel": "governance/events/write_failures_20260624.jsonl",
                    "pending_lines": 310,
                    "oldest_pending_age_seconds": 52642.0,
                },
                {
                    "source_rel": "governance/events/premarket_token_guard_20260624.jsonl",
                    "pending_lines": 68,
                    "oldest_pending_age_seconds": 47973.315,
                },
            ],
            "top_deferred_pending_files": [
                {
                    "source_rel": "decision_explanations/shadow_compound_options_equities/decision_explanations_20260624.jsonl",
                    "pending_lines": 2741,
                    "oldest_pending_age_seconds": 52821.37,
                }
            ],
        },
    )
    _write_json(health / "ingestion_storage_control_latest.json", {"severity": "critical"})

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 6, 25, 13, 15, tzinfo=timezone.utc),
    )

    assert payload["overall_status"] == "ready"
    assert payload["metrics"]["raw_live_expansion_guard"]["active"] is False
    assert payload["active_drainer"]["name"] == "governance_execution_drainer"
    assert payload["active_drainer"]["pending_lines"] == 1706
    assert "auth_events_20260624" in payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARD_GOVERNANCE_PATH_CONTAINS"]
    assert "execution_lane_stale_skips_20260624" in payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARD_GOVERNANCE_PATH_CONTAINS"]
    assert "write_failures_20260624" in payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARD_GOVERNANCE_PATH_CONTAINS"]


def test_backpressure_drainer_fleet_routes_sub_100_stale_execution_skip_tail(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 108,
            "pending_lines_total": 143,
            "pending_lines_deferred": 37,
            "oldest_pending_age_seconds": 0.0,
            "oldest_age_threshold_seconds": 240.0,
            "top_pending_files": [
                {
                    "source_rel": "governance/events/execution_lane_stale_skips_20260729.jsonl",
                    "pending_lines": 80,
                    "oldest_pending_age_seconds": 1589.96,
                },
                {
                    "source_rel": "governance/events/premarket_token_guard_20260729.jsonl",
                    "pending_lines": 4,
                    "oldest_pending_age_seconds": 25.0,
                },
            ],
            "top_deferred_pending_files": [
                {
                    "source_rel": "governance/channels/runtime/intraday_aggressive_equities_schwab/runtime_20260729.jsonl",
                    "pending_lines": 320,
                    "oldest_pending_age_seconds": 9.6,
                }
            ],
        },
    )
    _write_json(health / "ingestion_storage_control_latest.json", {"severity": "stable"})

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 7, 29, 22, 5, tzinfo=timezone.utc),
    )

    governance = next(row for row in payload["candidate_drainers"] if row["name"] == "governance_execution_drainer")
    assert governance["status"] == "ready"
    assert governance["min_pending_lines"] == 25
    assert governance["pending_lines"] == 84
    assert payload["active_drainer"]["name"] == "governance_execution_drainer"
    assert "execution_lane_stale_skips_20260729" in payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARD_GOVERNANCE_PATH_CONTAINS"]


def test_backpressure_drainer_fleet_does_not_freeze_cold_stage_for_tiny_raw_live_tail(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 10,
            "pending_lines_total": 120010,
            "pending_lines_deferred": 120000,
            "oldest_pending_age_seconds": 86400.0,
            "oldest_age_threshold_seconds": 240.0,
            "top_pending_files": [
                {
                    "source_rel": "governance/channels/loop_state/conservative_equities_schwab/loop_state_20260625.jsonl",
                    "pending_lines": 10,
                    "oldest_pending_age_seconds": 86400.0,
                }
            ],
            "top_deferred_pending_files": [
                {
                    "source_rel": "decision_explanations/shadow_dividend_capture_equities/decision_explanations_20260624.jsonl",
                    "pending_lines": 120000,
                    "oldest_pending_age_seconds": 86400.0,
                }
            ],
        },
    )
    _write_json(health / "ingestion_storage_control_latest.json", {"severity": "critical"})

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 6, 25, 2, 0, tzinfo=timezone.utc),
    )

    assert payload["overall_status"] == "ready"
    assert payload["metrics"]["raw_live_expansion_guard"]["active"] is False
    assert payload["active_drainer"]["name"] == "cold_stage_drainer"
    cold = payload["active_drainer"]
    assert cold["raw_live_expansion_cold_penalty"] == 0
    assert cold["pending_lines"] == 120000


def test_backpressure_drainer_fleet_prioritizes_hot_raw_live_age_over_micro_stale_tail(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 411,
            "pending_lines_total": 690,
            "pending_lines_deferred": 279,
            "oldest_pending_age_seconds": 522.378,
            "oldest_age_threshold_seconds": 240.0,
            "top_pending_files": [
                {
                    "source_rel": "governance/events/auth_events_20260630.jsonl",
                    "pending_lines": 255,
                    "oldest_pending_age_seconds": 522.378,
                },
                {
                    "source_rel": "governance/events/premarket_token_guard_20260630.jsonl",
                    "pending_lines": 37,
                    "oldest_pending_age_seconds": 119.266,
                },
                {
                    "source_rel": "governance/events/paper_execution_guard_20260630.jsonl",
                    "pending_lines": 36,
                    "oldest_pending_age_seconds": 161.786,
                },
                {
                    "source_rel": "governance/channels/runtime/provider_adapter_verification_equities_schwab/runtime_20260630.jsonl",
                    "pending_lines": 4,
                    "oldest_pending_age_seconds": 2313.479,
                },
            ],
            "top_deferred_pending_files": [
                {
                    "source_rel": "governance/distillation/teacher_student_events_20260630.jsonl",
                    "pending_lines": 30,
                    "oldest_pending_age_seconds": 649.554,
                },
                {
                    "source_rel": "governance/walk_forward/promotion_readiness_history.jsonl",
                    "pending_lines": 28,
                    "oldest_pending_age_seconds": 332.213,
                },
            ],
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "severity": "high",
            "raw_live_expansion_contract": {
                "active": True,
                "targets": {
                    "core_reserve_lines": 4000,
                    "total_reserve_lines": 5500,
                    "oldest_age_reserve_seconds": 180.0,
                },
            },
        },
    )

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 6, 30, 23, 0, tzinfo=timezone.utc),
    )

    assert payload["overall_status"] == "ready"
    assert payload["metrics"]["raw_live_expansion_guard"]["active"] is True
    assert payload["active_drainer"]["name"] in {"governance_execution_drainer", "operations_guard_drainer"}
    assert payload["active_drainer"]["name"] != "data_quality_contract_drainer"
    assert payload["active_drainer"]["raw_live_expansion_priority_bonus"] > 0
    data_quality = next(row for row in payload["candidate_drainers"] if row["name"] == "data_quality_contract_drainer")
    assert data_quality["readiness_reason"] == "stale_tail"
    assert data_quality["raw_live_expansion_priority_bonus"] == 0


def test_backpressure_drainer_fleet_ignores_stale_hot_overlay_when_raw_core_is_smaller(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 11,
            "pending_lines_total": 120011,
            "pending_lines_deferred": 120000,
            "oldest_pending_age_seconds": 86400.0,
            "top_pending_files": [
                {
                    "source_rel": "governance/events/training/overnight_training_window_20260624.jsonl",
                    "pending_lines": 9,
                    "oldest_pending_age_seconds": 33729.395,
                }
            ],
            "top_deferred_pending_files": [
                {
                    "source_rel": "decision_explanations/shadow_dividend_capture_equities/decision_explanations_20260624.jsonl",
                    "pending_lines": 120000,
                    "oldest_pending_age_seconds": 86400.0,
                }
            ],
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "severity": "critical",
            "sql_ingestion_pending_overlay": {
                "active": True,
                "used_for_pressure": True,
                "source_count": 3,
                "fresh_source_count": 3,
                "stale_source_count": 0,
                "top_pending_files": [
                    {
                        "source_rel": "governance/events/auth_events_20260624.jsonl",
                        "pending_lines": 246,
                        "oldest_pending_age_seconds": 51539.739,
                    },
                    {
                        "source_rel": "governance/events/live_execution_guard_20260624.jsonl",
                        "pending_lines": 192,
                        "oldest_pending_age_seconds": 52833.682,
                    },
                    {
                        "source_rel": "governance/events/premarket_token_guard_20260624.jsonl",
                        "pending_lines": 68,
                        "oldest_pending_age_seconds": 47973.315,
                    },
                ],
            },
            "backlog_truth": {
                "authoritative_mode": "overlay_sql_ingestion",
                "overlay_decay": {"should_decay": False, "attribution_ratio": 1.0},
            },
        },
    )

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 6, 25, 2, 0, tzinfo=timezone.utc),
    )

    assert payload["overall_status"] == "ready"
    assert payload["active_drainer"]["name"] == "cold_stage_drainer"
    governance = next(row for row in payload["candidate_drainers"] if row["name"] == "governance_execution_drainer")
    assert governance["pending_lines"] == 0


def test_backpressure_drainer_fleet_trusts_fresh_overlay_with_zero_stale_pending(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 14051,
            "pending_lines_total": 96811,
            "top_pending_files": [
                {
                    "source_rel": "decisions/shadow_dividend_equities/trade_decisions_20260805.jsonl",
                    "pending_lines": 14051,
                    "oldest_pending_age_seconds": 1.0,
                }
            ],
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "severity": "critical",
            "sql_ingestion_pending_overlay": {
                "active": True,
                "used_for_pressure": True,
                "source_count": 21,
                "fresh_source_count": 17,
                "stale_source_count": 4,
                "fresh_path_contains": ["shadow_crypto/", "decisions/shadow_crypto/trade_decisions_20260805.jsonl"],
                "top_pending_files": [
                    {
                        "source_rel": "decisions/shadow_crypto/trade_decisions_20260805.jsonl",
                        "shard": "crypto_trading",
                        "pending_lines": 25673,
                        "oldest_pending_age_seconds": 179.668,
                    }
                ],
            },
            "backlog_truth": {
                "authoritative_mode": "overlay_fresh_shard_level",
                "overlay_decay": {
                    "should_decay": False,
                    "attribution_ratio": 1.0,
                    "stale_source_count": 4,
                    "stale_pending_lines": 0,
                },
            },
        },
    )

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 8, 5, 12, 40, tzinfo=timezone.utc),
    )

    assert payload["overall_status"] == "ready"
    assert payload["active_drainer"]["name"] == "core_decision_drainer"
    assert payload["active_drainer"]["pending_lines"] == 25673
    assert "crypto_trading" in payload["active_drainer"]["shards"]
    assert payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_PATH_CONTAINS"].endswith(
        "trade_decisions_20260805.jsonl"
    )


def test_backpressure_drainer_fleet_prioritizes_signal_generation_core_backlog(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 425461,
            "pending_lines_total": 570530,
            "top_pending_files": [
                {
                    "source_rel": "governance/events/signal_generation_20260524.jsonl",
                    "pending_lines": 230451,
                    "oldest_pending_age_seconds": 0.0,
                },
                {
                    "source_rel": "decisions/shadow_crypto/trade_decisions_20260524.jsonl",
                    "pending_lines": 102473,
                    "oldest_pending_age_seconds": 0.0,
                },
                {
                    "source_rel": "decisions/shadow_crypto_futures_crypto/trade_decisions_20260524.jsonl",
                    "pending_lines": 91847,
                    "oldest_pending_age_seconds": 0.0,
                },
            ],
            "top_deferred_pending_files": [
                {
                    "source_rel": "governance/channels/risk/crypto_futures_crypto_schwab/risk_20260524.jsonl",
                    "pending_lines": 142659,
                    "oldest_pending_age_seconds": 3.0,
                },
            ],
        },
    )
    _write_json(health / "ingestion_storage_control_latest.json", {"severity": "critical"})

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 5, 24, 15, 0, tzinfo=timezone.utc),
    )

    assert payload["overall_status"] == "ready"
    assert payload["active_drainer"]["name"] == "core_decision_drainer"
    assert payload["active_drainer"]["pending_lines"] == 424771
    assert "governance" in payload["active_drainer"]["shards"]
    assert "crypto_trading" in payload["active_drainer"]["shards"]
    env = payload["active_env_overrides"]
    assert env["SQL_LINK_SERVICE_SHARDS"].startswith("crypto_trading,governance")
    assert "signal_generation_20260524" in env["SQL_LINK_SERVICE_SHARD_GOVERNANCE_PATH_CONTAINS"]
    assert env["SQL_LINK_SERVICE_SHARD_GOVERNANCE_MAX_LINES_PER_FILE"] == "512000"
    assert env["SQL_LINK_SERVICE_SHARD_GOVERNANCE_MERGE_MAX_JSONL_ROWS"] == "256000"


def test_backpressure_drainer_fleet_routes_tiny_stale_bridge_tails(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 95,
            "pending_lines_total": 95,
            "pending_lines_threshold": 15000,
            "oldest_age_threshold_seconds": 240.0,
            "top_pending_files": [
                {
                    "source_rel": "paper_trades_paper.jsonl",
                    "pending_lines": 48,
                    "oldest_pending_age_seconds": 611.0,
                },
                {
                    "source_rel": "exports/paper_broker_bridge/paper/paper_bridge_orders_20260524.jsonl",
                    "pending_lines": 47,
                    "oldest_pending_age_seconds": 611.0,
                },
            ],
        },
    )

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 5, 24, 16, 0, tzinfo=timezone.utc),
    )

    assert payload["overall_status"] == "ready"
    assert payload["active_drainer"]["name"] == "fast_trade_bridge_drainer"
    assert payload["active_drainer"]["readiness_reason"] == "material_pending"
    assert payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARDS"].startswith("trading_fast,")


def test_backpressure_drainer_fleet_guards_cold_stage_during_market_hours(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 10,
            "pending_lines_total": 120000,
            "pending_lines_cold": 120000,
            "top_cold_pending_files": [
                {
                    "source_rel": "data/stale_stage/decision_explanations/project/decision_explanations/a.jsonl",
                    "pending_lines": 120000,
                    "oldest_pending_age_seconds": 86400.0,
                }
            ],
        },
    )

    payload = src.build_payload(
        project_root,
        apply=True,
        now_utc=datetime(2026, 5, 1, 15, 0, tzinfo=timezone.utc),
    )

    assert payload["overall_status"] == "blocked"
    assert payload["blocked_reasons"] == ["market_hours_guard"]
    assert not (health / "sql_link_service_request_latest.json").exists()


def test_backpressure_drainer_fleet_can_force_cold_stage_handoff(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 10,
            "pending_lines_total": 120000,
            "pending_lines_cold": 120000,
            "top_cold_pending_files": [
                {
                    "source_rel": "data/stale_stage/decision_explanations/project/decision_explanations/a.jsonl",
                    "pending_lines": 120000,
                    "oldest_pending_age_seconds": 86400.0,
                }
            ],
        },
    )

    payload = src.build_payload(
        project_root,
        apply=True,
        force_live_window=True,
        now_utc=datetime(2026, 5, 1, 15, 0, tzinfo=timezone.utc),
    )

    assert payload["overall_status"] == "handoff_requested"
    assert payload["active_drainer"]["name"] == "cold_stage_drainer"
    assert payload["service_request"]["env_overrides"]["SQL_LINK_SERVICE_SHARDS"].startswith("data,")
    assert payload["service_request"]["env_overrides"]["SQL_LINK_SERVICE_SHARD_EXPLANATIONS_PATH_CONTAINS"].endswith("a.jsonl")
    assert payload["service_request"]["env_overrides"]["SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_MAX_LINES_PER_FILE"] == "64000"



def test_backpressure_drainer_fleet_splits_api_ingress_from_runtime(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines_total": 2700,
            "top_pending_files": [
                {
                    "source_rel": "governance/channels/api/default_crypto_schwab/api_20260504.jsonl",
                    "pending_lines": 900,
                    "oldest_pending_age_seconds": 600.0,
                },
                {
                    "source_rel": "governance/channels/ingress/default_equities_schwab/ingress_20260504.jsonl",
                    "pending_lines": 700,
                    "oldest_pending_age_seconds": 600.0,
                },
                {
                    "source_rel": "governance/channels/runtime/default_equities_schwab/runtime_20260504.jsonl",
                    "pending_lines": 1100,
                    "oldest_pending_age_seconds": 600.0,
                },
            ],
        },
    )

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 5, 1, 15, 0, tzinfo=timezone.utc),
    )

    assert payload["overall_status"] == "ready"
    assert payload["active_drainer"]["name"] == "api_ingress_drainer"
    assert payload["active_drainer"]["shards"][:2] == ["crypto_api_ingress", "api_ingress"]
    assert "default_crypto_schwab" in payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARD_CRYPTO_API_INGRESS_PATH_CONTAINS"]
    assert "default_equities_schwab" in payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARD_API_INGRESS_PATH_CONTAINS"]
    runtime = next(row for row in payload["candidate_drainers"] if row["name"] == "runtime_channel_drainer")
    assert runtime["pending_lines"] == 1100


def test_backpressure_drainer_fleet_routes_schema_violations_to_isolated_shard(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines_total": 600,
            "top_pending_files": [
                {
                    "source_rel": "governance/events/channel_schema_violations_20260504.jsonl",
                    "pending_lines": 600,
                    "oldest_pending_age_seconds": 1200.0,
                }
            ],
        },
    )

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 5, 1, 15, 0, tzinfo=timezone.utc),
    )

    assert payload["active_drainer"]["name"] == "schema_violation_drainer"
    assert payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARDS"] == "schema_violations,health_fast"
    assert payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARD_SCHEMA_VIOLATIONS_MAX_LINES_PER_FILE"] == "16000"


def test_backpressure_drainer_fleet_routes_overlay_schema_violations_to_isolated_shard(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 20,
            "pending_lines_total": 20,
            "top_pending_files": [],
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "sql_ingestion_pending_overlay": {
                "top_pending_files": [
                    {
                        "source_rel": "governance/events/channel_schema_violations_20260528.jsonl",
                        "pending_lines": 16000,
                        "oldest_pending_age_seconds": 600.0,
                    }
                ],
            }
        },
    )

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 5, 1, 15, 0, tzinfo=timezone.utc),
    )

    assert payload["active_drainer"]["name"] == "schema_violation_drainer"
    assert payload["active_drainer"]["pending_lines"] == 16000
    assert payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARDS"] == "schema_violations,health_fast"


def test_backpressure_drainer_fleet_queues_secondary_live_safe_drainers(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines_total": 1200,
            "top_pending_files": [
                {
                    "source_rel": "governance/channels/paper_broker_bridge/default_equities_schwab/bridge_20260504.jsonl",
                    "pending_lines": 700,
                    "oldest_pending_age_seconds": 300.0,
                },
                {
                    "source_rel": "governance/events/shadow_pnl_attribution_default_equities_20260504.jsonl",
                    "pending_lines": 500,
                    "oldest_pending_age_seconds": 300.0,
                },
            ],
        },
    )

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 5, 1, 15, 0, tzinfo=timezone.utc),
    )

    assert payload["active_drainer"]["name"] == "fast_trade_bridge_drainer"
    assert payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARDS"] == "trading_fast,health_fast"
    assert payload["next_drainer_queue"][0]["name"] == "attribution_drainer"
    assert payload["metrics"]["expanded_lane_count"] >= 9


def test_backpressure_drainer_fleet_prioritizes_old_live_safe_age_pressure(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 4993,
            "pending_lines_total": 5022,
            "oldest_age_threshold_seconds": 240,
            "top_pending_files": [
                {
                    "source_rel": "decisions/shadow_cdo_squared_equities/trade_decisions_20260506.jsonl",
                    "pending_lines": 382,
                    "oldest_pending_age_seconds": 215.0,
                },
                {
                    "source_rel": "exports/paper_broker_bridge/paper/paper_bridge_orders_20260506.jsonl",
                    "pending_lines": 189,
                    "oldest_pending_age_seconds": 1088.0,
                },
                {
                    "source_rel": "paper_trades_paper.jsonl",
                    "pending_lines": 189,
                    "oldest_pending_age_seconds": 1088.0,
                },
            ],
        },
    )

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 5, 6, 23, 0, tzinfo=timezone.utc),
    )

    assert payload["active_drainer"]["name"] == "fast_trade_bridge_drainer"
    assert payload["active_drainer"]["age_pressure_priority_bonus"] > 0
    assert payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARDS"] == "trading_fast,health_fast"
    assert "paper_trades_paper.jsonl" in payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARD_TRADING_FAST_PATH_CONTAINS"]


def test_backpressure_drainer_fleet_routes_stale_provider_tails_even_when_tiny(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 245,
            "pending_lines_total": 245,
            "top_pending_files": [
                {
                    "source_rel": "decisions/shadow_conservative_equities/trade_decisions_20260505.jsonl",
                    "pending_lines": 237,
                    "oldest_pending_age_seconds": 9289.538,
                },
                {
                    "source_rel": "data/tradingeconomics/tradingeconomics_guest_rows_20260502.jsonl",
                    "pending_lines": 4,
                    "oldest_pending_age_seconds": 226918.124,
                },
                {
                    "source_rel": "data/tradingeconomics/tradingeconomics_guest_rows_20260503.jsonl",
                    "pending_lines": 4,
                    "oldest_pending_age_seconds": 140032.872,
                },
            ],
        },
    )
    _write_json(health / "ingestion_storage_control_latest.json", {"severity": "critical"})

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 5, 5, 15, 0, tzinfo=timezone.utc),
    )

    assert payload["active_drainer"]["name"] == "market_data_provider_drainer"
    assert payload["active_drainer"]["readiness_reason"] == "stale_tail"
    assert payload["active_drainer"]["pending_lines"] == 8
    assert payload["active_drainer"]["shards"] == ["data", "governance", "health_fast"]
    assert "tradingeconomics_guest_rows_20260502" in payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARD_DATA_PATH_CONTAINS"]
    assert payload["next_drainer_queue"][0]["name"] == "core_decision_drainer"
    assert payload["metrics"]["stale_tail_ready_count"] == 1
    assert payload["metrics"]["expanded_lane_count"] >= 15


def test_backpressure_drainer_fleet_prioritizes_source_attributed_stale_decision_logs(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 1200,
            "pending_lines_total": 1200,
            "oldest_age_threshold_seconds": 240,
            "top_pending_files": [
                {
                    "source_rel": "governance/channels/decision/swaptions_equities_schwab/decision_20260520.jsonl",
                    "pending_lines": 33,
                    "oldest_pending_age_seconds": 15896.0,
                }
            ],
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "severity": "critical",
            "stale_pending_locator": {
                "status": "attributed",
                "oldest_sources": [
                    {
                        "source_rel": "decisions/shadow_aggressive_equities/trade_decisions_20260515.jsonl",
                        "pending_lines": 19902,
                        "oldest_pending_age_seconds": 111456.0,
                    },
                    {
                        "source_rel": "decisions/shadow_swing_aggressive_equities/trade_decisions_20260515.jsonl",
                        "pending_lines": 24038,
                        "oldest_pending_age_seconds": 111437.0,
                    },
                    {
                        "source_rel": "decisions/shadow_conservative_equities/trade_decisions_20260515.jsonl",
                        "pending_lines": 6615,
                        "oldest_pending_age_seconds": 111445.0,
                    },
                ],
            },
        },
    )

    payload = src.build_payload(project_root, apply=False, now_utc=datetime(2026, 5, 21, 1, 0, tzinfo=timezone.utc))

    assert payload["active_drainer"]["name"] == "stale_decision_log_drainer"
    assert payload["active_drainer"]["pending_lines"] == 50555
    env = payload["active_env_overrides"]
    assert env["SQL_LINK_SERVICE_STALE_DECISION_SOURCE_CATCH_UP"] == "1"
    assert env["SQL_LINK_SERVICE_SHARD_LINK_TIMEOUT_SECONDS"] == "420"
    assert env["INGEST_MAX_BYTES_PER_FILE"] == str(1024 * 1024 * 1024)
    assert env["SQLITE_BATCH_MAX_BYTES"] == str(256 * 1024 * 1024)
    assert env["SQL_LINK_SERVICE_SHARD_TRADING_MAX_BYTES_PER_FILE"] == str(1024 * 1024 * 1024)
    assert env["SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_MAX_BYTES_PER_FILE"] == str(1024 * 1024 * 1024)
    assert env["SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_MAX_BYTES_PER_FILE"] == str(1024 * 1024 * 1024)
    assert env["SQL_LINK_SERVICE_SHARD_TRADING_SQLITE_BATCH_MAX_BYTES"] == str(256 * 1024 * 1024)
    assert env["SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_SQLITE_BATCH_MAX_BYTES"] == str(256 * 1024 * 1024)
    assert env["SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_SQLITE_BATCH_MAX_BYTES"] == str(256 * 1024 * 1024)
    assert "shadow_aggressive_equities" in env["SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_PATH_CONTAINS"]
    assert "shadow_conservative_equities" in env["SQL_LINK_SERVICE_SHARD_TRADING_PATH_CONTAINS"]
    assert payload["next_drainer_queue"][0]["name"] == "core_decision_drainer"


def test_backpressure_drainer_fleet_honors_source_attributed_shard_over_filename_family(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    source_rel = "governance/channels/decision/intraday_aggressive_equities_schwab/decision_20260803.jsonl"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 600,
            "pending_lines_total": 600,
            "oldest_age_threshold_seconds": 240,
            "top_pending_files": [],
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "severity": "critical",
            "stale_pending_locator": {
                "status": "attributed",
                "oldest_sources": [
                    {
                        "source_rel": source_rel,
                        "shard": "trading",
                        "pending_lines": 600,
                        "oldest_pending_age_seconds": 1300.0,
                    }
                ],
            },
        },
    )

    payload = src.build_payload(project_root, apply=False, now_utc=datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc))

    assert payload["active_drainer"]["name"] == "stale_decision_log_drainer"
    assert payload["active_drainer"]["shards"][0] == "trading"
    env = payload["active_env_overrides"]
    assert source_rel in env["SQL_LINK_SERVICE_SHARD_TRADING_PATH_CONTAINS"]
    assert "SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_PATH_CONTAINS" not in env
    assert env["SQL_LINK_SERVICE_SHARD_TRADING_MAX_BYTES_PER_FILE"] == str(1024 * 1024 * 1024)


def test_backpressure_drainer_fleet_keeps_stale_crypto_path_in_decision_handoff(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 6200,
            "pending_lines_total": 6200,
            "oldest_age_threshold_seconds": 240,
            "top_pending_files": [
                {
                    "source_rel": "governance/channels/decision/crypto_futures_crypto_schwab/decision_20260522.jsonl",
                    "pending_lines": 5200,
                    "oldest_pending_age_seconds": 20.0,
                },
                {
                    "source_rel": "governance/channels/decision/crypto_futures_crypto_schwab/decision_20260521.jsonl",
                    "pending_lines": 1308,
                    "oldest_pending_age_seconds": 525.0,
                }
            ],
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "severity": "high",
            "stale_pending_locator": {
                "status": "clear",
                "oldest_sources": [],
            },
        },
    )

    payload = src.build_payload(project_root, apply=False, now_utc=datetime(2026, 5, 22, 1, 0, tzinfo=timezone.utc))

    env = payload["active_env_overrides"]
    assert env["SQL_LINK_SERVICE_STALE_DECISION_SOURCE_CATCH_UP"] == "1"
    assert "decision_20260521.jsonl" in env["SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_PATH_CONTAINS"]
    assert "crypto_trading" in payload["active_drainer"]["shards"]


def test_backpressure_drainer_fleet_routes_mixed_stale_decision_sleeves_to_matching_shards(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 650,
            "pending_lines_total": 650,
            "oldest_age_threshold_seconds": 240,
            "top_pending_files": [
                {
                    "source_rel": "governance/channels/decision/crypto_futures_crypto_schwab/decision_20260730.jsonl",
                    "pending_lines": 68,
                    "oldest_pending_age_seconds": 1265.0,
                },
                {
                    "source_rel": "governance/channels/decision/swing_aggressive_equities_schwab/decision_20260730.jsonl",
                    "pending_lines": 58,
                    "oldest_pending_age_seconds": 1322.0,
                },
                {
                    "source_rel": "decisions/shadow_swing_aggressive_equities/trade_decisions_20260730.jsonl",
                    "pending_lines": 56,
                    "oldest_pending_age_seconds": 1322.0,
                },
                {
                    "source_rel": "decisions/shadow_conservative_equities/trade_decisions_20260730.jsonl",
                    "pending_lines": 52,
                    "oldest_pending_age_seconds": 1271.0,
                },
                {
                    "source_rel": "decisions/shadow_aggressive_equities/trade_decisions_20260730.jsonl",
                    "pending_lines": 51,
                    "oldest_pending_age_seconds": 1272.0,
                },
                {
                    "source_rel": "governance/channels/decision/conservative_equities_schwab/decision_20260730.jsonl",
                    "pending_lines": 33,
                    "oldest_pending_age_seconds": 1271.0,
                },
                {
                    "source_rel": "decisions/shadow_bond_equities/trade_decisions_20260730.jsonl",
                    "pending_lines": 33,
                    "oldest_pending_age_seconds": 1253.0,
                },
                {
                    "source_rel": "governance/channels/decision/dividend_capture_equities_schwab/decision_20260730.jsonl",
                    "pending_lines": 32,
                    "oldest_pending_age_seconds": 2055.0,
                },
            ],
        },
    )
    _write_json(health / "ingestion_storage_control_latest.json", {"severity": "elevated"})

    payload = src.build_payload(project_root, apply=False, now_utc=datetime(2026, 7, 30, 23, 58, tzinfo=timezone.utc))

    assert payload["active_drainer"]["name"] == "core_decision_drainer"
    env = payload["active_env_overrides"]
    assert "crypto_trading" in payload["active_drainer"]["shards"]
    assert "aggressive_trading" in payload["active_drainer"]["shards"]
    assert "trading" in payload["active_drainer"]["shards"]
    assert "crypto_futures_crypto_schwab" in env["SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_PATH_CONTAINS"]
    assert "swing_aggressive_equities_schwab" in env["SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_PATH_CONTAINS"]
    assert "shadow_swing_aggressive_equities" in env["SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_PATH_CONTAINS"]
    assert "shadow_aggressive_equities" in env["SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_PATH_CONTAINS"]
    assert "shadow_conservative_equities" in env["SQL_LINK_SERVICE_SHARD_TRADING_PATH_CONTAINS"]
    assert "shadow_bond_equities" in env["SQL_LINK_SERVICE_SHARD_TRADING_PATH_CONTAINS"]
    assert "dividend_capture_equities_schwab" in env["SQL_LINK_SERVICE_SHARD_TRADING_PATH_CONTAINS"]


def test_backpressure_drainer_fleet_routes_derivatives_to_focused_trading_shards(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines_total": 1700,
            "top_pending_files": [
                {
                    "source_rel": "decisions/shadow_options_on_futures_aggressive/trade_decisions_20260505.jsonl",
                    "pending_lines": 900,
                    "oldest_pending_age_seconds": 600.0,
                },
                {
                    "source_rel": "governance/quant_models/model_risk_validation/retrain_surface_20260505.jsonl",
                    "pending_lines": 800,
                    "oldest_pending_age_seconds": 900.0,
                },
            ],
        },
    )

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 5, 5, 15, 0, tzinfo=timezone.utc),
    )

    assert payload["active_drainer"]["name"] == "derivatives_surface_drainer"
    assert "trading" in payload["active_drainer"]["shards"]
    assert "options_on_futures" in payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARD_TRADING_PATH_CONTAINS"]
    model = next(row for row in payload["candidate_drainers"] if row["name"] == "model_research_drainer")
    assert model["status"] == "ready"
    assert model["assigned_pressure_lane"] == "model_retrain_research_backpressure"


def test_backpressure_drainer_fleet_routes_derivative_explainers_to_explanation_shards(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 12,
            "pending_lines_total": 9841,
            "pending_lines_deferred": 9829,
            "top_pending_files": [],
            "top_deferred_pending_files": [
                {
                    "source_rel": "decision_explanations/shadow_compound_options_equities/decision_explanations_20260624.jsonl",
                    "pending_lines": 2741,
                    "oldest_pending_age_seconds": 53453.502,
                },
                {
                    "source_rel": "decision_explanations/shadow_synthetic_cdo_equities/decision_explanations_20260624.jsonl",
                    "pending_lines": 2650,
                    "oldest_pending_age_seconds": 53444.388,
                },
            ],
        },
    )
    _write_json(health / "ingestion_storage_control_latest.json", {"severity": "critical"})

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 6, 25, 15, 0, tzinfo=timezone.utc),
    )

    assert payload["active_drainer"]["name"] == "derivatives_surface_drainer"
    assert "explanations" in payload["active_drainer"]["shards"]
    assert "trading" not in payload["active_drainer"]["shards"]
    env = payload["active_env_overrides"]
    assert env["SQL_LINK_SERVICE_SHARDS"] == "explanations,health_fast"
    assert "shadow_compound_options" in env["SQL_LINK_SERVICE_SHARD_EXPLANATIONS_PATH_CONTAINS"]
    assert env["SQL_LINK_SERVICE_SHARD_EXPLANATIONS_MAX_LINES_PER_FILE"] == "64000"


def test_backpressure_drainer_fleet_keeps_futures_loop_state_out_of_derivatives_drainer(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 10,
            "pending_lines_total": 930,
            "pending_lines_deferred": 920,
            "top_deferred_pending_files": [
                {
                    "source_rel": "governance/channels/loop_state/futures_commodity_macro_equities_schwab/loop_state_20260625.jsonl",
                    "pending_lines": 920,
                    "oldest_pending_age_seconds": 51.148,
                }
            ],
        },
    )
    _write_json(health / "ingestion_storage_control_latest.json", {"severity": "critical"})

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 6, 25, 15, 0, tzinfo=timezone.utc),
    )

    derivatives = next(row for row in payload["candidate_drainers"] if row["name"] == "derivatives_surface_drainer")
    assert derivatives["status"] == "idle"
    assert payload["active_drainer"]["name"] == "runtime_channel_drainer"
    assert "futures_commodity_macro" in payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARD_RUNTIME_PATH_CONTAINS"]
    assert payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARD_RUNTIME_MAX_LINES_PER_FILE"] == "24000"
    assert payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARD_RUNTIME_MAX_FILES"] == "8"
    assert payload["active_env_overrides"]["SQL_LINK_SERVICE_SKIP_FRESH_IDLE_SHARDS"] == "0"
    assert payload["active_env_overrides"]["SQL_LINK_SERVICE_IDLE_SHARD_MAX_AGE_SECONDS"] == "0"


def test_backpressure_drainer_fleet_keeps_fx_loop_state_out_of_provider_drainer(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines": 10,
            "pending_lines_total": 970,
            "pending_lines_deferred": 960,
            "top_deferred_pending_files": [
                {
                    "source_rel": "governance/channels/loop_state/fx_equities_schwab/loop_state_20260625.jsonl",
                    "pending_lines": 960,
                    "oldest_pending_age_seconds": 55.153,
                }
            ],
        },
    )
    _write_json(health / "ingestion_storage_control_latest.json", {"severity": "critical"})

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 6, 25, 15, 0, tzinfo=timezone.utc),
    )

    provider = next(row for row in payload["candidate_drainers"] if row["name"] == "market_data_provider_drainer")
    assert provider["status"] == "idle"
    assert payload["active_drainer"]["name"] == "runtime_channel_drainer"
    assert "fx_equities_schwab" in payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARD_RUNTIME_PATH_CONTAINS"]


def test_backpressure_drainer_fleet_keeps_report_cockpit_drainer_protected(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines_total": 900,
            "top_deferred_pending_files": [
                {
                    "source_rel": "governance/reports/operator_cockpit/report_ready_packets_20260505.jsonl",
                    "pending_lines": 900,
                    "oldest_pending_age_seconds": 12000.0,
                }
            ],
        },
    )

    payload = src.build_payload(
        project_root,
        apply=True,
        now_utc=datetime(2026, 5, 5, 15, 0, tzinfo=timezone.utc),
    )

    assert payload["active_drainer"]["name"] == "report_cockpit_drainer"
    assert payload["active_drainer"]["live_window_safe"] is False
    assert payload["overall_status"] == "blocked"
    assert payload["blocked_reasons"] == ["market_hours_guard"]

    forced = src.build_payload(
        project_root,
        apply=True,
        force_live_window=True,
        now_utc=datetime(2026, 5, 5, 15, 0, tzinfo=timezone.utc),
    )
    assert forced["overall_status"] == "handoff_requested"
    assert forced["service_request"]["assigned_pressure_lane"] == "report_cockpit_backpressure"


def test_backpressure_drainer_fleet_routes_settlement_reconciliation_to_dedicated_lane(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines_total": 1600,
            "top_pending_files": [
                {
                    "source_rel": "governance/reconciliation/positions/fills_20260506.jsonl",
                    "pending_lines": 1600,
                    "oldest_pending_age_seconds": 1800.0,
                }
            ],
        },
    )

    payload = src.build_payload(
        project_root,
        apply=True,
        now_utc=datetime(2026, 5, 6, 15, 0, tzinfo=timezone.utc),
    )

    assert payload["overall_status"] == "handoff_requested"
    assert payload["active_drainer"]["name"] == "settlement_reconciliation_drainer"
    assert payload["active_drainer"]["assigned_pressure_lane"] == "settlement_reconciliation_backpressure"
    assert payload["active_drainer"]["shards"] == ["governance", "trading_fast", "health_fast"]
    assert payload["active_drainer"]["self_accommodation"]["allowed_parallel_writers"] == 1
    assert payload["active_drainer"]["self_accommodation"]["starts_parallel_sql_writers"] is False
    assert payload["service_request"]["self_accommodation"]["coordination_model"] == "single_sql_writer_focused_handoff"
    assert payload["self_accommodation"]["next_safe_action"] == "single_writer_handoff_requested"
    assert "positions/fills_20260506" in payload["active_env_overrides"]["SQL_LINK_SERVICE_SHARD_GOVERNANCE_PATH_CONTAINS"]


def test_backpressure_drainer_fleet_exposes_self_accommodation_contracts_for_new_lanes(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines_total": 1800,
            "top_pending_files": [
                {
                    "source_rel": "governance/health/runtime_pressure/memory_efficiency_20260506.jsonl",
                    "pending_lines": 1000,
                    "oldest_pending_age_seconds": 2400.0,
                },
                {
                    "source_rel": "governance/data_quality/source_verification/provider_adapter_20260506.jsonl",
                    "pending_lines": 800,
                    "oldest_pending_age_seconds": 2400.0,
                },
            ],
        },
    )

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 5, 6, 15, 0, tzinfo=timezone.utc),
    )

    assert payload["overall_status"] == "ready"
    assert payload["metrics"]["expanded_lane_count"] >= 19
    assert payload["metrics"]["self_accommodating_lane_count"] == payload["metrics"]["expanded_lane_count"]
    assert payload["self_accommodation"]["allowed_parallel_writers"] == 1
    assert payload["self_accommodation"]["next_safe_action"] == "run_backpressure_drainer_fleet_apply_or_bounded_super_drainer_wave"
    memory_lane = next(row for row in payload["candidate_drainers"] if row["name"] == "memory_runtime_artifact_drainer")
    data_quality_lane = next(row for row in payload["candidate_drainers"] if row["name"] == "data_quality_contract_drainer")
    assert memory_lane["status"] == "ready"
    assert data_quality_lane["status"] == "ready"
    assert data_quality_lane["self_accommodation"]["safe_expansion_rule"].startswith("sequence_bounded_handoffs")


def test_backpressure_drainer_fleet_routes_predictive_and_self_healing_lanes(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines_total": 2700,
            "top_pending_files": [
                {
                    "source_rel": "governance/predictive_stability/pressure_trajectory_20260506.jsonl",
                    "pending_lines": 1400,
                    "oldest_pending_age_seconds": 2200.0,
                },
                {
                    "source_rel": "governance/self_healing/blocked_surface_recovery_plan_20260506.jsonl",
                    "pending_lines": 1300,
                    "oldest_pending_age_seconds": 2100.0,
                },
            ],
        },
    )

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 5, 6, 15, 0, tzinfo=timezone.utc),
    )

    assert payload["overall_status"] == "ready"
    assert payload["active_drainer"]["name"] == "predictive_stability_drainer"
    assert payload["active_drainer"]["assigned_pressure_lane"] == "predictive_stability_backpressure"
    assert payload["next_drainer_queue"][0]["name"] == "self_healing_recovery_drainer"
    assert payload["metrics"]["expanded_lane_count"] >= 25


def test_backpressure_drainer_fleet_routes_admission_and_writer_recovery_lanes(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines_total": 2000,
            "top_pending_files": [
                {
                    "source_rel": "governance/admission_evidence/new_bot_admission/sample_depth_20260506.jsonl",
                    "pending_lines": 1200,
                    "oldest_pending_age_seconds": 1900.0,
                },
                {
                    "source_rel": "governance/writer_progress/jsonl_sql_writer/merge_progress_20260506.jsonl",
                    "pending_lines": 800,
                    "oldest_pending_age_seconds": 950.0,
                },
            ],
        },
    )

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 5, 6, 15, 0, tzinfo=timezone.utc),
    )

    assert payload["active_drainer"]["name"] == "admission_evidence_drainer"
    writer_lane = next(row for row in payload["candidate_drainers"] if row["name"] == "writer_progress_recovery_drainer")
    assert writer_lane["status"] == "ready"
    assert writer_lane["assigned_pressure_lane"] == "writer_progress_recovery_backpressure"


def test_backpressure_drainer_fleet_routes_training_collection_storage_and_ingestion_lanes(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    health = project_root / "governance" / "health"
    _write_json(
        health / "ingestion_backpressure_latest.json",
        {
            "pending_lines_total": 6200,
            "top_pending_files": [
                {
                    "source_rel": "governance/training_lineage/lineage_manifest_20260506.jsonl",
                    "pending_lines": 1600,
                    "oldest_pending_age_seconds": 2200.0,
                },
                {
                    "source_rel": "governance/training_labeling_intelligence/label_coverage_20260506.jsonl",
                    "pending_lines": 1300,
                    "oldest_pending_age_seconds": 2100.0,
                },
                {
                    "source_rel": "governance/collector_telemetry/observation_rollup_20260506.jsonl",
                    "pending_lines": 1200,
                    "oldest_pending_age_seconds": 1900.0,
                },
                {
                    "source_rel": "governance/storage_route/split_brain_reconcile_20260506.jsonl",
                    "pending_lines": 1100,
                    "oldest_pending_age_seconds": 1800.0,
                },
                {
                    "source_rel": "governance/ingestion_priority/backlog_quarantine_20260506.jsonl",
                    "pending_lines": 1000,
                    "oldest_pending_age_seconds": 1700.0,
                },
            ],
        },
    )

    payload = src.build_payload(
        project_root,
        apply=False,
        now_utc=datetime(2026, 5, 6, 15, 0, tzinfo=timezone.utc),
    )

    assert payload["active_drainer"]["name"] == "training_lineage_drainer"
    assert payload["active_drainer"]["assigned_pressure_lane"] == "training_lineage_backpressure"
    lanes = {row["name"]: row for row in payload["candidate_drainers"]}
    assert lanes["label_contract_drainer"]["status"] == "ready"
    assert lanes["collector_telemetry_rollup_drainer"]["status"] == "ready"
    assert lanes["storage_route_reconcile_drainer"]["status"] == "ready"
    assert lanes["ingestion_priority_drainer"]["status"] == "ready"
    assert lanes["collector_telemetry_rollup_drainer"]["self_accommodation"]["allowed_parallel_writers"] == 1
    assert payload["metrics"]["expanded_lane_count"] >= 30
