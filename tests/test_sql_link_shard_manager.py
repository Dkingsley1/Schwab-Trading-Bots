import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.ops.sql_link_shard_manager as shard_manager


def test_retention_maintenance_pause_reads_live_swap_override(tmp_path, monkeypatch) -> None:
    override = tmp_path / ".env.swap_pressure_override"
    override.write_text(
        "\n".join(
            [
                "SWAP_PRESSURE_TIER=pause_research",
                "SWAP_PRESSURE_SWAP_USED_GB=19.1",
                "RETENTION_MAINTENANCE_PAUSED_FOR_SWAP=1",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("RETENTION_MAINTENANCE_PAUSED_FOR_SWAP", raising=False)
    paused, env = shard_manager._retention_maintenance_paused_for_swap(override_path=override)
    assert paused is True
    assert env["SWAP_PRESSURE_TIER"] == "pause_research"


def test_retention_maintenance_pause_can_clear_with_normal_override(tmp_path, monkeypatch) -> None:
    override = tmp_path / ".env.swap_pressure_override"
    override.write_text(
        "\n".join(
            [
                "SWAP_PRESSURE_TIER=normal",
                "RETENTION_MAINTENANCE_PAUSED_FOR_SWAP=0",
                "SWAP_PRESSURE_HEAVY_RESEARCH_PAUSED=0",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("RETENTION_MAINTENANCE_PAUSED_FOR_SWAP", "1")
    paused, env = shard_manager._retention_maintenance_paused_for_swap(override_path=override)
    assert paused is False
    assert env["RETENTION_MAINTENANCE_PAUSED_FOR_SWAP"] == "0"


def _create_shard_jsonl_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE jsonl_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT NOT NULL,
            source_rel TEXT NOT NULL,
            line_no INTEGER NOT NULL,
            ingested_at TEXT NOT NULL,
            payload_sha1 TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            run_id TEXT,
            iter_id TEXT,
            decision_id TEXT,
            parent_decision_id TEXT,
            log_schema_version INTEGER,
            UNIQUE(source_file, line_no)
        )
        """
    )
    conn.execute(
        """
        INSERT INTO jsonl_records (
            source_file, source_rel, line_no, ingested_at, payload_sha1, payload_json,
            run_id, iter_id, decision_id, parent_decision_id, log_schema_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "a.jsonl",
            "governance/events/a.jsonl",
            1,
            "2026-03-26T15:00:00+00:00",
            "sha1-a",
            "{}",
            "run-a",
            "iter-a",
            "decision-a",
            "",
            2,
        ),
    )
    conn.execute(
        """
        INSERT INTO jsonl_records (
            source_file, source_rel, line_no, ingested_at, payload_sha1, payload_json,
            run_id, iter_id, decision_id, parent_decision_id, log_schema_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "b.jsonl",
            "governance/events/b.jsonl",
            1,
            "2026-03-26T15:01:00+00:00",
            "sha1-b",
            "{}",
            "run-b",
            "iter-b",
            "decision-b",
            "",
            2,
        ),
    )
    conn.commit()
    conn.close()


def test_quarantine_shard_artifacts_moves_corrupt_db_and_state(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(shard_manager, "SHARD_DB_ROOT", tmp_path / "sql_link_shards")

    sqlite_db = tmp_path / "jsonl_link_governance.sqlite3"
    sqlite_db.write_bytes(b"not a sqlite database")
    state_file = tmp_path / "jsonl_sql_link_state_governance.json"
    state_file.write_text("{}", encoding="utf-8")
    health_file = tmp_path / "jsonl_sql_ingestion_health_governance_latest.json"
    health_file.write_text("{}", encoding="utf-8")

    recovery = shard_manager._quarantine_shard_artifacts(
        shard_name="governance",
        sqlite_db=sqlite_db,
        state_file=state_file,
        health_file=health_file,
    )

    assert recovery["triggered"] is True
    assert sqlite_db.exists() is False
    assert state_file.exists() is False
    assert health_file.exists() is False
    assert len(recovery["moved_paths"]) == 2
    assert Path(str(recovery["quarantine_root"])).exists()


def test_merge_shard_into_primary_resets_cursor_after_rebuild(tmp_path) -> None:
    primary_db = tmp_path / "primary.sqlite3"
    shard_db = tmp_path / "governance.sqlite3"

    conn = sqlite3.connect(str(primary_db))
    shard_manager._ensure_primary_schema(conn)
    conn.execute(
        """
        INSERT INTO shard_merge_state (shard_name, last_jsonl_id, last_json_file_id, updated_at)
        VALUES (?, ?, ?, ?)
        """,
        ("governance", 50, 9, "2026-03-26T15:00:00+00:00"),
    )
    conn.commit()
    conn.close()

    _create_shard_jsonl_db(shard_db)

    result = shard_manager._merge_shard_into_primary(
        shard_name="governance",
        shard_db=shard_db,
        primary_db=primary_db,
        sqlite_timeout_seconds=30,
    )

    assert result["ok"] is True
    assert result["jsonl_cursor_reset"] is True
    assert result["jsonl_rows_inserted"] == 2
    assert result["last_jsonl_id"] == 2

    conn = sqlite3.connect(str(primary_db))
    rows = conn.execute("SELECT COUNT(*) FROM jsonl_records").fetchone()[0]
    cursor = conn.execute(
        "SELECT last_jsonl_id FROM shard_merge_state WHERE shard_name = ?",
        ("governance",),
    ).fetchone()[0]
    conn.close()

    assert rows == 2
    assert cursor == 2


def test_quarantine_shard_artifacts_uses_light_probe_when_recent_integrity_marker_exists(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(shard_manager, "SHARD_DB_ROOT", tmp_path / "sql_link_shards")
    monkeypatch.setattr(shard_manager, "INTEGRITY_MARKER_ROOT", tmp_path / "health" / "sql_link_integrity")

    sqlite_db = tmp_path / "jsonl_link_trading.sqlite3"
    conn = sqlite3.connect(str(sqlite_db))
    conn.execute("CREATE TABLE jsonl_records (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()

    marker_path = shard_manager._integrity_marker_path("trading")
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(
        (
            '{'
            f'"checked_at_epoch": {time.time()}, '
            '"ok": true'
            '}'
        ),
        encoding="utf-8",
    )

    probe_modes: list[bool] = []

    def fake_integrity(path: Path, *, deep: bool) -> tuple[bool, str]:
        probe_modes.append(bool(deep))
        return True, "opened"

    monkeypatch.setattr(shard_manager, "_sqlite_integrity_status", fake_integrity)

    recovery = shard_manager._quarantine_shard_artifacts(
        shard_name="trading",
        sqlite_db=sqlite_db,
        state_file=tmp_path / "jsonl_sql_link_state_trading.json",
        health_file=tmp_path / "jsonl_sql_ingestion_health_trading_latest.json",
    )

    assert recovery["triggered"] is False
    assert recovery["integrity_probe_mode"] == "open_probe"
    assert probe_modes == [False]


def test_quarantine_shard_artifacts_skips_recent_probe_for_large_db(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(shard_manager, "SHARD_DB_ROOT", tmp_path / "sql_link_shards")
    monkeypatch.setattr(shard_manager, "INTEGRITY_MARKER_ROOT", tmp_path / "health" / "sql_link_integrity")

    sqlite_db = tmp_path / "jsonl_link_trading.sqlite3"
    conn = sqlite3.connect(str(sqlite_db))
    conn.execute("CREATE TABLE jsonl_records (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()

    marker_path = shard_manager._integrity_marker_path("trading")
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(
        (
            "{"
            f'"checked_at_epoch": {time.time()}, '
            '"ok": true'
            "}"
        ),
        encoding="utf-8",
    )

    probe_modes: list[bool] = []

    def fake_integrity(path: Path, *, deep: bool) -> tuple[bool, str]:
        probe_modes.append(bool(deep))
        return True, "opened"

    monkeypatch.setattr(shard_manager, "_sqlite_integrity_status", fake_integrity)
    monkeypatch.setattr(shard_manager, "_db_size_gb", lambda path: 35.0)

    recovery = shard_manager._quarantine_shard_artifacts(
        shard_name="trading",
        sqlite_db=sqlite_db,
        state_file=tmp_path / "jsonl_sql_link_state_trading.json",
        health_file=tmp_path / "jsonl_sql_ingestion_health_trading_latest.json",
    )

    assert recovery["triggered"] is False
    assert recovery["integrity_probe_mode"] == "recent_marker_skip"
    assert recovery["reason"] == "recent_ok_marker_skip"
    assert probe_modes == []


def test_quarantine_shard_artifacts_uses_light_probe_for_large_db_without_marker(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(shard_manager, "SHARD_DB_ROOT", tmp_path / "sql_link_shards")
    monkeypatch.setattr(shard_manager, "INTEGRITY_MARKER_ROOT", tmp_path / "health" / "sql_link_integrity")

    sqlite_db = tmp_path / "jsonl_link_crypto_trading.sqlite3"
    conn = sqlite3.connect(str(sqlite_db))
    conn.execute("CREATE TABLE jsonl_records (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()

    probe_modes: list[bool] = []

    def fake_integrity(path: Path, *, deep: bool) -> tuple[bool, str]:
        probe_modes.append(bool(deep))
        return True, "opened"

    monkeypatch.setattr(shard_manager, "_sqlite_integrity_status", fake_integrity)
    monkeypatch.setattr(shard_manager, "_db_size_gb", lambda path: 5.0)

    recovery = shard_manager._quarantine_shard_artifacts(
        shard_name="crypto_trading",
        sqlite_db=sqlite_db,
        state_file=tmp_path / "jsonl_sql_link_state_crypto_trading.json",
        health_file=tmp_path / "jsonl_sql_ingestion_health_crypto_trading_latest.json",
    )

    assert recovery["triggered"] is False


def test_configured_primary_db_path_preserves_routed_symlink_path(tmp_path, monkeypatch) -> None:
    routed_primary = tmp_path / "data" / "jsonl_link.sqlite3"
    fallback_primary = tmp_path / "local_fallback_storage" / "data" / "jsonl_link.sqlite3"
    fallback_primary.parent.mkdir(parents=True, exist_ok=True)
    fallback_primary.write_bytes(b"db")
    routed_primary.parent.mkdir(parents=True, exist_ok=True)
    routed_primary.symlink_to(fallback_primary)
    monkeypatch.setattr(shard_manager, "PRIMARY_DB_PATH", routed_primary)

    configured = shard_manager._configured_primary_db_path(str(routed_primary))

    assert configured == routed_primary
    assert configured.resolve(strict=False) == fallback_primary.resolve(strict=False)
    assert shard_manager._primary_db_role(configured, configured.resolve(strict=False)) == "compatibility_cache"


def test_normalized_shard_config_upgrades_old_default_layouts() -> None:
    assert shard_manager._normalized_shard_config("") == shard_manager.CURRENT_DEFAULT_SHARDS
    assert shard_manager._normalized_shard_config(shard_manager.LEGACY_DEFAULT_SHARDS) == shard_manager.CURRENT_DEFAULT_SHARDS
    assert shard_manager._normalized_shard_config(shard_manager.PRE_FAST_DEFAULT_SHARDS) == shard_manager.CURRENT_DEFAULT_SHARDS
    assert shard_manager._normalized_shard_config(shard_manager.PRE_BACKLOG_SPLIT_DEFAULT_SHARDS) == shard_manager.CURRENT_DEFAULT_SHARDS


def test_build_shards_separates_fast_trading_streams() -> None:
    shards = {
        row["name"]: row
        for row in shard_manager._build_shards(
                [
                    "health_fast",
                    "writer_progress",
                    "trading_fast",
                    "runtime",
                    "predictive_stability",
                    "self_healing",
                    "hot_path_storage",
                    "crypto_api_ingress",
                    "aggressive_trading",
                    "trading",
                    "crypto_runtime",
                    "crypto_trading_fast",
                    "crypto_trading",
                    "governance",
                    "support_watchdog",
                    "schema_violations",
                    "collector_utility",
                    "admission_evidence",
                    "crypto_explanations",
                    "explanations",
                    "reports",
                    "shadow_attribution",
                    "crypto_shadow_attribution",
                    "data",
                ]
        )
    }

    assert shards["health_fast"]["skip_json_files"] is False
    assert "data_ingress_latest_" in str(shards["health_fast"]["path_contains"])
    assert shards["health_fast"]["state_checkpoint_lines"] == 500
    assert shards["crypto_runtime"]["include_streams"] == "governance"
    assert "default_crypto_schwab" in str(shards["crypto_runtime"]["path_contains"])
    assert shards["crypto_runtime"]["max_lines_per_file"] == 12000
    assert shards["crypto_runtime"]["state_checkpoint_lines"] == 2000
    assert shards["crypto_runtime"]["merge_max_jsonl_rows"] == 8000
    assert shards["crypto_trading_fast"]["include_streams"] == "paper_broker_bridge,top_level_trade_links"
    assert shards["crypto_explanations"]["include_streams"] == "decision_explanations"
    assert shards["crypto_explanations"]["merge_hot_days"] == 3
    assert shards["crypto_explanations"]["merge_priority"] == "low"
    assert "shadow_pnl_attribution_" in str(shards["crypto_shadow_attribution"]["path_contains"])
    assert shards["crypto_shadow_attribution"]["merge_to_primary"] is False
    assert shards["runtime"]["include_streams"] == "governance"
    assert "governance/channels/runtime/" in str(shards["runtime"]["path_contains"])
    assert shards["runtime"]["max_lines_per_file"] == 12000
    assert shards["runtime"]["state_checkpoint_lines"] == 2000
    assert shards["runtime"]["merge_max_jsonl_rows"] == 8000
    assert shards["crypto_api_ingress"]["include_streams"] == "governance"
    assert "default_crypto_schwab" in str(shards["crypto_api_ingress"]["path_contains"])
    assert shards["crypto_api_ingress"]["merge_to_primary"] is False
    assert shards["schema_violations"]["include_streams"] == "schema_violations"
    assert "channel_schema_violations_" in str(shards["schema_violations"]["path_contains"])
    assert shards["schema_violations"]["merge_to_primary"] is False
    assert shards["writer_progress"]["skip_json_files"] is False
    assert "writer_cycle_coordinator_" in str(shards["writer_progress"]["path_contains"])
    assert shards["writer_progress"]["merge_max_json_file_rows"] == 96
    assert shards["predictive_stability"]["skip_json_files"] is False
    assert "pressure_trajectory" in str(shards["predictive_stability"]["path_contains"])
    assert shards["self_healing"]["include_streams"] == "governance,governance_events,governance_watchdog"
    assert "blackstart" in str(shards["self_healing"]["path_contains"])
    assert shards["collector_utility"]["merge_to_primary"] is False
    assert "collector_budget" in str(shards["collector_utility"]["path_contains"])
    assert shards["hot_path_storage"]["merge_max_jsonl_rows"] == 3000
    assert "write_budget" in str(shards["hot_path_storage"]["path_contains"])
    assert shards["admission_evidence"]["merge_to_primary"] is False
    assert "teacher_lineage" in str(shards["admission_evidence"]["path_contains"])
    assert shards["reports"]["merge_to_primary"] is False
    assert "exports/reports/" in str(shards["reports"]["path_contains"])
    assert shards["trading_fast"]["include_streams"] == "paper_broker_bridge,top_level_trade_links"
    assert shards["explanations"]["include_streams"] == "decision_explanations"
    assert shards["explanations"]["merge_hot_days"] == 3
    assert shards["explanations"]["hot_retention_hot_hours"] == 0
    assert shards["explanations"]["merge_priority"] == "low"
    assert "shadow_pnl_attribution_" in str(shards["shadow_attribution"]["path_contains"])
    assert shards["shadow_attribution"]["merge_to_primary"] is False
    assert shards["aggressive_trading"]["include_streams"] == "decisions,trade_logs"
    assert "shadow_intraday_aggressive_" in str(shards["aggressive_trading"]["path_contains"])
    assert shards["aggressive_trading"]["max_lines_per_file"] == 20000
    assert shards["aggressive_trading"]["state_checkpoint_lines"] == 2000
    assert shards["aggressive_trading"]["merge_max_jsonl_rows"] == 16000
    assert shards["crypto_trading"]["include_streams"] == "decisions,trade_logs"
    assert shards["crypto_trading"]["max_lines_per_file"] == 12000
    assert shards["crypto_trading"]["state_checkpoint_lines"] == 2000
    assert shards["crypto_trading"]["merge_max_jsonl_rows"] == 8000
    assert shards["trading"]["include_streams"] == "decisions,trade_logs"
    assert shards["trading"]["max_lines_per_file"] == 16000
    assert shards["trading"]["state_checkpoint_lines"] == 2000
    assert shards["trading"]["merge_max_jsonl_rows"] == 12000
    assert "shadow_intraday_aggressive_" in str(shards["trading"]["path_not_contains"])
    assert "governance_walk_forward" in str(shards["governance"]["include_streams"])
    assert "governance/watchdog/" in str(shards["governance"]["path_not_contains"])
    assert "governance/channels/runtime/" in str(shards["governance"]["path_not_contains"])
    assert "channel_schema_violations_" in str(shards["governance"]["path_not_contains"])
    assert shards["governance"]["max_lines_per_file"] == 8000
    assert shards["governance"]["state_checkpoint_lines"] == 2000
    assert shards["governance"]["merge_max_jsonl_rows"] == 6000
    assert shards["support_watchdog"]["include_streams"] == "governance_watchdog"
    assert "governance/watchdog/" in str(shards["support_watchdog"]["path_contains"])
    assert shards["support_watchdog"]["max_lines_per_file"] == 96000
    assert shards["support_watchdog"]["state_checkpoint_lines"] == 4000
    assert shards["support_watchdog"]["merge_max_jsonl_rows"] == 64000
    assert shards["support_watchdog"]["merge_priority"] == "low"
    assert shards["support_watchdog"]["merge_to_primary"] is False
    assert shards["data"]["skip_json_files"] is False
    assert "external_context" in str(shards["data"]["include_streams"])
    assert "exports/external_context/" in str(shards["data"]["path_contains"])
    assert shards["data"]["merge_priority"] == "low"
    assert shards["data"]["merge_to_primary"] is False


def test_load_active_request_sanitizes_live_drain_overrides(tmp_path) -> None:
    request_path = tmp_path / "sql_link_service_request_latest.json"
    request_path.write_text(
        json.dumps(
                {
                    "active": True,
                    "request_kind": "external_backlog_drain",
                    "requested_at": "2026-04-17T11:00:00+00:00",
                    "expires_utc": "2099-04-18T12:00:00+00:00",
                    "env_overrides": {
                    "SQL_LINK_SERVICE_SHARDS": "health_fast,runtime,trading",
                    "SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE": "25",
                    "BAD_KEY": "ignore-me",
                },
            }
        ),
        encoding="utf-8",
    )

    payload = shard_manager._load_active_request(request_path)

    assert payload["request_kind"] == "external_backlog_drain"
    assert payload["env_overrides"]["SQL_LINK_SERVICE_SHARDS"] == "health_fast,runtime,trading"
    assert payload["env_overrides"]["SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE"] == "25"
    assert "BAD_KEY" not in payload["env_overrides"]


def test_effective_cycle_args_applies_live_request_env() -> None:
    args = shard_manager.argparse.Namespace(
        interval_seconds=20,
        link_mode="sqlite",
        shards="health_fast,governance",
        low_priority_merge_skip_gb=120.0,
        merge_max_seconds_per_cycle=60.0,
        shard_link_timeout_seconds=180,
        auto_wal_checkpoint=True,
        wal_checkpoint_threshold_gb=2.0,
        wal_checkpoint_trigger_growth_gb=1.5,
        wal_checkpoint_trigger_rows=750000,
        wal_checkpoint_min_interval_seconds=900,
        wal_truncate_max_gb=8.0,
        wal_checkpoint_mode="auto",
        auto_hot_retention=True,
        hot_retention_max_db_gb=12.0,
        hot_retention_trigger_growth_gb=2.0,
        hot_retention_trigger_rows=500000,
        hot_retention_hot_days=3,
        hot_retention_hot_hours=0,
        hot_retention_batch_size=120000,
        hot_retention_max_rows=1000000,
        hot_retention_min_interval_seconds=180,
    )

    effective = shard_manager._effective_cycle_args(
        args,
        {
            "SQL_LINK_SERVICE_SHARDS": "health_fast,runtime,trading",
            "SQL_LINK_SERVICE_INTERVAL_SECONDS": "12",
            "SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE": "25",
            "SQL_LINK_SERVICE_SHARD_LINK_TIMEOUT_SECONDS": "45",
            "SQL_LINK_SERVICE_HOT_BATCH_SIZE": "240000",
        },
    )

    assert effective.shards == "health_fast,runtime,trading"
    assert effective.interval_seconds == 12
    assert effective.merge_max_seconds_per_cycle == 25.0
    assert effective.shard_link_timeout_seconds == 45
    assert effective.hot_retention_batch_size == 240000


def test_run_shard_links_records_timeout_and_continues(tmp_path, monkeypatch) -> None:
    health_file = tmp_path / "health.json"
    shard = {
        "name": "trading",
        "sqlite_db": tmp_path / "trading.sqlite3",
        "state_file": tmp_path / "state.json",
        "health_file": health_file,
        "journal_file": tmp_path / "journal.jsonl",
        "journal_events_file": tmp_path / "journal_events.jsonl",
        "invalid_log_file": tmp_path / "invalid.jsonl",
        "include_streams": "decisions",
        "path_contains": "decisions/paper/",
        "skip_json_files": True,
        "max_files": 1,
        "max_lines_per_file": 100,
        "state_checkpoint_lines": 10,
    }
    monkeypatch.setattr(
        shard_manager,
        "_quarantine_shard_artifacts",
        lambda **kwargs: {"triggered": False},
    )

    def fake_run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=["link"], timeout=1, output="working", stderr="slow")

    monkeypatch.setattr(shard_manager.subprocess, "run", fake_run)

    results = shard_manager._run_shard_links(
        shards=[shard],
        link_mode="sqlite",
        sqlite_timeout_seconds=30,
        sqlite_lock_retries=0,
        sqlite_lock_retry_delay_seconds=0.1,
        shard_link_timeout_seconds=1,
    )

    assert results[0]["rc"] == 124
    assert results[0]["timed_out"] is True
    assert results[0]["timeout_seconds"] == 1


def test_build_shards_reads_hourly_hot_retention_overrides(monkeypatch) -> None:
    monkeypatch.setenv("SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_HOT_HOURS", "2")
    monkeypatch.setenv("SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_MAX_ROWS", "900000")

    shards = {
        row["name"]: row
        for row in shard_manager._build_shards(["explanations"])
    }

    assert shards["explanations"]["hot_retention_hot_hours"] == 2
    assert shards["explanations"]["hot_retention_max_rows"] == 900000


def test_build_shards_uses_heat_map_to_expand_hot_shard_capacity(monkeypatch) -> None:
    monkeypatch.setattr(
        shard_manager.ops_data_plane,
        "load_shard_heat_map",
        lambda project_root: {
            "explanations": {
                "promotion_candidate": True,
                "last_heat_score": 3.4,
            }
        },
    )

    shards = {
        row["name"]: row
        for row in shard_manager._build_shards(["explanations"])
    }

    assert shards["explanations"]["heat_promotion_candidate"] is True
    assert shards["explanations"]["last_heat_score"] == 3.4
    assert shards["explanations"]["max_files"] == int(shard_manager.DEFAULT_SHARD_DEFS["explanations"]["max_files"]) + 2


def test_build_shards_fails_open_when_heat_map_unavailable(monkeypatch) -> None:
    def _raise(_project_root):
        raise sqlite3.DatabaseError("database disk image is malformed")

    monkeypatch.setattr(shard_manager.ops_data_plane, "load_shard_heat_map", _raise)

    shards = {
        row["name"]: row
        for row in shard_manager._build_shards(["trading"])
    }

    assert shards["trading"]["heat_promotion_candidate"] is False


def test_sql_link_service_payload_marks_mysql_disabled_in_sqlite_mode(tmp_path) -> None:
    health_root = tmp_path / "governance" / "health"
    health_root.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp_utc": "2026-04-18T13:00:00+00:00",
        "ok": True,
        "rc": 0,
        "mode": "sharded_merge",
        "link_mode": "sqlite",
        "sinks": {
            "sqlite": {"enabled": True, "status": "active"},
            "mysql": {"enabled": False, "status": "disabled_by_link_mode"},
        },
    }
    (health_root / "sql_link_service_latest.json").write_text(json.dumps(payload), encoding="utf-8")

    saved = json.loads((health_root / "sql_link_service_latest.json").read_text(encoding="utf-8"))

    assert saved["sinks"]["sqlite"]["enabled"] is True
    assert saved["sinks"]["mysql"]["enabled"] is False
    assert saved["sinks"]["mysql"]["status"] == "disabled_by_link_mode"


def test_should_skip_low_priority_merge_when_primary_db_is_large() -> None:
    skip, reason = shard_manager._should_skip_low_priority_merge(
        shard={"name": "data", "merge_priority": "low"},
        primary_db_size_gb=140.0,
        skip_threshold_gb=120.0,
    )
    keep, keep_reason = shard_manager._should_skip_low_priority_merge(
        shard={"name": "governance", "merge_priority": "normal"},
        primary_db_size_gb=140.0,
        skip_threshold_gb=120.0,
    )

    assert skip is True
    assert "120" in reason
    assert keep is False
    assert keep_reason == ""


def test_db_size_gb_prefers_live_page_usage_over_sparse_logical_size(tmp_path) -> None:
    db_path = tmp_path / "sparse.sqlite3"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE jsonl_records (id INTEGER PRIMARY KEY, payload_json TEXT)")
    conn.execute("INSERT INTO jsonl_records (payload_json) VALUES (?)", ("{}",))
    conn.commit()
    page_size = int(conn.execute("PRAGMA page_size").fetchone()[0])
    page_count = int(conn.execute("PRAGMA page_count").fetchone()[0])
    conn.close()

    with db_path.open("ab") as fh:
        fh.truncate((page_size * page_count) + (32 * 1024 * 1024))

    measured_bytes = shard_manager._db_size_gb(db_path) * (1024.0 ** 3)

    assert measured_bytes < db_path.stat().st_size
    assert measured_bytes == page_size * page_count


def test_load_maintenance_state_defaults_to_current_sizes(tmp_path) -> None:
    state = shard_manager._load_maintenance_state(
        tmp_path / "missing.json",
        db_size_gb=12.5,
        wal_size_gb=1.25,
    )

    assert state["wal_checkpoint"]["baseline_db_size_gb"] == 12.5
    assert state["wal_checkpoint"]["baseline_wal_size_gb"] == 1.25
    assert state["hot_retention"]["baseline_db_size_gb"] == 12.5
    assert state["hot_retention"]["rows_since_last_run"] == 0


def test_hot_retention_triggers_on_oversize_after_successful_run() -> None:
    reasons = shard_manager._hot_retention_trigger_reasons(
        db_size_gb=232.0,
        max_db_gb=25.0,
        db_growth_gb=0.8,
        growth_trigger_gb=12.0,
        rows_since_last_run=500000,
        row_trigger=2500000,
        has_successful_run=True,
    )

    assert reasons == ["db_size_gb>=25"]


def test_hot_retention_bootstraps_on_large_db_without_prior_run() -> None:
    reasons = shard_manager._hot_retention_trigger_reasons(
        db_size_gb=232.0,
        max_db_gb=25.0,
        db_growth_gb=0.0,
        growth_trigger_gb=12.0,
        rows_since_last_run=0,
        row_trigger=2500000,
        has_successful_run=False,
    )

    assert reasons == ["bootstrap_db_size_gb>=25"]


def test_wal_checkpoint_triggers_on_growth_or_rows() -> None:
    reasons = shard_manager._wal_checkpoint_trigger_reasons(
        wal_size_gb=0.9,
        wal_threshold_gb=2.0,
        wal_growth_gb=1.6,
        wal_growth_trigger_gb=1.5,
        rows_since_last_run=800000,
        row_trigger=750000,
    )

    assert "wal_growth_gb>=1.5" in reasons
    assert "rows_since_last_run>=750000" in reasons


def test_build_shards_splits_crypto_paths_from_generic_shards(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(shard_manager, "SHARD_DB_ROOT", tmp_path / "sql_link_shards")
    monkeypatch.setattr(shard_manager, "SHARD_STATE_ROOT", tmp_path / "state")
    monkeypatch.setattr(shard_manager, "HEALTH_ROOT", tmp_path / "health")
    monkeypatch.setattr(shard_manager, "EVENT_ROOT", tmp_path / "events")

    shards = shard_manager._build_shards(["trading", "crypto_trading", "governance", "support_watchdog", "crypto_governance", "data"])
    by_name = {str(row["name"]): row for row in shards}

    assert "crypto_trading" in by_name
    assert "crypto_governance" in by_name
    assert "shadow_crypto/" in str(by_name["crypto_trading"]["path_contains"])
    assert "shadow_crypto/" in str(by_name["trading"]["path_not_contains"])
    assert "default_crypto_coinbase" in str(by_name["crypto_governance"]["path_contains"])
    assert "default_crypto_schwab" in str(by_name["governance"]["path_not_contains"])
    assert "governance/watchdog/" in str(by_name["governance"]["path_not_contains"])
    assert by_name["support_watchdog"]["include_streams"] == "governance_watchdog"
    assert by_name["crypto_trading"]["include_streams"] == by_name["trading"]["include_streams"]
    assert by_name["crypto_governance"]["include_streams"] == by_name["governance"]["include_streams"]
    assert by_name["crypto_trading"]["max_files"] == 10
    assert by_name["crypto_governance"]["max_files"] == 12


def test_probe_shard_merge_state_detects_up_to_date_shard(tmp_path) -> None:
    primary_db = tmp_path / "primary.sqlite3"
    shard_db = tmp_path / "governance.sqlite3"

    _create_shard_jsonl_db(shard_db)

    conn = sqlite3.connect(str(primary_db))
    shard_manager._ensure_primary_schema(conn)
    conn.execute(
        """
        INSERT INTO shard_merge_state (shard_name, last_jsonl_id, last_json_file_id, updated_at)
        VALUES (?, ?, ?, ?)
        """,
        ("governance", 2, 0, "2026-03-29T16:00:00+00:00"),
    )
    conn.commit()
    conn.close()

    probe = shard_manager._probe_shard_merge_state(
        shard_name="governance",
        shard_db=shard_db,
        primary_db=primary_db,
        sqlite_timeout_seconds=30,
    )

    assert probe["ok"] is True
    assert probe["merge_required"] is False
    assert probe["max_jsonl_id"] == 2


def test_probe_shard_merge_state_detects_pending_merge(tmp_path) -> None:
    primary_db = tmp_path / "primary.sqlite3"
    shard_db = tmp_path / "governance.sqlite3"

    _create_shard_jsonl_db(shard_db)

    conn = sqlite3.connect(str(primary_db))
    shard_manager._ensure_primary_schema(conn)
    conn.execute(
        """
        INSERT INTO shard_merge_state (shard_name, last_jsonl_id, last_json_file_id, updated_at)
        VALUES (?, ?, ?, ?)
        """,
        ("governance", 1, 0, "2026-03-29T16:00:00+00:00"),
    )
    conn.commit()
    conn.close()

    probe = shard_manager._probe_shard_merge_state(
        shard_name="governance",
        shard_db=shard_db,
        primary_db=primary_db,
        sqlite_timeout_seconds=30,
    )

    assert probe["ok"] is True
    assert probe["merge_required"] is True


def test_merge_shard_into_primary_respects_merge_hot_cutoff(tmp_path) -> None:
    primary_db = tmp_path / "primary.sqlite3"
    shard_db = tmp_path / "governance.sqlite3"

    conn = sqlite3.connect(str(shard_db))
    conn.execute(
        """
        CREATE TABLE jsonl_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT NOT NULL,
            source_rel TEXT NOT NULL,
            line_no INTEGER NOT NULL,
            ingested_at TEXT NOT NULL,
            payload_sha1 TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            run_id TEXT,
            iter_id TEXT,
            decision_id TEXT,
            parent_decision_id TEXT,
            log_schema_version INTEGER,
            UNIQUE(source_file, line_no)
        )
        """
    )
    conn.executemany(
        """
        INSERT INTO jsonl_records (
            source_file, source_rel, line_no, ingested_at, payload_sha1, payload_json,
            run_id, iter_id, decision_id, parent_decision_id, log_schema_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("a.jsonl", "decision_explanations/a.jsonl", 1, "2026-03-01T00:00:00+00:00", "a", "{}", "", "", "", "", 2),
            ("b.jsonl", "decision_explanations/b.jsonl", 1, "2026-04-01T00:00:00+00:00", "b", "{}", "", "", "", "", 2),
        ],
    )
    conn.commit()
    conn.close()

    result = shard_manager._merge_shard_into_primary(
        shard_name="explanations",
        shard_db=shard_db,
        primary_db=primary_db,
        sqlite_timeout_seconds=30,
        merge_hot_cutoff_utc="2026-03-15T00:00:00+00:00",
    )

    assert result["ok"] is True
    assert result["jsonl_rows_inserted"] == 1
    conn = sqlite3.connect(str(primary_db))
    rows = conn.execute("SELECT COUNT(*) FROM jsonl_records").fetchone()[0]
    conn.close()
    assert rows == 1


def test_merge_shard_into_primary_caps_jsonl_rows_per_cycle(tmp_path) -> None:
    primary_db = tmp_path / "primary.sqlite3"
    shard_db = tmp_path / "governance.sqlite3"

    _create_shard_jsonl_db(shard_db)

    first = shard_manager._merge_shard_into_primary(
        shard_name="governance",
        shard_db=shard_db,
        primary_db=primary_db,
        sqlite_timeout_seconds=30,
        merge_max_jsonl_rows=1,
    )
    second = shard_manager._merge_shard_into_primary(
        shard_name="governance",
        shard_db=shard_db,
        primary_db=primary_db,
        sqlite_timeout_seconds=30,
        merge_max_jsonl_rows=1,
    )

    assert first["ok"] is True
    assert first["jsonl_rows_inserted"] == 1
    assert first["merge_target_jsonl_id"] == 1
    assert first["last_jsonl_id"] == 1
    assert first["merge_capped"] is True
    assert second["ok"] is True
    assert second["jsonl_rows_inserted"] == 1
    assert second["merge_target_jsonl_id"] == 2
    assert second["last_jsonl_id"] == 2
    conn = sqlite3.connect(str(primary_db))
    rows = conn.execute("SELECT COUNT(*) FROM jsonl_records").fetchone()[0]
    conn.close()
    assert rows == 2


def test_prune_stale_local_fallback_artifacts_deletes_old_files_only(tmp_path) -> None:
    health_root = tmp_path / "health"
    data_root = tmp_path / "data"
    health_root.mkdir()
    data_root.mkdir()

    old_path = health_root / "one_numbers_latest.json.local_fallback.1"
    fresh_path = data_root / "jsonl_link.sqlite3-wal.local_fallback"
    old_path.write_text("x", encoding="utf-8")
    fresh_path.write_text("x", encoding="utf-8")

    stale_epoch = time.time() - 100000
    fresh_epoch = time.time()
    old_path.touch()
    fresh_path.touch()
    import os
    os.utime(old_path, (stale_epoch, stale_epoch))
    os.utime(fresh_path, (fresh_epoch, fresh_epoch))

    summary = shard_manager._prune_stale_local_fallback_artifacts(
        roots=[health_root, data_root],
        older_than_seconds=3600,
        max_files=20,
    )

    assert summary["candidate_files"] == 1
    assert summary["deleted_files"] == 1
    assert old_path.exists() is False
    assert fresh_path.exists() is True


def test_normalized_shard_config_upgrades_legacy_default() -> None:
    assert shard_manager._normalized_shard_config("") == shard_manager.CURRENT_DEFAULT_SHARDS
    assert shard_manager._normalized_shard_config("trading,governance,data") == shard_manager.CURRENT_DEFAULT_SHARDS
    assert shard_manager._normalized_shard_config("trading,governance,data,custom") == "trading,governance,data,custom"
