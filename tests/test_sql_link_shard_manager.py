import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.ops.sql_link_shard_manager as shard_manager


def test_child_python_inherits_manager_runtime(monkeypatch, tmp_path: Path) -> None:
    fake_parent = tmp_path / ".venv314" / "bin" / "python"
    fake_parent.parent.mkdir(parents=True, exist_ok=True)
    fake_parent.write_text("", encoding="utf-8")

    monkeypatch.delenv("SQL_LINK_SERVICE_PYTHON_BIN", raising=False)
    monkeypatch.setattr(shard_manager.sys, "executable", str(fake_parent))
    monkeypatch.setattr(
        shard_manager,
        "resolve_runtime_python",
        lambda _root: (_ for _ in ()).throw(AssertionError("parent runtime should win")),
    )

    assert shard_manager._resolve_child_python() == fake_parent


def test_child_python_honors_explicit_service_bin(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(shard_manager, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("SQL_LINK_SERVICE_PYTHON_BIN", ".venv314/bin/python")

    assert shard_manager._resolve_child_python() == (tmp_path / ".venv314" / "bin" / "python").resolve()


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


def test_queue_retention_inline_vacuum_is_explicit(monkeypatch) -> None:
    monkeypatch.delenv("SQL_LINK_SERVICE_QUEUE_VACUUM_INLINE_ENABLED", raising=False)
    assert shard_manager._queue_retention_inline_vacuum_enabled() is False

    monkeypatch.setenv("SQL_LINK_SERVICE_QUEUE_VACUUM_INLINE_ENABLED", "1")
    assert shard_manager._queue_retention_inline_vacuum_enabled() is True


def test_queue_retention_inline_cleanup_is_bounded(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class Result:
        returncode = 0
        stdout = "{}"
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["timeout"] = kwargs.get("timeout")
        return Result()

    monkeypatch.setenv("SQL_LINK_SERVICE_QUEUE_INLINE_MAX_ROWS", "12345")
    monkeypatch.setenv("SQL_LINK_SERVICE_QUEUE_RETENTION_TIMEOUT_SECONDS", "17")
    monkeypatch.setattr(shard_manager.subprocess, "run", fake_run)

    rc, out, err = shard_manager._run_queue_retention(
        db_path="queue.sqlite3",
        acked_days=7,
        batch_size=25000,
        max_rows=240000,
        cleanup_consumer_state_days=30,
        prune_orphans=True,
        orphan_days=14,
        vacuum=False,
    )

    assert rc == 0
    assert out == "{}"
    assert err == ""
    assert captured["timeout"] == 17
    cmd = [str(item) for item in captured["cmd"]]
    assert cmd[cmd.index("--max-rows") + 1] == "12345"


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


def test_primary_schema_includes_route_label_columns(tmp_path) -> None:
    primary_db = tmp_path / "primary.sqlite3"
    conn = sqlite3.connect(str(primary_db))
    try:
        shard_manager._ensure_primary_schema(conn)
        cols = {str(row[1]) for row in conn.execute("PRAGMA table_info(jsonl_records)")}
    finally:
        conn.close()

    for col in (
        "source_day_utc",
        "source_stream",
        "source_partition_key",
        "source_broker",
        "source_provider",
        "source_venue",
        "asset_class",
        "routing_lane",
        "source_quality_label",
    ):
        assert col in cols


def test_merge_shard_into_primary_preserves_route_label_columns(tmp_path) -> None:
    primary_db = tmp_path / "primary.sqlite3"
    shard_db = tmp_path / "coinbase.sqlite3"
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
            source_day_utc TEXT,
            source_stream TEXT,
            source_partition_key TEXT,
            source_broker TEXT,
            source_provider TEXT,
            source_venue TEXT,
            asset_class TEXT,
            routing_lane TEXT,
            source_quality_label TEXT,
            UNIQUE(source_file, line_no)
        )
        """
    )
    conn.execute(
        """
        INSERT INTO jsonl_records (
            source_file, source_rel, line_no, ingested_at, payload_sha1, payload_json,
            run_id, iter_id, decision_id, parent_decision_id, log_schema_version,
            source_day_utc, source_stream, source_partition_key, source_broker,
            source_provider, source_venue, asset_class, routing_lane, source_quality_label
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "coinbase.jsonl",
            "governance/channels/api/default_crypto_coinbase/api_20260624.jsonl",
            1,
            "2026-06-24T14:30:00+00:00",
            "sha1-route",
            "{}",
            "run-route",
            "iter-route",
            "",
            "",
            2,
            "2026-06-24",
            "governance",
            "2026-06-24:governance",
            "coinbase",
            "coinbase_ticker",
            "coinbase",
            "crypto",
            "coinbase_crypto",
            "exchange_native",
        ),
    )
    conn.commit()
    conn.close()

    result = shard_manager._merge_shard_into_primary(
        shard_name="coinbase",
        shard_db=shard_db,
        primary_db=primary_db,
        sqlite_timeout_seconds=30,
    )
    conn = sqlite3.connect(str(primary_db))
    try:
        row = conn.execute(
            "SELECT source_broker, source_provider, source_venue, asset_class, routing_lane, source_quality_label FROM jsonl_records"
        ).fetchone()
    finally:
        conn.close()

    assert result["ok"] is True
    assert result["jsonl_rows_inserted"] == 1
    assert row == ("coinbase", "coinbase_ticker", "coinbase", "crypto", "coinbase_crypto", "exchange_native")


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


def test_configured_primary_db_path_uses_local_fallback_for_broken_route(tmp_path, monkeypatch) -> None:
    routed_primary = tmp_path / "data" / "jsonl_link.sqlite3"
    missing_external_primary = tmp_path / "missing_bot_logs" / "data" / "jsonl_link.sqlite3"
    fallback_primary = tmp_path / "local_fallback_storage" / "data" / "jsonl_link.sqlite3"
    routed_primary.parent.mkdir(parents=True, exist_ok=True)
    routed_primary.symlink_to(missing_external_primary)
    monkeypatch.setattr(shard_manager, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(shard_manager, "LOCAL_FALLBACK_ROOT", tmp_path / "local_fallback_storage")
    monkeypatch.setattr(shard_manager, "PRIMARY_DB_PATH", routed_primary)

    configured = shard_manager._configured_primary_db_path(str(routed_primary))

    assert configured == fallback_primary
    assert shard_manager._primary_db_role(configured, configured.resolve(strict=False)) == "compatibility_cache"


def test_routed_or_local_fallback_path_redirects_broken_shard_symlink(tmp_path, monkeypatch) -> None:
    routed_shards = tmp_path / "data" / "sql_link_shards"
    missing_external_shards = tmp_path / "missing_bot_logs" / "data" / "sql_link_shards"
    fallback_shards = tmp_path / "local_fallback_storage" / "data" / "sql_link_shards"
    routed_shards.parent.mkdir(parents=True, exist_ok=True)
    routed_shards.symlink_to(missing_external_shards)
    monkeypatch.setattr(shard_manager, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(shard_manager, "LOCAL_FALLBACK_ROOT", tmp_path / "local_fallback_storage")

    assert shard_manager._routed_or_local_fallback_path(routed_shards) == fallback_shards


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
                    "risk_support",
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
    assert "governance/channels/risk/" in str(shards["crypto_shadow_attribution"]["path_not_contains"])
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
    assert shards["aggressive_trading"]["max_bytes_per_file"] == 128 * 1024 * 1024
    assert shards["aggressive_trading"]["sqlite_batch_max_bytes"] == 32 * 1024 * 1024
    assert shards["aggressive_trading"]["state_checkpoint_lines"] == 2000
    assert shards["aggressive_trading"]["merge_max_jsonl_rows"] == 16000
    assert shards["crypto_trading"]["include_streams"] == "decisions,trade_logs"
    assert shards["crypto_trading"]["max_lines_per_file"] == 12000
    assert shards["crypto_trading"]["max_bytes_per_file"] == 128 * 1024 * 1024
    assert shards["crypto_trading"]["sqlite_batch_max_bytes"] == 32 * 1024 * 1024
    assert shards["crypto_trading"]["state_checkpoint_lines"] == 2000
    assert shards["crypto_trading"]["merge_max_jsonl_rows"] == 8000
    assert shards["trading"]["include_streams"] == "decisions,trade_logs"
    assert shards["trading"]["max_lines_per_file"] == 16000
    assert shards["trading"]["max_bytes_per_file"] == 128 * 1024 * 1024
    assert shards["trading"]["sqlite_batch_max_bytes"] == 32 * 1024 * 1024
    assert shards["trading"]["state_checkpoint_lines"] == 2000
    assert shards["trading"]["merge_max_jsonl_rows"] == 12000
    assert "shadow_intraday_aggressive_" in str(shards["trading"]["path_not_contains"])
    assert "governance_walk_forward" in str(shards["governance"]["include_streams"])
    assert "governance/watchdog/" in str(shards["governance"]["path_not_contains"])
    assert "governance/channels/risk/" in str(shards["governance"]["path_not_contains"])
    assert "governance/channels/runtime/" in str(shards["governance"]["path_not_contains"])
    assert "channel_schema_violations_" in str(shards["governance"]["path_not_contains"])
    assert shards["governance"]["max_lines_per_file"] == 8000
    assert shards["governance"]["state_checkpoint_lines"] == 2000
    assert shards["governance"]["merge_max_jsonl_rows"] == 6000
    assert shards["risk_support"]["include_streams"] == "governance"
    assert "governance/channels/risk/" in str(shards["risk_support"]["path_contains"])
    assert shards["risk_support"]["max_lines_per_file"] == 400000
    assert shards["risk_support"]["state_checkpoint_lines"] == 16000
    assert shards["risk_support"]["merge_max_jsonl_rows"] == 0
    assert shards["risk_support"]["merge_to_primary"] is False
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
                    "SQL_LINK_SERVICE_SINGLE_WRITER_ONLY": "1",
                    "SQL_LINK_SERVICE_PREPROCESS_WORKERS": "4",
                    "INGEST_MAX_BYTES_PER_FILE": "67108864",
                    "SQLITE_BATCH_MAX_BYTES": "16777216",
                    "BOT_OPS_SQLITE_CACHE_SIZE_KB": "8192",
                    "BACKLOG_PCORE_ALLOCATION_ACTIVE": "1",
                    "BACKLOG_DRAIN_SINGLE_WRITER_ONLY": "1",
                    "BACKLOG_PCORE_PREPROCESS_WORKERS": "4",
                    "BACKLOG_PCORE_BURST_MODE": "daily_driver_5",
                    "BACKLOG_PCORE_BURST_REASON": "normal daily-driver headroom",
                    "TRAINING_PCORE_ALLOWED_WHEN_BACKLOG_GREEN": "1",
                    "TRAINING_PCORE_MAX_WORKERS": "2",
                    "TRAINING_PCORE_NICE": "8",
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
    assert payload["env_overrides"]["INGEST_MAX_BYTES_PER_FILE"] == "67108864"
    assert payload["env_overrides"]["SQLITE_BATCH_MAX_BYTES"] == "16777216"
    assert payload["env_overrides"]["BOT_OPS_SQLITE_CACHE_SIZE_KB"] == "8192"
    assert payload["env_overrides"]["BACKLOG_PCORE_ALLOCATION_ACTIVE"] == "1"
    assert payload["env_overrides"]["SQL_LINK_SERVICE_PREPROCESS_WORKERS"] == "4"
    assert payload["env_overrides"]["BACKLOG_PCORE_BURST_MODE"] == "daily_driver_5"
    p_core = shard_manager._p_core_drain_contract(payload)
    assert p_core["active"] is True
    assert p_core["single_writer_only"] is True
    assert p_core["preprocess_worker_budget"] == 4
    assert p_core["p_core_burst_intelligence"]["mode"] == "daily_driver_5"
    assert p_core["training_pcore_gate"]["max_workers"] == 2
    assert "BAD_KEY" not in payload["env_overrides"]


def test_effective_cycle_args_applies_live_request_env() -> None:
    args = shard_manager.argparse.Namespace(
        interval_seconds=20,
        link_mode="sqlite",
        shards="health_fast,governance",
        low_priority_merge_skip_gb=120.0,
        merge_max_seconds_per_cycle=60.0,
        shard_link_timeout_seconds=180,
        preprocess_workers=1,
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
            "SQL_LINK_SERVICE_PREPROCESS_WORKERS": "4",
            "SQL_LINK_SERVICE_HOT_BATCH_SIZE": "240000",
        },
    )

    assert effective.shards == "health_fast,runtime,trading"
    assert effective.interval_seconds == 12
    assert effective.preprocess_workers == 4
    assert effective.merge_max_seconds_per_cycle == 25.0
    assert effective.shard_link_timeout_seconds == 45
    assert effective.hot_retention_batch_size == 240000


def test_temporary_env_overrides_clears_stale_shard_path_filters(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(shard_manager, "SHARD_DB_ROOT", tmp_path / "sql_link_shards")
    monkeypatch.setattr(shard_manager, "SHARD_STATE_ROOT", tmp_path / "state")
    monkeypatch.setattr(shard_manager, "HEALTH_ROOT", tmp_path / "health")
    monkeypatch.setattr(shard_manager, "EVENT_ROOT", tmp_path / "events")
    monkeypatch.setenv(
        "SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_PATH_CONTAINS",
        "decision_explanations/shadow_bond_equities/decision_explanations_20260612.jsonl",
    )
    overrides = {
        "SQL_LINK_SERVICE_SHARDS": "explanations,crypto_explanations",
        "SQL_LINK_SERVICE_SHARD_EXPLANATIONS_PATH_CONTAINS": (
            "decision_explanations/shadow_bond_equities/decision_explanations_20260612.jsonl"
        ),
    }

    with shard_manager._temporary_env_overrides(overrides):
        shards = shard_manager._build_shards(["explanations", "crypto_explanations"])
        by_name = {str(row["name"]): row for row in shards}

    assert "shadow_bond_equities" in str(by_name["explanations"]["path_contains"])
    assert "shadow_crypto/" in str(by_name["crypto_explanations"]["path_contains"])
    assert "shadow_bond_equities" not in str(by_name["crypto_explanations"]["path_contains"])
    assert (
        os.environ["SQL_LINK_SERVICE_SHARD_CRYPTO_EXPLANATIONS_PATH_CONTAINS"]
        == "decision_explanations/shadow_bond_equities/decision_explanations_20260612.jsonl"
    )


def test_cycle_runtime_overrides_reads_live_runtime_guard(tmp_path, monkeypatch) -> None:
    runtime_override = tmp_path / ".env.runtime_resource_guard_override"
    pressure_override = tmp_path / ".env.pressure_relief_override"
    runtime_override.write_text(
        "\n".join(
            [
                "SQL_LINK_SERVICE_HOST_COOLING_ACTIVE=1",
                "SQL_LINK_SERVICE_PREPROCESS_WORKERS=1",
                "SQL_LINK_SERVICE_INTERVAL_SECONDS=120",
                "BAD_KEY=ignored",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pressure_override.write_text(
        "SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE=20\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(shard_manager, "RUNTIME_RESOURCE_GUARD_OVERRIDE_PATH", runtime_override)
    monkeypatch.setattr(shard_manager, "PRESSURE_RELIEF_OVERRIDE_PATH", pressure_override)

    overrides = shard_manager._cycle_runtime_overrides({"env_overrides": {"SQL_LINK_SERVICE_PREPROCESS_WORKERS": "2"}})

    assert overrides["SQL_LINK_SERVICE_HOST_COOLING_ACTIVE"] == "1"
    assert overrides["SQL_LINK_SERVICE_PREPROCESS_WORKERS"] == "2"
    assert overrides["SQL_LINK_SERVICE_INTERVAL_SECONDS"] == "120"
    assert overrides["SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE"] == "20"
    assert "BAD_KEY" not in overrides


def test_cycle_runtime_overrides_preserves_child_cooling_controls(tmp_path, monkeypatch) -> None:
    runtime_override = tmp_path / ".env.runtime_resource_guard_override"
    pressure_override = tmp_path / ".env.pressure_relief_override"
    runtime_override.write_text(
        "\n".join(
            [
                "INGEST_HOST_LOAD_SOFT_CAP=6.0",
                "INGEST_HOST_LOAD_SLEEP_SECONDS=0.50",
                "INGEST_FLUSH_SLEEP_SECONDS=0.10",
                "INGEST_FILE_SLEEP_SECONDS=0.02",
                "SQL_LINK_WRITER_NICE=18",
                "SQL_LINK_WRITER_BACKGROUND_POLICY=1",
                "SQL_LINK_CHILD_WRITER_CPU_POLICY=background",
                "BOT_CPU_ALLOCATION_POLICY=efficiency",
                "BOT_CPU_QOS_POLICY=background",
                "UNRELATED_SHELL_CONTROL=ignored",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pressure_override.write_text("", encoding="utf-8")
    monkeypatch.setattr(shard_manager, "RUNTIME_RESOURCE_GUARD_OVERRIDE_PATH", runtime_override)
    monkeypatch.setattr(shard_manager, "PRESSURE_RELIEF_OVERRIDE_PATH", pressure_override)

    overrides = shard_manager._cycle_runtime_overrides({})

    assert overrides["INGEST_HOST_LOAD_SOFT_CAP"] == "6.0"
    assert overrides["INGEST_HOST_LOAD_SLEEP_SECONDS"] == "0.50"
    assert overrides["INGEST_FLUSH_SLEEP_SECONDS"] == "0.10"
    assert overrides["INGEST_FILE_SLEEP_SECONDS"] == "0.02"
    assert overrides["SQL_LINK_WRITER_NICE"] == "18"
    assert overrides["SQL_LINK_WRITER_BACKGROUND_POLICY"] == "1"
    assert overrides["SQL_LINK_CHILD_WRITER_CPU_POLICY"] == "background"
    assert overrides["BOT_CPU_ALLOCATION_POLICY"] == "efficiency"
    assert overrides["BOT_CPU_QOS_POLICY"] == "background"
    assert "UNRELATED_SHELL_CONTROL" not in overrides


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


def test_run_shard_links_applies_shard_specific_timeout(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SQL_LINK_SERVICE_SHARD_GOVERNANCE_TIMEOUT_SECONDS", "7")
    shard = {
        "name": "governance",
        "sqlite_db": tmp_path / "governance.sqlite3",
        "state_file": tmp_path / "governance_state.json",
        "health_file": tmp_path / "governance_health.json",
        "journal_file": tmp_path / "governance_journal.jsonl",
        "journal_events_file": tmp_path / "governance_journal_events.jsonl",
        "invalid_log_file": tmp_path / "governance_invalid.jsonl",
        "include_streams": "governance",
        "skip_json_files": False,
        "max_files": 1,
        "max_lines_per_file": 100,
        "state_checkpoint_lines": 10,
    }
    captured_timeouts: list[int] = []

    monkeypatch.setattr(
        shard_manager,
        "_quarantine_shard_artifacts",
        lambda **kwargs: {"triggered": False},
    )

    def fake_run(_cmd, *_args, **kwargs):
        captured_timeouts.append(int(kwargs["timeout"]))
        return subprocess.CompletedProcess(args=["link"], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(shard_manager.subprocess, "run", fake_run)

    results = shard_manager._run_shard_links(
        shards=[shard],
        link_mode="sqlite",
        sqlite_timeout_seconds=30,
        sqlite_lock_retries=0,
        sqlite_lock_retry_delay_seconds=0.1,
        shard_link_timeout_seconds=30,
    )

    assert captured_timeouts == [7]
    assert results[0]["timeout_seconds"] == 7


def test_run_shard_links_emits_active_shard_progress(tmp_path, monkeypatch) -> None:
    shards = []
    for name in ("data", "reports"):
        shards.append(
            {
                "name": name,
                "sqlite_db": tmp_path / f"{name}.sqlite3",
                "state_file": tmp_path / f"{name}_state.json",
                "health_file": tmp_path / f"{name}_health.json",
                "journal_file": tmp_path / f"{name}_journal.jsonl",
                "journal_events_file": tmp_path / f"{name}_journal_events.jsonl",
                "invalid_log_file": tmp_path / f"{name}_invalid.jsonl",
                "include_streams": name,
                "skip_json_files": False,
                "max_files": 1,
                "max_lines_per_file": 100,
                "state_checkpoint_lines": 10,
            }
        )
    events: list[list[dict[str, object]]] = []

    monkeypatch.setattr(
        shard_manager,
        "_quarantine_shard_artifacts",
        lambda **kwargs: {"triggered": False},
    )
    monkeypatch.setattr(
        shard_manager.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(args=["link"], returncode=0, stdout="", stderr=""),
    )

    def progress_callback(_rows, active_shard_links=None):
        events.append(list(active_shard_links or []))

    results = shard_manager._run_shard_links(
        shards=shards,
        link_mode="sqlite",
        sqlite_timeout_seconds=30,
        sqlite_lock_retries=0,
        sqlite_lock_retry_delay_seconds=0.1,
        shard_link_timeout_seconds=5,
        preprocess_workers=1,
        progress_callback=progress_callback,
    )

    assert [row["shard"] for row in results] == ["data", "reports"]
    assert any(event and event[0]["shard"] == "data" and event[0]["queued_shard_count"] == 1 for event in events)
    assert any(event and event[0]["shard"] == "reports" and event[0]["tail_shard"] is True for event in events)


def test_run_shard_links_uses_preprocess_worker_budget(tmp_path, monkeypatch) -> None:
    shards = []
    for idx, name in enumerate(["trading", "aggressive_trading", "crypto_trading"]):
        shards.append(
            {
                "name": name,
                "sqlite_db": tmp_path / f"{name}.sqlite3",
                "state_file": tmp_path / f"{name}_state.json",
                "health_file": tmp_path / f"{name}_health.json",
                "journal_file": tmp_path / f"{name}_journal.jsonl",
                "journal_events_file": tmp_path / f"{name}_journal_events.jsonl",
                "invalid_log_file": tmp_path / f"{name}_invalid.jsonl",
                "include_streams": "decisions",
                "path_contains": f"decisions/{idx}/",
                "skip_json_files": True,
                "max_files": 1,
                "max_lines_per_file": 100,
                "state_checkpoint_lines": 10,
            }
        )
    monkeypatch.setattr(
        shard_manager,
        "_quarantine_shard_artifacts",
        lambda **kwargs: {"triggered": False},
    )
    lock = threading.Lock()
    active = {"count": 0, "max": 0}

    def fake_run(*_args, **_kwargs):
        with lock:
            active["count"] += 1
            active["max"] = max(active["max"], active["count"])
        time.sleep(0.05)
        with lock:
            active["count"] -= 1
        return subprocess.CompletedProcess(args=["link"], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(shard_manager.subprocess, "run", fake_run)

    results = shard_manager._run_shard_links(
        shards=shards,
        link_mode="sqlite",
        sqlite_timeout_seconds=30,
        sqlite_lock_retries=0,
        sqlite_lock_retry_delay_seconds=0.1,
        shard_link_timeout_seconds=5,
        preprocess_workers=3,
    )

    assert [row["shard"] for row in results] == ["trading", "aggressive_trading", "crypto_trading"]
    assert active["max"] > 1
    assert all(row["parallel_preprocess"] is True for row in results)
    assert all(row["preprocess_worker_count"] == 3 for row in results)


def test_shard_writer_lane_contract_exposes_smart_parallelism(monkeypatch) -> None:
    monkeypatch.setenv("SQL_LINK_SERVICE_MAX_SHARD_WRITER_LANES", "4")
    monkeypatch.setenv("SQL_LINK_SERVICE_COLD_SHARD_LANE_CAP", "1")
    monkeypatch.setenv("SQL_LINK_SERVICE_WARM_SHARD_LANE_CAP", "2")

    contract = shard_manager._shard_writer_lane_contract(4)

    assert contract["single_primary_merge_writer"] is True
    assert contract["sqlite_primary_writer_count"] == 1
    smart = contract["smart_shard_parallelism"]
    assert smart["enabled"] is True
    assert smart["enforced_single_primary_merge_writer"] is True
    assert smart["tier_lane_caps"]["hot"] == 4
    assert smart["tier_lane_caps"]["warm"] == 2
    assert smart["tier_lane_caps"]["cold"] == 1


def test_run_shard_links_caps_cold_shard_parallelism(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SQL_LINK_SERVICE_SMART_SHARD_PARALLELISM", "1")
    monkeypatch.setenv("SQL_LINK_SERVICE_COLD_SHARD_LANE_CAP", "1")
    shards = []
    for name in ("data", "reports", "explanations"):
        shards.append(
            {
                "name": name,
                "sqlite_db": tmp_path / f"{name}.sqlite3",
                "state_file": tmp_path / f"{name}_state.json",
                "health_file": tmp_path / f"{name}_health.json",
                "journal_file": tmp_path / f"{name}_journal.jsonl",
                "journal_events_file": tmp_path / f"{name}_journal_events.jsonl",
                "invalid_log_file": tmp_path / f"{name}_invalid.jsonl",
                "include_streams": name,
                "skip_json_files": False,
                "max_files": 1,
                "max_lines_per_file": 100,
                "state_checkpoint_lines": 10,
            }
        )
    lock = threading.Lock()
    active = {"count": 0, "max": 0}

    monkeypatch.setattr(
        shard_manager,
        "_quarantine_shard_artifacts",
        lambda **kwargs: {"triggered": False},
    )

    def fake_run(*_args, **_kwargs):
        with lock:
            active["count"] += 1
            active["max"] = max(active["max"], active["count"])
        time.sleep(0.03)
        with lock:
            active["count"] -= 1
        return subprocess.CompletedProcess(args=["link"], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(shard_manager.subprocess, "run", fake_run)

    results = shard_manager._run_shard_links(
        shards=shards,
        link_mode="sqlite",
        sqlite_timeout_seconds=30,
        sqlite_lock_retries=0,
        sqlite_lock_retry_delay_seconds=0.1,
        shard_link_timeout_seconds=5,
        preprocess_workers=3,
    )

    assert [row["shard"] for row in results] == ["data", "reports", "explanations"]
    assert active["max"] == 1
    assert all(row["preprocess_worker_count"] == 3 for row in results)


def test_run_shard_links_lets_hot_shard_bypass_capped_cold_queue(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SQL_LINK_SERVICE_SMART_SHARD_PARALLELISM", "1")
    monkeypatch.setenv("SQL_LINK_SERVICE_COLD_SHARD_LANE_CAP", "1")
    shards = []
    for name in ("data", "reports", "trading"):
        shards.append(
            {
                "name": name,
                "sqlite_db": tmp_path / f"{name}.sqlite3",
                "state_file": tmp_path / f"{name}_state.json",
                "health_file": tmp_path / f"{name}_health.json",
                "journal_file": tmp_path / f"{name}_journal.jsonl",
                "journal_events_file": tmp_path / f"{name}_journal_events.jsonl",
                "invalid_log_file": tmp_path / f"{name}_invalid.jsonl",
                "include_streams": "decisions" if name == "trading" else name,
                "skip_json_files": name == "trading",
                "max_files": 1,
                "max_lines_per_file": 100,
                "state_checkpoint_lines": 10,
            }
        )
    started: list[str] = []
    lock = threading.Lock()

    monkeypatch.setattr(
        shard_manager,
        "_quarantine_shard_artifacts",
        lambda **kwargs: {"triggered": False},
    )

    def fake_run(cmd, *_args, **_kwargs):
        parts = [str(part) for part in cmd]
        db_name = Path(parts[parts.index("--sqlite-db") + 1]).stem
        shard_name = db_name.replace("jsonl_link_", "")
        with lock:
            started.append(shard_name)
        time.sleep(0.03)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(shard_manager.subprocess, "run", fake_run)

    shard_manager._run_shard_links(
        shards=shards,
        link_mode="sqlite",
        sqlite_timeout_seconds=30,
        sqlite_lock_retries=0,
        sqlite_lock_retry_delay_seconds=0.1,
        shard_link_timeout_seconds=5,
        preprocess_workers=3,
    )

    assert started.index("trading") < started.index("reports")


def test_run_shard_links_skips_fresh_idle_non_sentinel_shards(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SQL_LINK_SERVICE_SKIP_FRESH_IDLE_SHARDS", "1")
    monkeypatch.setenv("SQL_LINK_SERVICE_IDLE_SHARD_MAX_AGE_SECONDS", "120")
    health_file = tmp_path / "data_health.json"
    health_file.write_text(
        json.dumps(
            {
                "timestamp_utc": shard_manager._now_utc(),
                "overall_status": "ready",
                "sqlite": {"pending_lines": 0, "inserted": 0},
                "sqlite_json_files": {"pending_files": 0, "inserted": 0},
            }
        ),
        encoding="utf-8",
    )
    shard = {
        "name": "data",
        "sqlite_db": tmp_path / "data.sqlite3",
        "state_file": tmp_path / "data_state.json",
        "health_file": health_file,
        "journal_file": tmp_path / "data_journal.jsonl",
        "journal_events_file": tmp_path / "data_journal_events.jsonl",
        "invalid_log_file": tmp_path / "data_invalid.jsonl",
        "include_streams": "data",
        "skip_json_files": False,
        "max_files": 1,
        "max_lines_per_file": 100,
        "state_checkpoint_lines": 10,
    }
    calls = {"quarantine": 0, "subprocess": 0}

    def fake_quarantine(**_kwargs):
        calls["quarantine"] += 1
        return {"triggered": False}

    def fake_run(*_args, **_kwargs):
        calls["subprocess"] += 1
        return subprocess.CompletedProcess(args=["link"], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(shard_manager, "_quarantine_shard_artifacts", fake_quarantine)
    monkeypatch.setattr(shard_manager.subprocess, "run", fake_run)

    results = shard_manager._run_shard_links(
        shards=[shard],
        link_mode="sqlite",
        sqlite_timeout_seconds=30,
        sqlite_lock_retries=0,
        sqlite_lock_retry_delay_seconds=0.1,
        shard_link_timeout_seconds=5,
        preprocess_workers=2,
    )

    assert results[0]["rc"] == 0
    assert results[0]["skipped"] is True
    assert results[0]["skip_reason"] == "fresh_idle_health"
    assert results[0]["preprocess_worker_count"] == 1
    assert calls == {"quarantine": 0, "subprocess": 0}


def test_run_shard_links_does_not_skip_fresh_idle_dirty_health(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SQL_LINK_SERVICE_SKIP_FRESH_IDLE_SHARDS", "1")
    monkeypatch.setenv("SQL_LINK_SERVICE_IDLE_SHARD_MAX_AGE_SECONDS", "120")
    health_file = tmp_path / "governance_health.json"
    health_file.write_text(
        json.dumps(
            {
                "timestamp_utc": shard_manager._now_utc(),
                "overall_status": "ready",
                "sqlite": {"pending_lines": 0, "inserted": 0, "invalid": 1},
                "sqlite_json_files": {"pending_files": 0, "inserted": 0},
            }
        ),
        encoding="utf-8",
    )
    shard = {
        "name": "governance",
        "sqlite_db": tmp_path / "governance.sqlite3",
        "state_file": tmp_path / "governance_state.json",
        "health_file": health_file,
        "journal_file": tmp_path / "governance_journal.jsonl",
        "journal_events_file": tmp_path / "governance_journal_events.jsonl",
        "invalid_log_file": tmp_path / "governance_invalid.jsonl",
        "include_streams": "governance_events",
        "skip_json_files": False,
        "max_files": 1,
        "max_lines_per_file": 100,
        "state_checkpoint_lines": 10,
    }
    calls = {"quarantine": 0, "subprocess": 0}

    def fake_quarantine(**_kwargs):
        calls["quarantine"] += 1
        return {"triggered": False}

    def fake_run(*_args, **_kwargs):
        calls["subprocess"] += 1
        return subprocess.CompletedProcess(args=["link"], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(shard_manager, "_quarantine_shard_artifacts", fake_quarantine)
    monkeypatch.setattr(shard_manager.subprocess, "run", fake_run)

    results = shard_manager._run_shard_links(
        shards=[shard],
        link_mode="sqlite",
        sqlite_timeout_seconds=30,
        sqlite_lock_retries=0,
        sqlite_lock_retry_delay_seconds=0.1,
        shard_link_timeout_seconds=5,
        preprocess_workers=2,
    )

    assert results[0]["rc"] == 0
    assert results[0].get("skipped") is not True
    assert calls == {"quarantine": 1, "subprocess": 1}


def test_run_shard_links_does_not_skip_fresh_idle_when_filters_change(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SQL_LINK_SERVICE_SKIP_FRESH_IDLE_SHARDS", "1")
    monkeypatch.setenv("SQL_LINK_SERVICE_IDLE_SHARD_MAX_AGE_SECONDS", "120")
    health_file = tmp_path / "governance_health.json"
    health_file.write_text(
        json.dumps(
            {
                "timestamp_utc": shard_manager._now_utc(),
                "overall_status": "ready",
                "filters": {
                    "include_streams": ["governance_events"],
                    "path_contains": ["governance/events/auth_events_20260630.jsonl"],
                    "path_not_contains": [],
                },
                "sqlite": {"pending_lines": 0, "inserted": 0},
                "sqlite_json_files": {"pending_files": 0, "inserted": 0},
            }
        ),
        encoding="utf-8",
    )
    shard = {
        "name": "governance",
        "sqlite_db": tmp_path / "governance.sqlite3",
        "state_file": tmp_path / "governance_state.json",
        "health_file": health_file,
        "journal_file": tmp_path / "governance_journal.jsonl",
        "journal_events_file": tmp_path / "governance_journal_events.jsonl",
        "invalid_log_file": tmp_path / "governance_invalid.jsonl",
        "include_streams": "governance_events",
        "path_contains": "governance/events/auth_events_20260630.jsonl,governance/events/write_failures_20260630.jsonl",
        "skip_json_files": False,
        "max_files": 2,
        "max_lines_per_file": 100,
        "state_checkpoint_lines": 10,
    }
    calls = {"quarantine": 0, "subprocess": 0}

    def fake_quarantine(**_kwargs):
        calls["quarantine"] += 1
        return {"triggered": False}

    def fake_run(*_args, **_kwargs):
        calls["subprocess"] += 1
        return subprocess.CompletedProcess(args=["link"], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(shard_manager, "_quarantine_shard_artifacts", fake_quarantine)
    monkeypatch.setattr(shard_manager.subprocess, "run", fake_run)

    results = shard_manager._run_shard_links(
        shards=[shard],
        link_mode="sqlite",
        sqlite_timeout_seconds=30,
        sqlite_lock_retries=0,
        sqlite_lock_retry_delay_seconds=0.1,
        shard_link_timeout_seconds=5,
        preprocess_workers=2,
    )

    assert results[0]["rc"] == 0
    assert results[0].get("skipped") is not True
    assert calls == {"quarantine": 1, "subprocess": 1}


def test_run_shard_links_does_not_skip_stale_decision_catch_up(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SQL_LINK_SERVICE_SKIP_FRESH_IDLE_SHARDS", "1")
    monkeypatch.setenv("SQL_LINK_SERVICE_IDLE_SHARD_MAX_AGE_SECONDS", "120")
    monkeypatch.setenv("SQL_LINK_SERVICE_STALE_DECISION_SOURCE_CATCH_UP", "1")
    health_file = tmp_path / "crypto_health.json"
    health_file.write_text(
        json.dumps(
            {
                "timestamp_utc": shard_manager._now_utc(),
                "overall_status": "ready",
                "sqlite": {"pending_lines": 0, "inserted": 0},
                "sqlite_json_files": {"pending_files": 0, "inserted": 0},
            }
        ),
        encoding="utf-8",
    )
    shard = {
        "name": "crypto_trading",
        "sqlite_db": tmp_path / "crypto.sqlite3",
        "state_file": tmp_path / "crypto_state.json",
        "health_file": health_file,
        "journal_file": tmp_path / "crypto_journal.jsonl",
        "journal_events_file": tmp_path / "crypto_journal_events.jsonl",
        "invalid_log_file": tmp_path / "crypto_invalid.jsonl",
        "include_streams": "decisions,trade_logs",
        "path_contains": "governance/channels/decision/crypto_futures_crypto_schwab/decision_20260604.jsonl",
        "skip_json_files": True,
        "max_files": 1,
        "max_lines_per_file": 100,
        "state_checkpoint_lines": 10,
    }
    calls = {"quarantine": 0, "subprocess": 0}

    def fake_quarantine(**_kwargs):
        calls["quarantine"] += 1
        return {"triggered": False}

    def fake_run(*_args, **_kwargs):
        calls["subprocess"] += 1
        return subprocess.CompletedProcess(args=["link"], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(shard_manager, "_quarantine_shard_artifacts", fake_quarantine)
    monkeypatch.setattr(shard_manager.subprocess, "run", fake_run)

    results = shard_manager._run_shard_links(
        shards=[shard],
        link_mode="sqlite",
        sqlite_timeout_seconds=30,
        sqlite_lock_retries=0,
        sqlite_lock_retry_delay_seconds=0.1,
        shard_link_timeout_seconds=5,
        preprocess_workers=2,
    )

    assert results[0]["rc"] == 0
    assert "skipped" not in results[0]
    assert calls == {"quarantine": 1, "subprocess": 1}


def test_run_shard_links_forwards_sparse_decision_byte_caps(tmp_path, monkeypatch) -> None:
    shard = {
        "name": "crypto_trading",
        "sqlite_db": tmp_path / "crypto.sqlite3",
        "state_file": tmp_path / "crypto_state.json",
        "health_file": tmp_path / "crypto_health.json",
        "journal_file": tmp_path / "crypto_journal.jsonl",
        "journal_events_file": tmp_path / "crypto_journal_events.jsonl",
        "invalid_log_file": tmp_path / "crypto_invalid.jsonl",
        "include_streams": "decisions,trade_logs",
        "path_contains": "governance/channels/decision/crypto_futures_crypto_schwab/",
        "skip_json_files": True,
        "max_files": 2,
        "max_lines_per_file": 12000,
        "max_bytes_per_file": 128 * 1024 * 1024,
        "sqlite_batch_max_bytes": 32 * 1024 * 1024,
        "state_checkpoint_lines": 2000,
    }
    captured: list[list[str]] = []

    def fake_quarantine(**_kwargs):
        return {"triggered": False}

    def fake_run(cmd, *_args, **_kwargs):
        captured.append([str(part) for part in cmd])
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(shard_manager, "_quarantine_shard_artifacts", fake_quarantine)
    monkeypatch.setattr(shard_manager.subprocess, "run", fake_run)

    results = shard_manager._run_shard_links(
        shards=[shard],
        link_mode="sqlite",
        sqlite_timeout_seconds=30,
        sqlite_lock_retries=0,
        sqlite_lock_retry_delay_seconds=0.1,
        shard_link_timeout_seconds=5,
        preprocess_workers=1,
    )

    assert results[0]["rc"] == 0
    assert captured
    cmd = captured[0]
    assert cmd[cmd.index("--max-bytes-per-file") + 1] == str(128 * 1024 * 1024)
    assert cmd[cmd.index("--sqlite-batch-max-bytes") + 1] == str(32 * 1024 * 1024)


def test_timed_out_shard_with_confirmed_rows_is_merge_eligible() -> None:
    result = {
        "rc": 124,
        "timed_out": True,
        "health": {
            "sqlite": {
                "inserted": 2943,
                "pending_lines": 0,
            }
        },
    }

    assert shard_manager._shard_link_merge_eligible(result) is True
    assert shard_manager._shard_link_hard_failed(result) is False


def test_timed_out_shard_without_health_progress_is_hard_failed() -> None:
    result = {
        "rc": 124,
        "timed_out": True,
        "health": {
            "sqlite": {
                "inserted": 0,
                "pending_lines": 12,
            }
        },
    }

    assert shard_manager._shard_link_merge_eligible(result) is False
    assert shard_manager._shard_link_hard_failed(result) is True


def test_missing_shard_db_probe_is_noop_merge_skip(tmp_path: Path) -> None:
    result = shard_manager._probe_shard_merge_state(
        shard_name="crypto_trading",
        shard_db=tmp_path / "missing_shard.sqlite3",
        primary_db=tmp_path / "primary.sqlite3",
        sqlite_timeout_seconds=30,
    )

    assert result["ok"] is True
    assert result["merge_required"] is False
    assert result["skipped"] is True
    assert result["reason"] == "merge_shard_db_missing"


def test_missing_shard_db_direct_merge_is_noop_skip(tmp_path: Path) -> None:
    result = shard_manager._merge_shard_into_primary(
        shard_name="crypto_trading",
        shard_db=tmp_path / "missing_shard.sqlite3",
        primary_db=tmp_path / "primary.sqlite3",
        sqlite_timeout_seconds=30,
    )

    assert result["ok"] is True
    assert result["skipped"] is True
    assert result["reason"] == "merge_shard_db_missing"
    assert result["jsonl_rows_inserted"] == 0
    assert result["json_file_rows_inserted"] == 0


def test_merge_followup_summary_recommends_catch_up_for_capped_or_budgeted_merge() -> None:
    summary = shard_manager._merge_followup_summary(
        merge_results=[
            {"shard": "trading", "merge_capped": True},
            {"shard": "aggressive_trading", "reason": "merge_cycle_budget_exhausted:60.1s"},
        ],
        shard_results=[
            {"shard": "crypto_trading", "rc": 124, "timed_out": True, "health": {"sqlite": {"inserted": 12, "pending_lines": 0}}},
        ],
    )

    assert summary["followup_needed"] is True
    assert summary["catch_up_recommended"] is True
    assert summary["merge_capped_count"] == 1
    assert summary["merge_budget_exhausted_count"] == 1
    assert summary["partial_timeout_shard_count"] == 1
    assert "merge_row_cap_remaining" in summary["followup_reasons"]


def test_build_shards_reads_hourly_hot_retention_overrides(monkeypatch) -> None:
    monkeypatch.setenv("SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_HOT_HOURS", "2")
    monkeypatch.setenv("SQL_LINK_SERVICE_SHARD_EXPLANATIONS_HOT_RETENTION_MAX_ROWS", "900000")

    shards = {
        row["name"]: row
        for row in shard_manager._build_shards(["explanations"])
    }

    assert shards["explanations"]["hot_retention_hot_hours"] == 2
    assert shards["explanations"]["hot_retention_max_rows"] == 900000


def test_build_shards_bounds_governance_tail_work_by_default(monkeypatch) -> None:
    monkeypatch.setattr(shard_manager.ops_data_plane, "load_shard_heat_map", lambda _project_root: {})

    shards = {
        row["name"]: row
        for row in shard_manager._build_shards(["governance"])
    }

    assert shards["governance"]["max_files"] == 10
    assert shards["governance"]["max_bytes_per_file"] == 128 * 1024 * 1024
    assert shards["governance"]["sqlite_batch_max_bytes"] == 32 * 1024 * 1024


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


def test_prioritize_shards_for_linking_moves_sentinel_and_hot_pending_first(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("SQL_LINK_SERVICE_SHARD_ORDER_MODE", raising=False)
    monkeypatch.delenv("SQL_LINK_SERVICE_ADAPTIVE_SHARD_ORDER", raising=False)
    trading_health = tmp_path / "trading_health.json"
    data_health = tmp_path / "data_health.json"
    trading_health.write_text(json.dumps({"sqlite": {"pending_lines": 24000}}), encoding="utf-8")
    data_health.write_text(json.dumps({"sqlite": {"pending_lines": 48000}}), encoding="utf-8")
    shards = [
        {"name": "data", "health_file": data_health, "merge_priority": "low", "merge_to_primary": False},
        {"name": "health_fast", "health_file": tmp_path / "health_fast.json"},
        {"name": "trading", "health_file": trading_health, "last_heat_score": 1.5},
        {"name": "explanations", "health_file": tmp_path / "explanations.json", "merge_priority": "low"},
    ]

    ordered, plan = shard_manager._prioritize_shards_for_linking(shards)

    assert [row["name"] for row in ordered][:2] == ["health_fast", "trading"]
    assert [row["name"] for row in ordered].index("data") < [row["name"] for row in ordered].index("explanations")
    assert plan["enabled"] is True
    assert plan["policy"] == "adaptive_hot_pending_sentinel_first"
    trading_row = next(row for row in plan["priority_rows"] if row["shard"] == "trading")
    assert trading_row["pending_lines"] == 24000


def test_prioritize_shards_for_linking_can_preserve_stable_order(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SQL_LINK_SERVICE_SHARD_ORDER_MODE", "stable")
    shards = [
        {"name": "data", "health_file": tmp_path / "data.json"},
        {"name": "health_fast", "health_file": tmp_path / "health.json"},
        {"name": "trading", "health_file": tmp_path / "trading.json"},
    ]

    ordered, plan = shard_manager._prioritize_shards_for_linking(shards)

    assert [row["name"] for row in ordered] == ["data", "health_fast", "trading"]
    assert plan["enabled"] is False
    assert plan["policy"] == "stable_config_order"


def test_write_service_progress_exposes_shard_link_queue_details(tmp_path, monkeypatch) -> None:
    progress_path = tmp_path / "progress.json"
    monkeypatch.setattr(shard_manager, "PROGRESS_HEALTH", progress_path)
    shards = [{"name": "health_fast"}, {"name": "trading"}, {"name": "data"}]
    shard_results = [
        {"shard": "health_fast", "timed_out": False},
        {"shard": "trading", "timed_out": True},
    ]

    shard_manager._write_service_progress(
        cycle_started_utc="2026-05-26T13:00:00+00:00",
        current_step="shard_linking",
        lock_path=tmp_path / "writer.lock",
        primary_db=tmp_path / "primary.sqlite3",
        shards=shards,
        shard_results=shard_results,
        merge_results=[],
        shard_link_plan={"policy": "adaptive_hot_pending_sentinel_first", "planned_order": ["health_fast", "trading", "data"]},
        active_shard_links=[{"shard": "data", "elapsed_seconds": 12.345, "timeout_seconds": 60, "tail_shard": True}],
    )

    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert payload["planned_shard_count"] == 3
    assert payload["completed_shard_count"] == 2
    assert payload["pending_shard_count"] == 1
    assert payload["pending_shards"] == ["data"]
    assert payload["timed_out_shard_count"] == 1
    assert payload["timed_out_shards"] == ["trading"]
    assert payload["active_shard_count"] == 1
    assert payload["active_shards"] == ["data"]
    assert payload["max_active_shard_elapsed_seconds"] == 12.345
    assert payload["shard_link_plan"]["policy"] == "adaptive_hot_pending_sentinel_first"


def test_build_shards_fails_open_when_heat_map_unavailable(monkeypatch) -> None:
    def _raise(_project_root):
        raise sqlite3.DatabaseError("database disk image is malformed")

    monkeypatch.setattr(shard_manager.ops_data_plane, "load_shard_heat_map", _raise)

    shards = {
        row["name"]: row
        for row in shard_manager._build_shards(["trading"])
    }

    assert shards["trading"]["heat_promotion_candidate"] is False


def test_connect_primary_db_quarantines_malformed_primary_and_recreates(tmp_path, monkeypatch) -> None:
    primary_db = tmp_path / "data" / "jsonl_link.sqlite3"
    primary_db.parent.mkdir(parents=True, exist_ok=True)
    primary_db.write_bytes(b"not a sqlite database")
    Path(f"{primary_db}-wal").write_bytes(b"bad wal")
    Path(f"{primary_db}-shm").write_bytes(b"bad shm")
    monkeypatch.setattr(shard_manager, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(shard_manager, "HEALTH_ROOT", tmp_path / "governance" / "health")

    conn = shard_manager._connect_primary_db(primary_db, sqlite_timeout_seconds=30)
    try:
        assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    finally:
        conn.close()

    recovery = json.loads((tmp_path / "governance" / "health" / "sql_link_primary_recovery_latest.json").read_text(encoding="utf-8"))
    assert recovery["triggered"] is True
    assert recovery["primary_db"] == str(primary_db)
    assert "quarantined_malformed_primary" in recovery["recovery_action"]
    assert len(recovery["moved_paths"]) == 3
    assert primary_db.exists()
    assert list((primary_db.parent / "corrupt_quarantine").glob("primary_*/*.sqlite3"))


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
    assert "governance/channels/risk/" in str(by_name["crypto_governance"]["path_not_contains"])
    assert "default_crypto_schwab" in str(by_name["governance"]["path_not_contains"])
    assert "governance/watchdog/" in str(by_name["governance"]["path_not_contains"])
    assert by_name["support_watchdog"]["include_streams"] == "governance_watchdog"
    assert by_name["crypto_trading"]["include_streams"] == by_name["trading"]["include_streams"]
    assert by_name["crypto_governance"]["include_streams"] == by_name["governance"]["include_streams"]
    assert by_name["crypto_trading"]["max_files"] == 10
    assert by_name["crypto_governance"]["max_files"] == 12


def test_build_shards_ignores_blank_path_filter_overrides(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(shard_manager, "SHARD_DB_ROOT", tmp_path / "sql_link_shards")
    monkeypatch.setattr(shard_manager, "SHARD_STATE_ROOT", tmp_path / "state")
    monkeypatch.setattr(shard_manager, "HEALTH_ROOT", tmp_path / "health")
    monkeypatch.setattr(shard_manager, "EVENT_ROOT", tmp_path / "events")
    monkeypatch.setenv("SQL_LINK_SERVICE_SHARD_CRYPTO_TRADING_PATH_CONTAINS", "")
    monkeypatch.setenv("SQL_LINK_SERVICE_SHARD_TRADING_PATH_NOT_CONTAINS", "")

    shards = shard_manager._build_shards(["trading", "crypto_trading"])
    by_name = {str(row["name"]): row for row in shards}

    assert "shadow_crypto/" in str(by_name["crypto_trading"]["path_contains"])
    assert "default_crypto_schwab" in str(by_name["trading"]["path_not_contains"])


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
