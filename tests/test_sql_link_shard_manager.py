import sqlite3
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.ops.sql_link_shard_manager as shard_manager


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
                "crypto_trading_fast",
                "crypto_explanations",
                "crypto_shadow_attribution",
                "crypto_trading",
                "trading_fast",
                "explanations",
                "shadow_attribution",
                "aggressive_trading",
                "trading",
            ]
        )
    }

    assert shards["health_fast"]["skip_json_files"] is False
    assert "data_ingress_latest_" in str(shards["health_fast"]["path_contains"])
    assert shards["crypto_trading_fast"]["include_streams"] == "paper_broker_bridge,top_level_trade_links"
    assert shards["crypto_explanations"]["include_streams"] == "decision_explanations"
    assert "shadow_pnl_attribution_" in str(shards["crypto_shadow_attribution"]["path_contains"])
    assert shards["trading_fast"]["include_streams"] == "paper_broker_bridge,top_level_trade_links"
    assert shards["explanations"]["include_streams"] == "decision_explanations"
    assert "shadow_pnl_attribution_" in str(shards["shadow_attribution"]["path_contains"])
    assert shards["aggressive_trading"]["include_streams"] == "decisions,trade_logs"
    assert "shadow_intraday_aggressive_" in str(shards["aggressive_trading"]["path_contains"])
    assert shards["crypto_trading"]["include_streams"] == "decisions,trade_logs"
    assert shards["trading"]["include_streams"] == "decisions,trade_logs"
    assert "shadow_intraday_aggressive_" in str(shards["trading"]["path_not_contains"])


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


def test_hot_retention_requires_growth_after_successful_run() -> None:
    reasons = shard_manager._hot_retention_trigger_reasons(
        db_size_gb=232.0,
        max_db_gb=25.0,
        db_growth_gb=0.8,
        growth_trigger_gb=12.0,
        rows_since_last_run=500000,
        row_trigger=2500000,
        has_successful_run=True,
    )

    assert reasons == []


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

    shards = shard_manager._build_shards(["trading", "crypto_trading", "governance", "crypto_governance", "data"])
    by_name = {str(row["name"]): row for row in shards}

    assert "crypto_trading" in by_name
    assert "crypto_governance" in by_name
    assert "shadow_crypto/" in str(by_name["crypto_trading"]["path_contains"])
    assert "shadow_crypto/" in str(by_name["trading"]["path_not_contains"])
    assert "default_crypto_coinbase" in str(by_name["crypto_governance"]["path_contains"])
    assert "default_crypto_schwab" in str(by_name["governance"]["path_not_contains"])
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


def test_normalized_shard_config_upgrades_legacy_default() -> None:
    assert shard_manager._normalized_shard_config("") == shard_manager.CURRENT_DEFAULT_SHARDS
    assert shard_manager._normalized_shard_config("trading,governance,data") == shard_manager.CURRENT_DEFAULT_SHARDS
    assert shard_manager._normalized_shard_config("trading,governance,data,custom") == "trading,governance,data,custom"
