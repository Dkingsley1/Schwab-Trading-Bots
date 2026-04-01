import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts import session_ready_check as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_profile_heartbeat_uses_runtime_checkpoint_fallback(tmp_path: Path) -> None:
    original_root = src.PROJECT_ROOT
    original_db = src.DB_PATH
    try:
        src.PROJECT_ROOT = tmp_path
        src.DB_PATH = tmp_path / "data" / "jsonl_link.sqlite3"
        now = datetime.now(timezone.utc)
        _write_json(
            tmp_path / "governance" / "shadow_crypto" / "runtime_checkpoint.json",
            {"timestamp_utc": (now - timedelta(seconds=15)).isoformat()},
        )

        ok, details = src._profile_heartbeat_ok("crypto", 120.0)

        assert ok
        assert details.startswith("age_sec=")
    finally:
        src.PROJECT_ROOT = original_root
        src.DB_PATH = original_db


def test_resolve_expected_profiles_auto_prefers_recent_activity(tmp_path: Path) -> None:
    original_root = src.PROJECT_ROOT
    original_db = src.DB_PATH
    try:
        src.PROJECT_ROOT = tmp_path
        src.DB_PATH = tmp_path / "data" / "jsonl_link.sqlite3"
        now = datetime.now(timezone.utc)
        _write_json(
            tmp_path / "governance" / "shadow_crypto" / "runtime_checkpoint.json",
            {"timestamp_utc": (now - timedelta(seconds=30)).isoformat()},
        )
        _write_json(
            tmp_path / "governance" / "shadow_aggressive_equities" / "runtime_checkpoint.json",
            {"timestamp_utc": (now - timedelta(hours=8)).isoformat()},
        )

        activity = src._profile_activity_map()
        profiles = src._resolve_expected_profiles("auto", activity, 300.0)

        assert profiles == ["crypto"]
    finally:
        src.PROJECT_ROOT = original_root
        src.DB_PATH = original_db


def test_resolve_expected_profiles_auto_ignores_stale_checkpoint_only_profiles(tmp_path: Path) -> None:
    original_root = src.PROJECT_ROOT
    original_db = src.DB_PATH
    try:
        src.PROJECT_ROOT = tmp_path
        src.DB_PATH = tmp_path / "data" / "jsonl_link.sqlite3"
        now = datetime.now(timezone.utc)
        _write_json(
            tmp_path / "governance" / "shadow_crypto" / "runtime_checkpoint.json",
            {"timestamp_utc": (now - timedelta(seconds=420)).isoformat()},
        )
        _write_json(
            tmp_path / "governance" / "health" / "shadow_loop_default_crypto_coinbase_1.json",
            {"timestamp_utc": (now - timedelta(seconds=30)).isoformat()},
        )

        activity = src._profile_activity_map()
        profiles = src._resolve_expected_profiles("auto", activity, 300.0)

        assert profiles == ["default"]
    finally:
        src.PROJECT_ROOT = original_root
        src.DB_PATH = original_db


def test_resolve_expected_profiles_auto_ignores_stale_heartbeat_profiles(tmp_path: Path) -> None:
    original_root = src.PROJECT_ROOT
    original_db = src.DB_PATH
    try:
        src.PROJECT_ROOT = tmp_path
        src.DB_PATH = tmp_path / "data" / "jsonl_link.sqlite3"
        now = datetime.now(timezone.utc)
        _write_json(
            tmp_path / "governance" / "health" / "shadow_loop_aggressive_equities_schwab_1.json",
            {"timestamp_utc": (now - timedelta(seconds=420)).isoformat()},
        )
        _write_json(
            tmp_path / "governance" / "health" / "shadow_loop_default_crypto_coinbase_1.json",
            {"timestamp_utc": (now - timedelta(seconds=30)).isoformat()},
        )

        activity = src._profile_activity_map()
        profiles = src._resolve_expected_profiles("auto", activity, 300.0)

        assert profiles == ["default"]
    finally:
        src.PROJECT_ROOT = original_root
        src.DB_PATH = original_db


def test_command_invokes_target_ignores_watchdog_embedded_start_cmd() -> None:
    target = "scripts/run_parallel_shadows.py"
    direct = (
        "/opt/homebrew/bin/python3 "
        "/Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/run_parallel_shadows.py --simulate"
    )
    watchdog = (
        "/opt/homebrew/bin/python3 "
        "/Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/shadow_watchdog.py "
        "--schwab-start-cmd /Users/dankingsley/PycharmProjects/schwab_trading_bot/.venv312/bin/python "
        "/Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/run_parallel_shadows.py"
    )

    assert src._command_invokes_target(direct, target) is True
    assert src._command_invokes_target(watchdog, target) is False


def test_sql_writable_uses_nonblocking_probe_file(tmp_path: Path) -> None:
    original_db = src.DB_PATH
    try:
        db_path = tmp_path / "data" / "jsonl_link.sqlite3"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE sample(id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

        src.DB_PATH = db_path

        assert src._sql_writable() is True
        assert not (db_path.parent / ".session_ready_write_probe").exists()
    finally:
        src.DB_PATH = original_db
