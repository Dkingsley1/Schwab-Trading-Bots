from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_one_numbers_report as one_numbers


def test_data_quality_session_policy_is_strict_during_session(monkeypatch) -> None:
    monkeypatch.setenv("ONE_NUMBERS_SESSION_AWARE_DATA_QUALITY", "1")
    monkeypatch.setenv("ONE_NUMBERS_SESSION_TIMEZONE", "America/New_York")
    monkeypatch.setenv("ONE_NUMBERS_SESSION_START", "09:30")
    monkeypatch.setenv("ONE_NUMBERS_SESSION_END", "16:00")
    monkeypatch.setenv("ONE_NUMBERS_DECISION_STALE_GRACE_SECONDS", "120")
    monkeypatch.setenv("ONE_NUMBERS_GOVERNANCE_STALE_GRACE_SECONDS", "180")
    monkeypatch.setenv("ONE_NUMBERS_OFF_HOURS_STALE_GRACE_SECONDS", "259200")

    now_utc = datetime(2026, 3, 24, 15, 0, tzinfo=timezone.utc)
    policy = one_numbers._data_quality_session_policy(now_utc)

    assert policy["session_open"] is True
    assert policy["mode"] == "session_hours_strict"
    assert policy["decision_grace_seconds"] == 120
    assert policy["governance_grace_seconds"] == 180


def test_data_quality_session_policy_relaxes_after_hours(monkeypatch) -> None:
    monkeypatch.setenv("ONE_NUMBERS_SESSION_AWARE_DATA_QUALITY", "1")
    monkeypatch.setenv("ONE_NUMBERS_SESSION_TIMEZONE", "America/New_York")
    monkeypatch.setenv("ONE_NUMBERS_SESSION_START", "09:30")
    monkeypatch.setenv("ONE_NUMBERS_SESSION_END", "16:00")
    monkeypatch.setenv("ONE_NUMBERS_DECISION_STALE_GRACE_SECONDS", "120")
    monkeypatch.setenv("ONE_NUMBERS_GOVERNANCE_STALE_GRACE_SECONDS", "180")
    monkeypatch.setenv("ONE_NUMBERS_OFF_HOURS_STALE_GRACE_SECONDS", "259200")

    now_utc = datetime(2026, 3, 25, 1, 0, tzinfo=timezone.utc)
    policy = one_numbers._data_quality_session_policy(now_utc)

    assert policy["session_open"] is False
    assert policy["mode"] == "off_hours_relaxed"
    assert policy["decision_grace_seconds"] == 259200
    assert policy["governance_grace_seconds"] == 259200


def test_staleness_penalty_respects_grace_window() -> None:
    assert one_numbers._staleness_penalty(110, 120, 30.0, 20.0) == 0.0
    assert one_numbers._staleness_penalty(150, 120, 30.0, 20.0) == 1.0


def test_blocked_metrics_split_data_and_risk() -> None:
    metrics = one_numbers._blocked_metrics(
        {"BLOCKED": 30, "DATA_ONLY_BLOCKED": 50},
        100,
        observe_only_data_blocked_total=20,
    )

    assert metrics["risk_blocked_total"] == 30
    assert metrics["raw_data_blocked_total"] == 50
    assert metrics["observe_only_data_blocked_total"] == 20
    assert metrics["data_blocked_total"] == 30
    assert metrics["combined_blocked_total"] == 60
    assert metrics["risk_blocked_rate"] == 0.30
    assert metrics["raw_data_blocked_rate"] == 0.50
    assert metrics["observe_only_data_blocked_rate"] == 0.20
    assert metrics["data_blocked_rate"] == 0.30
    assert abs(float(metrics["effective_blocked_rate"]) - 0.375) < 1e-9


def test_resolve_sqlite_state_prefers_shard_progress_over_legacy(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(one_numbers, "PROJECT_ROOT", tmp_path)

    shard_root = tmp_path / "governance" / "sql_link_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    legacy_path = tmp_path / "governance" / "jsonl_sql_link_state.json"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)

    rel = "decision_explanations/shadow_crypto/decision_explanations_20260327.jsonl"
    trading_state = {
        "sqlite": {
            rel: {
                "last_line": 25,
                "last_offset_bytes": 2500,
                "mtime": 2.0,
                "file_size_bytes": 2500,
            }
        }
    }
    legacy_state = {
        "sqlite": {
            rel: {
                "last_line": 10,
                "last_offset_bytes": 1000,
                "mtime": 1.0,
                "file_size_bytes": 1000,
            }
        }
    }

    (shard_root / "jsonl_sql_link_state_trading.json").write_text(json.dumps(trading_state), encoding="utf-8")
    legacy_path.write_text(json.dumps(legacy_state), encoding="utf-8")

    sqlite_state = one_numbers._resolve_sqlite_state(tmp_path)

    assert rel in sqlite_state
    assert sqlite_state[rel]["last_line"] == 25


def test_register_sql_snapshot_returns_warning_on_locked_db() -> None:
    class LockedConn:
        def execute(self, *_args, **_kwargs):
            raise one_numbers.sqlite3.OperationalError("database is locked")

        def commit(self):
            raise AssertionError("commit should not run on locked db")

    ok, warning = one_numbers._register_sql_snapshot(
        LockedConn(),
        generated_utc="2026-03-27T11:00:00+00:00",
        day="20260327",
        decision_total_rows=1,
        stocks_decision_rows=1,
        crypto_decision_rows=0,
        watchdog_restarts=0,
        data_quality_score=95.9,
        alerts=[],
        metric_map={},
    )

    assert ok is False
    assert "locked" in warning.lower()


def test_freshest_json_payload_prefers_sql_link_progress_file(tmp_path: Path) -> None:
    latest = tmp_path / "sql_link_service_latest.json"
    progress = tmp_path / "sql_link_service_progress_latest.json"
    now = datetime.now(timezone.utc)

    latest.write_text(
        json.dumps({"timestamp_utc": (now.replace(microsecond=0)).isoformat(), "status": "ok"}),
        encoding="utf-8",
    )
    progress.write_text(
        json.dumps({"timestamp_utc": (now.replace(microsecond=0) + one_numbers.timedelta(minutes=5)).isoformat(), "status": "running"}),
        encoding="utf-8",
    )

    payload, path = one_numbers._freshest_json_payload([progress, latest])

    assert path == progress
    assert payload["status"] == "running"


def test_lightweight_main_uses_daily_summary_without_sqlite(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(one_numbers, "PROJECT_ROOT", tmp_path)
    out_dir = tmp_path / "exports" / "one_numbers"
    out_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "governance" / "health").mkdir(parents=True, exist_ok=True)
    (tmp_path / "exports" / "sql_reports").mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)

    (tmp_path / "governance" / "health" / "daily_runtime_summary_latest.json").write_text(
        json.dumps(
            {
                "day": "20260331",
                "watchdog": {"restarts": 1, "throttled": 0, "restart_errors": 0},
                "decision": {
                    "rows": 100,
                    "observe_only_data_blocked": 20,
                    "status_counts": {"DATA_ONLY_BLOCKED": 20, "BLOCKED": 5},
                    "stale_windows": 1,
                    "files": ["/tmp/decision_a.jsonl", "/tmp/decision_b.jsonl"],
                },
                "governance": {
                    "rows": 10,
                    "stale_windows": 2,
                    "files": ["/tmp/gov_a.jsonl"],
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "exports" / "sql_reports" / "daily_runtime_summary_20260330.json").write_text(
        json.dumps(
            {
                "day": "20260330",
                "watchdog": {"restarts": 0, "throttled": 0, "restart_errors": 0},
                "decision": {
                    "rows": 50,
                    "observe_only_data_blocked": 10,
                    "status_counts": {"DATA_ONLY_BLOCKED": 10, "BLOCKED": 2},
                    "stale_windows": 0,
                    "files": ["/tmp/decision_prev.jsonl"],
                },
                "governance": {"rows": 5, "stale_windows": 0, "files": ["/tmp/gov_prev.jsonl"]},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "governance" / "health" / "paper_performance_latest.json").write_text(
        json.dumps(
            {
                "history_daily_series": [
                    {
                        "day_utc": "20260331",
                        "executions": 77,
                        "ending_net_pnl_total": 12.34,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "governance" / "health" / "one_numbers_latest.json").write_text(
        json.dumps({"stocks_top_symbol_1": "SPY:10"}),
        encoding="utf-8",
    )

    def _fail_connect(*_args, **_kwargs):
        raise AssertionError("sqlite should not be opened in lightweight mode")

    monkeypatch.setattr(one_numbers.sqlite3, "connect", _fail_connect)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_one_numbers_report.py",
            "--day",
            "20260331",
            "--out-dir",
            str(out_dir),
            "--lightweight",
            "--no-sql-write",
        ],
    )

    rc = one_numbers.main()
    payload = json.loads((tmp_path / "governance" / "health" / "one_numbers_latest.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["day_utc"] == "20260331"
    assert payload["combined_decision_total_rows"] == "100"
    assert payload["data_blocked_total"] == "0"
    assert payload["risk_blocked_total"] == "5"
    assert payload["paper_executed_total"] == "77"
    assert payload["combined_pnl_proxy"] == "12.340000"
    assert payload["report_mode"] == "lightweight_cached"
