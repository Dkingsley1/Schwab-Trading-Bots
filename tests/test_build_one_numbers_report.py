from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
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


def test_session_aligned_recent_cutoff_uses_session_open(monkeypatch) -> None:
    monkeypatch.setenv("ONE_NUMBERS_SESSION_AWARE_DATA_QUALITY", "1")
    monkeypatch.setenv("ONE_NUMBERS_SESSION_TIMEZONE", "America/New_York")
    monkeypatch.setenv("ONE_NUMBERS_SESSION_START", "09:30")
    monkeypatch.setenv("ONE_NUMBERS_SESSION_END", "16:00")

    now_utc = datetime(2026, 4, 6, 14, 0, tzinfo=timezone.utc)
    cutoff = one_numbers._session_aligned_recent_cutoff(now_utc)

    assert cutoff == datetime(2026, 4, 6, 13, 30, tzinfo=timezone.utc)


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


def test_default_db_path_uses_local_fallback_for_broken_routed_symlink(tmp_path: Path, monkeypatch) -> None:
    routed_db = tmp_path / "data" / "jsonl_link.sqlite3"
    missing_external_db = tmp_path / "missing_bot_logs" / "data" / "jsonl_link.sqlite3"
    fallback_db = tmp_path / "local_fallback_storage" / "data" / "jsonl_link.sqlite3"
    routed_db.parent.mkdir(parents=True, exist_ok=True)
    routed_db.symlink_to(missing_external_db)
    fallback_db.parent.mkdir(parents=True, exist_ok=True)
    fallback_db.write_bytes(b"sqlite")
    monkeypatch.setattr(one_numbers, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(one_numbers, "DEFAULT_DB", routed_db)
    monkeypatch.setattr(one_numbers, "LOCAL_FALLBACK_ROOT", tmp_path / "local_fallback_storage")
    monkeypatch.delenv("SQL_LINK_SERVICE_PRIMARY_DB", raising=False)

    assert one_numbers._default_db_path() == fallback_db
    assert one_numbers._routed_or_local_fallback_path(routed_db) == fallback_db


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


def test_lightweight_metrics_ignore_preopen_stale_windows(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ONE_NUMBERS_SESSION_AWARE_DATA_QUALITY", "1")
    monkeypatch.setenv("ONE_NUMBERS_SESSION_TIMEZONE", "America/New_York")
    monkeypatch.setenv("ONE_NUMBERS_SESSION_START", "09:30")
    monkeypatch.setenv("ONE_NUMBERS_SESSION_END", "16:00")
    monkeypatch.setenv("ONE_NUMBERS_STALE_SECONDS", "180")

    decision_path = tmp_path / "decision.jsonl"
    governance_path = tmp_path / "governance.jsonl"
    now_utc = datetime(2026, 4, 6, 14, 0, tzinfo=timezone.utc)
    preopen = datetime(2026, 4, 6, 12, 0, tzinfo=timezone.utc)
    session_samples = [
        datetime(2026, 4, 6, 13, 31, tzinfo=timezone.utc),
        datetime(2026, 4, 6, 13, 33, tzinfo=timezone.utc),
        datetime(2026, 4, 6, 13, 35, tzinfo=timezone.utc),
    ]

    decision_rows = [
        {"timestamp_utc": preopen.isoformat(), "status": "SHADOW_ONLY"},
        {"timestamp_utc": (preopen + timedelta(minutes=5)).isoformat(), "status": "SHADOW_ONLY"},
    ] + [{"timestamp_utc": stamp.isoformat(), "status": "SHADOW_ONLY"} for stamp in session_samples]
    governance_rows = [
        {"timestamp_utc": preopen.isoformat()},
        {"timestamp_utc": (preopen + timedelta(minutes=5)).isoformat()},
    ] + [{"timestamp_utc": stamp.isoformat()} for stamp in session_samples]

    decision_path.write_text("\n".join(json.dumps(row) for row in decision_rows) + "\n", encoding="utf-8")
    governance_path.write_text("\n".join(json.dumps(row) for row in governance_rows) + "\n", encoding="utf-8")

    metrics = one_numbers._lightweight_metrics_from_daily_summary(
        {
            "decision": {
                "rows": len(decision_rows),
                "observe_only_data_blocked": 0,
                "status_counts": {"SHADOW_ONLY": len(decision_rows)},
                "stale_windows": 2,
                "files": [str(decision_path)],
            },
            "governance": {
                "rows": len(governance_rows),
                "stale_windows": 2,
                "files": [str(governance_path)],
            },
            "watchdog": {"restarts": 0, "throttled": 0, "restart_errors": 0},
        },
        {},
        now_utc=now_utc,
    )

    assert metrics["decision_stale_windows_4h"] == 0
    assert metrics["governance_stale_windows_4h"] == 0
    assert metrics["data_quality_score"] == 100.0


def test_lightweight_flag_runs_full_sqlite_backed_report(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(one_numbers, "PROJECT_ROOT", tmp_path)
    out_dir = tmp_path / "exports" / "one_numbers"
    db_path = tmp_path / "data" / "jsonl_link.sqlite3"
    out_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "governance" / "health").mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = one_numbers.sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE jsonl_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT NOT NULL,
            source_rel TEXT NOT NULL,
            line_no INTEGER NOT NULL,
            ingested_at TEXT NOT NULL,
            payload_sha1 TEXT NOT NULL,
            payload_json TEXT NOT NULL
        )
        """
    )
    conn.executemany(
        """
        INSERT INTO jsonl_records (source_file, source_rel, line_no, ingested_at, payload_sha1, payload_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "decision_explanations_20260331.jsonl",
                "decision_explanations/shadow_default/decision_explanations_20260331.jsonl",
                1,
                "2026-03-31T15:00:00+00:00",
                "sha1-a",
                json.dumps(
                    {
                        "timestamp_utc": "2026-03-31T15:00:00+00:00",
                        "action": "BUY",
                        "status": "PAPER_EXECUTED",
                        "symbol": "SPY",
                        "bot_id": "brain_refinery_v1",
                    }
                ),
            ),
            (
                "master_control_20260331.jsonl",
                "governance/shadow_default/master_control_20260331.jsonl",
                1,
                "2026-03-31T15:00:01+00:00",
                "sha1-b",
                json.dumps({"timestamp_utc": "2026-03-31T15:00:01+00:00", "master_action": "BUY"}),
            ),
        ],
    )
    conn.commit()
    conn.close()
    (tmp_path / "governance" / "health" / "ingestion_storage_control_latest.json").write_text(
        json.dumps(
            {
                "pressure_index": 0.042,
                "backpressure": {
                    "core_pending_lines": 605,
                    "estimated_total_drain_minutes": 0.706,
                    "stale_stage_pending_lines": 0,
                },
                "steady_state": {
                    "quality_score": 97.4,
                    "quality_label": "excellent",
                    "target_status": {
                        "steady_state_ready": True,
                        "target_breach_count": 0,
                        "target_breaches": [],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_one_numbers_report.py",
            "--day",
            "20260331",
            "--out-dir",
            str(out_dir),
            "--db",
            str(db_path),
            "--lightweight",
            "--no-sql-write",
        ],
    )

    rc = one_numbers.main()
    payload = json.loads((tmp_path / "governance" / "health" / "one_numbers_latest.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["day_utc"] == "20260331"
    assert payload["report_mode"] == "full"
    assert payload["combined_decision_total_rows"] == "1"
    assert payload["data_blocked_total"] == "0"
    assert payload["risk_blocked_total"] == "0"
    assert payload["paper_executed_total"] == "1"
    assert payload["backpressure_quality_score"] == "97.40"
    assert payload["pressure_index"] == "0.042"
    assert payload["core_pending_lines"] == "605"
    latest_csv = out_dir / "latest.csv"
    latest_metrics_csv = out_dir / "latest_metrics.csv"
    latest_md = out_dir / "latest.md"
    latest_xlsx = out_dir / "latest.xlsx"
    assert latest_csv.is_symlink()
    assert latest_metrics_csv.is_symlink()
    assert latest_md.is_symlink()
    assert latest_xlsx.is_symlink()
    latest_csv_text = latest_csv.read_text(encoding="utf-8")
    latest_metrics_csv_text = latest_metrics_csv.read_text(encoding="utf-8")
    latest_md_text = latest_md.read_text(encoding="utf-8")
    assert "label,value" in latest_csv_text
    assert "Report Metadata," in latest_csv_text
    assert "Report Day (UTC),20260331" in latest_csv_text
    assert "Report Mode,full" in latest_csv_text
    assert "Combined Decision Total Rows,1" in latest_csv_text
    assert "Month To Date," in latest_csv_text
    assert "Backpressure Scorecard," in latest_csv_text
    assert "Backpressure Quality Score,97.40" in latest_csv_text
    assert "section,label,value,metric" in latest_metrics_csv_text
    assert "Backpressure Scorecard,Backpressure Quality Score,97.40,backpressure_quality_score" in latest_metrics_csv_text
    assert "## Combined" in latest_md_text
    assert "## Backpressure Scorecard" in latest_md_text
    assert "- Backpressure quality score: 97.40/100" in latest_md_text
    assert "Report mode: lightweight" not in latest_md_text


def test_full_main_reports_trade_decision_day_when_explanations_lag(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(one_numbers, "PROJECT_ROOT", tmp_path)
    out_dir = tmp_path / "exports" / "one_numbers"
    db_path = tmp_path / "data" / "jsonl_link.sqlite3"
    out_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "governance" / "health").mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = one_numbers.sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE jsonl_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT NOT NULL,
            source_rel TEXT NOT NULL,
            line_no INTEGER NOT NULL,
            ingested_at TEXT NOT NULL,
            payload_sha1 TEXT NOT NULL,
            payload_json TEXT NOT NULL
        )
        """
    )
    conn.executemany(
        """
        INSERT INTO jsonl_records (source_file, source_rel, line_no, ingested_at, payload_sha1, payload_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "decision_explanations_20260330.jsonl",
                "decision_explanations/shadow_default/decision_explanations_20260330.jsonl",
                1,
                "2026-03-30T15:00:00+00:00",
                "sha1-a",
                json.dumps({"timestamp_utc": "2026-03-30T15:00:00+00:00", "action": "HOLD", "status": "SHADOW_ONLY", "symbol": "SPY"}),
            ),
            (
                "trade_decisions_20260331.jsonl",
                "decisions/shadow_default/trade_decisions_20260331.jsonl",
                1,
                "2026-03-31T15:00:00+00:00",
                "sha1-b",
                json.dumps({"timestamp_utc": "2026-03-31T15:00:00+00:00", "action": "BUY", "decision": "EXECUTE", "symbol": "QQQ"}),
            ),
            (
                "master_control_20260331.jsonl",
                "governance/shadow_default/master_control_20260331.jsonl",
                1,
                "2026-03-31T15:00:01+00:00",
                "sha1-c",
                json.dumps({"timestamp_utc": "2026-03-31T15:00:01+00:00", "master_action": "BUY"}),
            ),
        ],
    )
    conn.commit()
    conn.close()
    (tmp_path / "governance" / "health" / "ingestion_storage_control_latest.json").write_text(
        json.dumps({"pressure_index": 0.1, "backpressure": {}, "steady_state": {"quality_score": 99.0, "quality_label": "excellent", "target_status": {}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_one_numbers_report.py",
            "--day",
            "20260331",
            "--out-dir",
            str(out_dir),
            "--db",
            str(db_path),
            "--no-sql-write",
        ],
    )

    rc = one_numbers.main()
    payload = json.loads((tmp_path / "governance" / "health" / "one_numbers_latest.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["requested_day"] == "20260331"
    assert payload["resolved_day"] == "20260331"
    assert payload["day_fallback_applied"] == "false"
    assert payload["combined_decision_total_rows"] == "1"
    assert payload["detail_source"] == "trade_decision_fallback"
    assert payload["stocks_decision_rows"] == "1"


def test_full_main_ignores_stale_explanation_state_when_trade_rows_are_indexed(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(one_numbers, "PROJECT_ROOT", tmp_path)
    out_dir = tmp_path / "exports" / "one_numbers"
    db_path = tmp_path / "data" / "jsonl_link.sqlite3"
    health_dir = tmp_path / "governance" / "health"
    out_dir.mkdir(parents=True, exist_ok=True)
    health_dir.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = one_numbers.sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE jsonl_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT NOT NULL,
            source_rel TEXT NOT NULL,
            line_no INTEGER NOT NULL,
            ingested_at TEXT NOT NULL,
            payload_sha1 TEXT NOT NULL,
            payload_json TEXT NOT NULL
        )
        """
    )
    trade_rel = "decisions/shadow_default/trade_decisions_20260331.jsonl"
    conn.execute(
        """
        INSERT INTO jsonl_records (source_file, source_rel, line_no, ingested_at, payload_sha1, payload_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "trade_decisions_20260331.jsonl",
            trade_rel,
            1,
            "2026-03-31T15:00:00+00:00",
            "sha1-trade",
            json.dumps({"timestamp_utc": "2026-03-31T15:00:00+00:00", "action": "BUY", "decision": "EXECUTE", "symbol": "SPY"}),
        ),
    )
    conn.commit()
    conn.close()
    stale_explanation_rel = "decision_explanations/shadow_default/decision_explanations_20260331.jsonl"
    (tmp_path / "governance" / "jsonl_sql_link_state.json").write_text(
        json.dumps({"sqlite": {stale_explanation_rel: {}, trade_rel: {}}}),
        encoding="utf-8",
    )
    (health_dir / "ingestion_storage_control_latest.json").write_text(
        json.dumps({"pressure_index": 0.1, "backpressure": {}, "steady_state": {"quality_score": 99.0}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_one_numbers_report.py",
            "--day",
            "20260331",
            "--out-dir",
            str(out_dir),
            "--db",
            str(db_path),
            "--no-sql-write",
        ],
    )

    rc = one_numbers.main()
    payload = json.loads((health_dir / "one_numbers_latest.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["combined_decision_total_rows"] == "1"
    assert payload["decision_source_files"] == "1"
    assert payload["detail_source"] == "trade_decision_fallback"


def test_resolve_report_day_prefers_latest_decision_day_over_governance_only_today() -> None:
    sqlite_state = {
        "decision_explanations/shadow_default/decision_explanations_20260330.jsonl": {},
        "governance/shadow_default/master_control_20260331.jsonl": {},
    }

    day, sources = one_numbers._resolve_report_day("20260331", sqlite_state)

    assert day == "20260330"
    assert sources["decision"] == ["decision_explanations/shadow_default/decision_explanations_20260330.jsonl"]


def test_resolve_report_day_uses_trade_decisions_when_explanations_lag() -> None:
    sqlite_state = {
        "decision_explanations/shadow_default/decision_explanations_20260330.jsonl": {},
        "decisions/shadow_default/trade_decisions_20260331.jsonl": {},
        "governance/shadow_default/master_control_20260331.jsonl": {},
    }

    day, sources = one_numbers._resolve_report_day("20260331", sqlite_state)

    assert day == "20260331"
    assert sources["decision"] == []
    assert sources["decision_trade"] == ["decisions/shadow_default/trade_decisions_20260331.jsonl"]


def test_resolve_lightweight_report_day_prefers_latest_decision_day() -> None:
    history = {
        "20260330": {"decision": {"rows": 125}, "governance": {"rows": 18}},
        "20260331": {"decision": {"rows": 0}, "governance": {"rows": 20}},
    }

    day = one_numbers._resolve_lightweight_report_day("20260331", history)

    assert day == "20260330"


def test_resolve_raw_jsonl_report_day_prefers_exact_governance_source_day(tmp_path: Path) -> None:
    decision_root = tmp_path / "decision_explanations" / "shadow_default"
    governance_root = tmp_path / "governance" / "shadow_default"
    decision_root.mkdir(parents=True, exist_ok=True)
    governance_root.mkdir(parents=True, exist_ok=True)
    (decision_root / "decision_explanations_20260330.jsonl").write_text("{}\n", encoding="utf-8")
    (governance_root / "master_control_20260331.jsonl").write_text("{}\n", encoding="utf-8")

    day = one_numbers._resolve_raw_jsonl_report_day(tmp_path, "20260331")

    assert day == "20260331"


def test_requested_source_day_history_entry_covers_governance_only_requested_day(tmp_path: Path) -> None:
    governance_root = tmp_path / "governance" / "shadow_default"
    governance_root.mkdir(parents=True, exist_ok=True)
    (governance_root / "master_control_20260331.jsonl").write_text('{"status":"ready"}\n', encoding="utf-8")
    entries: dict[str, dict[str, object]] = {
        "20260330": {
            "day_utc": "20260330",
            "metrics": {"combined_decision_total_rows": 12},
        }
    }

    one_numbers._add_requested_source_day_history_entry(
        tmp_path,
        entries,
        requested_day="20260331",
        resolved_day="20260330",
        now_utc=datetime(2026, 3, 31, 20, 0, tzinfo=timezone.utc),
    )

    assert entries["20260331"]["report_mode"] == "source_day_coverage_stub"
    metrics = entries["20260331"]["metrics"]
    assert metrics["combined_decision_total_rows"] == 0
    assert metrics["combined_governance_total_rows"] == 1


def test_latest_report_day_from_db_prefers_latest_decision_day(tmp_path: Path) -> None:
    db_path = tmp_path / "jsonl_link.sqlite3"
    conn = one_numbers.sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE jsonl_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT NOT NULL,
            source_rel TEXT NOT NULL,
            line_no INTEGER NOT NULL,
            ingested_at TEXT NOT NULL,
            payload_sha1 TEXT NOT NULL,
            payload_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO jsonl_records (source_file, source_rel, line_no, ingested_at, payload_sha1, payload_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "decision_explanations_20260330.jsonl",
            "decision_explanations/shadow_default/decision_explanations_20260330.jsonl",
            1,
            "2026-03-30T15:00:00+00:00",
            "sha1-a",
            "{}",
        ),
    )
    conn.execute(
        """
        INSERT INTO jsonl_records (source_file, source_rel, line_no, ingested_at, payload_sha1, payload_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "master_control_20260331.jsonl",
            "governance/shadow_default/master_control_20260331.jsonl",
            1,
            "2026-03-31T15:00:00+00:00",
            "sha1-b",
            "{}",
        ),
    )
    conn.commit()

    day = one_numbers._latest_report_day_from_db(conn, "20260331")

    conn.close()
    assert day == "20260330"


def test_latest_report_day_from_db_uses_trade_decisions_when_explanations_lag(tmp_path: Path) -> None:
    db_path = tmp_path / "jsonl_link.sqlite3"
    conn = one_numbers.sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE jsonl_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT NOT NULL,
            source_rel TEXT NOT NULL,
            line_no INTEGER NOT NULL,
            ingested_at TEXT NOT NULL,
            payload_sha1 TEXT NOT NULL,
            payload_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO jsonl_records (source_file, source_rel, line_no, ingested_at, payload_sha1, payload_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "decision_explanations_20260330.jsonl",
            "decision_explanations/shadow_default/decision_explanations_20260330.jsonl",
            1,
            "2026-03-30T15:00:00+00:00",
            "sha1-a",
            "{}",
        ),
    )
    conn.execute(
        """
        INSERT INTO jsonl_records (source_file, source_rel, line_no, ingested_at, payload_sha1, payload_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "trade_decisions_20260331.jsonl",
            "decisions/shadow_default/trade_decisions_20260331.jsonl",
            1,
            "2026-03-31T15:00:00+00:00",
            "sha1-b",
            "{}",
        ),
    )
    conn.commit()

    day = one_numbers._latest_report_day_from_db(conn, "20260331")

    conn.close()
    assert day == "20260331"


def test_prefer_db_report_day_repairs_stale_shard_fallback() -> None:
    selected = one_numbers._prefer_db_report_day(
        "20260423",
        "20260422",
        "20260423",
    )

    assert selected == "20260423"


def test_one_numbers_coverage_metadata_marks_unpinned_history_incomplete(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ONE_NUMBERS_ORIGINAL_START_DAY", raising=False)
    monkeypatch.delenv("ONE_NUMBERS_EXPECTED_START_DAY", raising=False)
    monkeypatch.delenv("INFRA_SUPERVISOR_ONE_NUMBERS_START_DAY", raising=False)
    decision_root = tmp_path / "decision_explanations" / "shadow_default"
    decision_root.mkdir(parents=True, exist_ok=True)
    (decision_root / "decision_explanations_20260422.jsonl").write_text("{}\n", encoding="utf-8")
    (decision_root / "decision_explanations_20260423.jsonl").write_text("{}\n", encoding="utf-8")
    history = {
        "20260422": {"metrics": {"combined_decision_total_rows": 10}},
    }

    metadata = one_numbers._one_numbers_coverage_metadata(tmp_path, history)

    assert metadata["historical_coverage_status"] == "degraded"
    assert metadata["all_time_coverage_complete"] == "false"
    assert metadata["source_days_discovered"] == "2"
    assert metadata["source_days_missing_from_rollup_count"] == "1"
    assert "original start day is not pinned" in metadata["historical_coverage_detail"]


def test_one_numbers_coverage_metadata_ready_when_start_and_sources_are_covered(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ONE_NUMBERS_ORIGINAL_START_DAY", "20260422")
    decision_root = tmp_path / "decision_explanations" / "shadow_default"
    decision_root.mkdir(parents=True, exist_ok=True)
    (decision_root / "decision_explanations_20260422.jsonl").write_text("{}\n", encoding="utf-8")
    (decision_root / "decision_explanations_20260423.jsonl").write_text("{}\n", encoding="utf-8")
    history = {
        "20260422": {"metrics": {"combined_decision_total_rows": 10}},
        "20260423": {"metrics": {"combined_decision_total_rows": 20}},
    }

    metadata = one_numbers._one_numbers_coverage_metadata(tmp_path, history)

    assert metadata["historical_coverage_status"] == "ready"
    assert metadata["all_time_coverage_complete"] == "true"
    assert metadata["earliest_rollup_day"] == "20260422"
    assert metadata["latest_rollup_day"] == "20260423"


def test_model_drift_snapshot_requires_minimum_actionable_activity(monkeypatch) -> None:
    monkeypatch.setenv("ONE_NUMBERS_MODEL_DRIFT_MIN_ROWS_1H", "120")
    monkeypatch.setenv("ONE_NUMBERS_MODEL_DRIFT_MIN_ROWS_4H", "480")
    monkeypatch.setenv("ONE_NUMBERS_MODEL_DRIFT_MIN_ACTIONABLE_1H", "12")
    monkeypatch.setenv("ONE_NUMBERS_MODEL_DRIFT_MIN_ACTIONABLE_4H", "48")
    monkeypatch.setenv("ONE_NUMBERS_MODEL_DRIFT_THRESHOLD", "0.20")

    snapshot = one_numbers._model_drift_snapshot(
        (180, 0, 0, 180),
        (720, 160, 0, 560),
    )

    assert snapshot["buy_rate_drift_abs"] > 0.20
    assert snapshot["action_mix_drift_abs"] > 0.20
    assert snapshot["model_drift_flag"] is False
    assert snapshot["model_drift_reason"] == "low_actionable_activity"


def test_model_drift_snapshot_flags_large_action_mix_shift(monkeypatch) -> None:
    monkeypatch.setenv("ONE_NUMBERS_MODEL_DRIFT_MIN_ROWS_1H", "120")
    monkeypatch.setenv("ONE_NUMBERS_MODEL_DRIFT_MIN_ROWS_4H", "480")
    monkeypatch.setenv("ONE_NUMBERS_MODEL_DRIFT_MIN_ACTIONABLE_1H", "12")
    monkeypatch.setenv("ONE_NUMBERS_MODEL_DRIFT_MIN_ACTIONABLE_4H", "48")
    monkeypatch.setenv("ONE_NUMBERS_MODEL_DRIFT_THRESHOLD", "0.20")

    snapshot = one_numbers._model_drift_snapshot(
        (180, 72, 18, 30),
        (720, 72, 180, 468),
    )

    assert snapshot["buy_rate_drift_abs"] > 0.20
    assert snapshot["action_mix_drift_abs"] > 0.20
    assert snapshot["model_drift_flag"] is True
    assert snapshot["model_drift_reason"] == "action_mix_shift"


def test_build_lightweight_summary_payload_prefers_durable_rollup_history(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(one_numbers, "PROJECT_ROOT", tmp_path)
    (tmp_path / "governance" / "health").mkdir(parents=True, exist_ok=True)
    (tmp_path / "exports" / "sql_reports").mkdir(parents=True, exist_ok=True)

    (tmp_path / "governance" / "health" / "one_numbers_rollup_history.json").write_text(
        json.dumps(
            {
                "history_by_day": {
                    "20260330": {
                        "day_utc": "20260330",
                        "generated_utc": "2026-03-30T21:00:00+00:00",
                        "report_mode": "lightweight_cached",
                        "metrics": {
                            "combined_decision_total_rows": "90",
                            "combined_governance_total_rows": "12",
                            "combined_blocked_total": "8",
                            "data_blocked_total": "3",
                            "risk_blocked_total": "5",
                            "paper_executed_total": "44",
                            "watchdog_restarts": "0",
                            "data_quality_score": "97.0",
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "governance" / "health" / "daily_runtime_summary_latest.json").write_text(
        json.dumps(
            {
                "day": "20260331",
                "watchdog": {"restarts": 0, "throttled": 0, "restart_errors": 0},
                "decision": {
                    "rows": 100,
                    "observe_only_data_blocked": 0,
                    "status_counts": {"BLOCKED": 5},
                    "stale_windows": 0,
                    "files": [],
                },
                "governance": {"rows": 10, "stale_windows": 0, "files": []},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "governance" / "health" / "paper_performance_latest.json").write_text(
        json.dumps({"history_daily_series": []}),
        encoding="utf-8",
    )

    payload, _entries = one_numbers._build_lightweight_summary_payload(
        project_root=tmp_path,
        requested_day="20260331",
        db_path=tmp_path / "data" / "jsonl_link.sqlite3",
    )

    assert payload["rollup_history_source"] == "durable_history"
    assert payload["month_to_date_days_covered"] == "2"
    assert payload["all_time_days_covered"] == "2"
    assert payload["month_to_date_decision_total_rows"] == "190"
    assert payload["all_time_decision_total_rows"] == "190"


def test_full_main_persists_rollup_history_from_sqlite_day(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(one_numbers, "PROJECT_ROOT", tmp_path)
    out_dir = tmp_path / "exports" / "one_numbers"
    db_path = tmp_path / "data" / "jsonl_link.sqlite3"
    out_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "governance" / "health").mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "governance" / "health" / "one_numbers_rollup_history.json").write_text(
        json.dumps(
            {
                "history_by_day": {
                    "20260330": {
                        "day_utc": "20260330",
                        "metrics": {
                            "combined_decision_total_rows": "50",
                            "combined_governance_total_rows": "5",
                            "combined_blocked_total": "2",
                            "data_blocked_total": "0",
                            "risk_blocked_total": "2",
                            "paper_executed_total": "0",
                            "watchdog_restarts": "0",
                            "data_quality_score": "90.0",
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    conn = one_numbers.sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE jsonl_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT NOT NULL,
            source_rel TEXT NOT NULL,
            line_no INTEGER NOT NULL,
            ingested_at TEXT NOT NULL,
            payload_sha1 TEXT NOT NULL,
            payload_json TEXT NOT NULL
        )
        """
    )
    conn.executemany(
        """
        INSERT INTO jsonl_records (source_file, source_rel, line_no, ingested_at, payload_sha1, payload_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "decision_explanations_20260331.jsonl",
                "decision_explanations/shadow_default/decision_explanations_20260331.jsonl",
                1,
                "2026-03-31T15:00:00+00:00",
                "sha1-a",
                json.dumps({"timestamp_utc": "2026-03-31T15:00:00+00:00", "action": "HOLD", "status": "SHADOW_ONLY", "symbol": "SPY"}),
            ),
            (
                "master_control_20260331.jsonl",
                "governance/shadow_default/master_control_20260331.jsonl",
                1,
                "2026-03-31T15:00:01+00:00",
                "sha1-b",
                json.dumps({"timestamp_utc": "2026-03-31T15:00:01+00:00", "master_action": "HOLD"}),
            ),
        ],
    )
    conn.commit()
    conn.close()
    (tmp_path / "governance" / "health" / "ingestion_storage_control_latest.json").write_text(
        json.dumps({"pressure_index": 0.1, "backpressure": {}, "steady_state": {"quality_score": 99.0, "quality_label": "excellent", "target_status": {}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_one_numbers_report.py",
            "--day",
            "20260331",
            "--out-dir",
            str(out_dir),
            "--db",
            str(db_path),
            "--no-sql-write",
        ],
    )

    rc = one_numbers.main()
    history_payload = json.loads((tmp_path / "governance" / "health" / "one_numbers_rollup_history.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert sorted(history_payload["history_by_day"]) == ["20260330", "20260331"]


def test_persist_rollup_history_writes_trimmed_latest_days(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(one_numbers, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("ONE_NUMBERS_ROLLUP_HISTORY_MAX_DAYS", "2")
    (tmp_path / "governance" / "health").mkdir(parents=True, exist_ok=True)
    (tmp_path / "governance" / "health" / "one_numbers_rollup_history.json").write_text(
        json.dumps(
            {
                "history_by_day": {
                    "20260329": {
                        "day_utc": "20260329",
                        "metrics": {
                            "combined_decision_total_rows": "10",
                            "combined_governance_total_rows": "1",
                            "combined_blocked_total": "1",
                            "data_blocked_total": "0",
                            "risk_blocked_total": "1",
                            "paper_executed_total": "2",
                            "watchdog_restarts": "0",
                            "data_quality_score": "95.0",
                        },
                    },
                    "20260330": {
                        "day_utc": "20260330",
                        "metrics": {
                            "combined_decision_total_rows": "20",
                            "combined_governance_total_rows": "2",
                            "combined_blocked_total": "2",
                            "data_blocked_total": "1",
                            "risk_blocked_total": "1",
                            "paper_executed_total": "3",
                            "watchdog_restarts": "0",
                            "data_quality_score": "96.0",
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    history = one_numbers._persist_rollup_history(
        tmp_path,
        {
            "day_utc": "20260331",
            "generated_utc": "2026-03-31T21:00:00+00:00",
            "report_mode": "lightweight_cached",
            "combined_decision_total_rows": "30",
            "combined_governance_total_rows": "3",
            "combined_blocked_total": "3",
            "data_blocked_total": "1",
            "risk_blocked_total": "2",
            "paper_executed_total": "4",
            "watchdog_restarts": "0",
            "data_quality_score": "97.0",
        },
    )

    assert sorted(history) == ["20260330", "20260331"]
