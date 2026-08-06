import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from scripts import canary_rollout_guard as guard


def _create_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE jsonl_records (
                source_rel TEXT NOT NULL,
                payload_sha1 TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        for day, timestamp in (
            ("20260805", "2026-08-05T16:00:00+00:00"),
            ("20260806", "2026-08-06T16:00:00+00:00"),
        ):
            for profile, pnl in (("intraday_aggressive", 0.002), ("conservative", 0.001)):
                for index, symbol in enumerate(("SPY", "QQQ"), start=1):
                    payload = {
                        "timestamp_utc": timestamp,
                        "profile": profile,
                        "bot_id": f"bot_{profile}",
                        "snapshot_id": f"{profile}-{day}-{index}",
                        "symbol": symbol,
                        "pnl_proxy": pnl,
                        "action": "BUY",
                    }
                    conn.execute(
                        "INSERT INTO jsonl_records VALUES (?, ?, ?)",
                        (
                            f"governance/shadow_{profile}_equities/shadow_pnl_attribution_{day}.jsonl",
                            f"{profile}-{day}-{index}",
                            json.dumps(payload),
                        ),
                    )
        duplicate = {
            "timestamp_utc": "2026-08-06T16:00:00+00:00",
            "profile": "intraday_aggressive",
            "bot_id": "bot_intraday_aggressive",
            "snapshot_id": "intraday_aggressive-20260806-1",
            "symbol": "SPY",
            "pnl_proxy": 0.002,
            "action": "BUY",
        }
        conn.execute(
            "INSERT INTO jsonl_records VALUES (?, ?, ?)",
            (
                "governance/shadow_intraday_aggressive_equities/shadow_pnl_attribution_20260806.jsonl",
                "duplicate-payload",
                json.dumps(duplicate),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _candidate_state(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "candidate_id": "pc-test-g1",
                "generation": 1,
                "accepted_git_head": "abc123",
                "scope_windows_started_utc": {"promotion": "2026-08-05T00:00:00+00:00"},
            }
        ),
        encoding="utf-8",
    )


def test_schema_v2_profile_rows_are_counted_and_candidate_bound(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "jsonl_link.sqlite3"
    state_path = tmp_path / "governance" / "runtime" / "production_candidate_state.json"
    _create_db(db_path)
    _candidate_state(state_path)

    payload = guard.build_payload(
        db_path=db_path,
        candidate_state_path=state_path,
        end=datetime(2026, 8, 6, 23, 59, tzinfo=timezone.utc),
        lookback_days=14,
        canary_profiles=["intraday_aggressive"],
        baseline_profiles=["conservative"],
        minimum_samples=4,
        minimum_days=2,
        minimum_symbols=2,
        minimum_effective_samples=1.0,
        minimum_edge_delta=0.0,
    )

    assert payload["candidate_binding"]["candidate_id"] == "pc-test-g1"
    assert payload["canary_samples"] == 4
    assert payload["baseline_samples"] == 4
    assert payload["scan"]["duplicates_removed"] == 1
    assert payload["eligible"] is True
    assert payload["promote_canary"] is True
    assert payload["edge_statistics"]["lower_confidence_bound_95"] > 0.0


def test_missing_candidate_binding_fails_closed(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "jsonl_link.sqlite3"
    _create_db(db_path)

    payload = guard.build_payload(
        db_path=db_path,
        candidate_state_path=tmp_path / "missing.json",
        end=datetime(2026, 8, 6, 23, 59, tzinfo=timezone.utc),
        lookback_days=14,
        canary_profiles=["intraday_aggressive"],
        baseline_profiles=["conservative"],
        minimum_samples=1,
        minimum_days=1,
        minimum_symbols=1,
        minimum_effective_samples=1.0,
        minimum_edge_delta=0.0,
    )

    assert payload["eligible"] is False
    assert payload["promote_canary"] is False
    assert "production_candidate_binding_missing" in payload["blockers"]


def test_canary_query_keeps_the_source_path_indexable(tmp_path: Path) -> None:
    db_path = tmp_path / "jsonl_link.sqlite3"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE jsonl_records (source_rel TEXT NOT NULL, payload_sha1 TEXT NOT NULL, payload_json TEXT NOT NULL)"
        )
        conn.execute("CREATE INDEX idx_jsonl_records_source_rel ON jsonl_records(source_rel)")
        sql = guard.PROFILE_ROWS_SQL.format(profile_placeholders="?")
        plan = conn.execute(
            f"EXPLAIN QUERY PLAN {sql}",
            ("governance/*/shadow_pnl_attribution_20260806.jsonl", "intraday_aggressive"),
        ).fetchall()
    finally:
        conn.close()

    assert any("SEARCH jsonl_records USING INDEX" in str(row[-1]) for row in plan)


def test_live_jsonl_is_cached_incrementally_without_duplicate_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "jsonl_link.sqlite3"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE jsonl_records (source_rel TEXT NOT NULL, payload_sha1 TEXT NOT NULL, payload_json TEXT NOT NULL)"
        )
        conn.commit()
    finally:
        conn.close()
    state_path = tmp_path / "governance" / "runtime" / "production_candidate_state.json"
    scan_state_path = tmp_path / "governance" / "runtime" / "canary_rollout_scan_state.json"
    evidence_path = tmp_path / "governance" / "evidence" / "canary_rollout_observations.jsonl"
    _candidate_state(state_path)

    source_root = tmp_path / "governance"
    canary_file = source_root / "shadow_intraday_aggressive_equities" / "shadow_pnl_attribution_20260806.jsonl"
    baseline_file = source_root / "shadow_conservative_equities" / "shadow_pnl_attribution_20260806.jsonl"
    canary_file.parent.mkdir(parents=True, exist_ok=True)
    baseline_file.parent.mkdir(parents=True, exist_ok=True)
    canary_file.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-08-06T16:00:00+00:00",
                "profile": "intraday_aggressive",
                "bot_id": "canary",
                "snapshot_id": "canary-1",
                "symbol": "SPY",
                "pnl_proxy": 0.002,
                "action": "BUY",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    baseline_file.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-08-06T16:00:00+00:00",
                "profile": "conservative",
                "bot_id": "baseline",
                "snapshot_id": "baseline-1",
                "symbol": "QQQ",
                "pnl_proxy": 0.001,
                "action": "BUY",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    kwargs = {
        "db_path": db_path,
        "candidate_state_path": state_path,
        "end": datetime(2026, 8, 6, 23, 59, tzinfo=timezone.utc),
        "lookback_days": 14,
        "canary_profiles": ["intraday_aggressive"],
        "baseline_profiles": ["conservative"],
        "minimum_samples": 1,
        "minimum_days": 1,
        "minimum_symbols": 1,
        "minimum_effective_samples": 1.0,
        "minimum_edge_delta": 0.0,
        "scan_state_path": scan_state_path,
        "evidence_cache_path": evidence_path,
    }
    first = guard.build_payload(**kwargs)

    assert first["canary_samples"] == 1
    assert first["baseline_samples"] == 1
    assert first["scan"]["primary"]["new_rows"] == 2
    assert first["scan"]["fallback"]["skipped"] is True

    with canary_file.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "timestamp_utc": "2026-08-06T17:00:00+00:00",
                    "profile": "intraday_aggressive",
                    "bot_id": "canary",
                    "snapshot_id": "canary-2",
                    "symbol": "QQQ",
                    "pnl_proxy": 0.003,
                    "action": "BUY",
                }
            )
            + "\n"
        )

    second = guard.build_payload(**kwargs)

    assert second["canary_samples"] == 2
    assert second["baseline_samples"] == 1
    assert second["scan"]["primary"]["cached_rows_before"] == 2
    assert second["scan"]["primary"]["new_rows"] == 1
    assert len(evidence_path.read_text(encoding="utf-8").splitlines()) == 3
