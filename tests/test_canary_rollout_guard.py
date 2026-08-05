import json
import sqlite3
from pathlib import Path

from scripts import canary_rollout_guard


def test_canary_rollout_uses_indexable_source_glob_and_aggregates_profiles(tmp_path: Path) -> None:
    db_path = tmp_path / "jsonl_link.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE jsonl_records (source_rel TEXT NOT NULL, line_no INTEGER NOT NULL, payload_json TEXT NOT NULL)")
    conn.execute("CREATE INDEX idx_jsonl_records_source_rel_line ON jsonl_records(source_rel, line_no)")
    rows = [
        (
            "governance/alpha/shadow_pnl_attribution_20260731.jsonl",
            1,
            json.dumps({"shadow_profile": "intraday_aggressive", "pnl_proxy": 0.01}),
        ),
        (
            "governance/beta/shadow_pnl_attribution_20260731.jsonl",
            1,
            json.dumps({"shadow_profile": "intraday_aggressive", "pnl_proxy": 0.03}),
        ),
        (
            "governance/beta/shadow_pnl_attribution_20260730.jsonl",
            1,
            json.dumps({"shadow_profile": "intraday_aggressive", "pnl_proxy": 9.0}),
        ),
    ]
    conn.executemany("INSERT INTO jsonl_records(source_rel, line_no, payload_json) VALUES (?, ?, ?)", rows)
    plan = conn.execute(
        f"EXPLAIN QUERY PLAN {canary_rollout_guard.CANARY_PROFILE_SQL}",
        ("governance/*/shadow_pnl_attribution_20260731.jsonl",),
    ).fetchall()
    conn.commit()
    conn.close()

    stats = canary_rollout_guard._load_profile_stats(db_path, "20260731")

    assert stats["intraday_aggressive"]["n"] == 2
    assert stats["intraday_aggressive"]["avg_pnl"] == 0.02
    assert any("SEARCH jsonl_records USING INDEX" in str(row[-1]) for row in plan)
