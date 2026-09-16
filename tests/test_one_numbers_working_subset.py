import json
import sqlite3

import pytest

from scripts import build_one_numbers_report as report


@pytest.mark.parametrize("source_list", [True, False])
def test_compact_governance_preserves_report_metrics_and_source(source_list):
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE jsonl_records (id INTEGER, source_file TEXT, source_rel TEXT, "
        "line_no INTEGER, ingested_at TEXT, payload_sha1 TEXT, payload_json TEXT)"
    )
    governance = "governance/sleeve/master_control_20260909.jsonl"
    decision = "decision_explanations/sleeve/decision_explanations_20260909.jsonl"
    values = [
        {
            "timestamp_utc": "2026-09-09T15:00:00+00:00",
            "options_plan": {"options_style": "spread", "contracts": 2.5},
            "futures_plan": {"futures_style": "trend"},
            "options_master": {"action": "BUY"},
            "active_futures_sub_bots": 7,
            "active_options_sub_bots": "3",
            "large_diagnostic": "x" * 512_000 + "CANARY",
        },
        {"timestamp_utc": "2026-09-09T15:01:00Z", "options_plan": None},
        {"action": "HOLD", "status": "BLOCKED", "reason": "inside_no_trade_band"},
    ]
    sources = [governance, governance, decision]
    for index, (source, payload) in enumerate(zip(sources, values), 1):
        conn.execute(
            "INSERT INTO jsonl_records VALUES (?,?,?,?,?,?,?)",
            (index, source, source, index, "now", str(index), json.dumps(payload)),
        )
    before = conn.execute("SELECT * FROM main.jsonl_records ORDER BY id").fetchall()
    fields = (
        "timestamp_utc",
        "options_plan.options_style",
        "options_plan.contracts",
        "futures_plan.futures_style",
        "options_master.action",
        "active_futures_sub_bots",
        "active_options_sub_bots",
    )
    metrics = ",".join(f"json_extract(payload_json, '$.{field}')" for field in fields)
    expected = conn.execute(
        f"SELECT {metrics} FROM main.jsonl_records WHERE source_rel=? ORDER BY id",
        (governance,),
    ).fetchall()
    row_count = report._materialize_working_subset(
        conn,
        source_rel_values=sources if source_list else [],
        decision_like="decision_explanations/%",
        governance_like="governance/%/master_control_%",
        pnl_like="pnl/%",
        watchdog_like="watchdog/%",
    )
    assert row_count == 3
    assert (
        conn.execute(
            f"SELECT {metrics} FROM temp.jsonl_records WHERE source_rel=? ORDER BY id",
            (governance,),
        ).fetchall()
        == expected
    )
    assert conn.execute(
        "SELECT payload_contains_canary FROM temp.jsonl_records ORDER BY id"
    ).fetchall() == [(1,), (0,), (0,)]
    assert (
        conn.execute(
            "SELECT length(payload_json) FROM temp.jsonl_records WHERE id=1"
        ).fetchone()[0]
        < 1000
    )
    assert conn.execute(
        "SELECT payload_json FROM temp.jsonl_records WHERE id=3"
    ).fetchone()[0] == json.dumps(values[2])
    assert (
        conn.execute("SELECT * FROM main.jsonl_records ORDER BY id").fetchall()
        == before
    )
    conn.close()
