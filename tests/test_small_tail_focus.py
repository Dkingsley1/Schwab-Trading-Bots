import json
from datetime import datetime, timezone

from scripts.ops import sql_link_shard_manager as src


def test_old_small_tail_uses_existing_slot_only_with_material_focus(
    tmp_path, monkeypatch
):
    now = datetime.now(timezone.utc)
    path = tmp_path / "backpressure.json"
    monkeypatch.setenv("SQL_LINK_SERVICE_RAW_LIVE_PRIORITY_MIN_PENDING_LINES", "1000")
    monkeypatch.setenv("SQL_LINK_SERVICE_RAW_LIVE_PRIORITY_SOURCE_MIN_LINES", "100")
    monkeypatch.setenv("SQL_LINK_SERVICE_RAW_LIVE_PRIORITY_MAX_SOURCES_PER_SHARD", "2")
    rows = [
        dict(
            source_rel="decisions/large/trade_decisions_20260915.jsonl",
            pending_lines=800,
            oldest_pending_age_seconds=5,
        ),
        dict(
            source_rel="decisions/medium/trade_decisions_20260915.jsonl",
            pending_lines=700,
            oldest_pending_age_seconds=5,
        ),
        dict(
            source_rel="decisions/old/trade_decisions_20260915.jsonl",
            pending_lines=9,
            oldest_pending_age_seconds=3600,
        ),
        dict(
            source_rel="decisions/recent/trade_decisions_20260915.jsonl",
            pending_lines=9,
            oldest_pending_age_seconds=30,
        ),
        dict(
            source_rel="governance/events/old.jsonl",
            pending_lines=1,
            oldest_pending_age_seconds=7200,
        ),
    ]
    shards = [dict(name="trading", path_contains="", max_files=2)]
    payload = dict(
        timestamp_utc=now.isoformat(), pending_lines=1519, top_pending_files=rows
    )
    path.write_text(json.dumps(payload))
    focused, contract = src._apply_raw_live_priority_focus(
        shards, backpressure_path=path, now_utc=now
    )
    assert contract["applied"]
    assert focused[0]["raw_live_priority_sources"] == [
        rows[0]["source_rel"],
        rows[2]["source_rel"],
    ]
    assert focused[0]["max_files"] == 2
    assert contract["small_tail_reserved_shards"] == ["trading"]
    assert len(focused) == 1

    governance = dict(
        name="governance",
        max_files=3,
        max_lines_per_file=40,
        max_bytes_per_file=10000,
        sqlite_batch_max_bytes=5000,
    )
    focused, contract = src._apply_raw_live_priority_focus(
        [*shards, governance], backpressure_path=path, now_utc=now
    )
    assert focused[1]["raw_live_priority_sources"] == [rows[4]["source_rel"]]
    for key in (
        "max_files",
        "max_lines_per_file",
        "max_bytes_per_file",
        "sqlite_batch_max_bytes",
    ):
        assert focused[1][key] == governance[key]

    payload.update(pending_lines=19, top_pending_files=rows[2:])
    path.write_text(json.dumps(payload))
    _, contract = src._apply_raw_live_priority_focus(
        shards, backpressure_path=path, now_utc=now
    )
    assert not contract["applied"]
    assert contract["reason"] == "pressure_below_focus_threshold"


def test_single_slot_does_not_displace_material_source(tmp_path, monkeypatch):
    now = datetime.now(timezone.utc)
    path = tmp_path / "backpressure.json"
    monkeypatch.setenv("SQL_LINK_SERVICE_RAW_LIVE_PRIORITY_MIN_PENDING_LINES", "1000")
    monkeypatch.setenv("SQL_LINK_SERVICE_RAW_LIVE_PRIORITY_MAX_SOURCES_PER_SHARD", "1")
    path.write_text(
        json.dumps(
            dict(
                timestamp_utc=now.isoformat(),
                pending_lines=2001,
                top_pending_files=[
                    dict(
                        source_rel="decisions/large/trade_decisions_20260915.jsonl",
                        pending_lines=2000,
                        oldest_pending_age_seconds=5,
                    ),
                    dict(
                        source_rel="decisions/old/trade_decisions_20260915.jsonl",
                        pending_lines=1,
                        oldest_pending_age_seconds=7200,
                    ),
                ],
            )
        )
    )
    focused, contract = src._apply_raw_live_priority_focus(
        [dict(name="trading")], backpressure_path=path, now_utc=now
    )
    assert len(focused[0]["raw_live_priority_sources"]) == 1
    assert not contract["small_tail_reserved_shards"]
