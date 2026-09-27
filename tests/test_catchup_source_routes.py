import json
from datetime import datetime, timezone

import pytest

from scripts import link_jsonl_to_sql as writer
from scripts.ops import sql_link_shard_manager as manager

MANIFEST = "data/jsonl_link_archives/cold_archive_compaction_manifest.jsonl"
GOVERNANCE = [
    "governance/allocator/sleeve_allocator_events_20260915.jsonl",
    "governance/archive/lacie_deep_cold_manifest.jsonl",
    "governance/regime/regime_control_plane_history.jsonl",
    "governance/research/institutional_decision_flow/history_20260915.jsonl",
    "governance/risk/portfolio_risk_events_20260915.jsonl",
    "governance/health/preopen_replay_drift_history.jsonl",
    "governance/health/storage_eject_guard_events.jsonl",
    "governance/channels/loop_state/futures_rates_curve_equities_schwab/loop_state_20260913.jsonl",
]


def accepted(source, shard):
    defaults = manager.DEFAULT_SHARD_DEFS[shard]
    return writer._matches_rel_filters(
        source_rel=source,
        stream=writer._classify_stream(source),
        **{
            key: manager._filter_list(defaults.get(key, ""))
            for key in (
                "include_streams",
                "exclude_streams",
                "path_contains",
                "path_not_contains",
            )
        },
    )


@pytest.mark.parametrize(
    "source,shard", [(MANIFEST, "data"), *[(s, "governance") for s in GOVERNANCE]]
)
def test_pending_receipts_have_normal_and_priority_routes(source, shard):
    assert accepted(source, shard)
    assert manager._raw_live_priority_shard_for_source(source) == shard


@pytest.mark.parametrize(
    "sleeve,owner",
    [
        ("futures_rates_curve_equities_schwab", "governance"),
        ("default_crypto_coinbase", "crypto_governance"),
        ("crypto_futures_crypto_coinbase", "crypto_governance"),
        ("crypto_futures_basis_equities_schwab", "governance"),
    ],
)
def test_loop_state_uses_its_normal_governance_owner_not_runtime(sleeve, owner):
    source = f"governance/channels/loop_state/{sleeve}/loop_state_20260913.jsonl"
    assert accepted(source, owner)
    assert not accepted(source, "runtime")
    assert not accepted(source, "crypto_runtime")
    assert manager._raw_live_priority_shard_for_source(source) == owner
    runtime = f"governance/channels/runtime/{sleeve}/runtime_20260915.jsonl"
    assert manager._raw_live_priority_shard_for_source(runtime) in {
        "runtime",
        "crypto_runtime",
    }


def test_manifest_route_does_not_open_arbitrary_archives_or_primary_merge():
    unrelated = "data/jsonl_link_archives/other_history.jsonl"
    assert not accepted(unrelated, "data")
    assert manager._raw_live_priority_shard_for_source(unrelated) == ""
    assert manager.DEFAULT_SHARD_DEFS["data"]["merge_to_primary"] is False
    excluded = "governance/research/default_crypto_schwab/receipt.jsonl"
    assert not accepted(excluded, "governance")
    assert manager._raw_live_priority_shard_for_source(excluded) == ""
    assert (
        manager._raw_live_priority_shard_for_source(
            "governance/channels/risk/default_schwab/risk.jsonl"
        )
        == "risk_support"
    )
    assert (
        manager._raw_live_priority_shard_for_source(
            "governance/health/one_numbers_latest.json"
        )
        == "health_fast"
    )
    assert (
        manager._raw_live_priority_shard_for_source(
            "governance/health/jsonl_ingest_batch_journal_latest.jsonl"
        )
        == ""
    )


@pytest.mark.parametrize(
    "source,shard", [(MANIFEST, "data"), (GOVERNANCE[1], "governance")]
)
def test_receipt_tail_uses_only_requested_shard_and_existing_bounds(
    tmp_path, monkeypatch, source, shard
):
    now = datetime.now(timezone.utc)
    monkeypatch.setenv("SQL_LINK_SERVICE_RAW_LIVE_PRIORITY_MIN_PENDING_LINES", "1000")
    monkeypatch.setenv("SQL_LINK_SERVICE_RAW_LIVE_PRIORITY_SOURCE_MIN_LINES", "100")
    path = tmp_path / "backlog.json"
    path.write_text(
        json.dumps(
            {
                "timestamp_utc": now.isoformat(),
                "pending_lines": 2001,
                "top_pending_files": [
                    {
                        "source_rel": "decisions/main/trade_decisions.jsonl",
                        "pending_lines": 2000,
                    },
                    {
                        "source_rel": source,
                        "pending_lines": 1,
                        "oldest_pending_age_seconds": 3600,
                    },
                ],
            }
        )
    )
    original = dict(
        name=shard,
        max_files=2,
        max_lines_per_file=40,
        max_bytes_per_file=10000,
        sqlite_batch_max_bytes=5000,
    )
    focused, receipt = manager._apply_raw_live_priority_focus(
        [original], backpressure_path=path, now_utc=now
    )
    assert receipt["small_tail_reserved_shards"] == [shard]
    assert focused[0]["raw_live_priority_sources"] == [source]
    for key in (
        "max_files",
        "max_lines_per_file",
        "max_bytes_per_file",
        "sqlite_batch_max_bytes",
    ):
        assert focused[0][key] == original[key]
    other = dict(name="api_ingress")
    focused, receipt = manager._apply_raw_live_priority_focus(
        [other], backpressure_path=path, now_utc=now
    )
    assert focused == [other]
    assert not receipt["applied"]
