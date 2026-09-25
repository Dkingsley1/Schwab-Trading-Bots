import json
from datetime import datetime, timezone

import pytest

from scripts.ops import sql_link_shard_manager as manager


@pytest.mark.parametrize("checkpoint_age", [False, True])
@pytest.mark.parametrize(
    "source,owner",
    [
        ("governance/health/preopen_replay_drift_history.jsonl", "governance"),
        (
            "governance/channels/loop_state/futures_rates_curve_equities_schwab/loop_state_20260913.jsonl",
            "governance",
        ),
        (
            "governance/channels/loop_state/default_crypto_coinbase/loop_state_20260913.jsonl",
            "crypto_governance",
        ),
    ],
)
def test_deferred_receipt_uses_existing_governance_slot_only_after_material_admission(
    tmp_path, monkeypatch, checkpoint_age, source, owner
):
    now = datetime.now(timezone.utc)
    monkeypatch.setenv("SQL_LINK_SERVICE_RAW_LIVE_PRIORITY_MIN_PENDING_LINES", "1000")
    monkeypatch.setenv("SQL_LINK_SERVICE_RAW_LIVE_PRIORITY_SOURCE_MIN_LINES", "100")
    monkeypatch.setenv("SQL_LINK_SERVICE_RAW_LIVE_PRIORITY_MAX_SOURCES_PER_SHARD", "2")
    path = tmp_path / "backpressure.json"
    payload = dict(
        timestamp_utc=now.isoformat(),
        pending_lines=2000,
        top_pending_files=[dict(source_rel="decisions/main.jsonl", pending_lines=2000)],
        top_deferred_pending_files=[
            dict(source_rel=source, pending_lines=14, oldest_pending_age_seconds=7200),
            dict(
                source_rel="decision_explanations/old.jsonl",
                pending_lines=1,
                oldest_pending_age_seconds=9000,
            ),
        ],
    )
    if checkpoint_age:
        payload["top_deferred_pending_files"][0].update(
            oldest_pending_age_seconds=1, checkpoint_service_age_seconds=7200
        )
    path.write_text(json.dumps(payload))
    original = dict(
        name=owner,
        max_files=4,
        max_lines_per_file=8000,
        max_bytes_per_file=1024,
        sqlite_batch_max_bytes=512,
    )
    focused, result = manager._apply_raw_live_priority_focus(
        [original, dict(name="health_fast")], backpressure_path=path, now_utc=now
    )
    assert focused[0]["raw_live_priority_sources"] == [source]
    assert focused[1] == dict(name="health_fast")
    assert result["small_tail_reserved_shards"] == [owner]
    for key in (
        "max_files",
        "max_lines_per_file",
        "max_bytes_per_file",
        "sqlite_batch_max_bytes",
    ):
        assert focused[0][key] == original[key]
    _, result = manager._apply_raw_live_priority_focus(
        [dict(name="health_fast")], backpressure_path=path, now_utc=now
    )
    assert not result["applied"]
    payload.update(pending_lines=0, top_pending_files=[])
    path.write_text(json.dumps(payload))
    _, result = manager._apply_raw_live_priority_focus(
        [original], backpressure_path=path, now_utc=now
    )
    assert result["reason"] == "pressure_below_focus_threshold"
