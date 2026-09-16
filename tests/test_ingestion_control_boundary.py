import json
import sys
from pathlib import Path

import pytest

from scripts import link_jsonl_to_sql as writer

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from scripts import ingestion_backpressure_guard as guard


@pytest.mark.parametrize(
    "rel",
    [
        "governance/events/jsonl_ingest_batches_governance_20260915.jsonl",
        "governance/health/jsonl_ingest_batch_journal_latest.jsonl",
        "governance/health/jsonl_ingest_batch_journal_runtime_latest.jsonl",
        "governance/health/jsonl_ingest_batch_journal_runtime_latest.jsonl.resume_index.json",
        "governance/health/jsonl_ingest_batch_journal_runtime_latest.jsonl.backpressure_index.json",
    ],
)
def test_recovery_journals_cannot_be_selected_even_by_explicit_focus(rel):
    assert guard._should_ignore_backpressure_file(rel)
    assert not writer._matches_rel_filters(
        source_rel=rel,
        stream=writer._classify_stream(rel),
        include_streams=[],
        exclude_streams=[],
        path_contains=[rel],
        path_not_contains=[],
    )


@pytest.mark.parametrize(
    "rel",
    [
        "governance/events/signal_generation_20260915.jsonl",
        "governance/health/storage_eject_guard_events.jsonl",
        "governance/archive/lacie_deep_cold_manifest.jsonl",
        "governance/health/jsonl_sql_ingestion_health_governance_latest.json",
        "governance/events/custom_journal_20260915.jsonl",
    ],
)
def test_business_events_and_other_receipts_remain_eligible(rel):
    assert writer._matches_rel_filters(
        source_rel=rel,
        stream=writer._classify_stream(rel),
        include_streams=[],
        exclude_streams=[],
        path_contains=[],
        path_not_contains=[],
    )


def test_excluded_journal_is_still_readable_for_checkpoint_recovery(tmp_path):
    path = tmp_path / "governance/health/jsonl_ingest_batch_journal_latest.jsonl"
    path.parent.mkdir(parents=True)
    row = dict(
        event="file_checkpoint",
        source_rel="governance/events/test.jsonl",
        last_line=1,
        last_offset_bytes=12,
        file_inode=123,
        timestamp_utc="2026-09-15T19:00:00+00:00",
    )
    original = (json.dumps(row) + "\n").encode()
    path.write_bytes(original)
    progress, _ = writer._load_journal_resume_progress(path, persist_index=False)
    assert progress[row["source_rel"]]["last_offset_bytes"] == 12
    assert path.read_bytes() == original
