"""The duplicate log owner must never retire SQL or unassigned raw inputs."""

import gzip
import json
from pathlib import Path

import pytest

from scripts.ops import verified_duplicate_cleanup as src


@pytest.mark.parametrize(
    "relative",
    [
        "decisions/paper/rows_20260501.jsonl",
        "decision_explanations/paper/rows_20260501.jsonl",
        "governance/channels/runtime/test/rows_20260501.jsonl",
        "data/rows_20260501.jsonl",
        "exports/trade_logs/rows_20260501.jsonl",
        "paper_trades_20260501.jsonl",
        "live_orders_20260501.jsonl",
        "unknown/rows_20260501.jsonl",
        "logs/../data/rows_20260501.jsonl",
    ],
)
@pytest.mark.parametrize("checkpoint", [None, 1, 100])
def test_sql_and_unknown_sources_preserved_regardless_of_checkpoint(
    tmp_path, relative, checkpoint
):
    raw = tmp_path / relative
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(b"x" * 100)
    archive = Path(str(raw) + ".gz")
    archive.write_bytes(gzip.compress(raw.read_bytes()))
    state = tmp_path / "governance/sql_link_shards/jsonl_sql_link_state_other.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(
        json.dumps({"sqlite": {relative: {"last_offset_bytes": checkpoint}}})
    )
    with pytest.raises(RuntimeError, match="retirement_owner_required"):
        src.remove_pair(tmp_path, raw, archive, src.Budget())
    assert raw.read_bytes() == b"x" * 100


def test_post_release_close_failure_is_not_reported_as_untouched(tmp_path, monkeypatch):
    raw = tmp_path / "logs/rows_20260501.jsonl"
    raw.parent.mkdir()
    raw.write_bytes(b"content")
    archive = Path(str(raw) + ".gz")
    archive.write_bytes(gzip.compress(b"content"))
    monkeypatch.setattr(src.safety, "idle", lambda path: None)
    monkeypatch.setattr(src, "receipt", lambda *args: None)
    original = src.os.close

    def fail_after_close(fd):
        original(fd)
        raise OSError("close failed after release")

    monkeypatch.setattr(src.os, "close", fail_after_close)
    proof = src.remove_pair(tmp_path, raw, archive, src.Budget())
    assert proof["source_removed"] is True
    assert proof["release_persistence_complete"] is False
    assert not raw.exists()
