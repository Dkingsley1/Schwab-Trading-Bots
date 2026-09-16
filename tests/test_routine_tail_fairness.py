import os
import time
from pathlib import Path

import pytest

from scripts import link_jsonl_to_sql as writer


def source(root, rel, size, age=0):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    stamp = time.time() - age
    os.utime(path, (stamp, stamp))
    return path


def checkpoint(path, offset, age=7200):
    stat = path.stat()
    return dict(
        last_line=1,
        last_offset_bytes=offset,
        file_inode=stat.st_ino,
        file_size_bytes=offset,
        mtime=time.time() - age,
    )


def test_overdue_core_tail_spends_existing_second_slot_and_rotates(tmp_path):
    first = source(tmp_path, "decisions/main/trade_decisions_20260915.jsonl", 1000)
    second = source(tmp_path, "decisions/next/trade_decisions_20260915.jsonl", 900)
    tail = source(tmp_path, "governance/events/training/old_20260912.jsonl", 10, 7200)
    paths = [first, second, tail]
    ordered = writer._prioritize_jsonl_files_by_pending_bytes(
        paths, project_root=tmp_path, sqlite_state={}
    )
    assert ordered[:2] == [first, tail]
    assert writer._limit_prioritized_jsonl_files(
        ordered, project_root=tmp_path, max_files=1, max_deferred_files=0
    ) == [first]
    assert writer._limit_prioritized_jsonl_files(
        ordered, project_root=tmp_path, max_files=2, max_deferred_files=0
    ) == [first, tail]
    state = {str(tail.relative_to(tmp_path)): checkpoint(tail, 10)}
    state[str(tail.relative_to(tmp_path))]["mtime"] = tail.stat().st_mtime
    ordered = writer._prioritize_jsonl_files_by_pending_bytes(
        paths, project_root=tmp_path, sqlite_state=state
    )
    assert ordered[:2] == [first, second]


@pytest.mark.parametrize(
    "cursor_kind,selected",
    [("valid", True), ("replaced", False), ("nan", False), ("future", False)],
)
def test_appended_file_can_use_valid_checkpoint_observation_age(
    tmp_path, cursor_kind, selected
):
    first = source(tmp_path, "decisions/main/trade_decisions_20260915.jsonl", 1000)
    second = source(tmp_path, "decisions/next/trade_decisions_20260915.jsonl", 900)
    tail = source(tmp_path, "governance/events/other_20260915.jsonl", 10)
    progress = checkpoint(tail, 5)
    if cursor_kind == "replaced":
        progress["file_inode"] += 1
    elif cursor_kind == "nan":
        progress["mtime"] = float("nan")
    elif cursor_kind == "future":
        progress["mtime"] = time.time() + 100
    ordered = writer._prioritize_jsonl_files_by_pending_bytes(
        [first, second, tail],
        project_root=tmp_path,
        sqlite_state={str(tail.relative_to(tmp_path)): progress},
    )
    assert ordered[0] == first
    assert (ordered[1] == tail) is selected


def test_tail_ordering_does_not_cross_lane_or_cold_budget(tmp_path, monkeypatch):
    core = source(tmp_path, "decisions/main/trade_decisions_20260915.jsonl", 100)
    deferred = source(
        tmp_path, "governance/channels/api/main/api_20260915.jsonl", 20, 7200
    )
    cold = source(
        tmp_path,
        "governance/shadow_main/shadow_pnl_attribution_20260912.jsonl",
        30,
        9000,
    )
    monkeypatch.setenv("JSONL_SQL_MAX_COLD_LANE_FILES", "0")
    ordered = writer._prioritize_jsonl_files_by_pending_bytes(
        [cold, deferred, core], project_root=tmp_path, sqlite_state={}
    )
    assert writer._limit_prioritized_jsonl_files(
        ordered, project_root=tmp_path, max_files=2, max_deferred_files=0
    ) == [core]
    assert writer._limit_prioritized_jsonl_files(
        ordered, project_root=tmp_path, max_files=3, max_deferred_files=1
    ) == [core, deferred]


def test_protected_source_is_not_statted_for_priority(tmp_path, monkeypatch):
    blocked = tmp_path / "blocked.jsonl"
    real_stat = Path.stat

    def checked_stat(path, *args, **kwargs):
        assert path != blocked
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(
        writer, "inspect_storage_path", lambda path: {"status": "protected"}
    )
    monkeypatch.setattr(Path, "stat", checked_stat)
    assert writer._prioritize_jsonl_files_by_pending_bytes(
        [blocked], project_root=tmp_path, sqlite_state={}
    ) == [blocked]


def test_overdue_first_choice_does_not_starve_an_even_older_tail(tmp_path):
    first = source(
        tmp_path, "decisions/main/trade_decisions_20260915.jsonl", 1000, 2000
    )
    second = source(tmp_path, "decisions/next/trade_decisions_20260915.jsonl", 900)
    tail = source(tmp_path, "governance/events/training/old_20260912.jsonl", 10, 7200)
    ordered = writer._prioritize_jsonl_files_by_pending_bytes(
        [first, second, tail], project_root=tmp_path, sqlite_state={}
    )
    assert ordered[:2] == [first, tail]
