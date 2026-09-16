import json

import pytest

from scripts.ops import ingestion_storage_control as src
from scripts import link_jsonl_to_sql as linker


def setup_state(root, monkeypatch, **changes):
    path = root / "governance/events/old.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"record": 1}\n{"record": 2}\n')
    stat = path.stat()
    state = dict(
        last_line=2,
        last_offset_bytes=stat.st_size,
        file_size_bytes=stat.st_size,
        file_inode=stat.st_ino,
        mtime=stat.st_mtime,
    )
    state.update(changes)
    rel = str(path.relative_to(root))
    state_path = root / "state.json"
    state_path.write_text(json.dumps({"sqlite": {rel: state}}))
    monkeypatch.setattr(src, "_sql_ingestion_state_paths", lambda _: [state_path])
    return path, rel, state_path, state


@pytest.mark.parametrize(
    "mutation", ["inode", "shrink", "rewind", "offset", "malformed"]
)
def test_old_generation_cannot_clear_backlog(tmp_path, monkeypatch, mutation):
    path, rel, state_path, state = setup_state(tmp_path, monkeypatch)
    if mutation == "inode":
        state["file_inode"] += 1
    elif mutation == "shrink":
        state["file_size_bytes"] += 1
    elif mutation == "rewind":
        state["mtime"] += 10
    elif mutation == "offset":
        state["last_offset_bytes"] += 1
    else:
        state["last_line"] = "invalid"
    state_path.write_text(json.dumps({"sqlite": {rel: state}}))
    assert src._state_progress_for_source(tmp_path, rel) == {}


def test_valid_cursor_still_reconciles_and_invalid_larger_cursor_cannot_win(
    tmp_path, monkeypatch
):
    path, rel, valid, state = setup_state(tmp_path, monkeypatch)
    stale = tmp_path / "stale.json"
    stale.write_text(
        json.dumps(
            {
                "sqlite": {
                    rel: {
                        **state,
                        "last_line": 999,
                        "file_inode": state["file_inode"] + 1,
                    }
                }
            }
        )
    )
    monkeypatch.setattr(src, "_sql_ingestion_state_paths", lambda _: [stale, valid])
    payload = src._state_progress_for_source(tmp_path, rel)
    assert payload["reconciled"]
    assert payload["pending_lines"] == 0
    assert payload["last_line"] == 2


def test_source_change_during_exact_count_preserves_pending(tmp_path, monkeypatch):
    path, rel, _, _ = setup_state(
        tmp_path, monkeypatch, last_offset_bytes=0, last_line=1
    )

    def count(source, **kwargs):
        with source.open("a") as handle:
            handle.write('{"record": 3}\n')
        return 2

    monkeypatch.setattr(src, "_count_lines_bounded", count)
    assert src._state_progress_for_source(tmp_path, rel) == {}


def test_protected_source_is_rejected_before_state_reads(tmp_path, monkeypatch):
    (tmp_path / "blocked").symlink_to("/Volumes/VIDEO/private")
    monkeypatch.setattr(
        src, "_sql_ingestion_state_paths", lambda _: pytest.fail("state read")
    )
    assert src._state_progress_for_source(tmp_path, "blocked/file.jsonl") == {}


def test_changed_generation_is_prioritized_as_pending(tmp_path, monkeypatch):
    path, rel, _, stale = setup_state(tmp_path, monkeypatch)
    stale["file_inode"] += 1
    complete = path.with_name("complete.jsonl")
    complete.write_bytes(path.read_bytes())
    info = complete.stat()
    complete_rel = str(complete.relative_to(tmp_path))
    current = dict(
        last_line=2,
        last_offset_bytes=info.st_size,
        file_size_bytes=info.st_size,
        file_inode=info.st_ino,
        mtime=info.st_mtime,
    )
    ordered = linker._prioritize_jsonl_files_by_pending_bytes(
        [complete, path],
        project_root=tmp_path,
        sqlite_state={rel: stale, complete_rel: current},
    )
    assert ordered[0] == path
