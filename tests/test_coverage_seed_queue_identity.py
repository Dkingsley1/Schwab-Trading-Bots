import json
import os

import pytest

from scripts.ops import walk_forward_coverage_seed as seed


def test_unchanged_queue_preserves_identity_and_age(tmp_path):
    path = tmp_path / "queue.jsonl"
    rows = [{"bot": "example", "value": "x" * 150000}, {"value": 2}]
    seed._write_queue(path, rows)
    os.utime(path, (1000, 1000))
    before = path.stat()
    seed._write_queue(path, rows)
    after = path.stat()
    assert (before.st_ino, before.st_mtime_ns) == (after.st_ino, after.st_mtime_ns)
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("rows", [[{"value": "new"}], []])
def test_changed_queue_is_atomically_replaced(tmp_path, rows):
    path = tmp_path / "queue.jsonl"
    seed._write_queue(path, [{"value": "old"}])
    with path.open() as old_generation:
        old_inode = os.fstat(old_generation.fileno()).st_ino
        seed._write_queue(path, rows)
        assert path.stat().st_ino != old_inode
        assert json.loads(old_generation.read()) == {"value": "old"}
    assert [json.loads(line) for line in path.read_text().splitlines()] == rows
    assert list(tmp_path.iterdir()) == [path]


def test_serialization_failure_preserves_original_and_cleans_temp(tmp_path):
    path = tmp_path / "queue.jsonl"
    seed._write_queue(path, [{"value": "old"}])
    before = path.read_bytes()
    with pytest.raises(TypeError):
        seed._write_queue(path, [{"value": object()}])
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_comparison_rejects_replaced_source(tmp_path, monkeypatch):
    path = tmp_path / "queue.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    path.write_bytes(b"same\n")
    candidate.write_bytes(b"same\n")
    fstat = os.fstat
    reads = 0

    def racing_stat(fd):
        nonlocal reads
        reads += 1
        if reads == 3:
            replacement = tmp_path / "replacement"
            replacement.write_bytes(b"same\n")
            os.replace(replacement, path)
        return fstat(fd)

    monkeypatch.setattr(os, "fstat", racing_stat)
    assert not seed._queue_matches(path, candidate)


def test_symlink_is_not_treated_as_matching_owned_queue(tmp_path):
    target = tmp_path / "target"
    target.write_bytes(b"same\n")
    alias = tmp_path / "queue.jsonl"
    alias.symlink_to(target)
    assert not seed._queue_matches(alias, target)
