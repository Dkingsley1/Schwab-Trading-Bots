import fcntl
import json
import os
import sys

import pytest

from scripts.ops import lock_watchdog as src


@pytest.mark.parametrize("content", ["", "pid=999999999\n", f"pid={os.getpid()}\n"])
@pytest.mark.parametrize("held", [False, True])
def test_watchdog_never_unlinks_lock_anchors(tmp_path, monkeypatch, content, held):
    path = tmp_path / "governance/locks/storage_maintenance.lock"
    path.parent.mkdir(parents=True)
    path.write_text(content)
    inode = path.stat().st_ino
    out = tmp_path / "health.json"
    monkeypatch.setattr(src, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(src, "OUT", out)
    monkeypatch.setattr(sys, "argv", ["lock-watchdog", "--apply", "--json"])
    with path.open("a+") as owner:
        if held:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert src.main() == 0
        result = json.loads(out.read_text())
        assert result["removed"] == []
        assert result["lock_inode_preservation"] is True
        assert path.stat().st_ino == inode and path.read_text() == content
        if held:
            assert result["healthy_locks"][0]["reason"] == "kernel_lock_held"
            with path.open("a+") as contender:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with path.open("a+") as next_owner:
        fcntl.flock(next_owner, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_unknown_probe_and_symlink_fail_closed(tmp_path, monkeypatch):
    path = tmp_path / "governance/locks/worker.lock"
    path.parent.mkdir(parents=True)
    path.write_text("")
    alias = path.with_name("alias.lock")
    alias.symlink_to(path)
    monkeypatch.setattr(src, "PROJECT_ROOT", tmp_path)
    assert src._lock_candidates() == [path]
    assert src._kernel_lock_state(alias) == "unknown"
    assert src._kernel_lock_state(path.with_name("missing.lock")) == "unknown"
