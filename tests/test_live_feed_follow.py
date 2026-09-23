from pathlib import Path

from scripts.ops import live_feed_follow as src


def test_append_preserves_offsets_and_partial_lines(tmp_path):
    path = tmp_path / "feed.log"
    path.write_bytes(b"history\n")
    follower = src.Follower()
    assert follower.read(path, initialize=True) == b""
    with path.open("ab") as handle:
        handle.write(b"fresh")
    assert follower.read(path) == b""
    with path.open("ab") as handle:
        handle.write(b" event\n")
    assert follower.read(path) == b"fresh event\n"
    assert follower.read(path) == b""


def test_new_daily_file_and_atomic_rotation(tmp_path):
    follower = src.Follower()
    old = tmp_path / "day1.log"
    old.write_bytes(b"old\n")
    follower.read(old, initialize=True)
    new = tmp_path / "day2.log"
    new.write_bytes(b"new day\n")
    assert follower.read(new) == b"new day\n"
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"replacement\n")
    replacement.replace(new)
    assert follower.read(new) == b"replacement\n"
    new.write_bytes(b"short\n")
    assert follower.read(new) == b"short\n"


def test_read_and_partial_line_buffers_are_bounded(tmp_path):
    path = tmp_path / "wide.log"
    path.write_bytes(b"x" * (2 * src.MAX_READ))
    follower = src.Follower()
    assert len(follower.read(path)) < src.MAX_READ
    assert follower.cursors[path].offset == src.MAX_READ
    assert len(follower.cursors[path].pending) <= src.MAX_LINE


def test_protected_route_is_not_opened(monkeypatch):
    monkeypatch.setattr(
        src, "inspect_storage_path", lambda path: {"status": "protected"}
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("must not touch denied path")

    monkeypatch.setattr(src.os, "open", forbidden)
    assert src.Follower().read(Path("/denied")) == b""


def test_fifo_cannot_block_reader(tmp_path):
    import os

    fifo = tmp_path / "not_a_log"
    os.mkfifo(fifo)
    assert src.Follower().read(fifo) == b""


def test_discovery_failure_preserves_existing_selection(monkeypatch):
    monkeypatch.setattr(
        src,
        "run_bounded_process_group",
        lambda *a, **kw: {
            "rc": 124,
            "timed_out": True,
            "stdout": "/new\0",
        },
    )
    assert src.discover(["select"]) is None
    monkeypatch.setattr(
        src,
        "run_bounded_process_group",
        lambda *a, **kw: {
            "rc": 0,
            "timed_out": False,
            "stdout": "/a\0/b\0/a\0",
        },
    )
    assert src.discover(["select"]) == [Path("/a"), Path("/b")]
