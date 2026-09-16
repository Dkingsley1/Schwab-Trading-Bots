import hashlib
import os
import sys
import time
from types import SimpleNamespace

import psutil
import pytest

from scripts.ops import cold_sqlite_streaming_compressor as streaming
from scripts.ops import cold_sqlite_filesystem_compaction as compact


def test_only_pinned_executable_is_accepted(tmp_path, monkeypatch):
    path = tmp_path / "compressor"
    monkeypatch.setattr(streaming, "EXECUTABLE", path)
    with pytest.raises(RuntimeError, match="not_installed"):
        streaming.require_executable()
    path.write_bytes(b"fixture")
    path.chmod(0o700)
    assert compact.select_compressor("auto") == "applesauce"
    with pytest.raises(RuntimeError, match="hash_mismatch"):
        streaming.require_executable()
    monkeypatch.setattr(
        streaming, "EXECUTABLE_SHA256", hashlib.sha256(b"fixture").hexdigest()
    )
    assert streaming.require_executable() == str(path)
    link = tmp_path / "link"
    link.symlink_to(path)
    monkeypatch.setattr(streaming, "EXECUTABLE", link)
    with pytest.raises(RuntimeError, match="untrusted"):
        streaming.require_executable()


def test_scratch_and_compression_ratio_respect_resource_fork_limit(
    tmp_path, monkeypatch
):
    path = tmp_path / "large.sqlite3"
    with path.open("wb") as file:
        file.truncate(streaming.MAX_FILE_BYTES)
    captured = {}

    def run(command, **kwargs):
        captured.update(command=command, **kwargs)
        return {"rc": 0}

    monkeypatch.setattr(streaming, "run_paced", run)
    streaming.compress("/pinned", path, deadline=100, reserve_bytes=64 * streaming.GIB)
    command = captured["command"]
    ratio = float(command[command.index("--minimum-compression-ratio") + 1])
    assert ratio * path.stat().st_size <= streaming.MAX_COMPRESSED_BYTES
    # The owner verifies all logical bytes and SQLite integrity before publication.
    assert "--verify" not in command
    assert streaming.extra_scratch_bytes(path.stat().st_size) == (
        streaming.MAX_COMPRESSED_BYTES + streaming.OVERHEAD_BYTES
    )
    assert captured["parent"] == tmp_path


def test_large_file_is_rejected_before_start(tmp_path, monkeypatch):
    path = tmp_path / "too_large.sqlite3"
    with path.open("wb") as file:
        file.truncate(streaming.MAX_FILE_BYTES + 1)
    monkeypatch.setattr(
        streaming,
        "run_paced",
        lambda *a, **kw: pytest.fail("unsupported file must not launch"),
    )
    with pytest.raises(ValueError, match="compressor_size"):
        streaming.compress("unused", path, deadline=100, reserve_bytes=0)


def test_cpu_work_is_paced_and_reaped(tmp_path):
    result = streaming.run_paced(
        [
            sys.executable,
            "-c",
            "import time; s=time.process_time();\nwhile time.process_time()-s < .3: pass",
        ],
        parent=tmp_path,
        deadline=time.monotonic() + 10,
        reserve_bytes=0,
    )
    assert result["rc"] == 0
    assert result["process_reaped"]
    assert result["elapsed_seconds"] > 0.6
    assert result["pacing"]["pacing_sleep_seconds"] > 0.3
    assert result["peak_sampled_rss_bytes"] < streaming.MAX_RSS_BYTES


@pytest.mark.parametrize("failure", ["deadline", "memory", "reserve"])
def test_revocation_reaps_child(tmp_path, monkeypatch, failure):
    launched = []
    real_popen = streaming.subprocess.Popen

    def spawn(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        launched.append(proc)
        if failure == "reserve":
            monkeypatch.setattr(
                streaming.shutil, "disk_usage", lambda _: SimpleNamespace(free=0)
            )
        return proc

    monkeypatch.setattr(streaming.subprocess, "Popen", spawn)
    if failure == "memory":
        monkeypatch.setattr(streaming, "MAX_RSS_BYTES", 1)
    seconds = 0.3 if failure == "deadline" else 4
    with pytest.raises((RuntimeError, TimeoutError)):
        streaming.run_paced(
            [sys.executable, "-c", "import time; time.sleep(20)"],
            parent=tmp_path,
            deadline=time.monotonic() + seconds,
            reserve_bytes=1,
        )
    assert len(launched) == 1
    assert launched[0].poll() is not None
    assert not psutil.pid_exists(launched[0].pid)


def test_resource_probe_failure_prevents_start(tmp_path, monkeypatch):
    def fail(_):
        raise OSError("unavailable")

    monkeypatch.setattr(streaming.shutil, "disk_usage", fail)
    monkeypatch.setattr(
        streaming.subprocess, "Popen", lambda *a, **kw: pytest.fail("must not start")
    )
    with pytest.raises(OSError):
        streaming.run_paced(
            ["unused"], parent=tmp_path, deadline=time.monotonic() + 5, reserve_bytes=1
        )
