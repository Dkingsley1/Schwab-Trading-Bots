import hashlib
import json
import os
import plistlib
from pathlib import Path
import sqlite3
import sys
import shutil
import subprocess
import time
from contextlib import contextmanager

import pytest

from scripts.ops import cold_sqlite_filesystem_compaction as compact


@pytest.mark.parametrize("available,expected", [(True, "afsctool"), (False, "ditto")])
def test_auto_backend_prefers_installed_large_file_compressor(
    monkeypatch, available, expected
):
    monkeypatch.setattr(compact.streaming, "installed", lambda: False)
    monkeypatch.setattr(
        compact.shutil, "which", lambda _: "/tool/afsctool" if available else None
    )
    assert compact.select_compressor("auto") == expected
    assert compact.select_compressor("ditto") == "ditto"
    assert compact.select_compressor("afsctool") == "afsctool"
    with pytest.raises(ValueError, match="unknown_filesystem_compressor"):
        compact.select_compressor("unknown")


def test_builtin_compression_does_not_require_optional_afsctool(monkeypatch):
    monkeypatch.setattr(
        compact.os, "access", lambda path, mode: path == "/usr/bin/ditto"
    )
    monkeypatch.setattr(compact.shutil, "which", lambda name: None)
    assert compact.require_compressor("ditto") == "/usr/bin/ditto"
    with pytest.raises(RuntimeError, match="afsctool_not_installed"):
        compact.require_compressor("afsctool")


def test_optional_backend_still_requires_its_copy_dependency(monkeypatch):
    monkeypatch.setattr(compact.os, "access", lambda *a: False)
    monkeypatch.setattr(compact.shutil, "which", lambda name: "/tool/afsctool")
    with pytest.raises(RuntimeError, match="ditto_not_installed"):
        compact.require_compressor("afsctool")


@pytest.fixture
def native_apfs(tmp_path):
    try:
        device = (
            subprocess.run(
                ["df", "-P", str(tmp_path)],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            .stdout.splitlines()[-1]
            .split()[0]
        )
        result = subprocess.run(
            ["/usr/sbin/diskutil", "info", "-plist", device],
            capture_output=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"native APFS inspection unavailable: {type(exc).__name__}")
    if plistlib.loads(result.stdout).get("FilesystemType") != "apfs":
        pytest.skip("native compression integration requires an APFS test directory")


def _database(root):
    path = root / "cold.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE evidence (id INTEGER PRIMARY KEY, payload BLOB)")
    conn.execute("INSERT INTO evidence VALUES (1, zeroblob(2097152))")
    conn.commit()
    conn.close()
    old = time.time() - 3 * 86400
    os.utime(path, (old, old))
    return path


def _compact(path, root, **kwargs):
    return compact.compact_one(
        path,
        archive_root=root,
        manifest=root / "proof.jsonl",
        deadline=time.monotonic() + 60,
        **kwargs,
    )


def test_protected_alias_is_rejected_before_metadata(tmp_path, monkeypatch):
    alias = tmp_path / "alias"
    alias.symlink_to("/Volumes/VIDEO")
    original = Path.lstat

    def checked(path, *args, **kwargs):
        assert not str(path).lower().startswith("/volumes/video")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", checked)
    result = _compact(alias / "archive.sqlite3", tmp_path)
    assert result["status"] == "failed"
    assert not result["original_replaced"]


def test_recent_or_out_of_scope_file_is_preserved(tmp_path):
    path = _database(tmp_path)
    original = path.read_bytes()
    os.utime(path, None)
    assert "archive_not_old_enough" in _compact(path, tmp_path)["error"]
    assert "source_outside_cold_archive" in _compact(path, tmp_path / "other")["error"]
    assert path.read_bytes() == original


def test_first_oversized_file_does_not_bypass_wave_budget(tmp_path):
    path = _database(tmp_path)
    result = compact.build_payload(
        paths=[path],
        archive_root=tmp_path,
        manifest=tmp_path / "proof.jsonl",
        apply=True,
        max_files=1,
        max_raw_gb=0.0001,
        timeout_seconds=60,
        min_age_hours=24,
    )
    assert result["selected_paths"] == []
    assert result["allocated_bytes_reclaimed"] == 0
    assert path.exists()


def test_streaming_wave_budget_accounts_for_pacing_and_verification(tmp_path):
    large, fitting = tmp_path / "large.sqlite3", tmp_path / "fitting.sqlite3"
    for path, size in ((large, 3 * 1024**3), (fitting, 1024**3)):
        with path.open("wb") as handle:
            handle.truncate(size)
    result = compact.build_payload(
        paths=[large, fitting],
        archive_root=tmp_path,
        manifest=tmp_path / "proof.jsonl",
        apply=False,
        max_files=4,
        max_raw_gb=8,
        timeout_seconds=300,
        min_age_hours=24,
        compressor="applesauce",
    )
    assert result["selected_paths"] == [str(fitting)]
    assert result["requested_max_raw_gb"] == 8
    assert result["effective_max_raw_gb"] == 2


@pytest.mark.skipif(
    sys.platform != "darwin", reason="native APFS compression integration"
)
@pytest.mark.parametrize("compressor", ["ditto", "afsctool", "applesauce"])
def test_native_compression_keeps_sqlite_readable_and_proves_all_bytes(
    tmp_path, compressor, native_apfs
):
    if compressor == "afsctool" and not shutil.which("afsctool"):
        pytest.skip("optional afsctool backend not installed")
    if compressor == "applesauce" and not compact.streaming.installed():
        pytest.skip("pinned applesauce backend not installed")
    path = _database(tmp_path)
    original = path.read_bytes()
    result = _compact(path, tmp_path, compressor=compressor)
    assert result["status"] == "filesystem_compressed_verified", result.get(
        "error", result
    )
    assert result["allocated_bytes_reclaimed"] > 1024**2
    assert path.read_bytes() == original
    assert result["source_sha256"] == hashlib.sha256(original).hexdigest()
    conn = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
    assert conn.execute("SELECT id, length(payload) FROM evidence").fetchall() == [
        (1, 2097152)
    ]
    conn.close()
    receipts = [
        json.loads(line) for line in (tmp_path / "proof.jsonl").read_text().splitlines()
    ]
    assert [row["status"] for row in receipts] == [
        "verified_pending_replace",
        "filesystem_compressed_verified",
    ]
    assert not list(tmp_path.glob(".filesystem_compaction_*"))
    assert (
        _compact(path, tmp_path, compressor=compressor)["status"]
        == "already_compressed"
    )


def test_large_archive_is_rejected_before_expensive_copy(tmp_path, monkeypatch):
    path = _database(tmp_path)
    monkeypatch.setattr(compact, "MAX_NATIVE_FILE_BYTES", 1024)
    monkeypatch.setattr(
        compact, "_run", lambda *a, **kw: pytest.fail("unexpected subprocess")
    )
    result = _compact(path, tmp_path)
    assert "archive_exceeds_bounded_native_compression_size" in result["error"]
    assert not result["original_replaced"]


@pytest.mark.parametrize("source_changes", [False, True])
def test_isolated_copy_is_verified_before_short_publication_guard(
    tmp_path, monkeypatch, native_apfs, source_changes
):
    path = _database(tmp_path)
    original = path.read_bytes()
    publishing = [False]
    hash_calls = []
    original_hash = compact._hash

    def checked_hash(path, deadline):
        assert not publishing[0], "full-file reads must not hold the hot writer"
        hash_calls.append(path)
        return original_hash(path, deadline)

    @contextmanager
    def publish(deadline):
        assert len(hash_calls) == 2
        publishing[0] = True
        if source_changes:
            path.write_bytes(original + b"changed")
        try:
            yield deadline, {"scope": "verified_copy_publication_only"}
        finally:
            publishing[0] = False

    monkeypatch.setattr(compact, "_hash", checked_hash)
    result = _compact(path, tmp_path, publication_guard=publish)
    assert not publishing[0]
    if source_changes:
        assert "source_changed_during_compaction" in result["error"]
        assert not result["original_replaced"]
        assert path.read_bytes() == original + b"changed"
    else:
        assert result["status"] == "filesystem_compressed_verified", result
        assert path.read_bytes() == original
        assert (
            result["publication_handoff"]["scope"] == "verified_copy_publication_only"
        )


@pytest.mark.skipif(
    sys.platform != "darwin", reason="native APFS compression integration"
)
@pytest.mark.parametrize("operation", ["count", "retention", "integrity", "export"])
def test_archive_readers_and_noop_retention_preserve_compression(
    tmp_path, operation, native_apfs
):
    from datetime import datetime, timezone
    from scripts import sql_hot_retention as retention
    from scripts.ops import sql_link_shard_manager as manager

    path = _database(tmp_path)
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE jsonl_records (id INTEGER PRIMARY KEY, ingested_at TEXT)"
        )
        conn.execute(
            "INSERT INTO jsonl_records VALUES (1, ?)",
            (datetime.now(timezone.utc).isoformat(),),
        )
    conn.close()
    os.utime(path, (time.time() - 3 * 86400,) * 2)
    compressed = _compact(path, tmp_path)
    assert compressed["status"] == "filesystem_compressed_verified", compressed
    before = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if operation == "count":
        assert retention._count_archive_rows(path) == 1
    elif operation == "integrity":
        assert manager._sqlite_integrity_status(path, deep=True) == (True, "ok")
    elif operation == "export":
        pytest.importorskip("pyarrow")
        result = retention._export_sqlite_archive_to_parquet(
            path,
            out_path=tmp_path / "export.parquet",
            batch_size=1000,
            compression="zstd",
        )
        assert result["rows_exported"] == 1
    else:
        result = retention._prune_archive_storage(
            archive_db=path,
            archive_root=None,
            archive_retention_days=30,
            archive_prune_vacuum=True,
            cold_export_root=None,
            cold_export_format="parquet",
            cold_export_batch_size=1000,
            cold_export_compression="zstd",
        )
        assert result["pruned_rows"] == 0
        assert not result["errors"]
    assert path.stat().st_flags == before.st_flags
    assert path.stat().st_blocks == before.st_blocks
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_bounded_selection_skips_protected_alias_and_large_or_fresh_files(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(compact, "_require_idle", lambda *a: None)
    old = tmp_path / "old.sqlite3"
    fresh = tmp_path / "fresh.sqlite3"
    large = tmp_path / "large.sqlite3"
    for path, size in (
        (old, 110 * 1024**2),
        (fresh, 110 * 1024**2),
        (large, 3 * 1024**3),
    ):
        with path.open("wb") as handle:
            handle.truncate(size)
    for path in (old, large):
        os.utime(path, (time.time() - 172800,) * 2)
    (tmp_path / "protected_alias").symlink_to("/Volumes/VIDEO")
    assert compact.select_inactive_archives(
        tmp_path, max_files=4, max_raw_gb=8, min_age_hours=24
    ) == [old]
    assert set(
        compact.select_inactive_archives(
            tmp_path,
            max_files=4,
            max_raw_gb=8,
            min_age_hours=24,
            compressor="applesauce",
        )
    ) == {old, large}
    assert (
        compact.select_inactive_archives(
            tmp_path, max_files=4, max_raw_gb=0.01, min_age_hours=24
        )
        == []
    )


def test_bounded_selection_skips_busy_archive_before_spending_wave_budget(
    tmp_path, monkeypatch
):
    busy, idle = tmp_path / "a_busy.sqlite3", tmp_path / "b_idle.sqlite3"
    for path in (busy, idle):
        with path.open("wb") as handle:
            handle.truncate(110 * 1024**2)
        os.utime(path, (time.time() - 172800,) * 2)

    def probe(path, deadline):
        if path == busy:
            raise RuntimeError("cold_database_open_or_process_probe_failed")

    monkeypatch.setattr(compact, "_require_idle", probe)
    assert compact.select_inactive_archives(
        tmp_path, max_files=1, max_raw_gb=1, min_age_hours=24
    ) == [idle]


@pytest.mark.skipif(
    sys.platform != "darwin", reason="native APFS compression integration"
)
def test_cold_root_may_be_a_verified_storage_alias(tmp_path, native_apfs):
    root = tmp_path / "real"
    root.mkdir()
    path = _database(root)
    alias = tmp_path / "alias"
    alias.symlink_to(root)
    result = _compact(alias / path.name, alias)
    assert result["status"] == "filesystem_compressed_verified", result


@pytest.mark.skipif(
    sys.platform != "darwin", reason="native APFS compression integration"
)
@pytest.mark.parametrize(
    "failure", ["hash", "source_change", "manifest", "timeout", "open"]
)
@pytest.mark.parametrize("compressor", ["ditto", "applesauce"])
def test_failure_preserves_original(
    tmp_path, monkeypatch, failure, native_apfs, compressor
):
    if compressor == "applesauce" and not compact.streaming.installed():
        pytest.skip("pinned applesauce backend not installed")
    path = _database(tmp_path)
    original = path.read_bytes()
    if failure in {"hash", "source_change"}:
        real_hash = compact._hash

        def changed(candidate, deadline):
            value = real_hash(candidate, deadline)
            if candidate != path:
                if failure == "hash":
                    return "mismatch"
                os.utime(path, None)
            return value

        monkeypatch.setattr(compact, "_hash", changed)
    elif failure == "manifest":

        def fail_receipt(*_args):
            raise OSError("receipt failure")

        monkeypatch.setattr(compact, "_receipt", fail_receipt)
    elif failure == "open":

        def fail_idle(*_args):
            raise RuntimeError("open archive")

        monkeypatch.setattr(compact, "_require_idle", fail_idle)
    else:

        def expired(*_args):
            raise TimeoutError("deadline")

        monkeypatch.setattr(compact, "_remaining", expired)
    result = _compact(path, tmp_path, compressor=compressor)
    assert result["status"] == "failed"
    assert not result["original_replaced"]
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".filesystem_compaction_*"))
