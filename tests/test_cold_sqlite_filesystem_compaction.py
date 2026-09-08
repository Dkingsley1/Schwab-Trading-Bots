import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import shutil
import time

import pytest

from scripts.ops import cold_sqlite_filesystem_compaction as compact


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
        **kwargs
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


@pytest.mark.skipif(
    sys.platform != "darwin", reason="native APFS compression integration"
)
@pytest.mark.parametrize("compressor", ["ditto", "afsctool"])
def test_native_compression_keeps_sqlite_readable_and_proves_all_bytes(
    tmp_path, compressor
):
    if compressor == "afsctool" and not shutil.which("afsctool"):
        pytest.skip("optional afsctool backend not installed")
    path = _database(tmp_path)
    original = path.read_bytes()
    result = _compact(path, tmp_path, compressor=compressor)
    assert result["status"] == "filesystem_compressed_verified", result
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
    assert _compact(path, tmp_path)["status"] == "already_compressed"


def test_large_archive_is_rejected_before_expensive_copy(tmp_path, monkeypatch):
    path = _database(tmp_path)
    monkeypatch.setattr(compact, "MAX_NATIVE_FILE_BYTES", 1024)
    monkeypatch.setattr(
        compact, "_run", lambda *a, **kw: pytest.fail("unexpected subprocess")
    )
    result = _compact(path, tmp_path)
    assert "archive_exceeds_bounded_native_compression_size" in result["error"]
    assert not result["original_replaced"]


@pytest.mark.skipif(
    sys.platform != "darwin", reason="native APFS compression integration"
)
@pytest.mark.parametrize("operation", ["count", "retention", "integrity", "export"])
def test_archive_readers_and_noop_retention_preserve_compression(tmp_path, operation):
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


def test_bounded_selection_skips_protected_alias_and_large_or_fresh_files(tmp_path):
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
    assert (
        compact.select_inactive_archives(
            tmp_path, max_files=4, max_raw_gb=0.01, min_age_hours=24
        )
        == []
    )


@pytest.mark.skipif(
    sys.platform != "darwin", reason="native APFS compression integration"
)
def test_cold_root_may_be_a_verified_storage_alias(tmp_path):
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
def test_failure_preserves_original(tmp_path, monkeypatch, failure):
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
    result = _compact(path, tmp_path)
    assert result["status"] == "failed"
    assert not result["original_replaced"]
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".filesystem_compaction_*"))
