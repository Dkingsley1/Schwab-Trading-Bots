from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from pathlib import Path
import pytest

from scripts.ops import governance_lifecycle_compactor as src


@pytest.fixture(autouse=True)
def resources(monkeypatch):
    class Guard:
        reserve = 16 * 1024**3

        def check(self):
            pass

        def progress(self, *args):
            pass

    monkeypatch.setattr(src.verified, "Guard", lambda *args: Guard())
    monkeypatch.setattr(src.verified, "background_policy", lambda: {})
    monkeypatch.setattr(src.verified, "idle", lambda path: None)


def _write_backup(
    project_root: Path, stamp: str, suffix: str = "coverage_gap_stage"
) -> Path:
    path = (
        project_root
        / "governance"
        / "lifecycle"
        / f"master_bot_registry.{suffix}_backup_{stamp}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"registry":"backup","bots":[{"id":"v10"}]}\n' * 64, encoding="utf-8"
    )
    old_epoch = time.time() - 3 * 86400
    os.utime(path, (old_epoch, old_epoch))
    return path


def test_lifecycle_compactor_dry_run_selects_old_backups(tmp_path: Path) -> None:
    source = _write_backup(tmp_path, "20260519_120000")

    payload = src.build_payload(
        project_root=tmp_path,
        apply=False,
        min_file_mb=0.000001,
        target_free_gb=0,
        max_files=4,
        min_age_hours=0,
        keep_latest=0,
    )

    assert payload["overall_status"] == "planned"
    assert payload["summary"]["candidate_count"] == 1
    assert (
        payload["records"][0]["relative_path"] == "governance/lifecycle/" + source.name
    )
    assert source.exists()


def test_lifecycle_compactor_apply_gzips_backup_in_place(tmp_path: Path) -> None:
    source = _write_backup(tmp_path, "20260519_120000")
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    payload = src.build_payload(
        project_root=tmp_path,
        apply=True,
        min_file_mb=0.000001,
        target_free_gb=0,
        max_files=4,
        min_age_hours=0,
        keep_latest=0,
        compression_level=1,
    )

    archive = source.with_name(source.name + ".gz")
    assert payload["overall_status"] == "applied"
    assert payload["summary"]["compacted_count"] == 1
    assert not source.exists()
    assert archive.exists()
    with gzip.open(archive, "rt", encoding="utf-8") as handle:
        content = handle.read()
    assert '"registry":"backup"' in content
    assert (
        payload["records"][0]["restore_proof"]["sha256_uncompressed"] == original_hash
    )
    assert (
        payload["records"][0]["restore_proof"]["verification"]
        == "full_gzip_restore_sha256_stable_idle_source"
    )
    receipts = [
        json.loads(line)
        for line in (
            tmp_path / "governance/storage_recovery/cold_evidence_compression.jsonl"
        )
        .read_text()
        .splitlines()
    ]
    assert [r["event"] for r in receipts] == [
        "verified_before_release",
        "original_replaced",
    ]


def test_lifecycle_compactor_keeps_recent_backups(tmp_path: Path) -> None:
    _write_backup(tmp_path, "20260518_120000")
    newest = _write_backup(tmp_path, "20260519_120000")
    now = time.time()
    os.utime(newest, (now, now))

    payload = src.build_payload(
        project_root=tmp_path,
        apply=False,
        min_file_mb=0.000001,
        target_free_gb=0,
        max_files=4,
        min_age_hours=0,
        keep_latest=1,
    )

    assert payload["summary"]["candidate_count"] == 1
    assert payload["records"][0]["relative_path"].endswith("20260518_120000.json")


def run_apply(root):
    return src.build_payload(
        project_root=root,
        apply=True,
        min_file_mb=0.000001,
        target_free_gb=0,
        max_files=4,
        min_age_hours=0,
        keep_latest=0,
    )


@pytest.mark.parametrize("matching", [False, True])
def test_existing_gzip_is_never_overwritten(tmp_path, matching):
    source = _write_backup(tmp_path, "20260519_120000")
    archive = source.with_name(source.name + ".gz")
    archive.write_bytes(
        gzip.compress(source.read_bytes() if matching else b"different content")
    )
    original = archive.read_bytes()
    result = run_apply(tmp_path)
    assert archive.read_bytes() == original
    assert result["ok"] is matching
    assert source.exists() is not matching


@pytest.mark.parametrize(
    "failure", ["receipt", "source_mutation", "resource", "open", "deadline"]
)
def test_failed_verification_or_resource_hold_preserves_source(
    tmp_path, monkeypatch, failure
):
    source = _write_backup(tmp_path, "20260519_120000")
    if failure == "receipt":
        monkeypatch.setattr(
            src.verified,
            "receipt",
            lambda *a: (_ for _ in ()).throw(OSError("receipt failed")),
        )
    elif failure == "source_mutation":
        digest = src.verified.digest_stream

        def mutate(*args):
            result = digest(*args)
            with source.open("ab") as f:
                f.write(b"new")
            return result

        monkeypatch.setattr(src.verified, "digest_stream", mutate)
    elif failure == "open":
        monkeypatch.setattr(
            src.verified,
            "idle",
            lambda *a: (_ for _ in ()).throw(src.verified.Deferred("file_open")),
        )
    else:

        class Held:
            def check(self):
                raise src.verified.Deferred(failure)

        monkeypatch.setattr(src.verified, "Guard", lambda *a: Held())
    result = run_apply(tmp_path)
    assert source.exists()
    assert not result["batch_complete"]
    assert not list(source.parent.glob(".cold_compact_*"))
    if failure in {"resource", "open", "deadline"}:
        assert result["ok"] and result["overall_status"] == "deferred"
    else:
        assert not result["ok"]


def test_lifecycle_inventory_rejects_symlinks_and_non_backups(tmp_path):
    source = _write_backup(tmp_path, "20260519_120000")
    alias = source.parent / "master_bot_registry.alias_backup_20260519.json"
    alias.symlink_to(source)
    (source.parent / "active.json").write_bytes(source.read_bytes())
    assert src._iter_lifecycle_json(tmp_path) == [source]


def test_protected_lifecycle_root_rejected_before_traversal(tmp_path):
    root = tmp_path / "governance/lifecycle"
    root.parent.mkdir()
    root.symlink_to("/Volumes/VIDEO/do-not-inspect", target_is_directory=True)
    with pytest.raises(RuntimeError, match="protected"):
        src._iter_lifecycle_json(tmp_path)


def test_shared_storage_lock_prevents_overlapping_apply(tmp_path):
    import fcntl

    source = _write_backup(tmp_path, "20260519_120000")
    lock = tmp_path / "governance/locks/storage_maintenance.lock"
    lock.parent.mkdir()
    with lock.open("a+") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = run_apply(tmp_path)
    assert result["overall_status"] == "deferred" and source.exists()
    assert result["records"][0]["reason"] == "storage_maintenance_lock_busy"
