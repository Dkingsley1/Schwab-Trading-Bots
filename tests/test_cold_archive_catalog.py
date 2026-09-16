import csv
import fcntl
import json
import os
from pathlib import Path

import pytest

from scripts.ops import cold_archive_catalog as catalog
from scripts.ops import cold_archive_compactor as compactor


def test_catalog_groups_dates_and_preserves_all_source_files(tmp_path):
    raw = (
        tmp_path
        / "sql_link_shards/archives/crypto_trading/jsonl_link_archive_2026_08_05.sqlite3"
    )
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"source bytes")
    wal = raw.with_name(raw.name + "-wal")
    wal.write_bytes(b"wal bytes")
    pending = (
        tmp_path
        / "sql_link_primary/.cold_20260914T000000Z.parquet.partial_dataset/part_1.parquet"
    )
    pending.parent.mkdir(parents=True)
    pending.write_bytes(b"unfinished")
    original = {
        path: (path.read_bytes(), path.stat().st_ino) for path in (raw, wal, pending)
    }
    result = catalog.build_catalog(tmp_path, apply=True)
    assert result["ok"] and result["complete"]
    assert result["file_count"] == 3
    by_path = {row["relative_path"]: row for row in result["files"]}
    row = by_path[str(raw.relative_to(tmp_path))]
    assert row["dataset"] == "sql_link_crypto_trading"
    assert row["filename_date"] == "2026-08-05"
    assert row["verification"] == "not_verified_by_catalog"
    assert (
        by_path[str(wal.relative_to(tmp_path))]["lifecycle"]
        == "sqlite_sidecar_keep_with_database"
    )
    assert (
        by_path[str(pending.relative_to(tmp_path))]["lifecycle"]
        == "incomplete_maintenance"
    )
    for path, (data, inode) in original.items():
        assert path.read_bytes() == data and path.stat().st_ino == inode
    assert all((tmp_path / name).is_file() for name in catalog.OUTPUTS)
    again = catalog.build_catalog(tmp_path, apply=True)
    assert again["file_count"] == 3


def test_catalog_skips_links_without_inspecting_targets(tmp_path):
    (tmp_path / "outside").symlink_to(
        "/nonexistent/protected_target", target_is_directory=True
    )
    result = catalog.build_catalog(tmp_path, apply=False)
    assert result["ok"] and result["symlinks_skipped"] == 1
    assert result["files"] == []


def test_protected_root_rejected_before_open(tmp_path, monkeypatch):
    monkeypatch.setattr(
        catalog, "inspect_storage_path", lambda path: {"status": "protected"}
    )
    monkeypatch.setattr(
        catalog.os, "open", lambda *a, **kw: pytest.fail("must not open")
    )
    assert (
        catalog.build_catalog(tmp_path, apply=True)["overall_status"]
        == "blocked_protected_root"
    )


def test_metadata_limit_is_explicit_partial_inventory(tmp_path):
    for number in range(3):
        (tmp_path / f"row_{number}.json").write_text("{}")
    result = catalog.build_catalog(tmp_path, apply=False, max_entries=1)
    assert not result["ok"] and not result["complete"]
    assert result["overall_status"] == "partial_inventory"
    assert result["file_count"] == 1


def test_output_symlink_is_not_followed_or_replaced(tmp_path):
    target = tmp_path / "original.txt"
    target.write_text("keep")
    (tmp_path / catalog.OUTPUTS[0]).symlink_to(target)
    with pytest.raises(ValueError, match="catalog_output_not_regular"):
        catalog.build_catalog(tmp_path, apply=True)
    assert target.read_text() == "keep"
    assert (tmp_path / catalog.OUTPUTS[0]).is_symlink()


def test_catalog_lease_does_not_preempt_or_replace_another_owner(tmp_path):
    path = tmp_path / catalog.LOCK_NAME
    with path.open("w") as owner:
        inode = path.stat().st_ino
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert (
            catalog.build_catalog(tmp_path, apply=True)["overall_status"]
            == "catalog_busy"
        )
        assert path.stat().st_ino == inode


def test_receipt_requires_complete_hash_proof(tmp_path):
    db = tmp_path / "archive.sqlite3"
    db.write_bytes(b"unchanged")
    proof = {
        "path": str(db),
        "status": "filesystem_compressed_verified",
        "original_replaced": True,
        "source_sha256": "a" * 64,
        "verified_copy_sha256": "b" * 64,
        "sqlite_quick_check": "ok",
    }
    (tmp_path / catalog.MANIFEST).write_text(json.dumps(proof) + "\n")
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        receipts, complete = catalog._receipt_index(fd, tmp_path, float("inf"))
    finally:
        os.close(fd)
    assert complete and not receipts


def test_csv_cannot_evaluate_filename_as_formula(tmp_path):
    (tmp_path / "=example.json").write_text("{}")
    catalog.build_catalog(tmp_path, apply=True)
    with (tmp_path / catalog.OUTPUTS[1]).open(newline="") as source:
        rows = list(csv.DictReader(source))
    assert rows[0]["relative_path"] == "'=example.json"
    assert "=example.json" in (tmp_path / catalog.OUTPUTS[2]).read_text()


def test_index_only_cli_never_requests_writer_or_maintenance_hold(
    tmp_path, monkeypatch
):
    (tmp_path / "archive_2026_09_14.json").write_text("{}")
    out = tmp_path / "health/receipt.json"
    monkeypatch.setattr(
        compactor.sys,
        "argv",
        [
            "cold-archive-compactor",
            "--index-only",
            "--apply",
            "--archive-root",
            str(tmp_path),
            "--out-file",
            str(out),
            "--json",
        ],
    )
    monkeypatch.setattr(
        compactor,
        "writer_state_snapshot",
        lambda *a: pytest.fail("metadata must not acquire writer"),
    )
    monkeypatch.setattr(
        compactor,
        "engage_maintenance_hold",
        lambda *a, **kw: pytest.fail("metadata must not acquire hold"),
    )
    assert compactor.main() == 0
    result = json.loads(out.read_text())
    assert (
        result["mode"] == "cold_archive_metadata_catalog" and result["file_count"] == 1
    )


@pytest.mark.parametrize("name", ["2026_99_99.sqlite3", "plain.sqlite3"])
def test_invalid_or_absent_dates_stay_undated(name):
    assert catalog._filename_date(Path(name)) == "undated"


@pytest.mark.parametrize(
    "path,expected",
    [
        ("content_store/hash.tmp.538", "incomplete_maintenance"),
        ("storage_split_brain/2026-08-25/old.log", "quarantined"),
        ("stateful_corrupt/old.sqlite3", "quarantined"),
        ("old.sqlite3.corrupt", "quarantined"),
        ("archive_2026_09_14.sqlite3.bak", "retained_backup"),
        ("archive_2026_09_14.sqlite3", "retained_archive"),
    ],
)
def test_retained_recovery_artifacts_are_separately_labeled(path, expected):
    assert catalog._lifecycle(Path(path), "other") == expected
