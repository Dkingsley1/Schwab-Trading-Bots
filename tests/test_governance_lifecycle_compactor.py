from __future__ import annotations

import gzip
import os
import time
from pathlib import Path

from scripts.ops import governance_lifecycle_compactor as src


def _write_backup(project_root: Path, stamp: str, suffix: str = "coverage_gap_stage") -> Path:
    path = project_root / "governance" / "lifecycle" / f"master_bot_registry.{suffix}_backup_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"registry":"backup","bots":[{"id":"v10"}]}\n' * 64, encoding="utf-8")
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
    assert payload["records"][0]["relative_path"] == "governance/lifecycle/" + source.name
    assert source.exists()


def test_lifecycle_compactor_apply_gzips_backup_in_place(tmp_path: Path) -> None:
    source = _write_backup(tmp_path, "20260519_120000")

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
