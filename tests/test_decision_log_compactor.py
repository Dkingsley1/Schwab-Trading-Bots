from __future__ import annotations

import gzip
import io
import json
import os
from pathlib import Path

import pytest

from scripts.ops import decision_log_compactor as src
from scripts.ops import cold_evidence_compactor as verified


def _write_decision_file(
    project_root: Path, day: str = "20260519", profile: str = "paper"
) -> Path:
    path = project_root / "decisions" / profile / f"trade_decisions_{day}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"timestamp_utc":"2026-05-19T00:00:00+00:00","symbol":"BTC-USD","action":"HOLD"}\n'
        * 16,
        encoding="utf-8",
    )
    return path


def _write_fallback_file(project_root: Path, day: str = "20260519") -> Path:
    path = (
        project_root
        / "decision_explanations"
        / "shadow_crypto"
        / f"decision_explanations_{day}.jsonl.local_fallback.1"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"timestamp_utc":"2026-05-19T00:00:00+00:00","symbol":"BTC-USD","reason":"fallback"}\n'
        * 16,
        encoding="utf-8",
    )
    return path


def _write_shadow_pnl_file(project_root: Path, day: str = "20260519") -> Path:
    path = (
        project_root
        / "governance"
        / "shadow_crypto"
        / f"shadow_pnl_attribution_{day}.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"timestamp_utc":"2026-05-19T00:00:00+00:00","symbol":"BTC-USD","pnl":0.0}\n'
        * 16,
        encoding="utf-8",
    )
    return path


def test_decision_log_compactor_dry_run_selects_old_large_files(tmp_path: Path) -> None:
    source = _write_decision_file(tmp_path)

    payload = src.build_payload(
        project_root=tmp_path,
        apply=False,
        min_file_mb=0.000001,
        target_free_gb=0,
        max_files=4,
        min_age_minutes=0,
    )

    assert payload["overall_status"] == "planned"
    assert payload["summary"]["candidate_count"] == 1
    assert payload["summary"]["selected_count"] == 1
    assert payload["records"][0]["relative_path"] == "decisions/paper/" + source.name
    assert source.exists()


def test_decision_log_compactor_apply_gzips_in_place(tmp_path: Path) -> None:
    source = _write_decision_file(tmp_path)

    payload = src.build_payload(
        project_root=tmp_path,
        apply=True,
        min_file_mb=0.000001,
        target_free_gb=0,
        max_files=4,
        min_age_minutes=0,
        compression_level=1,
    )

    archive = source.with_name(source.name + ".gz")
    assert payload["overall_status"] == "applied"
    assert payload["summary"]["compacted_count"] == 1
    assert not source.exists()
    assert archive.exists()
    with gzip.open(archive, "rt", encoding="utf-8") as handle:
        content = handle.read()
    assert content.count("\n") == 16
    assert "BTC-USD" in content


def test_decision_log_compactor_preserves_conflicting_existing_archive(
    tmp_path: Path,
) -> None:
    source = _write_decision_file(tmp_path)
    archive = source.with_name(source.name + ".gz")
    with gzip.open(archive, "wt", encoding="utf-8") as handle:
        handle.write('{"symbol":"LEGACY_SYMBOL"}\n')

    payload = src.build_payload(
        project_root=tmp_path,
        apply=True,
        min_file_mb=0.000001,
        target_free_gb=0,
        max_files=4,
        min_age_minutes=0,
        compression_level=1,
    )

    assert payload["overall_status"] == "degraded"
    assert payload["records"][0]["error"] == "full_gzip_restore_sha256_mismatch"
    assert source.exists()
    with gzip.open(archive, "rt", encoding="utf-8") as handle:
        content = handle.read()
    assert "LEGACY_SYMBOL" in content
    assert not list(source.parent.glob(".decision_compact_*.tmp"))


def test_decision_log_compactor_skips_current_day_by_default(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(src, "_today_stamp", lambda: "20260522")
    _write_decision_file(tmp_path, day="20260522")

    payload = src.build_payload(
        project_root=tmp_path,
        apply=False,
        min_file_mb=0.000001,
        target_free_gb=0,
        max_files=4,
        min_age_minutes=0,
    )

    assert payload["overall_status"] == "nothing_to_do"
    assert payload["summary"]["candidate_count"] == 0


def test_decision_log_compactor_includes_local_fallback_spillover(
    tmp_path: Path,
) -> None:
    source = _write_fallback_file(tmp_path)

    payload = src.build_payload(
        project_root=tmp_path,
        apply=False,
        families=["decision_explanations"],
        min_file_mb=0.000001,
        target_free_gb=0,
        max_files=4,
        min_age_minutes=0,
    )

    assert payload["overall_status"] == "planned"
    assert (
        payload["records"][0]["relative_path"]
        == "decision_explanations/shadow_crypto/" + source.name
    )
    assert payload["records"][0]["fallback_copy"] is True


def test_decision_log_compactor_applies_to_local_fallback_spillover(
    tmp_path: Path,
) -> None:
    source = _write_fallback_file(tmp_path)

    payload = src.build_payload(
        project_root=tmp_path,
        apply=True,
        families=["decision_explanations"],
        min_file_mb=0.000001,
        target_free_gb=0,
        max_files=4,
        min_age_minutes=0,
        compression_level=1,
    )

    archive = source.with_name(source.name + ".gz")
    assert payload["overall_status"] == "applied"
    assert not source.exists()
    assert archive.exists()


def test_decision_log_compactor_includes_shadow_pnl_attribution_family(
    tmp_path: Path,
) -> None:
    source = _write_shadow_pnl_file(tmp_path)

    payload = src.build_payload(
        project_root=tmp_path,
        apply=False,
        families=["shadow_pnl_attribution"],
        min_file_mb=0.000001,
        target_free_gb=0,
        max_files=4,
        min_age_minutes=0,
    )

    assert payload["overall_status"] == "planned"
    assert payload["summary"]["candidate_count"] == 1
    assert (
        payload["records"][0]["relative_path"]
        == "governance/shadow_crypto/" + source.name
    )


def _checkpoint(root, source, **changes):
    row = {
        "last_offset_bytes": source.stat().st_size,
        "file_size_bytes": source.stat().st_size,
        "file_inode": source.stat().st_ino,
        "mtime": source.stat().st_mtime,
    }
    row.update(changes)
    state = root / "governance/sql_link_shards/jsonl_sql_link_state_trading.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps({"sqlite": {str(source.relative_to(root)): row}}))


@pytest.mark.parametrize(
    "changes",
    [
        {"last_offset_bytes": 0},
        {"file_size_bytes": 0},
        {"file_inode": 1},
        {"last_offset_bytes": "invalid"},
        {"file_inode": None},
        {"mtime": 0},
        {"mtime": float("nan")},
        {"file_size_bytes": True},
    ],
)
def test_known_pending_or_invalid_checkpoint_skips_even_old_logs(tmp_path, changes):
    source = _write_decision_file(tmp_path)
    _checkpoint(tmp_path, source, **changes)
    payload = src.build_payload(
        project_root=tmp_path, min_file_mb=0.000001, min_age_minutes=0
    )
    assert payload["summary"]["selected_count"] == 0
    assert source.exists()


def test_exact_checkpoint_allows_old_file_and_durable_restore_proof(tmp_path):
    source = _write_decision_file(tmp_path)
    raw = source.read_bytes()
    _checkpoint(tmp_path, source)
    payload = src.build_payload(
        project_root=tmp_path, apply=True, min_file_mb=0.000001, min_age_minutes=0
    )
    proof = payload["records"][0]["restore_proof"]
    assert payload["overall_status"] == "applied"
    assert proof["verified_restored_bytes"] == len(raw)
    assert gzip.decompress(Path(str(source) + ".gz").read_bytes()) == raw
    receipts = [
        json.loads(line)
        for line in (
            tmp_path / "governance/storage_recovery/cold_evidence_compression.jsonl"
        )
        .read_text()
        .splitlines()
    ]
    assert [r["event"] for r in receipts] == [
        "decision_verified_before_release",
        "decision_original_replaced",
    ]


def test_matching_existing_archive_reused_without_overwrite(tmp_path):
    source = _write_decision_file(tmp_path)
    archive = Path(str(source) + ".gz")
    archive.write_bytes(gzip.compress(source.read_bytes()))
    before = archive.read_bytes(), archive.stat().st_ino
    result = src._compact_one(
        project_root=tmp_path,
        source_rel=str(source.relative_to(tmp_path)),
        compression_level=1,
    )
    assert result["status"] == "compacted"
    assert result["matching_archive_reused"] is True
    assert result["archive_replaced"] is False
    assert (archive.read_bytes(), archive.stat().st_ino) == before


def test_corrupt_restored_bytes_preserve_original(tmp_path, monkeypatch):
    source = _write_decision_file(tmp_path)
    raw = source.read_bytes()
    original = gzip.open

    def corrupt(path, mode, **kwargs):
        return (
            io.BytesIO(b"x" * len(raw))
            if mode == "rb"
            else original(path, mode, **kwargs)
        )

    monkeypatch.setattr(src.gzip, "open", corrupt)
    result = src._compact_one(
        project_root=tmp_path,
        source_rel=str(source.relative_to(tmp_path)),
        compression_level=1,
    )
    assert result["status"] == "error"
    assert source.read_bytes() == raw
    assert not Path(str(source) + ".gz").exists()
    assert not list(source.parent.glob(".decision_compact_*.tmp"))


def test_archive_publication_race_preserves_both_sources(tmp_path, monkeypatch):
    source = _write_decision_file(tmp_path)
    raw = source.read_bytes()
    archive = Path(str(source) + ".gz")
    original_link = os.link

    def race(src_path, dst, **kwargs):
        Path(dst).write_bytes(b"another archive")
        return original_link(src_path, dst, **kwargs)

    monkeypatch.setattr(src.os, "link", race)
    result = src._compact_one(
        project_root=tmp_path,
        source_rel=str(source.relative_to(tmp_path)),
        compression_level=1,
    )
    assert result["status"] == "error"
    assert source.read_bytes() == raw
    assert archive.read_bytes() == b"another archive"


def test_source_mutation_after_proof_preserves_original(tmp_path, monkeypatch):
    source = _write_decision_file(tmp_path)
    original_receipt = verified.receipt

    def mutate(root, payload):
        original_receipt(root, payload)
        source.write_bytes(source.read_bytes() + b"new row\n")

    monkeypatch.setattr(verified, "receipt", mutate)
    result = src._compact_one(
        project_root=tmp_path,
        source_rel=str(source.relative_to(tmp_path)),
        compression_level=1,
    )
    assert result["error"] == "source_or_archive_changed_before_release"
    assert source.read_bytes().endswith(b"new row\n")


def test_failed_durable_proof_preserves_original(tmp_path, monkeypatch):
    source = _write_decision_file(tmp_path)

    def fail(*args):
        raise OSError("receipt disk failure")

    monkeypatch.setattr(verified, "receipt", fail)
    result = src._compact_one(
        project_root=tmp_path,
        source_rel=str(source.relative_to(tmp_path)),
        compression_level=1,
    )
    assert result["error"] == "receipt disk failure"
    assert source.exists()


def test_pending_checkpoint_rechecked_before_release(tmp_path, monkeypatch):
    source = _write_decision_file(tmp_path)
    _checkpoint(tmp_path, source)
    original_receipt = verified.receipt

    def reset(root, payload):
        original_receipt(root, payload)
        _checkpoint(tmp_path, source, last_offset_bytes=0)

    monkeypatch.setattr(verified, "receipt", reset)
    result = src._compact_one(
        project_root=tmp_path,
        source_rel=str(source.relative_to(tmp_path)),
        compression_level=1,
    )
    assert result["error"] == "ingestion_checkpoint_changed_before_release"
    assert source.exists()


def test_missing_checkpoint_after_proof_preserves_original(tmp_path, monkeypatch):
    source = _write_decision_file(tmp_path)
    _checkpoint(tmp_path, source)
    original_receipt = verified.receipt

    def remove(root, payload):
        original_receipt(root, payload)
        (
            tmp_path / "governance/sql_link_shards/jsonl_sql_link_state_trading.json"
        ).unlink()

    monkeypatch.setattr(verified, "receipt", remove)
    result = src._compact_one(
        project_root=tmp_path,
        source_rel=str(source.relative_to(tmp_path)),
        compression_level=1,
    )
    assert result["error"] == "ingestion_checkpoint_changed_before_release"
    assert source.exists()


def test_busy_source_preserved(tmp_path, monkeypatch):
    source = _write_decision_file(tmp_path)

    def busy(path):
        raise RuntimeError("source_open_by_process")

    monkeypatch.setattr(verified, "idle", busy)
    result = src._compact_one(
        project_root=tmp_path,
        source_rel=str(source.relative_to(tmp_path)),
        compression_level=1,
    )
    assert result["error"] == "source_open_by_process"
    assert source.exists()
    assert not Path(str(source) + ".gz").exists()


def test_interrupt_removes_scratch_not_source(tmp_path, monkeypatch):
    source = _write_decision_file(tmp_path)

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(src.gzip, "open", interrupt)
    with pytest.raises(KeyboardInterrupt):
        src._compact_one(
            project_root=tmp_path,
            source_rel=str(source.relative_to(tmp_path)),
            compression_level=1,
        )
    assert source.exists()
    assert not list(source.parent.glob(".decision_compact_*.tmp"))


def test_checkpoint_change_before_copy_is_visible_deferral(tmp_path, monkeypatch):
    source = _write_decision_file(tmp_path)
    candidate = src._candidate_rows(
        project_root=tmp_path,
        min_file_bytes=1,
        include_current_day=False,
        min_age_minutes=0,
        families=["decisions"],
        require_current_day_safe=True,
    )
    _checkpoint(tmp_path, source, last_offset_bytes=0)
    monkeypatch.setattr(src, "_candidate_rows", lambda **kwargs: candidate)
    result = src.build_payload(project_root=tmp_path, apply=True)
    assert result["overall_status"] == "deferred"
    assert result["ok"] is False
    assert result["summary"]["deferred_count"] == 1
    assert result["summary"]["compacted_count"] == 0
    assert source.exists()


def test_safe_directory_route_preserves_external_archive_support(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    external = tmp_path / "external"
    source = _write_decision_file(external)
    (root / "decisions").symlink_to(external / "decisions", target_is_directory=True)
    result = src._compact_one(
        project_root=root,
        source_rel="decisions/paper/" + source.name,
        compression_level=1,
    )
    assert result["status"] == "compacted"
    assert (root / "decisions").is_symlink()
    assert Path(str(source) + ".gz").exists()


def test_file_symlink_is_preserved(tmp_path):
    source = _write_decision_file(tmp_path)
    alias = source.with_name("alias.jsonl")
    alias.symlink_to(source)
    result = src._compact_one(
        project_root=tmp_path,
        source_rel=str(alias.relative_to(tmp_path)),
        compression_level=1,
    )
    assert result["error"] == "source_file_symlink_not_eligible"
    assert source.exists() and alias.is_symlink()


def test_retargeted_directory_route_preserves_original(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    external, other = tmp_path / "external", tmp_path / "other"
    source = _write_decision_file(external)
    _write_decision_file(other)
    alias = root / "decisions"
    alias.symlink_to(external / "decisions", target_is_directory=True)
    original_receipt = verified.receipt

    def retarget(root, payload):
        original_receipt(root, payload)
        alias.unlink()
        alias.symlink_to(other / "decisions", target_is_directory=True)

    monkeypatch.setattr(verified, "receipt", retarget)
    result = src._compact_one(
        project_root=root,
        source_rel="decisions/paper/" + source.name,
        compression_level=1,
    )
    assert result["error"] == "source_route_changed_before_release"
    assert source.exists()
