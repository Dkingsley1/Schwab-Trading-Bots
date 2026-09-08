import argparse
import gzip
import hashlib
import json
import os
import time
from pathlib import Path

import pytest

from scripts.ops import raw_training_compaction_intelligence as raw_compaction


@pytest.fixture(autouse=True)
def sufficient_compaction_scratch(monkeypatch, tmp_path):
    usage = raw_compaction.shutil.disk_usage(tmp_path)
    monkeypatch.setattr(
        raw_compaction.shutil,
        "disk_usage",
        lambda _path: type(usage)(1024**4, 0, 1024**4),
    )
    monkeypatch.delenv("BOT_LOCAL_STORAGE_EMERGENCY_FREE_GB", raising=False)


def test_compaction_respects_higher_reserve_and_rejects_invalid_values(monkeypatch):
    monkeypatch.setenv("BOT_LOCAL_STORAGE_EMERGENCY_FREE_GB", "64")
    assert raw_compaction._compaction_reserve_bytes() == 64 * 1024**3
    for value in ("nan", "inf", "-1", "broken"):
        monkeypatch.setenv("BOT_LOCAL_STORAGE_EMERGENCY_FREE_GB", value)
        with pytest.raises(ValueError):
            raw_compaction._compaction_reserve_bytes()


def test_directory_sync_failure_preserves_raw(monkeypatch, tmp_path):
    path, target = tmp_path / "source.jsonl", tmp_path / "source.jsonl.gz"
    path.write_bytes(b"evidence")

    def fail_sync(_path):
        raise OSError("sync failed")

    monkeypatch.setattr(raw_compaction, "_sync_directory", fail_sync)
    result = raw_compaction._compress_and_clear(
        path, target, compress_level=1, keep_raw=False
    )
    assert result["status"] == "failed" and path.exists()
    assert gzip.decompress(target.read_bytes()) == b"evidence"


def test_gzip_verification_reads_trailer_and_rejects_truncation(tmp_path: Path) -> None:
    path = tmp_path / "truncated.gz"
    path.write_bytes(gzip.compress(b"evidence\n" * 10000)[:-8])
    assert not raw_compaction._verify_gzip(path)


def test_scan_does_not_follow_nested_directory_links(monkeypatch, tmp_path):
    (tmp_path / "forbidden").symlink_to("/Volumes/VIDEO")
    evidence = tmp_path / "evidence.jsonl"
    evidence.write_bytes(b"{}\n")

    def forbidden_walk(*_args, **_kwargs):
        raise AssertionError("os.walk can inspect symlink target metadata")

    monkeypatch.setattr(raw_compaction.os, "walk", forbidden_walk)
    assert list(raw_compaction._iter_jsonl_files(tmp_path)) == [evidence]


def test_duplicate_with_matching_prefix_and_different_tail_is_preserved(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source.jsonl"
    prefix = b"same prefix\n" * 1024
    path.write_bytes(prefix + b"original tail\n")
    target = tmp_path / "source.jsonl.gz"
    target.write_bytes(gzip.compress(prefix + b"different tail\n"))
    result = raw_compaction._remove_duplicate_raw(
        path,
        target,
        expected_prefix_sha256=hashlib.sha256(prefix[:4096]).hexdigest(),
        sample_bytes=4096,
    )
    assert result["reason"] == "compressed_sibling_content_mismatch"
    assert path.read_bytes() == prefix + b"original tail\n"
    assert gzip.decompress(target.read_bytes()) == prefix + b"different tail\n"


def test_compaction_preserves_existing_divergent_target(tmp_path: Path) -> None:
    path = tmp_path / "source.jsonl"
    path.write_bytes(b"new\n")
    target = tmp_path / "source.jsonl.gz"
    target.write_bytes(gzip.compress(b"old\n"))
    result = raw_compaction._compress_and_clear(
        path, target, compress_level=1, keep_raw=False
    )
    assert not result["raw_removed"]
    assert path.exists()
    assert gzip.decompress(target.read_bytes()) == b"old\n"


def test_keep_raw_applies_to_verified_duplicates(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    path = root / "evidence_20200101.jsonl"
    _write_old_jsonl(path)
    target = path.with_suffix(".jsonl.gz")
    target.write_bytes(gzip.compress(path.read_bytes()))
    result = raw_compaction.build_report(
        _args(tmp_path, root, apply=True, keep_raw_after_compress=True)
    )
    assert path.exists()
    assert not result["apply_records"][0]["raw_removed"]


def test_compaction_detects_same_size_source_change(
    monkeypatch, tmp_path: Path
) -> None:
    path = tmp_path / "source.jsonl"
    path.write_bytes(b"old\n")
    target = tmp_path / "source.jsonl.gz"
    digest = raw_compaction._digest_stream

    def mutate_after_verify(handle, deadline):
        result = digest(handle, deadline)
        path.write_bytes(b"new\n")
        return result

    monkeypatch.setattr(raw_compaction, "_digest_stream", mutate_after_verify)
    result = raw_compaction._compress_and_clear(
        path, target, compress_level=1, keep_raw=False
    )
    assert result["status"] == "failed"
    assert "source_changed" in result["reason"]
    assert path.read_bytes() == b"new\n"
    assert not target.exists()
    assert not list(tmp_path.glob(".raw_compact_*"))


def test_compaction_records_full_restore_hash(tmp_path: Path) -> None:
    path = tmp_path / "source.jsonl"
    content = b"full evidence\n" * 5000
    path.write_bytes(content)
    target = tmp_path / "source.jsonl.gz"
    result = raw_compaction._compress_and_clear(
        path, target, compress_level=1, keep_raw=False
    )
    assert result["sha256_uncompressed"] == hashlib.sha256(content).hexdigest()
    assert result["verified_raw_bytes"] == len(content)
    assert gzip.decompress(target.read_bytes()) == content
    assert target.stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob(".raw_compact_*"))


def test_compaction_denies_protected_alias_before_target_metadata(
    monkeypatch, tmp_path: Path
) -> None:
    alias = tmp_path / "forbidden"
    alias.symlink_to("/Volumes/VIDEO", target_is_directory=True)
    original = Path.lstat

    def guarded_lstat(path, *args, **kwargs):
        assert not str(path).lower().startswith("/volumes/video")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", guarded_lstat)
    rows, roots = raw_compaction._build_rows([alias], min_age_hours=0, sample_bytes=32)
    assert not rows and roots[0]["protected"]
    path = tmp_path / "source.jsonl"
    path.write_bytes(b"preserve\n")
    result = raw_compaction._compress_and_clear(
        path, alias / "out.gz", compress_level=1, keep_raw=False
    )
    assert result["status"] == "failed" and path.exists()


def test_compaction_scratch_shortage_preserves_source(
    monkeypatch, tmp_path: Path
) -> None:
    path = tmp_path / "source.jsonl"
    path.write_bytes(b"preserve\n")
    usage = raw_compaction.shutil.disk_usage(tmp_path)
    monkeypatch.setattr(
        raw_compaction.shutil,
        "disk_usage",
        lambda _path: type(usage)(usage.total, usage.used, 0),
    )
    result = raw_compaction._compress_and_clear(
        path, tmp_path / "out.gz", compress_level=1, keep_raw=False
    )
    assert "insufficient_compaction_scratch_reserve" in result["reason"]
    assert path.exists()


def test_compaction_deadline_preserves_source(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "source.jsonl"
    path.write_bytes(b"preserve\n")
    clock = iter([0.0, 301.0])
    monkeypatch.setattr(raw_compaction.time, "monotonic", lambda: next(clock))
    result = raw_compaction._compress_and_clear(
        path, tmp_path / "out.gz", compress_level=1, keep_raw=False
    )
    assert "compaction_deadline" in result["reason"]
    assert path.exists() and not list(tmp_path.glob(".raw_compact_*"))


def _args(tmp_path: Path, root: Path, **overrides):
    base = {
        "apply": False,
        "json": True,
        "bot_logs_root": str(root),
        "scan_root": [],
        "max_files": 12,
        "max_gb": 1.0,
        "jumbo_gb": 0.0,
        "min_age_hours": 1.0,
        "sample_bytes": 1024,
        "compress_level": 1,
        "compaction_workers": 1,
        "keep_raw_after_compress": False,
        "health_path": str(tmp_path / "health.json"),
        "manifest_path": str(tmp_path / "manifest.json"),
        "source_queue_path": str(tmp_path / "raw_training_source_queue_latest.jsonl"),
        "eligible_queue_path": str(
            tmp_path / "raw_training_eligible_source_queue_latest.jsonl"
        ),
        "write_history": False,
        "history_dir": str(tmp_path / "history"),
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _write_old_jsonl(
    path: Path, body: str = '{"x": 1}\n', age_seconds: int = 7200
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    old = time.time() - age_seconds
    os.utime(path, (old, old))


def test_queue_all_raw_sources_but_only_eligible_old_sources_compact(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bot_logs"
    old_decisions = root / "sleeve_a" / "decision_events_20260525.jsonl"
    fallback = root / "local_fallback" / "decision_events_20260525.jsonl"
    today = raw_compaction._utc_day()
    current_day = root / "sleeve_b" / f"decision_events_{today}.jsonl"
    _write_old_jsonl(old_decisions)
    _write_old_jsonl(fallback)
    _write_old_jsonl(current_day)

    payload = raw_compaction.build_report(_args(tmp_path, root))

    assert payload["raw_summary"]["raw_jsonl_count"] == 3
    assert payload["next_training_manifest"]["raw_source_queue_count"] == 3
    assert payload["next_training_manifest"]["raw_eligible_source_queue_count"] == 2
    assert payload["raw_summary"]["compression_candidate_count"] == 1
    assert payload["raw_summary"]["current_day_protected_count"] == 1
    assert payload["raw_summary"]["local_fallback_reconciliation_count"] == 1
    assert payload["decision_packet"]["blocked_reasons"] == []
    assert payload["decision_packet"]["managed_debts"] == [
        "submaterial_raw_compaction_tail"
    ]
    assert payload["raw_summary"]["raw_source_queue_coverage_ratio"] == 1.0
    assert payload["overall_status"] == "ready"
    assert payload["overall_grade"] == "A+"


def test_external_fallback_archive_is_evidence_not_live_reconciliation_debt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bot_logs"
    archived = (
        root
        / "local_fallback_storage"
        / "governance"
        / "shadow_dividend_equities"
        / "shadow_pnl_attribution_20260801.jsonl"
    )
    _write_old_jsonl(archived)

    payload = raw_compaction.build_report(_args(tmp_path, root))

    assert payload["raw_summary"]["local_fallback_reconciliation_count"] == 0
    assert payload["raw_summary"]["archived_fallback_evidence_count"] == 1
    assert payload["raw_summary"]["eligible_training_source_count"] == 1
    assert payload["raw_summary"]["compression_candidate_count"] == 1
    row = payload["top_training_sources"][0]
    assert row["archived_fallback_evidence"] is True
    assert row["local_fallback_reconciliation_required"] is False


def test_material_unapplied_compaction_debt_still_degrades(monkeypatch) -> None:
    monkeypatch.setenv("BOT_RAW_TRAINING_MATERIAL_COMPACTION_GB", "1.0")
    score, status, blockers, _actions = raw_compaction._score_report(
        {
            "raw_jsonl_count": 10,
            "raw_source_queue_count": 10,
            "compression_candidate_count": 3,
            "compression_candidate_gb": 2.0,
            "selected_compaction_count": 3,
            "apply_failed_count": 0,
        },
        False,
    )

    assert status == "needs_work"
    assert score < 90.0
    assert blockers == ["raw_compaction_not_applied"]


def test_apply_compresses_and_removes_eligible_raw_source(tmp_path: Path) -> None:
    root = tmp_path / "bot_logs"
    raw_path = root / "sleeve_a" / "shadow_decisions_20260525.jsonl"
    _write_old_jsonl(raw_path, '{"symbol":"SPY","action":"HOLD"}\n')

    payload = raw_compaction.build_report(
        _args(tmp_path, root, apply=True, max_files=1, max_gb=1.0)
    )
    gz_path = raw_path.with_name(raw_path.name + ".gz")

    assert payload["raw_summary"]["apply_record_count"] == 1
    assert payload["raw_summary"]["apply_failed_count"] == 0
    assert payload["raw_summary"]["raw_bytes_cleared"] > 0
    assert not raw_path.exists()
    assert gz_path.exists()
    with gzip.open(gz_path, "rt", encoding="utf-8") as handle:
        assert "SPY" in handle.read()


def test_apply_parallel_compaction_uses_bounded_independent_workers(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bot_logs"
    paths = [
        root / "sleeve_a" / f"shadow_decisions_2026052{index}.jsonl"
        for index in range(3)
    ]
    for index, raw_path in enumerate(paths):
        _write_old_jsonl(raw_path, f'{{"symbol":"SYM{index}","action":"HOLD"}}\n')

    payload = raw_compaction.build_report(
        _args(
            tmp_path,
            root,
            apply=True,
            max_files=3,
            max_gb=1.0,
            compaction_workers=2,
        )
    )

    assert payload["decision_packet"]["bounded_apply_caps"]["compaction_workers"] == 2
    assert (
        "parallel_workers_only_touch_independent_raw_files"
        in payload["decision_packet"]["risk_flags"]
    )
    assert payload["raw_summary"]["apply_record_count"] == 3
    assert payload["raw_summary"]["apply_failed_count"] == 0
    for raw_path in paths:
        assert not raw_path.exists()
        assert raw_path.with_name(raw_path.name + ".gz").exists()


def test_apply_never_compacts_active_latest_jsonl(tmp_path: Path) -> None:
    root = tmp_path / "bot_logs"
    raw_path = root / "exports" / "training" / "runtime_training_snapshot_latest.jsonl"
    _write_old_jsonl(raw_path, '{"sequence_id":"s1"}\n', age_seconds=7 * 86400)

    payload = raw_compaction.build_report(
        _args(tmp_path, root, apply=True, max_files=1, max_gb=1.0)
    )

    row = next(
        item
        for item in payload["top_training_sources"]
        if item["path"] == str(raw_path)
    )
    assert row["active_latest_artifact_protected"] is True
    assert row["compression_candidate"] is False
    assert "active_latest_artifact_protected" in row["compaction_blockers"]
    assert raw_path.exists()
    assert not raw_path.with_name(raw_path.name + ".gz").exists()


def test_training_clearance_resolves_external_snapshot_contract(tmp_path: Path) -> None:
    root = tmp_path / "bot_logs"
    runtime_path = (
        root / "exports" / "training" / "runtime_training_snapshot_latest.jsonl"
    )
    _write_old_jsonl(runtime_path, '{"sequence_id":"s1"}\n')
    health_path = tmp_path / "runtime_training_snapshot_latest.json"
    health_path.write_text(
        json.dumps(
            {
                "rows_path": str(tmp_path / "missing.jsonl"),
                "row_count": 10530,
                "sequence_count": 857,
            }
        ),
        encoding="utf-8",
    )

    clearance = raw_compaction._training_clearance_snapshot(
        [root], snapshot_health_path=health_path
    )

    assert clearance["runtime_snapshot_ready"] is True
    assert clearance["runtime_snapshot_path"] == str(runtime_path)
    assert clearance["runtime_snapshot_row_count"] == 10530
    assert clearance["runtime_snapshot_sequence_count"] == 857
    assert clearance["runtime_snapshot_source"] == "health_contract_and_resolved_rows"


def test_apply_removes_raw_duplicate_when_compressed_sibling_is_valid(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bot_logs"
    raw_path = root / "sleeve_a" / "paper_execution_intents_20260525.jsonl"
    _write_old_jsonl(raw_path, '{"symbol":"QQQ"}\n')
    gz_path = raw_path.with_name(raw_path.name + ".gz")
    with gzip.open(gz_path, "wt", encoding="utf-8") as handle:
        handle.write(raw_path.read_text(encoding="utf-8"))

    payload = raw_compaction.build_report(
        _args(tmp_path, root, apply=True, max_files=1, max_gb=1.0)
    )

    assert (
        payload["apply_records"][0]["action"]
        == "remove_raw_duplicate_of_compressed_sibling"
    )
    assert not raw_path.exists()
    assert gz_path.exists()


def test_apply_keeps_raw_when_compressed_sibling_prefix_mismatches(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bot_logs"
    raw_path = root / "sleeve_a" / "paper_execution_intents_20260525.jsonl"
    _write_old_jsonl(raw_path, '{"symbol":"QQQ"}\n')
    gz_path = raw_path.with_name(raw_path.name + ".gz")
    with gzip.open(gz_path, "wt", encoding="utf-8") as handle:
        handle.write('{"symbol":"SPY"}\n')

    payload = raw_compaction.build_report(
        _args(tmp_path, root, apply=True, max_files=1, max_gb=1.0)
    )
    repacked_path = raw_path.with_name(raw_path.name + ".raw-training.gz")

    assert (
        payload["apply_records"][0]["action"]
        == "repack_mismatched_sibling_then_remove_raw"
    )
    assert (
        payload["apply_records"][0]["original_compressed_sibling_reason"]
        == "compressed_sibling_prefix_mismatch"
    )
    assert not raw_path.exists()
    assert gz_path.exists()
    assert repacked_path.exists()
    with gzip.open(repacked_path, "rt", encoding="utf-8") as handle:
        assert "QQQ" in handle.read()


def test_video_volume_is_hard_protected() -> None:
    assert raw_compaction._is_under_protected_volume(Path("/Volumes/VIDEO"))
    assert raw_compaction._is_under_protected_volume(
        Path("/Volumes/VIDEO/schwab_trading_bot/raw.jsonl")
    )
    assert not raw_compaction._is_under_protected_volume(
        Path("/Volumes/BOT_LOGS/schwab_trading_bot/raw.jsonl")
    )


def test_opsctl_exposes_raw_training_compaction_command() -> None:
    opsctl = raw_compaction.PROJECT_ROOT / "scripts" / "ops" / "opsctl.sh"
    text = opsctl.read_text(encoding="utf-8")

    assert "raw-training-compaction" in text
    assert "raw-training-clear" in text
