import argparse
import gzip
import os
import time
from pathlib import Path

from scripts.ops import raw_training_compaction_intelligence as raw_compaction


def _args(tmp_path: Path, root: Path, **overrides):
    base = {
        "apply": False,
        "json": True,
        "bot_logs_root": str(root),
        "scan_root": [],
        "max_files": 12,
        "max_gb": 1.0,
        "min_age_hours": 1.0,
        "sample_bytes": 1024,
        "compress_level": 1,
        "keep_raw_after_compress": False,
        "health_path": str(tmp_path / "health.json"),
        "manifest_path": str(tmp_path / "manifest.json"),
        "source_queue_path": str(tmp_path / "raw_training_source_queue_latest.jsonl"),
        "eligible_queue_path": str(tmp_path / "raw_training_eligible_source_queue_latest.jsonl"),
        "write_history": False,
        "history_dir": str(tmp_path / "history"),
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _write_old_jsonl(path: Path, body: str = '{"x": 1}\n', age_seconds: int = 7200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    old = time.time() - age_seconds
    os.utime(path, (old, old))


def test_queue_all_raw_sources_but_only_eligible_old_sources_compact(tmp_path: Path) -> None:
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
    assert "raw_compaction_not_applied" in payload["decision_packet"]["blocked_reasons"]


def test_apply_compresses_and_removes_eligible_raw_source(tmp_path: Path) -> None:
    root = tmp_path / "bot_logs"
    raw_path = root / "sleeve_a" / "shadow_decisions_20260525.jsonl"
    _write_old_jsonl(raw_path, '{"symbol":"SPY","action":"HOLD"}\n')

    payload = raw_compaction.build_report(_args(tmp_path, root, apply=True, max_files=1, max_gb=1.0))
    gz_path = raw_path.with_name(raw_path.name + ".gz")

    assert payload["raw_summary"]["apply_record_count"] == 1
    assert payload["raw_summary"]["apply_failed_count"] == 0
    assert payload["raw_summary"]["raw_bytes_cleared"] > 0
    assert not raw_path.exists()
    assert gz_path.exists()
    with gzip.open(gz_path, "rt", encoding="utf-8") as handle:
        assert "SPY" in handle.read()


def test_apply_removes_raw_duplicate_when_compressed_sibling_is_valid(tmp_path: Path) -> None:
    root = tmp_path / "bot_logs"
    raw_path = root / "sleeve_a" / "paper_execution_intents_20260525.jsonl"
    _write_old_jsonl(raw_path, '{"symbol":"QQQ"}\n')
    gz_path = raw_path.with_name(raw_path.name + ".gz")
    with gzip.open(gz_path, "wt", encoding="utf-8") as handle:
        handle.write(raw_path.read_text(encoding="utf-8"))

    payload = raw_compaction.build_report(_args(tmp_path, root, apply=True, max_files=1, max_gb=1.0))

    assert payload["apply_records"][0]["action"] == "remove_raw_duplicate_of_compressed_sibling"
    assert not raw_path.exists()
    assert gz_path.exists()


def test_apply_keeps_raw_when_compressed_sibling_prefix_mismatches(tmp_path: Path) -> None:
    root = tmp_path / "bot_logs"
    raw_path = root / "sleeve_a" / "paper_execution_intents_20260525.jsonl"
    _write_old_jsonl(raw_path, '{"symbol":"QQQ"}\n')
    gz_path = raw_path.with_name(raw_path.name + ".gz")
    with gzip.open(gz_path, "wt", encoding="utf-8") as handle:
        handle.write('{"symbol":"SPY"}\n')

    payload = raw_compaction.build_report(_args(tmp_path, root, apply=True, max_files=1, max_gb=1.0))
    repacked_path = raw_path.with_name(raw_path.name + ".raw-training.gz")

    assert payload["apply_records"][0]["action"] == "repack_mismatched_sibling_then_remove_raw"
    assert payload["apply_records"][0]["original_compressed_sibling_reason"] == "compressed_sibling_prefix_mismatch"
    assert not raw_path.exists()
    assert gz_path.exists()
    assert repacked_path.exists()
    with gzip.open(repacked_path, "rt", encoding="utf-8") as handle:
        assert "QQQ" in handle.read()


def test_video_volume_is_hard_protected() -> None:
    assert raw_compaction._is_under_protected_volume(Path("/Volumes/VIDEO"))
    assert raw_compaction._is_under_protected_volume(Path("/Volumes/VIDEO/schwab_trading_bot/raw.jsonl"))
    assert not raw_compaction._is_under_protected_volume(Path("/Volumes/BOT_LOGS/schwab_trading_bot/raw.jsonl"))


def test_opsctl_exposes_raw_training_compaction_command() -> None:
    opsctl = raw_compaction.PROJECT_ROOT / "scripts" / "ops" / "opsctl.sh"
    text = opsctl.read_text(encoding="utf-8")

    assert "raw-training-compaction" in text
    assert "raw-training-clear" in text
