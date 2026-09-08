import gzip
import os
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import bot_logs_cleanup_intelligence as cleanup


def _fake_disk(path: Path) -> dict:
    return {
        "path": str(path),
        "exists": True,
        "total_bytes": 10 * 1024**3,
        "used_bytes": 9 * 1024**3,
        "free_bytes": 1,
        "free_gb": 0.0,
        "used_gb": 9.0,
        "capacity_pct": 99.0,
    }


def _write_gzip_pair(raw_path: Path, content: bytes, *, age_hours: float = 48.0) -> None:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(content)
    with gzip.open(str(raw_path) + ".gz", "wb") as handle:
        handle.write(content)
    old_ts = time.time() - (age_hours * 3600.0)
    os.utime(raw_path, (old_ts, old_ts))
    os.utime(str(raw_path) + ".gz", (old_ts, old_ts))


def test_cleanup_selects_old_verified_gzip_duplicates(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    monkeypatch.setattr(cleanup, "_disk_snapshot", _fake_disk)
    _write_gzip_pair(root / "governance" / "execution_lanes" / "execution_results_20260501.jsonl", b'{"ok": true}\n' * 200)
    (root / "decisions" / "paper").mkdir(parents=True)
    (root / "decisions" / "paper" / "trade_decisions_20260501.jsonl").write_text("raw only\n", encoding="utf-8")

    payload = cleanup.build_payload(
        tmp_path,
        bot_logs_root=root,
        target_free_gb=1.0,
        max_tier=1,
        min_age_hours=12.0,
        out_path=tmp_path / "latest.json",
        history_path=tmp_path / "history.jsonl",
    )

    assert payload["candidate_summary"]["eligible_count"] == 1
    assert payload["selected_count"] == 1
    assert payload["selected_candidates"][0]["relative_path"] == "governance/execution_lanes/execution_results_20260501.jsonl"


def test_cleanup_apply_deletes_raw_and_keeps_gzip(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    monkeypatch.setattr(cleanup, "_disk_snapshot", _fake_disk)
    raw_path = root / "governance" / "execution_lanes" / "execution_results_20260501.jsonl"
    _write_gzip_pair(raw_path, b'{"row": 1}\n' * 300)

    payload = cleanup.build_payload(
        tmp_path,
        bot_logs_root=root,
        apply=True,
        target_free_gb=1.0,
        max_tier=1,
        min_age_hours=12.0,
        out_path=tmp_path / "latest.json",
        history_path=tmp_path / "history.jsonl",
    )

    assert payload["apply_result"]["deleted_files"] == 1
    assert not raw_path.exists()
    assert Path(str(raw_path) + ".gz").exists()
    assert (tmp_path / "history.jsonl").read_text(encoding="utf-8").strip()


def test_cleanup_protects_current_day_duplicates(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    monkeypatch.setattr(cleanup, "_disk_snapshot", _fake_disk)
    raw_path = root / "decisions" / "paper" / "trade_decisions_20990101.jsonl"
    _write_gzip_pair(raw_path, b'{"active": true}\n' * 100, age_hours=72.0)
    monkeypatch.setattr(cleanup, "_today_tokens", lambda now=None: {"20990101"})

    payload = cleanup.build_payload(
        tmp_path,
        bot_logs_root=root,
        target_free_gb=1.0,
        max_tier=1,
        min_age_hours=12.0,
        out_path=tmp_path / "latest.json",
        history_path=tmp_path / "history.jsonl",
    )

    assert payload["candidate_summary"]["eligible_count"] == 0
    assert payload["selected_count"] == 0
    assert payload["top_candidates"][0]["blocked_reasons"] == ["current_day_protected"]


def test_cleanup_offloads_external_local_fallback_conflicts(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    monkeypatch.setattr(cleanup, "_disk_snapshot", _fake_disk)
    monkeypatch.setattr(
        cleanup,
        "_quarantine_disk_snapshot",
        lambda path: {
            "path": str(path),
            "exists": True,
            "total_bytes": 200 * 1024**3,
            "used_bytes": 1 * 1024**3,
            "free_bytes": 199 * 1024**3,
            "free_gb": 199.0,
            "used_gb": 1.0,
            "capacity_pct": 0.5,
        },
    )
    conflict_path = (
        root
        / "decision_explanations"
        / "shadow_crypto"
        / "decision_explanations_20260507.jsonl.local_fallback.2"
    )
    conflict_path.parent.mkdir(parents=True, exist_ok=True)
    conflict_path.write_text("conflict-copy\n", encoding="utf-8")
    quarantine_root = tmp_path / "local_fallback_storage" / "quarantine" / "bot_logs_cleanup"

    payload = cleanup.build_payload(
        tmp_path,
        bot_logs_root=root,
        apply=True,
        target_free_gb=1.0,
        max_tier=2,
        fallback_quarantine_root=quarantine_root,
        out_path=tmp_path / "latest.json",
        history_path=tmp_path / "history.jsonl",
    )

    quarantined = quarantine_root / "decision_explanations" / "shadow_crypto" / conflict_path.name
    assert payload["apply_result"]["offloaded_files"] == 1
    assert payload["apply_result"]["deleted_files"] == 0
    assert not conflict_path.exists()
    assert quarantined.read_text(encoding="utf-8") == "conflict-copy\n"


def test_cleanup_blocks_conflict_offload_when_local_quarantine_low_space(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    monkeypatch.setattr(cleanup, "_disk_snapshot", _fake_disk)
    monkeypatch.setattr(
        cleanup,
        "_quarantine_disk_snapshot",
        lambda path: {
            "path": str(path),
            "exists": True,
            "total_bytes": 10 * 1024**3,
            "used_bytes": 9 * 1024**3,
            "free_bytes": 1,
            "free_gb": 0.0,
            "used_gb": 9.0,
            "capacity_pct": 99.0,
        },
    )
    conflict_path = (
        root
        / "decision_explanations"
        / "shadow_crypto"
        / "decision_explanations_20260507.jsonl.local_fallback.2"
    )
    conflict_path.parent.mkdir(parents=True, exist_ok=True)
    conflict_path.write_text("conflict-copy\n", encoding="utf-8")

    payload = cleanup.build_payload(
        tmp_path,
        bot_logs_root=root,
        apply=True,
        target_free_gb=1.0,
        max_tier=2,
        fallback_quarantine_root=tmp_path / "local_fallback_storage" / "quarantine" / "bot_logs_cleanup",
        out_path=tmp_path / "latest.json",
        history_path=tmp_path / "history.jsonl",
    )

    assert payload["apply_result"]["offloaded_files"] == 0
    assert conflict_path.exists()
    assert payload["top_candidates"][0]["blocked_reasons"] == ["quarantine_root_low_free_space"]


def test_cleanup_selection_respects_cumulative_quarantine_headroom(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    monkeypatch.setattr(cleanup, "_disk_snapshot", _fake_disk)
    first = root / "decisions" / "shadow_crypto" / "trade_decisions_20260507.jsonl.local_fallback.1"
    second = root / "decisions" / "shadow_crypto" / "trade_decisions_20260507.jsonl.local_fallback.2"
    first.parent.mkdir(parents=True, exist_ok=True)
    first.write_bytes(b"a" * 1024)
    second.write_bytes(b"b" * 1024)
    monkeypatch.setattr(
        cleanup,
        "_quarantine_disk_snapshot",
        lambda path: {
            "path": str(path),
            "exists": True,
            "total_bytes": (125 * 1024**3) + 1536,
            "used_bytes": 0,
            "free_bytes": (125 * 1024**3) + 1536,
            "free_gb": 125.0,
            "used_gb": 0.0,
            "capacity_pct": 0.0,
        },
    )

    payload = cleanup.build_payload(
        tmp_path,
        bot_logs_root=root,
        target_free_gb=1.0,
        max_tier=2,
        fallback_quarantine_root=tmp_path / "local_fallback_storage" / "quarantine" / "bot_logs_cleanup",
        out_path=tmp_path / "latest.json",
        history_path=tmp_path / "history.jsonl",
    )

    assert payload["candidate_summary"]["eligible_count"] == 2
    assert payload["selected_count"] == 1
    assert payload["selected_reclaimable_bytes"] == 1024


def test_cleanup_blocks_quarantine_when_reserve_budget_is_zero(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    conflict = root / "decisions" / "paper" / "trade_decisions_20260507.jsonl.local_fallback.1"
    conflict.parent.mkdir(parents=True)
    conflict.write_bytes(b"conflict" * 1024)
    old_ts = time.time() - (60 * 24 * 3600.0)
    os.utime(conflict, (old_ts, old_ts))
    monkeypatch.setattr(cleanup, "_disk_snapshot", _fake_disk)
    monkeypatch.setattr(
        cleanup,
        "_quarantine_disk_snapshot",
        lambda path: {
            "path": str(path),
            "exists": True,
            "total_bytes": 125 * 1024**3,
            "used_bytes": 0,
            "free_bytes": 125 * 1024**3,
            "free_gb": 125.0,
            "used_gb": 0.0,
            "capacity_pct": 0.0,
        },
    )

    payload = cleanup.build_payload(
        tmp_path,
        bot_logs_root=root,
        target_free_gb=1.0,
        max_tier=2,
        fallback_quarantine_root=tmp_path / "local_fallback_storage" / "quarantine" / "bot_logs_cleanup",
        out_path=tmp_path / "latest.json",
        history_path=tmp_path / "history.jsonl",
    )

    assert payload["selected_count"] == 0


def test_cleanup_prefers_stale_stage_deletion_over_local_quarantine(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    stale = root / "data" / "stale_stage" / "old" / "expired.jsonl.gz"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale" * 1024)
    conflict = root / "decisions" / "paper" / "trade_decisions_20260507.jsonl.local_fallback.1"
    conflict.parent.mkdir(parents=True)
    conflict.write_bytes(b"conflict" * 2048)
    old_ts = time.time() - (60 * 24 * 3600.0)
    os.utime(stale, (old_ts, old_ts))
    os.utime(conflict, (old_ts, old_ts))
    monkeypatch.setattr(cleanup, "_disk_snapshot", _fake_disk)
    monkeypatch.setattr(
        cleanup,
        "_quarantine_disk_snapshot",
        lambda path: {
            "path": str(path),
            "exists": True,
            "total_bytes": 300 * 1024**3,
            "used_bytes": 1 * 1024**3,
            "free_bytes": 299 * 1024**3,
            "free_gb": 299.0,
            "used_gb": 1.0,
            "capacity_pct": 0.4,
        },
    )

    payload = cleanup.build_payload(
        tmp_path,
        bot_logs_root=root,
        target_free_gb=0.000001,
        max_tier=2,
        fallback_quarantine_root=tmp_path / "local_fallback_storage" / "quarantine" / "bot_logs_cleanup",
        out_path=tmp_path / "latest.json",
        history_path=tmp_path / "history.jsonl",
    )

    assert payload["selected_count"] == 1
    assert payload["selected_candidates"][0]["tier_name"] == "stale_stage_reaper"


def test_cleanup_does_not_count_sparse_logical_bytes_as_reclaimable(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    sparse = root / "data" / "stale_stage" / "old" / "sparse.jsonl.gz"
    sparse.parent.mkdir(parents=True)
    with sparse.open("wb") as handle:
        handle.seek((8 * 1024**3) - 1)
        handle.write(b"\0")
    old_ts = time.time() - (60 * 24 * 3600.0)
    os.utime(sparse, (old_ts, old_ts))
    monkeypatch.setattr(cleanup, "_disk_snapshot", _fake_disk)
    monkeypatch.setattr(cleanup, "_file_allocated_size", lambda path: 0 if path == sparse else cleanup._file_size(path))

    payload = cleanup.build_payload(
        tmp_path,
        bot_logs_root=root,
        target_free_gb=1.0,
        max_tier=2,
        fallback_quarantine_root=tmp_path / "local_fallback_storage" / "quarantine" / "bot_logs_cleanup",
        out_path=tmp_path / "latest.json",
        history_path=tmp_path / "history.jsonl",
    )

    assert payload["candidate_summary"]["by_family"]["stale_stage"]["bytes"] == 0
    assert payload["selected_count"] == 0


def test_cleanup_skips_candidate_changed_after_scan(tmp_path: Path) -> None:
    candidate = tmp_path / "stale.jsonl.gz"
    candidate.write_bytes(b"first")
    row = {
        "tier": 2,
        "tier_name": "stale_stage_reaper",
        "relative_path": candidate.name,
        "path": str(candidate),
        "action": "delete",
        "reclaimable_bytes": cleanup._file_allocated_size(candidate),
        "source_identity": cleanup._file_identity(candidate),
    }
    candidate.write_bytes(b"changed")

    result = cleanup._apply_selected([row])

    assert result["deleted_files"] == 0
    assert result["skipped_files"] == 1
    assert candidate.exists()


def test_cleanup_apply_uses_actual_free_space_and_continues_after_shared_extent_estimate(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    first = root / "data" / "stale_stage" / "old" / "first.jsonl.gz"
    second = root / "data" / "stale_stage" / "old" / "second.jsonl.gz"
    first.parent.mkdir(parents=True)
    first.write_bytes(b"a" * 1024)
    second.write_bytes(b"b" * 2048)
    old_ts = time.time() - (60 * 24 * 3600.0)
    os.utime(first, (old_ts, old_ts))
    os.utime(second, (old_ts, old_ts))
    snapshots = iter(
        [
            {**_fake_disk(root), "free_bytes": 1},
            {**_fake_disk(root), "free_bytes": 1},
            {**_fake_disk(root), "free_bytes": 4096},
        ]
    )
    monkeypatch.setattr(cleanup, "_disk_snapshot", lambda path: next(snapshots))
    monkeypatch.setattr(cleanup, "_quarantine_disk_snapshot", _fake_disk)

    payload = cleanup.build_payload(
        tmp_path,
        bot_logs_root=root,
        apply=True,
        target_free_gb=0.0000009,
        max_tier=2,
        out_path=tmp_path / "latest.json",
        history_path=tmp_path / "history.jsonl",
    )

    assert payload["overall_status"] == "ready"
    assert payload["apply_result"]["apply_rounds"] == 2
    assert payload["apply_result"]["deleted_files"] == 2
    assert payload["apply_result"]["actual_reclaimed_bytes"] == 4095


def test_cleanup_quarantines_old_stateful_corrupt_sqlite_copy(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    data_root = root / "data"
    data_root.mkdir(parents=True)
    active = data_root / "bot_channel_queue.sqlite3"
    corrupt = data_root / "bot_channel_queue.sqlite3.corrupt-20260630193319311611"
    active.write_bytes(b"active-sqlite")
    corrupt.write_bytes(b"corrupt-copy")
    old_ts = time.time() - (48 * 3600.0)
    os.utime(corrupt, (old_ts, old_ts))
    monkeypatch.setattr(cleanup, "_disk_snapshot", _fake_disk)
    monkeypatch.setattr(
        cleanup,
        "_quarantine_disk_snapshot",
        lambda path: {
            "path": str(path),
            "exists": True,
            "total_bytes": 200 * 1024**3,
            "used_bytes": 1 * 1024**3,
            "free_bytes": 199 * 1024**3,
            "free_gb": 199.0,
            "used_gb": 1.0,
            "capacity_pct": 0.5,
        },
    )
    quarantine_root = tmp_path / "local_fallback_storage" / "quarantine" / "bot_logs_cleanup"

    payload = cleanup.build_payload(
        tmp_path,
        bot_logs_root=root,
        apply=True,
        target_free_gb=1.0,
        max_tier=2,
        fallback_quarantine_root=quarantine_root,
        out_path=tmp_path / "latest.json",
        history_path=tmp_path / "history.jsonl",
    )

    quarantined = quarantine_root / "stateful_corrupt" / "data" / corrupt.name
    assert payload["corrupt_sqlite_quarantine"]["eligible_count"] == 1
    assert payload["apply_result"]["offloaded_files"] == 1
    assert active.exists()
    assert not corrupt.exists()
    assert quarantined.read_bytes() == b"corrupt-copy"


def test_cleanup_blocks_young_stateful_corrupt_sqlite_copy(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    data_root = root / "data"
    data_root.mkdir(parents=True)
    active = data_root / "bot_channel_queue.sqlite3"
    corrupt = data_root / "bot_channel_queue.sqlite3.corrupt-20260630193319311611"
    active.write_bytes(b"active-sqlite")
    corrupt.write_bytes(b"corrupt-copy")
    monkeypatch.setattr(cleanup, "_disk_snapshot", _fake_disk)
    monkeypatch.setattr(
        cleanup,
        "_quarantine_disk_snapshot",
        lambda path: {
            "path": str(path),
            "exists": True,
            "total_bytes": 200 * 1024**3,
            "used_bytes": 1 * 1024**3,
            "free_bytes": 199 * 1024**3,
            "free_gb": 199.0,
            "used_gb": 1.0,
            "capacity_pct": 0.5,
        },
    )

    payload = cleanup.build_payload(
        tmp_path,
        bot_logs_root=root,
        target_free_gb=1.0,
        max_tier=2,
        fallback_quarantine_root=tmp_path / "local_fallback_storage" / "quarantine" / "bot_logs_cleanup",
        out_path=tmp_path / "latest.json",
        history_path=tmp_path / "history.jsonl",
    )

    assert payload["corrupt_sqlite_quarantine"]["candidate_count"] == 1
    assert payload["corrupt_sqlite_quarantine"]["eligible_count"] == 0
    assert payload["selected_count"] == 0
    assert payload["top_candidates"][0]["blocked_reasons"] == ["corrupt_sqlite_min_age_not_met"]
