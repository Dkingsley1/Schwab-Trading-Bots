import gzip
import fcntl
import json
import os
import sys
import time
from pathlib import Path
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import bot_logs_cleanup_intelligence as cleanup


@pytest.fixture(autouse=True)
def admitted_cleanup(monkeypatch):
    class Guard:
        def __init__(self, *args):
            pass

        def check(self):
            pass

    monkeypatch.setattr(cleanup.verified.safety, "Guard", Guard)
    monkeypatch.setattr(cleanup.verified.safety, "background_policy", lambda: None)
    monkeypatch.setattr(cleanup.verified.safety, "idle", lambda path: None)


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
    _write_gzip_pair(root / "logs" / "runtime_20260501.jsonl", b'{"ok": true}\n' * 200)
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
    assert payload["selected_candidates"][0]["relative_path"] == "logs/runtime_20260501.jsonl"


def test_cleanup_apply_deletes_raw_and_keeps_gzip(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "BOT_LOGS" / "schwab_trading_bot"
    monkeypatch.setattr(cleanup, "_disk_snapshot", _fake_disk)
    raw_path = root / "logs" / "runtime_20260501.jsonl"
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
    raw_path = root / "logs" / "runtime_20990101.jsonl"
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


def test_cleanup_preserves_conflicts_for_verified_offload_owner(tmp_path: Path, monkeypatch) -> None:
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
    assert payload["apply_result"]["offloaded_files"] == 0
    assert payload["apply_result"]["deleted_files"] == 0
    assert conflict_path.read_text(encoding="utf-8") == "conflict-copy\n"
    assert not quarantined.exists()
    assert "verified_offload_owner_required" in payload["top_candidates"][0]["blocked_reasons"]


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
    assert "quarantine_root_low_free_space" in payload["top_candidates"][0]["blocked_reasons"]


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

    assert payload["candidate_summary"]["eligible_count"] == 0
    assert payload["selected_count"] == 0
    assert payload["selected_reclaimable_bytes"] == 0
    rows = cleanup._scan_external_local_fallback_copies(root, project_root=tmp_path,
        fallback_quarantine_root=tmp_path / "local_fallback_storage/quarantine/bot_logs_cleanup")
    assert len(cleanup._select_candidates(rows, free_bytes=0, target_free_bytes=1024**3,
                                         max_tier=2, max_delete_bytes=1024**3)) == 1


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


def test_cleanup_delegates_stale_stage_to_manifest_retention_owner(tmp_path: Path, monkeypatch) -> None:
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

    assert payload["selected_count"] == 0
    assert stale.exists()
    assert any("manifest_retention_owner_required" in row["blocked_reasons"] for row in payload["top_candidates"])


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
    first = root / "logs" / "first_20260501.jsonl"
    second = root / "logs" / "second_20260501.jsonl"
    _write_gzip_pair(first, b"a" * 1024)
    _write_gzip_pair(second, b"b" * 2048)
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


def test_cleanup_preserves_corrupt_sqlite_for_verified_offload_owner(tmp_path: Path, monkeypatch) -> None:
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
    assert payload["corrupt_sqlite_quarantine"]["eligible_count"] == 0
    assert payload["apply_result"]["offloaded_files"] == 0
    assert active.exists()
    assert corrupt.read_bytes() == b"corrupt-copy"
    assert not quarantined.exists()


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
    assert "corrupt_sqlite_min_age_not_met" in payload["top_candidates"][0]["blocked_reasons"]


@pytest.mark.parametrize("damage", ["tail", "short", "crc", "truncated", "extra_member"])
def test_full_verification_rejects_matching_prefix_with_bad_archive(tmp_path, monkeypatch, damage):
    raw = tmp_path / "decisions/rows_20260501.jsonl"
    content = b"a" * 65536 + b"original-tail"
    _write_gzip_pair(raw, content)
    archive = Path(str(raw) + ".gz")
    if damage in {"tail", "short"}:
        replacement = content[:-1] + b"x" if damage == "tail" else content[:-1]
        archive.write_bytes(gzip.compress(replacement))
    elif damage == "crc":
        data = bytearray(archive.read_bytes())
        data[-8] ^= 1
        archive.write_bytes(data)
    elif damage == "truncated":
        archive.write_bytes(archive.read_bytes()[:-4])
    else:
        archive.write_bytes(archive.read_bytes() + gzip.compress(b"extra"))
    proof = cleanup._gzip_duplicate_verification(raw, archive, prefix_bytes=1)
    assert proof["ok"] is False
    with pytest.raises((OSError, RuntimeError, EOFError)):
        cleanup.verified.remove_pair(tmp_path, raw, archive, cleanup.verified.Budget())
    assert raw.read_bytes() == content


@pytest.mark.parametrize("changed", ["raw", "archive", "receipt_failure"])
def test_verification_rechecks_both_files_after_durable_proof(tmp_path, monkeypatch, changed):
    raw = tmp_path / "logs/rows_20260501.jsonl"
    _write_gzip_pair(raw, b"original")
    archive = Path(str(raw) + ".gz")

    def mutate(root, record):
        if changed == "receipt_failure":
            raise OSError("fsync failed")
        target = raw if changed == "raw" else archive
        target.write_bytes(b"changed")

    monkeypatch.setattr(cleanup.verified, "receipt", mutate)
    with pytest.raises((OSError, RuntimeError)):
        cleanup.verified.remove_pair(tmp_path, raw, archive, cleanup.verified.Budget())
    assert raw.exists()


def test_valid_duplicate_has_complete_durable_proof(tmp_path):
    raw = tmp_path / "logs/rows_20260501.jsonl"
    _write_gzip_pair(raw, b"content" * 1024)
    archive = Path(str(raw) + ".gz")
    proof = cleanup.verified.remove_pair(tmp_path, raw, archive, cleanup.verified.Budget())
    rows = [json.loads(line) for line in (tmp_path / "governance/storage_recovery/verified_duplicate_cleanup.jsonl").read_text().splitlines()]
    assert [row["phase"] for row in rows] == ["verified_before_release", "source_released"]
    assert proof["verified_bytes"] == 7 * 1024
    assert rows[0]["sha256"] == rows[1]["sha256"] == proof["sha256"]
    assert not raw.exists() and gzip.decompress(archive.read_bytes()) == b"content" * 1024


@pytest.mark.parametrize("reason", ["source_open", "idle_probe_unknown"])
def test_open_or_unknown_handles_prevent_release(tmp_path, monkeypatch, reason):
    raw = tmp_path / "logs/rows_20260501.jsonl"
    _write_gzip_pair(raw, b"content")
    def held(path):
        raise cleanup.verified.safety.Deferred(reason)
    monkeypatch.setattr(cleanup.verified.safety, "idle", held)
    with pytest.raises(RuntimeError, match=reason):
        cleanup.verified.remove_pair(tmp_path, raw, Path(str(raw) + ".gz"), cleanup.verified.Budget())
    assert raw.exists()


@pytest.mark.parametrize("kind", ["deadline", "bytes", "hardlink", "symlink"])
def test_budgets_and_linked_sources_preserve_raw(tmp_path, kind):
    raw = tmp_path / "logs/rows_20260501.jsonl"
    _write_gzip_pair(raw, b"content" * 100)
    archive = Path(str(raw) + ".gz")
    budget = cleanup.verified.Budget(seconds=-1 if kind == "deadline" else 30,
                                     max_bytes=10 if kind == "bytes" else 1024**2)
    if kind == "hardlink":
        os.link(raw, tmp_path / "alias")
    if kind == "symlink":
        archive.rename(tmp_path / "retained.gz")
        archive.symlink_to(tmp_path / "retained.gz")
    with pytest.raises(RuntimeError):
        cleanup.verified.remove_pair(tmp_path, raw, archive, budget)
    assert raw.exists()


def test_protected_root_rejected_before_disk_probe(tmp_path, monkeypatch):
    def forbidden_probe(path):
        raise AssertionError("must not probe protected root")
    monkeypatch.setattr(cleanup, "_disk_snapshot", forbidden_probe)
    payload = cleanup.build_payload(tmp_path, bot_logs_root=Path("/Volumes/VIDEO/never-access"),
                                    out_path=tmp_path / "latest.json", history_path=tmp_path / "history.jsonl")
    assert payload["overall_status"] == "deferred"
    assert payload["assessment_complete"] is False


def test_inventory_does_not_follow_protected_alias_and_is_bounded(tmp_path):
    (tmp_path / "alias").symlink_to("/Volumes/VIDEO/never-access", target_is_directory=True)
    (tmp_path / "safe").write_bytes(b"safe")
    assert cleanup.verified.inventory(tmp_path) == [tmp_path / "safe"]
    with pytest.raises(RuntimeError, match="inventory_budget"):
        cleanup.verified.inventory(tmp_path, max_entries=1)


def test_storage_lock_contention_and_admission_denial_defer(tmp_path, monkeypatch):
    lock_path = tmp_path / "governance/locks/storage_maintenance.lock"
    lock_path.parent.mkdir(parents=True)
    kwargs = dict(bot_logs_root=tmp_path, apply=True, out_path=tmp_path / "latest.json", history_path=tmp_path / "history.jsonl")
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        payload = cleanup.build_payload(tmp_path, **kwargs)
        assert payload["reason"] == "storage_maintenance_lock_busy"
    class Denied:
        def __init__(self, *args):
            pass
        def check(self):
            raise cleanup.verified.safety.Deferred("maintenance_hold")
    monkeypatch.setattr(cleanup.verified.safety, "Guard", Denied)
    assert cleanup.build_payload(tmp_path, **kwargs)["reason"] == "maintenance_hold"


def test_first_oversized_file_cannot_bypass_delete_cap():
    row = {"eligible": True, "tier": 1, "reclaimable_bytes": 1000}
    assert cleanup._select_candidates([row], free_bytes=0, target_free_bytes=2000,
                                      max_tier=1, max_delete_bytes=100) == []


def test_retained_and_latest_artifacts_are_not_duplicate_cleanup_candidates(tmp_path, monkeypatch):
    monkeypatch.setattr(cleanup, "_disk_snapshot", _fake_disk)
    for name in ("exports/training/rows_20260501.jsonl", "cold_archive/rows_20260501.jsonl",
                 "governance/rows_latest_20260501.jsonl", "data/stale_stage/rows_20260501.jsonl"):
        _write_gzip_pair(tmp_path / name, b"content")
    payload = cleanup.build_payload(tmp_path, bot_logs_root=tmp_path, max_tier=2,
                                    out_path=tmp_path / "latest.json", history_path=tmp_path / "history.jsonl")
    assert payload["selected_count"] == 0


def test_preview_never_verifies_or_claims_projected_capacity_as_ready(tmp_path, monkeypatch):
    _write_gzip_pair(tmp_path / "logs/rows_20260501.jsonl", b"content" * 1000)
    monkeypatch.setattr(cleanup, "_disk_snapshot", _fake_disk)
    def forbidden(*args):
        raise AssertionError("preview must not read full files")
    monkeypatch.setattr(cleanup.verified, "verify_pair", forbidden)
    payload = cleanup.build_payload(tmp_path, bot_logs_root=tmp_path, target_free_gb=0.000001,
                                    out_path=tmp_path / "latest.json", history_path=tmp_path / "history.jsonl")
    assert payload["selected_count"] == 1
    assert payload["ok"] is False
    assert payload["selected_candidates"][0]["verification_state"] == "full_verification_required_at_apply"


def test_nested_protected_directory_is_never_probed(tmp_path, monkeypatch):
    (tmp_path / "data").symlink_to("/Volumes/VIDEO/never-access", target_is_directory=True)
    original_stat = Path.stat
    def checked_stat(path, *args, **kwargs):
        if path == tmp_path / "data" or str(path).startswith(str(tmp_path / "data") + "/"):
            raise AssertionError("protected directory target must not be probed")
        return original_stat(path, *args, **kwargs)
    monkeypatch.setattr(Path, "stat", checked_stat)
    payload = cleanup.build_payload(tmp_path, bot_logs_root=tmp_path, max_tier=2,
                                    out_path=tmp_path / "latest.json", history_path=tmp_path / "history.jsonl")
    assert payload["selected_count"] == 0


def test_protected_default_environment_is_rejected_before_resolver_probe(tmp_path, monkeypatch):
    monkeypatch.setenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", "/Volumes/VIDEO/never-access")
    original = Path.exists
    def forbidden(path):
        if str(path).startswith("/Volumes/VIDEO"):
            raise AssertionError("must validate before existence probe")
        return original(path)
    monkeypatch.setattr(Path, "exists", forbidden)
    payload = cleanup.build_payload(tmp_path, out_path=tmp_path / "latest.json", history_path=tmp_path / "history.jsonl")
    assert payload["overall_status"] == "deferred"
    assert payload["reason"] == "protected_or_unavailable_external_route"


@pytest.mark.parametrize("padding", [b"\0" * (2 * 1024**2), gzip.compress(b"") * 10000])
def test_compressed_input_is_included_in_verification_budget(tmp_path, padding):
    raw = tmp_path / "logs/rows_20260501.jsonl"
    _write_gzip_pair(raw, b"a")
    archive = Path(str(raw) + ".gz")
    archive.write_bytes(archive.read_bytes() + padding)
    with pytest.raises(RuntimeError, match="verification_byte_budget"):
        cleanup.verified.remove_pair(tmp_path, raw, archive, cleanup.verified.Budget(max_bytes=2))
    assert raw.exists()


def test_compressed_reader_checks_deadline_inside_padding_reads(tmp_path):
    raw = tmp_path / "rows_20260501.jsonl"
    _write_gzip_pair(raw, b"a")
    archive = Path(str(raw) + ".gz")
    archive.write_bytes(archive.read_bytes() + b"\0" * (2 * 1024**2))
    class Budget(cleanup.verified.Budget):
        def consume(self, count):
            super().consume(count)
            if count > 1024:
                self.deadline = 0
    with pytest.raises(RuntimeError, match="verification_deadline"):
        cleanup.verified.verify_pair(raw, archive, Budget())
    assert raw.exists()


def test_post_unlink_receipt_failure_is_reported_as_removed_not_skipped(tmp_path, monkeypatch):
    root = tmp_path / "data_root"
    raw = root / "logs/rows_20260501.jsonl"
    _write_gzip_pair(raw, b"content" * 100)
    original = cleanup.verified.receipt
    def fail_after_release(root, record):
        if record["phase"] == "source_released":
            raise OSError("post-unlink persistence failure")
        original(root, record)
    monkeypatch.setattr(cleanup.verified, "receipt", fail_after_release)
    monkeypatch.setattr(cleanup, "_disk_snapshot", _fake_disk)
    payload = cleanup.build_payload(tmp_path, bot_logs_root=root, apply=True,
                                    out_path=tmp_path / "latest.json", history_path=tmp_path / "history.jsonl")
    assert not raw.exists()
    assert payload["apply_result"]["deleted_files"] == 1
    assert payload["apply_result"]["skipped_files"] == 0
    assert payload["apply_result"]["errors"][0]["source_removed"] is True
    assert payload["cleanup_pass_complete"] is False


@pytest.mark.parametrize("name", ["deep_cold_storage_layer_latest.json", "retention_intelligence_v2_latest.json"])
def test_auxiliary_protected_report_alias_is_not_read(tmp_path, monkeypatch, name):
    report = tmp_path / "governance/health" / name
    report.parent.mkdir(parents=True)
    report.symlink_to("/Volumes/VIDEO/never-access")
    original = cleanup.load_json
    def checked_read(path):
        assert path != report, "protected leaf must be rejected before reading"
        return original(path)
    monkeypatch.setattr(cleanup, "load_json", checked_read)
    payload = cleanup.build_payload(tmp_path, bot_logs_root=tmp_path,
                                    out_path=tmp_path / "latest.json", history_path=tmp_path / "history.jsonl")
    assert payload["assessment_complete"] is False
    assert payload["overall_status"] == "deferred"


@pytest.mark.parametrize("phase", ["scan", "final"])
def test_empty_assessment_cannot_outlive_shared_budget(tmp_path, monkeypatch, phase):
    budgets = []
    class Budget(cleanup.verified.Budget):
        def __init__(self, *args):
            super().__init__(*args)
            budgets.append(self)
    monkeypatch.setattr(cleanup.verified, "Budget", Budget)
    if phase == "scan":
        original = cleanup._scan_duplicate_jsonl_gzip
        def expire(*args, **kwargs):
            budgets[0].deadline = 0
            return original(*args, **kwargs)
        monkeypatch.setattr(cleanup, "_scan_duplicate_jsonl_gzip", expire)
    else:
        original = cleanup._summarize_candidates
        def expire(*args, **kwargs):
            budgets[0].deadline = 0
            return original(*args, **kwargs)
        monkeypatch.setattr(cleanup, "_summarize_candidates", expire)
    payload = cleanup.build_payload(tmp_path, bot_logs_root=tmp_path,
                                    out_path=tmp_path / "latest.json", history_path=tmp_path / "history.jsonl")
    assert payload["assessment_complete"] is False
    assert not payload.get("cleanup_pass_complete")
    assert payload["reason"] == "verification_deadline"
