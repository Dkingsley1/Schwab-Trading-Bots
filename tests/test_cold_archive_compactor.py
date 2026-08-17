from __future__ import annotations

import gzip
import json
import shutil
import sqlite3
from pathlib import Path

import scripts.ops.cold_archive_compactor as compactor
from scripts.ops.cold_archive_compactor import archive_root_available, build_payload, writer_blocks_compaction


def test_compactor_defers_heavy_work_while_single_writer_is_active() -> None:
    assert writer_blocks_compaction({"active": True}) is True
    assert writer_blocks_compaction({"active": True}, allow_active_writer=True) is False
    assert writer_blocks_compaction({"active": False}) is False


def test_compactor_requires_existing_archive_root_before_unattended_apply(tmp_path: Path) -> None:
    assert archive_root_available(tmp_path / "missing") is False
    assert archive_root_available(tmp_path) is True


def test_compactor_rejects_protected_archive_root_before_scanning() -> None:
    payload = build_payload(
        archive_root=Path("/Volumes/VIDEO/schwab_trading_bot_cold"),
        apply=True,
    )

    assert payload["ok"] is False
    assert payload["overall_status"] == "blocked_protected_volume"
    assert payload["blockers"] == ["protected_archive_volume_rejected"]


def test_stable_file_work_preflight_ignores_quarantine_and_finds_pending(tmp_path: Path) -> None:
    root = tmp_path / "cold"
    pending = root / "evidence.jsonl.gz.tmp"
    pending.parent.mkdir(parents=True)
    pending.write_bytes(b"pending")
    manifest = root / "cold_archive_compaction_manifest.jsonl"
    manifest.write_text('{"status":"compacted_verified"}\n', encoding="utf-8")
    quarantined = (
        root
        / "quarantine"
        / "corrupt_gzip_orphans"
        / "evidence.jsonl.gz.tmp.abc.corrupt-gzip-fragment"
    )
    quarantined.parent.mkdir(parents=True)
    quarantined.write_bytes(b"preserved")

    candidates = compactor.stable_file_work_candidates(
        root,
        min_age_hours=0,
        include_plain_jsonl=True,
        excluded_paths={manifest},
    )

    assert candidates == [pending]


def test_hourly_retention_lane_runs_bounded_writer_aware_cold_compaction() -> None:
    runner = (compactor.PROJECT_ROOT / "scripts" / "ops" / "run_data_retention_launchd.sh").read_text(
        encoding="utf-8"
    )

    assert "cold_archive_compactor.py" in runner
    assert "--coordinate-writer-handoff" in runner
    assert "BOT_COLD_ARCHIVE_MIN_AGE_HOURS" in runner
    assert "BOT_COLD_ARCHIVE_MAX_FILES" in runner


def test_writer_handoff_wait_returns_when_writer_releases(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(compactor, "writer_state_snapshot", lambda _root: {"active": False})

    result = compactor.wait_for_writer_handoff(tmp_path, timeout_seconds=0, poll_seconds=0.1)

    assert result["ready"] is True
    assert result["status"] == "writer_handoff_complete"


def test_writer_handoff_wait_times_out_without_bypassing_writer(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(compactor, "writer_state_snapshot", lambda _root: {"active": True})

    result = compactor.wait_for_writer_handoff(tmp_path, timeout_seconds=0, poll_seconds=0.1)

    assert result["ready"] is False
    assert result["status"] == "writer_handoff_timeout"


def test_compactor_losslessly_compresses_jsonl_and_writes_reader_index(tmp_path: Path) -> None:
    root = tmp_path / "cold"
    source = root / "data" / "deep_cold" / "manifest_backed" / "decision_explanations" / "paper" / "decisions_20260701.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text('{"id":1}\n{"id":2}\n', encoding="utf-8")

    payload = build_payload(
        archive_root=root,
        apply=True,
        min_age_hours=0,
        max_files=4,
        max_raw_gb=1,
        sqlite_inventory_limit=0,
    )

    target = source.with_suffix(".jsonl.gz")
    assert payload["ok"] is True
    assert source.exists() is False
    assert target.exists() is True
    with gzip.open(target, "rt", encoding="utf-8") as handle:
        assert handle.read() == '{"id":1}\n{"id":2}\n'
    assert (root / "COLD_ARCHIVE_README.txt").exists()
    assert "immutable=1" in (root / "COLD_ARCHIVE_README.txt").read_text(encoding="utf-8")
    manifest_rows = [json.loads(line) for line in (root / "cold_archive_compaction_manifest.jsonl").read_text().splitlines()]
    assert manifest_rows[0]["status"] == "compacted_verified"
    assert manifest_rows[0]["line_count"] == 2
    assert manifest_rows[0]["sha256_uncompressed"]
    families = {row["data_family"]: row for row in payload["archive_inventory"]["data_families"]}
    assert families["decision_explanations"]["file_count"] == 1


def test_compactor_inventory_reports_formats_families_and_tiers(tmp_path: Path) -> None:
    root = tmp_path / "cold"
    evidence = (
        root
        / "data"
        / "deep_cold"
        / "manifest_backed"
        / "governance_channels"
        / "governance"
        / "channels"
        / "risk"
        / "paper"
        / "risk_20260701.jsonl.gz"
    )
    evidence.parent.mkdir(parents=True)
    with gzip.open(evidence, "wt", encoding="utf-8") as handle:
        handle.write('{"risk":"bounded"}\n')

    trade_decisions = (
        root
        / "deep_cold"
        / "stale_stage"
        / "project"
        / "data"
        / "stale_stage"
        / "decisions"
        / "project"
        / "decisions"
        / "paper"
        / "trade_decisions_20260701.jsonl.gz"
    )
    trade_decisions.parent.mkdir(parents=True)
    with gzip.open(trade_decisions, "wt", encoding="utf-8") as handle:
        handle.write('{"decision":"hold"}\n')

    database = root / "data" / "sql_hot_archive" / "jsonl_link_runtime" / "archive.sqlite3"
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE jsonl_records (payload_json TEXT)")

    payload = build_payload(
        archive_root=root,
        apply=True,
        min_age_hours=0,
        max_files=0,
        include_plain_jsonl=False,
    )

    inventory = payload["archive_inventory"]
    formats = {row["format"]: row for row in inventory["formats"]}
    families = {row["data_family"]: row for row in inventory["data_families"]}
    tiers = {row["tier"]: row for row in inventory["tiers"]}
    assert formats["jsonl_gzip"]["file_count"] == 2
    assert formats["sqlite_database"]["file_count"] == 1
    assert families["risk_governance_events"]["file_count"] == 1
    assert families["trade_decisions"]["file_count"] == 1
    assert families["sql_link_runtime"]["file_count"] == 1
    assert tiers["manifest_backed_evidence"]["file_count"] == 1
    assert tiers["sql_hot_archive"]["file_count"] == 1
    assert tiers["stale_stage_and_quarantine"]["file_count"] == 1
    assert "AppleDouble" in " ".join(inventory["excluded_from_inventory"])
    assert "immutable=1" in payload["reader_commands"]["sqlite"]


def test_compactor_recovers_orphaned_pending_without_empty_placeholder(tmp_path: Path) -> None:
    root = tmp_path / "cold"
    pending = root / "governance" / "risk.jsonl.compact_pending_20260701T010101Z_42"
    pending.parent.mkdir(parents=True)
    pending.write_text('{"risk":"ok"}\n', encoding="utf-8")

    payload = build_payload(
        archive_root=root,
        apply=True,
        min_age_hours=0,
        sqlite_inventory_limit=0,
    )

    target = pending.with_name("risk.jsonl.gz")
    assert payload["summary"]["successful_action_count"] == 1
    assert pending.exists() is False
    assert pending.with_name("risk.jsonl").exists() is False
    with gzip.open(target, "rt", encoding="utf-8") as handle:
        assert json.loads(handle.readline()) == {"risk": "ok"}


def test_compactor_removes_only_exact_tmp_duplicate(tmp_path: Path) -> None:
    root = tmp_path / "cold"
    final = root / "archive.jsonl.gz"
    final.parent.mkdir(parents=True)
    with gzip.open(final, "wt", encoding="utf-8") as handle:
        handle.write('{"id":1}\n')
    duplicate = final.with_name(f".{final.name}.tmp.tmp")
    shutil.copy2(final, duplicate)

    payload = build_payload(
        archive_root=root,
        apply=True,
        min_age_hours=0,
        max_files=0,
        include_plain_jsonl=False,
        sqlite_inventory_limit=0,
    )

    assert payload["ok"] is True
    assert final.exists() is True
    assert duplicate.exists() is False
    assert payload["summary"]["released_gb"] >= 0


def test_compactor_recovers_verified_gzip_finalize_orphan(tmp_path: Path) -> None:
    root = tmp_path / "cold"
    pending = root / "governance" / "risk_20260701.jsonl.gz.tmp"
    pending.parent.mkdir(parents=True)
    with gzip.open(pending, "wt", encoding="utf-8") as handle:
        handle.write('{"risk":"bounded"}\n')

    payload = build_payload(
        archive_root=root,
        apply=True,
        min_age_hours=0,
        max_files=0,
        include_plain_jsonl=False,
        sqlite_inventory_limit=0,
    )

    target = pending.with_name("risk_20260701.jsonl.gz")
    assert pending.exists() is False
    assert target.exists() is True
    with gzip.open(target, "rt", encoding="utf-8") as handle:
        assert json.loads(handle.readline()) == {"risk": "bounded"}
    recovered = [row for row in payload["actions"] if row["status"] == "recovered_verified_orphan"]
    assert len(recovered) == 1
    assert recovered[0]["line_count"] == 1
    assert recovered[0]["sha256_uncompressed"]


def test_compactor_quarantines_invalid_gzip_finalize_orphan_with_provenance(tmp_path: Path) -> None:
    root = tmp_path / "cold"
    pending = root / "governance" / "risk_20260701.jsonl.gz.tmp"
    pending.parent.mkdir(parents=True)
    pending.write_bytes(b"not-a-gzip")

    payload = build_payload(
        archive_root=root,
        apply=True,
        min_age_hours=0,
        max_files=0,
        include_plain_jsonl=False,
        sqlite_inventory_limit=0,
    )

    assert payload["ok"] is True
    assert pending.exists() is False
    assert pending.with_name("risk_20260701.jsonl.gz").exists() is False
    assert payload["summary"]["error_count"] == 0
    assert payload["summary"]["quarantined_corrupt_orphan_count"] == 1
    quarantined = [row for row in payload["actions"] if row["status"] == "quarantined_corrupt_orphan"]
    assert len(quarantined) == 1
    quarantine_path = Path(quarantined[0]["quarantine_path"])
    metadata_path = Path(quarantined[0]["metadata_path"])
    assert quarantine_path.read_bytes() == b"not-a-gzip"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["source_relative_path"] == "governance/risk_20260701.jsonl.gz.tmp"
    assert metadata["sha256_compressed_fragment"] == quarantined[0]["sha256_compressed_fragment"]

    followup = build_payload(
        archive_root=root,
        apply=False,
        min_age_hours=0,
        max_files=0,
        include_plain_jsonl=False,
        sqlite_inventory_limit=0,
    )
    assert followup["summary"]["gzip_finalize_candidate_count"] == 0


def test_compactor_vacuums_only_integrity_checked_reclaimable_sqlite(tmp_path: Path) -> None:
    root = tmp_path / "cold"
    db = root / "archive.sqlite3"
    root.mkdir(parents=True)
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE evidence (payload TEXT)")
        conn.executemany("INSERT INTO evidence VALUES (?)", [("x" * 4000,) for _ in range(800)])
        conn.execute("DELETE FROM evidence WHERE rowid <= 700")
    before = db.stat().st_size

    payload = build_payload(
        archive_root=root,
        apply=True,
        min_age_hours=0,
        max_files=0,
        include_plain_jsonl=False,
        vacuum_sqlite=True,
        sqlite_min_reclaim_mb=0,
        sqlite_min_reclaim_ratio=0.01,
    )

    assert payload["ok"] is True
    assert payload["summary"]["sqlite_vacuum_eligible_count"] == 1
    assert db.stat().st_size < before
    with sqlite3.connect(db) as conn:
        assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
