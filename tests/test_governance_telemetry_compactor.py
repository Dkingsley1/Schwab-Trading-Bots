from __future__ import annotations

import gzip
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import pytest

from scripts.ops import governance_telemetry_compactor as src


@pytest.fixture(autouse=True)
def idle_probe(monkeypatch):
    monkeypatch.setattr(src, "_require_rotated_idle", lambda path: None)


def _write_channel_file(project_root: Path, *, channel: str = "decision", profile: str = "default_crypto_schwab") -> Path:
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = project_root / "governance" / "channels" / channel / profile / f"{channel}_{day}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"timestamp_utc":"2026-05-22T00:00:00+00:00","symbol":"BTC-USD","action":"HOLD"}\n' * 8, encoding="utf-8")
    return path


def _write_master_control_file(project_root: Path) -> Path:
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = project_root / "governance" / "shadow_crypto" / f"master_control_{day}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"timestamp_utc":"2026-05-22T00:00:00+00:00","mode":"shadow_crypto"}\n' * 8, encoding="utf-8")
    return path


def _write_execution_lane_file(project_root: Path) -> Path:
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = project_root / "governance" / "execution_lanes" / f"execution_intents_{day}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"timestamp_utc":"2026-05-22T00:00:00+00:00","intent":"paper"}\n' * 8, encoding="utf-8")
    return path


def test_compactor_dry_run_selects_oversized_current_day_channel(tmp_path: Path) -> None:
    source = _write_channel_file(tmp_path)

    payload = src.build_payload(
        project_root=tmp_path,
        apply=False,
        channels=["decision"],
        min_file_mb=0.000001,
        target_free_gb=0,
        max_files=4,
        include_current_day=True,
    )

    assert payload["overall_status"] == "planned"
    assert payload["summary"]["candidate_count"] == 1
    assert payload["summary"]["selected_count"] == 1
    assert payload["records"][0]["relative_path"] == "governance/channels/decision/default_crypto_schwab/" + source.name
    assert source.exists()
    assert source.read_text(encoding="utf-8").count("\n") == 8


def test_compactor_default_discovers_risk_channel(tmp_path: Path) -> None:
    source = _write_channel_file(tmp_path, channel="risk")

    payload = src.build_payload(
        project_root=tmp_path,
        apply=False,
        min_file_mb=0.000001,
        target_free_gb=0,
        max_files=4,
        include_current_day=True,
    )

    assert payload["overall_status"] == "planned"
    assert payload["policy"]["requested_channels"] == ["all"]
    assert "risk" in payload["policy"]["channels"]
    assert payload["records"][0]["relative_path"] == "governance/channels/risk/default_crypto_schwab/" + source.name


def test_compactor_apply_rotates_to_stale_stage_and_keeps_fresh_path(tmp_path: Path) -> None:
    source = _write_channel_file(tmp_path)

    payload = src.build_payload(
        project_root=tmp_path,
        apply=True,
        channels=["decision"],
        min_file_mb=0.000001,
        target_free_gb=0,
        max_files=4,
        include_current_day=True,
        compression_level=1,
    )

    assert payload["overall_status"] == "applied"
    assert payload["summary"]["archived_count"] == 1
    assert payload["summary"]["raw_archived_bytes"] > 0
    assert source.exists()
    assert source.read_text(encoding="utf-8") == ""

    archive_rel = payload["records"][0]["archive_path"]
    archive_path = tmp_path / archive_rel
    assert archive_path.exists()
    assert "data/stale_stage/governance_telemetry_compactor/" in archive_rel
    with gzip.open(archive_path, "rt", encoding="utf-8") as handle:
        archived = handle.read()
    assert archived.count("\n") == 8
    assert "BTC-USD" in archived


def test_compactor_recovers_orphaned_pending_file_after_interrupted_run(tmp_path: Path) -> None:
    source = _write_channel_file(tmp_path, channel="risk")
    original = source.read_text(encoding="utf-8")
    pending = source.with_name(f"{source.name}.compact_pending_20260730T150733Z_55014")
    source.rename(pending)
    source.touch()

    payload = src.build_payload(
        project_root=tmp_path,
        apply=True,
        channels=["risk"],
        min_file_mb=0.000001,
        target_free_gb=0,
        max_files=4,
        include_current_day=True,
        compression_level=1,
    )

    assert payload["overall_status"] == "applied"
    assert payload["summary"]["orphaned_compaction_candidate_count"] == 1
    assert payload["summary"]["orphaned_compaction_recovered_count"] == 1
    assert not pending.exists()
    assert source.exists()
    assert source.read_text(encoding="utf-8") == ""
    record = payload["records"][0]
    assert record["orphaned_compaction_recovered"] is True
    assert ".compact_pending_" not in record["archive_path"]
    assert ".segment_" in record["archive_path"]
    with gzip.open(tmp_path / record["archive_path"], "rt", encoding="utf-8") as handle:
        assert handle.read() == original


def test_compactor_keeps_active_and_orphaned_segments_in_distinct_archives(tmp_path: Path) -> None:
    source = _write_channel_file(tmp_path, channel="risk")
    active_payload = source.read_text(encoding="utf-8")
    pending = source.with_name(f"{source.name}.compact_pending_20260730T150733Z_55014")
    pending_payload = json.dumps({"segment": "orphaned"}) + "\n"
    pending.write_text(pending_payload, encoding="utf-8")

    payload = src.build_payload(
        project_root=tmp_path,
        apply=True,
        channels=["risk"],
        min_file_mb=0.000001,
        target_free_gb=0,
        max_files=4,
        include_current_day=True,
        compression_level=1,
    )

    assert payload["summary"]["archived_count"] == 2
    archive_paths = [tmp_path / row["archive_path"] for row in payload["records"]]
    assert len({str(path) for path in archive_paths}) == 2
    archived_payloads = []
    for path in archive_paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            archived_payloads.append(handle.read())
    assert active_payload in archived_payloads
    assert pending_payload in archived_payloads


def test_compactor_can_skip_current_day_files(tmp_path: Path) -> None:
    _write_channel_file(tmp_path)

    payload = src.build_payload(
        project_root=tmp_path,
        apply=False,
        channels=["decision"],
        min_file_mb=0.000001,
        target_free_gb=0,
        max_files=4,
        include_current_day=False,
    )

    assert payload["overall_status"] == "nothing_to_do"
    assert payload["summary"]["candidate_count"] == 0


def test_compactor_covers_master_control_and_execution_lanes(tmp_path: Path) -> None:
    master_control = _write_master_control_file(tmp_path)
    execution_lane = _write_execution_lane_file(tmp_path)

    payload = src.build_payload(
        project_root=tmp_path,
        apply=False,
        families=["master_control", "execution_lanes"],
        min_file_mb=0.000001,
        target_free_gb=0,
        max_files=4,
        include_current_day=True,
    )

    rels = {row["relative_path"]: row["family"] for row in payload["records"]}
    assert payload["overall_status"] == "planned"
    assert rels[f"governance/shadow_crypto/{master_control.name}"] == "master_control"
    assert rels[f"governance/execution_lanes/{execution_lane.name}"] == "execution_lanes"


def test_compactor_applies_to_master_control_without_losing_active_path(tmp_path: Path) -> None:
    source = _write_master_control_file(tmp_path)

    payload = src.build_payload(
        project_root=tmp_path,
        apply=True,
        families=["master_control"],
        min_file_mb=0.000001,
        target_free_gb=0,
        max_files=4,
        include_current_day=True,
        compression_level=1,
    )

    assert payload["overall_status"] == "applied"
    assert payload["summary"]["archived_count"] == 1
    assert source.exists()
    assert source.read_text(encoding="utf-8") == ""
    archive_path = tmp_path / payload["records"][0]["archive_path"]
    assert archive_path.exists()


def test_rotation_emits_full_restore_proof(tmp_path):
    source = _write_master_control_file(tmp_path)
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    payload = src.build_payload(project_root=tmp_path, apply=True, min_file_mb=0.000001)
    proof = payload["records"][0]["verification"]
    assert proof["sha256_uncompressed"] == expected
    assert proof["raw_removed"] is True
    assert (
        proof["verification_basis"]
        == "full_gzip_restore_sha256_and_stable_source_identity"
    )


@pytest.mark.parametrize("failure", ["verification", "open_file"])
def test_rotation_failure_preserves_pending_contents(tmp_path, monkeypatch, failure):
    source = _write_master_control_file(tmp_path)
    original = source.read_bytes()
    if failure == "verification":
        monkeypatch.setattr(
            src,
            "_compress_and_clear",
            lambda *a, **kw: {"status": "failed", "reason": "restore_mismatch"},
        )
    else:

        def busy(path):
            raise RuntimeError("rotated_file_still_open")

        monkeypatch.setattr(src, "_require_rotated_idle", busy)
    payload = src.build_payload(project_root=tmp_path, apply=True, min_file_mb=0.000001)
    assert payload["overall_status"] == "degraded"
    assert payload["summary"]["archived_count"] == 0
    assert source.read_bytes() == b""
    pending = list(source.parent.glob("*.compact_pending_*"))
    assert len(pending) == 1 and pending[0].read_bytes() == original


def test_protected_alias_is_rejected_before_metadata(tmp_path, monkeypatch):
    root = tmp_path / "governance" / "shadow_alias"
    root.parent.mkdir()
    root.symlink_to("/Volumes/VIDEO")
    original = Path.stat

    def guarded(path, *args, **kwargs):
        assert not str(path).startswith("/Volumes/VIDEO")
        assert not (path == root and kwargs.get("follow_symlinks", True))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", guarded)
    assert src._iter_master_control_files(tmp_path) == []
    record = src._archive_one(
        project_root=tmp_path,
        source_rel="governance/shadow_alias/example.jsonl",
        archive_root=tmp_path / "archive",
        compression_level=1,
        stamp="test",
    )
    assert record["status"] == "error"


def test_idle_probe_rejects_open_files_and_probe_errors(tmp_path, monkeypatch):
    monkeypatch.undo()
    monkeypatch.setattr(src.shutil, "which", lambda name: "/usr/sbin/lsof")
    for result in (
        {"rc": 0, "stdout": "123", "stderr": ""},
        {"rc": 1, "stdout": "", "stderr": "error"},
    ):
        monkeypatch.setattr(src, "run_bounded_process_group", lambda *a, **kw: result)
        with pytest.raises(RuntimeError, match="idle_probe_failed"):
            src._require_rotated_idle(tmp_path / "rotated.jsonl")
    monkeypatch.setattr(
        src,
        "run_bounded_process_group",
        lambda *a, **kw: {"rc": 1, "stdout": "", "stderr": ""},
    )
    src._require_rotated_idle(tmp_path / "rotated.jsonl")
