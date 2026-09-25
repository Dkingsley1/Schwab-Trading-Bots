import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import data_retention_policy as retention


def staged(tmp_path, name="logs/old.log", **overrides):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("closed data")
    os.utime(path, (1_735_689_600, 1_735_689_600))
    row = {
        "event": "staged",
        "staged_path": str(path),
        "sha256": retention._path_sha256(path),
        "integrity_verified": True,
        "protected_evidence": False,
        "economic_value": "low",
        **overrides,
    }
    return path, row


def manifest(tmp_path, rows):
    path = tmp_path / "stale_manifest.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def purge(tmp_path, receipt, **kwargs):
    return retention._purge_old_stale_stage(
        stale_root=tmp_path, manifest_path=receipt, older_than_days=1, **kwargs
    )


def test_compaction_preserves_reused_path_event_order(tmp_path):
    path, row = staged(tmp_path)
    receipt = manifest(
        tmp_path, [row, {"event": "purged", "staged_path": str(path)}, row]
    )
    before = retention._active_stale_manifest_rows(receipt)
    retention._compact_stale_manifest(manifest_path=receipt)
    assert retention._active_stale_manifest_rows(receipt) == before


def test_compaction_retains_unavailable_path_without_probing_it(tmp_path, monkeypatch):
    missing = "/Volumes/VIDEO/do-not-inspect"
    row = {"event": "staged", "staged_path": missing}
    receipt = manifest(tmp_path, [row])
    original = Path.exists

    def exists(path):
        assert str(path) != missing
        return original(path)

    monkeypatch.setattr(Path, "exists", exists)
    retention._compact_stale_manifest(manifest_path=receipt)
    assert retention._active_stale_manifest_rows(receipt) == {missing: row}


def test_corrupt_manifest_is_preserved_and_blocks_purge(tmp_path):
    path, row = staged(tmp_path)
    receipt = manifest(tmp_path, [row])
    with receipt.open("a") as handle:
        handle.write('{"event":"purged",')
    before = receipt.read_bytes()
    with pytest.raises(ValueError, match="manifest"):
        purge(tmp_path, receipt)
    assert path.exists()
    with pytest.raises(ValueError, match="manifest"):
        retention._compact_stale_manifest(manifest_path=receipt)
    assert receipt.read_bytes() == before


@pytest.mark.parametrize(
    "name", ["governance_health/risk_latest.json", "decisions/old.jsonl"]
)
def test_current_protection_overrides_legacy_false_flag(tmp_path, name):
    path, row = staged(tmp_path, name)
    result = purge(tmp_path, manifest(tmp_path, [row]))
    assert path.exists()
    assert result["skipped_protected_evidence_files"] == 1


def test_verification_reads_only_budgeted_candidates(tmp_path, monkeypatch):
    first, row_a = staged(tmp_path, "logs/a.log")
    _, row_b = staged(tmp_path, "logs/b.log")
    receipt = manifest(tmp_path, [row_a, row_b])
    hashed = []
    original = retention._path_sha256

    def hash_file(path):
        hashed.append(path)
        return original(path)

    monkeypatch.setattr(retention, "_path_sha256", hash_file)
    result = purge(tmp_path, receipt, max_files=1)
    assert hashed == [first]
    assert result["deleted_files"] == 1
    assert result["skipped_by_budget_files"] == 1


def test_identity_change_after_hash_preserves_source(tmp_path, monkeypatch):
    path, row = staged(tmp_path)
    receipt = manifest(tmp_path, [row])
    original = retention._path_sha256

    def hash_then_change(source):
        digest = original(source)
        source.write_text("new active content")
        return digest

    monkeypatch.setattr(retention, "_path_sha256", hash_then_change)
    result = purge(tmp_path, receipt)
    assert path.read_text() == "new active content"
    assert result["deleted_files"] == 0
    assert result["delete_errors"] == 1


def test_hardlink_not_deleted_or_counted_as_reclaimed(tmp_path):
    path, row = staged(tmp_path)
    os.link(path, tmp_path / "retained-copy")
    result = purge(tmp_path, manifest(tmp_path, [row]))
    assert path.exists()
    assert result["deleted_bytes"] == 0


def test_symlink_root_rejected_before_scanning(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    path, row = staged(target)
    receipt = manifest(target, [row])
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(RuntimeError, match="route"):
        purge(alias, receipt)
    assert path.exists()


def test_symlink_payload_not_reindexed(tmp_path):
    path = tmp_path / "logs" / "link.log"
    path.parent.mkdir()
    target = tmp_path / "outside-stage"
    target.write_text("not a staged payload")
    path.symlink_to(target)
    result = retention._reindex_legacy_stale_stage(
        stale_root=tmp_path, manifest_path=tmp_path / "stale_manifest.jsonl"
    )
    rows = retention._active_stale_manifest_rows(tmp_path / "stale_manifest.jsonl")
    assert str(path) not in rows
    assert path.is_symlink()


def test_manifest_truthy_string_does_not_authorize_deletion(tmp_path):
    path, row = staged(tmp_path, integrity_verified="false")
    result = purge(tmp_path, manifest(tmp_path, [row]))
    assert path.exists()
    assert result["skipped_unverified_manifest_files"] == 1


@pytest.mark.parametrize("name", ["risk.json.local_fallback", "audit/events.jsonl"])
def test_fallback_suffix_and_generic_name_do_not_bypass_evidence_protection(
    tmp_path, name
):
    path, row = staged(tmp_path, "logs/" + name)
    result = purge(tmp_path, manifest(tmp_path, [row]))
    assert path.exists()
    assert result["skipped_protected_evidence_files"] == 1


@pytest.mark.parametrize("value", [0, -1, 1e-12, float("nan"), float("inf")])
def test_reaper_rejects_unbounded_budget_before_work(tmp_path, monkeypatch, value):
    from scripts.ops import stale_artifact_reaper_bot as reaper

    monkeypatch.setattr(
        reaper, "_build_payload", lambda *a, **k: pytest.fail("must not run")
    )
    result = reaper.build_payload(
        tmp_path, stale_stage_root=tmp_path, max_delete_gb=value
    )
    assert result["ok"] is False
    assert "invalid_bounded_retention_budget" in result["error"]


def test_reaper_reports_corrupt_manifest_without_claiming_success(tmp_path):
    from scripts.ops import stale_artifact_reaper_bot as reaper

    (tmp_path / "stale_manifest.jsonl").write_text("broken json\n")
    result = reaper.build_payload(
        tmp_path, stale_stage_root=tmp_path, stale_stage_manifest="", stale_purge_days=1
    )
    assert result["ok"] is False
    assert result["reason"] == "stale_retention_failed_closed"
    assert result["summary"]["work_totals_complete"] is False


def test_busy_reaper_preserves_active_owner_receipt(tmp_path, monkeypatch):
    from scripts.ops import stale_artifact_reaper_bot as reaper

    output = tmp_path / "health.json"
    output.write_text('{"owner":"active","ok":false}')
    before = output.read_bytes()

    def busy(*args):
        raise BlockingIOError("already owned")

    monkeypatch.setattr(reaper.fcntl, "flock", busy)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reaper",
            "--project-root",
            str(tmp_path),
            "--out-file",
            str(output),
            "--lock-file",
            str(tmp_path / "owner.lock"),
            "--no-include-external-stale-root",
        ],
    )
    assert reaper.main() == 0
    assert output.read_bytes() == before


@pytest.mark.parametrize("hold", ["nan", "inf", "not-a-number"])
def test_invalid_legacy_hold_cannot_expire_file(tmp_path, hold):
    path, row = staged(
        tmp_path,
        legacy_reindexed=True,
        timestamp_utc="2025-01-01T00:00:00+00:00",
        legacy_reindex_hold_hours=hold,
    )
    result = purge(tmp_path, manifest(tmp_path, [row]))
    assert path.exists()
    assert result["skipped_legacy_reindex_hold_files"] == 1


@pytest.mark.parametrize(
    "name",
    ["backlog_quarantine_manifest.jsonl", "jsonl_link_trading.sqlite3.local_fallback"],
)
def test_custody_manifest_and_trading_fallback_are_not_disposable(tmp_path, name):
    path, row = staged(tmp_path, "logs/" + name)
    result = purge(tmp_path, manifest(tmp_path, [row]))
    assert path.exists()
    assert result["skipped_protected_evidence_files"] == 1


def test_owned_local_alias_preserves_logical_manifest_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(retention, "PROJECT_ROOT", tmp_path)
    physical = tmp_path / "local_fallback_storage/data/stale_stage"
    physical.mkdir(parents=True)
    logical = tmp_path / "data/stale_stage"
    logical.parent.mkdir()
    logical.symlink_to(physical, target_is_directory=True)
    path, row = staged(logical)
    result = purge(logical, manifest(logical, [row]))
    assert result["deleted_files"] == 1
    assert not path.exists()


def test_owned_alias_name_cannot_point_to_arbitrary_destination(tmp_path, monkeypatch):
    monkeypatch.setattr(retention, "PROJECT_ROOT", tmp_path)
    arbitrary = tmp_path / "other"
    arbitrary.mkdir()
    logical = tmp_path / "data/stale_stage"
    logical.parent.mkdir()
    logical.symlink_to(arbitrary, target_is_directory=True)
    with pytest.raises(RuntimeError, match="route"):
        retention._stale_route(logical)


def test_offloaded_manifest_restored_locally_without_changing_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(retention, "PROJECT_ROOT", tmp_path)
    parent = tmp_path / "local_fallback_storage/data/stale_stage"
    parent.mkdir(parents=True)
    source = tmp_path / "cold_archive/deep_cold/stale_stage/project/data/stale_stage/stale_manifest.saved.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text('{"event":"purged","staged_path":"/old"}\n')
    original = source.read_bytes()
    target = parent / "stale_manifest.jsonl"
    target.symlink_to(source)
    result = retention._restore_offloaded_stale_manifest(tmp_path, target)
    assert result["restored"] is True
    assert not target.is_symlink()
    assert target.read_bytes() == original == source.read_bytes()
    assert retention._restore_offloaded_stale_manifest(tmp_path, target)["restored"] is False


def test_offloaded_manifest_unknown_link_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(retention, "PROJECT_ROOT", tmp_path)
    parent = tmp_path / "local_fallback_storage/data/stale_stage"
    parent.mkdir(parents=True)
    source = tmp_path / "unowned.jsonl"
    source.write_text("{}")
    target = parent / "stale_manifest.jsonl"
    target.symlink_to(source)
    with pytest.raises(RuntimeError, match="unrecognized"):
        retention._restore_offloaded_stale_manifest(tmp_path, target)
    assert target.is_symlink()


def test_failed_manifest_restore_proof_keeps_link_and_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(retention, "PROJECT_ROOT", tmp_path)
    parent = tmp_path / "local_fallback_storage/data/stale_stage"
    parent.mkdir(parents=True)
    source = tmp_path / "cold_archive/deep_cold/stale_stage/project/data/stale_stage/stale_manifest.saved.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text("original archived manifest")
    target = parent / "stale_manifest.jsonl"
    target.symlink_to(source)
    monkeypatch.setattr(retention, "_path_sha256", lambda path: "bad proof")
    with pytest.raises(RuntimeError, match="changed_during_restore"):
        retention._restore_offloaded_stale_manifest(tmp_path, target)
    assert target.is_symlink()
    assert source.read_text() == "original archived manifest"
    assert not list(parent.glob(".stale_manifest.restore.*"))


def test_deep_cold_excludes_mutable_retention_controls(tmp_path):
    from scripts.ops import deep_cold_storage_layer as cold
    names = ["stale_manifest.jsonl", "backlog_quarantine_manifest.jsonl", "job_state.json", "status_latest.json", "owner.lock", "closed.jsonl"]
    for name in names:
        (tmp_path / name).write_text("closed data")
    assert cold._iter_candidate_files(tmp_path, min_size_bytes=1) == [tmp_path / "closed.jsonl"]
