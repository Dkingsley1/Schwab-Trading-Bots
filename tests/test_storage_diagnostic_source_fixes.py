import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import accountability, storage_router
from scripts import daily_state_snapshot_drill as drill
from scripts.ops import data_plane_recovery_controller as data_plane
from scripts.ops import state_snapshot_capacity as compact


@pytest.fixture(autouse=True)
def forbid_external_metadata(monkeypatch):
    original_stat, original_lstat = os.stat, os.lstat

    def guarded(original):
        def call(path, *args, **kwargs):
            if not isinstance(path, int):
                assert not os.fsdecode(path).casefold().startswith("/volumes"), path
            return original(path, *args, **kwargs)
        return call

    monkeypatch.setattr(os, "stat", guarded(original_stat))
    monkeypatch.setattr(os, "lstat", guarded(original_lstat))


@pytest.fixture
def restore_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(drill, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(compact, "reserve_bytes", lambda *args: (64 * 1024**3, "test"))
    monkeypatch.setattr(compact, "CopyGuard", lambda *args, **kwargs: SimpleNamespace(check=lambda: None))
    root = tmp_path / "archives"
    run = root / "20260915_120000_000001"
    run.mkdir(parents=True)
    archive = run / "state.gz"
    archive.write_bytes(b"retained archive")
    source = str(tmp_path / "state.json")
    payload = {
        "timestamp_utc": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
        "ok": True, "files_checked": 1, "files_restore_verified": 1,
        "latest_write_verified": True, "published_latest_write_verified": True,
        "rows": [{
            "requested_source": source, "snapshot": str(archive),
            "snapshot_sha256": "a" * 64, "restore_sha256": "a" * 64,
            "restore_verified": True, "error": "",
            "archive_proof": {
                "archive_path": str(archive), "archive_bytes": archive.stat().st_size,
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "decoded_sha256": "a" * 64,
                "verification": "full_decoded_bytes_match_verified_sqlite_or_file_restore",
            },
        }],
    }
    manifest = run / "manifest.json"
    manifest.write_text(json.dumps(payload))
    published = tmp_path / "published.json"
    published.write_text('{"preserved": true}')
    args = SimpleNamespace(targets=[source], operation_seconds=60)
    return args, root, manifest, archive, published, payload


@pytest.mark.parametrize("content", [None, "{", "[]", "{}", '{"ok":false}'])
def test_restore_skips_incomplete_candidate_without_renewing_producer_time(restore_receipt, content):
    args, root, _, _, published, payload = restore_receipt
    orphan = root / "20260915_130000_000001"
    orphan.mkdir()
    if content is not None:
        (orphan / "manifest.json").write_text(content)
    assert drill._recover_latest_verified(args, root, published) == 0
    result = json.loads(published.read_text())
    assert result["timestamp_utc"] == payload["timestamp_utc"]
    assert len(result["skipped_restore_candidates"]) == 1
    assert result["skipped_restore_candidates"][0]["path"] == str(orphan / "manifest.json")


def test_missing_archive_root_reports_exact_route(restore_receipt, tmp_path):
    args, _, _, _, published, _ = restore_receipt
    missing = tmp_path / "missing-archive"
    assert drill._recover_latest_verified(args, missing, published) == 2
    result = json.loads((tmp_path / "governance/health/state_snapshot_drill_attempt_latest.json").read_text())
    assert result["failure_context"] == {
        "stage": "archive_root", "path": str(missing),
        "route_status": "missing", "resolved_path": str(missing),
    }
    assert json.loads(published.read_text()) == {"preserved": True}


@pytest.mark.parametrize("fault", ["missing", "corrupt", "escape"])
def test_selected_archive_failure_is_fatal_even_with_older_valid_candidate(restore_receipt, tmp_path, fault):
    args, root, manifest, archive, published, payload = restore_receipt
    older = root / "20260915_110000_000001"
    older.mkdir()
    old_payload = json.loads(json.dumps(payload))
    old_payload["timestamp_utc"] = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    old_archive = older / "state.gz"
    old_archive.write_bytes(archive.read_bytes())
    old_payload["rows"][0]["snapshot"] = str(old_archive)
    old_payload["rows"][0]["archive_proof"]["archive_path"] = str(old_archive)
    (older / "manifest.json").write_text(json.dumps(old_payload))
    if fault == "missing":
        archive.unlink()
    elif fault == "corrupt":
        archive.write_bytes(b"x" * archive.stat().st_size)
    else:
        archive.unlink()
        archive.symlink_to("/Volumes/UNAVAILABLE/secret.gz")
    assert drill._recover_latest_verified(args, root, published) == 2
    result = json.loads((tmp_path / "governance/health/state_snapshot_drill_attempt_latest.json").read_text())
    assert result["failure_context"]["stage"] == "selected_archive"
    assert result["failure_context"]["path"] == str(archive)
    assert result["failure_context"]["route_status"] == {
        "missing": "missing", "corrupt": "present", "escape": "boundary_escape",
    }[fault]
    assert json.loads(published.read_text()) == {"preserved": True}


@pytest.mark.parametrize("target", ["/Volumes/VIDEO/forbidden", "/Volumes/UNAVAILABLE/archive", "../outside"])
def test_candidate_boundary_escape_is_fatal_without_target_probe(restore_receipt, target):
    args, root, _, _, published, _ = restore_receipt
    (root / "20260915_130000_000001").symlink_to(target)
    assert drill._recover_latest_verified(args, root, published) == 2
    assert json.loads(published.read_text()) == {"preserved": True}


@pytest.mark.parametrize("rel", ["decision_explanations", "governance", "data/snapshot_context.sqlite3"])
def test_local_fallback_redirect_rejected_before_any_mutation(tmp_path, monkeypatch, rel):
    root = tmp_path / "project"
    fallback = root / "local_fallback_storage"
    fallback.mkdir(parents=True)
    link = fallback / rel
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to("/Volumes/UNAVAILABLE/redirect")
    monkeypatch.setenv("BOT_LOGS_PREFER_EXTERNAL", "0")
    monkeypatch.setenv("BOT_LOGS_LOCAL_FALLBACK_ROOT", str(fallback))
    monkeypatch.setattr(storage_router, "maintenance_hold_snapshot", lambda root: {"active": False})
    def forbidden(*args, **kwargs):
        pytest.fail("mutation reached before fallback validation")
    monkeypatch.setattr(storage_router, "_is_writable_directory", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(Path, "unlink", forbidden)
    monkeypatch.setattr(Path, "symlink_to", forbidden)
    with pytest.raises(RuntimeError, match="local_fallback_route_rejected"):
        storage_router.route_runtime_storage(root)
    assert os.readlink(link) == "/Volumes/UNAVAILABLE/redirect"


def test_internal_fallback_alias_is_allowed(tmp_path):
    (tmp_path / "real").mkdir()
    (tmp_path / "alias").symlink_to("real")
    storage_router._require_local_fallback_path(tmp_path / "alias/new", tmp_path)
    assert not (tmp_path / "real/new").exists()


def test_atomic_writers_have_independent_same_directory_temps(tmp_path, monkeypatch):
    target = tmp_path / "latest.json"
    barrier = threading.Barrier(4)
    original = Path.replace
    temporary_paths = []
    def replace(path, destination):
        temporary_paths.append(path)
        assert path.parent == tmp_path
        if destination == target:
            barrier.wait(timeout=5)
        return original(path, destination)
    monkeypatch.setattr(Path, "replace", replace)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda n: accountability.safe_write_json_atomic(
            str(target), {"number": n}, marker=True), range(4)))
    assert all(results)
    assert len(set(temporary_paths)) == 8
    assert json.loads(target.read_text())["number"] in range(4)
    assert isinstance(json.loads((tmp_path / "latest.json.ok").read_text()), dict)
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_failure_preserves_existing_file_and_cleans_only_owned_temp(tmp_path, monkeypatch):
    target = tmp_path / "latest.json"
    target.write_text('{"old":true}')
    unrelated = tmp_path / "latest.json.tmp"
    unrelated.write_text("another writer")
    events = []
    monkeypatch.setattr(accountability, "_emit_write_failure_event", lambda **kwargs: events.append(kwargs))
    def fail(*args):
        raise OSError("publication failed")
    monkeypatch.setattr(Path, "replace", fail)
    assert not accountability.safe_write_json_atomic(str(target), {"new": True})
    assert json.loads(target.read_text()) == {"old": True}
    assert list(tmp_path.glob("*.tmp")) == [unrelated]
    assert len(events) == 1


def write_journal(tmp_path, rows):
    events = tmp_path / "governance/events"
    events.mkdir(parents=True, exist_ok=True)
    path = events / "write_failures_20260915.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def failure(**overrides):
    return {"event": "write_failure", "timestamp_utc": "2026-09-15T15:13:16.100000+00:00",
            "source": "writer", "run_id": "run", "iter_id": "run:1",
            "target_path": "/logical/failed.jsonl", "error": "channel_batch_append_failed", **overrides}


def test_journal_count_survives_display_rolloff_and_deduplicates_only_paired_errors(tmp_path):
    rows = [failure(), failure(error="batch_write_failed", timestamp_utc="2026-09-15T15:13:16.101000+00:00")]
    rows += [failure(iter_id=f"run:{n}") for n in range(2, 32)]
    rows += [failure(error="channel_mirror_append_failed")]
    write_journal(tmp_path, rows)
    result = data_plane._write_failure_history(tmp_path, {"recent_incidents": [{"summary": "info"}] * 20})
    assert result["complete"]
    assert result["count"] == 32
    assert result["raw_event_count"] == 33
    assert result["duplicate_count"] == 1


def test_repeated_failures_are_not_deduplicated_across_time_or_targets(tmp_path):
    write_journal(tmp_path, [failure(), failure(error="batch_write_failed", timestamp_utc="2026-09-15T15:13:18+00:00"),
                             failure(target_path="/another/path")])
    assert data_plane._write_failure_history(tmp_path, {})["count"] == 3


def test_journal_failures_are_not_cleared_by_healthy_sql_storage(tmp_path):
    write_journal(tmp_path, [failure()])
    health = tmp_path / "governance/health"
    health.mkdir()
    (health / "ingestion_storage_control_latest.json").write_text(json.dumps({
        "overall_status": "ready", "severity": "stable", "backpressure_quality_score": 100,
        "recovery_quality_score": 96, "steady_state": {"target_status": {"steady_state_ready": True}},
        "external_route_verification": {"verification_state": "active_local_ready"},
    }))
    result = data_plane.build_payload(tmp_path)
    assert result["raw_write_failure_count"] == result["write_failure_count"] == 1
    assert not result["write_path_recovered_by_storage"]
    assert not result["ok"]


@pytest.mark.parametrize("fault", ["symlink", "invalid", "budget", "deleted"])
def test_incomplete_history_cannot_clear_previous_failures(tmp_path, monkeypatch, fault):
    journal = write_journal(tmp_path, [failure()])
    health = tmp_path / "governance/health"
    health.mkdir()
    (health / "data_plane_recovery_controller_latest.json").write_text(json.dumps({"write_failure_history": {"count": 7}}))
    if fault == "symlink":
        journal.unlink()
        journal.symlink_to("/Volumes/UNAVAILABLE/events.jsonl")
    elif fault == "invalid":
        journal.write_text("{\n")
    elif fault == "budget":
        monkeypatch.setattr(data_plane, "WRITE_HISTORY_MAX_BYTES", 4)
    else:
        journal.unlink()
    result = data_plane.build_payload(tmp_path)
    assert result["raw_write_failure_count"] == result["write_failure_count"] == 7
    assert result["overall_status"] == "blocked"
    assert not result["write_failure_history"]["complete"]


def test_external_compressed_history_is_rejected_without_reading_it(tmp_path):
    journal = write_journal(tmp_path, [])
    archive = journal.with_suffix(".jsonl.gz")
    archive.symlink_to("/Volumes/UNAVAILABLE/history.gz")
    history = data_plane._write_failure_history(tmp_path, {})
    assert not history["complete"]
    assert history["errors"][0]["route_status"] == "external_path"


def test_event_directory_redirect_is_rejected_before_scanning(tmp_path):
    (tmp_path / "governance").mkdir()
    (tmp_path / "governance/events").symlink_to("/Volumes/UNAVAILABLE/events")
    history = data_plane._write_failure_history(tmp_path, {})
    assert not history["complete"]
    assert history["file_count"] == 0


def test_history_file_budget_is_not_reported_as_empty_healthy_history(tmp_path, monkeypatch):
    write_journal(tmp_path, [failure()])
    monkeypatch.setattr(data_plane, "WRITE_HISTORY_MAX_FILES", 0)
    result = data_plane.build_payload(tmp_path)
    assert result["overall_status"] == "blocked"
    assert not result["write_failure_history"]["complete"]


def test_empty_journals_cannot_erase_preupgrade_failure_count(tmp_path):
    write_journal(tmp_path, [])
    health = tmp_path / "governance/health"
    health.mkdir()
    (health / "data_plane_recovery_controller_latest.json").write_text('{"raw_write_failure_count":14}')
    result = data_plane.build_payload(tmp_path)
    assert result["write_failure_count"] == 14
    assert result["overall_status"] == "blocked"


def test_history_row_budget_counts_records_before_deduplication(tmp_path, monkeypatch):
    write_journal(tmp_path, [failure()] * 3)
    monkeypatch.setattr(data_plane, "WRITE_HISTORY_MAX_ROWS", 2)
    result = data_plane.build_payload(tmp_path)
    assert result["overall_status"] == "blocked"
    assert not result["write_failure_history"]["complete"]


def test_complete_journal_census_supersedes_preupgrade_display_sample(tmp_path):
    write_journal(tmp_path, [failure(iter_id=f"run:{n}") for n in range(13)])
    health = tmp_path / "governance/health"
    health.mkdir()
    (health / "data_plane_recovery_controller_latest.json").write_text('{"raw_write_failure_count":14}')
    result = data_plane._write_failure_history(tmp_path, {})
    assert result["count"] == 13
    assert result["complete"]


@pytest.mark.parametrize("bad_row", [{}, {"error": "write failure"}, {"event": "write_failure"},
                                   failure(source=""), failure(target_path=None),
                                   failure(timestamp_utc="bad"), failure(error={"detail": "bad"})])
def test_malformed_schema_preserves_preupgrade_debt(tmp_path, bad_row):
    write_journal(tmp_path, [failure(), bad_row])
    health = tmp_path / "governance/health"
    health.mkdir()
    (health / "data_plane_recovery_controller_latest.json").write_text('{"raw_write_failure_count":14}')
    result = data_plane.build_payload(tmp_path)
    assert result["write_failure_count"] == 14
    assert result["overall_status"] == "blocked"
    assert not result["write_failure_history"]["complete"]
    assert result["write_failure_history"]["errors"][0]["reason"] == "invalid_write_failure_schema"


def test_unavailable_first_journal_census_preserves_sampled_debt(tmp_path):
    health = tmp_path / "governance/health"
    health.mkdir(parents=True)
    (health / "data_plane_recovery_controller_latest.json").write_text('{"raw_write_failure_count":14}')
    result = data_plane.build_payload(tmp_path)
    assert result["write_failure_count"] == 14
    assert result["overall_status"] == "blocked"
    assert not result["write_failure_history"]["complete"]


def test_recent_compressed_families_are_read_without_scanning_old_archives(tmp_path):
    import gzip

    path = write_journal(tmp_path, [failure()])
    path.with_suffix(".jsonl.gz").write_bytes(gzip.compress(path.read_bytes()))
    earlier = path.with_name("write_failures_20260914.jsonl.gz")
    earlier.write_bytes(gzip.compress((json.dumps(failure(iter_id="earlier", timestamp_utc="2026-09-14T12:00:00Z")) + "\n").encode()))
    path.with_name("write_failures_20260913.jsonl.gz").write_bytes(b"invalid older archive")
    history = data_plane._write_failure_history(tmp_path, {})
    assert history["complete"]
    assert history["count"] == 2
    assert history["duplicate_count"] == 1
    assert history["scope_dates"] == ["20260915", "20260914"]
    assert history["file_count"] == 3


def test_corrupt_recent_gzip_remains_incomplete(tmp_path):
    path = write_journal(tmp_path, [failure()])
    path.with_suffix(".jsonl.gz").write_bytes(b"not gzip")
    history = data_plane._write_failure_history(tmp_path, {})
    assert not history["complete"]
    assert history["count"] == 1


def test_future_rows_cannot_count_as_current_write_failures(tmp_path):
    write_journal(tmp_path, [failure(), failure(iter_id="future", timestamp_utc="2999-01-01T00:00:00Z")])
    history = data_plane._write_failure_history(tmp_path, {})
    assert history["count"] == 1
    assert not history["complete"]
    assert history["errors"][0]["reason"] == "future_write_failure"


def test_complete_new_file_scope_does_not_inherit_all_time_failure_counts(tmp_path):
    write_journal(tmp_path, [failure()])
    health = tmp_path / "governance/health"
    health.mkdir()
    (health / "data_plane_recovery_controller_latest.json").write_text(json.dumps({
        "write_failure_history": {"count": 100, "scope_dates": ["20260901", "20260831"]},
    }))
    history = data_plane._write_failure_history(tmp_path, {})
    assert history["count"] == 1
    assert history["complete"]
