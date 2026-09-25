import hashlib
import json
import shutil
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import daily_state_snapshot_drill as drill
from scripts.ops import state_snapshot_capacity as compact


class Guard:
    def __init__(self, *args, **kwargs):
        self.pace = SimpleNamespace(snapshot=lambda: {})

    def check(self):
        pass


@pytest.fixture
def retained(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    out = root / "drills"
    run = out / "20260911_124327_915974"
    snapshot = run / "snapshot/db.sqlite3"
    restored = run / "restore_probe/db.sqlite3"
    snapshot.parent.mkdir(parents=True)
    restored.parent.mkdir()
    with sqlite3.connect(snapshot) as db:
        db.execute("create table evidence (id integer)")
        db.execute("insert into evidence values (123)")
    shutil.copyfile(snapshot, restored)
    digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    partial = snapshot.with_suffix(".sqlite3.gz")
    partial.write_bytes(b"preserved partial archive")
    payload = {
        "timestamp_utc": "2026-09-11T12:52:04+00:00",
        "run_dir": str(run),
        "ok": False,
        "files_checked": 1,
        "files_restore_verified": 0,
        "full_platform_restore_verified": False,
        "missing_files": [],
        "rows": [
            {
                "snapshot": str(snapshot),
                "restored": str(restored),
                "copy_mode": "compact_sqlite_logical_snapshot_restore",
                "snapshot_sha256": digest,
                "restore_sha256": digest,
                "sqlite_integrity_verified": True,
                "restore_verified": False,
                "capacity_plan": {"archive_limit_bytes": 2 * compact.GIB},
                "error": "compressed_snapshot_archive_budget_exceeded",
            }
        ],
    }
    (run / "manifest.json").write_text(json.dumps(payload))
    monkeypatch.setattr(drill, "PROJECT_ROOT", root)
    monkeypatch.setattr(compact, "CopyGuard", Guard)
    monkeypatch.setattr(compact, "reserve_bytes", lambda *a: (64 * compact.GIB, "test"))
    monkeypatch.setattr(
        shutil, "disk_usage", lambda *a: SimpleNamespace(free=100 * compact.GIB)
    )
    args = SimpleNamespace(resume_run=str(run), operation_seconds=60)
    return args, out, run, snapshot, restored, partial, payload


def test_resume_reverifies_and_seals_without_recopy_or_deleting_partial(retained):
    args, out, run, snapshot, restored, partial, original = retained
    published = out / "published.json"
    assert drill._resume_archive(args, out, publish_latest=published) == 0
    payload = json.loads(published.read_text())
    assert compact.complete_restore_evidence(payload)
    assert payload["timestamp_utc"] == original["timestamp_utc"]
    assert payload["full_platform_restore_verified"] is False
    assert not snapshot.exists() and not restored.exists()
    assert partial.read_bytes() == b"preserved partial archive"
    assert len(payload["resume_from_manifest_sha256"]) == 64
    assert len(list(run.glob("manifest.before-resume-*.json"))) == 1


def test_resume_changed_copy_preserves_both_and_old_manifest(retained):
    args, out, run, snapshot, restored, partial, original = retained
    restored.write_bytes(b"changed")
    assert drill._resume_archive(args, out, publish_latest=out / "published.json") == 2
    assert snapshot.exists() and restored.exists() and partial.exists()
    assert json.loads((run / "manifest.json").read_text()) == original
    assert not (out / "published.json").exists()


def test_resume_requires_reserved_archive_capacity(retained, monkeypatch):
    args, out, run, snapshot, restored, partial, original = retained
    monkeypatch.setattr(
        shutil, "disk_usage", lambda *a: SimpleNamespace(free=67 * compact.GIB)
    )
    assert drill._resume_archive(args, out, publish_latest=out / "published.json") == 2
    assert snapshot.exists() and restored.exists()
    assert not list(snapshot.parent.glob("*.resume-*.gz"))


def test_resume_rejects_file_outside_owned_run(retained):
    args, out, run, snapshot, restored, partial, original = retained
    original["rows"][0]["snapshot"] = str(out / "unowned.sqlite3")
    shutil.copyfile(snapshot, out / "unowned.sqlite3")
    (run / "manifest.json").write_text(json.dumps(original))
    assert drill._resume_archive(args, out, publish_latest=out / "published.json") == 2
    assert (out / "unowned.sqlite3").exists()


def test_resume_revalidates_previously_successful_archive_before_releasing_copies(
    retained,
):
    args, out, run, snapshot, restored, partial, original = retained
    other = run / "snapshot/other.json"
    probe = run / "restore_probe/other.json"
    other.write_bytes(b'{"known":"evidence"}')
    shutil.copyfile(other, probe)
    digest = hashlib.sha256(other.read_bytes()).hexdigest()
    proof = compact.seal_compressed_snapshot(other, probe, digest, Guard(), 1024**2)
    original["rows"].append(
        {
            "snapshot": proof["archive_path"],
            "restore_verified": True,
            "archive_proof": proof,
            "snapshot_sha256": digest,
            "restore_sha256": digest,
        }
    )
    original["files_checked"] = 2
    (run / "manifest.json").write_text(json.dumps(original))
    Path(proof["archive_path"]).write_bytes(b"corrupt retained archive")
    assert drill._resume_archive(args, out, publish_latest=out / "published.json") == 2
    assert snapshot.exists() and restored.exists()
    assert not list(snapshot.parent.glob("*.resume-*.gz"))
