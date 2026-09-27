from datetime import datetime, timedelta, timezone
import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts import daily_state_snapshot_drill as drill
from scripts.ops import state_snapshot_capacity as compact


@pytest.fixture
def receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(drill, "PROJECT_ROOT", tmp_path)
    def check(guard):
        assert isinstance(guard.reserve, int)

    monkeypatch.setattr(compact.CopyGuard, "check", check)
    root = tmp_path / "archive"
    run = root / "20260911_120000_000000"
    run.mkdir(parents=True)
    archive = run / "state.json.gz"
    archive.write_bytes(b"verified archive bytes")
    source = str(tmp_path / "state.json")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    payload = {
        "timestamp_utc": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
        "ok": True, "files_checked": 1, "files_restore_verified": 1,
        "latest_write_verified": True, "published_latest_write_verified": True,
        "rows": [{"requested_source": source, "snapshot": str(archive),
                  "snapshot_sha256": "a" * 64, "restore_sha256": "a" * 64,
                  "restore_verified": True, "error": "",
                  "archive_proof": {"archive_path": str(archive),
                                    "archive_bytes": archive.stat().st_size,
                                    "archive_sha256": digest, "decoded_sha256": "a" * 64,
                                    "verification": "full_decoded_bytes_match_verified_sqlite_or_file_restore"}}],
    }
    manifest = run / "manifest.json"
    manifest.write_text(json.dumps(payload))
    published = tmp_path / "exports/latest.json"
    args = SimpleNamespace(targets=[source], operation_seconds=60)
    return args, root, manifest, payload, published, archive


def test_verified_archive_receipt_recovers_original_age(receipt):
    args, root, manifest, payload, published, _ = receipt
    original = manifest.read_bytes()
    assert drill._recover_latest_verified(args, root, published) == 0
    recovered = json.loads(published.read_text())
    assert recovered["timestamp_utc"] == payload["timestamp_utc"]
    assert recovered["receipt_republished_at_utc"] != recovered["timestamp_utc"]
    assert recovered["receipt_recovery_scope"] == "retained_archive_hashes_match_original_restore_proof"
    assert manifest.read_bytes() == original
    assert compact.complete_restore_evidence(recovered)


@pytest.mark.parametrize("approved", [False, True])
def test_receipt_verifier_passes_only_explicit_approval_to_resource_guard(receipt, monkeypatch, approved):
    args, root, _, _, published, _ = receipt
    args.operator_approved_recovery = approved
    probes = []

    def check(guard):
        probes.append(guard.operator_approved)
        assert guard.operator_approved is approved

    monkeypatch.setattr(compact.CopyGuard, "check", check)
    assert drill._recover_latest_verified(args, root, published) == 0
    assert probes


def test_receipt_verifier_does_not_publish_when_approved_resources_are_withdrawn(receipt, monkeypatch):
    args, root, _, _, published, _ = receipt
    args.operator_approved_recovery = True
    published.parent.mkdir()
    published.write_text('{"preserved": true}')

    def check(guard):
        assert guard.operator_approved
        raise RuntimeError("approved_recovery_hard_resource_admission_withdrawn")

    monkeypatch.setattr(compact.CopyGuard, "check", check)
    assert drill._recover_latest_verified(args, root, published) == 2
    assert json.loads(published.read_text()) == {"preserved": True}


@pytest.mark.parametrize("failure", ["corrupt", "missing", "future", "expired", "partial", "scope", "hash", "outside"])
def test_invalid_retained_proof_never_replaces_existing_receipt(receipt, failure):
    args, root, manifest, payload, published, archive = receipt
    published.parent.mkdir()
    published.write_text('{"preserved": true}')
    if failure == "corrupt":
        archive.write_bytes(b"x" * archive.stat().st_size)
    elif failure == "missing":
        archive.unlink()
    elif failure in {"future", "expired"}:
        payload["timestamp_utc"] = (datetime.now(timezone.utc) + timedelta(hours=1 if failure == "future" else -169)).isoformat()
    elif failure == "partial":
        payload["files_restore_verified"] = 0
    elif failure == "scope":
        args.targets.append("another required target")
    elif failure == "hash":
        payload["rows"][0]["restore_sha256"] = "b" * 64
    else:
        outside = root / "outside.gz"
        outside.write_bytes(archive.read_bytes())
        payload["rows"][0]["snapshot"] = str(outside)
        payload["rows"][0]["archive_proof"]["archive_path"] = str(outside)
    manifest.write_text(json.dumps(payload))
    assert drill._recover_latest_verified(args, root, published) == 2
    assert json.loads(published.read_text()) == {"preserved": True}


@pytest.mark.parametrize("allow_metadata", [False, True])
def test_partial_new_drill_preserves_successful_latest(receipt, monkeypatch, allow_metadata):
    args, root, _, payload, published, _ = receipt
    published.parent.mkdir()
    published.write_text(json.dumps(payload))
    (root / "latest.json").write_text(json.dumps(payload))
    args.__dict__.update(compact_sqlite=False, clone_restore=False, max_copy_bytes=1,
                         allow_large_metadata_only=allow_metadata, keep_runs=5, json=True)
    source = root / "too-large.bin"
    source.write_bytes(b"too large")
    args.targets = [str(source)]
    monkeypatch.setattr(drill, "_capacity_preflight", lambda *a: {"sufficient": True})
    assert drill._run_drill(args, root, publish_latest=published) == 2
    assert json.loads(published.read_text()) == payload
    attempt = json.loads((drill.PROJECT_ROOT / "governance/health/state_snapshot_drill_attempt_latest.json").read_text())
    assert not compact.complete_restore_evidence(attempt)
    assert attempt["previous_restore_evidence_unchanged"]
