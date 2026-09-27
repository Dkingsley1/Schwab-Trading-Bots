import hashlib
import json
from pathlib import Path

import pytest
import zstandard

from core import sqlite_primary_retirement as src


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    root, target, backup_root = [
        tmp_path / name for name in ("project", "ssd", "backup")
    ]
    directory = root / "governance/storage_recovery"
    directory.mkdir(parents=True)
    (root / "governance/locks").mkdir()
    relative = "sql_link_shards/jsonl_link_runtime.sqlite3"
    source = root / "local_fallback_storage/data" / relative
    source.parent.mkdir(parents=True)
    source.write_bytes(b"verified sqlite standby" * 100)
    active = target / "data" / relative
    active.parent.mkdir(parents=True)
    active.write_bytes(source.read_bytes())
    backup_root.mkdir()
    backup = backup_root / (source.name + ".zst")
    backup.write_bytes(zstandard.ZstdCompressor().compress(source.read_bytes()))
    row = {
        "relative": relative,
        "source": str(source),
        "source_identity": src.primary._identity(source),
        "backup": str(backup),
        "backup_identity": src.primary._identity(backup),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "restored_bytes": source.stat().st_size,
        "full_restore_hash_verified": True,
    }
    proof = directory / "backups.json"
    proof.write_text(
        json.dumps({"purpose": "sqlite_primary_independent_backups", "files": [row]})
    )
    (directory / "sqlite_primary_cutover_reviewed.json").write_text(
        json.dumps(
            {
                "files": [row],
                "target_root": str(target),
                "source_root": str(root / "local_fallback_storage/data"),
                "volume_uuid": "uuid",
            }
        )
    )
    (directory / "sqlite_primary_handoff_result.json").write_text(
        json.dumps(
            {
                "ok": True,
                "route_mutation_performed": True,
                "payload_hashes_verified": 1,
                "target_root": str(target),
            }
        )
    )
    (directory / "sqlite_primary_io_verification.json").write_text(
        json.dumps(
            {
                "ok": True,
                "queue_write_read_ack_verified": True,
                "standby_unchanged": True,
                "target_root": str(target),
                "volume_uuid": "uuid",
            }
        )
    )
    monkeypatch.setenv("BOT_LOGS_EXTERNAL_VOLUME_UUID", "uuid")
    monkeypatch.setattr(src.primary, "enabled", lambda: True)
    monkeypatch.setattr(src.primary, "_hold", lambda root: None)
    monkeypatch.setattr(src.primary, "require_ready", lambda root: target)
    monkeypatch.setattr(src.primary, "_validated_target", lambda: target)
    monkeypatch.setattr(src, "_backup_volume", lambda row, target, source: backup)
    monkeypatch.setattr(src, "_idle", lambda path: None)
    return root, proof, source, backup, active


def test_plan_preserves_source_and_apply_preserves_active_and_backup(cohort):
    root, proof, source, backup, active = cohort
    assert src.retire_standbys(root, proof)["retired"] == []
    assert source.exists()
    result = src.retire_standbys(root, proof, apply=True)
    assert len(result["retired"]) == 1
    assert not source.exists()
    assert backup.exists() and active.exists()
    assert json.loads(Path(result["journal"]).read_text())["phase"] == "complete"


@pytest.mark.parametrize(
    "damage", ["source", "backup", "active", "io", "route_binding", "duplicate"]
)
def test_missing_or_changed_evidence_preserves_source(cohort, damage):
    root, proof, source, backup, active = cohort
    if damage == "source":
        source.write_bytes(b"new source")
    elif damage == "backup":
        backup.write_bytes(b"damaged")
    elif damage == "active":
        active.unlink()
    elif damage == "io":
        (proof.parent / "sqlite_primary_io_verification.json").write_text("{}")
    elif damage == "route_binding":
        path = proof.parent / "sqlite_primary_cutover_reviewed.json"
        data = json.loads(path.read_text())
        data["target_root"] = "/wrong"
        path.write_text(json.dumps(data))
    else:
        data = json.loads(proof.read_text())
        data["files"] *= 2
        proof.write_text(json.dumps(data))
    with pytest.raises((ValueError, OSError)):
        src.retire_standbys(root, proof, apply=True)
    assert source.exists()


def test_hold_loss_before_retirement_preserves_source(cohort, monkeypatch):
    root, proof, source, backup, active = cohort
    calls = 0

    def hold(root):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("hold lost")

    monkeypatch.setattr(src.primary, "_hold", hold)
    with pytest.raises(RuntimeError, match="hold lost"):
        src.retire_standbys(root, proof, apply=True)
    assert source.exists()


def test_corrupt_restore_cannot_pass_even_with_updated_backup_identity(cohort):
    root, proof, source, backup, active = cohort
    backup.write_bytes(zstandard.ZstdCompressor().compress(b"wrong"))
    data = json.loads(proof.read_text())
    data["files"][0]["backup_identity"] = src.primary._identity(backup)
    proof.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="restore_size_mismatch"):
        src.retire_standbys(root, proof, apply=True)
    assert source.exists()


def test_video_rejected_without_any_metadata_access(monkeypatch):
    monkeypatch.setattr(
        src.primary, "_physical", lambda path: pytest.fail("metadata touched")
    )
    with pytest.raises(ValueError, match="mount_invalid"):
        src._backup_volume(
            {"backup_mount": "/Volumes/VIDEO", "backup": "/Volumes/VIDEO/file.zst"},
            Path("/ssd"),
            Path("/source"),
        )


def test_real_idle_guard_rejects_nonempty_sidecar(tmp_path, monkeypatch):
    from types import SimpleNamespace

    source = tmp_path / "db.sqlite3"
    Path(str(source) + "-shm").write_bytes(b"busy")
    monkeypatch.setattr(
        src.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout=b"", stderr=b""),
    )
    with pytest.raises(RuntimeError, match="journal_present"):
        src._idle(source)


@pytest.mark.parametrize(
    "backup_store,allowed", [("disk4s2", False), ("disk8s2", True)]
)
def test_backup_partition_is_not_independent_device(monkeypatch, backup_store, allowed):
    from types import SimpleNamespace
    import plistlib

    mount = Path("/Volumes/Backup")
    source = Path("/internal/data/db.sqlite3")
    target = Path("/Volumes/SSD/schwab_trading_bot")
    row = {
        "backup_mount": str(mount),
        "backup_volume_uuid": "backup-uuid",
        "backup": str(
            mount
            / "schwab_trading_bot/cold_archive/sqlite_primary_recovery/db.sqlite3.zst"
        ),
    }
    monkeypatch.setattr(src.primary, "_physical", lambda path: None)
    monkeypatch.setattr(src.os.path, "ismount", lambda path: True)
    monkeypatch.setattr(
        Path,
        "stat",
        lambda path: SimpleNamespace(
            st_dev=(
                1
                if str(path).startswith("/internal")
                else (2 if str(path).startswith("/Volumes/SSD") else 3)
            )
        ),
    )

    def info(command, **kwargs):
        if command[-1] == str(mount):
            return plistlib.dumps(
                {
                    "VolumeUUID": "backup-uuid",
                    "MountPoint": str(mount),
                    "Internal": False,
                    "APFSPhysicalStores": [{"APFSPhysicalStore": backup_store}],
                }
            )
        return plistlib.dumps(
            {"APFSPhysicalStores": [{"APFSPhysicalStore": "disk4s1"}]}
        )

    monkeypatch.setattr(src.subprocess, "check_output", info)
    if allowed:
        assert src._backup_volume(row, target, source) == Path(row["backup"])
    else:
        with pytest.raises(ValueError, match="independence_unproven"):
            src._backup_volume(row, target, source)
