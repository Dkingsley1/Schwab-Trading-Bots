import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from scripts.ops import deep_cold_storage_layer as src

NOW = datetime(2026, 9, 15, 17, tzinfo=timezone.utc)


def backup(root, name, *, age=172800):
    path = root / "backups" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'{"backup": true}\n')
    stamp = NOW.timestamp() - age
    os.utime(path, (stamp, stamp))
    return path


def test_old_named_backups_only_and_newest_per_family_stays_local(tmp_path):
    old = backup(
        tmp_path,
        "master_bot_registry_before_whole_system_governor_20260910_000000.json",
    )
    backup(
        tmp_path,
        "master_bot_registry_before_whole_system_governor_20260911_000000.json",
    )
    backup(tmp_path, "master_bot_registry_before_training_20260910_000000.json")
    backup(tmp_path, "unrelated.json")
    backup(tmp_path, "master_bot_registry_before_other_20269999_000000.json")
    assert src._closed_registry_backups(tmp_path, now=NOW, min_size_bytes=1) == [old]


def test_name_and_mtime_must_both_be_old(tmp_path):
    backup(tmp_path, "master_bot_registry_before_test_20260910_000000.json", age=10)
    backup(tmp_path, "master_bot_registry_before_test_20260915_000000.json")
    backup(tmp_path, "master_bot_registry_before_test_20260915_010000.json")
    assert src._closed_registry_backups(tmp_path, now=NOW, min_size_bytes=1) == []


def test_protected_backup_root_is_not_enumerated(tmp_path, monkeypatch):
    (tmp_path / "backups").symlink_to("/Volumes/VIDEO/private")
    monkeypatch.setattr(
        Path,
        "iterdir",
        lambda _: (_ for _ in ()).throw(AssertionError("enumerated protected root")),
    )
    assert src._closed_registry_backups(tmp_path, now=NOW, min_size_bytes=1) == []


def test_verified_move_is_opt_in_preserves_bytes_and_keeps_latest(
    tmp_path, monkeypatch
):
    root = tmp_path / "project"
    old = backup(root, "master_bot_registry_before_governor_20260910_000000.json")
    latest = backup(root, "master_bot_registry_before_governor_20260911_000000.json")
    external = tmp_path / "external"
    target = tmp_path / "lacie"
    monkeypatch.setattr(
        src, "resolve_external_storage", lambda: SimpleNamespace(external_root=external)
    )
    monkeypatch.setattr(
        src.shutil,
        "disk_usage",
        lambda _: SimpleNamespace(
            total=1024**4, free=512 * 1024**3, used=512 * 1024**3
        ),
    )
    options = dict(
        apply=True,
        min_size_mb=0.000001,
        move_to_second_cold=True,
        second_cold_root=target,
        max_move_gb=1,
    )
    assert src.build_payload(root, **options)["second_cold_move"]["moved_files"] == 0
    result = src.build_payload(root, include_registry_backups=True, **options)
    assert result["second_cold_move"]["moved_files"] == 1
    assert old.is_symlink()
    assert "registry_backups" in str(old.resolve())
    assert old.read_bytes() == b'{"backup": true}\n'
    assert not latest.is_symlink()
    assert result["second_cold_move"]["actions"][0]["verified_sha256_match"]
