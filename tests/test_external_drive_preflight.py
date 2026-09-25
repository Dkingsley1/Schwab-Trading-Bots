import json
import plistlib
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.ops import external_drive_preflight as src

VOLUME = Path("/Volumes/BOT_DATA")
VOLUME_UUID = "A133EDB3-4CAA-42EF-A5B8-1F0BFD118BB5"
NOW = datetime(2026, 9, 24, 22, tzinfo=timezone.utc)


def test_unconnected_plan_has_no_external_probe(monkeypatch, tmp_path):
    monkeypatch.setattr(
        src, "_disk_info", lambda *_: pytest.fail("unexpected disk probe")
    )
    result = src.build_payload(tmp_path, now=NOW)
    assert result["state"] == "awaiting_drive_selection"
    assert result["role"] == "primary_platform_data_with_cold_archives"
    assert "data/sql_link_shards" in result["planned_external_families"]
    assert "eligible_cold_archives" in result["planned_external_families"]
    assert not result["activation_ready"]
    assert not any(result["authority"].values())
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "mount",
    [
        "/Volumes/VIDEO",
        "/Volumes/video",
        "/Volumes/BOT_LOGS",
        "/Volumes/VIDEO/../BOT_DATA",
        "/Volumes",
        "/",
        "BOT_DATA",
        "/Volumes/BOT_DATA/subdirectory",
    ],
)
def test_unsafe_mount_rejected_before_target_metadata(monkeypatch, mount):
    monkeypatch.setattr(src, "_inventory", lambda *_: {})
    monkeypatch.setattr(
        src, "inspect_storage_path", lambda *_: pytest.fail("metadata touched")
    )
    result = src.build_payload(mount=mount)
    assert result["state"] == "blocked"
    assert not result["preflight_ok"]


@pytest.fixture
def volume(monkeypatch):
    monkeypatch.setattr(src, "_inventory", lambda *_: {})
    monkeypatch.delenv("BOT_LOGS_EXTERNAL_VOLUME_UUID", raising=False)
    monkeypatch.setattr(
        src,
        "inspect_storage_path",
        lambda *_: {
            "status": "present",
            "kind": "directory",
            "symlinks": [],
        },
    )
    original_mount, original_stat = Path.is_mount, Path.stat
    monkeypatch.setattr(
        Path, "is_mount", lambda p: True if p == VOLUME else original_mount(p)
    )
    monkeypatch.setattr(
        Path,
        "stat",
        lambda p, **kw: (
            SimpleNamespace(st_dev=9, st_ino=2)
            if p == VOLUME
            else original_stat(p, **kw)
        ),
    )
    info = {
        "Internal": False,
        "MountPoint": str(VOLUME),
        "FilesystemType": "apfs",
        "WritableVolume": True,
        "VolumeUUID": VOLUME_UUID,
    }
    monkeypatch.setattr(src, "_disk_info", lambda *_: info)
    monkeypatch.setattr(
        src.shutil,
        "disk_usage",
        lambda *_: SimpleNamespace(total=2_000_000_000_000, free=1_900_000_000_000),
    )
    return info


def test_metadata_pass_never_authorizes_activation(volume):
    result = src.build_payload(mount=str(VOLUME), expected_uuid=VOLUME_UUID.lower())
    assert result["preflight_ok"]
    assert result["state"] == "metadata_preflight_passed"
    assert result["volume"]["planned_reserve_bytes"] == 200_000_000_000
    assert not result["volume"]["write_test_performed"]
    assert not result["activation_ready"]
    assert not any(result["authority"].values())


@pytest.mark.parametrize(
    ("key", "value", "blocker"),
    [
        ("Internal", True, "external_device_not_confirmed"),
        ("Internal", None, "external_device_not_confirmed"),
        ("WritableVolume", False, "writable_volume_not_confirmed"),
        ("FilesystemType", "exfat", "apfs_required_for_primary_data_plan"),
        ("MountPoint", "/Volumes/DIFFERENT", "mount_identity_mismatch"),
    ],
)
def test_unqualified_volume_blocks(volume, key, value, blocker):
    volume[key] = value
    result = src.build_payload(mount=str(VOLUME), expected_uuid=VOLUME_UUID)
    assert blocker in result["blockers"]
    assert not result["preflight_ok"]


@pytest.mark.parametrize(
    "expected", ["", "F133EDB3-4CAA-42EF-A5B8-1F0BFD118BB5", "invalid"]
)
def test_identity_is_operator_confirmed_and_fail_closed(volume, expected):
    assert not src.build_payload(mount=str(VOLUME), expected_uuid=expected)[
        "preflight_ok"
    ]


def test_existing_volume_is_not_adopted_under_new_name(volume, monkeypatch):
    monkeypatch.setenv("BOT_LOGS_EXTERNAL_VOLUME_UUID", VOLUME_UUID)
    result = src.build_payload(mount=str(VOLUME), expected_uuid=VOLUME_UUID)
    assert "existing_bot_logs_volume_is_not_the_new_drive" in result["blockers"]


@pytest.mark.parametrize("status", ["missing", "protected_path", "inspection_error"])
def test_missing_and_protected_routes_do_not_probe(volume, monkeypatch, status):
    monkeypatch.setattr(src, "inspect_storage_path", lambda *_: {"status": status})
    monkeypatch.setattr(src, "_disk_info", lambda *_: pytest.fail("disk touched"))
    assert not src.build_payload(mount=str(VOLUME))["preflight_ok"]


def test_symlink_alias_is_rejected(volume, monkeypatch):
    monkeypatch.setattr(
        src,
        "inspect_storage_path",
        lambda *_: {
            "status": "present",
            "kind": "directory",
            "symlinks": [{"target": "/Volumes/VIDEO"}],
        },
    )
    monkeypatch.setattr(src, "_disk_info", lambda *_: pytest.fail("alias touched"))
    assert not src.build_payload(mount=str(VOLUME))["preflight_ok"]


def test_unmounted_directory_cannot_pass(volume, monkeypatch):
    monkeypatch.setattr(Path, "is_mount", lambda _: False)
    monkeypatch.setattr(
        src, "_disk_info", lambda *_: pytest.fail("unmounted directory touched")
    )
    assert not src.build_payload(mount=str(VOLUME))["preflight_ok"]


def test_disconnect_or_remount_during_probe_blocks(volume, monkeypatch):
    original = Path.stat
    count = 0

    def stat(path, **kwargs):
        nonlocal count
        if path != VOLUME:
            return original(path, **kwargs)
        count += 1
        return SimpleNamespace(st_dev=count, st_ino=2)

    monkeypatch.setattr(Path, "stat", stat)
    assert not src.build_payload(mount=str(VOLUME), expected_uuid=VOLUME_UUID)[
        "preflight_ok"
    ]


@pytest.mark.parametrize(
    ("total", "free", "blocker"),
    [
        (
            1_000_000_000_000,
            900_000_000_000,
            "capacity_smaller_than_expected_2tb_drive",
        ),
        (2_000_000_000_000, 150 * src.GIB, "insufficient_destination_headroom"),
    ],
)
def test_capacity_and_reserve_stay_explicit(volume, monkeypatch, total, free, blocker):
    monkeypatch.setattr(
        src.shutil, "disk_usage", lambda *_: SimpleNamespace(total=total, free=free)
    )
    result = src.build_payload(mount=str(VOLUME), expected_uuid=VOLUME_UUID)
    assert blocker in result["blockers"]


def test_disk_query_is_targeted_read_only_and_has_timeout(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0, stdout=plistlib.dumps({"VolumeUUID": VOLUME_UUID})
        )

    monkeypatch.setattr(src.subprocess, "run", run)
    assert src._disk_info(VOLUME)["VolumeUUID"] == VOLUME_UUID
    assert calls[0][0] == ["/usr/sbin/diskutil", "info", "-plist", str(VOLUME)]
    assert calls[0][1]["timeout"] == 10


def test_probe_timeout_is_not_readiness(volume, monkeypatch):
    def timeout(*_):
        raise subprocess.TimeoutExpired("diskutil", 10)

    monkeypatch.setattr(src, "_disk_info", timeout)
    assert not src.build_payload(mount=str(VOLUME), expected_uuid=VOLUME_UUID)[
        "preflight_ok"
    ]


@pytest.mark.parametrize("age", [0, 7200, -60])
def test_inventory_preserves_original_age_and_does_not_claim_reclaimable_space(
    tmp_path, age
):
    path = tmp_path / "governance/health/storage_tier_policy_latest.json"
    path.parent.mkdir(parents=True)
    raw = json.dumps(
        {
            "timestamp_utc": (NOW - timedelta(seconds=age)).isoformat(),
            "by_family": {"sql_link_shards": {"bytes": 300 * src.GIB}},
        }
    )
    path.write_text(raw)
    result = src._inventory(tmp_path, NOW)
    assert result["state"] == ("fresh" if age == 0 else "stale_or_future")
    assert result["reported_bytes"] == 300 * src.GIB
    assert result["reclaimable_bytes"] is None
    assert path.read_text() == raw


def test_inventory_rejects_oversized_report(tmp_path, monkeypatch):
    path = tmp_path / "governance/health/storage_tier_policy_latest.json"
    path.parent.mkdir(parents=True)
    path.write_text("x" * 17)
    monkeypatch.setattr(src, "REPORT_LIMIT", 16)
    assert src._inventory(tmp_path, NOW)["state"] == "report_exceeds_read_budget"


def test_no_apply_or_format_interface():
    with pytest.raises(SystemExit) as exc:
        src.main(["--apply"])
    assert exc.value.code == 2
