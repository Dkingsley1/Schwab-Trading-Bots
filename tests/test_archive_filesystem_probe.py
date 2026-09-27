import plistlib

import pytest

from scripts.ops import cold_sqlite_filesystem_compaction as compactor


@pytest.mark.parametrize(
    "rc,stdout,timed_out,error",
    [
        (1, "", False, "filesystem_type_probe_failed"),
        (0, "", True, "filesystem_type_probe_failed"),
        (0, "not a plist", False, "filesystem_type_probe_incomplete"),
        (0, plistlib.dumps({}).decode(), False, "filesystem_type_probe_incomplete"),
        (
            0,
            plistlib.dumps({"FilesystemType": "exfat"}).decode(),
            False,
            "transparent_compression_requires_apfs",
        ),
    ],
)
def test_unknown_filesystem_remains_blocked_without_claiming_unsupported_format(
    tmp_path, monkeypatch, rc, stdout, timed_out, error
):
    monkeypatch.setattr(
        compactor, "_run", lambda *args: dict(rc=rc, stdout=stdout, timed_out=timed_out)
    )
    with pytest.raises(RuntimeError, match=f"^{error}$"):
        compactor._require_apfs("/dev/disk5s1", tmp_path, 100)


def test_successful_apfs_probe_is_required(tmp_path, monkeypatch):
    monkeypatch.setattr(
        compactor,
        "_run",
        lambda *args: dict(
            rc=0, stdout=plistlib.dumps({"FilesystemType": "apfs"}).decode()
        ),
    )
    compactor._require_apfs("/dev/disk5s1", tmp_path, 100)
