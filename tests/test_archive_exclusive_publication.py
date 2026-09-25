import errno
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import pytest

from scripts.ops import deep_cold_storage_layer as src


@pytest.fixture
def copy_fallback(monkeypatch):
    monkeypatch.setattr(src.sys, "platform", "linux")

    def unsupported(*args, **kwargs):
        raise OSError(errno.ENOTSUP, "filesystem does not support links")

    monkeypatch.setattr(src.os, "link", unsupported)


def test_exclusive_copy_fallback_verifies_and_preserves_reserve(
    tmp_path, copy_fallback
):
    partial, target = tmp_path / "partial", tmp_path / "archive"
    partial.write_bytes(b"retained evidence")
    assert (
        src._publish_verified_copy(partial, target, reserve_bytes=1)
        == "verified_exclusive_copy"
    )
    assert not partial.exists()
    assert target.read_bytes() == b"retained evidence"


def test_exclusive_copy_fallback_never_overwrites(tmp_path, copy_fallback):
    partial, target = tmp_path / "partial", tmp_path / "archive"
    partial.write_bytes(b"new")
    target.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        src._publish_verified_copy(partial, target)
    assert partial.read_bytes() == b"new"
    assert target.read_bytes() == b"existing"


@pytest.mark.parametrize("free", [[10], [100, 10]])
def test_exclusive_copy_reserve_failure_preserves_verified_partial(
    tmp_path, monkeypatch, copy_fallback, free
):
    partial, target = tmp_path / "partial", tmp_path / "archive"
    partial.write_bytes(b"retained evidence")
    values = iter(free)
    monkeypatch.setattr(
        src.shutil, "disk_usage", lambda path: SimpleNamespace(free=next(values))
    )
    with pytest.raises(RuntimeError, match="reserve"):
        src._publish_verified_copy(partial, target, reserve_bytes=10)
    assert partial.read_bytes() == b"retained evidence"
    assert not target.exists()


def test_exclusive_copy_hash_failure_cleans_only_new_target(
    tmp_path, monkeypatch, copy_fallback
):
    partial, target = tmp_path / "partial", tmp_path / "archive"
    partial.write_bytes(b"retained evidence")
    monkeypatch.setattr(src, "_sha256", lambda path: str(path))
    with pytest.raises(RuntimeError, match="verification_failed"):
        src._publish_verified_copy(partial, target)
    assert partial.read_bytes() == b"retained evidence"
    assert not target.exists()


def test_permission_error_never_uses_copy_fallback(
    tmp_path, monkeypatch, copy_fallback
):
    partial, target = tmp_path / "partial", tmp_path / "archive"
    partial.write_bytes(b"retained evidence")

    def denied(*args, **kwargs):
        raise PermissionError(errno.EACCES, "denied")

    monkeypatch.setattr(src.os, "link", denied)
    with pytest.raises(PermissionError):
        src._publish_verified_copy(partial, target)
    assert partial.exists()
    assert not target.exists()


def test_initial_copy_keeps_original_on_reserve_failure(tmp_path, monkeypatch):
    original, partial = tmp_path / "source", tmp_path / "partial"
    original.write_bytes(b"retained evidence")
    monkeypatch.setattr(src.shutil, "disk_usage", lambda path: SimpleNamespace(free=10))
    with pytest.raises(RuntimeError, match="reserve"):
        src._copy_with_sha256(original, partial, reserve_bytes=10)
    assert original.read_bytes() == b"retained evidence"
    assert partial.stat().st_size == 0


def test_absent_external_mount_never_uses_local_disk_or_creates_folder(
    tmp_path, monkeypatch
):
    target = Path("/Volumes/unmounted-archive-test/platform/archive")
    monkeypatch.setattr(src, "_is_protected_volume", lambda path: False)
    monkeypatch.setattr(Path, "is_mount", lambda path: False)

    def forbidden(*args, **kwargs):
        pytest.fail("must not probe local fallback capacity or create a fake mount")

    monkeypatch.setattr(src.shutil, "disk_usage", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    assert src._disk_usage_snapshot(target)["free_bytes"] is None
    result = src._copy_verify_then_symlink(tmp_path / "source", target)
    assert result["reason"] == "destination_volume_unavailable"
    assert not result["source_replaced_with_symlink"]


def test_exclusive_publication_never_overwrites_existing_target(tmp_path):
    source, target = tmp_path / "partial", tmp_path / "archive"
    source.write_bytes(b"verified new data")
    target.write_bytes(b"existing user data")
    with pytest.raises(FileExistsError):
        src._publish_verified_copy(source, target)
    assert source.read_bytes() == b"verified new data"
    assert target.read_bytes() == b"existing user data"


def test_offload_root_overrides_compression_root_without_changing_it(tmp_path, monkeypatch):
    compression, offload = tmp_path / "apfs", tmp_path / "separate"
    monkeypatch.setenv("BOT_SECOND_COLD_ROOT", str(compression))
    monkeypatch.setenv("BOT_DEEP_COLD_OFFLOAD_ROOT", str(offload))
    monkeypatch.setattr(src, "resolve_external_storage", lambda: SimpleNamespace(external_root=tmp_path / "external"))
    payload = src.build_payload(tmp_path / "project", apply=False, move_to_second_cold=True)
    assert payload["second_cold_move"]["second_cold_root"] == str(offload)
    assert os.environ["BOT_SECOND_COLD_ROOT"] == str(compression)


def test_publication_failure_preserves_original_and_verified_partial(
    tmp_path, monkeypatch
):
    source, target = tmp_path / "original", tmp_path / "archive"
    source.write_bytes(b"retained evidence")

    def fail(*args, **kwargs):
        raise OSError(errno.ENOTSUP, "unsupported")

    monkeypatch.setattr(src, "_publish_verified_copy", fail)
    result = src._copy_verify_then_symlink(source, target)
    assert not result["source_replaced_with_symlink"]
    assert source.read_bytes() == b"retained evidence"
    assert not source.is_symlink()
    assert not target.exists()
    assert (tmp_path / ".archive.tmp").read_bytes() == source.read_bytes()


def test_native_archive_publication_and_restore_on_operator_selected_volume(tmp_path):
    root = os.getenv("TEST_VERIFIED_ARCHIVE_ROOT")
    if not root:
        pytest.skip("operator-selected test archive volume not provided")
    root = Path(root)
    assert src.inspect_storage_path(root)["status"] in {"present", "missing"}
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".publication-test-", dir=root) as folder:
        target = Path(folder) / "archive.bin"
        source = tmp_path / "source.bin"
        payload = bytes(range(256)) * 4096
        source.write_bytes(payload)
        result = src._copy_verify_then_symlink(source, target)
        assert result["source_replaced_with_symlink"], result.get("reason")
        assert result["verified_sha256_match"]
        assert source.is_symlink()
        assert source.read_bytes() == target.read_bytes() == payload
        assert Path(result["restore_proof_path"]).is_file()
        if sys.platform == "darwin":
            assert result["publication_method"] in {
                "exclusive_rename",
                "verified_exclusive_copy",
            }
        conflict = Path(folder) / "conflict.bin"
        conflict.write_bytes(b"must remain")
        with pytest.raises(FileExistsError):
            src._publish_verified_copy(conflict, target)
        assert conflict.read_bytes() == b"must remain"
        assert target.read_bytes() == payload
