from types import SimpleNamespace
import subprocess

import pytest

from scripts import sql_hot_retention as retention
from scripts.ops import sql_link_shard_manager as manager
from scripts.ops import storage_maintenance_lane as lane


def _run_options(tmp_path):
    return dict(
        db_path=tmp_path / "source.sqlite3",
        hot_days=5,
        hot_hours=18,
        batch_size=30000,
        max_rows=5400000,
        archive_db=str(tmp_path / "archive.sqlite3"),
        archive_root=str(tmp_path),
        archive_period="day",
        archive_retention_days=10,
        archive_prune_vacuum=True,
        cold_export_root="",
        cold_export_format="parquet",
        cold_export_batch_size=50000,
        cold_export_compression="zstd",
        vacuum=True,
    )


def test_maintenance_retention_is_bounded_and_preserves_archive_history(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SQL_LINK_SERVICE_BOUNDED_STORAGE_RETENTION", "1")
    captured = {}

    def run(cmd, **kwargs):
        captured.update(cmd=cmd, **kwargs)
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(manager.subprocess, "run", run)
    result = manager._run_hot_retention(**_run_options(tmp_path))
    cmd = captured["cmd"]
    assert result[0] == 0
    assert captured["timeout"] == 120
    assert cmd[cmd.index("--batch-size") + 1] == "1000"
    assert cmd[cmd.index("--max-rows") + 1] == "5000"
    assert cmd[cmd.index("--archive-retention-days") + 1] == "0"
    assert cmd[cmd.index("--min-archive-free-gb") + 1] == "64"
    assert "--vacuum" not in cmd
    assert "--archive-prune-vacuum" not in cmd
    assert "--preserve-conflicting-versions" in cmd


def test_maintenance_timeout_cannot_claim_retention_completion(tmp_path, monkeypatch):
    monkeypatch.setenv("SQL_LINK_SERVICE_BOUNDED_STORAGE_RETENTION", "1")

    def timeout(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 120)

    monkeypatch.setattr(manager.subprocess, "run", timeout)
    rc, out, err = manager._run_hot_retention(**_run_options(tmp_path))
    assert rc == 124
    assert out == ""
    assert "source_or_verified_archive_preserved" in err


def test_ordinary_one_pass_writers_keep_deferring_retention(monkeypatch):
    monkeypatch.delenv("SQL_LINK_SERVICE_ONCE_INLINE_RETENTION", raising=False)
    assert manager._once_inline_retention_enabled(SimpleNamespace(once=True)) is False


@pytest.mark.parametrize("free", [64 * 1024**3, None])
def test_archive_batch_reserves_scratch_and_fails_closed(tmp_path, monkeypatch, free):
    def usage(path):
        if free is None:
            raise OSError("volume unavailable")
        return SimpleNamespace(free=free)

    monkeypatch.setattr(retention.shutil, "disk_usage", usage)
    with pytest.raises((RuntimeError, OSError)):
        retention._archive_batch_capacity(
            tmp_path / "archive.sqlite3", [(1, "payload")], reserve_gb=64
        )


def test_archive_batch_capacity_admits_only_with_headroom(tmp_path, monkeypatch):
    monkeypatch.setattr(
        retention.shutil, "disk_usage", lambda _: SimpleNamespace(free=65 * 1024**3)
    )
    retention._archive_batch_capacity(
        tmp_path / "archive.sqlite3", [(1, "payload", b"bytes", None)], reserve_gb=64
    )


def test_ingestion_success_does_not_mask_failed_retention():
    result = {
        "rc": 0,
        "payload": {"ok": True, "hot_retention": {"ran": True, "rc": 124}},
    }
    normalized = lane._normalize_retention_result(result)
    assert normalized["rc"] == 1
    assert normalized["payload"]["ok"] is False
    assert normalized["payload"]["reason"] == "storage_retention_batch_incomplete"
    assert result["payload"]["ok"] is True


@pytest.mark.parametrize(
    "admitted,force", [(False, False), (False, True), (True, True)]
)
def test_force_does_not_bypass_retention_resource_admission(
    tmp_path, monkeypatch, admitted, force
):
    envs = []
    monkeypatch.setattr(lane, "_storage_roots", lambda _: (tmp_path, tmp_path))
    monkeypatch.setattr(lane, "_priority_retention_focus", lambda *args: {})

    def run(cmd, **kwargs):
        resource = any("resource_guard.py" in str(part) for part in cmd)
        if any("sql_link_shard_manager.py" in str(part) for part in cmd):
            envs.append(kwargs["env_overrides"])
        return {"cmd": cmd, "rc": 0, "payload": {"ok": admitted if resource else True}}

    monkeypatch.setattr(lane, "_run_json_command", run)
    lane.build_storage_maintenance_payload(
        tmp_path, resource_profile="optional", force=force, vacuum=False
    )
    if not admitted and not force:
        assert envs == []
    else:
        assert envs[0]["SQL_LINK_SERVICE_ONCE_INLINE_RETENTION"] == (
            "1" if admitted else "0"
        )
