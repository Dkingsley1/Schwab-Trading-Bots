from types import SimpleNamespace

import pytest

from scripts import sql_hot_retention as retention
from scripts.ops import sql_link_shard_manager as manager

POLICIES = [
    ("aggressive_trading", 7, 32.0),
    ("trading", 7, 32.0),
    ("governance", 14, 16.0),
    ("crypto_governance", 14, 16.0),
    ("runtime", 7, 4.0),
    ("crypto_runtime", 7, 2.0),
    ("support_watchdog", 14, 2.0),
    ("api_ingress", 7, 1.0),
    ("crypto_api_ingress", 7, 1.0),
]


@pytest.fixture(autouse=True)
def isolated_configuration(tmp_path, monkeypatch):
    for key in list(manager.os.environ):
        if key.startswith("SQL_LINK_SERVICE_"):
            monkeypatch.delenv(key)
    monkeypatch.setattr(manager, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(manager, "SHARD_DB_ROOT", tmp_path / "shards")
    monkeypatch.setattr(manager, "SHARD_STATE_ROOT", tmp_path / "state")
    monkeypatch.setattr(manager, "HEALTH_ROOT", tmp_path / "health")
    monkeypatch.setattr(manager, "EVENT_ROOT", tmp_path / "events")
    monkeypatch.setattr(manager.ops_data_plane, "load_shard_heat_map", lambda _: {})
    monkeypatch.setattr(
        manager,
        "_approved_second_cold_sql_root",
        lambda name, *, kind: tmp_path / "cold" / kind / name,
    )


@pytest.mark.parametrize("name,days,size_trigger", POLICIES)
def test_audited_shards_have_bounded_preserving_hot_policies(name, days, size_trigger):
    spec = manager._build_shards([name])[0]
    assert spec["hot_retention_enabled"] is True
    assert spec["hot_retention_hot_days"] == days
    assert spec["hot_retention_hot_hours"] == 0
    assert spec["hot_retention_max_db_gb"] == size_trigger
    assert spec["hot_retention_trigger_growth_gb"] > 0
    assert spec["hot_retention_trigger_rows"] == 100000
    assert spec["hot_retention_archive_period"] == "day"
    assert spec["hot_retention_archive_retention_days"] == 0
    assert spec["hot_retention_batch_size"] == 1000
    assert spec["hot_retention_max_rows"] == 5000
    assert spec["hot_retention_min_interval_seconds"] == 300
    assert spec["hot_retention_vacuum_threshold_gb"] == 0
    assert str(spec["hot_retention_archive_root"]).endswith("archives/" + name)
    assert spec["hot_retention_cold_export_root"]


@pytest.mark.parametrize("name,days,size_trigger", POLICIES)
def test_new_policies_reach_existing_bounded_archive_command(
    tmp_path, monkeypatch, name, days, size_trigger
):
    spec = manager._build_shards([name])[0]
    monkeypatch.setenv("SQL_LINK_SERVICE_BOUNDED_STORAGE_RETENTION", "1")
    captured = {}

    def run(cmd, **kwargs):
        captured.update(cmd=cmd, **kwargs)
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(manager.subprocess, "run", run)
    manager._run_hot_retention(
        db_path=spec["sqlite_db"],
        hot_days=spec["hot_retention_hot_days"],
        hot_hours=spec["hot_retention_hot_hours"],
        batch_size=spec["hot_retention_batch_size"],
        max_rows=spec["hot_retention_max_rows"],
        archive_db=str(tmp_path / "unused.sqlite3"),
        archive_root=spec["hot_retention_archive_root"],
        archive_period=spec["hot_retention_archive_period"],
        archive_retention_days=spec["hot_retention_archive_retention_days"],
        archive_prune_vacuum=True,
        cold_export_root=spec["hot_retention_cold_export_root"],
        cold_export_format="parquet",
        cold_export_batch_size=1000,
        cold_export_compression="zstd",
        vacuum=False,
    )
    cmd = captured["cmd"]
    assert cmd[cmd.index("--hot-days") + 1] == str(days)
    assert cmd[cmd.index("--archive-retention-days") + 1] == "0"
    assert cmd[cmd.index("--batch-size") + 1] == "1000"
    assert cmd[cmd.index("--max-rows") + 1] == "5000"
    assert cmd[cmd.index("--min-archive-free-gb") + 1] == "64"
    assert captured["timeout"] == 120
    assert "--preserve-conflicting-versions" in cmd
    assert "--vacuum" not in cmd
    assert "--archive-prune-vacuum" not in cmd


@pytest.mark.parametrize("days", [0, 60])
def test_explicit_archive_retention_override_preserves_zero(monkeypatch, days):
    monkeypatch.setenv(
        "SQL_LINK_SERVICE_SHARD_TRADING_HOT_RETENTION_ARCHIVE_RETENTION_DAYS", str(days)
    )
    assert (
        manager._build_shards(["trading"])[0]["hot_retention_archive_retention_days"]
        == days
    )


def test_existing_retention_and_merge_contracts_are_unchanged():
    specs = {
        r["name"]: r
        for r in manager._build_shards(
            ["crypto_trading", "risk_support", "explanations", "data", "health_fast"]
        )
    }
    assert specs["crypto_trading"]["hot_retention_hot_days"] == 7
    assert specs["crypto_trading"]["hot_retention_archive_retention_days"] == 180
    assert specs["crypto_trading"]["hot_retention_max_rows"] == 2500000
    assert specs["risk_support"]["hot_retention_archive_retention_days"] == 180
    assert specs["explanations"]["hot_retention_archive_retention_days"] == 365
    assert specs["data"]["hot_retention_enabled"] is False
    assert specs["health_fast"]["hot_retention_enabled"] is False


def test_zero_expiry_never_opens_or_prunes_an_archive(tmp_path, monkeypatch):
    archive = tmp_path / "historical.sqlite3"
    archive.write_bytes(b"historical evidence left untouched")
    before = archive.read_bytes()

    def unexpected(*args, **kwargs):
        raise AssertionError("archive expiry must not open a database")

    monkeypatch.setattr(retention, "_connect", unexpected)
    result = retention._prune_archive_storage(
        archive_db=archive,
        archive_root=tmp_path,
        archive_retention_days=0,
        archive_prune_vacuum=True,
        cold_export_root=tmp_path / "cold",
        cold_export_format="parquet",
        cold_export_batch_size=1000,
        cold_export_compression="zstd",
    )
    assert result["enabled"] is False
    assert result["pruned_rows"] == 0
    assert result["deleted_archive_files"] == []
    assert archive.read_bytes() == before
