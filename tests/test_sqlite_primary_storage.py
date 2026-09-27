import hashlib
import json
import os
import plistlib
import shutil
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import sqlite_primary_storage as src

UUID = "CF28B097-41B2-4A8B-8E9F-210FC6DE7D8D"


def managed_config(root):
    from core.storage_target_override import build_storage_target_override_text

    path = root / "config/.env.storage_target_override"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        build_storage_target_override_text(
            mount_root="/Volumes/Fixture SSD",
            volume_uuid=UUID,
            route_profile="sqlite_primary",
        )
    )
    return path


def test_saved_profile_prevents_stale_launcher_route_reversal(cohort, monkeypatch):
    from core.storage_router import route_runtime_storage

    root, source, target, receipt = cohort
    managed_config(root)
    monkeypatch.setenv("BOT_STORAGE_ROUTE_PROFILE", "")
    monkeypatch.setenv("BOT_LOGS_PREFER_EXTERNAL", "0")
    before = {name: os.readlink(root / name) for name in src.LINKS}
    with pytest.raises(RuntimeError, match="sqlite_primary_deferred"):
        route_runtime_storage(root)
    assert before == {name: os.readlink(root / name) for name in src.LINKS}
    src.commit_routes(root, receipt)
    result = route_runtime_storage(root)
    assert result.mode == "sqlite_primary"
    assert all(
        (root / name).resolve(strict=False) == target / name for name in src.LINKS
    )
    assert os.environ["BOT_STORAGE_ROUTE_PROFILE"] == ""


def test_saved_target_is_context_local_and_honors_spaces(tmp_path, monkeypatch):
    first, other = tmp_path / "one", tmp_path / "two"
    managed_config(first)
    monkeypatch.setenv("BOT_STORAGE_ROUTE_PROFILE", "")
    monkeypatch.setenv("BOT_LOGS_EXTERNAL_MOUNT", "/Volumes/Old")
    assert src.enabled(first)
    assert src.configured_target() == Path("/Volumes/Fixture SSD/schwab_trading_bot")
    assert os.environ["BOT_LOGS_EXTERNAL_MOUNT"] == "/Volumes/Old"
    assert not src.enabled(other)


def test_route_health_contract_is_scoped_and_fail_closed(cohort):
    root, source, target, receipt = cohort
    pending = src.observe(root)["route_verification"]
    assert pending["verification_state"] == "blocked"
    assert pending["ready_count"] == 0
    assert pending["blockers"]
    src.commit_routes(root, receipt)
    ready = src.observe(root)["route_verification"]
    assert ready["verification_state"] == "ready"
    assert ready["certified_mode"] == "sqlite_primary"
    assert ready["scope"] == "declared_sqlite_routes_only"
    assert ready["ready_count"] == ready["tracked_count"] == len(src.LINKS)
    assert ready["coverage_ratio"] == 1.0
    assert ready["integrity_verified"] is False
    assert ready["ingestion_verified"] is False
    (root / src.LINKS[0]).unlink()
    failed = src.observe(root)["route_verification"]
    assert failed["verification_state"] == "blocked"
    assert failed["coverage_ratio"] == 0.0
    assert failed["mismatches"]


@pytest.mark.parametrize("damage", ["duplicate", "incomplete", "oversized", "symlink"])
def test_invalid_saved_selection_cannot_fall_back_to_legacy(
    tmp_path, monkeypatch, damage
):
    path = managed_config(tmp_path)
    monkeypatch.delenv("BOT_STORAGE_ROUTE_PROFILE", raising=False)
    if damage == "duplicate":
        path.write_text(path.read_text() + "BOT_STORAGE_ROUTE_PROFILE=\n")
    elif damage == "incomplete":
        path.write_text("BOT_STORAGE_ROUTE_PROFILE=sqlite_primary\n")
    elif damage == "oversized":
        path.write_text("#" * 20000)
    else:
        other = tmp_path / "target"
        path.rename(other)
        path.symlink_to(other)
    with pytest.raises(ValueError):
        src.enabled(tmp_path)


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    root, target = tmp_path / "project", tmp_path / "mount" / "platform"
    source = root / "local_fallback_storage/data"
    for path in (
        source / "sql_link_shards",
        target / "data/sql_link_shards",
        root / "data",
        root / "governance/locks",
        root / "governance/runtime",
        root / "governance/health",
        root / "governance/storage_recovery",
    ):
        path.mkdir(parents=True)
    monkeypatch.setenv("BOT_STORAGE_ROUTE_PROFILE", src.PROFILE)
    monkeypatch.setenv("BOT_LOGS_EXTERNAL_VOLUME_UUID", UUID)
    monkeypatch.setattr(src, "configured_target", lambda: target)
    monkeypatch.setattr(src, "_validated_target", lambda: target)
    monkeypatch.setattr(src, "_hold", lambda _: None)
    monkeypatch.setattr(
        src.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout=b"", stderr=b""),
    )
    rows = []
    for rel in (*src.DATABASES, "sql_link_shards/trading.sqlite3"):
        path = source / rel
        connection = sqlite3.connect(path)
        connection.execute("CREATE TABLE evidence(value INTEGER)")
        connection.execute("INSERT INTO evidence VALUES (42)")
        connection.commit()
        connection.close()
        dest = target / "data" / rel
        shutil.copyfile(path, dest)
        rows.append(
            {
                "relative": rel,
                "source_identity": src._identity(path),
                "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
                "quick_check": "ok",
            }
        )
    for rel in src.LINKS:
        (root / rel).symlink_to(root / "local_fallback_storage" / rel)
    receipt = root / "governance/storage_recovery/handoff.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "sqlite_primary_cutover",
                "source_root": str(source),
                "target_root": str(target),
                "volume_uuid": UUID,
                "files": rows,
            }
        )
    )
    return root, source, target, receipt


def test_observation_is_read_only_and_not_adoption(cohort):
    root, source, target, _ = cohort
    before = {r: os.readlink(root / r) for r in src.LINKS}
    observed = src.observe(root)
    assert observed["mode"] == "sqlite_primary_pending"
    assert not observed["ok"]
    assert not observed["route_mutation_performed"]
    assert before == {r: os.readlink(root / r) for r in src.LINKS}


def test_failback_cli_observes_without_implicit_adoption(cohort, monkeypatch, capsys):
    from scripts.ops import storage_failback_sync as owner

    root, source, _, _ = cohort
    monkeypatch.setattr(owner, "PROJECT_ROOT", root)
    monkeypatch.delenv("STORAGE_FAILBACK_SYNC_LOCK_PATH", raising=False)
    monkeypatch.setattr(sys, "argv", ["storage_failback_sync", "--json"])
    monkeypatch.setattr(src, "commit_routes", lambda *a: pytest.fail("implicit commit"))
    assert owner.main() == 2
    result = json.loads(capsys.readouterr().out)
    assert result["observation_only"] and not result["route_mutation_performed"]
    assert result["certified_mode"] == "sqlite_primary_pending"
    assert (root / "data/sql_link_shards").resolve() == source / "sql_link_shards"


@pytest.mark.parametrize("flags", [[], ["--verify-only", "--apply"]])
def test_failback_cli_requires_explicit_handoff_mode(cohort, monkeypatch, flags):
    from scripts.ops import storage_failback_sync as owner

    _, _, _, receipt = cohort
    monkeypatch.setattr(
        sys,
        "argv",
        ["storage_failback_sync", "--sqlite-primary-receipt", str(receipt), *flags],
    )
    monkeypatch.setattr(
        owner,
        "_acquire_singleton_lock",
        lambda *a: pytest.fail("lock before argument validation"),
    )
    with pytest.raises(SystemExit) as result:
        owner.main()
    assert result.value.code == 2


def test_inventory_scan_failure_cannot_be_treated_as_complete(cohort, monkeypatch):
    root, _, _, receipt = cohort

    def walk(path, *, followlinks, onerror):
        onerror(PermissionError("inaccessible shard directory"))
        return iter(())

    monkeypatch.setattr(src.os, "walk", walk)
    with pytest.raises(PermissionError, match="inaccessible"):
        src.commit_routes(root, receipt)


def test_verified_handoff_preserves_sources_and_controls(cohort):
    root, source, target, receipt = cohort
    before = (source / src.DATABASES[0]).read_bytes()
    result = src.commit_routes(root, receipt)
    assert result["ok"] and result["route_mutation_performed"]
    assert result["mode"] == src.PROFILE
    assert not result["source_retired"]
    assert (source / src.DATABASES[0]).read_bytes() == before
    assert (root / "data/sql_link_shards").resolve() == target / "data/sql_link_shards"
    assert not (root / "governance").is_symlink()


@pytest.mark.parametrize(
    "damage",
    ["target", "source", "missing_row", "extra_target", "alias", "wal", "no_integrity"],
)
def test_incomplete_or_changed_custody_cannot_relink(cohort, damage):
    root, source, target, receipt = cohort
    before = {r: os.readlink(root / r) for r in src.LINKS}
    if damage == "target":
        with (target / "data/jsonl_link.sqlite3").open("ab") as stream:
            stream.write(b"changed")
    elif damage == "source":
        with (source / "jsonl_link.sqlite3").open("ab") as stream:
            stream.write(b"changed")
    elif damage == "extra_target":
        (target / "data/sql_link_shards/unreviewed.db").write_bytes(b"unknown")
    elif damage == "alias":
        (source / "sql_link_shards/alias").symlink_to(target)
    elif damage == "wal":
        (source / "jsonl_link.sqlite3-wal").write_bytes(b"pending")
    else:
        payload = json.loads(receipt.read_text())
        if damage == "missing_row":
            payload["files"].pop()
        else:
            payload["files"][0]["quick_check"] = "unknown"
        receipt.write_text(json.dumps(payload))
    with pytest.raises((RuntimeError, ValueError)):
        src.commit_routes(root, receipt)
    assert before == {r: os.readlink(root / r) for r in src.LINKS}


def test_interrupted_route_commit_rolls_back(cohort, monkeypatch):
    root, _, _, receipt = cohort
    before = {r: os.readlink(root / r) for r in src.LINKS}
    real = src.os.replace
    calls = []

    def replace(old, new):
        calls.append(str(new))
        if len(calls) == 3:
            raise OSError("injected publication failure")
        real(old, new)

    monkeypatch.setattr(src.os, "replace", replace)
    with pytest.raises(OSError, match="injected"):
        src.commit_routes(root, receipt)
    assert before == {r: os.readlink(root / r) for r in src.LINKS}


def test_busy_or_unknown_handles_block(cohort, monkeypatch):
    root, _, _, receipt = cohort
    monkeypatch.setattr(
        src.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=1, stdout=b"", stderr=b"permission denied"
        ),
    )
    with pytest.raises(RuntimeError, match="handles_present_or_unknown"):
        src.commit_routes(root, receipt)


def test_receipt_cannot_replace_an_unrelated_active_source(cohort):
    root, _, _, receipt = cohort
    link = root / "data/jsonl_link.sqlite3"
    link.unlink()
    link.symlink_to(root / "another_active_database.sqlite3")
    with pytest.raises(ValueError, match="source_route_mismatch"):
        src.commit_routes(root, receipt)
    assert os.readlink(link) == str(root / "another_active_database.sqlite3")


def test_router_never_implicitly_adopts_or_falls_back(cohort, monkeypatch):
    from core import storage_router

    root, _, _, receipt = cohort
    with pytest.raises(RuntimeError, match="handoff_required"):
        storage_router.route_runtime_storage(root, allow_autosync=True)
    src.commit_routes(root, receipt)
    assert (
        storage_router.route_runtime_storage(root, allow_autosync=True).mode
        == src.PROFILE
    )
    monkeypatch.setattr(
        src,
        "_validated_target",
        lambda: (_ for _ in ()).throw(ValueError("mount missing")),
    )
    with pytest.raises(RuntimeError, match="mount missing"):
        storage_router.route_runtime_storage(root)
    assert (root / "data/sql_link_shards").is_symlink()


def test_sqlite_connection_rejects_standby_write_but_preserves_read_only_target(cohort):
    from core.sqlite_runtime import connect_sqlite

    root, source, _, receipt = cohort
    src.commit_routes(root, receipt)
    with pytest.raises(RuntimeError, match="standby_write_forbidden"):
        connect_sqlite(source / "jsonl_link.sqlite3", project_root=root)
    with connect_sqlite(
        source / "jsonl_link.sqlite3", project_root=root, readonly=True
    ) as connection:
        assert connection.execute("SELECT value FROM evidence").fetchone() == (42,)


def test_missing_drive_blocks_connection_before_mkdir(cohort, monkeypatch):
    from core.sqlite_runtime import connect_sqlite

    root, _, _, _ = cohort
    monkeypatch.setattr(
        src,
        "_validated_target",
        lambda: (_ for _ in ()).throw(ValueError("mount missing")),
    )
    with pytest.raises(RuntimeError, match="mount missing"):
        connect_sqlite(root / "data/sql_link_shards/new.sqlite3", project_root=root)
    assert not (
        root / "local_fallback_storage/data/sql_link_shards/new.sqlite3"
    ).exists()


def test_execution_controls_ignore_external_preference(cohort, monkeypatch):
    from core.execution_lane_pipeline import execution_lane_root

    root, _, target, _ = cohort
    monkeypatch.setenv("BOT_LOGS_PREFER_EXTERNAL", "1")
    monkeypatch.setenv("EXECUTION_LANE_ROOT", str(target / "governance"))
    assert execution_lane_root(root) == root / "governance/execution_lanes"


def test_external_control_alias_blocks_handoff(cohort):
    root, _, target, receipt = cohort
    (root / "governance/execution_lanes").symlink_to(target)
    with pytest.raises(RuntimeError, match="control_route"):
        src.commit_routes(root, receipt)


def test_stateful_repair_cannot_mutate_primary_profile(cohort, monkeypatch):
    from scripts.ops import stateful_storage_regression_guard as guard

    root, _, _, _ = cohort
    monkeypatch.setattr(
        guard, "_repair_stateful_path", lambda **k: pytest.fail("legacy repair invoked")
    )
    result = guard.build_payload(root, apply=True)
    assert not result["route_mutation_performed"]
    assert result["overall_status"] == "blocked"


def test_standby_pruners_reject_legacy_proof(cohort):
    from scripts.ops import local_sql_shard_standby_prune, storage_standby_prune

    root, source, _, _ = cohort
    for owner in (local_sql_shard_standby_prune, storage_standby_prune):
        result = owner.build_payload(root, apply=True)
        assert result["overall_status"] == "blocked"
        assert result["source_files_removed"] == 0
    assert (source / "sql_link_shards/trading.sqlite3").exists()


def test_maintenance_and_off_are_mandatory(tmp_path, monkeypatch):
    (tmp_path / "governance/runtime").mkdir(parents=True)
    monkeypatch.delenv("RUNTIME_MAINTENANCE_HOLD_PATH", raising=False)
    monkeypatch.delenv("SQL_LINK_SERVICE_MAINTENANCE_HOLD_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="maintenance_required"):
        src._hold(tmp_path)


def test_live_switch_prevents_commit_even_with_maintenance_token(tmp_path, monkeypatch):
    from core import runtime_maintenance

    (tmp_path / "governance/runtime").mkdir(parents=True)
    monkeypatch.delenv("RUNTIME_MAINTENANCE_HOLD_PATH", raising=False)
    hold = runtime_maintenance.engage_maintenance_hold(tmp_path, reason="test")
    monkeypatch.setenv("SQL_LINK_SERVICE_MAINTENANCE_HOLD_TOKEN", hold["token"])
    switch = tmp_path / "governance/runtime/live_execution_switch_state.json"
    switch.write_text('{"requested_on": true}')
    with pytest.raises(RuntimeError, match="trading_off_required"):
        src._hold(tmp_path)
    switch.write_text('{"requested_on": false}')
    src._hold(tmp_path)


def test_late_wal_creation_blocks_commit(cohort, monkeypatch):
    root, source, _, receipt = cohort
    real = src._digest

    def digest(path, deadline):
        result = real(path, deadline)
        (source / "jsonl_link.sqlite3-wal").write_bytes(b"new write")
        return result

    monkeypatch.setattr(src, "_digest", digest)
    with pytest.raises(ValueError, match="journal"):
        src.commit_routes(root, receipt)
    assert (root / "data/sql_link_shards").resolve() == source / "sql_link_shards"


@pytest.mark.parametrize(
    "change", ["source_inventory", "target_inventory", "verified_target"]
)
def test_late_namespace_or_verified_target_changes_block_commit(
    cohort, monkeypatch, change
):
    root, source, target, receipt = cohort
    real = src._digest

    def digest(path, deadline):
        result = real(path, deadline)
        if path.name == "trading.sqlite3":
            if change == "verified_target":
                with (target / "data/jsonl_link.sqlite3").open("ab") as stream:
                    stream.write(b"late change")
            else:
                base = source if change == "source_inventory" else target / "data"
                (base / "sql_link_shards/new.sqlite3").write_bytes(b"new")
        return result

    monkeypatch.setattr(src, "_digest", digest)
    with pytest.raises(ValueError, match="changed_before_commit"):
        src.commit_routes(root, receipt)
    assert (root / "data/sql_link_shards").resolve() == source / "sql_link_shards"


def test_writer_admission_adds_route_debt_without_relaxing_reserve(cohort, monkeypatch):
    from scripts.ops.sql_writer_admission import storage_admission

    root, _, _, _ = cohort
    monkeypatch.setenv("SQL_LINK_SERVICE_FORCE_LOCAL_FALLBACK", "1")
    result = storage_admission(root)
    assert not result["writer_start_allowed"]
    assert "sqlite_primary_handoff_required" in result["blockers"]
    assert "sqlite_primary_conflicting_force_local_setting" in result["blockers"]


def test_queue_constructor_refuses_stale_standby(cohort):
    from core.channel_queue import ChannelQueue

    root, source, _, _ = cohort
    with pytest.raises(RuntimeError, match="standby_write_forbidden"):
        ChannelQueue(source / "bot_channel_queue.sqlite3", project_root=root)


def test_unknown_profile_is_not_silent_legacy_fallback(monkeypatch):
    monkeypatch.setenv("BOT_STORAGE_ROUTE_PROFILE", "sqlite_priamry")
    with pytest.raises(ValueError, match="unsupported"):
        src.enabled()


def test_watchdog_parks_unavailable_primary_without_readiness_credit(
    cohort, monkeypatch
):
    from scripts.ops import process_watchdog as watchdog

    root, _, _, _ = cohort
    monkeypatch.setattr(watchdog, "PROJECT_ROOT", root)
    monkeypatch.setattr(watchdog, "OPERATOR_STOP_FLAG", root / "operator.flag")
    monkeypatch.setattr(watchdog, "GLOBAL_HALT_FLAG", root / "halt.flag")
    monkeypatch.delenv("SQL_LINK_SERVICE_PAUSED_FOR_LOCAL_STORAGE", raising=False)
    state = watchdog._safety_pause_state()
    assert state["active"] and state["sqlite_primary_route_unavailable"]
    hold = watchdog._sql_writer_restart_hold(state)
    assert hold["active"] and not hold["writer_ready"]
    assert hold["reason"] == "sqlite_primary_route_unavailable"


def test_generic_switch_cannot_stop_or_relink_primary_profile(cohort, monkeypatch):
    from scripts.ops import storage_switch_orchestrator as owner

    root, _, _, _ = cohort
    monkeypatch.setattr(
        owner,
        "engage_maintenance_hold",
        lambda *a, **k: pytest.fail("legacy stop/hold invoked"),
    )
    result = owner.build_payload(
        root, target_mode="external", restart=True, eject=False
    )
    assert result["overall_status"] == "blocked"
    assert not result["route_mutation_performed"]


def test_disaster_owner_cannot_rewrite_primary_target(cohort, monkeypatch):
    from scripts.ops import storage_disaster_recovery as owner

    root, _, _, _ = cohort
    monkeypatch.setattr(
        owner, "_probe_storage", lambda: pytest.fail("legacy probe invoked")
    )
    result, _ = owner.build_payload(
        root,
        apply=True,
        recovery_root=root / "recovery",
        state_path=root / "state.json",
        mount_cooldown_seconds=30,
        snapshot_cooldown_seconds=30,
    )
    assert result["overall_status"] == "blocked"
    assert not result["route_mutation_performed"]


@pytest.mark.parametrize(
    "mount",
    [
        "/Volumes/VIDEO",
        "/Volumes/video",
        "/Volumes/..",
        "/tmp/Fake",
        "/Volumes/SSD/subdir",
    ],
)
def test_invalid_config_rejected_without_inspection(monkeypatch, mount):
    monkeypatch.setenv("BOT_LOGS_EXTERNAL_MOUNT", mount)
    monkeypatch.delenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", raising=False)
    monkeypatch.setattr(
        src, "_physical", lambda _: pytest.fail("protected path inspected")
    )
    with pytest.raises(ValueError):
        src._validated_target()


@pytest.mark.parametrize(
    "failure",
    [
        "uuid",
        "mount",
        "internal",
        "readonly",
        "format",
        "unmounted",
        "reserve",
        "higher_reserve",
        "none",
    ],
)
def test_actual_identity_gate_fails_closed(tmp_path, monkeypatch, failure):
    mount = tmp_path / "SSD"
    target = mount / "platform"
    target.mkdir(parents=True)
    monkeypatch.setattr(src, "configured_target", lambda: target)
    monkeypatch.setenv("BOT_LOGS_EXTERNAL_VOLUME_UUID", UUID)
    monkeypatch.delenv("BOT_LOGS_EXTERNAL_MIN_FREE_BYTES", raising=False)
    monkeypatch.setenv(
        "BOT_LOGS_EXTERNAL_MIN_FREE_GB",
        "1000" if failure == "higher_reserve" else "100",
    )
    monkeypatch.setattr(src.os.path, "ismount", lambda p: failure != "unmounted")
    info = {
        "VolumeUUID": UUID,
        "MountPoint": str(mount),
        "FilesystemType": "apfs",
        "Internal": False,
        "WritableVolume": True,
    }
    changes = {
        "uuid": ("VolumeUUID", "wrong"),
        "mount": ("MountPoint", "/elsewhere"),
        "internal": ("Internal", True),
        "readonly": ("WritableVolume", False),
        "format": ("FilesystemType", "exfat"),
    }
    if failure in changes:
        key, value = changes[failure]
        info[key] = value
    monkeypatch.setattr(
        src.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout=plistlib.dumps(info)),
    )
    monkeypatch.setattr(
        src.shutil,
        "disk_usage",
        lambda p: SimpleNamespace(
            free=(100 if failure == "reserve" else 900) * 1024**3
        ),
    )
    if failure == "none":
        assert src._validated_target() == target
    else:
        with pytest.raises(ValueError):
            src._validated_target()
