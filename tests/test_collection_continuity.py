import gzip
import json
import sqlite3
from pathlib import Path

import pytest

from core import collection_continuity as src


@pytest.fixture
def storage(tmp_path, monkeypatch):
    root, target = tmp_path / "project", tmp_path / "external/platform"
    root.mkdir()
    target.mkdir(parents=True)
    monkeypatch.setattr(src.primary, "enabled", lambda _: True)
    monkeypatch.setattr(src.primary, "require_ready", lambda _: target)
    monkeypatch.setattr(src, "_free_bytes", lambda _: 500 * 1024**3)
    return root, target


def buffer(root, value=1):
    return src.preserve(
        root,
        channel="decision",
        source_path=str(root / "decisions/test.jsonl"),
        payloads=[{"message_id": f"m-{value}", "action": "BUY", "value": value}],
    )


def pending(root):
    with sqlite3.connect(root / src.RELATIVE_DB) as conn:
        return conn.execute("SELECT COUNT(*) FROM pending").fetchone()[0]


def test_reconnect_archives_verified_evidence_without_replaying_decisions(storage):
    root, target = storage
    first = buffer(root)
    assert buffer(root) == first
    assert pending(root) == 1
    result = src.reconcile(root)
    assert result["ok"] and result["pending_batches"] == 0
    archive = target / f"cold_archive/collection_buffer/{first}.json.gz"
    body = json.loads(gzip.decompress(archive.read_bytes()))
    assert body["payloads"][0]["action"] == "BUY"
    assert body["queue_replay_authorized"] is False
    assert not result["canonical_ingestion_verified"]
    assert not (target / "data/bot_channel_queue.sqlite3").exists()
    assert src.reconcile(root)["archived_batches"] == 0
    buffer(root)
    assert pending(root) == 1
    assert src.reconcile(root)["archived_batches"] == 1
    assert len(list(target.rglob("*.gz"))) == 1
    assert pending(root) == 0


@pytest.mark.parametrize(
    "reason", ["mount_missing", "wrong_uuid", "destination_reserve", "route_mismatch"]
)
def test_unavailable_or_wrong_device_preserves_local_data(storage, monkeypatch, reason):
    root, target = storage
    buffer(root)

    def reject(_):
        raise RuntimeError(reason)

    monkeypatch.setattr(src.primary, "require_ready", reject)
    assert not src.reconcile(root)["ok"]
    assert pending(root) == 1
    assert not list(target.rglob("*.gz"))


def test_disconnect_after_copy_preserves_local_then_idempotently_recovers(
    storage, monkeypatch
):
    root, target = storage
    buffer(root)
    publish = src._publish

    def disconnect(*args):
        result = publish(*args)
        monkeypatch.setattr(
            src.primary,
            "require_ready",
            lambda _: (_ for _ in ()).throw(RuntimeError("unplugged")),
        )
        return result

    monkeypatch.setattr(src, "_publish", disconnect)
    assert not src.reconcile(root)["ok"]
    assert pending(root) == 1
    monkeypatch.setattr(src.primary, "require_ready", lambda _: target)
    monkeypatch.setattr(src, "_publish", publish)
    assert src.reconcile(root)["archived_batches"] == 1
    assert len(list(target.rglob("*.gz"))) == 1


def test_conflicting_archive_keeps_both_copies(storage):
    root, target = storage
    batch = buffer(root)
    archive = target / f"cold_archive/collection_buffer/{batch}.json.gz"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(gzip.compress(b"different"))
    assert not src.reconcile(root)["ok"]
    assert pending(root) == 1
    assert gzip.decompress(archive.read_bytes()) == b"different"


def test_archived_parent_alias_cannot_escape_target(storage):
    root, target = storage
    buffer(root)
    outside = root / "untouched"
    outside.mkdir()
    (target / "cold_archive").symlink_to(outside)
    assert not src.reconcile(root)["ok"]
    assert pending(root) == 1
    assert list(outside.iterdir()) == []


def test_collection_start_refuses_requested_live_authority(storage, monkeypatch):
    from core import live_execution_switch

    root, _ = storage
    monkeypatch.setattr(
        live_execution_switch,
        "switch_status",
        lambda _: {"ok": True, "switch": "ON", "requested_on": True},
    )
    with pytest.raises(RuntimeError, match="execution_off"):
        src.collection_start_allowed(root)


def test_pending_capacity_reserve_and_alias_are_not_bypassed(storage, monkeypatch):
    root, target = storage
    monkeypatch.setattr(src, "_free_bytes", lambda _: 124 * 1024**3)
    with pytest.raises(ValueError, match="internal_reserve"):
        buffer(root)
    monkeypatch.setattr(src, "_free_bytes", lambda _: 500 * 1024**3)
    path = root / src.RELATIVE_DB
    path.parent.mkdir(parents=True)
    path.symlink_to(target / "foreign.sqlite3")
    with pytest.raises(ValueError):
        buffer(root)
    assert not (target / "foreign.sqlite3").exists()


def test_bounded_batch_pass_leaves_pending_visible(storage):
    root, _ = storage
    buffer(root, 1)
    buffer(root, 2)
    result = src.reconcile(root, max_batches=1)
    assert result["overall_status"] == "catching_up"
    assert result["pending_batches"] == 1
    assert src.reconcile(root)["pending_batches"] == 0
    with sqlite3.connect(root / src.RELATIVE_DB) as conn:
        assert conn.execute("SELECT pending_bytes FROM capacity").fetchone()[0] == 0


def test_internal_outbox_alias_cannot_modify_an_unrelated_database(storage):
    root, _ = storage
    foreign = root / "unrelated.sqlite3"
    with sqlite3.connect(foreign) as conn:
        conn.execute("CREATE TABLE protected(value)")
    before = foreign.read_bytes()
    path = root / src.RELATIVE_DB
    path.parent.mkdir(parents=True)
    path.symlink_to(foreign)
    with pytest.raises(ValueError, match="physical_internal_path"):
        buffer(root)
    assert foreign.read_bytes() == before


def test_concurrent_drain_is_deferred(storage):
    import fcntl

    root, _ = storage
    buffer(root)
    with (root / src.RELATIVE_DB).with_name("drain.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert src.reconcile(root)["reason"] == "collection_buffer_drain_busy"
    assert pending(root) == 1


@pytest.mark.parametrize("reserve", ["NaN", "inf", "0", "-1"])
def test_invalid_reserve_never_admits_buffering(storage, monkeypatch, reserve):
    root, _ = storage
    monkeypatch.setenv("BOT_LOCAL_STORAGE_TARGET_FREE_GB", reserve)
    with pytest.raises(ValueError, match="invalid_reserve"):
        buffer(root)


def test_outbox_refuses_order_submission_channels(storage):
    root, _ = storage
    with pytest.raises(ValueError, match="scope_rejected"):
        src.preserve(
            root,
            channel="execution_order",
            source_path=str(root / "orders.jsonl"),
            payloads=[{"message_id": "x"}],
        )


def test_queue_outage_uses_outbox_not_failure_or_action_replay(storage, monkeypatch):
    from core import accountability, channel_queue

    root, _ = storage
    monkeypatch.setattr(channel_queue, "queue_enabled", lambda: True)
    monkeypatch.setattr(
        channel_queue,
        "default_queue_db_path",
        lambda _: str(root / "data/bot_channel_queue.sqlite3"),
    )
    monkeypatch.setattr(
        channel_queue,
        "ChannelQueue",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("missing_ssd")),
    )
    monkeypatch.setattr(src.primary, "observe", lambda _: {"ok": False})
    failures = []
    monkeypatch.setattr(
        accountability,
        "_emit_write_failure_event",
        lambda **kwargs: failures.append(kwargs),
    )
    accountability._queue_publish(
        project_root=str(root),
        channel="decision",
        source_path=str(root / "decisions/test.jsonl"),
        payloads=[{"message_id": "m-1", "action": "BUY"}],
    )
    assert pending(root) == 1
    assert not failures


def test_collection_start_never_opens_standby_databases(storage, monkeypatch):
    from core import storage_router

    root, _ = storage
    monkeypatch.setattr(
        src.primary,
        "require_ready",
        lambda _: (_ for _ in ()).throw(RuntimeError("unplugged")),
    )
    with pytest.raises(RuntimeError, match="unplugged"):
        storage_router.route_runtime_storage(root)
    route = storage_router.route_runtime_storage(root, collection_only=True)
    assert route.mode == "sqlite_primary_collection_buffer"
    assert route.switched_links == ()
    assert not (root / "local_fallback_storage/data").exists()


def test_selected_profile_replaces_only_legacy_repair_step(storage):
    from scripts.ops import readiness_evidence_refresh as refresh

    root, _ = storage
    before = refresh.profile_steps("accrual")
    after = refresh._storage_profile_steps(root, before)
    changed = [row for row, old in zip(after, before) if row != old]
    assert len(changed) == 1
    assert changed[0]["name"] == "storage_fallback_repair"
    assert changed[0]["args"] == ["--drain-collection-buffer", "--apply", "--json"]
    assert (
        "--repair-local-fallback-aliases"
        in next(row for row in before if row["name"] == "storage_fallback_repair")[
            "args"
        ]
    )
