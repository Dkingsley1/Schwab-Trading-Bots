import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from core import accountability as writer
from core import write_path_recovery as recovery
from scripts.ops import data_plane_recovery_controller as controller


@pytest.fixture(autouse=True)
def reset_cache():
    recovery._REQUEST_CACHE.clear()
    recovery._LAST_RECEIPT.clear()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def domain(root, *, now=None, records=None, kind="jsonl"):
    now = now or datetime.now(timezone.utc)
    target = str(root / f"governance/events/owned.{kind}")
    result = {
        "id": recovery.domain_id("owner", target, "20260916"),
        "source": "owner",
        "target_path": target,
        "day": "20260916",
        "count": 1,
        "generation": "generation-1",
        "latest_failure_utc": (now - timedelta(seconds=10)).isoformat(),
        "failed_records": [],
        "record_checkpoint_complete": False,
    }
    if records:
        result.update(recovery.failed_records(records))
    return result


def history(*domains, **extra):
    return {"complete": True, "domains": list(domains), **extra}


def receipt(root, item, *, observed, rows=None, kind="jsonl"):
    rows = rows or [{"message_id": "one", "timestamp_utc": observed.isoformat()}]
    data = (
        "".join(json.dumps(r) + "\n" for r in rows)
        if kind == "jsonl"
        else json.dumps(rows[0])
    ).encode()
    path = Path(item["target_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    info = path.stat()
    import hashlib

    proof = {
        "timestamp_utc": observed.isoformat(),
        "source": item["source"],
        "target_path": item["target_path"],
        "generation": item["generation"],
        "device": info.st_dev,
        "inode": info.st_ino,
        "offset": 0,
        "length": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "fsync_readback": True,
        "kind": kind,
    }
    proof_path = root / recovery.RECEIPTS / f"{item['id']}.json"
    write_json(proof_path, proof)
    return proof_path, proof


def test_observation_only_has_no_mutation(tmp_path):
    result = recovery.recovery_pass(tmp_path, history(domain(tmp_path)), apply=False)
    assert result["authority"] == "observation_only"
    assert list(tmp_path.iterdir()) == []


def test_actual_owner_write_gets_fsync_and_readback_proof(tmp_path, monkeypatch):
    item = domain(tmp_path)
    recovery.recovery_pass(tmp_path, history(item), apply=True)
    synced = []
    real_fsync = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (synced.append(fd), real_fsync(fd))[-1])
    assert (
        writer.safe_append_jsonl_batch(
            item["target_path"],
            [
                {
                    "message_id": "a",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
            ],
            project_root=str(tmp_path),
            source="owner",
        )
        == 1
    )
    result = recovery.verify_receipt(tmp_path, item, datetime.now(timezone.utc))
    assert result["verified"]
    assert not result["checkpoint_reconciled"]
    assert len(synced) >= 4


def test_unrequested_writes_do_not_publish_proof(tmp_path):
    item = domain(tmp_path)
    assert writer.safe_write_json_atomic(
        item["target_path"], {"value": 1}, project_root=str(tmp_path), source="owner"
    )
    assert not (tmp_path / recovery.RECEIPTS).exists()


def test_atomic_owner_write_is_verified_without_claiming_old_payload_recovery(tmp_path):
    item = domain(tmp_path, kind="json")
    recovery.recovery_pass(tmp_path, history(item), apply=True)
    assert writer.safe_write_json_atomic(
        item["target_path"], {"value": 1}, project_root=str(tmp_path), source="owner"
    )
    result = recovery.verify_receipt(tmp_path, item, datetime.now(timezone.utc))
    assert result["verified"] and not result["checkpoint_reconciled"]


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("source", "other", "owner_or_generation_mismatch"),
        ("generation", "old", "owner_or_generation_mismatch"),
        ("fsync_readback", False, "owner_or_generation_mismatch"),
        ("inode", -1, "source_generation_changed"),
        ("offset", -1, "invalid_checkpoint_bounds"),
        ("length", recovery.MAX_BYTES + 1, "invalid_checkpoint_bounds"),
        ("sha256", "0" * 64, "checkpoint_readback_mismatch"),
    ],
)
def test_invalid_proof_cannot_release(tmp_path, field, value, reason):
    now = datetime.now(timezone.utc)
    item = domain(tmp_path, now=now)
    path, proof = receipt(tmp_path, item, observed=now)
    proof[field] = value
    write_json(path, proof)
    assert recovery.verify_receipt(tmp_path, item, now) == {
        "verified": False,
        "reason": reason,
    }


@pytest.mark.parametrize("delta", [-3600, 1])
def test_stale_or_future_proof_cannot_release(tmp_path, delta):
    now = datetime.now(timezone.utc)
    item = domain(tmp_path, now=now - timedelta(hours=2))
    receipt(tmp_path, item, observed=now + timedelta(seconds=delta))
    assert not recovery.verify_receipt(tmp_path, item, now)["verified"]


def test_two_independent_observations_required_for_probation(tmp_path):
    now = datetime.now(timezone.utc)
    row = {"message_id": "failed-operation", "value": 17}
    item = domain(tmp_path, now=now, records=[row])
    receipt(tmp_path, item, observed=now, rows=[row])
    first = recovery.recovery_pass(tmp_path, history(item), apply=True, now=now)
    assert first["state"]["domains"][0]["phase"] == "probation"
    repeat = recovery.recovery_pass(
        tmp_path, history(item), apply=True, now=now + timedelta(seconds=60)
    )
    assert repeat["state"]["domains"][0]["phase"] == "probation"
    receipt(tmp_path, item, observed=now + timedelta(seconds=120), rows=[row])
    final = recovery.recovery_pass(
        tmp_path, history(item), apply=True, now=now + timedelta(seconds=120)
    )
    assert final["state"]["domains"][0]["phase"] == "path_recovered"
    assert final["state"]["unreconciled_failure_count"] == 0
    assert final["state"]["automatic_replay_allowed"] is False


def test_same_id_with_different_payload_does_not_reconcile(tmp_path):
    now = datetime.now(timezone.utc)
    item = domain(tmp_path, now=now, records=[{"message_id": "one", "value": 1}])
    receipt(tmp_path, item, observed=now, rows=[{"message_id": "one", "value": 2}])
    assert not recovery.verify_receipt(tmp_path, item, now)["checkpoint_reconciled"]


def test_duplicate_record_does_not_earn_exact_once_reconciliation(tmp_path):
    now = datetime.now(timezone.utc)
    row = {"message_id": "one", "value": 1}
    item = domain(tmp_path, now=now, records=[row])
    receipt(tmp_path, item, observed=now, rows=[row, row])
    result = recovery.verify_receipt(tmp_path, item, now)
    assert result["duplicate_or_conflicting_ids"]
    assert not result["checkpoint_reconciled"]


def test_locked_queue_schema_never_reports_ready(tmp_path, monkeypatch):
    import sqlite3
    from core.channel_queue import ChannelQueue

    path = tmp_path / "queue.sqlite3"
    path.write_bytes(b"present")
    queue = ChannelQueue.__new__(ChannelQueue)
    queue.db_path = path

    def locked(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(sqlite3, "connect", locked)
    with pytest.raises(sqlite3.OperationalError, match="schema_unverified_locked"):
        queue._schema_ready()


def test_new_failure_resets_generation_and_probation(tmp_path):
    now = datetime.now(timezone.utc)
    item = domain(tmp_path, now=now)
    receipt(tmp_path, item, observed=now)
    recovery.recovery_pass(tmp_path, history(item), apply=True, now=now)
    item.update(generation="new", count=2)
    result = recovery.recovery_pass(
        tmp_path, history(item), apply=True, now=now + timedelta(seconds=61)
    )
    saved = result["state"]["domains"][0]
    assert saved["phase"] == "awaiting_owner_write"
    assert "probation_started_utc" not in saved
    assert result["state"]["unreconciled_failure_count"] == 2


def test_rollover_retains_old_unreconciled_debt(tmp_path):
    item = domain(tmp_path)
    recovery.recovery_pass(
        tmp_path, history(item, unmapped_prior_failure_count=10), apply=True
    )
    result = recovery.recovery_pass(tmp_path, history(), apply=True)
    assert result["state"]["unreconciled_failure_count"] == 11
    assert (
        result["state"]["count_basis"]
        == "conservative_includes_unmapped_migration_debt"
    )


def test_incomplete_census_preserves_state(tmp_path):
    item = domain(tmp_path)
    recovery.recovery_pass(tmp_path, history(item), apply=True)
    before = (tmp_path / recovery.STATE).read_bytes()
    result = recovery.recovery_pass(tmp_path, history(complete=False), apply=True)
    assert result["overall_status"] == "blocked"
    assert (tmp_path / recovery.STATE).read_bytes() == before


def test_retry_budget_backoff_and_escalation(tmp_path):
    now = datetime.now(timezone.utc)
    item = domain(tmp_path, now=now)
    result = recovery.recovery_pass(tmp_path, history(item), apply=True, now=now)
    assert result["state"]["domains"][0]["attempts"] == 1
    assert (
        recovery.recovery_pass(tmp_path, history(item), apply=True, now=now)[
            "checked_count"
        ]
        == 0
    )
    for _ in range(5):
        now = recovery.timestamp(result["state"]["domains"][0]["next_check_utc"])
        result = recovery.recovery_pass(tmp_path, history(item), apply=True, now=now)
    assert result["state"]["escalated_count"] == 1
    assert result["state"]["requests"] == []


def test_rejected_target_is_not_opened(tmp_path, monkeypatch):
    item = domain(tmp_path)
    item["target_path"] = str(tmp_path / "rejected")
    real_inspect = recovery.inspect_storage_path

    def inspect(path, **kwargs):
        if str(path) == item["target_path"]:
            return {"status": "blocked_external", "resolved_path": str(path)}
        return real_inspect(path, **kwargs)

    monkeypatch.setattr(recovery, "inspect_storage_path", inspect)
    result = recovery.recovery_pass(tmp_path, history(item), apply=True)
    assert (
        result["state"]["domains"][0]["verification"]["reason"]
        == "route_rejected:blocked_external"
    )
    assert not Path(item["target_path"]).exists()


def test_lock_contention_is_deferred(tmp_path):
    import fcntl

    path = tmp_path / "governance/health/write_path_recovery.lock"
    path.parent.mkdir(parents=True)
    with path.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert (
            recovery.recovery_pass(tmp_path, history(), apply=True)["reason"]
            == "recovery_owner_busy"
        )


def test_failed_append_retains_exact_record_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(writer, "_write_lines", lambda *_: False)
    item = domain(tmp_path)
    assert (
        writer.safe_append_jsonl_batch(
            item["target_path"],
            [{"message_id": "known-id", "value": 9}],
            project_root=str(tmp_path),
            source="owner",
        )
        == 0
    )
    result = controller._write_failure_history(tmp_path, {})
    assert result["complete"]
    assert result["domains"][0]["record_checkpoint_complete"]
    assert result["domains"][0]["failed_records"][0]["message_id"] == "known-id"


def test_record_budget_is_explicit_not_partial_success():
    result = recovery.failed_records([{"message_id": str(i)} for i in range(129)])
    assert not result["record_checkpoint_complete"]
    assert result["failed_records"] == []


def test_native_schedule_observes_before_early_holds():
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts/ops/run_sql_link_writer_launchd.sh").read_text()
    assert script.index("-m scripts.ops.sql_writer_admission") < script.index(
        "runtime_maintenance_hold.py"
    )
    assert 'data_plane_recovery_controller.py" --apply --json' in script
