"""Synthetic safety tests; never open the real inventory's source files."""

from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

STAGED = Path(__file__).with_name("raw_inventory_cleanup.py")
if STAGED.exists():
    spec = importlib.util.spec_from_file_location(
        "raw_inventory_cleanup_staged", STAGED
    )
    src = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(src)
else:
    from scripts.ops import raw_inventory_cleanup as src


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    root = tmp_path / "sources"
    root.mkdir()
    monkeypatch.setattr(src, "SOURCE_ROOT", root)
    monkeypatch.setattr(src.safety, "idle", lambda path: None)
    return root


def make_row(
    root,
    relative="governance/channels/decision/profile/decision_20260101.jsonl",
    content=b"",
):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    stamp = datetime(2026, 1, 2, tzinfo=timezone.utc).timestamp()
    os.utime(path, (stamp, stamp))
    info = path.stat()
    return {
        "path": str(path),
        "device": info.st_dev,
        "inode": info.st_ino,
        "current_bytes": len(content),
        "mtime_utc": datetime.fromtimestamp(info.st_mtime, timezone.utc).isoformat(),
        "full_sha256": hashlib.sha256(content).hexdigest(),
        "inventory_blockers": ["empty_source"] if not content else [],
        "status": "present",
        "stable_during_read": True,
    }


def authorize(monkeypatch, rows, groups=None):
    monkeypatch.setattr(
        src,
        "read_audit",
        lambda path: (
            {r["path"]: r for r in rows},
            [r for r in rows if r["current_bytes"] == 0],
            groups or [],
        ),
    )


def retire(root, row):
    return src.retire_empty(
        root, row, src.native.Budget(), lambda: None, audit="synthetic"
    )


def test_retire_has_durable_tombstone_before_unlink(sandbox, monkeypatch):
    row = make_row(sandbox)
    authorize(monkeypatch, [row])
    path = Path(row["path"])
    calls = []
    original = src.native.receipt

    def receipt(root, payload):
        calls.append((payload["phase"], path.exists()))
        original(root, payload)

    monkeypatch.setattr(src.native, "receipt", receipt)
    result = retire(sandbox, row)
    assert result["source_removed"] and result["persistence_complete"]
    assert calls == [("empty_retirement_prepared", True), ("empty_retired", False)]
    records = [
        json.loads(line)
        for line in (
            sandbox / "governance/storage_recovery/verified_duplicate_cleanup.jsonl"
        )
        .read_text()
        .splitlines()
    ]
    assert records[0]["restore"]["content_hex"] == ""
    assert records[0]["audit_sha256"] == src.REVIEWED_AUDIT_SHA256
    assert records[0]["proof"]["identity"][:3] == [row["device"], row["inode"], 0]
    assert not path.exists()


@pytest.mark.parametrize(
    "relative,reason",
    [
        ("data/deep_cold/deep_cold_manifest.jsonl", "custody_manifest"),
        (
            "exports/independent_fill_inbox/market_replay_fills_current.jsonl",
            "active_inbox",
        ),
        ("cold_archive/governance/events/event_20260101.jsonl", "retention_locked"),
        ("data/stale_stage/governance/decision_20260101.jsonl", "retention_locked"),
        ("quarantine/decision_20260101.jsonl", "retention_locked"),
        (
            "governance/execution_lanes/execution_intents_20260101.jsonl",
            "execution_or_financial",
        ),
        (
            "exports/trade_logs/independent_fills/paper_trades_20260101.jsonl",
            "execution_or_financial",
        ),
        ("logs/random_20260101.jsonl", "unassigned"),
    ],
)
def test_protected_empty_is_never_probed_or_removed(
    sandbox, monkeypatch, relative, reason
):
    row = make_row(sandbox, relative)
    authorize(monkeypatch, [row])
    monkeypatch.setattr(
        src.safety, "idle", lambda p: pytest.fail("protected source probed")
    )
    with pytest.raises(src.safety.Deferred, match=reason):
        retire(sandbox, row)
    assert Path(row["path"]).exists()


def test_arbitrary_path_not_authorized_by_zero_length(sandbox, monkeypatch):
    row = make_row(sandbox)
    authorize(monkeypatch, [])
    with pytest.raises(ValueError, match="exact_reviewed_row"):
        retire(sandbox, row)
    assert Path(row["path"]).exists()


def test_current_date_and_recent_mtime_are_preserved(sandbox):
    row = make_row(sandbox)
    assert (
        src.empty_blocker(row, datetime(2026, 1, 3, tzinfo=timezone.utc))
        == "not_closed_historical_placeholder"
    )
    row["mtime_utc"] = datetime.now(timezone.utc).isoformat()
    assert src.empty_blocker(row) == "not_closed_historical_placeholder"


def test_nonempty_refused_even_with_faked_empty_digest(sandbox, monkeypatch):
    row = make_row(sandbox, content=b"nonempty")
    row["full_sha256"] = src.EMPTY_SHA256
    authorize(monkeypatch, [row])
    with pytest.raises(src.safety.Deferred, match="not_verified_empty"):
        retire(sandbox, row)


def test_changed_since_inventory_refused(sandbox, monkeypatch):
    row = make_row(sandbox)
    authorize(monkeypatch, [row])
    Path(row["path"]).write_bytes(b"new data")
    with pytest.raises(RuntimeError, match="changed_since_frozen"):
        retire(sandbox, row)


@pytest.mark.parametrize("stage", ["before", "after"])
def test_active_handle_preserves_source(sandbox, monkeypatch, stage):
    row = make_row(sandbox)
    authorize(monkeypatch, [row])
    calls = []

    def idle(path):
        calls.append(path)
        if len(calls) == (1 if stage == "before" else 2):
            raise src.safety.Deferred("source_open_or_handle_probe_failed")

    monkeypatch.setattr(src.safety, "idle", idle)
    with pytest.raises(src.safety.Deferred, match="source_open"):
        retire(sandbox, row)
    assert Path(row["path"]).exists()


def test_prepared_receipt_failure_preserves_source(sandbox, monkeypatch):
    row = make_row(sandbox)
    authorize(monkeypatch, [row])
    monkeypatch.setattr(
        src.native, "receipt", lambda *args: (_ for _ in ()).throw(OSError("disk full"))
    )
    with pytest.raises(OSError, match="disk full"):
        retire(sandbox, row)
    assert Path(row["path"]).exists()


def test_source_change_after_receipt_preserved(sandbox, monkeypatch):
    row = make_row(sandbox)
    authorize(monkeypatch, [row])

    def receipt(root, payload):
        Path(row["path"]).write_bytes(b"writer returned")

    monkeypatch.setattr(src.native, "receipt", receipt)
    with pytest.raises(RuntimeError, match="changed_before_retirement"):
        retire(sandbox, row)
    assert Path(row["path"]).read_bytes() == b"writer returned"


def test_completion_receipt_failure_reports_actual_removal(sandbox, monkeypatch):
    row = make_row(sandbox)
    authorize(monkeypatch, [row])
    original = src.native.receipt

    def receipt(root, payload):
        if payload["phase"] == "empty_retired":
            raise OSError("completion unavailable")
        original(root, payload)

    monkeypatch.setattr(src.native, "receipt", receipt)
    result = retire(sandbox, row)
    assert result["source_removed"] is True
    assert result["persistence_complete"] is False
    assert not Path(row["path"]).exists()


def test_pair_full_read_catches_unsampled_interior_difference(sandbox):
    a = make_row(
        sandbox,
        "governance/channels/decision/profile/decision_20260101.jsonl",
        b"a" * (3 * src.native.CHUNK),
    )
    b = make_row(
        sandbox,
        "governance/shadow_profile/master_control_20260101.jsonl",
        b"a" * 400000 + b"b" + b"a" * (3 * src.native.CHUNK - 400001),
    )
    with pytest.raises(RuntimeError, match="full_content_mismatch"):
        src.verify_pair([a, b], src.native.Budget(max_bytes=10 * src.native.CHUNK))


def test_pair_success_has_full_identity_and_hash(sandbox):
    a = make_row(sandbox, content=b"data")
    b = make_row(
        sandbox, "governance/shadow_profile/master_control_20260101.jsonl", b"data"
    )
    proofs = src.verify_pair([a, b], src.native.Budget())
    assert all(
        p["full_stream_read"] and p["verified_bytes"] == 4 and len(p["identity"]) == 5
        for p in proofs
    )
    assert (
        proofs[0]["sha256"]
        == proofs[1]["sha256"]
        == hashlib.sha256(b"data").hexdigest()
    )


def test_budget_preflight_does_not_open_sources(sandbox, monkeypatch):
    a = make_row(sandbox, content=b"data")
    b = make_row(
        sandbox, "governance/shadow_profile/master_control_20260101.jsonl", b"data"
    )
    monkeypatch.setattr(
        src, "hash_source", lambda *args: pytest.fail("byte budget must precede reads")
    )
    with pytest.raises(src.safety.Deferred, match="byte_budget"):
        src.verify_pair([a, b], src.native.Budget(max_bytes=7))


@pytest.mark.parametrize("linked_parent", [False, True])
def test_links_fail_closed(sandbox, linked_parent):
    row = make_row(sandbox)
    path = Path(row["path"])
    if linked_parent:
        path.parent.rename(path.parent.with_name("actual"))
        path.parent.symlink_to(path.parent.with_name("actual"))
    else:
        actual = path.with_suffix(".payload")
        path.rename(actual)
        path.symlink_to(actual)
    with pytest.raises(RuntimeError, match="symlink"):
        src.hash_source(row, src.native.Budget())


def test_hardlinked_source_rejected(sandbox):
    row = make_row(sandbox)
    os.link(row["path"], sandbox / "alias")
    with pytest.raises(RuntimeError, match="single_link"):
        src.hash_source(row, src.native.Budget())


def test_verify_only_never_acquires_locks_or_writes_receipts(sandbox, monkeypatch):
    row = make_row(sandbox)
    authorize(monkeypatch, [row])
    monkeypatch.setattr(
        src, "owner_locks", lambda *args: pytest.fail("read-only locks")
    )
    monkeypatch.setattr(
        src.native, "receipt", lambda *args: pytest.fail("read-only receipt")
    )
    result = src.run("synthetic", root=sandbox)
    assert result["results"][0]["status"] == "verified_empty"
    assert Path(row["path"]).exists()


def pair_fixture(sandbox, monkeypatch):
    a = make_row(
        sandbox,
        "local_fallback_storage/governance/channels/decision/profile/decision_20260101.jsonl",
        b"data",
    )
    b = make_row(
        sandbox,
        "local_fallback_storage/governance/shadow_profile/master_control_20260101.jsonl",
        b"data",
    )
    authorize(monkeypatch, [a, b], [{"paths": [a["path"], b["path"]]}])
    monkeypatch.setattr(src, "require_reserve", lambda extra=0: None)
    return [a, b]


def consolidate(sandbox, rows, **kwargs):
    return src.consolidate_pair(
        sandbox, rows, src.native.Budget(), lambda: None, audit="synthetic", **kwargs
    )


def test_duplicate_apply_owns_retained_payload_and_both_aliases(sandbox, monkeypatch):
    rows = pair_fixture(sandbox, monkeypatch)
    result = consolidate(sandbox, rows)
    target = Path(result["canonical"])
    assert target.parent == sandbox / src.PAYLOAD_REL
    assert (
        target.suffix == ".payload"
        and target.stem == hashlib.sha256(b"data").hexdigest()
    )
    assert target.read_bytes() == b"data" and not target.is_symlink()
    assert target.stat().st_nlink == 1
    assert len(result["aliases_replaced"]) == 2
    assert result["persistence_complete"]
    assert all(
        Path(r["path"]).is_symlink() and Path(r["path"]).read_bytes() == b"data"
        for r in rows
    )
    assert src.has_custody(target, rows, result["sha256"])


def test_duplicate_apply_idempotent_with_two_existing_aliases(sandbox, monkeypatch):
    rows = pair_fixture(sandbox, monkeypatch)
    first = consolidate(sandbox, rows)
    target = Path(first["canonical"])
    identity = src.native.identity(target)
    second = consolidate(sandbox, rows)
    assert second["aliases_replaced"] == [] and len(second["already_linked"]) == 2
    assert src.native.identity(target) == identity
    assert second["released_payload_bytes"] == 0


def test_resume_after_first_alias_publication(sandbox, monkeypatch):
    rows = pair_fixture(sandbox, monkeypatch)
    original = src.replace_alias
    calls = []

    def interrupt(path, *args):
        calls.append(path)
        if len(calls) == 2:
            raise RuntimeError("interrupted")
        return original(path, *args)

    monkeypatch.setattr(src, "replace_alias", interrupt)
    first = consolidate(sandbox, rows)
    assert not first["persistence_complete"] and len(first["aliases_replaced"]) == 1
    assert Path(rows[0]["path"]).is_symlink()
    assert not Path(rows[1]["path"]).is_symlink()
    monkeypatch.setattr(src, "replace_alias", original)
    second = consolidate(sandbox, rows)
    assert second["persistence_complete"] and len(second["aliases_replaced"]) == 1
    assert all(Path(r["path"]).read_bytes() == b"data" for r in rows)


def test_existing_payload_without_alias_can_resume(sandbox, monkeypatch):
    rows = pair_fixture(sandbox, monkeypatch)
    target = (
        sandbox / src.PAYLOAD_REL / (hashlib.sha256(b"data").hexdigest() + ".payload")
    )
    target.parent.mkdir(parents=True)
    target.write_bytes(b"data")
    monkeypatch.setattr(
        src, "publish_payload", lambda *args: pytest.fail("existing payload copied")
    )
    result = consolidate(sandbox, rows)
    assert result["persistence_complete"]


def test_corrupt_existing_payload_never_replaces_sources(sandbox, monkeypatch):
    rows = pair_fixture(sandbox, monkeypatch)
    target = (
        sandbox / src.PAYLOAD_REL / (hashlib.sha256(b"data").hexdigest() + ".payload")
    )
    target.parent.mkdir(parents=True)
    target.write_bytes(b"evil")
    with pytest.raises(RuntimeError, match="canonical_content_mismatch"):
        consolidate(sandbox, rows)
    assert all(not Path(r["path"]).is_symlink() for r in rows)


def test_custody_failure_leaves_both_raw_sources(sandbox, monkeypatch):
    rows = pair_fixture(sandbox, monkeypatch)
    monkeypatch.setattr(
        src,
        "custody_receipt",
        lambda *args: (_ for _ in ()).throw(OSError("receipt unavailable")),
    )
    result = consolidate(sandbox, rows)
    assert result["canonical_created"] and result["retained_unaliased_payload"]
    assert result["released_payload_bytes"] == -4
    assert not result["persistence_complete"]
    assert Path(result["canonical"]).read_bytes() == b"data"
    receipts = (
        sandbox / "governance/storage_recovery/verified_duplicate_cleanup.jsonl"
    ).read_text()
    assert "alias_transaction_incomplete" in receipts
    assert all(not Path(r["path"]).is_symlink() for r in rows)


def test_alias_without_original_custody_never_admitted(sandbox, monkeypatch):
    rows = pair_fixture(sandbox, monkeypatch)
    result = consolidate(sandbox, rows)
    target = Path(result["canonical"])
    target.with_name(target.name + ".restore_proofs.receipt").write_text("{}\n")
    with pytest.raises(RuntimeError, match="matching_custody"):
        consolidate(sandbox, rows)


def test_external_reserve_prevents_copy_and_alias(sandbox, monkeypatch):
    rows = pair_fixture(sandbox, monkeypatch)
    monkeypatch.setattr(
        src,
        "require_reserve",
        lambda extra=0: (_ for _ in ()).throw(
            src.safety.Deferred("external_32gib_reserve_required")
        ),
    )
    with pytest.raises(src.safety.Deferred, match="32gib"):
        consolidate(sandbox, rows)
    assert all(not Path(r["path"]).is_symlink() for r in rows)
    assert not (sandbox / src.PAYLOAD_REL).exists()


def test_duplicate_budget_precedes_all_hashing(sandbox, monkeypatch):
    rows = pair_fixture(sandbox, monkeypatch)
    monkeypatch.setattr(
        src, "hash_source", lambda *args: pytest.fail("insufficient budget hash")
    )
    with pytest.raises(src.safety.Deferred, match="byte_budget"):
        src.consolidate_pair(
            sandbox,
            rows,
            src.native.Budget(max_bytes=15),
            lambda: None,
            audit="synthetic",
        )


def test_unknown_duplicate_owner_never_mutates(sandbox, monkeypatch):
    rows = [
        make_row(sandbox, content=b"data"),
        make_row(sandbox, "logs/arbitrary.jsonl", b"data"),
    ]
    authorize(monkeypatch, rows, [{"paths": [r["path"] for r in rows]}])
    with pytest.raises(src.safety.Deferred, match="owner_boundary"):
        consolidate(sandbox, rows)


def test_post_alias_receipt_failure_records_partial_change(sandbox, monkeypatch):
    rows = pair_fixture(sandbox, monkeypatch)
    original = src.custody_receipt

    def fail_after_alias(target, record):
        if record["phase"] == "source_replaced_with_canonical_alias":
            raise OSError("completion unavailable")
        original(target, record)

    monkeypatch.setattr(src, "custody_receipt", fail_after_alias)
    result = consolidate(sandbox, rows)
    assert not result["persistence_complete"]
    assert len(result["aliases_replaced"]) == 1
    assert Path(rows[0]["path"]).is_symlink()
    assert not Path(rows[1]["path"]).is_symlink()


def test_end_to_end_apply_uses_resource_guard_and_shared_locks(sandbox, monkeypatch):
    rows = pair_fixture(sandbox, monkeypatch)
    monkeypatch.setattr(src.safety, "admission", lambda root: (True, "admitted"))
    checks = []
    monkeypatch.setattr(
        src.safety,
        "Guard",
        lambda root, seconds: SimpleNamespace(check=lambda: checks.append(True)),
    )
    result = src.run("synthetic", kind="duplicates", apply=True, root=sandbox)
    assert result["raw_files_replaced_with_aliases"] == 2
    assert result["released_payload_bytes"] == 4
    assert result["results"][0]["status"] == "consolidated"
    assert checks
    assert (sandbox / "governance/locks/storage_maintenance.lock").exists()
    assert all(Path(r["path"]).is_symlink() for r in rows)


def test_resource_admission_failure_has_no_source_changes(sandbox, monkeypatch):
    rows = pair_fixture(sandbox, monkeypatch)
    monkeypatch.setattr(
        src.safety, "admission", lambda root: (False, "memory_admission_not_ready")
    )
    with pytest.raises(src.safety.Deferred, match="memory_admission"):
        src.run("synthetic", kind="duplicates", apply=True, root=sandbox)
    assert all(not Path(r["path"]).is_symlink() for r in rows)
    assert not (sandbox / src.PAYLOAD_REL).exists()


def test_real_reserve_calculation_deducts_full_transient_copy(sandbox, monkeypatch):
    monkeypatch.setattr(
        src.shutil,
        "disk_usage",
        lambda root: SimpleNamespace(free=src.RESERVE_BYTES + 3),
    )
    src.require_reserve(3)
    with pytest.raises(src.safety.Deferred, match="32gib"):
        src.require_reserve(4)


def test_revoke_lease_before_alias_preserves_raw(sandbox, monkeypatch):
    rows = pair_fixture(sandbox, monkeypatch)
    calls = []

    def check():
        calls.append(True)
        if len(calls) >= 4:
            raise src.safety.Deferred("lease_revoked")

    try:
        result = src.consolidate_pair(
            sandbox, rows, src.native.Budget(), check, audit="synthetic"
        )
        assert not result["persistence_complete"]
    except src.safety.Deferred:
        pass
    assert all(not Path(r["path"]).is_symlink() for r in rows)


def test_unrecognized_alias_to_raw_file_is_never_canonical(sandbox, monkeypatch):
    rows = pair_fixture(sandbox, monkeypatch)
    primary = Path(rows[0]["path"])
    primary.unlink()
    primary.symlink_to(rows[1]["path"])
    with pytest.raises(RuntimeError, match="unrecognized_source_alias"):
        consolidate(sandbox, rows)
    assert Path(rows[1]["path"]).read_bytes() == b"data"


def test_guard_expiration_during_copy_removes_only_owned_scratch(sandbox, monkeypatch):
    rows = pair_fixture(sandbox, monkeypatch)
    original = src.publish_payload

    def expire(source, expected, target, digest, budget, check_locks):
        def expired():
            raise src.safety.Deferred("verification_deadline")

        budget.check = expired
        return original(source, expected, target, digest, budget, check_locks)

    monkeypatch.setattr(src, "publish_payload", expire)
    with pytest.raises(src.safety.Deferred, match="deadline"):
        consolidate(sandbox, rows)
    assert all(not Path(r["path"]).is_symlink() for r in rows)
    assert list((sandbox / src.PAYLOAD_REL).glob("*.building")) == []


def test_late_payload_collision_is_not_overwritten(sandbox, monkeypatch):
    rows = pair_fixture(sandbox, monkeypatch)
    original = src.os.link

    def collide(source, target, **kwargs):
        fd = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=kwargs["dst_dir_fd"],
        )
        with os.fdopen(fd, "wb") as handle:
            handle.write(b"collision")
        return original(source, target, **kwargs)

    monkeypatch.setattr(src.os, "link", collide)
    with pytest.raises(FileExistsError):
        consolidate(sandbox, rows)
    target = (
        sandbox / src.PAYLOAD_REL / (hashlib.sha256(b"data").hexdigest() + ".payload")
    )
    assert target.read_bytes() == b"collision"
    assert all(not Path(r["path"]).is_symlink() for r in rows)


def test_payload_aliases_and_custody_are_not_raw_compaction_candidates(
    sandbox, monkeypatch
):
    from scripts.ops import raw_training_compaction_intelligence as raw
    from scripts.ops import cold_archive_compactor as cold

    rows = pair_fixture(sandbox, monkeypatch)
    result = consolidate(sandbox, rows)
    target = Path(result["canonical"])
    custody = target.with_name(target.name + ".restore_proofs.receipt")
    raw_candidates = set(raw._iter_jsonl_files(sandbox))
    cold_candidates = set(
        cold.stable_file_work_candidates(
            sandbox, min_age_hours=0, include_plain_jsonl=True
        )
    )
    for path in (target, custody, *(Path(r["path"]) for r in rows)):
        assert path not in raw_candidates
        assert path not in cold_candidates


def test_post_publication_sync_failure_reports_retained_payload(sandbox, monkeypatch):
    rows = pair_fixture(sandbox, monkeypatch)
    original = src.safety.sync_dir

    def fail_payload_sync(path):
        if path == sandbox / src.PAYLOAD_REL:
            raise OSError("payload directory sync failed")
        return original(path)

    monkeypatch.setattr(src.safety, "sync_dir", fail_payload_sync)
    result = consolidate(sandbox, rows)
    assert result["retained_unaliased_payload"]
    assert result["released_payload_bytes"] == -4
    assert not result["persistence_complete"]
    assert Path(result["canonical"]).read_bytes() == b"data"
    assert all(not Path(r["path"]).is_symlink() for r in rows)


def test_busy_native_retention_lock_preserves_source(sandbox):
    folder = sandbox / "governance/locks"
    folder.mkdir(parents=True)
    with (folder / "data_retention.lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(src.safety.Deferred, match="data_retention.lock_busy"):
            with src.owner_locks(sandbox):
                pytest.fail("busy lock admitted")


def test_replaced_lock_anchor_is_rejected(sandbox):
    with src.owner_locks(sandbox) as check:
        path = sandbox / "governance/locks/storage_maintenance.lock"
        path.rename(path.with_suffix(".old"))
        path.touch()
        with pytest.raises(src.safety.Deferred, match="anchor_changed"):
            check()


def test_video_rejected_before_any_metadata(monkeypatch):
    monkeypatch.setattr(
        src.safety,
        "allowed",
        lambda *args, **kwargs: pytest.fail("protected path probed"),
    )
    with pytest.raises(ValueError, match="protected_volume"):
        src.read_audit(Path("/Volumes/VIDEO/inventory.json"))
    with pytest.raises(ValueError, match="protected_volume"):
        src.source_path("/Volumes/VIDEO/payload.jsonl")


@pytest.mark.parametrize(
    "options",
    [
        {"limit": 0},
        {"limit": 415},
        {"seconds": float("nan")},
        {"seconds": -1},
        {"max_bytes": 0},
        {"start": -1},
    ],
)
def test_invalid_budgets_fail_before_inventory_io(monkeypatch, options):
    monkeypatch.setattr(
        src, "read_audit", lambda *args: pytest.fail("inventory IO before validation")
    )
    with pytest.raises(ValueError):
        src.run("synthetic", **options)


def test_modified_audit_rejected_without_probing_sources(tmp_path):
    path = tmp_path / "inventory.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="not_exact_reviewed_inventory"):
        src.read_audit(path)
