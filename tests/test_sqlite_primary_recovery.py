import json
from pathlib import Path
import sqlite3

import pytest

from core import sqlite_primary_recovery as recovery
from core import sqlite_primary_storage as primary
from tests.test_sqlite_primary_storage import cohort


@pytest.fixture
def committed(cohort):
    root, source, target, receipt = cohort
    payload = json.loads(receipt.read_text())
    for row in payload["files"]:
        row["target_identity"] = primary._identity(target / "data" / row["relative"])
    receipt.write_text(json.dumps(payload))
    primary.commit_routes(root, receipt)
    directory = root / "governance/storage_recovery"
    transaction = next(directory.glob("sqlite_primary_transaction_*.json"))
    retired = directory / "retired.json"
    retired.write_text(
        json.dumps(
            {
                "purpose": "sqlite_primary_standby_retirement",
                "ok": True,
                "phase": "complete",
                "apply": True,
                "retired": [],
                "verified": [],
            }
        )
    )
    reconciliation = directory / "reconciled.json"
    reconciliation.write_text(
        json.dumps(
            {
                "purpose": "sqlite_primary_route_recovery_reconciliation",
                "all_standby_payloads_preserved": True,
                "primary_cursors_not_regressed": True,
                "source_identity": primary._identity(source / "jsonl_link.sqlite3"),
                "target_identity_after": primary._identity(
                    target / "data/jsonl_link.sqlite3"
                ),
            }
        )
    )
    for rel in primary.LINKS:
        (root / rel).unlink()
        (root / rel).symlink_to(root / "local_fallback_storage" / rel)
    arguments = dict(
        handoff_path=receipt,
        transaction_path=transaction,
        retirement_path=retired,
        reconciliation_path=reconciliation,
    )
    return root, source, target, arguments


def test_explicit_recovery_preserves_both_payload_copies(committed):
    root, source, target, arguments = committed
    before = {p.name: p.read_bytes() for p in source.glob("*.sqlite3")}
    result = recovery.restore_committed_routes(root, **arguments)
    assert result["ok"] and result["routes_restored"]
    assert result["ingestion_verified"] is False
    assert before == {p.name: p.read_bytes() for p in source.glob("*.sqlite3")}
    assert primary.observe(root)["ok"]


def test_unchanged_bytes_reuse_committed_integrity_only_after_full_hash(
    committed, monkeypatch
):
    root, source, target, arguments = committed
    hashed = []
    real_digest = primary._digest

    def digest(path, deadline):
        hashed.append(path)
        return real_digest(path, deadline)

    def unexpected_connect(*args, **kwargs):
        raise AssertionError("Identical verified bytes need no second structural scan")

    monkeypatch.setattr(primary, "_digest", digest)
    monkeypatch.setattr(recovery.sqlite3, "connect", unexpected_connect)
    result = recovery.restore_committed_routes(root, **arguments)
    assert len(hashed) == len(result["files"])
    assert all(
        row["integrity_basis"].startswith("fresh_full_hash_matches_")
        for row in result["files"]
    )


def test_changed_bytes_require_fresh_sqlite_integrity(committed):
    root, source, target, arguments = committed
    path = target / "data/snapshot_context.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE added_after_cutover(value INTEGER)")
    conn.close()
    result = recovery.restore_committed_routes(root, **arguments)
    row = next(r for r in result["files"] if r["relative"] == path.name)
    assert row["integrity_basis"] == "fresh_sqlite_quick_check"


@pytest.mark.parametrize("budget", [0, -1, 2401, 7201, True, None])
def test_unrecognized_verification_budget_rejected_before_io(tmp_path, budget):
    with pytest.raises(ValueError, match="verification_budget_invalid"):
        recovery.restore_committed_routes(
            tmp_path,
            handoff_path=tmp_path / "missing",
            transaction_path=tmp_path / "missing",
            retirement_path=tmp_path / "missing",
            reconciliation_path=tmp_path / "missing",
            verification_timeout_seconds=budget,
        )


def test_explicit_extended_verification_retains_all_proof_requirements(committed):
    root, source, target, arguments = committed
    result = recovery.restore_committed_routes(
        root, **arguments, verification_timeout_seconds=7200
    )
    assert result["verification_timeout_seconds"] == 7200
    assert result["routes_restored"] and result["files"]
    assert result["live_execution_authority"] is False


@pytest.mark.parametrize(
    "damage",
    [
        "journal",
        "source",
        "missing",
        "target",
        "extra",
        "foreign_link",
        "reconciliation",
    ],
)
def test_recovery_fails_closed_and_preserves_routes(committed, damage):
    root, source, target, arguments = committed
    if damage == "journal":
        path = arguments["transaction_path"]
        value = json.loads(path.read_text())
        value["phase"] = "prepared"
        path.write_text(json.dumps(value))
    elif damage == "source":
        (source / "snapshot_context.sqlite3").write_bytes(b"changed")
    elif damage == "missing":
        (source / "snapshot_context.sqlite3").unlink()
    elif damage == "target":
        (target / "data/snapshot_context.sqlite3").write_bytes(b"corrupt")
    elif damage == "extra":
        (target / "data/sql_link_shards/unknown").write_bytes(b"unknown")
    elif damage == "foreign_link":
        path = root / "data/snapshot_context.sqlite3"
        path.unlink()
        path.symlink_to(root / "foreign")
    else:
        arguments["reconciliation_path"].write_text("{}")
    before = {rel: (root / rel).readlink() for rel in primary.LINKS}
    with pytest.raises(Exception):
        recovery.restore_committed_routes(root, **arguments)
    assert before == {rel: (root / rel).readlink() for rel in primary.LINKS}


def test_recovery_accepts_only_documented_retired_source(committed):
    root, source, target, arguments = committed
    relative = "sql_link_shards/trading.sqlite3"
    identity = primary._identity(source / relative)
    (source / relative).unlink()
    path = arguments["retirement_path"]
    data = json.loads(path.read_text())
    data["retired"] = [relative]
    data["verified"] = [{"relative": relative, "source_identity": identity}]
    path.write_text(json.dumps(data))
    assert recovery.restore_committed_routes(root, **arguments)["routes_restored"]


def test_recovery_hold_loss_never_publishes(committed, monkeypatch):
    root, source, target, arguments = committed

    def unavailable(root):
        raise RuntimeError("hold lost")

    monkeypatch.setattr(primary, "_hold", unavailable)
    with pytest.raises(RuntimeError, match="hold lost"):
        recovery.restore_committed_routes(root, **arguments)
    assert (
        root / "data/jsonl_link.sqlite3"
    ).readlink() == source / "jsonl_link.sqlite3"
