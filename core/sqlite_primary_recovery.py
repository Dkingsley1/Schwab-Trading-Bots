"""Explicit recovery of a previously committed primary; never automatic replay."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time

from core import sqlite_primary_storage as primary
from core.sqlite_primary_retirement import _owned_json
from core.write_path_recovery import durable_json


def _quiet(source, target, rows, source_identities):
    for base in (source, target / "data"):
        result = subprocess.run(
            ["lsof", "+D", str(base)], capture_output=True, timeout=30
        )
        if result.returncode != 1 or result.stdout or result.stderr:
            raise RuntimeError("primary_recovery_handles_present_or_unknown")
    for row in rows:
        relative = row["relative"]
        old = source / relative
        primary._physical(old.parent)
        actual = primary._identity(old) if os.path.lexists(old) else None
        if actual != source_identities[relative]:
            raise ValueError("primary_recovery_standby_changed")
        for base in (source, target / "data"):
            primary._physical((base / relative).parent)
            for suffix in ("-wal", "-shm", "-journal"):
                sidecar = Path(str(base / relative) + suffix)
                if os.path.lexists(sidecar) and primary._identity(sidecar)[2]:
                    raise ValueError("primary_recovery_nonempty_journal")


def restore_committed_routes(
    root: Path,
    *,
    handoff_path: Path,
    transaction_path: Path,
    retirement_path: Path,
    reconciliation_path: Path
) -> dict:
    root = Path(root).absolute()
    if not primary.enabled(root):
        raise ValueError("primary_recovery_profile_required")
    primary._hold(root)
    target = primary._validated_target()
    source = root / "local_fallback_storage/data"
    primary._physical(source)
    primary._physical(source / "sql_link_shards")
    handoff = _owned_json(root, handoff_path)
    transaction = _owned_json(root, transaction_path)
    retired = _owned_json(root, retirement_path)
    reconciled = _owned_json(root, reconciliation_path)
    if (
        handoff.get("purpose") != "sqlite_primary_cutover"
        or handoff.get("source_root") != str(source)
        or handoff.get("target_root") != str(target)
        or handoff.get("volume_uuid")
        != primary._getenv("BOT_LOGS_EXTERNAL_VOLUME_UUID")
        or transaction.get("purpose") != "sqlite_primary_route_transaction"
        or transaction.get("phase") != "committed"
        or transaction.get("target_root") != str(target)
        or transaction.get("receipt_sha256")
        != hashlib.sha256(handoff_path.read_bytes()).hexdigest()
        or retired.get("purpose") != "sqlite_primary_standby_retirement"
        or retired.get("phase") != "complete"
        or retired.get("ok") is not True
        or retired.get("apply") is not True
        or reconciled.get("purpose") != "sqlite_primary_route_recovery_reconciliation"
        or reconciled.get("all_standby_payloads_preserved") is not True
        or reconciled.get("primary_cursors_not_regressed") is not True
    ):
        raise ValueError("primary_recovery_custody_incomplete")
    rows = handoff["files"]
    if not 3 < len(rows) <= 100 or len({r["relative"] for r in rows}) != len(rows):
        raise ValueError("primary_recovery_inventory_invalid")
    expected = set(primary.DATABASES)
    for row in rows:
        rel = Path(row["relative"])
        if (
            rel.is_absolute()
            or ".." in rel.parts
            or (str(rel) not in primary.DATABASES and rel.parts[0] != "sql_link_shards")
        ):
            raise ValueError("primary_recovery_path_invalid")
        expected.add(str(rel))
    deadline = time.monotonic() + 2400
    source_ids = {}
    retired_rows = {r["relative"]: r for r in retired.get("verified", [])}
    for row in rows:
        relative = row["relative"]
        old = source / relative
        primary._physical(old.parent)
        if not os.path.lexists(old):
            if (
                relative not in retired.get("retired", [])
                or relative not in retired_rows
                or retired_rows[relative]["source_identity"] != row["source_identity"]
            ):
                raise ValueError("primary_recovery_unexplained_missing_standby")
            source_ids[relative] = None
        else:
            identity = primary._identity(old)
            approved = (
                reconciled["source_identity"]
                if relative == "jsonl_link.sqlite3"
                else row["source_identity"]
            )
            if identity != approved:
                raise ValueError("primary_recovery_unreconciled_standby")
            source_ids[relative] = identity
    if (
        primary._identity(target / "data/jsonl_link.sqlite3")
        != reconciled["target_identity_after"]
    ):
        raise ValueError("primary_recovery_reconciled_target_changed")
    primary._physical(root / "governance/locks")
    with (root / "governance/locks/sqlite_primary_route.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if primary._shard_inventory(
            target / "data/sql_link_shards", deadline
        ) != expected - set(primary.DATABASES):
            raise ValueError("primary_recovery_target_inventory_changed")
        _quiet(source, target, rows, source_ids)
        verified = []
        for row in rows:
            primary._hold(root)
            path = target / "data" / row["relative"]
            primary._physical(path.parent)
            before = primary._identity(path)
            if before[:2] != row["target_identity"][:2]:
                raise ValueError("primary_recovery_target_replaced")
            digest = primary._digest(path, deadline)
            # Exact fresh byte equality preserves the committed snapshot's
            # structural proof; metadata equality alone never earns this credit.
            identical_verified_snapshot = (
                digest == row.get("sha256") and row.get("quick_check") == "ok"
            )
            if not identical_verified_snapshot:
                conn = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
                try:
                    conn.execute("PRAGMA cache_size=-8192")
                    conn.set_progress_handler(
                        lambda: int(time.monotonic() >= deadline), 10000
                    )
                    if conn.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                        raise ValueError("primary_recovery_integrity_failed")
                finally:
                    conn.close()
            if primary._identity(path) != before:
                raise ValueError("primary_recovery_target_changed")
            verified.append(
                {
                    "relative": row["relative"],
                    "identity": before,
                    "sha256": digest,
                    "quick_check": "ok",
                    "integrity_basis": (
                        "fresh_full_hash_matches_committed_integrity_checked_snapshot"
                        if identical_verified_snapshot
                        else "fresh_sqlite_quick_check"
                    ),
                }
            )
            print(
                json.dumps(
                    {"phase": "primary_recovery_verified", "relative": row["relative"]}
                ),
                flush=True,
            )
        primary._hold(root)
        primary._validated_target()
        _quiet(source, target, rows, source_ids)
        if primary._shard_inventory(
            target / "data/sql_link_shards", deadline
        ) != expected - set(primary.DATABASES):
            raise ValueError("primary_recovery_target_inventory_changed")
        for row in verified:
            if primary._identity(target / "data" / row["relative"]) != row["identity"]:
                raise ValueError("primary_recovery_target_changed_before_publication")
        proof = {
            "purpose": "sqlite_primary_committed_route_recovery",
            "files": verified,
            "source_identities": source_ids,
            "prior_transaction": str(transaction_path),
            "target_root": str(target),
            "live_execution_authority": False,
            "source_retired": False,
            "ingestion_verified": False,
        }
        path = (
            root / "governance/storage_recovery/sqlite_primary_recovery_verified.json"
        )
        durable_json(root, path, proof)
        journal = primary._publish_verified_links(
            root, target, hashlib.sha256(path.read_bytes()).hexdigest()
        )
        return {**proof, "ok": True, "journal": str(journal), "routes_restored": True}
