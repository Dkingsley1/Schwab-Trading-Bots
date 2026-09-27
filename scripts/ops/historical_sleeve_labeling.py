#!/usr/bin/env python3
"""Bounded, resumable historical labeling. Never calls trainers or order APIs."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.historical_sleeve_labels import (
    AUTHORITY,
    VERSION,
    annotate,
    canonical_json,
    digest,
    epoch,
    label_contracts,
    price_context,
)
from core.storage_router import inspect_storage_path

MAX_LINE = 2 * 1024 * 1024
MAX_DATABASE = 512 * 1024 * 1024
MAX_BATCH_BYTES = 512 * 1024 * 1024
MAX_RSS = 256 * 1024 * 1024
STATE_REL = "governance/training/historical_sleeve_labels"
REPORT_REL = "governance/health/historical_sleeve_labeling_latest.json"


def implementation_receipt() -> str:
    return digest(
        [
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                ROOT / "core/historical_sleeve_labels.py",
                ROOT / "core/background_work_budget.py",
                ROOT / "core/sleeve_strategy_specialization.py",
            )
        ]
    )


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_path(path: Path, *, missing: bool = False) -> Path:
    route = inspect_storage_path(path)
    if route["status"] not in ({"present", "missing"} if missing else {"present"}):
        raise ValueError("protected_or_unavailable_route")
    return Path(route["resolved_path"])


def load(path: Path) -> dict:
    try:
        with safe_path(path).open() as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (ValueError, OSError):
        return {}


def atomic_json(path: Path, value: dict) -> None:
    path = safe_path(path, missing=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("x") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def fingerprint(path: Path) -> list[int]:
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def source_roots(root: Path) -> list[Path]:
    relative = [
        "decisions",
        "decision_explanations",
        "governance/channels",
        "governance/research",
        "governance/training/generation_fill_learning",
        "governance/shadow",
        "governance/shadow_attribution",
        "data/trade_history",
        "data/sql_link_shards",
        "data/sql_link_sleeves",
        "data/jsonl_link_archives",
        "data/sql_link_archives",
        "data/jsonl_link.sqlite3",
    ]
    roots = [root / rel for rel in relative]
    # Consult retained-storage owners, not a mount-wide search. Never follow a
    # route until the protected-volume guard has checked every link component.
    raw = load(
        root / "governance/health/raw_training_compaction_intelligence_latest.json"
    )
    for entry in raw.get("scan_roots", []):
        path = Path(str(entry.get("path") or ""))
        if path.name == "schwab_trading_bot" and path != root:
            roots.extend(path / rel for rel in relative)
            roots.extend(path / "local_fallback_storage" / rel for rel in relative)
    archive = load(root / "governance/health/sql_link_sleeves_archive_latest.json")
    if archive.get("archive_root"):
        roots.append(Path(archive["archive_root"]))
    service = load(root / "governance/health/sql_link_service_latest.json")
    for entry in service.get("shard_hot_retention", []):
        for key in ("archive_root", "cold_export_root"):
            if entry.get(key):
                roots.append(Path(entry[key]))
    return roots


def inventory(
    root: Path, contracts: dict, aliases: dict, *, include_parquet=False
) -> dict:
    sources, routes, seen = [], [], set()
    deadline = time.monotonic() + 30
    stack = [(p, 0) for p in source_roots(root)]
    scanned = 0
    while stack:
        if time.monotonic() > deadline or scanned >= 50000:
            routes.append(
                {"status": "inventory_budget_exhausted", "remaining_paths": len(stack)}
            )
            break
        path, depth = stack.pop()
        scanned += 1
        route = inspect_storage_path(path)
        status = str(route["status"])
        if status != "present":
            routes.append({"path": str(path), "status": status})
            continue
        resolved = Path(route["resolved_path"])
        if str(resolved) in seen:
            continue
        seen.add(str(resolved))
        if route.get("kind") == "directory":
            if depth >= 6:
                routes.append({"path": str(path), "status": "depth_limit"})
                continue
            with os.scandir(resolved) as entries:
                for entry in entries:
                    if len(stack) + scanned >= 50000:
                        routes.append({"status": "inventory_entry_limit"})
                        break
                    stack.append((Path(entry.path), depth + 1))
            continue
        name = resolved.name
        kind = (
            "sqlite"
            if name.endswith(".sqlite3")
            else (
                "gzip"
                if name.endswith((".jsonl.gz", ".jsonl.raw-training.gz"))
                else (
                    "jsonl"
                    if name.endswith(".jsonl")
                    else (
                        "parquet"
                        if include_parquet and name.endswith(".parquet")
                        else ""
                    )
                )
            )
        )
        if not kind:
            continue
        fp = fingerprint(resolved)
        # Do not hold a read transaction on live SQL writers or manufacture SHM.
        wal = (
            inspect_storage_path(Path(str(resolved) + "-wal"))
            if kind == "sqlite"
            else {}
        )
        status = (
            "deferred_active_sqlite" if wal.get("status") == "present" else "pending"
        )
        if wal.get("status") not in (None, "present", "missing"):
            status = "deferred_unsafe_sqlite_route"
        sources.append(
            {
                "path": str(resolved),
                "kind": kind,
                "fingerprint": fp,
                "source_id": digest([str(resolved), fp]),
                "status": status,
            }
        )
    # Small historical files first gives every available sleeve a turn before
    # multi-gigabyte archives. The frozen manifest, not a lookback, sets scope.
    sources.sort(key=lambda row: (row["fingerprint"][2], row["path"]))
    result = {
        "version": VERSION,
        "created_at_utc": utc(),
        "contracts": contracts,
        "implementation_sha256": implementation_receipt(),
        "aliases": aliases,
        "sources": sources,
        "unavailable_routes": routes,
        "inventory_complete_for_declared_roots": not any(
            r["status"]
            in {
                "inventory_budget_exhausted",
                "inventory_entry_limit",
                "depth_limit",
                "inspection_error",
            }
            for r in routes
        ),
        "scope": "retained_decision_execution_channel_research_and_sql_history_at_frozen_file_inventory",
        "excluded_scope": [
            "credentials",
            "unconfigured_or_unmounted_history",
            "protected_storage",
            "formats_outside_jsonl_gzip_sqlite_and_declared_parquet",
            "records_written_after_inventory",
        ],
        "authority": AUTHORITY,
    }
    result["inventory_sha256"] = digest(result)
    return result


def admission(root: Path) -> list[str]:
    from scripts.ops.support_maintenance_gate import support_maintenance_freeze_contract

    reasons = []
    freeze = support_maintenance_freeze_contract(root, "historical_sleeve_labeling")
    if freeze.get("active"):
        reasons.append(str(freeze["reason"]))
    now = time.time()
    grade = load(root / "governance/health/grade_regression_guard_latest.json")
    grade_time = epoch(grade.get("timestamp_utc"))
    storage = next(
        (s for s in grade.get("surfaces", []) if s.get("surface") == "storage_control"),
        {},
    )
    if grade_time is None or not 0 <= now - grade_time <= 3600:
        reasons.append("stale_or_missing_storage_regression_admission")
    if storage.get("state") not in {"ready", "degraded"}:
        reasons.append("storage_regression_gate_not_admitted")
    training = load(root / "governance/health/training_runtime_control_latest.json")
    ts = epoch(training.get("timestamp_utc"))
    if ts is None or not 0 <= now - ts <= 6 * 3600:
        reasons.append("stale_or_missing_preparation_admission")
    if training.get("prep_allowed") is not True:
        reasons.append("preparation_not_admitted")
    reserve = load(root / "governance/health/local_storage_reserve_guard_latest.json")
    ts = epoch(reserve.get("timestamp_utc"))
    if ts is None or not 0 <= now - ts <= 3600:
        reasons.append("stale_or_missing_storage_admission")
    reserve = reserve.get("local_storage_reserve", {})
    if (
        reserve.get("pause_nonessential_writers") is not False
        or reserve.get("hard_block") is not False
    ):
        reasons.append("storage_writers_not_admitted")
    try:
        floor = max(float(reserve.get("pressure_free_gb", 64)), 64) * 1024**3
        if not 64 * 1024**3 <= floor <= 1024**5:
            raise ValueError("invalid_reserve")
    except (ValueError, TypeError, OverflowError):
        reasons.append("invalid_storage_reserve_contract")
        floor = 64 * 1024**3
    if shutil.disk_usage(root).free < floor + MAX_DATABASE:
        reasons.append("insufficient_storage_headroom")
    for name in ("RUNTIME_MAINTENANCE_HOLD.flag", "OPERATOR_STOP.flag"):
        if (root / "governance/health" / name).exists() or (root / name).exists():
            reasons.append("maintenance_or_operator_hold")
    if os.environ.get("RUNTIME_MAINTENANCE_HOLD") == "1":
        reasons.append("maintenance_environment_hold")
    return reasons


def connect(path: Path) -> sqlite3.Connection:
    path = safe_path(path, missing=True)
    db = sqlite3.connect(path, timeout=2)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA cache_size=-4096")
    db.execute("PRAGMA temp_store=FILE")
    db.execute(f"PRAGMA max_page_count={MAX_DATABASE // 4096}")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS sources (
          source_id TEXT PRIMARY KEY, status TEXT, cursor INTEGER DEFAULT 0,
          rows INTEGER DEFAULT 0, bytes INTEGER DEFAULT 0, reason TEXT DEFAULT '');
        CREATE TABLE IF NOT EXISTS observations (
          id INTEGER PRIMARY KEY, row_sha256 TEXT UNIQUE, sleeve TEXT,
          symbol TEXT, provider TEXT, instrument TEXT, candidate TEXT,
          ts REAL, price REAL, payload TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS observation_time ON observations
          (sleeve,symbol,provider,instrument,candidate,ts);
        CREATE TABLE IF NOT EXISTS receipts (
          source_id TEXT, ordinal INTEGER, row_sha256 TEXT,
          PRIMARY KEY(source_id,ordinal));
        CREATE TABLE IF NOT EXISTS dispositions (
          source_id TEXT, reason TEXT, n INTEGER,
          PRIMARY KEY(source_id,reason));
        CREATE TABLE IF NOT EXISTS contexts (
          observation_id INTEGER, horizon INTEGER, status TEXT, payload TEXT, revision INTEGER,
          PRIMARY KEY(observation_id,horizon));
        CREATE TABLE IF NOT EXISTS label_progress (
          revision INTEGER PRIMARY KEY, cursor INTEGER DEFAULT 0);
    """)
    return db


def _rss() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def database_headroom(db) -> bool:
    pages = db.execute("PRAGMA page_count").fetchone()[0]
    size = db.execute("PRAGMA page_size").fetchone()[0]
    return pages * size < MAX_DATABASE - 16 * 1024 * 1024


def ingest_source(
    db, source, manifest, deadline, budget: int, root: Path | None = None
) -> dict:
    sid = source["source_id"]
    state = dict(
        db.execute("SELECT * FROM sources WHERE source_id=?", (sid,)).fetchone()
    )
    try:
        path = safe_path(Path(source["path"]))
        unchanged = fingerprint(path) == source["fingerprint"]
    except (OSError, ValueError):
        unchanged = False
    if not unchanged:
        db.execute(
            "UPDATE sources SET status='changed',reason='source_changed_since_inventory' WHERE source_id=?",
            (sid,),
        )
        db.commit()
        return {"bytes": 0, "rows": 0}
    counters, used, processed = Counter(), 0, 0
    cursor, ordinal = state["cursor"], state["rows"]
    status, reason = "pending", "batch_budget"
    file_handle = sql = None
    last_admission = time.monotonic()
    try:
        if source["kind"] == "sqlite":
            if inspect_storage_path(Path(str(path) + "-wal"))["status"] != "missing":
                raise ValueError("sqlite_wal_appeared")
            sql = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=1)
            sql.execute("PRAGMA query_only=ON")
            sql.execute("PRAGMA cache_size=-2048")
            sql.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_LINE + 65536)
            sql.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
            columns = {r[1] for r in sql.execute("PRAGMA table_info(jsonl_records)")}
            if not {"id", "source_rel", "payload_json"}.issubset(columns):
                raise ValueError("unsupported_sqlite_schema")
            iterator = iter(
                sql.execute(
                    "SELECT id,source_rel,payload_json FROM jsonl_records WHERE id>? ORDER BY id",
                    (cursor,),
                )
            )
        else:
            file_handle = (
                gzip.open(path, "rb") if source["kind"] == "gzip" else path.open("rb")
            )
            # Gzip seek replays decompression. Seeking is deadline-bounded by
            # the parent CLI process watchdog; it never publishes partial rows.
            file_handle.seek(cursor)
            iterator = None
        db.execute("BEGIN")
        while time.monotonic() < deadline and used < budget and _rss() < MAX_RSS:
            if not database_headroom(db):
                reason = "sidecar_storage_budget_reached"
                break
            if root is not None and time.monotonic() - last_admission > 2:
                if admission(root):
                    reason = "preparation_admission_withdrawn"
                    break
                last_admission = time.monotonic()
            if iterator is not None:
                record = next(iterator, None)
                if record is None:
                    status, reason = "complete", ""
                    break
                next_cursor, rel, raw = record
                raw = raw.encode() if isinstance(raw, str) else bytes(raw or b"")
            else:
                raw = file_handle.readline(MAX_LINE + 1)
                if not raw:
                    status, reason = "complete", ""
                    break
                next_cursor, rel = file_handle.tell(), source["path"]
            used += len(raw)
            if len(raw) > MAX_LINE:
                # Do not split an overlarge line into fake independent records.
                status, reason = "blocked", "record_exceeds_line_budget"
                break
            ordinal += 1
            processed += 1
            sha = hashlib.sha256(raw.strip()).hexdigest()
            try:
                row = json.loads(raw)
                if not isinstance(row, dict):
                    raise ValueError("non_object")
            except (ValueError, UnicodeError):
                counters["malformed_or_nonobject_record"] += 1
                row = None
            if row is not None:
                annotation = annotate(
                    row, manifest["contracts"], manifest["aliases"], str(rel)
                )
                for why in annotation["reasons"] or ["annotated_context_only"]:
                    counters[why] += 1
                # Unassigned rows are accounted by source, not copied with raw
                # potentially sensitive payloads into the training sidecar.
                if annotation["sleeve_id"]:
                    sha = digest([sha, annotation["sleeve_id"]])
                    db.execute(
                        "INSERT OR IGNORE INTO observations(row_sha256,sleeve,symbol,provider,instrument,candidate,ts,price,payload) VALUES(?,?,?,?,?,?,?,?,?)",
                        (
                            sha,
                            annotation["sleeve_id"],
                            annotation["symbol"],
                            annotation["provider"],
                            annotation["instrument_type"],
                            annotation["source_candidate_id"],
                            annotation["epoch"],
                            annotation["price"],
                            canonical_json(annotation),
                        ),
                    )
            db.execute("INSERT INTO receipts VALUES(?,?,?)", (sid, ordinal, sha))
            cursor = next_cursor
        sql_became_live = (
            source["kind"] == "sqlite"
            and inspect_storage_path(Path(str(path) + "-wal"))["status"] != "missing"
        )
        if fingerprint(path) != source["fingerprint"] or sql_became_live:
            db.rollback()
            db.execute(
                "UPDATE sources SET status='changed',reason='source_changed_during_batch' WHERE source_id=?",
                (sid,),
            )
            db.commit()
            return {"bytes": used, "rows": 0}
        for why, count in counters.items():
            db.execute(
                "INSERT INTO dispositions VALUES(?,?,?) ON CONFLICT(source_id,reason) DO UPDATE SET n=n+excluded.n",
                (sid, why, count),
            )
        db.execute(
            "UPDATE sources SET status=?,cursor=?,rows=?,bytes=bytes+?,reason=? WHERE source_id=?",
            (status, cursor, ordinal, used, reason, sid),
        )
        db.commit()
    except (OSError, ValueError, sqlite3.Error, EOFError) as exc:
        db.rollback()
        db.execute(
            "UPDATE sources SET status='blocked',reason=? WHERE source_id=?",
            (str(exc) if isinstance(exc, ValueError) else type(exc).__name__, sid),
        )
        db.commit()
        processed = 0
    finally:
        if file_handle:
            file_handle.close()
        if sql:
            sql.close()
    return {"bytes": used, "rows": processed}


def label_contexts(db, manifest, deadline) -> int:
    count = 0
    revision = db.execute("SELECT COALESCE(MAX(id),0) FROM observations").fetchone()[0]
    db.execute("INSERT OR IGNORE INTO label_progress(revision) VALUES(?)", (revision,))
    cursor = db.execute(
        "SELECT cursor FROM label_progress WHERE revision=?", (revision,)
    ).fetchone()[0]
    # New history can reveal conflicting marks. Only targets computed against
    # the current indexed revision count in coverage; older revisions stay out.
    rows = db.execute(
        "SELECT id,payload FROM observations WHERE id>? ORDER BY id", (cursor,)
    )
    for record in rows:
        if (
            time.monotonic() >= deadline
            or _rss() >= MAX_RSS
            or not database_headroom(db)
        ):
            break
        anchor = json.loads(record["payload"])
        contract = manifest["contracts"][anchor["sleeve_id"]]
        horizons = (
            contract["supplemental_price_horizons_seconds"]
            if anchor["epoch"] is not None and anchor["price"]
            else []
        )
        for horizon in horizons:
            target = anchor["epoch"] + horizon
            match = db.execute(
                "SELECT ts,payload FROM observations WHERE sleeve=? AND symbol=? AND provider=? AND instrument=? AND candidate=? AND ts>=? AND ts<=? AND price>0 ORDER BY ts,id LIMIT 1",
                (
                    anchor["sleeve_id"],
                    anchor["symbol"],
                    anchor["provider"],
                    anchor["instrument_type"],
                    anchor["source_candidate_id"],
                    target,
                    target + min(300, max(30, horizon * 0.01)),
                ),
            ).fetchone()
            outcome = json.loads(match["payload"]) if match else None
            result = price_context(anchor, outcome, horizon, contract)
            result["indexed_observation_revision"] = revision
            if match:
                # Conflicting marks at either endpoint must not be arbitrarily
                # resolved by input order or by whichever archive was read first.
                conflicts = db.execute(
                    "SELECT ts,COUNT(DISTINCT price) AS n FROM observations WHERE sleeve=? AND symbol=? AND provider=? AND instrument=? AND candidate=? AND ts IN (?,?) GROUP BY ts HAVING n>1",
                    (
                        anchor["sleeve_id"],
                        anchor["symbol"],
                        anchor["provider"],
                        anchor["instrument_type"],
                        anchor["source_candidate_id"],
                        anchor["epoch"],
                        match["ts"],
                    ),
                ).fetchone()
                if conflicts:
                    result.update(
                        status="quarantined",
                        value=None,
                        reason="conflicting_endpoint_prices",
                    )
            db.execute(
                "INSERT INTO contexts VALUES(?,?,?,?,?) ON CONFLICT(observation_id,horizon) DO UPDATE SET status=excluded.status,payload=excluded.payload,revision=excluded.revision",
                (
                    record["id"],
                    horizon,
                    result["status"],
                    canonical_json(result),
                    revision,
                ),
            )
            count += 1
        db.execute(
            "UPDATE label_progress SET cursor=? WHERE revision=?",
            (record["id"], revision),
        )
        if count and count % 200 == 0:
            db.commit()
    db.commit()
    return count


def report(db, manifest, run_dir, blockers: list[str]) -> dict:
    source_counts = dict(
        db.execute("SELECT status,COUNT(*) FROM sources GROUP BY status")
    )
    counts = {
        r[0]: dict(r)
        for r in db.execute(
            "SELECT sleeve,COUNT(*) AS records,MIN(ts) AS first_epoch,MAX(ts) AS last_epoch FROM observations GROUP BY sleeve"
        )
    }
    revision = db.execute("SELECT COALESCE(MAX(id),0) FROM observations").fetchone()[0]
    context_counts = dict(
        db.execute(
            "SELECT status,COUNT(*) FROM contexts WHERE revision=? GROUP BY status",
            (revision,),
        )
    )
    incomplete = sum(v for k, v in source_counts.items() if k != "complete")
    rows = []
    for sleeve, contract in manifest["contracts"].items():
        item = {
            "sleeve_id": sleeve,
            "contract": contract,
            **counts.get(sleeve, {"records": 0}),
            "primary_outcome_status": "authority_specific_materialization_pending",
            "historical_coverage": (
                "partial" if incomplete else "declared_inventory_scanned"
            ),
            "training_eligible": False,
        }
        item["supplemental_context_counts"] = dict(
            db.execute(
                "SELECT c.status,COUNT(*) FROM contexts c JOIN observations o ON o.id=c.observation_id WHERE o.sleeve=? AND c.revision=? GROUP BY c.status",
                (sleeve, revision),
            )
        )
        if not item["records"]:
            item["historical_coverage"] = "no_attributed_records_in_processed_sources"
        rows.append(item)
    total = db.execute(
        "SELECT COALESCE(SUM(rows),0),COALESCE(SUM(bytes),0) FROM sources"
    ).fetchone()
    result = {
        "version": VERSION,
        "timestamp_utc": utc(),
        "run_directory": str(run_dir),
        "inventory_sha256": manifest["inventory_sha256"],
        "inventory_created_at_utc": manifest["created_at_utc"],
        "overall_status": "blocked" if blockers else "partial",
        "blockers": blockers,
        "sleeve_count": len(rows),
        "sleeves_with_attributed_records": len(counts),
        "source_status_counts": source_counts,
        "source_records_processed": total[0],
        "source_bytes_processed": total[1],
        "unique_attributed_records": sum(row["records"] for row in rows),
        "supplemental_context_counts": context_counts,
        "indexed_observation_revision": revision,
        "disposition_counts": dict(
            db.execute("SELECT reason,SUM(n) FROM dispositions GROUP BY reason")
        ),
        "declared_sources_scanned": incomplete == 0
        and manifest["inventory_complete_for_declared_roots"],
        "historical_backfill_complete": False,
        "completion_blockers": [
            "authority_specific_primary_outcomes_not_materialized",
            "non_jsonl_non_sqlite_history_not_scanned",
        ]
        + (
            ["historical_sources_pending_or_unavailable"]
            if incomplete or manifest["unavailable_routes"]
            else []
        ),
        "unavailable_route_count": len(manifest["unavailable_routes"]),
        "unresolved_research_horizon_sleeves": sum(
            c["unresolved_horizon"] for c in manifest["contracts"].values()
        ),
        "data_file": str(run_dir / "labels.sqlite3"),
        "implementation_sha256": manifest["implementation_sha256"],
        "authority": AUTHORITY,
        "sleeves": rows,
    }
    return result


def run(
    root: Path, *, execute: bool, seconds: int = 120, new_inventory: bool = False
) -> dict:
    base = safe_path(root / STATE_REL, missing=True)
    base.mkdir(parents=True, exist_ok=True)
    with (base / "backfill.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        blockers = admission(root) if execute else []
        if blockers:
            # Keep the last batch's evidence timestamp and receipts intact.
            # Reporting a denied run must not create another empty inventory or
            # disguise existing annotations as newly materialized evidence.
            prior = load(root / REPORT_REL)
            payload = {
                **prior,
                "timestamp_utc": utc(),
                "evidence_timestamp_utc": prior.get("evidence_timestamp_utc")
                or prior.get("timestamp_utc"),
                "previous_evidence_not_recomputed": True,
                "overall_status": "blocked",
                "blockers": blockers,
                "execute_requested": True,
                "batch_fresh_payload_bytes_processed": 0,
                "runner_implementation_sha256": implementation_receipt(),
                "historical_backfill_complete": False,
                "authority": AUTHORITY,
            }
            atomic_json(root / REPORT_REL, payload)
            return payload
        contracts, aliases = label_contracts(root)
        pointer = load(base / "latest_run.json")
        name = str(pointer.get("run_id") or "")
        if name and not name.isalnum():
            raise ValueError("invalid_run_pointer")
        if not name or new_inventory:
            name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        state = safe_path(base / "runs" / name, missing=True)
        state.mkdir(parents=True, exist_ok=True)
        manifest_path = state / "inventory.json"
        manifest = load(manifest_path)
        if not manifest:
            manifest = inventory(root, contracts, aliases)
            atomic_json(manifest_path, manifest)
        if manifest.get("contracts") != contracts or manifest.get("aliases") != aliases:
            raise ValueError("label_contract_changed_new_versioned_inventory_required")
        if manifest.get("implementation_sha256") != implementation_receipt():
            raise ValueError(
                "label_implementation_changed_new_versioned_inventory_required"
            )
        body = {k: v for k, v in manifest.items() if k != "inventory_sha256"}
        if digest(body) != manifest.get("inventory_sha256"):
            raise ValueError("inventory_integrity_failure")
        atomic_json(base / "latest_run.json", {"run_id": name})
        with closing(connect(state / "labels.sqlite3")) as db:
            for source in manifest["sources"]:
                db.execute(
                    "INSERT OR IGNORE INTO sources(source_id,status) VALUES(?,?)",
                    (source["source_id"], source["status"]),
                )
            db.commit()
            blockers = admission(root) if execute else []
            if execute and not database_headroom(db):
                blockers.append("sidecar_storage_budget_reached")
            deadline = time.monotonic() + min(max(seconds, 10), 300)
            used = 0
            if execute and not blockers:
                for source in manifest["sources"]:
                    if time.monotonic() > deadline - 20 or used >= MAX_BATCH_BYTES:
                        break
                    status = db.execute(
                        "SELECT status FROM sources WHERE source_id=?",
                        (source["source_id"],),
                    ).fetchone()[0]
                    if status != "pending":
                        continue
                    blockers = admission(root)
                    if blockers:
                        break
                    progress = ingest_source(
                        db,
                        source,
                        manifest,
                        deadline - 20,
                        MAX_BATCH_BYTES - used,
                        root=root,
                    )
                    used += progress["bytes"]
                    if not database_headroom(db):
                        blockers.append("sidecar_storage_budget_reached")
                        break
                    if _rss() >= MAX_RSS:
                        blockers.append("resident_memory_budget_reached")
                        break
                blockers.extend(
                    reason for reason in admission(root) if reason not in blockers
                )
                if not blockers:
                    label_contexts(db, manifest, deadline)
                if (
                    not database_headroom(db)
                    and "sidecar_storage_budget_reached" not in blockers
                ):
                    blockers.append("sidecar_storage_budget_reached")
            payload = report(db, manifest, state, blockers)
            payload["execute_requested"] = execute
            payload["batch_fresh_payload_bytes_processed"] = used
            payload["rss_high_water_bytes"] = _rss()
            atomic_json(state / "coverage.json", payload)
            atomic_json(root / REPORT_REL, payload)
            return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Process one admitted bounded batch; repeat to resume.",
    )
    parser.add_argument(
        "--new-inventory",
        action="store_true",
        help="Start a new frozen inventory, preserving all earlier runs.",
    )
    parser.add_argument("--seconds", type=int, default=120)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Process the complete retained inventory under bounded admission; resume completed compressed partitions.",
    )
    args = parser.parse_args()
    # A parent watchdog also bounds slow archive I/O and gzip seek, which cannot
    # be interrupted reliably by a Python loop's monotonic check.
    if not os.environ.get("HISTORICAL_LABEL_CHILD"):
        import subprocess

        child = subprocess.Popen(
            [sys.executable, __file__, *sys.argv[1:]],
            env={**os.environ, "HISTORICAL_LABEL_CHILD": "1"},
        )
        try:
            return child.wait(
                timeout=min(max(args.seconds, 10), 7200 if args.full else 300) + 60
            )
        except subprocess.TimeoutExpired:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            print(
                canonical_json(
                    {
                        "overall_status": "blocked",
                        "reason": "batch_deadline_exceeded",
                        "historical_backfill_complete": False,
                    }
                )
            )
            return 2
    try:
        if args.full:
            if not args.run:
                parser.error("--full requires --run")
            from scripts.ops.historical_sleeve_backfill import run_full

            from core.background_work_budget import background_policy

            work_policy = background_policy()
            payload = run_full(
                ROOT,
                seconds=args.seconds,
                new_inventory=args.new_inventory,
                work_policy=work_policy,
            )
        else:
            payload = run(
                ROOT,
                execute=args.run,
                seconds=args.seconds,
                new_inventory=args.new_inventory,
            )
        print(
            json.dumps({k: v for k, v in payload.items() if k != "sleeves"}, indent=2)
        )
        return 2 if payload["blockers"] else 0
    except (ValueError, OSError, sqlite3.Error, RuntimeError) as exc:
        print(
            canonical_json(
                {
                    "overall_status": "blocked",
                    "reason": str(exc),
                    "historical_backfill_complete": False,
                }
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
