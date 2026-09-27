"""Full source-granular backfill with compressed labels and a deduplicated mark index."""

from __future__ import annotations

from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time

import orjson

from core.historical_sleeve_labels import (
    AUTHORITY,
    annotate,
    digest,
    label_contracts,
    price_context,
)
from scripts.ops import historical_sleeve_labeling as common

MAX_OUTPUT = 8 * 1024**3
MAX_MARK_INDEX = 512 * 1024**2
_WORK_BUDGET = None
_WORK_POLICY = {}


def _pace():
    if _WORK_BUDGET is not None:
        _WORK_BUDGET.tick()


class Paused(RuntimeError):
    pass


def _check(root, directory, deadline):
    _pace()
    if time.monotonic() >= deadline:
        raise Paused("run_time_budget_reached")
    if common._rss() >= common.MAX_RSS:
        raise Paused("resident_memory_budget_reached")
    reasons = common.admission(root)
    if reasons:
        raise Paused(",".join(reasons))
    if shutil.disk_usage(directory).free < 64 * 1024**3 + common.MAX_DATABASE:
        raise Paused("storage_reserve_reached")


def _sha(path, check=None):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            if check:
                check()
            h.update(block)
    return h.hexdigest()


def _raw_records(source, deadline):
    path = common.safe_path(Path(source["path"]))
    if source["kind"] in {"jsonl", "gzip"}:
        if common.fingerprint(path) != source["fingerprint"]:
            raise Paused("source_changed_since_inventory")
        opener = gzip.open if source["kind"] == "gzip" else open
        with opener(path, "rb") as handle:
            ordinal = 0
            while True:
                _pace()
                raw = handle.readline(common.MAX_LINE + 1)
                if not raw:
                    break
                ordinal += 1
                h = hashlib.sha256(raw)
                size = len(raw)
                oversized = size > common.MAX_LINE
                while not raw.endswith(b"\n") and len(raw) == common.MAX_LINE + 1:
                    if time.monotonic() >= deadline:
                        raise Paused("run_time_budget_reached")
                    raw = handle.readline(common.MAX_LINE + 1)
                    h.update(raw)
                    size += len(raw)
                yield ordinal, source["path"], (
                    None if oversized else raw
                ), size, h.hexdigest()
        if common.fingerprint(path) != source["fingerprint"]:
            raise Paused("source_changed_during_scan")
    elif source["kind"] == "sqlite":
        # A real read transaction includes committed WAL rows. Never use
        # immutable=1, checkpoint a writer, or drop sidecars to make a read work.
        original = common.fingerprint(path)
        if original[:2] != source["fingerprint"][:2]:
            raise Paused("source_replaced_since_inventory")
        for suffix in ("-wal", "-shm"):
            common.safe_path(Path(str(path) + suffix), missing=True)
        with closing(
            sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=2)
        ) as db:
            db.execute("PRAGMA query_only=ON")
            db.execute("PRAGMA cache_size=-2048")
            db.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
            db.execute("BEGIN")
            tables = []
            for name in ("jsonl_records", "json_file_records"):
                columns = {r[1] for r in db.execute(f"PRAGMA table_info({name})")}
                if columns and not {"id", "source_rel", "payload_json"} <= columns:
                    raise Paused("unsupported_sqlite_schema")
                if columns:
                    tables.append(name)
            if not tables:
                raise Paused("unsupported_sqlite_schema")
            initial_wal = common.inspect_storage_path(Path(str(path) + "-wal"))
            initial_size = initial_wal.get("size_bytes") or 0
            for name in tables:
                query = f"SELECT id,source_rel,CASE WHEN length(CAST(payload_json AS BLOB))<=? THEN payload_json ELSE NULL END,length(CAST(payload_json AS BLOB)) FROM {name} ORDER BY id"
                for count, (ordinal, rel, payload, size) in enumerate(
                    db.execute(query, (common.MAX_LINE,)), 1
                ):
                    _pace()
                    raw = payload.encode() if isinstance(payload, str) else payload
                    yield f"{name}:{ordinal}", rel, raw, size or 0, (
                        hashlib.sha256(raw).hexdigest() if raw is not None else None
                    )
                    if count % 1000 == 0:
                        wal = common.inspect_storage_path(Path(str(path) + "-wal"))
                        if (wal.get("size_bytes") or 0) - initial_size > 256 * 1024**2:
                            raise Paused("live_sqlite_wal_growth_budget_reached")
            db.rollback()
        if common.fingerprint(path)[:2] != original[:2]:
            raise Paused("source_replaced_during_scan")
    elif source["kind"] == "parquet":
        if common.fingerprint(path) != source["fingerprint"]:
            raise Paused("source_changed_since_inventory")
        import pyarrow.parquet as pq

        with pq.ParquetFile(path) as file:
            if any(
                file.metadata.row_group(i).total_byte_size > common.MAX_RSS // 2
                for i in range(file.metadata.num_row_groups)
            ):
                raise Paused("parquet_row_group_exceeds_memory_budget")
            if not {"source_rel", "payload_json"} <= set(file.schema_arrow.names):
                raise Paused("unsupported_parquet_schema")
            ordinal = 0
            for batch in file.iter_batches(
                batch_size=1, columns=["source_rel", "payload_json"], use_threads=False
            ):
                for record in batch.to_pylist():
                    ordinal += 1
                    raw = record["payload_json"]
                    raw = raw.encode() if isinstance(raw, str) else raw
                    size = len(raw) if isinstance(raw, bytes) else 0
                    sha = (
                        hashlib.sha256(raw).hexdigest()
                        if isinstance(raw, bytes)
                        else None
                    )
                    yield ordinal, record["source_rel"], (
                        raw if size <= common.MAX_LINE else None
                    ), size, sha
        if common.fingerprint(path) != source["fingerprint"]:
            raise Paused("source_changed_during_scan")
    else:
        raise Paused("unsupported_source_format")


def _scan_source(source, manifest, directory, root, deadline, committed_bytes):
    sid = source["source_id"]
    destination = directory / f"{sid}.labels.jsonl.gz"
    partial = destination.with_suffix(destination.suffix + ".part")
    rows, size = 0, 0
    sleeves, dispositions = Counter(), Counter()
    chain = hashlib.sha256()
    last_check = time.monotonic()
    started = common.utc()
    try:
        with partial.open("wb") as raw_out:
            with gzip.GzipFile(
                filename="", fileobj=raw_out, mode="wb", compresslevel=1, mtime=0
            ) as out:
                for ordinal, rel, raw, nbytes, sha in _raw_records(source, deadline):
                    if time.monotonic() - last_check >= 2:
                        _check(root, directory, deadline)
                        if committed_bytes + raw_out.tell() >= MAX_OUTPUT:
                            raise Paused("compressed_output_budget_reached")
                        last_check = time.monotonic()
                    rows += 1
                    size += nbytes
                    try:
                        row = orjson.loads(raw) if raw is not None else None
                        if not isinstance(row, dict):
                            raise ValueError("nonobject")
                        result = annotate(
                            row, manifest["contracts"], manifest["aliases"], str(rel)
                        )
                    except (ValueError, TypeError, orjson.JSONDecodeError):
                        result = {
                            "sleeve_id": "",
                            "record_status": "quarantined",
                            "reasons": [
                                (
                                    "oversized_source_record"
                                    if raw is None
                                    else "malformed_or_nonobject_record"
                                )
                            ],
                            "primary_label": {"status": "unavailable", "value": None},
                            "authority": AUTHORITY,
                        }
                    result["source_ordinal"] = ordinal
                    result["source_row_sha256"] = sha
                    result["source_id"] = sid
                    encoded = orjson.dumps(result) + b"\n"
                    out.write(encoded)
                    chain.update(encoded)
                    if result["sleeve_id"]:
                        sleeves[result["sleeve_id"]] += 1
                    for why in result["reasons"] or ["annotated_context_only"]:
                        dispositions[why] += 1
            raw_out.flush()
            os.fsync(raw_out.fileno())
        _check(root, directory, deadline)
        if committed_bytes + partial.stat().st_size > MAX_OUTPUT:
            raise Paused("compressed_output_budget_reached")
        os.replace(partial, destination)
        receipt = {
            "source_id": sid,
            "status": "complete",
            "rows": rows,
            "source_payload_bytes": size,
            "sleeve_counts": dict(sleeves),
            "dispositions": dict(dispositions),
            "labels_file": str(destination),
            "label_rows_sha256": chain.hexdigest(),
            "labels_file_sha256": _sha(
                destination, lambda: _check(root, directory, deadline)
            ),
            "output_bytes": destination.stat().st_size,
            "started_at_utc": started,
            "source_inventory_fingerprint": source["fingerprint"],
            "completed_at_utc": common.utc(),
            "source_kind": source["kind"],
            "sqlite_read_policy": (
                "per_source_read_transaction_including_wal"
                if source["kind"] == "sqlite"
                else None
            ),
            "scope": "source_records_observed_during_this_source_snapshot",
            "authority": AUTHORITY,
        }
        common.atomic_json(directory / f"{sid}.receipt.json", receipt)
        return receipt
    finally:
        partial.unlink(missing_ok=True)


def _labels(receipt, check=None):
    path = common.safe_path(Path(receipt["labels_file"]))
    if _sha(path, check) != receipt["labels_file_sha256"]:
        raise Paused("label_partition_integrity_failure")
    chain = hashlib.sha256()
    count = 0
    with gzip.open(path, "rb") as handle:
        for raw in handle:
            chain.update(raw)
            count += 1
            yield orjson.loads(raw)
    if count != receipt["rows"] or chain.hexdigest() != receipt["label_rows_sha256"]:
        raise Paused("label_partition_row_integrity_failure")


def _context_stage(receipts, manifest, directory, root, deadline):
    index_path = directory / "market_marks.sqlite3"
    with closing(sqlite3.connect(index_path, timeout=2)) as db:
        db.execute("PRAGMA cache_size=-4096")
        db.execute(f"PRAGMA max_page_count={MAX_MARK_INDEX // 4096}")
        db.executescript("""
            CREATE TABLE IF NOT EXISTS marks(seq TEXT,ts REAL,price REAL,snapshot TEXT,payload BLOB,
                PRIMARY KEY(seq,ts,price,snapshot)) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS indexed(source_id TEXT PRIMARY KEY,receipt_sha TEXT);
        """)
        for receipt in receipts:
            sid = receipt["source_id"]
            old = db.execute(
                "SELECT receipt_sha FROM indexed WHERE source_id=?", (sid,)
            ).fetchone()
            if old:
                if old[0] != digest(receipt):
                    raise Paused("indexed_receipt_changed")
                continue
            _check(root, directory, deadline)
            last_check = time.monotonic()
            for row in _labels(receipt, lambda: _check(root, directory, deadline)):
                _pace()
                if time.monotonic() - last_check >= 2:
                    _check(root, directory, deadline)
                    last_check = time.monotonic()
                keys = (
                    "sleeve_id",
                    "symbol",
                    "provider",
                    "instrument_type",
                    "source_candidate_id",
                    "snapshot_id",
                )
                if (
                    not all(row.get(k) for k in keys)
                    or row.get("epoch") is None
                    or not row.get("price")
                    or row.get("record_status") == "quarantined"
                ):
                    continue
                quote_time = common.epoch(row.get("quote_timestamp_utc"))
                if quote_time is None or not 0 <= row["epoch"] - quote_time <= 120:
                    continue
                seq = digest([row[k] for k in keys[:5]])
                db.execute(
                    "INSERT OR IGNORE INTO marks VALUES(?,?,?,?,?)",
                    (
                        seq,
                        row["epoch"],
                        row["price"],
                        row["snapshot_id"],
                        orjson.dumps(row),
                    ),
                )
            db.execute("INSERT INTO indexed VALUES(?,?)", (sid, digest(receipt)))
            db.commit()
        revision = digest([r["labels_file_sha256"] for r in receipts])
        target = directory / f"price_context_{revision}.jsonl.gz"
        summary_path = target.with_suffix(".receipt.json")
        prior = common.load(summary_path)
        if prior and prior.get("sha256") == _sha(
            target, lambda: _check(root, directory, deadline)
        ):
            return prior
        counts, per_sleeve = Counter(), {}
        partial = target.with_suffix(target.suffix + ".part")
        last_check = time.monotonic()
        try:
            with gzip.open(partial, "wb", compresslevel=1) as out:
                for seq, ts, payload in db.execute(
                    "SELECT seq,ts,payload FROM marks ORDER BY seq,ts"
                ):
                    if time.monotonic() - last_check >= 2:
                        _check(root, directory, deadline)
                        if (
                            sum(r["output_bytes"] for r in receipts)
                            + partial.stat().st_size
                            > MAX_OUTPUT
                        ):
                            raise Paused("compressed_output_budget_reached")
                        last_check = time.monotonic()
                    _pace()
                    anchor = orjson.loads(payload)
                    contract = manifest["contracts"][anchor["sleeve_id"]]
                    for horizon in contract["supplemental_price_horizons_seconds"]:
                        end = ts + horizon
                        tolerance = min(300, max(30, horizon * 0.01))
                        match = db.execute(
                            "SELECT ts,payload FROM marks WHERE seq=? AND ts>=? AND ts<=? ORDER BY ts LIMIT 1",
                            (seq, end, end + tolerance),
                        ).fetchone()
                        result = price_context(
                            anchor,
                            orjson.loads(match[1]) if match else None,
                            horizon,
                            contract,
                        )
                        if (
                            match
                            and db.execute(
                                "SELECT ts FROM marks WHERE seq=? AND ts IN (?,?) GROUP BY ts HAVING COUNT(DISTINCT price)>1 LIMIT 1",
                                (seq, ts, match[0]),
                            ).fetchone()
                        ):
                            result.update(
                                status="quarantined",
                                value=None,
                                reason="conflicting_endpoint_prices",
                            )
                        result.update(
                            sleeve_id=anchor["sleeve_id"],
                            anchor_snapshot_id=anchor["snapshot_id"],
                            feature_timestamp_utc=anchor["timestamp_utc"],
                            source_id=anchor["source_id"],
                            source_row_sha256=anchor["source_row_sha256"],
                            source_ordinal=anchor["source_ordinal"],
                            input_revision=revision,
                        )
                        out.write(orjson.dumps(result) + b"\n")
                        counts[result["status"]] += 1
                        per_sleeve.setdefault(anchor["sleeve_id"], Counter())[
                            result["status"]
                        ] += 1
            if (
                sum(r["output_bytes"] for r in receipts) + partial.stat().st_size
                > MAX_OUTPUT
            ):
                raise Paused("compressed_output_budget_reached")
            with partial.open("rb") as sync:
                os.fsync(sync.fileno())
            os.replace(partial, target)
        finally:
            partial.unlink(missing_ok=True)
        result = {
            "status": "complete",
            "counts": dict(counts),
            "per_sleeve": per_sleeve,
            "file": str(target),
            "sha256": _sha(target, lambda: _check(root, directory, deadline)),
            "input_revision": revision,
            "unique_market_marks": db.execute("SELECT COUNT(*) FROM marks").fetchone()[
                0
            ],
            "authority": AUTHORITY,
        }
        common.atomic_json(summary_path, result)
        return result


def _publish(root, directory, manifest, receipts, failures, contexts, blockers):
    counts, dispositions = Counter(), Counter()
    for r in receipts:
        counts.update(r["sleeve_counts"])
        dispositions.update(r["dispositions"])
    scan_complete = (
        len(receipts) == len(manifest["sources"])
        and manifest["inventory_complete_for_declared_roots"]
    )
    result = {
        "version": "historical_full_backfill_v2",
        "timestamp_utc": common.utc(),
        "run_directory": str(directory),
        "inventory_sha256": manifest["inventory_sha256"],
        "inventory_created_at_utc": manifest["created_at_utc"],
        "overall_status": (
            "blocked"
            if blockers
            else (
                "complete_with_outcome_gaps"
                if scan_complete and contexts.get("status") == "complete"
                else "partial"
            )
        ),
        "blockers": blockers,
        "source_count": len(manifest["sources"]),
        "completed_sources": len(receipts),
        "source_records_processed": sum(r["rows"] for r in receipts),
        "source_payload_bytes_processed": sum(
            r["source_payload_bytes"] for r in receipts
        ),
        "compressed_label_bytes": sum(r["output_bytes"] for r in receipts),
        "sleeves_with_attributed_records": len(counts),
        "sleeve_count": len(manifest["contracts"]),
        "unresolved_research_horizon_sleeves": 0,
        "historical_source_scan_complete": scan_complete,
        "context_materialization": contexts,
        "worker_policy": _WORK_POLICY,
        "worker_budget": _WORK_BUDGET.snapshot() if _WORK_BUDGET else {},
        "scope": "declared_retained_inventory_with_per_source_read_snapshots_not_global_atomic_snapshot",
        "excluded_scope": manifest["excluded_scope"],
        "historical_backfill_complete": False,
        "verified_primary_training_targets": 0,
        "completion_blockers": ["authority_specific_primary_outcomes_not_materialized"]
        + ([] if scan_complete else ["retained_sources_not_fully_scanned"]),
        "failed_or_deferred_sources": failures,
        "unavailable_routes": manifest["unavailable_routes"],
        "disposition_counts": dict(dispositions),
        "authority": AUTHORITY,
        "sleeves": [
            {
                "sleeve_id": sid,
                "contract": c,
                "attributed_records": counts[sid],
                "primary_outcome_status": "authority_specific_materialization_pending",
                "research_horizon": c["research_horizon"],
            }
            for sid, c in manifest["contracts"].items()
        ],
    }
    common.atomic_json(directory / "coverage.json", result)
    common.atomic_json(root / common.REPORT_REL, result)
    return result


def run_full(root, seconds=7200, new_inventory=False, work_policy=None):
    global _WORK_BUDGET, _WORK_POLICY
    from core.background_work_budget import WorkBudget

    _WORK_POLICY = work_policy or {}
    _WORK_BUDGET = WorkBudget() if work_policy else None
    base = common.safe_path(root / common.STATE_REL, missing=True)
    base.mkdir(parents=True, exist_ok=True)
    with (base / "backfill.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        contracts, aliases = label_contracts(root)
        pointer = common.load(base / "latest_full_run.json")
        name = str(pointer.get("run_id") or "")
        if name and not name.isalnum():
            raise Paused("invalid_full_run_pointer")
        if new_inventory or not name:
            name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        directory = common.safe_path(base / "full_runs" / name, missing=True)
        directory.mkdir(parents=True, exist_ok=True)
        manifest = common.load(directory / "inventory.json")
        code_sha = digest([common.implementation_receipt(), _sha(Path(__file__))])
        if not manifest:
            manifest = common.inventory(root, contracts, aliases, include_parquet=True)
            manifest.pop("inventory_sha256")
            manifest["full_runner_sha256"] = code_sha
            manifest["inventory_sha256"] = digest(manifest)
            common.atomic_json(directory / "inventory.json", manifest)
        if (
            digest({k: v for k, v in manifest.items() if k != "inventory_sha256"})
            != manifest["inventory_sha256"]
            or manifest.get("full_runner_sha256") != code_sha
            or manifest["contracts"] != contracts
        ):
            raise Paused("full_run_contract_or_code_changed_new_inventory_required")
        common.atomic_json(base / "latest_full_run.json", {"run_id": name})
        receipts, failures, contexts, blockers = [], [], {}, []
        deadline = time.monotonic() + min(max(int(seconds), 30), 7200)
        publish_horizons(root, contracts)
        last_progress = time.monotonic()
        prior_verified = False
        try:
            _check(root, directory, deadline)
            completed_ids = set()
            for source in manifest["sources"]:
                _check(root, directory, deadline)
                receipt = common.load(directory / f"{source['source_id']}.receipt.json")
                if receipt:
                    if (
                        receipt.get("source_id") != source["source_id"]
                        or Path(receipt["labels_file"])
                        != directory / f"{source['source_id']}.labels.jsonl.gz"
                        or _sha(
                            common.safe_path(Path(receipt["labels_file"])),
                            lambda: _check(root, directory, deadline),
                        )
                        != receipt.get("labels_file_sha256")
                    ):
                        raise Paused("completed_source_receipt_integrity_failure")
                    receipts.append(receipt)
                    completed_ids.add(source["source_id"])
            prior_verified = True
            for source in manifest["sources"]:
                if source["source_id"] in completed_ids:
                    continue
                _check(root, directory, deadline)
                try:
                    receipt = _scan_source(
                        source,
                        manifest,
                        directory,
                        root,
                        deadline,
                        sum(r["output_bytes"] for r in receipts),
                    )
                    receipts.append(receipt)
                except (OSError, sqlite3.Error, ValueError, Paused) as exc:
                    reason = str(exc) if isinstance(exc, Paused) else type(exc).__name__
                    failures.append(
                        {"source_id": source["source_id"], "reason": reason}
                    )
                    if isinstance(exc, Paused) and reason not in {
                        "source_changed_since_inventory",
                        "source_changed_during_scan",
                        "source_replaced_since_inventory",
                        "source_replaced_during_scan",
                        "unsupported_sqlite_schema",
                        "unsupported_parquet_schema",
                        "unsupported_source_format",
                        "parquet_row_group_exceeds_memory_budget",
                        "live_sqlite_wal_growth_budget_reached",
                    }:
                        raise
                if time.monotonic() - last_progress >= 30:
                    print(
                        json.dumps(
                            {
                                "phase": "historical_scan",
                                "completed_sources": len(receipts),
                                "source_count": len(manifest["sources"]),
                                "records": sum(r["rows"] for r in receipts),
                                "source_failures": len(failures),
                            }
                        ),
                        flush=True,
                    )
                    _publish(
                        root, directory, manifest, receipts, failures, contexts, []
                    )
                    last_progress = time.monotonic()
            contexts = _context_stage(receipts, manifest, directory, root, deadline)
        except (Paused, OSError, sqlite3.Error) as exc:
            blockers.append(str(exc) if isinstance(exc, Paused) else type(exc).__name__)
        if blockers and not prior_verified:
            prior = common.load(directory / "coverage.json")
            if prior:
                prior.update(
                    overall_status="blocked",
                    blockers=blockers,
                    last_attempt_utc=common.utc(),
                    previous_evidence_not_recomputed=True,
                )
                common.atomic_json(root / common.REPORT_REL, prior)
                return prior
        return _publish(
            root, directory, manifest, receipts, failures, contexts, blockers
        )


def publish_horizons(root, contracts):
    target = root / common.STATE_REL / "research_horizons.json"
    common.atomic_json(
        target,
        {
            "timestamp_utc": common.utc(),
            "sleeve_count": len(contracts),
            "purpose": "Predeclared research windows, not validated optimal holding periods or a training-readiness countdown.",
            "authority": AUTHORITY,
            "sleeves": {s: c["research_horizon"] for s, c in contracts.items()},
        },
    )
    lines = [
        "# Sleeve Research Horizons",
        "",
        "Research-only windows. Days are elapsed calendar days, not exchange sessions. Missing or unverified outcomes remain unknown. These windows neither change execution holding periods nor guarantee enough training data. Live money remains locked.",
        "",
        "| Sleeve | Primary | Secondary | Required Endpoint |",
        "| --- | --- | --- | --- |",
    ]

    def duration(seconds):
        for divisor, suffix in ((86400, "d"), (3600, "h"), (60, "m")):
            if seconds % divisor == 0:
                return f"{seconds // divisor}{suffix}"
        return f"{seconds}s"

    for sleeve, contract in sorted(contracts.items()):
        h = contract["research_horizon"]
        lines.append(
            f"| {sleeve} | {duration(h['primary_seconds'])} | {', '.join(duration(s) for s in h['secondary_seconds'])} | {h['endpoint']} |"
        )
    path = common.safe_path(target.with_suffix(".md"), missing=True)
    temporary = path.with_suffix(".md.tmp")
    try:
        with temporary.open("w") as handle:
            handle.write("\n".join(lines) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
