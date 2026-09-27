"""Bounded, read-only evidence for newly committed JSONL/JSON SQL rows."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.storage_router import inspect_storage_path

TABLES = {"jsonl_records": "source_stream", "json_file_records": "stream"}


def _utc(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.microsecond:
        raise ValueError("verification timestamps must be timezone-aware whole seconds")
    return result.astimezone(timezone.utc)


def _receipt(root: Path, name: str, now: datetime) -> dict[str, Any]:
    route = inspect_storage_path(root / "governance/health" / name)
    if route["status"] != "present" or route.get("size_bytes", 0) > 2 * 1024**2:
        return {"status": "unavailable", "path": name}
    try:
        with Path(route["resolved_path"]).open("rb") as source:
            raw = source.read(2 * 1024**2 + 1)
        if len(raw) > 2 * 1024**2:
            raise ValueError("receipt too large")
        value = json.loads(raw)
        stamp = datetime.fromisoformat(value["timestamp_utc"].replace("Z", "+00:00"))
        age = (now - stamp).total_seconds()
        return {
            "status": "fresh" if 0 <= age <= 180 else "stale",
            "age_seconds": age,
            "payload": value,
        }
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return {"status": "unavailable", "path": name}


def _databases(root: Path) -> tuple[list[dict], list[dict]]:
    candidates = []
    findings = []
    local = Path(
        os.getenv("BOT_LOGS_LOCAL_FALLBACK_ROOT", str(root / "local_fallback_storage"))
    ).expanduser()
    for base in (root, local):
        candidates.append(base / "data/jsonl_link.sqlite3")
        directory = base / "data/sql_link_shards"
        route = inspect_storage_path(directory)
        if route["status"] == "missing" and not route.get("symlinks"):
            continue
        if route["status"] != "present":
            findings.append({"path": str(directory), "status": route["status"]})
            continue
        try:
            # Enumerate this directory only, never descend into archives or other volumes.
            with os.scandir(route["resolved_path"]) as entries:
                for index, entry in enumerate(entries):
                    if index >= 256:
                        findings.append(
                            {"path": str(directory), "status": "directory_entry_limit"}
                        )
                        break
                    if entry.name.startswith("jsonl_link_") and entry.name.endswith(
                        ".sqlite3"
                    ):
                        candidates.append(directory / entry.name)
        except OSError as exc:
            findings.append({"path": str(directory), "status": type(exc).__name__})
    physical = {}
    for path in candidates:
        route = inspect_storage_path(path)
        if route["status"] == "missing" and not route.get("symlinks"):
            continue
        if route["status"] != "present":
            findings.append({"path": str(path), "status": route["status"]})
            continue
        try:
            stat = Path(route["resolved_path"]).stat()
            key = (stat.st_dev, stat.st_ino)
            row = physical.setdefault(
                key,
                {
                    "path": route["resolved_path"],
                    "aliases": [],
                    "device": key[0],
                    "inode": key[1],
                },
            )
            row["aliases"].append(str(path))
        except OSError as exc:
            findings.append({"path": str(path), "status": type(exc).__name__})
    return list(physical.values()), findings


def _verify_database(row: dict, lower: str, upper: str, budget: dict) -> dict:
    result = {**row, "status": "incomplete", "tables": []}
    if time.monotonic() >= budget["deadline"]:
        return {**result, "reason": "deadline"}
    conn = None
    try:
        # Reinspect just before opening; URI mode=ro must never create a missing DB.
        route = inspect_storage_path(row["path"])
        if route["status"] != "present":
            raise ValueError("database_route_unavailable")
        path = Path(route["resolved_path"])
        stat = path.stat()
        if (stat.st_dev, stat.st_ino) != (row["device"], row["inode"]):
            raise ValueError("database_identity_changed")
        conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=0.2)
        conn.execute("PRAGMA query_only=ON")
        conn.execute("PRAGMA cache_size=-2048")
        deadline = min(budget["deadline"], time.monotonic() + 15)
        conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
        conn.execute("BEGIN")
        available = {
            item[0]
            for item in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        for table, stream in TABLES.items():
            if table not in available:
                continue
            index = f"idx_{table}_ingested_at"
            indexes = {item[1] for item in conn.execute(f"PRAGMA index_list({table})")}
            if index not in indexes or [
                item[2] for item in conn.execute(f"PRAGMA index_info({index})")
            ] != ["ingested_at"]:
                result["tables"].append(
                    {
                        "table": table,
                        "status": "incomplete",
                        "reason": "required_time_index_missing",
                    }
                )
                continue
            # Writer timestamps are UTC ISO strings. Whole-second prefix bounds include
            # every fractional timestamp at the start, and exclude the entire end second.
            where = "ingested_at >= ? AND ingested_at < ?"
            query = f"FROM {table} INDEXED BY {index} WHERE {where}"
            total = conn.execute(f"SELECT COUNT(*) {query}", (lower, upper)).fetchone()[
                0
            ]
            evidence = {
                "table": table,
                "committed_rows": total,
                "checked_rows": 0,
                "hash_mismatches": 0,
                "invalid_json_rows": 0,
                "checked_payload_bytes": 0,
            }
            result["tables"].append(evidence)
            counts = Counter()
            cursor = conn.execute(
                f"SELECT id, length(CAST(payload_json AS BLOB)), {stream} {query} ORDER BY ingested_at",
                (lower, upper),
            )
            for row_id, size, family in cursor:
                if time.monotonic() >= deadline:
                    evidence["reason"] = "deadline"
                    break
                if type(size) is not int or size < 0:
                    evidence["reason"] = "invalid_payload_size"
                    break
                if size > 8 * 1024**2 or size > budget["bytes"] or budget["rows"] <= 0:
                    evidence["reason"] = "payload_or_row_budget"
                    break
                payload, digest = conn.execute(
                    f"SELECT payload_json, payload_sha1 FROM {table} WHERE id=?",
                    (row_id,),
                ).fetchone()
                if not isinstance(payload, str):
                    evidence["reason"] = "non_text_payload"
                    break
                raw = payload.encode("utf-8")
                budget["bytes"] -= len(raw)
                budget["rows"] -= 1
                evidence["checked_rows"] += 1
                evidence["checked_payload_bytes"] += len(raw)
                evidence["hash_mismatches"] += int(
                    hashlib.sha1(raw).hexdigest() != digest
                )
                try:
                    json.loads(payload)
                except (ValueError, RecursionError):
                    evidence["invalid_json_rows"] += 1
                label = str(family or "unclassified")[:120]
                counts[
                    label if label in counts or len(counts) < 64 else "other_streams"
                ] += 1
            evidence["checked_rows_by_stream"] = dict(counts)
            evidence["status"] = (
                "failed"
                if evidence["hash_mismatches"] or evidence["invalid_json_rows"]
                else "verified" if evidence["checked_rows"] == total else "incomplete"
            )
        after = path.stat()
        if (after.st_dev, after.st_ino) != (row["device"], row["inode"]):
            raise ValueError("database_identity_changed")
        statuses = [item["status"] for item in result["tables"]]
        result["status"] = (
            "verified"
            if statuses and all(item == "verified" for item in statuses)
            else "failed" if "failed" in statuses else "incomplete"
        )
        if not statuses:
            result["reason"] = "no_supported_ingestion_tables"
    except (sqlite3.Error, OSError, ValueError, TypeError) as exc:
        result.update(status="incomplete", reason=f"{type(exc).__name__}:{exc}")
    finally:
        if conn is not None:
            conn.close()
    return result


def build_verification(
    root: Path,
    *,
    since: str,
    until: str | None = None,
    max_seconds: int = 90,
    max_payload_mib: int = 256,
    max_rows: int = 100000,
) -> dict:
    start = _utc(since)
    now = datetime.now(timezone.utc)
    end = _utc(until) if until else now.replace(microsecond=0)
    if not start < end <= now:
        raise ValueError("verification requires since < until <= now")
    if (
        not 1 <= max_seconds <= 300
        or not 1 <= max_payload_mib <= 1024
        or not 1 <= max_rows <= 1000000
    ):
        raise ValueError("verification budget outside supported bounds")
    budget = {
        "deadline": time.monotonic() + max_seconds,
        "bytes": max_payload_mib * 1024**2,
        "rows": max_rows,
    }
    candidates, findings = _databases(root)
    databases = [
        _verify_database(row, start.isoformat()[:19], end.isoformat()[:19], budget)
        for row in candidates
    ]
    for database in databases:
        name = Path(database["path"]).stem.removeprefix("jsonl_link_")
        receipt_name = (
            "jsonl_sql_ingestion_health_latest.json"
            if name == "jsonl_link"
            else f"jsonl_sql_ingestion_health_{name}_latest.json"
        )
        receipt = _receipt(root, receipt_name, now)
        producer = receipt.pop("payload", {})
        receipt.update(
            timestamp_utc=producer.get("timestamp_utc"),
            ingest_run_id=producer.get("ingest_run_id"),
            checkpoint_mode=producer.get("checkpoint_mode"),
            counters_scope="latest producer pass only, not the complete verification window",
            counters=(
                {
                    key: (producer.get("sqlite") or {}).get(key)
                    for key in (
                        "inserted",
                        "invalid",
                        "oversize_payloads",
                        "ops_write_failures",
                        "pending_lines",
                    )
                }
                if isinstance(producer.get("sqlite"), dict)
                else {}
            ),
        )
        database["producer_receipt"] = receipt
    checked = (
        bool(databases)
        and not findings
        and all(row["status"] == "verified" for row in databases)
    )
    census = _receipt(root, "ingestion_backpressure_latest.json", now)
    observed = census.pop("payload", {})
    census.update(
        {
            key: observed.get(key)
            for key in (
                "timestamp_utc",
                "pending_lines_total",
                "pending_files_total",
                "scan_selection",
                "lane_accounting",
                "line_estimation",
            )
        }
    )
    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "window": {
            "since_inclusive_utc": start.isoformat(),
            "until_exclusive_utc": end.isoformat(),
            "timestamp_basis": "writer-owned UTC ingested_at, not source event time",
        },
        "overall_status": "verified_stored_rows" if checked else "incomplete",
        "stored_row_verification_complete": checked,
        "all_new_source_data_ingested": False,
        "scope": {
            "discovery": "canonical and configured local-fallback primary/shard paths; physical aliases deduplicated",
            "proof": "committed SQL snapshot rows, stored payload SHA-1 consistency and JSON parsing within the window",
            "not_verified": [
                "source-to-SQL completeness",
                "global event deduplication",
                "primary merge equivalence",
                "transport receipts and durable channel queues",
                "unlisted or custom-table sinks",
                "archive restorability",
                "source semantic quality",
                "live readiness",
            ],
            "counting": "per physical database; primary and shard copies are not unique global events",
            "snapshot": "independent read transactions, not a cross-database atomic snapshot; late commits require another pass",
        },
        "budget": {
            "max_seconds": max_seconds,
            "max_payload_mib": max_payload_mib,
            "max_rows": max_rows,
        },
        "route_findings": findings,
        "databases": databases,
        "source_census": census,
        "authority": {
            "source_write": False,
            "cursor_advance": False,
            "route_mutation": False,
            "source_delete": False,
            "trading": False,
        },
    }
