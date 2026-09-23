#!/usr/bin/env python3
"""Bounded producer-age and repair-owner census. Never executes reported commands."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.storage_router import inspect_storage_path
from core.accountability import safe_write_json_atomic
from core.status_label_contract import evidence_label
from scripts.ops.ingestion_data_contract import declared_intake_catalog

OUT = "governance/health/self_healing_gap_audit_latest.json"


def read(root, relative, max_bytes=2 * 1024**2):
    path = root / relative
    route = inspect_storage_path(path, boundary_root=root, allow_external=False)
    if route.get("status") != "present" or route.get("symlinks"):
        return {}, "missing_or_nonlocal"
    try:
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                return {}, "not_regular"
            raw = handle.read(max_bytes + 1)
        if len(raw) > max_bytes:
            return {}, "size_budget"
        payload = json.loads(raw)
        return (payload, "") if isinstance(payload, dict) else ({}, "invalid_object")
    except (ValueError, OSError):
        return {}, "unreadable"


def age(payload, now, max_age_seconds=86400):
    label = evidence_label(payload, scope="artifact_census", source="producer",
                           max_age_seconds=max_age_seconds, now=now)
    seconds = label["age_seconds"]
    state = label["evidence_status"]
    if state == "stale" and max_age_seconds == 86400:
        state = "stale_over_24h"
    return (round(seconds / 3600, 6) if seconds is not None else None), state


def commands(payload):
    rows, pending, visits = [], [("", payload)], 0
    while pending and visits < 10000:
        prefix, node = pending.pop()
        visits += 1
        if isinstance(node, dict):
            for key, value in node.items():
                field = f"{prefix}.{key}".lstrip(".")
                if (
                    key in {"owner_command", "next_best_command"}
                    and isinstance(value, (str, list))
                    and value
                ):
                    rows.append({"field": field, "command": value})
                elif isinstance(value, (dict, list)):
                    pending.append((field, value))
        elif isinstance(node, list):
            pending.extend(
                (f"{prefix}[{i}]", value)
                for i, value in enumerate(node[:1000])
                if isinstance(value, (dict, list))
            )
    return rows


def build(root: Path, *, now=None, seconds=10):
    now = now or datetime.now(timezone.utc)
    start = time.monotonic()
    health = root / "governance/health"
    route = inspect_storage_path(health, boundary_root=root, allow_external=False)
    if route.get("status") != "present" or route.get("symlinks"):
        return {
            "ok": False,
            "reason": "health_route_unavailable",
            "live_execution_authority": False,
        }
    rows, refs, incomplete = [], [], []
    paths = sorted(health.glob("*_latest.json"))
    for path in paths[:2000]:
        if path.name == Path(OUT).name:
            continue
        if time.monotonic() - start >= seconds:
            incomplete.append("census_deadline")
            break
        rel = str(path.relative_to(root))
        payload, error = read(root, rel)
        hours, state = age(payload, now) if not error else (None, error)
        rows.append({"path": rel, "age_hours": hours, "state": state})
        refs.extend({"artifact": rel, **row} for row in commands(payload))
    if len(paths) > 2000:
        incomplete.append("file_count_budget")
    catalog = declared_intake_catalog(root)
    artifacts = []
    for row in catalog["artifact_producers"]:
        payload, error = read(root, row["payload"]["relative_path"])
        hours, state = age(payload, now, row["max_age_minutes"] * 60) if not error else (None, error)
        artifacts.append(
            {
                "producer_id": row["producer_id"],
                "owner_command": row["owner_command"],
                "age_hours": hours,
                "state": state,
                "max_age_minutes": row["max_age_minutes"],
                "owner_refresh_overdue": state != "fresh",
                "disposition": "refresh_through_registered_owner" if row["owner_command"] else "owner_mapping_required",
                "deletion_authorized": False,
                "missing_owner_alert": not bool(row["owner_command"]),
            }
        )
    collectors = [
        {
            "producer_id": row["producer_id"],
            "proof_declared": bool(
                row.get("capability_evidence_contract") or row.get("capability_proofs")
            ),
            "proof_requirement": (
                "runtime_evidence_required"
                if row.get("capability_evidence_contract")
                or row.get("collector_contract", {}).get("required")
                else "advisory_only_not_execution_evidence"
            ),
            "runtime_conformance_verified": False,
            "scope": "declaration_only_use_capability_materialization_for_runtime_proof",
        }
        for row in catalog["collectors"]
    ]
    storage, _ = read(root, "governance/health/local_storage_reserve_guard_latest.json")
    route_report, _ = read(root, "governance/health/storage_failback_sync_latest.json")
    cold, _ = read(root, "governance/health/deep_cold_storage_layer_latest.json")
    stale = [row for row in rows if row["state"] != "fresh"]
    return {
        "timestamp_utc": now.isoformat(),
        "ok": not incomplete,
        "overall_status": (
            "incomplete" if incomplete else "needs_attention" if stale else "ready"
        ),
        "scan_complete": not incomplete,
        "incomplete_reasons": incomplete,
        "inspected_count": len(rows),
        "stale_or_unverified_count": len(stale),
        "artifacts": rows,
        "repair_command_references": refs,
        "repair_commands_executed": False,
        "stale_artifact_policy": "refresh_current_evidence_through_owners; retain_unknown_and_historical_evidence; only_manifest_retention_owner_may_delete",
        "circularity_review": "references_are_inventory_not_proof_of_circular_preconditions",
        "artifact_producers": artifacts,
        "collectors": collectors,
        "sql_overlay": [
            row
            for row in rows
            if Path(row["path"]).name.startswith("jsonl_sql_ingestion_health_")
        ],
        "storage_pause": storage.get("local_storage_reserve", {}),
        "route_verification": route_report.get("route_verification")
        or {
            "verification_state": "not_observed",
            "reason": route_report.get("reason", "producer_missing"),
        },
        "deep_cold": {
            "mode": cold.get("policy", {}).get("second_cold_move_policy"),
            "movement": cold.get("second_cold_move", {}),
            "summary": cold.get("summary", {}),
        },
        "lock_contract": "backpressure_drainer_fleet and storage_backpressure_autopilot use flock; lock file existence is not ownership; never unlink held locks",
        "declaration_is_runtime_proof": False,
        "fresh_timestamp_is_ingestion_proof": False,
        "live_execution_authority": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = build(ROOT)
    destination = ROOT / OUT
    route = inspect_storage_path(destination, boundary_root=ROOT, allow_external=False)
    if route.get("status") not in {"present", "missing"} or route.get("symlinks"):
        raise ValueError("unsafe_audit_output")
    if (
        safe_write_json_atomic(
            str(destination),
            result,
            project_root=str(ROOT),
            source="self_healing_gap_audit",
        )
        is False
    ):
        result.update(ok=False, reason="audit_publication_failed")
    print(json.dumps(result, indent=None if args.json else 2))
    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
