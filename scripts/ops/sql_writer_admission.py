"""Enforce the storage owner's pause for direct and scheduled ingestion writers."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from core import local_storage_reserve as reserve
from core.storage_router import inspect_storage_path
from scripts.ops.long_runtime_common import iso_now, write_payload
from core.status_label_contract import evidence_label, read_label_source
from core.write_path_recovery import durable_json


def storage_admission(project_root: Path) -> dict:
    receipt_path = (
        project_root / "governance/health/local_storage_reserve_guard_latest.json"
    )
    blockers = []
    policy = {}
    snapshot = {}
    try:
        if inspect_storage_path(project_root).get("status") != "present":
            raise ValueError("project_route_unavailable")
        route = inspect_storage_path(receipt_path).get("status")
        if route not in {"present", "missing"}:
            raise ValueError("storage_owner_route_unavailable")
        if route == "present":
            with receipt_path.open("rb") as handle:
                raw = handle.read(2 * 1024 * 1024 + 1)
            if len(raw) > 2 * 1024 * 1024:
                raise ValueError("storage_owner_receipt_oversized")
            owner = json.loads(raw)["local_storage_reserve"]
            for field in ("target", "pressure", "hard", "emergency"):
                key = f"{field}_free_gb"
                value = owner[key]
                if (
                    type(value) not in {int, float}
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError("invalid_storage_owner_policy")
                policy[key] = float(value)
        for field in ("target", "pressure", "hard", "emergency"):
            key = f"{field}_free_gb"
            default = getattr(reserve, f"DEFAULT_{field.upper()}_FREE_GB")
            configured = float(
                os.getenv(f"BOT_LOCAL_STORAGE_{field.upper()}_FREE_GB", default)
            )
            if not math.isfinite(configured) or configured <= 0:
                raise ValueError("invalid_storage_policy")
            policy[key] = max(policy.get(key, default), configured)
        if (
            not policy["target_free_gb"]
            >= policy["pressure_free_gb"]
            >= policy["hard_free_gb"]
            >= policy["emergency_free_gb"]
        ):
            raise ValueError("unordered_storage_policy")
        # Reuse only policy thresholds, never the receipt's old disk observation.
        snapshot = reserve.local_storage_reserve_contract(project_root, **policy)
        if snapshot["pause_nonessential_writers"]:
            blockers.append("local_storage_reserve_pause")
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        blockers.append(f"storage_admission_unavailable:{type(exc).__name__}")
    if str(
        os.getenv("SQL_LINK_SERVICE_PAUSED_FOR_LOCAL_STORAGE", "")
    ).strip().lower() in {"1", "true", "yes", "on"}:
        blockers.append("sql_writer_storage_pause_requested")
    return {
        "timestamp_utc": iso_now(),
        "ok": not blockers,
        "overall_status": "ready" if not blockers else "deferred",
        "reason": "" if not blockers else "local_storage_writer_admission",
        "blockers": blockers,
        "writer_start_allowed": not blockers,
        "local_storage_reserve": snapshot,
        "policy_source": str(receipt_path),
        "policy": "fresh disk observation; owner thresholds and explicit pauses cannot be relaxed by recovery requests or maintenance tokens",
    }


def defer_storage_writer(project_root: Path, *, owner: str) -> dict:
    payload = storage_admission(project_root)
    deferred = not payload["writer_start_allowed"]
    payload.update(
        owner=owner, deferred=deferred, running=False, rc=75 if deferred else 0
    )
    write_payload(
        project_root / "governance/health/sql_link_storage_admission_latest.json",
        payload,
    )
    publish_writer_observation(project_root, owner=owner, admission=payload)
    return payload if deferred else {}


def publish_writer_observation(
    project_root: Path, *, owner: str, admission: dict | None = None
) -> dict:
    from core.runtime_maintenance import maintenance_hold_snapshot

    admission = admission if admission is not None else storage_admission(project_root)
    hold = maintenance_hold_snapshot(project_root)
    progress = read_label_source(
        project_root, "governance/health/sql_link_service_progress_latest.json"
    )
    label = evidence_label(
        progress,
        scope="writer_progress",
        source="sql_link_service_progress_latest.json",
        max_age_seconds=300,
    )
    reasons = list(admission.get("blockers", []))
    if hold.get("active"):
        reasons.append("runtime_maintenance_hold_active")
    payload = {
        "timestamp_utc": iso_now(),
        "owner": owner,
        "overall_status": "deferred" if reasons else "observed",
        "reason": ",".join(reasons),
        "blockers": reasons,
        "storage_admission": admission,
        "writer_progress_evidence": label,
        "last_writer_progress_utc": label.get("observation_timestamp_utc"),
        "writer_progress_fresh": label["fresh"],
        "current_sql_failure_count": None,
        "scope": "fresh_admission_observation_not_writer_progress_or_sql_success",
        "authority": "observation_only_no_writer_start_or_hold_release",
    }
    durable_json(
        project_root,
        project_root / "governance/health/sql_writer_observation_latest.json",
        payload,
    )
    return payload


if __name__ == "__main__":
    print(
        json.dumps(publish_writer_observation(Path.cwd(), owner="scheduled_sql_writer"))
    )
