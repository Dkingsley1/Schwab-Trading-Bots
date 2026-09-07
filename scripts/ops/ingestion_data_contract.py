"""Read-only definitions assembled from the existing data-plane owners."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.collector_capability_routing import load_ingestion_routing_policy
from core.storage_router import (
    DEFAULT_LINK_DIRS,
    DEFAULT_LOCAL_FALLBACK,
    NESTED_SQLITE_ROUTE_RELS,
    _configured_external_project_root_no_io,
    inspect_storage_path,
)
from core.tiered_ingestion_lifecycle import load_lifecycle_policy

INGESTION_STAGES = (
    {
        "stage": "source_declared",
        "owner": "scripts/collector_contracts.py",
        "evidence": "collector identity, payload path, freshness budget, coverage and owner command",
        "does_not_prove": "a declared source is running or licensed for every use",
    },
    {
        "stage": "transport_fetched",
        "owner": "core/collector_transport.py",
        "evidence": "bounded response, request identity, payload digest and fetch timestamp",
        "does_not_prove": "HTTP success or a fetch watermark is SQL durability or event-time usability",
    },
    {
        "stage": "event_qualified",
        "owner": "collector schema validators and core/event_time_control.py",
        "evidence": "schema, provenance, source event time and stateful event-time acceptance",
        "does_not_prove": "all legacy collectors invoke event-time qualification; invocation evidence is required",
    },
    {
        "stage": "capability_routed",
        "owner": "core/collector_capability_routing.py",
        "evidence": "family-specific capability coverage, quality, freshness, lineage and route receipt",
        "does_not_prove": "paper evidence meets independent live evidence requirements",
    },
    {
        "stage": "sql_checkpoint_committed",
        "owner": "scripts/link_jsonl_to_sql.py",
        "evidence": "SQL commit precedes line/byte/inode checkpoint; inspect inserted, invalid and ops-write-failure counts",
        "does_not_prove": "every consumed line was accepted, or shard rows have merged into the primary",
    },
    {
        "stage": "merge_confirmed",
        "owner": "scripts/ops/sql_link_shard_manager.py",
        "evidence": "writer-owned merge progress; acknowledge applicable dispatch work only after merge confirmation",
        "does_not_prove": "dispatch queue status alone independently verifies a SQL merge",
    },
    {
        "stage": "archive_verified",
        "owner": "scripts/storage_tier_policy.py",
        "evidence": "sealed input, manifest, stable source fingerprint, size/digest verification and restore proof",
        "does_not_prove": "file age, an archive name, or a successful copy grants source deletion authority",
    },
)

ROUTE_ROLES = {
    "logs": ("runtime_logs", "runtime producers and guarded retention"),
    "decisions": (
        "decision_evidence",
        "decision producers and scripts/storage_tier_policy.py",
    ),
    "decision_explanations": (
        "explanation_evidence",
        "explanation producers and scripts/storage_tier_policy.py",
    ),
    "governance": ("control_evidence", "per-artifact control owner"),
    "exports": (
        "shared_source_payloads_and_exports",
        "source collectors and export owners",
    ),
    "data": ("mixed_data_root", "per-file writer owner"),
    "models": ("model_artifacts", "training and promotion owners"),
    "data/jsonl_link.sqlite3": (
        "active_sql_primary",
        "scripts/ops/sql_link_shard_manager.py",
    ),
    "data/bot_channel_queue.sqlite3": (
        "durable_channel_queue",
        "channel producers and SQL writer",
    ),
    "data/snapshot_context.sqlite3": (
        "shared_context_snapshot",
        "snapshot context writer",
    ),
    "data/sql_link_shards": (
        "active_sql_shards",
        "scripts/ops/sql_link_shard_manager.py",
    ),
    "data/jsonl_link_archives": (
        "archive_directory_may_contain_mutable_files",
        "scripts/sql_hot_retention.py",
    ),
    "data/deep_cold": (
        "manifest_backed_cold_data",
        "scripts/ops/deep_cold_storage_layer.py",
    ),
    "governance/ops_data_plane.sqlite3": (
        "transport_and_ingestion_receipts",
        "scripts/ops_data_plane.py",
    ),
    "governance/queues/ingestion_priority_queue.sqlite3": (
        "bounded_dispatch_index",
        "scripts/ops/ingestion_priority_queue.py",
    ),
}


def _policy_definition(project_root: Path) -> dict[str, Any]:
    routing_rel = "config/sleeve_ingestion_routing_v2.json"
    lifecycle_rel = "config/tiered_ingestion_lifecycle_v1.json"
    routing_path = inspect_storage_path(project_root / routing_rel)
    routing = (
        load_ingestion_routing_policy(Path(str(routing_path["resolved_path"])))
        if routing_path["status"] == "present"
        else {}
    )
    errors: list[str] = []
    lanes = routing.get("lane_contracts")
    families = routing.get("family_routes")
    if routing.get("schema_version") != 2:
        errors.append("missing_or_unsupported_ingestion_routing_policy")
    if not isinstance(lanes, dict) or set(lanes) != {"core", "deferred", "cold"}:
        errors.append("invalid_lane_contracts")
        lanes = {}
    for lane, contract in lanes.items():
        if not isinstance(contract, dict) or not all(
            contract.get(field)
            for field in (
                "priority",
                "latency_class",
                "storage_temperature",
                "backpressure_policy",
                "failure_policy",
            )
        ):
            errors.append(f"incomplete_lane_contract:{lane}")
    if not isinstance(families, dict) or not families:
        errors.append("missing_family_routes")
        families = {}
    for family, route in families.items():
        if (
            not isinstance(route, dict)
            or not isinstance(route.get("lane"), str)
            or route["lane"] not in lanes
        ):
            errors.append(f"undefined_family_lane:{family}")
    try:
        lifecycle_path = inspect_storage_path(project_root / lifecycle_rel)
        if lifecycle_path["status"] != "present":
            raise ValueError("lifecycle policy is unavailable or protected")
        lifecycle = load_lifecycle_policy(Path(str(lifecycle_path["resolved_path"])))
    except (OSError, ValueError, TypeError) as exc:
        lifecycle = {}
        errors.append(f"invalid_lifecycle_policy:{type(exc).__name__}")
    return {
        "definition_status": "defined" if not errors else "needs_attention",
        "definition_errors": errors,
        "routing_policy_path": routing_rel,
        "routing_policy_sha256": (
            hashlib.sha256(
                json.dumps(routing, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if routing
            else ""
        ),
        "lane_contracts": lanes,
        "family_routes": {
            name: {
                key: route.get(key) for key in ("lane", "cadence", "degradation_policy")
            }
            for name, route in families.items()
            if isinstance(route, dict)
        },
        "transport_contract": routing.get("transport_contract", {}),
        "lifecycle_policy_path": lifecycle_rel,
        "lifecycle_policy_sha256": lifecycle.get("policy_sha256", ""),
        "lifecycle_thresholds": lifecycle.get("thresholds", {}),
        "lifecycle_authority": lifecycle.get("authority", {}),
    }


def build_data_plane_definition(project_root: Path) -> dict[str, Any]:
    """Do not import writer modules, open databases, scan trees, or apply policy."""
    root = Path(project_root).absolute()
    local = Path(
        os.getenv("BOT_LOGS_LOCAL_FALLBACK_ROOT", str(root / DEFAULT_LOCAL_FALLBACK))
    ).expanduser()
    external = _configured_external_project_root_no_io()
    roots = {
        "local_fallback": inspect_storage_path(local),
        "configured_external": inspect_storage_path(external),
        "project_local": inspect_storage_path(root),
    }
    paths = dict.fromkeys((*DEFAULT_LINK_DIRS, *NESTED_SQLITE_ROUTE_RELS, *ROUTE_ROLES))
    observations = []
    for rel in paths:
        row = inspect_storage_path(root / rel)
        base = rel.removesuffix("-wal").removesuffix("-shm")
        role, owner = ROUTE_ROLES[base]
        location = "unassigned"
        if row["status"] == "present":
            resolved = Path(str(row["resolved_path"]))
            for name, observed_root in roots.items():
                if observed_root["status"] == "present" and resolved.is_relative_to(
                    str(observed_root["resolved_path"])
                ):
                    location = name
                    break
        row.update(
            relative_path=rel, role=role, writer_owner=owner, observed_location=location
        )
        if rel != base:
            row.update(
                role="sqlite_sidecar", database_route=base, absence_can_be_normal=True
            )
        observations.append(row)
    findings = [
        {"relative_path": row["relative_path"], "status": row["status"]}
        for row in observations
        if row["status"] not in {"present", "missing"}
        or (
            row["status"] == "missing"
            and row["symlinks"]
            and not row.get("absence_can_be_normal")
        )
    ]
    by_path = {row["relative_path"]: row for row in observations}
    for row in observations:
        if row["role"] != "sqlite_sidecar" or row["status"] != "present":
            continue
        database = by_path[row["database_route"]]
        suffix = row["relative_path"][len(row["database_route"]) :]
        if (
            database["status"] != "present"
            or row["resolved_path"] != database["resolved_path"] + suffix
        ):
            findings.append(
                {
                    "relative_path": row["relative_path"],
                    "status": "sqlite_sidecar_route_mismatch",
                    "database_route": row["database_route"],
                }
            )
    policy = _policy_definition(root)
    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **policy,
        "observation_scope": "bounded canonical paths only; directories do not certify descendants; no process environment or integrity audit",
        "configured_intent": {
            "environment_scope": "calling process; opsctl loads the runtime profile; running daemons may differ",
            "prefer_external": os.getenv("BOT_LOGS_PREFER_EXTERNAL", "1")
            .strip()
            .lower()
            not in {"0", "false", "no", "off"},
            "active_mode_hint": os.getenv("BOT_LOGS_ACTIVE_MODE", ""),
            "local_root": str(local),
            "external_root": str(external),
        },
        "route_observations": observations,
        "route_observation_findings": findings,
        "ingestion_stages": [dict(stage) for stage in INGESTION_STAGES],
        "durability_contract": {
            "http_fetch_watermark_is_sql_checkpoint": False,
            "exactly_once_end_to_end_claimed": False,
            "jsonl_sql_dedupe_key": ["source_file", "line_no"],
            "resume_identity": [
                "source_rel",
                "file_inode",
                "last_line",
                "last_offset_bytes",
            ],
            "dispatch_queue_is_complete_backlog_inventory": False,
            "transport_receipt_persistence": "best_effort; skipped or failed writes are not durable proof",
            "replay_rule": "validate source identity and boundary against committed progress; do not advance from HTTP success",
        },
        "control_owners": {
            "route_mutation": "core/storage_router.py via scripts/ops/storage_switch_orchestrator.py",
            "intake_throttles": "scripts/ops/ingestion_storage_governor.py",
            "backlog_accounting": "scripts/ingestion_backpressure_guard.py and scripts/ops/ingestion_storage_control.py",
            "local_reserve": "core/local_storage_reserve.py",
            "sqlite_reclaim": "scripts/ops/sqlite_reclaim_control.py via guarded sqlite_maintenance_launchd.sh",
            "standby_retirement": "scripts/ops/storage_standby_prune.py",
        },
        "authority": {
            "route_mutation": False,
            "intake_throttle": False,
            "source_delete": False,
            "paper_execution": False,
            "live_execution": False,
        },
    }
