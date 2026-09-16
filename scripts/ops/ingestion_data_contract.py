"""Read-only definitions assembled from the existing data-plane owners."""

from __future__ import annotations

import hashlib
import json
import math
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
from scripts.collector_contracts import declared_collector_definitions


def declared_intake_catalog(project_root: Path) -> dict[str, Any]:
    """Join owning declarations, never inspect producer payloads or run collectors."""
    rel = "config/collector_capability_catalog_v1.json"
    errors, sources, artifacts = [], [], []
    try:
        definitions = declared_collector_definitions()
    except (KeyError, TypeError, ValueError):
        definitions = []
        errors.append("invalid_collector_definitions")
    by_name = {}
    for definition in definitions:
        if not isinstance(definition, dict):
            errors.append("invalid_collector_definition")
            continue
        name = definition.get("name")
        if not isinstance(name, str) or not name or name in by_name:
            errors.append("invalid_or_duplicate_collector_name")
            continue
        by_name[name] = definition
        age = definition.get("freshness_minutes")
        command = definition.get("owner_command")
        if (
            type(age) not in (int, float)
            or not math.isfinite(age)
            or age <= 0
            or not isinstance(command, list)
            or not command
            or any(not isinstance(arg, str) or not arg for arg in command)
        ):
            errors.append(f"{name}:invalid_freshness_or_owner_command")
    catalog, raw = {}, b""
    route = inspect_storage_path(project_root / rel)
    try:
        if route["status"] != "present" or route.get("symlinks"):
            raise ValueError("catalog_route_unavailable")
        with (project_root / rel).open("rb") as handle:
            raw = handle.read(2 * 1024 * 1024 + 1)
        if len(raw) > 2 * 1024 * 1024:
            raise ValueError("catalog_size_limit")
        catalog = json.loads(raw)
        if (
            not isinstance(catalog, dict)
            or type(catalog.get("schema_version")) is not int
            or catalog.get("schema_version") != 1
        ):
            raise ValueError("catalog_schema_invalid")
        if not isinstance(catalog.get("producers"), list):
            raise ValueError("catalog_producers_invalid")
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
        catalog = {}

    def logical_path(value, label):
        if not isinstance(value, str) or not value:
            errors.append(f"{label}:missing_logical_path")
            return None
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            errors.append(f"{label}:nonlocal_logical_path")
            return None
        return {
            "relative_path": path.as_posix(),
            "logical_path": str(project_root / path),
            "observed": False,
        }

    seen, bound = set(), set()
    capabilities = set()
    planes = catalog.get("planes", [])
    if not isinstance(planes, list) or not planes:
        errors.append("invalid_capability_planes")
        planes = []
    for plane in planes:
        caps = plane.get("capabilities") if isinstance(plane, dict) else None
        if not isinstance(caps, list) or any(
            not isinstance(cap, str) or not cap for cap in caps
        ):
            errors.append("invalid_capability_plane")
        else:
            capabilities.update(caps)
    for producer in catalog.get("producers", []):
        if not isinstance(producer, dict):
            errors.append("invalid_producer_declaration")
            continue
        pid = producer.get("producer_id")
        if not isinstance(pid, str) or not pid or pid in seen:
            errors.append("invalid_or_duplicate_producer_id")
            continue
        seen.add(pid)
        caps = producer.get("capabilities")
        age = producer.get("max_age_minutes")
        if (
            not isinstance(caps, list)
            or not caps
            or any(not isinstance(cap, str) or not cap for cap in caps)
        ):
            errors.append(f"{pid}:invalid_capabilities")
        elif len(caps) != len(set(caps)) or not set(caps) <= capabilities:
            errors.append(f"{pid}:unknown_or_duplicate_capability")
        if type(age) not in (int, float) or not math.isfinite(age) or age <= 0:
            errors.append(f"{pid}:invalid_freshness")
        row = {
            "producer_id": pid,
            **{
                key: producer.get(key)
                for key in (
                    "producer_kind",
                    "source_kind",
                    "cadence",
                    "resource_class",
                    "max_age_minutes",
                    "capabilities",
                    "cache_contract",
                    "fallback_policy",
                    "capability_evidence_contract",
                    "capability_proofs",
                )
            },
            "handoff_owner": "core/collector_capability_routing.py",
            "raw_response_preservation": "not_established_by_declaration",
            "runtime_conformance_verified": False,
            "ingestion_lane_inferred": False,
        }
        kind = producer.get("producer_kind")
        if kind == "collector":
            name = producer.get("collector_name")
            if not isinstance(name, str) or name not in by_name or name in bound:
                errors.append(f"{pid}:unmatched_or_duplicate_collector_binding")
                continue
            bound.add(name)
            definition = by_name[name]
            row.update(
                data_class="declared_collector_snapshot",
                collector_contract=definition,
                payload=logical_path(definition.get("payload_path"), pid + ":payload"),
                health=logical_path(definition.get("health_path"), pid + ":health"),
            )
            sources.append(row)
        elif kind == "artifact":
            row.update(
                data_class="derived_artifact",
                owner_command=producer.get("owner_command"),
                owner_command_status=(
                    "declared"
                    if producer.get("owner_command")
                    else "not_declared_in_catalog"
                ),
                payload=logical_path(producer.get("artifact_path"), pid + ":artifact"),
            )
            artifacts.append(row)
        else:
            errors.append(f"{pid}:invalid_producer_kind")
    unmatched = sorted(set(by_name) - bound)
    errors.extend(f"{name}:collector_without_capability_binding" for name in unmatched)
    return {
        "definition_status": "defined" if not errors else "needs_attention",
        "definition_errors": errors,
        "collector_count": len(sources),
        "artifact_producer_count": len(artifacts),
        "collectors": sources,
        "artifact_producers": artifacts,
        "unmatched_collectors": unmatched,
        "definition_sources": ["scripts/collector_contracts.py", rel],
        "catalog_sha256": hashlib.sha256(raw).hexdigest() if raw else None,
        "collector_definitions_sha256": hashlib.sha256(
            json.dumps(definitions, sort_keys=True).encode()
        ).hexdigest(),
        "payloads_inspected": False,
        "new_scheduler": False,
        "live_execution_authority": False,
    }


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
    intake = declared_intake_catalog(root)
    policy["definition_errors"].extend(intake["definition_errors"])
    if policy["definition_errors"]:
        policy["definition_status"] = "needs_attention"
    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **policy,
        "declared_intake_catalog": intake,
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
        "data_classes": [
            {
                "class": "source_payload",
                "owner": "scripts/collector_contracts.py",
                "completion": "source-owned persistence and qualification; not SQL ingestion",
            },
            {
                "class": "append_only_evidence",
                "owner": "scripts/link_jsonl_to_sql.py",
                "completion": "committed jsonl_records plus validated source cursor; rejects remain separate",
            },
            {
                "class": "versioned_json_snapshot",
                "owner": "scripts/link_jsonl_to_sql.py",
                "completion": "json_file_records keyed by source_rel and payload_sha1; not a line cursor",
            },
            {
                "class": "durable_queue",
                "owner": "core/channel_queue.py",
                "completion": "queue-owner acknowledgment and destination commit; not a dispatch-index count",
            },
            {
                "class": "analytical_mirror",
                "owner": "scripts/ops/sql_analytics_mirror.py",
                "completion": "atomic DuckDB mirror generation; not the operational authority",
            },
            {
                "class": "sealed_history",
                "owner": "scripts/sql_hot_retention.py and cold archive owners",
                "completion": "manifest-bound stable bytes and verified restoration; not age alone",
            },
        ],
        "verification_contract": {
            "owner": "scripts/ops/ingestion_verification.py",
            "command": "ingestion-storage-control --verify-new-ingestion --since ISO_UTC --json",
            "artifact": "governance/health/ingestion_verification_latest.json",
            "window": "inclusive ingestion timestamp start and exclusive end; not market event time",
            "scope": "bounded indexed reads of discovered primary/shard jsonl_records and json_file_records",
            "proof": "per-database committed rows, stored payload hashes and parseability; not full source reconciliation",
            "incomplete": "missing index, route error, unreadable DB, deadline or payload/row cap cannot certify complete verification",
            "global_unique_event_count": False,
            "new_scheduler": False,
        },
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
