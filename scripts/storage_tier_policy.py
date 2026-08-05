#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from link_jsonl_to_sql import (
    _ingestion_lane_label,
    _storage_temperature_label,
    _storage_tier_label,
    discover_jsonl_files,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCAN_ROOTS = (
    "decisions",
    "decision_explanations",
    "exports/paper_broker_bridge",
    "governance/events",
    "governance/channels",
    "data/sql_link_shards",
    "governance/content_store",
    "data/stale_stage",
    "data/deep_cold",
)
DEFAULT_STORAGE_SUFFIXES = {
    ".json",
    ".jsonl",
    ".sqlite",
    ".sqlite3",
    ".db",
    ".wal",
    ".shm",
    ".log",
    ".txt",
}
DEFAULT_OFFLOAD_MANIFEST_PATH = PROJECT_ROOT / "governance" / "health" / "storage_tier_offload_manifest_latest.json"
GIB = 1024**3


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _truthy(raw: Any, default: bool = False) -> bool:
    if isinstance(raw, bool):
        return raw
    text = str(raw or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _status_ready(payload: dict[str, Any]) -> bool:
    return bool(payload.get("ready", False)) or str(payload.get("status") or "").strip().lower() == "ready"


def _collector_intake_soak_safe(
    *,
    ingestion_payload: dict[str, Any],
    ingestion_inputs: dict[str, Any],
    backlog_relief_clear: bool,
) -> tuple[bool, dict[str, Any]]:
    audit = (
        ingestion_payload.get("collector_intake_enforcement_audit")
        if isinstance(ingestion_payload.get("collector_intake_enforcement_audit"), dict)
        else {}
    )
    status = (
        str(ingestion_inputs.get("collector_intake_status") or audit.get("status") or "")
        .strip()
        .lower()
    )
    required = bool(audit.get("required", False))
    mismatch_count = int(max(_safe_float(audit.get("mismatch_count"), 0.0), 0.0))
    enforced = status == "enforced"
    safely_optional = bool(
        status == "not_required"
        and not required
        and mismatch_count <= 0
        and backlog_relief_clear
    )
    return bool(enforced or safely_optional), {
        "status": status,
        "required": required,
        "mismatch_count": mismatch_count,
        "enforced": enforced,
        "safely_optional": safely_optional,
        "backlog_relief_clear": bool(backlog_relief_clear),
    }


def _managed_hot_path_budget_contract(
    *,
    project_root: Path,
    configured_hot_budget_bytes: int,
    live_hot_path_bytes: int,
    by_service_role: dict[str, dict[str, int]],
    active_explanation_buffer_bytes: int = 0,
) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    retention = _load_json(health_root / "storage_retention_unison_latest.json")
    ingestion = _load_json(health_root / "ingestion_storage_control_latest.json")

    retention_contract = retention.get("continuous_run_contract") if isinstance(retention.get("continuous_run_contract"), dict) else {}
    ingestion_contract = (
        ingestion.get("continuous_run_soak_contract")
        if isinstance(ingestion.get("continuous_run_soak_contract"), dict)
        else {}
    )
    ingestion_inputs = ingestion_contract.get("inputs") if isinstance(ingestion_contract.get("inputs"), dict) else {}
    storage_efficiency = (
        ingestion.get("storage_efficiency_contract")
        if isinstance(ingestion.get("storage_efficiency_contract"), dict)
        else {}
    )

    retention_ready = _status_ready(retention_contract)
    ingestion_ready = _status_ready(ingestion_contract)
    storage_efficiency_ready = (
        str(ingestion_inputs.get("storage_efficiency_status") or storage_efficiency.get("overall_status") or "")
        .strip()
        .lower()
        == "ready"
    )
    backlog_relief_clear = not _truthy(ingestion_inputs.get("backlog_relief_active"), False)
    collector_soak_safe, collector_details = _collector_intake_soak_safe(
        ingestion_payload=ingestion,
        ingestion_inputs=ingestion_inputs,
        backlog_relief_clear=backlog_relief_clear,
    )

    margin_gb = max(
        _safe_float(retention_contract.get("available_margin_gb"), 0.0),
        _safe_float((ingestion_contract.get("forecast") or {}).get("continuous_run_margin_gb") if isinstance(ingestion_contract.get("forecast"), dict) else 0.0, 0.0),
        0.0,
    )
    hot_lane = _load_json(health_root / "hot_lane_retention_control_latest.json")
    hot_lane_mode = str(hot_lane.get("mode") or "").strip()
    hot_lane_status = str(hot_lane.get("overall_status") or "").strip()
    hot_lane_active = bool(
        hot_lane.get("ok", False)
        and hot_lane_mode in {"thin_optional_sub_bot_decisions", "emergency_hot_thin"}
        and hot_lane_status in {"active", "critical", "watching", "ready"}
    )
    explanation_allowance_bytes = int(
        max(_safe_float(os.getenv("STORAGE_TIER_ACTIVE_EXPLANATION_BUFFER_ALLOWANCE_GB"), 16.0), 0.0)
        * float(GIB)
    )
    protected_explanation_bytes = (
        min(max(int(active_explanation_buffer_bytes), 0), explanation_allowance_bytes)
        if hot_lane_active
        else 0
    )
    protected_hot_floor_bytes = int(
        (by_service_role.get("live_decisioning") or {}).get("bytes", 0)
        + (by_service_role.get("stateful_sql") or {}).get("bytes", 0)
        + protected_explanation_bytes
    )
    raw_over_budget_bytes = max(int(live_hot_path_bytes) - int(configured_hot_budget_bytes), 0)
    blockers = [
        "storage_retention_continuous_run_not_ready" if not retention_ready else "",
        "ingestion_continuous_run_not_ready" if not ingestion_ready else "",
        "storage_efficiency_not_ready" if not storage_efficiency_ready else "",
        "collector_intake_not_soak_safe" if not collector_soak_safe else "",
        "backlog_relief_active" if not backlog_relief_clear else "",
    ]
    blockers = [item for item in blockers if item]

    margin_ratio = max(_safe_float(os.getenv("STORAGE_TIER_MANAGED_HOT_MARGIN_RATIO"), 0.25), 0.0)
    allowance_cap_gb = max(_safe_float(os.getenv("STORAGE_TIER_MANAGED_HOT_ALLOWANCE_CAP_GB"), 96.0), 0.0)
    managed_allowance_bytes = 0
    if not blockers:
        managed_allowance_bytes = int(min(margin_gb * float(GIB) * margin_ratio, allowance_cap_gb * float(GIB)))
    effective_hot_budget_bytes = max(int(configured_hot_budget_bytes), protected_hot_floor_bytes + managed_allowance_bytes)
    controlled_over_budget_bytes = max(int(live_hot_path_bytes) - int(effective_hot_budget_bytes), 0)

    if blockers:
        status = "fixed_budget"
    elif controlled_over_budget_bytes > 0:
        status = "managed_over_budget"
    elif raw_over_budget_bytes > 0:
        status = "managed_ready"
    else:
        status = "ready"

    return {
        "active": not bool(blockers),
        "status": status,
        "configured_hot_budget_bytes": int(configured_hot_budget_bytes),
        "effective_hot_budget_bytes": int(effective_hot_budget_bytes),
        "protected_hot_floor_bytes": int(protected_hot_floor_bytes),
        "managed_margin_allowance_bytes": int(managed_allowance_bytes),
        "active_explanation_buffer_contract": {
            "hot_lane_active": bool(hot_lane_active),
            "hot_lane_mode": hot_lane_mode,
            "active_bytes": max(int(active_explanation_buffer_bytes), 0),
            "allowance_bytes": int(explanation_allowance_bytes),
            "protected_bytes": int(protected_explanation_bytes),
            "within_allowance": bool(int(active_explanation_buffer_bytes) <= explanation_allowance_bytes),
            "policy": "bounded current-day explanation writes are protected only while targeted hot-lane containment is active",
        },
        "live_hot_path_bytes": int(live_hot_path_bytes),
        "raw_hot_path_over_budget_bytes": int(raw_over_budget_bytes),
        "hot_path_over_budget_bytes": int(controlled_over_budget_bytes),
        "continuous_run_margin_gb": round(float(margin_gb), 3),
        "margin_ratio": float(margin_ratio),
        "allowance_cap_gb": float(allowance_cap_gb),
        "blockers": blockers,
        "inputs": {
            "retention_continuous_run_ready": bool(retention_ready),
            "ingestion_continuous_run_ready": bool(ingestion_ready),
            "storage_efficiency_ready": bool(storage_efficiency_ready),
            "collector_intake_enforced": bool(collector_details.get("enforced", False)),
            "collector_intake_soak_safe": bool(collector_soak_safe),
            "collector_intake": collector_details,
            "backlog_relief_clear": bool(backlog_relief_clear),
        },
        "policy": "raw fixed hot budget remains visible; continuous-run-ready systems use protected floor plus bounded margin allowance for blocking decisions",
    }


def _path_size_bytes(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except Exception:
        return 0


def _economic_value(source_rel: str) -> str:
    rel = str(source_rel or "")
    if rel.startswith("data/deep_cold/"):
        return "medium"
    if rel.startswith("data/stale_stage/governance_telemetry_compactor/"):
        return "medium"
    if rel.startswith("decisions/") or rel.startswith("exports/paper_broker_bridge/"):
        return "critical"
    if rel.startswith("decision_explanations/"):
        return "high"
    if rel.startswith("governance/"):
        return "medium"
    if rel.startswith("data/"):
        return "medium"
    return "low"


def _path_family(source_rel: str) -> str:
    rel = str(source_rel or "")
    if rel.startswith("data/deep_cold/"):
        return "deep_cold_archive"
    if rel.startswith("data/stale_stage/governance_telemetry_compactor/"):
        return "deep_cold_archive"
    if rel.startswith("decisions/"):
        return "decisions"
    if rel.startswith("decision_explanations/"):
        return "decision_explanations"
    if rel.startswith("data/sql_link_shards/"):
        return "sql_link_shards"
    if rel.startswith("governance/content_store/"):
        return "content_store"
    if rel.startswith("governance/events/"):
        return "governance_events"
    if rel.startswith("governance/channels/"):
        return "governance_channels"
    if rel.startswith("exports/paper_broker_bridge/"):
        return "paper_bridge"
    if rel.startswith("data/stale_stage/"):
        return "stale_stage"
    if rel.startswith("data/"):
        return "data"
    if rel.startswith("governance/"):
        return "governance"
    if rel.startswith("exports/"):
        return "exports"
    return "other"


def _service_role(source_rel: str) -> str:
    rel = str(source_rel or "")
    if rel.startswith("data/deep_cold/") or rel.startswith("data/stale_stage/governance_telemetry_compactor/"):
        return "deep_cold_archive"
    if rel.startswith("decisions/") or rel.startswith("exports/paper_broker_bridge/"):
        return "live_decisioning"
    if rel.startswith("decision_explanations/"):
        return "explainability"
    if rel.startswith("data/sql_link_shards/"):
        return "stateful_sql"
    if rel.startswith("governance/content_store/"):
        return "artifact_store"
    if rel.startswith("governance/events/") or rel.startswith("governance/channels/"):
        return "governance_telemetry"
    if rel.startswith("data/stale_stage/"):
        return "staging_reaper"
    return "analytics"


def _storage_semantic_overrides(path: Path, source_rel: str) -> dict[str, str]:
    """Keep archives and quarantine targets out of the live-write budget."""
    rel = str(source_rel or "").replace("\\", "/")
    try:
        resolved = str(path.resolve(strict=False)).replace("\\", "/")
    except Exception:
        resolved = str(path).replace("\\", "/")

    if any(marker in resolved for marker in ("/quarantine/", "/data/stale_stage/")):
        return {
            "temperature": "cold",
            "storage_tier": "archive_cold",
            "ingestion_lane": "deferred_lane",
            "economic_value": "medium",
            "family": "stale_stage",
            "service_role": "staging_reaper",
        }

    if rel.startswith("decision_explanations/") and (
        path.name.endswith(".gz") or path.name.startswith("latest_decisions.log")
    ):
        return {
            "temperature": "cool",
            "storage_tier": "compatibility_cool",
            "ingestion_lane": "nearline_lane",
            "economic_value": "high",
            "family": "decision_explanations",
            "service_role": "explainability_archive",
        }
    return {}


def _is_current_day_explanation(path: Path, source_rel: str, tokens: set[str]) -> bool:
    rel = str(source_rel or "").replace("\\", "/")
    return bool(
        rel.startswith("decision_explanations/")
        and path.name.endswith(".jsonl")
        and not any(marker in path.name for marker in (".local_fallback", ".tmp.", ".compact_pending"))
        and any(token in path.name for token in tokens)
    )


def _recommended_action(*, role: str, value: str, lane: str) -> str:
    if role == "deep_cold_archive":
        return "keep_indexed_in_deep_cold_manifest"
    if role == "stateful_sql":
        return "compact_or_mirror_sqlite_shards"
    if role == "explainability":
        return "offload_explanation_history_to_cold_tier"
    if role == "artifact_store":
        return "garbage_collect_or_externalize_artifact_blobs"
    if role == "governance_telemetry":
        return "archive_governance_telemetry"
    if role == "staging_reaper":
        return "reap_stale_stage_artifacts"
    if value == "critical" or lane == "hot_lane":
        return "keep_on_hot_path"
    return "move_to_async_cold_path"


def _is_storage_file(path: Path, *, project_root: Path) -> bool:
    if not path.is_file():
        return False
    try:
        rel = str(path.relative_to(project_root))
    except Exception:
        rel = str(path)
    if rel.startswith("governance/content_store/"):
        return True
    lowered_name = path.name.lower()
    if lowered_name.endswith((".sqlite3-wal", ".sqlite3-shm", ".db-wal", ".db-shm")):
        return True
    suffixes = {suffix.lower() for suffix in path.suffixes}
    return bool(suffixes & DEFAULT_STORAGE_SUFFIXES)


def discover_storage_files(project_root: Path) -> list[Path]:
    seen: dict[str, Path] = {}
    for path in discover_jsonl_files(project_root):
        seen[str(path.resolve())] = path
    for rel_root in DEFAULT_SCAN_ROOTS:
        root = project_root / rel_root
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not _is_storage_file(path, project_root=project_root):
                continue
            seen[str(path.resolve())] = path
    return sorted(seen.values(), key=lambda path: str(path))


def _candidate_priority(row: dict[str, Any]) -> tuple[int, int, str]:
    action = str(row.get("recommended_action") or "")
    role = str(row.get("service_role") or "")
    priority = {
        "garbage_collect_or_externalize_artifact_blobs": 5,
        "offload_explanation_history_to_cold_tier": 4,
        "archive_governance_telemetry": 3,
        "compact_or_mirror_sqlite_shards": 2,
        "reap_stale_stage_artifacts": 1,
        "keep_indexed_in_deep_cold_manifest": 0,
    }.get(action, 0)
    role_bonus = {
        "artifact_store": 3,
        "explainability": 2,
        "governance_telemetry": 1,
        "stateful_sql": 1,
        "deep_cold_archive": 0,
    }.get(role, 0)
    return (priority, role_bonus, str(row.get("relative_path") or ""))


def _date_age_days(source_rel: str, *, now_utc: datetime) -> int | None:
    matches = re.findall(r"(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)", str(source_rel or ""))
    if not matches:
        return None
    year, month, day = matches[-1]
    try:
        parsed = datetime(int(year), int(month), int(day), tzinfo=timezone.utc).date()
    except Exception:
        return None
    return int((now_utc.date() - parsed).days)


def _offload_target_rel(source_rel: str, *, family: str) -> str:
    cleaned = str(source_rel or "").lstrip("/")
    if family and cleaned.startswith(f"{family}/"):
        return f"data/deep_cold/manifest_backed/{cleaned}"
    return f"data/deep_cold/manifest_backed/{family}/{cleaned}"


def _manifest_entry(
    row: dict[str, Any],
    *,
    project_root: Path,
    now_utc: datetime,
    min_bytes: int,
) -> dict[str, Any]:
    rel = str(row.get("relative_path") or "")
    role = str(row.get("service_role") or "")
    value = str(row.get("economic_value") or "")
    family = str(row.get("family") or "")
    action = str(row.get("recommended_action") or "")
    size_bytes = int(row.get("size_bytes", 0) or 0)
    age_days = _date_age_days(rel, now_utc=now_utc)
    source_path = project_root / rel
    stat: dict[str, Any] = {"size_bytes": size_bytes}
    try:
        info = source_path.stat()
        stat.update(
            {
                "mtime_ns": int(info.st_mtime_ns),
                "inode": int(info.st_ino),
            }
        )
    except Exception:
        stat.update({"mtime_ns": 0, "inode": 0})

    blockers: list[str] = []
    classification = "deferred_manifest_review"
    allowed_actions: list[str] = ["read_only"]
    delete_allowed = False
    proof_required = {
        "pre_copy_stat_fingerprint": False,
        "post_copy_size_match": False,
        "post_copy_sha256_match": False,
        "restore_probe": False,
        "retention_gate_before_source_delete": True,
    }

    if role == "deep_cold_archive":
        classification = "manifest_index_only"
        allowed_actions = ["refresh_deep_cold_manifest", "restore_probe_sample"]
    elif role == "stateful_sql":
        classification = "stateful_sql_compaction_only"
        allowed_actions = ["sqlite_checkpoint", "sqlite_vacuum_or_incremental_vacuum", "mirror_verified_copy"]
        blockers.append("stateful_sql_requires_checkpoint_or_mirror")
    elif value == "critical" or role == "live_decisioning":
        classification = "keep_hot_critical"
        allowed_actions = ["read_only", "hot_lane_retention_watch"]
        blockers.append("critical_live_decisioning_stays_hot")
    elif role == "staging_reaper":
        classification = "cleanup_review_required"
        allowed_actions = ["refresh_retention_manifest", "tiered_cleanup_with_retention_gate"]
        blockers.append("stale_stage_delete_requires_retention_owner")
    elif action in {
        "offload_explanation_history_to_cold_tier",
        "archive_governance_telemetry",
        "garbage_collect_or_externalize_artifact_blobs",
        "move_to_async_cold_path",
    }:
        if size_bytes < min_bytes:
            classification = "below_manifest_candidate_floor"
            blockers.append("below_manifest_candidate_min_bytes")
        elif age_days is not None and age_days <= 0:
            classification = "current_day_hot_path_hold"
            blockers.append("current_day_file_stays_hot")
        else:
            classification = "eligible_manifest_backed_offload"
            allowed_actions = [
                "copy_to_cold_tier",
                "verify_size",
                "verify_sha256",
                "write_restore_proof",
                "source_delete_requires_retention_gate",
            ]
            proof_required = {
                "pre_copy_stat_fingerprint": True,
                "post_copy_size_match": True,
                "post_copy_sha256_match": True,
                "restore_probe": True,
                "retention_gate_before_source_delete": True,
            }
    elif size_bytes < min_bytes:
        classification = "below_manifest_candidate_floor"
        blockers.append("below_manifest_candidate_min_bytes")

    return {
        "relative_path": rel,
        "planned_cold_relative_path": _offload_target_rel(rel, family=family),
        "size_bytes": size_bytes,
        "family": family,
        "service_role": role,
        "economic_value": value,
        "recommended_action": action,
        "classification": classification,
        "age_days": age_days,
        "source_stat_fingerprint": stat,
        "allowed_actions": allowed_actions,
        "delete_allowed_by_policy": delete_allowed,
        "proof_required": proof_required,
        "blockers": blockers,
    }


def _manifest_summary(entries: list[dict[str, Any]], *, omitted_count: int) -> dict[str, Any]:
    buckets: dict[str, dict[str, int]] = {}
    for entry in entries:
        key = str(entry.get("classification") or "unknown")
        bucket = buckets.setdefault(key, {"files": 0, "bytes": 0})
        bucket["files"] += 1
        bucket["bytes"] += int(entry.get("size_bytes", 0) or 0)
    eligible = buckets.get("eligible_manifest_backed_offload", {})
    compaction = buckets.get("stateful_sql_compaction_only", {})
    keep_hot = buckets.get("keep_hot_critical", {})
    cleanup_review = buckets.get("cleanup_review_required", {})
    return {
        "entry_count": len(entries),
        "omitted_count": int(omitted_count),
        "by_classification": buckets,
        "eligible_offload_files": int(eligible.get("files", 0)),
        "eligible_offload_bytes": int(eligible.get("bytes", 0)),
        "compaction_only_files": int(compaction.get("files", 0)),
        "compaction_only_bytes": int(compaction.get("bytes", 0)),
        "keep_hot_files": int(keep_hot.get("files", 0)),
        "cleanup_review_files": int(cleanup_review.get("files", 0)),
    }


def _build_offload_manifest_contract(
    *,
    project_root: Path,
    rows: list[dict[str, Any]],
    now_utc: datetime,
    manifest_path: Path,
    min_bytes: int,
    max_files: int,
    max_bytes: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            -_candidate_priority(row)[0],
            -_candidate_priority(row)[1],
            -int(row.get("size_bytes", 0) or 0),
            str(row.get("relative_path") or ""),
        ),
    )
    entries: list[dict[str, Any]] = []
    included_bytes = 0
    for row in ordered:
        if len(entries) >= max(int(max_files), 1):
            break
        size_bytes = int(row.get("size_bytes", 0) or 0)
        if included_bytes + size_bytes > max(int(max_bytes), 1) and entries:
            break
        entry = _manifest_entry(row, project_root=project_root, now_utc=now_utc, min_bytes=min_bytes)
        entries.append(entry)
        included_bytes += size_bytes

    summary = _manifest_summary(entries, omitted_count=max(len(ordered) - len(entries), 0))
    eligible_files = int(summary.get("eligible_offload_files", 0) or 0)
    compaction_files = int(summary.get("compaction_only_files", 0) or 0)
    status = "planned" if eligible_files or compaction_files else "watch"
    contract = {
        "mode": "manifest_backed_hot_path_offload_compaction_v1",
        "status": status,
        "manifest_path": str(manifest_path),
        "policy_script_is_read_only": True,
        "apply_owner": "storage-retention-unison",
        "eligible_offload_files": eligible_files,
        "eligible_offload_gb": round(int(summary.get("eligible_offload_bytes", 0) or 0) / (1024.0**3), 4),
        "compaction_only_files": compaction_files,
        "compaction_only_gb": round(int(summary.get("compaction_only_bytes", 0) or 0) / (1024.0**3), 4),
        "delete_requires": [
            "verified_cold_copy",
            "sha256_match",
            "restore_probe",
            "retention_gate",
            "non_critical_non_stateful_classification",
        ],
        "never_delete_classes": [
            "keep_hot_critical",
            "stateful_sql_compaction_only",
            "manifest_index_only",
            "current_day_hot_path_hold",
        ],
        "stateful_sql_policy": "checkpoint, vacuum, incremental vacuum, or verified mirror only; never source-delete from this policy",
        "target_policy": "cold target must not resolve to a protected volume and must write a restore-proof manifest before any source delete is considered",
        "next_action": (
            "run storage-retention-unison --apply for bounded compactors; run manifest-backed-offload --apply only after BOT_SECOND_COLD_ROOT is ready"
            if status == "planned"
            else "keep refreshing storage-tier-policy until offload or compaction candidates appear"
        ),
    }
    manifest_payload = {
        "timestamp_utc": now_utc.isoformat(),
        "schema_version": 1,
        "project_root": str(project_root),
        "contract": contract,
        "summary": summary,
        "entries": entries,
    }
    return contract, manifest_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize hot-path debt and cold-tier candidates across active storage lanes.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--hot-budget-gb", type=float, default=25.0)
    parser.add_argument("--cold-candidate-min-mb", type=float, default=128.0)
    parser.add_argument("--offload-manifest-file", default=str(DEFAULT_OFFLOAD_MANIFEST_PATH))
    parser.add_argument("--offload-manifest-max-files", type=int, default=5000)
    parser.add_argument("--offload-manifest-max-gb", type=float, default=512.0)
    parser.add_argument("--offload-manifest-min-mb", type=float, default=128.0)
    parser.add_argument("--no-write-offload-manifest", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    now_utc = datetime.now(timezone.utc)
    files = discover_storage_files(project_root)
    by_temperature: dict[str, dict[str, int]] = {}
    by_storage_tier: dict[str, dict[str, int]] = {}
    by_lane: dict[str, dict[str, int]] = {}
    by_value: dict[str, dict[str, int]] = {}
    by_family: dict[str, dict[str, int]] = {}
    by_service_role: dict[str, dict[str, int]] = {}
    all_rows: list[dict[str, Any]] = []
    cold_path_candidates: list[dict[str, Any]] = []
    async_offload_bytes = 0
    live_hot_path_bytes = 0
    active_explanation_buffer_bytes = 0
    today_tokens = {now_utc.strftime("%Y%m%d")}
    try:
        today_tokens.add(now_utc.astimezone().strftime("%Y%m%d"))
    except Exception:
        pass
    candidate_min_bytes = max(int(float(args.cold_candidate_min_mb) * 1024 * 1024), 1)

    for path in files:
        try:
            rel = str(path.relative_to(project_root))
        except Exception:
            rel = str(path)
        size_bytes = _path_size_bytes(path)
        temperature = _storage_temperature_label(rel)
        tier = _storage_tier_label(rel)
        lane = _ingestion_lane_label(rel)
        value = _economic_value(rel)
        family = _path_family(rel)
        service_role = _service_role(rel)
        semantic_overrides = _storage_semantic_overrides(path, rel)
        temperature = semantic_overrides.get("temperature", temperature)
        tier = semantic_overrides.get("storage_tier", tier)
        lane = semantic_overrides.get("ingestion_lane", lane)
        value = semantic_overrides.get("economic_value", value)
        family = semantic_overrides.get("family", family)
        service_role = semantic_overrides.get("service_role", service_role)
        row = {
            "relative_path": rel,
            "size_bytes": int(size_bytes),
            "temperature": temperature,
            "storage_tier": tier,
            "ingestion_lane": lane,
            "economic_value": value,
            "family": family,
            "service_role": service_role,
            "recommended_action": _recommended_action(role=service_role, value=value, lane=lane),
        }
        for bucket, key in (
            (by_temperature, temperature),
            (by_storage_tier, tier),
            (by_lane, lane),
            (by_value, value),
            (by_family, family),
            (by_service_role, service_role),
        ):
            entry = bucket.setdefault(key, {"files": 0, "bytes": 0})
            entry["files"] += 1
            entry["bytes"] += int(size_bytes)
        all_rows.append(row)
        if service_role in {"live_decisioning", "stateful_sql", "explainability"}:
            live_hot_path_bytes += int(size_bytes)
        if service_role == "explainability" and _is_current_day_explanation(path, rel, today_tokens):
            active_explanation_buffer_bytes += int(size_bytes)
        if (
            int(size_bytes) >= candidate_min_bytes
            and value != "critical"
            and row["recommended_action"] != "keep_on_hot_path"
            and service_role != "deep_cold_archive"
        ):
            cold_path_candidates.append(row)
            async_offload_bytes += int(size_bytes)

    all_rows.sort(key=lambda row: (-int(row.get("size_bytes", 0) or 0), str(row.get("relative_path") or "")))
    cold_path_candidates.sort(
        key=lambda row: (
            -_candidate_priority(row)[0],
            -_candidate_priority(row)[1],
            -int(row.get("size_bytes", 0) or 0),
            str(row.get("relative_path") or ""),
        )
    )
    configured_hot_budget_bytes = max(int(float(args.hot_budget_gb) * GIB), 1)
    budget_contract = _managed_hot_path_budget_contract(
        project_root=project_root,
        configured_hot_budget_bytes=configured_hot_budget_bytes,
        live_hot_path_bytes=int(live_hot_path_bytes),
        by_service_role=by_service_role,
        active_explanation_buffer_bytes=int(active_explanation_buffer_bytes),
    )
    hot_budget_bytes = int(budget_contract.get("effective_hot_budget_bytes", configured_hot_budget_bytes) or configured_hot_budget_bytes)
    hot_path_over_budget_bytes = int(budget_contract.get("hot_path_over_budget_bytes", 0) or 0)
    raw_hot_path_over_budget_bytes = int(
        budget_contract.get("raw_hot_path_over_budget_bytes", max(int(live_hot_path_bytes) - configured_hot_budget_bytes, 0))
        or 0
    )
    overall_status = "ready"
    if hot_path_over_budget_bytes > 0:
        overall_status = "degraded"
    if live_hot_path_bytes > hot_budget_bytes * 2:
        overall_status = "blocked"

    recommended_actions: list[str] = []
    if hot_path_over_budget_bytes > 0:
        recommended_actions.append(
            "trim live hot-path storage by offloading explanation, telemetry, and artifact-store payloads before they compete with decisions and SQLite writers"
        )
    elif raw_hot_path_over_budget_bytes > 0:
        recommended_actions.append(
            "keep bounded compaction and manifest-backed offload available; raw fixed hot-budget pressure is covered by the continuous-run margin contract"
        )
    if any(str(row.get("service_role") or "") == "stateful_sql" for row in all_rows[: max(int(args.top_n), 1)]):
        recommended_actions.append(
            "treat SQL link shards as stateful hot-path debt: compact, checkpoint, or mirror them instead of leaving the writer tier to absorb full historical growth"
        )
    if any(str(row.get("service_role") or "") == "artifact_store" for row in cold_path_candidates[: max(int(args.top_n), 1)]):
        recommended_actions.append(
            "run content-store GC aggressively so large immutable blobs move off the trading path first"
        )
    if any(str(row.get("service_role") or "") == "governance_telemetry" for row in cold_path_candidates[: max(int(args.top_n), 1)]):
        recommended_actions.append(
            "archive governance telemetry on an async cadence rather than keeping nearline event history on the same device as live writes"
        )

    manifest_path = Path(args.offload_manifest_file).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = project_root / manifest_path
    offload_contract, offload_manifest = _build_offload_manifest_contract(
        project_root=project_root,
        rows=all_rows,
        now_utc=now_utc,
        manifest_path=manifest_path,
        min_bytes=max(int(float(args.offload_manifest_min_mb) * 1024 * 1024), 1),
        max_files=max(int(args.offload_manifest_max_files), 1),
        max_bytes=max(int(float(args.offload_manifest_max_gb) * 1024 * 1024 * 1024), 1),
    )
    if not bool(args.no_write_offload_manifest):
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(offload_manifest, ensure_ascii=True, indent=2), encoding="utf-8")

    payload = {
        "timestamp_utc": now_utc.isoformat(),
        "schema_version": 2,
        "overall_status": overall_status,
        "file_count": len(files),
        "by_temperature": by_temperature,
        "by_storage_tier": by_storage_tier,
        "by_lane": by_lane,
        "by_economic_value": by_value,
        "by_family": by_family,
        "by_service_role": by_service_role,
        "top_files": all_rows[: max(int(args.top_n), 1)],
        "cold_path_candidates": cold_path_candidates[: max(int(args.top_n), 1)],
        "pressure": {
            "hot_files": int((by_temperature.get("hot") or {}).get("files", 0)),
            "hot_bytes": int((by_temperature.get("hot") or {}).get("bytes", 0)),
            "warm_bytes": int((by_temperature.get("warm") or {}).get("bytes", 0)),
            "cold_lane_bytes": int((by_lane.get("cold_lane") or {}).get("bytes", 0)),
            "deep_cold_bytes": int((by_family.get("deep_cold_archive") or {}).get("bytes", 0)),
            "live_hot_path_bytes": int(live_hot_path_bytes),
            "configured_hot_budget_bytes": int(configured_hot_budget_bytes),
            "hot_budget_bytes": int(hot_budget_bytes),
            "hot_path_over_budget_bytes": int(hot_path_over_budget_bytes),
            "raw_hot_path_over_budget_bytes": int(raw_hot_path_over_budget_bytes),
            "async_offload_candidate_bytes": int(async_offload_bytes),
        },
        "hot_path_budget_contract": budget_contract,
        "manifest_backed_offload_contract": offload_contract,
        "offload_manifest_summary": offload_manifest["summary"],
        "upgrade_plan": {
            "storage_split_target": "keep decisions and active SQL state on the hot path while explanations, telemetry, and artifact blobs drain asynchronously",
            "deep_cold_target": "stale-stage archives stay manifest-indexed as deep-cold evidence so retention-locked files stop acting like hot-path storage debt",
            "top_hot_path_families": sorted(
                [
                    {"family": key, **value}
                    for key, value in by_family.items()
                    if key in {"decisions", "decision_explanations", "sql_link_shards", "paper_bridge"}
                ],
                key=lambda row: (-int(row.get("bytes", 0) or 0), str(row.get("family") or "")),
            )[:5],
            "recommended_actions": recommended_actions,
        },
    }
    out = project_root / "governance" / "health" / "storage_tier_policy_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "storage_tier_policy file_count={files} hot_bytes={hot} warm_bytes={warm} cold_lane_bytes={cold}".format(
                files=payload["file_count"],
                hot=payload["pressure"]["hot_bytes"],
                warm=payload["pressure"]["warm_bytes"],
                cold=payload["pressure"]["cold_lane_bytes"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
