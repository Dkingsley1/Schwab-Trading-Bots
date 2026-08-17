#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


FOUNDER_BOT_ID = "brain_refinery_v1"
DNA_VERSION = "founder_dna_v1"
DNA_SCOPE = "brain_refinery_v1_to_full_fleet"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "bot_founder_dna_lineage_latest.json"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "governance" / "lineage" / "bot_founder_dna_manifest_latest.json"

DNA_TRAITS = (
    "market_data_observation",
    "paper_first_safety",
    "global_halt_awareness",
    "resource_throttle_awareness",
    "training_lineage",
    "decision_explanation_contract",
    "data_collection_before_training",
    "registry_auditable_identity",
)

TARGET_FUNCTION_CONTRACT = (
    "bot_founder_dna_lineage",
    "training_lineage_manifest",
    "platform_intelligence_expansion",
)

DATA_COLLECTION_CONTRACT = (
    "founder_dna_lineage_manifest",
    "bot_genome_trace",
    "first_bot_trait_contract",
    "cross_sleeve_inheritance_map",
)

STORAGE_CONTRACT = (
    "governance/lineage",
    "governance/health",
)

CORRELATION_CONTRACT = (
    "founder_dna_lineage",
    "registry_identity_contract",
)

EARLY_BOT_RE = re.compile(r"brain_refinery_v(\d+)\b")


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _normalize_id(raw: Any) -> str:
    return str(raw or "").strip()


def _safe_ratio(part: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(float(part) / float(total), 6)


def _registry_rows(registry: dict[str, Any]) -> list[dict[str, Any]]:
    rows = registry.get("sub_bots")
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _early_bot_ids_from_value(raw: Any) -> set[str]:
    text = json.dumps(raw, ensure_ascii=True, sort_keys=True) if isinstance(raw, (dict, list)) else str(raw or "")
    early: set[str] = set()
    for match in EARLY_BOT_RE.finditer(text):
        try:
            number = int(match.group(1))
        except Exception:
            continue
        if 1 <= number <= 107:
            early.add(f"brain_refinery_v{number}")
    return early


def _lineage_evidence(row: dict[str, Any]) -> dict[str, Any]:
    bot_id = _normalize_id(row.get("bot_id"))
    explicit = (
        _normalize_id(row.get("founder_bot_id")) == FOUNDER_BOT_ID
        and _normalize_id(row.get("founder_dna_version")) == DNA_VERSION
    )
    legacy_bridge = bool(_normalize_id(row.get("legacy_harmonization_scope")) or _normalize_id(row.get("legacy_harmonization_version")))

    evidence_fields = {
        key: row.get(key)
        for key in (
            "bot_id",
            "bot_role",
            "core_module_path",
            "target_functions",
            "data_intake_collections",
            "bootstrap_teacher_bot_ids",
            "teacher_bot_ids",
            "teacher_lineage",
            "lineage_teacher_bot_ids",
            "lineage_root_bot_id",
            "parent_bot_ids",
            "source_bot_ids",
            "correlation_peer_sleeves",
        )
        if key in row
    }
    early_ids = sorted(_early_bot_ids_from_value(evidence_fields))

    if bot_id == FOUNDER_BOT_ID:
        source = "self_founder"
        confidence = 1.0
    elif explicit:
        source = _normalize_id(row.get("founder_dna_source")) or "explicit_founder_contract"
        try:
            confidence = float(row.get("founder_dna_confidence"))
        except Exception:
            confidence = 0.95
    elif legacy_bridge:
        source = "explicit_legacy_bridge"
        confidence = 0.9
    elif early_ids:
        source = "early_bot_teacher_or_registry_reference"
        confidence = 0.82
    else:
        source = "registry_inferred_full_fleet_contract"
        confidence = 0.7

    return {
        "bot_id": bot_id,
        "explicit_founder_dna": explicit,
        "legacy_bridge": legacy_bridge,
        "early_lineage_bot_ids": early_ids,
        "source": source,
        "confidence": round(confidence, 3),
    }


def _contract_gaps(row: dict[str, Any]) -> list[str]:
    gaps: list[str] = []
    if _normalize_id(row.get("founder_bot_id")) != FOUNDER_BOT_ID:
        gaps.append("founder_bot_id")
    if _normalize_id(row.get("founder_dna_version")) != DNA_VERSION:
        gaps.append("founder_dna_version")
    if _normalize_id(row.get("founder_dna_scope")) != DNA_SCOPE:
        gaps.append("founder_dna_scope")
    existing_traits = set(str(item) for item in _as_list(row.get("founder_dna_traits")))
    missing_traits = [trait for trait in DNA_TRAITS if trait not in existing_traits]
    if missing_traits:
        gaps.append("founder_dna_traits")
    if not bool(row.get("lineage_guard_enabled", False)):
        gaps.append("lineage_guard_enabled")
    if _normalize_id(row.get("lineage_root_bot_id")) != FOUNDER_BOT_ID:
        gaps.append("lineage_root_bot_id")
    return gaps


def _stamp_row(row: dict[str, Any], *, timestamp_utc: str) -> bool:
    before = json.dumps(row, ensure_ascii=True, sort_keys=True)
    evidence = _lineage_evidence(row)
    bot_id = evidence["bot_id"]
    source = evidence["source"]

    row["founder_bot_id"] = FOUNDER_BOT_ID
    row["founder_dna_version"] = DNA_VERSION
    row["founder_dna_scope"] = DNA_SCOPE
    row["founder_dna_source"] = source
    row["founder_dna_confidence"] = evidence["confidence"]
    row["founder_dna_traits"] = list(DNA_TRAITS)
    row["founder_dna_inheritance_mode"] = "explicit_contract_metadata"
    if not _normalize_id(row.get("founder_dna_applied_utc")):
        row["founder_dna_applied_utc"] = timestamp_utc
    row["lineage_root_bot_id"] = FOUNDER_BOT_ID
    row["lineage_guard_enabled"] = True
    row["lineage_revalidation_command"] = "./scripts/ops/opsctl.sh bot-founder-dna --json"
    row["lineage_regression_guard"] = "fail_if_founder_dna_missing_or_stale"
    row["lineage_generation"] = 0 if bot_id == FOUNDER_BOT_ID else 1

    row["target_functions"] = ordered_unique([*map(str, _as_list(row.get("target_functions"))), *TARGET_FUNCTION_CONTRACT])
    row["data_intake_collections"] = ordered_unique(
        [*map(str, _as_list(row.get("data_intake_collections"))), *DATA_COLLECTION_CONTRACT]
    )
    row["storage_targets"] = ordered_unique([*map(str, _as_list(row.get("storage_targets"))), *STORAGE_CONTRACT])
    row["correlation_dependencies"] = ordered_unique(
        [*map(str, _as_list(row.get("correlation_dependencies"))), *CORRELATION_CONTRACT]
    )

    after = json.dumps(row, ensure_ascii=True, sort_keys=True)
    return before != after


def _lineage_rows(registry_rows: list[dict[str, Any]], *, max_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in registry_rows:
        bot_id = _normalize_id(row.get("bot_id"))
        if not bot_id:
            continue
        evidence = _lineage_evidence(row)
        gaps = _contract_gaps(row)
        rows.append(
            {
                "bot_id": bot_id,
                "bot_role": str(row.get("bot_role") or "unknown"),
                "active": bool(row.get("active", False)),
                "lifecycle_state": str(row.get("lifecycle_state") or ""),
                "explicit_founder_dna": evidence["explicit_founder_dna"],
                "legacy_bridge": evidence["legacy_bridge"],
                "lineage_source": evidence["source"],
                "lineage_confidence": evidence["confidence"],
                "early_lineage_bot_ids": evidence["early_lineage_bot_ids"],
                "contract_gaps": gaps,
            }
        )
    rows.sort(
        key=lambda row: (
            bool(row.get("explicit_founder_dna")),
            -len(_as_list(row.get("contract_gaps"))),
            str(row.get("bot_id") or ""),
        )
    )
    return rows[:max_rows]


def _summary(registry_rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_counts: Counter[str] = Counter()
    explicit_count = 0
    legacy_count = 0
    missing_contract_count = 0
    active_count = 0
    for row in registry_rows:
        if bool(row.get("active", False)):
            active_count += 1
        evidence = _lineage_evidence(row)
        source_counts[str(evidence["source"])] += 1
        if evidence["explicit_founder_dna"]:
            explicit_count += 1
        if evidence["legacy_bridge"]:
            legacy_count += 1
        if _contract_gaps(row):
            missing_contract_count += 1

    total = len(registry_rows)
    founder_present = any(_normalize_id(row.get("bot_id")) == FOUNDER_BOT_ID for row in registry_rows)
    coverage_ratio = _safe_ratio(explicit_count, total)
    if not founder_present:
        overall_status = "blocked"
    elif explicit_count == total and total > 0:
        overall_status = "ready"
    else:
        overall_status = "needs_work"

    return {
        "overall_status": overall_status,
        "founder_bot_id": FOUNDER_BOT_ID,
        "founder_dna_version": DNA_VERSION,
        "founder_dna_scope": DNA_SCOPE,
        "founder_present": founder_present,
        "total_bots": total,
        "active_bots": active_count,
        "explicit_founder_dna_count": explicit_count,
        "missing_founder_dna_count": max(total - explicit_count, 0),
        "missing_contract_count": missing_contract_count,
        "legacy_bridge_count": legacy_count,
        "coverage_ratio": coverage_ratio,
        "all_have_founder_dna": bool(total > 0 and explicit_count == total),
        "source_counts": dict(sorted(source_counts.items())),
        "trait_count": len(DNA_TRAITS),
    }


def build_payload(project_root: Path = PROJECT_ROOT, *, max_rows: int = 25) -> dict[str, Any]:
    registry_path = project_root / "master_bot_registry.json"
    health_artifact = project_root / "governance" / "health" / "bot_founder_dna_lineage_latest.json"
    manifest_artifact = project_root / "governance" / "lineage" / "bot_founder_dna_manifest_latest.json"
    registry = load_json(registry_path)
    rows = _registry_rows(registry)
    summary = _summary(rows)
    lineage_rows = _lineage_rows(rows, max_rows=max(max_rows, 1))

    sections = {
        "founder_contract": {
            "overall_status": "ready" if summary["founder_present"] else "blocked",
            "founder_bot_id": FOUNDER_BOT_ID,
            "founder_dna_version": DNA_VERSION,
            "founder_dna_scope": DNA_SCOPE,
            "founder_traits": list(DNA_TRAITS),
            "contract_mode": "metadata_lineage_no_runtime_loop",
        },
        "inheritance_coverage": {
            "overall_status": summary["overall_status"],
            "total_bots": summary["total_bots"],
            "explicit_founder_dna_count": summary["explicit_founder_dna_count"],
            "missing_founder_dna_count": summary["missing_founder_dna_count"],
            "coverage_ratio": summary["coverage_ratio"],
        },
        "lineage_sources": {
            "overall_status": "ready",
            "source_counts": summary["source_counts"],
            "sampled_lineage_rows": lineage_rows,
        },
        "regression_guard": {
            "overall_status": "ready" if summary["founder_present"] else "blocked",
            "guard_name": "bot_founder_dna_lineage_guard",
            "failure_condition": "any registry bot missing founder_bot_id, founder_dna_version, or lineage_guard_enabled",
            "command": "./scripts/ops/opsctl.sh bot-founder-dna --json",
        },
        "infrastructure_bot": {
            "overall_status": "ready",
            "infrastructure_bot_id": "brain_refinery_founder_dna_lineage_infrabot",
            "mode": "advisory_registry_guard",
            "writes_runtime_loops": False,
            "purpose": "keep bot ancestry, training lineage, and reporting DNA visible as the fleet expands",
        },
    }

    top_actions = []
    if not summary["founder_present"]:
        top_actions.append("restore brain_refinery_v1 before stamping founder DNA fleet-wide")
    elif summary["missing_founder_dna_count"]:
        top_actions.append("run ./scripts/ops/opsctl.sh bot-founder-dna --apply-registry --json to stamp the full fleet")
    else:
        top_actions.append("keep founder DNA lineage in dashboard refreshes before major bot expansions")

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": summary["overall_status"] in {"ready", "needs_work"},
        "overall_status": summary["overall_status"],
        "mode": "advisory_read_only_unless_apply_registry",
        "summary": summary,
        "sections": sections,
        "lineage_rows": lineage_rows,
        "top_actions": top_actions,
        "source_files": {
            "master_bot_registry": str(registry_path),
            "health_artifact": str(health_artifact),
            "manifest_artifact": str(manifest_artifact),
        },
    }


def apply_registry_contract(project_root: Path, *, timestamp_utc: str) -> dict[str, Any]:
    registry_path = project_root / "master_bot_registry.json"
    registry = load_json(registry_path)
    rows = _registry_rows(registry)
    backup_path = project_root / "governance" / "lineage" / f"master_bot_registry_before_founder_dna_{timestamp_utc.replace(':', '').replace('+', '_')}.json"
    write_payload(backup_path, registry)

    changed_rows = 0
    for row in rows:
        if _stamp_row(row, timestamp_utc=timestamp_utc):
            changed_rows += 1

    summary = _summary(rows)
    top_summary = _as_dict(registry.get("summary")).copy()
    top_summary["total_bots"] = len(rows)
    top_summary["founder_dna_covered_bots"] = summary["explicit_founder_dna_count"]
    top_summary["founder_dna_missing_bots"] = summary["missing_founder_dna_count"]
    top_summary["founder_dna_coverage_ratio"] = summary["coverage_ratio"]
    top_summary["founder_dna_version"] = DNA_VERSION
    registry["summary"] = top_summary
    registry["founder_dna_policy"] = {
        "updated_at_utc": timestamp_utc,
        "founder_bot_id": FOUNDER_BOT_ID,
        "founder_dna_version": DNA_VERSION,
        "founder_dna_scope": DNA_SCOPE,
        "coverage_ratio": summary["coverage_ratio"],
        "lineage_guard_required": True,
        "revalidation_command": "./scripts/ops/opsctl.sh bot-founder-dna --json",
    }
    master_policy = _as_dict(registry.get("master_policy")).copy()
    master_policy["founder_dna_lineage_required"] = True
    master_policy["founder_dna_version"] = DNA_VERSION
    master_policy["founder_dna_guard_command"] = "./scripts/ops/opsctl.sh bot-founder-dna --json"
    registry["master_policy"] = master_policy

    assignments = _as_dict(registry.get("backpressure_infrabot_assignments")).copy()
    assignments["bot_founder_dna_lineage"] = {
        "infrastructure_bot_id": "brain_refinery_founder_dna_lineage_infrabot",
        "command": "./scripts/ops/opsctl.sh bot-founder-dna --json",
        "purpose": "detect missing founder DNA lineage before bot admissions, retrains, or presentation reports",
        "cadence": "before_major_expansion_and_daily_dashboard",
        "writes_runtime_loops": False,
        "updated_at_utc": timestamp_utc,
    }
    registry["backpressure_infrabot_assignments"] = assignments
    registry["updated_at_utc"] = timestamp_utc

    write_payload(registry_path, registry)
    return {
        "applied": True,
        "changed_rows": changed_rows,
        "registry_path": str(registry_path),
        "backup_path": str(backup_path),
    }


def write_artifacts(project_root: Path, payload: dict[str, Any], *, out_file: Path) -> dict[str, str]:
    manifest_path = project_root / "governance" / "lineage" / "bot_founder_dna_manifest_latest.json"
    manifest = {
        "timestamp_utc": payload.get("timestamp_utc"),
        "schema_version": 1,
        "summary": payload.get("summary"),
        "lineage_rows": payload.get("lineage_rows") if isinstance(payload.get("lineage_rows"), list) else [],
        "founder_traits": list(DNA_TRAITS),
        "command": "./scripts/ops/opsctl.sh bot-founder-dna --json",
    }
    write_payload(out_file, payload)
    write_payload(manifest_path, manifest)
    return {
        "health_artifact": str(out_file),
        "manifest_artifact": str(manifest_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and optionally apply the founder-DNA lineage contract for every bot.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--max-rows", type=int, default=25)
    parser.add_argument("--apply-registry", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    out_file = Path(args.out_file).expanduser()
    timestamp = iso_now()
    apply_result = {"applied": False, "changed_rows": 0}
    if args.apply_registry:
        apply_result = apply_registry_contract(project_root, timestamp_utc=timestamp)

    payload = build_payload(project_root, max_rows=max(int(args.max_rows), 1))
    payload["apply_result"] = apply_result
    payload["artifact_paths"] = write_artifacts(project_root, payload, out_file=out_file)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        summary = _as_dict(payload.get("summary"))
        print(
            "bot_founder_dna_lineage "
            f"status={payload.get('overall_status', '')} "
            f"bots={int(summary.get('total_bots', 0) or 0)} "
            f"coverage={float(summary.get('coverage_ratio', 0.0) or 0.0):.3f} "
            f"missing={int(summary.get('missing_founder_dna_count', 0) or 0)} "
            f"changed={int(apply_result.get('changed_rows', 0) or 0)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
