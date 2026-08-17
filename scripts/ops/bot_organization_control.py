#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.bot_organization import canonical_hash, organize_registry
    from core.hierarchical_ensemble import aggregate_shadow_votes
    from scripts.ops.long_runtime_common import iso_now, load_json, ordered_unique, write_payload
else:
    from core.bot_organization import canonical_hash, organize_registry
    from core.hierarchical_ensemble import aggregate_shadow_votes
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "bot_organization_v1.json"
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "master_bot_registry.json"
DEFAULT_CATALOG_INPUT_PATH = PROJECT_ROOT / "core" / "bot_catalog.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "bot_organization_latest.json"
DEFAULT_HIERARCHY_OUT_PATH = PROJECT_ROOT / "governance" / "bot_organization" / "bot_hierarchy_latest.json"


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _shadow_integrity_self_test(policy: dict[str, Any]) -> dict[str, Any]:
    matching_profile = {
        "schema_version": 1,
        "profile_id": "self_test_bull_normal_liquid",
        "scope": "market_signal",
        "axes": {
            "market_direction": {
                "values": ["bull_trend"],
                "not_applicable": False,
            },
            "volatility_state": {
                "values": ["normal"],
                "not_applicable": False,
            },
            "liquidity_state": {
                "values": ["normal"],
                "not_applicable": False,
            },
        },
    }
    assignments = {
        "alpha_a": {
            "sleeve_id": "equity",
            "sub_sleeve_id": "trend",
            "cohort_id": "daily_all_regimes",
            "correlation_cluster_id": "equity/trend/daily",
            "shadow_vote_eligible": True,
            "regime_profile": matching_profile,
        },
        "alpha_a_duplicate": {
            "sleeve_id": "equity",
            "sub_sleeve_id": "trend",
            "cohort_id": "daily_all_regimes",
            "correlation_cluster_id": "equity/trend/daily",
            "shadow_vote_eligible": True,
            "regime_profile": matching_profile,
        },
        "alpha_b": {
            "sleeve_id": "equity",
            "sub_sleeve_id": "mean_reversion",
            "cohort_id": "intraday_all_regimes",
            "correlation_cluster_id": "equity/mean_reversion/intraday",
            "shadow_vote_eligible": True,
            "regime_profile": matching_profile,
        },
    }
    baseline = aggregate_shadow_votes(
        [
            {"vote_id": "a", "bot_id": "alpha_a", "score": 0.7, "confidence": 0.9, "weight": 1.0},
            {"vote_id": "b", "bot_id": "alpha_b", "score": 0.4, "confidence": 0.9, "weight": 1.0},
        ],
        assignments,
        policy,
    )
    duplicated = aggregate_shadow_votes(
        [
            {"vote_id": "a", "bot_id": "alpha_a", "score": 0.7, "confidence": 0.9, "weight": 1.0},
            {
                "vote_id": "a2",
                "bot_id": "alpha_a_duplicate",
                "score": 0.7,
                "confidence": 0.9,
                "weight": 1.0,
            },
            {"vote_id": "b", "bot_id": "alpha_b", "score": 0.4, "confidence": 0.9, "weight": 1.0},
        ],
        assignments,
        policy,
    )
    regime_votes = [
        {"vote_id": "a", "bot_id": "alpha_a", "score": 0.7, "confidence": 0.9, "weight": 1.0},
        {"vote_id": "b", "bot_id": "alpha_b", "score": 0.4, "confidence": 0.9, "weight": 1.0},
    ]
    regime_matching = aggregate_shadow_votes(
        regime_votes,
        assignments,
        policy,
        regime_context={
            "axes": {
                "market_direction": ["bull_trend"],
                "volatility_state": ["normal"],
                "liquidity_state": ["normal"],
            }
        },
    )
    regime_mismatch = aggregate_shadow_votes(
        regime_votes,
        assignments,
        policy,
        regime_context={
            "axes": {
                "market_direction": ["bear_trend"],
                "volatility_state": ["crisis"],
                "liquidity_state": ["dislocated"],
            }
        },
    )
    duplicate_invariant = abs(float(baseline.get("score", 0.0)) - float(duplicated.get("score", 0.0))) < 1e-12
    authority_locked = bool(
        baseline.get("authority", {}).get("paper_execution_authority") is False
        and baseline.get("authority", {}).get("live_execution_authority") is False
        and baseline.get("authority", {}).get("order_payload_created") is False
    )
    regime_filter_ready = bool(
        regime_matching.get("accepted_vote_count") == 2
        and regime_matching.get("regime_compatible_vote_count") == 2
        and regime_mismatch.get("accepted_vote_count") == 0
        and regime_mismatch.get("regime_incompatible_vote_count") == 2
        and regime_mismatch.get("authority", {}).get("paper_execution_authority") is False
        and regime_mismatch.get("authority", {}).get("live_execution_authority") is False
    )
    return {
        "ok": bool(duplicate_invariant and authority_locked and regime_filter_ready),
        "duplicate_cluster_invariant": duplicate_invariant,
        "authority_locked": authority_locked,
        "regime_filter_ready": regime_filter_ready,
        "regime_matching_accepted_vote_count": regime_matching.get("accepted_vote_count"),
        "regime_mismatch_accepted_vote_count": regime_mismatch.get("accepted_vote_count"),
        "baseline_score": baseline.get("score"),
        "duplicated_score": duplicated.get("score"),
        "baseline_action": baseline.get("action"),
        "duplicated_action": duplicated.get("action"),
    }


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    config_path: Path | None = None,
    registry_path: Path | None = None,
    catalog_input_path: Path | None = None,
    hierarchy_out_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    project_root = project_root.resolve()
    config_path = config_path or project_root / "config" / DEFAULT_CONFIG_PATH.name
    registry_path = registry_path or project_root / DEFAULT_REGISTRY_PATH.name
    catalog_input_path = catalog_input_path or project_root / "core" / DEFAULT_CATALOG_INPUT_PATH.name
    hierarchy_out_path = hierarchy_out_path or project_root / "governance" / "bot_organization" / DEFAULT_HIERARCHY_OUT_PATH.name
    policy = load_json(config_path)
    registry = load_json(registry_path)
    catalog = load_json(catalog_input_path)
    result = organize_registry(
        registry,
        policy,
        catalog=catalog,
        project_root=project_root,
    )
    assignments = list(result.pop("assignments", []))
    self_test = _shadow_integrity_self_test(policy)
    blockers = ordered_unique(
        list(result.get("blockers") or [])
        + (["hierarchical_shadow_integrity_self_test_failed"] if not self_test["ok"] else [])
    )
    ok = not blockers
    receipt_input = {
        "policy_sha256": _sha256(config_path),
        "registry_sha256": _sha256(registry_path),
        "catalog_input_sha256": _sha256(catalog_input_path),
        "assignment_receipt_sha256": result.get("assignment_receipt_sha256"),
        "shadow_integrity_self_test": self_test,
    }
    receipt = canonical_hash(receipt_input)
    hierarchy_catalog = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "policy_id": str(policy.get("policy_id") or ""),
        "operating_mode": str(policy.get("operating_mode") or ""),
        "registry_bot_count": result.get("registry_bot_count"),
        "assignment_count": len(assignments),
        "assignment_receipt_sha256": result.get("assignment_receipt_sha256"),
        "input_receipts": receipt_input,
        "hierarchy_levels": list((policy.get("hierarchy") or {}).get("levels") or []),
        "regime_model_id": str((policy.get("regime_model") or {}).get("model_id") or ""),
        "regime_model_contract": {
            "mode": str((policy.get("regime_model") or {}).get("mode") or ""),
            "axis_ids": [
                str(row.get("axis_id") or "")
                for row in (policy.get("regime_model") or {}).get("axes", [])
                if isinstance(row, dict)
            ],
            "compatibility_mode": str(
                ((policy.get("regime_model") or {}).get("compatibility_policy") or {}).get("mode")
                or ""
            ),
            "scenario_partition_version": str(
                ((policy.get("regime_model") or {}).get("scenario_partition_contract") or {}).get(
                    "version"
                )
                or ""
            ),
            "scenario_partition_mode": str(
                ((policy.get("regime_model") or {}).get("scenario_partition_contract") or {}).get(
                    "mode"
                )
                or ""
            ),
            "metadata_access_version": str(
                ((policy.get("regime_model") or {}).get("metadata_access_contract") or {}).get(
                    "version"
                )
                or ""
            ),
            "metadata_access_mode": str(
                ((policy.get("regime_model") or {}).get("metadata_access_contract") or {}).get(
                    "mode"
                )
                or ""
            ),
            "paper_execution_authority": False,
            "live_execution_authority": False,
        },
        "assignments": assignments,
        "authority_contract": {
            "metadata_only": True,
            "paper_execution_authority": False,
            "live_execution_authority": False,
        },
    }
    health = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": ok,
        "overall_status": str(result.get("overall_status") or ("ready" if ok else "blocked")),
        "grade": str(result.get("grade") or ("A+" if ok else "F")),
        "policy_id": str(policy.get("policy_id") or ""),
        **{key: value for key, value in result.items() if key not in {"ok", "overall_status", "grade", "blockers"}},
        "blockers": blockers,
        "hierarchy_catalog": {
            "path": str(hierarchy_out_path),
            "assignment_count": len(assignments),
            "assignment_receipt_sha256": result.get("assignment_receipt_sha256"),
        },
        "hierarchy_contract": {
            "levels": list((policy.get("hierarchy") or {}).get("levels") or []),
            "one_assignment_per_registered_bot": bool(result.get("unique_assignment_ratio") == 1.0),
            "full_registry_coverage": bool(result.get("organization_coverage_ratio") == 1.0),
            "classification_provenance_recorded": all(bool(row.get("provenance")) for row in assignments),
            "correlation_cluster_recorded": all(bool(row.get("correlation_cluster_id")) for row in assignments),
            "multi_axis_regime_profile_recorded": all(
                bool(row.get("regime_profile_id")) and bool(row.get("regime_profile"))
                for row in assignments
            ),
            "declared_scenario_partitions_are_bounded": all(
                not bool(row.get("regime_scenario_partitioned", False))
                or (
                    int(row.get("regime_scenario_count", 0) or 0) >= 2
                    and not list(
                        ((row.get("regime_profile") or {}).get("scenario_contract_errors") or [])
                    )
                )
                for row in assignments
            ),
            "regime_metadata_access_recorded": all(
                bool((row.get("regime_metadata_access") or {}).get("access_ready"))
                for row in assignments
            ),
        },
        "regime_model_contract": {
            "model_id": str((policy.get("regime_model") or {}).get("model_id") or ""),
            "mode": str((policy.get("regime_model") or {}).get("mode") or ""),
            "axis_ids": [
                str(row.get("axis_id") or "")
                for row in (policy.get("regime_model") or {}).get("axes", [])
                if isinstance(row, dict)
            ],
            "scope_counts": (result.get("counts") or {}).get("regime_scopes") or {},
            "axis_coverage_ratio": result.get("regime_axis_coverage_ratio"),
            "axis_specificity_ratio": result.get("regime_axis_specificity_ratio"),
            "quality_grade": result.get("regime_quality_grade"),
            "review_count": result.get("regime_review_count"),
            "scenario_profile_count": result.get("regime_scenario_profile_count"),
            "scenario_count": result.get("regime_scenario_count"),
            "scenario_review_count": result.get("regime_scenario_review_count"),
            "invalid_scenario_profile_count": result.get(
                "invalid_regime_scenario_profile_count"
            ),
            "overbroad_profile_count": result.get("overbroad_regime_profile_count"),
            "compatibility_mode": str(
                ((policy.get("regime_model") or {}).get("compatibility_policy") or {}).get("mode")
                or ""
            ),
            "scenario_partition_version": str(
                ((policy.get("regime_model") or {}).get("scenario_partition_contract") or {}).get(
                    "version"
                )
                or ""
            ),
            "metadata_access_version": str(
                ((policy.get("regime_model") or {}).get("metadata_access_contract") or {}).get(
                    "version"
                )
                or ""
            ),
            "metadata_access_mode": str(
                ((policy.get("regime_model") or {}).get("metadata_access_contract") or {}).get(
                    "mode"
                )
                or ""
            ),
            "metadata_access_ready_count": result.get(
                "regime_metadata_access_ready_count"
            ),
            "metadata_access_ratio": result.get("regime_metadata_access_ratio"),
            "paper_execution_authority": False,
            "live_execution_authority": False,
        },
        "resource_budget_contract": policy.get("resource_budgets") or {},
        "ensemble_contract": policy.get("ensemble_policy") or {},
        "admission_contract": policy.get("admission_policy") or {},
        "safety_contract": policy.get("safety_contract") or {},
        "shadow_integrity_self_test": self_test,
        "evidence_epoch": {
            "id": f"bot-organization:{receipt[:16]}",
            "receipt_sha256": receipt,
            **receipt_input,
        },
        "recommended_actions": ordered_unique(
            [
                "review low-confidence bot assignments before changing runtime routing"
                if result.get("review_queue_count")
                else "",
                "replace unknown or overbroad regime axes with evidence-backed registry metadata"
                if result.get("regime_review_count")
                else "",
                "rank marginal contribution and park excess shadow voters in oversubscribed cells"
                if result.get("oversubscribed_shadow_cells")
                else "",
                "keep the hierarchy in shadow mode until locked replay proves post-cost improvement",
            ]
        ),
    }
    return health, hierarchy_catalog


def _resolve(project_root: Path, raw: Path | None, default: str) -> Path:
    path = raw or Path(default)
    return path if path.is_absolute() else project_root / path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and validate the production-grade hierarchical bot organization catalog."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--catalog-input", type=Path)
    parser.add_argument("--out-file", type=Path)
    parser.add_argument("--hierarchy-out", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    config_path = _resolve(project_root, args.config, "config/bot_organization_v1.json")
    registry_path = _resolve(project_root, args.registry, "master_bot_registry.json")
    catalog_input_path = _resolve(project_root, args.catalog_input, "core/bot_catalog.json")
    out_path = _resolve(project_root, args.out_file, "governance/health/bot_organization_latest.json")
    hierarchy_out_path = _resolve(
        project_root,
        args.hierarchy_out,
        "governance/bot_organization/bot_hierarchy_latest.json",
    )
    health, hierarchy = build_payload(
        project_root,
        config_path=config_path,
        registry_path=registry_path,
        catalog_input_path=catalog_input_path,
        hierarchy_out_path=hierarchy_out_path,
    )
    write_payload(hierarchy_out_path, hierarchy)
    write_payload(out_path, health)
    if args.json:
        print(json.dumps(health, ensure_ascii=True))
    else:
        print(
            "bot_organization_control "
            f"status={health['overall_status']} grade={health['grade']} "
            f"organized={health['organized_bot_count']}/{health['registry_bot_count']} "
            f"review={health['review_queue_count']}"
        )
    return 0 if health["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
