#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.bot_organization import canonical_hash, organize_registry
    from core.bot_definition_contracts import audit_definitions, render_audit_markdown
    from core.hierarchical_ensemble import aggregate_shadow_votes
    from scripts.ops.long_runtime_common import (
        iso_now,
        load_json,
        ordered_unique,
        write_payload,
        write_text_atomic,
    )
else:
    from core.bot_organization import canonical_hash, organize_registry
    from core.bot_definition_contracts import audit_definitions, render_audit_markdown
    from core.hierarchical_ensemble import aggregate_shadow_votes
    from .long_runtime_common import (
        PROJECT_ROOT,
        iso_now,
        load_json,
        ordered_unique,
        write_payload,
        write_text_atomic,
    )


from core.bot_operating_definitions import (
    CATALOG_PATH as OPERATING_CATALOG_PATH,
    POLICY_PATH as OPERATING_POLICY_PATH,
    compile_catalog as compile_operating_catalog,
    expand_definition,
    validate_catalog as validate_operating_catalog,
)
from core.bot_process_definitions import expand_definition as expand_process_definition
from core.status_label_contract import bot_definition_labels

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "bot_organization_v1.json"
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "master_bot_registry.json"
DEFAULT_CATALOG_INPUT_PATH = PROJECT_ROOT / "core" / "bot_catalog.json"
DEFAULT_OUT_PATH = (
    PROJECT_ROOT / "governance" / "health" / "bot_organization_latest.json"
)
DEFAULT_HIERARCHY_OUT_PATH = (
    PROJECT_ROOT / "governance" / "bot_organization" / "bot_hierarchy_latest.json"
)


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
            {
                "vote_id": "a",
                "bot_id": "alpha_a",
                "score": 0.7,
                "confidence": 0.9,
                "weight": 1.0,
            },
            {
                "vote_id": "b",
                "bot_id": "alpha_b",
                "score": 0.4,
                "confidence": 0.9,
                "weight": 1.0,
            },
        ],
        assignments,
        policy,
    )
    duplicated = aggregate_shadow_votes(
        [
            {
                "vote_id": "a",
                "bot_id": "alpha_a",
                "score": 0.7,
                "confidence": 0.9,
                "weight": 1.0,
            },
            {
                "vote_id": "a2",
                "bot_id": "alpha_a_duplicate",
                "score": 0.7,
                "confidence": 0.9,
                "weight": 1.0,
            },
            {
                "vote_id": "b",
                "bot_id": "alpha_b",
                "score": 0.4,
                "confidence": 0.9,
                "weight": 1.0,
            },
        ],
        assignments,
        policy,
    )
    regime_votes = [
        {
            "vote_id": "a",
            "bot_id": "alpha_a",
            "score": 0.7,
            "confidence": 0.9,
            "weight": 1.0,
        },
        {
            "vote_id": "b",
            "bot_id": "alpha_b",
            "score": 0.4,
            "confidence": 0.9,
            "weight": 1.0,
        },
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
    duplicate_invariant = (
        abs(float(baseline.get("score", 0.0)) - float(duplicated.get("score", 0.0)))
        < 1e-12
    )
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
        and regime_mismatch.get("authority", {}).get("paper_execution_authority")
        is False
        and regime_mismatch.get("authority", {}).get("live_execution_authority")
        is False
    )
    return {
        "ok": bool(duplicate_invariant and authority_locked and regime_filter_ready),
        "duplicate_cluster_invariant": duplicate_invariant,
        "authority_locked": authority_locked,
        "regime_filter_ready": regime_filter_ready,
        "regime_matching_accepted_vote_count": regime_matching.get(
            "accepted_vote_count"
        ),
        "regime_mismatch_accepted_vote_count": regime_mismatch.get(
            "accepted_vote_count"
        ),
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
    catalog_input_path = (
        catalog_input_path or project_root / "core" / DEFAULT_CATALOG_INPUT_PATH.name
    )
    hierarchy_out_path = (
        hierarchy_out_path
        or project_root
        / "governance"
        / "bot_organization"
        / DEFAULT_HIERARCHY_OUT_PATH.name
    )
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
    trading_mandate_audit = audit_definitions(
        registry,
        catalog,
        assignments,
        policy.get("definition_audit_contract") or {},
        project_root,
    )
    definition_audit = validate_operating_catalog(
        registry,
        catalog,
        trading_mandate_audit,
        load_json(project_root / OPERATING_POLICY_PATH),
        load_json(project_root / OPERATING_CATALOG_PATH),
        project_root,
    )
    registry_by_id = {
        str(bot.get("bot_id") or bot.get("id") or ""): bot
        for bot in registry.get("sub_bots", []) if isinstance(bot, dict)
    }
    definition_by_id = {row["bot_id"]: row for row in definition_audit["records"]}
    for assignment in assignments:
        bot_id = assignment["bot_id"]
        assignment["status_labels"] = bot_definition_labels(
            registry_by_id.get(bot_id, {}), definition_by_id.get(bot_id, {})
        )
    label_audit = {
        "scope": "all_registry_assignments_not_all_runtime_or_training_outcomes",
        "registered_count": len(registry_by_id),
        "labeled_count": len(assignments),
        "coverage_complete": set(registry_by_id) == {row["bot_id"] for row in assignments},
        "counts": {
            key: dict(Counter(row["status_labels"][key] for row in assignments))
            for key in ("registry", "collection", "definition", "implementation", "process", "runtime", "economic_evidence")
        },
        "definition_completeness_is_not_runtime_or_economic_evidence": True,
    }
    self_test = _shadow_integrity_self_test(policy)
    blockers = ordered_unique(
        list(result.get("blockers") or [])
        + (
            ["hierarchical_shadow_integrity_self_test_failed"]
            if not self_test["ok"]
            else []
        )
    )
    ok = not blockers
    receipt_input = {
        "policy_sha256": _sha256(config_path),
        "registry_sha256": _sha256(registry_path),
        "catalog_input_sha256": _sha256(catalog_input_path),
        "assignment_receipt_sha256": result.get("assignment_receipt_sha256"),
        "definition_audit_sha256": definition_audit.get("audit_sha256"),
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
        "regime_model_id": str(
            (policy.get("regime_model") or {}).get("model_id") or ""
        ),
        "regime_model_contract": {
            "mode": str((policy.get("regime_model") or {}).get("mode") or ""),
            "axis_ids": [
                str(row.get("axis_id") or "")
                for row in (policy.get("regime_model") or {}).get("axes", [])
                if isinstance(row, dict)
            ],
            "compatibility_mode": str(
                (
                    (policy.get("regime_model") or {}).get("compatibility_policy") or {}
                ).get("mode")
                or ""
            ),
            "scenario_partition_version": str(
                (
                    (policy.get("regime_model") or {}).get(
                        "scenario_partition_contract"
                    )
                    or {}
                ).get("version")
                or ""
            ),
            "scenario_partition_mode": str(
                (
                    (policy.get("regime_model") or {}).get(
                        "scenario_partition_contract"
                    )
                    or {}
                ).get("mode")
                or ""
            ),
            "metadata_access_version": str(
                (
                    (policy.get("regime_model") or {}).get("metadata_access_contract")
                    or {}
                ).get("version")
                or ""
            ),
            "metadata_access_mode": str(
                (
                    (policy.get("regime_model") or {}).get("metadata_access_contract")
                    or {}
                ).get("mode")
                or ""
            ),
            "paper_execution_authority": False,
            "live_execution_authority": False,
        },
        "bot_setup_contract": result.get("bot_setup_contract") or {},
        "bot_setup_summary": result.get("bot_setup_summary") or {},
        "tripwire_contract": result.get("tripwire_contract") or {},
        "tripwire_summary": result.get("tripwire_summary") or {},
        "assignments": assignments,
        "status_label_audit": label_audit,
        "definition_audit": definition_audit,
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
        "definition_audit": {
            key: value
            for key, value in definition_audit.items()
            if key not in {"records", "shared_source_contracts"}
        },
        "grade_scope": "organization_structure_not_definition_completeness_or_economic_evidence",
        "status_label_audit": label_audit,
        "overall_status": str(
            result.get("overall_status") or ("ready" if ok else "blocked")
        ),
        "grade": str(result.get("grade") or ("A+" if ok else "F")),
        "policy_id": str(policy.get("policy_id") or ""),
        **{
            key: value
            for key, value in result.items()
            if key not in {"ok", "overall_status", "grade", "blockers"}
        },
        "blockers": blockers,
        "hierarchy_catalog": {
            "path": str(hierarchy_out_path),
            "assignment_count": len(assignments),
            "assignment_receipt_sha256": result.get("assignment_receipt_sha256"),
        },
        "hierarchy_contract": {
            "levels": list((policy.get("hierarchy") or {}).get("levels") or []),
            "one_assignment_per_registered_bot": bool(
                result.get("unique_assignment_ratio") == 1.0
            ),
            "full_registry_coverage": bool(
                result.get("organization_coverage_ratio") == 1.0
            ),
            "classification_provenance_recorded": all(
                bool(row.get("provenance")) for row in assignments
            ),
            "correlation_cluster_recorded": all(
                bool(row.get("correlation_cluster_id")) for row in assignments
            ),
            "bot_setup_profile_recorded": all(
                bool(row.get("setup_tier"))
                and bool(row.get("setup_role_group"))
                and bool(row.get("setup_lifecycle_state"))
                for row in assignments
            ),
            "bot_setup_hardening_ready": str(
                ((result.get("bot_setup_summary") or {}).get("hardening") or {}).get(
                    "overall_status"
                )
                or ""
            )
            == "ready",
            "tripwire_contract_recorded": bool(result.get("tripwire_contract")),
            "tripwire_hardening_ready": str(
                ((result.get("tripwire_summary") or {}).get("hardening") or {}).get(
                    "overall_status"
                )
                or ""
            )
            == "ready",
            "active_tripwire_count": int(result.get("active_tripwire_count", 0) or 0),
            "blocking_tripwire_count": int(
                result.get("blocking_tripwire_count", 0) or 0
            ),
            "multi_axis_regime_profile_recorded": all(
                bool(row.get("regime_profile_id")) and bool(row.get("regime_profile"))
                for row in assignments
            ),
            "declared_scenario_partitions_are_bounded": all(
                not bool(row.get("regime_scenario_partitioned", False))
                or (
                    int(row.get("regime_scenario_count", 0) or 0) >= 2
                    and not list(
                        (
                            (row.get("regime_profile") or {}).get(
                                "scenario_contract_errors"
                            )
                            or []
                        )
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
                (
                    (policy.get("regime_model") or {}).get("compatibility_policy") or {}
                ).get("mode")
                or ""
            ),
            "scenario_partition_version": str(
                (
                    (policy.get("regime_model") or {}).get(
                        "scenario_partition_contract"
                    )
                    or {}
                ).get("version")
                or ""
            ),
            "metadata_access_version": str(
                (
                    (policy.get("regime_model") or {}).get("metadata_access_contract")
                    or {}
                ).get("version")
                or ""
            ),
            "metadata_access_mode": str(
                (
                    (policy.get("regime_model") or {}).get("metadata_access_contract")
                    or {}
                ).get("mode")
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
                (
                    "review low-confidence bot assignments before changing runtime routing"
                    if result.get("review_queue_count")
                    else ""
                ),
                (
                    "replace unknown or overbroad regime axes with evidence-backed registry metadata"
                    if result.get("regime_review_count")
                    else ""
                ),
                (
                    "rank marginal contribution and park excess shadow voters in oversubscribed cells"
                    if result.get("oversubscribed_shadow_cells")
                    else ""
                ),
                (
                    "repair bot setup contract metadata before changing hierarchy"
                    if result.get("setup_hardening_failed_checks")
                    else ""
                ),
                (
                    "work active bot tripwires by owner, action, and evidence before changing runtime routing"
                    if result.get("active_tripwire_count")
                    else ""
                ),
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
    parser.add_argument("--definition-markdown", type=Path)
    parser.add_argument(
        "--materialize-operating-definitions",
        action="store_true",
        help="Explicitly author/rebind the operating-definition catalog from reviewed local sources; never run by routine refresh.",
    )
    parser.add_argument(
        "--require-trading-mandate-complete",
        action="store_true",
        help="Require the original standalone trading mandates independently of operating definitions; no economic clearance.",
    )
    parser.add_argument(
        "--bot-definition", help="Print one expanded seven-area operating definition."
    )
    parser.add_argument(
        "--require-definition-complete",
        action="store_true",
        help="Require source-bound operating definitions for all registered bots, not trading-mandate or economic readiness; does not alter runtime gates.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    config_path = _resolve(project_root, args.config, "config/bot_organization_v1.json")
    registry_path = _resolve(project_root, args.registry, "master_bot_registry.json")
    catalog_input_path = _resolve(
        project_root, args.catalog_input, "core/bot_catalog.json"
    )
    out_path = _resolve(
        project_root, args.out_file, "governance/health/bot_organization_latest.json"
    )
    hierarchy_out_path = _resolve(
        project_root,
        args.hierarchy_out,
        "governance/bot_organization/bot_hierarchy_latest.json",
    )
    if args.materialize_operating_definitions:
        manifest = compile_operating_catalog(
            load_json(registry_path),
            load_json(catalog_input_path),
            load_json(project_root / OPERATING_POLICY_PATH),
            project_root,
        )
        write_payload(project_root / OPERATING_CATALOG_PATH, manifest, compact=True)
    health, hierarchy = build_payload(
        project_root,
        config_path=config_path,
        registry_path=registry_path,
        catalog_input_path=catalog_input_path,
        hierarchy_out_path=hierarchy_out_path,
    )
    write_payload(hierarchy_out_path, hierarchy, compact=True)
    write_payload(out_path, health)
    markdown_path = _resolve(
        project_root,
        args.definition_markdown,
        "exports/reports/operator/bot_definition_audit_latest.md",
    )
    write_text_atomic(
        markdown_path, render_audit_markdown(hierarchy["definition_audit"])
    )
    if args.bot_definition:
        manifest = load_json(project_root / OPERATING_CATALOG_PATH)
        entry = (manifest.get("entries") or {}).get(args.bot_definition)
        record = next(
            (
                row
                for row in hierarchy["definition_audit"]["records"]
                if row["bot_id"] == args.bot_definition
            ),
            None,
        )
        if not entry or not record or not record["definition_complete"]:
            print(
                json.dumps(
                    {
                        "bot_id": args.bot_definition,
                        "definition_complete": False,
                        "issues": (record or {}).get("issues", ["bot_not_found"]),
                    }
                )
            )
            return 2
        expanded_entry = {
            "binding": record["binding"],
            "binding_sha256": record["binding_sha256"],
        }
        print(
            json.dumps(
                {
                    "bot_id": args.bot_definition,
                    "definition_sha256": record["definition_sha256"],
                    "completion_scope": record["completion_scope"],
                    "status_labels": next(
                        row["status_labels"] for row in hierarchy["assignments"]
                        if row["bot_id"] == args.bot_definition
                    ),
                    "areas": expand_definition(
                        expanded_entry, load_json(project_root / OPERATING_POLICY_PATH)
                    ),
                    "processes": expand_process_definition(
                        record["binding"],
                        hierarchy["definition_audit"]["process_contract"],
                    ),
                    "shared_source_contracts": hierarchy["definition_audit"][
                        "shared_source_contracts"
                    ],
                    "economic_evidence": record["economic_evidence"],
                    "standalone_trading_mandate_complete": record[
                        "standalone_trading_mandate"
                    ].get("definition_complete", False),
                },
                ensure_ascii=True,
            )
        )
    elif args.json:
        print(json.dumps(health, ensure_ascii=True))
    else:
        print(
            "bot_organization_control "
            f"status={health['overall_status']} grade={health['grade']} "
            f"organized={health['organized_bot_count']}/{health['registry_bot_count']} "
            f"review={health['review_queue_count']}"
            f" operating_definitions={health['definition_audit'].get('definition_complete_count', 0)}/{health['registry_bot_count']}"
        )
    if args.require_definition_complete and not health["definition_audit"].get(
        "definition_complete"
    ):
        return 2
    if args.require_trading_mandate_complete and not health["definition_audit"].get(
        "standalone_trading_mandate_summary", {}
    ).get("definition_complete"):
        return 2
    return 0 if health["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
