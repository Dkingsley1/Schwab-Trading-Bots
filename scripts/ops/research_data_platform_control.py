#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

from core.accountability import safe_write_json_atomic
from core.research_data_platform import (
    ResearchDataCatalog,
    build_reproducibility_receipt,
    evaluate_feed_slo,
    evaluate_source_value,
    load_policy,
    structural_probe,
)

DEFAULT_CONFIG = Path("config/research_data_platform_v1.json")
DEFAULT_OUT = Path("governance/health/research_data_platform_control_latest.json")
DEFAULT_MARKDOWN = Path("governance/health/research_data_platform_control_latest.md")

SOURCE_ALIASES = {
    "bls_census": "official_macro_context",
    "tradingeconomics_guest": "public_macro_feeds",
    "bond_reference_context": "public_policy_context",
    "macro_cross_asset_context": "macro_crossstack",
}

INTERNAL_SOURCE_ARTIFACTS = {
    "feature_store_manifest": "governance/feature_store/latest.json",
    "point_in_time_event_store": "governance/health/point_in_time_event_store_latest.json",
    "decision_explanations": "logs/decision_explanations_latest.json",
    "paper_performance": "governance/health/paper_performance_latest.json",
    "paper_execution_calibration": "governance/health/paper_execution_calibration_latest.json",
    "multiple_testing_guard": "governance/research/multiple_testing_guard_latest.json",
}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _artifact(
    project_root: Path, relative: str, *, now: datetime, max_age_seconds: float
) -> dict[str, Any]:
    path = project_root / relative
    payload = _load_json(path)
    observed = _parse_timestamp(payload.get("timestamp_utc"))
    if observed is None and path.is_file():
        observed = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    age = max((now - observed).total_seconds(), 0.0) if observed else None
    return {
        "path": str(path),
        "payload": payload,
        "present": bool(payload),
        "age_seconds": round(age, 3) if age is not None else None,
        "fresh": bool(payload and age is not None and age <= max_age_seconds),
    }


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _candidate(project_root: Path) -> dict[str, Any]:
    path = project_root / "governance/runtime/production_candidate_state.json"
    payload = _load_json(path)
    windows = (
        payload.get("scope_windows_started_utc")
        if isinstance(payload.get("scope_windows_started_utc"), dict)
        else {}
    )
    cutoffs = [
        value
        for value in (_parse_timestamp(raw) for raw in windows.values())
        if value is not None
    ]
    candidate_id = str(payload.get("candidate_id") or "").strip()
    return {
        "candidate_id": candidate_id,
        "generation": int(payload.get("generation") or 0),
        "cutoff_utc": max(cutoffs).isoformat() if cutoffs else "",
        "bound": bool(candidate_id and cutoffs),
        "state_path": str(path),
        "state_sha256": _file_sha256(path),
    }


def _source_rows(source_verification: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = source_verification.get("sources") or []
    return {
        str(row.get("source_id") or ""): dict(row)
        for row in rows
        if isinstance(row, Mapping) and row.get("source_id")
    }


def _source_observation(
    source_id: str,
    *,
    rows: Mapping[str, Mapping[str, Any]],
    source_artifact: Mapping[str, Any],
    internal_artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    canonical_id = SOURCE_ALIASES.get(source_id, source_id)
    row = rows.get(canonical_id)
    if isinstance(row, Mapping):
        confidence = float(row.get("source_confidence_score") or 0.0)
        fresh = bool(row.get("fresh", False)) and bool(source_artifact.get("fresh", False))
        ok = bool(row.get("ok", False))
        return {
            "source_id": canonical_id,
            "age_seconds": float(source_artifact.get("age_seconds") or 0.0)
            if fresh
            else 1e12,
            "completeness_ratio": 1.0 if ok else 0.0,
            "validity_ratio": confidence,
            "availability_ratio": 1.0 if fresh else 0.0,
            "correction_ratio": 0.0,
        }
    internal = internal_artifacts.get(source_id)
    if isinstance(internal, Mapping) and internal.get("present"):
        payload = internal.get("payload") if isinstance(internal.get("payload"), dict) else {}
        ok = bool(
            payload.get("ok", payload.get("complete", payload.get("overall_status") == "ready"))
        )
        fresh = bool(internal.get("fresh", False))
        return {
            "source_id": source_id,
            "age_seconds": float(internal.get("age_seconds") or 0.0)
            if fresh
            else 1e12,
            "completeness_ratio": 1.0 if ok else 0.0,
            "validity_ratio": 1.0 if ok else 0.0,
            "availability_ratio": 1.0 if fresh else 0.0,
            "correction_ratio": 0.0,
        }
    return None


def _product_runtime_evidence(
    *,
    catalog: ResearchDataCatalog,
    policy: Mapping[str, Any],
    source_rows: Mapping[str, Mapping[str, Any]],
    source_artifact: Mapping[str, Any],
    internal_artifacts: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for product in catalog.query_catalog():
        observations = [
            observation
            for source_id in product.get("source_ids") or []
            if (
                observation := _source_observation(
                    str(source_id),
                    rows=source_rows,
                    source_artifact=source_artifact,
                    internal_artifacts=internal_artifacts,
                )
            )
            is not None
        ]
        evaluations = [
            evaluate_feed_slo(product, observation, policy) for observation in observations
        ]
        best = next((row for row in evaluations if row.get("ok")), None)
        reports.append(
            {
                "dataset_id": product["dataset_id"],
                "observed_source_count": len(observations),
                "ready": best is not None,
                "status": "ready" if best else "collecting" if observations else "unobserved",
                "source_ids": [row["source_id"] for row in observations],
                "feed_slo": best or (evaluations[0] if evaluations else None),
            }
        )
    return reports


def _reproducibility(
    project_root: Path,
    policy: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    material_paths = {
        "code_revision": project_root / "core/research_data_platform.py",
        "dataset_receipts": project_root / "config/research_data_platform_v1.json",
        "parameter_receipt": project_root / "config/production_excellence_v1.json",
        "label_contract_receipt": project_root / "core/alpha_evidence_contract.py",
        "cost_model_receipt": project_root
        / "governance/health/paper_execution_calibration_latest.json",
        "result_receipt": project_root / "governance/health/paper_performance_latest.json",
    }
    hashes = {key: _file_sha256(path) for key, path in material_paths.items()}
    missing = [key for key, value in hashes.items() if not value]
    if not candidate.get("bound"):
        missing.append("candidate_id")
    if missing:
        return {
            "ready": False,
            "status": "collecting",
            "missing_materials": sorted(set(missing)),
            "receipt": {},
        }
    receipt = build_reproducibility_receipt(
        {"candidate_id": candidate["candidate_id"], **hashes}, policy
    )
    return {
        "ready": True,
        "status": "local_receipt_ready",
        "missing_materials": [],
        "receipt": receipt,
        "material_paths": {key: str(path) for key, path in material_paths.items()},
    }


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    config_path: Path = DEFAULT_CONFIG,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    config_file = config_path if config_path.is_absolute() else project_root / config_path
    policy = load_policy(config_file)
    probe = structural_probe(policy)
    catalog = ResearchDataCatalog(policy)
    candidate = _candidate(project_root)
    source_artifact = _artifact(
        project_root,
        "governance/health/source_verification_latest.json",
        now=now,
        max_age_seconds=4 * 60 * 60,
    )
    source_rows = _source_rows(source_artifact.get("payload") or {})
    internal_artifacts = {
        source_id: _artifact(
            project_root,
            relative,
            now=now,
            max_age_seconds=24 * 60 * 60,
        )
        for source_id, relative in INTERNAL_SOURCE_ARTIFACTS.items()
    }
    products = _product_runtime_evidence(
        catalog=catalog,
        policy=policy,
        source_rows=source_rows,
        source_artifact=source_artifact,
        internal_artifacts=internal_artifacts,
    )
    product_ready_count = sum(1 for row in products if row["ready"])

    entitlement_states: dict[str, Any] = {}
    market_source = source_rows.get("market_quote_profiles") or {}
    if market_source.get("ok") and market_source.get("fresh"):
        entitlement_states["broker_market_data_access"] = "active"
    authorization_reports = [
        catalog.authorize_use(
            row["dataset_id"],
            "paper_decision_context",
            entitlement_states=entitlement_states,
        )
        for row in catalog.query_catalog()
    ]

    feature_artifact = internal_artifacts["feature_store_manifest"]
    feature_payload = feature_artifact.get("payload") or {}
    point_contract = (
        feature_payload.get("point_in_time_contract")
        if isinstance(feature_payload.get("point_in_time_contract"), dict)
        else {}
    )
    event_artifact = internal_artifacts["point_in_time_event_store"]
    event_payload = event_artifact.get("payload") or {}
    event_contract = (
        event_payload.get("point_in_time_contract")
        if isinstance(event_payload.get("point_in_time_contract"), dict)
        else {}
    )
    pit_ready = bool(
        feature_artifact["fresh"]
        and event_artifact["fresh"]
        and event_payload.get("ok")
        and event_contract.get("point_in_time_only")
        and int(event_contract.get("future_event_count") or 0) == 0
        and point_contract.get("complete")
    )
    bitemporal_observed = bool(
        pit_ready
        and int(point_contract.get("revision_history_count") or 0) > 0
        and int(point_contract.get("future_revision_count") or 0) == 0
    )

    lifecycle_artifact = _artifact(
        project_root,
        "governance/research/alpha_lifecycle_latest.json",
        now=now,
        max_age_seconds=24 * 60 * 60,
    )
    lifecycle_payload = lifecycle_artifact.get("payload") or {}
    lifecycle_ready = bool(
        lifecycle_artifact["fresh"]
        and candidate.get("bound")
        and str((lifecycle_payload.get("candidate_binding") or {}).get("candidate_id") or "")
        == candidate.get("candidate_id")
        and lifecycle_payload.get("chain_ok")
    )

    source_value_artifact = _artifact(
        project_root,
        "governance/research/source_value_ledger_latest.json",
        now=now,
        max_age_seconds=24 * 60 * 60,
    )
    source_value_payload = source_value_artifact.get("payload") or {}
    source_value_rows = []
    for source_id, source in sorted(source_rows.items()):
        metrics = {
            "candidate_id": str(source_value_payload.get("candidate_id") or ""),
            "candidate_bound_samples": int(
                ((source_value_payload.get("sources") or {}).get(source_id) or {}).get(
                    "candidate_bound_samples", 0
                )
                if isinstance(source_value_payload.get("sources"), dict)
                else 0
            ),
            "quality": float(source.get("source_confidence_score") or 0.0),
            "freshness": 1.0 if source.get("fresh") else 0.0,
            "availability": 1.0 if source.get("ok") else 0.0,
        }
        source_metrics = (
            (source_value_payload.get("sources") or {}).get(source_id) or {}
            if isinstance(source_value_payload.get("sources"), dict)
            else {}
        )
        for key in (
            "incremental_information",
            "net_post_cost_contribution",
            "nonredundancy",
        ):
            if key in source_metrics:
                metrics[key] = source_metrics[key]
        source_value_rows.append(evaluate_source_value(source_id, metrics, policy))
    source_value_ready = bool(
        source_value_rows
        and candidate.get("bound")
        and source_value_payload.get("candidate_id") == candidate.get("candidate_id")
        and all(row["evidence_ready"] for row in source_value_rows)
    )

    portfolio_artifact = _artifact(
        project_root,
        "governance/health/portfolio_alpha_advisory_latest.json",
        now=now,
        max_age_seconds=24 * 60 * 60,
    )
    portfolio_payload = portfolio_artifact.get("payload") or {}
    portfolio_ready = bool(
        portfolio_artifact["fresh"]
        and portfolio_payload.get("ok")
        and portfolio_payload.get("advisory_only")
        and not portfolio_payload.get("execution_authority", False)
        and portfolio_payload.get("candidate_id") == candidate.get("candidate_id")
    )

    equivalence_artifact = _artifact(
        project_root,
        "governance/health/paper_live_equivalence_latest.json",
        now=now,
        max_age_seconds=24 * 60 * 60,
    )
    equivalence_payload = equivalence_artifact.get("payload") or {}
    simulation_ready = bool(
        equivalence_artifact["fresh"]
        and equivalence_payload.get("structural_ready")
        and equivalence_payload.get("empirical_ready")
        and int(equivalence_payload.get("paired_count") or 0) > 0
    )
    feed_ready = bool(
        source_artifact["fresh"]
        and source_artifact.get("payload", {}).get("overall_status") == "ready"
        and sum(1 for row in products if row["ready"] and row["dataset_id"] not in {
            "point_in_time_feature_store_v1",
            "candidate_outcome_evidence_v2",
        })
        >= 8
    )
    reproducibility = _reproducibility(project_root, policy, candidate)

    evidence = {
        "canonical_data_catalog": {
            "ready": product_ready_count == len(products),
            "status": "ready" if product_ready_count == len(products) else "collecting",
            "observed_products": product_ready_count,
            "required_products": len(products),
        },
        "license_entitlement_registry": {
            "ready": all(row["authorized"] for row in authorization_reports),
            "status": "ready"
            if all(row["authorized"] for row in authorization_reports)
            else "human_review_pending",
            "authorized_products": sum(1 for row in authorization_reports if row["authorized"]),
            "product_count": len(authorization_reports),
            "reports": authorization_reports,
        },
        "point_in_time_research_api": {
            "ready": pit_ready,
            "status": "ready" if pit_ready else "coverage_accruing",
            "snapshot_coverage_ratio": point_contract.get("snapshot_coverage_ratio"),
            "snapshot_coverage_floor": point_contract.get("snapshot_coverage_floor"),
            "future_event_count": int(event_contract.get("future_event_count") or 0),
        },
        "bitemporal_revision_history": {
            "ready": bitemporal_observed,
            "status": "ready" if bitemporal_observed else "revision_evidence_accruing",
            "revision_history_count": int(point_contract.get("revision_history_count") or 0),
        },
        "alpha_lifecycle_governance": {
            "ready": lifecycle_ready,
            "status": "ready" if lifecycle_ready else "candidate_lifecycle_evidence_accruing",
            "candidate_id": candidate.get("candidate_id"),
        },
        "source_value_accounting": {
            "ready": source_value_ready,
            "status": "ready" if source_value_ready else "candidate_outcomes_accruing",
            "qualified_sources": sum(1 for row in source_value_rows if row["qualified"]),
            "evaluated_sources": sum(1 for row in source_value_rows if row["evidence_ready"]),
            "source_count": len(source_value_rows),
            "automatic_purchase_authority": False,
            "automatic_retirement_authority": False,
        },
        "portfolio_alpha_combination": {
            "ready": portfolio_ready,
            "status": "ready" if portfolio_ready else "candidate_advisory_evidence_accruing",
        },
        "unified_simulation_semantics": {
            "ready": simulation_ready,
            "status": "ready" if simulation_ready else "observed_pair_evidence_accruing",
            "paired_count": int(equivalence_payload.get("paired_count") or 0),
        },
        "feed_service_levels": {
            "ready": feed_ready,
            "status": "ready" if feed_ready else "feed_evidence_attention",
            "ready_products": product_ready_count,
            "product_count": len(products),
        },
        "research_reproducibility": {
            "ready": bool(reproducibility["ready"]),
            "status": reproducibility["status"],
            "external_attestation": False,
        },
    }
    evidence_ready_count = sum(1 for row in evidence.values() if row["ready"])
    implementation_ready = bool(probe["ok"])
    paper_soak_ready = implementation_ready
    overall_status = (
        "blocked"
        if not implementation_ready
        else "ready"
        if evidence_ready_count == len(evidence)
        else "ready_with_evidence_debt"
    )
    return {
        "timestamp_utc": now.isoformat(),
        "schema_version": 1,
        "policy_id": str(policy.get("policy_id") or ""),
        "ok": implementation_ready,
        "overall_status": overall_status,
        "implementation_grade": "A+" if implementation_ready else "F",
        "implementation_ready_count": int(probe["ready_count"]),
        "implementation_control_count": int(probe["control_count"]),
        "implementation_controls": probe["controls"],
        "evidence_status": "earned" if evidence_ready_count == len(evidence) else "accruing",
        "evidence_ready_count": evidence_ready_count,
        "evidence_control_count": len(evidence),
        "evidence_controls": evidence,
        "catalog": {
            "data_product_count": len(products),
            "ready_product_count": product_ready_count,
            "decision_family_count": int(probe["validation"]["decision_family_count"]),
            "decision_family_product_counts": probe["validation"]["family_product_counts"],
            "products": products,
        },
        "candidate_binding": candidate,
        "source_value": {
            "qualified_count": sum(1 for row in source_value_rows if row["qualified"]),
            "evidence_ready_count": sum(1 for row in source_value_rows if row["evidence_ready"]),
            "source_count": len(source_value_rows),
            "rows": source_value_rows,
        },
        "reproducibility": reproducibility,
        "paper_soak_ready": paper_soak_ready,
        "paper_impact": "none",
        "live_promotion_ready": False,
        "authority": dict(policy.get("authority") or {}),
        "live_execution_authority": False,
        "soak_acceptance": {
            "classification": "additive_observability_and_research_governance",
            "reset_soak_clock": False,
            "preserve_cumulative_history": True,
            "new_evidence_accrues_forward": True,
            "changes_signal_or_order_semantics": False,
        },
        "evidence_semantics": {
            "implementation_is_not_profitability_evidence": True,
            "catalog_presence_is_not_source_value": True,
            "source_count_is_not_alpha": True,
            "local_receipt_is_not_external_attestation": True,
            "live_representation_is_not_order_authority": True,
        },
        "actions": [
            "continue_candidate_bound_collection_for_pit_coverage_source_value_and_alpha_lifecycle",
            "record_explicit_license_terms_review_receipts_before_new_restricted_data_uses",
            "retain_live_execution_lock_until_independent_promotion_controls_pass",
        ],
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Research Data Platform Control",
        "",
        f"- Status: `{payload.get('overall_status')}`",
        f"- Implementation: `{payload.get('implementation_ready_count')}/{payload.get('implementation_control_count')}` (`{payload.get('implementation_grade')}`)",
        f"- Earned evidence: `{payload.get('evidence_ready_count')}/{payload.get('evidence_control_count')}`",
        f"- Data products: `{(payload.get('catalog') or {}).get('ready_product_count')}/{(payload.get('catalog') or {}).get('data_product_count')}` observed",
        f"- Decision families: `{(payload.get('catalog') or {}).get('decision_family_count')}`",
        f"- Paper soak impact: `{payload.get('paper_impact')}`",
        f"- Live promotion ready: `{str(bool(payload.get('live_promotion_ready'))).lower()}`",
        "",
        "## Capability Evidence",
        "",
    ]
    for capability_id, row in (payload.get("evidence_controls") or {}).items():
        marker = "ready" if row.get("ready") else "accruing"
        lines.append(f"- `{capability_id}`: **{marker}** (`{row.get('status')}`)")
    lines.extend(
        [
            "",
            "Implementation readiness is structural. Profitability, external attestation, and live-promotion evidence must still be earned from candidate-bound observations.",
        ]
    )
    return "\n".join(lines) + "\n"


def _atomic_write_text(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(body, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the ten-contract research data and alpha governance platform."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT))
    parser.add_argument("--markdown-file", default=str(DEFAULT_MARKDOWN))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    payload = build_payload(root, config_path=Path(args.config).expanduser())
    out = Path(args.out_file).expanduser()
    markdown = Path(args.markdown_file).expanduser()
    if not out.is_absolute():
        out = root / out
    if not markdown.is_absolute():
        markdown = root / markdown
    if not safe_write_json_atomic(
        str(out),
        payload,
        project_root=str(root),
        source="research_data_platform_control",
    ):
        return 2
    _atomic_write_text(markdown, render_markdown(payload))
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    else:
        print(
            f"research_data_platform status={payload['overall_status']} "
            f"implementation={payload['implementation_ready_count']}/{payload['implementation_control_count']} "
            f"evidence={payload['evidence_ready_count']}/{payload['evidence_control_count']}"
        )
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
