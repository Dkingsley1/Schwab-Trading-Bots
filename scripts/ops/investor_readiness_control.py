#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "investor_readiness_v1.json"
HEALTH_ARTIFACT_NAME = "investor_readiness_control_latest.json"
PACKET_ARTIFACT_NAME = "investor_readiness_packet_latest.md"
TEAR_SHEET_ARTIFACT_NAME = "paper_performance_tear_sheet_latest.md"
DATA_ROOM_INDEX_ARTIFACT_NAME = "index_latest.json"
DATA_ROOM_README_ARTIFACT_NAME = "README_latest.md"
STATUS_READY = "ready"
STATUS_EVIDENCE_PENDING = "implemented_evidence_pending"
STATUS_EXTERNAL_REQUIRED = "external_action_required"
STATUS_IMPLEMENTATION_GAP = "implementation_gap"
CONTROL_STATUSES = {
    STATUS_READY,
    STATUS_EVIDENCE_PENDING,
    STATUS_EXTERNAL_REQUIRED,
    STATUS_IMPLEMENTATION_GAP,
}
HEX_64 = re.compile(r"^[0-9a-fA-F]{64}$")


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _project_path(project_root: Path, raw: Any) -> Path:
    path = Path(str(raw or ""))
    return path if path.is_absolute() else project_root / path


def _relative(project_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except Exception:
        return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _status_ready(payload: dict[str, Any]) -> bool:
    if not payload:
        return False
    status = str(payload.get("overall_status") or payload.get("status") or "").strip().lower()
    if payload.get("ok") is False:
        return False
    return status in {"ready", "ready_locked", "ok", "stable", "pass", "guarded_ready"} or payload.get("ok") is True


def _control_rows(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("control_id")): row
        for row in _as_list(config.get("controls"))
        if isinstance(row, dict) and str(row.get("control_id") or "").strip()
    }


def _load_sources(project_root: Path, config: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    payloads: dict[str, dict[str, Any]] = {}
    paths: dict[str, str] = {}
    for source_id, raw_path in _as_dict(config.get("source_artifacts")).items():
        path = _project_path(project_root, raw_path)
        payloads[str(source_id)] = load_json(path)
        paths[str(source_id)] = _relative(project_root, path)
    return payloads, paths


def validate_external_attestation(
    project_root: Path,
    attestation_path: Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    payload = load_json(attestation_path)
    required = [str(item) for item in _as_list(contract.get("required_fields")) if str(item).strip()]
    ready_statuses = {str(item).strip().lower() for item in _as_list(contract.get("ready_statuses"))}
    blockers: list[str] = []
    if not payload:
        blockers.append("attestation_missing")
    status = str(payload.get("status") or "").strip().lower()
    if payload and status not in ready_statuses:
        blockers.append("attestation_status_not_approved")
    if payload and bool(contract.get("independent_must_be_true", True)) and payload.get("independent") is not True:
        blockers.append("attestation_not_independent")
    for field in required:
        if payload and not str(payload.get(field) or "").strip():
            blockers.append(f"attestation_field_missing={field}")

    raw_document_path = str(payload.get("document_path") or "").strip()
    document_path = _project_path(project_root, raw_document_path) if raw_document_path else None
    expected_hash = str(payload.get("document_sha256") or "").strip()
    hash_format_ready = bool(HEX_64.fullmatch(expected_hash))
    if payload and expected_hash and not hash_format_ready:
        blockers.append("document_sha256_invalid")
    document_present = bool(document_path and document_path.is_file())
    if payload and raw_document_path and not document_present:
        blockers.append("attested_document_missing")
    observed_hash = _sha256(document_path) if document_present else ""
    hash_matches = bool(document_present and hash_format_ready and observed_hash.lower() == expected_hash.lower())
    if payload and document_present and hash_format_ready and not hash_matches:
        blockers.append("attested_document_hash_mismatch")

    blockers = ordered_unique(blockers)
    return {
        "path": _relative(project_root, attestation_path),
        "present": bool(payload),
        "valid": bool(payload and not blockers),
        "status": status or "missing",
        "independent": payload.get("independent") is True,
        "provider": str(payload.get("provider") or ""),
        "signed_by": str(payload.get("signed_by") or ""),
        "signed_at_utc": str(payload.get("signed_at_utc") or ""),
        "document_path": _relative(project_root, document_path) if document_path else "",
        "document_present": document_present,
        "document_sha256_matches": hash_matches,
        "blockers": blockers,
    }


def _attestations(project_root: Path, config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    contract = _as_dict(config.get("external_attestation_contract"))
    return {
        str(attestation_id): validate_external_attestation(project_root, _project_path(project_root, raw_path), contract)
        for attestation_id, raw_path in _as_dict(config.get("external_attestations")).items()
    }


def _result(
    row: dict[str, Any],
    *,
    implementation_ready: bool,
    organic_evidence_ready: bool | None = None,
    external_evidence_ready: bool | None = None,
    blockers: list[str] | None = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    completion_class = str(row.get("completion_class") or "")
    if not implementation_ready:
        status = STATUS_IMPLEMENTATION_GAP
    elif organic_evidence_ready is False:
        status = STATUS_EVIDENCE_PENDING
    elif external_evidence_ready is False:
        status = STATUS_EXTERNAL_REQUIRED
    else:
        status = STATUS_READY
    clean_blockers = ordered_unique(str(item or "").strip() for item in (blockers or []) if str(item or "").strip())
    if status == STATUS_READY:
        clean_blockers = []
    return {
        "control_id": str(row.get("control_id") or ""),
        "section": str(row.get("section") or ""),
        "title": str(row.get("title") or ""),
        "completion_class": completion_class,
        "owner": str(row.get("owner") or ""),
        "status": status,
        "ready": status == STATUS_READY,
        "implementation_ready": bool(implementation_ready),
        "organic_evidence_ready": organic_evidence_ready,
        "external_evidence_ready": external_evidence_ready,
        "blockers": clean_blockers,
        "evidence": evidence or {},
    }


def _manifest_row(rows: dict[str, dict[str, Any]], control_id: str) -> dict[str, Any]:
    return rows.get(
        control_id,
        {
            "control_id": control_id,
            "section": "unknown",
            "title": control_id,
            "completion_class": "unknown",
            "owner": "unassigned",
        },
    )


def _qualified_sleeves(firewall: dict[str, Any], maximum: int) -> list[dict[str, Any]]:
    proposal = _as_dict(firewall.get("allocation_proposal"))
    rows: list[dict[str, Any]] = []
    for raw in _as_list(proposal.get("qualified_sleeves")):
        if isinstance(raw, dict):
            sleeve = str(raw.get("sleeve") or raw.get("profile") or raw.get("name") or "").strip()
            if sleeve:
                rows.append({"sleeve": sleeve, **raw})
        else:
            sleeve = str(raw or "").strip()
            if sleeve:
                rows.append({"sleeve": sleeve})
    return rows[: max(0, maximum)]


def _control_results(
    config: dict[str, Any],
    sources: dict[str, dict[str, Any]],
    source_paths: dict[str, str],
    attestations: dict[str, dict[str, Any]],
    project_root: Path,
) -> list[dict[str, Any]]:
    rows = _control_rows(config)
    paper = sources.get("paper_performance", {})
    validator = sources.get("profitability_independent_validator", {})
    firewall = sources.get("profitability_evidence_firewall", {})
    multiple = sources.get("multiple_testing_guard", {})
    capacity = sources.get("portfolio_capacity_curve", {})
    transition = sources.get("live_transition_integrity", {})
    canary = sources.get("live_canary_control", {})
    soak = sources.get("continuous_soak_integrity", {})
    soak_history = _as_dict(soak.get("historical_soak_evidence"))
    resilience = sources.get("production_resilience", {})
    commercial = sources.get("commercial_readiness", {})
    experiment = sources.get("immutable_experiment_ledger", {})
    fills = sources.get("independent_fill_acquisition", {})
    broker = sources.get("broker_shared_truth", {})
    live_ledger = sources.get("live_order_ledger", {})
    roles = sources.get("system_role_contract", {})
    monitor = sources.get("independent_runtime_monitor", {})

    future = _as_dict(config.get("future_live_evidence"))
    live_performance_path = _project_path(project_root, future.get("broker_verified_performance"))
    divergence_path = _project_path(project_root, future.get("paper_live_divergence"))
    live_performance = load_json(live_performance_path)
    divergence = load_json(divergence_path)

    post_cost = _as_dict(paper.get("post_cost_expectancy"))
    accounting = _as_dict(paper.get("accounting_views"))
    candidate_flow = _as_dict(accounting.get("candidate_forward_flow"))
    candidate_window = _as_dict(paper.get("profitability_evidence_window"))
    risk_of_ruin = _as_dict(validator.get("risk_of_ruin"))
    proposal = _as_dict(firewall.get("allocation_proposal"))
    capacity_summary = _as_dict(capacity.get("summary"))
    current_stage = _as_dict(transition.get("current_canary_stage"))
    release_checks = _as_dict(_as_dict(transition.get("release_interlock")).get("checks"))
    shortlisted = _qualified_sleeves(firewall, _safe_int(config.get("maximum_shortlisted_sleeves"), 3))

    post_cost_ready = bool(
        post_cost.get("available")
        and post_cost.get("evidence_sufficient")
        and post_cost.get("positive_clustered_lower_confidence_bound_95")
        and validator.get("evidence_ready")
    )
    drawdown_ready = bool(
        risk_of_ruin.get("available")
        and risk_of_ruin.get("passes")
        and not release_checks.get("drawdown_limit_breached", False)
    )
    replay_ready = bool(
        experiment.get("append_only_ready")
        and experiment.get("latest_signature_ready")
        and experiment.get("latest_exact_replay_ready")
        and _safe_int(experiment.get("ledger_row_count")) > 0
    )
    candidate_fill_ready = _safe_int(fills.get("candidate_eligible_ledger_records")) > 0
    performance_attestation = attestations.get("independent_performance_verification", {})
    accounting_attestation = attestations.get("independent_accounting_review", {})
    ip_attestation = attestations.get("ip_ownership_review", {})
    legal_attestation = attestations.get("legal_structure_review", {})
    live_result_ready = bool(
        live_performance.get("broker_verified") is True
        and _safe_int(live_performance.get("live_sample_count")) > 0
        and live_performance.get("net_of_costs_available") is True
        and str(live_performance.get("candidate_id") or "") == str(candidate_window.get("candidate_id") or "")
    )
    divergence_ready = bool(
        _safe_int(divergence.get("live_sample_count")) > 0
        and divergence.get("within_limits") is True
        and str(divergence.get("candidate_id") or "") == str(candidate_window.get("candidate_id") or "")
    )

    results: list[dict[str, Any]] = []
    results.append(
        _result(
            _manifest_row(rows, "i01_broker_verified_live_results"),
            implementation_ready=bool(broker and transition),
            organic_evidence_ready=live_result_ready,
            external_evidence_ready=bool(performance_attestation.get("valid")),
            blockers=[
                "broker_truth_or_live_transition_control_missing" if not (broker and transition) else "",
                "candidate_bound_broker_verified_live_results_pending" if not live_result_ready else "",
                "independent_performance_verification_pending" if not performance_attestation.get("valid") else "",
            ],
            evidence={
                "broker_truth_path": source_paths.get("broker_shared_truth", ""),
                "future_live_performance_path": _relative(project_root, live_performance_path),
                "live_sample_count": _safe_int(live_performance.get("live_sample_count")),
                "candidate_bound": bool(live_performance and str(live_performance.get("candidate_id") or "") == str(candidate_window.get("candidate_id") or "")),
                "attestation": performance_attestation,
            },
        )
    )
    results.append(
        _result(
            _manifest_row(rows, "i02_net_of_all_costs"),
            implementation_ready=bool(paper and validator.get("implementation_ready")),
            organic_evidence_ready=post_cost_ready,
            blockers=[
                "paper_performance_or_validator_implementation_missing" if not (paper and validator.get("implementation_ready")) else "",
                *[str(item) for item in _as_list(post_cost.get("promotion_blockers"))],
                *[str(item) for item in _as_list(validator.get("blockers"))],
            ],
            evidence={
                "candidate_id": candidate_window.get("candidate_id"),
                "candidate_sample_count": _safe_int(candidate_flow.get("sample_count")),
                "post_cost_expectancy": post_cost,
            },
        )
    )
    results.append(
        _result(
            _manifest_row(rows, "i03_controlled_drawdowns"),
            implementation_ready=bool(validator.get("implementation_ready") and transition),
            organic_evidence_ready=drawdown_ready,
            blockers=[
                "risk_or_transition_implementation_missing" if not (validator.get("implementation_ready") and transition) else "",
                *[str(item) for item in _as_list(risk_of_ruin.get("blockers"))],
                "drawdown_limit_breached" if release_checks.get("drawdown_limit_breached", False) else "",
            ],
            evidence={"risk_of_ruin": risk_of_ruin, "drawdown_limit_breached": bool(release_checks.get("drawdown_limit_breached", False))},
        )
    )
    results.append(
        _result(
            _manifest_row(rows, "i04_statistically_credible_edge"),
            implementation_ready=bool(multiple and multiple.get("contract_present")),
            organic_evidence_ready=bool(multiple.get("statistical_evidence_ready")),
            blockers=[
                "multiple_testing_contract_missing" if not (multiple and multiple.get("contract_present")) else "",
                *[str(item) for item in _as_list(multiple.get("statistical_evidence_blockers"))],
            ],
            evidence={
                "family_size": _safe_int(multiple.get("family_size")),
                "deflated_sharpe_available_by_sleeve": multiple.get("deflated_sharpe_available_by_sleeve", {}),
                "probability_of_backtest_overfitting": multiple.get("probability_of_backtest_overfitting", {}),
            },
        )
    )
    results.append(
        _result(
            _manifest_row(rows, "i05_capacity_evidence"),
            implementation_ready=bool(capacity and capacity_summary),
            organic_evidence_ready=bool(capacity_summary.get("allocator_ready") and _safe_int(capacity_summary.get("curve_count")) > 0),
            blockers=[
                "capacity_curve_control_missing" if not (capacity and capacity_summary) else "",
                "capacity_curves_pending" if _safe_int(capacity_summary.get("curve_count")) <= 0 else "",
                "allocator_capacity_clearance_pending" if not capacity_summary.get("allocator_ready") else "",
            ],
            evidence={"summary": capacity_summary},
        )
    )
    results.append(
        _result(
            _manifest_row(rows, "i06_diversification_evidence"),
            implementation_ready=bool(firewall.get("control_implementation_ready")),
            organic_evidence_ready=bool(proposal.get("ready") and shortlisted),
            blockers=[
                "profitability_evidence_firewall_implementation_missing" if not firewall.get("control_implementation_ready") else "",
                "qualified_diversified_sleeves_pending" if not (proposal.get("ready") and shortlisted) else "",
            ],
            evidence={"allocation_proposal": proposal, "shortlisted_sleeves": shortlisted},
        )
    )
    results.append(
        _result(
            _manifest_row(rows, "i07_independently_checkable_records"),
            implementation_ready=bool(experiment.get("append_only_ready") and fills),
            organic_evidence_ready=bool(replay_ready and candidate_fill_ready),
            external_evidence_ready=bool(performance_attestation.get("valid")),
            blockers=[
                "immutable_ledger_or_fill_acquisition_missing" if not (experiment.get("append_only_ready") and fills) else "",
                "latest_exact_replay_pending" if not replay_ready else "",
                "candidate_eligible_independent_fills_pending" if not candidate_fill_ready else "",
                "independent_performance_verification_pending" if not performance_attestation.get("valid") else "",
            ],
            evidence={
                "experiment_ledger": experiment,
                "candidate_eligible_fill_records": _safe_int(fills.get("candidate_eligible_ledger_records")),
                "attestation": performance_attestation,
            },
        )
    )
    bounded_ready = bool(
        _status_ready(roles)
        and _status_ready(live_ledger)
        and live_ledger.get("live_execution_authority") is False
        and current_stage.get("automatic_scaling_allowed") is False
        and current_stage.get("operator_release_required_for_each_stage") is True
    )
    results.append(
        _result(
            _manifest_row(rows, "i08_bounded_automation"),
            implementation_ready=bounded_ready,
            blockers=["bounded_automation_contract_incomplete" if not bounded_ready else ""],
            evidence={
                "system_role_status": roles.get("overall_status", "missing"),
                "live_order_ledger_status": live_ledger.get("overall_status", "missing"),
                "automatic_scaling_allowed": current_stage.get("automatic_scaling_allowed"),
                "operator_release_required_for_each_stage": current_stage.get("operator_release_required_for_each_stage"),
                "live_execution_authority": False,
            },
        )
    )
    resilience_implementation = bool(resilience.get("implementation_ready"))
    resilience_evidence = bool(resilience.get("paper_soak_ready") and monitor.get("production_monitor_ready"))
    results.append(
        _result(
            _manifest_row(rows, "i09_operational_resilience"),
            implementation_ready=resilience_implementation,
            organic_evidence_ready=resilience_evidence,
            blockers=[
                "production_resilience_implementation_missing" if not resilience_implementation else "",
                *[str(item) for item in _as_list(resilience.get("paper_blockers"))],
                "independent_production_monitor_pending" if not monitor.get("production_monitor_ready") else "",
            ],
            evidence={
                "paper_soak_ready": bool(resilience.get("paper_soak_ready")),
                "local_monitor_ready": bool(monitor.get("local_monitor_ready")),
                "production_monitor_ready": bool(monitor.get("production_monitor_ready")),
            },
        )
    )
    results.append(
        _result(
            _manifest_row(rows, "i10_commercial_defensibility"),
            implementation_ready=bool(commercial),
            external_evidence_ready=bool(ip_attestation.get("valid")),
            blockers=[
                "commercial_readiness_framework_missing" if not commercial else "",
                "independent_ip_ownership_review_pending" if not ip_attestation.get("valid") else "",
            ],
            evidence={"commercial_product_mode": commercial.get("commercial_product_mode", "missing"), "attestation": ip_attestation},
        )
    )

    results.append(
        _result(
            _manifest_row(rows, "r01_shortlist_strong_sleeves"),
            implementation_ready=bool(firewall.get("control_implementation_ready")),
            organic_evidence_ready=bool(proposal.get("ready") and shortlisted),
            blockers=[
                "profitability_evidence_firewall_implementation_missing" if not firewall.get("control_implementation_ready") else "",
                "no_independently_qualified_sleeves" if not shortlisted else "",
            ],
            evidence={
                "shortlisted_sleeves": shortlisted,
                "maximum_shortlist_size": _safe_int(config.get("maximum_shortlisted_sleeves"), 3),
                "selection_source": "profitability_evidence_firewall.allocation_proposal.qualified_sleeves_only",
                "lifetime_leaderboard_fallback_allowed": False,
            },
        )
    )
    results.append(
        _result(
            _manifest_row(rows, "r02_record_every_experiment"),
            implementation_ready=bool(experiment.get("append_only_ready")),
            organic_evidence_ready=replay_ready,
            blockers=[
                "append_only_experiment_ledger_missing" if not experiment.get("append_only_ready") else "",
                "latest_signature_or_exact_replay_pending" if not replay_ready else "",
            ],
            evidence=experiment,
        )
    )
    results.append(
        _result(
            _manifest_row(rows, "r03_positive_post_cost_expectancy"),
            implementation_ready=bool(validator.get("implementation_ready") and paper),
            organic_evidence_ready=post_cost_ready,
            blockers=[
                "post_cost_validator_implementation_missing" if not (validator.get("implementation_ready") and paper) else "",
                *[str(item) for item in _as_list(post_cost.get("promotion_blockers"))],
                *[str(item) for item in _as_list(validator.get("blockers"))],
            ],
            evidence={"candidate_flow": candidate_flow, "post_cost_expectancy": post_cost},
        )
    )
    soak_ready = bool(soak.get("clean_720_hours_complete") and canary.get("supervised_canary_ready"))
    results.append(
        _result(
            _manifest_row(rows, "r04_complete_soak_before_canary"),
            implementation_ready=bool(soak and canary),
            organic_evidence_ready=soak_ready,
            blockers=[
                "soak_or_canary_control_missing" if not (soak and canary) else "",
                "clean_720_hour_soak_pending" if not soak.get("clean_720_hours_complete") else "",
                *[str(item) for item in _as_list(canary.get("blocking_reasons"))],
            ],
            evidence={
                "main_soak_elapsed_hours": _safe_float(soak.get("main_soak_elapsed_hours")),
                "main_soak_elapsed_days": _safe_float(soak.get("main_soak_elapsed_days")),
                "main_soak_progress_percent": _safe_float(soak.get("main_soak_progress_percent")),
                "main_soak_includes_pre_reset_time": bool(
                    soak.get("main_soak_includes_pre_reset_time", False)
                ),
                "main_soak_count_is_promotion_credit": bool(
                    soak.get("main_soak_count_is_promotion_credit", False)
                ),
                "clean_720_hours_complete": bool(soak.get("clean_720_hours_complete")),
                "clean_window_elapsed_hours": _safe_float(soak.get("clean_window_elapsed_hours")),
                "observed_window_elapsed_hours": _safe_float(soak.get("observed_window_elapsed_hours")),
                "historical_segmented_wall_clock_hours": _safe_float(
                    soak_history.get("historical_segmented_wall_clock_hours")
                ),
                "historical_segmented_wall_clock_days": _safe_float(
                    soak_history.get("historical_segmented_wall_clock_days")
                ),
                "historical_segment_count": int(_safe_float(soak_history.get("segment_count"))),
                "historical_counts_toward_clean_720_hours": bool(
                    soak_history.get("counts_toward_current_clean_720_hours", False)
                ),
                "supervised_canary_ready": bool(canary.get("supervised_canary_ready")),
            },
        )
    )
    results.append(
        _result(
            _manifest_row(rows, "r05_measure_paper_live_divergence"),
            implementation_ready=bool(transition and future.get("paper_live_divergence")),
            organic_evidence_ready=divergence_ready,
            blockers=[
                "paper_live_divergence_contract_missing" if not (transition and future.get("paper_live_divergence")) else "",
                "candidate_bound_paper_live_samples_pending" if not divergence_ready else "",
            ],
            evidence={
                "path": _relative(project_root, divergence_path),
                "live_sample_count": _safe_int(divergence.get("live_sample_count")),
                "within_limits": divergence.get("within_limits"),
                "candidate_bound": bool(divergence and str(divergence.get("candidate_id") or "") == str(candidate_window.get("candidate_id") or "")),
            },
        )
    )
    stages = _as_list(current_stage.get("stages"))
    scaling_contract = _as_dict(config.get("capital_scaling_contract"))
    scaling_ready = bool(
        stages
        and current_stage.get("automatic_scaling_allowed") is False
        and current_stage.get("operator_release_required_for_each_stage") is True
        and scaling_contract.get("automatic_scaling_allowed") is False
        and scaling_contract.get("account_growth_alone_may_increase_weight") is False
    )
    results.append(
        _result(
            _manifest_row(rows, "r06_predetermined_scaling_gates"),
            implementation_ready=scaling_ready,
            blockers=["predetermined_scaling_contract_incomplete" if not scaling_ready else ""],
            evidence={"stages": stages, "capital_scaling_contract": scaling_contract, "live_execution_authority": False},
        )
    )
    candidate_samples = _safe_int(candidate_flow.get("sample_count"))
    results.append(
        _result(
            _manifest_row(rows, "r07_monthly_investor_tear_sheet"),
            implementation_ready=True,
            organic_evidence_ready=candidate_samples > 0,
            blockers=["candidate_forward_tear_sheet_samples_pending" if candidate_samples <= 0 else ""],
            evidence={
                "output": str(_as_dict(config.get("outputs")).get("tear_sheet") or ""),
                "candidate_sample_count": candidate_samples,
                "paper_hypothetical_label_required": True,
            },
        )
    )
    results.append(
        _result(
            _manifest_row(rows, "r08_independent_accounting_review"),
            implementation_ready=bool(validator.get("implementation_ready")),
            external_evidence_ready=bool(accounting_attestation.get("valid")),
            blockers=[
                "internal_accounting_validator_missing" if not validator.get("implementation_ready") else "",
                "independent_accounting_review_pending" if not accounting_attestation.get("valid") else "",
            ],
            evidence={"internal_validator_evidence_ready": bool(validator.get("evidence_ready")), "attestation": accounting_attestation},
        )
    )
    required_data_room_sources = [
        "paper_performance",
        "profitability_independent_validator",
        "profitability_evidence_firewall",
        "multiple_testing_guard",
        "portfolio_capacity_curve",
        "production_resilience",
        "immutable_experiment_ledger",
    ]
    missing_data_room_sources = [source_id for source_id in required_data_room_sources if not sources.get(source_id)]
    invalid_attestations = [attestation_id for attestation_id, evidence in attestations.items() if not evidence.get("valid")]
    data_room_organic_complete = bool(not missing_data_room_sources and live_result_ready and divergence_ready)
    data_room_external_complete = not invalid_attestations
    results.append(
        _result(
            _manifest_row(rows, "r09_investor_data_room"),
            implementation_ready=True,
            organic_evidence_ready=data_room_organic_complete,
            external_evidence_ready=data_room_external_complete,
            blockers=[
                *[f"data_room_source_missing={source_id}" for source_id in missing_data_room_sources],
                *[f"external_attestation_pending={attestation_id}" for attestation_id in invalid_attestations],
                "broker_verified_live_performance_pending" if not live_result_ready else "",
                "paper_live_divergence_pending" if not divergence_ready else "",
            ],
            evidence={
                "output": str(_as_dict(config.get("outputs")).get("data_room_index") or ""),
                "required_source_count": len(required_data_room_sources),
                "present_source_count": len(required_data_room_sources) - len(missing_data_room_sources),
                "valid_external_attestation_count": len(attestations) - len(invalid_attestations),
                "external_attestation_count": len(attestations),
            },
        )
    )
    results.append(
        _result(
            _manifest_row(rows, "r10_legal_business_structure"),
            implementation_ready=bool(commercial),
            external_evidence_ready=bool(legal_attestation.get("valid")),
            blockers=[
                "commercial_mode_framework_missing" if not commercial else "",
                "qualified_legal_structure_review_pending" if not legal_attestation.get("valid") else "",
            ],
            evidence={"commercial_product_mode": commercial.get("commercial_product_mode", "missing"), "attestation": legal_attestation},
        )
    )
    return results


def _stage(results: list[dict[str, Any]]) -> str:
    by_id = {str(row.get("control_id")): row for row in results}
    if results and all(row.get("ready") for row in results):
        return "external_due_diligence_ready"
    if by_id.get("i01_broker_verified_live_results", {}).get("organic_evidence_ready"):
        return "live_evidence_building"
    if by_id.get("r04_complete_soak_before_canary", {}).get("ready"):
        return "founder_canary_ready"
    return "research_evidence_building"


def _recommended_actions(results: list[dict[str, Any]]) -> list[str]:
    status_by_id = {str(row.get("control_id")): row for row in results}
    actions = [
        "continue candidate-bound collection until post-cost expectancy, risk-of-ruin, and multiple-testing evidence become statistically available"
        if any(status_by_id.get(control_id, {}).get("status") == STATUS_EVIDENCE_PENDING for control_id in ("i02_net_of_all_costs", "i03_controlled_drawdowns", "i04_statistically_credible_edge"))
        else "",
        "do not select sleeves from lifetime rankings; shortlist only sleeves emitted by the profitability evidence firewall"
        if status_by_id.get("r01_shortlist_strong_sleeves", {}).get("status") != STATUS_READY
        else "",
        "collect capacity curves before increasing capital or concentration"
        if status_by_id.get("i05_capacity_evidence", {}).get("status") != STATUS_READY
        else "",
        "complete the unchanged-candidate 720-hour soak before any founder-funded canary"
        if status_by_id.get("r04_complete_soak_before_canary", {}).get("status") != STATUS_READY
        else "",
        "obtain real independent performance, accounting, IP, and legal reviews; the system cannot self-issue them"
        if any(row.get("status") == STATUS_EXTERNAL_REQUIRED or row.get("external_evidence_ready") is False for row in results)
        else "",
        "treat the $200 canary as an execution-validation envelope, not an income target or permission to scale",
        "future deposits may increase account equity but never increase strategy weight without every predefined evidence gate and an explicit operator release",
        "keep live execution disabled until the separate live transition and canary controls clear",
    ]
    return ordered_unique(item for item in actions if item)


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    config = load_json(config_path)
    sources, source_paths = _load_sources(project_root, config)
    attestations = _attestations(project_root, config)
    results = _control_results(config, sources, source_paths, attestations, project_root)
    counts = Counter(str(row.get("status") or STATUS_IMPLEMENTATION_GAP) for row in results)
    section_counts: dict[str, dict[str, int]] = {}
    for section in {str(row.get("section") or "unknown") for row in results}:
        section_rows = [row for row in results if row.get("section") == section]
        section_counts[section] = {
            "control_count": len(section_rows),
            "ready_count": sum(1 for row in section_rows if row.get("ready")),
        }
    implementation_gap_count = counts.get(STATUS_IMPLEMENTATION_GAP, 0)
    all_ready = bool(results and len(results) == 20 and counts.get(STATUS_READY, 0) == len(results))
    outputs = _as_dict(config.get("outputs"))
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "policy_id": str(config.get("policy_id") or "investor_readiness_v1"),
        "source": "investor_readiness_control",
        "ok": implementation_gap_count == 0,
        "overall_status": "ready" if all_ready else ("implementation_gap" if implementation_gap_count else "evidence_building"),
        "readiness_stage": _stage(results),
        "investor_due_diligence_ready": all_ready,
        "readiness_percentage_published": False,
        "readiness_percentage": None,
        "control_count": len(results),
        "status_counts": {status: counts.get(status, 0) for status in sorted(CONTROL_STATUSES)},
        "evidence_facet_counts": {
            "implementation_ready": sum(1 for row in results if row.get("implementation_ready") is True),
            "organic_evidence_ready": sum(1 for row in results if row.get("organic_evidence_ready") is True),
            "organic_evidence_pending": sum(1 for row in results if row.get("organic_evidence_ready") is False),
            "external_evidence_ready": sum(1 for row in results if row.get("external_evidence_ready") is True),
            "external_evidence_pending": sum(1 for row in results if row.get("external_evidence_ready") is False),
        },
        "section_counts": section_counts,
        "controls": results,
        "shortlisted_sleeves": _qualified_sleeves(
            sources.get("profitability_evidence_firewall", {}),
            _safe_int(config.get("maximum_shortlisted_sleeves"), 3),
        ),
        "capital_scaling_contract": _as_dict(config.get("capital_scaling_contract")),
        "safety_contract": _as_dict(config.get("safety_contract")),
        "external_attestations": attestations,
        "source_artifacts": source_paths,
        "artifact_paths": {
            key: str(value)
            for key, value in outputs.items()
        },
        "recommended_actions": _recommended_actions(results),
        "truth_contract": {
            "implementation_organic_live_and_external_evidence_are_separate": True,
            "control_grades_do_not_rewrite_economic_results": True,
            "lifetime_current_candidate_and_inventory_scopes_are_separate": True,
            "lifetime_rankings_cannot_select_canary_sleeves": True,
            "paper_results_are_hypothetical": True,
            "profitability_is_not_guaranteed": True,
            "this_control_grants_live_execution_authority": False,
            "this_control_grants_automatic_scaling_authority": False,
            "external_attestations_cannot_be_self_issued": True,
        },
    }


def _markdown_value(raw: Any) -> str:
    return str(raw if raw is not None else "").replace("|", "\\|").replace("\n", " ")


def render_packet(payload: dict[str, Any]) -> str:
    lines = [
        "# Investor Readiness Packet",
        "",
        f"Generated UTC: `{payload.get('timestamp_utc', '')}`",
        f"Stage: `{payload.get('readiness_stage', '')}`",
        f"Investor due-diligence ready: `{payload.get('investor_due_diligence_ready', False)}`",
        "Readiness percentage: `not published`; implementation, organic evidence, live evidence, and external review are intentionally separate.",
        "",
        "## Capital Boundary",
        "",
        "The initial `$200` canary is an execution-validation envelope, not an income target or permanent portfolio size. Future deposits may increase account equity, but account growth alone cannot increase strategy weight. Every stage still requires the predefined evidence gates and explicit operator release.",
        "",
        "## Twenty Controls",
        "",
        "| ID | Control | Class | Status | Blocking truth |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in _as_list(payload.get("controls")):
        if not isinstance(row, dict):
            continue
        blockers = ", ".join(str(item) for item in _as_list(row.get("blockers"))) or "none"
        lines.append(
            f"| `{_markdown_value(row.get('control_id'))}` | {_markdown_value(row.get('title'))} | "
            f"`{_markdown_value(row.get('completion_class'))}` | `{_markdown_value(row.get('status'))}` | {_markdown_value(blockers)} |"
        )
    lines.extend(["", "## Shortlist", ""])
    shortlist = _as_list(payload.get("shortlisted_sleeves"))
    if shortlist:
        for row in shortlist:
            lines.append(f"- `{_markdown_value(_as_dict(row).get('sleeve'))}`: independently qualified by the profitability evidence firewall")
    else:
        lines.append("- No sleeve is independently qualified yet. Lifetime rankings are not an allowed fallback.")
    lines.extend(["", "## Required Next Actions", ""])
    for action in _as_list(payload.get("recommended_actions")):
        lines.append(f"- {action}")
    lines.extend(
        [
            "",
            "## Authority Boundary",
            "",
            "This packet is read-only evidence. It does not enable live orders, select an allocation, promote a sleeve, approve marketing, accept customer funds, provide legal approval, or guarantee profitability.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_tear_sheet(payload: dict[str, Any], paper: dict[str, Any] | None = None) -> str:
    paper = paper or {}
    accounting = _as_dict(paper.get("accounting_views"))
    candidate = _as_dict(accounting.get("candidate_forward_flow"))
    current = _as_dict(accounting.get("current_day_flow"))
    lifetime = _as_dict(accounting.get("lifetime_flow"))
    inventory = _as_dict(accounting.get("active_book_snapshot"))
    post_cost = _as_dict(paper.get("post_cost_expectancy"))
    window = _as_dict(paper.get("profitability_evidence_window"))
    lines = [
        "# PAPER / HYPOTHETICAL - NOT LIVE PERFORMANCE",
        "",
        f"Generated UTC: `{payload.get('timestamp_utc', '')}`",
        f"Candidate ID: `{window.get('candidate_id', 'unavailable')}`",
        f"Candidate cutoff UTC: `{window.get('candidate_cutoff_utc', 'unavailable')}`",
        "",
        "## Candidate-Forward Promotion Scope",
        "",
        f"- Samples: `{_safe_int(candidate.get('sample_count'))}`",
        f"- Observed days: `{_safe_int(candidate.get('observed_days'))}`",
        f"- Post-cost P&L delta: `${_safe_float(candidate.get('post_cost_pnl_delta_total')):.2f}`",
        f"- Post-cost expectancy available: `{bool(post_cost.get('available'))}`",
        f"- Evidence sufficient: `{bool(post_cost.get('evidence_sufficient'))}`",
        f"- Positive clustered 95% lower bound: `{bool(post_cost.get('positive_clustered_lower_confidence_bound_95'))}`",
        "",
        "Only identity-bound candidate-forward observations may support promotion. No data means pending evidence, not a zero-return result and not readiness.",
        "",
        "## Separate Non-Promotion Scopes",
        "",
        f"- Current-day flow: samples=`{_safe_int(current.get('sample_count'))}`, post-cost P&L delta=`${_safe_float(current.get('post_cost_pnl_delta_total')):.2f}`.",
        f"- Lifetime schema-v2 flow: samples=`{_safe_int(lifetime.get('sample_count'))}`, post-cost P&L delta=`${_safe_float(lifetime.get('post_cost_pnl_delta_total')):.2f}`.",
        f"- Active inventory snapshot: ending net P&L=`${_safe_float(inventory.get('ending_net_pnl_total')):.2f}`, candidate-grade eligible=`{bool(inventory.get('candidate_grade_eligible'))}`.",
        "",
        "Lifetime history, current-day activity, candidate-forward flow, and carried inventory are different accounting truths and are never blended into one promotion number.",
        "",
        "## Qualified Sleeve Shortlist",
        "",
    ]
    shortlist = _as_list(payload.get("shortlisted_sleeves"))
    if shortlist:
        lines.extend(f"- `{_as_dict(row).get('sleeve', '')}`" for row in shortlist)
    else:
        lines.append("- None. The system will not substitute lifetime winners for candidate-qualified sleeves.")
    lines.extend(
        [
            "",
            "## Disclosure",
            "",
            "These are simulated paper results and research evidence, not live returns, audited performance, investment advice, an offer, or a promise of future results. Trading can lose money. Independent verification remains a separate external requirement.",
        ]
    )
    return "\n".join(lines) + "\n"


def _index_entry(project_root: Path, category: str, artifact_id: str, raw_path: Any, *, expected: bool = True) -> dict[str, Any]:
    path = _project_path(project_root, raw_path)
    present = path.is_file()
    return {
        "category": category,
        "artifact_id": artifact_id,
        "path": _relative(project_root, path),
        "expected": expected,
        "present": present,
        "size_bytes": path.stat().st_size if present else 0,
        "sha256": _sha256(path) if present else "",
    }


def build_data_room_index(project_root: Path, config: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    documents: list[dict[str, Any]] = []
    documents.append(_index_entry(project_root, "policy", "investor_readiness_policy", "config/investor_readiness_v1.json"))
    for artifact_id, raw_path in _as_dict(config.get("source_artifacts")).items():
        documents.append(_index_entry(project_root, "system_evidence", str(artifact_id), raw_path))
    for artifact_id, raw_path in _as_dict(config.get("outputs")).items():
        if str(artifact_id) in {"data_room_index", "data_room_readme"}:
            continue
        documents.append(_index_entry(project_root, "generated_packet", str(artifact_id), raw_path))
    for artifact_id, raw_path in _as_dict(config.get("future_live_evidence")).items():
        documents.append(_index_entry(project_root, "future_live_evidence", str(artifact_id), raw_path))
    for artifact_id, raw_path in _as_dict(config.get("external_attestations")).items():
        documents.append(_index_entry(project_root, "external_attestation", str(artifact_id), raw_path))
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "policy_id": str(config.get("policy_id") or "investor_readiness_v1"),
        "source": "investor_readiness_control",
        "readiness_stage": payload.get("readiness_stage"),
        "investor_due_diligence_ready": bool(payload.get("investor_due_diligence_ready")),
        "document_count": len(documents),
        "present_document_count": sum(1 for row in documents if row.get("present")),
        "missing_document_count": sum(1 for row in documents if not row.get("present")),
        "documents": documents,
        "authority_contract": {
            "index_is_not_external_verification": True,
            "missing_documents_remain_missing": True,
            "self_attestation_allowed": False,
            "live_execution_authority": False,
        },
    }


def render_data_room_readme(payload: dict[str, Any], index: dict[str, Any]) -> str:
    lines = [
        "# Investor Data Room Index",
        "",
        f"Generated UTC: `{index.get('timestamp_utc', '')}`",
        f"Readiness stage: `{payload.get('readiness_stage', '')}`",
        f"Present documents: `{index.get('present_document_count', 0)}/{index.get('document_count', 0)}`",
        "",
        "This directory is an index over canonical evidence. Missing live records and independent attestations are deliberately shown as missing; the software cannot manufacture or self-approve them.",
        "",
        "| Category | Artifact | Present | Path |",
        "| --- | --- | --- | --- |",
    ]
    for row in _as_list(index.get("documents")):
        if not isinstance(row, dict):
            continue
        lines.append(
            f"| `{_markdown_value(row.get('category'))}` | `{_markdown_value(row.get('artifact_id'))}` | "
            f"`{bool(row.get('present'))}` | `{_markdown_value(row.get('path'))}` |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This index is not an audit opinion, legal advice, marketing approval, live-trading authorization, or evidence of future profitability.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(
    project_root: Path,
    config: dict[str, Any],
    payload: dict[str, Any],
    *,
    out_path: Path,
    packet_path: Path,
    tear_sheet_path: Path,
    data_room_index_path: Path,
    data_room_readme_path: Path,
) -> dict[str, Any]:
    write_payload(out_path, payload)
    _write_text(packet_path, render_packet(payload))
    paper_path = _project_path(project_root, _as_dict(config.get("source_artifacts")).get("paper_performance"))
    _write_text(tear_sheet_path, render_tear_sheet(payload, load_json(paper_path)))
    data_room_index_path.parent.mkdir(parents=True, exist_ok=True)
    index = build_data_room_index(project_root, config, payload)
    write_payload(data_room_index_path, index)
    _write_text(data_room_readme_path, render_data_room_readme(payload, index))
    return index


def _output(project_root: Path, config: dict[str, Any], key: str, override: Path | None) -> Path:
    if override is not None:
        return override
    return _project_path(project_root, _as_dict(config.get("outputs")).get(key))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate twenty investor-readiness controls without granting live or scaling authority.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--packet", type=Path, default=None)
    parser.add_argument("--tear-sheet", type=Path, default=None)
    parser.add_argument("--data-room-index", type=Path, default=None)
    parser.add_argument("--data-room-readme", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    project_root = args.project_root.resolve()
    config_path = args.config or project_root / "config" / "investor_readiness_v1.json"
    config = load_json(config_path)
    payload = build_payload(project_root, config_path=config_path)
    write_outputs(
        project_root,
        config,
        payload,
        out_path=_output(project_root, config, "health", args.out),
        packet_path=_output(project_root, config, "packet", args.packet),
        tear_sheet_path=_output(project_root, config, "tear_sheet", args.tear_sheet),
        data_room_index_path=_output(project_root, config, "data_room_index", args.data_room_index),
        data_room_readme_path=_output(project_root, config, "data_room_readme", args.data_room_readme),
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        counts = _as_dict(payload.get("status_counts"))
        print(
            "investor_readiness_control "
            f"status={payload.get('overall_status')} "
            f"stage={payload.get('readiness_stage')} "
            f"ready={counts.get(STATUS_READY, 0)}/{payload.get('control_count', 0)} "
            f"evidence_pending={counts.get(STATUS_EVIDENCE_PENDING, 0)} "
            f"external_required={counts.get(STATUS_EXTERNAL_REQUIRED, 0)} "
            f"implementation_gaps={counts.get(STATUS_IMPLEMENTATION_GAP, 0)} "
            "live_authority=0"
        )
    # Evidence debt and external actions are expected lifecycle states, not process failures.
    return 0 if payload.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
