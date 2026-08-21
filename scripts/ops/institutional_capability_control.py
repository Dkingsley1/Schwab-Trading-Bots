#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import load_json, ordered_unique, parse_iso_utc, payload_age_minutes, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, load_json, ordered_unique, parse_iso_utc, payload_age_minutes, write_payload


DEFAULT_CONFIG = Path("config/institutional_capability_control_v1.json")
DEFAULT_OUT = Path("governance/health/institutional_capability_control_latest.json")
SCHEMA_VERSION = 1

ARTIFACT_SPECS: dict[str, tuple[str, float, str]] = {
    "strategy_specialization": (
        "governance/research/sleeve_strategy_specialization_latest.json",
        120.0,
        "./scripts/ops/opsctl.sh sleeve-strategy-specialization --json",
    ),
    "quantitative_challenger": (
        "governance/research/quantitative_challenger_latest.json",
        120.0,
        "./scripts/ops/opsctl.sh quantitative-challengers --json",
    ),
    "point_in_time_store": (
        "governance/health/point_in_time_event_store_latest.json",
        60.0,
        ".venv314/bin/python scripts/point_in_time_event_store.py --json",
    ),
    "immutable_experiment_ledger": (
        "governance/experiments/immutable_experiment_ledger_latest.json",
        64800.0,
        ".venv314/bin/python scripts/experiment_tracker.py --help",
    ),
    "multiple_testing": (
        "governance/research/multiple_testing_guard_latest.json",
        120.0,
        "./scripts/ops/opsctl.sh multiple-testing --json",
    ),
    "source_verification": (
        "governance/health/source_verification_latest.json",
        60.0,
        "./scripts/ops/opsctl.sh source-verification --json",
    ),
    "collector_capabilities": (
        "governance/health/collector_capability_control_latest.json",
        60.0,
        "./scripts/ops/opsctl.sh collector-capability-control --json",
    ),
    "independent_fills": (
        "governance/health/independent_fill_evidence_acquisition_latest.json",
        30.0,
        ".venv314/bin/python scripts/ops/independent_fill_evidence_acquisition.py --apply --json",
    ),
    "paper_calibration": (
        "governance/health/paper_execution_calibration_latest.json",
        60.0,
        ".venv314/bin/python scripts/paper_execution_calibration_report.py --json",
    ),
    "resource_governor": (
        "governance/health/autonomic_resource_governor_latest.json",
        30.0,
        "./scripts/ops/opsctl.sh autonomic-resource-governor --apply --json",
    ),
    "system_roles": (
        "governance/health/system_role_contract_latest.json",
        120.0,
        "./scripts/ops/opsctl.sh system-role-contract --json",
    ),
    "control_ownership": (
        "governance/health/control_surface_ownership_latest.json",
        120.0,
        "./scripts/ops/opsctl.sh control-surface-ownership --json",
    ),
    "live_order_ledger": (
        "governance/health/live_order_ledger_control_latest.json",
        60.0,
        "./scripts/ops/opsctl.sh live-order-ledger --json",
    ),
    "risk_service_boundary": (
        "governance/risk/risk_service_boundary_latest.json",
        60.0,
        ".venv314/bin/python scripts/risk_service_boundary.py --json",
    ),
    "live_reconciliation": (
        "governance/health/live_reconciliation_slo_latest.json",
        60.0,
        ".venv314/bin/python scripts/live_reconciliation_slo_guard.py --json",
    ),
    "live_canary": (
        "governance/health/live_canary_control_latest.json",
        120.0,
        "./scripts/ops/opsctl.sh live-canary-control --json",
    ),
}


def _resolve(project_root: Path, path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else project_root / candidate


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


def _status(payload: dict[str, Any]) -> str:
    return str(payload.get("overall_status") or payload.get("status") or "missing").strip().lower()


def _artifact(project_root: Path, relative: str, max_age_minutes: float, now: datetime) -> dict[str, Any]:
    path = _resolve(project_root, relative)
    payload = load_json(path)
    age = payload_age_minutes(payload, path, now=now) if payload else None
    present = bool(payload)
    fresh = bool(present and age is not None and age <= max_age_minutes)
    return {
        "path": str(path),
        "payload": payload,
        "present": present,
        "fresh": fresh,
        "state": "fresh" if fresh else ("stale" if present else "missing"),
        "age_minutes": round(float(age), 3) if age is not None else None,
        "max_age_minutes": float(max_age_minutes),
    }


def _candidate(project_root: Path) -> dict[str, Any]:
    path = project_root / "governance" / "runtime" / "production_candidate_state.json"
    payload = load_json(path)
    windows = payload.get("scope_windows_started_utc") if isinstance(payload.get("scope_windows_started_utc"), dict) else {}
    cutoffs = [parse_iso_utc(windows.get(scope)) for scope in ("execution", "data", "dependencies", "strategy")]
    cutoff = max((value for value in cutoffs if value is not None), default=None)
    candidate_id = str(payload.get("candidate_id") or "").strip()
    return {
        "candidate_id": candidate_id,
        "generation": _safe_int(payload.get("generation")),
        "cutoff_utc": cutoff.isoformat() if cutoff is not None else "",
        "bound": bool(candidate_id and cutoff is not None),
        "state_path": str(path),
    }


def _binding(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("candidate_binding")
    return value if isinstance(value, dict) else {}


def _candidate_matches(payload: dict[str, Any], candidate: dict[str, Any]) -> bool:
    binding = _binding(payload)
    observed = str(binding.get("candidate_id") or "").strip()
    return bool(
        candidate.get("bound")
        and observed == candidate.get("candidate_id")
        and binding.get("bound", True)
    )


def _pillar(
    pillar_id: str,
    title: str,
    *,
    implementation_ready: bool,
    paper_ready: bool,
    evidence_ready: bool,
    live_ready: bool,
    implementation_blockers: list[str],
    paper_blockers: list[str],
    evidence_blockers: list[str],
    live_blockers: list[str],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "pillar_id": pillar_id,
        "title": title,
        "implementation_ready": bool(implementation_ready),
        "paper_soak_ready": bool(paper_ready),
        "candidate_evidence_ready": bool(evidence_ready),
        "live_promotion_ready": bool(live_ready),
        "implementation_blockers": ordered_unique(implementation_blockers),
        "paper_blockers": ordered_unique(paper_blockers),
        "candidate_evidence_blockers": ordered_unique(evidence_blockers),
        "live_promotion_blockers": ordered_unique(live_blockers),
        "metrics": metrics,
    }


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    config_path: Path = DEFAULT_CONFIG,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    config_file = _resolve(project_root, config_path)
    config = load_json(config_file)
    candidate = _candidate(project_root)
    artifacts = {
        name: _artifact(project_root, relative, max_age, current)
        for name, (relative, max_age, _command) in ARTIFACT_SPECS.items()
    }
    payloads = {name: artifact["payload"] for name, artifact in artifacts.items()}

    specialization = payloads["strategy_specialization"]
    coverage = specialization.get("contract_coverage") if isinstance(specialization.get("contract_coverage"), dict) else {}
    library = specialization.get("strategy_library") if isinstance(specialization.get("strategy_library"), dict) else {}
    quality = specialization.get("quality_summary") if isinstance(specialization.get("quality_summary"), dict) else {}
    point_in_time = payloads["point_in_time_store"]
    immutable = payloads["immutable_experiment_ledger"]
    challenger = payloads["quantitative_challenger"]
    multiple = payloads["multiple_testing"]
    research_impl_blockers: list[str] = []
    if not artifacts["strategy_specialization"]["fresh"] or not specialization.get("ok", False):
        research_impl_blockers.append("strategy_specialization_not_fresh_and_ready")
    if _safe_int(coverage.get("complete_contract_count")) != _safe_int(coverage.get("strategy_count")) or _safe_int(coverage.get("strategy_count")) <= 0:
        research_impl_blockers.append("strategy_contract_coverage_incomplete")
    if _safe_int(library.get("complete_contract_count")) != _safe_int(library.get("strategy_count")) or _safe_int(library.get("strategy_count")) <= 0:
        research_impl_blockers.append("cold_strategy_library_lineage_incomplete")
    if not artifacts["point_in_time_store"]["fresh"] or not point_in_time.get("ok", False):
        research_impl_blockers.append("point_in_time_store_not_fresh_and_ready")
    ledger_rows = _safe_int(immutable.get("ledger_row_count"))
    if not immutable.get("append_only_ready", False) or ledger_rows <= 0 or _safe_int(immutable.get("signed_row_count")) != ledger_rows:
        research_impl_blockers.append("immutable_experiment_ledger_not_signed_and_append_only")
    research_implementation = not research_impl_blockers
    research_paper_blockers: list[str] = []
    if not _candidate_matches(specialization, candidate):
        research_paper_blockers.append("strategy_specialization_candidate_mismatch")
    if not _candidate_matches(challenger, candidate):
        research_paper_blockers.append("quantitative_challenger_candidate_mismatch")
    research_paper = bool(research_implementation and not research_paper_blockers)
    research_evidence_blockers: list[str] = []
    if not multiple.get("statistical_evidence_ready", False):
        research_evidence_blockers.append("candidate_statistical_evidence_pending")
    if _safe_int(quality.get("validated_good_count")) <= 0:
        research_evidence_blockers.append("validated_strategy_evidence_pending")
    if not immutable.get("latest_exact_replay_ready", False):
        research_evidence_blockers.append("exact_replay_bundle_pending")
    research_evidence = bool(research_paper and not research_evidence_blockers)
    research_live_blockers = list(research_evidence_blockers)
    if not immutable.get("latest_attestation_ready", False):
        research_live_blockers.append("independent_release_attestation_pending")
    research_live = bool(research_evidence and not research_live_blockers)

    source = payloads["source_verification"]
    source_rows = source.get("sources") if isinstance(source.get("sources"), list) else []
    verified_source_bundles = sum(1 for row in source_rows if isinstance(row, dict) and row.get("ok", False) and row.get("fresh", False))
    collector = payloads["collector_capabilities"]
    collector_summary = collector.get("summary") if isinstance(collector.get("summary"), dict) else {}
    coverage_debt = collector.get("coverage_debt") if isinstance(collector.get("coverage_debt"), dict) else {}
    data_impl_blockers: list[str] = []
    if not artifacts["source_verification"]["fresh"] or not source.get("ok", False):
        data_impl_blockers.append("source_verification_not_fresh_and_ready")
    if not artifacts["collector_capabilities"]["fresh"] or not collector.get("ok", False):
        data_impl_blockers.append("collector_capability_control_not_fresh_and_ready")
    if min(
        _safe_int(collector_summary.get("plane_count")),
        _safe_int(collector_summary.get("capability_count")),
        _safe_int(collector_summary.get("assignment_count")),
    ) <= 0:
        data_impl_blockers.append("collector_routing_inventory_empty")
    data_implementation = not data_impl_blockers
    data_paper_blockers: list[str] = []
    if coverage_debt.get("blocks_guarded_paper_soak", False):
        data_paper_blockers.append("required_paper_capability_gap")
    data_paper = bool(data_implementation and not data_paper_blockers)
    provider_policy = config.get("provider_policy") if isinstance(config.get("provider_policy"), dict) else {}
    provider_min = max(_safe_int(provider_policy.get("authoritative_provider_family_target_min"), 15), 1)
    data_evidence_blockers: list[str] = []
    if verified_source_bundles < provider_min:
        data_evidence_blockers.append("verified_authoritative_source_bundle_floor_pending")
    if _safe_float(collector_summary.get("required_capability_usable_ratio")) < 0.70:
        data_evidence_blockers.append("required_capability_usable_ratio_below_candidate_floor")
    data_evidence = bool(data_paper and not data_evidence_blockers)
    data_live_blockers = list(data_evidence_blockers)
    if _safe_int(collector_summary.get("runtime_live_ready_route_count")) <= 0:
        data_live_blockers.append("no_live_ready_data_routes")
    if _safe_int(coverage_debt.get("candidate_blocking_gap_count")) > 0:
        data_live_blockers.append("candidate_blocking_capability_debt")
    data_live = bool(data_evidence and not data_live_blockers)

    fills = payloads["independent_fills"]
    calibration = payloads["paper_calibration"]
    fill_contract = fills.get("control_contract") if isinstance(fills.get("control_contract"), dict) else {}
    execution_impl_blockers: list[str] = []
    for owner in (
        project_root / "core" / "execution_simulator.py",
        project_root / "scripts" / "ops" / "independent_fill_evidence_acquisition.py",
        project_root / "scripts" / "paper_execution_calibration_report.py",
    ):
        if not owner.is_file():
            execution_impl_blockers.append(f"owner_source_missing:{owner.name}")
    if not artifacts["independent_fills"]["fresh"] or not fill_contract.get("exact_candidate_identity_required", False):
        execution_impl_blockers.append("independent_fill_candidate_identity_contract_not_fresh")
    if not artifacts["paper_calibration"]["fresh"]:
        execution_impl_blockers.append("paper_calibration_not_fresh")
    execution_implementation = not execution_impl_blockers
    execution_paper_blockers: list[str] = []
    if not _candidate_matches(fills, candidate):
        execution_paper_blockers.append("independent_fill_artifact_candidate_mismatch")
    if not _candidate_matches(calibration, candidate):
        execution_paper_blockers.append("calibration_artifact_candidate_mismatch")
    if _safe_int(fills.get("conflict_count")) > 0:
        execution_paper_blockers.append("immutable_fill_identity_conflict")
    execution_paper = bool(execution_implementation and not execution_paper_blockers)
    execution_evidence_blockers: list[str] = []
    if _safe_int(fills.get("candidate_eligible_ledger_records")) < _safe_int(calibration.get("minimum_independent_samples"), 30):
        execution_evidence_blockers.append("minimum_independent_fill_sample_floor_pending")
    if not calibration.get("independent_evidence_ready", False):
        execution_evidence_blockers.append("independent_slippage_calibration_pending")
    execution_evidence = bool(execution_paper and not execution_evidence_blockers)
    execution_live_blockers = list(execution_evidence_blockers)
    live_reconciliation = payloads["live_reconciliation"]
    live_metrics = live_reconciliation.get("metrics") if isinstance(live_reconciliation.get("metrics"), dict) else {}
    if not live_reconciliation.get("ok", False) or _safe_int(live_metrics.get("reconcile_events")) <= 0:
        execution_live_blockers.append("broker_verified_live_reconciliation_evidence_pending")
    execution_live = bool(execution_evidence and not execution_live_blockers)

    overfit_impl_blockers: list[str] = []
    if not artifacts["multiple_testing"]["fresh"] or not multiple.get("contract_present", False):
        overfit_impl_blockers.append("multiple_testing_contract_not_fresh")
    if _safe_int(multiple.get("family_size")) <= 0:
        overfit_impl_blockers.append("experiment_family_empty")
    if not immutable.get("append_only_ready", False):
        overfit_impl_blockers.append("immutable_experiment_lineage_missing")
    overfit_implementation = not overfit_impl_blockers
    overfit_paper_blockers: list[str] = []
    if not _candidate_matches(multiple, candidate):
        overfit_paper_blockers.append("multiple_testing_candidate_mismatch")
    overfit_paper = bool(overfit_implementation and not overfit_paper_blockers)
    overfit_evidence_blockers: list[str] = []
    if not multiple.get("statistical_evidence_ready", False):
        overfit_evidence_blockers.extend(
            str(item) for item in multiple.get("statistical_evidence_blockers", []) if str(item).strip()
        )
    if not immutable.get("latest_exact_replay_ready", False):
        overfit_evidence_blockers.append("exact_replay_bundle_pending")
    overfit_evidence = bool(overfit_paper and not overfit_evidence_blockers)
    overfit_live_blockers = list(overfit_evidence_blockers)
    if not immutable.get("latest_attestation_ready", False):
        overfit_live_blockers.append("independent_experiment_attestation_pending")
    overfit_live = bool(overfit_evidence and not overfit_live_blockers)

    resource = payloads["resource_governor"]
    roles = payloads["system_roles"]
    ownership = payloads["control_ownership"]
    resource_budgets = resource.get("budgets") if isinstance(resource.get("budgets"), dict) else {}
    pressure = resource_budgets.get("runtime_pressure_source") if isinstance(resource_budgets.get("runtime_pressure_source"), dict) else {}
    pressure_attribution = pressure.get("attribution") if isinstance(pressure.get("attribution"), dict) else {}
    collector_budget = resource_budgets.get("collectors") if isinstance(resource_budgets.get("collectors"), dict) else {}
    live_loop_budget = resource_budgets.get("live_loops") if isinstance(resource_budgets.get("live_loops"), dict) else {}
    resource_impl_blockers: list[str] = []
    if not artifacts["resource_governor"]["fresh"] or not resource.get("ok", False):
        resource_impl_blockers.append("resource_governor_not_fresh_and_ready")
    if not artifacts["system_roles"]["fresh"] or not roles.get("ok", False):
        resource_impl_blockers.append("system_role_contract_not_fresh_and_ready")
    if not artifacts["control_ownership"]["fresh"] or not ownership.get("ok", False):
        resource_impl_blockers.append("control_surface_ownership_not_fresh_and_ready")
    resource_implementation = not resource_impl_blockers
    resource_paper_blockers: list[str] = []
    paper_pressure_dominant = bool(
        pressure_attribution.get("paper_execution_pressure_dominant", False)
        or pressure.get("protected_work_hot", False)
    )
    if pressure.get("runtime_hot", False) and paper_pressure_dominant:
        resource_paper_blockers.append("paper_runtime_pressure_hot")
    if str(pressure.get("memory_pressure_level") or "unknown").lower() not in {"normal", "low"}:
        resource_paper_blockers.append("memory_pressure_not_normal")
    collector_mode = str(collector_budget.get("mode") or "unknown").lower()
    if collector_mode in {"blocked", "halted", "paused"}:
        resource_paper_blockers.append("required_collection_lane_not_running")
    live_loop_mode = str(live_loop_budget.get("mode") or "unknown").lower()
    if live_loop_mode in {"blocked", "halted", "paused"}:
        resource_paper_blockers.append("required_observation_lane_not_running")
    resource_paper = bool(resource_implementation and not resource_paper_blockers)
    resource_evidence = resource_paper
    resource_live = resource_paper

    live_ledger = payloads["live_order_ledger"]
    risk = payloads["risk_service_boundary"]
    service_boundary = risk.get("independent_service_boundary") if isinstance(risk.get("independent_service_boundary"), dict) else {}
    pre_trade = (risk.get("services") or {}).get("pre_trade_service", {}) if isinstance(risk.get("services"), dict) else {}
    market_impl_blockers: list[str] = []
    if not artifacts["live_order_ledger"]["fresh"] or not live_ledger.get("ok", False):
        market_impl_blockers.append("transactional_live_order_ledger_not_fresh_and_ready")
    if not artifacts["risk_service_boundary"]["fresh"] or not service_boundary.get("service_isolation_ready", False):
        market_impl_blockers.append("independent_risk_service_boundary_not_ready")
    for owner in (
        project_root / "core" / "live_execution_controls.py",
        project_root / "scripts" / "global_risk_killswitch.py",
    ):
        if not owner.is_file():
            market_impl_blockers.append(f"owner_source_missing:{owner.name}")
    market_implementation = not market_impl_blockers
    market_paper_blockers: list[str] = []
    if live_ledger.get("live_execution_authority", False):
        market_paper_blockers.append("unexpected_live_execution_authority")
    market_paper = bool(market_implementation and not market_paper_blockers)
    market_evidence_blockers: list[str] = []
    if _safe_int(pre_trade.get("evaluated_orders")) <= 0:
        market_evidence_blockers.append("pre_trade_decision_evidence_pending")
    input_health = risk.get("input_health") if isinstance(risk.get("input_health"), dict) else {}
    if not input_health.get("sources_ready", False):
        market_evidence_blockers.append("risk_service_operational_inputs_not_ready")
    market_evidence = bool(market_paper and not market_evidence_blockers)
    market_live_blockers = list(market_evidence_blockers)
    live_canary = payloads["live_canary"]
    if not live_canary.get("live_execution_allowed", False):
        market_live_blockers.append("operator_live_release_and_canary_clearance_pending")
    market_live = bool(market_evidence and not market_live_blockers)

    configured_pillars = {
        str(row.get("pillar_id") or ""): row
        for row in config.get("pillars", [])
        if isinstance(row, dict)
    }
    pillars = [
        _pillar(
            "scientific_research_platform",
            str((configured_pillars.get("scientific_research_platform") or {}).get("title") or "Scientific research and reproducibility"),
            implementation_ready=research_implementation,
            paper_ready=research_paper,
            evidence_ready=research_evidence,
            live_ready=research_live,
            implementation_blockers=research_impl_blockers,
            paper_blockers=research_paper_blockers,
            evidence_blockers=research_evidence_blockers,
            live_blockers=research_live_blockers,
            metrics={
                "hot_strategy_contracts": _safe_int(coverage.get("strategy_count")),
                "conceptual_strategy_hypotheses": _safe_int(library.get("strategy_count")),
                "validated_good_strategies": _safe_int(quality.get("validated_good_count")),
                "immutable_experiment_rows": ledger_rows,
            },
        ),
        _pillar(
            "market_visibility_and_data_lineage",
            str((configured_pillars.get("market_visibility_and_data_lineage") or {}).get("title") or "Market visibility, point-in-time data, and lineage"),
            implementation_ready=data_implementation,
            paper_ready=data_paper,
            evidence_ready=data_evidence,
            live_ready=data_live,
            implementation_blockers=data_impl_blockers,
            paper_blockers=data_paper_blockers,
            evidence_blockers=data_evidence_blockers,
            live_blockers=data_live_blockers,
            metrics={
                "verified_source_bundles": verified_source_bundles,
                "collector_planes": _safe_int(collector_summary.get("plane_count")),
                "logical_capabilities": _safe_int(collector_summary.get("capability_count")),
                "shared_producers": _safe_int(collector_summary.get("producer_count")),
                "required_capability_usable_ratio": _safe_float(collector_summary.get("required_capability_usable_ratio")),
                "candidate_blocking_capability_gaps": _safe_int(coverage_debt.get("candidate_blocking_gap_count")),
            },
        ),
        _pillar(
            "independent_execution_evidence",
            str((configured_pillars.get("independent_execution_evidence") or {}).get("title") or "Independent fills and transaction-cost calibration"),
            implementation_ready=execution_implementation,
            paper_ready=execution_paper,
            evidence_ready=execution_evidence,
            live_ready=execution_live,
            implementation_blockers=execution_impl_blockers,
            paper_blockers=execution_paper_blockers,
            evidence_blockers=execution_evidence_blockers,
            live_blockers=execution_live_blockers,
            metrics={
                "lifetime_independent_fill_records": _safe_int(fills.get("accepted_ledger_records")),
                "candidate_independent_fill_records": _safe_int(fills.get("candidate_eligible_ledger_records")),
                "minimum_independent_fill_records": _safe_int(calibration.get("minimum_independent_samples"), 30),
                "calibration_mae_bps": _safe_float((calibration.get("metrics") or {}).get("mae_bps") if isinstance(calibration.get("metrics"), dict) else 0.0),
            },
        ),
        _pillar(
            "selection_bias_and_overfit_control",
            str((configured_pillars.get("selection_bias_and_overfit_control") or {}).get("title") or "Selection-bias and overfit control"),
            implementation_ready=overfit_implementation,
            paper_ready=overfit_paper,
            evidence_ready=overfit_evidence,
            live_ready=overfit_live,
            implementation_blockers=overfit_impl_blockers,
            paper_blockers=overfit_paper_blockers,
            evidence_blockers=overfit_evidence_blockers,
            live_blockers=overfit_live_blockers,
            metrics={
                "conservative_hypothesis_family_size": _safe_int(multiple.get("family_size")),
                "correction_method": str(multiple.get("correction_method") or "missing"),
                "statistical_evidence_ready": bool(multiple.get("statistical_evidence_ready", False)),
                "exact_replay_ready": bool(immutable.get("latest_exact_replay_ready", False)),
            },
        ),
        _pillar(
            "resource_routing_and_role_separation",
            str((configured_pillars.get("resource_routing_and_role_separation") or {}).get("title") or "Bounded compute routing and role separation"),
            implementation_ready=resource_implementation,
            paper_ready=resource_paper,
            evidence_ready=resource_evidence,
            live_ready=resource_live,
            implementation_blockers=resource_impl_blockers,
            paper_blockers=resource_paper_blockers,
            evidence_blockers=[],
            live_blockers=[],
            metrics={
                "resource_status": _status(resource),
                "runtime_hot": bool(pressure.get("runtime_hot", False)),
                "runtime_hot_advisory_only": bool(pressure.get("runtime_hot", False) and not paper_pressure_dominant),
                "paper_pressure_dominant": paper_pressure_dominant,
                "memory_pressure_level": str(pressure.get("memory_pressure_level") or "unknown"),
                "collector_mode": collector_mode,
                "live_loop_mode": live_loop_mode,
                "role_count": _safe_int((roles.get("summary") or {}).get("role_count") if isinstance(roles.get("summary"), dict) else 0),
                "authority_conflicts": _safe_int((roles.get("summary") or {}).get("authority_conflict_count") if isinstance(roles.get("summary"), dict) else 0),
            },
        ),
        _pillar(
            "market_access_risk_controls",
            str((configured_pillars.get("market_access_risk_controls") or {}).get("title") or "Pre-trade market-access risk controls"),
            implementation_ready=market_implementation,
            paper_ready=market_paper,
            evidence_ready=market_evidence,
            live_ready=market_live,
            implementation_blockers=market_impl_blockers,
            paper_blockers=market_paper_blockers,
            evidence_blockers=market_evidence_blockers,
            live_blockers=market_live_blockers,
            metrics={
                "isolated_risk_service_count": _safe_int(service_boundary.get("service_count")),
                "pre_trade_orders_evaluated": _safe_int(pre_trade.get("evaluated_orders")),
                "risk_inputs_ready": bool(input_health.get("sources_ready", False)),
                "live_order_ledger_ready": bool(live_ledger.get("ok", False)),
                "live_execution_authority": False,
            },
        ),
    ]

    implementation_ready_count = sum(1 for row in pillars if row["implementation_ready"])
    paper_ready_count = sum(1 for row in pillars if row["paper_soak_ready"])
    evidence_ready_count = sum(1 for row in pillars if row["candidate_evidence_ready"])
    live_ready_count = sum(1 for row in pillars if row["live_promotion_ready"])
    pillar_count = len(pillars)
    local_refresh_actions = []
    for name, artifact in artifacts.items():
        if artifact["state"] in {"missing", "stale"}:
            local_refresh_actions.append(
                {
                    "artifact_id": name,
                    "state": artifact["state"],
                    "command": ARTIFACT_SPECS[name][2],
                    "automatic_live_authority": False,
                }
            )

    entitlements = config.get("conditional_external_entitlements") if isinstance(config.get("conditional_external_entitlements"), list) else []
    conditional_entitlements = [
        {
            "entitlement_id": str(row.get("entitlement_id") or ""),
            "provider": str(row.get("provider") or ""),
            "status": str(row.get("status") or "unknown"),
            "required_only_for_live_families": list(row.get("required_only_for_live_families") or []),
            "blocks_guarded_paper_soak": False,
            "self_healable": False,
        }
        for row in entitlements
        if isinstance(row, dict)
    ]
    external_actions = [
        {
            "need": "independent_fill_evidence",
            "reason": "broker paper receipts or licensed venue replay must be observed outside the expected-fill model",
            "self_healable": False,
        },
        {
            "need": "exact_replay_and_independent_attestation",
            "reason": "the system cannot self-issue independent validation",
            "self_healable": False,
        },
        {
            "need": "conditional_market_data_entitlements",
            "reason": "purchase only when an activated live family declares depth, low-latency news, estimates, or borrow as required",
            "self_healable": False,
        },
        {
            "need": "operator_live_release",
            "reason": "no control plane may unlock live execution automatically",
            "self_healable": False,
        },
    ]

    if implementation_ready_count < pillar_count:
        overall_status = "blocked"
    elif paper_ready_count < pillar_count:
        overall_status = "paper_attention"
    elif live_ready_count == pillar_count:
        overall_status = "ready"
    else:
        overall_status = "ready_with_evidence_debt"
    artifact_public = {
        name: {key: value for key, value in artifact.items() if key != "payload"}
        for name, artifact in artifacts.items()
    }
    provider_max = max(_safe_int(provider_policy.get("authoritative_provider_family_target_max"), 30), provider_min)
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": current.isoformat(),
        "policy_id": str(config.get("policy_id") or "missing"),
        "overall_status": overall_status,
        "ok": bool(implementation_ready_count == pillar_count and paper_ready_count == pillar_count),
        "candidate_binding": candidate,
        "summary": {
            "pillar_count": pillar_count,
            "implementation_ready_count": implementation_ready_count,
            "paper_soak_ready_count": paper_ready_count,
            "candidate_evidence_ready_count": evidence_ready_count,
            "live_promotion_ready_count": live_ready_count,
            "verified_source_bundle_count": verified_source_bundles,
            "conditional_external_entitlement_count": len(conditional_entitlements),
            "local_refresh_action_count": len(local_refresh_actions),
        },
        "provider_policy": {
            **provider_policy,
            "target_range": [provider_min, provider_max],
            "verified_source_bundle_count": verified_source_bundles,
            "ten_thousand_sources_required": False,
            "readiness_interpretation": "authority, freshness, lineage, independent corroboration, and strategy need matter more than raw source count",
        },
        "pillars": pillars,
        "paper_soak_ready": paper_ready_count == pillar_count,
        "candidate_evidence_ready": evidence_ready_count == pillar_count,
        "live_promotion_ready": live_ready_count == pillar_count,
        "paper_blockers": [row["pillar_id"] for row in pillars if not row["paper_soak_ready"]],
        "candidate_evidence_debt": [row["pillar_id"] for row in pillars if not row["candidate_evidence_ready"]],
        "live_promotion_debt": [row["pillar_id"] for row in pillars if not row["live_promotion_ready"]],
        "conditional_external_entitlements": conditional_entitlements,
        "bounded_local_refresh_actions": local_refresh_actions,
        "external_or_human_actions": external_actions,
        "artifacts": artifact_public,
        "control_contract": {
            "implementation_paper_evidence_entitlement_and_live_states_are_separate": True,
            "source_count_does_not_create_readiness_credit": True,
            "optional_unsubscribed_depth_does_not_block_guarded_paper_soak": True,
            "local_refresh_routes_are_bounded_and_non_mutating_to_strategy_or_orders": True,
            "subscriptions_fills_attestations_and_operator_release_are_not_self_healable": True,
            "candidate_mismatch_fails_closed": True,
            "automatic_source_purchase_authority": False,
            "automatic_strategy_admission_authority": False,
            "automatic_live_execution_authority": False,
            "profitability_is_not_guaranteed": True,
        },
        "source_files": {
            "config": str(config_file),
            "owner": str(project_root / "scripts" / "ops" / "institutional_capability_control.py"),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the unified institutional-capability readiness contract.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-file", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    project_root = args.project_root.expanduser().resolve()
    payload = build_payload(project_root, config_path=args.config)
    write_payload(_resolve(project_root, args.out_file), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        summary = payload["summary"]
        print(
            "institutional_capability_control "
            f"status={payload['overall_status']} "
            f"implementation={summary['implementation_ready_count']}/{summary['pillar_count']} "
            f"paper={summary['paper_soak_ready_count']}/{summary['pillar_count']} "
            f"evidence={summary['candidate_evidence_ready_count']}/{summary['pillar_count']} "
            f"live={summary['live_promotion_ready_count']}/{summary['pillar_count']}"
        )
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
