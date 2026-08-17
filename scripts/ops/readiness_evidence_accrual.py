#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import load_json, ordered_unique, parse_iso_utc, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, load_json, ordered_unique, parse_iso_utc, write_payload


DEFAULT_OUT = Path("governance/health/readiness_evidence_accrual_latest.json")
DEFAULT_STATE = Path("governance/runtime/readiness_evidence_accrual_state.json")
DEFAULT_HISTORY = Path("governance/evidence/readiness_evidence_accrual_history.jsonl")
SCHEMA_VERSION = 2
EASTERN = ZoneInfo("America/New_York")


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        value = float(raw)
    except Exception:
        return float(default)
    return value if math.isfinite(value) else float(default)


def _safe_int(raw: Any, default: int = 0) -> int:
    return int(_safe_float(raw, float(default)))


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _resolve(project_root: Path, path: Path) -> Path:
    return path.expanduser() if path.is_absolute() else project_root / path


def _candidate(project_root: Path) -> dict[str, Any]:
    state = load_json(project_root / "governance" / "runtime" / "production_candidate_state.json")
    production = load_json(project_root / "governance" / "health" / "production_excellence_control_latest.json")
    production_candidate = _as_dict(production.get("candidate"))
    windows = _as_dict(state.get("scope_windows_started_utc"))
    window_values = [parse_iso_utc(value) for value in windows.values()]
    soak_started = max((value for value in window_values if value is not None), default=None)
    identity_bound = bool(str(state.get("candidate_id") or "").strip() and soak_started is not None)
    production_contract_present = bool(production_candidate)
    production_candidate_matches = bool(
        str(production_candidate.get("candidate_id") or "").strip()
        == str(state.get("candidate_id") or "").strip()
    )
    credit_eligible = bool(
        identity_bound
        and (
            not production_contract_present
            or (
                production_candidate_matches
                and production_candidate.get("candidate_ready", False)
                and not production_candidate.get("candidate_drift", True)
            )
        )
    )
    return {
        "candidate_id": str(state.get("candidate_id") or "").strip(),
        "generation": _safe_int(state.get("generation"), 0),
        "soak_started_utc": soak_started.isoformat() if soak_started is not None else "",
        "soak_started": soak_started,
        "bound": identity_bound,
        "credit_eligible": credit_eligible,
        "production_contract_present": production_contract_present,
        "production_candidate_matches": production_candidate_matches,
        "candidate_drift": bool(production_candidate.get("candidate_drift", False)),
    }


def _runtime_collection_active(project_root: Path) -> tuple[bool, dict[str, Any]]:
    watchdog = load_json(project_root / "governance" / "health" / "process_watchdog_latest.json")
    live_runtime = load_json(project_root / "governance" / "health" / "live_runtime_status_latest.json")
    watchdog_status = str(watchdog.get("overall_status") or watchdog.get("status") or "").strip().lower()
    runtime_status = str(live_runtime.get("overall_status") or live_runtime.get("status") or "").strip().lower()
    process_count = max(
        _safe_int(watchdog.get("active_process_count"), 0),
        _safe_int(_as_dict(watchdog.get("summary")).get("active_process_count"), 0),
    )
    active = bool(watchdog_status in {"ready", "ok", "active"} or runtime_status in {"ready", "ok", "active"} or process_count > 0)
    return active, {
        "watchdog_status": watchdog_status,
        "live_runtime_status": runtime_status,
        "active_process_count": process_count,
    }


def _metric(
    metric_id: str,
    label: str,
    current: float,
    target: float,
    *,
    unit: str,
    kind: str,
    producer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    remaining = max(float(target) - float(current), 0.0)
    progress = 1.0 if target <= 0.0 and current >= target else max(0.0, min(float(current) / float(target), 1.0)) if target > 0.0 else 0.0
    return {
        "metric_id": metric_id,
        "label": label,
        "current": round(float(current), 6),
        "target": round(float(target), 6),
        "remaining": round(remaining, 6),
        "progress_ratio": round(progress, 6),
        "unit": unit,
        "kind": kind,
        "complete": bool(current >= target),
        "producer": producer
        or {"producer_id": "clock", "ready": True, "schedule_active": True, "monotonic_within_candidate": True},
    }


def _producer(
    producer_id: str,
    *,
    ready: bool,
    status: str,
    reason: str = "",
    cadence_hours: float = 6.0,
    schedule_active: bool = True,
    schedule: str = "continuous",
    monotonic_within_candidate: bool = True,
    binding_key: str = "",
) -> dict[str, Any]:
    return {
        "producer_id": producer_id,
        "ready": bool(ready),
        "status": str(status or "unknown"),
        "reason": str(reason or ""),
        "expected_cadence_hours": max(float(cadence_hours), 0.25),
        "schedule": schedule,
        "schedule_active": bool(schedule_active),
        "monotonic_within_candidate": bool(monotonic_within_candidate),
        "binding_key": str(binding_key or ""),
    }


def _equity_evidence_schedule_active(now: datetime) -> bool:
    local = now.astimezone(EASTERN)
    minutes = local.hour * 60 + local.minute
    return bool(local.weekday() < 5 and 4 * 60 <= minutes <= 20 * 60 + 15)


def _candidate_bound_artifact_state(
    payload: dict[str, Any],
    *,
    candidate: dict[str, Any],
    cutoff_utc: Any = "",
) -> tuple[bool, str]:
    """Return candidate coherence and a stable producer-window key."""
    candidate_id = str(candidate.get("candidate_id") or "").strip()
    candidate_started_utc = str(candidate.get("soak_started_utc") or "").strip()
    binding = _as_dict(payload.get("candidate_binding"))
    bound_candidate_id = str(binding.get("candidate_id") or "").strip()
    bound_cutoff_utc = str(binding.get("cutoff_utc") or cutoff_utc or "").strip()

    if bound_candidate_id:
        current = bool(
            candidate_id
            and bound_candidate_id == candidate_id
            and binding.get("bound", True)
        )
        return current, ":".join(
            [bound_candidate_id, bound_cutoff_utc, "current" if current else "stale"]
        )

    explicit_cutoff = parse_iso_utc(bound_cutoff_utc)
    candidate_started = candidate.get("soak_started") if isinstance(candidate.get("soak_started"), datetime) else None
    if explicit_cutoff is not None and candidate_started is not None:
        current = explicit_cutoff == candidate_started
        return current, ":".join(
            [candidate_id, bound_cutoff_utc, "current" if current else "stale"]
        )

    artifact_timestamp = parse_iso_utc(payload.get("timestamp_utc"))
    if artifact_timestamp is not None and candidate_started is not None:
        current = artifact_timestamp >= candidate_started
        return current, ":".join(
            [candidate_id, candidate_started_utc, "current" if current else "stale"]
        )

    # Older artifacts and compact unit-test fixtures may not carry an epoch yet.
    current = bool(payload)
    return current, ":".join(
        [candidate_id, candidate_started_utc, "legacy" if current else "missing"]
    )


def _artifact_producer(
    producer_id: str,
    payload: dict[str, Any],
    *,
    candidate: dict[str, Any],
    cutoff_utc: Any = "",
    cadence_hours: float = 6.0,
    schedule_active: bool = True,
    schedule: str = "continuous",
    monotonic_within_candidate: bool = False,
) -> tuple[dict[str, Any], bool]:
    current, binding_key = _candidate_bound_artifact_state(
        payload,
        candidate=candidate,
        cutoff_utc=cutoff_utc,
    )
    status = str(payload.get("overall_status") or payload.get("status") or "ready" if payload else "missing")
    if payload and not current:
        status = "stale_candidate_epoch"
    reason = "" if payload and current else (
        f"{producer_id}_candidate_epoch_stale" if payload else f"{producer_id}_artifact_missing"
    )
    return (
        _producer(
            producer_id,
            ready=bool(payload and current),
            status=status,
            reason=reason,
            cadence_hours=cadence_hours,
            schedule_active=schedule_active,
            schedule=schedule,
            monotonic_within_candidate=monotonic_within_candidate,
            binding_key=binding_key,
        ),
        current,
    )


def _raw_metrics(project_root: Path, *, candidate: dict[str, Any], now: datetime) -> list[dict[str, Any]]:
    acquisition = load_json(project_root / "governance" / "health" / "independent_fill_evidence_acquisition_latest.json")
    replay_capture = load_json(project_root / "governance" / "health" / "market_replay_fill_capture_latest.json")
    calibration = load_json(project_root / "governance" / "health" / "paper_execution_calibration_latest.json")
    performance = load_json(project_root / "governance" / "health" / "paper_performance_latest.json")
    post_cost = _as_dict(performance.get("post_cost_expectancy"))
    robust = _as_dict(post_cost.get("robust_statistics"))
    robust_thresholds = _as_dict(robust.get("thresholds"))
    promotion = load_json(project_root / "governance" / "health" / "promotion_quality_gate_latest.json")
    promotion_details = _as_dict(promotion.get("details"))
    promotion_scope = _as_dict(promotion_details.get("promotion"))
    advancement = load_json(project_root / "governance" / "health" / "promotion_candidate_advancement_latest.json")
    rollout = load_json(project_root / "governance" / "health" / "canary_rollout_latest.json")
    rollout_thresholds = _as_dict(rollout.get("thresholds"))
    rollout_canary = _as_dict(rollout.get("canary_statistics"))
    profitability = load_json(project_root / "governance" / "health" / "paper_profitability_control_latest.json")
    profitability_firewall = load_json(
        project_root / "governance" / "health" / "profitability_evidence_firewall_latest.json"
    )
    profitability_policy = load_json(project_root / "config" / "profitability_evidence_firewall_v1.json")
    multiple_testing = load_json(project_root / "governance" / "research" / "multiple_testing_guard_latest.json")
    holdout = load_json(project_root / "governance" / "research" / "profitability_holdout_vault_latest.json")
    benchmark = load_json(project_root / "governance" / "research" / "profitability_benchmark_hurdle_latest.json")
    decay = load_json(project_root / "governance" / "research" / "decay_monitor_latest.json")
    validator = load_json(project_root / "governance" / "health" / "profitability_independent_validator_latest.json")
    target_contract = _as_dict(profitability.get("a_plus_target_contract"))
    target_thresholds = _as_dict(target_contract.get("thresholds"))
    target_current = _as_dict(target_contract.get("current"))
    soak_started = candidate.get("soak_started") if isinstance(candidate.get("soak_started"), datetime) else None
    soak_hours = (
        max((now - soak_started).total_seconds() / 3600.0, 0.0)
        if soak_started is not None and candidate.get("credit_eligible", False)
        else 0.0
    )
    minimum_cohort_samples = _safe_float(rollout_thresholds.get("minimum_samples_per_cohort"), 400.0)
    minimum_canary_days = _safe_float(rollout_thresholds.get("minimum_independent_days"), 3.0)
    minimum_canary_ess = _safe_float(rollout_thresholds.get("minimum_effective_samples"), 50.0)
    candidate_id = str(candidate.get("candidate_id") or "")
    candidate_started = candidate.get("soak_started") if isinstance(candidate.get("soak_started"), datetime) else None

    acquisition_current, acquisition_binding_key = _candidate_bound_artifact_state(
        acquisition,
        candidate=candidate,
    )
    calibration_window = _as_dict(calibration.get("calibration_window"))
    calibration_current, calibration_binding_key = _candidate_bound_artifact_state(
        calibration,
        candidate=candidate,
        cutoff_utc=calibration_window.get("cutoff_utc"),
    )
    performance_window = _as_dict(performance.get("profitability_evidence_window"))
    performance_producer, performance_current = _artifact_producer(
        "paper_performance",
        performance,
        candidate=candidate,
        cutoff_utc=performance_window.get("candidate_cutoff_utc"),
    )
    if not performance_current:
        post_cost = {}
        robust = {}
        robust_thresholds = {}
    daily_performance_producer = {**performance_producer, "expected_cadence_hours": 30.0}
    cumulative_performance_producer = {
        **daily_performance_producer,
        "monotonic_within_candidate": True,
    }

    profitability_producer, profitability_firewall_current = _artifact_producer(
        "profitability_evidence_firewall",
        profitability_firewall,
        candidate=candidate,
    )
    multiple_testing_producer, multiple_testing_current = _artifact_producer(
        "multiple_testing_guard",
        multiple_testing,
        candidate=candidate,
    )
    holdout_producer, holdout_current = _artifact_producer(
        "profitability_holdout_vault",
        holdout,
        candidate=candidate,
    )
    benchmark_producer, benchmark_current = _artifact_producer(
        "profitability_benchmark_hurdle",
        benchmark,
        candidate=candidate,
        cadence_hours=30.0,
        monotonic_within_candidate=True,
    )
    benchmark_gate_producer = {**benchmark_producer, "monotonic_within_candidate": False}
    decay_producer, decay_current = _artifact_producer(
        "decay_monitor",
        decay,
        candidate=candidate,
    )
    validator_producer, validator_current = _artifact_producer(
        "profitability_independent_validator",
        validator,
        candidate=candidate,
    )
    cumulative_validator_producer = {
        **validator_producer,
        "expected_cadence_hours": 30.0,
        "monotonic_within_candidate": True,
    }

    advancement_timestamp = parse_iso_utc(advancement.get("timestamp_utc"))
    promotion_timestamp = parse_iso_utc(promotion.get("timestamp_utc"))
    advancement_current = bool(
        advancement
        and (
            candidate_started is None
            or advancement_timestamp is None
            or advancement_timestamp >= candidate_started
        )
    )
    promotion_current = bool(
        promotion
        and (
            candidate_started is None
            or promotion_timestamp is None
            or promotion_timestamp >= candidate_started
        )
    )
    advancement_epoch = "fresh" if advancement_current and promotion_current else "pre_candidate"
    acquisition_status = str(acquisition.get("overall_status") or acquisition.get("status") or "missing").strip().lower()
    acquisition_ready = bool(
        acquisition
        and acquisition_current
        and acquisition_status not in {"waiting_for_source", "missing", "blocked", "failed", "error"}
    )
    replay_status = str(replay_capture.get("overall_status") or replay_capture.get("status") or "missing").strip().lower()
    replay_current, replay_binding_key = _candidate_bound_artifact_state(
        replay_capture,
        candidate=candidate,
    )
    replay_ready = bool(
        replay_capture.get("ok", False)
        and replay_current
        and replay_status in {"ready", "waiting_for_observations", "waiting_for_paper_orders"}
    )
    calibrated_source_ready = bool(calibration and calibration_current)
    fill_producer = _producer(
        "independent_fill_acquisition_with_market_replay",
        ready=bool(calibrated_source_ready and (acquisition_ready or replay_ready)),
        status=(
            acquisition_status
            if acquisition_ready
            else f"market_replay_{replay_status}"
            if replay_ready
            else "candidate_evidence_source_pending"
        ),
        reason=(
            "paper_execution_calibration_candidate_epoch_stale"
            if calibration and not calibration_current
            else "independent_fill_and_market_replay_sources_unavailable"
            if not acquisition_ready and not replay_ready
            else ""
        ),
        cadence_hours=6.0,
        schedule_active=_equity_evidence_schedule_active(now),
        schedule="weekday_equity_extended_hours_event_driven",
        binding_key=f"{calibration_binding_key}:{acquisition_binding_key}:{replay_binding_key}",
    )
    promotion_scope_active = bool(
        advancement_current
        and promotion_current
        and promotion_scope.get("promotion_scope_active", False)
    )
    promotion_producer = _producer(
        "promotion_pipeline",
        ready=promotion_scope_active,
        status=str(advancement.get("overall_status") or "inactive"),
        reason="promotion_scope_inactive" if not promotion_scope_active else "",
        cadence_hours=24.0,
        schedule="runtime_budgeted",
        monotonic_within_candidate=False,
        binding_key=f"{candidate_id}:{advancement_epoch}",
    )
    rollout_binding = _as_dict(rollout.get("candidate_binding"))
    rollout_current, rollout_binding_key = _candidate_bound_artifact_state(
        rollout,
        candidate=candidate,
        cutoff_utc=rollout_binding.get("promotion_window_started_utc"),
    )
    coverage = _as_dict(rollout.get("cohort_source_coverage"))
    canary_coverage = _as_dict(coverage.get("canary"))
    baseline_coverage = _as_dict(coverage.get("baseline"))
    scan_primary = _as_dict(_as_dict(rollout.get("scan")).get("primary"))
    source_files_seen = _safe_int(scan_primary.get("files_seen"), 0) > 0
    equity_schedule_active = _equity_evidence_schedule_active(now)

    def cohort_producer(cohort: str, row: dict[str, Any], samples: float) -> dict[str, Any]:
        source_ready = bool(
            rollout_current
            and (row.get("source_ready", False) or samples > 0 or (not coverage and source_files_seen))
        )
        return _producer(
            f"canary_rollout_{cohort}",
            ready=source_ready,
            status="source_ready" if source_ready else "stale_candidate_epoch" if not rollout_current else "source_missing",
            reason=(
                f"canary_rollout_{cohort}_candidate_epoch_stale"
                if not rollout_current
                else f"{cohort}_cohort_source_missing"
                if not source_ready
                else ""
            ),
            cadence_hours=6.0,
            schedule_active=equity_schedule_active,
            schedule="weekday_equity_extended_hours",
            monotonic_within_candidate=False,
            binding_key=rollout_binding_key,
        )

    canary_samples = _safe_float(rollout.get("canary_samples")) if rollout_current else 0.0
    baseline_samples = _safe_float(rollout.get("baseline_samples")) if rollout_current else 0.0
    canary_producer = cohort_producer("canary", canary_coverage, canary_samples)
    baseline_producer = cohort_producer("baseline", baseline_coverage, baseline_samples)
    canary_daily_producer = {**canary_producer, "expected_cadence_hours": 30.0}
    soak_credit_eligible = bool(candidate.get("credit_eligible", False))
    soak_producer = _producer(
        "accepted_candidate_clock",
        ready=soak_credit_eligible,
        status="accepted" if soak_credit_eligible else "candidate_drift_or_unbound",
        reason="candidate_acceptance_required_before_soak_credit" if not soak_credit_eligible else "",
        cadence_hours=1.0,
        binding_key=f"{candidate_id}:{candidate.get('soak_started_utc', '')}:{int(soak_credit_eligible)}",
    )
    strict_policy = _as_dict(profitability_policy.get("strict_graduation"))
    active_firewall = profitability_firewall if profitability_firewall_current else {}
    active_multiple_testing = multiple_testing if multiple_testing_current else {}
    active_holdout = holdout if holdout_current else {}
    active_benchmark = benchmark if benchmark_current else {}
    active_decay = decay if decay_current else {}
    active_validator = validator if validator_current else {}
    active_rollout_canary = rollout_canary if rollout_current else {}
    allocation = _as_dict(active_firewall.get("allocation_proposal"))
    epoch_contract = _as_dict(active_firewall.get("evidence_epoch_contract"))
    risk_of_ruin = _as_dict(active_validator.get("risk_of_ruin"))
    benchmark_thresholds = _as_dict(benchmark.get("thresholds"))
    edge_contract = _as_dict(active_decay.get("edge_decay_contract"))
    firewall_controls = {
        str(row.get("control_id") or ""): bool(row.get("evidence_ready", False))
        for row in (active_firewall.get("baseline_controls") or [])
        if isinstance(row, dict)
    }
    firewall_controls.update(
        {
            str(row.get("control_id") or ""): bool(row.get("evidence_ready", False))
            for row in (active_firewall.get("controls") or [])
            if isinstance(row, dict)
        }
    )
    return [
        _metric("soak_elapsed_hours", "Frozen-candidate soak time", soak_hours, 720.0, unit="hours", kind="elapsed_time", producer=soak_producer),
        _metric("independent_fills", "Independent paper/replay fills", _safe_float(calibration.get("independent_samples")) if calibration_current else 0.0, 100.0, unit="fills", kind="evidence", producer=fill_producer),
        _metric("post_cost_samples", "Post-cost trade observations", _safe_float(post_cost.get("sample_count")), _safe_float(robust_thresholds.get("minimum_samples"), 30.0), unit="observations", kind="evidence", producer=performance_producer),
        _metric("post_cost_days", "Independent post-cost days", _safe_float(robust.get("unique_day_count")), _safe_float(robust_thresholds.get("minimum_days"), 7.0), unit="days", kind="evidence", producer=daily_performance_producer),
        _metric("post_cost_symbols", "Post-cost symbol breadth", _safe_float(robust.get("unique_symbol_count")), _safe_float(robust_thresholds.get("minimum_symbols"), 5.0), unit="symbols", kind="evidence", producer=performance_producer),
        _metric("post_cost_effective_samples", "Cluster-effective post-cost samples", _safe_float(robust.get("effective_sample_size")), _safe_float(robust_thresholds.get("minimum_effective_samples"), 20.0), unit="effective_samples", kind="evidence", producer=daily_performance_producer),
        _metric("considered_bots", "Independently considered promotion bots", _safe_float(promotion_scope.get("considered_bots")) if promotion_scope_active else 0.0, _safe_float(promotion_scope.get("min_considered_bots"), 4.0), unit="bots", kind="evidence", producer=promotion_producer),
        _metric("promotion_candidates", "Qualified promotion candidates", float(len(promotion_details.get("promotion_candidate_ids") or [])) if promotion_scope_active else 0.0, _safe_float(promotion_scope.get("min_considered_bots"), 4.0), unit="bots", kind="evidence", producer=promotion_producer),
        _metric("canary_samples", "Candidate canary observations", canary_samples, minimum_cohort_samples, unit="observations", kind="evidence", producer=canary_producer),
        _metric("baseline_samples", "Baseline canary observations", baseline_samples, minimum_cohort_samples, unit="observations", kind="evidence", producer=baseline_producer),
        _metric("canary_independent_days", "Canary independent days", _safe_float(active_rollout_canary.get("unique_day_count")), minimum_canary_days, unit="days", kind="evidence", producer=canary_daily_producer),
        _metric("canary_effective_samples", "Canary effective samples", _safe_float(active_rollout_canary.get("effective_sample_size")), minimum_canary_ess, unit="effective_samples", kind="evidence", producer=canary_daily_producer),
        _metric("strict_post_cost_samples", "Strict post-cost observations", _safe_float(post_cost.get("sample_count")), _safe_float(strict_policy.get("minimum_post_cost_samples"), 200.0), unit="observations", kind="evidence", producer=cumulative_performance_producer),
        _metric("strict_independent_days", "Strict independent post-cost days", _safe_float(robust.get("unique_day_count")), _safe_float(strict_policy.get("minimum_independent_days"), 30.0), unit="days", kind="evidence", producer=cumulative_performance_producer),
        _metric("strict_symbol_breadth", "Strict post-cost symbol breadth", _safe_float(robust.get("unique_symbol_count")), _safe_float(strict_policy.get("minimum_symbols"), 10.0), unit="symbols", kind="evidence", producer=cumulative_performance_producer),
        _metric("strict_effective_samples", "Strict cluster-effective samples", _safe_float(robust.get("effective_sample_size")), _safe_float(strict_policy.get("minimum_effective_samples"), 100.0), unit="effective_samples", kind="evidence", producer=cumulative_performance_producer),
        _metric("strict_regime_breadth", "Strict independent regime breadth", _safe_float(robust.get("unique_regime_count")), _safe_float(strict_policy.get("minimum_regimes"), 3.0), unit="regimes", kind="evidence", producer=cumulative_performance_producer),
        _metric("strict_profitable_sleeves", "Strict independently profitable sleeves", _safe_float(allocation.get("qualified_sleeve_count", len(allocation.get("qualified_sleeves") or []))), _safe_float(strict_policy.get("minimum_profitable_sleeves"), 4.0), unit="sleeves", kind="evidence", producer=profitability_producer),
        _metric("positive_conservative_lcb", "Positive conservative post-cost lower bound", float(bool(robust.get("positive_clustered_lower_confidence_bound_95", False))), 1.0, unit="gate", kind="evidence", producer=performance_producer),
        _metric("stressed_expectancy_gate", "Adversarial stressed expectancy", float(firewall_controls.get("06_stressed_post_cost_expectancy", False)), 1.0, unit="gate", kind="evidence", producer=profitability_producer),
        _metric("multiple_testing_gate", "FDR and overfit evidence", float(bool(active_multiple_testing.get("statistical_evidence_ready", False))), 1.0, unit="gate", kind="evidence", producer=multiple_testing_producer),
        _metric("holdout_vault_gate", "Sealed holdout evaluation evidence", float(bool(active_holdout.get("evidence_ready", False))), 1.0, unit="gate", kind="evidence", producer=holdout_producer),
        _metric("benchmark_common_days", "Cash and passive benchmark common days", _safe_float(active_benchmark.get("common_day_count")), _safe_float(benchmark_thresholds.get("minimum_common_days"), 30.0), unit="days", kind="evidence", producer=benchmark_producer),
        _metric("benchmark_hurdle_gate", "Cash and passive benchmark hurdle", float(bool(active_benchmark.get("evidence_ready", False))), 1.0, unit="gate", kind="evidence", producer=benchmark_gate_producer),
        _metric("edge_decay_gate", "Edge-decay evidence and containment", float(bool(edge_contract.get("evidence_ready", False) and edge_contract.get("automatic_demotion_ready", False))), 1.0, unit="gate", kind="evidence", producer=decay_producer),
        _metric("risk_of_ruin_days", "Risk-of-ruin independent days", _safe_float(risk_of_ruin.get("day_count")), _safe_float(_as_dict(risk_of_ruin.get("thresholds")).get("minimum_days"), 30.0), unit="days", kind="evidence", producer=cumulative_validator_producer),
        _metric("risk_of_ruin_gate", "Risk-of-ruin and drawdown stress", float(bool(risk_of_ruin.get("available", False) and risk_of_ruin.get("passes", False))), 1.0, unit="gate", kind="evidence", producer=validator_producer),
        _metric("tail_concentration_gate", "Tail concentration and conservative allocation", float(firewall_controls.get("h09_tail_concentration", False)), 1.0, unit="gate", kind="evidence", producer=profitability_producer),
        _metric("profitability_epoch_coherence", "Atomic profitability evidence epoch", float(bool(epoch_contract.get("ready", False))), 1.0, unit="gate", kind="evidence", producer=profitability_producer),
        _metric("raw_net_pnl", "Raw paper net PnL", _safe_float(target_current.get("net_pnl")), _safe_float(target_thresholds.get("min_net_pnl"), 50000.0), unit="usd", kind="outcome"),
    ]


def _enrich_rates(
    metrics: list[dict[str, Any]],
    *,
    prior_state: dict[str, Any],
    candidate_id: str,
    now: datetime,
    collection_active: bool,
    stall_hours: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prior_candidate = str(prior_state.get("candidate_id") or "")
    prior_timestamp = parse_iso_utc(prior_state.get("timestamp_utc"))
    prior_metrics = prior_state.get("metrics") if isinstance(prior_state.get("metrics"), dict) else {}
    same_candidate = bool(candidate_id and candidate_id == prior_candidate)
    elapsed_hours = max((now - prior_timestamp).total_seconds() / 3600.0, 0.0) if same_candidate and prior_timestamp is not None else 0.0
    next_metric_state: dict[str, Any] = {}
    enriched: list[dict[str, Any]] = []
    for raw in metrics:
        row = dict(raw)
        metric_id = str(row["metric_id"])
        prior = prior_metrics.get(metric_id) if isinstance(prior_metrics.get(metric_id), dict) else {}
        producer = _as_dict(row.get("producer"))
        producer_binding_key = str(producer.get("binding_key") or "")
        prior_binding_key = str(prior.get("producer_binding_key") or "")
        producer_binding_changed = bool(
            same_candidate and producer_binding_key and producer_binding_key != prior_binding_key
        )
        comparable = bool(same_candidate and not producer_binding_changed)
        prior_value = _safe_float(prior.get("value"), row["current"])
        delta = float(row["current"]) - prior_value if comparable else 0.0
        rate = delta / elapsed_hours if comparable and elapsed_hours > 0.0 and delta > 0.0 else None
        producer_ready = bool(producer.get("ready", True))
        schedule_active = bool(producer.get("schedule_active", True))
        prior_producer_ready = bool(prior.get("producer_ready", producer_ready))
        prior_schedule_active = bool(prior.get("schedule_active", schedule_active))
        producer_resumed = bool(comparable and producer_ready and not prior_producer_ready)
        schedule_resumed = bool(comparable and schedule_active and not prior_schedule_active)
        unchanged_since = (
            parse_iso_utc(prior.get("unchanged_since_utc"))
            if comparable and delta <= 0.0 and not producer_resumed and not schedule_resumed
            else now
        )
        if unchanged_since is None:
            unchanged_since = now
        unchanged_hours = max((now - unchanged_since).total_seconds() / 3600.0, 0.0)
        cadence_hours = max(_safe_float(producer.get("expected_cadence_hours"), stall_hours), 0.25)
        stall_threshold = max(float(stall_hours), cadence_hours, 0.25)
        expected_to_move = bool(
            row["kind"] in {"evidence", "elapsed_time"}
            and not row["complete"]
            and producer_ready
            and schedule_active
        )
        regressed = bool(
            comparable
            and row["kind"] == "evidence"
            and bool(producer.get("monotonic_within_candidate", True))
            and delta < 0.0
        )
        stalled = bool(collection_active and expected_to_move and not regressed and unchanged_hours >= stall_threshold)
        eta_hours = float(row["remaining"]) / rate if rate is not None and rate > 0.0 and row["remaining"] > 0.0 else 0.0 if row["complete"] else None
        if row["complete"]:
            accrual_state = "complete"
        elif regressed:
            accrual_state = "counter_regression"
        elif not collection_active and row["kind"] in {"evidence", "elapsed_time"}:
            accrual_state = "collection_paused"
        elif not producer_ready and row["kind"] in {"evidence", "elapsed_time"}:
            accrual_state = "waiting_precondition"
        elif not schedule_active and row["kind"] == "evidence":
            accrual_state = "outside_producer_schedule"
        elif stalled:
            accrual_state = "stalled"
        elif delta > 0.0:
            accrual_state = "advancing"
        else:
            accrual_state = "accumulating"
        row.update(
            {
                "delta_since_previous": round(delta, 6) if comparable else None,
                "rate_per_hour": round(rate, 6) if rate is not None else None,
                "eta_hours": round(eta_hours, 3) if eta_hours is not None else None,
                "eta_available": eta_hours is not None,
                "unchanged_hours": round(unchanged_hours, 3),
                "stalled": stalled,
                "regressed": regressed,
                "expected_to_move_now": expected_to_move,
                "stall_threshold_hours": round(stall_threshold, 3),
                "accrual_state": accrual_state,
                "producer_resumed": producer_resumed,
                "schedule_resumed": schedule_resumed,
                "producer_binding_changed": producer_binding_changed,
            }
        )
        next_metric_state[metric_id] = {
            "value": row["current"],
            "unchanged_since_utc": unchanged_since.isoformat(),
            "producer_ready": producer_ready,
            "schedule_active": schedule_active,
            "producer_binding_key": producer_binding_key,
        }
        enriched.append(row)
    return enriched, next_metric_state


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _load_history(path: Path, limit: int = 512) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[-max(int(limit), 1):]
    except Exception:
        return []
    for line in lines:
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    state_path: Path = DEFAULT_STATE,
    history_path: Path = DEFAULT_HISTORY,
    apply: bool = False,
    stall_hours: float = 6.0,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = now or datetime.now(timezone.utc)
    effective_state = _resolve(project_root, state_path)
    effective_history = _resolve(project_root, history_path)
    prior_state = load_json(effective_state)
    candidate = _candidate(project_root)
    collection_active, collection = _runtime_collection_active(project_root)
    metrics = _raw_metrics(project_root, candidate=candidate, now=current)
    metrics, metric_state = _enrich_rates(
        metrics,
        prior_state=prior_state,
        candidate_id=str(candidate.get("candidate_id") or ""),
        now=current,
        collection_active=collection_active,
        stall_hours=stall_hours,
    )
    stalled_ids = [str(row["metric_id"]) for row in metrics if row.get("stalled")]
    regressed_ids = [str(row["metric_id"]) for row in metrics if row.get("regressed")]
    waiting_ids = [str(row["metric_id"]) for row in metrics if row.get("accrual_state") == "waiting_precondition"]
    pending_ids = [str(row["metric_id"]) for row in metrics if not row.get("complete")]
    status = (
        "blocked"
        if not candidate.get("credit_eligible", False)
        else "regressed"
        if regressed_ids
        else "stalled"
        if stalled_ids
        else "complete"
        if not pending_ids
        else "accumulating"
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": current.isoformat(),
        "overall_status": status,
        "ok": bool(candidate.get("credit_eligible", False) and not stalled_ids and not regressed_ids),
        "candidate_binding": {key: value for key, value in candidate.items() if key != "soak_started"},
        "collection_active": collection_active,
        "collection_evidence": collection,
        "metric_count": len(metrics),
        "complete_metric_count": sum(1 for row in metrics if row.get("complete")),
        "pending_metric_ids": pending_ids,
        "stalled_metric_ids": stalled_ids,
        "regressed_metric_ids": regressed_ids,
        "waiting_precondition_metric_ids": waiting_ids,
        "metrics": metrics,
        "control_contract": {
            "eta_requires_observed_positive_rate": True,
            "candidate_change_resets_rate_history": True,
            "unaccepted_candidate_drift_receives_zero_soak_credit": True,
            "all_profitability_promotion_gates_are_explicitly_tracked": True,
            "producer_candidate_or_window_rebind_resets_metric_history": True,
            "stale_candidate_artifacts_are_quarantined_before_metric_credit": True,
            "each_profitability_metric_tracks_its_actual_producer_epoch": True,
            "stalls_require_active_collection_and_elapsed_window": True,
            "stalls_require_ready_producer_and_active_schedule": True,
            "daily_and_event_driven_evidence_use_source_specific_cadence": True,
            "same_candidate_cumulative_counter_regressions_fail_closed": True,
            "raw_counts_are_not_relabelled_as_independent_evidence": True,
            "market_replay_source_health_is_tracked_without_relabelling_fill_counts": True,
            "live_execution_authority": False,
        },
        "recommended_actions": ordered_unique(
            [
                "investigate the producer or route for stalled evidence metrics" if stalled_ids else "",
                "investigate evidence counter regression before continuing promotion" if regressed_ids else "",
                "satisfy explicit source prerequisites for waiting evidence metrics" if waiting_ids else "",
                "continue candidate-bound collection for pending evidence metrics" if pending_ids else "",
                "do not infer an ETA until a metric has a positive observed accrual rate",
            ]
        ),
    }
    if apply:
        next_state = {
            "schema_version": SCHEMA_VERSION,
            "timestamp_utc": current.isoformat(),
            "candidate_id": str(candidate.get("candidate_id") or ""),
            "generation": _safe_int(candidate.get("generation"), 0),
            "metrics": metric_state,
        }
        write_payload(effective_state, next_state)
        history = _load_history(effective_history)
        history.append(
            {
                "timestamp_utc": current.isoformat(),
                "candidate_id": str(candidate.get("candidate_id") or ""),
                "overall_status": status,
                "values": {str(row["metric_id"]): row["current"] for row in metrics},
                "stalled_metric_ids": stalled_ids,
                "regressed_metric_ids": regressed_ids,
            }
        )
        _atomic_write_jsonl(effective_history, history[-512:])
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Track candidate-bound readiness evidence accrual and stalls.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--out-file", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--history-file", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--stall-hours", type=float, default=6.0)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    project_root = args.project_root.expanduser().resolve()
    payload = build_payload(
        project_root,
        state_path=args.state_file,
        history_path=args.history_file,
        apply=bool(args.apply),
        stall_hours=float(args.stall_hours),
    )
    write_payload(_resolve(project_root, args.out_file), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "readiness_evidence_accrual "
            f"status={payload['overall_status']} complete={payload['complete_metric_count']}/{payload['metric_count']} "
            f"stalled={len(payload['stalled_metric_ids'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
