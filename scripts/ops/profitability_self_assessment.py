#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import load_json, parse_iso_utc, write_payload
    from scripts.ops.production_excellence_control import verify_candidate_event_chain
else:
    from .long_runtime_common import PROJECT_ROOT, load_json, parse_iso_utc, write_payload
    from .production_excellence_control import verify_candidate_event_chain


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "profitability_self_assessment_v1.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "profitability_self_assessment_latest.json"
DEFAULT_MARKDOWN_PATH = PROJECT_ROOT / "exports" / "reports" / "operator" / "profitability_self_assessment_latest.md"
DEFAULT_CANDIDATE_EVENT_PATH = PROJECT_ROOT / "governance" / "evidence" / "production_candidate_events.jsonl"

SOURCE_PATHS = {
    "production_candidate": "governance/runtime/production_candidate_state.json",
    "paper_performance": "governance/health/paper_performance_latest.json",
    "paper_execution_calibration": "governance/health/paper_execution_calibration_latest.json",
    "calibration_control": "governance/health/calibration_abstention_control_latest.json",
    "calibration_overrides": "governance/health/calibration_abstention_overrides_latest.json",
    "profitability_firewall": "governance/health/profitability_evidence_firewall_latest.json",
    "paper_profitability": "governance/health/paper_profitability_control_latest.json",
    "counterfactual_replay": "governance/health/counterfactual_replay_latest.json",
    "sleeve_dashboard": "governance/health/sleeve_profitability_dashboard_latest.json",
    "live_money_readiness": "governance/health/live_money_readiness_contract_latest.json",
    "risk_service": "governance/risk/risk_service_boundary_latest.json",
    "decision_policy": "config/institutional_decision_flow_v1.json",
    "profitability_policy": "config/profitability_evidence_firewall_v1.json",
    "broker_capabilities": "config/broker_capability_contracts_v1.json",
    "collector_capabilities": "governance/health/collector_capability_control_latest.json",
}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _grade(score: float, *, complete: bool = False) -> str:
    if complete and score >= 100.0:
        return "A+"
    if score >= 90.0:
        return "A"
    if score >= 80.0:
        return "B"
    if score >= 70.0:
        return "C"
    if score >= 60.0:
        return "D"
    return "F"


def _timestamp(payload: dict[str, Any], path: Path) -> datetime | None:
    for key in (
        "timestamp_utc",
        "accepted_at_utc",
        "updated_at_utc",
        "created_at_utc",
        "initialized_at_utc",
    ):
        parsed = parse_iso_utc(payload.get(key))
        if parsed is not None:
            return parsed
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return None


def _receipt(path: Path, payload: dict[str, Any], *, max_age_hours: float, now: datetime) -> dict[str, Any]:
    present = bool(payload) and path.is_file()
    timestamp = _timestamp(payload, path) if present else None
    age_seconds = max((now - timestamp).total_seconds(), 0.0) if timestamp else None
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if present else ""
    except OSError:
        digest = ""
    fresh = bool(
        present
        and age_seconds is not None
        and age_seconds <= max(max_age_hours, 0.0) * 3600.0
    )
    return {
        "path": str(path),
        "present": present,
        "fresh": fresh,
        "timestamp_utc": timestamp.isoformat() if timestamp else "",
        "age_seconds": round(age_seconds, 3) if age_seconds is not None else None,
        "maximum_age_hours": round(max_age_hours, 3),
        "sha256": digest,
    }


def _source_candidate_id(name: str, payload: dict[str, Any]) -> str:
    if name == "production_candidate":
        return str(payload.get("candidate_id") or "").strip()
    if name == "paper_performance":
        return str(_as_dict(payload.get("profitability_evidence_window")).get("candidate_id") or "").strip()
    if name == "paper_execution_calibration":
        return str(_as_dict(payload.get("candidate_binding")).get("candidate_id") or "").strip()
    if name == "calibration_overrides":
        binding = _as_dict(payload.get("candidate_binding"))
        return str(binding.get("valid_candidate_id") or binding.get("candidate_id") or "").strip()
    if name == "paper_profitability":
        scaling = _as_dict(payload.get("sleeve_strategy_profitability_scaling_contract"))
        return str(_as_dict(scaling.get("candidate_binding")).get("candidate_id") or "").strip()
    return ""


def _candidate_binding(sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidate = sources["production_candidate"]
    candidate_id = str(candidate.get("candidate_id") or "").strip()
    generation = _safe_int(candidate.get("generation"), 0)
    rows: list[dict[str, Any]] = []
    required = {"paper_performance", "paper_execution_calibration", "paper_profitability"}
    mismatches: list[str] = []
    optional_mismatches: list[str] = []
    missing_required: list[str] = []
    for name in ("paper_performance", "paper_execution_calibration", "calibration_overrides", "paper_profitability"):
        declared = _source_candidate_id(name, sources[name])
        if name in required and not declared:
            missing_required.append(name)
        matches = bool(candidate_id and declared and declared == candidate_id)
        if declared and candidate_id and not matches:
            if name in required:
                mismatches.append(name)
            else:
                optional_mismatches.append(name)
        rows.append(
            {
                "source": name,
                "declared_candidate_id": declared,
                "matches_current_candidate": matches,
                "required": name in required,
            }
        )
    identity_consistent = bool(candidate_id and not mismatches)
    identity_complete = bool(identity_consistent and not missing_required)
    return {
        "candidate_id": candidate_id,
        "generation": generation,
        "accepted_at_utc": str(candidate.get("accepted_at_utc") or ""),
        "candidate_state_receipt_sha256": str(candidate.get("overall_sha256") or ""),
        "live_execution_authority": bool(candidate.get("live_execution_authority", False)),
        "identity_consistent": identity_consistent,
        "identity_complete": identity_complete,
        "mismatch_sources": mismatches,
        "optional_mismatch_sources": optional_mismatches,
        "optional_mismatch_policy": "stale optional inputs are ignored and reported without blocking required-source identity",
        "missing_required_bindings": missing_required,
        "source_bindings": rows,
        "policy": "current-candidate economic evidence is accepted only when every required source declares the same candidate identity",
    }


def _market_sample_counts(calibration: dict[str, Any], broker_capabilities: dict[str, Any]) -> tuple[list[str], dict[str, int]]:
    schwab = _as_dict(_as_dict(broker_capabilities.get("brokers")).get("schwab"))
    paper = _as_dict(schwab.get("paper"))
    required = [str(item or "").strip().upper() for item in _as_list(paper.get("asset_classes")) if str(item or "").strip()]
    raw_counts = _as_dict(calibration.get("by_market_kind"))
    counts: dict[str, int] = {}
    for key, value in raw_counts.items():
        row = _as_dict(value)
        count = _safe_int(
            row.get("independent_samples"),
            _safe_int(row.get("sample_count"), len(value) if isinstance(value, list) else 0),
        )
        counts[str(key).strip().upper()] = count
    return sorted(set(required)), counts


def _lane(
    lane_id: str,
    title: str,
    *,
    implementation_ready: bool,
    evidence_ready: bool,
    current_state: dict[str, Any],
    next_gate: str,
) -> dict[str, Any]:
    return {
        "lane_id": lane_id,
        "title": title,
        "implementation_ready": bool(implementation_ready),
        "implementation_status": "ready" if implementation_ready else "blocked",
        "evidence_ready": bool(evidence_ready),
        "evidence_status": "ready" if evidence_ready else "collecting",
        "current_state": current_state,
        "next_gate": next_gate,
        "paper_only": True,
        "live_execution_authority": False,
    }


def _need(
    blocker: str,
    *,
    exact_file: str,
    exact_shard: str,
    command: list[str],
    expected_impact: str,
    when_to_stop: str,
    classification: str,
    candidate_id: str,
    auto_apply_allowed: bool = False,
    risk_level: str = "low",
) -> dict[str, Any]:
    return {
        "blocker": blocker,
        "exact_file": exact_file,
        "exact_shard": exact_shard,
        "command": command,
        "expected_impact": expected_impact,
        "risk_level": risk_level,
        "when_to_stop": when_to_stop,
        "source": "profitability_self_assessment",
        "classification": classification,
        "candidate_id": candidate_id,
        "auto_apply_allowed": bool(auto_apply_allowed),
        "candidate_semantics_change": False,
        "soak_effect": "preserve cumulative soak history and current candidate window; never relabel historical evidence",
        "live_execution_allowed": False,
    }


def _read_candidate_events(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return []
    rows: list[dict[str, Any]] = []
    for line in lines:
        try:
            row = json.loads(line)
        except (TypeError, ValueError):
            continue
        if not isinstance(row, dict):
            continue
        if parse_iso_utc(row.get("timestamp_utc")) is None:
            continue
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: parse_iso_utc(row.get("timestamp_utc"))
        or datetime.min.replace(tzinfo=timezone.utc),
    )


def _developmental_action(
    action_id: str,
    *,
    owner: str,
    command: list[str],
    trigger: str,
    auto_apply_allowed: bool,
    risk_level: str = "low",
) -> dict[str, Any]:
    return {
        "action_id": action_id,
        "owner": owner,
        "command": command,
        "trigger": trigger,
        "risk_level": risk_level,
        "auto_apply_allowed": bool(auto_apply_allowed),
        "paper_only": True,
        "candidate_semantics_change": False,
        "force_trade_allowed": False,
        "loss_recovery_size_increase_allowed": False,
        "direct_threshold_loosen_allowed": False,
        "live_execution_allowed": False,
        "promotion_authority": False,
    }


def _developmental_soak_learning(
    project_root: Path,
    *,
    policy: dict[str, Any],
    performance: dict[str, Any],
    current_candidate_id: str,
    current_generation: int,
    current_candidate_samples: int,
    minimum_candidate_samples: int,
    missing_market_types: list[str],
    replay_tradeability_ready: bool,
    weak_control_count: int,
    now: datetime,
) -> dict[str, Any]:
    soak_policy = _as_dict(policy.get("soak_contract"))
    enabled = bool(soak_policy.get("developmental_change_learning_enabled", False))
    event_relative = str(
        soak_policy.get("candidate_event_log_path")
        or "governance/evidence/production_candidate_events.jsonl"
    )
    event_path = Path(event_relative)
    if not event_path.is_absolute():
        event_path = project_root / event_path
    chain = verify_candidate_event_chain(event_path)
    chain_valid = bool(chain.get("ok", False) and _safe_int(chain.get("event_count"), 0) > 0)
    events = _read_candidate_events(event_path) if chain_valid else []
    generation_events = [
        row
        for row in events
        if str(row.get("event_type") or "")
        in {"candidate_change_accepted", "candidate_chain_recovery_anchor"}
    ]
    event_by_candidate = {
        str(row.get("candidate_id") or "").strip(): (index, row)
        for index, row in enumerate(generation_events)
        if str(row.get("candidate_id") or "").strip()
    }
    flow_contract = _as_dict(performance.get("developmental_generation_flows"))
    raw_flows = [
        row
        for row in _as_list(flow_contract.get("generation_flows"))
        if isinstance(row, dict)
    ]
    flow_by_candidate = {
        str(row.get("candidate_id") or "").strip(): row
        for row in raw_flows
        if str(row.get("candidate_id") or "").strip()
    }
    minimum_days = max(
        _safe_int(soak_policy.get("minimum_developmental_observed_days"), 2),
        1,
    )
    minimum_samples = max(
        _safe_int(
            soak_policy.get("minimum_developmental_post_cost_samples"),
            minimum_candidate_samples,
        ),
        1,
    )

    rows: list[dict[str, Any]] = []
    attributable_count = 0
    mature_count = 0
    observed_positive_count = 0
    observed_negative_count = 0
    observed_flat_count = 0
    for index, event in enumerate(generation_events):
        candidate_id = str(event.get("candidate_id") or "").strip()
        generation = _safe_int(event.get("generation"), 0)
        accepted_at = parse_iso_utc(event.get("timestamp_utc"))
        next_accepted_at = (
            parse_iso_utc(generation_events[index + 1].get("timestamp_utc"))
            if index + 1 < len(generation_events)
            else None
        )
        flow = _as_dict(flow_by_candidate.get(candidate_id))
        sample_count = _safe_int(flow.get("sample_count"), 0)
        observed_days = _safe_int(flow.get("observed_days"), 0)
        pnl_delta = _safe_float(flow.get("post_cost_pnl_delta_total"), 0.0)
        first_observation = parse_iso_utc(flow.get("first_observation_utc"))
        last_observation = parse_iso_utc(flow.get("last_observation_utc"))
        generation_matches = bool(
            generation > 0
            and _safe_int(flow.get("candidate_generation"), 0) == generation
            and bool(flow.get("candidate_generation_consistent", False))
        )
        temporal_binding_valid = bool(
            accepted_at is not None
            and first_observation is not None
            and last_observation is not None
            and first_observation >= accepted_at
            and last_observation >= first_observation
            and (next_accepted_at is None or last_observation < next_accepted_at)
        )
        attributable = bool(
            enabled
            and chain_valid
            and sample_count > 0
            and generation_matches
            and temporal_binding_valid
            and flow.get("developmental_attribution_eligible", False)
        )
        mature = bool(
            attributable
            and sample_count >= minimum_samples
            and observed_days >= minimum_days
        )
        if not flow:
            status = "collecting_no_identity_bound_outcomes"
        elif not generation_matches:
            status = "rejected_generation_identity_conflict"
        elif not temporal_binding_valid:
            status = "rejected_outside_accepted_generation_window"
        elif not mature:
            status = "collecting_developmental_evidence"
        elif pnl_delta > 0.0:
            status = "observed_positive_post_cost_delta"
        elif pnl_delta < 0.0:
            status = "observed_negative_post_cost_delta"
        else:
            status = "observed_flat_post_cost_delta"
        if attributable:
            attributable_count += 1
        if mature:
            mature_count += 1
            if pnl_delta > 0.0:
                observed_positive_count += 1
            elif pnl_delta < 0.0:
                observed_negative_count += 1
            else:
                observed_flat_count += 1
        rows.append(
            {
                "candidate_id": candidate_id,
                "generation": generation,
                "current_candidate": bool(candidate_id == current_candidate_id),
                "event_type": str(event.get("event_type") or ""),
                "accepted_at_utc": accepted_at.isoformat() if accepted_at else "",
                "ended_at_utc": next_accepted_at.isoformat() if next_accepted_at else now.isoformat(),
                "changed_scopes": [
                    str(item) for item in _as_list(event.get("changed_scopes"))
                ],
                "change_reason": str(event.get("change_reason") or ""),
                "sample_count": sample_count,
                "observed_days": observed_days,
                "post_cost_pnl_delta": round(pnl_delta, 6),
                "generation_identity_matches": generation_matches,
                "temporal_binding_valid": temporal_binding_valid,
                "developmental_attribution_eligible": attributable,
                "developmental_evidence_mature": mature,
                "developmental_status": status,
                "association_is_causal_proof": False,
                "counts_toward_current_candidate_economic_grade": False,
                "counts_toward_clean_720_hour_promotion_window": False,
                "live_promotion_authority": False,
            }
        )

    unmatched_flows: list[dict[str, Any]] = []
    for candidate_id, flow in sorted(flow_by_candidate.items()):
        if candidate_id in event_by_candidate:
            continue
        unmatched_flows.append(
            {
                "candidate_id": candidate_id,
                "candidate_generation": _safe_int(flow.get("candidate_generation"), 0),
                "sample_count": _safe_int(flow.get("sample_count"), 0),
                "reason": "candidate_missing_from_verified_event_chain",
                "developmental_attribution_eligible": False,
                "live_promotion_authority": False,
            }
        )

    allowed_actions = {
        str(item)
        for item in _as_list(soak_policy.get("bounded_paper_actions"))
        if str(item)
    }
    actions: list[dict[str, Any]] = []

    def add_action(action: dict[str, Any]) -> None:
        if action["action_id"] in allowed_actions:
            actions.append(action)

    if current_candidate_samples < minimum_candidate_samples:
        add_action(
            _developmental_action(
                "collect_current_candidate_post_cost_outcomes",
                owner="paper_performance_refresh",
                command=["./scripts/ops/opsctl.sh", "paper-performance", "--week-days", "7", "--json"],
                trigger="current candidate has insufficient identity-bound post-cost outcomes",
                auto_apply_allowed=True,
            )
        )
    if not replay_tradeability_ready or current_candidate_samples < minimum_candidate_samples or observed_negative_count:
        add_action(
            _developmental_action(
                "refresh_candidate_counterfactual_replay",
                owner="counterfactual_replay",
                command=["./scripts/ops/opsctl.sh", "counterfactual-replay", "--json"],
                trigger="candidate evidence is incomplete or a mature accepted generation has a negative observed delta",
                auto_apply_allowed=True,
            )
        )
    if missing_market_types:
        add_action(
            _developmental_action(
                "acquire_candidate_independent_fills",
                owner="independent_fill_evidence_acquisition",
                command=["./scripts/ops/opsctl.sh", "independent-fill-acquisition", "--apply", "--json"],
                trigger=f"independent fill evidence is incomplete for {','.join(missing_market_types)}",
                auto_apply_allowed=True,
                risk_level="medium",
            )
        )
    if observed_negative_count or weak_control_count > 0:
        add_action(
            _developmental_action(
                "maintain_candidate_bound_weak_sleeve_containment",
                owner="paper_profitability_control",
                command=["./scripts/ops/opsctl.sh", "paper-profitability-control", "--apply", "--json"],
                trigger="negative developmental outcomes or weak strategy controls remain",
                auto_apply_allowed=True,
                risk_level="medium",
            )
        )
    if observed_negative_count:
        add_action(
            _developmental_action(
                "prioritize_loss_and_missed_opportunity_labels",
                owner="training_data_intake_labeling",
                command=["./scripts/ops/opsctl.sh", "training-data-intake", "--apply", "--focus-limit", "160", "--json"],
                trigger="mature accepted generations contain negative post-cost developmental outcomes",
                auto_apply_allowed=True,
                risk_level="medium",
            )
        )
    if enabled and not chain_valid:
        actions.insert(
            0,
            _developmental_action(
                "repair_candidate_event_chain",
                owner="production_excellence_control",
                command=["./scripts/ops/opsctl.sh", "production-excellence", "--json"],
                trigger="candidate generation event chain is missing or invalid",
                auto_apply_allowed=False,
                risk_level="high",
            ),
        )

    current_row = next(
        (row for row in reversed(rows) if row.get("candidate_id") == current_candidate_id),
        {},
    )
    return {
        "enabled": enabled,
        "status": (
            "disabled"
            if not enabled
            else "blocked"
            if not chain_valid
            else "ready"
            if attributable_count > 0
            else "collecting"
        ),
        "candidate_event_chain": {
            "valid": chain_valid,
            "event_count": _safe_int(chain.get("event_count"), 0),
            "chain_head": str(chain.get("chain_head") or ""),
            "errors": [str(item) for item in _as_list(chain.get("errors"))],
            "path": str(event_path),
        },
        "current_candidate_id": current_candidate_id,
        "current_generation": current_generation,
        "accepted_generation_count": len(generation_events),
        "attributable_generation_count": attributable_count,
        "mature_developmental_generation_count": mature_count,
        "observed_positive_delta_generation_count": observed_positive_count,
        "observed_negative_delta_generation_count": observed_negative_count,
        "observed_flat_delta_generation_count": observed_flat_count,
        "unbound_schema_v2_sample_count": _safe_int(
            flow_contract.get("unbound_schema_v2_sample_count"), 0
        ),
        "metadata_conflict_count": _safe_int(
            flow_contract.get("metadata_conflict_count"), 0
        ),
        "unmatched_generation_flows": unmatched_flows,
        "generation_rows": rows,
        "current_generation_row": current_row,
        "bounded_paper_action_plan": actions,
        "automatic_action_count": sum(
            1 for row in actions if row.get("auto_apply_allowed", False)
        ),
        "policy": {
            "accepted_generations_feed_developmental_learning": enabled,
            "identity_and_time_binding_required": True,
            "association_is_not_causation": True,
            "mixed_or_unbound_history_is_diagnostic_only": True,
            "historical_generations_grade_current_candidate": False,
            "historical_generations_earn_clean_720_hour_credit": False,
            "clean_720_hour_live_promotion_gate_unchanged": True,
            "current_candidate_economic_firewall_unchanged": True,
            "profitability_guaranteed": False,
            "force_trades": False,
            "loss_recovery_size_increase_allowed": False,
            "direct_threshold_loosen_allowed": False,
            "live_execution_allowed": False,
        },
    }


def build_payload(project_root: Path = PROJECT_ROOT, *, config_path: Path | None = None) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    policy_path = config_path or project_root / "config" / DEFAULT_CONFIG_PATH.name
    policy = load_json(policy_path)
    freshness = _as_dict(policy.get("source_freshness_hours"))
    sources: dict[str, dict[str, Any]] = {}
    receipts: dict[str, dict[str, Any]] = {}
    for name, relative in SOURCE_PATHS.items():
        path = project_root / relative
        payload = load_json(path)
        sources[name] = payload
        receipts[name] = _receipt(
            path,
            payload,
            max_age_hours=_safe_float(freshness.get(name), 24.0),
            now=now,
        )

    binding = _candidate_binding(sources)
    candidate_id = str(binding.get("candidate_id") or "")
    performance = sources["paper_performance"]
    expectancy = _as_dict(performance.get("post_cost_expectancy"))
    accounting = _as_dict(performance.get("accounting_views"))
    candidate_flow = _as_dict(accounting.get("candidate_forward_flow"))
    active_book = _as_dict(accounting.get("active_book_snapshot"))
    candidate_samples = _safe_int(expectancy.get("sample_count"), _safe_int(candidate_flow.get("sample_count"), 0))
    minimum_candidate_samples = _safe_int(expectancy.get("minimum_samples"), 30)
    candidate_pnl = _safe_float(candidate_flow.get("post_cost_pnl_delta_total"), 0.0)

    calibration = sources["paper_execution_calibration"]
    calibration_control = sources["calibration_control"]
    calibration_overrides = sources["calibration_overrides"]
    calibration_recommendations = _as_list(calibration_control.get("recommendations"))
    family_recommendations = _as_list(calibration_control.get("family_recommendations"))
    override_binding = _as_dict(calibration_overrides.get("candidate_binding"))
    override_candidate = str(override_binding.get("valid_candidate_id") or override_binding.get("candidate_id") or "")
    override_binding_valid = bool(
        _safe_int(calibration_overrides.get("schema_version"), 0) >= 2
        and candidate_id
        and override_candidate == candidate_id
        and bool(override_binding.get("valid_until_candidate_changes", False))
    )
    direct_loosen_present = any(
        isinstance(row, dict)
        and (
            str(row.get("mode") or "").strip().lower() == "loosen"
            or _safe_float(row.get("acted_prob_threshold_uplift"), 0.0) < 0.0
        )
        for group in (
            _as_dict(calibration_overrides.get("bot_overrides")).values(),
            _as_dict(calibration_overrides.get("family_overrides")).values(),
        )
        for row in group
    )

    decision = sources["decision_policy"]
    stages = {
        str(row.get("stage_id") or "")
        for row in _as_list(decision.get("stages"))
        if isinstance(row, dict)
    }
    profile_map = _as_dict(decision.get("profile_policy_map"))
    policy_families = _as_dict(decision.get("sleeve_policy_families"))
    counterfactual = sources["counterfactual_replay"]
    replay_candidates = _as_list(counterfactual.get("top_candidates"))
    replay_tradeability_ready = bool(
        replay_candidates
        and all(
            isinstance(row, dict) and "tradeability_floor" in row and "max_conflict_norm" in row
            for row in replay_candidates
        )
    )

    profit_policy = sources["profitability_policy"]
    configured_labels = {
        str(item or "") for item in _as_list(profit_policy.get("counterfactual_labels"))
    }
    required_exit_labels = {
        str(item or "")
        for item in _as_list(_as_dict(policy.get("exit_learning")).get("required_labels"))
    }
    paper_profitability = sources["paper_profitability"]
    harvest_regret = _as_dict(paper_profitability.get("profit_harvest_regret_replay_contract"))
    scaling = _as_dict(paper_profitability.get("sleeve_strategy_profitability_scaling_contract"))
    sizing_policy = _as_dict(policy.get("sizing"))
    current_entry_cap = _safe_float(scaling.get("global_entry_size_cap_norm"), -1.0)
    current_max_scale = _safe_float(scaling.get("maximum_above_baseline_entry_size_multiplier_norm"), -1.0)
    expected_entry_cap = _safe_float(sizing_policy.get("paper_entry_cap_norm"), 0.25)
    expected_max_scale = _safe_float(sizing_policy.get("maximum_evidence_validated_scale_norm"), 1.10)

    required_market_types, market_counts = _market_sample_counts(calibration, sources["broker_capabilities"])
    min_fills_per_market = _safe_int(
        _as_dict(policy.get("execution_realism")).get("minimum_independent_fills_per_market_type"),
        30,
    )
    market_requirements = {
        market: {
            "independent_samples": _safe_int(market_counts.get(market), 0),
            "minimum_samples": min_fills_per_market,
            "ready": _safe_int(market_counts.get(market), 0) >= min_fills_per_market,
        }
        for market in required_market_types
    }
    market_evidence_ready = bool(market_requirements) and all(
        bool(row.get("ready", False)) for row in market_requirements.values()
    )

    firewall = sources["profitability_firewall"]
    allocation = _as_dict(firewall.get("allocation_proposal"))
    minimum_profitable_sleeves = _safe_int(
        _as_dict(policy.get("portfolio_allocation")).get("minimum_independently_profitable_sleeves"),
        4,
    )
    qualified_sleeves = _safe_int(allocation.get("qualified_sleeve_count"), 0)
    income_policy = _as_dict(policy.get("income_sleeves"))
    income_uplift = _safe_float(income_policy.get("paper_confidence_threshold_uplift"), 0.08)
    income_abstention = _safe_float(income_policy.get("paper_abstention_budget"), 0.88)
    family_rows = {
        str(row.get("family") or "").strip().lower(): row
        for row in family_recommendations
        if isinstance(row, dict)
    }
    override_families = _as_dict(calibration_overrides.get("family_overrides"))
    income_controls_ready = all(
        str(profile_map.get(family) or "") == "long_horizon_income"
        and math.isclose(
            _safe_float(_as_dict(override_families.get(family)).get("acted_prob_threshold_uplift"), -1.0),
            income_uplift,
            abs_tol=1e-9,
        )
        and math.isclose(
            _safe_float(_as_dict(override_families.get(family)).get("recommended_abstention_budget"), -1.0),
            income_abstention,
            abs_tol=1e-9,
        )
        for family in ("bond", "dividend")
    )

    lanes = [
        _lane(
            "01_confidence_and_abstention",
            "Confidence thresholds and abstention",
            implementation_ready=bool(calibration_control) and not direct_loosen_present,
            evidence_ready=bool(override_binding_valid and not calibration_recommendations),
            current_state={
                "overacting_count": _safe_int(calibration_control.get("overacting_count"), 0),
                "underacting_count": _safe_int(calibration_control.get("underacting_count"), 0),
                "recommendation_count": len(calibration_recommendations),
                "candidate_bound_overrides": override_binding_valid,
                "direct_loosen_present": direct_loosen_present,
            },
            next_gate="apply paper-only tightening, then require candidate-bound replay before any future loosening",
        ),
        _lane(
            "02_sleeve_and_regime_thresholds",
            "Sleeve and regime-specific thresholds",
            implementation_ready=bool(profile_map and policy_families and "04_consensus_and_regime" in stages),
            evidence_ready=bool(candidate_samples >= minimum_candidate_samples and replay_tradeability_ready),
            current_state={
                "profile_policy_count": len(profile_map),
                "policy_family_count": len(policy_families),
                "regime_stage_present": "04_consensus_and_regime" in stages,
                "regime_override_count": len(_as_dict(calibration_overrides.get("regime_overrides"))),
                "direct_loosen_allowed": False,
            },
            next_gate="collect candidate outcomes by resolved sleeve family and regime before changing acceptance thresholds",
        ),
        _lane(
            "03_bond_and_dividend_acceptance",
            "Bond and dividend acceptance",
            implementation_ready=income_controls_ready,
            evidence_ready=bool(candidate_samples >= minimum_candidate_samples and replay_tradeability_ready),
            current_state={
                "policy_family": "long_horizon_income",
                "confidence_threshold_uplift": income_uplift,
                "abstention_budget": income_abstention,
                "candidate_bound_overrides": override_binding_valid,
                "diagnostic_recommendations": sorted(set(family_rows) & {"bond", "dividend"}),
            },
            next_gate="retain +0.08 tightening and 88% abstention until candidate-bound replay supports a safer setting",
        ),
        _lane(
            "04_tradeability_and_conflict",
            "Tradeability and conflict limits",
            implementation_ready=bool(
                {"06_execution_feasibility", "07_portfolio_fit"} <= stages
                and _as_dict(profit_policy.get("entry_quality")).get("unknown_evidence_fails_closed", False)
            ),
            evidence_ready=bool(replay_tradeability_ready and candidate_samples >= minimum_candidate_samples),
            current_state={
                "execution_stage_present": "06_execution_feasibility" in stages,
                "portfolio_conflict_stage_present": "07_portfolio_fit" in stages,
                "replay_tradeability_and_conflict_fields_present": replay_tradeability_ready,
                "unknown_evidence_fails_closed": bool(
                    _as_dict(profit_policy.get("entry_quality")).get("unknown_evidence_fails_closed", False)
                ),
            },
            next_gate="validate tradeability floors and conflict ceilings on current-candidate post-cost replay",
        ),
        _lane(
            "05_exit_timing_learning",
            "MAE, MFE, continuation, harvest-regret, and regime-aware exits",
            implementation_ready=bool(required_exit_labels <= configured_labels and harvest_regret),
            evidence_ready=bool(candidate_samples >= minimum_candidate_samples and replay_tradeability_ready),
            current_state={
                "required_labels": sorted(required_exit_labels),
                "configured_labels": sorted(configured_labels & required_exit_labels),
                "harvest_regret_control_present": bool(harvest_regret),
                "exit_paths_open": bool(scaling.get("keep_sells_and_reduce_only_paths_open", False)),
            },
            next_gate="join current-candidate excursions and exit outcomes before changing exit timing",
        ),
        _lane(
            "06_execution_realism",
            "Independent execution realism by market type",
            implementation_ready=bool(_as_dict(calibration.get("candidate_binding")).get("required", False)),
            evidence_ready=market_evidence_ready,
            current_state={
                "independent_samples_total": _safe_int(calibration.get("independent_samples"), 0),
                "minimum_samples_per_market_type": min_fills_per_market,
                "market_types": market_requirements,
                "model_defaults_count_as_evidence": False,
            },
            next_gate="collect at least 30 independent candidate-bound fills in every supported Schwab paper market type",
        ),
        _lane(
            "07_evidence_gated_sizing",
            "Evidence-gated position sizing",
            implementation_ready=bool(
                math.isclose(current_entry_cap, expected_entry_cap, abs_tol=1e-9)
                and math.isclose(current_max_scale, expected_max_scale, abs_tol=1e-9)
                and bool(scaling.get("entry_only", False))
                and bool(scaling.get("keep_sells_and_reduce_only_paths_open", False))
            ),
            evidence_ready=bool(scaling.get("scale_up_ready", False)),
            current_state={
                "paper_entry_cap_norm": current_entry_cap,
                "maximum_evidence_validated_scale_norm": current_max_scale,
                "scale_up_ready": bool(scaling.get("scale_up_ready", False)),
                "above_baseline_ready_count": _safe_int(scaling.get("above_baseline_ready_count"), 0),
                "loss_recovery_size_increase_allowed": False,
            },
            next_gate="retain the 0.25 entry cap; permit at most 1.10 only after robust candidate-bound evidence",
        ),
        _lane(
            "08_portfolio_allocation",
            "Evidence-gated portfolio allocation",
            implementation_ready=bool(
                _safe_int(_as_dict(allocation.get("thresholds")).get("minimum_profitable_sleeves"), 0)
                >= minimum_profitable_sleeves
                and allocation.get("automatic_allocation_allowed") is False
            ),
            evidence_ready=bool(allocation.get("ready", False)),
            current_state={
                "qualified_sleeves": _as_list(allocation.get("qualified_sleeves")),
                "qualified_sleeve_count": qualified_sleeves,
                "minimum_profitable_sleeves": minimum_profitable_sleeves,
                "suggested_cash_weight": _safe_float(allocation.get("suggested_cash_weight"), 1.0),
                "automatic_allocation_allowed": False,
            },
            next_gate="keep 100% unallocated until at least four independently profitable low-correlation sleeves qualify",
        ),
    ]

    implementation_ready_count = sum(1 for row in lanes if row["implementation_ready"])
    evidence_ready_count = sum(1 for row in lanes if row["evidence_ready"])
    implementation_score = round(100.0 * implementation_ready_count / max(len(lanes), 1), 3)
    economic_score = _safe_float(firewall.get("economic_evidence_score"), 0.0)
    economic_grade = str(firewall.get("economic_evidence_grade") or _grade(economic_score)).upper()
    economic_ready = bool(firewall.get("economic_evidence_ready", False))
    collector_capabilities = sources["collector_capabilities"]
    economic_context = _as_dict(collector_capabilities.get("economic_context_contract"))
    economic_context_policy = _as_dict(economic_context.get("policy"))
    economic_family_count = _safe_int(economic_context.get("family_count"), 0)
    economic_configured_families = _safe_int(economic_context.get("configured_family_count"), 0)
    economic_ready_families = _safe_int(economic_context.get("ready_family_count"), 0)
    economic_runtime_routes = _safe_int(economic_context.get("runtime_route_count"), 0)
    economic_ready_runtime_routes = _safe_int(economic_context.get("runtime_ready_route_count"), 0)
    economic_source_count = _safe_int(economic_context.get("selected_source_count"), 0)
    minimum_economic_sources = max(
        _safe_int(economic_context_policy.get("minimum_distinct_selected_sources"), 2),
        2,
    )
    economic_context_checks = {
        "artifact_fresh": bool(receipts["collector_capabilities"].get("fresh", False)),
        "contract_present": bool(str(economic_context_policy.get("contract_id") or "")),
        "all_families_configured": bool(
            economic_family_count > 0 and economic_configured_families == economic_family_count
        ),
        "all_families_ready": bool(
            economic_family_count > 0 and economic_ready_families == economic_family_count
        ),
        "all_runtime_routes_ready": bool(
            economic_runtime_routes > 0 and economic_ready_runtime_routes == economic_runtime_routes
        ),
        "source_pool_ready": economic_source_count >= minimum_economic_sources,
        "receipt_present": bool(str(economic_context.get("contract_receipt_sha256") or "")),
        "authority_safe": bool(
            economic_context.get("context_changes_strategy_signal") is False
            and economic_context.get("paper_execution_authority") is False
            and economic_context.get("live_execution_authority") is False
            and economic_context.get("automatic_promotion_authority") is False
            and economic_context.get("economic_profitability_grade_authority") is False
        ),
    }
    economic_context_ready = all(economic_context_checks.values())
    economic_context_score = round(
        100.0 * sum(1 for ready in economic_context_checks.values() if ready) / len(economic_context_checks),
        3,
    )
    tier_counts = _as_dict(scaling.get("tier_counts"))
    weak_control_count = max(
        _safe_int(scaling.get("blocked_control_count"), 0),
        _safe_int(tier_counts.get("quarantine"), 0),
    )
    missing_market_types = [
        market for market, row in market_requirements.items() if not row["ready"]
    ]
    developmental_learning = _developmental_soak_learning(
        project_root,
        policy=policy,
        performance=performance,
        current_candidate_id=candidate_id,
        current_generation=_safe_int(binding.get("generation"), 0),
        current_candidate_samples=candidate_samples,
        minimum_candidate_samples=minimum_candidate_samples,
        missing_market_types=missing_market_types,
        replay_tradeability_ready=replay_tradeability_ready,
        weak_control_count=weak_control_count,
        now=now,
    )
    developmental_action_ids = {
        str(row.get("action_id") or "")
        for row in _as_list(developmental_learning.get("bounded_paper_action_plan"))
        if isinstance(row, dict)
    }

    needs: list[dict[str, Any]] = []
    if not binding.get("identity_consistent", False) or not binding.get("identity_complete", False):
        needs.append(
            _need(
                "candidate_identity_binding_incomplete",
                exact_file="governance/health/profitability_self_assessment_latest.json",
                exact_shard="candidate_binding",
                command=["./scripts/ops/opsctl.sh", "profitability-self-assessment", "--json"],
                expected_impact="Prevents cross-candidate outcomes, fills, or overrides from grading the current candidate.",
                when_to_stop="all required candidate-bearing sources declare the current production candidate with zero mismatches",
                classification="repair",
                candidate_id=candidate_id,
            )
        )
    if calibration_recommendations and not override_binding_valid:
        needs.append(
            _need(
                "candidate_bound_abstention_tightening_pending",
                exact_file="governance/health/calibration_abstention_overrides_latest.json",
                exact_shard="candidate_binding",
                command=["./scripts/ops/opsctl.sh", "calibration-control", "--apply", "--json"],
                expected_impact="Applies only non-negative paper threshold uplifts and retires unsafe direct-loosen overrides for this candidate.",
                when_to_stop="override schema is candidate-bound, direct-loosen overrides are absent, and the candidate identity matches",
                classification="tune",
                candidate_id=candidate_id,
                auto_apply_allowed=True,
            )
        )
    if candidate_samples < minimum_candidate_samples:
        needs.append(
            _need(
                "candidate_post_cost_observations_collecting",
                exact_file="governance/health/paper_performance_latest.json",
                exact_shard="post_cost_expectancy",
                command=["./scripts/ops/opsctl.sh", "paper-performance", "--week-days", "7", "--json"],
                expected_impact="Measures current-candidate post-cost expectancy without mixing the historical paper ledger into the grade.",
                when_to_stop=f"at least {minimum_candidate_samples} current-candidate schema-v2 post-cost observations exist",
                classification="collect",
                candidate_id=candidate_id,
            )
        )
    if missing_market_types:
        needs.append(
            _need(
                "independent_fill_evidence_by_market_collecting",
                exact_file="governance/health/paper_execution_calibration_latest.json",
                exact_shard="by_market_kind",
                command=["./scripts/ops/opsctl.sh", "independent-fill-acquisition", "--apply", "--json"],
                expected_impact="Replaces model-default fill assumptions with candidate-bound observed or replay fill evidence by market type.",
                when_to_stop=f"each required market type has at least {min_fills_per_market} independent fills: {','.join(missing_market_types)}",
                classification="collect",
                candidate_id=candidate_id,
            )
        )
    if not replay_tradeability_ready or candidate_samples < minimum_candidate_samples:
        needs.append(
            _need(
                "candidate_bound_threshold_and_exit_replay_collecting",
                exact_file="governance/health/counterfactual_replay_latest.json",
                exact_shard="top_candidates",
                command=["./scripts/ops/opsctl.sh", "counterfactual-replay", "--json"],
                expected_impact="Tests threshold, tradeability, conflict, and exit choices before any loosening or scale change.",
                when_to_stop="replay is bound to current-candidate outcomes and supports positive post-cost expectancy",
                classification="collect",
                candidate_id=candidate_id,
            )
        )
    if not economic_ready:
        needs.append(
            _need(
                "economic_profitability_evidence_collecting",
                exact_file="governance/health/profitability_evidence_firewall_latest.json",
                exact_shard="economic_evidence_blockers",
                command=["./scripts/ops/opsctl.sh", "profitability-evidence-firewall", "--json"],
                expected_impact="Keeps implementation quality separate from externally testable post-cost profitability proof.",
                when_to_stop="economic evidence is ready with a positive conservative lower confidence bound and all firewall controls pass",
                classification="collect",
                candidate_id=candidate_id,
            )
        )
    if not economic_context_ready:
        needs.append(
            _need(
                "sleeve_economic_context_incomplete",
                exact_file="governance/health/collector_capability_control_latest.json",
                exact_shard="economic_context_contract",
                command=["./scripts/ops/opsctl.sh", "collector-capability-control", "--json"],
                expected_impact="Restores fresh, receipt-bound economic context for every decision family without changing trade authority or profitability evidence.",
                when_to_stop="all decision families and runtime routes have complete economic context with at least two distinct sources and a safe authority contract",
                classification="repair",
                candidate_id=candidate_id,
            )
        )
    if qualified_sleeves < minimum_profitable_sleeves:
        needs.append(
            _need(
                "independently_profitable_sleeve_breadth_collecting",
                exact_file="governance/health/profitability_evidence_firewall_latest.json",
                exact_shard="allocation_proposal",
                command=["./scripts/ops/opsctl.sh", "sleeve-profitability-dashboard", "--json"],
                expected_impact="Prevents a single sleeve or correlated cluster from receiving portfolio allocation prematurely.",
                when_to_stop=f"at least {minimum_profitable_sleeves} independently profitable, sufficiently observed, low-correlation sleeves qualify",
                classification="collect",
                candidate_id=candidate_id,
            )
        )
    if (
        developmental_learning.get("enabled", False)
        and not _as_dict(developmental_learning.get("candidate_event_chain")).get("valid", False)
    ):
        needs.append(
            _need(
                "developmental_candidate_event_chain_unavailable",
                exact_file="governance/evidence/production_candidate_events.jsonl",
                exact_shard="candidate_event_chain",
                command=["./scripts/ops/opsctl.sh", "production-excellence", "--json"],
                expected_impact="Restores verified candidate-generation boundaries before historical outcomes are associated with accepted soak changes.",
                when_to_stop="the candidate event chain is valid and its identity-bound generation windows can be joined without conflicts",
                classification="repair",
                candidate_id=candidate_id,
                risk_level="high",
            )
        )
    if "maintain_candidate_bound_weak_sleeve_containment" in developmental_action_ids:
        needs.append(
            _need(
                "developmental_weak_sleeve_containment_active",
                exact_file="governance/health/paper_profitability_control_latest.json",
                exact_shard="sleeve_strategy_profitability_scaling_contract",
                command=["./scripts/ops/opsctl.sh", "paper-profitability-control", "--apply", "--json"],
                expected_impact="Keeps weak candidate-bound sleeve and strategy entries quarantined or deweighted while preserving full-size exits and continued collection.",
                when_to_stop="negative developmental cohorts clear their configured evidence gates and weak controls no longer require quarantine or probation",
                classification="tune",
                candidate_id=candidate_id,
                auto_apply_allowed=True,
                risk_level="medium",
            )
        )
    if "prioritize_loss_and_missed_opportunity_labels" in developmental_action_ids:
        needs.append(
            _need(
                "developmental_loss_label_priority_active",
                exact_file="governance/health/training_data_intake_expansion_latest.json",
                exact_shard="paper_loss_hard_negative_ingress",
                command=["./scripts/ops/opsctl.sh", "training-data-intake", "--apply", "--focus-limit", "160", "--json"],
                expected_impact="Routes mature negative accepted-generation outcomes into hard-negative and missed-opportunity labeling without changing current trade authority.",
                when_to_stop="the prioritized negative cohorts have point-in-time labels and a candidate-bound replay receipt",
                classification="train",
                candidate_id=candidate_id,
                auto_apply_allowed=True,
                risk_level="medium",
            )
        )

    if not candidate_id:
        statement = "The system cannot attribute profitability because no accepted production candidate is present."
    elif not binding.get("identity_consistent", False):
        statement = "The system found a candidate identity conflict and refuses to combine the affected evidence."
    elif candidate_samples <= 0:
        statement = (
            f"Candidate {candidate_id} is guarded for paper collection, but it has no current-candidate schema-v2 "
            "post-cost outcomes yet, so live-grade profitability cannot be estimated. Earlier accepted generations "
            "remain available for verified developmental attribution and bounded paper-only remediation."
        )
    elif not economic_ready:
        statement = (
            f"Candidate {candidate_id} has {candidate_samples} post-cost outcomes, but the evidence is not yet "
            "sufficient to claim positive expectancy or allocate capital."
        )
    else:
        statement = (
            f"Candidate {candidate_id} has passed the configured economic evidence firewall; live execution remains "
            "locked until the separate readiness and operator-release contracts pass."
        )

    historical_baseline = _as_dict(sources["production_candidate"].get("profitability_baseline"))
    assessment_ready = bool(
        binding.get("identity_consistent", False)
        and binding.get("identity_complete", False)
    )
    if not assessment_ready:
        overall_status = "blocked"
    elif economic_ready:
        overall_status = "ready"
    else:
        overall_status = "collecting"
    live_sections = {
        str(row.get("section_id") or ""): row
        for row in _as_list(sources["live_money_readiness"].get("sections"))
        if isinstance(row, dict)
    }
    payload = {
        "timestamp_utc": now.isoformat(),
        "schema_version": 2,
        "policy_id": str(policy.get("policy_id") or "candidate_bound_profitability_self_assessment_v1"),
        "overall_status": overall_status,
        "assessment_status": "ready" if assessment_ready else "blocked",
        "ok": assessment_ready,
        "system_statement": statement,
        "candidate_binding": binding,
        "measurement": {
            "candidate_id": candidate_id,
            "candidate_post_cost_sample_count": candidate_samples,
            "candidate_post_cost_minimum_samples": minimum_candidate_samples,
            "candidate_post_cost_pnl": round(candidate_pnl, 6),
            "candidate_expectancy_status": str(expectancy.get("status") or "unavailable"),
            "candidate_expectancy_estimable": bool(expectancy.get("evidence_sufficient", False)),
            "historical_active_book_net_pnl": round(_safe_float(active_book.get("ending_net_pnl_total"), 0.0), 6),
            "historical_active_book_candidate_grade_eligible": bool(active_book.get("candidate_grade_eligible", False)),
            "historical_baseline": historical_baseline,
            "accounting_policy": "historical inventory remains visible for risk and exits but cannot grade the current candidate",
        },
        "grades": {
            "implementation_grade": _grade(implementation_score, complete=implementation_ready_count == len(lanes)),
            "implementation_score": implementation_score,
            "implementation_ready_lanes": implementation_ready_count,
            "implementation_lane_count": len(lanes),
            "economic_evidence_grade": economic_grade,
            "economic_evidence_score": round(economic_score, 3),
            "economic_evidence_ready": economic_ready,
            "economic_evidence_ready_controls": _safe_int(firewall.get("evidence_ready_control_count"), 0),
            "economic_evidence_control_count": _safe_int(firewall.get("control_count"), 0),
            "economic_context_source_grade": _grade(
                economic_context_score,
                complete=economic_context_ready,
            ),
            "economic_context_source_score": economic_context_score,
            "economic_context_source_ready": economic_context_ready,
            "economic_context_ready_families": economic_ready_families,
            "economic_context_family_count": economic_family_count,
            "economic_context_ready_runtime_routes": economic_ready_runtime_routes,
            "economic_context_runtime_route_count": economic_runtime_routes,
            "economic_context_selected_source_count": economic_source_count,
            "evidence_ready_lanes": evidence_ready_count,
            "evidence_lane_count": len(lanes),
            "grade_separation_policy": "implementation and source-context completeness never upgrade the economic profitability grade",
        },
        "economic_context_sources": {
            "contract_id": str(economic_context_policy.get("contract_id") or ""),
            "contract_receipt_sha256": str(economic_context.get("contract_receipt_sha256") or ""),
            "selected_source_ids": sorted(
                str(source_id)
                for source_id in _as_list(economic_context.get("selected_source_ids"))
                if str(source_id)
            ),
            "minimum_distinct_selected_sources": minimum_economic_sources,
            "checks": economic_context_checks,
            "context_only": True,
            "profitability_grade_authority": False,
            "live_execution_authority": False,
        },
        "developmental_soak_learning": developmental_learning,
        "eight_lane_program": lanes,
        "needs": needs,
        "next_safe_action": needs[0] if needs else {},
        "allocation_posture": {
            "qualified_sleeve_count": qualified_sleeves,
            "minimum_qualified_sleeves": minimum_profitable_sleeves,
            "suggested_cash_weight": _safe_float(allocation.get("suggested_cash_weight"), 1.0),
            "automatic_allocation_allowed": False,
        },
        "live_readiness_context": {
            "overall_status": str(sources["live_money_readiness"].get("overall_status") or "missing"),
            "live_money_locked": bool(sources["live_money_readiness"].get("live_money_locked", True)),
            "paper_profitability_grade": str(_as_dict(live_sections.get("paper_profitability_control")).get("grade") or ""),
            "risk_controls_grade": str(_as_dict(live_sections.get("risk_controls")).get("grade") or ""),
            "continuous_soak_grade": str(_as_dict(live_sections.get("continuous_soak")).get("grade") or ""),
        },
        "claims": {
            "profitability_guaranteed": False,
            "positive_expectancy_established": economic_ready,
            "economic_context_available": economic_context_ready,
            "economic_context_is_profitability_evidence": False,
            "historical_loss_is_current_candidate_evidence": False,
            "accepted_generation_history_informs_developmental_actions": bool(
                developmental_learning.get("enabled", False)
            ),
            "accepted_generation_history_is_live_promotion_evidence": False,
            "developmental_association_is_causal_proof": False,
            "safe_to_loosen_thresholds_without_replay": False,
            "safe_to_scale_without_evidence": False,
            "automatic_allocation_allowed": False,
            "live_execution_authority": False,
        },
        "source_receipts": receipts,
        "control_contract": {
            "candidate_bound": True,
            "source_receipts_hashed": True,
            "fail_closed_on_candidate_mismatch": True,
            "paper_only_tuning": True,
            "direct_threshold_loosen_allowed": False,
            "cumulative_soak_history_preserved": True,
            "full_soak_clock_reset_requested": False,
            "historical_pnl_preserved_but_not_regraded": True,
            "accepted_generation_outcomes_feed_developmental_learning": bool(
                developmental_learning.get("enabled", False)
            ),
            "developmental_learning_is_identity_and_time_bound": True,
            "historical_generation_outcomes_grant_promotion": False,
            "clean_720_hour_live_promotion_gate_unchanged": True,
            "live_execution_allowed": False,
            "grant_promotion": False,
        },
    }
    payload["assessment_sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    grades = _as_dict(payload.get("grades"))
    measurement = _as_dict(payload.get("measurement"))
    developmental = _as_dict(payload.get("developmental_soak_learning"))
    lines = [
        "# Profitability Self Assessment",
        "",
        f"- Candidate: `{measurement.get('candidate_id') or 'none'}`",
        f"- Implementation: `{grades.get('implementation_grade')}` ({grades.get('implementation_score')}%)",
        f"- Economic evidence: `{grades.get('economic_evidence_grade')}` ({grades.get('economic_evidence_score')}%)",
        f"- Economic source context: `{grades.get('economic_context_source_grade')}` ({grades.get('economic_context_source_score')}%)",
        f"- Candidate post-cost samples: `{measurement.get('candidate_post_cost_sample_count')}/{measurement.get('candidate_post_cost_minimum_samples')}`",
        f"- Live execution authority: `{_as_dict(payload.get('claims')).get('live_execution_authority')}`",
        "",
        str(payload.get("system_statement") or ""),
        "",
        "## Developmental Soak Learning",
        "",
        f"- Status: `{developmental.get('status') or 'missing'}`",
        f"- Verified accepted generations: `{developmental.get('accepted_generation_count', 0)}`",
        f"- Generations with attributable outcomes: `{developmental.get('attributable_generation_count', 0)}`",
        f"- Mature developmental generations: `{developmental.get('mature_developmental_generation_count', 0)}`",
        f"- Bounded paper actions: `{len(_as_list(developmental.get('bounded_paper_action_plan')))}`",
        "- Historical generations earn current economic-grade or clean 720-hour credit: `False`",
        "",
        "## Eight-Lane Program",
        "",
    ]
    for row in _as_list(payload.get("eight_lane_program")):
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `{row.get('lane_id')}` {row.get('title')}: implementation `{row.get('implementation_status')}`, evidence `{row.get('evidence_status')}`"
        )
    lines.extend(["", "## Needs", ""])
    needs = _as_list(payload.get("needs"))
    if not needs:
        lines.append("- No unresolved need in this assessment.")
    for row in needs:
        if isinstance(row, dict):
            lines.append(f"- `{row.get('blocker')}`: {row.get('expected_impact')}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish candidate-bound profitability self-awareness and eight-lane tuning needs.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--config", default="")
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN_PATH))
    parser.add_argument("--no-markdown", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve() if args.config else None
    payload = build_payload(project_root, config_path=config_path)
    write_payload(Path(args.out_file).expanduser(), payload)
    if not args.no_markdown:
        markdown_path = Path(args.markdown_out).expanduser()
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        grades = _as_dict(payload.get("grades"))
        print(
            "profitability_self_assessment "
            f"status={payload.get('overall_status')} "
            f"candidate={_as_dict(payload.get('measurement')).get('candidate_id') or 'none'} "
            f"implementation={grades.get('implementation_grade')} "
            f"economic={grades.get('economic_evidence_grade')} "
            f"economic_sources={grades.get('economic_context_source_grade')} "
            f"needs={len(_as_list(payload.get('needs')))}"
        )
    return 0 if payload.get("ok", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
