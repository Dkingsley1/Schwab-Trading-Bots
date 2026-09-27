from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

POLICY_SCHEMA_VERSION = 1
DEFAULT_POLICY_PATH = Path("config/paper_behavior_intervention_drill_v1.json")
KNOWN_INTERVENTIONS = (
    "stale_data_abstention",
    "post_cost_edge_gate",
    "evidence_quality_gate",
    "liquidity_size_throttle",
    "drawdown_loss_streak_throttle",
    "crowding_concentration_cap",
    "regime_transition_hysteresis",
    "staged_recovery_reentry",
    "partial_fill_inventory_guard",
    "winner_add_discipline",
    "horizon_conflict_abstention",
    "candidate_maturity_scaling",
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def _clamp(value: Any, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return min(max(_float(value, minimum), minimum), maximum)


def _parse_utc(value: Any) -> datetime | None:
    raw = str(value or "").strip().replace("Z", "+00:00")
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _canonical_payload(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_payload(value)).hexdigest()


def file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def load_policy(
    *,
    project_root: str | Path,
    policy_path: str | Path = DEFAULT_POLICY_PATH,
) -> dict[str, Any]:
    root = Path(project_root)
    path = Path(policy_path).expanduser()
    resolved = path if path.is_absolute() else root / path
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def contract_receipt_sha256(contract: Mapping[str, Any]) -> str:
    payload = dict(contract)
    payload.pop("contract_receipt_sha256", None)
    return canonical_sha256(payload)


def build_runtime_overlay_proposal(
    *,
    policy: Mapping[str, Any],
    candidate_binding: Mapping[str, Any],
    policy_sha256: str,
    generated_at_utc: datetime | None = None,
) -> dict[str, Any]:
    now = generated_at_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    ttl_seconds = max(
        int(_float(policy.get("runtime_contract_ttl_seconds"), 86400)), 300
    )
    invariants = _mapping(policy.get("runtime_invariants"))
    contract = {
        "schema_version": POLICY_SCHEMA_VERSION,
        "contract_id": "paper_behavior_intervention_runtime_v1",
        "active": True,
        "admission_status": "paper_probation",
        "generated_at_utc": now.isoformat(),
        "expires_at_utc": (now + timedelta(seconds=ttl_seconds)).isoformat(),
        "paper_only": True,
        "live_execution_allowed": False,
        "policy_id": str(policy.get("policy_id") or ""),
        "policy_sha256": str(policy_sha256 or ""),
        "candidate_binding": _mapping(candidate_binding),
        "interventions": {
            str(key): _mapping(value)
            for key, value in _mapping(policy.get("interventions")).items()
            if str(key) in KNOWN_INTERVENTIONS
        },
        "invariants": invariants,
        "forward_evidence_contract": {
            "status": "candidate_bound_forward_evidence_pending",
            "minimum_post_cost_fills_for_review": max(
                int(
                    _float(
                        policy.get(
                            "minimum_candidate_bound_post_cost_fills_for_review"
                        ),
                        30,
                    )
                ),
                1,
            ),
            "intervention_specific_labels_required": True,
            "automatic_risk_widening_allowed": False,
            "profitability_claim_allowed": False,
        },
        "rollback_contract": {
            "disable_on_candidate_mismatch": True,
            "disable_on_policy_mismatch": True,
            "disable_on_expiry": True,
            "disable_on_authority_violation": True,
            "invalid_contract_behavior": "no_op_preserve_existing_profitability_controls",
        },
    }
    contract["contract_receipt_sha256"] = contract_receipt_sha256(contract)
    return contract


def validate_runtime_overlay(
    contract: Mapping[str, Any] | None,
    *,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    row = _mapping(contract)
    failures: list[str] = []
    if int(_float(row.get("schema_version"), 0)) != POLICY_SCHEMA_VERSION:
        failures.append("schema_version_invalid")
    if str(row.get("contract_id") or "") != "paper_behavior_intervention_runtime_v1":
        failures.append("contract_id_invalid")
    if not bool(row.get("active", False)):
        failures.append("contract_inactive")
    if str(row.get("admission_status") or "") not in {
        "paper_probation",
        "paper_active",
    }:
        failures.append("admission_status_invalid")
    if row.get("paper_only") is not True:
        failures.append("paper_only_required")
    if row.get("live_execution_allowed") is not False:
        failures.append("live_execution_must_be_false")
    invariants = _mapping(row.get("invariants"))
    required_false = (
        "can_originate_action",
        "can_reverse_action",
        "can_enlarge_entry",
    )
    for key in required_false:
        if invariants.get(key) is not False:
            failures.append(f"{key}_must_be_false")
    for key in ("preserve_hold", "preserve_sell_and_reduce_only"):
        if invariants.get(key) is not True:
            failures.append(f"{key}_must_be_true")
    max_multiplier = _float(invariants.get("max_entry_size_multiplier_norm"), -1.0)
    if max_multiplier < 0.0 or max_multiplier > 1.0:
        failures.append("max_entry_size_multiplier_out_of_range")
    receipt = str(row.get("contract_receipt_sha256") or "")
    if not receipt or receipt != contract_receipt_sha256(row):
        failures.append("contract_receipt_mismatch")
    interventions = _mapping(row.get("interventions"))
    unknown = sorted(set(interventions) - set(KNOWN_INTERVENTIONS))
    if unknown:
        failures.append("unknown_intervention:" + ",".join(unknown))
    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    expires = _parse_utc(row.get("expires_at_utc"))
    if expires is None:
        failures.append("expires_at_invalid")
    elif now.astimezone(timezone.utc) > expires:
        failures.append("contract_expired")
    binding = _mapping(row.get("candidate_binding"))
    if not str(binding.get("candidate_id") or ""):
        failures.append("candidate_id_missing")
    if int(_float(binding.get("candidate_generation"), 0)) <= 0:
        failures.append("candidate_generation_invalid")
    if not str(binding.get("candidate_state_sha256") or ""):
        failures.append("candidate_state_sha256_missing")
    return {
        "ready": not failures,
        "failures": failures,
        "contract_receipt_sha256": receipt,
    }


def _first_feature(
    features: Mapping[str, Any], keys: Sequence[str]
) -> tuple[bool, float, str]:
    for key in keys:
        if key not in features or features.get(key) is None:
            continue
        value = _float(features.get(key), math.nan)
        if math.isfinite(value):
            return True, value, key
    return False, 0.0, ""


def _minimum_feature(
    features: Mapping[str, Any], keys: Sequence[str]
) -> tuple[bool, float, list[str]]:
    found: list[tuple[str, float]] = []
    for key in keys:
        if key not in features or features.get(key) is None:
            continue
        value = _float(features.get(key), math.nan)
        if math.isfinite(value):
            found.append((key, value))
    if not found:
        return False, 0.0, []
    return True, min(value for _, value in found), [key for key, _ in found]


def _maximum_feature(
    features: Mapping[str, Any], keys: Sequence[str]
) -> tuple[bool, float, list[str]]:
    found: list[tuple[str, float]] = []
    for key in keys:
        if key not in features or features.get(key) is None:
            continue
        value = _float(features.get(key), math.nan)
        if math.isfinite(value):
            found.append((key, value))
    if not found:
        return False, 0.0, []
    return True, max(value for _, value in found), [key for key, _ in found]


def _bounded_pressure_multiplier(
    value: float,
    *,
    soft_floor: float,
    hard_floor: float,
    minimum_multiplier: float,
) -> float:
    soft = _float(soft_floor)
    hard = max(_float(hard_floor), soft + 1e-9)
    minimum = _clamp(minimum_multiplier)
    if value <= soft:
        return 1.0
    if value >= hard:
        return 0.0
    remaining = 1.0 - ((value - soft) / (hard - soft))
    return minimum + (1.0 - minimum) * remaining


def _bounded_quality_multiplier(
    value: float,
    *,
    hard_floor: float,
    soft_floor: float,
    minimum_multiplier: float,
) -> float:
    hard = _clamp(hard_floor)
    soft = max(_clamp(soft_floor), hard + 1e-9)
    minimum = _clamp(minimum_multiplier)
    if value < hard:
        return 0.0
    if value >= soft:
        return 1.0
    progress = (value - hard) / (soft - hard)
    return minimum + (1.0 - minimum) * progress


def evaluate_paper_behavior_interventions(
    *,
    action: str,
    features: Mapping[str, Any] | None,
    runtime_contract: Mapping[str, Any] | None,
    profile: str = "default",
    strategy: str = "",
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    requested_action = str(action or "HOLD").strip().upper()
    if requested_action not in {"BUY", "SELL"}:
        requested_action = "HOLD"
    feature_map = _mapping(features)
    validation = validate_runtime_overlay(runtime_contract, now_utc=now_utc)
    contract = _mapping(runtime_contract)
    base = {
        "requested_action": requested_action,
        "action": requested_action,
        "entry_size_multiplier_norm": 1.0,
        "blocked": False,
        "active": bool(validation.get("ready", False)),
        "triggered_interventions": [],
        "observations": [],
        "reasons": [],
        "profile": str(profile or "default").strip().lower() or "default",
        "strategy": str(strategy or ""),
        "contract_receipt_sha256": str(contract.get("contract_receipt_sha256") or ""),
        "validation_failures": list(validation.get("failures") or []),
    }
    if not validation.get("ready", False):
        base["reasons"] = ["paper_behavior_intervention_overlay_disabled"]
        return base
    if requested_action == "SELL":
        base["reasons"] = ["paper_behavior_intervention_exit_path_preserved"]
        return base
    if requested_action == "HOLD":
        base["reasons"] = ["paper_behavior_intervention_hold_preserved"]
        return base

    interventions = _mapping(contract.get("interventions"))
    max_multiplier = _clamp(
        _mapping(contract.get("invariants")).get("max_entry_size_multiplier_norm", 1.0)
    )
    multiplier = max_multiplier
    blockers: list[str] = []
    triggered: list[str] = []
    observations: list[dict[str, Any]] = []

    def enabled(intervention_id: str) -> tuple[bool, dict[str, Any]]:
        row = _mapping(interventions.get(intervention_id))
        return bool(row.get("enabled", False)), row

    def trigger(
        intervention_id: str,
        *,
        disposition: str,
        value: float | None = None,
        threshold: float | None = None,
        feature_keys: Sequence[str] = (),
        size_cap: float | None = None,
    ) -> None:
        nonlocal multiplier
        if intervention_id not in triggered:
            triggered.append(intervention_id)
        if disposition == "block":
            blockers.append(intervention_id)
            multiplier = 0.0
        elif size_cap is not None:
            multiplier = min(multiplier, _clamp(size_cap))
        observations.append(
            {
                "intervention_id": intervention_id,
                "disposition": disposition,
                "value": round(value, 8) if value is not None else None,
                "threshold": round(threshold, 8) if threshold is not None else None,
                "feature_keys": list(feature_keys),
                "size_cap_norm": (
                    round(_clamp(size_cap), 8) if size_cap is not None else None
                ),
            }
        )

    is_enabled, policy = enabled("stale_data_abstention")
    if is_enabled:
        present, quote_age, key = _first_feature(
            feature_map, ("quote_age_ms", "market_quote_age_ms")
        )
        quote_limit = max(_float(policy.get("max_quote_age_ms"), 1500.0), 1.0)
        if present and quote_age > quote_limit:
            trigger(
                "stale_data_abstention",
                disposition="block",
                value=quote_age,
                threshold=quote_limit,
                feature_keys=(key,),
            )
        present, data_age, key = _first_feature(
            feature_map,
            (
                "market_data_age_seconds",
                "data_age_seconds",
                "snapshot_age_seconds",
            ),
        )
        data_limit = max(_float(policy.get("max_market_data_age_seconds"), 5.0), 0.1)
        if present and data_age > data_limit:
            trigger(
                "stale_data_abstention",
                disposition="block",
                value=data_age,
                threshold=data_limit,
                feature_keys=(key,),
            )
        present, freshness, key = _first_feature(
            feature_map,
            ("source_freshness_norm", "data_freshness_norm"),
        )
        freshness_floor = _clamp(policy.get("minimum_source_freshness_norm", 0.55))
        if present and freshness < freshness_floor:
            trigger(
                "stale_data_abstention",
                disposition="block",
                value=freshness,
                threshold=freshness_floor,
                feature_keys=(key,),
            )

    is_enabled, policy = enabled("post_cost_edge_gate")
    if is_enabled:
        edge_present, edge_bps, edge_key = _first_feature(
            feature_map,
            (
                "predicted_edge_lower_confidence_bound_bps",
                "predicted_edge_lcb_bps",
                "net_edge_lcb_bps",
            ),
        )
        cost_present, cost_bps, cost_key = _first_feature(
            feature_map,
            ("round_trip_cost_bps", "expected_round_trip_cost_bps"),
        )
        if edge_present and cost_present:
            required = max(
                max(cost_bps, 0.0)
                * max(_float(policy.get("minimum_edge_cost_multiple"), 1.5), 1.0),
                max(cost_bps, 0.0)
                + max(_float(policy.get("minimum_edge_margin_bps"), 2.0), 0.0),
            )
            if edge_bps < required:
                trigger(
                    "post_cost_edge_gate",
                    disposition="block",
                    value=edge_bps,
                    threshold=required,
                    feature_keys=(edge_key, cost_key),
                )

    is_enabled, policy = enabled("evidence_quality_gate")
    if is_enabled:
        present, quality, keys = _minimum_feature(
            feature_map,
            (
                "profitability_evidence_quality_norm",
                "news_source_quality_norm",
                "source_quality_norm",
            ),
        )
        if present:
            hard = _clamp(policy.get("hard_floor_norm", 0.5))
            soft = _clamp(policy.get("soft_floor_norm", 0.7))
            quality_multiplier = _bounded_quality_multiplier(
                quality,
                hard_floor=hard,
                soft_floor=soft,
                minimum_multiplier=_clamp(
                    policy.get("soft_floor_size_multiplier_norm", 0.55)
                ),
            )
            if quality_multiplier <= 0.0:
                trigger(
                    "evidence_quality_gate",
                    disposition="block",
                    value=quality,
                    threshold=hard,
                    feature_keys=keys,
                )
            elif quality_multiplier < 1.0:
                trigger(
                    "evidence_quality_gate",
                    disposition="throttle",
                    value=quality,
                    threshold=soft,
                    feature_keys=keys,
                    size_cap=quality_multiplier,
                )
        present, channel_count, key = _first_feature(
            feature_map,
            (
                "independent_evidence_channel_count",
                "paper_profitability_independent_evidence_channel_count",
            ),
        )
        channel_floor = max(
            int(_float(policy.get("minimum_independent_channels"), 3)), 1
        )
        if present and int(channel_count) < channel_floor:
            trigger(
                "evidence_quality_gate",
                disposition="block",
                value=channel_count,
                threshold=float(channel_floor),
                feature_keys=(key,),
            )

    is_enabled, policy = enabled("liquidity_size_throttle")
    if is_enabled:
        present, liquidity, keys = _minimum_feature(
            feature_map,
            (
                "market_micro_tradeability_score_norm",
                "execution_fitness_norm",
                "liquidity_quality_norm",
            ),
        )
        if present:
            hard = _clamp(policy.get("hard_floor_norm", 0.35))
            soft = _clamp(policy.get("soft_floor_norm", 0.65))
            if liquidity < hard:
                trigger(
                    "liquidity_size_throttle",
                    disposition="block",
                    value=liquidity,
                    threshold=hard,
                    feature_keys=keys,
                )
            elif liquidity < soft:
                cap = _bounded_quality_multiplier(
                    liquidity,
                    hard_floor=hard,
                    soft_floor=soft,
                    minimum_multiplier=_clamp(
                        policy.get("minimum_size_multiplier_norm", 0.35)
                    ),
                )
                trigger(
                    "liquidity_size_throttle",
                    disposition="throttle",
                    value=liquidity,
                    threshold=soft,
                    feature_keys=keys,
                    size_cap=cap,
                )

    is_enabled, policy = enabled("drawdown_loss_streak_throttle")
    if is_enabled:
        present, drawdown, keys = _maximum_feature(
            feature_map,
            (
                "portfolio_drawdown_pressure_norm",
                "drawdown_pressure_norm",
                "intraday_drawdown_pressure_norm",
            ),
        )
        if present:
            soft = _clamp(policy.get("drawdown_soft_floor_norm", 0.55))
            hard = _clamp(policy.get("drawdown_hard_floor_norm", 0.85))
            if drawdown >= hard:
                trigger(
                    "drawdown_loss_streak_throttle",
                    disposition="block",
                    value=drawdown,
                    threshold=hard,
                    feature_keys=keys,
                )
            elif drawdown > soft:
                cap = _bounded_pressure_multiplier(
                    drawdown,
                    soft_floor=soft,
                    hard_floor=hard,
                    minimum_multiplier=_clamp(
                        policy.get("minimum_size_multiplier_norm", 0.25)
                    ),
                )
                trigger(
                    "drawdown_loss_streak_throttle",
                    disposition="throttle",
                    value=drawdown,
                    threshold=soft,
                    feature_keys=keys,
                    size_cap=cap,
                )
        present, loss_streak, key = _first_feature(
            feature_map,
            ("lane_loss_streak", "strategy_loss_streak", "paper_loss_streak"),
        )
        if present:
            soft_count = max(int(_float(policy.get("loss_streak_soft_count"), 3)), 1)
            hard_count = max(
                int(_float(policy.get("loss_streak_hard_count"), 6)),
                soft_count + 1,
            )
            if loss_streak >= hard_count:
                trigger(
                    "drawdown_loss_streak_throttle",
                    disposition="block",
                    value=loss_streak,
                    threshold=float(hard_count),
                    feature_keys=(key,),
                )
            elif loss_streak >= soft_count:
                cap = _bounded_pressure_multiplier(
                    loss_streak,
                    soft_floor=float(soft_count - 1),
                    hard_floor=float(hard_count),
                    minimum_multiplier=_clamp(
                        policy.get("minimum_size_multiplier_norm", 0.25)
                    ),
                )
                trigger(
                    "drawdown_loss_streak_throttle",
                    disposition="throttle",
                    value=loss_streak,
                    threshold=float(soft_count),
                    feature_keys=(key,),
                    size_cap=cap,
                )

    for intervention_id, keys in (
        (
            "crowding_concentration_cap",
            (
                "core_portfolio_overlap_pressure_norm",
                "portfolio_crowding_norm",
                "position_concentration_norm",
                "correlation_pressure_norm",
            ),
        ),
        (
            "regime_transition_hysteresis",
            (
                "regime_transition_risk_norm",
                "regime_dislocation_norm",
                "lead_lag_break_norm",
            ),
        ),
        (
            "partial_fill_inventory_guard",
            (
                "unresolved_partial_fill_inventory_norm",
                "partial_fill_inventory_pressure_norm",
            ),
        ),
        (
            "horizon_conflict_abstention",
            (
                "cross_horizon_conflict_norm",
                "horizon_ownership_conflict_norm",
            ),
        ),
    ):
        is_enabled, policy = enabled(intervention_id)
        if not is_enabled:
            continue
        present, pressure, used_keys = _maximum_feature(feature_map, keys)
        if not present:
            continue
        soft = _clamp(policy.get("soft_floor_norm", 0.5))
        hard = _clamp(policy.get("hard_floor_norm", 0.8))
        if pressure >= hard:
            trigger(
                intervention_id,
                disposition="block",
                value=pressure,
                threshold=hard,
                feature_keys=used_keys,
            )
        elif pressure > soft:
            cap = _bounded_pressure_multiplier(
                pressure,
                soft_floor=soft,
                hard_floor=hard,
                minimum_multiplier=_clamp(
                    policy.get("minimum_size_multiplier_norm", 0.35)
                ),
            )
            trigger(
                intervention_id,
                disposition="throttle",
                value=pressure,
                threshold=soft,
                feature_keys=used_keys,
                size_cap=cap,
            )

    is_enabled, policy = enabled("staged_recovery_reentry")
    if is_enabled:
        present, confirmation, key = _first_feature(
            feature_map,
            (
                "recovery_reentry_confirmation_norm",
                "recovery_confirmation_norm",
            ),
        )
        floor = _clamp(policy.get("minimum_confirmation_norm", 0.6))
        if present and confirmation < floor:
            trigger(
                "staged_recovery_reentry",
                disposition="block",
                value=confirmation,
                threshold=floor,
                feature_keys=(key,),
            )
        present, recovery_cap, key = _first_feature(
            feature_map,
            (
                "paper_debt_recovery_entry_size_multiplier_norm",
                "recovery_entry_size_multiplier_norm",
            ),
        )
        if present and recovery_cap < 1.0:
            trigger(
                "staged_recovery_reentry",
                disposition="throttle",
                value=recovery_cap,
                threshold=1.0,
                feature_keys=(key,),
                size_cap=recovery_cap,
            )

    is_enabled, policy = enabled("winner_add_discipline")
    if is_enabled:
        runner_present, runner, runner_key = _first_feature(
            feature_map,
            (
                "paper_strategy_profit_harvest_runner_protected_norm",
                "profitable_runner_active_norm",
            ),
        )
        confirm_present, confirmation, confirmation_key = _first_feature(
            feature_map,
            (
                "core_cross_asset_confirmation_norm",
                "cross_asset_confirmation_norm",
            ),
        )
        floor = _clamp(policy.get("minimum_confirmation_norm", 0.65))
        if (
            runner_present
            and runner >= 0.5
            and confirm_present
            and confirmation < floor
        ):
            trigger(
                "winner_add_discipline",
                disposition="block",
                value=confirmation,
                threshold=floor,
                feature_keys=(runner_key, confirmation_key),
            )

    is_enabled, policy = enabled("candidate_maturity_scaling")
    if is_enabled:
        samples_present, samples, samples_key = _first_feature(
            feature_map,
            (
                "candidate_bound_post_cost_samples",
                "post_cost_samples",
            ),
        )
        minimum_samples = max(
            int(_float(policy.get("minimum_post_cost_samples"), 30)), 1
        )
        if samples_present and samples < minimum_samples:
            cap = _clamp(policy.get("immature_size_multiplier_norm", 0.25))
            trigger(
                "candidate_maturity_scaling",
                disposition="throttle",
                value=samples,
                threshold=float(minimum_samples),
                feature_keys=(samples_key,),
                size_cap=cap,
            )
        lcb_present, lcb, lcb_key = _first_feature(
            feature_map,
            (
                "candidate_bound_post_cost_lower_confidence_bound",
                "post_cost_lower_confidence_bound",
            ),
        )
        lcb_floor = _float(policy.get("mature_lower_confidence_bound_floor"), 0.0)
        if (
            samples_present
            and samples >= minimum_samples
            and lcb_present
            and lcb <= lcb_floor
        ):
            trigger(
                "candidate_maturity_scaling",
                disposition="block",
                value=lcb,
                threshold=lcb_floor,
                feature_keys=(samples_key, lcb_key),
            )

    blocked = bool(blockers) or multiplier <= 1e-9
    final_action = "HOLD" if blocked else "BUY"
    return {
        **base,
        "action": final_action,
        "entry_size_multiplier_norm": round(0.0 if blocked else multiplier, 8),
        "blocked": blocked,
        "triggered_interventions": triggered,
        "observations": observations,
        "reasons": [
            f"paper_behavior_intervention:{intervention_id}"
            for intervention_id in triggered
        ]
        + (["paper_behavior_intervention_entry_blocked"] if blocked else []),
    }


def intervention_telemetry_features(result: Mapping[str, Any]) -> dict[str, float]:
    triggered = {str(value) for value in result.get("triggered_interventions", [])}
    features = {
        "paper_behavior_intervention_active_norm": (
            1.0 if bool(result.get("active", False)) else 0.0
        ),
        "paper_behavior_intervention_blocked_norm": (
            1.0 if bool(result.get("blocked", False)) else 0.0
        ),
        "paper_behavior_intervention_size_multiplier_norm": _clamp(
            result.get("entry_size_multiplier_norm", 1.0)
        ),
        "paper_behavior_intervention_trigger_count_norm": _clamp(
            len(triggered) / max(len(KNOWN_INTERVENTIONS), 1)
        ),
    }
    for intervention_id in KNOWN_INTERVENTIONS:
        features[f"paper_behavior_{intervention_id}_active_norm"] = (
            1.0 if intervention_id in triggered else 0.0
        )
    return features
