from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


POLICY_VERSION = "profitability_hardening_v1"
PAPER_EXECUTION_AUTHORITY_VERSION = "paper_execution_authority_v2"

PAPER_SIGNAL_ROLES = frozenset(
    {
        "signal_sub_bot",
        "options_sub_bot",
        "futures_sub_bot",
        "crypto_sub_bot",
        "macro_sub_bot",
    }
)
PAPER_CONTROL_TOKENS = frozenset(
    {
        "allocator",
        "auth",
        "backlog",
        "calibrator",
        "cockpit",
        "controller",
        "coordinator",
        "dashboard",
        "data_quality",
        "governance",
        "guard",
        "infrastructure",
        "lifecycle",
        "memory",
        "monitor",
        "operator",
        "orchestrat",
        "platform_organ",
        "pruner",
        "registry",
        "report",
        "router",
        "sentinel",
        "storage",
        "supervisor",
        "telemetry",
        "watchdog",
    }
)
PAPER_CONTROL_TRAINING_LANES = frozenset({"governance_effect", "operational_effect"})
PAPER_CONTROL_OBJECTIVES = frozenset({"governance_effect", "operational_effect", "control_outcome"})

FUTURES_CONTRACT_MULTIPLIERS: dict[str, float] = {
    "ES": 50.0,
    "MES": 5.0,
    "NQ": 20.0,
    "MNQ": 2.0,
    "YM": 5.0,
    "MYM": 0.5,
    "RTY": 50.0,
    "M2K": 5.0,
    "CL": 1000.0,
    "MCL": 100.0,
    "GC": 100.0,
    "MGC": 10.0,
    "SI": 5000.0,
    "SIL": 1000.0,
    "HG": 25000.0,
    "NG": 10000.0,
    "RB": 42000.0,
    "HO": 42000.0,
    "ZB": 1000.0,
    "ZN": 1000.0,
    "ZF": 1000.0,
    "ZT": 2000.0,
    "6E": 125000.0,
    "6J": 12500000.0,
    "6B": 62500.0,
}

_FUTURES_CONTRACT_RE = re.compile(r"^/?([A-Z0-9]+?)([FGHJKMNQUVXZ])(\d{1,4})$")
_OCC_CONTRACT_RE = re.compile(r"^[A-Z]{1,6}\d{6}[CP]\d{8}$")


def _float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _clamp01(value: Any) -> float:
    return min(max(_float(value), 0.0), 1.0)


def _field(row: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(key, default)
    return getattr(row, key, default)


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


def evaluate_paper_execution_authority(
    bot: Mapping[str, Any] | Any,
    *,
    segment: str = "core",
    minimum_accuracy: float = 0.56,
    minimum_quality_score: float = 0.50,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Fail closed unless a market-signal bot has explicit, current paper authority."""

    bot_id = str(_field(bot, "bot_id", "") or "").strip()
    role = str(_field(bot, "bot_role", "") or "").strip().lower()
    lane = str(_field(bot, "training_lane", "") or "").strip().lower()
    lifecycle = str(_field(bot, "lifecycle_state", "") or "").strip().lower()
    objective = str(_field(bot, "training_objective_class", "") or "").strip().lower()
    label_family = str(_field(bot, "label_family", "") or "").strip().lower()
    segment_key = str(segment or "core").strip().lower()
    explicit = bool(_field(bot, "paper_execution_authority", False))
    probation = bool(_field(bot, "paper_probation_authority", False))
    probation_requalification = bool(
        _field(bot, "paper_probation_requalification_allowed", False)
    )
    accuracy_raw = _field(bot, "test_accuracy", None)
    quality_raw = _field(bot, "quality_score", None)
    accuracy = _clamp01(accuracy_raw) if accuracy_raw is not None else None
    quality = _clamp01(quality_raw) if quality_raw is not None else None
    evidence = _field(bot, "paper_execution_evidence", {})
    evidence = dict(evidence) if isinstance(evidence, Mapping) else {}
    post_cost_samples = max(int(_float(evidence.get("post_cost_samples"), 0.0)), 0)
    post_cost_lcb = _float(evidence.get("post_cost_lower_confidence_bound"), 0.0)
    positive_post_cost = post_cost_samples >= 30 and post_cost_lcb > 0.0

    reasons: list[str] = []
    if not bot_id:
        reasons.append("missing_bot_id")
    if not bool(_field(bot, "active", False)):
        reasons.append("bot_inactive")
    if bool(_field(bot, "deleted_from_rotation", False)):
        reasons.append("deleted_from_rotation")
    if lifecycle in {"data_collection_only", "retired", "deleted", "quarantined"}:
        reasons.append(f"lifecycle_not_executable:{lifecycle}")
    training_excluded = bool(
        _field(bot, "training_excluded", False)
        or _field(bot, "exclude_from_training", False)
    )
    if training_excluded and not (probation and probation_requalification):
        reasons.append("training_or_quality_excluded")
    if role not in PAPER_SIGNAL_ROLES:
        reasons.append(f"non_signal_role:{role or 'missing'}")
    if lane in PAPER_CONTROL_TRAINING_LANES:
        reasons.append(f"control_training_lane:{lane}")
    if objective in PAPER_CONTROL_OBJECTIVES:
        reasons.append(f"control_objective:{objective}")
    if label_family.startswith(("operational_", "governance_", "control_")):
        reasons.append(f"non_market_label_family:{label_family}")
    lowered_id = bot_id.lower()
    blocked_token = next((token for token in sorted(PAPER_CONTROL_TOKENS) if token in lowered_id), "")
    if blocked_token:
        reasons.append(f"control_identity_token:{blocked_token}")

    if segment_key == "options" and role != "options_sub_bot":
        reasons.append(f"segment_role_mismatch:{segment_key}:{role}")
    elif segment_key == "futures" and role != "futures_sub_bot":
        reasons.append(f"segment_role_mismatch:{segment_key}:{role}")
    elif segment_key in {"core", "all_active"} and role in {"options_sub_bot", "futures_sub_bot"}:
        reasons.append(f"segment_role_mismatch:{segment_key}:{role}")

    if not (explicit or probation):
        reasons.append("explicit_paper_execution_authority_missing")
    if accuracy is None and not positive_post_cost:
        reasons.append("test_accuracy_missing")
    elif accuracy is not None and accuracy < max(float(minimum_accuracy), 0.0) and not positive_post_cost:
        reasons.append(f"test_accuracy_below_floor:{accuracy:.6f}")
    if quality is not None and quality < max(float(minimum_quality_score), 0.0) and not positive_post_cost:
        reasons.append(f"quality_score_below_floor:{quality:.6f}")

    expires = _parse_utc(_field(bot, "paper_execution_authority_expires_utc", ""))
    reference_now = now_utc or datetime.now(timezone.utc)
    if expires is not None and expires <= reference_now.astimezone(timezone.utc):
        reasons.append("paper_execution_authority_expired")

    tier = "qualified" if explicit else ("probation" if probation else "observation_only")
    return {
        "policy_version": PAPER_EXECUTION_AUTHORITY_VERSION,
        "allowed": not reasons,
        "bot_id": bot_id,
        "segment": segment_key,
        "tier": tier,
        "reasons": reasons,
        "test_accuracy": accuracy,
        "quality_score": quality,
        "positive_post_cost_evidence": positive_post_cost,
        "post_cost_samples": post_cost_samples,
        "post_cost_lower_confidence_bound": post_cost_lcb,
        "expires_utc": expires.isoformat() if expires is not None else "",
        "all_bot_observation_allowed": True,
        "paper_execution_authority": bool(explicit),
        "paper_probation_authority": bool(probation),
        "paper_probation_requalification_allowed": bool(probation_requalification),
        "training_excluded": training_excluded,
        "live_execution_authority": False,
    }


def _first_number(source: Mapping[str, Any], keys: Sequence[str]) -> tuple[float, str]:
    for key in keys:
        value = _float(source.get(key), 0.0)
        if value > 0.0:
            return value, key
    return 0.0, ""


def normalize_futures_root(symbol: str) -> str:
    raw = str(symbol or "").strip().upper()
    if raw.startswith("/"):
        raw = raw[1:]
    if raw.endswith("=F"):
        raw = raw[:-2]
    match = _FUTURES_CONTRACT_RE.fullmatch(raw)
    return match.group(1) if match else raw


def resolve_contract_valuation(
    symbol: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve dollar P&L scaling without guessing an unknown derivative contract."""
    meta = dict(metadata or {})
    raw_symbol = str(meta.get("contract_symbol") or symbol or "").strip().upper().replace(" ", "")
    declared_asset = str(
        meta.get("asset_type")
        or meta.get("assetType")
        or meta.get("instrument_type")
        or meta.get("asset_class")
        or ""
    ).strip().upper()
    is_option = "OPTION" in declared_asset or bool(_OCC_CONTRACT_RE.fullmatch(raw_symbol))
    is_future = "FUTURE" in declared_asset or str(symbol or "").strip().startswith("/")
    if not is_future and raw_symbol.endswith("=F"):
        is_future = True

    explicit_multiplier, explicit_key = _first_number(
        meta,
        ("contract_multiplier", "notional_multiplier", "point_value", "pnl_multiplier"),
    )
    if is_option:
        multiplier = explicit_multiplier or 100.0
        return {
            "policy_version": POLICY_VERSION,
            "asset_type": "OPTION",
            "contract_symbol": raw_symbol,
            "contract_root": "",
            "contract_multiplier": float(multiplier),
            "valuation_ready": True,
            "multiplier_source": explicit_key or "listed_option_standard",
            "reason": "ok",
        }

    if is_future:
        root = normalize_futures_root(raw_symbol or symbol)
        multiplier = explicit_multiplier or FUTURES_CONTRACT_MULTIPLIERS.get(root, 0.0)
        ready = bool(multiplier > 0.0)
        return {
            "policy_version": POLICY_VERSION,
            "asset_type": "FUTURE",
            "contract_symbol": raw_symbol,
            "contract_root": root,
            "contract_multiplier": float(multiplier),
            "valuation_ready": ready,
            "multiplier_source": explicit_key or ("curated_contract_spec" if ready else ""),
            "reason": "ok" if ready else f"unknown_futures_contract_multiplier:{root or 'missing_root'}",
        }

    return {
        "policy_version": POLICY_VERSION,
        "asset_type": declared_asset or "SPOT",
        "contract_symbol": raw_symbol,
        "contract_root": "",
        "contract_multiplier": 1.0,
        "valuation_ready": True,
        "multiplier_source": "spot_unit",
        "reason": "ok",
    }


def position_valuation_compatible(
    position: Mapping[str, Any] | None,
    valuation: Mapping[str, Any],
) -> tuple[bool, str]:
    row = dict(position or {})
    if abs(_float(row.get("qty"), 0.0)) <= 1e-12:
        return True, "flat_position"
    expected = _float(valuation.get("contract_multiplier"), 0.0)
    if expected <= 0.0:
        return False, "unresolved_contract_multiplier"
    if "contract_multiplier" not in row:
        if expected == 1.0:
            return True, "legacy_spot_position"
        return False, "legacy_derivative_position_requires_reconciliation"
    current = _float(row.get("contract_multiplier"), 0.0)
    if current <= 0.0 or not math.isclose(current, expected, rel_tol=1e-9, abs_tol=1e-9):
        return False, f"position_multiplier_mismatch:{current:g}!={expected:g}"
    return True, "ok"


def _feature(features: Mapping[str, Any], *keys: str, default: float = 0.5) -> tuple[float, bool]:
    for key in keys:
        if key not in features or features.get(key) in {None, ""}:
            continue
        return _clamp01(features.get(key)), True
    return _clamp01(default), False


def evaluate_profitability_entry(
    *,
    profile: str,
    features: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Fail closed on explicit execution, regime, or overlap hazards for new exposure."""
    values = dict(features or {})
    profile_key = str(profile or "default").strip().lower() or "default"
    tradeability, trade_known = _feature(
        values,
        "market_micro_tradeability_score_norm",
        "tradeability_score",
        default=0.5,
    )
    execution, execution_known = _feature(values, "execution_fitness_norm", default=tradeability)
    source, source_known = _feature(values, "news_source_quality_norm", "source_quality_norm", default=0.5)
    confirmation, confirmation_known = _feature(
        values,
        "core_cross_asset_confirmation_norm",
        "cross_asset_confirmation_norm",
        default=0.5,
    )
    overlap, overlap_known = _feature(
        values,
        "core_portfolio_overlap_pressure_norm",
        "portfolio_overlap_pressure_norm",
        "overlap_pressure_norm",
        default=0.0,
    )
    conflict, conflict_known = _feature(
        values,
        "cross_bot_conflict_norm",
        "allocation_conflict_norm",
        default=0.0,
    )
    spread_known = any(values.get(key) not in {None, ""} for key in ("spread_bps", "model_spread_bps"))
    spread_bps = max(_float(values.get("spread_bps"), _float(values.get("model_spread_bps"), 8.0)), 0.0)
    quote_age_known = values.get("quote_age_ms") not in {None, ""}
    quote_age_ms = max(_float(values.get("quote_age_ms"), 0.0), 0.0)
    liquidity, liquidity_known = _feature(
        values,
        "liquidity_quality_norm",
        "market_micro_liquidity_norm",
        "depth_quality_norm",
        default=0.5,
    )
    session_quality, session_quality_known = _feature(
        values,
        "session_quality_norm",
        "session_edge_norm",
        default=0.5,
    )
    session = str(values.get("market_session") or values.get("session") or "unknown").strip().lower()
    strict_evidence = bool(
        values.get("profitability_strict_evidence_required", False)
        or values.get("paper_execution_authority_v2", False)
    )

    edge_bps = None
    for key in (
        "predicted_edge_lower_confidence_bound_bps",
        "expected_edge_lower_confidence_bound_bps",
        "predicted_edge_bps",
        "expected_alpha_bps",
    ):
        if values.get(key) in {None, ""}:
            continue
        edge_bps = _float(values.get(key), 0.0)
        break
    explicit_round_trip_cost = None
    for key in ("round_trip_cost_bps", "expected_round_trip_cost_bps"):
        if values.get(key) in {None, ""}:
            continue
        explicit_round_trip_cost = max(_float(values.get(key), 0.0), 0.0)
        break
    slippage_bps = max(
        _float(values.get("expected_slippage_bps"), _float(values.get("modeled_slippage_bps"), 0.0)),
        0.0,
    )
    fee_bps = max(_float(values.get("fee_bps"), 0.0), 0.0)
    estimated_round_trip_cost_bps = (
        explicit_round_trip_cost
        if explicit_round_trip_cost is not None
        else (2.0 * spread_bps + 2.0 * slippage_bps + fee_bps)
    )
    edge_cost_multiple = _float(values.get("minimum_edge_cost_multiple"), 2.0)
    edge_cost_multiple = min(max(edge_cost_multiple, 1.0), 5.0)

    trend, trend_known = _feature(
        values,
        "core_regime_specialist_blend_norm",
        "day_regime_trend_norm",
        "market_micro_trend_persistence_norm",
        default=0.5,
    )
    chop, chop_known = _feature(values, "day_regime_chop_norm", "regime_chop_norm", default=0.5)
    event, event_known = _feature(
        values,
        "core_event_reaction_norm",
        "calendar_event_proximity_norm",
        "event_proximity_norm",
        default=0.5,
    )
    futures_edge, futures_known = _feature(values, "core_futures_regime_edge_norm", default=0.5)
    futures_curve, curve_known = _feature(values, "core_futures_curve_alignment_norm", default=0.5)
    options_edge, options_known = _feature(values, "core_options_structure_edge_norm", default=0.5)
    unwind, unwind_known = _feature(values, "core_crypto_unwind_risk_norm", default=0.0)

    profile_family = "general"
    regime_evidence_known = trend_known or confirmation_known
    regime_fit = 0.30 * tradeability + 0.25 * execution + 0.25 * confirmation + 0.20 * trend
    regime_floor = 0.38
    if any(token in profile_key for token in ("futures", "basis", "rates_curve")):
        profile_family = "futures"
        regime_fit = 0.40 * futures_edge + 0.24 * futures_curve + 0.18 * execution + 0.18 * confirmation
        regime_evidence_known = futures_known or curve_known
        if "crypto" in profile_key:
            profile_family = "crypto_futures"
            regime_fit = max(regime_fit - (0.25 * unwind), 0.0)
            regime_evidence_known = regime_evidence_known or unwind_known
        regime_floor = 0.46
    elif any(token in profile_key for token in ("option", "volatility", "variance", "gamma", "vanna")):
        profile_family = "options_volatility"
        regime_fit = 0.38 * options_edge + 0.22 * event + 0.22 * execution + 0.18 * source
        regime_evidence_known = options_known or event_known
        regime_floor = 0.44
    elif any(token in profile_key for token in ("event", "earnings", "dividend_capture")):
        profile_family = "event"
        regime_fit = 0.38 * event + 0.24 * source + 0.22 * execution + 0.16 * confirmation
        regime_evidence_known = event_known or source_known
        regime_floor = 0.45
    elif any(token in profile_key for token in ("stat_arb", "pairs", "mean_revert", "market_neutral")):
        profile_family = "relative_value"
        regime_fit = 0.32 * chop + 0.24 * (1.0 - overlap) + 0.24 * execution + 0.20 * source
        regime_evidence_known = chop_known or overlap_known
        regime_floor = 0.42
    elif any(token in profile_key for token in ("aggressive", "intraday", "swing", "momentum", "trend")):
        profile_family = "directional"
        regime_fit = 0.38 * trend + 0.24 * (1.0 - chop) + 0.22 * execution + 0.16 * confirmation
        regime_evidence_known = trend_known or chop_known
        regime_floor = 0.44
    elif any(token in profile_key for token in ("conservative", "dividend", "bond", "cash_rotation")):
        profile_family = "defensive"
        regime_fit = 0.28 * source + 0.26 * execution + 0.22 * tradeability + 0.14 * confirmation + 0.10 * (1.0 - conflict)
        regime_evidence_known = source_known or execution_known
        regime_floor = 0.43

    execution_style = "marketable_limit"
    if spread_bps >= 12.0 or execution < 0.60:
        execution_style = "passive_limit"
    if quote_age_ms > 5000.0:
        execution_style = "refresh_quote_then_limit"

    spread_ceiling_bps = {
        "defensive": 25.0,
        "directional": 30.0,
        "relative_value": 25.0,
        "event": 40.0,
        "futures": 35.0,
        "crypto_futures": 50.0,
        "options_volatility": 60.0,
    }.get(profile_family, 35.0)
    quote_age_ceiling_ms = 3000.0 if profile_family == "options_volatility" else 5000.0

    blockers: list[str] = []
    missing_evidence: list[str] = []
    for known, name in (
        (trade_known, "tradeability"),
        (execution_known, "execution_fitness"),
        (spread_known, "spread"),
        (quote_age_known, "quote_age"),
        (liquidity_known, "liquidity"),
        (session_quality_known, "session_quality"),
    ):
        if not known:
            missing_evidence.append(name)
    if strict_evidence and missing_evidence:
        blockers.extend(f"{name}_unknown" for name in missing_evidence)
    if trade_known and tradeability < 0.55:
        blockers.append(f"tradeability={tradeability:.3f}<0.550")
    if execution_known and execution < 0.55:
        blockers.append(f"execution_fitness={execution:.3f}<0.550")
    if liquidity_known and liquidity < 0.50:
        blockers.append(f"liquidity={liquidity:.3f}<0.500")
    if session_quality_known and session_quality < 0.50:
        blockers.append(f"session_quality={session_quality:.3f}<0.500")
    if quote_age_known and quote_age_ms > quote_age_ceiling_ms:
        blockers.append(f"quote_age_ms={quote_age_ms:.0f}>{quote_age_ceiling_ms:.0f}")
    if spread_known and spread_bps > spread_ceiling_bps:
        blockers.append(f"spread_bps={spread_bps:.3f}>{spread_ceiling_bps:.3f}")
    if session in {"premarket", "after_hours", "overnight"} and not bool(
        values.get("extended_session_validated", False)
    ):
        blockers.append("extended_session_not_independently_validated")
    if edge_bps is None:
        if strict_evidence:
            blockers.append("predicted_edge_lower_bound_unknown")
    elif edge_bps <= estimated_round_trip_cost_bps * edge_cost_multiple:
        blockers.append(
            f"edge_cost_margin={edge_bps:.3f}<={estimated_round_trip_cost_bps * edge_cost_multiple:.3f}"
        )
    if overlap_known and overlap >= 0.78:
        blockers.append(f"portfolio_overlap={overlap:.3f}>=0.780")
    if conflict_known and conflict >= 0.84:
        blockers.append(f"portfolio_conflict={conflict:.3f}>=0.840")
    if regime_evidence_known and regime_fit < regime_floor:
        blockers.append(f"{profile_family}_regime_fit={regime_fit:.3f}<{regime_floor:.3f}")

    evidence_quality = _clamp01(
        0.25 * tradeability
        + 0.25 * execution
        + 0.18 * source
        + 0.17 * confirmation
        + 0.15 * (1.0 - conflict)
    )
    risk_multiplier = _clamp01(
        (0.30 + 0.70 * evidence_quality)
        * (0.45 + 0.55 * _clamp01(regime_fit))
        * (1.0 - 0.70 * overlap)
        * (1.0 - 0.55 * conflict)
    )
    return {
        "policy_version": POLICY_VERSION,
        "allowed": not blockers,
        "blockers": blockers,
        "profile": profile_key,
        "profile_family": profile_family,
        "regime_fit_norm": round(_clamp01(regime_fit), 6),
        "regime_floor_norm": round(regime_floor, 6),
        "evidence_quality_norm": round(evidence_quality, 6),
        "risk_multiplier_norm": round(risk_multiplier, 6),
        "overlap_pressure_norm": round(overlap, 6),
        "conflict_pressure_norm": round(conflict, 6),
        "missing_evidence": missing_evidence,
        "strict_evidence_required": strict_evidence,
        "entry_economics": {
            "predicted_edge_lower_confidence_bound_bps": edge_bps,
            "estimated_round_trip_cost_bps": round(estimated_round_trip_cost_bps, 6),
            "minimum_edge_cost_multiple": round(edge_cost_multiple, 6),
            "required_edge_bps": round(estimated_round_trip_cost_bps * edge_cost_multiple, 6),
            "positive_cost_margin": bool(
                edge_bps is not None and edge_bps > estimated_round_trip_cost_bps * edge_cost_multiple
            ),
        },
        "execution_plan": {
            "style": execution_style,
            "spread_bps": round(spread_bps, 6),
            "spread_ceiling_bps": round(spread_ceiling_bps, 6),
            "quote_age_ms": round(quote_age_ms, 3),
            "quote_age_ceiling_ms": round(quote_age_ceiling_ms, 3),
            "market_orders_allowed": False,
            "refresh_quote_required": execution_style == "refresh_quote_then_limit",
        },
    }


def coalesce_paper_intents(
    intents: Sequence[Mapping[str, Any]],
    *,
    min_consensus_ratio: float = 0.62,
    min_net_vote_ratio: float = 0.18,
    max_bot_weight: float = 0.15,
    max_correlation_cluster_weight: float = 0.25,
    max_sub_sleeve_weight: float = 0.40,
    max_sleeve_weight: float = 0.55,
    minimum_distinct_clusters: int = 2,
    minimum_distinct_sub_sleeves: int = 1,
    require_hierarchy_identity: bool = False,
) -> dict[str, Any]:
    eligible: list[dict[str, Any]] = []
    skipped = 0
    skipped_reasons: Counter[str] = Counter()
    for raw in intents:
        row = dict(raw or {})
        action = str(row.get("action") or "HOLD").strip().upper()
        if action not in {"BUY", "SELL"} or not bool(row.get("eligible", True)):
            skipped += 1
            skipped_reasons["non_directional_or_ineligible"] += 1
            continue
        if require_hierarchy_identity and any(
            not str(row.get(key) or "").strip()
            for key in ("correlation_cluster_id", "sub_sleeve_id", "sleeve_id")
        ):
            skipped += 1
            skipped_reasons["hierarchy_identity_missing"] += 1
            continue
        score = _clamp01(row.get("score", 0.5))
        base_weight = max(_float(row.get("weight"), 0.0), 0.01)
        accuracy = _clamp01(row.get("test_accuracy", 0.5))
        features = row.get("features") if isinstance(row.get("features"), Mapping) else {}
        strategy_size = _clamp01(features.get("paper_profitability_strategy_size_multiplier_norm", 1.0))
        regime_fit = _clamp01(features.get("profitability_regime_fit_norm", 1.0))
        execution_quality = _clamp01(
            features.get(
                "execution_fitness_norm",
                features.get("market_micro_tradeability_score_norm", 0.5),
            )
        )
        conviction = max(abs(score - 0.5) * 2.0, 0.05)
        post_cost_samples = max(int(_float(row.get("post_cost_samples"), 0.0)), 0)
        post_cost_lcb_raw = row.get("post_cost_lower_confidence_bound")
        post_cost_lcb = _float(post_cost_lcb_raw, 0.0)
        if post_cost_lcb_raw is not None and post_cost_samples >= 30 and post_cost_lcb <= 0.0:
            skipped += 1
            skipped_reasons["nonpositive_post_cost_lower_bound"] += 1
            continue
        evidence_multiplier = 1.0 if post_cost_samples >= 30 and post_cost_lcb > 0.0 else 0.35
        if str(row.get("paper_execution_tier") or "").strip().lower() == "probation":
            evidence_multiplier = min(evidence_multiplier, 0.25)
        effective_weight = (
            base_weight
            * conviction
            * (0.50 + 0.50 * accuracy)
            * strategy_size
            * (0.35 + 0.65 * regime_fit)
            * (0.35 + 0.65 * execution_quality)
            * evidence_multiplier
        )
        if effective_weight <= 1e-12:
            skipped += 1
            skipped_reasons["nonpositive_effective_weight"] += 1
            continue
        row["effective_weight"] = effective_weight
        row["score"] = score
        bot_id = str(row.get("bot_id") or "").strip()
        row["correlation_cluster_id"] = str(
            row.get("correlation_cluster_id") or bot_id or "unknown_cluster"
        )
        row["sub_sleeve_id"] = str(row.get("sub_sleeve_id") or bot_id or "unknown_sub_sleeve")
        row["sleeve_id"] = str(row.get("sleeve_id") or row.get("profile") or "unknown_sleeve")
        row["signal_fingerprint"] = str(
            row.get("signal_fingerprint")
            or hashlib.sha256(
                json.dumps(
                    {
                        "cluster": row["correlation_cluster_id"],
                        "action": action,
                        "score_bucket": round(score, 3),
                        "threshold_bucket": round(_float(row.get("threshold"), 0.55), 3),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        )
        row["evidence_multiplier"] = evidence_multiplier
        eligible.append(row)

    deduplicated: list[dict[str, Any]] = []
    duplicate_groups: Counter[str] = Counter()
    by_fingerprint: dict[str, dict[str, Any]] = {}
    for row in eligible:
        fingerprint = str(row.get("signal_fingerprint") or "")
        current = by_fingerprint.get(fingerprint)
        if current is None or _float(row.get("effective_weight")) > _float(current.get("effective_weight")):
            if current is not None:
                duplicate_groups[fingerprint] += 1
            by_fingerprint[fingerprint] = row
        else:
            duplicate_groups[fingerprint] += 1
    deduplicated.extend(by_fingerprint.values())
    skipped += sum(duplicate_groups.values())
    if duplicate_groups:
        skipped_reasons["duplicate_or_near_duplicate_signal"] += sum(duplicate_groups.values())
    eligible = deduplicated

    raw_total = sum(_float(row.get("effective_weight")) for row in eligible)
    if raw_total > 1e-12:
        for row in eligible:
            row["effective_weight"] = _float(row.get("effective_weight")) / raw_total

    cap_events: list[dict[str, Any]] = []

    def apply_group_cap(key: str, cap: float) -> None:
        bounded_cap = _clamp01(cap)
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in eligible:
            groups[str(item.get(key) or "unknown")].append(item)
        for group_id, members in groups.items():
            group_weight = sum(_float(item.get("effective_weight")) for item in members)
            if group_weight <= bounded_cap + 1e-12 or group_weight <= 0.0:
                continue
            scale = bounded_cap / group_weight
            for item in members:
                item["effective_weight"] = _float(item.get("effective_weight")) * scale
            cap_events.append(
                {
                    "dimension": key,
                    "group_id": group_id,
                    "weight_before": round(group_weight, 8),
                    "weight_after": round(bounded_cap, 8),
                }
            )

    apply_group_cap("bot_id", max_bot_weight)
    apply_group_cap("correlation_cluster_id", max_correlation_cluster_weight)
    apply_group_cap("sub_sleeve_id", max_sub_sleeve_weight)
    apply_group_cap("sleeve_id", max_sleeve_weight)

    buy_weight = sum(_float(row.get("effective_weight")) for row in eligible if row.get("action") == "BUY")
    sell_weight = sum(_float(row.get("effective_weight")) for row in eligible if row.get("action") == "SELL")
    total_weight = buy_weight + sell_weight
    buy_count = sum(1 for row in eligible if row.get("action") == "BUY")
    sell_count = sum(1 for row in eligible if row.get("action") == "SELL")
    ids = sorted({str(row.get("bot_id") or "").strip() for row in eligible if str(row.get("bot_id") or "").strip()})
    ids_hash = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("utf-8")).hexdigest()
    distinct_clusters = len({str(row.get("correlation_cluster_id") or "") for row in eligible})
    distinct_sub_sleeves = len({str(row.get("sub_sleeve_id") or "") for row in eligible})
    diversity_ready = bool(
        distinct_clusters >= max(int(minimum_distinct_clusters), 1)
        and distinct_sub_sleeves >= max(int(minimum_distinct_sub_sleeves), 1)
    )
    if total_weight <= 1e-12:
        return {
            "policy_version": POLICY_VERSION,
            "action": "HOLD",
            "score": 0.5,
            "threshold": 0.55,
            "quantity_multiplier": 0.0,
            "reason": "no_eligible_directional_intents",
            "constituent_count": 0,
            "skipped_count": skipped,
            "skipped_reasons": dict(sorted(skipped_reasons.items())),
            "constituent_bot_ids": [],
            "constituent_bot_ids_sha256": ids_hash,
            "diversity_ready": False,
            "distinct_correlation_clusters": distinct_clusters,
            "distinct_sub_sleeves": distinct_sub_sleeves,
            "weight_cap_events": cap_events,
            "hierarchy_identity_required": bool(require_hierarchy_identity),
        }

    winning_weight = max(buy_weight, sell_weight)
    consensus_ratio = winning_weight / total_weight
    net_vote_ratio = abs(buy_weight - sell_weight) / total_weight
    action = "BUY" if buy_weight > sell_weight else "SELL"
    if not diversity_ready:
        action = "HOLD"
    elif consensus_ratio < _clamp01(min_consensus_ratio) or net_vote_ratio < _clamp01(min_net_vote_ratio):
        action = "HOLD"
    direction = 1.0 if action == "BUY" else (-1.0 if action == "SELL" else 0.0)
    threshold = sum(
        _float(row.get("threshold"), 0.55) * _float(row.get("effective_weight")) for row in eligible
    ) / total_weight
    score_distance = min(max(0.055 + 0.30 * net_vote_ratio, 0.055), 0.45)
    score = 0.5 + direction * score_distance if direction else 0.5
    weighted_size = sum(
        _clamp01(
            (row.get("features") or {}).get("paper_profitability_strategy_size_multiplier_norm", 1.0)
            if isinstance(row.get("features"), Mapping)
            else 1.0
        )
        * _float(row.get("effective_weight"))
        for row in eligible
    ) / total_weight
    quantity_multiplier = _clamp01(weighted_size * (0.55 + 0.45 * consensus_ratio)) if action != "HOLD" else 0.0
    reason = (
        "portfolio_consensus"
        if action != "HOLD"
        else "insufficient_hierarchical_diversity"
        if not diversity_ready
        else "portfolio_consensus_abstention"
    )
    winning_rows = [row for row in eligible if row.get("action") == action]
    winning_total = sum(_float(row.get("effective_weight")) for row in winning_rows)
    constituent_attribution = [
        {
            "bot_id": str(row.get("bot_id") or ""),
            "action": str(row.get("action") or ""),
            "weight_share": round(_float(row.get("effective_weight")) / max(winning_total, 1e-12), 8),
            "score": round(_float(row.get("score"), 0.5), 8),
        }
        for row in sorted(winning_rows, key=lambda item: str(item.get("bot_id") or ""))[:128]
    ]
    return {
        "policy_version": POLICY_VERSION,
        "action": action,
        "score": round(score, 8),
        "threshold": round(min(max(threshold, 0.50), 0.90), 8),
        "quantity_multiplier": round(quantity_multiplier, 6),
        "reason": reason,
        "constituent_count": len(eligible),
        "skipped_count": skipped,
        "skipped_reasons": dict(sorted(skipped_reasons.items())),
        "buy_count": buy_count,
        "sell_count": sell_count,
        "buy_weight": round(buy_weight, 8),
        "sell_weight": round(sell_weight, 8),
        "consensus_ratio": round(consensus_ratio, 6),
        "net_vote_ratio": round(net_vote_ratio, 6),
        "constituent_bot_ids": ids[:64],
        "constituent_bot_ids_truncated": len(ids) > 64,
        "constituent_bot_ids_sha256": ids_hash,
        "constituent_attribution": constituent_attribution,
        "constituent_attribution_truncated": len(winning_rows) > 128,
        "diversity_ready": diversity_ready,
        "distinct_correlation_clusters": distinct_clusters,
        "distinct_sub_sleeves": distinct_sub_sleeves,
        "weight_cap_events": cap_events,
        "duplicate_signal_count": sum(duplicate_groups.values()),
        "correlation_weight_capped": any(
            row.get("dimension") == "correlation_cluster_id" for row in cap_events
        ),
        "hierarchy_identity_required": bool(require_hierarchy_identity),
    }


def post_cost_adjusted_forward_return(
    *,
    action: str,
    forward_return: float,
    entry_cost_bps: float,
    exit_cost_bps: float | None = None,
    fee_bps: float = 0.0,
) -> dict[str, float]:
    side = str(action or "HOLD").strip().upper()
    round_trip_cost_bps = max(_float(entry_cost_bps), 0.0) + max(
        _float(exit_cost_bps, entry_cost_bps),
        0.0,
    ) + max(_float(fee_bps), 0.0)
    raw = _float(forward_return)
    if side == "BUY":
        adjusted = raw - round_trip_cost_bps / 10000.0
    elif side == "SELL":
        adjusted = raw + round_trip_cost_bps / 10000.0
    else:
        adjusted = raw
        round_trip_cost_bps = 0.0
    return {
        "raw_forward_return": raw,
        "post_cost_forward_return": adjusted,
        "round_trip_cost_bps": round_trip_cost_bps,
    }


def evaluate_retirement_evidence(
    evidence: Mapping[str, Any] | None,
    *,
    minimum_samples: int = 100,
    minimum_observed_days: int = 10,
    minimum_failed_retests: int = 3,
) -> dict[str, Any]:
    row = dict(evidence or {})
    samples = max(int(_float(row.get("post_cost_samples"), _float(row.get("samples"), 0.0))), 0)
    observed_days = max(int(_float(row.get("observed_days"), 0.0)), 0)
    failed_retests = max(int(_float(row.get("failed_retests"), row.get("no_improvement_streak", 0))), 0)
    expectancy = _float(row.get("post_cost_expectancy"), row.get("net_expectancy", 0.0))
    lower_bound = _float(row.get("post_cost_lower_confidence_bound"), row.get("lower_confidence_bound", 0.0))
    requirements = {
        "sample_floor": samples >= max(int(minimum_samples), 1),
        "observed_day_floor": observed_days >= max(int(minimum_observed_days), 1),
        "negative_post_cost_expectancy": expectancy < 0.0,
        "negative_post_cost_lower_bound": lower_bound < 0.0,
        "failed_retest_floor": failed_retests >= max(int(minimum_failed_retests), 1),
    }
    return {
        "policy_version": POLICY_VERSION,
        "retire": all(requirements.values()),
        "requirements": requirements,
        "post_cost_samples": samples,
        "observed_days": observed_days,
        "failed_retests": failed_retests,
        "post_cost_expectancy": expectancy,
        "post_cost_lower_confidence_bound": lower_bound,
        "policy": "retire_only_after_repeated_negative_post_cost_out_of_sample_evidence",
    }
