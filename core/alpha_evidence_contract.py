"""Candidate-bound alpha evidence and advisory cross-sleeve allocation contracts.

This module never changes a decision, size, allocation, promotion, or order. It
adds provenance to evidence rows and computes fail-closed research advice from
observed candidate-forward outcomes.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_POLICY_PATH = "config/alpha_generation_control_v1.json"
DEFAULT_CANDIDATE_PATH = "governance/runtime/production_candidate_state.json"
_JSON_CACHE: dict[str, dict[str, Any]] = {}


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _parse_timestamp(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _read_cached_json(path: Path, *, ttl_seconds: float = 5.0) -> dict[str, Any]:
    key = str(path.resolve())
    now = time.monotonic()
    try:
        stat = path.stat()
        fingerprint = (int(stat.st_mtime_ns), int(stat.st_size))
    except OSError:
        fingerprint = None
    cached = _JSON_CACHE.get(key, {})
    if fingerprint == cached.get("fingerprint") and now - float(
        cached.get("checked_at", 0.0) or 0.0
    ) < max(ttl_seconds, 0.0):
        return _as_dict(cached.get("payload"))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    payload = _as_dict(payload)
    _JSON_CACHE[key] = {
        "checked_at": now,
        "fingerprint": fingerprint,
        "payload": payload,
    }
    return dict(payload)


def clear_runtime_cache() -> None:
    _JSON_CACHE.clear()


def load_alpha_policy(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    return _read_cached_json(root / DEFAULT_POLICY_PATH)


def candidate_context(
    project_root: str | Path,
    *,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    resolved_policy = _as_dict(policy) or load_alpha_policy(root)
    binding_policy = _as_dict(resolved_policy.get("candidate_binding"))
    state_path = root / str(binding_policy.get("state_path") or DEFAULT_CANDIDATE_PATH)
    state = _read_cached_json(
        state_path,
        ttl_seconds=float(binding_policy.get("cache_seconds", 5) or 5),
    )
    windows = _as_dict(state.get("scope_windows_started_utc"))
    scopes = _as_list(binding_policy.get("scope_names")) or [
        "strategy",
        "execution",
        "risk",
        "data",
        "promotion",
        "dependencies",
    ]
    parsed_windows = [
        parsed
        for scope in scopes
        if (parsed := _parse_timestamp(windows.get(str(scope)))) is not None
    ]
    accepted = _parse_timestamp(state.get("accepted_at_utc"))
    if accepted is not None:
        parsed_windows.append(accepted)
    cutoff = max(parsed_windows, default=None)
    return {
        "candidate_id": str(state.get("candidate_id") or "").strip(),
        "generation": int(state.get("generation", 0) or 0),
        "accepted_at_utc": accepted.isoformat() if accepted else "",
        "scope_cutoff_utc": cutoff.isoformat() if cutoff else "",
        "state_receipt_sha256": str(state.get("overall_sha256") or ""),
        "state_path": str(state_path),
        "state_present": bool(state),
        "live_execution_authority": bool(state.get("live_execution_authority", False)),
    }


def _first_number(
    rows: Sequence[Mapping[str, Any]], keys: Iterable[str]
) -> tuple[float | None, str]:
    for row in rows:
        for key in keys:
            if key not in row:
                continue
            number = _safe_float(row.get(key))
            if number is not None:
                return number, str(key)
    return None, ""


def _evidence_sources(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    metadata = _as_dict(row.get("metadata"))
    return [
        dict(row),
        _as_dict(row.get("features")),
        metadata,
        _as_dict(row.get("execution")),
        _as_dict(row.get("paper_order")),
        _as_dict(metadata.get("execution_cost_model")),
        _as_dict(metadata.get("alpha_evidence")),
    ]


def build_net_edge_contract(
    row: Mapping[str, Any],
    *,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_policy = _as_dict(policy)
    edge_policy = _as_dict(resolved_policy.get("net_edge"))
    sources = _evidence_sources(row)
    gross_fields = _as_list(edge_policy.get("gross_edge_fields")) or [
        "expected_gross_edge_bps",
        "gross_edge_bps",
        "expected_edge_bps",
    ]
    post_cost_fields = _as_list(edge_policy.get("explicit_post_cost_edge_fields")) or [
        "expected_post_cost_edge_bps",
        "post_cost_edge_bps",
    ]
    gross_edge, gross_source = _first_number(sources, gross_fields)
    explicit_post_cost, post_cost_source = _first_number(sources, post_cost_fields)

    spread, spread_source = _first_number(
        sources,
        ("half_spread_bps", "spread_bps", "quoted_spread_bps", "bid_ask_spread_bps"),
    )
    spread_fraction = float(edge_policy.get("spread_charge_fraction", 0.5) or 0.5)
    if spread is not None and spread_source != "half_spread_bps":
        spread *= spread_fraction
    fees, fees_source = _first_number(
        sources, ("fee_bps", "fees_bps", "commission_bps", "expected_fee_bps")
    )
    slippage, slippage_source = _first_number(
        sources, ("slippage_bps", "expected_slippage_bps", "realized_slippage_bps")
    )
    impact, impact_source = _first_number(
        sources, ("market_impact_bps", "impact_bps", "expected_market_impact_bps")
    )
    financing, financing_source = _first_number(
        sources, ("financing_bps", "funding_bps", "carry_cost_bps")
    )
    borrow, borrow_source = _first_number(
        sources, ("borrow_bps", "borrow_cost_bps", "hard_to_borrow_bps")
    )
    components = {
        "half_spread": spread,
        "fees": fees,
        "slippage": slippage,
        "market_impact": impact,
        "financing": financing,
        "borrow": borrow,
    }
    component_sources = {
        "half_spread": spread_source,
        "fees": fees_source,
        "slippage": slippage_source,
        "market_impact": impact_source,
        "financing": financing_source,
        "borrow": borrow_source,
    }
    required = [
        str(item)
        for item in (
            _as_list(edge_policy.get("required_cost_components"))
            or [
                "half_spread",
                "fees",
                "slippage",
                "market_impact",
            ]
        )
    ]
    missing_required = [key for key in required if components.get(key) is None]
    known_cost = sum(value for value in components.values() if value is not None)
    uncertainty = float(edge_policy.get("uncertainty_buffer_bps", 1.0) or 0.0)
    basis = ""
    conservative_net: float | None = None
    estimable = False
    if explicit_post_cost is not None:
        basis = "explicit_post_cost_edge"
        conservative_net = explicit_post_cost - uncertainty
        estimable = True
    elif gross_edge is not None and not missing_required:
        basis = "gross_edge_minus_observed_costs"
        conservative_net = gross_edge - known_cost - uncertainty
        estimable = True
    elif gross_edge is None:
        basis = "gross_edge_missing"
    else:
        basis = "required_cost_components_missing"
    minimum_edge = float(
        edge_policy.get("minimum_conservative_net_edge_bps", 0.0) or 0.0
    )
    return {
        "schema_version": 1,
        "status": "estimable" if estimable else "non_estimable",
        "estimable": estimable,
        "basis": basis,
        "gross_edge_bps": gross_edge,
        "gross_edge_source": gross_source,
        "explicit_post_cost_edge_bps": explicit_post_cost,
        "explicit_post_cost_edge_source": post_cost_source,
        "cost_components_bps": components,
        "cost_component_sources": component_sources,
        "known_cost_bps": round(known_cost, 8),
        "required_cost_components": required,
        "missing_required_cost_components": missing_required,
        "uncertainty_buffer_bps": uncertainty,
        "conservative_net_edge_bps": (
            round(conservative_net, 8) if conservative_net is not None else None
        ),
        "positive_conservative_net_edge": bool(
            estimable
            and conservative_net is not None
            and conservative_net > minimum_edge
        ),
        "unknown_cost_defaults_used": False,
        "model_score_converted_to_edge": False,
    }


def _row_candidate_id(row: Mapping[str, Any]) -> str:
    metadata = _as_dict(row.get("metadata"))
    provenance = _as_dict(row.get("provenance"))
    for value in (
        row.get("production_candidate_id"),
        row.get("candidate_id"),
        metadata.get("production_candidate_id"),
        metadata.get("candidate_id"),
        provenance.get("production_candidate_id"),
        provenance.get("candidate_id"),
    ):
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _row_timestamp(row: Mapping[str, Any]) -> datetime | None:
    metadata = _as_dict(row.get("metadata"))
    for value in (
        row.get("timestamp_utc"),
        row.get("timestamp"),
        row.get("created_at_utc"),
        metadata.get("timestamp_utc"),
        metadata.get("decision_timestamp_utc"),
    ):
        parsed = _parse_timestamp(value)
        if parsed is not None:
            return parsed
    return None


def bind_candidate_identity(
    row: Mapping[str, Any],
    *,
    project_root: str | Path,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    out = dict(row)
    metadata = _as_dict(out.get("metadata"))
    context = candidate_context(project_root, policy=policy)
    expected = str(context.get("candidate_id") or "")
    actual = _row_candidate_id(out)
    timestamp = _row_timestamp(out)
    cutoff = _parse_timestamp(context.get("scope_cutoff_utc"))
    after_cutoff = bool(
        timestamp is not None and cutoff is not None and timestamp >= cutoff
    )
    attached = False
    status = "candidate_state_missing"
    if not expected:
        status = "candidate_state_missing"
    elif timestamp is None:
        status = "row_timestamp_missing"
    elif cutoff is None:
        status = "candidate_cutoff_missing"
    elif not after_cutoff:
        status = "pre_candidate_history"
    elif actual and actual != expected:
        status = "candidate_identity_mismatch"
    else:
        if not actual:
            actual = expected
            out["production_candidate_id"] = expected
            metadata["production_candidate_id"] = expected
            metadata["production_candidate_generation"] = int(
                context.get("generation", 0) or 0
            )
            metadata["production_candidate_scope_started_utc"] = str(
                context.get("scope_cutoff_utc") or ""
            )
            metadata["production_candidate_receipt_sha256"] = str(
                context.get("state_receipt_sha256") or ""
            )
            attached = True
        status = "candidate_bound"
    if metadata:
        out["metadata"] = metadata
    out["candidate_binding"] = {
        "schema_version": 1,
        "status": status,
        "candidate_bound": status == "candidate_bound",
        "identity_attached": attached,
        "expected_candidate_id": expected,
        "observed_candidate_id": actual,
        "generation": int(context.get("generation", 0) or 0),
        "row_timestamp_utc": timestamp.isoformat() if timestamp else "",
        "candidate_scope_cutoff_utc": str(context.get("scope_cutoff_utc") or ""),
        "historical_relabeling_allowed": False,
        "mismatched_identity_overwrite_allowed": False,
    }
    return out


def _clean_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    return "_".join(part for part in text.replace("-", "_").split("_") if part)


def _alpha_owner_contract(row: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _as_dict(row.get("metadata"))
    specialization = _as_dict(metadata.get("strategy_specialization"))
    profile = _clean_label(
        row.get("profile")
        or row.get("shadow_profile")
        or metadata.get("source_profile")
    )
    horizon = (
        _clean_label(
            row.get("horizon")
            or specialization.get("primary_horizon")
            or specialization.get("selected_horizon")
            or metadata.get("label_horizon")
        )
        or "unspecified"
    )
    alpha_family = (
        _clean_label(
            row.get("alpha_family")
            or specialization.get("alpha_family")
            or specialization.get("strategy_family")
            or row.get("strategy")
        )
        or "unspecified"
    )
    symbol = str(row.get("symbol") or "").strip().upper() or "UNSPECIFIED"
    key = f"{symbol}:{horizon}:{alpha_family}"
    return {
        "schema_version": 1,
        "sleeve": profile,
        "ownership_key": key,
        "ownership_key_sha256": hashlib.sha256(key.encode("utf-8")).hexdigest(),
        "shared_context_only": True,
        "shared_trade_logic_allowed": False,
        "exposure_owner_selected_from_candidate_bound_residual_alpha": True,
        "automatic_allocation_allowed": False,
    }


def _alpha_relevant(row: Mapping[str, Any], *, channel: str, path_hint: str) -> bool:
    haystack = " ".join(
        [
            str(channel or row.get("channel") or ""),
            str(row.get("event") or ""),
            str(row.get("status") or ""),
            str(path_hint or ""),
        ]
    ).lower()
    return any(
        token in haystack
        for token in (
            "decision",
            "execution",
            "paper_trade",
            "paper_bridge",
            "signal_generation",
            "shadow_pnl_attribution",
            "risk",
        )
    )


def enrich_alpha_evidence(
    row: Mapping[str, Any],
    *,
    project_root: str | Path,
    channel: str = "",
    path_hint: str = "",
) -> dict[str, Any]:
    out = dict(row)
    if not _alpha_relevant(out, channel=channel, path_hint=path_hint):
        return out
    policy = load_alpha_policy(project_root)
    out = bind_candidate_identity(out, project_root=project_root, policy=policy)
    out["alpha_evidence_contract"] = {
        "schema_version": 1,
        "objective": str(
            policy.get("primary_objective")
            or "candidate_bound_post_cost_residual_alpha"
        ),
        "candidate_binding_status": _as_dict(out.get("candidate_binding")).get(
            "status"
        ),
        "net_edge": build_net_edge_contract(out, policy=policy),
        "post_cost_outcomes_grade_alpha": True,
        "historical_book_totals_grade_current_candidate": False,
        "live_execution_authority": False,
    }
    out["cross_sleeve_alpha_contract"] = _alpha_owner_contract(out)
    return out


def _daily_series(rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for row in rows:
        day = str(row.get("day_utc") or "").strip()
        if not day:
            continue
        value = _safe_float(
            row.get("mean_post_cost_return_bps")
            if row.get("mean_post_cost_return_bps") is not None
            else row.get("post_cost_return_bps_total")
        )
        if value is not None:
            result[day] = value
    return result


def _mean_lcb(values: Sequence[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = statistics.fmean(values)
    if len(values) == 1:
        return mean, None
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    return mean, mean - 1.96 * standard_error


def _correlation(
    left: Mapping[str, float], right: Mapping[str, float]
) -> tuple[float | None, int]:
    days = sorted(set(left).intersection(right))
    if len(days) < 2:
        return None, len(days)
    left_values = [left[day] for day in days]
    right_values = [right[day] for day in days]
    left_mean = statistics.fmean(left_values)
    right_mean = statistics.fmean(right_values)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left_values, right_values)
    )
    left_ss = sum((value - left_mean) ** 2 for value in left_values)
    right_ss = sum((value - right_mean) ** 2 for value in right_values)
    denominator = math.sqrt(left_ss * right_ss)
    if denominator <= 0.0:
        return None, len(days)
    return numerator / denominator, len(days)


def _capped_weights(scores: Mapping[str, float], cap: float) -> dict[str, float]:
    remaining = set(scores)
    weights: dict[str, float] = {}
    remaining_weight = 1.0
    while remaining and remaining_weight > 1e-12:
        total = sum(max(scores[key], 0.0) for key in remaining)
        proposed = {
            key: (
                remaining_weight * max(scores[key], 0.0) / total
                if total > 0.0
                else remaining_weight / len(remaining)
            )
            for key in remaining
        }
        capped = [key for key, value in proposed.items() if value > cap]
        if not capped:
            weights.update(proposed)
            break
        for key in capped:
            weights[key] = cap
            remaining_weight -= cap
            remaining.remove(key)
    return {key: round(value, 8) for key, value in sorted(weights.items())}


def build_cross_sleeve_alpha_map(
    series_by_sleeve: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    statistically_qualified_sleeves: Iterable[str],
    minimum_profitable_sleeves: int = 4,
    minimum_common_days: int = 30,
    maximum_pairwise_correlation: float = 0.5,
    maximum_single_sleeve_weight: float = 0.25,
) -> dict[str, Any]:
    qualified = {str(item) for item in statistically_qualified_sleeves if str(item)}
    daily = {
        str(sleeve): _daily_series(rows)
        for sleeve, rows in series_by_sleeve.items()
        if str(sleeve) in qualified
    }
    all_days = sorted({day for values in daily.values() for day in values})
    common_component: dict[str, float] = {}
    for day in all_days:
        values = [series[day] for series in daily.values() if day in series]
        if len(values) >= minimum_profitable_sleeves:
            common_component[day] = statistics.median(values)

    sleeves: list[dict[str, Any]] = []
    for sleeve, series in sorted(daily.items()):
        total_values = list(series.values())
        residual_values = [
            series[day] - common_component[day]
            for day in sorted(set(series).intersection(common_component))
        ]
        total_mean, total_lcb = _mean_lcb(total_values)
        residual_mean, residual_lcb = _mean_lcb(residual_values)
        independently_positive = bool(
            len(residual_values) >= minimum_common_days
            and total_lcb is not None
            and total_lcb > 0.0
            and residual_lcb is not None
            and residual_lcb > 0.0
        )
        sleeves.append(
            {
                "sleeve": sleeve,
                "day_count": len(total_values),
                "common_day_count": len(residual_values),
                "mean_total_alpha_bps": (
                    round(total_mean, 8) if total_mean is not None else None
                ),
                "total_alpha_lcb_95_bps": (
                    round(total_lcb, 8) if total_lcb is not None else None
                ),
                "mean_residual_alpha_bps": (
                    round(residual_mean, 8) if residual_mean is not None else None
                ),
                "residual_alpha_lcb_95_bps": (
                    round(residual_lcb, 8) if residual_lcb is not None else None
                ),
                "independently_positive": independently_positive,
            }
        )

    correlations: list[dict[str, Any]] = []
    names = sorted(daily)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            value, common_days = _correlation(daily[left], daily[right])
            correlations.append(
                {
                    "left": left,
                    "right": right,
                    "common_days": common_days,
                    "correlation": round(value, 8) if value is not None else None,
                }
            )

    ranked = sorted(
        (row for row in sleeves if row["independently_positive"]),
        key=lambda row: (
            float(row.get("residual_alpha_lcb_95_bps") or -math.inf),
            str(row.get("sleeve") or ""),
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    duplicate_blocks: list[dict[str, Any]] = []
    correlation_lookup = {
        frozenset((str(row["left"]), str(row["right"]))): row for row in correlations
    }
    for row in ranked:
        sleeve = str(row["sleeve"])
        conflicts: list[dict[str, Any]] = []
        for existing in selected:
            peer = str(existing["sleeve"])
            pair = correlation_lookup.get(frozenset((sleeve, peer)), {})
            value = _safe_float(pair.get("correlation"))
            common_days = int(pair.get("common_days", 0) or 0)
            if (
                value is not None
                and common_days >= minimum_common_days
                and value > maximum_pairwise_correlation
            ):
                conflicts.append(
                    {
                        "kept_owner": peer,
                        "blocked_peer": sleeve,
                        "correlation": round(value, 8),
                        "common_days": common_days,
                    }
                )
        if conflicts:
            duplicate_blocks.extend(conflicts)
        else:
            selected.append(row)

    evidence_ready = len(selected) >= minimum_profitable_sleeves
    scores = {
        str(row["sleeve"]): float(row["residual_alpha_lcb_95_bps"]) for row in selected
    }
    weights = (
        _capped_weights(scores, maximum_single_sleeve_weight) if evidence_ready else {}
    )
    blockers: list[str] = []
    if len(qualified) < minimum_profitable_sleeves:
        blockers.append("minimum_statistically_qualified_sleeves_pending")
    if len(common_component) < minimum_common_days:
        blockers.append("minimum_common_candidate_days_pending")
    if len(ranked) < minimum_profitable_sleeves:
        blockers.append("minimum_positive_residual_sleeves_pending")
    if len(selected) < minimum_profitable_sleeves:
        blockers.append("correlation_adjusted_sleeve_floor_pending")
    return {
        "schema_version": 1,
        "status": "evidence_ready" if evidence_ready else "collecting",
        "evidence_ready": evidence_ready,
        "qualified_sleeve_count": len(qualified),
        "common_component_day_count": len(common_component),
        "independently_positive_sleeve_count": len(ranked),
        "selected_sleeve_count": len(selected),
        "selected_sleeves": [str(row["sleeve"]) for row in selected],
        "research_weights": weights,
        "cash_weight": round(max(1.0 - sum(weights.values()), 0.0), 8),
        "sleeve_decomposition": sleeves,
        "pairwise_correlations": correlations,
        "duplicate_exposure_blocks": duplicate_blocks,
        "blockers": blockers,
        "shared_context_only": True,
        "shared_trade_logic_allowed": False,
        "automatic_allocation_allowed": False,
        "live_execution_authority": False,
    }
