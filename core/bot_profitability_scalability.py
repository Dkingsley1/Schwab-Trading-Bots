"""Evidence-bound bot ranking, lifecycle, capacity, and bounded activation planning.

This module is intentionally execution-free. It turns observed paper outcomes and
existing control artifacts into advice that runtime routers may inspect, but it
cannot change the registry, allocate capital, or submit an order.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import threading
import time
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Sequence

from core.master_grandmaster_evidence import build_observed_regime_context
from core.regime_taxonomy import evaluate_regime_compatibility


PROFITABILITY_CONTROL_IDS = tuple(f"p{index:02d}" for index in range(1, 9))
SCALABILITY_CONTROL_IDS = tuple(f"s{index:02d}" for index in range(1, 9))
ALL_CONTROL_IDS = PROFITABILITY_CONTROL_IDS + SCALABILITY_CONTROL_IDS


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    return min(max(_safe_float(value), low), high)


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


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ordered_unique(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _grade(ratio: float, *, structurally_ready: bool = True) -> str:
    if not structurally_ready:
        return "F"
    if ratio >= 0.98:
        return "A+"
    if ratio >= 0.9:
        return "A"
    if ratio >= 0.8:
        return "B"
    if ratio >= 0.7:
        return "C"
    if ratio >= 0.6:
        return "D"
    return "F"


def validate_policy(policy: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    profitability = _as_dict(policy.get("profitability"))
    scalability = _as_dict(policy.get("scalability"))
    safety = _as_dict(policy.get("safety_contract"))
    ranking = _as_dict(profitability.get("forward_ranking"))
    weights = _as_dict(ranking.get("weights"))
    activation = _as_dict(scalability.get("activation"))
    resources = _as_dict(scalability.get("resource_budgets"))
    cache = _as_dict(scalability.get("model_cache"))

    if _safe_int(policy.get("schema_version")) != 1:
        errors.append("profitability_scalability_schema_version_invalid")
    if str(policy.get("operating_mode") or "") != "evidence_bound_shadow_planning":
        errors.append("profitability_scalability_operating_mode_invalid")
    if abs(sum(_safe_float(value) for value in weights.values()) - 1.0) > 1e-9:
        errors.append("forward_ranking_weights_must_sum_to_one")
    if set(weights) != {
        "post_cost_expectancy",
        "conservative_lcb",
        "drawdown_control",
        "turnover_efficiency",
        "confidence",
        "persistence",
    }:
        errors.append("forward_ranking_weight_dimensions_invalid")
    for key in (
        "maximum_active_bots_total",
        "maximum_active_bots_per_sleeve",
        "maximum_active_bots_per_cell",
    ):
        if _safe_int(activation.get(key)) < 1:
            errors.append(f"activation_{key}_invalid")
    if _safe_int(activation.get("maximum_active_bots_per_cell")) > _safe_int(
        activation.get("maximum_active_bots_per_sleeve")
    ):
        errors.append("activation_cell_limit_exceeds_sleeve_limit")
    if _safe_int(activation.get("maximum_active_bots_per_sleeve")) > _safe_int(
        activation.get("maximum_active_bots_total")
    ):
        errors.append("activation_sleeve_limit_exceeds_total_limit")
    if _safe_int(resources.get("maximum_parallel_training_jobs")) != 1:
        errors.append("global_training_must_be_single_flight")
    if _safe_int(resources.get("maximum_parallel_training_jobs_per_sub_sleeve")) != 1:
        errors.append("sub_sleeve_training_must_be_single_flight")
    if _safe_int(cache.get("maximum_loaded_models")) < 1:
        errors.append("model_cache_count_limit_invalid")
    if _safe_int(cache.get("maximum_cache_mb")) < 1:
        errors.append("model_cache_memory_limit_invalid")
    if _safe_float(cache.get("inactive_ttl_seconds")) <= 0.0:
        errors.append("model_cache_ttl_invalid")
    for key in (
        "changes_runtime_decisions",
        "automatic_registry_mutation",
        "automatic_source_code_changes",
        "paper_execution_authority",
        "live_execution_authority",
        "automatic_live_promotion",
        "automatic_allocation",
        "profitability_guaranteed",
    ):
        if safety.get(key) is not False:
            errors.append(f"safety_{key}_must_be_false")
    return _ordered_unique(errors)


@dataclass
class _ModelCacheEntry:
    value: Any
    estimated_bytes: int
    loaded_monotonic: float
    last_access_monotonic: float
    access_count: int = 1


class LazyModelCache:
    """Thread-safe bounded lazy cache with deterministic pressure eviction."""

    def __init__(
        self,
        *,
        maximum_models: int = 8,
        maximum_bytes: int = 512 * 1024 * 1024,
        inactive_ttl_seconds: float = 900.0,
    ) -> None:
        if maximum_models < 1 or maximum_bytes < 1 or inactive_ttl_seconds <= 0:
            raise ValueError("model cache limits must be positive")
        self.maximum_models = int(maximum_models)
        self.maximum_bytes = int(maximum_bytes)
        self.inactive_ttl_seconds = float(inactive_ttl_seconds)
        self._entries: OrderedDict[str, _ModelCacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._load_count = 0
        self._eviction_count = 0

    def _total_bytes(self) -> int:
        return sum(entry.estimated_bytes for entry in self._entries.values())

    def _evict_key(self, key: str) -> None:
        if key in self._entries:
            self._entries.pop(key)
            self._eviction_count += 1

    def evict_inactive(self, *, now_monotonic: float | None = None) -> list[str]:
        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        evicted: list[str] = []
        with self._lock:
            for key, entry in list(self._entries.items()):
                if now - entry.last_access_monotonic < self.inactive_ttl_seconds:
                    continue
                self._evict_key(key)
                evicted.append(key)
        return evicted

    def evict_for_pressure(self, pressure: str) -> list[str]:
        level = str(pressure or "normal").strip().lower()
        evicted: list[str] = []
        with self._lock:
            if level in {"critical", "emergency"}:
                evicted = list(self._entries)
            elif level in {"high", "severe"}:
                evicted = list(self._entries)[: max(len(self._entries) - 1, 0)]
            for key in evicted:
                self._evict_key(key)
        return evicted

    def get(
        self,
        key: str,
        loader: Callable[[], Any],
        *,
        estimated_bytes: int = 0,
        memory_pressure: str = "normal",
        now_monotonic: float | None = None,
    ) -> Any:
        cache_key = str(key or "").strip()
        if not cache_key:
            raise ValueError("model cache key is required")
        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        pressure = str(memory_pressure or "normal").strip().lower()
        with self._lock:
            self.evict_inactive(now_monotonic=now)
            self.evict_for_pressure(pressure)
            entry = self._entries.get(cache_key)
            if entry is not None:
                entry.last_access_monotonic = now
                entry.access_count += 1
                self._entries.move_to_end(cache_key)
                return entry.value

        value = loader()
        size = max(int(estimated_bytes), 0)
        if pressure in {"critical", "emergency"} or size > self.maximum_bytes:
            return value
        with self._lock:
            self._load_count += 1
            self._entries[cache_key] = _ModelCacheEntry(value, size, now, now)
            self._entries.move_to_end(cache_key)
            while len(self._entries) > self.maximum_models or self._total_bytes() > self.maximum_bytes:
                oldest = next(iter(self._entries))
                self._evict_key(oldest)
        return value

    def clear(self) -> None:
        with self._lock:
            for key in list(self._entries):
                self._evict_key(key)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "loaded_model_count": len(self._entries),
                "estimated_bytes": self._total_bytes(),
                "maximum_models": self.maximum_models,
                "maximum_bytes": self.maximum_bytes,
                "inactive_ttl_seconds": self.inactive_ttl_seconds,
                "load_count": self._load_count,
                "eviction_count": self._eviction_count,
                "keys": list(self._entries),
            }


def _decision_identity(row: Mapping[str, Any]) -> str:
    metadata = _as_dict(row.get("metadata"))
    explicit = str(
        row.get("decision_id")
        or metadata.get("decision_id")
        or row.get("message_id")
        or ""
    ).strip()
    if explicit:
        return explicit
    return canonical_hash(
        {
            "timestamp_utc": row.get("timestamp_utc"),
            "strategy": row.get("strategy"),
            "symbol": row.get("symbol"),
            "action": row.get("action"),
            "quantity": row.get("quantity"),
        }
    )


def _regime_signature(row: Mapping[str, Any]) -> str:
    metadata = _as_dict(row.get("metadata"))
    context = _as_dict(metadata.get("regime_context"))
    axes = _as_dict(context.get("axes")) or context
    axis_parts = []
    for key in sorted(axes):
        raw = axes.get(key)
        values = raw if isinstance(raw, list) else [raw]
        cleaned = sorted(str(value or "").strip() for value in values if str(value or "").strip())
        if cleaned:
            axis_parts.append(f"{key}={'|'.join(cleaned)}")
    if axis_parts:
        return ";".join(axis_parts)
    explicit = str(
        metadata.get("post_entry_regime_bucket")
        or metadata.get("regime_state")
        or metadata.get("regime_label")
        or row.get("post_entry_regime_bucket")
        or row.get("regime_state")
        or ""
    ).strip()
    if explicit:
        return explicit
    return "/".join(
        part
        for part in (
            str(row.get("paper_profile") or metadata.get("source_profile") or "unknown_profile"),
            str(row.get("spread_regime") or "unknown_spread"),
            str(row.get("asset_class") or row.get("asset_type") or "unknown_asset"),
        )
        if part
    )


def extract_bot_observations(
    rows: Iterable[Mapping[str, Any]],
    *,
    known_bot_ids: set[str] | None = None,
    candidate_cutoff_utc: str = "",
) -> dict[str, Any]:
    cutoff = _parse_timestamp(candidate_cutoff_utc)
    known = known_bot_ids or set()
    observations: list[dict[str, Any]] = []
    seen_decisions: set[str] = set()
    duplicate_rows = 0
    unattributed_rows = 0
    unknown_constituents = Counter()
    malformed_rows = 0

    for row in rows:
        timestamp = _parse_timestamp(row.get("timestamp_utc"))
        if timestamp is None:
            malformed_rows += 1
            continue
        decision_id = _decision_identity(row)
        if decision_id in seen_decisions:
            duplicate_rows += 1
            continue
        seen_decisions.add(decision_id)
        metadata = _as_dict(row.get("metadata"))
        constituents = [
            item
            for item in _as_list(metadata.get("constituent_attribution"))
            if isinstance(item, dict) and str(item.get("bot_id") or "").strip()
        ]
        direct_bot_id = str(row.get("bot_id") or metadata.get("bot_id") or "").strip()
        if not constituents and direct_bot_id:
            constituents = [{"bot_id": direct_bot_id, "weight_share": 1.0}]
        if not constituents:
            unattributed_rows += 1
            continue
        positive_weights = [max(_safe_float(item.get("weight_share"), 1.0), 0.0) for item in constituents]
        weight_total = sum(positive_weights) or float(len(constituents))
        for item, raw_weight in zip(constituents, positive_weights):
            bot_id = str(item.get("bot_id") or "").strip()
            if known and bot_id not in known:
                unknown_constituents[bot_id] += 1
                continue
            weight = (raw_weight if sum(positive_weights) > 0.0 else 1.0) / weight_total
            raw_score = _safe_float(item.get("confidence"), math.nan)
            if not math.isfinite(raw_score):
                score = _safe_float(item.get("score"), 0.0)
                raw_score = abs(score - 0.5) * 2.0 if 0.0 <= score <= 1.0 else abs(score)
            observations.append(
                {
                    "bot_id": bot_id,
                    "decision_id": decision_id,
                    "timestamp_utc": timestamp.isoformat(),
                    "day_utc": timestamp.date().isoformat(),
                    "candidate_bound": bool(cutoff is not None and timestamp >= cutoff),
                    "profile": str(row.get("paper_profile") or metadata.get("source_profile") or ""),
                    "strategy": str(row.get("paper_strategy") or row.get("strategy") or ""),
                    "regime": _regime_signature(row),
                    "weight_share": round(weight, 10),
                    "post_cost_pnl": _safe_float(row.get("post_cost_pnl_delta")) * weight,
                    "post_cost_return_bps": _safe_float(row.get("post_cost_return_bps")) * weight,
                    "notional": abs(_safe_float(row.get("execution_notional"))) * weight,
                    "confidence": _clamp(raw_score),
                    "slippage_bps": abs(_safe_float(row.get("realized_slippage_bps"))),
                    "spread_bps": abs(_safe_float(row.get("model_spread_bps"))),
                    "latency_ms": max(_safe_float(row.get("model_latency_ms")), 0.0),
                    "partial_fill_ratio": _clamp(
                        row.get("expected_partial_fill_ratio", 1.0)
                    ),
                    "tradeability": _clamp(row.get("tradeability_score")),
                    "source": str(row.get("paper_fill_source") or "unknown"),
                }
            )
    return {
        "observations": observations,
        "scan": {
            "unique_decision_count": len(seen_decisions),
            "observation_count": len(observations),
            "duplicate_row_count": duplicate_rows,
            "unattributed_row_count": unattributed_rows,
            "malformed_row_count": malformed_rows,
            "unknown_constituent_count": sum(unknown_constituents.values()),
            "unknown_constituents": dict(unknown_constituents.most_common(25)),
            "candidate_cutoff_utc": cutoff.isoformat() if cutoff is not None else "",
            "candidate_binding_available": cutoff is not None,
        },
    }


def _maximum_drawdown(values: Sequence[float]) -> float:
    cumulative = 0.0
    peak = 0.0
    drawdown = 0.0
    for value in values:
        cumulative += value
        peak = max(peak, cumulative)
        drawdown = max(drawdown, peak - cumulative)
    return drawdown


def _metric_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: str(row.get("timestamp_utc") or ""))
    pnl = [_safe_float(row.get("post_cost_pnl")) for row in ordered]
    returns = [_safe_float(row.get("post_cost_return_bps")) for row in ordered]
    days = sorted({str(row.get("day_utc") or "") for row in ordered if row.get("day_utc")})
    regimes = sorted({str(row.get("regime") or "") for row in ordered if row.get("regime")})
    daily_pnl: dict[str, float] = defaultdict(float)
    regime_pnl: dict[str, list[float]] = defaultdict(list)
    for row in ordered:
        daily_pnl[str(row.get("day_utc") or "")] += _safe_float(row.get("post_cost_pnl"))
        regime_pnl[str(row.get("regime") or "")].append(_safe_float(row.get("post_cost_pnl")))
    effective_samples = min(len(returns), max(len(days) * 3, 1))
    mean_return = statistics.fmean(returns) if returns else 0.0
    return_std = statistics.stdev(returns) if len(returns) >= 2 else None
    return_lcb = (
        mean_return - 1.96 * return_std / math.sqrt(effective_samples)
        if return_std is not None and effective_samples >= 2
        else None
    )
    positive_days = sum(1 for value in daily_pnl.values() if value > 0.0)
    profitable_regimes = sum(1 for values in regime_pnl.values() if sum(values) > 0.0)
    notional = sum(_safe_float(row.get("notional")) for row in ordered)
    total_pnl = sum(pnl)
    return {
        "sample_count": len(ordered),
        "effective_sample_count": effective_samples,
        "independent_day_count": len(days),
        "regime_count": len(regimes),
        "total_post_cost_pnl": round(total_pnl, 10),
        "mean_post_cost_pnl": round(statistics.fmean(pnl), 10) if pnl else 0.0,
        "mean_post_cost_return_bps": round(mean_return, 8),
        "post_cost_return_lcb_bps": round(return_lcb, 8) if return_lcb is not None else None,
        "maximum_drawdown": round(_maximum_drawdown(pnl), 10),
        "turnover_notional": round(notional, 8),
        "turnover_efficiency_bps": round(total_pnl / notional * 10000.0, 8) if notional > 0 else 0.0,
        "mean_confidence": round(
            statistics.fmean(_safe_float(row.get("confidence")) for row in ordered), 8
        )
        if ordered
        else 0.0,
        "positive_day_count": positive_days,
        "positive_day_ratio": round(positive_days / max(len(days), 1), 8),
        "profitable_regime_count": profitable_regimes,
        "days": days,
        "regimes": regimes,
        "daily_post_cost_pnl": dict(sorted(daily_pnl.items())),
    }


def _correlation(left: Mapping[str, float], right: Mapping[str, float], minimum_days: int) -> tuple[float | None, int]:
    common = sorted(set(left).intersection(right))
    if len(common) < minimum_days:
        return None, len(common)
    left_values = [_safe_float(left[day]) for day in common]
    right_values = [_safe_float(right[day]) for day in common]
    left_mean = statistics.fmean(left_values)
    right_mean = statistics.fmean(right_values)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left_values, right_values))
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left_values))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right_values))
    if left_scale <= 1e-12 or right_scale <= 1e-12:
        return None, len(common)
    return numerator / (left_scale * right_scale), len(common)


def _minmax(values: Mapping[str, float], key: str) -> dict[str, float]:
    finite = [value for value in values.values() if math.isfinite(value)]
    if not finite:
        return {bot_id: 0.0 for bot_id in values}
    low = min(finite)
    high = max(finite)
    if abs(high - low) <= 1e-12:
        return {bot_id: 0.5 for bot_id in values}
    return {bot_id: _clamp((value - low) / (high - low)) for bot_id, value in values.items()}


def build_bot_profiles(
    assignments: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
    *,
    independent_execution_ready: bool,
) -> list[dict[str, Any]]:
    profitability = _as_dict(policy.get("profitability"))
    learning = _as_dict(profitability.get("regime_learning"))
    ranking_policy = _as_dict(profitability.get("forward_ranking"))
    marginal_policy = _as_dict(profitability.get("marginal_contribution"))
    persistence_policy = _as_dict(profitability.get("persistence"))
    lifecycle_policy = _as_dict(profitability.get("lifecycle"))
    capacity_policy = _as_dict(profitability.get("capacity"))
    assignments_by_id = {
        str(row.get("bot_id") or ""): dict(row)
        for row in assignments
        if str(row.get("bot_id") or "")
    }
    rows_by_bot: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in observations:
        rows_by_bot[str(row.get("bot_id") or "")].append(row)

    profiles: list[dict[str, Any]] = []
    for bot_id, assignment in sorted(assignments_by_id.items()):
        all_rows = rows_by_bot.get(bot_id, [])
        candidate_rows = [row for row in all_rows if bool(row.get("candidate_bound", False))]
        historical = _metric_summary(all_rows)
        candidate = _metric_summary(candidate_rows)
        regime_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in candidate_rows:
            regime_rows[str(row.get("regime") or "unknown")].append(row)
        regime_outcomes = []
        for regime, rows in sorted(regime_rows.items()):
            metrics = _metric_summary(rows)
            regime_outcomes.append(
                {
                    "regime": regime,
                    **{
                        key: metrics[key]
                        for key in (
                            "sample_count",
                            "independent_day_count",
                            "total_post_cost_pnl",
                            "mean_post_cost_return_bps",
                            "post_cost_return_lcb_bps",
                        )
                    },
                }
            )
        regime_outcomes.sort(
            key=lambda row: (
                -_safe_float(row.get("post_cost_return_lcb_bps"), -1e12),
                -_safe_int(row.get("sample_count")),
                str(row.get("regime") or ""),
            )
        )
        preferred = [
            str(row.get("regime") or "")
            for row in regime_outcomes
            if _safe_int(row.get("sample_count")) >= _safe_int(learning.get("minimum_regime_samples"), 10)
            and row.get("post_cost_return_lcb_bps") is not None
            and _safe_float(row.get("post_cost_return_lcb_bps")) > 0.0
        ][:_safe_int(learning.get("maximum_preferred_regimes"), 4)]
        rank_evidence_ready = bool(
            candidate["sample_count"] >= _safe_int(ranking_policy.get("minimum_bot_samples"), 30)
            and candidate["independent_day_count"] >= _safe_int(ranking_policy.get("minimum_independent_days"), 3)
            and candidate["post_cost_return_lcb_bps"] is not None
        )
        persistence_ready = bool(
            rank_evidence_ready
            and candidate["positive_day_count"] >= _safe_int(persistence_policy.get("minimum_positive_days"), 3)
            and candidate["positive_day_ratio"] >= _safe_float(persistence_policy.get("minimum_positive_day_ratio"), 0.6)
            and candidate["profitable_regime_count"] >= _safe_int(persistence_policy.get("minimum_profitable_regimes"), 2)
            and _safe_float(candidate.get("post_cost_return_lcb_bps"), -1e12) > 0.0
        )
        profiles.append(
            {
                "bot_id": bot_id,
                "sleeve_id": str(assignment.get("sleeve_id") or ""),
                "sub_sleeve_id": str(assignment.get("sub_sleeve_id") or ""),
                "cell_id": str(assignment.get("cell_id") or ""),
                "correlation_cluster_id": str(assignment.get("correlation_cluster_id") or ""),
                "shadow_vote_eligible": bool(assignment.get("shadow_vote_eligible", False)),
                "regime_profile_id": str(assignment.get("regime_profile_id") or ""),
                "regime_profile": dict(_as_dict(assignment.get("regime_profile"))),
                "historical_diagnostics": historical,
                "candidate_evidence": candidate,
                "regime_outcomes": regime_outcomes,
                "learned_preferred_regimes": preferred,
                "regime_preference_evidence_ready": bool(
                    candidate["sample_count"] >= _safe_int(learning.get("minimum_bot_samples"), 30)
                    and candidate["independent_day_count"] >= _safe_int(learning.get("minimum_independent_days"), 3)
                    and bool(preferred)
                ),
                "rank_evidence_ready": rank_evidence_ready,
                "persistence_ready": persistence_ready,
            }
        )

    profile_by_id = {row["bot_id"]: row for row in profiles}
    clusters: dict[str, list[str]] = defaultdict(list)
    for row in profiles:
        clusters[row["correlation_cluster_id"]].append(row["bot_id"])
    minimum_common_days = _safe_int(marginal_policy.get("minimum_common_days"), 5)
    max_correlation = _safe_float(marginal_policy.get("maximum_peer_correlation"), 0.85)
    for members in clusters.values():
        for bot_id in members:
            current = profile_by_id[bot_id]
            daily = _as_dict(_as_dict(current.get("candidate_evidence")).get("daily_post_cost_pnl"))
            peer_rows = []
            for peer_id in members:
                if peer_id == bot_id:
                    continue
                peer_daily = _as_dict(
                    _as_dict(profile_by_id[peer_id].get("candidate_evidence")).get("daily_post_cost_pnl")
                )
                correlation, common_days = _correlation(daily, peer_daily, minimum_common_days)
                peer_rows.append(
                    {
                        "peer_bot_id": peer_id,
                        "correlation": round(correlation, 8) if correlation is not None else None,
                        "common_day_count": common_days,
                    }
                )
            measured = [
                _safe_float(row.get("correlation"))
                for row in peer_rows
                if row.get("correlation") is not None
            ]
            max_peer = max(measured) if measured else (0.0 if len(members) == 1 and len(daily) >= minimum_common_days else None)
            marginal_ready = bool(current["rank_evidence_ready"] and max_peer is not None)
            current["marginal_contribution"] = {
                "evidence_ready": marginal_ready,
                "maximum_peer_correlation": round(max_peer, 8) if max_peer is not None else None,
                "duplicate_cluster": bool(max_peer is not None and max_peer > max_correlation),
                "marginal_score": round(max(0.0, 1.0 - max(max_peer or 0.0, 0.0)), 8)
                if marginal_ready
                else 0.0,
                "peer_count": len(peer_rows),
                "measured_peer_count": len(measured),
                "peers": sorted(
                    peer_rows,
                    key=lambda row: (
                        row.get("correlation") is None,
                        -_safe_float(row.get("correlation")),
                        str(row.get("peer_bot_id") or ""),
                    ),
                )[:12],
            }

    eligible = [row for row in profiles if bool(row.get("rank_evidence_ready"))]
    expectancy_norm = _minmax(
        {row["bot_id"]: _safe_float(_as_dict(row["candidate_evidence"]).get("mean_post_cost_return_bps")) for row in eligible},
        "expectancy",
    )
    lcb_norm = _minmax(
        {row["bot_id"]: _safe_float(_as_dict(row["candidate_evidence"]).get("post_cost_return_lcb_bps"), -1e12) for row in eligible},
        "lcb",
    )
    turnover_norm = _minmax(
        {row["bot_id"]: _safe_float(_as_dict(row["candidate_evidence"]).get("turnover_efficiency_bps")) for row in eligible},
        "turnover",
    )
    weights = _as_dict(ranking_policy.get("weights"))
    for row in profiles:
        bot_id = row["bot_id"]
        metrics = _as_dict(row.get("candidate_evidence"))
        notional = max(_safe_float(metrics.get("turnover_notional")), 1e-9)
        drawdown_ratio = _safe_float(metrics.get("maximum_drawdown")) / notional
        components = {
            "post_cost_expectancy": expectancy_norm.get(bot_id, 0.0),
            "conservative_lcb": lcb_norm.get(bot_id, 0.0),
            "drawdown_control": 1.0 / (1.0 + drawdown_ratio * 100.0),
            "turnover_efficiency": turnover_norm.get(bot_id, 0.0),
            "confidence": _clamp(metrics.get("mean_confidence")),
            "persistence": 1.0 if row.get("persistence_ready") else 0.0,
        }
        score = sum(_safe_float(weights.get(key)) * value for key, value in components.items())
        marginal = _as_dict(row.get("marginal_contribution"))
        if bool(marginal.get("duplicate_cluster", False)):
            score *= _safe_float(marginal_policy.get("maximum_duplicate_weight"), 0.02)
        if not row.get("rank_evidence_ready"):
            score = 0.0
        row["forward_rank"] = {
            "score": round(_clamp(score), 8),
            "components": {key: round(value, 8) for key, value in components.items()},
            "candidate_bound": True,
            "evidence_ready": bool(row.get("rank_evidence_ready")),
        }

    ranked = sorted(
        profiles,
        key=lambda row: (
            not bool(row.get("rank_evidence_ready")),
            -_safe_float(_as_dict(row.get("forward_rank")).get("score")),
            str(row.get("bot_id") or ""),
        ),
    )
    for rank, row in enumerate(ranked, 1):
        row["forward_rank"]["global_rank"] = rank if row.get("rank_evidence_ready") else None

    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in profiles:
        by_cell[row["cell_id"]].append(row)
    for rows in by_cell.values():
        rows.sort(key=lambda row: (-_safe_float(_as_dict(row.get("forward_rank")).get("score")), row["bot_id"]))
        for rank, row in enumerate(rows, 1):
            row["forward_rank"]["cell_rank"] = rank if row.get("rank_evidence_ready") else None

    for row in profiles:
        metrics = _as_dict(row.get("candidate_evidence"))
        mean_lcb = metrics.get("post_cost_return_lcb_bps")
        positive_ratio = _safe_float(metrics.get("positive_day_ratio"))
        days = _safe_int(metrics.get("independent_day_count"))
        lifecycle_state = "probation_collecting"
        reason = "candidate_bound_sample_floor_pending"
        if (
            mean_lcb is not None
            and _safe_float(mean_lcb) < 0.0
            and days >= _safe_int(lifecycle_policy.get("retirement_minimum_days"), 10)
            and positive_ratio <= _safe_float(lifecycle_policy.get("retirement_maximum_positive_day_ratio"), 0.35)
        ):
            lifecycle_state = "retirement_review"
            reason = "persistent_negative_candidate_bound_post_cost_evidence"
        elif mean_lcb is not None and _safe_float(mean_lcb) < 0.0 and days >= _safe_int(
            lifecycle_policy.get("demotion_minimum_days"), 5
        ):
            lifecycle_state = "demotion_review"
            reason = "negative_candidate_bound_post_cost_lcb"
        elif row.get("persistence_ready") and _safe_int(_as_dict(row.get("forward_rank")).get("cell_rank")) == 1:
            lifecycle_state = "champion_candidate"
            reason = "top_persistent_candidate_bound_cell_rank"
        elif row.get("rank_evidence_ready"):
            lifecycle_state = "challenger"
            reason = "candidate_bound_rank_ready"
        row["lifecycle_advice"] = {
            "state": lifecycle_state,
            "reason": reason,
            "automatic_registry_mutation": False,
            "human_review_required": lifecycle_state in {"retirement_review", "demotion_review", "champion_candidate"},
        }

        source_rows = [item for item in rows_by_bot.get(row["bot_id"], []) if item.get("candidate_bound")]
        notionals = sorted(_safe_float(item.get("notional")) for item in source_rows if _safe_float(item.get("notional")) > 0)
        base_notional = statistics.median(notionals) if notionals else 0.0
        mean_slippage = statistics.fmean(_safe_float(item.get("slippage_bps")) for item in source_rows) if source_rows else 0.0
        mean_latency = statistics.fmean(_safe_float(item.get("latency_ms")) for item in source_rows) if source_rows else 0.0
        mean_partial = statistics.fmean(_safe_float(item.get("partial_fill_ratio"), 1.0) for item in source_rows) if source_rows else 0.0
        curve = []
        max_supported = 0.0
        capacity_evidence_ready = bool(
            independent_execution_ready
            and row.get("rank_evidence_ready")
            and len(source_rows) >= _safe_int(capacity_policy.get("minimum_capacity_samples"), 30)
            and mean_lcb is not None
            and base_notional > 0.0
        )
        for multiplier in _as_list(capacity_policy.get("notional_multipliers")):
            scale = max(_safe_float(multiplier), 0.0)
            impact = mean_slippage * max(math.sqrt(max(scale, 1.0)) - 1.0, 0.0)
            latency_penalty = min(mean_latency / 1000.0, 10.0) * 0.5
            partial_penalty = max(1.0 - mean_partial, 0.0) * 20.0
            stressed_cost = mean_slippage + impact + latency_penalty + partial_penalty
            projected_lcb = _safe_float(mean_lcb, -1e12) - impact - latency_penalty - partial_penalty
            supported = bool(
                capacity_evidence_ready
                and projected_lcb > 0.0
                and stressed_cost <= _safe_float(capacity_policy.get("maximum_stressed_slippage_bps"), 100.0)
                and scale <= _safe_float(capacity_policy.get("maximum_observed_notional_multiplier"), 4.0)
            )
            notional_value = base_notional * scale
            if supported:
                max_supported = max(max_supported, notional_value)
            curve.append(
                {
                    "notional_multiplier": scale,
                    "notional": round(notional_value, 8),
                    "stressed_execution_cost_bps": round(stressed_cost, 8),
                    "projected_post_cost_lcb_bps": round(projected_lcb, 8),
                    "supported": supported,
                }
            )
        row["capacity_curve"] = {
            "evidence_ready": capacity_evidence_ready,
            "independent_execution_calibration_ready": independent_execution_ready,
            "base_observed_notional": round(base_notional, 8),
            "maximum_supported_notional": round(max_supported, 8),
            "curve": curve,
            "account_size_alone_may_increase_weight": False,
        }
    return sorted(profiles, key=lambda row: str(row.get("bot_id") or ""))


def _existing_control(artifact: Mapping[str, Any], control_id: str) -> dict[str, Any]:
    for key in ("baseline_controls", "controls"):
        for row in _as_list(artifact.get(key)):
            if isinstance(row, dict) and str(row.get("control_id") or "") == control_id:
                return row
    return {}


def _control(
    control_id: str,
    title: str,
    *,
    evidence_ready: bool,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "control_id": control_id,
        "title": title,
        "implemented": True,
        "implementation_status": "ready",
        "evidence_ready": bool(evidence_ready),
        "evidence_status": "ready" if evidence_ready else "collecting",
        "evidence": dict(evidence),
    }


def _select_activation_plan(
    profiles: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
    *,
    current_regime_context: Mapping[str, Any],
    regime_model: Mapping[str, Any],
) -> dict[str, Any]:
    activation = _as_dict(_as_dict(policy.get("scalability")).get("activation"))
    total_limit = _safe_int(activation.get("maximum_active_bots_total"), 400)
    sleeve_limit = _safe_int(activation.get("maximum_active_bots_per_sleeve"), 48)
    cell_limit = _safe_int(activation.get("maximum_active_bots_per_cell"), 8)
    require_regime = bool(activation.get("require_current_regime_compatibility", True))
    regime_compatible_count = 0
    for row in profiles:
        if not isinstance(row, dict):
            continue
        profile = _as_dict(row.get("regime_profile"))
        if not require_regime:
            compatibility = {
                "compatible": True,
                "reason": "current_regime_compatibility_not_required",
                "score": 1.0,
                "compared_axis_count": 0,
            }
        elif not profile or not regime_model or not current_regime_context:
            compatibility = {
                "compatible": False,
                "reason": "regime_compatibility_evidence_missing",
                "score": 0.0,
                "compared_axis_count": 0,
            }
        else:
            compatibility = evaluate_regime_compatibility(
                profile,
                current_regime_context,
                regime_model,
            )
        row["current_regime_compatibility"] = compatibility
        if bool(compatibility.get("compatible", False)):
            regime_compatible_count += 1

    candidates = [
        dict(row)
        for row in profiles
        if bool(row.get("shadow_vote_eligible", False))
        and bool(row.get("rank_evidence_ready", False))
        and bool(row.get("persistence_ready", False))
        and bool(_as_dict(row.get("marginal_contribution")).get("evidence_ready", False))
        and not bool(_as_dict(row.get("marginal_contribution")).get("duplicate_cluster", False))
        and bool(_as_dict(row.get("capacity_curve")).get("evidence_ready", False))
        and bool(_as_dict(row.get("current_regime_compatibility")).get("compatible", False))
    ]
    candidates.sort(
        key=lambda row: (
            -_safe_float(_as_dict(row.get("forward_rank")).get("score")),
            str(row.get("bot_id") or ""),
        )
    )
    selected: list[dict[str, Any]] = []
    parked: list[dict[str, Any]] = []
    sleeve_counts = Counter()
    cell_counts = Counter()
    for row in candidates:
        sleeve = str(row.get("sleeve_id") or "")
        cell = str(row.get("cell_id") or "")
        reason = ""
        if len(selected) >= total_limit:
            reason = "global_activation_budget_reached"
        elif sleeve_counts[sleeve] >= sleeve_limit:
            reason = "sleeve_activation_budget_reached"
        elif cell_counts[cell] >= cell_limit:
            reason = "cell_activation_budget_reached"
        target = {
            "bot_id": str(row.get("bot_id") or ""),
            "sleeve_id": sleeve,
            "cell_id": cell,
            "rank_score": _safe_float(_as_dict(row.get("forward_rank")).get("score")),
            "maximum_supported_notional": _safe_float(
                _as_dict(row.get("capacity_curve")).get("maximum_supported_notional")
            ),
            "regime_compatibility_score": _safe_float(
                _as_dict(row.get("current_regime_compatibility")).get("score")
            ),
            "regime_compatibility_reason": str(
                _as_dict(row.get("current_regime_compatibility")).get("reason") or ""
            ),
            "reason": reason or "top_k_candidate",
            "paper_execution_authority": False,
            "live_execution_authority": False,
        }
        if reason:
            parked.append(target)
            continue
        selected.append(target)
        sleeve_counts[sleeve] += 1
        cell_counts[cell] += 1
    evidence_pending = sum(1 for row in profiles if row.get("shadow_vote_eligible")) - len(candidates)
    return {
        "selection_mode": str(activation.get("selection_mode") or "ranked_top_k_then_park"),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "parked_by_budget_count": len(parked),
        "evidence_pending_count": max(evidence_pending, 0),
        "regime_compatible_bot_count": regime_compatible_count,
        "current_regime_context": dict(current_regime_context),
        "limits": {
            "total": total_limit,
            "per_sleeve": sleeve_limit,
            "per_cell": cell_limit,
        },
        "selected": selected,
        "parked_by_budget": parked,
        "selection_receipt_sha256": canonical_hash(selected),
        "application_allowed": False,
        "paper_execution_authority": False,
        "live_execution_authority": False,
    }


def build_control_payload(
    policy: Mapping[str, Any],
    assignments: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
    artifacts: Mapping[str, Mapping[str, Any]],
    runtime_evidence: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    policy_errors = validate_policy(policy)
    firewall = _as_dict(artifacts.get("profitability_firewall"))
    calibration = _as_dict(artifacts.get("paper_execution_calibration"))
    feature_store = _as_dict(artifacts.get("feature_store"))
    runtime_throttle = _as_dict(artifacts.get("runtime_throttle"))
    resource_governor = _as_dict(artifacts.get("resource_governor"))
    cold_archive = _as_dict(artifacts.get("cold_archive"))
    sleeve_ingestion = _as_dict(artifacts.get("sleeve_ingestion"))
    regime_payload = _as_dict(artifacts.get("regime_context"))
    regime_model = _as_dict(_as_dict(artifacts.get("bot_organization_policy")).get("regime_model"))
    current_regime_context = (
        build_observed_regime_context(regime_payload, regime_model)
        if regime_payload and regime_model
        else {}
    )
    execution_policy = _as_dict(_as_dict(policy.get("profitability")).get("execution_realism"))
    independent_samples = _safe_int(calibration.get("independent_samples"))
    independent_execution_ready = bool(
        calibration.get("independent_evidence_ready", False)
        and independent_samples >= _safe_int(execution_policy.get("minimum_independent_fills"), 30)
    )
    profiles = build_bot_profiles(
        assignments,
        observations,
        policy,
        independent_execution_ready=independent_execution_ready,
    )
    activation_plan = _select_activation_plan(
        profiles,
        policy,
        current_regime_context=current_regime_context,
        regime_model=regime_model,
    )
    candidate_profiles = [row for row in profiles if _safe_int(_as_dict(row.get("candidate_evidence")).get("sample_count")) > 0]
    learned_profiles = [row for row in profiles if row.get("regime_preference_evidence_ready")]
    ranked_profiles = [row for row in profiles if row.get("rank_evidence_ready")]
    marginal_profiles = [row for row in profiles if _as_dict(row.get("marginal_contribution")).get("evidence_ready")]
    persistence_profiles = [row for row in profiles if row.get("persistence_ready")]
    capacity_profiles = [row for row in profiles if _as_dict(row.get("capacity_curve")).get("evidence_ready")]
    retirement_reviews = [row["bot_id"] for row in profiles if _as_dict(row.get("lifecycle_advice")).get("state") == "retirement_review"]
    demotion_reviews = [row["bot_id"] for row in profiles if _as_dict(row.get("lifecycle_advice")).get("state") == "demotion_review"]
    champion_candidates = [row["bot_id"] for row in profiles if _as_dict(row.get("lifecycle_advice")).get("state") == "champion_candidate"]

    holdout = _existing_control(firewall, "h04_locked_holdout_vault")
    multiple_testing = _existing_control(firewall, "08_multiple_testing_firewall")
    oos_regime = _existing_control(firewall, "09_oos_regime_lcb")
    fill_truth = _existing_control(firewall, "h01_independent_fill_truth")
    execution_stress = _existing_control(firewall, "h05_adversarial_execution_replay")
    statistical_ready = all(
        bool(row.get("evidence_ready", False))
        for row in (holdout, multiple_testing, oos_regime)
    )
    profitability_controls = [
        _control(
            "p01",
            "Learn regime preferences from candidate-bound observed outcomes",
            evidence_ready=bool(learned_profiles),
            evidence={
                "evaluated_bot_count": len(profiles),
                "candidate_observed_bot_count": len(candidate_profiles),
                "learned_preference_bot_count": len(learned_profiles),
                "manual_assignment_used_as_profitability_evidence": False,
            },
        ),
        _control(
            "p02",
            "Rank bots on forward post-cost expectancy, drawdown, turnover, and confidence",
            evidence_ready=bool(ranked_profiles),
            evidence={
                "ranked_bot_count": len(ranked_profiles),
                "candidate_bound": True,
                "ranking_dimensions": list(_as_dict(_as_dict(_as_dict(policy.get("profitability")).get("forward_ranking")).get("weights"))),
            },
        ),
        _control(
            "p03",
            "Measure marginal portfolio contribution and suppress correlated duplicates",
            evidence_ready=bool(marginal_profiles),
            evidence={
                "marginal_evidence_bot_count": len(marginal_profiles),
                "duplicate_cluster_bot_count": sum(
                    1 for row in profiles if _as_dict(row.get("marginal_contribution")).get("duplicate_cluster")
                ),
                "unknown_correlation_fails_closed": True,
            },
        ),
        _control(
            "p04",
            "Calibrate spread, slippage, latency, and partial-fill realism independently",
            evidence_ready=bool(independent_execution_ready and fill_truth.get("evidence_ready") and execution_stress.get("evidence_ready")),
            evidence={
                "independent_fill_samples": independent_samples,
                "independent_calibration_ready": independent_execution_ready,
                "fill_truth_ready": bool(fill_truth.get("evidence_ready", False)),
                "execution_stress_ready": bool(execution_stress.get("evidence_ready", False)),
                "dimensions": _as_list(execution_policy.get("required_dimensions")),
            },
        ),
        _control(
            "p05",
            "Enforce locked holdouts, walk-forward testing, and multiple-testing control",
            evidence_ready=statistical_ready,
            evidence={
                "locked_holdout_ready": bool(holdout.get("evidence_ready", False)),
                "multiple_testing_ready": bool(multiple_testing.get("evidence_ready", False)),
                "walk_forward_regime_lcb_ready": bool(oos_regime.get("evidence_ready", False)),
            },
        ),
        _control(
            "p06",
            "Drive probation, demotion, retirement, and champion-challenger advice",
            evidence_ready=bool(candidate_profiles),
            evidence={
                "evaluated_candidate_bot_count": len(candidate_profiles),
                "champion_candidate_count": len(champion_candidates),
                "demotion_review_count": len(demotion_reviews),
                "retirement_review_count": len(retirement_reviews),
                "automatic_registry_mutation": False,
            },
        ),
        _control(
            "p07",
            "Require positive post-cost persistence across days and regimes",
            evidence_ready=bool(persistence_profiles),
            evidence={
                "persistent_bot_count": len(persistence_profiles),
                "candidate_observed_bot_count": len(candidate_profiles),
                "positive_lcb_required": True,
            },
        ),
        _control(
            "p08",
            "Publish liquidity-aware capital-capacity curves",
            evidence_ready=bool(capacity_profiles),
            evidence={
                "capacity_ready_bot_count": len(capacity_profiles),
                "independent_execution_calibration_ready": independent_execution_ready,
                "unknown_capacity_weight": 0.0,
            },
        ),
    ]

    catalog_process = _as_dict(_as_dict(policy.get("scalability")).get("catalog_process_separation"))
    runtime_process_count = _safe_int(runtime_evidence.get("runtime_loop_process_count"))
    catalog_count = len(assignments)
    process_separation_ready = bool(
        runtime_process_count <= _safe_int(catalog_process.get("maximum_runtime_loop_processes"), 96)
        and (catalog_count > runtime_process_count or not catalog_count)
    )
    feature_hashes = _as_dict(feature_store.get("contract_hashes"))
    point_in_time = _as_dict(feature_store.get("point_in_time_contract"))
    shared_feature_ready = bool(
        feature_store.get("ok", False)
        and feature_hashes.get("dataset_manifest_sha256")
        and point_in_time.get("seed_ready", False)
    )
    resource_ready = bool(runtime_throttle.get("ok", False) and resource_governor.get("ok", False))
    checkpoint_ready = bool(
        _safe_int(runtime_evidence.get("runtime_checkpoint_count")) > 0
        and _safe_int(runtime_evidence.get("order_idempotency_registry_count")) > 0
        and _safe_float(runtime_evidence.get("decision_identity_coverage_ratio")) >= 1.0
    )
    data_tier = _as_dict(sleeve_ingestion.get("data_tier_contract"))
    raw_reader_commands = cold_archive.get("reader_commands")
    reader_command_count = (
        len(raw_reader_commands)
        if isinstance(raw_reader_commands, (dict, list))
        else 0
    )
    storage_ready = bool(
        cold_archive.get("ok", False)
        and cold_archive.get("archive_root")
        and cold_archive.get("manifest_path")
        and reader_command_count > 0
        and data_tier.get("core_priority")
        and data_tier.get("cold_budget")
    )
    cache_policy = _as_dict(_as_dict(policy.get("scalability")).get("model_cache"))
    model_cache_ready = bool(
        cache_policy.get("lazy_loading_required") is True
        and cache_policy.get("evict_all_under_critical_memory_pressure") is True
        and _safe_int(cache_policy.get("maximum_loaded_models")) > 0
        and _safe_int(cache_policy.get("maximum_cache_mb")) > 0
    )
    capacity_allocation_ready = bool(
        capacity_profiles
        and activation_plan["selected_count"] > 0
        and _as_dict(_as_dict(policy.get("scalability")).get("capacity_aware_allocation")).get(
            "automatic_allocation"
        )
        is False
    )
    scalability_controls = [
        _control(
            "s01",
            "Keep the registered bot catalog separate from bounded runtime processes",
            evidence_ready=process_separation_ready,
            evidence={
                "catalog_bot_count": catalog_count,
                "runtime_loop_process_count": runtime_process_count,
                "maximum_runtime_loop_processes": _safe_int(catalog_process.get("maximum_runtime_loop_processes"), 96),
            },
        ),
        _control(
            "s02",
            "Select only top eligible bots by sleeve, cell, current regime, and evidence",
            evidence_ready=activation_plan["selected_count"] > 0,
            evidence={
                "candidate_count": activation_plan["candidate_count"],
                "selected_count": activation_plan["selected_count"],
                "parked_by_budget_count": activation_plan["parked_by_budget_count"],
                "evidence_pending_count": activation_plan["evidence_pending_count"],
                "regime_compatible_bot_count": activation_plan["regime_compatible_bot_count"],
                "current_regime_known_axis_count": _safe_int(
                    _as_dict(activation_plan.get("current_regime_context")).get("known_axis_count")
                ),
                "current_regime_profile_id": str(
                    _as_dict(activation_plan.get("current_regime_context")).get("profile_id") or ""
                ),
                "limits": activation_plan["limits"],
            },
        ),
        _control(
            "s03",
            "Compute shared features once and distribute immutable snapshots",
            evidence_ready=shared_feature_ready,
            evidence={
                "feature_store_ok": bool(feature_store.get("ok", False)),
                "strict_status": str(feature_store.get("strict_status") or ""),
                "dataset_manifest_sha256": str(feature_hashes.get("dataset_manifest_sha256") or ""),
                "point_in_time_seed_ready": bool(point_in_time.get("seed_ready", False)),
            },
        ),
        _control(
            "s04",
            "Bound workers, training, memory, queues, and per-sleeve compute",
            evidence_ready=resource_ready,
            evidence={
                "runtime_throttle_ok": bool(runtime_throttle.get("ok", False)),
                "resource_governor_ok": bool(resource_governor.get("ok", False)),
                "runtime_throttle_profile": str(runtime_throttle.get("throttle_profile") or ""),
                "budgets": _as_dict(_as_dict(policy.get("scalability")).get("resource_budgets")),
            },
        ),
        _control(
            "s05",
            "Require idempotent checkpoints, decision identities, and order intents",
            evidence_ready=checkpoint_ready,
            evidence={
                "runtime_checkpoint_count": _safe_int(runtime_evidence.get("runtime_checkpoint_count")),
                "order_idempotency_registry_count": _safe_int(runtime_evidence.get("order_idempotency_registry_count")),
                "decision_identity_coverage_ratio": _safe_float(runtime_evidence.get("decision_identity_coverage_ratio")),
                "duplicate_source_row_count": _safe_int(runtime_evidence.get("duplicate_source_row_count")),
            },
        ),
        _control(
            "s06",
            "Separate hot data from compressed, readable cold archives",
            evidence_ready=storage_ready,
            evidence={
                "cold_archive_ok": bool(cold_archive.get("ok", False)),
                "archive_root": str(cold_archive.get("archive_root") or ""),
                "manifest_path": str(cold_archive.get("manifest_path") or ""),
                "reader_command_count": reader_command_count,
                "hot_priority": str(data_tier.get("core_priority") or ""),
                "cold_budget": str(data_tier.get("cold_budget") or ""),
            },
        ),
        _control(
            "s07",
            "Lazy-load models and evict inactive models under memory pressure",
            evidence_ready=model_cache_ready,
            evidence={
                "lazy_loading_required": bool(cache_policy.get("lazy_loading_required", False)),
                "maximum_loaded_models": _safe_int(cache_policy.get("maximum_loaded_models")),
                "maximum_cache_mb": _safe_int(cache_policy.get("maximum_cache_mb")),
                "inactive_ttl_seconds": _safe_float(cache_policy.get("inactive_ttl_seconds")),
                "critical_pressure_evict_all": bool(cache_policy.get("evict_all_under_critical_memory_pressure", False)),
            },
        ),
        _control(
            "s08",
            "Scale proposed allocation by liquidity and strategy capacity",
            evidence_ready=capacity_allocation_ready,
            evidence={
                "capacity_ready_bot_count": len(capacity_profiles),
                "selected_capacity_ready_bot_count": activation_plan["selected_count"],
                "account_size_alone_may_increase_weight": False,
                "automatic_allocation": False,
            },
        ),
    ]
    controls = profitability_controls + scalability_controls
    implemented_count = sum(1 for row in controls if row.get("implemented"))
    evidence_count = sum(1 for row in controls if row.get("evidence_ready"))
    structural_ready = not policy_errors and implemented_count == len(ALL_CONTROL_IDS)
    evidence_ready = evidence_count == len(ALL_CONTROL_IDS)
    blockers = list(policy_errors)
    if len(assignments) == 0:
        blockers.append("bot_hierarchy_empty")
    if len({str(row.get("bot_id") or "") for row in assignments}) != len(assignments):
        blockers.append("bot_hierarchy_duplicate_ids")
    structural_ready = structural_ready and not blockers
    live_allocation_ready = bool(
        structural_ready
        and evidence_ready
        and firewall.get("live_promotion_ready", False)
        and activation_plan["selected_count"] > 0
    )
    receipt_input = {
        "policy_id": policy.get("policy_id"),
        "assignment_count": len(assignments),
        "assignment_receipt": canonical_hash(
            [
                {
                    "bot_id": row.get("bot_id"),
                    "cell_id": row.get("cell_id"),
                    "correlation_cluster_id": row.get("correlation_cluster_id"),
                }
                for row in assignments
            ]
        ),
        "observation_receipt": canonical_hash(observations),
        "current_regime_context_receipt": canonical_hash(current_regime_context),
        "regime_model_receipt": canonical_hash(regime_model),
        "activation_receipt": activation_plan["selection_receipt_sha256"],
    }
    manifest = {
        "schema_version": 1,
        "policy_id": str(policy.get("policy_id") or ""),
        "catalog_bot_count": len(assignments),
        "candidate_observed_bot_count": len(candidate_profiles),
        "profiles": profiles,
        "activation_plan": activation_plan,
        "authority_contract": {
            "advisory_only": True,
            "automatic_registry_mutation": False,
            "paper_execution_authority": False,
            "live_execution_authority": False,
            "automatic_allocation": False,
        },
        "receipt_sha256": canonical_hash(receipt_input),
    }
    health = {
        "schema_version": 1,
        "policy_id": str(policy.get("policy_id") or ""),
        "ok": structural_ready,
        "overall_status": "ready" if structural_ready and evidence_ready else "ready_with_evidence_debt" if structural_ready else "blocked",
        "control_grade": _grade(implemented_count / len(ALL_CONTROL_IDS), structurally_ready=structural_ready),
        "control_score": round(implemented_count / len(ALL_CONTROL_IDS) * 100.0, 4),
        "economic_and_scale_evidence_grade": _grade(evidence_count / len(ALL_CONTROL_IDS)),
        "economic_and_scale_evidence_score": round(evidence_count / len(ALL_CONTROL_IDS) * 100.0, 4),
        "implemented_control_count": implemented_count,
        "evidence_ready_control_count": evidence_count,
        "control_count": len(ALL_CONTROL_IDS),
        "profitability_controls": profitability_controls,
        "scalability_controls": scalability_controls,
        "catalog_bot_count": len(assignments),
        "candidate_observed_bot_count": len(candidate_profiles),
        "learned_preference_bot_count": len(learned_profiles),
        "ranked_bot_count": len(ranked_profiles),
        "persistent_bot_count": len(persistence_profiles),
        "capacity_ready_bot_count": len(capacity_profiles),
        "planned_active_bot_count": activation_plan["selected_count"],
        "planned_parked_bot_count": activation_plan["parked_by_budget_count"] + activation_plan["evidence_pending_count"],
        "paper_collection_ready": structural_ready,
        "live_allocation_ready": live_allocation_ready,
        "automatic_allocation_allowed": False,
        "profitability_guaranteed": False,
        "blockers": _ordered_unique(blockers),
        "evidence_debt": [row["control_id"] for row in controls if not row.get("evidence_ready")],
        "safety_contract": dict(_as_dict(policy.get("safety_contract"))),
        "manifest_receipt_sha256": manifest["receipt_sha256"],
        "evidence_epoch": {
            "id": f"bot-profitability-scalability:{manifest['receipt_sha256'][:16]}",
            "receipt_sha256": manifest["receipt_sha256"],
        },
        "recommended_actions": _ordered_unique(
            [
                "continue candidate-bound paper collection until per-bot day, regime, and sample floors are met"
                if not ranked_profiles
                else "",
                "ingest independent broker-paper or licensed replay fills before enabling capacity evidence"
                if not independent_execution_ready
                else "",
                "keep all proposed bot weights at zero until locked holdout, multiple-testing, and walk-forward evidence pass"
                if not statistical_ready
                else "",
                "review lifecycle advice manually; this controller never mutates the bot registry",
            ]
        ),
    }
    return health, manifest
