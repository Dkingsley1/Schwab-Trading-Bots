#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin
from urllib.request import Request, urlopen

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "schwab_indicator_intelligence_latest.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.schwab_indicator_intelligence_override"

THINKMANUAL_ROOT = "https://toslc.thinkorswim.com"
TECH_INDICATORS_URL = f"{THINKMANUAL_ROOT}/center/reference/Tech-Indicators"
STUDIES_INDEX_URL = f"{TECH_INDICATORS_URL}/studies-library"
STRATEGIES_INDEX_URL = f"{TECH_INDICATORS_URL}/strategies"

STUDY_GROUPS = ("A-B", "C-D", "E-F", "G-L", "M-N", "O-Q", "R-S", "T-U", "V-Z")
STRATEGY_GROUPS = ("A-D", "E-K", "L-P", "R-S", "T-Z")
REQUIRED_GROUP_KEYS = tuple([f"study:{group}" for group in STUDY_GROUPS] + [f"strategy:{group}" for group in STRATEGY_GROUPS])

CATALOG_MINIMUM_COMPLETE_COUNT = 80
OFFICIAL_USER_AGENT = "schwab-trading-bot/1.0 schwab-indicator-intelligence"


FALLBACK_SEED_ITEMS: tuple[dict[str, str], ...] = (
    {"kind": "study", "name": "ADX", "group": "A-B"},
    {"kind": "study", "name": "ATR", "group": "A-B"},
    {"kind": "study", "name": "BollingerBands", "group": "A-B"},
    {"kind": "study", "name": "BollingerPercentB", "group": "A-B"},
    {"kind": "study", "name": "MACD", "group": "M-N"},
    {"kind": "study", "name": "MovingAvgExponential", "group": "M-N"},
    {"kind": "study", "name": "RSI", "group": "R-S"},
    {"kind": "study", "name": "StochasticFull", "group": "R-S"},
    {"kind": "study", "name": "VWAP", "group": "V-Z"},
    {"kind": "strategy", "name": "ADXTrend", "group": "A-D"},
    {"kind": "strategy", "name": "BollingerBandsLE", "group": "A-D"},
    {"kind": "strategy", "name": "BollingerBandsSE", "group": "A-D"},
    {"kind": "strategy", "name": "MACDStrat", "group": "L-P"},
    {"kind": "strategy", "name": "RSIStrat", "group": "R-S"},
    {"kind": "strategy", "name": "Stochastic", "group": "R-S"},
)


FAMILY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("trend", ("adx", "adxr", "aroon", "macd", "movingavg", "ema", "sma", "hull", "trend", "ichimoku", "onsettrend", "parabolicsar")),
    ("momentum", ("momentum", "roc", "rsi", "stoch", "oscillator", "pmo", "cci", "williams", "tsi", "ultimate")),
    ("volatility", ("atr", "bollinger", "volatility", "keltner", "standarddev", "deviation", "vix", "impvol", "true range")),
    ("volume_flow", ("volume", "vwap", "accum", "distribution", "moneyflow", "obv", "onbalance", "chaikin", "easeofmovement")),
    ("mean_reversion", ("meanreversion", "regression", "reverse", "percentb", "pricezone", "zscore", "bands", "channel")),
    ("breakout", ("breakout", "donchian", "highlow", "gap", "camarilla", "pivot", "insidebar", "flag", "keyrev")),
    ("breadth", ("advancedecline", "mcclellan", "arms", "trin", "breadth")),
    ("pair_correlation", ("correlation", "beta", "pairtrading", "multicurrency", "cointegration")),
    ("support_resistance", ("pivot", "camarilla", "fibonacci", "support", "resistance", "points")),
    ("seasonality", ("seasonal", "halloween", "month", "eightmonth", "longhaul")),
    ("account_risk", ("account", "netliq", "profit", "stoploss", "trailingstop", "target")),
    ("pattern", ("engulfing", "insidebar", "barsup", "barsdown", "swingthree", "candlestick")),
)

FAMILY_TO_INPUTS: dict[str, list[str]] = {
    "trend": ["ohlc_price_bars"],
    "momentum": ["ohlc_price_bars"],
    "volatility": ["ohlc_price_bars", "high_low_range"],
    "volume_flow": ["ohlc_price_bars", "volume"],
    "mean_reversion": ["ohlc_price_bars", "rolling_window_context"],
    "breakout": ["ohlc_price_bars", "session_high_low"],
    "breadth": ["market_breadth"],
    "pair_correlation": ["multi_symbol_history"],
    "support_resistance": ["ohlc_price_bars", "session_high_low"],
    "seasonality": ["calendar_context", "historical_returns"],
    "account_risk": ["account_or_position_context"],
    "pattern": ["ohlc_price_bars"],
}

FAMILY_TO_SLEEVES: dict[str, list[str]] = {
    "trend": ["trend_following", "swing", "quality_growth", "futures_macro", "trading_muscles"],
    "momentum": ["intraday_aggressive", "day_trading", "swing", "trading_muscles"],
    "volatility": ["options_income", "options_risk", "risk_guard", "intraday_aggressive", "futures_macro"],
    "volume_flow": ["intraday_aggressive", "day_trading", "microstructure", "execution_quality", "trading_muscles"],
    "mean_reversion": ["pairs", "stat_arb", "swing", "sector_rotation", "trading_muscles"],
    "breakout": ["intraday_aggressive", "day_trading", "swing", "event_driven"],
    "breadth": ["market_posture", "risk_guard", "sector_rotation", "macro"],
    "pair_correlation": ["pairs", "stat_arb", "portfolio_hedging", "sector_rotation"],
    "support_resistance": ["intraday_aggressive", "options_income", "swing", "execution_timing"],
    "seasonality": ["seasonality", "swing", "income", "portfolio_rotation"],
    "account_risk": ["risk_guard", "income", "portfolio_hedging", "execution_quality"],
    "pattern": ["intraday_aggressive", "swing", "event_driven"],
}

FAMILY_TO_CIRCUMSTANCES: dict[str, list[str]] = {
    "trend": [
        "persistent_directional_move",
        "higher_timeframe_confirmation",
        "avoid_when_choppy_or_low_trend_strength",
    ],
    "momentum": [
        "impulse_or_continuation_setup",
        "overbought_oversold_context_requires_confirmation",
        "avoid_when_spread_or_latency_is_unstable",
    ],
    "volatility": [
        "position_sizing_stop_width_or_options_premium_context",
        "volatility_expansion_or_compression",
        "avoid_when_range_inputs_are_stale",
    ],
    "volume_flow": [
        "liquidity_confirmation",
        "vwap_reclaim_rejection_or_accumulation_distribution_context",
        "avoid_when_volume_feed_is_missing_or_thin",
    ],
    "mean_reversion": [
        "range_bound_or_statistical_deviation_context",
        "pairs_or_channel_reversion_candidate",
        "avoid_when_breakout_pressure_is_high",
    ],
    "breakout": [
        "range_expansion_or_opening_drive_context",
        "session_high_low_or_gap_follow_through",
        "avoid_when_false_breakout_rate_is_elevated",
    ],
    "breadth": [
        "market_internal_confirmation",
        "risk_on_risk_off_filtering",
        "avoid_as_single_name_entry_trigger",
    ],
    "pair_correlation": [
        "relative_value_or_hedge_context",
        "spread_divergence_with_stable_relationship",
        "avoid_when_correlation_regime_has_broken",
    ],
    "support_resistance": [
        "entry_timing_near_defined_levels",
        "stop_or_profit_target_context",
        "avoid_when_levels_are_unconfirmed_or_stale",
    ],
    "seasonality": [
        "calendar_effect_context",
        "longer_horizon_bias_only",
        "avoid_as_standalone_trade_trigger",
    ],
    "account_risk": [
        "position_or_account_protection_context",
        "stop_loss_target_or_net_liq_awareness",
        "never_use_to_override_global_halts",
    ],
    "pattern": [
        "candlestick_or_bar_pattern_confirmation",
        "requires_volume_trend_or_regime_confirmation",
        "avoid_when_market_quality_is_poor",
    ],
}

FAMILY_MECHANISMS: dict[str, str] = {
    "trend": "Measures directional persistence with smoothed price, trend-strength, or crossover behavior; useful as a bias or continuation filter.",
    "momentum": "Measures rate of change or oscillator pressure; useful for impulse continuation, exhaustion checks, and confirmation timing.",
    "volatility": "Measures range, dispersion, or band expansion/compression; useful for sizing, stop width, options context, and regime filters.",
    "volume_flow": "Combines price with volume or participation; useful for confirming liquidity, accumulation/distribution, and VWAP-style intraday context.",
    "mean_reversion": "Measures distance from a moving center, band, channel, or statistical norm; useful when range or spread behavior is stable.",
    "breakout": "Measures movement through prior ranges, pivots, highs, lows, or gaps; useful when expansion and follow-through are the primary setup.",
    "breadth": "Measures market internals across many symbols; useful as a posture filter rather than a standalone single-name trigger.",
    "pair_correlation": "Measures relationship, beta, spread, or co-movement between instruments; useful for hedge, pair, and relative-value sleeves.",
    "support_resistance": "Maps reference levels from prior price structure; useful for entry timing, stops, targets, and risk/reward checks.",
    "seasonality": "Maps calendar or historical-period tendencies; useful as a longer-horizon bias that requires current-market confirmation.",
    "account_risk": "Uses account, position, target, or stop context; useful for protection and reporting but never for overriding safety gates.",
    "pattern": "Detects bar or candlestick formations; useful as a setup clue that requires volume, trend, or regime confirmation.",
    "strategy_signal": "Represents a thinkorswim strategy template; useful for backtest inspiration and candidate signals only after validation.",
    "general_technical": "General chart-derived technical context; useful for research until a tighter family, validation set, and sleeve contract are assigned.",
}

FAMILY_RISK_NOTES: dict[str, list[str]] = {
    "trend": ["lags reversals", "can whipsaw in range-bound markets"],
    "momentum": ["can stay overbought_or_oversold", "needs spread_and_latency_quality_for_intraday_use"],
    "volatility": ["range_inputs_must_be_fresh", "does_not_predict_direction_by_itself"],
    "volume_flow": ["requires_reliable_volume", "thin_symbols_can_distort_signal"],
    "mean_reversion": ["breakouts_can_invalidate_reversion", "needs_stable_regime_or_pair_relationship"],
    "breakout": ["false_breakout_risk", "needs_liquidity_and_follow_through_confirmation"],
    "breadth": ["portfolio_filter_not_single_name_trigger", "index_composition_can_shift"],
    "pair_correlation": ["relationship_break_risk", "requires_multi_symbol_history_quality"],
    "support_resistance": ["levels_can_be_crowded_or_stale", "needs_current_order_flow_confirmation"],
    "seasonality": ["historical_bias_only", "must_not_override_current_risk_state"],
    "account_risk": ["never_override_global_halt_or_operator_stop", "requires_account_data_freshness"],
    "pattern": ["pattern_frequency_can_overfit", "requires_confirmation"],
    "strategy_signal": ["requires_walk_forward_validation", "paper_gate_required_before_weight"],
    "general_technical": ["research_only_until_classified", "requires_validation_before_sleeve_weight"],
}


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[dict[str, str]] = []
        self._href_stack: list[str] = []
        self._text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = ""
        for key, value in attrs:
            if key.lower() == "href" and value:
                href = value
                break
        self._href_stack.append(href)
        self._text_parts = []

    def handle_data(self, data: str) -> None:
        if self._href_stack:
            self._text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or not self._href_stack:
            return
        href = self._href_stack.pop()
        text = " ".join(part.strip() for part in self._text_parts if part.strip()).strip()
        self._text_parts = []
        if href and text:
            self.links.append({"href": href, "text": text})


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name or "").strip().lower()).strip("_")


def _fetch_text(url: str, *, timeout: int = 20) -> str:
    request = Request(url, headers={"User-Agent": OFFICIAL_USER_AGENT})
    with urlopen(request, timeout=timeout) as response:  # nosec - official Schwab thinkManual URL, operator controlled.
        data = response.read()
    return data.decode("utf-8", errors="replace")


def _group_url(kind: str, group: str) -> str:
    if kind == "study":
        return f"{STUDIES_INDEX_URL}/{group}"
    return f"{STRATEGIES_INDEX_URL}/{group}"


def _catalog_path_fragment(kind: str, group: str) -> str:
    if kind == "study":
        return f"/center/reference/Tech-Indicators/studies-library/{group}/"
    return f"/center/reference/Tech-Indicators/strategies/{group}/"


def _parse_group_links(kind: str, group: str, html: str) -> list[dict[str, str]]:
    parser = _LinkParser()
    parser.feed(html)
    fragment = _catalog_path_fragment(kind, group)
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for link in parser.links:
        href = urljoin(THINKMANUAL_ROOT, link["href"])
        name = str(link["text"] or "").strip()
        if not name or href in seen:
            continue
        if fragment not in href:
            continue
        if name in {group, "Studies Library", "Strategies Library", "Tech Indicators"}:
            continue
        seen.add(href)
        rows.append({"kind": kind, "name": name, "group": group, "url": href})
    return rows


def _official_catalog(
    *,
    offline: bool = False,
    timeout: int = 20,
    retry_count: int = 1,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if offline:
        return [], {
            "mode": "offline_requested",
            "source_urls": [STUDIES_INDEX_URL, STRATEGIES_INDEX_URL],
            "groups_required": len(STUDY_GROUPS) + len(STRATEGY_GROUPS),
            "groups_fetched": 0,
            "groups_failed": len(STUDY_GROUPS) + len(STRATEGY_GROUPS),
            "failures": ["offline_requested"],
        }

    rows: list[dict[str, str]] = []
    failures: list[str] = []
    fetched = 0
    for kind, groups in (("study", STUDY_GROUPS), ("strategy", STRATEGY_GROUPS)):
        for group in groups:
            url = _group_url(kind, group)
            group_rows: list[dict[str, str]] = []
            last_error = ""
            for _attempt in range(max(int(retry_count), 0) + 1):
                try:
                    html = _fetch_text(url, timeout=timeout)
                    group_rows = _parse_group_links(kind, group, html)
                    last_error = ""
                    break
                except Exception as exc:
                    last_error = type(exc).__name__
            if last_error:
                failures.append(f"{kind}:{group}:{last_error}")
                continue
            fetched += 1
            rows.extend(group_rows)
    required = len(STUDY_GROUPS) + len(STRATEGY_GROUPS)
    return rows, {
        "mode": "official_fetch",
        "source_urls": [STUDIES_INDEX_URL, STRATEGIES_INDEX_URL],
        "groups_required": required,
        "groups_fetched": fetched,
        "groups_failed": required - fetched,
        "failures": failures,
    }


def _fallback_catalog() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in FALLBACK_SEED_ITEMS:
        kind = item["kind"]
        group = item["group"]
        name = item["name"]
        rows.append(
            {
                "kind": kind,
                "name": name,
                "group": group,
                "url": f"{_group_url(kind, group)}/{name}",
            }
        )
    return rows


def _classify_families(name: str, kind: str) -> list[str]:
    normalized = _slug(name).replace("_", "")
    families: list[str] = []
    for family, tokens in FAMILY_RULES:
        for token in tokens:
            if token.replace(" ", "").lower() in normalized:
                families.append(family)
                break
    if kind == "strategy":
        families.append("strategy_signal")
    if not families:
        families.append("general_technical")
    return ordered_unique(families)


def _strategy_direction(name: str, kind: str) -> str:
    if kind != "strategy":
        return ""
    if name.endswith("LE"):
        return "long_entry"
    if name.endswith("SE"):
        return "short_entry"
    if name.endswith("LX"):
        return "long_exit"
    if name.endswith("SX"):
        return "short_exit"
    return "strategy_or_system"


def _required_inputs(families: list[str], kind: str) -> list[str]:
    inputs: list[str] = []
    for family in families:
        inputs.extend(FAMILY_TO_INPUTS.get(family, ["ohlc_price_bars"]))
    if kind == "strategy":
        inputs.extend(["historical_signal_backtest", "paper_validation_evidence"])
    return ordered_unique(inputs)


def _applicable_sleeves(families: list[str]) -> list[str]:
    sleeves: list[str] = []
    for family in families:
        sleeves.extend(FAMILY_TO_SLEEVES.get(family, ["research", "feature_store"]))
    return ordered_unique(sleeves)


def _circumstances(families: list[str], kind: str) -> list[str]:
    circumstances: list[str] = []
    for family in families:
        circumstances.extend(FAMILY_TO_CIRCUMSTANCES.get(family, ["research_context_only"]))
    if kind == "strategy":
        circumstances.extend(
            [
                "use_as_backtest_or_candidate_signal_template",
                "requires_walk_forward_and_paper_gate_before_any_weight",
                "never_promotes_live_authority_by_itself",
            ]
        )
    return ordered_unique(circumstances)


def _mechanism_summary(families: list[str]) -> list[str]:
    return ordered_unique([FAMILY_MECHANISMS.get(family, FAMILY_MECHANISMS["general_technical"]) for family in families])


def _risk_notes(families: list[str], kind: str) -> list[str]:
    notes: list[str] = []
    for family in families:
        notes.extend(FAMILY_RISK_NOTES.get(family, FAMILY_RISK_NOTES["general_technical"]))
    if kind == "strategy":
        notes.extend(["strategy_template_not_execution_permission", "live_authority_remains_false"])
    return ordered_unique(notes)


def _catalog_item(row: dict[str, str]) -> dict[str, Any]:
    kind = str(row.get("kind") or "").strip().lower()
    name = str(row.get("name") or "").strip()
    families = _classify_families(name, kind)
    return {
        "id": f"schwab_{kind}_{_slug(name)}",
        "name": name,
        "kind": kind,
        "group": str(row.get("group") or ""),
        "url": str(row.get("url") or ""),
        "families": families,
        "mechanism_summary": _mechanism_summary(families),
        "required_inputs": _required_inputs(families, kind),
        "applicable_sleeves": _applicable_sleeves(families),
        "circumstance_triggers": _circumstances(families, kind),
        "risk_notes": _risk_notes(families, kind),
        "strategy_direction": _strategy_direction(name, kind),
        "authority": "advisory_feature_candidate",
    }


def _dedupe_items(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        item = _catalog_item(row)
        key = str(item["id"])
        if key not in by_key or (not by_key[key].get("url") and item.get("url")):
            by_key[key] = item
    return sorted(by_key.values(), key=lambda item: (str(item["kind"]), str(item["name"]).lower()))


def _previous_catalog_items(project_root: Path) -> list[dict[str, Any]]:
    payload = load_json(project_root / "governance" / "health" / "schwab_indicator_intelligence_latest.json")
    rows = payload.get("catalog_items") if isinstance(payload.get("catalog_items"), list) else []
    out: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict) and row.get("id") and row.get("kind") and row.get("group"):
            refreshed = _catalog_item(
                {
                    "kind": str(row.get("kind") or ""),
                    "name": str(row.get("name") or ""),
                    "group": str(row.get("group") or ""),
                    "url": str(row.get("url") or ""),
                }
            )
            out.append({**refreshed, "catalog_record_source": str(row.get("catalog_record_source") or "previous_artifact_cache")})
    return out


def _group_key(item: dict[str, Any]) -> str:
    return f"{str(item.get('kind') or '').strip()}:{str(item.get('group') or '').strip()}"


def _merge_current_with_previous(current: list[dict[str, Any]], previous: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    current_groups = {_group_key(item) for item in current if _group_key(item) in REQUIRED_GROUP_KEYS}
    by_id = {str(item.get("id")): {**item, "catalog_record_source": "current_official_or_seed"} for item in current}
    cached_groups: set[str] = set()
    cached_count = 0
    for item in previous:
        group_key = _group_key(item)
        if group_key not in REQUIRED_GROUP_KEYS or group_key in current_groups:
            continue
        item_id = str(item.get("id") or "")
        if not item_id or item_id in by_id:
            continue
        cached = dict(item)
        cached["catalog_record_source"] = "previous_artifact_cache"
        by_id[item_id] = cached
        cached_groups.add(group_key)
        cached_count += 1
    merged = sorted(by_id.values(), key=lambda item: (str(item.get("kind")), str(item.get("name")).lower()))
    return merged, {
        "current_group_count": len(current_groups),
        "cached_group_count": len(cached_groups),
        "cached_item_count": cached_count,
        "cached_group_keys": sorted(cached_groups),
    }


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _status(payload: dict[str, Any]) -> str:
    if not payload:
        return "missing"
    raw = payload.get("overall_status")
    if raw is None:
        raw = payload.get("status")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    if "ok" in payload:
        return "ready" if bool(payload.get("ok")) else "blocked"
    return "unknown"


def _load_runtime_context(project_root: Path) -> dict[str, Any]:
    health = project_root / "governance" / "health"
    runtime = load_json(health / "runtime_throttle_control_latest.json")
    health_fast = load_json(health / "health_fast_latest.json")
    market_posture = load_json(health / "market_posture_control_latest.json")
    paper_ramp = load_json(health / "paper_400_ramp_latest.json")
    paper_perf = load_json(health / "paper_performance_latest.json")
    runtime_level = str(runtime.get("memory_pressure_level") or runtime.get("compute_pressure_level") or "").lower()
    guarded = bool(
        _status(runtime) in {"blocked", "degraded", "needs_work"}
        or _status(health_fast) in {"blocked", "degraded", "needs_work"}
        or runtime_level in {"high", "elevated"}
    )
    return {
        "runtime_status": _status(runtime),
        "health_fast_status": _status(health_fast),
        "market_posture_status": _status(market_posture),
        "paper_ramp_status": _status(paper_ramp),
        "paper_performance_status": _status(paper_perf),
        "runtime_guarded": guarded,
        "posture_label": str(
            market_posture.get("posture")
            or market_posture.get("overall_posture")
            or market_posture.get("market_posture")
            or "unknown"
        ),
        "paper_ramp_stage": str(paper_ramp.get("paper_ramp_stage") or paper_ramp.get("stage") or "unknown"),
    }


def _registry_sleeve_profiles(project_root: Path) -> list[str]:
    rows = _as_list(load_json(project_root / "master_bot_registry.json").get("sub_bots"))
    profiles: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in ("sleeve_profile", "sleeve_family", "slot_kind"):
            value = str(row.get(key) or "").strip()
            if value:
                profiles.add(value)
    return sorted(profiles)


def _sleeve_matrix(items: list[dict[str, Any]], project_root: Path) -> list[dict[str, Any]]:
    registry_profiles = _registry_sleeve_profiles(project_root)
    canonical = ordered_unique(
        [
            "intraday_aggressive",
            "day_trading",
            "swing",
            "quality_growth",
            "options_income",
            "options_risk",
            "risk_guard",
            "pairs",
            "stat_arb",
            "sector_rotation",
            "portfolio_hedging",
            "futures_macro",
            "market_posture",
            "trading_muscles",
        ]
        + registry_profiles
    )
    rows: list[dict[str, Any]] = []
    for sleeve in canonical:
        relevant = [
            item
            for item in items
            if any(str(route) in sleeve or sleeve in str(route) for route in _as_list(item.get("applicable_sleeves")))
        ]
        families = sorted({family for item in relevant for family in _as_list(item.get("families"))})
        rows.append(
            {
                "sleeve": sleeve,
                "mapped_item_count": len(relevant),
                "mapped_families": families,
                "top_studies": [str(item.get("name")) for item in relevant if item.get("kind") == "study"][:12],
                "top_strategies": [str(item.get("name")) for item in relevant if item.get("kind") == "strategy"][:12],
                "apply_when": ordered_unique(
                    [
                        trigger
                        for item in relevant[:20]
                        for trigger in _as_list(item.get("circumstance_triggers"))
                        if not str(trigger).startswith("avoid_")
                    ]
                )[:12],
                "avoid_when": ordered_unique(
                    [
                        trigger
                        for item in relevant[:20]
                        for trigger in _as_list(item.get("circumstance_triggers"))
                        if str(trigger).startswith("avoid_") or str(trigger).startswith("never_")
                    ]
                )[:12],
                "authority": "advisory_feature_routing_no_execution_authority",
            }
        )
    return sorted(rows, key=lambda row: (-int(row["mapped_item_count"]), str(row["sleeve"])))


def _coverage_summary(
    items: list[dict[str, Any]],
    fetch_meta: dict[str, Any],
    *,
    used_fallback: bool,
    merge_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    by_kind = Counter(str(item.get("kind") or "unknown") for item in items)
    by_family = Counter(family for item in items for family in _as_list(item.get("families")))
    observed_group_keys = sorted({_group_key(item) for item in items if _group_key(item) in REQUIRED_GROUP_KEYS})
    missing_group_keys = [group for group in REQUIRED_GROUP_KEYS if group not in set(observed_group_keys)]
    current_fetch_complete = bool(not used_fallback and int(fetch_meta.get("groups_failed") or 0) == 0)
    group_coverage_complete = bool(not missing_group_keys and len(items) >= CATALOG_MINIMUM_COMPLETE_COUNT)
    complete = bool(not used_fallback and (current_fetch_complete or group_coverage_complete))
    if current_fetch_complete:
        coverage_status = "official_catalog_complete"
    elif group_coverage_complete:
        coverage_status = "official_catalog_complete_with_cached_groups"
    else:
        coverage_status = "partial_or_seed_catalog_needs_refresh"
    return {
        "coverage_status": coverage_status,
        "official_fetch_complete": current_fetch_complete,
        "catalog_group_coverage_complete": group_coverage_complete,
        "catalog_item_count": len(items),
        "study_count": int(by_kind.get("study", 0)),
        "strategy_count": int(by_kind.get("strategy", 0)),
        "family_counts": dict(sorted(by_family.items())),
        "minimum_complete_count": CATALOG_MINIMUM_COMPLETE_COUNT,
        "used_fallback_seed": used_fallback,
        "groups_required": int(fetch_meta.get("groups_required") or 0),
        "groups_fetched": int(fetch_meta.get("groups_fetched") or 0),
        "groups_failed": int(fetch_meta.get("groups_failed") or 0),
        "groups_observed_count": len(observed_group_keys),
        "groups_observed": observed_group_keys,
        "missing_group_keys": missing_group_keys,
        "cache_merge": merge_meta or {},
    }


def _routing_contract(runtime_context: dict[str, Any]) -> dict[str, Any]:
    guarded = bool(runtime_context.get("runtime_guarded", False))
    return {
        "authority_boundary": "advisory_intelligence_layer_no_live_or_paper_order_authority",
        "study_usage_policy": "studies_can_be_features_filters_or_context_only_after_input_quality_checks",
        "strategy_usage_policy": "strategies_are_backtest_templates_or_candidate_signals_until_walk_forward_and_paper_gates_clear",
        "runtime_action_mode": "observe_only_under_runtime_pressure" if guarded else "advisory_routing_ready",
        "paper_execution_authority": False,
        "live_execution_authority": False,
        "requires_training_or_walk_forward_before_weight": True,
        "blocks": ordered_unique(
            [
                "runtime_pressure_guarded" if guarded else "",
                "missing_required_inputs_for_study",
                "stale_catalog_fetch",
                "no_sleeve_mapping",
                "paper_or_live_gate_not_cleared",
            ]
        ),
    }


def _write_override(path: Path, payload: dict[str, Any]) -> None:
    coverage = _as_dict(payload.get("coverage"))
    contract = _as_dict(payload.get("routing_contract"))
    lines = [
        "# Generated by schwab_indicator_intelligence.py",
        f"SCHWAB_INDICATOR_INTELLIGENCE_READY={1 if payload.get('ok') else 0}",
        f"SCHWAB_INDICATOR_CATALOG_ITEMS={coverage.get('catalog_item_count', 0)}",
        f"SCHWAB_INDICATOR_STUDY_COUNT={coverage.get('study_count', 0)}",
        f"SCHWAB_INDICATOR_STRATEGY_COUNT={coverage.get('strategy_count', 0)}",
        f"SCHWAB_INDICATOR_OFFICIAL_FETCH_COMPLETE={1 if coverage.get('official_fetch_complete') else 0}",
        f"SCHWAB_INDICATOR_ACTION_MODE={contract.get('runtime_action_mode', '')}",
        "SCHWAB_INDICATOR_LIVE_EXECUTION_AUTHORITY=0",
        "SCHWAB_INDICATOR_PAPER_EXECUTION_AUTHORITY=0",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    offline: bool = False,
    apply: bool = False,
    override_path: Path = DEFAULT_OVERRIDE_PATH,
    fetch_timeout: int = 20,
    retry_count: int = 1,
) -> dict[str, Any]:
    official_rows, fetch_meta = _official_catalog(offline=offline, timeout=fetch_timeout, retry_count=retry_count)
    used_fallback = False
    if not official_rows:
        official_rows = _fallback_catalog()
        used_fallback = True
        fetch_meta = {**fetch_meta, "mode": "fallback_seed_partial", "fallback_reason": "official_catalog_unavailable_or_empty"}
    current_items = _dedupe_items(official_rows)
    previous_items = _previous_catalog_items(project_root)
    current_for_merge = [] if used_fallback and previous_items else current_items
    items, merge_meta = _merge_current_with_previous(current_for_merge, previous_items)
    if not items:
        items = current_items
    runtime_context = _load_runtime_context(project_root)
    matrix = _sleeve_matrix(items, project_root)
    coverage = _coverage_summary(items, fetch_meta, used_fallback=used_fallback, merge_meta=merge_meta)
    complete = bool(coverage.get("official_fetch_complete") or coverage.get("catalog_group_coverage_complete"))
    ok = bool(items) and (complete or used_fallback)
    overall_status = (
        "schwab_indicator_intelligence_ready"
        if bool(coverage.get("official_fetch_complete"))
        else "schwab_indicator_intelligence_ready_cached"
        if bool(coverage.get("catalog_group_coverage_complete"))
        else "schwab_indicator_intelligence_partial_catalog"
    )
    family_gaps = sorted(set(FAMILY_TO_SLEEVES) - set(_as_dict(coverage.get("family_counts")).keys()))
    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": ok,
        "overall_status": overall_status,
        "catalog_source": {
            **fetch_meta,
            "official_reference": {
                "tech_indicators_url": TECH_INDICATORS_URL,
                "studies_index_url": STUDIES_INDEX_URL,
                "strategies_index_url": STRATEGIES_INDEX_URL,
            },
            "retry_count": max(int(retry_count), 0),
        },
        "coverage": coverage,
        "runtime_context": runtime_context,
        "routing_contract": _routing_contract(runtime_context),
        "classification_model": {
            "model_version": "schwab_indicator_intelligence_v1",
            "families": sorted(FAMILY_TO_SLEEVES),
            "family_gap_count": len(family_gaps),
            "families_not_observed_in_catalog": family_gaps,
            "classification_method": "official_name_catalog_with_rule_based_family_input_sleeve_circumstance_routing",
        },
        "sleeve_applicability_matrix": matrix,
        "catalog_items": items,
        "recommended_commands": {
            "refresh_indicator_intelligence": ["./scripts/ops/opsctl.sh", "schwab-indicator-intelligence", "--json"],
            "apply_indicator_intelligence_env": ["./scripts/ops/opsctl.sh", "schwab-indicator-intelligence", "--apply", "--json"],
            "refresh_system_self_model": ["./scripts/ops/opsctl.sh", "big-platform-brain", "--json"],
            "refresh_architecture_contract_graph": ["./scripts/ops/opsctl.sh", "system-architecture-contract-graph", "--apply", "--json"],
        },
        "recommended_actions": ordered_unique(
            [
                "rerun with network access to refresh the official Schwab thinkManual catalog" if not complete else "",
                "use sleeve_applicability_matrix as feature-routing input, not as execution permission",
                "require walk-forward and paper validation before any strategy-derived weight",
                "block any study whose required_inputs are unavailable or stale for the sleeve",
            ]
        ),
    }
    if apply:
        _write_override(override_path, payload)
        payload["write_result"] = {"applied": True, "override_path": str(override_path)}
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Schwab thinkorswim study/strategy intelligence and sleeve routing.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--override-path", default=str(DEFAULT_OVERRIDE_PATH))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--fetch-timeout", type=int, default=20)
    parser.add_argument("--retry-count", type=int, default=1)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    payload = build_payload(
        project_root,
        offline=bool(args.offline),
        apply=bool(args.apply),
        override_path=Path(args.override_path).expanduser(),
        fetch_timeout=max(int(args.fetch_timeout), 1),
        retry_count=max(int(args.retry_count), 0),
    )
    write_payload(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        coverage = _as_dict(payload.get("coverage"))
        print(
            "schwab_indicator_intelligence "
            f"status={payload.get('overall_status')} "
            f"items={coverage.get('catalog_item_count', 0)} "
            f"studies={coverage.get('study_count', 0)} "
            f"strategies={coverage.get('strategy_count', 0)} "
            f"official_complete={1 if coverage.get('official_fetch_complete') else 0}"
        )
    return 0 if payload.get("ok", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
