#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import statistics
import sys
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.research_context_expansion import (  # noqa: E402
    COLLECTOR_BY_ID,
    COLLECTOR_IDS,
)

EXTERNAL_CONTEXT_ROOT = PROJECT_ROOT / "exports" / "external_context"
HEALTH_ROOT = PROJECT_ROOT / "governance" / "health"
AGGREGATE_PATH = HEALTH_ROOT / "research_context_expansion_latest.json"
FINRA_BASE_URL = "https://api.finra.org/data/group/fixedIncomeMarket/name"
BIS_GLI_URL = "https://data.bis.org/static/bulk/WS_GLI_csv_flat.zip"
USER_AGENT = "SchwabTradingBotResearchContext/1.0 local-research"

SECTOR_SYMBOLS = {
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
}
FACTOR_SYMBOLS = {"SPY", "QQQ", "IWM", "MTUM", "VLUE", "QUAL", "USMV"}
CORRELATION_SYMBOLS = ("QQQ", "IWM", "TLT", "GLD", "UUP")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _safe_float(raw: Any) -> float | None:
    try:
        value = float(raw)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _signed_norm(value: float, scale: float) -> float:
    return _clamp01(0.5 + (float(value) / max(float(scale), 1e-12)) * 0.5)


def _canonical_hash(payload: Any) -> str:
    raw = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(dict(payload), ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def _artifact_path(project_root: Path, collector_id: str) -> Path:
    return project_root / "exports" / "external_context" / f"{collector_id}_latest.json"


def _health_path(project_root: Path, collector_id: str) -> Path:
    return project_root / "governance" / "health" / f"{collector_id}_latest.json"


def _artifact_age_minutes(payload: Mapping[str, Any], now: datetime) -> float | None:
    timestamp = _parse_ts(payload.get("timestamp_utc"))
    if timestamp is None:
        return None
    return max((now - timestamp).total_seconds() / 60.0, 0.0)


def _proof_row(
    capability_id: str, usable: bool, evidence: Mapping[str, Any]
) -> dict[str, Any]:
    proof = {
        "capability_id": str(capability_id),
        "usable": bool(usable),
        "proof_semantics": (
            "direct_or_source_backed_derived"
            if usable
            else "insufficient_direct_evidence"
        ),
        "evidence": dict(evidence),
    }
    proof["proof_receipt_sha256"] = _canonical_hash(proof)
    return proof


def _capability_payload(
    collector_id: str,
    *,
    now: datetime,
    source_observed_at: datetime | None,
    capabilities: Mapping[str, tuple[bool, Mapping[str, Any]]],
    derived: Mapping[str, Any],
    metrics: Mapping[str, Any],
    sources: Mapping[str, Any],
    warnings: Iterable[str] = (),
) -> dict[str, Any]:
    definition = COLLECTOR_BY_ID[collector_id]
    rows = [
        _proof_row(
            capability_id,
            *capabilities.get(capability_id, (False, {"reason": "not_evaluated"})),
        )
        for capability_id in definition["capabilities"]
    ]
    usable_count = sum(1 for row in rows if row["usable"])
    payload: dict[str, Any] = {
        "timestamp_utc": now.isoformat(),
        "source_observed_at_utc": (
            source_observed_at.isoformat() if source_observed_at else None
        ),
        "schema_version": 1,
        "collector_id": collector_id,
        "title": definition["title"],
        "ok": usable_count > 0,
        "overall_status": (
            "ready"
            if usable_count == len(rows)
            else "ready_with_evidence_debt" if usable_count else "degraded"
        ),
        "capability_count": len(rows),
        "usable_capability_count": usable_count,
        "capabilities": rows,
        "derived": dict(derived),
        "metrics": dict(metrics),
        "sources": dict(sources),
        "warnings": sorted({str(item) for item in warnings if str(item).strip()}),
        "authority_contract": {
            "observation_only": True,
            "paper_execution_authority": False,
            "live_execution_authority": False,
            "automatic_promotion_authority": False,
            "registry_mutation_authority": False,
        },
        "evidence_contract": {
            "missing_dimensions_are_omitted_not_zero_filled": True,
            "source_timestamp_is_preserved": True,
            "unsupported_level2_is_never_inferred": True,
            "point_in_time_only": True,
        },
    }
    payload["snapshot_receipt_sha256"] = _canonical_hash(payload)
    return payload


def _tail_lines(path: Path, *, max_bytes: int, max_lines: int) -> list[str]:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(max(size - max(int(max_bytes), 1), 0))
            data = handle.read(max(int(max_bytes), 1))
    except OSError:
        return []
    if size > max_bytes:
        newline = data.find(b"\n")
        if newline >= 0:
            data = data[newline + 1 :]
    return data.decode("utf-8", errors="ignore").splitlines()[-max(int(max_lines), 1) :]


def load_recent_decisions(
    project_root: Path,
    *,
    now: datetime,
    lookback_hours: float = 18.0,
    max_files: int = 64,
    max_rows: int = 6000,
    max_bytes_per_file: int = 1_048_576,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = project_root / "decisions"
    try:
        candidates = sorted(
            root.glob("*/trade_decisions_*.jsonl"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )[: max(int(max_files), 1)]
    except OSError:
        candidates = []
    cutoff = now.timestamp() - max(float(lookback_hours), 0.25) * 3600.0
    rows: list[dict[str, Any]] = []
    paths_used: list[str] = []
    latest_observed: datetime | None = None
    for path in candidates:
        accepted_from_file = 0
        for raw in reversed(
            _tail_lines(path, max_bytes=max_bytes_per_file, max_lines=160)
        ):
            try:
                row = json.loads(raw)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            timestamp = _parse_ts(
                row.get("timestamp_utc") or row.get("generated_at_utc")
            )
            if timestamp is None or timestamp.timestamp() < cutoff:
                continue
            row["_collector_source_path"] = str(path.relative_to(project_root))
            rows.append(row)
            accepted_from_file += 1
            if latest_observed is None or timestamp > latest_observed:
                latest_observed = timestamp
            if len(rows) >= max_rows:
                break
        if accepted_from_file:
            paths_used.append(str(path.relative_to(project_root)))
        if len(rows) >= max_rows:
            break
    return rows, {
        "row_count": len(rows),
        "file_count": len(paths_used),
        "files": paths_used,
        "latest_observed_at_utc": (
            latest_observed.isoformat() if latest_observed else None
        ),
        "bounded": True,
        "max_files": int(max_files),
        "max_rows": int(max_rows),
        "max_bytes_per_file": int(max_bytes_per_file),
    }


def _features(row: Mapping[str, Any]) -> dict[str, Any]:
    value = row.get("features")
    return dict(value) if isinstance(value, Mapping) else {}


def _symbol(row: Mapping[str, Any]) -> str:
    return str(row.get("symbol") or "").strip().upper()


def _latest_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    latest: dict[str, Mapping[str, Any]] = {}
    latest_ts: dict[str, datetime] = {}
    for row in rows:
        symbol = _symbol(row)
        timestamp = _parse_ts(row.get("timestamp_utc"))
        if not symbol or timestamp is None:
            continue
        if symbol not in latest_ts or timestamp > latest_ts[symbol]:
            latest[symbol] = row
            latest_ts[symbol] = timestamp
    return latest


def _latest_observed(rows: Iterable[Mapping[str, Any]]) -> datetime | None:
    timestamps = [_parse_ts(row.get("timestamp_utc")) for row in rows]
    present = [value for value in timestamps if value is not None]
    return max(present) if present else None


def _return_value(features: Mapping[str, Any]) -> float | None:
    for key in ("pct_from_close", "return_1d", "mom_5m"):
        value = _safe_float(features.get(key))
        if value is not None:
            return value
    price = _safe_float(features.get("last_price"))
    previous = _safe_float(features.get("prev_close"))
    if price is not None and previous not in {None, 0.0}:
        return (price / float(previous)) - 1.0
    return None


def _minute_series(rows: Iterable[Mapping[str, Any]], symbol: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for row in rows:
        if _symbol(row) != symbol:
            continue
        timestamp = _parse_ts(row.get("timestamp_utc"))
        value = _return_value(_features(row))
        if timestamp is None or value is None:
            continue
        out[int(timestamp.timestamp() // 300)] = float(value)
    return out


def _correlation(left: Mapping[int, float], right: Mapping[int, float]) -> float | None:
    shared = sorted(set(left).intersection(right))
    if len(shared) < 5:
        return None
    xs = [float(left[key]) for key in shared]
    ys = [float(right[key]) for key in shared]
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = sum((x - mean_x) ** 2 for x in xs)
    denom_y = sum((y - mean_y) ** 2 for y in ys)
    denominator = math.sqrt(denom_x * denom_y)
    return numerator / denominator if denominator > 0.0 else None


def build_cross_asset_breadth_context(
    rows: list[dict[str, Any]], *, now: datetime
) -> dict[str, Any]:
    latest = _latest_rows(rows)
    returns: dict[str, float] = {}
    for symbol, row in latest.items():
        value = _return_value(_features(row))
        if value is not None:
            returns[symbol] = value
    for row in latest.values():
        for key, raw in _features(row).items():
            if not str(key).startswith("ctx_") or not str(key).endswith(
                "_pct_from_close"
            ):
                continue
            symbol = str(key)[4:-15].upper()
            value = _safe_float(raw)
            if symbol and value is not None:
                returns.setdefault(symbol, value)

    values = list(returns.values())
    advancers = sum(1 for value in values if value > 0.0)
    decliners = sum(1 for value in values if value < 0.0)
    unchanged = len(values) - advancers - decliners
    sector_returns = {
        key: value for key, value in returns.items() if key in SECTOR_SYMBOLS
    }
    factor_returns = {
        key: value for key, value in returns.items() if key in FACTOR_SYMBOLS
    }
    dispersion = statistics.pstdev(values) if len(values) >= 2 else None
    dislocation = (max(values) - min(values)) if len(values) >= 2 else None
    breadth_signed = (advancers - decliners) / max(advancers + decliners, 1)

    spy_series = _minute_series(rows, "SPY")
    correlations = [
        value
        for symbol in CORRELATION_SYMBOLS
        if (value := _correlation(spy_series, _minute_series(rows, symbol))) is not None
    ]
    correlation_mean = statistics.fmean(correlations) if correlations else None
    sector_spread = (
        max(sector_returns.values()) - min(sector_returns.values())
        if len(sector_returns) >= 2
        else None
    )
    factor_leader = max(factor_returns.values()) if factor_returns else None

    capabilities = {
        "market_breadth": (
            len(values) >= 6,
            {
                "distinct_symbol_count": len(values),
                "advancers": advancers,
                "decliners": decliners,
            },
        ),
        "cross_asset_dispersion": (
            dispersion is not None and len(values) >= 6,
            {"distinct_symbol_count": len(values), "dispersion": dispersion},
        ),
        "cross_asset_dislocation": (
            dislocation is not None and len(values) >= 6,
            {"distinct_symbol_count": len(values), "max_min_spread": dislocation},
        ),
        "sector_rotation": (
            len(sector_returns) >= 3,
            {
                "sector_symbol_count": len(sector_returns),
                "leader_laggard_spread": sector_spread,
            },
        ),
        "factor_leadership": (
            len(factor_returns) >= 3,
            {
                "factor_symbol_count": len(factor_returns),
                "leader_return": factor_leader,
            },
        ),
        "cross_asset_correlation": (
            len(correlations) >= 2,
            {"pair_count": len(correlations), "mean_correlation": correlation_mean},
        ),
        "risk_on_off_state": (
            len(values) >= 6,
            {
                "distinct_symbol_count": len(values),
                "advance_decline_signed": breadth_signed,
            },
        ),
    }
    global_features: dict[str, float] = {}
    if len(values) >= 6:
        global_features.update(
            {
                "research_cross_asset_available_norm": 1.0,
                "research_cross_asset_advance_decline_norm": _signed_norm(
                    breadth_signed, 1.0
                ),
                "research_risk_on_state_norm": _signed_norm(
                    statistics.fmean(values), 0.02
                ),
            }
        )
    if dispersion is not None:
        global_features["research_cross_asset_dispersion_norm"] = _clamp01(
            dispersion / 0.03
        )
    if dislocation is not None:
        global_features["research_cross_asset_dislocation_norm"] = _clamp01(
            dislocation / 0.08
        )
    if correlation_mean is not None:
        global_features["research_cross_asset_correlation_norm"] = _signed_norm(
            correlation_mean, 1.0
        )
    if sector_spread is not None:
        global_features["research_sector_rotation_norm"] = _clamp01(
            sector_spread / 0.05
        )
    if factor_leader is not None:
        global_features["research_factor_leadership_norm"] = _signed_norm(
            factor_leader, 0.03
        )
    symbol_features = {
        symbol: {
            "research_cross_asset_relative_strength_norm": _signed_norm(value, 0.05)
        }
        for symbol, value in returns.items()
    }
    payload = _capability_payload(
        "cross_asset_breadth_context",
        now=now,
        source_observed_at=_latest_observed(rows),
        capabilities=capabilities,
        derived={
            "global_features": global_features,
            "symbol_features": symbol_features,
        },
        metrics={
            "row_count": len(rows),
            "symbol_count": len(values),
            "advancers": advancers,
            "decliners": decliners,
            "unchanged": unchanged,
            "sector_symbol_count": len(sector_returns),
            "factor_symbol_count": len(factor_returns),
        },
        sources={"decision_tail": {"ok": bool(rows), "contract_participates": True}},
    )
    payload.update(
        {
            "advancers": advancers,
            "decliners": decliners,
            "sector_advancers": sum(
                1 for value in sector_returns.values() if value > 0.0
            ),
            "sector_decliners": sum(
                1 for value in sector_returns.values() if value < 0.0
            ),
            "sector_dispersion": (
                statistics.pstdev(sector_returns.values())
                if len(sector_returns) >= 2
                else None
            ),
            "sector_rotation_score": global_features.get(
                "research_sector_rotation_norm"
            ),
            "sector_leader_strength": (
                max(sector_returns.values()) if sector_returns else None
            ),
            "sector_laggard_strength": (
                min(sector_returns.values()) if sector_returns else None
            ),
            "index_alignment_score": global_features.get(
                "research_cross_asset_correlation_norm"
            ),
            "risk_on_score": global_features.get("research_risk_on_state_norm"),
        }
    )
    payload["snapshot_receipt_sha256"] = _canonical_hash(
        {
            key: value
            for key, value in payload.items()
            if key != "snapshot_receipt_sha256"
        }
    )
    return payload


def build_tape_liquidity_context(
    rows: list[dict[str, Any]], *, now: datetime
) -> dict[str, Any]:
    latest = _latest_rows(rows)
    symbol_features: dict[str, dict[str, float]] = {}
    spreads: list[float] = []
    quote_ages: list[float] = []
    realized_volatility: list[float] = []
    vwap_states: list[float] = []
    quality_scores: list[float] = []
    size_evidence = 0
    for symbol, row in latest.items():
        features = _features(row)
        derived: dict[str, float] = {}
        spread = next(
            (
                value
                for key in ("spread_bps", "futures_spread_bps", "expected_slippage_bps")
                if (value := _safe_float(features.get(key))) is not None
                and value >= 0.0
            ),
            None,
        )
        quote_age = _safe_float(features.get("quote_age_ms"))
        volatility_key = next(
            (
                key
                for key in (
                    "realized_volatility_norm",
                    "vol_30m",
                    "realized_volatility",
                )
                if (value := _safe_float(features.get(key))) is not None
                and value >= 0.0
            ),
            None,
        )
        volatility = (
            _safe_float(features.get(volatility_key)) if volatility_key else None
        )
        direct_vwap_state = next(
            (
                value
                for key in (
                    "market_micro_vwap_state_norm",
                    "vwap_state_norm",
                    "vwap_bias_norm",
                )
                if (value := _safe_float(features.get(key))) is not None
            ),
            None,
        )
        vwap_price = next(
            (
                value
                for key in ("vwap_60m", "vwap_30m", "session_vwap")
                if (value := _safe_float(features.get(key))) is not None and value > 0.0
            ),
            None,
        )
        last_price = _safe_float(features.get("last_price"))
        vwap_state = (
            _clamp01(direct_vwap_state)
            if direct_vwap_state is not None
            else (
                _signed_norm((last_price - vwap_price) / vwap_price, 0.03)
                if last_price is not None and vwap_price is not None
                else None
            )
        )
        route = (
            row.get("data_route") if isinstance(row.get("data_route"), Mapping) else {}
        )
        route_quality = _safe_float(route.get("source_quality_score"))
        quote_agreement = _safe_float(features.get("data_quality_quote_agreement_norm"))
        quality = (
            statistics.fmean(
                [
                    value
                    for value in (route_quality, quote_agreement)
                    if value is not None
                ]
            )
            if any(value is not None for value in (route_quality, quote_agreement))
            else None
        )
        bid_size = _safe_float(
            features.get("bid_size") or features.get("futures_bid_size")
        )
        ask_size = _safe_float(
            features.get("ask_size") or features.get("futures_ask_size")
        )
        if bid_size is not None and ask_size is not None and bid_size + ask_size > 0.0:
            size_evidence += 1
        if spread is not None:
            spreads.append(spread)
            derived["research_tape_spread_regime_norm"] = _clamp01(spread / 100.0)
        if quote_age is not None:
            quote_ages.append(quote_age)
            derived["research_tape_quote_freshness_norm"] = _clamp01(
                1.0 - quote_age / 30_000.0
            )
        if volatility is not None:
            normalized_volatility = _clamp01(
                volatility
                if volatility_key == "realized_volatility_norm"
                else volatility / 0.08
            )
            realized_volatility.append(normalized_volatility)
            derived["research_tape_realized_volatility_norm"] = normalized_volatility
        if vwap_state is not None:
            vwap_states.append(vwap_state)
            derived["research_tape_vwap_state_norm"] = vwap_state
        if quality is not None:
            quality_scores.append(quality)
            derived["research_tape_quality_norm"] = _clamp01(quality)
        if derived:
            symbol_features[symbol] = derived

    global_features: dict[str, float] = {}
    if symbol_features:
        global_features["research_tape_available_norm"] = 1.0
    if spreads:
        global_features["research_tape_spread_regime_norm"] = _clamp01(
            statistics.median(spreads) / 100.0
        )
    if quote_ages:
        global_features["research_tape_quote_freshness_norm"] = _clamp01(
            1.0 - statistics.median(quote_ages) / 30_000.0
        )
    if realized_volatility:
        global_features["research_tape_realized_volatility_norm"] = _clamp01(
            statistics.median(realized_volatility)
        )
    if vwap_states:
        global_features["research_tape_vwap_state_norm"] = _clamp01(
            statistics.median(vwap_states)
        )
    if quality_scores:
        global_features["research_tape_quality_norm"] = _clamp01(
            statistics.fmean(quality_scores)
        )
    if spreads and size_evidence:
        global_features["research_tape_liquidity_regime_norm"] = _clamp01(
            0.65 * (1.0 - global_features["research_tape_spread_regime_norm"])
            + 0.35 * min(size_evidence / max(len(latest), 1), 1.0)
        )
    capabilities = {
        "bid_ask_spread": (len(spreads) >= 3, {"symbol_count": len(spreads)}),
        "consolidated_tape_quality": (
            len(quality_scores) >= 3,
            {"symbol_count": len(quality_scores)},
        ),
        "quote_age": (len(quote_ages) >= 3, {"symbol_count": len(quote_ages)}),
        "realized_volatility": (
            len(realized_volatility) >= 3,
            {"symbol_count": len(realized_volatility)},
        ),
        "vwap_state": (len(vwap_states) >= 2, {"symbol_count": len(vwap_states)}),
        "liquidity_regime": (
            len(spreads) >= 3 and size_evidence >= 2,
            {"spread_symbol_count": len(spreads), "size_symbol_count": size_evidence},
        ),
    }
    return _capability_payload(
        "tape_liquidity_context",
        now=now,
        source_observed_at=_latest_observed(rows),
        capabilities=capabilities,
        derived={
            "global_features": global_features,
            "symbol_features": symbol_features,
        },
        metrics={
            "row_count": len(rows),
            "symbol_count": len(latest),
            "spread_symbol_count": len(spreads),
            "size_symbol_count": size_evidence,
        },
        sources={"decision_tail": {"ok": bool(rows), "contract_participates": True}},
        warnings=("level2_order_book_not_available_or_inferred",),
    )


def _feature_presence(
    rows: Iterable[Mapping[str, Any]], keys: Iterable[str]
) -> list[tuple[Mapping[str, Any], dict[str, Any]]]:
    required = tuple(keys)
    return [
        (row, _features(row))
        for row in rows
        if any(key in _features(row) for key in required)
    ]


def build_options_greeks_surface_context(
    rows: list[dict[str, Any]], *, now: datetime
) -> dict[str, Any]:
    latest = _latest_rows(rows)
    chain_rows = [
        (symbol, _features(row))
        for symbol, row in latest.items()
        if (_safe_float(_features(row).get("options_chain_available")) or 0.0) >= 0.5
        and any(
            key in _features(row)
            for key in (
                "options_contract_count",
                "options_contract_count_norm",
                "options_iv_atm",
            )
        )
    ]
    key_groups = {
        "implied_volatility_surface": ("options_iv_atm", "options_iv_mean"),
        "volatility_skew": ("options_iv_skew", "options_iv_skew_norm"),
        "volatility_term_structure": (
            "options_iv_term_structure",
            "options_iv_term_structure_norm",
        ),
        "option_greeks": (
            "options_delta_abs_mean_norm",
            "options_gamma_mean_norm",
            "options_theta_abs_mean_norm",
            "options_vega_mean_norm",
        ),
        "option_open_interest": (
            "options_open_interest_total",
            "options_open_interest_norm",
            "options_oi_concentration_norm",
        ),
        "realized_volatility": (
            "options_realized_volatility",
            "options_realized_volatility_norm",
            "vol_30m",
        ),
        "volatility_risk_premium": (
            "options_iv_realized_spread",
            "options_iv_realized_spread_norm",
            "volatility_risk_premium_norm",
        ),
    }
    capabilities: dict[str, tuple[bool, Mapping[str, Any]]] = {
        "options_chain": (
            len(chain_rows) >= 1,
            {"symbols_with_direct_chain": len(chain_rows)},
        )
    }
    for capability_id, keys in key_groups.items():
        count = sum(
            1 for _, features in chain_rows if any(key in features for key in keys)
        )
        capabilities[capability_id] = (
            count >= 1,
            {"symbols_with_field_evidence": count, "accepted_fields": list(keys)},
        )

    symbol_features: dict[str, dict[str, float]] = {}
    source_to_feature = {
        "options_iv_atm_norm": "research_options_iv_surface_norm",
        "options_iv_skew_norm": "research_options_skew_norm",
        "options_iv_term_structure_norm": "research_options_term_structure_norm",
        "options_open_interest_norm": "research_options_open_interest_norm",
        "options_iv_realized_spread_norm": "research_options_vrp_norm",
    }
    for symbol, features in chain_rows:
        derived = {
            target: _clamp01(value)
            for source, target in source_to_feature.items()
            if (value := _safe_float(features.get(source))) is not None
        }
        greek_values = [
            value
            for key in (
                "options_delta_abs_mean_norm",
                "options_gamma_mean_norm",
                "options_theta_abs_mean_norm",
                "options_vega_mean_norm",
            )
            if (value := _safe_float(features.get(key))) is not None
        ]
        if greek_values:
            derived["research_options_greeks_norm"] = _clamp01(
                statistics.fmean(greek_values)
            )
        derived["research_options_available_norm"] = 1.0
        symbol_features[symbol] = derived
    global_features: dict[str, float] = {}
    for key in COLLECTOR_BY_ID["options_greeks_surface_context"]["feature_keys"]:
        values = [
            features[key] for features in symbol_features.values() if key in features
        ]
        if values:
            global_features[key] = _clamp01(statistics.fmean(values))
    return _capability_payload(
        "options_greeks_surface_context",
        now=now,
        source_observed_at=_latest_observed(rows),
        capabilities=capabilities,
        derived={
            "global_features": global_features,
            "symbol_features": symbol_features,
        },
        metrics={"row_count": len(rows), "symbols_with_direct_chain": len(chain_rows)},
        sources={
            "broker_decision_chain_features": {
                "ok": bool(chain_rows),
                "contract_participates": True,
            }
        },
    )


def build_futures_curve_context(
    rows: list[dict[str, Any]], *, now: datetime
) -> dict[str, Any]:
    latest = _latest_rows(rows)
    futures_rows: list[tuple[str, dict[str, Any]]] = []
    for symbol, row in latest.items():
        route = (
            row.get("data_route") if isinstance(row.get("data_route"), Mapping) else {}
        )
        lane = str(route.get("routing_lane") or "").lower()
        if (
            str(row.get("asset_class") or "").lower() != "futures"
            and "futures" not in lane
        ):
            continue
        features = _features(row)
        anchors = [
            _safe_float(features.get(key))
            for key in (
                "futures_spread_bps",
                "futures_bid_size",
                "futures_ask_size",
                "futures_open_interest",
                "futures_expiry_days",
                "futures_basis_bps",
            )
        ]
        if not any(value is not None and abs(value) > 0.0 for value in anchors):
            continue
        futures_rows.append((symbol, features))
    key_groups = {
        "futures_term_structure": ("futures_term_structure_norm",),
        "futures_basis": ("futures_basis_bps", "futures_basis_bps_norm"),
        "roll_yield": ("futures_roll_yield_norm",),
        "futures_open_interest": (
            "futures_open_interest",
            "futures_open_interest_norm",
        ),
        "futures_volume_migration": (
            "futures_session_volume_profile_norm",
            "futures_volume_migration_norm",
        ),
        "expiry_state": ("futures_expiry_days", "futures_expiry_norm"),
        "carry_state": (
            "futures_roll_yield_norm",
            "futures_basis_bps_norm",
            "futures_basis_bps",
        ),
        "calendar_spreads": ("futures_calendar_spread_curve_norm",),
        "spot_futures_dislocation": (
            "futures_basis_dislocation_norm",
            "futures_mark_index_dislocation_norm",
        ),
    }
    capabilities: dict[str, tuple[bool, Mapping[str, Any]]] = {}
    for capability_id, keys in key_groups.items():
        count = sum(
            1
            for _, features in futures_rows
            if any(
                (value := _safe_float(features.get(key))) is not None
                and abs(value) > 1e-12
                for key in keys
            )
        )
        capabilities[capability_id] = (
            count >= 1,
            {
                "symbols_with_direct_futures_anchor": len(futures_rows),
                "symbols_with_nonzero_field_evidence": count,
                "default_zero_fields_do_not_prove_capability": True,
            },
        )
    source_to_feature = {
        "futures_term_structure_norm": "research_futures_term_structure_norm",
        "futures_basis_bps_norm": "research_futures_basis_norm",
        "futures_roll_yield_norm": "research_futures_roll_yield_norm",
        "futures_open_interest_norm": "research_futures_open_interest_norm",
        "futures_session_volume_profile_norm": "research_futures_volume_migration_norm",
        "futures_expiry_norm": "research_futures_expiry_norm",
        "futures_calendar_spread_curve_norm": "research_futures_calendar_spread_norm",
    }
    symbol_features: dict[str, dict[str, float]] = {}
    for symbol, features in futures_rows:
        derived = {
            target: _clamp01(value)
            for source, target in source_to_feature.items()
            if (value := _safe_float(features.get(source))) is not None
            and abs(value) > 1e-12
        }
        if "research_futures_basis_norm" not in derived:
            if (
                basis_bps := _safe_float(features.get("futures_basis_bps"))
            ) is not None and abs(basis_bps) > 1e-12:
                derived["research_futures_basis_norm"] = _signed_norm(basis_bps, 100.0)
        if "research_futures_open_interest_norm" not in derived:
            if (
                open_interest := _safe_float(features.get("futures_open_interest"))
            ) is not None and open_interest > 0.0:
                derived["research_futures_open_interest_norm"] = _clamp01(
                    math.log1p(open_interest) / 20.0
                )
        if "research_futures_expiry_norm" not in derived:
            if (
                expiry_days := _safe_float(features.get("futures_expiry_days"))
            ) is not None and expiry_days > 0.0:
                derived["research_futures_expiry_norm"] = _clamp01(expiry_days / 365.0)
        derived["research_futures_available_norm"] = 1.0
        symbol_features[symbol] = derived
    global_features = {
        key: _clamp01(statistics.fmean(values))
        for key in COLLECTOR_BY_ID["futures_curve_context"]["feature_keys"]
        if (
            values := [
                features[key]
                for features in symbol_features.values()
                if key in features
            ]
        )
    }
    return _capability_payload(
        "futures_curve_context",
        now=now,
        source_observed_at=_latest_observed(rows),
        capabilities=capabilities,
        derived={
            "global_features": global_features,
            "symbol_features": symbol_features,
        },
        metrics={
            "row_count": len(rows),
            "symbols_with_direct_futures_anchor": len(futures_rows),
        },
        sources={
            "broker_futures_decision_features": {
                "ok": bool(futures_rows),
                "contract_participates": True,
            }
        },
        warnings=(
            ()
            if futures_rows
            else ("futures_fields_present_but_no_nonzero_direct_market_anchor",)
        ),
    )


def build_earnings_event_context(
    rows: list[dict[str, Any]],
    *,
    now: datetime,
    sec_payload: Mapping[str, Any],
    analyst_payload: Mapping[str, Any],
) -> dict[str, Any]:
    sec_rows = (
        sec_payload.get("symbol_rows")
        if isinstance(sec_payload.get("symbol_rows"), list)
        else []
    )
    analyst_symbols = (
        analyst_payload.get("symbols")
        if isinstance(analyst_payload.get("symbols"), Mapping)
        else {}
    )
    calendar_evidence = any(
        (_safe_float(_features(row).get("calendar_feed_available")) or 0.0) > 0.0
        and "calendar_earnings_7d_norm" in _features(row)
        for row in rows
    )
    sec_ok = bool(
        (sec_payload.get("status") or {}).get("ok", sec_payload.get("ok", False))
    )
    analyst_ok = bool(analyst_payload.get("ok", False))
    reported_fields = sum(
        1 for row in sec_rows if isinstance(row, Mapping) and "earnings_7d" in row
    )
    guidance_fields = sum(
        1 for row in sec_rows if isinstance(row, Mapping) and "guidance_7d" in row
    )
    estimate_rows = sum(
        len(value.get("estimates") or [])
        for value in analyst_symbols.values()
        if isinstance(value, Mapping) and isinstance(value.get("estimates"), list)
    )
    analyst_features = (
        (analyst_payload.get("derived") or {}).get("symbol_features")
        if isinstance(analyst_payload.get("derived"), Mapping)
        else {}
    )
    analyst_features = analyst_features if isinstance(analyst_features, Mapping) else {}
    revision_fields = sum(
        1
        for value in analyst_features.values()
        if isinstance(value, Mapping)
        and _safe_float(value.get("consensus_revision_direction_norm")) is not None
    )
    dispersion_fields = sum(
        1
        for value in analyst_features.values()
        if isinstance(value, Mapping)
        and _safe_float(value.get("consensus_dispersion_norm")) is not None
    )
    capabilities = {
        "earnings_calendar": (
            calendar_evidence,
            {"calendar_feed_evidence": calendar_evidence},
        ),
        "reported_earnings": (
            sec_ok and reported_fields > 0,
            {"sec_symbol_field_count": reported_fields},
        ),
        "company_guidance": (
            sec_ok and guidance_fields > 0,
            {"sec_symbol_field_count": guidance_fields},
        ),
        "estimate_revisions": (
            analyst_ok and revision_fields > 0,
            {
                "estimate_row_count": estimate_rows,
                "symbols_with_revision_field": revision_fields,
            },
        ),
        "estimate_dispersion": (
            analyst_ok and dispersion_fields > 0,
            {
                "estimate_row_count": estimate_rows,
                "symbols_with_dispersion_field": dispersion_fields,
            },
        ),
    }
    sec_by_symbol = {
        str(row.get("symbol") or "").upper(): row
        for row in sec_rows
        if isinstance(row, Mapping) and str(row.get("symbol") or "")
    }
    symbol_features: dict[str, dict[str, float]] = {}
    for symbol in sorted(set(sec_by_symbol).union(analyst_symbols)):
        derived: dict[str, float] = {}
        sec_row = sec_by_symbol.get(symbol, {})
        if "earnings_7d" in sec_row:
            derived["research_earnings_report_activity_norm"] = _clamp01(
                float(sec_row.get("earnings_7d", 0) or 0) / 4.0
            )
        if "guidance_7d" in sec_row:
            derived["research_earnings_guidance_activity_norm"] = _clamp01(
                float(sec_row.get("guidance_7d", 0) or 0) / 4.0
            )
        raw_features = (
            analyst_features.get(symbol)
            if isinstance(analyst_features.get(symbol), Mapping)
            else {}
        )
        if (
            value := _safe_float(raw_features.get("consensus_dispersion_norm"))
        ) is not None:
            derived["research_estimate_dispersion_norm"] = _clamp01(value)
        if (
            value := _safe_float(raw_features.get("consensus_revision_direction_norm"))
        ) is not None:
            derived["research_estimate_revision_direction_norm"] = _clamp01(value)
        if derived:
            derived["research_earnings_available_norm"] = 1.0
            symbol_features[symbol] = derived
    latest_decision = _latest_rows(rows)
    proximity_values = [
        value
        for row in latest_decision.values()
        if (value := _safe_float(_features(row).get("calendar_earnings_7d_norm")))
        is not None
    ]
    global_features: dict[str, float] = {}
    if symbol_features:
        global_features["research_earnings_available_norm"] = 1.0
    if proximity_values:
        global_features["research_earnings_calendar_proximity_norm"] = _clamp01(
            max(proximity_values)
        )
    for key in (
        "research_earnings_report_activity_norm",
        "research_earnings_guidance_activity_norm",
        "research_estimate_dispersion_norm",
        "research_estimate_revision_direction_norm",
    ):
        values = [
            features[key] for features in symbol_features.values() if key in features
        ]
        if values:
            global_features[key] = _clamp01(statistics.fmean(values))
    observed = max(
        [
            value
            for value in (
                _parse_ts(sec_payload.get("timestamp_utc")),
                _parse_ts(analyst_payload.get("timestamp_utc")),
                _latest_observed(rows),
            )
            if value is not None
        ],
        default=None,
    )
    return _capability_payload(
        "earnings_event_context",
        now=now,
        source_observed_at=observed,
        capabilities=capabilities,
        derived={
            "global_features": global_features,
            "symbol_features": symbol_features,
        },
        metrics={
            "sec_symbol_count": len(sec_rows),
            "analyst_symbol_count": len(analyst_symbols),
            "estimate_row_count": estimate_rows,
        },
        sources={
            "sec_edgar_context": {"ok": sec_ok, "contract_participates": True},
            "analyst_consensus_context": {
                "ok": analyst_ok,
                "contract_participates": True,
            },
            "decision_calendar_features": {
                "ok": calendar_evidence,
                "contract_participates": True,
            },
        },
        warnings=("earnings_surprise_not_claimed_without_aligned_actual_and_estimate",),
    )


def build_portfolio_factor_risk_context(
    rows: list[dict[str, Any]],
    *,
    now: datetime,
    account_payload: Mapping[str, Any],
) -> dict[str, Any]:
    positions = (
        account_payload.get("positions")
        if isinstance(account_payload.get("positions"), list)
        else []
    )
    clean_positions = [
        row
        for row in positions
        if isinstance(row, Mapping) and str(row.get("symbol") or "")
    ]
    market_values = [
        abs(value)
        for row in clean_positions
        if (value := _safe_float(row.get("market_value"))) is not None
    ]
    gross = sum(market_values)
    signed = sum(
        (_safe_float(row.get("market_value")) or 0.0) for row in clean_positions
    )
    weights = [value / gross for value in market_values] if gross > 0.0 else []
    concentration = sum(weight * weight for weight in weights)

    symbol_series = {
        str(row.get("underlying") or row.get("symbol") or "").upper(): _minute_series(
            rows, str(row.get("underlying") or row.get("symbol") or "").upper()
        )
        for row in clean_positions
    }
    spy = _minute_series(rows, "SPY")
    correlations = [
        value
        for series in symbol_series.values()
        if (value := _correlation(spy, series)) is not None
    ]
    factor_correlations: list[float] = []
    for factor in FACTOR_SYMBOLS - {"SPY"}:
        factor_series = _minute_series(rows, factor)
        factor_correlations.extend(
            value
            for series in symbol_series.values()
            if (value := _correlation(factor_series, series)) is not None
        )
    latest = _latest_rows(rows)
    held_symbols = {
        str(row.get("underlying") or row.get("symbol") or "").upper()
        for row in clean_positions
    }
    held_spreads = [
        value
        for symbol in held_symbols
        if symbol in latest
        if (value := _safe_float(_features(latest[symbol]).get("spread_bps")))
        is not None
        and value >= 0.0
    ]
    capabilities = {
        "portfolio_exposure": (
            bool(clean_positions) and gross > 0.0,
            {"position_count": len(clean_positions), "gross_market_value": gross},
        ),
        "concentration_risk": (
            bool(weights),
            {"position_count": len(weights), "herfindahl_index": concentration},
        ),
        "portfolio_beta": (
            len(correlations) >= 2,
            {"position_pair_count": len(correlations)},
        ),
        "factor_exposure": (
            len(factor_correlations) >= 3,
            {"factor_pair_count": len(factor_correlations)},
        ),
        "correlation_risk": (
            len(correlations) >= 2,
            {"position_pair_count": len(correlations)},
        ),
        "portfolio_liquidity": (
            len(held_spreads) >= 2,
            {"held_symbol_spread_count": len(held_spreads)},
        ),
    }
    global_features: dict[str, float] = {}
    if gross > 0.0:
        global_features.update(
            {
                "research_portfolio_available_norm": 1.0,
                "research_portfolio_concentration_norm": _clamp01(concentration * 4.0),
                "research_portfolio_net_exposure_norm": _signed_norm(
                    signed / gross, 1.0
                ),
            }
        )
    if correlations:
        mean_correlation = statistics.fmean(correlations)
        global_features["research_portfolio_beta_norm"] = _signed_norm(
            mean_correlation, 1.0
        )
        global_features["research_portfolio_correlation_risk_norm"] = _clamp01(
            statistics.fmean(abs(value) for value in correlations)
        )
    if factor_correlations:
        global_features["research_portfolio_factor_exposure_norm"] = _clamp01(
            statistics.fmean(abs(value) for value in factor_correlations)
        )
    if held_spreads:
        global_features["research_portfolio_liquidity_norm"] = _clamp01(
            1.0 - statistics.median(held_spreads) / 100.0
        )
    symbol_features = {
        str(row.get("underlying") or row.get("symbol") or "").upper(): {
            "research_portfolio_weight_norm": _clamp01(
                abs(_safe_float(row.get("market_value")) or 0.0) / gross
            )
        }
        for row in clean_positions
        if gross > 0.0
    }
    return _capability_payload(
        "portfolio_factor_risk_context",
        now=now,
        source_observed_at=max(
            [
                value
                for value in (
                    _parse_ts(account_payload.get("timestamp_utc")),
                    _latest_observed(rows),
                )
                if value is not None
            ],
            default=None,
        ),
        capabilities=capabilities,
        derived={
            "global_features": global_features,
            "symbol_features": symbol_features,
        },
        metrics={
            "position_count": len(clean_positions),
            "gross_market_value": round(gross, 4),
            "net_market_value": round(signed, 4),
            "held_symbol_spread_count": len(held_spreads),
        },
        sources={
            "broker_account_position_study": {
                "ok": bool(account_payload.get("ok", False)),
                "contract_participates": True,
            },
            "decision_tail": {"ok": bool(rows), "contract_participates": True},
        },
    )


def _http_json(
    url: str,
    *,
    timeout_seconds: float,
    max_bytes: int = 8_388_608,
    headers: Mapping[str, str] | None = None,
) -> Any:
    request_headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    request_headers.update(dict(headers or {}))
    request = urllib.request.Request(url, headers=request_headers)
    with urllib.request.urlopen(
        request, timeout=max(float(timeout_seconds), 1.0)
    ) as response:
        declared = int(response.headers.get("Content-Length", "0") or 0)
        if declared > max_bytes:
            raise ValueError(f"response_too_large:{declared}>{max_bytes}")
        raw = response.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError(f"response_too_large:{len(raw)}>{max_bytes}")
    return json.loads(raw.decode("utf-8"))


def _http_bytes(
    url: str, *, timeout_seconds: float, max_bytes: int = 67_108_864
) -> bytes:
    request = urllib.request.Request(
        url, headers={"Accept": "application/zip", "User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(
        request, timeout=max(float(timeout_seconds), 1.0)
    ) as response:
        declared = int(response.headers.get("Content-Length", "0") or 0)
        if declared > max_bytes:
            raise ValueError(f"response_too_large:{declared}>{max_bytes}")
        raw = response.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError(f"response_too_large:{len(raw)}>{max_bytes}")
    return raw


def _date_from_row(row: Mapping[str, Any]) -> str:
    for key in (
        "tradeReportDate",
        "tradeDate",
        "reportDate",
        "beginningOfTheMonthDate",
    ):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def build_fixed_income_trace_context(
    *,
    now: datetime,
    fetch_json: Callable[..., Any] = _http_json,
    timeout_seconds: float = 8.0,
    access_token: str | None = None,
) -> dict[str, Any]:
    datasets: dict[str, list[dict[str, Any]]] = {}
    errors: list[str] = []
    token = str(
        access_token
        if access_token is not None
        else os.getenv("FINRA_API_ACCESS_TOKEN", "")
    ).strip()
    if not token:
        errors.append("finra_public_api_credential_required")
        datasets = {"corporateMarketBreadth": [], "treasuryDailyAggregates": []}
    else:
        for dataset in ("corporateMarketBreadth", "treasuryDailyAggregates"):
            url = f"{FINRA_BASE_URL}/{dataset}?{urllib.parse.urlencode({'limit': 500})}"
            try:
                raw = fetch_json(
                    url,
                    timeout_seconds=timeout_seconds,
                    headers={"Authorization": f"Bearer {token}"},
                )
                datasets[dataset] = (
                    [row for row in raw if isinstance(row, dict)]
                    if isinstance(raw, list)
                    else []
                )
            except Exception as exc:
                datasets[dataset] = []
                errors.append(f"{dataset}:{type(exc).__name__}:{exc}")

    corporate = datasets["corporateMarketBreadth"]
    treasury = datasets["treasuryDailyAggregates"]
    latest_corporate_date = max((_date_from_row(row) for row in corporate), default="")
    latest_treasury_date = max((_date_from_row(row) for row in treasury), default="")
    corporate_latest = [
        row for row in corporate if _date_from_row(row) == latest_corporate_date
    ]
    treasury_latest = [
        row for row in treasury if _date_from_row(row) == latest_treasury_date
    ]
    advances = sum(
        (_safe_float(row.get("advances")) or 0.0) for row in corporate_latest
    )
    declines = sum(
        (_safe_float(row.get("declines")) or 0.0) for row in corporate_latest
    )
    trade_count = sum(
        (_safe_float(row.get("totalTradeCount") or row.get("totalTrades")) or 0.0)
        for row in corporate_latest
    )
    treasury_volume = sum(
        (_safe_float(row.get("dealerCustomerVolume")) or 0.0)
        + (_safe_float(row.get("atsInterdealerVolume")) or 0.0)
        for row in treasury_latest
    )
    breadth_signed = (advances - declines) / max(advances + declines, 1.0)
    corporate_by_date: dict[str, float] = defaultdict(float)
    for row in corporate:
        corporate_by_date[_date_from_row(row)] += (
            _safe_float(row.get("totalTradeCount") or row.get("totalTrades")) or 0.0
        )
    history_counts = [
        value for key, value in sorted(corporate_by_date.items()) if key and value > 0.0
    ]
    turnover_ratio = (
        trade_count / statistics.median(history_counts[:-1] or history_counts)
        if history_counts
        and statistics.median(history_counts[:-1] or history_counts) > 0.0
        else None
    )

    capabilities = {
        "market_breadth": (
            bool(corporate_latest) and advances + declines > 0.0,
            {
                "latest_date": latest_corporate_date,
                "record_count": len(corporate_latest),
            },
        ),
        "turnover": (
            bool(treasury_latest) or trade_count > 0.0,
            {
                "treasury_record_count": len(treasury_latest),
                "corporate_trade_count": trade_count,
            },
        ),
        "liquidity_regime": (
            bool(treasury_latest) and treasury_volume > 0.0,
            {"latest_date": latest_treasury_date, "aggregate_volume": treasury_volume},
        ),
        "rates_credit_regime": (
            bool(corporate_latest) and advances + declines > 0.0,
            {"latest_date": latest_corporate_date, "breadth_signed": breadth_signed},
        ),
    }
    global_features: dict[str, float] = {}
    if corporate_latest or treasury_latest:
        global_features["research_trace_available_norm"] = 1.0
    if corporate_latest:
        global_features["research_trace_breadth_norm"] = _signed_norm(
            breadth_signed, 1.0
        )
        global_features["research_trace_credit_regime_norm"] = _signed_norm(
            breadth_signed, 1.0
        )
    if turnover_ratio is not None:
        global_features["research_trace_turnover_norm"] = _clamp01(turnover_ratio / 2.0)
    if treasury_volume > 0.0:
        global_features["research_trace_liquidity_norm"] = _clamp01(
            math.log1p(treasury_volume) / 10.0
        )
    observed = max(
        [
            value
            for value in (
                _parse_ts(latest_corporate_date),
                _parse_ts(latest_treasury_date),
            )
            if value is not None
        ],
        default=None,
    )
    return _capability_payload(
        "fixed_income_trace_context",
        now=now,
        source_observed_at=observed,
        capabilities=capabilities,
        derived={"global_features": global_features, "symbol_features": {}},
        metrics={
            "row_count": len(corporate) + len(treasury),
            "corporate_latest_date": latest_corporate_date,
            "treasury_latest_date": latest_treasury_date,
            "corporate_advances": advances,
            "corporate_declines": declines,
            "corporate_trade_count": trade_count,
            "treasury_aggregate_volume": treasury_volume,
        },
        sources={
            "finra_corporate_market_breadth": {
                "ok": bool(corporate),
                "contract_participates": True,
                "credential_configured": bool(token),
                "official_url": f"{FINRA_BASE_URL}/corporateMarketBreadth",
            },
            "finra_treasury_daily_aggregates": {
                "ok": bool(treasury),
                "contract_participates": True,
                "credential_configured": bool(token),
                "official_url": f"{FINRA_BASE_URL}/treasuryDailyAggregates",
            },
        },
        warnings=errors,
    )


def _period_datetime(period: str) -> datetime | None:
    text = str(period or "").strip().upper()
    for suffix, month, day in (
        ("-Q1", 3, 31),
        ("-Q2", 6, 30),
        ("-Q3", 9, 30),
        ("-Q4", 12, 31),
    ):
        if text.endswith(suffix) and len(text) >= 7:
            try:
                return datetime(int(text[:4]), month, day, tzinfo=timezone.utc)
            except Exception:
                return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        try:
            return datetime(int(text[:4]), 12, 31, tzinfo=timezone.utc)
        except Exception:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_bis_global_liquidity_zip(
    raw: bytes, *, max_rows: int = 250_000, max_series: int = 50_000
) -> dict[str, Any]:
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        candidates = [
            info
            for info in archive.infolist()
            if info.filename.lower().endswith(".csv")
        ]
        if not candidates:
            raise ValueError("bis_zip_missing_csv")
        info = max(candidates, key=lambda item: item.file_size)
        with archive.open(info) as handle:
            reader = csv.DictReader(
                io.TextIOWrapper(
                    handle, encoding="utf-8-sig", errors="replace", newline=""
                )
            )
            fieldnames = [str(value or "") for value in (reader.fieldnames or [])]
            by_code = {
                name.split(":", 1)[0].strip().upper(): name for name in fieldnames
            }
            time_field = next(
                (by_code[key] for key in ("TIME_PERIOD", "TIME") if key in by_code),
                "",
            )
            value_field = next(
                (by_code[key] for key in ("OBS_VALUE", "VALUE") if key in by_code), ""
            )
            if not time_field or not value_field:
                raise ValueError("bis_flat_csv_missing_time_or_value")
            ignored = {time_field, value_field}
            ignored.update(
                name
                for name in fieldnames
                if name.split(":", 1)[0].strip().upper().startswith("OBS_")
                or name.split(":", 1)[0].strip().upper()
                in {"CONF_STATUS", "DECIMALS"}
            )
            dimension_fields = [name for name in fieldnames if name not in ignored]
            series: dict[tuple[str, ...], list[tuple[str, float]]] = {}
            periods: set[str] = set()
            country_values: set[str] = set()
            currency_values: set[str] = set()
            row_count = 0
            for row in reader:
                row_count += 1
                if row_count > max_rows:
                    break
                period = str(row.get(time_field) or "").strip()
                value = _safe_float(row.get(value_field))
                if not period or value is None:
                    continue
                key = tuple(str(row.get(name) or "") for name in dimension_fields)
                if key not in series and len(series) >= max_series:
                    continue
                points = series.setdefault(key, [])
                points.append((period, value))
                points[:] = sorted(points, key=lambda item: item[0], reverse=True)[:2]
                periods.add(period)
                for name in dimension_fields:
                    upper = name.upper()
                    token = str(row.get(name) or "").strip()
                    if "CTY" in upper or "COUNTRY" in upper:
                        country_values.add(token)
                    if "CURRENCY" in upper:
                        currency_values.add(token)
    growth_rates: list[float] = []
    for points in series.values():
        if len(points) < 2 or points[1][1] == 0.0:
            continue
        growth_rates.append((points[0][1] - points[1][1]) / abs(points[1][1]))
    latest_period = max(periods, default="")
    median_growth = statistics.median(growth_rates) if growth_rates else None
    return {
        "row_count": min(row_count, max_rows),
        "series_count": len(series),
        "growth_pair_count": len(growth_rates),
        "latest_period": latest_period,
        "median_series_growth": median_growth,
        "country_count": len({value for value in country_values if value}),
        "currency_count": len({value for value in currency_values if value}),
        "csv_member": info.filename,
        "bounded": True,
        "max_rows": int(max_rows),
        "max_series": int(max_series),
    }


def build_bis_global_liquidity_context(
    *,
    now: datetime,
    fetch_bytes: Callable[..., bytes] = _http_bytes,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    raw = fetch_bytes(BIS_GLI_URL, timeout_seconds=timeout_seconds)
    parsed = parse_bis_global_liquidity_zip(raw)
    sufficient = parsed["series_count"] >= 10 and bool(parsed["latest_period"])
    growth_ready = (
        parsed["growth_pair_count"] >= 5 and parsed["median_series_growth"] is not None
    )
    capabilities = {
        "global_liquidity_regime": (
            growth_ready,
            {
                "series_count": parsed["series_count"],
                "growth_pair_count": parsed["growth_pair_count"],
            },
        ),
        "cross_border_capital_flows": (
            sufficient,
            {
                "series_count": parsed["series_count"],
                "country_count": parsed["country_count"],
            },
        ),
        "bank_credit_conditions": (
            growth_ready,
            {
                "series_count": parsed["series_count"],
                "growth_pair_count": parsed["growth_pair_count"],
            },
        ),
    }
    global_features: dict[str, float] = {}
    if sufficient:
        global_features.update(
            {
                "research_bis_available_norm": 1.0,
                "research_bis_country_coverage_norm": _clamp01(
                    parsed["country_count"] / 50.0
                ),
            }
        )
    if growth_ready:
        growth_norm = _signed_norm(float(parsed["median_series_growth"]), 0.20)
        global_features.update(
            {
                "research_bis_global_liquidity_impulse_norm": growth_norm,
                "research_bis_cross_border_flow_norm": growth_norm,
                "research_bis_bank_credit_conditions_norm": growth_norm,
            }
        )
    return _capability_payload(
        "bis_global_liquidity_context",
        now=now,
        source_observed_at=_period_datetime(parsed["latest_period"]),
        capabilities=capabilities,
        derived={"global_features": global_features, "symbol_features": {}},
        metrics=parsed,
        sources={
            "bis_global_liquidity_indicators": {
                "ok": sufficient,
                "contract_participates": True,
                "official_url": BIS_GLI_URL,
            }
        },
    )


def _write_success(
    project_root: Path, collector_id: str, payload: Mapping[str, Any]
) -> None:
    _atomic_write_json(_artifact_path(project_root, collector_id), payload)
    _atomic_write_json(_health_path(project_root, collector_id), payload)
    if collector_id == "cross_asset_breadth_context":
        _atomic_write_json(
            project_root / "data" / "external_context" / "market_breadth_latest.json",
            payload,
        )


def _write_failure_health(
    project_root: Path,
    collector_id: str,
    *,
    now: datetime,
    previous: Mapping[str, Any],
    error: Exception,
) -> dict[str, Any]:
    if previous:
        health = dict(previous)
        health["last_attempt_utc"] = now.isoformat()
        health["serving_last_good"] = True
        health["refresh_error"] = f"{type(error).__name__}:{error}"
        warnings = list(health.get("warnings") or [])
        warnings.append("refresh_failed_serving_bounded_last_good")
        health["warnings"] = sorted(set(str(item) for item in warnings))
        _atomic_write_json(_health_path(project_root, collector_id), health)
        return health
    payload = _capability_payload(
        collector_id,
        now=now,
        source_observed_at=None,
        capabilities={},
        derived={"global_features": {}, "symbol_features": {}},
        metrics={"row_count": 0},
        sources={"refresh_attempt": {"ok": False, "contract_participates": True}},
        warnings=(f"refresh_failed:{type(error).__name__}:{error}",),
    )
    _atomic_write_json(_health_path(project_root, collector_id), payload)
    return payload


def collect_one(
    collector_id: str,
    *,
    project_root: Path,
    now: datetime,
    decisions: list[dict[str, Any]],
    force: bool,
    offline: bool,
    timeout_seconds: float,
) -> dict[str, Any]:
    definition = COLLECTOR_BY_ID[collector_id]
    artifact_path = _artifact_path(project_root, collector_id)
    previous = _read_json(artifact_path)
    age_minutes = _artifact_age_minutes(previous, now)
    if (
        not force
        and age_minutes is not None
        and age_minutes <= float(definition["refresh_after_minutes"])
    ):
        return {"collector_id": collector_id, "cached": True, "payload": previous}
    try:
        if collector_id == "cross_asset_breadth_context":
            payload = build_cross_asset_breadth_context(decisions, now=now)
        elif collector_id == "tape_liquidity_context":
            payload = build_tape_liquidity_context(decisions, now=now)
        elif collector_id == "options_greeks_surface_context":
            payload = build_options_greeks_surface_context(decisions, now=now)
        elif collector_id == "futures_curve_context":
            payload = build_futures_curve_context(decisions, now=now)
        elif collector_id == "earnings_event_context":
            payload = build_earnings_event_context(
                decisions,
                now=now,
                sec_payload=_read_json(
                    project_root
                    / "exports"
                    / "external_context"
                    / "sec_edgar_latest.json"
                ),
                analyst_payload=_read_json(
                    project_root
                    / "exports"
                    / "external_context"
                    / "analyst_consensus_latest.json"
                ),
            )
        elif collector_id == "portfolio_factor_risk_context":
            payload = build_portfolio_factor_risk_context(
                decisions,
                now=now,
                account_payload=_read_json(
                    project_root
                    / "governance"
                    / "health"
                    / "account_position_study_latest.json"
                ),
            )
        elif collector_id == "fixed_income_trace_context":
            if offline:
                raise RuntimeError("offline_mode_external_fetch_skipped")
            payload = build_fixed_income_trace_context(
                now=now, timeout_seconds=timeout_seconds
            )
        elif collector_id == "bis_global_liquidity_context":
            if offline:
                raise RuntimeError("offline_mode_external_fetch_skipped")
            payload = build_bis_global_liquidity_context(
                now=now, timeout_seconds=max(timeout_seconds, 20.0)
            )
        else:  # pragma: no cover - argparse and the catalog constrain this branch.
            raise ValueError(f"unknown_collector:{collector_id}")
        _write_success(project_root, collector_id, payload)
        return {"collector_id": collector_id, "cached": False, "payload": payload}
    except Exception as exc:
        payload = _write_failure_health(
            project_root, collector_id, now=now, previous=previous, error=exc
        )
        return {
            "collector_id": collector_id,
            "cached": False,
            "error": f"{type(exc).__name__}:{exc}",
            "payload": payload,
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refresh eight bounded, context-only research collectors."
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--collector", choices=COLLECTOR_IDS)
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--lookback-hours", type=float, default=18.0)
    parser.add_argument("--timeout-seconds", type=float, default=8.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Refresh local materializers and serve bounded last-good external artifacts without network calls.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    now = _now_utc()
    selected = list(COLLECTOR_IDS) if args.all else [str(args.collector)]
    local_selected = [
        collector_id
        for collector_id in selected
        if collector_id
        not in {"fixed_income_trace_context", "bis_global_liquidity_context"}
    ]
    decisions, decision_contract = (
        load_recent_decisions(
            project_root,
            now=now,
            lookback_hours=float(args.lookback_hours),
        )
        if local_selected
        else ([], {"row_count": 0, "file_count": 0, "files": [], "bounded": True})
    )
    results = [
        collect_one(
            collector_id,
            project_root=project_root,
            now=now,
            decisions=decisions,
            force=bool(args.force),
            offline=bool(args.offline),
            timeout_seconds=float(args.timeout_seconds),
        )
        for collector_id in selected
    ]
    rows = [
        {
            "collector_id": result["collector_id"],
            "cached": bool(result.get("cached", False)),
            "ok": bool((result.get("payload") or {}).get("ok", False)),
            "overall_status": str(
                (result.get("payload") or {}).get("overall_status") or "missing"
            ),
            "usable_capability_count": int(
                (result.get("payload") or {}).get("usable_capability_count", 0) or 0
            ),
            "capability_count": int(
                (result.get("payload") or {}).get("capability_count", 0) or 0
            ),
            "error": result.get("error"),
            "artifact_path": str(_artifact_path(project_root, result["collector_id"])),
            "health_path": str(_health_path(project_root, result["collector_id"])),
        }
        for result in results
    ]
    aggregate = {
        "timestamp_utc": now.isoformat(),
        "schema_version": 1,
        "ok": True,
        "overall_status": (
            "ready"
            if all(row["ok"] for row in rows)
            else "ready_with_optional_evidence_debt"
        ),
        "collector_count": len(rows),
        "ready_collector_count": sum(1 for row in rows if row["ok"]),
        "optional_failure_count": sum(1 for row in rows if not row["ok"]),
        "collector_error_count": sum(1 for row in rows if row.get("error")),
        "cached_collector_count": sum(1 for row in rows if row["cached"]),
        "selected_collectors": selected,
        "decision_tail_contract": decision_contract,
        "collectors": rows,
        "authority_contract": {
            "observation_only": True,
            "paper_execution_authority": False,
            "live_execution_authority": False,
            "automatic_promotion_authority": False,
        },
        "resource_contract": {
            "single_bounded_process": True,
            "per_bot_network_fanout": False,
            "decision_tail_max_rows": 6000,
            "decision_tail_max_files": 64,
            "external_collectors_cache_between_runs": True,
        },
    }
    aggregate["snapshot_receipt_sha256"] = _canonical_hash(aggregate)
    _atomic_write_json(
        project_root / AGGREGATE_PATH.relative_to(PROJECT_ROOT), aggregate
    )
    if args.json:
        print(json.dumps(aggregate, ensure_ascii=True))
    else:
        print(
            f"research_context_expansion status={aggregate['overall_status']} "
            f"ready={aggregate['ready_collector_count']}/{aggregate['collector_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
