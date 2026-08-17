#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.collector_transport import fetch_json, fetch_text  # noqa: E402
from core.market_context_features import load_latest_external_context  # noqa: E402
from core.fx_twelve_data_guard import (  # noqa: E402
    classify_twelve_data_failure,
    mark_twelve_data_cooldown,
    twelve_data_cooldown_status,
)
from scripts import ops_data_plane  # noqa: E402


ECB_FX_HIST_90D_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist-90d.xml"
FED_H10_CURRENT_URL = "https://www.federalreserve.gov/releases/h10/current/default.htm"
FRANKFURTER_LATEST_URL = "https://api.frankfurter.app/latest?from=EUR&to=USD,JPY,GBP,CHF,CAD,AUD"
ALPHA_VANTAGE_FX_INTRADAY_URL = "https://www.alphavantage.co/query"
TWELVE_DATA_TIME_SERIES_URL = "https://api.twelvedata.com/time_series"
USER_AGENT = "schwab-trading-bot/1.0"
SOURCE_CONTRACTS = {
    "ecb": {"source_confidence_norm": 0.99, "schema_confidence_norm": 0.95},
    "fed_h10": {"source_confidence_norm": 0.98, "schema_confidence_norm": 0.9},
    "frankfurter": {"source_confidence_norm": 0.9, "schema_confidence_norm": 0.9},
    "alpha_vantage": {"source_confidence_norm": 0.87, "schema_confidence_norm": 0.86},
    "twelve_data": {"source_confidence_norm": 0.89, "schema_confidence_norm": 0.88},
}

FEATURE_KEYS = [
    "fx_official_data_available",
    "fx_eurusd_level_norm",
    "fx_eurusd_momentum_norm",
    "fx_usdjpy_level_norm",
    "fx_usdjpy_momentum_norm",
    "fx_gbpusd_level_norm",
    "fx_gbpusd_momentum_norm",
    "fx_usd_strength_norm",
    "fx_usd_broad_index_norm",
    "fx_proxy_agreement_norm",
    "fx_risk_on_alignment_norm",
    "fx_crypto_alignment_norm",
    "fx_macro_dispersion_norm",
    "fx_corr_confidence_norm",
    "fx_session_asia_norm",
    "fx_session_london_norm",
    "fx_session_ny_norm",
    "fx_rollover_risk_norm",
    "fx_dxy_yield_confirmation_norm",
    "fx_carry_proxy_norm",
]

PAIR_SYMBOLS = ("EURUSD", "USDJPY", "GBPUSD", "USDCHF", "USDCAD", "AUDUSD")
PROXY_SYMBOLS = ("UUP", "FXE", "FXY", "FXB", "FXC", "FXA", "CYB", "SPY", "QQQ", "TLT", "GLD", "BTC-USD", "ETH-USD", "SOL-USD")
PAIR_TWELVE_DATA_SYMBOLS = {
    "EURUSD": "EUR/USD",
    "USDJPY": "USD/JPY",
    "GBPUSD": "GBP/USD",
    "USDCHF": "USD/CHF",
    "USDCAD": "USD/CAD",
    "AUDUSD": "AUD/USD",
}
FED_H10_PAIR_MARKERS = {
    "EURUSD": "*EMU MEMBERS",
    "USDJPY": "JAPAN",
    "GBPUSD": "*UNITED KINGDOM",
    "USDCHF": "SWITZERLAND",
    "USDCAD": "CANADA",
    "AUDUSD": "*AUSTRALIA",
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(payload), ensure_ascii=True, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _safe_load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_int_env(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, str(default)) or str(default)))
    except Exception:
        return int(default)


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _file_age_seconds(path: Path, now: datetime) -> float:
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except Exception:
        return float("inf")
    return max(0.0, (now - mtime).total_seconds())


def _fx_payload_reusable(path: Path) -> bool:
    payload = _safe_load_json(path)
    derived = payload.get("derived")
    if not str(payload.get("timestamp_utc") or "").strip() or not isinstance(derived, Mapping):
        return False
    pair_values = derived.get("pair_values")
    source_contracts = (payload.get("collection_contract") or {}).get("source_contracts")
    return bool(isinstance(pair_values, Mapping) and pair_values and isinstance(source_contracts, Mapping) and source_contracts)


def _pressure_min_interval_active(external_path: Path, *, min_interval: int, now: datetime) -> bool:
    return bool(
        min_interval > 0
        and _fx_payload_reusable(external_path)
        and _file_age_seconds(external_path, now) < min_interval
    )


def _collector_pressure_contract(now: datetime) -> dict[str, Any]:
    runtime = _safe_load_json(PROJECT_ROOT / "governance" / "health" / "runtime_throttle_control_latest.json")
    mac = runtime.get("mac_fluidity_contract") if isinstance(runtime.get("mac_fluidity_contract"), dict) else {}
    score = _to_float(runtime.get("host_saturation_score"), 0.0)
    compute = str(runtime.get("compute_pressure_level") or "").strip().lower()
    memory = str(runtime.get("memory_pressure_level") or "").strip().lower()
    profile = str(os.getenv("BOT_RUNTIME_RESOURCE_GUARD_PROFILE") or runtime.get("throttle_profile") or "").strip().lower()
    mac_status = str(mac.get("overall_status") or "").strip().lower()
    mac_band = str(mac.get("fluidity_band") or "").strip().lower()
    env_mode = str(os.getenv("FX_MARKET_CONTEXT_PRESSURE_MODE") or "").strip().lower()
    if env_mode:
        mode = env_mode
    elif memory == "high" or compute == "high" or profile == "protect_live" or score >= 75.0 or mac_band == "protect":
        mode = "protect"
    elif compute == "elevated" or profile == "sustain" or score >= 60.0 or mac_status == "needs_work" or mac_band == "strained":
        mode = "guarded"
    elif profile == "soft_cap" or score >= 45.0 or mac_band == "guarded_smooth":
        mode = "calm"
    else:
        mode = "off"
    active = bool(mode in {"protect", "guarded", "calm"} or _bool_env("CONTEXT_COLLECTOR_PRESSURE_GOVERNOR_ENABLED", False))
    defaults = {
        "protect": {"min_interval": 900, "pairs_per_run": 1, "outputsize": 12, "timeout": 8.0},
        "guarded": {"min_interval": 600, "pairs_per_run": 1, "outputsize": 24, "timeout": 10.0},
        "calm": {"min_interval": 300, "pairs_per_run": 2, "outputsize": 36, "timeout": 12.0},
        "off": {"min_interval": 0, "pairs_per_run": 0, "outputsize": 72, "timeout": 20.0},
    }
    selected = defaults.get(mode, defaults["off"])
    min_interval = _safe_int_env("FX_MARKET_CONTEXT_MIN_INTERVAL_SECONDS", selected["min_interval"])
    configured_pairs_per_run = _safe_int_env("FX_TWELVE_DATA_MAX_PAIRS_PER_RUN", selected["pairs_per_run"])
    configured_outputsize = _safe_int_env("FX_TWELVE_DATA_OUTPUTSIZE", selected["outputsize"])
    configured_timeout = _to_float(os.getenv("FX_MARKET_CONTEXT_TIMEOUT_CAP_SECONDS"), float(selected["timeout"]))
    pairs_per_run = max(0, min(configured_pairs_per_run, int(selected["pairs_per_run"]))) if active else configured_pairs_per_run
    outputsize_cap = max(4, min(configured_outputsize, int(selected["outputsize"]))) if active else configured_outputsize
    timeout_cap = min(configured_timeout, float(selected["timeout"])) if active else configured_timeout
    return {
        "active": active,
        "mode": mode,
        "timestamp_utc": now.isoformat(),
        "host_saturation_score": round(score, 3),
        "compute_pressure_level": compute or "unknown",
        "memory_pressure_level": memory or "unknown",
        "runtime_profile": profile or "unknown",
        "mac_fluidity_status": mac_status or "unknown",
        "mac_fluidity_band": mac_band or "unknown",
        "min_interval_seconds": max(int(min_interval), 0) if active else 0,
        "max_pairs_per_run": int(pairs_per_run),
        "outputsize_cap": int(outputsize_cap),
        "timeout_cap_seconds": round(float(timeout_cap), 3),
        "lock_path": str(PROJECT_ROOT / "governance" / "locks" / "fx_market_context.lock"),
        "policy": "single_flight_and_min_interval_under_runtime_pressure",
    }


def _apply_collector_pressure_env(contract: Mapping[str, Any]) -> None:
    if not bool(contract.get("active", False)):
        return

    def cap_int_env(name: str, cap: int) -> None:
        current = _safe_int_env(name, cap)
        os.environ[name] = str(max(0, min(current, cap)))

    cap_int_env("FX_TWELVE_DATA_MAX_PAIRS_PER_RUN", int(contract.get("max_pairs_per_run") or 1))
    cap_int_env("FX_TWELVE_DATA_OUTPUTSIZE", int(contract.get("outputsize_cap") or 24))
    os.environ.setdefault("FX_MARKET_CONTEXT_ALPHA_VANTAGE_ENABLED", "0")
    os.environ["FX_MARKET_CONTEXT_PRESSURE_MODE"] = str(contract.get("mode") or "guarded")


def _pressure_skip_health(
    *,
    reason: str,
    contract: Mapping[str, Any],
    external_path: Path,
    health_path: Path,
    now: datetime,
) -> dict[str, Any]:
    previous_health = _safe_load_json(health_path)
    previous_payload = _safe_load_json(external_path)
    previous_derived = (
        previous_payload.get("derived")
        if isinstance(previous_payload.get("derived"), Mapping)
        else {}
    )
    previous_global_features = (
        previous_derived.get("global_features")
        if isinstance(previous_derived.get("global_features"), Mapping)
        else {}
    )
    previous_sources = (
        previous_health.get("sources")
        if isinstance(previous_health.get("sources"), Mapping)
        else previous_payload.get("sources")
        if isinstance(previous_payload.get("sources"), Mapping)
        else {}
    )
    previous_pair_values = (
        previous_derived.get("pair_values")
        if isinstance(previous_derived.get("pair_values"), Mapping)
        else {}
    )
    previous_market = (
        previous_derived.get("latest_market")
        if isinstance(previous_derived.get("latest_market"), Mapping)
        else {}
    )
    previous_reconciliation = (
        previous_derived.get("canonical_reconciliation")
        if isinstance(previous_derived.get("canonical_reconciliation"), Mapping)
        else {}
    )
    previous_contract = (
        previous_payload.get("collection_contract")
        if isinstance(previous_payload.get("collection_contract"), Mapping)
        else {}
    )
    ok = _fx_payload_reusable(external_path)
    return {
        **previous_health,
        "timestamp_utc": now.isoformat(),
        "ok": ok,
        "skipped": True,
        "skip_reason": reason,
        "serving_last_good_snapshot": ok,
        "last_good_snapshot_timestamp_utc": str(previous_payload.get("timestamp_utc") or ""),
        "pressure_contract": dict(contract),
        "previous_health_age_seconds": round(_file_age_seconds(health_path, now), 3),
        "previous_payload_age_seconds": round(_file_age_seconds(external_path, now), 3),
        "source_count": int(previous_health.get("source_count", len(previous_sources)) or len(previous_sources)),
        "ok_source_count": int(
            previous_health.get(
                "ok_source_count",
                sum(1 for row in previous_sources.values() if isinstance(row, Mapping) and bool(row.get("ok", False))),
            )
            or 0
        ),
        "official_pairs": int(previous_health.get("official_pairs", len(previous_pair_values)) or len(previous_pair_values)),
        "proxy_symbols_observed": int(
            previous_health.get("proxy_symbols_observed", len(previous_market)) or len(previous_market)
        ),
        "proxy_agreement_norm": round(
            _to_float(
                previous_health.get(
                    "proxy_agreement_norm",
                    previous_global_features.get("fx_proxy_agreement_norm"),
                ),
                0.0,
            ),
            6,
        ),
        "direct_forex_execution_supported": bool(
            previous_health.get("direct_forex_execution_supported", False)
        ),
        "direct_forex_execution_reason": str(
            previous_health.get("direct_forex_execution_reason")
            or "schwab_official_api_forex_unverified"
        ),
        "sources": dict(previous_sources),
        "canonical_pairs": int(
            previous_health.get("canonical_pairs", len(previous_reconciliation))
            or len(previous_reconciliation)
        ),
        "source_contracts": dict(
            previous_health.get("source_contracts")
            if isinstance(previous_health.get("source_contracts"), Mapping)
            else previous_contract.get("source_contracts")
            if isinstance(previous_contract.get("source_contracts"), Mapping)
            else {}
        ),
        "policy": "serve_a_complete_valid_last_good_fx_health_contract_when_pressure_blocks_redundant_collection",
    }


def _split_pair_symbols(raw: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for token in str(raw or "").split(","):
        pair = _normalize_pair_symbol(token)
        if pair and pair not in seen:
            seen.add(pair)
            out.append(pair)
    return out


def _configured_twelve_data_pairs() -> tuple[list[str], dict[str, Any]]:
    realtime_pairs = _split_pair_symbols(os.getenv("FX_REALTIME_SYMBOLS", ",".join(PAIR_SYMBOLS))) or list(PAIR_SYMBOLS)
    context_pairs = _split_pair_symbols(os.getenv("FX_REALTIME_CONTEXT_SYMBOLS", "EURUSD,USDJPY"))
    requested_pairs = context_pairs or realtime_pairs
    max_credits_per_minute = max(_safe_int_env("FX_TWELVE_DATA_MAX_CREDITS_PER_MINUTE", 8), 0)
    credit_reserve = max(_safe_int_env("FX_TWELVE_DATA_CREDIT_RESERVE", 3), 0)
    usable_budget = max(max_credits_per_minute - credit_reserve, 0)
    default_pairs_per_run = min(max(usable_budget, 1), len(requested_pairs)) if max_credits_per_minute > 0 else len(requested_pairs)
    pairs_per_run = max(_safe_int_env("FX_TWELVE_DATA_MAX_PAIRS_PER_RUN", default_pairs_per_run), 0)
    credit_budget_per_run = min(pairs_per_run, usable_budget) if max_credits_per_minute > 0 else pairs_per_run
    selected_pairs = requested_pairs[: max(credit_budget_per_run, 0)]
    deferred_pairs = requested_pairs[len(selected_pairs) :]
    return selected_pairs, {
        "requested_pairs": requested_pairs,
        "selected_pairs": selected_pairs,
        "deferred_pairs": deferred_pairs,
        "max_credits_per_minute": max_credits_per_minute,
        "credit_reserve": credit_reserve,
        "credit_budget_per_run": credit_budget_per_run,
        "pairs_per_run": pairs_per_run,
    }


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _signed_norm(value: float, scale: float) -> float:
    return _clamp01(0.5 + (float(value) / max(float(scale), 1e-8)))


def _pct_change(current: float | None, previous: float | None) -> float:
    if current is None or previous is None:
        return 0.0
    if abs(previous) <= 1e-12:
        return 0.0
    return (float(current) - float(previous)) / abs(float(previous))


def _source_contract_name(url: str) -> str:
    text = str(url or "")
    if "ecb.europa.eu" in text:
        return "ecb"
    if "federalreserve.gov" in text:
        return "fed_h10"
    if "frankfurter.app" in text:
        return "frankfurter"
    if "alphavantage.co" in text:
        return "alpha_vantage"
    if "twelvedata.com" in text:
        return "twelve_data"
    return "fed_h10"


def _source_contract(source_name: str) -> dict[str, float]:
    row = SOURCE_CONTRACTS.get(str(source_name or ""), {})
    return {
        "source_confidence_norm": float(row.get("source_confidence_norm", 0.9) or 0.9),
        "schema_confidence_norm": float(row.get("schema_confidence_norm", 0.9) or 0.9),
    }


def _fetch_text_result(url: str, *, timeout: float = 20.0) -> dict[str, Any]:
    source_name = _source_contract_name(url)
    contract = _source_contract(source_name)
    return fetch_text(
        url=url,
        user_agent=USER_AGENT,
        timeout=timeout,
        collector_key="fx_market_context",
        source_name=source_name,
        entity_key=url,
        project_root=PROJECT_ROOT,
        source_confidence_norm=contract["source_confidence_norm"],
        schema_confidence_norm=contract["schema_confidence_norm"],
    )


def _http_text(url: str, *, timeout: float = 20.0) -> str:
    result = _fetch_text_result(url, timeout=timeout)
    if not bool(result.get("ok", False)):
        raise RuntimeError(str(result.get("error") or "http_text_failed"))
    return str(result.get("text") or "")


def _http_json(url: str, *, timeout: float = 20.0) -> Any:
    source_name = _source_contract_name(url)
    contract = _source_contract(source_name)
    result = fetch_json(
        url=url,
        user_agent=USER_AGENT,
        timeout=timeout,
        collector_key="fx_market_context",
        source_name=source_name,
        entity_key=url,
        project_root=PROJECT_ROOT,
        source_confidence_norm=contract["source_confidence_norm"],
        schema_confidence_norm=contract["schema_confidence_norm"],
    )
    if not bool(result.get("ok", False)):
        raise RuntimeError(str(result.get("error") or "http_json_failed"))
    return result.get("json")


def _canonical_pair_reconciliation(
    *,
    ecb_pairs: Mapping[str, Any],
    fed_pairs: Mapping[str, Any],
    twelve_data_intraday: Mapping[str, Any],
    alpha_vantage_intraday: Mapping[str, Any],
    frankfurter_pairs: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    preferred_order = {"twelve_data": 0, "alpha_vantage": 1, "fed_h10": 2, "frankfurter": 3, "ecb": 4}
    provider_floor_recency = {"twelve_data": 0.75, "alpha_vantage": 0.7, "fed_h10": 0.45, "frankfurter": 0.35, "ecb": 0.2}

    def _provider_recency_score(raw_ts: Any) -> float:
        text = str(raw_ts or "").strip()
        if not text:
            return 0.15
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            age_seconds = max(datetime.now(timezone.utc).timestamp() - parsed.astimezone(timezone.utc).timestamp(), 0.0)
        except Exception:
            return 0.15
        if age_seconds <= 30.0 * 60.0:
            return 1.0
        if age_seconds <= 2.0 * 3600.0:
            return 0.9
        if age_seconds <= 12.0 * 3600.0:
            return 0.7
        if age_seconds <= 48.0 * 3600.0:
            return 0.45
        return 0.2

    out: dict[str, dict[str, Any]] = {}
    for pair in PAIR_SYMBOLS:
        provider_rows: dict[str, dict[str, float | str]] = {}
        ecb_value = _to_float(ecb_pairs.get(pair), 0.0)
        if ecb_value > 0.0:
            provider_rows["ecb"] = {
                "value": ecb_value,
                "latest_ts": "",
                "recency_score": 0.2,
            }
        fed_value = _to_float(fed_pairs.get(pair), 0.0)
        if fed_value > 0.0:
            provider_rows["fed_h10"] = {
                "value": fed_value,
                "latest_ts": "",
                "recency_score": 0.45,
            }
        frankfurter_value = _to_float((frankfurter_pairs or {}).get(pair), 0.0)
        if frankfurter_value > 0.0:
            provider_rows["frankfurter"] = {
                "value": frankfurter_value,
                "latest_ts": "",
                "recency_score": 0.35,
            }
        intraday_row = twelve_data_intraday.get(pair)
        if isinstance(intraday_row, Mapping) and intraday_row.get("ok"):
            td_value = _to_float(intraday_row.get("latest_close"), 0.0)
            if td_value > 0.0:
                provider_rows["twelve_data"] = {
                    "value": td_value,
                    "latest_ts": str(intraday_row.get("latest_ts") or ""),
                    "recency_score": max(
                        _provider_recency_score(intraday_row.get("latest_ts")),
                        float(provider_floor_recency["twelve_data"]),
                    ),
                }
        if pair == "EURUSD" and alpha_vantage_intraday.get("ok"):
            av_value = _to_float(alpha_vantage_intraday.get("latest_close"), 0.0)
            if av_value > 0.0:
                provider_rows["alpha_vantage"] = {
                    "value": av_value,
                    "latest_ts": str(alpha_vantage_intraday.get("latest_ts") or ""),
                    "recency_score": max(
                        _provider_recency_score(alpha_vantage_intraday.get("latest_ts")),
                        float(provider_floor_recency["alpha_vantage"]),
                    ),
                }
        if not provider_rows:
            continue
        sorted_values = sorted(float(row["value"]) for row in provider_rows.values())
        mid = len(sorted_values) // 2
        median_value = (
            sorted_values[mid]
            if len(sorted_values) % 2 == 1
            else (sorted_values[mid - 1] + sorted_values[mid]) / 2.0
        )
        candidate_rows: list[tuple[str, float, float]] = []
        for source, row in provider_rows.items():
            value = float(row["value"])
            delta_ratio = abs(value - float(median_value)) / max(float(median_value), 1e-9)
            if delta_ratio <= 0.0025:
                candidate_rows.append((source, value, delta_ratio))
        if not candidate_rows:
            candidate_rows = [
                (
                    source,
                    float(row["value"]),
                    abs(float(row["value"]) - float(median_value)) / max(float(median_value), 1e-9),
                )
                for source, row in provider_rows.items()
            ]
        canonical_source, canonical_value, _ = max(
            candidate_rows,
            key=lambda item: (
                float(provider_rows[item[0]].get("recency_score", 0.0) or 0.0),
                -float(item[2]),
                -float(preferred_order.get(str(item[0]), 99)),
            ),
        )
        max_value = max(float(row["value"]) for row in provider_rows.values())
        min_value = min(float(row["value"]) for row in provider_rows.values())
        divergence_ratio = 0.0 if median_value <= 0.0 else abs(max_value - min_value) / max(median_value, 1e-9)
        high_recency_values = [
            float(row["value"])
            for row in provider_rows.values()
            if float(row.get("recency_score", 0.0) or 0.0) >= 0.7
        ]
        high_recency_divergence_ratio = 0.0
        if len(high_recency_values) >= 2:
            high_recency_mid = sum(high_recency_values) / max(len(high_recency_values), 1)
            high_recency_divergence_ratio = (
                abs(max(high_recency_values) - min(high_recency_values))
                / max(float(high_recency_mid), 1e-9)
            )
        provider_count = len(provider_rows)
        confidence = _clamp01((min(provider_count / 4.0, 1.0) * 0.45) + max(1.0 - (divergence_ratio / 0.02), 0.0) * 0.55)
        provider_votes = {
            source: {
                "value": round(float(row["value"]), 6),
                "latest_ts": str(row.get("latest_ts") or ""),
                "recency_score": round(float(row.get("recency_score", 0.0) or 0.0), 6),
                "delta_to_canonical_bps": round(((float(row["value"]) - float(canonical_value)) / max(float(canonical_value), 1e-9)) * 10000.0, 6),
            }
            for source, row in provider_rows.items()
        }
        divergence_severity = "none"
        divergence_reason = ""
        if divergence_ratio >= 0.01:
            if len(high_recency_values) >= 2 and high_recency_divergence_ratio >= 0.01:
                divergence_severity = "warning"
                divergence_reason = "high_recency_provider_divergence"
            elif not high_recency_values:
                divergence_severity = "warning"
                divergence_reason = "stale_provider_divergence"
            else:
                divergence_severity = "basis_watch"
                divergence_reason = "intraday_official_basis_difference"
        out[pair] = {
            "canonical_source": str(canonical_source),
            "canonical_value": round(float(canonical_value), 6),
            "median_value": round(float(median_value), 6),
            "provider_count": provider_count,
            "divergence_ratio": round(float(divergence_ratio), 6),
            "high_recency_divergence_ratio": round(float(high_recency_divergence_ratio), 6),
            "divergence_severity": divergence_severity,
            "divergence_reason": divergence_reason,
            "confidence_norm": round(float(confidence), 6),
            "provider_votes": provider_votes,
        }
    return out


def _normalize_pair_symbol(raw: Any) -> str:
    token = str(raw or "").strip().upper().replace("/", "").replace("-", "")
    return token if token in PAIR_SYMBOLS else ""


def _twelve_data_time_series(
    *,
    api_key: str,
    pair_symbol: str,
    interval: str,
    outputsize: int,
    timeout: float,
) -> dict[str, Any]:
    normalized = _normalize_pair_symbol(pair_symbol)
    feed_symbol = PAIR_TWELVE_DATA_SYMBOLS.get(normalized, "")
    if not normalized or not feed_symbol:
        return {"ok": False, "error": "unsupported_pair", "pair_symbol": normalized}
    cooldown = twelve_data_cooldown_status(PROJECT_ROOT)
    if bool(cooldown.get("active")):
        return {
            "ok": False,
            "error": (
                "provider_cooldown_active:"
                f"{cooldown.get('kind', 'rate_limit')}:{int(float(cooldown.get('remaining_seconds', 0.0) or 0.0))}"
            ),
            "pair_symbol": normalized,
            "cooldown": cooldown,
        }
    query = (
        f"{TWELVE_DATA_TIME_SERIES_URL}"
        f"?symbol={feed_symbol}&interval={interval}&outputsize={max(int(outputsize), 2)}"
        f"&apikey={api_key}&format=JSON"
    )
    try:
        raw = _http_text(query, timeout=timeout)
    except (HTTPError, URLError, RuntimeError, TimeoutError, OSError) as exc:
        code = str(getattr(exc, "code", "") or "").strip()
        message = str(exc)
        failure_kind = classify_twelve_data_failure(code=code, message=message)
        marked = {}
        if failure_kind:
            marked = mark_twelve_data_cooldown(
                project_root=PROJECT_ROOT,
                kind=failure_kind,
                code=code,
                message=message,
                symbol=normalized,
                source="collect_fx_market_context",
            )
        return {
            "ok": False,
            "error": f"http_error:{code}:{message}" if code else f"http_error:{message}",
            "pair_symbol": normalized,
            "cooldown": marked,
        }
    try:
        payload = json.loads(raw)
    except Exception:
        return {"ok": False, "error": "invalid_json", "pair_symbol": normalized, "payload": {}}
    if not isinstance(payload, Mapping):
        return {"ok": False, "error": "invalid_payload", "pair_symbol": normalized, "payload": {}}
    if str(payload.get("status", "")).strip().lower() == "error":
        code = str(payload.get("code", "")).strip()
        message = str(payload.get("message", "")).strip() or "twelve_data_error"
        failure_kind = classify_twelve_data_failure(code=code, message=message)
        marked = {}
        if failure_kind:
            marked = mark_twelve_data_cooldown(
                project_root=PROJECT_ROOT,
                kind=failure_kind,
                code=code,
                message=message,
                symbol=normalized,
                source="collect_fx_market_context",
            )
        return {
            "ok": False,
            "error": f"{code}:{message}" if code else message,
            "pair_symbol": normalized,
            "payload": dict(payload),
            "cooldown": marked,
        }
    values = payload.get("values")
    if not isinstance(values, list):
        return {"ok": False, "error": "missing_values", "pair_symbol": normalized, "payload": dict(payload)}
    rows: list[tuple[str, float]] = []
    for row in values:
        if not isinstance(row, Mapping):
            continue
        ts = str(row.get("datetime") or "").strip()
        close_value = _to_float(row.get("close"), math.nan)
        if ts and math.isfinite(close_value) and close_value > 0.0:
            rows.append((ts, close_value))
    rows.sort(key=lambda item: item[0])
    if len(rows) < 2:
        return {"ok": False, "error": "insufficient_intraday_rows", "pair_symbol": normalized, "payload": dict(payload)}
    latest_ts, latest_close = rows[-1]
    previous_ts, previous_close = rows[-2]
    session_ts, session_close = rows[0]
    return {
        "ok": True,
        "error": None,
        "pair_symbol": normalized,
        "feed_symbol": feed_symbol,
        "rows": len(rows),
        "latest_ts": latest_ts,
        "previous_ts": previous_ts,
        "session_ts": session_ts,
        "latest_close": latest_close,
        "previous_close": previous_close,
        "session_close": session_close,
    }


def _alpha_vantage_intraday(
    *,
    api_key: str,
    from_symbol: str,
    to_symbol: str,
    interval: str,
    timeout: float,
) -> dict[str, Any]:
    query = (
        f"{ALPHA_VANTAGE_FX_INTRADAY_URL}"
        f"?function=FX_INTRADAY&from_symbol={from_symbol}&to_symbol={to_symbol}"
        f"&interval={interval}&outputsize=compact&apikey={api_key}"
    )
    raw = _http_text(query, timeout=timeout)
    try:
        payload = json.loads(raw)
    except Exception:
        return {"ok": False, "error": "invalid_json", "payload": {}}
    if not isinstance(payload, dict):
        return {"ok": False, "error": "invalid_payload", "payload": {}}
    if payload.get("Note"):
        return {"ok": False, "error": f"rate_limited:{payload.get('Note')}", "payload": payload}
    if payload.get("Information"):
        return {"ok": False, "error": str(payload.get("Information")), "payload": payload}
    if payload.get("Error Message"):
        return {"ok": False, "error": str(payload.get("Error Message")), "payload": payload}
    series_key = next((key for key in payload.keys() if str(key).startswith("Time Series FX")), "")
    series = payload.get(series_key) if isinstance(payload.get(series_key), Mapping) else {}
    rows: list[tuple[str, float]] = []
    for ts, row in series.items():
        if not isinstance(row, Mapping):
            continue
        close_value = _to_float(row.get("4. close"), math.nan)
        if math.isfinite(close_value) and close_value > 0.0:
            rows.append((str(ts), close_value))
    rows.sort(key=lambda item: item[0])
    if len(rows) < 2:
        return {"ok": False, "error": "insufficient_intraday_rows", "payload": payload}
    latest_ts, latest_close = rows[-1]
    previous_ts, previous_close = rows[-2]
    return {
        "ok": True,
        "error": None,
        "rows": len(rows),
        "latest_ts": latest_ts,
        "previous_ts": previous_ts,
        "latest_close": latest_close,
        "previous_close": previous_close,
    }


def _parse_fed_h10_current(html_text: str) -> dict[str, Any]:
    lines = []
    for raw in str(html_text or "").splitlines():
        text = re.sub(r"<[^>]+>", " ", raw)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            lines.append(text)
    pair_values: dict[str, float] = {}
    previous_values: dict[str, float] = {}
    broad_index = 0.0

    def _extract_tail_floats(text: str) -> list[float]:
        matches = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
        values = []
        for token in matches[-5:]:
            value = _to_float(token, math.nan)
            if math.isfinite(value):
                values.append(value)
        return values

    marker_starts = set(FED_H10_PAIR_MARKERS.values()) | {"1) BROAD"}

    def _extract_marker_values(start_idx: int) -> list[float]:
        # The live Fed H10 page often splits a logical row across multiple lines:
        # marker, currency/unit, then one numeric cell per line. Collect just that row.
        values: list[float] = []
        started_numeric = False
        for idx in range(start_idx, min(len(lines), start_idx + 10)):
            line = lines[idx]
            if idx > start_idx and line in marker_starts:
                break
            numeric_tokens = re.findall(r"[-+]?\d+\.\d+", line)
            if not numeric_tokens:
                if started_numeric:
                    break
                continue
            for token in numeric_tokens:
                value = _to_float(token, math.nan)
                if math.isfinite(value):
                    values.append(value)
            started_numeric = True
            if len(values) >= 5:
                break
        return values[-5:]

    for pair, marker in FED_H10_PAIR_MARKERS.items():
        for idx, line in enumerate(lines):
            if line.startswith(marker):
                values = _extract_marker_values(idx)
                if len(values) >= 2:
                    previous_values[pair] = float(values[-2])
                    pair_values[pair] = float(values[-1])
                break

    for idx, line in enumerate(lines):
        if line.startswith("1) BROAD"):
            values = _extract_marker_values(idx)
            if values:
                broad_index = float(values[-1])
            break

    ok = len(pair_values) >= 3
    return {
        "ok": ok,
        "pair_values": pair_values,
        "previous_pair_values": previous_values,
        "broad_index": broad_index,
        "pair_count": len(pair_values),
        "error": None if ok else "insufficient_h10_pairs",
    }


def _parse_ecb_hist_90d(xml_text: str) -> list[dict[str, Any]]:
    root = ET.fromstring(xml_text)
    rows: list[dict[str, Any]] = []
    for cube_time in root.findall(".//{*}Cube[@time]"):
        day = str(cube_time.attrib.get("time") or "").strip()
        if not day:
            continue
        rates: dict[str, float] = {}
        for cube in cube_time.findall("{*}Cube[@currency][@rate]"):
            currency = str(cube.attrib.get("currency") or "").strip().upper()
            rate = _to_float(cube.attrib.get("rate"), math.nan)
            if currency and math.isfinite(rate) and rate > 0.0:
                rates[currency] = rate
        if rates:
            rows.append({"date": day, "rates": rates})
    rows.sort(key=lambda row: str(row.get("date") or ""))
    return rows


def _parse_frankfurter_latest(payload: Mapping[str, Any]) -> dict[str, float]:
    rates = payload.get("rates") if isinstance(payload.get("rates"), Mapping) else {}
    return _pair_levels({key: _to_float(value, math.nan) for key, value in rates.items()})


def _pair_levels(rates: Mapping[str, float]) -> dict[str, float]:
    usd = _to_float(rates.get("USD"), math.nan)
    jpy = _to_float(rates.get("JPY"), math.nan)
    gbp = _to_float(rates.get("GBP"), math.nan)
    chf = _to_float(rates.get("CHF"), math.nan)
    cad = _to_float(rates.get("CAD"), math.nan)
    aud = _to_float(rates.get("AUD"), math.nan)
    out: dict[str, float] = {}
    if math.isfinite(usd) and usd > 0.0:
        out["EURUSD"] = usd
        if math.isfinite(jpy) and jpy > 0.0:
            out["USDJPY"] = jpy / usd
        if math.isfinite(gbp) and gbp > 0.0:
            out["GBPUSD"] = usd / gbp
        if math.isfinite(chf) and chf > 0.0:
            out["USDCHF"] = chf / usd
        if math.isfinite(cad) and cad > 0.0:
            out["USDCAD"] = cad / usd
        if math.isfinite(aud) and aud > 0.0:
            out["AUDUSD"] = usd / aud
    return out


def _latest_pair_history(rows: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, float]]:
    if not rows:
        return {}, {}
    latest = _pair_levels(rows[-1].get("rates") if isinstance(rows[-1].get("rates"), Mapping) else {})
    previous = {}
    if len(rows) >= 2:
        previous = _pair_levels(rows[-2].get("rates") if isinstance(rows[-2].get("rates"), Mapping) else {})
    return latest, previous


def _latest_currency_reference_history(
    rows: list[dict[str, Any]],
    *,
    as_of_date: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, float], list[str]]:
    eligible = [row for row in rows if str(row.get("date") or "") <= str(as_of_date)]
    future_dates = sorted(
        str(row.get("date") or "")
        for row in rows
        if str(row.get("date") or "") > str(as_of_date)
    )
    if not eligible:
        return {}, {}, future_dates
    latest = eligible[-1]
    previous = eligible[-2] if len(eligible) >= 2 else {}
    latest_rates = latest.get("rates") if isinstance(latest.get("rates"), Mapping) else {}
    previous_rates = previous.get("rates") if isinstance(previous.get("rates"), Mapping) else {}
    reference_rates: dict[str, dict[str, Any]] = {
        "EUR": {
            "date": str(latest.get("date") or ""),
            "units_per_eur": 1.0,
            "source": "European Central Bank reference rates",
        }
    }
    changes: dict[str, float] = {"EUR": 0.0}
    for currency, raw_value in latest_rates.items():
        value = _to_float(raw_value, math.nan)
        if not math.isfinite(value) or value <= 0.0:
            continue
        token = str(currency or "").strip().upper()
        if not token:
            continue
        reference_rates[token] = {
            "date": str(latest.get("date") or ""),
            "units_per_eur": float(value),
            "source": "European Central Bank reference rates",
        }
        prior = _to_float(previous_rates.get(token), math.nan)
        changes[token] = _pct_change(value, prior) if math.isfinite(prior) and prior > 0.0 else 0.0
    return reference_rates, changes, future_dates


def _latest_market_snapshot() -> dict[str, dict[str, float]]:
    snapshot = load_latest_external_context(PROJECT_ROOT, "market_crypto_correlation")
    derived = snapshot.get("derived") if isinstance(snapshot.get("derived"), Mapping) else {}
    latest_market = derived.get("latest_market") if isinstance(derived.get("latest_market"), Mapping) else {}
    if not latest_market:
        latest_market = derived.get("latest_snapshots") if isinstance(derived.get("latest_snapshots"), Mapping) else {}
    out: dict[str, dict[str, float]] = {}
    for symbol, row in latest_market.items():
        if not isinstance(row, Mapping):
            continue
        token = str(symbol or "").strip().upper()
        if token not in PROXY_SYMBOLS:
            continue
        out[token] = {
            "pct_from_close": _to_float(row.get("pct_from_close"), 0.0),
            "mom_5m": _to_float(row.get("mom_5m"), 0.0),
            "last_price": _to_float(row.get("last_price"), 0.0),
            "ts": _to_float(row.get("ts"), 0.0),
        }
    return out


def _macro_cross_asset_context() -> dict[str, Any]:
    return _safe_load_json(PROJECT_ROOT / "exports" / "external_context" / "macro_cross_asset_latest.json")


def _proxy_agreement(
    pair_changes: Mapping[str, float],
    latest_market: Mapping[str, Mapping[str, float]],
    usd_strength_raw: float,
) -> tuple[float, dict[str, bool]]:
    checks: dict[str, bool] = {}

    def _direction(value: float, *, invert: bool = False) -> bool | None:
        if abs(value) <= 1e-6:
            return None
        return (value < 0.0) if invert else (value > 0.0)

    comparisons = [
        ("EURUSD", "FXE", False),
        ("USDJPY", "FXY", True),
        ("GBPUSD", "FXB", False),
        ("AUDUSD", "FXA", False),
        ("USDCAD", "FXC", True),
    ]
    matches = 0
    total = 0
    for pair, proxy, invert in comparisons:
        if proxy not in latest_market:
            continue
        pair_dir = _direction(_to_float(pair_changes.get(pair), 0.0))
        proxy_dir = _direction(_to_float((latest_market.get(proxy) or {}).get("pct_from_close"), 0.0), invert=invert)
        if pair_dir is None or proxy_dir is None:
            continue
        ok = bool(pair_dir == proxy_dir)
        checks[f"{pair}_{proxy}"] = ok
        total += 1
        matches += 1 if ok else 0

    if "UUP" in latest_market:
        usd_dir = _direction(usd_strength_raw)
        proxy_dir = _direction(_to_float((latest_market.get("UUP") or {}).get("pct_from_close"), 0.0))
        if usd_dir is not None and proxy_dir is not None:
            ok = bool(usd_dir == proxy_dir)
            checks["USD_UUP"] = ok
            total += 1
            matches += 1 if ok else 0

    if total <= 0:
        return 0.0, checks
    return matches / total, checks


def _risk_alignment(latest_market: Mapping[str, Mapping[str, float]], usd_strength_raw: float) -> tuple[float, float]:
    risk_symbols = ("SPY", "QQQ")
    crypto_symbols = ("BTC-USD", "ETH-USD", "SOL-USD")
    risk_values = [_to_float((latest_market.get(symbol) or {}).get("pct_from_close"), 0.0) for symbol in risk_symbols if symbol in latest_market]
    crypto_values = [_to_float((latest_market.get(symbol) or {}).get("pct_from_close"), 0.0) for symbol in crypto_symbols if symbol in latest_market]
    risk_avg = sum(risk_values) / len(risk_values) if risk_values else 0.0
    crypto_avg = sum(crypto_values) / len(crypto_values) if crypto_values else 0.0
    risk_align = _clamp01(0.5 + (-usd_strength_raw * risk_avg * 120.0))
    crypto_align = _clamp01(0.5 + (-usd_strength_raw * crypto_avg * 120.0))
    return risk_align, crypto_align


def _fx_session_state_norms(now_utc: datetime) -> dict[str, float]:
    hour = float(now_utc.hour) + (float(now_utc.minute) / 60.0)
    asia = 1.0 if (hour >= 21.0 or hour < 7.0) else 0.0
    london = 1.0 if 7.0 <= hour < 16.0 else 0.0
    ny = 1.0 if 13.0 <= hour < 22.0 else 0.0
    rollover = 1.0 if 20.0 <= hour < 22.0 else 0.0
    return {
        "fx_session_asia_norm": asia,
        "fx_session_london_norm": london,
        "fx_session_ny_norm": ny,
        "fx_rollover_risk_norm": rollover,
    }


def _fx_dxy_yield_confirmation(
    latest_market: Mapping[str, Mapping[str, float]],
    *,
    usd_strength_raw: float,
    proxy_agreement_raw: float,
) -> float:
    usd_dir = 0
    if usd_strength_raw > 1e-6:
        usd_dir = 1
    elif usd_strength_raw < -1e-6:
        usd_dir = -1
    if usd_dir == 0:
        return _clamp01(proxy_agreement_raw)

    votes: list[float] = []
    uup_move = _to_float((latest_market.get("UUP") or {}).get("pct_from_close"), 0.0)
    if abs(uup_move) > 1e-6:
        votes.append(1.0 if ((1 if uup_move > 0.0 else -1) == usd_dir) else 0.0)
    tlt_move = _to_float((latest_market.get("TLT") or {}).get("pct_from_close"), 0.0)
    if abs(tlt_move) > 1e-6:
        votes.append(1.0 if ((1 if (-tlt_move) > 0.0 else -1) == usd_dir) else 0.0)
    if not votes:
        return _clamp01(proxy_agreement_raw)
    vote_mean = sum(votes) / max(len(votes), 1)
    return _clamp01((0.55 * proxy_agreement_raw) + (0.45 * vote_mean))


def _fx_carry_proxy(pair_changes: Mapping[str, float]) -> float:
    carry_components = [
        _to_float(pair_changes.get("USDJPY"), 0.0),
        _to_float(pair_changes.get("USDCHF"), 0.0),
        _to_float(pair_changes.get("USDCAD"), 0.0),
        -_to_float(pair_changes.get("EURUSD"), 0.0),
        -_to_float(pair_changes.get("GBPUSD"), 0.0),
        -_to_float(pair_changes.get("AUDUSD"), 0.0),
    ]
    finite = [value for value in carry_components if math.isfinite(value)]
    if not finite:
        return 0.5
    return _signed_norm(sum(finite) / len(finite), 0.04)


def collect_fx_market_context(*, timeout: float = 20.0) -> tuple[dict[str, Any], dict[str, Any]]:
    now = datetime.now(timezone.utc)
    warnings: list[str] = []
    pressure_contract = _collector_pressure_contract(now)
    _apply_collector_pressure_env(pressure_contract)
    if bool(pressure_contract.get("active", False)):
        timeout = min(float(timeout), _to_float(pressure_contract.get("timeout_cap_seconds"), float(timeout)))
    twelve_data_enabled = str(os.getenv("FX_MARKET_CONTEXT_TWELVE_DATA_ENABLED", "1")).strip().lower() in {"1", "true", "yes", "on"}
    twelve_data_api_key = str(os.getenv("TWELVE_DATA_API_KEY", "")).strip()
    twelve_data_pairs, twelve_data_budget = _configured_twelve_data_pairs()
    twelve_data_interval = str(os.getenv("FX_TWELVE_DATA_INTERVAL", "5min") or "5min").strip() or "5min"
    twelve_data_outputsize = max(int(str(os.getenv("FX_TWELVE_DATA_OUTPUTSIZE", "72") or "72")), 4)
    alpha_vantage_enabled = str(os.getenv("FX_MARKET_CONTEXT_ALPHA_VANTAGE_ENABLED", "0")).strip().lower() in {"1", "true", "yes", "on"}
    alpha_vantage_api_key = str(os.getenv("ALPHA_VANTAGE_API_KEY", "")).strip()
    source_status: dict[str, Any] = {
        "ecb": {"ok": False, "url": ECB_FX_HIST_90D_URL, "rows": 0, "error": None, **_source_contract("ecb"), "freshness_norm": 0.0},
        "fed_h10": {"ok": False, "url": FED_H10_CURRENT_URL, "pair_count": 0, "error": None, **_source_contract("fed_h10"), "freshness_norm": 0.0},
        "frankfurter": {"ok": False, "url": FRANKFURTER_LATEST_URL, "pair_count": 0, "error": None, **_source_contract("frankfurter"), "freshness_norm": 0.0},
        "macro_cross_asset": {"ok": False, "path": str(PROJECT_ROOT / "exports" / "external_context" / "macro_cross_asset_latest.json"), "error": None},
        "market_proxy": {"ok": False, "symbols": 0, "error": None},
    }
    if twelve_data_enabled:
        source_status["twelve_data"] = {
            "ok": False,
            "configured": bool(twelve_data_api_key),
            "pairs_requested": list(twelve_data_budget["requested_pairs"]),
            "pairs_selected": list(twelve_data_budget["selected_pairs"]),
            "pairs_deferred": list(twelve_data_budget["deferred_pairs"]),
            "pairs_ok": 0,
            "interval": twelve_data_interval,
            "outputsize": twelve_data_outputsize,
            "throttle_prevention": {
                "max_credits_per_minute": int(twelve_data_budget["max_credits_per_minute"]),
                "credit_reserve": int(twelve_data_budget["credit_reserve"]),
                "credit_budget_per_run": int(twelve_data_budget["credit_budget_per_run"]),
                "pairs_per_run": int(twelve_data_budget["pairs_per_run"]),
            },
            "error": None,
            **_source_contract("twelve_data"),
            "freshness_norm": 0.0,
        }
    if alpha_vantage_enabled:
        source_status["alpha_vantage"] = {
            "ok": False,
            "configured": bool(alpha_vantage_api_key),
            "pair": "EURUSD",
            "interval": "5min",
            "rows": 0,
            "error": None,
            **_source_contract("alpha_vantage"),
            "freshness_norm": 0.0,
        }

    ecb_rows: list[dict[str, Any]] = []
    try:
        ecb_rows = _parse_ecb_hist_90d(_http_text(ECB_FX_HIST_90D_URL, timeout=timeout))
        source_status["ecb"]["ok"] = len(ecb_rows) >= 2
        source_status["ecb"]["rows"] = len(ecb_rows)
        source_status["ecb"]["freshness_norm"] = 1.0 if len(ecb_rows) >= 2 else 0.0
        if not source_status["ecb"]["ok"]:
            source_status["ecb"]["error"] = "insufficient_rows"
    except (HTTPError, URLError, RuntimeError, TimeoutError, ET.ParseError, OSError) as exc:
        source_status["ecb"]["error"] = str(exc)

    currency_reference_rates, currency_reference_changes, ecb_future_dates = _latest_currency_reference_history(
        ecb_rows,
        as_of_date=now.date().isoformat(),
    )
    ecb_rows_as_of = [row for row in ecb_rows if str(row.get("date") or "") <= now.date().isoformat()]
    source_status["ecb"]["future_observations_excluded"] = ecb_future_dates
    source_status["ecb"]["future_observation_selected"] = False
    source_status["ecb"]["currency_count"] = len(currency_reference_rates)
    latest_pairs, previous_pairs = _latest_pair_history(ecb_rows_as_of)
    ecb_latest_pairs = dict(latest_pairs)
    pair_changes = {pair: _pct_change(latest_pairs.get(pair), previous_pairs.get(pair)) for pair in PAIR_SYMBOLS}

    alpha_vantage_intraday: dict[str, Any] = {}
    twelve_data_intraday: dict[str, Any] = {}
    frankfurter_latest_pairs: dict[str, float] = {}
    try:
        frankfurter_latest_pairs = _parse_frankfurter_latest(_http_json(FRANKFURTER_LATEST_URL, timeout=timeout))
        source_status["frankfurter"]["ok"] = len(frankfurter_latest_pairs) >= 3
        source_status["frankfurter"]["pair_count"] = len(frankfurter_latest_pairs)
        source_status["frankfurter"]["freshness_norm"] = 1.0 if source_status["frankfurter"]["ok"] else 0.0
        if not source_status["frankfurter"]["ok"]:
            source_status["frankfurter"]["error"] = "insufficient_pairs"
        for pair, value in frankfurter_latest_pairs.items():
            latest_pairs.setdefault(pair, value)
    except (RuntimeError, TimeoutError, OSError, ValueError) as exc:
        source_status["frankfurter"]["error"] = str(exc)

    fed_h10 = {}
    try:
        fed_h10 = _parse_fed_h10_current(_http_text(FED_H10_CURRENT_URL, timeout=timeout))
        source_status["fed_h10"]["ok"] = bool(fed_h10.get("ok"))
        source_status["fed_h10"]["pair_count"] = int(fed_h10.get("pair_count", 0) or 0)
        source_status["fed_h10"]["error"] = fed_h10.get("error")
        source_status["fed_h10"]["freshness_norm"] = 1.0 if bool(fed_h10.get("ok")) else 0.0
        for pair, value in (fed_h10.get("pair_values") or {}).items():
            latest_pairs[str(pair)] = _to_float(value, latest_pairs.get(str(pair), 0.0))
        for pair, value in (fed_h10.get("previous_pair_values") or {}).items():
            previous_pairs[str(pair)] = _to_float(value, previous_pairs.get(str(pair), 0.0))
        pair_changes = {pair: _pct_change(latest_pairs.get(pair), previous_pairs.get(pair)) for pair in PAIR_SYMBOLS}
    except (HTTPError, URLError, RuntimeError, TimeoutError, OSError) as exc:
        source_status["fed_h10"]["error"] = str(exc)

    if alpha_vantage_enabled and alpha_vantage_api_key:
        alpha_vantage_intraday = _alpha_vantage_intraday(
            api_key=alpha_vantage_api_key,
            from_symbol="EUR",
            to_symbol="USD",
            interval="5min",
            timeout=timeout,
        )
        source_status["alpha_vantage"]["ok"] = bool(alpha_vantage_intraday.get("ok"))
        source_status["alpha_vantage"]["rows"] = int(alpha_vantage_intraday.get("rows", 0) or 0)
        source_status["alpha_vantage"]["error"] = alpha_vantage_intraday.get("error")
        source_status["alpha_vantage"]["freshness_norm"] = 1.0 if bool(alpha_vantage_intraday.get("ok")) else 0.0
        if alpha_vantage_intraday.get("ok"):
            av_change = _pct_change(
                _to_float(alpha_vantage_intraday.get("latest_close"), 0.0),
                _to_float(alpha_vantage_intraday.get("previous_close"), 0.0),
            )
            if abs(av_change) > abs(pair_changes.get("EURUSD", 0.0)):
                pair_changes["EURUSD"] = av_change
            if _to_float(alpha_vantage_intraday.get("latest_close"), 0.0) > 0.0:
                latest_pairs["EURUSD"] = _to_float(alpha_vantage_intraday.get("latest_close"), 0.0)
            if _to_float(alpha_vantage_intraday.get("previous_close"), 0.0) > 0.0:
                previous_pairs["EURUSD"] = _to_float(alpha_vantage_intraday.get("previous_close"), 0.0)
            pair_changes["EURUSD"] = _pct_change(latest_pairs.get("EURUSD"), previous_pairs.get("EURUSD"))
    elif alpha_vantage_enabled:
        source_status["alpha_vantage"]["error"] = "api_key_missing"

    if twelve_data_enabled and twelve_data_api_key and twelve_data_pairs:
        twelve_data_errors: list[str] = []
        for pair in twelve_data_pairs:
            result = _twelve_data_time_series(
                api_key=twelve_data_api_key,
                pair_symbol=pair,
                interval=twelve_data_interval,
                outputsize=twelve_data_outputsize,
                timeout=timeout,
            )
            twelve_data_intraday[pair] = dict(result)
            if result.get("ok"):
                latest_pairs[pair] = _to_float(result.get("latest_close"), latest_pairs.get(pair, 0.0))
                previous_pairs[pair] = _to_float(result.get("session_close"), previous_pairs.get(pair, 0.0))
                pair_changes[pair] = _pct_change(latest_pairs.get(pair), previous_pairs.get(pair))
            else:
                twelve_data_errors.append(f"{pair}:{result.get('error')}")
        ok_pairs = sum(1 for row in twelve_data_intraday.values() if bool(row.get("ok")))
        source_status["twelve_data"]["ok"] = ok_pairs > 0
        source_status["twelve_data"]["pairs_ok"] = ok_pairs
        source_status["twelve_data"]["error"] = None if ok_pairs > 0 else (";".join(twelve_data_errors[:3]) or "no_pairs_ok")
        source_status["twelve_data"]["freshness_norm"] = 1.0 if ok_pairs > 0 else 0.0
    elif twelve_data_enabled and twelve_data_api_key:
        source_status["twelve_data"]["error"] = "credit_budget_reserved"
    elif twelve_data_enabled:
        source_status["twelve_data"]["error"] = "api_key_missing"

    usd_components = [
        -pair_changes.get("EURUSD", 0.0),
        pair_changes.get("USDJPY", 0.0),
        -pair_changes.get("GBPUSD", 0.0),
        pair_changes.get("USDCHF", 0.0),
        pair_changes.get("USDCAD", 0.0),
        -pair_changes.get("AUDUSD", 0.0),
    ]
    usd_strength_raw = sum(usd_components) / len([x for x in usd_components if math.isfinite(x)]) if usd_components else 0.0
    macro_dispersion_raw = 0.0
    if pair_changes:
        centered = [float(value) for value in pair_changes.values() if math.isfinite(float(value))]
        if centered:
            mean = sum(centered) / len(centered)
            macro_dispersion_raw = math.sqrt(sum((value - mean) ** 2 for value in centered) / max(len(centered), 1))

    macro_cross_asset = _macro_cross_asset_context()
    macro_cross_ok = bool(macro_cross_asset)
    source_status["macro_cross_asset"]["ok"] = macro_cross_ok
    if not macro_cross_ok:
        source_status["macro_cross_asset"]["error"] = "missing_macro_cross_asset_latest"
    dollar_index = 0.0
    if macro_cross_ok:
        cross_asset = macro_cross_asset.get("cross_asset") if isinstance(macro_cross_asset.get("cross_asset"), Mapping) else {}
        dollar_index = _to_float(fed_h10.get("broad_index"), 0.0) or _to_float(cross_asset.get("dollar_index_broad"), 0.0)

    latest_market = _latest_market_snapshot()
    source_status["market_proxy"]["ok"] = len(latest_market) > 0
    source_status["market_proxy"]["symbols"] = len(latest_market)
    if not latest_market:
        source_status["market_proxy"]["error"] = "missing_market_proxy_snapshot"

    proxy_agreement_raw, proxy_checks = _proxy_agreement(pair_changes, latest_market, usd_strength_raw)
    risk_alignment, crypto_alignment = _risk_alignment(latest_market, usd_strength_raw)
    session_norms = _fx_session_state_norms(now)
    dxy_yield_confirmation = _fx_dxy_yield_confirmation(
        latest_market,
        usd_strength_raw=usd_strength_raw,
        proxy_agreement_raw=proxy_agreement_raw,
    )
    carry_proxy = _fx_carry_proxy(pair_changes)

    confidence = 0.0
    if source_status["ecb"]["ok"]:
        confidence = 0.55
    if source_status["fed_h10"]["ok"]:
        confidence += 0.15
    if twelve_data_enabled and source_status.get("twelve_data", {}).get("ok"):
        confidence += 0.20
    if alpha_vantage_enabled and source_status.get("alpha_vantage", {}).get("ok"):
        confidence += 0.10
    if source_status["market_proxy"]["ok"]:
        confidence += 0.25
    if macro_cross_ok:
        confidence += 0.20
    confidence = _clamp01(confidence)

    if not source_status["ecb"]["ok"]:
        warnings.append("ecb_fx_feed_unavailable")
    if not source_status["fed_h10"]["ok"]:
        warnings.append("fed_h10_fx_feed_unavailable")
    if not source_status["frankfurter"]["ok"]:
        warnings.append("frankfurter_fx_reference_unavailable")
    if twelve_data_enabled and source_status["twelve_data"]["configured"] and not source_status["twelve_data"]["ok"]:
        td_error = str(source_status["twelve_data"].get("error") or "")
        warnings.append("twelve_data_credit_budget_reserved" if td_error == "credit_budget_reserved" else "twelve_data_fx_intraday_unavailable")
    if alpha_vantage_enabled and source_status["alpha_vantage"]["configured"] and not source_status["alpha_vantage"]["ok"]:
        warnings.append("alpha_vantage_fx_intraday_unavailable")
    if not source_status["market_proxy"]["ok"]:
        warnings.append("market_proxy_snapshot_missing")
    elif proxy_agreement_raw <= 0.0:
        warnings.append("proxy_agreement_sparse")

    canonical_reconciliation = _canonical_pair_reconciliation(
        ecb_pairs=ecb_latest_pairs,
        fed_pairs=fed_h10.get("pair_values") if isinstance(fed_h10.get("pair_values"), Mapping) else {},
        twelve_data_intraday=twelve_data_intraday,
        alpha_vantage_intraday=alpha_vantage_intraday,
        frankfurter_pairs=frankfurter_latest_pairs,
    )
    provider_divergence_warnings = [
        pair
        for pair, row in canonical_reconciliation.items()
        if str(row.get("divergence_severity") or "") == "warning"
    ]
    provider_divergence_basis_watch = [
        pair
        for pair, row in canonical_reconciliation.items()
        if str(row.get("divergence_severity") or "") == "basis_watch"
    ]
    if provider_divergence_warnings:
        warnings.append("fx_provider_divergence_detected")

    global_features = {
        "fx_official_data_available": 1.0 if source_status["ecb"]["ok"] else 0.0,
        "fx_eurusd_level_norm": _clamp01(_to_float(latest_pairs.get("EURUSD"), 0.0) / 2.0),
        "fx_eurusd_momentum_norm": _signed_norm(pair_changes.get("EURUSD", 0.0), 0.05),
        "fx_usdjpy_level_norm": _clamp01(_to_float(latest_pairs.get("USDJPY"), 0.0) / 200.0),
        "fx_usdjpy_momentum_norm": _signed_norm(pair_changes.get("USDJPY", 0.0), 0.05),
        "fx_gbpusd_level_norm": _clamp01(_to_float(latest_pairs.get("GBPUSD"), 0.0) / 2.0),
        "fx_gbpusd_momentum_norm": _signed_norm(pair_changes.get("GBPUSD", 0.0), 0.05),
        "fx_usd_strength_norm": _signed_norm(usd_strength_raw, 0.04),
        "fx_usd_broad_index_norm": _clamp01((dollar_index - 70.0) / 60.0) if dollar_index > 0.0 else _signed_norm(usd_strength_raw, 0.04),
        "fx_proxy_agreement_norm": _clamp01(proxy_agreement_raw),
        "fx_risk_on_alignment_norm": _clamp01(risk_alignment),
        "fx_crypto_alignment_norm": _clamp01(crypto_alignment),
        "fx_macro_dispersion_norm": _clamp01(macro_dispersion_raw / 0.03),
        "fx_corr_confidence_norm": confidence,
        "fx_session_asia_norm": float(session_norms["fx_session_asia_norm"]),
        "fx_session_london_norm": float(session_norms["fx_session_london_norm"]),
        "fx_session_ny_norm": float(session_norms["fx_session_ny_norm"]),
        "fx_rollover_risk_norm": float(session_norms["fx_rollover_risk_norm"]),
        "fx_dxy_yield_confirmation_norm": dxy_yield_confirmation,
        "fx_carry_proxy_norm": carry_proxy,
    }

    symbol_features = {
        symbol: dict(global_features)
        for symbol in set(PROXY_SYMBOLS) | set(PAIR_SYMBOLS) | {"SPY", "QQQ", "BTC-USD", "ETH-USD", "SOL-USD"}
    }
    for symbol, row in symbol_features.items():
        if symbol in {"UUP", "EUO", "YCS"}:
            row["fx_usd_strength_norm"] = _clamp01(min(global_features["fx_usd_strength_norm"] + 0.08, 1.0))
        elif symbol in {"FXE", "FXB", "FXA", "CYB"}:
            row["fx_usd_strength_norm"] = _clamp01(max(global_features["fx_usd_strength_norm"] - 0.08, 0.0))
        elif symbol == "FXY":
            row["fx_usdjpy_momentum_norm"] = _clamp01(1.0 - global_features["fx_usdjpy_momentum_norm"])

    payload = {
        "timestamp_utc": now.isoformat(),
        "provider": "fx_market_context",
        "collection_contract": {
            "source_contracts": {
                name: {
                    "source_confidence_norm": float((row or {}).get("source_confidence_norm", 0.0) or 0.0),
                    "schema_confidence_norm": float((row or {}).get("schema_confidence_norm", 0.0) or 0.0),
                    "freshness_norm": float((row or {}).get("freshness_norm", 0.0) or 0.0),
                }
                for name, row in source_status.items()
                if isinstance(row, Mapping) and "source_confidence_norm" in row
            },
            "provider_confidence_norm": round(
                sum(
                    float((row or {}).get("source_confidence_norm", 0.0) or 0.0)
                    for row in source_status.values()
                    if isinstance(row, Mapping) and bool(row.get("ok")) and "source_confidence_norm" in row
                ) / max(
                    sum(1 for row in source_status.values() if isinstance(row, Mapping) and bool(row.get("ok")) and "source_confidence_norm" in row),
                    1,
                ),
                6,
            ),
        },
        "sources": source_status,
        "derived": {
            "calendar_features": {},
            "news_features": {},
            "global_features": global_features,
            "symbol_features": symbol_features,
            "pair_values": {key: round(_to_float(value), 6) for key, value in latest_pairs.items()},
            "pair_changes": {key: round(_to_float(value), 6) for key, value in pair_changes.items()},
            "currency_reference_rates": currency_reference_rates,
            "currency_reference_changes": {
                key: round(_to_float(value), 8) for key, value in currency_reference_changes.items()
            },
            "pair_intraday_quotes": {
                key: {
                    "ok": bool(value.get("ok")),
                    "latest_ts": value.get("latest_ts"),
                    "latest_close": round(_to_float(value.get("latest_close"), 0.0), 6),
                    "session_close": round(_to_float(value.get("session_close"), 0.0), 6),
                    "rows": int(value.get("rows", 0) or 0),
                }
                for key, value in twelve_data_intraday.items()
                if isinstance(value, Mapping)
            },
            "intraday_reference": {
                "fed_h10": {
                    "ok": bool(fed_h10.get("ok")),
                    "pair_count": int(fed_h10.get("pair_count", 0) or 0),
                },
                "twelve_data": (
                    {
                        "ok": bool(source_status.get("twelve_data", {}).get("ok")),
                        "pairs_ok": int(source_status.get("twelve_data", {}).get("pairs_ok", 0) or 0),
                        "pairs_requested": list(twelve_data_budget["requested_pairs"]),
                        "pairs_selected": list(twelve_data_pairs),
                        "pairs_deferred": list(twelve_data_budget["deferred_pairs"]),
                        "interval": twelve_data_interval,
                    }
                    if twelve_data_enabled
                    else {"enabled": False}
                ),
                "alpha_vantage": (
                    {
                        "ok": bool(alpha_vantage_intraday.get("ok")),
                        "latest_ts": alpha_vantage_intraday.get("latest_ts"),
                        "rows": int(alpha_vantage_intraday.get("rows", 0) or 0),
                    }
                    if alpha_vantage_enabled
                    else {"enabled": False}
                ),
            },
            "proxy_checks": proxy_checks,
            "latest_market": latest_market,
            "canonical_reconciliation": canonical_reconciliation,
        },
    }
    health = {
        "timestamp_utc": now.isoformat(),
        "ok": bool(source_status["ecb"]["ok"] or source_status.get("twelve_data", {}).get("ok")),
        "source_count": len(source_status),
        "ok_source_count": sum(1 for row in source_status.values() if isinstance(row, Mapping) and bool(row.get("ok", False))),
        "official_pairs": len(latest_pairs),
        "proxy_symbols_observed": len(latest_market),
        "proxy_agreement_norm": round(proxy_agreement_raw, 6),
        "direct_forex_execution_supported": False,
        "direct_forex_execution_reason": "schwab_official_api_forex_unverified",
        "warning_count": len(warnings),
        "warnings": warnings,
        "sources": source_status,
        "canonical_pairs": len(canonical_reconciliation),
        "provider_divergence_pairs": provider_divergence_warnings,
        "provider_divergence_basis_watch_pairs": provider_divergence_basis_watch,
        "source_contracts": payload["collection_contract"]["source_contracts"],
        "pressure_contract": pressure_contract,
    }
    return payload, health


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect free FX context from official feeds plus live proxy markets.")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    external_path = PROJECT_ROOT / "exports" / "external_context" / "fx_market_context_latest.json"
    health_path = PROJECT_ROOT / "governance" / "health" / "fx_market_context_sync_latest.json"
    now = datetime.now(timezone.utc)
    pressure_contract = _collector_pressure_contract(now)
    lock_handle = None
    if bool(pressure_contract.get("active", False)):
        min_interval = int(pressure_contract.get("min_interval_seconds") or 0)
        if _pressure_min_interval_active(external_path, min_interval=min_interval, now=now):
            health = _pressure_skip_health(
                reason="pressure_min_interval_active",
                contract=pressure_contract,
                external_path=external_path,
                health_path=health_path,
                now=now,
            )
            _write_json(health_path, health)
            if args.json:
                print(json.dumps(health, ensure_ascii=True))
            else:
                print(
                    "fx_market_context "
                    f"skipped={health.get('skip_reason')} "
                    f"age_seconds={health.get('previous_health_age_seconds')}"
                )
                print(f"fx_market_context_latest={external_path}")
                print(f"fx_market_context_sync_latest={health_path}")
            return 0 if bool(health.get("ok", False)) else 1
        lock_path = Path(str(pressure_contract.get("lock_path") or PROJECT_ROOT / "governance" / "locks" / "fx_market_context.lock"))
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_handle = lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            health = _pressure_skip_health(
                reason="pressure_single_flight_lock_active",
                contract=pressure_contract,
                external_path=external_path,
                health_path=health_path,
                now=now,
            )
            _write_json(health_path, health)
            if args.json:
                print(json.dumps(health, ensure_ascii=True))
            else:
                print(f"fx_market_context skipped={health.get('skip_reason')}")
                print(f"fx_market_context_latest={external_path}")
                print(f"fx_market_context_sync_latest={health_path}")
            return 0 if bool(health.get("ok", False)) else 1
    try:
        payload, health = collect_fx_market_context(timeout=max(float(args.timeout), 5.0))
    finally:
        if lock_handle is not None:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                lock_handle.close()
            except Exception:
                pass

    _write_json(external_path, payload)
    _write_json(health_path, health)
    with ops_data_plane.connect(PROJECT_ROOT) as conn:
        run_uid = ops_data_plane.record_collector_run(
            conn,
            collector_key="fx_market_context",
            cache_key="fx_market_context",
            command=list(sys.argv),
            expect_paths=[str(external_path), str(health_path)],
            fingerprint_files=[],
            command_fingerprint=hashlib.sha256(" ".join(sys.argv).encode("utf-8")).hexdigest(),
            skipped=False,
            rc=0 if bool(health.get("ok", False)) else 1,
            started_utc=str(health.get("timestamp_utc") or ""),
            finished_utc=datetime.now(timezone.utc).isoformat(),
            payload_sha256=ops_data_plane.file_sha256(external_path),
            metadata={
                "health": health,
                "canonical_pairs": int(health.get("canonical_pairs", 0) or 0),
            },
            commit=False,
        )
        for source_name, row in (health.get("sources") or {}).items():
            if not isinstance(row, Mapping):
                continue
            source_key = ops_data_plane.normalize_entity_key(PROJECT_ROOT, source_name, namespace="source")
            ops_data_plane.record_watermark(
                conn,
                collector_key="fx_market_context",
                source_name=source_key,
                entity_key=source_key,
                watermark_type="collector_sync",
                watermark_value=str(health.get("timestamp_utc") or ""),
                payload_sha256=ops_data_plane.file_sha256(health_path),
                metadata={
                    "run_uid": run_uid,
                    "source_key": source_key,
                    "source_status": dict(row),
                },
                commit=False,
            )
        for pair, row in (payload.get("derived") or {}).get("canonical_reconciliation", {}).items():
            if not isinstance(row, Mapping):
                continue
            ops_data_plane.record_canonical_reconciliation(
                conn,
                domain="fx_pair",
                entity_key=str(pair),
                canonical_source=str(row.get("canonical_source") or ""),
                confidence=_to_float(row.get("confidence_norm"), 0.0),
                divergence_score=_to_float(row.get("divergence_ratio"), 0.0),
                canonical_payload={
                    "pair": pair,
                    "canonical_value": _to_float(row.get("canonical_value"), 0.0),
                    "median_value": _to_float(row.get("median_value"), 0.0),
                },
                provider_votes=row.get("provider_votes") if isinstance(row.get("provider_votes"), Mapping) else {},
                metadata={"run_uid": run_uid},
                commit=False,
            )
        conn.commit()

    if args.json:
        print(json.dumps(health, ensure_ascii=True))
    else:
        print(f"fx_market_context ok={health.get('ok')} proxy_symbols={health.get('proxy_symbols_observed')}")
        print(f"fx_market_context_latest={external_path}")
        print(f"fx_market_context_sync_latest={health_path}")
    return 0 if bool(health.get("ok", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
