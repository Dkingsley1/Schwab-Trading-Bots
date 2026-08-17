#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import gzip
import json
import math
import os
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "analyst_consensus_context_v1.json"
LATEST_PATH = PROJECT_ROOT / "exports" / "external_context" / "analyst_consensus_latest.json"
HEALTH_PATH = PROJECT_ROOT / "governance" / "health" / "analyst_consensus_latest.json"
CACHE_PATH = PROJECT_ROOT / "data" / "external_context" / "analyst_consensus_cache_latest.json"
HISTORY_ROOT = PROJECT_ROOT / "data" / "external_context" / "analyst_consensus_history"
UNIVERSE_PATH = PROJECT_ROOT / "governance" / "health" / "schwab_symbol_news_latest.json"
QUOTA_STATE_PATH = PROJECT_ROOT / "governance" / "runtime" / "analyst_consensus_quota_state.json"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _parse_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _safe_float(value: Any) -> float | None:
    try:
        result = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _first(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row.get(key) not in (None, "", "None"):
            return row.get(key)
    return None


def _estimate_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    direct = payload.get("estimates")
    if isinstance(direct, list):
        return [row for row in direct if isinstance(row, Mapping)]
    rows: list[Mapping[str, Any]] = []
    for key in (
        "quarterlyEarningsEstimates",
        "annualEarningsEstimates",
        "quarterly_earnings_estimates",
        "annual_earnings_estimates",
    ):
        values = payload.get(key)
        if isinstance(values, list):
            rows.extend(row for row in values if isinstance(row, Mapping))
    return rows


def parse_alpha_vantage_estimates(
    payload: Mapping[str, Any],
    *,
    symbol: str,
    collected_at_utc: datetime,
) -> dict[str, Any]:
    if any(payload.get(key) for key in ("Information", "Note", "Error Message")):
        return {}
    clean_rows: list[dict[str, Any]] = []
    for raw in _estimate_rows(payload):
        eps_average = _safe_float(_first(raw, "epsEstimateAverage", "eps_estimate_average"))
        revenue_average = _safe_float(_first(raw, "revenueEstimateAverage", "revenue_estimate_average"))
        if eps_average is None and revenue_average is None:
            continue
        row = {
            "date": str(_first(raw, "date", "fiscalDateEnding", "fiscal_date_ending") or ""),
            "horizon": str(_first(raw, "horizon", "period") or ""),
            "eps_estimate_average": eps_average,
            "eps_estimate_high": _safe_float(_first(raw, "epsEstimateHigh", "eps_estimate_high")),
            "eps_estimate_low": _safe_float(_first(raw, "epsEstimateLow", "eps_estimate_low")),
            "eps_analyst_count": _safe_float(_first(raw, "epsEstimateAnalystCount", "eps_estimate_analyst_count")),
            "eps_average_7_days_ago": _safe_float(_first(raw, "epsEstimateAverage7DaysAgo", "eps_estimate_average_7_days_ago")),
            "eps_average_30_days_ago": _safe_float(_first(raw, "epsEstimateAverage30DaysAgo", "eps_estimate_average_30_days_ago")),
            "eps_average_60_days_ago": _safe_float(_first(raw, "epsEstimateAverage60DaysAgo", "eps_estimate_average_60_days_ago")),
            "eps_average_90_days_ago": _safe_float(_first(raw, "epsEstimateAverage90DaysAgo", "eps_estimate_average_90_days_ago")),
            "eps_revision_up_7_days": _safe_float(_first(raw, "epsEstimateRevisionUpTrailing7Days", "eps_revision_up_trailing_7_days")),
            "eps_revision_down_7_days": _safe_float(_first(raw, "epsEstimateRevisionDownTrailing7Days", "eps_revision_down_trailing_7_days")),
            "eps_revision_up_30_days": _safe_float(_first(raw, "epsEstimateRevisionUpTrailing30Days", "eps_revision_up_trailing_30_days")),
            "eps_revision_down_30_days": _safe_float(_first(raw, "epsEstimateRevisionDownTrailing30Days", "eps_revision_down_trailing_30_days")),
            "revenue_estimate_average": revenue_average,
            "revenue_estimate_high": _safe_float(_first(raw, "revenueEstimateHigh", "revenue_estimate_high")),
            "revenue_estimate_low": _safe_float(_first(raw, "revenueEstimateLow", "revenue_estimate_low")),
            "revenue_analyst_count": _safe_float(_first(raw, "revenueEstimateAnalystCount", "revenue_estimate_analyst_count")),
        }
        clean_rows.append({key: value for key, value in row.items() if value is not None})
    if not clean_rows:
        return {}
    return {
        "symbol": str(payload.get("symbol") or symbol).upper(),
        "collected_at_utc": collected_at_utc.isoformat(),
        "provider": "alpha_vantage",
        "provider_function": "EARNINGS_ESTIMATES",
        "estimates": clean_rows[:12],
    }


def parse_nasdaq_earnings_forecast(
    payload: Mapping[str, Any],
    *,
    symbol: str,
    collected_at_utc: datetime,
) -> dict[str, Any]:
    status = payload.get("status") if isinstance(payload.get("status"), Mapping) else {}
    if _safe_float(status.get("rCode")) not in (None, 200.0):
        return {}
    data = payload.get("data") if isinstance(payload.get("data"), Mapping) else {}
    clean_rows: list[dict[str, Any]] = []
    for source_key, horizon in (("quarterlyForecast", "quarterly"), ("yearlyForecast", "annual")):
        section = data.get(source_key) if isinstance(data.get(source_key), Mapping) else {}
        rows = section.get("rows") if isinstance(section.get("rows"), list) else []
        for raw in rows:
            if not isinstance(raw, Mapping):
                continue
            average = _safe_float(raw.get("consensusEPSForecast"))
            if average is None:
                continue
            row = {
                "date": str(raw.get("fiscalEnd") or ""),
                "horizon": horizon,
                "eps_estimate_average": average,
                "eps_estimate_high": _safe_float(raw.get("highEPSForecast")),
                "eps_estimate_low": _safe_float(raw.get("lowEPSForecast")),
                "eps_analyst_count": _safe_float(raw.get("noOfEstimates")),
                "eps_revision_up_30_days": _safe_float(raw.get("up")),
                "eps_revision_down_30_days": _safe_float(raw.get("down")),
                "revision_window_days": 28,
            }
            clean_rows.append({key: value for key, value in row.items() if value is not None})
    if not clean_rows:
        return {}
    return {
        "symbol": str(data.get("symbol") or symbol).upper(),
        "collected_at_utc": collected_at_utc.isoformat(),
        "provider": "nasdaq_analyst_forecast",
        "provider_function": "earnings_forecast",
        "provider_documentation_url": (
            f"https://www.nasdaq.com/market-activity/stocks/{str(symbol).lower()}/earnings"
        ),
        "availability_time_policy": "collection_time_used_when_provider_as_of_is_absent",
        "estimates": clean_rows[:12],
    }


def _symbol_feature_row(row: Mapping[str, Any]) -> dict[str, float]:
    estimates = row.get("estimates") if isinstance(row.get("estimates"), list) else []
    analyst_counts: list[float] = []
    dispersions: list[float] = []
    revision_up = revision_down = 0.0
    revision_observations = 0
    for estimate in estimates:
        if not isinstance(estimate, Mapping):
            continue
        analyst_count = _safe_float(estimate.get("eps_analyst_count"))
        if analyst_count is None:
            analyst_count = _safe_float(estimate.get("revenue_analyst_count"))
        if analyst_count is not None:
            analyst_counts.append(max(analyst_count, 0.0))
        for prefix in ("eps", "revenue"):
            average = _safe_float(estimate.get(f"{prefix}_estimate_average"))
            high = _safe_float(estimate.get(f"{prefix}_estimate_high"))
            low = _safe_float(estimate.get(f"{prefix}_estimate_low"))
            if average is not None and high is not None and low is not None:
                dispersions.append(_clamp01(abs(high - low) / max(2.0 * abs(average), 1e-6)))
        up = sum(
            value
            for value in (
                _safe_float(estimate.get("eps_revision_up_7_days")),
                _safe_float(estimate.get("eps_revision_up_30_days")),
            )
            if value is not None
        )
        down = sum(
            value
            for value in (
                _safe_float(estimate.get("eps_revision_down_7_days")),
                _safe_float(estimate.get("eps_revision_down_30_days")),
            )
            if value is not None
        )
        revision_count_present = any(
            key in estimate
            for key in (
                "eps_revision_up_7_days",
                "eps_revision_down_7_days",
                "eps_revision_up_30_days",
                "eps_revision_down_30_days",
            )
        )
        if revision_count_present or any("days_ago" in str(key) for key in estimate):
            revision_observations += 1
        revision_up += up
        revision_down += down
    revisions = revision_up + revision_down
    direction = 0.5 if revisions <= 0.0 else _clamp01(0.5 + (revision_up - revision_down) / (2.0 * revisions))
    analyst_denominator = max(sum(analyst_counts) / max(len(analyst_counts), 1), 1.0)
    return {
        "consensus_analyst_coverage_norm": _clamp01(analyst_denominator / 20.0),
        "consensus_dispersion_norm": _clamp01(sum(dispersions) / max(len(dispersions), 1)),
        "consensus_revision_activity_norm": _clamp01(revisions / analyst_denominator),
        "consensus_revision_direction_norm": direction,
        "consensus_revision_history_norm": _clamp01(revision_observations / max(len(estimates), 1)),
    }


def _eligible_symbols(project_root: Path, config: Mapping[str, Any], override: str = "") -> list[str]:
    if override.strip():
        candidates = [token.strip().upper() for token in override.split(",")]
    elif isinstance(config.get("required_symbols"), list) and config.get("required_symbols"):
        candidates = [str(symbol).strip().upper() for symbol in config.get("required_symbols", [])]
    else:
        universe = _read_json(project_root / "governance" / "health" / UNIVERSE_PATH.name)
        rows = universe.get("symbols") if isinstance(universe.get("symbols"), Mapping) else {}
        group = str(config.get("universe_group") or "SEC_EDGAR_SYMBOLS")
        candidates = [
            str(symbol).upper()
            for symbol, row in rows.items()
            if isinstance(row, Mapping) and group in (row.get("groups") or [])
        ]
    excluded = {str(symbol).upper() for symbol in config.get("excluded_fund_symbols", [])}
    return sorted(
        {
            symbol
            for symbol in candidates
            if symbol
            and symbol not in excluded
            and symbol.replace(".", "").isalnum()
            and "-" not in symbol
            and "/" not in symbol
        }
    )


def _fresh_cache_rows(
    cache: Mapping[str, Any],
    *,
    now: datetime,
    maximum_age_days: float,
) -> dict[str, dict[str, Any]]:
    rows = cache.get("symbols") if isinstance(cache.get("symbols"), Mapping) else {}
    fresh: dict[str, dict[str, Any]] = {}
    for symbol, row in rows.items():
        if not isinstance(row, Mapping):
            continue
        timestamp = _parse_timestamp(row.get("collected_at_utc"))
        age_days = (now - timestamp).total_seconds() / 86400.0 if timestamp is not None else maximum_age_days + 1.0
        if -0.1 <= age_days <= maximum_age_days and row.get("estimates"):
            fresh[str(symbol).upper()] = dict(row)
    return fresh


def _normalized_quota_state(
    payload: Mapping[str, Any],
    *,
    now: datetime,
    maximum_requests: int,
    provider_id: str = "alpha_vantage",
) -> dict[str, Any]:
    day_utc = now.date().isoformat()
    claimed = 0
    if (
        str(payload.get("date_utc") or "") == day_utc
        and str(payload.get("provider_id") or provider_id) == provider_id
    ):
        try:
            claimed = max(int(payload.get("requests_claimed", 0) or 0), 0)
        except (TypeError, ValueError):
            claimed = 0
    maximum = max(int(maximum_requests), 0)
    claimed = min(claimed, maximum)
    return {
        "provider_id": provider_id,
        "date_utc": day_utc,
        "maximum_requests": maximum,
        "requests_claimed": claimed,
        "requests_remaining": max(maximum - claimed, 0),
        "last_claimed_at_utc": payload.get("last_claimed_at_utc") if claimed else None,
    }


def _daily_quota_status(
    project_root: Path,
    *,
    now: datetime,
    maximum_requests: int,
    provider_id: str = "alpha_vantage",
) -> dict[str, Any]:
    suffix = "" if provider_id == "alpha_vantage" else f"_{provider_id}"
    path = project_root / "governance" / "runtime" / f"analyst_consensus{suffix}_quota_state.json"
    return _normalized_quota_state(
        _read_json(path),
        now=now,
        maximum_requests=maximum_requests,
        provider_id=provider_id,
    )


def _claim_daily_request(
    project_root: Path,
    *,
    now: datetime,
    maximum_requests: int,
    provider_id: str = "alpha_vantage",
) -> tuple[bool, dict[str, Any]]:
    suffix = "" if provider_id == "alpha_vantage" else f"_{provider_id}"
    path = project_root / "governance" / "runtime" / f"analyst_consensus{suffix}_quota_state.json"
    lock_path = path.with_suffix(f"{path.suffix}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        state = _normalized_quota_state(
            _read_json(path),
            now=now,
            maximum_requests=maximum_requests,
            provider_id=provider_id,
        )
        if state["requests_remaining"] <= 0:
            return False, state
        state["requests_claimed"] += 1
        state["requests_remaining"] -= 1
        state["last_claimed_at_utc"] = now.isoformat()
        _atomic_write_json(path, state)
        return True, state


def build_analyst_consensus_payload(
    *,
    config: Mapping[str, Any],
    universe_symbols: list[str],
    cache_rows: Mapping[str, Mapping[str, Any]],
    now_utc: datetime,
    provider_enabled: bool,
    provider_configured: bool,
    attempted_symbols: list[str] | None = None,
    errors: Mapping[str, str] | None = None,
    quota_state: Mapping[str, Any] | None = None,
    provider_states: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    universe_set = set(universe_symbols)
    symbol_features = {
        symbol: _symbol_feature_row(row)
        for symbol, row in sorted(cache_rows.items())
        if symbol in universe_set
    }
    covered = len(symbol_features)
    universe_count = len(universe_symbols)
    coverage_ratio = covered / max(universe_count, 1)
    revision_ready = sum(
        1 for row in symbol_features.values() if row.get("consensus_revision_history_norm", 0.0) > 0.0
    )
    revision_ratio = revision_ready / max(covered, 1)
    minimum_symbols = int(config.get("minimum_covered_symbols", 16) or 16)
    minimum_coverage = float(config.get("minimum_universe_coverage_ratio", 0.75) or 0.75)
    minimum_revision = float(config.get("minimum_revision_history_ratio", 0.75) or 0.75)
    expected_universe_count = int(config.get("expected_universe_symbol_count", 0) or 0)
    required_symbols = {
        str(symbol).strip().upper()
        for symbol in config.get("required_symbols", [])
        if str(symbol).strip()
    }
    missing_required_symbols = sorted(required_symbols - set(symbol_features))
    reasons: list[str] = []
    if expected_universe_count and universe_count != expected_universe_count:
        reasons.append("governed_universe_count_mismatch")
    if required_symbols and universe_set != required_symbols:
        reasons.append("governed_universe_membership_mismatch")
    if covered < minimum_symbols:
        reasons.append("minimum_symbol_coverage_not_met")
    if coverage_ratio < minimum_coverage:
        reasons.append("universe_coverage_ratio_not_met")
    if revision_ratio < minimum_revision:
        reasons.append("revision_history_ratio_not_met")
    if missing_required_symbols:
        reasons.append("required_symbols_missing")
    direct_ready = bool(universe_count > 0 and not reasons)
    global_features = {
        key: round(sum(row[key] for row in symbol_features.values()) / max(covered, 1), 8)
        for key in (
            "consensus_analyst_coverage_norm",
            "consensus_dispersion_norm",
            "consensus_revision_activity_norm",
            "consensus_revision_direction_norm",
            "consensus_revision_history_norm",
        )
        if covered > 0
    }
    status = "ready" if direct_ready else "accumulating"
    if not provider_enabled and not covered:
        status = "not_configured"
    elif provider_enabled and not provider_configured and not covered:
        status = "credential_missing"
    return {
        "timestamp_utc": now_utc.isoformat(),
        "schema_version": 1,
        "context_id": config.get("context_id", "analyst_consensus_context_v1"),
        "provider": "Nasdaq Analyst Forecast with optional Alpha Vantage fallback",
        "provider_documentation_url": "https://www.nasdaq.com/market-activity/stocks/aapl/earnings",
        "ok": direct_ready,
        "overall_status": status,
        "direct_evidence_ready": direct_ready,
        "coverage": {
            "universe_symbol_count": universe_count,
            "covered_symbol_count": covered,
            "coverage_ratio": round(coverage_ratio, 6),
            "revision_history_symbol_count": revision_ready,
            "revision_history_ratio": round(revision_ratio, 6),
            "minimum_covered_symbols": minimum_symbols,
            "minimum_universe_coverage_ratio": minimum_coverage,
            "minimum_revision_history_ratio": minimum_revision,
            "expected_universe_symbol_count": expected_universe_count,
            "required_symbols": sorted(required_symbols),
            "missing_required_symbols": missing_required_symbols,
            "provider_symbol_counts": {
                provider: sum(
                    1
                    for symbol, row in cache_rows.items()
                    if symbol in universe_set and str(row.get("provider") or "") == provider
                )
                for provider in sorted(
                    {
                        str(row.get("provider") or "")
                        for symbol, row in cache_rows.items()
                        if symbol in universe_set and str(row.get("provider") or "")
                    }
                )
            },
        },
        "provider_runtime": {
            "enabled": provider_enabled,
            "credential_configured": provider_configured,
            "attempted_symbols": list(attempted_symbols or []),
            "error_count": len(errors or {}),
            "errors": dict(errors or {}),
            "maximum_requests_per_day": int(
                (quota_state or {}).get("maximum_requests", config.get("maximum_requests_per_day", 25)) or 25
            ),
            "daily_quota": dict(quota_state or {}),
            "daily_quota_exhausted": bool(
                quota_state and int(quota_state.get("requests_remaining", 0) or 0) <= 0
            ),
            "providers": dict(provider_states or {}),
            "api_key_exposed": False,
        },
        "point_in_time_contract": {
            "collection_time_is_availability_time": True,
            "forward_fiscal_horizons_are_targets_not_future_leakage": True,
            "append_only_collection_history": True,
            "cache_is_bounded_by_age": True,
            "missing_symbols_are_not_zero_filled": True,
            "full_governed_universe_required": True,
            "commercial_or_live_data_entitlement_inferred": False,
        },
        "derived": {
            "global_features": global_features,
            "symbol_features": symbol_features,
        },
        "symbols": {symbol: dict(row) for symbol, row in sorted(cache_rows.items()) if symbol in universe_set},
        "consumer_contract": {
            "ready": direct_ready,
            "reasons": reasons,
            "covered_symbol_count": covered,
            "coverage_ratio": round(coverage_ratio, 6),
            "revision_history_ratio": round(revision_ratio, 6),
            "required_symbols": sorted(required_symbols),
            "missing_required_symbols": missing_required_symbols,
        },
        "authority_contract": dict(config.get("authority_contract") or {}),
    }


def _fetch_alpha_vantage(
    endpoint: str,
    *,
    symbol: str,
    api_key: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    query = urllib.parse.urlencode(
        {"function": "EARNINGS_ESTIMATES", "symbol": symbol, "apikey": api_key}
    )
    request = urllib.request.Request(
        f"{endpoint}?{query}",
        headers={"User-Agent": "schwab-trading-bot-analyst-consensus/1.0"},
    )
    with urllib.request.urlopen(request, timeout=max(timeout_seconds, 3)) as response:
        payload = json.loads(response.read().decode("utf-8", errors="replace"))
    return payload if isinstance(payload, dict) else {}


def _fetch_nasdaq_forecast(
    endpoint_template: str,
    *,
    symbol: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    url = endpoint_template.format(symbol=urllib.parse.quote(symbol.upper(), safe=""))
    proc = subprocess.run(
        [
            "/usr/bin/curl",
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--compressed",
            "--connect-timeout",
            "5",
            "--max-time",
            str(max(int(timeout_seconds), 3)),
            "--retry",
            "1",
            "--retry-delay",
            "1",
            "--user-agent",
            "Mozilla/5.0",
            "--header",
            "Accept: application/json, text/plain, */*",
            "--header",
            f"Referer: https://www.nasdaq.com/market-activity/stocks/{symbol.lower()}/earnings",
            url,
        ],
        capture_output=True,
        text=True,
        timeout=max(int(timeout_seconds), 3) * 2 + 5,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or f"curl_exit_{proc.returncode}").strip())
    payload = json.loads(proc.stdout or "{}")
    return payload if isinstance(payload, dict) else {}


def _write_cache_snapshot(path: Path, rows: Mapping[str, Mapping[str, Any]], now: datetime) -> None:
    _atomic_write_json(
        path,
        {
            "timestamp_utc": now.isoformat(),
            "schema_version": 1,
            "provider": "nasdaq_with_alpha_vantage_fallback",
            "symbols": {symbol: dict(row) for symbol, row in sorted(rows.items())},
        },
    )


def _write_history(payload: Mapping[str, Any], history_root: Path) -> Path:
    timestamp = _parse_timestamp(payload.get("timestamp_utc")) or datetime.now(timezone.utc)
    path = history_root / f"analyst_consensus_{timestamp:%Y%m%d}.jsonl.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "at", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n")
    return path


def collect_analyst_consensus_context(
    project_root: Path = PROJECT_ROOT,
    *,
    allow_network: bool = True,
    symbols_override: str = "",
    maximum_symbols: int | None = None,
    timeout_seconds: int = 20,
    maximum_runtime_seconds: int = 180,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    started_monotonic = time.monotonic()
    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    config = _read_json(project_root / "config" / CONFIG_PATH.name)
    universe = _eligible_symbols(project_root, config, symbols_override)
    cache_path = project_root / "data" / "external_context" / CACHE_PATH.name
    cache = _read_json(cache_path)
    max_age_days = float(config.get("cache_max_age_days", 35) or 35)
    cache_rows = _fresh_cache_rows(cache, now=now, maximum_age_days=max_age_days)
    nasdaq_config = config.get("nasdaq_provider") if isinstance(config.get("nasdaq_provider"), Mapping) else {}
    nasdaq_default = "1" if nasdaq_config.get("enabled_by_default", True) else "0"
    nasdaq_enabled = os.getenv("ANALYST_CONSENSUS_NASDAQ_ENABLED", nasdaq_default).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    alpha_enabled = os.getenv("ANALYST_CONSENSUS_ALPHA_VANTAGE_ENABLED", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
    alpha_configured = bool(api_key)
    enabled = bool(nasdaq_enabled or alpha_enabled)
    configured = bool(nasdaq_enabled or (alpha_enabled and alpha_configured))
    attempted: list[str] = []
    errors: dict[str, str] = {}
    provider_states: dict[str, Any] = {}
    alpha_maximum_requests = int(config.get("maximum_requests_per_day", 25) or 25)
    alpha_quota = _daily_quota_status(
        project_root,
        now=now,
        maximum_requests=alpha_maximum_requests,
        provider_id="alpha_vantage",
    )
    nasdaq_maximum_requests = int(nasdaq_config.get("maximum_requests_per_day", 40) or 40)
    nasdaq_quota = _daily_quota_status(
        project_root,
        now=now,
        maximum_requests=nasdaq_maximum_requests,
        provider_id="nasdaq",
    )

    if allow_network and nasdaq_enabled:
        limit = min(
            int(maximum_symbols or nasdaq_config.get("maximum_symbols_per_run", 16) or 16),
            int(nasdaq_config.get("maximum_symbols_per_run", 16) or 16),
            nasdaq_quota["requests_remaining"],
        )
        ordered = sorted(
            universe,
            key=lambda symbol: (
                _parse_timestamp(cache_rows.get(symbol, {}).get("collected_at_utc"))
                or datetime.min.replace(tzinfo=timezone.utc),
                symbol,
            ),
        )[:limit]
        interval = max(float(nasdaq_config.get("request_interval_seconds", 1.5) or 1.5), 0.0)
        nasdaq_attempted: list[str] = []
        nasdaq_errors: dict[str, str] = {}
        for index, symbol in enumerate(ordered):
            if time.monotonic() - started_monotonic >= max(int(maximum_runtime_seconds), 1):
                nasdaq_errors[symbol] = "run_runtime_deadline_reached"
                break
            claimed, nasdaq_quota = _claim_daily_request(
                project_root,
                now=now,
                maximum_requests=nasdaq_maximum_requests,
                provider_id="nasdaq",
            )
            if not claimed:
                break
            attempted.append(symbol)
            nasdaq_attempted.append(symbol)
            try:
                raw = _fetch_nasdaq_forecast(
                    str(
                        nasdaq_config.get("endpoint_template")
                        or "https://api.nasdaq.com/api/analyst/{symbol}/earnings-forecast"
                    ),
                    symbol=symbol,
                    timeout_seconds=timeout_seconds,
                )
                parsed = parse_nasdaq_earnings_forecast(raw, symbol=symbol, collected_at_utc=now)
                if parsed:
                    cache_rows[symbol] = parsed
                else:
                    nasdaq_errors[symbol] = "provider_payload_unusable_or_rate_limited"
            except Exception as exc:
                nasdaq_errors[symbol] = f"{type(exc).__name__}:{exc}"
            _write_cache_snapshot(cache_path, cache_rows, now)
            if interval > 0.0 and index + 1 < len(ordered):
                time.sleep(interval)
        errors.update({f"nasdaq:{symbol}": error for symbol, error in nasdaq_errors.items()})
        provider_states["nasdaq"] = {
            "enabled": True,
            "credential_required": False,
            "configured": True,
            "attempted_symbols": nasdaq_attempted,
            "error_count": len(nasdaq_errors),
            "errors": nasdaq_errors,
            "daily_quota": nasdaq_quota,
            "usage_scope": nasdaq_config.get("usage_scope"),
            "commercial_or_live_entitlement_required": bool(
                nasdaq_config.get("commercial_or_live_entitlement_required", True)
            ),
            "commercial_or_live_entitlement_verified": False,
        }
    else:
        provider_states["nasdaq"] = {
            "enabled": nasdaq_enabled,
            "credential_required": False,
            "configured": True,
            "attempted_symbols": [],
            "error_count": 0,
            "errors": {},
            "daily_quota": nasdaq_quota,
            "usage_scope": nasdaq_config.get("usage_scope"),
            "commercial_or_live_entitlement_required": bool(
                nasdaq_config.get("commercial_or_live_entitlement_required", True)
            ),
            "commercial_or_live_entitlement_verified": False,
        }

    if allow_network and alpha_enabled and alpha_configured:
        missing_after_nasdaq = [symbol for symbol in universe if symbol not in cache_rows]
        alpha_limit = min(
            int(maximum_symbols or config.get("maximum_symbols_per_run", 20) or 20),
            alpha_quota["requests_remaining"],
            len(missing_after_nasdaq),
        )
        ordered = sorted(missing_after_nasdaq)[:alpha_limit]
        interval = max(float(config.get("request_interval_seconds", 12) or 12), 0.0)
        alpha_attempted: list[str] = []
        alpha_errors: dict[str, str] = {}
        for index, symbol in enumerate(ordered):
            if time.monotonic() - started_monotonic >= max(int(maximum_runtime_seconds), 1):
                alpha_errors[symbol] = "run_runtime_deadline_reached"
                break
            claimed, alpha_quota = _claim_daily_request(
                project_root,
                now=now,
                maximum_requests=alpha_maximum_requests,
                provider_id="alpha_vantage",
            )
            if not claimed:
                break
            attempted.append(symbol)
            alpha_attempted.append(symbol)
            try:
                raw = _fetch_alpha_vantage(
                    str(config.get("endpoint") or "https://www.alphavantage.co/query"),
                    symbol=symbol,
                    api_key=api_key,
                    timeout_seconds=timeout_seconds,
                )
                parsed = parse_alpha_vantage_estimates(raw, symbol=symbol, collected_at_utc=now)
                if parsed:
                    cache_rows[symbol] = parsed
                else:
                    alpha_errors[symbol] = "provider_payload_unusable_or_rate_limited"
            except Exception as exc:
                alpha_errors[symbol] = f"{type(exc).__name__}:{exc}"
            _write_cache_snapshot(cache_path, cache_rows, now)
            if interval > 0.0 and index + 1 < len(ordered):
                time.sleep(interval)
        errors.update({f"alpha_vantage:{symbol}": error for symbol, error in alpha_errors.items()})
        provider_states["alpha_vantage"] = {
            "enabled": True,
            "credential_required": True,
            "configured": True,
            "attempted_symbols": alpha_attempted,
            "error_count": len(alpha_errors),
            "errors": alpha_errors,
            "daily_quota": alpha_quota,
        }
    else:
        provider_states["alpha_vantage"] = {
            "enabled": alpha_enabled,
            "credential_required": True,
            "configured": alpha_configured,
            "attempted_symbols": [],
            "error_count": 0,
            "errors": {},
            "daily_quota": alpha_quota,
        }

    primary_quota = nasdaq_quota if nasdaq_enabled else alpha_quota
    payload = build_analyst_consensus_payload(
        config=config,
        universe_symbols=universe,
        cache_rows=cache_rows,
        now_utc=now,
        provider_enabled=enabled,
        provider_configured=configured,
        attempted_symbols=attempted,
        errors=errors,
        quota_state=primary_quota,
        provider_states=provider_states,
    )
    _write_cache_snapshot(cache_path, cache_rows, now)
    _atomic_write_json(project_root / "exports" / "external_context" / LATEST_PATH.name, payload)
    _atomic_write_json(project_root / "governance" / "health" / HEALTH_PATH.name, payload)
    _write_history(payload, project_root / "data" / "external_context" / HISTORY_ROOT.name)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect governed point-in-time analyst consensus estimates.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--symbols", default=os.getenv("ANALYST_CONSENSUS_SYMBOLS", ""))
    parser.add_argument("--max-symbols", type=int)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--max-runtime-seconds", type=int, default=180)
    parser.add_argument("--no-network", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = collect_analyst_consensus_context(
        Path(args.project_root).expanduser().resolve(),
        allow_network=not args.no_network,
        symbols_override=args.symbols,
        maximum_symbols=args.max_symbols,
        timeout_seconds=max(int(args.timeout), 3),
        maximum_runtime_seconds=max(int(args.max_runtime_seconds), 1),
    )
    summary = {
        "timestamp_utc": payload.get("timestamp_utc"),
        "ok": payload.get("ok", False),
        "overall_status": payload.get("overall_status"),
        "direct_evidence_ready": payload.get("direct_evidence_ready", False),
        "coverage": payload.get("coverage", {}),
        "provider_runtime": payload.get("provider_runtime", {}),
        "consumer_contract": payload.get("consumer_contract", {}),
        "safety_contract": payload.get("authority_contract", {}),
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    else:
        coverage = payload.get("coverage") or {}
        print(
            "analyst_consensus_context "
            f"status={payload.get('overall_status')} "
            f"covered={coverage.get('covered_symbol_count', 0)}/{coverage.get('universe_symbol_count', 0)} "
            f"revision_history={coverage.get('revision_history_ratio', 0.0)}"
        )
    configured_failure = bool(
        payload.get("provider_runtime", {}).get("enabled")
        and not payload.get("direct_evidence_ready")
    )
    return 2 if configured_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
