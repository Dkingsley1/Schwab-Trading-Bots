#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import os
import re
import sys
import tempfile
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.decision_context_mesh import (
    DECISION_CONTEXT_MESH_FEATURE_KEYS,
    PLANE_SIGNAL_FEATURE_KEYS,
    assess_decision_context_mesh,
    load_decision_context_mesh_config,
    percentage_grade,
)
from scripts.portfolio_capacity_curve_report import build_payload as build_capacity_payload


LATEST_PATH = PROJECT_ROOT / "exports" / "external_context" / "decision_context_mesh_latest.json"
HEALTH_PATH = PROJECT_ROOT / "governance" / "health" / "decision_context_mesh_latest.json"
HISTORY_ROOT = PROJECT_ROOT / "data" / "external_context" / "decision_context_mesh_history"
TREASURY_TIC_URL = "https://ticdata.treasury.gov/resource-center/data-chart-center/tic/Documents/npr_history.txt"
EIA_WPSR_URL = "https://ir.eia.gov/wpsr/table4.csv"
BTS_FREIGHT_URL = "https://data.bts.gov/resource/bw6n-ddqk.json?$limit=6&$order=obs_date%20DESC"

LOCAL_SOURCE_SPECS = {
    "official_macro_context": (
        "exports/external_context/official_macro_context_latest.json",
        8.0,
        24.0,
        "Federal Reserve, U.S. Treasury, BLS, and BEA",
        "https://www.federalreserve.gov/data.htm",
        "official_macro",
    ),
    "public_policy_context": (
        "exports/external_context/public_policy_context_latest.json",
        24.0,
        48.0,
        "U.S. Treasury FiscalData and World Bank",
        "https://fiscaldata.treasury.gov/",
        "official_public_policy",
    ),
    "market_micro_context": (
        "exports/external_context/market_micro_latest.json",
        24.0,
        48.0,
        "FINRA, Nasdaq, U.S. Treasury, and internal market observations",
        "https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data",
        "official_market_micro",
    ),
    "extended_quant_context": (
        "exports/external_context/extended_quant_context_latest.json",
        24.0,
        48.0,
        "CFTC, Federal Reserve Bank of New York, Cboe, Nasdaq, and SEC",
        "https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm",
        "official_quant_context",
    ),
    "options_flow_context": (
        "exports/external_context/options_flow_context_latest.json",
        8.0,
        24.0,
        "Options provider mesh",
        "https://www.cboe.com/delayed_quotes/",
        "options_market_context",
    ),
    "sec_edgar_context": (
        "exports/external_context/sec_edgar_latest.json",
        24.0,
        48.0,
        "U.S. Securities and Exchange Commission",
        "https://www.sec.gov/edgar/search/",
        "official_filings",
    ),
    "schwab_symbol_news": (
        "exports/external_context/schwab_symbol_news_latest.json",
        8.0,
        24.0,
        "Charles Schwab symbol news context",
        "https://www.schwab.com/",
        "broker_news",
    ),
    "bond_reference_context": (
        "exports/external_context/bond_reference_latest.json",
        24.0,
        48.0,
        "Federal Reserve Economic Data and U.S. Treasury",
        "https://fred.stlouisfed.org/",
        "official_rates_credit",
    ),
    "fx_market_context": (
        "exports/external_context/fx_market_context_latest.json",
        8.0,
        24.0,
        "Official and market FX context mesh",
        "https://www.federalreserve.gov/releases/h10/current/",
        "official_fx",
    ),
    "central_bank_cross_source_context": (
        "exports/external_context/central_bank_cross_source_latest.json",
        12.0,
        24.0,
        "BIS member central banks and synchronized official context",
        "https://data.bis.org/topics/CBPOL",
        "official_central_banks",
    ),
    "portfolio_capacity_curves": (
        "governance/allocator/portfolio_capacity_curve_latest.json",
        4.0,
        24.0,
        "Internal paper execution evidence",
        "",
        "internal_capacity",
    ),
    "paper_execution_calibration": (
        "governance/health/paper_execution_calibration_latest.json",
        4.0,
        24.0,
        "Independent paper execution calibration",
        "",
        "internal_execution_calibration",
    ),
}

LOCAL_SOURCE_FALLBACK_PATHS = {
    "schwab_symbol_news": ("governance/health/schwab_symbol_news_latest.json",),
}

ANALYST_CONSENSUS_PATHS = (
    "exports/external_context/analyst_consensus_latest.json",
    "governance/health/analyst_consensus_latest.json",
)

DIRECT_SOURCE_SPECS = {
    "treasury_tic": {
        "publisher": "U.S. Treasury International Capital System",
        "url": TREASURY_TIC_URL,
        "target_observation_age_days": 90.0,
        "maximum_observation_age_days": 120.0,
        "family": "official_cross_border",
    },
    "eia_weekly_petroleum": {
        "publisher": "U.S. Energy Information Administration",
        "url": EIA_WPSR_URL,
        "target_observation_age_days": 10.0,
        "maximum_observation_age_days": 21.0,
        "family": "official_energy_inventory",
    },
    "bts_freight_tsi": {
        "publisher": "U.S. Bureau of Transportation Statistics",
        "url": BTS_FREIGHT_URL,
        "target_observation_age_days": 90.0,
        "maximum_observation_age_days": 120.0,
        "family": "official_transportation",
    },
}


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _safe_float(value: Any) -> float | None:
    try:
        result = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _cadence_freshness(age: float | None, *, target_age: float, maximum_age: float) -> float:
    if age is None or age < -0.1 or age > maximum_age:
        return 0.0
    target = max(min(float(target_age), float(maximum_age)), 0.0)
    if age <= target or maximum_age <= target:
        return 1.0
    return _clamp01((maximum_age - age) / max(maximum_age - target, 1e-9))


def _signed_norm(value: float, scale: float) -> float:
    return _clamp01(0.5 + float(value) / (2.0 * max(abs(float(scale)), 1e-9)))


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


def _parse_observation_time(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    parsed = _parse_timestamp(text)
    if parsed is not None:
        return parsed
    for fmt in ("%Y-%b", "%m/%d/%y", "%Y-%m", "%Y-%m-%d"):
        try:
            candidate = datetime.strptime(text[:10] if fmt == "%Y-%m-%d" else text, fmt)
        except ValueError:
            continue
        if fmt == "%Y-%b":
            if candidate.month == 12:
                next_month = datetime(candidate.year + 1, 1, 1)
            else:
                next_month = datetime(candidate.year, candidate.month + 1, 1)
            candidate = next_month - timedelta(days=1)
        return candidate.replace(tzinfo=timezone.utc)
    return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _routed_source_path(
    project_root: Path,
    relative: str,
    *,
    extra_relatives: Iterable[str] = (),
) -> tuple[Path, str, int]:
    external_root = Path(
        os.getenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", "/Volumes/BOT_LOGS/schwab_trading_bot")
    ).expanduser()
    roots = (
        ("repo", project_root),
        ("local_fallback", project_root / "local_fallback_storage"),
        ("external", external_root),
    )
    relatives = tuple(dict.fromkeys((relative, *extra_relatives)))
    candidates: list[tuple[Path, str, datetime, bool]] = []
    seen: set[str] = set()
    for route, root in roots:
        for candidate_relative in relatives:
            canonical = root / candidate_relative
            route_candidates: list[tuple[Path, bool]] = [(canonical, False)]
            if canonical.parent.exists():
                route_candidates.extend(
                    (candidate, True)
                    for candidate in canonical.parent.glob(f"{canonical.name}.local_fallback*")
                )
            for candidate, is_fallback in route_candidates:
                key = str(candidate)
                if key in seen or not candidate.is_file():
                    continue
                seen.add(key)
                payload = _read_json(candidate)
                if not payload:
                    continue
                timestamp = _parse_timestamp(payload.get("timestamp_utc"))
                if timestamp is None:
                    timestamp = datetime.fromtimestamp(candidate.stat().st_mtime, tz=timezone.utc)
                candidates.append((candidate, route, timestamp, is_fallback))
    if not candidates:
        return project_root / relative, "missing", 0
    selection_now = datetime.now(timezone.utc)
    future_tolerance = timedelta(hours=0.1)
    current_candidates = [
        row for row in candidates if row[2] <= selection_now + future_tolerance
    ]
    selectable_candidates = current_candidates or candidates
    # Conflict-preserved files are valid writes, not archival debris. Select the
    # newest observation across every storage route; prefer canonical only when
    # two candidates carry the same timestamp. A future-dated candidate cannot
    # shadow a valid current observation; if every candidate is future-dated,
    # downstream point-in-time validation still fails closed.
    path, route, _, _ = max(selectable_candidates, key=lambda row: (row[2], not row[3]))
    return path, route, len(candidates)


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
        tmp_path = Path(handle.name)
    os.replace(tmp_path, path)


def _path_value(payload: Any, dotted_path: str) -> Any:
    current = payload
    for token in str(dotted_path or "").split("."):
        if not token or not isinstance(current, Mapping) or token not in current:
            return None
        current = current[token]
    return current


def _global_features(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    derived = payload.get("derived") if isinstance(payload.get("derived"), Mapping) else {}
    features = derived.get("global_features") if isinstance(derived.get("global_features"), Mapping) else {}
    if features:
        return features
    return payload.get("features") if isinstance(payload.get("features"), Mapping) else {}


def _symbol_features(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    derived = payload.get("derived") if isinstance(payload.get("derived"), Mapping) else {}
    features = derived.get("symbol_features") if isinstance(derived.get("symbol_features"), Mapping) else {}
    return features


def _payload_declares_ok(payload: Mapping[str, Any]) -> bool:
    if not payload:
        return False
    if payload.get("ok") is False:
        return False
    status = payload.get("status")
    if isinstance(status, str) and status.strip().lower() in {"failed", "error", "blocked"}:
        return False
    if isinstance(status, Mapping) and status.get("ok") is False:
        return False
    overall = str(payload.get("overall_status") or "").strip().lower()
    if overall in {"failed", "error", "blocked"}:
        return False
    return True


def _local_source_state(
    source_id: str,
    project_root: Path,
    now: datetime,
) -> tuple[dict[str, Any], dict[str, Any]]:
    relative, target_age_hours, maximum_age_hours, publisher, url, family = LOCAL_SOURCE_SPECS[source_id]
    path, storage_route, route_candidate_count = _routed_source_path(
        project_root,
        relative,
        extra_relatives=LOCAL_SOURCE_FALLBACK_PATHS.get(source_id, ()),
    )
    payload = _read_json(path)
    timestamp = _parse_timestamp(payload.get("timestamp_utc"))
    if timestamp is None and path.exists():
        timestamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    age_hours = (now - timestamp).total_seconds() / 3600.0 if timestamp is not None else None
    future = bool(age_hours is not None and age_hours < -0.1)
    fresh = bool(age_hours is not None and not future and age_hours <= maximum_age_hours)
    freshness = _cadence_freshness(
        age_hours,
        target_age=target_age_hours,
        maximum_age=maximum_age_hours,
    )
    recency = _clamp01(1.0 - max(float(age_hours or 0.0), 0.0) / maximum_age_hours) if fresh else 0.0
    state = {
        "source_id": source_id,
        "publisher": publisher,
        "url": url,
        "source_family": family,
        "artifact_path": str(path),
        "storage_route": storage_route,
        "storage_route_candidate_count": route_candidate_count,
        "storage_conflict_preserved_candidate_selected": ".local_fallback" in path.name,
        "storage_selection_policy": "newest_valid_observation_across_routes_canonical_tiebreak",
        "timestamp_utc": timestamp.isoformat() if timestamp is not None else None,
        "observation_time": timestamp.isoformat() if timestamp is not None else None,
        "age_hours": round(age_hours, 6) if age_hours is not None else None,
        "target_age_hours": target_age_hours,
        "maximum_age_hours": maximum_age_hours,
        "fresh": fresh,
        "freshness_norm": round(freshness, 6),
        "recency_norm": round(recency, 6),
        "freshness_policy": "full_credit_inside_expected_refresh_cadence_then_linear_decay_to_hard_slo",
        "timestamp_in_future": future,
        "ok": bool(fresh and _payload_declares_ok(payload)),
        "fallback": False,
    }
    return payload, state


def _analyst_consensus_source_state(
    project_root: Path,
    now: datetime,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path, storage_route, route_candidate_count = _routed_source_path(
        project_root,
        ANALYST_CONSENSUS_PATHS[0],
        extra_relatives=ANALYST_CONSENSUS_PATHS[1:],
    )
    if not path.exists():
        return {}, {}
    payload = _read_json(path)
    timestamp = _parse_timestamp(payload.get("timestamp_utc"))
    age_hours = (now - timestamp).total_seconds() / 3600.0 if timestamp is not None else None
    future = bool(age_hours is not None and age_hours < -0.1)
    target_age_hours = 36.0
    maximum_age_hours = 168.0
    fresh = bool(age_hours is not None and not future and age_hours <= maximum_age_hours)
    consumer = payload.get("consumer_contract") if isinstance(payload.get("consumer_contract"), Mapping) else {}
    direct_ready = bool(payload.get("direct_evidence_ready") is True and consumer.get("ready") is True)
    if not direct_ready:
        return {}, {}
    state = {
        "source_id": "analyst_consensus_context",
        "publisher": payload.get("provider", "configured analyst consensus provider"),
        "url": payload.get("provider_documentation_url", ""),
        "source_family": "configured_analyst_consensus_provider",
        "artifact_path": str(path),
        "storage_route": storage_route,
        "storage_route_candidate_count": route_candidate_count,
        "storage_conflict_preserved_candidate_selected": ".local_fallback" in path.name,
        "storage_selection_policy": "newest_valid_observation_across_routes_canonical_tiebreak",
        "timestamp_utc": timestamp.isoformat() if timestamp is not None else None,
        "observation_time": timestamp.isoformat() if timestamp is not None else None,
        "age_hours": round(age_hours, 6) if age_hours is not None else None,
        "target_age_hours": target_age_hours,
        "maximum_age_hours": maximum_age_hours,
        "fresh": fresh,
        "freshness_norm": round(
            _cadence_freshness(age_hours, target_age=target_age_hours, maximum_age=maximum_age_hours),
            6,
        ),
        "timestamp_in_future": future,
        "ok": bool(fresh and direct_ready),
        "fallback": False,
        "direct_consensus_ready": direct_ready,
    }
    return payload, state


def _http_text(url: str, *, timeout_seconds: int) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "schwab-trading-bot-context-mesh/1.0 research-contact=local-operator",
            "Accept": "application/json,text/csv,text/plain,*/*",
        },
    )
    with urllib.request.urlopen(request, timeout=max(int(timeout_seconds), 3)) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_treasury_tic_history(text: str, *, as_of: datetime) -> dict[str, Any]:
    candidates: list[tuple[datetime, list[str]]] = []
    for raw_row in csv.reader(io.StringIO(text), delimiter="\t"):
        row = [str(value or "").strip() for value in raw_row]
        if not row or not re.fullmatch(r"\d{4}-[A-Za-z]{3}", row[0]):
            continue
        observation = _parse_observation_time(row[0])
        if observation is None or observation > as_of or len(row) < 33:
            continue
        candidates.append((observation, row))
    if not candidates:
        return {}
    observation, row = max(candidates, key=lambda item: item[0])
    total = _safe_float(row[30])
    private = _safe_float(row[31])
    official = _safe_float(row[32])
    if None in {total, private, official}:
        return {}
    return {
        "observation_period": row[0],
        "observation_time": observation.isoformat(),
        "total_monthly_inflows_usd_millions": total,
        "private_monthly_inflows_usd_millions": private,
        "official_monthly_inflows_usd_millions": official,
    }


def parse_eia_weekly_petroleum(text: str, *, as_of: datetime) -> dict[str, Any]:
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or len(reader.fieldnames) < 4:
        return {}
    current_column = reader.fieldnames[1]
    difference_column = next((name for name in reader.fieldnames if str(name).lower() == "difference"), "Difference")
    observation = _parse_observation_time(current_column)
    if observation is None or observation > as_of:
        return {}
    wanted = {
        "Commercial (Excluding SPR)": "commercial_crude",
        "SPR": "strategic_petroleum_reserve",
        "Total Motor Gasoline": "motor_gasoline",
        "Distillate Fuel Oil": "distillate_fuel",
    }
    rows: dict[str, Any] = {}
    for row in reader:
        label = str(row.get("STUB_1") or "").strip()
        key = wanted.get(label)
        if not key:
            continue
        current = _safe_float(row.get(current_column))
        difference = _safe_float(row.get(difference_column))
        if current is None or difference is None:
            continue
        rows[key] = {
            "current_million_barrels": current,
            "weekly_change_million_barrels": difference,
        }
    if not rows:
        return {}
    return {
        "observation_date": observation.date().isoformat(),
        "observation_time": observation.isoformat(),
        "inventories": rows,
    }


def parse_bts_freight_tsi(text: str, *, as_of: datetime) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except Exception:
        return {}
    if not isinstance(payload, list):
        return {}
    candidates: list[tuple[datetime, Mapping[str, Any]]] = []
    for row in payload:
        if not isinstance(row, Mapping):
            continue
        observation = _parse_observation_time(row.get("obs_date"))
        if observation is not None and observation <= as_of:
            candidates.append((observation, row))
    if not candidates:
        return {}
    observation, row = max(candidates, key=lambda item: item[0])
    fields = {
        "tsi_freight": _safe_float(row.get("tsi_freight")),
        "tsi_freight_change_pct": _safe_float(row.get("tsi_freight_c")),
        "truck_index": _safe_float(row.get("truck_d11")),
        "rail_carloads_index": _safe_float(row.get("idx_rail_frt_carloads")),
        "rail_intermodal_index": _safe_float(row.get("idx_rail_frt_intermodal")),
    }
    clean = {key: value for key, value in fields.items() if value is not None}
    if "tsi_freight" not in clean:
        return {}
    return {
        "observation_date": observation.date().isoformat(),
        "observation_time": observation.isoformat(),
        **clean,
    }


def _direct_source_state(
    source_id: str,
    data: Mapping[str, Any],
    *,
    now: datetime,
    fallback: bool = False,
    error: str = "",
) -> dict[str, Any]:
    spec = DIRECT_SOURCE_SPECS[source_id]
    observation = _parse_observation_time(data.get("observation_time"))
    age_days = (now - observation).total_seconds() / 86400.0 if observation is not None else None
    target_age = float(spec["target_observation_age_days"])
    maximum_age = float(spec["maximum_observation_age_days"])
    future = bool(age_days is not None and age_days < -0.1)
    fresh = bool(age_days is not None and not future and age_days <= maximum_age)
    freshness = _cadence_freshness(age_days, target_age=target_age, maximum_age=maximum_age)
    recency = _clamp01(1.0 - max(float(age_days or 0.0), 0.0) / maximum_age) if fresh else 0.0
    return {
        "source_id": source_id,
        "publisher": spec["publisher"],
        "url": spec["url"],
        "source_family": spec["family"],
        "timestamp_utc": now.isoformat(),
        "observation_time": observation.isoformat() if observation is not None else None,
        "age_days": round(age_days, 6) if age_days is not None else None,
        "target_observation_age_days": target_age,
        "maximum_observation_age_days": maximum_age,
        "fresh": fresh,
        "freshness_norm": round(freshness, 6),
        "recency_norm": round(recency, 6),
        "freshness_policy": "full_credit_inside_expected_publication_cadence_then_linear_decay_to_hard_slo",
        "timestamp_in_future": future,
        "ok": bool(data and fresh),
        "fallback": bool(fallback),
        "error": str(error or ""),
        "data": dict(data),
    }


def _collect_direct_sources(
    *,
    now: datetime,
    timeout_seconds: int,
    allow_network: bool,
    previous: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    previous_sources = previous.get("sources") if isinstance(previous.get("sources"), Mapping) else {}
    parsers = {
        "treasury_tic": (TREASURY_TIC_URL, parse_treasury_tic_history),
        "eia_weekly_petroleum": (EIA_WPSR_URL, parse_eia_weekly_petroleum),
        "bts_freight_tsi": (BTS_FREIGHT_URL, parse_bts_freight_tsi),
    }
    out: dict[str, dict[str, Any]] = {}
    for source_id, (url, parser) in parsers.items():
        data: dict[str, Any] = {}
        error = "network_disabled"
        if allow_network:
            try:
                data = parser(_http_text(url, timeout_seconds=timeout_seconds), as_of=now)
                error = "" if data else "official_payload_unusable"
            except Exception as exc:
                error = f"{type(exc).__name__}:{exc}"
        if data:
            out[source_id] = _direct_source_state(source_id, data, now=now)
            continue
        previous_row = previous_sources.get(source_id) if isinstance(previous_sources.get(source_id), Mapping) else {}
        previous_data = previous_row.get("data") if isinstance(previous_row.get("data"), Mapping) else {}
        fallback_state = _direct_source_state(source_id, previous_data, now=now, fallback=True, error=error)
        out[source_id] = fallback_state
    return out


def _add_feature(
    target: dict[str, dict[str, Any]],
    key: str,
    value: Any,
    *,
    source_id: str,
    field_path: str,
    source_states: Mapping[str, Mapping[str, Any]],
    now: datetime,
    observation_time: Any = None,
    confidence: float = 0.9,
) -> None:
    number = _safe_float(value)
    if number is None:
        return
    state = source_states.get(source_id) if isinstance(source_states.get(source_id), Mapping) else {}
    if state.get("ok") is not True:
        return
    observation = _parse_observation_time(observation_time or state.get("observation_time") or state.get("timestamp_utc"))
    if observation is None or observation > now:
        return
    target[key] = {
        "value": _clamp01(number),
        "lineage": [
            {
                "source_id": source_id,
                "publisher": state.get("publisher"),
                "url": state.get("url"),
                "field_path": field_path,
                "observation_time": observation.isoformat(),
                "artifact_timestamp_utc": state.get("timestamp_utc"),
                "source_confidence_norm": _clamp01(confidence),
                "fallback": bool(state.get("fallback", False)),
                "point_in_time_valid": True,
            }
        ],
    }


def _mean_feature(target: dict[str, dict[str, Any]], key: str, source_keys: Iterable[str]) -> None:
    rows = [target[item] for item in source_keys if item in target]
    if not rows:
        return
    target[key] = {
        "value": _clamp01(sum(float(row["value"]) for row in rows) / len(rows)),
        "lineage": [entry for row in rows for entry in row.get("lineage", [])],
    }


def _feature_candidates(
    payloads: Mapping[str, Mapping[str, Any]],
    source_states: Mapping[str, Mapping[str, Any]],
    *,
    now: datetime,
) -> dict[str, dict[str, dict[str, Any]]]:
    official = _global_features(payloads.get("official_macro_context", {}))
    public = _global_features(payloads.get("public_policy_context", {}))
    micro = _global_features(payloads.get("market_micro_context", {}))
    extended_payload = payloads.get("extended_quant_context", {})
    extended = _global_features(extended_payload)
    extended_calendar = _path_value(extended_payload, "derived.calendar_features") or {}
    options = _global_features(payloads.get("options_flow_context", {}))
    sec = _global_features(payloads.get("sec_edgar_context", {}))
    bond = payloads.get("bond_reference_context", {})
    fx = _global_features(payloads.get("fx_market_context", {}))
    central = _global_features(payloads.get("central_bank_cross_source_context", {}))
    consensus = _global_features(payloads.get("analyst_consensus_context", {}))
    capacity = payloads.get("portfolio_capacity_curves", {})
    calibration = payloads.get("paper_execution_calibration", {})
    tic = _path_value(source_states, "treasury_tic.data") or {}
    eia = _path_value(source_states, "eia_weekly_petroleum.data") or {}
    bts = _path_value(source_states, "bts_freight_tsi.data") or {}

    planes: dict[str, dict[str, dict[str, Any]]] = {plane_id: {} for plane_id in (
        "fiscal_liquidity",
        "funding_stress",
        "cross_border_capital",
        "positioning_crowding",
        "securities_lending",
        "credit_curve",
        "volatility_surface",
        "passive_mechanical_flows",
        "market_calendar",
        "supply_chain_inventory",
        "estimates_dispersion",
        "capacity_market_impact",
    )}

    fiscal = planes["fiscal_liquidity"]
    _add_feature(fiscal, "fiscal_fed_net_liquidity_norm", official.get("fed_net_liquidity_impulse_norm"), source_id="official_macro_context", field_path="derived.global_features.fed_net_liquidity_impulse_norm", source_states=source_states, now=now, confidence=0.99)
    debt_change = _safe_float(public.get("us_public_debt_daily_change_usd"))
    _add_feature(fiscal, "fiscal_debt_issuance_impulse_norm", _signed_norm(debt_change, 100_000_000_000.0) if debt_change is not None else None, source_id="public_policy_context", field_path="features.us_public_debt_daily_change_usd", source_states=source_states, now=now, confidence=0.98)
    _add_feature(fiscal, "fiscal_tga_drain_impulse_norm", official.get("fed_tga_drain_impulse_norm"), source_id="official_macro_context", field_path="derived.global_features.fed_tga_drain_impulse_norm", source_states=source_states, now=now, confidence=0.99)

    funding = planes["funding_stress"]
    _add_feature(funding, "funding_fed_stress_norm", official.get("fed_funding_stress_norm"), source_id="official_macro_context", field_path="derived.global_features.fed_funding_stress_norm", source_states=source_states, now=now, confidence=0.99)
    _add_feature(funding, "funding_sofr_stress_norm", extended.get("sofr_funding_stress_norm"), source_id="extended_quant_context", field_path="derived.global_features.sofr_funding_stress_norm", source_states=source_states, now=now, confidence=0.98)
    _add_feature(funding, "funding_corridor_pressure_norm", official.get("fed_sofr_effr_spread_norm"), source_id="official_macro_context", field_path="derived.global_features.fed_sofr_effr_spread_norm", source_states=source_states, now=now, confidence=0.99)

    cross_border = planes["cross_border_capital"]
    tic_observation = tic.get("observation_time")
    total_inflows = _safe_float(tic.get("total_monthly_inflows_usd_millions"))
    private_inflows = _safe_float(tic.get("private_monthly_inflows_usd_millions"))
    official_inflows = _safe_float(tic.get("official_monthly_inflows_usd_millions"))
    _add_feature(cross_border, "cross_border_total_flow_norm", _signed_norm(total_inflows, 250_000.0) if total_inflows is not None else None, source_id="treasury_tic", field_path="data.total_monthly_inflows_usd_millions", source_states=source_states, now=now, observation_time=tic_observation, confidence=0.99)
    _add_feature(cross_border, "cross_border_private_flow_norm", _signed_norm(private_inflows, 250_000.0) if private_inflows is not None else None, source_id="treasury_tic", field_path="data.private_monthly_inflows_usd_millions", source_states=source_states, now=now, observation_time=tic_observation, confidence=0.99)
    _add_feature(cross_border, "cross_border_official_flow_norm", _signed_norm(official_inflows, 150_000.0) if official_inflows is not None else None, source_id="treasury_tic", field_path="data.official_monthly_inflows_usd_millions", source_states=source_states, now=now, observation_time=tic_observation, confidence=0.99)
    _add_feature(cross_border, "cross_border_fx_confirmation_norm", fx.get("fx_proxy_agreement_norm"), source_id="fx_market_context", field_path="derived.global_features.fx_proxy_agreement_norm", source_states=source_states, now=now, confidence=0.9)
    _add_feature(cross_border, "cross_border_external_balance_imbalance_norm", public.get("world_bank_current_account_imbalance_norm"), source_id="public_policy_context", field_path="features.world_bank_current_account_imbalance_norm", source_states=source_states, now=now, confidence=0.96)
    _add_feature(cross_border, "cross_border_policy_spillover_norm", central.get("central_bank_policy_spillover_risk_norm"), source_id="central_bank_cross_source_context", field_path="derived.global_features.central_bank_policy_spillover_risk_norm", source_states=source_states, now=now, confidence=0.95)

    positioning = planes["positioning_crowding"]
    _add_feature(positioning, "positioning_cot_crowding_norm", extended.get("cot_equity_crowding_norm"), source_id="extended_quant_context", field_path="derived.global_features.cot_equity_crowding_norm", source_states=source_states, now=now, confidence=0.98)
    _add_feature(positioning, "positioning_macro_stress_norm", extended.get("cot_macro_positioning_stress_norm"), source_id="extended_quant_context", field_path="derived.global_features.cot_macro_positioning_stress_norm", source_states=source_states, now=now, confidence=0.98)
    _add_feature(positioning, "positioning_put_call_stress_norm", extended.get("cboe_put_call_stress_norm"), source_id="extended_quant_context", field_path="derived.global_features.cboe_put_call_stress_norm", source_states=source_states, now=now, confidence=0.97)
    _add_feature(positioning, "positioning_etf_flow_pressure_norm", micro.get("etf_fund_family_flow_norm"), source_id="market_micro_context", field_path="derived.global_features.etf_fund_family_flow_norm", source_states=source_states, now=now, confidence=0.82)

    lending = planes["securities_lending"]
    borrow_availability = _safe_float(options.get("short_borrow_availability_norm"))
    _add_feature(lending, "lending_borrow_scarcity_norm", (1.0 - borrow_availability) if borrow_availability is not None else None, source_id="options_flow_context", field_path="derived.global_features.short_borrow_availability_norm", source_states=source_states, now=now, confidence=0.82)
    _add_feature(lending, "lending_borrow_fee_norm", options.get("short_borrow_fee_norm"), source_id="options_flow_context", field_path="derived.global_features.short_borrow_fee_norm", source_states=source_states, now=now, confidence=0.82)
    _add_feature(lending, "lending_utilization_norm", options.get("short_utilization_norm"), source_id="options_flow_context", field_path="derived.global_features.short_utilization_norm", source_states=source_states, now=now, confidence=0.82)
    _add_feature(lending, "lending_ftd_pressure_norm", extended.get("short_ftd_total_hits_norm"), source_id="extended_quant_context", field_path="derived.global_features.short_ftd_total_hits_norm", source_states=source_states, now=now, confidence=0.98)
    _add_feature(lending, "lending_short_volume_pressure_norm", micro.get("market_micro_short_pressure_norm"), source_id="market_micro_context", field_path="derived.global_features.market_micro_short_pressure_norm", source_states=source_states, now=now, confidence=0.9)

    credit = planes["credit_curve"]
    yields = bond.get("treasury_yields") if isinstance(bond.get("treasury_yields"), Mapping) else {}
    y2 = _safe_float(yields.get("2y"))
    y10 = _safe_float(yields.get("10y"))
    _add_feature(credit, "credit_curve_2s10s_norm", _signed_norm(y10 - y2, 2.0) if y2 is not None and y10 is not None else None, source_id="bond_reference_context", field_path="treasury_yields.2y|10y", source_states=source_states, now=now, confidence=0.98)
    spread = _safe_float(bond.get("credit_spread_bps"))
    _add_feature(credit, "credit_spread_pressure_norm", _clamp01(spread / 10.0) if spread is not None else None, source_id="bond_reference_context", field_path="credit_spread_bps", source_states=source_states, now=now, confidence=0.95)
    _add_feature(credit, "credit_flow_pressure_norm", micro.get("market_micro_credit_flow_norm"), source_id="market_micro_context", field_path="derived.global_features.market_micro_credit_flow_norm", source_states=source_states, now=now, confidence=0.84)

    volatility = planes["volatility_surface"]
    _add_feature(volatility, "volatility_surface_skew_norm", options.get("options_iv_skew_norm"), source_id="options_flow_context", field_path="derived.global_features.options_iv_skew_norm", source_states=source_states, now=now, confidence=0.9)
    _add_feature(volatility, "volatility_term_structure_norm", options.get("options_iv_term_structure_norm"), source_id="options_flow_context", field_path="derived.global_features.options_iv_term_structure_norm", source_states=source_states, now=now, confidence=0.9)
    _add_feature(volatility, "volatility_surface_change_norm", options.get("options_surface_change_norm"), source_id="options_flow_context", field_path="derived.global_features.options_surface_change_norm", source_states=source_states, now=now, confidence=0.88)
    _add_feature(volatility, "volatility_put_call_stress_norm", extended.get("cboe_put_call_stress_norm"), source_id="extended_quant_context", field_path="derived.global_features.cboe_put_call_stress_norm", source_states=source_states, now=now, confidence=0.97)

    passive = planes["passive_mechanical_flows"]
    _add_feature(passive, "passive_etf_creation_redemption_stress_norm", micro.get("etf_creation_redemption_stress_norm"), source_id="market_micro_context", field_path="derived.global_features.etf_creation_redemption_stress_norm", source_states=source_states, now=now, confidence=0.82)
    _add_feature(passive, "passive_etf_family_flow_norm", micro.get("etf_fund_family_flow_norm"), source_id="market_micro_context", field_path="derived.global_features.etf_fund_family_flow_norm", source_states=source_states, now=now, confidence=0.82)
    _add_feature(passive, "passive_index_rebalance_window_norm", extended_calendar.get("calendar_index_rebalance_window_norm"), source_id="extended_quant_context", field_path="derived.calendar_features.calendar_index_rebalance_window_norm", source_states=source_states, now=now, confidence=0.96)
    _add_feature(passive, "passive_expiry_pressure_norm", extended_calendar.get("calendar_opex_week_norm"), source_id="extended_quant_context", field_path="derived.calendar_features.calendar_opex_week_norm", source_states=source_states, now=now, confidence=0.96)

    calendar = planes["market_calendar"]
    official_calendar = _path_value(payloads.get("official_macro_context", {}), "derived.calendar_features") or {}
    _add_feature(calendar, "calendar_high_impact_event_norm", official_calendar.get("calendar_high_impact_24h_norm"), source_id="official_macro_context", field_path="derived.calendar_features.calendar_high_impact_24h_norm", source_states=source_states, now=now, confidence=0.99)
    _add_feature(calendar, "calendar_options_expiry_norm", extended_calendar.get("calendar_options_expiry_week_norm"), source_id="extended_quant_context", field_path="derived.calendar_features.calendar_options_expiry_week_norm", source_states=source_states, now=now, confidence=0.97)
    _add_feature(calendar, "calendar_futures_roll_norm", extended_calendar.get("calendar_futures_roll_window_norm"), source_id="extended_quant_context", field_path="derived.calendar_features.calendar_futures_roll_window_norm", source_states=source_states, now=now, confidence=0.97)
    _add_feature(calendar, "calendar_auction_pressure_norm", micro.get("market_micro_auction_print_pressure_norm"), source_id="market_micro_context", field_path="derived.global_features.market_micro_auction_print_pressure_norm", source_states=source_states, now=now, confidence=0.88)

    supply = planes["supply_chain_inventory"]
    eia_observation = eia.get("observation_time")
    crude_change = _safe_float(_path_value(eia, "inventories.commercial_crude.weekly_change_million_barrels"))
    gasoline_change = _safe_float(_path_value(eia, "inventories.motor_gasoline.weekly_change_million_barrels"))
    distillate_change = _safe_float(_path_value(eia, "inventories.distillate_fuel.weekly_change_million_barrels"))
    _add_feature(supply, "inventory_crude_change_norm", _signed_norm(crude_change, 20.0) if crude_change is not None else None, source_id="eia_weekly_petroleum", field_path="data.inventories.commercial_crude.weekly_change_million_barrels", source_states=source_states, now=now, observation_time=eia_observation, confidence=0.99)
    _add_feature(supply, "inventory_gasoline_change_norm", _signed_norm(gasoline_change, 10.0) if gasoline_change is not None else None, source_id="eia_weekly_petroleum", field_path="data.inventories.motor_gasoline.weekly_change_million_barrels", source_states=source_states, now=now, observation_time=eia_observation, confidence=0.99)
    _add_feature(supply, "inventory_distillate_change_norm", _signed_norm(distillate_change, 10.0) if distillate_change is not None else None, source_id="eia_weekly_petroleum", field_path="data.inventories.distillate_fuel.weekly_change_million_barrels", source_states=source_states, now=now, observation_time=eia_observation, confidence=0.99)
    bts_observation = bts.get("observation_time")
    freight_change = _safe_float(bts.get("tsi_freight_change_pct"))
    truck_index = _safe_float(bts.get("truck_index"))
    _add_feature(supply, "supply_freight_momentum_norm", _signed_norm(freight_change, 5.0) if freight_change is not None else None, source_id="bts_freight_tsi", field_path="data.tsi_freight_change_pct", source_states=source_states, now=now, observation_time=bts_observation, confidence=0.99)
    _add_feature(supply, "supply_truck_activity_norm", _clamp01(truck_index / 150.0) if truck_index is not None else None, source_id="bts_freight_tsi", field_path="data.truck_index", source_states=source_states, now=now, observation_time=bts_observation, confidence=0.99)

    estimates = planes["estimates_dispersion"]
    _add_feature(estimates, "estimates_sec_revision_proxy_norm", sec.get("sec_estimate_revision_drift_norm"), source_id="sec_edgar_context", field_path="derived.global_features.sec_estimate_revision_drift_norm", source_states=source_states, now=now, confidence=0.65)
    news_payload = payloads.get("schwab_symbol_news", {})
    items_by_symbol = news_payload.get("items_by_symbol") if isinstance(news_payload.get("items_by_symbol"), Mapping) else {}
    analyst_positive = analyst_negative = analyst_items = total_items = 0
    positive_terms = ("upgrade", "raises price target", "price target raised", "initiates outperform", "buy rating")
    negative_terms = ("downgrade", "cuts price target", "price target cut", "initiates underperform", "sell rating")
    analyst_terms = positive_terms + negative_terms + ("analyst", "price target", "rating", "estimate")
    sentiment_values: list[float] = []
    for raw_items in items_by_symbol.values():
        if not isinstance(raw_items, list):
            continue
        for item in raw_items:
            if not isinstance(item, Mapping):
                continue
            total_items += 1
            text = f"{item.get('headline', '')} {item.get('summary', '')}".lower()
            if any(term in text for term in analyst_terms):
                analyst_items += 1
            if any(term in text for term in positive_terms):
                analyst_positive += 1
            if any(term in text for term in negative_terms):
                analyst_negative += 1
    news_derived_symbols = _path_value(news_payload, "derived.symbol_features") or {}
    for row in news_derived_symbols.values() if isinstance(news_derived_symbols, Mapping) else []:
        if isinstance(row, Mapping):
            value = _safe_float(row.get("news_sentiment"))
            if value is not None:
                sentiment_values.append(value)
    direction = 0.5 if analyst_items <= 0 else _clamp01(0.5 + (analyst_positive - analyst_negative) / (2.0 * analyst_items))
    dispersion = 0.0
    if sentiment_values:
        mean = sum(sentiment_values) / len(sentiment_values)
        dispersion = _clamp01(math.sqrt(sum((value - mean) ** 2 for value in sentiment_values) / len(sentiment_values)))
    _add_feature(estimates, "estimates_analyst_activity_norm", _clamp01(analyst_items / max(min(total_items, 20), 1)), source_id="schwab_symbol_news", field_path="items_by_symbol.*.headline", source_states=source_states, now=now, confidence=0.68)
    _add_feature(estimates, "estimates_analyst_direction_norm", direction, source_id="schwab_symbol_news", field_path="items_by_symbol.*.headline.classified_direction", source_states=source_states, now=now, confidence=0.62)
    _add_feature(estimates, "estimates_news_dispersion_proxy_norm", dispersion, source_id="schwab_symbol_news", field_path="derived.symbol_features.*.news_sentiment", source_states=source_states, now=now, confidence=0.58)
    _add_feature(estimates, "estimates_consensus_analyst_coverage_norm", consensus.get("consensus_analyst_coverage_norm"), source_id="analyst_consensus_context", field_path="derived.global_features.consensus_analyst_coverage_norm", source_states=source_states, now=now, confidence=0.94)
    _add_feature(estimates, "estimates_consensus_revision_activity_norm", consensus.get("consensus_revision_activity_norm"), source_id="analyst_consensus_context", field_path="derived.global_features.consensus_revision_activity_norm", source_states=source_states, now=now, confidence=0.94)
    _add_feature(estimates, "estimates_consensus_revision_direction_norm", consensus.get("consensus_revision_direction_norm"), source_id="analyst_consensus_context", field_path="derived.global_features.consensus_revision_direction_norm", source_states=source_states, now=now, confidence=0.94)
    _add_feature(estimates, "estimates_consensus_dispersion_norm", consensus.get("consensus_dispersion_norm"), source_id="analyst_consensus_context", field_path="derived.global_features.consensus_dispersion_norm", source_states=source_states, now=now, confidence=0.94)

    impact = planes["capacity_market_impact"]
    curves = capacity.get("curves") if isinstance(capacity.get("curves"), list) else []
    curve_values = [
        value
        for row in curves
        if isinstance(row, Mapping)
        for value in [_safe_float(row.get("recommended_capacity_fraction"))]
        if value is not None
    ]
    mean_capacity = sum(curve_values) / len(curve_values) if curve_values else None
    _add_feature(impact, "capacity_curve_coverage_norm", _clamp01(len(curves) / 20.0), source_id="portfolio_capacity_curves", field_path="summary.curve_count", source_states=source_states, now=now, confidence=0.92)
    _add_feature(impact, "capacity_constraint_pressure_norm", (1.0 - mean_capacity) if mean_capacity is not None else None, source_id="portfolio_capacity_curves", field_path="curves.*.recommended_capacity_fraction", source_states=source_states, now=now, confidence=0.9)
    mae_bps = _safe_float(_path_value(calibration, "metrics.mae_bps"))
    p95_bps = _safe_float(_path_value(calibration, "metrics.p95_bps"))
    _add_feature(impact, "capacity_calibration_error_norm", _clamp01(mae_bps / 50.0) if mae_bps is not None else None, source_id="paper_execution_calibration", field_path="metrics.mae_bps", source_states=source_states, now=now, confidence=0.95)
    _add_feature(impact, "capacity_tail_slippage_norm", _clamp01(p95_bps / 100.0) if p95_bps is not None else None, source_id="paper_execution_calibration", field_path="metrics.p95_bps", source_states=source_states, now=now, confidence=0.95)
    tradeability = _safe_float(micro.get("market_micro_tradeability_score_norm"))
    _add_feature(impact, "capacity_microstructure_pressure_norm", (1.0 - tradeability) if tradeability is not None else None, source_id="market_micro_context", field_path="derived.global_features.market_micro_tradeability_score_norm", source_states=source_states, now=now, confidence=0.88)
    return planes


def _build_plane(
    spec: Mapping[str, Any],
    candidates: dict[str, dict[str, Any]],
    source_states: Mapping[str, Mapping[str, Any]],
    *,
    scoring: Mapping[str, Any],
    minimum_score: float,
) -> dict[str, Any]:
    source_ids = [str(value) for value in spec.get("source_ids", []) if str(value)]
    source_rows = [source_states.get(source_id, {}) for source_id in source_ids]
    source_health = sum(1.0 for row in source_rows if row.get("ok") is True) / max(len(source_ids), 1)
    required_feature_count = max(int(spec.get("required_feature_count", 3) or 3), 1)
    feature_completeness = min(len(candidates) / required_feature_count, 1.0)
    contributing_source_ids = {
        str(lineage.get("source_id") or "")
        for row in candidates.values()
        for lineage in row.get("lineage", [])
        if str(lineage.get("source_id") or "")
    }
    contributing_states = [source_states.get(source_id, {}) for source_id in contributing_source_ids]
    freshness = (
        sum(float(row.get("freshness_norm", 0.0) or 0.0) for row in contributing_states)
        / max(len(contributing_states), 1)
    )
    lineage_count = sum(
        1
        for row in candidates.values()
        if row.get("lineage") and all(bool(item.get("point_in_time_valid", False)) for item in row.get("lineage", []))
    )
    lineage_coverage = lineage_count / max(len(candidates), 1)
    routing = 1.0 if spec.get("target_domains") and spec.get("target_symbols") else 0.0
    distinct_families = {
        str(row.get("source_family") or "")
        for row in contributing_states
        if str(row.get("source_family") or "")
    }
    cross_verification = min(len(distinct_families) / 2.0, 1.0)
    components = {
        "source_health_norm": source_health,
        "feature_completeness_norm": feature_completeness,
        "freshness_norm": freshness,
        "point_in_time_lineage_norm": lineage_coverage,
        "routing_norm": routing,
        "cross_verification_norm": cross_verification,
    }
    weights = {
        "source_health_norm": float(scoring.get("source_health_weight", 0.30) or 0.30),
        "feature_completeness_norm": float(scoring.get("feature_completeness_weight", 0.25) or 0.25),
        "freshness_norm": float(scoring.get("freshness_weight", 0.15) or 0.15),
        "point_in_time_lineage_norm": float(scoring.get("point_in_time_lineage_weight", 0.15) or 0.15),
        "routing_norm": float(scoring.get("routing_weight", 0.10) or 0.10),
        "cross_verification_norm": float(scoring.get("cross_verification_weight", 0.05) or 0.05),
    }
    weight_sum = max(sum(weights.values()), 1e-9)
    score = 100.0 * sum(components[key] * weights[key] for key in components) / weight_sum
    caveats: list[str] = []
    direct_consensus_ready = bool(
        source_states.get("analyst_consensus_context", {}).get("direct_consensus_ready") is True
    )
    if spec.get("plane_id") == "estimates_dispersion" and not direct_consensus_ready:
        score = min(score, 87.0)
        caveats.append("direct_point_in_time_consensus_not_present; SEC and broker-news proxies are context-only")
    signal_key = str(spec.get("signal_key") or "")
    if candidates:
        _mean_feature(candidates, signal_key, list(candidates.keys()))
    features = {key: round(float(row["value"]), 8) for key, row in candidates.items() if row.get("value") is not None}
    lineage = {key: row.get("lineage", []) for key, row in candidates.items() if row.get("lineage")}
    return {
        "plane_id": spec.get("plane_id"),
        "title": spec.get("title"),
        "plane_class": spec.get("plane_class"),
        "signal_key": signal_key,
        "score_pct": round(score, 3),
        "grade": percentage_grade(score),
        "status": "ready" if score >= minimum_score and signal_key in features else "degraded",
        "scoring_components": {key: round(value, 6) for key, value in components.items()},
        "required_feature_count": required_feature_count,
        "observed_feature_count": max(len(features) - (1 if signal_key in features else 0), 0),
        "missing_required_feature_slots": max(required_feature_count - max(len(features) - 1, 0), 0),
        "source_ids": source_ids,
        "contributing_source_ids": sorted(contributing_source_ids),
        "distinct_source_family_count": len(distinct_families),
        "direct_consensus_ready": direct_consensus_ready if spec.get("plane_id") == "estimates_dispersion" else None,
        "features": features,
        "lineage": lineage,
        "routing": {
            "target_domains": list(spec.get("target_domains") or []),
            "target_symbols": list(spec.get("target_symbols") or []),
            "paper_and_training_context_only": True,
        },
        "caveats": caveats,
    }


def _symbol_context_features(payloads: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    micro = _symbol_features(payloads.get("market_micro_context", {}))
    extended = _symbol_features(payloads.get("extended_quant_context", {}))
    options = _symbol_features(payloads.get("options_flow_context", {}))
    sec = _symbol_features(payloads.get("sec_edgar_context", {}))
    news = _symbol_features(payloads.get("schwab_symbol_news", {}))
    consensus = _symbol_features(payloads.get("analyst_consensus_context", {}))
    capacity_rows = payloads.get("portfolio_capacity_curves", {}).get("curves")
    capacity_by_symbol = {
        str(row.get("symbol") or "").upper(): row
        for row in capacity_rows
        if isinstance(capacity_rows, list) and isinstance(row, Mapping) and str(row.get("symbol") or "")
    } if isinstance(capacity_rows, list) else {}
    symbols = sorted({str(symbol).upper() for mapping in (micro, extended, options, sec, news, consensus) for symbol in mapping.keys()} | set(capacity_by_symbol))
    out: dict[str, dict[str, float]] = {}
    for symbol in symbols:
        micro_row = micro.get(symbol) if isinstance(micro.get(symbol), Mapping) else {}
        extended_row = extended.get(symbol) if isinstance(extended.get(symbol), Mapping) else {}
        options_row = options.get(symbol) if isinstance(options.get(symbol), Mapping) else {}
        sec_row = sec.get(symbol) if isinstance(sec.get(symbol), Mapping) else {}
        news_row = news.get(symbol) if isinstance(news.get(symbol), Mapping) else {}
        consensus_row = consensus.get(symbol) if isinstance(consensus.get(symbol), Mapping) else {}
        row: dict[str, float] = {}
        borrow_availability = _safe_float(options_row.get("short_borrow_availability_norm"))
        lending_values = [
            value
            for value in (
                (1.0 - borrow_availability) if borrow_availability is not None else None,
                _safe_float(options_row.get("short_borrow_fee_norm")),
                _safe_float(options_row.get("short_utilization_norm")),
                _safe_float(extended_row.get("short_ftd_quantity_norm")),
                _safe_float(micro_row.get("market_micro_short_pressure_norm")),
            )
            if value is not None
        ]
        if lending_values:
            row["context_securities_lending_signal_norm"] = _clamp01(sum(lending_values) / len(lending_values))
        vol_values = [
            value
            for key in ("options_iv_skew_norm", "options_iv_term_structure_norm", "options_surface_change_norm")
            for value in [_safe_float(options_row.get(key))]
            if value is not None
        ]
        if vol_values:
            row["context_volatility_surface_signal_norm"] = _clamp01(sum(vol_values) / len(vol_values))
        passive_values = [
            value
            for key in ("etf_creation_redemption_stress_norm", "etf_fund_family_flow_norm", "etf_underlying_basket_stress_norm")
            for value in [_safe_float(micro_row.get(key))]
            if value is not None
        ]
        if passive_values:
            row["context_passive_mechanical_flow_signal_norm"] = _clamp01(sum(passive_values) / len(passive_values))
        positioning_values = [
            value
            for value in (
                _safe_float(extended_row.get("short_ftd_quantity_norm")),
                _safe_float(micro_row.get("market_micro_order_flow_imbalance_norm")),
                _safe_float(micro_row.get("etf_fund_family_flow_norm")),
            )
            if value is not None
        ]
        if positioning_values:
            row["context_positioning_crowding_signal_norm"] = _clamp01(sum(positioning_values) / len(positioning_values))
        estimate_values = [
            value
            for value in (
                _safe_float(consensus_row.get("consensus_revision_direction_norm")),
                _safe_float(consensus_row.get("consensus_dispersion_norm")),
                _safe_float(consensus_row.get("consensus_revision_activity_norm")),
                _safe_float(sec_row.get("sec_estimate_revision_drift_norm")),
                _safe_float(news_row.get("news_sentiment")),
                _safe_float(news_row.get("news_novelty_norm")),
            )
            if value is not None
        ]
        if estimate_values:
            normalized = [_clamp01(0.5 + value / 2.0) if value < 0.0 else _clamp01(value) for value in estimate_values]
            row["context_estimates_dispersion_signal_norm"] = _clamp01(sum(normalized) / len(normalized))
        capacity_row = capacity_by_symbol.get(symbol) if isinstance(capacity_by_symbol.get(symbol), Mapping) else {}
        capacity_fraction = _safe_float(capacity_row.get("recommended_capacity_fraction"))
        tradeability = _safe_float(micro_row.get("market_micro_tradeability_score_norm"))
        impact_values = [
            value
            for value in (
                (1.0 - capacity_fraction) if capacity_fraction is not None else None,
                (1.0 - tradeability) if tradeability is not None else None,
                _safe_float(micro_row.get("market_micro_spread_regime_norm")),
                _safe_float(micro_row.get("market_micro_depth_collapse_norm")),
            )
            if value is not None
        ]
        if impact_values:
            row["context_capacity_market_impact_signal_norm"] = _clamp01(sum(impact_values) / len(impact_values))
        if row:
            out[symbol] = {key: round(value, 8) for key, value in row.items()}
    return out


def build_decision_context_mesh(
    project_root: Path = PROJECT_ROOT,
    *,
    config: Mapping[str, Any] | None = None,
    now_utc: datetime | None = None,
    direct_sources: Mapping[str, Mapping[str, Any]] | None = None,
    refresh_capacity: bool = True,
) -> dict[str, Any]:
    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    mesh_config = dict(config or load_decision_context_mesh_config())
    contract = dict(mesh_config.get("contract") or {})
    scoring = dict(mesh_config.get("scoring") or {})

    if refresh_capacity:
        capacity_path = project_root / LOCAL_SOURCE_SPECS["portfolio_capacity_curves"][0]
        capacity_payload = build_capacity_payload(project_root)
        _atomic_write_json(capacity_path, capacity_payload)

    payloads: dict[str, dict[str, Any]] = {}
    source_states: dict[str, dict[str, Any]] = {}
    for source_id in LOCAL_SOURCE_SPECS:
        payload, state = _local_source_state(source_id, project_root, now)
        payloads[source_id] = payload
        source_states[source_id] = state
    consensus_payload, consensus_state = _analyst_consensus_source_state(project_root, now)
    if consensus_payload and consensus_state:
        payloads["analyst_consensus_context"] = consensus_payload
        source_states["analyst_consensus_context"] = consensus_state
    for source_id, state in (direct_sources or {}).items():
        if source_id in DIRECT_SOURCE_SPECS and isinstance(state, Mapping):
            source_states[source_id] = dict(state)

    candidate_planes = _feature_candidates(payloads, source_states, now=now)
    minimum_score = float(contract.get("minimum_plane_score_pct", 70.0) or 70.0)
    plane_rows = [
        _build_plane(
            spec,
            candidate_planes.get(str(spec.get("plane_id") or ""), {}),
            source_states,
            scoring=scoring,
            minimum_score=minimum_score,
        )
        for spec in mesh_config.get("planes", [])
        if isinstance(spec, Mapping)
    ]
    macro_rows = [row for row in plane_rows if row.get("plane_class") == "macro"]
    micro_rows = [row for row in plane_rows if row.get("plane_class") == "micro"]
    macro_pct = round(sum(float(row.get("score_pct", 0.0)) for row in macro_rows) / max(len(macro_rows), 1), 3)
    micro_pct = round(sum(float(row.get("score_pct", 0.0)) for row in micro_rows) / max(len(micro_rows), 1), 3)
    signal_features = {
        str(row.get("signal_key")): float((row.get("features") or {}).get(str(row.get("signal_key"))))
        for row in plane_rows
        if str(row.get("signal_key") or "") and _safe_float((row.get("features") or {}).get(str(row.get("signal_key")))) is not None
    }
    plane_coverage = len(signal_features) / max(len(PLANE_SIGNAL_FEATURE_KEYS), 1)
    average_score = (macro_pct + micro_pct) / 2.0
    average_freshness = sum(float((row.get("scoring_components") or {}).get("freshness_norm", 0.0)) for row in plane_rows) / max(len(plane_rows), 1)
    average_lineage = sum(float((row.get("scoring_components") or {}).get("point_in_time_lineage_norm", 0.0)) for row in plane_rows) / max(len(plane_rows), 1)
    average_cross = sum(float((row.get("scoring_components") or {}).get("cross_verification_norm", 0.0)) for row in plane_rows) / max(len(plane_rows), 1)
    global_features = {
        **signal_features,
        "context_mesh_available_norm": 1.0 if signal_features else 0.0,
        "context_mesh_macro_grade_norm": _clamp01(macro_pct / 100.0),
        "context_mesh_micro_grade_norm": _clamp01(micro_pct / 100.0),
        "context_mesh_coverage_norm": _clamp01(plane_coverage),
        "context_mesh_confidence_norm": _clamp01(average_score / 100.0),
        "context_mesh_freshness_norm": _clamp01(average_freshness),
        "context_mesh_lineage_coverage_norm": _clamp01(average_lineage),
        "context_mesh_cross_verification_norm": _clamp01(average_cross),
    }
    symbol_features = _symbol_context_features(payloads)
    future_excluded = {
        source_id: [str(state.get("observation_time") or state.get("timestamp_utc") or "")]
        for source_id, state in source_states.items()
        if state.get("timestamp_in_future") is True
    }
    payload = {
        "timestamp_utc": now.isoformat(),
        "schema_version": 1,
        "mesh_id": mesh_config.get("mesh_id", "decision_context_mesh_v1"),
        "provider": "governed_point_in_time_official_and_internal_context_mesh",
        "contract": contract,
        "methodology": {
            "point_in_time_only": True,
            "future_observations_rejected": True,
            "missing_values_are_not_zero_filled": True,
            "feature_values_are_normalized_zero_to_one": True,
            "scores_are_organic_weighted_evidence_not_operator_overrides": True,
            "direct_sources_use_bounded_last_good_fallback": True,
            "freshness_is_cadence_aware_with_hard_staleness_slos": True,
            "context_does_not_authorize_orders_or_promotion": True,
        },
        "coverage": {
            "required_plane_count": int(contract.get("required_plane_count", 12) or 12),
            "observed_plane_count": len(plane_rows),
            "ready_plane_count": sum(1 for row in plane_rows if row.get("status") == "ready"),
            "signal_feature_count": len(signal_features),
            "required_signal_feature_count": len(PLANE_SIGNAL_FEATURE_KEYS),
            "signal_coverage_ratio": round(plane_coverage, 6),
            "source_count": len(source_states),
            "healthy_source_count": sum(1 for row in source_states.values() if row.get("ok") is True),
            "future_observations_excluded": future_excluded,
            "future_observation_selected": False,
        },
        "grade_summary": {
            "macro_percentage": macro_pct,
            "macro_grade": percentage_grade(macro_pct),
            "micro_percentage": micro_pct,
            "micro_grade": percentage_grade(micro_pct),
            "combined_percentage": round(average_score, 3),
            "combined_grade": percentage_grade(average_score),
            "grade_scale": "A+>=97,A>=93,A->=90,B+>=87,B>=83,B->=80,C+>=77,C>=73,C->=70,D+>=67,D>=63,D->=60,F<60",
        },
        "sources": source_states,
        "planes": plane_rows,
        "derived": {
            "global_features": {key: round(value, 8) for key, value in global_features.items()},
            "symbol_features": symbol_features,
        },
        "routing": {
            "plane_routes": {
                str(row.get("plane_id")): row.get("routing", {}) for row in plane_rows
            },
            "global_feature_keys": sorted(global_features),
            "symbol_feature_count": len(symbol_features),
            "consumer_surfaces": [
                "paper_decision_context",
                "runtime_training_gap_fill",
                "behavior_dataset_point_in_time_features",
                "replay_and_research",
                "source_verification",
            ],
            "excluded_surfaces": ["paper_execution_authority", "live_execution_authority", "automatic_promotion"],
        },
    }
    assessment = assess_decision_context_mesh(payload, now_utc=now)
    payload["consumer_contract"] = assessment
    payload["ok"] = bool(assessment.get("ready", False))
    payload["overall_status"] = "ready" if payload["ok"] else "degraded"
    return payload


def _write_history(payload: Mapping[str, Any], history_root: Path) -> Path:
    timestamp = _parse_timestamp(payload.get("timestamp_utc")) or datetime.now(timezone.utc)
    path = history_root / f"decision_context_mesh_{timestamp:%Y%m%d}.jsonl.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "at", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n")
    return path


def collect_decision_context_mesh(
    project_root: Path = PROJECT_ROOT,
    *,
    timeout_seconds: int = 20,
    allow_network: bool = True,
    refresh_capacity: bool = True,
    now_utc: datetime | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    latest_path = project_root / "exports" / "external_context" / LATEST_PATH.name
    previous = _read_json(latest_path)
    direct_sources = _collect_direct_sources(
        now=now,
        timeout_seconds=timeout_seconds,
        allow_network=allow_network,
        previous=previous,
    )
    payload = build_decision_context_mesh(
        project_root,
        now_utc=now,
        direct_sources=direct_sources,
        refresh_capacity=refresh_capacity,
    )
    health = {
        "timestamp_utc": now.isoformat(),
        "schema_version": 1,
        "ok": payload.get("ok", False),
        "overall_status": payload.get("overall_status", "degraded"),
        "macro_percentage": (payload.get("grade_summary") or {}).get("macro_percentage", 0.0),
        "macro_grade": (payload.get("grade_summary") or {}).get("macro_grade", "F"),
        "micro_percentage": (payload.get("grade_summary") or {}).get("micro_percentage", 0.0),
        "micro_grade": (payload.get("grade_summary") or {}).get("micro_grade", "F"),
        "combined_percentage": (payload.get("grade_summary") or {}).get("combined_percentage", 0.0),
        "combined_grade": (payload.get("grade_summary") or {}).get("combined_grade", "F"),
        "plane_scores": {
            str(row.get("plane_id")): {
                "score_pct": row.get("score_pct"),
                "grade": row.get("grade"),
                "status": row.get("status"),
                "caveats": row.get("caveats", []),
            }
            for row in payload.get("planes", [])
            if isinstance(row, Mapping)
        },
        "source_health": {
            "healthy": (payload.get("coverage") or {}).get("healthy_source_count", 0),
            "total": (payload.get("coverage") or {}).get("source_count", 0),
            "failed_source_ids": sorted(
                source_id
                for source_id, row in (payload.get("sources") or {}).items()
                if isinstance(row, Mapping) and row.get("ok") is not True
            ),
        },
        "sources": {
            source_id: {
                "ok": row.get("ok") is True,
                "fresh": row.get("fresh") is True,
                "fallback": bool(row.get("fallback", False)),
                "contract_participates": True,
            }
            for source_id, row in (payload.get("sources") or {}).items()
            if isinstance(row, Mapping)
        },
        "consumer_contract": payload.get("consumer_contract", {}),
        "artifact_path": str(latest_path),
        "safety_contract": {
            "paper_execution_authority": False,
            "live_execution_authority": False,
            "automatic_promotion_authority": False,
            "profitability_guaranteed": False,
        },
    }
    _atomic_write_json(latest_path, payload)
    _atomic_write_json(project_root / "governance" / "health" / HEALTH_PATH.name, health)
    _write_history(payload, project_root / "data" / "external_context" / HISTORY_ROOT.name)
    return payload, health


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect and grade the governed twelve-plane macro/micro context mesh.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--no-network", action="store_true")
    parser.add_argument("--no-refresh-capacity", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    payload, health = collect_decision_context_mesh(
        project_root,
        timeout_seconds=max(int(args.timeout), 3),
        allow_network=not args.no_network,
        refresh_capacity=not args.no_refresh_capacity,
    )
    if args.json:
        print(json.dumps(health, ensure_ascii=True, sort_keys=True))
    else:
        print(
            "decision_context_mesh "
            f"status={health.get('overall_status')} "
            f"macro={health.get('macro_percentage')}%({health.get('macro_grade')}) "
            f"micro={health.get('micro_percentage')}%({health.get('micro_grade')}) "
            f"combined={health.get('combined_percentage')}%({health.get('combined_grade')})"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
