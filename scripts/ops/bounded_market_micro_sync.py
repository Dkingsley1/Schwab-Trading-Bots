#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HEALTH_PATH = PROJECT_ROOT / "governance" / "health" / "market_micro_sync_latest.json"
LAST_VERIFIED_HEALTH_PATH = PROJECT_ROOT / "governance" / "health" / "market_micro_sync_last_verified.json"
PAYLOAD_PATH = PROJECT_ROOT / "exports" / "external_context" / "market_micro_latest.json"
FALLBACK_SYMBOLS = (
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "TLT",
    "GLD",
    "USO",
    "UUP",
    "AAPL",
    "MSFT",
    "NVDA",
    "TSLA",
)


def _json_from_stdout(stdout: str) -> dict:
    for line in reversed([row.strip() for row in str(stdout or "").splitlines() if row.strip()]):
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _payload_timestamp(path: Path, payload: dict) -> float:
    for key in ("timestamp_utc", "generated_utc", "updated_at_utc", "updated_at", "created_at"):
        raw = str(payload.get(key) or "").strip()
        if not raw:
            continue
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()
        except Exception:
            continue
    try:
        return float(path.stat().st_mtime)
    except Exception:
        return 0.0


def _minimum_ok_sources(total: int, *, floor: int = 3, tolerate_failures: int = 1, min_ratio: float = 0.75) -> int:
    if int(total) <= 0:
        return 0
    ratio_target = int(round(float(total) * float(min_ratio)))
    tolerated_target = int(total) - max(int(tolerate_failures), 0)
    return min(int(total), max(int(floor), ratio_target, tolerated_target))


def _participating_source_counts(sources: Mapping[str, Any]) -> tuple[int, int]:
    ok_count = 0
    total_count = 0
    for row in sources.values():
        if not isinstance(row, Mapping):
            continue
        if not bool(row.get("contract_participates", True)):
            continue
        total_count += 1
        if bool(row.get("ok", False)):
            ok_count += 1
    return ok_count, total_count


def _sources_usable_for_market_micro(sources: Mapping[str, Any]) -> bool:
    ok_count, total_count = _participating_source_counts(sources)
    if total_count <= 0 or ok_count < _minimum_ok_sources(total_count):
        return False
    local_micro_ok = bool((sources.get("local_micro") or {}).get("ok", False))
    finra_ok = bool((sources.get("finra_short_volume") or {}).get("ok", False))
    nasdaq_halts_ok = bool((sources.get("nasdaq_trade_halts") or {}).get("ok", False))
    treasury_ok = bool((sources.get("treasury_auctions") or {}).get("ok", False))
    return bool(local_micro_ok and (finra_ok or nasdaq_halts_ok) and treasury_ok)


def _save_last_verified_health(payload: Mapping[str, Any]) -> None:
    sources = payload.get("sources") if isinstance(payload.get("sources"), Mapping) else {}
    if not _sources_usable_for_market_micro(sources):
        return
    LAST_VERIFIED_HEALTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    LAST_VERIFIED_HEALTH_PATH.write_text(json.dumps(dict(payload), ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _source_contract(payload: Mapping[str, Any], source_name: str) -> dict[str, float]:
    contracts = ((payload.get("collection_contract") or {}).get("source_contracts") or {}) if isinstance(payload.get("collection_contract"), Mapping) else {}
    row = contracts.get(source_name) if isinstance(contracts, Mapping) else {}
    if not isinstance(row, Mapping):
        row = {}
    return {
        "source_confidence_norm": float(row.get("source_confidence_norm", 0.9) or 0.9),
        "schema_confidence_norm": float(row.get("schema_confidence_norm", 0.9) or 0.9),
        "freshness_norm": float(row.get("freshness_norm", 0.0) or 0.0),
    }


def _reconstruct_sources_from_payload(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    collection_contract = payload.get("collection_contract") if isinstance(payload.get("collection_contract"), Mapping) else {}
    source_contracts = collection_contract.get("source_contracts") if isinstance(collection_contract.get("source_contracts"), Mapping) else {}
    if not source_contracts:
        return {}
    derived = payload.get("derived") if isinstance(payload.get("derived"), Mapping) else {}
    symbol_features = derived.get("symbol_features") if isinstance(derived.get("symbol_features"), Mapping) else {}
    treasury = derived.get("treasury_auctions") if isinstance(derived.get("treasury_auctions"), Mapping) else {}
    finra = derived.get("finra_short_volume") if isinstance(derived.get("finra_short_volume"), Mapping) else {}
    halts = derived.get("nasdaq_trade_halts") if isinstance(derived.get("nasdaq_trade_halts"), Mapping) else {}
    if not any(isinstance(row, Mapping) for row in (symbol_features, treasury, finra, halts)):
        return {}

    finra_rows = finra.get("rows") if isinstance(finra.get("rows"), Mapping) else {}
    halt_rows = halts.get("rows") if isinstance(halts.get("rows"), list) else []
    treasury_rows = treasury.get("rows") if isinstance(treasury.get("rows"), list) else []
    local_contract = _source_contract(payload, "local_micro")
    treasury_contract = _source_contract(payload, "treasury_auctions")
    finra_contract = _source_contract(payload, "finra_short_volume")
    halts_contract = _source_contract(payload, "nasdaq_trade_halts")
    return {
        "local_micro": {
            "ok": bool(symbol_features and local_contract.get("freshness_norm", 0.0) > 0.0),
            "symbol_count": len(symbol_features),
            "required": True,
            "contract_participates": True,
            **local_contract,
        },
        "treasury_auctions": {
            "ok": bool(treasury.get("ok", False) and treasury_rows),
            "rows": len(treasury_rows),
            "error": treasury.get("error"),
            "required": True,
            "contract_participates": True,
            **treasury_contract,
        },
        "finra_short_volume": {
            "ok": bool(finra.get("ok", False) and finra_rows),
            "symbol_count": len(finra_rows),
            "error": finra.get("error"),
            "required": True,
            "contract_participates": True,
            **finra_contract,
        },
        "nasdaq_trade_halts": {
            "ok": bool(halts.get("ok", False)),
            "rows": len(halt_rows),
            "all_rows_count": int(halts.get("all_rows_count", 0) or len(halt_rows)),
            "error": halts.get("error"),
            "required": False,
            "contract_participates": True,
            **halts_contract,
        },
    }


def _fallback_from_sources(
    base: Mapping[str, Any],
    *,
    now: datetime,
    sources: Mapping[str, Any],
    reason: str,
    fallback_age_seconds: float | None,
) -> dict[str, Any]:
    ok_count, total_count = _participating_source_counts(sources)
    source_contracts = {
        name: {
            "source_confidence_norm": float((row or {}).get("source_confidence_norm", 0.0) or 0.0),
            "schema_confidence_norm": float((row or {}).get("schema_confidence_norm", 0.0) or 0.0),
            "freshness_norm": float((row or {}).get("freshness_norm", 0.0) or 0.0),
        }
        for name, row in sources.items()
        if isinstance(row, Mapping)
    }
    return {
        **dict(base),
        "timestamp_utc": now.isoformat(),
        "ok": ok_count > 0,
        "fallback_used": True,
        "source_reconstruction_used": True,
        "partial_data": ok_count < total_count,
        "degraded": True,
        "warning": reason,
        "fallback_source_age_seconds": round(float(fallback_age_seconds), 3) if fallback_age_seconds is not None else None,
        "sources": dict(sources),
        "source_contracts": source_contracts,
    }


def _neutral_micro_features() -> dict:
    return {
        "market_micro_premarket_pressure_norm": 0.0,
        "market_micro_opening_auction_norm": 0.0,
        "market_micro_opening_auction_imbalance_norm": 0.5,
        "market_micro_opening_drive_pressure_norm": 0.0,
        "market_micro_power_hour_pressure_norm": 0.0,
        "market_micro_closing_auction_norm": 0.0,
        "market_micro_closing_auction_imbalance_norm": 0.5,
        "market_micro_closing_cross_pressure_norm": 0.0,
        "market_micro_auction_print_pressure_norm": 0.0,
        "market_micro_relative_volume_norm": 0.0,
        "market_micro_order_flow_imbalance_norm": 0.5,
        "market_micro_options_flow_norm": 0.0,
        "market_micro_short_pressure_norm": 0.0,
        "market_micro_credit_flow_norm": 0.0,
        "market_micro_gap_continuation_norm": 0.0,
        "market_micro_reversal_risk_norm": 0.0,
        "market_micro_trend_persistence_norm": 0.0,
        "market_micro_range_expansion_norm": 0.0,
        "market_micro_block_trade_norm": 0.0,
        "market_micro_trade_halt_norm": 0.0,
        "market_micro_luld_pause_norm": 0.0,
        "market_micro_ssr_active_norm": 0.0,
        "market_micro_resume_window_norm": 0.0,
        "market_micro_dark_pool_pressure_norm": 0.0,
        "market_micro_off_exchange_share_norm": 0.0,
        "market_micro_spread_regime_norm": 0.0,
        "market_micro_spread_widening_norm": 0.0,
        "market_micro_queue_depth_decay_norm": 0.0,
        "market_micro_depth_collapse_norm": 0.0,
        "market_micro_quote_fade_rate_norm": 0.0,
        "market_micro_tradeability_score_norm": 0.5,
        "market_micro_session_open_norm": 0.0,
        "market_micro_session_midday_norm": 0.0,
        "market_micro_session_power_hour_norm": 0.0,
        "market_micro_overnight_gap_norm": 0.0,
        "market_micro_post_event_drift_norm": 0.0,
        "market_micro_lunch_chop_norm": 0.0,
        "market_micro_open_close_imbalance_regime_norm": 0.0,
        "market_micro_symbol_cooldown_pressure_norm": 0.0,
        "market_micro_gap_fade_risk_norm": 0.0,
        "market_micro_overnight_event_hazard_norm": 0.0,
        "etf_nav_premium_discount_norm": 0.0,
        "etf_creation_redemption_stress_norm": 0.0,
        "etf_primary_secondary_liquidity_norm": 0.5,
        "etf_underlying_basket_stress_norm": 0.0,
        "etf_fund_family_flow_norm": 0.0,
    }


def _write_neutral_fallback_payload(now: datetime, *, reason: str) -> dict:
    features = _neutral_micro_features()
    payload = {
        "timestamp_utc": now.isoformat(),
        "provider": "market_micro_context",
        "fallback_used": True,
        "fallback_reason": reason,
        "degraded": True,
        "collection_contract": {
            "provider": "market_micro_context",
            "provider_confidence_norm": 0.5,
            "source_contracts": {
                "local_neutral_fallback": {
                    "source_confidence_norm": 0.5,
                    "schema_confidence_norm": 0.9,
                    "freshness_norm": 1.0,
                }
            },
        },
        "derived": {
            "global_features": dict(features),
            "symbol_features": {symbol: dict(features) for symbol in FALLBACK_SYMBOLS},
            "fallback": {
                "mode": "neutral_local_market_micro",
                "reason": reason,
            },
        },
    }
    PAYLOAD_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAYLOAD_PATH.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return payload


def _timeout_payload(exc: subprocess.TimeoutExpired, *, outer_timeout_seconds: int) -> dict:
    now = datetime.now(timezone.utc)
    existing_payload = _load_json(PAYLOAD_PATH)
    observed_ts = _payload_timestamp(PAYLOAD_PATH, existing_payload) if existing_payload else 0.0
    age_seconds = max(time.time() - observed_ts, 0.0) if observed_ts > 0.0 else None
    max_fallback_age = max(float(os.getenv("MARKET_MICRO_TIMEOUT_FALLBACK_MAX_AGE_SECONDS", "86400") or 86400), 0.0)
    fallback_ready = bool(existing_payload and age_seconds is not None and age_seconds <= max_fallback_age)
    base = {
        "timestamp_utc": now.isoformat(),
        "timeout": True,
        "outer_timeout_seconds": int(outer_timeout_seconds),
        "error": "bounded_market_micro_sync_timeout",
        "stdout_tail": "\n".join(str(exc.stdout or "").splitlines()[-8:]),
        "stderr_tail": "\n".join(str(exc.stderr or "").splitlines()[-8:]),
        "fallback_payload_path": str(PAYLOAD_PATH),
        "fallback_payload_age_seconds": round(float(age_seconds), 3) if age_seconds is not None else None,
        "fallback_max_age_seconds": round(float(max_fallback_age), 3),
    }
    verified_payload = _load_json(LAST_VERIFIED_HEALTH_PATH)
    verified_ts = _payload_timestamp(LAST_VERIFIED_HEALTH_PATH, verified_payload) if verified_payload else 0.0
    verified_age = max(time.time() - verified_ts, 0.0) if verified_ts > 0.0 else None
    verified_sources = verified_payload.get("sources") if isinstance(verified_payload.get("sources"), Mapping) else {}
    if (
        verified_sources
        and verified_age is not None
        and verified_age <= max_fallback_age
        and _sources_usable_for_market_micro(verified_sources)
    ):
        return _fallback_from_sources(
            base,
            now=now,
            sources=verified_sources,
            reason="collector_timeout_using_last_verified_market_micro_sources",
            fallback_age_seconds=verified_age,
        )
    reconstructed_sources = _reconstruct_sources_from_payload(existing_payload) if fallback_ready else {}
    if reconstructed_sources:
        return _fallback_from_sources(
            base,
            now=now,
            sources=reconstructed_sources,
            reason="collector_timeout_using_reconstructed_last_known_market_micro_sources",
            fallback_age_seconds=age_seconds,
        )
    if not fallback_ready and os.getenv("MARKET_MICRO_TIMEOUT_NEUTRAL_FALLBACK_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}:
        _write_neutral_fallback_payload(now, reason="bounded_market_micro_sync_timeout")
        return {
            **base,
            "ok": True,
            "fallback_used": True,
            "neutral_fallback_used": True,
            "partial_data": True,
            "degraded": True,
            "warning": "collector_timeout_using_neutral_local_market_micro_fallback",
            "sources": {
                "local_neutral_fallback": {
                    "ok": True,
                    "contract_participates": True,
                    "source_confidence_norm": 0.5,
                    "schema_confidence_norm": 0.9,
                }
            },
            "source_contracts": {
                "local_neutral_fallback": {
                    "source_confidence_norm": 0.5,
                    "schema_confidence_norm": 0.9,
                    "freshness_norm": 1.0,
                }
            },
        }
    if not fallback_ready:
        return {**base, "ok": False, "fallback_used": False, "sources": {}, "source_contracts": {}}
    return {
        **base,
        "ok": True,
        "fallback_used": True,
        "partial_data": True,
        "degraded": True,
        "warning": "collector_timeout_using_fresh_last_known_market_micro_payload",
        "sources": {
            "last_known_market_micro_payload": {
                "ok": True,
                "contract_participates": False,
                "age_seconds": round(float(age_seconds), 3),
                "path": str(PAYLOAD_PATH),
            }
        },
        "source_contracts": {
            "last_known_market_micro_payload": {
                "source_confidence_norm": 0.72,
                "schema_confidence_norm": 0.9,
                "freshness_norm": max(0.0, 1.0 - (float(age_seconds or 0.0) / max(max_fallback_age, 1.0))),
            }
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run market micro sync with a hard outer timeout.")
    parser.add_argument("--outer-timeout-seconds", type=int, default=90)
    args, passthrough = parser.parse_known_args()

    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "collect_market_micro_context.py"), *passthrough]
    if "--json" not in passthrough:
        cmd.append("--json")
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
            timeout=max(int(args.outer_timeout_seconds), 5),
        )
        payload = _json_from_stdout(proc.stdout)
        if payload:
            _save_last_verified_health(payload)
            print(json.dumps(payload, ensure_ascii=True))
        else:
            print((proc.stdout or "").strip())
        return int(proc.returncode)
    except subprocess.TimeoutExpired as exc:
        payload = _timeout_payload(exc, outer_timeout_seconds=int(args.outer_timeout_seconds))
        HEALTH_PATH.parent.mkdir(parents=True, exist_ok=True)
        HEALTH_PATH.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=True))
        return 0 if bool(payload.get("ok", False)) else 124


if __name__ == "__main__":
    raise SystemExit(main())
