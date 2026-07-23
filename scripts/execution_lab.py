#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.execution_simulator import simulate_execution


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "execution_lab_latest.json"


def build_payload() -> dict:
    scenarios = [
        {"venue": "schwab_equities", "broker": "schwab", "market_kind": "equities", "session": "open", "symbol": "AAPL"},
        {"venue": "schwab_equities", "broker": "schwab", "market_kind": "equities", "session": "regular", "symbol": "SPY"},
        {"venue": "coinbase_crypto", "broker": "coinbase", "market_kind": "crypto", "session": "regular", "symbol": "BTC-USD"},
        {"venue": "coinbase_crypto", "broker": "coinbase", "market_kind": "crypto", "session": "overnight_gap", "symbol": "ETH-USD"},
        {"venue": "schwab_options", "broker": "schwab", "market_kind": "options", "asset_class": "options", "session": "regular", "symbol": "NVDA_covered_call"},
    ]
    rows = []
    for scenario in scenarios:
        for order_type in ("market", "limit"):
            result = simulate_execution(
                action="BUY",
                last_price=100.0,
                return_1m=0.002,
                spread_bps=18.0 if scenario["market_kind"] == "crypto" else 6.0,
                volatility_1m=0.012 if scenario["market_kind"] == "crypto" else 0.006,
                latency_ms=260.0 if scenario["session"] == "open" else 120.0,
                bid_size=120.0,
                ask_size=90.0,
                order_size=65.0,
                broker=scenario["broker"],
                market_kind=scenario["market_kind"],
                symbol=scenario["symbol"],
                session=scenario["session"],
                order_type=order_type,
                live_fill_slippage_bps=1.5,
                asset_class=scenario.get("asset_class", ""),
                sleeve="covered_call" if scenario.get("asset_class") == "options" else scenario["market_kind"],
                quote_age_ms=3200.0 if scenario.get("asset_class") == "options" and order_type == "limit" else 250.0,
                market_volume=35.0 if scenario.get("asset_class") == "options" else 25000.0,
                avg_daily_volume=100.0 if scenario.get("asset_class") == "options" else 1000000.0,
                open_interest=20.0 if scenario.get("asset_class") == "options" else 0.0,
            )
            rows.append(
                {
                    "venue": scenario["venue"],
                    "session": scenario["session"],
                    "order_type": order_type,
                    "symbol": scenario["symbol"],
                    "slippage_bps": round(result.slippage_bps, 6),
                    "effective_fill_ratio": round(result.effective_fill_ratio, 6),
                    "paper_execution_status": result.paper_execution_status,
                    "paper_execution_score": round(result.paper_execution_score, 6),
                    "cancel_probability": round(result.cancel_probability, 6),
                    "requote_probability": round(result.requote_probability, 6),
                    "reject_probability": round(result.reject_probability, 6),
                    "stale_quote_probability": round(result.stale_quote_probability, 6),
                    "queue_priority_score": round(result.queue_priority_score, 6),
                    "queue_fill_probability": round(result.queue_fill_probability, 6),
                    "session_penalty_bps": round(result.session_penalty_bps, 6),
                    "crowding_penalty_bps": round(result.crowding_penalty_bps, 6),
                    "market_impact_bps": round(result.market_impact_bps, 6),
                    "option_liquidity_penalty_bps": round(result.option_liquidity_penalty_bps, 6),
                    "latency_bucket": result.latency_bucket,
                    "spread_regime": result.spread_regime,
                    "asset_class": result.asset_class,
                }
            )
    worst = sorted(rows, key=lambda row: float(row.get("slippage_bps", 0.0) or 0.0), reverse=True)[:6]
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": True,
        "scenario_count": len(rows),
        "top_worst_case_scenarios": worst,
        "capabilities": {
            "venue_session_specific_latency": True,
            "fee_spread_slippage_haircut": True,
            "partial_fill_modeling": True,
            "queue_priority_modeling": True,
            "market_impact_modeling": True,
            "requote_probability": True,
            "reject_cancel_stale_quote_modeling": True,
            "realistic_option_fills": True,
            "execution_quality_scoring": True,
            "sleeve_specific_friction": True,
            "live_shadow_calibration_inputs": True,
            "short_borrow_stress": True,
            "spread_and_crowding_calibration": True,
        },
        "rows": rows,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a research-grade execution-lab sweep across venue/session scenarios.")
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload()
    out_path = Path(args.out_file).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "execution_lab "
            f"scenario_count={int(payload.get('scenario_count', 0) or 0)} "
            f"worst_slippage_bps={float((payload.get('top_worst_case_scenarios') or [{}])[0].get('slippage_bps', 0.0) or 0.0):.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
