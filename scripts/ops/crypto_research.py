"""Bounded BTC historical research. Never writes training or execution evidence."""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

PROFILES = {
    "day": {"name": "Bitcoin Day", "granularity": 300, "bars": 288, "max_hold": 72},
    "swing": {
        "name": "Bitcoin Swing",
        "granularity": 21600,
        "bars": 280,
        "max_hold": 20,
    },
}
WARMUP = 50


def validate_request(raw: dict) -> dict:
    if not isinstance(raw, dict) or set(raw) != {
        "profile",
        "capital",
        "fee_bps",
        "spread_bps",
    }:
        raise ValueError("invalid_replay_request")
    if raw["profile"] not in PROFILES:
        raise ValueError("invalid_profile")
    result = {"profile": raw["profile"]}
    for key, low, high in (
        ("capital", 10, 100000),
        ("fee_bps", 0, 200),
        ("spread_bps", 1, 100),
    ):
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("invalid_replay_input")
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError("replay_input_out_of_range")
        result[key] = float(value)
    return result


def validate_candles(raw: list, *, start: int, end: int, step: int) -> list:
    if not isinstance(raw, list) or not 1 <= len(raw) <= 300:
        raise ValueError("invalid_candle_count")
    candles = []
    for row in raw:
        if not isinstance(row, list) or len(row) != 6:
            raise ValueError("invalid_candle_shape")
        if any(
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or not math.isfinite(x)
            for x in row
        ):
            raise ValueError("invalid_candle_value")
        ts, low, high, opening, close, volume = row
        if (
            ts != int(ts)
            or ts % step
            or low <= 0
            or not low <= min(opening, close) <= max(opening, close) <= high
            or volume < 0
        ):
            raise ValueError("invalid_candle_range")
        if start <= ts and ts + step <= end:
            candles.append(row)
    candles.sort(key=lambda row: row[0])
    if len(candles) < WARMUP + 10:
        raise ValueError("insufficient_closed_candles")
    if any(b[0] - a[0] != step for a, b in zip(candles, candles[1:])):
        raise ValueError("gapped_or_duplicate_candles")
    if candles[0][0] != start or candles[-1][0] + step != end:
        raise ValueError("incomplete_candle_window")
    return candles


def replay(candles: list, request: dict) -> dict:
    import pandas as pd
    from core.execution_simulator import simulate_execution

    request = validate_request(request)
    profile = PROFILES[request["profile"]]
    if len(candles) < WARMUP + 10:
        raise ValueError("insufficient_closed_candles")
    closes = pd.Series([row[4] for row in candles], dtype=float)
    ema20 = closes.ewm(span=20, adjust=False).mean()
    ema50 = closes.ewm(span=50, adjust=False).mean()
    prior_high = pd.Series([row[2] for row in candles]).rolling(12).max().shift(1)
    # The shared cost model expects one-minute volatility, not a 5m/6h return.
    vol = closes.pct_change().rolling(20).std().fillna(0.0) / math.sqrt(
        profile["granularity"] / 60
    )
    cash = initial = request["capital"]
    btc = 0.0
    cost_basis = 0.0
    entry_bar = 0
    fee_rate = request["fee_bps"] / 10000
    fills, curve, outcomes = [], [], []
    fees = friction = 0.0

    def execute(action: str, i: int, reference: float, reason: str) -> None:
        nonlocal cash, btc, cost_basis, entry_bar, fees, friction
        # The shared model embeds its fee in price. Remove that fee component,
        # then charge the explicitly chosen research fee exactly once.
        sim = simulate_execution(
            action=action,
            last_price=reference,
            return_1m=0,
            spread_bps=request["spread_bps"],
            volatility_1m=float(vol.iloc[i - 1]),
            broker="coinbase",
            market_kind="crypto",
            symbol="BTC-USD",
            session="24x7",
            asset_class="crypto",
            order_size=max(btc, cash / reference),
        )
        direction = 1 if action == "BUY" else -1
        price = (
            sim.expected_fill_price
            - direction
            * sim.touch_price
            * sim.fee_bps
            * sim.symbol_curve_multiplier
            / 10000
        )
        if price <= 0 or not math.isfinite(price):
            raise ValueError("invalid_execution_model_price")
        if action == "BUY":
            qty = cash / (price * (1 + fee_rate))
            fee = qty * price * fee_rate
            cost_basis = cash
            btc, cash, entry_bar = qty, 0.0, i
        else:
            qty = btc
            fee = qty * price * fee_rate
            cash += qty * price - fee
            outcomes.append(cash - cost_basis)
            btc = 0.0
        fees += fee
        friction += qty * abs(price - reference)
        fills.append(
            {
                "time": candles[i][0],
                "action": action,
                "quantity": qty,
                "reference_price": reference,
                "price": price,
                "fee": fee,
                "friction": qty * abs(price - reference),
                "reason": reason,
            }
        )

    # Decisions use only completed bar i-1; executions use the next bar open.
    # Candle-only research assumes full fills, never measured order-book liquidity.
    for i in range(WARMUP, len(candles)):
        prev = i - 1
        up = closes.iloc[prev] > ema20.iloc[prev] > ema50.iloc[prev]
        enter = up and (
            request["profile"] == "swing" or closes.iloc[prev] > prior_high.iloc[prev]
        )
        exit_signal = closes.iloc[prev] < ema20.iloc[prev]
        age = i - entry_bar
        timed_out = age >= profile["max_hold"]
        if btc > 0 and (exit_signal or timed_out):
            execute(
                "SELL", i, candles[i][3], "Time limit" if timed_out else "Trend exit"
            )
        elif btc == 0 and enter:
            execute(
                "BUY",
                i,
                candles[i][3],
                "Trend breakout" if request["profile"] == "day" else "EMA trend",
            )
        curve.append(
            {
                "time": candles[i][0] + profile["granularity"],
                "equity": cash + btc * candles[i][4],
            }
        )
    if btc > 0:
        execute("SELL", len(candles) - 1, candles[-1][4], "End-of-window liquidation")
        fills[-1]["time"] += profile["granularity"]
        curve[-1]["equity"] = cash
    peak = initial
    drawdown = 0.0
    for point in curve:
        peak = max(peak, point["equity"])
        drawdown = max(drawdown, (peak - point["equity"]) / peak)

    # Compare the exact same evaluation window, charging both sides the same
    # configured fees and shared model costs, not a frictionless benchmark.
    def benchmark_price(action: str, reference: float, i: int) -> float:
        sim = simulate_execution(
            action=action,
            last_price=reference,
            return_1m=0,
            spread_bps=request["spread_bps"],
            volatility_1m=float(vol.iloc[i - 1]),
            broker="coinbase",
            market_kind="crypto",
            symbol="BTC-USD",
            session="24x7",
            asset_class="crypto",
            order_size=initial / reference,
        )
        direction = 1 if action == "BUY" else -1
        return (
            sim.expected_fill_price
            - direction
            * sim.touch_price
            * sim.fee_bps
            * sim.symbol_curve_multiplier
            / 10000
        )

    entry = benchmark_price("BUY", candles[WARMUP][3], WARMUP)
    units = initial / (entry * (1 + fee_rate))
    benchmark_curve = [
        {"time": p["time"], "equity": units * candles[WARMUP + j][4]}
        for j, p in enumerate(curve)
    ]
    benchmark_end = (
        units
        * benchmark_price("SELL", candles[-1][4], len(candles) - 1)
        * (1 - fee_rate)
    )
    benchmark_curve[-1]["equity"] = benchmark_end
    return {
        "schema_version": 1,
        "research_contract": "btc_candle_research_v1",
        "granularity_seconds": profile["granularity"],
        "kind": "historical_research_only",
        "symbol": "BTC-USD",
        "profile": request["profile"],
        "parameters": request,
        "model_trained": False,
        "live_execution_allowed": False,
        "forward_paper_authority": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_sha256": hashlib.sha256(
            json.dumps(candles, separators=(",", ":")).encode()
        ).hexdigest(),
        "candles": len(candles),
        "warmup_bars": WARMUP,
        "start": candles[WARMUP][0],
        "end": candles[-1][0] + profile["granularity"],
        "initial": initial,
        "ending": cash,
        "net_pnl": cash - initial,
        "return_pct": (cash / initial - 1) * 100,
        "max_drawdown_pct": drawdown * 100,
        "fees": fees,
        "friction": friction,
        "round_trips": len(outcomes),
        "win_rate_pct": (
            100 * sum(x > 0 for x in outcomes) / len(outcomes) if outcomes else None
        ),
        "benchmark_return_pct": (benchmark_end / initial - 1) * 100,
        "curve": curve,
        "benchmark_curve": benchmark_curve,
        "fills": fills,
        "assumptions": [
            "Rule-based baseline; not a trained model",
            "Long-only spot; no leverage",
            "Closed-bar signals; next-open fills",
            "Full fills assumed; no observed book depth",
            "Fee input is an assumption, not your Coinbase fee tier",
            "Spread and other execution friction are modeled",
            "No training, soak or promotion credit",
        ],
    }


def fetch_and_replay(request: dict) -> dict:
    request = validate_request(request)
    profile = PROFILES[request["profile"]]
    step = profile["granularity"]
    end = int(datetime.now(timezone.utc).timestamp()) // step * step
    start = end - profile["bars"] * step
    # A narrower public-data transport than the streaming market client:
    # fixed host/path, no credentials, proxies or redirects, and bounded bytes.
    client = http.client.HTTPSConnection("api.exchange.coinbase.com", timeout=6)
    try:
        query = urlencode(
            {
                "start": datetime.fromtimestamp(start, timezone.utc).isoformat(),
                "end": datetime.fromtimestamp(end, timezone.utc).isoformat(),
                "granularity": step,
            }
        )
        client.request(
            "GET",
            "/products/BTC-USD/candles?" + query,
            headers={"User-Agent": "SchwabPlatform-CryptoResearch/1"},
        )
        response = client.getresponse()
        if response.status != 200:
            raise ValueError("public_market_data_unavailable")
        raw = response.read(512 * 1024 + 1)
        if len(raw) > 512 * 1024:
            raise ValueError("market_response_too_large")
        rows = json.loads(raw)
    finally:
        client.close()
    return replay(validate_candles(rows, start=start, end=end, step=step), request)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(sys.argv[1])))
    os.nice(10)
    try:
        request = json.loads(sys.stdin.read(2048))
        result = fetch_and_replay(request)
    except Exception as exc:
        known = (
            str(exc)
            if isinstance(exc, ValueError)
            else "public_market_data_or_replay_unavailable"
        )
        result = {
            "error": (
                known
                if known.replace("_", "").isalnum() and len(known) < 80
                else "replay_failed"
            )
        }
    print(json.dumps(result, allow_nan=False))
