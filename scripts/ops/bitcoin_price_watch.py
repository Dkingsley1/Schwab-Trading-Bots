"""Three bounded Bitcoin observers. No credentials, orders, or capital allocation."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.storage_router import inspect_storage_path
from scripts.ops.crypto_research import PROFILES, fetch_candles, validate_candles
from scripts.ops.long_runtime_common import write_payload

OUT = "governance/health/bitcoin_price_watch_latest.json"
LOCK = "governance/locks/bitcoin_price_watch.lock"
BOTS = (("btc_intraday_breakout_watch", "day"),
        ("btc_intraday_pullback_watch", "day"),
        ("btc_swing_trend_watch", "swing"))


def summarize(rows, profile, now):
    import pandas as pd

    step = PROFILES[profile]["granularity"]
    end = int(now.timestamp()) // step * step
    start = end - PROFILES[profile]["bars"] * step
    rows = validate_candles(rows, start=start, end=end, step=step)
    close = pd.Series([row[4] for row in rows], dtype=float)
    ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
    last = float(close.iloc[-1])
    mean = float(close.iloc[-21:-1].mean())
    std = float(close.iloc[-21:-1].std(ddof=0))
    return {
        "state": "observed", "timeframe_seconds": step,
        "source_timestamp_utc": datetime.fromtimestamp(end, timezone.utc).isoformat(),
        "source": "coinbase_public_closed_candles", "closed_bars": len(rows),
        "bars_sha256": hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest(),
        "last_closed_price": last, "ema20": ema20, "ema50": ema50,
        "trend": "up" if last > ema20 > ema50 else "down" if last < ema20 < ema50 else "mixed",
        "above_prior_12_bar_high": last > max(row[2] for row in rows[-13:-1]),
        "prior_20_bar_zscore": (last - mean) / std if std > 0 else None,
        "return_12_bars_bps": (last / float(close.iloc[-13]) - 1) * 10000,
        "executable_quote": False,
    }


def build(*, now=None, fetcher=fetch_candles):
    now = now or datetime.now(timezone.utc)
    profiles = {}
    for name in PROFILES:
        try:
            profiles[name] = summarize(fetcher(name, now=now), name, now)
        except Exception as exc:
            # Never reuse an old signal or expose provider response bodies as errors.
            profiles[name] = {"state": "unavailable", "error_type": type(exc).__name__,
                              "source_timestamp_utc": None}
    bots = []
    for bot_id, name in BOTS:
        data = profiles[name]
        observation = "unavailable"
        if data["state"] == "observed":
            if "breakout" in bot_id:
                observation = "breakout_observed" if data["above_prior_12_bar_high"] else "no_breakout"
            elif "pullback" in bot_id:
                z = data["prior_20_bar_zscore"]
                observation = "lower_range_observed" if z is not None and z <= -2 else "no_lower_range_extreme"
            else:
                observation = data["trend"]
        bots.append({"bot_id": bot_id, "profile": name, "observation": observation,
                     "mode": "observe_only", "source_timestamp_utc": data["source_timestamp_utc"],
                     "order_requested": False})
    ready = all(data["state"] == "observed" for data in profiles.values())
    return {
        "schema_version": 1, "timestamp_utc": now.isoformat(),
        "ok": ready, "overall_status": "observing" if ready else "data_unavailable",
        "symbol": "BTC-USD", "cadence_seconds": 900, "profiles": profiles, "bots": bots,
        "cost_and_capital": {"available_cash_verified": False, "fee_tier_verified": False,
                             "round_trip_cost_verified": False, "capital_allocated_usd": 0},
        "live_blockers": ["available_cash_and_holdings_unverified", "account_fee_tier_unverified",
                          "spread_and_slippage_unverified", "net_of_cost_forward_evidence_missing",
                          "loss_limits_and_live_execution_not_authorized"],
        "live_execution_authority": False, "paper_execution_authority": False,
        "profitability_proven": False, "account_credentials_used": False,
        "history_policy": "one_bounded_latest_report_no_raw_candle_duplication",
    }


def local(root, relative):
    path = root / relative
    route = inspect_storage_path(path, boundary_root=root, allow_external=False)
    if route.get("status") not in {"present", "missing"} or route.get("symlinks"):
        raise ValueError("unsafe_monitor_path")
    return path


def run(root=ROOT):
    output, lock = local(root, OUT), local(root, LOCK)
    if local(root, "governance/health/SYSTEM_POWER_OFF.flag").exists():
        return {"ok": False, "reason": "system_power_off", "live_execution_authority": False}
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError("observer_lock_requires_regular_file")
    with os.fdopen(fd, "a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"ok": False, "reason": "observer_busy", "live_execution_authority": False}
        payload = build()
        write_payload(output, payload)
        return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = run()
    print(json.dumps(payload, indent=None if args.json else 2, allow_nan=False))
    return 0 if payload.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
