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


def build(*, now=None, fetcher=fetch_candles, capture_sink=None):
    now = now or datetime.now(timezone.utc)
    profiles = {}
    capture = {"source": {"provider": "coinbase", "symbol": "BTC-USD",
                           "fetch_started_at_utc": now.isoformat(), "requests": []},
               "candles": {}}
    for name in PROFILES:
        try:
            raw = fetcher(name, now=now)
            profiles[name] = summarize(raw, name, now)
            step = PROFILES[name]["granularity"]
            end = int(now.timestamp()) // step * step
            rows = validate_candles(raw, start=end-PROFILES[name]["bars"]*step, end=end, step=step)
            frame = "5m" if step == 300 else "6h"
            capture["candles"][frame] = [dict(
                start_utc=datetime.fromtimestamp(r[0], timezone.utc).isoformat(),
                end_utc=datetime.fromtimestamp(r[0]+step, timezone.utc).isoformat(),
                low=r[1], high=r[2], open=r[3], close=r[4], volume=r[5]) for r in rows]
            from core.decision_price_evidence import digest
            capture["source"]["requests"].append(dict(timeframe=frame,
                endpoint="/products/BTC-USD/candles", payload_sha256=digest(raw), closed_bars=len(rows)))
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
    chart_context = {"state": "unavailable", "reason": "capture_sink_not_configured"}
    if capture_sink and capture["candles"]:
        try:
            capture["source"]["observed_at_utc"] = datetime.now(timezone.utc).isoformat()
            chart_context = capture_sink(capture)
        except (OSError, ValueError, KeyError, TypeError):
            chart_context = {"state": "unavailable", "reason": "bounded_capture_publication_failed"}
    for bot in bots:
        bot["candle_context"] = chart_context
        from core.decision_price_evidence import digest
        data = profiles[bot["profile"]]
        bot["decision"] = {
            "timestamp_utc": chart_context.get("observed_at_utc", now.isoformat()),
            "decision_id": "bitcoin-observation:" + digest({"bot": bot["bot_id"], "at": now.isoformat()}),
            "symbol": "BTC-USD", "action": "HOLD", "decision": "OBSERVE_ONLY",
            "strategy": bot["bot_id"], "quantity": 0,
            "reasons": [bot["observation"]], "gates": {"orders_authorized": False},
            "features": {k: data[k] for k in ("last_closed_price", "ema20", "ema50", "trend", "above_prior_12_bar_high", "prior_20_bar_zscore", "return_12_bars_bps") if k in data},
            "metadata": {"candle_context": chart_context},
        }
        if data["state"] == "observed":
            if "breakout" in bot["bot_id"]:
                rule = {"rule": "last close > prior 12-bar high", "result": data["above_prior_12_bar_high"]}
            elif "pullback" in bot["bot_id"]:
                rule = {"rule": "prior 20-bar z-score <= -2", "value": data["prior_20_bar_zscore"],
                        "threshold": -2, "result": bot["observation"] == "lower_range_observed"}
            else:
                rule = {"rule": "up: close > EMA20 > EMA50; down: close < EMA20 < EMA50; otherwise mixed",
                        "result": data["trend"]}
            bot["decision"]["metadata"]["indicator_reasoning"] = {
                "timeframe_seconds": data["timeframe_seconds"], "rules": [rule],
                "authority_reason": "Observation only; these conditions do not authorize an entry or exit.",
            }
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
        "chart_context": chart_context,
        "history_policy": "shared_bounded_hash_captures_not_per_bot_raw_duplication",
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
        from core.decision_candle_store import publish
        payload = build(capture_sink=lambda capture: publish(root, capture))
        from core.accountability import safe_append_channel_event
        from core.path_registry import decision_log_path
        day = datetime.fromisoformat(payload["timestamp_utc"]).strftime("%Y%m%d")
        accepted = 0
        for bot in payload["bots"]:
            accepted += bool(safe_append_channel_event(
                decision_log_path(str(root), "decisions/coinbase_observers", day=day),
                bot["decision"], project_root=str(root), source="bitcoin_price_watch",
                channel="decision", schema="decision",
            ))
        payload["decision_history"] = {"accepted_by_native_writer": accepted,
                                       "expected": len(payload["bots"]),
                                       "durability_certified": False}
        if accepted != len(payload["bots"]):
            payload.update(ok=False, overall_status="decision_history_write_incomplete")
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
