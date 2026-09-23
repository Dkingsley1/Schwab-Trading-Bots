"""Point-in-time price diagnostics, separate from a bot's recorded rationale."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from statistics import mean
from zoneinfo import ZoneInfo


def timestamp(value):
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timezone_required")
    return parsed.astimezone(timezone.utc)


def number(value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("finite_number_required")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError("finite_positive_number_required")
    return result


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def session_bounds(moment):
    return _bounds_for_day(
        moment.astimezone(ZoneInfo("America/New_York")).date().isoformat()
    )


@lru_cache(maxsize=8192)
def _bounds_for_day(day_text):
    import exchange_calendars
    import pandas as pd

    calendar = exchange_calendars.get_calendar("XNYS")
    day = pd.Timestamp(day_text)
    if not calendar.is_session(day):
        return None
    return (
        calendar.session_open(day).to_pydatetime(),
        calendar.session_close(day).to_pydatetime(),
    )


def latest_closed_end(asof, minutes):
    # Anchor intervals to the exchange open, including DST and early closes.
    for offset in range(12):
        bounds = session_bounds(asof - timedelta(days=offset))
        if bounds is None:
            continue
        opened, closed = bounds
        through = min(asof, closed)
        if minutes is None:
            if asof >= closed:
                return closed
        elif through >= opened + timedelta(minutes=minutes):
            count = int((through - opened).total_seconds() // (minutes * 60))
            return opened + timedelta(minutes=count * minutes)
    raise ValueError("recent_exchange_session_unavailable")


def validate_bars(rows, *, minutes, asof):
    if not isinstance(rows, list) or len(rows) > 6000:
        raise ValueError("candle_input_unbounded")
    result, previous = [], None
    for row in rows:
        start, end = timestamp(row["start_utc"]), timestamp(row["end_utc"])
        bounds = session_bounds(start)
        if bounds is None:
            raise ValueError("candle_outside_exchange_session")
        opened, closed = bounds
        if minutes is None:
            valid_interval = start == opened and end == closed
        else:
            step = minutes * 60
            valid_interval = (
                opened <= start < end <= closed
                and (end - start).total_seconds() == step
                and (start - opened).total_seconds() % step == 0
            )
        if not valid_interval or end > asof or (previous and start < previous):
            raise ValueError("candle_unclosed_overlapping_or_misaligned")
        values = {
            key: number(row[key], positive=True)
            for key in ("open", "high", "low", "close")
        }
        volume = number(row["volume"])
        if volume < 0 or not (
            values["low"]
            <= min(values["open"], values["close"])
            <= max(values["open"], values["close"])
            <= values["high"]
        ):
            raise ValueError("invalid_ohlcv")
        result.append(
            dict(
                values,
                volume=volume,
                start_utc=start.isoformat(),
                end_utc=end.isoformat(),
            )
        )
        previous = end
    return result


def aggregate_bars(rows, minutes):
    buckets = {}
    for row in rows:
        start = timestamp(row["start_utc"])
        opened, _ = session_bounds(start)
        bucket = opened + timedelta(
            minutes=int((start - opened).total_seconds() // (minutes * 60)) * minutes
        )
        buckets.setdefault(bucket, []).append(row)
    result, incomplete = [], 0
    for start, group in sorted(buckets.items()):
        expected = [start + timedelta(minutes=5 * i) for i in range(minutes // 5)]
        if [timestamp(row["start_utc"]) for row in group] != expected:
            incomplete += 1
            continue
        result.append(
            {
                "start_utc": start.isoformat(),
                "end_utc": (start + timedelta(minutes=minutes)).isoformat(),
                "open": group[0]["open"],
                "high": max(r["high"] for r in group),
                "low": min(r["low"] for r in group),
                "close": group[-1]["close"],
                "volume": sum(r["volume"] for r in group),
            }
        )
    return result, incomplete


def describe_candle(row):
    o, h, low, c = (row[k] for k in ("open", "high", "low", "close"))
    span = h - low
    return dict(
        row,
        direction="up" if c > o else "down" if c < o else "flat",
        body_usd=abs(c - o),
        range_usd=span,
        upper_wick_usd=h - max(o, c),
        lower_wick_usd=min(o, c) - low,
        body_fraction=abs(c - o) / span if span else 0,
        close_position_in_candle=(c - low) / span if span else None,
        open_to_close_bps=(c / o - 1) * 10000,
    )


def metrics(rows, *, asof, minutes, incomplete=0):
    if not rows:
        return {"status": "missing", "closed_bars": 0, "issues": ["missing_candles"]}
    closes = [r["close"] for r in rows]
    last, issues = closes[-1], []
    expected = latest_closed_end(asof, minutes)
    end = timestamp(rows[-1]["end_utc"])
    if end != expected:
        issues.append("latest_closed_candle_missing")
    if len(rows) < 21:
        issues.append("fewer_than_21_closed_bars")
    # Each adjacent bar must be the next complete exchange interval. Partial
    # end-of-session hourly buckets are deliberately not counted as full hours.
    gaps = sum(
        latest_closed_end(timestamp(b["end_utc"]) - timedelta(microseconds=1), minutes)
        != timestamp(a["end_utc"])
        for a, b in zip(rows, rows[1:])
    )
    if gaps:
        issues.append("candle_gaps")
    prior = rows[-21:-1]
    high = max(r["high"] for r in prior) if prior else None
    low = min(r["low"] for r in prior) if prior else None
    sma20 = mean(closes[-20:]) if len(closes) >= 20 else None
    sma50 = mean(closes[-50:]) if len(closes) >= 50 else None
    changes = [b - a for a, b in zip(closes, closes[1:])][-14:]
    rsi = atr = None
    if len(changes) == 14:
        gain = mean(max(v, 0) for v in changes)
        loss = mean(max(-v, 0) for v in changes)
        rsi = 100 * gain / (gain + loss) if gain + loss else 50
        atr = mean(
            max(
                b["high"] - b["low"],
                abs(b["high"] - a["close"]),
                abs(b["low"] - a["close"]),
            )
            for a, b in list(zip(rows, rows[1:]))[-14:]
        )
    avg_volume = mean(r["volume"] for r in prior) if prior else 0
    return {
        "status": "complete" if not issues else "incomplete",
        "issues": issues,
        "closed_bars": len(rows),
        "bars_sha256": digest(rows),
        "gap_count": gaps,
        "incomplete_buckets_excluded": incomplete,
        "latest_end_utc": end.isoformat(),
        "expected_latest_end_utc": expected.isoformat(),
        "age_seconds": (asof - end).total_seconds(),
        "recent_candles": [describe_candle(r) for r in rows[-5:]],
        "chart_candles": rows[-60:],
        "close": last,
        "sma20": sma20,
        "sma50": sma50,
        "distance_from_sma20_bps": (last / sma20 - 1) * 10000 if sma20 else None,
        "distance_from_sma50_bps": (last / sma50 - 1) * 10000 if sma50 else None,
        "trend": (
            "above_sma20"
            if sma20 and last > sma20
            else "below_sma20" if sma20 and last < sma20 else "flat_or_unavailable"
        ),
        "rsi14_simple": rsi,
        "atr14_simple_usd": atr,
        "return_3_bars_bps": (
            (last / closes[-4] - 1) * 10000 if len(closes) >= 4 else None
        ),
        "prior_20_bar_low": low,
        "prior_20_bar_high": high,
        "close_position_in_prior_20_bar_range": (
            (last - low) / (high - low) if high is not None and high > low else None
        ),
        "distance_to_prior_low_bps": (last / low - 1) * 10000 if low else None,
        "distance_to_prior_high_bps": (last / high - 1) * 10000 if high else None,
        "above_prior_high": last > high if high else None,
        "below_prior_low": last < low if low else None,
        "volume_relative_to_prior_20_bars": (
            rows[-1]["volume"] / avg_volume if avg_volume else None
        ),
    }


def quote_evidence(quote, *, asof, maximum_age=30):
    if quote.get("symbol") != "SCHD" or quote.get("provider") != "schwab":
        raise ValueError("schd_schwab_quote_required")
    if (
        not quote.get("snapshot_id")
        or quote.get("source_quality_label") != "broker_native"
    ):
        raise ValueError("quote_identity_and_native_provenance_required")
    when = timestamp(quote["timestamp_utc"])
    age = (asof - when).total_seconds()
    if not 0 <= age <= maximum_age:
        raise ValueError("quote_stale_or_future")
    bid, ask, last = (number(quote[k], positive=True) for k in ("bid", "ask", "last"))
    if ask <= bid:
        raise ValueError("quote_locked_or_crossed")
    sizes = {k: number(quote[k], positive=True) for k in ("bid_size", "ask_size")}
    mid = (ask + bid) / 2
    return dict(
        sizes,
        symbol="SCHD",
        provider="schwab",
        source_quality_label="broker_native",
        timestamp_utc=when.isoformat(),
        snapshot_id=quote["snapshot_id"],
        age_seconds=age,
        bid=bid,
        ask=ask,
        last=last,
        mid=mid,
        spread_usd=ask - bid,
        spread_bps=(ask - bid) / mid * 10000,
        source_sha256=digest(quote),
        provenance_verification="input_declares_provider_not_independently_authenticated",
    )


def long_horizon_context(daily, *, asof):
    """Calendar periods and trailing calendar days are not interchangeable."""
    import exchange_calendars
    import pandas as pd

    calendar = exchange_calendars.get_calendar("XNYS")
    zone = ZoneInfo("America/New_York")
    latest = latest_closed_end(asof, None).astimezone(zone).date()
    by_day = {timestamp(r["start_utc"]).astimezone(zone).date(): r for r in daily}

    def period(start, finish):
        sessions = [
            day.date()
            for day in calendar.sessions_in_range(
                pd.Timestamp(start), pd.Timestamp(finish)
            )
        ]
        if not sessions or any(day not in by_day for day in sessions):
            return None
        rows = [by_day[day] for day in sessions]
        return describe_candle(
            {
                "start_utc": rows[0]["start_utc"],
                "end_utc": rows[-1]["end_utc"],
                "open": rows[0]["open"],
                "high": max(r["high"] for r in rows),
                "low": min(r["low"] for r in rows),
                "close": rows[-1]["close"],
                "volume": sum(r["volume"] for r in rows),
                "trading_sessions": len(rows),
            }
        )

    frames = {}
    for name, frequency in (("1M", "M"), ("1Y", "Y")):
        current = pd.Period(latest, freq=frequency)
        completed = []
        missing = 0
        for offset in range(24 if name == "1M" else 5):
            window = current - offset
            sessions = calendar.sessions_in_range(
                window.start_time, window.end_time.normalize()
            )
            end = calendar.session_close(sessions[-1]).to_pydatetime()
            if end > asof:
                continue
            candle = period(window.start_time.date(), window.end_time.date())
            if candle:
                completed.append(dict(candle, period=str(window)))
            else:
                missing += 1
        completed.reverse()
        latest_window = current
        sessions = calendar.sessions_in_range(
            current.start_time, current.end_time.normalize()
        )
        if calendar.session_close(sessions[-1]).to_pydatetime() > asof:
            latest_window -= 1
        complete = bool(completed) and completed[-1]["period"] == str(latest_window)
        frames[name] = {
            "status": "complete" if complete else "incomplete",
            "closed_bars": len(completed),
            "expected_latest_period": str(latest_window),
            "missing_history_periods": missing,
            "recent_candles": completed[-5:],
            "period_candles": completed,
            "issues": [] if complete else ["latest_complete_calendar_period_missing"],
            "definition": (
                "Complete calendar months"
                if name == "1M"
                else "Complete calendar years; current year excluded"
            ),
            "indicator_limit": "No monthly/yearly moving averages or RSI inferred from insufficient period history.",
        }
    start = latest - timedelta(days=179)
    window = period(start, latest)
    frames["180d"] = {
        "status": "complete" if window else "incomplete",
        "closed_bars": 1 if window else 0,
        "window_start_date": start.isoformat(),
        "window_end_date": latest.isoformat(),
        "definition": "Trailing 180 calendar days through the latest closed daily bar, not 180 trading sessions or a calendar half-year",
        "recent_candles": [window] if window else [],
        "issues": [] if window else ["daily_coverage_incomplete_for_180_calendar_days"],
        "overlapping_window_not_independent_sample": True,
    }
    return frames


def build_evidence(packet, *, now):
    """No generated reasons are attributed to the model; no trading side effects."""
    blockers, frames, quote = [], {}, {}
    native = packet.get("native_validation", {})
    if native:
        if not isinstance(native, dict) or not isinstance(native.get("blockers"), list):
            raise ValueError("native_validation_contract_invalid")
        blockers.extend(str(item) for item in native["blockers"])
    if packet.get("price_basis") != "split_adjusted_dividends_unadjusted":
        blockers.append("comparable_split_adjusted_price_basis_required")
    decision = packet.get("decision", {})
    asof = timestamp(decision["timestamp_utc"])
    if not 0 <= (now - asof).total_seconds() <= 120:
        blockers.append("decision_stale_or_future")
    bounds = session_bounds(asof)
    if bounds is None or not bounds[0] <= asof < bounds[1]:
        blockers.append("regular_session_required_for_market_rehearsal")
    if decision.get("symbol") != "SCHD" or packet.get("symbol") != "SCHD":
        blockers.append("schd_only")
    metadata = decision.get("metadata", {})
    if not all(
        (
            decision.get("decision_id"),
            decision.get("strategy"),
            metadata.get("snapshot_id"),
            packet.get("candidate_id"),
        )
    ):
        blockers.append("decision_identity_incomplete")
    reasons = decision.get("reasons", [])
    if (
        not isinstance(reasons, list)
        or not reasons
        or not all(isinstance(r, str) and r.strip() for r in reasons)
    ):
        blockers.append("recorded_bot_reasons_missing")
    gates = decision.get("gates", {})
    if (
        not isinstance(gates, dict)
        or not gates
        or not all(value is True for value in gates.values())
    ):
        blockers.append("bot_gates_not_all_affirmative")
    if decision.get("decision") != "EXECUTE" or decision.get("action") not in {
        "BUY",
        "SELL",
    }:
        blockers.append("bot_did_not_request_execution")
    try:
        quote = quote_evidence(packet.get("quote", {}), asof=asof)
        if quote["snapshot_id"] != metadata.get("snapshot_id"):
            blockers.append("decision_quote_snapshot_mismatch")
        if quote["spread_bps"] > 25:
            blockers.append("spread_above_25_bps")
    except (KeyError, TypeError, ValueError) as exc:
        blockers.append(f"quote_unusable:{exc}")
    candles = packet.get("candles", {})
    for name, minutes in (("5m", 5), ("1d", None), ("1m", 1)):
        try:
            rows = validate_bars(candles.get(name, []), minutes=minutes, asof=asof)
            frames[name] = metrics(rows, asof=asof, minutes=minutes)
            if name == "1d":
                frames.update(long_horizon_context(rows, asof=asof))
            if name == "5m":
                for derived, size in (("15m", 15), ("1h", 60)):
                    aggregated, incomplete = aggregate_bars(rows, size)
                    frames[derived] = metrics(
                        aggregated, asof=asof, minutes=size, incomplete=incomplete
                    )
                current = [
                    r for r in rows if bounds and timestamp(r["start_utc"]) >= bounds[0]
                ]
                volume = sum(r["volume"] for r in current)
                frames[name]["session_closed_bar_vwap_proxy"] = (
                    sum(
                        (r["high"] + r["low"] + r["close"]) / 3 * r["volume"]
                        for r in current
                    )
                    / volume
                    if volume
                    else None
                )
        except (KeyError, TypeError, ValueError) as exc:
            frames[name] = {"status": "invalid", "issues": [str(exc)]}
    for name in ("5m", "15m", "1h", "1d", "1M", "180d", "1Y"):
        if frames.get(name, {}).get("status") != "complete":
            blockers.append(f"{name}_evidence_incomplete")
    action = decision.get("action", "WAIT")
    supporting, opposing = [], []
    for name, frame in frames.items():
        recent = frame.get("recent_candles", [])
        if recent and quote:
            candle = recent[-1]
            price, low, high = quote["last"], candle["low"], candle["high"]
            frame["current_quote_position"] = {
                "last_price": price,
                "versus_last_closed_candle_close_bps": (price / candle["close"] - 1)
                * 10000,
                "within_last_closed_candle_range": (
                    (price - low) / (high - low) if high > low else None
                ),
                "above_last_closed_high": price > high,
                "below_last_closed_low": price < low,
                "interpretation": "Outside [0,1] means price is outside that historical candle range; not a probability.",
            }
        if frame.get("status") != "complete" or "trend" not in frame:
            continue
        direction = frame["trend"]
        fact = {
            "timeframe": name,
            "fact": direction,
            "close": frame["close"],
            "sma20": frame["sma20"],
        }
        agrees = (action == "BUY" and direction == "above_sma20") or (
            action == "SELL" and direction == "below_sma20"
        )
        (supporting if agrees else opposing).append(fact)
        if (
            action == "BUY"
            and frame.get("rsi14_simple", 0) is not None
            and frame["rsi14_simple"] >= 70
        ):
            opposing.append(
                {
                    "timeframe": name,
                    "fact": "simple_RSI_at_least_70",
                    "value": frame["rsi14_simple"],
                    "interpretation": "Elevated recent gains; does not predict reversal.",
                }
            )
        if (
            action == "SELL"
            and frame.get("rsi14_simple") is not None
            and frame["rsi14_simple"] <= 30
        ):
            opposing.append(
                {
                    "timeframe": name,
                    "fact": "simple_RSI_at_most_30",
                    "value": frame["rsi14_simple"],
                    "interpretation": "Elevated recent losses; does not predict rebound.",
                }
            )
    return {
        "schema_version": 1,
        "symbol": "SCHD",
        "generated_at_utc": now.isoformat(),
        "as_of_utc": asof.isoformat(),
        "input_sha256": digest(packet),
        "candidate_id": packet.get("candidate_id"),
        "evidence_kind": packet.get("evidence_kind"),
        "native_validation": native,
        "decision_status": "WAIT" if blockers else f"SIMULATE_{action}",
        "blockers": blockers,
        "bot_record": {
            k: decision.get(k)
            for k in (
                "decision_id",
                "strategy",
                "action",
                "decision",
                "model_score",
                "threshold",
                "reasons",
                "gates",
                "features",
                "feature_compaction_contract",
            )
        },
        "snapshot_id": metadata.get("snapshot_id"),
        "quote": quote,
        "timeframes": frames,
        "bot_declared_invalidation": metadata.get(
            "invalidation_conditions", "not_recorded"
        ),
        "bot_declared_time_horizon": metadata.get("decision_horizon", "not_recorded"),
        "corporate_action_context": {
            "price_basis": packet.get("price_basis", "unknown"),
            "source_declared_events": packet.get("corporate_actions", []),
            "independently_verified": False,
            "warning": "Ex-dividend gaps can reflect a distribution rather than selling pressure. Buying on or after the ordinary cash ex-date does not earn that distribution. Raw price returns are not total returns; split/dividend adjustment must be checked before interpreting long-history levels.",
        },
        "context_supporting_direction": supporting,
        "context_opposing_or_neutral": opposing,
        "attribution": "Chart diagnostics are context, not proof the bot used them. Only bot_record contains its recorded rationale; model_score is not a calibrated win probability.",
        "invalidation": [
            "Any failed bot gate",
            "Missing, future, stale or gapped required source",
            "Spread above 25 bps",
            "Candidate or decision identity changes",
            "No later independent quote before the 60-second simulated order deadline",
        ],
        "unverified_context": [
            "News and macro event risk",
            "Dividend/ex-date and total return",
            "ETF valuation and holdings overlap",
            "Account cash, restrictions, settlement and tax",
            "Out-of-sample profitability and model calibration",
            "Authenticity of imported provider declarations",
        ],
        "definitions": {
            "candles": "Closed regular-session XNYS bars only; no fabricated 1m bars or partial hourly buckets.",
            "levels": "Prior 20 completed bars excluding the current bar; descriptive extremes, not validated support/resistance.",
            "rsi_atr": "14-period simple-average RSI and true range, not Wilder smoothing.",
            "volume": "Last volume / prior-20 mean, not a time-of-day adjusted volume score.",
            "vwap": "Current-session closed-5m typical-price volume-weighted proxy, not tick VWAP.",
        },
        "live_execution_authority": False,
        "profitability_proven": False,
    }
