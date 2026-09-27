"""Schwab read-only OHLC capture and data-derived candlestick diagrams."""

from __future__ import annotations

import contextlib
from datetime import datetime, timedelta, timezone
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from core.decision_price_evidence import (
    build_evidence,
    digest,
    number,
    session_bounds,
    timestamp,
    validate_bars,
)

READ_ONLY_ENV = {
    "ALLOW_ORDER_EXECUTION": "0",
    "MARKET_DATA_ONLY": "1",
    "TOP_BOT_ENABLE_LIVE_EXECUTION": "0",
    "EXECUTION_LANE_LIVE_ENABLED": "0",
    "RUN_ALL_SLEEVES_WITH_LIVE_EXECUTOR": "0",
    "SCHWAB_AUTH_INTERACTIVE": "0",
    "TOP_BOT_PAPER_TRADING_ENABLED": "0",
    "PAPER_BROKER_BRIDGE_ENABLED": "0",
}

FETCH_FAILURE_REASONS = {
    "schwab_provider_cooldown_active",
    "system_power_off",
    "unsafe_rehearsal_path",
    "unexpected_price_history_symbol",
    "empty_or_unbounded_schwab_history",
    "non_session_provider_candle",
    "candle_unclosed_overlapping_or_misaligned",
    "invalid_ohlcv",
    "finite_number_required",
    "finite_positive_number_required",
    "native_realtime_schd_quote_required",
}


def normalize_schwab_candles(payload, *, minutes, asof, symbol="SCHD"):
    if not isinstance(payload, dict) or payload.get("symbol") != symbol:
        raise ValueError("unexpected_price_history_symbol")
    raw = payload.get("candles")
    if not isinstance(raw, list) or not raw or len(raw) > 6000:
        raise ValueError("empty_or_unbounded_schwab_history")
    result, excluded = [], 0
    for candle in raw:
        moment = datetime.fromtimestamp(
            number(candle["datetime"], positive=True) / 1000, timezone.utc
        )
        bounds = session_bounds(moment)
        if bounds is None:
            raise ValueError("non_session_provider_candle")
        opened, closed = bounds
        start = opened if minutes is None else moment
        end = closed if minutes is None else start + timedelta(minutes=minutes)
        if end > asof:
            excluded += 1
            continue
        row = {k: candle[k] for k in ("open", "high", "low", "close", "volume")}
        result.append(dict(row, start_utc=start.isoformat(), end_utc=end.isoformat()))
    return validate_bars(result, minutes=minutes, asof=asof), excluded


def normalize_schwab_quote(payload):
    node = payload.get("SCHD", {})
    quote = node.get("quote", {})
    if node.get("symbol") != "SCHD" or node.get("realtime") is not True:
        raise ValueError("native_realtime_schd_quote_required")
    milliseconds = number(quote["quoteTime"], positive=True)
    stamp = datetime.fromtimestamp(milliseconds / 1000, timezone.utc)
    return {
        "symbol": "SCHD",
        "provider": "schwab",
        "source_quality_label": "broker_native",
        "timestamp_utc": stamp.isoformat(),
        "timestamp_basis": "schwab_quoteTime",
        "snapshot_id": "schwab-quote:" + digest(payload),
        **{
            target: number(quote[key], positive=True)
            for target, key in (
                ("last", "lastPrice"),
                ("bid", "bidPrice"),
                ("ask", "askPrice"),
                ("bid_size", "bidSize"),
                ("ask_size", "askSize"),
            )
        },
    }


def fetch_with_client(client, *, now, include_quote=False, symbol="SCHD"):
    from core.decision_candle_store import identity
    identity("schwab", symbol)
    if include_quote and symbol != "SCHD":
        raise ValueError("generic_capture_is_candles_only")
    result = {
        "candles": {},
        "source": {
            "provider": "schwab",
            "symbol": symbol,
            "fetch_started_at_utc": now.isoformat(),
            "daily_timestamp_mapping": "Provider epoch milliseconds mapped to America/New_York trading date, then XNYS open/close",
            "price_adjustment_basis": "provider_as_returned_not_independently_verified",
            "requests": [],
        },
    }
    calls = (
        (
            "1d",
            None,
            client.get_price_history_every_day,
            datetime(now.year - 2, 1, 1, tzinfo=timezone.utc),
        ),
        ("5m", 5, client.get_price_history_every_five_minutes, now - timedelta(days=9)),
        ("1m", 1, client.get_price_history_every_minute, now - timedelta(days=2)),
    )
    for name, minutes, method, start in calls:
        response = method(
            symbol,
            start_datetime=start,
            end_datetime=now,
            need_extended_hours_data=False,
            need_previous_close=True,
        )
        response.raise_for_status()
        payload = response.json()
        bars, excluded = normalize_schwab_candles(payload, minutes=minutes, asof=now, symbol=symbol)
        result["candles"][name] = bars
        result["source"]["requests"].append(
            {
                "timeframe": name,
                "endpoint": method.__name__,
                "payload_sha256": digest(payload),
                "closed_bars": len(bars),
                "unclosed_bars_excluded": excluded,
            }
        )
    if include_quote:
        response = client.get_quote("SCHD")
        response.raise_for_status()
        payload = response.json()
        result["quote"] = normalize_schwab_quote(payload)
        result["source"]["requests"].append(
            {"endpoint": "get_quote", "payload_sha256": digest(payload)}
        )
    return result


def fetch_child(root, *, include_quote=False, symbol="SCHD"):
    # This child exposes market-data GETs only. Never call an account/order API.
    os.environ.update(READ_ONLY_ENV)
    from core.provider_access_guard import provider_access_status, provider_request_slot
    from scripts.brokers.schwab.common import build_schwab_trader

    if provider_access_status(root, "schwab").get("active"):
        raise ValueError("schwab_provider_cooldown_active")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        trader = build_schwab_trader(root, mode="shadow")
        client = trader.authenticate()
        with provider_request_slot(
            root, "schwab", symbol, slot_count=2, wait_seconds=10
        ):
            return fetch_with_client(
                client, now=datetime.now(timezone.utc), include_quote=include_quote, symbol=symbol
            )


def fetch_bounded(root, *, include_quote=False, timeout_seconds=90):
    child = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/ops/schd_decision_rehearsal.py"),
            "--fetch-child",
            *(["--with-quote"] if include_quote else []),
        ],
        cwd=root,
        env=dict(os.environ, **READ_ONLY_ENV),
        capture_output=True,
        text=True,
        timeout=max(1, min(timeout_seconds, 90)),
        check=False,
    )
    if child.returncode != 0:
        # Do not expose authentication logs, provider bodies or credential errors.
        try:
            reason = json.loads(child.stdout).get("reason")
        except (ValueError, TypeError, AttributeError):
            reason = None
        if reason in FETCH_FAILURE_REASONS:
            raise ValueError(reason)
        raise ValueError("schwab_chart_fetch_failed_check_auth_and_provider_health")
    if len(child.stdout) > 6 * 1024 * 1024:
        raise ValueError("schwab_chart_result_exceeds_limit")
    return json.loads(child.stdout)


def chart_report(captured, *, now):
    source = captured["source"]
    packet = {
        "schema_version": 1,
        "symbol": "SCHD",
        "evidence_kind": "schwab_api_chart_only",
        "candidate_id": "chart_context_only_not_bound_to_bot",
        "price_basis": "provider_as_returned_unverified",
        "decision": {
            "timestamp_utc": source["fetch_started_at_utc"],
            "symbol": "SCHD",
            "action": "WAIT",
            "strategy": "NO_BOT_DECISION_ATTACHED",
            "reasons": [],
            "gates": {},
            "metadata": {},
        },
        "quote": {},
        "candles": captured["candles"],
    }
    report = build_evidence(packet, now=now)
    report["chart_source"] = source
    report["blockers"].append(
        "chart_capture_is_not_a_bot_decision_or_an_executable_quote"
    )
    return report


def render_charts(report, *, directory, prefix, checked_path):
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "schd-rehearsal-matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    outputs = {}
    source = (
        "SYNTHETIC TEST DATA"
        if report["evidence_kind"] == "synthetic"
        else (
            str(report["chart_source"].get("provider", "unknown")).upper() + " API"
            if report.get("chart_source")
            else "IMPORTED DATA - provenance declared"
        )
    )
    for name, frame in report["timeframes"].items():
        rows = frame.get(
            "chart_candles",
            frame.get("period_candles", frame.get("recent_candles", [])),
        )
        if not rows:
            continue
        rows = rows[-60:]
        path = checked_path(directory / f"{prefix}_{name}.png")
        fig, (price_ax, volume_ax) = plt.subplots(
            2, 1, figsize=(11, 5.8), gridspec_kw={"height_ratios": [4, 1]}, sharex=True
        )
        try:
            for i, row in enumerate(rows):
                opened, close = row["open"], row["close"]
                color = "#067a62" if close >= opened else "#c13b4a"
                price_ax.vlines(i, row["low"], row["high"], color=color, linewidth=1)
                price_ax.add_patch(
                    Rectangle(
                        (i - 0.32, min(opened, close)),
                        0.64,
                        max(abs(close - opened), 0.00001),
                        facecolor=color,
                        edgecolor=color,
                    )
                )
                volume_ax.bar(i, row["volume"], color=color, width=0.64)
            levels = (
                ("prior_20_bar_low", "Prior 20-bar low", "#78716c"),
                ("prior_20_bar_high", "Prior 20-bar high", "#78716c"),
                ("sma20", "Latest SMA20", "#b07800"),
                ("sma50", "Latest SMA50", "#426b9d"),
            )
            for key, label, color in levels:
                if frame.get(key) is not None:
                    price_ax.axhline(
                        frame[key],
                        color=color,
                        linestyle="--",
                        linewidth=0.8,
                        label=f"{label}: {frame[key]:.3f}",
                    )
            outside = []
            for event in report.get("decision_chart_markers", []):
                when = timestamp(event["timestamp_utc"])
                x = None
                for index, candle in enumerate(rows):
                    start = timestamp(candle["start_utc"])
                    end = timestamp(candle["end_utc"])
                    if start <= when < end or (index == len(rows) - 1 and when == end):
                        x = index - 0.5 + (when - start).total_seconds() / (end - start).total_seconds()
                        break
                proposed = event["kind"] == "proposed"
                color = "#00695c" if event["action"] == "BUY" else "#b52342"
                label = f"{'PROPOSED' if proposed else 'EXECUTED'} {event['action']}"
                if proposed:
                    label += " (time only; no price)"
                detail = label + " " + event["timestamp_utc"]
                if not proposed:
                    detail += f" | {event['quantity']:g} @ {event['price']:g}"
                if x is None:
                    outside.append(detail)
                    continue
                options = dict(
                    marker="D" if proposed else ("^" if event["action"] == "BUY" else "v"),
                    s=65, facecolors="none" if proposed else color,
                    edgecolors=color, label=label, zorder=5,
                )
                if proposed:
                    price_ax.scatter(x, 0.82, transform=price_ax.get_xaxis_transform(), **options)
                    price_ax.axvline(x, color=color, linestyle=":", linewidth=0.8)
                else:
                    price_ax.scatter(x, event["price"], **options)
            if outside:
                fig.text(0.08, 0.06, "Outside displayed candle intervals (not plotted):\n"
                         + "\n".join(outside), fontsize=7, color="#555555")
            if price_ax.get_legend_handles_labels()[0]:
                price_ax.legend(loc="best", fontsize=8, ncol=2)
            ticks = sorted(
                set(
                    [0, len(rows) - 1]
                    + list(range(0, len(rows), max(1, len(rows) // 5)))
                )
            )
            volume_ax.set_xticks(
                ticks,
                [
                    (
                        timestamp(rows[i]["end_utc"]).strftime("%m-%d\n%H:%M")
                        if name in {"1m", "5m", "15m", "1h"}
                        else timestamp(rows[i]["end_utc"]).strftime("%Y-%m-%d")
                    )
                    for i in ticks
                ],
                fontsize=8,
            )
            for ax in (price_ax, volume_ax):
                ax.grid(axis="y", alpha=0.15)
                ax.set_xlim(-1, len(rows))
                ax.spines[["top", "right"]].set_visible(False)
                ax.tick_params(labelsize=8)
            price_ax.set_ylabel("Price (USD)")
            volume_ax.set_ylabel("Volume", fontsize=8)
            volume_ax.set_xlabel(
                "Closed-candle time (UTC); missing intervals not synthesized"
                if report.get("chart_source", {}).get("provider") == "coinbase"
                else "Closed-candle time (UTC); non-trading gaps compressed"
            )
            title = (
                "180 calendar-day aggregate (one window)" if name == "180d" else name
            )
            fig.suptitle(
                f"{report.get('symbol', 'SCHD')} | {title} | {source}",
                x=0.08,
                ha="left",
                fontsize=13,
                fontweight="bold",
            )
            price_ax.set_title(
                f"As of {report['as_of_utc']} | {frame['status']} | Latest close: {rows[-1]['end_utc']}",
                loc="left",
                fontsize=8,
            )
            fig.text(
                0.08,
                0.015,
                (f"Recorded: {report.get('recorded_action')} / {report.get('recorded_gate_decision')}. "
                 "Dashed levels are review calculations, not claimed bot inputs."
                 if report.get("recorded_action") else
                 "Closed bars only. Price-adjustment and dividend context require review. Charts do not authorize orders."),
                fontsize=8,
                color="#555555",
            )
            footer = 0.10 + len(outside) * 0.025 if outside else 0.035
            if report.get("review_kind") == "retrospective_execution_review":
                price_ax.set_title("RETROSPECTIVE EXECUTION REVIEW - NOT DECISION INPUT | "
                                   + price_ax.get_title(loc="left"), loc="left", fontsize=8)
            fig.tight_layout(rect=(0, footer, 1, 0.95))
            # Atomic replacement keeps a interrupted renderer from publishing a half PNG.
            fd, temporary = tempfile.mkstemp(
                prefix=f".{prefix}_{name}.", suffix=".png", dir=directory
            )
            os.close(fd)
            try:
                fig.savefig(temporary, dpi=140, facecolor="white")
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            outputs[name] = str(path)
        finally:
            plt.close(fig)
    return outputs
