#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import html
import json
import os
import shutil
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON_PATH = PROJECT_ROOT / "governance" / "health" / "paper_performance_latest.json"
DEFAULT_MD_PATH = PROJECT_ROOT / "exports" / "reports" / "paper_performance_latest.md"
DEFAULT_HTML_PATH = PROJECT_ROOT / "exports" / "reports" / "paper_performance_latest.html"
DEFAULT_PDF_PATH = PROJECT_ROOT / "exports" / "reports" / "paper_performance_latest.pdf"
DEFAULT_WEEKLY_CHART_PATH = PROJECT_ROOT / "exports" / "reports" / "paper_performance_weekly_latest.png"
DEFAULT_MONTHLY_CHART_PATH = PROJECT_ROOT / "exports" / "reports" / "paper_performance_monthly_latest.png"
DEFAULT_QUARTERLY_CHART_PATH = PROJECT_ROOT / "exports" / "reports" / "paper_performance_quarterly_latest.png"
DEFAULT_SLEEVES_CHART_PATH = PROJECT_ROOT / "exports" / "reports" / "paper_performance_sleeves_latest.png"
DEFAULT_SLEEVE_ORDER = (
    "default",
    "conservative",
    "aggressive",
    "intraday_aggressive",
    "swing_aggressive",
    "dividend",
    "bond",
    "fx",
    "schwab_futures",
    "crypto_futures",
)
APP_BROWSER_CANDIDATES = (
    Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
    Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
    Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _run(cmd: list[str]) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()
    except Exception as exc:
        return 1, "", str(exc)


def _default_allow_gui_pdf_renderer() -> bool:
    return any(candidate.exists() for candidate in APP_BROWSER_CANDIDATES)


def _pdf_renderer_binary(allow_gui_renderer: bool) -> tuple[str, str]:
    env_override = (
        os.getenv("PAPER_PERFORMANCE_PDF_BIN", "").strip()
        or os.getenv("REPORT_PDF_BUNDLE_PDF_BIN", "").strip()
        or os.getenv("TRAINING_REPORT_PDF_BIN", "").strip()
    )
    if env_override:
        env_bin = Path(env_override).expanduser()
        if env_bin.exists():
            kind = "wkhtmltopdf" if env_bin.name == "wkhtmltopdf" else "browser"
            return str(env_bin), kind

    wkhtmltopdf = shutil.which("wkhtmltopdf")
    if wkhtmltopdf:
        return wkhtmltopdf, "wkhtmltopdf"

    for candidate in (
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        shutil.which("google-chrome"),
        shutil.which("google-chrome-stable"),
        shutil.which("microsoft-edge"),
        shutil.which("msedge"),
    ):
        if candidate:
            return candidate, "browser"

    if allow_gui_renderer:
        for candidate in APP_BROWSER_CANDIDATES:
            if candidate.exists():
                return str(candidate), "browser"

    return "", ""


def _render_pdf_from_html(html_path: Path, pdf_path: Path, *, allow_gui_renderer: bool) -> tuple[bool, str]:
    renderer, renderer_kind = _pdf_renderer_binary(allow_gui_renderer=allow_gui_renderer)
    if not renderer:
        return False, "pdf_renderer_not_found"
    html_uri = html_path.resolve().as_uri()
    if renderer_kind == "wkhtmltopdf":
        cmd = [renderer, html_uri, str(pdf_path)]
    else:
        cmd = [renderer, "--headless", "--disable-gpu", f"--print-to-pdf={pdf_path}", html_uri]
    rc, out, err = _run(cmd)
    if rc == 0 and pdf_path.exists() and pdf_path.stat().st_size > 0:
        return True, out or "ok"
    return False, err or out or f"rc={rc}"


def _parse_ts(raw: Any) -> datetime | None:
    text = str(raw or "").strip().replace("Z", "+00:00")
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _day_key(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%d")


def _paper_source_files(project_root: Path) -> tuple[list[Path], str]:
    bridge = sorted((project_root / "exports" / "paper_broker_bridge" / "paper").glob("paper_bridge_orders_*.jsonl"))
    if bridge:
        return bridge, "paper_broker_bridge"

    trade_logs = sorted((project_root / "exports" / "trade_logs").rglob("paper_trades_*.jsonl"))
    if trade_logs:
        return trade_logs, "trade_logs"

    root_files = sorted(project_root.glob("paper_trades_*.jsonl"))
    return root_files, "root_paper_trades"


def _iter_rows(files: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in files:
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                for raw in handle:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(row, dict):
                        yield row
        except Exception:
            continue


def _profile_of(row: dict[str, Any]) -> str:
    meta = row.get("metadata")
    if isinstance(meta, dict):
        text = str(meta.get("source_profile") or "").strip().lower()
        if text:
            return text
    text = str(row.get("profile") or "").strip().lower()
    return text or "default"


def _strategy_of(row: dict[str, Any]) -> str:
    text = str(row.get("strategy") or "").strip()
    if text:
        return text
    meta = row.get("metadata")
    if isinstance(meta, dict):
        meta_text = str(meta.get("strategy_id") or meta.get("bot_id") or "").strip()
        if meta_text:
            return meta_text
    symbol = str(row.get("symbol") or "").strip().upper() or "UNKNOWN"
    action = str(row.get("action") or "").strip().upper() or "UNKNOWN"
    return f"unknown::{symbol}::{action}"


def _net_total(row: dict[str, Any]) -> float:
    realized = _safe_float(row.get("realized_pnl_total"), _safe_float(row.get("realized_pnl")))
    unrealized = _safe_float(row.get("unrealized_pnl_total"), _safe_float(row.get("unrealized_pnl")))
    return float(realized + unrealized)


def _realized_total(row: dict[str, Any]) -> float:
    return _safe_float(row.get("realized_pnl_total"), _safe_float(row.get("realized_pnl")))


def _unrealized_total(row: dict[str, Any]) -> float:
    return _safe_float(row.get("unrealized_pnl_total"), _safe_float(row.get("unrealized_pnl")))


def _rank_counter(counter: Counter[str], limit: int = 5) -> list[dict[str, Any]]:
    return [
        {"name": key, "executions": int(count)}
        for key, count in counter.most_common(limit)
    ]


def _empty_stats() -> dict[str, Any]:
    return {
        "executions": 0,
        "buy_count": 0,
        "sell_count": 0,
        "profiles": Counter(),
        "symbols": Counter(),
        "strategies": Counter(),
    }


def _update_stats(stats: dict[str, Any], row: dict[str, Any]) -> None:
    stats["executions"] = int(stats.get("executions", 0)) + 1
    action = str(row.get("action") or "").upper().strip()
    if action.startswith("BUY"):
        stats["buy_count"] = int(stats.get("buy_count", 0)) + 1
    elif action.startswith("SELL"):
        stats["sell_count"] = int(stats.get("sell_count", 0)) + 1
    stats["profiles"][_profile_of(row)] += 1
    symbol = str(row.get("symbol") or "").strip().upper() or "UNKNOWN"
    strategy = str(row.get("strategy") or "").strip() or "unknown"
    stats["symbols"][symbol] += 1
    stats["strategies"][strategy] += 1


def _chart_day_label(day_utc: str) -> str:
    text = str(day_utc or "").strip()
    if len(text) != 8:
        return text
    return f"{text[4:6]}-{text[6:8]}"


def _week_start_key(day_utc: str) -> str:
    dt = datetime.strptime(str(day_utc), "%Y%m%d").date()
    return (dt - timedelta(days=dt.weekday())).strftime("%Y%m%d")


def _month_key(day_utc: str) -> str:
    text = str(day_utc or "").strip()
    if len(text) != 8:
        return text
    return text[:6]


def _quarter_key(day_utc: str) -> str:
    dt = datetime.strptime(str(day_utc), "%Y%m%d").date()
    quarter = ((dt.month - 1) // 3) + 1
    return f"{dt.year}Q{quarter}"


def _build_history_series(
    all_days: list[str],
    latest_by_day_profile: dict[str, dict[str, dict[str, tuple[datetime, dict[str, Any]]]]],
    stats_by_day: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    series: list[dict[str, Any]] = []
    prev_net = 0.0
    for dkey in all_days:
        totals = _profile_totals(latest_by_day_profile.get(dkey, {}))
        ending_net = float(totals["ending_net_pnl_total"])
        ending_realized = float(totals["ending_realized_pnl_total"])
        day_stats = stats_by_day.get(dkey, _empty_stats())
        series.append(
            {
                "day_utc": dkey,
                "executions": int(day_stats.get("executions", 0)),
                "ending_net_pnl_total": round(float(ending_net), 6),
                "ending_realized_pnl_total": round(float(ending_realized), 6),
                "change_vs_previous_day": round(float(ending_net - prev_net), 6),
            }
        )
        prev_net = ending_net
    return series


def _build_weekly_history_series(history_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _build_period_history_series(
        history_rows,
        key_builder=_week_start_key,
        label_name="week",
    )


def _build_period_history_series(
    history_rows: list[dict[str, Any]],
    *,
    key_builder,
    label_name: str,
) -> list[dict[str, Any]]:
    latest_by_period: dict[str, dict[str, Any]] = {}
    for row in history_rows:
        if not isinstance(row, dict):
            continue
        day_utc = str(row.get("day_utc", "")).strip()
        if len(day_utc) != 8:
            continue
        period_key = str(key_builder(day_utc))
        current = latest_by_period.get(period_key)
        if current is None or day_utc > str(current.get("day_utc", "")):
            latest_by_period[period_key] = row

    period_rows: list[dict[str, Any]] = []
    previous_end = 0.0
    for period_key in sorted(latest_by_period.keys()):
        row = latest_by_period[period_key]
        ending_net = float(row.get("ending_net_pnl_total", 0.0) or 0.0)
        period_rows.append(
            {
                f"{label_name}_key": period_key,
                f"{label_name}_end_day_utc": str(row.get("day_utc", "")),
                "ending_net_pnl_total": round(float(ending_net), 6),
                "change_vs_previous_period": round(float(ending_net - previous_end), 6),
            }
        )
        previous_end = ending_net
    return period_rows


def _build_period_change_series(
    *,
    selected_day: str,
    selected_net: float,
    week_to_date_change: float,
    all_days: list[str],
    history_by_day: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    selected_date = datetime.strptime(selected_day, "%Y%m%d").date()
    period_rows: list[dict[str, Any]] = [
        {
            "label": "WTD",
            "window_days": int(selected_date.weekday() + 1),
            "change": round(float(week_to_date_change), 6),
        }
    ]
    for window_days in (7, 14, 21, 30):
        start_day = (selected_date - timedelta(days=window_days - 1)).strftime("%Y%m%d")
        prior_day = max((d for d in all_days if d < start_day), default="")
        prior_net = float((history_by_day.get(prior_day) or {}).get("ending_net_pnl_total", 0.0) or 0.0)
        available_days = sum(1 for d in all_days if start_day <= d <= selected_day)
        period_rows.append(
            {
                "label": f"{window_days}D",
                "window_days": int(window_days),
                "available_days": int(available_days),
                "change": round(float(selected_net - prior_net), 6),
            }
        )
    return period_rows


def _profile_totals(profile_rows: dict[str, tuple[datetime, dict[str, Any]]] | None) -> dict[str, Any]:
    if not profile_rows:
        return {
            "available": False,
            "ending_timestamp_utc": "",
            "ending_realized_pnl_total": 0.0,
            "ending_unrealized_pnl_total": 0.0,
            "ending_net_pnl_total": 0.0,
        }

    latest_ts: datetime | None = None
    realized_sum = 0.0
    unrealized_sum = 0.0

    def _iter_leaves(node: Any) -> Iterable[tuple[datetime, dict[str, Any]]]:
        if isinstance(node, tuple) and len(node) == 2:
            ts, row = node
            if isinstance(ts, datetime) and isinstance(row, dict):
                yield ts, row
            return
        if isinstance(node, dict):
            for child in node.values():
                yield from _iter_leaves(child)

    for ts, row in _iter_leaves(profile_rows):
        realized_sum += _realized_total(row)
        unrealized_sum += _unrealized_total(row)
        if latest_ts is None or ts > latest_ts:
            latest_ts = ts
    return {
        "available": True,
        "ending_timestamp_utc": latest_ts.isoformat().replace("+00:00", "Z") if latest_ts is not None else "",
        "ending_realized_pnl_total": round(float(realized_sum), 6),
        "ending_unrealized_pnl_total": round(float(unrealized_sum), 6),
        "ending_net_pnl_total": round(float(realized_sum + unrealized_sum), 6),
    }


def _build_sleeve_daily_series(
    all_days: list[str],
    latest_by_day_profile: dict[str, dict[str, dict[str, tuple[datetime, dict[str, Any]]]]],
) -> dict[str, list[dict[str, Any]]]:
    sleeves: dict[str, list[dict[str, Any]]] = defaultdict(list)
    profiles = _ordered_profiles(latest_by_day_profile)
    for profile in profiles:
        prev_net = 0.0
        for dkey in all_days:
            row_map = latest_by_day_profile.get(dkey, {})
            current = row_map.get(profile)
            if current is None:
                continue
            totals = _profile_totals(current)
            ending_realized = float(totals.get("ending_realized_pnl_total", 0.0) or 0.0)
            ending_unrealized = float(totals.get("ending_unrealized_pnl_total", 0.0) or 0.0)
            ending_net = float(totals.get("ending_net_pnl_total", 0.0) or 0.0)
            sleeves[profile].append(
                {
                    "day_utc": dkey,
                    "ending_realized_pnl_total": round(float(ending_realized), 6),
                    "ending_unrealized_pnl_total": round(float(ending_unrealized), 6),
                    "ending_net_pnl_total": round(float(ending_net), 6),
                    "change_vs_previous_day": round(float(ending_net - prev_net), 6),
                }
            )
            prev_net = ending_net
    return {profile: sleeves.get(profile, []) for profile in profiles}


def _ordered_profiles(latest_by_day_profile: dict[str, dict[str, dict[str, tuple[datetime, dict[str, Any]]]]]) -> list[str]:
    seen = {profile for rows in latest_by_day_profile.values() for profile in rows.keys()}
    ordered = list(DEFAULT_SLEEVE_ORDER)
    extras = sorted(seen.difference(DEFAULT_SLEEVE_ORDER))
    return ordered + extras


def _latest_profile_rows(
    latest_by_day_profile: dict[str, dict[str, dict[str, tuple[datetime, dict[str, Any]]]]],
) -> dict[str, tuple[str, datetime, dict[str, dict[str, tuple[datetime, dict[str, Any]]]]]]:
    latest: dict[str, tuple[str, datetime, dict[str, tuple[datetime, dict[str, Any]]]]] = {}
    for day_key, profile_rows in latest_by_day_profile.items():
        for profile, strategy_rows in profile_rows.items():
            ts_values = [ts for ts, _row in strategy_rows.values()]
            if not ts_values:
                continue
            ts = max(ts_values)
            current = latest.get(profile)
            if current is None or ts > current[1]:
                latest[profile] = (day_key, ts, strategy_rows)
    return latest


def _build_sleeve_period_history(
    sleeve_daily_series: dict[str, list[dict[str, Any]]],
    *,
    key_builder,
    label_name: str,
) -> dict[str, list[dict[str, Any]]]:
    payload: dict[str, list[dict[str, Any]]] = {}
    for profile, rows in sleeve_daily_series.items():
        payload[profile] = _build_period_history_series(rows, key_builder=key_builder, label_name=label_name)
    return payload


def _sleeve_chart_profiles(sleeve_latest: list[dict[str, Any]]) -> list[str]:
    profiles: list[str] = []
    seen: set[str] = set()
    for row in sleeve_latest:
        if not isinstance(row, dict):
            continue
        profile = str(row.get("profile", "")).strip()
        if not profile or profile in seen:
            continue
        seen.add(profile)
        profiles.append(profile)
    return profiles


def _sleeve_snapshot_points(sleeve_latest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in sleeve_latest:
        if not isinstance(row, dict):
            continue
        profile = str(row.get("profile", "")).strip()
        if not profile:
            continue
        rows.append(
            {
                "profile": profile,
                "ending_net_pnl_total": float(row.get("ending_net_pnl_total", 0.0) or 0.0),
                "executions": int(row.get("executions", 0) or 0),
                "win_rate": row.get("win_rate"),
                "winning_strategy_count": int(row.get("winning_strategy_count", 0) or 0),
                "losing_strategy_count": int(row.get("losing_strategy_count", 0) or 0),
                "flat_strategy_count": int(row.get("flat_strategy_count", 0) or 0),
            }
        )
    return rows


def _strategy_outcome_summary(
    strategy_rows: dict[str, tuple[datetime, dict[str, Any]]] | None,
) -> dict[str, Any]:
    if not strategy_rows:
        return {
            "strategy_count": 0,
            "winning_strategy_count": 0,
            "losing_strategy_count": 0,
            "flat_strategy_count": 0,
            "non_flat_strategy_count": 0,
            "win_rate": None,
        }

    wins = 0
    losses = 0
    flats = 0
    for _strategy, (_ts, row) in strategy_rows.items():
        net = _net_total(row)
        if net > 0:
            wins += 1
        elif net < 0:
            losses += 1
        else:
            flats += 1

    non_flat = wins + losses
    win_rate = round(float(wins / non_flat), 6) if non_flat > 0 else None
    return {
        "strategy_count": int(len(strategy_rows)),
        "winning_strategy_count": int(wins),
        "losing_strategy_count": int(losses),
        "flat_strategy_count": int(flats),
        "non_flat_strategy_count": int(non_flat),
        "win_rate": win_rate,
    }


def _format_win_rate(raw: Any) -> str:
    try:
        if raw is None:
            return "n/a"
        return f"{float(raw) * 100.0:.1f}%"
    except Exception:
        return "n/a"


def _format_strategy_pnl_brief(rows: list[dict[str, Any]], *, positive: bool) -> str:
    if not rows:
        return "n/a"
    parts: list[str] = []
    for row in rows:
        strategy = str(row.get("strategy") or "").strip() or "unknown"
        value = float(row.get("ending_net_pnl_total", 0.0) or 0.0)
        parts.append(f"{strategy}({value:+.2f})")
    return ", ".join(parts)


def _strategy_snapshot_rankings(
    strategy_rows: dict[str, tuple[datetime, dict[str, Any]]] | None,
    *,
    limit: int = 3,
) -> dict[str, list[dict[str, Any]]]:
    if not strategy_rows:
        return {
            "top_winning_strategies": [],
            "top_losing_strategies": [],
        }

    ranked = [
        {
            "strategy": str(strategy or "").strip(),
            "ending_net_pnl_total": round(float(_net_total(row)), 6),
        }
        for strategy, (_ts, row) in strategy_rows.items()
    ]
    ranked.sort(key=lambda item: (float(item["ending_net_pnl_total"]), item["strategy"]), reverse=True)
    winners = [row for row in ranked if float(row.get("ending_net_pnl_total", 0.0) or 0.0) > 0.0][: max(int(limit), 1)]
    losers = [
        row
        for row in sorted(ranked, key=lambda item: (float(item["ending_net_pnl_total"]), item["strategy"]))
        if float(row.get("ending_net_pnl_total", 0.0) or 0.0) < 0.0
    ][: max(int(limit), 1)]
    return {
        "top_winning_strategies": winners,
        "top_losing_strategies": losers,
    }


def _build_sleeve_latest_summary(
    *,
    day: str,
    latest_by_day_profile: dict[str, dict[str, dict[str, tuple[datetime, dict[str, Any]]]]],
    stats_by_day: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    profile_rows = latest_by_day_profile.get(day, {})
    latest_rows = _latest_profile_rows(latest_by_day_profile)
    rows: list[dict[str, Any]] = []
    for profile in _ordered_profiles(latest_by_day_profile):
        current = profile_rows.get(profile)
        source_day = day
        profile_current: dict[str, tuple[datetime, dict[str, Any]]] | None = None
        current_day_available = False
        if current is not None:
            profile_current = current
            current_day_available = True
        else:
            latest = latest_rows.get(profile)
            if latest is not None:
                source_day, _ts, profile_current = latest
        source_stats = stats_by_day.get(source_day, _empty_stats())
        source_execs = (source_stats.get("profiles") or Counter())
        if profile_current is None:
            rows.append(
                {
                    "profile": profile,
                    "day_utc": "",
                    "current_day_available": False,
                    "data_status": "no_data",
                    "executions": 0,
                    "ending_realized_pnl_total": 0.0,
                    "ending_unrealized_pnl_total": 0.0,
                    "ending_net_pnl_total": 0.0,
                }
            )
            continue
        totals = _profile_totals(profile_current)
        outcome = _strategy_outcome_summary(profile_current)
        rankings = _strategy_snapshot_rankings(profile_current)
        ending_realized = float(totals.get("ending_realized_pnl_total", 0.0) or 0.0)
        ending_unrealized = float(totals.get("ending_unrealized_pnl_total", 0.0) or 0.0)
        ending_net = float(totals.get("ending_net_pnl_total", 0.0) or 0.0)
        rows.append(
            {
                "profile": profile,
                "day_utc": source_day,
                "current_day_available": current_day_available,
                "data_status": "current" if current_day_available else "latest_available",
                "executions": int(source_execs.get(profile, 0)),
                "ending_realized_pnl_total": round(float(ending_realized), 6),
                "ending_unrealized_pnl_total": round(float(ending_unrealized), 6),
                "ending_net_pnl_total": round(float(ending_net), 6),
                "strategy_count": int(outcome.get("strategy_count", 0) or 0),
                "winning_strategy_count": int(outcome.get("winning_strategy_count", 0) or 0),
                "losing_strategy_count": int(outcome.get("losing_strategy_count", 0) or 0),
                "flat_strategy_count": int(outcome.get("flat_strategy_count", 0) or 0),
                "non_flat_strategy_count": int(outcome.get("non_flat_strategy_count", 0) or 0),
                "win_rate": outcome.get("win_rate"),
                "win_rate_basis": "latest_non_flat_strategy_snapshots",
                "top_winning_strategies": rankings.get("top_winning_strategies", []),
                "top_losing_strategies": rankings.get("top_losing_strategies", []),
            }
        )
    return rows


def _summarize_day(
    *,
    day: str,
    profile_latest_rows: dict[str, dict[str, tuple[datetime, dict[str, Any]]]] | None,
    stats: dict[str, Any],
    previous_profile_latest_rows: dict[str, dict[str, tuple[datetime, dict[str, Any]]]] | None,
) -> dict[str, Any]:
    totals = _profile_totals(profile_latest_rows)
    previous_totals = _profile_totals(previous_profile_latest_rows)
    if not bool(totals.get("available", False)):
        return {
            "day_utc": day,
            "available": False,
            "executions": int(stats.get("executions", 0)),
            "buy_count": int(stats.get("buy_count", 0)),
            "sell_count": int(stats.get("sell_count", 0)),
            "unique_symbols": int(len(stats.get("symbols", {}))),
            "change_vs_previous_day": 0.0,
            "realized_change_vs_previous_day": 0.0,
            "ending_realized_pnl_total": 0.0,
            "ending_unrealized_pnl_total": 0.0,
            "ending_net_pnl_total": 0.0,
            "top_profiles": _rank_counter(stats.get("profiles", Counter())),
            "top_symbols": _rank_counter(stats.get("symbols", Counter())),
            "top_strategies": _rank_counter(stats.get("strategies", Counter())),
        }

    ending_realized = float(totals.get("ending_realized_pnl_total", 0.0) or 0.0)
    ending_unrealized = float(totals.get("ending_unrealized_pnl_total", 0.0) or 0.0)
    ending_net = float(totals.get("ending_net_pnl_total", 0.0) or 0.0)
    previous_net = float(previous_totals.get("ending_net_pnl_total", 0.0) or 0.0)
    previous_realized = float(previous_totals.get("ending_realized_pnl_total", 0.0) or 0.0)

    return {
        "day_utc": day,
        "available": True,
        "ending_timestamp_utc": str(totals.get("ending_timestamp_utc") or ""),
        "executions": int(stats.get("executions", 0)),
        "buy_count": int(stats.get("buy_count", 0)),
        "sell_count": int(stats.get("sell_count", 0)),
        "unique_symbols": int(len(stats.get("symbols", {}))),
        "change_vs_previous_day": round(float(ending_net - previous_net), 6),
        "realized_change_vs_previous_day": round(float(ending_realized - previous_realized), 6),
        "ending_realized_pnl_total": round(float(ending_realized), 6),
        "ending_unrealized_pnl_total": round(float(ending_unrealized), 6),
        "ending_net_pnl_total": round(float(ending_net), 6),
        "top_profiles": _rank_counter(stats.get("profiles", Counter())),
        "top_symbols": _rank_counter(stats.get("symbols", Counter())),
        "top_strategies": _rank_counter(stats.get("strategies", Counter())),
    }


def build_paper_performance_report(project_root: Path, *, day: str, week_days: int = 7) -> dict[str, Any]:
    files, source_kind = _paper_source_files(project_root)
    latest_by_day_profile: dict[str, dict[str, dict[str, tuple[datetime, dict[str, Any]]]]] = defaultdict(lambda: defaultdict(dict))
    stats_by_day: dict[str, dict[str, Any]] = defaultdict(_empty_stats)

    for row in _iter_rows(files):
        ts = _parse_ts(row.get("timestamp_utc"))
        if ts is None:
            continue
        dkey = _day_key(ts)
        profile = _profile_of(row)
        strategy = _strategy_of(row)
        _update_stats(stats_by_day[dkey], row)
        current = latest_by_day_profile[dkey][profile].get(strategy)
        if current is None or ts > current[0]:
            latest_by_day_profile[dkey][profile][strategy] = (ts, row)

    all_days = sorted(latest_by_day_profile.keys())
    selected_latest = latest_by_day_profile.get(day)
    previous_day = max((d for d in all_days if d < day), default="")
    previous_latest = latest_by_day_profile.get(previous_day)

    day_summary = _summarize_day(
        day=day,
        profile_latest_rows=selected_latest,
        stats=stats_by_day.get(day, _empty_stats()),
        previous_profile_latest_rows=previous_latest,
    )

    selected_date = datetime.strptime(day, "%Y%m%d").date()
    week_start_date = selected_date - timedelta(days=selected_date.weekday())
    week_start = week_start_date.strftime("%Y%m%d")
    rolling_start = (selected_date - timedelta(days=max(int(week_days), 1) - 1)).strftime("%Y%m%d")
    prior_week_day = max((d for d in all_days if d < week_start), default="")
    prior_rolling_day = max((d for d in all_days if d < rolling_start), default="")

    week_profiles: Counter[str] = Counter()
    week_symbols: Counter[str] = Counter()
    week_strategies: Counter[str] = Counter()
    week_exec = 0
    week_buys = 0
    week_sells = 0
    rolling_series: list[dict[str, Any]] = []

    for dkey in sorted(d for d in stats_by_day.keys() if week_start <= d <= day):
        day_stats = stats_by_day[dkey]
        week_exec += int(day_stats.get("executions", 0))
        week_buys += int(day_stats.get("buy_count", 0))
        week_sells += int(day_stats.get("sell_count", 0))
        week_profiles.update(day_stats.get("profiles", Counter()))
        week_symbols.update(day_stats.get("symbols", Counter()))
        week_strategies.update(day_stats.get("strategies", Counter()))

    history_series = _build_history_series(all_days, latest_by_day_profile, stats_by_day)
    history_by_day = {str(row.get("day_utc", "")): row for row in history_series if isinstance(row, dict)}
    for dkey in sorted(d for d in all_days if rolling_start <= d <= day):
        row = history_by_day.get(dkey)
        if isinstance(row, dict):
            rolling_series.append(
                {
                    "day_utc": dkey,
                    "ending_net_pnl_total": round(float(row.get("ending_net_pnl_total", 0.0) or 0.0), 6),
                    "change_vs_previous_day": round(float(row.get("change_vs_previous_day", 0.0) or 0.0), 6),
                }
            )

    selected_net = float(day_summary.get("ending_net_pnl_total", 0.0) or 0.0)
    selected_realized = float(day_summary.get("ending_realized_pnl_total", 0.0) or 0.0)
    prior_week_net = float((history_by_day.get(prior_week_day) or {}).get("ending_net_pnl_total", 0.0) or 0.0)
    prior_week_realized = float((history_by_day.get(prior_week_day) or {}).get("ending_realized_pnl_total", 0.0) or 0.0)
    prior_rolling_net = float((history_by_day.get(prior_rolling_day) or {}).get("ending_net_pnl_total", 0.0) or 0.0)
    weekly_history_series = _build_weekly_history_series(history_series)
    monthly_history_series = _build_period_history_series(
        history_series,
        key_builder=_month_key,
        label_name="month",
    )
    quarterly_history_series = _build_period_history_series(
        history_series,
        key_builder=_quarter_key,
        label_name="quarter",
    )
    sleeve_daily_series = _build_sleeve_daily_series(all_days, latest_by_day_profile)
    sleeve_weekly_history_series = _build_sleeve_period_history(
        sleeve_daily_series,
        key_builder=_week_start_key,
        label_name="week",
    )
    week_to_date_change = round(float(selected_net - prior_week_net), 6)

    week_summary = {
        "week_start_day_utc": week_start,
        "week_end_day_utc": day,
        "available": bool(day_summary.get("available", False)),
        "executions": int(week_exec),
        "buy_count": int(week_buys),
        "sell_count": int(week_sells),
        "week_to_date_change": week_to_date_change,
        "week_to_date_realized_change": round(float(selected_realized - prior_week_realized), 6),
        "rolling_change_days": int(max(int(week_days), 1)),
        "rolling_change": round(float(selected_net - prior_rolling_net), 6),
        "ending_net_pnl_total": round(float(selected_net), 6),
        "top_profiles": _rank_counter(week_profiles),
        "top_symbols": _rank_counter(week_symbols),
        "top_strategies": _rank_counter(week_strategies),
        "daily_series": rolling_series,
    }

    return {
        "timestamp_utc": _utc_now().isoformat(),
        "schema_version": 1,
        "ok": bool(selected_latest),
        "source_kind": source_kind,
        "source_files_scanned": int(len(files)),
        "source_files": [str(path) for path in files[:10]],
        "available_days": all_days[-14:],
        "history_daily_series": history_series[-60:],
        "weekly_history_series": weekly_history_series[-16:],
        "monthly_history_series": monthly_history_series[-18:],
        "quarterly_history_series": quarterly_history_series[-16:],
        "sleeve_daily_series": {profile: rows[-60:] for profile, rows in sorted(sleeve_daily_series.items())},
        "sleeve_weekly_history_series": {profile: rows[-16:] for profile, rows in sorted(sleeve_weekly_history_series.items())},
        "sleeve_latest": _build_sleeve_latest_summary(day=day, latest_by_day_profile=latest_by_day_profile, stats_by_day=stats_by_day),
        "period_change_series": _build_period_change_series(
            selected_day=day,
            selected_net=float(selected_net),
            week_to_date_change=float(week_to_date_change),
            all_days=all_days,
            history_by_day=history_by_day,
        ),
        "day": day_summary,
        "week": week_summary,
    }


def render_paper_performance_graphs(
    payload: dict[str, Any],
    *,
    weekly_chart_path: Path,
    monthly_chart_path: Path,
    quarterly_chart_path: Path,
    sleeves_chart_path: Path,
) -> dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {
            "available": False,
            "error": f"matplotlib_unavailable:{type(exc).__name__}:{exc}",
        }

    weekly_rows = [
        row
        for row in (payload.get("weekly_history_series") or [])
        if isinstance(row, dict)
    ]
    monthly_rows = [
        row
        for row in (payload.get("monthly_history_series") or [])
        if isinstance(row, dict)
    ]
    quarterly_rows = [
        row
        for row in (payload.get("quarterly_history_series") or [])
        if isinstance(row, dict)
    ]
    sleeve_weekly_rows = payload.get("sleeve_weekly_history_series") if isinstance(payload.get("sleeve_weekly_history_series"), dict) else {}
    sleeve_latest = payload.get("sleeve_latest") if isinstance(payload.get("sleeve_latest"), list) else []
    weekly_rows = weekly_rows[-12:]
    monthly_rows = monthly_rows[-18:]
    quarterly_rows = quarterly_rows[-16:]

    weekly_chart_path.parent.mkdir(parents=True, exist_ok=True)
    monthly_chart_path.parent.mkdir(parents=True, exist_ok=True)
    quarterly_chart_path.parent.mkdir(parents=True, exist_ok=True)
    sleeves_chart_path.parent.mkdir(parents=True, exist_ok=True)

    if weekly_rows:
        labels = [_chart_day_label(str(row.get("week_end_day_utc", ""))) for row in weekly_rows]
        ending_values = [float(row.get("ending_net_pnl_total", 0.0) or 0.0) for row in weekly_rows]

        fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=160)
        ax.plot(labels, ending_values, color="#0f766e", linestyle=":", linewidth=2.4, marker="o", markersize=6.0)
        ax.fill_between(labels, ending_values, [0.0] * len(ending_values), color="#99f6e4", alpha=0.18)
        ax.axhline(0.0, color="#243b53", linewidth=1.0, alpha=0.9)
        ax.set_title("Paper Weekly Performance")
        ax.set_ylabel("Ending Net PnL")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(weekly_chart_path, bbox_inches="tight")
        plt.close(fig)

    if monthly_rows:
        labels = [str(row.get("month_key", "")) for row in monthly_rows]
        ending_values = [float(row.get("ending_net_pnl_total", 0.0) or 0.0) for row in monthly_rows]

        fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=160)
        ax.plot(labels, ending_values, color="#7c3aed", linestyle=":", linewidth=2.4, marker="o", markersize=6.0)
        ax.fill_between(labels, ending_values, [0.0] * len(ending_values), color="#ddd6fe", alpha=0.18)
        ax.axhline(0.0, color="#243b53", linewidth=1.0, alpha=0.9)
        ax.set_title("Paper Monthly Performance")
        ax.set_ylabel("Ending Net PnL")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(monthly_chart_path, bbox_inches="tight")
        plt.close(fig)

    if quarterly_rows:
        labels = [str(row.get("quarter_key", "")) for row in quarterly_rows]
        ending_values = [float(row.get("ending_net_pnl_total", 0.0) or 0.0) for row in quarterly_rows]

        fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=160)
        ax.plot(labels, ending_values, color="#b45309", linestyle=":", linewidth=2.4, marker="o", markersize=6.0)
        ax.fill_between(labels, ending_values, [0.0] * len(ending_values), color="#fde68a", alpha=0.18)
        ax.axhline(0.0, color="#243b53", linewidth=1.0, alpha=0.9)
        ax.set_title("Paper Quarterly Performance")
        ax.set_ylabel("Ending Net PnL")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(quarterly_chart_path, bbox_inches="tight")
        plt.close(fig)

    top_sleeves = _sleeve_chart_profiles(sleeve_latest)
    if top_sleeves:
        snapshot_points = _sleeve_snapshot_points(sleeve_latest)
        fig_height = max(7.4, 5.1 + (0.36 * max(len(top_sleeves) - 4, 0)))
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(10.8, fig_height),
            dpi=160,
            gridspec_kw={"height_ratios": [2.2, 1.2]},
        )
        ax = axes[0]
        palette = [
            "#0f766e",
            "#2563eb",
            "#7c3aed",
            "#b45309",
            "#dc2626",
            "#0891b2",
            "#65a30d",
            "#9333ea",
        ]
        plotted = 0
        for idx, profile in enumerate(top_sleeves):
            rows = [
                row for row in (sleeve_weekly_rows.get(profile) or [])
                if isinstance(row, dict)
            ]
            if not rows:
                continue
            labels = [_chart_day_label(str(row.get("week_end_day_utc", ""))) for row in rows[-12:]]
            values = [float(row.get("ending_net_pnl_total", 0.0) or 0.0) for row in rows[-12:]]
            ax.plot(
                labels,
                values,
                color=palette[idx % len(palette)],
                linestyle=":",
                linewidth=2.0,
                marker="o",
                markersize=4.8,
                label=profile,
            )
            plotted += 1
        if plotted:
            ax.axhline(0.0, color="#243b53", linewidth=1.0, alpha=0.9)
            ax.set_title("Paper Sleeve Weekly Progression")
            ax.set_ylabel("Ending Net PnL")
            ax.grid(axis="y", linestyle="--", alpha=0.25)
            ax.tick_params(axis="x", rotation=45)
            ax.legend(loc="best", fontsize=8, frameon=False, ncol=2)
        snap_ax = axes[1]
        if snapshot_points:
            labels = [str(row["profile"]) for row in snapshot_points]
            values = [float(row["ending_net_pnl_total"]) for row in snapshot_points]
            colors = []
            for value in values:
                if value > 0:
                    colors.append("#0f766e")
                elif value < 0:
                    colors.append("#b91c1c")
                else:
                    colors.append("#64748b")
            ypos = list(range(len(labels)))
            snap_ax.barh(ypos, values, color=colors, alpha=0.85)
            snap_ax.axvline(0.0, color="#243b53", linewidth=1.0, alpha=0.9)
            snap_ax.set_yticks(ypos)
            snap_ax.set_yticklabels(labels, fontsize=8)
            snap_ax.invert_yaxis()
            snap_ax.set_title("Current Sleeve Snapshot")
            snap_ax.set_xlabel("Ending Net PnL")
            snap_ax.grid(axis="x", linestyle="--", alpha=0.25)
            max_abs = max((abs(v) for v in values), default=0.0)
            pad = max(1.0, max_abs * 0.12)
            snap_ax.set_xlim(min(values + [0.0]) - pad, max(values + [0.0]) + pad)
            for idx, row in enumerate(snapshot_points):
                value = float(row["ending_net_pnl_total"])
                executions = int(row["executions"])
                direction = "left" if value < 0 else "right"
                offset = -6 if value < 0 else 6
                snap_ax.annotate(
                    (
                        f"{value:.2f} | wr {_format_win_rate(row.get('win_rate'))}"
                        f" | {int(row.get('winning_strategy_count', 0) or 0)}W/"
                        f"{int(row.get('losing_strategy_count', 0) or 0)}L/"
                        f"{int(row.get('flat_strategy_count', 0) or 0)}F"
                    ),
                    xy=(value, idx),
                    xytext=(offset, 0),
                    textcoords="offset points",
                    ha=direction,
                    va="center",
                    fontsize=7.5,
                    color="#1f2933",
                )
        fig.tight_layout()
        fig.savefig(sleeves_chart_path, bbox_inches="tight")
        plt.close(fig)

    return {
        "available": bool(weekly_rows or monthly_rows or quarterly_rows),
        "weekly_png": str(weekly_chart_path),
        "monthly_png": str(monthly_chart_path),
        "quarterly_png": str(quarterly_chart_path),
        "sleeves_png": str(sleeves_chart_path),
    }


def render_paper_performance_markdown(payload: dict[str, Any]) -> str:
    day = payload.get("day") if isinstance(payload.get("day"), dict) else {}
    week = payload.get("week") if isinstance(payload.get("week"), dict) else {}
    graphs = payload.get("graphs") if isinstance(payload.get("graphs"), dict) else {}
    sleeve_latest = payload.get("sleeve_latest") if isinstance(payload.get("sleeve_latest"), list) else []

    lines = [
        "# Paper Performance Report",
        "",
        f"- generated_utc: {payload.get('timestamp_utc', '')}",
        f"- source_kind: {payload.get('source_kind', '')}",
        f"- source_files_scanned: {int(payload.get('source_files_scanned', 0) or 0)}",
        "",
        "## End Of Day",
        "",
        f"- day_utc: {day.get('day_utc', '')}",
        f"- available: {bool(day.get('available', False))}",
        f"- executions: {int(day.get('executions', 0) or 0)}",
        f"- buys/sells: {int(day.get('buy_count', 0) or 0)}/{int(day.get('sell_count', 0) or 0)}",
        f"- ending_realized_pnl_total: {float(day.get('ending_realized_pnl_total', 0.0) or 0.0):.6f}",
        f"- ending_unrealized_pnl_total: {float(day.get('ending_unrealized_pnl_total', 0.0) or 0.0):.6f}",
        f"- ending_net_pnl_total: {float(day.get('ending_net_pnl_total', 0.0) or 0.0):.6f}",
        f"- change_vs_previous_day: {float(day.get('change_vs_previous_day', 0.0) or 0.0):.6f}",
        "",
        "## Week",
        "",
        f"- week_start_day_utc: {week.get('week_start_day_utc', '')}",
        f"- week_end_day_utc: {week.get('week_end_day_utc', '')}",
        f"- executions: {int(week.get('executions', 0) or 0)}",
        f"- week_to_date_change: {float(week.get('week_to_date_change', 0.0) or 0.0):.6f}",
        f"- week_to_date_realized_change: {float(week.get('week_to_date_realized_change', 0.0) or 0.0):.6f}",
        f"- rolling_{int(week.get('rolling_change_days', 7) or 7)}d_change: {float(week.get('rolling_change', 0.0) or 0.0):.6f}",
        "",
        "## Graphs",
        "",
        f"- weekly_png: {graphs.get('weekly_png', '')}",
        f"- monthly_png: {graphs.get('monthly_png', '')}",
        f"- quarterly_png: {graphs.get('quarterly_png', '')}",
        f"- sleeves_png: {graphs.get('sleeves_png', '')}",
        "",
        "## Sleeve Progression",
        "",
    ]

    for row in sleeve_latest:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- {row.get('profile', '')}: "
            f"status={row.get('data_status', '')}, "
            f"day={row.get('day_utc', '') or 'n/a'}, "
            f"end_net={float(row.get('ending_net_pnl_total', 0.0) or 0.0):.6f}, "
            f"win_rate={_format_win_rate(row.get('win_rate'))}, "
            f"wins/losses/flats="
            f"{int(row.get('winning_strategy_count', 0) or 0)}/"
            f"{int(row.get('losing_strategy_count', 0) or 0)}/"
            f"{int(row.get('flat_strategy_count', 0) or 0)}, "
            f"best={_format_strategy_pnl_brief(row.get('top_winning_strategies', []), positive=True)}, "
            f"worst={_format_strategy_pnl_brief(row.get('top_losing_strategies', []), positive=False)}, "
            f"realized={float(row.get('ending_realized_pnl_total', 0.0) or 0.0):.6f}, "
            f"unrealized={float(row.get('ending_unrealized_pnl_total', 0.0) or 0.0):.6f}, "
            f"executions={int(row.get('executions', 0) or 0)}"
        )

    lines.extend(
        [
        "",
        "## Daily Series",
        "",
        ]
    )

    for row in week.get("daily_series", []) or []:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- {row.get('day_utc', '')}: end_net={float(row.get('ending_net_pnl_total', 0.0) or 0.0):.6f}, "
            f"change={float(row.get('change_vs_previous_day', 0.0) or 0.0):.6f}"
        )

    return "\n".join(lines).strip() + "\n"


def _path_uri(raw: Any) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    path = Path(text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve().as_uri()


def render_paper_performance_html(payload: dict[str, Any], *, source_path: Path, generated_utc: str) -> str:
    day = payload.get("day") if isinstance(payload.get("day"), dict) else {}
    week = payload.get("week") if isinstance(payload.get("week"), dict) else {}
    graphs = payload.get("graphs") if isinstance(payload.get("graphs"), dict) else {}
    sleeve_latest = payload.get("sleeve_latest") if isinstance(payload.get("sleeve_latest"), list) else []

    chart_specs = [
        ("Weekly Performance", graphs.get("weekly_png", "")),
        ("Monthly Performance", graphs.get("monthly_png", "")),
        ("Quarterly Performance", graphs.get("quarterly_png", "")),
        ("Sleeve Progression", graphs.get("sleeves_png", "")),
    ]
    chart_cards: list[str] = []
    for title, raw_path in chart_specs:
        uri = _path_uri(raw_path)
        if not uri:
            continue
        chart_cards.append(
            "<section class=\"chart-card\">"
            f"<h2>{html.escape(title)}</h2>"
            f"<img src=\"{html.escape(uri)}\" alt=\"{html.escape(title)}\" />"
            "</section>"
        )

    sleeve_rows = []
    for row in sleeve_latest:
        if not isinstance(row, dict):
            continue
        sleeve_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('profile', '')))}</td>"
            f"<td>{html.escape(str(row.get('data_status', '')))}</td>"
            f"<td>{html.escape(str(row.get('day_utc', '') or 'n/a'))}</td>"
            f"<td>{int(row.get('executions', 0) or 0)}</td>"
            f"<td>{html.escape(_format_win_rate(row.get('win_rate')))}</td>"
            f"<td>"
            f"{int(row.get('winning_strategy_count', 0) or 0)}/"
            f"{int(row.get('losing_strategy_count', 0) or 0)}/"
            f"{int(row.get('flat_strategy_count', 0) or 0)}"
            f"</td>"
            f"<td>{html.escape(_format_strategy_pnl_brief(row.get('top_winning_strategies', []), positive=True))}</td>"
            f"<td>{html.escape(_format_strategy_pnl_brief(row.get('top_losing_strategies', []), positive=False))}</td>"
            f"<td>{float(row.get('ending_realized_pnl_total', 0.0) or 0.0):.6f}</td>"
            f"<td>{float(row.get('ending_unrealized_pnl_total', 0.0) or 0.0):.6f}</td>"
            f"<td>{float(row.get('ending_net_pnl_total', 0.0) or 0.0):.6f}</td>"
            "</tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Paper Performance Report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f3efe6;
      --ink: #1f2933;
      --muted: #66737f;
      --card: #fffaf2;
      --line: #d7ccb9;
      --accent: #9a3412;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: linear-gradient(180deg, #efe8db 0%, #f7f3ec 100%); color: var(--ink); font: 15px/1.55 Georgia, 'Times New Roman', serif; }}
    .page {{ max-width: 1040px; margin: 0 auto; padding: 34px 24px 48px; }}
    .hero, .section, .chart-card {{ background: var(--card); border: 1px solid var(--line); border-radius: 18px; box-shadow: 0 10px 26px rgba(31, 41, 51, 0.08); }}
    .hero {{ padding: 24px 26px; }}
    .section {{ margin-top: 18px; padding: 18px 22px; }}
    .chart-grid {{ display: grid; grid-template-columns: 1fr; gap: 18px; margin-top: 18px; }}
    .chart-card {{ padding: 16px 18px; }}
    h1, h2, h3 {{ margin: 0 0 10px; font-family: 'Avenir Next', 'Segoe UI', sans-serif; }}
    h1 {{ font-size: 30px; }}
    h2 {{ font-size: 20px; }}
    p.meta {{ margin: 0; color: var(--muted); }}
    .path {{ margin-top: 10px; font: 12px/1.4 'SF Mono', 'Menlo', monospace; color: var(--accent); word-break: break-all; }}
    .stats {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
    .stat-block {{ background: #f7f1e7; border: 1px solid #eadfce; border-radius: 14px; padding: 14px 16px; }}
    .stat-block p {{ margin: 6px 0 0; }}
    img {{ width: 100%; height: auto; display: block; border-radius: 12px; border: 1px solid #eadfce; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px 6px; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-family: 'Avenir Next', 'Segoe UI', sans-serif; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Paper Performance Report</h1>
      <p class="meta">PDF generated {html.escape(generated_utc)}</p>
      <p class="path">Source: {html.escape(str(source_path))}</p>
    </section>
    <section class="section">
      <div class="stats">
        <div class="stat-block">
          <h2>End Of Day</h2>
          <p>day_utc: {html.escape(str(day.get('day_utc', '')))}</p>
          <p>executions: {int(day.get('executions', 0) or 0)}</p>
          <p>ending_net_pnl_total: {float(day.get('ending_net_pnl_total', 0.0) or 0.0):.6f}</p>
          <p>change_vs_previous_day: {float(day.get('change_vs_previous_day', 0.0) or 0.0):.6f}</p>
        </div>
        <div class="stat-block">
          <h2>Week</h2>
          <p>week_start_day_utc: {html.escape(str(week.get('week_start_day_utc', '')))}</p>
          <p>week_end_day_utc: {html.escape(str(week.get('week_end_day_utc', '')))}</p>
          <p>week_to_date_change: {float(week.get('week_to_date_change', 0.0) or 0.0):.6f}</p>
          <p>rolling_{int(week.get('rolling_change_days', 7) or 7)}d_change: {float(week.get('rolling_change', 0.0) or 0.0):.6f}</p>
        </div>
      </div>
    </section>
    <div class="chart-grid">
      {''.join(chart_cards)}
    </div>
    <section class="section">
      <h2>Sleeve Snapshot</h2>
      <table>
        <thead>
          <tr>
            <th>Sleeve</th>
            <th>Status</th>
            <th>Data Day</th>
            <th>Executions</th>
            <th>Win Rate</th>
            <th>W/L/F</th>
            <th>Top Winners</th>
            <th>Top Losers</th>
            <th>Realized</th>
            <th>Unrealized</th>
            <th>Ending Net</th>
          </tr>
        </thead>
        <tbody>
          {''.join(sleeve_rows)}
        </tbody>
      </table>
    </section>
  </div>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Paper trading performance snapshot and week-to-date report.")
    ap.add_argument("--day", default=_utc_now().strftime("%Y%m%d"))
    ap.add_argument("--week-days", type=int, default=7)
    ap.add_argument("--out-file", default=str(DEFAULT_JSON_PATH))
    ap.add_argument("--md-out-file", default=str(DEFAULT_MD_PATH))
    ap.add_argument("--html-out-file", default=str(DEFAULT_HTML_PATH))
    ap.add_argument("--pdf-out-file", default=str(DEFAULT_PDF_PATH))
    ap.add_argument("--weekly-chart-file", default=str(DEFAULT_WEEKLY_CHART_PATH))
    ap.add_argument("--monthly-chart-file", default=str(DEFAULT_MONTHLY_CHART_PATH))
    ap.add_argument("--quarterly-chart-file", default=str(DEFAULT_QUARTERLY_CHART_PATH))
    ap.add_argument("--sleeves-chart-file", default=str(DEFAULT_SLEEVES_CHART_PATH))
    ap.add_argument("--allow-gui-pdf-renderer", action=argparse.BooleanOptionalAction, default=_default_allow_gui_pdf_renderer())
    ap.add_argument("--json-only", action="store_true", help="Write the JSON snapshot only and skip charts/markdown/html/pdf artifacts.")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    payload = build_paper_performance_report(PROJECT_ROOT, day=str(args.day), week_days=max(int(args.week_days), 1))

    out_path = Path(args.out_file)
    md_path = Path(args.md_out_file)
    html_path = Path(args.html_out_file)
    pdf_path = Path(args.pdf_out_file)
    weekly_chart_path = Path(args.weekly_chart_file)
    monthly_chart_path = Path(args.monthly_chart_file)
    quarterly_chart_path = Path(args.quarterly_chart_file)
    sleeves_chart_path = Path(args.sleeves_chart_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.json_only:
        payload["graphs"] = {
            "mode": "json_only",
            "weekly_png": "",
            "monthly_png": "",
            "quarterly_png": "",
            "sleeves_png": "",
        }
        payload["pdf"] = {
            "available": False,
            "html_report_path": "",
            "pdf_path": str(pdf_path),
            "detail": "skipped_json_only",
        }
    else:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        payload["graphs"] = render_paper_performance_graphs(
            payload,
            weekly_chart_path=weekly_chart_path,
            monthly_chart_path=monthly_chart_path,
            quarterly_chart_path=quarterly_chart_path,
            sleeves_chart_path=sleeves_chart_path,
        )
        generated_utc = str(payload.get("timestamp_utc") or _utc_now().isoformat())
        md_text = render_paper_performance_markdown(payload)
        html_text = render_paper_performance_html(payload, source_path=out_path, generated_utc=generated_utc)
        md_path.write_text(md_text, encoding="utf-8")
        html_path.write_text(html_text, encoding="utf-8")
        if pdf_path.exists():
            pdf_path.unlink()
        pdf_ok, pdf_detail = _render_pdf_from_html(
            html_path,
            pdf_path,
            allow_gui_renderer=bool(args.allow_gui_pdf_renderer),
        )
        payload["pdf"] = {
            "available": bool(pdf_ok),
            "html_report_path": str(html_path),
            "pdf_path": str(pdf_path),
            "detail": str(pdf_detail),
        }
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        day_summary = payload.get("day", {}) if isinstance(payload.get("day"), dict) else {}
        week_summary = payload.get("week", {}) if isinstance(payload.get("week"), dict) else {}
        print(
            "paper_performance "
            f"day={day_summary.get('day_utc', '')} "
            f"eod_net={float(day_summary.get('ending_net_pnl_total', 0.0) or 0.0):.4f} "
            f"day_change={float(day_summary.get('change_vs_previous_day', 0.0) or 0.0):.4f} "
            f"wtd_change={float(week_summary.get('week_to_date_change', 0.0) or 0.0):.4f}"
        )
    return 0 if payload.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
