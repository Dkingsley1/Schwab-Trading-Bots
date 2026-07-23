#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "exports" / "reports"
HEALTH_DIR = PROJECT_ROOT / "governance" / "health"
INTAKE_DIR = PROJECT_ROOT / "governance" / "training_labeling_intelligence"

DEFAULT_PAPER = HEALTH_DIR / "paper_performance_latest.json"
DEFAULT_CONTROL = HEALTH_DIR / "paper_profitability_control_latest.json"
DEFAULT_RUNTIME = HEALTH_DIR / "paper_runtime_profitability_controls_latest.json"
DEFAULT_HEALTH = HEALTH_DIR / "health_fast_latest.json"
DEFAULT_BROKER = HEALTH_DIR / "broker_readiness_latest.json"
DEFAULT_INTAKE = INTAKE_DIR / "data_intake_focus_latest.json"
DEFAULT_PDF = REPORTS_DIR / "paper_trading_summary_latest.pdf"
DEFAULT_JSON = HEALTH_DIR / "paper_trading_summary_pdf_latest.json"

COLORS = {
    "ink": "#172033",
    "muted": "#5b6776",
    "line": "#d8e1e7",
    "paper": "#f8fbfc",
    "teal": "#247f7c",
    "blue": "#4b73a8",
    "green": "#1e7b50",
    "gold": "#a97916",
    "red": "#ae4343",
    "purple": "#6252bd",
    "white": "#ffffff",
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _clean(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text.replace("`", "")


def _date_label(day_utc: Any) -> str:
    text = re.sub(r"\D", "", str(day_utc or ""))
    if len(text) != 8:
        return str(day_utc or "")
    return f"{text[:4]}-{text[4:6]}-{text[6:]}"


def _money(value: Any) -> str:
    number = _safe_float(value)
    sign = "+" if number > 0 else ""
    return f"{sign}\\${number:,.2f}"


def _number(value: Any) -> str:
    return f"{_safe_int(value):,}"


def _pct(value: Any) -> str:
    number = _safe_float(value, float("nan"))
    if not math.isfinite(number):
        return "n/a"
    return f"{number * 100:.1f}%"


def _short_strategy(value: Any, limit: int = 42) -> str:
    text = _clean(value)
    for prefix in ("paper_mirror_options::", "paper_mirror_futures::", "paper_mirror::"):
        text = text.replace(prefix, "")
    return textwrap.shorten(text, width=limit, placeholder="...")


def _unique_strategy_controls(control: dict[str, Any]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    rows: list[dict[str, Any]] = []
    for row in _as_list(control.get("strategy_controls")):
        if not isinstance(row, dict):
            continue
        key = (
            str(row.get("profile") or ""),
            str(row.get("strategy") or ""),
            str(row.get("bot_id") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    rows.sort(key=lambda item: _safe_float(item.get("ending_net_pnl_total"), 0.0))
    return rows


def _sleeve_coverage_by_day(paper: dict[str, Any]) -> dict[str, dict[str, int]]:
    sleeve_series = _as_dict(paper.get("sleeve_daily_series"))
    total_profiles = len(sleeve_series)
    coverage: dict[str, dict[str, int]] = {}
    for profile, series in sleeve_series.items():
        if not profile:
            continue
        for row in _as_list(series):
            if not isinstance(row, dict):
                continue
            day = str(row.get("day_utc") or "").strip()
            if not day:
                continue
            record = coverage.setdefault(
                day,
                {
                    "represented_profile_count": 0,
                    "pnl_profile_count": 0,
                    "total_profile_count": total_profiles,
                },
            )
            record["represented_profile_count"] += 1
            has_pnl_state = any(
                abs(_safe_float(row.get(field), 0.0)) > 0.000001
                for field in (
                    "ending_net_pnl_total",
                    "ending_realized_pnl_total",
                    "ending_unrealized_pnl_total",
                    "change_vs_previous_day",
                )
            )
            if has_pnl_state:
                record["pnl_profile_count"] += 1
    return coverage


def _coverage_stats(paper: dict[str, Any]) -> dict[str, int]:
    coverage = _sleeve_coverage_by_day(paper)
    day = str(_as_dict(paper.get("day")).get("day_utc") or "")
    latest = coverage.get(day, {})
    total = _safe_int(latest.get("total_profile_count"), len(_as_dict(paper.get("sleeve_daily_series"))))
    active_running = _safe_int(paper.get("active_paper_profile_count_today"), 0)
    pnl_profiles = _safe_int(latest.get("pnl_profile_count"), 0)
    represented = _safe_int(latest.get("represented_profile_count"), 0)
    partial_days = sum(
        1
        for row in _as_list(paper.get("history_daily_series"))
        if isinstance(row, dict)
        and _safe_int(coverage.get(str(row.get("day_utc") or ""), {}).get("pnl_profile_count"), 0) < total
    )
    return {
        "total_profile_count": total,
        "active_running_profile_count": active_running,
        "latest_pnl_profile_count": pnl_profiles,
        "latest_represented_profile_count": represented,
        "partial_coverage_day_count": partial_days,
    }


def _new_page(title: str, subtitle: str = ""):
    fig = plt.figure(figsize=(8.5, 11), facecolor=COLORS["paper"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    fig.patches.append(Rectangle((0, 0.93), 1, 0.07, transform=fig.transFigure, color=COLORS["ink"], zorder=-1))
    fig.patches.append(Rectangle((0, 0), 1, 0.035, transform=fig.transFigure, color="#e5ebef", zorder=-1))
    fig.text(0.065, 0.957, title, fontsize=16, color=COLORS["white"], weight="bold", va="center")
    if subtitle:
        fig.text(0.065, 0.915, subtitle, fontsize=9.2, color=COLORS["muted"], va="top")
    return fig


def _save(pdf: PdfPages, fig, page_number: int, title: str) -> int:
    fig.text(0.065, 0.02, f"{title} - page {page_number}", fontsize=8.2, color=COLORS["muted"], va="center")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return page_number + 1


def _wrapped(value: Any, width: int) -> list[str]:
    text = _clean(value)
    if not text:
        return []
    return textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False) or [text]


def _body(fig, x: float, y: float, text: Any, *, width: int = 88, size: float = 9.0, color: str = COLORS["ink"], bullet: bool = False) -> float:
    lines = _wrapped(text, width)
    if not lines:
        return y
    prefix = "- " if bullet else ""
    fig.text(x, y, prefix + lines[0], fontsize=size, color=color, va="top")
    y -= 0.019
    for line in lines[1:]:
        fig.text(x + (0.018 if bullet else 0.0), y, line, fontsize=size, color=color, va="top")
        y -= 0.019
    return y


def _heading(fig, y: float, text: Any, *, x: float = 0.065, color: str = COLORS["teal"], size: float = 12.0) -> float:
    fig.text(x, y, _clean(text), fontsize=size, color=color, weight="bold", va="top")
    return y - 0.033


def _metric(fig, x: float, y: float, label: str, value: str, detail: str = "", color: str = COLORS["purple"]) -> None:
    fig.patches.append(Rectangle((x, y - 0.078), 0.275, 0.070, transform=fig.transFigure, facecolor=COLORS["white"], edgecolor=COLORS["line"], linewidth=1))
    fig.text(x + 0.012, y - 0.018, label, fontsize=7.9, color=COLORS["muted"], weight="bold", va="top")
    value_size = 10.0 if len(value) > 17 else 13.0
    fig.text(x + 0.012, y - 0.043, value, fontsize=value_size, color=color, weight="bold", va="top")
    if detail:
        fig.text(x + 0.012, y - 0.063, detail, fontsize=7.1, color=COLORS["muted"], va="top")


def _draw_table(
    fig,
    *,
    x: float,
    y: float,
    columns: list[tuple[str, float]],
    rows: list[list[str]],
    row_height: float = 0.036,
    size: float = 7.3,
    max_rows: int = 14,
) -> float:
    width = sum(col_width for _, col_width in columns)
    fig.patches.append(Rectangle((x, y - 0.030), width, 0.030, transform=fig.transFigure, facecolor=COLORS["ink"], edgecolor=COLORS["ink"], linewidth=0.8))
    cx = x
    for label, col_width in columns:
        fig.text(cx + 0.006, y - 0.008, label, fontsize=7.5, color=COLORS["white"], weight="bold", va="top")
        cx += col_width
    y -= 0.030
    for idx, row in enumerate(rows[:max_rows]):
        fill = COLORS["white"] if idx % 2 == 0 else "#eef4f6"
        fig.patches.append(Rectangle((x, y - row_height), width, row_height, transform=fig.transFigure, facecolor=fill, edgecolor=COLORS["line"], linewidth=0.6))
        cx = x
        for value, (_, col_width) in zip(row, columns):
            fig.text(cx + 0.006, y - 0.010, value, fontsize=size, color=COLORS["ink"], va="top")
            cx += col_width
        y -= row_height
    return y


def _cover_page(pdf: PdfPages, data: dict[str, Any], page_number: int) -> int:
    paper = data["paper"]
    control = data["control"]
    broker = data["broker"]
    health = data["health"]
    day = _as_dict(paper.get("day"))
    week = _as_dict(paper.get("week"))
    current = _as_dict(_as_dict(control.get("a_plus_target_contract")).get("current"))
    coverage = data["coverage_stats"]
    generated = data["generated_utc"]
    latest_day = _date_label(day.get("day_utc"))
    subtitle = f"Generated UTC: {generated} | Latest complete paper report: {latest_day or 'unknown'}"
    fig = _new_page("Paper Trading Performance Summary", subtitle)
    fig.text(0.065, 0.865, "Executive Readout", fontsize=22, color=COLORS["ink"], weight="bold", va="top")
    fig.text(
        0.065,
        0.825,
        "Paper trading is profitable on the latest complete report, with recovery controls active for weak sleeves and conditional strategy pairs.",
        fontsize=10.2,
        color=COLORS["muted"],
        va="top",
    )

    metrics = [
        ("Day Net P&L", _money(day.get("ending_net_pnl_total")), f"{_number(day.get('executions'))} execs", COLORS["green"]),
        ("Day Change", _money(day.get("change_vs_previous_day")), "vs prior paper day", COLORS["green"]),
        ("Realized P&L", _money(day.get("ending_realized_pnl_total")), "ending total", COLORS["blue"]),
        ("WTD Change", _money(week.get("week_to_date_change")), f"through {_date_label(week.get('week_end_day_utc'))}", COLORS["green"]),
        (
            "P&L Coverage",
            f"{_number(coverage.get('latest_pnl_profile_count'))}/{_number(coverage.get('total_profile_count'))}",
            f"{_number(coverage.get('active_running_profile_count'))} profiles running",
            COLORS["purple"],
        ),
        ("Grade", str(control.get("profitability_display_grade") or control.get("profitability_grade") or "n/a").replace("controlled", "ctrl"), "", COLORS["gold"]),
    ]
    for idx, metric in enumerate(metrics):
        row = idx // 3
        col = idx % 3
        _metric(fig, 0.065 + col * 0.300, 0.760 - row * 0.100, *metric)

    y = 0.535
    y = _heading(fig, y, "What The Numbers Say")
    bullets = [
        f"Latest complete paper day ({latest_day}) ended at {_money(day.get('ending_net_pnl_total'))} net P&L after {_number(day.get('executions'))} executions.",
        f"Coverage matters: latest P&L came from {_number(coverage.get('latest_pnl_profile_count'))} P&L-bearing sleeve out of {_number(coverage.get('total_profile_count'))} configured sleeves, with {_number(coverage.get('active_running_profile_count'))} profiles running.",
        f"The week ending {_date_label(week.get('week_end_day_utc'))} added {_money(week.get('week_to_date_change'))}; the 30-day change is {_money(data['periods'].get('30D'))}.",
        "The raw financial grade is evidence-based; the A+ displayed grade is a controlled recovery posture from containment, not permission to widen live risk.",
        "Schwab auth and broker readiness are clear; live execution remains intentionally disabled behind operator controls.",
    ]
    for bullet in bullets:
        y = _body(fig, 0.078, y, bullet, width=92, size=9.3, color=COLORS["ink"], bullet=True)
        y -= 0.010

    y -= 0.020
    y = _heading(fig, y, "Operational Snapshot")
    ops = [
        f"Health-fast status: {_clean(health.get('overall_status') or 'unknown')} | global halt: {_clean(_as_dict(health.get('global_halt')).get('halt_state') or 'unknown')}.",
        f"Broker readiness: auth_ok={bool(broker.get('auth_ok'))}, network_ok={bool(broker.get('network_ok'))}, ready_for_open={bool(broker.get('ready_for_open'))}.",
        f"Weak profiles under protection: {_number(_as_dict(control.get('weak_sleeve_a_plus_plus_strengthening_contract')).get('weak_profile_count'))}; losing strategy pairs under rehab: {_number(len(_unique_strategy_controls(control)))}.",
    ]
    for item in ops:
        y = _body(fig, 0.078, y, item, width=92, size=8.9, color=COLORS["muted"], bullet=True)
        y -= 0.006
    return _save(pdf, fig, page_number, "Paper Trading Summary")


def _performance_page(pdf: PdfPages, data: dict[str, Any], page_number: int) -> int:
    paper = data["paper"]
    daily = [row for row in _as_list(paper.get("history_daily_series")) if isinstance(row, dict)]
    coverage_by_day = data["coverage_by_day"]
    coverage_stats = data["coverage_stats"]
    periods = data["periods"]
    fig = _new_page("Coverage-Aware Performance Trend", "Daily P&L is marked against sleeve participation so inactive/partial days are not over-read")
    fig.text(
        0.075,
        0.865,
        "Grey bands mark days where not all configured sleeves had P&L-bearing paper state. Treat those as coverage gaps, not pure strategy drawdown or recovery.",
        fontsize=8.5,
        color=COLORS["muted"],
        va="top",
    )

    daily_rows = daily[-14:]
    x_values = list(range(len(daily_rows)))
    dates = [_date_label(row.get("day_utc"))[5:] for row in daily_rows]
    day_keys = [str(row.get("day_utc") or "") for row in daily_rows]
    net_values = [_safe_float(row.get("ending_net_pnl_total")) for row in daily_rows]
    change_values = [_safe_float(row.get("change_vs_previous_day")) for row in daily_rows]
    execution_values = [_safe_int(row.get("executions")) for row in daily_rows]
    total_profiles = max(_safe_int(coverage_stats.get("total_profile_count"), 0), 1)
    pnl_profile_counts = [
        _safe_int(coverage_by_day.get(day, {}).get("pnl_profile_count"), 0)
        for day in day_keys
    ]
    partial_days = [count < total_profiles for count in pnl_profile_counts]

    ax1 = fig.add_axes([0.075, 0.595, 0.850, 0.240], facecolor=COLORS["white"])
    for idx, partial in enumerate(partial_days):
        if partial:
            ax1.axvspan(idx - 0.45, idx + 0.45, color=COLORS["line"], alpha=0.30, linewidth=0)
    ax1.plot(x_values, net_values, color=COLORS["teal"], marker="o", linewidth=2.0, label="Ending net P&L")
    ax1.axhline(0, color=COLORS["line"], linewidth=1)
    ax1.set_title("Daily Ending Net P&L (coverage-marked)", loc="left", fontsize=10, color=COLORS["ink"], weight="bold")
    ax1.set_xticks(x_values)
    ax1.set_xticklabels(dates)
    ax1.tick_params(axis="x", labelrotation=35, labelsize=7)
    ax1.tick_params(axis="y", labelsize=7)
    ax1.grid(axis="y", color=COLORS["line"], linewidth=0.6)

    ax2 = fig.add_axes([0.075, 0.340, 0.850, 0.195], facecolor=COLORS["white"])
    for idx, partial in enumerate(partial_days):
        if partial:
            ax2.axvspan(idx - 0.45, idx + 0.45, color=COLORS["line"], alpha=0.30, linewidth=0)
    bar_colors = [COLORS["green"] if value >= 0 else COLORS["red"] for value in change_values]
    ax2.bar(x_values, change_values, color=bar_colors, width=0.65)
    ax2.axhline(0, color=COLORS["ink"], linewidth=0.8)
    ax2.set_title("Daily Change Vs Previous Paper Day (not all-sleeve comparable on grey days)", loc="left", fontsize=10, color=COLORS["ink"], weight="bold")
    ax2.set_xticks(x_values)
    ax2.set_xticklabels(dates)
    ax2.tick_params(axis="x", labelrotation=35, labelsize=7)
    ax2.tick_params(axis="y", labelsize=7)
    ax2.grid(axis="y", color=COLORS["line"], linewidth=0.6)

    ax3 = fig.add_axes([0.075, 0.105, 0.395, 0.165], facecolor=COLORS["white"])
    ax3.bar(x_values, pnl_profile_counts, color=COLORS["blue"], width=0.65)
    ax3.axhline(total_profiles, color=COLORS["muted"], linestyle="--", linewidth=0.8)
    ax3.set_ylim(0, max(total_profiles + 1, max(pnl_profile_counts or [0]) + 1))
    ax3.set_title("P&L-Bearing Sleeves / Configured Sleeves", loc="left", fontsize=9.5, color=COLORS["ink"], weight="bold")
    ax3.set_xticks(x_values)
    ax3.set_xticklabels(dates)
    ax3.tick_params(axis="x", labelrotation=35, labelsize=6.8)
    ax3.tick_params(axis="y", labelsize=6.8)
    ax3.grid(axis="y", color=COLORS["line"], linewidth=0.5)
    ax3b = ax3.twinx()
    ax3b.plot(x_values, execution_values, color=COLORS["gold"], linewidth=1.2, marker=".", alpha=0.85)
    ax3b.tick_params(axis="y", labelsize=6.5, colors=COLORS["gold"])
    ax3b.set_ylabel("Execs", fontsize=6.5, color=COLORS["gold"])

    ax4 = fig.add_axes([0.535, 0.105, 0.390, 0.165], facecolor=COLORS["white"])
    labels = list(periods.keys())
    values = [periods[label] for label in labels]
    ax4.bar(labels, values, color=[COLORS["green"] if value >= 0 else COLORS["red"] for value in values])
    ax4.axhline(0, color=COLORS["ink"], linewidth=0.8)
    ax4.set_title("Rolling Period Change", loc="left", fontsize=9.5, color=COLORS["ink"], weight="bold")
    ax4.tick_params(axis="x", labelsize=7)
    ax4.tick_params(axis="y", labelsize=6.8)
    ax4.grid(axis="y", color=COLORS["line"], linewidth=0.5)
    return _save(pdf, fig, page_number, "Paper Trading Summary")


def _sleeve_page(pdf: PdfPages, data: dict[str, Any], page_number: int) -> int:
    paper = data["paper"]
    control = data["control"]
    profile_controls = _as_dict(control.get("active_profile_controls"))
    rows = []
    sleeves = [row for row in _as_list(paper.get("sleeve_latest")) if isinstance(row, dict)]
    sleeves.sort(key=lambda row: (_safe_float(row.get("ending_net_pnl_total")), _safe_int(row.get("executions"))), reverse=True)
    for row in sleeves:
        profile = str(row.get("profile") or "")
        ctl = _as_dict(profile_controls.get(profile))
        action = str(ctl.get("action") or ("monitor" if _safe_float(row.get("ending_net_pnl_total")) >= 0 else "watch"))
        rows.append(
            [
                textwrap.shorten(profile, width=18, placeholder="..."),
                _date_label(row.get("day_utc")) or "n/a",
                textwrap.shorten(str(row.get("data_status") or ""), width=18, placeholder="..."),
                _number(row.get("executions")),
                _money(row.get("ending_net_pnl_total")),
                _money(row.get("ending_realized_pnl_total")),
                _money(row.get("ending_unrealized_pnl_total")),
                _pct(row.get("win_rate")),
                textwrap.shorten(action, width=20, placeholder="..."),
            ]
        )

    fig = _new_page("Sleeve And Strategy Surface", "Latest sleeve snapshots and active paper controls")
    y = 0.875
    y = _heading(fig, y, "Sleeve Snapshot")
    columns = [
        ("Profile", 0.130),
        ("Day", 0.073),
        ("Status", 0.115),
        ("Execs", 0.070),
        ("Net", 0.090),
        ("Realized", 0.092),
        ("Unrealized", 0.092),
        ("Win", 0.055),
        ("Control", 0.145),
    ]
    y = _draw_table(fig, x=0.055, y=y, columns=columns, rows=rows, row_height=0.034, size=6.6, max_rows=14)

    y -= 0.035
    y = _heading(fig, y, "Top Current Drivers", size=11.2)
    day = _as_dict(paper.get("day"))
    top_symbols = ", ".join(f"{item.get('name')} ({_number(item.get('executions'))})" for item in _as_list(day.get("top_symbols"))[:5] if isinstance(item, dict))
    top_strategies = ", ".join(f"{_short_strategy(item.get('name'), 34)} ({_number(item.get('executions'))})" for item in _as_list(day.get("top_strategies"))[:4] if isinstance(item, dict))
    notes = [
        f"Most active symbols on the latest complete day: {top_symbols or 'n/a'}.",
        f"Most active strategies: {top_strategies or 'n/a'}.",
        "Only the default sleeve had meaningful current-day paper activity; weak sleeves are visible but kept under protective controls.",
    ]
    for note in notes:
        y = _body(fig, 0.078, y, note, width=95, size=8.5, color=COLORS["muted"], bullet=True)
        y -= 0.005
    return _save(pdf, fig, page_number, "Paper Trading Summary")


def _rehab_page(pdf: PdfPages, data: dict[str, Any], page_number: int) -> int:
    control = data["control"]
    rows = []
    for row in _unique_strategy_controls(control):
        rehab = _as_dict(row.get("rehabilitation_contract"))
        session_gate = _as_dict(rehab.get("session_gate"))
        required = _as_list(rehab.get("required_before_reentry"))
        rows.append(
            [
                textwrap.shorten(str(row.get("profile") or ""), width=18, placeholder="..."),
                _short_strategy(row.get("strategy"), 36),
                _money(row.get("ending_net_pnl_total")),
                textwrap.shorten(str(rehab.get("focus_family") or "general"), width=25, placeholder="..."),
                textwrap.shorten(str(session_gate.get("mode") or ""), width=30, placeholder="..."),
                textwrap.shorten(", ".join(required[:4]), width=48, placeholder="..."),
            ]
        )

    fig = _new_page("Strategy Rehabilitation Plan", "Losing pairs are treated as conditional-fit candidates, not dead code")
    y = 0.875
    y = _heading(fig, y, "Paper-Only Rehab Contracts")
    columns = [
        ("Profile", 0.115),
        ("Strategy", 0.225),
        ("Net", 0.075),
        ("Repair Focus", 0.150),
        ("Session Gate", 0.175),
        ("Reentry Evidence", 0.255),
    ]
    y = _draw_table(fig, x=0.040, y=y, columns=columns, rows=rows, row_height=0.050, size=6.4, max_rows=10)

    y -= 0.040
    y = _heading(fig, y, "What Changed Today", size=11.2)
    notes = [
        "Every losing profile-strategy pair now has a rehabilitation contract with paper-only retests, session gates, and quality evidence gates.",
        "Common loss causes are now explicitly labeled: low source quality, unknown fill quality, low event proximity, portfolio conflict, and poor session fit.",
        "Reentry requires three profitable refreshes, positive pair-level paper results, source/fill/spread evidence, regime applicability labels, and portfolio-conflict clearance.",
    ]
    for note in notes:
        y = _body(fig, 0.078, y, note, width=92, size=8.8, color=COLORS["muted"], bullet=True)
        y -= 0.008
    return _save(pdf, fig, page_number, "Paper Trading Summary")


def _controls_page(pdf: PdfPages, data: dict[str, Any], page_number: int) -> int:
    control = data["control"]
    intake = data["intake"]
    scout = _as_dict(control.get("scout_collection_contract"))
    hardening = _as_dict(control.get("paper_profitability_hardening_contract"))
    labels = _as_list(scout.get("required_label_outputs"))
    contexts = _as_list(scout.get("required_context"))
    focus_records = _as_list(intake.get("focus_records"))
    trainable = _as_list(intake.get("trainable_candidates"))
    collect_first = _as_list(intake.get("collect_first_top"))

    fig = _new_page("Controls And Next Validation", "What remains before widening paper risk")
    y = 0.875
    y = _heading(fig, y, "Active Guardrails")
    guardrails = [
        f"Overall control status: {_clean(control.get('overall_status') or 'unknown')}; controlled grade: {_clean(control.get('profitability_display_grade') or control.get('profitability_grade') or 'n/a')}.",
        f"Estimated strategy-pair drag: {_money(hardening.get('estimated_strategy_pair_drag'))}; new entries blocked for quarantined weak sleeves and losing pairs.",
        f"Scout contract targets {_number(len(_as_list(scout.get('target_bot_ids'))))} bot ids and {_number(len(_as_list(scout.get('target_profiles'))))} profiles.",
        f"Training intake is in {str(intake.get('mode') or 'unknown')} mode with {_number(len(focus_records))} focus records, {_number(len(trainable))} trainable candidates, and {_number(len(collect_first))} collect-first records in the latest focus artifact.",
    ]
    for item in guardrails:
        y = _body(fig, 0.078, y, item, width=92, size=8.9, color=COLORS["ink"], bullet=True)
        y -= 0.008

    y -= 0.015
    y = _heading(fig, y, "New Rehab Data Requirements")
    label_text = ", ".join(str(item) for item in labels if item in {"strategy_reentry_retest_outcome", "strategy_regime_applicability_bucket", "session_gate_result", "source_fill_spread_quality_bucket", "independent_evidence_channel_count"})
    context_text = ", ".join(str(item) for item in contexts if item in {"strategy_reentry_attempt", "session_calendar", "market_regime_snapshot", "source_quality_snapshot", "fill_spread_snapshot", "portfolio_conflict_snapshot"})
    y = _body(fig, 0.078, y, f"Labels now required: {label_text or 'n/a'}.", width=92, size=8.8, color=COLORS["muted"], bullet=True)
    y -= 0.006
    y = _body(fig, 0.078, y, f"Context now required: {context_text or 'n/a'}.", width=92, size=8.8, color=COLORS["muted"], bullet=True)

    y -= 0.040
    y = _heading(fig, y, "Next Clean-Room Checks")
    checks = [
        "Refresh paper performance after the next full trading day and rerun paper-profitability-control --apply.",
        "Verify each rehabilitated strategy has session_gate_result, strategy_regime_applicability_bucket, and source_fill_spread_quality_bucket before any reentry.",
        "Keep live execution blocked; these controls are paper-repair and training-data contracts only.",
        "Promote only after clean paper refreshes reduce weak-profile count and strategy-pair drag to zero.",
    ]
    for check in checks:
        y = _body(fig, 0.078, y, check, width=92, size=8.8, color=COLORS["muted"], bullet=True)
        y -= 0.008
    return _save(pdf, fig, page_number, "Paper Trading Summary")


def _periods(paper: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for row in _as_list(paper.get("period_change_series")):
        if not isinstance(row, dict):
            continue
        label = str(row.get("label") or "").strip()
        if label:
            values[label] = _safe_float(row.get("change"))
    for label in ("WTD", "7D", "14D", "21D", "30D"):
        values.setdefault(label, 0.0)
    return {label: values[label] for label in ("WTD", "7D", "14D", "21D", "30D")}


def build_pdf(data: dict[str, Any], pdf_path: Path) -> int:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    page_number = 1
    with PdfPages(pdf_path) as pdf:
        page_number = _cover_page(pdf, data, page_number)
        page_number = _performance_page(pdf, data, page_number)
        page_number = _sleeve_page(pdf, data, page_number)
        page_number = _rehab_page(pdf, data, page_number)
        page_number = _controls_page(pdf, data, page_number)
    return page_number - 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a detailed paper-trading performance summary PDF.")
    parser.add_argument("--paper", default=str(DEFAULT_PAPER))
    parser.add_argument("--control", default=str(DEFAULT_CONTROL))
    parser.add_argument("--runtime", default=str(DEFAULT_RUNTIME))
    parser.add_argument("--health", default=str(DEFAULT_HEALTH))
    parser.add_argument("--broker", default=str(DEFAULT_BROKER))
    parser.add_argument("--intake", default=str(DEFAULT_INTAKE))
    parser.add_argument("--pdf", default=str(DEFAULT_PDF))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON))
    parser.add_argument("--timestamped", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    generated = datetime.now(timezone.utc)
    pdf_path = Path(args.pdf).expanduser()
    if args.timestamped:
        stem = pdf_path.stem.replace("_latest", "")
        pdf_path = pdf_path.with_name(f"{stem}_{generated.strftime('%Y%m%d_%H%M%S')}.pdf")

    paper = _load_json(Path(args.paper).expanduser())
    control = _load_json(Path(args.control).expanduser())
    data = {
        "paper": paper,
        "control": control,
        "runtime": _load_json(Path(args.runtime).expanduser()),
        "health": _load_json(Path(args.health).expanduser()),
        "broker": _load_json(Path(args.broker).expanduser()),
        "intake": _load_json(Path(args.intake).expanduser()),
        "periods": _periods(paper),
        "coverage_by_day": _sleeve_coverage_by_day(paper),
        "coverage_stats": _coverage_stats(paper),
        "generated_utc": generated.isoformat(),
    }
    pages = build_pdf(data, pdf_path)
    latest_path = Path(args.pdf).expanduser()
    if pdf_path != latest_path:
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pdf_path, latest_path)

    payload = {
        "timestamp_utc": generated.isoformat(),
        "ok": bool(pdf_path.exists() and pdf_path.stat().st_size > 20_000),
        "pdf_path": str(pdf_path),
        "latest_pdf_path": str(latest_path),
        "pdf_bytes": int(pdf_path.stat().st_size) if pdf_path.exists() else 0,
        "latest_pdf_bytes": int(latest_path.stat().st_size) if latest_path.exists() else 0,
        "page_count": pages,
        "paper_day_utc": _as_dict(paper.get("day")).get("day_utc", ""),
        "coverage_stats": data["coverage_stats"],
        "renderer": "matplotlib_pdfpages",
        "source_files": {
            "paper": str(Path(args.paper).expanduser()),
            "control": str(Path(args.control).expanduser()),
            "runtime": str(Path(args.runtime).expanduser()),
            "health": str(Path(args.health).expanduser()),
            "broker": str(Path(args.broker).expanduser()),
            "intake": str(Path(args.intake).expanduser()),
        },
    }
    out = Path(args.json_out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(f"paper_trading_summary_pdf ok={int(payload['ok'])} pages={pages} path={latest_path}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
