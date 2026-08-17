#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import html
import json
import math
import re
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import FuncFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
REPORTS_DIR = PROJECT_ROOT / "exports" / "reports"
SQL_REPORTS_DIR = PROJECT_ROOT / "exports" / "sql_reports"
GOVERNANCE_DIR = PROJECT_ROOT / "governance" / "health"
CATALOG_HTML = REPORTS_DIR / "report_pdf_bundle_latest.html"
CATALOG_PDF = REPORTS_DIR / "report_pdf_bundle_latest.pdf"
CATALOG_JSON = GOVERNANCE_DIR / "report_pdf_bundle_latest.json"


@dataclass(frozen=True)
class ReportSpec:
    slug: str
    title: str
    pdf_path: Path
    source_candidates: tuple[Path | str, ...]


def _latest(pattern: str) -> Path | None:
    paths = [Path(p) for p in glob.glob(str(pattern)) if Path(p).is_file()]
    if not paths:
        return None
    paths.sort(key=lambda path: (path.stat().st_mtime, path.name))
    return paths[-1]


def _first_existing(candidates: Iterable[Path | str]) -> Path | None:
    for candidate in candidates:
        if isinstance(candidate, str) and any(char in candidate for char in "*?[]"):
            found = _latest(candidate)
            if found:
                return found
            continue
        path = Path(candidate)
        if path.exists() and path.is_file():
            return path
    return None


def _clean(value: str) -> str:
    value = html.unescape(value or "")
    value = value.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    value = value.replace("\u2013", "-").replace("\u2014", "-").replace("\u2026", "...")
    value = re.sub(r"\s+", " ", value).strip()
    return value.replace("`", "")


def _html_lines(path: Path) -> list[str]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["style", "script", "svg"]):
        tag.decompose()
    lines: list[str] = []
    for node in soup.find_all(["h1", "h2", "h3", "p", "li", "th", "td"]):
        text = _clean(node.get_text(" ", strip=True))
        if not text:
            continue
        if node.name == "h1":
            lines.extend(["", f"# {text}", ""])
        elif node.name == "h2":
            lines.extend(["", f"## {text}"])
        elif node.name == "h3":
            lines.extend(["", f"### {text}"])
        elif node.name == "li":
            lines.append(f"- {text}")
        elif node.name in {"th", "td"}:
            lines.append(text)
        else:
            lines.append(text)
    return _trim_lines(lines)


def _markdown_lines(path: Path) -> list[str]:
    return _trim_lines(path.read_text(encoding="utf-8", errors="ignore").splitlines())


def _json_lines(path: Path) -> list[str]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return _markdown_lines(path)
    return json.dumps(obj, ensure_ascii=True, indent=2, sort_keys=True).splitlines()


def _source_lines(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix in {".html", ".htm"}:
        return _html_lines(path)
    if suffix == ".json":
        return _json_lines(path)
    return _markdown_lines(path)


def _trim_lines(lines: Iterable[str]) -> list[str]:
    out = [line.rstrip() for line in lines]
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return out


def _wrap_line(line: str, width: int = 104) -> list[str]:
    raw = _clean(line)
    if not raw:
        return [""]
    if raw.startswith("# "):
        return [raw]
    if raw.startswith("## "):
        return [raw]
    if raw.startswith("### "):
        return [raw]
    if raw.startswith("- "):
        chunks = textwrap.wrap(raw[2:], width=width - 4, break_long_words=False, break_on_hyphens=False)
        if not chunks:
            return [raw]
        return ["- " + chunks[0], *["  " + chunk for chunk in chunks[1:]]]
    return textwrap.wrap(raw, width=width, break_long_words=False, break_on_hyphens=False) or [raw]


def _paginated_lines(lines: list[str], *, max_source_lines: int = 900) -> list[str]:
    rows = list(lines[:max_source_lines])
    if len(lines) > max_source_lines:
        rows.extend(["", f"[Truncated for send-out PDF: {len(lines) - max_source_lines} source lines omitted.]"])
    wrapped: list[str] = []
    for line in rows:
        wrapped.extend(_wrap_line(line))
    return wrapped


def render_text_pdf(title: str, source_path: Path | None, pdf_path: Path, *, missing_detail: str = "") -> dict[str, object]:
    generated = datetime.now(timezone.utc).isoformat()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path is None:
        source_lines = [
            f"# {title}",
            "",
            "Source artifact was not found.",
            missing_detail or "The command is documented, but no current source file was available for this report.",
        ]
    else:
        source_lines = _source_lines(source_path)
        if not source_lines:
            source_lines = [f"# {title}", "", "Source artifact exists but contains no renderable text."]

    lines = [
        f"# {title}",
        "",
        f"Generated UTC: {generated}",
        f"Source: {source_path if source_path else 'missing'}",
        "",
        *source_lines,
    ]
    body_lines = _paginated_lines(lines)

    page_lines = 42
    chunks = [body_lines[i : i + page_lines] for i in range(0, len(body_lines), page_lines)] or [[]]
    with PdfPages(pdf_path) as pdf:
        for page_index, chunk in enumerate(chunks, start=1):
            fig = plt.figure(figsize=(8.5, 11), facecolor="#f8fbfc")
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis("off")
            fig.patches.append(Rectangle((0, 0.93), 1, 0.07, transform=fig.transFigure, color="#172033", zorder=-1))
            fig.patches.append(Rectangle((0, 0), 1, 0.035, transform=fig.transFigure, color="#e5ebef", zorder=-1))
            fig.text(0.065, 0.957, title[:90], fontsize=15, color="white", weight="bold", va="center")
            y = 0.895
            for line in chunk:
                if line.startswith("# "):
                    fig.text(0.065, y, line[2:], fontsize=14.5, color="#172033", weight="bold", va="top")
                    y -= 0.030
                elif line.startswith("## "):
                    fig.text(0.065, y, line[3:], fontsize=12.2, color="#2c8f8d", weight="bold", va="top")
                    y -= 0.026
                elif line.startswith("### "):
                    fig.text(0.075, y, line[4:], fontsize=10.8, color="#557fa8", weight="bold", va="top")
                    y -= 0.024
                elif line.startswith("- "):
                    fig.text(0.083, y, line, fontsize=8.8, color="#172033", va="top")
                    y -= 0.020
                elif line.startswith("  "):
                    fig.text(0.103, y, line.strip(), fontsize=8.8, color="#586474", va="top")
                    y -= 0.020
                elif not line.strip():
                    y -= 0.010
                else:
                    fig.text(0.075, y, line, fontsize=8.8, color="#172033", va="top")
                    y -= 0.020
            fig.text(0.065, 0.02, f"{title} - page {page_index} of {len(chunks)}", fontsize=8, color="#586474", va="center")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    return {
        "title": title,
        "source_path": str(source_path) if source_path else "",
        "pdf_path": str(pdf_path),
        "pdf_bytes": int(pdf_path.stat().st_size) if pdf_path.exists() else 0,
        "page_count": len(chunks),
        "ok": bool(pdf_path.exists() and pdf_path.stat().st_size > 10_000),
        "detail": "deterministic_text_pdf" if source_path else "missing_source_text_pdf",
    }


def _load_json(path: Path) -> dict[str, object]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _num(raw: object, default: float = 0.0) -> float:
    try:
        value = float(raw)  # type: ignore[arg-type]
    except Exception:
        return float(default)
    if not math.isfinite(value):
        return float(default)
    return value


def _int(raw: object, default: int = 0) -> int:
    try:
        return int(raw)  # type: ignore[arg-type]
    except Exception:
        return int(default)


def _fmt_amount(raw: object, digits: int = 2) -> str:
    value = _num(raw)
    return f"{value:+,.{digits}f}"


def _fmt_plain(raw: object, digits: int = 2) -> str:
    value = _num(raw)
    return f"{value:,.{digits}f}"


def _fmt_rate(raw: object) -> str:
    if raw is None:
        return "n/a"
    return f"{_num(raw) * 100.0:.1f}%"


def _short(raw: object, limit: int = 46) -> str:
    text = _clean(str(raw or ""))
    text = text.replace("paper_mirror::", "").replace("paper_mirror_futures::", "")
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 1)].rstrip() + "..."


def _safe_rows(raw: object) -> list[dict[str, object]]:
    if not isinstance(raw, list):
        return []
    return [row for row in raw if isinstance(row, dict)]


def _safe_dict(raw: object) -> dict[str, object]:
    return raw if isinstance(raw, dict) else {}


def _new_report_page(title: str, subtitle: str = "", *, landscape: bool = True):
    size = (11, 8.5) if landscape else (8.5, 11)
    fig = plt.figure(figsize=size, facecolor="#f7faf9")
    fig.patches.append(Rectangle((0, 0.925), 1, 0.075, transform=fig.transFigure, color="#122033", zorder=-1))
    fig.patches.append(Rectangle((0, 0), 1, 0.035, transform=fig.transFigure, color="#e5edf0", zorder=-1))
    fig.text(0.045, 0.963, title[:92], fontsize=17, color="white", weight="bold", va="center")
    if subtitle:
        fig.text(0.045, 0.932, subtitle[:130], fontsize=8.5, color="#cfe6ee", va="center")
    return fig


def _card(fig, x: float, y: float, w: float, h: float, label: str, value: str, detail: str = "", accent: str = "#1f7a8c") -> None:
    fig.patches.append(Rectangle((x, y), w, h, transform=fig.transFigure, facecolor="white", edgecolor="#d5e0e5", linewidth=1.0, zorder=-1))
    fig.patches.append(Rectangle((x, y + h - 0.012), w, 0.012, transform=fig.transFigure, facecolor=accent, edgecolor=accent, zorder=0))
    fig.text(x + 0.015, y + h - 0.032, label.upper(), fontsize=7.5, color="#5f6f7a", weight="bold", va="top")
    value_text = str(value or "")
    value_size = 15
    wrap_width = 18
    if len(value_text) > 18:
        value_size = 11
        wrap_width = 16
    if len(value_text) > 30:
        value_size = 9.2
        wrap_width = 19
    value_y = y + h - 0.073
    for line in textwrap.wrap(value_text, width=wrap_width, break_long_words=False, break_on_hyphens=False)[:2] or [""]:
        fig.text(x + 0.015, value_y, line, fontsize=value_size, color="#122033", weight="bold", va="top")
        value_y -= 0.029 if value_size >= 11 else 0.024
    if detail:
        fig.text(x + 0.015, y + 0.018, detail[:82], fontsize=7.7, color="#60707a", va="bottom")


def _wrapped_fig_text(fig, x: float, y: float, text: str, *, width: int = 98, size: float = 9.0, color: str = "#122033", weight: str = "normal", line_gap: float = 0.027) -> float:
    for line in textwrap.wrap(_clean(text), width=width, break_long_words=False, break_on_hyphens=False) or [""]:
        fig.text(x, y, line, fontsize=size, color=color, weight=weight, va="top")
        y -= line_gap
    return y


def _bullets(fig, x: float, y: float, rows: Iterable[str], *, width: int = 110, size: float = 8.7, color: str = "#122033") -> float:
    for row in rows:
        y = _wrapped_fig_text(fig, x, y, f"- {row}", width=width, size=size, color=color, line_gap=0.023)
        y -= 0.005
    return y


def _html_text(tag, selector: str) -> str:
    found = tag.select_one(selector) if tag else None
    return _clean(found.get_text(" ", strip=True)) if found else ""


def _html_texts(tag, selector: str) -> list[str]:
    if not tag:
        return []
    return [_clean(row.get_text(" ", strip=True)) for row in tag.select(selector) if _clean(row.get_text(" ", strip=True))]


def _framework_source(path: Path) -> dict[str, object]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    hero = soup.select_one(".hero")
    metrics = []
    for card in soup.select(".metric-card"):
        metrics.append(
            {
                "label": _html_text(card, ".label"),
                "value": _html_text(card, ".value"),
                "detail": _html_text(card, ".detail"),
            }
        )
    brief_cards = []
    for card in soup.select(".report-grid .brief-card")[:3]:
        brief_cards.append(
            {
                "heading": _html_text(card, "h2"),
                "paragraphs": _html_texts(card, "p"),
                "bullets": _html_texts(card, "li"),
            }
        )
    flow_cards = []
    for box in soup.select(".flow-wrap .box")[:8]:
        flow_cards.append(
            {
                "heading": _html_text(box, "h3"),
                "bullets": _html_texts(box, "li"),
            }
        )
    return {
        "title": _html_text(soup, "h1") or "Schwab Trading Bot Framework Map v2",
        "subtitle": " ".join(_html_texts(hero, ".sub")[:1]),
        "metrics": metrics,
        "brief_cards": brief_cards,
        "flow_cards": flow_cards,
    }


def _map_box(
    fig,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    detail: str,
    *,
    accent: str,
    face: str = "white",
    title_size: float = 9.0,
    detail_size: float = 7.25,
) -> None:
    fig.patches.append(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.010",
            transform=fig.transFigure,
            facecolor=face,
            edgecolor="#d3dde4",
            linewidth=1.0,
            zorder=-1,
        )
    )
    fig.patches.append(Rectangle((x, y + h - 0.014), w, 0.014, transform=fig.transFigure, facecolor=accent, edgecolor=accent, zorder=0))
    fig.text(x + 0.012, y + h - 0.033, _short(title, 32), fontsize=title_size, color="#122033", weight="bold", va="top")
    text_y = y + h - 0.058
    max_detail_lines = max(1, min(4, int(max(h - 0.060, 0.020) / 0.019)))
    for line in textwrap.wrap(_clean(detail), width=24, break_long_words=False, break_on_hyphens=False)[:max_detail_lines]:
        fig.text(x + 0.012, text_y, line, fontsize=detail_size, color="#475569", va="top")
        text_y -= 0.019


def _map_arrow(fig, start: tuple[float, float], end: tuple[float, float], *, color: str = "#78909c", label: str = "") -> None:
    fig.patches.append(
        FancyArrowPatch(
            start,
            end,
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.45,
            color=color,
            shrinkA=2,
            shrinkB=2,
            zorder=1,
        )
    )
    if label:
        fig.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.014, label, fontsize=6.9, color=color, weight="bold", ha="center")


def _axis_no_data(ax, title: str, detail: str = "No source rows available for this chart.") -> None:
    ax.set_title(title, fontsize=11, weight="bold", color="#122033")
    ax.text(0.5, 0.5, detail, transform=ax.transAxes, ha="center", va="center", fontsize=9, color="#60707a")
    ax.set_axis_off()


def _axis_money(ax) -> None:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:,.0f}"))
    ax.grid(axis="y", linestyle="--", alpha=0.24)
    ax.tick_params(axis="x", labelrotation=35, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _bar_colors(values: list[float]) -> list[str]:
    return ["#0f766e" if value > 0 else "#b42318" if value < 0 else "#64748b" for value in values]


def _plot_line(ax, rows: list[dict[str, object]], *, label_key: str, value_key: str, title: str, color: str) -> None:
    if not rows:
        _axis_no_data(ax, title)
        return
    labels = [str(row.get(label_key, ""))[-4:] if len(str(row.get(label_key, ""))) == 8 else str(row.get(label_key, "")) for row in rows]
    values = [_num(row.get(value_key)) for row in rows]
    ax.plot(labels, values, color=color, linewidth=2.5, marker="o", markersize=4.8)
    ax.fill_between(labels, values, [0.0] * len(values), color=color, alpha=0.10)
    ax.axhline(0.0, color="#122033", linewidth=0.9, alpha=0.7)
    ax.set_title(title, fontsize=11, weight="bold", color="#122033")
    ax.set_ylabel("Paper PnL", fontsize=8.5)
    if values:
        ax.annotate(_fmt_amount(values[-1]), xy=(labels[-1], values[-1]), xytext=(6, 6), textcoords="offset points", fontsize=8, color=color, weight="bold")
    _axis_money(ax)


def _plot_bars(ax, labels: list[str], values: list[float], *, title: str, xlabel: str = "Paper PnL") -> None:
    if not labels:
        _axis_no_data(ax, title)
        return
    ax.bar(labels, values, color=_bar_colors(values), alpha=0.86)
    ax.axhline(0.0, color="#122033", linewidth=0.9, alpha=0.7)
    ax.set_title(title, fontsize=11, weight="bold", color="#122033")
    ax.set_ylabel(xlabel, fontsize=8.5)
    for idx, value in enumerate(values):
        ax.annotate(_fmt_amount(value), xy=(idx, value), xytext=(0, 5 if value >= 0 else -12), textcoords="offset points", ha="center", fontsize=7.2, color="#122033")
    _axis_money(ax)


def render_paper_performance_ready_pdf(source_path: Path, pdf_path: Path) -> dict[str, object]:
    payload = _load_json(source_path)
    generated = datetime.now(timezone.utc).isoformat()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    day = _safe_dict(payload.get("day"))
    week = _safe_dict(payload.get("week"))
    active_profiles = _safe_rows(payload.get("active_paper_profiles_today"))
    sleeves = _safe_rows(payload.get("sleeve_latest"))
    sleeves = [
        row for row in sleeves
        if str(row.get("data_status") or "") != "no_data"
        and (_int(row.get("executions")) > 0 or abs(_num(row.get("ending_net_pnl_total"))) > 0.000001 or abs(_num(row.get("change_vs_previous_day"))) > 0.000001)
    ]
    sleeves.sort(key=lambda row: (str(row.get("data_status") or ""), -abs(_num(row.get("ending_net_pnl_total"))), str(row.get("profile") or "")))
    daily_rows = _safe_rows(payload.get("history_daily_series"))[-30:]
    weekly_rows = _safe_rows(payload.get("weekly_history_series"))[-12:]
    monthly_rows = _safe_rows(payload.get("monthly_history_series"))[-18:]
    quarterly_rows = _safe_rows(payload.get("quarterly_history_series"))[-16:]
    period_rows = _safe_rows(payload.get("period_change_series"))

    pages = 0
    with PdfPages(pdf_path) as pdf:
        fig = _new_report_page("Paper Performance Report", f"Generated UTC: {generated} | Source: {source_path}")
        _card(fig, 0.045, 0.765, 0.145, 0.105, "Ending net", _fmt_amount(day.get("ending_net_pnl_total")), "realized + unrealized", "#1f7a8c")
        _card(fig, 0.205, 0.765, 0.145, 0.105, "Day change", _fmt_amount(day.get("change_vs_previous_day")), "vs previous day", "#0f766e" if _num(day.get("change_vs_previous_day")) >= 0 else "#b42318")
        _card(fig, 0.365, 0.765, 0.145, 0.105, "WTD change", _fmt_amount(week.get("week_to_date_change")), f"since {week.get('week_start_day_utc', '')}", "#0f766e" if _num(week.get("week_to_date_change")) >= 0 else "#b42318")
        _card(fig, 0.525, 0.765, 0.145, 0.105, "Rolling change", _fmt_amount(week.get("rolling_change")), f"{_int(week.get('rolling_change_days'), 7)} day window", "#7c3aed")
        _card(fig, 0.685, 0.765, 0.115, 0.105, "Executions", f"{_int(day.get('executions')):,}", f"buys/sells {_int(day.get('buy_count'))}/{_int(day.get('sell_count'))}", "#d97706")
        _card(fig, 0.815, 0.765, 0.14, 0.105, "Active lanes", f"{len(active_profiles):,}", "heartbeat profiles today", "#2563eb")

        fig.text(0.045, 0.705, "Executive Readout", fontsize=13, color="#122033", weight="bold", va="top")
        top_profiles = ", ".join(f"{item.get('name')} ({item.get('executions')})" for item in _safe_rows(day.get("top_profiles"))[:3]) or "n/a"
        top_symbols = ", ".join(f"{item.get('name')} ({item.get('executions')})" for item in _safe_rows(day.get("top_symbols"))[:4]) or "n/a"
        bullets = [
            "Ending net PnL combines realized and unrealized paper PnL at the latest snapshot, so it is the clearest single-day health marker for the paper stack.",
            f"Today's move is {_fmt_amount(day.get('change_vs_previous_day'))}; WTD is {_fmt_amount(week.get('week_to_date_change'))}; rolling {_int(week.get('rolling_change_days'), 7)} day change is {_fmt_amount(week.get('rolling_change'))}.",
            f"Most active profiles today: {top_profiles}. Most active symbols: {top_symbols}.",
            "Use the sleeve page to see whether one lane is driving the headline result or whether the move is broad across the stack.",
        ]
        _bullets(fig, 0.06, 0.665, bullets, width=132, size=9.1)

        fig.text(0.045, 0.43, "How To Read The Charts", fontsize=13, color="#122033", weight="bold", va="top")
        chart_notes = [
            "Daily and weekly lines show ending net paper PnL, not live realized account PnL.",
            "The day-change bars isolate the daily move, which is better for spotting new drift or a single-day shock.",
            "The sleeve scoreboard ranks the active lanes by current impact and pairs net PnL with day-over-day change, execution count, win rate, and top loss causes.",
            "A report-ready green result means the lane improved on paper; it still needs promotion gates, guardrails, and broker reconciliation before live expansion.",
        ]
        _bullets(fig, 0.06, 0.39, chart_notes, width=132, size=8.8)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        pages += 1

        fig = _new_report_page("Paper Performance Trend Charts", "Ending net PnL and period changes")
        axes = fig.subplots(2, 2)
        fig.subplots_adjust(left=0.07, right=0.965, top=0.84, bottom=0.11, hspace=0.38, wspace=0.22)
        _plot_line(axes[0][0], daily_rows, label_key="day_utc", value_key="ending_net_pnl_total", title="Daily Ending Net PnL", color="#1d4ed8")
        _plot_bars(
            axes[0][1],
            [str(row.get("day_utc", ""))[-4:] for row in daily_rows[-12:]],
            [_num(row.get("change_vs_previous_day")) for row in daily_rows[-12:]],
            title="Recent Day-Over-Day Change",
        )
        _plot_line(axes[1][0], weekly_rows, label_key="week_end_day_utc", value_key="ending_net_pnl_total", title="Weekly Ending Net PnL", color="#0f766e")
        if period_rows:
            _plot_bars(
                axes[1][1],
                [str(row.get("label") or "") for row in period_rows],
                [_num(row.get("change")) for row in period_rows],
                title="Window Change Comparison",
            )
        else:
            labels = [str(row.get("month_key") or row.get("quarter_key") or "") for row in monthly_rows[-6:] or quarterly_rows[-6:]]
            values = [_num(row.get("change_vs_previous_period")) for row in monthly_rows[-6:] or quarterly_rows[-6:]]
            _plot_bars(axes[1][1], labels, values, title="Period Change Comparison")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        pages += 1

        fig = _new_report_page("Paper Sleeve Scoreboard", "Lane-level contribution, activity, and risk notes")
        top = sleeves[:12]
        labels = [_short(row.get("profile"), 24) for row in top]
        net_values = [_num(row.get("ending_net_pnl_total")) for row in top]
        day_values = [_num(row.get("change_vs_previous_day")) for row in top]
        axes = fig.subplots(1, 2)
        fig.subplots_adjust(left=0.18, right=0.96, top=0.82, bottom=0.13, wspace=0.35)
        for ax, values, title in ((axes[0], net_values, "Ending Net PnL By Sleeve"), (axes[1], day_values, "Day Change By Sleeve")):
            if not labels:
                _axis_no_data(ax, title)
                continue
            ypos = list(range(len(labels)))
            ax.barh(ypos, values, color=_bar_colors(values), alpha=0.88)
            ax.axvline(0.0, color="#122033", linewidth=0.9, alpha=0.75)
            ax.set_yticks(ypos)
            ax.set_yticklabels(labels, fontsize=8)
            ax.invert_yaxis()
            ax.set_title(title, fontsize=11, weight="bold", color="#122033")
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:,.0f}"))
            ax.grid(axis="x", linestyle="--", alpha=0.23)
            for idx, value in enumerate(values):
                ax.annotate(_fmt_amount(value), xy=(value, idx), xytext=(5 if value >= 0 else -5, 0), textcoords="offset points", ha="left" if value >= 0 else "right", va="center", fontsize=7.2)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        pages += 1

        fig = _new_report_page("Paper Sleeve Notes", "Plain-English context for the current active lanes")
        y = 0.84
        if not top:
            y = _wrapped_fig_text(fig, 0.055, y, "No active sleeve rows were available in the latest paper-performance snapshot.", width=120, size=10)
        for row in top:
            loss_causes = ", ".join(str(item.get("cause") or "") for item in _safe_rows(row.get("top_loss_causes"))[:3]) or "n/a"
            best = ", ".join(_short(item.get("strategy"), 30) for item in _safe_rows(row.get("top_winning_strategies"))[:2]) or "n/a"
            worst = ", ".join(_short(item.get("strategy"), 30) for item in _safe_rows(row.get("top_losing_strategies"))[:2]) or "n/a"
            line = (
                f"{row.get('profile', '')}: status={row.get('data_status', '')}, day={row.get('day_utc', '') or 'n/a'}, "
                f"net={_fmt_amount(row.get('ending_net_pnl_total'))}, day_change={_fmt_amount(row.get('change_vs_previous_day'))}, "
                f"exec={_int(row.get('executions'))}, win_rate={_fmt_rate(row.get('win_rate'))}, "
                f"best={best}, worst={worst}, loss_causes={loss_causes}."
            )
            y = _wrapped_fig_text(fig, 0.055, y, line, width=142, size=8.2, line_gap=0.021)
            y -= 0.014
            if y < 0.075:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                pages += 1
                fig = _new_report_page("Paper Sleeve Notes", "Continued")
                y = 0.84
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        pages += 1

    ok = bool(pdf_path.exists() and pdf_path.stat().st_size > 10_000)
    payload["pdf"] = {
        "available": bool(ok),
        "html_report_path": str(REPORTS_DIR / "paper_performance_latest.html"),
        "pdf_path": str(pdf_path),
        "detail": "report_ready_paper_performance_pdf",
    }
    try:
        source_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass

    return {
        "title": "Paper Performance",
        "source_path": str(source_path),
        "pdf_path": str(pdf_path),
        "pdf_bytes": int(pdf_path.stat().st_size) if pdf_path.exists() else 0,
        "page_count": int(pages),
        "ok": ok,
        "detail": "report_ready_paper_performance_pdf",
    }


def render_post_trade_ready_pdf(source_path: Path, pdf_path: Path) -> dict[str, object]:
    payload = _load_json(source_path)
    generated = datetime.now(timezone.utc).isoformat()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    summary = _safe_dict(payload.get("summary"))
    calibration = _safe_dict(payload.get("paper_execution_calibration"))
    runtime = _safe_dict(payload.get("daily_runtime_summary"))
    strategy = _safe_dict(payload.get("strategy_attribution"))
    softguard = _safe_dict(payload.get("softguard"))
    sources = _safe_dict(payload.get("sources"))
    pages = 0

    with PdfPages(pdf_path) as pdf:
        fig = _new_report_page("Post-Trade Analysis", f"Generated UTC: {generated} | Source: {source_path}")
        max_mae = _num(_safe_dict(calibration.get("thresholds")).get("max_mae_bps"))
        _card(fig, 0.045, 0.765, 0.145, 0.105, "PnL proxy", _fmt_amount(summary.get("total_pnl_proxy"), 4), f"top lane: {_short(summary.get('top_lane') or 'n/a', 28)}", "#1f7a8c")
        _card(fig, 0.205, 0.765, 0.145, 0.105, "Paper MAE", f"{_fmt_plain(summary.get('paper_mae_bps'), 2)} bps", f"guardrail {max_mae:.2f} bps", "#0f766e" if bool(summary.get("paper_ok")) else "#b42318")
        _card(fig, 0.365, 0.765, 0.145, 0.105, "Decision rows", f"{_int(summary.get('decision_rows')):,}", f"stale windows {_int(summary.get('decision_stale_windows'))}", "#2563eb")
        _card(fig, 0.525, 0.765, 0.145, 0.105, "Softguard events", f"{_int(summary.get('global_halt_events')):,}", _short(summary.get("top_halt_reason") or "n/a", 38), "#d97706")
        _card(fig, 0.685, 0.765, 0.115, 0.105, "Restarts", f"{_int(summary.get('watchdog_restarts')):,}", "watchdog", "#7c3aed")
        status = "Ready" if bool(payload.get("ok")) else "Review"
        _card(fig, 0.815, 0.765, 0.14, 0.105, "Status", status, "source completeness", "#0f766e" if bool(payload.get("ok")) else "#b42318")

        fig.text(0.045, 0.705, "Assessment", fontsize=13, color="#122033", weight="bold", va="top")
        assessment = [str(row) for row in (payload.get("assessment") or []) if str(row).strip()]
        if not assessment:
            assessment = ["No assessment rows were present in the source artifact."]
        _bullets(fig, 0.06, 0.665, assessment, width=132, size=9.0)

        fig.text(0.045, 0.405, "Report Interpretation", fontsize=13, color="#122033", weight="bold", va="top")
        interpretation = [
            "This report ties together strategy attribution, paper execution calibration, runtime health, and softguard activity for the selected day.",
            "Paper MAE shows how far expected execution cost is from observed paper slippage. Lower is better, and values under the guardrail mean the simulator is behaving inside tolerance.",
            "Decision rows and stale windows show whether the runtime produced enough current decisions to trust the day snapshot.",
            "Softguard events identify operational friction that can distort paper results even when strategy attribution is clean.",
        ]
        if sources.get("daily_runtime_summary_fallback_used"):
            interpretation.append("Runtime summary was loaded from the latest cached artifact because the live runtime summary command timed out.")
        if sources.get("paper_execution_calibration_fallback_used"):
            interpretation.append("Paper calibration was loaded from the latest cached artifact because the live calibration command did not return a fresh payload.")
        _bullets(fig, 0.06, 0.365, interpretation, width=132, size=8.7)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        pages += 1

        fig = _new_report_page("Post-Trade Charts", "Execution quality, runtime, softguard, and attribution views")
        axes = fig.subplots(2, 2)
        fig.subplots_adjust(left=0.08, right=0.965, top=0.84, bottom=0.11, hspace=0.38, wspace=0.25)

        by_profile = _safe_dict(calibration.get("by_profile"))
        profile_rows = sorted(
            [
                {"name": name, **_safe_dict(value)}
                for name, value in by_profile.items()
                if isinstance(value, dict)
            ],
            key=lambda row: (_int(row.get("samples")), _num(row.get("mae_bps"))),
            reverse=True,
        )[:10]
        _plot_bars(
            axes[0][0],
            [_short(row.get("name"), 18) for row in profile_rows],
            [_num(row.get("mae_bps")) for row in profile_rows],
            title="Execution Calibration MAE By Profile",
            xlabel="MAE bps",
        )

        top_symbols = _safe_rows(calibration.get("top_symbols"))[:10]
        _plot_bars(
            axes[0][1],
            [_short(row.get("symbol"), 18) for row in top_symbols],
            [_num(row.get("p95_bps")) for row in top_symbols],
            title="Top Symbol P95 Slippage",
            xlabel="p95 bps",
        )

        reason_counts = _safe_dict(softguard.get("reason_counts"))
        reasons = sorted(reason_counts.items(), key=lambda item: (-_int(item[1]), str(item[0])))[:8]
        _plot_bars(
            axes[1][0],
            [_short(name, 18) for name, _count in reasons],
            [float(_int(count)) for _name, count in reasons],
            title="Softguard Reasons",
            xlabel="events",
        )

        by_lane = _safe_rows(strategy.get("by_lane"))[:8]
        lane_labels = [_short(row.get("lane") or row.get("name") or row.get("profile"), 18) for row in by_lane]
        lane_values = [_num(row.get("pnl_proxy") or row.get("total_pnl_proxy")) for row in by_lane]
        _plot_bars(axes[1][1], lane_labels, lane_values, title="Strategy Attribution By Lane", xlabel="pnl proxy")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        pages += 1

        fig = _new_report_page("Post-Trade Source Notes", "What was available and what needs attention")
        y = 0.84
        source_lines = [
            f"Strategy attribution rows: {_int(strategy.get('row_count'))}; files scanned: {_int(strategy.get('file_count'))}; latest event: {strategy.get('latest_event_timestamp_utc') or 'n/a'}.",
            f"Calibration samples: {_int(calibration.get('samples'))}; MAE: {_fmt_plain(_safe_dict(calibration.get('metrics')).get('mae_bps'), 4)} bps; p95: {_fmt_plain(_safe_dict(calibration.get('metrics')).get('p95_bps'), 4)} bps.",
            f"Runtime decision rows: {_int(_safe_dict(runtime.get('decision')).get('rows'))}; stale windows: {_int(_safe_dict(runtime.get('decision')).get('stale_windows'))}; watchdog restarts: {_int(_safe_dict(runtime.get('watchdog')).get('restarts'))}.",
            f"Softguard rows: {_int(softguard.get('rows'))}; latest softguard timestamp: {softguard.get('latest_timestamp_utc') or 'n/a'}.",
            f"Subcommand status: calibration_rc={sources.get('paper_execution_calibration_rc', 'n/a')}, runtime_rc={sources.get('daily_runtime_summary_rc', 'n/a')}.",
        ]
        y = _bullets(fig, 0.06, y, source_lines, width=136, size=9.0)
        y -= 0.02
        fig.text(0.045, y, "Professional Readiness Notes", fontsize=13, color="#122033", weight="bold", va="top")
        y -= 0.04
        readiness = [
            "If attribution rows are zero, the report is still operationally useful, but it cannot explain which strategy produced PnL for that day.",
            "If softguard events are elevated, review halt reasons before interpreting paper PnL as strategy quality.",
            "If runtime was served from cache, refresh the runtime summary during a calmer maintenance window before sending a final packet.",
        ]
        _bullets(fig, 0.06, y, readiness, width=136, size=8.8)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        pages += 1

    ok = bool(pdf_path.exists() and pdf_path.stat().st_size > 10_000)
    payload["pdf"] = {
        "available": bool(ok),
        "pdf_path": str(pdf_path),
        "detail": "report_ready_post_trade_pdf",
    }
    try:
        source_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass

    return {
        "title": "Post-Trade Analysis",
        "source_path": str(source_path),
        "pdf_path": str(pdf_path),
        "pdf_bytes": int(pdf_path.stat().st_size) if pdf_path.exists() else 0,
        "page_count": int(pages),
        "ok": ok,
        "detail": "report_ready_post_trade_pdf",
    }


def _markdown_section(lines: list[str], heading: str) -> list[str]:
    marker = f"## {heading}"
    start = None
    for idx, line in enumerate(lines):
        if line.strip() == marker:
            start = idx + 1
            break
    if start is None:
        return []
    end = len(lines)
    for idx in range(start, len(lines)):
        if lines[idx].startswith("## "):
            end = idx
            break
    return _trim_lines(lines[start:end])


def _timeline_items(section_lines: list[str], limit: int = 12) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    pattern = re.compile(r"^\s*(?:\d+\.|-)\s+`?([^`|]+)`?\s*\|\s*`?([^`|]+)`?\s*\|\s*`?([^`|]+)`?\s*\|\s*(.+)$")
    for line in section_lines:
        match = pattern.match(line.strip())
        if not match:
            continue
        date, area, kind, detail = match.groups()
        rows.append(
            {
                "date": _clean(date),
                "area": _clean(area),
                "kind": _clean(kind),
                "detail": _clean(re.sub(r"\(ref:.*?\)", "", detail)),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def _plain_bullets(section_lines: list[str], limit: int = 8) -> list[str]:
    out: list[str] = []
    for line in section_lines:
        text = line.strip()
        if text.startswith("- "):
            out.append(_clean(text[2:]))
        if len(out) >= limit:
            break
    return out


def _timeline_table(fig, rows: list[dict[str, str]], *, title: str, y: float = 0.82) -> None:
    fig.text(0.045, y, title, fontsize=13, color="#122033", weight="bold", va="top")
    y -= 0.04
    if not rows:
        _wrapped_fig_text(fig, 0.06, y, "No timeline rows were available in the source artifact.", width=120, size=9)
        return
    for row in rows:
        fig.text(0.06, y, row["date"][:22], fontsize=8.0, color="#1f7a8c", weight="bold", va="top")
        fig.text(0.205, y, row["area"][:24], fontsize=8.0, color="#334155", weight="bold", va="top")
        fig.text(0.36, y, row["kind"][:28], fontsize=8.0, color="#64748b", va="top")
        y = _wrapped_fig_text(fig, 0.535, y, row["detail"], width=68, size=8.0, line_gap=0.019)
        y -= 0.013
        if y < 0.075:
            break


def _timeline_color(area: str) -> str:
    text = str(area or "").lower()
    if "training" in text or "promotion" in text:
        return "#7c3aed"
    if "intelligence" in text or "data" in text:
        return "#1d4ed8"
    if "ops" in text or "governance" in text:
        return "#0f766e"
    if "registry" in text or "core" in text:
        return "#d97706"
    return "#334155"


def _phase_card(fig, x: float, y: float, w: float, h: float, title: str, body: str, color: str) -> None:
    fig.patches.append(Rectangle((x, y), w, h, transform=fig.transFigure, facecolor="white", edgecolor="#d7e1e6", linewidth=1.0, zorder=-1))
    fig.patches.append(Rectangle((x, y + h - 0.018), w, 0.018, transform=fig.transFigure, facecolor=color, edgecolor=color, zorder=0))
    fig.text(x + 0.018, y + h - 0.047, title, fontsize=11.5, color="#122033", weight="bold", va="top")
    _wrapped_fig_text(fig, x + 0.018, y + h - 0.088, body, width=34, size=8.2, color="#40515d", line_gap=0.021)


def _presentation_bullets(fig, title: str, rows: list[str], *, x: float, y: float, width: int = 60) -> float:
    fig.text(x, y, title, fontsize=13, color="#122033", weight="bold", va="top")
    y -= 0.045
    return _bullets(fig, x + 0.012, y, rows, width=width, size=8.7, color="#243442")


def _timeline_span_value(raw: str) -> str:
    dates = re.findall(r"(\d{4})-(\d{2})-(\d{2})", raw)
    if len(dates) >= 2:
        start = f"{dates[0][1]}/{dates[0][2]}"
        end = f"{dates[-1][1]}/{dates[-1][2]}"
        return f"{start} - {end}"
    return _short(raw.replace("Project span:", ""), 22)


def _timeline_count_value(raw: str, prefix: str) -> str:
    match = re.search(r"(\d[\d,]*)", raw)
    return match.group(1) if match else _short(raw.replace(prefix, ""), 12)


def _timeline_branch_value(raw: str) -> str:
    value = raw.replace("Branch:", "").strip()
    value = value.replace("codex/", "")
    return _short(value, 22)


def render_project_timeline_ready_pdf(source_path: Path, pdf_path: Path) -> dict[str, object]:
    lines = source_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    generated = datetime.now(timezone.utc).isoformat()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = _plain_bullets(_markdown_section(lines, "Snapshot"), limit=8)
    buildout = _plain_bullets(_markdown_section(lines, "Buildout Summary"), limit=8)
    intelligence = _plain_bullets(_markdown_section(lines, "Cross-Sleeve Intelligence"), limit=6)
    milestones = _timeline_items(_markdown_section(lines, "Milestone Timeline"), limit=16)
    current = _timeline_items(_markdown_section(lines, "Current Phase"), limit=14)
    runtime = _plain_bullets(_markdown_section(lines, "Runtime and Gates"), limit=8)
    pages = 0

    with PdfPages(pdf_path) as pdf:
        fig = _new_report_page("Project Timeline", f"Presentation packet | Generated UTC: {generated}")
        span = next((item for item in snapshot if item.startswith("Project span:")), "Project span: n/a")
        commits = next((item for item in snapshot if item.startswith("Total commits:")), "Total commits: n/a")
        branch = next((item for item in snapshot if item.startswith("Branch:")), "Branch: n/a")
        modified = next((item for item in _plain_bullets(_markdown_section(lines, "Working Tree"), limit=3) if item.startswith("Modified:")), "Modified: n/a")

        fig.text(0.055, 0.805, "Schwab Trading Bot Platform Buildout", fontsize=24, color="#122033", weight="bold", va="top")
        _wrapped_fig_text(
            fig,
            0.058,
            0.745,
            "A milestone-level view of the system build from initial control room through sleeve expansion, training governance, operational automation, and current report-ready infrastructure.",
            width=108,
            size=11.5,
            color="#334155",
            line_gap=0.032,
        )
        _card(fig, 0.055, 0.545, 0.17, 0.115, "Project span", _timeline_span_value(span), "", "#1f7a8c")
        _card(fig, 0.245, 0.545, 0.14, 0.115, "Commits", _timeline_count_value(commits, "Total commits:"), "", "#2563eb")
        _card(fig, 0.405, 0.545, 0.16, 0.115, "Milestones", f"{len(milestones):,}", "", "#0f766e")
        _card(fig, 0.585, 0.545, 0.17, 0.115, "Current branch", _timeline_branch_value(branch), "", "#7c3aed")
        _card(fig, 0.775, 0.545, 0.17, 0.115, "Worktree", f"{_timeline_count_value(modified, 'Modified:')} modified", "", "#d97706")

        _phase_card(fig, 0.055, 0.29, 0.205, 0.155, "1. Foundation", "Control room, runtime entrypoints, shadow sleeves, and first orchestration layer.", "#1f7a8c")
        _phase_card(fig, 0.282, 0.29, 0.205, 0.155, "2. Expansion", "Dividend, bond, aggressive, futures, crypto, FX, and cross-sleeve market context.", "#2563eb")
        _phase_card(fig, 0.509, 0.29, 0.205, 0.155, "3. Governance", "Retrain gates, promotion controls, model cards, regression checks, and halt logic.", "#7c3aed")
        _phase_card(fig, 0.736, 0.29, 0.205, 0.155, "4. Operations", "Report automation, storage routing, watchdogs, command validation, and reporter quality bots.", "#0f766e")

        fig.text(0.055, 0.19, "Presentation Intent", fontsize=13, color="#122033", weight="bold", va="top")
        _wrapped_fig_text(
            fig,
            0.058,
            0.155,
            "This packet is designed for external review: it emphasizes build phases, evidence of platform maturity, and current operational posture instead of raw file-by-file change logs.",
            width=128,
            size=9.4,
            color="#40515d",
            line_gap=0.026,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        pages += 1

        fig = _new_report_page("Buildout Themes", "How the platform matured over the project")
        left_rows = buildout[:5] or ["No buildout summary lines found."]
        right_rows = intelligence[:4] or ["No cross-sleeve intelligence summary lines found."]
        _presentation_bullets(fig, "System Buildout", left_rows, x=0.055, y=0.83, width=60)
        _presentation_bullets(fig, "Cross-Sleeve Intelligence", right_rows, x=0.535, y=0.83, width=58)
        fig.patches.append(Rectangle((0.055, 0.14), 0.89, 0.13, transform=fig.transFigure, facecolor="#eaf4f4", edgecolor="#c7dbdf", zorder=-1))
        fig.text(0.075, 0.235, "What this says about the system", fontsize=12.5, color="#122033", weight="bold", va="top")
        _wrapped_fig_text(
            fig,
            0.075,
            0.198,
            "The project has moved beyond a single trading bot into a platform: multiple sleeves, shared market context, automated quality gates, operational observability, and a growing reporting layer.",
            width=124,
            size=9.5,
            color="#334155",
            line_gap=0.026,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        pages += 1

        fig = _new_report_page("Milestone Roadmap", "Major build steps from project start to current platform shape")
        y_positions = [0.78, 0.66, 0.54, 0.42, 0.30, 0.18]
        for idx, row in enumerate(milestones[:12]):
            col = 0 if idx < 6 else 1
            y = y_positions[idx % 6]
            x0 = 0.07 if col == 0 else 0.54
            color = _timeline_color(row["area"])
            fig.patches.append(Rectangle((x0, y - 0.018), 0.022, 0.022, transform=fig.transFigure, facecolor=color, edgecolor=color, zorder=1))
            fig.text(x0 + 0.035, y + 0.010, row["date"][:10], fontsize=8.0, color=color, weight="bold", va="top")
            fig.text(x0 + 0.135, y + 0.010, f"{row['area']} | {row['kind']}", fontsize=8.2, color="#475569", weight="bold", va="top")
            _wrapped_fig_text(fig, x0 + 0.035, y - 0.020, row["detail"], width=54, size=8.0, color="#122033", line_gap=0.019)
        fig.text(0.07, 0.095, "Color key: platform/core, intelligence/data, training/promotion, operations/governance.", fontsize=8, color="#64748b", va="center")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        pages += 1

        fig = _new_report_page("Current Phase And Readiness", "What is active now and how to interpret the platform state")
        _presentation_bullets(
            fig,
            "Current Phase",
            [f"{row['area']}: {row['detail']}" for row in current[:7]] or ["No current phase rows found."],
            x=0.055,
            y=0.83,
            width=64,
        )
        _presentation_bullets(fig, "Runtime And Gates", runtime[:7] or ["No runtime gate summary lines found."], x=0.535, y=0.83, width=58)
        fig.patches.append(Rectangle((0.055, 0.10), 0.89, 0.135, transform=fig.transFigure, facecolor="#fff7ed", edgecolor="#fed7aa", zorder=-1))
        fig.text(0.075, 0.205, "Reviewer Takeaway", fontsize=12.5, color="#122033", weight="bold", va="top")
        _wrapped_fig_text(
            fig,
            0.075,
            0.168,
            "The system is in an active growth phase. The most professional framing is not that every gate is green, but that the platform exposes gates, reports, promotion controls, and infrastructure bots that make the current risk posture visible.",
            width=126,
            size=9.4,
            color="#334155",
            line_gap=0.026,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        pages += 1

    return {
        "title": "Project Timeline",
        "source_path": str(source_path),
        "pdf_path": str(pdf_path),
        "pdf_bytes": int(pdf_path.stat().st_size) if pdf_path.exists() else 0,
        "page_count": int(pages),
        "ok": bool(pdf_path.exists() and pdf_path.stat().st_size > 10_000),
        "detail": "report_ready_project_timeline_pdf",
    }


def render_framework_map_ready_pdf(source_path: Path, pdf_path: Path) -> dict[str, object]:
    data = _framework_source(source_path)
    generated = datetime.now(timezone.utc).isoformat()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = list(data.get("metrics") or [])
    metric_by_label = {str(row.get("label") or ""): row for row in metrics if isinstance(row, dict)}
    pages = 0

    with PdfPages(pdf_path) as pdf:
        fig = _new_report_page("Framework Map v2", f"Actual architecture map | Generated UTC: {generated}")
        fig.text(0.055, 0.845, "Schwab Trading Bot Platform Framework", fontsize=22, color="#122033", weight="bold", va="top")
        _wrapped_fig_text(
            fig,
            0.058,
            0.805,
            "Architecture flow from intake to storage, decisions, risk, execution, training, and reporting.",
            width=124,
            size=9.6,
            color="#40515d",
            line_gap=0.025,
        )
        top_metrics = [
            ("Readiness", metric_by_label.get("Live Readiness", {}).get("value", "n/a"), "#0f766e"),
            ("Autonomy", metric_by_label.get("Autonomy", {}).get("value", "n/a"), "#7c3aed"),
            ("Collector quality", metric_by_label.get("Collector Quality", {}).get("value", "n/a"), "#2563eb"),
            ("Queue depth", metric_by_label.get("Queue Depth", {}).get("value", "n/a"), "#d97706"),
        ]
        for idx, (label, value, color) in enumerate(top_metrics):
            _card(fig, 0.055 + idx * 0.225, 0.675, 0.19, 0.095, label, str(value), "", color)

        nodes = [
            ("Market / Macro\nInputs", "C-SPAN, FRED, news, chain data, quotes, bars, earnings, calendar events.", "#1f7a8c"),
            ("Collectors", "Adapters normalize raw events, timestamp them, and route lane-ready facts.", "#2563eb"),
            ("Evidence Store", "Point-in-time artifacts, shards, SQLite rollups, snapshots, and cached fallbacks.", "#0f766e"),
            ("Sleeve Runtime", "Dividend, bond, futures, crypto, FX, intraday, swing, and options sleeves.", "#7c3aed"),
            ("Risk / Halt Gates", "Margin guards, tripwires, global halt, storage pressure, and promotion gates.", "#b42318"),
            ("Broker / Paper\nExecution", "Schwab/IBKR handshake, paper mirror, reconciliation, and broker truth.", "#d97706"),
        ]
        x0, y0, w, h, gap = 0.055, 0.44, 0.132, 0.16, 0.025
        centers: list[tuple[float, float]] = []
        for idx, (title, detail, color) in enumerate(nodes):
            x = x0 + idx * (w + gap)
            _map_box(fig, x, y0, w, h, title, detail, accent=color)
            centers.append((x + w / 2, y0 + h / 2))
            if idx:
                _map_arrow(fig, (x - gap + 0.003, y0 + h / 2), (x - 0.006, y0 + h / 2), label="feeds")

        control_y = 0.22
        controls = [
            ("Storage Guard", "moves, compresses, trims, and fails over before disk pressure breaks collection", "#1f7a8c"),
            ("Runtime Calmer", "CPU and memory governors smooth heavy views and fast-growing bot counts", "#2563eb"),
            ("Training Gate", "new bots collect first, train only after enough clean evidence is present", "#7c3aed"),
            ("Reporter Guard", "PDF integrity, report-ready renderers, command validation, and sendout checks", "#0f766e"),
        ]
        for idx, (title, detail, color) in enumerate(controls):
            x = 0.075 + idx * 0.225
            _map_box(fig, x, control_y, 0.19, 0.11, title, detail, accent=color, face="#fbfdfe", title_size=8.4, detail_size=6.9)
            _map_arrow(fig, (x + 0.095, control_y + 0.11), (centers[min(idx + 2, len(centers) - 1)][0], y0 - 0.006), color=color)

        fig.text(0.055, 0.14, "Reviewer takeaway", fontsize=12.5, color="#122033", weight="bold", va="top")
        _wrapped_fig_text(
            fig,
            0.058,
            0.105,
            "This is a platform map, not a single-bot diagram: each layer can be tested, halted, repaired, or explained without pretending the whole system is one indivisible process.",
            width=130,
            size=9.0,
            color="#40515d",
            line_gap=0.024,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        pages += 1

        fig = _new_report_page("Framework Map v2 - Control Loops", "How the system governs itself while collection and runtime continue")
        lanes = [
            ("Data Loop", "collect", "route", "compact", "snapshot", "#1f7a8c"),
            ("Decision Loop", "features", "sleeves", "portfolio intent", "paper/live gate", "#2563eb"),
            ("Risk Loop", "margin guard", "tripwire", "global halt", "clearance notice", "#b42318"),
            ("Learning Loop", "evidence window", "retrain", "model card", "promotion packet", "#7c3aed"),
            ("Reporting Loop", "artifact scan", "PDF render", "quality guard", "sendout packet", "#0f766e"),
        ]
        y = 0.75
        for lane, a, b, c, d, color in lanes:
            fig.text(0.055, y + 0.038, lane, fontsize=10.5, color=color, weight="bold", va="center")
            xs = [0.21, 0.39, 0.57, 0.75]
            labels = [a, b, c, d]
            for idx, label in enumerate(labels):
                _map_box(fig, xs[idx], y, 0.13, 0.075, label.title(), "", accent=color, face="#ffffff", title_size=8.0, detail_size=6.5)
                if idx:
                    _map_arrow(fig, (xs[idx - 1] + 0.13, y + 0.037), (xs[idx] - 0.006, y + 0.037), color=color)
            y -= 0.125
        fig.patches.append(Rectangle((0.055, 0.105), 0.89, 0.115, transform=fig.transFigure, facecolor="#eef7f7", edgecolor="#c7dbdf", zorder=-1))
        fig.text(0.075, 0.190, "Why this matters operationally", fontsize=12.0, color="#122033", weight="bold", va="top")
        _wrapped_fig_text(
            fig,
            0.075,
            0.158,
            "The loops separate collection, decisioning, risk response, training, and reporting. That keeps growth manageable: new bots can collect data immediately, while training and promotion remain gated until the evidence window is strong enough.",
            width=124,
            size=9.0,
            color="#334155",
            line_gap=0.024,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        pages += 1

        fig = _new_report_page("Framework Map v2 - Evidence And Ownership", "What proves the system state, and which layer consumes it")
        columns = [
            ("Evidence Sources", ["quotes/bars", "macro feeds", "C-SPAN events", "broker snapshots", "paper fills"], "#1f7a8c"),
            ("Storage Objects", ["raw artifacts", "SQLite shards", "health JSON", "cached summaries", "PDF sources"], "#2563eb"),
            ("Guards", ["margin guard", "tripwires", "global halt", "disk guard", "report guard"], "#b42318"),
            ("Consumers", ["runtime", "training", "promotion", "ops console", "sendout reports"], "#0f766e"),
        ]
        for idx, (title, rows, color) in enumerate(columns):
            x = 0.07 + idx * 0.225
            _map_box(fig, x, 0.64, 0.175, 0.115, title, "Owns this class of operational proof.", accent=color, title_size=8.8, detail_size=6.9)
            yy = 0.555
            for row in rows:
                _map_box(fig, x, yy, 0.175, 0.055, row.title(), "", accent=color, face="#fbfdfe", title_size=7.5, detail_size=6.4)
                yy -= 0.069
            if idx:
                _map_arrow(fig, (x - 0.046, 0.695), (x - 0.008, 0.695), color="#78909c", label="proves")

        notes = []
        for card in list(data.get("brief_cards") or [])[:3]:
            if not isinstance(card, dict):
                continue
            heading = str(card.get("heading") or "")
            text = " ".join(str(row) for row in list(card.get("paragraphs") or [])[:1])
            bullets = ", ".join(str(row) for row in list(card.get("bullets") or [])[:2])
            row = f"{heading}: {text or bullets}"
            if heading:
                notes.append(row)
        fig.text(0.055, 0.205, "Source report context", fontsize=12.5, color="#122033", weight="bold", va="top")
        _bullets(fig, 0.07, 0.168, notes[:3] or ["No executive context was found in the source HTML."], width=122, size=8.3, color="#334155")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        pages += 1

    return {
        "title": "Framework Map v2",
        "source_path": str(source_path),
        "pdf_path": str(pdf_path),
        "pdf_bytes": int(pdf_path.stat().st_size) if pdf_path.exists() else 0,
        "page_count": int(pages),
        "ok": bool(pdf_path.exists() and pdf_path.stat().st_size > 10_000),
        "detail": "report_ready_framework_map_pdf",
    }


def _specs() -> list[ReportSpec]:
    return [
        ReportSpec("active_bot_stack", "Active Bot Stack", PROJECT_ROOT / "exports" / "bot_stack_status" / "latest.pdf", (PROJECT_ROOT / "exports" / "bot_stack_status" / "latest.html", PROJECT_ROOT / "exports" / "bot_stack_status" / "latest.md", PROJECT_ROOT / "exports" / "bot_stack_status" / "latest.json")),
        ReportSpec("bot_explainability", "Bot Explainability", SQL_REPORTS_DIR / "bot_explainability_latest.pdf", (REPORTS_DIR / "pdf_render_sources" / "bot_explainability_latest.html", SQL_REPORTS_DIR / "bot_explainability_latest.json", str(SQL_REPORTS_DIR / "bot_explainability_*.json"), GOVERNANCE_DIR / "bot_explainability_latest.json")),
        ReportSpec("crash_report_digest", "Crash Report Digest", REPORTS_DIR / "crash_reports" / "crash_report_digest_latest.pdf", (REPORTS_DIR / "crash_reports" / "crash_report_digest_print_latest.html", REPORTS_DIR / "crash_reports" / "crash_report_digest_latest.md")),
        ReportSpec("daily_auto_verify", "Daily Auto Verify", SQL_REPORTS_DIR / "daily_auto_verify_latest.pdf", (REPORTS_DIR / "pdf_render_sources" / "daily_auto_verify_latest.html", GOVERNANCE_DIR / "daily_auto_verify_latest.json", str(SQL_REPORTS_DIR / "daily_auto_verify_*.json"), GOVERNANCE_DIR / "daily_auto_verify_progress_latest.json")),
        ReportSpec("daily_ops_report", "Daily Ops Report", REPORTS_DIR / "daily_ops_report_latest.pdf", (REPORTS_DIR / "daily_ops_report_latest.md", REPORTS_DIR / "daily_ops_report_latest.json", REPORTS_DIR / "pdf_render_sources" / "daily_ops_report_latest.html")),
        ReportSpec("daily_runtime_summary", "Daily Runtime Summary", SQL_REPORTS_DIR / "daily_runtime_summary_latest.pdf", (REPORTS_DIR / "pdf_render_sources" / "daily_runtime_summary_latest.html", GOVERNANCE_DIR / "daily_runtime_summary_latest.json", str(SQL_REPORTS_DIR / "daily_runtime_summary_*.json"))),
        ReportSpec("framework_map_v2", "Framework Map v2", REPORTS_DIR / "system_explainers" / "framework_map_v2_latest.pdf", (REPORTS_DIR / "system_explainers" / "framework_map_v2_latest.html",)),
        ReportSpec("incident_report", "Incident Report", REPORTS_DIR / "incident_report_latest.pdf", (REPORTS_DIR / "incident_report_latest.html", REPORTS_DIR / "incident_report_latest.md", GOVERNANCE_DIR / "incident_report_latest.json")),
        ReportSpec("incident_review_packet", "Incident Review Packet", REPORTS_DIR / "incident_review_packet_latest.pdf", (REPORTS_DIR / "pdf_render_sources" / "incident_review_packet_latest.html", GOVERNANCE_DIR / "incident_review_packet_latest.json")),
        ReportSpec("macro_crosscheck", "Macro Crosscheck", REPORTS_DIR / "macro_crosscheck_latest.pdf", (REPORTS_DIR / "macro_crosscheck_latest.md", REPORTS_DIR / "pdf_render_sources" / "macro_crosscheck_latest.html", GOVERNANCE_DIR / "macro_crosscheck_latest.json")),
        ReportSpec("market_crypto_correlation", "Market Crypto Correlation", REPORTS_DIR / "market_crypto_correlation_latest.pdf", (REPORTS_DIR / "market_crypto_correlation_latest.md", REPORTS_DIR / "pdf_render_sources" / "market_crypto_correlation_latest.html")),
        ReportSpec("model_card", "Model Card", SQL_REPORTS_DIR / "model_card_latest.pdf", (REPORTS_DIR / "pdf_render_sources" / "model_card_latest.html", GOVERNANCE_DIR / "model_card_latest.json", str(SQL_REPORTS_DIR / "model_card_*.json"))),
        ReportSpec("one_numbers", "One Numbers Report", PROJECT_ROOT / "exports" / "one_numbers" / "one_numbers_latest.pdf", (PROJECT_ROOT / "exports" / "one_numbers" / "latest.md", PROJECT_ROOT / "exports" / "one_numbers" / "latest" / "one_numbers_latest.md", REPORTS_DIR / "pdf_render_sources" / "one_numbers_latest.html", GOVERNANCE_DIR / "one_numbers_latest.json")),
        ReportSpec("paper_execution_calibration", "Paper Execution Calibration", SQL_REPORTS_DIR / "paper_execution_calibration_latest.pdf", (REPORTS_DIR / "pdf_render_sources" / "paper_execution_calibration_latest.html", GOVERNANCE_DIR / "paper_execution_calibration_latest.json")),
        ReportSpec("paper_performance", "Paper Performance", REPORTS_DIR / "paper_performance_latest.pdf", (GOVERNANCE_DIR / "paper_performance_latest.json", REPORTS_DIR / "paper_performance_latest.html", REPORTS_DIR / "paper_performance_latest.md")),
        ReportSpec("post_trade_analysis", "Post-Trade Analysis", REPORTS_DIR / "post_trade_analysis_latest.pdf", (GOVERNANCE_DIR / "post_trade_analysis_latest.json", REPORTS_DIR / "post_trade_analysis_latest.md", REPORTS_DIR / "pdf_render_sources" / "post_trade_analysis_latest.html")),
        ReportSpec("project_timeline", "Project Timeline", REPORTS_DIR / "project_timeline" / "project_timeline_latest.pdf", (REPORTS_DIR / "project_timeline" / "project_timeline_latest.md", REPORTS_DIR / "project_timeline" / "project_timeline_print_latest.html")),
        ReportSpec("quant_model_control", "Quant Model Control", REPORTS_DIR / "quant_model_control" / "quant_model_control_latest.pdf", (REPORTS_DIR / "quant_model_control" / "quant_model_control_latest.md", GOVERNANCE_DIR / "quant_model_control_latest.json")),
        ReportSpec("replay_feature_ablation", "Replay Feature Ablation", SQL_REPORTS_DIR / "replay_feature_ablation_latest.pdf", (REPORTS_DIR / "pdf_render_sources" / "replay_feature_ablation_latest.html", SQL_REPORTS_DIR / "replay_feature_ablation_latest.json", str(SQL_REPORTS_DIR / "replay_feature_ablation_*.json"))),
        ReportSpec("retrain_scorecard", "Retrain Scorecard", SQL_REPORTS_DIR / "retrain_scorecard_latest.pdf", (REPORTS_DIR / "pdf_render_sources" / "retrain_scorecard_latest.html", SQL_REPORTS_DIR / "retrain_scorecard_latest.md", str(SQL_REPORTS_DIR / "retrain_scorecard_*.md"), str(SQL_REPORTS_DIR / "retrain_scorecard_*.json"))),
        ReportSpec("sentiment_report", "Sentiment Report", REPORTS_DIR / "sentiment_report_latest.pdf", (REPORTS_DIR / "sentiment_report_latest.html", REPORTS_DIR / "sentiment_report_latest.md")),
        ReportSpec("source_verification", "Source Verification", REPORTS_DIR / "source_verification_latest.pdf", (REPORTS_DIR / "source_verification_latest.md", REPORTS_DIR / "pdf_render_sources" / "source_verification_latest.html")),
        ReportSpec("special_features", "Special Features And Highlights", REPORTS_DIR / "showcase" / "special_features_latest.pdf", (PROJECT_ROOT / "docs" / "showcase" / "generated" / "special_features_latest.html",)),
        ReportSpec("state_snapshot_drills", "State Snapshot Drills", PROJECT_ROOT / "exports" / "state_snapshot_drills" / "state_snapshot_drills_latest.pdf", (REPORTS_DIR / "pdf_render_sources" / "state_snapshot_drills_latest.html", PROJECT_ROOT / "exports" / "state_snapshot_drills" / "latest.json")),
        ReportSpec("strategy_attribution", "Strategy Attribution", REPORTS_DIR / "strategy_attribution_latest.pdf", (REPORTS_DIR / "strategy_attribution_latest.md", REPORTS_DIR / "pdf_render_sources" / "strategy_attribution_latest.html", GOVERNANCE_DIR / "strategy_attribution_latest.json")),
        ReportSpec("strategy_inventory", "Strategy Inventory", REPORTS_DIR / "strategy_inventory" / "strategy_inventory_latest.pdf", (REPORTS_DIR / "strategy_inventory" / "strategy_inventory_latest.md", GOVERNANCE_DIR / "strategy_inventory_latest.json")),
        ReportSpec("expansion_inventory", "Expansion Inventory", REPORTS_DIR / "expansion_inventory" / "expansion_inventory_latest.pdf", (REPORTS_DIR / "expansion_inventory" / "expansion_inventory_latest.md", GOVERNANCE_DIR / "expansion_inventory_latest.json")),
        ReportSpec("system_overview", "System Overview Weekly Platform History", REPORTS_DIR / "system_overview" / "system_overview_weekly_platform_history_latest.pdf", (REPORTS_DIR / "system_overview" / "system_overview_weekly_platform_history_latest.md",)),
        ReportSpec("system_summary", "Compiled System Summary", REPORTS_DIR / "system_summary" / "system_summary_latest.pdf", (REPORTS_DIR / "system_summary" / "system_summary_latest.html", GOVERNANCE_DIR / "system_summary_report_latest.json")),
        ReportSpec("training_report", "Training Report", REPORTS_DIR / "training_reports" / "training_report_latest.pdf", (REPORTS_DIR / "training_reports" / "training_report_print_latest.html", REPORTS_DIR / "training_reports" / "training_report_latest.md")),
        ReportSpec("unified_lane_scorecard", "Unified Lane Scorecard", SQL_REPORTS_DIR / "unified_lane_scorecard_latest.pdf", (SQL_REPORTS_DIR / "unified_lane_scorecard_latest.md", REPORTS_DIR / "pdf_render_sources" / "unified_lane_scorecard_latest.html")),
        ReportSpec("data_intake_and_shards", "Data Intake And Shards", REPORTS_DIR / "system_explainers" / "data_intake_and_shards_latest.pdf", (REPORTS_DIR / "system_explainers" / "data_intake_and_shards_latest.md",)),
        ReportSpec("health_gates_and_halt_logic", "Health Gates And Halt Logic", REPORTS_DIR / "system_explainers" / "health_gates_and_halt_logic_latest.pdf", (REPORTS_DIR / "system_explainers" / "health_gates_and_halt_logic_latest.md",)),
        ReportSpec("storage_routing_and_failover", "Storage Routing And Failover", REPORTS_DIR / "system_explainers" / "storage_routing_and_failover_latest.pdf", (REPORTS_DIR / "system_explainers" / "storage_routing_and_failover_latest.md",)),
        ReportSpec("broker_truth_and_reconciliation", "Broker Truth And Reconciliation", REPORTS_DIR / "system_explainers" / "broker_truth_and_reconciliation_latest.pdf", (REPORTS_DIR / "system_explainers" / "broker_truth_and_reconciliation_latest.md",)),
        ReportSpec("training_and_promotion", "Training And Promotion", REPORTS_DIR / "system_explainers" / "training_and_promotion_latest.pdf", (REPORTS_DIR / "system_explainers" / "training_and_promotion_latest.md",)),
        ReportSpec("runtime_hierarchy", "Runtime Hierarchy", REPORTS_DIR / "system_explainers" / "runtime_hierarchy_latest.pdf", (REPORTS_DIR / "system_explainers" / "runtime_hierarchy_latest.md",)),
    ]


def _write_catalog(entries: list[dict[str, object]]) -> None:
    generated = datetime.now(timezone.utc).isoformat()
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(entry['title']))}</td>"
        f"<td>{'ok' if entry['ok'] else 'missing source'}</td>"
        f"<td>{entry['pdf_bytes']}</td>"
        f"<td>{html.escape(str(entry['pdf_path']))}</td>"
        f"<td>{html.escape(str(entry['source_path']))}</td>"
        "</tr>"
        for entry in entries
    )
    CATALOG_HTML.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Trading System PDF Bundle</title>"
        "<style>body{font:14px -apple-system,BlinkMacSystemFont,sans-serif;margin:24px;color:#172033}"
        "table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #d7e0e6;padding:8px;text-align:left}"
        "th{font-size:12px;text-transform:uppercase;color:#586474}</style></head><body>"
        f"<h1>Trading System PDF Bundle</h1><p>Generated UTC: {html.escape(generated)}</p>"
        "<table><thead><tr><th>Report</th><th>Status</th><th>Bytes</th><th>PDF</th><th>Source</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></body></html>",
        encoding="utf-8",
    )
    catalog_lines = [
        "# Trading System PDF Bundle",
        "",
        f"Generated UTC: {generated}",
        "",
        *[
            f"- {entry['title']}: {'ok' if entry['ok'] else 'missing source'} | {entry['pdf_bytes']} bytes | {entry['pdf_path']}"
            for entry in entries
        ],
    ]
    tmp_source = REPORTS_DIR / "pdf_render_sources" / "report_pdf_bundle_latest.md"
    tmp_source.parent.mkdir(parents=True, exist_ok=True)
    tmp_source.write_text("\n".join(catalog_lines) + "\n", encoding="utf-8")
    render_text_pdf("Trading System PDF Bundle", tmp_source, CATALOG_PDF)


def refresh_pdfs(slugs: set[str] | None = None) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    specs = [spec for spec in _specs() if not slugs or spec.slug in slugs]
    for spec in specs:
        source = _first_existing(spec.source_candidates)
        if spec.slug == "paper_performance" and source and source.suffix.lower() == ".json":
            entry = render_paper_performance_ready_pdf(source, spec.pdf_path)
        elif spec.slug == "post_trade_analysis" and source and source.suffix.lower() == ".json":
            entry = render_post_trade_ready_pdf(source, spec.pdf_path)
        elif spec.slug == "project_timeline" and source and source.suffix.lower() == ".md":
            entry = render_project_timeline_ready_pdf(source, spec.pdf_path)
        elif spec.slug == "framework_map_v2" and source and source.suffix.lower() in {".html", ".htm"}:
            entry = render_framework_map_ready_pdf(source, spec.pdf_path)
        elif spec.slug == "expansion_inventory":
            from scripts.ops.expansion_list_report import render_expansion_inventory_ready_pdf

            entry = render_expansion_inventory_ready_pdf(source, spec.pdf_path)
        else:
            entry = render_text_pdf(spec.title, source, spec.pdf_path)
        entry["slug"] = spec.slug
        entries.append(entry)
    if not slugs:
        _write_catalog(entries)
        catalog_entry = render_text_pdf("Trading System PDF Bundle", CATALOG_HTML, CATALOG_PDF)
        catalog_entry["slug"] = "report_pdf_bundle"
        entries.insert(0, catalog_entry)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 2,
        "ok": all(bool(entry.get("ok")) for entry in entries),
        "overall_status": "ready" if all(bool(entry.get("ok")) for entry in entries) else "degraded",
        "renderer": "deterministic_matplotlib_text_pdf",
        "entry_count": len(entries),
        "missing_count": sum(1 for entry in entries if not entry.get("source_path")),
        "small_pdf_count": sum(1 for entry in entries if int(entry.get("pdf_bytes") or 0) < 10_000),
        "entries": entries,
        "index_html": str(CATALOG_HTML),
        "index_pdf": str(CATALOG_PDF),
    }
    CATALOG_JSON.parent.mkdir(parents=True, exist_ok=True)
    CATALOG_JSON.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate send-out-safe PDFs for the documented system reports.")
    parser.add_argument("--only", action="append", default=[], help="Render only a specific slug. May be passed multiple times.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--allow-gui-pdf-renderer", action="store_true", help="Accepted for compatibility; ignored.")
    args, _unknown = parser.parse_known_args()
    payload = refresh_pdfs(set(args.only or []) or None)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "sendout_pdf_refresh "
            f"status={payload['overall_status']} "
            f"entries={payload['entry_count']} "
            f"missing={payload['missing_count']} "
            f"small={payload['small_pdf_count']}"
        )
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
