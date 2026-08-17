#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HTML = PROJECT_ROOT / "exports" / "reports" / "system_explainers" / "framework_map_v2_latest.html"
DEFAULT_PDF = PROJECT_ROOT / "exports" / "reports" / "system_explainers" / "framework_map_v2_latest.pdf"
DEFAULT_JSON = PROJECT_ROOT / "governance" / "health" / "framework_map_pdf_latest.json"


COLORS = {
    "ink": "#172033",
    "muted": "#586474",
    "line": "#d7e0e6",
    "teal": "#2c8f8d",
    "blue": "#557fa8",
    "gold": "#b98920",
    "red": "#b04c4c",
    "green": "#1e7b50",
    "purple": "#6252bd",
    "paper": "#f8fbfc",
    "panel": "#ffffff",
}


def _clean(value: str) -> str:
    value = re.sub(r"\s+", " ", value or "").strip()
    return value.replace("`", "")


def _wrapped(value: str, width: int) -> list[str]:
    value = _clean(value)
    if not value:
        return []
    return textwrap.wrap(value, width=width, break_long_words=False, break_on_hyphens=False) or [value]


def _direct_text(tag, selector: str) -> str:
    found = tag.select_one(selector)
    return _clean(found.get_text(" ", strip=True)) if found else ""


def _children_texts(tag, selector: str) -> list[str]:
    return [_clean(row.get_text(" ", strip=True)) for row in tag.select(selector) if _clean(row.get_text(" ", strip=True))]


def _card_data(card) -> dict[str, object]:
    heading = _direct_text(card, "h2") or _direct_text(card, "h3")
    paragraphs = _children_texts(card, "p")
    bullets = _children_texts(card, "li")
    return {"heading": heading, "paragraphs": paragraphs, "bullets": bullets}


def parse_framework_map(html_path: Path) -> dict[str, object]:
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
    hero = soup.select_one(".hero")
    metrics = []
    for card in soup.select(".metric-card"):
        metrics.append(
            {
                "label": _direct_text(card, ".label"),
                "value": _direct_text(card, ".value"),
                "detail": _direct_text(card, ".detail"),
            }
        )

    flow = []
    flow_wrap = soup.select_one(".flow-wrap")
    for box in flow_wrap.select(".box") if flow_wrap else []:
        flow.append(_card_data(box))

    overview_cards = [_card_data(card) for card in soup.select(".report-grid .brief-card")]
    row_notes = [_card_data(card) for card in soup.select(".row .note")]
    section_cards = []
    for section in soup.select(".section-card"):
        heading = _direct_text(section, "h2")
        lead = _direct_text(section, ".section-lead") or _direct_text(section, ".mini-note")
        mini_cards = [_card_data(card) for card in section.select(".toc-card, .mini-box, .closing-card")]
        if heading or lead or mini_cards:
            section_cards.append({"heading": heading, "lead": lead, "cards": mini_cards})

    return {
        "title": _direct_text(soup, "h1") or "Schwab Trading Bot Framework Map v2",
        "eyebrow": _direct_text(soup, ".eyebrow"),
        "subtitles": _children_texts(hero, ".sub") if hero else [],
        "callouts": [_card_data(card) for card in soup.select(".hero-callout")],
        "metrics": metrics,
        "overview_cards": overview_cards,
        "flow": flow,
        "row_notes": row_notes,
        "section_cards": section_cards,
    }


def _new_page(pdf: PdfPages, title: str, subtitle: str = ""):
    fig = plt.figure(figsize=(8.5, 11), facecolor=COLORS["paper"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    fig.patches.append(Rectangle((0, 0.93), 1, 0.07, transform=fig.transFigure, color=COLORS["ink"], zorder=-1))
    fig.patches.append(Rectangle((0, 0), 1, 0.035, transform=fig.transFigure, color="#e5ebef", zorder=-1))
    fig.text(0.065, 0.957, title, fontsize=16, color="white", weight="bold", va="center")
    if subtitle:
        fig.text(0.065, 0.915, subtitle, fontsize=9.5, color=COLORS["muted"], va="top")
    return fig


def _save(pdf: PdfPages, fig, page_number: int) -> int:
    fig.text(0.065, 0.02, f"Framework Map v2 - page {page_number}", fontsize=8.5, color=COLORS["muted"], va="center")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return page_number + 1


def _body_text(fig, x: float, y: float, text: str, *, width: int = 86, size: float = 9.3, color: str | None = None, bullet: bool = False) -> float:
    prefix = "- " if bullet else ""
    indent = 0.018 if bullet else 0.0
    lines = _wrapped(text, width)
    if not lines:
        return y
    fig.text(x, y, prefix + lines[0], fontsize=size, color=color or COLORS["ink"], va="top")
    y -= 0.020
    for line in lines[1:]:
        fig.text(x + indent, y, line, fontsize=size, color=color or COLORS["ink"], va="top")
        y -= 0.020
    return y


def _section_heading(fig, x: float, y: float, text: str, *, color: str = COLORS["ink"], size: float = 13.0) -> float:
    fig.text(x, y, _clean(text), fontsize=size, color=color, weight="bold", va="top")
    return y - 0.032


def _draw_metrics_page(pdf: PdfPages, data: dict[str, object], page_number: int) -> int:
    fig = _new_page(pdf, "Schwab Trading Bot Framework Map v2", _clean(str(data.get("eyebrow") or "")))
    fig.text(0.065, 0.87, _clean(str(data.get("title") or "")), fontsize=24, color=COLORS["ink"], weight="bold", va="top")
    y = 0.82
    for subtitle in list(data.get("subtitles") or [])[:3]:
        y = _body_text(fig, 0.065, y, str(subtitle), width=92, size=10.0, color=COLORS["muted"])
        y -= 0.012

    callouts = list(data.get("callouts") or [])
    if callouts:
        x = 0.065
        for card in callouts[:2]:
            fig.patches.append(Rectangle((x, y - 0.105), 0.40, 0.09, transform=fig.transFigure, facecolor="white", edgecolor=COLORS["line"], linewidth=1))
            fig.text(x + 0.015, y - 0.026, str(card.get("heading") or ""), fontsize=10.0, color=COLORS["purple"], weight="bold", va="top")
            text = " ".join(str(v) for v in list(card.get("paragraphs") or [])[:1])
            _body_text(fig, x + 0.015, y - 0.050, text, width=38, size=8.4, color=COLORS["muted"])
            x += 0.43
        y -= 0.135

    fig.text(0.065, y, "Current Operating Metrics", fontsize=13, color=COLORS["ink"], weight="bold", va="top")
    y -= 0.035
    metrics = list(data.get("metrics") or [])
    for idx, metric in enumerate(metrics):
        col = idx % 2
        row = idx // 2
        x = 0.065 + col * 0.43
        yy = y - row * 0.105
        fig.patches.append(Rectangle((x, yy - 0.080), 0.40, 0.075, transform=fig.transFigure, facecolor="white", edgecolor=COLORS["line"], linewidth=1))
        fig.text(x + 0.014, yy - 0.019, str(metric.get("label") or ""), fontsize=8.5, color=COLORS["muted"], weight="bold", va="top")
        fig.text(x + 0.014, yy - 0.043, str(metric.get("value") or ""), fontsize=15, color=COLORS["teal"], weight="bold", va="top")
        fig.text(x + 0.160, yy - 0.044, str(metric.get("detail") or ""), fontsize=8.2, color=COLORS["muted"], va="top")
    return _save(pdf, fig, page_number)


def _draw_card_pages(pdf: PdfPages, title: str, cards: Iterable[dict[str, object]], page_number: int, *, subtitle: str = "") -> int:
    fig = _new_page(pdf, title, subtitle)
    y = 0.87
    for card in cards:
        needed = 0.055 + 0.022 * (len(list(card.get("paragraphs") or [])) + len(list(card.get("bullets") or [])))
        if y - needed < 0.08:
            page_number = _save(pdf, fig, page_number)
            fig = _new_page(pdf, title + " (cont.)", subtitle)
            y = 0.87
        heading = str(card.get("heading") or "")
        if heading:
            y = _section_heading(fig, 0.065, y, heading, color=COLORS["teal"], size=12.0)
        for paragraph in list(card.get("paragraphs") or []):
            y = _body_text(fig, 0.075, y, str(paragraph), width=92, size=9.0, color=COLORS["ink"])
            y -= 0.006
        for bullet in list(card.get("bullets") or []):
            y = _body_text(fig, 0.085, y, str(bullet), width=86, size=8.8, color=COLORS["muted"], bullet=True)
        y -= 0.025
    return _save(pdf, fig, page_number)


def build_pdf(data: dict[str, object], pdf_path: Path) -> int:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    page_number = 1
    with PdfPages(pdf_path) as pdf:
        page_number = _draw_metrics_page(pdf, data, page_number)
        page_number = _draw_card_pages(pdf, "Executive Overview", data.get("overview_cards") or [], page_number)
        page_number = _draw_card_pages(pdf, "Top-Level System Flow", data.get("flow") or [], page_number)
        page_number = _draw_card_pages(pdf, "Control And Interpretation", data.get("row_notes") or [], page_number)
        for section in list(data.get("section_cards") or []):
            cards = list(section.get("cards") or [])
            if not cards:
                continue
            page_number = _draw_card_pages(
                pdf,
                str(section.get("heading") or "Framework Detail"),
                cards,
                page_number,
                subtitle=str(section.get("lead") or ""),
            )
    return page_number - 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the framework map HTML into a deterministic readable PDF.")
    parser.add_argument("--html", default=str(DEFAULT_HTML))
    parser.add_argument("--pdf", default=str(DEFAULT_PDF))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    html_path = Path(args.html).expanduser()
    pdf_path = Path(args.pdf).expanduser()
    data = parse_framework_map(html_path)
    page_count = build_pdf(data, pdf_path)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ok": bool(pdf_path.exists() and pdf_path.stat().st_size > 10_000),
        "html_path": str(html_path),
        "pdf_path": str(pdf_path),
        "pdf_bytes": int(pdf_path.stat().st_size) if pdf_path.exists() else 0,
        "page_count": int(page_count),
        "renderer": "matplotlib_text_layout",
    }
    json_path = Path(args.json_out).expanduser()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(f"framework_map_pdf ok={int(payload['ok'])} pages={page_count} path={pdf_path}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
