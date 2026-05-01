#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HTML = PROJECT_ROOT / "docs" / "showcase" / "generated" / "special_features_latest.html"
DEFAULT_PDF = PROJECT_ROOT / "exports" / "reports" / "showcase" / "special_features_latest.pdf"
DEFAULT_JSON = PROJECT_ROOT / "governance" / "health" / "special_features_pdf_latest.json"

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
}


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip().replace("`", "")


def _wrapped(value: str, width: int) -> list[str]:
    value = _clean(value)
    if not value:
        return []
    return textwrap.wrap(value, width=width, break_long_words=False, break_on_hyphens=False) or [value]


def _text(tag, selector: str) -> str:
    found = tag.select_one(selector)
    return _clean(found.get_text(" ", strip=True)) if found else ""


def _texts(tag, selector: str) -> list[str]:
    return [_clean(row.get_text(" ", strip=True)) for row in tag.select(selector) if _clean(row.get_text(" ", strip=True))]


def _card(card) -> dict[str, object]:
    return {
        "heading": _text(card, "h2") or _text(card, "h3") or _text(card, "strong"),
        "paragraphs": _texts(card, "p"),
        "bullets": _texts(card, "li"),
    }


def parse_special_features(html_path: Path) -> dict[str, object]:
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
    hero = soup.select_one(".hero")
    metrics = []
    for box in soup.select(".mini-box"):
        metrics.append(
            {
                "label": _text(box, "h3"),
                "value": _text(box, ".metric"),
                "detail": _text(box, "p"),
            }
        )

    lineup = []
    for row in soup.select("table tbody tr"):
        cells = [_clean(cell.get_text(" ", strip=True)) for cell in row.select("td")]
        if cells:
            lineup.append(cells)

    return {
        "title": _text(soup, "h1") or "Special Features And Highlights",
        "eyebrow": _text(soup, ".eyebrow"),
        "subtitles": _texts(hero, ".sub") if hero else [],
        "summary": [_card(card) for card in soup.select(".brief-grid .brief-card")[:3]],
        "features": [_card(card) for card in soup.select(".feature-grid .box")],
        "metrics": metrics,
        "interpretation": [_card(card) for card in soup.select(".section-card:has(h2) .brief-grid .brief-card")][3:6],
        "highlights": _texts(soup.select_one(".section-card:has(h2)") or soup, "li"),
        "lineup": lineup,
        "recommendations": [_card(card) for card in soup.select(".section-card")[-1].select(".brief-card")] if soup.select(".section-card") else [],
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
    fig.text(0.065, 0.02, f"Special Features - page {page_number}", fontsize=8.5, color=COLORS["muted"], va="center")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return page_number + 1


def _body(fig, x: float, y: float, text: str, *, width: int = 88, size: float = 9.0, color: str = COLORS["ink"], bullet: bool = False) -> float:
    lines = _wrapped(text, width)
    if not lines:
        return y
    prefix = "- " if bullet else ""
    fig.text(x, y, prefix + lines[0], fontsize=size, color=color, va="top")
    y -= 0.020
    for line in lines[1:]:
        fig.text(x + (0.018 if bullet else 0), y, line, fontsize=size, color=color, va="top")
        y -= 0.020
    return y


def _heading(fig, y: float, text: str, *, x: float = 0.065, color: str = COLORS["teal"], size: float = 12.0) -> float:
    fig.text(x, y, _clean(text), fontsize=size, color=color, weight="bold", va="top")
    return y - 0.032


def _card_pages(pdf: PdfPages, title: str, cards: list[dict[str, object]], page_number: int, *, subtitle: str = "") -> int:
    fig = _new_page(pdf, title, subtitle)
    y = 0.87
    for card in cards:
        if y < 0.13:
            page_number = _save(pdf, fig, page_number)
            fig = _new_page(pdf, title + " (cont.)", subtitle)
            y = 0.87
        y = _heading(fig, y, str(card.get("heading") or ""))
        for paragraph in list(card.get("paragraphs") or []):
            y = _body(fig, 0.075, y, str(paragraph), width=92, size=9.0)
            y -= 0.006
        for bullet in list(card.get("bullets") or []):
            y = _body(fig, 0.085, y, str(bullet), width=86, size=8.7, color=COLORS["muted"], bullet=True)
        y -= 0.025
    return _save(pdf, fig, page_number)


def _cover(pdf: PdfPages, data: dict[str, object], page_number: int) -> int:
    fig = _new_page(pdf, "Special Features And Highlights", str(data.get("eyebrow") or "Executive Feature Report"))
    fig.text(0.065, 0.87, str(data.get("title") or ""), fontsize=24, color=COLORS["ink"], weight="bold", va="top")
    y = 0.82
    for subtitle in list(data.get("subtitles") or [])[:3]:
        y = _body(fig, 0.065, y, str(subtitle), width=92, size=10.0, color=COLORS["muted"])
        y -= 0.012
    y -= 0.008
    fig.text(0.065, y, "Proof Snapshot", fontsize=13, color=COLORS["ink"], weight="bold", va="top")
    y -= 0.035
    metrics = list(data.get("metrics") or [])
    for idx, metric in enumerate(metrics[:8]):
        col = idx % 2
        row = idx // 2
        x = 0.065 + col * 0.43
        yy = y - row * 0.098
        fig.patches.append(Rectangle((x, yy - 0.073), 0.40, 0.068, transform=fig.transFigure, facecolor="white", edgecolor=COLORS["line"], linewidth=1))
        fig.text(x + 0.014, yy - 0.018, str(metric.get("label") or ""), fontsize=8.5, color=COLORS["muted"], weight="bold", va="top")
        fig.text(x + 0.014, yy - 0.041, str(metric.get("value") or ""), fontsize=13.5, color=COLORS["purple"], weight="bold", va="top")
        fig.text(x + 0.165, yy - 0.042, str(metric.get("detail") or ""), fontsize=8.2, color=COLORS["muted"], va="top")
    return _save(pdf, fig, page_number)


def _lineup_page(pdf: PdfPages, data: dict[str, object], page_number: int) -> int:
    fig = _new_page(pdf, "Current Active Lineup")
    y = 0.87
    for cells in list(data.get("lineup") or [])[:10]:
        y = _heading(fig, y, cells[0], color=COLORS["blue"], size=10.5)
        detail = " | ".join(cells[1:])
        y = _body(fig, 0.080, y, detail, width=92, size=8.8, color=COLORS["muted"])
        y -= 0.018
    return _save(pdf, fig, page_number)


def build_pdf(data: dict[str, object], pdf_path: Path) -> int:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    page_number = 1
    with PdfPages(pdf_path) as pdf:
        page_number = _cover(pdf, data, page_number)
        page_number = _card_pages(pdf, "Executive Summary", list(data.get("summary") or []), page_number)
        page_number = _card_pages(pdf, "Feature Proof Surface", list(data.get("features") or []), page_number)
        highlights = [{"heading": "Current Highlights", "paragraphs": [], "bullets": list(data.get("highlights") or [])[:10]}]
        page_number = _card_pages(pdf, "Current Highlights", highlights, page_number)
        page_number = _lineup_page(pdf, data, page_number)
        page_number = _card_pages(pdf, "Recommendations", list(data.get("recommendations") or []), page_number)
    return page_number - 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the special-features HTML into a deterministic readable PDF.")
    parser.add_argument("--html", default=str(DEFAULT_HTML))
    parser.add_argument("--pdf", default=str(DEFAULT_PDF))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    html_path = Path(args.html).expanduser()
    pdf_path = Path(args.pdf).expanduser()
    data = parse_special_features(html_path)
    pages = build_pdf(data, pdf_path)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ok": bool(pdf_path.exists() and pdf_path.stat().st_size > 10_000),
        "html_path": str(html_path),
        "pdf_path": str(pdf_path),
        "pdf_bytes": int(pdf_path.stat().st_size) if pdf_path.exists() else 0,
        "page_count": int(pages),
        "renderer": "matplotlib_text_layout",
    }
    out = Path(args.json_out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(f"special_features_pdf ok={int(payload['ok'])} pages={pages} path={pdf_path}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
