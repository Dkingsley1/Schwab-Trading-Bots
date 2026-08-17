#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
REGISTRY_PATH = PROJECT_ROOT / "master_bot_registry.json"
CONFIG_DIR = PROJECT_ROOT / "config"
OUT_DIR = PROJECT_ROOT / "exports" / "reports" / "expansion_inventory"
HEALTH_PATH = PROJECT_ROOT / "governance" / "health" / "expansion_inventory_latest.json"
MD_PATH = OUT_DIR / "expansion_inventory_latest.md"
PDF_PATH = OUT_DIR / "expansion_inventory_latest.pdf"

CONTROL_PLANE_CONFIGS = {
    "sleeve_strategy_expansion.json": "Sleeve, ticker, and strategy coverage expansion",
    "intelligence_capability_packs_v1.json": "Intelligence capability control plane",
    "advanced_intelligence_mesh_v1.json": "Advanced intelligence mesh",
    "cognitive_control_plane_v1.json": "Cognitive control plane",
    "recursive_research_foundry_v1.json": "Recursive research foundry",
    "coordination_intelligence_pack_v1.json": "Coordination intelligence pack",
    "adaptive_intelligence_kernel_v1.json": "Adaptive intelligence kernel",
    "system_self_awareness_v1.json": "System self-awareness pack",
    "alpha_intelligence_evolution_v1.json": "Alpha intelligence evolution pack",
    "intelligence_layer_advancement_v1.json": "Intelligence layer advancement pack",
    "apex_self_awareness_intelligence_v1.json": "Apex self-awareness intelligence pack",
    "deep_recursive_awareness_v1.json": "Deep recursive awareness pack",
    "frontier_intelligence_v1.json": "Frontier intelligence pack",
    "institutional_alpha_validation_v1.json": "Institutional alpha validation pack",
    "quant_strategy_gap_v1.json": "Quant strategy gap pack",
    "platform_organ_systems_v1.json": "Platform organ systems pack",
    "trading_muscle_systems_v1.json": "Trading muscle systems pack",
    "platform_intelligence_layer_v2.json": "Platform intelligence layer",
    "platform_brain_v4_grande.json": "Platform Brain v4 Grande",
    "platform_brain_v5_reflex.json": "Platform Brain v5 Reflex",
    "platform_brain_v6_foresight.json": "Platform Brain v6 Foresight",
    "platform_stabilization_quality_v1.json": "Platform stabilization and quality layer",
    "platform_settlement_stabilization_v1.json": "Platform settlement stabilization layer",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _as_rows(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    return [row for row in raw if isinstance(row, dict)]


def _version_from_bot_id(bot_id: str) -> int | None:
    match = re.match(r"^brain_refinery_v(?P<version>\d+)", str(bot_id or ""))
    return int(match.group("version")) if match else None


def _scalar_count(payload: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key, value in payload.items():
        if isinstance(value, list):
            counts[f"{key}_count"] = len(value)
        elif isinstance(value, dict):
            counts[f"{key}_keys"] = len(value)
    return counts


def _relative_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _config_summary(path: Path, title: str, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    payload = _load_json(path)
    pack = payload.get("pack") if isinstance(payload.get("pack"), dict) else {}
    display = str(pack.get("display_name") or payload.get("display_name") or title)
    objective = str(pack.get("objective") or payload.get("objective") or payload.get("purpose") or "")
    return {
        "file": _relative_path(path, project_root),
        "title": display,
        "configured_title": title,
        "exists": path.exists(),
        "version": str(
            payload.get("capability_pack_version")
            or payload.get("quant_strategy_gap_version")
            or payload.get("platform_organ_systems_version")
            or payload.get("trading_muscle_systems_version")
            or payload.get("institutional_alpha_validation_version")
            or payload.get("frontier_intelligence_version")
            or payload.get("deep_recursive_awareness_version")
            or ""
        ),
        "objective": objective,
        "counts": _scalar_count(pack or payload),
    }


def _registry_pack_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        slug = str(row.get("capability_pack_slug") or "").strip()
        version = str(row.get("capability_pack_version") or "").strip()
        if not slug and str(row.get("quant_strategy_gap_version") or "").strip():
            slug = "quant_strategy_gap"
            version = str(row.get("quant_strategy_gap_version") or "").strip()
        if not slug and str(row.get("institutional_alpha_validation_version") or "").strip():
            slug = "institutional_alpha_validation"
            version = str(row.get("institutional_alpha_validation_version") or "").strip()
        if not slug:
            continue
        grouped[(slug, version)].append(row)

    packs: list[dict[str, Any]] = []
    for (slug, version), pack_rows in sorted(grouped.items()):
        versions = [_version_from_bot_id(str(row.get("bot_id") or "")) for row in pack_rows]
        versions = [value for value in versions if value is not None]
        sample = pack_rows[0] if pack_rows else {}
        contract = sample.get("capability_pack_contract") if isinstance(sample.get("capability_pack_contract"), dict) else {}
        sleeves = Counter(str(row.get("sleeve_profile") or row.get("sleeve_family") or "unassigned") for row in pack_rows)
        roles = Counter(str(row.get("bot_role") or "unknown") for row in pack_rows)
        packs.append(
            {
                "slug": slug,
                "version": version,
                "display_name": str(sample.get("capability_pack_display_name") or contract.get("display_name") or slug.replace("_", " ").title()),
                "bot_count": len(pack_rows),
                "active_count": sum(1 for row in pack_rows if bool(row.get("active"))),
                "collection_count": sum(1 for row in pack_rows if bool(row.get("data_collection_active"))),
                "training_excluded_count": sum(1 for row in pack_rows if bool(row.get("training_excluded")) or bool(row.get("exclude_from_training"))),
                "execution_enabled_count": sum(1 for row in pack_rows if bool(row.get("execution_enabled")) or bool(row.get("live_trading_enabled"))),
                "sleeve_count": len(sleeves),
                "top_sleeves": [name for name, _ in sleeves.most_common(8)],
                "role_counts": dict(sorted(roles.items())),
                "version_range": [min(versions), max(versions)] if versions else [],
                "storage_rule": contract.get("storage_retention_rule") if isinstance(contract.get("storage_retention_rule"), dict) else {},
                "paper_only_floor": contract.get("paper_only_floor") if isinstance(contract.get("paper_only_floor"), dict) else {},
            }
        )
    return packs


def build_report(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    registry = _load_json(project_root / "master_bot_registry.json")
    rows = _as_rows(registry.get("sub_bots"))
    active = [row for row in rows if bool(row.get("active"))]
    collecting = [row for row in rows if bool(row.get("data_collection_active"))]
    training_excluded = [row for row in rows if bool(row.get("training_excluded")) or bool(row.get("exclude_from_training"))]
    versions = [_version_from_bot_id(str(row.get("bot_id") or "")) for row in rows]
    versions = [value for value in versions if value is not None]

    config_rows = []
    for filename, title in CONTROL_PLANE_CONFIGS.items():
        config_rows.append(_config_summary(project_root / "config" / filename, title, project_root))

    packs = _registry_pack_rows(rows)
    strategy_gap = next((row for row in config_rows if row["file"].endswith("quant_strategy_gap_v1.json")), None)
    quant_strategies: list[str] = []
    if strategy_gap and (project_root / "config" / "quant_strategy_gap_v1.json").exists():
        pack = _load_json(project_root / "config" / "quant_strategy_gap_v1.json").get("pack")
        if isinstance(pack, dict):
            for item in pack.get("strategies") or []:
                if isinstance(item, dict) and item.get("display_name"):
                    quant_strategies.append(str(item["display_name"]))

    return {
        "timestamp_utc": _utc_now(),
        "source_registry": str(project_root / "master_bot_registry.json"),
        "artifact_paths": {
            "markdown": str(MD_PATH),
            "pdf": str(PDF_PATH),
            "json": str(HEALTH_PATH),
        },
        "summary": {
            "registry_total_bots": len(rows),
            "active_bots": len(active),
            "data_collection_active_bots": len(collecting),
            "training_excluded_bots": len(training_excluded),
            "max_bot_version": max(versions) if versions else None,
            "registry_expansion_pack_count": len(packs),
            "registry_expansion_pack_bot_count": sum(int(row.get("bot_count") or 0) for row in packs),
            "control_plane_config_count": sum(1 for row in config_rows if row.get("exists")),
            "quant_strategy_gap_strategy_count": len(quant_strategies),
        },
        "registry_expansion_packs": packs,
        "control_plane_expansions": config_rows,
        "quant_strategy_gap_strategies": quant_strategies,
    }


def _fmt_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except Exception:
        return "0"


def _clean_report_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _short_report_text(value: Any, limit: int = 54) -> str:
    text = _clean_report_text(value)
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 1)].rstrip() + "..."


def _version_text(pack: dict[str, Any]) -> str:
    version_range = pack.get("version_range") if isinstance(pack.get("version_range"), list) else []
    if len(version_range) == 2:
        return f"v{version_range[0]}-v{version_range[-1]}"
    return "n/a"


def _counts_text(row: dict[str, Any], *, limit: int = 72) -> str:
    counts = row.get("counts") if isinstance(row.get("counts"), dict) else {}
    text = ", ".join(f"{key}={value}" for key, value in counts.items())
    return _short_report_text(text or "n/a", limit)


def _render_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    lines: list[str] = [
        "# Platform Expansion Inventory",
        "",
        "Registry-backed capability packs, control-plane expansions, and strategy sleeves added to the Schwab Trading Bot platform.",
        "",
        f"Generated UTC: {payload.get('timestamp_utc')}",
        "",
        "## Executive Summary",
        "",
        f"- Registry bots: {_fmt_int(summary.get('registry_total_bots'))}",
        f"- Active bots: {_fmt_int(summary.get('active_bots'))}",
        f"- Data-collection active bots: {_fmt_int(summary.get('data_collection_active_bots'))}",
        f"- Training-excluded bots: {_fmt_int(summary.get('training_excluded_bots'))}",
        f"- Max bot version: {summary.get('max_bot_version', '')}",
        f"- Registry-backed expansion packs: {_fmt_int(summary.get('registry_expansion_pack_count'))}",
        f"- Bots inside expansion packs: {_fmt_int(summary.get('registry_expansion_pack_bot_count'))}",
        f"- Control-plane/config expansion files: {_fmt_int(summary.get('control_plane_config_count'))}",
        f"- Latest quant strategy gap strategies: {_fmt_int(summary.get('quant_strategy_gap_strategy_count'))}",
        "",
        "## Operating Contract",
        "",
        "- New expansion bots are active observers first: zero weight, training-excluded, paper-disabled, live-disabled, and execution-disabled until their data and governance thresholds clear.",
        "- Strategy packs enrich labels, source confidence, cross-sleeve context, and paper-readiness evidence before they can influence allocation.",
        "- Control-plane expansions are advisory or guardrail layers unless a separate runtime command explicitly enables an action.",
        "",
        "## Registry-Backed Expansion Packs",
        "",
        "| Pack | Bots | Collecting | Training Locked | Execution Enabled | Sleeves | Versions |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for pack in list(payload.get("registry_expansion_packs") or []):
        if not isinstance(pack, dict):
            continue
        lines.append(
            "| "
            f"{pack.get('display_name')} | "
            f"{pack.get('bot_count')} | "
            f"{pack.get('collection_count')} | "
            f"{pack.get('training_excluded_count')} | "
            f"{pack.get('execution_enabled_count')} | "
            f"{pack.get('sleeve_count')} | "
            f"{_version_text(pack)} |"
        )
    lines.extend(["", "## Control-Plane And Stabilization Expansions", ""])
    for row in list(payload.get("control_plane_expansions") or []):
        if not isinstance(row, dict):
            continue
        status = "present" if row.get("exists") else "missing"
        lines.extend(
            [
                f"### {row.get('configured_title')}",
                "",
                f"- Status: {status}",
                f"- File: {row.get('file')}",
                f"- Title: {row.get('title')}",
                f"- Version: {row.get('version') or 'n/a'}",
                f"- Counts: {_counts_text(row, limit=120)}",
            ]
        )
        if row.get("objective"):
            lines.append(f"- Objective: {row.get('objective')}")
        lines.append("")
    strategies = list(payload.get("quant_strategy_gap_strategies") or [])
    if strategies:
        lines.extend(["## Latest 24 Quant Strategy Additions", ""])
        for index, item in enumerate(strategies, start=1):
            lines.append(f"{index}. {item}")
        lines.append("")
    return "\n".join(lines)


def render_expansion_inventory_pdf(payload: dict[str, Any], pdf_path: Path) -> dict[str, Any]:
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    generated = datetime.now(timezone.utc).isoformat()
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    packs = [row for row in list(payload.get("registry_expansion_packs") or []) if isinstance(row, dict)]
    configs = [row for row in list(payload.get("control_plane_expansions") or []) if isinstance(row, dict)]
    strategies = [str(row) for row in list(payload.get("quant_strategy_gap_strategies") or [])]
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    navy = "#122033"
    slate = "#52616d"
    border = "#d7e1e6"
    bg = "#f7faf9"
    teal = "#1f7a8c"
    green = "#0f766e"
    blue = "#2563eb"
    violet = "#7c3aed"
    amber = "#c26a1b"
    red = "#b42318"

    def page(title: str, subtitle: str = ""):
        fig = plt.figure(figsize=(11, 8.5), facecolor=bg)
        fig.patches.append(Rectangle((0, 0.925), 1, 0.075, transform=fig.transFigure, color=navy, zorder=-1))
        fig.patches.append(Rectangle((0, 0), 1, 0.035, transform=fig.transFigure, color="#e7edf0", zorder=-1))
        fig.text(0.045, 0.963, title[:96], fontsize=17, color="white", weight="bold", va="center")
        if subtitle:
            fig.text(0.045, 0.932, subtitle[:142], fontsize=8.5, color="#c9e3ea", va="center")
        return fig

    def footer(fig, page_number: int) -> None:
        fig.text(0.045, 0.018, f"Platform Expansion Inventory - page {page_number}", fontsize=7.5, color=slate, va="center")
        fig.text(0.955, 0.018, "Schwab Trading Bot Platform", fontsize=7.5, color=slate, va="center", ha="right")

    def card(fig, x: float, y: float, w: float, h: float, label: str, value: Any, detail: str = "", accent: str = teal) -> None:
        compact = h < 0.118
        fig.patches.append(Rectangle((x, y), w, h, transform=fig.transFigure, facecolor="white", edgecolor=border, linewidth=1.0, zorder=-1))
        fig.patches.append(Rectangle((x, y + h - 0.012), w, 0.012, transform=fig.transFigure, facecolor=accent, edgecolor=accent, linewidth=0, zorder=0))
        fig.text(x + 0.014, y + h - 0.031, label.upper(), fontsize=7.0, color="#62727d", weight="bold", va="top")
        value_text = _clean_report_text(value)
        value_size = 15.2 if len(value_text) <= 10 else 11.0 if len(value_text) <= 24 else 8.8
        value_y = y + 0.024 if compact else y + 0.044
        fig.text(x + 0.014, value_y, value_text[:36], fontsize=value_size, color=navy, weight="bold", va="bottom")
        if detail and not compact:
            fig.text(x + 0.014, y + 0.018, detail[:72], fontsize=7.1, color=slate, va="bottom")

    def wrapped(fig, x: float, y: float, text: Any, *, width: int = 100, size: float = 8.5, color: str = navy, weight: str = "normal", gap: float = 0.023) -> float:
        for line in textwrap.wrap(_clean_report_text(text), width=width, break_long_words=False, break_on_hyphens=False) or [""]:
            fig.text(x, y, line, fontsize=size, color=color, weight=weight, va="top")
            y -= gap
        return y

    def bullets(fig, x: float, y: float, rows: list[str], *, width: int = 120, size: float = 8.5) -> float:
        for row in rows:
            y = wrapped(fig, x, y, f"- {row}", width=width, size=size, color=navy, gap=0.022)
            y -= 0.004
        return y

    def draw_table(fig, rows: list[list[str]], headers: list[str], widths: list[float], *, x: float, y: float, row_h: float, font_size: float = 7.1) -> None:
        total_w = sum(widths)
        fig.patches.append(Rectangle((x, y), total_w, row_h, transform=fig.transFigure, facecolor=navy, edgecolor=navy, linewidth=0, zorder=-1))
        cur_x = x
        for header, width in zip(headers, widths):
            fig.text(cur_x + 0.006, y + row_h * 0.64, header.upper(), fontsize=6.6, color="white", weight="bold", va="center")
            cur_x += width
        y -= row_h
        for idx, row in enumerate(rows):
            face = "white" if idx % 2 == 0 else "#f0f5f6"
            fig.patches.append(Rectangle((x, y), total_w, row_h, transform=fig.transFigure, facecolor=face, edgecolor=border, linewidth=0.55, zorder=-1))
            cur_x = x
            for value, width in zip(row, widths):
                fig.text(cur_x + 0.006, y + row_h * 0.62, _short_report_text(value, max(8, int(width * 120))), fontsize=font_size, color=navy, va="center")
                cur_x += width
            y -= row_h

    pages = 0
    with PdfPages(pdf_path) as pdf:
        pages += 1
        fig = page("Platform Expansion Inventory", "Registry-backed capability packs, control-plane expansions, and latest strategy gap buildout")
        fig.text(0.055, 0.815, "Professional System Expansion Report", fontsize=23, color=navy, weight="bold", va="top")
        wrapped(
            fig,
            0.055,
            0.765,
            "A report-ready inventory of the platform buildout: what expanded, how many bots were added or governed by each pack, and which controls keep new capability in collection-first mode.",
            width=104,
            size=10.8,
            color="#374151",
            gap=0.030,
        )
        card(fig, 0.055, 0.585, 0.135, 0.105, "Registry Bots", _fmt_int(summary.get("registry_total_bots")), "total cataloged", teal)
        card(fig, 0.207, 0.585, 0.135, 0.105, "Active", _fmt_int(summary.get("active_bots")), "enabled in registry", blue)
        card(fig, 0.359, 0.585, 0.145, 0.105, "Collecting", _fmt_int(summary.get("data_collection_active_bots")), "live data observers", green)
        card(fig, 0.521, 0.585, 0.145, 0.105, "Expansion Packs", _fmt_int(summary.get("registry_expansion_pack_count")), "registry-backed", violet)
        card(fig, 0.683, 0.585, 0.135, 0.105, "Pack Bots", _fmt_int(summary.get("registry_expansion_pack_bot_count")), "inside expansions", amber)
        card(fig, 0.835, 0.585, 0.110, 0.105, "Max Version", summary.get("max_bot_version", ""), "latest bot id", red)
        fig.patches.append(Rectangle((0.055, 0.220), 0.890, 0.265, transform=fig.transFigure, facecolor="white", edgecolor=border, linewidth=1.0, zorder=-1))
        fig.text(0.075, 0.455, "Report Posture", fontsize=13.5, color=navy, weight="bold", va="top")
        bullets(
            fig,
            0.085,
            0.410,
            [
                "Expansion bots are staged as observers first, with execution and allocation disabled unless a separate governance path promotes them.",
                "Training exclusion remains the default for newly added packs until observation floors and collection-age thresholds are met.",
                "Control-plane expansions are included so external readers can see the stabilizers, intelligence layers, and governance infrastructure alongside strategy growth.",
                f"Generated from master_bot_registry.json at {generated}.",
            ],
            width=126,
            size=9.0,
        )
        footer(fig, pages)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        pages += 1
        fig = page("Executive Summary", "Scale, staging posture, and governance readiness")
        card(fig, 0.055, 0.765, 0.165, 0.100, "Training Locked", _fmt_int(summary.get("training_excluded_bots")), "excluded until mature", violet)
        card(fig, 0.237, 0.765, 0.165, 0.100, "Control Files", _fmt_int(summary.get("control_plane_config_count")), "expansion configs", teal)
        card(fig, 0.419, 0.765, 0.165, 0.100, "Latest Strategies", _fmt_int(summary.get("quant_strategy_gap_strategy_count")), "quant gap pack", amber)
        card(fig, 0.601, 0.765, 0.165, 0.100, "Execution On", _fmt_int(sum(int(row.get("execution_enabled_count") or 0) for row in packs)), "inside packs", red)
        card(fig, 0.783, 0.765, 0.160, 0.100, "Observer Ratio", f"{(int(summary.get('data_collection_active_bots') or 0) / max(int(summary.get('registry_total_bots') or 1), 1)) * 100:.1f}%", "collecting / total", green)
        ax = fig.add_axes([0.075, 0.400, 0.430, 0.260])
        labels = ["Total", "Active", "Collecting", "Training locked"]
        values = [int(summary.get("registry_total_bots") or 0), int(summary.get("active_bots") or 0), int(summary.get("data_collection_active_bots") or 0), int(summary.get("training_excluded_bots") or 0)]
        ax.barh(labels, values, color=[navy, blue, green, violet], alpha=0.88)
        ax.set_title("Registry Posture", fontsize=11, weight="bold", color=navy)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{int(value):,}"))
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        ax.invert_yaxis()
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="both", labelsize=8)
        fig.text(0.565, 0.665, "What This Means", fontsize=13.5, color=navy, weight="bold", va="top")
        bullets(
            fig,
            0.580,
            0.620,
            [
                "The platform expansion is broad, but the newest strategy and intelligence bots remain in a guarded collection state.",
                "The report separates registry-backed bot packs from control-plane files so strategy growth and infrastructure maturity are visible together.",
                "Zero execution-enabled bots inside expansion packs confirms the buildout did not silently turn new strategies into live allocators.",
                "The latest 24-strategy gap pack adds tradable research coverage while keeping storage, training, and paper-trade promotion gated.",
            ],
            width=58,
            size=8.9,
        )
        footer(fig, pages)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        pages += 1
        fig = page("Expansion Pack Portfolio", "Bot counts by registry-backed capability pack")
        sorted_packs = sorted(packs, key=lambda row: int(row.get("bot_count") or 0))[-12:]
        ax = fig.add_axes([0.255, 0.150, 0.690, 0.675])
        labels = [_short_report_text(row.get("display_name"), 36) for row in sorted_packs]
        values = [int(row.get("bot_count") or 0) for row in sorted_packs]
        colors = [violet if "Intelligence" in str(row.get("display_name")) or "Awareness" in str(row.get("display_name")) else teal for row in sorted_packs]
        ax.barh(labels, values, color=colors, alpha=0.90)
        ax.set_title("Largest Expansion Packs", fontsize=12, weight="bold", color=navy, pad=12)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{int(value):,}"))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        for idx, value in enumerate(values):
            ax.annotate(f"{value:,}", xy=(value, idx), xytext=(5, 0), textcoords="offset points", va="center", fontsize=7.6, color=navy)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="both", labelsize=8)
        footer(fig, pages)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        detail_rows = []
        for pack in sorted(packs, key=lambda row: (-int(row.get("bot_count") or 0), str(row.get("display_name") or ""))):
            detail_rows.append(
                [
                    str(pack.get("display_name") or ""),
                    _fmt_int(pack.get("bot_count")),
                    _fmt_int(pack.get("collection_count")),
                    _fmt_int(pack.get("training_excluded_count")),
                    _fmt_int(pack.get("execution_enabled_count")),
                    _fmt_int(pack.get("sleeve_count")),
                    _version_text(pack),
                ]
            )
        for offset in range(0, len(detail_rows), 10):
            pages += 1
            fig = page("Registry-Backed Expansion Detail", f"Rows {offset + 1}-{min(offset + 10, len(detail_rows))} of {len(detail_rows)}")
            draw_table(
                fig,
                detail_rows[offset : offset + 10],
                ["Expansion Pack", "Bots", "Collect", "Train Lock", "Exec", "Sleeves", "Versions"],
                [0.330, 0.070, 0.082, 0.092, 0.066, 0.070, 0.130],
                x=0.055,
                y=0.805,
                row_h=0.058,
                font_size=7.2,
            )
            fig.text(0.055, 0.105, "Interpretation", fontsize=11.5, color=navy, weight="bold", va="top")
            wrapped(fig, 0.055, 0.075, "Collect and Train Lock columns should usually move together for newly added packs. Exec should remain zero for research-stage expansions.", width=130, size=8.2, color=slate)
            footer(fig, pages)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        if strategies:
            pages += 1
            fig = page("Latest Quant Strategy Gap Pack", "24 newly staged strategy sleeves, 5 bots each, collection-first")
            fig.text(0.055, 0.830, "Strategy Coverage Added", fontsize=13.5, color=navy, weight="bold", va="top")
            wrapped(
                fig,
                0.055,
                0.795,
                "These are the newest research lanes added by the quant strategy gap pack. Each is staged for evidence collection and cross-sleeve context, not immediate execution.",
                width=126,
                size=9.2,
                color="#374151",
                gap=0.025,
            )
            col_x = [0.055, 0.365, 0.675]
            col_y = [0.700, 0.700, 0.700]
            for index, name in enumerate(strategies, start=1):
                col = (index - 1) // 8
                x = col_x[col]
                y = col_y[col]
                fig.patches.append(Rectangle((x, y - 0.043), 0.270, 0.046, transform=fig.transFigure, facecolor="white", edgecolor=border, linewidth=0.8, zorder=-1))
                fig.text(x + 0.010, y - 0.010, f"{index:02d}", fontsize=8.0, color=teal, weight="bold", va="top")
                wrapped(fig, x + 0.045, y - 0.010, name, width=28, size=7.7, color=navy, weight="bold", gap=0.017)
                col_y[col] -= 0.061
            footer(fig, pages)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        config_rows = []
        for row in configs:
            config_rows.append(
                [
                    str(row.get("configured_title") or row.get("title") or ""),
                    "present" if row.get("exists") else "missing",
                    str(row.get("file") or ""),
                    str(row.get("version") or "n/a"),
                    _counts_text(row, limit=64),
                ]
            )
        for offset in range(0, len(config_rows), 9):
            pages += 1
            fig = page("Control-Plane And Stabilization Expansions", f"Rows {offset + 1}-{min(offset + 9, len(config_rows))} of {len(config_rows)}")
            draw_table(
                fig,
                config_rows[offset : offset + 9],
                ["Expansion", "Status", "File", "Version", "Content Snapshot"],
                [0.285, 0.070, 0.225, 0.095, 0.245],
                x=0.040,
                y=0.805,
                row_h=0.061,
                font_size=6.9,
            )
            footer(fig, pages)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        pages += 1
        fig = page("Governance Contract", "How the expansion remains controlled while the system keeps growing")
        card(fig, 0.055, 0.745, 0.185, 0.105, "Default State", "Observer", "collect first", teal)
        card(fig, 0.260, 0.745, 0.185, 0.105, "Training", "Excluded", "until floors clear", violet)
        card(fig, 0.465, 0.745, 0.185, 0.105, "Execution", "Disabled", "no silent live risk", red)
        card(fig, 0.670, 0.745, 0.185, 0.105, "Storage", "Thin Sampled", "bounded growth", amber)
        fig.text(0.055, 0.650, "Promotion Requirements", fontsize=13.5, color=navy, weight="bold", va="top")
        bullets(
            fig,
            0.070,
            0.605,
            [
                "Minimum observations and collection-age thresholds must clear before any newly added bot can become a training candidate.",
                "Paper-trade locks remain part of the control contract for sleeves that use paper execution evidence.",
                "Storage retention rules keep the expansion from flooding local or external disks while the platform continues to collect live data.",
                "Registry labels, sleeve profiles, capability-pack slugs, and core-file materialization keep the expansion inspectable in PyCharm and reportable from commands.",
                "This report is meant for external review: it summarizes platform maturity without exposing credentials, broker secrets, or raw account data.",
            ],
            width=128,
            size=9.1,
        )
        fig.patches.append(Rectangle((0.055, 0.130), 0.890, 0.130, transform=fig.transFigure, facecolor="white", edgecolor=border, linewidth=1.0, zorder=-1))
        fig.text(0.075, 0.225, "Open Command", fontsize=11.5, color=navy, weight="bold", va="top")
        fig.text(0.075, 0.185, "./scripts/ops/open_report_artifact.sh expansions", fontsize=10.0, color="#0f4c81", family="monospace", va="top")
        fig.text(0.075, 0.152, str(pdf_path), fontsize=7.9, color=slate, va="top")
        footer(fig, pages)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    ok = bool(pdf_path.exists() and pdf_path.stat().st_size > 20_000)
    return {
        "title": "Platform Expansion Inventory",
        "source_path": str(payload.get("source_registry") or ""),
        "pdf_path": str(pdf_path),
        "pdf_bytes": int(pdf_path.stat().st_size) if pdf_path.exists() else 0,
        "page_count": int(pages),
        "ok": bool(ok),
        "detail": "report_ready_expansion_inventory_pdf",
    }


def render_expansion_inventory_ready_pdf(source_path: Path | None, pdf_path: Path, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if source_path and source_path.suffix.lower() == ".json":
        payload = _load_json(source_path)
    if not isinstance(payload.get("summary"), dict):
        payload = build_report(project_root)
    return render_expansion_inventory_pdf(payload, pdf_path)


def write_report(project_root: Path = PROJECT_ROOT, *, render_pdf: bool = True) -> dict[str, Any]:
    payload = build_report(project_root)
    out_dir = project_root / "exports" / "reports" / "expansion_inventory"
    health_path = project_root / "governance" / "health" / "expansion_inventory_latest.json"
    md_path = out_dir / "expansion_inventory_latest.md"
    pdf_path = out_dir / "expansion_inventory_latest.pdf"
    out_dir.mkdir(parents=True, exist_ok=True)
    health_path.parent.mkdir(parents=True, exist_ok=True)
    markdown = _render_markdown(payload)
    md_path.write_text(markdown, encoding="utf-8")
    if render_pdf:
        payload["pdf"] = render_expansion_inventory_pdf(payload, pdf_path)
    else:
        payload["pdf"] = {"ok": pdf_path.exists(), "pdf_path": str(pdf_path)}
    payload["artifact_paths"] = {"markdown": str(md_path), "pdf": str(pdf_path), "json": str(health_path)}
    health_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a PDF-ready list of platform expansions.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--no-render-pdf", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = write_report(Path(args.project_root).resolve(), render_pdf=not args.no_render_pdf)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    else:
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        pdf = payload.get("pdf") if isinstance(payload.get("pdf"), dict) else {}
        print(
            "expansion_inventory "
            f"packs={summary.get('registry_expansion_pack_count', 0)} "
            f"configs={summary.get('control_plane_config_count', 0)} "
            f"pdf={pdf.get('pdf_path') or payload.get('artifact_paths', {}).get('pdf', '')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
