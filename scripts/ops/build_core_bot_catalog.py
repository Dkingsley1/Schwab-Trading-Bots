#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MD_OUT = PROJECT_ROOT / "core" / "BOT_CATALOG.md"
DEFAULT_JSON_OUT = PROJECT_ROOT / "core" / "bot_catalog.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _bot_version(bot_id: str) -> int:
    match = re.search(r"_v(\d+)", bot_id)
    return int(match.group(1)) if match else 0


def _core_file_for(bot_id: str, core_files: set[str]) -> str:
    exact = f"core/{bot_id}.py"
    if exact in core_files:
        return exact
    version = _bot_version(bot_id)
    if version:
        candidates = [path for path in core_files if Path(path).name.startswith(f"brain_refinery_v{version}_")]
        if len(candidates) == 1:
            return candidates[0]
    return ""


def _category_for(bot_id: str, role: str) -> str:
    text = bot_id.lower()
    if "infrastructure" in role or any(token in text for token in ("guard", "sentinel", "allocator", "router", "controller", "pruner")):
        return "infrastructure"
    if "option" in text or "gamma" in text or "iv_" in text or "straddle" in text or "strangle" in text:
        return "options"
    if "future" in text or "rates_curve" in text or "commodity" in text or "gold" in text or "oil" in text:
        return "futures_macro"
    if "dividend" in text or "reit" in text or "covered_call" in text:
        return "dividend_income"
    if "intraday" in text or "day_trading" in text or "scalp" in text or "vwap" in text or "opening_range" in text:
        return "intraday"
    if "swing" in text:
        return "swing"
    if "conservative" in text or "capital_preservation" in text or "defensive" in text:
        return "conservative"
    if "crypto" in text:
        return "crypto"
    if "macro" in text or "fed" in text or "inflation" in text or "credit" in text or "dollar" in text:
        return "macro"
    return "general_signal"


def _runner_map(project_root: Path) -> dict[str, list[str]]:
    runners = sorted(
        path
        for path in (project_root / "scripts").glob("run_*shadow.py")
        if path.is_file()
    )
    by_key: dict[str, list[str]] = {}
    for path in runners:
        rel = str(path.relative_to(project_root))
        stem = path.stem.removeprefix("run_").removesuffix("_shadow")
        tokens = [token for token in stem.split("_") if token]
        for token in tokens:
            by_key.setdefault(token, []).append(rel)
        by_key.setdefault(stem, []).append(rel)
    return by_key


def _runner_for(bot_id: str, runners_by_key: dict[str, list[str]]) -> str:
    text = bot_id.lower()
    preferred: list[str] = []
    for key, paths in runners_by_key.items():
        if key and key in text:
            preferred.extend(paths)
    seen: set[str] = set()
    unique = []
    for path in preferred:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return ", ".join(unique[:3])


def _ops_bots(project_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((project_root / "scripts" / "ops").glob("*.py")):
        name = path.stem
        if not any(token in name for token in ("bot", "guard", "autopilot", "supervisor", "watchdog", "recovery", "control", "controller")):
            continue
        rows.append(
            {
                "bot_id": name,
                "bot_role": "ops_infrastructure_bot",
                "active": None,
                "lifecycle_state": "scripted_control_plane",
                "category": "ops_infrastructure",
                "core_file": "",
                "source": str(path.relative_to(project_root)),
                "runner": "",
                "notes": "Ops/infrastructure bot; kept under scripts/ops but indexed here for PyCharm visibility.",
            }
        )
    return rows


def build_catalog(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    registry = _load_json(project_root / "master_bot_registry.json")
    sub_bots = registry.get("sub_bots") if isinstance(registry.get("sub_bots"), list) else []
    core_files = {str(path.relative_to(project_root)) for path in (project_root / "core").glob("brain_refinery*.py")}
    runners_by_key = _runner_map(project_root)

    registry_rows: list[dict[str, Any]] = []
    for row in sub_bots:
        if not isinstance(row, dict):
            continue
        bot_id = str(row.get("bot_id") or "").strip()
        if not bot_id:
            continue
        role = str(row.get("bot_role") or "").strip()
        core_file = _core_file_for(bot_id, core_files)
        registry_rows.append(
            {
                "bot_id": bot_id,
                "bot_role": role,
                "active": bool(row.get("active", False)),
                "lifecycle_state": str(row.get("lifecycle_state") or ""),
                "category": _category_for(bot_id, role),
                "core_file": core_file,
                "source": "master_bot_registry.json",
                "runner": _runner_for(bot_id, runners_by_key),
                "data_collection_active": bool(row.get("data_collection_active", False)),
                "trading_enabled": bool(row.get("trading_enabled", False)),
                "paper_trading_enabled": bool(row.get("paper_trading_enabled", False)),
                "live_trading_enabled": bool(row.get("live_trading_enabled", False)),
                "weight": row.get("weight"),
                "quality_score": row.get("quality_score"),
                "notes": "Physical core module found." if core_file else "Registry-backed bot; no dedicated core/*.py file yet.",
            }
        )

    ops_rows = _ops_bots(project_root)
    all_rows = sorted(registry_rows + ops_rows, key=lambda r: (str(r.get("category")), _bot_version(str(r.get("bot_id"))), str(r.get("bot_id"))))

    summary = {
        "generated_at_utc": _utc_now(),
        "registry_updated_at_utc": registry.get("updated_at_utc", ""),
        "registry_total_bots": len(registry_rows),
        "registry_active_bots": sum(1 for row in registry_rows if row.get("active") is True),
        "registry_data_collection_active": sum(1 for row in registry_rows if row.get("data_collection_active") is True),
        "registry_with_core_file": sum(1 for row in registry_rows if row.get("core_file")),
        "registry_without_core_file": sum(1 for row in registry_rows if not row.get("core_file")),
        "ops_infrastructure_bots": len(ops_rows),
        "total_indexed_rows": len(all_rows),
        "categories": dict(sorted(Counter(str(row.get("category") or "unknown") for row in all_rows).items())),
    }
    return {"summary": summary, "bots": all_rows}


def _md_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Bot | Role | State | Category | Location | Runner | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        active = row.get("active")
        if active is True:
            state = "active"
        elif active is False:
            state = "inactive"
        else:
            state = str(row.get("lifecycle_state") or "script")
        if row.get("data_collection_active"):
            state += " / collecting"
        location = str(row.get("core_file") or row.get("source") or "")
        lines.append(
            "| {bot} | {role} | {state} | {category} | {location} | {runner} | {notes} |".format(
                bot=str(row.get("bot_id") or ""),
                role=str(row.get("bot_role") or ""),
                state=state,
                category=str(row.get("category") or ""),
                location=location,
                runner=str(row.get("runner") or ""),
                notes=str(row.get("notes") or ""),
            )
        )
    return lines


def render_markdown(catalog: dict[str, Any]) -> str:
    summary = catalog.get("summary") if isinstance(catalog.get("summary"), dict) else {}
    rows = catalog.get("bots") if isinstance(catalog.get("bots"), list) else []
    categories = summary.get("categories") if isinstance(summary.get("categories"), dict) else {}
    out = [
        "# Core Bot Catalog",
        "",
        "This is the single PyCharm-visible index for every bot the platform knows about.",
        "",
        "Some newer bots are registry-backed collection slots, not dedicated `core/*.py` modules yet. Those rows still appear here with `master_bot_registry.json` as their source so they are visible from the `core` folder.",
        "",
        "## Summary",
        "",
        f"- Generated UTC: `{summary.get('generated_at_utc', '')}`",
        f"- Registry updated UTC: `{summary.get('registry_updated_at_utc', '')}`",
        f"- Registry bots: `{summary.get('registry_total_bots', 0)}`",
        f"- Active registry bots: `{summary.get('registry_active_bots', 0)}`",
        f"- Active data-collection bots: `{summary.get('registry_data_collection_active', 0)}`",
        f"- Registry bots with physical `core/*.py` files: `{summary.get('registry_with_core_file', 0)}`",
        f"- Registry-backed bots without dedicated core files: `{summary.get('registry_without_core_file', 0)}`",
        f"- Ops/infrastructure bots indexed here: `{summary.get('ops_infrastructure_bots', 0)}`",
        f"- Total indexed rows: `{summary.get('total_indexed_rows', 0)}`",
        "",
        "## Category Counts",
        "",
    ]
    for category, count in categories.items():
        out.append(f"- `{category}`: `{count}`")
    out.extend(["", "## All Bots", ""])
    out.extend(_md_table(rows))
    out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build core/BOT_CATALOG.md and core/bot_catalog.json.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    catalog = build_catalog(project_root)
    md_out = Path(args.md_out)
    json_out = Path(args.json_out)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(render_markdown(catalog), encoding="utf-8")
    json_out.write_text(json.dumps(catalog, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps({"ok": True, "md_out": str(md_out), "json_out": str(json_out), "summary": catalog["summary"]}, ensure_ascii=True))
    else:
        print(f"core_bot_catalog md={md_out} json={json_out} rows={catalog['summary']['total_indexed_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
