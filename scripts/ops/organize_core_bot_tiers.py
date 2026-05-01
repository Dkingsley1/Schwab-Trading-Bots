#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "core_bot_tier_organizer_latest.json"
TIER_ROOT = PROJECT_ROOT / "core" / "bot_tiers"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _tier_for_row(row: dict[str, Any]) -> str:
    bot_id = str(row.get("bot_id") or "").lower()
    role = str(row.get("bot_role") or "").lower()
    category = str(row.get("category") or "").lower()
    source = str(row.get("source") or "").lower()
    if "grand_master" in bot_id or "grand_master" in source:
        return "00_grand_master"
    if bot_id in {"master_bot"} or "master_bot" in bot_id or "master_" in bot_id:
        return "01_master"
    if "ops_infrastructure" in category or role == "ops_infrastructure_bot":
        return "04_ops_infrastructure"
    if "infrastructure" in role or category == "infrastructure":
        return "02_infrastructure"
    return "03_sub_bots"


def _target_path(project_root: Path, row: dict[str, Any]) -> Path | None:
    raw = str(row.get("core_file") or row.get("source") or "")
    if not raw:
        return None
    path = project_root / raw
    return path if path.exists() else None


def _safe_link_name(row: dict[str, Any], target: Path) -> str:
    bot_id = str(row.get("bot_id") or target.stem).strip() or target.stem
    suffix = target.suffix or ".py"
    return f"{bot_id}{suffix}"


def _clear_owned_links(tier_dir: Path) -> None:
    tier_dir.mkdir(parents=True, exist_ok=True)
    for path in tier_dir.iterdir():
        if path.name in {"README.md", "__init__.py"}:
            continue
        if path.is_symlink():
            path.unlink()


def _write_tier_index(tier_dir: Path, title: str, rows: list[dict[str, Any]]) -> None:
    lines = [
        f"# {title}",
        "",
        "PyCharm tier view generated from `master_bot_registry.json` and `core/bot_catalog.json`.",
        "",
        "| Bot | Role | State | Source |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        state = "active" if row.get("active") is True else "inactive" if row.get("active") is False else str(row.get("lifecycle_state") or "")
        if row.get("data_collection_active"):
            state += " / collecting"
        lines.append(
            "| {bot} | {role} | {state} | {source} |".format(
                bot=str(row.get("bot_id") or ""),
                role=str(row.get("bot_role") or ""),
                state=state,
                source=str(row.get("core_file") or row.get("source") or ""),
            )
        )
    tier_dir.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    tier_dir.joinpath("__init__.py").write_text('"""Generated PyCharm bot tier view."""\n', encoding="utf-8")


def build_tier_view(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    catalog = _load_json(project_root / "core" / "bot_catalog.json")
    rows = [row for row in catalog.get("bots", []) if isinstance(row, dict)]
    manual_rows = [
        {
            "bot_id": "master_bot",
            "bot_role": "master_control_bot",
            "active": None,
            "lifecycle_state": "master_control",
            "category": "master",
            "core_file": "core/master_bot.py",
            "source": "core/master_bot.py",
        }
    ]
    if (project_root / "core" / "grand_master_bot.py").exists():
        manual_rows.append(
            {
                "bot_id": "grand_master_bot",
                "bot_role": "grand_master_control_bot",
                "active": None,
                "lifecycle_state": "grand_master_control",
                "category": "grand_master",
                "core_file": "core/grand_master_bot.py",
                "source": "core/grand_master_bot.py",
            }
        )
    rows = manual_rows + rows

    tier_titles = {
        "00_grand_master": "Grand Master",
        "01_master": "Master",
        "02_infrastructure": "Infrastructure Bots",
        "03_sub_bots": "Sub Bots",
        "04_ops_infrastructure": "Ops Infrastructure",
    }
    grouped: dict[str, list[dict[str, Any]]] = {key: [] for key in tier_titles}
    for row in rows:
        grouped.setdefault(_tier_for_row(row), []).append(row)

    TIER_ROOT.mkdir(parents=True, exist_ok=True)
    created_links: list[str] = []
    skipped_missing: list[str] = []
    for tier, title in tier_titles.items():
        tier_dir = project_root / "core" / "bot_tiers" / tier
        _clear_owned_links(tier_dir)
        tier_rows = sorted(grouped.get(tier, []), key=lambda row: str(row.get("bot_id") or ""))
        for row in tier_rows:
            target = _target_path(project_root, row)
            if target is None:
                skipped_missing.append(str(row.get("bot_id") or ""))
                continue
            link = tier_dir / _safe_link_name(row, target)
            if link.exists() or link.is_symlink():
                continue
            link.symlink_to(Path(os.path.relpath(target, start=tier_dir)))
            created_links.append(str(link.relative_to(project_root)))
        _write_tier_index(tier_dir, title, tier_rows)

    readme_lines = [
        "# Bot Tiers",
        "",
        "Generated PyCharm view. The canonical Python files stay in `core/`; these folders are tiered links for navigation.",
        "",
        "- `00_grand_master`: grand master control layer when present",
        "- `01_master`: master control and master-style coordinator bots",
        "- `02_infrastructure`: guards, sentinels, allocators, routers, and runtime controls",
        "- `03_sub_bots`: signal, options, futures, crypto, dividend, conservative, day-trading, and swing bots",
        "- `04_ops_infrastructure`: operational scripts indexed as infrastructure bots",
        "",
    ]
    (project_root / "core" / "bot_tiers" / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")

    summary = {
        tier: len(rows)
        for tier, rows in grouped.items()
    }
    return {
        "overall_status": "ready",
        "generated_at_utc": _utc_now(),
        "tier_root": str((project_root / "core" / "bot_tiers").relative_to(project_root)),
        "summary": summary,
        "created_link_count": len(created_links),
        "skipped_missing_count": len(skipped_missing),
        "skipped_missing": skipped_missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create PyCharm tier folders for core bots without moving canonical files.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    payload = build_tier_view(project_root)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = project_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        print(f"core_bot_tier_organizer status={payload['overall_status']} links={payload['created_link_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
