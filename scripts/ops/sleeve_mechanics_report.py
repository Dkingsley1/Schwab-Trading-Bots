#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "config" / "sleeve_strategy_expansion.json"
REGISTRY_PATH = PROJECT_ROOT / "master_bot_registry.json"
OUT_DIR = PROJECT_ROOT / "exports" / "reports" / "sleeve_mechanics"
HEALTH_PATH = PROJECT_ROOT / "governance" / "health" / "sleeve_mechanics_latest.json"
MD_PATH = OUT_DIR / "sleeve_mechanics_latest.md"

EXPANSION_PACK_FILES = (
    "quant_strategy_gap_v1.json",
    "trading_muscle_systems_v1.json",
    "exotic_quant_safe_admission_v1.json",
    "institutional_alpha_validation_v1.json",
    "frontier_intelligence_v1.json",
    "platform_organ_systems_v1.json",
)


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


def _split_csv(raw: Any) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in str(raw or "").split(","):
        token = item.strip()
        if token and token not in seen:
            seen.add(token)
            out.append(token)
    return out


def _import_runtime_maps() -> tuple[dict[str, dict[str, Any]], set[str], set[str]]:
    try:
        from scripts import run_all_sleeves
        from scripts import run_specialized_sleeve_shadow as specialized
        from scripts.ops import sleeve_strategy_coverage_guard as coverage

        defaults = getattr(specialized, "SLEEVE_DEFAULTS", {})
        all_sleeves = set(getattr(run_all_sleeves, "SPECIALIZED_SLEEVE_PROFILES", ()))
        builtin = set(getattr(coverage, "NON_SPECIALIZED_RUNTIME_SLEEVES", set()))
        return defaults if isinstance(defaults, dict) else {}, all_sleeves, builtin
    except Exception:
        return {}, set(), set()


def _registry_rows(project_root: Path) -> list[dict[str, Any]]:
    return _as_rows(_load_json(project_root / "master_bot_registry.json").get("sub_bots"))


def _bot_counts_by_sleeve(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sleeve = str(row.get("sleeve_profile") or row.get("sleeve_family") or "unassigned").strip() or "unassigned"
        grouped[sleeve].append(row)

    out: dict[str, dict[str, Any]] = {}
    for sleeve, sleeve_rows in grouped.items():
        roles = Counter(str(row.get("bot_role") or "unknown") for row in sleeve_rows)
        out[sleeve] = {
            "bot_count": len(sleeve_rows),
            "active_bot_count": sum(1 for row in sleeve_rows if bool(row.get("active"))),
            "collection_bot_count": sum(1 for row in sleeve_rows if bool(row.get("data_collection_active"))),
            "training_excluded_bot_count": sum(
                1 for row in sleeve_rows if bool(row.get("training_excluded")) or bool(row.get("exclude_from_training"))
            ),
            "role_counts": dict(sorted(roles.items())),
        }
    return out


def _candidate_names_from_pack(pack: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for key in ("strategy_sleeves", "sleeves"):
        raw = pack.get(key)
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, str) and item.strip():
                    names.append(item.strip())
    for key in ("muscles", "systems", "organs", "capabilities", "strategies"):
        raw = pack.get(key)
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    name = str(item.get("slug") or item.get("name") or "").strip()
                    if name:
                        names.append(name)
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _expansion_candidates(project_root: Path, existing_sleeves: set[str], specialized_profiles: set[str]) -> list[dict[str, Any]]:
    rows_by_name: dict[str, dict[str, Any]] = {}
    for filename in EXPANSION_PACK_FILES:
        payload = _load_json(project_root / "config" / filename)
        pack = payload.get("pack") if isinstance(payload.get("pack"), dict) else payload
        pack_name = str(pack.get("display_name") or filename)
        pack_slug = str(pack.get("slug") or filename.removesuffix(".json"))
        for name in _candidate_names_from_pack(pack):
            existing = rows_by_name.get(name)
            if existing is None:
                rows_by_name[name] = {
                    "name": name,
                    "pack_slug": pack_slug,
                    "pack_name": pack_name,
                    "pack_slugs": [pack_slug],
                    "pack_names": [pack_name],
                    "already_in_manifest": name in existing_sleeves,
                    "already_has_launcher": name in specialized_profiles,
                    "candidate_status": "wired" if name in specialized_profiles else "candidate",
                }
                continue
            if pack_slug not in existing["pack_slugs"]:
                existing["pack_slugs"].append(pack_slug)
            if pack_name not in existing["pack_names"]:
                existing["pack_names"].append(pack_name)
            existing["already_in_manifest"] = bool(existing.get("already_in_manifest", False) or name in existing_sleeves)
            existing["already_has_launcher"] = bool(existing.get("already_has_launcher", False) or name in specialized_profiles)
            existing["candidate_status"] = "wired" if bool(existing["already_has_launcher"]) else "candidate"
    return list(rows_by_name.values())


def _sleeve_row(
    project_root: Path,
    raw: dict[str, Any],
    defaults: dict[str, dict[str, Any]],
    all_sleeves_profiles: set[str],
    builtin_sleeves: set[str],
    bot_counts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    name = str(raw.get("name") or "").strip()
    strategies = [str(item) for item in raw.get("strategies", []) if str(item).strip()] if isinstance(raw.get("strategies"), list) else []
    profile = defaults.get(name, {})
    has_specialized_defaults = name in defaults
    wrapper = project_root / "scripts" / f"run_{name}_shadow.py"
    domain = str(profile.get("domain") or ("built_in_runtime" if name in builtin_sleeves else "manifest_only"))
    source_gated = str(profile.get("source_gated") or "0") == "1"
    all_sleeves_registered = name in all_sleeves_profiles
    wrapper_exists = wrapper.exists() if has_specialized_defaults else name in builtin_sleeves
    launch_mode = (
        "all_sleeves_specialized_child"
        if all_sleeves_registered
        else "built_in_parent_or_non_specialized"
        if name in builtin_sleeves
        else "manifest_only"
    )
    if has_specialized_defaults:
        execution_posture = "market_data_only_no_order_execution"
    elif name in builtin_sleeves:
        execution_posture = "parent_gated_or_paper_executor_only"
    else:
        execution_posture = "not_launchable_until_defaults_and_wrapper_exist"

    counts = bot_counts.get(name, {})
    return {
        "name": name,
        "runtime_status": str(raw.get("runtime_status") or ""),
        "domain": domain,
        "family": str(profile.get("family") or name),
        "launch_mode": launch_mode,
        "launcher_ready": bool(wrapper_exists and (all_sleeves_registered or name in builtin_sleeves)),
        "wrapper": str(wrapper.relative_to(project_root)) if wrapper.exists() else "",
        "all_sleeves_registered": all_sleeves_registered,
        "strategy_count": len(strategies),
        "strategies": strategies,
        "symbol_count": len(_split_csv(profile.get("symbols"))),
        "context_symbol_count": len(_split_csv(profile.get("context_symbols"))),
        "interval_seconds": int(str(profile.get("interval") or "0") or 0),
        "min_interval_seconds": int(str(profile.get("min_interval") or "0") or 0),
        "source_gated": source_gated,
        "source_profile": str(profile.get("source_profile") or ""),
        "correlation_peers": _split_csv(profile.get("correlation_peers")),
        "execution_posture": execution_posture,
        "bot_count": int(counts.get("bot_count") or 0),
        "active_bot_count": int(counts.get("active_bot_count") or 0),
        "collection_bot_count": int(counts.get("collection_bot_count") or 0),
        "training_excluded_bot_count": int(counts.get("training_excluded_bot_count") or 0),
        "role_counts": counts.get("role_counts") if isinstance(counts.get("role_counts"), dict) else {},
    }


def build_report(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    config = _load_json(project_root / "config" / "sleeve_strategy_expansion.json")
    sleeves = _as_rows(config.get("sleeves"))
    defaults, all_sleeves_profiles, builtin_sleeves = _import_runtime_maps()
    bot_counts = _bot_counts_by_sleeve(_registry_rows(project_root))

    sleeve_rows = [
        _sleeve_row(project_root, row, defaults, all_sleeves_profiles, builtin_sleeves, bot_counts)
        for row in sleeves
        if str(row.get("name") or "").strip()
    ]
    existing_sleeves = {str(row.get("name") or "").strip() for row in sleeves if str(row.get("name") or "").strip()}
    specialized_profiles = set(defaults)
    candidates = _expansion_candidates(project_root, existing_sleeves, specialized_profiles)
    candidate_only = [row for row in candidates if not bool(row.get("already_has_launcher"))]
    domain_counts = Counter(str(row.get("domain") or "unknown") for row in sleeve_rows)
    runtime_counts = Counter(str(row.get("runtime_status") or "unknown") for row in sleeve_rows)
    launch_ready_count = sum(1 for row in sleeve_rows if bool(row.get("launcher_ready")))

    return {
        "timestamp_utc": _utc_now(),
        "schema_version": 1,
        "artifact_paths": {
            "json": str(HEALTH_PATH),
            "markdown": str(MD_PATH),
        },
        "summary": {
            "manifest_sleeve_count": len(sleeve_rows),
            "specialized_launcher_profile_count": len(specialized_profiles),
            "all_sleeves_profile_count": len(all_sleeves_profiles),
            "launcher_ready_count": launch_ready_count,
            "strategy_count": sum(int(row.get("strategy_count") or 0) for row in sleeve_rows),
            "runtime_status_counts": dict(sorted(runtime_counts.items())),
            "domain_counts": dict(sorted(domain_counts.items())),
            "expansion_candidate_count": len(candidate_only),
        },
        "how_sleeves_work": [
            {
                "step": "manifest",
                "description": "config/sleeve_strategy_expansion.json names the sleeve, its lifecycle status, and strategy list.",
            },
            {
                "step": "launcher defaults",
                "description": "scripts/run_specialized_sleeve_shadow.py gives launchable sleeves symbols, context symbols, domain, interval, source gates, and peer sleeves.",
            },
            {
                "step": "wrapper",
                "description": "scripts/run_<sleeve>_shadow.py pins a dedicated entrypoint while reusing the shared specialized runner.",
            },
            {
                "step": "orchestration",
                "description": "scripts/run_all_sleeves.py starts each registered specialized sleeve with low workers, nice priority, and collection-only env flags.",
            },
            {
                "step": "data collection",
                "description": "the sleeve writes shadow observations and context while order execution stays disabled.",
            },
            {
                "step": "gates",
                "description": "source verification, backpressure clearance, sample depth, replay determinism, paper-trade lock, and promotion gates decide whether a sleeve can progress.",
            },
            {
                "step": "allocation",
                "description": "cross-sleeve controls and master layers can consume compressed sleeve signals; live execution remains separately gated.",
            },
        ],
        "safety_contract": {
            "new_sleeves_start_state": "data_collection_only",
            "specialized_sleeves_market_data_only": True,
            "specialized_sleeves_allow_order_execution": False,
            "training_excluded_until_ready": True,
            "paper_or_live_authority": "requires separate promotion and execution gates",
        },
        "sleeves": sleeve_rows,
        "expansion_candidates": candidate_only,
        "recommended_expansion_sequence": [
            "add the sleeve to the manifest with runtime_status=active_data_collection",
            "add specialized launcher defaults with source gates, symbols, interval, and correlation peers",
            "add scripts/run_<sleeve>_shadow.py wrapper and register it in SPECIALIZED_SLEEVE_PROFILES",
            "run sleeve launcher coverage and sleeve mechanics reports",
            "collect samples until source verification, backpressure, replay, and training gates clear",
        ],
    }


def _fmt_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except Exception:
        return "0"


def _render_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    lines = [
        "# Sleeve Mechanics",
        "",
        f"Generated UTC: {payload.get('timestamp_utc')}",
        "",
        "## What This Means",
        "",
        "A sleeve is a bounded strategy lane. It has a manifest row, optional launcher defaults, a wrapper entrypoint, a collection lifecycle, source and storage gates, and promotion rules before it can influence allocation or execution.",
        "",
        "## Summary",
        "",
        f"- Manifest sleeves: {_fmt_int(summary.get('manifest_sleeve_count'))}",
        f"- Specialized launcher profiles: {_fmt_int(summary.get('specialized_launcher_profile_count'))}",
        f"- All-sleeves registered profiles: {_fmt_int(summary.get('all_sleeves_profile_count'))}",
        f"- Launcher-ready sleeves: {_fmt_int(summary.get('launcher_ready_count'))}",
        f"- Strategies: {_fmt_int(summary.get('strategy_count'))}",
        f"- Expansion candidates not yet launcher-wired: {_fmt_int(summary.get('expansion_candidate_count'))}",
        "",
        "## How Sleeves Work",
        "",
    ]
    for item in payload.get("how_sleeves_work", []):
        if isinstance(item, dict):
            lines.append(f"- {item.get('step')}: {item.get('description')}")
    lines.extend(["", "## Domain Counts", ""])
    domains = summary.get("domain_counts") if isinstance(summary.get("domain_counts"), dict) else {}
    for name, count in sorted(domains.items()):
        lines.append(f"- {name}: {_fmt_int(count)}")
    lines.extend(["", "## Expansion Candidates", ""])
    candidates = payload.get("expansion_candidates") if isinstance(payload.get("expansion_candidates"), list) else []
    for row in candidates[:30]:
        if isinstance(row, dict):
            lines.append(f"- {row.get('name')} ({row.get('pack_name')})")
    if not candidates:
        lines.append("- none")
    lines.extend(["", "## Launchable Sleeve Sample", ""])
    sleeves = payload.get("sleeves") if isinstance(payload.get("sleeves"), list) else []
    for row in [item for item in sleeves if isinstance(item, dict) and item.get("launcher_ready")][:30]:
        lines.append(
            f"- {row.get('name')}: domain={row.get('domain')} "
            f"strategies={_fmt_int(row.get('strategy_count'))} posture={row.get('execution_posture')}"
        )
    lines.append("")
    return "\n".join(lines)


def write_report(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    payload = build_report(project_root)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HEALTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=True, indent=2) + "\n"
    HEALTH_PATH.write_text(text, encoding="utf-8")
    MD_PATH.write_text(_render_markdown(payload), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Explain sleeve mechanics, launch wiring, safety gates, and expansion candidates.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = write_report(PROJECT_ROOT)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        print(
            "sleeve_mechanics "
            f"sleeves={summary.get('manifest_sleeve_count', 0)} "
            f"launchers={summary.get('specialized_launcher_profile_count', 0)} "
            f"ready={summary.get('launcher_ready_count', 0)} "
            f"candidates={summary.get('expansion_candidate_count', 0)}"
        )
        print(f"markdown={MD_PATH}")
        print(f"json={HEALTH_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
