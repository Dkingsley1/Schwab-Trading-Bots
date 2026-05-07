#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "sleeve_strategy_expansion.json"
SESSION_PATH = PROJECT_ROOT / "governance" / "session_configs" / "all_sleeves_latest.json"
REGISTRY_PATH = PROJECT_ROOT / "master_bot_registry.json"
OUT_PATH = PROJECT_ROOT / "governance" / "health" / "sleeve_strategy_coverage_latest.json"

NON_SPECIALIZED_RUNTIME_SLEEVES = {
    "equity_core",
    "intraday_aggressive",
    "day_trading",
    "swing_aggressive",
    "dividend_income",
    "dividend_capture",
    "bond_rates",
    "fx_macro",
    "options_flow",
    "crypto_spot",
    "crypto_futures",
    "schwab_futures",
    "sector_master",
    "position_lifecycle",
    "execution_quality",
    "infrastructure_risk",
    "conservative",
}
ACTIVE_LAUNCHER_STATUSES = {"active_runtime", "active_data_collection"}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _split_symbols(raw: Any) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    if isinstance(raw, list):
        items = raw
    else:
        items = str(raw or "").split(",")
    for item in items:
        sym = str(item or "").strip().upper()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


def _registry_summary(registry: dict[str, Any]) -> dict[str, Any]:
    rows = registry.get("sub_bots") if isinstance(registry.get("sub_bots"), list) else []
    active = [row for row in rows if isinstance(row, dict) and bool(row.get("active", False))]
    collect_only = [
        row
        for row in rows
        if isinstance(row, dict)
        and str(row.get("lifecycle_state") or "") == "data_collection_only"
        and bool(row.get("data_collection_active", False))
    ]
    options = [row for row in rows if isinstance(row, dict) and str(row.get("bot_role") or "") == "options_sub_bot"]
    infra = [row for row in rows if isinstance(row, dict) and str(row.get("bot_role") or "") == "infrastructure_sub_bot"]
    return {
        "total_bots": len(rows),
        "active_bots": len(active),
        "data_collection_only_bots": len(collect_only),
        "options_bots": len(options),
        "infrastructure_bots": len(infra),
    }


def _specialized_launcher_profiles(project_root: Path) -> set[str]:
    root_text = str(project_root)
    try:
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        from scripts import run_specialized_sleeve_shadow as specialized

        defaults = getattr(specialized, "SLEEVE_DEFAULTS", {})
        if isinstance(defaults, dict):
            return {str(name) for name in defaults}
    except Exception:
        return set()
    return set()


def _launcher_gap_reasons(project_root: Path, name: str, specialized_profiles: set[str]) -> list[str]:
    reasons: list[str] = []
    if name not in specialized_profiles:
        reasons.append("missing_specialized_defaults")
    if not (project_root / "scripts" / f"run_{name}_shadow.py").exists():
        reasons.append("missing_wrapper")
    return reasons


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    config = _load_json(project_root / "config" / "sleeve_strategy_expansion.json")
    session = _load_json(project_root / "governance" / "session_configs" / "all_sleeves_latest.json")
    registry = _load_json(project_root / "master_bot_registry.json")
    args = session.get("args") if isinstance(session.get("args"), dict) else {}
    env = session.get("env") if isinstance(session.get("env"), dict) else {}
    ticker_universes = config.get("ticker_universes") if isinstance(config.get("ticker_universes"), dict) else {}
    sleeves = config.get("sleeves") if isinstance(config.get("sleeves"), list) else []

    runtime_universes = {
        "core": _split_symbols(args.get("symbols_core") or env.get("SHADOW_SYMBOLS_CORE") or ticker_universes.get("core")),
        "volatile": _split_symbols(args.get("symbols_volatile") or env.get("SHADOW_SYMBOLS_VOLATILE") or ticker_universes.get("volatile")),
        "defensive": _split_symbols(args.get("symbols_defensive") or env.get("SHADOW_SYMBOLS_DEFENSIVE") or ticker_universes.get("defensive")),
        "dividend": _split_symbols(args.get("dividend_symbols") or env.get("DIVIDEND_SYMBOLS") or ticker_universes.get("dividend")),
        "bond": _split_symbols(args.get("bond_symbols") or env.get("BOND_SYMBOLS") or ticker_universes.get("bond")),
        "fx": _split_symbols(args.get("fx_symbols") or env.get("FX_SYMBOLS") or ticker_universes.get("fx")),
    }
    planned_universe_counts = {key: len(_split_symbols(value)) for key, value in ticker_universes.items()}
    runtime_universe_counts = {key: len(value) for key, value in runtime_universes.items()}
    specialized_launcher_profiles = _specialized_launcher_profiles(project_root)

    sleeve_rows: list[dict[str, Any]] = []
    missing_runtime_sleeves: list[str] = []
    needs_launcher: list[str] = []
    active_runtime_count = 0
    total_strategy_count = 0
    for raw in sleeves:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "").strip()
        status = str(raw.get("runtime_status") or "").strip()
        strategies = [str(item) for item in raw.get("strategies", []) if str(item).strip()] if isinstance(raw.get("strategies"), list) else []
        total_strategy_count += len(strategies)
        launcher_gap_reasons: list[str] = []
        if name and status in ACTIVE_LAUNCHER_STATUSES and name not in NON_SPECIALIZED_RUNTIME_SLEEVES:
            launcher_gap_reasons = _launcher_gap_reasons(project_root, name, specialized_launcher_profiles)
        if status == "active_runtime":
            active_runtime_count += 1
        elif status == "missing_runtime_sleeve":
            missing_runtime_sleeves.append(name)
        if "needs_dedicated_launcher" in status or launcher_gap_reasons:
            if name not in needs_launcher:
                needs_launcher.append(name)
        row = {"name": name, "runtime_status": status, "strategy_count": len(strategies), "strategies": strategies}
        if launcher_gap_reasons:
            row["launcher_status"] = "missing:" + ",".join(launcher_gap_reasons)
        elif name and status in ACTIVE_LAUNCHER_STATUSES and name not in NON_SPECIALIZED_RUNTIME_SLEEVES:
            row["launcher_status"] = "ready"
        sleeve_rows.append(row)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": len(missing_runtime_sleeves) == 0 and len(needs_launcher) == 0,
        "overall_status": (
            "needs_sleeve_expansion"
            if missing_runtime_sleeves
            else "needs_launcher_expansion"
            if needs_launcher
            else "ready"
        ),
        "collection_rule": config.get("collection_rule") if isinstance(config.get("collection_rule"), dict) else {},
        "planned_universe_counts": planned_universe_counts,
        "runtime_universe_counts": runtime_universe_counts,
        "registry": _registry_summary(registry),
        "sleeve_count": len(sleeve_rows),
        "active_runtime_sleeve_count": active_runtime_count,
        "specialized_launcher_profile_count": len(specialized_launcher_profiles),
        "strategy_count": total_strategy_count,
        "missing_runtime_sleeves": missing_runtime_sleeves,
        "strategy_covered_needs_launcher": needs_launcher,
        "sleeves": sleeve_rows,
        "recommended_actions": [
            "keep expanded tickers in collection-only posture while the data plane is backpressured",
            "keep every active data-collection sleeve wired through specialized defaults plus a wrapper launcher",
            "prioritize stat_arb_market_neutral and international_macro if you want genuinely new sleeve diversity",
        ],
    }
    out = project_root / "governance" / "health" / "sleeve_strategy_coverage_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit expanded ticker universes and sleeve/strategy coverage.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = build_payload(PROJECT_ROOT)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "sleeve_strategy_coverage "
            f"status={payload['overall_status']} "
            f"sleeves={payload['sleeve_count']} "
            f"strategies={payload['strategy_count']} "
            f"missing={len(payload['missing_runtime_sleeves'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
