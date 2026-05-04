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
OUT_DIR = PROJECT_ROOT / "exports" / "reports" / "strategy_inventory"
HEALTH_PATH = PROJECT_ROOT / "governance" / "health" / "strategy_inventory_latest.json"
MD_PATH = OUT_DIR / "strategy_inventory_latest.md"
PDF_PATH = OUT_DIR / "strategy_inventory_latest.pdf"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _as_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, dict)]


def _registry_rows() -> list[dict[str, Any]]:
    payload = _load_json(REGISTRY_PATH)
    return _as_rows(payload.get("sub_bots"))


def _planned_rows() -> list[dict[str, Any]]:
    try:
        from scripts.ops.roster_expansion_slots import DEFAULT_SLOT_SPECS, _slot_registry_row

        return [_slot_registry_row(row) for row in DEFAULT_SLOT_SPECS if isinstance(row, dict)]
    except Exception:
        return []


def _unique_bot_rows() -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in _registry_rows() + _planned_rows():
        bot_id = str(row.get("bot_id") or "").strip()
        if bot_id:
            by_id[bot_id] = row
    return list(by_id.values())


def _strategy_slug(value: Any) -> str:
    return str(value or "").strip()


def build_inventory() -> dict[str, Any]:
    config = _load_json(CONFIG_PATH)
    sleeves = _as_rows(config.get("sleeves"))
    ticker_universes = config.get("ticker_universes") if isinstance(config.get("ticker_universes"), dict) else {}
    bot_rows = _unique_bot_rows()

    bots_by_sleeve: dict[str, list[dict[str, Any]]] = defaultdict(list)
    active_by_sleeve: Counter[str] = Counter()
    collection_by_sleeve: Counter[str] = Counter()
    training_excluded_by_sleeve: Counter[str] = Counter()
    roles_by_sleeve: dict[str, Counter[str]] = defaultdict(Counter)
    for row in bot_rows:
        sleeve = str(row.get("sleeve_profile") or row.get("sleeve_family") or "unassigned").strip() or "unassigned"
        bots_by_sleeve[sleeve].append(row)
        if bool(row.get("active", False)):
            active_by_sleeve[sleeve] += 1
        if str(row.get("lifecycle_state") or "").strip().lower() == "data_collection_only":
            collection_by_sleeve[sleeve] += 1
        if bool(row.get("training_excluded", False)) or bool(row.get("exclude_from_training", False)):
            training_excluded_by_sleeve[sleeve] += 1
        roles_by_sleeve[sleeve][str(row.get("bot_role") or "unknown")] += 1

    sleeve_rows: list[dict[str, Any]] = []
    total_strategies = 0
    for sleeve in sleeves:
        name = str(sleeve.get("name") or "").strip()
        strategies = [_strategy_slug(item) for item in list(sleeve.get("strategies") or []) if _strategy_slug(item)]
        total_strategies += len(strategies)
        tickers = [str(item) for item in list(ticker_universes.get(name) or [])]
        row = {
            "name": name,
            "runtime_status": str(sleeve.get("runtime_status") or ""),
            "strategy_count": len(strategies),
            "strategies": strategies,
            "ticker_count": len(tickers),
            "ticker_universe": tickers,
            "bot_count": len(bots_by_sleeve.get(name, [])),
            "active_bot_count": int(active_by_sleeve.get(name, 0)),
            "collection_only_bot_count": int(collection_by_sleeve.get(name, 0)),
            "training_excluded_bot_count": int(training_excluded_by_sleeve.get(name, 0)),
            "bot_roles": dict(sorted(roles_by_sleeve.get(name, Counter()).items())),
        }
        sleeve_rows.append(row)

    uncovered_sleeves = [row["name"] for row in sleeve_rows if int(row["bot_count"]) <= 0]
    advanced_collection_sleeves = [
        row["name"]
        for row in sleeve_rows
        if row["runtime_status"] == "active_data_collection"
        and any(
            token in row["name"]
            for token in (
                "options",
                "swap",
                "cdo",
                "greek",
                "basis",
                "arbitrage",
                "hedging",
                "dispersion",
                "microstructure",
                "parity",
                "quant",
                "state_space",
                "tail_dependency",
                "adaptive",
                "adversarial",
                "latency",
                "alternative_data",
                "zkp",
                "gpu",
                "qemc",
                "transport",
                "topology",
                "neural_sde",
                "order_flow",
                "toxicity",
                "signature",
                "hawkes",
                "mean_field",
                "physics",
                "games",
                "lit",
                "transformer",
                "critic",
                "hmm",
                "pinsde",
                "causal",
                "omni",
                "symbolic",
                "rlbf",
                "dms",
                "equivariant",
                "dainn",
                "markovian",
                "durability",
                "information_geometry",
                "statistical_manifold",
                "graph_attention",
                "spillover",
                "agentic_wallet",
                "intent",
                "rough_path",
                "signature_kernel",
                "quantum_classical",
                "formal_verification",
                "smart_agent",
                "institutional",
                "plumbing",
                "feature_store",
                "flink",
                "lob",
                "dex",
                "lobdif",
                "crisis",
                "flash",
                "homology",
                "photonic",
                "replication",
                "market_gan",
                "correlation_convergence",
                "fed_2026",
                "xva",
                "counterparty",
                "credit_derivatives",
                "cdx",
                "cds",
                "securitized",
                "mbs",
                "abs",
                "clo",
                "repo",
                "securities_lending",
                "tape",
                "provider_adapter",
                "proof_quantum",
                "formal_backends",
                "model_risk",
                "validation",
                "transaction_cost",
                "slippage",
                "portfolio_construction",
                "event_intelligence",
                "feature_quality",
                "data_confidence",
                "liquidity_regime",
                "system_governor",
            )
        )
    ]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_config": str(CONFIG_PATH),
        "source_registry": str(REGISTRY_PATH),
        "sleeve_count": len(sleeve_rows),
        "strategy_count": total_strategies,
        "bot_count": len(bot_rows),
        "uncovered_sleeves": uncovered_sleeves,
        "advanced_collection_sleeves": advanced_collection_sleeves,
        "sleeves": sleeve_rows,
        "artifact_paths": {
            "markdown": str(MD_PATH),
            "pdf": str(PDF_PATH),
            "json": str(HEALTH_PATH),
        },
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Strategy Inventory",
        "",
        f"Generated UTC: {payload.get('timestamp_utc')}",
        "",
        "## Summary",
        "",
        f"- Sleeve count: {payload.get('sleeve_count')}",
        f"- Strategy count: {payload.get('strategy_count')}",
        f"- Bot count in registry/expansion plan: {payload.get('bot_count')}",
        f"- Advanced collection sleeves: {len(list(payload.get('advanced_collection_sleeves') or []))}",
        f"- Sleeves without mapped bots: {len(list(payload.get('uncovered_sleeves') or []))}",
        "",
        "## How To Read This",
        "",
        "- Runtime status comes from config/sleeve_strategy_expansion.json.",
        "- Bot counts combine the current registry with planned roster-expansion slots.",
        "- Collection-only bots are active observers; training remains blocked until each lane clears its data floor.",
        "- Advanced derivative sleeves are proxy-first and execution-blocked unless a future broker/capability gate explicitly changes that.",
        "",
        "## Sleeves And Strategies",
        "",
    ]
    for sleeve in list(payload.get("sleeves") or []):
        if not isinstance(sleeve, dict):
            continue
        roles = ", ".join(f"{key}:{value}" for key, value in dict(sleeve.get("bot_roles") or {}).items()) or "none"
        lines.extend(
            [
                f"### {sleeve.get('name')}",
                "",
                f"- Runtime status: {sleeve.get('runtime_status')}",
                f"- Strategies: {sleeve.get('strategy_count')}",
                f"- Ticker universe size: {sleeve.get('ticker_count')}",
                f"- Bots mapped: {sleeve.get('bot_count')} active={sleeve.get('active_bot_count')} collection_only={sleeve.get('collection_only_bot_count')} training_excluded={sleeve.get('training_excluded_bot_count')}",
                f"- Bot roles: {roles}",
            ]
        )
        for strategy in list(sleeve.get("strategies") or []):
            lines.append(f"- {strategy}")
        lines.append("")
    uncovered = list(payload.get("uncovered_sleeves") or [])
    if uncovered:
        lines.extend(["## Coverage Notes", ""])
        lines.append("- Sleeves without mapped bots: " + ", ".join(str(item) for item in uncovered))
        lines.append("")
    return "\n".join(lines)


def write_report(*, render_pdf: bool = True) -> dict[str, Any]:
    payload = build_inventory()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HEALTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    markdown = _render_markdown(payload)
    MD_PATH.write_text(markdown, encoding="utf-8")
    if render_pdf:
        from scripts.ops.sendout_pdf_refresh import render_text_pdf

        pdf_result = render_text_pdf("Strategy Inventory", MD_PATH, PDF_PATH)
        payload["pdf"] = pdf_result
    else:
        payload["pdf"] = {"ok": PDF_PATH.exists(), "pdf_path": str(PDF_PATH)}
    HEALTH_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a PDF-ready inventory of all configured system strategies.")
    parser.add_argument("--no-render-pdf", action="store_true", help="Write markdown/json only.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = write_report(render_pdf=not args.no_render_pdf)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        pdf = payload.get("pdf") if isinstance(payload.get("pdf"), dict) else {}
        print(
            "strategy_inventory "
            f"sleeves={payload.get('sleeve_count')} "
            f"strategies={payload.get('strategy_count')} "
            f"pdf={pdf.get('pdf_path') or PDF_PATH}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
