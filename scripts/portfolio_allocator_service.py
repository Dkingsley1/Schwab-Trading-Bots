#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.portfolio_optimizer import PortfolioIntent, allocate_portfolio_intents, allocated_rows_as_dicts


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "allocator" / "portfolio_allocator_service_latest.json"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _intent_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("intents") if isinstance(payload.get("intents"), list) else []
    return [row for row in rows if isinstance(row, dict)]


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    intents_file: Path | None = None,
    allocator_file: Path | None = None,
    risk_file: Path | None = None,
) -> dict[str, Any]:
    intents_path = intents_file or (project_root / "governance" / "allocator" / "portfolio_candidate_intents_latest.json")
    allocator_path = allocator_file or (project_root / "governance" / "allocator" / "sleeve_allocator_latest.json")
    risk_path = risk_file or (project_root / "governance" / "risk" / "portfolio_risk_latest.json")
    intents_payload = _load_json(intents_path)
    allocator = _load_json(allocator_path)
    risk = _load_json(risk_path)

    risk_limits = risk.get("limits") if isinstance(risk.get("limits"), dict) else {}
    symbol_budgets = {
        str(key).upper(): float(value)
        for key, value in ((risk_limits.get("symbol_budgets") or {}) if isinstance(risk_limits.get("symbol_budgets"), dict) else {}).items()
    }
    sector_budgets = {
        str(key).lower(): float(value)
        for key, value in ((risk_limits.get("sector_budgets") or {}) if isinstance(risk_limits.get("sector_budgets"), dict) else {}).items()
    }
    if not sector_budgets:
        sector_budgets = {"technology": 0.30, "financials": 0.20, "unknown": 0.25}

    intents = [
        PortfolioIntent(
            symbol=str(row.get("symbol") or ""),
            sleeve=str(row.get("sleeve") or ""),
            side=str(row.get("side") or "BUY"),
            raw_qty=max(_safe_float(row.get("raw_qty"), 0.0), 0.0),
            score=_safe_float(row.get("score"), 0.0),
            volatility_1m=max(_safe_float(row.get("volatility_1m"), 0.0), 1e-6),
            price=max(_safe_float(row.get("price"), 1.0), 1e-6),
            sector=str(row.get("sector") or "unknown"),
            factor_exposure=_safe_float(row.get("factor_exposure"), 0.0),
            capacity_fraction=max(min(_safe_float(row.get("capacity_fraction"), 1.0), 1.0), 0.0),
        )
        for row in _intent_rows(intents_payload)
    ]
    allocated = allocate_portfolio_intents(
        intents,
        gross_budget=max(min(_safe_float(allocator.get("gross_risk_budget"), 0.75), 1.0), 0.0),
        base_budget=0.25,
        symbol_budgets=symbol_budgets,
        sector_budgets=sector_budgets,
        factor_cap=max(_safe_float(risk_limits.get("max_factor_exposure"), 1.5), 0.0),
    )
    approved_rows = [row for row in allocated_rows_as_dicts(allocated) if _safe_float(row.get("approved_qty"), 0.0) > 0.0]
    rejected_rows = [row for row in allocated_rows_as_dicts(allocated) if _safe_float(row.get("approved_qty"), 0.0) <= 0.0]
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": True,
        "source_files": {
            "intents": str(intents_path),
            "allocator": str(allocator_path),
            "risk": str(risk_path),
        },
        "summary": {
            "input_intent_count": len(intents),
            "approved_intent_count": len(approved_rows),
            "rejected_intent_count": len(rejected_rows),
            "gross_budget": max(min(_safe_float(allocator.get("gross_risk_budget"), 0.75), 1.0), 0.0),
        },
        "approved_intents": approved_rows[:50],
        "rejected_intents": rejected_rows[:50],
        "top_actions": [
            "feed sleeve intents through one allocator so opposite signals net before order emission",
            "encode sector and factor caps in the allocator instead of relying on downstream guards alone",
        ],
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Net cross-sleeve intents into a portfolio-aware allocation plan.")
    parser.add_argument("--intents-file", default=str(PROJECT_ROOT / "governance" / "allocator" / "portfolio_candidate_intents_latest.json"))
    parser.add_argument("--allocator-file", default=str(PROJECT_ROOT / "governance" / "allocator" / "sleeve_allocator_latest.json"))
    parser.add_argument("--risk-file", default=str(PROJECT_ROOT / "governance" / "risk" / "portfolio_risk_latest.json"))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        PROJECT_ROOT,
        intents_file=Path(args.intents_file).expanduser(),
        allocator_file=Path(args.allocator_file).expanduser(),
        risk_file=Path(args.risk_file).expanduser(),
    )
    out_path = Path(args.out_file).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "portfolio_allocator_service "
            f"approved_intents={int(payload.get('summary', {}).get('approved_intent_count', 0) or 0)} "
            f"rejected_intents={int(payload.get('summary', {}).get('rejected_intent_count', 0) or 0)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
