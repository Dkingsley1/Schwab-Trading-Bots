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
SOURCE_MAX_AGE_SECONDS = {
    "intents": 1800.0,
    "allocator": 3600.0,
    "risk": 3600.0,
    "capacity_curves": 21600.0,
}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _source_freshness(
    path: Path,
    *,
    now: datetime,
    max_age_seconds: float,
    absence_is_valid: bool = False,
) -> dict[str, Any]:
    try:
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        age_seconds = max((now - modified).total_seconds(), 0.0)
        exists = True
    except Exception:
        modified = None
        age_seconds = None
        exists = False
    fresh = bool(
        (absence_is_valid and not exists)
        or (exists and age_seconds is not None and age_seconds <= max(float(max_age_seconds), 0.0))
    )
    return {
        "path": str(path),
        "exists": exists,
        "absence_is_valid": bool(absence_is_valid),
        "modified_utc": modified.isoformat() if modified is not None else None,
        "age_seconds": round(age_seconds, 3) if age_seconds is not None else None,
        "max_age_seconds": float(max_age_seconds),
        "fresh": fresh,
    }


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _first_float(*values: Any, default: float = 0.0) -> float:
    for value in values:
        if value in {None, ""}:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return float(default)


def _intent_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("intents") if isinstance(payload.get("intents"), list) else []
    return [row for row in rows if isinstance(row, dict)]


def _curve_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("curves") if isinstance(payload.get("curves"), list) else []
    return [row for row in rows if isinstance(row, dict)]


def _curve_key(symbol: Any, venue: Any, clock_bucket: Any, regime: Any) -> tuple[str, str, str, str]:
    return (
        str(symbol or "").strip().upper(),
        str(venue or "primary").strip().lower() or "primary",
        str(clock_bucket or "all_day").strip().lower() or "all_day",
        str(regime or "normal").strip().lower() or "normal",
    )


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    intents_file: Path | None = None,
    allocator_file: Path | None = None,
    risk_file: Path | None = None,
    capacity_curve_file: Path | None = None,
) -> dict[str, Any]:
    intents_path = intents_file or (project_root / "governance" / "allocator" / "portfolio_candidate_intents_latest.json")
    allocator_path = allocator_file or (project_root / "governance" / "allocator" / "sleeve_allocator_latest.json")
    risk_path = risk_file or (project_root / "governance" / "risk" / "portfolio_risk_latest.json")
    capacity_curve_path = capacity_curve_file or (project_root / "governance" / "allocator" / "portfolio_capacity_curve_latest.json")
    now = datetime.now(timezone.utc)
    input_freshness = {
        "intents": _source_freshness(
            intents_path,
            now=now,
            max_age_seconds=SOURCE_MAX_AGE_SECONDS["intents"],
            absence_is_valid=True,
        ),
        "allocator": _source_freshness(
            allocator_path,
            now=now,
            max_age_seconds=SOURCE_MAX_AGE_SECONDS["allocator"],
        ),
        "risk": _source_freshness(
            risk_path,
            now=now,
            max_age_seconds=SOURCE_MAX_AGE_SECONDS["risk"],
        ),
        "capacity_curves": _source_freshness(
            capacity_curve_path,
            now=now,
            max_age_seconds=SOURCE_MAX_AGE_SECONDS["capacity_curves"],
        ),
    }
    input_sources_ready = all(row["fresh"] for row in input_freshness.values())
    intents_payload = _load_json(intents_path)
    allocator = _load_json(allocator_path)
    risk = _load_json(risk_path)
    capacity_curves = _load_json(capacity_curve_path)

    risk_limits = risk.get("limits") if isinstance(risk.get("limits"), dict) else {}
    factor_cap = max(_safe_float(risk_limits.get("max_factor_exposure"), 1.5), 0.0)
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
    curve_index = {
        _curve_key(row.get("symbol"), row.get("venue"), row.get("clock_bucket"), row.get("regime")): row
        for row in _curve_rows(capacity_curves)
    }
    regime_budget_ready = bool(
        capacity_curves
        and int(((capacity_curves.get("summary") or {}).get("regime_count") or 0) > 0)
    )

    intents = [
        PortfolioIntent(
            symbol=str(row.get("symbol") or ""),
            sleeve=str(row.get("sleeve") or ""),
            side=str(row.get("side") or "BUY"),
            raw_qty=max(_safe_float(row.get("raw_qty"), 0.0), 0.0),
            score=(
                _safe_float(row.get("score"), 0.0)
                / max(
                    1.0 + (
                        _first_float(
                            (
                                curve_index.get(
                                    _curve_key(row.get("symbol"), row.get("venue"), row.get("clock_bucket"), row.get("regime"))
                                )
                                or {}
                            ).get("forward_cost_bps"),
                            row.get("forward_cost_bps"),
                        )
                        / 100.0
                    ),
                    1.0,
                )
            ),
            volatility_1m=max(_safe_float(row.get("volatility_1m"), 0.0), 1e-6),
            price=max(_safe_float(row.get("price"), 1.0), 1e-6),
            sector=str(row.get("sector") or "unknown"),
            factor_exposure=_safe_float(row.get("factor_exposure"), 0.0),
            capacity_fraction=max(
                min(
                    _first_float(
                        (
                            curve_index.get(_curve_key(row.get("symbol"), row.get("venue"), row.get("clock_bucket"), row.get("regime")))
                            or {}
                        ).get("recommended_capacity_fraction"),
                        row.get("capacity_fraction"),
                        default=1.0,
                    ),
                    1.0,
                ),
                0.0,
            ),
            venue=str(row.get("venue") or "primary"),
            clock_bucket=str(row.get("clock_bucket") or row.get("session_bucket") or "all_day"),
            regime=str(row.get("regime") or "normal"),
            forward_cost_bps=_first_float(
                (
                    curve_index.get(_curve_key(row.get("symbol"), row.get("venue"), row.get("clock_bucket"), row.get("regime")))
                    or {}
                ).get("forward_cost_bps"),
                row.get("forward_cost_bps"),
            ),
        ) for row in _intent_rows(intents_payload)
    ]
    allocated = allocate_portfolio_intents(
        intents,
        gross_budget=max(min(_safe_float(allocator.get("gross_risk_budget"), 0.75), 1.0), 0.0),
        base_budget=0.25,
        symbol_budgets=symbol_budgets,
        sector_budgets=sector_budgets,
        factor_cap=factor_cap,
    )
    approved_rows = [row for row in allocated_rows_as_dicts(allocated) if _safe_float(row.get("approved_qty"), 0.0) > 0.0]
    rejected_rows = [row for row in allocated_rows_as_dicts(allocated) if _safe_float(row.get("approved_qty"), 0.0) <= 0.0]
    curve_summary = dict(capacity_curves.get("summary") or {})
    idle_no_intents = len(intents) == 0
    allocator_contract = {
        "factor_budget_ready": factor_cap > 0.0,
        "factor_budget_source": ("explicit_risk_limit" if risk_limits.get("max_factor_exposure") is not None else "optimizer_default"),
        "factor_budget_limit": round(factor_cap, 6),
        "sector_budget_ready": bool(sector_budgets),
        "regime_budget_ready": regime_budget_ready,
        "capacity_curve_ready": bool(curve_summary.get("allocator_ready", False)),
        "venue_time_capacity_ready": int(curve_summary.get("curve_count") or 0) > 0,
        "seeded_intent_contract_ready": bool(curve_summary.get("curve_count") or 0) > 0,
    }
    active_allocation_ready = bool(
        allocator_contract["factor_budget_ready"]
        and allocator_contract["sector_budget_ready"]
        and allocator_contract["regime_budget_ready"]
        and allocator_contract["capacity_curve_ready"]
        and allocator_contract["venue_time_capacity_ready"]
        and input_sources_ready
    )
    idle_ready = bool(
        idle_no_intents
        and allocator_contract["factor_budget_ready"]
        and allocator_contract["sector_budget_ready"]
        and input_sources_ready
    )
    allocator_contract.update(
        {
            "operating_mode": "idle_no_intents" if idle_no_intents else "active_allocation",
            "idle_no_intents": idle_no_intents,
            "idle_ready": idle_ready,
            "active_allocation_ready": active_allocation_ready,
            "capacity_requirements_applicable": not idle_no_intents,
            "activation_requires_capacity_curves": bool(idle_no_intents and not active_allocation_ready),
        }
    )
    overall_status = "ready" if idle_ready or active_allocation_ready else "degraded"
    payload = {
        "timestamp_utc": now.isoformat(),
        "schema_version": 2,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "source_files": {
            "intents": str(intents_path),
            "allocator": str(allocator_path),
            "risk": str(risk_path),
            "capacity_curves": str(capacity_curve_path),
        },
        "input_freshness": {
            "sources_ready": input_sources_ready,
            "sources": input_freshness,
            "stale_sources": sorted(name for name, row in input_freshness.items() if not row["fresh"]),
            "fresh_wrapper_timestamp_does_not_override_stale_sources": True,
        },
        "summary": {
            "input_intent_count": len(intents),
            "approved_intent_count": len(approved_rows),
            "rejected_intent_count": len(rejected_rows),
            "gross_budget": max(min(_safe_float(allocator.get("gross_risk_budget"), 0.75), 1.0), 0.0),
            "capacity_curve_count": int(((capacity_curves.get("summary") or {}).get("curve_count") or 0)),
            "forward_capacity_ready": bool(((capacity_curves.get("summary") or {}).get("allocator_ready", False))),
            "seeded_intent_contract_ready": bool(curve_summary.get("curve_count") or 0),
        },
        "allocator_contract": allocator_contract,
        "capacity_curve_summary": curve_summary,
        "approved_intents": approved_rows[:50],
        "rejected_intents": rejected_rows[:50],
        "top_actions": [
            "feed sleeve intents through one allocator so opposite signals net before order emission",
            "encode sector, factor, regime, and venue-time capacity curves in the allocator instead of relying on downstream guards alone",
        ],
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Net cross-sleeve intents into a portfolio-aware allocation plan.")
    parser.add_argument("--intents-file", default=str(PROJECT_ROOT / "governance" / "allocator" / "portfolio_candidate_intents_latest.json"))
    parser.add_argument("--allocator-file", default=str(PROJECT_ROOT / "governance" / "allocator" / "sleeve_allocator_latest.json"))
    parser.add_argument("--risk-file", default=str(PROJECT_ROOT / "governance" / "risk" / "portfolio_risk_latest.json"))
    parser.add_argument("--capacity-curve-file", default=str(PROJECT_ROOT / "governance" / "allocator" / "portfolio_capacity_curve_latest.json"))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        PROJECT_ROOT,
        intents_file=Path(args.intents_file).expanduser(),
        allocator_file=Path(args.allocator_file).expanduser(),
        risk_file=Path(args.risk_file).expanduser(),
        capacity_curve_file=Path(args.capacity_curve_file).expanduser(),
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
