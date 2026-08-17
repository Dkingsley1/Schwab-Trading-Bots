#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "derived_state_latest.json"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def build_derived_state_snapshot(
    project_root: Path,
    *,
    allocator_path: Path,
    risk_path: Path,
    execution_budget_path: Path,
) -> dict[str, Any]:
    allocator = _read_json(allocator_path)
    risk = _read_json(risk_path)
    execution_budget = _read_json(execution_budget_path)

    weights = allocator.get("target_weights") if isinstance(allocator.get("target_weights"), dict) else {}
    sleeve_caps = ((risk.get("limits") or {}).get("sleeve_exposure_caps") or {}) if isinstance(risk, dict) else {}
    budget_sleeves = execution_budget.get("sleeves") if isinstance(execution_budget.get("sleeves"), dict) else {}

    sleeves: dict[str, dict[str, Any]] = {}
    sleeve_names = sorted({*weights.keys(), *sleeve_caps.keys(), *budget_sleeves.keys()})
    for sleeve in sleeve_names:
        budget_row = budget_sleeves.get(sleeve) if isinstance(budget_sleeves.get(sleeve), dict) else {}
        sleeves[str(sleeve)] = {
            "target_weight": round(_safe_float(weights.get(sleeve), 0.0), 6),
            "exposure_cap": round(_safe_float(sleeve_caps.get(sleeve), 0.0), 6),
            "max_actions_per_hour": int(budget_row.get("max_actions_per_hour", 0) or 0),
            "max_open_orders": int(budget_row.get("max_open_orders", 0) or 0),
        }

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ok": bool(allocator and risk and execution_budget),
        "project_root": str(project_root),
        "source_paths": {
            "allocator": str(allocator_path),
            "risk": str(risk_path),
            "execution_budget": str(execution_budget_path),
        },
        "gross_risk_budget": round(_safe_float(allocator.get("gross_risk_budget"), 0.0), 6),
        "risk_level": str(risk.get("risk_level") or ""),
        "risk_score": round(_safe_float(risk.get("risk_score"), 0.0), 6),
        "gross_exposure_cap": round(_safe_float(((risk.get("limits") or {}).get("gross_exposure_cap")), 0.0), 6),
        "max_single_symbol_share": round(
            _safe_float(((risk.get("limits") or {}).get("max_single_symbol_share")), 0.0),
            6,
        ),
        "max_intraday_turnover": round(
            _safe_float(((risk.get("limits") or {}).get("max_intraday_turnover")), 0.0),
            6,
        ),
        "max_total_actions_per_hour": int(((execution_budget.get("global") or {}).get("max_total_actions_per_hour", 0) or 0)),
        "max_total_open_orders": int(((execution_budget.get("global") or {}).get("max_total_open_orders", 0) or 0)),
        "execution_multiplier": round(_safe_float(((execution_budget.get("global") or {}).get("multiplier")), 0.0), 6),
        "allocator_reasons": [str(item) for item in (((allocator.get("policy") or {}).get("reasons")) or []) if str(item)],
        "sleeves": sleeves,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a unified derived-state snapshot from allocator, risk, and execution budgets.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--allocator", default=str(PROJECT_ROOT / "governance" / "allocator" / "sleeve_allocator_latest.json"))
    parser.add_argument("--risk", default=str(PROJECT_ROOT / "governance" / "risk" / "portfolio_risk_latest.json"))
    parser.add_argument("--execution-budget", default=str(PROJECT_ROOT / "governance" / "risk" / "execution_budget_latest.json"))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    payload = build_derived_state_snapshot(
        project_root,
        allocator_path=Path(args.allocator).expanduser(),
        risk_path=Path(args.risk).expanduser(),
        execution_budget_path=Path(args.execution_budget).expanduser(),
    )
    out_path = Path(args.out_file).expanduser()
    _write_json(out_path, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "derived_state "
            f"ok={int(bool(payload.get('ok', False)))} "
            f"risk_level={payload.get('risk_level', '') or 'unknown'} "
            f"actions={int(payload.get('max_total_actions_per_hour', 0) or 0)}"
        )
    return 0 if payload.get("ok", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
