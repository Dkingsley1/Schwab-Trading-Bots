#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.risk_engine import RiskEngine
from scripts.ops.long_runtime_common import payload_age_minutes


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "risk" / "risk_service_boundary_latest.json"
DEFAULT_MAX_INPUT_AGE_MINUTES = 120.0


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


def _approved_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("approved_intents") if isinstance(payload.get("approved_intents"), list) else []
    return [row for row in rows if isinstance(row, dict)]


def _sha_json_file(path: Path) -> str:
    try:
        blob = path.read_text(encoding="utf-8")
    except Exception:
        return ""
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _input_health(
    name: str,
    path: Path,
    payload: dict[str, Any],
    *,
    now: datetime,
    max_age_minutes: float,
) -> dict[str, Any]:
    age_minutes = payload_age_minutes(payload, path, now=now)
    status = str(payload.get("overall_status") or payload.get("status") or "").strip().lower()
    input_freshness = payload.get("input_freshness") if isinstance(payload.get("input_freshness"), dict) else {}
    blockers: list[str] = []
    if not payload:
        blockers.append("payload_missing_or_invalid")
    if age_minutes is None or age_minutes > max(float(max_age_minutes), 0.0):
        blockers.append("artifact_stale")
    if "ok" in payload and not bool(payload.get("ok", False)):
        blockers.append("upstream_not_ok")
    if status and status not in {"ready", "ok", "active"}:
        blockers.append(f"upstream_status_{status}")
    if "sources_ready" in input_freshness and not bool(input_freshness.get("sources_ready", False)):
        blockers.append("upstream_sources_not_ready")

    if name == "allocator" and not isinstance(payload.get("approved_intents"), list):
        blockers.append("approved_intents_contract_missing")
    elif name == "portfolio_risk" and not isinstance(payload.get("limits"), dict):
        blockers.append("risk_limits_contract_missing")
    elif name == "execution_budget" and not isinstance(payload.get("global"), dict):
        blockers.append("global_budget_contract_missing")
    elif name in {"live_reconciliation", "paper_reconciliation"} and not bool(payload.get("ok", False)):
        blockers.append("reconciliation_not_ready")

    blockers = list(dict.fromkeys(blockers))
    return {
        "path": str(path),
        "exists": path.exists(),
        "age_minutes": round(age_minutes, 3) if age_minutes is not None else None,
        "max_age_minutes": float(max_age_minutes),
        "reported_status": status,
        "reported_ok": payload.get("ok"),
        "ready": not blockers,
        "blockers": blockers,
    }


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    allocator_file: Path | None = None,
    portfolio_risk_file: Path | None = None,
    execution_budget_file: Path | None = None,
    max_input_age_minutes: float = DEFAULT_MAX_INPUT_AGE_MINUTES,
) -> dict[str, Any]:
    allocator_path = allocator_file or (project_root / "governance" / "allocator" / "portfolio_allocator_service_latest.json")
    portfolio_risk_path = portfolio_risk_file or (project_root / "governance" / "risk" / "portfolio_risk_latest.json")
    execution_budget_path = execution_budget_file or (project_root / "governance" / "risk" / "execution_budget_latest.json")
    allocator = _load_json(allocator_path)
    portfolio_risk = _load_json(portfolio_risk_path)
    execution_budget = _load_json(execution_budget_path)
    engine = RiskEngine.from_env()
    live_reconciliation_path = project_root / "governance" / "health" / "live_reconciliation_slo_latest.json"
    paper_reconciliation_path = project_root / "governance" / "health" / "paper_reconciliation_slo_latest.json"
    kill_switch_path = project_root / "scripts" / "global_risk_killswitch.py"
    now = datetime.now(timezone.utc)
    live_reconciliation = _load_json(live_reconciliation_path)
    paper_reconciliation = _load_json(paper_reconciliation_path)
    input_health = {
        "allocator": _input_health(
            "allocator", allocator_path, allocator, now=now, max_age_minutes=max_input_age_minutes
        ),
        "portfolio_risk": _input_health(
            "portfolio_risk", portfolio_risk_path, portfolio_risk, now=now, max_age_minutes=max_input_age_minutes
        ),
        "execution_budget": _input_health(
            "execution_budget", execution_budget_path, execution_budget, now=now, max_age_minutes=max_input_age_minutes
        ),
        "live_reconciliation": _input_health(
            "live_reconciliation", live_reconciliation_path, live_reconciliation, now=now, max_age_minutes=max_input_age_minutes
        ),
        "paper_reconciliation": _input_health(
            "paper_reconciliation", paper_reconciliation_path, paper_reconciliation, now=now, max_age_minutes=max_input_age_minutes
        ),
    }
    upstream_blockers = [
        f"{name}:{blocker}"
        for name, row in input_health.items()
        for blocker in row.get("blockers", [])
    ]
    upstream_ready = all(bool(row.get("ready", False)) for row in input_health.values())

    exposure_state: dict[str, int] = {}
    pre_trade: list[dict[str, Any]] = []
    for row in _approved_rows(allocator)[:50]:
        symbol = str(row.get("symbol") or "").upper()
        features = {
            "volatility_1m": _safe_float(row.get("volatility_1m"), 0.0),
            "factor_exposure": _safe_float(row.get("factor_exposure"), 0.0),
            "var_proxy": _safe_float(row.get("volatility_1m"), 0.0) * 1.65,
            "daily_loss_proxy": _safe_float(((portfolio_risk.get("metrics") or {}).get("combined_blocked_rate")), 0.0),
            "drawdown_proxy": _safe_float(((portfolio_risk.get("metrics") or {}).get("buy_rate_drift_abs")), 0.0),
        }
        result = engine.enforce(
            action=str(row.get("side") or "HOLD"),
            symbol=symbol,
            exposure_state=exposure_state,
            features=features,
        )
        if result.action in {"BUY", "SELL", "SELL_SHORT", "BUY_TO_OPEN", "SELL_TO_OPEN"}:
            exposure_state[symbol] = exposure_state.get(symbol, 0) + 1
        pre_trade.append(
            {
                "symbol": symbol,
                "requested_action": str(row.get("side") or ""),
                "approved_action": result.action,
                "risk_limit_ok": bool(result.risk_limit_ok),
                "reasons": result.reasons,
                "gates": result.gates,
            }
        )

    policy_hashes = {
        "allocator": _sha_json_file(allocator_path),
        "portfolio_risk": _sha_json_file(portfolio_risk_path),
        "execution_budget": _sha_json_file(execution_budget_path),
    }
    service_contracts = {
        "pre_trade_service": {
            "contract_version": 2,
            "endpoint_slug": "risk.pre_trade.approval",
            "input_surface": "portfolio_allocator_service approved intents",
            "evaluated_orders": len(pre_trade),
            "rejections": sum(1 for row in pre_trade if not bool(row.get("risk_limit_ok", False))),
            "policy_hashes": [policy_hashes["allocator"], policy_hashes["portfolio_risk"], policy_hashes["execution_budget"]],
        },
        "execution_budget_service": {
            "contract_version": 2,
            "endpoint_slug": "risk.execution_budget.enforcement",
            "budget_path": str(execution_budget_path),
            "policy_hashes": [policy_hashes["execution_budget"]],
        },
        "post_trade_reconciliation_service": {
            "contract_version": 2,
            "endpoint_slug": "risk.post_trade.reconciliation",
            "live_reconciliation_path": str(live_reconciliation_path),
            "paper_reconciliation_path": str(paper_reconciliation_path),
        },
        "kill_switch_service": {
            "contract_version": 2,
            "endpoint_slug": "risk.kill_switch",
            "path": str(kill_switch_path),
        },
        "exception_workflow": {
            "contract_version": 2,
            "endpoint_slug": "risk.exception.review",
            "requires_operator_review": True,
            "source_files": [str(allocator_path), str(portfolio_risk_path), str(execution_budget_path)],
        },
    }
    independent_boundary = {
        "contract_version": 2,
        "service_count": len(service_contracts),
        "policy_hash_count": len([value for value in policy_hashes.values() if str(value or "").strip()]),
        "deploy_surface_count": 4,
        "service_isolation_ready": bool(
            len([value for value in policy_hashes.values() if str(value or "").strip()]) >= 3
            and live_reconciliation_path.exists()
            and paper_reconciliation_path.exists()
        ),
        "operational_inputs_ready": upstream_ready,
    }
    overall_status = (
        "ready"
        if bool(independent_boundary.get("service_isolation_ready", False))
        and upstream_ready
        and int(independent_boundary.get("service_count", 0) or 0) >= 5
        and int(independent_boundary.get("policy_hash_count", 0) or 0) >= 3
        else "degraded"
    )
    payload = {
        "timestamp_utc": now.isoformat(),
        "schema_version": 2,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "services": service_contracts,
        "service_contracts": service_contracts,
        "policy_hashes": policy_hashes,
        "independent_service_boundary": independent_boundary,
        "input_health": {
            "sources_ready": upstream_ready,
            "sources": input_health,
            "blockers": upstream_blockers,
            "fresh_wrapper_timestamp_does_not_override_stale_inputs": True,
        },
        "pre_trade_decisions": pre_trade,
        "execution_budget": execution_budget,
        "portfolio_risk": portfolio_risk,
        "top_actions": [
            "treat risk approvals as a separate service contract instead of strategy-local branching",
            "feed allocator output through pre-trade approval before any live order bridge sees it",
            "keep post-trade reconciliation independent so execution exceptions cannot be masked by the strategy path",
        ],
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish an independent risk-service boundary and pre-trade approval surface.")
    parser.add_argument("--allocator-file", default=str(PROJECT_ROOT / "governance" / "allocator" / "portfolio_allocator_service_latest.json"))
    parser.add_argument("--portfolio-risk-file", default=str(PROJECT_ROOT / "governance" / "risk" / "portfolio_risk_latest.json"))
    parser.add_argument("--execution-budget-file", default=str(PROJECT_ROOT / "governance" / "risk" / "execution_budget_latest.json"))
    parser.add_argument("--max-input-age-minutes", type=float, default=DEFAULT_MAX_INPUT_AGE_MINUTES)
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        PROJECT_ROOT,
        allocator_file=Path(args.allocator_file).expanduser(),
        portfolio_risk_file=Path(args.portfolio_risk_file).expanduser(),
        execution_budget_file=Path(args.execution_budget_file).expanduser(),
        max_input_age_minutes=max(float(args.max_input_age_minutes), 0.0),
    )
    out_path = Path(args.out_file).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "risk_service_boundary "
            f"pre_trade_orders={len(payload.get('pre_trade_decisions') or [])} "
            f"rejections={int(payload.get('services', {}).get('pre_trade_service', {}).get('rejections', 0) or 0)}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
