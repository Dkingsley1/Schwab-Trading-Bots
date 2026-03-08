import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.portfolio_optimizer import allocate_quantity
from core.position_sizing import size_from_action
from core.risk_engine import apply_risk_limits


@dataclass
class ReplayRow:
    symbol: str
    features: Dict[str, float]
    action: str
    score: float
    threshold: float


def _default_payload() -> Dict[str, Any]:
    return {
        "equity_proxy": 100000.0,
        "max_notional_pct": 0.03,
        "portfolio_base_budget": 0.50,
        "symbol_budgets": {"AAPL": 0.45, "MSFT": 0.40, "NVDA": 0.35},
        "rows": [
            {
                "symbol": "AAPL",
                "action": "BUY",
                "score": 0.64,
                "threshold": 0.55,
                "features": {
                    "volatility_1m": 0.012,
                    "pct_from_close": 0.002,
                    "mom_5m": 0.001,
                    "vol": 0.012,
                    "daily_loss_proxy": 0.0,
                },
            },
            {
                "symbol": "MSFT",
                "action": "SELL",
                "score": 0.62,
                "threshold": 0.56,
                "features": {
                    "volatility_1m": 0.016,
                    "pct_from_close": -0.003,
                    "mom_5m": -0.001,
                    "vol": 0.016,
                    "daily_loss_proxy": 0.0,
                },
            },
            {
                "symbol": "NVDA",
                "action": "BUY",
                "score": 0.58,
                "threshold": 0.55,
                "features": {
                    "volatility_1m": 0.028,
                    "pct_from_close": 0.011,
                    "mom_5m": 0.005,
                    "vol": 0.028,
                    "daily_loss_proxy": 0.01,
                },
            },
        ],
    }


def _load_payload(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return _default_payload()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _default_payload()


def _to_replay_rows(payload: Dict[str, Any]) -> List[ReplayRow]:
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    out: List[ReplayRow] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        features = row.get("features") if isinstance(row.get("features"), dict) else {}
        out.append(
            ReplayRow(
                symbol=str(row.get("symbol", "")).upper(),
                features={k: float(v) for k, v in features.items() if isinstance(v, (int, float))},
                action=str(row.get("action", "HOLD")).upper(),
                score=float(row.get("score", 0.5) or 0.5),
                threshold=float(row.get("threshold", 0.55) or 0.55),
            )
        )
    return out


def run_replay(payload: Dict[str, Any]) -> Dict[str, Any]:
    rows = _to_replay_rows(payload)
    exposure_state: Dict[str, int] = {}

    equity_proxy = float(payload.get("equity_proxy", 100000.0) or 100000.0)
    max_notional_pct = float(payload.get("max_notional_pct", 0.03) or 0.03)
    base_budget = float(payload.get("portfolio_base_budget", 0.50) or 0.50)
    symbol_budgets = payload.get("symbol_budgets") if isinstance(payload.get("symbol_budgets"), dict) else {}

    results: List[Dict[str, Any]] = []
    for row in rows:
        risk_action, risk_reasons, risk_gates = apply_risk_limits(
            action=row.action,
            symbol=row.symbol,
            exposure_state=exposure_state,
            features={
                "volatility_1m": float(row.features.get("volatility_1m", row.features.get("vol", 0.0)) or 0.0),
                "drawdown_proxy": abs(float(row.features.get("pct_from_close", 0.0) or 0.0)),
                "var_proxy": abs(float(row.features.get("volatility_1m", row.features.get("vol", 0.0)) or 0.0)) * 1.65,
                "factor_exposure": abs(float(row.features.get("mom_5m", 0.0) or 0.0)) + abs(float(row.features.get("pct_from_close", 0.0) or 0.0)),
                "daily_loss_proxy": abs(float(row.features.get("daily_loss_proxy", 0.0) or 0.0)),
            },
        )

        vol = float(row.features.get("volatility_1m", row.features.get("vol", 0.0)) or 0.0)
        raw_qty = size_from_action(
            action=risk_action,
            score=row.score,
            threshold=row.threshold,
            volatility_1m=vol,
            equity_proxy=equity_proxy,
            max_notional_pct=max_notional_pct,
        )
        alloc_qty = allocate_quantity(
            raw_qty=raw_qty,
            symbol=row.symbol,
            score=row.score,
            volatility_1m=vol,
            base_budget=base_budget,
            symbol_budgets={str(k).upper(): float(v) for k, v in symbol_budgets.items() if isinstance(v, (int, float))},
        )

        if risk_action in {"BUY", "SELL"}:
            exposure_state[row.symbol] = int(exposure_state.get(row.symbol, 0) or 0) + 1

        results.append(
            {
                "symbol": row.symbol,
                "action_in": row.action,
                "action_out": risk_action,
                "score": round(row.score, 6),
                "threshold": round(row.threshold, 6),
                "raw_qty": round(float(raw_qty), 6),
                "alloc_qty": round(float(alloc_qty), 6),
                "risk_gates": risk_gates,
                "risk_reasons": risk_reasons,
            }
        )

    canonical = {
        "results": results,
        "exposure_state": exposure_state,
    }
    blob = json.dumps(canonical, sort_keys=True, ensure_ascii=True)
    replay_hash = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ok": True,
        "row_count": len(results),
        "replay_hash": replay_hash,
        "canonical": canonical,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic end-to-end replay (ingress->risk->sizing->allocation).")
    parser.add_argument("--in-file", default="", help="Input replay payload JSON file.")
    parser.add_argument("--expected-hash", default="", help="If set, fail when replay hash differs.")
    parser.add_argument("--out-file", default=str(PROJECT_ROOT / "governance" / "health" / "replay_end_to_end_latest.json"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    in_path = Path(args.in_file).expanduser().resolve() if args.in_file else None
    payload = _load_payload(in_path)
    out = run_replay(payload)

    expected = str(args.expected_hash or "").strip().lower()
    if expected:
        out["expected_hash"] = expected
        out["hash_match"] = bool(out["replay_hash"] == expected)
        out["ok"] = bool(out.get("ok", False)) and bool(out["hash_match"])

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(out, ensure_ascii=True))
    else:
        print(
            "replay_end_to_end "
            f"ok={int(bool(out.get('ok', False)))} row_count={int(out.get('row_count', 0))} "
            f"hash={out.get('replay_hash', '')}"
        )

    return 0 if bool(out.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
