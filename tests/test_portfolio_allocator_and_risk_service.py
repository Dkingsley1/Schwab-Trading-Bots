import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.portfolio_allocator_service as allocator_src
import scripts.risk_service_boundary as risk_src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_portfolio_allocator_service_nets_cross_sleeve_intents(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "governance" / "allocator" / "portfolio_candidate_intents_latest.json",
        {
            "intents": [
                {"symbol": "AAPL", "sleeve": "core", "side": "BUY", "raw_qty": 10, "score": 0.8, "volatility_1m": 0.01, "sector": "technology", "factor_exposure": 0.2},
                {"symbol": "AAPL", "sleeve": "aggressive", "side": "SELL", "raw_qty": 6, "score": 0.7, "volatility_1m": 0.02, "sector": "technology", "factor_exposure": 0.1},
            ]
        },
    )
    _write_json(project_root / "governance" / "allocator" / "sleeve_allocator_latest.json", {"gross_risk_budget": 0.75})
    _write_json(project_root / "governance" / "risk" / "portfolio_risk_latest.json", {"limits": {"sector_budgets": {"technology": 0.4}, "max_factor_exposure": 1.5}})

    payload = allocator_src.build_payload(project_root)

    assert payload["summary"]["input_intent_count"] == 2
    assert payload["approved_intents"]
    assert any(float(row["weight_scale"]) < 1.0 for row in payload["approved_intents"])


def test_risk_service_boundary_evaluates_allocator_output(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "governance" / "allocator" / "portfolio_allocator_service_latest.json",
        {
            "approved_intents": [
                {
                    "symbol": "AAPL",
                    "side": "BUY",
                    "approved_qty": 5,
                    "factor_exposure": 0.2,
                    "volatility_1m": 0.01,
                }
            ]
        },
    )
    _write_json(project_root / "governance" / "risk" / "portfolio_risk_latest.json", {"metrics": {"combined_blocked_rate": 0.01, "buy_rate_drift_abs": 0.02}})
    _write_json(project_root / "governance" / "risk" / "execution_budget_latest.json", {"global": {"max_total_actions_per_hour": 40}})

    payload = risk_src.build_payload(project_root)

    assert payload["services"]["pre_trade_service"]["evaluated_orders"] == 1
    assert payload["pre_trade_decisions"][0]["symbol"] == "AAPL"
