import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.portfolio_allocator_service as allocator_src
import scripts.risk_service_boundary as risk_src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_ready_risk_inputs(project_root: Path) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    _write_json(
        project_root / "governance" / "allocator" / "portfolio_allocator_service_latest.json",
        {
            "timestamp_utc": timestamp,
            "ok": True,
            "overall_status": "ready",
            "input_freshness": {"sources_ready": True},
            "approved_intents": [
                {
                    "symbol": "AAPL",
                    "side": "BUY",
                    "approved_qty": 5,
                    "factor_exposure": 0.2,
                    "volatility_1m": 0.01,
                }
            ],
        },
    )
    _write_json(
        project_root / "governance" / "risk" / "portfolio_risk_latest.json",
        {
            "timestamp_utc": timestamp,
            "ok": True,
            "overall_status": "ready",
            "input_freshness": {"sources_ready": True},
            "metrics": {"combined_blocked_rate": 0.01, "buy_rate_drift_abs": 0.02},
            "limits": {"gross_exposure_cap": 0.75},
        },
    )
    _write_json(
        project_root / "governance" / "risk" / "execution_budget_latest.json",
        {
            "timestamp_utc": timestamp,
            "global": {"max_total_actions_per_hour": 40},
            "sleeves": {"core": {"max_actions_per_hour": 40}},
        },
    )
    _write_json(
        project_root / "governance" / "health" / "live_reconciliation_slo_latest.json",
        {"timestamp_utc": timestamp, "ok": True},
    )
    _write_json(
        project_root / "governance" / "health" / "paper_reconciliation_slo_latest.json",
        {"timestamp_utc": timestamp, "ok": True},
    )


def test_portfolio_allocator_service_nets_cross_sleeve_intents(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "governance" / "allocator" / "portfolio_candidate_intents_latest.json",
        {
            "intents": [
                {"symbol": "AAPL", "sleeve": "core", "side": "BUY", "raw_qty": 10, "score": 0.8, "volatility_1m": 0.01, "price": 175.0, "sector": "technology", "factor_exposure": 0.2, "venue": "nasdaq", "clock_bucket": "open"},
                {"symbol": "AAPL", "sleeve": "aggressive", "side": "SELL", "raw_qty": 6, "score": 0.7, "volatility_1m": 0.02, "price": 175.0, "sector": "technology", "factor_exposure": 0.1, "venue": "nasdaq", "clock_bucket": "open"},
            ]
        },
    )
    _write_json(project_root / "governance" / "allocator" / "sleeve_allocator_latest.json", {"gross_risk_budget": 0.75})
    _write_json(project_root / "governance" / "risk" / "portfolio_risk_latest.json", {"limits": {"sector_budgets": {"technology": 0.4}, "max_factor_exposure": 1.5}})
    _write_json(
        project_root / "governance" / "allocator" / "portfolio_capacity_curve_latest.json",
        {
            "summary": {"curve_count": 1, "allocator_ready": True, "regime_count": 1},
            "curves": [
                {
                    "symbol": "AAPL",
                    "venue": "nasdaq",
                    "clock_bucket": "open",
                    "regime": "normal",
                    "forward_cost_bps": 12.0,
                    "recommended_capacity_fraction": 0.55,
                }
            ],
        },
    )

    payload = allocator_src.build_payload(project_root)

    assert payload["summary"]["input_intent_count"] == 2
    assert payload["approved_intents"]
    assert payload["summary"]["capacity_curve_count"] == 1
    assert payload["allocator_contract"]["capacity_curve_ready"] is True
    assert payload["allocator_contract"]["venue_time_capacity_ready"] is True
    assert payload["input_freshness"]["sources_ready"] is True
    assert any(float(row["weight_scale"]) < 1.0 for row in payload["approved_intents"])
    assert all(float(row["forward_cost_bps"]) >= 0.0 for row in payload["approved_intents"])
    assert all(float(row["price"]) == 175.0 for row in payload["approved_intents"])


def test_portfolio_allocator_service_reports_fresh_zero_intent_cycle_as_ready_idle(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(project_root / "governance" / "allocator" / "sleeve_allocator_latest.json", {"gross_risk_budget": 0.75})
    _write_json(project_root / "governance" / "risk" / "portfolio_risk_latest.json", {"limits": {}})
    _write_json(
        project_root / "governance" / "allocator" / "portfolio_capacity_curve_latest.json",
        {"summary": {"curve_count": 0, "allocator_ready": False, "regime_count": 0}, "curves": []},
    )

    payload = allocator_src.build_payload(project_root)

    assert payload["overall_status"] == "ready"
    assert payload["ok"] is True
    assert payload["summary"]["input_intent_count"] == 0
    assert payload["allocator_contract"]["operating_mode"] == "idle_no_intents"
    assert payload["allocator_contract"]["idle_ready"] is True
    assert payload["allocator_contract"]["active_allocation_ready"] is False
    assert payload["allocator_contract"]["capacity_requirements_applicable"] is False
    assert payload["allocator_contract"]["activation_requires_capacity_curves"] is True


def test_portfolio_allocator_service_blocks_active_intents_without_capacity_curves(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_json(
        project_root / "governance" / "allocator" / "portfolio_candidate_intents_latest.json",
        {
            "intents": [
                {
                    "symbol": "AAPL",
                    "sleeve": "core",
                    "side": "BUY",
                    "raw_qty": 1,
                    "score": 0.8,
                    "volatility_1m": 0.01,
                    "price": 175.0,
                    "sector": "technology",
                }
            ]
        },
    )
    _write_json(project_root / "governance" / "allocator" / "sleeve_allocator_latest.json", {"gross_risk_budget": 0.75})
    _write_json(project_root / "governance" / "risk" / "portfolio_risk_latest.json", {"limits": {}})
    _write_json(
        project_root / "governance" / "allocator" / "portfolio_capacity_curve_latest.json",
        {"summary": {"curve_count": 0, "allocator_ready": False, "regime_count": 0}, "curves": []},
    )

    payload = allocator_src.build_payload(project_root)

    assert payload["overall_status"] == "degraded"
    assert payload["ok"] is False
    assert payload["allocator_contract"]["operating_mode"] == "active_allocation"
    assert payload["allocator_contract"]["capacity_requirements_applicable"] is True
    assert payload["allocator_contract"]["active_allocation_ready"] is False


def test_risk_service_boundary_evaluates_allocator_output(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_ready_risk_inputs(project_root)

    payload = risk_src.build_payload(project_root)

    assert payload["ok"] is True
    assert payload["overall_status"] == "ready"
    assert payload["input_health"]["sources_ready"] is True
    assert payload["services"]["pre_trade_service"]["evaluated_orders"] == 1
    assert payload["pre_trade_decisions"][0]["symbol"] == "AAPL"
    assert payload["independent_service_boundary"]["service_isolation_ready"] is True
    assert payload["independent_service_boundary"]["operational_inputs_ready"] is True
    assert payload["independent_service_boundary"]["policy_hash_count"] >= 3


def test_risk_service_boundary_fails_closed_on_degraded_upstream_truth(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_ready_risk_inputs(project_root)
    risk_path = project_root / "governance" / "risk" / "portfolio_risk_latest.json"
    degraded = json.loads(risk_path.read_text(encoding="utf-8"))
    degraded["overall_status"] = "degraded"
    degraded["input_freshness"] = {"sources_ready": False, "stale_sources": ["one_numbers"]}
    _write_json(risk_path, degraded)

    payload = risk_src.build_payload(project_root)

    assert payload["ok"] is False
    assert payload["overall_status"] == "degraded"
    assert payload["independent_service_boundary"]["service_isolation_ready"] is True
    assert payload["independent_service_boundary"]["operational_inputs_ready"] is False
    assert "portfolio_risk:upstream_status_degraded" in payload["input_health"]["blockers"]
    assert "portfolio_risk:upstream_sources_not_ready" in payload["input_health"]["blockers"]
