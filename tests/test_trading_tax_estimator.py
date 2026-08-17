from __future__ import annotations

import json
from pathlib import Path

from scripts.ops.trading_tax_estimator import evaluate, validate_policy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY = json.loads((PROJECT_ROOT / "config" / "trading_tax_policy_us_federal_2026.json").read_text())


def _profile() -> dict:
    return {
        "tax_year": 2026,
        "filing_status": "single",
        "tax_residency": {"country": "US", "state": "FL", "locality": ""},
        "taxable_ordinary_income_before_trading_usd": 100000,
        "preferential_income_before_trading_usd": 0,
        "modified_adjusted_gross_income_before_trading_usd": 100000,
        "net_investment_income_before_trading_usd": 0,
        "short_term_capital_loss_carryover_usd": 0,
        "long_term_capital_loss_carryover_usd": 0,
        "state_tax_model": {"status": "configured", "method": "none", "effective_rate": None},
        "account_tax_treatment_by_label": {"taxable": "taxable", "roth": "tax_advantaged"},
    }


def _coverage() -> dict:
    return {
        "complete_for_tax_year": True,
        "all_relevant_accounts_included": True,
        "coverage_start": "2026-01-01T00:00:00+00:00",
        "coverage_end": "2026-08-04T00:00:00+00:00",
        "source": "test",
    }


def test_policy_is_current_and_structurally_valid() -> None:
    result = validate_policy(POLICY, requested_tax_year=2026)
    assert result["ok"] is True
    assert result["issues"] == []

    wrong_year = validate_policy(POLICY, requested_tax_year=2027)
    assert wrong_year["ok"] is False
    assert "tax_year_mismatch" in wrong_year["issues"]


def test_exact_estimate_separates_tax_characters_and_excludes_paper_and_roth() -> None:
    events = [
        {
            "event_id": "st",
            "environment": "actual",
            "account_label": "taxable",
            "tax_event_kind": "capital_disposition",
            "realized_gain_loss_usd": 1000,
            "tax_character": "short_term",
            "wash_sale_status": "verified_clear",
        },
        {
            "event_id": "lt",
            "environment": "actual",
            "account_label": "taxable",
            "tax_event_kind": "capital_disposition",
            "realized_gain_loss_usd": 2000,
            "tax_character": "long_term",
            "wash_sale_status": "verified_clear",
        },
        {
            "event_id": "qd",
            "environment": "actual",
            "account_label": "taxable",
            "tax_event_kind": "dividend",
            "amount_usd": 500,
            "qualified_dividend": True,
        },
        {
            "event_id": "interest",
            "environment": "actual",
            "account_label": "taxable",
            "tax_event_kind": "interest",
            "amount_usd": 100,
        },
        {
            "event_id": "futures",
            "environment": "actual",
            "account_label": "taxable",
            "tax_event_kind": "section_1256",
            "amount_usd": 1000,
            "realized_gain_loss_usd": 1000,
            "section_1256_verified": True,
        },
        {
            "event_id": "paper",
            "environment": "paper",
            "account_label": "taxable",
            "tax_event_kind": "capital_disposition",
            "realized_gain_loss_usd": 5000,
            "tax_character": "short_term",
        },
        {
            "event_id": "roth",
            "environment": "actual",
            "account_label": "roth",
            "tax_event_kind": "capital_disposition",
            "realized_gain_loss_usd": 10000,
            "tax_character": "short_term",
        },
    ]
    result = evaluate(events, profile=_profile(), policy=POLICY, ledger_metadata=_coverage())

    assert result["status"] == "ready"
    assert result["estimate_status"] == "exact_estimate_available"
    assert result["tax_owed_from_trading_estimate_usd"] is not None
    buckets = result["taxable_activity"]["upper_scenario_buckets_usd"]
    assert buckets["ordinary_investment_income"] == 100
    assert buckets["short_term_capital"] == 1400
    assert buckets["long_term_capital"] == 2600
    assert buckets["qualified_dividends"] == 500
    counts = result["taxable_activity"]["counts"]
    assert counts["paper_events"] == 1
    assert counts["tax_advantaged_events"] == 1
    assert result["non_taxable_now"]["paper_realized_pnl_usd"] == 5000


def test_unknown_holding_period_and_wash_sale_create_a_range() -> None:
    events = [
        {
            "event_id": "unknown_gain",
            "environment": "actual",
            "account_label": "taxable",
            "tax_event_kind": "capital_disposition",
            "symbol": "XYZ",
            "realized_gain_loss_usd": 2500,
            "wash_sale_status": "verified_clear",
        },
        {
            "event_id": "loss",
            "environment": "actual",
            "account_label": "taxable",
            "tax_event_kind": "capital_disposition",
            "symbol": "ABC",
            "realized_gain_loss_usd": -1000,
            "tax_character": "short_term",
            "transaction_date": "2026-06-01",
        },
        {
            "event_id": "rebuy",
            "environment": "actual",
            "account_label": "taxable",
            "tax_event_kind": "acquisition",
            "symbol": "ABC",
            "action": "BUY",
            "transaction_date": "2026-06-15",
            "realization_status": "not_realized",
        },
    ]
    result = evaluate(events, profile=_profile(), policy=POLICY, ledger_metadata=_coverage())

    assert result["estimate_status"] == "range_or_reserve_only"
    assert result["federal"]["estimate_upper_usd"] > result["federal"]["estimate_lower_usd"]
    reasons = {row["reason"] for row in result["unresolved_evidence"]}
    assert "holding_period_missing" in reasons
    assert "potential_wash_sale_or_cross_account_repurchase" in reasons


def test_provisional_broker_dividend_is_included_but_blocks_exact_claim() -> None:
    events = [
        {
            "event_id": "dividend",
            "environment": "actual",
            "account_label": "taxable",
            "tax_event_kind": "dividend",
            "amount_usd": 125,
            "qualified_dividend": False,
            "tax_amount_provisional": True,
        }
    ]
    result = evaluate(events, profile=_profile(), policy=POLICY, ledger_metadata=_coverage())

    assert result["taxable_activity"]["upper_scenario_buckets_usd"]["ordinary_investment_income"] == 125
    assert result["tax_owed_from_trading_estimate_usd"] is None
    assert any("provisional_amount" in row["reason"] for row in result["unresolved_evidence"])


def test_tax_amount_is_unavailable_when_account_treatment_is_unknown() -> None:
    profile = _profile()
    profile["account_tax_treatment_by_label"] = {}
    events = [
        {
            "event_id": "sale",
            "environment": "actual",
            "account_label": "account_1",
            "tax_event_kind": "capital_disposition",
            "realized_gain_loss_usd": 1000,
            "tax_character": "short_term",
        }
    ]
    result = evaluate(events, profile=profile, policy=POLICY, ledger_metadata=_coverage())
    assert result["tax_owed_from_trading_estimate_usd"] is None
    assert result["federal"]["maximum_rate_reserve_usd"] is None
    assert result["recommended_tax_reserve"]["amount_usd"] is None
    assert result["estimated_payment_watch"]["status"] == "insufficient_evidence"
    assert result["taxable_activity"]["counts"]["actual_taxable_events"] == 0
    assert result["unresolved_evidence"][0]["reason"] == "account_tax_treatment_unknown"
