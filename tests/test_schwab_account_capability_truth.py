from __future__ import annotations

from scripts.ops import schwab_account_capability_truth as truth


def _limited_margin_account(
    *, cash_balance: float = 0.0, accrued_interest: float = 0.0
) -> dict:
    return {
        "accountNumber": "not-for-output",
        "type": "MARGIN",
        "isIntradayMargin": True,
        "isPortfolioMargin": False,
        "isClosingOnlyRestricted": False,
        "currentBalances": {
            "cashBalance": cash_balance,
            "availableFunds": 9369.0,
            "buyingPower": 18738.0,
            "intradayBuyingPowerAmount": 48642.24,
            "marginBalance": -2488.85,
            "accruedInterest": accrued_interest,
            "pendingDeposits": 0.0,
            "maintenanceCall": 0.0,
            "regTCall": 0.0,
            "shortOptionMarketValue": -6282.5,
            "newSchwabBalanceMetric": 7.5,
        },
        "projectedBalances": {"stockBuyingPower": 18738.0},
        "initialBalances": {"cashAvailableForTrading": 0.0},
    }


def _alias() -> dict:
    return {
        "account_policy_key": "schwab_cash_account_1",
        "operator_account_label": "Cash Account 1",
        "operator_account_kind": "cash",
        "tax_wrapper": "taxable",
        "trading_access": "limited_margin",
        "borrowing_allowed": False,
        "cash_only_live_budget": True,
        "existing_positions_authority": "observe_only",
        "canary_candidate": True,
        "canary_cap_usd": 200,
        "allowed_live_routes": ["dividend_liquid_etf_candidate_v1"],
        "operator_verified": True,
        "verification_source": "operator",
    }


def _covered_positions() -> list[dict]:
    return [
        {
            "asset_type": "EQUITY",
            "symbol": "NVDA",
            "underlying": "NVDA",
            "quantity": 100.1,
            "market_value": 22500.0,
        },
        {
            "asset_type": "OPTION",
            "symbol": "NVDA CALL",
            "underlying": "NVDA",
            "quantity": -1.0,
            "market_value": -6282.5,
        },
    ]


def test_limited_margin_truth_preserves_provider_data_without_claiming_debt() -> None:
    payload = truth.build_account_capability_truth(
        _limited_margin_account(),
        alias=_alias(),
        positions=_covered_positions(),
    )

    classification = payload["operator_classification"]
    assert classification["account_kind"] == "cash"
    assert classification["tax_wrapper"] == "taxable"
    assert classification["trading_access"] == "limited_margin"
    assert classification["borrowing_allowed"] is False
    assert payload["provider_account"]["provider_account_type"] == "MARGIN"
    assert "accountNumber" not in payload["provider_account"]["fields"]
    assert payload["provider_balances"]["current"]["newSchwabBalanceMetric"] == 7.5
    assert (
        payload["debit_truth"]["status"]
        == "negative_provider_balance_not_confirmed_as_borrowing"
    )
    assert payload["debit_truth"]["interest_bearing_borrowing_confirmed"] is False
    assert payload["position_collateral_truth"]["covered_short_option_count"] == 1
    assert (
        payload["position_collateral_truth"]["short_option_close_mark_estimate"]
        == 6282.5
    )
    assert payload["canary_preflight"]["account_preflight_ready"] is False
    assert (
        "canary_cash_not_funded_and_settled" in payload["canary_preflight"]["blockers"]
    )
    assert payload["canary_preflight"]["live_execution_authority"] is False


def test_limited_margin_canary_preflight_uses_cash_cap_but_never_grants_execution() -> (
    None
):
    payload = truth.build_account_capability_truth(
        _limited_margin_account(cash_balance=200.0),
        alias=_alias(),
        positions=_covered_positions(),
    )

    preflight = payload["canary_preflight"]
    assert preflight["cash_proxy_ready"] is True
    assert preflight["account_preflight_ready"] is True
    assert preflight["blockers"] == []
    assert preflight["live_execution_authority"] is False
    assert payload["balance_truth"]["settled_cash_claimed"] is False


def test_full_margin_with_accrued_interest_is_not_mislabeled_as_limited_margin() -> (
    None
):
    alias = _alias()
    alias.update(
        {
            "trading_access": "full_margin",
            "borrowing_allowed": True,
            "canary_candidate": False,
        }
    )
    payload = truth.build_account_capability_truth(
        _limited_margin_account(accrued_interest=12.25),
        alias=alias,
        positions=[],
    )

    assert payload["operator_classification"]["borrowing_allowed"] is True
    assert (
        payload["debit_truth"]["status"]
        == "interest_accrual_present_requires_broker_review"
    )
    assert payload["debit_truth"]["interest_bearing_borrowing_confirmed"] is True
