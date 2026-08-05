from __future__ import annotations

import json

from scripts.ops.schwab_tax_ledger_refresh import normalize_transaction


def test_trade_uses_signed_security_quantity_and_drops_currency_leg() -> None:
    tx = {
        "transactionId": 12345,
        "transactionDate": "2026-06-01T12:00:00Z",
        "type": "TRADE",
        "netAmount": -100.0,
        "description": "EXAMPLE INC",
        "transferItems": [
            {"amount": -100.0, "instrument": {"symbol": "CURRENCY_USD", "assetType": "CURRENCY"}},
            {
                "amount": 2.0,
                "price": 50.0,
                "positionEffect": "OPENING",
                "instrument": {"symbol": "XYZ", "assetType": "EQUITY"},
            },
        ],
    }
    events = normalize_transaction(tx, account_label="account_1", tax_treatment="taxable")
    assert len(events) == 1
    assert events[0]["tax_event_kind"] == "acquisition"
    assert events[0]["action"] == "BUY"
    assert events[0]["symbol"] == "XYZ"
    encoded = json.dumps(events[0])
    assert "12345" not in encoded


def test_closing_option_is_disposition_but_not_profit_without_basis() -> None:
    tx = {
        "transactionId": 99,
        "transactionDate": "2026-07-01T12:00:00Z",
        "type": "TRADE",
        "netAmount": 1075.0,
        "transferItems": [
            {
                "amount": -1.0,
                "price": 10.75,
                "positionEffect": "CLOSING",
                "instrument": {"symbol": "XYZ   260918C00100000", "assetType": "OPTION"},
            }
        ],
    }
    event = normalize_transaction(tx, account_label="account_1", tax_treatment="taxable")[0]
    assert event["tax_event_kind"] == "capital_disposition"
    assert event["proceeds_usd"] == 1075.0
    assert "realized_gain_loss_usd" not in event
    assert "adjusted_cost_basis_usd" not in event


def test_interest_income_and_margin_interest_expense_are_distinct() -> None:
    credit = {
        "transactionId": 1,
        "transactionDate": "2026-07-01T12:00:00Z",
        "type": "DIVIDEND_OR_INTEREST",
        "description": "BANK INT 0601-0630",
        "netAmount": 2.5,
        "transferItems": [{"amount": 2.5, "instrument": {"assetType": "CURRENCY"}}],
    }
    charge = {
        "transactionId": 2,
        "transactionDate": "2026-07-01T12:00:00Z",
        "type": "DIVIDEND_OR_INTEREST",
        "description": "INTEREST 0601 THRU 0630",
        "netAmount": -25.0,
        "transferItems": [{"amount": -25.0, "instrument": {"assetType": "CURRENCY"}}],
    }
    credit_event = normalize_transaction(credit, account_label="account_1", tax_treatment="taxable")[0]
    charge_event = normalize_transaction(charge, account_label="account_1", tax_treatment="taxable")[0]
    assert credit_event["tax_event_kind"] == "interest"
    assert credit_event["amount_usd"] == 2.5
    assert charge_event["tax_event_kind"] == "investment_interest_expense"
    assert charge_event["amount_usd"] == 25.0
