import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import collect_dividend_drip_state as drip


def test_classify_cash_dividend_transaction() -> None:
    event = drip._classify_dividend_transaction(
        {
            "transactionDate": "2026-03-20T14:30:00+00:00",
            "type": "DIVIDEND_OR_INTEREST",
            "description": "Cash Dividend",
            "netAmount": 14.25,
            "transferItems": [
                {
                    "instrument": {"symbol": "SCHD", "assetType": "EQUITY"},
                    "amount": 0.0,
                }
            ],
        }
    )

    assert event is not None
    assert event["symbol"] == "SCHD"
    assert event["event_type"] == "cash_dividend"
    assert event["cash_amount"] == 14.25
    assert event["share_credit"] == 0.0


def test_classify_drip_reinvest_transaction() -> None:
    event = drip._classify_dividend_transaction(
        {
            "transactionDate": "2026-03-21T14:30:00+00:00",
            "type": "REINVESTMENT",
            "description": "Dividend Reinvestment",
            "netAmount": 0.0,
            "transferItems": [
                {
                    "instrument": {"symbol": "VIG", "assetType": "EQUITY"},
                    "amount": 0.1834,
                    "description": "DRIP share credit",
                }
            ],
        }
    )

    assert event is not None
    assert event["symbol"] == "VIG"
    assert event["event_type"] == "drip_reinvest"
    assert event["share_credit"] == 0.1834
    assert event["confidence"] > 0.9


def test_aggregate_dividend_drip_prefers_symbol_level_reinvest_state() -> None:
    now_utc = datetime(2026, 3, 24, 15, 0, tzinfo=timezone.utc)
    payload, health = drip._aggregate_dividend_drip(
        [
            {
                "transactionDate": "2026-03-23T14:30:00+00:00",
                "type": "REINVESTMENT",
                "description": "Dividend Reinvestment",
                "netAmount": 0.0,
                "transferItems": [
                    {"instrument": {"symbol": "SCHD"}, "amount": 0.145},
                ],
                "account_number": "12345678",
            },
            {
                "transactionDate": "2026-03-10T14:30:00+00:00",
                "type": "DIVIDEND_OR_INTEREST",
                "description": "Cash Dividend",
                "netAmount": 11.4,
                "transferItems": [
                    {"instrument": {"symbol": "SCHD"}, "amount": 0.0},
                ],
                "account_number": "12345678",
            },
        ],
        now_utc=now_utc,
        recent_window_days=30,
    )

    features = payload["derived"]["symbol_features"]["SCHD"]

    assert health["ok"] is True
    assert payload["drip_detected"] is True
    assert features["dividend_drip_active_norm"] > 0.7
    assert features["dividend_drip_recent_reinvest_norm"] > 0.2
    assert features["dividend_drip_cash_only_norm"] < 1.0
    assert features["dividend_drip_share_credit_norm"] > 0.0
    assert features["dividend_drip_event_recency_norm"] > 0.9
    assert features["dividend_drip_confidence_norm"] > 0.7
