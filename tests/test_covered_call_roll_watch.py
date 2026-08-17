from __future__ import annotations

from argparse import Namespace
from datetime import date

from scripts.ops import covered_call_roll_watch as src


def _args() -> Namespace:
    return Namespace(
        early_dte=60,
        primary_start_dte=45,
        primary_end_dte=21,
        urgent_dte=14,
        itm_early_pct=2.0,
        deep_itm_pct=8.0,
    )


def _snapshot(*, underlying_price: float = 202.9203) -> dict:
    return {
        "fetched": {
            "payload": {
                "securitiesAccount": {
                    "positions": [
                        {
                            "instrument": {"assetType": "EQUITY", "symbol": "NVDA"},
                            "longQuantity": 110.07,
                            "shortQuantity": 0,
                            "marketValue": round(underlying_price * 110.07, 2),
                        },
                        {
                            "instrument": {"assetType": "OPTION", "symbol": "NVDA  260918C00195000"},
                            "longQuantity": 0,
                            "shortQuantity": 1,
                            "averagePrice": 29.0134,
                            "marketValue": -2365.0,
                        },
                    ]
                }
            }
        }
    }


def _multi_account_snapshot() -> dict:
    return {
        "fetched": {
            "payload": {
                "accounts": [
                    {
                        "_broker_account": {"account_label": "account_1_1111", "account_index": 0},
                        "securitiesAccount": {
                            "positions": [
                                {
                                    "instrument": {"assetType": "EQUITY", "symbol": "NVDA"},
                                    "longQuantity": 100,
                                    "shortQuantity": 0,
                                    "marketValue": 20200.0,
                                }
                            ]
                        },
                    },
                    {
                        "_broker_account": {"account_label": "account_2_2222", "account_index": 1},
                        "securitiesAccount": {
                            "positions": [
                                {
                                    "instrument": {"assetType": "OPTION", "symbol": "NVDA  260918C00195000"},
                                    "longQuantity": 0,
                                    "shortQuantity": 1,
                                    "averagePrice": 29.0134,
                                    "marketValue": -2365.0,
                                }
                            ]
                        },
                    },
                ]
            }
        }
    }


def test_roll_watch_gives_concrete_itm_pre_window_range() -> None:
    payload = src.evaluate(_snapshot(), today=date(2026, 6, 11), args=_args())

    assert payload["overall_status"] == "watch"
    assert payload["covered_call_count"] == 1
    row = payload["covered_calls"][0]
    assert row["underlying"] == "NVDA"
    assert row["covered"] is True
    assert row["status"] == "pre_window_itm_watch"
    assert row["roll_window"]["recommended_start"] == "2026-07-20"
    assert row["roll_window"]["recommended_end"] == "2026-08-28"
    assert row["roll_window"]["primary_start"] == "2026-08-04"
    assert row["roll_window"]["primary_end"] == "2026-08-28"
    assert row["roll_window"]["urgent_start"] == "2026-09-04"


def test_roll_watch_requires_coverage_in_same_account() -> None:
    payload = src.evaluate(_multi_account_snapshot(), today=date(2026, 6, 11), args=_args())

    assert payload["account_count"] == 2
    assert payload["overall_status"] == "critical"
    row = payload["covered_calls"][0]
    assert row["account_label"] == "account_2_2222"
    assert row["covered"] is False
    assert row["shares"] == 0.0
    assert row["status"] == "uncovered_short_call"


def test_operator_wait_price_holds_pre_window_roll_review() -> None:
    preferences = {
        "covered_call_roll_preferences": {
            "NVDA": {
                "operator_wait_for_underlying_price": 198.0,
                "operator_roll_bias": "wait_for_pullback_before_early_roll",
            }
        }
    }

    payload = src.evaluate(_snapshot(underlying_price=202.47), today=date(2026, 6, 11), args=_args(), preferences=preferences)

    assert payload["overall_status"] == "watch"
    assert payload["alert_count"] == 0
    row = payload["covered_calls"][0]
    assert row["status"] == "operator_wait_price_watch"
    assert row["severity"] == "warn"
    assert row["operator_roll_preference"]["wait_for_underlying_price"] == 198.0
    assert row["operator_roll_preference"]["trigger_hit"] is False
    assert "operator_wait_price_not_hit underlying=202.47>trigger=198.00" in row["reasons"]
    assert payload["recommended_actions"][0] == "wait for NVDA at or below 198.0 before voluntary early roll review"


def test_roll_watch_attaches_operator_account_alias() -> None:
    account_aliases = {
        "schwab_accounts": {
            "account_1": {
                "operator_account_label": "roth",
                "operator_account_kind": "roth",
                "trading_type": "cash",
            }
        }
    }

    payload = src.evaluate(_snapshot(), today=date(2026, 6, 11), args=_args(), account_aliases=account_aliases)

    row = payload["covered_calls"][0]
    assert row["operator_account_label"] == "roth"
    assert row["operator_account_kind"] == "roth"
    assert row["operator_trading_type"] == "cash"


def test_operator_wait_price_hit_emits_manual_review_alert() -> None:
    preferences = {
        "covered_call_roll_preferences": {
            "NVDA": {
                "operator_wait_for_underlying_price": 198.0,
                "operator_roll_bias": "wait_for_pullback_before_early_roll",
            }
        }
    }

    payload = src.evaluate(_snapshot(underlying_price=198.0), today=date(2026, 6, 11), args=_args(), preferences=preferences)

    assert payload["overall_status"] == "critical"
    assert payload["alert_count"] == 1
    row = payload["covered_calls"][0]
    assert row["status"] == "operator_price_review"
    assert row["severity"] == "critical"
    assert row["operator_roll_preference"]["trigger_hit"] is True
    assert row["operator_roll_preference"]["still_itm_at_trigger"] is True
    assert "operator_wait_price_hit underlying=198.00<=trigger=198.00" in row["reasons"]
    assert payload["recommended_actions"][0] == "NVDA is at or below the operator roll-review trigger 198.0"


def test_roll_watch_emits_critical_inside_recommended_window() -> None:
    payload = src.evaluate(_snapshot(), today=date(2026, 7, 20), args=_args())

    assert payload["overall_status"] == "critical"
    assert payload["alert_count"] == 1
    row = payload["covered_calls"][0]
    assert row["status"] == "roll_window_active"
    assert row["severity"] == "critical"
    assert "inside_recommended_roll_window dte=60" in row["reasons"]
