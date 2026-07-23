from __future__ import annotations

from scripts.ops import account_position_study as src


def test_position_study_uses_highest_severity_roll_context() -> None:
    snapshot = {
        "fetched": {
            "payload": {
                "securitiesAccount": {
                    "positions": [
                        {
                            "instrument": {"assetType": "EQUITY", "symbol": "NVDA"},
                            "longQuantity": 100,
                            "shortQuantity": 0,
                            "marketValue": 20250.0,
                        }
                    ]
                }
            }
        }
    }
    roll_watch = {
        "timestamp_utc": "2026-06-11T18:48:34Z",
        "covered_calls": [
            {
                "underlying": "NVDA",
                "status": "monitor",
                "severity": "info",
                "dte": 589,
                "strike": 210,
                "expiration": "2028-01-21",
                "underlying_price": 202.5,
                "moneyness_pct": -3.57,
            },
            {
                "underlying": "NVDA",
                "status": "operator_wait_price_watch",
                "severity": "warn",
                "dte": 99,
                "strike": 195,
                "expiration": "2026-09-18",
                "underlying_price": 202.5,
                "moneyness_pct": 3.85,
                "operator_roll_preference": {"wait_for_underlying_price": 198},
            },
        ],
    }

    payload = src.evaluate(snapshot=snapshot, roll_watch=roll_watch, profiles=[], day="20260611")

    nvda = next(row for row in payload["underlyings"] if row["underlying"] == "NVDA")
    stance = nvda["chart_context"]["stance"]
    assert stance["roll_watch_status"] == "operator_wait_price_watch"
    assert stance["roll_watch_severity"] == "warn"
    assert stance["covered_call_count_for_underlying"] == 2
    assert stance["roll_trigger"] == 198


def test_position_study_attaches_operator_account_alias() -> None:
    snapshot = {
        "fetched": {
            "payload": {
                "accounts": [
                    {
                        "_broker_account": {"account_label": "account_1_5625", "account_number_tail": "5625"},
                        "securitiesAccount": {
                            "positions": [
                                {
                                    "instrument": {"assetType": "EQUITY", "symbol": "NVDA"},
                                    "longQuantity": 100,
                                    "shortQuantity": 0,
                                    "marketValue": 20250.0,
                                }
                            ]
                        },
                    }
                ]
            }
        }
    }
    aliases = {
        "schwab_accounts": {
            "account_1_5625": {
                "operator_account_label": "roth",
                "operator_account_kind": "roth",
                "trading_type": "cash",
            }
        }
    }

    payload = src.evaluate(snapshot=snapshot, roll_watch={}, profiles=[], day="20260611", account_aliases=aliases)

    position = payload["positions"][0]
    assert position["operator_account_label"] == "roth"
    assert position["operator_account_kind"] == "roth"
    assert payload["underlyings"][0]["operator_accounts"] == ["roth"]
    assert payload["underlyings"][0]["account_kinds"] == ["roth"]
