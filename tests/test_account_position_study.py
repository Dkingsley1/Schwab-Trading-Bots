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

    payload = src.evaluate(
        snapshot=snapshot, roll_watch=roll_watch, profiles=[], day="20260611"
    )

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
                        "_broker_account": {
                            "account_label": "account_1_9999",
                            "account_number_tail": "9999",
                        },
                        "securitiesAccount": {
                            "positions": [
                                {
                                    "instrument": {
                                        "assetType": "EQUITY",
                                        "symbol": "NVDA",
                                    },
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
            "account_1_9999": {
                "operator_account_label": "roth",
                "operator_account_kind": "roth",
                "trading_type": "cash",
            }
        }
    }

    payload = src.evaluate(
        snapshot=snapshot,
        roll_watch={},
        profiles=[],
        day="20260611",
        account_aliases=aliases,
    )

    position = payload["positions"][0]
    assert position["operator_account_label"] == "roth"
    assert position["operator_account_kind"] == "roth"
    assert payload["underlyings"][0]["operator_accounts"] == ["roth"]
    assert payload["underlyings"][0]["account_kinds"] == ["roth"]


def test_position_study_emits_redacted_balance_truth_for_empty_accounts() -> None:
    snapshot = {
        "fetched": {
            "payload": {
                "accounts": [
                    {
                        "_broker_account": {
                            "account_label": "account_1_1234",
                            "account_number_tail": "1234",
                        },
                        "securitiesAccount": {
                            "type": "CASH",
                            "isClosingOnlyRestricted": False,
                            "currentBalances": {
                                "liquidationValue": 2500.0,
                                "equity": 2500.0,
                                "cashBalance": 2500.0,
                                "availableFunds": 2500.0,
                                "buyingPower": 2500.0,
                                "marginBalance": 0.0,
                            },
                            "positions": [],
                        },
                    }
                ]
            }
        }
    }

    payload = src.evaluate(
        snapshot=snapshot, roll_watch={}, profiles=[], day="20260804"
    )

    assert payload["schema_version"] == 3
    assert payload["account_count"] == 1
    assert payload["position_count"] == 0
    assert payload["accounts"][0]["account_label"] == "account_1_1234"
    assert payload["accounts"][0]["liquidation_value"] == 2500.0
    assert payload["accounts"][0]["cash_balance"] == 2500.0
    assert payload["accounts"][0]["position_count"] == 0
    assert payload["portfolio_summary"]["liquidation_value"] == 2500.0


def test_position_study_distinguishes_limited_margin_from_provider_margin_type() -> (
    None
):
    snapshot = {
        "fetched": {
            "payload": {
                "accounts": [
                    {
                        "_broker_account": {
                            "account_label": "account_1_0000",
                            "account_number_tail": "0000",
                        },
                        "securitiesAccount": {
                            "accountNumber": "secret",
                            "type": "MARGIN",
                            "isIntradayMargin": True,
                            "currentBalances": {
                                "cashBalance": 0.0,
                                "availableFunds": 500.0,
                                "buyingPower": 1000.0,
                                "marginBalance": -100.0,
                                "accruedInterest": 0.0,
                                "pendingDeposits": 0.0,
                                "maintenanceCall": 0.0,
                                "regTCall": 0.0,
                            },
                            "positions": [],
                        },
                    }
                ]
            }
        }
    }
    aliases = {
        "schwab_accounts": {
            "tail:0000": {
                "account_policy_key": "schwab_cash_account_1",
                "operator_account_label": "Cash Account 1",
                "operator_account_kind": "cash",
                "tax_wrapper": "taxable",
                "trading_access": "limited_margin",
                "borrowing_allowed": False,
                "cash_only_live_budget": True,
                "canary_candidate": True,
                "canary_cap_usd": 200,
                "allowed_live_routes": ["dividend_liquid_etf_candidate_v1"],
                "operator_verified": True,
                "verification_source": "operator",
            }
        }
    }

    payload = src.evaluate(
        snapshot=snapshot,
        roll_watch={},
        profiles=[],
        day="20260828",
        account_aliases=aliases,
    )
    account = payload["accounts"][0]

    assert account["account_type"] == "MARGIN"
    assert account["operator_trading_type"] == "limited_margin"
    assert account["borrowing_allowed"] is False
    assert (
        account["debit_status"]
        == "negative_provider_balance_not_confirmed_as_borrowing"
    )
    assert account["canary_preflight"]["account_preflight_ready"] is False
    assert (
        account["account_capability_truth"]["redaction"]["raw_account_number_emitted"]
        is False
    )


def test_compact_account_capability_context_is_redacted_and_fail_closed() -> None:
    payload = {
        "accounts": [
            {
                "_broker_account": {
                    "account_label": "account_1_0000",
                    "account_number_tail": "0000",
                },
                "securitiesAccount": {
                    "accountNumber": "must-not-appear",
                    "type": "MARGIN",
                    "currentBalances": {
                        "cashBalance": 200.0,
                        "availableFunds": 900.0,
                        "buyingPower": 1800.0,
                        "marginBalance": -100.0,
                        "accruedInterest": 0.0,
                    },
                    "positions": [],
                },
            }
        ]
    }
    aliases = {
        "schwab_accounts": {
            "tail:0000": {
                "account_policy_key": "schwab_cash_account_1",
                "operator_account_kind": "cash",
                "tax_wrapper": "taxable",
                "trading_access": "limited_margin",
                "borrowing_allowed": False,
                "cash_only_live_budget": True,
                "canary_candidate": True,
                "canary_cap_usd": 200.0,
                "allowed_live_routes": ["dividend_liquid_etf_candidate_v1"],
                "operator_verified": True,
                "verification_source": "operator",
            }
        }
    }

    context = src.build_account_capability_context(payload, account_aliases=aliases)

    assert context["status"] == "ready"
    assert context["limited_margin_account_count"] == 1
    assert context["borrowing_enabled_account_count"] == 0
    assert context["canary_preflight_ready_account_count"] == 1
    assert context["live_execution_authority"] is False
    assert context["accounts"][0]["provider_account_type"] == "MARGIN"
    assert context["accounts"][0]["trading_access"] == "limited_margin"
    assert context["accounts"][0]["live_execution_authority"] is False
    assert "must-not-appear" not in str(context)
