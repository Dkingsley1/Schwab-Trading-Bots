from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone

from scripts.ops.live_canary_dress_rehearsal import (
    READ_ONLY_ENVIRONMENT,
    build_dress_rehearsal_payload,
)

ACCOUNT_REFERENCE = "opaque-account-reference-for-test"


def _plan() -> dict:
    return {
        "policy_id": "live_canary_micro_200_v1",
        "account_capital_usd": 200.0,
        "account_policy_key": "schwab_cash_account_1",
        "execution_route_id": "dividend_liquid_etf_candidate_v1",
        "hard_limits": {
            "max_order_notional_usd": 100.0,
            "max_order_quantity": 1.0,
            "max_concurrent_positions": 1,
        },
        "stages": [{"stage": 1, "symbols": ["SCHD"]}],
        "activation_contract": {"unfilled_order_cancel_deadline_seconds": 60},
    }


def _firewall() -> dict:
    return {
        "max_quote_age_seconds": 15.0,
        "max_account_snapshot_age_seconds": 30.0,
        "max_spread_bps": 75.0,
        "max_future_clock_skew_seconds": 2.0,
        "live_execution_envelope_ttl_seconds": 15.0,
    }


def _study(now: datetime, *, cash: float = 200.0) -> dict:
    return {
        "timestamp_utc": now.isoformat(),
        "accounts": [
            {
                "account_policy_key": "schwab_cash_account_1",
                "operator_account_kind": "cash",
                "operator_trading_type": "limited_margin",
                "borrowing_allowed": False,
                "cash_balance": cash,
                "flags": {"closing_only": False},
                "canary_preflight": {
                    "blockers": (
                        [] if cash >= 200.0 else ["canary_cash_not_funded_and_settled"]
                    )
                },
                "account_capability_truth": {
                    "operator_classification": {
                        "account_kind": "cash",
                        "trading_access": "limited_margin",
                    },
                    "balance_truth": {
                        "cash_balance": cash,
                        "cash_available_for_trading": cash,
                        "pending_deposits": 0.0,
                    },
                    "debit_truth": {
                        "provider_margin_balance": -2512.65,
                        "status": "negative_provider_balance_not_confirmed_as_borrowing",
                        "interest_bearing_borrowing_confirmed": False,
                        "requires_broker_ui_confirmation": True,
                    },
                    "broker_call_truth": {"in_call": False},
                    "position_collateral_truth": {
                        "covered_short_option_count": 1,
                        "uncovered_short_option_count": 0,
                    },
                },
            }
        ],
        "positions": [
            {
                "account_policy_key": "schwab_cash_account_1",
                "symbol": "NVDA",
                "underlying": "NVDA",
                "asset_type": "EQUITY",
                "quantity": 100.1446,
                "short_quantity": 0.0,
            },
            {
                "account_policy_key": "schwab_cash_account_1",
                "symbol": "NVDA  280121C00195000",
                "underlying": "NVDA",
                "asset_type": "OPTION",
                "quantity": -1.0,
                "short_quantity": 1.0,
            },
        ],
    }


def _quote(now: datetime, *, observed_at: datetime | None = None) -> dict:
    provider_time = observed_at or now
    epoch_ms = int(provider_time.timestamp() * 1000)
    return {
        "ok": True,
        "operation": "get_quote",
        "method": "get_quote",
        "status_code": 200,
        "latency_ms": 12.5,
        "quote_snapshot": {
            "bid_price": 29.01,
            "ask_price": 29.03,
            "last_price": 29.02,
            "mark_price": 29.02,
            "raw_payload": {
                "SCHD": {
                    "realtime": True,
                    "quote": {
                        "bidPrice": 29.01,
                        "askPrice": 29.03,
                        "bidTime": epoch_ms,
                        "askTime": epoch_ms,
                        "askMICId": "XNAS",
                    },
                }
            },
        },
    }


def _preflight(*, ready: bool, blockers: list[str] | None = None) -> dict:
    return {
        "ready": ready,
        "receipt_sha256": "b" * 64,
        "account_policy_key": "schwab_cash_account_1",
        "account_reference_sha256": hashlib.sha256(
            ACCOUNT_REFERENCE.encode()
        ).hexdigest(),
        "equity_session": {"ready": ready, "state": "open" if ready else "closed"},
        "blockers": list(blockers or []),
    }


def _build(
    *,
    now: datetime,
    cash: float = 200.0,
    quote: dict | None = None,
    preflight: dict | None = None,
) -> dict:
    return build_dress_rehearsal_payload(
        now=now,
        plan=_plan(),
        firewall=_firewall(),
        candidate={"candidate_id": "pc-connected-rehearsal-test"},
        account_study=_study(now, cash=cash),
        account_study_sha256="a" * 64,
        account_reference=ACCOUNT_REFERENCE,
        expected_account_reference=ACCOUNT_REFERENCE,
        quote_result=quote or _quote(now),
        preflight_receipt=preflight or _preflight(ready=True),
        account_refresh_summary={"ok": True, "account_count": 3, "position_rows": 7},
        policy_sha256="c" * 64,
        symbol="SCHD",
    )


def test_funded_connected_rehearsal_builds_exact_cash_only_projection() -> None:
    now = datetime(2026, 8, 28, 15, 0, tzinfo=timezone.utc)
    payload = _build(now=now)

    assert payload["ok"] is True
    assert payload["canary_ready"] is True
    assert payload["exact_order_preview"]["order_spec"]["orderType"] == "LIMIT"
    assert payload["exact_order_preview"]["order_spec"]["session"] == "NORMAL"
    assert payload["exact_order_preview"]["order_spec"]["duration"] == "DAY"
    assert payload["exact_order_preview"]["quantity"] == 1.0
    assert payload["funding_and_settlement"]["candidate_order_notional_usd"] == 29.03
    assert (
        payload["funding_and_settlement"]["projected_settled_cash_after_fill_usd"]
        == 170.97
    )
    assert payload["live_execution_authority"] is False
    assert payload["live_order_attempted"] is False


def test_zero_cash_is_a_funding_shortfall_and_never_uses_buying_power() -> None:
    now = datetime(2026, 8, 28, 15, 0, tzinfo=timezone.utc)
    payload = _build(
        now=now,
        cash=0.0,
        preflight=_preflight(
            ready=False,
            blockers=["canary_settled_cash_not_broker_visible"],
        ),
    )

    funding = payload["funding_and_settlement"]
    assert payload["ok"] is True
    assert payload["canary_ready"] is False
    assert funding["canary_cap_shortfall_usd"] == 200.0
    assert funding["candidate_order_cash_shortfall_usd"] == 29.03
    assert funding["projected_settled_cash_after_fill_usd"] is None
    assert funding["borrowing_or_buying_power_used"] is False
    assert "candidate_order_not_fully_cash_funded" in payload["blockers"]


def test_provider_negative_balance_stays_diagnostic_not_debt_or_cash() -> None:
    now = datetime(2026, 8, 28, 15, 0, tzinfo=timezone.utc)
    payload = _build(now=now)
    diagnostic = payload["account"]["provider_balance_diagnostic"]

    assert diagnostic["provider_margin_balance"] == -2512.65
    assert diagnostic["classified_as_debt"] is False
    assert diagnostic["included_in_settled_cash"] is False
    assert diagnostic["interest_bearing_borrowing_confirmed"] is False
    assert payload["funding_and_settlement"]["settled_cash_broker_visible_usd"] == 200.0


def test_candidate_buy_preserves_existing_covered_nvda_collateral() -> None:
    now = datetime(2026, 8, 28, 15, 0, tzinfo=timezone.utc)
    payload = _build(now=now)
    safety = payload["post_fill_projection"]["covered_position_safety"]
    nvda = next(
        row for row in safety["collateral_by_underlying"] if row["underlying"] == "NVDA"
    )

    assert safety["candidate_order_reduces_existing_collateral"] is False
    assert safety["coverage_unchanged_by_candidate_buy"] is True
    assert safety["uncovered_short_option_count"] == 0
    assert nvda["reserved_equity_shares"] == 100.0
    assert nvda["unencumbered_equity_shares"] == 0.1446


def test_stale_provider_quote_blocks_without_turning_off_read_only_proof() -> None:
    now = datetime(2026, 8, 28, 15, 0, tzinfo=timezone.utc)
    payload = _build(
        now=now,
        quote=_quote(now, observed_at=now - timedelta(minutes=5)),
    )

    assert payload["ok"] is True
    assert payload["canary_ready"] is False
    assert "schwab_quote_stale" in payload["blockers"]
    assert "quote_is_stale" not in payload["blockers"]
    assert "order_intent_risk_decision_not_approved" not in payload["blockers"]
    assert payload["broker_network_read_only"] is True
    assert payload["broker_mutation_attempted"] is False


def test_wide_or_malformed_quote_fails_closed() -> None:
    now = datetime(2026, 8, 28, 15, 0, tzinfo=timezone.utc)
    malformed = _quote(now)
    malformed["quote_snapshot"]["bid_price"] = 28.0
    malformed["quote_snapshot"]["ask_price"] = 29.0
    payload = _build(now=now, quote=malformed)

    assert payload["canary_ready"] is False
    assert "schwab_quote_spread_not_canary_ready" in payload["blockers"]
    assert payload["live_order_attempted"] is False


def test_designated_account_reference_mismatch_blocks() -> None:
    now = datetime(2026, 8, 28, 15, 0, tzinfo=timezone.utc)
    payload = build_dress_rehearsal_payload(
        now=now,
        plan=_plan(),
        firewall=_firewall(),
        candidate={"candidate_id": "pc-connected-rehearsal-test"},
        account_study=_study(now),
        account_study_sha256="a" * 64,
        account_reference=ACCOUNT_REFERENCE,
        expected_account_reference="different-designated-reference",
        quote_result=_quote(now),
        preflight_receipt=_preflight(ready=True),
        account_refresh_summary={"ok": True, "account_count": 3, "position_rows": 7},
        policy_sha256="c" * 64,
        symbol="SCHD",
    )

    assert payload["canary_ready"] is False
    assert "live_account_not_designated_canary_account" in payload["blockers"]
    assert payload["account_reference_matches_designated_policy"] is False


def test_artifact_never_contains_raw_account_reference_or_provider_payload() -> None:
    now = datetime(2026, 8, 28, 15, 0, tzinfo=timezone.utc)
    payload = _build(now=now)
    encoded = json.dumps(payload, sort_keys=True)

    assert ACCOUNT_REFERENCE not in encoded
    assert "raw_payload" not in encoded
    assert payload["redaction"]["raw_account_hash_emitted"] is False
    assert payload["redaction"]["provider_quote_payload_emitted"] is False


def test_rehearsal_forces_every_live_runtime_switch_off() -> None:
    assert READ_ONLY_ENVIRONMENT == {
        "ALLOW_ORDER_EXECUTION": "0",
        "MARKET_DATA_ONLY": "1",
        "TOP_BOT_ENABLE_LIVE_EXECUTION": "0",
        "EXECUTION_LANE_LIVE_ENABLED": "0",
        "RUN_ALL_SLEEVES_WITH_LIVE_EXECUTOR": "0",
    }
