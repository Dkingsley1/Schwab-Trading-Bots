from __future__ import annotations

from datetime import datetime, timedelta, timezone

from scripts.ops import account_buildout_planner as src


NOW = datetime(2026, 8, 4, 14, 0, tzinfo=timezone.utc)


def _policy() -> dict:
    return {
        "enabled": True,
        "cash_reserve_fraction": 0.0,
        "max_deployable_equity_fraction": 1.0,
        "max_single_symbol_fraction": 0.2,
        "max_stage_account_fraction": 0.02,
        "max_reduction_account_fraction_per_cycle": 0.1,
        "minimum_order_notional": 1.0,
        "fractional_equities": True,
        "allow_margin_expansion": False,
        "freshness": {
            "account_study_max_age_seconds": 300,
            "position_opportunity_max_age_seconds": 300,
            "portfolio_allocator_max_age_seconds": 300,
            "portfolio_risk_max_age_seconds": 300,
        },
    }


def _account(label: str, equity: float, cash: float) -> dict:
    return {
        "account_label": label,
        "liquidation_value": equity,
        "equity": equity,
        "cash_balance": cash,
        "available_funds": cash,
        "account_type": "CASH",
        "flags": {"closing_only": False, "in_margin_call": False},
    }


def _inputs(*, gross_budget: float = 0.4) -> tuple[dict, dict, dict, dict]:
    timestamp = NOW.isoformat()
    study = {
        "timestamp_utc": timestamp,
        "ok": True,
        "accounts": [_account("small", 1_000.0, 1_000.0), _account("large", 100_000.0, 100_000.0)],
        "positions": [
            {
                "account_label": "small",
                "symbol": "AAPL",
                "underlying": "AAPL",
                "asset_type": "EQUITY",
                "quantity": 1.0,
                "market_value": 100.0,
            },
            {
                "account_label": "large",
                "symbol": "AAPL",
                "underlying": "AAPL",
                "asset_type": "EQUITY",
                "quantity": 100.0,
                "market_value": 10_000.0,
            },
        ],
        "underlyings": [],
    }
    opportunities = {"timestamp_utc": timestamp, "ok": True, "observations": [], "candidates": []}
    allocator = {
        "timestamp_utc": timestamp,
        "ok": True,
        "summary": {"gross_budget": gross_budget},
        "approved_intents": [
            {"symbol": "AAPL", "sleeve": "core", "side": "BUY", "approved_qty": 1.0, "price": 100.0},
            {"symbol": "MSFT", "sleeve": "core", "side": "BUY", "approved_qty": 1.0, "price": 100.0},
        ],
    }
    risk = {
        "timestamp_utc": timestamp,
        "limits": {"gross_exposure_cap": 0.4, "max_single_symbol_share": 0.2},
    }
    return study, opportunities, allocator, risk


def _evaluate(study: dict, opportunities: dict, allocator: dict, risk: dict, *, policy: dict | None = None) -> dict:
    return src.evaluate(
        study=study,
        opportunities=opportunities,
        allocator=allocator,
        risk=risk,
        policy=policy or _policy(),
        now=NOW,
    )


def test_buildout_plans_scale_with_account_equity_and_existing_positions() -> None:
    payload = _evaluate(*_inputs())

    assert payload["overall_status"] == "ready"
    assert payload["plan_state"] == "plan_ready"
    assert payload["action_count"] == 4
    small = {row["symbol"]: row for row in payload["actions"] if row["account_label"] == "small"}
    large = {row["symbol"]: row for row in payload["actions"] if row["account_label"] == "large"}
    assert small["AAPL"]["proposed_notional_change"] == 100.0
    assert small["MSFT"]["proposed_notional_change"] == 200.0
    assert large["AAPL"]["proposed_notional_change"] == 10_000.0
    assert large["MSFT"]["proposed_notional_change"] == 20_000.0
    assert large["AAPL"]["proposed_quantity_change"] == 100 * small["AAPL"]["proposed_quantity_change"]
    assert all(row["execution_allowed"] is False for row in payload["actions"])
    assert all(row["staging"]["expanded_rows_emitted"] is False for row in payload["actions"])


def test_fractional_equities_support_small_accounts_without_whole_share_assumptions() -> None:
    study, opportunities, allocator, risk = _inputs()
    study["accounts"] = [_account("micro", 25.0, 25.0)]
    study["positions"] = []
    allocator["approved_intents"] = [
        {"symbol": "AAPL", "sleeve": "core", "side": "BUY", "approved_qty": 1.0, "price": 200.0}
    ]
    policy = _policy()
    policy["max_single_symbol_fraction"] = 0.4
    risk["limits"]["max_single_symbol_share"] = 0.4

    payload = _evaluate(study, opportunities, allocator, risk, policy=policy)

    assert payload["action_count"] == 1
    assert payload["actions"][0]["proposed_quantity_change"] == 0.05
    assert payload["actions"][0]["proposed_notional_change"] == 10.0


def test_zero_allocator_budget_is_ready_observe_only_not_fake_degradation() -> None:
    payload = _evaluate(*_inputs(gross_budget=0.0))

    assert payload["ok"] is True
    assert payload["overall_status"] == "ready"
    assert payload["plan_state"] == "observe_only"
    assert payload["action_count"] == 0
    assert "allocator_gross_budget_zero" in payload["holds"]


def test_existing_exposure_over_cap_prevents_additions() -> None:
    study, opportunities, allocator, risk = _inputs()
    study["accounts"] = [_account("full", 1_000.0, 1_000.0)]
    study["positions"] = [
        {
            "account_label": "full",
            "symbol": "AAPL",
            "underlying": "AAPL",
            "asset_type": "EQUITY",
            "quantity": 5.0,
            "market_value": 500.0,
        }
    ]

    payload = _evaluate(study, opportunities, allocator, risk)

    assert payload["action_count"] == 0
    assert payload["accounts"][0]["gross_headroom"] == 0.0
    assert "existing_exposure_at_or_above_effective_gross_cap" in payload["accounts"][0]["holds"]


def test_sell_signal_cannot_open_a_new_short_position() -> None:
    study, opportunities, allocator, risk = _inputs()
    study["accounts"] = [_account("cash", 10_000.0, 10_000.0)]
    study["positions"] = []
    allocator["approved_intents"] = [
        {"symbol": "MSFT", "sleeve": "core", "side": "SELL", "approved_qty": 2.0, "price": 100.0}
    ]

    payload = _evaluate(study, opportunities, allocator, risk)

    assert payload["action_count"] == 0
    assert payload["skipped_signals"] == [
        {"account_label": "cash", "symbol": "MSFT", "reason": "sell_cannot_open_new_short"}
    ]
    assert payload["safety_contract"]["new_short_positions_allowed"] is False


def test_roll_reviews_are_never_given_automatic_quantities() -> None:
    study, opportunities, allocator, risk = _inputs(gross_budget=0.0)
    opportunities["candidates"] = [
        {
            "underlying": "NVDA",
            "accounts": ["small"],
            "position_action": "ROLL_REVIEW",
            "reason": "covered_call_operator_wait_price_watch",
        }
    ]

    payload = _evaluate(study, opportunities, allocator, risk)

    assert payload["plan_state"] == "review_only"
    assert payload["review_count"] == 1
    assert payload["reviews"][0]["quantity"] is None
    assert payload["reviews"][0]["execution_allowed"] is False


def test_round_trip_candidates_are_supplemental_reviews_without_quantities() -> None:
    study, opportunities, allocator, risk = _inputs(gross_budget=0.0)
    round_trips = {
        "timestamp_utc": NOW.isoformat(),
        "overall_status": "ready",
        "observations": [
            {
                "account_label": "small",
                "symbol": "AAPL",
                "action": "PAPER_REENTRY_CANDIDATE",
                "reasons": ["reentry_score_and_discount_satisfied"],
                "reentry_signal": {"score": 0.82},
                "zones": {"reentry": {"upper_price": 95.0}},
            }
        ],
    }

    payload = src.evaluate(
        study=study,
        opportunities=opportunities,
        round_trips=round_trips,
        allocator=allocator,
        risk=risk,
        policy=_policy(),
        now=NOW,
    )

    review = payload["reviews"][0]
    assert payload["round_trip_review_count"] == 1
    assert review["action"] == "PAPER_REENTRY_CANDIDATE"
    assert review["quantity"] is None
    assert review["notional"] is None
    assert review["execution_allowed"] is False


def test_stale_round_trip_input_is_suppressed_without_blocking_buildout() -> None:
    study, opportunities, allocator, risk = _inputs()
    round_trips = {
        "timestamp_utc": (NOW - timedelta(hours=1)).isoformat(),
        "observations": [{"account_label": "small", "symbol": "AAPL", "action": "PAPER_EXIT_CANDIDATE"}],
    }

    payload = src.evaluate(
        study=study,
        opportunities=opportunities,
        round_trips=round_trips,
        allocator=allocator,
        risk=risk,
        policy=_policy(),
        now=NOW,
    )

    assert payload["plan_state"] == "plan_ready"
    assert payload["round_trip_review_count"] == 0
    assert "position_round_trip_stale_review_suppressed" in payload["holds"]


def test_stale_account_truth_blocks_and_stale_planning_inputs_hold_actions() -> None:
    study, opportunities, allocator, risk = _inputs()
    study["timestamp_utc"] = (NOW - timedelta(hours=1)).isoformat()

    blocked = _evaluate(study, opportunities, allocator, risk)

    assert blocked["ok"] is False
    assert blocked["overall_status"] == "blocked"
    assert blocked["action_count"] == 0

    study["timestamp_utc"] = NOW.isoformat()
    allocator["timestamp_utc"] = (NOW - timedelta(hours=1)).isoformat()
    held = _evaluate(study, opportunities, allocator, risk)

    assert held["ok"] is True
    assert held["plan_state"] == "held_by_freshness"
    assert held["action_count"] == 0


def test_fresh_wrapper_timestamp_cannot_launder_stale_upstream_sources() -> None:
    study, opportunities, allocator, risk = _inputs()
    allocator["overall_status"] = "ready"
    allocator["input_freshness"] = {"sources_ready": False, "stale_sources": ["intents"]}
    risk["overall_status"] = "ready"
    risk["input_freshness"] = {"sources_ready": True, "stale_sources": []}

    payload = _evaluate(study, opportunities, allocator, risk)

    assert payload["ok"] is True
    assert payload["plan_state"] == "held_by_upstream_contract"
    assert payload["action_count"] == 0
    assert "portfolio_allocator_upstream_sources_stale" in payload["holds"]
    assert payload["regression_contract"]["fresh_wrapper_timestamps_cannot_launder_stale_upstream_sources"] is True
