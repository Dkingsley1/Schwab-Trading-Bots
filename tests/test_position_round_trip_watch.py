from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.ops.position_round_trip_watch import evaluate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY = json.loads((PROJECT_ROOT / "config" / "position_round_trip_policy_v1.json").read_text())


def _study(*, quantity: float = 10.0, short_calls: int = 0, average_price: float = 100.0, portfolio_value: float = 1_000_000.0) -> dict:
    positions = [
        {
            "account_label": "account_1_1111",
            "symbol": "XYZ",
            "underlying": "XYZ",
            "asset_type": "EQUITY",
            "quantity": quantity,
            "market_value": quantity * 150.0,
            "average_price": average_price,
        }
    ]
    if short_calls:
        positions.append(
            {
                "account_label": "account_1_1111",
                "symbol": "XYZ   261218C00150000",
                "underlying": "XYZ",
                "asset_type": "OPTION",
                "quantity": -float(short_calls),
                "market_value": -500.0 * short_calls,
                "average_price": 5.0,
            }
        )
    return {
        "timestamp_utc": "2026-08-04T15:00:00+00:00",
        "ok": True,
        "portfolio_summary": {"liquidation_value": portfolio_value, "equity": portfolio_value},
        "positions": positions,
        "underlyings": [
            {
                "underlying": "XYZ",
                "chart_context": {
                    "profile": "test",
                    "stance": {"master_action": "SELL", "master_score": 0.9},
                },
            }
        ],
    }


def _exit_chart(now: datetime) -> dict:
    daily = []
    for index in range(60):
        close = 100.0 + index
        daily.append({"open": close - 0.5, "high": close + 1.0, "low": close - 1.0, "close": close, "volume": 1000})
    intraday = []
    values = [160.0 + index * 0.05 for index in range(24)] + [161.0, 160.5, 160.0, 159.5, 159.0, 158.5]
    for value in values:
        intraday.append({"open": value, "high": value + 0.2, "low": value - 0.2, "close": value, "volume": 1000})
    return {"symbols": {"XYZ": {"fetched_at_utc": now.isoformat(), "daily_bars": daily, "intraday_bars": intraday}}}


def _reentry_chart(now: datetime) -> dict:
    daily = []
    for index in range(60):
        close = 150.0 - index
        daily.append({"open": close + 0.5, "high": close + 1.0, "low": close - 1.0, "close": close, "volume": 1000})
    values = [88.0 for _ in range(24)] + [88.0, 88.2, 88.5, 88.9, 89.4, 90.0]
    intraday = [
        {"open": value, "high": value + 0.2, "low": value - 0.2, "close": value, "volume": 1000}
        for value in values
    ]
    return {"symbols": {"XYZ": {"fetched_at_utc": now.isoformat(), "daily_bars": daily, "intraday_bars": intraday}}}


def _risk(max_share: float = 0.15) -> dict:
    return {"ok": True, "limits": {"max_single_symbol_share": max_share}}


def _profile(treatment: str = "taxable") -> dict:
    return {"account_tax_treatment_by_label": {"account_1_1111": treatment}}


def _evaluate(study: dict, charts: dict, *, now: datetime, state: dict | None = None, events: list | None = None):
    return evaluate(
        study,
        charts,
        risk=_risk(),
        tax_estimate={"status": "needs_taxpayer_or_ledger_evidence", "federal": {}, "taxable_activity": {}},
        tax_profile=_profile(),
        dividend_calendar={},
        state=state or {},
        paper_events=events or [],
        policy=POLICY,
        now=now,
    )


def test_exit_candidate_is_paper_only_and_prices_unknown_tax_friction() -> None:
    now = datetime(2026, 8, 4, 15, 0, tzinfo=timezone.utc)
    payload, _ = _evaluate(_study(), _exit_chart(now), now=now)
    observation = payload["observations"][0]
    assert observation["action"] == "PAPER_EXIT_CANDIDATE"
    assert observation["execution_contract"]["live_execution_allowed"] is False
    assert observation["execution_contract"]["quantity_recommendation"] is None
    assert observation["tax_and_cost"]["tax_friction_bps"] == 300.0
    assert observation["tax_and_cost"]["minimum_required_reentry_discount_fraction"] == 0.0324
    assert observation["tax_and_cost"]["estimated_exit_tax_reserve_usd"] is None
    assert observation["tax_and_cost"]["verified_tax_estimate_required_before_live"] is True


def test_short_call_coverage_reserves_underlying_before_exit() -> None:
    now = datetime(2026, 8, 4, 15, 0, tzinfo=timezone.utc)
    payload, _ = _evaluate(_study(quantity=100.0, short_calls=1), _exit_chart(now), now=now)
    observation = payload["observations"][0]
    assert observation["position"]["covered_call_reserved_quantity"] == 100.0
    assert observation["position"]["maximum_unencumbered_quantity"] == 0.0
    assert "all_equity_reserved_for_short_call_coverage" in observation["hard_holds"]
    assert observation["action"] == "HOLD"


def test_stale_chart_blocks_candidate_instead_of_reusing_old_signal() -> None:
    now = datetime(2026, 8, 4, 15, 0, tzinfo=timezone.utc)
    charts = _exit_chart(now - timedelta(hours=2))
    payload, _ = _evaluate(_study(), charts, now=now)
    observation = payload["observations"][0]
    assert observation["action"] == "HOLD"
    assert "chart_evidence_missing_or_stale" in observation["hard_holds"]
    assert payload["stale_chart_count"] == 1


def test_exit_fill_enforces_settlement_before_reentry() -> None:
    now = datetime(2026, 8, 4, 15, 0, tzinfo=timezone.utc)
    event = {
        "event_id": "exit-1",
        "round_trip_leg": "exit",
        "account_label": "account_1_1111",
        "symbol": "XYZ",
        "fill_price": 100.0,
        "quantity": 5.0,
        "timestamp_utc": now.isoformat(),
    }
    payload, state = _evaluate(_study(), _reentry_chart(now), now=now, events=[event])
    observation = payload["observations"][0]
    assert observation["action"] == "WAIT_REENTRY"
    assert "settlement_or_wash_sale_window_active" in observation["reasons"]
    assert state["positions"]["account_1_1111|XYZ"]["settlement_date"] == "2026-08-05"


def test_reentry_candidate_requires_discount_and_reversal_after_settlement() -> None:
    now = datetime(2026, 8, 10, 15, 0, tzinfo=timezone.utc)
    event = {
        "event_id": "exit-2",
        "round_trip_leg": "exit",
        "account_label": "account_1_1111",
        "symbol": "XYZ",
        "fill_price": 100.0,
        "quantity": 5.0,
        "timestamp_utc": "2026-08-03T15:00:00+00:00",
    }
    payload, _ = _evaluate(_study(average_price=80.0), _reentry_chart(now), now=now, events=[event])
    observation = payload["observations"][0]
    assert observation["action"] == "PAPER_REENTRY_CANDIDATE"
    assert observation["reentry_signal"]["discount_fraction"] == 0.1
    assert observation["zones"]["reentry"]["upper_price"] == 96.76


def test_loss_exit_holds_reentry_for_wash_sale_window() -> None:
    now = datetime(2026, 8, 10, 15, 0, tzinfo=timezone.utc)
    event = {
        "event_id": "exit-loss",
        "round_trip_leg": "exit",
        "account_label": "account_1_1111",
        "symbol": "XYZ",
        "fill_price": 90.0,
        "quantity": 5.0,
        "timestamp_utc": "2026-08-03T15:00:00+00:00",
    }
    payload, state = _evaluate(_study(average_price=100.0), _reentry_chart(now), now=now, events=[event])
    observation = payload["observations"][0]
    assert observation["action"] == "WAIT_REENTRY"
    assert state["positions"]["account_1_1111|XYZ"]["earliest_reentry_date"] == "2026-09-02"


def test_reentry_timeout_reaches_cooldown_after_second_failure() -> None:
    now = datetime(2026, 8, 30, 15, 0, tzinfo=timezone.utc)
    state = {
        "schema_version": 1,
        "positions": {
            "account_1_1111|XYZ": {
                "phase": "paper_exited_waiting_reentry",
                "position_round_trip_id": "rt-1",
                "exit_fill_price": 100.0,
                "exit_quantity": 5.0,
                "exit_filled_at_utc": "2026-08-01T15:00:00+00:00",
                "settlement_date": "2026-08-03",
                "failed_reentry_count": 1,
            }
        },
        "completed_round_trips": [],
    }
    payload, updated = _evaluate(_study(average_price=80.0), _reentry_chart(now), now=now, state=state)
    lifecycle = updated["positions"]["account_1_1111|XYZ"]
    assert "maximum_reentry_wait_exceeded" in payload["observations"][0]["hard_holds"]
    assert lifecycle["phase"] == "paper_reentry_failed_cooldown"
    assert lifecycle["failed_reentry_count"] == 2
    assert lifecycle["cooldown_until_utc"] is not None


def test_completed_fill_updates_paper_proof_without_enabling_live() -> None:
    now = datetime(2026, 8, 10, 15, 0, tzinfo=timezone.utc)
    state = {
        "schema_version": 1,
        "positions": {
            "account_1_1111|XYZ": {
                "phase": "paper_exited_waiting_reentry",
                "position_round_trip_id": "rt-2",
                "exit_fill_price": 100.0,
                "exit_quantity": 5.0,
                "exit_filled_at_utc": "2026-08-01T15:00:00+00:00",
                "settlement_date": "2026-08-03",
            }
        },
        "completed_round_trips": [],
    }
    event = {
        "event_id": "reentry-1",
        "round_trip_leg": "reentry",
        "account_label": "account_1_1111",
        "symbol": "XYZ",
        "fill_price": 90.0,
        "quantity": 5.0,
        "timestamp_utc": now.isoformat(),
    }
    payload, updated = _evaluate(_study(), _reentry_chart(now), now=now, state=state, events=[event])
    assert len(updated["completed_round_trips"]) == 1
    assert updated["completed_round_trips"][0]["post_cost_edge_bps"] == 676.0
    assert updated["completed_round_trips"][0]["edge_is_after_modeled_tax_hurdle"] is True
    assert payload["paper_proof"]["completed_round_trips"] == 1
    assert payload["paper_proof"]["paper_promotion_evidence_ready"] is False
    assert payload["paper_proof"]["live_execution_allowed"] is False
