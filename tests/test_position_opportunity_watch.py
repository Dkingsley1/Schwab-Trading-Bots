from __future__ import annotations

from datetime import datetime, timezone

from scripts.ops import position_opportunity_watch as src


NOW = datetime(2026, 8, 4, 14, 0, tzinfo=timezone.utc)


def _row(symbol: str, *, quantity: float, action: str, vote: float, age_minutes: int = 5) -> dict:
    timestamp = datetime(2026, 8, 4, 13, 55 - age_minutes + 5, tzinfo=timezone.utc).isoformat()
    return {
        "underlying": symbol,
        "accounts": ["account_1"],
        "equity_quantity": quantity,
        "option_contract_net": 0.0,
        "market_value": 1000.0,
        "chart_context": {
            "timestamp_utc": timestamp,
            "profile": "conservative_equities_schwab",
            "market": {"last_price": 100.0, "spread_bps": 1.0},
            "stance": {
                "master_action": action,
                "master_score": 0.7 if action == "BUY" else 0.3,
                "master_vote": vote,
                "directional_trigger": 0.4,
                "deployability": 0.8,
            },
        },
    }


def test_position_watch_maps_fresh_existing_decisions_to_position_aware_candidates() -> None:
    study = {
        "ok": True,
        "underlyings": [
            _row("AAPL", quantity=10.0, action="BUY", vote=0.6),
            _row("MSFT", quantity=8.0, action="SELL", vote=-0.6),
            {
                "underlying": "PG",
                "accounts": ["account_1"],
                "equity_quantity": 5.0,
                "chart_context": {"profile": "broker_position_mark", "market": {"last_price": 150.0}, "stance": {}},
            },
        ],
    }

    payload = src.evaluate(study, now=NOW)

    candidates = {row["underlying"]: row for row in payload["candidates"]}
    observations = {row["underlying"]: row for row in payload["observations"]}
    assert candidates["AAPL"]["position_action"] == "ADD"
    assert candidates["MSFT"]["position_action"] == "REDUCE"
    assert observations["PG"]["state"] == "abstain"
    assert observations["PG"]["reason"] == "no_fresh_model_decision_for_position"
    assert all(row["execution_contract"]["direct_intent_publish_allowed"] is False for row in payload["observations"])
    assert payload["safety_contract"]["live_execution_allowed"] is False


def test_position_watch_surfaces_covered_call_roll_as_review_only() -> None:
    study = {
        "ok": True,
        "underlyings": [
            {
                "underlying": "NVDA",
                "accounts": ["account_1"],
                "equity_quantity": 100.0,
                "option_contract_net": -1.0,
                "chart_context": {
                    "profile": "covered_call_roll_watch",
                    "market": {"last_price": 207.0},
                    "stance": {"roll_watch_status": "roll_window_active", "roll_watch_severity": "critical"},
                },
            }
        ],
    }

    payload = src.evaluate(study, now=NOW)
    candidate = payload["candidates"][0]

    assert candidate["position_action"] == "ROLL_REVIEW"
    assert candidate["state"] == "review_candidate"
    assert candidate["execution_contract"]["quantity_recommendation"] is None
    assert payload["safety_contract"]["covered_call_rolls_are_review_only"] is True


def test_position_watch_forces_stale_decisions_and_uncovered_sell_signals_to_abstain() -> None:
    stale = _row("AAPL", quantity=10.0, action="BUY", vote=0.8)
    stale["chart_context"]["timestamp_utc"] = "2026-08-04T12:00:00+00:00"
    uncovered_sell = _row("TSLA", quantity=0.0, action="SELL", vote=-0.8)

    payload = src.evaluate({"ok": True, "underlyings": [stale, uncovered_sell]}, now=NOW)
    rows = {row["underlying"]: row for row in payload["observations"]}

    assert rows["AAPL"]["reason"] == "position_model_decision_stale"
    assert rows["TSLA"]["reason"] == "sell_signal_without_long_equity_position"
    assert payload["candidate_count"] == 0
