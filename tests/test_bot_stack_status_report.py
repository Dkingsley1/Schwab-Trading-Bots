from datetime import datetime, timezone

from scripts.bot_stack_status_report import _classify_lane, _overall_health, _registry_summary


def test_classify_lane_prefers_role_specific_lanes() -> None:
    assert _classify_lane({"bot_role": "infrastructure_sub_bot", "bot_id": "brain_refinery_v56_meta_ranker"}) == "infrastructure"
    assert _classify_lane({"bot_role": "options_sub_bot", "bot_id": "brain_refinery_v27_term_structure_vol"}) == "options"
    assert _classify_lane({"bot_role": "signal_sub_bot", "bot_id": "brain_refinery_v48_position_1m_3m"}) == "swing"
    assert _classify_lane({"bot_role": "signal_sub_bot", "bot_id": "brain_refinery_v93_dividend_quality_compounder"}) == "long_term"


def test_registry_summary_groups_active_bots_by_lane() -> None:
    registry = {
        "sub_bots": [
            {"bot_id": "brain_refinery_v56_meta_ranker", "bot_role": "infrastructure_sub_bot", "active": True, "weight": 0.02, "quality_score": 0.87, "test_accuracy": 0.96, "reason": "canary"},
            {"bot_id": "brain_refinery_v27_term_structure_vol", "bot_role": "options_sub_bot", "active": True, "weight": 0.008, "quality_score": 0.22, "test_accuracy": 0.51, "reason": "options_floor"},
            {"bot_id": "brain_refinery_v48_position_1m_3m", "bot_role": "signal_sub_bot", "active": True, "weight": 0.01, "quality_score": 0.40, "test_accuracy": 0.53, "reason": "swing"},
            {"bot_id": "brain_refinery_v10_seasonal", "bot_role": "signal_sub_bot", "active": True, "weight": 0.013, "quality_score": 0.99, "test_accuracy": 0.94, "reason": "equities"},
        ]
    }

    summary = _registry_summary(registry, top_n=5)

    assert summary["lanes"]["infrastructure"]["active_count"] == 1
    assert summary["lanes"]["options"]["active_count"] == 1
    assert summary["lanes"]["swing"]["active_count"] == 1
    assert summary["lanes"]["equities"]["active_count"] == 1


def test_overall_health_treats_schwab_loops_as_off_session_on_weekends() -> None:
    now = datetime(2026, 3, 29, 16, 0, tzinfo=timezone.utc)
    payload = _overall_health(
        {"active": 35},
        {
            "schwab_conservative": {"latest_loop": None, "decision_lines": 0},
            "schwab_aggressive": {"latest_loop": None, "decision_lines": 0},
            "coinbase_crypto": {"latest_loop": None, "decision_lines": 50},
        },
        {"targets": [], "exists": False, "path": "/tmp/watchdog", "latest_timestamp_utc": None},
        now=now,
    )

    checks = {row["name"]: row for row in payload["checks"]}
    assert checks["live_shadow_loops"]["ok"] is True
    assert "schwab_conservative=off_session" in checks["live_shadow_loops"]["note"]
    assert checks["watchdog_schwab_live"]["ok"] is True
    assert checks["watchdog_schwab_live"]["note"] == "off_session"


def test_overall_health_falls_back_to_coinbase_activity_when_watchdog_missing() -> None:
    now = datetime(2026, 3, 30, 14, 0, tzinfo=timezone.utc)
    payload = _overall_health(
        {"active": 35},
        {
            "schwab_conservative": {"latest_loop": {"iter": 1}, "decision_lines": 10},
            "schwab_aggressive": {"latest_loop": {"iter": 2}, "decision_lines": 8},
            "coinbase_crypto": {"latest_loop": None, "decision_lines": 125},
        },
        {"targets": [], "exists": False, "path": "/tmp/watchdog", "latest_timestamp_utc": None},
        now=now,
    )

    checks = {row["name"]: row for row in payload["checks"]}
    assert checks["watchdog_coinbase_live"]["ok"] is True
    assert "fallback_activity=True" in checks["watchdog_coinbase_live"]["note"]
