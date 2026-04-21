from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.cohort_drift_baseline_guard as guard


def _history_snapshot(
    *,
    day_utc: str = "20260415",
    executions: int,
    win_rate: float,
    ending_net_pnl_total: float,
) -> dict[str, object]:
    return {
        "cohorts": [
            {
                "cohort_key": "conservative|conservative|unknown",
                "profile": "conservative",
                "family": "conservative",
                "timeframe": "conservative",
                "venue": "unknown",
                "day_utc": day_utc,
                "executions": executions,
                "win_rate": win_rate,
                "ending_net_pnl_total": ending_net_pnl_total,
                "mean_slippage_gap_bps": 0.0,
            }
        ]
    }


def test_partial_current_day_snapshot_is_deferred_instead_of_failed() -> None:
    payload, _snapshot = guard.build_payload(
        paper_performance={
            "sleeve_latest": [
                {
                    "profile": "conservative",
                    "day_utc": "20260416",
                    "current_day_available": True,
                    "data_status": "current",
                    "executions": 10,
                    "win_rate": 0.0,
                    "ending_net_pnl_total": -4.496662,
                    "tca_summary": {"mean_slippage_gap_bps": 0.0},
                }
            ]
        },
        history_rows=[
            _history_snapshot(day_utc="20260413", executions=210, win_rate=1.0, ending_net_pnl_total=52.658936),
            _history_snapshot(day_utc="20260414", executions=205, win_rate=1.0, ending_net_pnl_total=50.0),
            _history_snapshot(day_utc="20260415", executions=215, win_rate=1.0, ending_net_pnl_total=54.0),
        ],
        lookback_snapshots=7,
        min_history_points=3,
        min_executions=10,
        max_win_rate_drop=0.12,
        max_pnl_drop=5.0,
        max_slippage_gap_bps_increase=2.5,
        min_current_day_execution_completeness_ratio=0.5,
    )

    assert payload["ok"] is True
    assert payload["failed_checks"] == []
    assert payload["summary"]["deferred_current_day_cohort_count"] == 1
    assert payload["deferred_current_day_cohorts"][0]["profile"] == "conservative"
    assert payload["deferred_current_day_cohorts"][0]["baseline_executions"] == 210


def test_complete_current_day_snapshot_still_fails_when_drift_is_severe() -> None:
    payload, _snapshot = guard.build_payload(
        paper_performance={
            "sleeve_latest": [
                {
                    "profile": "conservative",
                    "day_utc": "20260416",
                    "current_day_available": True,
                    "data_status": "current",
                    "executions": 120,
                    "win_rate": 0.0,
                    "ending_realized_pnl_total": -66.3851,
                    "ending_unrealized_pnl_total": 0.0,
                    "ending_net_pnl_total": -4.496662,
                    "non_flat_strategy_count": 0,
                    "tca_summary": {"mean_slippage_gap_bps": 0.0},
                }
            ]
        },
        history_rows=[
            _history_snapshot(day_utc="20260413", executions=210, win_rate=1.0, ending_net_pnl_total=52.658936),
            _history_snapshot(day_utc="20260414", executions=205, win_rate=1.0, ending_net_pnl_total=50.0),
            _history_snapshot(day_utc="20260415", executions=215, win_rate=1.0, ending_net_pnl_total=54.0),
        ],
        lookback_snapshots=7,
        min_history_points=3,
        min_executions=10,
        max_win_rate_drop=0.12,
        max_pnl_drop=5.0,
        max_slippage_gap_bps_increase=2.5,
        min_current_day_execution_completeness_ratio=0.5,
    )

    assert payload["ok"] is False
    assert payload["failed_checks"] == ["cohort_drift_detected"]
    assert payload["summary"]["deferred_current_day_cohort_count"] == 0
    assert payload["drifted_cohorts"][0]["profile"] == "conservative"


def test_same_day_snapshots_do_not_pollute_the_baseline() -> None:
    payload, _snapshot = guard.build_payload(
        paper_performance={
            "sleeve_latest": [
                {
                    "profile": "conservative",
                    "day_utc": "20260416",
                    "current_day_available": True,
                    "data_status": "current",
                    "executions": 120,
                    "win_rate": 0.0,
                    "ending_realized_pnl_total": -66.3851,
                    "ending_unrealized_pnl_total": 0.0,
                    "ending_net_pnl_total": -66.3851,
                    "non_flat_strategy_count": 0,
                    "tca_summary": {"mean_slippage_gap_bps": 0.0},
                }
            ]
        },
        history_rows=[
            _history_snapshot(day_utc="20260415", executions=210, win_rate=1.0, ending_net_pnl_total=52.658936),
            _history_snapshot(day_utc="20260416", executions=10, win_rate=0.0, ending_net_pnl_total=-4.496662),
            _history_snapshot(day_utc="20260416", executions=14, win_rate=0.0, ending_net_pnl_total=-6.161179),
        ],
        lookback_snapshots=7,
        min_history_points=1,
        min_executions=10,
        max_win_rate_drop=0.12,
        max_pnl_drop=5.0,
        max_slippage_gap_bps_increase=2.5,
        min_current_day_execution_completeness_ratio=0.5,
    )

    assert payload["ok"] is False
    assert payload["drifted_cohorts"][0]["baseline"]["ending_net_pnl_total"] == 52.658936


def test_open_position_mark_to_market_current_day_is_deferred() -> None:
    payload, _snapshot = guard.build_payload(
        paper_performance={
            "sleeve_latest": [
                {
                    "profile": "conservative",
                    "day_utc": "20260416",
                    "current_day_available": True,
                    "data_status": "current",
                    "executions": 120,
                    "win_rate": 0.0,
                    "ending_realized_pnl_total": 0.0,
                    "ending_unrealized_pnl_total": -66.3851,
                    "ending_net_pnl_total": -66.3851,
                    "non_flat_strategy_count": 1,
                    "tca_summary": {"mean_slippage_gap_bps": 0.0},
                }
            ]
        },
        history_rows=[
            _history_snapshot(day_utc="20260413", executions=210, win_rate=1.0, ending_net_pnl_total=52.658936),
            _history_snapshot(day_utc="20260414", executions=205, win_rate=1.0, ending_net_pnl_total=50.0),
            _history_snapshot(day_utc="20260415", executions=215, win_rate=1.0, ending_net_pnl_total=54.0),
        ],
        lookback_snapshots=7,
        min_history_points=3,
        min_executions=10,
        max_win_rate_drop=0.12,
        max_pnl_drop=5.0,
        max_slippage_gap_bps_increase=2.5,
        min_current_day_execution_completeness_ratio=0.5,
    )

    assert payload["ok"] is True
    assert payload["failed_checks"] == []
    assert payload["summary"]["deferred_current_day_cohort_count"] == 1
    assert payload["deferred_current_day_cohorts"][0]["deferred_reason"] == "current_day_mark_to_market_open_positions"
