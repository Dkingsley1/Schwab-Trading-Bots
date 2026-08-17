import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.champion_challenger_probation_guard as src


def test_probation_guard_stays_green_for_healthy_probation_lane() -> None:
    payload = src.build_payload(
        champion_registry={
            "champion": {"name": "alpha", "rollback_candidate": "beta"},
            "probation_candidates": [{"bot_id": "brain_refinery_v43_intraday_ultrafast_proxy", "profile": "intraday_aggressive"}],
        },
        master_registry={"sub_bots": []},
        paper_execution_calibration={"ok": True, "metrics": {"mae_bps": 3.0, "mean_bias_bps": 1.0}, "by_profile": {}},
        health_gates={"hard_gate_triggered": False, "gates": {"priority_shard_latency": False}, "summary": {"worst_priority_latency_multiplier": 1.0}},
        paper_performance={
            "sleeve_latest": [
                {"profile": "intraday_aggressive", "executions": 20, "win_rate": 0.61, "ending_net_pnl_total": 4.5}
            ]
        },
        max_calibration_mae_bps=35.0,
        max_calibration_bias_bps=12.0,
        max_latency_multiplier=1.25,
        min_profile_executions=10,
        min_profile_win_rate=0.45,
        min_profile_net_pnl=0.0,
    )

    assert payload["ok"] is True
    assert payload["rollback_required"] is False
    assert payload["probation_cohort_count"] == 1


def test_probation_guard_triggers_rollback_on_drift_and_weak_paper_execution() -> None:
    payload = src.build_payload(
        champion_registry={
            "champion": {"name": "alpha", "rollback_candidate": "beta"},
            "probation_candidates": [{"bot_id": "brain_refinery_v43_intraday_ultrafast_proxy", "profile": "intraday_aggressive"}],
        },
        master_registry={"sub_bots": []},
        paper_execution_calibration={
            "ok": False,
            "metrics": {"mae_bps": 41.0, "mean_bias_bps": 15.0},
            "by_profile": {"intraday_aggressive": {"mae_bps": 41.0, "mean_bias_bps": 15.0}},
        },
        health_gates={
            "hard_gate_triggered": True,
            "gates": {"priority_shard_latency": True},
            "summary": {"worst_priority_latency_multiplier": 1.8, "priority_shard_latency_failures": ["shadow_attribution"]},
        },
        paper_performance={
            "sleeve_latest": [
                {"profile": "intraday_aggressive", "executions": 18, "win_rate": 0.31, "ending_net_pnl_total": -6.0}
            ]
        },
        max_calibration_mae_bps=35.0,
        max_calibration_bias_bps=12.0,
        max_latency_multiplier=1.25,
        min_profile_executions=10,
        min_profile_win_rate=0.45,
        min_profile_net_pnl=0.0,
    )

    assert payload["ok"] is False
    assert payload["rollback_required"] is True
    assert payload["rollback_candidate"] == "beta"
    assert payload["failed_checks"] == [
        "calibration_drift",
        "latency_drift",
        "weak_paper_execution",
    ]


def test_probation_guard_ignores_inactive_probation_backlog_without_rollout_scope() -> None:
    payload = src.build_payload(
        champion_registry={},
        master_registry={
            "sub_bots": [
                {
                    "bot_id": "brain_refinery_v45_intraday_open_close_regimes",
                    "active": False,
                    "lifecycle_state": "probation",
                }
            ]
        },
        paper_execution_calibration={"ok": True, "metrics": {"mae_bps": 3.0, "mean_bias_bps": 1.0}, "by_profile": {}},
        health_gates={"hard_gate_triggered": False, "gates": {"priority_shard_latency": False}, "summary": {"worst_priority_latency_multiplier": 1.0}},
        paper_performance={
            "sleeve_latest": [
                {"profile": "intraday_aggressive", "executions": 20, "win_rate": 0.0, "ending_net_pnl_total": -6.0}
            ]
        },
        max_calibration_mae_bps=35.0,
        max_calibration_bias_bps=12.0,
        max_latency_multiplier=1.25,
        min_profile_executions=10,
        min_profile_win_rate=0.45,
        min_profile_net_pnl=0.0,
    )

    assert payload["ok"] is True
    assert payload["overall_status"] == "idle"
    assert payload["probation_cohort_count"] == 0
