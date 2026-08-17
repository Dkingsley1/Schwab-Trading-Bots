import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.ops import readiness_evidence_accrual as accrual


START = datetime(2026, 8, 6, 16, 0, tzinfo=timezone.utc)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _seed(
    project_root: Path,
    *,
    fills: int = 0,
    acquisition_status: str = "collecting",
    post_cost_samples: int = 10,
    performance_cutoff: str = "",
) -> None:
    _write(
        project_root / "governance" / "runtime" / "production_candidate_state.json",
        {
            "candidate_id": "pc-test-g1",
            "generation": 1,
            "scope_windows_started_utc": {"promotion": START.isoformat(), "operations": START.isoformat()},
        },
    )
    health = project_root / "governance" / "health"
    _write(health / "process_watchdog_latest.json", {"overall_status": "ready", "active_process_count": 4})
    _write(
        health / "independent_fill_evidence_acquisition_latest.json",
        {"overall_status": acquisition_status, "rows_scanned": fills},
    )
    _write(health / "paper_execution_calibration_latest.json", {"independent_samples": fills})
    _write(
        health / "paper_performance_latest.json",
        {
            "profitability_evidence_window": {"candidate_cutoff_utc": performance_cutoff},
            "post_cost_expectancy": {
                "sample_count": post_cost_samples,
                "robust_statistics": {
                    "unique_day_count": 1,
                    "unique_symbol_count": 2,
                    "effective_sample_size": 1,
                    "thresholds": {"minimum_samples": 30, "minimum_days": 7, "minimum_symbols": 5, "minimum_effective_samples": 20},
                },
            }
        },
    )
    _write(
        health / "promotion_quality_gate_latest.json",
        {"details": {"promotion": {"considered_bots": 0, "min_considered_bots": 4}, "promotion_candidate_ids": []}},
    )
    _write(
        health / "canary_rollout_latest.json",
        {
            "canary_samples": 20,
            "baseline_samples": 20,
            "thresholds": {"minimum_samples_per_cohort": 400, "minimum_independent_days": 3, "minimum_effective_samples": 50},
            "canary_statistics": {"unique_day_count": 1, "effective_sample_size": 1},
        },
    )
    _write(
        health / "paper_profitability_control_latest.json",
        {"a_plus_target_contract": {"thresholds": {"min_net_pnl": 50000}, "current": {"net_pnl": -1000}}},
    )


def _by_id(payload: dict) -> dict[str, dict]:
    return {row["metric_id"]: row for row in payload["metrics"]}


def test_eta_requires_observed_positive_rate(tmp_path: Path) -> None:
    _seed(tmp_path, fills=0)
    first = accrual.build_payload(tmp_path, apply=True, now=START + timedelta(hours=1))
    assert _by_id(first)["independent_fills"]["eta_available"] is False

    _seed(tmp_path, fills=10)
    second = accrual.build_payload(tmp_path, apply=True, now=START + timedelta(hours=3))
    fill_metric = _by_id(second)["independent_fills"]

    assert fill_metric["rate_per_hour"] == 5.0
    assert fill_metric["eta_hours"] == 18.0
    assert fill_metric["stalled"] is False


def test_active_collection_marks_unchanged_evidence_stalled(tmp_path: Path) -> None:
    _seed(tmp_path, fills=0)
    accrual.build_payload(tmp_path, apply=True, now=START + timedelta(hours=1), stall_hours=6)
    payload = accrual.build_payload(tmp_path, apply=True, now=START + timedelta(hours=8), stall_hours=6)

    assert payload["overall_status"] == "stalled"
    assert "independent_fills" in payload["stalled_metric_ids"]
    assert _by_id(payload)["independent_fills"]["eta_available"] is False


def test_candidate_change_resets_stall_history(tmp_path: Path) -> None:
    _seed(tmp_path, fills=0)
    accrual.build_payload(tmp_path, apply=True, now=START + timedelta(hours=1), stall_hours=1)
    state_path = tmp_path / "governance" / "runtime" / "production_candidate_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["candidate_id"] = "pc-test-g2"
    state["generation"] = 2
    state["scope_windows_started_utc"] = {"promotion": (START + timedelta(hours=8)).isoformat()}
    state_path.write_text(json.dumps(state), encoding="utf-8")

    payload = accrual.build_payload(tmp_path, apply=True, now=START + timedelta(hours=8), stall_hours=1)

    assert payload["stalled_metric_ids"] == []
    assert _by_id(payload)["independent_fills"]["delta_since_previous"] is None


def test_missing_event_driven_source_waits_without_false_stall(tmp_path: Path) -> None:
    _seed(tmp_path, fills=0, acquisition_status="waiting_for_source")
    accrual.build_payload(tmp_path, apply=True, now=START + timedelta(hours=1), stall_hours=1)
    payload = accrual.build_payload(tmp_path, apply=True, now=START + timedelta(hours=12), stall_hours=1)
    fill_metric = _by_id(payload)["independent_fills"]

    assert fill_metric["stalled"] is False
    assert fill_metric["accrual_state"] == "waiting_precondition"
    assert "independent_fills" in payload["waiting_precondition_metric_ids"]


def test_daily_evidence_uses_daily_cadence_instead_of_global_stall_window(tmp_path: Path) -> None:
    _seed(tmp_path)
    accrual.build_payload(tmp_path, apply=True, now=START + timedelta(hours=1), stall_hours=1)
    payload = accrual.build_payload(tmp_path, apply=True, now=START + timedelta(hours=8), stall_hours=1)
    day_metric = _by_id(payload)["post_cost_days"]

    assert day_metric["stall_threshold_hours"] == 30.0
    assert day_metric["stalled"] is False


def test_same_candidate_counter_regression_fails_closed(tmp_path: Path) -> None:
    _seed(tmp_path, fills=10)
    accrual.build_payload(tmp_path, apply=True, now=START + timedelta(hours=1))
    _seed(tmp_path, fills=5)
    payload = accrual.build_payload(tmp_path, apply=True, now=START + timedelta(hours=2))

    assert payload["overall_status"] == "regressed"
    assert payload["ok"] is False
    assert payload["regressed_metric_ids"] == ["independent_fills"]
    assert _by_id(payload)["independent_fills"]["accrual_state"] == "counter_regression"


def test_schedule_resume_restarts_the_stall_clock(tmp_path: Path) -> None:
    _seed(tmp_path)
    accrual.build_payload(tmp_path, apply=True, now=START + timedelta(hours=12), stall_hours=1)
    payload = accrual.build_payload(tmp_path, apply=True, now=START + timedelta(hours=17), stall_hours=1)
    canary_metric = _by_id(payload)["canary_samples"]

    assert canary_metric["schedule_resumed"] is True
    assert canary_metric["unchanged_hours"] == 0.0
    assert canary_metric["stalled"] is False


def test_producer_window_rebind_resets_metric_history(tmp_path: Path) -> None:
    _seed(tmp_path, post_cost_samples=10, performance_cutoff="2026-08-06T16:00:00+00:00")
    accrual.build_payload(tmp_path, apply=True, now=START + timedelta(hours=1))
    _seed(tmp_path, post_cost_samples=0, performance_cutoff="2026-08-06T18:00:00+00:00")
    payload = accrual.build_payload(tmp_path, apply=True, now=START + timedelta(hours=2))
    sample_metric = _by_id(payload)["post_cost_samples"]

    assert sample_metric["producer_binding_changed"] is True
    assert sample_metric["delta_since_previous"] is None
    assert sample_metric["regressed"] is False
    assert "post_cost_samples" not in payload["regressed_metric_ids"]


def test_unaccepted_candidate_drift_receives_zero_soak_credit(tmp_path: Path) -> None:
    _seed(tmp_path)
    _write(
        tmp_path / "governance" / "health" / "production_excellence_control_latest.json",
        {
            "candidate": {
                "candidate_id": "pc-test-g1",
                "candidate_ready": False,
                "candidate_drift": True,
            }
        },
    )

    payload = accrual.build_payload(tmp_path, now=START + timedelta(hours=48))

    assert payload["overall_status"] == "blocked"
    assert payload["ok"] is False
    assert payload["candidate_binding"]["credit_eligible"] is False
    soak_metric = _by_id(payload)["soak_elapsed_hours"]
    assert soak_metric["current"] == 0.0
    assert soak_metric["accrual_state"] == "waiting_precondition"
    assert soak_metric["producer"]["reason"] == "candidate_acceptance_required_before_soak_credit"


def test_profitability_accrual_tracks_strict_promotion_evidence_targets(tmp_path: Path) -> None:
    _seed(tmp_path, post_cost_samples=357)
    health = tmp_path / "governance" / "health"
    research = tmp_path / "governance" / "research"
    config = tmp_path / "config"
    _write(
        health / "paper_performance_latest.json",
        {
            "post_cost_expectancy": {
                "sample_count": 357,
                "robust_statistics": {
                    "unique_day_count": 3,
                    "unique_symbol_count": 41,
                    "unique_regime_count": 0,
                    "effective_sample_size": 255.0,
                    "positive_clustered_lower_confidence_bound_95": False,
                },
            }
        },
    )
    _write(
        config / "profitability_evidence_firewall_v1.json",
        {
            "strict_graduation": {
                "minimum_post_cost_samples": 200,
                "minimum_independent_days": 30,
                "minimum_symbols": 10,
                "minimum_effective_samples": 100,
                "minimum_regimes": 3,
                "minimum_profitable_sleeves": 4,
            }
        },
    )
    _write(
        health / "profitability_evidence_firewall_latest.json",
        {
            "overall_status": "ready_with_evidence_debt",
            "allocation_proposal": {"qualified_sleeve_count": 0},
            "evidence_epoch_contract": {"ready": True},
            "baseline_controls": [{"control_id": "06_stressed_post_cost_expectancy", "evidence_ready": False}],
            "controls": [{"control_id": "h09_tail_concentration", "evidence_ready": False}],
        },
    )
    _write(health / "profitability_independent_validator_latest.json", {"risk_of_ruin": {"day_count": 3, "thresholds": {"minimum_days": 30}}})
    _write(research / "profitability_benchmark_hurdle_latest.json", {"common_day_count": 0, "thresholds": {"minimum_common_days": 30}})

    metrics = _by_id(accrual.build_payload(tmp_path, now=START + timedelta(hours=1)))

    assert metrics["strict_post_cost_samples"]["complete"] is True
    assert metrics["strict_independent_days"]["current"] == 3.0
    assert metrics["strict_independent_days"]["target"] == 30.0
    assert metrics["strict_regime_breadth"]["target"] == 3.0
    assert metrics["strict_profitable_sleeves"]["target"] == 4.0
    assert metrics["benchmark_common_days"]["target"] == 30.0
    assert metrics["risk_of_ruin_days"]["target"] == 30.0
    assert metrics["profitability_epoch_coherence"]["complete"] is True


def test_stale_candidate_artifacts_are_quarantined_without_false_counter_regression(tmp_path: Path) -> None:
    _seed(tmp_path)
    health = tmp_path / "governance" / "health"
    _write(
        health / "profitability_independent_validator_latest.json",
        {
            "candidate_binding": {
                "candidate_id": "pc-test-g0",
                "generation": 0,
                "cutoff_utc": (START - timedelta(days=1)).isoformat(),
                "bound": True,
            },
            "risk_of_ruin": {"day_count": 3, "thresholds": {"minimum_days": 30}},
        },
    )
    _write(
        health / "canary_rollout_latest.json",
        {
            "candidate_binding": {
                "candidate_id": "pc-test-g0",
                "generation": 0,
                "cutoff_utc": (START - timedelta(days=1)).isoformat(),
                "bound": True,
            },
            "canary_samples": 5000,
            "baseline_samples": 5000,
            "thresholds": {"minimum_samples_per_cohort": 400},
        },
    )

    first = accrual.build_payload(tmp_path, apply=True, now=START + timedelta(hours=1))
    first_metrics = _by_id(first)
    assert first_metrics["risk_of_ruin_days"]["current"] == 0.0
    assert first_metrics["risk_of_ruin_days"]["accrual_state"] == "waiting_precondition"
    assert first_metrics["canary_samples"]["current"] == 0.0
    assert first_metrics["canary_samples"]["producer"]["reason"] == "canary_rollout_canary_candidate_epoch_stale"

    _write(
        health / "profitability_independent_validator_latest.json",
        {
            "candidate_binding": {
                "candidate_id": "pc-test-g1",
                "generation": 1,
                "cutoff_utc": START.isoformat(),
                "bound": True,
            },
            "risk_of_ruin": {"day_count": 0, "thresholds": {"minimum_days": 30}},
        },
    )
    second = accrual.build_payload(tmp_path, apply=True, now=START + timedelta(hours=2))
    risk_metric = _by_id(second)["risk_of_ruin_days"]

    assert risk_metric["producer_binding_changed"] is True
    assert risk_metric["delta_since_previous"] is None
    assert risk_metric["regressed"] is False
    assert "risk_of_ruin_days" not in second["regressed_metric_ids"]
