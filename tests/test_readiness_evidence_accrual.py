import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.ops import readiness_evidence_accrual as accrual


START = datetime(2026, 8, 6, 16, 0, tzinfo=timezone.utc)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _seed(project_root: Path, *, fills: int = 0) -> None:
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
    _write(health / "paper_execution_calibration_latest.json", {"independent_samples": fills})
    _write(
        health / "paper_performance_latest.json",
        {
            "post_cost_expectancy": {
                "sample_count": 10,
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
