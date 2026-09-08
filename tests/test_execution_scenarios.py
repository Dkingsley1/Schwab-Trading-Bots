from core.execution_scenarios import gap_guard, run_execution_scenarios


def test_all_deterministic_fault_scenarios_pass() -> None:
    report = run_execution_scenarios()

    assert report["ok"] is True
    assert report["passed_count"] == report["scenario_count"] == 8


def test_gap_guard_rejects_large_price_dislocation() -> None:
    result = gap_guard(
        reference_price=100.0, observed_price=105.0, maximum_gap_bps=200.0
    )

    assert result["ok"] is True
    assert result["allow_execute"] is False
    assert result["reason"] == "gap_budget_exceeded"
