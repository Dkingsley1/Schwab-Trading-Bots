from core.strategy_validity import (
    default_validity_contract,
    future_suffix_invariance,
    recursive_warmup_stability,
    scan_source_text,
)
from scripts.strategy_validity_control import build_payload


def _trailing(values: list[float]) -> list[float]:
    return [sum(values[max(0, i - 2) : i + 1]) for i in range(len(values))]


def _future_leaking(values: list[float]) -> list[float]:
    return [values[min(i + 1, len(values) - 1)] for i in range(len(values))]


def test_static_scan_rejects_high_confidence_future_access_patterns() -> None:
    source = """
shifted = frame.price.shift(-1)
centered = frame.price.rolling(5, center=True).mean()
filled = frame.price.bfill()
joined = merge_asof(left, right, direction='forward')
"""
    rule_ids = {row["rule_id"] for row in scan_source_text(source)}

    assert rule_ids == {
        "future_period_access",
        "centered_rolling_window",
        "backward_fill",
        "forward_asof_join",
    }


def test_future_suffix_invariance_detects_leakage_and_accepts_trailing_logic() -> None:
    values = [float(index) for index in range(20)]

    assert future_suffix_invariance(_trailing, values)["ok"] is True
    assert future_suffix_invariance(_future_leaking, values)["ok"] is False


def test_recursive_warmup_probe_detects_unstable_history_dependency() -> None:
    values = [float(index) for index in range(1, 65)]
    stable = recursive_warmup_stability(
        _trailing, values, startup_lengths=(16, 32, 64), comparison_points=1
    )
    unstable = recursive_warmup_stability(
        lambda rows: [sum(rows[: i + 1]) / (i + 1) for i in range(len(rows))],
        values,
        startup_lengths=(16, 32, 64),
        comparison_points=1,
    )

    assert stable["ok"] is True
    assert unstable["ok"] is False
    assert (
        default_validity_contract()["failure_behavior"]
        == "block_candidate_promotion_and_live_execution"
    )


def test_all_12000_strategy_contracts_inherit_the_validity_contract() -> None:
    payload = build_payload()

    assert payload["ok"] is True
    assert payload["strategy_contract_coverage"] == {
        "strategy_count": 12000,
        "validity_ready_count": 12000,
        "complete": True,
    }
