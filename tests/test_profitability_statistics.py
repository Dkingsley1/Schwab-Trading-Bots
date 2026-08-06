from datetime import datetime, timedelta, timezone

from core.profitability_statistics import (
    benjamini_hochberg,
    clustered_post_cost_statistics,
    probability_of_backtest_overfitting,
)


def _rows(*, days: int, per_day: int, pnl: float = 1.0) -> list[dict]:
    start = datetime(2026, 7, 1, 14, 0, tzinfo=timezone.utc)
    rows = []
    for day in range(days):
        for index in range(per_day):
            rows.append(
                {
                    "timestamp_utc": (start + timedelta(days=day, minutes=index)).isoformat(),
                    "symbol": f"SYM{index % 5}",
                    "strategy": f"strategy-{index % 2}",
                    "regime": "risk_on" if day % 2 == 0 else "risk_off",
                    "post_cost_pnl_delta": pnl,
                    "post_cost_return_bps": pnl * 10.0,
                }
            )
    return rows


def test_trade_count_cannot_replace_independent_days() -> None:
    result = clustered_post_cost_statistics(_rows(days=1, per_day=500))

    assert result["sample_count"] == 500
    assert result["promotion_evidence_sufficient"] is False
    assert result["positive_clustered_lower_confidence_bound_95"] is False
    assert "minimum_independent_days_pending" in result["blockers"]


def test_clustered_evidence_can_pass_only_with_breadth() -> None:
    result = clustered_post_cost_statistics(
        _rows(days=30, per_day=5),
        minimum_effective_samples=20,
    )

    assert result["promotion_evidence_sufficient"] is True
    assert result["positive_clustered_lower_confidence_bound_95"] is True
    assert result["unique_symbol_count"] == 5


def test_benjamini_hochberg_adjusts_the_complete_family() -> None:
    result = benjamini_hochberg({"a": 0.001, "b": 0.02, "c": 0.20})
    by_id = {row["hypothesis_id"]: row for row in result["rows"]}

    assert by_id["a"]["passes_fdr"] is True
    assert by_id["c"]["passes_fdr"] is False
    assert by_id["a"]["q_value"] <= by_id["b"]["q_value"] <= by_id["c"]["q_value"]


def test_probability_of_backtest_overfitting_fails_closed_without_periods() -> None:
    unavailable = probability_of_backtest_overfitting({"one": [1.0] * 4, "two": [0.0] * 4})
    available = probability_of_backtest_overfitting(
        {
            "stable": [1.0] * 10,
            "unstable": [4.0, -4.0, 4.0, -4.0, 4.0, -4.0, 4.0, -4.0, 4.0, -4.0],
        }
    )

    assert unavailable["available"] is False
    assert available["available"] is True
    assert 0.0 <= available["pbo"] <= 1.0
