from core.portfolio_advisory import build_multi_period_advisory


def _sleeves(candidate_id: str = "candidate-1") -> list[dict[str, object]]:
    return [
        {
            "sleeve_id": name,
            "candidate_id": candidate_id,
            "qualified": True,
            "independent_fills": 35,
            "expected_return_bps": edge,
            "cost_bps": 1.0,
        }
        for name, edge in (
            ("dividend", 8.0),
            ("bond", 6.0),
            ("fx", 7.0),
            ("volatility", 5.0),
        )
    ]


def _covariance() -> dict[str, dict[str, float]]:
    names = ["dividend", "bond", "fx", "volatility"]
    return {
        left: {right: (1.0 if left == right else 0.1) for right in names}
        for left in names
    }


def test_portfolio_advisory_is_candidate_bound_constrained_and_orderless() -> None:
    report = build_multi_period_advisory(
        candidate_id="candidate-1",
        sleeves=_sleeves(),
        covariance=_covariance(),
        current_weights={},
    )

    assert report["ok"] is True
    assert report["qualified_sleeve_count"] == 4
    assert max(report["target_weights"].values()) <= 0.25
    assert all(step["turnover"] <= 0.20 + 1e-12 for step in report["steps"])
    assert report["advisory_only"] is True
    assert report["execution_authority"] is False


def test_portfolio_advisory_abstains_without_candidate_bound_evidence() -> None:
    rows = _sleeves()
    rows[0]["candidate_id"] = "stale-candidate"
    report = build_multi_period_advisory(
        candidate_id="candidate-1",
        sleeves=rows,
        covariance=_covariance(),
        current_weights={},
    )

    assert report["ok"] is False
    assert report["status"] == "abstain"
