from __future__ import annotations

from scripts.strategy_library_query import filter_families, filter_rows


def _row(
    strategy: str,
    *,
    sleeve: str = "equity_core",
    verdict: str = "insufficient_evidence",
    tier: str = "hot_catalog",
    relevance: str = "neutral",
    score: float | None = None,
) -> dict[str, object]:
    return {
        "strategy_name": strategy,
        "sleeve_id": sleeve,
        "library_tier": tier,
        "quality_assessment": {
            "verdict": verdict,
            "quality_score": score,
            "evidence_maturity_percent": 50.0 if score is not None else 0.0,
        },
        "regime_assessment": {"relevance": relevance},
    }


def test_query_filters_good_bad_sleeve_tier_and_regime() -> None:
    rows = [
        _row("good", verdict="validated_good", relevance="aligned", score=95.0),
        _row("bad", verdict="retirement_candidate", relevance="guarded", score=10.0),
        _row(
            "cold",
            sleeve="crypto_spot",
            verdict="cold_untested",
            tier="cold_research",
            relevance="aligned",
        ),
    ]

    assert [row["strategy_name"] for row in filter_rows(rows, good_only=True)] == [
        "good"
    ]
    assert [row["strategy_name"] for row in filter_rows(rows, bad_only=True)] == [
        "bad"
    ]
    assert [
        row["strategy_name"]
        for row in filter_rows(
            rows,
            sleeve="crypto-spot",
            tier="cold research",
            relevance="aligned",
        )
    ] == ["cold"]


def test_family_query_filters_parent_catalog_without_touching_variants() -> None:
    families = [
        {
            "family_id": "family::crypto_spot::funding_carry::v1",
            "family_name": "Funding Carry Research Family",
            "archetype": "funding_carry",
            "sleeve_id": "crypto_spot",
            "objective_class": "digital_asset_alpha",
            "child_variants": [{"strategy_id": "cold-1"}],
        },
        {
            "family_id": "sleeve::dividend::quality_dividend::v1",
            "family_name": "Quality Dividend",
            "archetype": "quality_dividend",
            "sleeve_id": "dividend",
            "objective_class": "income_total_return",
            "child_variants": [{"strategy_id": "hot-1"}],
        },
    ]

    selected = filter_families(
        families,
        sleeve="crypto-spot",
        objective="digital asset alpha",
        family="funding carry",
    )

    assert [row["family_id"] for row in selected] == [
        "family::crypto_spot::funding_carry::v1"
    ]
    assert selected[0]["child_variants"] == [{"strategy_id": "cold-1"}]
