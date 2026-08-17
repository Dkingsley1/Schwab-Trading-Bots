from __future__ import annotations

import json
from pathlib import Path

from core.hierarchical_ensemble import aggregate_shadow_votes


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _policy() -> dict:
    return json.loads((PROJECT_ROOT / "config" / "bot_organization_v1.json").read_text(encoding="utf-8"))


def _assignment(sub_sleeve: str, cluster: str) -> dict:
    return {
        "sleeve_id": "equity",
        "sub_sleeve_id": sub_sleeve,
        "cohort_id": "daily_all_regimes",
        "correlation_cluster_id": cluster,
        "shadow_vote_eligible": True,
    }


def _regime_profile(direction: str, volatility: str, liquidity: str) -> dict:
    return {
        "profile_id": f"{direction}_{volatility}_{liquidity}",
        "scope": "market_signal",
        "axes": {
            "market_direction": {"values": [direction], "not_applicable": False},
            "volatility_state": {"values": [volatility], "not_applicable": False},
            "liquidity_state": {"values": [liquidity], "not_applicable": False},
        },
    }


def test_duplicate_correlated_votes_cannot_manufacture_weight() -> None:
    assignments = {
        "trend_a": _assignment("trend", "trend_cluster"),
        "trend_duplicate": _assignment("trend", "trend_cluster"),
        "reversion": _assignment("mean_reversion", "reversion_cluster"),
    }
    baseline = aggregate_shadow_votes(
        [
            {"vote_id": "a", "bot_id": "trend_a", "score": 0.8, "confidence": 1.0},
            {"vote_id": "b", "bot_id": "reversion", "score": 0.2, "confidence": 1.0},
        ],
        assignments,
        _policy(),
    )
    duplicated = aggregate_shadow_votes(
        [
            {"vote_id": "a", "bot_id": "trend_a", "score": 0.8, "confidence": 1.0},
            {"vote_id": "a2", "bot_id": "trend_duplicate", "score": 0.8, "confidence": 1.0},
            {"vote_id": "b", "bot_id": "reversion", "score": 0.2, "confidence": 1.0},
        ],
        assignments,
        _policy(),
    )

    assert duplicated["score"] == baseline["score"]
    assert duplicated["correlation_cluster_count"] == baseline["correlation_cluster_count"] == 2
    assert duplicated["authority"]["order_payload_created"] is False


def test_hierarchy_abstains_when_diversity_is_missing() -> None:
    assignments = {"trend": _assignment("trend", "trend_cluster")}

    result = aggregate_shadow_votes(
        [{"vote_id": "a", "bot_id": "trend", "score": 0.9, "confidence": 1.0}],
        assignments,
        _policy(),
    )

    assert result["action"] == "HOLD"
    assert result["reason"] == "insufficient_hierarchical_diversity"


def test_hierarchy_abstains_on_cross_cell_disagreement() -> None:
    assignments = {
        "trend": _assignment("trend", "trend_cluster"),
        "reversion": _assignment("mean_reversion", "reversion_cluster"),
    }

    result = aggregate_shadow_votes(
        [
            {"vote_id": "a", "bot_id": "trend", "score": 1.0, "confidence": 1.0},
            {"vote_id": "b", "bot_id": "reversion", "score": -1.0, "confidence": 1.0},
        ],
        assignments,
        _policy(),
    )

    assert result["action"] == "HOLD"
    assert result["reason"] == "cross_cell_disagreement_above_limit"
    assert result["authority"]["paper_execution_authority"] is False
    assert result["authority"]["live_execution_authority"] is False


def test_unknown_and_low_confidence_votes_are_excluded() -> None:
    assignments = {"known": _assignment("trend", "trend_cluster")}

    result = aggregate_shadow_votes(
        [
            {"vote_id": "unknown", "bot_id": "missing", "score": 0.5, "confidence": 1.0},
            {"vote_id": "weak", "bot_id": "known", "score": 0.5, "confidence": 0.1},
        ],
        assignments,
        _policy(),
    )

    assert result["action"] == "HOLD"
    assert result["reason"] == "no_eligible_votes"
    assert result["excluded_reasons"] == {
        "confidence_below_floor": 1,
        "missing_organization_assignment": 1,
    }


def test_regime_context_filters_only_shadow_evidence_and_fails_closed() -> None:
    matching = _assignment("trend", "trend_cluster")
    matching["regime_profile"] = _regime_profile("bull_trend", "normal", "normal")
    mismatch = _assignment("crisis", "crisis_cluster")
    mismatch["regime_profile"] = _regime_profile("bear_trend", "crisis", "dislocated")

    result = aggregate_shadow_votes(
        [
            {"vote_id": "match", "bot_id": "matching", "score": 0.8, "confidence": 1.0},
            {"vote_id": "miss", "bot_id": "mismatch", "score": -0.8, "confidence": 1.0},
        ],
        {"matching": matching, "mismatch": mismatch},
        _policy(),
        regime_context={
            "axes": {
                "market_direction": ["bull_trend"],
                "volatility_state": ["normal"],
                "liquidity_state": ["normal"],
            }
        },
    )

    assert result["regime_context_applied"] is True
    assert result["accepted_vote_count"] == 1
    assert result["regime_compatible_vote_count"] == 1
    assert result["regime_incompatible_vote_count"] == 1
    assert result["excluded_reasons"] == {"critical_regime_axis_mismatch": 1}
    assert result["authority"]["order_payload_created"] is False
    assert result["regime_contract"]["paper_execution_authority"] is False
    assert result["regime_contract"]["live_execution_authority"] is False
    assert result["regime_contract"]["metadata_access_version"] == (
        "regime_metadata_access_v1"
    )
    assert result["regime_metadata_access_ready_vote_count"] == 2
    assert result["regime_metadata_context_ready_vote_count"] == 2
    assert all(
        row["metadata_context_receipt_sha256"]
        for row in result["regime_compatibility_evidence"]
    )
