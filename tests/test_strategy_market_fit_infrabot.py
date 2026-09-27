from __future__ import annotations

import json
from pathlib import Path

from scripts.ops.strategy_market_fit_infrabot import build_payload

COHORT = [
    (
        "sleeve::stat_arb_market_neutral::research_residual_reversal_regime_conditioned::v1",
        "stat_arb_market_neutral",
        "residual_reversal",
        "mean_reversion",
    ),
    (
        "sleeve::international_macro::research_policy_surprise_regime_conditioned::v1",
        "international_macro",
        "policy_surprise",
        "event",
    ),
    (
        "sleeve::international_macro::research_carry_roll_down_stress_tested::v1",
        "international_macro",
        "carry_roll_down",
        "carry_value",
    ),
    (
        "sleeve::international_macro::research_liquidity_stress_cost_adjusted::v1",
        "international_macro",
        "liquidity_stress",
        "liquidity_execution",
    ),
    (
        "sleeve::variance_volatility_swaps::research_tail_convexity_cost_adjusted::v1",
        "variance_volatility_swaps",
        "tail_convexity",
        "volatility",
    ),
]


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _policy(expected_count: int) -> dict:
    return {
        "policy_id": "strategy_market_fit_test_v1",
        "sources": {
            "strategy_library_path": "governance/research/sleeve_strategy_library_latest.json",
            "strategy_policy_path": "config/sleeve_strategy_contracts_v1.json",
            "regime_path": "governance/health/regime_control_plane_latest.json",
            "alpha_inputs_path": "governance/research/alpha_concept_inputs_latest.json",
            "alpha_generation_path": "governance/health/alpha_generation_control_latest.json",
            "paper_performance_path": "governance/health/paper_performance_latest.json",
            "maximum_library_age_seconds": 86400,
            "maximum_regime_age_seconds": 3600,
            "maximum_alpha_input_age_seconds": 7200,
        },
        "catalog_contract": {
            "expected_strategy_count": expected_count,
            "batch_size": 4,
            "top_strategy_limit": 10,
            "top_family_limit": 10,
            "top_per_sleeve_limit": 2,
            "cache_when_source_signature_unchanged": True,
        },
        "challenger_cohort": {
            "mode": "existing_contract_shadow_observation_only",
            "maximum_slots": 5,
            "require_distinct_signal_families": True,
            "strategies": [
                {"strategy_id": strategy_id, "role": f"role_{index}"}
                for index, (strategy_id, _, _, _) in enumerate(COHORT, start=1)
            ],
        },
        "scoring": {
            "weights": {
                "regime_fit": 0.5,
                "candidate_forecast_evidence": 0.15,
                "quality_maturity": 0.15,
                "source_integrity": 0.1,
                "contract_integrity": 0.1,
            },
            "regime_scores": {
                "aligned": 1.0,
                "neutral": 0.6,
                "guarded": 0.2,
                "unknown": 0.0,
            },
            "quality_scores": {"cold_untested": 0.1},
            "minimum_sleeve_regime_observations": 20,
        },
        "proof_contract": {"market_fit_is_not_profitability": True},
        "authority": {
            "changes_active_action": False,
            "changes_position_size": False,
            "activates_cold_strategy": False,
            "submits_paper_orders": False,
            "submits_live_orders": False,
            "grants_promotion": False,
            "mutates_candidate": False,
        },
    }


def _strategy(
    strategy_id: str,
    sleeve_id: str,
    archetype: str,
    signal_family: str,
) -> dict:
    return {
        "strategy_id": strategy_id,
        "strategy_name": archetype.replace("_", " "),
        "sleeve_id": sleeve_id,
        "archetype": archetype,
        "signal_family": signal_family,
        "objective_class": "trade_candidate",
        "library_tier": "cold_research",
        "activation_state": "cold_untested",
        "conditioning_overlay": "regime_conditioned",
        "contract_receipt_sha256": f"receipt-{strategy_id}",
        "quality_assessment": {"verdict": "cold_untested"},
        "regime_assessment": {
            "current_regime": "risk_on",
            "relevance": "aligned",
        },
    }


def _root(tmp_path: Path, *, regime_status: str = "ready") -> Path:
    rows = [_strategy(*values) for values in COHORT]
    for index in range(5):
        rows.append(
            _strategy(
                f"sleeve::extra_{index}::research_trend_{index}::v1",
                f"extra_{index}",
                f"trend_{index}",
                "trend_momentum",
            )
        )
    _write(tmp_path / "config" / "strategy_market_fit_infrabot_v1.json", _policy(10))
    _write(tmp_path / "config" / "sleeve_strategy_contracts_v1.json", {})
    _write(
        tmp_path / "governance" / "research" / "sleeve_strategy_library_latest.json",
        {
            "timestamp_utc": "2026-08-31T12:00:00+00:00",
            "strategies": rows,
        },
    )
    _write(
        tmp_path / "governance" / "health" / "regime_control_plane_latest.json",
        {
            "timestamp_utc": "2026-08-31T12:04:00+00:00",
            "overall_status": regime_status,
            "regime_state": "risk_on",
        },
    )
    _write(
        tmp_path / "governance" / "research" / "alpha_concept_inputs_latest.json",
        {
            "timestamp_utc": "2026-08-31T12:03:00+00:00",
            "candidate_id": "pc-test-g1",
            "candidate_binding": {"candidate_id": "pc-test-g1", "bound": True},
            "diagnostics": {
                "regime_and_decay_research_routing": {"routes": []},
                "capacity_truth": {"capacity_remains_blocked_without_direct_adv": True},
            },
        },
    )
    _write(
        tmp_path / "governance" / "health" / "alpha_generation_control_latest.json",
        {
            "candidate_binding": {"candidate_id": "pc-test-g1"},
            "strategy_expansion_freeze": {"active": True},
        },
    )
    _write(
        tmp_path / "governance" / "health" / "paper_performance_latest.json",
        {"profitability_evidence_window": {"candidate_id": "pc-test-g1"}},
    )
    return tmp_path


def test_complete_catalog_is_checked_and_cohort_is_shadow_only(tmp_path: Path) -> None:
    root = _root(tmp_path)

    payload = build_payload(
        root,
        config_path=root / "config" / "strategy_market_fit_infrabot_v1.json",
        generated_at_utc="2026-08-31T12:05:00+00:00",
        use_cache=False,
    )

    catalog = payload["catalog_contract"]
    cohort = payload["challenger_cohort"]
    assert payload["overall_status"] == "ready"
    assert catalog["all_strategies_checked"] is True
    assert catalog["evaluated_strategy_count"] == 10
    assert catalog["batch_count"] == 3
    assert sum(row["strategy_count"] for row in payload["batch_receipts"]) == 10
    assert cohort["slot_count"] == 5
    assert cohort["distinct_signal_family_count"] == 5
    assert {row["cohort_state"] for row in cohort["strategies"]} == {"shadow_observe"}
    assert all(not value for value in payload["authority_contract"].values())
    assert all(not row["paper_order_authority"] for row in cohort["strategies"])
    assert all(not row["live_order_authority"] for row in cohort["strategies"])
    assert payload["proven_working_strategy_count"] == 0
    assert all(
        not row["profitability_claim_allowed"]
        for row in payload["top_strategy_rankings"]
    )


def test_thin_regime_queues_challengers_without_blocking_full_scan(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path, regime_status="thin")

    payload = build_payload(
        root,
        config_path=root / "config" / "strategy_market_fit_infrabot_v1.json",
        generated_at_utc="2026-08-31T12:05:00+00:00",
        use_cache=False,
    )

    assert payload["overall_status"] == "guarded"
    assert payload["catalog_contract"]["all_strategies_checked"] is True
    assert payload["current_regime"]["screening_allowed"] is True
    assert payload["current_regime"]["trusted_for_shadow_admission"] is False
    assert {
        row["cohort_state"] for row in payload["challenger_cohort"]["strategies"]
    } == {"queued_regime_source_not_ready"}


def test_unchanged_sources_reuse_the_verified_full_scan(tmp_path: Path) -> None:
    root = _root(tmp_path)
    prior_path = root / "governance" / "health" / "market_fit_prior.json"
    first = build_payload(
        root,
        config_path=root / "config" / "strategy_market_fit_infrabot_v1.json",
        generated_at_utc="2026-08-31T12:05:00+00:00",
        previous_path=prior_path,
        use_cache=True,
    )
    _write(prior_path, first)

    second = build_payload(
        root,
        config_path=root / "config" / "strategy_market_fit_infrabot_v1.json",
        generated_at_utc="2026-08-31T12:06:00+00:00",
        previous_path=prior_path,
        use_cache=True,
    )

    assert second["evaluation_mode"] == "cache_hit"
    assert second["cache"]["hit"] is True
    assert second["full_scan_receipt_sha256"] == first["full_scan_receipt_sha256"]

    stale = build_payload(
        root,
        config_path=root / "config" / "strategy_market_fit_infrabot_v1.json",
        generated_at_utc="2026-08-31T14:10:00+00:00",
        previous_path=prior_path,
        use_cache=True,
    )

    assert stale["evaluation_mode"] == "full_scan"
    assert stale["overall_status"] == "guarded"
    assert stale["current_regime"]["fresh"] is False
    assert stale["current_regime"]["trusted_for_shadow_admission"] is False


def test_missing_strategy_fails_catalog_and_cohort_closed(tmp_path: Path) -> None:
    root = _root(tmp_path)
    library_path = (
        root / "governance" / "research" / "sleeve_strategy_library_latest.json"
    )
    library = json.loads(library_path.read_text(encoding="utf-8"))
    library["strategies"] = library["strategies"][1:]
    _write(library_path, library)

    payload = build_payload(
        root,
        config_path=root / "config" / "strategy_market_fit_infrabot_v1.json",
        generated_at_utc="2026-08-31T12:05:00+00:00",
        use_cache=False,
    )

    assert payload["overall_status"] == "blocked"
    assert payload["catalog_contract"]["all_strategies_checked"] is False
    assert "strategy_count_mismatch:9/10" in payload["catalog_contract"]["blockers"]
    assert (
        "challenger_cohort_contract_incomplete"
        in payload["catalog_contract"]["blockers"]
    )
    assert payload["challenger_cohort"]["status"] == "blocked"


def test_working_now_requires_bound_robust_post_cost_and_capacity_evidence(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    library_path = (
        root / "governance" / "research" / "sleeve_strategy_library_latest.json"
    )
    library = json.loads(library_path.read_text(encoding="utf-8"))
    library["strategies"][0]["quality_assessment"] = {
        "verdict": "validated_good",
        "lower_confidence_bound_95_post_cost_return_bps": 1.25,
    }
    _write(library_path, library)
    alpha_path = root / "governance" / "research" / "alpha_concept_inputs_latest.json"
    alpha = json.loads(alpha_path.read_text(encoding="utf-8"))
    alpha["diagnostics"]["capacity_truth"] = {
        "capacity_remains_blocked_without_direct_adv": False
    }
    _write(alpha_path, alpha)

    payload = build_payload(
        root,
        config_path=root / "config" / "strategy_market_fit_infrabot_v1.json",
        generated_at_utc="2026-08-31T12:05:00+00:00",
        use_cache=False,
    )

    assert payload["proven_working_strategy_count"] == 1
    proven = [
        row for row in payload["top_strategy_rankings"] if row["proven_working_now"]
    ]
    assert len(proven) == 1
    assert proven[0]["profitability_claim_allowed"] is True
    assert proven[0]["post_cost_lower_confidence_bound_bps"] == 1.25
