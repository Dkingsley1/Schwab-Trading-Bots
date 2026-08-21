from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.institutional_decision_flow import evaluate_execution_policy_guard
from core.sleeve_strategy_specialization import (
    FORBIDDEN_AUTHORITY,
    attach_strategy_specialization,
    extract_current_regime,
    load_policy,
    materialize_strategy_contracts,
    materialize_strategy_library,
    rank_counterfactual_strategies,
    resolve_runtime_regime_context,
    resolve_strategy_contract,
    strategy_regime_assessment,
    strategy_specialization_guard_reasons,
)
from scripts.paper_performance_report import (
    _candidate_strategy_post_cost_daily_series,
    _strategy_of,
    _strategy_post_cost_latest,
)
from scripts.sleeve_strategy_specialization_report import (
    _lifecycle,
    _quality_assessment,
    build_family_payload,
    build_library_payload,
    build_payload,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _bound_performance(strategy_rows: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "timestamp_utc": "2026-08-18T20:00:00+00:00",
        "profitability_evidence_window": {
            "candidate_id": "pc-test-g1",
            "candidate_generation": 1,
            "candidate_cutoff_utc": "2026-08-01T00:00:00+00:00",
            "evidence_through_utc": "2026-08-18T20:00:00+00:00",
            "candidate_filter_active": True,
            "candidate_binding_required": True,
            "candidate_binding_mismatch_rows_excluded": 0,
        },
        "strategy_latest": strategy_rows or [],
    }


def test_all_active_catalog_strategies_and_additions_have_complete_unique_contracts() -> None:
    policy = load_policy()
    contracts = materialize_strategy_contracts(policy=policy)

    assert len(contracts) == 879
    assert len({row["sleeve_id"] for row in contracts.values()}) == 111
    assert sum(row["source_kind"] == "catalog" for row in contracts.values()) == 771
    assert sum(row["source_kind"] == "curated_addition" for row in contracts.values()) == 108
    assert all(row["contract_complete"] for row in contracts.values())
    assert len({row["contract_receipt_sha256"] for row in contracts.values()}) == len(contracts)
    assert not any(
        bool(row["authority"].get(key, False))
        for row in contracts.values()
        for key in FORBIDDEN_AUTHORITY
    )
    assert all(row["strategy_definition"] for row in contracts.values())
    assert all(row["library_tier"] == "hot_catalog" for row in contracts.values())


def test_full_library_is_exactly_12000_balanced_and_runtime_cold() -> None:
    policy = load_policy()
    library = materialize_strategy_library(policy=policy)
    counts: dict[str, int] = {}
    for row in library.values():
        sleeve_id = str(row["sleeve_id"])
        counts[sleeve_id] = counts.get(sleeve_id, 0) + 1

    assert len(library) == 12000
    assert len(counts) == 111
    assert min(counts.values()) == 108
    assert max(counts.values()) == 109
    assert sum(row["library_tier"] == "cold_research" for row in library.values()) == 11121
    assert all(row["contract_complete"] for row in library.values())
    assert not any(
        bool(row["authority"].get(key, False))
        for row in library.values()
        for key in FORBIDDEN_AUTHORITY
    )


def test_regime_alignment_is_point_in_time_and_cannot_activate_on_thin_source() -> None:
    policy = load_policy()
    contract = next(
        row
        for row in materialize_strategy_library(policy=policy).values()
        if row["sleeve_id"] == "swing_aggressive"
        and row["library_tier"] == "cold_research"
        and "trend" in row["taxonomy_groups"]
    )
    features = {"market_regime_snapshot": {"regime_state": "risk_on_trend"}}

    assert extract_current_regime(features) == "risk_on_trend"
    ready = strategy_regime_assessment(
        contract,
        "risk_on_trend",
        policy=policy,
        source_ready=True,
        source_status="ready",
    )
    thin = strategy_regime_assessment(
        contract,
        "risk_on_trend",
        policy=policy,
        source_ready=True,
        source_status="thin",
    )

    assert ready["relevance"] == "aligned"
    assert ready["cold_activation_eligible"] is True
    assert ready["execution_alignment_ready"] is True
    assert thin["relevance"] == "aligned"
    assert thin["cold_activation_eligible"] is False
    assert thin["execution_alignment_ready"] is False

    guarded_metadata = {
        "strategy_specialization": {
            "contract_complete": True,
            "selected_strategy_id": "strategy-test",
            "contract_receipt_sha256": "receipt-test",
            "contract_receipt": {
                "strategy_id": "strategy-test",
                "contract_receipt_sha256": "receipt-test",
                "candidate_id": "pc-test-g1",
            },
            "regime_assessment": thin,
            "action_or_quantity_mutated": False,
            "authority": {},
        }
    }
    assert (
        "strategy_current_regime_source_not_execution_ready"
        in strategy_specialization_guard_reasons(
            guarded_metadata,
            require_candidate=True,
            require_regime_alignment=True,
        )
    )


def test_runtime_regime_context_falls_back_to_fresh_control_plane_and_expires(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 8, 18, 20, 0, tzinfo=timezone.utc)
    source = tmp_path / "regime.json"
    source.write_text(
        json.dumps(
            {
                "timestamp_utc": now.isoformat(),
                "overall_status": "thin",
                "regime_state": "mixed_transition",
            }
        ),
        encoding="utf-8",
    )
    policy = deepcopy(load_policy())
    policy["regime_adaptation"]["source_path"] = str(source)
    policy["regime_adaptation"]["maximum_source_age_seconds"] = 3600

    fresh = resolve_runtime_regime_context({}, policy=policy, now_utc=now)
    stale = resolve_runtime_regime_context(
        {},
        policy=policy,
        now_utc=now + timedelta(seconds=3601),
    )

    assert fresh["current_regime"] == "mixed_transition"
    assert fresh["source_status"] == "thin"
    assert fresh["source_ready"] is True
    assert fresh["fresh"] is True
    assert stale["source_ready"] is False
    assert stale["fresh"] is False


def test_contract_materialization_and_counterfactual_ranking_are_deterministic() -> None:
    policy = load_policy()
    first = materialize_strategy_contracts(policy=policy)
    second = materialize_strategy_contracts(policy=deepcopy(policy))
    features = {"momentum_20d": 0.8, "yield": 0.04, "spread_bps": 2.0}

    assert first == second
    ranking = rank_counterfactual_strategies("dividend", features, policy=policy)
    assert ranking == rank_counterfactual_strategies(
        "dividend", features, policy=policy
    )
    assert all(row["strategy_id"] in first for row in ranking)
    assert all(first[row["strategy_id"]]["library_tier"] == "hot_catalog" for row in ranking)


def test_broad_master_is_not_falsely_credited_to_a_named_strategy() -> None:
    policy = load_policy()
    contract = resolve_strategy_contract("dividend", "grand_master_bot", policy=policy)

    assert contract["strategy_id"] == "sleeve::dividend_income::ensemble_champion::v1"
    assert contract["source_kind"] == "synthetic_ensemble"
    assert contract["strategy_name"] not in {
        "quality_dividend",
        "dividend_quality_value_composite",
    }


def test_default_crypto_context_resolves_to_crypto_spot_contract() -> None:
    metadata = attach_strategy_specialization(
        {
            "source_broker": "coinbase",
            "shadow_domain": "crypto",
            "production_candidate_id": "pc-test-g1",
        },
        profile="default",
        raw_strategy="grand_master_bot",
        features={"funding_rate_norm": 0.4},
        action="HOLD",
        quantity=0.0,
    )

    receipt = metadata["strategy_specialization"]
    assert receipt["profile"] == "crypto_spot"
    assert receipt["selected_strategy_id"] == "sleeve::crypto_spot::ensemble_champion::v1"
    assert receipt["objective_class"] == "digital_asset_alpha"


def test_objectives_do_not_force_fake_profit_on_control_hedge_or_cash_sleeves() -> None:
    policy = load_policy()

    infrastructure = resolve_strategy_contract(
        "infrastructure_risk", "margin_guard", policy=policy
    )
    hedge = resolve_strategy_contract(
        "short_bias_hedge", "beta_hedge_efficiency", policy=policy
    )
    cash = resolve_strategy_contract(
        "cash_rotation_tactical", "trend_to_cash", policy=policy
    )

    assert infrastructure["objective_class"] == "control_only"
    assert infrastructure["risk_budget"] == "zero_trading_risk"
    assert infrastructure["shorting_policy"] == "forbidden"
    assert hedge["objective_class"] == "hedge_utility"
    assert hedge["objective_scorecard"]["primary_metric"] == "drawdown_reduction_per_carry_cost"
    assert cash["objective_class"] == "capital_preservation"
    assert cash["objective_scorecard"]["primary_metric"] == "risk_adjusted_opportunity_cost_bps"


def test_attachment_preserves_action_quantity_and_has_no_authority() -> None:
    original = {
        "source_profile": "swing_aggressive",
        "production_candidate_id": "pc-test-g1",
    }
    attached = attach_strategy_specialization(
        original,
        profile="swing_aggressive",
        raw_strategy="grand_master_bot",
        features={"momentum_20d": 0.7, "volatility": 0.2},
        action="BUY",
        quantity=2.5,
    )
    specialization = attached["strategy_specialization"]

    assert original == {
        "source_profile": "swing_aggressive",
        "production_candidate_id": "pc-test-g1",
    }
    assert specialization["action_observed"] == "BUY"
    assert specialization["quantity_observed"] == 2.5
    assert specialization["action_or_quantity_mutated"] is False
    assert not any(specialization["authority"].values())
    assert len(specialization["counterfactual_ranking"]) == 3


def test_live_requires_contract_and_candidate_while_legacy_paper_stays_compatible() -> None:
    intent = {
        "action": "BUY",
        "quantity": 1.0,
        "strategy": "legacy",
        "metadata": {},
    }
    paper = evaluate_execution_policy_guard(intent=intent, target_mode="paper")
    live = evaluate_execution_policy_guard(intent=intent, target_mode="live")

    assert paper["status"] == "legacy_paper_passthrough"
    assert paper["allow_execute"] is True
    assert "strategy_specialization:strategy_specialization_missing" in live["reasons"]

    attached = attach_strategy_specialization(
        {"source_profile": "default", "production_candidate_id": "pc-test-g1"},
        profile="default",
        raw_strategy="grand_master_bot",
        features={},
        action="BUY",
        quantity=1.0,
    )
    assert strategy_specialization_guard_reasons(attached, require_candidate=True) == []


def test_paper_performance_prefers_contract_identity_and_emits_strategy_series() -> None:
    strategy_id = "sleeve::equity_core::ensemble_champion::v1"
    rows = [
        {
            "timestamp_utc": "2026-08-18T15:00:00+00:00",
            "symbol": "SPY",
            "strategy": "grand_master_bot",
            "paper_pnl_schema_version": 3,
            "post_cost_pnl_delta": 1.25,
            "post_cost_return_bps": 2.5,
            "metadata": {
                "source_profile": "default",
                "strategy_specialization": {
                    "selected_strategy_id": strategy_id,
                    "selected_strategy_name": "ensemble_champion",
                    "source_kind": "synthetic_ensemble",
                    "objective_class": "directional_alpha",
                    "contract_complete": True,
                    "contract_receipt_sha256": "abc123",
                },
            },
        },
        {
            "timestamp_utc": "2026-08-19T15:00:00+00:00",
            "symbol": "QQQ",
            "strategy": "grand_master_bot",
            "paper_pnl_schema_version": 3,
            "post_cost_pnl_delta": 0.75,
            "post_cost_return_bps": 1.5,
            "metadata": {
                "source_profile": "default",
                "strategy_specialization": {
                    "selected_strategy_id": strategy_id,
                    "selected_strategy_name": "ensemble_champion",
                    "source_kind": "synthetic_ensemble",
                    "objective_class": "directional_alpha",
                    "contract_complete": True,
                    "contract_receipt_sha256": "abc123",
                },
            },
        },
    ]

    assert _strategy_of(rows[0]) == strategy_id
    series = _candidate_strategy_post_cost_daily_series(rows)
    latest = _strategy_post_cost_latest(rows)
    assert list(series) == [strategy_id]
    assert len(series[strategy_id]) == 2
    assert latest[0]["strategy_id"] == strategy_id
    assert latest[0]["sample_count"] == 2
    assert latest[0]["independent_day_count"] == 2
    assert latest[0]["independent_symbol_count"] == 2


def test_report_is_candidate_bound_and_uses_objective_aware_lifecycle(tmp_path: Path) -> None:
    policy = load_policy()
    policy_path = tmp_path / "policy.json"
    performance_path = tmp_path / "performance.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    rows = [
        {
            "strategy_id": "sleeve::equity_core::trend_follow::v1",
            "sample_count": 120,
            "independent_day_count": 8,
            "independent_symbol_count": 4,
            "post_cost_expectancy": {
                "positive_clustered_lower_confidence_bound_95": True,
                "mean_post_cost_return_bps": 3.0,
            },
        },
        {
            "strategy_id": "sleeve::short_bias_hedge::beta_hedge_efficiency::v1",
            "sample_count": 120,
            "independent_day_count": 8,
            "independent_symbol_count": 4,
            "post_cost_expectancy": {
                "positive_clustered_lower_confidence_bound_95": True,
                "mean_post_cost_return_bps": 3.0,
            },
        },
    ]
    performance_path.write_text(
        json.dumps(_bound_performance(rows)), encoding="utf-8"
    )

    payload, contracts = build_payload(
        PROJECT_ROOT,
        policy_path=policy_path,
        performance_path=performance_path,
        generated_at_utc="2026-08-18T20:01:00+00:00",
    )
    by_id = {row["strategy_id"]: row for row in payload["strategy_rows"]}

    assert payload["ok"] is True
    assert payload["contract_coverage"]["grade"] == "A+"
    assert payload["candidate_binding"]["bound"] is True
    assert by_id["sleeve::equity_core::trend_follow::v1"]["lifecycle"] == "validated_candidate"
    assert by_id["sleeve::short_bias_hedge::beta_hedge_efficiency::v1"]["lifecycle"] == "probation"
    assert by_id["sleeve::infrastructure_risk::margin_guard::v1"]["lifecycle"] == "control_only"
    assert contracts["contract_count"] == 879
    assert not any(payload["authority_contract"].values())

    library = build_library_payload(
        PROJECT_ROOT,
        policy_path=policy_path,
        hot_strategy_rows=payload["strategy_rows"],
        generated_at_utc="2026-08-18T20:01:00+00:00",
    )
    assert library["ok"] is True
    assert library["library_contract"]["strategy_count"] == 12000
    assert library["library_contract"]["hot_strategy_count"] == 879
    assert library["library_contract"]["cold_strategy_count"] == 11121
    assert library["library_contract"]["minimum_strategies_per_sleeve"] == 108
    assert library["library_contract"]["maximum_strategies_per_sleeve"] == 109
    assert (
        library["regime_activation_summary"]["cold_activation_eligible_count"]
        == 0
    )

    families = build_family_payload(
        PROJECT_ROOT,
        policy_path=policy_path,
        hot_strategy_rows=payload["strategy_rows"],
        library_rows=library["strategies"],
        generated_at_utc="2026-08-18T20:01:00+00:00",
    )
    contract = families["consolidation_contract"]
    assert families["ok"] is True
    assert contract["conceptual_strategy_count"] == 12000
    assert contract["canonical_record_count"] == 1989
    assert contract["native_hot_family_count"] == 879
    assert contract["cold_parent_family_count"] == 1110
    assert contract["cold_child_variant_count"] == 11121
    assert contract["lineage_covered_strategy_count"] == 12000
    assert contract["lineage_missing_count"] == 0
    assert contract["lineage_duplicate_count"] == 0
    assert contract["runtime_identity_change_count"] == 0
    assert contract["evidence_pooling_allowed"] is False
    assert contract["runtime_authority"] is False
    coverage = families["condition_coverage"]
    assert coverage["configured_condition_count"] == 12
    assert coverage["all_cold_parent_families_support_all_conditions"] is True
    assert coverage["materialized_parent_counts"]["volatility_targeted"] == 0
    cold_families = [
        row
        for row in families["families"]
        if row["family_kind"] == "cold_generated_parent"
    ]
    assert len(cold_families) == 1110
    assert all(len(row["supported_conditions"]) == 12 for row in cold_families)
    assert all(row["family_evidence"]["evidence_pooling_allowed"] is False for row in cold_families)
    lineage = [
        child["strategy_id"]
        for family in families["families"]
        for child in family["child_variants"]
    ]
    assert set(lineage) == set(materialize_strategy_library(policy=policy))


def test_quality_assessment_never_calls_missing_evidence_bad() -> None:
    policy = load_policy()
    contract = resolve_strategy_contract(
        "equity_core", "trend_follow", policy=policy
    )

    unknown = _quality_assessment(
        contract,
        {},
        "parked_candidate",
        policy,
    )
    good = _quality_assessment(
        contract,
        {
            "sample_count": 120,
            "independent_day_count": 8,
            "independent_symbol_count": 4,
            "post_cost_expectancy": {
                "mean_post_cost_return_bps": 2.5,
                "lower_confidence_bound_95_post_cost_return_bps": 0.8,
                "positive_clustered_lower_confidence_bound_95": True,
            },
        },
        "validated_candidate",
        policy,
    )

    assert unknown["verdict"] == "insufficient_evidence"
    assert unknown["grade"] == "NE"
    assert good["verdict"] == "validated_good"
    assert good["grade"] == "A+"


def test_mature_negative_strategy_requires_retirement_review() -> None:
    contract = {"objective_class": "directional_alpha"}
    evidence = {
        "sample_count": 200,
        "independent_day_count": 14,
        "independent_symbol_count": 5,
        "post_cost_expectancy": {
            "mean_post_cost_return_bps": -2.0,
            "lower_confidence_bound_95_post_cost_return_bps": -4.0,
            "positive_clustered_lower_confidence_bound_95": False,
        },
    }
    policy = {
        "candidate_binding": {
            "minimum_probation_samples": 30,
            "minimum_validation_samples": 100,
            "minimum_independent_days": 7,
            "minimum_independent_symbols": 3,
        }
    }

    lifecycle, blockers, ready = _lifecycle(
        contract,
        evidence,
        {"bound": True},
        policy,
    )

    assert lifecycle == "retirement_review"
    assert blockers == ["mature_negative_candidate_forward_post_cost_expectancy"]
    assert ready is False


def test_shadow_attachment_failure_preserves_paper_intent(monkeypatch) -> None:
    from scripts import run_shadow_training_loop as shadow_loop

    def _fail(*_args, **_kwargs):
        raise ValueError("invalid test policy")

    monkeypatch.setattr(shadow_loop, "attach_strategy_specialization", _fail)
    metadata = shadow_loop._attach_strategy_specialization_safe(
        {"production_candidate_id": "pc-test-g1"},
        profile="equity_core",
        raw_strategy="trend_follow",
        features={"trend_score": 0.8},
        action="BUY",
        quantity=1.0,
    )

    receipt = metadata["strategy_specialization"]
    assert receipt["status"] == "unavailable"
    assert receipt["action_observed"] == "BUY"
    assert receipt["quantity_observed"] == 1.0
    assert receipt["action_or_quantity_mutated"] is False
    assert receipt["candidate_binding"]["candidate_id"] == "pc-test-g1"
    assert not any(receipt["authority"].values())
