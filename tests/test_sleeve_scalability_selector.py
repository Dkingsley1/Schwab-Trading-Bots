from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from core.sleeve_scalability_selector import (
    build_selector_payload,
    validate_policy,
)
from scripts.ops import artifact_freshness_slo
from scripts.ops import runtime_artifact_refresh
from scripts.ops import runtime_gate_dashboard
from scripts.ops import source_mutation_guard

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 8, 29, 16, 0, tzinfo=timezone.utc)
NOW_TEXT = NOW.isoformat()


def _policy() -> dict:
    return json.loads(
        (PROJECT_ROOT / "config" / "sleeve_scalability_selector_v1.json").read_text(
            encoding="utf-8"
        )
    )


def _profile(
    bot_id: str,
    sleeve_id: str,
    daily: dict[str, float],
    *,
    lcb_bps: float = 8.0,
    capacity_usd: float = 500.0,
    correlation_cluster_id: str = "",
) -> dict:
    return {
        "bot_id": bot_id,
        "sleeve_id": sleeve_id,
        "shadow_vote_eligible": True,
        "correlation_cluster_id": correlation_cluster_id or f"{sleeve_id}/cluster_a",
        "candidate_evidence": {
            "sample_count": 60,
            "effective_sample_count": 50,
            "independent_day_count": len(daily),
            "regime_count": 2,
            "post_cost_return_lcb_bps": lcb_bps,
            "turnover_notional": 1000.0,
            "daily_post_cost_pnl": daily,
            "regimes": ["trend_normal", "range_normal"],
        },
        "rank_evidence_ready": True,
        "persistence_ready": True,
        "marginal_contribution": {
            "evidence_ready": True,
            "duplicate_cluster": False,
        },
        "capacity_curve": {
            "evidence_ready": True,
            "maximum_supported_notional": capacity_usd,
        },
        "current_regime_compatibility": {
            "compatible": True,
            "score": 0.8,
        },
        "forward_rank": {"score": 0.8},
    }


def _artifacts(
    profiles: list[dict],
    *,
    tier: str = "micro_validation",
    capital_usd: float = 200.0,
    max_order_notional_usd: float = 100.0,
    candidate_id: str = "candidate-1",
    graduation_candidate_id: str | None = None,
    growth_metrics: dict | None = None,
) -> tuple[dict, dict, dict, dict, dict, dict]:
    graduation_candidate_id = graduation_candidate_id or candidate_id
    manifest = {
        "timestamp_utc": NOW_TEXT,
        "candidate_binding": {
            "candidate_id": candidate_id,
            "bound": True,
        },
        "profiles": profiles,
    }
    hierarchy = {
        "timestamp_utc": NOW_TEXT,
        "assignments": [
            {
                "bot_id": row["bot_id"],
                "sleeve_id": row["sleeve_id"],
                "role_id": "signal",
                "active": True,
            }
            for row in profiles
        ],
    }
    default_growth_metrics = {
        "reconciled_round_trip_count": 0,
        "independent_trading_day_count": 0,
        "distinct_regime_bucket_count": 0,
        "total_post_cost_pnl_usd": 0.0,
        "normal_approx_lcb_95_post_cost_return_bps": None,
        "maximum_cumulative_drawdown_usd": 0.0,
        "actual_fee_evidence_round_trip_count": 0,
        "benchmark_evidence_round_trip_count": 0,
        "total_benchmark_excess_return_bps": 0.0,
        "daily_post_cost_pnl_usd": {},
    }
    default_growth_metrics.update(growth_metrics or {})
    graduation_policy = json.loads(
        (PROJECT_ROOT / "config" / "live_canary_graduation_v1.json").read_text(
            encoding="utf-8"
        )
    )
    tier_rows = graduation_policy["capital_ladder"][:3]
    graduation = {
        "timestamp_utc": NOW_TEXT,
        "control_ok": True,
        "identity": {
            "candidate_id": graduation_candidate_id,
            "account_policy_key": "account-1",
            "execution_route_id": "dividend_liquid_etf_candidate_v1",
        },
        "blockers": [],
        "metrics": default_growth_metrics,
        "capital_ladder": {
            "active_policy_tier": tier,
            "evaluations": [
                {
                    "tier": row["tier"],
                    "proposed_limits": {
                        "account_capital_usd": (
                            capital_usd
                            if row["tier"] == tier
                            else row["account_capital_usd"]
                        ),
                        "max_order_notional_usd": (
                            max_order_notional_usd
                            if row["tier"] == tier
                            else row["max_order_notional_usd"]
                        ),
                    },
                    "policy_requirements": {
                        "minimum_reconciled_round_trips": row[
                            "minimum_reconciled_round_trips"
                        ],
                        "minimum_independent_trading_days": row[
                            "minimum_independent_trading_days"
                        ],
                        "minimum_distinct_regime_buckets": row[
                            "minimum_distinct_regime_buckets"
                        ],
                        "require_positive_normal_approx_lcb_95": row[
                            "require_positive_normal_approx_lcb_95"
                        ],
                        "require_positive_benchmark_excess": row[
                            "require_positive_benchmark_excess"
                        ],
                        "require_actual_fee_evidence": row[
                            "require_actual_fee_evidence"
                        ],
                    },
                    "scale_governance": {
                        "operating_class": "personal_brokerage",
                        "operating_class_cataloged": True,
                        "operating_class_enabled": True,
                        "required_controls": [],
                        "missing_controls": [],
                        "required_controls_evidenced": True,
                    },
                    "operator_review_eligible": True,
                    "limits_applied": row["tier"] == tier,
                    "automatic_scaling": False,
                }
                for row in tier_rows
            ],
        },
    }
    account_study = {
        "timestamp_utc": NOW_TEXT,
        "accounts": [
            {
                "account_policy_key": "account-1",
                "account_capability_truth": {
                    "operator_classification": {
                        "account_kind": "roth_ira",
                        "tax_wrapper": "roth_ira",
                        "trading_access": "limited_margin",
                        "borrowing_allowed": False,
                        "short_stock_allowed": False,
                        "classification_complete": True,
                        "canary_cap_usd": capital_usd,
                        "allowed_live_routes": ["dividend_liquid_etf_candidate_v1"],
                    },
                    "canary_preflight": {
                        "account_preflight_ready": True,
                        "blockers": [],
                    },
                },
            }
        ],
    }
    firewall = {
        "timestamp_utc": NOW_TEXT,
        "promotion_evidence_ready": True,
        "economic_evidence_ready": True,
    }
    calibration = {
        "timestamp_utc": NOW_TEXT,
        "independent_evidence_ready": True,
        "independent_samples": 30,
    }
    return manifest, hierarchy, graduation, account_study, firewall, calibration


def _build(
    profiles: list[dict],
    **artifact_kwargs,
) -> dict:
    artifacts = _artifacts(profiles, **artifact_kwargs)
    return build_selector_payload(_policy(), *artifacts, now_utc=NOW)


def test_policy_is_valid_and_zero_authority() -> None:
    policy = _policy()

    assert validate_policy(policy) == []
    assert policy["safety_contract"]["advisory_only"] is True
    assert policy["safety_contract"]["live_execution_authority"] is False
    assert policy["safety_contract"]["automatic_capital_scaling"] is False


def test_billion_scale_ladders_remain_aligned_and_fail_closed() -> None:
    selector = _policy()
    graduation = json.loads(
        (PROJECT_ROOT / "config" / "live_canary_graduation_v1.json").read_text(
            encoding="utf-8"
        )
    )
    growth_rows = selector["capital_growth"]["targets"]
    tier_rows = graduation["capital_ladder"]

    assert len(growth_rows) == len(tier_rows) == 22
    assert growth_rows[-1]["capital_usd"] == 1_000_000_000
    assert tier_rows[-1]["account_capital_usd"] == 1_000_000_000
    assert selector["capital_growth"]["deployment_requirements"][-1] == {
        "maximum_capital_usd": 1_000_000_000.0,
        "minimum_independent_sleeves": 8,
        "minimum_capacity_headroom_ratio": 5.0,
    }
    for growth, tier in zip(growth_rows, tier_rows, strict=True):
        assert growth["capital_usd"] == tier["account_capital_usd"]
        assert (
            growth["minimum_reconciled_round_trips"]
            == tier["minimum_reconciled_round_trips"]
        )
        assert (
            growth["minimum_independent_days"]
            == tier["minimum_independent_trading_days"]
        )
        assert (
            growth["minimum_distinct_regimes"]
            == tier["minimum_distinct_regime_buckets"]
        )
        assert (
            growth["require_positive_lcb_95"]
            == tier["require_positive_normal_approx_lcb_95"]
        )
        assert (
            growth["require_positive_benchmark_excess"]
            == tier["require_positive_benchmark_excess"]
        )
        assert (
            growth["require_actual_fee_evidence"] == tier["require_actual_fee_evidence"]
        )


def test_scaling_scope_covers_classified_accounts_without_pooling_identity() -> None:
    days = {
        "2026-08-24": 1.0,
        "2026-08-25": 2.0,
        "2026-08-26": 1.5,
        "2026-08-27": 2.5,
        "2026-08-28": 2.0,
    }
    artifacts = list(_artifacts([_profile("bot-a", "dividend_income", days)]))
    second = deepcopy(artifacts[3]["accounts"][0])
    second["account_policy_key"] = "account-2"
    classification = second["account_capability_truth"]["operator_classification"]
    classification["account_kind"] = "cash"
    classification["tax_wrapper"] = "taxable"
    artifacts[3]["accounts"].append(second)

    payload = build_selector_payload(_policy(), *artifacts, now_utc=NOW)
    scope = payload["account_scaling_scope"]

    assert scope["classified_account_policy_count"] == 2
    assert {row["account_policy_key"] for row in scope["accounts"]} == {
        "account-1",
        "account-2",
    }
    assert scope["organic_progress_isolated_by_account_policy_key"] is True
    assert scope["cross_account_evidence_pooling"] is False
    assert scope["raw_account_identifiers_present"] is False
    assert payload["host_portability"]["credential_material_portable"] is False
    assert (
        payload["host_portability"]["automatic_live_reactivation_on_new_host"] is False
    )


def test_growth_target_requires_capacity_headroom_not_bare_notional() -> None:
    days = {
        "2026-08-24": 1.0,
        "2026-08-25": 2.0,
        "2026-08-26": 1.5,
        "2026-08-27": 2.5,
        "2026-08-28": 2.0,
    }
    payload = _build([_profile("bot-a", "dividend_income", days, capacity_usd=499.0)])

    assert payload["selected_advisory_plan"] is not None
    assert payload["capital_growth_plan"]["target_sleeve_plan"] == {}
    assert (
        "target_capacity_headroom" in payload["capital_growth_plan"]["target_blockers"]
    )
    assert payload["growth_portfolio_search"]["capacity_rejection_count"] == 1


def test_missing_evidence_abstains_without_creating_degradation() -> None:
    manifest, hierarchy, graduation, account, firewall, calibration = _artifacts([])
    manifest["candidate_binding"]["bound"] = False
    firewall["promotion_evidence_ready"] = False
    firewall["economic_evidence_ready"] = False
    calibration["independent_evidence_ready"] = False
    calibration["independent_samples"] = 0

    payload = build_selector_payload(
        _policy(),
        manifest,
        hierarchy,
        graduation,
        account,
        firewall,
        calibration,
        now_utc=NOW,
    )

    assert payload["ok"] is True
    assert payload["overall_status"] == "ready_with_evidence_debt"
    assert payload["recommendation_ready"] is False
    assert payload["selected_advisory_plan"] is None
    assert payload["authority_contract"]["application_allowed"] is False
    assert payload["authority_contract"]["live_execution_authority"] is False
    contract = payload["operating_contract"]
    assert contract["complete"] is True
    assert "independent_execution_calibration_ready" in contract["evidence_missing"]
    parameter_contract = payload["sleeve_parameter_contract"]
    assert parameter_contract["contract_id"] == "sleeve_specific_parameter_contract_v1"
    assert parameter_contract["application_eligible_sleeve_count"] == 0
    assert parameter_contract["required_parameters"]["minimum_candidate_samples"] >= 1


def test_single_best_supported_sleeve_is_recommended_for_micro_tier() -> None:
    days = {
        "2026-08-24": 1.0,
        "2026-08-25": 2.0,
        "2026-08-26": 1.5,
        "2026-08-27": 2.5,
        "2026-08-28": 2.0,
    }
    payload = _build([_profile("bot-a", "dividend_income", days)])

    assert payload["overall_status"] == "ready"
    assert payload["recommendation_ready"] is True
    assert payload["selected_advisory_plan"]["plan_type"] == "single_sleeve"
    assert payload["selected_advisory_plan"]["sleeve_ids"] == ["dividend_income"]
    assert payload["selected_advisory_plan"]["application_allowed"] is False
    assert payload["operating_contract"]["why"] == "recommendation_ready"
    assert (
        payload["sleeve_parameter_contract"]["sleeves"][0]["status"]
        == "application_eligible"
    )


def test_low_correlation_sleeves_form_bounded_multi_sleeve_plan() -> None:
    first = {
        "2026-08-24": 1.0,
        "2026-08-25": 2.0,
        "2026-08-26": 3.0,
        "2026-08-27": 4.0,
        "2026-08-28": 5.0,
    }
    second = {
        "2026-08-24": 3.0,
        "2026-08-25": 1.0,
        "2026-08-26": 5.0,
        "2026-08-27": 2.0,
        "2026-08-28": 4.0,
    }
    payload = _build(
        [
            _profile("bot-a", "dividend_income", first),
            _profile("bot-b", "dividend_capture", second),
        ],
        tier="micro_two",
        capital_usd=400.0,
        max_order_notional_usd=150.0,
    )

    assert payload["recommendation_ready"] is True
    assert payload["selected_advisory_plan"]["plan_type"] == "multi_sleeve"
    assert set(payload["selected_advisory_plan"]["sleeve_ids"]) == {
        "dividend_income",
        "dividend_capture",
    }
    weights = [
        row["advisory_weight"]
        for row in payload["selected_advisory_plan"]["allocations"]
    ]
    assert abs(sum(weights) - 1.0) < 1e-7
    assert max(weights) <= 0.6 + 1e-9
    goal = next(
        row
        for row in payload["system_scalability_goals"]
        if row["goal_id"] == "g02_dual_sleeve_diversification"
    )
    assert goal["status"] == "earned"


def test_high_correlation_prevents_duplicate_multi_sleeve_plan() -> None:
    days = {
        "2026-08-24": 1.0,
        "2026-08-25": 2.0,
        "2026-08-26": 3.0,
        "2026-08-27": 4.0,
        "2026-08-28": 5.0,
    }
    payload = _build(
        [
            _profile("bot-a", "dividend_income", days),
            _profile("bot-b", "dividend_capture", days),
        ],
        tier="micro_two",
        capital_usd=400.0,
        max_order_notional_usd=150.0,
    )

    assert payload["selected_advisory_plan"]["plan_type"] == "single_sleeve"
    assert payload["portfolio_search"]["excess_correlation_rejection_count"] >= 1


def test_unknown_correlation_fails_closed_for_multi_sleeve_only() -> None:
    first = {
        "2026-08-26": 1.0,
        "2026-08-27": 2.0,
        "2026-08-28": 3.0,
    }
    second = {
        "2026-08-26": 3.0,
        "2026-08-27": 1.0,
        "2026-08-28": 2.0,
    }
    payload = _build(
        [
            _profile("bot-a", "dividend_income", first),
            _profile("bot-b", "dividend_capture", second),
        ],
        tier="micro_two",
        capital_usd=400.0,
        max_order_notional_usd=150.0,
    )

    assert payload["selected_advisory_plan"]["plan_type"] == "single_sleeve"
    assert payload["portfolio_search"]["unknown_correlation_rejection_count"] >= 1


def test_negative_lcb_and_insufficient_capacity_are_excluded() -> None:
    days = {
        "2026-08-24": 1.0,
        "2026-08-25": 2.0,
        "2026-08-26": 3.0,
        "2026-08-27": 4.0,
        "2026-08-28": 5.0,
    }
    payload = _build(
        [
            _profile("bot-negative", "dividend_income", days, lcb_bps=-1.0),
            _profile("bot-small", "dividend_capture", days, capacity_usd=50.0),
        ]
    )
    rankings = {row["sleeve_id"]: row for row in payload["sleeve_rankings"]}

    assert rankings["dividend_income"]["research_eligible"] is False
    assert "qualified_bot_count" in rankings["dividend_income"]["evidence_blockers"]
    assert rankings["dividend_capture"]["research_eligible"] is False
    assert "supported_notional" in rankings["dividend_capture"]["evidence_blockers"]
    assert payload["selected_advisory_plan"] is None


def test_route_mismatch_keeps_research_result_out_of_application() -> None:
    days = {
        "2026-08-24": 1.0,
        "2026-08-25": 2.0,
        "2026-08-26": 3.0,
        "2026-08-27": 4.0,
        "2026-08-28": 5.0,
    }
    payload = _build([_profile("bot-a", "equity_core", days)])
    ranking = payload["sleeve_rankings"][0]

    assert ranking["research_eligible"] is True
    assert ranking["route_match"] is False
    assert ranking["application_eligible"] is False
    assert "sleeve_not_allowed_for_execution_route" in ranking["application_blockers"]
    assert payload["selected_advisory_plan"] is None


def test_route_match_without_earned_evidence_is_not_labeled_best_supported() -> None:
    days = {
        "2026-08-24": -1.0,
        "2026-08-25": -2.0,
        "2026-08-26": -3.0,
        "2026-08-27": -4.0,
        "2026-08-28": -5.0,
    }
    payload = _build([_profile("bot-a", "dividend_income", days, lcb_bps=-2.0)])

    assert payload["summary"]["route_matched_sleeve_count"] == 1
    assert payload["best_supported_route_sleeve"] == ""
    assert payload["selected_advisory_plan"] is None


def test_candidate_identity_mismatch_fails_closed() -> None:
    days = {
        "2026-08-24": 1.0,
        "2026-08-25": 2.0,
        "2026-08-26": 3.0,
        "2026-08-27": 4.0,
        "2026-08-28": 5.0,
    }
    payload = _build(
        [_profile("bot-a", "dividend_income", days)],
        graduation_candidate_id="candidate-other",
    )

    assert payload["ok"] is False
    assert payload["overall_status"] == "blocked"
    assert "candidate_identity_mismatch" in payload["hard_integrity_blockers"]
    assert payload["selected_advisory_plan"] is None


def test_selection_receipt_is_deterministic() -> None:
    days = {
        "2026-08-24": 1.0,
        "2026-08-25": 2.0,
        "2026-08-26": 3.0,
        "2026-08-27": 4.0,
        "2026-08-28": 5.0,
    }
    profiles = [_profile("bot-a", "dividend_income", days)]

    first = _build(deepcopy(profiles))
    second = _build(deepcopy(profiles))

    assert first["selection_receipt_sha256"] == second["selection_receipt_sha256"]


def _earned_growth_metrics(total_post_cost_pnl_usd: float) -> dict:
    return {
        "reconciled_round_trip_count": 10,
        "independent_trading_day_count": 5,
        "distinct_regime_bucket_count": 2,
        "total_post_cost_pnl_usd": total_post_cost_pnl_usd,
        "normal_approx_lcb_95_post_cost_return_bps": 4.0,
        "maximum_cumulative_drawdown_usd": 2.0,
        "actual_fee_evidence_round_trip_count": 10,
        "benchmark_evidence_round_trip_count": 10,
        "total_benchmark_excess_return_bps": 20.0,
        "daily_post_cost_pnl_usd": {
            "2026-08-24": total_post_cost_pnl_usd * 0.10,
            "2026-08-25": total_post_cost_pnl_usd * 0.15,
            "2026-08-26": total_post_cost_pnl_usd * 0.20,
            "2026-08-27": total_post_cost_pnl_usd * 0.25,
            "2026-08-28": total_post_cost_pnl_usd * 0.30,
        },
    }


def test_growth_ladder_compounds_only_reconciled_post_cost_profit() -> None:
    days = {
        "2026-08-24": 1.0,
        "2026-08-25": 2.0,
        "2026-08-26": 1.5,
        "2026-08-27": 2.5,
        "2026-08-28": 2.0,
    }
    payload = _build(
        [_profile("bot-a", "dividend_income", days)],
        growth_metrics=_earned_growth_metrics(100.0),
    )
    growth = payload["capital_growth_plan"]

    assert growth["organic_capital_usd"] == 300.0
    assert growth["next_target_capital_usd"] == 400.0
    assert growth["progress_to_next_target_percent"] == 50.0
    assert growth["advisory_reinvestment_budget_usd"] == 50.0
    assert growth["advisory_profit_reserve_usd"] == 50.0
    assert growth["recommended_action"] == (
        "retain_and_compound_earned_profit_after_review"
    )
    assert growth["target_sleeve_plan"]["sleeve_ids"] == ["dividend_income"]
    assert (
        growth["accounting_contract"]["external_deposits_count_toward_organic_progress"]
        is False
    )
    assert growth["authority_contract"]["automatic_reinvestment"] is False
    assert growth["authority_contract"]["live_execution_authority"] is False


def test_growth_target_reached_requires_operator_review_not_auto_scaling() -> None:
    days = {
        "2026-08-24": 1.0,
        "2026-08-25": 2.0,
        "2026-08-26": 1.5,
        "2026-08-27": 2.5,
        "2026-08-28": 2.0,
    }
    payload = _build(
        [_profile("bot-a", "dividend_income", days)],
        growth_metrics=_earned_growth_metrics(200.0),
    )
    growth = payload["capital_growth_plan"]

    assert growth["organic_capital_usd"] == 400.0
    assert growth["progress_to_next_target_percent"] == 100.0
    assert growth["operator_review_ready"] is True
    assert growth["recommended_action"] == "operator_review_next_growth_tier"
    assert growth["authority_contract"]["automatic_capital_scaling"] is False


def test_growth_loss_defends_seed_without_reinvestment() -> None:
    days = {
        "2026-08-24": 1.0,
        "2026-08-25": 2.0,
        "2026-08-26": 1.5,
        "2026-08-27": 2.5,
        "2026-08-28": 2.0,
    }
    metrics = _earned_growth_metrics(-20.0)
    metrics["normal_approx_lcb_95_post_cost_return_bps"] = -4.0
    payload = _build(
        [_profile("bot-a", "dividend_income", days)],
        growth_metrics=metrics,
    )
    growth = payload["capital_growth_plan"]

    assert growth["organic_capital_usd"] == 180.0
    assert growth["advisory_reinvestment_budget_usd"] == 0.0
    assert growth["recommended_action"] == "defend_seed_and_reduce_risk"


def test_next_growth_tier_can_prefer_a_low_correlation_multi_sleeve_plan() -> None:
    first = {
        "2026-08-24": 1.0,
        "2026-08-25": 2.0,
        "2026-08-26": 3.0,
        "2026-08-27": 4.0,
        "2026-08-28": 5.0,
    }
    second = {
        "2026-08-24": 3.0,
        "2026-08-25": 1.0,
        "2026-08-26": 5.0,
        "2026-08-27": 2.0,
        "2026-08-28": 4.0,
    }
    payload = _build(
        [
            _profile("bot-a", "dividend_income", first),
            _profile("bot-b", "dividend_capture", second),
        ]
    )
    growth_plan = payload["capital_growth_plan"]["target_sleeve_plan"]

    assert payload["selected_advisory_plan"]["plan_type"] == "single_sleeve"
    assert growth_plan["target_capital_usd"] == 400.0
    assert growth_plan["plan_type"] == "multi_sleeve"
    assert set(growth_plan["sleeve_ids"]) == {
        "dividend_income",
        "dividend_capture",
    }


def test_repository_wiring_refreshes_protects_and_surfaces_selector() -> None:
    refresh_steps = {
        row["name"]: row for row in runtime_artifact_refresh._step_specs(PROJECT_ROOT)
    }
    freshness = artifact_freshness_slo._artifact_contract(PROJECT_ROOT)
    dashboard = runtime_gate_dashboard._artifact_config(PROJECT_ROOT)
    role_contract = json.loads(
        (PROJECT_ROOT / "config" / "system_role_contracts_v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert "sleeve_scalability_selector" in refresh_steps
    assert freshness["sleeve_scalability_selector"]["required"] is True
    assert dashboard["sleeve_scalability_selector"]["required"] is True
    assert (
        "core/sleeve_scalability_selector.py"
        in source_mutation_guard.DEFAULT_PROTECTED_PATHS
    )
    assert (
        role_contract["control_surface_bindings"]["sleeve_scalability_selector_health"]
        == "sleeve_scalability_selector_controller"
    )


def test_dashboard_summary_exposes_advice_without_execution_authority() -> None:
    summary = runtime_gate_dashboard._artifact_summary(
        "sleeve_scalability_selector",
        {
            "overall_status": "ready_with_evidence_debt",
            "control_ready": True,
            "recommendation_ready": True,
            "identity": {
                "candidate_id": "candidate-1",
                "active_capital_tier": "micro_two",
            },
            "summary": {
                "evaluated_sleeve_count": 12,
                "research_eligible_sleeve_count": 3,
                "application_eligible_sleeve_count": 2,
                "earned_scalability_goal_count": 1,
                "scalability_goal_count": 6,
            },
            "selected_advisory_plan": {
                "sleeve_ids": ["dividend_income", "dividend_capture"]
            },
            "capital_growth_plan": {
                "organic_capital_usd": 260.0,
                "next_target_capital_usd": 400.0,
                "long_range_target_capital_usd": 1_000_000_000.0,
                "next_target_operating_class": "personal_brokerage",
                "progress_to_next_target_percent": 30.0,
                "recommended_action": "retain_and_compound_earned_profit_after_review",
                "operator_review_ready": False,
                "target_sleeve_plan": {
                    "sleeve_ids": ["dividend_income", "dividend_capture"]
                },
                "authority_contract": {"automatic_reinvestment": False},
            },
            "account_scaling_scope": {
                "classified_account_policy_count": 3,
                "organic_progress_isolated_by_account_policy_key": True,
            },
            "host_portability": {
                "status": "portable_policy_requires_new_host_rebind",
                "automatic_live_reactivation_on_new_host": False,
            },
            "evidence_debt": ["g02"],
            "authority_contract": {
                "paper_execution_authority": False,
                "live_execution_authority": False,
            },
        },
    )

    assert summary["selected_sleeve_ids"] == [
        "dividend_income",
        "dividend_capture",
    ]
    assert summary["earned_scalability_goal_count"] == 1
    assert summary["organic_capital_usd"] == 260.0
    assert summary["next_organic_capital_target_usd"] == 400.0
    assert summary["long_range_organic_target_usd"] == 1_000_000_000.0
    assert summary["next_target_operating_class"] == "personal_brokerage"
    assert summary["growth_target_sleeve_ids"] == [
        "dividend_income",
        "dividend_capture",
    ]
    assert summary["automatic_reinvestment"] is False
    assert summary["classified_account_policy_count"] == 3
    assert summary["account_evidence_isolated"] is True
    assert summary["host_portability_status"] == (
        "portable_policy_requires_new_host_rebind"
    )
    assert summary["automatic_live_reactivation_on_new_host"] is False
    assert summary["paper_execution_authority"] is False
    assert summary["live_execution_authority"] is False
