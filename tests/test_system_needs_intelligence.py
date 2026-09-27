from __future__ import annotations

import json
from pathlib import Path

from scripts.ops import system_needs_intelligence as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _seed_paper_collection_controls(project_root: Path) -> None:
    _write_json(
        project_root / "config" / "paper_evidence_collection_controls_v1.json",
        {
            "policy_id": "paper_evidence_collection_controls_v1",
            "enabled": True,
            "paper_only": True,
            "live_execution_allowed": False,
            "defaults": {
                "force_trade_allowed": False,
                "loss_recovery_size_increase_allowed": False,
                "profitability_claim_allowed": False,
                "allowed_guard_reasons": ["paper_turnover_sampling_cap"],
                "allowed_entry_policy_blockers": ["paper_turnover_sampling_cap"],
                "allowed_clean_gate_failures": ["auxiliary_context_missing"],
                "minimum_model_score_edge_over_threshold": 0.02,
                "minimum_known_core_channels": 3,
            },
            "profiles": {
                "volatility": {"max_new_entries_per_window": 3},
                "intraday": {"max_new_entries_per_window": 2},
            },
            "non_relaxed_controls": [
                "live_execution",
                "position_size",
                "spread",
                "quote_age",
            ],
        },
    )


def _seed_operator_brain_boundary(project_root: Path) -> None:
    _write_json(
        project_root / "config" / "system_role_contracts_v1.json",
        {
            "operator_brain_boundary": {
                "boundary_id": "trading_brain_ops_brain_v1",
                "purpose": "Keep trading changes separate from operational hardening.",
                "trading_brain": {
                    "display_name": "Trading Brain",
                    "definition": "Decision layer that changes sleeve parameters, entry/exit logic, sizing, candidate identity, or profitability semantics.",
                    "scope": ["sleeve_parameters", "entry_exit_logic"],
                    "categories": [
                        {
                            "category_id": "market_decision",
                            "purpose": "Interprets market regime and sleeve priority.",
                            "scope": ["market_regime_interpretation"],
                            "current_posture": "read_only_or_paper_observation_only",
                            "allowed_actions": ["explain_market_read"],
                            "frozen_or_review_actions": ["change_regime_model"],
                        },
                        {
                            "category_id": "strategy_logic",
                            "purpose": "Owns entries, exits, thresholds, and sleeve parameters.",
                            "scope": ["entry_exit_logic"],
                            "current_posture": "frozen_except_candidate_preserving_bug_fixes",
                            "allowed_actions": ["inspect_logic"],
                            "frozen_or_review_actions": ["relax_thresholds"],
                        },
                        {
                            "category_id": "risk_execution_policy",
                            "purpose": "Controls sizing, paper admission, and live authority.",
                            "scope": ["position_sizing"],
                            "current_posture": "locked_no_live_authority_change",
                            "allowed_actions": ["read_only_review"],
                            "frozen_or_review_actions": ["grant_live_authority"],
                        },
                        {
                            "category_id": "candidate_evidence",
                            "purpose": "Protects candidate identity and post-cost evidence.",
                            "scope": ["candidate_identity"],
                            "current_posture": "collecting_do_not_reset",
                            "allowed_actions": ["refresh_reports"],
                            "frozen_or_review_actions": ["reset_candidate"],
                        },
                    ],
                    "current_posture": "freeze_except_bug_fixes_while_candidate_collects_evidence",
                    "allowed_during_collection": ["read_only_analysis"],
                    "requires_explicit_operator_review": [
                        "threshold_relaxation",
                        "live_execution_authority_change",
                    ],
                },
                "ops_brain": {
                    "display_name": "Ops Brain",
                    "definition": "Runtime, storage, queue, writer, reporting, and observability layer.",
                    "scope": ["storage_cleanup", "writer_locks"],
                    "categories": [
                        {
                            "category_id": "storage_runtime",
                            "purpose": "Keeps disk and runtime pressure inside safe bounds.",
                            "scope": ["disk_pressure"],
                            "current_posture": "safe_to_harden_with_guards",
                            "allowed_actions": ["guarded_cleanup"],
                            "never_actions": ["wipe_audit_trail"],
                        },
                        {
                            "category_id": "writer_queue",
                            "purpose": "Maintains single-writer safety and queue backpressure.",
                            "scope": ["writer_locks"],
                            "current_posture": "safe_to_repair_serially",
                            "allowed_actions": ["run_single_writer_drain"],
                            "never_actions": ["parallel_sql_writers"],
                        },
                        {
                            "category_id": "data_freshness",
                            "purpose": "Refreshes sources and stale artifacts.",
                            "scope": ["feed_freshness"],
                            "current_posture": "safe_to_refresh",
                            "allowed_actions": ["refresh_sources"],
                            "never_actions": ["invent_missing_source_truth"],
                        },
                        {
                            "category_id": "observability_reporting",
                            "purpose": "Improves operator reports and system communication.",
                            "scope": ["operator_reports"],
                            "current_posture": "safe_to_improve",
                            "allowed_actions": ["add_readouts"],
                            "never_actions": ["claim_profitability"],
                        },
                        {
                            "category_id": "process_recovery",
                            "purpose": "Supervises guarded restarts and resource-pressure relief.",
                            "scope": ["watchdogs"],
                            "current_posture": "safe_when_guarded",
                            "allowed_actions": [
                                "restart_supervised_processes_when_guarded"
                            ],
                            "never_actions": ["grant_live_authority"],
                        },
                    ],
                    "current_posture": "safe_to_harden_while_trading_brain_collects_evidence",
                    "allowed_during_collection": ["repair_storage_or_backpressure"],
                },
                "boundary_rules": {
                    "trading_brain_changes_may_reset_or_invalidate_candidate_evidence": True,
                    "ops_brain_never_changes_trade_logic": True,
                    "ops_brain_never_grants_live_authority": True,
                    "paper_collection_controls_may_collect_more_evidence_without_live_authority": True,
                    "live_money_authority_remains_separate": True,
                    "operator_question_default_route": "classify_first_then_apply_only_ops_safe_repairs",
                },
                "routing_matrix": {
                    "classification_rule": "Classify each operator request by brain and category.",
                    "safe_now": [
                        "ops_brain.storage_runtime",
                        "ops_brain.writer_queue",
                        "ops_brain.data_freshness",
                        "ops_brain.observability_reporting",
                        "ops_brain.process_recovery",
                    ],
                    "freeze_now": [
                        "trading_brain.strategy_logic",
                        "trading_brain.risk_execution_policy",
                        "trading_brain.candidate_evidence",
                    ],
                    "read_only_now": ["trading_brain.market_decision"],
                    "review_required": [
                        "trading_brain.strategy_logic.parameter_change",
                        "trading_brain.risk_execution_policy.authority_change",
                    ],
                },
                "hardening_invariants": {
                    "required_trading_categories": [
                        "market_decision",
                        "strategy_logic",
                        "risk_execution_policy",
                        "candidate_evidence",
                    ],
                    "required_ops_categories": [
                        "storage_runtime",
                        "writer_queue",
                        "data_freshness",
                        "observability_reporting",
                        "process_recovery",
                    ],
                },
                "handoff_language": {
                    "summary": "Trading Brain is frozen except bug fixes; Ops Brain can keep being hardened.",
                    "trading_freeze_reason": "Current candidate evidence needs a stable decision surface.",
                    "safe_work_now": "storage, writer locks, feed freshness, reports",
                    "frozen_work_now": "sleeve parameters, entry and exit logic, sizing",
                },
            }
        },
    )


def _seed_sleeve_strategy_contracts(project_root: Path) -> None:
    _write_json(
        project_root / "config" / "sleeve_strategy_contracts_v1.json",
        {
            "policy_id": "sleeve_strategy_specialization_v1",
            "sleeve_characteristics": {
                "purpose": "Give every sleeve an operator-readable market character.",
                "authority": {
                    "metadata_only": True,
                    "can_change_trade_logic": False,
                    "can_submit_order": False,
                    "can_claim_profitability": False,
                },
                "routing_defaults": {
                    "classification_rule": "Describe sleeve fit before changing parameters.",
                    "trading_brain_categories": ["market_decision"],
                    "ops_brain_dependencies": ["data_freshness"],
                },
                "objective_class_characteristics": {
                    "volatility_relative_value": {
                        "primary_character": "volatility_surface_and_tail_pricing",
                        "market_question": "Is volatility mispriced after costs?",
                        "naturally_helps_when": ["volatility_dislocation"],
                        "naturally_hurts_when": ["surface_stale"],
                        "sensitive_to": ["spread", "skew"],
                        "profitability_evidence_type": "realized_option_utility_after_costs",
                        "trading_brain_categories": [
                            "market_decision",
                            "risk_execution_policy",
                        ],
                        "ops_brain_dependencies": [
                            "data_freshness",
                            "observability_reporting",
                        ],
                    },
                    "execution_alpha": {
                        "primary_character": "microstructure_execution_edge",
                        "market_question": "Can the sleeve improve fills after latency?",
                        "naturally_helps_when": ["fresh_quotes"],
                        "naturally_hurts_when": ["wide_spreads"],
                        "sensitive_to": ["quote_age", "latency"],
                        "profitability_evidence_type": "execution_improvement_after_costs",
                        "trading_brain_categories": [
                            "market_decision",
                            "strategy_logic",
                        ],
                        "ops_brain_dependencies": ["data_freshness", "writer_queue"],
                    },
                },
            },
            "measurement_parameter_defaults": {
                "measurement_focus": ["candidate_forward_post_cost_expectancy"]
            },
            "sleeve_measurement_parameters": {
                "volatility": {"measurement_focus": ["implied_realized_spread"]},
                "intraday": {"measurement_focus": ["fill_quality"]},
            },
            "strategy_additions": {
                "volatility": ["variance_risk_premium"],
                "intraday": ["opening_auction_imbalance"],
            },
            "strategy_organization": {
                "purpose": "Group strategy hypotheses inside sleeves.",
                "authority": {
                    "metadata_only": True,
                    "can_change_trade_logic": False,
                    "can_change_sizing": False,
                    "can_submit_order": False,
                    "can_claim_profitability": False,
                    "can_promote_candidate": False,
                    "can_allocate_capital": False,
                },
                "groups": [
                    {
                        "group_id": "volatility_event_convexity",
                        "label": "Volatility and event convexity",
                        "objective_classes": ["volatility_relative_value"],
                        "primary_sleeves": ["volatility"],
                        "strategy_archetypes": ["variance_risk_premium"],
                        "expected_edge_source": "surface_dislocation_after_costs",
                        "measurement_focus": ["implied_realized_spread"],
                        "failure_modes": ["surface_stale"],
                        "master_bot_needs": ["option_surface_freshness"],
                        "grandmaster_bot_needs": ["convexity_budget_across_sleeves"],
                    },
                    {
                        "group_id": "execution_microstructure_liquidity",
                        "label": "Execution and liquidity",
                        "objective_classes": ["execution_alpha"],
                        "primary_sleeves": ["intraday"],
                        "strategy_archetypes": ["opening_auction"],
                        "expected_edge_source": "fresh_session_structure",
                        "measurement_focus": ["fill_quality"],
                        "failure_modes": ["wide_spreads"],
                        "master_bot_needs": ["independent_fill_quality"],
                        "grandmaster_bot_needs": ["runtime_headroom"],
                    },
                ],
                "hardening_invariants": {
                    "required_group_fields": [
                        "group_id",
                        "label",
                        "objective_classes",
                        "primary_sleeves",
                        "strategy_archetypes",
                        "expected_edge_source",
                        "measurement_focus",
                        "failure_modes",
                        "master_bot_needs",
                        "grandmaster_bot_needs",
                    ],
                    "metadata_authority_must_remain_true": True,
                    "every_trading_sleeve_must_be_in_at_least_one_group": True,
                    "every_sleeve_strategy_addition_must_inherit_a_group": True,
                },
            },
            "sleeves": {
                "volatility": {
                    "objective_class": "volatility_relative_value",
                    "economic_thesis": "Volatility dislocations can produce utility after costs.",
                    "universe": "options_volatility",
                    "decision_horizon": "intraday_to_daily",
                    "holding_horizon": "days_to_expiry",
                    "benchmark": "delta_hedged_option",
                    "risk_budget": "volatility",
                },
                "intraday": {
                    "objective_class": "execution_alpha",
                    "economic_thesis": "Fresh microstructure can improve fills after costs.",
                    "universe": "volatile",
                    "decision_horizon": "seconds_to_minutes",
                    "holding_horizon": "intraday_flat_by_close",
                    "benchmark": "arrival_price_and_vwap",
                    "risk_budget": "intraday",
                },
            },
        },
    )


def _seed_master_grandmaster_success(project_root: Path) -> None:
    _write_json(
        project_root / "config" / "master_grandmaster_evidence_v2.json",
        {
            "policy_id": "master_grandmaster_evidence_v2",
            "coordination_success_needs": {
                "purpose": "Define tier success needs beyond raw data collection.",
                "authority": {
                    "metadata_only": True,
                    "can_submit_order": False,
                    "can_change_trade_logic": False,
                    "can_change_sizing": False,
                    "can_allocate_capital": False,
                    "can_promote_candidate": False,
                    "can_claim_profitability": False,
                    "can_override_halt": False,
                },
                "master_bot": {
                    "view": "per_sleeve_local_optimizer_and_evidence_curator",
                    "needs": [
                        {
                            "need_id": "sleeve_strategy_taxonomy",
                            "why": "Master needs local strategy families.",
                            "evidence_artifacts": [
                                "config/sleeve_strategy_contracts_v1.json.strategy_organization"
                            ],
                            "success_signal": "every_trading_sleeve_is_grouped",
                        },
                        {
                            "need_id": "execution_cost_and_fill_quality",
                            "why": "Master needs fill quality and costs.",
                            "evidence_artifacts": [
                                "governance/health/execution_calibration_latest.json"
                            ],
                            "success_signal": "fills_meet_sleeve_thresholds",
                        },
                    ],
                    "must_not": ["submit_orders", "promote_candidate"],
                },
                "grandmaster_bot": {
                    "view": "cross_sleeve_allocator_referee_and_policy_coordinator",
                    "needs": [
                        {
                            "need_id": "cross_sleeve_correlation_and_exposure_map",
                            "why": "Grandmaster needs cross-sleeve exposure context.",
                            "evidence_artifacts": [
                                "governance/health/master_grandmaster_evidence_v2_latest.json"
                            ],
                            "success_signal": "correlation_clusters_are_current",
                        },
                        {
                            "need_id": "sleeve_conflict_resolution",
                            "why": "Grandmaster needs sleeve priority conflicts resolved.",
                            "evidence_artifacts": [
                                "governance/health/market_pattern_feedback_latest.json"
                            ],
                            "success_signal": "prioritized_and_downshift_sleeves_reported",
                        },
                    ],
                    "must_not": ["submit_orders", "grant_live_authority"],
                },
                "hardening_invariants": {
                    "required_need_fields": [
                        "need_id",
                        "why",
                        "evidence_artifacts",
                        "success_signal",
                    ],
                    "minimum_need_count_per_tier": 2,
                    "master_and_grandmaster_views_required": True,
                    "master_and_grandmaster_must_not_lists_required": True,
                },
            },
        },
    )


def test_system_needs_operator_packet_explains_profitability_and_market_patterns(
    tmp_path: Path,
) -> None:
    health = tmp_path / "governance" / "health"
    _seed_paper_collection_controls(tmp_path)
    _seed_operator_brain_boundary(tmp_path)
    _seed_sleeve_strategy_contracts(tmp_path)
    _seed_master_grandmaster_success(tmp_path)
    _write_json(
        health / "profitability_self_assessment_latest.json",
        {
            "overall_status": "collecting",
            "system_statement": (
                "Candidate pc-test is guarded for paper collection; live-grade "
                "profitability cannot be estimated without schema-v2 post-cost outcomes."
            ),
            "candidate": {
                "candidate_id": "pc-test",
                "identity_consistent": True,
            },
            "scorecard": {
                "implementation": {"grade": "A+", "score": 100.0},
                "economic_evidence": {"grade": "F", "score": 22.727, "ready": False},
            },
            "measurement": {
                "candidate_post_cost_sample_count": 0,
                "candidate_post_cost_minimum_samples": 30,
                "candidate_observed_days": 0,
                "candidate_minimum_observed_days": 3,
                "candidate_independent_fill_records": 0,
                "candidate_independent_fill_minimum_records": 30,
                "positive_post_cost_lower_confidence_bound_95": False,
                "historical_active_book_net_pnl": -123.45,
                "historical_active_book_candidate_grade_eligible": False,
            },
            "needs": [
                {
                    "blocker": "candidate_post_cost_observations_collecting",
                    "exact_file": "governance/health/paper_performance_report_latest.json",
                    "exact_shard": "current_candidate.schema_v2_trade_deltas",
                    "command": [
                        "./scripts/ops/opsctl.sh",
                        "paper-performance",
                        "--week-days",
                        "7",
                        "--json",
                    ],
                    "expected_impact": "Collects post-cost outcomes for the current candidate.",
                    "risk_level": "none",
                    "when_to_stop": "candidate_post_cost_sample_count reaches 30",
                }
            ],
        },
    )
    _write_json(
        health / "market_pattern_feedback_latest.json",
        {
            "overall_status": "ready",
            "pattern_count": 2,
            "observable_dimension_count": 10,
            "patterns": [
                {
                    "pattern_id": "defensive_high_vol_chop",
                    "label": "Defensive high-volatility chop",
                    "strength": 0.84,
                    "confidence": 0.74,
                    "direction": "mixed",
                },
                {
                    "pattern_id": "symbol_specific_evidence_gap",
                    "label": "Symbol-specific evidence gap",
                    "strength": 0.7,
                    "confidence": 0.6,
                    "direction": "unknown",
                },
            ],
            "dominant_patterns": [
                {
                    "pattern_id": "defensive_high_vol_chop",
                    "label": "Defensive high-volatility chop",
                    "strength": 0.84,
                    "confidence": 0.74,
                    "direction": "mixed",
                }
            ],
            "sleeve_feedback": [
                {
                    "sleeve": "volatility",
                    "paper_sampling_posture": "prioritize_bounded_paper_sampling",
                    "boost_score": 1.0,
                    "caution_score": 0.1,
                    "context_score": 0.0,
                    "pattern_ids": ["defensive_high_vol_chop"],
                    "collection_focus": ["post_cost_by_regime"],
                },
                {
                    "sleeve": "intraday",
                    "paper_sampling_posture": "downshift_or_context_first",
                    "boost_score": 0.0,
                    "caution_score": 1.0,
                    "context_score": 0.0,
                    "pattern_ids": ["defensive_high_vol_chop"],
                    "collection_focus": ["avoid_forced_entries"],
                },
            ],
            "profitability_evidence_gaps": {
                "candidate_post_cost_sample_count": 0,
                "candidate_post_cost_minimum_samples": 30,
                "candidate_observed_days": 0,
                "candidate_minimum_observed_days": 3,
            },
            "platform_feedback_contract": {
                "can_route_paper_collection_priority": True,
                "can_change_live_execution": False,
                "can_claim_profitability": False,
            },
            "paper_only": True,
            "live_execution_allowed": False,
            "profitability_claim_allowed": False,
        },
    )

    payload = src.build_payload(tmp_path)

    communication = payload["operator_communication"]
    assert (
        "not profitable on evidence yet" in communication["direct_profitability_answer"]
    )
    assert (
        "current_candidate_post_cost_samples=0/30"
        in communication["why_not_profitable_yet"]
    )
    assert "candidate_observed_days=0/3" in communication["why_not_profitable_yet"]
    assert "symbol_level_driver_evidence_gap" in communication["why_not_profitable_yet"]
    assert payload["current_candidate"]["candidate_id"] == "pc-test"
    assert (
        payload["profitability_requirements"][0]["need"]
        == "current_candidate_post_cost_outcomes"
    )
    assert payload["market_pattern_readout"]["pattern_count"] == 2
    assert (
        payload["paper_evidence_collection_readout"]["safe_for_evidence_collection"]
        is True
    )
    assert payload["frames_of_reference"]["market_pattern_feedback"][
        "dominant_pattern_ids"
    ] == ["defensive_high_vol_chop"]
    assert (
        payload["frames_of_reference"]["paper_evidence_collection_controls"][
            "safe_for_evidence_collection"
        ]
        is True
    )
    assert (
        communication["paper_evidence_collection_readout"][
            "safe_for_evidence_collection"
        ]
        is True
    )
    assert communication["market_pattern_readout"]["prioritized_sleeves"] == [
        "volatility"
    ]
    assert communication["market_pattern_readout"]["downshift_sleeves"] == ["intraday"]
    sleeve_chars = communication["sleeve_characteristics_readout"]
    assert sleeve_chars["characterized_sleeve_count"] == 2
    assert sleeve_chars["objective_class_count"] == 2
    assert sleeve_chars["prioritized_sleeves"][0]["sleeve_id"] == "volatility"
    assert (
        sleeve_chars["prioritized_sleeves"][0]["primary_character"]
        == "volatility_surface_and_tail_pricing"
    )
    assert sleeve_chars["downshift_sleeves"][0]["sleeve_id"] == "intraday"
    assert sleeve_chars["downshift_sleeves"][0]["ops_brain_dependencies"] == [
        "data_freshness",
        "writer_queue",
    ]
    assert sleeve_chars["hardening"]["overall_status"] == "ready"
    assert sleeve_chars["hardening"]["failed_check_count"] == 0
    strategy = communication["strategy_organization_readout"]
    assert strategy["group_count"] == 2
    assert strategy["mapped_sleeve_count"] == 2
    assert strategy["trading_sleeve_count"] == 2
    assert strategy["unmapped_trading_sleeves"] == []
    assert strategy["hardening"]["overall_status"] == "ready"
    assert strategy["group_ids"] == [
        "volatility_event_convexity",
        "execution_microstructure_liquidity",
    ]
    assert strategy["sleeve_group_map"][0]["sleeve_id"] == "intraday"
    assert strategy["sleeve_group_map"][0]["strategy_groups"] == [
        "execution_microstructure_liquidity"
    ]
    success = communication["master_grandmaster_success_readout"]
    assert success["master_need_count"] == 2
    assert success["grandmaster_need_count"] == 2
    assert success["hardening"]["overall_status"] == "ready"
    assert success["master_bot"]["view"] == (
        "per_sleeve_local_optimizer_and_evidence_curator"
    )
    assert success["grandmaster_bot"]["view"] == (
        "cross_sleeve_allocator_referee_and_policy_coordinator"
    )
    assert (
        payload["frames_of_reference"]["strategy_organization"]["hardening"][
            "overall_status"
        ]
        == "ready"
    )
    assert (
        payload["frames_of_reference"]["master_grandmaster_success_needs"]["hardening"][
            "overall_status"
        ]
        == "ready"
    )
    boundary = communication["brain_boundary_readout"]
    assert boundary["boundary_id"] == "trading_brain_ops_brain_v1"
    assert (
        boundary["trading_brain"]["current_posture"]
        == "freeze_except_bug_fixes_while_candidate_collects_evidence"
    )
    assert (
        boundary["ops_brain"]["current_posture"]
        == "safe_to_harden_while_trading_brain_collects_evidence"
    )
    assert boundary["hardening"]["overall_status"] == "ready"
    assert boundary["hardening"]["failed_check_count"] == 0
    assert boundary["trading_category_count"] == 4
    assert boundary["ops_category_count"] == 5
    assert boundary["trading_categories"] == [
        "market_decision",
        "strategy_logic",
        "risk_execution_policy",
        "candidate_evidence",
    ]
    assert boundary["ops_categories"] == [
        "storage_runtime",
        "writer_queue",
        "data_freshness",
        "observability_reporting",
        "process_recovery",
    ]
    assert boundary["routing_matrix"]["safe_now"] == [
        "ops_brain.storage_runtime",
        "ops_brain.writer_queue",
        "ops_brain.data_freshness",
        "ops_brain.observability_reporting",
        "ops_brain.process_recovery",
    ]
    assert (
        "do_not_change_trading_brain_while_current_candidate_collects_evidence"
        in communication["what_not_to_do"]
    )
    assert (
        "do_not_treat_strategy_groups_as_trade_authority"
        in communication["what_not_to_do"]
    )
    assert (
        "do_not_treat_master_grandmaster_success_needs_as_live_authority"
        in communication["what_not_to_do"]
    )
    assert (
        communication["priority_ladder"][0]["blocker"]
        == "candidate_post_cost_observations_collecting"
    )
    assert any(
        row["blocker"] == "symbol_level_driver_attribution_collecting"
        for row in communication["priority_ladder"]
    )

    markdown = src.render_operator_needs_markdown(payload)
    assert "# System Needs" in markdown
    assert "## Why Not Profitable Yet" in markdown
    assert "## Trading/Ops Boundary" in markdown
    assert "Trading Brain is frozen except bug fixes" in markdown
    assert "Trading `market_decision`" in markdown
    assert "Ops `storage_runtime`" in markdown
    assert "Boundary Hardening: `ready`" in markdown
    assert "Safe Now Categories:" in markdown
    assert "## Sleeve Characteristics" in markdown
    assert "Sleeve Hardening: `ready`" in markdown
    assert "volatility_surface_and_tail_pricing" in markdown
    assert "microstructure_execution_edge" in markdown
    assert "## Strategy Organization" in markdown
    assert "Strategy Hardening: `ready`" in markdown
    assert "volatility_event_convexity" in markdown
    assert "## Master/Grandmaster Needs" in markdown
    assert "Master/Grandmaster Hardening: `ready`" in markdown
    assert "sleeve_strategy_taxonomy" in markdown
    assert "cross_sleeve_correlation_and_exposure_map" in markdown
    assert "## Market Patterns" in markdown
    assert "defensive_high_vol_chop" in markdown


def test_system_needs_flags_missing_market_pattern_feedback(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _seed_paper_collection_controls(tmp_path)
    _seed_operator_brain_boundary(tmp_path)
    _seed_sleeve_strategy_contracts(tmp_path)
    _seed_master_grandmaster_success(tmp_path)
    _write_json(
        health / "profitability_self_assessment_latest.json",
        {
            "overall_status": "collecting",
            "candidate": {"candidate_id": "pc-test", "identity_consistent": True},
            "scorecard": {
                "implementation": {"grade": "A+", "score": 100.0},
                "economic_evidence": {"grade": "F", "score": 20.0, "ready": False},
            },
            "measurement": {"candidate_post_cost_sample_count": 0},
            "needs": [],
        },
    )

    payload = src.build_payload(tmp_path)

    assert any(
        row["blocker"] == "market_pattern_feedback_missing" for row in payload["needs"]
    )
    assert (
        payload["operator_communication"]["market_pattern_readout"]["overall_status"]
        == "missing"
    )


def test_system_needs_flags_operator_brain_boundary_hardening_failure(
    tmp_path: Path,
) -> None:
    _seed_paper_collection_controls(tmp_path)
    _seed_operator_brain_boundary(tmp_path)
    _seed_sleeve_strategy_contracts(tmp_path)
    _seed_master_grandmaster_success(tmp_path)
    path = tmp_path / "config" / "system_role_contracts_v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["operator_brain_boundary"]["routing_matrix"]["safe_now"] = [
        "trading_brain.strategy_logic"
    ]
    _write_json(path, payload)

    result = src.build_payload(tmp_path)

    boundary = result["operator_communication"]["brain_boundary_readout"]
    assert boundary["hardening"]["overall_status"] == "needs_action"
    assert "safe_now_routes_are_ops_only" in boundary["hardening"]["failed_checks"]
    assert any(
        row["blocker"] == "operator_brain_boundary_hardening_failed"
        for row in result["needs"]
    )


def test_system_needs_flags_sleeve_characteristics_hardening_failure(
    tmp_path: Path,
) -> None:
    _seed_paper_collection_controls(tmp_path)
    _seed_operator_brain_boundary(tmp_path)
    _seed_sleeve_strategy_contracts(tmp_path)
    _seed_master_grandmaster_success(tmp_path)
    path = tmp_path / "config" / "sleeve_strategy_contracts_v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["sleeve_characteristics"]["authority"]["can_claim_profitability"] = True
    _write_json(path, payload)

    result = src.build_payload(tmp_path)

    sleeve_chars = result["operator_communication"]["sleeve_characteristics_readout"]
    assert sleeve_chars["hardening"]["overall_status"] == "needs_action"
    assert "authority_is_metadata_only" in sleeve_chars["hardening"]["failed_checks"]
    assert any(
        row["blocker"] == "sleeve_characteristics_hardening_failed"
        for row in result["needs"]
    )


def test_system_needs_flags_strategy_organization_hardening_failure(
    tmp_path: Path,
) -> None:
    _seed_paper_collection_controls(tmp_path)
    _seed_operator_brain_boundary(tmp_path)
    _seed_sleeve_strategy_contracts(tmp_path)
    _seed_master_grandmaster_success(tmp_path)
    path = tmp_path / "config" / "sleeve_strategy_contracts_v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["strategy_organization"]["groups"][0]["primary_sleeves"] = []
    _write_json(path, payload)

    result = src.build_payload(tmp_path)

    strategy = result["operator_communication"]["strategy_organization_readout"]
    assert strategy["hardening"]["overall_status"] == "needs_action"
    assert (
        "strategy_groups_have_required_fields" in strategy["hardening"]["failed_checks"]
    )
    assert "every_trading_sleeve_is_grouped" in strategy["hardening"]["failed_checks"]
    assert strategy["unmapped_trading_sleeves"] == ["volatility"]
    assert any(
        row["blocker"] == "strategy_organization_hardening_failed"
        for row in result["needs"]
    )


def test_system_needs_flags_master_grandmaster_success_hardening_failure(
    tmp_path: Path,
) -> None:
    _seed_paper_collection_controls(tmp_path)
    _seed_operator_brain_boundary(tmp_path)
    _seed_sleeve_strategy_contracts(tmp_path)
    _seed_master_grandmaster_success(tmp_path)
    path = tmp_path / "config" / "master_grandmaster_evidence_v2.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["coordination_success_needs"]["authority"]["can_submit_order"] = True
    payload["coordination_success_needs"]["grandmaster_bot"]["needs"][0][
        "success_signal"
    ] = ""
    _write_json(path, payload)

    result = src.build_payload(tmp_path)

    success = result["operator_communication"]["master_grandmaster_success_readout"]
    assert success["hardening"]["overall_status"] == "needs_action"
    assert "authority_is_metadata_only" in success["hardening"]["failed_checks"]
    assert "success_needs_have_required_fields" in success["hardening"]["failed_checks"]
    assert any(
        row["blocker"] == "master_grandmaster_success_needs_hardening_failed"
        for row in result["needs"]
    )
