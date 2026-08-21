from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

from scripts.ops import profitability_self_assessment as assessment
from scripts.ops import system_needs_intelligence
from scripts.ops import system_self_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = PROJECT_ROOT / "core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

indicator_bot_common = importlib.import_module("core.indicator_bot_common")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _seed_candidate_assessment(project_root: Path, *, performance_candidate: str = "candidate-1") -> None:
    _write_json(
        project_root / "config" / "profitability_self_assessment_v1.json",
        {
            "policy_id": "candidate_bound_profitability_self_assessment_v1",
            "source_freshness_hours": {},
            "exit_learning": {
                "required_labels": [
                    "mae_bucket",
                    "mfe_bucket",
                    "exit_timing_bucket",
                    "post_entry_regime_bucket",
                ]
            },
            "execution_realism": {"minimum_independent_fills_per_market_type": 30},
            "income_sleeves": {
                "paper_confidence_threshold_uplift": 0.08,
                "paper_abstention_budget": 0.88,
            },
            "sizing": {
                "paper_entry_cap_norm": 0.25,
                "maximum_evidence_validated_scale_norm": 1.1,
            },
            "portfolio_allocation": {"minimum_independently_profitable_sleeves": 4},
        },
    )
    _write_json(
        project_root / "governance" / "runtime" / "production_candidate_state.json",
        {
            "candidate_id": "candidate-1",
            "generation": 1,
            "accepted_at_utc": "2026-08-21T12:00:00+00:00",
            "overall_sha256": "candidate-receipt",
            "live_execution_authority": False,
            "profitability_baseline": {"historical_net_pnl": -100.0},
        },
    )
    _write_json(
        project_root / "governance" / "health" / "paper_performance_latest.json",
        {
            "profitability_evidence_window": {"candidate_id": performance_candidate},
            "post_cost_expectancy": {
                "status": "no_schema_v2_trade_deltas",
                "sample_count": 0,
                "minimum_samples": 30,
                "evidence_sufficient": False,
            },
            "accounting_views": {
                "candidate_forward_flow": {
                    "candidate_id": performance_candidate,
                    "sample_count": 0,
                    "post_cost_pnl_delta_total": 0.0,
                },
                "active_book_snapshot": {
                    "ending_net_pnl_total": -100.0,
                    "candidate_grade_eligible": False,
                },
            },
        },
    )
    health = project_root / "governance" / "health"
    _write_json(
        health / "paper_execution_calibration_latest.json",
        {
            "candidate_binding": {"candidate_id": "candidate-1", "required": True},
            "independent_samples": 150,
            "by_market_kind": {
                market: {"independent_samples": 30}
                for market in ("EQUITY", "ETF", "OPTION", "FUTURE", "FOREX")
            },
        },
    )
    _write_json(
        health / "calibration_abstention_control_latest.json",
        {"recommendations": [], "family_recommendations": [], "overacting_count": 0, "underacting_count": 0},
    )
    _write_json(
        health / "calibration_abstention_overrides_latest.json",
        {
            "schema_version": 2,
            "candidate_binding": {
                "candidate_id": "candidate-1",
                "valid_candidate_id": "candidate-1",
                "valid_until_candidate_changes": True,
            },
            "bot_overrides": {},
            "family_overrides": {
                family: {
                    "mode": "tighten",
                    "acted_prob_threshold_uplift": 0.08,
                    "recommended_abstention_budget": 0.88,
                    "valid_candidate_id": "candidate-1",
                }
                for family in ("bond", "dividend")
            },
            "regime_overrides": {},
        },
    )
    _write_json(
        health / "profitability_evidence_firewall_latest.json",
        {
            "economic_evidence_grade": "F",
            "economic_evidence_score": 25.0,
            "economic_evidence_ready": False,
            "evidence_ready_control_count": 5,
            "control_count": 20,
            "allocation_proposal": {
                "ready": False,
                "qualified_sleeves": [],
                "qualified_sleeve_count": 0,
                "suggested_cash_weight": 1.0,
                "automatic_allocation_allowed": False,
                "thresholds": {"minimum_profitable_sleeves": 4},
            },
        },
    )
    _write_json(
        health / "paper_profitability_control_latest.json",
        {
            "sleeve_strategy_profitability_scaling_contract": {
                "candidate_binding": {"candidate_id": "candidate-1"},
                "global_entry_size_cap_norm": 0.25,
                "maximum_above_baseline_entry_size_multiplier_norm": 1.1,
                "entry_only": True,
                "keep_sells_and_reduce_only_paths_open": True,
                "scale_up_ready": False,
                "above_baseline_ready_count": 0,
            },
            "profit_harvest_regret_replay_contract": {"mode": "candidate_exit_learning"},
        },
    )
    _write_json(
        health / "counterfactual_replay_latest.json",
        {"top_candidates": [{"tradeability_floor": 0.6, "max_conflict_norm": 0.5}]},
    )
    _write_json(
        health / "live_money_readiness_contract_latest.json",
        {"overall_status": "blocked", "live_money_locked": True, "sections": []},
    )
    _write_json(
        project_root / "config" / "institutional_decision_flow_v1.json",
        {
            "active_paper_control": {"enabled": True},
            "profile_policy_map": {"bond": "long_horizon_income", "dividend": "long_horizon_income"},
            "sleeve_policy_families": {"long_horizon_income": {}, "balanced_directional": {}},
            "stages": [
                {"stage_id": "04_consensus_and_regime"},
                {"stage_id": "06_execution_feasibility"},
                {"stage_id": "07_portfolio_fit"},
            ],
        },
    )
    _write_json(
        project_root / "config" / "profitability_evidence_firewall_v1.json",
        {
            "entry_quality": {"unknown_evidence_fails_closed": True},
            "counterfactual_labels": [
                "mae_bucket",
                "mfe_bucket",
                "exit_timing_bucket",
                "post_entry_regime_bucket",
            ],
        },
    )
    _write_json(
        project_root / "config" / "broker_capability_contracts_v1.json",
        {
            "brokers": {
                "schwab": {
                    "paper": {
                        "asset_classes": ["EQUITY", "ETF", "OPTION", "FUTURE", "FOREX"]
                    }
                }
            }
        },
    )


def test_assessment_separates_historical_debt_from_current_candidate(tmp_path: Path) -> None:
    _seed_candidate_assessment(tmp_path)

    payload = assessment.build_payload(tmp_path)

    assert payload["overall_status"] == "collecting"
    assert payload["assessment_status"] == "ready"
    assert payload["grades"]["implementation_grade"] == "A+"
    assert payload["grades"]["economic_evidence_grade"] == "F"
    assert payload["measurement"]["candidate_post_cost_sample_count"] == 0
    assert payload["measurement"]["historical_active_book_net_pnl"] == -100.0
    assert payload["measurement"]["historical_active_book_candidate_grade_eligible"] is False
    assert payload["claims"]["profitability_guaranteed"] is False
    assert payload["claims"]["live_execution_authority"] is False
    assert len(payload["eight_lane_program"]) == 8
    assert "candidate_post_cost_observations_collecting" in {
        row["blocker"] for row in payload["needs"]
    }


def test_assessment_fails_closed_on_cross_candidate_performance(tmp_path: Path) -> None:
    _seed_candidate_assessment(tmp_path, performance_candidate="candidate-old")

    payload = assessment.build_payload(tmp_path)

    assert payload["overall_status"] == "blocked"
    assert payload["candidate_binding"]["identity_consistent"] is False
    assert payload["candidate_binding"]["mismatch_sources"] == ["paper_performance"]
    assert payload["needs"][0]["blocker"] == "candidate_identity_binding_incomplete"


def test_candidate_bound_override_is_ignored_after_candidate_changes(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "governance" / "runtime" / "production_candidate_state.json",
        {"candidate_id": "candidate-new"},
    )
    _write_json(
        tmp_path / "governance" / "health" / "calibration_abstention_overrides_latest.json",
        {
            "schema_version": 2,
            "candidate_binding": {
                "valid_candidate_id": "candidate-old",
                "valid_until_candidate_changes": True,
            },
            "family_overrides": {
                "dividend": {"mode": "tighten", "acted_prob_threshold_uplift": 0.08}
            },
        },
    )

    threshold, meta = indicator_bot_common._resolve_learned_acted_threshold(
        tmp_path,
        run_tag="income_bot",
        family="dividend",
        base_threshold=0.65,
    )

    assert threshold == 0.65
    assert meta["payload_applicable"] is False
    assert meta["rejected_sources"][0]["reason"] == "candidate_missing_or_changed"


def test_system_needs_prefers_candidate_assessment_over_historical_burn_down(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "profitability_self_assessment_latest.json",
        {
            "overall_status": "ready",
            "candidate_binding": {
                "candidate_id": "candidate-1",
                "identity_consistent": True,
                "identity_complete": True,
            },
            "grades": {"implementation_grade": "A+", "economic_evidence_grade": "F"},
            "measurement": {
                "candidate_post_cost_sample_count": 0,
                "historical_active_book_net_pnl": -100.0,
                "historical_active_book_candidate_grade_eligible": False,
            },
            "needs": [
                {
                    "blocker": "candidate_post_cost_observations_collecting",
                    "exact_file": "governance/health/paper_performance_latest.json",
                    "exact_shard": "post_cost_expectancy",
                    "command": ["./scripts/ops/opsctl.sh", "paper-performance", "--json"],
                }
            ],
        },
    )
    _write_json(
        health / "paper_profitability_control_latest.json",
        {
            "raw_profitability_grade": "D",
            "financial_profitability_grade": "D",
            "a_plus_target_contract": {"current": {"net_pnl": -100.0}},
        },
    )

    payload = system_needs_intelligence.build_payload(
        tmp_path,
        fix_log_path=health / "system_needs_fix_log.jsonl",
    )
    blockers = {row["blocker"] for row in payload["needs"]}

    assert "candidate_post_cost_observations_collecting" in blockers
    assert "raw_profitability_burn_down" not in blockers
    assert payload["frames_of_reference"]["raw_profitability_recovery"]["historical_context_only"] is True


def test_self_model_consumes_candidate_profitability_statement(tmp_path: Path) -> None:
    _write_json(tmp_path / "config" / "profitability_self_assessment_v1.json", {"schema_version": 1})
    _write_json(
        tmp_path / "governance" / "health" / "profitability_self_assessment_latest.json",
        {
            "overall_status": "collecting",
            "assessment_status": "ready",
            "system_statement": "Current candidate needs post-cost observations.",
            "candidate_binding": {
                "candidate_id": "candidate-1",
                "identity_consistent": True,
                "identity_complete": True,
            },
            "grades": {
                "implementation_grade": "A+",
                "economic_evidence_grade": "F",
                "economic_evidence_ready": False,
            },
            "measurement": {"candidate_post_cost_sample_count": 0},
            "claims": {
                "historical_loss_is_current_candidate_evidence": False,
                "live_execution_authority": False,
            },
            "needs": [{"blocker": "candidate_post_cost_observations_collecting"}],
        },
    )

    payload = system_self_model.build_payload(tmp_path)
    domain = payload["awareness_domains"]["profitability_awareness"]

    assert domain["status"] == "advisory"
    assert domain["assessment_status"] == "ready"
    assert domain["candidate_evidence_status"] == "collecting"
    assert domain["candidate_id"] == "candidate-1"
    assert domain["economic_evidence_grade"] == "F"
    assert "candidate profitability candidate-1" in payload["self_summary"]
