from __future__ import annotations

import importlib
import hashlib
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


def _write_candidate_event_chain(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    previous_hash = ""
    rows: list[str] = []
    for event in events:
        row = {
            "schema_version": 1,
            "previous_event_hash": previous_hash,
            **event,
        }
        encoded = json.dumps(
            row,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        event_hash = hashlib.sha256(encoded).hexdigest()
        row["event_hash"] = event_hash
        rows.append(json.dumps(row, ensure_ascii=True, sort_keys=True))
        previous_hash = event_hash
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


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
        health / "collector_capability_control_latest.json",
        {
            "economic_context_contract": {
                "policy": {
                    "contract_id": "sleeve_economic_context_v1",
                    "minimum_distinct_selected_sources": 2,
                },
                "family_count": 15,
                "configured_family_count": 15,
                "ready_family_count": 15,
                "runtime_route_count": 104,
                "runtime_ready_route_count": 104,
                "selected_source_count": 4,
                "selected_source_ids": [
                    "official_macro_context",
                    "central_bank_cross_source_context",
                    "bond_reference_context",
                    "macro_cross_asset_context",
                ],
                "context_changes_strategy_signal": False,
                "paper_execution_authority": False,
                "live_execution_authority": False,
                "automatic_promotion_authority": False,
                "economic_profitability_grade_authority": False,
                "contract_receipt_sha256": "economic-context-receipt",
            },
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
    assert payload["grades"]["economic_context_source_grade"] == "A+"
    assert payload["grades"]["economic_context_source_ready"] is True
    assert payload["grades"]["economic_context_ready_families"] == 15
    assert payload["grades"]["economic_context_ready_runtime_routes"] == 104
    assert payload["claims"]["economic_context_is_profitability_evidence"] is False
    assert payload["measurement"]["candidate_post_cost_sample_count"] == 0
    assert payload["measurement"]["historical_active_book_net_pnl"] == -100.0
    assert payload["measurement"]["historical_active_book_candidate_grade_eligible"] is False
    assert payload["claims"]["profitability_guaranteed"] is False
    assert payload["claims"]["live_execution_authority"] is False
    assert len(payload["eight_lane_program"]) == 8
    assert "candidate_post_cost_observations_collecting" in {
        row["blocker"] for row in payload["needs"]
    }


def test_main_preserves_json_when_operator_markdown_storage_fails(
    tmp_path: Path, monkeypatch
) -> None:
    out_file = (
        tmp_path / "governance" / "health" / "profitability_self_assessment_latest.json"
    )
    blocked_parent = tmp_path / "exports" / "reports" / "operator"
    blocked_parent.parent.mkdir(parents=True, exist_ok=True)
    blocked_parent.write_text("not a directory", encoding="utf-8")
    markdown_out = blocked_parent / "profitability_self_assessment_latest.md"

    def _fake_payload(_project_root, *, config_path=None):
        return {
            "ok": True,
            "overall_status": "collecting",
            "grades": {},
            "measurement": {},
            "needs": [],
        }

    monkeypatch.setattr(assessment, "build_payload", _fake_payload)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "profitability_self_assessment.py",
            "--project-root",
            str(tmp_path),
            "--out-file",
            str(out_file),
            "--markdown-out",
            str(markdown_out),
            "--json",
        ],
    )

    rc = assessment.main()
    payload = json.loads(out_file.read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["operator_markdown_report"]["attempted"] is True
    assert payload["operator_markdown_report"]["available"] is False
    assert "FileExistsError" in payload["operator_markdown_report"]["error"]
    assert (
        "profitability_self_assessment_markdown_storage_unavailable"
        in payload["warnings"]
    )


def test_assessment_uses_verified_generations_for_bounded_developmental_actions(
    tmp_path: Path,
) -> None:
    _seed_candidate_assessment(tmp_path)
    config_path = tmp_path / "config" / "profitability_self_assessment_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["soak_contract"] = {
        "developmental_change_learning_enabled": True,
        "candidate_event_log_path": "governance/evidence/production_candidate_events.jsonl",
        "minimum_developmental_post_cost_samples": 30,
        "minimum_developmental_observed_days": 2,
        "clean_720_hour_live_promotion_gate_unchanged": True,
        "bounded_paper_actions": [
            "collect_current_candidate_post_cost_outcomes",
            "refresh_candidate_counterfactual_replay",
            "acquire_candidate_independent_fills",
            "maintain_candidate_bound_weak_sleeve_containment",
            "prioritize_loss_and_missed_opportunity_labels",
        ],
    }
    _write_json(config_path, config)
    candidate_path = (
        tmp_path / "governance" / "runtime" / "production_candidate_state.json"
    )
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["generation"] = 3
    candidate["accepted_at_utc"] = "2026-08-21T12:00:00+00:00"
    _write_json(candidate_path, candidate)
    _write_candidate_event_chain(
        tmp_path / "governance" / "evidence" / "production_candidate_events.jsonl",
        [
            {
                "event_type": "candidate_change_accepted",
                "candidate_id": "candidate-old-positive",
                "generation": 1,
                "timestamp_utc": "2026-08-18T12:00:00+00:00",
                "changed_scopes": ["strategy"],
                "change_reason": "accepted positive developmental cohort",
            },
            {
                "event_type": "candidate_change_accepted",
                "candidate_id": "candidate-old-negative",
                "generation": 2,
                "timestamp_utc": "2026-08-20T00:00:00+00:00",
                "changed_scopes": ["strategy", "execution"],
                "change_reason": "accepted negative developmental cohort",
            },
            {
                "event_type": "candidate_change_accepted",
                "candidate_id": "candidate-1",
                "generation": 3,
                "timestamp_utc": "2026-08-21T12:00:00+00:00",
                "changed_scopes": ["operations"],
                "change_reason": "current candidate",
            },
        ],
    )
    performance_path = (
        tmp_path / "governance" / "health" / "paper_performance_latest.json"
    )
    performance = json.loads(performance_path.read_text(encoding="utf-8"))
    performance["developmental_generation_flows"] = {
        "generation_flows": [
            {
                "candidate_id": "candidate-old-positive",
                "candidate_generation": 1,
                "candidate_generation_consistent": True,
                "developmental_attribution_eligible": True,
                "sample_count": 45,
                "observed_days": 2,
                "first_observation_utc": "2026-08-18T13:00:00+00:00",
                "last_observation_utc": "2026-08-19T20:00:00+00:00",
                "post_cost_pnl_delta_total": 8.5,
            },
            {
                "candidate_id": "candidate-old-negative",
                "candidate_generation": 2,
                "candidate_generation_consistent": True,
                "developmental_attribution_eligible": True,
                "sample_count": 38,
                "observed_days": 2,
                "first_observation_utc": "2026-08-20T01:00:00+00:00",
                "last_observation_utc": "2026-08-21T11:00:00+00:00",
                "post_cost_pnl_delta_total": -4.25,
            },
            {
                "candidate_id": "candidate-not-in-chain",
                "candidate_generation": 99,
                "candidate_generation_consistent": True,
                "developmental_attribution_eligible": True,
                "sample_count": 100,
                "observed_days": 4,
                "first_observation_utc": "2026-08-17T01:00:00+00:00",
                "last_observation_utc": "2026-08-17T20:00:00+00:00",
                "post_cost_pnl_delta_total": 100.0,
            },
        ],
        "unbound_schema_v2_sample_count": 7,
        "metadata_conflict_count": 0,
    }
    _write_json(performance_path, performance)

    payload = assessment.build_payload(tmp_path)
    learning = payload["developmental_soak_learning"]

    assert learning["status"] == "ready"
    assert learning["candidate_event_chain"]["valid"] is True
    assert learning["accepted_generation_count"] == 3
    assert learning["attributable_generation_count"] == 2
    assert learning["mature_developmental_generation_count"] == 2
    assert learning["observed_positive_delta_generation_count"] == 1
    assert learning["observed_negative_delta_generation_count"] == 1
    assert learning["unbound_schema_v2_sample_count"] == 7
    assert learning["unmatched_generation_flows"][0]["candidate_id"] == "candidate-not-in-chain"
    assert learning["policy"]["historical_generations_grade_current_candidate"] is False
    assert learning["policy"]["clean_720_hour_live_promotion_gate_unchanged"] is True
    action_ids = {
        row["action_id"] for row in learning["bounded_paper_action_plan"]
    }
    assert "collect_current_candidate_post_cost_outcomes" in action_ids
    assert "refresh_candidate_counterfactual_replay" in action_ids
    assert "maintain_candidate_bound_weak_sleeve_containment" in action_ids
    assert "prioritize_loss_and_missed_opportunity_labels" in action_ids
    assert all(row["paper_only"] for row in learning["bounded_paper_action_plan"])
    assert all(
        row["force_trade_allowed"] is False
        and row["loss_recovery_size_increase_allowed"] is False
        and row["direct_threshold_loosen_allowed"] is False
        and row["live_execution_allowed"] is False
        and row["promotion_authority"] is False
        for row in learning["bounded_paper_action_plan"]
    )
    assert payload["claims"]["accepted_generation_history_is_live_promotion_evidence"] is False
    assert payload["control_contract"]["clean_720_hour_live_promotion_gate_unchanged"] is True


def test_developmental_attribution_rejects_outcomes_after_generation_ended(
    tmp_path: Path,
) -> None:
    _write_candidate_event_chain(
        tmp_path / "governance" / "evidence" / "production_candidate_events.jsonl",
        [
            {
                "event_type": "candidate_change_accepted",
                "candidate_id": "candidate-1",
                "generation": 1,
                "timestamp_utc": "2026-08-18T12:00:00+00:00",
            },
            {
                "event_type": "candidate_change_accepted",
                "candidate_id": "candidate-2",
                "generation": 2,
                "timestamp_utc": "2026-08-20T12:00:00+00:00",
            },
        ],
    )
    learning = assessment._developmental_soak_learning(
        tmp_path,
        policy={
            "soak_contract": {
                "developmental_change_learning_enabled": True,
                "minimum_developmental_post_cost_samples": 1,
                "minimum_developmental_observed_days": 1,
                "bounded_paper_actions": [],
            }
        },
        performance={
            "developmental_generation_flows": {
                "generation_flows": [
                    {
                        "candidate_id": "candidate-1",
                        "candidate_generation": 1,
                        "candidate_generation_consistent": True,
                        "developmental_attribution_eligible": True,
                        "sample_count": 10,
                        "observed_days": 1,
                        "first_observation_utc": "2026-08-18T13:00:00+00:00",
                        "last_observation_utc": "2026-08-20T13:00:00+00:00",
                        "post_cost_pnl_delta_total": 5.0,
                    }
                ]
            }
        },
        current_candidate_id="candidate-2",
        current_generation=2,
        current_candidate_samples=0,
        minimum_candidate_samples=30,
        missing_market_types=[],
        replay_tradeability_ready=True,
        weak_control_count=0,
        now=assessment.parse_iso_utc("2026-08-21T12:00:00+00:00"),
    )

    first = learning["generation_rows"][0]
    assert first["temporal_binding_valid"] is False
    assert first["developmental_attribution_eligible"] is False
    assert first["developmental_status"] == "rejected_outside_accepted_generation_window"
    assert learning["attributable_generation_count"] == 0
    assert learning["observed_positive_delta_generation_count"] == 0


def test_assessment_fails_closed_on_cross_candidate_performance(tmp_path: Path) -> None:
    _seed_candidate_assessment(tmp_path, performance_candidate="candidate-old")

    payload = assessment.build_payload(tmp_path)

    assert payload["overall_status"] == "blocked"
    assert payload["candidate_binding"]["identity_consistent"] is False
    assert payload["candidate_binding"]["mismatch_sources"] == ["paper_performance"]
    assert payload["needs"][0]["blocker"] == "candidate_identity_binding_incomplete"


def test_stale_optional_override_is_reported_and_ignored_after_candidate_change(
    tmp_path: Path,
) -> None:
    _seed_candidate_assessment(tmp_path)
    override_path = (
        tmp_path
        / "governance"
        / "health"
        / "calibration_abstention_overrides_latest.json"
    )
    override = json.loads(override_path.read_text(encoding="utf-8"))
    override["candidate_binding"]["candidate_id"] = "candidate-old"
    override["candidate_binding"]["valid_candidate_id"] = "candidate-old"
    _write_json(override_path, override)

    payload = assessment.build_payload(tmp_path)

    assert payload["overall_status"] == "collecting"
    assert payload["candidate_binding"]["identity_consistent"] is True
    assert payload["candidate_binding"]["identity_complete"] is True
    assert payload["candidate_binding"]["mismatch_sources"] == []
    assert payload["candidate_binding"]["optional_mismatch_sources"] == [
        "calibration_overrides"
    ]


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
            "developmental_soak_learning": {
                "status": "ready",
                "accepted_generation_count": 12,
                "attributable_generation_count": 4,
                "mature_developmental_generation_count": 2,
                "observed_negative_delta_generation_count": 1,
                "bounded_paper_action_plan": [{"action_id": "refresh_candidate_counterfactual_replay"}],
            },
            "claims": {
                "historical_loss_is_current_candidate_evidence": False,
                "accepted_generation_history_informs_developmental_actions": True,
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
    assert domain["developmental_learning_status"] == "ready"
    assert domain["attributable_generation_count"] == 4
    assert domain["accepted_generation_history_informs_developmental_actions"] is True
    assert domain["accepted_generation_history_is_live_promotion_evidence"] is False
    assert "candidate profitability candidate-1" in payload["self_summary"]
    assert "developmental_generations=4/12" in payload["self_summary"]
