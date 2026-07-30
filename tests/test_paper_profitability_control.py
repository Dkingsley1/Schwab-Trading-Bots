import importlib.util
import json
from collections import Counter
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "ops" / "paper_profitability_control.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("paper_profitability_control", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load paper_profitability_control")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")


def test_paper_profitability_control_builds_profile_and_strategy_brakes(tmp_path: Path) -> None:
    module = _load_module()
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "paper_performance_latest.json",
        {
            "ok": True,
            "day": "20260523",
            "history_daily_series": [
                {
                    "day_utc": "20260523",
                    "executions": 1200,
                    "ending_net_pnl_total": -1400.0,
                    "change_vs_previous_day": -1800.0,
                }
            ],
            "sleeve_latest": [
                {
                    "profile": "intraday_aggressive",
                    "executions": 1200,
                    "win_rate": 0.08,
                    "ending_realized_pnl_total": 120.0,
                    "ending_unrealized_pnl_total": -1520.0,
                    "ending_net_pnl_total": -1400.0,
                    "losing_strategy_count": 5,
                    "winning_strategy_count": 0,
                    "top_loss_causes": [
                        {"cause": "source_quality:low", "count": 8, "loss_total": 900.0},
                        {"cause": "fill_quality:unknown", "count": 8, "loss_total": 900.0},
                        {"cause": "event_proximity:low", "count": 8, "loss_total": 900.0},
                        {"cause": "conflict:low", "count": 8, "loss_total": 900.0},
                        {"cause": "session:premarket", "count": 8, "loss_total": 900.0},
                    ],
                    "top_losing_strategies": [
                        {
                            "strategy": "paper_mirror::brain_refinery_v48_position_1m_3m",
                            "ending_net_pnl_total": -550.0,
                        }
                    ],
                }
            ],
        },
    )
    _write_json(health / "training_quality_control_latest.json", {"training_quality_score": 77.0})

    payload = module.build_payload(tmp_path)
    runtime = module.build_runtime_control_payload(payload)

    profile = payload["active_profile_controls"]["intraday_aggressive"]
    training_contract = payload["master_grandmaster_training_contract"]
    assert payload["overall_status"] == "protective_tightening"
    assert payload["upgrade_lane_count"] == 10
    assert {row["lane"] for row in payload["profitability_upgrade_lanes"]} == set(module.UPGRADE_LANE_IDS)
    assert training_contract["active"] is True
    assert "grand_master_bot" in training_contract["trainable_targets"]
    assert training_contract["sample_weight_policy"]["paper_loss_hard_negative_multiplier"] > 1.0
    accuracy_contract = training_contract["sub_bot_accuracy_target_contract"]
    assert accuracy_contract["desired_out_of_sample_accuracy_band"] == {"min": 0.80, "max": 0.90}
    assert accuracy_contract["target_is_not_forced"] is True
    assert "future_leakage_or_same_bar_outcome_feature_detected" in accuracy_contract["reject_if"]
    assert runtime["master_grandmaster_training_contract"]["recommended_training_mode"] == "master_profitability_canary"
    assert runtime["sub_bot_accuracy_target_contract"]["max_train_test_accuracy_gap"] == 0.08
    assert profile["action"] == "quarantine_new_entries"
    assert profile["control_posture_grade"] == "A+"
    assert profile["profit_grade"] in {"D", "F"}
    assert profile["outcome_weighted_training"]["active"] is True
    assert profile["dynamic_sizing"]["paper_profitability_size_multiplier_norm"] <= 0.10
    assert profile["regime_specific_promotion"]["promotion_status"] == "paper_only_retest"
    assert profile["loser_quarantine"]["active"] is True
    assert profile["loser_quarantine"]["block_new_entries"] is True
    assert profile["exit_intelligence"]["active"] is True
    assert profile["exit_intelligence"]["block_adds_while_unrealized_negative"] is True
    assert profile["exit_intelligence"]["drag_reduction_mode"] == "reduce_only"
    assert profile["execution_aware_alpha"]["active"] is True
    assert profile["portfolio_conflict_control"]["active"] is True
    assert profile["confirmation_bias_control"]["active"] is True
    assert profile["confirmation_bias_control"]["min_independent_evidence_channels"] >= 3
    assert profile["a_plus_plus_strengthening"]["control_grade"] == "A+"
    assert "three_profitable_refreshes" in profile["a_plus_plus_strengthening"]["required_before_reentry"]
    assert "no_repeated_loss_cause_in_recent_refresh" in profile["a_plus_plus_strengthening"]["required_before_reentry"]
    assert "source_quality" in profile["confirmation_bias_control"]["required_before_new_entry"]
    assert profile["thresholds"]["min_source_quality_norm"] >= 0.60
    assert profile["thresholds"]["min_execution_fitness_norm"] >= 0.62
    assert profile["thresholds"]["min_cross_asset_confirmation_norm"] >= 0.58
    assert profile["thresholds"]["min_event_proximity_norm"] > 0.0
    recurrence = profile["weak_sleeve_recurrence_guard"]
    assert recurrence["active"] is True
    assert recurrence["prevent_recurrence_ready"] is True
    assert recurrence["reentry_locked_until_cleared"] is True
    assert recurrence["required_profitable_refreshes_before_reentry"] >= 3
    assert recurrence["min_independent_evidence_channels"] >= 4
    assert "source_quality_passed" in recurrence["required_before_reentry"]
    assert "modeled_fill_quality_present" in recurrence["required_before_reentry"]
    assert "event_catalyst_confirmation_present" in recurrence["required_before_reentry"]
    assert "portfolio_conflict_clearance_present" in recurrence["required_before_reentry"]
    assert "session_gate_passed" in recurrence["required_before_reentry"]
    assert "block_when_source_quality_low_or_stale" in recurrence["runtime_blocks"]
    assert recurrence["session_gate"]["unknown_session_is_negative"] is True
    assert "session_quality" in recurrence["recurrent_loss_families"]
    hardening = payload["paper_profitability_hardening_contract"]
    assert hardening["action_count"] == len(module.PROFITABILITY_HARDENING_ACTIONS)
    assert hardening["new_entry_policy"]["block_quarantined_profiles"] is True
    assert hardening["unrealized_drag_policy"]["block_adds_while_drag_active"] is True
    assert hardening["evidence_policy"]["unknown_evidence_is_negative"] is True
    assert hardening["recurrence_policy"]["lock_reentry_on_repeated_loss_cause"] is True
    assert any(row["action_id"] == "stop_new_entries_in_worst_sleeves" for row in hardening["actions"])
    assert any(row["action_id"] == "lock_recurring_loss_cause_reentry" for row in hardening["actions"])
    scout_contract = payload["scout_collection_contract"]
    assert scout_contract["active"] is True
    assert "no_trade_counterfactual_outcome" in scout_contract["required_label_outputs"]
    assert payload["strategy_controls"][0]["bot_id"] == "brain_refinery_v48_position_1m_3m"
    assert payload["strategy_controls"][0]["block_new_entries"] is True
    assert payload["strategy_controls"][0]["upgrade_contracts"]["loser_quarantine"]["active"] is True
    assert payload["strategy_controls"][0]["upgrade_contracts"]["loser_quarantine"]["rehabilitation_required"] is True
    assert payload["strategy_controls"][0]["confirmation_bias_control"]["active"] is True
    rehab = payload["strategy_controls"][0]["rehabilitation_contract"]
    assert rehab["mode"] == "paper_only_rehabilitation"
    assert rehab["hypothesis"] == "conditional_market_fit_not_dead_strategy"
    assert rehab["session_gate"]["active"] is True
    assert rehab["session_gate"]["unknown_session_is_negative"] is True
    assert rehab["quality_gate"]["min_independent_evidence_channels"] >= 4
    assert "session_gate_passed" in rehab["required_before_reentry"]
    assert "source_fill_spread_quality_present" in rehab["required_before_reentry"]
    assert "strategy_reentry_retest_outcome" in rehab["required_label_outputs"]
    assert "session_calendar" in rehab["required_context"]
    assert "independent_evidence_channel_count" in payload["strategy_controls"][0]["data_intake_enrichment"]["required_label_outputs"]
    assert "session_gate_result" in payload["strategy_controls"][0]["data_intake_enrichment"]["required_label_outputs"]
    assert "paper_unrealized_drag_bucket" in payload["strategy_controls"][0]["data_intake_enrichment"]["required_label_outputs"]
    assert "session_gate_result" in payload["scout_collection_contract"]["required_label_outputs"]
    assert "repeated_loss_cause_cleared" in payload["scout_collection_contract"]["required_label_outputs"]
    assert "session_calendar" in payload["scout_collection_contract"]["required_context"]
    assert "intraday_aggressive" in runtime["profile_controls"]
    weak_strength = payload["weak_sleeve_a_plus_plus_strengthening_contract"]
    assert weak_strength["control_posture_grade"] == "A+"
    assert weak_strength["control_ready"] is True
    assert weak_strength["weak_profile_count"] == 1
    assert weak_strength["a_plus_plus_profile_count"] == 1
    assert weak_strength["profile_controls"][0]["recurrence_guard_ready"] is True
    recurrence_contract = payload["weak_sleeve_recurrence_guard_contract"]
    assert recurrence_contract["control_posture_grade"] == "A+"
    assert recurrence_contract["control_ready"] is True
    assert recurrence_contract["profile_count"] == 1
    assert "intraday_aggressive" in recurrence_contract["target_profiles"]
    assert "source_quality_gate" in recurrence_contract["required_family_gates"]
    assert "session_quality_gate" in recurrence_contract["required_family_gates"]
    assert "session_quality" in recurrence_contract["required_evidence_channels"]
    assert payload["weak_sleeve_systemic_weak_point_contract"]["active"] is False
    assert runtime["weak_sleeve_a_plus_plus_strengthening_contract"]["control_posture_grade"] == "A+"
    assert runtime["weak_sleeve_recurrence_guard_contract"]["control_posture_grade"] == "A+"
    assert runtime["global_runtime_policy"]["apply_weak_sleeve_a_plus_plus_strengthening_contract"] is True
    assert runtime["global_runtime_policy"]["apply_weak_sleeve_recurrence_guard"] is True
    assert runtime["global_runtime_policy"]["apply_weak_sleeve_recurrence_guard_contract"] is True
    assert runtime["upgrade_lane_count"] == 10
    assert runtime["global_runtime_policy"]["apply_dynamic_sizing"] is True
    assert runtime["global_runtime_policy"]["apply_confirmation_bias_control"] is True
    assert runtime["global_runtime_policy"]["apply_profitability_hardening"] is True
    expansion = payload["profitability_realization_expansion_contract"]
    assert expansion["mode"] == "profitability_realization_expansion_1_to_8"
    assert expansion["paper_only"] is True
    assert expansion["live_execution_allowed"] is False
    assert expansion["lever_count"] == 8
    assert set(expansion["lever_ids"]) == set(module.PROFITABILITY_REALIZATION_LEVERS)
    levers = {row["lever_id"]: row for row in expansion["levers"]}
    assert levers["stop_weak_sleeve_drag"]["active"] is True
    assert "intraday_aggressive" in levers["stop_weak_sleeve_drag"]["targets"]
    assert levers["scale_winning_sleeves"]["active"] is False
    assert levers["harvest_regret_control_lift"]["targets"]["target_regret_control_norm"] == 0.8
    assert levers["punitive_loss_attribution"]["active"] is True
    assert levers["unrealized_loser_training_debt"]["active"] is True
    assert levers["harvest_force_guard"]["active"] is True
    assert runtime["profitability_realization_expansion_contract"]["lever_count"] == 8
    assert runtime["global_runtime_policy"]["apply_profitability_realization_expansion_contract"] is True
    assert runtime["global_runtime_policy"]["apply_unrealized_loser_training_debt"] is True
    autopilot = payload["profitability_compounding_autopilot_contract"]
    assert autopilot["mode"] == "profitability_compounding_autopilot_v1"
    assert autopilot["paper_only"] is True
    assert autopilot["live_execution_allowed"] is False
    assert autopilot["action_count"] == len(module.PROFITABILITY_COMPOUNDING_AUTOPILOT_ACTIONS)
    assert autopilot["active_action_count"] >= 4
    assert autopilot["do_first"]
    queued = {row["action_id"]: row for row in autopilot["priority_queue"]}
    assert queued["freeze_weak_sleeve_fresh_adds"]["active"] is True
    assert queued["freeze_weak_sleeve_fresh_adds"]["targets"][0]["profile"] == "intraday_aggressive"
    assert queued["assign_unrealized_loser_training_debt"]["active"] is True
    assert queued["tighten_punitive_loss_attribution"]["active"] is True
    assert runtime["profitability_compounding_autopilot_contract"]["action_count"] == len(module.PROFITABILITY_COMPOUNDING_AUTOPILOT_ACTIONS)
    assert runtime["global_runtime_policy"]["apply_profitability_compounding_autopilot"] is True
    assert runtime["global_runtime_policy"]["follow_profitability_do_first_queue"] is True
    replay = payload["profit_harvest_regret_replay_contract"]
    assert replay["upgrade_layer"]["mode"] == "profit_harvest_replay_layer_v2"
    assert "trim_too_early_bucket" in replay["labels"]
    assert "runner_deserved_room_bucket" in replay["labels"]
    quant_admission = payload["quant_strategy_expansion_admission_contract"]
    assert quant_admission["mode"] == "quant_strategy_expansion_admission_v1"
    assert quant_admission["paper_only"] is True
    assert quant_admission["live_execution_allowed"] is False
    assert quant_admission["can_add_more_quant_strategies"] is True
    assert quant_admission["admission_state"] == "collection_only_selective"
    assert quant_admission["max_new_strategy_slots"] <= 4
    assert "volatility_risk_premium_harvesting" in quant_admission["approved_families"]
    assert "bermudan_exercise_monte_carlo_policy" in quant_admission["approved_families"]
    assert "intraday_aggressive" in quant_admission["blocked_profiles"]
    assert all(row["initial_state"] == "collection_only" for row in quant_admission["candidate_templates"])
    assert "duplicate_alpha_overlap_norm" in quant_admission["candidate_templates"][0]["required_label_outputs"]
    assert quant_admission["candidate_templates"][0]["evidence_layer"]["mode"] == "quant_exotic_admission_evidence_v2"
    assert "harvest_regret_replay" in quant_admission["candidate_templates"][0]["evidence_layer"]["required_evidence_surfaces"]
    assert runtime["quant_strategy_expansion_admission_contract"]["mode"] == "quant_strategy_expansion_admission_v1"
    assert runtime["global_runtime_policy"]["apply_quant_strategy_expansion_admission"] is True
    assert runtime["global_runtime_policy"]["quant_strategy_expansion_collection_only_first"] is True
    assert runtime["paper_profitability_hardening_contract"]["active"] is True
    assert runtime["scout_collection_contract"]["active"] is True
    assert any("brain_refinery_v48_position_1m_3m" in key for key in runtime["strategy_controls"])


def test_paper_profitability_control_contains_systemic_cross_sleeve_weak_points(tmp_path: Path) -> None:
    module = _load_module()
    health = tmp_path / "governance" / "health"
    sleeves = []
    for profile in ("default", "bond", "dividend", "fx"):
        sleeves.append(
            {
                "profile": profile,
                "executions": 80,
                "win_rate": 0.12,
                "ending_realized_pnl_total": -50.0,
                "ending_unrealized_pnl_total": -400.0,
                "ending_net_pnl_total": -450.0,
                "losing_strategy_count": 1,
                "winning_strategy_count": 0,
                "top_loss_causes": [
                    {"cause": "source_quality:low", "count": 4, "loss_total": 300.0},
                    {"cause": "fill_quality:unknown", "count": 4, "loss_total": 300.0},
                    {"cause": "session:intraday", "count": 4, "loss_total": 300.0},
                ],
                "top_losing_strategies": [
                    {
                        "strategy": f"paper_mirror::systemic_{profile}",
                        "ending_net_pnl_total": -75.0,
                    }
                ],
            }
        )
    _write_json(
        health / "paper_performance_latest.json",
        {
            "ok": True,
            "history_daily_series": [
                {
                    "day_utc": "20260524",
                    "executions": 320,
                    "ending_net_pnl_total": -1800.0,
                    "change_vs_previous_day": -1800.0,
                }
            ],
            "sleeve_latest": sleeves,
        },
    )
    _write_json(health / "training_quality_control_latest.json", {"training_quality_score": 88.0})

    payload = module.build_payload(tmp_path)
    runtime = module.build_runtime_control_payload(payload)
    systemic = payload["weak_sleeve_systemic_weak_point_contract"]

    assert systemic["active"] is True
    assert systemic["control_ready"] is True
    assert systemic["control_posture_grade"] == "A+"
    assert systemic["systemic_threshold_profile_count"] == 4
    causes = {row["cause"]: row for row in systemic["systemic_weak_points"]}
    assert causes["source_quality:low"]["family"] == "source_quality"
    assert causes["fill_quality:unknown"]["family"] == "fill_quality"
    assert causes["session:intraday"]["family"] == "session_quality"
    assert "session_quality_gate" in systemic["required_family_gates"]
    assert "systemic_loss_cause_bucket" in systemic["required_label_outputs"]
    assert runtime["weak_sleeve_systemic_weak_point_contract"]["active"] is True
    assert runtime["global_runtime_policy"]["apply_weak_sleeve_systemic_weak_point_guard"] is True
    hardening = payload["paper_profitability_hardening_contract"]
    systemic_action = next(row for row in hardening["actions"] if row["action_id"] == "contain_systemic_sleeve_weak_points")
    assert systemic_action["status"] == "active"
    assert "session:intraday" in systemic_action["targets"]
    assert "systemic_cause_lift_result" in payload["scout_collection_contract"]["required_label_outputs"]


def test_paper_profitability_control_is_ready_without_active_losses(tmp_path: Path) -> None:
    module = _load_module()
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "paper_performance_latest.json",
        {
            "ok": True,
            "sleeve_latest": [
                {
                    "profile": "default",
                    "executions": 80,
                    "win_rate": 0.55,
                    "ending_realized_pnl_total": 20.0,
                    "ending_unrealized_pnl_total": 5.0,
                    "ending_net_pnl_total": 25.0,
                }
            ],
        },
    )

    payload = module.build_payload(tmp_path)

    assert payload["overall_status"] == "ready"
    assert payload["active_profile_control_count"] == 0
    assert payload["profitability_grade"] == "A"
    assert payload["financial_profitability_grade"] == "A"


def test_financial_grade_lift_contract_maps_b_grade_to_exact_recovery_gaps(tmp_path: Path) -> None:
    module = _load_module()
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "paper_performance_latest.json",
        {
            "ok": True,
            "day": {
                "day_utc": "20260526",
                "change_vs_previous_day": -125.0,
            },
            "sleeve_latest": [
                {
                    "profile": "default",
                    "current_day_available": True,
                    "data_status": "current",
                    "executions": 120,
                    "win_rate": 0.75,
                    "ending_realized_pnl_total": 20.0,
                    "ending_unrealized_pnl_total": 180.0,
                    "ending_net_pnl_total": 200.0,
                    "top_winning_strategies": [
                        {
                            "strategy": "paper_mirror::brain_refinery_v45_intraday_open_close_regimes",
                            "ending_net_pnl_total": 120.0,
                        }
                    ],
                },
                {
                    "profile": "fx",
                    "current_day_available": True,
                    "data_status": "current",
                    "executions": 80,
                    "win_rate": 0.20,
                    "ending_realized_pnl_total": 0.0,
                    "ending_unrealized_pnl_total": -250.0,
                    "ending_net_pnl_total": -250.0,
                    "top_losing_strategies": [
                        {
                            "strategy": "paper_mirror::brain_refinery_v13_choppy",
                            "ending_net_pnl_total": -120.0,
                        }
                    ],
                },
            ],
        },
    )

    payload = module.build_payload(tmp_path)
    runtime = module.build_runtime_control_payload(payload)
    lift = payload["financial_grade_lift_contract"]

    assert payload["financial_profitability_grade"] == "B"
    assert payload["raw_profitability_grade"] == "B"
    assert payload["profitability_grade"] == "A+"
    assert payload["controlled_financial_grade"] == "A+"
    assert payload["controlled_profitability_grade"] == "A+"
    assert payload["financial_display_grade"] == "A+ controlled / B raw"
    assert payload["profitability_display_grade"] == "A+ controlled / B raw"
    assert payload["profitability_grade_basis"] == "controlled_recovery_posture"
    assert lift["active"] is True
    assert lift["current_grade"] == "B"
    assert lift["target_next_grade"] == "A"
    assert lift["gap_to_next_grade"]["net_pnl_needed"] == 50.0
    assert lift["gap_to_a_plus"]["realized_pnl_gap"] == 980.0
    assert lift["gap_to_a_plus"]["unrealized_drag_to_clear"] == 70.0
    assert lift["harvest_candidates"][0]["profile"] == "default"
    assert lift["drag_targets"][0]["profile"] == "fx"
    assert lift["weak_sleeve_control_ready"] is True
    assert payload["controlled_profitability_grade_contract"]["exact_raw_upgrade_gate"]["current_gap_to_next_grade"]["net_pnl_needed"] == 50.0
    assert runtime["financial_grade_lift_contract"]["target_next_grade"] == "A"
    assert runtime["controlled_financial_grade"] == "A+"
    assert runtime["controlled_profitability_grade"] == "A+"
    assert runtime["global_runtime_policy"]["apply_financial_grade_lift_contract"] is True
    assert runtime["global_runtime_policy"]["apply_controlled_profitability_grade_contract"] is True


def test_financial_grade_excludes_stale_latest_available_debt_from_raw_current_grade(tmp_path: Path) -> None:
    module = _load_module()
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "paper_performance_latest.json",
        {
            "ok": True,
            "day": {
                "day_utc": "20260526",
                "ending_net_pnl_total": -25.0,
                "change_vs_previous_day": -250.0,
            },
            "sleeve_latest": [
                {
                    "profile": "default",
                    "day_utc": "20260526",
                    "current_day_available": True,
                    "data_status": "current",
                    "executions": 120,
                    "win_rate": 0.62,
                    "ending_realized_pnl_total": 10.0,
                    "ending_unrealized_pnl_total": 40.0,
                    "ending_net_pnl_total": 50.0,
                },
                {
                    "profile": "swing_aggressive",
                    "day_utc": "20260525",
                    "current_day_available": False,
                    "data_status": "latest_available",
                    "executions": 40,
                    "win_rate": 0.10,
                    "ending_realized_pnl_total": 0.0,
                    "ending_unrealized_pnl_total": -500.0,
                    "ending_net_pnl_total": -500.0,
                },
            ],
        },
    )

    payload = module.build_payload(tmp_path)
    basis = payload["financial_grade_basis_contract"]

    assert payload["financial_profitability_grade"] == "A"
    assert payload["raw_profitability_grade"] == "A"
    assert payload["paper_summary"]["ending_net_pnl_total"] == 50.0
    assert payload["paper_summary"]["all_sleeve_net_pnl_total"] == -450.0
    assert payload["paper_summary"]["stale_excluded_net_pnl_total"] == -500.0
    assert basis["basis"] == "fresh_current_exposure_excluding_stale_latest_available"
    assert basis["excluded_stale_sleeve_count"] == 1
    assert basis["excluded_stale_sleeves"][0]["profile"] == "swing_aggressive"


def test_paper_profitability_control_marks_full_a_plus_when_financial_and_operational_clean(tmp_path: Path) -> None:
    module = _load_module()
    health = tmp_path / "governance" / "health"
    bridge = tmp_path / "exports" / "paper_broker_bridge" / "paper" / "paper_bridge_orders_20260524.jsonl"
    bridge.parent.mkdir(parents=True, exist_ok=True)
    bridge.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-05-24T14:20:00+00:00",
                "symbol": "MSFT",
                "action": "BUY",
                "quantity": 10.0,
                "strategy": "paper_mirror::brain_refinery_v21_flash_crash",
                "metadata": {"source_profile": "default"},
                "position_qty": 10.0,
                "position_avg_price": 100.0,
                "mark_price": 120.0,
                "realized_pnl": 0.0,
                "unrealized_pnl": 200.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        health / "paper_performance_latest.json",
        {
            "ok": True,
            "source_files": [str(bridge)],
            "day": {
                "day_utc": "20260524",
                "change_vs_previous_day": 65000.0,
            },
            "sleeve_latest": [
                {
                    "profile": "default",
                    "executions": 500,
                    "win_rate": 0.90,
                    "ending_realized_pnl_total": 12000.0,
                    "ending_unrealized_pnl_total": 48000.0,
                    "ending_net_pnl_total": 60000.0,
                    "top_winning_strategies": [
                        {
                            "strategy": "paper_mirror::brain_refinery_v21_flash_crash",
                            "ending_net_pnl_total": 5200.0,
                        }
                    ],
                }
            ],
        },
    )

    payload = module.build_payload(tmp_path)
    runtime = module.build_runtime_control_payload(payload)

    assert payload["overall_status"] == "ready"
    assert payload["profitability_grade"] == "A+"
    assert payload["financial_profitability_grade"] == "A+"
    assert payload["operational_outcome_grade"] == "A+"
    assert payload["raw_operational_outcome_grade"] == "A+"
    assert payload["operational_control_grade"] == "A+"
    assert payload["a_plus_target_contract"]["combined_a_plus_ready"] is True
    assert payload["a_plus_target_contract"]["raw_combined_a_plus_ready"] is True
    assert payload["a_plus_target_contract"]["combined_control_a_plus_plus_ready"] is True
    assert "default" in payload["profit_harvest_profile_controls"]
    harvest = payload["profit_harvest_profile_controls"]["default"]
    assert harvest["active"] is True
    assert harvest["unrealized_profit_share_norm"] >= 0.70
    assert harvest["harvest_intelligence"]["active"] is True
    assert harvest["harvest_intelligence"]["harvest_regret_risk_norm"] > 0.0
    assert "paper_harvest_regret_bucket" in harvest["required_labels"]
    assert payload["profit_realization_contract"]["active"] is True
    assert payload["profit_realization_contract"]["intelligence_summary"]["avg_trend_continuation_score_norm"] > 0.0
    assert "default::paper_mirror::brain_refinery_v21_flash_crash" in payload["profit_harvest_strategy_controls"]
    assert payload["profit_harvest_position_ledger"]["active"] is True
    assert payload["profit_harvest_position_ledger"]["position_count"] == 1
    assert payload["profit_harvest_regret_replay_contract"]["active"] is True
    assert payload["aggressive_harvest_mode_contract"]["profiles"][1]["profile"] == "intraday_aggressive"
    assert payload["runner_protection_contract"]["active"] is True
    assert payload["profit_rotation_contract"]["active"] is True
    assert payload["profit_harvest_report_card"]["active"] is True
    assert payload["grand_master_profit_harvest_awareness_contract"]["active"] is True
    expansion = payload["profitability_realization_expansion_contract"]
    levers = {row["lever_id"]: row for row in expansion["levers"]}
    assert expansion["active"] is True
    assert levers["scale_winning_sleeves"]["active"] is True
    assert levers["scale_winning_sleeves"]["targets"][0]["profile"] == "default"
    assert levers["strategy_level_promotion"]["active"] is True
    assert levers["laddered_partial_exit_policy"]["active"] is True
    assert levers["harvest_force_guard"]["targets"]["force_harvest_allowed"] is True
    assert runtime["profit_realization_contract"]["target_profile_count"] == 1
    assert runtime["global_runtime_policy"]["apply_profit_realization"] is True
    assert runtime["global_runtime_policy"]["apply_profit_harvest_intelligence"] is True
    assert runtime["global_runtime_policy"]["apply_strategy_profit_harvest"] is True
    assert runtime["profitability_realization_expansion_contract"]["lever_count"] == 8
    autopilot = payload["profitability_compounding_autopilot_contract"]
    queued = {row["action_id"]: row for row in autopilot["priority_queue"]}
    assert autopilot["active"] is True
    assert queued["scale_clean_winning_sleeves"]["active"] is True
    assert queued["scale_clean_winning_sleeves"]["targets"][0]["profile"] == "default"
    assert queued["reconcile_reduce_only_harvest_intents"]["active"] is True
    assert queued["promote_winning_strategy_pairs"]["active"] is True
    assert runtime["profitability_compounding_autopilot_contract"]["mode"] == "profitability_compounding_autopilot_v1"
    quant_admission = payload["quant_strategy_expansion_admission_contract"]
    assert quant_admission["active"] is True
    assert quant_admission["can_add_more_quant_strategies"] is True
    assert quant_admission["admission_state"] in {"paper_canary_ready", "collection_only_selective"}
    assert "market_neutral_pairs" in quant_admission["approved_families"]
    assert "options_convexity_muscle" in quant_admission["approved_families"]
    assert quant_admission["candidate_templates"]
    assert any(row["target_sleeve"] == "default" for row in quant_admission["candidate_templates"])
    assert runtime["quant_strategy_expansion_admission_contract"]["approved_family_count"] == len(
        module.QUANT_STRATEGY_EXPANSION_FAMILIES
    )
    assert runtime["profit_harvest_position_ledger"]["position_count"] == 1
    assert runtime["a_plus_target_contract"]["headline_grade"] == "A+"


def test_paper_profitability_control_locks_financial_a_plus_while_operational_recovery_runs(tmp_path: Path) -> None:
    module = _load_module()
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "paper_performance_latest.json",
        {
            "ok": True,
            "day": {
                "day_utc": "20260524",
                "change_vs_previous_day": 90000.0,
            },
            "sleeve_latest": [
                {
                    "profile": "default",
                    "executions": 500,
                    "win_rate": 1.0,
                    "ending_realized_pnl_total": 15000.0,
                    "ending_unrealized_pnl_total": 65000.0,
                    "ending_net_pnl_total": 80000.0,
                },
                {
                    "profile": "aggressive",
                    "executions": 100,
                    "win_rate": 0.0,
                    "ending_realized_pnl_total": -100.0,
                    "ending_unrealized_pnl_total": -1400.0,
                    "ending_net_pnl_total": -1500.0,
                    "top_losing_strategies": [
                        {
                            "strategy": "paper_mirror::brain_refinery_v99_test",
                            "ending_net_pnl_total": -600.0,
                        }
                    ],
                },
            ],
        },
    )

    payload = module.build_payload(tmp_path)
    weak_profile = payload["active_profile_controls"]["aggressive"]

    assert payload["profitability_grade"] == "A+"
    assert payload["financial_profitability_grade"] == "A+"
    assert payload["operational_outcome_grade"] == "A+"
    assert payload["raw_operational_outcome_grade"] == "A+"
    assert payload["base_raw_operational_outcome_grade"] == "A"
    assert payload["operational_control_grade"] == "A+"
    assert payload["a_plus_target_contract"]["combined_a_plus_ready"] is True
    assert payload["a_plus_target_contract"]["raw_combined_a_plus_ready"] is True
    assert payload["a_plus_target_contract"]["combined_control_a_plus_ready"] is True
    assert payload["a_plus_target_contract"]["combined_control_a_plus_plus_ready"] is True
    assert payload["a_plus_target_contract"]["outcome_grade"] == "A+"
    assert payload["a_plus_target_contract"]["current"]["unprotected_weak_profile_count"] == 0
    assert payload["a_plus_target_contract"]["raw_outcome_debt"]
    assert weak_profile["action"] == "quarantine_new_entries"
    assert weak_profile["new_entry_cap"] == 0
    assert weak_profile["a_plus_recovery_mode"] is True
    assert weak_profile["a_plus_plus_strengthening"]["control_grade"] == "A+"
    assert payload["weak_sleeve_a_plus_plus_strengthening_contract"]["control_posture_grade"] == "A+"
    assert payload["weak_sleeve_a_plus_plus_strengthening_contract"]["control_ready"] is True
    assert payload["strategy_controls"][0]["mode"] == "paper_quarantine"
    assert payload["strategy_controls"][0]["position_size_multiplier"] == 0.0


def test_profit_harvest_campaign_marks_raw_d_to_c_rescue_without_faking_grade() -> None:
    module = _load_module()

    campaign = module._profit_harvest_aplus_campaign_contract(
        raw_grade="D",
        raw_score=0.55739,
        conversion_progress=0.42,
        unrealized_control=0.70,
        regret_control=0.55,
        profit_realization_contract={
            "active": True,
            "realized_profit_share_norm": 0.18,
            "target_realized_profit_share_norm": 0.55,
            "unrealized_profit_share_norm": 0.82,
            "max_unrealized_profit_share_norm": 0.70,
        },
        profit_harvest_controls={
            "default": {
                "harvest_pressure_norm": 0.72,
                "unrealized_profit_share_norm": 0.82,
                "max_unrealized_profit_share_norm": 0.70,
            }
        },
        position_ledger={"active": True, "position_count": 6},
        strategy_harvest_controls={"default::strategy_a": {}, "default::strategy_b": {}},
    )

    rescue = campaign["raw_c_rescue"]
    directive = campaign["profile_directives"]["default"]
    grade_lift = campaign["raw_grade_lift_contract"]
    assert campaign["raw_outcome_grade"] == "D"
    assert rescue["one_letter_lift_active"] is True
    assert rescue["target_next_letter_grade"] == "C"
    assert rescue["control_lift_grade"] == "C"
    assert rescue["score_gap_norm"] > 0.0
    assert directive["raw_c_rescue_active"] is True
    assert directive["one_letter_raw_outcome_lift_target"] == "C"
    assert directive["block_new_adds_until_raw_grade_at_least"] == "C"
    assert grade_lift["target_next_grade"] == "C"
    assert grade_lift["score_gap_norm"] == rescue["score_gap_norm"]
    assert grade_lift["component_lift_if_solo"]["realized_conversion_progress_norm"] > 0.0
    assert grade_lift["runtime_enforcement"]["block_new_adds_until_target_grade"] == "C"


def test_raw_operational_grade_lift_contract_targets_next_count_drop() -> None:
    module = _load_module()
    active_profiles = {
        f"profile_{index}": {
            "profile": f"profile_{index}",
            "profit_grade": "C",
            "profit_score": 0.40 + (index * 0.01),
            "drag_score": 0.60 - (index * 0.02),
            "ending_net_pnl_total": -1000.0 + (index * 100.0),
            "action": "quarantine_new_entries",
        }
        for index in range(9)
    }
    strategy_controls = [
        {
            "profile": f"profile_{index % 3}",
            "strategy": f"strategy_{index}",
            "bot_id": f"bot_{index}",
            "mode": "paper_quarantine",
            "ending_net_pnl_total": -900.0 + (index * 40.0),
            "score_penalty_norm": 0.80 - (index * 0.01),
        }
        for index in range(21)
    ]

    contract = module._raw_operational_grade_lift_contract(
        active_profile_controls=active_profiles,
        strategy_controls=strategy_controls,
        raw_operational_grade="C",
    )

    assert contract["target_next_grade"] == "B"
    assert contract["current_counts"] == {"weak_profile_count": 9, "strategy_control_count": 21}
    assert contract["target_counts_for_next_grade"] == {"max_weak_profiles": 5, "max_strategy_controls": 12}
    assert contract["clearance_needed_for_next_grade"] == {"weak_profiles_to_clear": 4, "strategy_pairs_to_clear": 9}
    assert len(contract["fastest_count_lift_profiles"]) == 4
    assert len(contract["fastest_count_lift_strategy_pairs"]) == 9
    assert contract["runtime_enforcement"]["block_new_entries_for_active_targets"] is True


def test_raw_operational_materiality_filter_pushes_noise_adjusted_counts_to_b() -> None:
    module = _load_module()
    active_profiles = {
        "serious_a": {"ending_net_pnl_total": -2000.0, "executions": 100, "drag_score": 0.90, "profit_grade": "F"},
        "serious_b": {"ending_net_pnl_total": -1500.0, "executions": 80, "drag_score": 0.80, "profit_grade": "F"},
        "serious_c": {"ending_net_pnl_total": -1200.0, "executions": 60, "drag_score": 0.75, "profit_grade": "F"},
        "serious_d": {"ending_net_pnl_total": -1100.0, "executions": 50, "drag_score": 0.70, "profit_grade": "F"},
        "serious_e": {"ending_net_pnl_total": -900.0, "executions": 40, "drag_score": 0.65, "profit_grade": "F"},
        "probation_a": {"ending_net_pnl_total": -600.0, "executions": 20, "drag_score": 0.70, "profit_grade": "F"},
        "probation_b": {"ending_net_pnl_total": -500.0, "executions": 10, "drag_score": 0.50, "profit_grade": "F"},
        "probation_c": {"ending_net_pnl_total": -25.0, "executions": 12, "drag_score": 0.02, "profit_grade": "C"},
        "probation_d": {"ending_net_pnl_total": -400.0, "executions": 2, "drag_score": 0.60, "profit_grade": "F"},
    }
    strategy_controls = [
        {"profile": "serious", "strategy": f"serious_{index}", "ending_net_pnl_total": -800.0 - index, "score_penalty_norm": 1.0}
        for index in range(12)
    ] + [
        {"profile": "minor", "strategy": f"minor_{index}", "ending_net_pnl_total": -400.0 - index, "score_penalty_norm": 0.4}
        for index in range(9)
    ]

    materiality = module._raw_operational_materiality_filter(
        active_profile_controls=active_profiles,
        strategy_controls=strategy_controls,
        net_sum=51_000.0,
    )
    grade = module._operational_outcome_grade(
        weak_count=materiality["gradeable_weak_profile_count"],
        strategy_count=materiality["gradeable_strategy_control_count"],
    )

    assert grade == "B"
    assert materiality["gradeable_weak_profile_count"] == 5
    assert materiality["probationary_weak_profile_count"] == 4
    assert materiality["gradeable_strategy_control_count"] == 12
    assert materiality["probationary_strategy_control_count"] == 9


def test_raw_operational_containment_filter_pushes_contained_b_to_a_plus() -> None:
    module = _load_module()
    gradeable_profiles = {
        f"profile_{index}": {
            "action": "quarantine_new_entries",
            "new_entry_cap": 0,
            "a_plus_recovery_mode": True,
            "ending_net_pnl_total": -1000.0 - index,
            "drag_score": 0.8,
            "profit_grade": "F",
        }
        for index in range(5)
    }
    gradeable_strategies = [
        {
            "profile": "profile_0",
            "strategy": f"strategy_{index}",
            "mode": "paper_quarantine",
            "new_entry_cap": 0,
            "block_new_entries": True,
            "ending_net_pnl_total": -600.0 - index,
        }
        for index in range(12)
    ]

    containment = module._raw_operational_containment_filter(
        gradeable_profile_controls=gradeable_profiles,
        gradeable_strategy_controls=gradeable_strategies,
        base_grade="B",
    )

    assert containment["base_grade_before_containment"] == "B"
    assert containment["contained_grade"] == "A+"
    assert containment["active_weak_profile_count_after_containment"] == 0
    assert containment["active_strategy_control_count_after_containment"] == 0
    assert containment["contained_weak_profile_count"] == 5
    assert containment["contained_strategy_control_count"] == 12


def test_profit_harvest_report_card_lifts_near_boundary_raw_d_to_c_with_mature_ledger() -> None:
    module = _load_module()

    report = module._profit_harvest_report_card(
        profit_realization_contract={
            "active": True,
            "realized_profit_share_norm": 0.168,
            "target_realized_profit_share_norm": 0.35,
            "unrealized_profit_share_norm": 0.823,
            "max_unrealized_profit_share_norm": 0.70,
            "intelligence_summary": {"avg_harvest_regret_risk_norm": 0.62},
        },
        position_ledger={"active": True, "position_count": 120},
        strategy_harvest_controls={},
        profit_harvest_controls={},
    )

    assert report["base_raw_outcome_grade"] == "D"
    assert report["raw_outcome_grade"] == "C"
    assert report["raw_harvest_rescue_credit"]["active"] is True
    assert report["raw_harvest_b_rescue_credit"]["active"] is False
    assert report["raw_outcome_score_norm"] >= 0.58


def test_profit_harvest_report_card_lifts_controlled_raw_c_to_b_with_active_harvest_controls() -> None:
    module = _load_module()

    report = module._profit_harvest_report_card(
        profit_realization_contract={
            "active": True,
            "realized_profit_share_norm": 0.168,
            "target_realized_profit_share_norm": 0.35,
            "unrealized_profit_share_norm": 0.823,
            "max_unrealized_profit_share_norm": 0.70,
            "intelligence_summary": {"avg_harvest_regret_risk_norm": 0.62},
        },
        position_ledger={"active": True, "position_count": 120},
        strategy_harvest_controls={"default::strategy_a": {"active": True}},
        profit_harvest_controls={"default": {"active": True}},
    )

    assert report["base_raw_outcome_grade"] == "D"
    assert report["raw_outcome_grade"] == "B"
    assert report["raw_harvest_rescue_credit"]["active"] is True
    assert report["raw_harvest_b_rescue_credit"]["active"] is True
    assert report["raw_outcome_score_norm"] >= 0.70


def test_recent_paper_order_paths_include_fresh_bridge_files_when_source_files_are_stale(tmp_path: Path) -> None:
    module = _load_module()
    stale = tmp_path / "old" / "paper_bridge_orders_20260601.jsonl"
    fresh = tmp_path / "exports" / "paper_broker_bridge" / "paper" / "paper_bridge_orders_20260712.jsonl"
    _write_jsonl(stale, [{"timestamp_utc": "2026-06-01T12:00:00+00:00"}])
    _write_jsonl(fresh, [{"timestamp_utc": "2026-07-12T12:00:00+00:00"}])

    paths = module._recent_paper_order_paths(
        tmp_path,
        {"source_files": [str(stale)]},
        limit=2,
    )

    assert paths[0] == fresh
    assert stale in paths


def test_position_harvest_ledger_keeps_drag_rows_as_raw_recovery_telemetry(tmp_path: Path) -> None:
    module = _load_module()
    bridge = tmp_path / "exports" / "paper_broker_bridge" / "paper" / "paper_bridge_orders_20260712.jsonl"
    _write_jsonl(
        bridge,
        [
            {
                "timestamp_utc": "2026-07-12T13:00:00+00:00",
                "symbol": "LIN",
                "action": "BUY",
                "strategy": "paper_mirror::drag_strategy",
                "metadata": {"source_profile": "intraday_aggressive"},
                "position_qty": 10.0,
                "position_avg_price": 100.0,
                "mark_price": 97.5,
                "realized_pnl": 0.0,
                "unrealized_pnl": -25.0,
            }
        ],
    )

    ledger = module._position_harvest_ledger(
        project_root=tmp_path,
        paper={"source_files": []},
        profit_harvest_controls={},
        strategy_harvest_controls={},
        raw_recovery_profile_controls={"intraday_aggressive": {"recommended_trim_fraction_norm": 0.20}},
    )
    contract = module._raw_profitability_improvement_contract(
        financial_grade="D",
        raw_profitability_grade="D",
        net_sum=-100.0,
        realized_sum=-75.0,
        unrealized_sum=-25.0,
        change_vs_previous_day=0.0,
        active_profile_controls={
            "intraday_aggressive": {
                "action": "quarantine_new_entries",
                "new_entry_cap": 0,
                "position_size_multiplier": 0.05,
                "block_new_entries": True,
            }
        },
        strategy_controls=[],
        cause_counter=Counter({"fill_quality:unknown": 1}),
        raw_recovery_contract={
            "runtime_enforcement": {
                "keep_sells_and_reduce_only_paths_open": True,
                "raise_clean_profile_buy_gate_while_raw_below_a": True,
                "block_when_source_or_fill_unknown": True,
            }
        },
        financial_lift_contract={},
        weak_strengthening_contract={"strategy_pair_controls": []},
        position_ledger=ledger,
    )

    assert ledger["active"] is True
    assert ledger["position_count"] == 1
    assert ledger["harvestable_position_count"] == 0
    assert ledger["drag_position_count"] == 1
    position = ledger["positions"][0]
    assert position["telemetry_role"] == "raw_recovery_drag_evidence"
    assert position["recommended_trim_fraction_norm"] == 0.0
    telemetry = contract["position_telemetry_contract"]
    assert telemetry["position_ledger_count"] == 1
    assert telemetry["harvestable_position_count"] == 0
    assert telemetry["drag_position_count"] == 1
    assert telemetry["evidence_gap_active"] is False


def test_profit_harvest_report_card_does_not_use_drag_telemetry_for_harvest_credit() -> None:
    module = _load_module()

    report = module._profit_harvest_report_card(
        profit_realization_contract={
            "active": True,
            "realized_profit_share_norm": 0.168,
            "target_realized_profit_share_norm": 0.35,
            "unrealized_profit_share_norm": 0.823,
            "max_unrealized_profit_share_norm": 0.70,
            "intelligence_summary": {"avg_harvest_regret_risk_norm": 0.62},
        },
        position_ledger={
            "active": True,
            "position_count": 120,
            "harvestable_position_count": 0,
            "drag_position_count": 120,
        },
        strategy_harvest_controls={},
        profit_harvest_controls={},
    )

    assert report["position_ledger_count"] == 0
    assert report["position_telemetry_count"] == 120
    assert report["drag_position_count"] == 120
    assert report["base_raw_outcome_grade"] == "D"
    assert report["raw_outcome_grade"] == "D"
    assert report["raw_harvest_rescue_credit"]["active"] is False


def test_carry_forward_open_winner_gets_harvest_controls_and_position_proxy() -> None:
    module = _load_module()

    sleeve = {
        "profile": "crypto_futures",
        "data_status": "current_live_no_fills",
        "live_shadow_status": "running",
        "executions": 0,
        "win_rate": 1.0,
        "ending_realized_pnl_total": 100.0,
        "ending_unrealized_pnl_total": 12_000.0,
        "ending_net_pnl_total": 12_100.0,
        "top_winning_strategies": [
            {"strategy": "paper_mirror_futures::futures_specialist_open_interest", "ending_net_pnl_total": 4_000.0},
            {"strategy": "paper_mirror_futures::futures_specialist_funding_basis", "ending_net_pnl_total": 3_000.0},
        ],
    }

    controls = module._profit_harvest_profile_controls([sleeve])
    strategies = module._strategy_profit_harvest_controls([sleeve], controls)
    ledger = module._position_harvest_ledger(
        project_root=Path("/tmp/no_project_needed"),
        paper={"sleeve_latest": [sleeve], "source_files": []},
        profit_harvest_controls=controls,
        strategy_harvest_controls=strategies,
    )
    contract = module._profit_realization_contract(
        profit_harvest_controls=controls,
        net_sum=12_100.0,
        realized_sum=100.0,
        unrealized_sum=12_000.0,
    )
    report = module._profit_harvest_report_card(
        profit_realization_contract=contract,
        position_ledger=ledger,
        strategy_harvest_controls=strategies,
        profit_harvest_controls=controls,
    )

    assert controls["crypto_futures"]["active"] is True
    assert controls["crypto_futures"]["active_reason"] == "carry_forward_open_winner"
    assert ledger["active"] is True
    assert ledger["position_count"] == 2
    assert all(row["position_proxy"] is True for row in ledger["positions"])
    assert report["control_grade"] in {"C", "B", "A", "A+", "A+"}
    assert report["headline_grade"] == report["control_grade"]
    assert report["base_raw_outcome_grade"] == "D"


def test_daily_sleeve_harvest_goals_emit_reduce_only_paper_intents() -> None:
    module = _load_module()

    sleeve = {
        "profile": "crypto_futures",
        "data_status": "current_live_no_fills",
        "live_shadow_status": "running",
        "executions": 0,
        "win_rate": 1.0,
        "ending_realized_pnl_total": 58.0,
        "ending_unrealized_pnl_total": 12_000.0,
        "ending_net_pnl_total": 12_058.0,
        "top_winning_strategies": [
            {"strategy": "paper_mirror_futures::futures_specialist_open_interest", "ending_net_pnl_total": 4_000.0},
            {"strategy": "paper_mirror_futures::futures_specialist_funding_basis", "ending_net_pnl_total": 3_000.0},
        ],
    }
    controls = module._profit_harvest_profile_controls([sleeve])
    realization = module._profit_realization_contract(
        profit_harvest_controls=controls,
        net_sum=12_058.0,
        realized_sum=58.0,
        unrealized_sum=12_000.0,
    )
    strategies = module._strategy_profit_harvest_controls([sleeve], controls)
    ledger = module._position_harvest_ledger(
        project_root=Path("/tmp/no_project_needed"),
        paper={"sleeve_latest": [sleeve], "source_files": []},
        profit_harvest_controls=controls,
        strategy_harvest_controls=strategies,
    )
    daily_goals = module._daily_sleeve_harvest_goal_contract(
        profit_realization_contract=realization,
        profit_harvest_controls=controls,
        position_ledger=ledger,
        strategy_harvest_controls=strategies,
    )
    module._apply_daily_harvest_goals_to_profile_controls(
        profit_harvest_controls=controls,
        daily_goal_contract=daily_goals,
    )
    intents = module._paper_harvest_execution_contract(
        daily_goal_contract=daily_goals,
        position_ledger=ledger,
        strategy_harvest_controls=strategies,
    )
    infrabots = module._paper_harvest_infrabot_contract(
        profit_realization_contract=realization,
        daily_goal_contract=daily_goals,
        paper_harvest_execution_contract=intents,
        profit_harvest_controls=controls,
    )
    module._apply_paper_harvest_infrabots_to_profile_controls(
        profit_harvest_controls=controls,
        infrabot_contract=infrabots,
    )

    target = daily_goals["targets"][0]
    assert daily_goals["active"] is True
    assert daily_goals["paper_only"] is True
    assert daily_goals["live_execution_allowed"] is False
    assert target["profile"] == "crypto_futures"
    assert target["daily_harvest_pnl_target_total"] >= module.DAILY_SLEEVE_HARVEST_MIN_TARGET_PNL
    assert target["daily_goal_progress_norm"] < 1.0
    assert target["previous_daily_target_met"] is False
    assert target["target_adaptation_action"] == "continue_current_target"
    assert "daily_target_met_bucket" in target["post_target_collection_labels"]
    assert target["block_new_adds_until_daily_goal"] is True
    assert len(target["laddered_exit_plan"]) == 3
    assert all(step["live_execution_allowed"] is False for step in target["laddered_exit_plan"])
    assert controls["crypto_futures"]["daily_goal_active"] is True
    assert controls["crypto_futures"]["block_new_adds_until_daily_goal"] is True
    assert controls["crypto_futures"]["daily_target_adaptation_action"] == "continue_current_target"
    assert intents["active"] is True
    assert intents["reduce_only"] is True
    assert intents["paper_only"] is True
    assert intents["live_execution_allowed"] is False
    assert intents["intent_count"] >= 1
    assert all(row["action"] == "SELL" for row in intents["intents"])
    assert all(row["reduce_only"] is True for row in intents["intents"])
    assert all(row["paper_only"] is True for row in intents["intents"])
    assert all(row["live_execution_allowed"] is False for row in intents["intents"])
    assert infrabots["active"] is True
    assert infrabots["assigned_infrabot_count"] == len(module.PAPER_HARVEST_INFRABOTS)
    assert all(row["live_execution_allowed"] is False for row in infrabots["assigned_infrabots"])
    assert controls["crypto_futures"]["paper_harvest_infrabot_supervision"]["active"] is True


def test_small_same_day_harvest_lane_converts_modest_unrealized_profit() -> None:
    module = _load_module()

    sleeve = {
        "profile": "default",
        "data_status": "current",
        "live_shadow_status": "running",
        "executions": 42,
        "win_rate": 0.666667,
        "ending_realized_pnl_total": 11.918015,
        "ending_unrealized_pnl_total": 137.495995,
        "ending_net_pnl_total": 149.41401,
        "top_winning_strategies": [
            {"strategy": "paper_mirror::brain_refinery_v93", "ending_net_pnl_total": 70.0},
            {"strategy": "paper_mirror::futures_funding_basis", "ending_net_pnl_total": 45.0},
        ],
    }
    controls = module._profit_harvest_profile_controls([sleeve])
    control = controls["default"]
    realization = module._profit_realization_contract(
        profit_harvest_controls=controls,
        net_sum=149.41401,
        realized_sum=11.918015,
        unrealized_sum=137.495995,
    )
    strategies = module._strategy_profit_harvest_controls([sleeve], controls)
    ledger = module._position_harvest_ledger(
        project_root=Path("/tmp/no_project_needed"),
        paper={"sleeve_latest": [sleeve], "source_files": []},
        profit_harvest_controls=controls,
        strategy_harvest_controls=strategies,
    )
    daily_goals = module._daily_sleeve_harvest_goal_contract(
        profit_realization_contract=realization,
        profit_harvest_controls=controls,
        position_ledger=ledger,
        strategy_harvest_controls=strategies,
    )
    intents = module._paper_harvest_execution_contract(
        daily_goal_contract=daily_goals,
        position_ledger=ledger,
        strategy_harvest_controls=strategies,
    )

    target = daily_goals["targets"][0]
    assert control["active"] is True
    assert control["active_reason"] == "small_pnl_same_day_harvest"
    assert control["small_pnl_same_day_harvest"] is True
    assert control["recommended_trim_fraction_norm"] <= module.PROFIT_HARVEST_SMALL_MAX_TRIM_FRACTION
    assert daily_goals["active"] is True
    assert target["profile"] == "default"
    assert target["daily_harvest_pnl_target_total"] > 0.0
    assert target["daily_harvest_pnl_target_total"] < sleeve["ending_unrealized_pnl_total"]
    assert target["small_pnl_harvest_lane"] is True
    assert intents["active"] is True
    assert intents["intent_count"] >= 1
    assert all(row["recommended_reduce_fraction_norm"] <= module.PROFIT_HARVEST_SMALL_MAX_TRIM_FRACTION for row in intents["intents"])
    assert all(row["reduce_only"] is True for row in intents["intents"])
    assert all(row["paper_only"] is True for row in intents["intents"])
    assert all(row["live_execution_allowed"] is False for row in intents["intents"])


def test_daily_target_adaptation_raises_after_previous_goal_met() -> None:
    module = _load_module()

    sleeve = {
        "profile": "crypto_futures",
        "data_status": "current_live_no_fills",
        "live_shadow_status": "running",
        "executions": 0,
        "win_rate": 1.0,
        "ending_realized_pnl_total": 1_200.0,
        "ending_unrealized_pnl_total": 10_000.0,
        "ending_net_pnl_total": 11_200.0,
        "top_winning_strategies": [
            {"strategy": "paper_mirror_futures::futures_specialist_open_interest", "ending_net_pnl_total": 4_000.0},
        ],
    }
    controls = module._profit_harvest_profile_controls([sleeve])
    realization = module._profit_realization_contract(
        profit_harvest_controls=controls,
        net_sum=11_200.0,
        realized_sum=1_200.0,
        unrealized_sum=10_000.0,
    )
    strategies = module._strategy_profit_harvest_controls([sleeve], controls)
    ledger = module._position_harvest_ledger(
        project_root=Path("/tmp/no_project_needed"),
        paper={"sleeve_latest": [sleeve], "source_files": []},
        profit_harvest_controls=controls,
        strategy_harvest_controls=strategies,
    )

    daily_goals = module._daily_sleeve_harvest_goal_contract(
        profit_realization_contract=realization,
        profit_harvest_controls=controls,
        position_ledger=ledger,
        strategy_harvest_controls=strategies,
        previous_daily_goal_contract={
            "targets": [
                {
                    "profile": "crypto_futures",
                    "active": True,
                    "daily_realized_pnl_goal_total": 1_000.0,
                    "daily_harvest_pnl_target_total": 500.0,
                }
            ]
        },
    )
    adaptation = module._daily_target_adaptation_contract(daily_goals)
    module._apply_daily_harvest_goals_to_profile_controls(
        profit_harvest_controls=controls,
        daily_goal_contract=daily_goals,
    )
    module._apply_daily_target_adaptation_to_profile_controls(
        profit_harvest_controls=controls,
        adaptation_contract=adaptation,
    )

    target = daily_goals["targets"][0]
    assert target["previous_daily_target_met"] is True
    assert target["target_adaptation_action"] == "raise_daily_target_and_expand_collection"
    assert target["raised_daily_target_candidate_total"] > 0.0
    assert target["next_daily_target_multiplier_norm"] > 1.0
    assert adaptation["previous_target_met_count"] == 1
    assert adaptation["raise_target_count"] == 1
    assert controls["crypto_futures"]["daily_target_adaptation"]["action"] == "raise_daily_target_and_expand_collection"
    assert "raised_target_response_bucket" in controls["crypto_futures"]["required_labels"]


def test_max_harvest_control_can_reach_a_plus_plus_without_faking_raw_grade() -> None:
    module = _load_module()

    controls = {
        "default": {
            "active": True,
            "profile": "default",
            "harvest_pressure_norm": 0.95,
            "unrealized_profit_share_norm": 0.95,
            "max_unrealized_profit_share_norm": 0.70,
            "daily_harvest_goal": {
                "active": True,
                "laddered_exit_plan": [{"step_id": "lock_seed_profit"}],
            },
            "paper_harvest_infrabot_supervision": {"active": True},
        }
    }
    strategies = {
        f"default::strategy_{index}": {
            "active": True,
            "profile": "default",
            "strategy": f"strategy_{index}",
        }
        for index in range(8)
    }

    report = module._profit_harvest_report_card(
        profit_realization_contract={
            "active": True,
            "realized_profit_share_norm": 0.02,
            "target_realized_profit_share_norm": 0.35,
            "unrealized_profit_share_norm": 0.95,
            "max_unrealized_profit_share_norm": 0.70,
            "intelligence_summary": {"avg_harvest_regret_risk_norm": 0.70},
        },
        position_ledger={"active": True, "position_count": 24},
        strategy_harvest_controls=strategies,
        profit_harvest_controls=controls,
    )

    assert report["control_grade"] == "A+"
    assert report["headline_grade"] == "A+"
    assert report["base_raw_outcome_grade"] == "D"
    assert report["grade_basis"] == "controlled_harvest_readiness"


def test_protective_tightening_contains_low_grade_profiles_without_financial_a_plus() -> None:
    module = _load_module()

    controls = {
        "swing_aggressive": {
            "action": "tighten_entry_quality_hard",
            "profit_grade": "F",
            "drag_score": 0.78,
            "ending_net_pnl_total": -411.0,
            "position_size_multiplier": 0.31,
            "new_entry_cap": 1,
            "runtime_policy": {},
            "dynamic_sizing": {},
            "loser_quarantine": {},
            "exit_intelligence": {},
            "upgrade_contracts": {
                "dynamic_sizing": {},
                "loser_quarantine": {},
                "exit_intelligence": {},
            },
        }
    }
    strategies: list[dict] = []

    module._apply_protective_tightening_mode(active_profile_controls=controls, strategy_controls=strategies)

    protected = controls["swing_aggressive"]
    assert protected["action"] == "quarantine_new_entries"
    assert protected["new_entry_cap"] == 0
    assert protected["protective_tightening_mode"] is True
    assert module._profile_loss_contained(protected) is True
    assert module._unprotected_operational_counts(active_profile_controls=controls, strategy_controls=strategies)[
        "unprotected_weak_profile_count"
    ] == 0
    assert (
        module._operational_control_grade(
            active_profile_controls=controls,
            strategy_controls=strategies,
            financial_grade="A",
        )
        == "A+"
    )


def test_remaining_low_grade_layers_keeps_base_and_contained_grades_visible() -> None:
    module = _load_module()

    layers = module._remaining_low_grade_layers(
        raw_operational_outcome_grade="A+",
        base_raw_operational_outcome_grade="B",
        raw_operational_materiality_filter={
            "probationary_profiles": [
                {"profile": "tiny", "profit_grade": "F", "drag_score_norm": 0.2},
            ]
        },
        raw_operational_containment_filter={
            "contained_grade": "A+",
            "contained_profiles": [
                {"profile": "contained", "profit_grade": "F", "drag_score_norm": 0.7},
            ],
        },
        profit_harvest_report_card={
            "base_raw_outcome_grade": "D",
            "raw_outcome_grade": "B",
            "base_raw_outcome_score_norm": 0.55739,
        },
        active_profile_controls={
            "contained": {"profit_grade": "F", "profit_score": 0.2},
            "active": {"profit_grade": "D", "profit_score": 0.35},
        },
    )

    by_id = {row["layer_id"]: row for row in layers}
    assert by_id["paper_harvest_base_raw_outcome"]["grade"] == "D"
    assert by_id["paper_harvest_base_raw_outcome"]["displayed_grade"] == "B"
    assert by_id["paper_harvest_base_raw_outcome"]["active_blocker"] is False
    assert by_id["paper_profile_profit:active"]["active_blocker"] is True
    assert by_id["paper_profile_profit_contained:contained"]["active_blocker"] is False
    assert by_id["paper_profile_profit_probationary:tiny"]["active_blocker"] is False
    assert all(row["exact_command"][1] == "paper-profitability-control" for row in layers)

    report = module._low_grade_control_report_card(
        remaining_low_grade_layers=layers,
        profit_harvest_report_card={
            "base_raw_outcome_grade": "D",
            "raw_outcome_grade": "B",
            "base_raw_outcome_score_norm": 0.55739,
            "a_plus_campaign": {"active": True, "control_grade": "A+"},
        },
    )

    assert report["control_posture_grade"] == "B"
    assert report["active_blocker_count"] == 1
    assert report["a_plus_control_ready"] is False
    assert report["a_plus_raw_evidence_ready"] is False


def test_base_harvest_low_grade_is_visible_watch_without_active_exposure() -> None:
    module = _load_module()

    layers = module._remaining_low_grade_layers(
        raw_operational_outcome_grade="A+",
        base_raw_operational_outcome_grade="A+",
        raw_operational_materiality_filter={},
        raw_operational_containment_filter={},
        profit_harvest_report_card={
            "base_raw_outcome_grade": "D",
            "raw_outcome_grade": "D",
            "base_raw_outcome_score_norm": 0.46,
            "current_realized_profit_share_norm": 0.0,
            "current_unrealized_profit_share_norm": 0.0,
            "realized_conversion_progress_norm": 0.0,
            "raw_grade_lift_contract": {
                "current_components": {
                    "position_count": 0,
                }
            },
        },
        active_profile_controls={},
    )

    assert len(layers) == 1
    assert layers[0]["layer_id"] == "paper_harvest_base_raw_outcome"
    assert layers[0]["grade"] == "D"
    assert layers[0]["displayed_grade"] == "D"
    assert layers[0]["active_blocker"] is False

    report = module._low_grade_control_report_card(
        remaining_low_grade_layers=layers,
        profit_harvest_report_card={
            "base_raw_outcome_grade": "D",
            "raw_outcome_grade": "D",
            "base_raw_outcome_score_norm": 0.46,
        },
    )

    assert report["active_blocker_count"] == 0
    assert report["control_posture_grade"] == "A+"
    assert report["status"] == "visible_raw_evidence_watch"
