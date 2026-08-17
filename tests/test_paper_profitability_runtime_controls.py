from scripts.ops import paper_profitability_control as control


def test_runtime_profile_controls_harden_raw_df_quarantines() -> None:
    hardened = control._runtime_profile_controls(
        {
            "Aggressive": {
                "profile": "Aggressive",
                "active": True,
                "action": "quarantine_new_entries",
                "profit_grade": "F",
                "position_size_multiplier": 0.08,
                "new_entry_cap": 0,
                "dynamic_sizing": {"paper_profitability_size_multiplier_norm": 0.08},
                "loser_quarantine": {"block_new_entries": True},
            }
        }
    )

    aggressive = hardened["aggressive"]
    assert aggressive["position_size_multiplier"] == 0.0
    assert aggressive["new_entry_cap"] == 0
    assert aggressive["block_new_entries"] is True
    assert aggressive["dynamic_sizing"]["paper_profitability_size_multiplier_norm"] == 0.0
    assert aggressive["dynamic_sizing"]["max_new_entry_multiplier_norm"] == 0.0
    assert aggressive["loser_quarantine"]["block_new_entries"] is True
    assert aggressive["a_plus_plus_strengthening"]["max_position_size_multiplier_norm"] == 0.0


def test_build_runtime_control_payload_uses_hardened_profile_controls() -> None:
    payload = {
        "active_profile_controls": {
            "default": {
                "profile": "default",
                "active": True,
                "action": "quarantine_new_entries",
                "profit_grade": "D",
                "position_size_multiplier": 0.08,
                "new_entry_cap": 0,
            }
        },
        "strategy_controls": [],
    }

    runtime = control.build_runtime_control_payload(payload)

    default = runtime["profile_controls"]["default"]
    assert default["position_size_multiplier"] == 0.0
    assert default["new_entry_cap"] == 0
    assert default["block_new_entries"] is True
    assert default["loser_quarantine"]["mode"] == "quarantine_new_entries"


def test_raw_profitability_a_recovery_contract_tracks_gap_and_runtime_policy() -> None:
    contract = control._raw_profitability_a_recovery_contract(
        financial_grade="D",
        raw_profitability_grade="D",
        net_sum=-1250.25,
        realized_sum=-500.0,
        unrealized_sum=-750.25,
        change_vs_previous_day=-1200.0,
        active_profile_controls={"aggressive": {"action": "quarantine_new_entries"}},
        strategy_controls=[{"mode": "paper_quarantine"}],
        cause_counter=control.Counter({"source_quality:low": 3}),
    )

    assert contract["active"] is True
    assert contract["gap_to_raw_a"]["net_pnl_gap"] == 1250.25
    assert contract["runtime_enforcement"]["block_new_entries_on_weak_profiles"] is True
    assert contract["runtime_enforcement"]["min_quality_gate_norm"] >= 0.70
    assert contract["raw_grade_remains_evidence_based"] is True


def test_raw_profitability_improvement_contract_locks_one_to_seven_controls() -> None:
    raw_recovery = control._raw_profitability_a_recovery_contract(
        financial_grade="D",
        raw_profitability_grade="D",
        net_sum=-900.0,
        realized_sum=-250.0,
        unrealized_sum=-650.0,
        change_vs_previous_day=-700.0,
        active_profile_controls={
            "aggressive": {
                "action": "quarantine_new_entries",
                "new_entry_cap": 0,
                "position_size_multiplier": 0.0,
                "block_new_entries": True,
            }
        },
        strategy_controls=[{"mode": "paper_quarantine", "new_entry_cap": 0, "position_size_multiplier_norm": 0.0}],
        cause_counter=control.Counter({"fill_quality:unknown": 2, "source_quality:low": 1}),
    )

    contract = control._raw_profitability_improvement_contract(
        financial_grade="D",
        raw_profitability_grade="D",
        net_sum=-900.0,
        realized_sum=-250.0,
        unrealized_sum=-650.0,
        change_vs_previous_day=-700.0,
        active_profile_controls={
            "aggressive": {
                "action": "quarantine_new_entries",
                "new_entry_cap": 0,
                "position_size_multiplier": 0.0,
                "block_new_entries": True,
                "ending_net_pnl_total": -900.0,
                "ending_unrealized_pnl_total": -650.0,
                "loser_quarantine": {"block_new_entries": True},
            }
        },
        strategy_controls=[{"mode": "paper_quarantine", "new_entry_cap": 0, "position_size_multiplier_norm": 0.0}],
        cause_counter=control.Counter({"fill_quality:unknown": 2, "source_quality:low": 1}),
        raw_recovery_contract=raw_recovery,
        financial_lift_contract={"drag_targets": [{"profile": "aggressive", "net_pnl_total": -900.0}]},
        weak_strengthening_contract={
            "strategy_pair_controls": [
                {
                    "profile": "aggressive",
                    "strategy": "paper_mirror::loss_bot",
                    "mode": "paper_quarantine",
                    "new_entry_cap": 0,
                    "position_size_multiplier_norm": 0.0,
                    "protected": True,
                }
            ]
        },
        position_ledger={"active": False, "position_count": 0, "source_file_count": 0, "records_scanned": 0},
    )

    assert contract["control_ready"] is True
    assert len(contract["requirements"]) == 7
    assert contract["weak_sleeve_zero_entry_contract"]["ready"] is True
    assert contract["clean_sleeve_strict_buy_gate_contract"]["min_quality_gate_norm"] >= 0.72
    assert contract["position_telemetry_contract"]["evidence_gap_active"] is True
    assert contract["position_telemetry_contract"]["does_not_pause_safe_paper_trading_by_itself"] is True
    assert contract["loss_cause_training_feedback_contract"]["feed_hard_negative_training_labels"] is True
    assert contract["losing_strategy_pair_quarantine_contract"]["required_profitable_refreshes_before_reentry"] == 3
    assert contract["burn_down_contract"]["required_average_daily_net_improvement"] == 30.0
    raw_d_ladder = contract["raw_d_recovery_ladder_contract"]
    assert raw_d_ladder["active"] is True
    assert raw_d_ladder["contract_ready"] is True
    assert raw_d_ladder["daily_net_improvement_target"] == 30.0
    assert raw_d_ladder["drag_reduction_target_count"] == 1
    assert raw_d_ladder["profile_level_drag_target_count"] == 1
    assert raw_d_ladder["runtime_enforcement"]["apply_raw_d_recovery_ladder"] is True
    assert raw_d_ladder["runtime_enforcement"]["force_profit_harvest_on_raw_d"] is False
    assert raw_d_ladder["runtime_enforcement"]["do_not_force_trades"] is True
    six_point = contract["six_point_recovery_contract"]
    assert six_point["control_ready"] is True
    assert six_point["rule_count"] == 6
    assert [row["id"] for row in six_point["rules"]] == [
        "1_block_weak_profile_fresh_buys",
        "2_keep_sell_reduce_only_paths_open",
        "3_clean_profile_buys_require_all_gates",
        "4_top_loss_causes_get_specific_filters",
        "5_realized_conversion_uses_partial_reduce_only_trims",
        "6_do_not_force_trades",
    ]
    assert six_point["runtime_enforcement"]["apply_loss_cause_specific_entry_filters"] is True
    assert six_point["runtime_enforcement"]["emit_partial_reduce_only_profit_trims"] is True
    assert six_point["runtime_enforcement"]["do_not_force_trades"] is True
    assert six_point["runtime_enforcement"]["force_profit_harvest_on_raw_d"] is False


def test_build_runtime_control_payload_exports_raw_a_recovery_policy() -> None:
    payload = {
        "active_profile_controls": {},
        "strategy_controls": [],
        "raw_profitability_a_recovery_contract": {
            "active": True,
            "runtime_enforcement": {
                "raise_clean_profile_buy_gate_while_raw_below_a": True,
            },
        },
        "raw_profitability_improvement_contract": {
            "active": True,
            "control_ready": True,
            "runtime_enforcement": {
                "require_position_telemetry_on_paper_fills": True,
                "track_raw_gap_burn_down": True,
                "apply_raw_d_recovery_ladder": True,
                "force_profit_harvest_on_raw_d": False,
                "do_not_force_trades": True,
                "accelerate_drag_reduction_on_raw_d": True,
                "block_widening_while_raw_d": True,
                "raise_harvest_trim_urgency_while_raw_d": True,
                "emit_reduce_only_for_raw_d_drag_positions": True,
                "raw_d_recovery_pressure_norm": 0.91,
                "raw_d_recovery_trim_boost_norm": 0.12,
                "raw_d_daily_net_improvement_target": 322.95,
            },
            "raw_d_recovery_ladder_contract": {
                "active": True,
                "recovery_pressure_norm": 0.91,
                "trim_boost_norm": 0.12,
                "daily_net_improvement_target": 322.95,
            },
        },
        "raw_profitability_six_point_recovery_contract": {
            "active": True,
            "control_ready": True,
            "runtime_enforcement": {
                "apply_loss_cause_specific_entry_filters": True,
                "emit_partial_reduce_only_profit_trims": True,
                "do_not_force_trades": True,
                "force_profit_harvest_on_raw_d": False,
            },
        },
    }

    runtime = control.build_runtime_control_payload(payload)

    assert runtime["raw_profitability_a_recovery_contract"]["active"] is True
    assert runtime["raw_profitability_improvement_contract"]["control_ready"] is True
    assert runtime["raw_profitability_six_point_recovery_contract"]["control_ready"] is True
    assert runtime["global_runtime_policy"]["apply_raw_profitability_a_recovery"] is True
    assert runtime["global_runtime_policy"]["apply_raw_profitability_improvement_contract"] is True
    assert runtime["global_runtime_policy"]["apply_raw_profitability_six_point_recovery"] is True
    assert runtime["global_runtime_policy"]["raise_clean_profile_buy_gate_while_raw_below_a"] is True
    assert runtime["global_runtime_policy"]["require_position_telemetry_on_paper_fills_for_raw_recovery"] is True
    assert runtime["global_runtime_policy"]["track_raw_profitability_burn_down"] is True
    assert runtime["raw_d_recovery_ladder_contract"]["active"] is True
    assert runtime["global_runtime_policy"]["apply_raw_d_recovery_ladder"] is True
    assert runtime["global_runtime_policy"]["force_profit_harvest_on_raw_d"] is False
    assert runtime["global_runtime_policy"]["do_not_force_trades_for_raw_recovery"] is True
    assert runtime["global_runtime_policy"]["apply_loss_cause_specific_entry_filters"] is True
    assert runtime["global_runtime_policy"]["emit_partial_reduce_only_profit_trims_for_raw_recovery"] is True
    assert runtime["global_runtime_policy"]["accelerate_drag_reduction_on_raw_d"] is True
    assert runtime["global_runtime_policy"]["block_widening_while_raw_d"] is True
    assert runtime["global_runtime_policy"]["raw_d_recovery_pressure_norm"] == 0.91
    assert runtime["global_runtime_policy"]["raw_d_recovery_trim_boost_norm"] == 0.12
    assert runtime["global_runtime_policy"]["raw_d_daily_net_improvement_target"] == 322.95
