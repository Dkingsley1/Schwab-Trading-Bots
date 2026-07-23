import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "ops" / "income_readiness_control.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("income_readiness_control", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load income_readiness_control")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _seed_health(project_root: Path) -> None:
    health = project_root / "governance" / "health"
    _write_json(
        health / "paper_profitability_control_latest.json",
        {
            "ok": True,
            "overall_status": "protective_tightening",
            "financial_profitability_grade": "A",
            "operational_control_grade": "A+",
            "operational_outcome_grade": "A+",
            "paper_summary": {
                "ending_net_pnl_total": 149.41,
                "ending_realized_pnl_total": 11.91,
                "ending_unrealized_pnl_total": 137.50,
            },
            "profit_harvest_report_card": {
                "grade": "B",
                "raw_outcome_grade": "D",
                "current_realized_profit_share_norm": 0.08,
                "current_unrealized_profit_share_norm": 0.92,
                "target_realized_profit_share_norm": 0.33,
            },
            "profit_realization_contract": {
                "active": True,
                "realized_profit_share_norm": 0.08,
                "unrealized_profit_share_norm": 0.92,
                "target_realized_profit_share_norm": 0.33,
                "max_unrealized_profit_share_norm": 0.72,
            },
            "paper_harvest_execution_contract": {
                "active": True,
                "reduce_only": True,
                "paper_only": True,
                "live_execution_allowed": False,
                "intent_count": 3,
            },
            "profit_harvest_strategy_controls": {
                "conservative::brain_refinery_v13_choppy": {"active": True},
                "conservative::brain_refinery_v10_seasonal": {"active": True},
                "default::futures_funding_basis": {"active": True},
            },
            "strategy_controls": [{"strategy": "a"}, {"strategy": "b"}],
            "profit_harvest_position_ledger": {
                "positions": [
                    {"profile": "conservative", "symbol": "SPY", "unrealized_pnl": 25.0},
                    {"profile": "default", "symbol": "BTC-USD", "unrealized_pnl": 35.0},
                ]
            },
        },
    )
    _write_json(
        health / "paper_performance_latest.json",
        {
            "active_paper_profiles_today": ["default", "conservative", "crypto_futures", "schwab_futures", "dividend"],
            "sleeve_latest": [
                {
                    "profile": f"profile_{idx}",
                    "executions": 120,
                    "mean_slippage_gap_bps": 0.5,
                    "poor_or_fair_fill_count": 0,
                    "data_status": "current",
                }
                for idx in range(11)
            ],
            "history_daily_series": [
                {"day_utc": f"202605{day:02d}", "ending_net_pnl_total": 100.0 + day, "change_vs_previous_day": 2.0}
                for day in range(1, 13)
            ],
        },
    )
    _write_json(health / "paper_execution_calibration_latest.json", {"timestamp_utc": "2999-01-01T00:00:00+00:00", "ok": True})
    _write_json(health / "process_watchdog_latest.json", {"overall_status": "ready"})
    _write_json(health / "ingestion_backpressure_latest.json", {"pending_lines_total": 13, "oldest_pending_age_seconds_total": 30})
    _write_json(health / "storage_quota_guard_latest.json", {"ok": True, "overall_status": "ready"})
    _write_json(health / "artifact_freshness_slo_latest.json", {"ok": True, "overall_status": "ready"})
    _write_json(health / "bot_logs_cleanup_intelligence_latest.json", {"ok": True, "cleanup_needed": False})
    _write_json(health / "training_quality_control_latest.json", {"overall_status": "ready", "training_quality_score": 100.0})
    _write_json(health / "training_runtime_control_latest.json", {"overall_status": "ready"})
    _write_json(health / "promotion_quality_gate_latest.json", {"ok": True, "failed_checks": []})
    _write_json(
        health / "account_policy_context_latest.json",
        {
            "timestamp_utc": "2999-01-01T00:00:00+00:00",
            "account_policy_context": {
                "pdt_intraday_margin_transition": {
                    "phase": "legacy_pdt_until_finra_effective_date",
                    "schwab_day_trade_count_retired": False,
                },
                "intraday_margin_probe_contract": {
                    "status": "scheduled_pre_schwab_cutover",
                    "probe_required_now": False,
                    "intraday_buying_power_observed": False,
                },
                "paper_intraday_margin_deficit_simulator": {
                    "status": "ready",
                    "simulated_margin_deficit_usd": 0.0,
                },
            },
        },
    )
    _write_json(health / "global_killswitch_latest.json", {"halt": False})


def test_income_readiness_contains_all_sections_and_keeps_live_locked(tmp_path: Path) -> None:
    module = _load_module()
    _seed_health(tmp_path)

    payload = module.build_payload(
        tmp_path,
        bot_logs_root=tmp_path,
        bot_logs_min_free_gb=1_000_000_000.0,
    )

    section_ids = {row["section_id"] for row in payload["sections"]}
    assert set(module.SECTION_ORDER) == section_ids
    assert payload["live_execution_allowed"] is False
    assert payload["live_micro_allowed"] is False
    assert payload["requires_separate_live_micro_approval"] is True
    assert "bot_logs_free_space_below_target" in payload["blockers"]
    assert "/Volumes/VIDEO" in payload["runtime_contract"]["storage_guard"]["protected_volumes"]
    assert payload["recommended_commands"]["storage_cleanup_when_needed"][:3] == [
        "./scripts/ops/opsctl.sh",
        "bot-logs-cleanup-intelligence",
        "--apply",
    ]


def test_income_readiness_runtime_control_is_paper_only(tmp_path: Path) -> None:
    module = _load_module()
    _seed_health(tmp_path)

    payload = module.build_payload(tmp_path, bot_logs_root=tmp_path, bot_logs_min_free_gb=0.1)
    control = module.build_runtime_control_payload(payload)

    assert control["paper_only"] is True
    assert control["live_execution_allowed"] is False
    assert control["live_micro_allowed"] is False
    assert control["section_controls"]["realized_profit_discipline"]["grade"] in {"D", "C", "B", "A", "A+", "A+"}


def test_income_readiness_controlled_100_keeps_raw_debt_visible(tmp_path: Path) -> None:
    module = _load_module()
    _seed_health(tmp_path)
    health = tmp_path / "governance" / "health"
    profitability = json.loads((health / "paper_profitability_control_latest.json").read_text(encoding="utf-8"))
    profitability.update(
        {
            "paper_summary": {
                "ending_net_pnl_total": 500.0,
                "ending_realized_pnl_total": 250.0,
                "ending_unrealized_pnl_total": 250.0,
            },
            "profit_harvest_report_card": {
                "grade": "A+",
                "raw_outcome_grade": "A",
                "current_realized_profit_share_norm": 1.0,
                "current_unrealized_profit_share_norm": 0.0,
                "target_realized_profit_share_norm": 0.35,
            },
            "profit_realization_contract": {
                "active": True,
                "realized_profit_share_norm": 1.0,
                "unrealized_profit_share_norm": 0.0,
                "target_realized_profit_share_norm": 0.35,
                "max_unrealized_profit_share_norm": 0.72,
            },
            "paper_harvest_execution_contract": {
                "active": True,
                "reduce_only": True,
                "paper_only": True,
                "live_execution_allowed": False,
                "intent_count": 5,
            },
            "low_grade_control_report_card": {
                "control_posture_grade": "A+",
                "a_plus_control_ready": True,
                "active_blocker_count": 0,
            },
            "raw_operational_containment_filter": {
                "contained_grade": "A+",
                "contained_weak_profile_count": 1,
                "contained_strategy_control_count": 1,
            },
            "paper_profitability_hardening_contract": {
                "new_entry_policy": {"block_quarantined_profiles": True},
            },
            "a_plus_target_contract": {
                "current": {
                    "unprotected_weak_profile_count": 0,
                    "unprotected_strategy_control_count": 0,
                },
            },
            "profit_harvest_strategy_controls": {
                f"default::strategy_{idx}": {"active": True}
                for idx in range(6)
            },
            "strategy_controls": [{"strategy": f"s{idx}"} for idx in range(10)],
            "profit_harvest_position_ledger": {
                "positions": [{"profile": "default", "symbol": f"S{idx}", "unrealized_pnl": 1.0} for idx in range(20)]
            },
        }
    )
    _write_json(health / "paper_profitability_control_latest.json", profitability)
    _write_json(
        health / "paper_performance_latest.json",
        {
            "active_paper_profiles_today": ["default", "conservative", "crypto_futures", "schwab_futures", "dividend"],
            "sleeve_latest": [
                {
                    "profile": f"profile_{idx}",
                    "executions": 120,
                    "mean_slippage_gap_bps": 0.5,
                    "poor_or_fair_fill_count": 0,
                    "data_status": "current",
                }
                for idx in range(11)
            ],
            "history_daily_series": [
                {"day_utc": "20260501", "ending_net_pnl_total": 100000.0, "change_vs_previous_day": 100000.0},
                {"day_utc": "20260502", "ending_net_pnl_total": 100.0, "change_vs_previous_day": -99900.0},
            ]
            + [
                {"day_utc": f"202605{day:02d}", "ending_net_pnl_total": 500.0 + day, "change_vs_previous_day": 1.0}
                for day in range(3, 31)
            ],
        },
    )
    _write_json(
        health / "training_runtime_control_latest.json",
        {
            "overall_status": "constrained",
            "training_launch_contract": {"prep_allowed": True},
            "pretraining_drain_buffer": {"safe_to_launch_now": True},
        },
    )

    payload = module.build_payload(tmp_path, bot_logs_root=tmp_path, bot_logs_min_free_gb=0.1)
    by_id = {row["section_id"]: row for row in payload["sections"]}

    assert payload["income_readiness_score"] == 100.0
    assert payload["income_readiness_grade"] == "A+"
    assert payload["live_execution_allowed"] is False
    assert payload["live_micro_allowed"] is False
    assert by_id["drawdown_governor"]["evidence"]["raw_drawdown_grade"] == "F"
    assert by_id["drawdown_governor"]["evidence"]["controlled_drawdown_ready"] is True
    assert "raw_paper_drawdown_ratio_needs_clean_refreshes" in by_id["drawdown_governor"]["blockers"]
    assert by_id["promotion_rules_for_money"]["evidence"]["raw_promotion_rules_score"] < 100.0
    assert by_id["promotion_rules_for_money"]["evidence"]["controlled_money_promotion_ready"] is True
