import json
from pathlib import Path

from scripts.ops import profitability_evidence_firewall as firewall


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_firewall_separates_implemented_controls_from_earned_profitability(tmp_path: Path) -> None:
    config = json.loads(firewall.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    config_path = tmp_path / "config" / firewall.DEFAULT_CONFIG_PATH.name
    _write_json(config_path, config)
    base_trader = tmp_path / "core" / "base_trader.py"
    base_trader.parent.mkdir(parents=True, exist_ok=True)
    base_trader.write_text("paper_profitability_clean_profile_evidence_block = True\n", encoding="utf-8")
    shadow_loop = tmp_path / "scripts" / "run_shadow_training_loop.py"
    shadow_loop.parent.mkdir(parents=True, exist_ok=True)
    shadow_loop.write_text(
        "evaluate_paper_execution_authority require_hierarchy_identity=True PAPER_EXECUTION_COHORT_MAX_PER_SEGMENT\n",
        encoding="utf-8",
    )
    performance_report = tmp_path / "scripts" / "paper_performance_report.py"
    performance_report.write_text(
        "lifetime_flow current_day_flow candidate_forward_flow active_book_snapshot financial_grade_eligible carried_forward\n",
        encoding="utf-8",
    )
    for relative in (
        "scripts/ops/independent_fill_evidence_acquisition.py",
        "scripts/ops/profitability_holdout_vault.py",
        "scripts/ops/profitability_benchmark_capture.py",
        "scripts/ops/profitability_benchmark_hurdle.py",
        "scripts/ops/profitability_independent_validator.py",
        "scripts/multiple_testing_guard.py",
        "scripts/decay_monitor.py",
        "core/profitability_statistics.py",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# producer fixture\n", encoding="utf-8")
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "source_verification_latest.json",
        {"ok": False, "overall": {"all_verified": False, "mean_source_confidence_score": 0.2, "min_source_confidence_score": 0.1}},
    )
    _write_json(
        health / "paper_execution_calibration_latest.json",
        {"independent_samples": 0, "independent_evidence_ready": False},
    )
    _write_json(
        health / "paper_performance_latest.json",
        {
            "post_cost_expectancy": {
                "sample_count": 0,
                "status": "no_schema_v2_trade_deltas",
            },
            "sleeve_latest": [],
            "sleeve_daily_series": {},
            "accounting_views": {
                "lifetime_flow": {},
                "current_day_flow": {},
                "candidate_forward_flow": {"candidate_id": "candidate-test", "row_count": 0},
                "active_book_snapshot": {},
            },
        },
    )
    _write_json(
        health / "paper_live_data_standard_latest.json",
        {
            "ok": True,
            "safety_contract": {
                "paper_execution_authority_version": "paper_execution_authority_v2",
                "paper_mirror_all_active_sub_bots": "0",
                "unauthorized_execution_bot_ids": [],
                "execution_authority_count": 9,
                "execution_authority_hard_cap": 24,
            },
        },
    )
    _write_json(
        health / "paper_runtime_profitability_controls_latest.json",
        {
            "raw_profitability_grade": "D",
            "raw_profitability_improvement_contract": {
                "clean_sleeve_strict_buy_gate_contract": {
                    "active": True,
                    "enforced": True,
                    "allow_buy_only_when_all_gates_pass": True,
                },
                "weak_sleeve_zero_entry_contract": {
                    "profiles": [
                        {"profile": "futures_event_reaction", "block_new_entries": True, "new_entry_cap": 0},
                        {"profile": "options_on_futures_aggressive", "block_new_entries": True, "new_entry_cap": 0},
                    ]
                },
            },
            "scout_collection_contract": {
                "required_label_outputs": config["counterfactual_labels"],
            },
        },
    )
    _write_json(health / "counterfactual_replay_latest.json", {"ok": True, "candidate_count": 4})
    _write_json(
        tmp_path / "governance" / "research" / "multiple_testing_guard_latest.json",
        {"statistical_evidence_ready": False},
    )

    payload = firewall.build_payload(tmp_path, config_path=config_path)

    assert payload["control_grade"] == "A+"
    assert payload["overall_status"] == "ready_with_evidence_debt"
    assert payload["control_implementation_ready"] is True
    assert payload["economic_evidence_ready"] is False
    assert payload["live_promotion_ready"] is False
    assert payload["promotion_evidence_ready"] is False
    assert payload["economic_evidence_grade"] != "A+"
    assert payload["raw_profitability_grade"] == "D"
    assert payload["raw_profitability_grade_overridden"] is False
    assert "baseline:07_cluster_effective_samples" in payload["blockers"]
    assert payload["future_profitability_hardening"]["control_grade"] == "A+"
    assert payload["future_profitability_hardening"]["economic_evidence_grade"] != "A+"


def test_profitability_epoch_contract_rejects_cross_cycle_evidence(tmp_path: Path) -> None:
    (tmp_path / "master_bot_registry.json").write_text("{}", encoding="utf-8")
    artifacts = {
        "performance": ({"timestamp_utc": "2026-08-10T12:00:00+00:00", "evidence_epoch": {"id": "epoch-a"}}, tmp_path / "performance.json"),
        "validator": ({"timestamp_utc": "2026-08-10T12:01:00+00:00", "evidence_epoch": {"id": "epoch-b"}}, tmp_path / "validator.json"),
    }

    contract = firewall._profitability_evidence_epoch_contract(tmp_path, artifacts)

    assert contract["enforced"] is True
    assert contract["ready"] is False
    assert contract["epoch_ids"] == ["epoch-a", "epoch-b"]
    assert len(contract["receipt_sha256"]) == 64
