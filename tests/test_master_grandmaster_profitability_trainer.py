import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "ops" / "master_grandmaster_profitability_trainer.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("master_grandmaster_profitability_trainer", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load master_grandmaster_profitability_trainer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_master_grandmaster_trainer_builds_guarded_calibration(tmp_path: Path) -> None:
    module = _load_module()
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "paper_runtime_profitability_controls_latest.json",
        {
            "profile_controls": {"aggressive": {}, "crypto": {}},
            "strategy_controls": {"aggressive::brain_refinery_v10": {}},
            "master_grandmaster_training_contract": {
                "active": True,
                "trainable_targets": ["master_trend_bot", "grand_master_bot"],
                "mean_profit_score_norm": 0.40,
                "max_drag_score_norm": 0.75,
                "mean_position_size_multiplier_norm": 0.44,
                "sample_weight_policy": {
                    "paper_loss_hard_negative_multiplier": 2.5,
                    "paper_profit_positive_multiplier": 0.9,
                    "strategy_quarantine_multiplier": 1.5,
                },
                "promotion_gate_policy": {
                    "require_profit_score_floor_norm": 0.62,
                    "require_drag_score_below_norm": 0.38,
                },
            },
            "sub_bot_accuracy_target_contract": {
                "active": True,
                "desired_out_of_sample_accuracy_band": {"min": 0.80, "max": 0.90},
                "target_is_not_forced": True,
                "min_regime_count": 3,
                "min_oos_samples": 300,
                "max_single_side_action_share": 0.70,
            },
        },
    )
    _write_json(
        health / "training_runtime_control_latest.json",
        {
            "overall_status": "ready",
            "snapshot_ready": True,
            "snapshot_age_minutes": 4.0,
            "snapshot": {"row_count": 1200, "sequence_count": 120},
            "host_training_headroom_gate": {
                "safe_for_training": True,
                "batch_cap": 1,
                "selected_training_profile": "coverage_micro_canary",
            },
            "training_launch_contract": {
                "recommended_retrain_command": ["./scripts/ops/opsctl.sh", "retrain-force-targeted"],
                "canary_batch": [{"bot_id": "brain_refinery_v45_intraday_open_close_regimes"}],
            },
        },
    )
    feature_names = ["x", *module.PAPER_PROFITABILITY_FEATURES]
    _write_json(
        tmp_path / "data" / "trade_history" / "trade_learning_dataset.json",
        {
            "timestamp_utc": "2026-05-23T00:00:00+00:00",
            "rows": 360,
            "feature_dim": len(feature_names),
            "feature_names": feature_names,
            "label_counts": {"positive": 120, "negative": 120, "neutral": 120},
            "regime_label_counts": {
                "trend": {"positive": 40, "negative": 40, "neutral": 40},
                "mean_revert": {"positive": 40, "negative": 40, "neutral": 40},
                "shock": {"positive": 40, "negative": 40, "neutral": 40},
            },
            "data": [],
        },
    )

    payload = module.build_payload(tmp_path)

    assert payload["overall_status"] == "trained_protective_calibration"
    assert payload["trained_targets"] == ["master_trend_bot", "grand_master_bot"]
    assert payload["anti_overfit_assessment"]["overall_status"] == "clean"
    assert payload["learned_calibration"]["sample_weight_policy"]["paper_loss_hard_negative_multiplier"] == 2.5
    assert payload["learned_calibration"]["master_layer"]["position_size_multiplier_norm"] == 0.44
    assert payload["runtime_training_gate"]["recommended_retrain_command"] == ["./scripts/ops/opsctl.sh", "retrain-force-targeted"]


def test_master_grandmaster_trainer_flags_dataset_refresh_needed(tmp_path: Path) -> None:
    module = _load_module()
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "paper_runtime_profitability_controls_latest.json",
        {"master_grandmaster_training_contract": {"active": True}},
    )
    _write_json(
        health / "training_runtime_control_latest.json",
        {"snapshot": {"row_count": 500}},
    )
    _write_json(
        tmp_path / "data" / "trade_history" / "trade_learning_dataset.json",
        {
            "rows": 20,
            "feature_names": ["pnl_proxy"],
            "label_counts": {"positive": 20},
            "regime_label_counts": {"shock": {"positive": 20}},
        },
    )

    payload = module.build_payload(tmp_path)

    assert payload["anti_overfit_assessment"]["overall_status"] == "guarded"
    assert "behavior_dataset_refresh_needed_for_paper_profitability_features" in payload["blockers_for_full_80_90_release"]


def test_master_grandmaster_trainer_consumes_overfitting_awareness(tmp_path: Path) -> None:
    module = _load_module()
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "paper_runtime_profitability_controls_latest.json",
        {"master_grandmaster_training_contract": {"active": True}},
    )
    _write_json(health / "training_runtime_control_latest.json", {"snapshot": {"row_count": 500}})
    _write_json(
        health / "overfitting_awareness_latest.json",
        {
            "overall_status": "guarded",
            "risk_bot_count": 2,
            "hard_risk_bot_count": 0,
            "blocked_teacher_bot_count": 2,
            "teacher_ineligible_bot_count": 5,
        },
    )
    feature_names = ["x", *module.PAPER_PROFITABILITY_FEATURES]
    _write_json(
        tmp_path / "data" / "trade_history" / "trade_learning_dataset.json",
        {
            "rows": 360,
            "feature_names": feature_names,
            "label_counts": {"positive": 120, "negative": 120, "neutral": 120},
            "regime_label_counts": {
                "trend": {"positive": 40, "negative": 40, "neutral": 40},
                "mean_revert": {"positive": 40, "negative": 40, "neutral": 40},
                "shock": {"positive": 40, "negative": 40, "neutral": 40},
            },
        },
    )

    payload = module.build_payload(tmp_path)

    assert payload["overfitting_awareness"]["overall_status"] == "guarded"
    assert payload["overfitting_awareness"]["risk_bot_count"] == 2
    assert "overfitting_awareness_risk" in payload["blockers_for_full_80_90_release"]
