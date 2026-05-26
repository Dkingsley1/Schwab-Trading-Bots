#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, load_json, ordered_unique, write_payload


DEFAULT_HEALTH_OUT = PROJECT_ROOT / "governance" / "health" / "master_grandmaster_profitability_training_latest.json"
DEFAULT_MODEL_OUT = PROJECT_ROOT / "governance" / "models" / "master_grandmaster_profitability_calibration_latest.json"
DEFAULT_CONTROL_PATH = PROJECT_ROOT / "governance" / "health" / "paper_runtime_profitability_controls_latest.json"
DEFAULT_TRAINING_RUNTIME_PATH = PROJECT_ROOT / "governance" / "health" / "training_runtime_control_latest.json"
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "trade_history" / "trade_learning_dataset.json"

TARGETS = [
    "master_trend_bot",
    "master_mean_revert_bot",
    "master_shock_bot",
    "grand_master_bot",
]

PAPER_PROFITABILITY_FEATURES = [
    "paper_profitability_master_awareness_active_norm",
    "paper_profitability_master_profit_score_norm",
    "paper_profitability_master_drag_norm",
    "paper_profitability_master_training_weight_norm",
    "paper_profitability_master_size_multiplier_norm",
    "paper_profitability_master_risk_norm",
    "paper_profitability_grandmaster_awareness_active_norm",
    "paper_profitability_grandmaster_profit_score_norm",
    "paper_profitability_grandmaster_drag_norm",
    "paper_profitability_grandmaster_training_weight_norm",
    "paper_profitability_grandmaster_size_multiplier_norm",
    "paper_profitability_grandmaster_risk_norm",
    "paper_profitability_grandmaster_exit_pressure_norm",
    "paper_profitability_grandmaster_execution_discount_norm",
    "paper_profitability_grandmaster_conflict_cap_norm",
]

DEFAULT_SUB_BOT_ACCURACY_TARGET_CONTRACT = {
    "active": True,
    "desired_out_of_sample_accuracy_band": {"min": 0.80, "max": 0.90},
    "target_is_not_forced": True,
    "min_walk_forward_runs": 12,
    "min_regime_count": 3,
    "min_oos_samples": 300,
    "max_train_test_accuracy_gap": 0.08,
    "max_single_side_action_share": 0.70,
    "min_side_precision": 0.50,
    "min_calibration_score": 0.68,
    "max_duplicate_alpha_overlap_norm": 0.82,
    "accept_only_if": [
        "walk_forward_out_of_sample_accuracy_in_80_90_band",
        "train_test_gap_at_or_below_0_08",
        "no_single_side_or_overacted_collapse",
        "label_balance_and_side_precision_pass",
        "cross_regime_validation_passes",
        "duplicate_alpha_overlap_below_cap",
        "paper_profitability_drag_controls_are_clean_or_deweighted",
    ],
    "reject_if": [
        "accuracy_above_0_90_without_large_cross_regime_sample",
        "accuracy_above_0_90_with_train_test_gap_breach",
        "positive_or_negative_label_collapse",
        "overacted_or_one_sided_decision_surface",
        "future_leakage_or_same_bar_outcome_feature_detected",
        "duplicate_alpha_overlap_cluster_is_high",
    ],
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if out != out:
        return float(default)
    return out


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(max(float(value), low), high)


def _normal_counts(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    return {
        str(key): max(_safe_int(value, 0), 0)
        for key, value in raw.items()
        if str(key).strip()
    }


def _dataset_stats(project_root: Path) -> dict[str, Any]:
    dataset_path = project_root / "data" / "trade_history" / "trade_learning_dataset.json"
    dataset = load_json(dataset_path)
    data_rows = dataset.get("data") if isinstance(dataset.get("data"), list) else []
    label_counts = _normal_counts(dataset.get("label_counts"))
    if not label_counts and data_rows:
        label_counts = dict(Counter(str(row.get("label") or "unknown") for row in data_rows if isinstance(row, dict)))

    regime_counts_raw = dataset.get("regime_label_counts") if isinstance(dataset.get("regime_label_counts"), dict) else {}
    regime_count = len([key for key, value in regime_counts_raw.items() if str(key).strip() and isinstance(value, dict)])
    if regime_count <= 0 and data_rows:
        regime_count = len({str(row.get("regime") or "") for row in data_rows if isinstance(row, dict) and str(row.get("regime") or "").strip()})

    feature_names = dataset.get("feature_names") if isinstance(dataset.get("feature_names"), list) else []
    feature_name_set = {str(name) for name in feature_names}
    missing_paper_features = [name for name in PAPER_PROFITABILITY_FEATURES if name not in feature_name_set]
    rows = _safe_int(dataset.get("rows"), len(data_rows))
    total_labels = sum(label_counts.values())
    max_label_share = (max(label_counts.values()) / total_labels) if total_labels else 0.0
    positive_share = label_counts.get("positive", 0) / total_labels if total_labels else 0.0
    negative_share = label_counts.get("negative", 0) / total_labels if total_labels else 0.0
    neutral_share = label_counts.get("neutral", 0) / total_labels if total_labels else 0.0
    return {
        "path": str(dataset_path),
        "exists": dataset_path.exists(),
        "timestamp_utc": str(dataset.get("timestamp_utc") or ""),
        "rows": rows,
        "label_counts": label_counts,
        "positive_share": round(positive_share, 6),
        "negative_share": round(negative_share, 6),
        "neutral_share": round(neutral_share, 6),
        "max_label_share": round(max_label_share, 6),
        "regime_count": regime_count,
        "feature_dim": _safe_int(dataset.get("feature_dim"), len(feature_names)),
        "paper_profitability_feature_count": len(PAPER_PROFITABILITY_FEATURES) - len(missing_paper_features),
        "missing_paper_profitability_features": missing_paper_features,
        "refresh_needed_for_new_profitability_features": bool(missing_paper_features),
    }


def _training_runtime_stats(project_root: Path) -> dict[str, Any]:
    runtime_path = project_root / "governance" / "health" / "training_runtime_control_latest.json"
    runtime = load_json(runtime_path)
    snapshot = runtime.get("snapshot") if isinstance(runtime.get("snapshot"), dict) else {}
    headroom = runtime.get("host_training_headroom_gate") if isinstance(runtime.get("host_training_headroom_gate"), dict) else {}
    launch = runtime.get("training_launch_contract") if isinstance(runtime.get("training_launch_contract"), dict) else {}
    canary_batch = launch.get("canary_batch") if isinstance(launch.get("canary_batch"), list) else []
    return {
        "path": str(runtime_path),
        "overall_status": str(runtime.get("overall_status") or ""),
        "snapshot_ready": bool(runtime.get("snapshot_ready", False)),
        "snapshot_age_minutes": round(_safe_float(runtime.get("snapshot_age_minutes"), 0.0), 3),
        "snapshot_row_count": _safe_int(snapshot.get("row_count"), 0),
        "snapshot_sequence_count": _safe_int(snapshot.get("sequence_count"), 0),
        "safe_for_training": bool(headroom.get("safe_for_training", False)),
        "batch_cap": _safe_int(headroom.get("batch_cap"), 0),
        "selected_training_profile": str(headroom.get("selected_training_profile") or ""),
        "small_batch_training_safe": bool(headroom.get("small_batch_training_safe", False)),
        "batch10_training_safe": bool(headroom.get("batch10_training_safe", False)),
        "batch20_training_safe": bool(headroom.get("batch20_training_safe", False)),
        "recommended_retrain_command": launch.get("recommended_retrain_command") if isinstance(launch.get("recommended_retrain_command"), list) else [],
        "canary_batch": canary_batch,
    }


def _control_contract(project_root: Path) -> dict[str, Any]:
    control_path = project_root / "governance" / "health" / "paper_runtime_profitability_controls_latest.json"
    control = load_json(control_path)
    contract = (
        control.get("master_grandmaster_training_contract")
        if isinstance(control.get("master_grandmaster_training_contract"), dict)
        else {}
    )
    sub_contract = (
        control.get("sub_bot_accuracy_target_contract")
        if isinstance(control.get("sub_bot_accuracy_target_contract"), dict)
        else contract.get("sub_bot_accuracy_target_contract")
        if isinstance(contract.get("sub_bot_accuracy_target_contract"), dict)
        else DEFAULT_SUB_BOT_ACCURACY_TARGET_CONTRACT
    )
    return {
        "path": str(control_path),
        "control": control,
        "contract": contract,
        "sub_bot_accuracy_target_contract": sub_contract,
    }


def _overfitting_awareness(project_root: Path) -> dict[str, Any]:
    payload = load_json(project_root / "governance" / "health" / "overfitting_awareness_latest.json")
    if not payload:
        return {
            "overall_status": "missing",
            "risk_bot_count": 0,
            "hard_risk_bot_count": 0,
            "blocked_teacher_bot_count": 0,
            "teacher_ineligible_bot_count": 0,
            "policy": "missing awareness artifact; retain conservative anti-overfit defaults",
        }
    return {
        "overall_status": str(payload.get("overall_status") or "ready"),
        "risk_bot_count": _safe_int(payload.get("risk_bot_count"), 0),
        "hard_risk_bot_count": _safe_int(payload.get("hard_risk_bot_count"), 0),
        "blocked_teacher_bot_count": _safe_int(payload.get("blocked_teacher_bot_count"), 0),
        "teacher_ineligible_bot_count": _safe_int(payload.get("teacher_ineligible_bot_count"), 0),
        "active_status_counts": payload.get("active_status_counts") if isinstance(payload.get("active_status_counts"), dict) else {},
        "top_risk_bots": payload.get("top_risk_bots")[:8] if isinstance(payload.get("top_risk_bots"), list) else [],
        "policy": "masters and Grand Master must downweight overfit-risk bot votes and block full-time promotion while guarded/blocked",
    }


def _anti_overfit_assessment(dataset: dict[str, Any], sub_contract: dict[str, Any]) -> dict[str, Any]:
    min_samples = _safe_int(sub_contract.get("min_oos_samples"), 300)
    min_regimes = _safe_int(sub_contract.get("min_regime_count"), 3)
    max_label_share_cap = _safe_float(sub_contract.get("max_single_side_action_share"), 0.70)
    checks = [
        {
            "check": "min_labeled_oos_samples",
            "passed": dataset.get("rows", 0) >= min_samples,
            "observed": dataset.get("rows", 0),
            "required": min_samples,
        },
        {
            "check": "min_regime_count",
            "passed": dataset.get("regime_count", 0) >= min_regimes,
            "observed": dataset.get("regime_count", 0),
            "required": min_regimes,
        },
        {
            "check": "label_balance_not_collapsed",
            "passed": _safe_float(dataset.get("max_label_share"), 1.0) <= max_label_share_cap,
            "observed": dataset.get("max_label_share", 0.0),
            "required_max": max_label_share_cap,
        },
        {
            "check": "paper_profitability_features_present",
            "passed": not bool(dataset.get("missing_paper_profitability_features")),
            "observed_missing_count": len(dataset.get("missing_paper_profitability_features") or []),
            "required_missing_count": 0,
        },
    ]
    passed = [row["check"] for row in checks if row.get("passed")]
    failed = [row["check"] for row in checks if not row.get("passed")]
    return {
        "overall_status": "clean" if not failed else "guarded",
        "checks": checks,
        "passed_checks": passed,
        "failed_checks": failed,
        "accuracy_target_policy": sub_contract,
        "interpretation": "80-90% is admissible only after these checks pass out of sample; above 90% is treated as suspicious unless sample and regime coverage are large.",
    }


def _learned_calibration(control_bundle: dict[str, Any], dataset: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    contract = control_bundle["contract"]
    sample_policy = contract.get("sample_weight_policy") if isinstance(contract.get("sample_weight_policy"), dict) else {}
    gate_policy = contract.get("promotion_gate_policy") if isinstance(contract.get("promotion_gate_policy"), dict) else {}
    mean_profit = _clamp(_safe_float(contract.get("mean_profit_score_norm"), 0.5))
    max_drag = _clamp(_safe_float(contract.get("max_drag_score_norm"), 0.0))
    mean_size = _clamp(_safe_float(contract.get("mean_position_size_multiplier_norm"), 1.0))
    labeled_rows = _safe_int(dataset.get("rows"), 0)
    snapshot_rows = _safe_int(runtime.get("snapshot_row_count"), 0)
    evidence_rows = max(labeled_rows, snapshot_rows)
    evidence_confidence = _clamp(evidence_rows / 5000.0)
    hard_negative = max(1.0, _safe_float(sample_policy.get("paper_loss_hard_negative_multiplier"), 1.0 + (2.0 * max_drag)))
    positive_mult = max(0.5, _safe_float(sample_policy.get("paper_profit_positive_multiplier"), 1.0))
    quarantine_mult = max(1.0, _safe_float(sample_policy.get("strategy_quarantine_multiplier"), 1.0))
    return {
        "artifact_kind": "master_grandmaster_profitability_calibration",
        "trained_on": {
            "labeled_behavior_rows": labeled_rows,
            "runtime_snapshot_rows": snapshot_rows,
            "evidence_confidence_norm": round(evidence_confidence, 6),
            "paper_profile_controls_active": len(control_bundle["control"].get("profile_controls") or {}),
            "paper_strategy_controls_active": len(control_bundle["control"].get("strategy_controls") or {}),
        },
        "paper_profitability_state": {
            "mean_profit_score_norm": round(mean_profit, 6),
            "max_drag_score_norm": round(max_drag, 6),
            "mean_position_size_multiplier_norm": round(mean_size, 6),
        },
        "master_layer": {
            "profit_score_floor_norm": round(_safe_float(gate_policy.get("require_profit_score_floor_norm"), 0.62), 6),
            "drag_score_ceiling_norm": round(_safe_float(gate_policy.get("require_drag_score_below_norm"), 0.38), 6),
            "risk_damp_scale_norm": round(_clamp(0.35 + (0.55 * max_drag)), 6),
            "vote_weight_multiplier_norm": round(_clamp(0.55 + (0.35 * mean_profit) - (0.45 * max_drag), 0.10, 0.95), 6),
            "position_size_multiplier_norm": round(max(0.05, min(mean_size, 1.0)), 6),
        },
        "grandmaster_layer": {
            "hold_or_block_drag_threshold_norm": round(_clamp(0.48 + (0.18 * max_drag)), 6),
            "exit_pressure_multiplier_norm": round(_clamp(0.30 + (0.60 * max_drag)), 6),
            "execution_discount_norm": round(_clamp(0.12 + (0.40 * max_drag)), 6),
            "conflict_cap_norm": round(_clamp(0.74 - (0.24 * max_drag), 0.42, 0.74), 6),
            "release_requires_no_active_quarantine": True,
        },
        "sample_weight_policy": {
            "paper_loss_hard_negative_multiplier": round(hard_negative, 6),
            "paper_profit_positive_multiplier": round(positive_mult, 6),
            "strategy_quarantine_multiplier": round(quarantine_mult, 6),
            "max_effective_weight_cap": round(min(max(hard_negative, quarantine_mult), 3.50), 6),
        },
        "sub_bot_target_policy": control_bundle["sub_bot_accuracy_target_contract"],
    }


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    dataset = _dataset_stats(project_root)
    runtime = _training_runtime_stats(project_root)
    control_bundle = _control_contract(project_root)
    overfit_awareness = _overfitting_awareness(project_root)
    contract = control_bundle["contract"]
    targets = contract.get("trainable_targets") if isinstance(contract.get("trainable_targets"), list) else TARGETS
    targets = ordered_unique(str(target) for target in targets) or TARGETS
    calibration = _learned_calibration(control_bundle, dataset, runtime)
    anti_overfit = _anti_overfit_assessment(dataset, control_bundle["sub_bot_accuracy_target_contract"])
    active = bool(contract.get("active", False))
    if not active:
        overall_status = "ready_no_active_profitability_drag"
    elif runtime.get("snapshot_row_count", 0) >= 300:
        overall_status = "trained_protective_calibration"
    else:
        overall_status = "prepared_waiting_for_more_runtime_rows"
    blockers = []
    if dataset.get("refresh_needed_for_new_profitability_features"):
        blockers.append("behavior_dataset_refresh_needed_for_paper_profitability_features")
    blockers.extend(anti_overfit.get("failed_checks") or [])
    if _safe_int(overfit_awareness.get("risk_bot_count"), 0) > 0 or str(overfit_awareness.get("overall_status") or "") in {"guarded", "blocked"}:
        blockers.append("overfitting_awareness_risk")
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": overall_status in {
            "ready_no_active_profitability_drag",
            "trained_protective_calibration",
            "prepared_waiting_for_more_runtime_rows",
        },
        "overall_status": overall_status,
        "trained_targets": targets,
        "training_mode": "master_profitability_canary",
        "dataset": dataset,
        "runtime_training_gate": runtime,
        "master_grandmaster_training_contract": contract,
        "anti_overfit_assessment": anti_overfit,
        "overfitting_awareness": overfit_awareness,
        "learned_calibration": calibration,
        "blockers_for_full_80_90_release": ordered_unique(blockers),
        "recommended_next_actions": ordered_unique(
            [
                "apply the learned protective calibration to the master and Grand Master profitability artifact",
                "refresh behavior dataset so the new paper profitability features appear in feature_names"
                if dataset.get("refresh_needed_for_new_profitability_features")
                else "",
                "run the governor-approved micro-canary sub-bot training before widening",
                "do not accept 80-90% sub-bot accuracy unless walk-forward and anti-overfit checks pass",
                "block master/Grand Master promotion votes from overfit-risk bots until the awareness layer returns ready"
                if str(overfit_awareness.get("overall_status") or "") in {"guarded", "blocked"}
                else "",
            ]
        ),
        "recommended_retrain_command": runtime.get("recommended_retrain_command") or [],
        "source_files": {
            "paper_runtime_profitability_controls": str(project_root / "governance" / "health" / "paper_runtime_profitability_controls_latest.json"),
            "overfitting_awareness": str(project_root / "governance" / "health" / "overfitting_awareness_latest.json"),
            "training_runtime_control": str(project_root / "governance" / "health" / "training_runtime_control_latest.json"),
            "trade_learning_dataset": str(project_root / "data" / "trade_history" / "trade_learning_dataset.json"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a guarded profitability calibration for master and Grand Master decision layers.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_HEALTH_OUT))
    parser.add_argument("--model-out", default=str(DEFAULT_MODEL_OUT))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    payload = build_payload(project_root)
    out_path = Path(args.out_file).expanduser()
    if not out_path.is_absolute():
        out_path = project_root / out_path
    model_path = Path(args.model_out).expanduser()
    if not model_path.is_absolute():
        model_path = project_root / model_path

    if args.apply:
        write_payload(model_path, payload["learned_calibration"])
        payload["applied_model_file"] = str(model_path)
        payload["applied_model_summary"] = {
            "trained_targets": payload.get("trained_targets", []),
            "training_mode": payload.get("training_mode", ""),
            "overall_status": payload.get("overall_status", ""),
        }

    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "master_grandmaster_profitability_trainer "
            f"overall_status={payload.get('overall_status', '')} "
            f"targets={len(payload.get('trained_targets') or [])} "
            f"snapshot_rows={payload.get('runtime_training_gate', {}).get('snapshot_row_count', 0)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
