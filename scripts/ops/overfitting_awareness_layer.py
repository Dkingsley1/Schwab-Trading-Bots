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
    from scripts.ops.long_runtime_common import PROJECT_ROOT, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "overfitting_awareness_latest.json"

TIER_ORDER = ("infrastructure", "sub", "teacher", "master", "grand_master")
GRAND_MARKERS = ("grandmaster", "grand_master", "grand master")
MASTER_MARKERS = ("sleeve_master", "master_bot", "master_coordination", "per_sleeve_master_bots")
INFRA_MARKERS = ("infrastructure", "infra", "guard", "watchdog", "supervisor", "validator")


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        out = float(raw)
    except Exception:
        return float(default)
    if out != out:
        return float(default)
    return out


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(max(float(value), low), high)


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _registry_rows(project_root: Path) -> list[dict[str, Any]]:
    payload = load_json(project_root / "master_bot_registry.json")
    rows = payload.get("sub_bots") if isinstance(payload.get("sub_bots"), list) else []
    return [row for row in rows if isinstance(row, dict)]


def _bot_id(row: dict[str, Any]) -> str:
    return str(row.get("bot_id") or "").strip()


def _is_active(row: dict[str, Any]) -> bool:
    if row.get("deleted") is True:
        return False
    lifecycle = str(row.get("lifecycle_state") or "").strip().lower()
    if lifecycle in {"deleted", "retired", "archived"}:
        return False
    return bool(row.get("active", False))


def _str_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    text = str(raw or "").strip()
    return [text] if text else []


def _classify_tier(row: dict[str, Any]) -> str:
    text = " ".join(
        [
            str(row.get("bot_id") or ""),
            str(row.get("bot_role") or ""),
            str(row.get("slot_kind") or ""),
            str(row.get("bot_intelligence_layer") or ""),
            str(row.get("sleeve_profile") or ""),
        ]
        + _str_list(row.get("target_functions"))
    ).lower()
    role = str(row.get("bot_role") or "").strip().lower()
    if any(marker in text for marker in GRAND_MARKERS):
        return "grand_master"
    if any(marker in text for marker in MASTER_MARKERS):
        return "master"
    if role == "infrastructure_sub_bot" or any(marker in text for marker in INFRA_MARKERS):
        return "infrastructure"
    return "sub"


def _contract_from_runtime(project_root: Path) -> dict[str, Any]:
    paper_control = load_json(project_root / "governance" / "health" / "paper_runtime_profitability_controls_latest.json")
    contract = paper_control.get("sub_bot_accuracy_target_contract")
    if not isinstance(contract, dict):
        master_contract = paper_control.get("master_grandmaster_training_contract") if isinstance(paper_control.get("master_grandmaster_training_contract"), dict) else {}
        contract = master_contract.get("sub_bot_accuracy_target_contract") if isinstance(master_contract.get("sub_bot_accuracy_target_contract"), dict) else {}
    return {
        "active": bool(contract.get("active", True)),
        "desired_out_of_sample_accuracy_band": contract.get("desired_out_of_sample_accuracy_band") or {"min": 0.80, "max": 0.90},
        "target_is_not_forced": bool(contract.get("target_is_not_forced", True)),
        "min_walk_forward_runs": _safe_int(contract.get("min_walk_forward_runs"), 12),
        "min_regime_count": _safe_int(contract.get("min_regime_count"), 3),
        "min_oos_samples": _safe_int(contract.get("min_oos_samples"), 300),
        "max_train_test_accuracy_gap": _safe_float(contract.get("max_train_test_accuracy_gap"), 0.08),
        "max_single_side_action_share": _safe_float(contract.get("max_single_side_action_share"), 0.70),
        "min_side_precision": _safe_float(contract.get("min_side_precision"), 0.50),
        "min_calibration_score": _safe_float(contract.get("min_calibration_score"), 0.68),
        "max_duplicate_alpha_overlap_norm": _safe_float(contract.get("max_duplicate_alpha_overlap_norm"), 0.82),
    }


def _example_map(rows: list[Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        bot_id = str(row.get("bot_id") or "").strip().lower()
        if bot_id:
            out[bot_id] = row
    return out


def _diagnostic_for(project_root: Path, bot_id: str) -> dict[str, Any]:
    return load_json(project_root / "governance" / "training_diagnostics" / f"{bot_id}_latest.json")


def _risk_status(
    *,
    bot_id: str,
    wf_row: dict[str, Any],
    leak_guard: dict[str, Any],
    registry_row: dict[str, Any],
    diagnostic_row: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    leak_like = _example_map(_as_list(leak_guard.get("leak_like_examples")))
    severe = _example_map(_as_list(leak_guard.get("severe_overfit_examples")))
    overfit = _example_map(_as_list(leak_guard.get("overfit_examples")))
    bot_key = bot_id.lower()
    runs = _safe_int(wf_row.get("runs"), 0)
    train_mean = _safe_float(wf_row.get("train_mean"), 0.0)
    forward_mean = _safe_float(wf_row.get("forward_mean"), 0.0)
    delta = _safe_float(wf_row.get("delta"), 0.0)
    gap = train_mean - forward_mean
    registry_accuracy = max(
        _safe_float(registry_row.get("test_accuracy"), 0.0),
        _safe_float(registry_row.get("candidate_test_accuracy"), 0.0),
    )
    quality_score = max(
        _safe_float(registry_row.get("quality_score"), 0.0),
        _safe_float(registry_row.get("candidate_quality_score"), 0.0),
    )
    max_gap = _safe_float(thresholds.get("max_overfit_gap"), 0.08)
    severe_gap = _safe_float(thresholds.get("max_severe_overfit_gap"), 0.14)
    high_train = _safe_float(thresholds.get("high_train_threshold"), 0.90)
    target_band = thresholds.get("desired_out_of_sample_accuracy_band") if isinstance(thresholds.get("desired_out_of_sample_accuracy_band"), dict) else {}
    target_max = _safe_float(target_band.get("max"), 0.90)
    min_runs = _safe_int(thresholds.get("min_walk_forward_runs"), 12)
    min_oos_samples = _safe_int(thresholds.get("min_oos_samples"), 300)
    min_regime_count = _safe_int(thresholds.get("min_regime_count"), 3)
    runtime_meta = _as_dict(diagnostic_row.get("runtime_meta"))
    sample_count = max(_safe_int(diagnostic_row.get("sample_count"), 0), _safe_int(runtime_meta.get("sample_count"), 0))
    observation_count = max(
        _safe_int(diagnostic_row.get("observation_count"), 0),
        _safe_int(runtime_meta.get("observation_count"), 0),
        _safe_int(registry_row.get("data_collection_observations"), 0),
        _safe_int(registry_row.get("collected_observation_count"), 0),
    )
    eligible_sequences = max(
        _safe_int(diagnostic_row.get("eligible_sequences"), 0),
        _safe_int(runtime_meta.get("eligible_sequences"), 0),
        _safe_int(diagnostic_row.get("sequence_count"), 0),
    )
    sample_starved = bool(sample_count < min_oos_samples and observation_count < min_oos_samples)
    sequence_starved = bool(eligible_sequences > 0 and eligible_sequences < min_regime_count)

    reasons: list[str] = []
    status = "insufficient_evidence"
    if bot_key in leak_like:
        status = "leak_like"
        reasons.append("high train score with weak forward score")
    elif bot_key in severe or gap > severe_gap:
        status = "severe_overfit"
        reasons.append("train-forward gap above severe threshold")
    elif bot_key in overfit or gap > max_gap:
        status = "overfit_watch"
        reasons.append("train-forward gap above overfit threshold")
    elif registry_accuracy > target_max and runs < min_runs and sample_starved:
        status = "insufficient_evidence"
        reasons.append("high registry score ignored until sample and observation floors clear")
    elif registry_accuracy > target_max and runs < max(min_runs * 2, 20):
        status = "high_accuracy_guarded"
        reasons.append("high accuracy needs extra cross-regime confirmation before trust")
        if sequence_starved:
            reasons.append("eligible sequence coverage is below cross-regime floor")
    elif runs >= min_runs:
        status = "generalization_clean"
    elif registry_accuracy > 0.0 or quality_score > 0.0:
        status = "registry_only_guarded"

    risk_score = _clamp((max(gap, 0.0) / max(severe_gap, 0.001)) * 0.70)
    if status == "leak_like":
        risk_score = max(risk_score, 1.0)
    elif status == "severe_overfit":
        risk_score = max(risk_score, 0.85)
    elif status == "overfit_watch":
        risk_score = max(risk_score, 0.58)
    elif status == "high_accuracy_guarded":
        risk_score = max(risk_score, 0.42)
    elif status in {"insufficient_evidence", "registry_only_guarded"}:
        risk_score = max(risk_score, 0.25)

    may_teach = status == "generalization_clean" and risk_score < 0.35
    may_promote = status == "generalization_clean" and risk_score < 0.35
    may_expand_live = may_promote and forward_mean >= _safe_float(target_band.get("min"), 0.80)
    if status in {"insufficient_evidence", "registry_only_guarded"}:
        reasons.append("needs walk-forward/out-of-sample evidence before higher-trust use")
    return {
        "bot_id": bot_id,
        "status": status,
        "risk_score": round(risk_score, 6),
        "runs": runs,
        "train_mean": round(train_mean, 6),
        "forward_mean": round(forward_mean, 6),
        "delta": round(delta, 6),
        "train_forward_gap": round(gap, 6),
        "registry_accuracy": round(registry_accuracy, 6),
        "quality_score": round(quality_score, 6),
        "sample_count": sample_count,
        "observation_count": observation_count,
        "eligible_sequences": eligible_sequences,
        "reasons": ordered_unique(reasons),
        "policy": {
            "may_teach": may_teach,
            "may_promote": may_promote,
            "may_expand_live": may_expand_live,
            "must_use_shadow_or_paper_only": not may_expand_live,
            "requires_generalization_canary": status in {"overfit_watch", "severe_overfit", "leak_like", "high_accuracy_guarded"},
            "requires_dataset_or_label_repair": status in {"leak_like", "severe_overfit"},
        },
        "next_action": (
            "inspect label leakage and remove same-bar/future-leaking features"
            if status == "leak_like"
            else "reduce complexity, tighten feature set, and rerun walk-forward canary"
            if status in {"severe_overfit", "overfit_watch"}
            else "run more cross-regime walk-forward evidence before allowing teacher or promotion duty"
            if status in {"high_accuracy_guarded", "insufficient_evidence", "registry_only_guarded"}
            else "eligible for teacher/promotion consideration if other gates also pass"
        ),
    }


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    rows = _registry_rows(project_root)
    registry_map = {_bot_id(row).lower(): row for row in rows if _bot_id(row)}
    wf = load_json(project_root / "governance" / "walk_forward" / "walk_forward_latest.json")
    wf_bots = wf.get("bots") if isinstance(wf.get("bots"), dict) else {}
    leak_guard = load_json(project_root / "governance" / "health" / "leak_overfit_guard_latest.json")
    master_training = load_json(project_root / "governance" / "health" / "master_grandmaster_profitability_training_latest.json")
    thresholds = dict(_contract_from_runtime(project_root))
    leak_thresholds = leak_guard.get("thresholds") if isinstance(leak_guard.get("thresholds"), dict) else {}
    thresholds.update(
        {
            "max_overfit_gap": _safe_float(leak_thresholds.get("max_overfit_gap"), thresholds["max_train_test_accuracy_gap"]),
            "max_severe_overfit_gap": _safe_float(leak_thresholds.get("max_severe_overfit_gap"), 0.14),
            "high_train_threshold": _safe_float(leak_thresholds.get("high_train_threshold"), 0.90),
            "low_forward_threshold": _safe_float(leak_thresholds.get("low_forward_threshold"), 0.55),
        }
    )

    risk_rows: list[dict[str, Any]] = []
    tier_counts: dict[str, Counter[str]] = {tier: Counter() for tier in TIER_ORDER}
    for bot_id, reg_row in registry_map.items():
        display_id = _bot_id(reg_row)
        wf_row = wf_bots.get(bot_id) if isinstance(wf_bots.get(bot_id), dict) else wf_bots.get(display_id) if isinstance(wf_bots.get(display_id), dict) else {}
        risk = _risk_status(
            bot_id=display_id,
            wf_row=wf_row if isinstance(wf_row, dict) else {},
            leak_guard=leak_guard,
            registry_row=reg_row,
            diagnostic_row=_diagnostic_for(project_root, display_id),
            thresholds=thresholds,
        )
        tier = _classify_tier(reg_row)
        if _is_active(reg_row):
            tier_counts[tier][risk["status"]] += 1
        risk["tier"] = tier
        risk["active"] = _is_active(reg_row)
        risk_rows.append(risk)

    status_counts = Counter(row["status"] for row in risk_rows if row.get("active"))
    hard_count = status_counts.get("leak_like", 0) + status_counts.get("severe_overfit", 0)
    overfit_watch_count = hard_count + status_counts.get("overfit_watch", 0)
    high_accuracy_guarded_count = status_counts.get("high_accuracy_guarded", 0)
    guarded_count = overfit_watch_count + high_accuracy_guarded_count
    overall_status = "ready"
    if hard_count:
        overall_status = "blocked"
    elif guarded_count:
        overall_status = "guarded"

    risk_rows.sort(key=lambda row: (-_safe_float(row.get("risk_score"), 0.0), str(row.get("bot_id") or "")))
    teacher_ineligible_ids = [
        str(row.get("bot_id") or "")
        for row in risk_rows
        if row.get("active") and not bool(_as_dict(row.get("policy")).get("may_teach", False))
    ]
    overfit_blocked_teacher_ids = [
        str(row.get("bot_id") or "")
        for row in risk_rows
        if row.get("active")
        and str(row.get("status") or "") in {"leak_like", "severe_overfit", "overfit_watch", "high_accuracy_guarded"}
        and not bool(_as_dict(row.get("policy")).get("may_teach", False))
    ]
    master_anti = master_training.get("anti_overfit_assessment") if isinstance(master_training.get("anti_overfit_assessment"), dict) else {}
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": overall_status != "blocked",
        "overall_status": overall_status,
        "awareness_status": overall_status,
        "thresholds": thresholds,
        "active_status_counts": dict(status_counts),
        "tier_status_counts": {tier: dict(counter) for tier, counter in tier_counts.items()},
        "risk_bot_count": overfit_watch_count,
        "hard_risk_bot_count": hard_count,
        "guarded_bot_count": guarded_count,
        "high_accuracy_guarded_bot_count": high_accuracy_guarded_count,
        "blocked_teacher_bot_count": len(overfit_blocked_teacher_ids),
        "blocked_teacher_bot_ids": overfit_blocked_teacher_ids[:80],
        "teacher_ineligible_bot_count": len(teacher_ineligible_ids),
        "teacher_ineligible_bot_ids": teacher_ineligible_ids[:80],
        "top_risk_bots": risk_rows[:40],
        "bot_risk": risk_rows,
        "master_grandmaster_awareness": {
            "anti_overfit_status": str(master_anti.get("overall_status") or ""),
            "failed_checks": master_anti.get("failed_checks") if isinstance(master_anti.get("failed_checks"), list) else [],
            "policy": "masters and Grand Master must treat high accuracy as provisional until cross-regime walk-forward, leakage, and paper-drag checks are clean",
        },
        "broadcast_contract": {
            "applies_to_tiers": list(TIER_ORDER),
            "sub_bot_rule": "optimize for out-of-sample lift and calibrated action quality; do not chase in-sample accuracy",
            "teacher_rule": "bots with overfit, leak-like, or high-accuracy-guarded status may not teach students",
            "master_rule": "masters downweight or ignore votes from overfit-risk sub-bots",
            "grand_master_rule": "Grand Master blocks promotion/full-time use when overfit awareness is guarded or blocked",
            "infrastructure_rule": "infra bots surface this artifact before training, promotion, and sleeve widening",
        },
        "recommended_actions": ordered_unique(
            [
                "run leak_overfit_guard before promotion and training batches",
                "keep high-accuracy bots provisional until cross-regime out-of-sample evidence proves the score",
                "exclude overfit-risk bots from teacher duty and master promotion voting",
                "feed bot risk rows into bot-needs, teacher-quality, and the bot intelligence mesh",
            ]
        ),
        "source_files": {
            "walk_forward": str(project_root / "governance" / "walk_forward" / "walk_forward_latest.json"),
            "leak_overfit_guard": str(project_root / "governance" / "health" / "leak_overfit_guard_latest.json"),
            "paper_runtime_profitability_controls": str(project_root / "governance" / "health" / "paper_runtime_profitability_controls_latest.json"),
            "master_grandmaster_profitability_training": str(project_root / "governance" / "health" / "master_grandmaster_profitability_training_latest.json"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Broadcast overfitting awareness to sub, teacher, master, Grand Master, and infra layers.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    payload = build_payload(project_root)
    out_path = Path(args.out_file).expanduser()
    if not out_path.is_absolute():
        out_path = project_root / out_path
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "overfitting_awareness "
            f"overall_status={payload.get('overall_status', '')} "
            f"risk_bots={payload.get('risk_bot_count', 0)} "
            f"hard_risk_bots={payload.get('hard_risk_bot_count', 0)}"
        )
    return 0 if payload.get("overall_status") != "blocked" else 2


if __name__ == "__main__":
    raise SystemExit(main())
