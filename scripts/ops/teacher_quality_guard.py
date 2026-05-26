#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "distillation" / "teacher_quality_latest.json"
OVERFIT_BLOCKING_STATUSES = {"leak_like", "severe_overfit", "overfit_watch", "high_accuracy_guarded"}


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _registry_rows(project_root: Path) -> list[dict[str, Any]]:
    payload = load_json(project_root / "master_bot_registry.json")
    rows = payload.get("sub_bots") if isinstance(payload.get("sub_bots"), list) else []
    return [row for row in rows if isinstance(row, dict)]


def _registry_row_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        bot_id = str(row.get("bot_id") or "").strip().lower()
        if bot_id:
            out[bot_id] = row
    return out


def _paper_strategy_bot_id(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    if "::" in value:
        value = value.rsplit("::", 1)[-1].strip()
    return value.lower()


def _paper_bonus_map(paper_payload: dict[str, Any]) -> dict[str, float]:
    bonuses: dict[str, float] = {}
    sleeve_rows = paper_payload.get("sleeve_latest") if isinstance(paper_payload.get("sleeve_latest"), list) else []
    for sleeve in sleeve_rows:
        if not isinstance(sleeve, dict):
            continue
        for row in sleeve.get("top_winning_strategies") or []:
            if not isinstance(row, dict):
                continue
            bot_id = _paper_strategy_bot_id(str(row.get("strategy") or ""))
            if not bot_id:
                continue
            pnl = max(_safe_float(row.get("ending_net_pnl_total"), 0.0), 0.0)
            bonuses[bot_id] = bonuses.get(bot_id, 0.0) + min(pnl / 5000.0, 0.05)
    return bonuses


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _overfit_awareness(project_root: Path) -> dict[str, Any]:
    return load_json(project_root / "governance" / "health" / "overfitting_awareness_latest.json")


def _overfit_awareness_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    rows = payload.get("bot_risk") if isinstance(payload.get("bot_risk"), list) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        bot_id = str(row.get("bot_id") or "").strip().lower()
        if bot_id:
            out[bot_id] = row
    return out


def _teacher_overfit_blocked(row: dict[str, Any]) -> bool:
    if not row:
        return False
    status = str(row.get("status") or "").strip().lower()
    policy = _as_dict(row.get("policy"))
    return status in OVERFIT_BLOCKING_STATUSES and not bool(policy.get("may_teach", False))


def _teacher_overfit_fields(row: dict[str, Any]) -> dict[str, Any]:
    if not row:
        return {
            "overfit_status": "unknown",
            "overfit_risk_score": 0.0,
            "overfit_train_forward_gap": 0.0,
            "overfit_policy": {
                "may_teach": True,
                "may_promote": True,
                "requires_generalization_canary": False,
            },
        }
    policy = _as_dict(row.get("policy"))
    return {
        "overfit_status": str(row.get("status") or "unknown"),
        "overfit_risk_score": round(_safe_float(row.get("risk_score"), 0.0), 6),
        "overfit_train_forward_gap": round(_safe_float(row.get("train_forward_gap"), 0.0), 6),
        "overfit_policy": {
            "may_teach": bool(policy.get("may_teach", False)),
            "may_promote": bool(policy.get("may_promote", False)),
            "requires_generalization_canary": bool(policy.get("requires_generalization_canary", False)),
        },
    }


def _teacher_score(
    *,
    forward_mean: float,
    registry_accuracy: float,
    quality_score: float,
    runs: int,
    delta: float,
    active: bool,
    paper_bonus: float,
) -> float:
    run_norm = min(max(float(runs), 0.0) / 20.0, 1.0)
    delta_norm = min(max((float(delta) + 0.05) / 0.10, 0.0), 1.0)
    active_bonus = 0.03 if active else 0.0
    score = (
        (0.42 * max(forward_mean, 0.0))
        + (0.18 * max(registry_accuracy, 0.0))
        + (0.18 * max(quality_score, 0.0))
        + (0.10 * run_norm)
        + (0.07 * delta_norm)
        + active_bonus
        + min(max(paper_bonus, 0.0), 0.05)
    )
    return round(min(max(score, 0.0), 1.0), 6)


def _teacher_grade(score: float, *, forward_mean: float, quality_score: float, runs: int) -> str:
    if score >= 0.72 or (forward_mean >= 0.56 and quality_score >= 0.78 and runs >= 12):
        return "elite"
    if score >= 0.64:
        return "strong"
    return "qualified"


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    teacher_min_forward_mean: float = 0.53,
    teacher_min_runs: int = 8,
    teacher_min_delta: float = -0.03,
    teacher_min_registry_accuracy: float = 0.56,
    teacher_min_registry_quality: float = 0.65,
    teacher_min_score: float = 0.58,
    teacher_max: int = 16,
) -> dict[str, Any]:
    walk_forward = load_json(project_root / "governance" / "walk_forward" / "walk_forward_latest.json")
    training_quality = load_json(project_root / "governance" / "health" / "training_quality_control_latest.json")
    paper_performance = load_json(project_root / "governance" / "health" / "paper_performance_latest.json")
    current_plan = load_json(project_root / "governance" / "distillation" / "teacher_student_plan_latest.json")
    overfit_payload = _overfit_awareness(project_root)
    overfit_map = _overfit_awareness_map(overfit_payload)
    registry_rows = _registry_rows(project_root)
    registry_map = _registry_row_map(registry_rows)
    blocked_ids = set()
    targeted_actions = training_quality.get("targeted_actions") if isinstance(training_quality.get("targeted_actions"), dict) else {}
    for key in (
        "refresh_diagnostics_bot_ids",
        "repair_runtime_input_bot_ids",
        "runtime_input_depth_debt_bot_ids",
        "quality_probation_bot_ids",
        "targeted_retrain_bot_ids",
    ):
        for raw in targeted_actions.get(key) or []:
            bot_id = str(raw or "").strip().lower()
            if bot_id:
                blocked_ids.add(bot_id)
    paper_bonus = _paper_bonus_map(paper_performance)

    candidate_map: dict[str, dict[str, Any]] = {}
    rejected_rows: list[dict[str, Any]] = []
    rejected_reasons = Counter()
    overfit_rejected_statuses = Counter()
    overfit_rejected_ids: set[str] = set()

    def reject(bot_id: str, role: str, reason: str) -> None:
        text = str(reason or "").strip()
        if not text:
            return
        rejected_reasons[text] += 1
        if bot_id:
            rejected_rows.append({"bot_id": bot_id, "bot_role": role, "reason": text})

    def maybe_store(candidate: dict[str, Any]) -> None:
        bot_id = str(candidate.get("bot_id") or "").strip().lower()
        if not bot_id:
            return
        prev = candidate_map.get(bot_id)
        if prev is None or float(candidate.get("teacher_score", 0.0) or 0.0) > float(prev.get("teacher_score", 0.0) or 0.0):
            candidate_map[bot_id] = candidate

    wf_bots = walk_forward.get("bots") if isinstance(walk_forward.get("bots"), dict) else {}
    for raw_bot_id, raw in wf_bots.items():
        if not isinstance(raw, dict):
            continue
        bot_id = str(raw_bot_id or "").strip().lower()
        reg_row = registry_map.get(bot_id, {})
        role = str(reg_row.get("bot_role") or "unknown").strip() or "unknown"
        status = str(raw.get("status") or "").strip().lower()
        runs = _safe_int(raw.get("runs"), 0)
        forward_mean = _safe_float(raw.get("forward_mean"), 0.0)
        delta = _safe_float(raw.get("delta"), 0.0)
        registry_accuracy = max(
            _safe_float(reg_row.get("test_accuracy"), 0.0),
            _safe_float(reg_row.get("candidate_test_accuracy"), 0.0),
            forward_mean,
        )
        quality_score = max(
            _safe_float(reg_row.get("quality_score"), 0.0),
            _safe_float(reg_row.get("candidate_quality_score"), 0.0),
            _safe_float(raw.get("trading_quality_score"), 0.0),
        )
        if bot_id in blocked_ids:
            reject(bot_id, role, "quality_guard_blocked")
            continue
        overfit_row = overfit_map.get(bot_id, {})
        if _teacher_overfit_blocked(overfit_row):
            if bot_id not in overfit_rejected_ids:
                overfit_rejected_ids.add(bot_id)
                overfit_rejected_statuses[str(overfit_row.get("status") or "unknown")] += 1
            reject(bot_id, role, "overfit_risk_blocked")
            continue
        if status != "pass":
            reject(bot_id, role, f"walk_forward_status:{status or 'unknown'}")
            continue
        if runs < max(int(teacher_min_runs), 1):
            reject(bot_id, role, "insufficient_walk_forward_runs")
            continue
        if forward_mean < float(teacher_min_forward_mean):
            reject(bot_id, role, "walk_forward_mean_below_teacher_floor")
            continue
        if delta < float(teacher_min_delta):
            reject(bot_id, role, "walk_forward_delta_too_negative")
            continue
        active = bool(reg_row.get("active", False))
        score = _teacher_score(
            forward_mean=forward_mean,
            registry_accuracy=registry_accuracy,
            quality_score=quality_score,
            runs=runs,
            delta=delta,
            active=active,
            paper_bonus=paper_bonus.get(bot_id, 0.0),
        )
        if score < float(teacher_min_score):
            reject(bot_id, role, "teacher_score_below_floor")
            continue
        maybe_store(
            {
                "bot_id": bot_id,
                "bot_role": role,
                "teacher_score": score,
                "teacher_grade": _teacher_grade(score, forward_mean=forward_mean, quality_score=quality_score, runs=runs),
                "source": "walk_forward",
                "sources": ["walk_forward"],
                "walk_forward_runs": runs,
                "walk_forward_forward_mean": round(forward_mean, 6),
                "walk_forward_delta": round(delta, 6),
                "registry_accuracy": round(registry_accuracy, 6),
                "quality_score": round(quality_score, 6),
                "active": active,
                "lifecycle_state": str(reg_row.get("lifecycle_state") or ""),
                "paper_bonus": round(paper_bonus.get(bot_id, 0.0), 6),
                **_teacher_overfit_fields(overfit_row),
            }
        )

    for reg_row in registry_rows:
        bot_id = str(reg_row.get("bot_id") or "").strip().lower()
        if not bot_id:
            continue
        role = str(reg_row.get("bot_role") or "unknown").strip() or "unknown"
        lifecycle_state = str(reg_row.get("lifecycle_state") or "").strip().lower()
        active = bool(reg_row.get("active", False))
        if bot_id in blocked_ids:
            reject(bot_id, role, "quality_guard_blocked")
            continue
        overfit_row = overfit_map.get(bot_id, {})
        if _teacher_overfit_blocked(overfit_row):
            if bot_id not in overfit_rejected_ids:
                overfit_rejected_ids.add(bot_id)
                overfit_rejected_statuses[str(overfit_row.get("status") or "unknown")] += 1
            reject(bot_id, role, "overfit_risk_blocked")
            continue
        if bool(reg_row.get("deleted_from_rotation", False)) or lifecycle_state in {"probation", "retired", "deleted", "deactivated"}:
            reject(bot_id, role, "inactive_for_teacher_duty")
            continue
        registry_accuracy = max(
            _safe_float(reg_row.get("test_accuracy"), 0.0),
            _safe_float(reg_row.get("candidate_test_accuracy"), 0.0),
        )
        quality_score = max(
            _safe_float(reg_row.get("quality_score"), 0.0),
            _safe_float(reg_row.get("candidate_quality_score"), 0.0),
        )
        if registry_accuracy < float(teacher_min_registry_accuracy):
            continue
        if quality_score < float(teacher_min_registry_quality):
            continue
        wf_row = wf_bots.get(bot_id) if isinstance(wf_bots.get(bot_id), dict) else {}
        runs = max(_safe_int(wf_row.get("runs"), 0), max(int(teacher_min_runs), 1))
        forward_mean = max(_safe_float(wf_row.get("forward_mean"), 0.0), registry_accuracy)
        delta = _safe_float(wf_row.get("delta"), 0.0)
        score = _teacher_score(
            forward_mean=forward_mean,
            registry_accuracy=registry_accuracy,
            quality_score=quality_score,
            runs=runs,
            delta=delta,
            active=active,
            paper_bonus=paper_bonus.get(bot_id, 0.0),
        )
        if score < float(teacher_min_score):
            continue
        source_label = "registry_active" if active else "registry_candidate"
        maybe_store(
            {
                "bot_id": bot_id,
                "bot_role": role,
                "teacher_score": score,
                "teacher_grade": _teacher_grade(score, forward_mean=forward_mean, quality_score=quality_score, runs=runs),
                "source": source_label,
                "sources": [source_label],
                "walk_forward_runs": _safe_int(wf_row.get("runs"), 0),
                "walk_forward_forward_mean": round(_safe_float(wf_row.get("forward_mean"), 0.0), 6),
                "walk_forward_delta": round(_safe_float(wf_row.get("delta"), 0.0), 6),
                "registry_accuracy": round(registry_accuracy, 6),
                "quality_score": round(quality_score, 6),
                "active": active,
                "lifecycle_state": lifecycle_state,
                "paper_bonus": round(paper_bonus.get(bot_id, 0.0), 6),
                **_teacher_overfit_fields(overfit_row),
            }
        )

    teachers = sorted(
        candidate_map.values(),
        key=lambda row: (
            float(row.get("teacher_score", 0.0) or 0.0),
            float(row.get("walk_forward_forward_mean", 0.0) or 0.0),
            float(row.get("quality_score", 0.0) or 0.0),
            str(row.get("bot_id") or ""),
        ),
        reverse=True,
    )[: max(int(teacher_max), 1)]

    role_map: dict[str, list[dict[str, Any]]] = {}
    for row in teachers:
        role_map.setdefault(str(row.get("bot_role") or "unknown"), []).append(row)

    student_roles = Counter()
    for row in current_plan.get("assignments") or []:
        if not isinstance(row, dict):
            continue
        role = str(row.get("student_role") or "unknown").strip() or "unknown"
        student_roles[role] += 1
    uncovered_roles = [role for role in student_roles if role not in role_map]

    elite_count = sum(1 for row in teachers if str(row.get("teacher_grade") or "") == "elite")
    strong_count = sum(1 for row in teachers if str(row.get("teacher_grade") or "") == "strong")
    overall_status = "ready"
    if not teachers:
        overall_status = "blocked"
    elif elite_count == 0 or uncovered_roles:
        overall_status = "degraded"

    recommended_actions = ordered_unique(
        [
            "refresh walk-forward coverage and requalification so the teacher pool includes current strong performers" if not teachers else "",
            "promote at least one elite teacher-quality bot so student distillation is anchored to proven behavior" if elite_count == 0 else "",
            "seed teachers for uncovered student roles before expanding the student roster" if uncovered_roles else "",
            "keep bots that are on quality probation or runtime-input repair out of the teacher pool until they recover" if blocked_ids else "",
            "keep overfit-risk or high-accuracy-guarded bots out of teacher duty until generalization canaries clear" if overfit_rejected_statuses else "",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "inputs": {
            "teacher_min_forward_mean": float(teacher_min_forward_mean),
            "teacher_min_runs": int(teacher_min_runs),
            "teacher_min_delta": float(teacher_min_delta),
            "teacher_min_registry_accuracy": float(teacher_min_registry_accuracy),
            "teacher_min_registry_quality": float(teacher_min_registry_quality),
            "teacher_min_score": float(teacher_min_score),
            "teacher_max": int(teacher_max),
        },
        "summary": {
            "qualified_teacher_count": len(teachers),
            "elite_teacher_count": elite_count,
            "strong_teacher_count": strong_count,
            "student_role_count": len(student_roles),
            "uncovered_student_role_count": len(uncovered_roles),
            "excluded_bot_count": len(rejected_rows),
            "overfit_blocked_teacher_count": len(overfit_rejected_ids),
        },
        "overfitting_awareness": {
            "overall_status": str(overfit_payload.get("overall_status") if overfit_payload else "missing") or "ready",
            "risk_bot_count": _safe_int(overfit_payload.get("risk_bot_count"), 0),
            "hard_risk_bot_count": _safe_int(overfit_payload.get("hard_risk_bot_count"), 0),
            "blocked_teacher_count": len(overfit_rejected_ids),
            "blocked_status_counts": dict(overfit_rejected_statuses),
            "policy": "overfit-risk, leak-like, severe-overfit, and high-accuracy-guarded bots cannot teach students",
        },
        "qualified_teachers": teachers,
        "role_coverage": [
            {
                "bot_role": role,
                "teacher_count": len(rows),
                "top_teacher_ids": [str(row.get("bot_id") or "") for row in rows[:3]],
            }
            for role, rows in sorted(role_map.items(), key=lambda item: (-len(item[1]), item[0]))
        ],
        "student_role_coverage": {
            "role_counts": dict(student_roles),
            "uncovered_roles": uncovered_roles,
        },
        "excluded_reasons": [
            {"reason": reason, "count": count}
            for reason, count in rejected_reasons.most_common(10)
        ],
        "excluded_bots": rejected_rows[:20],
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Curate high-quality teacher bots from current walk-forward and registry performance.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--teacher-min-forward-mean", type=float, default=0.53)
    parser.add_argument("--teacher-min-runs", type=int, default=8)
    parser.add_argument("--teacher-min-delta", type=float, default=-0.03)
    parser.add_argument("--teacher-min-registry-accuracy", type=float, default=0.56)
    parser.add_argument("--teacher-min-registry-quality", type=float, default=0.65)
    parser.add_argument("--teacher-min-score", type=float, default=0.58)
    parser.add_argument("--teacher-max", type=int, default=16)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        Path(args.project_root).resolve(),
        teacher_min_forward_mean=float(args.teacher_min_forward_mean),
        teacher_min_runs=int(args.teacher_min_runs),
        teacher_min_delta=float(args.teacher_min_delta),
        teacher_min_registry_accuracy=float(args.teacher_min_registry_accuracy),
        teacher_min_registry_quality=float(args.teacher_min_registry_quality),
        teacher_min_score=float(args.teacher_min_score),
        teacher_max=int(args.teacher_max),
    )
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "teacher_quality_guard "
            f"overall_status={payload.get('overall_status', '')} "
            f"qualified_teachers={int(((payload.get('summary') or {}).get('qualified_teacher_count', 0) or 0))}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
