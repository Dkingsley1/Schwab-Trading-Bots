#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "supportability_control_latest.json"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


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


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in items:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def build_payload(project_root: Path = PROJECT_ROOT, *, limit: int = 8) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    lifecycle_root = project_root / "governance" / "lifecycle"
    distillation_root = project_root / "governance" / "distillation"

    lifecycle = _load_json(lifecycle_root / "model_lifecycle_latest.json")
    distillation = _load_json(distillation_root / "teacher_student_plan_latest.json")
    teacher_quality = _load_json(distillation_root / "teacher_quality_latest.json")
    training_quality = _load_json(health_root / "training_quality_control_latest.json")
    requalification = _load_json(health_root / "training_requalification_latest.json")

    supportability = training_quality.get("supportability") if isinstance(training_quality.get("supportability"), dict) else {}
    repair = lifecycle.get("repair") if isinstance(lifecycle.get("repair"), dict) else {}
    assignments = [row for row in (distillation.get("assignments") or []) if isinstance(row, dict)]
    teachers = [row for row in (distillation.get("teachers") or []) if isinstance(row, dict)]
    students_without_teachers = [row for row in assignments if not list(row.get("teachers") or [])]
    teacher_quality_summary = teacher_quality.get("summary") if isinstance(teacher_quality.get("summary"), dict) else {}

    teacher_gaps = Counter()
    for row in students_without_teachers:
        role = str(row.get("student_role") or "unknown").strip() or "unknown"
        teacher_gaps[role] += 1

    overall_status = "ready"
    active_supportability_score = _safe_float(supportability.get("active_supportability_score"), 0.0)
    if active_supportability_score < 0.5 or students_without_teachers:
        overall_status = "needs_work"
    if active_supportability_score <= 0.0 and not teachers and students_without_teachers:
        overall_status = "blocked"

    recommended_actions: list[str] = []
    if active_supportability_score < 0.5:
        recommended_actions.append("expand the supportable active roster before promotion or retrain work depends on a single fragile champion")
    if students_without_teachers:
        recommended_actions.append("assign qualified teachers for student bots or relax teacher thresholds so distillation can actually contribute signal")
    if _safe_int(teacher_quality_summary.get("elite_teacher_count"), 0) <= 0:
        recommended_actions.append("upgrade the teacher pool so at least one elite, high-performing bot can mentor students")
    if _safe_int(lifecycle.get("stale_active_training_diagnostics"), 0) > 0:
        recommended_actions.append("refresh stale training diagnostics before deciding whether a bot should remain active or move to probation")
    if _safe_int(requalification.get("reactivation_ready_count"), 0) <= 0:
        recommended_actions.append("repair runtime inputs for the best inactive candidates so at least a few bots become reactivation-ready")
    if _safe_int(lifecycle.get("missing_active_artifacts_total"), 0) > 0:
        recommended_actions.append("repair missing model or log artifacts before relying on lifecycle status in the active roster")

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "overall_status": overall_status,
        "supportability": {
            "active_bots": _safe_int(supportability.get("active_bots"), 0),
            "active_supportable_bots": _safe_int(supportability.get("active_supportable_bots"), 0),
            "active_supportability_score": round(active_supportability_score, 6),
            "tier_counts": dict(supportability.get("tier_counts") or {}),
        },
        "lifecycle": {
            "missing_active_artifacts_total": _safe_int(lifecycle.get("missing_active_artifacts_total"), 0),
            "missing_log_only_artifacts": _safe_int(lifecycle.get("missing_log_only_artifacts"), 0),
            "stale_active_training_diagnostics": _safe_int(lifecycle.get("stale_active_training_diagnostics"), 0),
            "repair_enabled": bool(repair.get("enabled", False)),
            "registry_updated": bool(repair.get("registry_updated", False)),
        },
        "teacher_student": {
            "teacher_count": len(teachers),
            "student_count": _safe_int((distillation.get("summary") or {}).get("student_count"), len(assignments)),
            "assignment_count": _safe_int((distillation.get("summary") or {}).get("assignment_count"), len(assignments)),
            "students_without_teachers": len(students_without_teachers),
            "teacher_gap_by_role": [
                {"student_role": role, "missing_assignments": count}
                for role, count in teacher_gaps.most_common(max(int(limit), 1))
            ],
            "uncovered_students": [
                {
                    "student_bot_id": str(row.get("student_bot_id") or ""),
                    "student_role": str(row.get("student_role") or ""),
                    "student_runs": _safe_int(row.get("student_runs"), 0),
                }
                for row in students_without_teachers[: max(int(limit), 1)]
            ],
        },
        "teacher_quality": {
            "overall_status": str(teacher_quality.get("overall_status") or ""),
            "qualified_teacher_count": _safe_int(teacher_quality_summary.get("qualified_teacher_count"), 0),
            "elite_teacher_count": _safe_int(teacher_quality_summary.get("elite_teacher_count"), 0),
            "strong_teacher_count": _safe_int(teacher_quality_summary.get("strong_teacher_count"), 0),
            "uncovered_student_role_count": _safe_int(teacher_quality_summary.get("uncovered_student_role_count"), 0),
        },
        "reactivation_lane": {
            "candidate_count": _safe_int(requalification.get("candidate_count"), 0),
            "reactivation_ready_count": _safe_int(requalification.get("reactivation_ready_count"), 0),
            "top_candidates": list(requalification.get("top_candidates") or [])[: max(int(limit), 1)],
        },
        "recommended_actions": _ordered_unique(recommended_actions),
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Combine lifecycle hygiene, teacher-student coverage, and reactivation readiness into a single supportability surface.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve(), limit=int(args.limit))
    out_path = Path(args.out_file).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "supportability_control "
            f"overall_status={payload.get('overall_status', '')} "
            f"students_without_teachers={int(((payload.get('teacher_student') or {}).get('students_without_teachers', 0) or 0))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
