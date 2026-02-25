import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _event_day() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _role_map(registry: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in registry.get("sub_bots", []) if isinstance(registry, dict) else []:
        bot_id = str((row or {}).get("bot_id", "")).strip()
        role = str((row or {}).get("bot_role", "unknown")).strip() or "unknown"
        if bot_id:
            out[bot_id] = role
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Teacher-student distillation planner for new bots.")
    parser.add_argument("--walk-forward", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "walk_forward_latest.json"))
    parser.add_argument("--registry", default=str(PROJECT_ROOT / "master_bot_registry.json"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "governance" / "distillation" / "teacher_student_plan_latest.json"))
    parser.add_argument("--teacher-min-forward-mean", type=float, default=0.53)
    parser.add_argument("--teacher-min-runs", type=int, default=10)
    parser.add_argument("--teacher-max", type=int, default=12)
    parser.add_argument("--student-max-runs", type=int, default=6)
    parser.add_argument("--teachers-per-student", type=int, default=3)
    parser.add_argument("--teacher-weight", type=float, default=0.30, help="Soft-target blend weight for teacher signals.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    wf = _load_json(Path(args.walk_forward), default={})
    bots = (wf.get("bots") or {}) if isinstance(wf, dict) else {}
    registry = _load_json(Path(args.registry), default={})
    role_by_bot = _role_map(registry)

    teacher_candidates: list[dict[str, Any]] = []
    students: list[dict[str, Any]] = []

    for bot_id, row in bots.items():
        if not isinstance(row, dict):
            continue
        runs = _safe_int(row.get("runs"), 0)
        fwd = _safe_float(row.get("forward_mean"), 0.0)
        delta = _safe_float(row.get("delta"), 0.0)
        status = str(row.get("status", ""))
        role = role_by_bot.get(bot_id, "unknown")

        if runs >= max(args.teacher_min_runs, 1) and fwd >= args.teacher_min_forward_mean and delta >= -0.02 and status == "pass":
            teacher_candidates.append(
                {
                    "bot_id": bot_id,
                    "role": role,
                    "runs": runs,
                    "forward_mean": fwd,
                    "delta": delta,
                    "score": fwd + max(delta, -0.02),
                }
            )

        is_new = status == "insufficient_runs" or runs <= max(args.student_max_runs, 0)
        if is_new:
            students.append(
                {
                    "bot_id": bot_id,
                    "role": role,
                    "runs": runs,
                    "status": status,
                }
            )

    teacher_candidates.sort(key=lambda x: (x["score"], x["forward_mean"], x["runs"]), reverse=True)
    teachers = teacher_candidates[: max(args.teacher_max, 1)]

    assignments: list[dict[str, Any]] = []
    per_student_n = max(args.teachers_per_student, 1)
    for s in students:
        same_role = [t for t in teachers if t.get("role") == s.get("role")]
        pool = same_role if same_role else teachers
        selected = pool[:per_student_n]
        assignments.append(
            {
                "student_bot_id": s["bot_id"],
                "student_role": s["role"],
                "student_runs": s["runs"],
                "teacher_blend_weight": round(max(min(args.teacher_weight, 0.9), 0.0), 3),
                "teachers": [
                    {
                        "bot_id": t["bot_id"],
                        "role": t["role"],
                        "forward_mean": round(_safe_float(t["forward_mean"]), 6),
                        "delta": round(_safe_float(t["delta"]), 6),
                    }
                    for t in selected
                ],
            }
        )

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "walk_forward": str(Path(args.walk_forward)),
            "registry": str(Path(args.registry)),
            "teacher_min_forward_mean": args.teacher_min_forward_mean,
            "teacher_min_runs": args.teacher_min_runs,
            "teacher_max": args.teacher_max,
            "student_max_runs": args.student_max_runs,
            "teachers_per_student": args.teachers_per_student,
            "teacher_weight": args.teacher_weight,
        },
        "summary": {
            "teacher_count": len(teachers),
            "student_count": len(students),
            "assignment_count": len(assignments),
        },
        "teachers": teachers,
        "assignments": assignments,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    events = PROJECT_ROOT / "governance" / "distillation" / f"teacher_student_events_{_event_day()}.jsonl"
    events.parent.mkdir(parents=True, exist_ok=True)
    with events.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    env_hint = PROJECT_ROOT / "governance" / "distillation" / "teacher_student_env.sh"
    env_hint.write_text(
        "\n".join(
            [
                "# generated by distill_new_bots.py",
                f"export DISTILLATION_TEACHER_WEIGHT={round(max(min(args.teacher_weight, 0.9), 0.0), 3)}",
                f"export DISTILLATION_TEACHER_COUNT={len(teachers)}",
                f"export DISTILLATION_STUDENT_COUNT={len(students)}",
                f"export DISTILLATION_PLAN_PATH={out_path}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "distillation_plan_ok=True teachers={t} students={s} assignments={a} out={o}".format(
                t=len(teachers),
                s=len(students),
                a=len(assignments),
                o=out_path,
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
