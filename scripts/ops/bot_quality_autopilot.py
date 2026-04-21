#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "bot_quality_autopilot_latest.json"
PYTHON_BIN = Path(sys.executable)


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


def _load_registry_roles(project_root: Path) -> dict[str, str]:
    payload = load_json(project_root / "master_bot_registry.json")
    rows = payload.get("sub_bots") if isinstance(payload.get("sub_bots"), list) else []
    out: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        bot_id = str(row.get("bot_id") or "").strip().lower()
        if bot_id:
            out[bot_id] = str(row.get("bot_role") or "unknown").strip() or "unknown"
    return out


def _run_json(cmd: list[str], *, cwd: Path, timeout_sec: int) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=max(int(timeout_sec), 1),
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        rc = int(proc.returncode)
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        rc = 124
        timed_out = True
    payload = {}
    for raw in reversed([line.strip() for line in stdout.splitlines() if line.strip()]):
        try:
            parsed = json.loads(raw)
        except Exception:
            continue
        if isinstance(parsed, dict):
            payload = parsed
            break
    return {
        "cmd": list(cmd),
        "rc": rc,
        "timed_out": timed_out,
        "stdout_tail": "\n".join(stdout.splitlines()[-10:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-10:]),
        "payload": payload,
    }


def _mentor_ids(teacher_quality: dict[str, Any], *, bot_role: str, limit: int = 3) -> list[str]:
    rows = teacher_quality.get("qualified_teachers") if isinstance(teacher_quality.get("qualified_teachers"), list) else []
    selected = [
        str(row.get("bot_id") or "")
        for row in rows
        if isinstance(row, dict) and str(row.get("bot_role") or "") == bot_role and str(row.get("bot_id") or "").strip()
    ]
    if not selected:
        selected = [
            str(row.get("bot_id") or "")
            for row in rows
            if isinstance(row, dict) and str(row.get("bot_id") or "").strip()
        ]
    return selected[: max(int(limit), 1)]


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool = False,
    timeout_sec: int = 900,
    mentor_limit: int = 3,
) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    training_quality = load_json(health_root / "training_quality_control_latest.json")
    supportability = load_json(health_root / "supportability_control_latest.json")
    teacher_quality = load_json(project_root / "governance" / "distillation" / "teacher_quality_latest.json")
    requalification = load_json(health_root / "training_requalification_latest.json")
    coverage_seed = load_json(project_root / "governance" / "walk_forward" / "coverage_seed_latest.json")
    runtime_control = load_json(health_root / "training_runtime_control_latest.json")
    role_by_bot = _load_registry_roles(project_root)
    coverage_seed_rows = coverage_seed.get("seed_queue") if isinstance(coverage_seed.get("seed_queue"), list) else []

    targeted = training_quality.get("targeted_actions") if isinstance(training_quality.get("targeted_actions"), dict) else {}
    refresh_ids = [str(raw or "").strip().lower() for raw in targeted.get("refresh_diagnostics_bot_ids") or [] if str(raw or "").strip()]
    repair_ids = [str(raw or "").strip().lower() for raw in targeted.get("repair_runtime_input_bot_ids") or [] if str(raw or "").strip()]
    probation_ids = [str(raw or "").strip().lower() for raw in targeted.get("quality_probation_bot_ids") or [] if str(raw or "").strip()]
    retrain_ids = [str(raw or "").strip().lower() for raw in targeted.get("targeted_retrain_bot_ids") or [] if str(raw or "").strip()]
    teacher_student = supportability.get("teacher_student") if isinstance(supportability.get("teacher_student"), dict) else {}
    uncovered_students = teacher_student.get("uncovered_students") if isinstance(teacher_student.get("uncovered_students"), list) else []
    uncovered_students = [row for row in uncovered_students if isinstance(row, dict)]

    queue_map: dict[str, dict[str, Any]] = {}

    def add_queue(bot_id: str, *, reason: str, priority: float, next_step: str) -> None:
        text = str(bot_id or "").strip().lower()
        if not text:
            return
        row = queue_map.setdefault(
            text,
            {
                "bot_id": text,
                "bot_role": role_by_bot.get(text, "unknown"),
                "priority": float(priority),
                "next_step": next_step,
                "reasons": [],
            },
        )
        row["priority"] = max(float(row.get("priority", 0.0) or 0.0), float(priority))
        if next_step == "targeted_retrain" or str(row.get("next_step") or "") == "assign_teacher":
            row["next_step"] = next_step
        if reason not in row["reasons"]:
            row["reasons"].append(reason)

    for bot_id in refresh_ids:
        add_queue(bot_id, reason="stale_diagnostics", priority=100.0, next_step="refresh_diagnostics")
    for bot_id in repair_ids:
        add_queue(bot_id, reason="runtime_input_gap", priority=98.0, next_step="repair_runtime_inputs")
    for bot_id in probation_ids:
        add_queue(bot_id, reason="quality_probation", priority=95.0, next_step="targeted_retrain")
    for bot_id in retrain_ids:
        add_queue(bot_id, reason="targeted_retrain_shortlist", priority=92.0, next_step="targeted_retrain")

    for row in requalification.get("top_candidates") or []:
        if not isinstance(row, dict):
            continue
        bot_id = str(row.get("bot_id") or "").strip().lower()
        actions = [str(raw or "").strip() for raw in row.get("actions") or [] if str(raw or "").strip()]
        if "seed_walk_forward_coverage" in actions:
            add_queue(bot_id, reason="reactivation_ready", priority=_safe_float(row.get("priority"), 70.0), next_step="seed_walk_forward_coverage")
        elif "targeted_retrain" in actions:
            add_queue(bot_id, reason="requalification_targeted_retrain", priority=_safe_float(row.get("priority"), 68.0), next_step="targeted_retrain")
        elif "repair_runtime_inputs" in actions:
            add_queue(bot_id, reason="requalification_runtime_input_gap", priority=_safe_float(row.get("priority"), 66.0), next_step="repair_runtime_inputs")

    for row in uncovered_students:
        bot_id = str(row.get("student_bot_id") or "").strip().lower()
        add_queue(bot_id, reason="student_without_teacher", priority=88.0, next_step="assign_teacher")
        queue_map[bot_id]["bot_role"] = str(row.get("student_role") or queue_map[bot_id].get("bot_role") or "unknown").strip() or "unknown"

    infrastructure_helper_count = 0
    for row in coverage_seed_rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("bot_role") or "").strip() != "infrastructure_sub_bot":
            continue
        bot_id = str(row.get("bot_id") or "").strip().lower()
        if not bot_id:
            continue
        next_step = "repair_runtime_inputs" if bool(row.get("needs_runtime_input_repair", False)) else "seed_walk_forward_coverage"
        add_queue(
            bot_id,
            reason="infrastructure_helper_lane",
            priority=min(_safe_float(row.get("priority"), 72.0), 90.0),
            next_step=next_step,
        )
        queue_map[bot_id]["queue_bucket"] = "infrastructure_support"
        infrastructure_helper_count += 1

    quality_queue = sorted(
        queue_map.values(),
        key=lambda row: (-float(row.get("priority", 0.0) or 0.0), str(row.get("bot_id") or "")),
    )
    for row in quality_queue:
        row["recommended_teacher_bot_ids"] = _mentor_ids(
            teacher_quality,
            bot_role=str(row.get("bot_role") or "unknown"),
            limit=mentor_limit,
        )

    assignment_preview = [
        {
            "student_bot_id": str(row.get("student_bot_id") or "").strip().lower(),
            "student_role": str(row.get("student_role") or "unknown").strip() or "unknown",
            "suggested_teacher_bot_ids": _mentor_ids(
                teacher_quality,
                bot_role=str(row.get("student_role") or "unknown"),
                limit=mentor_limit,
            ),
        }
        for row in uncovered_students[:8]
    ]

    teacher_summary = teacher_quality.get("summary") if isinstance(teacher_quality.get("summary"), dict) else {}
    qualified_teachers = _safe_int(teacher_summary.get("qualified_teacher_count"), 0)
    elite_teachers = _safe_int(teacher_summary.get("elite_teacher_count"), 0)
    students_without_teachers = _safe_int(teacher_student.get("students_without_teachers"), len(uncovered_students))
    training_status = str(training_quality.get("overall_status") or "")
    snapshot_ready = bool(runtime_control.get("snapshot_ready", False))
    coverage_shortfall_bots = _safe_int(coverage_seed.get("coverage_shortfall_bots"), 0)

    attempts: list[dict[str, Any]] = []
    if apply:
        apply_steps: list[list[str]] = []
        if repair_ids or not snapshot_ready:
            apply_steps.append([str(PYTHON_BIN), str(project_root / "scripts" / "build_runtime_training_snapshot.py"), "--json"])
            apply_steps.append([str(PYTHON_BIN), str(project_root / "scripts" / "ops" / "training_runtime_control.py"), "--json"])
        apply_steps.extend(
            [
                [str(PYTHON_BIN), str(project_root / "scripts" / "ops" / "teacher_quality_guard.py"), "--json"],
                [str(PYTHON_BIN), str(project_root / "scripts" / "distill_new_bots.py"), "--json"],
                [str(PYTHON_BIN), str(project_root / "scripts" / "ops" / "training_requalification_lane.py"), "--write-queue", "--json"],
                [str(PYTHON_BIN), str(project_root / "scripts" / "ops" / "walk_forward_coverage_seed.py"), "--write-queue", "--json"],
                [str(PYTHON_BIN), str(project_root / "scripts" / "ops" / "supportability_control.py"), "--json"],
                [str(PYTHON_BIN), str(project_root / "scripts" / "ops" / "training_quality_control.py"), "--json"],
            ]
        )
        for cmd in apply_steps:
            attempts.append(_run_json(cmd, cwd=project_root, timeout_sec=timeout_sec))
        teacher_quality = load_json(project_root / "governance" / "distillation" / "teacher_quality_latest.json")
        supportability = load_json(health_root / "supportability_control_latest.json")
        training_quality = load_json(health_root / "training_quality_control_latest.json")
        teacher_student = supportability.get("teacher_student") if isinstance(supportability.get("teacher_student"), dict) else {}
        teacher_summary = teacher_quality.get("summary") if isinstance(teacher_quality.get("summary"), dict) else {}
        qualified_teachers = _safe_int(teacher_summary.get("qualified_teacher_count"), 0)
        elite_teachers = _safe_int(teacher_summary.get("elite_teacher_count"), 0)
        students_without_teachers = _safe_int(teacher_student.get("students_without_teachers"), students_without_teachers)
        training_status = str(training_quality.get("overall_status") or training_status)

    overall_status = "ready"
    if training_status == "blocked" or qualified_teachers <= 0 or students_without_teachers > 0:
        overall_status = "blocked"
    elif quality_queue or elite_teachers <= 0 or coverage_shortfall_bots > 0:
        overall_status = "degraded"

    if apply and attempts:
        if any(int(row.get("rc", 1)) != 0 for row in attempts):
            overall_status = "blocked"
        elif overall_status != "ready":
            overall_status = "degraded"

    recommended_actions = ordered_unique(
        [
            "run the bot-quality autopilot in apply mode on a timer so teacher curation and requalification queues stay fresh",
            "repair runtime inputs before targeted retrains so sample-starved bots are not repeatedly retrained on broken inputs" if repair_ids else "",
            "assign high-quality teachers to uncovered students before expanding distillation breadth" if students_without_teachers > 0 else "",
            "seed walk-forward coverage continuously so teacher selection reflects current regime winners" if coverage_shortfall_bots > 0 else "",
            "keep infrastructure helper bots in their own retrain lane so control-plane recovery does not crowd out signal promotion" if infrastructure_helper_count > 0 else "",
            "promote or reactivate at least one elite teacher bot so mentorship is anchored to proven performers" if elite_teachers <= 0 else "",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "apply": bool(apply),
        "quality_blockers": {
            "refresh_diagnostics_bot_ids": refresh_ids,
            "repair_runtime_input_bot_ids": repair_ids,
            "quality_probation_bot_ids": probation_ids,
            "targeted_retrain_bot_ids": retrain_ids,
            "students_without_teachers": students_without_teachers,
            "coverage_shortfall_bots": coverage_shortfall_bots,
            "infrastructure_helper_count": infrastructure_helper_count,
        },
        "teacher_summary": {
            "qualified_teacher_count": qualified_teachers,
            "elite_teacher_count": elite_teachers,
            "teacher_quality_status": str(teacher_quality.get("overall_status") or ""),
        },
        "quality_upgrade_queue": quality_queue[:20],
        "infrastructure_helper_queue": [
            row for row in quality_queue
            if str(row.get("queue_bucket") or "") == "infrastructure_support"
        ][:8],
        "assignment_preview": assignment_preview,
        "attempts": [
            {
                "cmd": list(row.get("cmd") or []),
                "rc": int(row.get("rc", 1)),
                "timed_out": bool(row.get("timed_out", False)),
            }
            for row in attempts
        ],
        "infra_bots": [
            "bot_quality_autopilot",
            "teacher_quality_guard",
            "distill_new_bots",
            "training_requalification_lane",
            "walk_forward_coverage_seed",
            "retrain_lane_scheduler",
            "supportability_control",
            "training_quality_control",
        ],
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Automate bot-quality upkeep, teacher curation, and requalification queue refreshes.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--mentor-limit", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        Path(args.project_root).resolve(),
        apply=bool(args.apply),
        timeout_sec=int(args.timeout_sec),
        mentor_limit=int(args.mentor_limit),
    )
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "bot_quality_autopilot "
            f"overall_status={payload.get('overall_status', '')} "
            f"quality_queue={len(payload.get('quality_upgrade_queue') or [])}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
