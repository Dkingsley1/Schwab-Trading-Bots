#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops import section_grade_guard
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, ordered_unique, run_bounded_process_group, write_payload
else:
    from . import section_grade_guard
    from .long_runtime_common import PROJECT_ROOT, iso_now, ordered_unique, run_bounded_process_group, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "section_grade_autopilot_latest.json"
Runner = Callable[[list[str], Path, int], dict[str, Any]]
GuardBuilder = Callable[[Path], dict[str, Any]]


def _parse_json_output(text: str) -> dict[str, Any]:
    for raw in reversed([line.strip() for line in str(text or "").splitlines() if line.strip()]):
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _safe_int(raw: Any, default: int) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _bounded_timeout(raw: Any, *, max_step_timeout_sec: int) -> int:
    timeout = _safe_int(raw, max_step_timeout_sec)
    return max(1, min(timeout, max(int(max_step_timeout_sec), 1)))


def _bounded_repair_command(raw_cmd: list[str]) -> list[str]:
    cmd = [str(part) for part in raw_cmd]
    if "storage-backpressure-autopilot" not in cmd or "--apply" not in cmd or "--quick-bounded" in cmd:
        return cmd
    insert_at = cmd.index("--json") if "--json" in cmd else len(cmd)
    cmd.insert(insert_at, "--quick-bounded")
    return cmd


def _run(cmd: list[str], project_root: Path, timeout_sec: int) -> dict[str, Any]:
    result = run_bounded_process_group(cmd, cwd=project_root, timeout_seconds=max(int(timeout_sec), 1))
    stdout = str(result.get("stdout") or "")
    stderr = str(result.get("stderr") or "")
    return {
        "cmd": list(cmd),
        "rc": int(result.get("rc", 1)),
        "payload": _parse_json_output(stdout),
        "stdout_tail": "\n".join(stdout.splitlines()[-12:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-12:]) or ("timeout" if result.get("timed_out") else ""),
        "timeout_cleanup": result.get("timeout_cleanup") if isinstance(result.get("timeout_cleanup"), dict) else {},
    }


def _repair_plan(guard_payload: dict[str, Any]) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    seen: set[str] = set()
    below_floor_sections = {
        str(item).strip()
        for item in (guard_payload.get("below_floor_sections") or [])
        if str(item).strip()
    }
    advisory_below_floor_sections = {
        str(item).strip()
        for item in (guard_payload.get("advisory_below_floor_sections") or [])
        if str(item).strip()
    }
    blocking_below_floor_sections = {
        str(item).strip()
        for item in (guard_payload.get("blocking_below_floor_sections") or [])
        if str(item).strip()
    }
    if not blocking_below_floor_sections and not advisory_below_floor_sections:
        blocking_below_floor_sections = set(below_floor_sections)

    def add(section: str, reason: str, cmd: list[str], timeout_sec: int) -> None:
        key = " ".join(cmd)
        if key in seen:
            return
        seen.add(key)
        plan.append(
            {
                "section": section,
                "reason": reason,
                "cmd": list(cmd),
                "timeout_sec": int(timeout_sec),
            }
        )

    if blocking_below_floor_sections or (guard_payload.get("protected_by_floor_count", 0) or 0) > 0:
        add("floor_baseline", "refresh_grade_regression_repairs", ["./scripts/ops/opsctl.sh", "grade-regression-autopilot", "--apply", "--json"], 1800)
        add("floor_baseline", "refresh_artifacts", ["./scripts/ops/opsctl.sh", "dashboard-refresh"], 900)

    for row in guard_payload.get("sections") or []:
        if not isinstance(row, dict):
            continue
        state = str(row.get("state") or "")
        if state == "at_floor":
            continue
        section = str(row.get("section") or "")
        if state == "below_floor" and section not in blocking_below_floor_sections:
            continue
        reason = state or "below_floor"
        for cmd in row.get("recommended_commands") or []:
            if isinstance(cmd, list) and cmd:
                add(section, reason, _bounded_repair_command(cmd), 1200)
    return plan


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool = False,
    max_step_timeout_sec: int = 300,
    runner: Runner | None = None,
    guard_builder: GuardBuilder | None = None,
) -> dict[str, Any]:
    run_step = runner or _run
    build_guard = guard_builder or section_grade_guard.build_payload

    initial_guard = build_guard(project_root)
    repair_plan = _repair_plan(initial_guard)
    attempts: list[dict[str, Any]] = []
    if apply:
        for step in repair_plan:
            timeout_sec = _bounded_timeout(step.get("timeout_sec", 1200), max_step_timeout_sec=max_step_timeout_sec)
            result = run_step(list(step["cmd"]), project_root, timeout_sec)
            payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
            attempts.append(
                {
                    "section": step["section"],
                    "reason": step["reason"],
                    "cmd": list(result.get("cmd") or []),
                    "rc": int(result.get("rc", 1)),
                    "timeout_sec": timeout_sec,
                    "payload_summary": {
                        key: payload.get(key)
                        for key in (
                            "overall_status",
                            "overall_letter_grade",
                            "below_floor_count",
                            "protected_by_floor_count",
                            "readiness_score",
                            "training_quality_score",
                            "autonomy_score",
                        )
                        if key in payload
                    },
                    "stdout_tail": str(result.get("stdout_tail") or ""),
                    "stderr_tail": str(result.get("stderr_tail") or ""),
                }
            )

    final_guard = build_guard(project_root)
    final_blocking_below_floor_count = int(
        final_guard.get("blocking_below_floor_count", final_guard.get("below_floor_count", 0)) or 0
    )
    final_advisory_below_floor_count = int(final_guard.get("advisory_below_floor_count", 0) or 0)
    recommended_actions = ordered_unique(
        [
            "leave the section-grade floor bot in apply mode so bounded recovery keeps A-/A sections from slipping back"
            if apply and repair_plan
            else "",
            "focus on the sections below floor first; floor-protected sections are still meeting the target contract"
            if final_blocking_below_floor_count > 0
            else "",
        ]
        + [str(item or "") for item in (final_guard.get("recommended_actions") or [])]
    )
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": bool(final_guard.get("ok", final_blocking_below_floor_count <= 0)),
        "overall_status": str(final_guard.get("overall_status") or ""),
        "apply": bool(apply),
        "max_step_timeout_sec": int(max_step_timeout_sec),
        "repair_step_count": len(repair_plan),
        "attempt_count": len(attempts),
        "initial_guard": {
            "overall_status": str(initial_guard.get("overall_status") or ""),
            "below_floor_count": int(initial_guard.get("below_floor_count", 0) or 0),
            "blocking_below_floor_count": int(
                initial_guard.get("blocking_below_floor_count", initial_guard.get("below_floor_count", 0)) or 0
            ),
            "advisory_below_floor_count": int(initial_guard.get("advisory_below_floor_count", 0) or 0),
            "protected_by_floor_count": int(initial_guard.get("protected_by_floor_count", 0) or 0),
            "overall_letter_grade": str(initial_guard.get("overall_letter_grade") or ""),
        },
        "final_guard": {
            "overall_status": str(final_guard.get("overall_status") or ""),
            "below_floor_count": int(final_guard.get("below_floor_count", 0) or 0),
            "blocking_below_floor_count": final_blocking_below_floor_count,
            "advisory_below_floor_count": final_advisory_below_floor_count,
            "protected_by_floor_count": int(final_guard.get("protected_by_floor_count", 0) or 0),
            "overall_letter_grade": str(final_guard.get("overall_letter_grade") or ""),
        },
        "repair_plan": repair_plan,
        "attempts": attempts,
        "section_floor_autopilot_contract": {
            "generation": "section_grade_floor_autopilot_v2",
            "floor_aware_repairs": True,
            "bounded_step_timeouts": True,
            "bounded_storage_repairs": all(
                "storage-backpressure-autopilot" not in (step.get("cmd") or [])
                or "--quick-bounded" in (step.get("cmd") or [])
                for step in repair_plan
            ),
            "delegates_regression_repairs": any(
                "grade-regression-autopilot" in " ".join(step.get("cmd") or [])
                for step in repair_plan
            ),
        },
        "upgrade_track": {
            "family": "infrabots",
            "upgradeable": True,
            "current_generation": "section_grade_floor_autopilot_v2",
            "co_managed_with": [
                "section_grade_guard",
                "grade_regression_autopilot",
                "runtime_artifact_refresh",
            ],
            "future_upgrade_paths": [
                "adaptive retry budgets keyed to protected-vs-below-floor sections",
                "release-window awareness for live/canary floor recovery commands",
                "tenant-facing floor notifications from the licensing API grade contract",
            ],
        },
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Keep the A-/A section-grade floor alive by triggering targeted repair lanes before sections regress.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--max-step-timeout-seconds",
        type=int,
        default=_safe_int(os.getenv("SECTION_GRADE_AUTOPILOT_STEP_TIMEOUT_SECONDS"), 300),
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        Path(args.project_root).resolve(),
        apply=bool(args.apply),
        max_step_timeout_sec=int(args.max_step_timeout_seconds),
    )
    write_payload(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "section_grade_autopilot "
            f"overall_status={payload.get('overall_status', '')} "
            f"repair_step_count={payload.get('repair_step_count', 0)} "
            f"attempt_count={payload.get('attempt_count', 0)}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
