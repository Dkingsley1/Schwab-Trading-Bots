#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import (
        PROJECT_ROOT,
        load_json,
        run_bounded_process_group,
    )
    from scripts.ops.scheduled_lifecycle_common import (
        command_tail,
        infer_deferred_reason,
        interval_seconds,
        lifecycle_receipt,
        stamp_artifact,
        utc_now,
    )
else:
    from .long_runtime_common import PROJECT_ROOT, load_json, run_bounded_process_group
    from .scheduled_lifecycle_common import (
        command_tail,
        infer_deferred_reason,
        interval_seconds,
        lifecycle_receipt,
        stamp_artifact,
        utc_now,
    )


def _resolve(project_root: Path, raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else project_root / path


def _terminal_status(
    *,
    rc: int,
    timed_out: bool,
    artifact_present_after: bool,
    deferred_reason: str,
    base_payload: dict[str, Any] | None = None,
) -> tuple[str, str, bool]:
    if timed_out:
        return "timed_out", "command_timed_out", False
    if not artifact_present_after:
        return "artifact_missing_after_run", "artifact_missing_after_run", False
    if deferred_reason:
        return "deferred", "", True
    if rc == 0:
        return "completed", "", True
    if rc == 2 and isinstance(base_payload, dict) and base_payload:
        return "completed_with_findings", "", True
    if rc != 0:
        return "failed", f"command_rc_{rc}", False
    return "completed", "", True


def run_command(
    command: list[str], *, cwd: Path, timeout_seconds: int
) -> dict[str, Any]:
    try:
        return run_bounded_process_group(
            command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        return {
            "rc": 127,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "timed_out": False,
        }


def build_payload(
    *,
    project_root: Path,
    job_id: str,
    artifact: Path,
    schedule_interval_seconds: float,
    deadline_seconds: int,
    command: list[str],
) -> tuple[dict[str, Any], int]:
    started = utc_now()
    artifact_present_before = artifact.exists()
    result = run_command(command, cwd=project_root, timeout_seconds=deadline_seconds)
    completed = utc_now()
    artifact_present_after = artifact.exists()
    stdout_tail = command_tail(str(result.get("stdout") or ""))
    stderr_tail = command_tail(str(result.get("stderr") or ""))
    deferred_reason = infer_deferred_reason(
        stdout=stdout_tail,
        stderr=stderr_tail,
        rc=int(result.get("rc", 1)),
    )
    base_payload = load_json(artifact) if artifact_present_after else {}
    terminal_status, failure_reason, ok = _terminal_status(
        rc=int(result.get("rc", 1)),
        timed_out=bool(result.get("timed_out", False)),
        artifact_present_after=artifact_present_after,
        deferred_reason=deferred_reason,
        base_payload=base_payload,
    )
    receipt = lifecycle_receipt(
        job_id=job_id,
        scheduled=True,
        started_utc=started.astimezone(timezone.utc),
        completed_utc=completed.astimezone(timezone.utc),
        schedule_interval_seconds=schedule_interval_seconds,
        rc=int(result.get("rc", 1)),
        terminal_status=terminal_status,
        ok=ok,
        deferred_reason=deferred_reason,
        failure_reason=failure_reason,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        artifact_present_before=artifact_present_before,
        artifact_present_after=artifact_present_after,
        deadline_seconds=deadline_seconds,
        command=command,
        timed_out=bool(result.get("timed_out", False)),
    )
    updated = stamp_artifact(artifact, receipt, base_payload=base_payload)
    return updated, 0 if ok else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a scheduled ops command and stamp its latest artifact with lifecycle evidence."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--schedule-interval-seconds", type=float, default=300.0)
    parser.add_argument("--deadline-seconds", type=int, default=300)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    if args.command and args.command[0] == "--":
        command = args.command[1:]
    else:
        command = args.command
    if not command:
        print("run_scheduled_lifecycle_job: missing command after --", file=sys.stderr)
        return 2

    project_root = args.project_root.expanduser().resolve()
    artifact = _resolve(project_root, args.artifact)
    payload, rc = build_payload(
        project_root=project_root,
        job_id=args.job_id,
        artifact=artifact,
        schedule_interval_seconds=interval_seconds(args.schedule_interval_seconds),
        deadline_seconds=max(int(args.deadline_seconds), 1),
        command=command,
    )

    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        lifecycle = payload.get("job_lifecycle") if isinstance(payload, dict) else {}
        print(
            "scheduled_lifecycle_job "
            f"job_id={args.job_id} "
            f"status={lifecycle.get('terminal_status', '') if isinstance(lifecycle, dict) else ''} "
            f"rc={lifecycle.get('rc', '') if isinstance(lifecycle, dict) else ''}"
        )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
