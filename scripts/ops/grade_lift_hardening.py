#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.runtime_python import resolve_runtime_python
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, ordered_unique, write_payload
else:
    from core.runtime_python import resolve_runtime_python
    from .long_runtime_common import PROJECT_ROOT, iso_now, ordered_unique, write_payload


PY = resolve_runtime_python(PROJECT_ROOT)
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "grade_lift_hardening_latest.json"
Runner = Callable[[list[str], Path, int], dict[str, Any]]


def _parse_json_output(text: str) -> dict[str, Any]:
    for raw in reversed([line.strip() for line in str(text or "").splitlines() if line.strip()]):
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _run(cmd: list[str], project_root: Path, timeout_sec: int) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=max(int(timeout_sec), 1),
        )
        return {
            "cmd": list(cmd),
            "rc": int(proc.returncode),
            "payload": _parse_json_output(proc.stdout or ""),
            "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-12:]),
            "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-12:]),
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        return {
            "cmd": list(cmd),
            "rc": 124,
            "payload": _parse_json_output(stdout),
            "stdout_tail": "\n".join(stdout.splitlines()[-12:]),
            "stderr_tail": "\n".join(stderr.splitlines()[-12:]) or "timeout",
        }


def _step_specs(project_root: Path, *, apply_storage_remediations: bool) -> list[dict[str, Any]]:
    ops_root = project_root / "scripts" / "ops"
    storage_cmd = [str(PY), str(ops_root / "storage_backpressure_autopilot.py"), "--json"]
    if apply_storage_remediations:
        storage_cmd.insert(-1, "--apply")
    return [
        {"name": "feature_store_manifest", "cmd": [str(PY), str(project_root / "scripts" / "feature_store_manifest.py"), "--json"], "timeout_sec": 180},
        {"name": "snapshot_coverage", "cmd": [str(PY), str(project_root / "scripts" / "snapshot_coverage_sentinel.py"), "--json"], "timeout_sec": 180},
        {"name": "multiple_testing_guard", "cmd": [str(PY), str(project_root / "scripts" / "multiple_testing_guard.py"), "--json"], "timeout_sec": 180},
        {"name": "decay_monitor", "cmd": [str(PY), str(project_root / "scripts" / "decay_monitor.py"), "--json"], "timeout_sec": 180},
        {"name": "training_lineage_manifest", "cmd": [str(PY), str(ops_root / "training_lineage_manifest.py"), "--json"], "timeout_sec": 180},
        {"name": "broker_readiness", "cmd": [str(PY), str(ops_root / "premarket_token_guard.py"), "--json"], "timeout_sec": 180},
        {"name": "session_ready", "cmd": [str(PY), str(project_root / "scripts" / "session_ready_check.py"), "--json"], "timeout_sec": 180},
        {"name": "storage_failback_sync", "cmd": [str(PY), str(ops_root / "storage_failback_sync.py"), "--json"], "timeout_sec": 180},
        {"name": "promotion_packet_builder", "cmd": [str(PY), str(project_root / "scripts" / "promotion_packet_builder.py"), "--json"], "timeout_sec": 180},
        {"name": "promotion_autopilot_packet", "cmd": [str(PY), str(ops_root / "promotion_autopilot_packet.py"), "--json"], "timeout_sec": 180},
        {"name": "canary_rollout_guard", "cmd": [str(PY), str(project_root / "scripts" / "canary_rollout_guard.py")], "timeout_sec": 45, "optional": True},
        {"name": "storage_backpressure_autopilot", "cmd": storage_cmd, "timeout_sec": 900},
        {"name": "ingestion_storage_control", "cmd": [str(PY), str(ops_root / "ingestion_storage_control.py"), "--json"], "timeout_sec": 180},
        {"name": "storage_resilience_control", "cmd": [str(PY), str(ops_root / "storage_resilience_control.py"), "--fast", "--json"], "timeout_sec": 180},
        {"name": "security_evidence_autofix", "cmd": [str(PY), str(ops_root / "security_evidence_autofix.py"), "--json"], "timeout_sec": 900},
        {"name": "security_audit", "cmd": [str(PY), str(project_root / "scripts" / "security_hardening_audit.py")], "timeout_sec": 180},
        {"name": "incident_timeline", "cmd": [str(PY), str(ops_root / "incident_timeline.py"), "--json"], "timeout_sec": 180},
        {"name": "incident_review_packet", "cmd": [str(PY), str(ops_root / "incident_review_packet.py"), "--json"], "timeout_sec": 180},
        {"name": "incident_closeout_autopilot", "cmd": [str(PY), str(ops_root / "incident_closeout_autopilot.py"), "--json"], "timeout_sec": 180},
        {"name": "canary_auto_tuner", "cmd": [str(PY), str(ops_root / "canary_auto_tuner.py"), "--json"], "timeout_sec": 180},
        {"name": "live_canary_control", "cmd": [str(PY), str(ops_root / "live_canary_control.py"), "--json"], "timeout_sec": 180},
        {"name": "training_quality_control", "cmd": [str(PY), str(ops_root / "training_quality_control.py"), "--json"], "timeout_sec": 180},
        {"name": "runtime_throttle_control", "cmd": [str(PY), str(ops_root / "runtime_throttle_control.py"), "--json"], "timeout_sec": 180},
        {"name": "platform_control_plane", "cmd": [str(PY), str(project_root / "scripts" / "platform_control_plane_report.py"), "--json"], "timeout_sec": 180},
        {"name": "live_readiness_smoke", "cmd": [str(PY), str(project_root / "scripts" / "live_readiness_smoke.py"), "--json"], "timeout_sec": 180},
        {"name": "autonomy_control_plane", "cmd": [str(PY), str(ops_root / "autonomy_control_plane.py"), "--json"], "timeout_sec": 180},
        {"name": "runtime_gate_dashboard", "cmd": [str(PY), str(ops_root / "runtime_gate_dashboard.py"), "--json"], "timeout_sec": 180},
    ]


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply_storage_remediations: bool = True,
    runner: Runner | None = None,
) -> dict[str, Any]:
    run_step = runner or _run
    steps: list[dict[str, Any]] = []
    blocked = 0
    errors = 0
    for spec in _step_specs(project_root, apply_storage_remediations=apply_storage_remediations):
        result = run_step(list(spec["cmd"]), project_root, int(spec.get("timeout_sec", 180)))
        payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
        status = str(payload.get("overall_status") or "").strip().lower()
        if not status:
            status = "ready" if int(result.get("rc", 1)) == 0 else "error"
        if status == "error" and bool(spec.get("optional", False)):
            status = "degraded"
        if status in {"blocked", "critical"}:
            blocked += 1
        if int(result.get("rc", 1)) not in {0, 2} and status not in {"blocked", "critical", "degraded", "ready"}:
            errors += 1
        steps.append(
            {
                "name": spec["name"],
                "status": status,
                "rc": int(result.get("rc", 1)),
                "optional": bool(spec.get("optional", False)),
                "cmd": list(result.get("cmd") or []),
                "payload_summary": {
                    key: payload.get(key)
                    for key in ("overall_status", "ok", "readiness_score", "training_quality_score", "lineage_score", "autonomy_score")
                    if key in payload
                },
                "stdout_tail": str(result.get("stdout_tail") or ""),
                "stderr_tail": str(result.get("stderr_tail") or ""),
            }
        )

    overall_status = "ready"
    if blocked:
        overall_status = "blocked"
    elif errors:
        overall_status = "degraded"

    recommended_actions = ordered_unique(
        [
            "repair training lineage, replay proof, and promotion packet completeness before expecting training grades to move materially"
            if any(step["name"] in {"training_lineage_manifest", "training_quality_control", "promotion_autopilot_packet"} and step["status"] != "ready" for step in steps)
            else "",
            "keep the storage backpressure lane in apply mode until the core queue is back under target"
            if any(step["name"] == "ingestion_storage_control" and step["status"] != "ready" for step in steps)
            else "",
            "keep security evidence autofix fresh so secret-scan and mutation-journal proof stop dragging governance grades"
            if any(step["name"] in {"security_evidence_autofix", "security_audit"} and step["status"] != "ready" for step in steps)
            else "",
            "clear the incident closeout blockers before expecting autonomy and reporting grades to rise"
            if any(step["name"] == "incident_closeout_autopilot" and step["status"] != "ready" for step in steps)
            else "",
            "use supervised canary mode, not full live submit, as the next step once live canary control turns ready"
            if any(step["name"] == "live_canary_control" for step in steps)
            else "",
            "refresh the runtime throttle artifact so autonomy scoring reflects the active throttling bot instead of a missing surface"
            if any(step["name"] == "runtime_throttle_control" and step["status"] != "ready" for step in steps)
            else "",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "apply_storage_remediations": bool(apply_storage_remediations),
        "step_count": len(steps),
        "blocked_step_count": blocked,
        "error_step_count": errors,
        "steps": steps,
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the multi-surface hardening lane used to raise system grades.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--no-apply-storage-remediations", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        Path(args.project_root).resolve(),
        apply_storage_remediations=not bool(args.no_apply_storage_remediations),
    )
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "grade_lift_hardening "
            f"overall_status={payload.get('overall_status', '')} "
            f"step_count={payload.get('step_count', 0)} "
            f"blocked_step_count={payload.get('blocked_step_count', 0)}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
