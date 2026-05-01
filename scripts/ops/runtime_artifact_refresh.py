#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.runtime_python import resolve_runtime_python
    from scripts.ops.long_runtime_common import iso_now, ordered_unique, write_payload
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from core.runtime_python import resolve_runtime_python
    from .long_runtime_common import iso_now, ordered_unique, write_payload


PY = resolve_runtime_python(PROJECT_ROOT)
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "runtime_artifact_refresh_latest.json"


RefreshRunner = Callable[[dict[str, Any], Path], dict[str, Any]]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_json_output(text: str) -> dict[str, Any]:
    for raw in reversed([line.strip() for line in str(text or "").splitlines() if line.strip()]):
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _artifact_present(path: Path) -> bool:
    return path.exists() and bool(_load_json(path))


def _step_specs(project_root: Path) -> list[dict[str, Any]]:
    ops_root = project_root / "scripts" / "ops"
    health_root = project_root / "governance" / "health"
    return [
        {
            "name": "runtime_access_mode",
            "payload_path": health_root / "runtime_access_mode_latest.json",
            "cmd": [str(PY), str(ops_root / "runtime_access_mode.py"), "status", "--json"],
        },
        {
            "name": "apple_silicon_profile",
            "payload_path": health_root / "apple_silicon_profile_latest.json",
            "cmd": [str(PY), str(ops_root / "apple_silicon_profile.py"), "status", "--json"],
        },
        {
            "name": "memory_efficiency_control",
            "payload_path": health_root / "memory_efficiency_control_latest.json",
            "cmd": [str(PY), str(ops_root / "memory_efficiency_control.py"), "status", "--json"],
        },
        {
            "name": "training_lineage_manifest",
            "payload_path": health_root / "training_lineage_manifest_latest.json",
            "cmd": [str(PY), str(ops_root / "training_lineage_manifest.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "training_quality_control",
            "payload_path": health_root / "training_quality_control_latest.json",
            "cmd": [str(PY), str(ops_root / "training_quality_control.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "architecture_upgrade_scoreboard",
            "payload_path": health_root / "architecture_upgrade_scoreboard_latest.json",
            "cmd": [str(PY), str(ops_root / "architecture_upgrade_scoreboard.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "portfolio_capacity_curve_report",
            "payload_path": project_root / "governance" / "allocator" / "portfolio_capacity_curve_latest.json",
            "cmd": [str(PY), str(project_root / "scripts" / "portfolio_capacity_curve_report.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "cross_host_parity_report",
            "payload_path": health_root / "cross_host_parity_report_latest.json",
            "cmd": [str(PY), str(ops_root / "cross_host_parity_report.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "cost_telemetry",
            "payload_path": health_root / "cost_telemetry_latest.json",
            "cmd": [str(PY), str(ops_root / "cost_telemetry.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "broker_readiness",
            "payload_path": health_root / "broker_readiness_latest.json",
            "cmd": [str(PY), str(ops_root / "premarket_token_guard.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "session_ready",
            "payload_path": health_root / "session_ready_latest.json",
            "cmd": [str(PY), str(project_root / "scripts" / "session_ready_check.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "storage_failback_sync",
            "payload_path": health_root / "storage_failback_sync_latest.json",
            "cmd": [str(PY), str(ops_root / "storage_failback_sync.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "canary_auto_tuner",
            "payload_path": health_root / "canary_auto_tuner_latest.json",
            "cmd": [str(PY), str(ops_root / "canary_auto_tuner.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "canary_rollout_guard",
            "payload_path": health_root / "canary_rollout_latest.json",
            "cmd": [str(PY), str(project_root / "scripts" / "canary_rollout_guard.py")],
            "timeout_sec": 45,
            "optional": True,
        },
        {
            "name": "promotion_autopilot_packet",
            "payload_path": project_root / "governance" / "champion_challenger" / "promotion_autopilot_packet_latest.json",
            "cmd": [str(PY), str(ops_root / "promotion_autopilot_packet.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "training_report",
            "payload_path": health_root / "training_report_latest.json",
            "cmd": [str(PY), str(ops_root / "training_report.py"), "--no-render-pdf", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "platform_control_plane",
            "payload_path": health_root / "platform_control_plane_latest.json",
            "cmd": [str(PY), str(project_root / "scripts" / "platform_control_plane_report.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "security_evidence_autofix",
            "payload_path": health_root / "security_evidence_autofix_latest.json",
            "cmd": [str(PY), str(ops_root / "security_evidence_autofix.py"), "--json"],
            "timeout_sec": 900,
        },
        {
            "name": "security_audit",
            "payload_path": health_root / "security_audit_latest.json",
            "cmd": [str(PY), str(project_root / "scripts" / "security_hardening_audit.py")],
            "timeout_sec": 180,
        },
        {
            "name": "ingestion_storage_control",
            "payload_path": health_root / "ingestion_storage_control_latest.json",
            "cmd": [str(PY), str(ops_root / "ingestion_storage_control.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "storage_pressure_clearance",
            "payload_path": health_root / "storage_pressure_clearance_latest.json",
            "cmd": [str(PY), str(ops_root / "storage_pressure_clearance_bot.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "storage_resilience_control",
            "payload_path": health_root / "storage_resilience_control_latest.json",
            "cmd": [str(PY), str(ops_root / "storage_resilience_control.py"), "--fast", "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "live_runtime_separation_control",
            "payload_path": health_root / "live_runtime_separation_control_latest.json",
            "cmd": [str(PY), str(ops_root / "live_runtime_separation_control.py"), "--json"],
        },
        {
            "name": "auth_lease_manager",
            "payload_path": health_root / "auth_lease_manager_latest.json",
            "cmd": [str(PY), str(ops_root / "auth_lease_manager.py"), "--json"],
        },
        {
            "name": "schwab_auth_supervisor",
            "payload_path": health_root / "schwab_auth_supervisor_latest.json",
            "cmd": [str(PY), str(ops_root / "schwab_auth_supervisor.py"), "--json"],
        },
        {
            "name": "blackstart_recovery",
            "payload_path": health_root / "blackstart_recovery_latest.json",
            "cmd": [str(PY), str(ops_root / "blackstart_recovery.py"), "--json"],
        },
        {
            "name": "sleeve_isolation_guard",
            "payload_path": health_root / "sleeve_isolation_guard_latest.json",
            "cmd": [str(PY), str(ops_root / "sleeve_isolation_guard.py"), "--json"],
        },
        {
            "name": "artifact_freshness_slo",
            "payload_path": health_root / "artifact_freshness_slo_latest.json",
            "cmd": [str(PY), str(ops_root / "artifact_freshness_slo.py"), "--json"],
        },
        {
            "name": "runtime_snapshot_cache_control",
            "payload_path": health_root / "runtime_snapshot_cache_control_latest.json",
            "cmd": [str(PY), str(ops_root / "runtime_snapshot_cache_control.py"), "--json"],
        },
        {
            "name": "remote_alert_control",
            "payload_path": health_root / "remote_alert_control_latest.json",
            "cmd": [str(PY), str(ops_root / "remote_alert_control.py"), "--json"],
        },
        {
            "name": "storage_quota_guard",
            "payload_path": health_root / "storage_quota_guard_latest.json",
            "cmd": [str(PY), str(ops_root / "storage_quota_guard.py"), "--json"],
        },
        {
            "name": "release_freeze_guard",
            "payload_path": health_root / "release_freeze_guard_latest.json",
            "cmd": [str(PY), str(ops_root / "release_freeze_guard.py"), "--json"],
        },
        {
            "name": "roster_resilience_planner",
            "payload_path": health_root / "roster_resilience_planner_latest.json",
            "cmd": [str(PY), str(ops_root / "roster_resilience_planner.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "chaos_drill_coordinator",
            "payload_path": health_root / "chaos_drill_coordinator_latest.json",
            "cmd": [str(PY), str(ops_root / "chaos_drill_coordinator.py"), "--json"],
        },
        {
            "name": "rolling_restart_controller",
            "payload_path": health_root / "rolling_restart_controller_latest.json",
            "cmd": [str(PY), str(ops_root / "rolling_restart_controller.py"), "--json"],
        },
        {
            "name": "incident_timeline",
            "payload_path": health_root / "incident_timeline_latest.json",
            "cmd": [str(PY), str(ops_root / "incident_timeline.py"), "--json"],
        },
        {
            "name": "incident_review_packet",
            "payload_path": health_root / "incident_review_packet_latest.json",
            "cmd": [str(PY), str(ops_root / "incident_review_packet.py"), "--json"],
        },
        {
            "name": "incident_closeout_autopilot",
            "payload_path": health_root / "incident_closeout_autopilot_latest.json",
            "cmd": [str(PY), str(ops_root / "incident_closeout_autopilot.py"), "--json"],
        },
        {
            "name": "live_canary_control",
            "payload_path": health_root / "live_canary_control_latest.json",
            "cmd": [str(PY), str(ops_root / "live_canary_control.py"), "--json"],
        },
        {
            "name": "live_readiness_smoke",
            "payload_path": health_root / "live_readiness_smoke_latest.json",
            "cmd": [str(PY), str(project_root / "scripts" / "live_readiness_smoke.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "runtime_throttle_control",
            "payload_path": health_root / "runtime_throttle_control_latest.json",
            "cmd": [str(PY), str(ops_root / "runtime_throttle_control.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "chrome_headless_guard",
            "payload_path": health_root / "chrome_headless_guard_latest.json",
            "cmd": [str(PY), str(ops_root / "chrome_headless_guard.py"), "--apply", "--json"],
            "timeout_sec": 90,
        },
        {
            "name": "multiple_testing_guard",
            "payload_path": project_root / "governance" / "research" / "multiple_testing_guard_latest.json",
            "cmd": [str(PY), str(project_root / "scripts" / "multiple_testing_guard.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "decay_monitor",
            "payload_path": project_root / "governance" / "research" / "decay_monitor_latest.json",
            "cmd": [str(PY), str(project_root / "scripts" / "decay_monitor.py"), "--json"],
            "timeout_sec": 180,
        },
        {
            "name": "operator_cockpit",
            "payload_path": health_root / "operator_cockpit_latest.json",
            "cmd": [str(PY), str(ops_root / "operator_cockpit.py"), "--json"],
            "timeout_sec": 180,
        },
    ]


def _run_spec(spec: dict[str, Any], project_root: Path) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    payload_path = Path(spec["payload_path"]).expanduser()
    try:
        proc = subprocess.run(
            list(spec["cmd"]),
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=max(int(spec.get("timeout_sec", 120) or 120), 1),
        )
        payload = _parse_json_output(proc.stdout or "")
        if not payload:
            payload = _load_json(payload_path)
        rc = int(proc.returncode)
        stdout_tail = "\n".join((proc.stdout or "").splitlines()[-12:])
        stderr_tail = "\n".join((proc.stderr or "").splitlines()[-12:])
    except subprocess.TimeoutExpired as exc:
        rc = 124
        payload = _load_json(payload_path)
        stdout_tail = "\n".join((exc.stdout or "").splitlines()[-12:]) if isinstance(exc.stdout, str) else ""
        stderr_tail = "\n".join((exc.stderr or "").splitlines()[-12:]) if isinstance(exc.stderr, str) else "timeout"
    duration_ms = round((datetime.now(timezone.utc) - started).total_seconds() * 1000.0, 3)
    return {
        "cmd": list(spec["cmd"]),
        "rc": rc,
        "payload": payload,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "duration_ms": duration_ms,
    }


def _step_status(result: dict[str, Any]) -> str:
    payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
    if int(result.get("rc", 1)) != 0 and not payload:
        return "error"
    if bool(payload.get("busy", False)):
        return "busy"
    if bool(payload.get("skipped", False)):
        return "skipped"
    status = str(payload.get("overall_status") or "").strip().lower()
    if status:
        return status
    if "ok" in payload:
        return "ready" if bool(payload.get("ok", False)) else "blocked"
    return "ok" if int(result.get("rc", 1)) == 0 else "error"


def _payload_summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in ("overall_status", "ok", "timestamp_utc", "mode", "lease_state", "resilience_score"):
        if key in payload:
            summary[key] = payload.get(key)
    return summary


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    specs: list[dict[str, Any]] | None = None,
    runner: RefreshRunner | None = None,
) -> dict[str, Any]:
    refresh_specs = list(specs or _step_specs(project_root))
    run_step = runner or _run_spec
    missing_before = [str(spec["name"]) for spec in refresh_specs if not _artifact_present(Path(spec["payload_path"]))]

    steps: list[dict[str, Any]] = []
    statuses: list[str] = []
    missing_after: list[str] = []
    recovered = 0
    for spec in refresh_specs:
        payload_path = Path(spec["payload_path"])
        result = run_step(spec, project_root)
        payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
        present_after = _artifact_present(payload_path)
        if str(spec["name"]) in missing_before and present_after:
            recovered += 1
        if not present_after:
            missing_after.append(str(spec["name"]))
        status = _step_status(result)
        if status == "error" and bool(spec.get("optional", False)):
            status = "degraded"
        statuses.append(status)
        steps.append(
            {
                "name": str(spec["name"]),
                "status": status,
                "rc": int(result.get("rc", 1)),
                "duration_ms": float(result.get("duration_ms", 0.0) or 0.0),
                "payload_path": str(payload_path),
                "optional": bool(spec.get("optional", False)),
                "artifact_present": present_after,
                "payload_summary": _payload_summary(payload),
                "cmd": list(result.get("cmd") or []),
                "stdout_tail": str(result.get("stdout_tail") or ""),
                "stderr_tail": str(result.get("stderr_tail") or ""),
            }
        )

    optional_names = {str(spec["name"]) for spec in refresh_specs if bool(spec.get("optional", False))}
    required_missing_after = [name for name in missing_after if name not in optional_names]
    error_statuses = {"error"}
    degraded_statuses = {"warn", "thin", "degraded", "needs_work", "needs_review", "blocked", "busy", "skipped"}
    error_step_count = sum(1 for status in statuses if status in error_statuses)
    degraded_step_count = sum(1 for status in statuses if status in degraded_statuses)
    blocked_step_count = sum(1 for status in statuses if status == "blocked")
    overall_status = "ready"
    if error_step_count > 0 or required_missing_after:
        overall_status = "blocked"
    elif degraded_step_count > 0:
        overall_status = "degraded"

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "project_root": str(project_root),
        "ok": overall_status != "blocked",
        "overall_status": overall_status,
        "target_artifact_count": len(refresh_specs),
        "artifact_present_count_after": len(refresh_specs) - len(missing_after),
        "artifacts_recovered_count": recovered,
        "missing_before": missing_before,
        "missing_after": missing_after,
        "required_missing_after": required_missing_after,
        "blocked_step_count": blocked_step_count,
        "degraded_step_count": degraded_step_count,
        "error_step_count": error_step_count,
        "recommended_actions": ordered_unique(
            [
                "./scripts/ops/opsctl.sh dashboard" if not missing_after and error_step_count == 0 else "",
                "inspect the step stderr tails for the artifacts that are still missing" if required_missing_after else "",
                "treat optional proof steps like canary rollout diagnostics as advisory when they time out under live load" if any(name in optional_names for name in missing_after) else "",
                "treat blocked refresh outputs as real runtime issues instead of silent dashboard omissions" if blocked_step_count else "",
            ]
        ),
        "steps": steps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh the runtime dashboard's prerequisite artifacts before grading the live system.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve())
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "runtime_artifact_refresh "
            f"overall_status={payload.get('overall_status', '')} "
            f"recovered={int(payload.get('artifacts_recovered_count', 0) or 0)} "
            f"missing_after={len(payload.get('missing_after') or [])}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
