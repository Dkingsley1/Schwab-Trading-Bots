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
    from scripts.ops.long_runtime_common import (
        iso_now,
        load_json,
        ordered_unique,
        write_payload,
    )
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from .long_runtime_common import iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = (
    PROJECT_ROOT / "governance" / "health" / "storage_reconnect_infrabot_latest.json"
)
PYTHON_BIN = Path(sys.executable)
REGRESSION_GUARD = (
    Path(__file__).resolve().with_name("storage_reconnect_regression_guard.py")
)
HALT_STATUS_FILES = [
    "global_killswitch_latest.json",
    "global_halt_auto_clear_latest.json",
    "global_risk_killswitch_latest.json",
]


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _tail_text(text: str, *, max_lines: int = 10, max_chars: int = 4000) -> str:
    tail = "\n".join(str(text or "").splitlines()[-max_lines:])
    if len(tail) <= max_chars:
        return tail
    omitted = len(tail) - max_chars
    return f"...<truncated {omitted} chars>\n{tail[-max_chars:]}"


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
        stdout = (
            exc.stdout.decode("utf-8", errors="ignore")
            if isinstance(exc.stdout, bytes)
            else str(exc.stdout or "")
        )
        stderr = (
            exc.stderr.decode("utf-8", errors="ignore")
            if isinstance(exc.stderr, bytes)
            else str(exc.stderr or "")
        )
        rc = 124
        timed_out = True
    payload: dict[str, Any] = {}
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
        "payload": payload,
        "stdout_tail": _tail_text(stdout),
        "stderr_tail": _tail_text(stderr),
    }


def _load_halt_status(project_root: Path) -> dict[str, Any]:
    for name in HALT_STATUS_FILES:
        payload = load_json(project_root / "governance" / "health" / name)
        if payload:
            out = dict(payload)
            out["_source_file"] = name
            return out
    return {}


def _halt_has_explicit_state(payload: dict[str, Any]) -> bool:
    return any(
        key in payload
        for key in ("halt", "halt_latched", "halt_required", "would_rehalt")
    )


def _halt_latched_or_required(payload: dict[str, Any]) -> bool:
    return bool(
        payload.get("halt", False)
        or payload.get("halt_latched", False)
        or payload.get("halt_required", False)
        or payload.get("would_rehalt", False)
    )


def _halt_clear_needed(payload: dict[str, Any], clear_blockers: list[Any]) -> bool:
    if not clear_blockers:
        return False
    if not _halt_has_explicit_state(payload):
        return True
    return _halt_latched_or_required(payload)


def _guard_payload(project_root: Path, *, timeout_sec: int) -> dict[str, Any]:
    result = _run_json(
        [
            str(PYTHON_BIN),
            str(REGRESSION_GUARD),
            "--project-root",
            str(project_root),
            "--json",
        ],
        cwd=project_root,
        timeout_sec=min(int(timeout_sec), 120),
    )
    payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
    return payload or load_json(
        project_root
        / "governance"
        / "health"
        / "storage_reconnect_regression_guard_latest.json"
    )


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool = False,
    timeout_sec: int = 900,
) -> dict[str, Any]:
    bounded_timeout_sec = max(min(int(timeout_sec), 120), 1)
    guard = _guard_payload(project_root, timeout_sec=timeout_sec)
    storage_control = load_json(
        project_root / "governance" / "health" / "ingestion_storage_control_latest.json"
    )
    halt_status = _load_halt_status(project_root)
    data_plane = load_json(
        project_root
        / "governance"
        / "health"
        / "data_plane_recovery_controller_latest.json"
    )

    live = (
        guard.get("live_recovery")
        if isinstance(guard.get("live_recovery"), dict)
        else {}
    )
    automation = (
        guard.get("automation") if isinstance(guard.get("automation"), dict) else {}
    )
    launchd = (
        automation.get("launchd") if isinstance(automation.get("launchd"), dict) else {}
    )
    runtime_binary = (
        automation.get("runtime_binary")
        if isinstance(automation.get("runtime_binary"), dict)
        else {}
    )
    transition_state = (
        automation.get("transition_state")
        if isinstance(automation.get("transition_state"), dict)
        else {}
    )
    storage_backpressure = (
        storage_control.get("backpressure")
        if isinstance(storage_control.get("backpressure"), dict)
        else {}
    )
    halt_clear_blockers = (
        halt_status.get("clear_blockers")
        if isinstance(halt_status.get("clear_blockers"), list)
        else []
    )
    halt_clear_needed = _halt_clear_needed(halt_status, halt_clear_blockers)
    halt_latched_or_required = _halt_latched_or_required(halt_status)
    data_plane_status = str(data_plane.get("overall_status") or "")
    data_plane_recovery_state = str(data_plane.get("recovery_state") or "")
    data_plane_write_failures = _safe_int(data_plane.get("write_failure_count"), 0)
    data_plane_hot_path_over_budget = _safe_int(
        data_plane.get("hot_path_over_budget_bytes"), 0
    )
    data_plane_storage_halt_needed = bool(
        data_plane_status == "blocked"
        or data_plane_write_failures > 0
        or data_plane_hot_path_over_budget > 0
    )

    repair_plan: list[dict[str, Any]] = []

    def add_plan(name: str, reason: str, cmd: list[str], timeout: int) -> None:
        repair_plan.append(
            {"name": name, "reason": reason, "cmd": cmd, "timeout_sec": timeout}
        )

    if not bool(guard.get("contract_ok", False)):
        add_plan(
            "storage_reconnect_regression_guard",
            f"missing_contracts={len(guard.get('missing_contracts') or [])}",
            [
                str(PYTHON_BIN),
                str(REGRESSION_GUARD),
                "--project-root",
                str(project_root),
                "--json",
            ],
            bounded_timeout_sec,
        )

    if (
        not bool(launchd.get("running", False))
        or not bool(launchd.get("plist_exists", False))
        or not bool(runtime_binary.get("ready", True))
        or str(transition_state.get("status") or "")
        in {"blocked", "critical", "failed"}
    ):
        add_plan(
            "install_storage_eject_guard_launchd",
            "automatic_storage_eject_guard_not_running_or_runtime_not_current",
            [str(project_root / "scripts" / "install_storage_eject_guard_launchd.sh")],
            bounded_timeout_sec,
        )

    if bool(live.get("flap_cooldown_active", False)):
        add_plan(
            "storage_flap_cooldown_status_refresh",
            (
                "bot_logs_flap_cooldown_active "
                f"recent_disappear_count={_safe_int(live.get('recent_disappear_count'), 0)}"
            ),
            [
                str(PYTHON_BIN),
                str(REGRESSION_GUARD),
                "--project-root",
                str(project_root),
                "--json",
            ],
            min(bounded_timeout_sec, 60),
        )

    if _safe_int(live.get("split_brain_unresolved_conflicts"), 0) > 0:
        add_plan(
            "split_brain_reconcile",
            "unresolved_storage_split_brain_conflicts",
            [
                str(project_root / "scripts" / "ops" / "opsctl.sh"),
                "split-brain-reconcile",
                "--force-failback-if-hashes-match",
                "--force-failback-timeout-sec",
                "45",
                "--json",
            ],
            min(bounded_timeout_sec, 75),
        )

    storage_status = str(
        storage_control.get("overall_status")
        or live.get("storage_control_status")
        or ""
    )
    total_pending = _safe_int(
        storage_backpressure.get("total_pending_lines"),
        _safe_int(live.get("total_pending_lines"), 0),
    )
    if storage_status in {"blocked", "critical", "degraded"} or total_pending > 20000:
        add_plan(
            "storage_pressure_clearance",
            f"storage_status={storage_status or 'missing'} total_pending_lines={total_pending}",
            [
                str(project_root / "scripts" / "ops" / "opsctl.sh"),
                "storage-pressure-clearance",
                "--apply",
                "--max-cycles",
                "2",
                "--poll-seconds",
                "5",
                "--wait-timeout-seconds",
                "45",
                "--json",
            ],
            min(max(bounded_timeout_sec, 90), 150),
        )

    if (
        data_plane_storage_halt_needed
        and storage_status not in {"blocked", "critical", "degraded"}
        and total_pending <= 20000
    ):
        add_plan(
            "data_plane_recovery_aftercare",
            (
                f"data_plane_status={data_plane_status or 'missing'} "
                f"write_failures={data_plane_write_failures} "
                f"hot_path_over_budget_bytes={data_plane_hot_path_over_budget}"
            ),
            [
                str(project_root / "scripts" / "ops" / "opsctl.sh"),
                "external-backlog-drain",
                "--json",
            ],
            min(bounded_timeout_sec, 90),
        )

    if halt_clear_needed:
        add_plan(
            "global_halt_safe_refresh",
            (
                f"halt_clear_blockers={len(halt_clear_blockers)} "
                f"data_plane_status={data_plane_status or 'missing'} "
                f"write_failures={data_plane_write_failures} "
                f"hot_path_over_budget_bytes={data_plane_hot_path_over_budget}"
            ),
            [
                str(project_root / "scripts" / "ops" / "opsctl.sh"),
                "global-halt-refresh",
                "--json",
            ],
            min(bounded_timeout_sec, 90),
        )
        add_plan(
            "global_halt_safe_auto_clear",
            "clear_only_when_refreshed_blockers_are_safe",
            [
                str(project_root / "scripts" / "ops" / "opsctl.sh"),
                "global-halt-auto-clear",
                "--json",
            ],
            min(bounded_timeout_sec, 90),
        )

    attempts: list[dict[str, Any]] = []
    if apply:
        for row in repair_plan:
            attempts.append(
                _run_json(
                    list(row.get("cmd") or []),
                    cwd=project_root,
                    timeout_sec=int(row.get("timeout_sec") or timeout_sec),
                )
            )
        attempts.append(
            _run_json(
                [
                    str(PYTHON_BIN),
                    str(REGRESSION_GUARD),
                    "--project-root",
                    str(project_root),
                    "--json",
                ],
                cwd=project_root,
                timeout_sec=min(int(timeout_sec), 120),
            )
        )

    hard_failures = [
        row
        for row in attempts
        if bool(row.get("timed_out", False)) or int(row.get("rc", 1)) not in {0, 2}
    ]
    degraded_attempts = [
        row
        for row in attempts
        if int(row.get("rc", 1)) == 2 and not bool(row.get("timed_out", False))
    ]

    overall_status = "ready"
    if not bool(guard.get("contract_ok", False)) or hard_failures:
        overall_status = "blocked"
    elif (
        repair_plan
        or degraded_attempts
        or str(guard.get("overall_status") or "") == "degraded"
    ):
        overall_status = "degraded"

    recommended_actions = ordered_unique(
        [
            "keep this infrabot on the infrastructure autofix path so storage reconnect drift is caught before the next unplug/replug",
            "keep the reconnect guard LaunchAgent running; it is the automatic handoff from Finder eject/reconnect into storage recovery",
            (
                "leave global halt clearing safe-gated until storage pressure and write-path recovery are actually clear"
                if halt_clear_needed or total_pending > 20000
                else ""
            ),
            (
                "let data-plane aftercare confirm write failures have recovered before widening write-heavy storage lanes"
                if data_plane_storage_halt_needed and not halt_clear_needed
                else ""
            ),
            (
                "keep BOT_LOGS on local fallback or standby until the flap cooldown clears"
                if bool(live.get("flap_cooldown_active", False))
                else ""
            ),
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "apply": bool(apply),
        "guard_status": str(guard.get("overall_status") or "missing"),
        "contract_ok": bool(guard.get("contract_ok", False)),
        "repair_plan": repair_plan,
        "attempts": [
            {
                "cmd": list(row.get("cmd") or []),
                "rc": int(row.get("rc", 1)),
                "timed_out": bool(row.get("timed_out", False)),
            }
            for row in attempts
        ],
        "metrics": {
            "repair_plan_count": len(repair_plan),
            "max_repair_step_timeout_sec": max(
                [int(row.get("timeout_sec") or 0) for row in repair_plan] or [0]
            ),
            "missing_contract_count": len(guard.get("missing_contracts") or []),
            "automation_running": bool(launchd.get("running", False)),
            "runtime_binary_ready": bool(runtime_binary.get("ready", True)),
            "last_transition_status": str(transition_state.get("status") or "missing"),
            "last_transition_event": str(transition_state.get("event") or ""),
            "total_pending_lines": total_pending,
            "halt_clear_blocker_count": len(halt_clear_blockers),
            "halt_status_source": str(halt_status.get("_source_file") or ""),
            "halt_latched_or_required": halt_latched_or_required,
            "halt_clear_needed": halt_clear_needed,
            "data_plane_queue_depth": _safe_int(data_plane.get("queue_depth"), 0),
            "data_plane_storage_halt_needed": bool(data_plane_storage_halt_needed),
            "data_plane_recovery_state": data_plane_recovery_state,
            "data_plane_write_failure_count": data_plane_write_failures,
            "data_plane_hot_path_over_budget_bytes": data_plane_hot_path_over_budget,
            "flap_cooldown_active": bool(live.get("flap_cooldown_active", False)),
            "recent_disappear_count": _safe_int(live.get("recent_disappear_count"), 0),
        },
        "infra_bots": [
            "storage_reconnect_infrabot",
            "storage_reconnect_regression_guard",
            "storage_pressure_clearance_bot",
            "storage_backpressure_autopilot",
            "storage_split_brain_reconciler",
            "global_risk_killswitch_auto_clear",
        ],
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Infrastructure bot for automatic BOT_LOGS reconnect/eject recovery."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        Path(args.project_root).resolve(),
        apply=bool(args.apply),
        timeout_sec=int(args.timeout_sec),
    )
    write_payload(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "storage_reconnect_infrabot "
            f"overall_status={payload.get('overall_status', '')} "
            f"repair_plan={len(payload.get('repair_plan') or [])}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
