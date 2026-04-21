#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.runtime_python import resolve_runtime_python
    from scripts.ops import writer_cycle_coordinator as writer_src
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from core.runtime_python import resolve_runtime_python
    from scripts.ops import writer_cycle_coordinator as writer_src


PY = resolve_runtime_python(PROJECT_ROOT)
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "storage_switch_orchestrator_latest.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.storage_override"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _parse_json_output(text: str) -> dict[str, Any]:
    for line in reversed([raw.strip() for raw in str(text or "").splitlines() if raw.strip()]):
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _run_command(
    cmd: list[str],
    *,
    cwd: Path,
    timeout_sec: int,
    env_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    env = os.environ.copy()
    if env_overrides:
        env.update({str(key): str(value) for key, value in env_overrides.items()})

    started = datetime.now(timezone.utc)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=max(int(timeout_sec), 1),
            env=env,
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

    payload = _parse_json_output(stdout)
    duration_ms = round((datetime.now(timezone.utc) - started).total_seconds() * 1000.0, 3)
    return {
        "cmd": list(cmd),
        "rc": rc,
        "duration_ms": duration_ms,
        "payload": payload,
        "stdout_tail": "\n".join(stdout.splitlines()[-12:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-12:]),
        "timed_out": timed_out,
    }


def _step_status(result: dict[str, Any]) -> str:
    if bool(result.get("timed_out", False)):
        return "timed_out"
    if int(result.get("rc", 1)) != 0:
        return "error"
    return "ok"


def _step_record(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": _step_status(result),
        "rc": int(result.get("rc", 1)),
        "duration_ms": float(result.get("duration_ms", 0.0) or 0.0),
        "timed_out": bool(result.get("timed_out", False)),
        "cmd": list(result.get("cmd") or []),
        "stdout_tail": str(result.get("stdout_tail") or ""),
        "stderr_tail": str(result.get("stderr_tail") or ""),
    }


def _mode_matches_target(actual_mode: str, target_mode: str) -> bool:
    actual = str(actual_mode or "").strip()
    target = str(target_mode or "").strip()
    if target == "local":
        return actual in {"local_fallback", "local_fallback_split_brain"}
    return actual == "external"


def _write_storage_override(mode: str, override_path: Path) -> dict[str, Any]:
    override_path.parent.mkdir(parents=True, exist_ok=True)
    changed = False
    if mode == "local":
        body = "# Auto-managed by storage_switch_orchestrator.py\nBOT_LOGS_PREFER_EXTERNAL=0\n"
        current = ""
        if override_path.exists():
            try:
                current = override_path.read_text(encoding="utf-8")
            except Exception:
                current = ""
        if current != body:
            override_path.write_text(body, encoding="utf-8")
            changed = True
    else:
        if override_path.exists():
            override_path.unlink()
            changed = True
    return {
        "path": str(override_path),
        "mode": mode,
        "changed": changed,
        "exists": bool(override_path.exists()),
    }


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    target_mode: str,
    restart: bool,
    eject: bool,
    quiesce_only: bool = False,
    override_path: Path = DEFAULT_OVERRIDE_PATH,
    mount_root: str = "",
    poll_seconds: float = 2.0,
    wait_timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    target_mode = str(target_mode or "external").strip().lower()
    if target_mode not in {"local", "external"}:
        raise ValueError(f"unsupported target_mode: {target_mode}")

    if eject and target_mode != "local":
        raise ValueError("eject is only supported when target_mode=local")

    mount_root = str(mount_root or os.getenv("BOT_LOGS_EXTERNAL_MOUNT", "/Volumes/BOT_LOGS")).strip() or "/Volumes/BOT_LOGS"
    opsctl = project_root / "scripts" / "ops" / "opsctl.sh"
    health_root = project_root / "governance" / "health"
    should_stop = bool(restart or quiesce_only)
    should_restart = bool(restart and not quiesce_only)

    writer_before = writer_src.writer_state_snapshot(project_root)
    wait_for_writer = {
        "requested": False,
        "completed": not bool(writer_before.get("active", False)) if should_stop else True,
        "timed_out": False,
        "attempts": 0,
        "waited_seconds": 0.0,
        "final_state": writer_before,
    }
    writer_after_wait = writer_before
    steps: dict[str, Any] = {}

    if should_stop:
        stop = _run_command([str(opsctl), "stop"], cwd=project_root, timeout_sec=120)
        steps["stop_stack"] = _step_record(stop)
        wait_for_writer = writer_src._wait_for_writer_idle(
            project_root,
            poll_seconds=float(poll_seconds),
            wait_timeout_seconds=float(wait_timeout_seconds),
        )
        writer_after_wait = wait_for_writer.get("final_state") if isinstance(wait_for_writer.get("final_state"), dict) else writer_before

    override_result = _write_storage_override(target_mode, override_path)
    prefer_external = "1" if target_mode == "external" else "0"
    failback = _run_command(
        [str(PY), str(project_root / "scripts" / "ops" / "storage_failback_sync.py"), "--json"],
        cwd=project_root,
        timeout_sec=240,
        env_overrides={"BOT_LOGS_PREFER_EXTERNAL": prefer_external},
    )
    steps["storage_failback_sync"] = _step_record(failback)
    failback_payload = failback.get("payload") if isinstance(failback.get("payload"), dict) else {}
    achieved_target_mode = _mode_matches_target(str(failback_payload.get("mode") or ""), target_mode)

    if target_mode == "external":
        reconcile = _run_command(
            [str(PY), str(project_root / "scripts" / "ops" / "storage_split_brain_reconciler.py"), "--json"],
            cwd=project_root,
            timeout_sec=180,
        )
        steps["storage_split_brain_reconciler"] = _step_record(reconcile)

    if should_restart:
        refresh = _run_command(
            [str(opsctl), "feed-refresh", "--source", "all"],
            cwd=project_root,
            timeout_sec=300,
        )
        steps["feed_refresh"] = _step_record(refresh)
        watchdog = _run_command(
            [str(PY), str(project_root / "scripts" / "ops" / "process_watchdog.py"), "--json"],
            cwd=project_root,
            timeout_sec=180,
            env_overrides={"OPS_WATCHDOG_REFRESH_REPORTS": "0"},
        )
        steps["process_watchdog"] = _step_record(watchdog)
        transition = _run_command(
            [
                str(PY),
                str(project_root / "scripts" / "ops" / "storage_transition_coordinator.py"),
                "--transition-mode",
                target_mode,
                "--apply",
                "--json",
            ],
            cwd=project_root,
            timeout_sec=240,
        )
        steps["storage_transition_coordinator"] = _step_record(transition)

    disk_eject_attempted = False
    disk_eject_completed = False
    if eject and achieved_target_mode and not bool(wait_for_writer.get("timed_out", False)):
        disk_eject_attempted = True
        eject_result = _run_command(
            ["diskutil", "eject", mount_root],
            cwd=project_root,
            timeout_sec=90,
        )
        steps["disk_eject"] = _step_record(eject_result)
        disk_eject_completed = steps["disk_eject"]["status"] == "ok"

    writer_after = writer_src.writer_state_snapshot(project_root)
    step_statuses = [str(row.get("status") or "") for row in steps.values() if isinstance(row, dict)]
    hard_fail = any(status in {"error", "timed_out"} for status in step_statuses)

    if quiesce_only and achieved_target_mode and not hard_fail:
        overall_status = "quiesced_switched"
        ok = True
    elif achieved_target_mode and not hard_fail and (not eject or disk_eject_completed or not disk_eject_attempted):
        overall_status = "switched"
        ok = True
    elif achieved_target_mode and not hard_fail and eject and not disk_eject_attempted:
        overall_status = "switched_eject_deferred"
        ok = False
    elif achieved_target_mode:
        overall_status = "switched_with_followups"
        ok = False
    else:
        overall_status = "target_mode_not_achieved"
        ok = False

    payload = {
        "timestamp_utc": _utc_now(),
        "schema_version": 1,
        "ok": ok,
        "overall_status": overall_status,
        "target_mode": target_mode,
        "restart": bool(restart),
        "quiesce_only": bool(quiesce_only),
        "eject": bool(eject),
        "mount_root": mount_root,
        "override": override_result,
        "writer_state_before": writer_before,
        "wait_for_writer": wait_for_writer,
        "writer_state_after_wait": writer_after_wait,
        "writer_state_after": writer_after,
        "achieved_target_mode": achieved_target_mode,
        "applied_mode": str(failback_payload.get("mode") or ""),
        "active_root": str(failback_payload.get("active_root") or ""),
        "steps": steps,
        "storage_failback_sync": failback_payload,
        "disk_eject_attempted": disk_eject_attempted,
        "disk_eject_completed": disk_eject_completed,
        "out_file": str(health_root / "storage_switch_orchestrator_latest.json"),
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Coordinate BOT_LOGS storage switching by quiescing the stack before failback and restart.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--target-mode", choices=("local", "external"), required=True)
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--override-file", default=str(DEFAULT_OVERRIDE_PATH))
    parser.add_argument("--mount-root", default=os.getenv("BOT_LOGS_EXTERNAL_MOUNT", "/Volumes/BOT_LOGS"))
    parser.add_argument("--no-restart", action="store_true")
    parser.add_argument("--quiesce-only", action="store_true")
    parser.add_argument("--eject", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--wait-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    payload = build_payload(
        project_root,
        target_mode=args.target_mode,
        restart=not bool(args.no_restart),
        eject=bool(args.eject),
        quiesce_only=bool(args.quiesce_only),
        override_path=Path(args.override_file).expanduser(),
        mount_root=str(args.mount_root or ""),
        poll_seconds=float(args.poll_seconds),
        wait_timeout_seconds=float(args.wait_timeout_seconds),
    )

    out_file = Path(args.out_file).expanduser()
    _write_json(out_file, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "storage_switch_orchestrator "
            f"target_mode={payload.get('target_mode')} "
            f"achieved_target_mode={int(bool(payload.get('achieved_target_mode', False)))} "
            f"applied_mode={payload.get('applied_mode', '')} "
            f"overall_status={payload.get('overall_status', '')}"
        )
    return 0 if bool(payload.get("ok", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
