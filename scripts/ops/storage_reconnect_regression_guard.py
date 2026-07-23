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
    from scripts.ops.long_runtime_common import iso_now, load_json, ordered_unique, write_payload
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from .long_runtime_common import iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "storage_reconnect_regression_guard_latest.json"
DEFAULT_LABEL = "com.dankingsley.storage_eject_guard"
DEFAULT_PLIST = Path.home() / "Library" / "LaunchAgents" / f"{DEFAULT_LABEL}.plist"

REQUIRED_GUARD_SNIPPETS = {
    "disk_appearance_handler": "handleObservedDiskAppeared",
    "disappearance_grace": "confirmDisappearAndRestartLocal",
    "false_disappear_suppression": "external_still_available_after_disappear",
    "mount_poll_timer": "startMountPollTimer",
    "external_switch": "storage-switch-external --no-refresh",
    "local_switch": "storage-switch-local --no-refresh",
    "transition_coordinator": "storage-transition-coordinator --transition-mode external --apply --json",
    "split_brain_reconcile": "split-brain-reconcile --force-failback-if-hashes-match --json",
    "external_backlog_drain": "external-backlog-drain --apply --follow-through",
    "storage_pressure_clearance": "storage-pressure-clearance --apply --max-cycles",
    "halt_refresh": "global-halt-refresh --json",
    "halt_auto_clear": "global-halt-auto-clear --json",
    "operator_cockpit": "operator-cockpit --json",
}


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _run(cmd: list[str], *, cwd: Path, timeout_sec: int = 15) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=max(int(timeout_sec), 1),
        )
        return {
            "cmd": list(cmd),
            "rc": int(proc.returncode),
            "timed_out": False,
            "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-10:]),
            "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-10:]),
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        return {
            "cmd": list(cmd),
            "rc": 124,
            "timed_out": True,
            "stdout_tail": "\n".join(stdout.splitlines()[-10:]),
            "stderr_tail": "\n".join(stderr.splitlines()[-10:]),
        }
    except Exception as exc:
        return {"cmd": list(cmd), "rc": 127, "timed_out": False, "stdout_tail": "", "stderr_tail": str(exc)}


def _launchd_state(project_root: Path, *, check_launchd: bool) -> dict[str, Any]:
    plist_path = Path.home() / "Library" / "LaunchAgents" / f"{DEFAULT_LABEL}.plist"
    if not check_launchd:
        return {
            "checked": False,
            "plist_path": str(plist_path),
            "plist_exists": plist_path.exists(),
            "running": False,
            "status": "skipped",
        }
    uid_result = _run(["id", "-u"], cwd=project_root)
    uid = str(uid_result.get("stdout_tail") or "").strip().splitlines()[-1:] or [""]
    label_ref = f"gui/{uid[0]}/{DEFAULT_LABEL}" if uid[0] else DEFAULT_LABEL
    try:
        proc = subprocess.run(
            ["launchctl", "print", label_ref],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        rc = int(proc.returncode)
        text = f"{proc.stdout or ''}\n{proc.stderr or ''}"
    except Exception as exc:
        rc = 127
        text = str(exc)
    running = rc == 0 and "state = running" in text
    return {
        "checked": True,
        "plist_path": str(plist_path),
        "plist_exists": plist_path.exists(),
        "running": running,
        "status": "ready" if running else "degraded",
        "rc": rc,
    }


def _swift_parse(project_root: Path, guard_path: Path, *, check_swift_parse: bool) -> dict[str, Any]:
    if not check_swift_parse:
        return {"checked": False, "ok": True, "status": "skipped"}
    if not guard_path.exists():
        return {"checked": True, "ok": False, "status": "missing", "rc": 127}
    result = _run(["/usr/bin/swiftc", "-parse", str(guard_path)], cwd=project_root, timeout_sec=30)
    ok = int(result.get("rc", 1)) == 0
    return {
        "checked": True,
        "ok": ok,
        "status": "ready" if ok else "blocked",
        "rc": int(result.get("rc", 1)),
        "stderr_tail": result.get("stderr_tail", ""),
    }


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    check_launchd: bool = True,
    check_swift_parse: bool = True,
) -> dict[str, Any]:
    guard_path = project_root / "scripts" / "ops" / "storage_eject_guard.swift"
    install_path = project_root / "scripts" / "install_storage_eject_guard_launchd.sh"
    runner_path = project_root / "scripts" / "ops" / "run_storage_eject_guard_launchd.sh"
    text = guard_path.read_text(encoding="utf-8") if guard_path.exists() else ""

    contract_rows = []
    for name, snippet in REQUIRED_GUARD_SNIPPETS.items():
        present = snippet in text
        contract_rows.append({"name": name, "required_snippet": snippet, "present": present})

    missing = [row["name"] for row in contract_rows if not bool(row.get("present", False))]
    launchd = _launchd_state(project_root, check_launchd=check_launchd)
    swift_parse = _swift_parse(project_root, guard_path, check_swift_parse=check_swift_parse)

    storage_mount = load_json(project_root / "governance" / "health" / "storage_mount_guard_latest.json")
    storage_control = load_json(project_root / "governance" / "health" / "ingestion_storage_control_latest.json")
    split_brain = load_json(project_root / "governance" / "health" / "storage_split_brain_reconciler_latest.json")
    halt_status = load_json(project_root / "governance" / "health" / "global_risk_killswitch_latest.json")

    split_summary = split_brain.get("summary") if isinstance(split_brain.get("summary"), dict) else {}
    unresolved_conflicts = _safe_int(split_summary.get("unresolved_conflicts"), 0)
    backpressure = storage_control.get("backpressure") if isinstance(storage_control.get("backpressure"), dict) else {}
    storage_status = str(storage_control.get("overall_status") or "missing")
    halt_clear_blockers = halt_status.get("clear_blockers") if isinstance(halt_status.get("clear_blockers"), list) else []
    live_recovery_blockers = ordered_unique(
        [
            "external_mount_unavailable"
            if storage_mount and not bool(storage_mount.get("external_available", storage_mount.get("mount_present", True)))
            else "",
            "split_brain_unresolved" if unresolved_conflicts > 0 else "",
            "storage_pressure_active" if storage_status in {"blocked", "critical"} else "",
            "global_halt_clear_blocked" if halt_clear_blockers else "",
        ]
    )

    contract_ok = not missing and guard_path.exists() and install_path.exists() and runner_path.exists() and bool(swift_parse.get("ok", True))
    automation_installed = bool(launchd.get("plist_exists", False))
    automation_running = bool(launchd.get("running", False)) if check_launchd else True

    overall_status = "ready"
    if not contract_ok:
        overall_status = "blocked"
    elif not automation_installed or not automation_running:
        overall_status = "degraded"
    elif live_recovery_blockers:
        overall_status = "degraded"

    recommended_actions = ordered_unique(
        [
            "reinstall the storage eject guard LaunchAgent so reconnect/eject events keep running automatically"
            if not automation_installed or not automation_running
            else "",
            "repair the reconnect aftercare snippets before trusting automatic failback" if missing else "",
            "let storage-pressure-clearance and external-backlog-drain finish before safe halt auto-clear"
            if "storage_pressure_active" in live_recovery_blockers
            else "",
            "run split-brain reconciliation before deleting local fallback artifacts" if unresolved_conflicts > 0 else "",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "contract_ok": contract_ok,
        "contract_rows": contract_rows,
        "missing_contracts": missing,
        "automation": {
            "launchd_label": DEFAULT_LABEL,
            "install_script": str(install_path),
            "install_script_exists": install_path.exists(),
            "runner_script": str(runner_path),
            "runner_script_exists": runner_path.exists(),
            "launchd": launchd,
            "swift_parse": swift_parse,
        },
        "live_recovery": {
            "blockers": live_recovery_blockers,
            "storage_mode": str(storage_mount.get("storage_mode") or storage_mount.get("mode") or ""),
            "external_available": bool(storage_mount.get("external_available", False)) if storage_mount else False,
            "split_brain_unresolved_conflicts": unresolved_conflicts,
            "storage_control_status": storage_status,
            "core_pending_lines": _safe_int(backpressure.get("core_pending_lines"), 0),
            "total_pending_lines": _safe_int(backpressure.get("total_pending_lines"), 0),
            "halt_clear_blockers": halt_clear_blockers,
        },
        "metrics": {
            "missing_contract_count": len(missing),
            "live_recovery_blocker_count": len(live_recovery_blockers),
            "split_brain_unresolved_conflicts": unresolved_conflicts,
            "total_pending_lines": _safe_int(backpressure.get("total_pending_lines"), 0),
        },
        "regression_guard_contract": {
            "requires_split_brain_reconcile": True,
            "requires_backlog_drain": True,
            "requires_storage_pressure_clearance": True,
            "requires_global_halt_refresh": True,
            "requires_global_halt_auto_clear": True,
            "requires_operator_cockpit_refresh": True,
        },
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Guard the BOT_LOGS eject/reconnect automatic recovery contract.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--skip-launchd", action="store_true")
    parser.add_argument("--skip-swift-parse", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    payload = build_payload(
        project_root,
        check_launchd=not bool(args.skip_launchd),
        check_swift_parse=not bool(args.skip_swift_parse),
    )
    write_payload(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "storage_reconnect_regression_guard "
            f"overall_status={payload.get('overall_status', '')} "
            f"missing_contracts={len(payload.get('missing_contracts') or [])}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
