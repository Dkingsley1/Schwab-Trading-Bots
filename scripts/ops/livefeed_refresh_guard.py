#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, load_json, payload_age_minutes, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, payload_age_minutes, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "livefeed_refresh_guard_latest.json"


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _tail(text: str, lines: int = 20) -> str:
    return "\n".join((text or "").splitlines()[-max(int(lines), 1) :])


def _process_snapshot(project_root: Path, *, source: str, health_pid: int | None = None) -> dict[str, Any]:
    script_path = str(project_root / "scripts" / "ops" / "live_feed_tail.sh")
    guarded_path = str(project_root / "scripts" / "ops" / "live_feed_heavy_guarded.sh")
    source_arg = f"--source {source}"
    try:
        proc = subprocess.run(
            ["ps", "-axo", "pid,ppid,stat,etime,command"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        lines = proc.stdout.splitlines()[1:] if proc.returncode == 0 else []
    except Exception:
        lines = []

    local_rows: list[dict[str, Any]] = []
    heavy_rows: list[dict[str, Any]] = []
    guarded_rows: list[dict[str, Any]] = []
    health_pid_alive = False
    for line in lines:
        parts = line.strip().split(maxsplit=4)
        if len(parts) < 5:
            continue
        pid_text, ppid_text, stat, etime, command = parts
        pid = _safe_int(pid_text, -1)
        if pid <= 0:
            continue
        is_tail = script_path in command or "scripts/ops/live_feed_tail.sh" in command
        is_guarded = guarded_path in command or "scripts/ops/live_feed_heavy_guarded.sh" in command
        if not is_tail and not is_guarded:
            continue
        row = {
            "pid": pid,
            "ppid": _safe_int(ppid_text, 0),
            "stat": stat,
            "etime": etime,
            "command": command[:240],
        }
        if health_pid is not None and pid == int(health_pid):
            health_pid_alive = True
        if is_guarded:
            guarded_rows.append(row)
        if is_tail and "--heavy" in command:
            heavy_rows.append(row)
        elif is_tail and "--snapshot" not in command and (source_arg in command or not source):
            local_rows.append(row)

    return {
        "source": source,
        "health_pid": health_pid,
        "health_pid_alive": health_pid_alive if health_pid is not None else None,
        "local_mirror_process_count": len(local_rows),
        "heavy_process_count": len(heavy_rows),
        "guarded_heavy_process_count": len(guarded_rows),
        "process_count": len(local_rows) + len(heavy_rows) + len(guarded_rows),
        "local_mirror_processes": local_rows[:6],
        "heavy_processes": heavy_rows[:6],
        "guarded_heavy_processes": guarded_rows[:6],
    }


def _run_command(command: list[str], *, cwd: Path, timeout_sec: int) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=max(int(timeout_sec), 1),
        )
        return {
            "command": command,
            "returncode": int(proc.returncode),
            "timed_out": False,
            "stdout_tail": _tail(proc.stdout),
            "stderr_tail": _tail(proc.stderr),
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        return {
            "command": command,
            "returncode": 124,
            "timed_out": True,
            "stdout_tail": _tail(stdout),
            "stderr_tail": _tail(stderr),
        }


def _route_specs(opsctl: Path) -> list[dict[str, Any]]:
    return [
        {"name": "feed_refresh_all", "command": [str(opsctl), "feed-refresh", "--dry-run", "--source", "all"], "expects": ["feed_refresh_dry_run=1", "source=all", "mirror_only=0"]},
        {"name": "feed_refresh_schwab", "command": [str(opsctl), "feed-refresh", "--dry-run", "--source", "schwab"], "expects": ["feed_refresh_dry_run=1", "source=schwab", "mirror_only=0"]},
        {"name": "feed_refresh_coinbase", "command": [str(opsctl), "feed-refresh", "--dry-run", "--source", "coinbase"], "expects": ["feed_refresh_dry_run=1", "source=coinbase", "mirror_only=0"]},
        {"name": "feed_refresh_fx", "command": [str(opsctl), "feed-refresh", "--dry-run", "--source", "fx"], "expects": ["feed_refresh_dry_run=1", "source=fx", "mirror_only=0"]},
        {"name": "livefeed_refresh_alias", "command": [str(opsctl), "livefeed-refresh", "--dry-run"], "expects": ["feed_refresh_dry_run=1", "cmd=livefeed-refresh", "source=all", "mirror_only=1"]},
        {"name": "live_feed_refresh_alias", "command": [str(opsctl), "live-feed-refresh", "--dry-run"], "expects": ["feed_refresh_dry_run=1", "cmd=live-feed-refresh", "source=all", "mirror_only=1"]},
    ]


def _check_routes(project_root: Path, *, timeout_sec: int) -> tuple[list[dict[str, Any]], list[str]]:
    opsctl = project_root / "scripts" / "ops" / "opsctl.sh"
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for spec in _route_specs(opsctl):
        result = _run_command(spec["command"], cwd=project_root, timeout_sec=timeout_sec)
        stdout = str(result.get("stdout_tail") or "")
        missing = [token for token in spec["expects"] if token not in stdout]
        ok = int(result.get("returncode", 1)) == 0 and not missing and not bool(result.get("timed_out", False))
        row = {
            "name": spec["name"],
            "ok": ok,
            "missing_tokens": missing,
            **result,
        }
        rows.append(row)
        if not ok:
            blockers.append(f"route_failed:{spec['name']}")
    return rows, blockers


def _health_check(project_root: Path, *, freshness_minutes: float) -> dict[str, Any]:
    path = project_root / "governance" / "health" / "livefeed_local_latest.json"
    payload = load_json(path)
    age = payload_age_minutes(payload, path) if path.exists() else None
    running = str(payload.get("status") or "").lower() == "running"
    alive = bool(payload.get("alive", False))
    health_writer = bool(payload.get("health_writer", False))
    writer_mode = str(payload.get("writer_mode") or "")
    source = str(payload.get("source") or "main")
    pid = _safe_int(payload.get("pid"), 0)
    process = _process_snapshot(project_root, source=source, health_pid=pid if pid > 0 else None)
    skipped = _safe_int(payload.get("skipped_file_count", payload.get("skipped_unreadable_count")), 0)
    stale = _safe_int(payload.get("stale_count"), 0)
    pid_known_dead = pid > 0 and process.get("health_pid_alive") is False
    local_mirror_process_count = int(process.get("local_mirror_process_count", 0) or 0)
    pid_rotated_to_helper = bool(pid_known_dead and local_mirror_process_count > 0)
    ok = bool(
        running
        and alive
        and health_writer
        and writer_mode == "local_mirror"
        and age is not None
        and float(age) <= float(freshness_minutes)
        and skipped == 0
        and stale == 0
        and (not pid_known_dead or pid_rotated_to_helper)
    )
    blockers: list[str] = []
    warnings: list[str] = []
    if not running or not alive:
        blockers.append("livefeed_not_running")
    if not health_writer or writer_mode != "local_mirror":
        blockers.append("livefeed_health_writer_not_supervised")
    if age is None or float(age) > float(freshness_minutes):
        warnings.append("livefeed_health_stale")
    if skipped > 0:
        warnings.append("livefeed_skipped_unreadable_files")
    if stale > 0:
        warnings.append("livefeed_stale_sources")
    if pid_known_dead and not pid_rotated_to_helper:
        blockers.append("livefeed_health_pid_not_running")
    if process.get("heavy_process_count", 0) and local_mirror_process_count == 0:
        blockers.append("livefeed_supervised_mirror_missing_while_heavy_active")
    if process.get("guarded_heavy_process_count", 0) and local_mirror_process_count == 0:
        blockers.append("livefeed_supervised_mirror_missing_while_guarded_heavy_active")
    if ok:
        operating_mode = "supervised_local_mirror"
    elif process.get("heavy_process_count", 0) or process.get("guarded_heavy_process_count", 0):
        operating_mode = "operator_heavy_viewer_only"
    else:
        operating_mode = "stopped_or_stale"
    return {
        "ok": ok,
        "path": str(path),
        "age_minutes": age,
        "status": payload.get("status"),
        "alive": alive,
        "health_writer": health_writer,
        "writer_mode": writer_mode,
        "source": source,
        "heavy": payload.get("heavy"),
        "pid": pid or None,
        "pid_known_dead": pid_known_dead,
        "pid_rotated_to_helper": pid_rotated_to_helper,
        "operating_mode": operating_mode,
        "skipped_unreadable_count": skipped,
        "stale_count": stale,
        "process": process,
        "blockers": blockers,
        "warnings": warnings,
    }


def _recommended_actions(blockers: list[str], warnings: list[str]) -> list[str]:
    actions: list[str] = []
    issue_set = set(blockers) | set(warnings)
    if issue_set & {
        "livefeed_not_running",
        "livefeed_health_stale",
        "livefeed_health_pid_not_running",
        "livefeed_supervised_mirror_missing_while_heavy_active",
        "livefeed_supervised_mirror_missing_while_guarded_heavy_active",
        "livefeed_health_writer_not_supervised",
    }:
        actions.append("./scripts/ops/opsctl.sh livefeed-refresh-guard --apply --json")
    if "livefeed_skipped_unreadable_files" in issue_set:
        actions.append("check local livefeed log file permissions, then rerun livefeed-refresh-guard --apply")
    if any(item.startswith("route_failed:") for item in issue_set):
        actions.append("fix the failed opsctl feed-refresh dry-run route before relying on automation")
    return actions


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool = False,
    force_restart: bool = False,
    timeout_sec: int = 180,
    freshness_minutes: float = 10.0,
    out_path: Path = DEFAULT_OUT_PATH,
) -> dict[str, Any]:
    route_checks, route_blockers = _check_routes(project_root, timeout_sec=min(int(timeout_sec), 30))
    apply_result: dict[str, Any] = {
        "ran": False,
        "ok": True,
        "returncode": 0,
        "stdout_tail": "",
        "stderr_tail": "",
    }
    if apply:
        command = [str(project_root / "scripts" / "ops" / "opsctl.sh"), "livefeed-refresh"]
        if force_restart:
            command.append("--force-restart")
        raw = _run_command(command, cwd=project_root, timeout_sec=timeout_sec)
        stdout = str(raw.get("stdout_tail") or "")
        apply_result = {
            "ran": True,
            "ok": bool(
                int(raw.get("returncode", 1)) == 0
                and "livefeed_refresh_completed" in stdout
                and "local_mirror=ready" in stdout
            ),
            **raw,
        }
    health = _health_check(project_root, freshness_minutes=freshness_minutes)
    blockers = list(route_blockers)
    warnings: list[str] = []
    if apply and not bool(apply_result.get("ok", False)):
        blockers.append("livefeed_refresh_apply_failed")
    if not bool(health.get("ok", False)):
        warnings.extend(str(item) for item in health.get("warnings", []))
        blockers.extend(str(item) for item in health.get("blockers", []))
    status = "ready" if not blockers and not warnings else ("advisory" if not blockers else "blocked")
    recommended_actions = _recommended_actions(blockers, warnings)
    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": bool(status == "ready"),
        "overall_status": status,
        "apply": bool(apply),
        "force_restart": bool(force_restart),
        "route_checks": route_checks,
        "route_ok_count": sum(1 for row in route_checks if bool(row.get("ok", False))),
        "route_count": len(route_checks),
        "apply_result": apply_result,
        "health": health,
        "blockers": blockers,
        "warnings": warnings,
        "recommended_actions": recommended_actions,
        "degradation": {
            "active": bool(blockers or warnings),
            "blocker_count": len(blockers),
            "warning_count": len(warnings),
            "operating_mode": health.get("operating_mode"),
            "supervised_local_mirror": bool(health.get("ok", False)),
            "operator_heavy_viewer_count": int(((health.get("process") or {}).get("heavy_process_count", 0)) or 0),
            "guarded_heavy_process_count": int(((health.get("process") or {}).get("guarded_heavy_process_count", 0)) or 0),
        },
        "contract": {
            "validates_all_refresh_routes": True,
            "livefeed_refresh_alias_enforces_all_sources": True,
            "apply_uses_supervised_refresh": True,
            "apply_refreshes_local_livefeed_mirror": True,
            "force_restart_is_explicit_only": True,
            "checks_health_artifact_after_refresh": True,
            "verifies_health_pid_when_present": True,
            "separates_operator_heavy_viewer_from_supervised_mirror": True,
        },
        "next_action": (
            "livefeed refresh routes and health are ready"
            if status == "ready"
            else (recommended_actions[0] if recommended_actions else "resolve listed livefeed blockers/warnings")
        ),
    }
    write_payload(out_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate every livefeed refresh route and optionally perform a supervised refresh.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--force-restart", action="store_true")
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--freshness-minutes", type=float, default=10.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        Path(args.project_root).resolve(),
        apply=bool(args.apply),
        force_restart=bool(args.force_restart),
        timeout_sec=int(args.timeout_sec),
        freshness_minutes=float(args.freshness_minutes),
        out_path=Path(args.out_file).expanduser(),
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "livefeed_refresh_guard "
            f"status={payload.get('overall_status', '')} "
            f"routes={payload.get('route_ok_count', 0)}/{payload.get('route_count', 0)} "
            f"apply={int(bool(payload.get('apply', False)))}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
