#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
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
    from core.runtime_python import resolve_runtime_python
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from core.runtime_python import resolve_runtime_python


PY = resolve_runtime_python(PROJECT_ROOT)
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "storage_transition_coordinator_latest.json"
DEFAULT_LOCK_PATH = PROJECT_ROOT / "governance" / "locks" / "storage_transition_coordinator.lock"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _parse_json_output(text: str) -> dict[str, Any]:
    for raw in reversed([line.strip() for line in str(text or "").splitlines() if line.strip()]):
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _run_json_command(cmd: list[str], *, cwd: Path, payload_path: Path | None = None, timeout_sec: int = 120) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
        timeout=max(int(timeout_sec), 1),
    )
    payload = _parse_json_output(proc.stdout or "")
    if not payload and payload_path is not None:
        payload = _load_json(payload_path)
    duration_ms = round((datetime.now(timezone.utc) - started).total_seconds() * 1000.0, 3)
    return {
        "cmd": list(cmd),
        "rc": int(proc.returncode),
        "duration_ms": duration_ms,
        "payload": payload,
        "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-12:]),
        "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-12:]),
    }


def _artifact_status(name: str, payload: dict[str, Any]) -> str:
    if not payload:
        return "missing"
    if name == "storage_split_brain_reconciler":
        unresolved = int((((payload.get("summary") or {}).get("unresolved_conflicts", 0)) or 0))
        return "ready" if unresolved == 0 else "needs_review"
    status = str(payload.get("overall_status") or "").strip()
    if status:
        return status
    if "ok" in payload:
        return "ready" if bool(payload.get("ok", False)) else "needs_work"
    return "ready"


def _artifact_ok(name: str, payload: dict[str, Any]) -> bool:
    if not payload:
        return False
    if name == "storage_split_brain_reconciler":
        return int((((payload.get("summary") or {}).get("unresolved_conflicts", 0)) or 0)) == 0
    if "ok" in payload:
        return bool(payload.get("ok", False))
    status = str(payload.get("overall_status") or "").strip()
    return status not in {"blocked", "error", "needs_work", "needs_review"}


def _artifact_timestamp(payload: dict[str, Any]) -> str:
    return str(payload.get("timestamp_utc") or payload.get("generated_utc") or "")


def _ordered_unique(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        value = str(raw or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _assistant_specs(project_root: Path, transition_mode: str) -> list[dict[str, Any]]:
    specs = [
        {
            "name": "storage_split_brain_reconciler",
            "responsibility": "confirm there are no unresolved local-vs-external conflict copies before and after storage handoff",
            "critical": True,
            "health_path": str(project_root / "governance" / "health" / "storage_split_brain_reconciler_latest.json"),
            "refresh_cmd": [
                str(PY),
                str(project_root / "scripts" / "ops" / "storage_split_brain_reconciler.py"),
                "--json",
            ],
            "refresh_timeout_sec": 120,
        },
        {
            "name": "storage_resilience_control",
            "responsibility": "score dual-root readiness and restore-drill freshness for the active storage route",
            "critical": True,
            "health_path": str(project_root / "governance" / "health" / "storage_resilience_control_latest.json"),
            "refresh_cmd": [
                str(PY),
                str(project_root / "scripts" / "ops" / "storage_resilience_control.py"),
                "--json",
            ],
            "refresh_timeout_sec": 120,
        },
    ]
    if transition_mode == "local":
        specs.extend(
            [
                {
                    "name": "storage_quota_guard",
                    "responsibility": "watch local SSD quota pressure while BOT_LOGS is routed to local_fallback_storage",
                    "critical": False,
                    "health_path": str(project_root / "governance" / "health" / "storage_quota_guard_latest.json"),
                    "refresh_cmd": [
                        str(PY),
                        str(project_root / "scripts" / "ops" / "storage_quota_guard.py"),
                        "--json",
                    ],
                    "refresh_timeout_sec": 120,
                },
                {
                    "name": "storage_backpressure_autopilot",
                    "responsibility": "preview the backlog and retention-shaping bots that should keep internal-drive fallback stable",
                    "critical": False,
                    "health_path": str(project_root / "governance" / "health" / "storage_backpressure_autopilot_latest.json"),
                    "refresh_cmd": [
                        str(PY),
                        str(project_root / "scripts" / "ops" / "storage_backpressure_autopilot.py"),
                        "--json",
                    ],
                    "refresh_timeout_sec": 120,
                },
            ]
        )
    else:
        specs.append(
            {
                "name": "ops_coordinator",
                "responsibility": "refresh top-level ops health after BOT_LOGS is restored to the external route",
                "critical": False,
                "health_path": str(project_root / "governance" / "health" / "ops_coordinator_latest.json"),
                "refresh_cmd": [
                    str(PY),
                    str(project_root / "scripts" / "ops" / "ops_coordinator.py"),
                    "--json",
                ],
                "refresh_timeout_sec": 180,
            }
        )
    return specs


def _load_assigned_bots(project_root: Path, *, transition_mode: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in _assistant_specs(project_root, transition_mode):
        payload = _load_json(Path(spec["health_path"]))
        rows.append(
            {
                "name": spec["name"],
                "responsibility": spec["responsibility"],
                "critical": bool(spec.get("critical", False)),
                "health_path": spec["health_path"],
                "status": _artifact_status(spec["name"], payload),
                "ok": _artifact_ok(spec["name"], payload),
                "timestamp_utc": _artifact_timestamp(payload),
                "recommended_actions": list(payload.get("recommended_actions") or payload.get("top_actions") or []),
                "summary": payload.get("summary") if isinstance(payload.get("summary"), dict) else {},
            }
        )
    return rows


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    transition_mode: str,
    apply: bool = False,
) -> dict[str, Any]:
    transition_mode = str(transition_mode or "local").strip().lower()
    if transition_mode not in {"local", "external"}:
        raise ValueError(f"unsupported transition mode: {transition_mode}")

    mount_guard = _load_json(project_root / "governance" / "health" / "storage_mount_guard_latest.json")
    failback = _load_json(project_root / "governance" / "health" / "storage_failback_sync_latest.json")
    assigned_bots = _load_assigned_bots(project_root, transition_mode=transition_mode)
    recommended_actions = _ordered_unique(
        [
            *(
                str(item)
                for row in assigned_bots
                for item in list(row.get("recommended_actions") or [])
            ),
            "keep BOT_LOGS on local_fallback_storage only long enough to cover the external-drive maintenance window" if transition_mode == "local" else "",
            "reconcile split-brain state before pruning fallback copies after external storage returns" if transition_mode == "external" else "",
        ]
    )

    overall_status = "ready"
    if any(bool(row.get("critical", False)) and str(row.get("status") or "") in {"missing", "blocked", "error"} for row in assigned_bots):
        overall_status = "blocked"
    elif any(str(row.get("status") or "") not in {"ready", "ok"} for row in assigned_bots):
        overall_status = "degraded"

    payload = {
        "timestamp_utc": _utc_now(),
        "schema_version": 1,
        "project_root": str(project_root),
        "transition_mode": transition_mode,
        "apply_requested": bool(apply),
        "ok": overall_status != "blocked",
        "overall_status": overall_status,
        "current_storage_mode": str(mount_guard.get("storage_mode") or failback.get("mode") or ""),
        "mount_guard": {
            "external_available": bool(mount_guard.get("external_available", False)),
            "mount_present": bool(mount_guard.get("mount_present", False)),
            "storage_mode": str(mount_guard.get("storage_mode") or ""),
            "external_root": str(mount_guard.get("external_root") or ""),
        },
        "assigned_bots": assigned_bots,
        "recommended_actions": recommended_actions,
        "refresh_plan": [
            {
                "name": spec["name"],
                "cmd": list(spec["refresh_cmd"]),
                "timeout_sec": int(spec["refresh_timeout_sec"]),
            }
            for spec in _assistant_specs(project_root, transition_mode)
        ],
        "metrics": {
            "assigned_bot_count": len(assigned_bots),
            "ready_bot_count": sum(1 for row in assigned_bots if bool(row.get("ok", False))),
        },
    }
    return payload


def _apply_refresh(project_root: Path, *, transition_mode: str) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    watchdog_cmd = [
        str(PY),
        str(project_root / "scripts" / "ops" / "process_watchdog.py"),
        "--json",
    ]
    watchdog_result = _run_json_command(
        watchdog_cmd,
        cwd=project_root,
        payload_path=project_root / "governance" / "health" / "process_watchdog_latest.json",
        timeout_sec=120,
    )
    attempts.append(
        {
            "name": "process_watchdog",
            "rc": int(watchdog_result.get("rc", 1)),
            "duration_ms": float(watchdog_result.get("duration_ms", 0.0) or 0.0),
            "stdout_tail": str(watchdog_result.get("stdout_tail") or ""),
            "stderr_tail": str(watchdog_result.get("stderr_tail") or ""),
        }
    )
    for spec in _assistant_specs(project_root, transition_mode):
        result = _run_json_command(
            list(spec["refresh_cmd"]),
            cwd=project_root,
            payload_path=Path(spec["health_path"]),
            timeout_sec=int(spec["refresh_timeout_sec"]),
        )
        attempts.append(
            {
                "name": spec["name"],
                "rc": int(result.get("rc", 1)),
                "duration_ms": float(result.get("duration_ms", 0.0) or 0.0),
                "stdout_tail": str(result.get("stdout_tail") or ""),
                "stderr_tail": str(result.get("stderr_tail") or ""),
            }
        )
    return attempts


def main() -> int:
    parser = argparse.ArgumentParser(description="Assign and refresh infrastructure bots around a BOT_LOGS storage transition.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--transition-mode", choices=("local", "external"), required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--lock-file", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_file = Path(args.out_file).expanduser()
    lock_file = Path(args.lock_file).expanduser()
    lock_file.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "timestamp_utc": _utc_now(),
        "schema_version": 1,
        "project_root": str(project_root),
        "transition_mode": str(args.transition_mode),
        "ok": True,
        "overall_status": "pending",
    }

    with lock_file.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            payload.update({"busy": True, "overall_status": "already_running"})
            _write_json(out_file, payload)
            if args.json:
                print(json.dumps(payload, ensure_ascii=True))
            else:
                print("storage_transition_coordinator overall_status=already_running")
            return 0

        if not bool(args.apply):
            payload = build_payload(project_root, transition_mode=str(args.transition_mode), apply=False)
            _write_json(out_file, payload)
        else:
            running_payload = build_payload(project_root, transition_mode=str(args.transition_mode), apply=True)
            running_payload.update({"busy": True, "overall_status": "running"})
            _write_json(out_file, running_payload)

            attempts = _apply_refresh(project_root, transition_mode=str(args.transition_mode))
            payload = build_payload(project_root, transition_mode=str(args.transition_mode), apply=True)
            payload["attempts"] = attempts
            payload["metrics"]["attempted_step_count"] = len(attempts)
            if any(int(row.get("rc", 1)) != 0 for row in attempts):
                payload["ok"] = False
                payload["overall_status"] = "degraded" if str(payload.get("overall_status") or "") == "ready" else payload.get("overall_status")
            _write_json(out_file, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "storage_transition_coordinator "
            f"transition_mode={payload.get('transition_mode', '')} "
            f"overall_status={payload.get('overall_status', '')} "
            f"assigned_bot_count={int(((payload.get('metrics') or {}).get('assigned_bot_count', 0) or 0))}"
        )
    return 0 if str(payload.get("overall_status") or "") in {"ready", "degraded", "already_running"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
