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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_python import resolve_runtime_python


PY = resolve_runtime_python(PROJECT_ROOT)
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "stale_artifact_sweeper_bot_latest.json"
DEFAULT_LOCK_PATH = PROJECT_ROOT / "governance" / "locks" / "stale_artifact_sweeper_bot.lock"
DEFAULT_STALE_STAGE_ROOT = PROJECT_ROOT / "data" / "stale_stage"


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
    for line in reversed([raw.strip() for raw in str(text or "").splitlines() if raw.strip()]):
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _run_json_command(cmd: list[str], *, cwd: Path, payload_path: Path) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    payload = _parse_json_output(proc.stdout or "")
    if not payload:
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


def build_payload(
    project_root: Path,
    *,
    stale_stage_sections: str,
    stale_stage_root: Path,
    stale_stage_manifest: str,
) -> dict[str, Any]:
    cmd = [
        str(PY),
        str(project_root / "scripts" / "data_retention_policy.py"),
        "--apply",
        "--skip-sqlite-vacuum",
        "--no-archive-prune",
        "--stale-stage",
        "--stale-stage-only",
        "--no-stale-purge",
        "--stale-stage-sections",
        str(stale_stage_sections or "all"),
        "--stale-stage-root",
        str(stale_stage_root),
        "--json",
    ]
    if str(stale_stage_manifest or "").strip():
        cmd.extend(["--stale-stage-manifest", str(stale_stage_manifest).strip()])
    result = _run_json_command(
        cmd,
        cwd=project_root,
        payload_path=project_root / "governance" / "health" / "data_retention_latest.json",
    )
    retention_payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
    stale_stage = retention_payload.get("stale_stage") if isinstance(retention_payload.get("stale_stage"), dict) else {}
    staged_by_label = stale_stage.get("staged_by_label") if isinstance(stale_stage.get("staged_by_label"), dict) else {}
    top_labels = sorted(
        (
            {
                "label": str(label),
                "staged_files": int((row or {}).get("staged_files", 0) or 0),
                "staged_bytes": int((row or {}).get("staged_bytes", 0) or 0),
                "candidate_files": int((row or {}).get("candidate_files", 0) or 0),
            }
            for label, row in staged_by_label.items()
        ),
        key=lambda item: (-int(item["staged_files"]), -int(item["staged_bytes"]), str(item["label"])),
    )
    busy = bool(retention_payload.get("busy", False))
    ok = int(result.get("rc", 1)) == 0 and not busy and int(stale_stage.get("delete_errors", 0) or 0) == 0
    reason = "ok"
    if busy:
        reason = str(retention_payload.get("skipped_reason") or "lock_busy")
    elif int(result.get("rc", 1)) != 0:
        reason = "data_retention_failed"
    elif int(stale_stage.get("delete_errors", 0) or 0) > 0:
        reason = "stale_stage_move_errors"

    return {
        "timestamp_utc": _utc_now(),
        "project_root": str(project_root),
        "ok": bool(ok),
        "busy": busy,
        "reason": reason,
        "cmd": list(result.get("cmd") or []),
        "duration_ms": float(result.get("duration_ms", 0.0) or 0.0),
        "summary": {
            "candidate_files": int(stale_stage.get("candidate_files", 0) or 0),
            "candidate_bytes": int(stale_stage.get("candidate_bytes", 0) or 0),
            "staged_files": int(stale_stage.get("staged_files", 0) or 0),
            "staged_bytes": int(stale_stage.get("staged_bytes", 0) or 0),
            "delete_errors": int(stale_stage.get("delete_errors", 0) or 0),
            "stage_sections": list(stale_stage.get("sections") or []),
        },
        "stale_stage": stale_stage,
        "top_labels": top_labels[:12],
        "artifacts": {
            "data_retention": str(project_root / "governance" / "health" / "data_retention_latest.json"),
            "stale_manifest": str(stale_stage.get("manifest_path") or ""),
            "stale_root": str(stale_stage.get("root") or ""),
        },
        "stdout_tail": str(result.get("stdout_tail") or ""),
        "stderr_tail": str(result.get("stderr_tail") or ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage stale retention candidates into the stale_stage holding area without hard deleting them.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--stale-stage-sections", default="all")
    parser.add_argument("--stale-stage-root", default=str(DEFAULT_STALE_STAGE_ROOT))
    parser.add_argument("--stale-stage-manifest", default="")
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
        "project_root": str(project_root),
        "ok": True,
        "busy": False,
        "reason": "pending",
    }

    with lock_file.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            payload.update({"busy": True, "reason": "already_running"})
            _write_json(out_file, payload)
            if args.json:
                print(json.dumps(payload, ensure_ascii=True))
            else:
                print("stale_artifact_sweeper_bot busy=1 reason=already_running")
            return 0

        payload = build_payload(
            project_root,
            stale_stage_sections=str(args.stale_stage_sections or "all"),
            stale_stage_root=Path(args.stale_stage_root).expanduser(),
            stale_stage_manifest=str(args.stale_stage_manifest or ""),
        )
        _write_json(out_file, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        print(
            "stale_artifact_sweeper_bot "
            f"ok={int(bool(payload.get('ok', False)))} "
            f"staged_files={int(summary.get('staged_files', 0) or 0)} "
            f"candidate_files={int(summary.get('candidate_files', 0) or 0)}"
        )
    return 0 if bool(payload.get("ok", False) or payload.get("busy", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
