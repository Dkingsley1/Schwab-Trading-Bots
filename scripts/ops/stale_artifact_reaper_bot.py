#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import data_retention_policy as retention


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "stale_artifact_reaper_bot_latest.json"
DEFAULT_LOCK_PATH = PROJECT_ROOT / "governance" / "locks" / "stale_artifact_reaper_bot.lock"
DEFAULT_STALE_STAGE_ROOT = PROJECT_ROOT / "data" / "stale_stage"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def build_payload(
    project_root: Path,
    *,
    stale_stage_root: Path,
    stale_stage_manifest: str,
    stale_purge_days: int,
) -> dict[str, Any]:
    manifest_path = retention._stale_manifest_path(stale_stage_root, stale_stage_manifest)
    purge = retention._purge_old_stale_stage(
        stale_root=stale_stage_root,
        manifest_path=manifest_path,
        older_than_days=int(stale_purge_days),
    )
    ok = int(purge.get("delete_errors", 0) or 0) == 0
    return {
        "timestamp_utc": _utc_now(),
        "project_root": str(project_root),
        "ok": bool(ok),
        "busy": False,
        "reason": ("ok" if ok else "purge_errors"),
        "summary": {
            "candidate_files": int(purge.get("candidate_files", 0) or 0),
            "candidate_bytes": int(purge.get("candidate_bytes", 0) or 0),
            "deleted_files": int(purge.get("deleted_files", 0) or 0),
            "deleted_bytes": int(purge.get("deleted_bytes", 0) or 0),
            "delete_errors": int(purge.get("delete_errors", 0) or 0),
            "older_than_days": int(purge.get("older_than_days", stale_purge_days) or stale_purge_days),
            "manifest_lines_after": int(((purge.get("manifest_compaction") or {}).get("lines_after", 0) or 0)),
        },
        "purge": purge,
        "artifacts": {
            "stale_root": str(stale_stage_root),
            "stale_manifest": str(manifest_path),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Purge aged files that are already sitting inside stale_stage.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--stale-stage-root", default=str(DEFAULT_STALE_STAGE_ROOT))
    parser.add_argument("--stale-stage-manifest", default="")
    parser.add_argument("--stale-purge-days", type=int, default=int(os.getenv("RETENTION_STALE_PURGE_DAYS", "30")))
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
                print("stale_artifact_reaper_bot busy=1 reason=already_running")
            return 0

        payload = build_payload(
            project_root,
            stale_stage_root=Path(args.stale_stage_root).expanduser(),
            stale_stage_manifest=str(args.stale_stage_manifest or ""),
            stale_purge_days=int(args.stale_purge_days),
        )
        _write_json(out_file, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        print(
            "stale_artifact_reaper_bot "
            f"ok={int(bool(payload.get('ok', False)))} "
            f"deleted_files={int(summary.get('deleted_files', 0) or 0)} "
            f"candidate_files={int(summary.get('candidate_files', 0) or 0)}"
        )
    return 0 if bool(payload.get("ok", False) or payload.get("busy", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
