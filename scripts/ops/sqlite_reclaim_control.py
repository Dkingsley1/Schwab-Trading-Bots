#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_maintenance import MAINTENANCE_HOLD_TOKEN_ENV
from core.storage_mounts import resolve_external_storage
from scripts import sqlite_performance_maintenance as maintenance
from scripts.ops.storage_maintenance_lane import (
    _coordinate_priority_retention_handoff,
    _release_priority_retention_handoff,
)


def database_space(path: Path) -> dict:
    if maintenance._protected_storage_path(path):
        raise ValueError("protected_volume")
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=2)
    try:
        page_size = conn.execute("PRAGMA page_size").fetchone()[0]
        page_count = conn.execute("PRAGMA page_count").fetchone()[0]
        free_count = conn.execute("PRAGMA freelist_count").fetchone()[0]
    finally:
        conn.close()
    return {
        "physical_gb": path.stat().st_size / 2**30,
        "live_gb": (page_count - free_count) * page_size / 2**30,
        "reclaimable_gb": free_count * page_size / 2**30,
        "reclaimable_ratio": free_count / max(page_count, 1),
    }


def reclaim_blockers(
    space: dict,
    *,
    internal_free_gb: float,
    scratch_free_gb: float,
    same_filesystem: bool,
    memory_ready: bool,
) -> list[str]:
    blockers = []
    if space["reclaimable_gb"] < 2 or space["reclaimable_ratio"] < 0.10:
        blockers.append("below_material_reclaim_threshold")
    if not memory_ready:
        blockers.append("memory_pressure")
    # Keep room for the compacted image and WAL without consuming the hard reserve.
    rewrite_required = 32 + 2.2 * space["live_gb"]
    scratch_required = max(space["physical_gb"] * 1.15, space["physical_gb"] + 8)
    if internal_free_gb < rewrite_required:
        blockers.append("insufficient_database_volume_reserve")
    if same_filesystem:
        scratch_required = max(scratch_required, rewrite_required)
    else:
        scratch_required = max(scratch_required, 125 + space["live_gb"])
    if scratch_free_gb < scratch_required:
        blockers.append("insufficient_scratch_reserve")
    return blockers


def build_payload(project_root: Path, db: Path, scratch: Path, *, apply: bool) -> dict:
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "apply": apply,
        "ok": True,
        "overall_status": "nothing_to_do",
        "database": str(db),
        "scratch_directory": str(scratch),
        "blockers": [],
        "vacuum_ran": False,
    }
    if maintenance._protected_storage_path(scratch):
        raise ValueError("protected_volume")
    space = database_space(db)
    payload["before"] = space
    if space["reclaimable_gb"] < 2 or space["reclaimable_ratio"] < 0.10:
        return payload
    scratch_probe = scratch if scratch.exists() else scratch.parent
    if not scratch_probe.is_dir():
        payload.update(
            overall_status="deferred", blockers=["scratch_volume_unavailable"]
        )
        return payload
    settings = maintenance.resolve_runtime_settings(project_root)
    blockers = reclaim_blockers(
        space,
        internal_free_gb=shutil.disk_usage(db.parent).free / 2**30,
        scratch_free_gb=shutil.disk_usage(scratch_probe).free / 2**30,
        same_filesystem=db.stat().st_dev == scratch_probe.stat().st_dev,
        memory_ready=bool(settings["auto_vacuum_allowed"]),
    )
    payload.update(
        blockers=blockers, overall_status="deferred" if blockers else "ready_to_reclaim"
    )
    if blockers or not apply:
        return payload
    handoff = {}
    try:
        with (project_root / "governance/locks/storage_maintenance.lock").open(
            "a+"
        ) as lane:
            fcntl.flock(lane, fcntl.LOCK_EX | fcntl.LOCK_NB)
            handoff = _coordinate_priority_retention_handoff(
                project_root, enabled=True, poll_seconds=2, wait_timeout_seconds=60
            )
            if not handoff.get("ready"):
                payload.update(
                    overall_status="deferred",
                    blockers=[handoff.get("reason", "writer_busy")],
                )
                return payload
            with (project_root / "governance/locks/jsonl_sql_writer.lock").open(
                "a+"
            ) as writer:
                fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
                space = database_space(db)
                blockers = reclaim_blockers(
                    space,
                    internal_free_gb=shutil.disk_usage(db.parent).free / 2**30,
                    scratch_free_gb=shutil.disk_usage(scratch_probe).free / 2**30,
                    same_filesystem=db.stat().st_dev == scratch_probe.stat().st_dev,
                    memory_ready=bool(
                        maintenance.resolve_runtime_settings(project_root)[
                            "auto_vacuum_allowed"
                        ]
                    ),
                )
                payload.update(before=space, blockers=blockers)
                if blockers:
                    payload["overall_status"] = "deferred"
                    return payload
                env = dict(os.environ)
                env.update(
                    {
                        MAINTENANCE_HOLD_TOKEN_ENV: handoff["token"],
                        "SQLITE_OPTIMIZE_ENABLED": "0",
                    }
                )
                result = subprocess.run(
                    [
                        sys.executable,
                        str(project_root / "scripts/sqlite_performance_maintenance.py"),
                        "--db",
                        str(db),
                        "--vacuum",
                        "--vacuum-temp-dir",
                        str(scratch),
                        "--skip-analyze",
                        "--skip-row-count",
                        "--max-runtime-seconds",
                        "900",
                        "--json",
                    ],
                    cwd=project_root,
                    env=env,
                    capture_output=True,
                    text=True,
                )
                records = []
                for line in result.stdout.splitlines():
                    try:
                        records.append(json.loads(line))
                    except ValueError:
                        pass
                details = records[-1] if records else {}
                payload.update(
                    maintenance=details,
                    returncode=result.returncode,
                    vacuum_ran=bool(details.get("vacuum_ran")),
                    after=database_space(db),
                )
                payload["ok"] = result.returncode == 0 and payload["vacuum_ran"]
                payload["overall_status"] = "reclaimed" if payload["ok"] else "error"
                payload["reclaimed_gb"] = max(
                    space["physical_gb"] - payload["after"]["physical_gb"], 0
                )
                if not payload["ok"]:
                    payload["stderr_tail"] = result.stderr[-2000:]
    except BlockingIOError:
        payload.update(
            overall_status="deferred", blockers=["maintenance_or_writer_lock_busy"]
        )
    finally:
        payload["hold_release"] = _release_priority_retention_handoff(
            project_root, handoff
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reclaim material SQLite free pages under writer ownership and storage reserves."
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    db = maintenance._default_db_path().resolve()
    external = resolve_external_storage()
    scratch = Path(
        os.getenv("SQLITE_VACUUM_TMPDIR") or str(external.external_root / ".sqlite_tmp")
    )
    try:
        payload = build_payload(PROJECT_ROOT, db, scratch, apply=args.apply)
    except Exception as exc:
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "ok": False,
            "overall_status": "error",
            "error": str(exc),
            "vacuum_ran": False,
        }
    out = PROJECT_ROOT / "governance/health/sqlite_reclaim_control_latest.json"
    temporary = out.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(out)
    print(json.dumps(payload))
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
