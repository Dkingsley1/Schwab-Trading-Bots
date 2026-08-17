#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import storage_router
from scripts import ops_data_plane


DEFAULT_OUT = PROJECT_ROOT / "governance" / "health" / "storage_route_chaos_drill_latest.json"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def build_payload(project_root: Path = PROJECT_ROOT, *, scenario: str = "external_unavailable") -> dict[str, Any]:
    root = Path(project_root).resolve()
    external_root = storage_router._external_project_root()
    local_root = Path(
        os.getenv(
            "BOT_LOGS_LOCAL_FALLBACK_ROOT",
            str(root / storage_router.DEFAULT_LOCAL_FALLBACK),
        )
    ).expanduser()
    split_brain = _load_json(root / "governance" / "health" / "storage_split_brain_reconciler_latest.json")
    mount_guard = _load_json(root / "governance" / "health" / "storage_mount_guard_latest.json")
    route_status = _load_json(root / "governance" / "health" / "storage_route_status_latest.json")
    ops_db_path = ops_data_plane.resolve_db_path(root)
    checks = {
        "local_root_writable": bool(storage_router._is_writable_directory(local_root)),
        "ops_control_writable": bool(storage_router._is_writable_directory(ops_db_path.parent)),
        "split_brain_clear": int(((split_brain.get("summary") or {}).get("unresolved_conflicts", 0) or 0)) == 0,
        "external_root_known": bool(str(external_root)),
        "latest_route_status_present": bool(route_status),
        "mount_guard_present": bool(mount_guard),
    }
    ok = all(checks.values())
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": ok,
        "scenario": str(scenario or "external_unavailable"),
        "project_root": str(root),
        "local_root": str(local_root),
        "external_root": str(external_root),
        "ops_db_path": str(ops_db_path),
        "checks": checks,
        "top_actions": [
            "keep local_fallback_storage writable so the external_unavailable scenario can cut over cleanly",
            "treat storage_split_brain_reconciler_latest.json as a precondition for failback drills",
            "keep storage_route_events flowing into ops_data_plane.sqlite3 so drill history is auditable",
        ],
    }
    with ops_data_plane.connect(root) as conn:
        ops_data_plane.record_storage_route_event(
            conn,
            project_root=root,
            mode="chaos_drill",
            active_root=local_root if scenario == "external_unavailable" else external_root,
            switched_links=[],
            passthrough_paths=[],
            split_brain_conflicts=int(not checks["split_brain_clear"]),
            metadata={
                "chaos_drill": True,
                "scenario": str(scenario or "external_unavailable"),
                "checks": checks,
            },
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Record storage-route chaos-drill readiness for fallback scenarios.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--scenario", default="external_unavailable")
    parser.add_argument("--out-file", default=str(DEFAULT_OUT))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve(), scenario=str(args.scenario or "external_unavailable"))
    out_path = Path(args.out_file).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "storage_route_chaos_drill "
            f"scenario={payload.get('scenario', '')} ok={str(bool(payload.get('ok', False))).lower()}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
