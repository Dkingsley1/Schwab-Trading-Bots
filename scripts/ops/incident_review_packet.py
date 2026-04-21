#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "incident_review_packet_latest.json"


def _stable_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")).hexdigest()


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    timeline_path = health_root / "incident_timeline_latest.json"
    runtime_path = health_root / "live_runtime_separation_control_latest.json"
    auth_path = health_root / "auth_lease_manager_latest.json"
    alerts_path = health_root / "remote_alert_control_latest.json"
    thaw_path = health_root / "lane_thaw_controller_latest.json"
    data_plane_path = health_root / "data_plane_recovery_controller_latest.json"

    timeline = load_json(timeline_path)
    runtime = load_json(runtime_path)
    auth = load_json(auth_path)
    alerts = load_json(alerts_path)
    thaw = load_json(thaw_path)
    data_plane = load_json(data_plane_path)

    source_snapshot = {
        "timeline": {
            "overall_status": str(timeline.get("overall_status") or ""),
            "recent_incident_count": int(timeline.get("recent_incident_count", 0) or 0),
            "open_incident_count": int(timeline.get("open_incident_count", 0) or 0),
            "open_surfaces": timeline.get("open_surfaces") if isinstance(timeline.get("open_surfaces"), list) else [],
            "recent_incidents": timeline.get("recent_incidents") if isinstance(timeline.get("recent_incidents"), list) else [],
        },
        "runtime": {
            "overall_status": str(runtime.get("overall_status") or ""),
            "clearance_state": str(((runtime.get("clearance_plan") or {}).get("clearance_state") or "")),
        },
        "auth": {
            "overall_status": str(auth.get("overall_status") or ""),
            "lease_state": str(auth.get("lease_state") or ""),
        },
        "alerts": {
            "overall_status": str(alerts.get("overall_status") or ""),
            "critical_backlog": alerts.get("critical_backlog") if isinstance(alerts.get("critical_backlog"), dict) else {},
        },
        "lane_thaw": {
            "overall_status": str(thaw.get("overall_status") or ""),
            "paused_lane_count": int(thaw.get("paused_lane_count", 0) or 0),
            "candidate_count": int(thaw.get("candidate_count", 0) or 0),
        },
        "data_plane": {
            "overall_status": str(data_plane.get("overall_status") or ""),
            "write_failure_count": int(data_plane.get("write_failure_count", 0) or 0),
            "account_snapshot_failure_count": int(data_plane.get("account_snapshot_failure_count", 0) or 0),
        },
    }
    packet_sha256 = _stable_hash(source_snapshot)
    open_incident_count = int(((source_snapshot.get("timeline") or {}).get("open_incident_count", 0) or 0))
    recent_incidents = ((source_snapshot.get("timeline") or {}).get("recent_incidents") or [])
    recent_categories = sorted(
        {
            str((row or {}).get("category") or "").strip().lower()
            for row in recent_incidents
            if isinstance(row, dict) and str((row or {}).get("category") or "").strip()
        }
    )
    overall_status = str(timeline.get("overall_status") or "ready")
    review_required = open_incident_count > 0 or overall_status in {"blocked", "degraded"}
    auto_close = ((timeline.get("auto_close_contract") or {}) if isinstance(timeline.get("auto_close_contract"), dict) else {})
    closure_contract = {
        "closure_ready": bool(auto_close.get("closure_ready", False)) and not review_required,
        "candidate_count": int(auto_close.get("candidate_count", 0) or 0),
        "review_required": review_required,
        "closure_reason": str(auto_close.get("closure_reason") or ""),
    }

    recommended_actions = ordered_unique(
        list(timeline.get("recommended_actions") or [])[:2]
        + [
            "treat this packet hash as the immutable incident-review anchor when you discuss interventions or approvals" if review_required else "",
            "close the runtime and auth blockers before archiving the incident packet" if overall_status == "blocked" else "",
            "archive the packet hash and mark the incident closed once the auto-close contract stays green" if bool(closure_contract.get("closure_ready", False)) else "",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "review_required": review_required,
        "review_state": ("awaiting_remediation" if review_required else "ready_to_archive"),
        "open_incident_count": open_incident_count,
        "recent_categories": recent_categories,
        "packet_sha256": packet_sha256,
        "closure_contract": closure_contract,
        "immutability_contract": {
            "hash_algorithm": "sha256",
            "source_paths": [str(timeline_path), str(runtime_path), str(auth_path), str(alerts_path), str(thaw_path), str(data_plane_path)],
            "source_snapshot_bytes": len(json.dumps(source_snapshot, ensure_ascii=True).encode("utf-8")),
        },
        "source_snapshot": source_snapshot,
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish an immutable review packet for the current incident timeline.")
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
            "incident_review_packet "
            f"overall_status={payload.get('overall_status', '')} "
            f"review_required={int(bool(payload.get('review_required', False)))}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
