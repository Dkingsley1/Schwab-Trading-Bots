#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "sleeve_isolation_guard_latest.json"
NON_RUNTIME_DAILY_VERIFY_CHECKS = {
    "promotion_quality_gate",
    "retrain_schema_compatibility_guard",
}


def _ingress_rows(project_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((project_root / "governance" / "health").glob("data_ingress_latest_*.json")):
        payload = load_json(path)
        if not payload:
            continue
        rows.append(
            {
                "artifact": path.name,
                "profile": str(payload.get("profile") or ""),
                "domain": str(payload.get("domain") or ""),
                "broker": str(payload.get("broker") or ""),
                "loop_state": str(payload.get("loop_state") or ""),
                "pause_reason": str(payload.get("pause_reason") or ""),
                "iter_error_rate": float(payload.get("iter_error_rate", 0.0) or 0.0),
            }
        )
    return rows


def build_payload(project_root: Path = PROJECT_ROOT, *, max_quarantine_events: int = 120) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    quarantine_pressure = load_json(health_root / "quarantine_pressure_latest.json")
    daily_verify = load_json(health_root / "daily_auto_verify_latest.json")
    lane_thaw = load_json(health_root / "lane_thaw_controller_latest.json")
    ingress_rows = _ingress_rows(project_root)

    isolated_lanes = [
        row
        for row in ingress_rows
        if "paused" in str(row.get("loop_state") or "") or "killswitch" in str(row.get("loop_state") or "")
    ]
    running_lanes = [row for row in ingress_rows if str(row.get("loop_state") or "") == "running"]
    quarantine_events = int(quarantine_pressure.get("quarantine_events", 0) or 0)
    raw_failed_checks = daily_verify.get("failed_checks") if isinstance(daily_verify.get("failed_checks"), list) else []
    unresolved_checks = [
        name
        for name in [str(item or "").strip() for item in raw_failed_checks]
        if name and name not in NON_RUNTIME_DAILY_VERIFY_CHECKS
    ]

    overall_status = "ready"
    if len(isolated_lanes) >= 2 or quarantine_events > int(max_quarantine_events):
        overall_status = "blocked"
    elif isolated_lanes or quarantine_events > 0:
        overall_status = "degraded"
    isolated_lane_count = len(isolated_lanes)
    running_lane_count = len(running_lanes)
    total_lane_count = max(isolated_lane_count + running_lane_count, 1)
    blast_radius_score = round((running_lane_count / total_lane_count) * 100.0, 2)
    thaw_candidates = lane_thaw.get("candidates") if isinstance(lane_thaw.get("candidates"), list) else []
    thaw_blocked = lane_thaw.get("blocked") if isinstance(lane_thaw.get("blocked"), list) else []
    repeatable_thaw_ready = bool(
        isolated_lane_count > 0
        and len(thaw_candidates) > 0
        and not unresolved_checks
        and all(str(row.get("decision") or "").strip().lower() in {"allow", "ready"} for row in thaw_candidates)
    )

    recommended_actions = ordered_unique(
        [
            "keep healthy sleeves running while anomaly-killed lanes stay quarantined" if isolated_lanes else "",
            "route investigation to the paused sleeves instead of draining the entire runtime" if len(isolated_lanes) >= 1 else "",
            "reduce quarantine churn before expanding sleeve count again" if quarantine_events > int(max_quarantine_events) else "",
            "clear unresolved daily verify blockers before reenabling isolated sleeves" if unresolved_checks else "",
            "only thaw isolated sleeves through the repeatable thaw contract once feed, broker, and cooldown checks are green" if isolated_lanes else "",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "blast_radius_score": blast_radius_score,
        "quarantine_pressure": {
            "events": quarantine_events,
            "max_quarantine_events": int(max_quarantine_events),
            "top_symbols": quarantine_pressure.get("top_symbols") if isinstance(quarantine_pressure.get("top_symbols"), list) else [],
        },
        "sleeve_matrix": {
            "isolated_lanes": isolated_lanes,
            "isolated_lane_count": isolated_lane_count,
            "running_lane_count": running_lane_count,
            "running_examples": running_lanes[:6],
        },
        "gates": {
            "unresolved_daily_verify_checks": unresolved_checks,
            "isolation_required": bool(isolated_lanes),
        },
        "repeatable_thaw_contract": {
            "ready": repeatable_thaw_ready,
            "candidate_count": len(thaw_candidates),
            "blocked_count": len(thaw_blocked),
            "candidate_examples": thaw_candidates[:4],
            "blocked_examples": thaw_blocked[:4],
        },
        "infra_bots": ["sleeve_isolation_guard", "quarantine_pressure_bot", "data_ingress_latest_*"],
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Track sleeve quarantine and isolation so one failing lane does not poison the runtime.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--max-quarantine-events", type=int, default=120)
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve(), max_quarantine_events=int(args.max_quarantine_events))
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "sleeve_isolation_guard "
            f"overall_status={payload.get('overall_status', '')} "
            f"isolated_lane_count={int(((payload.get('sleeve_matrix') or {}).get('isolated_lane_count', 0) or 0))}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
