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
    from scripts.ops.long_runtime_common import iso_now, write_payload
    from scripts.ops import infrabot_gap_roster, system_cleanliness_autopilot
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from .long_runtime_common import iso_now, write_payload
    from . import infrabot_gap_roster, system_cleanliness_autopilot


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "system_cleanliness_infrabot_latest.json"


def build_payload(project_root: Path = PROJECT_ROOT, *, apply: bool = False, timeout_sec: int = 300) -> dict[str, Any]:
    autopilot = system_cleanliness_autopilot.build_payload(project_root, apply=apply, timeout_sec=timeout_sec)
    gap_roster = infrabot_gap_roster.build_payload(project_root, apply=False, timeout_sec=timeout_sec)
    layer_statuses = autopilot.get("layer_statuses") if isinstance(autopilot.get("layer_statuses"), dict) else {}
    blocked = [name for name, status in layer_statuses.items() if str(status) == "blocked"]
    degraded = [name for name, status in layer_statuses.items() if str(status) == "degraded"]
    overall_status = "ready"
    if blocked:
        overall_status = "blocked"
    elif degraded or int(autopilot.get("applyable_repair_count", 0) or 0) > 0:
        overall_status = "degraded"
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "apply": bool(apply),
        "assigned_scope": [
            "storage_backpressure",
            "collectors_sources",
            "training_eligibility",
            "paper_feedback",
            "promotion_replay",
            "infrabot_gap_roster",
        ],
        "autopilot_artifact": str(project_root / "governance" / "health" / "system_cleanliness_autopilot_latest.json"),
        "autopilot_status": autopilot.get("overall_status"),
        "gap_roster_artifact": str(project_root / "governance" / "health" / "infrabot_gap_roster_latest.json"),
        "gap_roster_status": gap_roster.get("overall_status"),
        "gap_roster_active_count": int(gap_roster.get("active_count", 0) or 0),
        "gap_infrabots": gap_roster.get("assigned_infrabots") if isinstance(gap_roster.get("assigned_infrabots"), list) else [],
        "active_gap_infrabots": gap_roster.get("active_infrabots") if isinstance(gap_roster.get("active_infrabots"), list) else [],
        "layer_statuses": layer_statuses,
        "blocked_layers": blocked,
        "degraded_layers": degraded,
        "repair_count": int(autopilot.get("applyable_repair_count", 0) or 0),
        "operator_followups": autopilot.get("operator_followups") if isinstance(autopilot.get("operator_followups"), list) else [],
        "supervision_contract": {
            "owner_bot": "system_cleanliness_infrabot",
            "must_not_start_broad_retrain_when_blocked": True,
            "must_keep_new_sleeves_collect_only_until_ready": True,
            "safe_apply_supported": True,
            "destructive_actions_operator_gated": True,
            "delegated_infrabots": gap_roster.get("assigned_infrabots") if isinstance(gap_roster.get("assigned_infrabots"), list) else [],
            "live_execution_authority": False,
            "protected_volume_denylist": ["/Volumes/VIDEO"],
        },
        "recommended_actions": [
            *(autopilot.get("recommended_actions") if isinstance(autopilot.get("recommended_actions"), list) else []),
            *(gap_roster.get("recommended_actions") if isinstance(gap_roster.get("recommended_actions"), list) else []),
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Infrastructure bot supervisor for the system cleanliness autopilot.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    payload = build_payload(project_root, apply=bool(args.apply), timeout_sec=int(args.timeout_sec))
    write_payload(Path(args.out_file), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "system_cleanliness_infrabot "
            f"overall_status={payload.get('overall_status')} "
            f"repairs={payload.get('repair_count')}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
