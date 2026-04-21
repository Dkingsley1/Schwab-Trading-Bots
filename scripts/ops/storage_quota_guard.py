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
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, ordered_unique, load_json, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, ordered_unique, load_json, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "storage_quota_guard_latest.json"


DEFAULT_QUOTAS_GB = {
    "sql_link_shards": {"soft": 320.0, "hard": 380.0},
    "decision_explanations": {"soft": 24.0, "hard": 48.0},
    "decisions": {"soft": 20.0, "hard": 36.0},
    "governance_telemetry": {"soft": 8.0, "hard": 12.0},
    "artifact_store": {"soft": 4.0, "hard": 10.0},
}

FAMILY_TO_ROLE = {
    "sql_link_shards": "stateful_sql",
    "decision_explanations": "explainability",
    "decisions": "live_decisioning",
    "governance": "governance_telemetry",
    "governance_events": "governance_telemetry",
    "governance_channels": "governance_telemetry",
    "content_store": "artifact_store",
}


def _role_bytes(storage_tier: dict[str, Any], role: str) -> int:
    by_role = storage_tier.get("by_service_role") if isinstance(storage_tier.get("by_service_role"), dict) else {}
    return int(((by_role.get(role) or {}).get("bytes", 0)) or 0)


def _family_bytes(storage_tier: dict[str, Any], family: str) -> int:
    by_family = storage_tier.get("by_family") if isinstance(storage_tier.get("by_family"), dict) else {}
    return int(((by_family.get(family) or {}).get("bytes", 0)) or 0)


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    storage_tier = load_json(project_root / "governance" / "health" / "storage_tier_policy_latest.json")
    lanes: list[dict[str, Any]] = []
    hard_breaches = 0
    soft_breaches = 0
    for family, quota in DEFAULT_QUOTAS_GB.items():
        bytes_used = _family_bytes(storage_tier, family)
        if bytes_used == 0:
            bytes_used = _role_bytes(storage_tier, FAMILY_TO_ROLE.get(family, family))
        used_gb = float(bytes_used) / float(1024**3)
        soft_gb = float(quota["soft"])
        hard_gb = float(quota["hard"])
        status = "ready"
        if used_gb >= hard_gb:
            status = "blocked"
            hard_breaches += 1
        elif used_gb >= soft_gb:
            status = "degraded"
            soft_breaches += 1
        lanes.append(
            {
                "family": family,
                "used_gb": round(used_gb, 3),
                "soft_quota_gb": soft_gb,
                "hard_quota_gb": hard_gb,
                "status": status,
            }
        )

    overall_status = "ready"
    if hard_breaches > 0:
        overall_status = "blocked"
    elif soft_breaches > 0:
        overall_status = "degraded"

    recommended_actions = ordered_unique(
        [
            "tighten explanation retention or cold-tier offload before hot-path quotas spill further" if any(row["family"] == "decision_explanations" and row["status"] != "ready" for row in lanes) else "",
            "checkpoint and compact sql_link shards before the stateful_sql quota breaches become runtime blocking" if any(row["family"] == "sql_link_shards" and row["status"] != "ready" for row in lanes) else "",
            "garbage-collect artifact store blobs proactively during long-run windows" if any(row["family"] == "artifact_store" and row["status"] != "ready" for row in lanes) else "",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "quota_summary": {
            "hard_breaches": hard_breaches,
            "soft_breaches": soft_breaches,
            "tracked_lane_count": len(lanes),
        },
        "lanes": lanes,
        "infra_bots": ["storage_quota_guard", "storage_tier_policy", "retention_debt_sheriff"],
        "recommended_actions": recommended_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply hard storage quotas per lane for long-running runtime windows.")
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
            "storage_quota_guard "
            f"overall_status={payload.get('overall_status', '')} "
            f"hard_breaches={int(((payload.get('quota_summary') or {}).get('hard_breaches', 0) or 0))}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
