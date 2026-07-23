#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
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
SOFT_ADVISORY_TOLERANCE_GB = 2.0
SOFT_ADVISORY_TOLERANCE_RATIO = 0.10
DEFAULT_ACTIVE_DECISION_BUFFER_ALLOWANCE_GB = float(os.getenv("STORAGE_QUOTA_ACTIVE_DECISION_BUFFER_ALLOWANCE_GB", "16"))
DEFAULT_ACTIVE_GOVERNANCE_BUFFER_ALLOWANCE_GB = float(os.getenv("STORAGE_QUOTA_ACTIVE_GOVERNANCE_BUFFER_ALLOWANCE_GB", "24"))
DEFAULT_ACTIVE_EXPLANATION_BUFFER_ALLOWANCE_GB = float(os.getenv("STORAGE_QUOTA_ACTIVE_EXPLANATION_BUFFER_ALLOWANCE_GB", "16"))

FAMILY_TO_ROLE = {
    "sql_link_shards": "stateful_sql",
    "decision_explanations": "explainability",
    "decisions": "live_decisioning",
    "governance": "governance_telemetry",
    "governance_events": "governance_telemetry",
    "governance_channels": "governance_telemetry",
    "content_store": "artifact_store",
}


def _round_gb(value: float) -> float:
    return round(float(value), 3)


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _role_bytes(storage_tier: dict[str, Any], role: str) -> int:
    by_role = storage_tier.get("by_service_role") if isinstance(storage_tier.get("by_service_role"), dict) else {}
    return int(((by_role.get(role) or {}).get("bytes", 0)) or 0)


def _family_bytes(storage_tier: dict[str, Any], family: str) -> int:
    by_family = storage_tier.get("by_family") if isinstance(storage_tier.get("by_family"), dict) else {}
    return int(((by_family.get(family) or {}).get("bytes", 0)) or 0)


def _compressed_archive_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    try:
        iterator = root.rglob("*.gz")
    except Exception:
        return 0
    for path in iterator:
        try:
            if path.is_file() and not path.is_symlink():
                total += max(int(path.stat().st_size), 0)
        except Exception:
            continue
    return total


def _today_tokens() -> set[str]:
    now = datetime.now(timezone.utc)
    tokens = {now.strftime("%Y%m%d")}
    try:
        tokens.add(now.astimezone().strftime("%Y%m%d"))
    except Exception:
        pass
    return tokens


def _active_current_day_decision_bytes(project_root: Path) -> int:
    root = project_root / "decisions"
    try:
        scan_root = root.resolve() if root.exists() else root
    except Exception:
        scan_root = root
    if not scan_root.exists():
        return 0
    tokens = _today_tokens()
    total = 0
    try:
        iterator = scan_root.rglob("*.jsonl")
    except Exception:
        return 0
    for path in iterator:
        try:
            if not path.is_file() or path.is_symlink():
                continue
            if not any(token in path.name for token in tokens):
                continue
            if ".local_fallback" in path.name or ".tmp." in path.name or ".compact_pending" in path.name:
                continue
            total += max(int(path.stat().st_size), 0)
        except Exception:
            continue
    return total


def _active_current_day_governance_channel_bytes(project_root: Path) -> int:
    root = project_root / "governance" / "channels"
    try:
        scan_root = root.resolve() if root.exists() else root
    except Exception:
        scan_root = root
    if not scan_root.exists():
        return 0
    tokens = _today_tokens()
    total = 0
    try:
        iterator = scan_root.rglob("*.jsonl")
    except Exception:
        return 0
    for path in iterator:
        try:
            if not path.is_file() or path.is_symlink():
                continue
            if not any(token in path.name for token in tokens):
                continue
            if ".local_fallback" in path.name or ".tmp." in path.name or ".compact_pending" in path.name:
                continue
            total += max(int(path.stat().st_size), 0)
        except Exception:
            continue
    return total


def _active_current_day_explanation_bytes(project_root: Path) -> int:
    root = project_root / "decision_explanations"
    try:
        scan_root = root.resolve() if root.exists() else root
    except Exception:
        scan_root = root
    if not scan_root.exists():
        return 0
    tokens = _today_tokens()
    total = 0
    try:
        iterator = scan_root.rglob("*.jsonl")
    except Exception:
        return 0
    for path in iterator:
        try:
            if not path.is_file() or path.is_symlink():
                continue
            if not any(token in path.name for token in tokens):
                continue
            if ".local_fallback" in path.name or ".tmp." in path.name or ".compact_pending" in path.name:
                continue
            total += max(int(path.stat().st_size), 0)
        except Exception:
            continue
    return total


def _quota_lane_action(lane: dict[str, Any]) -> str:
    family = str(lane.get("family") or "")
    status = str(lane.get("status") or "")
    if status == "ready":
        return ""
    if family == "decisions":
        return "prioritize ingestion-storage-control and the core decision drainer before widening decision log producers"
    if family == "governance_telemetry":
        return "shed verbose governance telemetry by running governance-telemetry-compactor to rotate oversized governance channel telemetry before trusting the support telemetry quota"
    if family == "decision_explanations":
        return "tighten explanation retention or cold-tier offload before hot-path quotas spill further"
    if family == "sql_link_shards":
        return "checkpoint and compact sql_link shards before the stateful_sql quota becomes runtime blocking"
    if family == "artifact_store":
        return "garbage-collect artifact store blobs proactively during long-run windows"
    return f"reduce {family} storage before allowing growth lanes to widen"


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    storage_tier = load_json(project_root / "governance" / "health" / "storage_tier_policy_latest.json")
    hot_lane_retention = load_json(project_root / "governance" / "health" / "hot_lane_retention_control_latest.json")
    data_collection_storage_guard = load_json(project_root / "governance" / "health" / "data_collection_storage_guard_latest.json")
    safe_space_recovery = (
        data_collection_storage_guard.get("safe_space_recovery")
        if isinstance(data_collection_storage_guard.get("safe_space_recovery"), dict)
        else {}
    )
    duplicate_cleanup = (
        data_collection_storage_guard.get("duplicate_cleanup")
        if isinstance(data_collection_storage_guard.get("duplicate_cleanup"), dict)
        else {}
    )
    safe_space_scan = safe_space_recovery.get("scan") if isinstance(safe_space_recovery.get("scan"), dict) else {}
    safe_space_by_reason = (
        safe_space_recovery.get("by_reason")
        if isinstance(safe_space_recovery.get("by_reason"), dict)
        else {}
    )
    duplicate_bucket = (
        safe_space_by_reason.get("duplicate_local_fallback_artifact")
        if isinstance(safe_space_by_reason.get("duplicate_local_fallback_artifact"), dict)
        else {}
    )
    unbacked_fallback_gb = _safe_float(safe_space_scan.get("unbacked_duplicate_gb"), 0.0)
    fallback_reconciliation_bytes = max(
        int(max(unbacked_fallback_gb, 0.0) * (1024**3)),
        int(max(_safe_float(duplicate_cleanup.get("candidate_gb"), 0.0), 0.0) * (1024**3)),
        int(max(int(duplicate_bucket.get("bytes") or 0), 0)),
    )
    compressed_archive_offsets = {
        "governance_telemetry": _compressed_archive_bytes(project_root / "governance" / "channels"),
        "decisions": _compressed_archive_bytes(project_root / "decisions"),
        "decision_explanations": _compressed_archive_bytes(project_root / "decision_explanations"),
    }
    active_current_day_decision_bytes = _active_current_day_decision_bytes(project_root)
    active_current_day_governance_channel_bytes = _active_current_day_governance_channel_bytes(project_root)
    active_current_day_explanation_bytes = _active_current_day_explanation_bytes(project_root)
    storage_tier_status = str(storage_tier.get("overall_status") or "").strip()
    pressure = storage_tier.get("pressure") if isinstance(storage_tier.get("pressure"), dict) else {}
    hot_path_over_budget_bytes = int(max(_safe_float(pressure.get("hot_path_over_budget_bytes"), 0.0), 0.0))
    live_hot_path_bytes = int(max(_safe_float(pressure.get("live_hot_path_bytes"), 0.0), 0.0))
    hot_budget_bytes = int(max(_safe_float(pressure.get("hot_budget_bytes"), 0.0), 0.0))
    hot_path_green = bool(
        storage_tier_status == "ready"
        and hot_path_over_budget_bytes <= 0
        and (hot_budget_bytes <= 0 or live_hot_path_bytes <= hot_budget_bytes)
    )
    hot_lane_mode = str(hot_lane_retention.get("mode") or "").strip()
    hot_lane_status = str(hot_lane_retention.get("overall_status") or "").strip()
    hot_lane_control_active = bool(
        hot_lane_retention.get("ok", False)
        and hot_lane_mode in {"thin_optional_sub_bot_decisions", "emergency_hot_thin"}
        and hot_lane_status in {"active", "critical", "watching", "ready"}
    )
    hot_lane_full_evidence_current_day_governance_relief = bool(
        hot_lane_retention.get("ok", False)
        and hot_lane_mode == "full_decision_evidence"
        and hot_lane_status == "ready"
        and hot_path_green
    )
    lanes: list[dict[str, Any]] = []
    hard_breaches = 0
    soft_breaches = 0
    advisory_breaches = 0
    for family, quota in DEFAULT_QUOTAS_GB.items():
        bytes_used = _family_bytes(storage_tier, family)
        if bytes_used == 0:
            bytes_used = _role_bytes(storage_tier, FAMILY_TO_ROLE.get(family, family))
        adjustments: list[dict[str, Any]] = []
        if family == "decisions" and fallback_reconciliation_bytes > 0:
            applied = min(bytes_used, fallback_reconciliation_bytes)
            bytes_used = max(bytes_used - applied, 0)
            adjustments.append(
                {
                    "reason": "exclude_local_fallback_reconciliation_artifacts_from_hot_quota",
                    "gb": _round_gb(float(applied) / float(1024**3)),
                }
            )
        compressed_offset = min(bytes_used, int(compressed_archive_offsets.get(family, 0) or 0))
        if compressed_offset > 0:
            bytes_used = max(bytes_used - compressed_offset, 0)
            adjustments.append(
                {
                    "reason": "exclude_compressed_archive_history_from_hot_quota",
                    "gb": _round_gb(float(compressed_offset) / float(1024**3)),
                }
            )
        if family == "decisions" and active_current_day_decision_bytes > 0 and hot_lane_control_active:
            applied = min(bytes_used, active_current_day_decision_bytes)
            if applied > 0:
                bytes_used = max(bytes_used - applied, 0)
                adjustments.append(
                    {
                        "reason": "exclude_current_day_active_decision_buffer_under_hot_lane_retention",
                        "gb": _round_gb(float(applied) / float(1024**3)),
                        "hot_lane_mode": hot_lane_mode,
                        "hard_quota_protected": False,
                        "still_counted_by_disk_free_forecast": True,
                    }
                )
        if family == "governance_telemetry" and active_current_day_governance_channel_bytes > 0:
            applied = 0
            reason = ""
            if hot_lane_control_active:
                applied = min(bytes_used, active_current_day_governance_channel_bytes)
                reason = "exclude_current_day_active_governance_channels_under_hot_lane_retention"
            elif hot_lane_full_evidence_current_day_governance_relief:
                allowance_bytes = int(max(DEFAULT_ACTIVE_GOVERNANCE_BUFFER_ALLOWANCE_GB, 0.0) * (1024**3))
                applied = min(bytes_used, active_current_day_governance_channel_bytes, allowance_bytes)
                reason = "exclude_bounded_current_day_active_governance_channels_under_green_full_evidence_hot_lane"
            if applied > 0:
                bytes_used = max(bytes_used - applied, 0)
                adjustments.append(
                    {
                        "reason": reason,
                        "gb": _round_gb(float(applied) / float(1024**3)),
                        "hot_lane_mode": hot_lane_mode,
                        "hot_path_over_budget_gb": _round_gb(float(hot_path_over_budget_bytes) / float(1024**3)),
                        "allowance_gb": _round_gb(DEFAULT_ACTIVE_GOVERNANCE_BUFFER_ALLOWANCE_GB),
                        "still_counted_by_disk_free_forecast": True,
                    }
                )
        if family == "decision_explanations" and active_current_day_explanation_bytes > 0 and hot_lane_control_active:
            raw_used_gb = float(bytes_used) / float(1024**3)
            hard_gb = float(quota["hard"])
            if raw_used_gb < hard_gb:
                allowance_bytes = int(max(DEFAULT_ACTIVE_EXPLANATION_BUFFER_ALLOWANCE_GB, 0.0) * (1024**3))
                applied = min(bytes_used, active_current_day_explanation_bytes, allowance_bytes)
                if applied > 0:
                    bytes_used = max(bytes_used - applied, 0)
                    adjustments.append(
                        {
                            "reason": "exclude_bounded_current_day_explanation_buffer_under_hot_lane_retention",
                            "gb": _round_gb(float(applied) / float(1024**3)),
                            "hot_lane_mode": hot_lane_mode,
                            "allowance_gb": _round_gb(DEFAULT_ACTIVE_EXPLANATION_BUFFER_ALLOWANCE_GB),
                            "hard_quota_protected": True,
                            "still_counted_by_disk_free_forecast": True,
                        }
                    )
        elif family == "decisions" and active_current_day_decision_bytes > 0 and not hot_lane_control_active:
            # Current-day active decision buffers are hot evidence, not old quota
            # debt. Exclude only a bounded soft-quota allowance after compressed
            # archive history is removed, and never use it to hide a hard breach.
            raw_used_gb = float(bytes_used) / float(1024**3)
            hard_gb = float(quota["hard"])
            if raw_used_gb < hard_gb:
                allowance_bytes = int(max(DEFAULT_ACTIVE_DECISION_BUFFER_ALLOWANCE_GB, 0.0) * (1024**3))
                applied = min(bytes_used, active_current_day_decision_bytes, allowance_bytes)
                if applied > 0:
                    bytes_used = max(bytes_used - applied, 0)
                    adjustments.append(
                        {
                            "reason": "exclude_bounded_current_day_active_decision_buffer_from_soft_quota",
                            "gb": _round_gb(float(applied) / float(1024**3)),
                            "hard_quota_protected": True,
                        }
                    )
        used_gb = float(bytes_used) / float(1024**3)
        soft_gb = float(quota["soft"])
        hard_gb = float(quota["hard"])
        over_soft_gb = max(used_gb - soft_gb, 0.0)
        over_hard_gb = max(used_gb - hard_gb, 0.0)
        soft_ratio = used_gb / max(soft_gb, 0.001)
        hard_ratio = used_gb / max(hard_gb, 0.001)
        status = "ready"
        if used_gb >= hard_gb:
            status = "blocked"
            hard_breaches += 1
        elif used_gb >= soft_gb:
            if over_soft_gb <= SOFT_ADVISORY_TOLERANCE_GB and soft_ratio <= (1.0 + SOFT_ADVISORY_TOLERANCE_RATIO):
                status = "advisory"
                advisory_breaches += 1
            else:
                status = "degraded"
                soft_breaches += 1
        lanes.append(
            {
                "family": family,
                "used_gb": _round_gb(used_gb),
                "soft_quota_gb": soft_gb,
                "hard_quota_gb": hard_gb,
                "over_soft_gb": _round_gb(over_soft_gb),
                "over_hard_gb": _round_gb(over_hard_gb),
                "soft_ratio": round(soft_ratio, 3),
                "hard_ratio": round(hard_ratio, 3),
                "status": status,
                "adjustments": adjustments,
            }
        )

    overall_status = "ready"
    if hard_breaches > 0:
        overall_status = "blocked"
    elif soft_breaches > 0:
        overall_status = "degraded"

    blocked_lanes = [row for row in lanes if str(row.get("status") or "") == "blocked"]
    degraded_lanes = [row for row in lanes if str(row.get("status") or "") == "degraded"]
    advisory_lanes = [row for row in lanes if str(row.get("status") or "") == "advisory"]
    ranked_breaches = sorted(
        [*blocked_lanes, *degraded_lanes],
        key=lambda row: (
            _safe_float(row.get("over_hard_gb"), 0.0),
            _safe_float(row.get("over_soft_gb"), 0.0),
            _safe_float(row.get("hard_ratio"), 0.0),
        ),
        reverse=True,
    )
    recommended_actions = ordered_unique(_quota_lane_action(row) for row in ranked_breaches)
    if blocked_lanes:
        recommended_actions.append("keep expansion and heavy training gated until blocked storage quota lanes fall below hard quota")
    recommended_actions = ordered_unique(recommended_actions)

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "quota_summary": {
            "hard_breaches": hard_breaches,
            "soft_breaches": soft_breaches,
            "advisory_breaches": advisory_breaches,
            "tracked_lane_count": len(lanes),
            "blocked_families": [str(row.get("family") or "") for row in blocked_lanes],
            "degraded_families": [str(row.get("family") or "") for row in degraded_lanes],
            "advisory_families": [str(row.get("family") or "") for row in advisory_lanes],
            "soft_advisory_tolerance_gb": SOFT_ADVISORY_TOLERANCE_GB,
            "soft_advisory_tolerance_ratio": SOFT_ADVISORY_TOLERANCE_RATIO,
            "worst_over_hard_gb": _round_gb(max((_safe_float(row.get("over_hard_gb"), 0.0) for row in lanes), default=0.0)),
            "worst_hard_ratio": round(max((_safe_float(row.get("hard_ratio"), 0.0) for row in lanes), default=0.0), 3),
        },
        "lanes": lanes,
        "active_hot_buffer_containment": {
            "hot_lane_control_active": bool(hot_lane_control_active),
            "hot_lane_status": hot_lane_status,
            "hot_lane_mode": hot_lane_mode,
            "active_current_day_decision_gb": _round_gb(float(active_current_day_decision_bytes) / float(1024**3)),
            "active_current_day_governance_channel_gb": _round_gb(float(active_current_day_governance_channel_bytes) / float(1024**3)),
            "active_current_day_explanation_gb": _round_gb(float(active_current_day_explanation_bytes) / float(1024**3)),
            "hot_path_green": bool(hot_path_green),
            "hot_path_over_budget_gb": _round_gb(float(hot_path_over_budget_bytes) / float(1024**3)),
            "hot_lane_full_evidence_current_day_governance_relief": bool(hot_lane_full_evidence_current_day_governance_relief),
            "active_governance_buffer_allowance_gb": _round_gb(DEFAULT_ACTIVE_GOVERNANCE_BUFFER_ALLOWANCE_GB),
            "active_explanation_buffer_allowance_gb": _round_gb(DEFAULT_ACTIVE_EXPLANATION_BUFFER_ALLOWANCE_GB),
            "policy": "current-day decision bytes are excluded from lane quota only when hot-lane retention is actively throttling their source; bounded current-day explanation buffers are excluded from soft quota only while hot-lane retention is active and cannot hide hard breaches; bounded current-day governance evidence is also excluded when full-evidence mode is ready and the hot path is green; disk free forecast still counts the bytes",
        },
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
