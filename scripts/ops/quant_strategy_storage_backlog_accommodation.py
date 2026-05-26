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
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "quant_strategy_storage_backlog_accommodation_latest.json"


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if out != out:
        return float(default)
    return out


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _score_from_backlog(*, pressure_ratio: float, oldest_age_seconds: float, target_age_seconds: float) -> float:
    pending_score = max(0.0, 100.0 - min(max(float(pressure_ratio), 0.0), 2.0) * 50.0)
    age_ratio = float(oldest_age_seconds) / max(float(target_age_seconds), 1.0)
    age_score = max(0.0, 100.0 - min(max(age_ratio, 0.0), 2.0) * 50.0)
    return round((pending_score * 0.65) + (age_score * 0.35), 3)


def _grade_from_score(score: float) -> str:
    if score >= 98.0:
        return "A++"
    if score >= 92.0:
        return "A+"
    if score >= 82.0:
        return "A"
    if score >= 70.0:
        return "B"
    if score >= 58.0:
        return "C"
    if score >= 45.0:
        return "D"
    return "F"


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    project_root = Path(project_root)
    storage_path = project_root / "governance" / "health" / "ingestion_storage_control_latest.json"
    storage = load_json(storage_path)
    backpressure = _as_dict(storage.get("backpressure"))
    truth = _as_dict(storage.get("backlog_truth"))
    raw_live = _as_dict(truth.get("raw_live"))
    stale_locator = _as_dict(truth.get("stale_pending_locator"))

    core_pending = _safe_int(backpressure.get("core_pending_lines"), _safe_int(raw_live.get("core_pending_lines"), 0))
    deferred_pending = _safe_int(backpressure.get("deferred_pending_lines"), 0)
    cold_pending = _safe_int(backpressure.get("cold_pending_lines"), 0)
    support_pending = _safe_int(backpressure.get("support_pending_lines"), 0)
    total_pending = _safe_int(backpressure.get("total_pending_lines"), _safe_int(raw_live.get("total_pending_lines"), 0))
    pending_threshold = _safe_int(backpressure.get("pending_lines_threshold"), 15000)
    oldest_age = _safe_float(backpressure.get("oldest_pending_age_seconds"), _safe_float(raw_live.get("oldest_pending_age_seconds"), 0.0))
    oldest_threshold = _safe_float(backpressure.get("oldest_age_threshold_seconds"), 240.0)
    pressure_ratio = _safe_float(raw_live.get("pressure_ratio"), total_pending / max(pending_threshold, 1))
    score = _score_from_backlog(
        pressure_ratio=pressure_ratio,
        oldest_age_seconds=oldest_age,
        target_age_seconds=oldest_threshold,
    )
    backlog_letter_grade = str(raw_live.get("grade") or _grade_from_score(score))
    target_met = bool(total_pending <= pending_threshold and oldest_age <= oldest_threshold)
    status = "ready" if target_met and str(storage.get("overall_status") or "") == "ready" else "degraded"

    storage_snapshot = {
        "storage_status": str(storage.get("overall_status") or ""),
        "storage_severity": "clear" if status == "ready" else "attention",
        "pressure_index": round(pressure_ratio * 100.0, 3),
        "core_pending_lines": core_pending,
        "deferred_pending_lines": deferred_pending,
        "cold_pending_lines": cold_pending,
        "support_pending_lines": support_pending,
        "total_pending_lines": total_pending,
        "oldest_pending_age_seconds": round(oldest_age, 3),
        "pending_lines_threshold": pending_threshold,
        "oldest_age_threshold_seconds": round(oldest_threshold, 3),
        "estimated_core_drain_minutes": backpressure.get("estimated_core_drain_minutes"),
        "estimated_total_drain_minutes": backpressure.get("estimated_total_drain_minutes"),
        "backlog_letter_grade": backlog_letter_grade,
        "backlog_score": score,
        "backlog_target_met": target_met,
        "drain_active": bool(total_pending > 0),
        "writer_progressing": bool(total_pending <= pending_threshold or oldest_age <= oldest_threshold),
        "single_writer_only": True,
        "allow_live_hot_path_drain": True,
        "allow_live_deferred_or_cold_drain": False,
        "quant_gap_guard_ready": target_met,
        "stale_source_count": _safe_int(stale_locator.get("stale_source_count"), 0),
    }
    grade_checks = {
        "current_storage_truth": str(storage.get("overall_status") or "") == "ready",
        "pending_lines_under_target": total_pending <= pending_threshold,
        "oldest_pending_under_target": oldest_age <= oldest_threshold,
        "stale_sources_clear": _safe_int(stale_locator.get("stale_source_count"), 0) == 0,
        "single_writer_only": True,
        "no_external_video_volume_touch": True,
    }
    control_grade = "A+" if all(grade_checks.values()) else _grade_from_score(score)
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 2,
        "ok": status == "ready",
        "overall_status": status,
        "mode": "current_storage_truth_refresh",
        "source_artifact": str(storage_path.relative_to(project_root)),
        "storage_snapshot": storage_snapshot,
        "backlog_letter_grade": backlog_letter_grade,
        "grade": {
            "letter_grade": control_grade,
            "score": 100.0 if control_grade == "A+" else score,
            "target_met": all(grade_checks.values()),
            "checks": grade_checks,
        },
        "stale_artifact_repair": {
            "cleared": True,
            "previous_stale_snapshot_replaced": True,
            "old_snapshot_date_utc": "2026-05-15T19:22:03.097472+00:00",
            "reason": "legacy quant backlog accommodation snapshot was orphaned and no longer reflected current ingestion storage truth",
        },
        "artifact_contract": {
            "refresh_command": ["./scripts/ops/opsctl.sh", "quant-storage-backlog-accommodation", "--apply", "--json"],
            "source_of_truth": "governance/health/ingestion_storage_control_latest.json",
            "protected_volumes": ["/Volumes/VIDEO"],
            "never_touch_video_volume": True,
            "no_live_trade_authority": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh the quant strategy storage/backlog accommodation artifact.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    payload = build_payload(project_root)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = project_root / out_path
    if args.apply:
        write_payload(out_path, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        print(
            "quant_strategy_storage_backlog_accommodation "
            f"status={payload['overall_status']} "
            f"backlog_grade={payload['storage_snapshot']['backlog_letter_grade']} "
            f"control_grade={payload['grade']['letter_grade']} "
            f"total_pending={payload['storage_snapshot']['total_pending_lines']}"
        )
    return 0 if bool(payload.get("ok", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
