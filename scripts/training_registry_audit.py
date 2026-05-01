#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.training_quality_thresholds import (
    STAGED_SUPPORT_RECOVERY_TEST_ACCURACY_FLOOR,
    STRONG_TEST_ACCURACY_FLOOR,
    TARGET_QUALITY_SCORE_FLOOR,
)

DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "master_bot_registry.json"
DEFAULT_DIAGNOSTICS_DIR = PROJECT_ROOT / "governance" / "training_diagnostics"
DEFAULT_SNAPSHOT_HEALTH = PROJECT_ROOT / "governance" / "health" / "runtime_training_snapshot_latest.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "governance" / "health" / "training_registry_audit_latest.json"

STRONG_QUALITY_SCORE_FLOOR = 0.20
STAGED_SUPPORT_RECOVERY_QUALITY_FLOOR = 0.15


def _safe_json_load(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _load_registry_rows(path: Path) -> list[dict[str, Any]]:
    obj = _safe_json_load(path)
    rows = obj.get("sub_bots") if isinstance(obj.get("sub_bots"), list) else []
    return [row for row in rows if isinstance(row, dict)]


def _diagnostic_path(diag_dir: Path, bot_id: str) -> Path:
    return diag_dir / f"{bot_id}_latest.json"


def _age_hours(path: Path) -> float | None:
    try:
        return max((datetime.now(timezone.utc) - datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)).total_seconds() / 3600.0, 0.0)
    except Exception:
        return None


def _parse_iso_utc(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _top_failure_excerpt(diag: dict[str, Any]) -> list[str]:
    failures = diag.get("quality_failures") if isinstance(diag.get("quality_failures"), list) else []
    return [str(x) for x in failures[:3]]


def _best_score(*values: Any) -> float:
    best = 0.0
    for raw in values:
        try:
            value = float(raw)
        except Exception:
            continue
        best = max(best, value)
    return best


def _support_recovery_reason(row: dict[str, Any]) -> bool:
    reason_bits = " ".join(
        [
            str(row.get("registry_reason") or "").strip().lower(),
            str(row.get("promotion_reason") or "").strip().lower(),
        ]
    )
    return any(
        token in reason_bits
        for token in (
            "supportable_recovery",
            "role_floor_",
            "manual_canary_restore",
            "manual_collection_restore",
        )
    )


def _infer_cause(diag: dict[str, Any], registry_row: dict[str, Any]) -> str:
    status = str(diag.get("status") or "").strip().lower()
    if not diag:
        if str(registry_row.get("promotion_reason") or "").strip().lower() == "new_runtime_candidate":
            return "new_runtime_candidate"
        return "missing_diagnostic"
    if status == "deferred_sample_starved":
        sample_count = int(diag.get("sample_count", 0) or 0)
        eligible_sequences = int(diag.get("eligible_sequences", 0) or 0)
        sequence_count = int(diag.get("sequence_count", 0) or 0)
        skipped_filtered = int(diag.get("skipped_filtered", 0) or 0)
        skipped_low_conf = int(diag.get("skipped_low_confidence", 0) or 0)
        skipped_labels = int(diag.get("skipped_labels", 0) or 0)
        positive_rate = float(diag.get("positive_rate", 0.0) or 0.0)
        if sample_count == 0 and eligible_sequences == 0 and sequence_count == 0:
            return "shared_runtime_input_gap"
        if sample_count == 0 and eligible_sequences == 0:
            return "sequence_depth_gap"
        if skipped_filtered > max(skipped_labels, skipped_low_conf):
            return "sample_filter_too_strict"
        if skipped_low_conf > max(skipped_filtered, skipped_labels):
            return "confidence_gate_too_strict"
        if skipped_labels > 0:
            if positive_rate <= 0.02 or positive_rate >= 0.98:
                return "label_balance_gap"
            return "label_builder_too_strict"
        return "dataset_floor_gap"
    if status == "failed":
        return "quality_guard_failure"
    if status in {"ok", "passed", "complete", "completed"}:
        return "passed"
    return status or "unknown"


def _tier_for_row(row: dict[str, Any]) -> str:
    if bool(row.get("active")):
        if not bool(row.get("diagnostic_fresh", False)):
            return "active_stale"
        cause = str(row.get("inferred_cause") or "")
        if cause == "passed":
            return "active_production"
        if cause == "quality_guard_failure":
            return "active_probation"
        return "active_repair"
    cause = str(row.get("inferred_cause") or "")
    lifecycle_state = str(row.get("lifecycle_state") or "").strip().lower()
    if cause == "new_runtime_candidate":
        return "research_candidate"
    if lifecycle_state in {"retired", "deleted", "deactivated"} or bool(row.get("deleted_from_rotation", False)):
        return "retired"
    return "inactive_backlog"


def _supportability_for_row(row: dict[str, Any], *, snapshot_ready: bool = False) -> str:
    if not bool(row.get("active")):
        return "inactive"
    best_quality_score = _best_score(row.get("registry_quality_score"), row.get("candidate_quality_score"))
    best_test_accuracy = _best_score(row.get("registry_test_accuracy"), row.get("candidate_test_accuracy"))
    if not bool(row.get("diagnostic_fresh", False)):
        if (
            bool(row.get("model_artifact_exists", False))
            and best_quality_score >= TARGET_QUALITY_SCORE_FLOOR
            and best_test_accuracy >= STRONG_TEST_ACCURACY_FLOOR
            and str(row.get("inferred_cause") or "") not in {"shared_runtime_input_gap", "sequence_depth_gap", "quality_guard_failure"}
        ):
            return "artifact_backed_active"
        if (
            snapshot_ready
            and _support_recovery_reason(row)
            and best_quality_score >= STAGED_SUPPORT_RECOVERY_QUALITY_FLOOR
            and best_test_accuracy >= STAGED_SUPPORT_RECOVERY_TEST_ACCURACY_FLOOR
            and str(row.get("inferred_cause") or "") not in {"shared_runtime_input_gap", "sequence_depth_gap", "quality_guard_failure"}
        ):
            return "staged_support_recovery"
        if (
            snapshot_ready
            and best_quality_score >= STRONG_QUALITY_SCORE_FLOOR
            and best_test_accuracy >= STRONG_TEST_ACCURACY_FLOOR
            and str(row.get("inferred_cause") or "") not in {"shared_runtime_input_gap", "sequence_depth_gap", "quality_guard_failure"}
        ):
            return "registry_seeded_active"
        return "unsupported_stale_diagnostics"
    cause = str(row.get("inferred_cause") or "")
    if cause in {"shared_runtime_input_gap", "sequence_depth_gap"}:
        return "unsupported_runtime_inputs"
    if cause in {"sample_filter_too_strict", "confidence_gate_too_strict", "label_builder_too_strict", "label_balance_gap", "dataset_floor_gap"}:
        return "unsupported_labeling"
    if cause == "quality_guard_failure":
        return "supported_but_quality_failing"
    if cause == "passed":
        return "supportable_active"
    return "needs_review"


def _audit_row(registry_row: dict[str, Any], diagnostics_dir: Path) -> dict[str, Any]:
    bot_id = str(registry_row.get("bot_id") or "").strip().lower()
    path = _diagnostic_path(diagnostics_dir, bot_id)
    diag = _safe_json_load(path) if path.exists() else {}
    status = str(diag.get("status") or ("missing_diagnostic" if not diag else "unknown")).strip().lower()
    quality_failures = _top_failure_excerpt(diag)
    diagnostic_age_hours = _age_hours(path) if path.exists() else None
    model_path = Path(str(registry_row.get("model_path") or "").strip()) if str(registry_row.get("model_path") or "").strip() else None
    log_path = Path(str(registry_row.get("log_file") or "").strip()) if str(registry_row.get("log_file") or "").strip() else None
    out = {
        "bot_id": bot_id,
        "bot_role": str(registry_row.get("bot_role") or ""),
        "active": bool(registry_row.get("active", False)),
        "lifecycle_state": str(registry_row.get("lifecycle_state") or ""),
        "registry_reason": str(registry_row.get("reason") or ""),
        "promotion_reason": str(registry_row.get("promotion_reason") or ""),
        "deleted_from_rotation": bool(registry_row.get("deleted_from_rotation", False)),
        "status": status,
        "sample_count": int(diag.get("sample_count", 0) or 0),
        "eligible_sequences": int(diag.get("eligible_sequences", 0) or 0),
        "sequence_count": int(diag.get("sequence_count", 0) or 0),
        "observation_count": int(diag.get("observation_count", 0) or 0),
        "positive_rate": float(diag.get("positive_rate", 0.0) or 0.0),
        "skipped_filtered": int(diag.get("skipped_filtered", 0) or 0),
        "skipped_low_confidence": int(diag.get("skipped_low_confidence", 0) or 0),
        "skipped_labels": int(diag.get("skipped_labels", 0) or 0),
        "quality_failures": quality_failures,
        "failure_categories": list(diag.get("failure_categories") or []),
        "diagnostics_path": str(path) if path.exists() else "",
        "diagnostic_age_hours": round(float(diagnostic_age_hours), 3) if diagnostic_age_hours is not None else None,
        "diagnostic_fresh": False,
        "model_artifact_exists": bool(model_path and model_path.exists()),
        "log_artifact_exists": bool(log_path and log_path.exists()),
        "registry_quality_score": float(registry_row.get("quality_score", 0.0) or 0.0),
        "registry_test_accuracy": float(registry_row.get("test_accuracy", 0.0) or 0.0),
        "candidate_quality_score": float(registry_row.get("candidate_quality_score", 0.0) or 0.0),
        "candidate_test_accuracy": float(registry_row.get("candidate_test_accuracy", 0.0) or 0.0),
    }
    out["best_quality_score"] = _best_score(out.get("registry_quality_score"), out.get("candidate_quality_score"))
    out["best_test_accuracy"] = _best_score(out.get("registry_test_accuracy"), out.get("candidate_test_accuracy"))
    out["inferred_cause"] = _infer_cause(diag, registry_row)
    return out


def _recommendations(rows: list[dict[str, Any]], snapshot: dict[str, Any]) -> list[str]:
    active_rows = [row for row in rows if bool(row.get("active"))]
    active_counter = Counter(str(row.get("inferred_cause") or "") for row in active_rows)
    active_stale = sum(1 for row in active_rows if not bool(row.get("diagnostic_fresh", False)))
    recs: list[str] = []
    if active_counter.get("shared_runtime_input_gap", 0) >= 1:
        recs.append("rebuild_runtime_training_snapshot_and_rerun_targeted_retrain")
    if active_stale > 0:
        recs.append("refresh_or_downgrade_active_bots_with_stale_diagnostics")
    if active_counter.get("sequence_depth_gap", 0) >= 2:
        recs.append("increase_runtime_sequence_depth_or_snapshot_coverage")
    if active_counter.get("sample_filter_too_strict", 0) >= 2 or active_counter.get("confidence_gate_too_strict", 0) >= 2:
        recs.append("relax_runtime_filters_before_collecting_more_data")
    if active_counter.get("label_builder_too_strict", 0) >= 2 or active_counter.get("label_balance_gap", 0) >= 2:
        recs.append("audit_label_builders_and_positive_rate_thresholds")
    if active_counter.get("quality_guard_failure", 0) >= 2:
        recs.append("focus_on_quality_guard_failures_after_sample_starvation_clears")
    if snapshot and int(snapshot.get("row_count", 0) or 0) == 0:
        recs.append("block_full_retrain_until_runtime_snapshot_has_rows")
    return recs


def build_audit_payload(
    *,
    registry_path: Path,
    diagnostics_dir: Path,
    snapshot_health_path: Path,
    max_diagnostic_age_hours: float = 72.0,
) -> dict[str, Any]:
    rows = _load_registry_rows(registry_path)
    audits = [_audit_row(row, diagnostics_dir) for row in rows if str(row.get("bot_id") or "").strip()]
    snapshot = _safe_json_load(snapshot_health_path)
    snapshot_ts = _parse_iso_utc(snapshot.get("timestamp_utc"))
    snapshot_age_hours = (
        max((datetime.now(timezone.utc) - snapshot_ts).total_seconds() / 3600.0, 0.0)
        if snapshot_ts is not None
        else None
    )
    snapshot_ready = bool(
        int(snapshot.get("row_count", 0) or 0) > 0
        and snapshot_age_hours is not None
        and snapshot_age_hours <= 36.0
    )
    for row in audits:
        age = row.get("diagnostic_age_hours")
        row["diagnostic_fresh"] = bool(age is not None and float(age) <= max(float(max_diagnostic_age_hours), 0.0))
        row["tier"] = _tier_for_row(row)
        row["supportability_status"] = _supportability_for_row(row, snapshot_ready=snapshot_ready)
    status_counts = Counter(str(row.get("status") or "") for row in audits)
    cause_counts = Counter(str(row.get("inferred_cause") or "") for row in audits)
    tier_counts = Counter(str(row.get("tier") or "") for row in audits)
    supportability_counts = Counter(str(row.get("supportability_status") or "") for row in audits)
    active_rows = [row for row in audits if bool(row.get("active"))]
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "registry_path": str(registry_path),
        "diagnostics_dir": str(diagnostics_dir),
        "snapshot_health_path": str(snapshot_health_path) if snapshot_health_path.exists() else "",
        "max_diagnostic_age_hours": float(max_diagnostic_age_hours),
        "runtime_snapshot_ready": snapshot_ready,
        "runtime_snapshot_age_hours": round(float(snapshot_age_hours), 3) if snapshot_age_hours is not None else None,
        "registry_total_bots": len(audits),
        "registry_active_bots": sum(1 for row in audits if bool(row.get("active"))),
        "status_counts": dict(sorted(status_counts.items())),
        "inferred_cause_counts": dict(sorted(cause_counts.items())),
        "tier_counts": dict(sorted(tier_counts.items())),
        "supportability_counts": dict(sorted(supportability_counts.items())),
        "runtime_snapshot": snapshot,
        "active_sample_starved": [
            row for row in active_rows if str(row.get("status")) == "deferred_sample_starved"
        ][:25],
        "active_quality_failed": [
            row for row in active_rows if str(row.get("inferred_cause")) == "quality_guard_failure"
        ][:25],
        "active_stale_diagnostics": [
            row for row in active_rows if not bool(row.get("diagnostic_fresh", False))
        ][:25],
        "active_registry_seeded": [
            row for row in active_rows if str(row.get("supportability_status") or "") == "registry_seeded_active"
        ][:25],
        "active_staged_support_recovery": [
            row for row in active_rows if str(row.get("supportability_status") or "") == "staged_support_recovery"
        ][:25],
        "tiers": {
            "active_production": [row for row in audits if str(row.get("tier")) == "active_production"][:25],
            "active_probation": [row for row in audits if str(row.get("tier")) == "active_probation"][:25],
            "active_repair": [row for row in audits if str(row.get("tier")) == "active_repair"][:25],
            "research_candidate": [row for row in audits if str(row.get("tier")) == "research_candidate"][:25],
            "retired": [row for row in audits if str(row.get("tier")) == "retired"][:25],
        },
        "inactive_new_candidates": [
            row for row in audits if str(row.get("inferred_cause")) == "new_runtime_candidate"
        ][:25],
        "recommendations": [],
    }
    payload["recommendations"] = _recommendations(audits, snapshot)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit registry-wide training readiness and shared failure modes.")
    parser.add_argument("--registry-path", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--diagnostics-dir", default=str(DEFAULT_DIAGNOSTICS_DIR))
    parser.add_argument("--snapshot-health-path", default=str(DEFAULT_SNAPSHOT_HEALTH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--max-diagnostic-age-hours", type=float, default=72.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_audit_payload(
        registry_path=Path(args.registry_path).expanduser(),
        diagnostics_dir=Path(args.diagnostics_dir).expanduser(),
        snapshot_health_path=Path(args.snapshot_health_path).expanduser(),
        max_diagnostic_age_hours=float(args.max_diagnostic_age_hours),
    )
    output_path = Path(args.output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "training_registry_audit "
            f"bots={int(payload['registry_total_bots'])} "
            f"active_sample_starved={len(payload['active_sample_starved'])} "
            f"active_quality_failed={len(payload['active_quality_failed'])} "
            f"output={output_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
