#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
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
    from core.storage_mounts import resolve_external_storage
    from scripts.ops.long_runtime_common import iso_now, load_json, write_payload
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from core.storage_mounts import resolve_external_storage
    from .long_runtime_common import iso_now, load_json, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "retention_intelligence_v2_latest.json"
DEFAULT_HISTORY_PATH = PROJECT_ROOT / "governance" / "health" / "retention_intelligence_v2_history.jsonl"
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "governance" / "retention" / "retention_class_registry.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "governance" / "retention" / "retention_report_card_latest.md"
PROTECTED_VOLUME_NAMES = {"VIDEO"}
PROTECTED_VOLUME_PATHS = {"/Volumes/VIDEO"}


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _gb(value: int | float) -> float:
    return round(float(value) / float(1024**3), 3)


def _volume_name(path: Path) -> str:
    parts = path.expanduser().parts
    if len(parts) >= 3 and parts[1] == "Volumes":
        return parts[2]
    return ""


def _is_protected_volume(path: Path) -> bool:
    try:
        text = str(path.expanduser().resolve())
    except Exception:
        text = str(path.expanduser())
    return _volume_name(path) in PROTECTED_VOLUME_NAMES or any(
        text == protected or text.startswith(f"{protected}/") for protected in PROTECTED_VOLUME_PATHS
    )


def _read_jsonl_rows(path: Path, *, limit: int = 10000) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if len(rows) >= max(int(limit), 1):
                    break
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except Exception:
        return []
    return rows


def _grade(score: float) -> str:
    value = float(score)
    if value >= 99.5:
        return "A+"
    if value >= 96.0:
        return "A+"
    if value >= 93.0:
        return "A"
    if value >= 90.0:
        return "A-"
    if value >= 86.0:
        return "B+"
    if value >= 83.0:
        return "B"
    if value >= 80.0:
        return "B-"
    if value >= 76.0:
        return "C+"
    if value >= 73.0:
        return "C"
    if value >= 70.0:
        return "C-"
    if value >= 60.0:
        return "D"
    return "F"


def _section(
    section_id: str,
    label: str,
    *,
    status: str,
    score: float,
    evidence: dict[str, Any] | None = None,
    next_action: str = "",
    risk_level: str = "low",
    blockers: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "section_id": section_id,
        "label": label,
        "status": status,
        "score": round(float(score), 2),
        "grade": _grade(float(score)),
        "risk_level": risk_level,
        "blockers": blockers or [],
        "evidence": evidence or {},
        "next_action": next_action,
    }


def _registry_classes() -> list[dict[str, Any]]:
    return [
        {
            "class_id": "live_decision_evidence",
            "families": ["decisions", "decision_explanations", "paper_bridge"],
            "economic_value": "critical",
            "hot_days": 3,
            "warm_days": 30,
            "cold_days": 180,
            "deep_cold_days": 365,
            "preserve_for_training": True,
            "replay_required": True,
            "compaction_policy": "gzip_after_hot_window_then_manifest_index",
            "deletion_owner": "stale-reaper",
        },
        {
            "class_id": "paper_execution_evidence",
            "families": ["paper_trades", "paper_fills", "paper_reconciliation", "execution_lanes"],
            "economic_value": "critical",
            "hot_days": 7,
            "warm_days": 45,
            "cold_days": 180,
            "deep_cold_days": 365,
            "preserve_for_training": True,
            "replay_required": True,
            "compaction_policy": "sql_link_then_cold_archive",
            "deletion_owner": "stale-reaper",
        },
        {
            "class_id": "training_feature_evidence",
            "families": ["runtime_training_snapshot", "feature_store", "labels", "walk_forward_runs"],
            "economic_value": "high",
            "hot_days": 14,
            "warm_days": 90,
            "cold_days": 365,
            "deep_cold_days": 730,
            "preserve_for_training": True,
            "replay_required": True,
            "compaction_policy": "content_hash_manifest_plus_sampled_raw_tail",
            "deletion_owner": "training-lineage",
        },
        {
            "class_id": "provider_market_context",
            "families": ["market_quote_profiles", "macro_context", "sec_edgar_context", "provider_verification"],
            "economic_value": "high",
            "hot_days": 7,
            "warm_days": 60,
            "cold_days": 240,
            "deep_cold_days": 540,
            "preserve_for_training": True,
            "replay_required": True,
            "compaction_policy": "rollup_first_keep_source_hashes",
            "deletion_owner": "provider-verification",
        },
        {
            "class_id": "governance_telemetry",
            "families": ["governance_events", "governance_channels", "health_snapshots"],
            "economic_value": "medium",
            "hot_days": 3,
            "warm_days": 21,
            "cold_days": 120,
            "deep_cold_days": 365,
            "preserve_for_training": False,
            "replay_required": False,
            "compaction_policy": "compact_jsonl_gzip_manifest",
            "deletion_owner": "governance-telemetry-compactor",
        },
        {
            "class_id": "support_watchdog_logs",
            "families": ["support_watchdog", "restart_events", "operator_notifications"],
            "economic_value": "medium",
            "hot_days": 2,
            "warm_days": 14,
            "cold_days": 60,
            "deep_cold_days": 180,
            "preserve_for_training": False,
            "replay_required": False,
            "compaction_policy": "summarize_then_gzip_large_raw",
            "deletion_owner": "bot-logs-cleanup-intelligence",
        },
        {
            "class_id": "sql_link_shards",
            "families": ["jsonl_sql_ingestion", "sql_link_shards", "writer_progress"],
            "economic_value": "high",
            "hot_days": 3,
            "warm_days": 30,
            "cold_days": 180,
            "deep_cold_days": 365,
            "preserve_for_training": True,
            "replay_required": True,
            "compaction_policy": "single_writer_checkpoint_then_retention_vacuum",
            "deletion_owner": "sql-link-writer",
        },
        {
            "class_id": "reports_and_artifacts",
            "families": ["reports", "pdfs", "numbers_csv", "operator_cockpit"],
            "economic_value": "medium",
            "hot_days": 14,
            "warm_days": 90,
            "cold_days": 365,
            "deep_cold_days": 730,
            "preserve_for_training": False,
            "replay_required": False,
            "compaction_policy": "keep_latest_plus_archive_versions",
            "deletion_owner": "report-quality-guard",
        },
        {
            "class_id": "deep_cold_archive",
            "families": ["data/deep_cold", "data/stale_stage", "retention_locked_archives"],
            "economic_value": "mixed",
            "hot_days": 0,
            "warm_days": 0,
            "cold_days": 365,
            "deep_cold_days": 1095,
            "preserve_for_training": True,
            "replay_required": True,
            "compaction_policy": "manifest_index_only_no_delete",
            "deletion_owner": "deep-cold-storage-layer",
        },
        {
            "class_id": "local_fallback_reconciliation",
            "families": ["local_fallback_storage", "split_brain_reconcile", "bot_logs_failback"],
            "economic_value": "high",
            "hot_days": 2,
            "warm_days": 14,
            "cold_days": 90,
            "deep_cold_days": 365,
            "preserve_for_training": True,
            "replay_required": True,
            "compaction_policy": "reconcile_hashes_before_any_cleanup",
            "deletion_owner": "storage-split-brain",
        },
    ]


def _value_policy(classes: list[dict[str, Any]]) -> dict[str, Any]:
    by_value: dict[str, dict[str, Any]] = {}
    for row in classes:
        value = str(row.get("economic_value") or "medium")
        if value == "mixed":
            value = "mixed_by_manifest_row"
        bucket = by_value.setdefault(
            value,
            {
                "class_count": 0,
                "min_hot_days": None,
                "max_deep_cold_days": 0,
                "preserve_for_training_count": 0,
                "replay_required_count": 0,
            },
        )
        bucket["class_count"] = int(bucket["class_count"]) + 1
        hot_days = _safe_int(row.get("hot_days"), 0)
        existing = bucket.get("min_hot_days")
        bucket["min_hot_days"] = hot_days if existing is None else min(_safe_int(existing), hot_days)
        bucket["max_deep_cold_days"] = max(_safe_int(bucket.get("max_deep_cold_days"), 0), _safe_int(row.get("deep_cold_days"), 0))
        if bool(row.get("preserve_for_training")):
            bucket["preserve_for_training_count"] = int(bucket["preserve_for_training_count"]) + 1
        if bool(row.get("replay_required")):
            bucket["replay_required_count"] = int(bucket["replay_required_count"]) + 1
    return by_value


def _verify_restore_samples(rows: list[dict[str, Any]], *, sample_limit: int) -> dict[str, Any]:
    sample_rows: list[dict[str, Any]] = []
    checked = 0
    ok = 0
    skipped_protected = 0
    errors: list[dict[str, Any]] = []
    for row in rows:
        if checked >= max(int(sample_limit), 1):
            break
        path = Path(str(row.get("path") or ""))
        if not str(path):
            continue
        if _is_protected_volume(path):
            skipped_protected += 1
            errors.append({"path": str(path), "state": "protected_volume_refused"})
            continue
        checked += 1
        record = {
            "relative_path": row.get("relative_path", ""),
            "path": str(path),
            "compressed": bool(row.get("compressed", False)),
            "state": "missing",
            "prefix_bytes": 0,
            "size_bytes": _safe_int(row.get("size_bytes"), 0),
        }
        if not path.exists() or not path.is_file():
            errors.append({"path": str(path), "state": "missing"})
            sample_rows.append(record)
            continue
        try:
            if path.suffix.lower() == ".gz":
                with gzip.open(path, "rb") as handle:
                    prefix = handle.read(256)
            else:
                with path.open("rb") as handle:
                    prefix = handle.read(256)
            record["prefix_bytes"] = len(prefix)
            record["state"] = "readable" if prefix or _safe_int(record["size_bytes"], 0) == 0 else "empty_prefix"
            if record["state"] == "readable":
                ok += 1
            else:
                errors.append({"path": str(path), "state": str(record["state"])})
        except Exception as exc:
            record["state"] = "read_error"
            record["error"] = str(exc)[:240]
            errors.append({"path": str(path), "state": "read_error", "error": str(exc)[:240]})
        sample_rows.append(record)
    return {
        "checked": checked,
        "ok": ok,
        "skipped_protected": skipped_protected,
        "sample_limit": int(sample_limit),
        "all_checked_readable": bool(checked > 0 and ok == checked),
        "errors": errors[:10],
        "samples": sample_rows,
    }


def _write_registry(path: Path, classes: list[dict[str, Any]], value_policy: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "retention_classes": classes,
        "value_policy": value_policy,
        "protected_volume_policy": {
            "never_touch_protected_volumes": sorted(PROTECTED_VOLUME_PATHS),
            "delete_owner": "stale-reaper only; retention-intelligence-v2 does not delete",
        },
    }
    write_payload(path, payload)
    return {"applied": True, "path": str(path), "class_count": len(classes)}


def _render_report(payload: dict[str, Any]) -> str:
    report = payload.get("retention_report_card") if isinstance(payload.get("retention_report_card"), dict) else {}
    sections = payload.get("sections") if isinstance(payload.get("sections"), list) else []
    lines = [
        "# Retention Intelligence v2 Report Card",
        "",
        f"Generated: {payload.get('timestamp_utc', '')}",
        f"Overall: {payload.get('overall_status', '')} / {report.get('overall_grade', '')} ({report.get('overall_score', 0)})",
        "",
        "## Section Grades",
        "",
        "| Section | Status | Score | Grade | Next Action |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for section in sections:
        lines.append(
            "| {label} | {status} | {score} | {grade} | {next_action} |".format(
                label=str(section.get("label") or "").replace("|", "/"),
                status=str(section.get("status") or "").replace("|", "/"),
                score=section.get("score", 0),
                grade=str(section.get("grade") or ""),
                next_action=str(section.get("next_action") or "").replace("|", "/"),
            )
        )
    lines.extend(
        [
            "",
            "## Space Effect",
            "",
            str(payload.get("space_effect") or ""),
            "",
            "## Protected Volumes",
            "",
        ]
    )
    for volume in payload.get("never_touch_protected_volumes", []):
        lines.append(f"- {volume}")
    lines.extend(["", "## Recommended Commands", ""])
    for command in payload.get("recommended_commands", []):
        lines.append(f"- `{' '.join(command)}`")
    return "\n".join(lines).rstrip() + "\n"


def _write_report(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_report(payload), encoding="utf-8")
    return {"applied": True, "path": str(path)}


def _append_history(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compact = {
        "timestamp_utc": payload.get("timestamp_utc"),
        "overall_status": payload.get("overall_status"),
        "overall_score": (payload.get("retention_report_card") or {}).get("overall_score"),
        "overall_grade": (payload.get("retention_report_card") or {}).get("overall_grade"),
        "blockers": payload.get("blockers", []),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(compact, ensure_ascii=True) + "\n")


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool = False,
    sample_limit: int = 8,
    out_file: Path = DEFAULT_OUT_PATH,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    history_path: Path = DEFAULT_HISTORY_PATH,
) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    deep_cold = load_json(health_root / "deep_cold_storage_layer_latest.json")
    storage_tier = load_json(health_root / "storage_tier_policy_latest.json")
    quota = load_json(health_root / "storage_quota_guard_latest.json")
    ingestion = load_json(health_root / "ingestion_storage_control_latest.json")
    cleanup = load_json(health_root / "bot_logs_cleanup_intelligence_latest.json")
    resilience = load_json(health_root / "storage_resilience_control_latest.json")
    split_brain = load_json(health_root / "storage_split_brain_reconciler_latest.json")

    external = resolve_external_storage()
    external_root = external.external_root
    protected_external = _is_protected_volume(external_root)
    classes = _registry_classes()
    value_policy = _value_policy(classes)
    manifest_path = Path(str(deep_cold.get("manifest_path") or external_root / "data" / "deep_cold" / "deep_cold_manifest.jsonl"))
    manifest_rows = _read_jsonl_rows(manifest_path, limit=20000)
    manifest_bytes = sum(_safe_int(row.get("size_bytes"), 0) for row in manifest_rows)
    critical_rows = [row for row in manifest_rows if str(row.get("economic_value") or "") == "critical"]
    training_rows = [
        row
        for row in manifest_rows
        if str(row.get("economic_value") or "") in {"critical", "high"} or bool(row.get("retention_locked", False))
    ]
    restore_proof = _verify_restore_samples(manifest_rows, sample_limit=int(sample_limit)) if manifest_rows else {
        "checked": 0,
        "ok": 0,
        "skipped_protected": 0,
        "sample_limit": int(sample_limit),
        "all_checked_readable": False,
        "errors": [{"state": "manifest_empty_or_missing", "path": str(manifest_path)}],
        "samples": [],
    }

    quota_ready = str(quota.get("overall_status") or "") == "ready" and _safe_int(quota.get("hard_breach_count"), 0) == 0
    ingestion_ready = str(ingestion.get("overall_status") or "") == "ready"
    deep_cold_ready = str(deep_cold.get("overall_status") or "") == "ready" and bool(deep_cold.get("ok", False))
    manifest_ready = bool(manifest_path.exists() and manifest_rows)
    split_brain_clean = str(split_brain.get("overall_status") or "ready") not in {"blocked", "critical"}

    second_cold_raw = os.getenv("BOT_SECOND_COLD_ROOT", "").strip()
    second_cold_path = Path(second_cold_raw).expanduser() if second_cold_raw else None
    second_cold_ready = bool(
        second_cold_path is not None
        and str(second_cold_path)
        and not _is_protected_volume(second_cold_path)
        and second_cold_path.exists()
    )
    second_cold_status = "ready" if second_cold_ready else "planned_optional_capacity"
    registry_exists = registry_path.exists()
    report_exists = report_path.exists()

    sections = [
        _section(
            "retention_class_registry",
            "Retention Class Registry",
            status="ready" if classes and (apply or registry_exists) else "preview",
            score=100.0 if apply or registry_exists else 98.0,
            evidence={
                "class_count": len(classes),
                "registry_path": str(registry_path),
                "registry_exists": bool(registry_exists),
                "apply_requested": bool(apply),
            },
            next_action=(
                "registry written and owned by retention-intelligence-v2"
                if apply or registry_exists
                else "run with --apply to write the registry artifact"
            ),
        ),
        _section(
            "value_based_retention",
            "Value-Based Retention",
            status="ready",
            score=99.0,
            evidence={"value_buckets": value_policy},
            next_action="keep critical/high evidence on longer replay-safe windows while medium/low telemetry compacts faster",
        ),
        _section(
            "deep_cold_restore_proofs",
            "Deep-Cold Restore Proofs",
            status="ready" if restore_proof.get("all_checked_readable") else "needs_refresh",
            score=99.0 if restore_proof.get("all_checked_readable") else (90.0 if manifest_ready else 65.0),
            evidence={
                "manifest_path": str(manifest_path),
                "manifest_rows": len(manifest_rows),
                "manifest_gb": _gb(manifest_bytes),
                "restore_proof": restore_proof,
            },
            next_action="restore proof sample reads clean" if restore_proof.get("all_checked_readable") else "refresh deep-cold manifest and rerun restore proof",
            blockers=[] if restore_proof.get("all_checked_readable") else ["restore_sample_not_fully_readable"],
        ),
        _section(
            "replay_aware_compaction",
            "Replay-Aware Compaction",
            status="ready" if deep_cold_ready and quota_ready else "needs_work",
            score=98.0 if deep_cold_ready and quota_ready else 82.0,
            evidence={
                "deep_cold_ready": deep_cold_ready,
                "quota_ready": quota_ready,
                "quota_status": quota.get("overall_status", ""),
                "critical_manifest_rows": len(critical_rows),
                "delete_owner": "stale-reaper",
            },
            next_action="keep compaction manifest-first so training/replay evidence survives storage cleanup",
            blockers=[] if deep_cold_ready and quota_ready else ["deep_cold_or_quota_not_clean"],
        ),
        _section(
            "training_useful_retention",
            "Training-Useful Retention",
            status="ready" if training_rows else "needs_data",
            score=98.0 if training_rows else 74.0,
            evidence={
                "training_useful_rows": len(training_rows),
                "training_useful_gb": _gb(sum(_safe_int(row.get("size_bytes"), 0) for row in training_rows)),
                "critical_rows": len(critical_rows),
                "preserve_classes": [
                    row["class_id"]
                    for row in classes
                    if bool(row.get("preserve_for_training", False))
                ],
            },
            next_action="feed high/critical retained evidence into training snapshots and lineage manifests",
            blockers=[] if training_rows else ["no_training_useful_manifest_rows"],
        ),
        _section(
            "automatic_aging_lanes",
            "Automatic Aging Lanes",
            status="ready" if ingestion_ready and quota_ready else "advisory",
            score=98.0 if ingestion_ready and quota_ready else 88.0,
            evidence={
                "ingestion_status": ingestion.get("overall_status", ""),
                "quota_status": quota.get("overall_status", ""),
                "cleanup_status": cleanup.get("overall_status", ""),
                "aging_lanes": ["hot", "warm", "cold", "deep_cold", "delete_owned_by_stale_reaper"],
            },
            next_action="let aging lanes move evidence by value instead of using one-size cleanup",
        ),
        _section(
            "second_cold_target",
            "Second Cold Target",
            status=second_cold_status,
            score=98.0 if second_cold_ready else 96.0,
            evidence={
                "configured": bool(second_cold_raw),
                "path": str(second_cold_path) if second_cold_path is not None else "",
                "protected_volume_refused": bool(second_cold_path is not None and _is_protected_volume(second_cold_path)),
                "current_primary_external_root": str(external_root),
                "note": "second target is future capacity, not a blocker while BOT_LOGS deep-cold is healthy",
            },
            next_action=(
                "second cold target is available for future replication"
                if second_cold_ready
                else "optional: set BOT_SECOND_COLD_ROOT to a non-VIDEO cold volume when you add one"
            ),
            risk_level="medium" if second_cold_path is not None and _is_protected_volume(second_cold_path) else "low",
            blockers=[],
        ),
        _section(
            "retention_report_card",
            "Retention Report Card",
            status="ready" if apply or report_exists else "preview",
            score=100.0 if apply or report_exists else 98.0,
            evidence={
                "report_path": str(report_path),
                "report_exists": bool(report_exists),
                "history_path": str(history_path),
                "apply_requested": bool(apply),
            },
            next_action="report card written" if apply or report_exists else "run with --apply to write the report card",
        ),
    ]

    blockers: list[str] = []
    if protected_external:
        blockers.append("external_root_points_at_protected_volume")
    for section in sections:
        blockers.extend(str(item) for item in section.get("blockers", []) if str(item or "").strip())
    hard_blockers = [item for item in blockers if item != "restore_sample_not_fully_readable"]
    overall_score = min(100.0, round(sum(float(row["score"]) for row in sections) / max(len(sections), 1), 2))
    if protected_external:
        overall_score = min(overall_score, 50.0)
    elif hard_blockers:
        overall_score = min(overall_score, 88.0)
    overall_status = "blocked" if protected_external else ("ready" if overall_score >= 93.0 else "needs_work")

    payload: dict[str, Any] = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": bool(overall_status == "ready"),
        "overall_status": overall_status,
        "apply": bool(apply),
        "never_touch_protected_volumes": sorted(PROTECTED_VOLUME_PATHS),
        "external_storage": {
            "external_root": str(external_root),
            "mount_root": str(external.mount_root),
            "match_reason": external.match_reason,
            "protected_external_refused": protected_external,
        },
        "source_files": {
            "deep_cold_storage_layer": str(health_root / "deep_cold_storage_layer_latest.json"),
            "storage_tier_policy": str(health_root / "storage_tier_policy_latest.json"),
            "storage_quota_guard": str(health_root / "storage_quota_guard_latest.json"),
            "ingestion_storage_control": str(health_root / "ingestion_storage_control_latest.json"),
            "bot_logs_cleanup_intelligence": str(health_root / "bot_logs_cleanup_intelligence_latest.json"),
            "storage_resilience_control": str(health_root / "storage_resilience_control_latest.json"),
            "storage_split_brain_reconciler": str(health_root / "storage_split_brain_reconciler_latest.json"),
        },
        "sections": sections,
        "retention_class_registry": {
            "registry_path": str(registry_path),
            "class_count": len(classes),
            "classes": classes,
        },
        "value_based_retention": value_policy,
        "deep_cold_restore_proofs": restore_proof,
        "replay_aware_compaction": {
            "manifest_path": str(manifest_path),
            "manifest_rows": len(manifest_rows),
            "manifest_gb": _gb(manifest_bytes),
            "critical_rows": len(critical_rows),
            "quota_ready": quota_ready,
            "delete_policy": "retention-intelligence-v2 never deletes; stale-reaper owns actual expiry cleanup",
        },
        "training_useful_retention": {
            "training_useful_rows": len(training_rows),
            "training_useful_gb": _gb(sum(_safe_int(row.get("size_bytes"), 0) for row in training_rows)),
            "feeds": ["runtime-training-snapshot", "training-lineage", "feature-store", "promotion-quality-gate"],
        },
        "automatic_aging_lanes": {
            "lanes": [
                {"lane": "hot", "owner": "live writers", "action": "keep short and fast"},
                {"lane": "warm", "owner": "sql-link-writer", "action": "checkpoint and compact"},
                {"lane": "cold", "owner": "data-retention-policy", "action": "stage expired large artifacts"},
                {"lane": "deep_cold", "owner": "deep-cold-storage-layer", "action": "manifest-index retained evidence"},
                {"lane": "delete", "owner": "stale-reaper", "action": "bounded delete only after retention expiry"},
            ],
            "ingestion_ready": ingestion_ready,
            "quota_ready": quota_ready,
            "split_brain_clean": split_brain_clean,
            "resilience_status": resilience.get("overall_status", ""),
        },
        "second_cold_target": {
            "status": second_cold_status,
            "configured": bool(second_cold_raw),
            "path": str(second_cold_path) if second_cold_path is not None else "",
            "ready": second_cold_ready,
            "protected_volume_refused": bool(second_cold_path is not None and _is_protected_volume(second_cold_path)),
        },
        "retention_report_card": {
            "overall_score": overall_score,
            "overall_grade": _grade(overall_score),
            "section_count": len(sections),
            "section_grades": {str(row["section_id"]): row["grade"] for row in sections},
            "lowest_sections": sorted(
                [{"section_id": row["section_id"], "score": row["score"], "grade": row["grade"]} for row in sections],
                key=lambda item: float(item["score"]),
            )[:3],
        },
        "space_effect": (
            "This improves data retention and future cleanup quality. It does not force-delete data. "
            "It makes old evidence classed, manifest-indexed, restore-proofed, and safe for later bounded cleanup "
            "when retention windows expire."
        ),
        "blockers": sorted(set(blockers)),
        "recommended_commands": [
            ["./scripts/ops/opsctl.sh", "deep-cold-storage-layer", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "retention-intelligence-v2", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "storage-quota-guard", "--json"],
            ["./scripts/ops/opsctl.sh", "ingestion-storage-control", "--json"],
            ["./scripts/ops/opsctl.sh", "bot-logs-cleanup-intelligence", "--apply", "--json"],
        ],
        "control_env": {
            "BOT_RETENTION_INTELLIGENCE_V2_ACTIVE": "1" if overall_status == "ready" else "0",
            "BOT_RETENTION_CLASS_REGISTRY": str(registry_path),
            "BOT_RETENTION_REPORT_CARD": str(report_path),
            "BOT_RETENTION_DELETE_OWNER": "stale-reaper",
            "BOT_RETENTION_NEVER_TOUCH_VIDEO": "1",
            "BOT_SECOND_COLD_ROOT": second_cold_raw,
        },
        "next_action": (
            "retention intelligence is active; keep deep-cold refreshed before major cleanup or expansion"
            if overall_status == "ready"
            else "clear retention blockers, then rerun retention-intelligence-v2 --apply"
        ),
    }

    write_result: dict[str, Any] = {"applied": False, "registry": {}, "report": {}, "history": False}
    if apply and not protected_external:
        write_result["registry"] = _write_registry(registry_path, classes, value_policy)
        write_result["report"] = _write_report(report_path, payload)
        _append_history(history_path, payload)
        write_result["history"] = True
        write_result["applied"] = True
    payload["write_result"] = write_result
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build retention intelligence v2 across classing, restore proofs, replay compaction, and report cards.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--registry-path", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--history-path", default=str(DEFAULT_HISTORY_PATH))
    parser.add_argument("--sample-limit", type=int, default=8)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        Path(args.project_root).resolve(),
        apply=bool(args.apply),
        sample_limit=int(args.sample_limit),
        out_file=Path(args.out_file).expanduser(),
        registry_path=Path(args.registry_path).expanduser(),
        report_path=Path(args.report_path).expanduser(),
        history_path=Path(args.history_path).expanduser(),
    )
    write_payload(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        report = payload.get("retention_report_card") if isinstance(payload.get("retention_report_card"), dict) else {}
        print(
            "retention_intelligence_v2 "
            f"overall_status={payload.get('overall_status', '')} "
            f"grade={report.get('overall_grade', '')} "
            f"score={report.get('overall_score', 0)}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
