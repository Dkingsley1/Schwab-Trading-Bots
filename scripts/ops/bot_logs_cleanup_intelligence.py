#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from core.storage_mounts import resolve_external_storage
    from scripts.ops.long_runtime_common import iso_now, load_json, ordered_unique, write_payload
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from core.storage_mounts import resolve_external_storage
    from .long_runtime_common import iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "bot_logs_cleanup_intelligence_latest.json"
DEFAULT_HISTORY_PATH = PROJECT_ROOT / "governance" / "health" / "bot_logs_cleanup_intelligence_history.jsonl"
DEFAULT_TARGET_FREE_GB = 100.0
DEFAULT_MIN_AGE_HOURS = 12.0
DEFAULT_PREFIX_VERIFY_BYTES = 65536
DEFAULT_FALLBACK_QUARANTINE_ROOT = PROJECT_ROOT / "local_fallback_storage" / "quarantine" / "bot_logs_cleanup"


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


def _disk_snapshot(path: Path) -> dict[str, Any]:
    try:
        usage = shutil.disk_usage(path)
    except Exception:
        return {
            "path": str(path),
            "exists": bool(path.exists()),
            "total_bytes": 0,
            "used_bytes": 0,
            "free_bytes": 0,
            "free_gb": 0.0,
            "used_gb": 0.0,
            "capacity_pct": 0.0,
        }
    capacity_pct = 100.0 * float(usage.used) / max(float(usage.total), 1.0)
    return {
        "path": str(path),
        "exists": bool(path.exists()),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
        "free_gb": _gb(usage.free),
        "used_gb": _gb(usage.used),
        "capacity_pct": round(capacity_pct, 3),
    }


def _file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except Exception:
        return 0


def _file_age_hours(path: Path, *, now: datetime | None = None) -> float:
    try:
        mt = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except Exception:
        return 0.0
    current = now or datetime.now(timezone.utc)
    return max((current - mt).total_seconds() / 3600.0, 0.0)


def _today_tokens(now: datetime | None = None) -> set[str]:
    current = now or datetime.now(timezone.utc)
    tokens = {current.strftime("%Y%m%d")}
    try:
        local = current.astimezone()
        tokens.add(local.strftime("%Y%m%d"))
    except Exception:
        pass
    return tokens


def _protects_current_day(path: Path, *, now: datetime | None = None) -> bool:
    name = path.name
    return any(token in name for token in _today_tokens(now))


def _read_prefix(path: Path, limit: int) -> bytes:
    try:
        with path.open("rb") as handle:
            return handle.read(max(int(limit), 1))
    except Exception:
        return b""


def _gzip_prefix(path: Path, limit: int) -> tuple[bytes, str]:
    try:
        with gzip.open(path, "rb") as handle:
            return handle.read(max(int(limit), 1)), ""
    except Exception as exc:
        return b"", str(exc)


def _gzip_duplicate_verification(raw_path: Path, gz_path: Path, *, prefix_bytes: int) -> dict[str, Any]:
    raw_size = _file_size(raw_path)
    gz_size = _file_size(gz_path)
    if not raw_path.exists() or not gz_path.exists():
        return {"ok": False, "state": "missing_pair", "reason": "raw or gzip path is missing"}
    if raw_size <= 0 or gz_size <= 0:
        return {"ok": False, "state": "empty_file", "reason": "raw or gzip path is empty"}
    raw_prefix = _read_prefix(raw_path, prefix_bytes)
    gz_prefix, error = _gzip_prefix(gz_path, prefix_bytes)
    if error:
        return {"ok": False, "state": "gzip_unreadable", "reason": error[:240]}
    if not raw_prefix or not gz_prefix:
        return {"ok": False, "state": "prefix_unreadable", "reason": "could not read comparable prefixes"}
    if raw_prefix != gz_prefix[: len(raw_prefix)]:
        return {"ok": False, "state": "prefix_mismatch", "reason": "raw and gzip prefixes do not match"}
    return {
        "ok": True,
        "state": "prefix_match",
        "reason": f"first {len(raw_prefix)} bytes match compressed sibling",
    }


def _candidate_family(path: Path, root: Path) -> str:
    try:
        rel = str(path.relative_to(root))
    except Exception:
        rel = str(path)
    if rel.startswith("data/stale_stage/"):
        return "stale_stage"
    if rel.startswith("decision_explanations/"):
        return "decision_explanations"
    if rel.startswith("decisions/"):
        return "decisions"
    if rel.startswith("governance/execution_lanes/"):
        return "execution_lanes"
    if rel.startswith("governance/channels/"):
        return "governance_channels"
    if rel.startswith("governance/"):
        return "governance"
    if rel.startswith("data/"):
        return "data"
    return "other"


def _risk_score(*, tier: int, family: str, current_day: bool, age_hours: float) -> int:
    score = 0
    if tier >= 3:
        score += 4
    if family in {"decisions", "data"}:
        score += 2
    if current_day:
        score += 5
    if age_hours < 24.0:
        score += 2
    return score


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def _scan_duplicate_jsonl_gzip(
    root: Path,
    *,
    min_age_hours: float,
    protect_current_day: bool,
    prefix_verify_bytes: int,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    current = now or datetime.now(timezone.utc)
    for raw_path in sorted(root.rglob("*.jsonl")):
        if not raw_path.is_file():
            continue
        gz_path = raw_path.with_suffix(raw_path.suffix + ".gz")
        if not gz_path.is_file():
            continue
        age_hours = _file_age_hours(raw_path, now=current)
        current_day = _protects_current_day(raw_path, now=current)
        family = _candidate_family(raw_path, root)
        verification = _gzip_duplicate_verification(raw_path, gz_path, prefix_bytes=prefix_verify_bytes)
        eligible = bool(verification.get("ok", False))
        blocked_reasons = []
        if age_hours < float(min_age_hours):
            eligible = False
            blocked_reasons.append("too_recent")
        if protect_current_day and current_day:
            eligible = False
            blocked_reasons.append("current_day_protected")
        if not bool(verification.get("ok", False)):
            blocked_reasons.append(str(verification.get("state") or "verification_failed"))
        raw_size = _file_size(raw_path)
        gz_size = _file_size(gz_path)
        rows.append(
            {
                "tier": 1,
                "tier_name": "lossless_duplicate_raw_jsonl",
                "family": family,
                "relative_path": _relative(raw_path, root),
                "path": str(raw_path),
                "compressed_path": str(gz_path),
                "size_bytes": int(raw_size),
                "compressed_size_bytes": int(gz_size),
                "reclaimable_bytes": int(raw_size),
                "age_hours": round(age_hours, 3),
                "current_day": bool(current_day),
                "eligible": bool(eligible),
                "verification": verification,
                "blocked_reasons": ordered_unique(blocked_reasons),
                "risk_score": _risk_score(tier=1, family=family, current_day=current_day, age_hours=age_hours),
            }
        )
    return rows


def _stale_stage_value(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root / "data" / "stale_stage")
        label = str(rel.parts[0] if rel.parts else "")
    except Exception:
        label = ""
    if label.startswith("decision_explanations") or label == "decision_explanations":
        return "high"
    if label.startswith("decisions") or label == "decisions":
        return "critical"
    if label.startswith("governance") or label == "governance":
        return "medium"
    return "low"


def _value_window_hours(value: str) -> float:
    return {
        "low": 24.0,
        "medium": 5.0 * 24.0,
        "high": 14.0 * 24.0,
        "critical": 45.0 * 24.0,
    }.get(str(value or "").strip().lower(), 14.0 * 24.0)


def _scan_stale_stage(root: Path, *, now: datetime | None = None) -> list[dict[str, Any]]:
    stale_root = root / "data" / "stale_stage"
    rows: list[dict[str, Any]] = []
    if not stale_root.exists():
        return rows
    current = now or datetime.now(timezone.utc)
    for path in sorted(stale_root.rglob("*")):
        if not path.is_file() or path.name == "stale_manifest.jsonl":
            continue
        value = _stale_stage_value(path, root)
        age_hours = _file_age_hours(path, now=current)
        min_age = _value_window_hours(value)
        eligible = age_hours >= min_age
        blocked = [] if eligible else [f"value_window_not_met:{value}"]
        rows.append(
            {
                "tier": 2,
                "tier_name": "stale_stage_reaper",
                "family": "stale_stage",
                "economic_value": value,
                "relative_path": _relative(path, root),
                "path": str(path),
                "size_bytes": _file_size(path),
                "reclaimable_bytes": _file_size(path),
                "age_hours": round(age_hours, 3),
                "min_age_hours": min_age,
                "eligible": bool(eligible),
                "verification": {"ok": bool(eligible), "state": "age_policy", "reason": "stale-stage value window passed" if eligible else "stale-stage value window not met"},
                "blocked_reasons": blocked,
                "risk_score": _risk_score(tier=2, family="stale_stage", current_day=False, age_hours=age_hours),
            }
        )
    return rows


def _local_fallback_canonical_name(name: str) -> str:
    marker = ".local_fallback"
    if marker not in name:
        return name
    return name.split(marker, 1)[0]


def _scan_external_local_fallback_copies(
    root: Path,
    *,
    project_root: Path,
    fallback_quarantine_root: Path,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows

    current = now or datetime.now(timezone.utc)
    try:
        quarantine_resolved = fallback_quarantine_root.resolve(strict=False)
        root_resolved = root.resolve(strict=False)
        quarantine_inside_external = str(quarantine_resolved).startswith(str(root_resolved))
    except Exception:
        quarantine_inside_external = False

    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".local_fallback" not in path.name:
            continue
        rel_path = _relative(path, root)
        canonical_name = _local_fallback_canonical_name(path.name)
        canonical_rel = str(Path(rel_path).with_name(canonical_name))
        local_preservation_path = project_root / "local_fallback_storage" / canonical_rel
        external_canonical_path = root / canonical_rel
        age_hours = _file_age_hours(path, now=current)
        current_day = _protects_current_day(path, now=current)
        family = _candidate_family(external_canonical_path, root)
        destination_path = fallback_quarantine_root / rel_path
        blocked_reasons: list[str] = []
        eligible = True
        verification_state = "quarantine_preserves_copy"
        verification_reason = (
            "external failback conflict copy can be moved to local quarantine before removal from BOT_LOGS"
        )
        if quarantine_inside_external:
            eligible = False
            blocked_reasons.append("quarantine_root_inside_external")
            verification_state = "unsafe_quarantine_root"
            verification_reason = "quarantine root must be outside BOT_LOGS to reclaim space"
        rows.append(
            {
                "tier": 2,
                "tier_name": "external_failback_conflict_quarantine",
                "family": family,
                "relative_path": rel_path,
                "path": str(path),
                "action": "quarantine",
                "destination_path": str(destination_path),
                "canonical_relative_path": canonical_rel,
                "local_preservation_path": str(local_preservation_path),
                "local_preservation_exists": bool(local_preservation_path.exists()),
                "external_canonical_exists": bool(external_canonical_path.exists()),
                "size_bytes": _file_size(path),
                "reclaimable_bytes": _file_size(path),
                "age_hours": round(age_hours, 3),
                "current_day": bool(current_day),
                "eligible": bool(eligible),
                "verification": {
                    "ok": bool(eligible),
                    "state": verification_state,
                    "reason": verification_reason,
                },
                "blocked_reasons": ordered_unique(blocked_reasons),
                "risk_score": 1,
            }
        )
    return rows


def _select_candidates(
    candidates: list[dict[str, Any]],
    *,
    free_bytes: int,
    target_free_bytes: int,
    max_tier: int,
    max_delete_bytes: int,
) -> list[dict[str, Any]]:
    needed = max(int(target_free_bytes) - int(free_bytes), 0)
    if needed <= 0:
        return []
    eligible = [
        row for row in candidates
        if bool(row.get("eligible", False)) and _safe_int(row.get("tier"), 99) <= int(max_tier)
    ]
    eligible.sort(
        key=lambda row: (
            _safe_int(row.get("tier"), 99),
            _safe_int(row.get("risk_score"), 99),
            -_safe_int(row.get("reclaimable_bytes"), 0),
            str(row.get("relative_path") or ""),
        )
    )
    selected: list[dict[str, Any]] = []
    selected_bytes = 0
    max_bytes = max(int(max_delete_bytes), 0)
    for row in eligible:
        reclaimable = _safe_int(row.get("reclaimable_bytes"), 0)
        if reclaimable <= 0:
            continue
        if max_bytes and selected_bytes + reclaimable > max_bytes and selected:
            continue
        selected_row = dict(row)
        selected_row["selected"] = True
        selected.append(selected_row)
        selected_bytes += reclaimable
        if int(free_bytes) + selected_bytes >= int(target_free_bytes):
            break
        if max_bytes and selected_bytes >= max_bytes:
            break
    return selected


def _unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    seq = 1
    while True:
        candidate = path.with_name(f"{path.name}.dupe.{seq}")
        if not candidate.exists():
            return candidate
        seq += 1


def _apply_selected(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deleted_files = 0
    deleted_bytes = 0
    offloaded_files = 0
    offloaded_bytes = 0
    errors: list[dict[str, str]] = []
    deleted_rows: list[dict[str, Any]] = []
    offloaded_rows: list[dict[str, Any]] = []
    for row in rows:
        path = Path(str(row.get("path") or "")).expanduser()
        if not path.exists() or not path.is_file():
            continue
        size_bytes = _file_size(path)
        action = str(row.get("action") or "delete")
        if action == "quarantine":
            destination = _unique_destination(Path(str(row.get("destination_path") or "")).expanduser())
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(destination))
            except Exception as exc:
                errors.append({"path": str(path), "error": str(exc)})
                continue
            offloaded_files += 1
            offloaded_bytes += size_bytes
            offloaded_rows.append(
                {
                    "tier": _safe_int(row.get("tier"), 0),
                    "tier_name": str(row.get("tier_name") or ""),
                    "relative_path": str(row.get("relative_path") or ""),
                    "destination_path": str(destination),
                    "offloaded_bytes": int(size_bytes),
                }
            )
            continue
        try:
            path.unlink()
        except Exception as exc:
            errors.append({"path": str(path), "error": str(exc)})
            continue
        deleted_files += 1
        deleted_bytes += size_bytes
        deleted_rows.append(
            {
                "tier": _safe_int(row.get("tier"), 0),
                "tier_name": str(row.get("tier_name") or ""),
                "relative_path": str(row.get("relative_path") or ""),
                "deleted_bytes": int(size_bytes),
            }
        )
    reclaimed_bytes = int(deleted_bytes + offloaded_bytes)
    return {
        "deleted_files": int(deleted_files),
        "deleted_bytes": int(deleted_bytes),
        "deleted_gb": _gb(deleted_bytes),
        "offloaded_files": int(offloaded_files),
        "offloaded_bytes": int(offloaded_bytes),
        "offloaded_gb": _gb(offloaded_bytes),
        "reclaimed_bytes": int(reclaimed_bytes),
        "reclaimed_gb": _gb(reclaimed_bytes),
        "errors": errors,
        "deleted_rows": deleted_rows[:50],
        "offloaded_rows": offloaded_rows[:50],
    }


def _summarize_candidates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_tier: dict[str, dict[str, Any]] = {}
    by_family: dict[str, dict[str, Any]] = {}
    eligible_bytes = 0
    blocked_count = 0
    for row in rows:
        tier_key = f"tier_{_safe_int(row.get('tier'), 0)}"
        family = str(row.get("family") or "unknown")
        reclaimable = _safe_int(row.get("reclaimable_bytes"), 0)
        for bucket, key in ((by_tier, tier_key), (by_family, family)):
            entry = bucket.setdefault(key, {"files": 0, "eligible_files": 0, "bytes": 0, "eligible_bytes": 0})
            entry["files"] += 1
            entry["bytes"] += reclaimable
            if bool(row.get("eligible", False)):
                entry["eligible_files"] += 1
                entry["eligible_bytes"] += reclaimable
        if bool(row.get("eligible", False)):
            eligible_bytes += reclaimable
        else:
            blocked_count += 1
    return {
        "candidate_count": len(rows),
        "eligible_count": sum(1 for row in rows if bool(row.get("eligible", False))),
        "blocked_count": blocked_count,
        "eligible_bytes": int(eligible_bytes),
        "eligible_gb": _gb(eligible_bytes),
        "by_tier": by_tier,
        "by_family": by_family,
    }


def _top_rows(rows: list[dict[str, Any]], *, limit: int = 20) -> list[dict[str, Any]]:
    out = []
    for row in sorted(rows, key=lambda item: (-_safe_int(item.get("reclaimable_bytes"), 0), str(item.get("relative_path") or "")))[: max(int(limit), 1)]:
        out.append(
            {
                "tier": _safe_int(row.get("tier"), 0),
                "tier_name": str(row.get("tier_name") or ""),
                "family": str(row.get("family") or ""),
                "relative_path": str(row.get("relative_path") or ""),
                "reclaimable_gb": _gb(_safe_int(row.get("reclaimable_bytes"), 0)),
                "eligible": bool(row.get("eligible", False)),
                "blocked_reasons": list(row.get("blocked_reasons") or []),
                "verification_state": str((row.get("verification") or {}).get("state") or ""),
            }
        )
    return out


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    bot_logs_root: Path | None = None,
    apply: bool = False,
    target_free_gb: float = DEFAULT_TARGET_FREE_GB,
    max_tier: int = 1,
    max_delete_gb: float = 0.0,
    min_age_hours: float = DEFAULT_MIN_AGE_HOURS,
    protect_current_day: bool = True,
    prefix_verify_bytes: int = DEFAULT_PREFIX_VERIFY_BYTES,
    fallback_quarantine_root: Path = DEFAULT_FALLBACK_QUARANTINE_ROOT,
    out_path: Path = DEFAULT_OUT_PATH,
    history_path: Path = DEFAULT_HISTORY_PATH,
) -> dict[str, Any]:
    external_root = bot_logs_root or resolve_external_storage().external_root
    disk_before = _disk_snapshot(external_root)
    target_free_bytes = int(max(float(target_free_gb), 0.0) * (1024**3))
    free_bytes = _safe_int(disk_before.get("free_bytes"), 0)
    max_delete_bytes = int(max(float(max_delete_gb), 0.0) * (1024**3))
    if max_delete_bytes <= 0:
        max_delete_bytes = max(target_free_bytes - free_bytes + int(10 * 1024**3), 0)

    duplicate_rows = _scan_duplicate_jsonl_gzip(
        external_root,
        min_age_hours=float(min_age_hours),
        protect_current_day=bool(protect_current_day),
        prefix_verify_bytes=max(int(prefix_verify_bytes), 1),
    )
    fallback_rows = _scan_external_local_fallback_copies(
        external_root,
        project_root=project_root,
        fallback_quarantine_root=fallback_quarantine_root,
    )
    stale_rows = _scan_stale_stage(external_root)
    all_candidates = duplicate_rows + fallback_rows + stale_rows
    selected = _select_candidates(
        all_candidates,
        free_bytes=free_bytes,
        target_free_bytes=target_free_bytes,
        max_tier=max(int(max_tier), 1),
        max_delete_bytes=max_delete_bytes,
    )
    selected_bytes = sum(_safe_int(row.get("reclaimable_bytes"), 0) for row in selected)
    projected_free_bytes = int(free_bytes + selected_bytes)
    cleanup_needed = free_bytes < target_free_bytes
    disk_after = dict(disk_before)
    apply_result = {
        "applied": False,
        "deleted_files": 0,
        "deleted_bytes": 0,
        "deleted_gb": 0.0,
        "offloaded_files": 0,
        "offloaded_bytes": 0,
        "offloaded_gb": 0.0,
        "reclaimed_bytes": 0,
        "reclaimed_gb": 0.0,
        "errors": [],
        "deleted_rows": [],
        "offloaded_rows": [],
    }
    if apply and selected:
        apply_result = {"applied": True, **_apply_selected(selected)}
        disk_after = _disk_snapshot(external_root)
    elif apply:
        apply_result["applied"] = True

    actual_free_bytes = _safe_int(disk_after.get("free_bytes"), projected_free_bytes if not apply else free_bytes)
    comparison_free_bytes = actual_free_bytes if apply else projected_free_bytes
    still_needed_bytes = max(target_free_bytes - comparison_free_bytes, 0)
    if comparison_free_bytes >= target_free_bytes:
        status = "ready"
    elif selected:
        status = "degraded"
    else:
        status = "blocked" if cleanup_needed else "ready"

    selected_top = _top_rows(selected, limit=30)
    candidates_summary = _summarize_candidates(all_candidates)
    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": status == "ready",
        "overall_status": status,
        "apply_requested": bool(apply),
        "bot_logs_root": str(external_root),
        "target_free_gb": round(float(target_free_gb), 3),
        "max_tier": int(max_tier),
        "guardrails": {
            "tier_1": "delete raw .jsonl only when a matching .jsonl.gz sibling exists and the prefix matches",
            "tier_2": "offload external .local_fallback* conflict copies to local quarantine, then delete stale-stage files only after value-based age windows pass",
            "tier_3": "recommend SQL compaction or offload; do not delete stateful SQLite files here",
            "fallback_quarantine_root": str(fallback_quarantine_root),
            "protect_current_day": bool(protect_current_day),
            "min_age_hours": float(min_age_hours),
            "prefix_verify_bytes": int(prefix_verify_bytes),
        },
        "disk_before": disk_before,
        "disk_after": disk_after,
        "cleanup_needed": bool(cleanup_needed),
        "selected_count": len(selected),
        "selected_reclaimable_bytes": int(selected_bytes),
        "selected_reclaimable_gb": _gb(selected_bytes),
        "projected_free_gb": _gb(projected_free_bytes),
        "remaining_to_target_gb": _gb(still_needed_bytes),
        "candidate_summary": candidates_summary,
        "selected_candidates": selected_top,
        "top_candidates": _top_rows(all_candidates, limit=30),
        "apply_result": apply_result,
        "intelligence_layer": {
            "decision": (
                "target_reached" if comparison_free_bytes >= target_free_bytes and apply
                else "ready_to_apply_selected_tiers" if comparison_free_bytes >= target_free_bytes
                else "run_next_tier_or_compact_stateful_sql" if still_needed_bytes > 0 and int(max_tier) < 2
                else "manual_review_required"
            ),
            "pressure_level": (
                "critical" if _safe_float(disk_before.get("capacity_pct"), 0.0) >= 98.0
                else "elevated" if _safe_float(disk_before.get("capacity_pct"), 0.0) >= 90.0
                else "normal"
            ),
            "self_updates": [
                "history rows record applied/deleted bytes so future cleanup can measure which tier actually helped",
                "current-day raw JSONL protection prevents the cleanup layer from racing active writers",
                "external failback conflict copies are offloaded to local quarantine instead of being destroyed",
                "tier selection stops as soon as the target free-space floor is projected or achieved",
            ],
            "next_actions": ordered_unique(
                [
                    "refresh storage-tier-policy and storage-quota-guard after cleanup",
                    "run max-tier 2 only if tier 1 does not recover enough space",
                    "keep autosync disabled or space-gated until BOT_LOGS has enough free space"
                    if len(fallback_rows) > 0 else "",
                    "checkpoint and compact jsonl_link.sqlite3 separately; it is stateful and intentionally outside this delete lane"
                    if still_needed_bytes > 0 else "",
                ]
            ),
        },
    }
    write_payload(out_path, payload)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "timestamp_utc": payload["timestamp_utc"],
                    "apply_requested": bool(apply),
                    "overall_status": status,
                    "bot_logs_root": str(external_root),
                    "free_gb_before": disk_before.get("free_gb"),
                    "free_gb_after": disk_after.get("free_gb"),
                    "selected_reclaimable_gb": payload["selected_reclaimable_gb"],
                    "deleted_gb": apply_result.get("deleted_gb", 0.0),
                    "offloaded_gb": apply_result.get("offloaded_gb", 0.0),
                    "reclaimed_gb": apply_result.get("reclaimed_gb", 0.0),
                    "selected_count": len(selected),
                    "deleted_files": apply_result.get("deleted_files", 0),
                    "offloaded_files": apply_result.get("offloaded_files", 0),
                    "max_tier": int(max_tier),
                },
                ensure_ascii=True,
            )
            + "\n"
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Tiered BOT_LOGS cleanup with guarded cleanup intelligence.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--bot-logs-root", default="")
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--history-file", default=str(DEFAULT_HISTORY_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--target-free-gb", type=float, default=DEFAULT_TARGET_FREE_GB)
    parser.add_argument("--max-tier", type=int, default=1)
    parser.add_argument("--max-delete-gb", type=float, default=0.0)
    parser.add_argument("--min-age-hours", type=float, default=DEFAULT_MIN_AGE_HOURS)
    parser.add_argument("--protect-current-day", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefix-verify-bytes", type=int, default=DEFAULT_PREFIX_VERIFY_BYTES)
    parser.add_argument("--fallback-quarantine-root", default=str(DEFAULT_FALLBACK_QUARANTINE_ROOT))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    bot_logs_root = Path(args.bot_logs_root).expanduser() if str(args.bot_logs_root or "").strip() else None
    payload = build_payload(
        project_root,
        bot_logs_root=bot_logs_root,
        apply=bool(args.apply),
        target_free_gb=float(args.target_free_gb),
        max_tier=max(int(args.max_tier), 1),
        max_delete_gb=float(args.max_delete_gb),
        min_age_hours=float(args.min_age_hours),
        protect_current_day=bool(args.protect_current_day),
        prefix_verify_bytes=max(int(args.prefix_verify_bytes), 1),
        fallback_quarantine_root=Path(args.fallback_quarantine_root).expanduser(),
        out_path=Path(args.out_file).expanduser(),
        history_path=Path(args.history_file).expanduser(),
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "bot_logs_cleanup_intelligence "
            f"overall_status={payload.get('overall_status', '')} "
            f"selected_gb={payload.get('selected_reclaimable_gb', 0)} "
            f"free_after_gb={((payload.get('disk_after') or {}).get('free_gb', 0))}"
        )
    return 0 if str(payload.get("overall_status") or "") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
