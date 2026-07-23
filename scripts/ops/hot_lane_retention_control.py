#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
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
    from scripts.ops.long_runtime_common import iso_now, load_json, write_payload
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from core.storage_mounts import resolve_external_storage
    from .long_runtime_common import iso_now, load_json, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "hot_lane_retention_control_latest.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.hot_lane_retention_override"
DEFAULT_STORAGE_TIER_PATH = PROJECT_ROOT / "governance" / "health" / "storage_tier_policy_latest.json"
PROTECTED_VOLUME_PREFIXES = ("/Volumes/VIDEO",)
DECISION_LOG_RE = re.compile(r"^trade_decisions_(\d{8})\.jsonl$")


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


def _grade(score: float) -> str:
    value = max(min(float(score), 100.0), 0.0)
    if value >= 99.0:
        return "A+"
    if value >= 97.0:
        return "A+"
    if value >= 93.0:
        return "A"
    if value >= 85.0:
        return "B"
    if value >= 75.0:
        return "C"
    if value >= 65.0:
        return "D"
    return "F"


def _real_path_text(path: Path) -> str:
    try:
        return os.path.realpath(str(path.expanduser()))
    except Exception:
        return str(path.expanduser())


def _is_protected(path: Path) -> bool:
    raw = _real_path_text(path)
    return any(raw == prefix or raw.startswith(f"{prefix}/") for prefix in PROTECTED_VOLUME_PREFIXES)


def _nearest_existing_parent(path: Path) -> Path:
    current = path.expanduser()
    for candidate in (current, *current.parents):
        if candidate.exists():
            return candidate
    return current


def _disk_snapshot(path: Path) -> dict[str, Any]:
    candidate = path.expanduser()
    parent = _nearest_existing_parent(candidate)
    protected = _is_protected(candidate)
    if protected:
        return {
            "path": str(candidate),
            "checked_path": str(parent),
            "protected": True,
            "total_gb": 0.0,
            "used_gb": 0.0,
            "free_gb": 0.0,
            "used_percent": 0.0,
        }
    try:
        usage = shutil.disk_usage(parent)
    except Exception:
        return {
            "path": str(candidate),
            "checked_path": str(parent),
            "protected": False,
            "total_gb": 0.0,
            "used_gb": 0.0,
            "free_gb": 0.0,
            "used_percent": 0.0,
        }
    total = max(float(usage.total), 1.0)
    used = float(usage.used)
    return {
        "path": str(candidate),
        "checked_path": str(parent),
        "protected": False,
        "total_gb": round(total / (1024.0**3), 3),
        "used_gb": round(used / (1024.0**3), 3),
        "free_gb": round(float(usage.free) / (1024.0**3), 3),
        "used_percent": round((used / total) * 100.0, 3),
    }


def _file_row(path: Path, *, project_root: Path, now: datetime) -> dict[str, Any]:
    try:
        stat = path.stat()
        size_bytes = int(stat.st_size)
        mtime = datetime.fromtimestamp(float(stat.st_mtime), tz=timezone.utc)
    except Exception:
        size_bytes = 0
        mtime = datetime.fromtimestamp(0, tz=timezone.utc)
    try:
        rel = str(path.relative_to(project_root))
    except Exception:
        rel = str(path)
    match = DECISION_LOG_RE.match(path.name)
    day_utc = match.group(1) if match else ""
    today_utc = now.strftime("%Y%m%d")
    age_minutes = max((now - mtime).total_seconds() / 60.0, 0.0)
    return {
        "relative_path": rel,
        "real_path": _real_path_text(path),
        "size_bytes": size_bytes,
        "size_gb": round(size_bytes / (1024.0**3), 4),
        "mtime_utc": mtime.isoformat(),
        "age_minutes": round(age_minutes, 3),
        "day_utc": day_utc,
        "is_current_day": bool(day_utc == today_utc),
        "protected": _is_protected(path),
    }


def _scan_decision_logs(project_root: Path, *, now: datetime, active_age_minutes: float) -> list[dict[str, Any]]:
    decisions_root = project_root / "decisions"
    if not decisions_root.exists() or _is_protected(decisions_root):
        return []
    rows: list[dict[str, Any]] = []
    try:
        iterator = decisions_root.rglob("trade_decisions_*.jsonl")
        for path in iterator:
            if not path.is_file() or _is_protected(path):
                continue
            row = _file_row(path, project_root=project_root, now=now)
            if not row.get("is_current_day") and _safe_float(row.get("age_minutes"), 999999.0) > float(active_age_minutes):
                continue
            rows.append(row)
    except Exception:
        return rows
    rows.sort(key=lambda item: (-_safe_int(item.get("size_bytes"), 0), str(item.get("relative_path") or "")))
    return rows


def _scan_sql_shards(project_root: Path) -> list[dict[str, Any]]:
    shard_root = project_root / "data" / "sql_link_shards"
    if not shard_root.exists() or _is_protected(shard_root):
        return []
    rows: list[dict[str, Any]] = []
    for path in shard_root.glob("*"):
        if not path.is_file() or _is_protected(path):
            continue
        try:
            size_bytes = int(path.stat().st_size)
        except Exception:
            size_bytes = 0
        if size_bytes <= 0:
            continue
        try:
            rel = str(path.relative_to(project_root))
        except Exception:
            rel = str(path)
        rows.append(
            {
                "relative_path": rel,
                "size_bytes": size_bytes,
                "size_gb": round(size_bytes / (1024.0**3), 4),
                "recommended_action": "checkpoint_compact_or_mirror_stateful_sql_shard",
                "protected": _is_protected(path),
            }
        )
    rows.sort(key=lambda item: (-_safe_int(item.get("size_bytes"), 0), str(item.get("relative_path") or "")))
    return rows


def _storage_tier_summary(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    pressure = payload.get("pressure") if isinstance(payload.get("pressure"), dict) else {}
    top_files = payload.get("top_files") if isinstance(payload.get("top_files"), list) else []
    return {
        "status": str(payload.get("overall_status") or "missing"),
        "timestamp_utc": payload.get("timestamp_utc", ""),
        "live_hot_path_gb": round(_safe_float(pressure.get("live_hot_path_bytes"), 0.0) / (1024.0**3), 4),
        "hot_path_over_budget_gb": round(_safe_float(pressure.get("hot_path_over_budget_bytes"), 0.0) / (1024.0**3), 4),
        "top_files": top_files[:8],
    }


def _mode_for_pressure(
    *,
    free_gb: float,
    target_free_gb: float,
    pressure_free_gb: float,
    current_day_decision_gb: float,
    top_current_day_gb: float,
    storage_tier_status: str = "",
    live_hot_path_gb: float = 0.0,
    hot_path_over_budget_gb: float = 0.0,
    hot_total_thin_gb: float,
    hot_file_thin_gb: float,
    restore_total_gb: float,
    restore_file_gb: float,
) -> tuple[str, str, float, list[str]]:
    reasons: list[str] = []
    storage_blocked = str(storage_tier_status or "").strip() in {"blocked", "critical"}
    if hot_path_over_budget_gb > 0.0:
        reasons.append("storage_tier_hot_path_over_budget")
    if storage_blocked:
        reasons.append("storage_tier_blocked")
    if live_hot_path_gb >= hot_total_thin_gb:
        reasons.append("live_hot_path_large")
    if free_gb <= pressure_free_gb:
        reasons.append("external_free_below_pressure_floor")
    if free_gb <= target_free_gb:
        reasons.append("external_free_below_target")
    if current_day_decision_gb >= hot_total_thin_gb * 1.75:
        reasons.append("current_day_decision_logs_extreme")
    elif current_day_decision_gb >= hot_total_thin_gb:
        reasons.append("current_day_decision_logs_large")
    if top_current_day_gb >= hot_file_thin_gb * 1.75:
        reasons.append("single_decision_lane_extreme")
    elif top_current_day_gb >= hot_file_thin_gb:
        reasons.append("single_decision_lane_large")
    if free_gb <= target_free_gb + 10.0 and current_day_decision_gb >= restore_total_gb:
        reasons.append("free_space_near_target_with_hot_growth")

    if (
        "external_free_below_pressure_floor" in reasons
        or "current_day_decision_logs_extreme" in reasons
        or "single_decision_lane_extreme" in reasons
        or storage_blocked
        or hot_path_over_budget_gb >= max(hot_total_thin_gb * 0.75, 1.0)
    ):
        return "emergency_hot_thin", "critical", 84.0, reasons
    if reasons:
        return "thin_optional_sub_bot_decisions", "active", 94.0, reasons
    if (
        free_gb >= target_free_gb + 25.0
        and current_day_decision_gb <= restore_total_gb
        and top_current_day_gb <= restore_file_gb
        and hot_path_over_budget_gb <= 0.0
        and not storage_blocked
    ):
        return "full_decision_evidence", "ready", 99.0, ["storage_headroom_green"]
    return "watch", "watching", 97.0, ["hot_lane_within_bounds"]


def _env_for_mode(mode: str, *, reasons: list[str], top_lanes: list[dict[str, Any]]) -> dict[str, str]:
    top_names = ",".join(str(row.get("relative_path") or "") for row in top_lanes[:5])
    base = {
        "HOT_LANE_RETENTION_ACTIVE": "1" if mode in {"thin_optional_sub_bot_decisions", "emergency_hot_thin"} else "0",
        "HOT_LANE_RETENTION_MODE": mode,
        "HOT_LANE_RETENTION_REASON": ",".join(reasons[:8]),
        "HOT_LANE_TOP_PRESSURE_LANES": top_names,
        "BOT_NEVER_TOUCH_VIDEO": "1",
        "LOG_MASTER_VARIANT_DECISIONS": "1",
        "LOG_GRAND_MASTER_DECISIONS": "1",
        "LOG_OPTIONS_MASTER_DECISIONS": "1",
        "LOG_FUTURES_MASTER_DECISIONS": "1",
    }
    if mode == "emergency_hot_thin":
        base.update(
            {
                "LOG_SUB_BOT_DECISIONS": "0",
                "LOG_API_CALLS": "0",
                "LOG_LOOP_STATE": "0",
                "LOG_GATE_EVALUATIONS": "0",
                "LOG_GATE_PASSES": "0",
                "LOG_DATA_INGRESS": "0",
                "LOG_DECISION_EXPLANATIONS": "0",
                "LOG_SHADOW_PNL_ATTRIBUTION": "0",
                "DECISION_LOG_FEATURE_MODE": "minimal",
            }
        )
    elif mode == "thin_optional_sub_bot_decisions":
        base.update(
            {
                "LOG_SUB_BOT_DECISIONS": "0",
                "LOG_API_CALLS": "0",
                "LOG_LOOP_STATE": "0",
                "LOG_GATE_EVALUATIONS": "0",
                "LOG_GATE_PASSES": "0",
                "LOG_DATA_INGRESS": "1",
                "LOG_DECISION_EXPLANATIONS": "1",
                "LOG_SHADOW_PNL_ATTRIBUTION": "1",
                "DECISION_LOG_FEATURE_MODE": "essential",
            }
        )
    else:
        base.update(
            {
                "LOG_SUB_BOT_DECISIONS": "1",
                "LOG_API_CALLS": "1",
                "LOG_LOOP_STATE": "1",
                "LOG_GATE_EVALUATIONS": "1",
                "LOG_GATE_PASSES": "1",
                "LOG_DATA_INGRESS": "1",
                "LOG_DECISION_EXPLANATIONS": "1",
                "LOG_SHADOW_PNL_ATTRIBUTION": "1",
                "DECISION_LOG_FEATURE_MODE": "full",
            }
        )
    return base


def _write_override(path: Path, env: dict[str, str], *, payload: dict[str, Any]) -> bool:
    lines = [
        "# Auto-managed by scripts/ops/hot_lane_retention_control.py",
        f"# updated_at_utc={payload.get('timestamp_utc')}",
        "# Active loops read this file dynamically; do not use it for manual storage routes.",
    ]
    for key, value in sorted(env.items()):
        lines.append(f"{key}={shlex.quote(str(value))}")
    content = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if current == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool = False,
    target_free_gb: float = 125.0,
    pressure_free_gb: float = 64.0,
    hot_total_thin_gb: float = 100.0,
    hot_file_thin_gb: float = 12.0,
    restore_total_gb: float = 40.0,
    restore_file_gb: float = 8.0,
    active_age_minutes: float = 180.0,
    out_path: Path = DEFAULT_OUT_PATH,
    override_path: Path = DEFAULT_OVERRIDE_PATH,
    storage_tier_path: Path = DEFAULT_STORAGE_TIER_PATH,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    external = resolve_external_storage()
    disk = {
        "external": _disk_snapshot(external.external_root),
        "internal_project": _disk_snapshot(project_root),
    }
    decision_rows = _scan_decision_logs(project_root, now=now, active_age_minutes=float(active_age_minutes))
    current_day_rows = [row for row in decision_rows if bool(row.get("is_current_day"))]
    active_rows = current_day_rows or decision_rows
    current_day_bytes = sum(_safe_int(row.get("size_bytes"), 0) for row in active_rows)
    top_current_day_bytes = max((_safe_int(row.get("size_bytes"), 0) for row in active_rows), default=0)
    current_day_decision_gb = current_day_bytes / (1024.0**3)
    top_current_day_gb = top_current_day_bytes / (1024.0**3)
    free_gb = _safe_float(disk["external"].get("free_gb"), 0.0)
    storage_tier = _storage_tier_summary(storage_tier_path)
    storage_tier_status = str(storage_tier.get("status") or "").strip()
    live_hot_path_gb = _safe_float(storage_tier.get("live_hot_path_gb"), 0.0)
    hot_path_over_budget_gb = _safe_float(storage_tier.get("hot_path_over_budget_gb"), 0.0)

    mode, status, score, reasons = _mode_for_pressure(
        free_gb=free_gb,
        target_free_gb=float(target_free_gb),
        pressure_free_gb=float(pressure_free_gb),
        current_day_decision_gb=current_day_decision_gb,
        top_current_day_gb=top_current_day_gb,
        storage_tier_status=storage_tier_status,
        live_hot_path_gb=live_hot_path_gb,
        hot_path_over_budget_gb=hot_path_over_budget_gb,
        hot_total_thin_gb=float(hot_total_thin_gb),
        hot_file_thin_gb=float(hot_file_thin_gb),
        restore_total_gb=float(restore_total_gb),
        restore_file_gb=float(restore_file_gb),
    )
    env = _env_for_mode(mode, reasons=reasons, top_lanes=active_rows)

    protected_external = bool(disk["external"].get("protected", False))
    if protected_external:
        status = "blocked"
        score = min(score, 50.0)
        reasons.append("external_storage_points_at_VIDEO")

    payload: dict[str, Any] = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": bool(status in {"ready", "watching", "active", "critical"}),
        "overall_status": status,
        "overall_score": round(score, 2),
        "overall_grade": _grade(score),
        "apply": bool(apply),
        "mode": mode,
        "reasons": reasons,
        "disk": disk,
        "thresholds": {
            "target_free_gb": round(float(target_free_gb), 3),
            "pressure_free_gb": round(float(pressure_free_gb), 3),
            "hot_total_thin_gb": round(float(hot_total_thin_gb), 3),
            "hot_file_thin_gb": round(float(hot_file_thin_gb), 3),
            "restore_total_gb": round(float(restore_total_gb), 3),
            "restore_file_gb": round(float(restore_file_gb), 3),
            "active_age_minutes": round(float(active_age_minutes), 3),
        },
        "hot_decision_pressure": {
            "active_file_count": len(active_rows),
            "current_day_file_count": len(current_day_rows),
            "active_decision_gb": round(current_day_decision_gb, 4),
            "largest_active_file_gb": round(top_current_day_gb, 4),
            "top_lanes": active_rows[:10],
        },
        "sql_shard_pressure": {
            "top_shards": _scan_sql_shards(project_root)[:8],
            "action": "stateful SQL shards should be checkpointed/compacted/mirrored by writer tooling, not deleted by retention",
        },
        "storage_tier_policy": storage_tier,
        "control_env": env,
        "override_path": str(override_path),
        "override_applied": False,
        "safety_contract": {
            "deletes_active_files": False,
            "compacts_active_current_day_logs": False,
            "thins_optional_sub_bot_decision_logging": mode in {"thin_optional_sub_bot_decisions", "emergency_hot_thin"},
            "keeps_master_evidence": True,
            "never_touch_protected_volumes": list(PROTECTED_VOLUME_PREFIXES),
        },
        "when_to_stop": (
            "restore full decision evidence when external free space is at least target+25GB and active decision logs are below restore thresholds"
        ),
        "recommended_commands": {
            "apply_hot_lane_control": ["./scripts/ops/opsctl.sh", "hot-lane-retention-control", "--apply", "--json"],
            "check_storage_tier": ["./scripts/ops/opsctl.sh", "storage-tier-policy", "--json"],
            "repeat_unison": ["./scripts/ops/opsctl.sh", "storage-retention-unison", "--apply", "--json"],
        },
        "next_action": (
            "hot-lane retention is active; keep master evidence and let optional sub-bot logs thin until free space/headroom recover"
            if mode in {"thin_optional_sub_bot_decisions", "emergency_hot_thin"}
            else "hot-lane retention is watching; full decision evidence can stay on"
        ),
    }
    if apply:
        payload["override_applied"] = _write_override(override_path, env, payload=payload)
    write_payload(out_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Control active hot-lane decision-log growth without deleting current-day files.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--override-file", default=str(DEFAULT_OVERRIDE_PATH))
    parser.add_argument("--storage-tier-file", default=str(DEFAULT_STORAGE_TIER_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--target-free-gb", type=float, default=float(os.getenv("HOT_LANE_TARGET_FREE_GB", "125.0")))
    parser.add_argument("--pressure-free-gb", type=float, default=float(os.getenv("HOT_LANE_PRESSURE_FREE_GB", "64.0")))
    parser.add_argument("--hot-total-thin-gb", type=float, default=float(os.getenv("HOT_LANE_TOTAL_THIN_GB", "100.0")))
    parser.add_argument("--hot-file-thin-gb", type=float, default=float(os.getenv("HOT_LANE_FILE_THIN_GB", "12.0")))
    parser.add_argument("--restore-total-gb", type=float, default=float(os.getenv("HOT_LANE_RESTORE_TOTAL_GB", "40.0")))
    parser.add_argument("--restore-file-gb", type=float, default=float(os.getenv("HOT_LANE_RESTORE_FILE_GB", "8.0")))
    parser.add_argument("--active-age-minutes", type=float, default=float(os.getenv("HOT_LANE_ACTIVE_AGE_MINUTES", "180.0")))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        Path(args.project_root).resolve(),
        apply=bool(args.apply),
        target_free_gb=float(args.target_free_gb),
        pressure_free_gb=float(args.pressure_free_gb),
        hot_total_thin_gb=float(args.hot_total_thin_gb),
        hot_file_thin_gb=float(args.hot_file_thin_gb),
        restore_total_gb=float(args.restore_total_gb),
        restore_file_gb=float(args.restore_file_gb),
        active_age_minutes=float(args.active_age_minutes),
        out_path=Path(args.out_file).expanduser(),
        override_path=Path(args.override_file).expanduser(),
        storage_tier_path=Path(args.storage_tier_file).expanduser(),
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "hot_lane_retention_control "
            f"status={payload.get('overall_status', '')} "
            f"mode={payload.get('mode', '')} "
            f"grade={payload.get('overall_grade', '')} "
            f"active_decision_gb={(payload.get('hot_decision_pressure') or {}).get('active_decision_gb', 0)}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
