#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "data_collection_storage_guard_latest.json"
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "master_bot_registry.json"
DEFAULT_EXTERNAL_ROOT = Path("/Volumes/BOT_LOGS/schwab_trading_bot")


def _disk_usage(path: Path) -> dict[str, Any]:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        usage = os.statvfs(str(probe))
    except Exception:
        return {
            "path": str(path),
            "probe_path": str(probe),
            "available_bytes": 0,
            "total_bytes": 0,
            "used_bytes": 0,
            "used_ratio": 1.0,
            "ok": False,
        }
    total = int(usage.f_frsize * usage.f_blocks)
    available = int(usage.f_frsize * usage.f_bavail)
    used = max(total - available, 0)
    return {
        "path": str(path),
        "probe_path": str(probe),
        "available_bytes": available,
        "total_bytes": total,
        "used_bytes": used,
        "used_ratio": (float(used) / float(total)) if total > 0 else 1.0,
        "ok": True,
    }


def _gb(raw: int | float) -> float:
    return float(raw) / float(1024**3)


def _mode_for_space(*, available_gb: float, used_ratio: float, warn_gb: float, throttle_gb: float, critical_gb: float) -> str:
    if available_gb <= critical_gb or used_ratio >= 0.98:
        return "critical"
    if available_gb <= throttle_gb or used_ratio >= 0.94:
        return "throttle"
    if available_gb <= warn_gb or used_ratio >= 0.90:
        return "watch"
    return "normal"


def _collector_kind(row: dict[str, Any]) -> str:
    kind = str(row.get("slot_kind") or "").strip().lower()
    role = str(row.get("bot_role") or "").strip().lower()
    label_contract = str(row.get("data_label_contract_version") or "").strip().lower()
    collections = ",".join(str(item or "").strip().lower() for item in list(row.get("data_intake_collections") or []))
    if (
        "quant" in kind
        or label_contract.startswith("quant_")
        or any(token in collections for token in ("mlx_library", "mlx_graph", "mlx_snn", "mlx_vision", "esig_rough_path"))
    ):
        return "quant_research"
    if "aggressive_intraday" in kind:
        return "aggressive_intraday"
    if role == "options_sub_bot" or "options" in kind:
        return "options"
    if role == "infrastructure_sub_bot":
        return "infrastructure"
    return "standard"


def _guard_profile(mode: str, kind: str) -> dict[str, Any]:
    if mode == "normal":
        if kind == "quant_research":
            return {
                "capture_mode": "sampled",
                "max_daily_storage_mb": 80,
                "freshness_floor_seconds": 900,
                "retention_profile": "hot_quant_sampled_2d_warm_30d",
                "sample_rate": 0.35,
            }
        return {
            "capture_mode": "full",
            "max_daily_storage_mb": 250 if kind == "aggressive_intraday" else 150,
            "freshness_floor_seconds": 60 if kind == "aggressive_intraday" else (180 if kind == "options" else 300),
            "retention_profile": "",
            "sample_rate": 1.0,
        }
    if mode == "watch":
        if kind == "quant_research":
            return {
                "capture_mode": "thin_sample",
                "max_daily_storage_mb": 45,
                "freshness_floor_seconds": 1200,
                "retention_profile": "hot_quant_thin_1d_warm_21d",
                "sample_rate": 0.18,
            }
        return {
            "capture_mode": "sampled",
            "max_daily_storage_mb": 100 if kind == "aggressive_intraday" else 80,
            "freshness_floor_seconds": 180 if kind == "aggressive_intraday" else (300 if kind == "options" else 600),
            "retention_profile": "hot_sampled_3d_warm_45d",
            "sample_rate": 0.5,
        }
    if mode == "throttle":
        if kind == "quant_research":
            return {
                "capture_mode": "metadata_only",
                "max_daily_storage_mb": 20,
                "freshness_floor_seconds": 2400,
                "retention_profile": "hot_quant_metadata_12h_warm_14d",
                "sample_rate": 0.08,
            }
        return {
            "capture_mode": "thin_sample",
            "max_daily_storage_mb": 50 if kind == "aggressive_intraday" else 40,
            "freshness_floor_seconds": 600 if kind == "aggressive_intraday" else (900 if kind == "options" else 1200),
            "retention_profile": "hot_thin_1d_warm_30d",
            "sample_rate": 0.2,
        }
    return {
        "capture_mode": "metadata_only",
        "max_daily_storage_mb": 10 if kind == "quant_research" else (15 if kind == "aggressive_intraday" else 20),
        "freshness_floor_seconds": 1800 if kind == "aggressive_intraday" else (1800 if kind == "options" else 3600),
        "retention_profile": "hot_quant_metadata_6h_warm_7d" if kind == "quant_research" else "hot_metadata_12h_warm_14d",
        "sample_rate": 0.03 if kind == "quant_research" else 0.05,
    }


def _compute_guard_floor(row: dict[str, Any]) -> dict[str, Any] | None:
    mode = str(row.get("data_collection_compute_guard_mode") or "").strip().lower()
    if mode == "protect_live":
        return {
            "capture_mode": "thin_sample",
            "sample_rate": 0.15,
            "max_daily_storage_mb": 35,
            "freshness_floor_seconds": 1800,
            "reason": "compute_guard=protect_live",
        }
    if mode == "sustain":
        return {
            "capture_mode": "sampled",
            "sample_rate": 0.3,
            "max_daily_storage_mb": 60,
            "freshness_floor_seconds": 900,
            "reason": "compute_guard=sustain",
        }
    if mode == "soft_cap":
        return {
            "capture_mode": "sampled",
            "sample_rate": 0.5,
            "max_daily_storage_mb": 90,
            "freshness_floor_seconds": 600,
            "reason": "compute_guard=soft_cap",
        }
    return None


def _registry_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("sub_bots") if isinstance(payload.get("sub_bots"), list) else []
    return [row for row in rows if isinstance(row, dict)]


def _refresh_summary(payload: dict[str, Any]) -> None:
    rows = _registry_rows(payload)
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    active_rows = [row for row in rows if bool(row.get("active", False))]
    summary["total_bots"] = len(rows)
    summary["active_bots"] = len(active_rows)
    summary["inactive_bots"] = max(len(rows) - len(active_rows), 0)
    summary["active_signal_sub_bots"] = sum(1 for row in active_rows if str(row.get("bot_role") or "") == "signal_sub_bot")
    summary["active_infrastructure_sub_bots"] = sum(1 for row in active_rows if str(row.get("bot_role") or "") == "infrastructure_sub_bot")
    summary["active_options_sub_bots"] = sum(1 for row in active_rows if str(row.get("bot_role") or "") == "options_sub_bot")
    summary["inactive_signal_sub_bots"] = sum(
        1 for row in rows if not bool(row.get("active", False)) and str(row.get("bot_role") or "") == "signal_sub_bot"
    )
    summary["inactive_infrastructure_sub_bots"] = sum(
        1 for row in rows if not bool(row.get("active", False)) and str(row.get("bot_role") or "") == "infrastructure_sub_bot"
    )
    summary["inactive_options_sub_bots"] = sum(
        1 for row in rows if not bool(row.get("active", False)) and str(row.get("bot_role") or "") == "options_sub_bot"
    )
    summary["data_collection_only_bots"] = sum(1 for row in rows if str(row.get("lifecycle_state") or "") == "data_collection_only")
    summary["training_excluded_bots"] = sum(1 for row in rows if bool(row.get("training_excluded", False)))
    summary["storage_guarded_collectors"] = sum(1 for row in rows if bool(row.get("data_collection_storage_guarded", False)))
    summary["storage_guard_metadata_only_collectors"] = sum(
        1 for row in rows if str(row.get("data_collection_capture_mode") or "") == "metadata_only"
    )
    payload["summary"] = summary
    payload["updated_at_utc"] = iso_now()


def _duplicate_fallback_files(root: Path, *, limit: int = 50000) -> list[Path]:
    if not root.exists():
        return []
    out: list[Path] = []
    for path in root.rglob("*.local_fallback*"):
        if path.is_file():
            out.append(path)
            if len(out) >= limit:
                break
    return out


def build_payload(
    *,
    external_root: Path,
    registry_path: Path,
    warn_gb: float,
    throttle_gb: float,
    critical_gb: float,
    apply: bool,
    cleanup_duplicates: bool,
) -> dict[str, Any]:
    disk = _disk_usage(external_root)
    available_gb = _gb(int(disk.get("available_bytes") or 0))
    used_ratio = float(disk.get("used_ratio") or 1.0)
    mode = _mode_for_space(
        available_gb=available_gb,
        used_ratio=used_ratio,
        warn_gb=warn_gb,
        throttle_gb=throttle_gb,
        critical_gb=critical_gb,
    )
    registry = load_json(registry_path)
    rows = _registry_rows(registry)
    collectors = [
        row
        for row in rows
        if bool(row.get("active", False))
        and bool(row.get("data_collection_active", False))
        and str(row.get("lifecycle_state") or "") == "data_collection_only"
    ]
    changes: list[dict[str, Any]] = []
    now = iso_now()
    for row in collectors:
        kind = _collector_kind(row)
        profile = _guard_profile(mode, kind)
        compute_floor = _compute_guard_floor(row)
        if compute_floor:
            profile = {
                **profile,
                "capture_mode": compute_floor["capture_mode"],
                "sample_rate": min(float(profile["sample_rate"]), float(compute_floor["sample_rate"])),
                "max_daily_storage_mb": min(int(profile["max_daily_storage_mb"]), int(compute_floor["max_daily_storage_mb"])),
                "freshness_floor_seconds": max(int(profile["freshness_floor_seconds"]), int(compute_floor["freshness_floor_seconds"])),
            }
        desired = {
            "data_collection_storage_guarded": True,
            "data_collection_storage_guard_mode": mode,
            "data_collection_capture_mode": profile["capture_mode"],
            "data_collection_sample_rate": profile["sample_rate"],
            "data_collection_max_daily_storage_mb": profile["max_daily_storage_mb"],
            "data_collection_storage_guard_updated_utc": now,
            "data_collection_runtime_dependency_profile": (
                "mlx_optional_research_only" if kind == "quant_research" else str(row.get("data_collection_runtime_dependency_profile") or "")
            ),
            "storage_pressure_capture_reason": (
                f"external_available_gb={available_gb:.2f};mode={mode};{compute_floor['reason']}"
                if compute_floor
                else f"external_available_gb={available_gb:.2f};mode={mode}"
            ),
            "freshness_slo_seconds": max(int(row.get("freshness_slo_seconds") or 0), int(profile["freshness_floor_seconds"])),
        }
        if profile["retention_profile"]:
            desired["retention_profile"] = profile["retention_profile"]
        delta = {key: value for key, value in desired.items() if row.get(key) != value}
        if delta:
            changes.append({"bot_id": str(row.get("bot_id") or ""), "kind": kind, "updates": delta})
            if apply:
                row.update(delta)

    duplicate_files = _duplicate_fallback_files(external_root) if cleanup_duplicates else []
    duplicate_bytes = 0
    deleted_duplicates: list[str] = []
    for path in duplicate_files:
        try:
            duplicate_bytes += int(path.stat().st_size)
        except Exception:
            continue
    if apply and cleanup_duplicates:
        for path in duplicate_files:
            try:
                path.unlink()
                deleted_duplicates.append(str(path))
            except Exception:
                continue

    backup_path = ""
    if apply and changes:
        backup = registry_path.parent / "governance" / "lifecycle" / f"master_bot_registry.data_collection_storage_guard_backup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        backup.parent.mkdir(parents=True, exist_ok=True)
        if registry_path.exists():
            backup.write_text(registry_path.read_text(encoding="utf-8"), encoding="utf-8")
            backup_path = str(backup)
        _refresh_summary(registry)
        registry_path.write_text(json.dumps(registry, ensure_ascii=True, indent=2), encoding="utf-8")

    status = "ready" if mode == "normal" else ("degraded" if mode in {"watch", "throttle"} else "blocked")
    return {
        "timestamp_utc": now,
        "schema_version": 1,
        "ok": mode != "critical",
        "overall_status": status,
        "apply_requested": bool(apply),
        "external_root": str(external_root),
        "disk": {**disk, "available_gb": round(available_gb, 3), "used_percent": round(used_ratio * 100.0, 3)},
        "thresholds": {"warn_gb": warn_gb, "throttle_gb": throttle_gb, "critical_gb": critical_gb},
        "guard_mode": mode,
        "collector_count": len(collectors),
        "planned_changes": changes[:200],
        "changed_count": len(changes),
        "registry_backup_path": backup_path,
        "duplicate_cleanup": {
            "enabled": bool(cleanup_duplicates),
            "candidate_count": len(duplicate_files),
            "candidate_bytes": duplicate_bytes,
            "candidate_gb": round(_gb(duplicate_bytes), 3),
            "deleted_count": len(deleted_duplicates),
            "deleted_gb": round(_gb(duplicate_bytes), 3) if apply else 0.0,
        },
        "recommended_actions": [
            "keep data-collection-only bots in metadata_only or thin_sample mode until external free space is above the throttle threshold"
            if mode in {"critical", "throttle"}
            else "",
            "remove duplicate .local_fallback files from the external route; they are fallback-copy artifacts, not canonical live files"
            if cleanup_duplicates and duplicate_files and not apply
            else "",
            "run storage-tier-policy after cleanup to find the next archive/compact targets",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Guard new data-collection bots from exhausting external storage.")
    parser.add_argument("--external-root", default=str(DEFAULT_EXTERNAL_ROOT))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--warn-gb", type=float, default=120.0)
    parser.add_argument("--throttle-gb", type=float, default=80.0)
    parser.add_argument("--critical-gb", type=float, default=40.0)
    parser.add_argument("--cleanup-duplicates", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        external_root=Path(args.external_root).expanduser(),
        registry_path=Path(args.registry).expanduser(),
        warn_gb=float(args.warn_gb),
        throttle_gb=float(args.throttle_gb),
        critical_gb=float(args.critical_gb),
        apply=bool(args.apply),
        cleanup_duplicates=bool(args.cleanup_duplicates),
    )
    write_payload(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "data_collection_storage_guard "
            f"overall_status={payload.get('overall_status')} "
            f"guard_mode={payload.get('guard_mode')} "
            f"collector_count={payload.get('collector_count')} "
            f"changed_count={payload.get('changed_count')}"
        )
    return 0 if payload.get("overall_status") in {"ready", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
