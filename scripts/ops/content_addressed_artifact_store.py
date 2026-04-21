#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import os


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORE_ROOT = PROJECT_ROOT / "governance" / "content_store" / "sha256"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "content_store" / "latest.json"
DEFAULT_TRACKED_PATHS = [
    "governance/health/runtime_gate_dashboard_latest.json",
    "governance/health/platform_control_plane_latest.json",
    "governance/health/training_report_latest.json",
    "governance/health/retrain_scorecard_latest.json",
    "governance/health/training_success_latest.json",
    "governance/health/schema_migration_guard_latest.json",
    "governance/health/ingestion_storage_control_latest.json",
    "governance/health/daily_auto_verify_latest.json",
    "governance/health/replay_end_to_end_latest.json",
    "governance/health/replay_hash_registry_guard_latest.json",
    "governance/health/bot_support_owner_guard_latest.json",
    "governance/health/new_bot_admission_guard_latest.json",
    "governance/health/retrain_schema_compatibility_latest.json",
    "governance/health/golden_replay_regression_latest.json",
    "governance/health/cohort_drift_baseline_latest.json",
    "governance/health/champion_challenger_probation_latest.json",
    "governance/health/champion_challenger_probation_action_latest.json",
    "governance/health/retrain_lane_scheduler_latest.json",
    "governance/feature_store/latest.json",
    "governance/walk_forward/promotion_readiness_latest.json",
    "exports/training/runtime_training_snapshot_latest.jsonl",
]


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _default_max_blob_bytes() -> int:
    raw_bytes = os.getenv("CONTENT_STORE_MAX_BLOB_BYTES", "").strip()
    if raw_bytes:
        try:
            return max(int(float(raw_bytes)), 0)
        except Exception:
            return 0
    raw_gb = os.getenv("CONTENT_STORE_MAX_BLOB_GB", "0.5").strip()
    try:
        return max(int(float(raw_gb) * (1024 ** 3)), 0)
    except Exception:
        return 0


def _default_gc_grace_days() -> float:
    raw = os.getenv("CONTENT_STORE_GC_GRACE_DAYS", "1").strip()
    try:
        return max(float(raw), 0.0)
    except Exception:
        return 1.0


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_path(project_root: Path, raw: str) -> Path:
    path = Path(str(raw or "").strip()).expanduser()
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return path


def _relative_label(project_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except Exception:
        return str(path.resolve())


def _mtime_utc(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    except Exception:
        return ""


def _artifact_row(
    project_root: Path,
    *,
    path: Path,
    store_root: Path,
    materialize: bool,
    max_blob_bytes: int,
) -> tuple[dict[str, Any], Path | None, bool]:
    size_bytes = int(path.stat().st_size)
    row: dict[str, Any] = {
        "path": _relative_label(project_root, path),
        "size_bytes": size_bytes,
        "source_mtime_utc": _mtime_utc(path),
        "sha256": "",
        "blob_path": "",
        "materialized": False,
        "skipped_reason": "",
    }
    if max_blob_bytes > 0 and size_bytes > max_blob_bytes:
        row["skipped_reason"] = "size_over_limit"
        return row, None, False

    digest = _sha(path)
    blob_path = store_root / digest[:2] / digest
    copied = False
    if materialize and not blob_path.exists():
        blob_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, blob_path)
        copied = True
    row.update(
        {
            "sha256": digest,
            "blob_path": str(blob_path),
            "materialized": bool(blob_path.exists()),
        }
    )
    return row, blob_path, copied


def _prune_unreferenced_blobs(
    *,
    store_root: Path,
    referenced_blobs: set[str],
    enabled: bool,
    grace_days: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "enabled": bool(enabled),
        "grace_days": round(float(grace_days), 3),
        "candidate_count": 0,
        "deleted_blob_count": 0,
        "deleted_bytes": 0,
        "delete_errors": 0,
        "skipped_recent_count": 0,
        "referenced_blob_count": int(len(referenced_blobs)),
    }
    if not enabled or not store_root.exists():
        return payload

    cutoff_ts = datetime.now(timezone.utc).timestamp() - max(float(grace_days), 0.0) * 86400.0
    for blob_path in sorted(path for path in store_root.rglob("*") if path.is_file()):
        blob_key = str(blob_path)
        if blob_key in referenced_blobs:
            continue
        payload["candidate_count"] = int(payload["candidate_count"]) + 1
        try:
            mtime_ts = blob_path.stat().st_mtime
        except Exception:
            payload["delete_errors"] = int(payload["delete_errors"]) + 1
            continue
        if mtime_ts > cutoff_ts:
            payload["skipped_recent_count"] = int(payload["skipped_recent_count"]) + 1
            continue
        try:
            deleted_bytes = int(blob_path.stat().st_size)
        except Exception:
            deleted_bytes = 0
        try:
            blob_path.unlink()
            payload["deleted_blob_count"] = int(payload["deleted_blob_count"]) + 1
            payload["deleted_bytes"] = int(payload["deleted_bytes"]) + deleted_bytes
        except OSError:
            payload["delete_errors"] = int(payload["delete_errors"]) + 1
    return payload


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    store_root: Path = DEFAULT_STORE_ROOT,
    tracked_paths: list[str] | None = None,
    materialize: bool = True,
    gc: bool = True,
    gc_grace_days: float | None = None,
    max_blob_bytes: int | None = None,
) -> dict[str, Any]:
    store_root.mkdir(parents=True, exist_ok=True)
    artifact_rows: list[dict[str, Any]] = []
    copied = 0
    skipped_blob_count = 0
    skipped_blob_bytes = 0
    max_blob_bytes = _default_max_blob_bytes() if max_blob_bytes is None else max(int(max_blob_bytes), 0)
    gc_grace_days = _default_gc_grace_days() if gc_grace_days is None else max(float(gc_grace_days), 0.0)
    referenced_blobs: set[str] = set()
    for raw in list(tracked_paths or DEFAULT_TRACKED_PATHS):
        path = _resolve_path(project_root, raw)
        if not path.exists() or not path.is_file():
            continue
        row, blob_path, copied_blob = _artifact_row(
            project_root,
            path=path,
            store_root=store_root,
            materialize=bool(materialize),
            max_blob_bytes=max_blob_bytes,
        )
        copied += int(bool(copied_blob))
        if blob_path is not None:
            referenced_blobs.add(str(blob_path))
        elif str(row.get("skipped_reason") or "") == "size_over_limit":
            skipped_blob_count += 1
            skipped_blob_bytes += int(row.get("size_bytes", 0) or 0)
        artifact_rows.append(row)
    artifact_rows.sort(key=lambda row: str(row.get("path") or ""))
    manifest_hash = hashlib.sha256(
        json.dumps(artifact_rows, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    gc_payload = _prune_unreferenced_blobs(
        store_root=store_root,
        referenced_blobs=referenced_blobs,
        enabled=bool(gc),
        grace_days=float(gc_grace_days),
    )
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": True,
        "store_root": str(store_root),
        "artifact_count": len(artifact_rows),
        "copied_blob_count": copied,
        "skipped_blob_count": int(skipped_blob_count),
        "skipped_blob_bytes": int(skipped_blob_bytes),
        "max_blob_bytes": int(max_blob_bytes),
        "manifest_hash": manifest_hash,
        "artifacts": artifact_rows,
        "gc": gc_payload,
        "top_actions": [
            "bind training, replay, and promotion bundles to immutable sha256 blobs before rollout",
            "use manifest_hash as the rollback/replay bundle identifier in governance events",
        ],
    }
    if skipped_blob_count > 0:
        payload["top_actions"].append("keep oversized runtime artifacts referenced by metadata only so the content store does not duplicate multi-gigabyte files")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize tracked artifacts into a content-addressed sha256 store.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--store-root", default=str(DEFAULT_STORE_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--path", action="append", default=[], help="Extra artifact path to content-address.")
    parser.add_argument("--no-materialize", action="store_true")
    parser.add_argument("--gc", action=argparse.BooleanOptionalAction, default=_env_flag("CONTENT_STORE_GC_ENABLED", "1"))
    parser.add_argument("--gc-grace-days", type=float, default=_default_gc_grace_days())
    parser.add_argument("--max-blob-bytes", type=int, default=_default_max_blob_bytes())
    parser.add_argument("--max-blob-gb", type=float, default=0.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    tracked = list(DEFAULT_TRACKED_PATHS) + list(args.path or [])
    max_blob_bytes = max(int(args.max_blob_bytes), 0)
    if float(args.max_blob_gb or 0.0) > 0.0:
        max_blob_bytes = max(int(float(args.max_blob_gb) * (1024 ** 3)), 0)
    payload = build_payload(
        Path(args.project_root).resolve(),
        store_root=Path(args.store_root).expanduser(),
        tracked_paths=tracked,
        materialize=not bool(args.no_materialize),
        gc=bool(args.gc),
        gc_grace_days=float(args.gc_grace_days),
        max_blob_bytes=max_blob_bytes,
    )
    out_path = Path(args.out_file).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "content_addressed_artifact_store "
            f"artifact_count={int(payload.get('artifact_count', 0) or 0)} "
            f"copied_blob_count={int(payload.get('copied_blob_count', 0) or 0)} "
            f"deleted_blob_count={int(((payload.get('gc') or {}).get('deleted_blob_count', 0)) or 0)} "
            f"manifest_hash={str(payload.get('manifest_hash') or '')[:16]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
