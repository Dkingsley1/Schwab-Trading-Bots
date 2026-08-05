#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import data_retention_policy as retention


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "stale_artifact_reaper_bot_latest.json"
DEFAULT_LOCK_PATH = Path(
    os.getenv(
        "DATA_RETENTION_LOCK_PATH",
        str(PROJECT_ROOT / "governance" / "locks" / "data_retention.lock"),
    )
)
DEFAULT_STALE_STAGE_ROOT = PROJECT_ROOT / "data" / "stale_stage"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True, indent=2))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _same_root(left: Path, right: Path) -> bool:
    try:
        return left.resolve(strict=False) == right.resolve(strict=False)
    except Exception:
        return str(left) == str(right)


def _merge_additional_root(payload: dict[str, Any], additional: dict[str, Any]) -> dict[str, Any]:
    merged = dict(payload)
    summary = dict(merged.get("summary") or {})
    additional_summary = dict(additional.get("summary") or {})
    additive_keys = {
        "candidate_files",
        "candidate_bytes",
        "candidate_files_raw",
        "candidate_bytes_raw",
        "deleted_files",
        "deleted_bytes",
        "delete_errors",
        "skipped_by_budget_files",
        "skipped_by_tier_files",
        "skipped_unmanifested_files",
        "skipped_unverified_manifest_files",
        "skipped_hash_mismatch_files",
        "skipped_protected_evidence_files",
        "skipped_legacy_reindex_hold_files",
        "legacy_reindexed_files",
        "legacy_reindexed_bytes",
        "legacy_reindex_remaining_files",
        "legacy_reindex_candidate_files",
        "legacy_reindex_candidate_bytes",
        "legacy_reindex_selected_files",
        "legacy_reindex_selected_bytes",
        "legacy_reindex_protected_files",
        "legacy_reindex_oversized_files",
        "legacy_reindex_errors",
    }
    for key in additive_keys:
        summary[key] = int(summary.get(key, 0) or 0) + int(additional_summary.get(key, 0) or 0)
    summary["budget_limited"] = bool(
        summary.get("budget_limited", False) or additional_summary.get("budget_limited", False)
    )
    merged["summary"] = summary
    merged["ok"] = bool(merged.get("ok", False) and additional.get("ok", False))
    merged["reason"] = "ok" if merged["ok"] else "one_or_more_stale_roots_failed"
    root_results = list(merged.get("root_results") or [])
    if not root_results:
        root_results.append(
            {
                "stale_root": str((merged.get("artifacts") or {}).get("stale_root") or ""),
                "summary": dict(payload.get("summary") or {}),
                "ok": bool(payload.get("ok", False)),
            }
        )
    root_results.append(
        {
            "stale_root": str((additional.get("artifacts") or {}).get("stale_root") or ""),
            "summary": additional_summary,
            "ok": bool(additional.get("ok", False)),
        }
    )
    merged["root_results"] = root_results
    summary["root_count"] = len(root_results)
    summary["all_roots_ok"] = bool(merged["ok"])
    return merged


def build_payload(
    project_root: Path,
    *,
    stale_stage_root: Path,
    stale_stage_manifest: str,
    stale_purge_days: int,
    stale_purge_low_value_days: int | None = None,
    stale_purge_medium_value_days: int | None = None,
    stale_purge_high_value_days: int | None = None,
    stale_purge_critical_value_days: int | None = None,
    max_delete_files: int = 5000,
    max_delete_gb: float = 10.0,
    max_reindex_files: int = 2048,
    max_reindex_gb: float = 4.0,
) -> dict[str, Any]:
    manifest_path = retention._stale_manifest_path(stale_stage_root, stale_stage_manifest)
    legacy_reindex = retention._reindex_legacy_stale_stage(
        stale_root=stale_stage_root,
        manifest_path=manifest_path,
        max_files=max_reindex_files,
        max_bytes=int(max(float(max_reindex_gb), 0.0) * (1024**3)),
    )
    purge = retention._purge_old_stale_stage(
        stale_root=stale_stage_root,
        manifest_path=manifest_path,
        older_than_days=int(stale_purge_days),
        low_value_days=stale_purge_low_value_days,
        medium_value_days=stale_purge_medium_value_days,
        high_value_days=stale_purge_high_value_days,
        critical_value_days=stale_purge_critical_value_days,
        max_files=max_delete_files,
        max_bytes=int(max(float(max_delete_gb), 0.0) * (1024**3)),
    )
    reindex_errors = list(legacy_reindex.get("errors", []) or [])
    ok = int(purge.get("delete_errors", 0) or 0) == 0 and not reindex_errors
    return {
        "timestamp_utc": _utc_now(),
        "project_root": str(project_root),
        "ok": bool(ok),
        "busy": False,
        "reason": ("ok" if ok else "reindex_or_purge_errors"),
        "summary": {
            "candidate_files": int(purge.get("candidate_files", 0) or 0),
            "candidate_bytes": int(purge.get("candidate_bytes", 0) or 0),
            "candidate_files_raw": int(purge.get("candidate_files_raw", purge.get("candidate_files", 0)) or 0),
            "candidate_bytes_raw": int(purge.get("candidate_bytes_raw", purge.get("candidate_bytes", 0)) or 0),
            "deleted_files": int(purge.get("deleted_files", 0) or 0),
            "deleted_bytes": int(purge.get("deleted_bytes", 0) or 0),
            "delete_errors": int(purge.get("delete_errors", 0) or 0),
            "older_than_days": int(purge.get("older_than_days", stale_purge_days) or stale_purge_days),
            "budget_limited": bool(purge.get("budget_limited", False)),
            "skipped_by_budget_files": int(purge.get("skipped_by_budget_files", 0) or 0),
            "skipped_by_tier_files": int(purge.get("skipped_by_tier_files", 0) or 0),
            "skipped_unmanifested_files": int(purge.get("skipped_unmanifested_files", 0) or 0),
            "skipped_unverified_manifest_files": int(purge.get("skipped_unverified_manifest_files", 0) or 0),
            "skipped_hash_mismatch_files": int(purge.get("skipped_hash_mismatch_files", 0) or 0),
            "skipped_protected_evidence_files": int(purge.get("skipped_protected_evidence_files", 0) or 0),
            "skipped_legacy_reindex_hold_files": int(purge.get("skipped_legacy_reindex_hold_files", 0) or 0),
            "legacy_reindexed_files": int(legacy_reindex.get("reindexed_files", 0) or 0),
            "legacy_reindexed_bytes": int(legacy_reindex.get("reindexed_bytes", 0) or 0),
            "legacy_reindex_remaining_files": int(legacy_reindex.get("remaining_files", 0) or 0),
            "legacy_reindex_candidate_files": int(legacy_reindex.get("candidate_files", 0) or 0),
            "legacy_reindex_candidate_bytes": int(legacy_reindex.get("candidate_bytes", 0) or 0),
            "legacy_reindex_selected_files": int(legacy_reindex.get("selected_files", 0) or 0),
            "legacy_reindex_selected_bytes": int(legacy_reindex.get("selected_bytes", 0) or 0),
            "legacy_reindex_protected_files": int(legacy_reindex.get("protected_reindexed_files", 0) or 0),
            "legacy_reindex_oversized_files": int(legacy_reindex.get("oversized_candidate_files", 0) or 0),
            "legacy_reindex_errors": len(reindex_errors),
            "purge_policy": purge.get("purge_policy") if isinstance(purge.get("purge_policy"), dict) else {},
            "manifest_lines_after": int(((purge.get("manifest_compaction") or {}).get("lines_after", 0) or 0)),
        },
        "purge": purge,
        "legacy_manifest_reindex": legacy_reindex,
        "artifacts": {
            "stale_root": str(stale_stage_root),
            "stale_manifest": str(manifest_path),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Purge aged files that are already sitting inside stale_stage.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--stale-stage-root", default=str(DEFAULT_STALE_STAGE_ROOT))
    parser.add_argument("--stale-stage-manifest", default="")
    parser.add_argument("--stale-purge-days", type=int, default=int(os.getenv("RETENTION_STALE_PURGE_DAYS", "30")))
    parser.add_argument("--stale-purge-low-value-days", type=int, default=int(os.getenv("RETENTION_STALE_PURGE_LOW_VALUE_DAYS", "3")))
    parser.add_argument("--stale-purge-medium-value-days", type=int, default=int(os.getenv("RETENTION_STALE_PURGE_MEDIUM_VALUE_DAYS", "14")))
    parser.add_argument("--stale-purge-high-value-days", type=int, default=int(os.getenv("RETENTION_STALE_PURGE_HIGH_VALUE_DAYS", "30")))
    parser.add_argument("--stale-purge-critical-value-days", type=int, default=int(os.getenv("RETENTION_STALE_PURGE_CRITICAL_VALUE_DAYS", "90")))
    parser.add_argument("--max-delete-files", type=int, default=int(os.getenv("RETENTION_STALE_PURGE_MAX_FILES", "5000")))
    parser.add_argument("--max-delete-gb", type=float, default=float(os.getenv("RETENTION_STALE_PURGE_MAX_GB", "10")))
    parser.add_argument("--max-reindex-files", type=int, default=int(os.getenv("RETENTION_STALE_REINDEX_MAX_FILES", "2048")))
    parser.add_argument("--max-reindex-gb", type=float, default=float(os.getenv("RETENTION_STALE_REINDEX_MAX_GB", "4")))
    parser.add_argument(
        "--include-external-stale-root",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("RETENTION_INCLUDE_EXTERNAL_STALE_ROOT", "1").strip() == "1",
    )
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--lock-file", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_file = Path(args.out_file).expanduser()
    lock_file = Path(args.lock_file).expanduser()
    lock_file.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "timestamp_utc": _utc_now(),
        "project_root": str(project_root),
        "ok": True,
        "busy": False,
        "reason": "pending",
    }

    with lock_file.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            payload.update({"busy": True, "reason": "already_running"})
            _write_json(out_file, payload)
            if args.json:
                print(json.dumps(payload, ensure_ascii=True))
            else:
                print("stale_artifact_reaper_bot busy=1 reason=already_running")
            return 0

        primary_stale_root = Path(args.stale_stage_root).expanduser()
        payload = build_payload(
            project_root,
            stale_stage_root=primary_stale_root,
            stale_stage_manifest=str(args.stale_stage_manifest or ""),
            stale_purge_days=int(args.stale_purge_days),
            stale_purge_low_value_days=args.stale_purge_low_value_days if args.stale_purge_low_value_days is None else int(args.stale_purge_low_value_days),
            stale_purge_medium_value_days=args.stale_purge_medium_value_days if args.stale_purge_medium_value_days is None else int(args.stale_purge_medium_value_days),
            stale_purge_high_value_days=args.stale_purge_high_value_days if args.stale_purge_high_value_days is None else int(args.stale_purge_high_value_days),
            stale_purge_critical_value_days=args.stale_purge_critical_value_days if args.stale_purge_critical_value_days is None else int(args.stale_purge_critical_value_days),
            max_delete_files=int(args.max_delete_files),
            max_delete_gb=float(args.max_delete_gb),
            max_reindex_files=int(args.max_reindex_files),
            max_reindex_gb=float(args.max_reindex_gb),
        )
        if args.include_external_stale_root:
            external_stale_root = retention._resolve_external_project_root() / "data" / "stale_stage"
            if external_stale_root.exists() and not _same_root(primary_stale_root, external_stale_root):
                additional = build_payload(
                    project_root,
                    stale_stage_root=external_stale_root,
                    stale_stage_manifest="",
                    stale_purge_days=int(args.stale_purge_days),
                    stale_purge_low_value_days=int(args.stale_purge_low_value_days),
                    stale_purge_medium_value_days=int(args.stale_purge_medium_value_days),
                    stale_purge_high_value_days=int(args.stale_purge_high_value_days),
                    stale_purge_critical_value_days=int(args.stale_purge_critical_value_days),
                    max_delete_files=int(args.max_delete_files),
                    max_delete_gb=float(args.max_delete_gb),
                    max_reindex_files=int(args.max_reindex_files),
                    max_reindex_gb=float(args.max_reindex_gb),
                )
                payload = _merge_additional_root(payload, additional)
        _write_json(out_file, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        print(
            "stale_artifact_reaper_bot "
            f"ok={int(bool(payload.get('ok', False)))} "
            f"deleted_files={int(summary.get('deleted_files', 0) or 0)} "
            f"candidate_files={int(summary.get('candidate_files', 0) or 0)}"
        )
    return 0 if bool(payload.get("ok", False) or payload.get("busy", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
