#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
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
    from scripts.ops.long_runtime_common import iso_now, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "decision_log_compactor_latest.json"
DEFAULT_LOCK_PATH = PROJECT_ROOT / "governance" / "locks" / "decision_log_compactor.lock"


def _gb(raw_bytes: int) -> float:
    return round(float(raw_bytes) / float(1024**3), 3)


def _today_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _relative(project_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(project_root)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _canonical_source_rel(project_root: Path, path: Path) -> str:
    rel = _relative(project_root, path)
    for family in ("decisions", "decision_explanations"):
        fallback_prefix = f"local_fallback_storage/{family}/"
        if rel.startswith(fallback_prefix):
            return f"{family}/{rel[len(fallback_prefix):]}"
    return rel


def _effective_runtime_overrides(project_root: Path) -> dict[str, str]:
    paths = (
        project_root / "config" / ".env.hot_lane_retention_override",
        project_root / "config" / ".env.storage_pressure_override",
        project_root / "config" / ".env.storage_override",
        project_root / "config" / ".env.runtime_resource_guard_override",
        project_root / "config" / ".env.local_storage_reserve_override",
    )
    values: dict[str, str] = {}
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for raw in lines:
            line = str(raw or "").strip()
            if (not line) or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip():
                values[key.strip()] = value.strip().strip("'\"")
    return values


def _sqlite_progress_index(project_root: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    state_root = project_root / "governance" / "sql_link_shards"
    for path in state_root.glob("jsonl_sql_link_state*.json") if state_root.exists() else ():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows = payload.get("sqlite") if isinstance(payload, dict) else {}
        if not isinstance(rows, dict):
            continue
        for rel, raw_progress in rows.items():
            if not isinstance(raw_progress, dict):
                continue
            progress = dict(raw_progress)
            progress["state_path"] = _relative(project_root, path)
            previous = index.get(str(rel))
            if previous is None or int(progress.get("last_offset_bytes", 0) or 0) > int(
                previous.get("last_offset_bytes", 0) or 0
            ):
                index[str(rel)] = progress
    return index


def _current_day_logging_disabled(source_rel: str, overrides: dict[str, str]) -> tuple[bool, str]:
    rel = str(source_rel or "")
    if rel.startswith("decision_explanations/"):
        key = "LOG_DECISION_EXPLANATIONS"
    elif rel.startswith("governance/") and "shadow_pnl_attribution_" in rel:
        key = "LOG_SHADOW_PNL_ATTRIBUTION"
    else:
        return False, "current_day_family_not_rotation_safe"
    raw = overrides.get(key)
    if raw is None:
        return False, f"{key.lower()}_not_explicitly_disabled"
    disabled = str(raw).strip().lower() in {"0", "false", "no", "off"}
    return disabled, ("logging_disabled" if disabled else f"{key.lower()}_still_enabled")


def _file_day(path: Path) -> str:
    for token in path.name.replace(".", "_").split("_"):
        if len(token) == 8 and token.startswith("20") and token.isdigit():
            return token
    return ""


def _path_age_minutes(path: Path) -> float:
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return 0.0
    return max((datetime.now(timezone.utc) - mtime).total_seconds() / 60.0, 0.0)


def _parse_families(raw: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if isinstance(raw, (list, tuple)):
        rows = [str(item).strip() for item in raw if str(item).strip()]
    else:
        rows = [part.strip() for part in str(raw or "").split(",") if part.strip()]
    return rows or ["decisions", "decision_explanations", "paper_bridge", "shadow_pnl_attribution"]


def _family_roots(project_root: Path, families: list[str]) -> list[Path]:
    roots: list[Path] = []
    for family in families:
        if family == "decisions":
            roots.append(project_root / "decisions")
        elif family == "decision_explanations":
            roots.append(project_root / "decision_explanations")
        elif family in {"paper_bridge", "exports_paper_broker_bridge"}:
            roots.append(project_root / "exports" / "paper_broker_bridge")
    resolved_roots: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        candidates = [root]
        try:
            resolved = root.resolve()
            if resolved != root:
                candidates.append(resolved)
        except Exception:
            pass
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            resolved_roots.append(candidate)
    return resolved_roots


def _iter_family_files(project_root: Path, families: list[str]) -> list[Path]:
    paths: list[Path] = []
    for family in families:
        if family == "shadow_pnl_attribution":
            governance_root = project_root / "governance"
            if governance_root.exists():
                paths.extend(
                    path
                    for path in governance_root.glob("shadow*/shadow_pnl_attribution_*.jsonl")
                    if ".__external_symlink_backup" not in path.parent.name
                )
            continue
        for root in _family_roots(project_root, [family]):
            if not root.exists():
                continue
            paths.extend(path for path in root.rglob("*") if _is_candidate_log_path(path))
    return sorted(paths, key=lambda path: str(path))


def _candidate_rows(
    *,
    project_root: Path,
    min_file_bytes: int,
    include_current_day: bool,
    min_age_minutes: float,
    families: list[str],
    require_current_day_safe: bool,
) -> list[dict[str, Any]]:
    today = _today_stamp()
    progress_index = _sqlite_progress_index(project_root)
    runtime_overrides = _effective_runtime_overrides(project_root)
    rows: list[dict[str, Any]] = []
    seen_files: set[tuple[int, int]] = set()
    for path in _iter_family_files(project_root, families):
        try:
            stat = path.stat()
            identity = (int(stat.st_dev), int(stat.st_ino))
        except OSError:
            continue
        if identity in seen_files:
            continue
        seen_files.add(identity)
        row = _candidate_row(
            project_root=project_root,
            path=path,
            min_file_bytes=min_file_bytes,
            include_current_day=include_current_day,
            min_age_minutes=min_age_minutes,
            today=today,
            require_current_day_safe=require_current_day_safe,
            progress_index=progress_index,
            runtime_overrides=runtime_overrides,
        )
        if row:
            rows.append(row)
    rows.sort(key=lambda row: (-int(row.get("size_bytes", 0) or 0), str(row.get("relative_path") or "")))
    return rows


def _candidate_row(
    *,
    project_root: Path,
    path: Path,
    min_file_bytes: int,
    include_current_day: bool,
    min_age_minutes: float,
    today: str,
    require_current_day_safe: bool,
    progress_index: dict[str, dict[str, Any]],
    runtime_overrides: dict[str, str],
) -> dict[str, Any] | None:
    if not _is_candidate_log_path(path):
        return None
    try:
        stat = path.stat()
        size_bytes = int(stat.st_size)
    except OSError:
        return None
    if size_bytes < min_file_bytes:
        return None
    age_minutes = _path_age_minutes(path)
    if age_minutes < float(min_age_minutes):
        return None
    day = _file_day(path)
    is_current_day = bool(day and day >= today)
    if is_current_day and not include_current_day:
        return None
    source_rel = _canonical_source_rel(project_root, path)
    current_day_safety: dict[str, Any] = {
        "required": bool(is_current_day and require_current_day_safe),
        "ready": not bool(is_current_day and require_current_day_safe),
        "reason": "not_current_day",
    }
    if is_current_day and require_current_day_safe:
        logging_disabled, logging_reason = _current_day_logging_disabled(source_rel, runtime_overrides)
        progress = dict(progress_index.get(source_rel) or {})
        checkpoint_offset = int(progress.get("last_offset_bytes", 0) or 0)
        checkpoint_size = int(progress.get("file_size_bytes", 0) or 0)
        checkpoint_inode = int(progress.get("file_inode", 0) or 0)
        fully_ingested = bool(
            checkpoint_offset >= size_bytes
            and checkpoint_size == size_bytes
            and (checkpoint_inode <= 0 or checkpoint_inode == int(stat.st_ino))
        )
        current_day_safety = {
            "required": True,
            "ready": bool(logging_disabled and fully_ingested),
            "reason": (
                "inert_and_fully_ingested"
                if logging_disabled and fully_ingested
                else logging_reason
                if not logging_disabled
                else "sqlite_checkpoint_not_at_exact_eof"
            ),
            "logging_disabled": bool(logging_disabled),
            "logging_reason": logging_reason,
            "fully_ingested": bool(fully_ingested),
            "checkpoint_offset_bytes": checkpoint_offset,
            "checkpoint_file_size_bytes": checkpoint_size,
            "checkpoint_state_path": str(progress.get("state_path") or ""),
        }
        if not current_day_safety["ready"]:
            return None
    return {
        "relative_path": source_rel,
        "fallback_copy": ".jsonl.local_fallback" in path.name,
        "size_bytes": size_bytes,
        "size_gb": _gb(size_bytes),
        "day": day,
        "current_day": is_current_day,
        "age_minutes": round(age_minutes, 3),
        "source_fingerprint": {
            "device": int(stat.st_dev),
            "inode": int(stat.st_ino),
            "size_bytes": size_bytes,
            "mtime_ns": int(stat.st_mtime_ns),
        },
        "current_day_safety": current_day_safety,
        "action": "gzip_compact_decision_explanation_or_bridge_log_in_place",
    }


def _is_candidate_log_path(path: Path) -> bool:
    if not path.is_file():
        return False
    name = path.name
    if name.endswith(".gz") or ".tmp." in name or ".compact_pending" in name:
        return False
    return name.endswith(".jsonl") or ".jsonl.local_fallback" in name


def _select_rows(rows: list[dict[str, Any]], *, target_free_bytes: int, max_files: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_bytes = 0
    for row in rows:
        if max_files > 0 and len(selected) >= max_files:
            break
        selected.append(dict(row))
        selected_bytes += int(row.get("size_bytes", 0) or 0)
        if target_free_bytes > 0 and selected_bytes >= target_free_bytes:
            break
    return selected


def _compact_one(
    *,
    project_root: Path,
    source_rel: str,
    compression_level: int,
    source_fingerprint: dict[str, Any] | None = None,
    current_day: bool = False,
) -> dict[str, Any]:
    source_path = project_root / source_rel
    if not source_path.exists():
        return {"relative_path": source_rel, "status": "missing", "error": "source_missing"}
    archive_path = source_path.with_name(f"{source_path.name}.gz")
    if current_day and archive_path.exists():
        stem = source_path.name[:-6] if source_path.name.endswith(".jsonl") else source_path.name
        for part in range(1, 10000):
            candidate = source_path.with_name(f"{stem}.part{part:03d}.jsonl.gz")
            if not candidate.exists():
                archive_path = candidate
                break
    archive_preexisting = archive_path.exists()

    try:
        raw_bytes = int(source_path.stat().st_size)
    except OSError as exc:
        return {"relative_path": source_rel, "status": "error", "error": str(exc)}

    expected = dict(source_fingerprint or {})
    try:
        before = source_path.stat()
    except OSError as exc:
        return {"relative_path": source_rel, "status": "error", "error": str(exc)}
    if expected and any(
        (
            int(expected.get("device", before.st_dev) or before.st_dev) != int(before.st_dev),
            int(expected.get("inode", before.st_ino) or before.st_ino) != int(before.st_ino),
            int(expected.get("size_bytes", before.st_size) or before.st_size) != int(before.st_size),
            int(expected.get("mtime_ns", before.st_mtime_ns) or before.st_mtime_ns) != int(before.st_mtime_ns),
        )
    ):
        return {
            "relative_path": source_rel,
            "status": "deferred_source_changed",
            "error": "source_fingerprint_changed_before_compaction",
        }

    tmp_archive = archive_path.with_name(f"{archive_path.name}.tmp.{os.getpid()}")
    try:
        digest = hashlib.sha256()
        copied_bytes = 0
        with source_path.open("rb") as src, gzip.open(
            tmp_archive,
            "wb",
            compresslevel=max(min(int(compression_level), 9), 1),
        ) as dst:
            while True:
                chunk = src.read(4 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                copied_bytes += len(chunk)
                dst.write(chunk)
        after = source_path.stat()
        unchanged = bool(
            int(before.st_dev) == int(after.st_dev)
            and int(before.st_ino) == int(after.st_ino)
            and int(before.st_size) == int(after.st_size)
            and int(before.st_mtime_ns) == int(after.st_mtime_ns)
            and copied_bytes == int(before.st_size)
        )
        if not unchanged:
            tmp_archive.unlink(missing_ok=True)
            return {
                "relative_path": source_rel,
                "status": "deferred_source_changed",
                "error": "source_changed_during_compaction",
                "copied_bytes": copied_bytes,
                "expected_bytes": int(before.st_size),
            }
        tmp_archive.replace(archive_path)
        archive_bytes = int(archive_path.stat().st_size)
        source_path.unlink()
        return {
            "relative_path": source_rel,
            "status": "compacted",
            "archive_replaced": bool(archive_preexisting),
            "raw_bytes": raw_bytes,
            "raw_gb": _gb(raw_bytes),
            "archive_path": _relative(project_root, archive_path),
            "archive_bytes": archive_bytes,
            "archive_gb": _gb(archive_bytes),
            "source_sha256": digest.hexdigest(),
            "source_fingerprint_verified": True,
            "estimated_reduction_bytes": max(raw_bytes - archive_bytes, 0),
            "estimated_reduction_gb": _gb(max(raw_bytes - archive_bytes, 0)),
        }
    except Exception as exc:
        try:
            tmp_archive.unlink(missing_ok=True)
        except Exception:
            pass
        return {
            "relative_path": source_rel,
            "status": "error",
            "raw_bytes": raw_bytes,
            "raw_gb": _gb(raw_bytes),
            "archive_path": _relative(project_root, archive_path),
            "error": str(exc),
        }


def build_payload(
    *,
    project_root: Path = PROJECT_ROOT,
    apply: bool = False,
    min_file_mb: float = 128.0,
    target_free_gb: float = 9.0,
    max_files: int = 8,
    include_current_day: bool = False,
    min_age_minutes: float = 60.0,
    compression_level: int = 1,
    families: str | list[str] | tuple[str, ...] | None = None,
    require_current_day_safe: bool = True,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    min_file_bytes = max(int(float(min_file_mb) * 1024 * 1024), 1)
    target_free_bytes = max(int(float(target_free_gb) * 1024 * 1024 * 1024), 0)
    family_list = _parse_families(families)
    candidates = _candidate_rows(
        project_root=project_root,
        min_file_bytes=min_file_bytes,
        include_current_day=bool(include_current_day),
        min_age_minutes=float(min_age_minutes),
        families=family_list,
        require_current_day_safe=bool(require_current_day_safe),
    )
    selected = _select_rows(candidates, target_free_bytes=target_free_bytes, max_files=max(int(max_files), 0))

    if apply and selected:
        records = [
            _compact_one(
                project_root=project_root,
                source_rel=str(row.get("relative_path") or ""),
                compression_level=int(compression_level),
                source_fingerprint=(
                    row.get("source_fingerprint")
                    if isinstance(row.get("source_fingerprint"), dict)
                    else None
                ),
                current_day=bool(row.get("current_day", False)),
            )
            for row in selected
        ]
    else:
        records = [dict(row, status="planned") for row in selected]

    compacted = [row for row in records if str(row.get("status") or "") == "compacted"]
    errors = [row for row in records if str(row.get("status") or "") == "error"]
    selected_bytes = sum(int(row.get("size_bytes", row.get("raw_bytes", 0)) or 0) for row in selected)
    raw_compacted_bytes = sum(int(row.get("raw_bytes", 0) or 0) for row in compacted)
    archive_bytes = sum(int(row.get("archive_bytes", 0) or 0) for row in compacted)
    reduction_bytes = sum(int(row.get("estimated_reduction_bytes", 0) or 0) for row in compacted)
    if errors:
        overall_status = "degraded"
    elif apply and compacted:
        overall_status = "applied"
    elif selected:
        overall_status = "planned"
    else:
        overall_status = "nothing_to_do"

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": not errors,
        "overall_status": overall_status,
        "apply": bool(apply),
        "policy": {
            "min_file_mb": float(min_file_mb),
            "target_free_gb": float(target_free_gb),
            "max_files": int(max_files),
            "include_current_day": bool(include_current_day),
            "require_current_day_safe": bool(require_current_day_safe),
            "min_age_minutes": float(min_age_minutes),
            "compression_level": int(compression_level),
            "families": family_list,
            "compaction_policy": "gzip_old_logs_in_place; current-day logs require explicit producer disablement, exact SQL EOF, minimum inert age, and unchanged-file verification",
        },
        "summary": {
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "selected_bytes": int(selected_bytes),
            "selected_gb": _gb(selected_bytes),
            "compacted_count": len(compacted),
            "raw_compacted_bytes": int(raw_compacted_bytes),
            "raw_compacted_gb": _gb(raw_compacted_bytes),
            "archive_bytes": int(archive_bytes),
            "archive_gb": _gb(archive_bytes),
            "estimated_reduction_bytes": int(reduction_bytes),
            "estimated_reduction_gb": _gb(reduction_bytes),
            "error_count": len(errors),
        },
        "records": records,
        "next_action": (
            "refresh storage-tier-policy and storage-quota-guard"
            if apply and compacted
            else "run with --apply to compact old decision logs"
            if selected and not apply
            else "monitor decisions quota"
        ),
    }


def _acquire_lock(path: Path) -> tuple[Any | None, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fh.seek(0)
        fh.truncate()
        fh.write(f"pid={os.getpid()} started={iso_now()}\n")
        fh.flush()
        return fh, ""
    except BlockingIOError:
        try:
            fh.seek(0)
            owner = fh.read().strip()
        except Exception:
            owner = ""
        try:
            fh.close()
        except Exception:
            pass
        return None, owner


def main() -> int:
    parser = argparse.ArgumentParser(description="Gzip-compact older decisions JSONL files while preserving current-day hot logs.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--lock-path", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument("--min-file-mb", type=float, default=float(os.getenv("DECISION_LOG_COMPACTOR_MIN_FILE_MB", "128")))
    parser.add_argument("--target-free-gb", type=float, default=float(os.getenv("DECISION_LOG_COMPACTOR_TARGET_FREE_GB", "9")))
    parser.add_argument("--max-files", type=int, default=int(os.getenv("DECISION_LOG_COMPACTOR_MAX_FILES", "8")))
    parser.add_argument("--compression-level", type=int, default=int(os.getenv("DECISION_LOG_COMPACTOR_GZIP_LEVEL", "1")))
    parser.add_argument("--include-current-day", action=argparse.BooleanOptionalAction, default=os.getenv("DECISION_LOG_COMPACTOR_INCLUDE_CURRENT_DAY", "0").strip() == "1")
    parser.add_argument(
        "--require-current-day-safe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require explicit logging disablement and an exact SQLite EOF checkpoint before current-day rotation.",
    )
    parser.add_argument("--min-age-minutes", type=float, default=float(os.getenv("DECISION_LOG_COMPACTOR_MIN_AGE_MINUTES", "60")))
    parser.add_argument(
        "--families",
        default=os.getenv(
            "DECISION_LOG_COMPACTOR_FAMILIES",
            "decisions,decision_explanations,paper_bridge,shadow_pnl_attribution",
        ),
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out_file).expanduser()
    lock_fh = None
    if args.apply:
        lock_fh, owner = _acquire_lock(Path(args.lock_path).expanduser())
        if lock_fh is None:
            payload = {
                "timestamp_utc": iso_now(),
                "schema_version": 1,
                "ok": False,
                "overall_status": "busy",
                "apply": True,
                "lock_owner": owner,
            }
            write_payload(out_path, payload)
            print(json.dumps(payload, ensure_ascii=True))
            return 2
    try:
        payload = build_payload(
            project_root=Path(args.project_root).resolve(),
            apply=bool(args.apply),
            min_file_mb=float(args.min_file_mb),
            target_free_gb=float(args.target_free_gb),
            max_files=int(args.max_files),
            include_current_day=bool(args.include_current_day),
            min_age_minutes=float(args.min_age_minutes),
            compression_level=int(args.compression_level),
            families=args.families,
            require_current_day_safe=bool(args.require_current_day_safe),
        )
        write_payload(out_path, payload)
        if args.json:
            print(json.dumps(payload, ensure_ascii=True))
        else:
            summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
            print(
                "decision_log_compactor "
                f"overall_status={payload.get('overall_status', '')} "
                f"selected_gb={summary.get('selected_gb', 0)} "
                f"compacted_gb={summary.get('raw_compacted_gb', 0)}"
            )
        return 0 if payload.get("ok", False) else 2
    finally:
        if lock_fh is not None:
            try:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                lock_fh.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
