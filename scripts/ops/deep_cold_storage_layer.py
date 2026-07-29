#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    from scripts.ops.long_runtime_common import iso_now, write_payload
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    from core.storage_mounts import resolve_external_storage
    from .long_runtime_common import iso_now, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "deep_cold_storage_layer_latest.json"
DEFAULT_MANIFEST_NAME = "deep_cold_manifest.jsonl"
DEFAULT_VIDEO_COLD_ARCHIVE_ROOT = "/Volumes/VIDEO/schwab_trading_bot_cold"
PROTECTED_VOLUME_NAMES = {"VIDEO"}


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_truthy(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes", "y", "on"}


def _gb(value: int | float) -> float:
    return round(float(value) / float(1024**3), 3)


def _file_age_days(path: Path, *, now: datetime) -> float:
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except Exception:
        return 0.0
    return max((now - mtime).total_seconds() / 86400.0, 0.0)


def _volume_name(path: Path) -> str:
    parts = path.expanduser().parts
    if len(parts) >= 3 and parts[1] == "Volumes":
        return parts[2]
    return ""


def _is_protected_volume(path: Path) -> bool:
    volume = _volume_name(path)
    if volume == "VIDEO" and _approved_video_cold_archive(path):
        return False
    return volume in PROTECTED_VOLUME_NAMES


def _approved_video_cold_archive(path: Path) -> bool:
    if not _env_truthy("BOT_ALLOW_VIDEO_COLD_ARCHIVE"):
        return False
    allowed_root = Path(os.getenv("BOT_VIDEO_COLD_ARCHIVE_ROOT", DEFAULT_VIDEO_COLD_ARCHIVE_ROOT)).expanduser()
    try:
        raw = path.expanduser().resolve(strict=False)
        allowed = allowed_root.resolve(strict=False)
    except Exception:
        raw = path.expanduser()
        allowed = allowed_root
    return bool(raw == allowed or allowed in raw.parents)


def _relative_to_any(path: Path, roots: list[tuple[str, Path]]) -> str:
    for label, root in sorted(roots, key=lambda item: len(str(item[1])), reverse=True):
        try:
            return str(Path(label) / path.relative_to(root))
        except Exception:
            continue
    return str(path)


def _economic_value_from_stale_path(rel: str) -> str:
    lowered = rel.lower()
    if "/decisions/" in lowered or lowered.startswith("data/stale_stage/decisions/"):
        return "critical"
    if "/decision_explanations/" in lowered:
        return "high"
    if "/governance/" in lowered or "governance_telemetry_compactor" in lowered:
        return "medium"
    if "/exports/" in lowered or "/logs/" in lowered:
        return "low"
    return "medium"


def _retention_days_for_value(value: str) -> int:
    return {
        "low": 3,
        "medium": 14,
        "high": 30,
        "critical": 90,
    }.get(str(value or "medium"), 14)


def _deep_cold_state(*, rel: str, value: str, age_days: float, suffix: str) -> str:
    if value == "critical":
        return "nearline_retained_critical"
    if suffix == ".gz":
        return "manifest_indexed_compressed"
    if age_days >= _retention_days_for_value(value):
        return "retention_mature_review"
    return "manifest_indexed_retention_locked"


def _iter_candidate_files(stale_root: Path, *, min_size_bytes: int) -> list[Path]:
    if not stale_root.exists():
        return []
    rows: list[Path] = []
    for path in stale_root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            size = path.stat().st_size
        except Exception:
            continue
        if size < min_size_bytes:
            continue
        if path.name == DEFAULT_MANIFEST_NAME:
            continue
        rows.append(path)
    return sorted(rows, key=lambda item: (-_safe_int(item.stat().st_size if item.exists() else 0), str(item)))


def _second_cold_target_for_row(row: dict[str, Any], *, second_cold_root: Path) -> Path:
    rel = str(row.get("relative_path") or row.get("path") or "").strip().lstrip("/")
    clean_parts = [part for part in Path(rel).parts if part not in {"", ".", ".."}]
    return second_cold_root / "deep_cold" / "stale_stage" / Path(*clean_parts)


def _copy_verify_then_symlink(source: Path, target: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "source": str(source),
        "target": str(target),
        "copied": False,
        "source_replaced_with_symlink": False,
        "verified_size_match": False,
        "skipped": False,
        "reason": "",
        "bytes": 0,
    }
    if source.is_symlink():
        result.update({"skipped": True, "reason": "source_already_symlink"})
        return result
    if not source.exists() or not source.is_file():
        result.update({"skipped": True, "reason": "source_missing_or_not_file"})
        return result

    source_size = _safe_int(source.stat().st_size)
    result["bytes"] = source_size
    target.parent.mkdir(parents=True, exist_ok=True)
    final_target = target
    if final_target.exists():
        target_size = _safe_int(final_target.stat().st_size)
        if target_size == source_size:
            result["verified_size_match"] = True
        else:
            stamp = iso_now().replace(":", "").replace("+", "_")
            final_target = target.with_name(f"{target.stem}.{stamp}{target.suffix}")

    if not result["verified_size_match"]:
        tmp = final_target.with_name(f".{final_target.name}.tmp")
        try:
            if tmp.exists():
                tmp.unlink()
            shutil.copy2(source, tmp)
            copied_size = _safe_int(tmp.stat().st_size)
            if copied_size != source_size:
                try:
                    tmp.unlink()
                except Exception:
                    pass
                result.update({"reason": f"copy_size_mismatch:{copied_size}!={source_size}"})
                return result
            os.replace(tmp, final_target)
            result.update({"copied": True, "verified_size_match": True, "target": str(final_target)})
        except Exception as exc:
            result.update({"reason": f"copy_failed:{exc}"})
            return result

    try:
        source.unlink()
        source.symlink_to(final_target)
        result["source_replaced_with_symlink"] = True
    except Exception as exc:
        result["reason"] = f"symlink_replace_failed:{exc}"
    return result


def _apply_second_cold_moves(
    rows: list[dict[str, Any]],
    *,
    second_cold_root: Path,
    max_move_gb: float,
    max_move_files: int,
    include_critical: bool,
) -> dict[str, Any]:
    if _is_protected_volume(second_cold_root):
        return {
            "enabled": True,
            "status": "blocked",
            "reason": "second_cold_root_protected_without_approved_subtree",
            "second_cold_root": str(second_cold_root),
            "moved_files": 0,
            "moved_gb": 0.0,
            "actions": [],
        }

    max_bytes = int(max(float(max_move_gb), 0.0) * (1024**3))
    max_files = max(int(max_move_files), 0)
    actions: list[dict[str, Any]] = []
    moved_bytes = 0
    candidates = [
        row
        for row in rows
        if (include_critical or str(row.get("economic_value") or "") != "critical")
        and str(row.get("path") or "").strip()
        and not bool(row.get("source_replaced_with_symlink", False))
    ]
    for row in candidates:
        if max_files and len(actions) >= max_files:
            break
        size = _safe_int(row.get("size_bytes"), 0)
        if max_bytes and moved_bytes + size > max_bytes and actions:
            break
        source = Path(str(row.get("path") or ""))
        target = _second_cold_target_for_row(row, second_cold_root=second_cold_root)
        action = _copy_verify_then_symlink(source, target)
        actions.append(action)
        row["second_cold_target"] = str(action.get("target") or target)
        row["second_cold_move"] = action
        row["source_replaced_with_symlink"] = bool(action.get("source_replaced_with_symlink", False))
        if bool(action.get("source_replaced_with_symlink", False)):
            moved_bytes += _safe_int(action.get("bytes"), size)

    failed = [row for row in actions if not bool(row.get("source_replaced_with_symlink", False)) and not bool(row.get("skipped", False))]
    return {
        "enabled": True,
        "status": "ready" if not failed else "partial",
        "reason": "" if not failed else "one_or_more_moves_failed",
        "second_cold_root": str(second_cold_root),
        "include_critical": bool(include_critical),
        "max_move_gb": round(float(max_move_gb), 3),
        "max_move_files": int(max_files),
        "candidate_files": len(candidates),
        "attempted_files": len(actions),
        "moved_files": sum(1 for row in actions if bool(row.get("source_replaced_with_symlink", False))),
        "moved_gb": _gb(moved_bytes),
        "failed_files": len(failed),
        "actions": actions[:50],
    }


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    apply: bool = False,
    min_size_mb: float = 25.0,
    top_n: int = 25,
    manifest_path: Path | None = None,
    move_to_second_cold: bool = False,
    second_cold_root: Path | None = None,
    max_move_gb: float = 64.0,
    max_move_files: int = 250,
    include_critical: bool = False,
) -> dict[str, Any]:
    external = resolve_external_storage()
    external_root = external.external_root
    if _is_protected_volume(external_root):
        payload = {
            "timestamp_utc": iso_now(),
            "schema_version": 1,
            "ok": False,
            "overall_status": "blocked",
            "apply": bool(apply),
            "blocked_reason": "protected_volume_refused",
            "protected_volume": _volume_name(external_root),
            "never_touch_protected_volumes": sorted(PROTECTED_VOLUME_NAMES),
        }
        return payload

    now = datetime.now(timezone.utc)
    stale_roots = [
        external_root / "data" / "stale_stage",
        project_root / "data" / "stale_stage",
    ]
    seen_roots: dict[str, Path] = {}
    for root in stale_roots:
        try:
            seen_roots[str(root.resolve())] = root
        except Exception:
            seen_roots[str(root)] = root
    roots = list(seen_roots.values())
    rel_roots = [("external", external_root), ("project", project_root)]
    min_size_bytes = max(int(float(min_size_mb) * 1024 * 1024), 1)
    rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for stale_root in roots:
        for path in _iter_candidate_files(stale_root, min_size_bytes=min_size_bytes):
            try:
                real = str(path.resolve())
            except Exception:
                real = str(path)
            if real in seen_paths:
                continue
            seen_paths.add(real)
            size = _safe_int(path.stat().st_size if path.exists() else 0)
            rel = _relative_to_any(path, rel_roots)
            value = _economic_value_from_stale_path(rel)
            age_days = _file_age_days(path, now=now)
            retention_days = _retention_days_for_value(value)
            suffix = path.suffix.lower()
            state = _deep_cold_state(rel=rel, value=value, age_days=age_days, suffix=suffix)
            rows.append(
                {
                    "relative_path": rel,
                    "path": str(path),
                    "size_bytes": size,
                    "size_gb": _gb(size),
                    "age_days": round(age_days, 3),
                    "economic_value": value,
                    "retention_days": retention_days,
                    "deep_cold_state": state,
                    "retention_locked": bool(age_days < retention_days),
                    "compressed": suffix == ".gz",
                    "eligible_for_delete": False,
                    "reason": "deep_cold_manifest_index_only_no_delete",
                }
            )

    rows.sort(key=lambda row: (-_safe_int(row.get("size_bytes"), 0), str(row.get("relative_path") or "")))
    managed_rows = [
        row
        for row in rows
        if str(row.get("deep_cold_state") or "") in {"manifest_indexed_compressed", "manifest_indexed_retention_locked"}
    ]
    critical_rows = [row for row in rows if str(row.get("economic_value") or "") == "critical"]
    retention_locked_rows = [row for row in rows if bool(row.get("retention_locked", False))]
    managed_bytes = sum(_safe_int(row.get("size_bytes"), 0) for row in managed_rows)
    total_bytes = sum(_safe_int(row.get("size_bytes"), 0) for row in rows)
    retention_locked_bytes = sum(_safe_int(row.get("size_bytes"), 0) for row in retention_locked_rows)
    deep_root = manifest_path.parent if manifest_path is not None else external_root / "data" / "deep_cold"
    final_manifest_path = manifest_path or deep_root / DEFAULT_MANIFEST_NAME
    second_cold_move: dict[str, Any] = {
        "enabled": False,
        "status": "disabled",
        "second_cold_root": str(second_cold_root or ""),
        "moved_files": 0,
        "moved_gb": 0.0,
        "actions": [],
    }
    if apply and move_to_second_cold:
        target_root = (
            second_cold_root
            or Path(os.getenv("BOT_SECOND_COLD_ROOT", "") or os.getenv("BOT_VIDEO_COLD_ARCHIVE_ROOT", DEFAULT_VIDEO_COLD_ARCHIVE_ROOT))
        ).expanduser()
        second_cold_move = _apply_second_cold_moves(
            rows,
            second_cold_root=target_root,
            max_move_gb=max_move_gb,
            max_move_files=max_move_files,
            include_critical=include_critical,
        )

    write_result: dict[str, Any] = {
        "applied": False,
        "manifest_path": str(final_manifest_path),
        "manifest_rows": 0,
        "error": "",
    }
    if apply:
        try:
            final_manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with final_manifest_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            write_result.update({"applied": True, "manifest_rows": len(rows)})
        except Exception as exc:
            write_result.update({"applied": False, "error": str(exc)})

    ready = bool(rows and (not apply or write_result.get("applied", False)))
    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": bool(ready),
        "overall_status": "ready" if ready else "needs_data",
        "apply": bool(apply),
        "external_root": str(external_root),
        "stale_roots": [str(root) for root in roots],
        "deep_cold_root": str(final_manifest_path.parent),
        "manifest_path": str(final_manifest_path),
        "min_size_mb": float(min_size_mb),
        "summary": {
            "candidate_count": len(rows),
            "candidate_gb": _gb(total_bytes),
            "managed_count": len(managed_rows),
            "managed_gb": _gb(managed_bytes),
            "retention_locked_count": len(retention_locked_rows),
            "retention_locked_gb": _gb(retention_locked_bytes),
            "critical_nearline_count": len(critical_rows),
            "critical_nearline_gb": _gb(sum(_safe_int(row.get("size_bytes"), 0) for row in critical_rows)),
        },
        "policy": {
            "mode": "manifest_indexed_deep_cold_no_delete",
            "purpose": "keep evidence discoverable while moving protected stale-stage archives out of active hot-path scoring",
            "delete_policy": "never delete from this layer; stale-reaper keeps owning retention deletion",
            "never_touch_protected_volumes": sorted(PROTECTED_VOLUME_NAMES),
            "protected_volume_checked": _volume_name(external_root),
            "second_cold_move_policy": (
                "copy-verify-retain-via-original-path-symlink"
                if move_to_second_cold
                else "manifest-only"
            ),
        },
        "second_cold_move": second_cold_move,
        "write_result": write_result,
        "top_rows": rows[: max(int(top_n), 1)],
        "control_env": {
            "BOT_DEEP_COLD_LAYER_ACTIVE": "1" if ready else "0",
            "BOT_DEEP_COLD_ROOT": str(final_manifest_path.parent),
            "BOT_DEEP_COLD_MANIFEST_PATH": str(final_manifest_path),
            "BOT_DEEP_COLD_MANAGED_GB": str(_gb(managed_bytes)),
            "BOT_DEEP_COLD_DELETE_POLICY": "never_delete_manifest_index_only",
            "BOT_DEEP_COLD_SECOND_COLD_MOVE_GB": str(second_cold_move.get("moved_gb", 0.0)),
        },
        "next_action": (
            "deep cold manifest is current; approved second-cold moves preserve original paths with symlinks"
            if move_to_second_cold and _safe_int(second_cold_move.get("moved_files"), 0) > 0
            else "deep cold manifest is current; storage control can treat retention-locked stale-stage archives as managed cold evidence"
            if ready
            else "run with --apply after stale-stage archives exist"
        ),
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the deep-cold manifest layer for protected BOT_LOGS archives.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--min-size-mb", type=float, default=25.0)
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--move-to-second-cold", action="store_true")
    parser.add_argument("--second-cold-root", default="")
    parser.add_argument("--max-move-gb", type=float, default=float(os.getenv("BOT_DEEP_COLD_MAX_MOVE_GB", "64.0")))
    parser.add_argument("--max-move-files", type=int, default=int(os.getenv("BOT_DEEP_COLD_MAX_MOVE_FILES", "250")))
    parser.add_argument("--include-critical", action="store_true", default=_env_truthy("BOT_DEEP_COLD_INCLUDE_CRITICAL"))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path).expanduser() if str(args.manifest_path or "").strip() else None
    second_cold_root = Path(args.second_cold_root).expanduser() if str(args.second_cold_root or "").strip() else None
    payload = build_payload(
        Path(args.project_root).resolve(),
        apply=bool(args.apply),
        min_size_mb=float(args.min_size_mb),
        top_n=int(args.top_n),
        manifest_path=manifest_path,
        move_to_second_cold=bool(args.move_to_second_cold),
        second_cold_root=second_cold_root,
        max_move_gb=float(args.max_move_gb),
        max_move_files=int(args.max_move_files),
        include_critical=bool(args.include_critical),
    )
    write_payload(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        print(
            "deep_cold_storage_layer "
            f"overall_status={payload.get('overall_status', '')} "
            f"managed_gb={summary.get('managed_gb', 0)} "
            f"manifest={payload.get('manifest_path', '')}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
