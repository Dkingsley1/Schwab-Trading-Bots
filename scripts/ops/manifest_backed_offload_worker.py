#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "governance" / "health" / "storage_tier_offload_manifest_latest.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "manifest_backed_offload_worker_latest.json"
DEFAULT_PROOF_PATH = PROJECT_ROOT / "governance" / "health" / "manifest_backed_offload_restore_proofs_latest.jsonl"
PROTECTED_VOLUME_PREFIXES = ("/Volumes/VIDEO",)
DEFAULT_COLD_TARGETS = (
    "/Volumes/BOT_COLD/schwab_trading_bot",
    "/Volumes/BOT_ARCHIVE/schwab_trading_bot",
    "/Volumes/BOT_RETENTION/schwab_trading_bot",
)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _gb(raw_bytes: int | float) -> float:
    return round(float(raw_bytes) / float(1024**3), 4)


def _is_protected(path: Path) -> bool:
    raw = str(path.expanduser())
    return any(raw == prefix or raw.startswith(f"{prefix}/") for prefix in PROTECTED_VOLUME_PREFIXES)


def _resolve_target(raw_target: str) -> tuple[Path | None, list[dict[str, Any]]]:
    configured = raw_target.strip() or os.getenv("BOT_SECOND_COLD_ROOT", "").strip()
    candidates = [configured] if configured else list(DEFAULT_COLD_TARGETS)
    rows: list[dict[str, Any]] = []
    for raw in candidates:
        if not raw:
            continue
        path = Path(raw).expanduser()
        protected = _is_protected(path)
        ready = bool(path.exists() and path.is_dir() and not protected)
        rows.append(
            {
                "path": str(path),
                "configured": bool(configured and raw == configured),
                "exists": bool(path.exists()),
                "protected": protected,
                "ready": ready,
            }
        )
        if ready:
            return path, rows
    return None, rows


def _load_json(path: Path) -> dict[str, Any]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _eligible_entries(manifest: dict[str, Any], *, project_root: Path) -> list[dict[str, Any]]:
    rows = manifest.get("entries") if isinstance(manifest.get("entries"), list) else []
    eligible: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("classification") or "") != "eligible_manifest_backed_offload":
            continue
        rel = str(row.get("relative_path") or "")
        if not rel:
            continue
        source = project_root / rel
        if not source.exists() or not source.is_file() or source.is_symlink():
            continue
        allowed = {str(item) for item in (row.get("allowed_actions") or [])}
        proof = row.get("proof_required") if isinstance(row.get("proof_required"), dict) else {}
        if "copy_to_cold_tier" not in allowed or not bool(proof.get("post_copy_sha256_match", False)):
            continue
        eligible.append(dict(row))
    eligible.sort(key=lambda item: (-_safe_int(item.get("size_bytes"), 0), str(item.get("relative_path") or "")))
    return eligible


def _select_entries(entries: list[dict[str, Any]], *, max_files: int, max_bytes: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    total = 0
    for row in entries:
        size = _safe_int(row.get("size_bytes"), 0)
        if len(selected) >= max(int(max_files), 1):
            break
        if selected and total + size > max(int(max_bytes), 1):
            break
        selected.append(row)
        total += size
    return selected


def _copy_verify_one(
    *,
    project_root: Path,
    target_root: Path,
    entry: dict[str, Any],
    proof_path: Path,
) -> dict[str, Any]:
    rel = str(entry.get("relative_path") or "")
    source = project_root / rel
    planned_rel = str(entry.get("planned_cold_relative_path") or rel)
    target = target_root / planned_rel
    try:
        source_size = int(source.stat().st_size)
        source_hash = _sha256(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_target = target.with_name(f"{target.name}.tmp.{os.getpid()}")
        shutil.copy2(source, tmp_target)
        copied_size = int(tmp_target.stat().st_size)
        copied_hash = _sha256(tmp_target)
        if copied_size != source_size or copied_hash != source_hash:
            tmp_target.unlink(missing_ok=True)
            status = "verify_failed"
        else:
            tmp_target.replace(target)
            status = "copied_verified"
    except Exception as exc:
        return {
            "relative_path": rel,
            "status": "error",
            "error": str(exc),
            "source_retained": True,
        }

    result = {
        "timestamp_utc": _iso_now(),
        "relative_path": rel,
        "target_path": str(target),
        "status": status,
        "source_bytes": source_size,
        "target_bytes": copied_size,
        "source_sha256": source_hash,
        "target_sha256": copied_hash,
        "source_retained": True,
        "source_delete_requires_retention_gate": True,
    }
    proof_path.parent.mkdir(parents=True, exist_ok=True)
    with proof_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=True) + "\n")
    return result


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    target_root: str = "",
    apply: bool = False,
    max_files: int = 4,
    max_gb: float = 4.0,
    out_path: Path = DEFAULT_OUT_PATH,
    proof_path: Path = DEFAULT_PROOF_PATH,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    target, target_candidates = _resolve_target(target_root)
    entries = _eligible_entries(manifest, project_root=project_root) if manifest else []
    selected = _select_entries(
        entries,
        max_files=max(int(max_files), 1),
        max_bytes=max(int(float(max_gb) * 1024 * 1024 * 1024), 1),
    )
    selected_bytes = sum(_safe_int(row.get("size_bytes"), 0) for row in selected)
    results: list[dict[str, Any]] = []
    if apply and target is not None:
        for entry in selected:
            results.append(_copy_verify_one(project_root=project_root, target_root=target, entry=entry, proof_path=proof_path))

    copied = [row for row in results if str(row.get("status") or "") == "copied_verified"]
    errors = [row for row in results if str(row.get("status") or "") not in {"copied_verified"}]
    if not manifest:
        status = "blocked_missing_manifest"
        ok = False
    elif not selected:
        status = "nothing_to_do"
        ok = True
    elif target is None:
        status = "blocked_waiting_for_cold_target"
        ok = False
    elif apply and errors:
        status = "degraded"
        ok = False
    elif apply:
        status = "applied"
        ok = True
    else:
        status = "planned"
        ok = True

    payload = {
        "timestamp_utc": _iso_now(),
        "schema_version": 1,
        "ok": ok,
        "overall_status": status,
        "apply": bool(apply),
        "manifest_path": str(manifest_path),
        "target_root": str(target) if target is not None else "",
        "target_candidates": target_candidates,
        "selected_count": len(selected),
        "selected_gb": _gb(selected_bytes),
        "eligible_count": len(entries),
        "apply_result": {
            "copied_count": len(copied),
            "copied_gb": _gb(sum(_safe_int(row.get("source_bytes"), 0) for row in copied)),
            "error_count": len(errors),
            "proof_path": str(proof_path),
            "source_delete_performed": False,
        },
        "policy": {
            "mode": "manifest_backed_copy_verify_restore_proof",
            "source_delete_policy": "never delete source in this worker; retention gate must consume restore proofs first",
            "never_touch_protected_volumes": list(PROTECTED_VOLUME_PREFIXES),
            "required_entry_classification": "eligible_manifest_backed_offload",
        },
        "selected_entries": selected[:25],
        "results": results[:25],
        "next_action": (
            "configure BOT_SECOND_COLD_ROOT or mount BOT_COLD/BOT_ARCHIVE before offload copy-verify can run"
            if status == "blocked_waiting_for_cold_target"
            else "offload copies have restore proofs; source retention remains separately gated"
            if status == "applied"
            else "run with --apply during a maintenance window to copy and verify selected offload candidates"
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy manifest-approved hot-path artifacts to a cold target with restore proofs.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--manifest-file", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--target-root", default="")
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--proof-file", default=str(DEFAULT_PROOF_PATH))
    parser.add_argument("--max-files", type=int, default=int(os.getenv("MANIFEST_BACKED_OFFLOAD_MAX_FILES", "4")))
    parser.add_argument("--max-gb", type=float, default=float(os.getenv("MANIFEST_BACKED_OFFLOAD_MAX_GB", "4.0")))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        Path(args.project_root).resolve(),
        manifest_path=Path(args.manifest_file).expanduser(),
        target_root=str(args.target_root or ""),
        apply=bool(args.apply),
        max_files=int(args.max_files),
        max_gb=float(args.max_gb),
        out_path=Path(args.out_file).expanduser(),
        proof_path=Path(args.proof_file).expanduser(),
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "manifest_backed_offload_worker "
            f"status={payload.get('overall_status', '')} "
            f"selected={payload.get('selected_count', 0)} "
            f"target={payload.get('target_root', '')}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
