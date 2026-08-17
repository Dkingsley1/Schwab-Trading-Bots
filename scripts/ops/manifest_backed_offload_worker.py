#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
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


def _env_truthy(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes", "y", "on"}


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
    release_source_after_verify: bool = False,
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
    if release_source_after_verify and status == "copied_verified":
        result.update(_release_source_with_retention_gate(source=source, target=target, entry=entry, result=result))
    proof_path.parent.mkdir(parents=True, exist_ok=True)
    with proof_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=True) + "\n")
    return result


def _release_source_with_retention_gate(*, source: Path, target: Path, entry: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    allowed = {str(item) for item in (entry.get("allowed_actions") or [])}
    proof = entry.get("proof_required") if isinstance(entry.get("proof_required"), dict) else {}
    if str(entry.get("classification") or "") != "eligible_manifest_backed_offload":
        return {"source_release_status": "blocked_not_manifest_eligible", "source_retained": True}
    if "source_delete_requires_retention_gate" not in allowed:
        return {"source_release_status": "blocked_missing_retention_gate_action", "source_retained": True}
    if not bool(proof.get("retention_gate_before_source_delete", False)):
        return {"source_release_status": "blocked_missing_retention_gate_proof", "source_retained": True}
    if not target.exists() or not source.exists():
        return {"source_release_status": "blocked_missing_source_or_target", "source_retained": True}
    try:
        source_hash = _sha256(source)
        target_hash = _sha256(target)
        source_size = int(source.stat().st_size)
        target_size = int(target.stat().st_size)
        if (
            source_hash != target_hash
            or source_hash != str(result.get("source_sha256") or "")
            or target_hash != str(result.get("target_sha256") or "")
            or source_size != target_size
        ):
            return {
                "source_release_status": "blocked_restore_proof_mismatch",
                "source_retained": True,
                "source_release_error": "hash_or_size_mismatch",
            }
        source.unlink()
    except Exception as exc:
        return {"source_release_status": "error", "source_retained": True, "source_release_error": str(exc)}
    return {
        "status": "copied_verified_source_released",
        "source_release_status": "released_after_restore_proof",
        "source_retained": False,
        "source_released": True,
        "source_release_bytes": source_size,
    }


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    target_root: str = "",
    apply: bool = False,
    max_files: int = 4,
    max_gb: float = 4.0,
    release_source_after_verify: bool = False,
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
            results.append(
                _copy_verify_one(
                    project_root=project_root,
                    target_root=target,
                    entry=entry,
                    proof_path=proof_path,
                    release_source_after_verify=bool(release_source_after_verify),
                )
            )

    copied_statuses = {"copied_verified", "copied_verified_source_released"}
    copied = [row for row in results if str(row.get("status") or "") in copied_statuses]
    released = [row for row in results if bool(row.get("source_released", False))]
    errors = [
        row
        for row in results
        if str(row.get("status") or "") not in copied_statuses
        or (release_source_after_verify and not bool(row.get("source_released", False)))
    ]
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
            "released_count": len(released),
            "released_gb": _gb(sum(_safe_int(row.get("source_release_bytes"), 0) for row in released)),
            "error_count": len(errors),
            "proof_path": str(proof_path),
            "source_delete_performed": bool(released),
            "release_source_after_verify": bool(release_source_after_verify),
        },
        "policy": {
            "mode": "manifest_backed_copy_verify_restore_proof",
            "source_delete_policy": (
                "release only manifest eligible sources after copy, hash verify, and restore proof"
                if release_source_after_verify
                else "never delete source in this worker; retention gate must consume restore proofs first"
            ),
            "never_touch_protected_volumes": list(PROTECTED_VOLUME_PREFIXES),
            "approved_video_cold_archive": {
                "enabled": False,
                "root": "",
                "scope": "forbidden",
            },
            "required_entry_classification": "eligible_manifest_backed_offload",
        },
        "selected_entries": selected[:25],
        "results": results[:25],
        "next_action": (
            "configure BOT_SECOND_COLD_ROOT or mount BOT_COLD/BOT_ARCHIVE before offload copy-verify can run"
            if status == "blocked_waiting_for_cold_target"
            else "offload copies have restore proofs; source retention remains separately gated"
            if status == "applied" and not released
            else "offload copies have restore proofs and eligible sources were released by retention gate"
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
    parser.add_argument("--release-source-after-verify", action="store_true")
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
        release_source_after_verify=bool(args.release_source_after_verify),
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
