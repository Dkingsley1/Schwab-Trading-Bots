#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import storage_router


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "storage_split_brain_reconciler_latest.json"
DEFAULT_MARKDOWN_PATH = PROJECT_ROOT / "exports" / "reports" / "storage" / "storage_split_brain_reconciler_latest.md"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _local_fallback_root(project_root: Path) -> Path:
    return Path(os.getenv("BOT_LOGS_LOCAL_FALLBACK_ROOT", str(project_root / storage_router.DEFAULT_LOCAL_FALLBACK))).expanduser()


def _strip_conflict_suffix(path: Path) -> Path:
    name = path.name
    if ".local_fallback" not in name:
        return path
    base = name.split(".local_fallback", 1)[0]
    return path.with_name(base)


def _iter_conflict_files(external_root: Path) -> list[Path]:
    if not external_root.exists():
        return []
    matches: list[Path] = []
    for root, _, files in os.walk(external_root):
        for fname in files:
            if ".local_fallback" in fname:
                matches.append(Path(root) / fname)
    return sorted(matches)


def _render_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    lines = [
        "# Storage Split-Brain Reconciliation",
        "",
        f"- Timestamp UTC: `{payload.get('timestamp_utc', '')}`",
        f"- External Root: `{payload.get('external_root', '')}`",
        f"- Local Fallback Root: `{payload.get('local_root', '')}`",
        f"- Conflict Files: `{int(summary.get('conflict_files', 0) or 0)}`",
        f"- Hash Match Ready: `{int(summary.get('hash_match_ready', 0) or 0)}`",
        f"- Unresolved Conflicts: `{int(summary.get('unresolved_conflicts', 0) or 0)}`",
        f"- Force Failback Eligible: `{bool(summary.get('force_failback_eligible', False))}`",
        "",
        "## Top Conflicts",
        "",
    ]
    for row in (payload.get("conflicts") or [])[:12]:
        if not isinstance(row, dict):
            continue
        lines.append(
            "- `{rel}`: class=`{cls}` local_hash_match=`{match}`".format(
                rel=str(row.get("relative_path") or ""),
                cls=str(row.get("classification") or ""),
                match=bool(row.get("hashes_match", False)),
            )
        )
    return "\n".join(lines) + "\n"


def build_payload(project_root: Path = PROJECT_ROOT, *, full_scan: bool = False) -> dict[str, Any]:
    external_root = storage_router._external_project_root()
    local_root = _local_fallback_root(project_root)
    failback_payload = _load_json(project_root / "governance" / "health" / "storage_failback_sync_latest.json")
    mount_payload = _load_json(project_root / "governance" / "health" / "storage_mount_guard_latest.json")

    reported_split_brain_conflicts = max(
        int(failback_payload.get("split_brain_conflicts", 0) or 0),
        int(mount_payload.get("storage_mount_transition", {}).get("recovery", {}).get("payload", {}).get("split_brain_conflicts", 0) or 0),
    )
    scan_mode = "full_scan" if full_scan else "manifest_fast_path"
    rows: list[dict[str, Any]] = []
    if full_scan or reported_split_brain_conflicts > 0:
        for conflict_path in _iter_conflict_files(external_root):
            canonical_external = _strip_conflict_suffix(conflict_path)
            relative_path = str(canonical_external.relative_to(external_root)) if canonical_external.exists() or external_root.exists() else canonical_external.name
            local_counterpart = local_root / relative_path
            hashes_match = False
            local_hash = ""
            external_hash = ""
            conflict_hash = ""
            if conflict_path.exists():
                conflict_hash = _sha(conflict_path)
            if local_counterpart.exists() and local_counterpart.is_file():
                local_hash = _sha(local_counterpart)
            if canonical_external.exists() and canonical_external.is_file():
                external_hash = _sha(canonical_external)
            if local_hash and external_hash and local_hash == external_hash:
                hashes_match = True
                classification = "ready_to_prune_local"
            elif conflict_hash and external_hash and conflict_hash == external_hash:
                hashes_match = True
                classification = "duplicate_conflict_copy"
            elif not canonical_external.exists():
                classification = "external_missing_keep_local"
            elif not local_counterpart.exists():
                classification = "conflict_copy_only_review"
            else:
                classification = "divergent_hash_manual_review"
            rows.append(
                {
                    "conflict_file": str(conflict_path),
                    "relative_path": relative_path,
                    "external_path": str(canonical_external),
                    "local_path": str(local_counterpart),
                    "classification": classification,
                    "hashes_match": hashes_match,
                    "local_hash": local_hash[:16],
                    "external_hash": external_hash[:16],
                    "conflict_hash": conflict_hash[:16],
                }
            )

    rows.sort(key=lambda row: (str(row.get("classification") or ""), str(row.get("relative_path") or "")))
    hash_match_ready = sum(1 for row in rows if bool(row.get("hashes_match", False)))
    unresolved_conflicts = sum(1 for row in rows if not bool(row.get("hashes_match", False)))
    split_brain_conflicts = max(
        reported_split_brain_conflicts,
        unresolved_conflicts,
    )
    force_failback_eligible = bool(mount_payload.get("external_available", False)) and split_brain_conflicts == 0

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "external_root": str(external_root),
        "local_root": str(local_root),
        "external_available": bool(mount_payload.get("external_available", False)),
        "current_storage_mode": str(mount_payload.get("storage_mode") or failback_payload.get("mode") or ""),
        "scan_mode": scan_mode,
        "summary": {
            "conflict_files": len(rows),
            "hash_match_ready": hash_match_ready,
            "unresolved_conflicts": unresolved_conflicts,
            "reported_split_brain_conflicts": split_brain_conflicts,
            "force_failback_eligible": force_failback_eligible,
        },
        "conflicts": rows[:50],
        "recommended_actions": [
            "prune duplicate local fallback copies when hashes already match external storage",
            "hold force failback until unresolved_conflicts reaches zero",
            "use the markdown report to review divergent hashes before deleting any conflict copies",
            "run with --full-scan when you want to sweep the entire external tree for forensic reconciliation",
        ],
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify and report BOT_LOGS split-brain conflicts.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN_PATH))
    parser.add_argument("--force-failback-if-hashes-match", action="store_true")
    parser.add_argument("--full-scan", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    payload = build_payload(project_root, full_scan=bool(args.full_scan))

    if args.force_failback_if_hashes_match and bool(payload.get("summary", {}).get("force_failback_eligible", False)):
        os.environ["BOT_LOGS_PREFER_EXTERNAL"] = "1"
        routing = storage_router.route_runtime_storage(project_root)
        payload["forced_failback"] = {
            "attempted": True,
            "mode": routing.mode,
            "active_root": str(routing.active_root),
            "split_brain_conflicts": int(routing.split_brain_conflicts),
        }
    else:
        payload["forced_failback"] = {
            "attempted": bool(args.force_failback_if_hashes_match),
            "mode": "",
            "active_root": "",
            "split_brain_conflicts": int(payload.get("summary", {}).get("reported_split_brain_conflicts", 0) or 0),
        }

    out_path = Path(args.out_file).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    markdown_path = Path(args.markdown_out).expanduser()
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "storage_split_brain_reconciler "
            f"unresolved_conflicts={int(payload.get('summary', {}).get('unresolved_conflicts', 0) or 0)} "
            f"force_failback_eligible={bool(payload.get('summary', {}).get('force_failback_eligible', False))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
