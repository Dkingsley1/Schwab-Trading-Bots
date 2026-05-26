#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.accountability import write_registry_mutation_journal

DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "master_bot_registry.json"
DEFAULT_AUDIT_PATH = PROJECT_ROOT / "governance" / "health" / "training_registry_audit_latest.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "training_probation_isolation_latest.json"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _audit_quality_failed_ids(audit: dict[str, Any]) -> list[str]:
    rows = audit.get("active_quality_failed") if isinstance(audit.get("active_quality_failed"), list) else []
    out: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        bot_id = str(row.get("bot_id") or "").strip().lower()
        if not bot_id or bot_id in seen:
            continue
        seen.add(bot_id)
        out.append(bot_id)
    return out


def build_payload(
    *,
    project_root: Path = PROJECT_ROOT,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    audit_path: Path = DEFAULT_AUDIT_PATH,
    apply: bool = False,
    limit: int = 0,
) -> dict[str, Any]:
    registry = _load_json(registry_path)
    audit = _load_json(audit_path)
    target_ids = _audit_quality_failed_ids(audit)
    if limit > 0:
        target_ids = target_ids[:limit]

    rows = registry.get("sub_bots") if isinstance(registry.get("sub_bots"), list) else []
    before = json.loads(json.dumps(registry)) if isinstance(registry, dict) else {}
    isolated: list[dict[str, Any]] = []
    already_isolated: list[dict[str, Any]] = []
    missing: list[str] = []
    target_set = set(target_ids)

    by_id = {str(row.get("bot_id") or "").strip().lower(): row for row in rows if isinstance(row, dict)}
    for bot_id in target_ids:
        row = by_id.get(bot_id)
        if not isinstance(row, dict):
            missing.append(bot_id)
            continue
        was_training_excluded = bool(row.get("training_excluded", False) or row.get("exclude_from_training", False))
        record = {
            "bot_id": bot_id,
            "was_training_excluded": was_training_excluded,
            "lifecycle_state": str(row.get("lifecycle_state") or ""),
            "reason": str(row.get("reason") or ""),
            "promotion_reason": str(row.get("promotion_reason") or ""),
        }
        if was_training_excluded:
            already_isolated.append(record)
            continue
        if apply:
            row["training_excluded"] = True
            row["exclude_from_training"] = True
            row["training_exclusion_reason"] = "quality_probation_isolation"
            row["promotion_blocked_reason"] = "quality_probation_isolation"
            row["quality_probation_isolated_at_utc"] = datetime.now(timezone.utc).isoformat()
        isolated.append(record)

    registry_updated = False
    backup_file = ""
    journal_error = ""
    if apply and isolated and isinstance(registry, dict):
        lifecycle_dir = project_root / "governance" / "lifecycle"
        lifecycle_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup = lifecycle_dir / f"master_bot_registry.quality_probation_isolation_backup_{stamp}.json"
        if registry_path.exists():
            backup.write_text(registry_path.read_text(encoding="utf-8"), encoding="utf-8")
            backup_file = str(backup)
        registry_path.write_text(json.dumps(registry, ensure_ascii=True, indent=2), encoding="utf-8")
        registry_updated = True
        try:
            write_registry_mutation_journal(
                project_root=str(project_root),
                actor="training_probation_isolation",
                reason="isolate_quality_probation_bots_from_training",
                before=before,
                after=registry,
                extra={"isolated_bot_ids": sorted(target_set), "isolated_rows": isolated[:80]},
            )
        except Exception as exc:
            journal_error = str(exc)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": True,
        "apply_requested": bool(apply),
        "registry_updated": bool(registry_updated),
        "backup_file": backup_file,
        "target_count": len(target_ids),
        "newly_isolated_count": len(isolated),
        "already_isolated_count": len(already_isolated),
        "missing_count": len(missing),
        "target_bot_ids": target_ids,
        "newly_isolated": isolated,
        "already_isolated": already_isolated,
        "missing_bot_ids": missing,
        "journal_error": journal_error,
        "policy": {
            "live_execution_changed": False,
            "paper_collection_allowed": True,
            "training_excluded_until_quality_gate_recovers": True,
        },
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Isolate active quality-probation bots from training/promotion without disabling paper observation.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--registry-path", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--audit-path", default=str(DEFAULT_AUDIT_PATH))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    payload = build_payload(
        project_root=project_root,
        registry_path=Path(args.registry_path).expanduser(),
        audit_path=Path(args.audit_path).expanduser(),
        apply=bool(args.apply),
        limit=int(args.limit),
    )
    _write_json(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "training_probation_isolation "
            f"targets={payload['target_count']} newly_isolated={payload['newly_isolated_count']} "
            f"registry_updated={str(payload['registry_updated']).lower()}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
