#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "core_bot_materialization_guard_latest.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _registry_rows(project_root: Path) -> list[dict[str, Any]]:
    registry = _load_json(project_root / "master_bot_registry.json")
    rows = registry.get("sub_bots") if isinstance(registry.get("sub_bots"), list) else []
    return [row for row in rows if isinstance(row, dict)]


def _should_have_core_file(row: dict[str, Any]) -> bool:
    bot_id = str(row.get("bot_id") or "")
    return (
        bot_id.startswith("brain_refinery_v")
        and bool(row.get("active"))
        and bool(row.get("data_collection_active"))
        and str(row.get("reason") or "") == "planned_roster_expansion_slot"
        and str(row.get("lifecycle_state") or "") == "data_collection_only"
    )


def _duplicate_core_versions(project_root: Path) -> dict[str, list[str]]:
    versions: dict[str, list[str]] = {}
    for path in sorted((project_root / "core").glob("brain_refinery_v*.py")):
        match = re.match(r"brain_refinery_v(\d+)_", path.name)
        if not match:
            continue
        versions.setdefault(match.group(1), []).append(path.name)
    return {version: names for version, names in versions.items() if len(names) > 1}


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    rows = _registry_rows(project_root)
    required = [row for row in rows if _should_have_core_file(row)]
    missing: list[str] = []
    present: list[str] = []
    for row in required:
        bot_id = str(row.get("bot_id") or "")
        rel = f"core/{bot_id}.py"
        if (project_root / rel).exists():
            present.append(bot_id)
        else:
            missing.append(bot_id)
    duplicate_versions = _duplicate_core_versions(project_root)
    overall_status = "ready" if not missing and not duplicate_versions else "degraded"
    return {
        "overall_status": overall_status,
        "generated_at_utc": _utc_now(),
        "summary": {
            "required_core_module_count": len(required),
            "present_core_module_count": len(present),
            "missing_core_module_count": len(missing),
            "duplicate_core_version_count": len(duplicate_versions),
        },
        "missing_core_modules": missing,
        "duplicate_core_versions": duplicate_versions,
        "present_core_modules": present,
        "recommended_actions": [
            *([] if not missing else ["run ./scripts/ops/opsctl.sh core-bot-materialize --json to create missing PyCharm-visible bot files"]),
            *([] if not duplicate_versions else ["renumber or archive duplicate brain_refinery_v### core files so PyCharm shows one bot per version"]),
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify registry-backed expansion bots have visible core/*.py modules.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    payload = build_payload(project_root)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = project_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        summary = payload["summary"]
        print(
            "core_bot_materialization_guard "
            f"status={payload['overall_status']} "
            f"present={summary['present_core_module_count']} "
            f"missing={summary['missing_core_module_count']} "
            f"required={summary['required_core_module_count']}"
        )
    return 0 if payload["overall_status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
