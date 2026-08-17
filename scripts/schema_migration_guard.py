#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "migrations" / "latest.json"

CONTRACT_SPECS = [
    {
        "name": "paper_performance",
        "path": Path("governance/health/paper_performance_latest.json"),
        "required_keys": ["schema_version", "ok", "sleeve_latest"],
        "compatibility": "consumer_backward_compatible",
    },
    {
        "name": "point_in_time_event_store",
        "path": Path("governance/health/point_in_time_event_store_latest.json"),
        "required_keys": ["ok", "event_count", "events"],
        "compatibility": "consumer_backward_compatible",
    },
    {
        "name": "training_quality_control",
        "path": Path("governance/health/training_quality_control_latest.json"),
        "required_keys": ["overall_status", "training_quality_score", "improvements"],
        "compatibility": "additive_only",
    },
    {
        "name": "platform_control_plane",
        "path": Path("governance/health/platform_control_plane_latest.json"),
        "required_keys": ["institutional_readiness", "institutional_domains_by_slug"],
        "compatibility": "additive_only",
    },
    {
        "name": "feature_store_manifest",
        "path": Path("governance/feature_store/latest.json"),
        "required_keys": ["schema_version", "dataset_contract", "point_in_time_contract"],
        "compatibility": "point_in_time_contract_stable",
    },
    {
        "name": "security_audit",
        "path": Path("governance/health/security_audit_latest.json"),
        "required_keys": ["overall_status", "checks"],
        "compatibility": "consumer_backward_compatible",
    },
]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    rows: list[dict[str, Any]] = []
    missing = 0
    legacy_unversioned = 0

    for spec in CONTRACT_SPECS:
        path = project_root / spec["path"]
        payload = _load_json(path)
        present = bool(payload)
        schema_version = payload.get("schema_version") if present else None
        missing_keys = []
        if present:
            for key in spec["required_keys"]:
                if key not in payload:
                    missing_keys.append(key)
        status = "ready"
        if not present:
            status = "missing"
            missing += 1
        elif schema_version is None:
            status = "legacy_unversioned"
            legacy_unversioned += 1
        elif missing_keys:
            status = "needs_work"
        rows.append(
            {
                "name": spec["name"],
                "path": str(path),
                "present": present,
                "schema_version": schema_version,
                "status": status,
                "compatibility": spec["compatibility"],
                "missing_keys": missing_keys,
            }
        )

    ok = missing == 0 and legacy_unversioned == 0
    overall_status = "ready" if ok else "needs_work"
    if missing > 0:
        overall_status = "blocked"

    return {
        "timestamp_utc": now.isoformat(),
        "schema_version": 1,
        "ok": ok,
        "overall_status": overall_status,
        "contracts": rows,
        "summary": {
            "contract_count": len(rows),
            "missing_contracts": missing,
            "legacy_unversioned_contracts": legacy_unversioned,
            "needs_work_contracts": sum(1 for row in rows if row["status"] == "needs_work"),
        },
        "recommendations": [
            "Require schema_version or contract_version on every new operator-facing artifact.",
            "Treat replay-sensitive contract changes as additive-only unless the migration manifest is updated in the same change.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a migration and schema contract manifest for operator-facing artifacts.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve())
    out_path = Path(args.out_file).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "schema_migration_guard "
            f"status={payload['overall_status']} "
            f"contracts={int(((payload.get('summary') or {}).get('contract_count', 0) or 0))}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
