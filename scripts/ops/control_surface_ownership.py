#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "control_surface_ownership_v1.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "control_surface_ownership_latest.json"
COORDINATION_MODES = {"file_lock", "sqlite_immediate_transaction", "atomic_replace", "operator_only"}
MUTABLE_MODES = {"mutable_state", "transactional_state"}


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _project_path(project_root: Path, raw: Any) -> Path:
    path = Path(str(raw or ""))
    return path if path.is_absolute() else project_root / path


def _command_routable(project_root: Path, command: list[Any]) -> tuple[bool, str]:
    parts = [str(item).strip() for item in command if str(item).strip()]
    if not parts:
        return False, "owner_command_missing"
    executable = _project_path(project_root, parts[0])
    if not executable.is_file():
        return False, f"owner_executable_missing:{parts[0]}"
    if executable.name != "opsctl.sh":
        return True, "direct_owner_present"
    if len(parts) < 2:
        return False, "opsctl_owner_route_missing"
    try:
        routed = parts[1] in executable.read_text(encoding="utf-8")
    except OSError:
        routed = False
    return routed, "opsctl_owner_route_present" if routed else f"opsctl_owner_route_missing:{parts[1]}"


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    config_path: Path | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    config_path = config_path or project_root / "config" / DEFAULT_CONFIG_PATH.name
    config = load_json(config_path)
    controls = [row for row in config.get("controls", []) if isinstance(row, dict)]
    control_ids = [str(row.get("control_id") or "").strip() for row in controls]
    resource_paths = [str(row.get("resource_path") or "").strip() for row in controls]
    duplicate_ids = sorted({item for item in control_ids if item and control_ids.count(item) > 1})
    duplicate_resources = sorted({item for item in resource_paths if item and resource_paths.count(item) > 1})
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    source_receipts: dict[str, str] = {}

    if not config:
        blockers.append("ownership_registry_missing_or_invalid")
    if duplicate_ids:
        blockers.extend(f"duplicate_control_id:{item}" for item in duplicate_ids)
    if duplicate_resources:
        blockers.extend(f"duplicate_resource_owner:{item}" for item in duplicate_resources)

    for index, spec in enumerate(controls):
        control_id = str(spec.get("control_id") or f"unnamed_{index}").strip()
        resource_path = str(spec.get("resource_path") or "").strip()
        owner_source_raw = str(spec.get("owner_source") or "").strip()
        owner_source = _project_path(project_root, owner_source_raw)
        owner_exists = bool(owner_source_raw and owner_source.is_file())
        marker = str(spec.get("owner_marker") or "").strip()
        try:
            owner_text = owner_source.read_text(encoding="utf-8") if owner_exists else ""
        except OSError:
            owner_text = ""
        marker_present = bool(marker and marker in owner_text)
        command_ready, command_detail = _command_routable(project_root, list(spec.get("owner_command") or []))
        mutation_mode = str(spec.get("mutation_mode") or "").strip()
        coordination = str(spec.get("coordination") or "").strip()
        lock_path = str(spec.get("lock_path") or "").strip()
        coordination_ready = bool(coordination in COORDINATION_MODES)
        if mutation_mode in MUTABLE_MODES and coordination == "file_lock":
            coordination_ready = bool(lock_path)
        row_blockers = ordered_unique(
            [
                "resource_path_missing" if not resource_path else "",
                "owner_source_missing" if not owner_exists else "",
                "owner_marker_missing" if owner_exists and not marker_present else "",
                command_detail if not command_ready else "",
                "coordination_contract_missing" if not coordination_ready else "",
                "duplicate_control_id" if control_id in duplicate_ids else "",
                "duplicate_resource_owner" if resource_path in duplicate_resources else "",
            ]
        )
        ready = not row_blockers
        rows.append(
            {
                "control_id": control_id,
                "resource_path": resource_path,
                "owner_source": owner_source_raw,
                "owner_command": list(spec.get("owner_command") or []),
                "mutation_mode": mutation_mode,
                "coordination": coordination,
                "lock_path": lock_path,
                "ready": ready,
                "blockers": row_blockers,
            }
        )
        if owner_exists:
            source_receipts[owner_source_raw] = _sha256(owner_source)
        blockers.extend(f"{control_id}:{item}" for item in row_blockers)

    receipt_input = {
        "config_sha256": _sha256(config_path),
        "source_receipts": dict(sorted(source_receipts.items())),
        "resources": sorted(resource_paths),
    }
    receipt_sha = hashlib.sha256(
        json.dumps(receipt_input, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    blockers = ordered_unique(blockers)
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": not blockers,
        "overall_status": "ready" if not blockers else "blocked",
        "grade": "A+" if not blockers else "F",
        "policy_id": str(config.get("policy_id") or ""),
        "control_count": len(rows),
        "ready_control_count": sum(1 for row in rows if row["ready"]),
        "controls": rows,
        "duplicate_control_ids": duplicate_ids,
        "duplicate_resource_paths": duplicate_resources,
        "blockers": blockers,
        "evidence_epoch": {
            "id": f"control-ownership:{receipt_sha[:16]}",
            "receipt_sha256": receipt_sha,
            "config_sha256": receipt_input["config_sha256"],
            "source_receipts": receipt_input["source_receipts"],
        },
        "control_contract": {
            "one_declared_writer_per_resource": not duplicate_resources,
            "owners_are_source_backed": all(row["ready"] for row in rows),
            "mutable_automation_is_coordinated": all(
                row["coordination"] in COORDINATION_MODES for row in rows
            ),
            "automatic_live_execution_authority": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate exclusive ownership of production control surfaces.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out-file", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config or Path("config/control_surface_ownership_v1.json")
    out_path = args.out_file or Path("governance/health/control_surface_ownership_latest.json")
    config_path = config_path if config_path.is_absolute() else project_root / config_path
    out_path = out_path if out_path.is_absolute() else project_root / out_path
    payload = build_payload(project_root, config_path=config_path)
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "control_surface_ownership "
            f"status={payload['overall_status']} ready={payload['ready_control_count']}/{payload['control_count']}"
        )
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
