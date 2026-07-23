from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT_ROOT / "governance" / "health" / "support_maintenance_gate_latest.json"


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    except Exception:
        return {}
    return values


def _fresh_enough(path: Path, *, max_age_seconds: float) -> bool:
    try:
        age = datetime.now(timezone.utc).timestamp() - path.stat().st_mtime
        return age <= max(max_age_seconds, 1.0)
    except Exception:
        return False


def support_maintenance_freeze_contract(project_root: Path, component: str) -> dict[str, Any]:
    """Return an active contract when noncritical support work should yield."""
    project_root = Path(project_root)
    max_age_seconds = float(os.getenv("SUPPORT_MAINTENANCE_FREEZE_MAX_AGE_SECONDS", "1800"))
    override_path = project_root / "config" / ".env.runtime_resource_guard_override"
    runtime_path = project_root / "governance" / "health" / "runtime_throttle_control_latest.json"
    env_file = _load_env_file(override_path)
    runtime = _load_json(runtime_path)
    mac = runtime.get("mac_fluidity_contract") if isinstance(runtime.get("mac_fluidity_contract"), dict) else {}
    support_pause = (
        runtime.get("apply_result", {}).get("support_maintenance_pause")
        if isinstance(runtime.get("apply_result"), dict)
        else {}
    )
    support_pause = support_pause if isinstance(support_pause, dict) else {}

    env_freeze = bool(
        _truthy(os.getenv("OPS_SUPPORT_MAINTENANCE_FREEZE"))
        or _truthy(os.getenv("MAC_FLUIDITY_SUPPORT_PAUSE"))
        or str(os.getenv("SUPPORT_MAINTENANCE_CONCURRENCY", "")).strip() == "0"
        or _truthy(env_file.get("OPS_SUPPORT_MAINTENANCE_FREEZE"))
        or _truthy(env_file.get("MAC_FLUIDITY_SUPPORT_PAUSE"))
        or str(env_file.get("SUPPORT_MAINTENANCE_CONCURRENCY", "")).strip() == "0"
    )
    runtime_freeze = bool(
        _truthy(mac.get("support_pause_recommended"))
        or _truthy(support_pause.get("pause_requested"))
        or (
            _truthy(os.getenv("SUPPORT_MAINTENANCE_FREEZE_ON_PROTECT"))
            and str(mac.get("overall_status") or "").strip().lower() == "needs_work"
            and str(mac.get("fluidity_band") or "").strip().lower() in {"strained", "protect"}
        )
    )
    fresh_runtime = _fresh_enough(runtime_path, max_age_seconds=max_age_seconds)
    fresh_override = _fresh_enough(override_path, max_age_seconds=max_age_seconds)
    active = bool((env_freeze and (fresh_override or fresh_runtime)) or (runtime_freeze and fresh_runtime))
    return {
        "active": active,
        "component": str(component),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "support_maintenance_frozen_for_mac_fluidity" if active else "support_maintenance_ready",
        "env_freeze": env_freeze,
        "runtime_freeze": runtime_freeze,
        "fresh_runtime": fresh_runtime,
        "fresh_override": fresh_override,
        "override_path": str(override_path),
        "runtime_path": str(runtime_path),
        "mac_fluidity": {
            "overall_status": mac.get("overall_status", ""),
            "fluidity_band": mac.get("fluidity_band", ""),
            "fluidity_score": mac.get("fluidity_score", 0.0),
            "support_pause_recommended": bool(mac.get("support_pause_recommended", False)),
        },
        "policy": "noncritical_support_jobs_exit_before_heavy_work_when_mac_fluidity_is_frozen",
    }


def frozen_health_payload(previous_path: Path, contract: dict[str, Any], *, ok: bool = True) -> dict[str, Any]:
    previous = _load_json(Path(previous_path))
    payload = dict(previous)
    for key in ("busy", "lock_owner", "lock_path"):
        payload.pop(key, None)
    payload.update(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "ok": bool(ok),
            "overall_status": "ready" if ok else "blocked",
            "support_maintenance_frozen": True,
            "skipped_reason": str(contract.get("reason") or "support_maintenance_frozen_for_mac_fluidity"),
            "support_maintenance_freeze_contract": contract,
        }
    )
    return payload


def build_payload(project_root: Path, *, component: str = "support_maintenance_gate") -> dict[str, Any]:
    contract = support_maintenance_freeze_contract(project_root, component)
    active = bool(contract.get("active", False))
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": True,
        "overall_status": "ready",
        "support_maintenance_freeze_active": active,
        "component": str(component),
        "freeze_reason": str(contract.get("reason") or ""),
        "support_maintenance_freeze_contract": contract,
        "memory_pressure_allocation_role": "yieldable_support_lane",
        "lane_policy": {
            "live_execution": "protected_read_only_gate",
            "paper_collection": "protected_collection",
            "sql_writer": "single_writer_guarded_priority",
            "research_training": "pause_or_downshift_first",
            "support_report_media": "off_hours_or_nice_20",
        },
        "recommended_actions": (
            ["keep support, report, media, and maintenance jobs out of the hot path until fluidity recovers"]
            if active
            else ["support maintenance is clear to run within normal runtime throttle limits"]
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report whether noncritical support maintenance should yield to runtime pressure.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--component", default="support_maintenance_gate")
    parser.add_argument("--out-file", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    payload = build_payload(args.project_root, component=args.component)
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        state = "active" if payload["support_maintenance_freeze_active"] else "ready"
        print(f"support_maintenance_gate={state} component={payload['component']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
