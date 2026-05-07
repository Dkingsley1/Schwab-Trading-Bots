#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "health_fast_latest.json"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _health(project_root: Path, name: str) -> dict[str, Any]:
    payload = load_json(project_root / "governance" / "health" / name)
    return payload if isinstance(payload, dict) else {}


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    process = _health(project_root, "process_watchdog_latest.json")
    runtime = _health(project_root, "runtime_throttle_control_latest.json")
    memory = _health(project_root, "memory_efficiency_control_latest.json")
    swap = _health(project_root, "swap_pressure_governor_latest.json")
    pressure = _health(project_root, "pressure_relief_control_latest.json")
    platform = _health(project_root, "platform_intelligence_expansion_latest.json")
    brain_v4 = _health(project_root, "platform_brain_v4_latest.json")
    brain_v5 = _health(project_root, "platform_brain_v5_latest.json")
    stabilizer = _health(project_root, "platform_stabilization_quality_latest.json")
    settlement = _health(project_root, "platform_settlement_stabilization_latest.json")
    rollup = _health(project_root, "data_collection_observation_rollup_latest.json")
    halt = _health(project_root, "global_halt_auto_clear_latest.json") or _health(project_root, "global_killswitch_latest.json")
    storage = _health(project_root, "ingestion_storage_control_latest.json")
    schwab_futures = _health(project_root, "data_ingress_latest_schwab_futures_equities_schwab.json")
    alerts = process.get("alerts") if isinstance(process.get("alerts"), list) else []
    safety = process.get("safety_pause") if isinstance(process.get("safety_pause"), dict) else {}
    swap_payload = swap.get("swap_pressure") if isinstance(swap.get("swap_pressure"), dict) else {}
    ready = bool(
        not alerts
        and not bool(safety.get("active", False))
        and not bool(halt.get("halt", False))
        and str(rollup.get("overall_status") or "").lower() in {"ready", ""}
        and str(swap_payload.get("tier") or "normal").lower() in {"normal", "calm", ""}
    )
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": ready,
        "overall_status": "ready" if ready else "degraded",
        "read_only": True,
        "started_heavy_reports": False,
        "global_halt": {
            "halt": bool(halt.get("halt", False)),
            "halt_state": halt.get("halt_state", "unknown"),
            "clear_blockers": halt.get("clear_blockers", []),
        },
        "process_watchdog": {
            "alerts": alerts,
            "safety_pause": safety,
            "status": process.get("status", []),
        },
        "runtime_pressure": {
            "overall_status": runtime.get("overall_status"),
            "tier": pressure.get("tier"),
            "host_saturation_score": _safe_float(runtime.get("host_saturation_score"), 0.0),
            "compute_pressure_level": runtime.get("compute_pressure_level"),
            "memory_pressure_level": runtime.get("memory_pressure_level"),
        },
        "memory": {
            "overall_status": memory.get("overall_status"),
            "recommended_profile": memory.get("recommended_profile"),
            "swap_tier": swap_payload.get("tier", "unknown"),
            "swap_used_gb": _safe_float(swap_payload.get("swap_used_gb"), 0.0),
        },
        "storage": {
            "severity": storage.get("severity"),
            "pressure_index": _safe_float(storage.get("pressure_index"), 0.0),
            "backpressure": storage.get("backpressure") if isinstance(storage.get("backpressure"), dict) else {},
        },
        "collection": {
            "overall_status": rollup.get("overall_status"),
            "collector_count": _safe_int(rollup.get("collector_count"), 0),
            "bots_with_observations": _safe_int(rollup.get("bots_with_observations"), 0),
            "zero_observation_count": _safe_int(rollup.get("zero_observation_count"), 0),
            "total_observations": _safe_int(rollup.get("total_observations"), 0),
        },
        "platform_intelligence": {
            "overall_status": platform.get("overall_status"),
            "expansion_count": _safe_int(platform.get("expansion_count"), 0),
            "control_count": _safe_int(platform.get("control_count"), 0),
        },
        "platform_brain_v4": {
            "overall_status": brain_v4.get("overall_status"),
            "section_count": _safe_int(brain_v4.get("section_count"), 0),
            "control_count": _safe_int(brain_v4.get("control_count"), 0),
            "next_best_command": (((brain_v4.get("sections") or {}).get("executive_meta_orchestrator") or {}).get("next_best_command") if isinstance(brain_v4.get("sections"), dict) else ""),
        },
        "platform_brain_v5": {
            "overall_status": brain_v5.get("overall_status"),
            "section_count": _safe_int(brain_v5.get("section_count"), 0),
            "control_count": _safe_int(brain_v5.get("control_count"), 0),
            "next_best_command": (((brain_v5.get("sections") or {}).get("reflex_action_router") or {}).get("next_best_command") if isinstance(brain_v5.get("sections"), dict) else ""),
        },
        "platform_stabilization_quality": {
            "overall_status": stabilizer.get("overall_status"),
            "section_count": _safe_int(stabilizer.get("section_count"), 0),
            "control_count": _safe_int(stabilizer.get("control_count"), 0),
            "next_best_command": stabilizer.get("next_best_command", ""),
            "expansion_allowed_now": (((stabilizer.get("sections") or {}).get("expansion_rehearsal_gate") or {}).get("expansion_allowed_now") if isinstance(stabilizer.get("sections"), dict) else None),
        },
        "platform_settlement_stabilization": {
            "overall_status": settlement.get("overall_status"),
            "section_count": _safe_int(settlement.get("section_count"), 0),
            "control_count": _safe_int(settlement.get("control_count"), 0),
            "next_best_command": settlement.get("next_best_command", ""),
            "queue_backpressure_active": (((settlement.get("sections") or {}).get("queue_decay_meter") or {}).get("queue_backpressure_active") if isinstance(settlement.get("sections"), dict) else None),
            "global_clear_status": (((settlement.get("sections") or {}).get("global_clear_settlement_guard") or {}).get("overall_status") if isinstance(settlement.get("sections"), dict) else None),
        },
        "schwab_futures": {
            "loop_state": schwab_futures.get("loop_state"),
            "pause_gate": schwab_futures.get("pause_gate"),
            "pause_reason": schwab_futures.get("pause_reason"),
            "total_counts": schwab_futures.get("total_counts", {}),
        },
        "recommended_commands": [
            ["./scripts/ops/opsctl.sh", "pressure-relief", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "global-halt-status", "--json"],
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast read-only health summary. Does not refresh reports.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    payload = build_payload(project_root)
    write_payload(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "health_fast "
            f"overall_status={payload.get('overall_status')} "
            f"halt={int(bool((payload.get('global_halt') or {}).get('halt')))} "
            f"collection={((payload.get('collection') or {}).get('overall_status') or 'unknown')} "
            f"pressure={((payload.get('runtime_pressure') or {}).get('tier') or 'unknown')}"
        )
    return 0 if payload.get("overall_status") == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
