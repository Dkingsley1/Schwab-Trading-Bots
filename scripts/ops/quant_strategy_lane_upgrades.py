#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "quant_strategy_lane_upgrades_v1.json"
OUT_PATH = PROJECT_ROOT / "governance" / "health" / "quant_strategy_lane_upgrades_latest.json"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _status(payload: dict[str, Any]) -> str:
    return str(payload.get("overall_status") or payload.get("status") or payload.get("stage") or "").strip().lower()


def _health(project_root: Path) -> dict[str, Any]:
    return {
        "health_fast": _load_json(project_root / "governance" / "health" / "health_fast_latest.json"),
        "paper_400_ramp": _load_json(project_root / "governance" / "health" / "paper_400_ramp_latest.json"),
        "maturation": _load_json(project_root / "governance" / "health" / "quant_strategy_maturation_latest.json"),
        "promotion_quality_gate": _load_json(project_root / "governance" / "health" / "promotion_quality_gate_latest.json"),
        "promotion_readiness": _load_json(project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json"),
        "promotion_packet": _load_json(project_root / "governance" / "champion_challenger" / "promotion_packet_latest.json"),
    }


def _promotion_quality_state(health: dict[str, Any]) -> dict[str, Any]:
    quality = health.get("promotion_quality_gate") if isinstance(health.get("promotion_quality_gate"), dict) else {}
    readiness = health.get("promotion_readiness") if isinstance(health.get("promotion_readiness"), dict) else {}
    packet = health.get("promotion_packet") if isinstance(health.get("promotion_packet"), dict) else {}

    quality_failed = quality.get("failed_checks") if isinstance(quality.get("failed_checks"), list) else []
    readiness_blockers = readiness.get("blocking_reasons") if isinstance(readiness.get("blocking_reasons"), list) else []
    signature = packet.get("signature") if isinstance(packet.get("signature"), dict) else {}
    quality_status = _status(quality)
    readiness_status = _status(readiness)
    packet_status = _status(packet)
    blocked_statuses = {"blocked", "critical", "failed", "fail", "error"}
    ready_statuses = {"ready", "ok", "green", "passed", "pass", "clear", "complete", "verified"}

    quality_ok = bool(quality) and bool(quality.get("ok")) and not quality_failed and quality_status not in blocked_statuses
    readiness_ok = (
        bool(readiness)
        and not readiness_blockers
        and readiness_status not in blocked_statuses
        and (bool(readiness.get("promote_ok")) or bool(readiness.get("ok")) or readiness_status in ready_statuses)
    )
    packet_ok = (
        bool(packet)
        and bool(packet.get("ok"))
        and bool(packet.get("ready_for_committee"))
        and bool(packet.get("packet_complete", True))
        and bool(signature.get("verified", True))
        and packet_status not in blocked_statuses
    )
    return {
        "promotion_quality_ready": quality_ok and readiness_ok and packet_ok,
        "quality_gate_ok": quality_ok,
        "promotion_readiness_ok": readiness_ok,
        "promotion_packet_ok": packet_ok,
        "promotion_quality_failed_checks": [str(item) for item in quality_failed],
        "promotion_readiness_blockers": [str(item) for item in readiness_blockers],
        "promotion_packet_ready_for_committee": bool(packet.get("ready_for_committee")),
        "promotion_packet_complete": bool(packet.get("packet_complete", False)),
    }


def _gate_state(health: dict[str, Any]) -> dict[str, Any]:
    health_fast = health.get("health_fast") if isinstance(health.get("health_fast"), dict) else {}
    paper_400 = health.get("paper_400_ramp") if isinstance(health.get("paper_400_ramp"), dict) else {}
    runtime = health_fast.get("runtime_pressure") if isinstance(health_fast.get("runtime_pressure"), dict) else {}
    storage = health_fast.get("storage") if isinstance(health_fast.get("storage"), dict) else {}
    backpressure = storage.get("backpressure") if isinstance(storage.get("backpressure"), dict) else {}
    global_halt = health_fast.get("global_halt") if isinstance(health_fast.get("global_halt"), dict) else {}
    clear_blockers = global_halt.get("clear_blockers") if isinstance(global_halt.get("clear_blockers"), list) else []
    threshold = float(backpressure.get("pending_lines_threshold") or 15000)
    total = float(backpressure.get("total_pending_lines") or 0)
    core = float(backpressure.get("core_pending_lines") or 0)
    oldest = float(backpressure.get("oldest_pending_age_seconds") or 0)
    age_threshold = float(backpressure.get("oldest_age_threshold_seconds") or 240)
    runtime_status = _status(runtime)
    compute = str(runtime.get("compute_pressure_level") or "").strip().lower()
    paper_blockers = paper_400.get("blockers") if isinstance(paper_400.get("blockers"), list) else []
    storage_green = (
        core <= threshold
        and total <= threshold
        and oldest <= age_threshold
        and str(storage.get("severity") or "").strip().lower() in {"stable", "ready", "green"}
    )
    runtime_green = runtime_status in {"ready", "ok", "green"} and compute in {"", "normal", "low"}
    paper_400_ready = bool(paper_400) and bool(paper_400.get("ok")) and _status(paper_400) not in {"blocked"}
    promotion_state = _promotion_quality_state(health)
    gate_state = {
        "global_halt_clear": bool(global_halt) and not bool(global_halt.get("halt")) and not clear_blockers,
        "storage_green": storage_green,
        "runtime_green": runtime_green,
        "paper_400_ready": paper_400_ready,
        "runtime_status": runtime_status,
        "compute_pressure_level": compute,
        "storage_severity": str(storage.get("severity") or ""),
        "core_pending_lines": int(core),
        "total_pending_lines": int(total),
        "oldest_pending_age_seconds": oldest,
        "paper_400_blockers": paper_blockers,
    }
    gate_state.update(promotion_state)
    return gate_state


def _lane_summary(lane: dict[str, Any]) -> dict[str, Any]:
    modules = lane.get("upgrade_modules") if isinstance(lane.get("upgrade_modules"), list) else []
    safe_now = [module for module in modules if isinstance(module, dict) and bool(module.get("safe_now"))]
    outputs = [
        str(output)
        for module in modules
        if isinstance(module, dict)
        for output in (module.get("outputs") if isinstance(module.get("outputs"), list) else [])
        if str(output)
    ]
    return {
        "priority": lane.get("priority"),
        "slug": lane.get("slug"),
        "current_state": lane.get("current_state"),
        "upgrade_module_count": len(modules),
        "safe_now_module_count": len(safe_now),
        "paper_trading_enabled": bool(lane.get("paper_trading_enabled")),
        "live_trading_enabled": bool(lane.get("live_trading_enabled")),
        "execution_enabled": bool(lane.get("execution_enabled")),
        "allocation_enabled": bool(lane.get("allocation_enabled")),
        "output_contract_count": len(set(outputs)),
    }


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    config = _load_json(project_root / "config" / "quant_strategy_lane_upgrades_v1.json")
    pack = config.get("pack") if isinstance(config.get("pack"), dict) else {}
    lanes = pack.get("lanes") if isinstance(pack.get("lanes"), list) else []
    lane_summaries = [_lane_summary(lane) for lane in lanes if isinstance(lane, dict)]
    health = _health(project_root)
    gate_state = _gate_state(health)
    total_modules = sum(int(lane.get("upgrade_module_count") or 0) for lane in lane_summaries)
    safe_now_modules = sum(int(lane.get("safe_now_module_count") or 0) for lane in lane_summaries)
    forbidden_enabled = [
        field
        for field in [
            "paper_trading_enabled",
            "live_trading_enabled",
            "execution_enabled",
            "allocation_enabled",
            "heavy_training_enabled",
            "new_high_volume_collectors_enabled",
            "registry_promotion_side_effects_allowed",
        ]
        if bool(pack.get(field))
    ]
    ok = not forbidden_enabled and len(lane_summaries) == 7 and total_modules >= 56
    collection_runtime_active = ok and safe_now_modules == total_modules and total_modules > 0
    paper_activation_ready = (
        ok
        and gate_state["global_halt_clear"]
        and gate_state["storage_green"]
        and gate_state["runtime_green"]
        and gate_state["paper_400_ready"]
        and gate_state["promotion_quality_ready"]
    )
    runtime_activation_ready = paper_activation_ready
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": ok,
        "overall_status": (
            "invalid_upgrade_contract"
            if not ok
            else "paper_activation_ready"
            if runtime_activation_ready
            else "collection_runtime_active_paper_activation_blocked"
            if collection_runtime_active
            else "manifest_upgrades_installed_runtime_activation_blocked"
        ),
        "quant_strategy_lane_upgrades_version": config.get(
            "quant_strategy_lane_upgrades_version", "quant_strategy_lane_upgrades_v1"
        ),
        "pack_slug": pack.get("slug", "quant_strategy_lane_upgrades"),
        "lane_count": len(lane_summaries),
        "total_upgrade_modules": total_modules,
        "safe_now_upgrade_modules": safe_now_modules,
        "collection_runtime_active": collection_runtime_active,
        "paper_activation_ready": paper_activation_ready,
        "runtime_activation_ready": runtime_activation_ready,
        "forbidden_enabled": forbidden_enabled,
        "gate_state": gate_state,
        "lanes": lane_summaries,
        "blocked_runtime_scope_until_gates_clear": (
            pack.get("blocked_runtime_scope_until_gates_clear")
            if isinstance(pack.get("blocked_runtime_scope_until_gates_clear"), list)
            else []
        ),
        "recommended_actions": (
            [
                "keep all 56 lane upgrades active as collection-only runtime manifests",
                "use the new feature, label, execution, source, portfolio, and promotion contracts for candidate review",
                "do not enable paper, live, allocation, heavy replay, heavy training, or high-volume collectors until runtime, paper-400, and promotion gates clear",
            ]
            if ok and not runtime_activation_ready
            else [
                "review one lane for a tiny paper canary packet",
                "keep live execution disabled",
                "require a signed promotion packet before registry changes",
            ]
        ),
    }
    _write_json(project_root / "governance" / "health" / "quant_strategy_lane_upgrades_latest.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate seven-lane quant strategy upgrade contracts.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = build_payload(PROJECT_ROOT)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "quant_strategy_lane_upgrades "
            f"status={payload['overall_status']} "
            f"lanes={payload['lane_count']} "
            f"modules={payload['total_upgrade_modules']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
