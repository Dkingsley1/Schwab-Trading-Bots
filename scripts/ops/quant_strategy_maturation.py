#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "quant_strategy_maturation_v1.json"
OUT_PATH = PROJECT_ROOT / "governance" / "health" / "quant_strategy_maturation_latest.json"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "active"}


def _status(payload: dict[str, Any]) -> str:
    return str(payload.get("overall_status") or payload.get("status") or payload.get("stage") or "").strip().lower()


def _health(project_root: Path) -> dict[str, Any]:
    paths = {
        "health_fast": project_root / "governance" / "health" / "health_fast_latest.json",
        "ingestion_storage": project_root / "governance" / "health" / "ingestion_storage_control_latest.json",
        "runtime_throttle": project_root / "governance" / "health" / "runtime_throttle_latest.json",
        "writer_cycle": project_root / "governance" / "health" / "writer_cycle_coordinator_latest.json",
        "paper_400_ramp": project_root / "governance" / "health" / "paper_400_ramp_latest.json",
        "promotion_quality_gate": project_root / "governance" / "health" / "promotion_quality_gate_latest.json",
        "promotion_readiness": project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json",
        "promotion_packet": project_root / "governance" / "champion_challenger" / "promotion_packet_latest.json",
    }
    return {name: _load_json(path) for name, path in paths.items()}


def _promotion_quality_state(health: dict[str, Any]) -> dict[str, Any]:
    quality = health.get("promotion_quality_gate") if isinstance(health.get("promotion_quality_gate"), dict) else {}
    readiness = health.get("promotion_readiness") if isinstance(health.get("promotion_readiness"), dict) else {}
    packet = health.get("promotion_packet") if isinstance(health.get("promotion_packet"), dict) else {}

    quality_failed = quality.get("failed_checks") if isinstance(quality.get("failed_checks"), list) else []
    readiness_blockers = readiness.get("blocking_reasons") if isinstance(readiness.get("blocking_reasons"), list) else []
    signature = packet.get("signature") if isinstance(packet.get("signature"), dict) else {}
    blocked_statuses = {"blocked", "critical", "failed", "fail", "error"}
    ready_statuses = {"ready", "ok", "green", "passed", "pass", "clear", "complete", "verified"}
    quality_status = _status(quality)
    readiness_status = _status(readiness)
    packet_status = _status(packet)

    quality_ok = bool(quality) and _bool(quality.get("ok")) and not quality_failed and quality_status not in blocked_statuses
    readiness_ok = (
        bool(readiness)
        and not readiness_blockers
        and readiness_status not in blocked_statuses
        and (_bool(readiness.get("promote_ok")) or _bool(readiness.get("ok")) or readiness_status in ready_statuses)
    )
    packet_ok = (
        bool(packet)
        and _bool(packet.get("ok"))
        and _bool(packet.get("ready_for_committee"))
        and _bool(packet.get("packet_complete", True))
        and _bool(signature.get("verified", True))
        and packet_status not in blocked_statuses
    )
    return {
        "promotion_quality_ready": quality_ok and readiness_ok and packet_ok,
        "quality_gate_ok": quality_ok,
        "promotion_readiness_ok": readiness_ok,
        "promotion_packet_ok": packet_ok,
        "promotion_quality_failed_checks": [str(item) for item in quality_failed],
        "promotion_readiness_blockers": [str(item) for item in readiness_blockers],
        "promotion_packet_ready_for_committee": _bool(packet.get("ready_for_committee")),
        "promotion_packet_complete": _bool(packet.get("packet_complete", False)),
    }


def _pending_snapshot(health: dict[str, Any]) -> dict[str, Any]:
    health_fast = health.get("health_fast") if isinstance(health.get("health_fast"), dict) else {}
    storage = health_fast.get("storage") if isinstance(health_fast.get("storage"), dict) else {}
    backpressure = storage.get("backpressure") if isinstance(storage.get("backpressure"), dict) else {}
    ingestion = health.get("ingestion_storage") if isinstance(health.get("ingestion_storage"), dict) else {}
    if not backpressure and isinstance(ingestion.get("backpressure"), dict):
        backpressure = ingestion["backpressure"]
    threshold = float(backpressure.get("pending_lines_threshold") or 15000)
    age_threshold = float(backpressure.get("oldest_age_threshold_seconds") or 240)
    total = float(backpressure.get("total_pending_lines") or 0)
    core = float(backpressure.get("core_pending_lines") or 0)
    oldest = float(backpressure.get("oldest_pending_age_seconds") or 0)
    return {
        "core_pending_lines": int(core),
        "total_pending_lines": int(total),
        "oldest_pending_age_seconds": oldest,
        "pending_lines_threshold": int(threshold),
        "oldest_age_threshold_seconds": age_threshold,
        "pending_green": core <= threshold and total <= threshold,
        "age_green": oldest <= age_threshold,
    }


def _writer_active(writer: dict[str, Any]) -> bool:
    candidates = [
        writer.get("writer_state_before"),
        writer.get("writer_state_after_wait"),
        writer.get("wait_for_writer", {}).get("final_state") if isinstance(writer.get("wait_for_writer"), dict) else {},
    ]
    for candidate in candidates:
        if isinstance(candidate, dict) and "active" in candidate:
            return _bool(candidate.get("active"))
    return _bool(writer.get("writer_active"))


def _gate_checks(health: dict[str, Any]) -> list[dict[str, Any]]:
    health_fast = health.get("health_fast") if isinstance(health.get("health_fast"), dict) else {}
    global_halt = health_fast.get("global_halt") if isinstance(health_fast.get("global_halt"), dict) else {}
    runtime = health_fast.get("runtime_pressure") if isinstance(health_fast.get("runtime_pressure"), dict) else {}
    storage = health_fast.get("storage") if isinstance(health_fast.get("storage"), dict) else {}
    runtime_throttle = health.get("runtime_throttle") if isinstance(health.get("runtime_throttle"), dict) else {}
    writer = health.get("writer_cycle") if isinstance(health.get("writer_cycle"), dict) else {}
    paper_400 = health.get("paper_400_ramp") if isinstance(health.get("paper_400_ramp"), dict) else {}
    pending = _pending_snapshot(health)

    clear_blockers = global_halt.get("clear_blockers") if isinstance(global_halt.get("clear_blockers"), list) else []
    global_clear = bool(global_halt) and not _bool(global_halt.get("halt")) and not clear_blockers
    queue_clear = pending["pending_green"] and pending["age_green"]
    storage_ready = (
        queue_clear
        and str(storage.get("severity") or "").strip().lower() not in {"critical", "blocked"}
        and _status(health.get("ingestion_storage", {})) not in {"blocked", "critical"}
    )
    runtime_status = _status(runtime)
    runtime_throttle_status = _status(runtime_throttle)
    compute_pressure = str(runtime.get("compute_pressure_level") or "").strip().lower()
    runtime_ready = (
        runtime_status in {"ready", "ok", "green"}
        and runtime_throttle_status in {"ready", "ok", "green"}
        and compute_pressure in {"", "normal", "low"}
    )
    writer_ready = bool(writer) and not _writer_active(writer)
    paper_400_ready = (
        bool(paper_400)
        and _status(paper_400) in {"ready", "ok", "green"}
        and not paper_400.get("blockers")
        and _bool(paper_400.get("ok", True))
    )
    promotion_quality = _promotion_quality_state(health)

    return [
        {
            "gate": "global_halt_clear",
            "ok": global_clear,
            "detail": "clear" if global_clear else "halt_active_or_clear_blockers_present",
        },
        {
            "gate": "queue_backpressure_clear",
            "ok": queue_clear,
            "detail": pending,
        },
        {
            "gate": "storage_pressure_below_paper_gate",
            "ok": storage_ready,
            "detail": {
                "storage_severity": str(storage.get("severity") or ""),
                "ingestion_storage_status": _status(health.get("ingestion_storage", {})),
            },
        },
        {
            "gate": "runtime_capacity_ready",
            "ok": runtime_ready,
            "detail": {
                "runtime_status": runtime_status,
                "runtime_throttle_status": runtime_throttle_status,
                "compute_pressure_level": compute_pressure,
            },
        },
        {
            "gate": "writer_idle_or_coordinated",
            "ok": writer_ready,
            "detail": {
                "writer_status": _status(writer),
                "writer_active": _writer_active(writer),
            },
        },
        {
            "gate": "paper_400_ramp_ready",
            "ok": paper_400_ready,
            "detail": {
                "paper_400_status": _status(paper_400),
                "blockers": paper_400.get("blockers", []),
            },
        },
        {
            "gate": "promotion_quality_gates_ready",
            "ok": promotion_quality["promotion_quality_ready"],
            "detail": promotion_quality,
        },
    ]


def _upgrade_summary(project_root: Path) -> dict[str, Any]:
    payload = _load_json(project_root / "config" / "quant_strategy_lane_upgrades_v1.json")
    pack = payload.get("pack") if isinstance(payload.get("pack"), dict) else {}
    lanes = pack.get("lanes") if isinstance(pack.get("lanes"), list) else []
    module_count = 0
    safe_now_count = 0
    for lane in lanes:
        if not isinstance(lane, dict):
            continue
        modules = lane.get("upgrade_modules") if isinstance(lane.get("upgrade_modules"), list) else []
        module_count += len(modules)
        safe_now_count += sum(1 for module in modules if isinstance(module, dict) and bool(module.get("safe_now")))
    return {
        "version": payload.get("quant_strategy_lane_upgrades_version", ""),
        "slug": pack.get("slug", ""),
        "lane_count": len(lanes),
        "total_upgrade_modules": module_count,
        "safe_now_upgrade_modules": safe_now_count,
        "paper_trading_enabled": bool(pack.get("paper_trading_enabled")),
        "live_trading_enabled": bool(pack.get("live_trading_enabled")),
        "execution_enabled": bool(pack.get("execution_enabled")),
        "allocation_enabled": bool(pack.get("allocation_enabled")),
    }


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    config = _load_json(project_root / "config" / "quant_strategy_maturation_v1.json")
    pack = config.get("pack") if isinstance(config.get("pack"), dict) else {}
    lanes = pack.get("lanes") if isinstance(pack.get("lanes"), list) else []
    health = _health(project_root)
    gates = _gate_checks(health)
    failed = [gate for gate in gates if not bool(gate.get("ok"))]
    blocked = bool(failed)
    lane_summaries = [
        {
            "priority": lane.get("priority"),
            "slug": lane.get("slug"),
            "current_state": lane.get("current_state"),
            "candidate_count": len(lane.get("candidate_bot_ids", [])) if isinstance(lane.get("candidate_bot_ids"), list) else 0,
            "paper_trading_enabled": bool(lane.get("paper_trading_enabled")),
            "live_trading_enabled": bool(lane.get("live_trading_enabled")),
            "execution_enabled": bool(lane.get("execution_enabled")),
            "allocation_enabled": bool(lane.get("allocation_enabled")),
        }
        for lane in lanes
        if isinstance(lane, dict)
    ]
    activation_policy = pack.get("activation_policy") if isinstance(pack.get("activation_policy"), dict) else {}
    collection_runtime_active = (
        len(lane_summaries) == 7
        and all(str(lane.get("current_state") or "") == "collection_only" for lane in lane_summaries)
        and not bool(pack.get("paper_trading_enabled"))
        and not bool(pack.get("live_trading_enabled"))
        and not bool(pack.get("execution_enabled"))
        and not bool(pack.get("allocation_enabled"))
        and not bool(activation_policy.get("registry_promotion_side_effects_allowed"))
    )
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": True,
        "overall_status": (
            "paper_canary_queue_ready"
            if not blocked
            else "collection_runtime_active_paper_canary_blocked"
            if collection_runtime_active
            else "collection_only_blocked"
        ),
        "quant_strategy_maturation_version": config.get("quant_strategy_maturation_version", "quant_strategy_maturation_v1"),
        "pack_slug": pack.get("slug", "quant_strategy_maturation"),
        "lane_count": len(lane_summaries),
        "activation_policy": activation_policy,
        "upgrade_pack": _upgrade_summary(project_root),
        "collection_runtime_active": collection_runtime_active,
        "paper_trading_enabled": bool(pack.get("paper_trading_enabled")),
        "live_trading_enabled": bool(pack.get("live_trading_enabled")),
        "execution_enabled": bool(pack.get("execution_enabled")),
        "allocation_enabled": bool(pack.get("allocation_enabled")),
        "gate_checks": gates,
        "failed_gates": [str(gate.get("gate") or "") for gate in failed],
        "lanes": lane_summaries,
        "recommended_actions": (
            [
                "keep all seven lanes collection-only",
                "continue single-writer backlog relief before paper-canary review",
                "re-run after queue, runtime, writer, paper-400, and promotion-quality gates clear",
            ]
            if blocked
            else [
                "review one lane and one bot for a min-size paper canary",
                "keep live execution disabled",
                "require promotion packet approval before changing registry flags",
            ]
        ),
    }
    _write_json(project_root / "governance" / "health" / "quant_strategy_maturation_latest.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the seven-lane quant strategy maturation queue.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = build_payload(PROJECT_ROOT)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "quant_strategy_maturation "
            f"status={payload['overall_status']} "
            f"lanes={payload['lane_count']} "
            f"failed_gates={len(payload['failed_gates'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
