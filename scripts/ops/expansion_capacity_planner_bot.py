#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, status_rank, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, status_rank, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "expansion_capacity_planner_latest.json"

SUPPORT_INFRABOT_IDS = [
    "brain_refinery_v674_expansion_capacity_planner_bot",
    "brain_refinery_v675_expansion_dependency_graph_guard_bot",
    "brain_refinery_v676_expansion_storage_budget_forecaster_bot",
    "brain_refinery_v677_expansion_training_maturity_gate_bot",
    "brain_refinery_v678_expansion_runtime_isolation_guard_bot",
]


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        value = float(raw)
    except Exception:
        return float(default)
    if value != value or value in {float("inf"), float("-inf")}:
        return float(default)
    return value


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on", "active", "enabled", "blocked"}


def _status(raw: Any, default: str = "missing") -> str:
    text = str(raw or "").strip().lower()
    return text or default


def _registry_rows(project_root: Path) -> list[dict[str, Any]]:
    payload = load_json(project_root / "master_bot_registry.json")
    rows = payload.get("sub_bots") if isinstance(payload.get("sub_bots"), list) else []
    return [row for row in rows if isinstance(row, dict)]


def _bot_version(bot_id: str) -> int:
    match = re.search(r"_v(\d+)", str(bot_id or ""))
    return int(match.group(1)) if match else 0


def _infer_sleeve(row: dict[str, Any]) -> str:
    for key in ("sleeve_profile", "sleeve", "profile", "strategy_sleeve", "slot_kind"):
        value = str(row.get(key) or "").strip().lower()
        if value:
            return value
    text = " ".join(
        str(part or "").lower()
        for part in [
            row.get("bot_id"),
            row.get("bot_role"),
            " ".join(str(item) for item in list(row.get("target_functions") or [])),
            " ".join(str(item) for item in list(row.get("data_intake_collections") or [])),
        ]
    )
    for token in (
        "options_on_futures",
        "intraday_aggressive",
        "day_trading",
        "dividend",
        "conservative",
        "futures",
        "options",
        "crypto",
        "fx",
        "bond",
        "macro",
        "liquidity",
        "execution",
        "feature_quality",
        "system_governor",
    ):
        if token in text:
            return token
    return "default"


def _pressure_snapshot(project_root: Path) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    platform_root = project_root / "governance" / "platform_intelligence"
    swap = load_json(health_root / "swap_pressure_governor_latest.json")
    runtime = load_json(health_root / "runtime_throttle_control_latest.json")
    memory = load_json(health_root / "memory_efficiency_control_latest.json")
    storage = load_json(health_root / "ingestion_storage_control_latest.json")
    backpressure = load_json(health_root / "ingestion_backpressure_latest.json")
    data_storage = load_json(health_root / "data_collection_storage_guard_latest.json")
    halt = load_json(health_root / "global_killswitch_latest.json")
    halt_auto_clear = load_json(health_root / "global_halt_auto_clear_latest.json")
    admission = load_json(health_root / "new_bot_admission_guard_latest.json")
    dashboard = load_json(health_root / "runtime_gate_dashboard_latest.json")
    platform_capacity = load_json(platform_root / "capacity_planner_latest.json")
    platform_stabilization = load_json(health_root / "platform_stabilization_quality_latest.json")
    settlement = load_json(health_root / "platform_settlement_stabilization_latest.json")

    swap_pressure = swap.get("swap_pressure") if isinstance(swap.get("swap_pressure"), dict) else {}
    swap_tier = _status(swap_pressure.get("tier") or swap.get("tier"), "normal")
    swap_gb = _safe_float(swap_pressure.get("swap_used_gb") or swap.get("swap_used_gb"), 0.0)
    runtime_status = _status(runtime.get("overall_status"))
    memory_status = _status(memory.get("overall_status"))
    storage_status = _status(storage.get("overall_status"))
    data_storage_status = _status(data_storage.get("overall_status"), storage_status)
    dashboard_status = _status(dashboard.get("overall_status"), "missing")
    platform_capacity_status = _status(platform_capacity.get("overall_status"), "missing")
    storage_pressure = _safe_float(storage.get("pressure_index"), 0.0)
    host_saturation = _safe_float(runtime.get("host_saturation_score"), 0.0)
    compute_level = _status(runtime.get("compute_pressure_level"), "unknown")
    memory_pressure_level = _status(runtime.get("memory_pressure_level"), "unknown")
    storage_backpressure = storage.get("backpressure") if isinstance(storage.get("backpressure"), dict) else {}
    total_pending_lines = max(
        _safe_int(storage_backpressure.get("total_pending_lines"), 0),
        _safe_int(backpressure.get("pending_lines_total"), 0),
        _safe_int(backpressure.get("pending_lines"), 0),
    )
    pending_threshold = max(_safe_int(storage_backpressure.get("pending_lines_threshold"), 15000), 1)
    pending_ratio = total_pending_lines / float(pending_threshold)
    queue_backpressure_active = total_pending_lines >= pending_threshold or _status(storage.get("severity"), "unknown") in {"high", "critical", "blocked"}
    admission_blocking = _safe_int(admission.get("blocking_candidate_count"), 0)
    admission_candidates = _safe_int(admission.get("candidate_bot_count"), 0)
    global_halt_active = any(
        _bool(halt.get(key))
        for key in ("global_halt_active", "halt_active", "hard_halt_active", "blocked", "killswitch_active")
    ) or _bool(halt_auto_clear.get("halt"))
    clear_blockers = halt_auto_clear.get("clear_blockers") if isinstance(halt_auto_clear.get("clear_blockers"), list) else []
    stabilization_sections = platform_stabilization.get("sections") if isinstance(platform_stabilization.get("sections"), dict) else {}
    stabilization_gate = stabilization_sections.get("expansion_rehearsal_gate") if isinstance(stabilization_sections.get("expansion_rehearsal_gate"), dict) else {}
    stabilization_says_no = bool(stabilization_gate) and not _bool(stabilization_gate.get("expansion_allowed_now"))
    settlement_sections = settlement.get("sections") if isinstance(settlement.get("sections"), dict) else {}
    settlement_queue = settlement_sections.get("queue_decay_meter") if isinstance(settlement_sections.get("queue_decay_meter"), dict) else {}
    settlement_queue_active = _bool(settlement_queue.get("queue_backpressure_active"))

    blocking_reasons: list[str] = []
    if global_halt_active:
        blocking_reasons.append("global_halt_active")
    if swap_tier in {"survival", "critical"} or swap_gb >= 20.0:
        blocking_reasons.append("swap_pressure_too_high_for_new_runtime")
    if runtime_status in {"blocked", "critical"} or host_saturation >= 85.0 or compute_level in {"high", "critical"}:
        blocking_reasons.append("runtime_pressure_too_high")
    if storage_status in {"blocked", "critical"} or data_storage_status in {"blocked", "critical"}:
        blocking_reasons.append("storage_pressure_too_high")
    if queue_backpressure_active:
        blocking_reasons.append("queue_backpressure_active")
    if clear_blockers:
        blocking_reasons.append("global_clear_blockers_present")
    if stabilization_says_no:
        blocking_reasons.append("pre_expansion_stabilization_gate_closed")
    if settlement_queue_active:
        blocking_reasons.append("settlement_queue_backpressure_active")

    watch_reasons: list[str] = []
    if memory_status in {"needs_work", "degraded", "blocked", "critical"}:
        watch_reasons.append("memory_governor_needs_attention")
    if swap_tier in {"calm", "constrained", "pause_research"} or swap_gb >= 12.0:
        watch_reasons.append("swap_pressure_elevated")
    if storage_pressure >= 0.40:
        watch_reasons.append("storage_pressure_index_elevated")
    if pending_ratio >= 0.50:
        watch_reasons.append("queue_pending_ratio_elevated")
    if host_saturation >= 65.0 or compute_level == "elevated" or memory_pressure_level == "elevated":
        watch_reasons.append("runtime_not_calm_enough_for_expansion")
    if admission_blocking > 0:
        watch_reasons.append("new_bot_admission_contracts_not_clear")
    if dashboard_status in {"warn", "needs_work", "degraded", "blocked", "critical"}:
        watch_reasons.append("runtime_gate_dashboard_has_attention_items")

    if blocking_reasons:
        overall_status = "blocked"
    elif watch_reasons:
        overall_status = "needs_work"
    else:
        overall_status = "ready"

    return {
        "overall_status": overall_status,
        "blocking_reasons": blocking_reasons,
        "watch_reasons": watch_reasons,
        "swap_tier": swap_tier,
        "swap_used_gb": round(swap_gb, 3),
        "runtime_status": runtime_status,
        "memory_status": memory_status,
        "storage_status": storage_status,
        "data_collection_storage_status": data_storage_status,
        "runtime_gate_dashboard_status": dashboard_status,
        "platform_capacity_status": platform_capacity_status,
        "storage_pressure_index": round(storage_pressure, 6),
        "host_saturation_score": round(host_saturation, 3),
        "compute_pressure_level": compute_level,
        "memory_pressure_level": memory_pressure_level,
        "queue_backpressure_active": queue_backpressure_active,
        "total_pending_lines": total_pending_lines,
        "pending_lines_threshold": pending_threshold,
        "pending_ratio": round(pending_ratio, 6),
        "global_halt_active": global_halt_active,
        "global_clear_blockers": [str(item) for item in clear_blockers],
        "pre_expansion_stabilization_gate_closed": stabilization_says_no,
        "settlement_queue_backpressure_active": settlement_queue_active,
        "admission_candidate_count": admission_candidates,
        "admission_blocking_candidate_count": admission_blocking,
        "source_files": {
            "swap_pressure_governor": str(health_root / "swap_pressure_governor_latest.json"),
            "runtime_throttle_control": str(health_root / "runtime_throttle_control_latest.json"),
            "memory_efficiency_control": str(health_root / "memory_efficiency_control_latest.json"),
            "ingestion_storage_control": str(health_root / "ingestion_storage_control_latest.json"),
            "data_collection_storage_guard": str(health_root / "data_collection_storage_guard_latest.json"),
            "global_killswitch": str(health_root / "global_killswitch_latest.json"),
            "new_bot_admission_guard": str(health_root / "new_bot_admission_guard_latest.json"),
            "runtime_gate_dashboard": str(health_root / "runtime_gate_dashboard_latest.json"),
            "platform_capacity_planner": str(platform_root / "capacity_planner_latest.json"),
            "platform_stabilization_quality": str(health_root / "platform_stabilization_quality_latest.json"),
            "platform_settlement_stabilization": str(health_root / "platform_settlement_stabilization_latest.json"),
        },
    }


def _capacity_contract(rows: list[dict[str, Any]], pressure: dict[str, Any], *, requested_wave_size: int) -> dict[str, Any]:
    total_bots = len(rows)
    active_bots = sum(1 for row in rows if bool(row.get("active", False)))
    collection_only = sum(1 for row in rows if str(row.get("lifecycle_state") or "").strip().lower() == "data_collection_only")
    training_excluded = sum(1 for row in rows if bool(row.get("training_excluded", False)) or bool(row.get("exclude_from_training", False)))
    max_version = max([_bot_version(str(row.get("bot_id") or "")) for row in rows] or [0])
    sleeve_counts = Counter(_infer_sleeve(row) for row in rows)

    max_new_collectors = 30
    if total_bots >= 900:
        max_new_collectors = 10
    elif total_bots >= 700:
        max_new_collectors = 18
    elif total_bots >= 500:
        max_new_collectors = 24

    if collection_only >= 550:
        max_new_collectors = min(max_new_collectors, 16)
    elif collection_only >= 450:
        max_new_collectors = min(max_new_collectors, 22)

    if pressure["overall_status"] == "blocked":
        max_new_collectors = 0
    elif pressure["overall_status"] == "needs_work":
        max_new_collectors = min(max_new_collectors, 20)

    if pressure["swap_used_gb"] >= 20.0:
        max_new_collectors = min(max_new_collectors, 12)
    elif pressure["swap_used_gb"] >= 16.0:
        max_new_collectors = min(max_new_collectors, 20)

    if pressure["global_halt_active"]:
        max_new_collectors = 0

    requested = max(int(requested_wave_size), 0)
    recommended = min(requested, max_new_collectors)
    if max_new_collectors <= 0:
        rollout_mode = "protect_live_no_new_runtime_loops"
    elif recommended < requested:
        rollout_mode = "trickle_collection_only_wave"
    else:
        rollout_mode = "collection_only_wave_allowed"

    training_policy = "training_locked_until_admission_and_minimum_observation_contracts_clear"
    if pressure["admission_blocking_candidate_count"] <= 0 and pressure["overall_status"] == "ready":
        training_policy = "training_still_locked_for_new_bots_until_minimum_collection_thresholds"

    sleeve_growth_budget = []
    for sleeve, count in sleeve_counts.most_common():
        if sleeve in {"system_governor_expansion", "feature_quality_data_confidence", "liquidity_regime", "transaction_cost_slippage_intelligence", "provider_adapter_verification"}:
            mode = "support_growth"
        elif count >= 45:
            mode = "thin_sample_only_until_backlog_drops"
        else:
            mode = "normal_collection"
        sleeve_growth_budget.append({"sleeve": sleeve, "registered_bots": count, "recommended_collection_mode": mode})

    status = "ready"
    if max_new_collectors <= 0:
        status = "blocked"
    elif recommended < requested:
        status = "needs_work"

    return {
        "overall_status": status,
        "requested_wave_size": requested,
        "max_new_collectors_now": max_new_collectors,
        "recommended_wave_size_now": recommended,
        "rollout_mode": rollout_mode,
        "training_policy": training_policy,
        "paper_trade_policy": "paper_locked_zero_weight_until_graduation",
        "live_trade_policy": "live_execution_disabled_for_new_expansion_slots",
        "next_bot_id_range": {
            "start": f"brain_refinery_v{max_version + 1}",
            "end_if_full_requested_wave_added": f"brain_refinery_v{max_version + requested}",
            "end_if_recommended_wave_added": f"brain_refinery_v{max_version + recommended}" if recommended else "",
        },
        "fleet_counts": {
            "total_bots": total_bots,
            "active_bots": active_bots,
            "data_collection_only_bots": collection_only,
            "training_excluded_bots": training_excluded,
            "max_bot_version": max_version,
            "sleeve_count": len(sleeve_counts),
        },
        "sleeve_growth_budget": sleeve_growth_budget[:40],
    }


def build_payload(project_root: Path = PROJECT_ROOT, *, requested_wave_size: int = 20) -> dict[str, Any]:
    rows = _registry_rows(project_root)
    pressure = _pressure_snapshot(project_root)
    capacity = _capacity_contract(rows, pressure, requested_wave_size=requested_wave_size)
    worst_rank = max(status_rank(str(pressure.get("overall_status"))), status_rank(str(capacity.get("overall_status"))))
    overall_status = "ready"
    if worst_rank >= status_rank("blocked"):
        overall_status = "blocked"
    elif worst_rank >= status_rank("needs_work"):
        overall_status = "needs_work"

    recommended_actions = ordered_unique(
        [
            "sync roster-expansion with --apply-registry, then materialize core bot files" if capacity["recommended_wave_size_now"] > 0 else "",
            "keep every new bot data_collection_only, zero weight, paper locked, and live disabled",
            "run expansion-capacity before each future wave so growth reacts to storage, swap, runtime, halts, and admission pressure",
            "pause any future wave while global halt or high swap pressure is active" if capacity["max_new_collectors_now"] <= 0 else "",
            "clear new-bot admission contracts before allowing any of the expanded roster into training" if pressure["admission_blocking_candidate_count"] > 0 else "",
            "refresh core bot catalog after materialization so PyCharm shows the new files in one place",
        ]
    )

    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall_status != "blocked",
        "overall_status": overall_status,
        "mode": "advisory_expansion_capacity_guard",
        "pressure_snapshot": pressure,
        "capacity_contract": capacity,
        "support_infrabots": SUPPORT_INFRABOT_IDS,
        "growth_invariants": [
            "new bots enter as data_collection_only",
            "new bots are excluded from training until minimum observations and days are met",
            "new bots keep zero allocation weight and live execution disabled",
            "future waves must pass storage, queue, swap, runtime, global halt, settlement, and admission gates",
            "pre-expansion stabilization must explicitly allow the wave before any roster growth",
            "registry sync and core materialization must run after each accepted wave",
        ],
        "recommended_actions": recommended_actions,
        "source_files": {
            "master_bot_registry": str(project_root / "master_bot_registry.json"),
            "artifact": str(DEFAULT_OUT_PATH),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan safe expansion capacity for new bot waves.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--wave-size", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    payload = build_payload(project_root, requested_wave_size=max(int(args.wave_size), 0))
    out_path = Path(args.out_file).expanduser()
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        contract = payload["capacity_contract"]
        print(
            "expansion_capacity_planner "
            f"status={payload.get('overall_status', '')} "
            f"recommended_wave={contract.get('recommended_wave_size_now', 0)} "
            f"max_new_collectors={contract.get('max_new_collectors_now', 0)} "
            f"mode={contract.get('rollout_mode', '')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
