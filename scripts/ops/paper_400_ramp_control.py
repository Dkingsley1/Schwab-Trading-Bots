#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "paper_400_ramp_latest.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.paper_400_ramp_override"
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "master_bot_registry.json"
EARLIEST_ACTIVATION_DATE = date(2026, 5, 11)
TARGET_PAPER_BOTS = 400

PAPER_ALLOCATION: dict[str, dict[str, Any]] = {
    "schwab_equities": {
        "target": 200,
        "top_n_env": "SCHWAB_TOP_BOT_PAPER_TRADING_TOP_N",
        "min_acc_env": "SCHWAB_TOP_BOT_PAPER_TRADING_MIN_ACC",
        "profiles_env": "SCHWAB_TOP_BOT_PAPER_TRADING_PROFILES",
        "min_acc": "0.56",
        "profiles": "default,conservative,aggressive,intraday_aggressive,swing_aggressive,dividend,bond,fx",
    },
    "schwab_options": {
        "target": 40,
        "top_n_env": "SCHWAB_OPTIONS_TOP_BOT_PAPER_TRADING_TOP_N",
        "min_acc_env": "SCHWAB_OPTIONS_TOP_BOT_PAPER_TRADING_MIN_ACC",
        "profiles_env": "SCHWAB_OPTIONS_TOP_BOT_PAPER_TRADING_PROFILES",
        "min_acc": "0.56",
        "profiles": "default,aggressive,intraday_aggressive,swing_aggressive,options_on_futures,options_on_futures_aggressive",
    },
    "schwab_futures": {
        "target": 80,
        "top_n_env": "SCHWAB_FUTURES_TOP_BOT_PAPER_TRADING_TOP_N",
        "min_acc_env": "SCHWAB_FUTURES_TOP_BOT_PAPER_TRADING_MIN_ACC",
        "profiles_env": "SCHWAB_FUTURES_TOP_BOT_PAPER_TRADING_PROFILES",
        "min_acc": "0.54",
        "profiles": "schwab_futures",
    },
    "coinbase_spot": {
        "target": 50,
        "top_n_env": "COINBASE_TOP_BOT_PAPER_TRADING_TOP_N",
        "min_acc_env": "COINBASE_TOP_BOT_PAPER_TRADING_MIN_ACC",
        "profiles_env": "COINBASE_TOP_BOT_PAPER_TRADING_PROFILES",
        "min_acc": "0.58",
        "profiles": "default",
    },
    "coinbase_futures": {
        "target": 30,
        "top_n_env": "COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_TOP_N",
        "min_acc_env": "COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_MIN_ACC",
        "profiles_env": "COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_PROFILES",
        "min_acc": "0.56",
        "profiles": "crypto_futures",
    },
}

CONTROL_PLANE_EXCLUDED_PROFILES = (
    "alpha_intelligence_evolution",
    "intelligence_layer_advancement",
    "apex_self_awareness_intelligence",
    "deep_recursive_awareness",
    "adaptive_intelligence_kernel",
    "system_self_awareness",
    "platform_brain",
)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _parse_today(raw: str | None) -> date:
    if raw:
        return date.fromisoformat(raw)
    return datetime.now(timezone.utc).date()


def _resolve_path(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _registry_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[Any] = []
    for key in ("sub_bots", "bots", "registry", "rows"):
        value = payload.get(key)
        if isinstance(value, list):
            candidates.extend(value)
    if not candidates and isinstance(payload.get("data"), list):
        candidates.extend(payload.get("data") or [])
    return [dict(row) for row in candidates if isinstance(row, dict)]


def _registry_counts(project_root: Path, registry_path: Path) -> dict[str, Any]:
    path = _resolve_path(registry_path, project_root)
    payload = load_json(path)
    rows = _registry_rows(payload)
    active_rows = [row for row in rows if bool(row.get("active", False))]
    paper_rows = [
        row
        for row in active_rows
        if bool(row.get("paper_live_data_enabled", False))
        or bool(row.get("paper_trading_enabled", False))
        or bool(row.get("paper_trade_enabled", False))
        or bool(row.get("paper_execution_allowed", False))
    ]
    control_plane_rows = [
        row
        for row in active_rows
        if any(
            profile in " ".join(str(row.get(key) or "").lower() for key in ("slot_kind", "sleeve_profile", "bot_id"))
            for profile in CONTROL_PLANE_EXCLUDED_PROFILES
        )
    ]
    data_collection_only_rows = [
        row
        for row in active_rows
        if str(row.get("lifecycle_state") or "").strip().lower() == "data_collection_only"
    ]
    return {
        "registry_path": str(path),
        "registered_bot_count": len(rows),
        "active_bot_count": len(active_rows),
        "paper_tagged_count": len(paper_rows),
        "control_plane_excluded_count": len(control_plane_rows),
        "data_collection_only_count": len(data_collection_only_rows),
    }


def _memory_gate(memory: dict[str, Any]) -> dict[str, Any]:
    snapshot = memory.get("memory_snapshot") if isinstance(memory.get("memory_snapshot"), dict) else {}
    compressed_store_gb = _safe_float(snapshot.get("compressed_store_gb"), 0.0)
    compressor_gb = _safe_float(snapshot.get("compressor_gb"), 0.0)
    swap_used_gb = _safe_float(snapshot.get("swap_used_gb"), 0.0)
    free_pct = _safe_float(snapshot.get("memory_free_pct"), 100.0)
    status = str(memory.get("overall_status") or "missing").strip().lower()
    recommended_profile = str(memory.get("recommended_profile") or "").strip().lower()
    hard_block = bool(
        status == "blocked"
        or compressed_store_gb >= 28.0
        or compressor_gb >= 16.0
        or swap_used_gb >= 12.0
        or free_pct < 12.0
    )
    advisory = bool(
        (compressed_store_gb >= 18.0 or compressor_gb >= 9.0 or swap_used_gb >= 4.0)
        and not hard_block
    )
    return {
        "ok": not hard_block,
        "status": "blocked" if hard_block else ("advisory" if advisory else "ready"),
        "overall_status": status,
        "recommended_profile": recommended_profile,
        "compressed_store_gb": compressed_store_gb,
        "compressor_gb": compressor_gb,
        "swap_used_gb": swap_used_gb,
        "memory_free_pct": free_pct,
    }


def _storage_gate(storage: dict[str, Any]) -> dict[str, Any]:
    backpressure = storage.get("backpressure") if isinstance(storage.get("backpressure"), dict) else {}
    severity = str(storage.get("severity") or storage.get("overall_status") or "missing").strip().lower()
    pressure_index = _safe_float(storage.get("pressure_index"), 0.0)
    core_pending = _safe_int(backpressure.get("core_pending_lines"), 0)
    total_pending = _safe_int(backpressure.get("total_pending_lines"), 0)
    hard_block = bool(
        severity in {"high", "critical", "blocked"}
        or pressure_index >= 0.35
        or core_pending >= 5000
        or total_pending >= 12000
    )
    return {
        "ok": not hard_block,
        "status": "blocked" if hard_block else "ready",
        "severity": severity,
        "pressure_index": round(pressure_index, 3),
        "core_pending_lines": core_pending,
        "total_pending_lines": total_pending,
    }


def _runtime_gate(runtime: dict[str, Any], registry_counts: dict[str, Any]) -> dict[str, Any]:
    contract = runtime.get("paper_capacity_contract") if isinstance(runtime.get("paper_capacity_contract"), dict) else {}
    compute_pressure = str(runtime.get("compute_pressure_level") or "").strip().lower()
    memory_pressure = str(runtime.get("memory_pressure_level") or "").strip().lower()
    throttle_profile = str(runtime.get("throttle_profile") or "").strip().lower()
    active_count = max(
        _safe_int(contract.get("active_bot_count"), 0),
        _safe_int(registry_counts.get("active_bot_count"), 0),
    )
    paper_tagged_count = max(
        _safe_int(contract.get("paper_tagged_count"), 0),
        _safe_int(registry_counts.get("paper_tagged_count"), 0),
    )
    ready_for_full_force = bool(contract.get("ready_for_700_bot_paper", False))
    pressure_limited = bool(contract.get("pressure_limited", False))
    hard_block = bool(
        active_count < TARGET_PAPER_BOTS
        or paper_tagged_count < TARGET_PAPER_BOTS
        or pressure_limited
        or throttle_profile == "protect_live"
        or compute_pressure == "high"
        or memory_pressure == "high"
    )
    if not ready_for_full_force and active_count >= 650 and not hard_block:
        ready_for_full_force = True
    return {
        "ok": not hard_block and ready_for_full_force,
        "status": "ready" if (not hard_block and ready_for_full_force) else "blocked",
        "ready_for_700_bot_paper": ready_for_full_force,
        "pressure_limited": pressure_limited,
        "throttle_profile": throttle_profile,
        "compute_pressure_level": compute_pressure,
        "memory_pressure_level": memory_pressure,
        "active_bot_count": active_count,
        "paper_tagged_count": paper_tagged_count,
    }


def _halt_gate(project_root: Path, global_halt: dict[str, Any]) -> dict[str, Any]:
    flag_path = project_root / "governance" / "health" / "GLOBAL_TRADING_HALT.flag"
    halt_active = bool(global_halt.get("halt", False) or global_halt.get("global_halt", False) or flag_path.exists())
    clear_blockers = global_halt.get("clear_blockers") if isinstance(global_halt.get("clear_blockers"), list) else []
    clear_ready = bool(global_halt.get("clear_ready", not halt_active))
    return {
        "ok": not halt_active and not clear_blockers,
        "status": "ready" if (not halt_active and not clear_blockers) else "blocked",
        "halt_active": halt_active,
        "clear_ready": clear_ready,
        "clear_blockers": clear_blockers,
        "flag_path": str(flag_path),
    }


def _blocker_list(gates: dict[str, dict[str, Any]]) -> list[str]:
    blockers: list[str] = []
    for name, gate in gates.items():
        if bool(gate.get("ok", False)):
            continue
        if name == "calendar":
            blockers.append("calendar_wait_until_2026-05-11")
        elif name == "runtime":
            blockers.append("runtime_capacity_not_ready_for_400_paper")
        elif name == "memory":
            blockers.append("memory_pressure_above_paper_400_gate")
        elif name == "storage":
            blockers.append("ingestion_or_backpressure_above_paper_400_gate")
        elif name == "global_halt":
            blockers.append("global_halt_or_clear_blocker_active")
        else:
            blockers.append(f"{name}_not_ready")
    return ordered_unique(blockers)


def _readiness_score(gates: dict[str, dict[str, Any]]) -> int:
    score = 100
    for gate in gates.values():
        if bool(gate.get("ok", False)):
            continue
        score -= 25 if str(gate.get("status") or "") == "blocked" else 10
    return max(score, 0)


def _allocation_summary() -> dict[str, Any]:
    total = sum(_safe_int(row.get("target"), 0) for row in PAPER_ALLOCATION.values())
    return {
        "target_total": total,
        "lanes": PAPER_ALLOCATION,
        "policy": "top_scored_per_lane_with_live_execution_locked",
    }


def _override_lines(payload: dict[str, Any]) -> list[str]:
    stage = str(payload.get("stage") or "planned")
    armed = bool(payload.get("armed", False))
    blockers = ",".join(str(item) for item in payload.get("blockers", []) if str(item).strip())
    base: dict[str, str] = {
        "PAPER_400_RAMP_ENABLED": "1",
        "PAPER_400_RAMP_STAGE": stage,
        "PAPER_400_RAMP_ARMED": "1" if armed else "0",
        "PAPER_400_RAMP_TARGET_BOTS": str(TARGET_PAPER_BOTS),
        "PAPER_400_RAMP_EARLIEST_DATE": EARLIEST_ACTIVATION_DATE.isoformat(),
        "PAPER_400_RAMP_READINESS_SCORE": str(_safe_int(payload.get("readiness_score"), 0)),
        "PAPER_400_RAMP_BLOCKERS": blockers,
        "PAPER_400_RAMP_SELECTION_POLICY": "top_scored_coverage_ready_control_plane_excluded",
        "PAPER_LIVE_DATA_STANDARD_ENABLED": "1",
        "PAPER_NEW_BOTS_REQUIRE_STANDARD": "1",
        "PAPER_STANDARD_SELECTION_POLICY": "legacy_established_or_promoted_after_standard",
        "PAPER_400_RAMP_OVERRIDE_SOURCE": "scripts/ops/paper_400_ramp_control.py",
        "ALLOW_ORDER_EXECUTION": "0",
        "MARKET_DATA_ONLY": "1",
        "PAPER_MIRROR_ALL_ACTIVE_SUB_BOTS": "0",
        "PAPER_BROKER_BRIDGE_ENABLED": "1",
        "PAPER_BROKER_BRIDGE_MODE": "jsonl",
        "PAPER_TRADE_LOCK": "1",
    }
    if armed:
        base.update(
            {
                "TOP_BOT_PAPER_TRADING_ENABLED": "1",
                "TOP_BOT_PAPER_TRADING_TOP_N": str(PAPER_ALLOCATION["schwab_equities"]["target"]),
                "TOP_BOT_PAPER_TRADING_MIN_ACC": str(PAPER_ALLOCATION["schwab_equities"]["min_acc"]),
                "TOP_BOT_PAPER_TRADING_PROFILES": str(PAPER_ALLOCATION["schwab_equities"]["profiles"]),
                "TOP_BOT_PAPER_TRADING_OPTIONS_ENABLED": "1",
                "TOP_BOT_PAPER_TRADING_OPTIONS_TOP_N": str(PAPER_ALLOCATION["schwab_options"]["target"]),
                "TOP_BOT_PAPER_TRADING_OPTIONS_MIN_ACC": str(PAPER_ALLOCATION["schwab_options"]["min_acc"]),
                "TOP_BOT_PAPER_TRADING_OPTIONS_PROFILES": str(PAPER_ALLOCATION["schwab_options"]["profiles"]),
                "PAPER_400_RAMP_AGGREGATE_TOP_N": str(TARGET_PAPER_BOTS),
                "PAPER_FULL_FORCE_STABILITY_MODE": "paper_400_buffered",
            }
        )
        for lane in PAPER_ALLOCATION.values():
            base[str(lane["top_n_env"])] = str(lane["target"])
            base[str(lane["min_acc_env"])] = str(lane["min_acc"])
            base[str(lane["profiles_env"])] = str(lane["profiles"])

    lines = [
        "# Auto-managed by scripts/ops/paper_400_ramp_control.py",
        f"# Generated at {payload.get('timestamp_utc') or iso_now()}",
    ]
    for key in sorted(base):
        lines.append(f"{key}={shlex.quote(str(base[key]))}")
    return lines


def write_override(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(_override_lines(payload)) + "\n", encoding="utf-8")


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    today: date | None = None,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    today_value = today or _parse_today(None)
    registry_counts = _registry_counts(project_root, registry_path)
    memory = load_json(health_root / "memory_efficiency_control_latest.json")
    runtime = load_json(health_root / "runtime_throttle_control_latest.json")
    storage = load_json(health_root / "ingestion_storage_control_latest.json")
    global_halt = load_json(health_root / "global_killswitch_latest.json")

    gates: dict[str, dict[str, Any]] = {
        "calendar": {
            "ok": today_value >= EARLIEST_ACTIVATION_DATE,
            "status": "ready" if today_value >= EARLIEST_ACTIVATION_DATE else "planned",
            "today": today_value.isoformat(),
            "earliest_activation_date": EARLIEST_ACTIVATION_DATE.isoformat(),
        },
        "global_halt": _halt_gate(project_root, global_halt),
        "memory": _memory_gate(memory),
        "storage": _storage_gate(storage),
        "runtime": _runtime_gate(runtime, registry_counts),
    }
    blockers = _blocker_list(gates)
    date_only_wait = blockers == ["calendar_wait_until_2026-05-11"]
    armed = bool(not blockers and today_value >= EARLIEST_ACTIVATION_DATE)
    stage = "armed" if armed else ("planned" if date_only_wait else "blocked")

    recommendations = ordered_unique(
        [
            "wait until Monday 2026-05-11 before arming the 400-bot paper target"
            if "calendar_wait_until_2026-05-11" in blockers
            else "",
            "keep the paper-trade lock active and live execution disabled while the ramp is armed"
            if armed
            else "",
            "./scripts/ops/opsctl.sh memory-efficiency apply --json"
            if "memory_pressure_above_paper_400_gate" in blockers
            else "",
            "./scripts/ops/opsctl.sh external-backlog-drain --apply --follow-through --json"
            if "ingestion_or_backpressure_above_paper_400_gate" in blockers
            else "",
            "./scripts/ops/opsctl.sh global-halt-refresh --json && ./scripts/ops/opsctl.sh global-halt-auto-clear --json"
            if "global_halt_or_clear_blocker_active" in blockers
            else "",
            "./scripts/ops/opsctl.sh runtime-throttle --apply --json"
            if "runtime_capacity_not_ready_for_400_paper" in blockers
            else "",
        ]
    )

    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": armed,
        "stage": stage,
        "armed": armed,
        "target_paper_bots": TARGET_PAPER_BOTS,
        "earliest_activation_date": EARLIEST_ACTIVATION_DATE.isoformat(),
        "today": today_value.isoformat(),
        "blockers": blockers,
        "readiness_score": _readiness_score(gates),
        "gates": gates,
        "registry_counts": registry_counts,
        "paper_allocation": _allocation_summary(),
        "self_awareness_contract": {
            "layer": "paper_400_ramp_cognitive_governor_v1",
            "purpose": "decide when the expanded bot fleet can move from collection-heavy mode into a 400-bot paper lane without causing halt, memory, or ingestion pressure",
            "reasoning_inputs": [
                "calendar activation window",
                "global halt clearance",
                "compressed memory and swap pressure",
                "runtime throttle full-force paper capacity",
                "ingestion backlog and storage pressure",
                "registry paper-tagged capacity",
            ],
            "intelligence_upgrades": [
                "calendar-aware future activation instead of manual top-n edits",
                "sticky override removal when gates degrade",
                "lane allocation across Schwab equities, options, futures, Coinbase spot, and Coinbase futures",
                "paper-trade lock reinforcement with live execution blocked",
                "explainable blockers for operator and self-model feedback loops",
            ],
            "next_upgrade_candidates": [
                "graduate from fixed lane allocation to rolling realized-latency allocation",
                "use paper PnL variance and rejection rate as dynamic throttles",
                "feed blocker history into the self-upgrade critic board",
            ],
        },
        "recommendations": recommendations,
    }
    return payload


def apply_payload(
    project_root: Path,
    payload: dict[str, Any],
    *,
    out_path: Path = DEFAULT_OUT_PATH,
    override_path: Path = DEFAULT_OVERRIDE_PATH,
) -> dict[str, Any]:
    resolved_out = _resolve_path(out_path, project_root)
    resolved_override = _resolve_path(override_path, project_root)
    write_override(resolved_override, payload)
    payload = dict(payload)
    payload["override"] = {
        "path": str(resolved_override),
        "written": True,
        "armed_values_written": bool(payload.get("armed", False)),
    }
    write_payload(resolved_out, payload)
    payload["out_path"] = str(resolved_out)
    return payload


def _print_human(payload: dict[str, Any]) -> None:
    print(f"paper_400_ramp stage={payload.get('stage')} armed={int(bool(payload.get('armed', False)))} target={TARGET_PAPER_BOTS}")
    blockers = payload.get("blockers") if isinstance(payload.get("blockers"), list) else []
    if blockers:
        print("blockers=" + ",".join(str(item) for item in blockers))
    allocation = payload.get("paper_allocation") if isinstance(payload.get("paper_allocation"), dict) else {}
    print(f"allocation_target_total={allocation.get('target_total', TARGET_PAPER_BOTS)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Guarded 400-bot paper trading ramp controller.")
    parser.add_argument("--apply", action="store_true", help="Write the health artifact and guarded env override.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    parser.add_argument("--today", help="Override the local date for tests or dry-run planning, YYYY-MM-DD.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH, help="Health artifact path.")
    parser.add_argument("--override", type=Path, default=DEFAULT_OVERRIDE_PATH, help="Runtime env override path.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH, help="Bot registry path.")
    args = parser.parse_args(argv)

    try:
        today_value = _parse_today(args.today)
    except Exception as exc:
        print(f"invalid --today date: {exc}", file=sys.stderr)
        return 2

    payload = build_payload(PROJECT_ROOT, today=today_value, registry_path=args.registry)
    if args.apply:
        payload = apply_payload(PROJECT_ROOT, payload, out_path=args.out, override_path=args.override)
    elif not args.apply:
        payload = {
            **payload,
            "override": {
                "path": str(_resolve_path(args.override, PROJECT_ROOT)),
                "written": False,
                "armed_values_written": False,
            },
            "out_path": str(_resolve_path(args.out, PROJECT_ROOT)),
        }

    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        _print_human(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
