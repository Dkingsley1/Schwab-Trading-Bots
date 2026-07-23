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
DEFAULT_PROMOTION_AUDIT_PATH = PROJECT_ROOT / "governance" / "health" / "paper_400_roster_promotion_latest.json"
DEFAULT_BACKUP_DIR = PROJECT_ROOT / "governance" / "lifecycle"
EARLIEST_ACTIVATION_DATE = date(2026, 5, 11)
TARGET_PAPER_BOTS = 400
PAPER_400_PROMOTION_COHORT = "paper_400_ramp_promoted"
OVERLAY_RAW_LIVE_MAX_CORE_LINES = 10_000
OVERLAY_RAW_LIVE_MAX_TOTAL_LINES = 15_000
OVERLAY_RAW_LIVE_MAX_AGE_SECONDS = 15 * 60
PAPER_PROFILE_DISABLED_SENTINEL = "__paper_profile_disabled_by_profitability_quarantine__"
COINBASE_PROBATIONARY_PROFILES = ("default", "crypto_futures")

CLEAN_SCHWAB_RUNTIME_PROFILES = (
    "volatility",
    "pairs_correlation",
    "stat_arb_market_neutral",
    "earnings_event",
    "commodity_inflation",
    "international_macro",
    "market_making_liquidity",
    "short_bias_hedge",
    "single_name_options_event",
    "rates_credit_macro",
    "cash_rotation_tactical",
    "futures_index_intraday",
    "futures_rates_curve",
    "futures_commodity_macro",
    "crypto_futures_basis",
    "futures_event_reaction",
    "options_on_futures_aggressive",
)
CLEAN_SCHWAB_OPTIONS_PROFILES = (
    "single_name_options_event",
    "options_on_futures_aggressive",
)
CLEAN_SCHWAB_FUTURES_PROFILES = (
    "schwab_futures",
    "futures_index_intraday",
    "futures_rates_curve",
    "futures_commodity_macro",
    "crypto_futures_basis",
    "futures_event_reaction",
)

PAPER_ALLOCATION: dict[str, dict[str, Any]] = {
    "schwab_equities": {
        "target": 200,
        "top_n_env": "SCHWAB_TOP_BOT_PAPER_TRADING_TOP_N",
        "min_acc_env": "SCHWAB_TOP_BOT_PAPER_TRADING_MIN_ACC",
        "profiles_env": "SCHWAB_TOP_BOT_PAPER_TRADING_PROFILES",
        "min_acc": "0.56",
        "profiles": ",".join(CLEAN_SCHWAB_RUNTIME_PROFILES),
    },
    "schwab_options": {
        "target": 40,
        "top_n_env": "SCHWAB_OPTIONS_TOP_BOT_PAPER_TRADING_TOP_N",
        "min_acc_env": "SCHWAB_OPTIONS_TOP_BOT_PAPER_TRADING_MIN_ACC",
        "profiles_env": "SCHWAB_OPTIONS_TOP_BOT_PAPER_TRADING_PROFILES",
        "min_acc": "0.56",
        "profiles": ",".join(CLEAN_SCHWAB_OPTIONS_PROFILES),
    },
    "schwab_futures": {
        "target": 80,
        "top_n_env": "SCHWAB_FUTURES_TOP_BOT_PAPER_TRADING_TOP_N",
        "min_acc_env": "SCHWAB_FUTURES_TOP_BOT_PAPER_TRADING_MIN_ACC",
        "profiles_env": "SCHWAB_FUTURES_TOP_BOT_PAPER_TRADING_PROFILES",
        "min_acc": "0.54",
        "profiles": ",".join(CLEAN_SCHWAB_FUTURES_PROFILES),
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


def _paper_enabled(row: dict[str, Any]) -> bool:
    return bool(
        row.get("paper_live_data_enabled", False)
        or row.get("paper_trading_enabled", False)
        or row.get("paper_trade_enabled", False)
        or row.get("paper_execution_allowed", False)
    )


def _is_control_plane_row(row: dict[str, Any]) -> bool:
    identity = " ".join(str(row.get(key) or "").lower() for key in ("slot_kind", "sleeve_profile", "bot_id"))
    return any(profile in identity for profile in CONTROL_PLANE_EXCLUDED_PROFILES)


def _registry_counts(project_root: Path, registry_path: Path) -> dict[str, Any]:
    path = _resolve_path(registry_path, project_root)
    payload = load_json(path)
    rows = _registry_rows(payload)
    active_rows = [row for row in rows if bool(row.get("active", False))]
    paper_rows = [row for row in active_rows if _paper_enabled(row)]
    control_plane_rows = [row for row in active_rows if _is_control_plane_row(row)]
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


def _bot_version(row: dict[str, Any]) -> int:
    import re

    match = re.search(r"(?:^|[^A-Za-z0-9])v(?P<version>\d+)", str(row.get("bot_id") or ""))
    if not match:
        return 10_000
    return _safe_int(match.group("version"), 10_000)


def _paper_roster_candidate_score(row: dict[str, Any]) -> tuple[float, float, float, int, int, str]:
    progress = row.get("data_collection_threshold_progress") if isinstance(row.get("data_collection_threshold_progress"), dict) else {}
    observations = max(
        _safe_int(row.get("data_collection_observations"), 0),
        _safe_int(row.get("collected_observation_count"), 0),
        _safe_int(progress.get("observations"), 0),
    )
    minimum_observations = max(
        _safe_int(row.get("minimum_training_observations"), 1000),
        _safe_int(progress.get("minimum_training_observations"), 1000),
        1,
    )
    observation_ratio = min(observations / max(minimum_observations, 1), 2.0)
    quality = max(_safe_float(row.get("quality_score"), 0.0), _safe_float(row.get("test_accuracy"), 0.0))
    training_ready = 1.0 if bool(row.get("data_collection_training_ready", False) or progress.get("training_ready", False)) else 0.0
    label_ready = 1.0 if bool(row.get("label_contract") or row.get("universal_label_contract")) else 0.0
    return (
        quality,
        training_ready,
        label_ready,
        observations,
        -_bot_version(row),
        str(row.get("bot_id") or ""),
    )


def _paper_roster_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if bool(row.get("active", False))
        and not bool(row.get("deleted_from_rotation", False))
        and not _paper_enabled(row)
        and not _is_control_plane_row(row)
        and not bool(row.get("paper_promotion_blocked", False))
    ]
    return sorted(candidates, key=_paper_roster_candidate_score, reverse=True)


def _mark_paper_400_promoted(row: dict[str, Any], *, now: str) -> None:
    prior_lifecycle = str(row.get("lifecycle_state") or "").strip()
    if prior_lifecycle and prior_lifecycle != "paper_live_data":
        row.setdefault("prior_lifecycle_state", prior_lifecycle)
    row["active"] = True
    row["lifecycle_state"] = "paper_live_data"
    row["paper_standard_cohort"] = PAPER_400_PROMOTION_COHORT
    row["paper_standard_status"] = "paper_live_data_enabled"
    row["paper_live_data_enabled"] = True
    row["paper_trading_enabled"] = True
    row["paper_trade_enabled"] = True
    row["paper_execution_allowed"] = True
    row["paper_runtime_stability_mode"] = "paper_400_guarded_buffered"
    row["paper_execution_queue_policy"] = "buffered_jsonl_batching"
    row["paper_live_data_source"] = "paper_400_ramp_guarded_roster_promotion"
    row["paper_400_ramp_promoted_utc"] = now
    row["paper_trade_lock_required"] = True
    row["paper_trade_lock_policy"] = "market_data_and_paper_only_until_explicit_graduation"
    row["direct_execution_allowed"] = False
    row["trading_enabled"] = False
    row["live_trading_enabled"] = False
    row["execution_enabled"] = False
    row["allocation_enabled"] = False
    row["live_rotation_blocked"] = True
    row["promotion_blocked_until"] = ""
    row["promotion_block_reason"] = ""


def promote_paper_roster(
    project_root: Path,
    registry_path: Path,
    *,
    target: int = TARGET_PAPER_BOTS,
    audit_path: Path = DEFAULT_PROMOTION_AUDIT_PATH,
    backup_dir: Path = DEFAULT_BACKUP_DIR,
) -> dict[str, Any]:
    path = _resolve_path(registry_path, project_root)
    payload = load_json(path)
    rows = payload.get("sub_bots") if isinstance(payload.get("sub_bots"), list) else []
    active_rows = [row for row in rows if isinstance(row, dict) and bool(row.get("active", False))]
    paper_rows_before = [row for row in active_rows if _paper_enabled(row)]
    needed = max(int(target) - len(paper_rows_before), 0)
    candidates = _paper_roster_candidates([row for row in rows if isinstance(row, dict)])
    selected = candidates[:needed]
    now = iso_now()
    audit = {
        "timestamp_utc": now,
        "schema_version": 1,
        "ok": needed == 0 or len(selected) >= needed,
        "overall_status": "ready" if needed == 0 else ("applied" if len(selected) >= needed else "blocked"),
        "target_paper_bots": int(target),
        "paper_count_before": len(paper_rows_before),
        "needed": needed,
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "selected_bot_ids": [str(row.get("bot_id") or "") for row in selected],
        "policy": "promote active non-control-plane bots to guarded paper-live-data only; live execution stays disabled",
        "registry_path": str(path),
    }
    if needed > 0 and len(selected) < needed:
        write_payload(_resolve_path(audit_path, project_root), audit)
        return audit

    if needed > 0:
        backup_root = _resolve_path(backup_dir, project_root)
        backup_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = backup_root / f"master_bot_registry.paper_400_ramp_{stamp}.json"
        backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        for row in selected:
            _mark_paper_400_promoted(row, now=now)
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        summary["paper_live_data_enabled_bots"] = sum(1 for row in active_rows if isinstance(row, dict) and _paper_enabled(row))
        summary["paper_400_ramp_promoted_bots"] = sum(
            1 for row in active_rows if isinstance(row, dict) and str(row.get("paper_standard_cohort") or "") == PAPER_400_PROMOTION_COHORT
        )
        payload["summary"] = summary
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        audit["backup_path"] = str(backup_path)

    active_rows_after = [row for row in rows if isinstance(row, dict) and bool(row.get("active", False))]
    audit["paper_count_after"] = sum(1 for row in active_rows_after if _paper_enabled(row))
    audit["live_execution_locked"] = all(
        not bool(row.get("live_trading_enabled", False))
        and not bool(row.get("direct_execution_allowed", False))
        and not bool(row.get("trading_enabled", False))
        for row in selected
    )
    write_payload(_resolve_path(audit_path, project_root), audit)
    return audit


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


def _overlay_only_storage_relief(backpressure: dict[str, Any]) -> dict[str, Any]:
    effective_raw_live = backpressure.get("effective_raw_live") if isinstance(backpressure.get("effective_raw_live"), dict) else {}
    effective_raw_live_estimate = (
        effective_raw_live.get("raw_live_estimate")
        if isinstance(effective_raw_live.get("raw_live_estimate"), dict)
        else {}
    )
    raw_live_source = "raw_live"
    if effective_raw_live_estimate:
        raw_live = effective_raw_live_estimate
        raw_live_source = "effective_raw_live.raw_live_estimate"
    elif effective_raw_live:
        raw_live = effective_raw_live
        raw_live_source = "effective_raw_live"
    else:
        raw_live = backpressure.get("raw_live") if isinstance(backpressure.get("raw_live"), dict) else {}
    raw_core = _safe_int(raw_live.get("core_pending_lines"), 0)
    raw_total = _safe_int(raw_live.get("total_pending_lines"), 0)
    raw_oldest = _safe_float(raw_live.get("oldest_pending_age_seconds"), 0.0)
    overlay_adjusted = bool(backpressure.get("overlay_adjusted", False))
    raw_live_clear = bool(
        raw_live
        and raw_core <= OVERLAY_RAW_LIVE_MAX_CORE_LINES
        and raw_total <= OVERLAY_RAW_LIVE_MAX_TOTAL_LINES
        and raw_oldest <= OVERLAY_RAW_LIVE_MAX_AGE_SECONDS
    )
    return {
        "active": bool(overlay_adjusted and raw_live_clear),
        "overlay_adjusted": overlay_adjusted,
        "raw_live_clear": raw_live_clear,
        "raw_live_source": str(raw_live.get("source") or raw_live_source),
        "raw_live": {
            "core_pending_lines": raw_core,
            "total_pending_lines": raw_total,
            "oldest_pending_age_seconds": round(raw_oldest, 3),
            "max_core_pending_lines": OVERLAY_RAW_LIVE_MAX_CORE_LINES,
            "max_total_pending_lines": OVERLAY_RAW_LIVE_MAX_TOTAL_LINES,
            "max_oldest_pending_age_seconds": OVERLAY_RAW_LIVE_MAX_AGE_SECONDS,
        },
        "policy": "allow paper admission gates to ignore SQL-overlay-only pressure only when raw live backlog is cool",
    }


def _storage_gate(storage: dict[str, Any]) -> dict[str, Any]:
    backpressure = storage.get("backpressure") if isinstance(storage.get("backpressure"), dict) else {}
    severity = str(storage.get("severity") or storage.get("overall_status") or "missing").strip().lower()
    pressure_index = _safe_float(storage.get("pressure_index"), 0.0)
    core_pending = _safe_int(backpressure.get("core_pending_lines"), 0)
    total_pending = _safe_int(backpressure.get("total_pending_lines"), 0)
    overlay_relief = _overlay_only_storage_relief(backpressure)
    pressure_advisory = bool(
        severity in {"stable", "ready", ""}
        and pressure_index >= 0.25
        and pressure_index < 0.50
        and (bool(overlay_relief.get("active", False)) or core_pending < 5000)
        and total_pending < 12000
    )
    hard_block = bool(
        not bool(overlay_relief.get("active", False))
        and (
            severity in {"high", "critical", "blocked"}
            or pressure_index >= 0.50
            or core_pending >= 5000
            or total_pending >= 12000
        )
        and not pressure_advisory
    )
    status = "blocked" if hard_block else ("overlay_drain_advisory" if bool(overlay_relief.get("active", False)) else ("storage_pressure_advisory" if pressure_advisory else "ready"))
    return {
        "ok": not hard_block,
        "status": status,
        "severity": severity,
        "pressure_index": round(pressure_index, 3),
        "pressure_target": 0.25,
        "pressure_advisory_ceiling": 0.50,
        "pressure_advisory": pressure_advisory,
        "core_pending_lines": core_pending,
        "total_pending_lines": total_pending,
        "overlay_only_relief": overlay_relief,
    }


def _runtime_gate(runtime: dict[str, Any], registry_counts: dict[str, Any]) -> dict[str, Any]:
    contract = runtime.get("paper_capacity_contract") if isinstance(runtime.get("paper_capacity_contract"), dict) else {}
    paper_policy = runtime.get("paper_execution_policy") if isinstance(runtime.get("paper_execution_policy"), dict) else {}
    runtime_policy = contract.get("runtime_policy") if isinstance(contract.get("runtime_policy"), dict) else {}
    advisory = (
        runtime.get("soft_cap_advisory_reclassification")
        if isinstance(runtime.get("soft_cap_advisory_reclassification"), dict)
        else {}
    )
    advisory_measurements = (
        advisory.get("measurements")
        if isinstance(advisory.get("measurements"), dict)
        else {}
    )
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
    attribution_capacity_advisory = bool(
        contract.get("attribution_capacity_advisory", False)
        or (
            advisory.get("active", False)
            and str(advisory.get("reason") or "")
            in {
                "external_high_compute_pressure_is_capacity_limited_advisory_not_bot_runtime_degradation",
                "external_high_compute_with_bounded_storage_overlay_is_capacity_limited_advisory",
            }
            and bool(advisory_measurements.get("external_high_compute_guarded", False))
            and bool(advisory_measurements.get("storage_ready_for_runtime_advisory", True))
        )
    )
    compute_pressure_ready = bool(compute_pressure != "high" or attribution_capacity_advisory)
    active_bot_capacity_ready = active_count >= TARGET_PAPER_BOTS
    paper_roster_ready = paper_tagged_count >= TARGET_PAPER_BOTS
    paper_policy_blockers = [str(item) for item in paper_policy.get("blockers") or []]
    paper_pressure_bypass = bool(
        paper_policy.get("paper_execution_allowed", False)
        and not bool(paper_policy.get("pause_paper_execution", False))
        and bool(paper_policy.get("pressure_pause_bypassed", False))
        and set(paper_policy_blockers).issubset({"runtime_capacity_not_ready_for_400_paper"})
        and "paper_ramp" in str(
            paper_policy.get("reason") or paper_policy.get("pressure_pause_bypass_reason") or ""
        ).lower()
    )
    paper_execution_clean = bool(
        (
            paper_policy.get("paper_execution_allowed", False)
            and not bool(paper_policy.get("pause_paper_execution", False))
            and bool(paper_policy.get("armed", False))
            and str(paper_policy.get("stage") or "").strip().lower() == "armed"
            and bool(paper_policy.get("ok", False))
            and not paper_policy_blockers
        )
        or paper_pressure_bypass
    )
    live_execution_locked = bool(runtime_policy.get("live_execution_blocked", False))
    capacity_limited_armed = bool(
        active_bot_capacity_ready
        and paper_roster_ready
        and paper_execution_clean
        and live_execution_locked
        and memory_pressure != "high"
        and throttle_profile != "protect_live"
    )
    runtime_pressure_ready = bool(
        (not pressure_limited or attribution_capacity_advisory)
        and throttle_profile != "protect_live"
        and compute_pressure_ready
        and memory_pressure != "high"
    )
    runtime_capacity_ready = bool(runtime_pressure_ready and (ready_for_full_force or active_count >= 650))
    if not ready_for_full_force and active_count >= 650 and runtime_pressure_ready:
        ready_for_full_force = True
        runtime_capacity_ready = True
    if capacity_limited_armed:
        runtime_capacity_ready = True
    hard_block = bool(not active_bot_capacity_ready or not paper_roster_ready or not runtime_capacity_ready)
    blockers: list[str] = []
    if not active_bot_capacity_ready:
        blockers.append("active_bot_count_below_400_target")
    if not paper_roster_ready:
        blockers.append("paper_roster_below_400_target")
    if not runtime_capacity_ready:
        blockers.append("runtime_capacity_not_ready_for_400_paper")
    return {
        "ok": not hard_block and (ready_for_full_force or capacity_limited_armed),
        "status": (
            "ready"
            if (not hard_block and ready_for_full_force)
            else ("capacity_limited_armed" if (not hard_block and capacity_limited_armed) else "blocked")
        ),
        "blockers": blockers,
        "active_bot_capacity_ready": active_bot_capacity_ready,
        "paper_roster_ready": paper_roster_ready,
        "runtime_pressure_ready": runtime_pressure_ready,
        "runtime_capacity_ready": runtime_capacity_ready,
        "ready_for_700_bot_paper": ready_for_full_force,
        "capacity_limited_armed": capacity_limited_armed,
        "paper_execution_clean": paper_execution_clean,
        "paper_pressure_bypass": paper_pressure_bypass,
        "live_execution_locked": live_execution_locked,
        "pressure_limited": pressure_limited,
        "attribution_capacity_advisory": attribution_capacity_advisory,
        "compute_pressure_ready": compute_pressure_ready,
        "throttle_profile": throttle_profile,
        "compute_pressure_level": compute_pressure,
        "memory_pressure_level": memory_pressure,
        "active_bot_count": active_count,
        "paper_tagged_count": paper_tagged_count,
    }


def _halt_clear_relief(global_halt: dict[str, Any], data_plane: dict[str, Any], plumbing: dict[str, Any] | None = None) -> dict[str, Any]:
    clear_blockers = global_halt.get("clear_blockers") if isinstance(global_halt.get("clear_blockers"), list) else []
    blocker_set = {str(item).strip() for item in clear_blockers if str(item).strip()}
    metrics = global_halt.get("metrics") if isinstance(global_halt.get("metrics"), dict) else {}
    plumbing = plumbing if isinstance(plumbing, dict) else {}
    plumbing_relief = plumbing.get("global_clear_relief") if isinstance(plumbing.get("global_clear_relief"), dict) else {}
    plumbing_paper_contract = (
        plumbing.get("paper_ramp_relief_contract")
        if isinstance(plumbing.get("paper_ramp_relief_contract"), dict)
        else {}
    )
    execution_expected = bool(metrics.get("execution_expected", False))
    restart_storm_isolation = metrics.get("restart_storm_isolation") if isinstance(metrics.get("restart_storm_isolation"), dict) else {}
    recovery_state = str(data_plane.get("recovery_state") or "").strip().lower()
    overall_status = str(data_plane.get("overall_status") or "").strip().lower()
    write_failures = _safe_int(data_plane.get("write_failure_count"), 0)
    snapshot_failures = _safe_int(data_plane.get("account_snapshot_failure_count"), 0)
    queue_depth = _safe_int(data_plane.get("queue_depth"), 0)
    write_path_requested = "write_path_recovery_pending" in blocker_set
    restart_storm_requested = "restart_storm_active" in blocker_set
    plumbing_bounded_write_recovery = bool(
        write_path_requested
        and plumbing.get("ok", False)
        and plumbing_relief.get("bounded_write_recovery", False)
        and plumbing_paper_contract.get("bounded_write_recovery", False)
    )
    bounded_write_recovery = bool(
        plumbing_bounded_write_recovery
        or (
            write_path_requested
            and not execution_expected
            and overall_status in {"ready", "degraded"}
            and recovery_state in {"stable", "recovering_under_guard", "recovering"}
            and write_failures <= 5
            and snapshot_failures <= 0
            and queue_depth < 10_000
        )
    )
    isolated_restart_storm = bool(
        restart_storm_requested
        and not execution_expected
        and bool(restart_storm_isolation.get("safe_to_clear_when_not_executing", False))
        and _safe_int(restart_storm_isolation.get("isolated_count"), 0) > 0
        and _safe_int(restart_storm_isolation.get("execution_blocking_count"), 0) == 0
    )
    relief_active = bool(
        blocker_set
        and blocker_set <= {"write_path_recovery_pending", "restart_storm_active"}
        and (not write_path_requested or bounded_write_recovery)
        and (not restart_storm_requested or isolated_restart_storm)
    )
    if relief_active and write_path_requested and restart_storm_requested:
        relief_status = "bounded_clear_blocker_advisory"
    elif relief_active and restart_storm_requested:
        relief_status = "restart_storm_isolation_advisory"
    elif relief_active and write_path_requested:
        relief_status = "write_path_recovery_advisory"
    else:
        relief_status = ""
    return {
        "active": relief_active,
        "status": relief_status,
        "clear_blockers": sorted(blocker_set),
        "bounded_write_recovery": bounded_write_recovery,
        "plumbing_bounded_write_recovery": plumbing_bounded_write_recovery,
        "isolated_restart_storm": isolated_restart_storm,
        "restart_storm_isolation": restart_storm_isolation,
        "data_plane_status": overall_status,
        "recovery_state": recovery_state,
        "write_failure_count": write_failures,
        "account_snapshot_failure_count": snapshot_failures,
        "queue_depth": queue_depth,
        "execution_expected": execution_expected,
        "plumbing_status": str(plumbing.get("overall_status") or ""),
        "plumbing_score": _safe_int(plumbing.get("plumbing_score"), 0),
        "policy": "allow paper ramp planning to treat inactive-halt bounded write-path recovery or isolated read-only restart storms as advisory while live execution is off",
    }


def _halt_gate(project_root: Path, global_halt: dict[str, Any], data_plane: dict[str, Any], plumbing: dict[str, Any] | None = None) -> dict[str, Any]:
    flag_path = project_root / "governance" / "health" / "GLOBAL_TRADING_HALT.flag"
    safe_clear = global_halt.get("safe_clear") if isinstance(global_halt.get("safe_clear"), dict) else {}
    safe_clear_hard_blockers = safe_clear.get("hard_blockers") if isinstance(safe_clear.get("hard_blockers"), list) else None
    raw_clear_blockers = global_halt.get("clear_blockers") if isinstance(global_halt.get("clear_blockers"), list) else []
    clear_blockers = list(safe_clear_hard_blockers) if safe_clear_hard_blockers is not None else list(raw_clear_blockers)
    halt_latched = bool(
        global_halt.get(
            "halt_latched",
            global_halt.get("halt", False) or global_halt.get("global_halt", False),
        )
    )
    halt_required = bool(global_halt.get("halt_required", False) or global_halt.get("would_rehalt", False))
    halt_active = bool(halt_latched or flag_path.exists())
    clear_ready = bool(global_halt.get("clear_ready", not halt_active and not halt_required))
    clear_relief = _halt_clear_relief(global_halt, data_plane, plumbing)
    ok = bool(not halt_active and not halt_required and (not clear_blockers or clear_relief.get("active", False)))
    if ok and not clear_blockers:
        status = "ready"
    elif ok:
        status = str(clear_relief.get("status") or "clear_blocker_advisory")
    elif halt_active:
        status = "halt_latched"
    elif halt_required:
        status = "halt_required"
    else:
        status = "blocked"
    return {
        "ok": ok,
        "status": status,
        "halt_active": halt_active,
        "halt_latched": halt_latched,
        "halt_required": halt_required,
        "would_rehalt": bool(global_halt.get("would_rehalt", False)),
        "halt_posture": str(global_halt.get("halt_posture") or ""),
        "clear_ready": clear_ready,
        "clear_blockers": clear_blockers,
        "raw_clear_blockers": raw_clear_blockers,
        "safe_clear": safe_clear,
        "clear_blocker_relief": clear_relief,
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
            runtime_blockers = gate.get("blockers") if isinstance(gate.get("blockers"), list) else []
            blockers.extend(str(item) for item in runtime_blockers if str(item).strip())
            if not runtime_blockers:
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


def _csv_profiles(raw: Any) -> list[str]:
    return ordered_unique(str(item).strip().lower() for item in str(raw or "").split(",") if str(item).strip())


def _weak_profiles_from_profitability_controls(project_root: Path) -> set[str]:
    health_root = project_root / "governance" / "health"
    weak: set[str] = set()
    for name in (
        "paper_runtime_profitability_controls_latest.json",
        "paper_profitability_control_latest.json",
    ):
        payload = load_json(health_root / name)
        if not isinstance(payload, dict):
            continue
        recovery = payload.get("raw_profitability_a_recovery_contract")
        if isinstance(recovery, dict):
            for profile in recovery.get("weak_profiles") if isinstance(recovery.get("weak_profiles"), list) else []:
                value = str(profile or "").strip().lower()
                if value:
                    weak.add(value)
        improvement = payload.get("raw_profitability_improvement_contract")
        if isinstance(improvement, dict):
            weak_contract = improvement.get("weak_sleeve_zero_entry_contract")
            if isinstance(weak_contract, dict):
                for row in weak_contract.get("profiles") if isinstance(weak_contract.get("profiles"), list) else []:
                    if isinstance(row, dict) and bool(row.get("block_new_entries", False)):
                        value = str(row.get("profile") or "").strip().lower()
                        if value:
                            weak.add(value)
        active_controls = payload.get("active_profile_controls")
        if isinstance(active_controls, dict):
            for profile, control in active_controls.items():
                if isinstance(control, dict) and (
                    str(control.get("action") or "").strip().lower() == "quarantine_new_entries"
                    or bool(control.get("block_new_entries", False))
                ):
                    value = str(profile or "").strip().lower()
                    if value:
                        weak.add(value)
        profile_controls = payload.get("profile_controls")
        if isinstance(profile_controls, dict):
            for profile, control in profile_controls.items():
                if isinstance(control, dict) and (
                    str(control.get("action") or "").strip().lower() == "quarantine_new_entries"
                    or bool(control.get("block_new_entries", False))
                ):
                    value = str(profile or "").strip().lower()
                    if value:
                        weak.add(value)
    return weak


def _clean_profile_csv(raw: Any, weak_profiles: set[str], *, allow_weak_profiles: set[str] | None = None) -> str:
    allowed = allow_weak_profiles or set()
    cleaned = [profile for profile in _csv_profiles(raw) if profile not in weak_profiles or profile in allowed]
    return ",".join(cleaned) if cleaned else PAPER_PROFILE_DISABLED_SENTINEL


def _paper_allocation_for_runtime(project_root: Path) -> dict[str, dict[str, Any]]:
    weak_profiles = _weak_profiles_from_profitability_controls(project_root)
    allocation: dict[str, dict[str, Any]] = {}
    for lane_name, lane in PAPER_ALLOCATION.items():
        lane_out = dict(lane)
        original_profiles = _csv_profiles(lane.get("profiles"))
        lane_out["profiles_before_profitability_quarantine"] = ",".join(original_profiles)
        coinbase_probationary = lane_name in {"coinbase_spot", "coinbase_futures"}
        allowed_weak = set(original_profiles) if coinbase_probationary else set()
        lane_out["profiles"] = _clean_profile_csv(lane.get("profiles"), weak_profiles, allow_weak_profiles=allowed_weak)
        lane_out["profitability_quarantined_profiles"] = sorted(profile for profile in original_profiles if profile in weak_profiles)
        lane_out["profile_quarantine_sentinel_active"] = lane_out["profiles"] == PAPER_PROFILE_DISABLED_SENTINEL
        lane_out["paper_probation_active"] = bool(
            coinbase_probationary
            and lane_out["profiles"] != PAPER_PROFILE_DISABLED_SENTINEL
            and any(profile in weak_profiles for profile in original_profiles)
        )
        if lane_out["paper_probation_active"]:
            lane_out["paper_probation_reason"] = "weak_coinbase_profile_guarded_paper_only_retest"
        allocation[lane_name] = lane_out
    return allocation


def _allocation_summary(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    lanes = _paper_allocation_for_runtime(project_root)
    total = sum(_safe_int(row.get("target"), 0) for row in lanes.values())
    return {
        "target_total": total,
        "lanes": lanes,
        "policy": "all_eligible_paper_live_data_with_live_execution_locked",
        "weak_profile_source": "paper_profitability_runtime_controls",
        "disabled_profile_sentinel": PAPER_PROFILE_DISABLED_SENTINEL,
    }


def _override_lines(payload: dict[str, Any]) -> list[str]:
    stage = str(payload.get("stage") or "planned")
    armed = bool(payload.get("armed", False))
    blockers = ",".join(str(item) for item in payload.get("blockers", []) if str(item).strip())
    gates = payload.get("gates") if isinstance(payload.get("gates"), dict) else {}
    runtime_gate = gates.get("runtime") if isinstance(gates.get("runtime"), dict) else {}
    paper_tagged_count = max(_safe_int(runtime_gate.get("paper_tagged_count"), TARGET_PAPER_BOTS), 0)
    eligible_top_n = max(TARGET_PAPER_BOTS, paper_tagged_count)
    allocation = payload.get("paper_allocation") if isinstance(payload.get("paper_allocation"), dict) else {}
    lanes = allocation.get("lanes") if isinstance(allocation.get("lanes"), dict) else PAPER_ALLOCATION
    schwab_soak_profiles = str(
        (lanes.get("schwab_equities") if isinstance(lanes.get("schwab_equities"), dict) else {}).get("profiles")
        or ""
    )
    base: dict[str, str] = {
        "PAPER_400_RAMP_ENABLED": "1",
        "PAPER_400_RAMP_STAGE": stage,
        "PAPER_400_RAMP_ARMED": "1" if armed else "0",
        "PAPER_400_RAMP_TARGET_BOTS": str(TARGET_PAPER_BOTS),
        "PAPER_400_RAMP_EARLIEST_DATE": EARLIEST_ACTIVATION_DATE.isoformat(),
        "PAPER_400_RAMP_READINESS_SCORE": str(_safe_int(payload.get("readiness_score"), 0)),
        "PAPER_400_RAMP_BLOCKERS": blockers,
        "PAPER_400_RAMP_SELECTION_POLICY": "all_eligible_paper_live_data_when_mirror_all_active_enabled",
        "PAPER_LIVE_DATA_STANDARD_ENABLED": "1",
        "PAPER_NEW_BOTS_REQUIRE_STANDARD": "1",
        "PAPER_STANDARD_SELECTION_POLICY": "legacy_established_or_promoted_after_standard",
        "PAPER_400_RAMP_OVERRIDE_SOURCE": "scripts/ops/paper_400_ramp_control.py",
        "ALLOW_ORDER_EXECUTION": "0",
        "MARKET_DATA_ONLY": "1",
        "PAPER_MIRROR_ALL_ACTIVE_SUB_BOTS": "1",
        "PAPER_BROKER_BRIDGE_ENABLED": "1",
        "PAPER_BROKER_BRIDGE_MODE": "jsonl",
        "PAPER_SOAK_SPECIALIZED_ALLOWLIST_BYPASS_FANOUT": "1",
        "RUN_ALL_SLEEVES_WITH_SPECIALIZED_SLEEVES": "1",
        "RUN_ALL_SLEEVES_SPECIALIZED_PROFILE_ALLOWLIST": schwab_soak_profiles,
        "COINBASE_PAPER_PROBATION_ENABLED": "1",
        "COINBASE_PAPER_PROBATION_REASON": "weak_profiles_allowed_for_guarded_paper_only_retest",
        "COINBASE_PAPER_PROBATIONARY_PROFILES": ",".join(COINBASE_PROBATIONARY_PROFILES),
        "PAPER_TRADE_LOCK": "1",
    }
    if armed:
        base.update(
            {
                "TOP_BOT_PAPER_TRADING_ENABLED": "1",
                "TOP_BOT_PAPER_TRADING_TOP_N": str(eligible_top_n),
                "TOP_BOT_PAPER_TRADING_MIN_ACC": str(lanes["schwab_equities"]["min_acc"]),
                "TOP_BOT_PAPER_TRADING_PROFILES": str(lanes["schwab_equities"]["profiles"]),
                "TOP_BOT_PAPER_TRADING_OPTIONS_ENABLED": "1",
                "TOP_BOT_PAPER_TRADING_OPTIONS_TOP_N": str(eligible_top_n),
                "TOP_BOT_PAPER_TRADING_OPTIONS_MIN_ACC": str(lanes["schwab_options"]["min_acc"]),
                "TOP_BOT_PAPER_TRADING_OPTIONS_PROFILES": str(lanes["schwab_options"]["profiles"]),
                "PAPER_400_RAMP_AGGREGATE_TOP_N": str(eligible_top_n),
                "PAPER_FULL_FORCE_STABILITY_MODE": "all_eligible_paper_buffered",
            }
        )
        for lane_name, lane in lanes.items():
            lane_top_n = _safe_int(lane.get("target"), eligible_top_n) if str(lane_name).startswith("coinbase_") else eligible_top_n
            base[str(lane["top_n_env"])] = str(lane_top_n)
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
    global_halt = load_json(health_root / "global_halt_auto_clear_latest.json") or load_json(health_root / "global_killswitch_latest.json")
    data_plane = load_json(health_root / "data_plane_recovery_controller_latest.json")
    plumbing = load_json(health_root / "system_plumbing_control_latest.json")

    gates: dict[str, dict[str, Any]] = {
        "calendar": {
            "ok": today_value >= EARLIEST_ACTIVATION_DATE,
            "status": "ready" if today_value >= EARLIEST_ACTIVATION_DATE else "planned",
            "today": today_value.isoformat(),
            "earliest_activation_date": EARLIEST_ACTIVATION_DATE.isoformat(),
        },
        "global_halt": _halt_gate(project_root, global_halt, data_plane, plumbing),
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
            "keep paper admissions closed until the paper roster is intentionally raised to the 400-bot target"
            if "paper_roster_below_400_target" in blockers
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
        "paper_allocation": _allocation_summary(project_root),
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
    parser.add_argument("--promote-roster", action="store_true", help="Promote enough guarded paper-only registry rows to satisfy the 400-bot paper target.")
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

    promotion_result: dict[str, Any] = {}
    if bool(args.promote_roster):
        if not bool(args.apply):
            promotion_result = {"ok": False, "overall_status": "preview_only_requires_apply", "apply_required": True}
        else:
            promotion_result = promote_paper_roster(PROJECT_ROOT, args.registry)

    payload = build_payload(PROJECT_ROOT, today=today_value, registry_path=args.registry)
    if promotion_result:
        payload["promotion_result"] = promotion_result
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
