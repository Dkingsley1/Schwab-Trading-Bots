#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "backlog_pump_infrabots_latest.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.backlog_pump_infrabots_override"
PROTECTED_VOLUMES = ["/Volumes/VIDEO"]


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _status(payload: dict[str, Any], default: str = "missing") -> str:
    for key in ("overall_status", "status"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    if payload.get("ok") is True:
        return "ready"
    if payload.get("ok") is False:
        return "blocked"
    return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "y"}


def _lock_open_enabled() -> bool:
    return any(
        _env_bool(name)
        for name in (
            "BACKLOG_PCORE_DRAIN_LOCK_OPEN",
            "BACKLOG_ACCELERATOR_LOCK_OPEN",
            "BACKLOG_DRAIN_LOCK_OPEN",
            "BACKLOG_FORCE_LOCK_OPEN",
            "OPERATOR_DRAIN_LOCK_OPEN",
        )
    )


def _infer_sleeve(source_rel: str) -> str:
    text = str(source_rel or "").strip()
    if not text:
        return "unknown"
    parts = [part for part in text.split("/") if part]
    stem = parts[1] if len(parts) > 1 and parts[0] == "decisions" else parts[0]
    stem = stem.removeprefix("shadow_")
    for suffix in ("_equities", "_crypto", "_schwab", "_latest"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem or "unknown"


def _source_rows(accelerator: dict[str, Any], writer: dict[str, Any]) -> list[dict[str, Any]]:
    storage = _as_dict(accelerator.get("storage_contract"))
    rows = _as_list(storage.get("oldest_sources"))
    if not rows:
        stale = _as_dict(_as_dict(writer.get("drain_effectiveness")).get("false_alarm_guard"))
        rows = _as_list(stale.get("oldest_sources"))
    out: list[dict[str, Any]] = []
    for item in rows:
        row = _as_dict(item)
        source_rel = str(row.get("source_rel") or "")
        out.append(
            {
                "source_rel": source_rel,
                "sleeve": _infer_sleeve(source_rel),
                "shard": str(row.get("shard") or ""),
                "pressure_lane": str(row.get("pressure_lane") or ""),
                "pending_lines": _safe_int(row.get("pending_lines"), 0),
                "oldest_pending_age_seconds": round(_safe_float(row.get("oldest_pending_age_seconds"), 0.0), 3),
                "age_ratio": round(_safe_float(row.get("age_ratio"), 0.0), 3),
                "total_lines": _safe_int(row.get("total_lines"), 0),
                "last_line": _safe_int(row.get("last_line"), 0),
            }
        )
    return out


def _storage_control_source_rows(storage_control: dict[str, Any]) -> list[dict[str, Any]]:
    relief = _as_dict(storage_control.get("backlog_relief_contract"))
    rows: list[dict[str, Any]] = []
    for issue in _as_list(relief.get("issues")):
        evidence = _as_dict(_as_dict(issue).get("evidence"))
        for item in _as_list(evidence.get("oldest_sources")):
            row = _as_dict(item)
            source_rel = str(row.get("source_rel") or "")
            if not source_rel:
                continue
            rows.append(
                {
                    "source_rel": source_rel,
                    "sleeve": _infer_sleeve(source_rel),
                    "shard": str(row.get("shard") or ""),
                    "pressure_lane": str(row.get("pressure_lane") or ""),
                    "pending_lines": _safe_int(row.get("pending_lines"), 0),
                    "oldest_pending_age_seconds": round(_safe_float(row.get("oldest_pending_age_seconds"), 0.0), 3),
                    "age_ratio": round(_safe_float(row.get("age_ratio"), 0.0), 3),
                    "total_lines": _safe_int(row.get("total_lines"), 0),
                    "last_line": _safe_int(row.get("last_line"), 0),
                }
            )
    return rows


def _merge_rows(primary: list[dict[str, Any]], secondary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in [*primary, *secondary]:
        source_rel = str(row.get("source_rel") or "")
        key = source_rel or f"{row.get('shard') or 'unknown'}:{len(merged)}"
        existing = merged.get(key)
        if not existing:
            merged[key] = dict(row)
            continue
        existing["pending_lines"] = max(_safe_int(existing.get("pending_lines"), 0), _safe_int(row.get("pending_lines"), 0))
        existing["oldest_pending_age_seconds"] = max(
            _safe_float(existing.get("oldest_pending_age_seconds"), 0.0),
            _safe_float(row.get("oldest_pending_age_seconds"), 0.0),
        )
        for field in ("sleeve", "shard", "pressure_lane", "total_lines", "last_line", "age_ratio"):
            if row.get(field) and not existing.get(field):
                existing[field] = row.get(field)
    return list(merged.values())


def _group_by_sleeve(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        sleeve = str(row.get("sleeve") or "unknown")
        bucket = grouped.setdefault(
            sleeve,
            {
                "sleeve": sleeve,
                "pending_lines": 0,
                "max_age_seconds": 0.0,
                "source_count": 0,
                "shards": set(),
                "top_sources": [],
            },
        )
        bucket["pending_lines"] = _safe_int(bucket.get("pending_lines"), 0) + _safe_int(row.get("pending_lines"), 0)
        bucket["max_age_seconds"] = max(_safe_float(bucket.get("max_age_seconds"), 0.0), _safe_float(row.get("oldest_pending_age_seconds"), 0.0))
        bucket["source_count"] = _safe_int(bucket.get("source_count"), 0) + 1
        if row.get("shard"):
            bucket["shards"].add(str(row.get("shard")))
        bucket["top_sources"].append(
            {
                "source_rel": str(row.get("source_rel") or ""),
                "pending_lines": _safe_int(row.get("pending_lines"), 0),
                "oldest_pending_age_seconds": _safe_float(row.get("oldest_pending_age_seconds"), 0.0),
            }
        )
    out: list[dict[str, Any]] = []
    for bucket in grouped.values():
        sources = sorted(_as_list(bucket.get("top_sources")), key=lambda item: _safe_int(_as_dict(item).get("pending_lines"), 0), reverse=True)
        out.append(
            {
                "sleeve": str(bucket.get("sleeve") or "unknown"),
                "pending_lines": _safe_int(bucket.get("pending_lines"), 0),
                "max_age_seconds": round(_safe_float(bucket.get("max_age_seconds"), 0.0), 3),
                "source_count": _safe_int(bucket.get("source_count"), 0),
                "shards": sorted(str(item) for item in bucket.get("shards", set())),
                "top_sources": sources[:3],
            }
        )
    return sorted(out, key=lambda item: (_safe_int(item.get("pending_lines"), 0), _safe_float(item.get("max_age_seconds"), 0.0)), reverse=True)


def _sleeve_pump_fairness(accelerator: dict[str, Any], sleeve_rows: list[dict[str, Any]]) -> dict[str, Any]:
    sleeve_pump = _as_dict(accelerator.get("sleeve_pump_contract"))
    max_slots = max(_safe_int(sleeve_pump.get("max_active_sleeves_per_wave"), _safe_int(os.getenv("BACKLOG_SLEEVE_PUMP_MAX_ACTIVE_SLEEVES"), 8)), 1)
    selected_slots = max(_safe_int(sleeve_pump.get("selected_active_sleeve_slots"), min(max_slots, 7)), 1)
    active = sleeve_rows[:selected_slots]
    standby = sleeve_rows[selected_slots:max_slots]
    total_pending = sum(_safe_int(row.get("pending_lines"), 0) for row in sleeve_rows)
    largest_share = max([_safe_int(row.get("pending_lines"), 0) / max(total_pending, 1) for row in sleeve_rows] or [0.0])
    score = 100
    blockers: list[str] = []
    if not bool(sleeve_pump.get("enabled", False)):
        score -= 40
        blockers.append("sleeve_pump_disabled")
    if largest_share > 0.55 and len(sleeve_rows) > 1:
        score -= 15
        blockers.append("one_sleeve_dominates_pump_queue")
    if selected_slots < min(len(sleeve_rows), max_slots):
        score -= 10
        blockers.append("not_all_hot_sleeves_have_active_slots")
    return {
        "bot": "sleeve_pump_fairness_bot",
        "status": "ready" if score >= 85 else "needs_work",
        "score": max(score, 0),
        "active_sleeve_slots": selected_slots,
        "max_active_sleeves_per_wave": max_slots,
        "active_sleeves": [row["sleeve"] for row in active],
        "standby_sleeves": [row["sleeve"] for row in standby],
        "largest_pending_share": round(largest_share, 4),
        "blockers": blockers,
        "policy": "hot_sleeves_first_then_round_robin_without_parallel_sqlite_writers",
    }


def _writer_throughput_sentinel(writer: dict[str, Any], writer_intel: dict[str, Any]) -> dict[str, Any]:
    state = _as_dict(writer.get("writer_state_before")) or _as_dict(writer.get("writer_state_after_wait"))
    health = _as_dict(writer_intel.get("writer_health"))
    current = health or state
    rows = _safe_int(current.get("merged_rows_this_cycle"), 0)
    cycle_age = max(_safe_float(current.get("cycle_age_minutes"), _safe_float(state.get("cycle_age_minutes"), 0.0)), 0.0)
    progress_age = max(_safe_float(current.get("progress_age_minutes"), _safe_float(state.get("progress_age_minutes"), 0.0)), 0.0)
    rows_per_minute = rows / max(cycle_age, 0.001)
    lane_contract = _as_dict(current.get("shard_writer_lane_contract"))
    selected_lanes = _safe_int(lane_contract.get("selected_shard_writer_lanes"), _safe_int(current.get("shard_link_writer_lanes"), 0))
    max_lanes = _safe_int(lane_contract.get("max_shard_writer_lanes"), 0)
    active = bool(current.get("active", False) or state.get("active", False))
    blockers: list[str] = []
    if active and progress_age >= 20:
        blockers.append("writer_progress_age_high")
    if active and rows_per_minute < 1000 and cycle_age >= 5:
        blockers.append("writer_rows_per_minute_low")
    return {
        "bot": "writer_throughput_sentinel",
        "status": "needs_work" if blockers else "ready",
        "writer_active": active,
        "current_step": str(current.get("current_step") or state.get("current_step") or ""),
        "merged_rows_this_cycle": rows,
        "cycle_age_minutes": round(cycle_age, 3),
        "progress_age_minutes": round(progress_age, 3),
        "rows_per_minute": round(rows_per_minute, 3),
        "selected_shard_writer_lanes": selected_lanes,
        "max_shard_writer_lanes": max_lanes,
        "blockers": blockers,
        "next_command": ["./scripts/ops/opsctl.sh", "writer-cycle-coordinator", "--json"],
    }


def _pump_regression_guard(accelerator: dict[str, Any], writer_intel: dict[str, Any]) -> dict[str, Any]:
    host = _as_dict(accelerator.get("host_lane_contract"))
    desired_workers = max(_safe_int(host.get("selected_p_core_preprocess_workers"), _safe_int(os.getenv("BACKLOG_PCORE_PREPROCESS_WORKERS"), 7)), 1)
    desired_max_lanes = max(_safe_int(os.getenv("SQL_LINK_SERVICE_MAX_SHARD_WRITER_LANES"), 8), desired_workers)
    health = _as_dict(writer_intel.get("writer_health"))
    lane_contract = _as_dict(health.get("shard_writer_lane_contract"))
    selected = _safe_int(lane_contract.get("selected_shard_writer_lanes"), _safe_int(health.get("shard_link_writer_lanes"), 0))
    max_lanes = _safe_int(lane_contract.get("max_shard_writer_lanes"), 0)
    active = bool(health.get("active", False))
    old_cycle = bool(active and selected > 0 and selected < desired_workers)
    regressed = bool(not active and selected > 0 and selected < desired_workers)
    status = "pending_next_cycle" if old_cycle else "blocked" if regressed else "ready"
    return {
        "bot": "pump_regression_guard",
        "status": status,
        "desired_p_core_workers": desired_workers,
        "desired_max_shard_lanes": desired_max_lanes,
        "observed_selected_shard_lanes": selected,
        "observed_max_shard_lanes": max_lanes,
        "writer_active": active,
        "regression_detected": regressed,
        "old_cycle_pending_new_contract": old_cycle,
        "next_command": ["./scripts/ops/opsctl.sh", "writer-cycle-coordinator", "--json"],
        "policy": "alert_when_next_idle_writer_cycle_does_not_pick_up_full_p_core_pump_contract",
    }


def _stale_source_hunter(rows: list[dict[str, Any]], accelerator: dict[str, Any]) -> dict[str, Any]:
    storage_accel = _as_dict(accelerator.get("storage_accelerator_contract"))
    trigger = _as_dict(storage_accel.get("trigger_context"))
    stale = [row for row in rows if _safe_int(row.get("pending_lines"), 0) > 0 or _safe_float(row.get("oldest_pending_age_seconds"), 0.0) > 240.0]
    top = sorted(stale, key=lambda item: (_safe_float(item.get("oldest_pending_age_seconds"), 0.0), _safe_int(item.get("pending_lines"), 0)), reverse=True)[:8]
    return {
        "bot": "stale_source_hunter",
        "status": "needs_work" if top else "ready",
        "stale_source_count": len(stale),
        "sparse_huge_jsonl_active": bool(trigger.get("sparse_active", False)),
        "sparse_pending_bytes": _safe_int(trigger.get("sparse_pending_bytes"), 0),
        "top_sources": top,
        "next_command": ["./scripts/ops/opsctl.sh", "writer-cycle-coordinator", "--apply", "--json"] if top else [],
        "policy": "pin_oldest_exact_sources_before_broad_sleeve_pump_rotation",
    }


def _sleeve_intake_governor(sleeve_rows: list[dict[str, Any]], accelerator: dict[str, Any]) -> dict[str, Any]:
    storage = _as_dict(accelerator.get("storage_contract"))
    green = bool(storage.get("green", False))
    total_pending = sum(_safe_int(row.get("pending_lines"), 0) for row in sleeve_rows)
    max_age = max([_safe_float(row.get("max_age_seconds"), 0.0) for row in sleeve_rows] or [0.0])
    if not green or total_pending > 1_000_000 or max_age > 900:
        active_ratio = 0.16
        mode = "protect_writer"
    elif total_pending > 250_000:
        active_ratio = 0.22
        mode = "bounded_catchup"
    else:
        active_ratio = 0.30
        mode = "normal"
    sleeve_caps = []
    for row in sleeve_rows[:10]:
        pending = _safe_int(row.get("pending_lines"), 0)
        cap = active_ratio
        if pending > 750_000:
            cap = min(cap, 0.12)
        elif pending > 250_000:
            cap = min(cap, 0.16)
        sleeve_caps.append({"sleeve": row.get("sleeve"), "pending_lines": pending, "max_active_ratio": round(cap, 2)})
    return {
        "bot": "sleeve_intake_governor",
        "status": "ready",
        "mode": mode,
        "global_max_active_ratio": round(active_ratio, 2),
        "sleeve_caps": sleeve_caps,
        "control_env": {
            "SLEEVE_INTAKE_GOVERNOR_ENABLED": "1",
            "BOT_COLLECTION_DUTY_CYCLE_ENABLED": "1",
            "BOT_COLLECTION_DUTY_CYCLE_MAX_ACTIVE_RATIO": f"{active_ratio:.2f}",
        },
        "policy": "throttle_noisy_sleeves_until_writer_age_and_pending_lines_stay_green",
    }


def _wal_sqlite_steward(accelerator: dict[str, Any]) -> dict[str, Any]:
    writer_tuning = _as_dict(accelerator.get("single_writer_tuning_contract"))
    wal = _as_dict(writer_tuning.get("wal_checkpoint"))
    memory = _as_dict(writer_tuning.get("sqlite_memory"))
    blockers: list[str] = []
    if _safe_int(writer_tuning.get("hot_batch_size"), 0) < 240000:
        blockers.append("hot_batch_below_backlog_pump_floor")
    if _safe_int(writer_tuning.get("sqlite_timeout_seconds"), 0) < 300:
        blockers.append("sqlite_timeout_too_short_for_heavy_writer")
    if not bool(wal.get("enabled", True)):
        blockers.append("wal_checkpoint_disabled")
    if _safe_int(memory.get("cache_size_kb"), 0) < 32768:
        blockers.append("sqlite_cache_too_small")
    return {
        "bot": "wal_sqlite_steward",
        "status": "needs_work" if blockers else "ready",
        "blockers": blockers,
        "writer_tuning": writer_tuning,
        "policy": "keep_single_writer_fast_with_cache_mmap_wal_and_lock_patience_not_parallel_writes",
    }


def _active_issue_ids(accelerator: dict[str, Any], storage_control: dict[str, Any]) -> list[str]:
    storage_accel = _as_dict(accelerator.get("storage_accelerator_contract"))
    trigger = _as_dict(storage_accel.get("trigger_context"))
    relief = _as_dict(storage_control.get("backlog_relief_contract"))
    issue_ids = [
        *[str(item) for item in _as_list(trigger.get("active_issue_ids")) if str(item).strip()],
        *[str(item) for item in _as_list(relief.get("active_issue_ids")) if str(item).strip()],
    ]
    return ordered_unique(issue_ids)


def _preferred_accelerator_contract(accelerator: dict[str, Any], storage_control: dict[str, Any]) -> dict[str, Any]:
    relief = _as_dict(storage_control.get("backlog_relief_contract"))
    control_contract = _as_dict(relief.get("accelerator_contract"))
    direct_contract = _as_dict(accelerator.get("storage_accelerator_contract"))
    if control_contract.get("enabled") or _as_dict(control_contract.get("trigger_context")).get("active_issue_ids"):
        return control_contract
    return direct_contract


def _backpressure_view(storage_control: dict[str, Any]) -> dict[str, Any]:
    backpressure = _as_dict(storage_control.get("backpressure"))
    raw_live = _as_dict(backpressure.get("raw_live"))
    effective = _as_dict(backpressure.get("effective_raw_live"))
    return {
        "core_pending_lines": _safe_int(effective.get("core_pending_lines"), _safe_int(backpressure.get("core_pending_lines"), 0)),
        "total_pending_lines": _safe_int(effective.get("total_pending_lines"), _safe_int(backpressure.get("total_pending_lines"), 0)),
        "oldest_pending_age_seconds": round(
            _safe_float(effective.get("oldest_pending_age_seconds"), _safe_float(backpressure.get("oldest_pending_age_seconds"), 0.0)),
            3,
        ),
        "pending_lines_threshold": _safe_int(backpressure.get("pending_lines_threshold"), 15000),
        "oldest_age_threshold_seconds": round(_safe_float(backpressure.get("oldest_age_threshold_seconds"), 240.0), 3),
        "raw_live_total_pending_lines": _safe_int(raw_live.get("total_pending_lines"), 0),
        "raw_live_oldest_pending_age_seconds": round(_safe_float(raw_live.get("oldest_pending_age_seconds"), 0.0), 3),
        "artifact_age_seconds": round(_safe_float(effective.get("artifact_age_seconds"), _safe_float(raw_live.get("artifact_age_seconds"), 0.0)), 3),
        "artifact_stale_for_overlay_reconciliation": bool(
            effective.get("artifact_stale_for_overlay_reconciliation", raw_live.get("artifact_stale_for_overlay_reconciliation", False))
        ),
        "effective_raw_live_source": str(backpressure.get("effective_raw_live_source") or effective.get("source") or ""),
    }


def _shard_hotness_router(rows: list[dict[str, Any]], storage_control: dict[str, Any], accelerator: dict[str, Any]) -> dict[str, Any]:
    active_issues = set(_active_issue_ids(accelerator, storage_control))
    shard_map: dict[str, dict[str, Any]] = {}
    for row in rows:
        shard = str(row.get("shard") or row.get("sleeve") or "unknown")
        bucket = shard_map.setdefault(
            shard,
            {
                "shard": shard,
                "pending_lines": 0,
                "max_age_seconds": 0.0,
                "source_count": 0,
                "pressure_score": 0.0,
                "top_sources": [],
            },
        )
        pending = _safe_int(row.get("pending_lines"), 0)
        age = _safe_float(row.get("oldest_pending_age_seconds"), 0.0)
        score = float(pending) + max(age - 120.0, 0.0) * 20.0
        if str(row.get("pressure_lane") or "") == "core":
            score += 5000.0
        if "stale_old_pending_work" in active_issues and age > 240.0:
            score += 10000.0
        if "sparse_huge_jsonl_files" in active_issues and pending > 0:
            score += 5000.0
        bucket["pending_lines"] = _safe_int(bucket.get("pending_lines"), 0) + pending
        bucket["max_age_seconds"] = max(_safe_float(bucket.get("max_age_seconds"), 0.0), age)
        bucket["source_count"] = _safe_int(bucket.get("source_count"), 0) + 1
        bucket["pressure_score"] = _safe_float(bucket.get("pressure_score"), 0.0) + score
        bucket["top_sources"].append(
            {
                "source_rel": str(row.get("source_rel") or ""),
                "pending_lines": pending,
                "oldest_pending_age_seconds": round(age, 3),
                "pressure_lane": str(row.get("pressure_lane") or ""),
            }
        )
    ranked = []
    for bucket in shard_map.values():
        sources = sorted(
            _as_list(bucket.get("top_sources")),
            key=lambda item: (_safe_int(_as_dict(item).get("pending_lines"), 0), _safe_float(_as_dict(item).get("oldest_pending_age_seconds"), 0.0)),
            reverse=True,
        )
        ranked.append(
            {
                "shard": str(bucket.get("shard") or "unknown"),
                "pending_lines": _safe_int(bucket.get("pending_lines"), 0),
                "max_age_seconds": round(_safe_float(bucket.get("max_age_seconds"), 0.0), 3),
                "source_count": _safe_int(bucket.get("source_count"), 0),
                "pressure_score": round(_safe_float(bucket.get("pressure_score"), 0.0), 3),
                "top_sources": sources[:4],
            }
        )
    ranked = sorted(ranked, key=lambda item: (_safe_float(item.get("pressure_score"), 0.0), _safe_int(item.get("pending_lines"), 0)), reverse=True)
    focused_sources = []
    for shard in ranked[:4]:
        focused_sources.extend(_as_list(shard.get("top_sources"))[:2])
    hot_shards = [str(item.get("shard") or "") for item in ranked[:6] if str(item.get("shard") or "").strip()]
    return {
        "bot": "shard_hotness_router_bot",
        "status": "ready",
        "ranked_shards": ranked[:8],
        "focused_sources": focused_sources[:8],
        "active_issue_ids": sorted(active_issues),
        "control_env": {
            "BACKLOG_SHARD_HOTNESS_ROUTER_ENABLED": "1",
            "SQL_LINK_SERVICE_HOT_SHARD_PRIORITY": ",".join(hot_shards),
            "SQL_LINK_SERVICE_PIN_HOT_SOURCES": "1" if focused_sources else "0",
            "SQL_LINK_SERVICE_RAW_LIVE_PRIORITY_BOOST": "1" if "raw_live_expansion_headroom" in active_issues else "0",
            "SQL_LINK_SERVICE_COLD_STAGE_YIELDS_TO_RAW_LIVE": "1" if "raw_live_expansion_headroom" in active_issues else "0",
        },
        "policy": "rank_hot_shards_and_sources_before_each_single_writer_catch_up_wave",
    }


def _catch_up_wave_budget_bot(accelerator: dict[str, Any], writer: dict[str, Any], storage_control: dict[str, Any]) -> dict[str, Any]:
    active_issues = set(_active_issue_ids(accelerator, storage_control))
    storage_accel = _preferred_accelerator_contract(accelerator, storage_control)
    wave = _as_dict(storage_accel.get("catch_up_wave_controller"))
    lock_open = _lock_open_enabled()
    pressure = _backpressure_view(storage_control)
    drain = _as_dict(writer.get("drain_effectiveness"))
    waves_run = _safe_int(drain.get("waves_run"), 0)
    wave_enabled = bool(wave.get("enabled", False)) or lock_open
    base_limit = max(_safe_int(wave.get("max_waves"), 1), 1)
    target_limit = base_limit
    if lock_open:
        target_limit = max(target_limit, 9)
    if {"stale_old_pending_work", "raw_live_expansion_headroom"} & active_issues:
        target_limit = max(target_limit, 3)
    if "sparse_huge_jsonl_files" in active_issues:
        target_limit = max(target_limit, 5)
    target_limit = min(max(target_limit, 1), 9 if lock_open else 6)
    seconds_per_cycle = max(_safe_int(wave.get("max_seconds_per_writer_cycle"), 30), 30)
    if lock_open:
        seconds_per_cycle = max(seconds_per_cycle, 240)
    if active_issues:
        seconds_per_cycle = max(seconds_per_cycle, 90)
    if "sparse_huge_jsonl_files" in active_issues:
        seconds_per_cycle = max(seconds_per_cycle, 120)
    blockers: list[str] = []
    if active_issues and not wave_enabled:
        blockers.append("catch_up_wave_disabled_for_active_backpressure")
    if str(drain.get("status") or "").lower() == "no_progress" and (
        _safe_int(drain.get("pending_after"), pressure["total_pending_lines"]) > pressure["pending_lines_threshold"]
        or _safe_float(drain.get("oldest_pending_age_after_seconds"), pressure["oldest_pending_age_seconds"])
        > pressure["oldest_age_threshold_seconds"]
    ):
        blockers.append("writer_effectiveness_needs_rescore_before_more_waves")
    return {
        "bot": "catch_up_wave_budget_bot",
        "status": "needs_work" if blockers else "ready",
        "active_issue_ids": sorted(active_issues),
        "wave_enabled": wave_enabled,
        "waves_run_last_cycle": waves_run,
        "recommended_wave_limit": target_limit,
        "recommended_max_seconds_per_cycle": seconds_per_cycle,
        "backpressure": pressure,
        "blockers": blockers,
        "control_env": {
            "BACKLOG_CATCH_UP_WAVE_BUDGET_ENABLED": "1",
            "BACKLOG_PCORE_DRAIN_LOCK_OPEN": "1" if lock_open else "0",
            "BACKLOG_ACCELERATOR_LOCK_OPEN": "1" if lock_open else "0",
            "BACKLOG_DRAIN_LOCK_OPEN": "1" if lock_open else "0",
            "SQL_LINK_SERVICE_CATCH_UP_WAVE": "1" if (active_issues or lock_open) else "0",
            "WRITER_CYCLE_MAX_CATCH_UP_WAVES": str(target_limit),
            "BACKLOG_CATCH_UP_WAVE_LIMIT": str(target_limit),
            "BACKLOG_ACCELERATOR_MAX_SECONDS_PER_CYCLE": str(seconds_per_cycle),
            "SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE": str(seconds_per_cycle),
            "SQL_LINK_SERVICE_SINGLE_WRITER_ONLY": "1",
        },
        "next_command": ["./scripts/ops/opsctl.sh", "writer-cycle-coordinator", "--json"] if blockers else [],
        "policy": "bounded_sequential_catch_up_waves_after_fresh_pressure_scoring",
    }


def _stale_signal_arbitrator_bot(accelerator: dict[str, Any], writer: dict[str, Any], storage_control: dict[str, Any]) -> dict[str, Any]:
    storage = _as_dict(accelerator.get("storage_contract"))
    pressure = _backpressure_view(storage_control)
    accelerator_total = _safe_int(storage.get("total_pending_lines"), 0)
    accelerator_oldest = _safe_float(storage.get("oldest_pending_age_seconds"), 0.0)
    effective_total = _safe_int(pressure.get("total_pending_lines"), 0)
    effective_oldest = _safe_float(pressure.get("oldest_pending_age_seconds"), 0.0)
    total_drift = abs(accelerator_total - effective_total) if accelerator_total and effective_total else 0
    oldest_drift = abs(accelerator_oldest - effective_oldest) if accelerator_oldest and effective_oldest else 0.0
    blockers: list[str] = []
    if bool(pressure.get("artifact_stale_for_overlay_reconciliation")) or _safe_float(pressure.get("artifact_age_seconds"), 0.0) > 900:
        blockers.append("raw_live_snapshot_stale_for_overlay_reconciliation")
    if total_drift > max(2000, min(accelerator_total, effective_total) * 0.5):
        blockers.append("accelerator_storage_total_drift")
    if oldest_drift > 180:
        blockers.append("accelerator_storage_age_drift")
    drain = _as_dict(writer.get("drain_effectiveness"))
    if str(drain.get("status") or "").lower() == "no_progress" and pressure["oldest_pending_age_seconds"] > pressure["oldest_age_threshold_seconds"]:
        blockers.append("writer_no_progress_needs_fresh_storage_arbitration")
    return {
        "bot": "stale_signal_arbitrator_bot",
        "status": "needs_work" if blockers else "ready",
        "accelerator_storage_total_pending_lines": accelerator_total,
        "effective_storage_total_pending_lines": effective_total,
        "total_pending_drift": total_drift,
        "accelerator_oldest_pending_age_seconds": round(accelerator_oldest, 3),
        "effective_oldest_pending_age_seconds": round(effective_oldest, 3),
        "oldest_age_drift_seconds": round(oldest_drift, 3),
        "artifact_age_seconds": pressure["artifact_age_seconds"],
        "blockers": blockers,
        "control_env": {
            "BACKLOG_STALE_SIGNAL_ARBITRATOR_ENABLED": "1",
            "BACKLOG_PUMP_REQUIRE_FRESH_STORAGE_SNAPSHOT": "1" if blockers else "0",
            "SQL_LINK_SERVICE_RECHECK_BACKPRESSURE_BEFORE_WAVE": "1",
        },
        "next_commands": [
            ["./scripts/ops/opsctl.sh", "ingestion-storage-control", "--json"],
            ["./scripts/ops/opsctl.sh", "backlog-pcore-accelerator", "--apply", "--json"],
            ["./scripts/ops/opsctl.sh", "writer-cycle-coordinator", "--json"],
        ]
        if blockers
        else [],
        "policy": "refresh_and_reconcile_storage_signals_before_speeding_up_drain_waves",
    }


def _writer_lane_preflight_bot(accelerator: dict[str, Any], writer: dict[str, Any], writer_intel: dict[str, Any], storage_control: dict[str, Any]) -> dict[str, Any]:
    host = _as_dict(accelerator.get("host_lane_contract"))
    storage_accel = _preferred_accelerator_contract(accelerator, storage_control)
    relief = _as_dict(storage_control.get("backlog_relief_contract"))
    pcore = _as_dict(relief.get("p_core_backlog_allocation_contract"))
    desired_workers = max(
        _safe_int(pcore.get("shard_link_writer_lanes"), 0),
        _safe_int(storage_accel.get("p_core_preprocess_workers"), 0),
        _safe_int(host.get("selected_p_core_preprocess_workers"), 0),
        _safe_int(os.getenv("BACKLOG_PCORE_PREPROCESS_WORKERS"), 0),
        1,
    )
    desired_max_lanes = max(
        _safe_int(pcore.get("max_shard_link_writer_lanes"), 0),
        _safe_int(storage_accel.get("max_shard_writer_lanes"), 0),
        _safe_int(os.getenv("SQL_LINK_SERVICE_MAX_SHARD_WRITER_LANES"), 0),
        desired_workers,
    )
    health = _as_dict(writer_intel.get("writer_health"))
    state = _as_dict(writer.get("writer_state_after_wait")) or _as_dict(writer.get("writer_state_before"))
    current = health or state
    lane_contract = _as_dict(current.get("shard_writer_lane_contract"))
    selected = _safe_int(lane_contract.get("selected_shard_writer_lanes"), _safe_int(current.get("shard_link_writer_lanes"), 0))
    max_lanes = _safe_int(lane_contract.get("max_shard_writer_lanes"), 0)
    active = bool(current.get("active", False) or state.get("active", False))
    old_cycle = bool(active and selected > 0 and selected < desired_workers)
    blockers: list[str] = []
    if selected < desired_workers:
        blockers.append("selected_writer_lanes_below_current_pcore_contract")
    if max_lanes and max_lanes < desired_max_lanes:
        blockers.append("max_writer_lanes_below_current_pcore_contract")
    if active and blockers:
        status = "pending_next_cycle"
    elif blockers:
        status = "needs_work"
    else:
        status = "ready"
    return {
        "bot": "writer_lane_preflight_bot",
        "status": status,
        "writer_active": active,
        "old_cycle_pending_new_contract": old_cycle,
        "desired_shard_writer_lanes": desired_workers,
        "desired_max_shard_writer_lanes": desired_max_lanes,
        "observed_selected_shard_lanes": selected,
        "observed_max_shard_lanes": max_lanes,
        "blockers": blockers,
        "control_env": {
            "WRITER_LANE_PREFLIGHT_ENABLED": "1",
            "BACKLOG_DRAIN_SINGLE_WRITER_ONLY": "1",
            "SQL_LINK_SERVICE_SINGLE_WRITER_ONLY": "1",
            "SQL_LINK_SERVICE_PREPROCESS_WORKERS": str(desired_workers),
            "SQL_LINK_SERVICE_SHARD_WRITER_LANES": str(desired_workers),
            "SQL_LINK_SERVICE_MAX_SHARD_WRITER_LANES": str(desired_max_lanes),
            "SQL_LINK_CHILD_WRITER_CPU_POLICY": "performance_core_primary",
            "SQL_LINK_WRITER_BACKGROUND_POLICY": "0",
        },
        "next_command": ["./scripts/ops/opsctl.sh", "writer-cycle-coordinator", "--json"] if blockers else [],
        "policy": "preflight_next_writer_cycle_for_full_current_pcore_preprocess_lane_contract",
    }


def _env_lines(payload: dict[str, Any]) -> list[str]:
    fairness = _as_dict(_as_dict(payload.get("bots")).get("sleeve_pump_fairness_bot"))
    lock_open = _lock_open_enabled()
    env = {
        "BACKLOG_PCORE_DRAIN_LOCK_OPEN": "1" if lock_open else "0",
        "BACKLOG_ACCELERATOR_LOCK_OPEN": "1" if lock_open else "0",
        "BACKLOG_DRAIN_LOCK_OPEN": "1" if lock_open else "0",
        "BACKLOG_PUMP_INFRABOTS_ENABLED": "1",
        "BACKLOG_SLEEVE_PUMP_FAIRNESS_ENABLED": "1",
        "WRITER_THROUGHPUT_SENTINEL_ENABLED": "1",
        "BACKLOG_PUMP_REGRESSION_GUARD_ENABLED": "1",
        "BACKLOG_STALE_SOURCE_HUNTER_ENABLED": "1",
        "SLEEVE_INTAKE_GOVERNOR_ENABLED": "1",
        "SQLITE_WAL_STEWARD_ENABLED": "1",
        "BACKLOG_SHARD_HOTNESS_ROUTER_ENABLED": "1",
        "BACKLOG_CATCH_UP_WAVE_BUDGET_ENABLED": "1",
        "BACKLOG_STALE_SIGNAL_ARBITRATOR_ENABLED": "1",
        "WRITER_LANE_PREFLIGHT_ENABLED": "1",
        "BACKLOG_SLEEVE_PUMP_MAX_ACTIVE_SLEEVES": str(fairness.get("max_active_sleeves_per_wave") or os.getenv("BACKLOG_SLEEVE_PUMP_MAX_ACTIVE_SLEEVES") or "8"),
        "SQL_LINK_SERVICE_SLEEVE_PUMP_MAX_ACTIVE_SLEEVES": str(fairness.get("max_active_sleeves_per_wave") or os.getenv("SQL_LINK_SERVICE_SLEEVE_PUMP_MAX_ACTIVE_SLEEVES") or "8"),
    }
    for bot_name in (
        "sleeve_intake_governor",
        "shard_hotness_router_bot",
        "catch_up_wave_budget_bot",
        "stale_signal_arbitrator_bot",
        "writer_lane_preflight_bot",
    ):
        bot = _as_dict(_as_dict(payload.get("bots")).get(bot_name))
        env.update({str(key): str(value) for key, value in _as_dict(bot.get("control_env")).items()})
    env["BOT_PROTECTED_VOLUME_DENYLIST"] = ",".join(PROTECTED_VOLUMES)
    env["BOT_NEVER_TOUCH_VIDEO"] = "1"
    if lock_open:
        env.update(
            {
                "BACKLOG_PCORE_ALLOCATION_ACTIVE": "1",
                "BACKLOG_PCORE_PREPROCESS_WORKERS": str(max(_safe_int(os.getenv("BACKLOG_PCORE_PREPROCESS_WORKERS"), 8), 8)),
                "SQL_LINK_SERVICE_PREPROCESS_WORKERS": str(max(_safe_int(os.getenv("SQL_LINK_SERVICE_PREPROCESS_WORKERS"), 8), 8)),
                "SQL_LINK_SERVICE_SHARD_WRITER_LANES": str(max(_safe_int(os.getenv("SQL_LINK_SERVICE_SHARD_WRITER_LANES"), 8), 8)),
                "SQL_LINK_SERVICE_MAX_SHARD_WRITER_LANES": str(max(_safe_int(os.getenv("SQL_LINK_SERVICE_MAX_SHARD_WRITER_LANES"), 8), 8)),
                "SQL_LINK_SERVICE_CATCH_UP_WAVE": "1",
                "WRITER_CYCLE_MAX_CATCH_UP_WAVES": "9",
                "BACKLOG_CATCH_UP_WAVE_LIMIT": "9",
                "BACKLOG_ACCELERATOR_MAX_SECONDS_PER_CYCLE": "240",
                "SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE": "240",
                "BACKLOG_PUMP_REQUIRE_FRESH_STORAGE_SNAPSHOT": "0",
            }
        )
    return [f"{key}={shlex.quote(str(value))}" for key, value in env.items()]


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    health = project_root / "governance" / "health"
    accelerator = load_json(health / "backlog_pcore_accelerator_latest.json")
    writer = load_json(health / "writer_cycle_coordinator_latest.json")
    writer_intel = load_json(health / "writer_process_intelligence_latest.json")
    storage_control = load_json(health / "ingestion_storage_control_latest.json")
    rows = _merge_rows(_source_rows(accelerator, writer), _storage_control_source_rows(storage_control))
    sleeve_rows = _group_by_sleeve(rows)
    bots = {
        "sleeve_pump_fairness_bot": _sleeve_pump_fairness(accelerator, sleeve_rows),
        "writer_throughput_sentinel": _writer_throughput_sentinel(writer, writer_intel),
        "pump_regression_guard": _pump_regression_guard(accelerator, writer_intel),
        "stale_source_hunter": _stale_source_hunter(rows, accelerator),
        "sleeve_intake_governor": _sleeve_intake_governor(sleeve_rows, accelerator),
        "wal_sqlite_steward": _wal_sqlite_steward(accelerator),
        "shard_hotness_router_bot": _shard_hotness_router(rows, storage_control, accelerator),
        "catch_up_wave_budget_bot": _catch_up_wave_budget_bot(accelerator, writer, storage_control),
        "stale_signal_arbitrator_bot": _stale_signal_arbitrator_bot(accelerator, writer, storage_control),
        "writer_lane_preflight_bot": _writer_lane_preflight_bot(accelerator, writer, writer_intel, storage_control),
    }
    statuses = [str(_as_dict(value).get("status") or "missing") for value in bots.values()]
    if any(status == "blocked" for status in statuses):
        overall = "blocked"
    elif any(status in {"needs_work", "pending_next_cycle"} for status in statuses):
        overall = "advisory"
    else:
        overall = "ready"
    recommended = ordered_unique(
        [
            "let the active old writer finish so the next cycle can pick up the 7/8 pump contract"
            if _as_dict(bots["pump_regression_guard"]).get("old_cycle_pending_new_contract")
            else "",
            "keep sleeve pumps at 8 active slots; add intelligence and fairness, not more SQLite writers",
            "route next drain waves by shard hotness before broad sleeve rotation",
            "budget bounded catch-up waves from fresh backpressure snapshots before each writer handoff",
            "preflight writer lanes so the next cycle starts with the current P-core lane contract",
            "pin oldest sleeve sources first, then rotate hot sleeves round-robin",
            "use the WAL/SQLite steward to tune the one writer before widening intake",
            "keep /Volumes/VIDEO denied from all storage and cleanup flows",
        ]
    )
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": overall == "ready",
        "overall_status": overall,
        "mode": "backlog_pump_infrabots",
        "input_contracts": {
            "backlog_pcore_accelerator": _status(accelerator),
            "writer_cycle_coordinator": _status(writer),
            "writer_process_intelligence": _status(writer_intel),
            "ingestion_storage_control": _status(storage_control),
        },
        "sleeve_backlog_rows": sleeve_rows,
        "bots": bots,
        "integration_contract": {
            "adds_parallel_sqlite_writers": False,
            "single_sqlite_writer_only": True,
            "added_speed_infrabot_count": 4,
            "drain_speed_helpers": [
                "shard_hotness_router_bot",
                "catch_up_wave_budget_bot",
                "stale_signal_arbitrator_bot",
                "writer_lane_preflight_bot",
            ],
            "p_cores_are_primary": True,
            "e_cores_are_spillover": True,
            "max_active_sleeve_pump_slots": _safe_int(_as_dict(bots["sleeve_pump_fairness_bot"]).get("max_active_sleeves_per_wave"), 8),
            "never_touch_protected_volumes": PROTECTED_VOLUMES,
            "feeds_system_needs_intelligence": True,
            "feeds_backlog_pcore_accelerator": True,
            "feeds_writer_process_intelligence": True,
        },
        "recommended_actions": recommended,
    }


def write_outputs(payload: dict[str, Any], *, out_path: Path, override_path: Path, apply: bool = False) -> dict[str, Any]:
    write_payload(out_path, payload)
    applied = False
    if apply:
        lines = [
            "# Managed by scripts/ops/backlog_pump_infrabots.py",
            f"# updated_at_utc={payload.get('timestamp_utc')}",
            *_env_lines(payload),
            "",
        ]
        override_path.parent.mkdir(parents=True, exist_ok=True)
        override_path.write_text("\n".join(lines), encoding="utf-8")
        applied = True
    return {"out_path": str(out_path), "override_path": str(override_path), "applied": applied}


def main() -> int:
    parser = argparse.ArgumentParser(description="Coordinate pump infra bots around sleeve fairness, writer throughput, stale sources, intake, and WAL/SQLite tuning.")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--override", default=str(DEFAULT_OVERRIDE_PATH))
    args = parser.parse_args()
    payload = build_payload(PROJECT_ROOT)
    result = write_outputs(payload, out_path=Path(args.out), override_path=Path(args.override), apply=bool(args.apply))
    payload["write_result"] = result
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        statuses = {
            key: _as_dict(value).get("status")
            for key, value in _as_dict(payload.get("bots")).items()
        }
        print(
            "backlog_pump_infrabots "
            f"status={payload.get('overall_status')} "
            f"applied={result['applied']} "
            f"bots={statuses}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
