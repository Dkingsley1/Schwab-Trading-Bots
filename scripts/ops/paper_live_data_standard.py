#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
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


DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "master_bot_registry.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "paper_live_data_standard_latest.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.paper_live_data_standard_override"
DEFAULT_BACKUP_DIR = PROJECT_ROOT / "governance" / "lifecycle"
SOURCE_REGISTRY_PATH = PROJECT_ROOT / "master_bot_registry.json"
DEFAULT_CANDIDATE_REGISTRY_PATH = PROJECT_ROOT / "governance" / "health" / "paper_live_data_standard_registry_candidate_latest.json"
DEFAULT_SOURCE_WRITE_GUARD_PATH = PROJECT_ROOT / "governance" / "health" / "paper_live_data_standard_source_write_guard_latest.json"
STANDARD_VERSION = "paper_live_data_standard_v1"

PAPER_LOCK_POLICY = "market_data_and_paper_only_until_explicit_graduation"
COLLECTION_ONLY_BLOCK = "paper_live_data_standard_met"
TARGET_PAPER_BOTS = 40
MIN_PAPER_BOTS = 30
MAX_PAPER_BOTS = 50
LEGACY_COHORT = "legacy_established"
BOOTSTRAP_COHORT = "legacy_bootstrap"
PROMOTED_COHORT = "standard_promoted"
COLLECTION_COHORT = "collection_until_standard_met"
DELETED_COHORT = "deleted_preserved"
PAPER_PROFILE_DISABLED_SENTINEL = "__paper_profile_disabled_by_profitability_quarantine__"
COINBASE_PROBATIONARY_SPOT_PROFILES = ("default",)
COINBASE_PROBATIONARY_FUTURES_PROFILES = ("crypto_futures",)
COINBASE_PROBATIONARY_SPOT_TOP_N = 50
COINBASE_PROBATIONARY_FUTURES_TOP_N = 30
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


def _resolve_path(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _registry_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("sub_bots") if isinstance(payload.get("sub_bots"), list) else []
    return [row for row in rows if isinstance(row, dict)]


def _truthy(row: dict[str, Any], key: str) -> bool:
    return bool(row.get(key, False))


def _is_deleted(row: dict[str, Any]) -> bool:
    return _truthy(row, "deleted_from_rotation")


def _is_explicit_paper(row: dict[str, Any]) -> bool:
    return any(
        bool(row.get(key, False))
        for key in (
            "paper_live_data_enabled",
            "paper_trading_enabled",
            "paper_trade_enabled",
            "paper_execution_allowed",
        )
    )


def _is_legacy_established(row: dict[str, Any]) -> bool:
    cohort = str(row.get("paper_standard_cohort") or "").strip().lower()
    if cohort == PROMOTED_COHORT:
        return False
    lifecycle = str(row.get("lifecycle_state") or "").strip().lower()
    if lifecycle == "active" and not _is_deleted(row):
        return True
    if cohort == LEGACY_COHORT:
        return True
    return _is_explicit_paper(row) and str(row.get("paper_standard_status") or "").strip().lower() == "paper_live_data_enabled"


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


def _bot_version(row: dict[str, Any]) -> int | None:
    match = re.search(r"(?:^|[^A-Za-z0-9])v(?P<version>\d+)", str(row.get("bot_id") or ""))
    if not match:
        return None
    try:
        return int(match.group("version"))
    except Exception:
        return None


def _paper_score(row: dict[str, Any]) -> tuple[float, float, int, str]:
    return (
        _safe_float(row.get("test_accuracy"), 0.0),
        _safe_float(row.get("quality_score"), 0.0),
        -_safe_int(_bot_version(row), 10_000),
        str(row.get("bot_id") or ""),
    )


def _is_legacy_bootstrap_candidate(row: dict[str, Any]) -> bool:
    if _is_deleted(row) or row.get("test_accuracy") is None:
        return False
    version = _bot_version(row)
    if version is None or version > 99:
        return False
    if str(row.get("paper_standard_cohort") or "").strip().lower() == DELETED_COHORT:
        return False
    return True


def _select_paper_bootstrap_ids(rows: list[dict[str, Any]]) -> set[str]:
    candidates = [row for row in rows if _is_legacy_bootstrap_candidate(row)]
    candidates = sorted(candidates, key=_paper_score, reverse=True)
    if len(candidates) <= MAX_PAPER_BOTS:
        selected = candidates
    else:
        selected = candidates[:TARGET_PAPER_BOTS]
    return {str(row.get("bot_id") or "") for row in selected if str(row.get("bot_id") or "").strip()}


def _meets_paper_promotion_standard(row: dict[str, Any]) -> bool:
    if _is_deleted(row) or _is_legacy_established(row):
        return False
    progress = row.get("data_collection_threshold_progress")
    progress = progress if isinstance(progress, dict) else {}
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
    minimum_days = max(_safe_float(row.get("minimum_data_collection_days"), 7.0), 0.0)
    days_ready = bool(progress.get("days_ready", False)) or _safe_float(progress.get("collection_age_days"), 0.0) >= minimum_days
    observations_ready = bool(progress.get("observations_ready", False)) or observations >= minimum_observations
    training_ready = bool(row.get("data_collection_training_ready", False) or progress.get("training_ready", False))
    label_ready = bool(row.get("label_contract") or row.get("universal_label_contract"))
    quality_score = _safe_float(row.get("quality_score"), 0.0)
    test_accuracy = _safe_float(row.get("test_accuracy"), 0.0)
    quality_ready = quality_score >= 0.50 or test_accuracy >= 0.56
    blocked = bool(row.get("paper_promotion_blocked", False))
    return bool(training_ready and observations_ready and days_ready and label_ready and quality_ready and not blocked)


def _ensure_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return ordered_unique(str(item) for item in value)
    return []


def _append_unique(row: dict[str, Any], key: str, values: list[str]) -> None:
    row[key] = ordered_unique(_ensure_list(row.get(key)) + values)


def _set_collection_floor(row: dict[str, Any], *, now: str) -> None:
    row["active"] = True
    row["data_collection_active"] = True
    row.setdefault("data_collection_started_utc", now)
    row["data_collection_mode"] = str(row.get("data_collection_mode") or "active_observer")
    row["active_data_collection_standard"] = True
    row["paper_trade_lock_required"] = True
    row["paper_trade_lock_policy"] = PAPER_LOCK_POLICY
    row["direct_execution_allowed"] = False
    row["trading_enabled"] = False
    row["live_trading_enabled"] = False
    row["execution_enabled"] = False
    row["allocation_enabled"] = False
    row["paper_live_data_standard_version"] = STANDARD_VERSION
    row["paper_live_data_standard_applied_utc"] = now
    _append_unique(
        row,
        "target_functions",
        ["paper_live_data_standard", "paper_trade_lock", "data_collection_floor"],
    )


def _mark_legacy_paper(row: dict[str, Any], *, now: str) -> None:
    _set_collection_floor(row, now=now)
    row["paper_standard_cohort"] = LEGACY_COHORT
    row["paper_standard_status"] = "paper_live_data_enabled"
    row["paper_live_data_enabled"] = True
    row["paper_trading_enabled"] = True
    row["paper_trade_enabled"] = True
    row["paper_execution_allowed"] = True
    row["paper_runtime_stability_mode"] = str(row.get("paper_runtime_stability_mode") or "full_force_guarded")
    row["paper_execution_queue_policy"] = str(row.get("paper_execution_queue_policy") or "buffered_jsonl_batching")
    row["paper_live_data_source"] = "legacy_established_standard"
    row["training_excluded"] = bool(row.get("training_excluded", False))
    row["exclude_from_training"] = bool(row.get("exclude_from_training", False))
    row["rotation_blocked"] = bool(row.get("rotation_blocked", False))
    row["promotion_blocked_until"] = str(row.get("promotion_blocked_until") or "")
    row["promotion_block_reason"] = str(row.get("promotion_block_reason") or "")


def _mark_standard_promoted(row: dict[str, Any], *, now: str) -> None:
    _set_collection_floor(row, now=now)
    row["paper_standard_cohort"] = PROMOTED_COHORT
    row["paper_standard_status"] = "paper_live_data_enabled"
    row["paper_live_data_enabled"] = True
    row["paper_trading_enabled"] = True
    row["paper_trade_enabled"] = True
    row["paper_execution_allowed"] = True
    row["paper_runtime_stability_mode"] = str(row.get("paper_runtime_stability_mode") or "standard_promoted_guarded")
    row["paper_execution_queue_policy"] = str(row.get("paper_execution_queue_policy") or "buffered_jsonl_batching")
    row["paper_live_data_source"] = "data_collection_promotion_standard"
    row["training_excluded"] = False
    row["exclude_from_training"] = False
    row["rotation_blocked"] = False
    row["promotion_blocked_until"] = ""
    row["promotion_block_reason"] = ""
    row["paper_promotion_approved_utc"] = now


def _mark_legacy_bootstrap_paper(row: dict[str, Any], *, now: str) -> None:
    _set_collection_floor(row, now=now)
    prior_lifecycle = str(row.get("lifecycle_state") or "").strip()
    if prior_lifecycle and prior_lifecycle != "paper_live_data":
        row.setdefault("prior_lifecycle_state", prior_lifecycle)
    row["lifecycle_state"] = "paper_live_data"
    row["paper_standard_cohort"] = BOOTSTRAP_COHORT
    row["paper_standard_status"] = "paper_live_data_enabled"
    row["paper_live_data_enabled"] = True
    row["paper_trading_enabled"] = True
    row["paper_trade_enabled"] = True
    row["paper_execution_allowed"] = True
    row["paper_runtime_stability_mode"] = str(row.get("paper_runtime_stability_mode") or "legacy_bootstrap_guarded")
    row["paper_execution_queue_policy"] = str(row.get("paper_execution_queue_policy") or "buffered_jsonl_batching")
    row["paper_live_data_source"] = "legacy_bootstrap_30_50_standard"
    row["paper_bootstrap_reason"] = "legacy_row_with_real_test_history_selected_for_30_to_50_bot_paper_lane"
    row["live_rotation_blocked"] = True
    row["training_candidate_after_threshold"] = True
    row["promotion_blocked_until"] = ""
    row["promotion_block_reason"] = ""
    row["paper_promotion_approved_utc"] = now


def _mark_collection_only(row: dict[str, Any], *, now: str) -> None:
    prior_lifecycle = str(row.get("lifecycle_state") or "").strip()
    _set_collection_floor(row, now=now)
    if prior_lifecycle and prior_lifecycle != "data_collection_only":
        row.setdefault("prior_lifecycle_state", prior_lifecycle)
    row["lifecycle_state"] = "data_collection_only"
    row["paper_standard_cohort"] = COLLECTION_COHORT
    row["paper_standard_status"] = "collecting_only_until_standard_met"
    row["paper_live_data_enabled"] = False
    row["paper_trading_enabled"] = False
    row["paper_trade_enabled"] = False
    row["paper_execution_allowed"] = False
    row["trading_enabled"] = False
    row["live_trading_enabled"] = False
    row["execution_enabled"] = False
    row["allocation_enabled"] = False
    row["rotation_blocked"] = True
    row["rotation_block_reason"] = str(row.get("rotation_block_reason") or "paper_standard_collection_only_no_rotation")
    row["training_excluded"] = True
    row["exclude_from_training"] = True
    row["training_candidate_after_threshold"] = True
    row["training_exclusion_until"] = str(row.get("training_exclusion_until") or "minimum_data_collection_threshold_met")
    row["promotion_blocked_until"] = COLLECTION_ONLY_BLOCK
    row["promotion_block_reason"] = "collecting_live_data_until_paper_standard_met"
    row["paper_promotion_standard"] = {
        "version": STANDARD_VERSION,
        "minimum_observations": int(row.get("minimum_training_observations") or 1000),
        "minimum_collection_days": int(row.get("minimum_data_collection_days") or 7),
        "requires_point_in_time_labels": True,
        "requires_quality_gate": True,
        "requires_paper_lock": True,
        "requires_live_execution_disabled": True,
    }


def _mark_deleted(row: dict[str, Any], *, now: str) -> None:
    row["active"] = False
    row["data_collection_active"] = False
    row["paper_standard_cohort"] = DELETED_COHORT
    row["paper_standard_status"] = "deleted_preserved"
    row["paper_live_data_enabled"] = False
    row["paper_trading_enabled"] = False
    row["paper_trade_enabled"] = False
    row["paper_execution_allowed"] = False
    row["direct_execution_allowed"] = False
    row["live_trading_enabled"] = False
    row["execution_enabled"] = False
    row["paper_live_data_standard_version"] = STANDARD_VERSION
    row["paper_live_data_standard_applied_utc"] = now


def _summary_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    non_deleted = [row for row in rows if not _is_deleted(row)]
    active = [row for row in rows if bool(row.get("active", False))]
    collection_active = [row for row in non_deleted if bool(row.get("data_collection_active", False))]
    collection_only = [
        row
        for row in non_deleted
        if str(row.get("paper_standard_cohort") or "").strip().lower() == COLLECTION_COHORT
    ]
    legacy_paper = [
        row
        for row in non_deleted
        if str(row.get("paper_standard_cohort") or "").strip().lower() in {LEGACY_COHORT, BOOTSTRAP_COHORT, PROMOTED_COHORT}
        and _is_explicit_paper(row)
    ]
    legacy_bootstrap = [
        row
        for row in non_deleted
        if str(row.get("paper_standard_cohort") or "").strip().lower() == BOOTSTRAP_COHORT
        and _is_explicit_paper(row)
    ]
    standard_promoted = [
        row
        for row in non_deleted
        if str(row.get("paper_standard_cohort") or "").strip().lower() == PROMOTED_COHORT
        and _is_explicit_paper(row)
    ]
    return {
        "total_bots": len(rows),
        "non_deleted_bots": len(non_deleted),
        "active_bots": len(active),
        "inactive_bots": len(rows) - len(active),
        "deleted_from_rotation": len(rows) - len(non_deleted),
        "data_collection_active_bots": len(collection_active),
        "paper_live_data_enabled_bots": len(legacy_paper),
        "legacy_bootstrap_paper_bots": len(legacy_bootstrap),
        "standard_promoted_paper_bots": len(standard_promoted),
        "collection_until_standard_bots": len(collection_only),
        "direct_execution_allowed_bots": len([row for row in rows if bool(row.get("direct_execution_allowed", False))]),
        "live_trading_enabled_bots": len([row for row in rows if bool(row.get("live_trading_enabled", False))]),
    }


def _override_lines(payload: dict[str, Any]) -> list[str]:
    counts = payload.get("counts_after") if isinstance(payload.get("counts_after"), dict) else {}
    target = payload.get("paper_lane_target") if isinstance(payload.get("paper_lane_target"), dict) else {}
    paper_count = max(_safe_int(counts.get("paper_live_data_enabled_bots"), 0), 0)
    core_top_n = max(paper_count, TARGET_PAPER_BOTS if paper_count >= MIN_PAPER_BOTS else MIN_PAPER_BOTS)
    registry_path = Path(str(payload.get("registry_path") or DEFAULT_REGISTRY_PATH))
    project_root = registry_path.parent if registry_path.name == "master_bot_registry.json" else PROJECT_ROOT
    weak_profiles = _weak_profiles_from_profitability_controls(project_root)
    schwab_profiles = _clean_profile_csv(",".join(CLEAN_SCHWAB_RUNTIME_PROFILES), weak_profiles)
    schwab_options_profiles = _clean_profile_csv(",".join(CLEAN_SCHWAB_OPTIONS_PROFILES), weak_profiles)
    schwab_futures_profiles = _clean_profile_csv(",".join(CLEAN_SCHWAB_FUTURES_PROFILES), weak_profiles)
    coinbase_profiles = _clean_profile_csv(
        ",".join(COINBASE_PROBATIONARY_SPOT_PROFILES),
        weak_profiles,
        allow_weak_profiles=set(COINBASE_PROBATIONARY_SPOT_PROFILES),
    )
    coinbase_futures_profiles = _clean_profile_csv(
        ",".join(COINBASE_PROBATIONARY_FUTURES_PROFILES),
        weak_profiles,
        allow_weak_profiles=set(COINBASE_PROBATIONARY_FUTURES_PROFILES),
    )
    coinbase_top_n = COINBASE_PROBATIONARY_SPOT_TOP_N
    coinbase_futures_top_n = COINBASE_PROBATIONARY_FUTURES_TOP_N
    values = {
        "PAPER_LIVE_DATA_STANDARD_ENABLED": "1",
        "PAPER_LIVE_DATA_STANDARD_VERSION": STANDARD_VERSION,
        "PAPER_LIVE_DATA_STANDARD_TARGET_BOTS": str(target.get("target") or TARGET_PAPER_BOTS),
        "PAPER_LIVE_DATA_STANDARD_TARGET_MIN": str(target.get("minimum") or MIN_PAPER_BOTS),
        "PAPER_LIVE_DATA_STANDARD_TARGET_MAX": str(target.get("maximum") or MAX_PAPER_BOTS),
        "PAPER_LIVE_DATA_STANDARD_ACTUAL_BOTS": str(paper_count),
        "PAPER_LIVE_DATA_STANDARD_WITHIN_BAND": "1" if bool(target.get("within_target_band", False)) else "0",
        "PAPER_LIVE_DATA_STANDARD_SELECTION_POLICY": "explicit_registry_paper_flags_only",
        "PAPER_NEW_BOTS_REQUIRE_STANDARD": "1",
        "TOP_BOT_PAPER_TRADING_ENABLED": "1",
        "TOP_BOT_PAPER_TRADING_TOP_N": str(core_top_n),
        "TOP_BOT_PAPER_TRADING_MIN_ACC": "0.0",
        "TOP_BOT_PAPER_TRADING_PROFILES": schwab_profiles,
        "TOP_BOT_PAPER_TRADING_OPTIONS_ENABLED": "1",
        "TOP_BOT_PAPER_TRADING_OPTIONS_TOP_N": str(core_top_n),
        "TOP_BOT_PAPER_TRADING_OPTIONS_MIN_ACC": "0.0",
        "TOP_BOT_PAPER_TRADING_OPTIONS_PROFILES": schwab_options_profiles,
        "SCHWAB_TOP_BOT_PAPER_TRADING_TOP_N": str(core_top_n),
        "SCHWAB_TOP_BOT_PAPER_TRADING_MIN_ACC": "0.0",
        "SCHWAB_TOP_BOT_PAPER_TRADING_PROFILES": schwab_profiles,
        "SCHWAB_OPTIONS_TOP_BOT_PAPER_TRADING_TOP_N": str(core_top_n),
        "SCHWAB_OPTIONS_TOP_BOT_PAPER_TRADING_MIN_ACC": "0.0",
        "SCHWAB_OPTIONS_TOP_BOT_PAPER_TRADING_PROFILES": schwab_options_profiles,
        "SCHWAB_FUTURES_TOP_BOT_PAPER_TRADING_TOP_N": str(core_top_n),
        "SCHWAB_FUTURES_TOP_BOT_PAPER_TRADING_MIN_ACC": "0.0",
        "SCHWAB_FUTURES_TOP_BOT_PAPER_TRADING_PROFILES": schwab_futures_profiles,
        "COINBASE_TOP_BOT_PAPER_TRADING_TOP_N": str(coinbase_top_n),
        "COINBASE_TOP_BOT_PAPER_TRADING_MIN_ACC": "0.0",
        "COINBASE_TOP_BOT_PAPER_TRADING_PROFILES": coinbase_profiles,
        "COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_TOP_N": str(coinbase_futures_top_n),
        "COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_MIN_ACC": "0.0",
        "COINBASE_FUTURES_TOP_BOT_PAPER_TRADING_PROFILES": coinbase_futures_profiles,
        "COINBASE_PAPER_PROBATION_ENABLED": "1",
        "COINBASE_PAPER_PROBATION_REASON": "weak_profiles_allowed_for_guarded_paper_only_retest",
        "COINBASE_PAPER_PROBATIONARY_PROFILES": ",".join(
            COINBASE_PROBATIONARY_SPOT_PROFILES + COINBASE_PROBATIONARY_FUTURES_PROFILES
        ),
        "COINBASE_PAPER_PROBATION_SPOT_TOP_N": str(COINBASE_PROBATIONARY_SPOT_TOP_N),
        "COINBASE_PAPER_PROBATION_FUTURES_TOP_N": str(COINBASE_PROBATIONARY_FUTURES_TOP_N),
        "PAPER_PROFILE_DISABLED_SENTINEL": PAPER_PROFILE_DISABLED_SENTINEL,
        "PAPER_PROFITABILITY_WEAK_PROFILES": ",".join(sorted(weak_profiles)),
        "PAPER_MIRROR_ALL_ACTIVE_SUB_BOTS": "1",
        "PAPER_BROKER_BRIDGE_ENABLED": "1",
        "PAPER_BROKER_BRIDGE_MODE": "jsonl",
        "PAPER_SOAK_SPECIALIZED_ALLOWLIST_BYPASS_FANOUT": "1",
        "RUN_ALL_SLEEVES_WITH_SPECIALIZED_SLEEVES": "1",
        "RUN_ALL_SLEEVES_SPECIALIZED_PROFILE_ALLOWLIST": schwab_profiles,
        "PAPER_TRADE_LOCK": "1",
        "MARKET_DATA_ONLY": "1",
        "ALLOW_ORDER_EXECUTION": "0",
    }
    lines = [
        "# Auto-managed by scripts/ops/paper_live_data_standard.py",
        f"# Generated at {payload.get('timestamp_utc') or iso_now()}",
    ]
    for key in sorted(values):
        lines.append(f"{key}={shlex.quote(str(values[key]))}")
    return lines


def write_override(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(_override_lines(payload)) + "\n", encoding="utf-8")


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, Any]:
    path = _resolve_path(registry_path, project_root)
    registry = load_json(path)
    rows = _registry_rows(registry)
    counts_before = _summary_counts(rows)
    legacy_candidates = [row for row in rows if _is_legacy_established(row) and not _is_deleted(row)]
    inactive_non_deleted = [row for row in rows if not _is_deleted(row) and not bool(row.get("active", False))]

    projected_rows = [dict(row) for row in rows]
    bootstrap_ids = _select_paper_bootstrap_ids(projected_rows)
    now = iso_now()
    for row in projected_rows:
        if _is_deleted(row):
            _mark_deleted(row, now=now)
        elif _is_legacy_established(row):
            _mark_legacy_paper(row, now=now)
        elif str(row.get("bot_id") or "") in bootstrap_ids:
            _mark_legacy_bootstrap_paper(row, now=now)
        elif _meets_paper_promotion_standard(row):
            _mark_standard_promoted(row, now=now)
        else:
            _mark_collection_only(row, now=now)

    counts_after = _summary_counts(projected_rows)
    blockers = []
    if counts_after["direct_execution_allowed_bots"] > 0:
        blockers.append("direct_execution_allowed_remaining")
    if counts_after["live_trading_enabled_bots"] > 0:
        blockers.append("live_trading_enabled_remaining")
    if counts_after["data_collection_active_bots"] < counts_after["non_deleted_bots"]:
        blockers.append("non_deleted_data_collection_not_fully_active")

    return {
        "timestamp_utc": now,
        "schema_version": 1,
        "ok": not blockers,
        "overall_status": "ready" if not blockers else "blocked",
        "standard_version": STANDARD_VERSION,
        "registry_path": str(path),
        "counts_before": counts_before,
        "counts_after": counts_after,
        "changed_counts": {
            "activated_for_collection": max(counts_after["data_collection_active_bots"] - counts_before["data_collection_active_bots"], 0),
            "legacy_paper_enabled": counts_after["paper_live_data_enabled_bots"],
            "legacy_bootstrap_paper_enabled": counts_after["legacy_bootstrap_paper_bots"],
            "standard_promoted_paper_enabled": counts_after["standard_promoted_paper_bots"],
            "collection_only_standardized": counts_after["collection_until_standard_bots"],
            "inactive_non_deleted_before": len(inactive_non_deleted),
            "deleted_preserved": counts_after["deleted_from_rotation"],
        },
        "paper_lane_target": {
            "target": TARGET_PAPER_BOTS,
            "minimum": MIN_PAPER_BOTS,
            "maximum": MAX_PAPER_BOTS,
            "actual": counts_after["paper_live_data_enabled_bots"],
            "within_target_band": MIN_PAPER_BOTS <= counts_after["paper_live_data_enabled_bots"] <= MAX_PAPER_BOTS,
            "selection_policy": "legacy_v1_to_v99_with_real_test_history_then_standard_promotions",
        },
        "legacy_paper_cohort_sample": [str(row.get("bot_id") or "") for row in legacy_candidates[:20]],
        "legacy_bootstrap_cohort_sample": [
            str(row.get("bot_id") or "")
            for row in sorted(
                [row for row in projected_rows if str(row.get("paper_standard_cohort") or "") == BOOTSTRAP_COHORT],
                key=_paper_score,
                reverse=True,
            )[:20]
        ],
        "inactive_non_deleted_sample": [str(row.get("bot_id") or "") for row in inactive_non_deleted[:20]],
        "safety_contract": {
            "allow_order_execution": "0",
            "market_data_only": "1",
            "paper_trade_lock": "1",
            "paper_mirror_all_active_sub_bots": "1",
            "live_execution_allowed": False,
            "deleted_bots_reactivated": False,
            "policy": "every non-deleted bot collects live data; every eligible paper-live-data bot may paper trade on live data; live/direct execution remains disabled",
        },
        "standard_rules": [
            "non-deleted registry rows are active live-data collectors",
            "legacy active rows stay in the paper-live-data cohort",
            "legacy v1-v99 rows with real test history bootstrap the initial paper-live-data lane",
            "new and restored rows collect only until the paper standard is met",
            "collection-only rows promote into paper-live-data when observation, age, label, and quality gates are ready",
            "deleted_from_rotation rows stay inactive and cannot paper trade",
            "direct/live execution remains disabled for every row",
        ],
        "blockers": blockers,
        "projected_registry": {**registry, "sub_bots": projected_rows},
    }


def apply_payload(
    project_root: Path,
    payload: dict[str, Any],
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    out_path: Path = DEFAULT_OUT_PATH,
    override_path: Path = DEFAULT_OVERRIDE_PATH,
    backup_dir: Path = DEFAULT_BACKUP_DIR,
    candidate_registry_path: Path = DEFAULT_CANDIDATE_REGISTRY_PATH,
    source_write_guard_path: Path = DEFAULT_SOURCE_WRITE_GUARD_PATH,
    allow_source_registry_write: bool = False,
) -> dict[str, Any]:
    registry_out = _resolve_path(registry_path, project_root)
    health_out = _resolve_path(out_path, project_root)
    override_out = _resolve_path(override_path, project_root)
    backup_root = _resolve_path(backup_dir, project_root)
    candidate_out = _resolve_path(candidate_registry_path, project_root)
    guard_out = _resolve_path(source_write_guard_path, project_root)
    projected = payload.get("projected_registry") if isinstance(payload.get("projected_registry"), dict) else {}
    if not projected:
        raise RuntimeError("projected registry missing")

    summary = projected.get("summary") if isinstance(projected.get("summary"), dict) else {}
    counts = payload.get("counts_after") if isinstance(payload.get("counts_after"), dict) else {}
    projected["summary"] = {
        **summary,
        "active_bots": int(counts.get("active_bots", summary.get("active_bots", 0)) or 0),
        "inactive_bots": int(counts.get("inactive_bots", summary.get("inactive_bots", 0)) or 0),
        "deleted_from_rotation": int(counts.get("deleted_from_rotation", summary.get("deleted_from_rotation", 0)) or 0),
        "data_collection_active_bots": int(
            counts.get("data_collection_active_bots", summary.get("data_collection_active_bots", 0)) or 0
        ),
        "paper_live_data_enabled_bots": int(counts.get("paper_live_data_enabled_bots", 0) or 0),
        "legacy_bootstrap_paper_bots": int(counts.get("legacy_bootstrap_paper_bots", 0) or 0),
        "standard_promoted_paper_bots": int(counts.get("standard_promoted_paper_bots", 0) or 0),
        "collection_until_standard_bots": int(counts.get("collection_until_standard_bots", 0) or 0),
        "paper_live_data_standard_version": STANDARD_VERSION,
        "paper_live_data_standard_applied_utc": str(payload.get("timestamp_utc") or iso_now()),
    }

    source_write_blocked = _canonical_registry_write_blocked(registry_out, allow_source_registry_write)
    backup_path = backup_root / f"master_bot_registry.{STANDARD_VERSION}.backup.json"
    if source_write_blocked:
        candidate_out.parent.mkdir(parents=True, exist_ok=True)
        guard_out.parent.mkdir(parents=True, exist_ok=True)
        candidate_out.write_text(json.dumps(projected, ensure_ascii=True, indent=2), encoding="utf-8")
        guard_out.write_text(
            json.dumps(
                {
                    "timestamp_utc": iso_now(),
                    "ok": True,
                    "overall_status": "ready",
                    "source_write_blocked": True,
                    "source_path": str(registry_out),
                    "candidate_path": str(candidate_out),
                    "reason": "canonical_registry_requires_explicit_source_write",
                    "allow_env": "PAPER_LIVE_DATA_ALLOW_SOURCE_REGISTRY_WRITE=1",
                    "allow_cli": "scripts/ops/paper_live_data_standard.py --apply --allow-source-registry-write",
                },
                ensure_ascii=True,
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        backup_root.mkdir(parents=True, exist_ok=True)
        if registry_out.exists() and not backup_path.exists():
            backup_path.write_text(registry_out.read_text(encoding="utf-8"), encoding="utf-8")
        registry_out.write_text(json.dumps(projected, ensure_ascii=True, indent=2), encoding="utf-8")
    write_override(override_out, payload)
    payload = {key: value for key, value in payload.items() if key != "projected_registry"}
    payload["apply_result"] = {
        "applied": True,
        "registry_source_write_blocked": source_write_blocked,
        "registry_path": str(registry_out),
        "candidate_registry_path": str(candidate_out) if source_write_blocked else "",
        "backup_path": "" if source_write_blocked else str(backup_path),
        "health_path": str(health_out),
        "override_path": str(override_out),
    }
    write_payload(health_out, payload)
    payload["out_path"] = str(health_out)
    return payload


def _canonical_registry_write_blocked(registry_out: Path, allow_source_registry_write: bool) -> bool:
    if allow_source_registry_write:
        return False
    try:
        return registry_out.resolve() == SOURCE_REGISTRY_PATH.resolve()
    except Exception:
        return False


def preview_payload(
    project_root: Path,
    payload: dict[str, Any],
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    out_path: Path = DEFAULT_OUT_PATH,
    override_path: Path = DEFAULT_OVERRIDE_PATH,
) -> dict[str, Any]:
    registry_out = _resolve_path(registry_path, project_root)
    health_out = _resolve_path(out_path, project_root)
    override_out = _resolve_path(override_path, project_root)
    payload = {
        key: value
        for key, value in payload.items()
        if key != "projected_registry"
    }
    payload["apply_result"] = {
        "applied": False,
        "registry_path": str(registry_out),
        "health_path": str(health_out),
        "override_path": str(override_out),
        "mode": "preview_no_registry_change",
    }
    payload["out_path"] = str(health_out)
    write_payload(health_out, payload)
    return payload


def _print_human(payload: dict[str, Any]) -> None:
    counts = payload.get("counts_after") if isinstance(payload.get("counts_after"), dict) else {}
    print(
        "paper_live_data_standard "
        f"status={payload.get('overall_status')} "
        f"active={counts.get('active_bots')} "
        f"collection={counts.get('data_collection_active_bots')} "
        f"paper={counts.get('paper_live_data_enabled_bots')} "
        f"collection_only={counts.get('collection_until_standard_bots')}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Enforce the fleet paper-on-live-data and collection baseline standard.")
    parser.add_argument("--apply", action="store_true", help="Update the registry and write the health artifact.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH, help="Registry path.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH, help="Health artifact path.")
    parser.add_argument("--override", type=Path, default=DEFAULT_OVERRIDE_PATH, help="Runtime env override path.")
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR, help="Registry backup directory.")
    parser.add_argument(
        "--allow-source-registry-write",
        action="store_true",
        default=os.getenv("PAPER_LIVE_DATA_ALLOW_SOURCE_REGISTRY_WRITE", "0").strip() == "1",
        help="Allow this intentional operator command to update the tracked master_bot_registry.json source file.",
    )
    args = parser.parse_args(argv)

    payload = build_payload(PROJECT_ROOT, registry_path=args.registry)
    if args.apply:
        payload = apply_payload(
            PROJECT_ROOT,
            payload,
            registry_path=args.registry,
            out_path=args.out,
            override_path=args.override,
            backup_dir=args.backup_dir,
            allow_source_registry_write=args.allow_source_registry_write,
        )
    else:
        payload = preview_payload(
            PROJECT_ROOT,
            payload,
            registry_path=args.registry,
            out_path=args.out,
            override_path=args.override,
        )

    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        _print_human(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
