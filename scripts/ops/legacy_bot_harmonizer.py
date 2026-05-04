#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import iso_now, load_json, ordered_unique, write_payload
    PROJECT_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "master_bot_registry.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "legacy_bot_harmonizer_latest.json"
LEGACY_MIN_VERSION = 1
LEGACY_MAX_VERSION = 107
HARMONIZATION_VERSION = "legacy_v1_107_expansion_bridge_v1"
HARMONIZATION_SCOPE = "brain_refinery_v1_to_v107"

TARGET_FUNCTIONS = [
    "legacy_bot_harmonization",
    "quant_model_control",
    "global_halt_refresh",
    "runtime_throttle",
    "memory_efficiency",
    "paper_trade_lock",
]

DATA_COLLECTIONS = [
    "legacy_bot_harmonization_bridge",
    "quant_model_feature_surface",
    "fed_2026_source_plumbing_map",
    "fed_2026_stress_module_map",
    "covid_2020_pandemic_replay_trace",
    "stress_replay_confidence",
    "global_halt_pressure_reducer",
    "cpu_memory_backlog_pressure",
    "feature_confidence_matrix",
    "paper_live_slippage_gap",
    "arxiv_qfin_recent_research_intake",
    "ssrn_market_infrastructure_reference",
    "quantlib_pricing_benchmark",
]

STORAGE_TARGETS = [
    "governance/legacy_bot_harmonization",
    "governance/quant_models",
    "governance/resource",
    "governance/health",
]

CORRELATION_PEERS = [
    "system_governor_expansion",
    "macro_crisis_scenario_lab",
    "feature_quality_data_confidence",
    "model_risk_validation",
]

CORRELATION_DEPENDENCIES = [
    "cross_sleeve_correlation_matrix",
    "resource_profile",
    "label_contract_quality",
    "global_halt_pressure_reducer",
]


def _version_from_text(raw: str) -> int | None:
    match = re.search(r"brain_refinery_v(\d+)", str(raw or ""))
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _legacy_version(row: dict[str, Any]) -> int | None:
    version = _version_from_text(str(row.get("bot_id") or ""))
    if version is None or not (LEGACY_MIN_VERSION <= version <= LEGACY_MAX_VERSION):
        return None
    return version


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("sub_bots") if isinstance(payload.get("sub_bots"), list) else []
    return [row for row in rows if isinstance(row, dict)]


def _row_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        bot_id = str(row.get("bot_id") or "").strip()
        if bot_id:
            out[bot_id] = row
    return out


def _legacy_core_modules(project_root: Path) -> dict[str, str]:
    core_dir = project_root / "core"
    discovered: dict[str, str] = {}
    for path in sorted(core_dir.glob("brain_refinery_v*.py")):
        version = _version_from_text(path.name)
        if version is None or not (LEGACY_MIN_VERSION <= version <= LEGACY_MAX_VERSION):
            continue
        discovered[path.stem] = str(path.relative_to(project_root))
    return discovered


def _infer_role(bot_id: str) -> str:
    normalized = bot_id.lower()
    infra_tokens = (
        "guard",
        "sentinel",
        "monitor",
        "validator",
        "quarantine",
        "allocator",
        "risk_budget",
        "drift",
        "latency",
        "consensus",
        "execution_filter",
    )
    return "infrastructure_sub_bot" if any(token in normalized for token in infra_tokens) else "signal_sub_bot"


def _merge_list(row: dict[str, Any], key: str, values: list[str]) -> bool:
    existing = row.get(key) if isinstance(row.get(key), list) else []
    merged = ordered_unique([str(item or "") for item in existing] + values)
    if row.get(key) != merged:
        row[key] = merged
        return True
    return False


def _activation_candidate(row: dict[str, Any], *, added_from_core: bool = False) -> bool:
    if added_from_core:
        return True
    reason = str(row.get("reason") or "").strip().lower()
    promotion_reason = str(row.get("promotion_reason") or "").strip().lower()
    coverage_active = bool(row.get("coverage_candidate_active", False))
    coverage_reason = str(row.get("coverage_candidate_reason") or "").strip().lower()
    return (
        reason in {"new_runtime_candidate", "coverage_candidate_active", "legacy_core_file_discovered"}
        or promotion_reason in {"new_runtime_candidate", "coverage_candidate_active", "legacy_core_file_discovered"}
        or coverage_active
        or coverage_reason == "coverage_gap_closer"
    )


def _preserve_inactive(row: dict[str, Any]) -> bool:
    reason = str(row.get("reason") or "").strip().lower()
    promotion_reason = str(row.get("promotion_reason") or "").strip().lower()
    return reason == "no_classification_accuracy" or promotion_reason == "no_candidate_accuracy"


def _new_row_from_core(bot_id: str, core_module_path: str) -> dict[str, Any]:
    return {
        "bot_id": bot_id,
        "bot_role": _infer_role(bot_id),
        "active": True,
        "reason": "legacy_core_file_discovered",
        "promotion_reason": "legacy_core_file_discovered",
        "weight": 0.0,
        "preference_score": 0.0,
        "quality_score": 0.0,
        "test_accuracy": None,
        "candidate_test_accuracy": None,
        "candidate_quality_score": 0.0,
        "previous_best_accuracy": None,
        "no_improvement_streak": 0,
        "deleted_from_rotation": False,
        "delete_reason": "",
        "promoted": False,
        "model_path": "",
        "log_file": "",
        "candidate_log_file": "",
        "core_module_path": core_module_path,
        "lifecycle_state": "data_collection_only",
    }


def _harmonize_row(row: dict[str, Any], *, added_from_core: bool = False) -> tuple[bool, bool]:
    before = copy.deepcopy(row)
    row["legacy_harmonization_version"] = HARMONIZATION_VERSION
    row["legacy_harmonization_scope"] = HARMONIZATION_SCOPE
    row["quant_model_context_enabled"] = True
    row["stress_scenario_context_enabled"] = True
    row["global_halt_aware"] = True
    row["resource_throttle_aware"] = True
    row["paper_trade_lock_required"] = True
    row["paper_trade_lock_policy"] = "market_data_and_paper_only_until_explicit_graduation"
    row["direct_execution_allowed"] = False
    row["execution_policy_label"] = "legacy_harmonized_collection_or_existing_active_no_new_execution"
    row["retention_profile"] = str(row.get("retention_profile") or "hot_legacy_bridge_14d_warm_365d")
    row["freshness_slo_seconds"] = max(_safe_int(row.get("freshness_slo_seconds"), 900), 300)

    _merge_list(row, "target_functions", TARGET_FUNCTIONS)
    _merge_list(row, "data_intake_collections", DATA_COLLECTIONS)
    _merge_list(row, "storage_targets", STORAGE_TARGETS)
    _merge_list(row, "correlation_peer_sleeves", CORRELATION_PEERS)
    _merge_list(row, "correlation_dependencies", CORRELATION_DEPENDENCIES)

    activated = False
    if _activation_candidate(row, added_from_core=added_from_core) and not _preserve_inactive(row):
        activated = not bool(row.get("active", False))
        row.update(
            {
                "active": True,
                "lifecycle_state": "data_collection_only",
                "data_collection_active": True,
                "data_collection_started_utc": str(row.get("data_collection_started_utc") or iso_now()),
                "data_collection_observations": _safe_int(row.get("data_collection_observations"), 0),
                "data_collection_mode": "active_observer",
                "data_collection_reason": "legacy_bridge_active_observer_until_minimum_samples",
                "trading_enabled": False,
                "paper_trading_enabled": False,
                "live_trading_enabled": False,
                "allocation_enabled": False,
                "execution_enabled": False,
                "weight": 0.0,
                "preference_score": 0.0,
                "rotation_blocked": True,
                "rotation_block_reason": "legacy_bridge_data_collection_only_zero_weight",
                "training_excluded": True,
                "exclude_from_training": True,
                "training_candidate_after_threshold": True,
                "training_exclusion_reason": "legacy_bridge_collecting_observations_before_training",
                "training_exclusion_until": "minimum_data_collection_threshold_met",
                "training_threshold_policy": "eligible_when_minimum_observations_and_days_met",
                "minimum_training_observations": max(_safe_int(row.get("minimum_training_observations"), 1000), 1000),
                "minimum_data_collection_days": max(_safe_int(row.get("minimum_data_collection_days"), 7), 7),
                "promotion_blocked_until": "minimum_data_collection_threshold_met",
                "promotion_block_reason": "legacy_bridge_collection_only_no_training_yet",
            }
        )
    else:
        row["legacy_harmonization_mode"] = "metadata_only_preserve_existing_lifecycle"

    return row != before, activated


def _refresh_summary(payload: dict[str, Any]) -> None:
    rows = _rows(payload)
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    active = [row for row in rows if bool(row.get("active", False))]
    data_collection = [row for row in rows if str(row.get("lifecycle_state") or "").strip().lower() == "data_collection_only"]
    summary["total_bots"] = len(rows)
    summary["active_bots"] = len(active)
    summary["inactive_bots"] = max(len(rows) - len(active), 0)
    summary["active_signal_sub_bots"] = sum(1 for row in active if str(row.get("bot_role") or "") == "signal_sub_bot")
    summary["active_infrastructure_sub_bots"] = sum(1 for row in active if str(row.get("bot_role") or "") == "infrastructure_sub_bot")
    summary["active_options_sub_bots"] = sum(1 for row in active if str(row.get("bot_role") or "") == "options_sub_bot")
    summary["active_futures_sub_bots"] = sum(1 for row in active if str(row.get("bot_role") or "") == "futures_sub_bot")
    summary["inactive_signal_sub_bots"] = sum(1 for row in rows if not bool(row.get("active", False)) and str(row.get("bot_role") or "") == "signal_sub_bot")
    summary["inactive_infrastructure_sub_bots"] = sum(1 for row in rows if not bool(row.get("active", False)) and str(row.get("bot_role") or "") == "infrastructure_sub_bot")
    summary["data_collection_only_bots"] = len(data_collection)
    summary["training_excluded_bots"] = sum(1 for row in rows if bool(row.get("training_excluded", False)) or bool(row.get("exclude_from_training", False)))
    summary["legacy_harmonized_bots"] = sum(1 for row in rows if row.get("legacy_harmonization_version") == HARMONIZATION_VERSION)
    payload["summary"] = summary
    payload["updated_at_utc"] = iso_now()


def harmonize(
    project_root: Path = PROJECT_ROOT,
    *,
    registry_path: Path | None = None,
    apply: bool = False,
    out_path: Path | None = None,
) -> dict[str, Any]:
    registry_path = registry_path or project_root / "master_bot_registry.json"
    out_path = out_path or project_root / "governance" / "health" / "legacy_bot_harmonizer_latest.json"
    original = load_json(registry_path)
    payload = copy.deepcopy(original)
    rows = _rows(payload)
    by_id = _row_map(rows)
    core_modules = _legacy_core_modules(project_root)

    added_rows: list[str] = []
    for bot_id, rel_path in core_modules.items():
        if bot_id in by_id:
            by_id[bot_id]["core_module_path"] = str(by_id[bot_id].get("core_module_path") or rel_path)
            continue
        row = _new_row_from_core(bot_id, rel_path)
        rows.append(row)
        by_id[bot_id] = row
        added_rows.append(bot_id)

    harmonized: list[str] = []
    activated: list[str] = []
    preserved_inactive: list[str] = []
    for row in rows:
        version = _legacy_version(row)
        if version is None:
            continue
        changed, was_activated = _harmonize_row(row, added_from_core=str(row.get("bot_id") or "") in added_rows)
        if changed:
            harmonized.append(str(row.get("bot_id") or ""))
        if was_activated:
            activated.append(str(row.get("bot_id") or ""))
        if _preserve_inactive(row) and not bool(row.get("active", False)):
            preserved_inactive.append(str(row.get("bot_id") or ""))

    payload["sub_bots"] = rows
    _refresh_summary(payload)
    backup_path = ""
    if apply:
        lifecycle_dir = project_root / "governance" / "lifecycle"
        lifecycle_dir.mkdir(parents=True, exist_ok=True)
        stamp = iso_now().replace(":", "").replace("+00:00", "Z")
        backup = lifecycle_dir / f"master_bot_registry.legacy_harmonizer_backup_{stamp}.json"
        if registry_path.exists():
            backup.write_text(registry_path.read_text(encoding="utf-8"), encoding="utf-8")
            backup_path = str(backup.relative_to(project_root))
        write_payload(registry_path, payload)

    result = {
        "ok": True,
        "applied": bool(apply),
        "generated_at_utc": iso_now(),
        "harmonization_version": HARMONIZATION_VERSION,
        "harmonization_scope": HARMONIZATION_SCOPE,
        "summary": {
            "legacy_core_module_count": len(core_modules),
            "added_missing_registry_rows": len(added_rows),
            "harmonized_registry_rows": len(harmonized),
            "activated_collection_rows": len(activated),
            "preserved_inactive_rows": len(preserved_inactive),
        },
        "added_missing_registry_rows": added_rows,
        "harmonized_registry_rows": harmonized,
        "activated_collection_rows": activated,
        "preserved_inactive_rows": preserved_inactive,
        "backup_path": backup_path,
        "recommended_actions": [
            "run ./scripts/ops/opsctl.sh core-bot-materialize --json after applying so PyCharm-visible modules stay synchronized",
            "keep legacy v1-v107 bots collection-only until the minimum observation and day thresholds are met",
        ],
    }
    write_payload(out_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Harmonize legacy brain_refinery_v1-v107 bots with the expanded collection and safety metadata.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--registry", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    registry_path = Path(args.registry).resolve() if args.registry else project_root / "master_bot_registry.json"
    out_path = Path(args.out).resolve() if args.out else project_root / "governance" / "health" / "legacy_bot_harmonizer_latest.json"
    result = harmonize(project_root, registry_path=registry_path, apply=bool(args.apply), out_path=out_path)
    if args.json:
        print(json.dumps(result, ensure_ascii=True, indent=2))
    else:
        summary = result["summary"]
        print(
            "legacy_bot_harmonizer "
            f"applied={result['applied']} "
            f"harmonized={summary['harmonized_registry_rows']} "
            f"activated={summary['activated_collection_rows']} "
            f"added={summary['added_missing_registry_rows']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
