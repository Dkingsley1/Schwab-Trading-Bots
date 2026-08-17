#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import ops_data_plane

HEALTH_ROOT = PROJECT_ROOT / "governance" / "health"
EXTERNAL_CONTEXT_ROOT = PROJECT_ROOT / "exports" / "external_context"
EXTERNAL_FEEDS_ROOT = PROJECT_ROOT / "exports" / "external_feeds"
LOCAL_EXTERNAL_CONTEXT_ROOT = PROJECT_ROOT / "data" / "external_context"

COLLECTOR_SPECS = [
    {
        "name": "tradingeconomics_guest",
        "health_path": HEALTH_ROOT / "tradingeconomics_guest_sync_latest.json",
        "payload_path": LOCAL_EXTERNAL_CONTEXT_ROOT / "tradingeconomics_latest.json",
        "freshness_minutes": 1440,
        "required": False,
        "safe_to_degrade": True,
    },
    {
        "name": "bls_census",
        "health_path": EXTERNAL_FEEDS_ROOT / "latest_status.json",
        "payload_path": EXTERNAL_FEEDS_ROOT / "fred" / "latest.json",
        "freshness_minutes": 1440,
        "required": False,
        "safe_to_degrade": True,
    },
    {
        "name": "official_macro_context",
        "health_path": HEALTH_ROOT / "official_macro_context_sync_latest.json",
        "payload_path": EXTERNAL_CONTEXT_ROOT / "official_macro_context_latest.json",
        "freshness_minutes": 1440,
        "required": True,
        "safe_to_degrade": False,
        "min_source_coverage_ratio": 0.80,
        "max_failed_sources": 1,
    },
    {
        "name": "global_central_bank_context",
        "health_path": HEALTH_ROOT / "global_central_bank_context_sync_latest.json",
        "payload_path": EXTERNAL_CONTEXT_ROOT / "global_central_bank_context_latest.json",
        "freshness_minutes": 2880,
        "required": False,
        "safe_to_degrade": True,
        "min_source_coverage_ratio": 0.80,
        "max_failed_sources": 1,
    },
    {
        "name": "central_bank_cross_source_context",
        "health_path": HEALTH_ROOT / "central_bank_cross_source_sync_latest.json",
        "payload_path": EXTERNAL_CONTEXT_ROOT / "central_bank_cross_source_latest.json",
        "freshness_minutes": 1440,
        "required": False,
        "safe_to_degrade": True,
        "min_source_coverage_ratio": 0.60,
        "max_failed_sources": 1,
    },
    {
        "name": "decision_context_mesh",
        "health_path": HEALTH_ROOT / "decision_context_mesh_latest.json",
        "payload_path": EXTERNAL_CONTEXT_ROOT / "decision_context_mesh_latest.json",
        "freshness_minutes": 1440,
        "required": True,
        "safe_to_degrade": False,
        "min_source_coverage_ratio": 0.80,
        "max_failed_sources": 1,
        "collector_class": "decision_critical_source_context",
        "data_plane_key": "decision_context_mesh",
        "evidence_domains": ["source_verification", "training_models", "profitability_research", "risk_controls"],
        "owner_command": ["./scripts/ops/opsctl.sh", "decision-context-sync", "--json"],
    },
    {
        "name": "schwab_education_context",
        "health_path": HEALTH_ROOT / "schwab_education_context_sync_latest.json",
        "payload_path": EXTERNAL_CONTEXT_ROOT / "schwab_education_context_latest.json",
        "freshness_minutes": 720,
        "required": False,
        "safe_to_degrade": True,
    },
    {
        "name": "market_micro_context",
        "health_path": HEALTH_ROOT / "market_micro_sync_latest.json",
        "payload_path": EXTERNAL_CONTEXT_ROOT / "market_micro_latest.json",
        "freshness_minutes": 1440,
        "required": True,
        "safe_to_degrade": False,
        "min_source_coverage_ratio": 0.75,
        "max_failed_sources": 1,
    },
    {
        "name": "sec_edgar_context",
        "health_path": HEALTH_ROOT / "sec_edgar_sync_latest.json",
        "payload_path": EXTERNAL_CONTEXT_ROOT / "sec_edgar_latest.json",
        "freshness_minutes": 1440,
        "required": False,
        "safe_to_degrade": True,
    },
    {
        "name": "extended_quant_context",
        "health_path": HEALTH_ROOT / "extended_quant_context_sync_latest.json",
        "payload_path": EXTERNAL_CONTEXT_ROOT / "extended_quant_context_latest.json",
        "freshness_minutes": 1440,
        "required": False,
        "safe_to_degrade": True,
    },
    {
        "name": "options_flow_context",
        "health_path": HEALTH_ROOT / "options_flow_context_sync_latest.json",
        "payload_path": EXTERNAL_CONTEXT_ROOT / "options_flow_context_latest.json",
        "freshness_minutes": 240,
        "required": False,
        "safe_to_degrade": True,
    },
    {
        "name": "crypto_market_context",
        "health_path": HEALTH_ROOT / "crypto_market_context_sync_latest.json",
        "payload_path": EXTERNAL_CONTEXT_ROOT / "crypto_market_context_latest.json",
        "freshness_minutes": 1440,
        "required": True,
        "safe_to_degrade": False,
    },
    {
        "name": "free_equity_reference_context",
        "health_path": HEALTH_ROOT / "free_equity_reference_context_latest.json",
        "payload_path": EXTERNAL_CONTEXT_ROOT / "free_equity_reference_context_latest.json",
        "freshness_minutes": 720,
        "required": False,
        "safe_to_degrade": True,
        "min_source_coverage_ratio": 0.50,
        "max_failed_sources": 1,
    },
    {
        "name": "market_crypto_correlation",
        "health_path": HEALTH_ROOT / "market_crypto_correlation_sync_latest.json",
        "payload_path": EXTERNAL_CONTEXT_ROOT / "market_crypto_correlation_latest.json",
        "freshness_minutes": 720,
        "required": False,
        "safe_to_degrade": True,
    },
    {
        "name": "fx_market_context",
        "health_path": HEALTH_ROOT / "fx_market_context_sync_latest.json",
        "payload_path": EXTERNAL_CONTEXT_ROOT / "fx_market_context_latest.json",
        "freshness_minutes": 360,
        "required": True,
        "safe_to_degrade": False,
    },
]

ORGANIC_EVIDENCE_COLLECTOR_SPECS = [
    {
        "name": "bond_reference_context",
        "health_path": EXTERNAL_CONTEXT_ROOT / "bond_reference_latest.json",
        "payload_path": EXTERNAL_CONTEXT_ROOT / "bond_reference_latest.json",
        "freshness_minutes": 1440,
        "required": False,
        "safe_to_degrade": True,
        "collector_class": "source_context",
        "data_plane_key": "official_macro_context",
        "evidence_domains": ["source_verification", "training_models", "profitability_research"],
        "owner_command": ["./scripts/ops/opsctl.sh", "macro-context-sync", "--json"],
    },
    {
        "name": "dividend_drip_context",
        "health_path": HEALTH_ROOT / "dividend_drip_state_sync_latest.json",
        "payload_path": EXTERNAL_CONTEXT_ROOT / "dividend_drip_state_latest.json",
        "freshness_minutes": 1440,
        "required": False,
        "safe_to_degrade": True,
        "collector_class": "source_context",
        "data_plane_key": "dividend_drip_context",
        "evidence_domains": ["training_models", "profitability_research"],
        "owner_command": ["./scripts/ops/opsctl.sh", "dividend-drip-sync", "--json"],
    },
    {
        "name": "macro_cross_asset_context",
        "health_path": EXTERNAL_CONTEXT_ROOT / "macro_cross_asset_latest.json",
        "payload_path": EXTERNAL_CONTEXT_ROOT / "macro_cross_asset_latest.json",
        "freshness_minutes": 1440,
        "required": False,
        "safe_to_degrade": True,
        "collector_class": "source_context",
        "data_plane_key": "bls_census",
        "evidence_domains": ["source_verification", "training_models", "profitability_research"],
        "owner_command": ["./scripts/ops/opsctl.sh", "macro-context-sync", "--json"],
    },
    {
        "name": "central_bank_liquidity_context",
        "health_path": HEALTH_ROOT / "official_macro_context_sync_latest.json",
        "payload_path": EXTERNAL_CONTEXT_ROOT / "central_bank_liquidity_latest.json",
        "freshness_minutes": 1440,
        "required": True,
        "safe_to_degrade": False,
        "collector_class": "decision_critical_source_context",
        "data_plane_key": "official_macro_context",
        "evidence_domains": ["source_verification", "training_models", "profitability_research", "risk_controls"],
        "owner_command": ["./scripts/ops/opsctl.sh", "macro-context-sync", "--json"],
    },
    {
        "name": "public_policy_context",
        "health_path": HEALTH_ROOT / "public_policy_context_sync_latest.json",
        "payload_path": EXTERNAL_CONTEXT_ROOT / "public_policy_context_latest.json",
        "freshness_minutes": 1440,
        "required": False,
        "safe_to_degrade": True,
        "collector_class": "source_context",
        "data_plane_key": "public_policy_context",
        "evidence_domains": ["source_verification", "training_models", "profitability_research"],
        "owner_command": ["./scripts/ops/opsctl.sh", "public-policy-sync", "--json"],
    },
    {
        "name": "schwab_symbol_news",
        "health_path": HEALTH_ROOT / "schwab_symbol_news_latest.json",
        "payload_path": EXTERNAL_CONTEXT_ROOT / "schwab_symbol_news_latest.json",
        "freshness_minutes": 720,
        "required": False,
        "safe_to_degrade": True,
        "collector_class": "source_context",
        "data_plane_key": "schwab_symbol_news",
        "evidence_domains": ["source_verification", "training_models", "profitability_research"],
        "owner_command": ["./scripts/ops/opsctl.sh", "schwab-symbol-news-sync", "--json"],
    },
    {
        "name": "ticker_news_context",
        "health_path": HEALTH_ROOT / "ticker_news_context_latest.json",
        "payload_path": EXTERNAL_CONTEXT_ROOT / "ticker_news_context_latest.json",
        "freshness_minutes": 720,
        "required": False,
        "safe_to_degrade": True,
        "collector_class": "source_context",
        "data_plane_key": "ticker_news_context",
        "evidence_domains": ["source_verification", "training_models", "profitability_research"],
        "owner_command": ["./scripts/ops/opsctl.sh", "ticker-news-sync", "--json"],
    },
    {
        "name": "point_in_time_event_store",
        "health_path": HEALTH_ROOT / "point_in_time_event_store_latest.json",
        "payload_path": HEALTH_ROOT / "point_in_time_event_store_latest.json",
        "freshness_minutes": 30,
        "required": False,
        "safe_to_degrade": True,
        "collector_class": "evidence_accrual",
        "data_plane_key": "point_in_time_event_store",
        "evidence_domains": ["training_models", "promotion_release"],
        "organic_minimums": {"event_count": 100},
        "owner_command": ["./scripts/ops/opsctl.sh", "point-in-time-event-store", "--json"],
    },
    {
        "name": "feature_store_lineage",
        "health_path": PROJECT_ROOT / "governance" / "feature_store" / "latest.json",
        "payload_path": PROJECT_ROOT / "governance" / "feature_store" / "latest.json",
        "freshness_minutes": 60,
        "required": False,
        "safe_to_degrade": True,
        "collector_class": "evidence_accrual",
        "data_plane_key": "feature_store_lineage",
        "evidence_domains": ["training_models", "promotion_release"],
        "organic_truthy_paths": ["strict_ok"],
        "organic_ratio_targets": {"point_in_time_contract.snapshot_coverage_ratio": 0.75},
        "owner_command": ["./scripts/ops/opsctl.sh", "feature-store", "--json"],
    },
    {
        "name": "candidate_fill_replay",
        "health_path": HEALTH_ROOT / "market_replay_fill_capture_latest.json",
        "payload_path": HEALTH_ROOT / "market_replay_fill_capture_latest.json",
        "freshness_minutes": 10,
        "required": False,
        "safe_to_degrade": True,
        "collector_class": "evidence_accrual",
        "data_plane_key": "candidate_fill_replay",
        "evidence_domains": ["profitability_research", "promotion_release"],
        "organic_minimums": {"capture_count": 100},
        "owner_command": ["./scripts/ops/opsctl.sh", "market-replay-fill-capture", "--apply", "--json"],
    },
]

COLLECTOR_SPECS += ORGANIC_EVIDENCE_COLLECTOR_SPECS

_PAYLOAD_META_KEYS = {
    "timestamp_utc",
    "generated_utc",
    "updated_at_utc",
    "updated_at",
    "created_at",
    "schema_version",
    "log_schema_version",
    "ok",
    "errors",
    "warnings",
    "status",
    "sources",
    "health",
}
_COUNT_SIGNAL_KEYS = (
    "rows",
    "row_count",
    "record_count",
    "records",
    "item_count",
    "items",
    "series_count",
    "pair_count",
    "symbol_count",
    "headline_count",
    "article_count",
    "file_count",
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_spec_path(project_root: Path, raw_path: Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        return project_root / path
    try:
        rel = path.relative_to(PROJECT_ROOT)
        return project_root / rel
    except Exception:
        return path


def _sha256_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_timestamp(path: Path, payload: dict[str, Any]) -> float:
    for key in ("timestamp_utc", "generated_utc", "updated_at_utc", "updated_at", "created_at"):
        raw = str(payload.get(key) or "").strip()
        if not raw:
            continue
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()
        except Exception:
            continue
    try:
        return float(path.stat().st_mtime)
    except Exception:
        return 0.0


def _observed_timestamp(*, health_path: Path, health_payload: dict[str, Any], payload_path: Path, payload_body: dict[str, Any]) -> float:
    return max(
        _payload_timestamp(health_path, health_payload) if health_payload else 0.0,
        _payload_timestamp(payload_path, payload_body) if payload_body else 0.0,
    )


def _status_ok(name: str, payload: dict[str, Any]) -> bool:
    if not payload:
        return False
    if name == "bls_census":
        return all(bool((payload.get(section) or {}).get("ok", False)) for section in ("bls", "census", "fred", "bea"))
    if "ok" in payload:
        return bool(payload.get("ok", False))
    return True


def _partial_data(spec: dict[str, Any], name: str, payload: dict[str, Any]) -> bool:
    if not payload:
        return False
    if name == "bls_census":
        return not _status_ok(name, payload)
    if isinstance(payload.get("errors"), list) and payload.get("errors"):
        return True
    source_status = payload.get("sources")
    if isinstance(source_status, dict):
        participating_rows = [
            row
            for row in source_status.values()
            if isinstance(row, dict) and bool(row.get("contract_participates", True))
        ]
        ok_count = sum(1 for row in participating_rows if bool(row.get("ok", False)))
        total_count = len(participating_rows)
        failed_count = max(total_count - ok_count, 0)
        min_ratio = float(spec.get("min_source_coverage_ratio", 1.0) or 1.0)
        max_failed = int(spec.get("max_failed_sources", 0) or 0)
        if total_count <= 0:
            return False
        coverage_ratio = ok_count / max(total_count, 1)
        return bool(coverage_ratio < min_ratio or failed_count > max_failed)
    return False


def _clamp01(value: float) -> float:
    return max(min(float(value), 1.0), 0.0)


def _source_status_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    source_status = payload.get("sources")
    if not isinstance(source_status, dict) or not source_status:
        return {"total": 0, "ok": 0, "coverage_ratio": 1.0}

    participating_rows = [
        row
        for row in source_status.values()
        if isinstance(row, dict) and bool(row.get("contract_participates", True))
    ]
    total = 0
    ok_count = 0
    for row in participating_rows:
        total += 1
        if bool(row.get("ok", False)):
            ok_count += 1
    if total <= 0:
        return {"total": 0, "ok": 0, "coverage_ratio": 1.0}
    return {
        "total": total,
        "ok": ok_count,
        "coverage_ratio": round(ok_count / max(total, 1), 6),
    }


def _payload_shape_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        return {
            "nonempty": False,
            "semantic_key_count": 0,
            "nonempty_child_count": 0,
            "count_signal": None,
            "payload_score": 0.0,
        }

    semantic_keys = [key for key in payload.keys() if str(key) not in _PAYLOAD_META_KEYS]
    nonempty_child_count = 0
    for value in payload.values():
        if isinstance(value, dict) and value:
            nonempty_child_count += 1
        elif isinstance(value, list) and value:
            nonempty_child_count += 1
        elif isinstance(value, str) and value.strip():
            nonempty_child_count += 1
        elif isinstance(value, (int, float)) and float(value) != 0.0:
            nonempty_child_count += 1

    count_values: list[float] = []
    for key in _COUNT_SIGNAL_KEYS:
        raw = payload.get(key)
        if raw is None:
            continue
        try:
            count_values.append(float(raw))
        except Exception:
            continue

    count_signal = None
    if count_values:
        count_signal = 1.0 if max(count_values) > 0.0 else 0.0

    payload_score = 0.2
    if semantic_keys:
        payload_score += 0.35
    if nonempty_child_count > 0:
        payload_score += 0.3
    if count_signal is None or count_signal > 0.0:
        payload_score += 0.15
    return {
        "nonempty": bool(semantic_keys or nonempty_child_count > 0),
        "semantic_key_count": len(semantic_keys),
        "nonempty_child_count": nonempty_child_count,
        "count_signal": count_signal,
        "payload_score": round(_clamp01(payload_score), 6),
    }


def _path_value(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for token in str(path or "").split("."):
        if not token or not isinstance(current, dict) or token not in current:
            return None
        current = current[token]
    return current


def _organic_readiness(
    spec: dict[str, Any],
    *,
    fresh: bool,
    health_ok: bool,
    payload_present: bool,
    payload_nonempty: bool,
    health_payload: dict[str, Any],
    payload_body: dict[str, Any],
) -> dict[str, Any]:
    organic_required = bool(spec.get("organic_required", spec.get("collector_class") in {"source_context", "evidence_accrual"}))
    if not organic_required:
        return {
            "required": False,
            "ready": True,
            "status": "not_scored",
            "progress": 1.0,
            "blockers": [],
            "minimums": {},
            "ratio_targets": {},
            "observed": {},
        }

    blockers: list[str] = []
    if not fresh:
        blockers.append("collector_stale")
    if not health_ok:
        blockers.append("collector_health_not_ok")
    if not payload_present:
        blockers.append("payload_missing")
    elif not payload_nonempty:
        blockers.append("payload_empty")

    observed: dict[str, Any] = {}
    truthy_results: list[bool] = []
    quantitative_progress: list[float] = []
    for path in spec.get("organic_truthy_paths", []) or []:
        value = _path_value(health_payload, str(path))
        if value is None:
            value = _path_value(payload_body, str(path))
        observed[str(path)] = value
        passed = bool(value)
        truthy_results.append(passed)
        if not passed:
            blockers.append(f"truthy_requirement_not_met:{path}")

    minimums = spec.get("organic_minimums") if isinstance(spec.get("organic_minimums"), dict) else {}
    for path, raw_minimum in minimums.items():
        value = _path_value(health_payload, str(path))
        if value is None:
            value = _path_value(payload_body, str(path))
        try:
            observed_value = float(value or 0.0)
            minimum = max(float(raw_minimum or 0.0), 0.0)
        except Exception:
            observed_value = 0.0
            minimum = max(float(raw_minimum or 0.0), 0.0)
        observed[str(path)] = observed_value
        passed = observed_value >= minimum
        quantitative_progress.append(1.0 if minimum <= 0.0 else _clamp01(observed_value / minimum))
        if not passed:
            blockers.append(f"minimum_not_met:{path}:{observed_value:g}/{minimum:g}")

    ratio_targets = spec.get("organic_ratio_targets") if isinstance(spec.get("organic_ratio_targets"), dict) else {}
    for path, raw_target in ratio_targets.items():
        value = _path_value(health_payload, str(path))
        if value is None:
            value = _path_value(payload_body, str(path))
        try:
            observed_value = max(float(value or 0.0), 0.0)
            target = max(float(raw_target or 0.0), 0.0)
        except Exception:
            observed_value = 0.0
            target = max(float(raw_target or 0.0), 0.0)
        observed[str(path)] = observed_value
        passed = observed_value >= target
        quantitative_progress.append(1.0 if target <= 0.0 else _clamp01(observed_value / target))
        if not passed:
            blockers.append(f"ratio_target_not_met:{path}:{observed_value:g}/{target:g}")

    evidence_accrual = str(spec.get("collector_class") or "") == "evidence_accrual"
    prerequisites_ready = bool(
        fresh
        and payload_present
        and payload_nonempty
        and (health_ok or evidence_accrual)
    )
    progress = (
        min(quantitative_progress)
        if quantitative_progress
        else (1.0 if all(truthy_results) else 0.0)
        if truthy_results
        else 1.0
    )
    if not prerequisites_ready:
        progress = 0.0
    ready = bool(not blockers)
    return {
        "required": True,
        "ready": ready,
        "status": "ready" if ready else "accumulating",
        "progress": round(progress, 6),
        "blockers": blockers,
        "minimums": {str(key): value for key, value in minimums.items()},
        "ratio_targets": {str(key): value for key, value in ratio_targets.items()},
        "observed": observed,
        "policy": "scores rise only from fresh source-backed payloads and real evidence counts",
    }


def _empty_error_budget() -> dict[str, Any]:
    return {
        "collector_key": "",
        "run_count": 0,
        "error_count": 0,
        "error_rate": 0.0,
        "error_budget_remaining": 1.0,
        "source": "data_plane_lookup_skipped",
    }


def _data_plane_context(
    project_root: Path,
    *,
    collector_key: str,
    include_data_plane: bool,
    connection: sqlite3.Connection | None = None,
    connection_attempted: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not include_data_plane:
        budget = _empty_error_budget()
        budget["collector_key"] = str(collector_key)
        return {}, budget
    if connection_attempted and connection is None:
        budget = _empty_error_budget()
        budget["collector_key"] = str(collector_key)
        budget["source"] = "shared_data_plane_connection_failed"
        return {}, budget
    try:
        latest_run = ops_data_plane.latest_collector_run(
            project_root,
            collector_key=collector_key,
            connection=connection,
        )
        error_budget = ops_data_plane.collector_error_budget(
            project_root,
            collector_key=collector_key,
            connection=connection,
        )
    except Exception:
        latest_run = {}
        error_budget = _empty_error_budget()
        error_budget["collector_key"] = str(collector_key)
        error_budget["source"] = "data_plane_lookup_failed"
    return latest_run, error_budget


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize collector freshness contracts for daily ops.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument(
        "--out-file",
        default="",
        help="Optional override for where the health payload is written.",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--include-data-plane", action="store_true", help="Enrich rows with SQLite data-plane latest-run/error-budget context.")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    now_ts = datetime.now(timezone.utc).timestamp()
    rows: list[dict[str, Any]] = []
    required_failures: list[str] = []
    soft_failures: list[str] = []
    data_plane_connection: sqlite3.Connection | None = None
    data_plane_connection_attempted = bool(args.include_data_plane)
    if args.include_data_plane:
        try:
            data_plane_path = ops_data_plane.resolve_db_path(project_root)
            if data_plane_path.exists():
                data_plane_connection = ops_data_plane.connect(
                    project_root,
                    db_path=data_plane_path,
                    quick_check=False,
                )
        except Exception:
            data_plane_connection = None

    try:
        for spec in COLLECTOR_SPECS:
            name = str(spec["name"])
            health_path = _resolve_spec_path(project_root, Path(spec["health_path"]))
            payload_path = _resolve_spec_path(project_root, Path(spec["payload_path"]))
            health_payload = _load_json(health_path)
            payload_body = _load_json(payload_path)
            health_ts = _observed_timestamp(
                health_path=health_path,
                health_payload=health_payload,
                payload_path=payload_path,
                payload_body=payload_body,
            )
            age_seconds = max(now_ts - health_ts, 0.0) if health_ts > 0.0 else float("inf")
            fresh = bool(health_ts > 0.0 and age_seconds <= (float(spec["freshness_minutes"]) * 60.0))
            ok = _status_ok(name, health_payload)
            partial = _partial_data(spec, name, health_payload)
            required = bool(spec["required"])
            safe_to_degrade = bool(spec["safe_to_degrade"])
            contract_ok = bool(fresh and (ok or safe_to_degrade))
            collector_class = str(spec.get("collector_class") or "core_context")
            data_plane_key = str(spec.get("data_plane_key") or name)
            latest_run, error_budget = _data_plane_context(
                project_root,
                collector_key=data_plane_key,
                include_data_plane=bool(args.include_data_plane),
                connection=data_plane_connection,
                connection_attempted=data_plane_connection_attempted,
            )
            payload_present = bool(payload_path.exists())
            payload_size_bytes = int(payload_path.stat().st_size) if payload_present else 0
            payload_sha256 = _sha256_file(payload_path) if payload_present else ""
            source_status = _source_status_metrics(health_payload)
            payload_shape = _payload_shape_metrics(payload_body)
            freshness_score = 0.0
            if age_seconds != float("inf") and spec["freshness_minutes"]:
                freshness_score = _clamp01(1.0 - (float(age_seconds) / (float(spec["freshness_minutes"]) * 60.0 * 2.0)))
            elif age_seconds != float("inf"):
                freshness_score = 1.0
            latest_run_score = 1.0
            if latest_run and int(latest_run.get("rc", 0) or 0) != 0 and not bool(latest_run.get("skipped", False)):
                latest_run_score = 0.0
            quality_score = (
                freshness_score * 0.35
                + (1.0 if ok else 0.0) * 0.15
                + float(source_status.get("coverage_ratio", 1.0) or 1.0) * 0.2
                + float(payload_shape.get("payload_score", 0.0) or 0.0) * 0.2
                + float(error_budget.get("error_budget_remaining", 1.0) or 1.0) * 0.05
                + latest_run_score * 0.05
            )
            if partial:
                quality_score -= 0.15
            quality_score = _clamp01(quality_score)
            organic_readiness = _organic_readiness(
                spec,
                fresh=fresh,
                health_ok=ok,
                payload_present=payload_present,
                payload_nonempty=bool(payload_shape["nonempty"]),
                health_payload=health_payload,
                payload_body=payload_body,
            )
            if not contract_ok:
                if required:
                    required_failures.append(name)
                else:
                    soft_failures.append(name)
            rows.append(
                {
                    "name": name,
                    "required": required,
                    "safe_to_degrade": safe_to_degrade,
                    "collector_class": collector_class,
                    "data_plane_key": data_plane_key,
                    "evidence_domains": [str(value) for value in spec.get("evidence_domains", []) or []],
                    "owner_command": [str(value) for value in spec.get("owner_command", []) or []],
                    "health_path": str(health_path),
                    "payload_path": str(payload_path),
                    "health_present": bool(health_payload),
                    "payload_present": payload_present,
                    "payload_size_bytes": payload_size_bytes,
                    "payload_sha256": payload_sha256,
                    "freshness_minutes": int(spec["freshness_minutes"]),
                    "age_seconds": None if age_seconds == float("inf") else round(float(age_seconds), 3),
                    "fresh": fresh,
                    "ok": ok,
                    "partial_data": partial,
                    "contract_ok": contract_ok,
                    "log_schema_version": health_payload.get("log_schema_version")
                    if isinstance(health_payload, dict)
                    else None,
                    "quality_score": round(float(quality_score), 6),
                    "source_status": source_status,
                    "payload_nonempty": bool(payload_shape["nonempty"]),
                    "payload_semantic_key_count": int(payload_shape["semantic_key_count"]),
                    "payload_nonempty_child_count": int(payload_shape["nonempty_child_count"]),
                    "payload_count_signal": payload_shape["count_signal"],
                    "intake_score_components": {
                        "freshness_score": round(float(freshness_score), 6),
                        "ok_score": 1.0 if ok else 0.0,
                        "source_coverage_score": round(float(source_status.get("coverage_ratio", 1.0) or 1.0), 6),
                        "payload_score": round(float(payload_shape.get("payload_score", 0.0) or 0.0), 6),
                        "error_budget_score": round(float(error_budget.get("error_budget_remaining", 1.0) or 1.0), 6),
                        "latest_run_score": round(float(latest_run_score), 6),
                        "partial_penalty_applied": bool(partial),
                    },
                    "latest_run": latest_run,
                    "error_budget": error_budget,
                    "organic_readiness": organic_readiness,
                    "authority_contract": {
                        "mode": "observation_only",
                        "live_execution_authority": False,
                        "automatic_promotion_authority": False,
                        "may_rewrite_historical_outcomes": False,
                    },
                }
            )
    finally:
        if data_plane_connection is not None:
            data_plane_connection.close()

    low_quality_collectors = [row["name"] for row in rows if float(row.get("quality_score", 0.0) or 0.0) < 0.65]
    organic_rows = [row for row in rows if bool((row.get("organic_readiness") or {}).get("required", False))]
    organic_ready_rows = [row for row in organic_rows if bool((row.get("organic_readiness") or {}).get("ready", False))]
    organic_score = round(
        100.0
        * sum(float((row.get("organic_readiness") or {}).get("progress", 0.0) or 0.0) for row in organic_rows)
        / max(len(organic_rows), 1),
        3,
    )
    collector_names = [str(row.get("name") or "") for row in rows]
    configured_expansion_names = {
        str(spec.get("name") or "") for spec in ORGANIC_EVIDENCE_COLLECTOR_SPECS if str(spec.get("name") or "")
    }
    configured_added_count = sum(1 for name in collector_names if name in configured_expansion_names)
    duplicate_names = sorted({name for name in collector_names if collector_names.count(name) > 1})
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "collector_count": len(rows),
        "base_collector_count": len(rows) - configured_added_count,
        "added_collector_count": configured_added_count,
        "required_failure_count": len(required_failures),
        "soft_failure_count": len(soft_failures),
        "average_quality_score": round(
            sum(float(row.get("quality_score", 0.0) or 0.0) for row in rows) / max(len(rows), 1),
            6,
        ),
        "low_quality_collectors": low_quality_collectors,
        "required_failures": required_failures,
        "soft_failures": soft_failures,
        "organic_readiness": {
            "status": "ready" if len(organic_ready_rows) == len(organic_rows) and organic_rows else "accumulating",
            "score": organic_score,
            "ready_collector_count": len(organic_ready_rows),
            "collector_count": len(organic_rows),
            "pending_collectors": [
                {
                    "name": str(row.get("name") or ""),
                    "evidence_domains": list(row.get("evidence_domains") or []),
                    "blockers": list((row.get("organic_readiness") or {}).get("blockers") or []),
                    "progress": float((row.get("organic_readiness") or {}).get("progress", 0.0) or 0.0),
                }
                for row in organic_rows
                if not bool((row.get("organic_readiness") or {}).get("ready", False))
            ],
            "policy": "100 requires every organically scored collector to publish fresh source-backed evidence and meet its real sample target",
        },
        "collector_expansion_contract": {
            "version": "organic_collector_expansion_v1",
            "requested_added_collectors": len(ORGANIC_EVIDENCE_COLLECTOR_SPECS),
            "configured_added_collectors": len(ORGANIC_EVIDENCE_COLLECTOR_SPECS),
            "baseline_requested_organic_collectors": 9,
            "decision_critical_source_context_collectors": sum(
                1
                for spec in ORGANIC_EVIDENCE_COLLECTOR_SPECS
                if str(spec.get("collector_class") or "") == "decision_critical_source_context"
            ),
            "duplicate_names": duplicate_names,
            "bounded_refresh": True,
            "observation_only": True,
            "live_execution_authority": False,
            "automatic_promotion_authority": False,
            "historical_outcome_rewrite_allowed": False,
        },
        "rows": rows,
    }
    out = Path(args.out_file).expanduser() if str(args.out_file or "").strip() else project_root / "governance" / "health" / "collector_contracts_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            f"collector_contracts required_failures={len(required_failures)} "
            f"soft_failures={len(soft_failures)} collectors={len(rows)}"
        )
    return 2 if required_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
