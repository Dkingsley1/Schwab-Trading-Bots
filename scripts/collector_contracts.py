#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize collector freshness contracts for daily ops.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    now_ts = datetime.now(timezone.utc).timestamp()
    rows: list[dict[str, Any]] = []
    required_failures: list[str] = []
    soft_failures: list[str] = []

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
        latest_run = ops_data_plane.latest_collector_run(project_root, collector_key=name)
        error_budget = ops_data_plane.collector_error_budget(project_root, collector_key=name)
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
                "log_schema_version": health_payload.get("log_schema_version") if isinstance(health_payload, dict) else None,
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
            }
        )

    low_quality_collectors = [row["name"] for row in rows if float(row.get("quality_score", 0.0) or 0.0) < 0.65]
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "collector_count": len(rows),
        "required_failure_count": len(required_failures),
        "soft_failure_count": len(soft_failures),
        "average_quality_score": round(
            sum(float(row.get("quality_score", 0.0) or 0.0) for row in rows) / max(len(rows), 1),
            6,
        ),
        "low_quality_collectors": low_quality_collectors,
        "required_failures": required_failures,
        "soft_failures": soft_failures,
        "rows": rows,
    }
    out = project_root / "governance" / "health" / "collector_contracts_latest.json"
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
