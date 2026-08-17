#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "data_intelligence_layer_latest.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.data_intelligence_override"
DEFAULT_SQLITE_DB = PROJECT_ROOT / "data" / "jsonl_link.sqlite3"
LAYER_VERSION = "data_intelligence_layer_v2"

ROUTE_COLUMNS = (
    "source_broker",
    "source_provider",
    "source_venue",
    "asset_class",
    "routing_lane",
    "source_quality_label",
)
ROUTE_SCHEMA_COLUMNS = {
    "source_broker": "TEXT",
    "source_provider": "TEXT",
    "source_venue": "TEXT",
    "asset_class": "TEXT",
    "routing_lane": "TEXT",
    "source_quality_label": "TEXT",
}

EXPANDED_CRYPTO_SYMBOLS = (
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "XRP-USD",
    "DOGE-USD",
    "ADA-USD",
    "AVAX-USD",
    "LINK-USD",
    "LTC-USD",
    "BCH-USD",
    "UNI-USD",
    "AAVE-USD",
    "ETC-USD",
    "ATOM-USD",
    "DOT-USD",
    "NEAR-USD",
    "OP-USD",
    "ARB-USD",
    "INJ-USD",
    "FIL-USD",
    "SUI-USD",
    "HBAR-USD",
    "MKR-USD",
    "COMP-USD",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return parsed


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "active", "enabled", "ready", "ok"}


def _status(payload: dict[str, Any], default: str = "missing") -> str:
    if not payload:
        return default
    for key in ("overall_status", "status", "state"):
        raw = payload.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip().lower()
    if payload.get("ok") is True:
        return "ready"
    if payload.get("ok") is False:
        return "degraded"
    return default


def _counter_dict(counter: Counter[str], *, limit: int = 20) -> dict[str, int]:
    return {key: int(value) for key, value in counter.most_common(max(int(limit), 1))}


def _sqlite_table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


def _sqlite_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    except Exception:
        return set()


def _repair_sql_route_schema(project_root: Path, *, db_path: Path, create_indexes: bool = False) -> dict[str, Any]:
    resolved_db = db_path if db_path.is_absolute() else project_root / db_path
    result: dict[str, Any] = {
        "ok": False,
        "overall_status": "missing",
        "db_path": str(resolved_db),
        "added_columns": [],
        "created_indexes": [],
    }
    if not resolved_db.exists():
        return {**result, "reason": "sqlite_db_missing"}
    try:
        conn = sqlite3.connect(str(resolved_db), timeout=10.0)
    except Exception as exc:
        return {**result, "overall_status": "blocked", "error": f"sqlite_open_failed:{type(exc).__name__}:{exc}"}
    try:
        if not _sqlite_table_exists(conn, "jsonl_records"):
            return {**result, "reason": "jsonl_records_table_missing"}
        columns = _sqlite_columns(conn, "jsonl_records")
        added: list[str] = []
        for column, column_type in ROUTE_SCHEMA_COLUMNS.items():
            if column in columns:
                continue
            conn.execute(f"ALTER TABLE jsonl_records ADD COLUMN {column} {column_type}")
            added.append(column)
        created_indexes = []
        if create_indexes:
            indexes = [
                ("idx_jsonl_records_source_broker", "source_broker"),
                ("idx_jsonl_records_routing_lane", "routing_lane"),
                ("idx_jsonl_records_asset_class", "asset_class"),
            ]
            for index_name, column in indexes:
                conn.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON jsonl_records({column})")
                created_indexes.append(index_name)
        conn.commit()
        return {
            **result,
            "ok": True,
            "overall_status": "ready",
            "added_columns": added,
            "created_indexes": created_indexes,
        }
    except Exception as exc:
        try:
            conn.rollback()
        except Exception:
            pass
        return {**result, "overall_status": "blocked", "error": f"sqlite_schema_repair_failed:{type(exc).__name__}:{exc}"}
    finally:
        conn.close()


def _route_family(row: dict[str, Any]) -> str:
    parts = " ".join(str(row.get(key) or "") for key in ROUTE_COLUMNS).lower()
    if "coinbase" in parts:
        return "coinbase"
    if "schwab_crypto_bridge" in parts or "schwab_crypto" in parts:
        return "schwab_crypto_bridge"
    if "schwab" in parts:
        return "schwab"
    if "crypto" in parts:
        return "crypto"
    if "fx" in parts or "forex" in parts:
        return "fx"
    if "options" in parts:
        return "options"
    if "futures" in parts:
        return "futures"
    if "equities" in parts or "equity" in parts:
        return "equities"
    return "other"


def _route_row_is_unknown(row: dict[str, Any]) -> bool:
    return bool(
        str(row.get("source_broker") or "") == "unknown"
        and str(row.get("asset_class") or "") == "unknown"
        and str(row.get("routing_lane") or "") in {"unknown", "unclassified"}
    )


def _merge_route_rows(*row_sets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], dict[str, Any]] = {}
    for rows in row_sets:
        for row in rows:
            key = tuple(str(row.get(column) or "unknown") for column in ROUTE_COLUMNS)
            count = _safe_int(row.get("row_count"), 0)
            existing = grouped.get(key)
            if existing is None:
                merged = {column: key[idx] for idx, column in enumerate(ROUTE_COLUMNS)}
                merged["row_count"] = count
                merged["family"] = _route_family(merged)
                grouped[key] = merged
            else:
                existing["row_count"] = max(_safe_int(existing.get("row_count"), 0), count)
    return sorted(grouped.values(), key=lambda item: _safe_int(item.get("row_count"), 0), reverse=True)[:500]


def _route_quality_score(label: str) -> float:
    token = str(label or "").strip().lower()
    if token in {"broker_native", "exchange_native"}:
        return 0.95
    if token in {"multi_source_context", "broker_bridge"}:
        return 0.82
    if token in {"public_context"}:
        return 0.65
    if token in {"synthetic_or_simulated"}:
        return 0.50
    if token and token != "unknown":
        return 0.60
    return 0.35


def _pick_route_text(obj: dict[str, Any], route: dict[str, Any], metadata: dict[str, Any], *keys: str) -> str:
    for key in keys:
        for bucket in (obj, route, metadata):
            value = bucket.get(key) if isinstance(bucket, dict) else None
            text = str(value or "").strip()
            if text:
                return text
        return ""


def _route_hint_text(obj: dict[str, Any], route: dict[str, Any], metadata: dict[str, Any], *, source_rel: str) -> tuple[str, str]:
    path_hints = [
        source_rel,
        obj.get("source_path"),
        obj.get("target_path"),
        obj.get("file_path"),
        obj.get("path"),
        obj.get("decision_path"),
        metadata.get("source_path"),
        metadata.get("target_path"),
        metadata.get("file_path"),
        route.get("source_path"),
        route.get("route_key"),
    ]
    context_hints = [
        obj.get("event"),
        obj.get("strategy"),
        obj.get("profile"),
        obj.get("shadow_profile"),
        obj.get("domain"),
        obj.get("shadow_domain"),
        obj.get("source_stream"),
        obj.get("source_partition_key"),
        route.get("channel"),
        route.get("profile"),
        route.get("domain"),
        metadata.get("event"),
        metadata.get("strategy"),
        metadata.get("source_stream"),
    ]
    path_text = " ".join(str(item or "") for item in path_hints if str(item or "").strip()).lower()
    context_text = " ".join(str(item or "") for item in context_hints if str(item or "").strip()).lower()
    return path_text, " ".join(part for part in (path_text, context_text) if part).lower()


def _infer_route_fields(obj: dict[str, Any], *, source_rel: str = "") -> dict[str, str]:
    route = obj.get("data_route") if isinstance(obj.get("data_route"), dict) else {}
    metadata = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
    source_broker = _pick_route_text(obj, route, metadata, "source_broker", "broker")
    source_provider = _pick_route_text(obj, route, metadata, "source_provider", "provider", "source")
    source_venue = _pick_route_text(obj, route, metadata, "source_venue", "venue")
    asset_class = _pick_route_text(obj, route, metadata, "asset_class", "market_kind", "instrument_class", "domain")
    routing_lane = _pick_route_text(obj, route, metadata, "routing_lane", "lane")
    source_quality_label = _pick_route_text(obj, route, metadata, "source_quality_label", "quality_label")

    symbol = str(obj.get("symbol") or obj.get("underlying_symbol") or "").strip().upper()
    path_haystack, route_haystack = _route_hint_text(obj, route, metadata, source_rel=source_rel)
    haystack = " ".join(
        [route_haystack, source_broker, source_provider, source_venue, asset_class, routing_lane, symbol]
    ).lower()
    if not source_broker:
        if "coinbase" in haystack:
            source_broker = "coinbase"
        elif "schwab" in haystack:
            source_broker = "schwab"
        elif "crypto" not in haystack and ("equities" in path_haystack or "equity" in path_haystack):
            source_broker = "schwab"
    if not source_provider:
        if "coinbase" in haystack:
            source_provider = "coinbase"
        elif "schwab_crypto" in haystack:
            source_provider = "schwab_crypto"
        elif "schwab" in haystack:
            source_provider = "schwab"
        else:
            source_provider = source_broker or "unknown"
    if not source_venue:
        if "schwab_crypto" in haystack:
            source_venue = "schwab_crypto_bridge"
        elif "coinbase" in haystack:
            source_venue = "coinbase"
        elif "schwab" in haystack:
            source_venue = "schwab"
        else:
            source_venue = source_provider or source_broker or "unknown"
    if not asset_class:
        if symbol.startswith("/"):
            asset_class = "futures"
        elif (
            "shadow_crypto" in path_haystack
            or "crypto_coinbase" in path_haystack
            or source_broker == "coinbase"
            or symbol.endswith(("-USD", "-USDT", "-USDC"))
        ):
            asset_class = "crypto"
        elif "equities" in path_haystack or "equity" in path_haystack:
            asset_class = "equities"
        elif "schwab_futures" in path_haystack or "_futures" in path_haystack:
            asset_class = "futures"
        elif "option" in haystack or "options" in haystack:
            asset_class = "options"
        elif "fx" in haystack or "forex" in haystack:
            asset_class = "fx"
        elif source_broker == "schwab":
            asset_class = "equities"
        else:
            asset_class = "unknown"
    if not routing_lane:
        if source_venue == "schwab_crypto_bridge":
            routing_lane = "schwab_crypto_bridge"
        elif source_broker and asset_class and asset_class != "unknown":
            routing_lane = f"{source_broker}_{asset_class}"
        elif source_provider and asset_class and asset_class != "unknown":
            routing_lane = f"{source_provider}_{asset_class}"
        else:
            routing_lane = "unclassified"
    if not source_quality_label:
        if source_venue == "schwab_crypto_bridge":
            source_quality_label = "broker_bridge"
        elif source_broker == "schwab":
            source_quality_label = "broker_native"
        elif source_broker == "coinbase":
            source_quality_label = "exchange_native"
        elif "crypto_market_context" in haystack:
            source_quality_label = "multi_source_context"
        elif "news" in haystack or "external_context" in haystack:
            source_quality_label = "public_context"
        else:
            source_quality_label = "unknown"
    return {
        "source_broker": source_broker or "unknown",
        "source_provider": source_provider or "unknown",
        "source_venue": source_venue or "unknown",
        "asset_class": asset_class or "unknown",
        "routing_lane": routing_lane or "unclassified",
        "source_quality_label": source_quality_label or "unknown",
    }


def _fallback_route_rows_from_payload(
    conn: sqlite3.Connection,
    *,
    columns: set[str],
    lookback_hours: float,
    limit: int,
) -> list[dict[str, Any]]:
    if "payload_json" not in columns:
        return []
    select_cols = ["payload_json"]
    if "source_rel" in columns:
        select_cols.append("source_rel")
    else:
        select_cols.append("'' AS source_rel")
    scan_limit = max(_safe_int(os.getenv("DATA_INTELLIGENCE_ROUTE_SCAN_ROWS"), 50000), 1000)
    params: list[Any] = []
    where_sql = ""
    if "id" in columns:
        order_sql = "ORDER BY id DESC"
        params.append(scan_limit)
        query = f"""
            SELECT {", ".join(select_cols)}
            FROM jsonl_records
            {order_sql}
            LIMIT ?
        """
    elif "ingested_at" in columns and lookback_hours > 0:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=float(lookback_hours))).isoformat()
        where_sql = "WHERE ingested_at >= ?"
        params.append(cutoff)
        query = f"""
            SELECT {", ".join(select_cols)}
            FROM jsonl_records
            {where_sql}
            LIMIT ?
        """
        params.append(min(max(int(limit), 1), scan_limit))
    else:
        query = f"""
            SELECT {", ".join(select_cols)}
            FROM jsonl_records
            LIMIT ?
        """
        params.append(min(max(int(limit), 1), scan_limit))
    grouped: Counter[tuple[str, str, str, str, str, str]] = Counter()
    for payload_json, source_rel in conn.execute(query, tuple(params)).fetchall():
        try:
            obj = json.loads(str(payload_json or "{}"))
        except Exception:
            obj = {}
        if not isinstance(obj, dict):
            obj = {}
        fields = _infer_route_fields(obj, source_rel=str(source_rel or ""))
        key = tuple(fields[column] for column in ROUTE_COLUMNS)
        grouped[key] += 1
    rows: list[dict[str, Any]] = []
    for key, count in grouped.most_common(500):
        row = {column: str(key[idx] or "unknown") for idx, column in enumerate(ROUTE_COLUMNS)}
        row["row_count"] = int(count)
        row["family"] = _route_family(row)
        rows.append(row)
    return rows


def _route_coverage(project_root: Path, *, db_path: Path, lookback_hours: float) -> dict[str, Any]:
    resolved_db = db_path if db_path.is_absolute() else project_root / db_path
    base: dict[str, Any] = {
        "ok": False,
        "overall_status": "missing",
        "db_path": str(resolved_db),
        "lookback_hours": round(float(lookback_hours), 3),
        "rows_total": 0,
        "sample_rows": [],
        "by_source_broker": {},
        "by_source_provider": {},
        "by_source_venue": {},
        "by_asset_class": {},
        "by_routing_lane": {},
        "by_source_quality_label": {},
        "by_family": {},
        "family_rows": {},
        "coverage_gaps": [],
    }
    if not resolved_db.exists():
        base["coverage_gaps"] = ["jsonl_link_sqlite_missing"]
        return base

    try:
        conn = sqlite3.connect(f"file:{resolved_db}?mode=ro", uri=True)
    except Exception as exc:
        return {**base, "overall_status": "blocked", "error": f"sqlite_open_failed:{type(exc).__name__}:{exc}"}

    try:
        if not _sqlite_table_exists(conn, "jsonl_records"):
            return {**base, "overall_status": "missing", "coverage_gaps": ["jsonl_records_table_missing"]}
        columns = _sqlite_columns(conn, "jsonl_records")
        missing_route_columns = [column for column in ROUTE_COLUMNS if column not in columns]
        coverage_mode = "materialized_route_columns"
        db_size_gb = resolved_db.stat().st_size / (1024.0**3)
        large_db_threshold_gb = max(_safe_float(os.getenv("DATA_INTELLIGENCE_PAYLOAD_INFERENCE_DB_SIZE_GB"), 25.0), 0.0)
        prefer_payload_inference = bool(
            _bool(os.getenv("DATA_INTELLIGENCE_ENABLE_LARGE_DB_PAYLOAD_INFERENCE"))
            and "payload_json" in columns
            and db_size_gb >= large_db_threshold_gb
        )
        materialized_rows = []
        fallback_limit = max(_safe_int(os.getenv("DATA_INTELLIGENCE_ROUTE_FALLBACK_ROWS"), 10000), 1000)
        materialized_known_only = bool(prefer_payload_inference and not missing_route_columns and "source_broker" in columns)
        select_exprs = []
        for column in ROUTE_COLUMNS:
            if column in columns:
                select_exprs.append(f"COALESCE(NULLIF({column}, ''), 'unknown') AS {column}")
            else:
                select_exprs.append(f"'unknown' AS {column}")
        params: list[Any] = []
        scan_limit = max(_safe_int(os.getenv("DATA_INTELLIGENCE_ROUTE_SCAN_ROWS"), 50000), 1000)
        if "id" in columns:
            inner_query = f"""
                SELECT {", ".join(select_exprs)}
                FROM jsonl_records
                ORDER BY id DESC
                LIMIT ?
            """
            params.append(scan_limit)
            query = f"""
                SELECT {", ".join(ROUTE_COLUMNS)}, COUNT(*) AS row_count
                FROM ({inner_query}) recent_route_rows
                GROUP BY {", ".join(ROUTE_COLUMNS)}
                ORDER BY row_count DESC
                LIMIT 500
            """
            coverage_mode = "materialized_route_columns_latest_id_sample"
            if materialized_known_only:
                coverage_mode = "materialized_route_columns_latest_known_id_sample"
        elif materialized_known_only:
            query = f"""
                SELECT {", ".join(select_exprs)}, COUNT(*) AS row_count
                FROM jsonl_records
                WHERE source_broker IS NOT NULL AND source_broker != ''
                GROUP BY {", ".join(ROUTE_COLUMNS)}
                ORDER BY row_count DESC
                LIMIT 500
            """
            coverage_mode = "materialized_route_columns_all_time"
        elif "ingested_at" in columns and lookback_hours > 0:
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=float(lookback_hours))).isoformat()
            params.append(cutoff)
            query = f"""
                SELECT {", ".join(select_exprs)}, COUNT(*) AS row_count
                FROM jsonl_records
                WHERE ingested_at >= ?
                GROUP BY {", ".join(ROUTE_COLUMNS)}
                ORDER BY row_count DESC
                LIMIT 500
            """
        else:
            query = f"""
                SELECT {", ".join(select_exprs)}, COUNT(*) AS row_count
                FROM jsonl_records
                GROUP BY {", ".join(ROUTE_COLUMNS)}
                ORDER BY row_count DESC
                LIMIT 500
            """
        for raw in conn.execute(query, tuple(params)).fetchall():
            row = {column: str(raw[idx] or "unknown") for idx, column in enumerate(ROUTE_COLUMNS)}
            row["row_count"] = int(raw[len(ROUTE_COLUMNS)] or 0)
            row["family"] = _route_family(row)
            materialized_rows.append(row)
        known_materialized_rows = [row for row in materialized_rows if not _route_row_is_unknown(row)]
        all_unknown = bool(materialized_rows) and not known_materialized_rows
        rows = materialized_rows
        if prefer_payload_inference or missing_route_columns or all_unknown:
            fallback_rows = _fallback_route_rows_from_payload(
                conn,
                columns=columns,
                lookback_hours=lookback_hours,
                limit=fallback_limit,
            )
            if fallback_rows:
                rows = _merge_route_rows(known_materialized_rows, fallback_rows)
                coverage_mode = (
                    (
                        coverage_mode + "_plus_payload_json_source_rel_inference"
                        if materialized_known_only
                        else "materialized_route_columns_plus_payload_json_source_rel_inference"
                    )
                    if known_materialized_rows
                    else "payload_json_source_rel_inference"
                )
    except Exception as exc:
        return {
            **base,
            "overall_status": "blocked",
            "error": f"sqlite_route_coverage_failed:{type(exc).__name__}:{exc}",
        }
    finally:
        conn.close()

    total = sum(int(row["row_count"]) for row in rows)
    counters = {column: Counter() for column in ROUTE_COLUMNS}
    family_counter: Counter[str] = Counter()
    quality_weighted = 0.0
    for row in rows:
        count = int(row["row_count"])
        for column in ROUTE_COLUMNS:
            counters[column][str(row[column] or "unknown")] += count
        family = str(row.get("family") or "other")
        family_counter[family] += count
        quality_weighted += _route_quality_score(str(row.get("source_quality_label") or "")) * count

    gaps = []
    if total <= 0:
        gaps.append("no_recent_route_labeled_rows")
    if family_counter.get("schwab", 0) <= 0 and family_counter.get("schwab_crypto_bridge", 0) <= 0:
        gaps.append("schwab_route_rows_missing")
    if family_counter.get("coinbase", 0) <= 0:
        gaps.append("coinbase_route_rows_missing")
    if family_counter.get("crypto", 0) <= 0 and family_counter.get("coinbase", 0) <= 0 and family_counter.get("schwab_crypto_bridge", 0) <= 0:
        gaps.append("crypto_route_rows_missing")
    if missing_route_columns:
        gaps.append("route_label_columns_missing:" + ",".join(missing_route_columns))

    return {
        **base,
        "ok": total > 0 and not missing_route_columns,
        "overall_status": "ready" if total > 0 and not gaps else "thin" if total > 0 else "missing",
        "coverage_mode": coverage_mode,
        "db_size_gb": round(db_size_gb, 6),
        "rows_total": int(total),
        "sample_rows": rows[:40],
        "by_source_broker": _counter_dict(counters["source_broker"]),
        "by_source_provider": _counter_dict(counters["source_provider"]),
        "by_source_venue": _counter_dict(counters["source_venue"]),
        "by_asset_class": _counter_dict(counters["asset_class"]),
        "by_routing_lane": _counter_dict(counters["routing_lane"]),
        "by_source_quality_label": _counter_dict(counters["source_quality_label"]),
        "by_family": _counter_dict(family_counter),
        "family_rows": {key: int(value) for key, value in sorted(family_counter.items())},
        "mean_route_quality_score": round(quality_weighted / max(total, 1), 6),
        "coverage_gaps": ordered_unique(gaps),
    }


def _load_source_verification(project_root: Path) -> dict[str, Any]:
    payload = load_json(project_root / "governance" / "health" / "source_verification_latest.json")
    if payload:
        return payload
    try:
        from scripts.ops import source_verification_report

        return source_verification_report.build_source_verification_payload(project_root)
    except Exception:
        return {}


def _source_rows_by_id(source_verification: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = source_verification.get("sources") if isinstance(source_verification.get("sources"), list) else []
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        source_id = str(row.get("source_id") or "").strip()
        if source_id:
            out[source_id] = row
    return out


def _health_artifacts(project_root: Path) -> dict[str, dict[str, Any]]:
    health = project_root / "governance" / "health"
    return {
        "crypto_market_context": load_json(health / "crypto_market_context_sync_latest.json"),
        "schwab_symbol_news": load_json(health / "schwab_symbol_news_latest.json"),
        "ticker_news_context": load_json(health / "ticker_news_context_latest.json"),
        "coinbase_api_health": load_json(health / "coinbase_api_health_latest.json"),
        "market_micro_context": load_json(health / "market_micro_sync_latest.json"),
    }


def _artifact_score(payload: dict[str, Any]) -> float:
    status = _status(payload)
    if status in {"ready", "ok", "active"}:
        return 1.0
    if status in {"thin", "warn", "degraded", "partial"}:
        return 0.58
    if status in {"missing"}:
        return 0.20
    return 0.10


def _verification_score(row: dict[str, Any]) -> float:
    if not row:
        return 0.25
    raw = row.get("source_confidence_score")
    if raw is not None:
        return max(0.0, min(_safe_float(raw, 0.25), 1.0))
    status = str(row.get("verification_status") or "").strip().lower()
    if status == "cross_verified":
        return 1.0
    if status == "single_source_verified":
        return 0.78
    if status == "single_source_unverified":
        return 0.25
    return 0.50 if bool(row.get("ok")) else 0.25


def _count_norm(count: int, target: int) -> float:
    return round(min(max(int(count), 0) / float(max(int(target), 1)), 1.0), 6)


def _source_scorecards(
    *,
    route_coverage: dict[str, Any],
    source_verification: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    rows_by_id = _source_rows_by_id(source_verification)
    family_rows = _as_dict(route_coverage.get("family_rows"))
    schwab_rows = _safe_int(family_rows.get("schwab"), 0) + _safe_int(family_rows.get("schwab_crypto_bridge"), 0)
    coinbase_rows = _safe_int(family_rows.get("coinbase"), 0)
    crypto_rows = coinbase_rows + _safe_int(family_rows.get("crypto"), 0) + _safe_int(family_rows.get("schwab_crypto_bridge"), 0)

    specs = {
        "schwab": {
            "route_rows": schwab_rows,
            "route_target_rows": 500,
            "verification_ids": ["schwab_symbol_news", "market_quote_profiles", "market_micro_context"],
            "artifact_ids": ["schwab_symbol_news", "market_micro_context"],
        },
        "coinbase": {
            "route_rows": coinbase_rows,
            "route_target_rows": 250,
            "verification_ids": ["crypto_market_context"],
            "artifact_ids": ["coinbase_api_health", "crypto_market_context"],
        },
        "crypto": {
            "route_rows": crypto_rows,
            "route_target_rows": 350,
            "verification_ids": ["crypto_market_context"],
            "artifact_ids": ["crypto_market_context", "coinbase_api_health"],
        },
        "news_context": {
            "route_rows": _safe_int(family_rows.get("schwab"), 0),
            "route_target_rows": 250,
            "verification_ids": ["schwab_symbol_news", "ticker_news_context"],
            "artifact_ids": ["schwab_symbol_news", "ticker_news_context"],
        },
    }
    scorecards: dict[str, dict[str, Any]] = {}
    for name, spec in specs.items():
        verification_ids = [str(item) for item in spec["verification_ids"]]
        artifact_ids = [str(item) for item in spec["artifact_ids"]]
        verification_scores = [_verification_score(rows_by_id.get(source_id, {})) for source_id in verification_ids]
        artifact_scores = [_artifact_score(artifacts.get(artifact_id, {})) for artifact_id in artifact_ids]
        route_score = _count_norm(_safe_int(spec["route_rows"], 0), _safe_int(spec["route_target_rows"], 1))
        confidence = round(
            0.36 * route_score
            + 0.34 * (sum(verification_scores) / max(len(verification_scores), 1))
            + 0.30 * (sum(artifact_scores) / max(len(artifact_scores), 1)),
            6,
        )
        if confidence >= 0.78:
            status = "ready"
        elif confidence >= 0.52:
            status = "thin"
        else:
            status = "needs_coverage"
        scorecards[name] = {
            "overall_status": status,
            "confidence_score": confidence,
            "route_rows": _safe_int(spec["route_rows"], 0),
            "route_target_rows": _safe_int(spec["route_target_rows"], 1),
            "route_score": route_score,
            "verification_scores": {source_id: _verification_score(rows_by_id.get(source_id, {})) for source_id in verification_ids},
            "artifact_statuses": {artifact_id: _status(artifacts.get(artifact_id, {})) for artifact_id in artifact_ids},
        }
    return scorecards


def _source_refresh_command(project_root: Path, source_id: str) -> str:
    opsctl = str(project_root / "scripts" / "ops" / "opsctl.sh")
    mapping = {
        "market_quote_profiles": [str(project_root / ".venv314" / "bin" / "python"), str(project_root / "scripts" / "data_source_divergence_bot.py"), "--json"],
        "options_context_mesh": [opsctl, "options-flow-sync", "--json"],
        "macro_crossstack": [opsctl, "macro-crosscheck", "--json"],
        "crypto_market_context": [opsctl, "crypto-market-sync", "--json"],
        "free_equity_reference_context": [
            opsctl,
            "free-equity-reference-sync",
            "--max-symbols",
            "40",
            "--timeout",
            "2.5",
            "--max-runtime-seconds",
            "45",
            "--json",
        ],
        "fx_market_context": [opsctl, "fx-market-sync", "--json"],
        "public_macro_feeds": [opsctl, "macro-context-sync", "--json"],
        "official_macro_context": [opsctl, "macro-context-sync", "--json"],
        "schwab_education_context": [opsctl, "schwab-education-sync", "--json"],
        "schwab_symbol_news": [opsctl, "schwab-symbol-news-sync", "--max-runtime-seconds", "240", "--json"],
        "ticker_news_context": [opsctl, "ticker-news-sync", "--max-runtime-seconds", "300", "--json"],
        "market_micro_context": [opsctl, "market-micro-sync", "--json"],
        "sec_edgar_context": [opsctl, "sec-edgar-sync", "--json"],
        "extended_quant_context": [opsctl, "extended-quant-sync", "--json"],
        "public_policy_context": [opsctl, "public-policy-sync", "--json"],
    }
    command = mapping.get(str(source_id), [opsctl, "source-verification", "--json"])
    return " ".join(shlex.quote(str(part)) for part in command)


def _source_provider_ratio(evidence: dict[str, Any]) -> tuple[int, int, float]:
    total = max(
        _safe_int(evidence.get("total_sources"), 0),
        _safe_int(evidence.get("source_count"), 0),
        _safe_int(evidence.get("provider_total"), 0),
    )
    ok = max(
        _safe_int(evidence.get("ok_sources"), 0),
        _safe_int(evidence.get("effective_ok_sources"), 0),
        _safe_int(evidence.get("provider_ok"), 0),
    )
    if total <= 0 and _safe_int(evidence.get("symbols_with_chain"), 0) > 0:
        total = 1
        ok = 1
    ratio = min(max(ok / float(max(total, 1)), 0.0), 1.0) if total > 0 else 1.0
    return ok, total, ratio


def _coverage_ratio(evidence: dict[str, Any]) -> float | None:
    raw = evidence.get("coverage_ratio")
    if raw is not None:
        return max(0.0, min(_safe_float(raw, 0.0), 1.0))
    requested = max(
        _safe_int(evidence.get("requested_symbols"), 0),
        _safe_int(evidence.get("symbols_requested"), 0),
    )
    covered = max(
        _safe_int(evidence.get("symbols_with_news"), 0),
        _safe_int(evidence.get("symbols_with_reference"), 0),
        _safe_int(evidence.get("symbols_with_chain"), 0),
        _safe_int(evidence.get("tracked_symbols"), 0),
        _safe_int(evidence.get("tracked_assets"), 0),
    )
    if requested <= 0:
        return None
    return max(0.0, min(covered / float(max(requested, 1)), 1.0))


def _source_risk_rows(project_root: Path, source_verification: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _as_list(source_verification.get("sources"))
    risk_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        source_id = str(row.get("source_id") or "").strip()
        if not source_id:
            continue
        evidence = _as_dict(row.get("evidence"))
        notes = [str(item or "").strip() for item in _as_list(row.get("notes")) if str(item or "").strip()]
        confidence = _verification_score(row)
        verification_status = str(row.get("verification_status") or "").strip().lower()
        ok = bool(row.get("ok", False))
        fresh = bool(row.get("fresh", False))
        provider_ok, provider_total, provider_ratio = _source_provider_ratio(evidence)
        coverage = _coverage_ratio(evidence)
        risk = max(1.0 - confidence, 0.0)
        reasons: list[str] = []

        if not ok:
            risk += 0.35
            reasons.append("source_not_ok")
        if not fresh:
            risk += 0.25
            reasons.append("stale_source")
        if verification_status == "single_source_unverified":
            risk += 0.35
            reasons.append("single_source_unverified")
        elif verification_status == "single_source_verified":
            risk += 0.04
            reasons.append("single_source_only")
        if provider_total > 0 and provider_ratio < 1.0:
            risk += min((1.0 - provider_ratio) * 0.28, 0.22)
            reasons.append(f"partial_provider_mesh={provider_ok}/{provider_total}")
        if notes:
            risk += min(len(notes) * 0.04, 0.16)
            reasons.extend(notes[:4])
        if coverage is not None:
            if coverage <= 0.0:
                risk += 0.20
                reasons.append("zero_symbol_coverage")
            elif coverage < 0.05:
                risk += 0.12
                reasons.append(f"very_low_symbol_coverage={coverage:.3f}")
            elif coverage < 0.12:
                risk += 0.06
                reasons.append(f"low_symbol_coverage={coverage:.3f}")
        if evidence.get("broker_native_news_endpoint_available") is False and evidence.get("fallback_active") is True:
            risk += 0.08
            reasons.append("broker_native_endpoint_unavailable_using_fallback")
        risk = round(max(0.0, min(risk, 1.0)), 6)
        if ok and fresh and verification_status != "single_source_unverified":
            risk = min(risk, 0.74)

        if risk >= 0.75 or (not ok and verification_status == "single_source_unverified"):
            tier = "bad_data_blocker"
        elif risk >= 0.45:
            tier = "degraded_suspect"
        elif risk >= 0.22:
            tier = "watch"
        else:
            tier = "trusted"

        label_weight = max(0.15, min(1.0, confidence - (risk * 0.35)))
        if tier == "bad_data_blocker":
            label_weight = min(label_weight, 0.25)
        elif tier == "degraded_suspect":
            label_weight = min(label_weight, 0.55)
        elif tier == "watch":
            label_weight = min(label_weight, 0.78)
        risk_rows.append(
            {
                "source_id": source_id,
                "risk_tier": tier,
                "risk_score": risk,
                "label_weight": round(label_weight, 6),
                "source_confidence_score": round(confidence, 6),
                "verification_status": verification_status,
                "ok": ok,
                "fresh": fresh,
                "provider_ok_sources": provider_ok,
                "provider_total_sources": provider_total,
                "provider_ok_ratio": round(provider_ratio, 6),
                "coverage_ratio": round(coverage, 6) if coverage is not None else None,
                "reasons": ordered_unique(reasons),
                "refresh_command": _source_refresh_command(project_root, source_id),
            }
        )
    return sorted(risk_rows, key=lambda item: (-_safe_float(item.get("risk_score"), 0.0), str(item.get("source_id") or "")))


def _degradation_intelligence(
    *,
    project_root: Path,
    route_coverage: dict[str, Any],
    source_verification: dict[str, Any],
    scorecards: dict[str, dict[str, Any]],
    pressure: dict[str, Any],
    volume_plan: dict[str, Any],
) -> dict[str, Any]:
    source_risks = _source_risk_rows(project_root, source_verification)
    bad = [row for row in source_risks if str(row.get("risk_tier")) == "bad_data_blocker"]
    suspects = [row for row in source_risks if str(row.get("risk_tier")) == "degraded_suspect"]
    watch = [row for row in source_risks if str(row.get("risk_tier")) == "watch"]
    weak_scorecards = [
        {"source_group": name, **row}
        for name, row in scorecards.items()
        if str(row.get("overall_status") or "") != "ready"
    ]
    route_gaps = [str(item) for item in _as_list(route_coverage.get("coverage_gaps")) if str(item).strip()]
    pressure_blockers, pressure_severe = _pressure_blockers(pressure)

    if bad:
        status = "blocked"
    elif suspects or pressure_severe or str(volume_plan.get("profile") or "") == "deferred":
        status = "degraded"
    elif watch or weak_scorecards or route_gaps or pressure_blockers:
        status = "watch"
    else:
        status = "ready"

    label_weight_overrides = {
        str(row["source_id"]): float(row["label_weight"])
        for row in source_risks
        if str(row.get("risk_tier")) != "trusted"
    }
    root_causes = []
    if bad:
        root_causes.append("bad_source_data_blockers")
    if suspects:
        root_causes.append("degraded_source_suspects")
    if route_gaps:
        root_causes.append("route_coverage_gaps")
    if weak_scorecards:
        root_causes.append("weak_source_scorecards")
    if pressure_severe or pressure_blockers:
        root_causes.append("runtime_or_storage_pressure")

    actions = []
    for row in [*bad, *suspects, *watch][:8]:
        actions.append(str(row.get("refresh_command") or ""))
    if route_gaps:
        actions.append(str(project_root / "scripts" / "ops" / "data_intelligence_layer.py") + " --apply --json")
    actions.append(str(project_root / "scripts" / "ops" / "opsctl.sh") + " source-verification --json")

    return {
        "enabled": True,
        "mode": "max",
        "overall_status": status,
        "bad_source_data_present": bool(bad),
        "bad_source_data_suspects": bad,
        "degraded_source_suspects": suspects,
        "watch_sources": watch,
        "source_risk_rows": source_risks[:80],
        "label_weight_overrides": label_weight_overrides,
        "route_coverage_gaps": route_gaps,
        "weak_source_scorecards": weak_scorecards,
        "pressure_blockers": pressure_blockers,
        "pressure_severe_reasons": pressure_severe,
        "root_causes": ordered_unique(root_causes),
        "recommended_commands": ordered_unique([action for action in actions if action])[:12],
        "policy": {
            "bad_data_blocker": "exclude_or_require_refresh_before_training_labels",
            "degraded_suspect": "downweight_and_prefer_cross_verified_alternates",
            "watch": "keep_active_but_reduce_training_label_weight",
            "trusted": "normal_weight",
        },
    }


def _pressure_context(project_root: Path) -> dict[str, Any]:
    health = project_root / "governance" / "health"
    runtime = load_json(health / "runtime_throttle_control_latest.json")
    storage = load_json(health / "ingestion_storage_control_latest.json")
    backpressure = load_json(health / "ingestion_backpressure_latest.json")
    storage_auto = load_json(health / "storage_backpressure_autopilot_latest.json")
    global_halt = load_json(health / "global_killswitch_latest.json") or load_json(health / "global_halt_auto_clear_latest.json")

    storage_bp = _as_dict(storage.get("backpressure"))
    effective_raw_live = _as_dict(storage_bp.get("effective_raw_live"))
    auto_bp = _as_dict(storage_auto.get("backpressure"))
    if effective_raw_live:
        pending_total = _safe_int(effective_raw_live.get("total_pending_lines"), _safe_int(storage_bp.get("total_pending_lines"), 0))
        pending_source = str(storage_bp.get("effective_raw_live_source") or effective_raw_live.get("source") or "effective_raw_live")
    else:
        pending_total = max(
            _safe_int(storage.get("pending_lines_total"), 0),
            _safe_int(backpressure.get("pending_lines_total"), 0),
            _safe_int(storage_bp.get("total_pending_lines"), 0),
            _safe_int(auto_bp.get("total_pending_lines"), 0),
            _safe_int(storage_auto.get("total_pending_lines"), 0),
        )
        pending_source = "reported_pending_lines"
    threshold = max(
        _safe_int(storage.get("pending_lines_threshold"), 0),
        _safe_int(backpressure.get("pending_lines_threshold"), 0),
        _safe_int(storage_bp.get("pending_lines_threshold"), 0),
        _safe_int(auto_bp.get("pending_lines_threshold"), 0),
        15000,
    )
    pending_ratio = pending_total / float(max(threshold, 1))
    host_saturation = max(
        _safe_float(runtime.get("host_saturation_score"), 0.0),
        _safe_float(runtime.get("saturation_score"), 0.0),
    )
    return {
        "runtime_status": _status(runtime),
        "host_saturation_score": round(host_saturation, 3),
        "compute_pressure_level": str(runtime.get("compute_pressure_level") or "unknown").strip().lower(),
        "memory_pressure_level": str(runtime.get("memory_pressure_level") or "unknown").strip().lower(),
        "storage_status": _status(storage),
        "storage_severity": str(storage.get("severity") or storage_auto.get("severity") or "unknown").strip().lower(),
        "storage_pressure_index": round(max(_safe_float(storage.get("pressure_index"), 0.0), _safe_float(storage_auto.get("pressure_index"), 0.0)), 6),
        "pending_lines_total": int(pending_total),
        "pending_lines_source": pending_source,
        "pending_lines_threshold": int(threshold),
        "pending_ratio": round(pending_ratio, 6),
        "global_halt_active": _bool(global_halt.get("halt")) or _bool(global_halt.get("global_halt_active")),
        "global_halt_state": str(global_halt.get("halt_state") or global_halt.get("state") or "unknown"),
    }


def _pressure_blockers(pressure: dict[str, Any]) -> tuple[list[str], list[str]]:
    blockers: list[str] = []
    severe: list[str] = []
    runtime_status = str(pressure.get("runtime_status") or "")
    if runtime_status in {"blocked", "critical"}:
        severe.append("runtime_status_" + runtime_status)
    elif runtime_status in {"degraded"}:
        blockers.append("runtime_status_degraded")
    host_score = _safe_float(pressure.get("host_saturation_score"), 0.0)
    if host_score >= 85.0:
        severe.append("host_saturation_extreme")
    elif host_score >= 65.0:
        blockers.append("host_saturation_high")
    for key in ("compute_pressure_level", "memory_pressure_level"):
        level = str(pressure.get(key) or "").lower()
        if level in {"critical", "hard", "emergency"}:
            severe.append(f"{key}_{level}")
        elif level in {"high", "relief", "swap_relief", "hard_relief"}:
            blockers.append(f"{key}_{level}")
    pending_ratio = _safe_float(pressure.get("pending_ratio"), 0.0)
    if pending_ratio >= 3.0:
        severe.append("ingestion_backpressure_extreme")
    elif pending_ratio >= 1.0:
        blockers.append("ingestion_backpressure_active")
    if str(pressure.get("storage_severity") or "") in {"critical", "blocked"}:
        severe.append("storage_pressure_critical")
    elif str(pressure.get("storage_status") or "") in {"degraded", "blocked", "critical"}:
        blockers.append("storage_status_" + str(pressure.get("storage_status")))
    if _bool(pressure.get("global_halt_active")):
        severe.append("global_halt_active")
    return ordered_unique(blockers), ordered_unique(severe)


def _compute_pressure_high_guarded_for_expansion(pressure: dict[str, Any]) -> bool:
    runtime_status = str(pressure.get("runtime_status") or "").strip().lower()
    compute_level = str(pressure.get("compute_pressure_level") or "").strip().lower()
    memory_level = str(pressure.get("memory_pressure_level") or "").strip().lower()
    storage_status = str(pressure.get("storage_status") or "").strip().lower()
    storage_severity = str(pressure.get("storage_severity") or "").strip().lower()
    return bool(
        compute_level == "high"
        and runtime_status in {"ready", "advisory"}
        and memory_level in {"normal", "unknown", ""}
        and storage_status in {"ready", "advisory"}
        and storage_severity in {"stable", "ready", "low", "unknown", ""}
        and _safe_float(pressure.get("host_saturation_score"), 0.0) < 55.0
        and _safe_float(pressure.get("storage_pressure_index"), 0.0) < 0.25
        and _safe_float(pressure.get("pending_ratio"), 0.0) < 0.05
        and not _bool(pressure.get("global_halt_active"))
    )


def _profile_env(profile: str) -> dict[str, str]:
    intelligence_env = {
        "DATA_DEGRADATION_INTELLIGENCE_MODE": "max",
        "DATA_SOURCE_BAD_DATA_DOWNWEIGHT_ENABLED": "1",
        "DATA_SOURCE_LOW_CONFIDENCE_THRESHOLD": "0.72",
        "DATA_SOURCE_SUSPECT_RISK_THRESHOLD": "0.45",
        "DATA_SOURCE_BAD_DATA_RISK_THRESHOLD": "0.75",
        "TRAINING_SOURCE_LABEL_WEIGHTING_ENABLED": "1",
        "TRAINING_LOW_CONFIDENCE_SOURCE_MIN_WEIGHT": "0.25",
    }
    symbols = ",".join(EXPANDED_CRYPTO_SYMBOLS)
    if profile == "deferred":
        return {
            **intelligence_env,
            "DATA_INTELLIGENCE_LAYER_ENABLED": "1",
            "DATA_VOLUME_PROFILE": "deferred",
            "DATA_VOLUME_PULLS_DEFERRED": "1",
            "CRYPTO_MARKET_CONTEXT_MAX_SYMBOLS": "8",
            "CRYPTO_MARKET_CONTEXT_SYMBOLS": ",".join(EXPANDED_CRYPTO_SYMBOLS[:8]),
            "CRYPTO_MARKET_CONTEXT_COINBASE_QUOTE_MAX_ASSETS": "8",
            "CRYPTO_MARKET_CONTEXT_TIMEOUT_SECONDS": "8",
            "CRYPTO_MARKET_CONTEXT_MAX_RUNTIME_SECONDS": "45",
            "SCHWAB_SYMBOL_NEWS_MAX_SYMBOLS": "80",
            "SCHWAB_SYMBOL_NEWS_LIMIT_PER_SYMBOL": "20",
            "SCHWAB_SYMBOL_NEWS_MAX_RUNTIME_SECONDS": "120",
            "SCHWAB_SYMBOL_NEWS_SLEEP_SECONDS": "0.03",
            "TICKER_NEWS_MAX_SYMBOLS": "80",
            "TICKER_NEWS_LIMIT_PER_SYMBOL": "8",
            "TICKER_NEWS_MAX_RUNTIME_SECONDS": "120",
            "TICKER_NEWS_TIMEOUT_SECONDS": "4",
            "TICKER_NEWS_SLEEP_SECONDS": "0.03",
            "MARKET_MICRO_LOOKBACK_DAYS": "7",
            "MARKET_MICRO_FINRA_LOOKBACK_DAYS": "3",
            "MARKET_MICRO_TIMEOUT_SECONDS": "3",
            "MARKET_MICRO_MAX_RUNTIME_SECONDS": "45",
            "MARKET_MICRO_OUTER_TIMEOUT_SECONDS": "60",
        }
    if profile == "conservative":
        return {
            **intelligence_env,
            "DATA_INTELLIGENCE_LAYER_ENABLED": "1",
            "DATA_VOLUME_PROFILE": "conservative",
            "DATA_VOLUME_PULLS_DEFERRED": "0",
            "CRYPTO_MARKET_CONTEXT_MAX_SYMBOLS": "18",
            "CRYPTO_MARKET_CONTEXT_SYMBOLS": ",".join(EXPANDED_CRYPTO_SYMBOLS[:18]),
            "CRYPTO_MARKET_CONTEXT_COINBASE_QUOTE_MAX_ASSETS": "18",
            "CRYPTO_MARKET_CONTEXT_TIMEOUT_SECONDS": "12",
            "CRYPTO_MARKET_CONTEXT_MAX_RUNTIME_SECONDS": "75",
            "SCHWAB_SYMBOL_NEWS_MAX_SYMBOLS": "180",
            "SCHWAB_SYMBOL_NEWS_LIMIT_PER_SYMBOL": "50",
            "SCHWAB_SYMBOL_NEWS_MAX_RUNTIME_SECONDS": "240",
            "SCHWAB_SYMBOL_NEWS_SLEEP_SECONDS": "0.02",
            "TICKER_NEWS_MAX_SYMBOLS": "180",
            "TICKER_NEWS_LIMIT_PER_SYMBOL": "12",
            "TICKER_NEWS_MAX_RUNTIME_SECONDS": "240",
            "TICKER_NEWS_TIMEOUT_SECONDS": "5",
            "TICKER_NEWS_SLEEP_SECONDS": "0.02",
            "MARKET_MICRO_LOOKBACK_DAYS": "21",
            "MARKET_MICRO_FINRA_LOOKBACK_DAYS": "10",
            "MARKET_MICRO_TIMEOUT_SECONDS": "4",
            "MARKET_MICRO_MAX_RUNTIME_SECONDS": "75",
            "MARKET_MICRO_OUTER_TIMEOUT_SECONDS": "90",
        }
    return {
        **intelligence_env,
        "DATA_INTELLIGENCE_LAYER_ENABLED": "1",
        "DATA_VOLUME_PROFILE": "expanded",
        "DATA_VOLUME_PULLS_DEFERRED": "0",
        "CRYPTO_MARKET_CONTEXT_MAX_SYMBOLS": "36",
        "CRYPTO_MARKET_CONTEXT_SYMBOLS": symbols,
        "CRYPTO_MARKET_CONTEXT_COINBASE_QUOTE_MAX_ASSETS": "36",
        "CRYPTO_MARKET_CONTEXT_TIMEOUT_SECONDS": "18",
        "CRYPTO_MARKET_CONTEXT_MAX_RUNTIME_SECONDS": "120",
        "SCHWAB_SYMBOL_NEWS_MAX_SYMBOLS": "0",
        "SCHWAB_SYMBOL_NEWS_LIMIT_PER_SYMBOL": "80",
        "SCHWAB_SYMBOL_NEWS_MAX_RUNTIME_SECONDS": "420",
        "SCHWAB_SYMBOL_NEWS_SLEEP_SECONDS": "0.015",
        "TICKER_NEWS_MAX_SYMBOLS": "0",
        "TICKER_NEWS_LIMIT_PER_SYMBOL": "20",
        "TICKER_NEWS_MAX_RUNTIME_SECONDS": "420",
        "TICKER_NEWS_TIMEOUT_SECONDS": "6",
        "TICKER_NEWS_SLEEP_SECONDS": "0.015",
        "MARKET_MICRO_LOOKBACK_DAYS": "45",
        "MARKET_MICRO_FINRA_LOOKBACK_DAYS": "15",
        "MARKET_MICRO_TIMEOUT_SECONDS": "6",
        "MARKET_MICRO_MAX_RUNTIME_SECONDS": "120",
        "MARKET_MICRO_OUTER_TIMEOUT_SECONDS": "150",
    }


def _choose_profile(
    *,
    forced_profile: str,
    route_coverage: dict[str, Any],
    source_verification: dict[str, Any],
    scorecards: dict[str, dict[str, Any]],
    pressure: dict[str, Any],
) -> tuple[str, list[str], list[str]]:
    blockers, severe = _pressure_blockers(pressure)
    guarded_compute_expansion = _compute_pressure_high_guarded_for_expansion(pressure)
    if guarded_compute_expansion:
        blockers = [item for item in blockers if item != "compute_pressure_level_high"]
    if forced_profile != "auto":
        return forced_profile, blockers, severe
    if severe:
        return "deferred", blockers, severe
    if blockers:
        return "conservative", blockers, severe
    reasons = []
    if str(route_coverage.get("overall_status") or "") in {"missing", "thin", "needs_coverage"}:
        reasons.append("route_coverage_thin")
    if str(source_verification.get("overall_status") or "") in {"missing", "degraded", "blocked", "critical"}:
        reasons.append("source_verification_needs_refresh")
    weak_scorecards = [name for name, row in scorecards.items() if str(row.get("overall_status") or "") != "ready"]
    if weak_scorecards:
        reasons.append("scorecards_need_more_observations:" + ",".join(weak_scorecards))
    if guarded_compute_expansion:
        reasons.append("compute_pressure_high_guarded_by_runtime_advisory_storage_clear")
    return "expanded", blockers + reasons, severe


def _collector_commands(project_root: Path, profile: str) -> list[dict[str, Any]]:
    opsctl = str(project_root / "scripts" / "ops" / "opsctl.sh")
    commands: list[list[str]] = [
        [opsctl, "coinbase-api-health", "--snapshot", "--json"],
        [opsctl, "crypto-market-sync", "--json"],
        [opsctl, "schwab-symbol-news-sync", "--json"],
        [opsctl, "ticker-news-sync", "--json"],
        [opsctl, "market-micro-sync", "--json"],
        [opsctl, "source-verification", "--json"],
    ]
    if profile == "expanded":
        commands[3].insert(-1, "--include-optional-global-feeds")
    if profile == "deferred":
        commands = [[opsctl, "coinbase-api-health", "--json"], [opsctl, "source-verification", "--json"]]
    return [
        {
            "argv": command,
            "shell": " ".join(shlex.quote(part) for part in command),
        }
        for command in commands
    ]


def _volume_plan(
    *,
    project_root: Path,
    profile: str,
    reasons: list[str],
    severe_reasons: list[str],
) -> dict[str, Any]:
    env = _profile_env(profile)
    return {
        "profile": profile,
        "profile_reasons": ordered_unique(reasons),
        "severe_guardrail_reasons": ordered_unique(severe_reasons),
        "guardrail_policy": "expand_when_runtime_and_storage_are_calm; downshift_when_backpressure_or_host_pressure_is_active",
        "override_env": env,
        "collector_commands": _collector_commands(project_root, profile),
    }


def _artifact_summary(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    crypto = artifacts.get("crypto_market_context", {})
    schwab = artifacts.get("schwab_symbol_news", {})
    ticker = artifacts.get("ticker_news_context", {})
    coinbase = artifacts.get("coinbase_api_health", {})
    coinbase_public = _as_dict(coinbase.get("public_market_data"))
    return {
        "crypto_market_context": {
            "overall_status": _status(crypto),
            "tracked_symbols": _safe_int(crypto.get("tracked_symbols"), 0),
            "ok_source_count": _safe_int(crypto.get("ok_source_count"), 0),
            "source_count": _safe_int(crypto.get("source_count"), 0),
            "compared_assets": _safe_int(crypto.get("compared_assets"), 0),
            "news_row_count": _safe_int(crypto.get("news_row_count"), 0),
        },
        "schwab_symbol_news": {
            "overall_status": _status(schwab),
            "attempted_symbol_count": _safe_int(schwab.get("attempted_symbol_count"), 0),
            "requested_symbol_count": _safe_int(schwab.get("requested_symbol_count"), 0),
            "symbols_with_news": _safe_int(schwab.get("symbols_with_news"), 0),
            "total_news_items": _safe_int(schwab.get("total_news_items"), 0),
            "coverage_ratio": round(_safe_float(schwab.get("coverage_ratio"), 0.0), 6),
        },
        "ticker_news_context": {
            "overall_status": _status(ticker),
            "symbols_with_news": _safe_int(ticker.get("symbols_with_news"), 0),
            "requested_symbol_count": _safe_int(ticker.get("requested_symbol_count"), 0),
            "total_news_items": _safe_int(ticker.get("total_news_items"), 0),
            "ok_source_count": _safe_int(ticker.get("ok_source_count"), 0),
            "source_count": _safe_int(ticker.get("source_count"), 0),
        },
        "coinbase_api_health": {
            "overall_status": _status(coinbase),
            "public_ok": _bool(coinbase_public.get("ok")),
            "symbol": str(coinbase_public.get("symbol") or ""),
            "latency_ms": _safe_float(coinbase_public.get("latency_ms"), 0.0),
            "snapshot_requested": _bool(coinbase_public.get("snapshot_requested")),
        },
    }


def _write_override(path: Path, env: dict[str, str], *, profile: str, timestamp: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Auto-managed by scripts/ops/data_intelligence_layer.py",
        f"# layer_version={LAYER_VERSION}",
        f"# profile={profile}",
        f"# applied_at_utc={timestamp}",
    ]
    full_env = {
        **env,
        "DATA_INTELLIGENCE_LAYER_VERSION": LAYER_VERSION,
        "DATA_INTELLIGENCE_LAST_APPLIED_UTC": timestamp,
    }
    for key in sorted(full_env):
        lines.append(f"{key}={shlex.quote(str(full_env[key]))}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    db_path: Path = DEFAULT_SQLITE_DB,
    out_path: Path = DEFAULT_OUT_PATH,
    override_path: Path = DEFAULT_OVERRIDE_PATH,
    lookback_hours: float = 48.0,
    apply: bool = False,
    profile: str = "auto",
    repair_sql_schema: bool = False,
    repair_sql_indexes: bool = False,
) -> dict[str, Any]:
    timestamp = iso_now()
    schema_repair = (
        _repair_sql_route_schema(project_root, db_path=db_path, create_indexes=repair_sql_indexes)
        if apply and repair_sql_schema
        else {"ok": False, "overall_status": "not_requested", "added_columns": [], "created_indexes": []}
    )
    source_verification = _load_source_verification(project_root)
    artifacts = _health_artifacts(project_root)
    route_coverage = _route_coverage(project_root, db_path=db_path, lookback_hours=lookback_hours)
    scorecards = _source_scorecards(route_coverage=route_coverage, source_verification=source_verification, artifacts=artifacts)
    pressure = _pressure_context(project_root)
    selected_profile, reasons, severe = _choose_profile(
        forced_profile=profile,
        route_coverage=route_coverage,
        source_verification=source_verification,
        scorecards=scorecards,
        pressure=pressure,
    )
    volume_plan = _volume_plan(
        project_root=project_root,
        profile=selected_profile,
        reasons=reasons,
        severe_reasons=severe,
    )
    degradation_intelligence = _degradation_intelligence(
        project_root=project_root,
        route_coverage=route_coverage,
        source_verification=source_verification,
        scorecards=scorecards,
        pressure=pressure,
        volume_plan=volume_plan,
    )
    weak_sources = [name for name, row in scorecards.items() if str(row.get("overall_status") or "") != "ready"]
    overall_status = "ready"
    if selected_profile == "deferred":
        overall_status = "degraded"
    elif weak_sources or str(route_coverage.get("overall_status") or "") != "ready":
        overall_status = "thin"
    payload: dict[str, Any] = {
        "timestamp_utc": timestamp,
        "schema_version": 1,
        "layer_version": LAYER_VERSION,
        "ok": overall_status in {"ready", "thin"},
        "overall_status": overall_status,
        "apply_requested": bool(apply),
        "applied": False,
        "project_root": str(project_root),
        "out_path": str(out_path),
        "override_path": str(override_path),
        "sql_route_schema_repair": schema_repair,
        "route_coverage": route_coverage,
        "source_verification_summary": {
            "overall_status": _status(source_verification),
            "unverified_sources": _as_list(_as_dict(source_verification.get("overall")).get("unverified_sources")),
            "stale_sources": _as_list(_as_dict(source_verification.get("overall")).get("stale_sources")),
            "mean_source_confidence_score": _safe_float(_as_dict(source_verification.get("overall")).get("mean_source_confidence_score"), 0.0),
        },
        "artifact_summary": _artifact_summary(artifacts),
        "source_scorecards": scorecards,
        "pressure_context": pressure,
        "volume_plan": volume_plan,
        "degradation_intelligence": degradation_intelligence,
        "training_label_bridge": {
            "enabled": True,
            "policy": "route_labels_and_source_scorecards_feed_training_context_weighting",
            "degradation_intelligence_mode": degradation_intelligence.get("mode"),
            "source_label_weighting_enabled": True,
            "source_label_weight_overrides": degradation_intelligence.get("label_weight_overrides", {}),
            "bad_source_data_policy": degradation_intelligence.get("policy", {}),
            "preferred_contexts": [
                "source_broker",
                "source_provider",
                "source_venue",
                "asset_class",
                "routing_lane",
                "source_quality_label",
                "source_quality_score",
                "data_labels",
            ],
        },
        "recommended_actions": [
            "apply the managed data intelligence override before collector refreshes" if not apply else "",
            "run collector_commands from volume_plan to immediately refill weak Schwab, Coinbase, crypto, news, and microstructure lanes",
            "keep source-verification current after high-volume pulls so training labels downweight thin sources correctly",
            "refresh or downweight degraded source suspects before using them as high-confidence labels"
            if degradation_intelligence.get("bad_source_data_suspects") or degradation_intelligence.get("degraded_source_suspects")
            else "",
        ],
    }
    payload["recommended_actions"] = [action for action in payload["recommended_actions"] if action]
    if apply:
        _write_override(override_path, volume_plan["override_env"], profile=selected_profile, timestamp=timestamp)
        payload["applied"] = True
        payload["applied_override_path"] = str(override_path)
    write_payload(out_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and optionally apply the stronger source/data intelligence volume layer.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--sqlite-db", default=str(DEFAULT_SQLITE_DB))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--override-file", default=str(DEFAULT_OVERRIDE_PATH))
    parser.add_argument("--lookback-hours", type=float, default=float(os.getenv("DATA_INTELLIGENCE_ROUTE_LOOKBACK_HOURS", "48") or 48))
    parser.add_argument("--profile", choices=["auto", "expanded", "conservative", "deferred"], default=os.getenv("DATA_INTELLIGENCE_FORCE_PROFILE", "auto"))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--repair-sql-schema", action="store_true", help="Add missing route columns before scanning. Off by default for live bounded operation.")
    parser.add_argument("--no-repair-sql-schema", action="store_true", help="Compatibility flag; schema repair is already off unless --repair-sql-schema is passed.")
    parser.add_argument("--repair-sql-indexes", action="store_true", help="Also build route indexes; leave off for very large SQLite files.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    payload = build_payload(
        project_root,
        db_path=Path(args.sqlite_db).expanduser(),
        out_path=Path(args.out_file).expanduser(),
        override_path=Path(args.override_file).expanduser(),
        lookback_hours=float(args.lookback_hours),
        apply=bool(args.apply),
        profile=str(args.profile or "auto"),
        repair_sql_schema=bool(args.repair_sql_schema) and not bool(args.no_repair_sql_schema),
        repair_sql_indexes=bool(args.repair_sql_indexes),
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        plan = _as_dict(payload.get("volume_plan"))
        coverage = _as_dict(payload.get("route_coverage"))
        print(
            "data_intelligence_layer "
            f"status={payload.get('overall_status')} "
            f"profile={plan.get('profile')} "
            f"route_rows={coverage.get('rows_total')} "
            f"applied={int(bool(payload.get('applied')))}"
        )
    return 0 if payload.get("overall_status") in {"ready", "thin", "degraded"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
