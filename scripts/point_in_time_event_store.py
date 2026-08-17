#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HEALTH_EVENT_PATTERNS = (
    "governance/health/ingestion_*latest.json",
    "governance/health/backpressure_*latest.json",
    "governance/health/jsonl_sql_ingestion_health_*_latest.json",
    "governance/health/storage_*latest.json",
    "governance/health/runtime_training_snapshot_latest.json",
    "governance/health/collector_contracts_latest.json",
    "governance/health/source_verification_latest.json",
    "governance/health/data_source_divergence*_latest.json",
    "governance/health/broker_truth_*_latest.json",
    "governance/health/*_sync_latest.json",
    "governance/health/shadow_watchdog_halt_recovery_latest.json",
    "governance/health/incident_auto_halt_state.json",
    "governance/health/provider_mesh_latest.json",
    "governance/health/service_control_plane_latest.json",
    "governance/health/retrain_pipeline_latest.json",
    "governance/health/execution_lane_*_latest.json",
    "governance/feature_store/latest.json",
)


def _parse_ts(raw: Any) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
    except Exception:
        return text


def _parse_datetime(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _sha256_json(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except Exception:
        return []
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _norm_text(raw: Any) -> str:
    return str(raw or "").strip().lower()


def _string_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


def _json_value_candidates(payload: dict[str, Any], keys: Iterable[str]) -> list[str]:
    out: list[str] = []
    for key in keys:
        raw = payload.get(key)
        if isinstance(raw, str) and raw.strip():
            out.append(raw.strip())
        elif isinstance(raw, list):
            out.extend(_string_list(raw))
    return out


def _normalize_event_category(row: dict[str, Any], *, source_path: Path) -> str:
    explicit = str(row.get("category") or "").strip()
    if explicit and explicit.lower() != "uncategorized":
        return explicit

    event_type = _norm_text(row.get("event_type") or row.get("type") or source_path.stem)
    source = _norm_text(row.get("source"))
    speaker = _norm_text(row.get("speaker"))
    template = _norm_text(row.get("template"))
    signal_types = " ".join(_string_list(row.get("market_signal_types")))
    joined = " ".join(part for part in [event_type, source, speaker, template, signal_types] if part).strip()

    if (
        "premarket_token_guard" in event_type
        or "broker_truth" in event_type
        or "broker_readiness" in joined
        or "token" in joined
        or "auth" in joined
        or "account snapshot" in joined
    ):
        return "broker_readiness"
    if any(token in joined for token in ("retrain", "distillation", "promotion", "champion", "challenger", "model registry", "canary", "training_success")):
        return "training_control"
    if any(token in joined for token in ("storage", "retention", "stale_stage", "archive", "vacuum", "sqlite_maintenance", "data_retention")):
        return "storage_control"
    if any(token in joined for token in ("sql_sync", "backpressure", "ingestion", "latency", "queue", "link_jsonl_to_sql", "sql_link")):
        return "ingestion_control"
    if any(token in joined for token in ("supreme court", "c-span", "legal", "justice", "court")):
        return "legal_policy"
    if any(token in joined for token in ("white house", "donald trump", "president", "fed", "fomc", "treasury", "powell")):
        return "policy_macro"
    if any(token in joined for token in ("options", "0dte", "gamma", "straddle", "strangle", "call wall", "put wall", "assignment")):
        return "options_event"
    if any(token in joined for token in ("futures", "basis", "roll", "inventory", "curve shift", "calendar spread")):
        return "futures_event"
    if any(token in joined for token in ("dividend", "drip", "payout", "ex-date", "yield trap")):
        return "dividend_event"
    if any(token in joined for token in ("long_term", "dca", "accumulation", "allocation", "compounder", "rebalance overlap")):
        return "long_term_allocation"
    if any(token in joined for token in ("earnings_call", "earnings", "analyst_day")):
        return "earnings"
    if any(token in joined for token in ("ceo_interview", "investor relations", "tim cook", "apple", "nvidia", "tesla")):
        return "issuer_event"
    if any(token in joined for token in ("8-k", "10-q", "10-k", "6-k", "edgar", "filing")):
        return "filing"
    if any(token in joined for token in ("halt", "luld", "ssr")):
        return "tradeability"
    if any(token in joined for token in ("opex", "rebalance", "russell", "msci", "roll")):
        return "calendar_regime"
    if any(token in joined for token in ("verification", "divergence", "coverage", "collector")):
        return "source_quality"
    if any(token in joined for token in ("provider_mesh", "service_control_plane", "ops_coordinator", "operator_cockpit", "platform_control_plane")):
        return "control_plane"
    if any(token in joined for token in ("execution_lane", "portfolio_allocator_service", "risk_service_boundary", "execution_gateway")):
        return "execution_control"
    if any(token in joined for token in ("fx_market", "eurusd", "usdjpy", "gbpusd")):
        return "fx_context"
    if any(token in joined for token in ("crypto_market", "defillama", "coingecko", "coinmetrics", "market_crypto")):
        return "crypto_context"
    if any(token in joined for token in ("market_micro", "relative_volume", "opening_auction", "block_trade")):
        return "market_micro_context"
    if any(token in joined for token in ("education", "schwab network", "schwab coaching")):
        return "education_media"
    if any(token in joined for token in ("quant", "cboe", "sofr", "cot", "threshold")):
        return "quant_context"
    return explicit or "uncategorized"


def _derive_join_key(row: dict[str, Any], *, category: str, timestamp_utc: str) -> str:
    resolution = row.get("event_resolution_join") if isinstance(row.get("event_resolution_join"), dict) else {}
    explicit = str(resolution.get("join_key") or "").strip()
    if explicit:
        return explicit
    source = str(row.get("source") or "").strip()
    speaker = str(row.get("speaker") or "").strip()
    symbols = sorted(_string_list(row.get("symbols")))
    ts_bucket = str(timestamp_utc or "")[:13]
    if symbols:
        return f"{category}:{','.join(symbols)}:{ts_bucket}"
    if source or speaker:
        return f"{category}:{source}:{speaker}:{ts_bucket}"
    event_type = str(row.get("event_type") or row.get("type") or "").strip()
    return f"{category}:{event_type}:{ts_bucket}"


def _event_row_from_payload(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    stem = path.stem
    timestamp_utc = _parse_ts(
        payload.get("timestamp_utc")
        or payload.get("generated_utc")
        or payload.get("updated_at_utc")
        or payload.get("last_updated_utc")
        or payload.get("checked_at_utc")
    )
    notes = _json_value_candidates(payload, ("notes", "warnings", "errors", "soft_failures", "required_failures"))
    event_row = {
        "timestamp_utc": timestamp_utc,
        "event_type": stem,
        "source": str(payload.get("source") or payload.get("title") or stem).strip(),
        "speaker": str(payload.get("source_id") or payload.get("name") or payload.get("category") or "").strip(),
        "symbols": _string_list(payload.get("symbols")),
        "market_signal_types": _json_value_candidates(payload, ("signal_types", "market_signal_types")) + notes,
        "broad_market": bool(payload.get("broad_market", False)),
        "artifact_path": str(path),
        "artifact_ok": bool(payload.get("ok", False)),
    }
    category = _normalize_event_category(event_row, source_path=path)
    if category == "uncategorized":
        if "market_breadth" in stem or "bond_reference" in stem:
            category = "policy_macro"
        elif "feature_store" in stem or "point_in_time_event_store" in stem:
            category = "source_quality"
        elif "sync_latest" in stem:
            category = "collector_sync"
    event_row["category"] = category
    event_row["join_key"] = _derive_join_key(event_row, category=category, timestamp_utc=timestamp_utc)
    return event_row


def _iter_health_artifact_events(project_root: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for pattern in HEALTH_EVENT_PATTERNS:
        for path in sorted(project_root.glob(pattern)):
            if not path.is_file():
                continue
            path_key = str(path.resolve())
            if path_key in seen_paths:
                continue
            seen_paths.add(path_key)
            payload = _read_json(path)
            if not payload:
                continue
            event_row = _event_row_from_payload(path, payload)
            if str(event_row.get("timestamp_utc") or "").strip():
                events.append(event_row)
    return events


def _dedupe_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in events:
        category = str(row.get("category") or "uncategorized").strip()
        join_key = str(row.get("join_key") or "").strip()
        source = str(row.get("source") or "").strip()
        speaker = str(row.get("speaker") or "").strip()
        dedupe_key = (category, join_key, source, speaker)
        existing = deduped.get(dedupe_key)
        if existing is None or str(row.get("timestamp_utc") or "") > str(existing.get("timestamp_utc") or ""):
            deduped[dedupe_key] = row
    return list(deduped.values())


def build_event_store(
    project_root: Path,
    *,
    limit: int,
    now: datetime | None = None,
    future_tolerance_seconds: float = 300.0,
) -> dict[str, Any]:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    patterns = (
        "governance/events/live_macro_events_*.jsonl",
        "governance/events/live_macro_media_events_*.jsonl",
        "governance/events/premarket_token_guard_*.jsonl",
    )
    events: list[dict[str, Any]] = []
    for pattern in patterns:
        for path in sorted(project_root.glob(pattern))[-3:]:
            for row in _read_jsonl(path):
                event_type = str(row.get("event_type") or row.get("type") or path.stem).strip()
                symbols = row.get("symbols") if isinstance(row.get("symbols"), list) else []
                timestamp_utc = _parse_ts(row.get("timestamp_utc"))
                category = _normalize_event_category(row, source_path=path)
                events.append(
                    {
                        "timestamp_utc": timestamp_utc,
                        "event_type": event_type,
                        "category": category,
                        "source": str(row.get("source") or "").strip(),
                        "speaker": str(row.get("speaker") or "").strip(),
                        "symbols": symbols,
                        "join_key": _derive_join_key(row, category=category, timestamp_utc=timestamp_utc),
                        "broad_market": bool(row.get("market_broad_market", row.get("broad_market", False))),
                    }
                )
    events.extend(_iter_health_artifact_events(project_root))
    input_event_count = len(events)
    source_manifest_sha256 = _sha256_json(
        sorted(
            (
                {
                    "timestamp_utc": str(row.get("timestamp_utc") or ""),
                    "event_type": str(row.get("event_type") or ""),
                    "category": str(row.get("category") or ""),
                    "join_key": str(row.get("join_key") or ""),
                    "source": str(row.get("source") or ""),
                }
                for row in events
            ),
            key=lambda row: (
                row["timestamp_utc"],
                row["category"],
                row["join_key"],
                row["event_type"],
                row["source"],
            ),
        )
    )
    future_cutoff = now.timestamp() + max(float(future_tolerance_seconds), 0.0)
    accepted_events: list[dict[str, Any]] = []
    quarantined_events: list[dict[str, Any]] = []
    future_event_count = 0
    invalid_timestamp_count = 0
    for row in events:
        parsed = _parse_datetime(row.get("timestamp_utc"))
        if parsed is None:
            invalid_timestamp_count += 1
            quarantined_events.append({**row, "quarantine_reason": "invalid_effective_timestamp"})
            continue
        if parsed.timestamp() > future_cutoff:
            future_event_count += 1
            quarantined_events.append({**row, "quarantine_reason": "future_effective_timestamp"})
            continue
        accepted_events.append(row)
    events = _dedupe_events(accepted_events)
    events.sort(key=lambda row: row.get("timestamp_utc", ""), reverse=True)
    category_counts: dict[str, int] = {}
    latest_by_category: dict[str, dict[str, Any]] = {}
    for row in events:
        category = str(row.get("category") or "uncategorized")
        category_counts[category] = category_counts.get(category, 0) + 1
        latest_by_category.setdefault(
            category,
            {
                "timestamp_utc": str(row.get("timestamp_utc") or ""),
                "event_type": str(row.get("event_type") or ""),
                "join_key": str(row.get("join_key") or ""),
            },
        )
    point_in_time_only = future_event_count == 0 and invalid_timestamp_count == 0
    event_manifest_sha256 = _sha256_json(events)
    return {
        "timestamp_utc": now.isoformat(),
        "schema_version": 2,
        "ok": point_in_time_only,
        "overall_status": "ready" if point_in_time_only else "blocked_quarantined_events",
        "input_event_count": input_event_count,
        "event_count": int(len(events)),
        "category_counts": category_counts,
        "latest_by_category": latest_by_category,
        "events": events[: max(int(limit), 1)],
        "quarantined_event_count": len(quarantined_events),
        "quarantined_events": quarantined_events[:20],
        "point_in_time_contract": {
            "point_in_time_only": point_in_time_only,
            "future_event_count": future_event_count,
            "invalid_timestamp_count": invalid_timestamp_count,
            "quarantined_count": len(quarantined_events),
            "future_tolerance_seconds": max(float(future_tolerance_seconds), 0.0),
            "effective_cutoff_utc": datetime.fromtimestamp(future_cutoff, tz=timezone.utc).isoformat(),
            "source_manifest_sha256": source_manifest_sha256,
            "event_manifest_sha256": event_manifest_sha256,
            "join_policy": "event effective time must be less than or equal to the decision snapshot time",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a point-in-time normalized event store from recent event streams.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--out-file", default=str(PROJECT_ROOT / "governance" / "health" / "point_in_time_event_store_latest.json"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_event_store(Path(args.project_root).resolve(), limit=int(args.limit))
    out_path = Path(args.out_file)
    _write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(f"point_in_time_event_store events={int(payload.get('event_count', 0) or 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
