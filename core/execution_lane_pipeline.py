from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from core.accountability import safe_append_jsonl, safe_write_json_atomic
from core.channel_queue import ChannelMessage, ChannelQueue, default_queue_db_path
from core.profitability_hardening import (
    PAPER_EXECUTION_AUTHORITY_VERSION,
    evaluate_paper_execution_authority,
)


EXECUTION_INTENT_CHANNEL = "execution_intent"
EXECUTION_RESULT_CHANNEL = "execution_result"
EXECUTION_PROMOTION_CHANNEL = "execution_promotion"
EXECUTION_PROMOTED_CHANNEL = "execution_promoted"
EXECUTION_TRANSPORT_FEATURE_KEYS = frozenset(
    {
        "allocation_conflict_norm",
        "ask",
        "ask_price",
        "ask_size",
        "best_ask",
        "best_bid",
        "bid",
        "bid_price",
        "bid_size",
        "calendar_event_proximity_norm",
        "close_price",
        "entry_price",
        "last_price",
        "latency_ms",
        "mark_price",
        "news_source_quality_norm",
        "offer",
        "offer_price",
        "price",
        "queue_depth",
        "quote_age_ms",
        "spread_bps",
        "tradeability_score",
        "vol_30m",
        "volatility_1m",
    }
)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_ts(raw: Any) -> Optional[datetime]:
    text = str(raw or "").strip().replace("Z", "+00:00")
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) != 0.0
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_latest(project_root: str, name: str, payload: Dict[str, Any]) -> None:
    out = Path(project_root) / "governance" / "health" / name
    safe_write_json_atomic(
        str(out),
        payload,
        project_root=project_root,
        source=f"execution_lane_pipeline.{name}",
    )


def _execution_transport_payload(channel: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(payload or {})
    if str(channel or "") not in {EXECUTION_INTENT_CHANNEL, EXECUTION_PROMOTED_CHANNEL}:
        return row

    features = row.get("features") if isinstance(row.get("features"), dict) else {}
    retained = {key: value for key, value in features.items() if key in EXECUTION_TRANSPORT_FEATURE_KEYS}
    encoded_features = json.dumps(features, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    row["features"] = retained
    row["execution_transport"] = {
        "schema_version": 2,
        "compacted": len(retained) < len(features),
        "source_feature_count": len(features),
        "transport_feature_count": len(retained),
        "source_features_sha256": hashlib.sha256(encoded_features.encode("utf-8")).hexdigest(),
        "canonical_evidence_policy": "full_features_remain_in_source_decision_telemetry",
    }
    return row


def execution_lane_root(project_root: str | Path) -> Path:
    root = Path(project_root)
    override = os.getenv("EXECUTION_LANE_ROOT", "").strip()
    if override:
        return Path(override).expanduser()

    prefer_default = os.getenv("BOT_LOGS_PREFER_EXTERNAL", "1")
    if _env_flag("EXECUTION_LANE_PREFER_EXTERNAL", prefer_default):
        external_project = os.getenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", "").strip()
        if external_project:
            external_root = Path(external_project).expanduser()
        else:
            mount = Path(os.getenv("BOT_LOGS_EXTERNAL_MOUNT", "/Volumes/BOT_LOGS")).expanduser()
            project_dir = os.getenv("BOT_LOGS_EXTERNAL_PROJECT_DIR", "schwab_trading_bot").strip() or "schwab_trading_bot"
            external_root = mount / project_dir
        if external_root.exists() or _env_flag("EXECUTION_LANE_CREATE_EXTERNAL_ROOT", "1"):
            return external_root / "governance" / "execution_lanes"

    return root / "governance" / "execution_lanes"


def execution_lane_daily_path(project_root: str | Path, stem: str, *, day: str = "") -> str:
    stamp = str(day or datetime.now(timezone.utc).strftime("%Y%m%d"))
    base = execution_lane_root(project_root)
    return str(base / f"{stem}_{stamp}.jsonl")


def _execution_result_evidence(project_root: str, mode: str, now: datetime) -> Dict[str, Any]:
    max_rows = max(_safe_int(os.getenv("EXECUTION_LANE_HEALTH_RESULT_EVIDENCE_MAX_ROWS", "5000"), 5000), 100)
    freshness_seconds = max(
        _safe_float(os.getenv("EXECUTION_LANE_HEALTH_RESULT_FRESH_SECONDS", "900"), 900.0),
        60.0,
    )
    path = Path(execution_lane_daily_path(project_root, "execution_results"))
    evidence: Dict[str, Any] = {
        "path": str(path),
        "path_exists": path.exists(),
        "rows_scanned": 0,
        "mode_rows": 0,
        "stale_skip_rows": 0,
        "non_stale_rows": 0,
        "paper_executed_rows": 0,
        "latest_result_status": "",
        "latest_result_age_seconds": None,
        "latest_non_stale_status": "",
        "latest_non_stale_age_seconds": None,
        "latest_paper_executed_age_seconds": None,
        "freshness_seconds": float(freshness_seconds),
        "fresh_non_stale_activity": False,
        "fresh_paper_executed": False,
        "historical_stale_skip_only": False,
        "stale_skip_only": False,
        "activity_status": "missing_result_file",
    }
    if not path.exists() or not path.is_file():
        return evidence

    try:
        with path.open("r", encoding="utf-8") as handle:
            rows = deque(handle, maxlen=max_rows)
    except Exception as exc:
        evidence["activity_status"] = "result_file_unreadable"
        evidence["error_type"] = type(exc).__name__
        evidence["error"] = str(exc)
        return evidence

    latest_any_dt: datetime | None = None
    latest_non_stale_dt: datetime | None = None
    latest_executed_dt: datetime | None = None
    mode_text = str(mode or "").strip().lower()
    evidence["rows_scanned"] = len(rows)
    for line in rows:
        try:
            row = json.loads(str(line or "").strip())
        except Exception:
            continue
        if not isinstance(row, dict):
            continue
        row_mode = str(row.get("mode") or "").strip().lower()
        if mode_text and row_mode and row_mode != mode_text:
            continue
        status = str(row.get("result_status") or "").strip().upper()
        result = row.get("result") if isinstance(row.get("result"), dict) else {}
        result_reason = str(result.get("reason") or row.get("reason") or "").strip().lower()
        is_stale_skip = bool(status == "STALE_INTENT_SKIPPED" or result_reason == "stale_execution_intent")
        ts = _parse_ts(row.get("timestamp_utc"))

        evidence["mode_rows"] = int(evidence["mode_rows"]) + 1
        if ts is not None and (latest_any_dt is None or ts >= latest_any_dt):
            latest_any_dt = ts
            evidence["latest_result_status"] = status
        if is_stale_skip:
            evidence["stale_skip_rows"] = int(evidence["stale_skip_rows"]) + 1
            continue

        evidence["non_stale_rows"] = int(evidence["non_stale_rows"]) + 1
        if ts is not None and (latest_non_stale_dt is None or ts >= latest_non_stale_dt):
            latest_non_stale_dt = ts
            evidence["latest_non_stale_status"] = status
        if status == "PAPER_EXECUTED":
            evidence["paper_executed_rows"] = int(evidence["paper_executed_rows"]) + 1
            if ts is not None and (latest_executed_dt is None or ts >= latest_executed_dt):
                latest_executed_dt = ts

    if latest_any_dt is not None:
        evidence["latest_result_age_seconds"] = round(max((now - latest_any_dt).total_seconds(), 0.0), 3)
    if latest_non_stale_dt is not None:
        evidence["latest_non_stale_age_seconds"] = round(max((now - latest_non_stale_dt).total_seconds(), 0.0), 3)
    if latest_executed_dt is not None:
        evidence["latest_paper_executed_age_seconds"] = round(max((now - latest_executed_dt).total_seconds(), 0.0), 3)

    latest_non_stale_age = evidence["latest_non_stale_age_seconds"]
    latest_executed_age = evidence["latest_paper_executed_age_seconds"]
    evidence["fresh_non_stale_activity"] = bool(
        latest_non_stale_age is not None and float(latest_non_stale_age) <= freshness_seconds
    )
    evidence["fresh_paper_executed"] = bool(
        latest_executed_age is not None and float(latest_executed_age) <= freshness_seconds
    )
    latest_result_age = evidence["latest_result_age_seconds"]
    evidence["historical_stale_skip_only"] = bool(evidence["stale_skip_rows"] and not evidence["non_stale_rows"])
    evidence["stale_skip_only"] = bool(
        evidence["historical_stale_skip_only"]
        and latest_result_age is not None
        and float(latest_result_age) <= freshness_seconds
    )
    if evidence["fresh_paper_executed"]:
        evidence["activity_status"] = "fresh_paper_executed"
    elif evidence["fresh_non_stale_activity"]:
        evidence["activity_status"] = "fresh_non_stale_paper_activity"
    elif evidence["stale_skip_only"]:
        evidence["activity_status"] = "stale_skip_only"
    elif evidence["historical_stale_skip_only"]:
        evidence["activity_status"] = "old_stale_skip_audit_only"
    elif evidence["mode_rows"]:
        evidence["activity_status"] = "stale_or_old_paper_activity"
    else:
        evidence["activity_status"] = "no_mode_results"
    return evidence


def queue_db_path(project_root: str | Path, override: str = "") -> str:
    return str(override or default_queue_db_path(project_root))


def _enqueue_channel(
    *,
    project_root: str,
    channel: str,
    payload: Dict[str, Any],
    queue_db_override: str = "",
    source_path: str = "",
) -> str:
    queue = ChannelQueue(queue_db_path(project_root, queue_db_override))
    attempts = max(int(os.getenv("EXECUTION_LANE_QUEUE_ENQUEUE_RETRIES", "8") or 8), 1)
    base_sleep = max(float(os.getenv("EXECUTION_LANE_QUEUE_ENQUEUE_SLEEP_SECONDS", "0.25") or 0.25), 0.05)
    last_error = ""
    for attempt in range(attempts):
        try:
            return queue.enqueue(
                channel=channel,
                payload=payload,
                source_path=source_path,
                message_id=str(payload.get("message_id") or ""),
                parent_message_id=str(payload.get("parent_message_id") or ""),
                run_id=str(payload.get("run_id") or ""),
                iter_id=str(payload.get("iter_id") or ""),
            )
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower():
                raise
            last_error = str(exc)
            if attempt >= attempts - 1:
                break
            time.sleep(min(base_sleep * (attempt + 1), 2.0))

    safe_append_jsonl(
        execution_lane_daily_path(project_root, "execution_queue_enqueue_failures"),
        {
            "timestamp_utc": _now_utc(),
            "channel": channel,
            "message_id": str(payload.get("message_id") or ""),
            "source_path": source_path,
            "error": last_error or "sqlite_queue_enqueue_locked",
        },
        project_root=project_root,
        source="execution_lane_pipeline.enqueue_failure",
    )
    return str(payload.get("message_id") or "")


def publish_channel_payload(
    *,
    project_root: str,
    channel: str,
    payload: Dict[str, Any],
    stem: str,
    queue_db_override: str = "",
) -> Dict[str, Any]:
    row = _execution_transport_payload(channel, payload)
    row.setdefault("timestamp_utc", _now_utc())
    out_path = execution_lane_daily_path(project_root, stem)
    safe_append_jsonl(
        out_path,
        row,
        project_root=project_root,
        source=f"execution_lane_pipeline.{channel}",
    )
    message_id = _enqueue_channel(
        project_root=project_root,
        channel=channel,
        payload=row,
        queue_db_override=queue_db_override,
        source_path=out_path,
    )
    row["message_id"] = str(message_id or row.get("message_id") or "")
    return row


def publish_execution_intent(
    *,
    project_root: str,
    payload: Dict[str, Any],
    queue_db_override: str = "",
) -> Dict[str, Any]:
    row = dict(payload or {})
    row.setdefault("target_mode", "paper")
    row.setdefault("source_mode", "shadow")
    row.setdefault("intent_kind", "master")
    return publish_channel_payload(
        project_root=project_root,
        channel=EXECUTION_INTENT_CHANNEL,
        payload=row,
        stem="execution_intents",
        queue_db_override=queue_db_override,
    )


def publish_execution_result(
    *,
    project_root: str,
    payload: Dict[str, Any],
    queue_db_override: str = "",
) -> Dict[str, Any]:
    return publish_channel_payload(
        project_root=project_root,
        channel=EXECUTION_RESULT_CHANNEL,
        payload=payload,
        stem="execution_results",
        queue_db_override=queue_db_override,
    )


def publish_execution_promotion(
    *,
    project_root: str,
    payload: Dict[str, Any],
    queue_db_override: str = "",
) -> Dict[str, Any]:
    return publish_channel_payload(
        project_root=project_root,
        channel=EXECUTION_PROMOTION_CHANNEL,
        payload=payload,
        stem="execution_promotions",
        queue_db_override=queue_db_override,
    )


def publish_promoted_execution_intent(
    *,
    project_root: str,
    payload: Dict[str, Any],
    queue_db_override: str = "",
) -> Dict[str, Any]:
    row = dict(payload or {})
    row["target_mode"] = "live"
    return publish_channel_payload(
        project_root=project_root,
        channel=EXECUTION_PROMOTED_CHANNEL,
        payload=row,
        stem="execution_promoted",
        queue_db_override=queue_db_override,
    )


def emit_paper_reconciliation_heartbeat(
    *,
    project_root: str,
    trader: Any,
    last_emit_monotonic: float = 0.0,
    min_interval_seconds: float = 180.0,
    reason: str = "execution_lane_heartbeat",
) -> float:
    now_mono = time.monotonic()
    if last_emit_monotonic > 0.0 and (now_mono - float(last_emit_monotonic)) < max(float(min_interval_seconds), 0.0):
        return float(last_emit_monotonic)

    guard = getattr(trader, "live_guard", None)
    status = "guard_unavailable"
    reconciliation: Dict[str, Any] = {"ok": False, "error": "live_guard_unavailable"}
    if guard is not None and hasattr(guard, "reconcile_order_lifecycle"):
        try:
            raw = guard.reconcile_order_lifecycle(broker_open_orders=[])
            reconciliation = dict(raw) if isinstance(raw, dict) else {"ok": False, "raw": raw}
            status = "ok" if bool(reconciliation.get("ok", False)) else "mismatch"
        except Exception as exc:
            reconciliation = {"ok": False, "error": str(exc)}
            status = "error"

    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_path = Path(project_root) / "governance" / "events" / f"paper_execution_guard_{day}.jsonl"
    safe_append_jsonl(
        str(out_path),
        {
            "timestamp_utc": _now_utc(),
            "event": "paper_order_lifecycle_reconcile",
            "status": status,
            "mode": str(getattr(trader, "mode_label", getattr(trader, "mode", "paper")) or "paper"),
            "account_hash": str(getattr(trader, "live_account_hash", "") or ""),
            "details": {
                "heartbeat": True,
                "reason": str(reason or "execution_lane_heartbeat"),
                "broker_open_orders_total": 0,
                "order_lifecycle_reconcile": reconciliation,
            },
        },
        project_root=project_root,
        source="execution_lane_pipeline.paper_reconciliation_heartbeat",
    )
    return now_mono


def _registry_rows(project_root: str) -> dict[str, Dict[str, Any]]:
    registry_path = Path(project_root) / "master_bot_registry.json"
    registry = _read_json(registry_path)
    rows = registry.get("sub_bots") if isinstance(registry.get("sub_bots"), list) else []
    out: dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        bot_id = str(row.get("bot_id") or "").strip()
        if not bot_id:
            continue
        out[bot_id] = row
    return out


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _paper_standard_registry_rows(
    project_root: str,
) -> tuple[dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Load the generated paper cohort only when its source and candidate hashes match."""

    root = Path(project_root)
    source_path = root / "master_bot_registry.json"
    source = _read_json(source_path)
    source_rows = _registry_rows(project_root)
    candidate_path = root / "governance" / "health" / "paper_live_data_standard_registry_candidate_latest.json"
    guard_path = root / "governance" / "health" / "paper_live_data_standard_source_write_guard_latest.json"
    health_path = root / "governance" / "health" / "paper_live_data_standard_latest.json"
    candidate = _read_json(candidate_path)
    guard = _read_json(guard_path)
    health = _read_json(health_path)
    reasons: list[str] = []

    if not candidate:
        reasons.append("candidate_registry_missing")
    if not guard:
        reasons.append("source_write_guard_missing")
    if not health:
        reasons.append("paper_standard_health_missing")
    elif not bool(health.get("ok", False)):
        reasons.append("paper_standard_health_not_ok")
    if guard and not bool(guard.get("source_write_blocked", False)):
        reasons.append("source_write_guard_not_active")
    guarded_source_path = str(guard.get("source_path") or "").strip()
    if guarded_source_path and Path(guarded_source_path).resolve() != source_path.resolve():
        reasons.append("source_registry_path_mismatch")
    guarded_candidate_path = str(guard.get("candidate_path") or "").strip()
    if not guarded_candidate_path or Path(guarded_candidate_path).resolve() != candidate_path.resolve():
        reasons.append("candidate_registry_path_mismatch")
    source_sha256 = _sha256_file(source_path)
    candidate_sha256 = _sha256_file(candidate_path)
    if source_sha256 != str(guard.get("source_sha256") or ""):
        reasons.append("source_registry_hash_mismatch")
    if candidate_sha256 != str(guard.get("candidate_sha256") or ""):
        reasons.append("candidate_registry_hash_mismatch")

    summary = candidate.get("summary") if isinstance(candidate.get("summary"), dict) else {}
    if summary.get("paper_live_data_standard_version") != "paper_live_data_standard_v2":
        reasons.append("candidate_registry_version_mismatch")
    candidate_list = candidate.get("sub_bots") if isinstance(candidate.get("sub_bots"), list) else []
    source_list = source.get("sub_bots") if isinstance(source.get("sub_bots"), list) else []
    source_ids = {
        str(row.get("bot_id") or "").strip()
        for row in source_list
        if isinstance(row, dict) and str(row.get("bot_id") or "").strip()
    }
    candidate_ids = {
        str(row.get("bot_id") or "").strip()
        for row in candidate_list
        if isinstance(row, dict) and str(row.get("bot_id") or "").strip()
    }
    if len(source_list) != len(candidate_list) or source_ids != candidate_ids:
        reasons.append("candidate_registry_membership_mismatch")

    if reasons:
        return source_rows, {
            "source": "canonical_registry",
            "candidate_overlay_valid": False,
            "reasons": reasons,
        }

    rows: dict[str, Dict[str, Any]] = {}
    for row in candidate_list:
        if not isinstance(row, dict):
            continue
        bot_id = str(row.get("bot_id") or "").strip()
        if bot_id:
            rows[bot_id] = row
    return rows, {
        "source": "hash_bound_candidate_overlay",
        "candidate_overlay_valid": True,
        "source_sha256": source_sha256,
        "candidate_sha256": candidate_sha256,
        "reasons": [],
    }


def _execution_gateway_paths(project_root: str) -> tuple[Path, Path]:
    root = Path(project_root)
    return (
        root / "governance" / "allocator" / "portfolio_allocator_service_latest.json",
        root / "governance" / "risk" / "risk_service_boundary_latest.json",
    )


def _intent_side(intent: Dict[str, Any]) -> str:
    return str(intent.get("action") or intent.get("side") or "").strip().upper()


def _pre_trade_match(rows: list[Dict[str, Any]], *, symbol: str, side: str) -> Dict[str, Any]:
    symbol_upper = str(symbol or "").strip().upper()
    side_upper = str(side or "").strip().upper()
    for row in rows:
        row_symbol = str(row.get("symbol") or "").strip().upper()
        requested_action = str(row.get("requested_action") or row.get("side") or "").strip().upper()
        approved_action = str(row.get("approved_action") or "").strip().upper()
        if row_symbol != symbol_upper:
            continue
        if requested_action and side_upper and requested_action != side_upper:
            continue
        if approved_action and approved_action not in {"HOLD", "REJECT"}:
            return row
        if bool(row.get("risk_limit_ok", False)):
            return row
    return {}


def _allocator_match(rows: list[Dict[str, Any]], *, symbol: str, side: str) -> Dict[str, Any]:
    symbol_upper = str(symbol or "").strip().upper()
    side_upper = str(side or "").strip().upper()
    for row in rows:
        row_symbol = str(row.get("symbol") or "").strip().upper()
        row_side = str(row.get("side") or "").strip().upper()
        approved_qty = _safe_float(row.get("approved_qty"), 0.0)
        if row_symbol != symbol_upper or approved_qty <= 0.0:
            continue
        if row_side and side_upper and row_side != side_upper:
            continue
        return row
    return {}


def evaluate_execution_gateway(
    *,
    project_root: str,
    intent: Dict[str, Any],
    mode: str,
) -> Dict[str, Any]:
    allocator_path, risk_path = _execution_gateway_paths(project_root)
    allocator = _read_json(allocator_path)
    risk_boundary = _read_json(risk_path)
    approved_rows = allocator.get("approved_intents") if isinstance(allocator.get("approved_intents"), list) else []
    pre_trade_rows = risk_boundary.get("pre_trade_decisions") if isinstance(risk_boundary.get("pre_trade_decisions"), list) else []
    symbol = str(intent.get("symbol") or "").strip().upper()
    side = _intent_side(intent)
    allocator_match = _allocator_match(approved_rows, symbol=symbol, side=side)
    pre_trade_match = _pre_trade_match(pre_trade_rows, symbol=symbol, side=side)

    reasons: list[str] = []
    allocator_ok = bool(allocator.get("ok", False))
    risk_ok = bool(risk_boundary.get("ok", False))
    if not allocator:
        reasons.append("allocator_contract_missing")
    elif not allocator_ok:
        reasons.append("allocator_contract_not_ok")
    if not risk_boundary:
        reasons.append("risk_boundary_missing")
    elif not risk_ok:
        reasons.append("risk_boundary_not_ok")
    if mode.strip().lower() == "live":
        if not allocator_match:
            reasons.append("allocator_missing_matching_intent")
        if not pre_trade_match:
            reasons.append("risk_boundary_missing_pretrade_match")

    allow_execute = True
    if mode.strip().lower() == "live":
        allow_execute = len(reasons) == 0

    approved_action = str(pre_trade_match.get("approved_action") or "").strip().upper()
    return {
        "timestamp_utc": _now_utc(),
        "contract_version": 1,
        "mode": str(mode),
        "symbol": symbol,
        "side": side,
        "allow_execute": bool(allow_execute),
        "allocator_ok": allocator_ok,
        "risk_boundary_ok": risk_ok,
        "allocator_match_found": bool(allocator_match),
        "pre_trade_match_found": bool(pre_trade_match),
        "approved_action": approved_action,
        "reasons": reasons,
        "source_files": {
            "allocator": str(allocator_path),
            "risk_boundary": str(risk_path),
        },
    }


def _extract_bot_id(intent: Dict[str, Any]) -> str:
    metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
    candidates = [
        metadata.get("bot_id"),
        intent.get("bot_id"),
    ]
    strategy = str(intent.get("strategy") or "").strip()
    if "::" in strategy:
        candidates.append(strategy.split("::", 1)[1].strip())
    for raw in candidates:
        bot_id = str(raw or "").strip()
        if bot_id:
            return bot_id
    return ""


def _paper_standard_segment(intent: Dict[str, Any], row: Dict[str, Any]) -> str:
    metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
    declared = str(metadata.get("signal_segment") or "").strip().lower()
    if declared in {"core", "options", "futures"}:
        return declared
    role = str(row.get("bot_role") or metadata.get("bot_role") or intent.get("bot_role") or "").strip().lower()
    if role == "options_sub_bot":
        return "options"
    if role == "futures_sub_bot":
        return "futures"
    return "core"


def _paper_registry_authority(row: Dict[str, Any], intent: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(row or {})
    materialization = (
        normalized.get("training_label_materialization_contract")
        if isinstance(normalized.get("training_label_materialization_contract"), dict)
        else {}
    )
    label_contract = normalized.get("label_contract") if isinstance(normalized.get("label_contract"), dict) else {}
    normalized.setdefault("training_objective_class", materialization.get("objective_class"))
    normalized.setdefault(
        "label_family",
        materialization.get("label_family") or label_contract.get("label_family"),
    )
    verdict = evaluate_paper_execution_authority(
        normalized,
        segment=_paper_standard_segment(intent, normalized),
        minimum_accuracy=max(_safe_float(os.getenv("PAPER_EXECUTION_AUTHORITY_MIN_ACC", "0.56"), 0.56), 0.0),
        minimum_quality_score=max(
            _safe_float(os.getenv("PAPER_EXECUTION_AUTHORITY_MIN_QUALITY", "0.50"), 0.50),
            0.0,
        ),
    )
    reasons = list(verdict.get("reasons") or [])
    authority_version = str(normalized.get("paper_execution_authority_version") or "").strip()
    if authority_version != PAPER_EXECUTION_AUTHORITY_VERSION:
        reasons.append("paper_execution_authority_version_mismatch")
    if bool(normalized.get("direct_execution_allowed", False)):
        reasons.append("direct_execution_authority_present")
    if bool(normalized.get("live_trading_enabled", False)):
        reasons.append("live_execution_authority_present")
    verdict["allowed"] = not reasons
    verdict["reasons"] = reasons
    verdict["declared_authority_version"] = authority_version
    return verdict


def _candidate_identity_reasons(project_root: str, metadata: Dict[str, Any]) -> list[str]:
    candidate_state = _read_json(Path(project_root) / "governance" / "runtime" / "production_candidate_state.json")
    expected_candidate_id = str(candidate_state.get("candidate_id") or "").strip()
    if not expected_candidate_id:
        return []
    actual_candidate_id = str(metadata.get("production_candidate_id") or "").strip()
    if not actual_candidate_id:
        return ["paper_standard_production_candidate_id_missing"]
    if actual_candidate_id != expected_candidate_id:
        return ["paper_standard_production_candidate_id_mismatch"]
    return []


def evaluate_paper_standard_gateway(*, project_root: str, intent: Dict[str, Any]) -> Dict[str, Any]:
    enabled = _env_flag("PAPER_LIVE_DATA_STANDARD_ENABLED", "0")
    bot_id = _extract_bot_id(intent)
    if not enabled:
        return {
            "enabled": False,
            "allow_execute": True,
            "bot_id": bot_id,
            "reasons": [],
        }

    metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
    constituent_ids = sorted(
        {
            str(item or "").strip()
            for item in metadata.get("constituent_bot_ids", [])
            if str(item or "").strip()
        }
    )
    is_portfolio_consensus = str(metadata.get("layer") or "").strip().lower() == "paper_portfolio_consensus"
    if is_portfolio_consensus:
        reasons: list[str] = []
        invalid_ids: list[str] = []
        authority_failures: dict[str, list[str]] = {}
        registry_rows, registry_provenance = _paper_standard_registry_rows(project_root)
        if not constituent_ids:
            reasons.append("paper_standard_consensus_missing_constituents")
        if len(constituent_ids) < 2:
            reasons.append("paper_standard_consensus_constituent_count_below_diversity_floor")
        if _safe_int(metadata.get("constituent_count"), len(constituent_ids)) != len(constituent_ids):
            reasons.append("paper_standard_consensus_constituent_count_mismatch")
        if bool(metadata.get("constituent_bot_ids_truncated", False)):
            reasons.append("paper_standard_consensus_constituents_truncated")
        if str(metadata.get("paper_execution_authority_version") or "") != PAPER_EXECUTION_AUTHORITY_VERSION:
            reasons.append("paper_standard_consensus_authority_version_mismatch")
        if not bool(metadata.get("paper_execution_diversity_ready", False)):
            reasons.append("paper_standard_consensus_diversity_not_ready")
        distinct_clusters = _safe_int(metadata.get("paper_execution_distinct_correlation_clusters"), 0)
        if distinct_clusters < 2:
            reasons.append("paper_standard_consensus_correlation_diversity_below_floor")
        if distinct_clusters > len(constituent_ids):
            reasons.append("paper_standard_consensus_correlation_diversity_invalid")

        expected_ids_sha256 = hashlib.sha256(
            json.dumps(constituent_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if str(metadata.get("constituent_bot_ids_sha256") or "") != expected_ids_sha256:
            reasons.append("paper_standard_consensus_constituent_hash_mismatch")
        manifest = (
            metadata.get("paper_execution_cohort_manifest")
            if isinstance(metadata.get("paper_execution_cohort_manifest"), dict)
            else {}
        )
        manifest_sha256 = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if not manifest or str(metadata.get("paper_execution_cohort_manifest_sha256") or "") != manifest_sha256:
            reasons.append("paper_standard_consensus_manifest_hash_mismatch")
        if str(manifest.get("policy") or "") != PAPER_EXECUTION_AUTHORITY_VERSION:
            reasons.append("paper_standard_consensus_manifest_policy_mismatch")
        manifest_ids = sorted(
            str(item or "").strip()
            for item in manifest.get("constituent_bot_ids", [])
            if str(item or "").strip()
        )
        if manifest_ids != constituent_ids:
            reasons.append("paper_standard_consensus_manifest_membership_mismatch")
        if str(manifest.get("segment") or "").strip().lower() != str(
            metadata.get("signal_segment") or ""
        ).strip().lower():
            reasons.append("paper_standard_consensus_manifest_segment_mismatch")
        if str(manifest.get("profile") or "").strip().lower() != str(
            metadata.get("source_profile") or ""
        ).strip().lower():
            reasons.append("paper_standard_consensus_manifest_profile_mismatch")
        reasons.extend(_candidate_identity_reasons(project_root, metadata))
        for constituent_id in constituent_ids:
            registry_row = registry_rows.get(constituent_id, {})
            authority = _paper_registry_authority(registry_row, intent)
            if not bool(authority.get("allowed", False)):
                invalid_ids.append(constituent_id)
                authority_failures[constituent_id] = list(authority.get("reasons") or [])
        if invalid_ids:
            reasons.append("paper_standard_consensus_contains_ineligible_bot")
        return {
            "enabled": True,
            "allow_execute": len(reasons) == 0,
            "bot_id": "paper_portfolio_consensus",
            "consensus_constituent_count": len(constituent_ids),
            "consensus_invalid_bot_ids": invalid_ids[:64],
            "consensus_invalid_bot_ids_truncated": len(invalid_ids) > 64,
            "consensus_authority_failures": authority_failures,
            "registry_provenance": registry_provenance,
            "reasons": reasons,
        }

    reasons = []
    registry_row: Dict[str, Any] = {}
    authority: Dict[str, Any] = {}
    registry_rows, registry_provenance = _paper_standard_registry_rows(project_root)
    if not bot_id:
        reasons.append("paper_standard_missing_bot_id")
    else:
        registry_row = registry_rows.get(bot_id, {})
        if not registry_row:
            reasons.append("paper_standard_bot_missing_from_registry")
        else:
            authority = _paper_registry_authority(registry_row, intent)
            if not bool(authority.get("allowed", False)):
                reasons.append("paper_standard_bot_not_in_explicit_paper_cohort")
        reasons.extend(_candidate_identity_reasons(project_root, metadata))

    return {
        "enabled": True,
        "allow_execute": len(reasons) == 0,
        "bot_id": bot_id,
        "virtual_allowed": False,
        "paper_standard_cohort": str(registry_row.get("paper_standard_cohort") or "") if registry_row else "",
        "paper_live_data_enabled": bool(registry_row.get("paper_live_data_enabled", False)) if registry_row else None,
        "paper_execution_authority": authority,
        "registry_provenance": registry_provenance,
        "reasons": reasons,
    }


def evaluate_live_promotion(
    *,
    project_root: str,
    intent: Dict[str, Any],
    paper_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
    intent_kind = str(intent.get("intent_kind") or metadata.get("intent_kind") or "master").strip().lower()
    lane = str(metadata.get("runtime_lane") or metadata.get("lane") or "default").strip().lower() or "default"
    bot_id = _extract_bot_id(intent)
    reasons: list[str] = []

    if not _safe_bool(metadata.get("allow_live_promotion", intent_kind == "master")):
        reasons.append("intent_marked_paper_only")

    action = str(intent.get("action") or "").strip().upper()
    if action not in {"BUY", "SELL", "SELL_SHORT", "BUY_TO_COVER", "BUY_TO_OPEN", "BUY_TO_CLOSE", "SELL_TO_OPEN", "SELL_TO_CLOSE", "CLOSE", "ROLL"}:
        reasons.append("non_trade_action")

    if paper_result is not None:
        result_status = str(paper_result.get("status") or "").strip().upper()
        if result_status != "PAPER_EXECUTED":
            reasons.append(f"paper_status_not_executed:{result_status or 'unknown'}")
        paper_order = paper_result.get("paper_order") if isinstance(paper_result.get("paper_order"), dict) else {}
        realism_status = str(paper_order.get("paper_realism_status") or "").strip()
        filled_quantity = _safe_float(paper_order.get("filled_quantity"), _safe_float(paper_order.get("quantity"), 0.0))
        if realism_status and realism_status not in {"filled", "full_fill", "partial_fill"}:
            reasons.append(f"paper_realism_not_filled:{realism_status}")
        elif filled_quantity <= 0.0 and result_status == "PAPER_EXECUTED":
            reasons.append("paper_realism_not_filled:zero_fill")
        min_realism_score = _safe_float(os.getenv("PAPER_REALISM_MIN_PROMOTION_SCORE", "0"), 0.0)
        realism_score = _safe_float(paper_order.get("paper_realism_score"), 100.0)
        if min_realism_score > 0.0 and realism_score < min_realism_score:
            reasons.append("paper_realism_quality_below_threshold")

    gateway = evaluate_execution_gateway(
        project_root=project_root,
        intent=intent,
        mode="live",
    )
    if not bool(gateway.get("allow_execute", False)):
        reasons.append("execution_gateway_blocked")

    promotion_gate = _read_json(Path(project_root) / "governance" / "walk_forward" / "promotion_gate_latest.json")
    lane_gate = _read_json(Path(project_root) / "governance" / "walk_forward" / "lane_promotion_gate_latest.json")
    quality_gate = _read_json(Path(project_root) / "governance" / "health" / "promotion_quality_gate_latest.json")

    if not bool(promotion_gate.get("promote_ok", False)):
        reasons.append("promotion_gate_blocked")
    if not bool(lane_gate.get("promote_ok", False)):
        reasons.append("lane_promotion_gate_blocked")
    if not bool(lane_gate.get("coverage_ok", False)):
        reasons.append("lane_promotion_coverage_missing")
    if not bool(quality_gate.get("ok", False)):
        reasons.append("promotion_quality_gate_blocked")

    lane_payload = lane_gate.get("lanes") if isinstance(lane_gate.get("lanes"), dict) else {}
    lane_detail = lane_payload.get(lane) if isinstance(lane_payload.get(lane), dict) else {}
    if lane_detail:
        if not bool(lane_detail.get("promote_ok", False)):
            reasons.append(f"lane_blocked:{lane}")
        if not bool(lane_detail.get("coverage_ok", False)):
            reasons.append(f"lane_uncovered:{lane}")

    registry_row = {}
    if bot_id:
        registry_row = _registry_rows(project_root).get(bot_id, {})
        if not registry_row:
            reasons.append("bot_missing_from_registry")
        else:
            if not bool(registry_row.get("active", False)):
                reasons.append("bot_inactive_in_registry")
            if bool(registry_row.get("deleted_from_rotation", False)):
                reasons.append("bot_deleted_from_rotation")
            if bool(registry_row.get("training_excluded", False) or registry_row.get("exclude_from_training", False)):
                reasons.append("bot_training_or_quality_excluded")
            promotion_status = str(registry_row.get("promotion_status") or "").strip().lower()
            if promotion_status and promotion_status not in {"live", "live_ready", "promoted"}:
                reasons.append(f"bot_promotion_status_not_live:{promotion_status}")
            if not bool(registry_row.get("promoted", False)):
                reasons.append("bot_not_promoted")

    promote_ok = len(reasons) == 0
    return {
        "timestamp_utc": _now_utc(),
        "promote_ok": bool(promote_ok),
        "intent_kind": intent_kind,
        "lane": lane,
        "bot_id": bot_id,
        "reasons": reasons,
        "gate_snapshot": {
            "promotion_gate": {
                "promote_ok": bool(promotion_gate.get("promote_ok", False)),
                "coverage_ok": bool(promotion_gate.get("coverage_ok", False)),
                "considered_bots": _safe_int(promotion_gate.get("considered_bots"), 0),
            },
            "lane_promotion_gate": {
                "promote_ok": bool(lane_gate.get("promote_ok", False)),
                "coverage_ok": bool(lane_gate.get("coverage_ok", False)),
                "lane_detail": lane_detail,
            },
            "promotion_quality_gate": {
                "ok": bool(quality_gate.get("ok", False)),
                "failed_checks": quality_gate.get("failed_checks", []),
            },
        },
        "registry_row": {
            "active": bool(registry_row.get("active", False)) if registry_row else None,
            "promoted": bool(registry_row.get("promoted", False)) if registry_row else None,
            "deleted_from_rotation": bool(registry_row.get("deleted_from_rotation", False)) if registry_row else None,
        },
        "execution_gateway": gateway,
    }


def configure_trader_for_lane(trader: Any, mode: str) -> Any:
    trader.set_mode(mode)
    trader.execution_enabled = True
    trader.market_data_only = False
    return trader


def intent_to_decision_kwargs(intent: Dict[str, Any]) -> Dict[str, Any]:
    metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
    features = intent.get("features") if isinstance(intent.get("features"), dict) else {}
    gates = intent.get("gates") if isinstance(intent.get("gates"), dict) else {}
    reasons = intent.get("reasons") if isinstance(intent.get("reasons"), list) else []
    return {
        "symbol": str(intent.get("symbol") or ""),
        "action": str(intent.get("action") or "HOLD"),
        "quantity": _safe_float(intent.get("quantity"), 0.0),
        "model_score": _safe_float(intent.get("model_score"), 0.5),
        "threshold": _safe_float(intent.get("threshold"), 0.55),
        "features": features,
        "gates": gates,
        "reasons": [str(r) for r in reasons],
        "strategy": str(intent.get("strategy") or metadata.get("strategy") or "execution_lane"),
        "metadata": metadata,
    }


def _annotate_paper_realism(intent: Dict[str, Any], result: Dict[str, Any]) -> None:
    if str(result.get("status") or "").strip().upper() != "PAPER_EXECUTED":
        return
    paper_order = result.get("paper_order")
    if not isinstance(paper_order, dict):
        return

    metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
    features = intent.get("features") if isinstance(intent.get("features"), dict) else {}
    asset_class = str(metadata.get("asset_class") or intent.get("asset_class") or "").strip().lower()
    quantity = _safe_float(paper_order.get("quantity", intent.get("quantity")), 0.0)
    quote_age_ms = max(
        _safe_float(features.get("quote_age_ms"), 0.0),
        _safe_float(metadata.get("quote_age_ms"), 0.0),
    )
    max_quote_age_ms = max(_safe_float(os.getenv("PAPER_REALISM_MAX_QUOTE_AGE_MS", "5000"), 5000.0), 100.0)

    if asset_class == "options" and quote_age_ms > max_quote_age_ms:
        paper_order["paper_realism_status"] = "stale_quote_rejected"
        paper_order["paper_realism_score"] = 0.0
        paper_order["filled_quantity"] = 0.0
        paper_order["paper_realism_reason"] = "option_quote_age_above_limit"
        paper_order["paper_realism_quote_age_ms"] = float(quote_age_ms)
        paper_order["paper_realism_max_quote_age_ms"] = float(max_quote_age_ms)
        return

    paper_order.setdefault("paper_realism_status", "filled")
    paper_order.setdefault("paper_realism_score", 100.0)
    paper_order.setdefault("filled_quantity", float(quantity))


def process_execution_intent(
    *,
    project_root: str,
    trader: Any,
    mode: str,
    message: ChannelMessage,
    queue_db_override: str = "",
) -> Dict[str, Any]:
    intent = dict(message.payload or {})
    kwargs = intent_to_decision_kwargs(intent)
    paper_standard_gateway: Dict[str, Any] = {}
    if str(mode).strip().lower() == "paper":
        paper_standard_gateway = evaluate_paper_standard_gateway(
            project_root=project_root,
            intent=intent,
        )
    gateway = evaluate_execution_gateway(
        project_root=project_root,
        intent=intent,
        mode=mode,
    )
    if str(mode).strip().lower() == "paper" and not bool(paper_standard_gateway.get("allow_execute", True)):
        result = {
            "status": "PAPER_STANDARD_BLOCKED",
            "reason": "paper_live_data_standard_blocked",
            "paper_standard_gateway": paper_standard_gateway,
        }
    elif str(mode).strip().lower() == "live" and not bool(gateway.get("allow_execute", False)):
        result = {
            "status": "LIVE_GATEWAY_BLOCKED",
            "reason": "execution_gateway_blocked",
            "execution_gateway": gateway,
        }
    else:
        result = trader.execute_decision(**kwargs)
    if str(mode).strip().lower() == "paper":
        _annotate_paper_realism(intent, result)

    result_payload = {
        "timestamp_utc": _now_utc(),
        "mode": str(mode),
        "consumer": f"execution_lane_{mode}",
        "intent_channel": str(message.channel),
        "intent_message_id": str(message.message_id),
        "intent_created_at": str(message.created_at),
        "intent": intent,
        "result_status": str(result.get("status") or ""),
        "result": result,
        "execution_gateway": gateway,
        "paper_standard_gateway": paper_standard_gateway,
    }
    publish_execution_result(
        project_root=project_root,
        payload=result_payload,
        queue_db_override=queue_db_override,
    )

    promotion_payload = {}
    if str(mode).strip().lower() == "paper":
        promotion = evaluate_live_promotion(
            project_root=project_root,
            intent=intent,
            paper_result=result,
        )
        promotion_payload = {
            "timestamp_utc": _now_utc(),
            "intent_message_id": str(message.message_id),
            "intent_channel": str(message.channel),
            "intent": intent,
            "paper_result_status": str(result.get("status") or ""),
            "promotion": promotion,
            "execution_gateway": gateway,
        }
        publish_execution_promotion(
            project_root=project_root,
            payload=promotion_payload,
            queue_db_override=queue_db_override,
        )
        if bool(promotion.get("promote_ok", False)):
            promoted_payload = {
                **intent,
                "timestamp_utc": _now_utc(),
                "source_intent_message_id": str(message.message_id),
                "promotion": promotion,
                "target_mode": "live",
                "parent_message_id": str(message.message_id),
            }
            publish_promoted_execution_intent(
                project_root=project_root,
                payload=promoted_payload,
                queue_db_override=queue_db_override,
            )

    return {
        "result": result_payload,
        "promotion": promotion_payload,
    }


def update_lane_health(
    *,
    project_root: str,
    mode: str,
    processed_count: int,
    queue_channel: str,
    queue_db_override: str = "",
    auth_ok: Optional[bool] = None,
    auth_error: str = "",
) -> None:
    consumer = f"execution_lane_{mode}"
    now = datetime.now(timezone.utc)
    queue_stats: dict[str, Any] = {}
    consumer_state: dict[str, Any] = {}
    pending_rows = 0
    pending_rows_unknown = False
    queue_stats_available = False
    queue_stats_status = "skipped"
    queue_stats_skip_reason = ""
    queue_stats_error_type = ""
    queue_stats_error = ""

    if _env_flag("EXECUTION_LANE_HEALTH_QUEUE_STATS_ENABLED", "0"):
        try:
            q = ChannelQueue(queue_db_path(project_root, queue_db_override))
            queue_stats = q.queue_stats(channel=queue_channel)
            consumer_state = q.consumer_state(consumer=consumer, channel=queue_channel)
            pending_rows = q.pending_count(consumer=consumer, channel=queue_channel)
            queue_stats_available = True
            queue_stats_status = "ready"
        except Exception as exc:
            pending_rows_unknown = True
            queue_stats_available = False
            queue_stats_status = "error"
            queue_stats_error_type = type(exc).__name__
            queue_stats_error = str(exc)
    else:
        pending_rows_unknown = True
        queue_stats_skip_reason = "disabled_for_nonblocking_execution_lane_heartbeat"

    queue_oldest_dt = _parse_ts((queue_stats or {}).get("oldest_created_at"))
    queue_newest_dt = _parse_ts((queue_stats or {}).get("newest_created_at"))
    consumer_updated_dt = _parse_ts((consumer_state or {}).get("updated_at"))
    queue_oldest_age_seconds = (
        round(max((now - queue_oldest_dt).total_seconds(), 0.0), 3)
        if queue_oldest_dt is not None
        else None
    )
    queue_newest_age_seconds = (
        round(max((now - queue_newest_dt).total_seconds(), 0.0), 3)
        if queue_newest_dt is not None
        else None
    )
    consumer_idle_seconds = (
        round(max((now - consumer_updated_dt).total_seconds(), 0.0), 3)
        if consumer_updated_dt is not None
        else None
    )
    stale_after_seconds = max(int(os.getenv("EXECUTION_LANE_STALE_AFTER_SECONDS", "180") or 180), 30)
    stale_grace_seconds = 0
    known_pending_rows = 0 if pending_rows_unknown else int(pending_rows)
    if known_pending_rows > 0:
        # Large active queues naturally create short idle gaps between acks; don't
        # label the lane stale while fresh intents are still flowing in.
        if queue_newest_age_seconds is not None and float(queue_newest_age_seconds) <= max(float(stale_after_seconds), 300.0):
            stale_grace_seconds += int(stale_after_seconds)
        backlog_scale = min(max(known_pending_rows // 25000, 0), 10)
        stale_grace_seconds += int(backlog_scale * 60)
    effective_stale_after_seconds = int(stale_after_seconds + stale_grace_seconds)
    stale = bool(
        known_pending_rows > 0
        and consumer_idle_seconds is not None
        and float(consumer_idle_seconds) >= float(effective_stale_after_seconds)
    )
    payload = {
        "timestamp_utc": _now_utc(),
        "mode": str(mode),
        "consumer": consumer,
        "processed_count": int(processed_count),
        "queue_channel": str(queue_channel),
        "queue_stats_available": bool(queue_stats_available),
        "queue_stats_status": queue_stats_status,
        "queue_stats_skip_reason": queue_stats_skip_reason,
        "queue_stats_error_type": queue_stats_error_type,
        "queue_stats_error": queue_stats_error,
        "queue_stats": queue_stats,
        "consumer_state": consumer_state,
        "pending_rows": int(known_pending_rows),
        "pending_rows_unknown": bool(pending_rows_unknown),
        "queue_oldest_age_seconds": queue_oldest_age_seconds,
        "queue_newest_age_seconds": queue_newest_age_seconds,
        "consumer_idle_seconds": consumer_idle_seconds,
        "stale_after_seconds": int(effective_stale_after_seconds),
        "stale_base_seconds": int(stale_after_seconds),
        "stale_grace_seconds": int(stale_grace_seconds),
        "stale": bool(stale),
    }
    allocator_path, risk_path = _execution_gateway_paths(project_root)
    allocator = _read_json(allocator_path)
    risk_boundary = _read_json(risk_path)
    pre_trade_rows = risk_boundary.get("pre_trade_decisions") if isinstance(risk_boundary.get("pre_trade_decisions"), list) else []
    payload["execution_gateway"] = {
        "allocator_ok": bool(allocator.get("ok", False)),
        "risk_boundary_ok": bool(risk_boundary.get("ok", False)),
        "approved_intents": len(allocator.get("approved_intents") or []) if isinstance(allocator.get("approved_intents"), list) else 0,
        "pre_trade_orders": len(pre_trade_rows),
    }
    result_evidence = _execution_result_evidence(project_root, mode, now)
    payload["execution_result_evidence"] = result_evidence
    payload["fresh_non_stale_result_activity"] = bool(result_evidence.get("fresh_non_stale_activity", False))
    payload["fresh_paper_executed"] = bool(result_evidence.get("fresh_paper_executed", False))
    payload["stale_skip_only_result_activity"] = bool(result_evidence.get("stale_skip_only", False))
    payload["result_activity_status"] = str(result_evidence.get("activity_status") or "")
    if auth_ok is not None:
        payload["auth_ok"] = bool(auth_ok)
        payload["auth_error"] = str(auth_error or "")
    _write_latest(project_root, f"execution_lane_{mode}_latest.json", payload)
