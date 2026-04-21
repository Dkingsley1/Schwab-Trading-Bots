from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from core.accountability import safe_append_jsonl, safe_write_json_atomic
from core.channel_queue import ChannelMessage, ChannelQueue, default_queue_db_path


EXECUTION_INTENT_CHANNEL = "execution_intent"
EXECUTION_RESULT_CHANNEL = "execution_result"
EXECUTION_PROMOTION_CHANNEL = "execution_promotion"
EXECUTION_PROMOTED_CHANNEL = "execution_promoted"


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


def execution_lane_daily_path(project_root: str | Path, stem: str, *, day: str = "") -> str:
    stamp = str(day or datetime.now(timezone.utc).strftime("%Y%m%d"))
    base = Path(project_root) / "governance" / "execution_lanes"
    return str(base / f"{stem}_{stamp}.jsonl")


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
    row = dict(payload or {})
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
    gateway = evaluate_execution_gateway(
        project_root=project_root,
        intent=intent,
        mode=mode,
    )
    if str(mode).strip().lower() == "live" and not bool(gateway.get("allow_execute", False)):
        result = {
            "status": "LIVE_GATEWAY_BLOCKED",
            "reason": "execution_gateway_blocked",
            "execution_gateway": gateway,
        }
    else:
        result = trader.execute_decision(**kwargs)

    result_payload = {
        "timestamp_utc": _now_utc(),
        "mode": str(mode),
        "consumer": str(mode),
        "intent_channel": str(message.channel),
        "intent_message_id": str(message.message_id),
        "intent_created_at": str(message.created_at),
        "intent": intent,
        "result_status": str(result.get("status") or ""),
        "result": result,
        "execution_gateway": gateway,
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
    q = ChannelQueue(queue_db_path(project_root, queue_db_override))
    consumer = f"execution_lane_{mode}"
    queue_stats = q.queue_stats(channel=queue_channel)
    consumer_state = q.consumer_state(consumer=consumer, channel=queue_channel)
    pending_rows = q.pending_count(consumer=consumer, channel=queue_channel)
    now = datetime.now(timezone.utc)
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
    if int(pending_rows) > 0:
        # Large active queues naturally create short idle gaps between acks; don't
        # label the lane stale while fresh intents are still flowing in.
        if queue_newest_age_seconds is not None and float(queue_newest_age_seconds) <= max(float(stale_after_seconds), 300.0):
            stale_grace_seconds += int(stale_after_seconds)
        backlog_scale = min(max(int(pending_rows) // 25000, 0), 10)
        stale_grace_seconds += int(backlog_scale * 60)
    effective_stale_after_seconds = int(stale_after_seconds + stale_grace_seconds)
    stale = bool(
        int(pending_rows) > 0
        and consumer_idle_seconds is not None
        and float(consumer_idle_seconds) >= float(effective_stale_after_seconds)
    )
    payload = {
        "timestamp_utc": _now_utc(),
        "mode": str(mode),
        "consumer": consumer,
        "processed_count": int(processed_count),
        "queue_channel": str(queue_channel),
        "queue_stats": queue_stats,
        "consumer_state": consumer_state,
        "pending_rows": int(pending_rows),
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
    if auth_ok is not None:
        payload["auth_ok"] = bool(auth_ok)
        payload["auth_error"] = str(auth_error or "")
    _write_latest(project_root, f"execution_lane_{mode}_latest.json", payload)
