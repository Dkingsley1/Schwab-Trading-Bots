import hashlib
import json
import os
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_schema_version() -> int:
    try:
        return max(int(os.getenv("LOG_SCHEMA_VERSION", "2")), 1)
    except Exception:
        return 2


def current_correlation() -> Dict[str, str]:
    return {
        "run_id": str(os.getenv("CORRELATION_RUN_ID", "") or "").strip(),
        "iter_id": str(os.getenv("CORRELATION_ITER_ID", "") or "").strip(),
    }


def _clean_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    out: list[str] = []
    prev_sep = False
    for ch in text:
        if ch.isalnum():
            out.append(ch)
            prev_sep = False
        elif ch in {"-", "_", ".", "/", " ", ":"} and not prev_sep:
            out.append("_")
            prev_sep = True
    return "".join(out).strip("_")


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _nested_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _path_hint_text(out: Dict[str, Any], path_hint: str = "") -> str:
    metadata = _nested_dict(out.get("metadata"))
    candidates = [
        path_hint,
        out.get("source_path"),
        out.get("target_path"),
        out.get("file_path"),
        out.get("path"),
        metadata.get("source_path"),
        metadata.get("target_path"),
    ]
    return " ".join(str(item or "") for item in candidates if str(item or "").strip()).lower()


def _symbol_text(out: Dict[str, Any]) -> str:
    return str(out.get("symbol") or out.get("underlying_symbol") or "").strip().upper()


def _infer_source_broker(out: Dict[str, Any], *, path_hint: str = "") -> str:
    route = _nested_dict(out.get("data_route"))
    direct = _clean_label(
        _first_text(
            out.get("source_broker"),
            route.get("source_broker"),
            out.get("broker"),
            route.get("broker"),
        )
    )
    if direct:
        return direct
    haystack = " ".join(
        [
            str(out.get("source") or ""),
            str(out.get("provider") or ""),
            str(out.get("source_provider") or ""),
            str(out.get("endpoint") or ""),
            _path_hint_text(out, path_hint),
        ]
    ).lower()
    if "coinbase" in haystack:
        return "coinbase"
    if "schwab" in haystack or "charles_schwab" in haystack:
        return "schwab"
    if "crypto" not in haystack and ("equities" in haystack or "equity" in haystack):
        return "schwab"
    return ""


def _infer_source_provider(out: Dict[str, Any], *, broker: str, path_hint: str = "") -> str:
    route = _nested_dict(out.get("data_route"))
    direct = _clean_label(
        _first_text(
            out.get("source_provider"),
            route.get("source_provider"),
            out.get("provider"),
            out.get("source"),
            route.get("provider"),
        )
    )
    if direct:
        return direct
    haystack = _path_hint_text(out, path_hint)
    for token in (
        "schwab_crypto",
        "coinbase",
        "schwab",
        "kraken",
        "binance",
        "okx",
        "deribit",
        "coingecko",
        "coinmetrics",
        "hyperliquid",
    ):
        if token in haystack:
            return token
    if broker == "schwab" or ("crypto" not in haystack and ("equities" in haystack or "equity" in haystack)):
        return "schwab"
    return broker or "unknown"


def _infer_source_venue(out: Dict[str, Any], *, broker: str, provider: str, path_hint: str = "") -> str:
    route = _nested_dict(out.get("data_route"))
    direct = _clean_label(_first_text(out.get("source_venue"), route.get("source_venue"), out.get("venue")))
    if direct:
        return direct
    haystack = " ".join([provider, broker, _path_hint_text(out, path_hint)]).lower()
    if "schwab_crypto" in haystack:
        return "schwab_crypto_bridge"
    if "coinbase" in haystack:
        return "coinbase"
    if "schwab" in haystack:
        return "schwab"
    if provider == "schwab" or broker == "schwab":
        return "schwab"
    return provider or broker or "unknown"


def _infer_asset_class(out: Dict[str, Any], *, broker: str, provider: str, path_hint: str = "") -> str:
    route = _nested_dict(out.get("data_route"))
    direct = _clean_label(
        _first_text(
            out.get("asset_class"),
            route.get("asset_class"),
            out.get("market_kind"),
            out.get("instrument_class"),
        )
    )
    if direct:
        return "equities" if direct == "equity" else direct

    symbol = _symbol_text(out)
    profile = _clean_label(out.get("profile") or out.get("shadow_profile"))
    domain = _clean_label(out.get("domain") or out.get("shadow_domain"))
    haystack = " ".join([symbol, profile, domain, broker, provider, _path_hint_text(out, path_hint)]).lower()
    if symbol.startswith("/"):
        return "futures"
    if "schwab_futures" in haystack or "_futures" in haystack:
        return "futures"
    if "crypto" in haystack or broker == "coinbase" or symbol.endswith(("-USD", "-USDT", "-USDC", "-BTC", "-ETH")):
        return "crypto"
    if "option" in haystack or (" C00" in symbol or " P00" in symbol):
        return "options"
    if "fx" in haystack or "forex" in haystack:
        return "fx"
    if "equities" in haystack or "equity" in haystack:
        return "equities"
    if domain:
        return "equities" if domain == "equity" else domain
    return "unknown"


def _infer_routing_lane(
    out: Dict[str, Any],
    *,
    broker: str,
    provider: str,
    venue: str,
    asset_class: str,
    path_hint: str = "",
) -> str:
    route = _nested_dict(out.get("data_route"))
    direct = _clean_label(_first_text(out.get("routing_lane"), route.get("routing_lane")))
    if direct:
        return direct
    haystack = _path_hint_text(out, path_hint)
    if "paper_broker_bridge" in haystack:
        return "paper_broker_bridge"
    if venue == "schwab_crypto_bridge" or provider == "schwab_crypto":
        return "schwab_crypto_bridge"
    if broker and asset_class and asset_class != "unknown":
        return f"{broker}_{asset_class}"
    if provider and asset_class and asset_class != "unknown":
        return f"{provider}_{asset_class}"
    channel = _clean_label(out.get("channel"))
    return channel or "unclassified"


def _source_quality(label_seed: str) -> tuple[str, float]:
    seed = _clean_label(label_seed)
    if "schwab_crypto_bridge" in seed:
        return "broker_bridge", 0.80
    if seed in {"schwab", "schwab_equities", "schwab_futures", "schwab_options"} or seed.startswith("schwab_"):
        return "broker_native", 0.95
    if seed in {"coinbase", "coinbase_crypto"} or seed.startswith("coinbase_"):
        return "exchange_native", 0.92
    if "crypto_market_context" in seed or "cross_provider" in seed:
        return "multi_source_context", 0.82
    if "public" in seed or "news" in seed or "external_context" in seed:
        return "public_context", 0.65
    if "sim" in seed or "synthetic" in seed:
        return "synthetic_or_simulated", 0.50
    return "unclassified", 0.50


def _merge_labels(existing: Any, additions: Sequence[str]) -> list[str]:
    labels: list[str] = []
    if isinstance(existing, (list, tuple, set)):
        labels.extend(str(item) for item in existing if str(item or "").strip())
    elif str(existing or "").strip():
        labels.append(str(existing).strip())
    labels.extend(str(item) for item in additions if str(item or "").strip())
    seen: set[str] = set()
    out: list[str] = []
    for label in labels:
        cleaned = _clean_label(label)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            out.append(cleaned)
    return out


def _enrich_data_route(out: Dict[str, Any], *, path_hint: str = "", channel: str = "") -> None:
    if channel and not str(out.get("channel") or "").strip():
        out["channel"] = channel

    broker = _infer_source_broker(out, path_hint=path_hint)
    provider = _infer_source_provider(out, broker=broker, path_hint=path_hint)
    venue = _infer_source_venue(out, broker=broker, provider=provider, path_hint=path_hint)
    asset_class = _infer_asset_class(out, broker=broker, provider=provider, path_hint=path_hint)
    lane = _infer_routing_lane(
        out,
        broker=broker,
        provider=provider,
        venue=venue,
        asset_class=asset_class,
        path_hint=path_hint,
    )

    explicit_quality_label = _clean_label(out.get("source_quality_label"))
    explicit_quality_score = out.get("source_quality_score")
    inferred_quality_label, inferred_quality_score = _source_quality(" ".join([lane, venue, provider, broker]))
    quality_label = explicit_quality_label or inferred_quality_label
    try:
        quality_score = float(explicit_quality_score)
    except Exception:
        quality_score = inferred_quality_score

    channel_label = _clean_label(out.get("channel"))
    profile = _clean_label(out.get("profile") or out.get("shadow_profile"))
    domain = _clean_label(out.get("domain") or out.get("shadow_domain"))
    additions = [
        f"broker:{broker}" if broker else "",
        f"provider:{provider}" if provider else "",
        f"venue:{venue}" if venue else "",
        f"asset:{asset_class}" if asset_class else "",
        f"lane:{lane}" if lane else "",
        f"channel:{channel_label}" if channel_label else "",
        f"profile:{profile}" if profile else "",
        f"domain:{domain}" if domain else "",
        f"quality:{quality_label}" if quality_label else "",
    ]
    labels = _merge_labels(out.get("data_labels") or out.get("labels"), additions)

    if broker and not str(out.get("source_broker") or "").strip():
        out["source_broker"] = broker
    if provider and not str(out.get("source_provider") or "").strip():
        out["source_provider"] = provider
    if venue and not str(out.get("source_venue") or "").strip():
        out["source_venue"] = venue
    if asset_class and not str(out.get("asset_class") or "").strip():
        out["asset_class"] = asset_class
    if lane and not str(out.get("routing_lane") or "").strip():
        out["routing_lane"] = lane
    if quality_label and not str(out.get("source_quality_label") or "").strip():
        out["source_quality_label"] = quality_label
    if "source_quality_score" not in out:
        out["source_quality_score"] = round(float(quality_score), 3)
    out["data_labels"] = labels

    route = _nested_dict(out.get("data_route"))
    route.setdefault("schema_version", 1)
    route.setdefault("source_broker", broker)
    route.setdefault("source_provider", provider)
    route.setdefault("source_venue", venue)
    route.setdefault("asset_class", asset_class)
    route.setdefault("routing_lane", lane)
    route.setdefault("channel", channel_label)
    route.setdefault("profile", profile)
    route.setdefault("domain", domain)
    route.setdefault("source_quality_label", quality_label)
    route.setdefault("source_quality_score", round(float(quality_score), 3))
    route.setdefault("route_key", ":".join(part for part in (lane, channel_label, provider) if part))
    route.setdefault("labels", labels)
    linked_provider = _clean_label(_first_text(out.get("linked_provider"), route.get("linked_provider")))
    if linked_provider:
        route.setdefault("linked_provider", linked_provider)
    out["data_route"] = route


def _ensure_message_contract(out: Dict[str, Any]) -> None:
    msg_id = str(out.get("message_id") or "").strip()
    if not msg_id:
        msg_id = str(uuid.uuid4())
        out["message_id"] = msg_id

    parent = str(out.get("parent_message_id") or "").strip()
    if not parent:
        parent = str(out.get("parent_decision_id") or "").strip()
    if (not parent) and isinstance(out.get("metadata"), dict):
        parent = str(out["metadata"].get("parent_message_id") or out["metadata"].get("parent_decision_id") or "").strip()
    if parent and ("parent_message_id" not in out):
        out["parent_message_id"] = parent


def enrich_log_row(
    row: Dict[str, Any],
    *,
    include_correlation: bool = True,
    include_schema: bool = True,
    path_hint: str = "",
    channel: str = "",
) -> Dict[str, Any]:
    out = dict(row or {})
    if include_schema and ("log_schema_version" not in out):
        out["log_schema_version"] = log_schema_version()

    if include_correlation:
        corr = current_correlation()
        if corr.get("run_id") and ("run_id" not in out):
            out["run_id"] = corr["run_id"]
        if corr.get("iter_id") and ("iter_id" not in out):
            out["iter_id"] = corr["iter_id"]

    _enrich_data_route(out, path_hint=path_hint, channel=channel)
    _ensure_message_contract(out)
    return out


CHANNEL_SCHEMA_REQUIRED: Dict[str, tuple[str, ...]] = {
    "runtime": ("timestamp_utc", "event", "message_id"),
    "gate": ("timestamp_utc", "symbol", "gate", "message_id"),
    "ingress": ("timestamp_utc", "symbol", "status", "message_id"),
    "api": ("timestamp_utc", "symbol", "endpoint", "status", "message_id"),
    "loop_state": ("timestamp_utc", "state", "iter", "message_id"),
    "decision": ("timestamp_utc", "symbol", "action", "message_id"),
    "risk": ("timestamp_utc", "message_id"),
    "execution_guard": ("timestamp_utc", "event", "status", "message_id"),
    "softguard": ("timestamp_utc", "event", "status", "message_id"),
    "auth": ("timestamp_utc", "event", "status", "message_id"),
}

HOT_QUEUE_CHANNELS = {
    "runtime",
    "gate",
    "ingress",
    "api",
    "loop_state",
    "decision",
    "risk",
}


_LOW_SIGNAL_RECENT: Dict[str, float] = {}
_LOW_SIGNAL_RECENT_LOCK = threading.Lock()
_SCHEMA_VIOLATION_RECENT: Dict[str, float] = {}
_SCHEMA_VIOLATION_RECENT_LOCK = threading.Lock()
SIGNAL_GENERATION_STATUSES = {
    "PAPER_EXECUTED",
    "LIVE_EXECUTED",
    "SHADOW_ONLY",
    "DATA_ONLY_BLOCKED",
    "PAPER_GUARD_BLOCKED",
    "LIVE_GUARD_BLOCKED",
    "BLOCKED",
    "HOLD",
}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) != 0.0
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _low_signal_thinning_enabled() -> bool:
    return os.getenv("LOW_SIGNAL_LOG_THINNING_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}


def _low_signal_decision_window_seconds() -> float:
    return max(float(os.getenv("LOW_SIGNAL_DECISION_WINDOW_SECONDS", "60") or 60.0), 1.0)


def _low_signal_execution_guard_window_seconds() -> float:
    return max(float(os.getenv("LOW_SIGNAL_EXECUTION_GUARD_WINDOW_SECONDS", "60") or 60.0), 1.0)


def _schema_violation_window_seconds() -> float:
    return max(float(os.getenv("CHANNEL_SCHEMA_VIOLATION_WINDOW_SECONDS", "300") or 300.0), 1.0)


def _signal_generation_bad_signal_thinning_enabled() -> bool:
    return os.getenv("SIGNAL_GENERATION_BAD_SIGNAL_THINNING_ENABLED", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _signal_generation_bad_signal_window_seconds() -> float:
    return max(float(os.getenv("SIGNAL_GENERATION_BAD_SIGNAL_WINDOW_SECONDS", "300") or 300.0), 1.0)


def _signal_generation_bad_signal_batch_cap() -> int:
    try:
        return max(int(os.getenv("SIGNAL_GENERATION_BAD_SIGNAL_BATCH_CAP", "128") or 128), 0)
    except Exception:
        return 128


def _should_emit_signal_generation_event(
    payload: Dict[str, Any],
    *,
    classification: str,
    reason: str,
    emitted_bad_count: int = 0,
) -> bool:
    if classification != "bad_signal" or not _signal_generation_bad_signal_thinning_enabled():
        return True
    batch_cap = _signal_generation_bad_signal_batch_cap()
    if batch_cap > 0 and int(emitted_bad_count) >= batch_cap:
        return False
    now_ts = time.time()
    window = _signal_generation_bad_signal_window_seconds()
    symbol = str(payload.get("symbol") or "UNKNOWN").strip()
    action = str(payload.get("action") or "UNKNOWN").strip().upper()
    strategy = str(payload.get("strategy") or payload.get("bot_id") or "UNKNOWN").strip()
    status = str(payload.get("status") or "").strip().upper()
    signature = f"signal_generation:{classification}:{reason}:{symbol}:{action}:{strategy}:{status}"
    with _LOW_SIGNAL_RECENT_LOCK:
        last_seen = _LOW_SIGNAL_RECENT.get(signature)
        if last_seen is not None and (now_ts - float(last_seen)) < window:
            return False
        _LOW_SIGNAL_RECENT[signature] = now_ts
    return True


def _schema_violation_signature(
    *,
    source: str,
    channel: str,
    target_path: str,
    payload: Dict[str, Any],
    errors: Sequence[str],
) -> str:
    seed = {
        "source": str(source or ""),
        "channel": str(channel or ""),
        "target_path": str(target_path or ""),
        "errors": [str(item or "").strip() for item in errors if str(item or "").strip()],
        "symbol": str(payload.get("symbol") or ""),
        "event": str(payload.get("event") or ""),
        "status": str(payload.get("status") or ""),
        "gate": str(payload.get("gate") or ""),
        "action": str(payload.get("action") or ""),
    }
    return hashlib.sha1(json.dumps(seed, ensure_ascii=True, sort_keys=True).encode("utf-8")).hexdigest()


def _schema_violation_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for key in (
        "timestamp_utc",
        "symbol",
        "event",
        "status",
        "gate",
        "action",
        "strategy",
        "broker",
        "profile",
        "domain",
        "run_id",
        "iter_id",
        "message_id",
        "parent_message_id",
        "decision_id",
        "parent_decision_id",
        "log_schema_version",
    ):
        value = payload.get(key)
        if value in (None, "", [], {}):
            continue
        summary[key] = value
    summary["payload_key_count"] = len(payload)
    summary["payload_keys"] = sorted(str(key) for key in payload.keys())[:20]
    return summary


def _should_log_schema_violation(signature: str) -> bool:
    now_ts = time.time()
    window = _schema_violation_window_seconds()
    with _SCHEMA_VIOLATION_RECENT_LOCK:
        last_ts = _SCHEMA_VIOLATION_RECENT.get(signature)
        if last_ts is not None and (now_ts - float(last_ts)) < window:
            return False
        _SCHEMA_VIOLATION_RECENT[signature] = now_ts
        stale_cutoff = now_ts - max(window * 2.0, 60.0)
        stale_keys = [key for key, ts in _SCHEMA_VIOLATION_RECENT.items() if float(ts) < stale_cutoff]
        for key in stale_keys[:512]:
            _SCHEMA_VIOLATION_RECENT.pop(key, None)
    return True


def _low_signal_signature(path: str, payload: Dict[str, Any]) -> tuple[str, float] | None:
    norm_path = str(path or "").replace("\\", "/")
    status = str(payload.get("status") or "").strip()

    if "/decision_explanations/" in norm_path and status in {"DATA_ONLY_BLOCKED", "SHADOW_ONLY", "PAPER_GUARD_BLOCKED"}:
        safety = payload.get("safety") if isinstance(payload.get("safety"), dict) else {}
        observe_only = _as_bool(safety.get("market_data_only")) and (not _as_bool(safety.get("execution_enabled")))
        if status == "DATA_ONLY_BLOCKED" and (not observe_only):
            return None
        symbol = str(payload.get("symbol") or "UNKNOWN").strip()
        action = str(payload.get("action") or "UNKNOWN").strip()
        strategy = str(payload.get("strategy") or "UNKNOWN").strip()
        reasons = payload.get("reasons") if isinstance(payload.get("reasons"), list) else []
        reason = str(reasons[0] or "").strip() if reasons else ""
        signature = f"decision:{status}:{symbol}:{action}:{strategy}:{reason}"
        return signature, _low_signal_decision_window_seconds()

    if os.path.basename(norm_path).startswith("paper_execution_guard_"):
        event = str(payload.get("event") or "").strip()
        if event != "pre_trade_check":
            return None
        guard_status = str(payload.get("status") or "").strip().lower()
        if guard_status not in {"blocked", "skip", "skipped"}:
            return None
        details = payload.get("details") if isinstance(payload.get("details"), dict) else {}
        symbol = str(details.get("symbol") or payload.get("symbol") or "UNKNOWN").strip()
        action = str(details.get("action") or payload.get("action") or "UNKNOWN").strip()
        reason = str(payload.get("reason") or details.get("reason") or "").strip()
        gate = str(details.get("gate") or "").strip()
        mode = str(payload.get("mode") or "").strip()
        signature = f"execution_guard:{guard_status}:{mode}:{symbol}:{action}:{reason}:{gate}"
        return signature, _low_signal_execution_guard_window_seconds()

    return None


def _thin_low_signal_payloads(path: str, payloads: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = [dict(p or {}) for p in payloads]
    if (not rows) or (not _low_signal_thinning_enabled()):
        return rows

    now = time.time()
    retention_window = max(_low_signal_decision_window_seconds(), _low_signal_execution_guard_window_seconds(), 300.0) * 2.0
    kept: List[Dict[str, Any]] = []
    norm_path = os.path.abspath(path)

    with _LOW_SIGNAL_RECENT_LOCK:
        stale_keys = [key for key, ts in _LOW_SIGNAL_RECENT.items() if (now - ts) >= retention_window]
        for key in stale_keys:
            _LOW_SIGNAL_RECENT.pop(key, None)

        for payload in rows:
            sig = _low_signal_signature(norm_path, payload)
            if sig is None:
                kept.append(payload)
                continue
            signature, window_seconds = sig
            cache_key = f"{norm_path}:{signature}"
            last_seen = _LOW_SIGNAL_RECENT.get(cache_key)
            if last_seen is not None and (now - last_seen) < window_seconds:
                continue
            _LOW_SIGNAL_RECENT[cache_key] = now
            kept.append(payload)

    return kept


def _signal_generation_classification(payload: Dict[str, Any]) -> tuple[str, str]:
    status = str(payload.get("status") or "").strip().upper()
    action = str(payload.get("action") or "").strip().upper()
    score = payload.get("score")
    threshold = payload.get("threshold")
    generated_trade_intent = action in {"BUY", "SELL"} or str(payload.get("intent_action") or "").strip().upper() in {"BUY", "SELL"}
    blocked = "BLOCKED" in status or "GUARD" in status
    executed = status in {"PAPER_EXECUTED", "LIVE_EXECUTED"}
    shadow_intent = status == "SHADOW_ONLY" and generated_trade_intent
    if executed or shadow_intent:
        return "good_signal", "trade_intent_generated"
    if blocked and generated_trade_intent:
        return "bad_signal", "trade_intent_blocked"
    if status == "DATA_ONLY_BLOCKED":
        return "bad_signal", "data_only_blocked"
    if action == "HOLD" or status == "HOLD":
        return "bad_signal", "hold_or_no_trade_signal"
    if score is not None and threshold is not None:
        try:
            if float(score) >= float(threshold):
                return "good_signal", "score_above_threshold"
            return "bad_signal", "score_below_threshold"
        except Exception:
            pass
    return "bad_signal", "no_executable_signal"


def _signal_generation_events(
    *,
    project_root: str,
    target_path: str,
    payloads: Sequence[Dict[str, Any]],
) -> None:
    if not project_root or not payloads:
        return
    norm_path = str(target_path or "").replace("\\", "/")
    if "decision_explanations/" not in norm_path and "/decisions/" not in norm_path and "trade_decisions_" not in norm_path:
        return

    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_path = os.path.join(project_root, "governance", "events", f"signal_generation_{day}.jsonl")
    lines: list[str] = []
    emitted_bad_count = 0
    for payload in payloads:
        status = str(payload.get("status") or "").strip().upper()
        action = str(payload.get("action") or "").strip().upper()
        if status and status not in SIGNAL_GENERATION_STATUSES and action not in {"BUY", "SELL", "HOLD"}:
            continue
        classification, reason = _signal_generation_classification(payload)
        if not _should_emit_signal_generation_event(
            payload,
            classification=classification,
            reason=reason,
            emitted_bad_count=emitted_bad_count,
        ):
            continue
        if classification == "bad_signal":
            emitted_bad_count += 1
        row: Dict[str, Any] = {
            "timestamp_utc": now_utc_iso(),
            "event": "signal_generation",
            "signal_quality": classification,
            "reason": reason,
            "source_path": str(target_path or ""),
            "symbol": str(payload.get("symbol") or ""),
            "action": action,
            "status": status,
            "strategy": str(payload.get("strategy") or payload.get("bot_id") or ""),
            "score": payload.get("score"),
            "threshold": payload.get("threshold"),
            "message_id": str(payload.get("message_id") or ""),
            "parent_message_id": str(payload.get("parent_message_id") or payload.get("parent_decision_id") or ""),
            "log_schema_version": log_schema_version(),
        }
        corr = current_correlation()
        if corr.get("run_id"):
            row["run_id"] = corr["run_id"]
        if corr.get("iter_id"):
            row["iter_id"] = corr["iter_id"]
        lines.append(json.dumps(enrich_log_row(row), ensure_ascii=True) + "\n")
    _write_lines(out_path, lines)


def _schema_errors(payload: Dict[str, Any], *, schema: str) -> List[str]:
    req = CHANNEL_SCHEMA_REQUIRED.get(str(schema or "").strip(), ())
    if not req:
        return []
    errors: List[str] = []
    for key in req:
        val = payload.get(key)
        if val is None:
            errors.append(f"missing:{key}")
            continue
        if isinstance(val, str) and (not val.strip()):
            errors.append(f"missing:{key}")
    return errors


def _write_lines(path: str, lines: Sequence[str]) -> bool:
    if not lines:
        return True
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write("".join(lines))
        return True
    except Exception:
        return False


def safe_append_jsonl_batch(
    path: str,
    rows: Iterable[Dict[str, Any]],
    *,
    project_root: str = "",
    source: str = "",
) -> int:
    payloads = [enrich_log_row(dict(r or {}), path_hint=path) for r in rows]
    if not payloads:
        return 0

    lines = [json.dumps(p, ensure_ascii=True) + "\n" for p in payloads]
    if _write_lines(path, lines):
        _signal_generation_events(project_root=project_root, target_path=path, payloads=payloads)
        return len(payloads)

    _emit_write_failure_event(
        project_root=project_root,
        source=source or "jsonl_writer_batch",
        target_path=path,
        error=RuntimeError("batch_append_failed"),
    )
    return 0


def _default_queue_db(project_root: str) -> str:
    if not project_root:
        return ""
    from core.channel_queue import default_queue_db_path

    return default_queue_db_path(project_root)


def _queue_publish(
    *,
    project_root: str,
    channel: str,
    source_path: str,
    payloads: Sequence[Dict[str, Any]],
    queue_db_path: str = "",
) -> None:
    if not project_root:
        return
    if str(channel or "") not in HOT_QUEUE_CHANNELS:
        return

    try:
        from core.channel_queue import ChannelQueue, default_queue_db_path, queue_enabled

        if not queue_enabled():
            return

        db_path = queue_db_path or default_queue_db_path(project_root)
        q = ChannelQueue(db_path)

        require_consumer = os.getenv("BOT_CHANNEL_QUEUE_REQUIRE_RECENT_CONSUMER", "1").strip().lower() in {"1", "true", "yes", "on"}
        consumer_max_age_seconds = max(int(os.getenv("BOT_CHANNEL_QUEUE_CONSUMER_MAX_AGE_SECONDS", "86400")), 60)
        if require_consumer and not q.has_recent_consumer(channel=channel, max_age_seconds=consumer_max_age_seconds):
            return

        q.enqueue_batch(
            channel=channel,
            payloads=list(payloads),
            source_path=source_path,
        )
    except Exception as exc:
        _emit_write_failure_event(
            project_root=project_root,
            source="channel_queue.enqueue",
            target_path=queue_db_path or _default_queue_db(project_root),
            error=exc,
        )


def _schema_violation_log(
    *,
    project_root: str,
    source: str,
    channel: str,
    target_path: str,
    payload: Dict[str, Any],
    errors: Sequence[str],
) -> None:
    if not project_root:
        return
    signature = _schema_violation_signature(
        source=source,
        channel=channel,
        target_path=target_path,
        payload=payload,
        errors=errors,
    )
    if not _should_log_schema_violation(signature):
        return
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_path = os.path.join(project_root, "governance", "events", f"channel_schema_violations_{day}.jsonl")
    row = {
        "timestamp_utc": now_utc_iso(),
        "event": "channel_schema_violation",
        "source": str(source or "unknown"),
        "channel": str(channel or ""),
        "target_path": str(target_path or ""),
        "signature": signature,
        "errors": list(errors),
        "payload": _schema_violation_summary(payload),
        "log_schema_version": log_schema_version(),
    }
    corr = current_correlation()
    if corr.get("run_id"):
        row["run_id"] = corr["run_id"]
    if corr.get("iter_id"):
        row["iter_id"] = corr["iter_id"]
    safe_append_jsonl(out_path, row, project_root=project_root, source="channel_schema_violation")


def _schema_strict_enabled(explicit: Optional[bool]) -> bool:
    if explicit is not None:
        return bool(explicit)
    return os.getenv("CHANNEL_SCHEMA_STRICT", "0").strip().lower() in {"1", "true", "yes", "on"}


def safe_append_channel_batch(
    path: str,
    rows: Iterable[Dict[str, Any]],
    *,
    project_root: str = "",
    source: str = "",
    channel: str = "",
    schema: str = "",
    mirror_paths: Optional[Sequence[str]] = None,
    strict_schema: Optional[bool] = None,
    queue_db_path: str = "",
) -> int:
    raw_payloads = [dict(r or {}) for r in rows]
    if not raw_payloads:
        return 0

    ch = str(channel or "").strip()
    sch = str(schema or ch).strip()
    strict = _schema_strict_enabled(strict_schema)

    valid_payloads: List[Dict[str, Any]] = []
    for raw in raw_payloads:
        if ch and ("channel" not in raw):
            raw["channel"] = ch
        payload = enrich_log_row(raw, path_hint=path, channel=ch)
        errors = _schema_errors(payload, schema=sch)
        if errors:
            _schema_violation_log(
                project_root=project_root,
                source=source,
                channel=ch or sch,
                target_path=path,
                payload=payload,
                errors=errors,
            )
            if strict:
                continue
            payload["schema_errors"] = list(errors)
            payload["schema_valid"] = False
        else:
            payload["schema_valid"] = True
        valid_payloads.append(payload)

    if not valid_payloads:
        return 0

    valid_payloads = _thin_low_signal_payloads(path, valid_payloads)
    if not valid_payloads:
        return 0
    lines = [json.dumps(p, ensure_ascii=True) + "\n" for p in valid_payloads]
    if not _write_lines(path, lines):
        _emit_write_failure_event(
            project_root=project_root,
            source=source or "channel_writer",
            target_path=path,
            error=RuntimeError("channel_batch_append_failed"),
        )
        return 0
    _signal_generation_events(project_root=project_root, target_path=path, payloads=valid_payloads)

    mirrors = [str(p) for p in (mirror_paths or []) if str(p or "").strip()]
    for mirror in mirrors:
        if os.path.abspath(mirror) == os.path.abspath(path):
            continue
        if not _write_lines(mirror, lines):
            _emit_write_failure_event(
                project_root=project_root,
                source=source or "channel_writer_mirror",
                target_path=mirror,
                error=RuntimeError("channel_mirror_append_failed"),
            )

    if ch:
        _queue_publish(
            project_root=project_root,
            channel=ch,
            source_path=path,
            payloads=valid_payloads,
            queue_db_path=queue_db_path,
        )

    return len(valid_payloads)


def safe_append_channel_event(
    path: str,
    row: Dict[str, Any],
    *,
    project_root: str = "",
    source: str = "",
    channel: str = "",
    schema: str = "",
    mirror_paths: Optional[Sequence[str]] = None,
    strict_schema: Optional[bool] = None,
    queue_db_path: str = "",
) -> bool:
    wrote = safe_append_channel_batch(
        path,
        [row],
        project_root=project_root,
        source=source,
        channel=channel,
        schema=schema,
        mirror_paths=mirror_paths,
        strict_schema=strict_schema,
        queue_db_path=queue_db_path,
    )
    return wrote > 0


def safe_append_jsonl(
    path: str,
    row: Dict[str, Any],
    *,
    project_root: str = "",
    source: str = "",
) -> bool:
    wrote = safe_append_jsonl_batch(path, [row], project_root=project_root, source=source)
    return wrote > 0


def safe_write_json(
    path: str,
    payload: Dict[str, Any],
    *,
    project_root: str = "",
    source: str = "",
    indent: int = 2,
) -> bool:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=indent)
        return True
    except Exception as exc:
        _emit_write_failure_event(
            project_root=project_root,
            source=source or "json_writer",
            target_path=path,
            error=exc,
        )
        return False


def safe_write_json_atomic(
    path: str,
    payload: Dict[str, Any],
    *,
    project_root: str = "",
    source: str = "",
    indent: int = 2,
    marker: bool = True,
) -> bool:
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=indent), encoding="utf-8")
        tmp.replace(target)

        if marker:
            marker_path = target.with_suffix(target.suffix + ".ok")
            marker_payload = {
                "timestamp_utc": now_utc_iso(),
                "source": str(source or "json_writer_atomic"),
                "payload_sha256": sha256_json_obj(payload),
                "target": str(target),
            }
            marker_path.write_text(json.dumps(marker_payload, ensure_ascii=True, indent=2), encoding="utf-8")
        return True
    except Exception as exc:
        _emit_write_failure_event(
            project_root=project_root,
            source=source or "json_writer_atomic",
            target_path=path,
            error=exc,
        )
        return False


def _emit_write_failure_event(
    *,
    project_root: str,
    source: str,
    target_path: str,
    error: Exception,
) -> None:
    if not project_root:
        print(f"[WriteFail] source={source} target={target_path} err={error}")
        return

    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    fail_path = os.path.join(project_root, "governance", "events", f"write_failures_{day}.jsonl")
    row = {
        "timestamp_utc": now_utc_iso(),
        "event": "write_failure",
        "source": source,
        "target_path": target_path,
        "error": str(error),
        "error_type": type(error).__name__,
        "log_schema_version": log_schema_version(),
    }
    corr = current_correlation()
    if corr.get("run_id"):
        row["run_id"] = corr["run_id"]
    if corr.get("iter_id"):
        row["iter_id"] = corr["iter_id"]

    try:
        os.makedirs(os.path.dirname(fail_path), exist_ok=True)
        with open(fail_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    except Exception as inner_exc:
        print(
            f"[WriteFail] source={source} target={target_path} err={error} "
            f"failure_event_write_err={inner_exc}"
        )


def git_commit(project_root: str) -> str:
    if not project_root:
        return ""
    try:
        proc = subprocess.run(
            ["git", "-C", project_root, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            return str(proc.stdout or "").strip()
        return ""
    except Exception:
        return ""


def sha256_file(path: str) -> str:
    if not path or (not os.path.exists(path)):
        return ""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def sha256_json_obj(obj: Any) -> str:
    try:
        encoded = json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
    except Exception:
        return ""


def _bot_map(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    rows = payload.get("sub_bots") if isinstance(payload.get("sub_bots"), list) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        bot_id = str(row.get("bot_id") or "").strip()
        if not bot_id:
            continue
        out[bot_id] = row
    return out


def _bot_field_diff(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    changed: Dict[str, Dict[str, Any]] = {}
    keys = sorted(set(before.keys()) | set(after.keys()))
    for key in keys:
        b = before.get(key)
        a = after.get(key)
        if b != a:
            changed[key] = {"before": b, "after": a}
    return changed


def compute_registry_mutation(
    *,
    before: Dict[str, Any],
    after: Dict[str, Any],
) -> Dict[str, Any]:
    before_map = _bot_map(before)
    after_map = _bot_map(after)

    bot_diffs = []
    for bot_id in sorted(set(before_map.keys()) | set(after_map.keys())):
        b = before_map.get(bot_id)
        a = after_map.get(bot_id)
        if b is None and a is not None:
            bot_diffs.append({"bot_id": bot_id, "change_type": "added", "after": a})
            continue
        if a is None and b is not None:
            bot_diffs.append({"bot_id": bot_id, "change_type": "removed", "before": b})
            continue
        if b is None or a is None:
            continue
        fields = _bot_field_diff(b, a)
        if fields:
            bot_diffs.append(
                {
                    "bot_id": bot_id,
                    "change_type": "updated",
                    "changed_fields": fields,
                }
            )

    return {
        "bots_total_before": int(len(before_map)),
        "bots_total_after": int(len(after_map)),
        "bot_diff_count": int(len(bot_diffs)),
        "bot_diffs": bot_diffs,
        "registry_sha256_before": sha256_json_obj(before),
        "registry_sha256_after": sha256_json_obj(after),
    }


def write_registry_mutation_journal(
    *,
    project_root: str,
    actor: str,
    reason: str,
    before: Dict[str, Any],
    after: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    mutation = compute_registry_mutation(before=before, after=after)
    payload: Dict[str, Any] = {
        "timestamp_utc": now_utc_iso(),
        "actor": actor,
        "reason": reason,
        "mutation": mutation,
        "log_schema_version": log_schema_version(),
    }
    if extra:
        payload["extra"] = extra

    corr = current_correlation()
    if corr.get("run_id"):
        payload["run_id"] = corr["run_id"]
    if corr.get("iter_id"):
        payload["iter_id"] = corr["iter_id"]

    audit_dir = os.path.join(project_root, "governance", "audits")
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    journal_path = os.path.join(audit_dir, f"registry_mutation_journal_{day}.jsonl")
    latest_path = os.path.join(audit_dir, "registry_mutation_latest.json")

    safe_append_jsonl(
        journal_path,
        payload,
        project_root=project_root,
        source="registry_mutation_journal",
    )
    safe_write_json(
        latest_path,
        payload,
        project_root=project_root,
        source="registry_mutation_latest",
    )
    return journal_path
