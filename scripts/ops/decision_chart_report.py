"""Offline, bounded decision charts from original provider-bound captures.

Call publish_decision_chart(root, decision, executions=(), now=...) from the
explanation/report worker, never the trading loop. Persist only its small receipt
in the decision explanation. No broker client, fetch, ledger update or order path.
Execution receipts must come from the successful broker reconciliation owner;
the raw order response and original observation time must be retained there.
"""

from __future__ import annotations

from contextlib import closing
import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import fcntl
import gzip
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import stat
import sqlite3
import tempfile
import time

from core.decision_price_evidence import digest, metrics, timestamp, validate_bars
from core.schd_capture_store import DIRECTORY as CAPTURE_DIRECTORY
from core.schd_capture_store import checked, decision_capture, read_capture
from scripts.ops.schd_candle_report import render_charts

DIRECTORY = Path("governance/reports/decision_charts")
MAX_FILES = 128
MAX_TOTAL_BYTES = 32 * 1024 * 1024
MAX_REPORT_BYTES = 2 * 1024 * 1024
MAX_EXECUTIONS = 8
MAX_INPUT_BYTES = 256 * 1024
MAX_LOG_BYTES = 8 * 1024 * 1024
HASH = re.compile(r"[0-9a-f]{64}")
AUTHORITY = {"live_execution_authority": False, "order_authority": False}


def read_ledger_executions(root, decision_id):
    """Read bounded existing reconciliation events without opening a writer."""
    from core.live_order_ledger import LiveOrderLedger

    path = checked(root, Path(root) / "governance/runtime/live_order_ledger.sqlite3")
    if not path.exists():
        return []
    for suffix in ("-wal", "-shm"):
        checked(root, Path(str(path) + suffix))
    deadline = time.monotonic() + 2
    receipts = []
    with closing(
        sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=1)
    ) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
        events = connection.execute(
            "SELECT * FROM (SELECT * FROM order_events ORDER BY event_id DESC LIMIT 256) "
            "WHERE length(details_json) <= 264192"
        )
        for row in events:
            if time.monotonic() >= deadline:
                raise ValueError("execution_receipt_read_deadline")
            details = json.loads(row["details_json"])
            receipt = details.get("decision_chart_execution")
            if (
                not isinstance(receipt, dict)
                or receipt.get("decision_id") != decision_id
            ):
                continue
            event = {
                key: row[key]
                for key in (
                    "intent_id",
                    "timestamp_utc",
                    "from_state",
                    "to_state",
                    "previous_event_hash",
                )
            }
            event["details"] = details
            if LiveOrderLedger._event_hash(event) != row["event_hash"]:
                raise ValueError("execution_receipt_event_hash_mismatch")
            receipts.append(receipt)
            if len(receipts) >= MAX_EXECUTIONS:
                break
    return receipts


def _positive(value):
    if isinstance(value, bool):
        raise ValueError("invalid_execution_number")
    try:
        value = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("invalid_execution_number") from exc
    if not value.is_finite() or value <= 0 or not math.isfinite(float(value)):
        raise ValueError("invalid_execution_number")
    return value


def _bounded(value):
    raw = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError("chart_input_budget_exceeded")


def execution_markers(decision, receipts, *, now):
    """Validate an explicit reconciliation receipt; status/limit alone is no fill.

    Receipt contract: owner, provider, mode, reconciled, decision_id,
    broker_order_id, observed_at_utc, broker_order_sha256, broker_order.
    These fields are owner provenance, not independent broker authentication.
    """
    if not isinstance(receipts, (list, tuple)) or len(receipts) > MAX_EXECUTIONS:
        raise ValueError("execution_receipt_budget_exceeded")
    markers, issues, seen = [], [], {}
    for receipt in receipts:
        try:
            if (
                decision.get("metadata", {}).get("candle_context", {}).get("provider")
                == "coinbase"
            ):
                raise ValueError("coinbase_execution_receipt_owner_not_implemented")
            order = receipt["broker_order"]
            observed = timestamp(receipt["observed_at_utc"])
            if (
                receipt.get("owner") != "supervised_broker_test.reconcile_order"
                or receipt.get("provider") != "schwab"
                or receipt.get("mode") != "live"
                or receipt.get("reconciled") is not True
                or receipt.get("decision_id") != decision["decision_id"]
                or not receipt.get("broker_order_id")
                or str(receipt["broker_order_id"]) != str(order.get("orderId"))
                or receipt.get("broker_order_sha256") != digest(order)
                or not timestamp(decision["timestamp_utc"]) <= observed <= now
            ):
                raise ValueError("execution_identity_or_provenance_invalid")
            legs = order["orderLegCollection"]
            if (
                len(legs) != 1
                or legs[0]["instrument"].get("symbol") != decision["symbol"]
                or legs[0]["instrument"].get("assetType") != "EQUITY"
                or legs[0].get("instruction") != decision["action"]
                or decision["action"] not in {"BUY", "SELL"}
                or legs[0].get("legId") is None
            ):
                raise ValueError("execution_order_leg_mismatch")
            filled = _positive(order["filledQuantity"])
            requested = _positive(legs[0]["quantity"])
            if filled > requested or (
                order.get("status") == "FILLED" and filled != requested
            ):
                raise ValueError("execution_quantity_mismatch")
            if order.get("status") not in {
                "FILLED",
                "PARTIALLY_FILLED",
                "CANCELED",
                "CANCELLED",
                "EXPIRED",
                "WORKING",
                "PENDING_CANCEL",
            }:
                raise ValueError("execution_status_unusable")
            pending, identities, total = [], set(), Decimal(0)
            for activity in order.get("orderActivityCollection", []):
                if activity.get("activityType") != "EXECUTION":
                    continue
                if activity.get("executionType") != "FILL" or not activity.get(
                    "activityId"
                ):
                    raise ValueError("execution_activity_proof_missing")
                for leg in activity.get("executionLegs", []):
                    when = timestamp(leg["time"])
                    price, quantity = _positive(leg["price"]), _positive(
                        leg["quantity"]
                    )
                    if (
                        leg.get("legId") != legs[0]["legId"]
                        or not timestamp(decision["timestamp_utc"]) <= when <= observed
                    ):
                        raise ValueError("execution_time_or_leg_mismatch")
                    identity = (
                        str(order["orderId"]),
                        str(activity["activityId"]),
                        str(leg["legId"]),
                        leg["time"],
                    )
                    if identity in identities:
                        raise ValueError("duplicate_execution_leg")
                    identities.add(identity)
                    total += quantity
                    pending.append(
                        (
                            identity,
                            {
                                "kind": "executed",
                                "action": decision["action"],
                                "timestamp_utc": leg["time"],
                                "price": float(price),
                                "quantity": float(quantity),
                                "broker_order_id": str(order["orderId"]),
                                "activity_id": str(activity["activityId"]),
                            },
                        )
                    )
            if total != filled or not pending:
                raise ValueError("actual_execution_legs_incomplete")
            for identity, marker in pending:
                if identity in seen and seen[identity] != marker:
                    return [], issues + ["conflicting_broker_execution_evidence"]
                if identity not in seen:
                    seen[identity] = marker
                    markers.append(marker)
        except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
            issues.append("broker_execution_evidence_invalid_or_unbound")
    if len(markers) > MAX_EXECUTIONS:
        return [], issues + ["execution_marker_budget_exceeded"]
    return sorted(markers, key=lambda m: timestamp(m["timestamp_utc"])), issues


def _frames(capture, *, asof, available_at):
    if capture["source"].get("provider") == "coinbase":
        return _coinbase_frames(capture, asof=asof, available_at=available_at)
    frames = {}
    requests = capture["source"].get("requests", [])
    for name, minutes, endpoint in (
        ("1m", 1, "get_price_history_every_minute"),
        ("5m", 5, "get_price_history_every_five_minutes"),
        ("1d", None, "get_price_history_every_day"),
    ):
        try:
            receipts = [r for r in requests if r.get("timeframe") == name]
            if (
                len(receipts) != 1
                or receipts[0].get("endpoint") != endpoint
                or not HASH.fullmatch(str(receipts[0].get("payload_sha256", "")))
            ):
                raise ValueError("native_history_request_receipt_missing")
            rows = validate_bars(
                capture.get("candles", {}).get(name, []),
                minutes=minutes,
                asof=min(asof, available_at),
            )
            if len(rows) != receipts[0].get("closed_bars"):
                raise ValueError("capture_bar_count_mismatch")
            frames[name] = metrics(rows, asof=asof, minutes=minutes)
        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            frames[name] = {"status": "unavailable", "issues": [str(exc)]}
    return frames


def _coinbase_frames(capture, *, asof, available_at):
    from core.decision_price_evidence import number

    frames = {}
    product = capture["source"]["symbol"]
    requests = capture["source"].get("requests", [])
    for name, step in (
        ("1m", 60),
        ("5m", 300),
        ("15m", 900),
        ("1h", 3600),
        ("6h", 21600),
        ("1d", 86400),
    ):
        if name not in capture.get("candles", {}):
            continue
        try:
            receipts = [r for r in requests if r.get("timeframe") == name]
            rows = capture["candles"][name]
            if (
                len(receipts) != 1
                or receipts[0].get("endpoint") != f"/products/{product}/candles"
                or not HASH.fullmatch(str(receipts[0].get("payload_sha256", "")))
                or not isinstance(rows, list)
                or not 0 < len(rows) <= 300
                or len(rows) != receipts[0].get("closed_bars")
            ):
                raise ValueError("coinbase_capture_receipt_invalid")
            previous = None
            for row in rows:
                start, end = timestamp(row["start_utc"]), timestamp(row["end_utc"])
                values = [
                    number(row[k], positive=True)
                    for k in ("open", "high", "low", "close")
                ]
                opening, high, low, close = values
                if (
                    start.timestamp() % step
                    or (end - start).total_seconds() != step
                    or end > min(asof, available_at)
                    or (previous and start < previous)
                    or number(row["volume"]) < 0
                    or not low <= min(opening, close) <= max(opening, close) <= high
                ):
                    raise ValueError("coinbase_candle_invalid_or_unclosed")
                previous = end
            frames[name] = metrics(rows, asof=asof, minutes=step // 60, continuous=True)
        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            frames[name] = {"status": "unavailable", "issues": [str(exc)]}
    return frames


def build_decision_chart(root, decision, *, executions=(), now=None):
    """Read the immutable capture named by the original decision, never latest."""
    now = now or datetime.now(timezone.utc)
    report = {
        "schema_version": 1,
        "symbol": decision.get("symbol"),
        "decision_id": decision.get("decision_id"),
        "as_of_utc": decision.get("timestamp_utc"),
        "original_decision_timestamp_utc": decision.get("timestamp_utc"),
        "recorded_action": decision.get("action"),
        "recorded_gate_decision": decision.get("decision"),
        "evidence_kind": "original_provider_capture",
        "decision_input_eligible": False,
        "timeframes": {},
        "decision_chart_markers": [],
        "issues": [],
        "execution_status": "unavailable",
        **AUTHORITY,
        "interpretation": (
            "Original closed-candle context only. Proposed actions are not fills. "
            "Fills are plotted only within displayed candle intervals; other events are listed. "
            "Price adjustment is unverified. Stored owner provenance is not "
            "independent broker authentication. Charts do not authorize orders."
        ),
    }
    try:
        _bounded(decision)
        _bounded(executions)
        report["recorded_reasoning"] = {
            "source": "original_decision_record_not_reconstructed",
            "strategy": decision.get("strategy"),
            "model_score": decision.get("model_score"),
            "threshold": decision.get("threshold"),
            "reasons": decision.get("reasons", []),
            "gates": decision.get("gates", {}),
            "indicator_features": decision.get("features", {}),
            "feature_compaction_contract": decision.get(
                "feature_compaction_contract", {}
            ),
            "explicit_indicator_reasoning": decision.get("metadata", {}).get(
                "indicator_reasoning", {}
            ),
            "causal_attribution": "Only explicit recorded reasons establish claimed indicator use; feature presence alone does not.",
        }
        when = timestamp(decision["timestamp_utc"])
        if when > now or not decision.get("decision_id"):
            raise ValueError("original_decision_identity_or_time_invalid")
        if decision.get("action") in {"BUY", "SELL"}:
            report["decision_chart_markers"].append(
                {
                    "kind": "proposed",
                    "action": decision["action"],
                    "timestamp_utc": decision["timestamp_utc"],
                }
            )
        fills, issues = execution_markers(decision, executions, now=now)
        report["decision_chart_markers"].extend(fills)
        report["issues"].extend(issues)
        report["execution_status"] = (
            "confirmed_execution_legs"
            if fills
            else (
                "not_applicable_no_order"
                if decision.get("action") == "HOLD"
                else "unavailable"
            )
        )
        if not fills and decision.get("action") != "HOLD":
            report["issues"].append("confirmed_broker_executions_unavailable")
        if "candle_context" in decision.get("metadata", {}):
            from core.decision_candle_store import read_bound

            capture = read_bound(root, decision)
            source = capture["source"]
            captured = timestamp(source["fetch_started_at_utc"])
            report["capture_sha256"] = digest(capture)
            report["chart_source"] = source
            report["timeframes"] = _frames(capture, asof=when, available_at=captured)
            report["status"] = (
                "available"
                if any(f.get("chart_candles") for f in report["timeframes"].values())
                else "unavailable"
            )
            return report
        if decision.get("symbol") != "SCHD":
            raise ValueError("original_capture_binding_missing")
        context = decision.get("metadata", {}).get("schd_candle_context", {})
        identity = context.get("capture_sha256", "")
        if not isinstance(identity, str) or not HASH.fullmatch(identity):
            raise ValueError("original_capture_binding_missing")
        if context.get("state") != "observed_context_not_claimed_model_input":
            raise ValueError("original_capture_binding_invalid")
        capture = decision_capture(root, decision)
        if not capture or digest(capture) != identity:
            raise ValueError("original_capture_unavailable")
        source = capture["source"]
        captured = timestamp(source["fetch_started_at_utc"])
        if source.get("provider") != "schwab" or source.get("symbol") != "SCHD":
            raise ValueError("original_capture_wrong_provider_or_symbol")
        if (
            not captured <= timestamp(context["observed_at_utc"]) <= when
            or not 0 <= (when - captured).total_seconds() <= 300
            or timestamp(context["fetch_started_at_utc"]) != captured
        ):
            raise ValueError("original_capture_stale_future_or_reconstructed")
        report["capture_sha256"] = identity
        report["chart_source"] = source
        report["timeframes"] = _frames(capture, asof=when, available_at=captured)
    except (OSError, KeyError, ValueError, TypeError, AttributeError) as exc:
        report["issues"].append(str(exc))
    report["status"] = (
        "available"
        if any(f.get("chart_candles") for f in report["timeframes"].values())
        else "unavailable"
    )
    return report


def build_execution_review(root, decision, *, capture_sha256, executions=(), now=None):
    """Explicit later capture, isolated from the frozen decision-time evidence."""
    now = now or datetime.now(timezone.utc)
    report = build_decision_chart(root, decision, executions=executions, now=now)
    report.update(
        review_kind="retrospective_execution_review",
        timeframes={},
        status="unavailable",
        decision_input_eligible=False,
    )
    report["interpretation"] = (
        "RETROSPECTIVE EXECUTION REVIEW - NOT DECISION INPUT. "
        "Later captured candles locate confirmed broker fills at actual times and prices. "
        "The original decision timestamp is unchanged. " + report["interpretation"]
    )
    try:
        if decision.get("symbol") != "SCHD":
            raise ValueError("symbol_capture_owner_unavailable")
        if not isinstance(capture_sha256, str) or not HASH.fullmatch(capture_sha256):
            raise ValueError("review_capture_identity_invalid")
        capture = read_capture(
            root, Path(root) / CAPTURE_DIRECTORY / "captures" / f"{capture_sha256}.json"
        )
        source = capture["source"]
        captured = timestamp(source["fetch_started_at_utc"])
        if (
            digest(capture) != capture_sha256
            or source.get("provider") != "schwab"
            or source.get("symbol") != decision["symbol"]
            or not timestamp(decision["timestamp_utc"]) <= captured <= now
        ):
            raise ValueError("review_capture_identity_source_or_time_invalid")
        report["chart_source"] = source
        report["capture_sha256"] = capture_sha256
        report["as_of_utc"] = source["fetch_started_at_utc"]
        report["timeframes"] = _frames(capture, asof=captured, available_at=captured)
        if any(f.get("chart_candles") for f in report["timeframes"].values()):
            report["status"] = "available"
    except (OSError, KeyError, ValueError, TypeError, AttributeError) as exc:
        report["issues"].append(str(exc))
    return report


def _inventory(root, directory):
    count, size = 0, 0
    with os.scandir(directory) as entries:
        for entry in entries:
            path = checked(root, directory / entry.name)
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError("unsafe_decision_chart_cache_entry")
            count += 1
            size += info.st_size
            if count > MAX_FILES or size > MAX_TOTAL_BYTES:
                raise ValueError("decision_chart_cache_budget_exceeded")
    return count, size


def _page(report, images):
    escape = lambda value: html.escape(str(value), quote=True)
    parts = [
        "<!doctype html><html lang='en'><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width'>",
        "<title>Decision candle report</title>",
        "<style>body{font:16px system-ui;margin:24px;max-width:1200px}"
        "img{width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>",
        f"<h1>{escape(report['symbol'])} decision chart</h1>",
        f"<p>Decision {escape(report['decision_id'])} at {escape(report['original_decision_timestamp_utc'])}</p>",
        f"<p>Chart as of {escape(report['as_of_utc'])}</p>",
        f"<p>Action: {escape(report['recorded_action'])}; gate decision: {escape(report['recorded_gate_decision'])}</p>",
        f"<p>{escape(report['interpretation'])}</p>",
        f"<p>Chart: {escape(report['status'])}; executions: {escape(report['execution_status'])}</p>",
    ]
    for name, frame in report["timeframes"].items():
        parts.append(f"<h2>{escape(name)}: {escape(frame['status'])}</h2>")
        parts.append(f"<p>{escape(', '.join(frame.get('issues', [])))}</p>")
        if name in images:
            parts.append(
                f"<img alt='{escape(name)} original candles and decision events' src='{escape(images[name])}'>"
            )
        diagnostics = {
            key: frame[key]
            for key in (
                "close",
                "sma20",
                "sma50",
                "rsi14_simple",
                "atr14_simple_usd",
                "prior_20_bar_low",
                "prior_20_bar_high",
                "trend",
                "gap_count",
                "recent_candles",
                "distance_from_sma20_bps",
                "distance_from_sma50_bps",
            )
            if key in frame
        }
        parts.append(
            "<h3>Review calculations, not claimed bot inputs</h3><pre>"
            + escape(json.dumps(diagnostics, indent=2))
            + "</pre>"
        )
    parts.append(
        "<h2>Recorded Indicator Reasoning</h2><pre>"
        + escape(json.dumps(report.get("recorded_reasoning", {}), indent=2))
        + "</pre>"
    )
    parts.append(
        "<pre>"
        + escape(
            json.dumps(
                {
                    "issues": report["issues"],
                    "events": report["decision_chart_markers"],
                },
                indent=2,
            )
        )
        + "</pre></html>"
    )
    return "\n".join(parts).encode()


def _asset_hash(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("unsafe_decision_chart_asset")
        raw = stream.read(MAX_REPORT_BYTES + 1)
    if len(raw) > MAX_REPORT_BYTES:
        raise ValueError("decision_chart_asset_budget_exceeded")
    return hashlib.sha256(raw).hexdigest()


def publish_decision_chart(
    root, decision, *, executions=(), now=None, review_capture_sha256=None
):
    """Return a small link receipt. Cache is content-addressed and hard bounded.

    Capacity exhaustion returns an explicit unavailable receipt; this report
    owner never prunes decisions, captures, logs or existing report links.
    """
    root = Path(root)
    ledger_issue = None
    if not executions:
        try:
            executions = read_ledger_executions(root, decision.get("decision_id"))
        except (OSError, ValueError, sqlite3.Error) as exc:
            ledger_issue = "execution_receipt_read_unavailable:" + type(exc).__name__
    report = (
        build_execution_review(
            root,
            decision,
            capture_sha256=review_capture_sha256,
            executions=executions,
            now=now,
        )
        if review_capture_sha256 is not None
        else build_decision_chart(root, decision, executions=executions, now=now)
    )
    if ledger_issue:
        report["issues"].append(ledger_issue)
    identity = digest(report)
    receipt = {
        "schema_version": 1,
        "report_sha256": identity,
        "decision_id": decision.get("decision_id"),
        "original_decision_timestamp_utc": decision.get("timestamp_utc"),
        "status": report["status"],
        "execution_status": report["execution_status"],
        "issues": report["issues"],
        "report_path": None,
        "sidecar_path": None,
        **AUTHORITY,
    }
    try:
        directory = checked(root, root / DIRECTORY)
        directory.mkdir(parents=True, exist_ok=True)
        lock = checked(root, directory / "writer.lock")
        fd = os.open(
            lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600
        )
        with os.fdopen(fd, "r+") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise ValueError("unsafe_decision_chart_lock")
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            count, size = _inventory(root, directory)
            page = checked(root, directory / f"{identity}.html")
            manifest = checked(root, directory / f"{identity}.json")
            if manifest.exists() and page.exists():
                saved = read_capture(root, manifest)
                assets = saved.pop("chart_asset_sha256", {})
                expected = {page.name} | {
                    f"{identity}_{name}.png"
                    for name, frame in report["timeframes"].items()
                    if frame.get("chart_candles")
                }
                if (
                    digest(saved) != identity
                    or set(assets) != expected
                    or any(
                        _asset_hash(checked(root, directory / name)) != assets[name]
                        for name in expected
                    )
                ):
                    raise ValueError("decision_chart_cache_corrupt_or_incomplete")
                receipt["report_path"] = str(page.relative_to(root))
                receipt["sidecar_path"] = str(manifest.relative_to(root))
                return receipt
            asset_count = 2 + sum(
                bool(f.get("chart_candles")) for f in report["timeframes"].values()
            )
            if (
                count + asset_count > MAX_FILES
                or size + MAX_REPORT_BYTES > MAX_TOTAL_BYTES
            ):
                raise ValueError("decision_chart_cache_budget_exceeded")
            # Scratch is bounded to one three-frame render under the writer lock.
            with tempfile.TemporaryDirectory(prefix="decision-chart-") as scratch:
                scratch = Path(scratch)
                images = render_charts(
                    report, directory=scratch, prefix=identity, checked_path=lambda p: p
                )
                payloads = {Path(p).name: Path(p).read_bytes() for p in images.values()}
                payloads[page.name] = _page(
                    report, {k: Path(p).name for k, p in images.items()}
                )
                stored = dict(
                    report,
                    chart_asset_sha256={
                        name: hashlib.sha256(raw).hexdigest()
                        for name, raw in payloads.items()
                    },
                )
                payloads[manifest.name] = json.dumps(
                    stored, sort_keys=True, allow_nan=False
                ).encode()
                if sum(map(len, payloads.values())) > MAX_REPORT_BYTES:
                    raise ValueError("decision_chart_report_budget_exceeded")
                for name, raw in payloads.items():
                    target = checked(root, directory / name)
                    temporary_fd, temporary = tempfile.mkstemp(
                        prefix=".chart-", dir=directory
                    )
                    try:
                        with os.fdopen(temporary_fd, "wb") as stream:
                            stream.write(raw)
                        os.replace(temporary, target)
                    finally:
                        if os.path.exists(temporary):
                            os.unlink(temporary)
            receipt["report_path"] = str(page.relative_to(root))
            receipt["sidecar_path"] = str(manifest.relative_to(root))
    except (OSError, ValueError) as exc:
        receipt.update(status="unavailable", issues=receipt["issues"] + [str(exc)])
    return receipt


def read_original_decision(root, path, *, decision_id):
    """Select an exact ID from a bounded JSONL tail or bounded gzip stream."""
    path = checked(root, path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("regular_decision_log_required")
        if path.suffix == ".gz":
            with gzip.GzipFile(fileobj=stream) as decoded:
                raw = decoded.read(MAX_LOG_BYTES + 1)
            if len(raw) > MAX_LOG_BYTES:
                raise ValueError("compressed_decision_scan_budget_exceeded")
        else:
            if info.st_size > MAX_LOG_BYTES:
                stream.seek(info.st_size - MAX_LOG_BYTES)
                stream.readline(MAX_INPUT_BYTES + 1)
            raw = stream.read(MAX_LOG_BYTES + 1)
        after = os.fstat(stream.fileno())
        if (info.st_size, info.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("decision_log_changed_during_scan_retry_later")
    selected = None
    for line in raw.splitlines():
        if not line.strip():
            continue
        if len(line) > MAX_INPUT_BYTES:
            raise ValueError("decision_line_budget_exceeded")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("decision_log_object_required")
        if row.get("decision_id") == decision_id:
            if selected is not None and digest(selected) != digest(row):
                raise ValueError("ambiguous_original_decision_id")
            selected = row
    if selected is None:
        raise ValueError("original_decision_not_found_in_bounded_scan")
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--decision-log",
        type=Path,
        help="Existing original decision JSONL or bounded .jsonl.gz; not explanation logs",
    )
    parser.add_argument("--decision-id")
    parser.add_argument(
        "--bitcoin-bot", help="Existing Bitcoin observer ID in its latest native report"
    )
    parser.add_argument(
        "--executions-json",
        type=Path,
        help="Existing reconciliation owner object with a receipts array",
    )
    parser.add_argument(
        "--review-capture-sha256",
        help="Explicit later stored capture; writes a distinct retrospective report",
    )
    args = parser.parse_args(argv)
    root = args.root.absolute()
    local = lambda p: p if p.is_absolute() else root / p
    try:
        if args.bitcoin_bot:
            if args.decision_log or args.decision_id:
                raise ValueError("choose_original_log_or_bitcoin_observation")
            watch = read_capture(
                root, root / "governance/health/bitcoin_price_watch_latest.json"
            )
            rows = [
                b["decision"]
                for b in watch.get("bots", [])
                if b.get("bot_id") == args.bitcoin_bot
            ]
            if len(rows) != 1:
                raise ValueError("bitcoin_observation_not_available")
            decision = rows[0]
        else:
            if not args.decision_log or not args.decision_id:
                raise ValueError("original_decision_log_and_id_required")
            decision = read_original_decision(
                root, local(args.decision_log), decision_id=args.decision_id
            )
        executions = ()
        if args.executions_json:
            executions = read_capture(root, local(args.executions_json))["receipts"]
        receipt = publish_decision_chart(
            root,
            decision,
            executions=executions,
            review_capture_sha256=args.review_capture_sha256,
        )
    except (OSError, KeyError, ValueError, TypeError) as exc:
        receipt = {
            "status": "unavailable",
            "issues": [str(exc)],
            "report_path": None,
            "sidecar_path": None,
            **AUTHORITY,
        }
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))
    return 0 if receipt["report_path"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
