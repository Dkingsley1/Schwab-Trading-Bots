"""Offline, bounded source census and targeted owner-response reconciliation.

No network, auth, scheduler or training writes. The coordinating owner performs
approved GETs, then supplies the exact request and response to import_response.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timedelta, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.decision_price_evidence import digest, number, session_bounds, timestamp
from core.research_data_platform import ResearchDataCatalog
from core.storage_router import inspect_storage_path

MAX_BYTES = 16 * 1024**2
MAX_TOTAL_BYTES = 64 * 1024**2
MAX_ROWS = 50000
MAX_FILES = 32
MAX_JOBS = 32
FRAMES = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "6h": 21600, "1d": 86400}
SCHWAB_METHODS = {
    "1m": "get_price_history_every_minute",
    "5m": "get_price_history_every_five_minutes",
    "1d": "get_price_history_every_day",
}
AUTHORITY = {
    "network_requests_performed": False,
    "historical_live_evidence": False,
    "training_readiness": False,
    "live_execution_authority": False,
}


class Budget:
    def __init__(self, seconds=20):
        if not 0 < number(seconds) <= 60:
            raise ValueError("invalid_deadline")
        self.deadline = time.monotonic() + seconds
        self.bytes = 0
        self.rows = 0

    def check(self, *, size=0, rows=0):
        self.bytes += size
        self.rows += rows
        if time.monotonic() >= self.deadline:
            raise ValueError("deadline_reached")
        if self.bytes > MAX_TOTAL_BYTES or self.rows > MAX_ROWS:
            raise ValueError("scan_budget_reached")


def safe_path(path, *, missing=False):
    route = inspect_storage_path(path)
    if route.get("status") not in (
        {"present", "missing"} if missing else {"present"}
    ) or route.get("symlinks"):
        raise ValueError("unsafe_or_unavailable_source")
    return Path(route["resolved_path"])


def _fingerprint(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def read_json(path, budget):
    """Hash complete physical bytes; limit both compressed and expanded input."""
    budget.check()
    path = safe_path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_BYTES:
            raise ValueError("bounded_regular_source_required")
        raw = stream.read(MAX_BYTES + 1)
        budget.check(size=len(raw))
        if len(raw) > MAX_BYTES or _fingerprint(before) != _fingerprint(
            os.fstat(stream.fileno())
        ):
            raise ValueError("source_changed_or_oversized")
    if _fingerprint(before) != _fingerprint(safe_path(path).stat()):
        raise ValueError("source_changed_during_scan")
    identity = hashlib.sha256(raw).hexdigest()
    if path.suffix == ".gz":
        import io

        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
            raw = stream.read(MAX_BYTES + 1)
        budget.check(size=len(raw))
        if len(raw) > MAX_BYTES:
            raise ValueError("expanded_source_too_large")
    value = json.loads(raw)
    digest(value)
    budget.check()
    return value, {
        "path": str(path),
        "sha256": identity,
        "bytes": before.st_size,
        "verification": "full_file_read",
        "format": "json.gz" if path.suffix == ".gz" else "json",
    }


def symbol(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,14}", value):
        raise ValueError("invalid_symbol")
    return value


def candle_scope(raw, now):
    provider, sym, frame = raw["provider"], symbol(raw["symbol"]), raw["timeframe"]
    if provider == "coinbase_exchange":
        if sym != "BTC-USD" or frame not in FRAMES:
            raise ValueError("unsupported_coinbase_scope")
    elif provider == "schwab":
        if sym == "BTC-USD" or frame not in SCHWAB_METHODS:
            raise ValueError("unsupported_schwab_scope")
    else:
        raise ValueError("unsupported_provider")
    start, end = timestamp(raw["start_utc"]), timestamp(raw["end_utc"])
    if not start < end <= now or end - start > timedelta(
        days=370 if frame == "1d" else 31
    ):
        raise ValueError("invalid_or_unbounded_window")
    if provider == "coinbase_exchange" and any(
        t.timestamp() % FRAMES[frame] for t in (start, end)
    ):
        raise ValueError("unaligned_crypto_window")
    return {
        "provider": provider,
        "symbol": sym,
        "timeframe": frame,
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "session": "XNYS_regular" if provider == "schwab" else "24x7",
    }


def intervals(scope, budget):
    start, end = timestamp(scope["start_utc"]), timestamp(scope["end_utc"])
    step = timedelta(seconds=FRAMES[scope["timeframe"]])
    result = []
    if scope["provider"] == "coinbase_exchange":
        cursor = start
        while cursor + step <= end:
            budget.check(rows=1)
            result.append((cursor.isoformat(), (cursor + step).isoformat()))
            cursor += step
    else:
        # Noon UTC identifies the NY session date without a UTC-midnight shift.
        cursor = datetime.combine(
            start.date() - timedelta(days=1), datetime.min.time(), timezone.utc
        ) + timedelta(hours=12)
        while cursor.date() <= end.date():
            budget.check()
            bounds = session_bounds(cursor)
            cursor += timedelta(days=1)
            if bounds is None:
                continue
            opened, closed = bounds
            if scope["timeframe"] == "1d":
                if start <= opened and closed <= end:
                    budget.check(rows=1)
                    result.append((opened.isoformat(), closed.isoformat()))
            else:
                bar = opened
                while bar + step <= closed:
                    if start <= bar and bar + step <= end:
                        budget.check(rows=1)
                        result.append((bar.isoformat(), (bar + step).isoformat()))
                    bar += step
    return result


def _ohlcv(raw):
    values = {
        key: number(raw[key], positive=True) for key in ("open", "high", "low", "close")
    }
    values["volume"] = number(raw["volume"])
    if (
        values["volume"] < 0
        or not values["low"]
        <= min(values["open"], values["close"])
        <= max(values["open"], values["close"])
        <= values["high"]
    ):
        raise ValueError("invalid_ohlcv")
    return values


def normalize_candles(payload, scope, observed, budget, *, native=False):
    """Return distinct bars; conflicting duplicates never earn coverage."""
    provider, frame = scope["provider"], scope["timeframe"]
    allowed = set(intervals(scope, budget))
    if native:
        raw = payload
    elif provider == "schwab":
        if not isinstance(payload, dict) or payload.get("symbol") != scope["symbol"]:
            raise ValueError("response_symbol_mismatch")
        raw = payload.get("candles")
    else:
        raw = payload
    if not isinstance(raw, list) or len(raw) > (6000 if provider == "schwab" else 300):
        raise ValueError("unbounded_or_invalid_candles")
    bars, conflicts, counts = {}, set(), Counter()
    for row in raw:
        budget.check(rows=1)
        if native:
            start, end = timestamp(row["start_utc"]), timestamp(row["end_utc"])
            values = _ohlcv(row)
        elif provider == "schwab":
            epoch = number(row["datetime"], positive=True)
            if epoch != int(epoch):
                raise ValueError("invalid_provider_timestamp")
            start = datetime.fromtimestamp(epoch / 1000, timezone.utc)
            if frame == "1d":
                bounds = session_bounds(start)
                if bounds is None:
                    raise ValueError("non_session_daily_bar")
                start, end = bounds
            else:
                end = start + timedelta(seconds=FRAMES[frame])
            values = _ohlcv(row)
        else:
            if not isinstance(row, list) or len(row) != 6:
                raise ValueError("invalid_exchange_candle")
            epoch = number(row[0], positive=True)
            if epoch != int(epoch) or epoch % FRAMES[frame]:
                raise ValueError("unaligned_exchange_candle")
            start = datetime.fromtimestamp(epoch, timezone.utc)
            end = start + timedelta(seconds=FRAMES[frame])
            values = _ohlcv(
                dict(zip(("low", "high", "open", "close", "volume"), row[1:]))
            )
        key = (start.isoformat(), end.isoformat())
        if end > observed:
            counts["unclosed_at_observation"] += 1
            continue
        if key not in allowed:
            counts["outside_requested_session_or_interval"] += 1
            continue
        bar = dict(values, start_utc=key[0], end_utc=key[1])
        if key in conflicts:
            continue
        if key in bars:
            if bars[key] == bar:
                counts["identical_duplicates"] += 1
            else:
                counts["conflicting_duplicates"] += 1
                conflicts.add(key)
                del bars[key]
        else:
            bars[key] = bar
    return list(bars.values()), dict(counts), conflicts


def candle_request(scope):
    body = dict(
        scope,
        product="candles",
        max_response_bytes=MAX_BYTES,
        timeout_seconds=10,
        retries=0,
    )
    if scope["provider"] == "schwab":
        start_arg, end_arg = scope["start_utc"], scope["end_utc"]
        if scope["timeframe"] == "1d":
            # Daily provider stamps precede session open. Request the exact NY
            # date while retaining session open/close as the coverage key.
            ny = ZoneInfo("America/New_York")
            expected = intervals(scope, Budget())
            if not expected:
                raise ValueError("no_closed_daily_session")
            first = (
                timestamp(expected[0][0])
                .astimezone(ny)
                .replace(hour=0, minute=0, second=0, microsecond=0)
            )
            last = (
                timestamp(expected[-1][1])
                .astimezone(ny)
                .replace(hour=0, minute=0, second=0, microsecond=0)
            )
            start_arg = first.astimezone(timezone.utc).isoformat()
            end_arg = (
                (last + timedelta(days=1) - timedelta(milliseconds=1))
                .astimezone(timezone.utc)
                .isoformat()
            )
        body.update(
            owner="market_data_collectors",
            endpoint=SCHWAB_METHODS[scope["timeframe"]],
            dataset_id="broker_market_observations_v1",
            request_parameters={
                "symbol": scope["symbol"],
                "start_datetime": start_arg,
                "end_datetime": end_arg,
                "need_extended_hours_data": False,
                "need_previous_close": False,
            },
        )
    else:
        body.update(
            owner="scripts/ops/crypto_research.py",
            endpoint="https://api.exchange.coinbase.com/products/BTC-USD/candles",
            dataset_id="broker_market_observations_v1",
            request_parameters={
                "start": scope["start_utc"],
                "end": scope["end_utc"],
                "granularity": FRAMES[scope["timeframe"]],
            },
        )
    return dict(body, request_id=digest(body))


def missing_requests(scope, missing):
    groups = []
    for key in missing:
        # One regular session per request; no request bridges observed bars.
        if groups and groups[-1][-1][1] == key[0] and len(groups[-1]) < 299:
            groups[-1].append(key)
        else:
            groups.append([key])
    return [
        candle_request(dict(scope, start_utc=g[0][0], end_utc=g[-1][1])) for g in groups
    ]


def fred_request(raw, now):
    series = symbol(raw["series_id"])
    start, end, vintage = (
        date.fromisoformat(raw[k])
        for k in ("observation_start", "observation_end", "vintage_date")
    )
    if (
        not start <= end <= now.date()
        or vintage > now.date()
        or end - start > timedelta(days=370)
    ):
        raise ValueError("invalid_fred_window")
    body = {
        "product": "fred_vintage",
        "provider": "fred",
        "series_id": series,
        "owner": "scripts/collect_bls_census_data.py",
        "dataset_id": "official_us_macro_v1",
        "endpoint": "https://api.stlouisfed.org/fred/series/observations",
        "request_parameters": {
            "series_id": series,
            "observation_start": start.isoformat(),
            "observation_end": end.isoformat(),
            "realtime_start": vintage.isoformat(),
            "realtime_end": vintage.isoformat(),
            "file_type": "json",
            "output_type": 1,
            "limit": 1000,
            "offset": 0,
            "sort_order": "asc",
        },
        "max_response_bytes": MAX_BYTES,
        "timeout_seconds": 10,
        "retries": 0,
    }
    return dict(body, request_id=digest(body))


def dividend_request(raw, now):
    sym = symbol(raw["symbol"])
    start, end = timestamp(raw["start_utc"]), timestamp(raw["end_utc"])
    account = raw["account_reference_sha256"]
    if sym == "BTC-USD" or not start < end <= now or end - start > timedelta(days=31):
        raise ValueError("invalid_dividend_window")
    if not isinstance(account, str) or not re.fullmatch(r"[0-9a-f]{64}", account):
        raise ValueError("opaque_account_binding_required")
    body = {
        "product": "schwab_dividends",
        "provider": "schwab",
        "symbol": sym,
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "account_reference_sha256": account,
        "endpoint": "get_transactions",
        "owner": "scripts/ops/supervised_broker_test.py:transaction_observations",
        "dataset_id": "broker_market_observations_v1",
        "request_parameters": {
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        },
        "max_response_bytes": MAX_BYTES,
        "timeout_seconds": 10,
        "retries": 0,
    }
    return dict(body, request_id=digest(body))


def normalize_dividends(job, payload, observed, budget):
    from scripts.collect_dividend_drip_state import _classify_dividend_transaction

    if (
        not isinstance(payload, dict)
        or payload.get("account_reference_sha256") != job["account_reference_sha256"]
        or payload.get("window_start_utc") != job["start_utc"]
        or payload.get("window_end_utc") != job["end_utc"]
    ):
        raise ValueError("transaction_owner_binding_mismatch")
    original = payload.get("rows")
    if not isinstance(original, list) or len(original) > 1000:
        raise ValueError("bounded_transaction_response_required")
    if observed < timestamp(job["end_utc"]):
        raise ValueError("response_observed_before_requested_end")
    issues = (
        []
        if payload.get("source_complete") is True and len(original) < 1000
        else ["transaction_source_incomplete"]
    )
    rows, seen = [], {}
    for raw in original:
        budget.check(rows=1)
        if not isinstance(raw, dict):
            raise ValueError("invalid_transaction_row")
        event = _classify_dividend_transaction(raw)
        if not event or event["symbol"] != job["symbol"]:
            continue
        event_id = raw.get("activityId") or raw.get("transactionId")
        when = (
            raw.get("transactionDate")
            or raw.get("settlementDate")
            or raw.get("tradeDate")
        )
        try:
            # The native classifier accepts naive dates. Do not turn that into a
            # certified UTC instant for this stricter historical receipt.
            event_time = timestamp(when)
            if not event_id or isinstance(raw.get("netAmount"), bool):
                raise ValueError("transaction_identity_or_amount_missing")
            amount = number(float(raw["netAmount"]))
        except (ValueError, TypeError, KeyError):
            issues.append("unresolved_dividend_transaction")
            continue
        if not timestamp(job["start_utc"]) <= event_time < timestamp(job["end_utc"]):
            issues.append("dividend_transaction_outside_window")
            continue
        identity = digest(str(event_id))
        row = {
            "symbol": job["symbol"],
            "transaction_id_sha256": identity,
            "event_type": event["event_type"],
            "effective_at_utc": event_time.isoformat(),
            "signed_net_amount": amount,
            "issuer_ex_date": None,
            "publication_at_utc": None,
        }
        if identity in seen:
            if seen[identity] != row:
                raise ValueError("conflicting_dividend_transaction")
            continue
        seen[identity] = row
        rows.append(row)
    return (
        rows,
        issues,
        {
            "source_rows": len(original),
            "distinct_dividend_postings": len(rows),
            "issuer_corporate_action_history_complete": False,
        },
    )


def _check_request(job, now):
    if job.get("product") == "candles":
        scope = candle_scope(job, now)
        if job != candle_request(scope) or len(intervals(scope, Budget())) > 299:
            raise ValueError("request_binding_or_size_invalid")
    elif job.get("product") == "fred_vintage":
        p = job["request_parameters"]
        expected = fred_request(
            {
                "series_id": job["series_id"],
                "observation_start": p["observation_start"],
                "observation_end": p["observation_end"],
                "vintage_date": p["realtime_start"],
            },
            now,
        )
        if job != expected:
            raise ValueError("request_binding_invalid")
    elif job.get("product") == "schwab_dividends":
        if job != dividend_request(job, now):
            raise ValueError("request_binding_invalid")
    else:
        raise ValueError("unsupported_response_product")


def import_response(
    job, payload, *, observed_at_utc, source_endpoint, now=None, budget=None
):
    """Validate an owner-supplied response; never manufacture collection times.

    observed_at_utc is the real fetch-completion time supplied by that owner.
    The receipt's hash binds bytes and scope, not authenticity or entitlement.
    """
    now = now or datetime.now(timezone.utc)
    budget = budget or Budget()
    _check_request(job, now)
    observed = timestamp(observed_at_utc)
    if observed > now or source_endpoint != job["endpoint"]:
        raise ValueError("future_observation_or_source_mismatch")
    raw = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
    budget.check(size=len(raw))
    if len(raw) > MAX_BYTES:
        raise ValueError("response_too_large")
    issues = []
    if job["product"] == "candles":
        scope = candle_scope(job, now)
        if observed < timestamp(scope["end_utc"]):
            raise ValueError("response_observed_before_requested_end")
        rows, counts, conflicts = normalize_candles(payload, scope, observed, budget)
        expected = intervals(scope, budget)
        have = {(r["start_utc"], r["end_utc"]) for r in rows}
        missing = [k for k in expected if k not in have]
        if missing:
            issues.append("missing_buckets_no_trade_or_unavailable_not_established")
        if conflicts:
            issues.append("conflicting_duplicate_bars")
        coverage = {
            "expected": len(expected),
            "present": len(rows),
            "missing": len(missing),
            "counts": counts,
        }
    elif job["product"] == "schwab_dividends":
        rows, issues, coverage = normalize_dividends(job, payload, observed, budget)
    else:
        p = job["request_parameters"]
        if date.fromisoformat(p["realtime_start"]) > observed.date():
            raise ValueError("vintage_after_observation")
        if not isinstance(payload, dict) or not isinstance(
            payload.get("observations"), list
        ):
            raise ValueError("invalid_fred_response")
        if (
            payload.get("realtime_start") != p["realtime_start"]
            or payload.get("realtime_end") != p["realtime_end"]
        ):
            raise ValueError("fred_vintage_mismatch")
        original = payload["observations"]
        if (
            len(original) > 1000
            or payload.get("offset") != 0
            or type(payload.get("count")) is not int
        ):
            raise ValueError("invalid_fred_pagination")
        if payload["count"] != len(original):
            issues.append("fred_pagination_incomplete")
        rows, seen = [], {}
        for row in original:
            budget.check(rows=1)
            period = date.fromisoformat(row["date"])
            if not p["observation_start"] <= period.isoformat() <= p["observation_end"]:
                raise ValueError("fred_period_outside_request")
            if (
                row.get("realtime_start") != p["realtime_start"]
                or row.get("realtime_end") != p["realtime_end"]
            ):
                raise ValueError("fred_row_vintage_mismatch")
            value = None if row["value"] == "." else number(float(row["value"]))
            if value is None:
                issues.append("fred_missing_value")
            if period in seen:
                if seen[period] != value:
                    raise ValueError("fred_conflicting_duplicate")
                continue
            seen[period] = value
            rows.append(
                {
                    "series_id": job["series_id"],
                    "period": period.isoformat(),
                    "value": value,
                    "vintage_date": p["realtime_start"],
                    "provider_realtime_end": p["realtime_end"],
                    "effective_at_utc": period.isoformat() + "T00:00:00+00:00",
                    "event_time_precision": "date",
                    "publication_at_utc": None,
                }
            )
        if not rows:
            issues.append("empty_fred_response")
        coverage = {
            "provider_count": payload["count"],
            "distinct_periods": len(rows),
            "release_calendar_completeness": "not_established",
        }
    for row in rows:
        row.update(
            observed_at_utc=observed.isoformat(),
            known_at_utc=observed.isoformat(),
            evidence_kind="historical_backfill",
            historical_live_evidence=False,
        )
    body = {
        "schema_version": 1,
        "kind": "targeted_collection_receipt",
        "request": job,
        "source_endpoint": source_endpoint,
        "observed_at_utc": observed.isoformat(),
        "payload_sha256": digest(payload),
        "rows": rows,
        "coverage": coverage,
        "issues": sorted(set(issues)),
        "status": "incomplete" if issues else "response_reconciled",
        "price_adjustment_basis": "provider_as_returned_not_independently_verified",
        "authenticity": "owner_supplied_response_not_independently_attested",
        **AUTHORITY,
    }
    budget.check()
    return dict(body, receipt_sha256=digest(body))


def validate_receipt(value, now, budget):
    body = {k: v for k, v in value.items() if k != "receipt_sha256"}
    if value.get("kind") != "targeted_collection_receipt" or digest(body) != value.get(
        "receipt_sha256"
    ):
        raise ValueError("invalid_receipt_digest")
    _check_request(value["request"], now)
    if value.get("source_endpoint") != value["request"]["endpoint"]:
        raise ValueError("receipt_source_mismatch")
    observed = timestamp(value["observed_at_utc"])
    authority_keys = set(AUTHORITY) - {"network_requests_performed"}
    network = value.get("network_requests_performed")
    if (
        observed > now
        or any(value.get(k) is not False for k in authority_keys)
        or type(network) is not bool
        or (
            network
            and value.get("collection_owner")
            not in {
                "collection_gap_census.fetch_public_coinbase_receipt",
                "collection_gap_census.fetch_schwab_receipt",
            }
        )
    ):
        raise ValueError("receipt_time_or_authority_invalid")
    if not isinstance(value.get("rows"), list) or len(value["rows"]) > 1000:
        raise ValueError("receipt_rows_invalid")
    for row in value["rows"]:
        budget.check(rows=1)
        if (
            row.get("observed_at_utc") != observed.isoformat()
            or row.get("known_at_utc") != observed.isoformat()
            or row.get("historical_live_evidence") is not False
        ):
            raise ValueError("receipt_knowledge_time_invalid")
    if value["request"]["product"] == "fred_vintage":
        request = value["request"]
        p = request["request_parameters"]
        if date.fromisoformat(p["realtime_start"]) > observed.date():
            raise ValueError("vintage_after_observation")
        periods = set()
        for row in value["rows"]:
            period = date.fromisoformat(row["period"]).isoformat()
            if (
                row.get("series_id") != request["series_id"]
                or row.get("vintage_date") != p["realtime_start"]
                or not p["observation_start"] <= period <= p["observation_end"]
                or period in periods
            ):
                raise ValueError("invalid_vintage_receipt_row")
            periods.add(period)
            if row.get("value") is not None:
                number(row["value"])
        if value["status"] == "response_reconciled" and (
            not periods
            or value["issues"]
            or any(r.get("value") is None for r in value["rows"])
            or value["coverage"].get("provider_count") != len(periods)
        ):
            raise ValueError("incomplete_vintage_receipt")
    return value


def reconcile_symbols(root, selected, now, budget):
    """Reuse native position parsing; expose no account identifiers or balances."""
    from scripts.ops.account_position_study import _positions

    declared, failures = set(), []
    for relative in (
        "config/supervised_broker_test_v1.json",
        "config/supervised_schd_broker_test_v1.json",
    ):
        try:
            value, _ = read_json(root / relative, budget)
            declared.add(symbol(value["symbol"]))
        except (OSError, ValueError, KeyError, TypeError):
            failures.append(relative + ":unavailable")
    # This fixed native observer contract is explicitly BTC-only.
    if safe_path(root / "scripts/ops/bitcoin_price_watch.py", missing=True).exists():
        declared.add("BTC-USD")
    held, excluded = set(), []
    source_time, source_sha, state = None, None, "unavailable"
    try:
        snapshot, receipt = read_json(
            root / "governance/health/broker_truth_shared_snapshot_schwab_latest.json",
            budget,
        )
        source_sha = receipt["sha256"]
        fetched = snapshot["fetched"]
        if snapshot.get("broker") != "schwab" or fetched.get("ok") is not True:
            raise ValueError("invalid_broker_snapshot")
        source_time = timestamp(
            fetched.get("_forced_account_snapshot_refreshed_at_utc")
            or snapshot["timestamp_utc"]
        )
        age = (now - source_time).total_seconds()
        state = "fresh" if 0 <= age <= 900 else "stale_or_future"
        if fetched.get("account_snapshot_partial") is not False:
            state = "partial"
        for position in _positions(snapshot):
            budget.check(rows=1)
            if position["asset_type"] != "EQUITY":
                excluded.append(
                    {"symbol": position["symbol"], "asset_type": position["asset_type"]}
                )
                continue
            held.add(symbol(position["symbol"]))
    except (OSError, ValueError, KeyError, TypeError):
        failures.append("broker_holdings_unavailable_or_incomplete")
    return {
        "selected_research_symbols": sorted(selected),
        "native_research_symbols": sorted(declared),
        "held_equity_symbols": sorted(held),
        "held_but_not_selected": sorted(held - selected),
        "selected_not_in_native_research": sorted(selected - declared),
        "holdings_state": state,
        "holdings_source_at_utc": source_time.isoformat() if source_time else None,
        "holdings_source_sha256": source_sha,
        "excluded_non_equity_positions": excluded,
        "coinbase_holdings": "not_verified",
        "issues": failures,
        "automatic_universe_expansion": False,
    }


def census(root, manifest, *, now=None, budget=None):
    now = now or datetime.now(timezone.utc)
    budget = budget or Budget()
    if not isinstance(manifest, dict):
        raise ValueError("manifest_object_required")
    scopes_raw, sources = manifest.get("candles", []), manifest.get("sources", [])
    if (
        not isinstance(scopes_raw, list)
        or not 1 <= len(scopes_raw) <= 12
        or not isinstance(sources, list)
        or len(sources) > MAX_FILES
    ):
        raise ValueError("bounded_explicit_manifest_required")
    scopes = [candle_scope(s, now) for s in scopes_raw]
    if len({digest(s) for s in scopes}) != len(scopes):
        raise ValueError("duplicate_scope")
    policy, _ = read_json(root / "config/research_data_platform_v1.json", budget)
    catalog = ResearchDataCatalog(policy)
    authorization = {
        k: catalog.authorize_use(k, "research")
        for k in ("broker_market_observations_v1", "official_us_macro_v1")
    }
    inventory, loaded, blocked = [], [], False
    for item in sources:
        budget.check()
        try:
            path = Path(item["path"])
            value, receipt = read_json(
                path if path.is_absolute() else root / path, budget
            )
            if not isinstance(value, dict):
                raise ValueError("source_object_required")
            if item["kind"] == "receipt":
                validate_receipt(value, now, budget)
            elif item["kind"] == "native_schwab_capture":
                source = value["source"]
                if (
                    source.get("provider") != "schwab"
                    or timestamp(source["fetch_started_at_utc"]) > now
                ):
                    raise ValueError("native_source_invalid")
                symbol(source["symbol"])
            else:
                raise ValueError("unsupported_source_format")
            inventory.append(dict(receipt, kind=item["kind"], status="read"))
            loaded.append((item["kind"], value, receipt["sha256"]))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            budget.check()
            blocked = True
            inventory.append(
                {
                    "path": item.get("path"),
                    "status": "unavailable_or_unsupported",
                    "error_type": type(exc).__name__,
                }
            )
    products, jobs = [], []
    for scope in scopes:
        expected = intervals(scope, budget)
        by_key, conflicts, counts, physical = {}, set(), Counter(), []
        for kind, value, sha in loaded:
            budget.check()
            if kind == "native_schwab_capture":
                source = value["source"]
                if scope["provider"] != "schwab" or source["symbol"] != scope["symbol"]:
                    continue
                rows = value.get("candles", {}).get(scope["timeframe"], [])
                observed = timestamp(source["fetch_started_at_utc"])
            else:
                req = value["request"]
                if req["product"] != "candles" or any(
                    req.get(k) != scope[k] for k in ("provider", "symbol", "timeframe")
                ):
                    continue
                rows = value["rows"]
                observed = timestamp(value["observed_at_utc"])
            try:
                bars, count, bad = normalize_candles(
                    rows, scope, observed, budget, native=True
                )
            except (ValueError, KeyError, TypeError):
                budget.check()
                blocked = True
                counts["invalid_source_payload"] += 1
                continue
            physical.append(sha)
            counts.update(count)
            conflicts.update(bad)
            for row in bars:
                key = (row["start_utc"], row["end_utc"])
                if key in by_key:
                    if by_key[key] != row:
                        conflicts.add(key)
                    else:
                        counts["duplicate_physical_bar"] += 1
                else:
                    by_key[key] = row
        have = set(by_key) - conflicts
        missing = [k for k in expected if k not in have]
        requests = missing_requests(scope, missing)
        jobs.extend(requests)
        products.append(
            {
                "scope": scope,
                "expected_closed_bars": len(expected),
                "distinct_bars": len(have),
                "missing_bars": len(missing),
                "conflicting_bars": len(conflicts),
                "counts": dict(counts),
                "physical_source_sha256": sorted(set(physical)),
                "missing_intervals": [
                    {"start_utc": j["start_utc"], "end_utc": j["end_utc"]}
                    for j in requests
                ],
                "status": (
                    "unresolved_gaps"
                    if missing or conflicts
                    else "covered_in_selected_sources"
                ),
                "price_adjustment_basis": "unverified",
                "missing_bucket_cause": "not_established",
            }
        )
    macro = manifest.get("fred_vintages", [])
    if not isinstance(macro, list) or len(macro) > 8:
        raise ValueError("bounded_fred_scope_required")
    macro_products = []
    for request in macro:
        job = fred_request(request, now)
        matches = [
            v for kind, v, _ in loaded if kind == "receipt" and v["request"] == job
        ]
        ready = any(v["status"] == "response_reconciled" for v in matches)
        macro_products.append(
            {
                "series_id": job["series_id"],
                "vintage_date": request["vintage_date"],
                "status": "response_reconciled" if ready else "missing_or_incomplete",
                "release_calendar_completeness": "not_established",
            }
        )
        if not ready:
            jobs.append(job)
    dividends = manifest.get("dividends", [])
    if not isinstance(dividends, list) or len(dividends) > 8:
        raise ValueError("bounded_dividend_scope_required")
    dividend_products = []
    for request in dividends:
        job = dividend_request(request, now)
        if job["symbol"] not in {
            s["symbol"] for s in scopes if s["provider"] == "schwab"
        }:
            raise ValueError("dividend_symbol_outside_selected_scope")
        matches = [
            v for kind, v, _ in loaded if kind == "receipt" and v["request"] == job
        ]
        ready = any(v["status"] == "response_reconciled" for v in matches)
        dividend_products.append(
            {
                "symbol": job["symbol"],
                "request_id": job["request_id"],
                "status": (
                    "account_postings_reconciled" if ready else "missing_or_incomplete"
                ),
                "issuer_action_history": "unverified",
            }
        )
        if not ready:
            jobs.append(job)
    selected = {s["symbol"] for s in scopes}
    equities = sorted({s["symbol"] for s in scopes if s["provider"] == "schwab"})
    jobs = list({job["request_id"]: job for job in jobs}.values())
    dispatch = {
        "state": "blocked_pending_main_coordination",
        "catalog_authorizations": authorization,
        "required_checks": [
            "fresh_owner_storage_admission",
            "current_provider_entitlement",
            "hard_child_deadline",
            "reconcile_catalog_before_dispatch",
        ],
        "download_authorized": False,
    }
    budget.check()
    return {
        "schema_version": 1,
        "timestamp_utc": now.isoformat(),
        "manifest_sha256": digest(manifest),
        "scope": "explicit_selected_sources_only_not_lifetime_archive",
        "lifetime_coverage_certified": False,
        "inventory_status": "incomplete" if blocked else "selected_sources_read",
        "catalog": inventory,
        "symbols": reconcile_symbols(root, selected, now, budget),
        "candles": products,
        "fred_vintages": macro_products,
        "dividends": dividend_products,
        "corporate_actions": {
            "symbols": equities,
            "status": "unavailable_not_certified",
            "dividend_owner": "scripts/collect_dividend_drip_state.py",
            "targeted_transaction_owner": "scripts/ops/supervised_broker_test.py:transaction_observations",
            "blockers": [
                "account_transactions_are_not_a_complete_issuer_action_calendar",
                "historical_ex_dates_splits_and_adjustment_entitlement_unverified",
            ],
            "automatic_download": False,
        },
        "requests": jobs[:MAX_JOBS],
        "dispatch": dispatch,
        "deferred_request_count": max(0, len(jobs) - MAX_JOBS),
        "request_count": len(jobs),
        "downloads_enabled": False,
        **AUTHORITY,
    }


def fetch_public_coinbase_receipt(job):
    """One explicitly selected missing interval, GET only, no automatic retries.

    This collects quarantined research evidence; catalog authorization and
    historical point-in-time eligibility are not granted by a successful GET.
    """
    import http.client
    from urllib.parse import urlencode

    _check_request(job, datetime.now(timezone.utc))
    if job.get("provider") != "coinbase_exchange" or job.get("product") != "candles":
        raise ValueError("public_coinbase_candle_request_required")
    connection = http.client.HTTPSConnection("api.exchange.coinbase.com", timeout=10)
    try:
        connection.request(
            "GET",
            "/products/BTC-USD/candles?" + urlencode(job["request_parameters"]),
            headers={"User-Agent": "SchwabPlatform-TargetedBackfill/1"},
        )
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError("public_backfill_provider_unavailable")
        raw = response.read(512 * 1024 + 1)
        if len(raw) > 512 * 1024:
            raise ValueError("public_backfill_response_oversize")
        observed = datetime.now(timezone.utc)
        payload = json.loads(raw)
    finally:
        connection.close()
    result = import_response(
        job,
        payload,
        observed_at_utc=observed.isoformat(),
        source_endpoint=job["endpoint"],
        now=observed,
    )
    result.pop("receipt_sha256")
    result.update(
        network_requests_performed=True,
        collection_owner="collection_gap_census.fetch_public_coinbase_receipt",
    )
    return dict(result, receipt_sha256=digest(result))


def fetch_schwab_receipt(root, job):
    """Isolated child: exact reviewed history interval, no account/order APIs."""
    import contextlib
    import io
    from scripts.ops.schd_candle_report import READ_ONLY_ENV

    os.environ.update(READ_ONLY_ENV)
    _check_request(job, datetime.now(timezone.utc))
    if job.get("provider") != "schwab" or job.get("product") != "candles":
        raise ValueError("schwab_candle_request_required")
    from core.provider_access_guard import provider_access_status, provider_request_slot
    from scripts.brokers.schwab.common import build_schwab_trader

    if provider_access_status(root, "schwab").get("active"):
        raise ValueError("schwab_provider_cooldown_active")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        client = build_schwab_trader(root, mode="shadow").authenticate()
        with provider_request_slot(
            root, "schwab", job["symbol"], slot_count=2, wait_seconds=10
        ):
            parameters = job["request_parameters"]
            response = getattr(client, job["endpoint"])(
                job["symbol"],
                start_datetime=timestamp(parameters["start_datetime"]),
                end_datetime=timestamp(parameters["end_datetime"]),
                need_extended_hours_data=False,
                need_previous_close=False,
            )
            response.raise_for_status()
            payload = response.json()
    observed = datetime.now(timezone.utc)
    result = import_response(
        job,
        payload,
        observed_at_utc=observed.isoformat(),
        source_endpoint=job["endpoint"],
        now=observed,
    )
    result.pop("receipt_sha256")
    result.update(
        network_requests_performed=True,
        collection_owner="collection_gap_census.fetch_schwab_receipt",
    )
    return dict(result, receipt_sha256=digest(result))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--response", type=Path)
    parser.add_argument("--observed-at-utc")
    parser.add_argument("--source-endpoint")
    parser.add_argument(
        "--fetch-public-coinbase",
        action="store_true",
        help="One explicit missing-interval request; no orders or research-use authorization",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="New receipt under root/governance/collection_backfills only",
    )
    parser.add_argument(
        "--fetch-schwab",
        action="store_true",
        help="One explicit Schwab candle interval in a read-only bounded child",
    )
    parser.add_argument("--schwab-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        budget = Budget()
        if args.schwab_child:
            from core.schd_capture_store import checked

            if checked(
                args.root, args.root / "governance/health/SYSTEM_POWER_OFF.flag"
            ).exists():
                raise ValueError("system_power_off")
            job, _ = read_json(args.request, budget)
            print(json.dumps(fetch_schwab_receipt(args.root, job), allow_nan=False))
            return 0
        if args.fetch_public_coinbase or args.fetch_schwab:
            from core.schd_capture_store import checked
            from core.decision_candle_store import _atomic

            if (
                not args.request
                or not args.output
                or any(
                    (
                        args.manifest,
                        args.response,
                        args.observed_at_utc,
                        args.source_endpoint,
                    )
                )
            ):
                raise ValueError("explicit_request_and_new_receipt_output_required")
            root = args.root.absolute()
            if checked(root, root / "governance/health/SYSTEM_POWER_OFF.flag").exists():
                raise ValueError("system_power_off")
            output = args.output if args.output.is_absolute() else root / args.output
            output = Path(os.path.abspath(output))
            output.relative_to(root / "governance/collection_backfills")
            checked(root, output)
            if output.exists():
                raise ValueError("backfill_receipt_already_exists_reconcile_first")
            job, _ = read_json(args.request, budget)
            if args.fetch_public_coinbase and args.fetch_schwab:
                raise ValueError("one_provider_per_request")
            if args.fetch_schwab:
                import subprocess
                from scripts.ops.schd_candle_report import READ_ONLY_ENV

                _check_request(job, datetime.now(timezone.utc))
                try:
                    child = subprocess.run(
                        [
                            sys.executable,
                            __file__,
                            "--root",
                            str(root),
                            "--request",
                            str(args.request.absolute()),
                            "--schwab-child",
                        ],
                        cwd=root,
                        env=dict(os.environ, **READ_ONLY_ENV),
                        capture_output=True,
                        text=True,
                        timeout=90,
                        check=False,
                    )
                except subprocess.TimeoutExpired:
                    raise ValueError("bounded_schwab_backfill_timeout") from None
                if child.returncode or len(child.stdout) > 1024 * 1024:
                    raise ValueError("schwab_backfill_failed_check_provider_health")
                result = json.loads(child.stdout)
                validate_receipt(result, datetime.now(timezone.utc), Budget())
            else:
                result = fetch_public_coinbase_receipt(job)
            output.parent.mkdir(parents=True, exist_ok=True)
            _atomic(root, output, json.dumps(result, sort_keys=True, allow_nan=False).encode(), immutable=True)
            print(
                json.dumps(
                    {
                        "status": result.get("status"),
                        "coverage": result.get("coverage"),
                        "output": str(output),
                        "issues": result.get("issues"),
                        **AUTHORITY,
                        "network_requests_performed": True,
                    }
                )
            )
            return 0
        if args.output:
            raise ValueError("output_requires_explicit_public_backfill")
        if args.manifest and not any(
            (args.request, args.response, args.observed_at_utc, args.source_endpoint)
        ):
            manifest, _ = read_json(args.manifest, budget)
            result = census(args.root, manifest, budget=budget)
        elif not args.manifest and all(
            (args.request, args.response, args.observed_at_utc, args.source_endpoint)
        ):
            job, _ = read_json(args.request, budget)
            payload, _ = read_json(args.response, budget)
            result = import_response(
                job,
                payload,
                observed_at_utc=args.observed_at_utc,
                source_endpoint=args.source_endpoint,
                budget=budget,
            )
        else:
            raise ValueError("choose_manifest_or_complete_response_import")
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # Never print provider bodies, credentials, account IDs or arbitrary paths.
        print(
            json.dumps(
                {"status": "incomplete", "error_type": type(exc).__name__, **AUTHORITY}
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
