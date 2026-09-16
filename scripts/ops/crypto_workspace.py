"""Local-only Crypto UI: cached account read, evidence read, bounded research."""

from __future__ import annotations

import ipaddress
import json
import math
import os
import re
import stat
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import urlparse

ASSETS = Path(__file__).with_name("crypto_assets")
RESEARCH = Path(__file__).with_name("crypto_research.py")
REPLAY_LOCK = threading.Lock()
REPLAYS: dict = {}
LAST_ATTEMPT: dict = {}


def age_seconds(value, now: datetime) -> float | None:
    try:
        measured = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if measured.tzinfo is None:
            return None
        age = (now - measured).total_seconds()
        return round(age, 1) if age >= 0 else None
    except (ValueError, TypeError, OverflowError):
        return None


def evidence(
    root: Path, filename: str, now: datetime, max_age=300
) -> tuple[dict, dict]:
    # Fixed local files only. No storage routing, broad scans or volume fallbacks.
    path = root / "governance" / "health" / filename
    payload = {}
    try:
        for parent in (path, *path.parents):
            if parent.is_symlink():
                raise ValueError()
        with os.fdopen(
            os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb"
        ) as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise ValueError()
            data = handle.read(4 * 1024 * 1024 + 1)
        if len(data) > 4 * 1024 * 1024:
            raise ValueError()
        payload = json.loads(data)
        if not isinstance(payload, dict):
            raise ValueError()
    except (OSError, ValueError):
        payload = {}
    timestamp = payload.get("timestamp_utc")
    age = age_seconds(timestamp, now)
    return payload, {
        "source": filename,
        "timestamp_utc": timestamp if isinstance(timestamp, str) else None,
        "age_seconds": age,
        "fresh": age is not None and age <= max_age,
        "present": bool(payload),
    }


def safe_codes(raw) -> list:
    if not isinstance(raw, list):
        return []
    return [
        x
        for x in raw[:50]
        if isinstance(x, str) and re.fullmatch(r"[a-zA-Z0-9_:. -]{1,160}", x)
    ]


def count(value) -> int | None:
    return value if type(value) is int and 0 <= value < 10**12 else None


def portfolio(state_dir=None, now=None) -> dict:
    from scripts.ops.coinbase_account_link import DEFAULT_STATE_DIR, _read_json
    from core.coinbase_account import CoinbaseAccountError

    now = now or datetime.now(timezone.utc)
    state_dir = state_dir or DEFAULT_STATE_DIR
    result = {
        "status": "unavailable",
        "configured": False,
        "fresh": False,
        "verified_at": None,
        "age_seconds": None,
        "balances": [],
        "valuation_included": False,
        "live_execution_allowed": False,
        "transfers_allowed": False,
    }
    try:
        connection = _read_json(state_dir / "connection.json")
        snapshot = connection.get("snapshot")
        if not isinstance(snapshot, dict) or snapshot.get("complete") is not True:
            raise ValueError()
        rows = snapshot.get("accounts")
        if not isinstance(rows, list) or len(rows) > 2500:
            raise ValueError()
        stamp = connection.get("verified_at")
        age = age_seconds(stamp, now)
        if age is None:
            raise ValueError()
        totals = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError()
            currency = row["currency"]
            if not isinstance(currency, str) or not re.fullmatch(
                r"[A-Z0-9]{1,20}", currency
            ):
                raise ValueError()
            amounts = [Decimal(str(row[key])) for key in ("available", "hold")]
            if any(not x.is_finite() or x < 0 or x > Decimal("1e18") for x in amounts):
                raise ValueError()
            current = totals.setdefault(currency, [Decimal(0), Decimal(0)])
            for i, amount in enumerate(amounts):
                current[i] += amount
        balances = [
            {
                "currency": currency,
                "available": str(values[0]),
                "hold": str(values[1]),
                "total": str(sum(values)),
            }
            for currency, values in sorted(totals.items())
            if sum(values) > 0 or currency == "BTC"
        ]
        result.update(
            configured=True,
            verified_at=stamp,
            age_seconds=age,
            balances=balances,
            status="fresh" if age <= 300 else "stale",
            fresh=age <= 300,
        )
        try:
            receipt = _read_json(state_dir / "status.json")
            if (
                receipt.get("overall_status") != "ready"
                or receipt.get("verified_at") != stamp
            ):
                result.update(status="refresh_unverified", fresh=False)
        except CoinbaseAccountError:
            result.update(status="refresh_unverified", fresh=False)
    except (CoinbaseAccountError, KeyError, TypeError, ValueError, InvalidOperation):
        pass
    return result


def schwab_context(root: Path, now: datetime) -> dict:
    from core.schwab_crypto_data import INSTRUMENTS, MAX_QUOTE_AGE, number

    payload, meta = evidence(root, "crypto_market_context_sync_latest.json", now)
    sources = payload.get("sources")
    source = sources.get("schwab_crypto", {}) if isinstance(sources, dict) else {}
    if not isinstance(source, dict):
        source = {}
    supported = (
        source.get("schema_version") == 1 and source.get("data_role") == "context_only"
    )
    source_age = age_seconds(source.get("timestamp_utc"), now)
    source_fresh = meta["fresh"] and source_age is not None and source_age <= 300
    rows, seen = [], set()
    raw_rows = source.get("instruments")
    allowed = set(INSTRUMENTS)
    for raw in (raw_rows if supported and isinstance(raw_rows, list) else [])[:8]:
        if not isinstance(raw, dict):
            continue
        identity = (
            raw.get("asset"),
            raw.get("requested_symbol"),
            raw.get("instrument_type"),
        )
        if (
            any(not isinstance(x, str) for x in identity)
            or identity not in allowed
            or identity in seen
        ):
            continue
        seen.add(identity)
        quote_age = age_seconds(raw.get("quote_timestamp_utc"), now)
        book_ages = [
            age_seconds(raw.get(key), now)
            for key in ("bid_timestamp_utc", "ask_timestamp_utc")
        ]
        usable = (
            source_fresh
            and raw.get("usable") is True
            and raw.get("realtime") is True
            and quote_age is not None
            and quote_age <= MAX_QUOTE_AGE
            and all(value is not None and value <= MAX_QUOTE_AGE for value in book_ages)
        )
        symbol = raw.get("source_symbol")
        quality = safe_codes([raw.get("quality")])
        rows.append(
            {
                "asset": identity[0],
                "requested_symbol": identity[1],
                "instrument_type": identity[2],
                "source_symbol": (
                    symbol
                    if isinstance(symbol, str)
                    and re.fullmatch(r"/?[A-Z0-9]{1,20}", symbol)
                    else None
                ),
                "mark": number(raw.get("mark")),
                "spread_bps": number(raw.get("spread_bps")),
                "quote_timestamp_utc": (
                    raw.get("quote_timestamp_utc") if quote_age is not None else None
                ),
                "age_seconds": quote_age,
                "usable": usable,
                "quality": (
                    "fresh"
                    if usable
                    else (
                        "stale_evidence"
                        if not source_fresh
                        else (
                            quality[0]
                            if quality and quality[0] != "fresh"
                            else "stale_quote"
                        )
                    )
                ),
            }
        )
    usable_count = sum(row["usable"] for row in rows)
    states = safe_codes([source.get("state")])
    state = "not_collected" if not source else states[0] if states else "unavailable"
    if rows:
        state = (
            "ready"
            if usable_count == len(rows)
            else "partial" if usable_count else "stale_or_unusable"
        )
    elif not meta["fresh"] and source:
        state = "stale_evidence"
    return {
        "state": state,
        "evidence": meta,
        "usable_instruments": usable_count,
        "instruments": rows,
        "warnings": safe_codes(source.get("warnings")),
        "data_role": "context_only",
        "spot_feed_verified": False,
        "live_execution_allowed": False,
        "transfers_allowed": False,
    }


def build_snapshot(root: Path, state_dir=None, now=None) -> dict:
    now = now or datetime.now(timezone.utc)
    breaker, breaker_meta = evidence(root, "execution_runtime_breaker_latest.json", now)
    launcher, launcher_meta = evidence(root, "all_sleeves_launcher_latest.json", now)
    training, training_meta = evidence(
        root, "training_runtime_control_latest.json", now
    )
    labels, labels_meta = evidence(
        root, "training_dataset_preflight_latest.json", now, 86400
    )
    ingress, ingress_meta = evidence(
        root, "data_ingress_latest_default_crypto_coinbase.json", now
    )
    blocks = []
    if not breaker_meta["fresh"]:
        blocks.append("execution_breaker_evidence_stale_or_missing")
    if breaker.get("active") is not False:
        blocks.extend(
            safe_codes(breaker.get("reasons")) or ["execution_breaker_not_clear"]
        )
    if not launcher_meta["fresh"] or launcher.get("paper_execution_ready") is not True:
        blocks.append("paper_executor_not_ready")
    # Historical research is never admission into an execution cohort.
    blocks.append("btc_day_and_swing_forward_cohorts_not_admitted")
    label_rows = []
    results = labels.get("results")
    for row in (results if isinstance(results, list) else [])[:100]:
        if not isinstance(row, dict) or "crypto" not in str(row.get("bot_id", "")):
            continue
        label_rows.append(
            {
                "bot_id": str(row.get("bot_id", ""))[:180],
                "data_checks_passed": row.get("data_checks_passed") is True,
                "observations": count(row.get("observation_count")),
                "blockers": safe_codes(row.get("blockers")),
            }
        )
    return {
        "schema_version": 1,
        "timestamp_utc": now.isoformat(),
        "symbol": "BTC-USD",
        "capabilities": {
            "account_read_only": True,
            "historical_research": True,
            "forward_paper_execution": False,
            "live_execution": False,
            "transfers": False,
        },
        "portfolio": portfolio(state_dir, now),
        "schwab_context": schwab_context(root, now),
        "paper": {
            "status": "held",
            "blockers": list(dict.fromkeys(blocks)),
            "breaker": breaker_meta,
            "launcher": launcher_meta,
        },
        "training": {
            "launch_allowed": training_meta["fresh"]
            and training.get("launch_allowed") is True,
            "evidence": training_meta,
            "blockers": safe_codes(training.get("launch_blockers")),
            "label_evidence": labels_meta,
            "assessed": count(labels.get("audited_bot_count")),
            "passed": count(labels.get("data_checks_passed_count")),
            "samples": count(labels.get("materialized_sample_count")),
            "bots": label_rows,
        },
        "collection": {
            "evidence": ingress_meta,
            "symbols": count(ingress.get("symbols_total")),
            "running": ingress_meta["fresh"] and ingress.get("loop_state") == "running",
        },
    }


def run_replay(root: Path, raw: dict) -> tuple[int, dict]:
    from scripts.ops.crypto_research import validate_request

    try:
        request = validate_request(raw)
    except (ValueError, TypeError):
        return 400, {"error": "invalid_replay_request"}
    if not REPLAY_LOCK.acquire(blocking=False):
        return 409, {"error": "research_replay_already_running"}
    try:
        profile = request["profile"]
        cached = REPLAYS.get(profile)
        if (
            cached
            and cached["parameters"] == request
            and time.monotonic() - LAST_ATTEMPT.get(profile, 0) < 300
        ):
            return 200, cached
        if time.monotonic() - LAST_ATTEMPT.get(profile, -1000) < 60:
            return 429, {"error": "replay_cooldown_60_seconds"}
        LAST_ATTEMPT[profile] = time.monotonic()
        env = {
            key: os.environ[key]
            for key in ("PATH", "HOME", "LANG", "TMPDIR")
            if key in os.environ
        }
        env.update(
            PYTHONDONTWRITEBYTECODE="1",
            MPLBACKEND="Agg",
            COINBASE_WEBSOCKET_ENABLED="0",
            COINBASE_HTTP_RETRIES="0",
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            VECLIB_MAXIMUM_THREADS="1",
        )
        result = subprocess.run(
            [sys.executable, "-B", str(RESEARCH), str(root)],
            input=json.dumps(request),
            text=True,
            capture_output=True,
            timeout=25,
            env=env,
            cwd=root,
        )
        if result.returncode or len(result.stdout) > 512 * 1024:
            return 502, {"error": "research_replay_failed"}
        payload = json.loads(result.stdout)
        if not isinstance(payload, dict) or payload.get("error"):
            return 502, {"error": "public_candles_unavailable_or_failed_quality_checks"}
        if (
            payload.get("kind") != "historical_research_only"
            or payload.get("live_execution_allowed") is not False
        ):
            return 502, {"error": "invalid_replay_result"}
        REPLAYS[profile] = payload
        return 200, payload
    except subprocess.TimeoutExpired:
        return 504, {"error": "research_replay_deadline_exceeded"}
    except (OSError, ValueError):
        return 502, {"error": "research_replay_unavailable"}
    finally:
        REPLAY_LOCK.release()


def local_request(handler, *, mutation=False) -> bool:
    try:
        if not ipaddress.ip_address(handler.client_address[0]).is_loopback:
            return False
    except ValueError:
        return False
    port = handler.server.server_address[1]
    host = handler.headers.get("Host", "")
    allowed = {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}
    if host not in allowed or handler.headers.get("Sec-Fetch-Site") == "cross-site":
        return False
    origin = handler.headers.get("Origin")
    if origin is not None and origin != f"http://{host}":
        return False
    if mutation and (
        origin != f"http://{host}"
        or handler.headers.get("Content-Type", "").split(";")[0] != "application/json"
    ):
        return False
    return True


def respond(
    handler, status: int, body: bytes, mime: str, filename: str | None = None
) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", mime)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    if filename:
        handler.send_header("Content-Disposition", f'attachment; filename="{filename}"')
    handler.send_header("X-Content-Type-Options", "nosniff")
    handler.send_header("Referrer-Policy", "no-referrer")
    handler.send_header(
        "Content-Security-Policy",
        "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
    )
    handler.end_headers()
    handler.wfile.write(body)


def handle_request(handler, root: Path, *, post=False) -> bool:
    path = urlparse(handler.path).path
    if path != "/crypto" and not path.startswith(("/crypto/", "/api/crypto/")):
        return False

    def reply(status, payload):
        respond(
            handler,
            status,
            json.dumps(payload, allow_nan=False).encode(),
            "application/json",
        )

    if not local_request(handler, mutation=post):
        reply(403, {"error": "crypto_workspace_local_only"})
        return True
    # The HTML shell contains no private data and shares the feed's token input.
    public_assets = {
        "/crypto": ("index.html", "text/html"),
        "/crypto/": ("index.html", "text/html"),
        "/crypto/app.js": ("app.js", "application/javascript"),
        "/crypto/style.css": ("style.css", "text/css"),
        "/crypto/icons.svg": ("icons.svg", "image/svg+xml"),
    }
    if not post and path in public_assets:
        filename, mime = public_assets[path]
        respond(handler, 200, (ASSETS / filename).read_bytes(), mime)
        return True
    if not handler._require_auth():
        return True
    if post:
        if path != "/api/crypto/replay":
            reply(405, {"error": "operation_not_available"})
            return True
        try:
            length = int(handler.headers.get("Content-Length", "0"))
            if not 0 < length <= 2048 or handler.headers.get("Transfer-Encoding"):
                raise ValueError()
            handler.connection.settimeout(5)
            payload = json.loads(handler.rfile.read(length))
            code, payload = run_replay(root, payload)
            reply(code, payload)
        except (ValueError, TimeoutError, OSError):
            reply(400, {"error": "invalid_replay_request"})
        return True
    if path == "/api/crypto/status":
        reply(200, build_snapshot(root))
    elif path in {"/api/crypto/export/day", "/api/crypto/export/swing"}:
        profile = path.rsplit("/", 1)[-1]
        with REPLAY_LOCK:
            payload = REPLAYS.get(profile)
        if payload is None:
            reply(404, {"error": "research_result_unavailable"})
        else:
            respond(
                handler,
                200,
                json.dumps(payload, allow_nan=False, indent=2).encode(),
                "application/json",
                filename=f"btc-{profile}-historical-research.json",
            )
    elif path == "/api/crypto/replays":
        with REPLAY_LOCK:
            payload = dict(REPLAYS)
        reply(200, payload)
    else:
        reply(404, {"error": "not_found"})
    return True
