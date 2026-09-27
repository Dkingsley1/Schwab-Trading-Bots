"""Shared, bounded provider captures and small decision-time references.

Collectors publish under one lock. Decision logging reads only a small pointer;
reports validate the hash-bound candles offline. No orders or network calls.
"""

from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import stat
import tempfile

from core.decision_price_evidence import digest, timestamp
from core.schd_capture_store import checked, read_capture

DIRECTORY = Path("governance/market/decision_candles")
MAX_FILES = 512
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_CAPTURE_BYTES = 1024 * 1024
MAX_AGE_SECONDS = 900


def identity(provider, symbol):
    if provider not in {"schwab", "coinbase"} or not re.fullmatch(
        r"[A-Z0-9][A-Z0-9.-]{0,31}", str(symbol)
    ):
        raise ValueError("unsupported_chart_provider_or_instrument")
    if provider == "coinbase" and not re.fullmatch(r"[A-Z0-9]+-USD", symbol):
        raise ValueError("exact_coinbase_usd_product_required")
    return provider + "_" + symbol


def _atomic(root, path, raw, *, immutable=False):
    checked(root, path)
    fd, temporary = tempfile.mkstemp(prefix=".building-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if immutable:
            os.link(temporary, path, follow_symlinks=False)
        else:
            os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def publish(root, capture):
    """Never evict captures referenced by historical decisions to claim space."""
    source = capture["source"]
    key = identity(source["provider"], source["symbol"])
    observed = timestamp(source.get("observed_at_utc", source["fetch_started_at_utc"]))
    if (
        not timestamp(source["fetch_started_at_utc"])
        <= observed
        <= datetime.now(timezone.utc)
    ):
        raise ValueError("invalid_capture_observation_time")
    sha = digest(capture)
    raw = json.dumps(capture, sort_keys=True, allow_nan=False).encode()
    if len(raw) > MAX_CAPTURE_BYTES:
        raise ValueError("decision_capture_size_exceeded")
    directory = checked(root, Path(root) / DIRECTORY)
    directory.mkdir(parents=True, exist_ok=True)
    lock = checked(root, directory / "writer.lock")
    fd = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    with os.fdopen(fd, "a+") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError("unsafe_decision_capture_lock")
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        count = total = 0
        for entry in os.scandir(directory):
            info = checked(root, directory / entry.name).lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError("unsafe_decision_capture_entry")
            count += 1
            total += info.st_size
            if count > MAX_FILES or total > MAX_TOTAL_BYTES:
                raise ValueError("decision_capture_capacity_exhausted")
        target = checked(root, directory / (sha + ".json"))
        pointer = checked(root, directory / (key + ".latest.json"))
        if pointer.exists():
            previous = read_capture(root, pointer)
            if timestamp(previous["observed_at_utc"]) > observed:
                raise ValueError("decision_capture_out_of_order_publication")
        receipt = dict(
            state="observed_context_not_claimed_model_input",
            provider=source["provider"],
            symbol=source["symbol"],
            capture_sha256=sha,
            fetch_started_at_utc=source["fetch_started_at_utc"],
            observed_at_utc=observed.isoformat(),
        )
        pointer_raw = json.dumps(receipt, sort_keys=True).encode()
        additional = int(not target.exists()) + int(not pointer.exists())
        if (
            count + additional > MAX_FILES
            or total + len(raw) + len(pointer_raw) > MAX_TOTAL_BYTES
        ):
            raise ValueError("decision_capture_capacity_exhausted")
        if target.exists():
            if digest(read_capture(root, target)) != sha:
                raise ValueError("decision_capture_hash_mismatch")
        else:
            _atomic(root, target, raw, immutable=True)
        _atomic(root, pointer, pointer_raw)
        return receipt


def context_receipt(root, symbol, *, provider=None, now=None):
    now = now or datetime.now(timezone.utc)
    # A hyphenated USD product denotes Coinbase spot, never an equity proxy.
    provider = provider or ("coinbase" if str(symbol).endswith("-USD") else "schwab")
    try:
        key = identity(provider, symbol)
        path = checked(root, Path(root) / DIRECTORY / (key + ".latest.json"))
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise ValueError("unsafe_decision_capture_pointer")
            raw = handle.read(2049)
        if len(raw) > 2048:
            raise ValueError("decision_capture_pointer_oversize")
        receipt = json.loads(raw)
        if receipt.get("provider") != provider or receipt.get("symbol") != symbol:
            raise ValueError("decision_capture_pointer_identity_mismatch")
        if (
            not 0
            <= (now - timestamp(receipt["observed_at_utc"])).total_seconds()
            <= MAX_AGE_SECONDS
        ):
            raise ValueError("decision_capture_stale_or_future")
        return receipt
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return {
            "state": "unavailable",
            "provider": provider,
            "symbol": symbol,
            "reason": (
                str(exc)
                if isinstance(exc, ValueError)
                else "decision_capture_not_available"
            ),
        }


def read_bound(root, decision):
    receipt = decision.get("metadata", {}).get("candle_context", {})
    sha = receipt.get("capture_sha256", "")
    if not isinstance(sha, str) or not re.fullmatch(r"[a-f0-9]{64}", sha):
        raise ValueError("original_capture_binding_missing")
    capture = read_capture(root, Path(root) / DIRECTORY / (sha + ".json"))
    source = capture["source"]
    identity(source["provider"], source["symbol"])
    when = timestamp(decision["timestamp_utc"])
    observed = timestamp(source.get("observed_at_utc", source["fetch_started_at_utc"]))
    if (
        digest(capture) != sha
        or source["symbol"] != decision["symbol"]
        or source["provider"] != receipt.get("provider")
        or receipt.get("symbol") != decision["symbol"]
        or receipt.get("state") != "observed_context_not_claimed_model_input"
        or timestamp(receipt["observed_at_utc"]) != observed
        or timestamp(receipt["fetch_started_at_utc"])
        != timestamp(source["fetch_started_at_utc"])
        or not timestamp(source["fetch_started_at_utc"]) <= observed <= when
        or not 0 <= (when - observed).total_seconds() <= MAX_AGE_SECONDS
    ):
        raise ValueError("original_capture_identity_or_time_invalid")
    return capture
