from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.base_trader import BaseTrader
from core.brokers import BrokerCredentials


INVALID_SCHWAB_CREDENTIAL_VALUES = {
    "",
    "YOUR_KEY_HERE",
    "YOUR_SECRET_HERE",
    "YOUR_REAL_KEY",
    "YOUR_REAL_SECRET",
    "<real_key>",
    "<real_secret>",
}


def schwab_credentials_from_env() -> BrokerCredentials:
    return BrokerCredentials(
        api_key=os.getenv("SCHWAB_API_KEY", "YOUR_KEY_HERE").strip(),
        app_secret=os.getenv("SCHWAB_SECRET", "YOUR_SECRET_HERE").strip(),
        callback_url=(
            os.getenv("SCHWAB_CALLBACK_URL", "").strip()
            or os.getenv("SCHWAB_REDIRECT", "https://127.0.0.1:8182").strip()
        ),
    )


def credentials_ready(credentials: BrokerCredentials) -> bool:
    return (
        str(credentials.api_key or "").strip() not in INVALID_SCHWAB_CREDENTIAL_VALUES
        and str(credentials.app_secret or "").strip() not in INVALID_SCHWAB_CREDENTIAL_VALUES
    )


def build_schwab_trader(
    project_root: Path,
    *,
    mode: str = "shadow",
    token_path: Optional[Path] = None,
    require_credentials: bool = True,
    missing_credentials_message: str = "Schwab credentials are required",
) -> BaseTrader:
    credentials = schwab_credentials_from_env()
    if require_credentials and not credentials_ready(credentials):
        raise RuntimeError(str(missing_credentials_message or "Schwab credentials are required"))

    trader = BaseTrader.from_env(mode=mode, broker="schwab")
    trader.token_path = str((token_path or (project_root / "token.json")).expanduser().resolve())
    return trader


def resp_json(resp: Any) -> Any:
    if resp is None:
        return None
    if hasattr(resp, "json"):
        try:
            return resp.json()
        except Exception:
            return None
    return resp


def fetch_account_rows(client: Any) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not hasattr(client, "get_account_numbers"):
        return rows

    data = resp_json(client.get_account_numbers())
    if not isinstance(data, list):
        return rows

    for row in data:
        if not isinstance(row, dict):
            continue
        account_reference = str(row.get("hashValue") or row.get("account_hash") or "").strip()
        if not account_reference:
            continue
        rows.append(
            {
                "account_number": str(row.get("accountNumber") or row.get("account_number") or "").strip(),
                "account_hash": account_reference,
            }
        )
    return rows


def fetch_transactions_for_account(
    client: Any,
    account_hash: str,
    start_dt: datetime,
    end_dt: datetime,
) -> List[Dict[str, Any]]:
    if not hasattr(client, "get_transactions"):
        return []

    window_days = 59
    cursor = start_dt
    rows: List[Dict[str, Any]] = []
    seen = set()

    while cursor <= end_dt:
        window_end = min(cursor + timedelta(days=window_days), end_dt)
        attempts = [
            {
                "account_hash": account_hash,
                "start_date": cursor,
                "end_date": window_end,
            },
            {
                "account_hash": account_hash,
                "startDate": cursor.isoformat(),
                "endDate": window_end.isoformat(),
            },
            {
                "account_hash": account_hash,
                "start_datetime": cursor,
                "end_datetime": window_end,
            },
        ]

        payload = None
        for kwargs in attempts:
            try:
                obj = resp_json(client.get_transactions(**kwargs))
                if isinstance(obj, list):
                    payload = obj
                    break
            except Exception:
                continue

        if isinstance(payload, list):
            for tx in payload:
                if not isinstance(tx, dict):
                    continue
                tx_id = str(tx.get("transactionId") or tx.get("activityId") or "")
                tx_ts = str(tx.get("transactionDate") or tx.get("tradeDate") or tx.get("settlementDate") or "")
                tx_type = str(tx.get("type") or tx.get("transactionSubType") or "")
                desc = str(tx.get("description") or "")
                key = (tx_id, tx_ts, tx_type, desc)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(dict(tx))

        cursor = window_end + timedelta(seconds=1)

    return rows


def token_status(path: Path) -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "token_path": str(path),
        "exists": path.exists(),
        "size_bytes": 0,
        "age_seconds": None,
        "expires_at": "",
        "expires_in_seconds": None,
    }
    if not path.exists():
        return status

    try:
        stat = path.stat()
        status["size_bytes"] = int(stat.st_size)
        status["age_seconds"] = max(datetime.now(timezone.utc).timestamp() - float(stat.st_mtime), 0.0)
    except Exception:
        return status

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return status

    if isinstance(payload, dict):
        expiry_sources = [payload]
        nested = payload.get("token")
        if isinstance(nested, dict):
            expiry_sources.insert(0, nested)

        exp_value: Any = ""
        for source in expiry_sources:
            for key in ("expires_at", "expiresAt", "expires", "expires_time"):
                raw = source.get(key)
                if raw not in (None, ""):
                    exp_value = raw
                    break
            if exp_value not in (None, ""):
                break

        if exp_value not in (None, ""):
            status["expires_at"] = str(exp_value)
            try:
                if isinstance(exp_value, (int, float)):
                    expires_epoch = float(exp_value)
                else:
                    normalized = str(exp_value).strip().replace("Z", "+00:00")
                    if normalized.replace(".", "", 1).isdigit():
                        expires_epoch = float(normalized)
                    else:
                        dt = datetime.fromisoformat(normalized)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        expires_epoch = dt.astimezone(timezone.utc).timestamp()
                status["expires_in_seconds"] = expires_epoch - datetime.now(timezone.utc).timestamp()
            except Exception:
                status["expires_in_seconds"] = None

    return status


def token_needs_refresh(
    status: Dict[str, Any],
    *,
    min_expires_seconds: float,
    max_age_seconds: Optional[float] = None,
    ready_reason: str = "token_ready",
) -> tuple[bool, str]:
    if not bool(status.get("exists")):
        return True, "missing_token"
    if int(status.get("size_bytes") or 0) < 64:
        return True, "token_too_small"

    if max_age_seconds is not None:
        age = status.get("age_seconds")
        if age is not None and float(age) > max(float(max_age_seconds), 0.0):
            return True, f"token_age_high:{float(age):.1f}"

    expires_in = status.get("expires_in_seconds")
    if expires_in is not None and float(expires_in) <= max(float(min_expires_seconds), 0.0):
        return True, f"token_expiring_soon:{float(expires_in):.1f}"

    return False, str(ready_reason or "token_ready")
