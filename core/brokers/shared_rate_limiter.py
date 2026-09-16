from __future__ import annotations

import json
import os
import sqlite3
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _enabled(value: object, *, default: bool = False) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return default
    return text not in {"0", "false", "no", "off"}


def broker_operation_pool(operation: str) -> str:
    key = str(operation or "").strip().lower()
    if any(token in key for token in ("order", "cancel", "replace")):
        return "orders"
    if any(
        token in key
        for token in ("account", "position", "transaction", "balance")
    ):
        return "accounts"
    return "market_data"


def _configured_limit(
    project_root: Path,
    broker: str,
    pool: str,
    env: Mapping[str, str],
) -> int:
    env_name = f"BROKER_RATE_LIMIT_{broker}_{pool}_PER_MINUTE".upper()
    raw = str(env.get(env_name, "") or "").strip()
    if raw:
        try:
            return max(int(raw), 1)
        except ValueError:
            pass
    path = project_root / "config" / "broker_capability_contracts_v1.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        configured = (
            payload.get("brokers", {})
            .get(broker, {})
            .get("rate_limit_pools", {})
            .get(pool)
        )
        return max(int(configured), 1)
    except Exception:
        return 60


def acquire_broker_rate_limit(
    *,
    project_root: str | Path,
    broker: str,
    operation: str,
    env: Mapping[str, str] | None = None,
    now_epoch: float | None = None,
    database_path: str | Path | None = None,
) -> dict[str, Any]:
    """Reserve one shared provider request slot using a durable minute window."""
    values = env if env is not None else os.environ
    broker_key = str(broker or "generic").strip().lower() or "generic"
    pool = broker_operation_pool(operation)
    enabled = _enabled(
        values.get("BROKER_SHARED_RATE_LIMIT_ENABLED", "0"),
        default=False,
    )
    if not enabled:
        return {
            "enabled": False,
            "allowed": True,
            "broker": broker_key,
            "pool": pool,
            "operation": str(operation or ""),
        }

    root = Path(project_root)
    path = (
        Path(database_path)
        if database_path is not None
        else root / "governance" / "runtime" / "broker_rate_limits.sqlite3"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    limit = _configured_limit(root, broker_key, pool, values)
    now = float(time.time() if now_epoch is None else now_epoch)
    window_start = int(now // 60.0) * 60
    retry_after = max(float(window_start + 60) - now, 0.0)

    conn = sqlite3.connect(str(path), timeout=10.0)
    try:
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rate_limit_windows (
                broker TEXT NOT NULL,
                pool TEXT NOT NULL,
                window_start_epoch INTEGER NOT NULL,
                request_count INTEGER NOT NULL,
                updated_at_utc TEXT NOT NULL,
                PRIMARY KEY (broker, pool, window_start_epoch)
            )
            """
        )
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "DELETE FROM rate_limit_windows WHERE window_start_epoch < ?",
            (window_start - 600,),
        )
        row = conn.execute(
            """
            SELECT request_count FROM rate_limit_windows
            WHERE broker = ? AND pool = ? AND window_start_epoch = ?
            """,
            (broker_key, pool, window_start),
        ).fetchone()
        used_before = int(row[0]) if row else 0
        allowed = used_before < limit
        used_after = used_before + 1 if allowed else used_before
        if row is None:
            conn.execute(
                """
                INSERT INTO rate_limit_windows (
                    broker, pool, window_start_epoch, request_count, updated_at_utc
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    broker_key,
                    pool,
                    window_start,
                    used_after,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        elif allowed:
            conn.execute(
                """
                UPDATE rate_limit_windows
                SET request_count = ?, updated_at_utc = ?
                WHERE broker = ? AND pool = ? AND window_start_epoch = ?
                """,
                (
                    used_after,
                    datetime.now(timezone.utc).isoformat(),
                    broker_key,
                    pool,
                    window_start,
                ),
            )
        conn.commit()
    finally:
        conn.close()

    return {
        "enabled": True,
        "allowed": allowed,
        "broker": broker_key,
        "pool": pool,
        "operation": str(operation or ""),
        "limit_per_minute": limit,
        "used": used_after,
        "remaining": max(limit - used_after, 0),
        "window_start_epoch": window_start,
        "retry_after_seconds": round(retry_after, 3) if not allowed else 0.0,
    }
