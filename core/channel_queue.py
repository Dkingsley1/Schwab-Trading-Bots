from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

DEFAULT_QUEUE_DB_NAME = "bot_channel_queue.sqlite3"
DEFAULT_LOCAL_FALLBACK_ROOT = "local_fallback_storage"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_ts_utc(raw: str) -> Optional[datetime]:
    text = str(raw or '').strip()
    if not text:
        return None
    if text.endswith('Z'):
        text = text[:-1] + '+00:00'
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _decode_details(raw: Any) -> Dict[str, Any]:
    try:
        value = json.loads(str(raw or "{}"))
    except Exception:
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _local_queue_root(project_root: str | Path) -> Path:
    configured = str(os.getenv("BOT_CHANNEL_QUEUE_LOCAL_ROOT", "") or "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path(project_root) / DEFAULT_LOCAL_FALLBACK_ROOT


def local_queue_db_path(project_root: str | Path) -> str:
    root = Path(project_root) if str(project_root or "").strip() else Path()
    if not str(root):
        return ""
    return str(_local_queue_root(root) / "data" / DEFAULT_QUEUE_DB_NAME)


def routed_queue_db_path(project_root: str | Path) -> str:
    root = Path(project_root) if str(project_root or "").strip() else Path()
    if not str(root):
        return ""
    return str(root / "data" / DEFAULT_QUEUE_DB_NAME)


def _queue_prefer_local_override() -> Optional[bool]:
    raw = str(os.getenv("BOT_CHANNEL_QUEUE_PREFER_LOCAL", "") or "").strip()
    if not raw:
        return None
    return raw.lower() in {"1", "true", "yes", "on"}


@dataclass
class ChannelMessage:
    id: int
    channel: str
    message_id: str
    parent_message_id: str
    run_id: str
    iter_id: str
    source_path: str
    payload: Dict[str, Any]
    created_at: str


class ChannelQueue:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.last_repair: Dict[str, Any] = {"active": False, "moved": [], "reason": ""}
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._schema_ready():
            self._ensure_schema()

    def _wal_retry_count(self) -> int:
        try:
            return max(int(float(os.getenv("BOT_CHANNEL_QUEUE_WAL_RETRY_COUNT", "6") or 6)), 1)
        except Exception:
            return 6

    def _wal_retry_sleep_seconds(self) -> float:
        try:
            return max(float(os.getenv("BOT_CHANNEL_QUEUE_WAL_RETRY_SLEEP_SECONDS", "0.25") or 0.25), 0.05)
        except Exception:
            return 0.25

    def _schema_ready(self) -> bool:
        if not self.db_path.exists() or self.db_path.stat().st_size <= 0:
            return False
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=5.0)
            try:
                conn.execute("PRAGMA busy_timeout=5000")
                rows = conn.execute(
                    """
                    SELECT name
                    FROM sqlite_master
                    WHERE type='table' AND name IN (
                        'channel_messages',
                        'channel_consumer_state',
                        'channel_processing_claims'
                    )
                    """
                ).fetchall()
            finally:
                conn.close()
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc).lower():
                return True
            return False
        except sqlite3.DatabaseError as exc:
            self._quarantine_corrupt_db(str(exc))
            return False
        return {str(row[0] or "") for row in rows} >= {
            "channel_messages",
            "channel_consumer_state",
            "channel_processing_claims",
        }

    def _quarantine_corrupt_db(self, reason: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        moved: List[Dict[str, Any]] = []

        candidates: list[tuple[Path, str, bool]] = []
        if self.db_path.is_symlink():
            candidates.append((self.db_path.resolve(strict=False), str(self.db_path), True))
        else:
            candidates.append((self.db_path, str(self.db_path), False))
        for suffix in ("-wal", "-shm"):
            sidecar = Path(f"{self.db_path}{suffix}")
            if sidecar.exists() or sidecar.is_symlink():
                candidates.append((sidecar.resolve(strict=False) if sidecar.is_symlink() else sidecar, str(sidecar), sidecar.is_symlink()))

        for path, original_path, via_symlink in candidates:
            try:
                if not path.exists() and not path.is_symlink():
                    continue
                target = path.with_name(f"{path.name}.corrupt-{stamp}")
                path.rename(target)
                moved.append(
                    {
                        "original_path": original_path,
                        "moved_path": str(target),
                        "via_symlink": bool(via_symlink),
                    }
                )
            except Exception as exc:
                moved.append(
                    {
                        "original_path": original_path,
                        "moved_path": "",
                        "via_symlink": bool(via_symlink),
                        "error": str(exc),
                    }
                )

        self.last_repair = {
            "active": bool(moved),
            "reason": str(reason or "sqlite_database_error"),
            "moved": moved,
        }

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.execute("PRAGMA busy_timeout=30000")
        last_locked_error: sqlite3.OperationalError | None = None
        for attempt in range(self._wal_retry_count()):
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                last_locked_error = None
                break
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower():
                    conn.close()
                    raise
                last_locked_error = exc
                if attempt >= self._wal_retry_count() - 1:
                    break
                time.sleep(min(self._wal_retry_sleep_seconds() * (attempt + 1), 2.0))
        if last_locked_error is None:
            conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _read_connect(self, *, timeout_seconds: float = 1.0) -> sqlite3.Connection:
        """Open a bounded read path without renegotiating SQLite journal mode."""

        timeout = min(max(float(timeout_seconds), 0.05), 5.0)
        conn = sqlite3.connect(str(self.db_path), timeout=timeout)
        conn.execute(f"PRAGMA busy_timeout={max(int(timeout * 1000), 50)}")
        conn.execute("PRAGMA query_only=ON")
        return conn

    def _ensure_schema(self) -> None:
        conn = self._connect()
        try:
            conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS channel_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel TEXT NOT NULL,
                    message_id TEXT NOT NULL UNIQUE,
                    parent_message_id TEXT,
                    run_id TEXT,
                    iter_id TEXT,
                    source_path TEXT,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS channel_consumer_state (
                    consumer TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    last_id INTEGER NOT NULL DEFAULT 0,
                    last_message_id TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (consumer, channel)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS channel_processing_claims (
                    consumer TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    message_row_id INTEGER NOT NULL DEFAULT 0,
                    state TEXT NOT NULL,
                    claimed_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    finalized_at TEXT NOT NULL DEFAULT '',
                    outcome_status TEXT NOT NULL DEFAULT '',
                    outcome_message_id TEXT NOT NULL DEFAULT '',
                    details_json TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY (consumer, channel, message_id)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_channel_messages_channel_id ON channel_messages(channel, id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_channel_messages_created_at ON channel_messages(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_channel_consumer_state_channel ON channel_consumer_state(channel, last_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_channel_consumer_state_updated_at ON channel_consumer_state(updated_at)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_channel_processing_claims_state_updated "
                "ON channel_processing_claims(consumer, channel, state, updated_at)"
            )
            conn.commit()
        finally:
            conn.close()

    def _normalize_payload_row(
        self,
        *,
        channel: str,
        payload: Dict[str, Any],
        source_path: str = "",
        message_id: str = "",
        parent_message_id: str = "",
        run_id: str = "",
        iter_id: str = "",
    ) -> tuple[str, str, str, str, str, str, str, str]:
        ch = str(channel or '').strip()
        if not ch:
            raise ValueError('channel is required')

        msg_id = str(message_id or payload.get('message_id') or uuid.uuid4())
        parent_id = str(parent_message_id or payload.get('parent_message_id') or payload.get('parent_decision_id') or '')
        run = str(run_id or payload.get('run_id') or '')
        itr = str(iter_id or payload.get('iter_id') or '')
        created_at = str(payload.get('timestamp_utc') or _now_utc())
        payload_json = json.dumps(payload, ensure_ascii=True, separators=(',', ':'))
        return (ch, msg_id, parent_id, run, itr, str(source_path or ''), payload_json, created_at)

    def enqueue(
        self,
        *,
        channel: str,
        payload: Dict[str, Any],
        source_path: str = "",
        message_id: str = "",
        parent_message_id: str = "",
        run_id: str = "",
        iter_id: str = "",
    ) -> str:
        row = self._normalize_payload_row(
            channel=channel,
            payload=payload,
            source_path=source_path,
            message_id=message_id,
            parent_message_id=parent_message_id,
            run_id=run_id,
            iter_id=iter_id,
        )

        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO channel_messages(
                    channel, message_id, parent_message_id, run_id, iter_id, source_path, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row,
            )
            conn.commit()
        finally:
            conn.close()
        return str(row[1])

    def enqueue_batch(
        self,
        *,
        channel: str,
        payloads: Sequence[Dict[str, Any]],
        source_path: str = '',
    ) -> List[str]:
        batch = [
            self._normalize_payload_row(channel=channel, payload=dict(payload or {}), source_path=source_path)
            for payload in payloads
            if isinstance(payload, dict) and payload
        ]
        if not batch:
            return []

        conn = self._connect()
        try:
            conn.executemany(
                """
                INSERT OR IGNORE INTO channel_messages(
                    channel, message_id, parent_message_id, run_id, iter_id, source_path, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                batch,
            )
            conn.commit()
        finally:
            conn.close()
        return [str(row[1]) for row in batch]

    def active_consumer_count(self, *, channel: str = '', max_age_seconds: int = 86400) -> int:
        conn = self._connect()
        try:
            cutoff = (_parse_ts_utc(_now_utc()) or datetime.now(timezone.utc)) - timedelta(seconds=max(int(max_age_seconds), 60))
            cutoff_iso = cutoff.isoformat()
            if channel:
                row = conn.execute(
                    "SELECT COUNT(*) FROM channel_consumer_state WHERE channel=? AND updated_at>=?",
                    (str(channel), cutoff_iso),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) FROM channel_consumer_state WHERE updated_at>=?",
                    (cutoff_iso,),
                ).fetchone()
        finally:
            conn.close()
        return int(row[0] if row and row[0] is not None else 0)

    def has_recent_consumer(self, *, channel: str = '', max_age_seconds: int = 86400) -> bool:
        return self.active_consumer_count(channel=channel, max_age_seconds=max_age_seconds) > 0

    def read_from_cursor(self, *, consumer: str, channel: str, limit: int = 500) -> List[ChannelMessage]:
        cons = str(consumer or "").strip()
        ch = str(channel or "").strip()
        if not cons or not ch:
            return []

        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT last_id FROM channel_consumer_state WHERE consumer=? AND channel=?",
                (cons, ch),
            ).fetchone()
            last_id = int(row[0]) if row else 0

            rows = conn.execute(
                """
                SELECT id, channel, message_id, parent_message_id, run_id, iter_id, source_path, payload_json, created_at
                FROM channel_messages
                WHERE channel=? AND id>?
                ORDER BY id ASC
                LIMIT ?
                """,
                (ch, last_id, max(int(limit), 1)),
            ).fetchall()
        finally:
            conn.close()

        out: List[ChannelMessage] = []
        for r in rows:
            try:
                payload = json.loads(str(r[7]))
            except Exception:
                payload = {}
            out.append(
                ChannelMessage(
                    id=int(r[0]),
                    channel=str(r[1]),
                    message_id=str(r[2]),
                    parent_message_id=str(r[3] or ''),
                    run_id=str(r[4] or ''),
                    iter_id=str(r[5] or ''),
                    source_path=str(r[6] or ''),
                    payload=payload,
                    created_at=str(r[8]),
                )
            )
        return out

    def ack_through(
        self,
        *,
        consumer: str,
        channel: str,
        last_id: int,
        last_message_id: str = "",
    ) -> None:
        cons = str(consumer or "").strip()
        ch = str(channel or "").strip()
        if not cons or not ch:
            return

        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO channel_consumer_state(consumer, channel, last_id, last_message_id, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(consumer, channel)
                DO UPDATE SET
                    last_id=MAX(channel_consumer_state.last_id, excluded.last_id),
                    last_message_id=CASE
                        WHEN excluded.last_id >= channel_consumer_state.last_id
                        THEN excluded.last_message_id
                        ELSE channel_consumer_state.last_message_id
                    END,
                    updated_at=CASE
                        WHEN excluded.last_id >= channel_consumer_state.last_id
                        THEN excluded.updated_at
                        ELSE channel_consumer_state.updated_at
                    END
                """,
                (cons, ch, max(int(last_id), 0), str(last_message_id or ""), _now_utc()),
            )
            conn.commit()
        finally:
            conn.close()

    def ack_messages(self, *, consumer: str, channel: str, messages: List[ChannelMessage]) -> None:
        if not messages:
            return
        last = messages[-1]
        self.ack_through(consumer=consumer, channel=channel, last_id=int(last.id), last_message_id=str(last.message_id))

    def claim_message_processing(
        self,
        *,
        consumer: str,
        channel: str,
        message: ChannelMessage,
    ) -> Dict[str, Any]:
        """Durably claim an intent before any execution side effect occurs."""

        cons = str(consumer or "").strip()
        ch = str(channel or "").strip()
        message_id = str(message.message_id or "").strip()
        if not cons or not ch or not message_id:
            raise ValueError("consumer, channel, and message_id are required")

        now = _now_utc()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                """
                SELECT message_row_id, state, claimed_at, updated_at, finalized_at,
                       outcome_status, outcome_message_id, details_json
                FROM channel_processing_claims
                WHERE consumer=? AND channel=? AND message_id=?
                """,
                (cons, ch, message_id),
            ).fetchone()
            if existing is not None:
                conn.commit()
                return {
                    "claimed": False,
                    "consumer": cons,
                    "channel": ch,
                    "message_id": message_id,
                    "message_row_id": int(existing[0] or 0),
                    "state": str(existing[1] or ""),
                    "claimed_at": str(existing[2] or ""),
                    "updated_at": str(existing[3] or ""),
                    "finalized_at": str(existing[4] or ""),
                    "outcome_status": str(existing[5] or ""),
                    "outcome_message_id": str(existing[6] or ""),
                    "details": _decode_details(existing[7]),
                }
            conn.execute(
                """
                INSERT INTO channel_processing_claims(
                    consumer, channel, message_id, message_row_id, state,
                    claimed_at, updated_at
                ) VALUES (?, ?, ?, ?, 'processing', ?, ?)
                """,
                (cons, ch, message_id, max(int(message.id), 0), now, now),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return {
            "claimed": True,
            "consumer": cons,
            "channel": ch,
            "message_id": message_id,
            "message_row_id": max(int(message.id), 0),
            "state": "processing",
            "claimed_at": now,
            "updated_at": now,
            "finalized_at": "",
            "outcome_status": "",
            "outcome_message_id": "",
            "details": {},
        }

    def finalize_message_processing(
        self,
        *,
        consumer: str,
        channel: str,
        message: ChannelMessage,
        state: str,
        outcome_status: str = "",
        outcome_message_id: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cons = str(consumer or "").strip()
        ch = str(channel or "").strip()
        message_id = str(message.message_id or "").strip()
        state_text = str(state or "").strip().lower()
        allowed_states = {
            "completed",
            "dead_lettered",
            "outcome_ambiguous",
            "replay_suppressed",
        }
        if not cons or not ch or not message_id:
            raise ValueError("consumer, channel, and message_id are required")
        if state_text not in allowed_states:
            raise ValueError(f"unsupported processing claim state: {state_text}")

        now = _now_utc()
        details_json = json.dumps(
            dict(details or {}), ensure_ascii=True, separators=(",", ":")
        )
        conn = self._connect()
        try:
            cursor = conn.execute(
                """
                UPDATE channel_processing_claims
                SET state=?, updated_at=?, finalized_at=?, outcome_status=?,
                    outcome_message_id=?, details_json=?
                WHERE consumer=? AND channel=? AND message_id=?
                """,
                (
                    state_text,
                    now,
                    now,
                    str(outcome_status or ""),
                    str(outcome_message_id or ""),
                    details_json,
                    cons,
                    ch,
                    message_id,
                ),
            )
            if int(cursor.rowcount or 0) != 1:
                raise KeyError(f"processing claim not found: {cons}:{ch}:{message_id}")
            conn.commit()
        finally:
            conn.close()
        return {
            "consumer": cons,
            "channel": ch,
            "message_id": message_id,
            "message_row_id": max(int(message.id), 0),
            "state": state_text,
            "updated_at": now,
            "finalized_at": now,
            "outcome_status": str(outcome_status or ""),
            "outcome_message_id": str(outcome_message_id or ""),
            "details": dict(details or {}),
        }

    def processing_claim_stats(self, *, consumer: str, channel: str) -> Dict[str, Any]:
        cons = str(consumer or "").strip()
        ch = str(channel or "").strip()
        if not cons or not ch:
            return {"consumer": cons, "channel": ch, "total": 0, "state_counts": {}}
        try:
            timeout_seconds = float(
                os.getenv("BOT_CHANNEL_QUEUE_HEALTH_READ_TIMEOUT_SECONDS", "1") or 1
            )
        except Exception:
            timeout_seconds = 1.0
        conn = self._read_connect(timeout_seconds=timeout_seconds)
        try:
            rows = conn.execute(
                """
                SELECT state, COUNT(*)
                FROM channel_processing_claims
                WHERE consumer=? AND channel=?
                GROUP BY state
                """,
                (cons, ch),
            ).fetchall()
            latest = conn.execute(
                """
                SELECT message_id, message_row_id, state, updated_at, outcome_status
                FROM channel_processing_claims
                WHERE consumer=? AND channel=?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (cons, ch),
            ).fetchone()
        finally:
            conn.close()
        state_counts = {str(row[0] or "unknown"): int(row[1] or 0) for row in rows}
        return {
            "consumer": cons,
            "channel": ch,
            "total": sum(state_counts.values()),
            "state_counts": state_counts,
            "processing": int(state_counts.get("processing", 0)),
            "ambiguous": int(state_counts.get("outcome_ambiguous", 0)),
            "latest": (
                {
                    "message_id": str(latest[0] or ""),
                    "message_row_id": int(latest[1] or 0),
                    "state": str(latest[2] or ""),
                    "updated_at": str(latest[3] or ""),
                    "outcome_status": str(latest[4] or ""),
                }
                if latest
                else {}
            ),
        }

    def consumer_state(self, *, consumer: str, channel: str) -> Dict[str, Any]:
        cons = str(consumer or "").strip()
        ch = str(channel or "").strip()
        if not cons or not ch:
            return {"consumer": cons, "channel": ch, "last_id": 0, "last_message_id": "", "updated_at": ""}

        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT last_id, last_message_id, updated_at FROM channel_consumer_state WHERE consumer=? AND channel=?",
                (cons, ch),
            ).fetchone()
        finally:
            conn.close()

        if not row:
            return {"consumer": cons, "channel": ch, "last_id": 0, "last_message_id": "", "updated_at": ""}

        return {
            "consumer": cons,
            "channel": ch,
            "last_id": int(row[0] or 0),
            "last_message_id": str(row[1] or ""),
            "updated_at": str(row[2] or ""),
        }

    def queue_stats(self, *, channel: str = "") -> Dict[str, Any]:
        conn = self._connect()
        try:
            if channel:
                row = conn.execute(
                    "SELECT COUNT(*), MIN(created_at), MAX(created_at), MAX(id) FROM channel_messages WHERE channel=?",
                    (channel,),
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*), MIN(created_at), MAX(created_at), MAX(id) FROM channel_messages").fetchone()
        finally:
            conn.close()

        return {
            "channel": str(channel or ""),
            "rows": int(row[0] if row and row[0] is not None else 0),
            "oldest_created_at": str(row[1] or "") if row else "",
            "newest_created_at": str(row[2] or "") if row else "",
            "max_id": int(row[3] if row and row[3] is not None else 0),
            "db_path": str(self.db_path),
        }

    def pending_count(self, *, consumer: str, channel: str) -> int:
        cons = str(consumer or "").strip()
        ch = str(channel or "").strip()
        if not cons or not ch:
            return 0

        state = self.consumer_state(consumer=cons, channel=ch)
        last_id = int(state.get("last_id") or 0)

        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM channel_messages WHERE channel=? AND id>?",
                (ch, last_id),
            ).fetchone()
        finally:
            conn.close()
        return int(row[0] if row and row[0] is not None else 0)

    def stale_prefix(
        self,
        *,
        consumer: str,
        channel: str,
        stale_before: datetime,
        limit: int = 5000,
    ) -> Dict[str, Any]:
        cons = str(consumer or "").strip()
        ch = str(channel or "").strip()
        if not cons or not ch:
            return {"count": 0, "last_id": 0, "last_message_id": "", "oldest_created_at": "", "newest_created_at": ""}

        state = self.consumer_state(consumer=cons, channel=ch)
        last_seen_id = int(state.get("last_id") or 0)
        max_rows = max(int(limit), 1)

        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT id, message_id, created_at
                FROM channel_messages
                WHERE channel=? AND id>?
                ORDER BY id
                LIMIT ?
                """,
                (ch, last_seen_id, max_rows),
            ).fetchall()
        finally:
            conn.close()

        stale_rows: list[tuple[int, str, str]] = []
        stopped_at_fresh = False
        for row in rows:
            created_at = str(row[2] or "")
            parsed = _parse_ts_utc(created_at)
            if parsed is None or parsed >= stale_before:
                stopped_at_fresh = True
                break
            stale_rows.append((int(row[0] or 0), str(row[1] or ""), created_at))

        if not stale_rows:
            return {
                "count": 0,
                "last_id": 0,
                "last_message_id": "",
                "oldest_created_at": "",
                "newest_created_at": "",
                "scanned_rows": len(rows),
                "stopped_at_fresh": bool(stopped_at_fresh),
            }

        return {
            "count": len(stale_rows),
            "last_id": stale_rows[-1][0],
            "last_message_id": stale_rows[-1][1],
            "first_id": stale_rows[0][0],
            "first_message_id": stale_rows[0][1],
            "oldest_created_at": min(row[2] for row in stale_rows),
            "newest_created_at": max(row[2] for row in stale_rows),
            "scanned_rows": len(rows),
            "stopped_at_fresh": bool(stopped_at_fresh),
        }


def default_queue_db_path(project_root: str | Path) -> str:
    override = str(os.getenv("BOT_CHANNEL_QUEUE_DB", "") or "").strip()
    if override:
        return str(Path(override).expanduser())
    prefer_local = _queue_prefer_local_override()
    if prefer_local is True:
        return local_queue_db_path(project_root)
    if prefer_local is False:
        return routed_queue_db_path(project_root)
    if _env_flag("BOT_LOGS_PREFER_EXTERNAL", "1"):
        return routed_queue_db_path(project_root)
    return local_queue_db_path(project_root)


def queue_enabled() -> bool:
    return os.getenv("BOT_CHANNEL_QUEUE_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
