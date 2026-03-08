from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _ensure_schema(self) -> None:
        conn = self._connect()
        try:
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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_channel_messages_channel_id ON channel_messages(channel, id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_channel_messages_created_at ON channel_messages(created_at)")
            conn.commit()
        finally:
            conn.close()

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
        ch = str(channel or "").strip()
        if not ch:
            raise ValueError("channel is required")

        msg_id = str(message_id or payload.get("message_id") or uuid.uuid4())
        parent_id = str(parent_message_id or payload.get("parent_message_id") or payload.get("parent_decision_id") or "")
        run = str(run_id or payload.get("run_id") or "")
        itr = str(iter_id or payload.get("iter_id") or "")
        created_at = str(payload.get("timestamp_utc") or _now_utc())
        payload_json = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))

        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO channel_messages(
                    channel, message_id, parent_message_id, run_id, iter_id, source_path, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (ch, msg_id, parent_id, run, itr, str(source_path or ""), payload_json, created_at),
            )
            conn.commit()
        finally:
            conn.close()
        return msg_id

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
                    parent_message_id=str(r[3] or ""),
                    run_id=str(r[4] or ""),
                    iter_id=str(r[5] or ""),
                    source_path=str(r[6] or ""),
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
                    last_id=excluded.last_id,
                    last_message_id=excluded.last_message_id,
                    updated_at=excluded.updated_at
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


def default_queue_db_path(project_root: str | Path) -> str:
    return str(Path(project_root) / "data" / "bot_channel_queue.sqlite3")


def queue_enabled() -> bool:
    return os.getenv("BOT_CHANNEL_QUEUE_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
