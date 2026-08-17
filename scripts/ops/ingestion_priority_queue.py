#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "governance" / "queues" / "ingestion_priority_queue.sqlite3"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "ingestion_priority_queue_latest.json"

LANE_SPECS = {
    "core": {"quota": 0.60, "boost": 1.00, "source_key": "top_pending_files"},
    "deferred": {"quota": 0.25, "boost": 0.65, "source_key": "top_deferred_pending_files"},
    "cold": {"quota": 0.15, "boost": 0.35, "source_key": "top_cold_pending_files"},
}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS queue_items (
            source_rel TEXT PRIMARY KEY,
            lane TEXT NOT NULL,
            dedupe_key TEXT NOT NULL,
            quota_share REAL NOT NULL,
            priority_score REAL NOT NULL,
            pending_lines INTEGER NOT NULL,
            total_lines INTEGER NOT NULL,
            oldest_pending_age_seconds REAL NOT NULL,
            replay_from_line INTEGER NOT NULL,
            replay_to_line INTEGER NOT NULL,
            retry_count INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'ready',
            first_seen_utc TEXT NOT NULL,
            last_seen_utc TEXT NOT NULL,
            last_ack_utc TEXT NOT NULL DEFAULT ''
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS queue_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp_utc TEXT NOT NULL,
            source_rel TEXT NOT NULL,
            event TEXT NOT NULL,
            details TEXT NOT NULL DEFAULT ''
        )
        """
    )
    conn.commit()


def _priority_score(row: dict[str, Any], *, lane: str) -> float:
    pending = max(_safe_float(row.get("pending_lines"), 0.0), 0.0)
    total = max(_safe_float(row.get("total_lines"), pending), max(pending, 1.0))
    age = max(_safe_float(row.get("oldest_pending_age_seconds"), 0.0), 0.0)
    lane_boost = float(LANE_SPECS.get(lane, {}).get("boost", 1.0))
    return round(lane_boost * ((pending / 1000.0) + (pending / total * 40.0) + (age / 60.0)), 6)


def _fairness_bonus(row: dict[str, Any]) -> float:
    age_bonus = min(max(_safe_float(row.get("oldest_pending_age_seconds"), 0.0), 0.0) / 1800.0, 1.0)
    retry_bonus = min(max(_safe_int(row.get("retry_count"), 0), 0), 4) * 0.2
    return round(age_bonus + retry_bonus, 6)


def _adaptive_lane_shares(queue_rows: list[dict[str, Any]]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for lane, spec in LANE_SPECS.items():
        lane_rows = [row for row in queue_rows if str(row.get("lane")) == lane and str(row.get("status")) != "acked"]
        pending_lines = sum(max(int(row.get("pending_lines", 0) or 0), 0) for row in lane_rows)
        oldest_age = max((_safe_float(row.get("oldest_pending_age_seconds"), 0.0) for row in lane_rows), default=0.0)
        pressure_weight = float(spec["quota"]) + min(pending_lines / 2500.0, 4.0) * 0.12 + min(oldest_age / 1200.0, 3.0) * 0.08
        weights[lane] = max(pressure_weight, 0.01)
    total_weight = sum(weights.values()) or 1.0
    return {lane: round(weight / total_weight, 6) for lane, weight in weights.items()}


def _dispatch_plan(queue_rows: list[dict[str, Any]], *, top_n: int, adaptive_shares: dict[str, float]) -> list[dict[str, Any]]:
    limit = min(len(queue_rows), max(int(top_n), 1))
    if limit <= 0:
        return []

    per_lane_rows: dict[str, list[dict[str, Any]]] = {}
    for lane in LANE_SPECS:
        rows = [row for row in queue_rows if str(row.get("lane")) == lane and str(row.get("status")) != "acked"]
        rows.sort(
            key=lambda row: (
                -float(row.get("effective_priority_score", 0.0)),
                -int(row.get("pending_lines", 0) or 0),
                str(row.get("source_rel") or ""),
            )
        )
        per_lane_rows[lane] = rows

    budgets: dict[str, int] = {}
    remaining = limit
    lanes_with_rows = [lane for lane, rows in per_lane_rows.items() if rows]
    for idx, lane in enumerate(lanes_with_rows):
        if idx == len(lanes_with_rows) - 1:
            budget = remaining
        else:
            budget = int(round(limit * float(adaptive_shares.get(lane, 0.0))))
            if budget <= 0:
                budget = 1
            budget = min(budget, remaining)
        budgets[lane] = budget
        remaining -= budget

    plan: list[dict[str, Any]] = []
    for lane in LANE_SPECS:
        lane_rows = per_lane_rows.get(lane, [])
        for row in lane_rows[: max(budgets.get(lane, 0), 0)]:
            plan.append(dict(row))

    if len(plan) < limit:
        used = {str(row.get("source_rel") or "") for row in plan}
        remaining_rows = [row for row in queue_rows if str(row.get("source_rel") or "") not in used and str(row.get("status")) != "acked"]
        remaining_rows.sort(
            key=lambda row: (
                -float(row.get("effective_priority_score", 0.0)),
                -int(row.get("pending_lines", 0) or 0),
                str(row.get("source_rel") or ""),
            )
        )
        plan.extend(dict(row) for row in remaining_rows[: limit - len(plan)])
    return plan[:limit]


def _sync_rows(conn: sqlite3.Connection, payload: dict[str, Any], *, top_n: int) -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc).isoformat()
    synced: list[dict[str, Any]] = []
    seen: set[str] = set()
    for lane, spec in LANE_SPECS.items():
        raw_rows = payload.get(spec["source_key"])
        if not isinstance(raw_rows, list):
            continue
        for raw in raw_rows[: max(int(top_n), 1)]:
            if not isinstance(raw, dict):
                continue
            source_rel = str(raw.get("source_rel") or "").strip()
            if not source_rel:
                continue
            seen.add(source_rel)
            pending_lines = max(_safe_int(raw.get("pending_lines"), 0), 0)
            total_lines = max(_safe_int(raw.get("total_lines"), pending_lines), pending_lines)
            replay_from_line = max(_safe_int(raw.get("last_line"), 0), 0)
            replay_to_line = max(total_lines, replay_from_line)
            row = {
                "source_rel": source_rel,
                "lane": lane,
                "dedupe_key": source_rel,
                "quota_share": round(float(spec["quota"]), 4),
                "priority_score": _priority_score(raw, lane=lane),
                "pending_lines": pending_lines,
                "total_lines": total_lines,
                "oldest_pending_age_seconds": round(max(_safe_float(raw.get("oldest_pending_age_seconds"), 0.0), 0.0), 6),
                "replay_from_line": replay_from_line,
                "replay_to_line": replay_to_line,
            }
            conn.execute(
                """
                INSERT INTO queue_items (
                    source_rel, lane, dedupe_key, quota_share, priority_score, pending_lines, total_lines,
                    oldest_pending_age_seconds, replay_from_line, replay_to_line, retry_count, status,
                    first_seen_utc, last_seen_utc
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'ready', ?, ?)
                ON CONFLICT(source_rel) DO UPDATE SET
                    lane=excluded.lane,
                    dedupe_key=excluded.dedupe_key,
                    quota_share=excluded.quota_share,
                    priority_score=excluded.priority_score,
                    pending_lines=excluded.pending_lines,
                    total_lines=excluded.total_lines,
                    oldest_pending_age_seconds=excluded.oldest_pending_age_seconds,
                    replay_from_line=excluded.replay_from_line,
                    replay_to_line=excluded.replay_to_line,
                    last_seen_utc=excluded.last_seen_utc,
                    status=CASE
                        WHEN queue_items.status='acked' AND excluded.pending_lines=0 THEN 'acked'
                        WHEN excluded.pending_lines>0 AND queue_items.status='acked' THEN 'ready'
                        WHEN queue_items.status='retry' THEN 'retry'
                        ELSE 'ready'
                    END
                """,
                (
                    row["source_rel"],
                    row["lane"],
                    row["dedupe_key"],
                    row["quota_share"],
                    row["priority_score"],
                    row["pending_lines"],
                    row["total_lines"],
                    row["oldest_pending_age_seconds"],
                    row["replay_from_line"],
                    row["replay_to_line"],
                    now,
                    now,
                ),
            )
            synced.append(row)

    conn.execute(
        "UPDATE queue_items SET status='stale', last_seen_utc=? WHERE source_rel NOT IN ({})".format(
            ",".join("?" for _ in seen) or "''"
        ),
        (now, *sorted(seen)),
    )
    conn.commit()
    return synced


def _record_event(conn: sqlite3.Connection, source_rel: str, event: str, details: str = "") -> None:
    conn.execute(
        "INSERT INTO queue_events(timestamp_utc, source_rel, event, details) VALUES (?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), str(source_rel), str(event), str(details)),
    )
    conn.commit()


def _mark_retry(conn: sqlite3.Connection, source_rel: str) -> bool:
    cur = conn.execute(
        """
        UPDATE queue_items
        SET retry_count=retry_count+1, status='retry', last_seen_utc=?
        WHERE source_rel=?
        """,
        (datetime.now(timezone.utc).isoformat(), str(source_rel)),
    )
    changed = cur.rowcount > 0
    if changed:
        _record_event(conn, source_rel, "retry", "")
    return changed


def _ack(conn: sqlite3.Connection, source_rel: str) -> bool:
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        """
        UPDATE queue_items
        SET status='acked', pending_lines=0, replay_from_line=replay_to_line, last_ack_utc=?, last_seen_utc=?
        WHERE source_rel=?
        """,
        (now, now, str(source_rel)),
    )
    changed = cur.rowcount > 0
    if changed:
        _record_event(conn, source_rel, "ack", "")
        conn.commit()
    return changed


def build_payload(project_root: Path = PROJECT_ROOT, *, db_path: Path = DEFAULT_DB_PATH, top_n: int = 24) -> dict[str, Any]:
    health_root = project_root / "governance" / "health"
    backpressure_path = health_root / "ingestion_backpressure_latest.json"
    backpressure = _load_json(backpressure_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(str(db_path)) as conn:
        _ensure_schema(conn)
        synced = _sync_rows(conn, backpressure, top_n=top_n)
        queue_rows = [
            {
                "source_rel": row[0],
                "lane": row[1],
                "priority_score": float(row[2]),
                "quota_share": float(row[3]),
                "pending_lines": int(row[4]),
                "oldest_pending_age_seconds": float(row[5]),
                "retry_count": int(row[6]),
                "status": str(row[7]),
                "replay_from_line": int(row[8]),
                "replay_to_line": int(row[9]),
            }
            for row in conn.execute(
                """
                SELECT source_rel, lane, priority_score, quota_share, pending_lines,
                       oldest_pending_age_seconds, retry_count, status, replay_from_line, replay_to_line
                FROM queue_items
                WHERE status != 'stale'
                ORDER BY
                    CASE lane WHEN 'core' THEN 0 WHEN 'deferred' THEN 1 ELSE 2 END,
                    priority_score DESC,
                    pending_lines DESC,
                    source_rel ASC
                """
            ).fetchall()
        ]
        event_count = int(conn.execute("SELECT COUNT(*) FROM queue_events").fetchone()[0] or 0)

    adaptive_shares = _adaptive_lane_shares(queue_rows)
    for row in queue_rows:
        fairness_bonus = _fairness_bonus(row)
        row["fairness_bonus"] = fairness_bonus
        row["effective_priority_score"] = round(float(row["priority_score"]) + fairness_bonus, 6)

    lane_counts: dict[str, dict[str, Any]] = {}
    for lane in LANE_SPECS:
        lane_rows = [row for row in queue_rows if row["lane"] == lane]
        lane_counts[lane] = {
            "items": len(lane_rows),
            "pending_lines": sum(int(row["pending_lines"]) for row in lane_rows),
            "quota_share": float(LANE_SPECS[lane]["quota"]),
            "adaptive_quota_share": float(adaptive_shares.get(lane, 0.0)),
            "avg_fairness_bonus": round(
                sum(float(row.get("fairness_bonus", 0.0)) for row in lane_rows) / max(len(lane_rows), 1),
                6,
            ),
        }

    top_dispatch = _dispatch_plan(queue_rows, top_n=top_n, adaptive_shares=adaptive_shares)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": True,
        "queue_db": str(db_path),
        "source_file": str(backpressure_path),
        "items_synced": len(synced),
        "queue_depth": len(queue_rows),
        "event_count": event_count,
        "lane_counts": lane_counts,
        "dispatch_plan": top_dispatch,
        "retry_candidates": [row for row in queue_rows if int(row["retry_count"]) > 0][:10],
        "acked_items": [row for row in queue_rows if str(row["status"]) == "acked"][:10],
        "top_actions": [
            "drain core lane first until queue_depth_core is within quota",
            "treat deferred lane as quota-limited rather than FIFO so explanations and channels stop starving hot trading rows",
            "replay from replay_from_line for retried rows so ingestion recoveries stay deterministic",
            "ack queue items only after merge confirmation to keep dedupe state durable across restarts",
            "use adaptive_quota_share and fairness_bonus to keep old retries and pressured lanes from starving behind raw pending-line volume",
        ],
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Persist a durable prioritized ingestion queue from backpressure artifacts.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--top-n", type=int, default=24)
    parser.add_argument("--mark-retry", default="")
    parser.add_argument("--ack", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    db_path = Path(args.db_path).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        _ensure_schema(conn)
        if str(args.mark_retry or "").strip():
            _mark_retry(conn, str(args.mark_retry).strip())
        if str(args.ack or "").strip():
            _ack(conn, str(args.ack).strip())

    payload = build_payload(project_root, db_path=db_path, top_n=int(args.top_n))
    out_path = Path(args.out_file).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "ingestion_priority_queue "
            f"queue_depth={int(payload.get('queue_depth', 0) or 0)} "
            f"items_synced={int(payload.get('items_synced', 0) or 0)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
