import argparse
import csv
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "data" / "jsonl_link.sqlite3"


def _rows(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[tuple]:
    cur = conn.execute(sql, params)
    return cur.fetchall()


def _write_csv(path: Path, headers: list[str], rows: Iterable[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            w.writerow(list(r))


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _stale_windows(ts_rows: list[tuple], stale_seconds: int) -> int:
    stamps = []
    for (ts,) in ts_rows:
        dt = _parse_iso(ts)
        if dt is not None:
            stamps.append(dt)
    stamps.sort()
    if len(stamps) < 2:
        return 0
    gaps = 0
    for i in range(1, len(stamps)):
        if (stamps[i] - stamps[i - 1]).total_seconds() > stale_seconds:
            gaps += 1
    return gaps


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate SQL-based runtime log reports from jsonl_link.sqlite3")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--day", default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    parser.add_argument("--stale-seconds", type=int, default=180)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (PROJECT_ROOT / "exports" / "sql_reports" / stamp)
    out_dir.mkdir(parents=True, exist_ok=True)

    day = args.day
    day_decision_like = f"decision_explanations/%/decision_explanations_{day}.jsonl"
    day_governance_like = f"governance/%/master_control_{day}.jsonl"
    day_watchdog_like = f"governance/watchdog/watchdog_events_{day}.jsonl"

    conn = sqlite3.connect(str(db_path))

    total_rows = _rows(conn, "SELECT COUNT(*) FROM jsonl_records")[0][0]
    min_ing, max_ing = _rows(conn, "SELECT MIN(ingested_at), MAX(ingested_at) FROM jsonl_records")[0]

    top_sources = _rows(
        conn,
        """
        SELECT source_rel, COUNT(*) AS rows
        FROM jsonl_records
        GROUP BY source_rel
        ORDER BY rows DESC
        LIMIT 50
        """,
    )
    _write_csv(out_dir / "top_sources.csv", ["source_rel", "rows"], top_sources)

    decision_status = _rows(
        conn,
        """
        SELECT COALESCE(json_extract(payload_json, '$.status'), 'UNKNOWN') AS status, COUNT(*) AS rows
        FROM jsonl_records
        WHERE source_rel LIKE ?
        GROUP BY status
        ORDER BY rows DESC
        """,
        (day_decision_like,),
    )
    _write_csv(out_dir / "decision_status_counts.csv", ["status", "rows"], decision_status)

    decision_actions = _rows(
        conn,
        """
        SELECT COALESCE(json_extract(payload_json, '$.action'), 'UNKNOWN') AS action, COUNT(*) AS rows
        FROM jsonl_records
        WHERE source_rel LIKE ?
        GROUP BY action
        ORDER BY rows DESC
        """,
        (day_decision_like,),
    )
    _write_csv(out_dir / "decision_action_counts.csv", ["action", "rows"], decision_actions)

    top_symbols = _rows(
        conn,
        """
        SELECT COALESCE(json_extract(payload_json, '$.symbol'), 'UNKNOWN') AS symbol, COUNT(*) AS rows
        FROM jsonl_records
        WHERE source_rel LIKE ?
        GROUP BY symbol
        ORDER BY rows DESC
        LIMIT 40
        """,
        (day_decision_like,),
    )
    _write_csv(out_dir / "top_symbols.csv", ["symbol", "rows"], top_symbols)

    governance_actions = _rows(
        conn,
        """
        SELECT COALESCE(json_extract(payload_json, '$.master_action'), 'UNKNOWN') AS master_action, COUNT(*) AS rows
        FROM jsonl_records
        WHERE source_rel LIKE ?
        GROUP BY master_action
        ORDER BY rows DESC
        """,
        (day_governance_like,),
    )
    _write_csv(out_dir / "governance_master_action_counts.csv", ["master_action", "rows"], governance_actions)

    watchdog_actions = _rows(
        conn,
        """
        SELECT 
          COALESCE(json_extract(payload_json, '$.targets[0].action'), 'UNKNOWN') AS schwab_action,
          COALESCE(json_extract(payload_json, '$.targets[1].action'), 'UNKNOWN') AS coinbase_action,
          COUNT(*) AS rows
        FROM jsonl_records
        WHERE source_rel LIKE ?
        GROUP BY schwab_action, coinbase_action
        ORDER BY rows DESC
        """,
        (day_watchdog_like,),
    )
    _write_csv(out_dir / "watchdog_action_matrix.csv", ["schwab_action", "coinbase_action", "rows"], watchdog_actions)

    recent_watchdog_events = _rows(
        conn,
        """
        SELECT
          json_extract(payload_json, '$.timestamp_utc') AS ts,
          json_extract(payload_json, '$.targets[0].note') AS schwab_note,
          json_extract(payload_json, '$.targets[1].note') AS coinbase_note,
          json_extract(payload_json, '$.targets[0].action') AS schwab_action,
          json_extract(payload_json, '$.targets[1].action') AS coinbase_action
        FROM jsonl_records
        WHERE source_rel LIKE ?
        ORDER BY id DESC
        LIMIT 30
        """,
        (day_watchdog_like,),
    )
    _write_csv(
        out_dir / "recent_watchdog_events.csv",
        ["timestamp_utc", "schwab_note", "coinbase_note", "schwab_action", "coinbase_action"],
        recent_watchdog_events,
    )

    decision_ts_rows = _rows(
        conn,
        """
        SELECT json_extract(payload_json, '$.timestamp_utc') AS ts
        FROM jsonl_records
        WHERE source_rel LIKE ?
        """,
        (day_decision_like,),
    )
    governance_ts_rows = _rows(
        conn,
        """
        SELECT json_extract(payload_json, '$.timestamp_utc') AS ts
        FROM jsonl_records
        WHERE source_rel LIKE ?
        """,
        (day_governance_like,),
    )

    decision_stale = _stale_windows(decision_ts_rows, args.stale_seconds)
    governance_stale = _stale_windows(governance_ts_rows, args.stale_seconds)

    status_counter = Counter({k: int(v) for k, v in decision_status})
    action_counter = Counter({k: int(v) for k, v in decision_actions})
    gov_counter = Counter({k: int(v) for k, v in governance_actions})

    md_lines: List[str] = []
    md_lines.append(f"# SQL Runtime Log Report ({day})")
    md_lines.append("")
    md_lines.append(f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}")
    md_lines.append(f"Database: `{db_path}`")
    md_lines.append(f"Total rows in `jsonl_records`: **{total_rows:,}**")
    md_lines.append(f"Ingested range: `{min_ing}` to `{max_ing}`")
    md_lines.append("")
    md_lines.append("## What Is Going On")

    decision_total = sum(status_counter.values())
    blocked = status_counter.get("BLOCKED", 0) + status_counter.get("DATA_ONLY_BLOCKED", 0)
    shadow_only = status_counter.get("SHADOW_ONLY", 0)
    hold_actions = action_counter.get("HOLD", 0)
    buy_actions = action_counter.get("BUY", 0)
    sell_actions = action_counter.get("SELL", 0)

    md_lines.append(f"- Decision rows for day: **{decision_total:,}**")
    md_lines.append(f"- Status mix: SHADOW_ONLY={shadow_only:,}, blocked(data/risk gates)={blocked:,}")
    md_lines.append(f"- Action mix: HOLD={hold_actions:,}, BUY={buy_actions:,}, SELL={sell_actions:,}")
    md_lines.append(f"- Governance master action mix: {', '.join(f'{k}={v:,}' for k, v in gov_counter.items()) if gov_counter else 'none'}")
    md_lines.append(f"- Staleness signals (>{args.stale_seconds}s gap): decisions={decision_stale}, governance={governance_stale}")

    if recent_watchdog_events:
        latest = recent_watchdog_events[0]
        md_lines.append(
            f"- Latest watchdog snapshot: schwab_action={latest[3]}, coinbase_action={latest[4]} (ts={latest[0]})"
        )

    md_lines.append("")
    md_lines.append("## Top Symbols (By Decision Volume)")
    for sym, cnt in top_symbols[:15]:
        md_lines.append(f"- {sym}: {cnt:,}")

    md_lines.append("")
    md_lines.append("## Report Files")
    md_lines.append("- `top_sources.csv`: highest volume source files")
    md_lines.append("- `decision_status_counts.csv`: status distribution")
    md_lines.append("- `decision_action_counts.csv`: action distribution")
    md_lines.append("- `top_symbols.csv`: symbol activity ranking")
    md_lines.append("- `governance_master_action_counts.csv`: master decision distribution")
    md_lines.append("- `watchdog_action_matrix.csv`: watchdog action combinations")
    md_lines.append("- `recent_watchdog_events.csv`: recent watchdog notes/actions")

    report_path = out_dir / "runtime_sql_report.md"
    report_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Wrote report: {report_path}")
    print(f"Wrote CSV files under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
