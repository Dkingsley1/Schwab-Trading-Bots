import argparse
import csv
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "data" / "jsonl_link.sqlite3"


def _write_kv_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for r in rows:
            w.writerow(list(r))


def _q1(conn: sqlite3.Connection, sql: str, params: tuple = ()):
    row = conn.execute(sql, params).fetchone()
    return row[0] if row else None


def _qall(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[tuple]:
    return conn.execute(sql, params).fetchall()


def _safe_int(v, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _fmt_pct(v: float) -> str:
    return f"{v * 100.0:.2f}%"


def _is_crypto_symbol(symbol: str) -> bool:
    s = (symbol or "").upper().strip()
    return s.endswith("-USD") or s.endswith("-USDC") or s.endswith("-USDT")


def _stale_windows(ts_rows: Iterable[tuple], stale_seconds: int) -> int:
    stamps = []
    for (ts,) in ts_rows:
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            continue
        stamps.append(dt)
    stamps.sort()
    if len(stamps) < 2:
        return 0
    gaps = 0
    for i in range(1, len(stamps)):
        if (stamps[i] - stamps[i - 1]).total_seconds() > stale_seconds:
            gaps += 1
    return gaps


def _ensure_sql_snapshot_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS one_numbers_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            generated_utc TEXT NOT NULL,
            day_utc TEXT NOT NULL,
            source_report_dir TEXT NOT NULL,
            decision_total_rows INTEGER NOT NULL,
            stocks_decision_rows INTEGER NOT NULL,
            crypto_decision_rows INTEGER NOT NULL,
            watchdog_restarts INTEGER NOT NULL,
            data_quality_score REAL NOT NULL,
            alerts_json TEXT NOT NULL,
            metrics_json TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_one_numbers_day ON one_numbers_snapshots(day_utc)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_one_numbers_generated ON one_numbers_snapshots(generated_utc)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build one concise numbers file from SQL logs (stocks + crypto + alerts).")
    parser.add_argument("--day", default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "exports" / "one_numbers"))
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--stale-seconds", type=int, default=180)
    parser.add_argument("--no-sql-write", action="store_true", help="Do not persist summary snapshot into SQLite")
    args = parser.parse_args()

    day = args.day
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(args.db)

    if not db_path.exists():
        raise SystemExit(f"SQLite DB not found: {db_path}")

    conn = sqlite3.connect(str(db_path))

    decision_like = f"decision_explanations/%/decision_explanations_{day}.jsonl"
    governance_like = f"governance/%/master_control_{day}.jsonl"
    pnl_like = f"governance/%/shadow_pnl_attribution_{day}.jsonl"
    watchdog_like = f"governance/watchdog/watchdog_events_{day}.jsonl"

    # Combined totals
    decision_total_rows = _safe_int(_q1(conn, "SELECT COUNT(*) FROM jsonl_records WHERE source_rel LIKE ?", (decision_like,)), 0)
    governance_total_rows = _safe_int(_q1(conn, "SELECT COUNT(*) FROM jsonl_records WHERE source_rel LIKE ?", (governance_like,)), 0)

    status_rows = _qall(
        conn,
        """
        SELECT COALESCE(json_extract(payload_json, '$.status'), 'UNKNOWN') AS status, COUNT(*)
        FROM jsonl_records
        WHERE source_rel LIKE ?
        GROUP BY status
        """,
        (decision_like,),
    )
    status_counts = {str(k): _safe_int(v) for k, v in status_rows}

    action_rows = _qall(
        conn,
        """
        SELECT COALESCE(json_extract(payload_json, '$.action'), 'UNKNOWN') AS action, COUNT(*)
        FROM jsonl_records
        WHERE source_rel LIKE ?
        GROUP BY action
        """,
        (decision_like,),
    )
    action_counts = {str(k): _safe_int(v) for k, v in action_rows}

    # Exact stock/crypto split from decision symbols
    split_rows = _qall(
        conn,
        """
        SELECT
          CASE
            WHEN UPPER(COALESCE(json_extract(payload_json, '$.symbol'), '')) LIKE '%-USD'
              OR UPPER(COALESCE(json_extract(payload_json, '$.symbol'), '')) LIKE '%-USDC'
              OR UPPER(COALESCE(json_extract(payload_json, '$.symbol'), '')) LIKE '%-USDT'
            THEN 'crypto'
            ELSE 'stocks'
          END AS bucket,
          COUNT(*)
        FROM jsonl_records
        WHERE source_rel LIKE ?
        GROUP BY bucket
        """,
        (decision_like,),
    )
    split_counts = {str(k): _safe_int(v) for k, v in split_rows}
    stocks_decision_rows = split_counts.get("stocks", 0)
    crypto_decision_rows = split_counts.get("crypto", 0)

    split_action_rows = _qall(
        conn,
        """
        SELECT
          CASE
            WHEN UPPER(COALESCE(json_extract(payload_json, '$.symbol'), '')) LIKE '%-USD'
              OR UPPER(COALESCE(json_extract(payload_json, '$.symbol'), '')) LIKE '%-USDC'
              OR UPPER(COALESCE(json_extract(payload_json, '$.symbol'), '')) LIKE '%-USDT'
            THEN 'crypto'
            ELSE 'stocks'
          END AS bucket,
          COALESCE(json_extract(payload_json, '$.action'), 'UNKNOWN') AS action,
          COUNT(*)
        FROM jsonl_records
        WHERE source_rel LIKE ?
        GROUP BY bucket, action
        """,
        (decision_like,),
    )

    stocks_actions = {"BUY": 0, "SELL": 0, "HOLD": 0}
    crypto_actions = {"BUY": 0, "SELL": 0, "HOLD": 0}
    for bucket, action, cnt in split_action_rows:
        b = str(bucket)
        a = str(action)
        if b == "stocks" and a in stocks_actions:
            stocks_actions[a] = _safe_int(cnt)
        if b == "crypto" and a in crypto_actions:
            crypto_actions[a] = _safe_int(cnt)

    # Top symbols and concentration
    top_symbols_rows = _qall(
        conn,
        """
        SELECT COALESCE(json_extract(payload_json, '$.symbol'), 'UNKNOWN') AS symbol, COUNT(*) AS rows
        FROM jsonl_records
        WHERE source_rel LIKE ?
        GROUP BY symbol
        ORDER BY rows DESC
        LIMIT 50
        """,
        (decision_like,),
    )
    top_symbols = [(str(sym), _safe_int(cnt)) for sym, cnt in top_symbols_rows]
    top3_total = sum(cnt for _, cnt in top_symbols[:3])
    symbol_concentration_top3_share = (top3_total / max(decision_total_rows, 1))

    stocks_top = [(s, c) for s, c in top_symbols if not _is_crypto_symbol(s)]
    crypto_top = [(s, c) for s, c in top_symbols if _is_crypto_symbol(s)]

    # Governance action mix
    gov_action_rows = _qall(
        conn,
        """
        SELECT COALESCE(json_extract(payload_json, '$.master_action'), 'UNKNOWN') AS action, COUNT(*)
        FROM jsonl_records
        WHERE source_rel LIKE ?
        GROUP BY action
        """,
        (governance_like,),
    )
    gov_actions = {str(k): _safe_int(v) for k, v in gov_action_rows}

    # PnL proxy splits and by-strategy
    pnl_split_rows = _qall(
        conn,
        """
        SELECT
          CASE
            WHEN UPPER(COALESCE(json_extract(payload_json, '$.symbol'), '')) LIKE '%-USD'
              OR UPPER(COALESCE(json_extract(payload_json, '$.symbol'), '')) LIKE '%-USDC'
              OR UPPER(COALESCE(json_extract(payload_json, '$.symbol'), '')) LIKE '%-USDT'
            THEN 'crypto'
            ELSE 'stocks'
          END AS bucket,
          SUM(CAST(COALESCE(json_extract(payload_json, '$.pnl_proxy'), 0.0) AS REAL))
        FROM jsonl_records
        WHERE source_rel LIKE ?
        GROUP BY bucket
        """,
        (pnl_like,),
    )
    pnl_split = {str(k): _safe_float(v) for k, v in pnl_split_rows}
    stocks_pnl_proxy = pnl_split.get("stocks", 0.0)
    crypto_pnl_proxy = pnl_split.get("crypto", 0.0)

    pnl_strategy_rows = _qall(
        conn,
        """
        SELECT
          COALESCE(json_extract(payload_json, '$.bot_id'), 'UNKNOWN') AS bot_id,
          SUM(CAST(COALESCE(json_extract(payload_json, '$.pnl_proxy'), 0.0) AS REAL)) AS pnl
        FROM jsonl_records
        WHERE source_rel LIKE ?
        GROUP BY bot_id
        ORDER BY ABS(pnl) DESC
        LIMIT 8
        """,
        (pnl_like,),
    )

    # Watchdog signals
    wd_counts = _qall(
        conn,
        """
        SELECT
          COALESCE(json_extract(payload_json, '$.targets[0].action'), 'none') AS a0,
          COALESCE(json_extract(payload_json, '$.targets[1].action'), 'none') AS a1,
          COUNT(*)
        FROM jsonl_records
        WHERE source_rel LIKE ?
        GROUP BY a0, a1
        """,
        (watchdog_like,),
    )
    watchdog_restarts = 0
    watchdog_throttled = 0
    watchdog_restart_errors = 0
    for a0, a1, cnt in wd_counts:
        actions = {str(a0), str(a1)}
        c = _safe_int(cnt)
        if "restart" in actions:
            watchdog_restarts += c
        if "throttled" in actions:
            watchdog_throttled += c
        if "error" in actions:
            watchdog_restart_errors += c

    # Time-sliced stability metrics
    now_utc = datetime.now(timezone.utc)

    def _slice_counts(minutes: int) -> tuple[int, int, int, int]:
        cutoff = (now_utc - timedelta(minutes=minutes)).isoformat()
        row = _qall(
            conn,
            """
            SELECT
              COUNT(*) AS n,
              SUM(CASE WHEN COALESCE(json_extract(payload_json, '$.action'), '')='BUY' THEN 1 ELSE 0 END) AS buy_n,
              SUM(CASE WHEN COALESCE(json_extract(payload_json, '$.action'), '')='SELL' THEN 1 ELSE 0 END) AS sell_n,
              SUM(CASE WHEN COALESCE(json_extract(payload_json, '$.status'), '') IN ('BLOCKED','DATA_ONLY_BLOCKED') THEN 1 ELSE 0 END) AS blocked_n
            FROM jsonl_records
            WHERE source_rel LIKE ?
              AND julianday(replace(COALESCE(json_extract(payload_json, '$.timestamp_utc'), ''), 'Z', '+00:00')) >= julianday(?)
            """,
            (decision_like, cutoff),
        )[0]
        return _safe_int(row[0]), _safe_int(row[1]), _safe_int(row[2]), _safe_int(row[3])

    s15 = _slice_counts(15)
    s60 = _slice_counts(60)
    s240 = _slice_counts(240)

    def _imbalance(b: int, s: int) -> float:
        denom = max(b + s, 1)
        return (b - s) / denom

    # Stale windows in the last 4h for decisions/governance
    cutoff_4h = (now_utc - timedelta(hours=4)).isoformat()
    decision_ts_rows = _qall(
        conn,
        """
        SELECT COALESCE(json_extract(payload_json, '$.timestamp_utc'), '')
        FROM jsonl_records
        WHERE source_rel LIKE ?
          AND julianday(replace(COALESCE(json_extract(payload_json, '$.timestamp_utc'), ''), 'Z', '+00:00')) >= julianday(?)
        ORDER BY id ASC
        """,
        (decision_like, cutoff_4h),
    )
    governance_ts_rows = _qall(
        conn,
        """
        SELECT COALESCE(json_extract(payload_json, '$.timestamp_utc'), '')
        FROM jsonl_records
        WHERE source_rel LIKE ?
          AND julianday(replace(COALESCE(json_extract(payload_json, '$.timestamp_utc'), ''), 'Z', '+00:00')) >= julianday(?)
        ORDER BY id ASC
        """,
        (governance_like, cutoff_4h),
    )
    decision_stale_windows = _stale_windows(decision_ts_rows, args.stale_seconds)
    governance_stale_windows = _stale_windows(governance_ts_rows, args.stale_seconds)

    # Hold-no-edge diagnostic
    hold_no_edge = _safe_int(
        _q1(
            conn,
            """
            SELECT COUNT(*)
            FROM jsonl_records
            WHERE source_rel LIKE ?
              AND COALESCE(json_extract(payload_json, '$.action'), '')='HOLD'
              AND (
                payload_json LIKE '%inside_no_trade_band%'
                OR payload_json LIKE '%options_filter_no_clear_edge%'
              )
            """,
            (decision_like,),
        ),
        0,
    )

    blocked_total = status_counts.get("BLOCKED", 0) + status_counts.get("DATA_ONLY_BLOCKED", 0)
    blocked_rate = blocked_total / max(decision_total_rows, 1)
    hold_no_edge_rate = hold_no_edge / max(action_counts.get("HOLD", 0), 1)

    # Drift flag: compare buy rate last 1h vs last 4h baseline.
    buy_rate_1h = s60[1] / max(s60[0], 1)
    buy_rate_4h = s240[1] / max(s240[0], 1)
    buy_rate_drift_abs = abs(buy_rate_1h - buy_rate_4h)
    model_drift_flag = buy_rate_drift_abs >= 0.20

    # Freshness ages
    last_decision_ts = _q1(
        conn,
        """
        SELECT MAX(COALESCE(json_extract(payload_json, '$.timestamp_utc'), ''))
        FROM jsonl_records
        WHERE source_rel LIKE ?
        """,
        (decision_like,),
    )
    last_governance_ts = _q1(
        conn,
        """
        SELECT MAX(COALESCE(json_extract(payload_json, '$.timestamp_utc'), ''))
        FROM jsonl_records
        WHERE source_rel LIKE ?
        """,
        (governance_like,),
    )

    def _age_seconds(ts_raw) -> int:
        if not ts_raw:
            return 10**9
        try:
            dt = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            return 10**9
        return max(int((now_utc - dt).total_seconds()), 0)

    decision_last_age_sec = _age_seconds(last_decision_ts)
    governance_last_age_sec = _age_seconds(last_governance_ts)

    # Heartbeat health (filesystem)
    hb_dir = PROJECT_ROOT / "governance" / "health"
    heartbeat_files = list(hb_dir.glob("shadow_loop_*.json")) if hb_dir.exists() else []
    heartbeat_recent = 0

    # Bot stack status report integration (masterbots + sub-bots)
    bot_stack_status = "unknown"
    bot_stack_active_sub_bots = 0
    bot_stack_watchdog_schwab_live = False
    bot_stack_watchdog_coinbase_live = False
    bot_stack_latest_json = PROJECT_ROOT / "exports" / "bot_stack_status" / "latest.json"
    if bot_stack_latest_json.exists():
        try:
            bot_obj = json.loads(bot_stack_latest_json.read_text(encoding="utf-8"))
            bot_stack_status = str((bot_obj.get("overall_health") or {}).get("status") or "unknown")
            bot_stack_active_sub_bots = _safe_int(((bot_obj.get("registry") or {}).get("counts") or {}).get("active"), 0)
            checks = ((bot_obj.get("overall_health") or {}).get("checks") or [])
            for chk in checks:
                name = str((chk or {}).get("name", ""))
                ok = bool((chk or {}).get("ok"))
                if name == "watchdog_schwab_live":
                    bot_stack_watchdog_schwab_live = ok
                elif name == "watchdog_coinbase_live":
                    bot_stack_watchdog_coinbase_live = ok
        except Exception:
            pass
    for fp in heartbeat_files:
        try:
            obj = json.loads(fp.read_text(encoding="utf-8"))
            ts = datetime.fromisoformat(str(obj.get("timestamp_utc", "")).replace("Z", "+00:00")).astimezone(timezone.utc)
            if (now_utc - ts).total_seconds() <= 180:
                heartbeat_recent += 1
        except Exception:
            continue

    # Data quality score
    score = 100.0
    if decision_total_rows == 0:
        score -= 40
    if governance_total_rows == 0:
        score -= 25
    score -= min(max(decision_last_age_sec - 120, 0) / 30.0, 20.0)
    score -= min(max(governance_last_age_sec - 180, 0) / 45.0, 15.0)
    if heartbeat_recent == 0:
        score -= 15
    score -= min(watchdog_restarts * 1.0, 8.0)
    score -= min(watchdog_throttled * 2.0, 10.0)
    score -= min(watchdog_restart_errors * 3.0, 12.0)
    data_quality_score = max(min(score, 100.0), 0.0)

    # Alert flags
    alerts = {
        "ALERT_WATCHDOG_RESTARTS": watchdog_restarts > 0,
        "ALERT_STALE_WINDOWS": (decision_stale_windows + governance_stale_windows) > 0,
        "ALERT_BLOCKED_RATE": blocked_rate > 0.25,
        "ALERT_SYMBOL_CONCENTRATION": symbol_concentration_top3_share > 0.75,
        "ALERT_MODEL_DRIFT": model_drift_flag,
        "ALERT_DATA_QUALITY": data_quality_score < 80.0,
    }

    # Build output rows
    generated_utc = now_utc.isoformat()
    rows: list[tuple[str, str]] = [
        ("day_utc", day),
        ("generated_utc", generated_utc),
        ("db_path", str(db_path)),
        ("combined_decision_total_rows", str(decision_total_rows)),
        ("combined_governance_total_rows", str(governance_total_rows)),
        ("combined_action_buy", str(action_counts.get("BUY", 0))),
        ("combined_action_sell", str(action_counts.get("SELL", 0))),
        ("combined_action_hold", str(action_counts.get("HOLD", 0))),
        ("combined_blocked_total", str(blocked_total)),
        ("combined_blocked_rate", f"{blocked_rate:.6f}"),
        ("stocks_decision_rows", str(stocks_decision_rows)),
        ("stocks_action_buy", str(stocks_actions.get("BUY", 0))),
        ("stocks_action_sell", str(stocks_actions.get("SELL", 0))),
        ("stocks_action_hold", str(stocks_actions.get("HOLD", 0))),
        ("crypto_decision_rows", str(crypto_decision_rows)),
        ("crypto_action_buy", str(crypto_actions.get("BUY", 0))),
        ("crypto_action_sell", str(crypto_actions.get("SELL", 0))),
        ("crypto_action_hold", str(crypto_actions.get("HOLD", 0))),
        ("stocks_pnl_proxy", f"{stocks_pnl_proxy:.6f}"),
        ("crypto_pnl_proxy", f"{crypto_pnl_proxy:.6f}"),
        ("timeslice_15m_rows", str(s15[0])),
        ("timeslice_15m_buy_sell_imbalance", f"{_imbalance(s15[1], s15[2]):.6f}"),
        ("timeslice_15m_blocked_rate", f"{(s15[3] / max(s15[0], 1)):.6f}"),
        ("timeslice_1h_rows", str(s60[0])),
        ("timeslice_1h_buy_sell_imbalance", f"{_imbalance(s60[1], s60[2]):.6f}"),
        ("timeslice_1h_blocked_rate", f"{(s60[3] / max(s60[0], 1)):.6f}"),
        ("timeslice_4h_rows", str(s240[0])),
        ("timeslice_4h_buy_sell_imbalance", f"{_imbalance(s240[1], s240[2]):.6f}"),
        ("timeslice_4h_blocked_rate", f"{(s240[3] / max(s240[0], 1)):.6f}"),
        ("decision_stale_windows_4h", str(decision_stale_windows)),
        ("governance_stale_windows_4h", str(governance_stale_windows)),
        ("hold_no_edge_rate", f"{hold_no_edge_rate:.6f}"),
        ("symbol_concentration_top3_share", f"{symbol_concentration_top3_share:.6f}"),
        ("buy_rate_1h", f"{buy_rate_1h:.6f}"),
        ("buy_rate_4h", f"{buy_rate_4h:.6f}"),
        ("buy_rate_drift_abs", f"{buy_rate_drift_abs:.6f}"),
        ("model_drift_flag", str(model_drift_flag).lower()),
        ("watchdog_restarts", str(watchdog_restarts)),
        ("watchdog_throttled", str(watchdog_throttled)),
        ("watchdog_restart_errors", str(watchdog_restart_errors)),
        ("decision_last_age_sec", str(decision_last_age_sec)),
        ("governance_last_age_sec", str(governance_last_age_sec)),
        ("heartbeat_recent_count", str(heartbeat_recent)),
        ("data_quality_score", f"{data_quality_score:.2f}"),
        ("bot_stack_overall_status", bot_stack_status),
        ("bot_stack_active_sub_bots", str(bot_stack_active_sub_bots)),
        ("bot_stack_watchdog_schwab_live", str(bot_stack_watchdog_schwab_live).lower()),
        ("bot_stack_watchdog_coinbase_live", str(bot_stack_watchdog_coinbase_live).lower()),
    ]

    for k, v in alerts.items():
        rows.append((k, str(v).lower()))

    for i, (sym, cnt) in enumerate(stocks_top[:3], start=1):
        rows.append((f"stocks_top_symbol_{i}", f"{sym}:{cnt}"))
    for i in range(len(stocks_top[:3]) + 1, 4):
        rows.append((f"stocks_top_symbol_{i}", "n/a"))

    for i, (sym, cnt) in enumerate(crypto_top[:3], start=1):
        rows.append((f"crypto_top_symbol_{i}", f"{sym}:{cnt}"))
    for i in range(len(crypto_top[:3]) + 1, 4):
        rows.append((f"crypto_top_symbol_{i}", "n/a"))

    for i, (bot_id, pnl) in enumerate(pnl_strategy_rows[:5], start=1):
        rows.append((f"pnl_strategy_{i}", f"{bot_id}:{_safe_float(pnl):.6f}"))
    for i in range(len(pnl_strategy_rows[:5]) + 1, 6):
        rows.append((f"pnl_strategy_{i}", "n/a"))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"one_numbers_{day}_{stamp}.csv"
    md_path = out_dir / f"one_numbers_{day}_{stamp}.md"

    _write_kv_csv(csv_path, rows)

    md_lines = [
        f"# One Numbers Report ({day})",
        "",
        f"Generated: {generated_utc}",
        "",
        "## Combined",
        f"- Decisions: {decision_total_rows}",
        f"- Actions: BUY={action_counts.get('BUY',0)}, SELL={action_counts.get('SELL',0)}, HOLD={action_counts.get('HOLD',0)}",
        f"- Blocked: {blocked_total} ({_fmt_pct(blocked_rate)})",
        f"- Data quality score: {data_quality_score:.2f}/100",
        "",
        "## Stocks",
        f"- Rows: {stocks_decision_rows}",
        f"- Actions: BUY={stocks_actions.get('BUY',0)}, SELL={stocks_actions.get('SELL',0)}, HOLD={stocks_actions.get('HOLD',0)}",
        f"- PnL proxy: {stocks_pnl_proxy:.6f}",
        f"- Top symbols: {rows[-11][1]}, {rows[-10][1]}, {rows[-9][1]}",
        "",
        "## Crypto",
        f"- Rows: {crypto_decision_rows}",
        f"- Actions: BUY={crypto_actions.get('BUY',0)}, SELL={crypto_actions.get('SELL',0)}, HOLD={crypto_actions.get('HOLD',0)}",
        f"- PnL proxy: {crypto_pnl_proxy:.6f}",
        f"- Top symbols: {rows[-8][1]}, {rows[-7][1]}, {rows[-6][1]}",
        "",
        "## Stability (15m / 1h / 4h)",
        f"- Rows: {s15[0]} / {s60[0]} / {s240[0]}",
        f"- Buy-sell imbalance: {_imbalance(s15[1], s15[2]):.4f} / {_imbalance(s60[1], s60[2]):.4f} / {_imbalance(s240[1], s240[2]):.4f}",
        f"- Blocked rate: {_fmt_pct(s15[3]/max(s15[0],1))} / {_fmt_pct(s60[3]/max(s60[0],1))} / {_fmt_pct(s240[3]/max(s240[0],1))}",
        f"- Stale windows (decision/governance): {decision_stale_windows}/{governance_stale_windows}",
        "",
        "## Risk/Diagnostics",
        f"- Hold-no-edge rate: {_fmt_pct(hold_no_edge_rate)}",
        f"- Symbol concentration top3 share: {_fmt_pct(symbol_concentration_top3_share)}",
        f"- Drift abs (buy_rate 1h vs 4h): {buy_rate_drift_abs:.4f} (flag={str(model_drift_flag).lower()})",
        "",
        "## Bot Stack",
        f"- Overall status: {bot_stack_status}",
        f"- Active sub-bots: {bot_stack_active_sub_bots}",
        f"- Watchdog live (schwab/coinbase): {str(bot_stack_watchdog_schwab_live).lower()}/{str(bot_stack_watchdog_coinbase_live).lower()}",
        f"- Source: {bot_stack_latest_json}",
        "",
        "## Alerts",
    ]
    md_lines.extend([f"- {k}: {str(v).lower()}" for k, v in alerts.items()])
    md_lines.append("")
    md_lines.append(f"CSV: `{csv_path}`")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    latest_csv = out_dir / "latest.csv"
    latest_md = out_dir / "latest.md"
    latest_json = out_dir / "one_numbers_summary.json"

    metric_map = {k: v for k, v in rows}
    summary_payload = {
        "generated_utc": generated_utc,
        "day_utc": day,
        **metric_map,
    }
    latest_json.write_text(json.dumps(summary_payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if latest_csv.exists() or latest_csv.is_symlink():
        latest_csv.unlink()
    if latest_md.exists() or latest_md.is_symlink():
        latest_md.unlink()
    latest_csv.symlink_to(csv_path)
    latest_md.symlink_to(md_path)

    # SQL register snapshot
    if not args.no_sql_write:
        _ensure_sql_snapshot_table(conn)
        conn.execute(
            """
            INSERT INTO one_numbers_snapshots (
                generated_utc, day_utc, source_report_dir,
                decision_total_rows, stocks_decision_rows, crypto_decision_rows,
                watchdog_restarts, data_quality_score,
                alerts_json, metrics_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                generated_utc,
                day,
                str(PROJECT_ROOT / "exports" / "sql_reports" / "latest"),
                decision_total_rows,
                stocks_decision_rows,
                crypto_decision_rows,
                watchdog_restarts,
                data_quality_score,
                json.dumps(alerts, ensure_ascii=True),
                json.dumps(metric_map, ensure_ascii=True),
            ),
        )
        conn.commit()

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")
    print(f"Latest CSV: {latest_csv}")
    print(f"Latest MD: {latest_md}")
    print(f"Latest JSON: {latest_json}")
    if not args.no_sql_write:
        print("Registered snapshot in SQLite table: one_numbers_snapshots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
