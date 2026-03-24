import argparse
import csv
import fcntl
import json
import os
import re
import sqlite3
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "data" / "jsonl_link.sqlite3"
DEFAULT_REPORT_TIMEZONE = "America/New_York"
DAY_SUFFIX_RE = re.compile(r"_(\d{8})\.jsonl$")


def _acquire_singleton_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        try:
            fh.seek(0)
            owner = fh.read().strip()
        except Exception:
            owner = "unknown"
        fh.close()
        raise RuntimeError(f"one_numbers lock busy lock_path={lock_path} owner={owner or 'unknown'}")

    fh.seek(0)
    fh.truncate(0)
    fh.write(f"pid={os.getpid()} started={datetime.now(timezone.utc).isoformat()} cmd=build_one_numbers_report")
    fh.flush()
    return fh


def _write_kv_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for r in rows:
            w.writerow(list(r))


def _xlsx_col_name(index: int) -> str:
    name = ""
    current = max(index, 1)
    while current:
        current, rem = divmod(current - 1, 26)
        name = chr(65 + rem) + name
    return name


def _xlsx_inline_cell(ref: str, value: str, style: int = 0) -> str:
    escaped = escape(str(value))
    style_attr = f' s="{style}"' if style else ""
    return f'<c r="{ref}" t="inlineStr"{style_attr}><is><t xml:space="preserve">{escaped}</t></is></c>'


def _humanize_metric_label(metric: str) -> str:
    text = str(metric or "").strip().replace("_", " ")
    if not text:
        return ""
    text = text.title()
    replacements = {
        "Utc": "UTC",
        "Pnl": "PnL",
        "Sql": "SQL",
        "Db": "DB",
        "Pct": "Pct",
        "1H": "1H",
        "4H": "4H",
        "15M": "15M",
        "Mtd": "MTD",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _report_title_date(day_value: str) -> str:
    raw = str(day_value or "").strip()
    if not raw:
        return "Unknown Date"
    try:
        return datetime.strptime(raw, "%Y%m%d").strftime("%B %d, %Y")
    except Exception:
        return raw


def _write_one_numbers_xlsx(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row_map = {str(metric): str(value) for metric, value in rows}
    logical_rows: list[tuple[str, str, int]] = []

    day = row_map.get("resolved_day") or row_map.get("day_utc") or row_map.get("requested_day") or "unknown"
    logical_rows.append((f"One Numbers Report ({_report_title_date(day)})", "", 2))
    logical_rows.append(("", "", 0))
    for label, metric in [
        ("Generated", "generated_utc"),
        ("Requested Day", "requested_day"),
        ("Resolved Day", "resolved_day"),
        ("Day Fallback Applied", "day_fallback_applied"),
    ]:
        if metric in row_map:
            logical_rows.append((label, row_map[metric], 0))
    if "db_path" in row_map:
        logical_rows.append(("Database Path", row_map["db_path"], 0))
    logical_rows.append(("", "", 0))

    section_name_map = {
        "Report Metadata": "Report Metadata",
        "Current Day": "Combined",
        "Month To Date": "Month To Date",
        "All Time": "All Time",
        "Detailed Metrics": "Detailed Metrics",
    }
    current_section = ""
    first_section = True
    for metric, value in rows:
        metric_str = str(metric)
        value_str = str(value)
        if metric_str.startswith("report_section_"):
            section_title = section_name_map.get(value_str, value_str)
            if section_title == "Report Metadata":
                current_section = section_title
                continue
            if not first_section:
                logical_rows.append(("", "", 0))
            logical_rows.append((section_title, "", 1))
            first_section = False
            current_section = section_title
            continue
        if metric_str in {"day_utc", "generated_utc", "requested_day", "resolved_day", "day_fallback_applied", "db_path"}:
            continue
        label = _humanize_metric_label(metric_str)
        logical_rows.append((label, value_str, 0))

    row_xml: list[str] = []
    used_rows = max(len(logical_rows), 1)
    for row_index, (left, right, style) in enumerate(logical_rows, start=1):
        cells = [
            _xlsx_inline_cell(f"{_xlsx_col_name(1)}{row_index}", left, style),
            _xlsx_inline_cell(f"{_xlsx_col_name(2)}{row_index}", right, style),
        ]
        row_xml.append(f'<row r="{row_index}" spans="1:2">{"".join(cells)}</row>')

    worksheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="A1:B{used_rows}"/>'
        f'<sheetViews><sheetView workbookViewId="0" tabSelected="1"><selection activeCell="A1" sqref="A1:B{used_rows}"/></sheetView></sheetViews>'
        '<sheetFormatPr defaultRowHeight="15"/>'
        '<cols>'
        '<col min="1" max="1" width="38" customWidth="1"/>'
        '<col min="2" max="2" width="68" customWidth="1"/>'
        '<col min="3" max="16384" width="0" hidden="1" customWidth="1"/>'
        '</cols>'
        f'<sheetData>{"".join(row_xml)}</sheetData>'
        '</worksheet>'
    )
    styles_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="3">'
        '<font><sz val="11"/><name val="Aptos"/></font>'
        '<font><b/><sz val="11"/><name val="Aptos"/></font>'
        '<font><b/><sz val="15"/><name val="Aptos"/></font>'
        '</fonts>'
        '<fills count="3">'
        '<fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FFDEDEDE"/><bgColor indexed="64"/></patternFill></fill>'
        '</fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="3">'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"/>'
        '<xf numFmtId="0" fontId="2" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"/>'
        '</cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        '</styleSheet>'
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="One Numbers" sheetId="1" r:id="rId1"/></sheets>'
        '</workbook>'
    )
    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
        '</Relationships>'
    )
    root_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '</Relationships>'
    )
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        '</Types>'
    )

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", root_rels_xml)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        zf.writestr("xl/styles.xml", styles_xml)
        zf.writestr("xl/worksheets/sheet1.xml", worksheet_xml)


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


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _report_timezone() -> timezone | ZoneInfo:
    tz_name = str(os.getenv("ONE_NUMBERS_REPORT_TIMEZONE", DEFAULT_REPORT_TIMEZONE) or DEFAULT_REPORT_TIMEZONE).strip()
    try:
        return ZoneInfo(tz_name)
    except Exception:
        return timezone.utc


def _default_report_day() -> str:
    return datetime.now(_report_timezone()).strftime("%Y%m%d")


def _empty_day_sources() -> dict[str, list[str]]:
    return {
        "decision": [],
        "governance": [],
        "pnl": [],
        "watchdog": [],
    }


def _extract_day_from_source_rel(source_rel: str) -> str:
    match = DAY_SUFFIX_RE.search(str(source_rel or ""))
    return match.group(1) if match else ""


def _sqlite_state_sources_by_day(sqlite_state: dict) -> dict[str, dict[str, list[str]]]:
    by_day: dict[str, dict[str, list[str]]] = {}
    for rel in sqlite_state:
        source_rel = str(rel)
        bucket = ""
        if source_rel.startswith("decision_explanations/") and "/decision_explanations_" in source_rel:
            bucket = "decision"
        elif source_rel.startswith("governance/") and "/master_control_" in source_rel:
            bucket = "governance"
        elif source_rel.startswith("governance/") and "/shadow_pnl_attribution_" in source_rel:
            bucket = "pnl"
        elif source_rel.startswith("governance/watchdog/watchdog_events_"):
            bucket = "watchdog"
        if not bucket:
            continue
        day = _extract_day_from_source_rel(source_rel)
        if not day:
            continue
        day_entry = by_day.setdefault(day, _empty_day_sources())
        day_entry[bucket].append(source_rel)
    for day_entry in by_day.values():
        for bucket in day_entry:
            day_entry[bucket] = sorted(day_entry[bucket])
    return by_day


def _resolve_report_day(requested_day: str, sqlite_state: dict) -> tuple[str, dict[str, list[str]]]:
    day_sources = _sqlite_state_sources_by_day(sqlite_state)
    requested = str(requested_day or "").strip() or _default_report_day()
    selected = day_sources.get(requested, _empty_day_sources())
    if selected["decision"] or selected["governance"]:
        return requested, selected

    candidates = sorted(
        day
        for day, entry in day_sources.items()
        if entry["decision"] or entry["governance"]
    )
    if not candidates:
        return requested, selected

    prior_or_equal = [day for day in candidates if day <= requested]
    resolved_day = prior_or_equal[-1] if prior_or_equal else candidates[-1]
    return resolved_day, day_sources.get(resolved_day, _empty_day_sources())


def _chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), max(size, 1))]


def _materialize_working_subset(
    conn: sqlite3.Connection,
    *,
    source_rel_values: list[str],
    decision_like: str,
    governance_like: str,
    pnl_like: str,
    watchdog_like: str,
) -> int:
    # Build a temp subset table named jsonl_records so all downstream queries stay unchanged.
    # This dramatically reduces runtime on very large main tables.
    conn.execute("DROP TABLE IF EXISTS temp.jsonl_records")
    conn.execute(
        """
        CREATE TEMP TABLE jsonl_records (
            id INTEGER,
            source_file TEXT,
            source_rel TEXT,
            line_no INTEGER,
            ingested_at TEXT,
            payload_sha1 TEXT,
            payload_json TEXT
        )
        """
    )

    inserted = 0
    unique_sources = sorted({str(s) for s in source_rel_values if str(s).strip()})
    if unique_sources:
        for chunk in _chunked(unique_sources, 200):
            placeholders = ",".join(["?"] * len(chunk))
            conn.execute(
                f"""
                INSERT INTO temp.jsonl_records (
                    id, source_file, source_rel, line_no, ingested_at, payload_sha1, payload_json
                )
                SELECT
                    id, source_file, source_rel, line_no, ingested_at, payload_sha1, payload_json
                FROM main.jsonl_records
                WHERE source_rel IN ({placeholders})
                """,
                tuple(chunk),
            )
        inserted = _safe_int(conn.execute("SELECT COUNT(*) FROM temp.jsonl_records").fetchone()[0], 0)

    if inserted == 0:
        conn.execute(
            """
            INSERT INTO temp.jsonl_records (
                id, source_file, source_rel, line_no, ingested_at, payload_sha1, payload_json
            )
            SELECT
                id, source_file, source_rel, line_no, ingested_at, payload_sha1, payload_json
            FROM main.jsonl_records
            WHERE source_rel LIKE ?
               OR source_rel LIKE ?
               OR source_rel LIKE ?
               OR source_rel LIKE ?
               OR source_rel='governance/health/preopen_replay_drift_history.jsonl'
            """,
            (decision_like, governance_like, pnl_like, watchdog_like),
        )
        inserted = _safe_int(conn.execute("SELECT COUNT(*) FROM temp.jsonl_records").fetchone()[0], 0)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_tmp_jsonl_source_rel ON jsonl_records(source_rel)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tmp_jsonl_id ON jsonl_records(id)")
    return inserted


def _is_crypto_symbol(symbol: str) -> bool:
    s = (symbol or "").upper().strip()
    return s.endswith("-USD") or s.endswith("-USDC") or s.endswith("-USDT")


def _is_futures_symbol(symbol: str) -> bool:
    s = (symbol or "").upper().strip()
    return s.startswith("/") or s.endswith("=F") or s.endswith("1!") or s.endswith("2!")


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


def _latest_daily_snapshots(conn: sqlite3.Connection) -> dict[str, dict]:
    try:
        rows = conn.execute(
            """
            SELECT s.day_utc, s.generated_utc, s.metrics_json
            FROM one_numbers_snapshots s
            JOIN (
                SELECT day_utc, MAX(id) AS max_id
                FROM one_numbers_snapshots
                GROUP BY day_utc
            ) latest
              ON latest.max_id = s.id
            """
        ).fetchall()
    except sqlite3.OperationalError:
        return {}

    out: dict[str, dict] = {}
    for day_utc, generated_utc, metrics_json in rows:
        metrics = {}
        try:
            metrics = json.loads(metrics_json or "{}")
        except Exception:
            metrics = {}
        out[str(day_utc)] = {
            "day_utc": str(day_utc),
            "generated_utc": str(generated_utc or ""),
            "metrics": metrics if isinstance(metrics, dict) else {},
        }
    return out


def _rollup_metric_int(metrics: dict, key: str) -> int:
    return _safe_int(metrics.get(key, 0), 0)


def _rollup_metric_float(metrics: dict, key: str) -> float:
    return _safe_float(metrics.get(key, 0.0), 0.0)


def _aggregate_rollup(entries: list[dict]) -> dict[str, str]:
    if not entries:
        return {
            "days_covered": "0",
            "decision_total_rows": "0",
            "governance_total_rows": "0",
            "blocked_total": "0",
            "paper_executed_total": "0",
            "watchdog_restarts": "0",
            "avg_data_quality_score": "0.00",
        }

    days_covered = len(entries)
    decision_total_rows = sum(_rollup_metric_int(e["metrics"], "combined_decision_total_rows") for e in entries)
    governance_total_rows = sum(_rollup_metric_int(e["metrics"], "combined_governance_total_rows") for e in entries)
    blocked_total = sum(_rollup_metric_int(e["metrics"], "combined_blocked_total") for e in entries)
    paper_executed_total = sum(_rollup_metric_int(e["metrics"], "paper_executed_total") for e in entries)
    watchdog_restarts = sum(_rollup_metric_int(e["metrics"], "watchdog_restarts") for e in entries)
    avg_data_quality_score = sum(_rollup_metric_float(e["metrics"], "data_quality_score") for e in entries) / max(days_covered, 1)
    return {
        "days_covered": str(days_covered),
        "decision_total_rows": str(decision_total_rows),
        "governance_total_rows": str(governance_total_rows),
        "blocked_total": str(blocked_total),
        "paper_executed_total": str(paper_executed_total),
        "watchdog_restarts": str(watchdog_restarts),
        "avg_data_quality_score": f"{avg_data_quality_score:.2f}",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build one concise numbers file from SQL logs (stocks + crypto + futures + options + alerts).")
    parser.add_argument("--day", default="", help="Preferred session day in YYYYMMDD. Defaults to the report timezone day and can fall back to the latest linked day with data.")
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "exports" / "one_numbers"))
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--stale-seconds", type=int, default=180)
    parser.add_argument("--no-sql-write", action="store_true", help="Do not persist summary snapshot into SQLite")
    args = parser.parse_args()

    lock_path = Path(os.getenv("ONE_NUMBERS_LOCK_PATH", str(PROJECT_ROOT / "governance" / "locks" / "one_numbers.lock")))
    lock_fh = None
    try:
        lock_fh = _acquire_singleton_lock(lock_path)
    except Exception as exc:
        print(f"{exc}")
        return 1

    requested_day = str(args.day or "").strip() or _default_report_day()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(args.db)

    if not db_path.exists():
        try:
            if lock_fh is not None:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
                lock_fh.close()
        except Exception:
            pass
        raise SystemExit(f"SQLite DB not found: {db_path}")

    conn = sqlite3.connect(str(db_path))

    day = requested_day
    decision_bucket_case = """
    CASE
      WHEN LOWER(COALESCE(source_rel, '')) LIKE '%futures%'
        OR LOWER(COALESCE(json_extract(payload_json, '$.mode'), '')) LIKE '%futures%'
        OR COALESCE(json_extract(payload_json, '$.symbol'), '') LIKE '/%'
        OR UPPER(COALESCE(json_extract(payload_json, '$.symbol'), '')) LIKE '%=F'
        OR UPPER(COALESCE(json_extract(payload_json, '$.symbol'), '')) LIKE '%1!'
      THEN 'futures'
      WHEN UPPER(COALESCE(json_extract(payload_json, '$.symbol'), '')) LIKE '%-USD'
        OR UPPER(COALESCE(json_extract(payload_json, '$.symbol'), '')) LIKE '%-USDC'
        OR UPPER(COALESCE(json_extract(payload_json, '$.symbol'), '')) LIKE '%-USDT'
      THEN 'crypto'
      ELSE 'stocks'
    END
    """
    pnl_bucket_case = """
    CASE
      WHEN LOWER(COALESCE(source_rel, '')) LIKE '%futures%'
        OR COALESCE(json_extract(payload_json, '$.symbol'), '') LIKE '/%'
        OR UPPER(COALESCE(json_extract(payload_json, '$.symbol'), '')) LIKE '%=F'
        OR UPPER(COALESCE(json_extract(payload_json, '$.symbol'), '')) LIKE '%1!'
      THEN 'futures'
      WHEN UPPER(COALESCE(json_extract(payload_json, '$.symbol'), '')) LIKE '%-USD'
        OR UPPER(COALESCE(json_extract(payload_json, '$.symbol'), '')) LIKE '%-USDC'
        OR UPPER(COALESCE(json_extract(payload_json, '$.symbol'), '')) LIKE '%-USDT'
      THEN 'crypto'
      ELSE 'stocks'
    END
    """

    state_obj = _read_json(PROJECT_ROOT / "governance" / "jsonl_sql_link_state.json")
    sqlite_state = state_obj.get("sqlite") if isinstance(state_obj.get("sqlite"), dict) else {}
    day, day_sources = _resolve_report_day(requested_day, sqlite_state)
    decision_sources_day = day_sources["decision"]
    governance_sources_day = day_sources["governance"]
    pnl_sources_day = day_sources["pnl"]
    watchdog_sources_day = day_sources["watchdog"]

    decision_like = f"decision_explanations/%/decision_explanations_{day}.jsonl"
    governance_like = f"governance/%/master_control_{day}.jsonl"
    pnl_like = f"governance/%/shadow_pnl_attribution_{day}.jsonl"
    watchdog_like = f"governance/watchdog/watchdog_events_{day}.jsonl"

    linked_source_files_total = len(sqlite_state) if sqlite_state else 0
    decision_source_files = len(decision_sources_day)
    governance_source_files = len(governance_sources_day)

    _ = _materialize_working_subset(
        conn,
        source_rel_values=(
            decision_sources_day
            + governance_sources_day
            + pnl_sources_day
            + watchdog_sources_day
            + ["governance/health/preopen_replay_drift_history.jsonl"]
        ),
        decision_like=decision_like,
        governance_like=governance_like,
        pnl_like=pnl_like,
        watchdog_like=watchdog_like,
    )

    # Combined totals (computed against the temp working subset).
    decision_total_rows = _safe_int(_q1(conn, "SELECT COUNT(*) FROM jsonl_records WHERE source_rel LIKE ?", (decision_like,)), 0)
    governance_total_rows = _safe_int(_q1(conn, "SELECT COUNT(*) FROM jsonl_records WHERE source_rel LIKE ?", (governance_like,)), 0)

    if decision_source_files == 0:
        decision_source_files = _safe_int(_q1(conn, "SELECT COUNT(DISTINCT source_rel) FROM jsonl_records WHERE source_rel LIKE ?", (decision_like,)), 0)
    if governance_source_files == 0:
        governance_source_files = _safe_int(_q1(conn, "SELECT COUNT(DISTINCT source_rel) FROM jsonl_records WHERE source_rel LIKE ?", (governance_like,)), 0)

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

    # Pre-compute decision bucket table once to avoid repeated full scans.
    conn.execute("DROP TABLE IF EXISTS temp.decision_bucketed")
    conn.execute(
        f"""
        CREATE TEMP TABLE decision_bucketed AS
        SELECT
          {decision_bucket_case} AS bucket,
          COALESCE(json_extract(payload_json, '$.symbol'), 'UNKNOWN') AS symbol,
          COALESCE(json_extract(payload_json, '$.action'), 'UNKNOWN') AS action,
          COALESCE(json_extract(payload_json, '$.status'), 'UNKNOWN') AS status
        FROM jsonl_records
        WHERE source_rel LIKE ?
        """,
        (decision_like,),
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tmp_decision_bucketed_bucket ON decision_bucketed(bucket)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tmp_decision_bucketed_symbol ON decision_bucketed(symbol)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tmp_decision_bucketed_action ON decision_bucketed(action)")

    # Exact stocks/crypto/futures split from decision payload + source path.
    split_rows = _qall(
        conn,
        """
        SELECT bucket, COUNT(*)
        FROM decision_bucketed
        GROUP BY bucket
        """,
    )
    split_counts = {str(k): _safe_int(v) for k, v in split_rows}
    stocks_decision_rows = split_counts.get("stocks", 0)
    crypto_decision_rows = split_counts.get("crypto", 0)
    futures_decision_rows = split_counts.get("futures", 0)

    split_action_rows = _qall(
        conn,
        """
        SELECT bucket, action, COUNT(*)
        FROM decision_bucketed
        GROUP BY bucket, action
        """,
    )

    stocks_actions = {"BUY": 0, "SELL": 0, "HOLD": 0}
    crypto_actions = {"BUY": 0, "SELL": 0, "HOLD": 0}
    futures_actions = {"BUY": 0, "SELL": 0, "HOLD": 0}
    for bucket, action, cnt in split_action_rows:
        b = str(bucket)
        a = str(action)
        if b == "stocks" and a in stocks_actions:
            stocks_actions[a] = _safe_int(cnt)
        if b == "crypto" and a in crypto_actions:
            crypto_actions[a] = _safe_int(cnt)
        if b == "futures" and a in futures_actions:
            futures_actions[a] = _safe_int(cnt)

    # Top symbols and concentration
    top_symbols_rows = _qall(
        conn,
        """
        SELECT symbol, COUNT(*) AS rows
        FROM decision_bucketed
        GROUP BY symbol
        ORDER BY rows DESC
        LIMIT 50
        """,
    )
    top_symbols = [(str(sym), _safe_int(cnt)) for sym, cnt in top_symbols_rows]
    top3_total = sum(cnt for _, cnt in top_symbols[:3])
    symbol_concentration_top3_share = (top3_total / max(decision_total_rows, 1))

    def _bucket_top_symbols(bucket: str) -> list[tuple[str, int]]:
        rows = _qall(
            conn,
            """
            SELECT symbol, COUNT(*) AS rows
            FROM decision_bucketed
            WHERE bucket=?
            GROUP BY symbol
            ORDER BY rows DESC
            LIMIT 50
            """,
            (bucket,),
        )
        return [(str(sym), _safe_int(cnt)) for sym, cnt in rows]

    stocks_top = _bucket_top_symbols("stocks")
    crypto_top = _bucket_top_symbols("crypto")
    futures_top = _bucket_top_symbols("futures")

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
    options_style_rows = _qall(
        conn,
        """
        SELECT COALESCE(NULLIF(json_extract(payload_json, '$.options_plan.options_style'), ''), 'NONE') AS style, COUNT(*)
        FROM jsonl_records
        WHERE source_rel LIKE ?
        GROUP BY style
        ORDER BY COUNT(*) DESC
        """,
        (governance_like,),
    )
    options_styles = [(str(style), _safe_int(cnt)) for style, cnt in options_style_rows]
    options_active_styles = [(style, cnt) for style, cnt in options_styles if style.upper() != "NONE"]
    options_decision_rows = sum(cnt for _, cnt in options_active_styles)
    options_none_rows = sum(cnt for style, cnt in options_styles if style.upper() == "NONE")

    futures_style_rows = _qall(
        conn,
        """
        SELECT COALESCE(NULLIF(json_extract(payload_json, '$.futures_plan.futures_style'), ''), 'NONE') AS style, COUNT(*)
        FROM jsonl_records
        WHERE source_rel LIKE ?
        GROUP BY style
        ORDER BY COUNT(*) DESC
        """,
        (governance_like,),
    )
    futures_styles = [(str(style), _safe_int(cnt)) for style, cnt in futures_style_rows]
    futures_active_styles = [(style, cnt) for style, cnt in futures_styles if style.upper() != "NONE"]
    futures_strategy_rows = sum(cnt for _, cnt in futures_active_styles)
    futures_none_rows = sum(cnt for style, cnt in futures_styles if style.upper() == "NONE")
    options_contracts_total = _safe_float(
        _q1(
            conn,
            """
            SELECT SUM(CAST(COALESCE(json_extract(payload_json, '$.options_plan.contracts'), 0.0) AS REAL))
            FROM jsonl_records
            WHERE source_rel LIKE ?
            """,
            (governance_like,),
        ),
        0.0,
    )
    options_master_action_rows = _qall(
        conn,
        """
        SELECT COALESCE(json_extract(payload_json, '$.options_master.action'), 'UNKNOWN') AS action, COUNT(*)
        FROM jsonl_records
        WHERE source_rel LIKE ?
        GROUP BY action
        """,
        (governance_like,),
    )
    options_master_actions = {"BUY": 0, "SELL": 0, "HOLD": 0}
    for action, cnt in options_master_action_rows:
        a = str(action)
        if a in options_master_actions:
            options_master_actions[a] = _safe_int(cnt)
    futures_governance_rows = _safe_int(
        _q1(
            conn,
            """
            SELECT COUNT(*)
            FROM jsonl_records
            WHERE source_rel LIKE ?
              AND (
                LOWER(source_rel) LIKE '%futures%'
                OR CAST(COALESCE(json_extract(payload_json, '$.active_futures_sub_bots'), 0) AS INTEGER) > 0
              )
            """,
            (governance_like,),
        ),
        0,
    )
    options_specialist_active_rows = _safe_int(
        _q1(
            conn,
            """
            SELECT COUNT(*)
            FROM jsonl_records
            WHERE source_rel LIKE ?
              AND CAST(COALESCE(json_extract(payload_json, '$.active_options_sub_bots'), 0) AS INTEGER) > 0
            """,
            (governance_like,),
        ),
        0,
    )
    futures_specialist_active_rows = _safe_int(
        _q1(
            conn,
            """
            SELECT COUNT(*)
            FROM jsonl_records
            WHERE source_rel LIKE ?
              AND CAST(COALESCE(json_extract(payload_json, '$.active_futures_sub_bots'), 0) AS INTEGER) > 0
            """,
            (governance_like,),
        ),
        0,
    )

    # PnL proxy splits and by-strategy
    pnl_split_rows = _qall(
        conn,
        f"""
        SELECT
          {pnl_bucket_case} AS bucket,
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
    futures_pnl_proxy = pnl_split.get("futures", 0.0)

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

    # Paper execution (for paper-trading visibility in one-page report).
    paper_executed_total = _safe_int(
        _q1(
            conn,
            """
            SELECT COUNT(*)
            FROM jsonl_records
            WHERE source_rel LIKE ?
              AND COALESCE(json_extract(payload_json, '$.status'), '')='PAPER_EXECUTED'
            """,
            (decision_like,),
        ),
        0,
    )
    paper_executed_crypto = _safe_int(
        _q1(
            conn,
            """
            SELECT COUNT(*)
            FROM jsonl_records
            WHERE source_rel LIKE ?
              AND COALESCE(json_extract(payload_json, '$.status'), '')='PAPER_EXECUTED'
              AND (
                UPPER(COALESCE(json_extract(payload_json, '$.symbol'), '')) LIKE '%-USD'
                OR UPPER(COALESCE(json_extract(payload_json, '$.symbol'), '')) LIKE '%-USDC'
                OR UPPER(COALESCE(json_extract(payload_json, '$.symbol'), '')) LIKE '%-USDT'
              )
            """,
            (decision_like,),
        ),
        0,
    )
    paper_bot_rows = _qall(
        conn,
        """
        SELECT COALESCE(json_extract(payload_json, '$.bot_id'), 'UNKNOWN') AS bot_id, COUNT(*)
        FROM jsonl_records
        WHERE source_rel LIKE ?
          AND COALESCE(json_extract(payload_json, '$.status'), '')='PAPER_EXECUTED'
        GROUP BY bot_id
        ORDER BY COUNT(*) DESC
        LIMIT 5
        """,
        (decision_like,),
    )

    # Guardrail counters for quick verification in the one-page report.
    guardrail_master_latency_slo_fail = _safe_int(
        _q1(
            conn,
            """
            SELECT COUNT(*)
            FROM jsonl_records
            WHERE source_rel LIKE ?
              AND payload_json LIKE '%master_latency_slo_ok=FAIL%'
            """,
            (decision_like,),
        ),
        0,
    )
    guardrail_feature_freshness_fail = _safe_int(
        _q1(
            conn,
            """
            SELECT COUNT(*)
            FROM jsonl_records
            WHERE source_rel LIKE ?
              AND payload_json LIKE '%feature_freshness_ok=FAIL%'
            """,
            (decision_like,),
        ),
        0,
    )
    guardrail_event_lock_hits = _safe_int(
        _q1(
            conn,
            """
            SELECT COUNT(*)
            FROM jsonl_records
            WHERE source_rel LIKE ?
              AND (
                payload_json LIKE '%event_lock_window%'
                OR payload_json LIKE '%event_lock%'
              )
            """,
            (decision_like,),
        ),
        0,
    )
    guardrail_canary_mentions = _safe_int(
        _q1(
            conn,
            """
            SELECT COUNT(*)
            FROM jsonl_records
            WHERE source_rel LIKE ?
              AND payload_json LIKE '%canary%'
            """,
            (governance_like,),
        ),
        0,
    )

    cutoff_24h = (now_utc - timedelta(hours=24)).isoformat()
    preopen_replay_rows_24h = _safe_int(
        _q1(
            conn,
            """
            SELECT COUNT(*)
            FROM jsonl_records
            WHERE source_rel='governance/health/preopen_replay_drift_history.jsonl'
              AND julianday(replace(COALESCE(json_extract(payload_json, '$.timestamp_utc'), ''), 'Z', '+00:00')) >= julianday(?)
            """,
            (cutoff_24h,),
        ),
        0,
    )
    preopen_replay_fail_24h = _safe_int(
        _q1(
            conn,
            """
            SELECT COUNT(*)
            FROM jsonl_records
            WHERE source_rel='governance/health/preopen_replay_drift_history.jsonl'
              AND julianday(replace(COALESCE(json_extract(payload_json, '$.timestamp_utc'), ''), 'Z', '+00:00')) >= julianday(?)
              AND (
                LOWER(COALESCE(json_extract(payload_json, '$.status'), '')) IN ('fail','failed','error')
                OR COALESCE(json_extract(payload_json, '$.ok'), 1)=0
                OR COALESCE(json_extract(payload_json, '$.passed'), 1)=0
              )
            """,
            (cutoff_24h,),
        ),
        0,
    )

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

    # Ops metadata (storage + SQL link writer health + canary state).
    logs_root = PROJECT_ROOT / "logs"
    try:
        storage_logs_target = str(logs_root.resolve())
    except Exception:
        storage_logs_target = str(logs_root)
    storage_mode = "external" if storage_logs_target.startswith("/Volumes/") else "local"

    sql_link_health = _read_json(PROJECT_ROOT / "governance" / "health" / "sql_link_service_latest.json")
    sql_link_ok = bool(sql_link_health.get("ok"))
    sql_link_rc = _safe_int(sql_link_health.get("rc"), -1)
    sql_link_db_size_gb = _safe_float(sql_link_health.get("sqlite_db_size_gb"), 0.0)
    hot_retention = sql_link_health.get("hot_retention") or {}
    hot_retention_ran = bool(hot_retention.get("ran"))
    hot_retention_rc = _safe_int(hot_retention.get("rc"), 0)
    hot_retention_db_after = _safe_float(hot_retention.get("db_size_gb_after"), 0.0)

    canary_state = _read_json(PROJECT_ROOT / "governance" / "canary_state.json")
    canary_weight = _safe_float(
        canary_state.get("weight", canary_state.get("current_weight", canary_state.get("applied_weight", 0.0))),
        0.0,
    )
    canary_enabled = bool(canary_state.get("enabled", canary_state.get("active", canary_weight > 0.0)))

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

    current_rollup_metrics = {
        "combined_decision_total_rows": decision_total_rows,
        "combined_governance_total_rows": governance_total_rows,
        "combined_blocked_total": blocked_total,
        "paper_executed_total": paper_executed_total,
        "watchdog_restarts": watchdog_restarts,
        "data_quality_score": data_quality_score,
    }
    latest_snapshots = _latest_daily_snapshots(conn)
    latest_snapshots[day] = {
        "day_utc": day,
        "generated_utc": now_utc.isoformat(),
        "metrics": current_rollup_metrics,
    }
    month_prefix = day[:6]
    month_rollup = _aggregate_rollup([entry for entry_day, entry in latest_snapshots.items() if str(entry_day).startswith(month_prefix)])
    all_time_rollup = _aggregate_rollup(list(latest_snapshots.values()))

    # Build output rows
    generated_utc = now_utc.isoformat()
    metadata_rows: list[tuple[str, str]] = [
        ("report_section_01", "Report Metadata"),
        ("day_utc", day),
        ("requested_day", requested_day),
        ("resolved_day", day),
        ("day_fallback_applied", str(requested_day != day).lower()),
        ("generated_utc", generated_utc),
        ("db_path", str(db_path)),
    ]
    summary_rows: list[tuple[str, str]] = [
        ("report_section_02", "Current Day"),
        ("combined_decision_total_rows", str(decision_total_rows)),
        ("combined_governance_total_rows", str(governance_total_rows)),
        ("combined_blocked_total", str(blocked_total)),
        ("combined_blocked_rate", f"{blocked_rate:.6f}"),
        ("data_quality_score", f"{data_quality_score:.2f}"),
        ("paper_executed_total", str(paper_executed_total)),
        ("watchdog_restarts", str(watchdog_restarts)),
        ("report_section_03", "Month To Date"),
        ("month_to_date_days_covered", month_rollup["days_covered"]),
        ("month_to_date_decision_total_rows", month_rollup["decision_total_rows"]),
        ("month_to_date_governance_total_rows", month_rollup["governance_total_rows"]),
        ("month_to_date_blocked_total", month_rollup["blocked_total"]),
        ("month_to_date_paper_executed_total", month_rollup["paper_executed_total"]),
        ("month_to_date_watchdog_restarts", month_rollup["watchdog_restarts"]),
        ("month_to_date_avg_data_quality_score", month_rollup["avg_data_quality_score"]),
        ("report_section_04", "All Time"),
        ("all_time_days_covered", all_time_rollup["days_covered"]),
        ("all_time_decision_total_rows", all_time_rollup["decision_total_rows"]),
        ("all_time_governance_total_rows", all_time_rollup["governance_total_rows"]),
        ("all_time_blocked_total", all_time_rollup["blocked_total"]),
        ("all_time_paper_executed_total", all_time_rollup["paper_executed_total"]),
        ("all_time_watchdog_restarts", all_time_rollup["watchdog_restarts"]),
        ("all_time_avg_data_quality_score", all_time_rollup["avg_data_quality_score"]),
    ]
    detail_rows: list[tuple[str, str]] = [
        ("report_section_05", "Detailed Metrics"),
        ("linked_source_files_total", str(linked_source_files_total)),
        ("decision_source_files", str(decision_source_files)),
        ("governance_source_files", str(governance_source_files)),
        ("combined_action_buy", str(action_counts.get("BUY", 0))),
        ("combined_action_sell", str(action_counts.get("SELL", 0))),
        ("combined_action_hold", str(action_counts.get("HOLD", 0))),
        ("stocks_decision_rows", str(stocks_decision_rows)),
        ("stocks_action_buy", str(stocks_actions.get("BUY", 0))),
        ("stocks_action_sell", str(stocks_actions.get("SELL", 0))),
        ("stocks_action_hold", str(stocks_actions.get("HOLD", 0))),
        ("crypto_decision_rows", str(crypto_decision_rows)),
        ("crypto_action_buy", str(crypto_actions.get("BUY", 0))),
        ("crypto_action_sell", str(crypto_actions.get("SELL", 0))),
        ("crypto_action_hold", str(crypto_actions.get("HOLD", 0))),
        ("futures_decision_rows", str(futures_decision_rows)),
        ("futures_action_buy", str(futures_actions.get("BUY", 0))),
        ("futures_action_sell", str(futures_actions.get("SELL", 0))),
        ("futures_action_hold", str(futures_actions.get("HOLD", 0))),
        ("stocks_pnl_proxy", f"{stocks_pnl_proxy:.6f}"),
        ("crypto_pnl_proxy", f"{crypto_pnl_proxy:.6f}"),
        ("futures_pnl_proxy", f"{futures_pnl_proxy:.6f}"),
        ("futures_strategy_rows", str(futures_strategy_rows)),
        ("futures_strategy_none_rows", str(futures_none_rows)),
        ("options_decision_rows", str(options_decision_rows)),
        ("options_none_rows", str(options_none_rows)),
        ("options_contracts_total", f"{options_contracts_total:.2f}"),
        ("options_master_action_buy", str(options_master_actions.get("BUY", 0))),
        ("options_master_action_sell", str(options_master_actions.get("SELL", 0))),
        ("options_master_action_hold", str(options_master_actions.get("HOLD", 0))),
        ("options_specialist_active_rows", str(options_specialist_active_rows)),
        ("futures_specialist_active_rows", str(futures_specialist_active_rows)),
        ("futures_governance_rows", str(futures_governance_rows)),
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
        ("watchdog_throttled", str(watchdog_throttled)),
        ("watchdog_restart_errors", str(watchdog_restart_errors)),
        ("decision_last_age_sec", str(decision_last_age_sec)),
        ("governance_last_age_sec", str(governance_last_age_sec)),
        ("heartbeat_recent_count", str(heartbeat_recent)),
        ("bot_stack_overall_status", bot_stack_status),
        ("bot_stack_active_sub_bots", str(bot_stack_active_sub_bots)),
        ("bot_stack_watchdog_schwab_live", str(bot_stack_watchdog_schwab_live).lower()),
        ("bot_stack_watchdog_coinbase_live", str(bot_stack_watchdog_coinbase_live).lower()),
        ("paper_executed_crypto", str(paper_executed_crypto)),
        ("guardrail_master_latency_slo_fail", str(guardrail_master_latency_slo_fail)),
        ("guardrail_feature_freshness_fail", str(guardrail_feature_freshness_fail)),
        ("guardrail_canary_mentions", str(guardrail_canary_mentions)),
        ("guardrail_event_lock_hits", str(guardrail_event_lock_hits)),
        ("guardrail_preopen_replay_rows_24h", str(preopen_replay_rows_24h)),
        ("guardrail_preopen_replay_fail_24h", str(preopen_replay_fail_24h)),
        ("ops_storage_mode", storage_mode),
        ("ops_storage_logs_target", storage_logs_target),
        ("ops_sql_link_ok", str(sql_link_ok).lower()),
        ("ops_sql_link_rc", str(sql_link_rc)),
        ("ops_sql_link_db_size_gb", f"{sql_link_db_size_gb:.3f}"),
        ("ops_hot_retention_ran", str(hot_retention_ran).lower()),
        ("ops_hot_retention_rc", str(hot_retention_rc)),
        ("ops_hot_retention_db_size_gb_after", f"{hot_retention_db_after:.3f}"),
        ("ops_canary_enabled", str(canary_enabled).lower()),
        ("ops_canary_weight", f"{canary_weight:.6f}"),
    ]
    rows: list[tuple[str, str]] = metadata_rows + summary_rows + detail_rows

    for k, v in alerts.items():
        rows.append((k, str(v).lower()))

    for i, (sym, cnt) in enumerate(stocks_top[:5], start=1):
        rows.append((f"stocks_top_symbol_{i}", f"{sym}:{cnt}"))
    for i in range(len(stocks_top[:5]) + 1, 6):
        rows.append((f"stocks_top_symbol_{i}", "n/a"))

    for i, (sym, cnt) in enumerate(crypto_top[:5], start=1):
        rows.append((f"crypto_top_symbol_{i}", f"{sym}:{cnt}"))
    for i in range(len(crypto_top[:5]) + 1, 6):
        rows.append((f"crypto_top_symbol_{i}", "n/a"))
    for i, (sym, cnt) in enumerate(futures_top[:5], start=1):
        rows.append((f"futures_top_symbol_{i}", f"{sym}:{cnt}"))
    for i in range(len(futures_top[:5]) + 1, 6):
        rows.append((f"futures_top_symbol_{i}", "n/a"))
    for i, (style, cnt) in enumerate(futures_active_styles[:5], start=1):
        rows.append((f"futures_style_{i}", f"{style}:{cnt}"))
    for i in range(len(futures_active_styles[:5]) + 1, 6):
        rows.append((f"futures_style_{i}", "n/a"))
    for i, (style, cnt) in enumerate(options_active_styles[:5], start=1):
        rows.append((f"options_style_{i}", f"{style}:{cnt}"))
    for i in range(len(options_active_styles[:5]) + 1, 6):
        rows.append((f"options_style_{i}", "n/a"))

    for i, (bot_id, pnl) in enumerate(pnl_strategy_rows[:5], start=1):
        rows.append((f"pnl_strategy_{i}", f"{bot_id}:{_safe_float(pnl):.6f}"))
    for i in range(len(pnl_strategy_rows[:5]) + 1, 6):
        rows.append((f"pnl_strategy_{i}", "n/a"))
    for i, (bot_id, cnt) in enumerate(paper_bot_rows[:5], start=1):
        rows.append((f"paper_bot_{i}", f"{bot_id}:{_safe_int(cnt)}"))
    for i in range(len(paper_bot_rows[:5]) + 1, 6):
        rows.append((f"paper_bot_{i}", "n/a"))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"one_numbers_{day}_{stamp}.csv"
    md_path = out_dir / f"one_numbers_{day}_{stamp}.md"
    xlsx_path = out_dir / f"one_numbers_{day}_{stamp}.xlsx"

    _write_kv_csv(csv_path, rows)
    _write_one_numbers_xlsx(xlsx_path, rows)

    stocks_top_md = ", ".join(f"{sym}:{cnt}" for sym, cnt in stocks_top[:5]) if stocks_top else "n/a"
    crypto_top_md = ", ".join(f"{sym}:{cnt}" for sym, cnt in crypto_top[:5]) if crypto_top else "n/a"
    futures_top_md = ", ".join(f"{sym}:{cnt}" for sym, cnt in futures_top[:5]) if futures_top else "n/a"
    futures_styles_md = ", ".join(f"{style}:{cnt}" for style, cnt in futures_active_styles[:5]) if futures_active_styles else "n/a"
    options_styles_md = ", ".join(f"{style}:{cnt}" for style, cnt in options_active_styles[:5]) if options_active_styles else "n/a"

    md_lines = [
        f"# One Numbers Report ({day})",
        "",
        f"Generated: {generated_utc}",
        f"Requested day: {requested_day}",
        f"Resolved day: {day}",
        "",
        "## Combined",
        f"- Decisions: {decision_total_rows}",
        f"- Actions: BUY={action_counts.get('BUY',0)}, SELL={action_counts.get('SELL',0)}, HOLD={action_counts.get('HOLD',0)}",
        f"- Blocked: {blocked_total} ({_fmt_pct(blocked_rate)})",
        f"- Data quality score: {data_quality_score:.2f}/100",
        "",
        "## Month To Date",
        f"- Days covered: {month_rollup['days_covered']}",
        f"- Decisions: {month_rollup['decision_total_rows']}",
        f"- Governance rows: {month_rollup['governance_total_rows']}",
        f"- Blocked total: {month_rollup['blocked_total']}",
        f"- Paper executions: {month_rollup['paper_executed_total']}",
        f"- Watchdog restarts: {month_rollup['watchdog_restarts']}",
        f"- Avg data quality score: {month_rollup['avg_data_quality_score']}/100",
        "",
        "## All Time",
        f"- Days covered: {all_time_rollup['days_covered']}",
        f"- Decisions: {all_time_rollup['decision_total_rows']}",
        f"- Governance rows: {all_time_rollup['governance_total_rows']}",
        f"- Blocked total: {all_time_rollup['blocked_total']}",
        f"- Paper executions: {all_time_rollup['paper_executed_total']}",
        f"- Watchdog restarts: {all_time_rollup['watchdog_restarts']}",
        f"- Avg data quality score: {all_time_rollup['avg_data_quality_score']}/100",
        "",
        "## Stocks",
        f"- Rows: {stocks_decision_rows}",
        f"- Actions: BUY={stocks_actions.get('BUY',0)}, SELL={stocks_actions.get('SELL',0)}, HOLD={stocks_actions.get('HOLD',0)}",
        f"- PnL proxy: {stocks_pnl_proxy:.6f}",
        f"- Top symbols: {stocks_top_md}",
        "",
        "## Crypto",
        f"- Rows: {crypto_decision_rows}",
        f"- Actions: BUY={crypto_actions.get('BUY',0)}, SELL={crypto_actions.get('SELL',0)}, HOLD={crypto_actions.get('HOLD',0)}",
        f"- PnL proxy: {crypto_pnl_proxy:.6f}",
        f"- Top symbols: {crypto_top_md}",
        "",
        "## Futures",
        f"- Rows: {futures_decision_rows}",
        f"- Actions: BUY={futures_actions.get('BUY',0)}, SELL={futures_actions.get('SELL',0)}, HOLD={futures_actions.get('HOLD',0)}",
        f"- PnL proxy: {futures_pnl_proxy:.6f}",
        f"- Strategy decisions (style!=NONE): {futures_strategy_rows}",
        f"- Strategy decisions NONE: {futures_none_rows}",
        f"- Top symbols: {futures_top_md}",
        f"- Active futures styles: {futures_styles_md}",
        f"- Governance rows tagged futures: {futures_governance_rows}",
        "",
        "## Options",
        f"- Strategy decisions (style!=NONE): {options_decision_rows}",
        f"- Strategy decisions NONE: {options_none_rows}",
        f"- Total contracts: {options_contracts_total:.2f}",
        f"- Options master actions: BUY={options_master_actions.get('BUY',0)}, SELL={options_master_actions.get('SELL',0)}, HOLD={options_master_actions.get('HOLD',0)}",
        f"- Active options styles: {options_styles_md}",
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
        "## Paper + Guardrails",
        f"- Paper executed (total/crypto): {paper_executed_total}/{paper_executed_crypto}",
        f"- Guardrail hits: latency_slo_fail={guardrail_master_latency_slo_fail}, feature_freshness_fail={guardrail_feature_freshness_fail}, canary_mentions={guardrail_canary_mentions}, event_lock_hits={guardrail_event_lock_hits}",
        f"- Preopen replay sanity (24h rows/failures): {preopen_replay_rows_24h}/{preopen_replay_fail_24h}",
        "",
        "## Ops/Storage",
        f"- Storage mode: {storage_mode}",
        f"- Logs target: {storage_logs_target}",
        f"- SQL link service: ok={str(sql_link_ok).lower()} rc={sql_link_rc} db_size_gb={sql_link_db_size_gb:.3f}",
        f"- Hot retention: ran={str(hot_retention_ran).lower()} rc={hot_retention_rc} db_after_gb={hot_retention_db_after:.3f}",
        f"- Canary state: enabled={str(canary_enabled).lower()} weight={canary_weight:.4f}",
        "",
        "## Alerts",
    ]
    md_lines.extend([f"- {k}: {str(v).lower()}" for k, v in alerts.items()])
    md_lines.append("")
    md_lines.append(f"CSV: `{csv_path}`")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    latest_csv = out_dir / "latest.csv"
    latest_md = out_dir / "latest.md"
    latest_xlsx = out_dir / "latest.xlsx"
    latest_json = out_dir / "one_numbers_summary.json"
    health_latest_json = PROJECT_ROOT / "governance" / "health" / "one_numbers_latest.json"
    legacy_latest_dir = out_dir / "latest"
    legacy_latest_json = legacy_latest_dir / "one_numbers_summary.json"

    metric_map = {k: v for k, v in rows}
    summary_payload = {
        "generated_utc": generated_utc,
        "day_utc": day,
        "requested_day": requested_day,
        "resolved_day": day,
        "day_fallback_applied": requested_day != day,
        **metric_map,
    }
    payload_text = json.dumps(summary_payload, ensure_ascii=True, indent=2)
    latest_json.write_text(payload_text, encoding="utf-8")

    health_latest_json.parent.mkdir(parents=True, exist_ok=True)
    health_latest_json.write_text(payload_text, encoding="utf-8")

    legacy_latest_dir.mkdir(parents=True, exist_ok=True)
    legacy_latest_json.write_text(payload_text, encoding="utf-8")

    if latest_csv.exists() or latest_csv.is_symlink():
        latest_csv.unlink()
    if latest_md.exists() or latest_md.is_symlink():
        latest_md.unlink()
    if latest_xlsx.exists() or latest_xlsx.is_symlink():
        latest_xlsx.unlink()
    latest_csv.symlink_to(csv_path)
    latest_md.symlink_to(md_path)
    latest_xlsx.symlink_to(xlsx_path)

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
    print(f"Wrote: {xlsx_path}")
    print(f"Latest CSV: {latest_csv}")
    print(f"Latest MD: {latest_md}")
    print(f"Latest XLSX: {latest_xlsx}")
    print(f"Latest JSON: {latest_json}")
    if not args.no_sql_write:
        print("Registered snapshot in SQLite table: one_numbers_snapshots")

    try:
        if lock_fh is not None:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
            lock_fh.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
