#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import html
import json
import os
import shutil
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HEALTH_DIR = PROJECT_ROOT / "governance" / "health"
WATCHDOG_DIR = PROJECT_ROOT / "governance" / "watchdog"
ALERTS_DIR = PROJECT_ROOT / "governance" / "alerts"
DEFAULT_OUT_DIR = PROJECT_ROOT / "exports" / "reports" / "crash_reports"


def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _run(cmd: List[str]) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()
    except Exception as exc:
        return 1, "", str(exc)


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8") as fh:  # type: ignore[arg-type]
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield obj
    except Exception:
        return


def _parse_ts(raw: Any) -> datetime | None:
    if not raw:
        return None
    txt = str(raw).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(txt)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _fmt_ts_local(raw: Any) -> str:
    dt = _parse_ts(raw)
    if dt is None:
        return ""
    return dt.astimezone().isoformat(timespec="seconds")


def _fmt_num(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value or "")


def _title_token(value: Any) -> str:
    txt = str(value or "").strip().replace("-", "_")
    if not txt:
        return "Unknown"
    return " ".join(part for part in txt.replace("_", " " ).title().split())


def _critical_alert_label(event: Any) -> str:
    token = str(event or "").strip()
    labels = {
        "lane_kill_switch_engaged": "Lane Kill Switch",
        "options_margin_guard": "Options Margin Guard",
        "futures_margin_guard": "Futures Margin Guard",
        "critical_alert": "Critical Alert",
    }
    return labels.get(token, _title_token(token))


def _critical_alert_detail(row: Dict[str, Any]) -> str:
    details = row.get("details") if isinstance(row.get("details"), dict) else {}
    parts: List[str] = []
    message = str(row.get("message") or "").strip()
    if message:
        parts.append(message)
    symbol = str(details.get("symbol") or "").strip()
    if symbol:
        parts.append(f"symbol={symbol}")
    lane = str(details.get("lane") or "").strip()
    if lane and f"lane={lane}" not in message:
        parts.append(f"lane={lane}")
    source = str(details.get("source") or "").strip()
    if source:
        parts.append(f"source={source}")
    reasons = details.get("reasons") if isinstance(details.get("reasons"), list) else []
    if reasons:
        reason_text = ",".join(str(item) for item in reasons if str(item).strip())
        if reason_text:
            parts.append(f"reasons={reason_text}")
    if "required_margin_proxy" in details:
        parts.append(f"required_margin_proxy={_fmt_num(details.get('required_margin_proxy'))}")
    if "available_margin_proxy" in details:
        parts.append(f"available_margin_proxy={_fmt_num(details.get('available_margin_proxy'))}")
    return " ".join(part for part in parts if part).strip()


def _incident_class_label(name: str) -> str:
    labels = {
        "bot_limit_guardrail": "Bot Limit / Guardrail",
        "crash_restart": "Crash / Restart",
        "operational_alert": "Operational Alert",
    }
    return labels.get(name, _title_token(name))


def _pdf_renderer_binary(allow_gui_renderer: bool) -> tuple[str, str]:
    env_override = os.getenv("CRASH_REPORT_PDF_BIN", "").strip() or os.getenv("PROJECT_TIMELINE_PDF_BIN", "").strip()
    if env_override:
        env_bin = Path(env_override).expanduser()
        if env_bin.exists():
            kind = "wkhtmltopdf" if env_bin.name == "wkhtmltopdf" else "browser"
            return str(env_bin), kind

    wkhtmltopdf = shutil.which("wkhtmltopdf")
    if wkhtmltopdf:
        return wkhtmltopdf, "wkhtmltopdf"

    for candidate in (
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        shutil.which("google-chrome"),
        shutil.which("google-chrome-stable"),
        shutil.which("microsoft-edge"),
        shutil.which("msedge"),
    ):
        if candidate:
            return candidate, "browser"

    if allow_gui_renderer:
        for candidate in (
            Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
            Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
            Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
        ):
            if candidate.exists():
                return str(candidate), "browser"

    return "", ""


def _render_pdf_from_html(html_path: Path, pdf_path: Path, *, allow_gui_renderer: bool) -> tuple[bool, str]:
    renderer, renderer_kind = _pdf_renderer_binary(allow_gui_renderer=allow_gui_renderer)
    if not renderer:
        return False, "pdf_renderer_not_found"
    html_uri = html_path.resolve().as_uri()
    if renderer_kind == "wkhtmltopdf":
        cmd = [renderer, html_uri, str(pdf_path)]
    else:
        cmd = [
            renderer,
            "--headless",
            "--disable-gpu",
            f"--print-to-pdf={pdf_path}",
            html_uri,
        ]
    rc, out, err = _run(cmd)
    if rc == 0 and pdf_path.exists() and pdf_path.stat().st_size > 0:
        return True, out or "ok"
    return False, err or out or f"rc={rc}"


def _source_inventory() -> List[Path]:
    rows: List[Path] = []
    for candidate in (
        HEALTH_DIR / "process_watchdog_latest.json",
        HEALTH_DIR / "shadow_watchdog_tripwire_latest.json",
        HEALTH_DIR / "shadow_watchdog_halt_recovery_latest.json",
        HEALTH_DIR / "operator_control_latest.json",
        HEALTH_DIR / "incident_auto_halt_state.json",
        ALERTS_DIR / "incident_auto_halt_latest.json",
        WATCHDOG_DIR / "shadow_watchdog_tripwire_events.jsonl",
        WATCHDOG_DIR / "shadow_watchdog_halt_recovery_events.jsonl",
        WATCHDOG_DIR / "incident_auto_halt_events.jsonl",
        WATCHDOG_DIR / "incident_auto_halt_events.jsonl.gz",
    ):
        if candidate.exists():
            rows.append(candidate)
    rows.extend(sorted(WATCHDOG_DIR.glob("watchdog_events_*.jsonl")))
    rows.extend(sorted(ALERTS_DIR.glob("critical_events_*.jsonl")))
    rows.extend(sorted(ALERTS_DIR.glob("critical_latest_*.json")))
    return rows


def _collect_watchdog_rows(since_utc: datetime) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(WATCHDOG_DIR.glob("watchdog_events_*.jsonl")):
        for obj in _iter_jsonl(path):
            ts = _parse_ts(obj.get("timestamp_utc"))
            if ts is None or ts < since_utc:
                continue
            rows.append(obj)
    rows.sort(key=lambda row: str(row.get("timestamp_utc") or ""))
    return rows


def _collect_tripwire_rows(since_utc: datetime) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for obj in _iter_jsonl(WATCHDOG_DIR / "shadow_watchdog_tripwire_events.jsonl"):
        ts = _parse_ts(obj.get("timestamp_utc"))
        if ts is None or ts < since_utc:
            continue
        rows.append(obj)
    rows.sort(key=lambda row: str(row.get("timestamp_utc") or ""))
    return rows


def _collect_halt_recovery_rows(since_utc: datetime) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for obj in _iter_jsonl(WATCHDOG_DIR / "shadow_watchdog_halt_recovery_events.jsonl"):
        ts = _parse_ts(obj.get("timestamp_utc"))
        if ts is None or ts < since_utc:
            continue
        rows.append(obj)
    rows.sort(key=lambda row: str(row.get("timestamp_utc") or ""))
    return rows


def _collect_incident_auto_halt_rows(since_utc: datetime) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in (
        WATCHDOG_DIR / "incident_auto_halt_events.jsonl",
        WATCHDOG_DIR / "incident_auto_halt_events.jsonl.gz",
    ):
        if not path.exists():
            continue
        for obj in _iter_jsonl(path):
            ts = _parse_ts(obj.get("timestamp_utc"))
            if ts is None or ts < since_utc:
                continue
            rows.append(obj)
    rows.sort(key=lambda row: str(row.get("timestamp_utc") or ""))
    return rows


def _collect_critical_alert_rows(since_utc: datetime) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str, str, str, str]] = set()

    def add_row(obj: Dict[str, Any]) -> None:
        ts = _parse_ts(obj.get("timestamp_utc"))
        if ts is None or ts < since_utc:
            return
        details = obj.get("details") if isinstance(obj.get("details"), dict) else {}
        dedupe_key = (
            ts.isoformat(),
            str(obj.get("profile") or ""),
            str(obj.get("broker") or ""),
            str(obj.get("event") or ""),
            str(obj.get("message") or ""),
            json.dumps(details, sort_keys=True, separators=(",", ":")),
        )
        if dedupe_key in seen:
            return
        seen.add(dedupe_key)
        rows.append(obj)

    for path in sorted(ALERTS_DIR.glob("critical_events_*.jsonl")):
        for obj in _iter_jsonl(path):
            add_row(obj)
    for path in sorted(ALERTS_DIR.glob("critical_latest_*.json")):
        payload = _load_json(path)
        if payload:
            add_row(payload)

    rows.sort(key=lambda row: str(row.get("timestamp_utc") or ""))
    return rows


def _watchdog_action_summary(rows: List[Dict[str, Any]]) -> tuple[Dict[str, Counter], List[Dict[str, Any]]]:
    counts: Dict[str, Counter] = defaultdict(Counter)
    incidents: List[Dict[str, Any]] = []
    for row in rows:
        ts = row.get("timestamp_utc")
        halt = row.get("global_halt_recovery") if isinstance(row.get("global_halt_recovery"), dict) else {}
        for target in row.get("targets", []) or []:
            if not isinstance(target, dict):
                continue
            name = str(target.get("name") or "unknown")
            action = str(target.get("action") or "none")
            note = str(target.get("note") or "")
            if action != "none":
                counts[name][action] += 1
            if action != "none" or "global_halt_active" in note or "restart_rate_limit" in note or "restart_attempted" in note:
                incidents.append(
                    {
                        "timestamp_utc": ts,
                        "source": "watchdog",
                        "target": name,
                        "event": action,
                        "detail": note,
                        "halt_reason": str(halt.get("decision_reason") or ""),
                    }
                )
    incidents.sort(key=lambda row: str(row.get("timestamp_utc") or ""), reverse=True)
    return counts, incidents


def _tripwire_summary(rows: List[Dict[str, Any]]) -> tuple[Dict[str, Counter], List[Dict[str, Any]]]:
    counts: Dict[str, Counter] = defaultdict(Counter)
    incidents: List[Dict[str, Any]] = []
    for row in rows:
        target = str(row.get("target") or "unknown")
        event = str(row.get("event") or "")
        if not event:
            continue
        counts[target][event] += 1
        incidents.append(
            {
                "timestamp_utc": row.get("timestamp_utc"),
                "source": "tripwire",
                "target": target,
                "event": event,
                "detail": str(row.get("note") or ""),
            }
        )
    incidents.sort(key=lambda row: str(row.get("timestamp_utc") or ""), reverse=True)
    return counts, incidents


def _normalize_halt_decision_reason(raw: Any) -> str:
    txt = str(raw or "").strip()
    if txt.startswith("cooldown_not_elapsed:"):
        return "cooldown_not_elapsed"
    if txt.startswith("reason_not_allowed:"):
        suffix = txt.split(":", 1)[1].strip() or "unknown"
        return f"reason_not_allowed:{suffix}"
    if txt.startswith("malformed_payload_eligible:"):
        suffix = txt.split(":", 1)[1].strip() or "unknown"
        return f"malformed_payload_eligible:{suffix}"
    return txt or "unknown"


def _infer_halt_root_cause(first_row: Dict[str, Any], *, auto_halt_rows: List[Dict[str, Any]]) -> tuple[str, str]:
    decision_reason = str(first_row.get("decision_reason") or "").strip()
    normalized_decision = _normalize_halt_decision_reason(decision_reason)
    halt_reason = str(first_row.get("halt_reason") or "").strip()
    payload_error = str(first_row.get("halt_payload_error") or "").strip()
    start_ts = _parse_ts(first_row.get("timestamp_utc"))

    if start_ts is not None:
        nearest_failed: tuple[float, List[str]] | None = None
        for row in auto_halt_rows:
            row_ts = _parse_ts(row.get("timestamp_utc"))
            if row_ts is None:
                continue
            delta_seconds = (start_ts - row_ts).total_seconds()
            if delta_seconds < -15.0 or delta_seconds > 120.0:
                continue
            failed_checks = [str(x) for x in (row.get("failed_checks") or []) if str(x)] if isinstance(row.get("failed_checks"), list) else []
            if not failed_checks and not bool(row.get("halt", False)):
                continue
            if nearest_failed is None or delta_seconds < nearest_failed[0]:
                nearest_failed = (delta_seconds, failed_checks)
        if nearest_failed is not None:
            delta_seconds, failed_checks = nearest_failed
            label = ",".join(failed_checks) if failed_checks else "incident_auto_halt"
            return (
                f"incident_auto_halt:{label}",
                f"Incident auto-halt checks fired {delta_seconds:.0f}s before the halt: {label}.",
            )

    if halt_reason:
        if normalized_decision == "cooldown_not_elapsed":
            return halt_reason, f"Global halt reason {halt_reason} was active and still inside cooldown."
        if normalized_decision.startswith("reason_not_allowed:"):
            return halt_reason, f"Global halt reason {halt_reason} was not eligible for auto-clear under the configured allowlist."
        return halt_reason, f"Global halt reason {halt_reason}."

    if normalized_decision == "reason_not_allowed:unknown":
        reason_hint = payload_error or "missing_reason"
        return (
            "missing_halt_reason",
            "Global halt payload had no usable reason, so the watchdog classified it as reason_not_allowed:unknown and would not auto-clear it"
            f" ({reason_hint}).",
        )
    if normalized_decision.startswith("malformed_payload_eligible"):
        reason_hint = payload_error or normalized_decision.split(":", 1)[1]
        return "malformed_halt_payload", f"Malformed global halt payload was eligible for auto-clear ({reason_hint})."
    if normalized_decision == "cooldown_not_elapsed":
        return "global_halt_cooldown", "Global halt was active and still inside cooldown."
    if normalized_decision.startswith("reason_not_allowed:"):
        blocked_reason = normalized_decision.split(":", 1)[1].strip() or "unknown"
        return (
            f"blocked_halt_reason:{blocked_reason}",
            f"Global halt reason {blocked_reason} was not eligible for auto-clear.",
        )
    return "global_halt", f"Global halt state persisted with decision {decision_reason or 'unknown'}."


def _summarize_halt_root_causes(halt_windows: List[Dict[str, Any]]) -> tuple[Counter, Dict[str, str]]:
    counts: Counter = Counter()
    examples: Dict[str, str] = {}
    for row in halt_windows:
        root_cause = str(row.get("root_cause") or "global_halt")
        counts[root_cause] += 1
        if root_cause not in examples:
            examples[root_cause] = str(row.get("root_detail") or row.get("detail") or "")
    return counts, examples


def _halt_recovery_windows(rows: List[Dict[str, Any]], *, auto_halt_rows: List[Dict[str, Any]]) -> tuple[Counter, List[Dict[str, Any]]]:
    counts: Counter = Counter()
    windows: List[Dict[str, Any]] = []
    active_rows = [
        row for row in rows
        if bool(row.get("halt_active", False)) or str(row.get("action") or "") != "halt_not_set"
    ]
    active_rows.sort(key=lambda row: str(row.get("timestamp_utc") or ""))
    current: Dict[str, Any] | None = None
    for row in active_rows:
        raw_decision = str(row.get("decision_reason") or "")
        normalized_decision = _normalize_halt_decision_reason(raw_decision)
        key = (
            str(row.get("action") or ""),
            normalized_decision,
            str(row.get("halt_reason") or ""),
        )
        counts[f"{key[0]}|{key[1]}"] += 1
        ts = row.get("timestamp_utc")
        if current and current["key"] == key:
            current["last_timestamp_utc"] = ts
            current["last_decision_reason"] = raw_decision
            current["count"] += 1
            continue
        if current:
            root_cause, root_detail = _infer_halt_root_cause(current["first_row"], auto_halt_rows=auto_halt_rows)
            detail_parts = [
                f"root_cause={root_cause}",
                root_detail,
                f"decision={current['normalized_decision_reason'] or 'unknown'}",
                f"halt_reason={current['halt_reason'] or 'n/a'}",
            ]
            current["root_cause"] = root_cause
            current["root_detail"] = root_detail
            current["detail"] = " ".join(part for part in detail_parts if part)
            current.pop("first_row", None)
            current.pop("key", None)
            current.pop("last_decision_reason", None)
            windows.append(current)
        current = {
            "key": key,
            "first_row": row,
            "timestamp_utc": ts,
            "last_timestamp_utc": ts,
            "source": "halt_recovery",
            "target": "global_halt",
            "event": key[0] or "unknown",
            "decision_reason": raw_decision,
            "normalized_decision_reason": normalized_decision,
            "halt_reason": key[2],
            "detail": "",
            "count": 1,
            "last_decision_reason": raw_decision,
        }
    if current:
        root_cause, root_detail = _infer_halt_root_cause(current["first_row"], auto_halt_rows=auto_halt_rows)
        detail_parts = [
            f"root_cause={root_cause}",
            root_detail,
            f"decision={current['normalized_decision_reason'] or 'unknown'}",
            f"halt_reason={current['halt_reason'] or 'n/a'}",
        ]
        current["root_cause"] = root_cause
        current["root_detail"] = root_detail
        current["detail"] = " ".join(part for part in detail_parts if part)
        current.pop("first_row", None)
        current.pop("key", None)
        current.pop("last_decision_reason", None)
        windows.append(current)
    windows.sort(key=lambda row: str(row.get("timestamp_utc") or ""), reverse=True)
    return counts, windows


def _incident_auto_halt_summary(rows: List[Dict[str, Any]]) -> tuple[Counter, List[Dict[str, Any]]]:
    counts: Counter = Counter()
    incidents: List[Dict[str, Any]] = []
    for row in rows:
        failed = row.get("failed_checks") if isinstance(row.get("failed_checks"), list) else []
        if not failed and not bool(row.get("halt", False)):
            continue
        label = ",".join(str(x) for x in failed) if failed else "halted"
        counts[label] += 1
        incidents.append(
            {
                "timestamp_utc": row.get("timestamp_utc"),
                "source": "incident_auto_halt",
                "target": "quality_gate",
                "event": str(row.get("event") or "state_update"),
                "detail": label,
            }
        )
    incidents.sort(key=lambda row: str(row.get("timestamp_utc") or ""), reverse=True)
    return counts, incidents


def _critical_alert_summary(rows: List[Dict[str, Any]]) -> tuple[Counter, List[Dict[str, Any]]]:
    counts: Counter = Counter()
    incidents: List[Dict[str, Any]] = []
    for row in rows:
        event_key = str(row.get("event") or "").strip()
        label = _critical_alert_label(event_key)
        target = f"{_title_token(row.get('profile'))} / {_title_token(row.get('broker'))}"
        counts[f"{target}|{label}"] += 1
        incidents.append(
            {
                "timestamp_utc": row.get("timestamp_utc"),
                "source": "critical_alert",
                "target": target,
                "event": label,
                "event_key": event_key,
                "detail": _critical_alert_detail(row),
            }
        )
    incidents.sort(key=lambda row: str(row.get("timestamp_utc") or ""), reverse=True)
    return counts, incidents


def _incident_class(row: Dict[str, Any]) -> str:
    source = str(row.get("source") or "")
    event_key = str(row.get("event_key") or row.get("event") or "").strip()
    event = event_key.lower()
    detail = " ".join(str(row.get(key) or "") for key in ("detail", "halt_reason")).lower()

    if source in {"incident_auto_halt", "halt_recovery"}:
        return "bot_limit_guardrail"
    if source == "critical_alert":
        if event_key in {"lane_kill_switch_engaged", "options_margin_guard", "futures_margin_guard"}:
            return "bot_limit_guardrail"
        if event.startswith("lane_") or event.endswith("_guard") or event.endswith("_pause") or "kill_switch" in event or "margin_guard" in event:
            return "bot_limit_guardrail"
        return "operational_alert"
    if source == "watchdog":
        if "global_halt_active" in detail or event.startswith("halt"):
            return "bot_limit_guardrail"
        return "crash_restart"
    if source == "tripwire":
        return "crash_restart"
    return "crash_restart"


def _partition_incidents_by_class(
    incidents: List[Dict[str, Any]], recent_limit: int
) -> tuple[Counter, List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    limit = max(recent_limit, 1)
    class_counts: Counter = Counter()
    recent_limit_incidents: List[Dict[str, Any]] = []
    recent_crash_incidents: List[Dict[str, Any]] = []
    recent_other_incidents: List[Dict[str, Any]] = []

    for row in incidents:
        tagged = dict(row)
        incident_class = _incident_class(tagged)
        tagged["incident_class"] = incident_class
        class_counts[incident_class] += 1
        if incident_class == "bot_limit_guardrail":
            if len(recent_limit_incidents) < limit:
                recent_limit_incidents.append(tagged)
            continue
        if incident_class == "crash_restart":
            if len(recent_crash_incidents) < limit:
                recent_crash_incidents.append(tagged)
            continue
        if len(recent_other_incidents) < limit:
            recent_other_incidents.append(tagged)

    return class_counts, recent_limit_incidents, recent_crash_incidents, recent_other_incidents


def _process_status_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = payload.get("status")
    return rows if isinstance(rows, list) else []


def _summary_points(current: Dict[str, Any]) -> List[str]:
    points: List[str] = []
    process_watchdog = current.get("process_watchdog") if isinstance(current.get("process_watchdog"), dict) else {}
    halt_recovery = current.get("halt_recovery") if isinstance(current.get("halt_recovery"), dict) else {}
    tripwire = current.get("tripwire") if isinstance(current.get("tripwire"), dict) else {}
    operator = current.get("operator_control") if isinstance(current.get("operator_control"), dict) else {}

    status_rows = _process_status_rows(process_watchdog)
    all_sleeves = next((row for row in status_rows if str(row.get("name")) == "all_sleeves"), {})
    coinbase_loop = next((row for row in status_rows if str(row.get("name")) == "coinbase_loop"), {})

    if operator:
        action = str(operator.get("global_action") or "none")
        if action != "none":
            points.append(
                f"Operator control latest action: {action} at {operator.get('timestamp_utc', '')}."
            )
    points.append(
        "Global halt active: {halt_active}; market_data_only: {mdo}; allow_order_execution: {aoe}.".format(
            halt_active=_fmt_num(bool(halt_recovery.get("halt_active", False))),
            mdo=_fmt_num(bool(halt_recovery.get("market_data_only", False))),
            aoe=_fmt_num(bool(halt_recovery.get("allow_order_execution", False))),
        )
    )
    points.append(
        "Tripwire active: {active}; active incidents: {count}.".format(
            active=_fmt_num(bool(tripwire.get("active", False))),
            count=len(tripwire.get("active_incidents", []) if isinstance(tripwire.get("active_incidents"), list) else []),
        )
    )
    if all_sleeves:
        points.append(
            "All sleeves heartbeat_ok={hb} alt_running={alt} at {ts}.".format(
                hb=_fmt_num(bool(all_sleeves.get("heartbeat_ok", False))),
                alt=_fmt_num(all_sleeves.get("alt_running", 0)),
                ts=process_watchdog.get("timestamp_utc", ""),
            )
        )
    if coinbase_loop:
        points.append(
            "Coinbase loop running={running} heartbeat_ok={hb}.".format(
                running=_fmt_num(coinbase_loop.get("running", 0)),
                hb=_fmt_num(bool(coinbase_loop.get("heartbeat_ok", False))),
            )
        )
    return points


def _render_markdown(context: Dict[str, Any]) -> str:
    current = context["current"]
    summary_points = context["summary_points"]
    watchdog_counts = context["watchdog_counts"]
    tripwire_counts = context["tripwire_counts"]
    halt_counts = context["halt_counts"]
    halt_root_cause_counts = context["halt_root_cause_counts"]
    halt_root_cause_examples = context["halt_root_cause_examples"]
    auto_halt_counts = context["auto_halt_counts"]
    critical_alert_counts = context["critical_alert_counts"]
    incident_class_counts = context["incident_class_counts"]
    recent_limit_incidents = context["recent_limit_incidents"]
    recent_crash_incidents = context["recent_crash_incidents"]
    recent_other_incidents = context["recent_other_incidents"]
    source_files = context["source_files"]
    status_rows = _process_status_rows(current["process_watchdog"])

    lines: List[str] = [
        "# Crash Report Digest",
        "",
        f"- Generated (UTC): `{context['generated_utc']}`",
        f"- Generated (Local): `{context['generated_local']}`",
        f"- Project root: `{PROJECT_ROOT}`",
        f"- Lookback days: `{context['lookback_days']}`",
        "",
        "## Executive Summary",
    ]

    def append_incident_table(title: str, rows: List[Dict[str, Any]], empty_message: str) -> None:
        lines.extend(
            [
                "",
                title,
                "",
                "| Date (Local) | Source | Target | Event | Detail |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for row in rows:
            detail = str(row.get("detail") or "")
            if row.get("count"):
                detail = f"{detail} window_count={row.get('count')}".strip()
            lines.append(
                "| {date_local} | {source} | {target} | {event} | {detail} |".format(
                    date_local=_fmt_ts_local(row.get("timestamp_utc")),
                    source=str(row.get("source") or ""),
                    target=str(row.get("target") or ""),
                    event=str(row.get("event") or ""),
                    detail=detail.replace("|", "/"),
                )
            )
        if not rows:
            lines.append(f"| n/a | n/a | n/a | n/a | {empty_message} |")

    for point in summary_points:
        lines.append(f"- {point}")

    incident_summary_bits = [
        f"{incident_class_counts.get('bot_limit_guardrail', 0)} bot limit/guardrail",
        f"{incident_class_counts.get('crash_restart', 0)} crash/restart",
    ]
    if incident_class_counts.get("operational_alert", 0):
        incident_summary_bits.append(f"{incident_class_counts.get('operational_alert', 0)} operational alert")
    lines.append(f"- Lookback incident classes: {', '.join(incident_summary_bits)}.")

    lines.extend(
        [
            "",
            "## Current Watchdog Status",
            "",
            "| Name | Running | Alt Running | Heartbeat OK | Heartbeat Age Seconds | Restart Reason |",
            "| --- | ---: | ---: | --- | ---: | --- |",
        ]
    )
    for row in status_rows:
        lines.append(
            "| {name} | {running} | {alt} | {hb} | {age} | {reason} |".format(
                name=str(row.get("name") or ""),
                running=_fmt_num(row.get("running", 0)),
                alt=_fmt_num(row.get("alt_running", 0)),
                hb=_fmt_num(bool(row.get("heartbeat_ok", False))),
                age=_fmt_num(row.get("heartbeat_age_seconds", 0.0)),
                reason=str(row.get("restart_reason") or ""),
            )
        )

    lines.extend(
        [
            "",
            "## Watchdog Target Action Counts",
            "",
            "| Target | Restart | Throttled | Halted |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for target in sorted(watchdog_counts):
        counts = watchdog_counts[target]
        lines.append(
            f"| {target} | {counts.get('restart', 0)} | {counts.get('throttled', 0)} | {counts.get('halted', 0)} |"
        )
    if not watchdog_counts:
        lines.append("| none | 0 | 0 | 0 |")

    lines.extend(
        [
            "",
            "## Tripwire Counts",
            "",
            "| Target | Opened | Cleared |",
            "| --- | ---: | ---: |",
        ]
    )
    for target in sorted(tripwire_counts):
        counts = tripwire_counts[target]
        lines.append(
            f"| {target} | {counts.get('loop_tripwire_opened', 0)} | {counts.get('loop_tripwire_cleared', 0)} |"
        )
    if not tripwire_counts:
        lines.append("| none | 0 | 0 |")

    lines.extend(
        [
            "",
            "## Halt Recovery Decisions",
            "",
            "| Action / Decision | Count |",
            "| --- | ---: |",
        ]
    )
    for key, count in halt_counts.most_common():
        action, decision = key.split("|", 1)
        lines.append(f"| `{action}` / `{decision}` | {count} |")
    if not halt_counts:
        lines.append("| none | 0 |")

    lines.extend(
        [
            "",
            "## Inferred Halt Root Causes",
            "",
            "| Root Cause | Windows | Example |",
            "| --- | ---: | --- |",
        ]
    )
    for root_cause, count in halt_root_cause_counts.most_common():
        example = str(halt_root_cause_examples.get(root_cause) or "")
        lines.append(f"| `{root_cause}` | {count} | {example.replace('|', '/')} |")
    if not halt_root_cause_counts:
        lines.append("| none | 0 | No halt windows found. |")

    lines.extend(
        [
            "",
            "## Incident Auto Halt Checks",
            "",
            "| Failed Checks | Count |",
            "| --- | ---: |",
        ]
    )
    for key, count in auto_halt_counts.most_common():
        lines.append(f"| `{key}` | {count} |")
    if not auto_halt_counts:
        lines.append("| none | 0 |")

    lines.extend(
        [
            "",
            "## Critical Alert Counts",
            "",
            "| Target | Event | Count |",
            "| --- | --- | ---: |",
        ]
    )
    for key, count in critical_alert_counts.most_common():
        target, event = key.split("|", 1)
        lines.append(f"| {target} | {event} | {count} |")
    if not critical_alert_counts:
        lines.append("| none | none | 0 |")

    lines.extend(
        [
            "",
            "## Incident Classes",
            "",
            "| Class | Count |",
            "| --- | ---: |",
        ]
    )
    for key, count in incident_class_counts.most_common():
        lines.append(f"| {_incident_class_label(key)} | {count} |")
    if not incident_class_counts:
        lines.append("| none | 0 |")

    append_incident_table(
        "## Recent Bot Limit / Guardrail Events",
        recent_limit_incidents,
        "No bot limit or guardrail rows found.",
    )
    append_incident_table(
        "## Recent Crash / Restart Events",
        recent_crash_incidents,
        "No crash or restart rows found.",
    )
    append_incident_table(
        "## Recent Other Operational Alerts",
        recent_other_incidents,
        "No other operational alerts found.",
    )

    lines.extend(
        [
            "",
            "## Start Commands",
            "",
            "- Full stack restart: `./scripts/ops/opsctl.sh start-live --force-restart`",
            "- Coinbase spot feed only: `./scripts/ops/opsctl.sh coinbase-start --paper --live-data --top-n 5 --min-acc 0.58 --profiles default`",
            "- Coinbase futures feed only: `./scripts/ops/opsctl.sh coinbase-futures-start --paper --live-data --top-n 10 --min-acc 0.56 --profiles crypto_futures`",
            "- Feed tail: `./scripts/ops/opsctl.sh feed --source all --lines 40`",
            "",
            "## Source Files",
            "",
            "| Path | Modified (Local) | Size Bytes |",
            "| --- | --- | ---: |",
        ]
    )
    for path in source_files:
        st = path.stat()
        lines.append(
            f"| `{path.relative_to(PROJECT_ROOT)}` | `{datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).astimezone().isoformat(timespec='seconds')}` | {st.st_size} |"
        )
    return "\n".join(lines) + "\n"


def _render_html(context: Dict[str, Any]) -> str:
    current = context["current"]
    summary_points = context["summary_points"]
    watchdog_counts = context["watchdog_counts"]
    tripwire_counts = context["tripwire_counts"]
    halt_counts = context["halt_counts"]
    halt_root_cause_counts = context["halt_root_cause_counts"]
    halt_root_cause_examples = context["halt_root_cause_examples"]
    auto_halt_counts = context["auto_halt_counts"]
    critical_alert_counts = context["critical_alert_counts"]
    incident_class_counts = context["incident_class_counts"]
    recent_limit_incidents = context["recent_limit_incidents"]
    recent_crash_incidents = context["recent_crash_incidents"]
    recent_other_incidents = context["recent_other_incidents"]
    source_files = context["source_files"]
    status_rows = _process_status_rows(current["process_watchdog"])

    def li(label: str, value: Any) -> str:
        return f"<li><b>{html.escape(label)}:</b> <code>{html.escape(_fmt_num(value))}</code></li>"

    def recent_table_html(rows: List[Dict[str, Any]], empty_message: str) -> str:
        recent_rows: List[str] = []
        for row in rows:
            detail = str(row.get("detail") or "")
            if row.get("count"):
                detail = f"{detail} window_count={row.get('count')}".strip()
            recent_rows.append(
                "<tr>"
                f"<td><code>{html.escape(_fmt_ts_local(row.get('timestamp_utc')))}</code></td>"
                f"<td>{html.escape(str(row.get('source') or ''))}</td>"
                f"<td>{html.escape(str(row.get('target') or ''))}</td>"
                f"<td>{html.escape(str(row.get('event') or ''))}</td>"
                f"<td>{html.escape(detail)}</td>"
                "</tr>"
            )
        return "".join(recent_rows) or f"<tr><td colspan='5'>{html.escape(empty_message)}</td></tr>"

    summary_items = list(summary_points)
    incident_summary_bits = [
        f"{incident_class_counts.get('bot_limit_guardrail', 0)} bot limit/guardrail",
        f"{incident_class_counts.get('crash_restart', 0)} crash/restart",
    ]
    if incident_class_counts.get("operational_alert", 0):
        incident_summary_bits.append(f"{incident_class_counts.get('operational_alert', 0)} operational alert")
    summary_items.append(f"Lookback incident classes: {', '.join(incident_summary_bits)}.")

    summary_html = "".join(f"<li>{html.escape(point)}</li>" for point in summary_items)
    status_html = "".join(
        "<tr>"
        f"<td>{html.escape(str(row.get('name') or ''))}</td>"
        f"<td>{html.escape(_fmt_num(row.get('running', 0)))}</td>"
        f"<td>{html.escape(_fmt_num(row.get('alt_running', 0)))}</td>"
        f"<td>{html.escape(_fmt_num(bool(row.get('heartbeat_ok', False))))}</td>"
        f"<td>{html.escape(_fmt_num(row.get('heartbeat_age_seconds', 0.0)))}</td>"
        f"<td>{html.escape(str(row.get('restart_reason') or ''))}</td>"
        "</tr>"
        for row in status_rows
    ) or "<tr><td colspan='6'>No current watchdog status rows.</td></tr>"

    watchdog_html = "".join(
        "<tr>"
        f"<td>{html.escape(target)}</td>"
        f"<td>{watchdog_counts[target].get('restart', 0)}</td>"
        f"<td>{watchdog_counts[target].get('throttled', 0)}</td>"
        f"<td>{watchdog_counts[target].get('halted', 0)}</td>"
        "</tr>"
        for target in sorted(watchdog_counts)
    ) or "<tr><td colspan='4'>No watchdog action rows.</td></tr>"

    tripwire_html = "".join(
        "<tr>"
        f"<td>{html.escape(target)}</td>"
        f"<td>{tripwire_counts[target].get('loop_tripwire_opened', 0)}</td>"
        f"<td>{tripwire_counts[target].get('loop_tripwire_cleared', 0)}</td>"
        "</tr>"
        for target in sorted(tripwire_counts)
    ) or "<tr><td colspan='3'>No tripwire events.</td></tr>"

    halt_html = "".join(
        "<tr>"
        f"<td><code>{html.escape(key.split('|', 1)[0])}</code> / <code>{html.escape(key.split('|', 1)[1])}</code></td>"
        f"<td>{count}</td>"
        "</tr>"
        for key, count in halt_counts.most_common()
    ) or "<tr><td colspan='2'>No halt recovery incidents.</td></tr>"

    halt_root_cause_html = "".join(
        "<tr>"
        f"<td><code>{html.escape(root_cause)}</code></td>"
        f"<td>{count}</td>"
        f"<td>{html.escape(str(halt_root_cause_examples.get(root_cause) or ''))}</td>"
        "</tr>"
        for root_cause, count in halt_root_cause_counts.most_common()
    ) or "<tr><td colspan='3'>No inferred halt root causes.</td></tr>"

    auto_halt_html = "".join(
        "<tr>"
        f"<td><code>{html.escape(key)}</code></td>"
        f"<td>{count}</td>"
        "</tr>"
        for key, count in auto_halt_counts.most_common()
    ) or "<tr><td colspan='2'>No incident auto-halt rows.</td></tr>"

    critical_alert_html = "".join(
        "<tr>"
        f"<td>{html.escape(key.split('|', 1)[0])}</td>"
        f"<td>{html.escape(key.split('|', 1)[1])}</td>"
        f"<td>{count}</td>"
        "</tr>"
        for key, count in critical_alert_counts.most_common()
    ) or "<tr><td colspan='3'>No critical alert rows.</td></tr>"

    incident_class_html = "".join(
        "<tr>"
        f"<td>{html.escape(_incident_class_label(key))}</td>"
        f"<td>{count}</td>"
        "</tr>"
        for key, count in incident_class_counts.most_common()
    ) or "<tr><td colspan='2'>No incident classifications.</td></tr>"

    recent_limit_html = recent_table_html(recent_limit_incidents, "No bot limit or guardrail rows found.")
    recent_crash_html = recent_table_html(recent_crash_incidents, "No crash or restart rows found.")
    recent_other_html = recent_table_html(recent_other_incidents, "No other operational alerts found.")

    sources_html = "".join(
        "<tr>"
        f"<td><code>{html.escape(str(path.relative_to(PROJECT_ROOT)))}</code></td>"
        f"<td><code>{html.escape(datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).astimezone().isoformat(timespec='seconds'))}</code></td>"
        f"<td>{path.stat().st_size}</td>"
        "</tr>"
        for path in source_files
    )

    html_doc = f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Crash Report Digest</title>
  <style>
    :root {{
      --text: #111;
      --muted: #555;
      --line: #ddd;
      --bg: #fff;
      --accent: #8b0000;
    }}
    body {{
      color: var(--text);
      background: var(--bg);
      font-family: "Georgia", "Times New Roman", serif;
      margin: 24px;
      line-height: 1.4;
    }}
    h1, h2, h3 {{
      margin: 0.6em 0 0.35em;
      page-break-after: avoid;
    }}
    h1 {{
      color: var(--accent);
    }}
    .meta {{
      color: var(--muted);
      font-size: 0.95rem;
      margin-bottom: 16px;
    }}
    code {{
      font-family: "Menlo", "Consolas", monospace;
      font-size: 0.92em;
      white-space: normal;
      overflow-wrap: anywhere;
      word-break: break-word;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 10px 0 18px;
      table-layout: fixed;
      page-break-inside: auto;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 6px 8px;
      vertical-align: top;
      font-size: 0.95rem;
      overflow-wrap: anywhere;
      word-break: break-word;
    }}
    th {{
      text-align: left;
      background: #f7f7f7;
    }}
    thead {{
      display: table-header-group;
    }}
    tr {{
      page-break-inside: avoid;
      page-break-after: auto;
    }}
    .wrap-table {{
      table-layout: fixed;
      page-break-inside: auto;
    }}
    .recent-incidents-table {{
      font-size: 0.9rem;
    }}
    .recent-incidents-table th:nth-child(1),
    .recent-incidents-table td:nth-child(1) {{
      width: 15%;
    }}
    .recent-incidents-table th:nth-child(2),
    .recent-incidents-table td:nth-child(2) {{
      width: 10%;
    }}
    .recent-incidents-table th:nth-child(3),
    .recent-incidents-table td:nth-child(3) {{
      width: 12%;
    }}
    .recent-incidents-table th:nth-child(4),
    .recent-incidents-table td:nth-child(4) {{
      width: 11%;
    }}
    .recent-incidents-table th:nth-child(5),
    .recent-incidents-table td:nth-child(5) {{
      width: 52%;
    }}
    .root-causes-table th:nth-child(1),
    .root-causes-table td:nth-child(1) {{
      width: 24%;
    }}
    .root-causes-table th:nth-child(2),
    .root-causes-table td:nth-child(2) {{
      width: 8%;
    }}
    .root-causes-table th:nth-child(3),
    .root-causes-table td:nth-child(3) {{
      width: 68%;
    }}
    .source-files-table th:nth-child(1),
    .source-files-table td:nth-child(1) {{
      width: 60%;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }}
    .panel {{
      border: 1px solid var(--line);
      padding: 10px 12px;
      page-break-inside: avoid;
    }}
    ul {{
      margin-top: 8px;
    }}
    @page {{
      size: landscape;
      margin: 0.35in;
    }}
    @media print {{
      body {{
        margin: 0.35in;
        font-size: 11px;
      }}
      .wrap-table {{
        font-size: 0.82rem;
      }}
      .recent-incidents-table th,
      .recent-incidents-table td,
      .root-causes-table th,
      .root-causes-table td,
      .source-files-table th,
      .source-files-table td {{
        padding: 4px 5px;
      }}
    }}
  </style>
</head>
<body>
  <h1>Crash Report Digest</h1>
  <div class="meta">
    Generated (UTC): <code>{html.escape(context['generated_utc'])}</code><br>
    Generated (Local): <code>{html.escape(context['generated_local'])}</code><br>
    Project root: <code>{html.escape(str(PROJECT_ROOT))}</code><br>
    Lookback days: <code>{context['lookback_days']}</code>
  </div>

  <h2>Executive Summary</h2>
  <ul>{summary_html}</ul>

  <h2>Current Snapshot</h2>
  <div class="grid">
    <div class="panel">
      <ul>
        {li("Process watchdog timestamp", current["process_watchdog"].get("timestamp_utc", ""))}
        {li("Tripwire active", bool(current["tripwire"].get("active", False)))}
        {li("Global halt active", bool(current["halt_recovery"].get("halt_active", False)))}
        {li("Market data only", bool(current["halt_recovery"].get("market_data_only", False)))}
        {li("Allow order execution", bool(current["halt_recovery"].get("allow_order_execution", False)))}
      </ul>
    </div>
    <div class="panel">
      <ul>
        {li("Operator global action", current["operator_control"].get("global_action", ""))}
        {li("Operator timestamp", current["operator_control"].get("timestamp_utc", ""))}
        {li("Incident auto halt ok", bool(current["incident_auto_halt"].get("ok", False)))}
        {li("Incident auto halt failed checks", ",".join(current["incident_auto_halt"].get("failed_checks", []) if isinstance(current["incident_auto_halt"].get("failed_checks"), list) else []))}
      </ul>
    </div>
  </div>

  <h2>Current Watchdog Status</h2>
  <table>
    <thead><tr><th>Name</th><th>Running</th><th>Alt Running</th><th>Heartbeat OK</th><th>Heartbeat Age Seconds</th><th>Restart Reason</th></tr></thead>
    <tbody>{status_html}</tbody>
  </table>

  <h2>Watchdog Target Action Counts</h2>
  <table>
    <thead><tr><th>Target</th><th>Restart</th><th>Throttled</th><th>Halted</th></tr></thead>
    <tbody>{watchdog_html}</tbody>
  </table>

  <h2>Tripwire Counts</h2>
  <table>
    <thead><tr><th>Target</th><th>Opened</th><th>Cleared</th></tr></thead>
    <tbody>{tripwire_html}</tbody>
  </table>

  <h2>Halt Recovery Decisions</h2>
  <table>
    <thead><tr><th>Action / Decision</th><th>Count</th></tr></thead>
    <tbody>{halt_html}</tbody>
  </table>

  <h2>Inferred Halt Root Causes</h2>
  <table class="wrap-table root-causes-table">
    <thead><tr><th>Root Cause</th><th>Windows</th><th>Example</th></tr></thead>
    <tbody>{halt_root_cause_html}</tbody>
  </table>

  <h2>Incident Auto Halt Checks</h2>
  <table>
    <thead><tr><th>Failed Checks</th><th>Count</th></tr></thead>
    <tbody>{auto_halt_html}</tbody>
  </table>

  <h2>Critical Alert Counts</h2>
  <table>
    <thead><tr><th>Target</th><th>Event</th><th>Count</th></tr></thead>
    <tbody>{critical_alert_html}</tbody>
  </table>

  <h2>Incident Classes</h2>
  <table>
    <thead><tr><th>Class</th><th>Count</th></tr></thead>
    <tbody>{incident_class_html}</tbody>
  </table>

  <h2>Recent Bot Limit / Guardrail Events</h2>
  <table class="wrap-table recent-incidents-table">
    <thead><tr><th>Date (Local)</th><th>Source</th><th>Target</th><th>Event</th><th>Detail</th></tr></thead>
    <tbody>{recent_limit_html}</tbody>
  </table>

  <h2>Recent Crash / Restart Events</h2>
  <table class="wrap-table recent-incidents-table">
    <thead><tr><th>Date (Local)</th><th>Source</th><th>Target</th><th>Event</th><th>Detail</th></tr></thead>
    <tbody>{recent_crash_html}</tbody>
  </table>

  <h2>Recent Other Operational Alerts</h2>
  <table class="wrap-table recent-incidents-table">
    <thead><tr><th>Date (Local)</th><th>Source</th><th>Target</th><th>Event</th><th>Detail</th></tr></thead>
    <tbody>{recent_other_html}</tbody>
  </table>

  <h2>Start Commands</h2>
  <ul>
    <li><code>./scripts/ops/opsctl.sh start-live --force-restart</code></li>
    <li><code>./scripts/ops/opsctl.sh coinbase-start --paper --live-data --top-n 5 --min-acc 0.58 --profiles default</code></li>
    <li><code>./scripts/ops/opsctl.sh coinbase-futures-start --paper --live-data --top-n 10 --min-acc 0.56 --profiles crypto_futures</code></li>
    <li><code>./scripts/ops/opsctl.sh feed --source all --lines 40</code></li>
  </ul>

  <h2>Source Files</h2>
  <table class="wrap-table source-files-table">
    <thead><tr><th>Path</th><th>Modified (Local)</th><th>Size Bytes</th></tr></thead>
    <tbody>{sources_html}</tbody>
  </table>
</body>
</html>
'''
    return html_doc


def _build_context(lookback_days: int, recent_limit: int) -> Dict[str, Any]:
    now_utc = datetime.now(timezone.utc)
    since_utc = now_utc - timedelta(days=max(lookback_days, 1))

    current = {
        "process_watchdog": _load_json(HEALTH_DIR / "process_watchdog_latest.json"),
        "tripwire": _load_json(HEALTH_DIR / "shadow_watchdog_tripwire_latest.json"),
        "halt_recovery": _load_json(HEALTH_DIR / "shadow_watchdog_halt_recovery_latest.json"),
        "operator_control": _load_json(HEALTH_DIR / "operator_control_latest.json"),
        "incident_auto_halt": _load_json(ALERTS_DIR / "incident_auto_halt_latest.json"),
    }

    watchdog_rows = _collect_watchdog_rows(since_utc)
    tripwire_rows = _collect_tripwire_rows(since_utc)
    halt_rows = _collect_halt_recovery_rows(since_utc)
    auto_halt_rows = _collect_incident_auto_halt_rows(since_utc)
    critical_alert_rows = _collect_critical_alert_rows(since_utc)

    watchdog_counts, watchdog_incidents = _watchdog_action_summary(watchdog_rows)
    tripwire_counts, tripwire_incidents = _tripwire_summary(tripwire_rows)
    halt_counts, halt_windows = _halt_recovery_windows(halt_rows, auto_halt_rows=auto_halt_rows)
    halt_root_cause_counts, halt_root_cause_examples = _summarize_halt_root_causes(halt_windows)
    auto_halt_counts, auto_halt_incidents = _incident_auto_halt_summary(auto_halt_rows)
    critical_alert_counts, critical_alert_incidents = _critical_alert_summary(critical_alert_rows)

    recent_incidents = sorted(
        watchdog_incidents + tripwire_incidents + halt_windows + auto_halt_incidents + critical_alert_incidents,
        key=lambda row: str(row.get("timestamp_utc") or ""),
        reverse=True,
    )
    incident_class_counts, recent_limit_incidents, recent_crash_incidents, recent_other_incidents = _partition_incidents_by_class(
        recent_incidents,
        recent_limit=max(recent_limit, 1),
    )

    return {
        "generated_utc": now_utc.isoformat(),
        "generated_local": datetime.now().astimezone().isoformat(),
        "lookback_days": int(lookback_days),
        "current": current,
        "summary_points": _summary_points(current),
        "watchdog_counts": watchdog_counts,
        "tripwire_counts": tripwire_counts,
        "halt_counts": halt_counts,
        "halt_root_cause_counts": halt_root_cause_counts,
        "halt_root_cause_examples": halt_root_cause_examples,
        "auto_halt_counts": auto_halt_counts,
        "critical_alert_counts": critical_alert_counts,
        "incident_class_counts": incident_class_counts,
        "recent_incidents": recent_incidents[: max(recent_limit, 1)],
        "recent_limit_incidents": recent_limit_incidents,
        "recent_crash_incidents": recent_crash_incidents,
        "recent_other_incidents": recent_other_incidents,
        "source_files": _source_inventory(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a crash-report digest (markdown + printable HTML + optional PDF).")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--lookback-days", type=int, default=int(os.getenv("CRASH_REPORT_LOOKBACK_DAYS", "30")))
    parser.add_argument("--recent-limit", type=int, default=int(os.getenv("CRASH_REPORT_RECENT_LIMIT", "120")))
    parser.add_argument(
        "--render-pdf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render a PDF alongside markdown and printable HTML.",
    )
    parser.add_argument(
        "--allow-gui-pdf-renderer",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow GUI browser app bundles when no CLI PDF renderer is available.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    allow_gui_pdf_renderer = _env_flag("CRASH_REPORT_ALLOW_GUI_PDF_RENDERER", "0") if args.allow_gui_pdf_renderer is None else bool(args.allow_gui_pdf_renderer)

    context = _build_context(lookback_days=int(args.lookback_days), recent_limit=int(args.recent_limit))
    md_text = _render_markdown(context)
    html_text = _render_html(context)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    latest_md = out_dir / "crash_report_digest_latest.md"
    latest_html = out_dir / "crash_report_digest_print_latest.html"
    latest_pdf = out_dir / "crash_report_digest_latest.pdf"
    ts_md = out_dir / f"crash_report_digest_{stamp}.md"
    ts_html = out_dir / f"crash_report_digest_print_{stamp}.html"
    ts_pdf = out_dir / f"crash_report_digest_{stamp}.pdf"

    latest_md.write_text(md_text, encoding="utf-8")
    latest_html.write_text(html_text, encoding="utf-8")
    ts_md.write_text(md_text, encoding="utf-8")
    ts_html.write_text(html_text, encoding="utf-8")

    if latest_pdf.exists():
        latest_pdf.unlink()
    if ts_pdf.exists():
        ts_pdf.unlink()

    pdf_ok = False
    pdf_detail = "pdf_render_disabled"
    if bool(args.render_pdf):
        pdf_ok, pdf_detail = _render_pdf_from_html(
            latest_html,
            latest_pdf,
            allow_gui_renderer=bool(allow_gui_pdf_renderer),
        )
        if pdf_ok:
            try:
                shutil.copy2(latest_pdf, ts_pdf)
            except Exception as exc:
                pdf_ok = False
                pdf_detail = f"timestamp_pdf_copy_failed:{exc}"
                ts_pdf = None
        else:
            ts_pdf = None
    else:
        ts_pdf = None

    payload = {
        "latest_markdown": str(latest_md),
        "latest_printable_html": str(latest_html),
        "latest_pdf": str(latest_pdf) if latest_pdf.exists() else "",
        "timestamped_markdown": str(ts_md),
        "timestamped_printable_html": str(ts_html),
        "timestamped_pdf": str(ts_pdf) if ts_pdf is not None else "",
        "pdf_ok": bool(pdf_ok),
        "pdf_detail": str(pdf_detail),
        "generated_utc": context["generated_utc"],
        "lookback_days": int(args.lookback_days),
        "recent_limit": int(args.recent_limit),
        "allow_gui_pdf_renderer": bool(allow_gui_pdf_renderer),
        "recent_incident_rows": len(context["recent_incidents"]),
        "source_file_count": len(context["source_files"]),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(f"Wrote: {ts_md}")
        print(f"Wrote: {ts_html}")
        if ts_pdf is not None:
            print(f"Wrote: {ts_pdf}")
        print(f"Latest MD: {latest_md}")
        print(f"Latest HTML: {latest_html}")
        if latest_pdf.exists():
            print(f"Latest PDF: {latest_pdf}")
        else:
            print(f"PDF: {pdf_detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
