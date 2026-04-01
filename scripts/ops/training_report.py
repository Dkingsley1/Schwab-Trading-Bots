#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import html
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = PROJECT_ROOT / "exports" / "reports" / "training_reports"
LATEST_METADATA_PATH = PROJECT_ROOT / "governance" / "health" / "training_report_latest.json"
OPERATOR_NOTES_PATH = PROJECT_ROOT / "governance" / "health" / "retrain_operator_notes_latest.json"


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
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _normalize_operator_notes(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict) or not payload:
        return {}
    observations = [str(item).strip() for item in (payload.get("observations") or []) if str(item).strip()]
    training_guidance = [str(item).strip() for item in (payload.get("training_guidance") or []) if str(item).strip()]
    tags = [str(item).strip() for item in (payload.get("tags") or []) if str(item).strip()]
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    out = {
        "title": str(payload.get("title", "") or "").strip(),
        "summary": str(payload.get("summary", "") or "").strip(),
        "timestamp_local": _fmt_ts_local(payload.get("timestamp_utc")) or str(payload.get("timestamp_local", "") or "").strip(),
        "source": str(payload.get("source", "") or "").strip(),
        "requested_by": str(payload.get("requested_by", "") or "").strip(),
        "tags": tags,
        "observations": observations,
        "training_guidance": training_guidance,
        "metrics": metrics,
    }
    return {k: v for k, v in out.items() if v not in ("", [], {}, None)}


def _latest(pattern: str) -> Path | None:
    rows = sorted(glob.glob(pattern))
    return Path(rows[-1]) if rows else None


def _parse_ts(raw: Any) -> datetime | None:
    if not raw:
        return None
    txt = str(raw).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(txt)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _fmt_ts_local(raw: Any) -> str:
    dt = _parse_ts(raw)
    if dt is None:
        return ""
    return dt.astimezone().isoformat(timespec="seconds")


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except Exception:
        return None


def _fmt_num(value: Any, digits: int = 4) -> str:
    numeric = _coerce_float(value)
    if numeric is None:
        return str(value or "")
    return f"{numeric:.{digits}f}"


def _pdf_renderer_binary(allow_gui_renderer: bool) -> tuple[str, str]:
    env_override = os.getenv("TRAINING_REPORT_PDF_BIN", "").strip() or os.getenv("PROJECT_TIMELINE_PDF_BIN", "").strip()
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
        cmd = [renderer, "--headless", "--disable-gpu", f"--print-to-pdf={pdf_path}", html_uri]
    rc, out, err = _run(cmd)
    if rc == 0 and pdf_path.exists() and pdf_path.stat().st_size > 0:
        return True, out or "ok"
    return False, err or out or f"rc={rc}"


def _resolve_trade_log(explicit_path: str, scorecard: Dict[str, Any]) -> Path | None:
    if explicit_path:
        return Path(explicit_path).expanduser()
    lineage = scorecard.get("lineage") if isinstance(scorecard.get("lineage"), dict) else {}
    candidate_txt = str(lineage.get("trade_behavior_log_latest", "") or "").strip()
    if candidate_txt:
        candidate = Path(candidate_txt).expanduser()
        if candidate.exists():
            return candidate
    return _latest(str(PROJECT_ROOT / "logs" / "trade_behavior_policy_*.json"))


def _load_lane_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    lanes = payload.get("lanes") if isinstance(payload.get("lanes"), dict) else {}
    rows: List[Dict[str, Any]] = []
    for lane_name in sorted(lanes):
        row = lanes.get(lane_name)
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "lane": str(lane_name),
                "rows": int(row.get("rows", 0) or 0),
                "hold_rows": int(row.get("hold_rows", 0) or 0),
                "avg_master_score": _coerce_float(row.get("avg_master_score")),
                "execution_guard_block_rate": _coerce_float(row.get("execution_guard_block_rate")),
                "portfolio_risk_block_rate": _coerce_float(row.get("portfolio_risk_block_rate")),
            }
        )
    return rows


def _assessment_lines(context: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    summary = context.get("summary") if isinstance(context.get("summary"), dict) else {}
    promotion_quality = context.get("promotion_quality") if isinstance(context.get("promotion_quality"), dict) else {}
    promotion_gate = context.get("promotion_gate") if isinstance(context.get("promotion_gate"), dict) else {}
    trade = context.get("trade_behavior") if isinstance(context.get("trade_behavior"), dict) else {}
    divergence = context.get("data_divergence") if isinstance(context.get("data_divergence"), dict) else {}
    operator_notes = context.get("operator_notes") if isinstance(context.get("operator_notes"), dict) else {}

    target_count = int(summary.get("target_count", 0) or 0)
    trained_count = int(summary.get("trained_count", 0) or 0)
    if bool(summary.get("confirmed_training_success", False)):
        lines.append(f"Confirmed training success was achieved with {trained_count}/{target_count} trained targets.")
    else:
        reason = str(summary.get("training_reason", "") or summary.get("master_update_status", "unknown")).strip()
        lines.append(f"Training ran {trained_count}/{target_count} targets, but the run was not confirmed successful ({reason}).")

    failed_checks = [str(item) for item in (promotion_quality.get("failed_checks") or []) if str(item).strip()]
    if failed_checks:
        lines.append("Promotion quality gate blockers: " + ", ".join(failed_checks[:6]))

    if promotion_gate:
        considered = int(promotion_gate.get("considered_bots", 0) or 0)
        min_needed = int(((promotion_gate.get("thresholds") or {}).get("min_considered_bots", 0) or 0))
        if not bool(promotion_gate.get("coverage_ok", True)):
            lines.append(f"Walk-forward coverage is still short: considered_bots={considered} required={min_needed}.")

    score_delta = trade.get("score_delta")
    candidate_score = trade.get("candidate_score")
    previous_score = trade.get("previous_score")
    if score_delta is not None and candidate_score is not None and previous_score is not None:
        if float(score_delta) > 0.0:
            lines.append(f"Trade behavior candidate improved by {float(score_delta):.4f} versus the previous score.")
        elif float(score_delta) < 0.0:
            lines.append(f"Trade behavior candidate regressed by {abs(float(score_delta)):.4f} versus the previous score.")
        else:
            lines.append("Trade behavior candidate matched the previous score.")
    elif candidate_score is not None and previous_score is None:
        lines.append("Trade behavior candidate has no prior deployed score baseline for direct comparison.")

    if bool(trade.get("deployed_from_previous", False)):
        lines.append("The previous deployed trade behavior model remained active after rollback.")
    elif bool(trade.get("promoted", False)):
        lines.append("The trade behavior candidate was promoted.")

    if divergence and (divergence.get("ok") is False):
        worst = _coerce_float(divergence.get("worst_relative_spread"))
        ceiling = _coerce_float(divergence.get("max_relative_spread"))
        if worst is not None and ceiling is not None:
            lines.append(f"Data divergence is above the allowed threshold ({worst:.4f} vs {ceiling:.4f}).")

    if operator_notes:
        note_summary = str(operator_notes.get("summary", "") or "").strip()
        if note_summary:
            lines.append(f"Operator note carried into this retrain: {note_summary}")

    return lines


def _build_context(
    *,
    scorecard_path: Path,
    training_success_path: Path,
    promotion_quality_path: Path,
    promotion_gate_path: Path,
    graduation_path: Path,
    daily_verify_path: Path,
    data_divergence_path: Path,
    lane_scorecard_path: Path,
    trade_log_path: str = "",
) -> Dict[str, Any]:
    scorecard = _load_json(scorecard_path)
    success = _load_json(training_success_path)
    promotion_quality = _load_json(promotion_quality_path)
    promotion_gate = _load_json(promotion_gate_path)
    graduation = _load_json(graduation_path)
    daily_verify = _load_json(daily_verify_path)
    data_divergence = _load_json(data_divergence_path)
    lane_scorecard = _load_json(lane_scorecard_path)
    fallback_operator_notes = _load_json(OPERATOR_NOTES_PATH)

    resolved_trade_log = _resolve_trade_log(trade_log_path, scorecard)
    trade = _load_json(resolved_trade_log) if resolved_trade_log is not None else {}
    operator_notes = _normalize_operator_notes(
        scorecard.get("operator_notes") if isinstance(scorecard.get("operator_notes"), dict)
        else success.get("operator_notes") if isinstance(success.get("operator_notes"), dict)
        else fallback_operator_notes
    )

    candidate_score = _coerce_float(trade.get("candidate_score"))
    previous_score = _coerce_float(trade.get("previous_score"))
    score_delta = None
    if candidate_score is not None and previous_score is not None:
        score_delta = candidate_score - previous_score

    test_metrics = trade.get("test_metrics") if isinstance(trade.get("test_metrics"), dict) else {}
    promotion_gate_meta = trade.get("promotion_gate") if isinstance(trade.get("promotion_gate"), dict) else {}
    promotion_reasons = [str(item) for item in (promotion_gate_meta.get("reasons") or []) if str(item).strip()]
    failed_checks = [str(item) for item in (promotion_quality.get("failed_checks") or []) if str(item).strip()]
    daily_verify_failed_checks = [str(item) for item in (daily_verify.get("failed_checks") or []) if str(item).strip()]

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "started_utc": str(scorecard.get("started_utc", "") or ""),
        "ended_utc": str(scorecard.get("ended_utc", "") or ""),
        "target_count": int(scorecard.get("target_count", 0) or 0),
        "trained_count": int(success.get("trained_count", 0) or 0),
        "failure_count": int(scorecard.get("failure_count", success.get("failure_count", 0)) or 0),
        "skipped_by_memory_count": int(scorecard.get("skipped_by_memory_count", success.get("skipped_by_memory_count", 0)) or 0),
        "confirmed_training_success": bool(success.get("confirmed_training_success", False)),
        "training_reason": str(success.get("reason", "") or ""),
        "master_update_status": str(scorecard.get("master_update_status", success.get("master_update_status", "")) or ""),
        "data_quality_ok": bool(success.get("data_quality_ok", False)),
        "promotion_quality_ok": bool(promotion_quality.get("ok", False)),
        "daily_verify_ok": bool(daily_verify.get("ok", False)),
        "run_timestamp_local": _fmt_ts_local(success.get("timestamp_utc") or scorecard.get("timestamp_utc")),
    }

    context: Dict[str, Any] = {
        "generated_utc": summary["generated_utc"],
        "summary": summary,
        "scorecard": {
            "status_counts": scorecard.get("status_counts") if isinstance(scorecard.get("status_counts"), dict) else {},
            "accuracy_delta": scorecard.get("accuracy_delta") if isinstance(scorecard.get("accuracy_delta"), dict) else {},
            "canary_priority_selected": int(scorecard.get("canary_priority_selected", 0) or 0),
            "distillation_priority_selected": int(scorecard.get("distillation_priority_selected", 0) or 0),
            "failures": [str(item) for item in (scorecard.get("failures") or []) if str(item).strip()],
            "skipped_by_memory": [str(item) for item in (scorecard.get("skipped_by_memory") or []) if str(item).strip()],
        },
        "promotion_quality": {
            "ok": bool(promotion_quality.get("ok", False)),
            "failed_checks": failed_checks,
            "details": promotion_quality.get("details") if isinstance(promotion_quality.get("details"), dict) else {},
        },
        "promotion_gate": {
            "promote_ok": bool(promotion_gate.get("promote_ok", False)),
            "coverage_ok": bool(promotion_gate.get("coverage_ok", False)),
            "considered_bots": int(promotion_gate.get("considered_bots", 0) or 0),
            "failed_bots": int(promotion_gate.get("failed_bots", 0) or 0),
            "fail_share": _coerce_float(promotion_gate.get("fail_share")),
            "mean_trading_quality_score": _coerce_float(promotion_gate.get("mean_trading_quality_score")),
            "thresholds": promotion_gate.get("thresholds") if isinstance(promotion_gate.get("thresholds"), dict) else {},
        },
        "new_bot_graduation": {
            "ok": bool(graduation.get("ok", False)),
            "immature_active_count": int(graduation.get("immature_active_count", 0) or 0),
            "maturity": graduation.get("maturity") if isinstance(graduation.get("maturity"), dict) else {},
        },
        "trade_behavior": {
            "log_path": str(resolved_trade_log) if resolved_trade_log is not None else "",
            "model_path": str(trade.get("model_path", "") or ""),
            "candidate_score": candidate_score,
            "previous_score": previous_score,
            "score_delta": score_delta,
            "promoted": bool(trade.get("promoted", False)),
            "deployed_from_previous": bool(trade.get("deployed_from_previous", False)),
            "deployed_previous_model": str(trade.get("deployed_previous_model", "") or ""),
            "champion_seed": trade.get("champion_seed"),
            "promotion_reasons": promotion_reasons,
            "test_metrics": {
                "accuracy": _coerce_float(test_metrics.get("accuracy")),
                "macro_f1": _coerce_float(test_metrics.get("macro_f1")),
                "balanced_accuracy": _coerce_float(test_metrics.get("balanced_accuracy")),
                "neutral_f1": _coerce_float(test_metrics.get("neutral_f1")),
                "positive_precision": _coerce_float(test_metrics.get("positive_precision")),
                "positive_recall": _coerce_float(test_metrics.get("positive_recall")),
            },
        },
        "lane_scorecard": {
            "ok": bool(lane_scorecard.get("ok", False)),
            "lookback_hours": _coerce_float(lane_scorecard.get("lookback_hours")),
            "rows_used": int(lane_scorecard.get("rows_used", 0) or 0),
            "lane_rows": _load_lane_rows(lane_scorecard),
        },
        "data_divergence": {
            "ok": bool(data_divergence.get("ok", False)),
            "window_hours": int(data_divergence.get("window_hours", 0) or 0),
            "worst_relative_spread": _coerce_float(data_divergence.get("worst_relative_spread")),
            "max_relative_spread": _coerce_float(data_divergence.get("max_relative_spread")),
            "offenders": data_divergence.get("offenders") if isinstance(data_divergence.get("offenders"), list) else [],
        },
        "daily_verify": {
            "ok": bool(daily_verify.get("ok", False)),
            "failed_checks": daily_verify_failed_checks,
        },
        "operator_notes": operator_notes,
        "sources": {
            "retrain_scorecard": str(scorecard_path),
            "training_success": str(training_success_path),
            "promotion_quality": str(promotion_quality_path),
            "promotion_gate": str(promotion_gate_path),
            "new_bot_graduation": str(graduation_path),
            "daily_verify": str(daily_verify_path),
            "data_divergence": str(data_divergence_path),
            "unified_lane_scorecard": str(lane_scorecard_path),
            "trade_behavior_log": str(resolved_trade_log) if resolved_trade_log is not None else "",
            "operator_notes": str(OPERATOR_NOTES_PATH),
        },
    }
    context["assessment"] = _assessment_lines(context)
    return context


def _render_markdown(context: Dict[str, Any]) -> str:
    summary = context["summary"]
    scorecard = context["scorecard"]
    promotion_quality = context["promotion_quality"]
    promotion_gate = context["promotion_gate"]
    graduation = context["new_bot_graduation"]
    trade = context["trade_behavior"]
    lane_scorecard = context["lane_scorecard"]
    divergence = context["data_divergence"]
    operator_notes = context.get("operator_notes") if isinstance(context.get("operator_notes"), dict) else {}

    lines = [
        f"# Training Report ({context['generated_utc']})",
        "",
        "## Summary",
        f"- Run window: {summary['started_utc']} -> {summary['ended_utc']}",
        f"- Run completed at: {summary['run_timestamp_local']}",
        f"- Targets trained: {summary['trained_count']} / {summary['target_count']}",
        f"- Confirmed training success: {summary['confirmed_training_success']}",
        f"- Master update status: {summary['master_update_status']}",
        f"- Promotion quality gate ok: {promotion_quality['ok']}",
        f"- Daily verify ok: {summary['daily_verify_ok']}",
        "",
        "## Assessment",
    ]
    for line in context["assessment"]:
        lines.append(f"- {line}")

    if operator_notes:
        lines.extend(["", "## Operator Notes"])
        title = str(operator_notes.get("title", "") or "").strip()
        summary_text = str(operator_notes.get("summary", "") or "").strip()
        if title:
            lines.append(f"- Title: {title}")
        if summary_text:
            lines.append(f"- Summary: {summary_text}")
        if operator_notes.get("timestamp_local"):
            lines.append(f"- Captured: {operator_notes['timestamp_local']}")
        tags = [str(item) for item in (operator_notes.get("tags") or []) if str(item).strip()]
        if tags:
            lines.append(f"- Tags: {', '.join(tags)}")
        for item in [str(item) for item in (operator_notes.get("observations") or []) if str(item).strip()]:
            lines.append(f"- Observation: {item}")
        for item in [str(item) for item in (operator_notes.get("training_guidance") or []) if str(item).strip()]:
            lines.append(f"- Training guidance: {item}")

    lines.extend(
        [
            "",
            "## Retrain Scorecard",
            f"- Status counts: {json.dumps(scorecard['status_counts'], ensure_ascii=True, sort_keys=True)}",
            f"- Accuracy delta: {json.dumps(scorecard['accuracy_delta'], ensure_ascii=True, sort_keys=True)}",
            f"- Canary-priority selected: {scorecard['canary_priority_selected']}",
            f"- Distillation-priority selected: {scorecard['distillation_priority_selected']}",
        ]
    )
    if scorecard["failures"]:
        lines.append("- Failures: " + ", ".join(scorecard["failures"][:10]))
    if scorecard["skipped_by_memory"]:
        lines.append("- Skipped by memory/thermal: " + ", ".join(scorecard["skipped_by_memory"][:10]))

    lines.extend(
        [
            "",
            "## Promotion Gates",
            f"- Promotion quality failed checks: {', '.join(promotion_quality['failed_checks']) or 'none'}",
            f"- Walk-forward promote_ok={promotion_gate['promote_ok']} coverage_ok={promotion_gate['coverage_ok']} considered_bots={promotion_gate['considered_bots']}",
            f"- Walk-forward fail_share={_fmt_num(promotion_gate['fail_share'])} mean_trading_quality_score={_fmt_num(promotion_gate['mean_trading_quality_score'])}",
            f"- New-bot graduation ok: {graduation['ok']} immature_active_count={graduation['immature_active_count']}",
        ]
    )

    test_metrics = trade["test_metrics"]
    lines.extend(
        [
            "",
            "## Trade Behavior Model",
            f"- Log: {trade['log_path']}",
            f"- Model path: {trade['model_path']}",
            f"- Candidate score: {_fmt_num(trade['candidate_score'])}",
            f"- Previous score: {_fmt_num(trade['previous_score'])}",
            f"- Score delta: {_fmt_num(trade['score_delta'])}",
            f"- Promoted: {trade['promoted']}",
            f"- Deployed from previous: {trade['deployed_from_previous']}",
            f"- Test metrics: accuracy={_fmt_num(test_metrics['accuracy'])} macro_f1={_fmt_num(test_metrics['macro_f1'])} balanced_accuracy={_fmt_num(test_metrics['balanced_accuracy'])}",
        ]
    )
    if trade["deployed_previous_model"]:
        lines.append(f"- Previous deployed model: {trade['deployed_previous_model']}")
    if trade["promotion_reasons"]:
        lines.append("- Promotion reasons: " + "; ".join(trade["promotion_reasons"][:8]))

    lines.extend(
        [
            "",
            "## Data Quality",
            f"- Divergence ok: {divergence['ok']} worst_relative_spread={_fmt_num(divergence['worst_relative_spread'])} max_relative_spread={_fmt_num(divergence['max_relative_spread'])}",
        ]
    )
    offenders = divergence.get("offenders") or []
    for row in offenders[:5]:
        if not isinstance(row, dict):
            continue
        lines.append(
            "- Offender: "
            f"symbol={row.get('symbol', '')} minute={row.get('minute', '')} rel_spread={_fmt_num(row.get('rel_spread'))} n={row.get('n', '')}"
        )

    lines.extend(["", "## Lane Scorecard"])
    for row in lane_scorecard["lane_rows"]:
        lines.append(
            "- "
            f"{row['lane']}: rows={row['rows']} hold_rows={row['hold_rows']} avg_master_score={_fmt_num(row['avg_master_score'])} "
            f"execution_guard_block_rate={_fmt_num(row['execution_guard_block_rate'])} portfolio_risk_block_rate={_fmt_num(row['portfolio_risk_block_rate'])}"
        )

    lines.extend(["", "## Sources"])
    for label, path in context["sources"].items():
        lines.append(f"- {label}: {path}")
    lines.append("")
    return "\n".join(lines)


def _lane_rows_html(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "<p>No lane scorecard rows available.</p>"
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row['lane']))}</td>"
            f"<td>{row['rows']}</td>"
            f"<td>{row['hold_rows']}</td>"
            f"<td>{html.escape(_fmt_num(row['avg_master_score']))}</td>"
            f"<td>{html.escape(_fmt_num(row['execution_guard_block_rate']))}</td>"
            f"<td>{html.escape(_fmt_num(row['portfolio_risk_block_rate']))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Lane</th><th>Rows</th><th>Hold Rows</th><th>Avg Score</th>"
        "<th>Exec Guard Block</th><th>Risk Block</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def _render_html(context: Dict[str, Any]) -> str:
    summary = context["summary"]
    promotion_quality = context["promotion_quality"]
    promotion_gate = context["promotion_gate"]
    graduation = context["new_bot_graduation"]
    trade = context["trade_behavior"]
    test_metrics = trade["test_metrics"]
    divergence = context["data_divergence"]
    operator_notes = context.get("operator_notes") if isinstance(context.get("operator_notes"), dict) else {}

    assessment_html = "".join(f"<li>{html.escape(line)}</li>" for line in context["assessment"])
    promotion_reasons_html = "".join(f"<li>{html.escape(str(line))}</li>" for line in trade["promotion_reasons"][:8])
    failed_checks_html = "".join(f"<li>{html.escape(str(line))}</li>" for line in promotion_quality["failed_checks"][:10])
    offenders_html = "".join(
        "<li>"
        f"symbol={html.escape(str(row.get('symbol', '')))} minute={html.escape(str(row.get('minute', '')))} "
        f"rel_spread={html.escape(_fmt_num(row.get('rel_spread')))} n={html.escape(str(row.get('n', '')))}"
        "</li>"
        for row in (divergence.get("offenders") or [])[:5]
        if isinstance(row, dict)
    )
    source_html = "".join(
        f"<li><strong>{html.escape(label)}</strong>: {html.escape(str(path))}</li>"
        for label, path in context["sources"].items()
    )
    operator_notes_html = ""
    if operator_notes:
        operator_lines: list[str] = []
        title = str(operator_notes.get("title", "") or "").strip()
        summary_text = str(operator_notes.get("summary", "") or "").strip()
        if title:
            operator_lines.append(f"<li><strong>Title</strong>: {html.escape(title)}</li>")
        if summary_text:
            operator_lines.append(f"<li><strong>Summary</strong>: {html.escape(summary_text)}</li>")
        if operator_notes.get("timestamp_local"):
            operator_lines.append(f"<li><strong>Captured</strong>: {html.escape(str(operator_notes['timestamp_local']))}</li>")
        tags = [str(item) for item in (operator_notes.get("tags") or []) if str(item).strip()]
        if tags:
            operator_lines.append(f"<li><strong>Tags</strong>: {html.escape(', '.join(tags))}</li>")
        operator_lines.extend(
            f"<li><strong>Observation</strong>: {html.escape(str(item))}</li>"
            for item in (operator_notes.get("observations") or [])
            if str(item).strip()
        )
        operator_lines.extend(
            f"<li><strong>Training guidance</strong>: {html.escape(str(item))}</li>"
            for item in (operator_notes.get("training_guidance") or [])
            if str(item).strip()
        )
        operator_notes_html = (
            "\n    <section class=\"section\">\n"
            "      <h2>Operator Notes</h2>\n"
            f"      <ul>{''.join(operator_lines)}</ul>\n"
            "    </section>\n"
        )

    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Training Report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f1e8;
      --ink: #1d2a33;
      --muted: #5a6872;
      --card: #fffaf2;
      --line: #d8cdbb;
      --accent: #9a3412;
      --accent-soft: #fde7d8;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: linear-gradient(180deg, #f3efe5 0%, #efe7d8 100%); color: var(--ink); font: 15px/1.5 Georgia, 'Times New Roman', serif; }}
    .page {{ max-width: 980px; margin: 0 auto; padding: 40px 28px 56px; }}
    .hero {{ background: var(--card); border: 1px solid var(--line); border-radius: 18px; padding: 28px; box-shadow: 0 12px 30px rgba(29, 42, 51, 0.08); }}
    h1, h2 {{ margin: 0 0 12px; font-family: 'Avenir Next', 'Segoe UI', sans-serif; }}
    h1 {{ font-size: 34px; letter-spacing: 0.02em; }}
    h2 {{ margin-top: 28px; font-size: 20px; }}
    p.meta {{ margin: 0; color: var(--muted); }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin-top: 18px; }}
    .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 16px; }}
    .label {{ color: var(--muted); font-family: 'Avenir Next', 'Segoe UI', sans-serif; font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .value {{ font-size: 24px; margin-top: 6px; font-weight: 700; }}
    .section {{ margin-top: 22px; background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: 20px 22px; }}
    ul {{ margin: 10px 0 0 18px; padding: 0; }}
    li {{ margin: 4px 0; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px 6px; text-align: left; }}
    th {{ font-family: 'Avenir Next', 'Segoe UI', sans-serif; font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }}
    .pill {{ display: inline-block; padding: 2px 8px; border-radius: 999px; background: var(--accent-soft); color: var(--accent); font-family: 'Avenir Next', 'Segoe UI', sans-serif; font-size: 12px; }}
    .mono {{ font-family: 'SF Mono', 'Menlo', monospace; font-size: 12px; word-break: break-all; }}
  </style>
</head>
<body>
  <div class=\"page\">
    <section class=\"hero\">
      <h1>Training Report</h1>
      <p class=\"meta\">Generated {html.escape(summary['run_timestamp_local'] or context['generated_utc'])}</p>
      <div class=\"grid\">
        <div class=\"card\"><div class=\"label\">Targets</div><div class=\"value\">{summary['trained_count']} / {summary['target_count']}</div></div>
        <div class=\"card\"><div class=\"label\">Confirmed Success</div><div class=\"value\">{html.escape(str(summary['confirmed_training_success']))}</div></div>
        <div class=\"card\"><div class=\"label\">Master Update</div><div class=\"value\">{html.escape(summary['master_update_status'] or 'unknown')}</div></div>
      </div>
    </section>

    <section class=\"section\">
      <h2>Assessment</h2>
      <ul>{assessment_html}</ul>
    </section>
{operator_notes_html}

    <section class=\"section\">
      <h2>Gate Summary</h2>
      <ul>
        <li>Promotion quality ok: <span class=\"pill\">{html.escape(str(promotion_quality['ok']))}</span></li>
        <li>Walk-forward promote_ok={html.escape(str(promotion_gate['promote_ok']))} coverage_ok={html.escape(str(promotion_gate['coverage_ok']))} considered_bots={promotion_gate['considered_bots']}</li>
        <li>New-bot graduation ok: {html.escape(str(graduation['ok']))} immature_active_count={graduation['immature_active_count']}</li>
      </ul>
      <ul>{failed_checks_html or '<li>No promotion-quality failed checks recorded.</li>'}</ul>
    </section>

    <section class=\"section\">
      <h2>Trade Behavior Candidate</h2>
      <ul>
        <li>Candidate score: {html.escape(_fmt_num(trade['candidate_score']))}</li>
        <li>Previous score: {html.escape(_fmt_num(trade['previous_score']))}</li>
        <li>Score delta: {html.escape(_fmt_num(trade['score_delta']))}</li>
        <li>Promoted: {html.escape(str(trade['promoted']))}</li>
        <li>Deployed from previous: {html.escape(str(trade['deployed_from_previous']))}</li>
        <li>Accuracy: {html.escape(_fmt_num(test_metrics['accuracy']))}</li>
        <li>Macro F1: {html.escape(_fmt_num(test_metrics['macro_f1']))}</li>
        <li>Balanced accuracy: {html.escape(_fmt_num(test_metrics['balanced_accuracy']))}</li>
      </ul>
      <ul>{promotion_reasons_html or '<li>No trade-behavior promotion reasons recorded.</li>'}</ul>
      <p class=\"mono\">{html.escape(trade['model_path'])}</p>
    </section>

    <section class=\"section\">
      <h2>Data Quality</h2>
      <ul>
        <li>Divergence ok: {html.escape(str(divergence['ok']))}</li>
        <li>Worst relative spread: {html.escape(_fmt_num(divergence['worst_relative_spread']))}</li>
        <li>Allowed max spread: {html.escape(_fmt_num(divergence['max_relative_spread']))}</li>
      </ul>
      <ul>{offenders_html or '<li>No divergence offenders recorded.</li>'}</ul>
    </section>

    <section class=\"section\">
      <h2>Lane Scorecard</h2>
      {_lane_rows_html(context['lane_scorecard']['lane_rows'])}
    </section>

    <section class=\"section\">
      <h2>Sources</h2>
      <ul>{source_html}</ul>
    </section>
  </div>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a training report with markdown, HTML, and optional PDF output.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--retrain-scorecard", default=str(PROJECT_ROOT / "governance" / "health" / "retrain_scorecard_latest.json"))
    parser.add_argument("--training-success", default=str(PROJECT_ROOT / "governance" / "health" / "training_success_latest.json"))
    parser.add_argument("--promotion-quality", default=str(PROJECT_ROOT / "governance" / "health" / "promotion_quality_gate_latest.json"))
    parser.add_argument("--promotion-gate", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "promotion_gate_latest.json"))
    parser.add_argument("--new-bot-graduation", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "new_bot_graduation_latest.json"))
    parser.add_argument("--daily-verify", default=str(PROJECT_ROOT / "governance" / "health" / "daily_auto_verify_latest.json"))
    parser.add_argument("--data-divergence", default=str(PROJECT_ROOT / "governance" / "health" / "data_source_divergence_latest.json"))
    parser.add_argument("--lane-scorecard", default=str(PROJECT_ROOT / "governance" / "health" / "unified_lane_scorecard_latest.json"))
    parser.add_argument("--trade-log", default="")
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

    allow_gui_pdf_renderer = _env_flag("TRAINING_REPORT_ALLOW_GUI_PDF_RENDERER", "0") if args.allow_gui_pdf_renderer is None else bool(args.allow_gui_pdf_renderer)

    context = _build_context(
        scorecard_path=Path(args.retrain_scorecard),
        training_success_path=Path(args.training_success),
        promotion_quality_path=Path(args.promotion_quality),
        promotion_gate_path=Path(args.promotion_gate),
        graduation_path=Path(args.new_bot_graduation),
        daily_verify_path=Path(args.daily_verify),
        data_divergence_path=Path(args.data_divergence),
        lane_scorecard_path=Path(args.lane_scorecard),
        trade_log_path=str(args.trade_log),
    )
    md_text = _render_markdown(context)
    html_text = _render_html(context)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    latest_md = out_dir / "training_report_latest.md"
    latest_html = out_dir / "training_report_print_latest.html"
    latest_pdf = out_dir / "training_report_latest.pdf"
    ts_md = out_dir / f"training_report_{stamp}.md"
    ts_html = out_dir / f"training_report_print_{stamp}.html"
    ts_pdf = out_dir / f"training_report_{stamp}.pdf"

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
        pdf_ok, pdf_detail = _render_pdf_from_html(latest_html, latest_pdf, allow_gui_renderer=bool(allow_gui_pdf_renderer))
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
        "allow_gui_pdf_renderer": bool(allow_gui_pdf_renderer),
        "summary": context["summary"],
        "trade_behavior": {
            "candidate_score": context["trade_behavior"].get("candidate_score"),
            "previous_score": context["trade_behavior"].get("previous_score"),
            "score_delta": context["trade_behavior"].get("score_delta"),
            "promoted": context["trade_behavior"].get("promoted"),
            "deployed_from_previous": context["trade_behavior"].get("deployed_from_previous"),
        },
    }
    LATEST_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    LATEST_METADATA_PATH.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

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
