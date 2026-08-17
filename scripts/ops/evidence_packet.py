#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, load_json, ordered_unique, payload_age_minutes, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, payload_age_minutes, write_payload


DEFAULT_JSON_OUT = PROJECT_ROOT / "governance" / "health" / "evidence_packet_latest.json"
DEFAULT_MD_OUT = PROJECT_ROOT / "exports" / "evidence_packets" / "evidence_packet_latest.md"
DEFAULT_HISTORY_DIR = PROJECT_ROOT / "exports" / "evidence_packets"

SOURCE_FILES = {
    "paper_performance": "governance/health/paper_performance_latest.json",
    "sleeve_profitability": "governance/health/sleeve_profitability_dashboard_latest.json",
    "paper_profitability": "governance/health/paper_profitability_control_latest.json",
    "income_operating_platform": "governance/health/income_operating_platform_latest.json",
    "runtime_gate": "governance/health/runtime_gate_dashboard_latest.json",
    "ingestion_storage": "governance/health/ingestion_storage_control_latest.json",
    "training_quality": "governance/health/training_quality_control_latest.json",
    "training_runtime": "governance/health/training_runtime_control_latest.json",
    "promotion_quality_gate": "governance/health/promotion_quality_gate_latest.json",
    "promotion_packet": "governance/champion_challenger/promotion_packet_latest.json",
    "capital_growth": "governance/health/capital_growth_intelligence_latest.json",
    "memory_pressure": "governance/health/memory_pressure_intelligence_latest.json",
}


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


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


def _first_present(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def _grade(score: float) -> str:
    score = _safe_float(score)
    if score >= 97.0:
        return "A+"
    if score >= 92.0:
        return "A+"
    if score >= 85.0:
        return "A"
    if score >= 75.0:
        return "B"
    if score >= 65.0:
        return "C"
    if score >= 50.0:
        return "D"
    return "F"


def _status_from_score(score: float, blockers: list[str]) -> str:
    if blockers and score < 75.0:
        return "needs_work"
    if blockers:
        return "watch"
    if score >= 85.0:
        return "ready"
    if score >= 65.0:
        return "forming"
    return "needs_work"


def _parse_day(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(text[:10], fmt).replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None


def _load_sources(project_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    metas: dict[str, Any] = {}
    for source_id, rel_path in SOURCE_FILES.items():
        path = project_root / rel_path
        payload = load_json(path)
        payloads[source_id] = payload
        metas[source_id] = {
            "path": str(path),
            "present": bool(payload),
            "age_minutes": None if not payload else payload_age_minutes(payload, path),
            "overall_status": str(payload.get("overall_status") or payload.get("status") or ""),
            "ok": payload.get("ok"),
        }
    return payloads, metas


def _daily_rows(paper: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [row for row in _as_list(paper.get("history_daily_series")) if isinstance(row, dict)]
    rows.sort(key=lambda row: str(row.get("day_utc") or row.get("day") or ""))
    return rows


def _day_key(row: dict[str, Any]) -> str:
    return str(row.get("day_utc") or row.get("day") or "").strip()


def _track_record_windows(paper: dict[str, Any], *, now: datetime | None = None) -> list[dict[str, Any]]:
    current = now or datetime.now(timezone.utc)
    rows = _daily_rows(paper)
    by_day = {_day_key(row): row for row in rows if _day_key(row)}
    available_days = sorted(str(day) for day in _as_list(paper.get("available_days")) if str(day).strip())
    if not available_days:
        available_days = sorted(by_day)

    windows: list[dict[str, Any]] = []
    for window_days in (30, 60, 90):
        cutoff = current - timedelta(days=window_days - 1)
        in_window = []
        for day in available_days:
            parsed = _parse_day(day)
            if parsed is not None and parsed >= cutoff:
                in_window.append(day)
        window_rows = [by_day[day] for day in in_window if day in by_day]
        coverage_ratio = min(len(in_window) / float(window_days), 1.0)
        change = sum(_safe_float(row.get("change_vs_previous_day"), 0.0) for row in window_rows)
        realized_changes = [
            _safe_float(row.get("realized_change_vs_previous_day"), 0.0)
            for row in window_rows
            if row.get("realized_change_vs_previous_day") is not None
        ]
        realized = sum(realized_changes)
        executions = sum(_safe_int(row.get("executions"), 0) for row in window_rows)
        if coverage_ratio >= 0.80:
            status = "credible_window"
        elif in_window:
            status = "forming"
        else:
            status = "missing"
        windows.append(
            {
                "window_days": window_days,
                "status": status,
                "observed_days": len(in_window),
                "coverage_ratio": round(coverage_ratio, 4),
                "observed_start_day_utc": in_window[0] if in_window else "",
                "observed_end_day_utc": in_window[-1] if in_window else "",
                "paper_change_sum": round(change, 6),
                "realized_pnl_sum": round(realized, 6),
                "realized_pnl_sum_method": "daily_realized_change" if realized_changes else "unavailable_exact_daily_change",
                "execution_count": executions,
                "evidence_note": "enough daily evidence for the window" if status == "credible_window" else "keep collecting consecutive paper days",
            }
        )
    return windows


def _paper_snapshot(sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    paper = sources["paper_performance"]
    profitability = sources["paper_profitability"]
    day = _as_dict(paper.get("day"))
    summary = _as_dict(profitability.get("paper_summary"))
    return {
        "day_utc": str(day.get("day_utc") or summary.get("day_utc") or ""),
        "executions": _safe_int(_first_present(day.get("executions"), summary.get("executions"), default=0), 0),
        "buy_count": _safe_int(day.get("buy_count"), 0),
        "sell_count": _safe_int(day.get("sell_count"), 0),
        "unique_symbols": _safe_int(day.get("unique_symbols"), 0),
        "ending_net_pnl_total": round(
            _safe_float(_first_present(day.get("ending_net_pnl_total"), summary.get("ending_net_pnl_total"), default=0.0), 0.0),
            6,
        ),
        "ending_realized_pnl_total": round(
            _safe_float(
                _first_present(day.get("ending_realized_pnl_total"), summary.get("ending_realized_pnl_total"), default=0.0),
                0.0,
            ),
            6,
        ),
        "ending_unrealized_pnl_total": round(
            _safe_float(
                _first_present(day.get("ending_unrealized_pnl_total"), summary.get("ending_unrealized_pnl_total"), default=0.0),
                0.0,
            ),
            6,
        ),
        "change_vs_previous_day": round(_safe_float(day.get("change_vs_previous_day"), 0.0), 6),
        "realized_change_vs_previous_day": round(_safe_float(day.get("realized_change_vs_previous_day"), 0.0), 6),
        "active_paper_profiles_today": _safe_int(paper.get("active_paper_profile_count_today"), 0),
        "available_days": len(_as_list(paper.get("available_days"))),
        "source_kind": str(paper.get("source_kind") or ""),
        "paper_control_grade": str(profitability.get("profitability_grade") or ""),
        "raw_profitability_grade": str(profitability.get("raw_profitability_grade") or ""),
        "financial_profitability_grade": str(profitability.get("financial_profitability_grade") or ""),
    }


def _sleeve_snapshot(sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sleeve = sources["sleeve_profitability"]
    totals = _as_dict(sleeve.get("totals"))
    top_rows = [row for row in _as_list(sleeve.get("top_sleeves")) if isinstance(row, dict)]
    bottom_rows = [row for row in _as_list(sleeve.get("bottom_sleeves")) if isinstance(row, dict)]
    return {
        "status": str(sleeve.get("overall_status") or ""),
        "profitability_grade": str(sleeve.get("profitability_grade") or ""),
        "sleeve_count": _safe_int(totals.get("sleeve_count"), 0),
        "execution_count": _safe_int(totals.get("execution_count"), 0),
        "realized_pnl_total": round(_safe_float(totals.get("realized_pnl_total"), 0.0), 6),
        "unrealized_pnl_total": round(_safe_float(totals.get("unrealized_pnl_total"), 0.0), 6),
        "net_pnl_total": round(_safe_float(totals.get("net_pnl_total"), 0.0), 6),
        "weak_sleeve_count": _safe_int(sleeve.get("weak_sleeve_count"), 0),
        "harvest_attention_count": _safe_int(sleeve.get("harvest_attention_count"), 0),
        "top_sleeves": top_rows[:5],
        "bottom_sleeves": bottom_rows[:5],
    }


def _harvest_snapshot(sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    profitability = sources["paper_profitability"]
    harvest = _as_dict(profitability.get("profit_harvest_report_card"))
    paper_summary = _as_dict(profitability.get("paper_summary"))
    realized = _safe_float(paper_summary.get("all_sleeve_realized_pnl_total"), 0.0)
    unrealized = _safe_float(paper_summary.get("all_sleeve_unrealized_pnl_total"), 0.0)
    denominator = max(abs(realized) + abs(unrealized), 1e-9)
    realized_share = realized / denominator
    return {
        "grade": str(harvest.get("grade") or ""),
        "raw_outcome_grade": str(harvest.get("raw_outcome_grade") or harvest.get("base_raw_outcome_grade") or ""),
        "control_grade": str(harvest.get("control_grade") or ""),
        "active": bool(harvest.get("active")),
        "realized_conversion_progress_norm": round(_safe_float(harvest.get("realized_conversion_progress_norm"), 0.0), 6),
        "target_realized_profit_share_norm": round(_safe_float(harvest.get("target_realized_profit_share_norm"), 0.0), 6),
        "current_realized_profit_share_norm": round(_safe_float(harvest.get("current_realized_profit_share_norm"), realized_share), 6),
        "computed_all_sleeve_realized_share": round(realized_share, 6),
        "next_action": str(harvest.get("next_action") or ""),
    }


def _risk_ops_snapshot(sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    income = sources["income_operating_platform"]
    runtime = sources["runtime_gate"]
    ingestion = sources["ingestion_storage"]
    runtime_overall = _as_dict(runtime.get("overall"))
    runtime_memory = _as_dict(runtime.get("memory"))
    runtime_storage = _as_dict(runtime.get("storage"))
    backpressure = _as_dict(ingestion.get("backpressure"))
    raw_live = _as_dict(backpressure.get("raw_live"))
    effective_raw_live = _as_dict(backpressure.get("effective_raw_live")) or raw_live
    return {
        "income_operating_status": str(income.get("overall_status") or ""),
        "income_operating_grade": str(income.get("income_operating_grade") or ""),
        "income_operating_score": round(_safe_float(income.get("income_operating_score"), 0.0), 3),
        "paper_only": bool(income.get("paper_only", True)),
        "live_execution_allowed": bool(income.get("live_execution_allowed")),
        "income_hard_blockers": [str(item) for item in _as_list(income.get("hard_blockers"))],
        "income_blockers": [str(item) for item in _as_list(income.get("blockers"))],
        "runtime_overall_status": str(runtime_overall.get("status") or runtime.get("overall_status") or ""),
        "runtime_attention": [str(item) for item in _as_list(runtime_overall.get("attention"))],
        "memory_status": str(runtime_memory.get("status") or ""),
        "memory_pressure_state": str(runtime_memory.get("memory_pressure_state") or ""),
        "swap_used_gb": round(_safe_float(runtime_memory.get("swap_used_gb"), 0.0), 3),
        "storage_status": str(runtime_storage.get("status") or ingestion.get("overall_status") or ""),
        "storage_pressure_profile": str(runtime_storage.get("pressure_profile") or ""),
        "total_pending_lines": _safe_int(backpressure.get("total_pending_lines"), 0),
        "raw_live_pending_lines": _safe_int(effective_raw_live.get("total_pending_lines"), 0),
        "raw_live_pending_lines_raw": _safe_int(raw_live.get("total_pending_lines"), 0),
        "raw_live_pending_lines_source": str(
            backpressure.get("effective_raw_live_source") or effective_raw_live.get("source") or ("raw_live" if raw_live else "")
        ),
        "oldest_pending_age_seconds": round(_safe_float(backpressure.get("oldest_pending_age_seconds"), 0.0), 3),
        "estimated_total_drain_minutes": backpressure.get("estimated_total_drain_minutes"),
    }


def _training_promotion_snapshot(sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    training = sources["training_quality"]
    training_runtime = sources["training_runtime"]
    quality_gate = sources["promotion_quality_gate"]
    packet = sources["promotion_packet"]
    details = _as_dict(quality_gate.get("details"))
    promotion_details = _as_dict(details.get("promotion"))
    committee = _as_dict(packet.get("committee"))
    replayability = _as_dict(packet.get("replayability_contract"))
    signature = _as_dict(packet.get("signature"))
    return {
        "training_quality_status": str(training.get("overall_status") or ""),
        "training_quality_score": round(_safe_float(training.get("training_quality_score"), 0.0), 3),
        "training_top_priorities": [str(item) for item in _as_list(training.get("top_priorities"))],
        "training_runtime_status": str(training_runtime.get("overall_status") or ""),
        "training_launch_allowed": bool(training_runtime.get("launch_allowed")),
        "training_runtime_blockers": [str(item) for item in _as_list(training_runtime.get("launch_blockers"))],
        "promotion_gate_ok": bool(quality_gate.get("ok")),
        "promotion_failed_checks": [str(item) for item in _as_list(quality_gate.get("failed_checks"))],
        "promotion_candidate_ids": [str(item) for item in _as_list(details.get("promotion_candidate_ids"))],
        "promotion_considered_bots": _safe_int(promotion_details.get("considered_bots"), 0),
        "promotion_packet_complete": bool(packet.get("packet_complete")),
        "ready_for_committee": bool(packet.get("ready_for_committee") or committee.get("ready_for_committee")),
        "packet_signature_verified": str(signature.get("status") or "") == "verified" or bool(signature.get("verified")),
        "exact_replay_ready": bool(replayability.get("exact_replay_ready") or packet.get("exact_replay_ready")),
        "trained_models_complete": bool(packet.get("trained_models_complete")),
    }


def _checklist(
    *,
    track_windows: list[dict[str, Any]],
    paper: dict[str, Any],
    sleeves: dict[str, Any],
    harvest: dict[str, Any],
    risk_ops: dict[str, Any],
    training_promotion: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = [
        {
            "id": "thirty_day_paper_track_record",
            "ready": any(row["window_days"] == 30 and row["status"] == "credible_window" for row in track_windows),
            "evidence": f"{next((row['observed_days'] for row in track_windows if row['window_days'] == 30), 0)} observed days",
            "next_step": "continue daily paper-performance refresh until 30-day coverage is credible",
        },
        {
            "id": "sixty_ninety_day_soak",
            "ready": all(
                any(row["window_days"] == day and row["observed_days"] >= min(day, 45) for row in track_windows)
                for day in (60, 90)
            ),
            "evidence": ", ".join(f"{row['window_days']}d={row['observed_days']}" for row in track_windows),
            "next_step": "let paper trading build the 60/90-day audit trail",
        },
        {
            "id": "sleeve_level_attribution",
            "ready": _safe_int(sleeves.get("sleeve_count"), 0) > 0 and _safe_int(sleeves.get("execution_count"), 0) > 0,
            "evidence": f"{sleeves.get('sleeve_count')} sleeves, {sleeves.get('execution_count')} executions",
            "next_step": "keep sleeve-profitability-dashboard current",
        },
        {
            "id": "realized_profit_conversion",
            "ready": _safe_float(harvest.get("computed_all_sleeve_realized_share"), 0.0) >= 0.35
            or str(harvest.get("raw_outcome_grade")) in {"B", "A", "A+", "A++"},
            "evidence": f"realized_share={harvest.get('computed_all_sleeve_realized_share')} raw={harvest.get('raw_outcome_grade')}",
            "next_step": "increase paper partial-harvest confirmations without breaking runner protection",
        },
        {
            "id": "drawdown_and_income_controls",
            "ready": str(risk_ops.get("income_operating_grade")) in {"A", "A+", "A++"}
            and not _as_list(risk_ops.get("income_hard_blockers")),
            "evidence": f"grade={risk_ops.get('income_operating_grade')} hard_blockers={len(_as_list(risk_ops.get('income_hard_blockers')))}",
            "next_step": "clear income hard blockers before any live-income dependence claim",
        },
        {
            "id": "operational_stability",
            "ready": str(risk_ops.get("memory_status")) == "ready"
            and _safe_int(risk_ops.get("raw_live_pending_lines"), 0) < 1000
            and str(risk_ops.get("storage_status")) in {"ready", "ok", "watch", "advisory"},
            "evidence": f"memory={risk_ops.get('memory_status')} raw_pending={risk_ops.get('raw_live_pending_lines')} storage={risk_ops.get('storage_status')}",
            "next_step": "clear stale storage gate mismatch and keep raw-live backlog below target",
        },
        {
            "id": "promotion_lineage_and_replay",
            "ready": bool(training_promotion.get("promotion_gate_ok"))
            and bool(training_promotion.get("promotion_packet_complete"))
            and bool(training_promotion.get("packet_signature_verified")),
            "evidence": f"gate_ok={training_promotion.get('promotion_gate_ok')} packet={training_promotion.get('promotion_packet_complete')} signature={training_promotion.get('packet_signature_verified')}",
            "next_step": "finish promotion packet builder/daily verify before calling candidates production-ready",
        },
    ]
    for row in rows:
        row["status"] = "ready" if row["ready"] else "needs_evidence"
    return rows


def _recommended_commands(checklist: list[dict[str, Any]], risk_ops: dict[str, Any], training_promotion: dict[str, Any]) -> list[list[str]]:
    commands: list[list[str]] = [
        ["./scripts/ops/opsctl.sh", "paper-profitability-control", "--apply", "--json"],
        ["./scripts/ops/opsctl.sh", "sleeve-profitability-dashboard", "--json"],
        ["./scripts/ops/opsctl.sh", "income-operating-platform", "--apply", "--json"],
    ]
    if str(risk_ops.get("storage_status")) in {"blocked", "critical"} or _safe_int(risk_ops.get("total_pending_lines"), 0) > 0:
        commands.append(["./scripts/ops/opsctl.sh", "writer-cycle-coordinator", "--apply", "--json"])
        commands.append(["./scripts/ops/opsctl.sh", "ingestion-storage-control", "--json"])
    if not bool(training_promotion.get("promotion_packet_complete")):
        commands.append(["./scripts/ops/opsctl.sh", "promotion-quality-gate", "--json"])
    if any(not row["ready"] for row in checklist):
        commands.append(["./scripts/ops/opsctl.sh", "evidence-packet", "--json"])
    return commands


def _truth_statement(sources: dict[str, dict[str, Any]], risk_ops: dict[str, Any]) -> str:
    live_allowed = bool(risk_ops.get("live_execution_allowed"))
    if live_allowed:
        return "Evidence packet includes paper and operational evidence; verify broker fill parity before relying on live income."
    return "Evidence packet is paper-mode proof, not live-income proof; live execution remains blocked/read-only."


def build_payload(project_root: Path = PROJECT_ROOT, *, now_utc: datetime | None = None) -> dict[str, Any]:
    sources, source_meta = _load_sources(project_root)
    track_windows = _track_record_windows(sources["paper_performance"], now=now_utc)
    paper = _paper_snapshot(sources)
    sleeves = _sleeve_snapshot(sources)
    harvest = _harvest_snapshot(sources)
    risk_ops = _risk_ops_snapshot(sources)
    training_promotion = _training_promotion_snapshot(sources)
    checklist = _checklist(
        track_windows=track_windows,
        paper=paper,
        sleeves=sleeves,
        harvest=harvest,
        risk_ops=risk_ops,
        training_promotion=training_promotion,
    )
    ready_count = sum(1 for row in checklist if row["ready"])
    readiness_score = round(100.0 * ready_count / max(len(checklist), 1), 3)
    blockers = [row["id"] for row in checklist if not row["ready"]]
    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": readiness_score >= 85.0 and not blockers,
        "overall_status": _status_from_score(readiness_score, blockers),
        "readiness_score": readiness_score,
        "readiness_grade": _grade(readiness_score),
        "truth_statement": _truth_statement(sources, risk_ops),
        "paper_snapshot": paper,
        "track_record_windows": track_windows,
        "sleeve_attribution": sleeves,
        "harvest_and_realization": harvest,
        "risk_and_operations": risk_ops,
        "training_and_promotion": training_promotion,
        "evidence_checklist": checklist,
        "blockers": blockers,
        "recommended_commands": _recommended_commands(checklist, risk_ops, training_promotion),
        "source_artifacts": source_meta,
        "contract": {
            "mode": "repeatable_evidence_packet_v1",
            "read_only": True,
            "live_execution_allowed": bool(risk_ops.get("live_execution_allowed")),
            "protected_volumes": {"VIDEO": "never_touched"},
            "intended_use": [
                "30/60/90-day paper proof tracking",
                "sleeve-level attribution",
                "drawdown and realized-conversion readiness",
                "operational stability evidence",
                "promotion lineage and replay readiness",
            ],
            "non_claims": [
                "does not claim live profitability",
                "does not bypass live execution guards",
                "does not replace broker fill verification",
            ],
        },
    }
    return payload


def _fmt_money(raw: Any) -> str:
    value = _safe_float(raw, 0.0)
    return f"${value:,.4f}"


def render_markdown(payload: dict[str, Any]) -> str:
    paper = _as_dict(payload.get("paper_snapshot"))
    sleeves = _as_dict(payload.get("sleeve_attribution"))
    harvest = _as_dict(payload.get("harvest_and_realization"))
    risk_ops = _as_dict(payload.get("risk_and_operations"))
    promotion = _as_dict(payload.get("training_and_promotion"))
    lines = [
        "# Trading System Evidence Packet",
        "",
        f"Generated UTC: `{payload.get('timestamp_utc')}`",
        "",
        "## Executive Status",
        "",
        f"- Status: `{payload.get('overall_status')}`",
        f"- Readiness: `{payload.get('readiness_score')}` / 100 (`{payload.get('readiness_grade')}`)",
        f"- Truth statement: {payload.get('truth_statement')}",
        "",
        "## 30/60/90-Day Paper Track Record",
        "",
        "| Window | Status | Observed Days | Paper Change | Realized Sum | Executions |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in _as_list(payload.get("track_record_windows")):
        if not isinstance(row, dict):
            continue
        lines.append(
            f"| {row.get('window_days')}d | {row.get('status')} | {row.get('observed_days')} | "
            f"{_fmt_money(row.get('paper_change_sum'))} | {_fmt_money(row.get('realized_pnl_sum'))} | {row.get('execution_count')} |"
        )
    lines.extend(
        [
            "",
            "## Current Paper Snapshot",
            "",
            f"- Day: `{paper.get('day_utc')}`",
            f"- Executions: `{paper.get('executions')}` across `{paper.get('unique_symbols')}` symbols",
            f"- Net / realized / unrealized: `{_fmt_money(paper.get('ending_net_pnl_total'))}` / `{_fmt_money(paper.get('ending_realized_pnl_total'))}` / `{_fmt_money(paper.get('ending_unrealized_pnl_total'))}`",
            f"- Paper control grade: `{paper.get('paper_control_grade')}`; raw profitability: `{paper.get('raw_profitability_grade')}`; financial: `{paper.get('financial_profitability_grade')}`",
            "",
            "## Sleeve Attribution",
            "",
            f"- Sleeves: `{sleeves.get('sleeve_count')}`",
            f"- Sleeve totals net / realized / unrealized: `{_fmt_money(sleeves.get('net_pnl_total'))}` / `{_fmt_money(sleeves.get('realized_pnl_total'))}` / `{_fmt_money(sleeves.get('unrealized_pnl_total'))}`",
            f"- Weak sleeves: `{sleeves.get('weak_sleeve_count')}`; harvest attention rows: `{sleeves.get('harvest_attention_count')}`",
            "",
            "## Harvest And Risk",
            "",
            f"- Harvest grade: `{harvest.get('grade')}`; raw outcome: `{harvest.get('raw_outcome_grade')}`; control: `{harvest.get('control_grade')}`",
            f"- Realized share: `{harvest.get('computed_all_sleeve_realized_share')}`; target: `{harvest.get('target_realized_profit_share_norm')}`",
            f"- Income platform: `{risk_ops.get('income_operating_status')}` / `{risk_ops.get('income_operating_grade')}`",
            f"- Live execution allowed: `{risk_ops.get('live_execution_allowed')}`",
            "",
            "## Operations",
            "",
            f"- Runtime: `{risk_ops.get('runtime_overall_status')}`; memory: `{risk_ops.get('memory_status')}` / `{risk_ops.get('memory_pressure_state')}`; swap GB: `{risk_ops.get('swap_used_gb')}`",
            f"- Storage: `{risk_ops.get('storage_status')}` / `{risk_ops.get('storage_pressure_profile')}`",
            f"- Pending lines: total `{risk_ops.get('total_pending_lines')}`, raw-live `{risk_ops.get('raw_live_pending_lines')}`; estimated drain minutes `{risk_ops.get('estimated_total_drain_minutes')}`",
            "",
            "## Training And Promotion",
            "",
            f"- Training quality: `{promotion.get('training_quality_status')}` score `{promotion.get('training_quality_score')}`",
            f"- Promotion gate ok: `{promotion.get('promotion_gate_ok')}`; packet complete: `{promotion.get('promotion_packet_complete')}`; signature verified: `{promotion.get('packet_signature_verified')}`",
            f"- Candidates: `{', '.join(str(item) for item in _as_list(promotion.get('promotion_candidate_ids')))}`",
            "",
            "## Evidence Checklist",
            "",
            "| Check | Status | Evidence | Next Step |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in _as_list(payload.get("evidence_checklist")):
        if not isinstance(row, dict):
            continue
        lines.append(f"| `{row.get('id')}` | `{row.get('status')}` | {row.get('evidence')} | {row.get('next_step')} |")
    lines.extend(["", "## Recommended Commands", ""])
    for command in _as_list(payload.get("recommended_commands")):
        if isinstance(command, list):
            lines.append(f"- `{' '.join(str(part) for part in command)}`")
    lines.append("")
    return "\n".join(lines)


def _history_path(history_dir: Path, timestamp_utc: str) -> Path:
    safe = timestamp_utc.replace(":", "").replace("+", "Z").replace("-", "").replace(".", "_")
    return history_dir / f"evidence_packet_{safe}.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a repeatable paper/evidence packet for operator or program-head review.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    parser.add_argument("--history-dir", default=str(DEFAULT_HISTORY_DIR))
    parser.add_argument("--no-md", action="store_true")
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    payload = build_payload(project_root)
    json_out = Path(args.json_out).expanduser()
    write_payload(json_out, payload)
    payload["artifacts"] = {"json": str(json_out)}

    if not args.no_history:
        history_path = _history_path(Path(args.history_dir).expanduser(), str(payload.get("timestamp_utc") or "latest"))
        write_payload(history_path, payload)
        payload["artifacts"]["history_json"] = str(history_path)

    if not args.no_md:
        md_out = Path(args.md_out).expanduser()
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(render_markdown(payload), encoding="utf-8")
        payload["artifacts"]["markdown"] = str(md_out)

    write_payload(json_out, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        print(
            "evidence_packet "
            f"status={payload['overall_status']} score={payload['readiness_score']} "
            f"grade={payload['readiness_grade']} json={json_out}"
        )
        if not args.no_md:
            print(f"markdown={Path(args.md_out).expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
