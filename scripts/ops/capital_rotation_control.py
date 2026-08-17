#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, payload_timestamp, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, payload_timestamp, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "capital_rotation_control_latest.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.capital_rotation_control_override"

LIVE_LOCK_REASON = "capital_rotation_control_is_advisory_and_paper_only_until_explicit_live_graduation"


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


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _signed_clamp(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def _status(payload: dict[str, Any]) -> str:
    for key in ("overall_status", "status"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if payload.get("ok") is True:
        return "ready"
    if payload.get("ok") is False:
        return "blocked"
    return "missing"


def _nested(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return current if current is not None else default


def _newer_than(primary: dict[str, Any], secondary: dict[str, Any], *, min_seconds: float = 60.0) -> bool:
    primary_ts = payload_timestamp(primary)
    secondary_ts = payload_timestamp(secondary)
    if primary_ts is None or secondary_ts is None:
        return False
    return (primary_ts - secondary_ts).total_seconds() >= min_seconds


def _ramp_stage(ramp: dict[str, Any]) -> str:
    return str(ramp.get("paper_ramp_stage") or ramp.get("stage") or ramp.get("status") or ramp.get("overall_status") or "").strip().lower()


def _load_sources(project_root: Path) -> dict[str, Any]:
    health = project_root / "governance" / "health"
    return {
        "capital_growth": load_json(health / "capital_growth_intelligence_latest.json"),
        "capital_awareness": load_json(health / "capital_growth_awareness_bridge_latest.json"),
        "paper_profitability": load_json(health / "paper_profitability_control_latest.json"),
        "paper_performance": load_json(health / "paper_performance_latest.json"),
        "sleeve_dashboard": load_json(health / "sleeve_profitability_dashboard_latest.json"),
        "whole_system_governor": load_json(health / "whole_system_governor_latest.json"),
        "paper_ramp": load_json(health / "paper_400_ramp_latest.json"),
        "runtime_throttle": load_json(health / "runtime_throttle_control_latest.json"),
        "health_fast": load_json(health / "health_fast_latest.json"),
    }


def _profile(row: dict[str, Any]) -> str:
    for key in ("profile", "cohort_key", "sleeve", "family", "group"):
        value = str(row.get(key) or "").strip().lower()
        if value:
            return value
    return "unknown"


def _paper_sleeve_rows(sources: dict[str, Any]) -> list[dict[str, Any]]:
    growth_rows = [
        row
        for row in _as_list(_as_dict(sources.get("capital_growth")).get("sleeve_growth_plan"))
        if isinstance(row, dict)
    ]
    if growth_rows:
        return growth_rows

    paper_rows = [
        row
        for row in _as_list(_as_dict(sources.get("paper_performance")).get("sleeve_latest"))
        if isinstance(row, dict)
    ]
    if paper_rows:
        return paper_rows

    dashboard = _as_dict(sources.get("sleeve_dashboard"))
    merged = _as_list(dashboard.get("top_sleeves")) + _as_list(dashboard.get("bottom_sleeves"))
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for row in merged:
        if not isinstance(row, dict):
            continue
        key = _profile(row)
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


def _weak_profiles(profitability: dict[str, Any]) -> set[str]:
    weak = {
        str(item or "").strip().lower()
        for item in _as_list(_nested(profitability, "a_plus_target_contract", "weak_profiles", default=[]))
        if str(item or "").strip()
    }
    active_controls = _as_dict(profitability.get("active_profile_controls"))
    for profile, control in active_controls.items():
        if not isinstance(control, dict):
            continue
        if (
            bool(control.get("block_new_entries"))
            or "quarantine" in str(control.get("action") or "").lower()
            or _safe_float(control.get("position_size_multiplier"), 1.0) <= 0.10
        ):
            weak.add(str(profile or "").strip().lower())
    return weak


def _sleeve_budgets(governor: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = governor.get("sleeve_budgets") if isinstance(governor.get("sleeve_budgets"), list) else []
    return {str(row.get("group") or "").strip().lower(): row for row in rows if isinstance(row, dict)}


def _runtime_pressure(sources: dict[str, Any]) -> dict[str, Any]:
    throttle = _as_dict(sources.get("runtime_throttle"))
    health = _as_dict(sources.get("health_fast"))
    health_ignored_as_stale = bool(throttle and health and _newer_than(throttle, health))
    storage = {} if health_ignored_as_stale else _as_dict(health.get("storage"))
    runtime = {} if health_ignored_as_stale else _as_dict(health.get("runtime_pressure"))
    memory = {} if health_ignored_as_stale else _as_dict(health.get("memory"))
    storage_pressure = _safe_float(
        storage.get("pressure_index"),
        _safe_float(_nested(throttle, "mac_fluidity_contract", "measurements", "storage_pressure_index", default=0.0)),
    )
    pending_lines = _safe_int(
        _nested(storage, "backpressure", "total_pending_lines", default=0),
        _safe_int(_nested(throttle, "mac_fluidity_contract", "measurements", "storage_total_pending_lines", default=0)),
    )
    runtime_status = _status(throttle) if throttle else str(runtime.get("overall_status") or "missing")
    compute_level = str(throttle.get("compute_pressure_level") or runtime.get("compute_pressure_level") or "").lower()
    memory_level = str(throttle.get("memory_pressure_level") or runtime.get("memory_pressure_level") or "").lower()
    if health_ignored_as_stale:
        memory_status = "needs_work" if memory_level in {"elevated", "high", "critical"} else "ready"
    else:
        memory_status = str(memory.get("overall_status") or "")
    guarded = bool(
        runtime_status in {"degraded", "blocked", "needs_work"}
        or memory_status in {"needs_work", "degraded", "blocked"}
        or compute_level in {"high", "critical"}
        or memory_level in {"high", "critical"}
        or storage_pressure >= 0.70
    )
    return {
        "runtime_status": runtime_status,
        "memory_status": memory_status,
        "compute_pressure_level": compute_level or "unknown",
        "memory_pressure_level": memory_level or "unknown",
        "storage_pressure_index": round(storage_pressure, 3),
        "pending_lines": pending_lines,
        "guarded": guarded,
        "source": "runtime_throttle_control" if health_ignored_as_stale else "health_fast_and_runtime_throttle",
        "health_fast_ignored_as_stale": health_ignored_as_stale,
    }


def _paper_rotation_gate(sources: dict[str, Any], pressure: dict[str, Any]) -> dict[str, Any]:
    ramp = _as_dict(sources.get("paper_ramp"))
    health = _as_dict(sources.get("health_fast"))
    guarded_paper = _as_dict(_nested(health, "operational_readiness", "guarded_paper", default={}))
    direct_ramp_stage = _ramp_stage(ramp)
    guarded_ramp_stage = str(guarded_paper.get("paper_ramp_stage") or "").strip().lower()
    ramp_stage = direct_ramp_stage or guarded_ramp_stage
    blockers = [str(item) for item in _as_list(ramp.get("blockers"))]
    direct_ramp_clear = bool(ramp) and not blockers and (
        direct_ramp_stage in {"armed", "ready", "active", "ok"} or bool(ramp.get("ok")) is True
    )
    stale_fast_gate_ignored = bool(direct_ramp_clear and health and _newer_than(ramp, health))
    if not stale_fast_gate_ignored:
        blockers.extend(str(item) for item in _as_list(guarded_paper.get("blockers")))
        blockers.extend(str(item) for item in _as_list(guarded_paper.get("paper_ramp_blockers")))
    if bool(pressure.get("guarded")):
        blockers.append("runtime_or_memory_storage_pressure_guarded")
    allowed = bool(ramp) and direct_ramp_clear and ramp_stage not in {"blocked", "halted"} and not blockers
    mode = "paper_budget_tilt_allowed" if allowed else "advisory_only_pressure_or_ramp_guarded"
    return {
        "allowed": allowed,
        "mode": mode,
        "paper_ramp_stage": ramp_stage or "unknown",
        "blockers": ordered_unique(blockers),
        "direct_paper_ramp_stage": direct_ramp_stage or "unknown",
        "guarded_paper_stage": guarded_ramp_stage or "unknown",
        "stale_health_fast_paper_gate_ignored": stale_fast_gate_ignored,
    }


def _live_money_gate(sources: dict[str, Any]) -> dict[str, Any]:
    growth = _as_dict(sources.get("capital_growth"))
    health = _as_dict(sources.get("health_fast"))
    growth_live = _as_dict(growth.get("live_money_scaling"))
    live_execution = _as_dict(_nested(health, "operational_readiness", "live_execution", default={}))
    blockers = [str(item) for item in _as_list(growth_live.get("blockers"))]
    blockers.extend(str(item) for item in _as_list(live_execution.get("blockers")))
    blockers.append(LIVE_LOCK_REASON)
    return {
        "allowed": False,
        "operator_approval_required": True,
        "source_live_money_allowed": bool(growth_live.get("allowed", False)),
        "source_live_execution_status": str(live_execution.get("status") or "unknown"),
        "blockers": ordered_unique(blockers),
        "policy": "never_convert_paper_rotation_to_live_money_without_explicit_operator_promotion_and_micro_live_gate",
    }


def _row_metrics(row: dict[str, Any]) -> dict[str, Any]:
    realized = _safe_float(row.get("realized_pnl"), _safe_float(row.get("realized_pnl_total"), _safe_float(row.get("ending_realized_pnl_total"), 0.0)))
    unrealized = _safe_float(row.get("unrealized_pnl"), _safe_float(row.get("unrealized_pnl_total"), _safe_float(row.get("ending_unrealized_pnl_total"), 0.0)))
    net = _safe_float(row.get("net_pnl"), _safe_float(row.get("net_pnl_total"), _safe_float(row.get("ending_net_pnl_total"), realized + unrealized)))
    return {
        "profile": _profile(row),
        "growth_score": _safe_float(row.get("growth_score"), _safe_float(row.get("score"), 50.0)),
        "growth_grade": str(row.get("growth_grade") or row.get("grade") or ""),
        "capital_action": str(row.get("capital_action") or "").strip(),
        "budget_reason": str(row.get("budget_reason") or "").strip(),
        "executions": _safe_int(row.get("executions"), 0),
        "win_rate": None if row.get("win_rate") is None else round(_safe_float(row.get("win_rate"), 0.0), 6),
        "realized_pnl": round(realized, 6),
        "unrealized_pnl": round(unrealized, 6),
        "net_pnl": round(net, 6),
        "paper_sim_budget_pct": _safe_float(row.get("paper_sim_budget_pct"), 0.0),
        "paper_sim_budget_usd": round(_safe_float(row.get("paper_sim_budget_usd"), 0.0), 2),
    }


def _rotation_row(
    row: dict[str, Any],
    *,
    weak: set[str],
    budgets: dict[str, dict[str, Any]],
    pressure: dict[str, Any],
    paper_gate: dict[str, Any],
) -> dict[str, Any]:
    metrics = _row_metrics(row)
    profile = metrics["profile"]
    budget = budgets.get(profile, {})
    growth_norm = _clamp(metrics["growth_score"] / 100.0)
    net = _safe_float(metrics["net_pnl"])
    realized = _safe_float(metrics["realized_pnl"])
    unrealized = _safe_float(metrics["unrealized_pnl"])
    win_rate = metrics.get("win_rate")
    win_norm = 0.5 if win_rate is None else _clamp(_safe_float(win_rate))
    action = str(metrics.get("capital_action") or "")
    weak_or_quarantined = bool(profile in weak or action == "cap_or_quarantine")
    value_score = _safe_float(budget.get("value_score"), growth_norm)
    cost_score = _safe_float(budget.get("cost_score"), 0.35)
    risk_score = _safe_float(budget.get("risk_score"), 0.0)
    drag_norm = _clamp(abs(min(unrealized, 0.0)) / 100.0)
    net_positive_norm = _clamp(net / 100.0)
    net_negative_norm = _clamp(abs(min(net, 0.0)) / 100.0)
    realized_norm = _clamp(realized / 50.0)
    pressure_penalty = 0.20 if bool(pressure.get("guarded")) else 0.0

    inflow_pressure = _clamp(
        0.28 * growth_norm
        + 0.20 * realized_norm
        + 0.16 * net_positive_norm
        + 0.14 * win_norm
        + 0.14 * _clamp(value_score)
        + (0.08 if action == "candidate_for_growth" else 0.0)
    )
    outflow_pressure = _clamp(
        0.24 * drag_norm
        + 0.22 * net_negative_norm
        + 0.18 * (1.0 if weak_or_quarantined else 0.0)
        + 0.14 * _clamp(risk_score)
        + 0.12 * _clamp(cost_score)
        + pressure_penalty
    )
    signed_pressure = _signed_clamp(inflow_pressure - outflow_pressure)
    if weak_or_quarantined or signed_pressure <= -0.25:
        direction = "outflow_or_quarantine"
        recommended_action = "quarantine_or_reduce_only"
    elif signed_pressure >= 0.30 and bool(paper_gate.get("allowed")):
        direction = "inflow_candidate"
        recommended_action = "paper_expand_candidate"
    elif signed_pressure >= 0.30:
        direction = "latent_inflow_candidate"
        recommended_action = "hold_inflow_until_pressure_clears"
    else:
        direction = "hold_or_observe"
        recommended_action = "hold_or_collect_more_evidence"

    max_delta = 0.035 if bool(paper_gate.get("allowed")) else 0.0
    if recommended_action == "quarantine_or_reduce_only":
        paper_delta = max(-0.05, min(signed_pressure * 0.04, -0.005))
    else:
        paper_delta = max(0.0, min(signed_pressure * 0.04, max_delta))

    return {
        **metrics,
        "direction": direction,
        "inflow_pressure_norm": round(inflow_pressure, 4),
        "outflow_pressure_norm": round(outflow_pressure, 4),
        "signed_rotation_pressure_norm": round(signed_pressure, 4),
        "paper_rotation_delta_norm": round(paper_delta, 5),
        "live_rotation_delta_norm": 0.0,
        "recommended_action": recommended_action,
        "why": ordered_unique(
            [
                str(metrics.get("budget_reason") or ""),
                "weak_or_quarantined_profile" if weak_or_quarantined else "",
                "runtime_pressure_guarded" if bool(pressure.get("guarded")) else "",
                "paper_gate_blocked" if not bool(paper_gate.get("allowed")) else "",
            ]
        ),
        "governor_budget": {
            "capture_tier": str(budget.get("capture_tier") or ""),
            "governor_action": str(budget.get("governor_action") or ""),
            "value_score": round(value_score, 4),
            "cost_score": round(cost_score, 4),
            "risk_score": round(risk_score, 4),
        },
    }


def _rotation_rows(sources: dict[str, Any], pressure: dict[str, Any], paper_gate: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _paper_sleeve_rows(sources)
    weak = _weak_profiles(_as_dict(sources.get("paper_profitability")))
    budgets = _sleeve_budgets(_as_dict(sources.get("whole_system_governor")))
    return sorted(
        [_rotation_row(row, weak=weak, budgets=budgets, pressure=pressure, paper_gate=paper_gate) for row in rows],
        key=lambda item: _safe_float(item.get("signed_rotation_pressure_norm")),
        reverse=True,
    )


def _portfolio_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    inflow = [row for row in rows if row.get("direction") in {"inflow_candidate", "latent_inflow_candidate"}]
    outflow = [row for row in rows if row.get("direction") == "outflow_or_quarantine"]
    paper_delta = sum(_safe_float(row.get("paper_rotation_delta_norm"), 0.0) for row in rows)
    return {
        "sleeve_count": len(rows),
        "inflow_candidate_count": len(inflow),
        "outflow_or_quarantine_count": len(outflow),
        "hold_count": max(len(rows) - len(inflow) - len(outflow), 0),
        "net_paper_rotation_delta_norm": round(paper_delta, 5),
        "net_live_rotation_delta_norm": 0.0,
        "strongest_inflow_profiles": [str(row.get("profile")) for row in inflow[:5]],
        "largest_outflow_profiles": [str(row.get("profile")) for row in sorted(outflow, key=lambda item: _safe_float(item.get("signed_rotation_pressure_norm")))[:5]],
    }


def _write_override(path: Path, payload: dict[str, Any]) -> None:
    runtime = _as_dict(payload.get("runtime_contract"))
    lines = [
        "# Generated by capital_rotation_control.py",
        f"CAPITAL_ROTATION_CONTROL_READY={1 if payload.get('ok') else 0}",
        f"CAPITAL_ROTATION_ACTION_MODE={runtime.get('paper_rotation_action_mode', '')}",
        f"CAPITAL_ROTATION_PAPER_TILT_ALLOWED={1 if runtime.get('paper_rotation_allowed') else 0}",
        "CAPITAL_ROTATION_LIVE_MONEY_ALLOWED=0",
        "CAPITAL_ROTATION_LIVE_EXECUTION_ALLOWED=0",
        f"CAPITAL_ROTATION_MAX_PAPER_DELTA_NORM={runtime.get('max_single_sleeve_paper_delta_norm', 0)}",
        f"CAPITAL_ROTATION_NET_PAPER_DELTA_NORM={runtime.get('net_paper_rotation_delta_norm', 0)}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_payload(project_root: Path = PROJECT_ROOT, *, apply: bool = False, override_path: Path = DEFAULT_OVERRIDE_PATH) -> dict[str, Any]:
    sources = _load_sources(project_root)
    pressure = _runtime_pressure(sources)
    paper_gate = _paper_rotation_gate(sources, pressure)
    live_gate = _live_money_gate(sources)
    rows = _rotation_rows(sources, pressure, paper_gate)
    summary = _portfolio_summary(rows)
    ok = bool(rows)
    status = "capital_rotation_ready" if ok and bool(paper_gate.get("allowed")) else "capital_rotation_advisory_only" if ok else "needs_capital_growth_plan"
    payload = {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": ok,
        "overall_status": status,
        "authority_boundary": "advisory_and_paper_rotation_only_no_live_money_movement",
        "capital_rotation_model": {
            "model_version": "capital_rotation_control_v1",
            "sees_movement_through": [
                "paper sleeve realized and unrealized PnL",
                "capital-growth sleeve scores and candidate/cap actions",
                "weak-sleeve profitability quarantines",
                "whole-system sleeve budget cost/value/risk scores",
                "runtime, memory, storage, and paper-ramp gates",
            ],
            "acts_on_movement_through": [
                "paper budget tilt recommendations",
                "reduce-only or quarantine recommendations",
                "hold-under-pressure decisions",
                "operator-review packets for future live micro promotion",
            ],
            "does_not_do": [
                "does_not_move_live_money",
                "does_not_clear_live_execution",
                "does_not_override_paper_trade_locks",
                "does_not_expand_when_runtime_or_storage_pressure_is_guarded",
            ],
        },
        "paper_rotation_gate": paper_gate,
        "live_money_promotion_gate": live_gate,
        "runtime_pressure": pressure,
        "portfolio_rotation": summary,
        "sleeve_rotation_plan": rows,
        "runtime_contract": {
            "paper_rotation_allowed": bool(paper_gate.get("allowed")),
            "paper_rotation_action_mode": str(paper_gate.get("mode") or ""),
            "live_money_rotation_allowed": False,
            "live_execution_allowed": False,
            "requires_operator_approval_for_live_money": True,
            "max_single_sleeve_paper_delta_norm": 0.035 if bool(paper_gate.get("allowed")) else 0.0,
            "net_paper_rotation_delta_norm": summary["net_paper_rotation_delta_norm"],
            "net_live_rotation_delta_norm": 0.0,
        },
        "recommended_commands": {
            "refresh_growth_plan": ["./scripts/ops/opsctl.sh", "capital-growth-intelligence", "--apply", "--json"],
            "refresh_rotation_control": ["./scripts/ops/opsctl.sh", "capital-rotation-control", "--json"],
            "apply_advisory_rotation_override": ["./scripts/ops/opsctl.sh", "capital-rotation-control", "--apply", "--json"],
            "refresh_paper_profitability": ["./scripts/ops/opsctl.sh", "paper-profitability-control", "--apply", "--json"],
            "refresh_whole_system_governor": ["./scripts/ops/opsctl.sh", "whole-system-governor", "--apply", "--json"],
        },
        "source_status": {name: _status(payload) for name, payload in sources.items()},
    }
    if apply:
        _write_override(override_path, payload)
        payload["write_result"] = {"applied": True, "override_path": str(override_path)}
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the paper-only capital rotation control surface.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--override-path", default=str(DEFAULT_OVERRIDE_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    payload = build_payload(project_root, apply=bool(args.apply), override_path=Path(args.override_path).expanduser())
    write_payload(Path(args.out_file).expanduser(), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        summary = _as_dict(payload.get("portfolio_rotation"))
        print(
            "capital_rotation_control "
            f"status={payload.get('overall_status')} "
            f"sleeves={_safe_int(summary.get('sleeve_count'), 0)} "
            f"inflow={_safe_int(summary.get('inflow_candidate_count'), 0)} "
            f"outflow={_safe_int(summary.get('outflow_or_quarantine_count'), 0)} "
            f"live_allowed=0"
        )
    return 0 if payload.get("ok", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
