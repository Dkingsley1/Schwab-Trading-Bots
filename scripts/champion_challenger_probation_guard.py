#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _infer_profile_from_bot_id(bot_id: str) -> str:
    text = str(bot_id or "").strip().lower()
    if "crypto_futures" in text:
        return "crypto_futures"
    if "crypto" in text:
        return "crypto"
    if "dividend" in text or "income" in text or "drip" in text:
        return "dividend"
    if "bond" in text or "yield" in text or "credit" in text:
        return "bond"
    if "fx" in text:
        return "fx"
    if "swing" in text:
        return "swing_aggressive"
    if "intraday" in text or "ultrafast" in text or "position_1m_3m" in text:
        return "intraday_aggressive"
    if "aggressive" in text:
        return "aggressive"
    return ""


def _resolve_profile(row: dict[str, Any]) -> tuple[str, str]:
    for key in ("profile", "source_profile", "paper_profile", "mode"):
        value = str(row.get(key) or "").strip().lower()
        if value:
            return value, key
    inferred = _infer_profile_from_bot_id(str(row.get("bot_id") or ""))
    return inferred, ("bot_id_inference" if inferred else "")


def _probation_scope_requested(row: dict[str, Any]) -> bool:
    if bool(row.get("active", False)):
        return True
    promotion_status = str(row.get("promotion_status") or "").strip().lower()
    if promotion_status in {"challenger", "probation", "candidate", "shadow", "paper"}:
        return True
    return False


def _cohort_rows(champion_registry: dict[str, Any], master_registry: dict[str, Any]) -> list[dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}

    def add_row(raw: Any, *, stage: str) -> None:
        if isinstance(raw, str):
            row = {"bot_id": raw, "stage": stage}
        elif isinstance(raw, dict):
            bot_id = str(raw.get("bot_id") or raw.get("name") or "").strip()
            if not bot_id:
                return
            row = {
                "bot_id": bot_id,
                "stage": str(raw.get("stage") or stage or "").strip().lower() or stage,
                "profile": str(raw.get("profile") or raw.get("source_profile") or raw.get("paper_profile") or "").strip().lower(),
                "source": str(raw.get("source") or "").strip(),
            }
        else:
            return
        bot_id = str(row.get("bot_id") or "").strip()
        if not bot_id:
            return
        existing = out.get(bot_id, {})
        merged = dict(existing)
        merged.update({k: v for k, v in row.items() if v})
        merged.setdefault("source", "champion_registry")
        out[bot_id] = merged

    for raw in champion_registry.get("probation_candidates") or []:
        add_row(raw, stage="probation")
    for raw in champion_registry.get("challengers") or []:
        add_row(raw, stage="challenger")
    last_event = champion_registry.get("last_event") if isinstance(champion_registry.get("last_event"), dict) else {}
    if str(last_event.get("candidate") or "").strip():
        add_row(
            {
                "bot_id": str(last_event.get("candidate") or "").strip(),
                "stage": str(last_event.get("stage_transition") or last_event.get("action") or "candidate"),
                "source": "last_event",
            },
            stage="candidate",
        )

    sub_bots = master_registry.get("sub_bots") if isinstance(master_registry.get("sub_bots"), list) else []
    for raw in sub_bots:
        if not isinstance(raw, dict):
            continue
        lifecycle_state = str(raw.get("lifecycle_state") or "").strip().lower()
        if lifecycle_state != "probation":
            continue
        if not _probation_scope_requested(raw):
            continue
        bot_id = str(raw.get("bot_id") or "").strip()
        if not bot_id:
            continue
        add_row(
            {
                "bot_id": bot_id,
                "stage": "probation",
                "profile": str(raw.get("profile") or raw.get("mode") or "").strip().lower(),
                "source": "master_registry",
            },
            stage="probation",
        )

    rows = []
    for bot_id, row in sorted(out.items()):
        profile, profile_source = _resolve_profile(row)
        rows.append(
            {
                "bot_id": bot_id,
                "stage": str(row.get("stage") or "probation"),
                "profile": profile,
                "profile_source": profile_source,
                "source": str(row.get("source") or "champion_registry"),
            }
        )
    return rows


def build_payload(
    *,
    champion_registry: dict[str, Any],
    master_registry: dict[str, Any],
    paper_execution_calibration: dict[str, Any],
    health_gates: dict[str, Any],
    paper_performance: dict[str, Any],
    max_calibration_mae_bps: float,
    max_calibration_bias_bps: float,
    max_latency_multiplier: float,
    min_profile_executions: int,
    min_profile_win_rate: float,
    min_profile_net_pnl: float,
) -> dict[str, Any]:
    cohort = _cohort_rows(champion_registry, master_registry)
    by_profile_calibration = (
        paper_execution_calibration.get("by_profile")
        if isinstance(paper_execution_calibration.get("by_profile"), dict)
        else {}
    )
    sleeve_latest = paper_performance.get("sleeve_latest") if isinstance(paper_performance.get("sleeve_latest"), list) else []
    paper_profiles = {
        str(row.get("profile") or "").strip().lower(): row
        for row in sleeve_latest
        if isinstance(row, dict) and str(row.get("profile") or "").strip()
    }
    health_summary = health_gates.get("summary") if isinstance(health_gates.get("summary"), dict) else {}
    gate_flags = health_gates.get("gates") if isinstance(health_gates.get("gates"), dict) else {}

    global_calibration_metrics = paper_execution_calibration.get("metrics") if isinstance(paper_execution_calibration.get("metrics"), dict) else {}
    calibration_drift = bool(
        (not paper_execution_calibration.get("ok", True))
        or _to_float(global_calibration_metrics.get("mae_bps"), 0.0) > float(max_calibration_mae_bps)
        or abs(_to_float(global_calibration_metrics.get("mean_bias_bps"), 0.0)) > float(max_calibration_bias_bps)
    )
    latency_drift = bool(
        gate_flags.get("priority_shard_latency", False)
        or _to_float(health_summary.get("worst_priority_latency_multiplier"), 0.0) > float(max_latency_multiplier)
        or (
            bool(health_gates.get("hard_gate_triggered", False))
            and bool(health_summary.get("priority_shard_latency_failures"))
        )
    )

    monitored_rows: list[dict[str, Any]] = []
    weak_paper_execution = False
    for row in cohort:
        profile = str(row.get("profile") or "").strip().lower()
        calibration_profile = by_profile_calibration.get(profile) if isinstance(by_profile_calibration.get(profile), dict) else {}
        paper_profile = paper_profiles.get(profile) if isinstance(paper_profiles.get(profile), dict) else {}
        profile_calibration_drift = bool(
            profile
            and calibration_profile
            and (
                _to_float(calibration_profile.get("mae_bps"), 0.0) > float(max_calibration_mae_bps)
                or abs(_to_float(calibration_profile.get("mean_bias_bps"), 0.0)) > float(max_calibration_bias_bps)
            )
        )
        executions = _to_int(paper_profile.get("executions"), 0)
        win_rate = _to_float(paper_profile.get("win_rate"), 1.0)
        net_pnl = _to_float(paper_profile.get("ending_net_pnl_total"), 0.0)
        profile_weak_paper = bool(
            profile
            and paper_profile
            and executions >= int(min_profile_executions)
            and (win_rate < float(min_profile_win_rate) or net_pnl < float(min_profile_net_pnl))
        )
        weak_paper_execution = weak_paper_execution or profile_weak_paper
        monitored_rows.append(
            {
                **row,
                "profile_calibration_drift": profile_calibration_drift,
                "paper_execution_profile_found": bool(paper_profile),
                "paper_execution_quality_weak": profile_weak_paper,
                "paper_execution": {
                    "executions": executions,
                    "win_rate": round(win_rate, 6) if paper_profile else None,
                    "ending_net_pnl_total": round(net_pnl, 6) if paper_profile else None,
                },
            }
        )

    failed_checks: list[str] = []
    if cohort and calibration_drift:
        failed_checks.append("calibration_drift")
    if cohort and latency_drift:
        failed_checks.append("latency_drift")
    if cohort and weak_paper_execution:
        failed_checks.append("weak_paper_execution")

    rollback_candidate = str(
        ((champion_registry.get("champion") or {}).get("rollback_candidate") or "")
    ).strip()
    if not rollback_candidate:
        history = champion_registry.get("history") if isinstance(champion_registry.get("history"), list) else []
        if history:
            rollback_candidate = str((history[-1] or {}).get("name") or "").strip()

    rollback_required = bool(failed_checks)
    top_actions: list[str] = []
    if rollback_required and rollback_candidate:
        top_actions.append(f"rollback probation lane to {rollback_candidate} before widening promotion scope")
    elif rollback_required:
        top_actions.append("freeze probation lane and restore the last-known-good champion before widening promotion scope")
    if calibration_drift:
        top_actions.append("recalibrate execution and abstention controls before allowing the challenger to graduate")
    if latency_drift:
        top_actions.append("treat latency drift as a rollout blocker until the priority shard path returns inside its limit")
    if weak_paper_execution:
        top_actions.append("keep weak paper-execution challengers in probation until win rate and net pnl recover")

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": not rollback_required,
        "overall_status": ("idle" if not cohort else ("blocked" if rollback_required else "ready")),
        "rollback_required": rollback_required,
        "failed_checks": failed_checks,
        "thresholds": {
            "max_calibration_mae_bps": float(max_calibration_mae_bps),
            "max_calibration_bias_bps": float(max_calibration_bias_bps),
            "max_latency_multiplier": float(max_latency_multiplier),
            "min_profile_executions": int(min_profile_executions),
            "min_profile_win_rate": float(min_profile_win_rate),
            "min_profile_net_pnl": float(min_profile_net_pnl),
        },
        "probation_cohort_count": len(cohort),
        "rollback_candidate": rollback_candidate,
        "monitored_candidates": monitored_rows,
        "drift": {
            "calibration_drift": calibration_drift,
            "latency_drift": latency_drift,
            "weak_paper_execution": weak_paper_execution,
            "global_calibration_metrics": {
                "mae_bps": round(_to_float(global_calibration_metrics.get("mae_bps"), 0.0), 6),
                "mean_bias_bps": round(_to_float(global_calibration_metrics.get("mean_bias_bps"), 0.0), 6),
            },
            "priority_latency_failures": health_summary.get("priority_shard_latency_failures")
            if isinstance(health_summary.get("priority_shard_latency_failures"), list)
            else [],
            "worst_priority_latency_multiplier": round(
                _to_float(health_summary.get("worst_priority_latency_multiplier"), 0.0),
                6,
            ),
        },
        "top_actions": top_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Champion/challenger probation guard with automatic rollback triggers.")
    parser.add_argument("--registry", default=str(PROJECT_ROOT / "governance" / "champion_challenger" / "registry.json"))
    parser.add_argument("--master-registry", default=str(PROJECT_ROOT / "master_bot_registry.json"))
    parser.add_argument("--paper-execution-calibration", default=str(PROJECT_ROOT / "governance" / "health" / "paper_execution_calibration_latest.json"))
    parser.add_argument("--health-gates-file", default=str(PROJECT_ROOT / "governance" / "health" / "health_gates_latest.json"))
    parser.add_argument("--paper-performance-file", default=str(PROJECT_ROOT / "governance" / "health" / "paper_performance_latest.json"))
    parser.add_argument("--max-calibration-mae-bps", type=float, default=35.0)
    parser.add_argument("--max-calibration-bias-bps", type=float, default=12.0)
    parser.add_argument("--max-latency-multiplier", type=float, default=1.25)
    parser.add_argument("--min-profile-executions", type=int, default=10)
    parser.add_argument("--min-profile-win-rate", type=float, default=0.45)
    parser.add_argument("--min-profile-net-pnl", type=float, default=0.0)
    parser.add_argument("--out-file", default=str(PROJECT_ROOT / "governance" / "health" / "champion_challenger_probation_latest.json"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        champion_registry=_load_json(Path(args.registry)),
        master_registry=_load_json(Path(args.master_registry)),
        paper_execution_calibration=_load_json(Path(args.paper_execution_calibration)),
        health_gates=_load_json(Path(args.health_gates_file)),
        paper_performance=_load_json(Path(args.paper_performance_file)),
        max_calibration_mae_bps=float(args.max_calibration_mae_bps),
        max_calibration_bias_bps=float(args.max_calibration_bias_bps),
        max_latency_multiplier=float(args.max_latency_multiplier),
        min_profile_executions=int(args.min_profile_executions),
        min_profile_win_rate=float(args.min_profile_win_rate),
        min_profile_net_pnl=float(args.min_profile_net_pnl),
    )

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "champion_challenger_probation_guard "
            f"ok={str(payload['ok']).lower()} "
            f"cohort={int(payload['probation_cohort_count'])} "
            f"rollback_required={str(payload['rollback_required']).lower()}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
