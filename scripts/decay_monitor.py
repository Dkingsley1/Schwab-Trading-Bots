#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "research" / "decay_monitor_latest.json"


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


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _lcb95(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean - 1.96 * math.sqrt(max(variance, 0.0) / len(values))


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    health_root = project_root / "governance" / "health"

    paper = _load_json(health_root / "paper_performance_latest.json")
    promotion = _load_json(project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json")
    profitability_control = _load_json(health_root / "paper_profitability_control_latest.json")
    firewall_config = _load_json(project_root / "config" / "profitability_evidence_firewall_v1.json")
    decay_policy = firewall_config.get("edge_decay") if isinstance(firewall_config.get("edge_decay"), dict) else {}

    sleeve_latest = paper.get("sleeve_latest") if isinstance(paper.get("sleeve_latest"), list) else []
    history_daily = paper.get("history_daily_series") if isinstance(paper.get("history_daily_series"), list) else []
    period_change = paper.get("period_change_series") if isinstance(paper.get("period_change_series"), list) else []
    sleeve_daily = paper.get("sleeve_daily_series") if isinstance(paper.get("sleeve_daily_series"), dict) else {}
    candidate_daily = (
        paper.get("candidate_post_cost_daily_series")
        if isinstance(paper.get("candidate_post_cost_daily_series"), dict)
        else {}
    )
    evidence_window = (
        paper.get("profitability_evidence_window")
        if isinstance(paper.get("profitability_evidence_window"), dict)
        else {}
    )
    candidate_id = str(evidence_window.get("candidate_id") or "").strip()
    candidate_binding_required = bool(
        evidence_window.get("candidate_binding_required", False)
    )
    candidate_binding_mismatches = max(
        _safe_int(
            evidence_window.get("candidate_binding_mismatch_rows_excluded"),
            0,
        ),
        0,
    )
    candidate_bound = bool(
        not candidate_binding_required
        or (
            candidate_id
            and evidence_window.get("candidate_filter_active", False)
            and candidate_binding_mismatches == 0
        )
    )
    decay_daily = candidate_daily if candidate_binding_required else sleeve_daily
    decay_value_key = (
        "post_cost_pnl_delta_total"
        if candidate_binding_required
        else "change_vs_previous_day"
    )

    weak_sleeves: list[dict[str, Any]] = []
    active_sleeves = 0
    for row in sleeve_latest:
        if not isinstance(row, dict):
            continue
        profile = str(row.get("profile") or "").strip().lower()
        data_status = str(row.get("data_status") or "").strip().lower()
        if data_status not in {"current", "stale", "partial"}:
            continue
        active_sleeves += 1
        pnl = _safe_float(row.get("ending_net_pnl_total"), 0.0)
        win_rate_raw = row.get("win_rate")
        win_rate = _safe_float(win_rate_raw, -1.0) if win_rate_raw is not None else None
        if pnl < 0.0 or (win_rate is not None and win_rate < 0.45):
            weak_sleeves.append(
                {
                    "profile": profile,
                    "ending_net_pnl_total": round(pnl, 6),
                    "win_rate": round(win_rate, 6) if win_rate is not None else None,
                    "top_loss_causes": row.get("top_loss_causes") if isinstance(row.get("top_loss_causes"), list) else [],
                }
            )

    latest_change = _safe_float((history_daily[-1] if history_daily else {}).get("change_vs_previous_day"), 0.0)
    latest_net_pnl = _safe_float((history_daily[-1] if history_daily else {}).get("ending_net_pnl_total"), 0.0)
    previous_net_pnl = _safe_float((history_daily[-2] if len(history_daily) >= 2 else {}).get("ending_net_pnl_total"), 0.0)
    pnl_slope = round(latest_net_pnl - previous_net_pnl, 6) if len(history_daily) >= 2 else None

    sleeve_regimes = []
    for profile, rows in sleeve_daily.items():
        if not isinstance(rows, list) or not rows:
            continue
        latest = rows[-1] if isinstance(rows[-1], dict) else {}
        sleeve_regimes.append(
            {
                "profile": str(profile),
                "latest_net_pnl_total": round(_safe_float(latest.get("ending_net_pnl_total"), 0.0), 6),
                "change_vs_previous_day": round(_safe_float(latest.get("change_vs_previous_day"), 0.0), 6),
            }
        )
    sleeve_regimes.sort(key=lambda row: (float(row.get("latest_net_pnl_total", 0.0) or 0.0), str(row.get("profile") or "")))

    trailing_periods = []
    for row in period_change[:6]:
        if not isinstance(row, dict):
            continue
        trailing_periods.append(
            {
                "label": str(row.get("label") or ""),
                "window_days": _safe_int(row.get("window_days"), 0),
                "change": round(_safe_float(row.get("change"), 0.0), 6),
                "available_days": _safe_int(row.get("available_days"), 0),
            }
        )

    minimum_history = max(_safe_int(decay_policy.get("minimum_history_days"), 10), 3)
    recent_window = max(_safe_int(decay_policy.get("recent_window_days"), 3), 2)
    maximum_decay = max(0.0, min(_safe_float(decay_policy.get("maximum_mean_decay_fraction"), 0.5), 1.0))
    maximum_decayed_size = max(
        0.0,
        min(_safe_float(decay_policy.get("maximum_decayed_position_multiplier"), 0.1), 1.0),
    )
    active_controls = profitability_control.get("active_profile_controls")
    active_controls = active_controls if isinstance(active_controls, dict) else {}
    edge_rows: list[dict[str, Any]] = []
    insufficient_profiles: list[str] = []
    decayed_profiles: list[str] = []
    uncontained_profiles: list[str] = []
    known_profiles = {
        str(row.get("profile") or "").strip().lower()
        for row in sleeve_latest
        if isinstance(row, dict) and str(row.get("profile") or "").strip()
    }
    known_profiles.update(
        str(profile or "").strip().lower()
        for profile in decay_daily
        if str(profile or "").strip()
    )
    for profile in sorted(known_profiles):
        rows = decay_daily.get(profile) if isinstance(decay_daily, dict) else []
        if not isinstance(rows, list):
            insufficient_profiles.append(str(profile))
            continue
        values = [
            _safe_float(row.get(decay_value_key), 0.0)
            for row in rows
            if isinstance(row, dict)
        ]
        if len(values) < minimum_history or len(values) <= recent_window:
            insufficient_profiles.append(str(profile))
            continue
        prior = values[:-recent_window]
        recent = values[-recent_window:]
        prior_mean = _mean(prior)
        recent_mean = _mean(recent)
        recent_lcb = _lcb95(recent)
        decline_fraction = (
            max(0.0, (prior_mean - recent_mean) / max(abs(prior_mean), 1e-9))
            if prior_mean > 0.0
            else 0.0
        )
        decayed = bool(
            recent_mean < 0.0
            or (prior_mean > 0.0 and recent_lcb is not None and recent_lcb <= 0.0 and decline_fraction >= maximum_decay)
        )
        control = active_controls.get(str(profile)) if isinstance(active_controls.get(str(profile)), dict) else {}
        size_multiplier = _safe_float(control.get("position_size_multiplier"), 1.0)
        contained = bool(control.get("block_new_entries", False) or size_multiplier <= maximum_decayed_size)
        if decayed:
            decayed_profiles.append(str(profile))
            if not contained:
                uncontained_profiles.append(str(profile))
        edge_rows.append(
            {
                "profile": str(profile),
                "history_days": len(values),
                "prior_mean_daily_pnl": round(prior_mean, 8),
                "recent_mean_daily_pnl": round(recent_mean, 8),
                "recent_lower_confidence_bound_95": round(recent_lcb, 8) if recent_lcb is not None else None,
                "mean_decay_fraction": round(decline_fraction, 8),
                "decayed": decayed,
                "contained": contained,
                "position_size_multiplier": round(size_multiplier, 8),
                "automatic_action": "collect_only_or_reduce_only" if decayed else "retain_current_guarded_posture",
            }
        )
    edge_evidence_ready = bool(
        candidate_bound and decay_daily and not insufficient_profiles and edge_rows
    )
    automatic_demotion_ready = bool(profitability_control and not uncontained_profiles)
    edge_decay_contract = {
        "implementation_ready": bool(decay_policy),
        "evidence_ready": edge_evidence_ready,
        "automatic_demotion_ready": automatic_demotion_ready,
        "evaluated_profile_count": len(edge_rows),
        "insufficient_history_profiles": sorted(insufficient_profiles),
        "decayed_profiles": sorted(decayed_profiles),
        "uncontained_decayed_profiles": sorted(uncontained_profiles),
        "profiles": sorted(edge_rows, key=lambda row: str(row.get("profile") or "")),
        "thresholds": decay_policy,
        "evidence_scope": (
            "candidate_forward_profile_daily_post_cost_pnl"
            if candidate_binding_required
            else "legacy_lifetime_sleeve_daily_pnl"
        ),
        "policy": "edge decay automatically requires collect-only or reduce-only containment before promotion can remain eligible",
    }

    ok = bool(paper.get("ok", False) and len(history_daily) >= 1)
    overall_status = "ready"
    if not ok:
        overall_status = "blocked"
    elif weak_sleeves or (pnl_slope is not None and pnl_slope < 0.0) or latest_change < 0.0 or bool(promotion.get("promote_ok") is False):
        overall_status = "needs_work"

    payload = {
        "timestamp_utc": now.isoformat(),
        "schema_version": 2,
        "ok": ok,
        "overall_status": overall_status,
        "history_days_available": (
            max(
                (
                    len(rows)
                    for rows in decay_daily.values()
                    if isinstance(rows, list)
                ),
                default=0,
            )
            if candidate_binding_required
            else len(history_daily)
        ),
        "candidate_binding": {
            "candidate_id": candidate_id,
            "generation": _safe_int(
                evidence_window.get("candidate_generation"),
                0,
            ),
            "cutoff_utc": str(evidence_window.get("candidate_cutoff_utc") or ""),
            "evidence_through_utc": str(
                evidence_window.get("evidence_through_utc") or ""
            ),
            "required": candidate_binding_required,
            "bound": candidate_bound,
            "mismatch_rows_excluded": candidate_binding_mismatches,
            "series_scope": edge_decay_contract["evidence_scope"],
        },
        "active_sleeves": active_sleeves,
        "weak_sleeve_count": len(weak_sleeves),
        "weak_sleeves": weak_sleeves[:10],
        "latest_change_vs_previous_day": round(latest_change, 6),
        "latest_net_pnl_total": round(latest_net_pnl, 6),
        "pnl_slope": pnl_slope,
        "trailing_periods": trailing_periods,
        "regime_segments": sleeve_regimes[:12],
        "promotion_ready": bool(promotion.get("promote_ok", False)),
        "edge_decay_contract": edge_decay_contract,
        "recommendations": [
            "Refresh or demote sleeves that stay loss-making across consecutive periods instead of letting them quietly dilute training.",
            "Segment decay review by sleeve and regime before promoting threshold or label changes across the full registry.",
        ],
        "source_files": {
            "paper_performance": str(health_root / "paper_performance_latest.json"),
            "promotion_readiness": str(project_root / "governance" / "walk_forward" / "promotion_readiness_latest.json"),
            "paper_profitability_control": str(health_root / "paper_profitability_control_latest.json"),
        },
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a replay/paper decay monitor artifact.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(Path(args.project_root).resolve())
    out_path = Path(args.out_file).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "decay_monitor "
            f"status={payload['overall_status']} "
            f"weak_sleeves={int(payload.get('weak_sleeve_count', 0) or 0)} "
            f"history_days={int(payload.get('history_days_available', 0) or 0)}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
