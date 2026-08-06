#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, load_json, ordered_unique, parse_iso_utc, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, parse_iso_utc, write_payload


DEFAULT_OUT = Path("governance/health/readiness_evidence_accrual_latest.json")
DEFAULT_STATE = Path("governance/runtime/readiness_evidence_accrual_state.json")
DEFAULT_HISTORY = Path("governance/evidence/readiness_evidence_accrual_history.jsonl")
SCHEMA_VERSION = 1


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        value = float(raw)
    except Exception:
        return float(default)
    return value if math.isfinite(value) else float(default)


def _safe_int(raw: Any, default: int = 0) -> int:
    return int(_safe_float(raw, float(default)))


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _resolve(project_root: Path, path: Path) -> Path:
    return path.expanduser() if path.is_absolute() else project_root / path


def _candidate(project_root: Path) -> dict[str, Any]:
    state = load_json(project_root / "governance" / "runtime" / "production_candidate_state.json")
    windows = _as_dict(state.get("scope_windows_started_utc"))
    window_values = [parse_iso_utc(value) for value in windows.values()]
    soak_started = max((value for value in window_values if value is not None), default=None)
    return {
        "candidate_id": str(state.get("candidate_id") or "").strip(),
        "generation": _safe_int(state.get("generation"), 0),
        "soak_started_utc": soak_started.isoformat() if soak_started is not None else "",
        "soak_started": soak_started,
        "bound": bool(str(state.get("candidate_id") or "").strip() and soak_started is not None),
    }


def _runtime_collection_active(project_root: Path) -> tuple[bool, dict[str, Any]]:
    watchdog = load_json(project_root / "governance" / "health" / "process_watchdog_latest.json")
    live_runtime = load_json(project_root / "governance" / "health" / "live_runtime_status_latest.json")
    watchdog_status = str(watchdog.get("overall_status") or watchdog.get("status") or "").strip().lower()
    runtime_status = str(live_runtime.get("overall_status") or live_runtime.get("status") or "").strip().lower()
    process_count = max(
        _safe_int(watchdog.get("active_process_count"), 0),
        _safe_int(_as_dict(watchdog.get("summary")).get("active_process_count"), 0),
    )
    active = bool(watchdog_status in {"ready", "ok", "active"} or runtime_status in {"ready", "ok", "active"} or process_count > 0)
    return active, {
        "watchdog_status": watchdog_status,
        "live_runtime_status": runtime_status,
        "active_process_count": process_count,
    }


def _metric(metric_id: str, label: str, current: float, target: float, *, unit: str, kind: str) -> dict[str, Any]:
    remaining = max(float(target) - float(current), 0.0)
    progress = 1.0 if target <= 0.0 and current >= target else max(0.0, min(float(current) / float(target), 1.0)) if target > 0.0 else 0.0
    return {
        "metric_id": metric_id,
        "label": label,
        "current": round(float(current), 6),
        "target": round(float(target), 6),
        "remaining": round(remaining, 6),
        "progress_ratio": round(progress, 6),
        "unit": unit,
        "kind": kind,
        "complete": bool(current >= target),
    }


def _raw_metrics(project_root: Path, *, candidate: dict[str, Any], now: datetime) -> list[dict[str, Any]]:
    calibration = load_json(project_root / "governance" / "health" / "paper_execution_calibration_latest.json")
    performance = load_json(project_root / "governance" / "health" / "paper_performance_latest.json")
    post_cost = _as_dict(performance.get("post_cost_expectancy"))
    robust = _as_dict(post_cost.get("robust_statistics"))
    robust_thresholds = _as_dict(robust.get("thresholds"))
    promotion = load_json(project_root / "governance" / "health" / "promotion_quality_gate_latest.json")
    promotion_details = _as_dict(promotion.get("details"))
    promotion_scope = _as_dict(promotion_details.get("promotion"))
    rollout = load_json(project_root / "governance" / "health" / "canary_rollout_latest.json")
    rollout_thresholds = _as_dict(rollout.get("thresholds"))
    rollout_canary = _as_dict(rollout.get("canary_statistics"))
    profitability = load_json(project_root / "governance" / "health" / "paper_profitability_control_latest.json")
    target_contract = _as_dict(profitability.get("a_plus_target_contract"))
    target_thresholds = _as_dict(target_contract.get("thresholds"))
    target_current = _as_dict(target_contract.get("current"))
    soak_started = candidate.get("soak_started") if isinstance(candidate.get("soak_started"), datetime) else None
    soak_hours = max((now - soak_started).total_seconds() / 3600.0, 0.0) if soak_started is not None else 0.0
    minimum_cohort_samples = _safe_float(rollout_thresholds.get("minimum_samples_per_cohort"), 400.0)
    minimum_canary_days = _safe_float(rollout_thresholds.get("minimum_independent_days"), 3.0)
    minimum_canary_ess = _safe_float(rollout_thresholds.get("minimum_effective_samples"), 50.0)
    return [
        _metric("soak_elapsed_hours", "Frozen-candidate soak time", soak_hours, 720.0, unit="hours", kind="elapsed_time"),
        _metric("independent_fills", "Independent paper/replay fills", _safe_float(calibration.get("independent_samples")), 100.0, unit="fills", kind="evidence"),
        _metric("post_cost_samples", "Post-cost trade observations", _safe_float(post_cost.get("sample_count")), _safe_float(robust_thresholds.get("minimum_samples"), 30.0), unit="observations", kind="evidence"),
        _metric("post_cost_days", "Independent post-cost days", _safe_float(robust.get("unique_day_count")), _safe_float(robust_thresholds.get("minimum_days"), 7.0), unit="days", kind="evidence"),
        _metric("post_cost_symbols", "Post-cost symbol breadth", _safe_float(robust.get("unique_symbol_count")), _safe_float(robust_thresholds.get("minimum_symbols"), 5.0), unit="symbols", kind="evidence"),
        _metric("post_cost_effective_samples", "Cluster-effective post-cost samples", _safe_float(robust.get("effective_sample_size")), _safe_float(robust_thresholds.get("minimum_effective_samples"), 20.0), unit="effective_samples", kind="evidence"),
        _metric("considered_bots", "Independently considered promotion bots", _safe_float(promotion_scope.get("considered_bots")), _safe_float(promotion_scope.get("min_considered_bots"), 4.0), unit="bots", kind="evidence"),
        _metric("promotion_candidates", "Qualified promotion candidates", float(len(promotion_details.get("promotion_candidate_ids") or [])), _safe_float(promotion_scope.get("min_considered_bots"), 4.0), unit="bots", kind="evidence"),
        _metric("canary_samples", "Candidate canary observations", _safe_float(rollout.get("canary_samples")), minimum_cohort_samples, unit="observations", kind="evidence"),
        _metric("baseline_samples", "Baseline canary observations", _safe_float(rollout.get("baseline_samples")), minimum_cohort_samples, unit="observations", kind="evidence"),
        _metric("canary_independent_days", "Canary independent days", _safe_float(rollout_canary.get("unique_day_count")), minimum_canary_days, unit="days", kind="evidence"),
        _metric("canary_effective_samples", "Canary effective samples", _safe_float(rollout_canary.get("effective_sample_size")), minimum_canary_ess, unit="effective_samples", kind="evidence"),
        _metric("raw_net_pnl", "Raw paper net PnL", _safe_float(target_current.get("net_pnl")), _safe_float(target_thresholds.get("min_net_pnl"), 50000.0), unit="usd", kind="outcome"),
    ]


def _enrich_rates(
    metrics: list[dict[str, Any]],
    *,
    prior_state: dict[str, Any],
    candidate_id: str,
    now: datetime,
    collection_active: bool,
    stall_hours: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prior_candidate = str(prior_state.get("candidate_id") or "")
    prior_timestamp = parse_iso_utc(prior_state.get("timestamp_utc"))
    prior_metrics = prior_state.get("metrics") if isinstance(prior_state.get("metrics"), dict) else {}
    same_candidate = bool(candidate_id and candidate_id == prior_candidate)
    elapsed_hours = max((now - prior_timestamp).total_seconds() / 3600.0, 0.0) if same_candidate and prior_timestamp is not None else 0.0
    next_metric_state: dict[str, Any] = {}
    enriched: list[dict[str, Any]] = []
    for raw in metrics:
        row = dict(raw)
        metric_id = str(row["metric_id"])
        prior = prior_metrics.get(metric_id) if isinstance(prior_metrics.get(metric_id), dict) else {}
        prior_value = _safe_float(prior.get("value"), row["current"])
        delta = float(row["current"]) - prior_value if same_candidate else 0.0
        rate = delta / elapsed_hours if elapsed_hours > 0.0 and delta > 0.0 else None
        unchanged_since = parse_iso_utc(prior.get("unchanged_since_utc")) if same_candidate and delta <= 0.0 else now
        if unchanged_since is None:
            unchanged_since = now
        unchanged_hours = max((now - unchanged_since).total_seconds() / 3600.0, 0.0)
        expected_to_move = bool(row["kind"] in {"evidence", "elapsed_time"} and not row["complete"])
        stalled = bool(collection_active and expected_to_move and unchanged_hours >= max(float(stall_hours), 0.25))
        eta_hours = float(row["remaining"]) / rate if rate is not None and rate > 0.0 and row["remaining"] > 0.0 else 0.0 if row["complete"] else None
        row.update(
            {
                "delta_since_previous": round(delta, 6) if same_candidate else None,
                "rate_per_hour": round(rate, 6) if rate is not None else None,
                "eta_hours": round(eta_hours, 3) if eta_hours is not None else None,
                "eta_available": eta_hours is not None,
                "unchanged_hours": round(unchanged_hours, 3),
                "stalled": stalled,
            }
        )
        next_metric_state[metric_id] = {
            "value": row["current"],
            "unchanged_since_utc": unchanged_since.isoformat(),
        }
        enriched.append(row)
    return enriched, next_metric_state


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _load_history(path: Path, limit: int = 512) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[-max(int(limit), 1):]
    except Exception:
        return []
    for line in lines:
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    state_path: Path = DEFAULT_STATE,
    history_path: Path = DEFAULT_HISTORY,
    apply: bool = False,
    stall_hours: float = 6.0,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = now or datetime.now(timezone.utc)
    effective_state = _resolve(project_root, state_path)
    effective_history = _resolve(project_root, history_path)
    prior_state = load_json(effective_state)
    candidate = _candidate(project_root)
    collection_active, collection = _runtime_collection_active(project_root)
    metrics = _raw_metrics(project_root, candidate=candidate, now=current)
    metrics, metric_state = _enrich_rates(
        metrics,
        prior_state=prior_state,
        candidate_id=str(candidate.get("candidate_id") or ""),
        now=current,
        collection_active=collection_active,
        stall_hours=stall_hours,
    )
    stalled_ids = [str(row["metric_id"]) for row in metrics if row.get("stalled")]
    pending_ids = [str(row["metric_id"]) for row in metrics if not row.get("complete")]
    status = "blocked" if not candidate.get("bound", False) else "stalled" if stalled_ids else "complete" if not pending_ids else "accumulating"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": current.isoformat(),
        "overall_status": status,
        "ok": bool(candidate.get("bound", False) and not stalled_ids),
        "candidate_binding": {key: value for key, value in candidate.items() if key != "soak_started"},
        "collection_active": collection_active,
        "collection_evidence": collection,
        "metric_count": len(metrics),
        "complete_metric_count": sum(1 for row in metrics if row.get("complete")),
        "pending_metric_ids": pending_ids,
        "stalled_metric_ids": stalled_ids,
        "metrics": metrics,
        "control_contract": {
            "eta_requires_observed_positive_rate": True,
            "candidate_change_resets_rate_history": True,
            "stalls_require_active_collection_and_elapsed_window": True,
            "raw_counts_are_not_relabelled_as_independent_evidence": True,
            "live_execution_authority": False,
        },
        "recommended_actions": ordered_unique(
            [
                "investigate the producer or route for stalled evidence metrics" if stalled_ids else "",
                "continue candidate-bound collection for pending evidence metrics" if pending_ids else "",
                "do not infer an ETA until a metric has a positive observed accrual rate",
            ]
        ),
    }
    if apply:
        next_state = {
            "schema_version": SCHEMA_VERSION,
            "timestamp_utc": current.isoformat(),
            "candidate_id": str(candidate.get("candidate_id") or ""),
            "generation": _safe_int(candidate.get("generation"), 0),
            "metrics": metric_state,
        }
        write_payload(effective_state, next_state)
        history = _load_history(effective_history)
        history.append(
            {
                "timestamp_utc": current.isoformat(),
                "candidate_id": str(candidate.get("candidate_id") or ""),
                "overall_status": status,
                "values": {str(row["metric_id"]): row["current"] for row in metrics},
                "stalled_metric_ids": stalled_ids,
            }
        )
        _atomic_write_jsonl(effective_history, history[-512:])
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Track candidate-bound readiness evidence accrual and stalls.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--out-file", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--history-file", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--stall-hours", type=float, default=6.0)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    project_root = args.project_root.expanduser().resolve()
    payload = build_payload(
        project_root,
        state_path=args.state_file,
        history_path=args.history_file,
        apply=bool(args.apply),
        stall_hours=float(args.stall_hours),
    )
    write_payload(_resolve(project_root, args.out_file), payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "readiness_evidence_accrual "
            f"status={payload['overall_status']} complete={payload['complete_metric_count']}/{payload['metric_count']} "
            f"stalled={len(payload['stalled_metric_ids'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
