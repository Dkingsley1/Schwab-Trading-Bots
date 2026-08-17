#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, load_json, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, write_payload


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "profitability_evidence_firewall_v1.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "research" / "profitability_benchmark_hurdle_latest.json"


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        value = float(raw)
    except Exception:
        return float(default)
    return value if math.isfinite(value) else float(default)


def _resolve(project_root: Path, raw: Any) -> Path:
    path = Path(str(raw or "")).expanduser()
    return path if path.is_absolute() else project_root / path


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            try:
                row = json.loads(raw)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _compound(returns_bps: list[float]) -> float:
    value = 1.0
    for returns in returns_bps:
        value *= 1.0 + returns / 10_000.0
    return value - 1.0


def _max_drawdown(returns_bps: list[float]) -> float:
    equity = 1.0
    peak = 1.0
    maximum = 0.0
    for returns in returns_bps:
        equity *= 1.0 + returns / 10_000.0
        peak = max(peak, equity)
        maximum = max(maximum, (peak - equity) / max(peak, 1e-12))
    return maximum


def build_payload(project_root: Path = PROJECT_ROOT, *, config_path: Path | None = None) -> dict[str, Any]:
    config = load_json(config_path or project_root / "config" / DEFAULT_CONFIG_PATH.name)
    policy = _as_dict(config.get("benchmark_hurdle"))
    validator = load_json(project_root / "governance" / "health" / "profitability_independent_validator_latest.json")
    capture_path = _resolve(
        project_root,
        policy.get("capture_artifact") or "governance/research/profitability_benchmark_capture_latest.json",
    )
    capture = load_json(capture_path)
    candidate = _as_dict(validator.get("candidate_binding"))
    active_rows = _as_dict(validator.get("recomputed")).get("daily")
    active_rows = active_rows if isinstance(active_rows, list) else []
    active_by_day = {
        str(row.get("day_utc") or ""): _safe_float(row.get("active_return_bps"), float("nan"))
        for row in active_rows
        if isinstance(row, dict) and str(row.get("day_utc") or "")
    }
    active_by_day = {day: value for day, value in active_by_day.items() if math.isfinite(value)}
    series_path = _resolve(project_root, policy.get("series"))
    benchmark_rows = _load_jsonl(series_path)
    benchmark_by_day: dict[str, dict[str, float]] = {}
    rejected_candidate_rows = 0
    rejected_unbound_rows = 0
    for row in benchmark_rows:
        day = str(row.get("day_utc") or row.get("day") or "").strip()
        if not day:
            continue
        row_candidate = str(row.get("candidate_id") or "").strip()
        if row_candidate != str(candidate.get("candidate_id") or ""):
            rejected_candidate_rows += 1
            continue
        if not bool(row.get("candidate_full_session", False)):
            rejected_unbound_rows += 1
            continue
        passive = _safe_float(row.get("passive_return_bps"), float("nan"))
        cash = _safe_float(row.get("cash_return_bps"), float("nan"))
        if not math.isfinite(passive):
            continue
        if not math.isfinite(cash):
            cash = ((1.0 + _safe_float(policy.get("cash_annual_rate"), 0.04)) ** (1.0 / 252.0) - 1.0) * 10_000.0
        benchmark_by_day[day] = {"passive_return_bps": passive, "cash_return_bps": cash}
    common_days = sorted(set(active_by_day) & set(benchmark_by_day))
    active = [active_by_day[day] for day in common_days]
    passive = [benchmark_by_day[day]["passive_return_bps"] for day in common_days]
    cash = [benchmark_by_day[day]["cash_return_bps"] for day in common_days]
    active_return = _compound(active) if active else 0.0
    passive_return = _compound(passive) if passive else 0.0
    cash_return = _compound(cash) if cash else 0.0
    active_drawdown = _max_drawdown(active) if active else 0.0
    passive_drawdown = _max_drawdown(passive) if passive else 0.0
    minimum_days = max(int(_safe_float(policy.get("minimum_common_days"), 30)), 1)
    minimum_excess = _safe_float(policy.get("minimum_excess_return_bps"), 0.0) / 10_000.0
    drawdown_ratio_ceiling = _safe_float(policy.get("maximum_drawdown_ratio_to_passive"), 1.0)
    return_hurdle = active_return > max(passive_return, cash_return) + minimum_excess
    drawdown_hurdle = active_drawdown <= max(passive_drawdown, 1e-9) * drawdown_ratio_ceiling
    evidence_ready = bool(
        validator.get("evidence_ready", False)
        and len(common_days) >= minimum_days
        and return_hurdle
        and drawdown_hurdle
    )
    blockers = []
    if not validator.get("evidence_ready", False):
        blockers.append("independent_accounting_validation_pending")
    if not series_path.is_file():
        blockers.append("point_in_time_benchmark_series_pending")
    if len(common_days) < minimum_days:
        blockers.append("minimum_common_benchmark_days_pending")
    if len(common_days) >= minimum_days and not return_hurdle:
        blockers.append("active_return_does_not_clear_cash_and_passive_hurdle")
    if len(common_days) >= minimum_days and not drawdown_hurdle:
        blockers.append("active_drawdown_exceeds_passive_hurdle")
    implementation_ready = bool(
        policy
        and policy.get("series")
        and policy.get("artifact")
        and policy.get("capture_artifact")
        and _as_dict(policy.get("capture"))
    )
    evidence_ready = bool(evidence_ready and implementation_ready)
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": implementation_ready,
        "overall_status": "ready" if evidence_ready else "evidence_pending",
        "implementation_ready": implementation_ready,
        "evidence_ready": evidence_ready,
        "candidate_binding": candidate,
        "series_path": str(series_path),
        "capture_artifact_path": str(capture_path),
        "capture": capture,
        "source_row_count": len(benchmark_rows),
        "rejected_candidate_row_count": rejected_candidate_rows,
        "rejected_unbound_row_count": rejected_unbound_rows,
        "common_day_count": len(common_days),
        "common_days": common_days,
        "metrics": {
            "active_compound_return": round(active_return, 10),
            "passive_compound_return": round(passive_return, 10),
            "cash_compound_return": round(cash_return, 10),
            "active_excess_over_passive": round(active_return - passive_return, 10),
            "active_excess_over_cash": round(active_return - cash_return, 10),
            "active_max_drawdown": round(active_drawdown, 10),
            "passive_max_drawdown": round(passive_drawdown, 10),
        },
        "checks": {
            "minimum_common_days": len(common_days) >= minimum_days,
            "return_hurdle": return_hurdle,
            "drawdown_hurdle": drawdown_hurdle,
        },
        "thresholds": policy,
        "blockers": blockers,
        "control_contract": {
            "cash_and_passive_hurdles_required": True,
            "candidate_binding_enforced": True,
            "mid_session_candidate_freeze_days_rejected": True,
            "point_in_time_series_required": True,
            "automatic_broker_native_daily_capture_required": True,
            "drawdown_cannot_be_traded_for_cosmetic_return": True,
            "missing_benchmark_data_fails_closed": True,
            "live_execution_authority": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare candidate-bound paper returns with cash and passive benchmarks.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=Path("config") / DEFAULT_CONFIG_PATH.name)
    parser.add_argument("--out-file", type=Path, default=Path("governance/research") / DEFAULT_OUT_PATH.name)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    out_path = args.out_file if args.out_file.is_absolute() else project_root / args.out_file
    payload = build_payload(project_root, config_path=config_path)
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "profitability_benchmark_hurdle "
            f"status={payload['overall_status']} common_days={payload['common_day_count']}"
        )
    return 0 if payload.get("ok", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
