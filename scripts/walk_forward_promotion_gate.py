import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN_FILE = PROJECT_ROOT / "governance" / "walk_forward" / "walk_forward_latest.json"
DEFAULT_REGISTRY_FILE = PROJECT_ROOT / "master_bot_registry.json"
OPS_THRESHOLDS_FILE = PROJECT_ROOT / "governance" / "ops_thresholds.json"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _ops_thresholds() -> dict[str, Any]:
    payload = _load_json(OPS_THRESHOLDS_FILE)
    gates = payload.get("promotion_gates") if isinstance(payload.get("promotion_gates"), dict) else {}
    return gates.get("promotion_gate") if isinstance(gates.get("promotion_gate"), dict) else {}


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _i(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _near_pass_reason(
    *,
    runs: int,
    failed_gates: dict[str, bool],
    forward_mean: float,
    delta: float,
    trading_quality_score: float,
    min_forward_mean: float,
    min_delta: float,
    min_trading_quality_score: float,
    min_runs_per_bot: int,
    forward_slack: float,
    delta_slack: float,
    min_extra_runs: int,
    min_tq_cushion: float,
) -> str:
    failed = [name for name, is_failed in failed_gates.items() if bool(is_failed)]
    if len(failed) != 1:
        return ""
    if int(runs) < max(int(min_runs_per_bot) + int(min_extra_runs), int(min_runs_per_bot)):
        return ""
    if float(trading_quality_score) < (float(min_trading_quality_score) + float(min_tq_cushion)):
        return ""

    failed_gate = failed[0]
    if failed_gate == "forward_mean":
        miss = max(float(min_forward_mean) - float(forward_mean), 0.0)
        if miss <= float(forward_slack):
            return f"forward_mean_within_slack:{round(miss, 6)}"
    if failed_gate == "delta":
        miss = max(float(min_delta) - float(delta), 0.0)
        if miss <= float(delta_slack):
            return f"delta_within_slack:{round(miss, 6)}"
    return ""


def _normalize_registry_row(row: dict[str, Any]) -> dict[str, Any]:
    deleted = bool(row.get("deleted_from_rotation", False))
    active = bool(row.get("active", False)) and (not deleted)
    return {
        **row,
        "active": active,
        "deleted_from_rotation": deleted,
        "lifecycle_state": "deleted" if deleted else ("active" if active else "inactive"),
    }


def _manual_quarantine(row: dict[str, Any]) -> bool:
    marker = "active_streak_cap_quarantine_manual_"
    return str(row.get("reason") or "").startswith(marker) or str(row.get("delete_reason") or "").startswith(marker)


def _load_registry_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json(path)
    rows = payload.get("sub_bots") if isinstance(payload.get("sub_bots"), list) else []
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        bot_id = str(row.get("bot_id") or "").strip()
        if not bot_id:
            continue
        out[bot_id] = _normalize_registry_row(row)
    return out


def _should_use_registry_filter(in_file: Path, registry_file: Path) -> bool:
    if not registry_file.exists():
        return False
    custom_input = in_file.resolve() != DEFAULT_IN_FILE.resolve()
    using_default_registry = registry_file.resolve() == DEFAULT_REGISTRY_FILE.resolve()
    return not (custom_input and using_default_registry)


def _registry_gate_allowed(
    bot_id: str,
    registry_rows: dict[str, dict[str, Any]],
    *,
    require_active_registry: bool,
    include_infrastructure: bool,
) -> tuple[bool, str]:
    row = registry_rows.get(bot_id)
    if row is None:
        return True, "missing_registry_row"
    if bool(row.get("deleted_from_rotation", False)):
        return False, "deleted_from_rotation"
    if _manual_quarantine(row):
        return False, "manual_quarantine"
    if require_active_registry and (not bool(row.get("active", False))):
        return False, "inactive"
    if (not include_infrastructure) and str(row.get("bot_role") or "") == "infrastructure_sub_bot":
        return False, "infrastructure_sub_bot"
    return True, "eligible"


def main() -> int:
    defaults = _ops_thresholds()
    parser = argparse.ArgumentParser(description="Promotion gate based on walk-forward + trading quality metrics.")
    parser.add_argument("--in-file", default=str(DEFAULT_IN_FILE))
    parser.add_argument("--registry-file", default=str(DEFAULT_REGISTRY_FILE))
    parser.add_argument("--min-forward-mean", type=float, default=float(os.getenv("PROMOTION_GATE_MIN_FORWARD_MEAN", str(defaults.get("min_forward_mean", 0.53)))))
    parser.add_argument("--min-delta", type=float, default=float(os.getenv("PROMOTION_GATE_MIN_DELTA", str(defaults.get("min_delta", -0.01)))))
    parser.add_argument("--min-trading-quality-score", type=float, default=float(os.getenv("PROMOTION_GATE_MIN_TRADING_QUALITY_SCORE", str(defaults.get("min_trading_quality_score", 0.52)))))
    parser.add_argument("--max-overfit-gap", type=float, default=float(os.getenv("PROMOTION_GATE_MAX_OVERFIT_GAP", str(defaults.get("max_overfit_gap", 0.10)))))
    parser.add_argument("--max-fail-share", type=float, default=float(os.getenv("PROMOTION_GATE_MAX_FAIL_SHARE", str(defaults.get("max_fail_share", 0.25)))))
    parser.add_argument("--max-severe-overfit-share", type=float, default=float(os.getenv("PROMOTION_GATE_MAX_SEVERE_OVERFIT_SHARE", str(defaults.get("max_severe_overfit_share", 0.10)))))
    parser.add_argument("--min-runs-per-bot", type=int, default=int(os.getenv("PROMOTION_GATE_MIN_RUNS", str(defaults.get("min_runs_per_bot", 12)))))
    parser.add_argument("--min-considered-bots", type=int, default=int(os.getenv("PROMOTION_GATE_MIN_CONSIDERED", str(defaults.get("min_considered_bots", 4)))))
    parser.add_argument("--near-pass-forward-slack", type=float, default=float(os.getenv("PROMOTION_GATE_NEAR_PASS_FORWARD_SLACK", "0.025")))
    parser.add_argument("--near-pass-delta-slack", type=float, default=float(os.getenv("PROMOTION_GATE_NEAR_PASS_DELTA_SLACK", "0.015")))
    parser.add_argument("--near-pass-min-extra-runs", type=int, default=int(os.getenv("PROMOTION_GATE_NEAR_PASS_MIN_EXTRA_RUNS", "2")))
    parser.add_argument("--near-pass-min-tq-cushion", type=float, default=float(os.getenv("PROMOTION_GATE_NEAR_PASS_MIN_TQ_CUSHION", "0.06")))
    parser.add_argument(
        "--require-active-registry",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("PROMOTION_GATE_REQUIRE_ACTIVE_REGISTRY", "1" if bool(defaults.get("require_active_registry", True)) else "0").strip() == "1",
    )
    parser.add_argument(
        "--include-infrastructure",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("PROMOTION_GATE_INCLUDE_INFRASTRUCTURE", "1" if bool(defaults.get("include_infrastructure", False)) else "0").strip() == "1",
    )
    parser.add_argument("--out-file", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "promotion_gate_latest.json"))
    args = parser.parse_args()

    in_path = Path(args.in_file)
    if not in_path.exists():
        raise SystemExit(f"walk-forward file missing: {in_path}")

    payload = _load_json(in_path)
    bots = payload.get("bots", {}) if isinstance(payload, dict) else {}

    registry_path = Path(args.registry_file)
    use_registry_filter = _should_use_registry_filter(in_path, registry_path)
    registry_rows = _load_registry_rows(registry_path) if use_registry_filter else {}

    considered = 0
    fails = 0
    raw_fails = 0
    severe_overfit = 0
    tq_sum = 0.0
    fail_reasons = []
    near_pass_examples = []
    excluded_counts: dict[str, int] = {}

    for bot_id, row in bots.items():
        if not isinstance(row, dict):
            continue
        runs = _i(row.get("runs"), 0)
        if runs < int(args.min_runs_per_bot):
            continue
        if str(row.get("status", "")).strip().lower() == "insufficient_runs":
            continue

        if registry_rows:
            eligible, exclude_reason = _registry_gate_allowed(
                str(bot_id),
                registry_rows,
                require_active_registry=bool(args.require_active_registry),
                include_infrastructure=bool(args.include_infrastructure),
            )
            if not eligible:
                excluded_counts[exclude_reason] = excluded_counts.get(exclude_reason, 0) + 1
                continue

        considered += 1
        fwd = _f(row.get("forward_mean"), 0.0)
        delta = _f(row.get("delta"), 0.0)
        tq = _f(row.get("trading_quality_score"), 0.0)
        overfit_gap = _f(row.get("overfit_gap"), _f(row.get("train_mean"), 0.0) - fwd)
        tq_sum += tq

        gate_fwd = fwd >= float(args.min_forward_mean)
        gate_delta = delta >= float(args.min_delta)
        gate_tq = tq >= float(args.min_trading_quality_score)
        gate_overfit = overfit_gap <= float(args.max_overfit_gap)

        ok = gate_fwd and gate_delta and gate_tq and gate_overfit
        row_out = {
            "bot_id": bot_id,
            "runs": runs,
            "forward_mean": round(fwd, 6),
            "delta": round(delta, 6),
            "trading_quality_score": round(tq, 6),
            "overfit_gap": round(overfit_gap, 6),
            "failed_gates": {
                "forward_mean": not gate_fwd,
                "delta": not gate_delta,
                "trading_quality_score": not gate_tq,
                "overfit_gap": not gate_overfit,
            },
        }
        if not ok:
            raw_fails += 1
            near_pass_reason = _near_pass_reason(
                runs=runs,
                failed_gates=row_out["failed_gates"],
                forward_mean=fwd,
                delta=delta,
                trading_quality_score=tq,
                min_forward_mean=float(args.min_forward_mean),
                min_delta=float(args.min_delta),
                min_trading_quality_score=float(args.min_trading_quality_score),
                min_runs_per_bot=int(args.min_runs_per_bot),
                forward_slack=float(args.near_pass_forward_slack),
                delta_slack=float(args.near_pass_delta_slack),
                min_extra_runs=int(args.near_pass_min_extra_runs),
                min_tq_cushion=float(args.near_pass_min_tq_cushion),
            )
            if near_pass_reason:
                near_pass_examples.append({**row_out, "near_pass_reason": near_pass_reason})
            else:
                fails += 1
                fail_reasons.append(row_out)

        if overfit_gap > float(args.max_overfit_gap) * 1.5:
            severe_overfit += 1

    fail_share = fails / max(considered, 1)
    raw_fail_share = raw_fails / max(considered, 1)
    severe_overfit_share = severe_overfit / max(considered, 1)
    coverage_ok = considered >= int(args.min_considered_bots)
    mean_trading_quality_score = tq_sum / max(considered, 1)

    promote_ok = (
        coverage_ok
        and (fail_share <= float(args.max_fail_share))
        and (severe_overfit_share <= float(args.max_severe_overfit_share))
    )

    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "considered_bots": considered,
        "failed_bots": fails,
        "fail_share": round(fail_share, 6),
        "raw_failed_bots": raw_fails,
        "raw_fail_share": round(raw_fail_share, 6),
        "near_pass_bots": len(near_pass_examples),
        "severe_overfit_bots": severe_overfit,
        "severe_overfit_share": round(severe_overfit_share, 6),
        "mean_trading_quality_score": round(mean_trading_quality_score, 6),
        "coverage_ok": coverage_ok,
        "promote_ok": promote_ok,
        "thresholds": {
            "min_forward_mean": float(args.min_forward_mean),
            "min_delta": float(args.min_delta),
            "min_trading_quality_score": float(args.min_trading_quality_score),
            "max_overfit_gap": float(args.max_overfit_gap),
            "max_fail_share": float(args.max_fail_share),
            "max_severe_overfit_share": float(args.max_severe_overfit_share),
            "min_runs_per_bot": int(args.min_runs_per_bot),
            "min_considered_bots": int(args.min_considered_bots),
            "near_pass_forward_slack": float(args.near_pass_forward_slack),
            "near_pass_delta_slack": float(args.near_pass_delta_slack),
            "near_pass_min_extra_runs": int(args.near_pass_min_extra_runs),
            "near_pass_min_tq_cushion": float(args.near_pass_min_tq_cushion),
        },
        "registry_filter": {
            "enabled": bool(registry_rows),
            "require_active_registry": bool(args.require_active_registry),
            "include_infrastructure": bool(args.include_infrastructure),
        },
        "excluded_counts": excluded_counts,
        "fail_examples": fail_reasons[:30],
        "near_pass_examples": near_pass_examples[:30],
    }

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=True))
    return 0 if promote_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
