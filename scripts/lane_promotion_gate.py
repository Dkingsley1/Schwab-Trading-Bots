import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN_FILE = PROJECT_ROOT / "governance" / "walk_forward" / "walk_forward_latest.json"
DEFAULT_REGISTRY_FILE = PROJECT_ROOT / "master_bot_registry.json"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


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


def _classify_lane(bot_id: str) -> str:
    bid = str(bot_id or "").strip().lower()
    if any(tok in bid for tok in ("options", "greek", "iv_", "vol_surface", "put_call")):
        return "options"
    if any(tok in bid for tok in ("futures", "funding", "basis", "order_book", "open_interest", "term_structure")):
        return "futures"
    if any(tok in bid for tok in ("long_term", "dividend_quality_compounder", "dividend_yield_trap_avoidance")):
        return "long_term"
    if any(tok in bid for tok in ("intraday", "scalp", "open_close", "ultrafast", "day_trade", "daytrading")):
        return "day"
    if any(tok in bid for tok in ("swing", "position_1m_3m", "1w_3w", "2d_5d")):
        return "swing"
    return "equities"


def _parse_lane_float_caps(raw: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for part in (raw or "").split(","):
        seg = part.strip()
        if (not seg) or (":" not in seg):
            continue
        k, v = seg.split(":", 1)
        key = k.strip().lower()
        if not key:
            continue
        try:
            out[key] = max(float(v.strip()), 0.0)
        except Exception:
            continue
    return out


def _lane_row() -> Dict[str, Any]:
    return {
        "considered": 0,
        "failed": 0,
        "raw_failed": 0,
        "severe_overfit": 0,
        "trading_quality_sum": 0.0,
        "fail_examples": [],
        "near_pass_examples": [],
        "pass_examples": [],
    }


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Lane-aware promotion gate from walk-forward metrics.")
    parser.add_argument("--in-file", default=str(DEFAULT_IN_FILE))
    parser.add_argument("--registry-file", default=str(DEFAULT_REGISTRY_FILE))
    parser.add_argument("--min-forward-mean", type=float, default=float(os.getenv("LANE_PROMOTION_MIN_FORWARD_MEAN", "0.53")))
    parser.add_argument("--min-delta", type=float, default=float(os.getenv("LANE_PROMOTION_MIN_DELTA", "-0.01")))
    parser.add_argument("--min-trading-quality-score", type=float, default=float(os.getenv("LANE_PROMOTION_MIN_TRADING_QUALITY_SCORE", "0.52")))
    parser.add_argument("--max-overfit-gap", type=float, default=float(os.getenv("LANE_PROMOTION_MAX_OVERFIT_GAP", "0.10")))
    parser.add_argument(
        "--max-fail-share-by-lane",
        default=os.getenv(
            "LANE_PROMOTION_MAX_FAIL_SHARE_BY_LANE",
            "equities:0.28,day:0.32,swing:0.30,options:0.35,futures:0.35,long_term:0.25,default:0.30",
        ),
    )
    parser.add_argument("--min-runs-per-bot", type=int, default=int(os.getenv("LANE_PROMOTION_MIN_RUNS", "12")))
    parser.add_argument("--min-considered-per-lane", type=int, default=int(os.getenv("LANE_PROMOTION_MIN_CONSIDERED_PER_LANE", "4")))
    parser.add_argument("--min-covered-lanes", type=int, default=int(os.getenv("LANE_PROMOTION_MIN_COVERED_LANES", "3")))
    parser.add_argument("--near-pass-forward-slack", type=float, default=float(os.getenv("LANE_PROMOTION_NEAR_PASS_FORWARD_SLACK", "0.025")))
    parser.add_argument("--near-pass-delta-slack", type=float, default=float(os.getenv("LANE_PROMOTION_NEAR_PASS_DELTA_SLACK", "0.015")))
    parser.add_argument("--near-pass-min-extra-runs", type=int, default=int(os.getenv("LANE_PROMOTION_NEAR_PASS_MIN_EXTRA_RUNS", "2")))
    parser.add_argument("--near-pass-min-tq-cushion", type=float, default=float(os.getenv("LANE_PROMOTION_NEAR_PASS_MIN_TQ_CUSHION", "0.06")))
    parser.add_argument(
        "--require-active-registry",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("LANE_PROMOTION_REQUIRE_ACTIVE_REGISTRY", "1").strip() == "1",
    )
    parser.add_argument(
        "--include-infrastructure",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("LANE_PROMOTION_INCLUDE_INFRASTRUCTURE", "0").strip() == "1",
    )
    parser.add_argument("--out-file", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "lane_promotion_gate_latest.json"))
    parser.add_argument("--json", action="store_true", help="Print JSON payload.")
    args = parser.parse_args()

    in_path = Path(args.in_file)
    if not in_path.exists():
        raise SystemExit(f"walk-forward file missing: {in_path}")

    payload = _load_json(in_path)
    bots = payload.get("bots", {}) if isinstance(payload, dict) else {}

    registry_path = Path(args.registry_file)
    use_registry_filter = _should_use_registry_filter(in_path, registry_path)
    registry_rows = _load_registry_rows(registry_path) if use_registry_filter else {}

    lane_fail_caps = _parse_lane_float_caps(str(args.max_fail_share_by_lane))
    default_lane_fail_cap = float(lane_fail_caps.get("default", 0.30))

    lane_stats: Dict[str, Dict[str, Any]] = {}
    total_considered = 0
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

        lane = _classify_lane(str(bot_id))
        stats = lane_stats.setdefault(lane, _lane_row())

        total_considered += 1
        stats["considered"] += 1

        fwd = _f(row.get("forward_mean"), 0.0)
        delta = _f(row.get("delta"), 0.0)
        tq = _f(row.get("trading_quality_score"), 0.0)
        overfit_gap = _f(row.get("overfit_gap"), _f(row.get("train_mean"), 0.0) - fwd)
        stats["trading_quality_sum"] += tq

        gate_fwd = fwd >= float(args.min_forward_mean)
        gate_delta = delta >= float(args.min_delta)
        gate_tq = tq >= float(args.min_trading_quality_score)
        gate_overfit = overfit_gap <= float(args.max_overfit_gap)

        ok = gate_fwd and gate_delta and gate_tq and gate_overfit
        row_out = {
            "bot_id": str(bot_id),
            "runs": int(runs),
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

        if ok:
            if len(stats["pass_examples"]) < 5:
                stats["pass_examples"].append(row_out)
        else:
            stats["raw_failed"] += 1
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
                if len(stats["near_pass_examples"]) < 12:
                    stats["near_pass_examples"].append({**row_out, "near_pass_reason": near_pass_reason})
            else:
                stats["failed"] += 1
                if len(stats["fail_examples"]) < 12:
                    stats["fail_examples"].append(row_out)

        if overfit_gap > (float(args.max_overfit_gap) * 1.5):
            stats["severe_overfit"] += 1

    lane_payload: Dict[str, Any] = {}
    observed_lanes = sorted(lane_stats.keys())
    effective_min_covered_lanes = min(max(int(args.min_covered_lanes), 1), max(len(observed_lanes), 1)) if total_considered > 0 else max(int(args.min_covered_lanes), 1)
    covered_lanes = 0
    passing_covered_lanes = 0

    for lane in sorted(lane_stats.keys()):
        stats = lane_stats[lane]
        considered = int(stats.get("considered", 0) or 0)
        failed = int(stats.get("failed", 0) or 0)
        raw_failed = int(stats.get("raw_failed", 0) or 0)
        severe_overfit = int(stats.get("severe_overfit", 0) or 0)
        fail_share = float(failed) / max(considered, 1)
        raw_fail_share = float(raw_failed) / max(considered, 1)
        severe_overfit_share = float(severe_overfit) / max(considered, 1)
        mean_tq = float(stats.get("trading_quality_sum", 0.0) or 0.0) / max(considered, 1)
        lane_fail_cap = float(lane_fail_caps.get(lane, default_lane_fail_cap))
        effective_min_considered = min(max(int(args.min_considered_per_lane), 1), max(considered, 1))
        coverage_ok = considered >= effective_min_considered
        promote_ok = coverage_ok and (fail_share <= lane_fail_cap)
        if coverage_ok:
            covered_lanes += 1
            if promote_ok:
                passing_covered_lanes += 1

        lane_payload[lane] = {
            "considered_bots": considered,
            "failed_bots": failed,
            "fail_share": round(fail_share, 6),
            "raw_failed_bots": raw_failed,
            "raw_fail_share": round(raw_fail_share, 6),
            "near_pass_bots": len(stats.get("near_pass_examples", [])),
            "max_fail_share": round(lane_fail_cap, 6),
            "severe_overfit_bots": severe_overfit,
            "severe_overfit_share": round(severe_overfit_share, 6),
            "mean_trading_quality_score": round(mean_tq, 6),
            "coverage_ok": bool(coverage_ok),
            "promote_ok": bool(promote_ok),
            "effective_min_considered_bots": int(effective_min_considered),
            "fail_examples": list(stats.get("fail_examples", []))[:8],
            "near_pass_examples": list(stats.get("near_pass_examples", []))[:8],
            "pass_examples": list(stats.get("pass_examples", []))[:5],
        }

    coverage_ok = covered_lanes >= effective_min_covered_lanes
    lane_promote_ok = all(bool(v.get("promote_ok", False)) for v in lane_payload.values() if bool(v.get("coverage_ok", False)))
    promote_ok = bool(coverage_ok and lane_promote_ok)

    ranked_lanes = sorted(
        [
            {
                "lane": lane,
                "fail_share": float(v["fail_share"]) if v.get("fail_share") is not None else 1.0,
                "failed_bots": int(v.get("failed_bots", 0) or 0),
                "considered_bots": int(v.get("considered_bots", 0) or 0),
            }
            for lane, v in lane_payload.items()
        ],
        key=lambda r: (r["fail_share"], r["failed_bots"]),
        reverse=True,
    )

    top_lane_failures: List[Dict[str, Any]] = []
    for lane in sorted(lane_payload.keys()):
        lane_fail_examples = lane_payload[lane].get("fail_examples", [])
        if lane_fail_examples:
            top_lane_failures.append(
                {
                    "lane": lane,
                    "bot_ids": [str(x.get("bot_id", "")) for x in lane_fail_examples[:5] if str(x.get("bot_id", ""))],
                }
            )

    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "promote_ok": bool(promote_ok),
        "coverage_ok": bool(coverage_ok),
        "covered_lanes": int(covered_lanes),
        "passing_covered_lanes": int(passing_covered_lanes),
        "considered_bots": int(total_considered),
        "thresholds": {
            "min_forward_mean": float(args.min_forward_mean),
            "min_delta": float(args.min_delta),
            "min_trading_quality_score": float(args.min_trading_quality_score),
            "max_overfit_gap": float(args.max_overfit_gap),
            "min_runs_per_bot": int(args.min_runs_per_bot),
            "min_considered_per_lane": int(args.min_considered_per_lane),
            "min_covered_lanes": int(args.min_covered_lanes),
            "max_fail_share_by_lane": {k: float(v) for k, v in lane_fail_caps.items()},
            "near_pass_forward_slack": float(args.near_pass_forward_slack),
            "near_pass_delta_slack": float(args.near_pass_delta_slack),
            "near_pass_min_extra_runs": int(args.near_pass_min_extra_runs),
            "near_pass_min_tq_cushion": float(args.near_pass_min_tq_cushion),
        },
        "effective_thresholds": {
            "min_covered_lanes": int(effective_min_covered_lanes),
        },
        "registry_filter": {
            "enabled": bool(registry_rows),
            "require_active_registry": bool(args.require_active_registry),
            "include_infrastructure": bool(args.include_infrastructure),
        },
        "excluded_counts": excluded_counts,
        "lanes": lane_payload,
        "ranked_lanes": ranked_lanes,
        "canary_recommendations": top_lane_failures,
    }

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(out, ensure_ascii=True))
    else:
        print(json.dumps(out, ensure_ascii=True))

    return 0 if promote_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
