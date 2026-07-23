import argparse
import glob
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_ts(raw: Any) -> datetime | None:
    s = str(raw or "").strip().replace("Z", "+00:00")
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _calibration_cutoff_utc() -> datetime | None:
    raw = (
        os.getenv("PAPER_EXECUTION_CALIBRATION_MIN_TIMESTAMP_UTC", "").strip()
        or os.getenv("PAPER_EXECUTION_REALISTIC_FILL_CUTOFF_UTC", "").strip()
    )
    return _parse_ts(raw)


def _is_synthetic_guard_row(row: Dict[str, Any]) -> bool:
    strategy = str(row.get("strategy") or "").strip().lower()
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    bot_id = str(metadata.get("bot_id") or row.get("bot_id") or "").strip().lower()
    if strategy.startswith(("paper_guard_test", "paper_guard_block_test")):
        return True
    return bot_id == "test_bot" and strategy.startswith("paper_")


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _bps(fill: float, expected: float, action: str) -> float:
    if fill <= 0 or expected <= 0:
        return 0.0
    a = str(action or "").upper().strip()
    if a.startswith("BUY"):
        return max(((fill - expected) / expected) * 10000.0, 0.0)
    if a.startswith("SELL"):
        return max(((expected - fill) / expected) * 10000.0, 0.0)
    return abs((fill - expected) / expected) * 10000.0


def _market_kind_from_symbol(symbol: Any) -> str:
    text = str(symbol or "").strip().upper()
    if "-" in text:
        return "crypto"
    return "equities"


def _record_group(group: Dict[str, Any], observed_bps: float, expected_bps: float, abs_error_bps: float) -> None:
    group["samples"] = int(group.get("samples", 0)) + 1
    group["observed_sum"] = float(group.get("observed_sum", 0.0)) + float(observed_bps)
    group["expected_sum"] = float(group.get("expected_sum", 0.0)) + float(expected_bps)
    group["abs_error_sum"] = float(group.get("abs_error_sum", 0.0)) + float(abs_error_bps)
    vals = group.setdefault("abs_error_values", [])
    if isinstance(vals, list):
        vals.append(float(abs_error_bps))


def _finalize_group(group: Dict[str, Any]) -> Dict[str, Any]:
    samples = max(int(group.get("samples", 0)), 0)
    errors = sorted(float(v) for v in group.get("abs_error_values", []) if float(v) >= 0.0)
    observed_mean = (float(group.get("observed_sum", 0.0)) / samples) if samples > 0 else 0.0
    expected_mean = (float(group.get("expected_sum", 0.0)) / samples) if samples > 0 else 0.0
    mae = (float(group.get("abs_error_sum", 0.0)) / samples) if samples > 0 else 0.0
    p95 = errors[min(max(int(0.95 * len(errors)) - 1, 0), len(errors) - 1)] if errors else 0.0
    recommended_scale = 1.0
    if expected_mean > 0.0:
        recommended_scale = min(max(observed_mean / expected_mean, 0.25), 1.75)
    return {
        "samples": samples,
        "mean_observed_slippage_bps": round(float(observed_mean), 6),
        "mean_expected_slippage_bps": round(float(expected_mean), 6),
        "mean_bias_bps": round(float(observed_mean - expected_mean), 6),
        "mae_bps": round(float(mae), 6),
        "p95_bps": round(float(p95), 6),
        "recommended_slippage_scale": round(float(recommended_scale), 6),
    }


def _bucket_start(ts: datetime, bucket_hours: int) -> datetime:
    hours = max(int(bucket_hours), 1)
    bucket_hour = int(ts.hour // hours) * hours
    return ts.replace(hour=bucket_hour, minute=0, second=0, microsecond=0)


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in items:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Paper execution calibration drift report.")
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--bucket-hours", type=int, default=1)
    ap.add_argument("--max-mae-bps", type=float, default=35.0)
    ap.add_argument("--out-file", default=str(PROJECT_ROOT / "governance" / "health" / "paper_execution_calibration_latest.json"))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    since = datetime.now(timezone.utc) - timedelta(hours=max(int(args.hours), 1))
    cutoff_utc = _calibration_cutoff_utc()
    if cutoff_utc is not None and cutoff_utc > since:
        since = cutoff_utc
    vals: list[float] = []
    observed_vals: list[float] = []
    expected_vals: list[float] = []
    files_scanned = 0
    skipped_before_cutoff = 0
    skipped_synthetic_guard = 0
    by_market_kind: Dict[str, Dict[str, Any]] = {}
    by_profile: Dict[str, Dict[str, Any]] = {}
    by_symbol: Dict[str, Dict[str, Any]] = {}
    by_bucket: Dict[str, Dict[str, Any]] = {}
    for raw in sorted(glob.glob(str(PROJECT_ROOT / "exports" / "trade_logs" / "**" / "paper_trades_*.jsonl"), recursive=True)):
        files_scanned += 1
        p = Path(raw)
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        row = json.loads(s)
                    except Exception:
                        continue
                    if not isinstance(row, dict):
                        continue
                    ts = _parse_ts(row.get("timestamp_utc"))
                    if ts is None or ts < since:
                        if cutoff_utc is not None and ts is not None and ts < cutoff_utc:
                            skipped_before_cutoff += 1
                        continue
                    if _is_synthetic_guard_row(row):
                        skipped_synthetic_guard += 1
                        continue
                    action = row.get("action")
                    fill = float(row.get("fill_price", 0.0) or 0.0)
                    exp = float(row.get("expected_fill_price", 0.0) or 0.0)
                    ref = float(row.get("reference_price", row.get("intended_price", 0.0)) or 0.0)
                    model_bps = float(row.get("expected_slippage_bps", 0.0) or 0.0)
                    if fill <= 0.0 or exp <= 0.0 or ref <= 0.0:
                        continue
                    observed_bps = _bps(fill, ref, action)
                    expected_bps = model_bps if model_bps > 0.0 else _bps(exp, ref, action)
                    abs_error = abs(observed_bps - expected_bps)
                    vals.append(abs_error)
                    observed_vals.append(observed_bps)
                    expected_vals.append(expected_bps)

                    market_kind = _market_kind_from_symbol(row.get("symbol"))
                    profile = str(((row.get("metadata") or {}).get("source_profile") or "default")).strip().lower() or "default"
                    symbol = str(row.get("symbol") or "").strip().upper() or "UNKNOWN"

                    _record_group(by_market_kind.setdefault(market_kind, {}), observed_bps, expected_bps, abs_error)
                    _record_group(by_profile.setdefault(profile, {}), observed_bps, expected_bps, abs_error)
                    _record_group(by_symbol.setdefault(symbol, {}), observed_bps, expected_bps, abs_error)
                    bucket_key = _bucket_start(ts, int(args.bucket_hours)).isoformat()
                    _record_group(by_bucket.setdefault(bucket_key, {}), observed_bps, expected_bps, abs_error)
        except Exception:
            continue

    vals.sort()
    n = len(vals)
    mae = (sum(vals) / n) if n > 0 else 0.0
    p95 = vals[min(max(int(0.95 * n) - 1, 0), n - 1)] if n > 0 else 0.0
    observed_mean = (sum(observed_vals) / len(observed_vals)) if observed_vals else 0.0
    expected_mean = (sum(expected_vals) / len(expected_vals)) if expected_vals else 0.0

    failed = []
    if n > 0 and mae > float(args.max_mae_bps):
        failed.append("mae_bps")

    finalized_market_kind = {key: _finalize_group(group) for key, group in sorted(by_market_kind.items())}
    finalized_profile = {key: _finalize_group(group) for key, group in sorted(by_profile.items())}
    finalized_symbol_rows = [
        {"symbol": key, **_finalize_group(group)}
        for key, group in sorted(by_symbol.items(), key=lambda item: (-int(item[1].get("samples", 0)), item[0]))
    ]
    drift_series = [
        {"bucket_start_utc": key, **_finalize_group(group)}
        for key, group in sorted(by_bucket.items())
    ]

    recommendations = {
        "env": {
            "EXEC_SIM_SLIPPAGE_SCALE_CRYPTO": float(finalized_market_kind.get("crypto", {}).get("recommended_slippage_scale", 1.0)),
            "EXEC_SIM_SLIPPAGE_SCALE_EQUITIES": float(finalized_market_kind.get("equities", {}).get("recommended_slippage_scale", 1.0)),
        }
    }
    worst_profile = max(
        (
            {"profile": key, **value}
            for key, value in finalized_profile.items()
        ),
        key=lambda row: (float(row.get("mae_bps", 0.0)), float(abs(row.get("mean_bias_bps", 0.0)))),
        default={},
    )
    worst_symbol = finalized_symbol_rows[0] if finalized_symbol_rows else {}
    top_actions = []
    if n > 0 and mae > float(args.max_mae_bps):
        top_actions.append("tighten execution simulator slippage scales until realized mean absolute error returns below the guardrail")
    if worst_profile and _safe_float(worst_profile.get("mae_bps"), 0.0) > float(args.max_mae_bps) * 0.75:
        top_actions.append(f"prioritize profile-level recalibration for {str(worst_profile.get('profile') or 'default')}")
    if worst_symbol and _safe_float(worst_symbol.get("mae_bps"), 0.0) > float(args.max_mae_bps):
        top_actions.append(f"review symbol-specific fill assumptions for {str(worst_symbol.get('symbol') or 'UNKNOWN')}")
    if len(drift_series) >= 2:
        recent_bias = _safe_float(drift_series[-1].get("mean_bias_bps"), 0.0)
        prior_bias = _safe_float(drift_series[-2].get("mean_bias_bps"), 0.0)
        if abs(recent_bias) > abs(prior_bias) + 5.0:
            top_actions.append("recent slippage drift worsened versus the prior bucket, so treat execution realism as a live risk control rather than a static calibration")

    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": "ready" if len(failed) == 0 else "needs_tuning",
        "ok": len(failed) == 0,
        "failed_checks": failed,
        "lookback_hours": int(args.hours),
        "bucket_hours": max(int(args.bucket_hours), 1),
        "files_scanned": int(files_scanned),
        "samples": int(n),
        "calibration_window": {
            "cutoff_utc": cutoff_utc.isoformat() if cutoff_utc is not None else "",
            "reset_active": cutoff_utc is not None,
            "skipped_before_cutoff": int(skipped_before_cutoff),
            "skipped_synthetic_guard_rows": int(skipped_synthetic_guard),
            "policy": "when realistic paper fills are enabled, pre-fix perfect-fill samples are excluded from readiness blocking",
        },
        "metrics": {
            "mae_bps": round(float(mae), 6),
            "p95_bps": round(float(p95), 6),
            "mean_observed_slippage_bps": round(float(observed_mean), 6),
            "mean_expected_slippage_bps": round(float(expected_mean), 6),
            "mean_bias_bps": round(float(observed_mean - expected_mean), 6),
        },
        "thresholds": {"max_mae_bps": float(args.max_mae_bps)},
        "by_market_kind": finalized_market_kind,
        "by_profile": finalized_profile,
        "top_symbols": finalized_symbol_rows[:10],
        "drift_series": drift_series[-48:],
        "line_graph": {
            "x_key": "bucket_start_utc",
            "series": [
                {"key": "mean_observed_slippage_bps", "label": "Observed Slippage"},
                {"key": "mean_expected_slippage_bps", "label": "Expected Slippage"},
                {"key": "mae_bps", "label": "Absolute Error"},
            ],
            "points": drift_series[-48:],
        },
        "top_actions": _ordered_unique(top_actions),
        "recommendations": recommendations,
    }

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(out, ensure_ascii=True))
    else:
        print(f"paper_execution_calibration_ok={int(out['ok'])} mae_bps={out['metrics']['mae_bps']:.4f}/{float(args.max_mae_bps):.4f}")
    return 0 if out["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
