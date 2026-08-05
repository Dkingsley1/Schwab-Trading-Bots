import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _event_day() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _source_freshness(path: Path, *, now: datetime, max_age_seconds: float) -> dict:
    try:
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        age_seconds = max((now - modified).total_seconds(), 0.0)
        exists = True
    except Exception:
        modified = None
        age_seconds = None
        exists = False
    return {
        "path": str(path),
        "exists": exists,
        "modified_utc": modified.isoformat() if modified is not None else None,
        "age_seconds": round(age_seconds, 3) if age_seconds is not None else None,
        "max_age_seconds": float(max_age_seconds),
        "fresh": bool(exists and age_seconds is not None and age_seconds <= max(float(max_age_seconds), 0.0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Portfolio-level risk ledger across sleeves.")
    parser.add_argument("--allocator", default=str(PROJECT_ROOT / "governance" / "allocator" / "sleeve_allocator_latest.json"))
    parser.add_argument("--one-numbers", default=str(PROJECT_ROOT / "exports" / "one_numbers" / "one_numbers_summary.json"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "governance" / "risk" / "portfolio_risk_latest.json"))
    parser.add_argument("--allocator-max-age-seconds", type=float, default=3600.0)
    parser.add_argument("--one-numbers-max-age-seconds", type=float, default=21600.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    allocator_path = Path(args.allocator)
    one_numbers_path = Path(args.one_numbers)
    alloc = _read_json(allocator_path)
    one = _read_json(one_numbers_path)
    source_freshness = {
        "sleeve_allocator": _source_freshness(
            allocator_path,
            now=now,
            max_age_seconds=max(float(args.allocator_max_age_seconds), 0.0),
        ),
        "one_numbers": _source_freshness(
            one_numbers_path,
            now=now,
            max_age_seconds=max(float(args.one_numbers_max_age_seconds), 0.0),
        ),
    }
    sources_ready = all(row["fresh"] for row in source_freshness.values())

    weights = alloc.get("target_weights") or {}
    gross_budget = _safe_float(alloc.get("gross_risk_budget"), 0.75)

    blocked = _safe_float(one.get("combined_blocked_rate"), 0.0)
    concentration = _safe_float(one.get("symbol_concentration_top3_share"), 0.0)
    drift = _safe_float(one.get("buy_rate_drift_abs"), 0.0)
    dq = _safe_float(one.get("data_quality_score"), 0.0)
    stocks_pnl = _safe_float(one.get("stocks_pnl_proxy"), 0.0)
    crypto_pnl = _safe_float(one.get("crypto_pnl_proxy"), 0.0)

    risk_score = (
        max(0.0, 100.0 - dq) * 0.40
        + blocked * 100.0 * 0.25
        + concentration * 100.0 * 0.20
        + min(drift * 100.0, 100.0) * 0.15
    )

    if risk_score >= 55:
        risk_level = "high"
    elif risk_score >= 30:
        risk_level = "medium"
    else:
        risk_level = "low"

    gross_exposure_cap = max(0.20, min(gross_budget, 1.00))
    if risk_level == "high":
        gross_exposure_cap *= 0.70
    elif risk_level == "medium":
        gross_exposure_cap *= 0.85

    sleeve_caps = {k: round(float(v) * gross_exposure_cap, 6) for k, v in weights.items()}

    payload = {
        "timestamp_utc": now.isoformat(),
        "schema_version": 2,
        "ok": True,
        "overall_status": "ready" if sources_ready else "degraded",
        "risk_level": risk_level,
        "risk_score": round(risk_score, 4),
        "metrics": {
            "data_quality_score": dq,
            "combined_blocked_rate": blocked,
            "symbol_concentration_top3_share": concentration,
            "buy_rate_drift_abs": drift,
            "stocks_pnl_proxy": stocks_pnl,
            "crypto_pnl_proxy": crypto_pnl,
        },
        "limits": {
            "gross_exposure_cap": round(gross_exposure_cap, 6),
            "sleeve_exposure_caps": sleeve_caps,
            "max_single_symbol_share": 0.20 if risk_level == "low" else (0.15 if risk_level == "medium" else 0.10),
            "max_intraday_turnover": 1.20 if risk_level == "low" else (0.90 if risk_level == "medium" else 0.60),
        },
        "input_freshness": {
            "sources_ready": sources_ready,
            "sources": source_freshness,
            "stale_sources": sorted(name for name, row in source_freshness.items() if not row["fresh"]),
            "fresh_wrapper_timestamp_does_not_override_stale_sources": True,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    events = PROJECT_ROOT / "governance" / "risk" / f"portfolio_risk_events_{_event_day()}.jsonl"
    events.parent.mkdir(parents=True, exist_ok=True)
    with events.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(f"risk_ledger_ok=True risk_level={risk_level} risk_score={risk_score:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
