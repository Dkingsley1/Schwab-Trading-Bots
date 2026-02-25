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


def _normalize(weights: dict[str, float]) -> dict[str, float]:
    s = sum(max(v, 0.0) for v in weights.values())
    if s <= 0:
        n = max(len(weights), 1)
        return {k: 1.0 / n for k in weights}
    return {k: max(v, 0.0) / s for k, v in weights.items()}


def _event_day() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def main() -> int:
    parser = argparse.ArgumentParser(description="Portfolio sleeve allocator across core/aggressive/dividend/bond/crypto.")
    parser.add_argument("--one-numbers", default=str(PROJECT_ROOT / "exports" / "one_numbers" / "one_numbers_summary.json"))
    parser.add_argument("--slo", default=str(PROJECT_ROOT / "governance" / "watchdog" / "sleeve_slo_latest.json"))
    parser.add_argument("--bot-stack", default=str(PROJECT_ROOT / "exports" / "bot_stack_status" / "latest.json"))
    parser.add_argument("--broker", default="schwab", choices=["schwab", "coinbase"])
    parser.add_argument("--out", default=str(PROJECT_ROOT / "governance" / "allocator" / "sleeve_allocator_latest.json"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    one = _read_json(Path(args.one_numbers))
    slo = _read_json(Path(args.slo))
    stack = _read_json(Path(args.bot_stack))

    base = {
        "core": 0.45,
        "aggressive": 0.25,
        "dividend": 0.15,
        "bond": 0.15,
        "crypto": 0.00 if args.broker == "schwab" else 0.25,
    }

    dq = _safe_float(one.get("data_quality_score"), 0.0)
    blocked = _safe_float(one.get("combined_blocked_rate"), 0.0)
    drift = _safe_float(one.get("buy_rate_drift_abs"), 0.0)
    slo_alerts = len(slo.get("alerts") or []) if isinstance(slo.get("alerts"), list) else 0
    bot_status = str(((stack.get("overall_health") or {}).get("status") or "unknown")).lower()

    adjusted = dict(base)
    reasons: list[str] = []

    if dq < 80:
        adjusted["aggressive"] *= 0.65
        adjusted["bond"] *= 1.20
        adjusted["dividend"] *= 1.15
        reasons.append(f"dq_low:{dq:.2f}")

    if blocked > 0.25:
        adjusted["aggressive"] *= 0.70
        adjusted["core"] *= 0.90
        adjusted["bond"] *= 1.25
        reasons.append(f"blocked_high:{blocked:.4f}")

    if drift > 0.20:
        adjusted["aggressive"] *= 0.80
        adjusted["dividend"] *= 1.10
        adjusted["bond"] *= 1.10
        reasons.append(f"drift_high:{drift:.4f}")

    if slo_alerts > 0:
        adjusted["aggressive"] *= 0.60
        adjusted["core"] *= 0.90
        adjusted["bond"] *= 1.25
        reasons.append(f"slo_alerts:{slo_alerts}")

    if bot_status in {"warn", "error", "critical"}:
        adjusted["aggressive"] *= 0.80
        adjusted["core"] *= 0.95
        adjusted["bond"] *= 1.10
        reasons.append(f"bot_stack_status:{bot_status}")

    weights = _normalize(adjusted)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "broker": args.broker,
        "inputs": {
            "data_quality_score": dq,
            "combined_blocked_rate": blocked,
            "buy_rate_drift_abs": drift,
            "slo_alert_count": slo_alerts,
            "bot_stack_status": bot_status,
        },
        "policy": {
            "base_weights": base,
            "adjusted_weights": adjusted,
            "reasons": reasons,
        },
        "target_weights": {k: round(v, 6) for k, v in weights.items()},
        "gross_risk_budget": round(_safe_float(one.get("data_quality_score"), 75.0) / 100.0, 4),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    events = PROJECT_ROOT / "governance" / "allocator" / f"sleeve_allocator_events_{_event_day()}.jsonl"
    events.parent.mkdir(parents=True, exist_ok=True)
    with events.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(f"allocator_ok=True broker={args.broker} reasons={','.join(reasons) if reasons else 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
