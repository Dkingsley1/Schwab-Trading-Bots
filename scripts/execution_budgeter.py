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


def _event_day() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build execution budgets per sleeve from allocator + risk ledger.")
    parser.add_argument("--allocator", default=str(PROJECT_ROOT / "governance" / "allocator" / "sleeve_allocator_latest.json"))
    parser.add_argument("--risk", default=str(PROJECT_ROOT / "governance" / "risk" / "portfolio_risk_latest.json"))
    parser.add_argument("--slo", default=str(PROJECT_ROOT / "governance" / "watchdog" / "sleeve_slo_latest.json"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "governance" / "risk" / "execution_budget_latest.json"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    alloc = _read_json(Path(args.allocator))
    risk = _read_json(Path(args.risk))
    slo = _read_json(Path(args.slo))

    weights = alloc.get("target_weights") or {}
    caps = ((risk.get("limits") or {}).get("sleeve_exposure_caps") or {})
    risk_level = str(risk.get("risk_level", "medium"))
    alert_count = len(slo.get("alerts") or []) if isinstance(slo.get("alerts"), list) else 0

    base_actions = {
        "core": 80,
        "aggressive": 160,
        "dividend": 30,
        "bond": 30,
        "crypto": 120,
    }

    if risk_level == "high":
        mult = 0.55
    elif risk_level == "medium":
        mult = 0.75
    else:
        mult = 1.00

    if alert_count > 0:
        mult *= 0.85

    per_sleeve: dict[str, dict] = {}
    for sleeve, w in weights.items():
        base = base_actions.get(sleeve, 40)
        actions_per_hour = max(4, int(round(base * float(mult) * max(float(w), 0.05))))
        per_sleeve[sleeve] = {
            "target_weight": round(float(w), 6),
            "exposure_cap": round(float(caps.get(sleeve, 0.0)), 6),
            "max_actions_per_hour": actions_per_hour,
            "max_open_orders": max(2, int(actions_per_hour // 8)),
        }

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "risk_level": risk_level,
        "slo_alert_count": alert_count,
        "global": {
            "max_total_actions_per_hour": int(sum(v["max_actions_per_hour"] for v in per_sleeve.values())),
            "max_total_open_orders": int(sum(v["max_open_orders"] for v in per_sleeve.values())),
            "multiplier": round(mult, 4),
        },
        "sleeves": per_sleeve,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    events = PROJECT_ROOT / "governance" / "risk" / f"execution_budget_events_{_event_day()}.jsonl"
    events.parent.mkdir(parents=True, exist_ok=True)
    with events.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "execution_budget_ok=True risk_level={risk} total_actions={a} total_open_orders={o}".format(
                risk=risk_level,
                a=payload["global"]["max_total_actions_per_hour"],
                o=payload["global"]["max_total_open_orders"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
