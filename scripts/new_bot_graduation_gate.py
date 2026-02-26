import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _to_float(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _to_int(value, default=0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def main() -> int:
    parser = argparse.ArgumentParser(description="New-bot graduation gate for promotion safety.")
    parser.add_argument("--registry", default=str(PROJECT_ROOT / "master_bot_registry.json"))
    parser.add_argument("--walk-forward-file", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "walk_forward_latest.json"))
    parser.add_argument("--min-runs", type=int, default=int(__import__("os").getenv("GRADUATION_MIN_RUNS", "30")))
    parser.add_argument("--min-forward-mean", type=float, default=float(__import__("os").getenv("GRADUATION_MIN_FORWARD_MEAN", "0.52")))
    parser.add_argument("--min-delta", type=float, default=float(__import__("os").getenv("GRADUATION_MIN_DELTA", "-0.02")))
    parser.add_argument("--min-mature-bots", type=int, default=int(__import__("os").getenv("GRADUATION_MIN_MATURE_BOTS", "16")))
    parser.add_argument("--min-mature-pass-rate", type=float, default=float(__import__("os").getenv("GRADUATION_MIN_MATURE_PASS_RATE", "0.30")))
    parser.add_argument("--max-immature-active", type=int, default=int(__import__("os").getenv("GRADUATION_MAX_IMMATURE_ACTIVE", "0")))
    parser.add_argument("--out-file", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "new_bot_graduation_latest.json"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    registry = _load(Path(args.registry))
    wf = _load(Path(args.walk_forward_file))
    wf_bots = wf.get("bots") if isinstance(wf.get("bots"), dict) else {}
    sub_bots = registry.get("sub_bots") if isinstance(registry.get("sub_bots"), list) else []

    mature_count = 0
    mature_pass = 0
    for bot_id, row in wf_bots.items():
        if not isinstance(row, dict):
            continue
        runs = _to_int(row.get("runs"), 0)
        if runs < int(args.min_runs):
            continue
        mature_count += 1
        fwd = _to_float(row.get("forward_mean"), 0.0)
        delta = _to_float(row.get("delta"), 0.0)
        status = str(row.get("status", "")).lower()
        if status == "pass" and fwd >= float(args.min_forward_mean) and delta >= float(args.min_delta):
            mature_pass += 1

    mature_pass_rate = float(mature_pass / max(mature_count, 1))

    immature_active = []
    for row in sub_bots:
        if not isinstance(row, dict):
            continue
        if not bool(row.get("active", False)):
            continue
        bot_id = str(row.get("bot_id", "")).strip()
        if not bot_id:
            continue
        wf_row = wf_bots.get(bot_id, {}) if isinstance(wf_bots, dict) else {}
        runs = _to_int((wf_row or {}).get("runs"), 0)
        fwd = _to_float((wf_row or {}).get("forward_mean"), 0.0)
        delta = _to_float((wf_row or {}).get("delta"), 0.0)
        status = str((wf_row or {}).get("status", "")).lower()

        reasons = []
        if runs < int(args.min_runs):
            reasons.append(f"runs<{int(args.min_runs)}")
        if fwd < float(args.min_forward_mean):
            reasons.append(f"forward<{float(args.min_forward_mean):.3f}")
        if delta < float(args.min_delta):
            reasons.append(f"delta<{float(args.min_delta):.3f}")
        if status and status != "pass":
            reasons.append(f"status={status}")

        if reasons:
            immature_active.append(
                {
                    "bot_id": bot_id,
                    "runs": runs,
                    "forward_mean": round(fwd, 6),
                    "delta": round(delta, 6),
                    "status": status or "unknown",
                    "reasons": reasons,
                }
            )

    ok = (
        mature_count >= int(args.min_mature_bots)
        and mature_pass_rate >= float(args.min_mature_pass_rate)
        and len(immature_active) <= int(args.max_immature_active)
    )

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ok": bool(ok),
        "thresholds": {
            "min_runs": int(args.min_runs),
            "min_forward_mean": float(args.min_forward_mean),
            "min_delta": float(args.min_delta),
            "min_mature_bots": int(args.min_mature_bots),
            "min_mature_pass_rate": float(args.min_mature_pass_rate),
            "max_immature_active": int(args.max_immature_active),
        },
        "maturity": {
            "mature_bots": int(mature_count),
            "mature_pass_bots": int(mature_pass),
            "mature_pass_rate": round(mature_pass_rate, 6),
        },
        "immature_active_count": len(immature_active),
        "immature_active_examples": immature_active[:30],
    }

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "new_bot_graduation_gate "
            f"ok={str(payload['ok']).lower()} "
            f"mature={mature_count} pass_rate={mature_pass_rate:.3f} "
            f"immature_active={len(immature_active)}"
        )

    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
