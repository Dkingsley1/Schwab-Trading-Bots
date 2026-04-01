import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPS_THRESHOLDS_FILE = PROJECT_ROOT / "governance" / "ops_thresholds.json"
SCOPE_EXEMPT_TOKENS = (
    "collection_floor",
    "min_active_floor_override",
    "bucket_diversity",
    "manual_collection_restore",
    "manual_canary_restore",
)


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _threshold_defaults() -> dict:
    payload = _load(OPS_THRESHOLDS_FILE)
    gates = payload.get("promotion_gates") if isinstance(payload.get("promotion_gates"), dict) else {}
    return gates.get("graduation_gate") if isinstance(gates.get("graduation_gate"), dict) else {}


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


def _registry_row_map(sub_bots: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in sub_bots:
        if not isinstance(row, dict):
            continue
        bot_id = str(row.get("bot_id", "")).strip()
        if not bot_id:
            continue
        out[bot_id] = row
    return out


def _scope_exempt_reason(row: dict) -> str:
    tokens = " ".join(
        [
            str(row.get("reason", "") or ""),
            str(row.get("promotion_reason", "") or ""),
            str(row.get("promotion_status", "") or ""),
            str(row.get("bot_role", row.get("role", "")) or ""),
        ]
    ).lower()
    if "support_control" in tokens or "infrastructure_sub_bot" in tokens:
        return "support_control"
    if any(token in tokens for token in SCOPE_EXEMPT_TOKENS):
        return "coverage_exempt"
    return ""


def main() -> int:
    defaults = _threshold_defaults()
    parser = argparse.ArgumentParser(description="New-bot graduation gate for promotion safety.")
    parser.add_argument("--registry", default=str(PROJECT_ROOT / "master_bot_registry.json"))
    parser.add_argument("--walk-forward-file", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "walk_forward_latest.json"))
    parser.add_argument("--min-runs", type=int, default=int(os.getenv("GRADUATION_MIN_RUNS", str(defaults.get("min_runs", 24)))))
    parser.add_argument("--min-forward-mean", type=float, default=float(os.getenv("GRADUATION_MIN_FORWARD_MEAN", str(defaults.get("min_forward_mean", 0.52)))))
    parser.add_argument("--min-delta", type=float, default=float(os.getenv("GRADUATION_MIN_DELTA", str(defaults.get("min_delta", -0.02)))))
    parser.add_argument("--min-mature-bots", type=int, default=int(os.getenv("GRADUATION_MIN_MATURE_BOTS", str(defaults.get("min_mature_bots", 10)))))
    parser.add_argument("--min-mature-bots-floor", type=int, default=int(os.getenv("GRADUATION_MIN_MATURE_BOTS_FLOOR", "2")))
    parser.add_argument("--established-min-runs", type=int, default=int(os.getenv("GRADUATION_ESTABLISHED_MIN_RUNS", "16")))
    parser.add_argument("--min-mature-pass-rate", type=float, default=float(os.getenv("GRADUATION_MIN_MATURE_PASS_RATE", str(defaults.get("min_mature_pass_rate", 0.30)))))
    parser.add_argument("--max-immature-active", type=int, default=int(os.getenv("GRADUATION_MAX_IMMATURE_ACTIVE", str(defaults.get("max_immature_active", 0)))))
    parser.add_argument("--out-file", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "new_bot_graduation_latest.json"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    registry = _load(Path(args.registry))
    wf = _load(Path(args.walk_forward_file))
    wf_bots = wf.get("bots") if isinstance(wf.get("bots"), dict) else {}
    sub_bots = registry.get("sub_bots") if isinstance(registry.get("sub_bots"), list) else []
    registry_rows = _registry_row_map(sub_bots)

    active_rows = [row for row in sub_bots if isinstance(row, dict) and bool(row.get("active", False))]
    graduation_scope_active: list[dict] = []
    coverage_exempt_active: list[dict] = []
    for row in active_rows:
        exempt_reason = _scope_exempt_reason(row)
        target = coverage_exempt_active if exempt_reason else graduation_scope_active
        target.append(
            {
                "bot_id": str(row.get("bot_id", "")).strip(),
                "exempt_reason": exempt_reason,
                "promotion_status": str(row.get("promotion_status", "") or ""),
                "bot_role": str(row.get("bot_role", row.get("role", "")) or ""),
            }
        )

    mature_count = 0
    mature_pass = 0
    for bot_id, registry_row in registry_rows.items():
        row = wf_bots.get(bot_id, {}) if isinstance(wf_bots, dict) else {}
        if not isinstance(row, dict) or not bool(registry_row.get("active", False)):
            continue
        runs = _to_int(row.get("runs"), 0)
        fwd = _to_float(row.get("forward_mean"), 0.0)
        delta = _to_float(row.get("delta"), 0.0)
        status = str(row.get("status", "")).lower()
        exempt_reason = _scope_exempt_reason(registry_row)
        mature_run_floor = int(args.established_min_runs) if exempt_reason else int(args.min_runs)
        if runs < mature_run_floor:
            continue
        mature_count += 1
        if status == "pass" and fwd >= float(args.min_forward_mean) and delta >= float(args.min_delta):
            mature_pass += 1

    mature_pass_rate = float(mature_pass / max(mature_count, 1))
    graduation_scope_active_count = len(graduation_scope_active)
    effective_min_mature_bots = 0
    if graduation_scope_active_count > 0:
        effective_min_mature_bots = min(
            int(args.min_mature_bots),
            max(int(args.min_mature_bots_floor), graduation_scope_active_count),
        )

    immature_active = []
    for scoped in graduation_scope_active:
        bot_id = str(scoped.get("bot_id", "")).strip()
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
        mature_count >= int(effective_min_mature_bots)
        and mature_pass_rate >= float(args.min_mature_pass_rate)
        and len(immature_active) <= int(args.max_immature_active)
    )

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ok": bool(ok),
        "thresholds": {
            "min_runs": int(args.min_runs),
            "established_min_runs": int(args.established_min_runs),
            "min_forward_mean": float(args.min_forward_mean),
            "min_delta": float(args.min_delta),
            "min_mature_bots": int(args.min_mature_bots),
            "min_mature_bots_floor": int(args.min_mature_bots_floor),
            "effective_min_mature_bots": int(effective_min_mature_bots),
            "min_mature_pass_rate": float(args.min_mature_pass_rate),
            "max_immature_active": int(args.max_immature_active),
        },
        "maturity": {
            "mature_bots": int(mature_count),
            "mature_pass_bots": int(mature_pass),
            "mature_pass_rate": round(mature_pass_rate, 6),
        },
        "graduation_scope_active_count": int(graduation_scope_active_count),
        "coverage_exempt_active_count": int(len(coverage_exempt_active)),
        "coverage_exempt_examples": coverage_exempt_active[:30],
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
