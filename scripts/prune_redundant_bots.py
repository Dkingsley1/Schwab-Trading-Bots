import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def family_key(bot_id: str) -> str:
    name = re.sub(r"^brain_refinery_v\d+_", "", bot_id)

    if "trend_" in name and "_sentinel" in name:
        return "trend_sentinel"
    if name.startswith("intraday_"):
        return "intraday"
    if name.startswith("swing_"):
        return "swing"
    if name.startswith("investment_"):
        return "investment"
    if "risk" in name and "sentinel" in name:
        return "risk_sentinel"
    if "drift" in name:
        return "drift"
    if "correlation" in name:
        return "correlation"
    if any(x in name for x in ("bollinger", "keltner", "donchian", "breakout")):
        return "breakout_band"
    if any(x in name for x in ("macd", "tsi", "stoch", "smi")):
        return "oscillator"
    if "layer" in name:
        return "layer"
    if "sentinel" in name:
        return "sentinel"

    toks = name.split("_")
    return "_".join(toks[:2]) if len(toks) >= 2 else name


def score(row: dict) -> float:
    acc = row.get("test_accuracy")
    q = row.get("quality_score")
    a = float(acc) if acc is not None else 0.0
    qq = float(q) if q is not None else 0.0
    return 0.8 * a + 0.2 * qq


def main() -> int:
    parser = argparse.ArgumentParser(description="Soft-prune redundant bots by family clusters.")
    parser.add_argument("--registry", default=str(PROJECT_ROOT / "master_bot_registry.json"))
    parser.add_argument("--keep-signal", type=int, default=2)
    parser.add_argument("--keep-infra", type=int, default=2)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    reg_path = Path(args.registry)
    reg = json.loads(reg_path.read_text(encoding="utf-8"))

    grouped = defaultdict(list)
    for row in reg.get("sub_bots", []):
        if bool(row.get("deleted_from_rotation", False)):
            continue
        fam = family_key(str(row.get("bot_id", "")))
        role = str(row.get("bot_role", "signal_sub_bot"))
        grouped[(fam, role)].append(row)

    changes = []
    kept = []
    for (fam, role), rows in grouped.items():
        rows.sort(key=score, reverse=True)
        keep_n = args.keep_infra if role == "infrastructure_sub_bot" else args.keep_signal

        for i, r in enumerate(rows):
            bid = str(r.get("bot_id"))
            if i < keep_n:
                kept.append(bid)
                continue

            # Soft prune redundancy: inactive + weight zero + reason tag, no hard delete.
            prev_active = bool(r.get("active", False))
            prev_reason = str(r.get("reason", ""))
            r["active"] = False
            r["weight"] = 0.0
            r["reason"] = f"redundant_family_prune:{fam}"
            changes.append(
                {
                    "bot_id": bid,
                    "family": fam,
                    "role": role,
                    "was_active": prev_active,
                    "old_reason": prev_reason,
                    "acc": r.get("test_accuracy"),
                    "quality": r.get("quality_score"),
                }
            )

    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "apply": bool(args.apply),
        "keep_signal": int(args.keep_signal),
        "keep_infra": int(args.keep_infra),
        "changed_count": len(changes),
        "changed": changes[:200],
        "kept_count": len(kept),
    }

    if args.apply:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup = reg_path.with_name(f"master_bot_registry.backup_redundant_prune_{ts}.json")
        backup.write_text(json.dumps(reg, indent=2), encoding="utf-8")
        reg_path.write_text(json.dumps(reg, indent=2), encoding="utf-8")
        out["backup"] = str(backup)

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
