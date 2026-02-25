import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_ts(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _read_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    rows.append(json.loads(s))
                except Exception:
                    continue
    except Exception:
        return []
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Retire persistent underperformers using readiness history.")
    parser.add_argument("--registry", default=str(PROJECT_ROOT / "master_bot_registry.json"))
    parser.add_argument("--history-jsonl", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "promotion_readiness_history.jsonl"))
    parser.add_argument("--lookback-days", type=int, default=14)
    parser.add_argument("--min-fail-days", type=int, default=7)
    parser.add_argument("--min-no-improvement-streak", type=int, default=3)
    parser.add_argument("--max-retire-per-run", type=int, default=4)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "retire_persistent_losers_latest.json"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    registry_path = Path(args.registry)
    if not registry_path.exists():
        raise SystemExit(f"missing registry: {registry_path}")

    reg = json.loads(registry_path.read_text(encoding="utf-8"))
    sub_bots = reg.get("sub_bots") if isinstance(reg.get("sub_bots"), list) else []

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=max(int(args.lookback_days), 1))
    fail_days: dict[str, int] = {}

    for row in _read_history(Path(args.history_jsonl)):
        ts = _parse_ts(str(row.get("timestamp_utc", "")))
        if ts is None or ts < cutoff:
            continue
        bots = row.get("failed_bots_list") if isinstance(row.get("failed_bots_list"), list) else []
        seen = set()
        for b in bots:
            bot_id = str(b).strip().lower()
            if bot_id and bot_id not in seen:
                fail_days[bot_id] = fail_days.get(bot_id, 0) + 1
                seen.add(bot_id)

    candidates: list[dict] = []
    for row in sub_bots:
        if not isinstance(row, dict):
            continue
        bot_id = str(row.get("bot_id", "")).strip().lower()
        if not bot_id:
            continue
        if bool(row.get("deleted_from_rotation", False)):
            continue
        fd = int(fail_days.get(bot_id, 0))
        streak = int(row.get("no_improvement_streak", 0) or 0)
        if fd >= int(args.min_fail_days) and streak >= int(args.min_no_improvement_streak):
            candidates.append({
                "bot_id": bot_id,
                "fail_days": fd,
                "no_improvement_streak": streak,
            })

    candidates.sort(key=lambda x: (-x["fail_days"], -x["no_improvement_streak"], x["bot_id"]))
    selected = candidates[: max(int(args.max_retire_per_run), 0)]

    retired = []
    if args.apply and selected:
        selected_ids = {x["bot_id"] for x in selected}
        for row in sub_bots:
            if not isinstance(row, dict):
                continue
            bot_id = str(row.get("bot_id", "")).strip().lower()
            if bot_id not in selected_ids:
                continue
            row["active"] = False
            row["deleted_from_rotation"] = True
            row["delete_reason"] = f"auto_retire_persistent_fail_{int(args.min_fail_days)}d"
            row["reason"] = row.get("reason") or "auto_retired_persistent_fail"
            row["promotion_reason"] = "rotation_deleted"
            retired.append(bot_id)

        reg["updated_at_utc"] = now.isoformat()
        registry_path.write_text(json.dumps(reg, ensure_ascii=True, indent=2), encoding="utf-8")

    payload = {
        "timestamp_utc": now.isoformat(),
        "lookback_days": int(args.lookback_days),
        "min_fail_days": int(args.min_fail_days),
        "min_no_improvement_streak": int(args.min_no_improvement_streak),
        "candidate_count": len(candidates),
        "selected": selected,
        "applied": bool(args.apply),
        "retired": retired,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "retire_persistent_losers "
            f"candidates={payload['candidate_count']} "
            f"applied={payload['applied']} "
            f"retired={','.join(retired) if retired else 'none'}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
