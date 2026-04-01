import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.training_guard import check_confirmed_training_success, check_registry_row_state_before_deletion
from core.accountability import write_registry_mutation_journal


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


def _safe_write_json(path: Path, payload: dict) -> str:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        return str(path)
    except Exception as exc:
        fallback = Path("/tmp") / path.name
        try:
            fallback.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
            print(f"[RetirePersistentLosers] write_fallback path={path} fallback={fallback} err={exc}")
            return str(fallback)
        except Exception:
            print(f"[RetirePersistentLosers] write_failed path={path} err={exc}")
            return ""


def _parse_lane_int_map(raw: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for part in (raw or "").split(","):
        seg = part.strip()
        if not seg or ":" not in seg:
            continue
        k, v = seg.split(":", 1)
        key = k.strip().lower()
        if not key:
            continue
        try:
            out[key] = max(int(float(v.strip())), 0)
        except Exception:
            continue
    return out


def _infer_lane(row: dict) -> str:
    role = str((row or {}).get("bot_role", "")).strip().lower()
    bot_id = str((row or {}).get("bot_id", "")).strip().lower()

    if any(tok in bot_id for tok in ("long_term", "dividend_quality_compounder", "dividend_yield_trap_avoidance")):
        return "long_term"
    if role == "options_sub_bot" or any(tok in bot_id for tok in ("options", "greek", "iv_", "vol_surface", "put_call")):
        return "options"
    if role == "futures_sub_bot" or any(tok in bot_id for tok in ("futures", "funding", "basis", "order_book", "open_interest", "term_structure")):
        return "futures"
    if any(tok in bot_id for tok in ("intraday", "scalp", "open_close", "ultrafast", "day_trade", "daytrading")):
        return "day"
    if any(tok in bot_id for tok in ("swing", "position_1m_3m", "1w_3w", "2d_5d")):
        return "swing"
    return "equities"


def _is_canary_row(row: dict) -> bool:
    if bool((row or {}).get("promoted", False)):
        return True
    reason = str((row or {}).get("reason", "")).strip().lower()
    promotion_reason = str((row or {}).get("promotion_reason", "")).strip().lower()
    return ("canary" in reason) or ("canary" in promotion_reason)


def _counter_to_int_dict(counter: Counter[str]) -> dict[str, int]:
    return {k: int(counter[k]) for k in sorted(counter.keys())}


def _protected_collection_lanes(registry: dict) -> set[str]:
    master_policy = registry.get("master_policy") if isinstance(registry.get("master_policy"), dict) else {}
    raw = master_policy.get("protected_collection_lanes", ["options", "long_term"])
    if isinstance(raw, str):
        items = [part.strip().lower() for part in raw.split(",")]
    elif isinstance(raw, list):
        items = [str(part).strip().lower() for part in raw]
    else:
        items = []
    return {item for item in items if item}


def _protected_collection_lane_floors(registry: dict) -> dict[str, int]:
    master_policy = registry.get("master_policy") if isinstance(registry.get("master_policy"), dict) else {}
    raw = master_policy.get("protected_collection_lane_floors", {"options": 2, "long_term": 2})
    if isinstance(raw, dict):
        out: dict[str, int] = {}
        for key, value in raw.items():
            lane = str(key).strip().lower()
            if not lane:
                continue
            try:
                out[lane] = max(int(value), 0)
            except Exception:
                continue
        return out
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Retire persistent underperformers using readiness history.")
    parser.add_argument("--registry", default=str(PROJECT_ROOT / "master_bot_registry.json"))
    parser.add_argument("--history-jsonl", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "promotion_readiness_history.jsonl"))
    parser.add_argument("--lookback-days", type=int, default=14)
    parser.add_argument("--min-fail-days", type=int, default=7)
    parser.add_argument("--min-no-improvement-streak", type=int, default=3)
    parser.add_argument("--max-retire-per-run", type=int, default=4)
    parser.add_argument(
        "--lane-min-fail-days",
        default=os.getenv(
            "RETIRE_LANE_MIN_FAIL_DAYS",
            "equities:7,day:6,swing:6,options:5,futures:5,long_term:10,default:7",
        ),
        help="Per-lane minimum fail-days map (lane:int,comma-separated).",
    )
    parser.add_argument(
        "--lane-min-no-improvement-streak",
        default=os.getenv(
            "RETIRE_LANE_MIN_NO_IMPROVEMENT_STREAK",
            "equities:3,day:2,swing:2,options:2,futures:2,long_term:4,default:3",
        ),
        help="Per-lane minimum no-improvement streak map (lane:int,comma-separated).",
    )
    parser.add_argument(
        "--lane-max-retire-per-run",
        default=os.getenv(
            "RETIRE_LANE_MAX_PER_RUN",
            "equities:3,day:2,swing:2,options:2,futures:2,long_term:1,default:2",
        ),
        help="Per-lane max retire count map (lane:int,comma-separated).",
    )
    parser.add_argument(
        "--canary-min-fail-days",
        type=int,
        default=int(os.getenv("RETIRE_CANARY_MIN_FAIL_DAYS", "5")),
        help="Stricter fail-days floor for canary-tagged bots.",
    )
    parser.add_argument(
        "--canary-min-no-improvement-streak",
        type=int,
        default=int(os.getenv("RETIRE_CANARY_MIN_NO_IMPROVEMENT_STREAK", "2")),
        help="Stricter no-improvement streak floor for canary-tagged bots.",
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "retire_persistent_losers_latest.json"))
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--require-training-success",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("REQUIRE_CONFIRMED_TRAINING_SUCCESS", "1").strip() == "1",
        help="Block retirement deletes when recent confirmed training success is missing.",
    )
    parser.add_argument(
        "--max-training-age-hours",
        type=float,
        default=float(os.getenv("CONFIRMED_TRAINING_SUCCESS_MAX_AGE_HOURS", "72")),
    )
    parser.add_argument(
        "--training-success-file",
        default=str(PROJECT_ROOT / "governance" / "health" / "training_success_latest.json"),
    )
    parser.add_argument(
        "--training-scorecard-file",
        default=str(PROJECT_ROOT / "governance" / "health" / "retrain_scorecard_latest.json"),
    )
    args = parser.parse_args()

    lane_min_fail_days = _parse_lane_int_map(str(args.lane_min_fail_days))
    lane_min_streak = _parse_lane_int_map(str(args.lane_min_no_improvement_streak))
    lane_max_retire = _parse_lane_int_map(str(args.lane_max_retire_per_run))

    registry_path = Path(args.registry)
    if not registry_path.exists():
        raise SystemExit(f"missing registry: {registry_path}")

    reg = json.loads(registry_path.read_text(encoding="utf-8"))
    original_reg = json.loads(json.dumps(reg))
    sub_bots = reg.get("sub_bots") if isinstance(reg.get("sub_bots"), list) else []
    protected_lanes = _protected_collection_lanes(reg)
    protected_lane_floors = _protected_collection_lane_floors(reg)
    active_lane_counts: Counter[str] = Counter()
    for row in sub_bots:
        if not isinstance(row, dict):
            continue
        if bool(row.get("deleted_from_rotation", False)):
            continue
        if not bool(row.get("active", False)):
            continue
        active_lane_counts[_infer_lane(row)] += 1

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
    candidate_lane_counts: Counter[str] = Counter()
    for row in sub_bots:
        if not isinstance(row, dict):
            continue
        bot_id = str(row.get("bot_id", "")).strip().lower()
        if not bot_id:
            continue
        if bool(row.get("deleted_from_rotation", False)):
            continue

        lane = _infer_lane(row)
        is_canary = _is_canary_row(row)
        fd = int(fail_days.get(bot_id, 0))
        streak = int(row.get("no_improvement_streak", 0) or 0)

        lane_fail_req = int(lane_min_fail_days.get(lane, lane_min_fail_days.get("default", int(args.min_fail_days))))
        lane_streak_req = int(lane_min_streak.get(lane, lane_min_streak.get("default", int(args.min_no_improvement_streak))))
        if is_canary:
            lane_fail_req = min(lane_fail_req, max(int(args.canary_min_fail_days), 0))
            lane_streak_req = min(lane_streak_req, max(int(args.canary_min_no_improvement_streak), 0))

        if fd >= lane_fail_req and streak >= lane_streak_req:
            candidates.append({
                "bot_id": bot_id,
                "lane": lane,
                "is_canary": bool(is_canary),
                "is_active": bool(row.get("active", False)),
                "fail_days": fd,
                "no_improvement_streak": streak,
                "required_fail_days": lane_fail_req,
                "required_no_improvement_streak": lane_streak_req,
            })
            candidate_lane_counts[lane] += 1

    candidates.sort(
        key=lambda x: (
            int(x.get("is_canary", False)) * -1,
            -int(x.get("fail_days", 0)),
            -int(x.get("no_improvement_streak", 0)),
            str(x.get("bot_id", "")),
        )
    )

    selected: list[dict] = []
    selected_lane_counts: Counter[str] = Counter()
    selected_active_lane_counts: Counter[str] = Counter()
    max_retire_total = max(int(args.max_retire_per_run), 0)
    for row in candidates:
        if len(selected) >= max_retire_total:
            break
        lane = str(row.get("lane", "equities"))
        lane_cap = int(lane_max_retire.get(lane, lane_max_retire.get("default", max_retire_total)))
        if lane_cap == 0:
            continue
        if lane_cap > 0 and selected_lane_counts[lane] >= lane_cap:
            continue
        if bool(row.get("is_active", False)) and lane in protected_lanes:
            lane_floor = int(protected_lane_floors.get(lane, 0))
            remaining_active = int(active_lane_counts.get(lane, 0)) - int(selected_active_lane_counts.get(lane, 0))
            if lane_floor > 0 and (remaining_active - 1) < lane_floor:
                continue
        selected.append(row)
        selected_lane_counts[lane] += 1
        if bool(row.get("is_active", False)):
            selected_active_lane_counts[lane] += 1

    guard_ok = True
    guard_reason = "disabled"
    guard_details = {}
    if args.require_training_success:
        guard_ok, guard_reason, guard_details = check_confirmed_training_success(
            project_root=str(PROJECT_ROOT),
            marker_path=str(args.training_success_file),
            scorecard_path=str(args.training_scorecard_file),
            max_age_hours=float(args.max_training_age_hours),
            require_master_update=True,
            min_trained_bots=1,
        )

    retired = []
    blocked = []
    blocked_state = []
    apply_blocked = bool(args.apply and args.require_training_success and (not guard_ok) and bool(selected))

    if args.apply and selected and (not apply_blocked):
        selected_ids = {x["bot_id"] for x in selected}
        selected_by_id = {str(x.get("bot_id", "")).strip().lower(): x for x in selected}
        for row in sub_bots:
            if not isinstance(row, dict):
                continue
            bot_id = str(row.get("bot_id", "")).strip().lower()
            if bot_id not in selected_ids:
                continue

            sel = selected_by_id.get(bot_id, {})
            min_required_streak = int(sel.get("required_no_improvement_streak", int(args.min_no_improvement_streak)) or 0)
            state_ok, state_reason, state_details = check_registry_row_state_before_deletion(
                row,
                min_streak=min_required_streak,
            )
            if not state_ok:
                blocked_state.append(
                    {
                        "bot_id": bot_id,
                        "lane": str(sel.get("lane", _infer_lane(row))),
                        "state_reason": state_reason,
                        "state_details": state_details,
                    }
                )
                continue

            lane = str(sel.get("lane", _infer_lane(row)))
            required_fail_days = int(sel.get("required_fail_days", int(args.min_fail_days)) or 0)
            row["active"] = False
            row["deleted_from_rotation"] = True
            row["delete_reason"] = f"auto_retire_{lane}_persistent_fail_{required_fail_days}d"
            row["reason"] = row.get("reason") or f"auto_retired_persistent_fail_{lane}"
            row["promotion_reason"] = "rotation_deleted"
            retired.append(bot_id)

        reg["updated_at_utc"] = now.isoformat()
        backup = registry_path.with_name(f"master_bot_registry.backup_{now.strftime('%Y%m%d_%H%M%S')}.json")
        backup.write_text(json.dumps(original_reg, ensure_ascii=True, indent=2), encoding="utf-8")
        registry_path.write_text(json.dumps(reg, ensure_ascii=True, indent=2), encoding="utf-8")
        try:
            write_registry_mutation_journal(
                project_root=str(PROJECT_ROOT),
                actor="retire_persistent_losers",
                reason=f"apply_min_fail_days_{int(args.min_fail_days)}",
                before=original_reg if isinstance(original_reg, dict) else {},
                after=reg if isinstance(reg, dict) else {},
                extra={
                    "apply": bool(args.apply),
                    "apply_blocked": bool(apply_blocked),
                    "retired_count": len(retired),
                    "blocked_count": len(blocked),
                    "blocked_state_count": len(blocked_state),
                    "backup_path": str(backup),
                },
            )
        except Exception:
            pass
    elif apply_blocked and selected:
        blocked = [
            {
                "bot_id": str(x.get("bot_id", "")),
                "lane": str(x.get("lane", "equities")),
                "is_canary": bool(x.get("is_canary", False)),
                "fail_days": int(x.get("fail_days", 0) or 0),
                "no_improvement_streak": int(x.get("no_improvement_streak", 0) or 0),
                "required_fail_days": int(x.get("required_fail_days", int(args.min_fail_days)) or 0),
                "required_no_improvement_streak": int(x.get("required_no_improvement_streak", int(args.min_no_improvement_streak)) or 0),
                "guard_reason": guard_reason,
            }
            for x in selected
        ]

    selected_by_id_all = {str(x.get("bot_id", "")).strip().lower(): x for x in selected}
    retired_lane_counts: Counter[str] = Counter()
    for bot_id in retired:
        lane = str(selected_by_id_all.get(str(bot_id).strip().lower(), {}).get("lane", "equities"))
        retired_lane_counts[lane] += 1

    blocked_lane_counts: Counter[str] = Counter()
    for row in blocked:
        lane = str((row or {}).get("lane", "")).strip().lower()
        if lane:
            blocked_lane_counts[lane] += 1

    payload = {
        "timestamp_utc": now.isoformat(),
        "lookback_days": int(args.lookback_days),
        "min_fail_days": int(args.min_fail_days),
        "min_no_improvement_streak": int(args.min_no_improvement_streak),
        "max_retire_per_run": int(args.max_retire_per_run),
        "lane_min_fail_days": {k: int(v) for k, v in sorted(lane_min_fail_days.items())},
        "lane_min_no_improvement_streak": {k: int(v) for k, v in sorted(lane_min_streak.items())},
        "lane_max_retire_per_run": {k: int(v) for k, v in sorted(lane_max_retire.items())},
        "canary_thresholds": {
            "min_fail_days": int(args.canary_min_fail_days),
            "min_no_improvement_streak": int(args.canary_min_no_improvement_streak),
        },
        "candidate_count": len(candidates),
        "candidate_lane_counts": _counter_to_int_dict(candidate_lane_counts),
        "selected_count": len(selected),
        "selected_lane_counts": _counter_to_int_dict(selected_lane_counts),
        "selected": selected,
        "applied": bool(args.apply and (not apply_blocked)),
        "apply_blocked": apply_blocked,
        "retired_count": len(retired),
        "retired_lane_counts": _counter_to_int_dict(retired_lane_counts),
        "retired": retired,
        "blocked_count": len(blocked),
        "blocked_lane_counts": _counter_to_int_dict(blocked_lane_counts),
        "blocked": blocked,
        "blocked_state_count": len(blocked_state),
        "blocked_state": blocked_state,
        "deletion_guard": {
            "required": bool(args.require_training_success),
            "ok": bool(guard_ok),
            "reason": guard_reason,
            "details": guard_details,
        },
    }

    out = Path(args.out)
    payload["out_file"] = _safe_write_json(out, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "retire_persistent_losers "
            f"candidates={payload['candidate_count']} "
            f"applied={payload['applied']} "
            f"retired={','.join(retired) if retired else 'none'} "
            f"blocked={len(blocked)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
