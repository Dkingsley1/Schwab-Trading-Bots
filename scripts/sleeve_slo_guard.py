import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WATCHDOG_DIR = PROJECT_ROOT / "governance" / "watchdog"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _today() -> str:
    return _now_utc().strftime("%Y%m%d")


def _parse_note(note: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for token in (note or "").split(","):
        token = token.strip()
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _safe_int(v, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _restart_count_last_hour(events: list[dict], target_name: str) -> int:
    cutoff = _now_utc() - timedelta(hours=1)
    count = 0
    for evt in events:
        ts_raw = str(evt.get("timestamp_utc", ""))
        try:
            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            continue
        if ts < cutoff:
            continue
        for t in evt.get("targets", []) or []:
            if str(t.get("name", "")) != target_name:
                continue
            if str(t.get("action", "none")) == "restart":
                count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-sleeve SLO guard with sustained-breach alerting.")
    parser.add_argument("--day", default=_today())
    parser.add_argument("--event-log", default=None)
    parser.add_argument("--state-file", default=str(WATCHDOG_DIR / "sleeve_slo_state.json"))
    parser.add_argument("--out-file", default=str(WATCHDOG_DIR / "sleeve_slo_latest.json"))
    parser.add_argument("--required-consecutive-breaches", type=int, default=3)
    parser.add_argument("--max-heartbeat-age-seconds", type=float, default=240.0)
    parser.add_argument("--max-restarts-per-hour", type=int, default=4)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    event_path = Path(args.event_log) if args.event_log else (WATCHDOG_DIR / f"watchdog_events_{args.day}.jsonl")
    events = _read_jsonl(event_path)
    latest_evt = events[-1] if events else {"timestamp_utc": _now_utc().isoformat(), "targets": []}

    state_path = Path(args.state_file)
    state = _read_json(state_path)
    streaks = dict((state.get("streaks") or {}))

    entries: list[dict] = []
    alerts: list[dict] = []
    for target in latest_evt.get("targets", []) or []:
        name = str(target.get("name", "unknown"))
        live = bool(target.get("live", False))
        note_map = _parse_note(str(target.get("note", "")))
        hb_age = _safe_float(note_map.get("heartbeat_age_s"), 0.0)
        restarts_last_hour = _restart_count_last_hour(events, name)

        breaches: list[str] = []
        if not live:
            breaches.append("process_or_heartbeat_unhealthy")
        if hb_age > max(args.max_heartbeat_age_seconds, 1.0):
            breaches.append(f"heartbeat_age_high:{hb_age:.1f}")
        if restarts_last_hour > max(args.max_restarts_per_hour, 0):
            breaches.append(f"restart_rate_high:{restarts_last_hour}")

        current_streak = _safe_int(streaks.get(name, 0), 0)
        if breaches:
            current_streak += 1
        else:
            current_streak = 0
        streaks[name] = current_streak

        alert = current_streak >= max(args.required_consecutive_breaches, 1)
        if alert:
            alerts.append({"name": name, "breaches": breaches, "streak": current_streak})

        entries.append(
            {
                "name": name,
                "live": live,
                "heartbeat_age_s": round(hb_age, 1),
                "restarts_last_hour": restarts_last_hour,
                "breaches": breaches,
                "breach_streak": current_streak,
                "alert": alert,
            }
        )

    payload = {
        "timestamp_utc": _now_utc().isoformat(),
        "source_event_log": str(event_path),
        "required_consecutive_breaches": max(args.required_consecutive_breaches, 1),
        "max_heartbeat_age_seconds": max(args.max_heartbeat_age_seconds, 1.0),
        "max_restarts_per_hour": max(args.max_restarts_per_hour, 0),
        "overall_ok": len(alerts) == 0,
        "alerts": alerts,
        "targets": entries,
    }

    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "timestamp_utc": payload["timestamp_utc"],
                "streaks": streaks,
                "source_event_log": str(event_path),
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    events_out = WATCHDOG_DIR / f"sleeve_slo_events_{args.day}.jsonl"
    with events_out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(f"sleeve_slo_ok={payload['overall_ok']} alerts={len(alerts)} source={event_path}")
        for row in entries:
            print(
                " - {name}: live={live} hb_age_s={hb} restarts_1h={r} streak={s} alert={a} breaches={b}".format(
                    name=row["name"],
                    live=row["live"],
                    hb=row["heartbeat_age_s"],
                    r=row["restarts_last_hour"],
                    s=row["breach_streak"],
                    a=row["alert"],
                    b="|".join(row["breaches"]) if row["breaches"] else "none",
                )
            )

    return 0 if payload["overall_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
