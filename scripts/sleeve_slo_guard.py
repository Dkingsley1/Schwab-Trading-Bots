import argparse
import fcntl
import hashlib
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WATCHDOG_DIR = PROJECT_ROOT / "governance" / "watchdog"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.storage_router import inspect_storage_path
from scripts.ops.long_runtime_common import (
    evidence_freshness,
    parse_iso_utc,
    write_payload,
)


def local_path(root: Path, path: Path) -> Path:
    route = inspect_storage_path(path, boundary_root=root, allow_external=False)
    if route.get("status") not in {"present", "missing"} or route.get("symlinks"):
        raise ValueError("unsafe_risk_evidence_path")
    return path


def read_local_json(root: Path, path: Path) -> dict:
    path = local_path(root, path)
    if not path.exists():
        return {}
    if not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("risk_evidence_size_or_type_invalid")
    payload = _read_json(path)
    return payload if isinstance(payload, dict) else {}


def watchdog_payload(watchdog, state, *, now, required_breaches=3):
    """Consume the active watchdog's observation, never start or restart a process."""
    freshness = evidence_freshness(watchdog, now=now, max_age_minutes=6)
    rows = watchdog.get("status")
    valid = bool(
        freshness["fresh"]
        and isinstance(rows, list)
        and rows
        and all(
            isinstance(row, dict)
            and isinstance(row.get("name"), str)
            and row["name"]
            and isinstance(row.get("heartbeat_ok"), bool)
            for row in rows
        )
        and len({row["name"] for row in rows}) == len(rows)
        and any(row["name"] == "all_sleeves" for row in rows)
        and isinstance(watchdog.get("restart_storms"), list)
    )
    source_hash = hashlib.sha256(
        json.dumps(watchdog, sort_keys=True).encode()
    ).hexdigest()
    observed = parse_iso_utc(watchdog.get("timestamp_utc"))
    previous = parse_iso_utc(state.get("source_observed_at_utc"))
    new_observation = bool(observed and (previous is None or observed > previous))
    if (
        observed
        and previous
        and (
            observed < previous
            or (
                observed == previous
                and source_hash != state.get("source_receipt_sha256")
            )
        )
    ):
        valid = False
    streaks = dict(state.get("streaks") or {})
    targets, alerts = [], []
    if not valid:
        alerts.append(
            {
                "name": "process_watchdog",
                "breaches": ["source_missing_stale_or_incomplete"],
            }
        )
    else:
        storms = {
            row.get("name")
            for row in watchdog["restart_storms"]
            if isinstance(row, dict)
        }
        for row in rows:
            name = row["name"]
            live = row.get("effective_process_live", row.get("process_live")) is True
            heartbeat_ok = row["heartbeat_ok"]
            heartbeat_age = row.get("heartbeat_age_seconds")
            heartbeat_limit = row.get("heartbeat_max_age_seconds")
            if heartbeat_age is not None or heartbeat_limit is not None:
                try:
                    heartbeat_age = float(heartbeat_age) + freshness["age_minutes"] * 60
                    heartbeat_limit = float(heartbeat_limit)
                    heartbeat_ok = bool(
                        heartbeat_ok
                        and math.isfinite(heartbeat_age)
                        and math.isfinite(heartbeat_limit)
                        and 0 <= heartbeat_age <= heartbeat_limit
                    )
                except (ValueError, TypeError):
                    heartbeat_ok = False
            idle = row.get("writer_idle_health") or {}
            recovery = row.get("writer_recovery_health") or {}
            healthy = bool(
                heartbeat_ok
                and (live or idle.get("ok") is True or recovery.get("ok") is True)
            )
            breaches = [] if healthy else ["process_or_heartbeat_unhealthy"]
            if name in storms:
                breaches.append("watchdog_reported_restart_storm")
            streak = int(streaks.get(name, 0))
            if new_observation:
                streak = streak + 1 if breaches else 0
            streaks[name] = streak
            alert = bool(breaches and (streak >= required_breaches or name in storms))
            if alert:
                alerts.append({"name": name, "breaches": breaches, "streak": streak})
            targets.append(
                {
                    "name": name,
                    "live": live,
                    "heartbeat_ok": heartbeat_ok,
                    "heartbeat_age_s": heartbeat_age,
                    "breaches": breaches,
                    "breach_streak": streak,
                    "alert": alert,
                }
            )
    return {
        "timestamp_utc": now.isoformat(),
        "schema_version": 2,
        "source_kind": "active_process_watchdog_snapshot",
        "source_timestamp_utc": watchdog.get("timestamp_utc"),
        "source_observed_at_utc": watchdog.get("timestamp_utc"),
        "source_receipt_sha256": source_hash,
        "source_scope": "reported_watchdog_targets_not_every_individual_bot",
        "restart_rate_basis": "watchdog_reported_storms_not_reconstructed_history",
        "input_freshness": {"sources_ready": valid, "process_watchdog": freshness},
        "required_consecutive_breaches": required_breaches,
        "ok": valid and not alerts,
        "overall_ok": valid and not alerts,
        "alerts": alerts,
        "targets": targets,
        "live_execution_authority": False,
    }, {
        "timestamp_utc": now.isoformat(),
        "streaks": streaks,
        "source_receipt_sha256": (
            source_hash if valid else state.get("source_receipt_sha256")
        ),
        "source_observed_at_utc": (
            watchdog.get("timestamp_utc")
            if valid
            else state.get("source_observed_at_utc")
        ),
    }


def refresh_current(root: Path, *, out_path=None, state_path=None, required_breaches=3):
    out = local_path(
        root, out_path or root / "governance/watchdog/sleeve_slo_latest.json"
    )
    state_file = local_path(
        root, state_path or root / "governance/watchdog/sleeve_slo_state.json"
    )
    lock = local_path(root, state_file.with_suffix(".lock"))
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        watchdog = read_local_json(
            root, root / "governance/health/process_watchdog_latest.json"
        )
        state = read_local_json(root, state_file)
        payload, updated = watchdog_payload(
            watchdog, state, now=_now_utc(), required_breaches=required_breaches
        )
        write_payload(state_file, updated)
        write_payload(out, payload)
        return payload


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
            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00")).astimezone(
                timezone.utc
            )
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
    parser = argparse.ArgumentParser(
        description="Per-sleeve SLO guard with sustained-breach alerting."
    )
    parser.add_argument("--day", default=_today())
    parser.add_argument("--event-log", default=None)
    parser.add_argument(
        "--state-file", default=str(WATCHDOG_DIR / "sleeve_slo_state.json")
    )
    parser.add_argument(
        "--out-file", default=str(WATCHDOG_DIR / "sleeve_slo_latest.json")
    )
    parser.add_argument("--required-consecutive-breaches", type=int, default=3)
    parser.add_argument("--max-heartbeat-age-seconds", type=float, default=240.0)
    parser.add_argument("--max-restarts-per-hour", type=int, default=4)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    if args.event_log is None:
        payload = refresh_current(
            PROJECT_ROOT,
            out_path=Path(args.out_file),
            state_path=Path(args.state_file),
            required_breaches=max(args.required_consecutive_breaches, 1),
        )
        print(
            json.dumps(payload)
            if args.json
            else f"sleeve_slo_ok={payload['overall_ok']} alerts={len(payload['alerts'])} source=process_watchdog"
        )
        return 0 if payload["overall_ok"] else 2

    event_path = (
        Path(args.event_log)
        if args.event_log
        else (WATCHDOG_DIR / f"watchdog_events_{args.day}.jsonl")
    )
    events = _read_jsonl(event_path)
    latest_evt = (
        events[-1]
        if events
        else {"timestamp_utc": _now_utc().isoformat(), "targets": []}
    )

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
            alerts.append(
                {"name": name, "breaches": breaches, "streak": current_streak}
            )

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
    out_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
    )

    events_out = WATCHDOG_DIR / f"sleeve_slo_events_{args.day}.jsonl"
    with events_out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            f"sleeve_slo_ok={payload['overall_ok']} alerts={len(alerts)} source={event_path}"
        )
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
