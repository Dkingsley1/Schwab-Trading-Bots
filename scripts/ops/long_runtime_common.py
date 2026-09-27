#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
from collections import deque
from datetime import date, datetime, timedelta, time, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - zoneinfo is standard on supported runtimes
    ZoneInfo = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_TZ = ZoneInfo("America/New_York") if ZoneInfo is not None else timezone.utc

STATUS_RANK = {
    "ready": 0,
    "ok": 0,
    "active": 0,
    "warn": 1,
    "thin": 1,
    "needs_coverage": 1,
    "needs_work": 1,
    "degraded": 2,
    "inactive": 2,
    "missing": 2,
    "blocked": 3,
    "critical": 3,
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return utc_now().isoformat()


def parse_iso_utc(raw: Any) -> datetime | None:
    text = str(raw or "").strip().replace("Z", "+00:00")
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def load_recent_jsonl(path: Path, *, limit: int = 500) -> list[dict[str, Any]]:
    rows: deque[dict[str, Any]] = deque(maxlen=max(int(limit), 1))
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except Exception:
        return []
    return list(rows)


def payload_timestamp(payload: dict[str, Any], path: Path | None = None) -> datetime | None:
    for key in ("timestamp_utc", "updated_at_utc", "updated_at", "created_at", "ended_utc", "started_utc"):
        parsed = parse_iso_utc(payload.get(key))
        if parsed is not None:
            return parsed
    if path is None:
        return None
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except Exception:
        return None


def payload_age_minutes(payload: dict[str, Any], path: Path | None = None, *, now: datetime | None = None) -> float | None:
    ts = payload_timestamp(payload, path)
    if ts is None:
        return None
    current = now or utc_now()
    return max((current - ts).total_seconds() / 60.0, 0.0)


def evidence_freshness(
    payload: dict[str, Any], *, max_age_minutes: float = 60.0, now: datetime | None = None
) -> dict[str, Any]:
    """Use producer time, never file mtime, to qualify reported evidence."""
    field = next(
        (
            key
            for key in (
                "timestamp_utc",
                "updated_at_utc",
                "updated_at",
                "created_at",
                "ended_utc",
                "generated_utc",
                "generated_at_utc",
            )
            if key in payload
        ),
        "",
    )
    raw = payload.get(field) if field else None
    timestamp = parse_iso_utc(raw)
    age = ((now or utc_now()) - timestamp).total_seconds() / 60.0 if timestamp else None
    if not payload:
        status = "missing"
    elif not field:
        status = "timestamp_missing"
    elif timestamp is None:
        status = "timestamp_invalid"
    elif age < 0.0:
        status = "future_timestamp"
    elif age > max_age_minutes:
        status = "stale"
    else:
        status = "fresh"
    producer_timestamp = timestamp
    # Derived and deferred reports cannot renew the observations they copied.
    if status == "fresh" and "source_timestamp_utc" in payload:
        source = parse_iso_utc(payload.get("source_timestamp_utc"))
        if source is None:
            status = "source_timestamp_invalid"
        else:
            source_age = ((now or utc_now()) - source).total_seconds() / 60.0
            if source_age < 0:
                status = "future_source_timestamp"
            elif source_age > max_age_minutes:
                status = "stale"
            timestamp = min(timestamp, source)
            age = max(age, source_age)
    if status == "fresh" and payload.get("freshness_state") == "no_previous_measurement":
        status = "measurement_missing"
    if status == "fresh" and payload.get("input_evidence_ready") is False:
        status = "input_evidence_unavailable"
    if status == "fresh" and isinstance(payload.get("input_evidence"), dict):
        for row in payload["input_evidence"].values():
            if not isinstance(row, dict) or not row.get("fresh"):
                status = "input_evidence_unavailable"
                break
            try:
                ttl = min(float(row["max_age_minutes"]), max_age_minutes)
            except (KeyError, ValueError, TypeError):
                status = "input_evidence_unavailable"
                break
            source_evidence = evidence_freshness(
                {"timestamp_utc": row.get("source_timestamp_utc")}, max_age_minutes=ttl, now=now
            )
            if not source_evidence["fresh"]:
                status = "input_evidence_unavailable"
                break
    return {
        "status": status,
        "fresh": status == "fresh",
        "source_timestamp_utc": timestamp.isoformat() if timestamp else None,
        "producer_timestamp_utc": producer_timestamp.isoformat() if producer_timestamp else None,
        "age_minutes": round(age, 3) if age is not None else None,
        "max_age_minutes": max_age_minutes,
    }


def governor_observation_contract(
    inputs: dict[str, tuple[dict[str, Any], float]], *, now: datetime | None = None
) -> dict[str, Any]:
    """Carry required observation ages through a governor decision, without I/O."""
    current = now or utc_now()
    evidence = {
        name: evidence_freshness(payload, max_age_minutes=ttl / 60.0, now=current)
        for name, (payload, ttl) in inputs.items()
    }
    ready = bool(evidence) and all(row["fresh"] for row in evidence.values())
    timestamps = [row["source_timestamp_utc"] for row in evidence.values() if row["source_timestamp_utc"]]
    oldest = min(timestamps) if len(timestamps) == len(evidence) and timestamps else None
    return {
        "input_evidence_ready": ready,
        "input_evidence": evidence,
        "source_timestamp_utc": oldest,
        "measurement_refreshed": False,
        "freshness_state": "derived_from_observations" if ready else "awaiting_fresh_inputs",
    }


def standardize_grade_labels(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: standardize_grade_labels(item) for key, item in value.items()}
    if isinstance(value, list):
        return [standardize_grade_labels(item) for item in value]
    if isinstance(value, str):
        return value.replace("A++", "A+")
    return value


def governor_recovery_observation(
    previous: dict[str, Any],
    source_timestamp: Any,
    *,
    inputs_ready: bool = True,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Rate-limit healthy-sample credit without delaying adverse decisions."""
    current = parse_iso_utc(source_timestamp)
    now = now or utc_now()
    prior = previous.get("recovery_observation")
    prior = prior if isinstance(prior, dict) else {}
    prior_source = parse_iso_utc(prior.get("source_timestamp_utc") if prior else previous.get("source_timestamp_utc", previous.get("timestamp_utc")))
    prior_credit = parse_iso_utc(prior.get("last_credit_source_timestamp_utc")) if prior else prior_source
    prior_producer = parse_iso_utc(previous.get("timestamp_utc"))
    current_valid = bool(inputs_ready and current is not None and current <= now)
    history_valid = bool(
        current_valid and prior_source is not None and prior_credit is not None
        and prior_producer is not None and previous.get("input_evidence_ready") is not False
        and 0 <= (now - prior_producer).total_seconds() <= 300
        and 0 <= (current - prior_source).total_seconds() <= 300
        and prior_credit <= prior_source
    )
    credit_due = bool(current_valid and (not history_valid or (current - prior_credit).total_seconds() >= 60))
    last_credit = current if credit_due else prior_credit if history_valid else None
    return {
        "source_timestamp_utc": current.isoformat() if current_valid else None,
        "last_credit_source_timestamp_utc": last_credit.isoformat() if last_credit else None,
        "credit_due": credit_due,
        "reset_history": not history_valid,
        "minimum_credit_interval_seconds": 60,
    }


def write_payload(
    path: Path, payload: dict[str, Any], *, compact: bool = False
) -> None:
    normalized_payload = standardize_grade_labels(payload)
    serialized = json.dumps(
        normalized_payload,
        ensure_ascii=True,
        indent=None if compact else 2,
        separators=(",", ":") if compact else None,
    )
    write_text_atomic(path, serialized)


def write_text_atomic(path: Path, serialized: str) -> None:
    """Replace one complete control artifact while preserving its permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode = path.stat().st_mode & 0o777
    except OSError:
        mode = 0o644
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_path, mode)
        os.replace(temp_path, path)
        temp_path = None
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except OSError:
                pass


def _process_tree_targets(root_pid: int) -> tuple[set[int], set[int]]:
    try:
        snapshot = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,pgid="],
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
    except Exception:
        return {int(root_pid)}, {int(root_pid)}
    rows: dict[int, tuple[int, int]] = {}
    for raw_line in str(snapshot.stdout or "").splitlines():
        parts = raw_line.split()
        if len(parts) < 3:
            continue
        try:
            pid, ppid, pgid = (int(parts[0]), int(parts[1]), int(parts[2]))
        except Exception:
            continue
        rows[pid] = (ppid, pgid)
    descendants: set[int] = {int(root_pid)}
    frontier = deque([int(root_pid)])
    while frontier:
        parent = frontier.popleft()
        for pid, (ppid, _pgid) in rows.items():
            if ppid != parent or pid in descendants:
                continue
            descendants.add(pid)
            frontier.append(pid)
    pgids = {
        int(rows.get(pid, (0, pid))[1])
        for pid in descendants
        if int(rows.get(pid, (0, pid))[1]) > 0
    }
    return descendants, pgids


def _signal_process_tree_targets(pids: set[int], pgids: set[int], sig: int) -> None:
    own_pgid = os.getpgrp()
    for pgid in sorted(pgids, reverse=True):
        if pgid <= 0 or pgid == own_pgid:
            continue
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            pass
        except Exception:
            continue
    for pid in sorted(pids, reverse=True):
        if pid <= 0 or pid == os.getpid():
            continue
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass
        except Exception:
            continue


def run_bounded_process_group(
    cmd: list[str],
    *,
    cwd: Path,
    timeout_seconds: int,
    env: dict[str, str] | None = None,
    terminate_grace_seconds: float = 2.0,
) -> dict[str, Any]:
    """Run one maintenance command and reap its full descendant process group on timeout."""
    timeout = max(int(timeout_seconds), 1)
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return {
            "rc": int(proc.returncode),
            "stdout": stdout or "",
            "stderr": stderr or "",
            "timed_out": False,
            "timeout_cleanup": {"process_group": proc.pid, "signal": "", "reaped": True},
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        cleanup_signal = "SIGTERM"
        cleanup_pids, cleanup_pgids = _process_tree_targets(proc.pid)
        _signal_process_tree_targets(cleanup_pids, cleanup_pgids, signal.SIGTERM)
        try:
            stdout, stderr = proc.communicate(timeout=max(float(terminate_grace_seconds), 0.1))
        except subprocess.TimeoutExpired as final_exc:
            stdout = (
                final_exc.stdout.decode("utf-8", errors="ignore")
                if isinstance(final_exc.stdout, bytes)
                else str(final_exc.stdout or stdout)
            )
            stderr = (
                final_exc.stderr.decode("utf-8", errors="ignore")
                if isinstance(final_exc.stderr, bytes)
                else str(final_exc.stderr or stderr)
            )
            cleanup_signal = "SIGKILL"
            current_pids, current_pgids = _process_tree_targets(proc.pid)
            _signal_process_tree_targets(
                cleanup_pids | current_pids,
                cleanup_pgids | current_pgids,
                signal.SIGKILL,
            )
            try:
                final_stdout, final_stderr = proc.communicate(timeout=1)
                stdout = final_stdout or stdout
                stderr = final_stderr or stderr
            except Exception:
                pass
        return {
            "rc": 124,
            "stdout": stdout or "",
            "stderr": stderr or "",
            "timed_out": True,
            "timeout_cleanup": {
                "process_group": proc.pid,
                "signal": cleanup_signal,
                "reaped": proc.poll() is not None,
            },
        }
    except BaseException:
        cleanup_pids, cleanup_pgids = _process_tree_targets(proc.pid)
        _signal_process_tree_targets(cleanup_pids, cleanup_pgids, signal.SIGTERM)
        try:
            proc.communicate(timeout=max(float(terminate_grace_seconds), 0.1))
        except subprocess.TimeoutExpired:
            current_pids, current_pgids = _process_tree_targets(proc.pid)
            _signal_process_tree_targets(
                cleanup_pids | current_pids,
                cleanup_pgids | current_pgids,
                signal.SIGKILL,
            )
            try:
                proc.communicate(timeout=1)
            except Exception:
                pass
        raise


def ordered_unique(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def status_rank(status: str) -> int:
    return STATUS_RANK.get(str(status or "").strip().lower(), 1)


def bool_status(ok: bool) -> str:
    return "ready" if ok else "blocked"


def _observed_fixed_holiday(year: int, month: int, day: int) -> date:
    holiday = date(year, month, day)
    if holiday.weekday() == 5:
        return holiday - timedelta(days=1)
    if holiday.weekday() == 6:
        return holiday + timedelta(days=1)
    return holiday


def _nth_weekday(year: int, month: int, weekday: int, nth: int) -> date:
    current = date(year, month, 1)
    offset = (weekday - current.weekday()) % 7
    return current + timedelta(days=offset + (nth - 1) * 7)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    current = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    return current - timedelta(days=(current.weekday() - weekday) % 7)


def _easter_date(year: int) -> date:
    # Meeus/Jones/Butcher Gregorian computus.
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def us_equity_market_holiday(day: date) -> str:
    year = day.year
    holidays = {
        _observed_fixed_holiday(year, 1, 1): "new_years_day",
        _nth_weekday(year, 1, 0, 3): "martin_luther_king_jr_day",
        _nth_weekday(year, 2, 0, 3): "washingtons_birthday",
        _easter_date(year) - timedelta(days=2): "good_friday",
        _last_weekday(year, 5, 0): "memorial_day",
        _observed_fixed_holiday(year, 6, 19): "juneteenth",
        _observed_fixed_holiday(year, 7, 4): "independence_day",
        _nth_weekday(year, 9, 0, 1): "labor_day",
        _nth_weekday(year, 11, 3, 4): "thanksgiving_day",
        _observed_fixed_holiday(year, 12, 25): "christmas_day",
    }
    return holidays.get(day, "")


def eastern_off_hours_window(
    *,
    now: datetime | None = None,
    start_local: time = time(16, 15),
    end_local: time = time(9, 20),
) -> dict[str, Any]:
    current = (now or utc_now()).astimezone(LOCAL_TZ)
    local_clock = current.timetz().replace(tzinfo=None)
    is_weekend = current.weekday() >= 5
    holiday_name = us_equity_market_holiday(current.date())
    market_holiday = bool(holiday_name)
    active = bool(is_weekend or market_holiday or local_clock >= start_local or local_clock < end_local)
    return {
        "active": active,
        "is_weekend": is_weekend,
        "market_holiday": market_holiday,
        "market_holiday_name": holiday_name,
        "timezone": "America/New_York",
        "local_time": current.isoformat(),
        "window_start_local": start_local.strftime("%H:%M"),
        "window_end_local": end_local.strftime("%H:%M"),
        "label": "off_hours" if active else "market_hours",
    }
