"""Native read-only candle refresh and bounded online-cache retention."""

from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import stat
import subprocess

from core.decision_price_evidence import latest_closed_end, session_bounds, timestamp
from core.schd_capture_store import (
    DIRECTORY,
    checked,
    latest_capture,
    publish_capture,
    prune_captures,
    read_capture,
)
from scripts.ops.long_runtime_common import write_payload

RECEIPT = DIRECTORY / "evidence_maintenance_latest.json"
AUTHORITY = {
    "live_execution_authority": False,
    "broker_mutation_attempted": False,
    "autonomous_execution": False,
}


def producer_observation(root, *, now):
    """Latest producer state is diagnostic, never decision or liveness proof."""
    path = (
        Path(root)
        / "governance/health/data_ingress_latest_dividend_equities_schwab.json"
    )
    result = {
        "status": "unavailable",
        "source": str(path.relative_to(root)),
        "decision_freshness_credit": False,
        "automatic_restart_allowed": False,
    }
    try:
        payload = read_capture(root, path)
        if any(
            payload.get(key) != expected
            for key, expected in (
                ("broker", "schwab"),
                ("profile", "dividend"),
                ("domain", "equities"),
            )
        ) or not payload.get("run_id"):
            raise ValueError("producer_scope_mismatch")
        age = (now - timestamp(payload["timestamp_utc"])).total_seconds()
        result.update(timestamp_utc=payload["timestamp_utc"], age_seconds=age)
        if not 0 <= age <= 300:
            return dict(result, status="stale_or_future_observation")
        state = str(payload.get("loop_state", ""))
        activity = str(payload.get("activity", ""))
        result.update(
            status=(
                "pause_observed"
                if state.startswith("paused_") or state in {"halted", "resume_stagger"}
                else (
                    "interval_wait_observed"
                    if state == "running" and activity == "interval_wait"
                    else "producer_state_observed"
                )
            ),
            run_id=payload["run_id"],
            loop_state=state,
            activity=activity,
            pause_gate=payload.get("pause_gate", ""),
            pause_reason=payload.get("pause_reason", ""),
            collector_pacing=payload.get("collector_pacing", {}),
        )
        return result
    except (ValueError, OSError, KeyError, TypeError):
        return result


def refresh_due(market, *, now):
    try:
        source = market["source"]
        age = (now - timestamp(source["fetch_started_at_utc"])).total_seconds()
        return (
            source.get("provider") != "schwab"
            or source.get("symbol") != "SCHD"
            or not 0 <= age < 120
            or any(
                timestamp(market["candles"][name][-1]["end_utc"])
                != latest_closed_end(now, minutes)
                for name, minutes in (("1m", 1), ("5m", 5), ("1d", None))
            )
        )
    except (ValueError, KeyError, TypeError, IndexError):
        return True


def maintenance_metric(root, *, now=None):
    now = now or datetime.now(timezone.utc)
    root = Path(root)
    try:
        plan = checked(root, root / "config/supervised_schd_broker_test_v1.json")
        off = checked(root, root / "governance/health/SYSTEM_POWER_OFF.flag")
        if not plan.exists() or off.exists():
            return {
                "refresh_due": False,
                "stale": False,
                "state": "disabled_or_not_configured",
            }
        path = checked(root, root / RECEIPT)
        try:
            receipt = read_capture(root, path) if path.exists() else {}
            age = (
                (now - timestamp(receipt["timestamp_utc"])).total_seconds()
                if receipt
                else None
            )
        except (ValueError, OSError, KeyError, TypeError):
            age = None
        bounds = session_bounds(now)
        active = bool(bounds and bounds[0] <= now < bounds[1])
        due = age is None or age < 0 or age >= (60 if active else 3600)
        return {
            "refresh_due": due,
            "stale": due,
            "age_seconds": age,
            "regular_session": active,
            "state": "maintenance_due" if due else "recent_pass",
        }
    except (ValueError, OSError, KeyError, TypeError):
        return {
            "refresh_due": False,
            "stale": True,
            "state": "unsafe_or_invalid_maintenance_route",
        }


def maintain(root, *, now=None, fetcher=None):
    from scripts.ops.schd_candle_report import FETCH_FAILURE_REASONS, fetch_bounded
    from scripts.ops.schd_native_decision import read_latest

    fixed_now = now
    now = now or datetime.now(timezone.utc)
    root = Path(root)
    base = {
        "timestamp_utc": now.isoformat(),
        "purpose": "schd_evidence_maintenance",
        **AUTHORITY,
    }
    off = checked(root, root / "governance/health/SYSTEM_POWER_OFF.flag")
    if off.exists():
        return dict(base, ok=False, state="system_power_off")
    directory = checked(root, root / DIRECTORY)
    directory.mkdir(parents=True, exist_ok=True)
    fd = os.open(
        checked(root, directory / "writer.lock"),
        os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
        0o600,
    )
    with os.fdopen(fd, "a+") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("regular_schd_writer_lock_required")
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return dict(base, ok=False, state="writer_busy_previous_receipt_preserved")
        market = latest_capture(root)
        retention = prune_captures(root, now=now)
        bounds = session_bounds(now)
        active = bool(bounds and bounds[0] <= now < bounds[1])
        refresh = {
            "state": "outside_regular_session" if not active else "already_current"
        }
        if active and refresh_due(market, now=now):
            try:
                captured = (fetcher or fetch_bounded)(root, timeout_seconds=25)
                # Validate the clock and provider without changing the declared basis.
                source = captured["source"]
                observed_at = fixed_now or datetime.now(timezone.utc)
                age = (
                    observed_at - timestamp(source["fetch_started_at_utc"])
                ).total_seconds()
                if (
                    not 0 <= age <= 120
                    or source.get("provider") != "schwab"
                    or source.get("symbol") != "SCHD"
                ):
                    raise ValueError("invalid_fresh_schwab_capture")
                identity = publish_capture(root, captured)
                market = captured
                refresh = {"state": "refreshed", "capture_sha256": identity}
            except (
                ValueError,
                OSError,
                KeyError,
                TypeError,
                subprocess.TimeoutExpired,
            ) as exc:
                refresh = {
                    "state": "failed_previous_capture_preserved",
                    "error_type": type(exc).__name__,
                    "reason": (
                        str(exc)
                        if str(exc) in FETCH_FAILURE_REASONS
                        else "bounded_capture_or_publication_failed"
                    ),
                }
        observed_at = fixed_now or datetime.now(timezone.utc)
        try:
            row, _, scan = read_latest(root, now=observed_at)
        except (ValueError, OSError):
            row, scan = None, {"issues": ["native_decision_source_unavailable"]}
        age = (
            (observed_at - timestamp(row["timestamp_utc"])).total_seconds()
            if row
            else None
        )
        native = {
            "timestamp_utc": row.get("timestamp_utc") if row else None,
            "action": row.get("action") if row else None,
            "age_seconds": age,
            "fresh": age is not None and 0 <= age <= 120 and not scan["issues"],
            "scan_issues": scan["issues"],
            "source": "shadow_dividend_equities/grand_master_bot",
            "heartbeat_is_not_decision_freshness": True,
            "automatic_restart_or_timestamp_rewrite": False,
            "producer_observation": producer_observation(root, now=observed_at),
        }
        result = dict(
            base,
            timestamp_utc=observed_at.isoformat(),
            ok=refresh["state"] != "failed_previous_capture_preserved",
            state="maintenance_observation_not_trade_clearance",
            market_refresh=refresh,
            retention=retention,
            native_decision=native,
            market_source=market.get("source", {}),
            current_closed_candles_available=(
                not refresh_due(market, now=observed_at) if active else None
            ),
        )
        if active and not result["current_closed_candles_available"]:
            result["ok"] = False
        write_payload(checked(root, root / RECEIPT), result)
        return result
