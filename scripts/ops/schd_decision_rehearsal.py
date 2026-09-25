"""Offline SCHD evidence review and isolated conditional round-trip simulation."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.decision_price_evidence import digest, timestamp
from core.schd_capture_store import decision_capture, publish_capture
from core.schd_decision_rehearsal import (
    AUTHORITY,
    advance,
    initial_state,
    render_markdown,
    validate_state,
)
from core.storage_router import inspect_storage_path
from scripts.ops.long_runtime_common import write_text_atomic

DIRECTORY = "governance/rehearsals/schd"
MAX_BYTES = 6 * 1024 * 1024


def local(path, *, root=ROOT, require_root=True):
    path = Path(os.path.abspath(path))
    route = inspect_storage_path(
        path, boundary_root=root if require_root else None, allow_external=False
    )
    if route.get("status") not in {"present", "missing"} or route.get("symlinks"):
        raise ValueError("unsafe_rehearsal_path")
    return path


def read_json(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_BYTES:
            raise ValueError("bounded_regular_json_file_required")
        raw = handle.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError("input_exceeds_byte_limit")
    value = json.loads(
        raw,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json")),
    )
    if not isinstance(value, dict):
        raise ValueError("json_object_required")
    return value


def implementation_digest(root=ROOT):
    sources = {}
    for relative in (
        "core/decision_price_evidence.py",
        "core/schd_market_evidence.py",
        "core/schd_capture_store.py",
        "core/schd_decision_rehearsal.py",
        "core/execution_simulator.py",
        "scripts/ops/schd_decision_rehearsal.py",
        "scripts/ops/schd_candle_report.py",
        "scripts/ops/decision_chart_report.py",
        "scripts/ops/schd_native_decision.py",
    ):
        sources[relative] = hashlib.sha256(
            local(root / relative, root=root).read_bytes()
        ).hexdigest()
    # Only numeric simulator settings are included; no credential environment.
    simulator_env = {
        key: value for key, value in os.environ.items() if key.startswith("EXEC_SIM_")
    }
    return digest({"sources": sources, "simulator_environment": simulator_env})


def synthetic_packet():
    """Clearly fictional candles/decisions for wiring tests, never live evidence."""
    import exchange_calendars
    import pandas as pd

    asof = timestamp("2026-09-23T15:05:01+00:00")
    calendar = exchange_calendars.get_calendar("XNYS")
    sessions = calendar.sessions_in_range(
        pd.Timestamp("2025-01-01"), pd.Timestamp("2026-09-23")
    )
    daily, intraday = [], []
    for index, day in enumerate(sessions):
        opened = calendar.session_open(day).to_pydatetime()
        closed = calendar.session_close(day).to_pydatetime()
        price = 28 + index * 0.01
        if closed <= asof:
            daily.append(
                {
                    "start_utc": opened.isoformat(),
                    "end_utc": closed.isoformat(),
                    "open": price,
                    "high": price + 0.2,
                    "low": price - 0.1,
                    "close": price + 0.1,
                    "volume": 1_000_000,
                }
            )
        if index >= len(sessions) - 8:
            start = opened
            while start + timedelta(minutes=5) <= min(closed, asof):
                value = 32 + len(intraday) * 0.0005
                intraday.append(
                    {
                        "start_utc": start.isoformat(),
                        "end_utc": (start + timedelta(minutes=5)).isoformat(),
                        "open": value,
                        "high": value + 0.02,
                        "low": value - 0.01,
                        "close": value + 0.01,
                        "volume": 10000,
                    }
                )
                start += timedelta(minutes=5)
    return {
        "schema_version": 1,
        "symbol": "SCHD",
        "evidence_kind": "synthetic",
        "candidate_id": "SYNTHETIC_WIRING_TEST_NOT_A_RELEASE",
        "price_basis": "split_adjusted_dividends_unadjusted",
        "decision": {
            "timestamp_utc": asof.isoformat(),
            "symbol": "SCHD",
            "decision_id": "synthetic-buy-1",
            "strategy": "SYNTHETIC_FIXTURE_NOT_A_TRAINED_BOT",
            "action": "BUY",
            "decision": "EXECUTE",
            "model_score": 0.75,
            "threshold": 0.6,
            "gates": {"fixture_gate": True},
            "reasons": [
                "Synthetic BUY exercises the wiring; not an investment signal."
            ],
            "features": {"fixture": True},
            "metadata": {"snapshot_id": "synthetic-entry-quote"},
        },
        "quote": {
            "symbol": "SCHD",
            "provider": "schwab",
            "source_quality_label": "broker_native",
            "snapshot_id": "synthetic-entry-quote",
            "timestamp_utc": asof.isoformat(),
            "bid": 32.99,
            "ask": 33.0,
            "last": 33.0,
            "bid_size": 10000,
            "ask_size": 10000,
        },
        "candles": {"5m": intraday, "1d": daily},
        "corporate_actions": [],
    }


def demo(*, source_digest):
    packet = synthetic_packet()
    now = timestamp(packet["decision"]["timestamp_utc"])
    state = initial_state(
        candidate_id=packet["candidate_id"],
        evidence_kind="synthetic",
        implementation_sha256=source_digest,
    )
    state = advance(state, packet, now=now, implementation_sha256=source_digest)
    fill = {
        "candidate_id": packet["candidate_id"],
        "evidence_kind": "synthetic",
        "fill_quote": dict(
            packet["quote"],
            snapshot_id="synthetic-buy-fill",
            timestamp_utc=(now + timedelta(seconds=1)).isoformat(),
        ),
    }
    state = advance(
        state, fill, now=now + timedelta(seconds=1), implementation_sha256=source_digest
    )
    sell = deepcopy(packet)
    sell["decision"].update(
        action="SELL",
        decision_id="synthetic-sell-1",
        timestamp_utc=(now + timedelta(seconds=2)).isoformat(),
        reasons=[
            "Synthetic SELL tests the separate exit and reconciliation; not a prediction."
        ],
    )
    sell["decision"]["metadata"]["snapshot_id"] = "synthetic-exit-quote"
    sell["quote"].update(
        bid=33.1,
        ask=33.11,
        last=33.1,
        timestamp_utc=(now + timedelta(seconds=2)).isoformat(),
        snapshot_id="synthetic-exit-quote",
    )
    state = advance(
        state, sell, now=now + timedelta(seconds=2), implementation_sha256=source_digest
    )
    fill["fill_quote"] = dict(
        sell["quote"],
        snapshot_id="synthetic-sell-fill",
        timestamp_utc=(now + timedelta(seconds=3)).isoformat(),
    )
    return advance(
        state, fill, now=now + timedelta(seconds=3), implementation_sha256=source_digest
    )


def run(
    action,
    *,
    input_path=None,
    root=ROOT,
    now=None,
    source_digest=None,
    refresh_market_data=False,
):
    fixed_now = now
    now = now or datetime.now(timezone.utc)
    source_digest = source_digest or implementation_digest(root)
    destination = local(root / DIRECTORY, root=root)
    state_path = local(destination / "recorded_state.json", root=root)
    if refresh_market_data and action != "native":
        raise ValueError("market_refresh_requires_native_action")
    if action == "status":
        state = (
            read_json(state_path)
            if state_path.exists()
            else initial_state(
                candidate_id="unbound_no_recorded_decision",
                evidence_kind="recorded",
                implementation_sha256=source_digest,
            )
        )
        validate_state(state)
        return dict(
            state,
            implementation_current=state["implementation_sha256"] == source_digest,
            source_data_connected=bool(
                state.get("native_observation", {}).get("source_selected")
            ),
            native_decision_fresh=(
                bool(state.get("native_connection"))
                and bool(state.get("native_observation", {}).get("source_selected"))
                and 0
                <= (
                    now
                    - timestamp(state["native_connection"]["decision_timestamp_utc"])
                ).total_seconds()
                <= 120
            ),
        )
    if local(root / "governance/health/SYSTEM_POWER_OFF.flag", root=root).exists():
        raise ValueError("system_power_off")
    destination.mkdir(parents=True, exist_ok=True)
    lock = local(destination / "writer.lock", root=root)
    fd = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    with os.fdopen(fd, "a+") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError("regular_lock_required")
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if action == "native":
            from scripts.ops.schd_native_decision import (
                advance_native,
                native_packet,
                read_latest,
            )

            market_path = local(destination / "market_latest.json", root=root)
            market = read_json(market_path) if market_path.exists() else {}
            refresh = {"requested": refresh_market_data, "state": "cached_only"}
            if refresh_market_data:
                from scripts.ops.schd_candle_report import fetch_bounded

                try:
                    market = fetch_bounded(root, include_quote=True)
                    publish_capture(root, market)
                    refresh["state"] = "refreshed"
                except (
                    OSError,
                    ValueError,
                    KeyError,
                    subprocess.TimeoutExpired,
                ) as exc:
                    refresh.update(
                        state="failed_previous_capture_preserved",
                        error_type=type(exc).__name__,
                    )
                    from scripts.ops.schd_candle_report import FETCH_FAILURE_REASONS

                    refresh["reason"] = (
                        str(exc)
                        if str(exc) in FETCH_FAILURE_REASONS
                        else "market_data_refresh_unavailable"
                    )
            observed_at = fixed_now or datetime.now(timezone.utc)
            # The caller's `now` may be fixed in tests; production observation
            # must happen after the bounded market-data child has completed.
            if refresh_market_data and refresh["state"] == "refreshed":
                observed_at = max(
                    observed_at, timestamp(market["source"]["fetch_started_at_utc"])
                )
            row, receipt, scan = read_latest(root, now=observed_at)
            candidate_path = local(
                root / "governance/runtime/production_candidate_state.json", root=root
            )
            candidate = read_json(candidate_path) if candidate_path.exists() else {}
            state = read_json(state_path) if state_path.exists() else None
            if row is not None:
                original_market = decision_capture(root, row)
                packet = native_packet(
                    row, receipt, scan, original_market, candidate, now=observed_at
                )
                state = advance_native(
                    state, packet, market, now=observed_at, source_digest=source_digest
                )
                if state.get("last_report"):
                    state["last_report"]["chart_source"] = original_market.get("source", {})
                from scripts.ops.decision_chart_report import publish_decision_chart

                # This is the report worker, never the trading loop. The original
                # decision and its source timestamps remain unchanged.
                state["decision_chart_report"] = publish_decision_chart(
                    root, row, now=observed_at
                )
                chart_path = state["decision_chart_report"].get("report_path")
                if chart_path:
                    state["decision_chart_report"]["report_path"] = str(root / chart_path)
            else:
                state = state or initial_state(
                    candidate_id=candidate.get("candidate_id")
                    or "unbound_no_native_decision",
                    evidence_kind="recorded",
                    implementation_sha256=source_digest,
                )
                if state.get("pending"):
                    state = advance(
                        state,
                        {
                            "candidate_id": state["candidate_id"],
                            "evidence_kind": "recorded",
                        },
                        now=observed_at,
                        implementation_sha256=source_digest,
                    )
            state["native_observation"] = {
                "observed_at_utc": observed_at.isoformat(),
                "source_selected": row is not None,
                "scan": scan,
                "market_refresh": refresh,
                "readiness": (
                    "WAIT"
                    if not row
                    or not state.get("last_report")
                    or state["last_report"]["blockers"]
                    else "simulation_only"
                ),
                "live_execution_authority": False,
            }
        elif action == "charts":
            from scripts.ops.schd_candle_report import chart_report, fetch_bounded

            captured = fetch_bounded(root)
            publish_capture(root, captured)
            state = initial_state(
                candidate_id="chart_context_only_not_bound_to_bot",
                evidence_kind="recorded",
                implementation_sha256=source_digest,
            )
            state["last_report"] = chart_report(
                captured, now=datetime.now(timezone.utc)
            )
            from scripts.ops.schd_native_decision import read_latest

            observed_at = fixed_now or datetime.now(timezone.utc)
            row, receipt, scan = read_latest(root, now=observed_at)
            if row is not None:
                state["last_report"]["recorded_decision_sample"] = {
                    "observed_at_utc": observed_at.isoformat(),
                    "decision_age_seconds": (
                        observed_at - timestamp(row["timestamp_utc"])
                    ).total_seconds(),
                    "record": {
                        key: deepcopy(row.get(key))
                        for key in (
                            "timestamp_utc", "decision_id", "strategy", "symbol",
                            "action", "decision", "model_score", "threshold",
                            "reasons", "gates",
                        )
                    },
                    "source_receipt": receipt,
                    "scan_issues": scan["issues"],
                    "chart_binding": "separate_current_context_not_original_decision_input",
                    "live_execution_authority": False,
                }
            state["chart_capture_ok"] = True
            state["wait_reason"] = (
                "chart_context_only_no_bot_decision_or_executable_quote"
            )
            state_path = local(destination / "schwab_charts_latest.json", root=root)
        elif action == "demo":
            state = demo(source_digest=source_digest)
            state_path = local(destination / "synthetic_latest.json", root=root)
        else:
            if not input_path:
                raise ValueError("explicit_input_packet_required")
            packet = read_json(local(input_path, root=root, require_root=False))
            if packet.get("evidence_kind") != "recorded":
                raise ValueError("recorded_input_required_synthetic_uses_demo")
            state = (
                read_json(state_path)
                if state_path.exists()
                else initial_state(
                    candidate_id=packet.get("candidate_id"),
                    evidence_kind="recorded",
                    implementation_sha256=source_digest,
                )
            )
            state = advance(state, packet, now=now, implementation_sha256=source_digest)
        # One atomic state contains both sides and their evidence. Markdown is a
        # disposable rendering, never the authority for replay prevention.
        validate_state(state)
        write_text_atomic(
            state_path, json.dumps(state, sort_keys=True, allow_nan=False)
        )
        markdown = local(state_path.with_suffix(".md"), root=root)
        from scripts.ops.schd_candle_report import render_charts

        reports = [order["decision_evidence"] for order in state["orders"]]
        if state.get("last_report") and not any(
            r["input_sha256"] == state["last_report"]["input_sha256"] for r in reports
        ):
            reports.append(state["last_report"])
        chart_paths = {}
        for index, report in enumerate(reports):
            chart_paths[report["input_sha256"]] = render_charts(
                report,
                directory=destination,
                prefix=f"{state_path.stem}_{index}",
                checked_path=lambda p: local(p, root=root),
            )
        write_text_atomic(markdown, render_markdown(state, chart_paths=chart_paths))
        return dict(state, report_path=str(markdown), state_path=str(state_path))


def main():
    if sys.argv[1:] in (["--fetch-child"], ["--fetch-child", "--with-quote"]):
        from scripts.ops.schd_candle_report import FETCH_FAILURE_REASONS, fetch_child

        try:
            if local(ROOT / "governance/health/SYSTEM_POWER_OFF.flag").exists():
                raise ValueError("system_power_off")
            print(
                json.dumps(
                    fetch_child(ROOT, include_quote="--with-quote" in sys.argv),
                    allow_nan=False,
                )
            )
            return 0
        except Exception as exc:
            reason = (
                str(exc)
                if str(exc) in FETCH_FAILURE_REASONS
                else "read_only_capture_failed"
            )
            print(json.dumps({"error_type": type(exc).__name__, "reason": reason}))
            return 2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("status", "evaluate", "demo", "charts", "native", "maintain"),
        nargs="?",
        default="status",
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--refresh-market-data", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        if args.action == "maintain":
            from scripts.ops.schd_evidence_maintenance import maintain

            if args.input or args.refresh_market_data:
                raise ValueError("maintenance_has_fixed_read_only_scope")
            result = maintain(ROOT)
            print(json.dumps(result, indent=2))
            return 0 if result["ok"] else 2
        state = run(
            args.action,
            input_path=args.input,
            refresh_market_data=args.refresh_market_data,
        )
        summary = {
            k: v
            for k, v in state.items()
            if k not in {"last_report", "orders", "pending"}
        }
        summary["modeled_order_count"] = len(state["orders"])
        if args.action == "native":
            report = state.get("last_report") or {}
            summary["decision_summary"] = {
                "action": report.get("bot_record", {}).get("action"),
                "timestamp_utc": report.get("as_of_utc"),
                "readiness": report.get("decision_status", "WAIT"),
                "blockers": report.get("blockers", []),
                "reasons": report.get("bot_record", {}).get("reasons", []),
            }
        summary["ok"] = (
            args.action == "status"
            or state["phase"] == "complete"
            or state.get("chart_capture_ok", False)
        )
        print(json.dumps(summary, indent=2))
        return 0 if summary["ok"] else 2
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "state": "blocked",
                    "error_type": type(exc).__name__,
                    "reason": str(exc),
                    **AUTHORITY,
                }
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
