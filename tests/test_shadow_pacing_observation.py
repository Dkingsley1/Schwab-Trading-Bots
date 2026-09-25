import ast
from datetime import datetime, timedelta, timezone
import inspect
import json
from pathlib import Path

import pytest

import scripts.run_shadow_training_loop as loop
from scripts.ops.schd_evidence_maintenance import producer_observation


def test_external_pressure_release_does_not_change_adaptive_state():
    adaptive = 120
    assert (
        loop._collection_interval_seconds(
            adaptive_interval_seconds=adaptive,
            base_interval_seconds=120,
            external_extra_seconds=75,
        )
        == 195
    )
    assert (
        loop._collection_interval_seconds(
            adaptive_interval_seconds=adaptive,
            base_interval_seconds=120,
            external_extra_seconds=0,
        )
        == 120
    )
    # Removing external pressure must not undo a separate memory/overload throttle.
    assert (
        loop._collection_interval_seconds(
            adaptive_interval_seconds=240,
            base_interval_seconds=120,
            external_extra_seconds=0,
        )
        == 240
    )


def test_run_loop_preserves_separate_pacing_and_existing_duty_limit():
    source = inspect.getsource(loop.run_loop)
    assert "current_interval_seconds = external_floor" not in source
    assert "interval_seconds=scheduled_interval_seconds" in source
    assert 'activity="interval_wait", pacing=pacing, emit_summary=False' in source
    assert 'sleep_s = float(duty_cycle["sleep_seconds"])' in source
    assert "time.sleep(sleep_s)" in source


@pytest.mark.parametrize(
    "state,gate,reason",
    [
        ("paused_event_gate", "event_blackout", "event_lock_window"),
        ("paused_session_gate", "session_gate", "post_window"),
        (
            "paused_market_data_provider_cooldown",
            "market_data_provider_cooldown",
            "provider_http_401_403_429",
        ),
        ("paused_anomaly_killswitch", "anomaly_killswitch", "data_anomaly"),
        (
            "paused_runtime_backpressure",
            "cooperative_ingestion_backpressure",
            "storage_pressure",
        ),
    ],
)
def test_pause_publication_updates_heartbeat_and_ingress_together(state, gate, reason):
    # Execute the nested publisher alone; starting run_loop would create workers.
    tree = ast.parse(inspect.getsource(loop.run_loop))
    publisher = next(
        n
        for n in tree.body[0].body
        if isinstance(n, ast.FunctionDef) and n.name == "_publish_ingress_state"
    )
    heartbeats, ingress, events = [], [], []
    scope = dict(vars(loop))
    scope.update(
        loop_state=state,
        loop_state_reason=reason,
        iter_ingress={},
        ingress_totals={},
        iter_count=3,
        broker="schwab",
        symbols=["SCHD"],
        context_symbols=[],
        _write_heartbeat=lambda **kwargs: heartbeats.append(kwargs),
        _write_ingress_state=lambda **kwargs: ingress.append(kwargs["payload"]),
        _append_jsonl=lambda *args: events.append(args),
        _event_bus_path=lambda *args: "unused",
        _shadow_ingress_instance=lambda: "",
    )
    exec(
        compile(ast.Module(body=[publisher], type_ignores=[]), "<publisher>", "exec"),
        scope,
    )
    scope["_publish_ingress_state"](pause_gate=gate, pause_reason=reason)
    assert heartbeats[-1]["state"] == ingress[-1]["loop_state"] == state
    assert heartbeats[-1]["pause_gate"] == ingress[-1]["pause_gate"] == gate
    assert heartbeats[-1]["pause_reason"] == ingress[-1]["pause_reason"] == reason
    # A later pacing observation must not erase a still-active gate reason.
    scope["_publish_ingress_state"](
        activity="interval_wait",
        pacing={"scheduled_interval_seconds": 195},
        emit_summary=False,
    )
    assert heartbeats[-1]["pause_reason"] == reason
    assert ingress[-1]["collector_pacing"]["scheduled_interval_seconds"] == 195
    assert len(events) == 1


def test_heartbeat_clears_previous_pause_fields_when_work_resumes(monkeypatch):
    rows = []
    monkeypatch.setattr(
        loop, "safe_write_json_atomic", lambda path, payload, **kw: rows.append(payload)
    )
    args = dict(
        project_root="/unused",
        broker="schwab",
        iter_count=1,
        symbols_total=1,
        context_total=0,
    )
    loop._write_heartbeat(
        **args,
        state="paused_event_gate",
        pause_gate="event_blackout",
        pause_reason="event_lock_window"
    )
    loop._write_heartbeat(**args, state="running", progress_current=1, progress_total=1)
    assert rows[0]["pause_reason"] == "event_lock_window"
    assert "pause_reason" not in rows[1] and "pause_gate" not in rows[1]


def write_producer(root, now, **changes):
    value = dict(
        timestamp_utc=now.isoformat(),
        run_id="run-1",
        broker="schwab",
        profile="dividend",
        domain="equities",
        loop_state="paused_event_gate",
        pause_gate="event_blackout",
        pause_reason="event_lock_window",
    )
    value.update(changes)
    path = root / "governance/health/data_ingress_latest_dividend_equities_schwab.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.mark.parametrize("state", ["paused_event_gate", "resume_stagger"])
def test_recent_pause_is_diagnostic_not_fresh_decision_credit(tmp_path, state):
    now = datetime.now(timezone.utc)
    write_producer(tmp_path, now, loop_state=state)
    result = producer_observation(tmp_path, now=now)
    assert result["status"] == "pause_observed"
    assert result["pause_reason"] == "event_lock_window"
    assert not result["decision_freshness_credit"]
    assert not result["automatic_restart_allowed"]


@pytest.mark.parametrize("seconds", [-1, 301])
def test_old_or_future_pause_cannot_explain_current_state(tmp_path, seconds):
    now = datetime.now(timezone.utc)
    write_producer(tmp_path, now - timedelta(seconds=seconds))
    result = producer_observation(tmp_path, now=now)
    assert result["status"] == "stale_or_future_observation"
    assert "pause_reason" not in result


@pytest.mark.parametrize(
    "changes",
    [
        dict(profile="dividend_capture"),
        dict(broker="coinbase"),
        dict(domain="crypto"),
        dict(run_id=""),
        dict(timestamp_utc="bad"),
    ],
)
def test_invalid_producer_scope_or_clock_is_unavailable(tmp_path, changes):
    now = datetime.now(timezone.utc)
    write_producer(tmp_path, now, **changes)
    assert producer_observation(tmp_path, now=now)["status"] == "unavailable"


def test_pacing_report_retains_owner_components(tmp_path):
    now = datetime.now(timezone.utc)
    write_producer(
        tmp_path,
        now,
        loop_state="running",
        activity="interval_wait",
        pause_gate="",
        pause_reason="",
        collector_pacing={"scheduled_interval_seconds": 195},
    )
    result = producer_observation(tmp_path, now=now)
    assert result["status"] == "interval_wait_observed"
    assert result["collector_pacing"]["scheduled_interval_seconds"] == 195


def test_unsafe_missing_and_corrupt_producer_are_unavailable(tmp_path, monkeypatch):
    now = datetime.now(timezone.utc)
    assert producer_observation(tmp_path, now=now)["status"] == "unavailable"
    path = write_producer(tmp_path, now)
    path.write_text("{broken")
    assert producer_observation(tmp_path, now=now)["status"] == "unavailable"
    path.unlink()
    path.symlink_to("/Volumes/VIDEO/do-not-touch")
    original = Path.lstat

    def guarded(path, *args, **kwargs):
        assert not str(path).startswith("/Volumes/VIDEO")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", guarded)
    assert producer_observation(tmp_path, now=now)["status"] == "unavailable"
