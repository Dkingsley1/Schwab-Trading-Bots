import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts import observability_exporter as monitor


NOW = datetime(2026, 8, 10, 20, 0, tzinfo=timezone.utc)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _seed(project_root: Path, *, stale: str = "") -> None:
    for name, spec in monitor._surface_contract(project_root).items():
        timestamp = NOW - timedelta(hours=2) if name == stale else NOW
        status = "ready_idle" if name == "live_order_ledger_control" else "ready"
        _write(Path(spec["path"]), {"timestamp_utc": timestamp.isoformat(), "ok": True, "overall_status": status})


def test_local_deadman_is_ready_without_falsely_claiming_off_host_monitoring(tmp_path: Path) -> None:
    _seed(tmp_path)

    payload = monitor.build_payload(tmp_path, now=NOW)

    assert payload["local_monitor_ready"] is True
    assert payload["production_monitor_ready"] is False
    assert payload["overall_status"] == "degraded"
    assert payload["deadman_contract"]["paper_collection_blocked_by_receiver_absence"] is False


def test_local_storage_reserve_watch_is_locally_healthy(tmp_path: Path) -> None:
    _seed(tmp_path)
    storage_path = Path(monitor._surface_contract(tmp_path)["local_storage_reserve_guard"]["path"])
    _write(storage_path, {"timestamp_utc": NOW.isoformat(), "ok": True, "overall_status": "watch"})

    payload = monitor.build_payload(tmp_path, now=NOW)

    storage = next(row for row in payload["surfaces"] if row["name"] == "local_storage_reserve_guard")
    assert storage["ready"] is True
    assert payload["local_monitor_ready"] is True


def test_off_host_delivery_completes_production_monitor_contract(tmp_path: Path) -> None:
    _seed(tmp_path)

    def receiver(url: str, payload: dict, token: str, timeout: float) -> dict:
        assert url == "https://monitor.invalid/heartbeat"
        assert payload["live_execution_authority"] is False
        assert token == "secret"
        return {"ok": True, "status_code": 202, "error": ""}

    payload = monitor.build_payload(
        tmp_path,
        receiver_url="https://monitor.invalid/heartbeat",
        receiver_token="secret",
        deliver=True,
        receiver=receiver,
        now=NOW,
    )

    assert payload["production_monitor_ready"] is True
    assert payload["overall_status"] == "ready"
    assert payload["off_host_delivery"]["token_present"] is True


def test_stale_critical_surface_fails_local_monitor_closed(tmp_path: Path) -> None:
    _seed(tmp_path, stale="process_watchdog")

    payload = monitor.build_payload(tmp_path, now=NOW)

    assert payload["local_monitor_ready"] is False
    assert payload["overall_status"] == "blocked"
    assert "process_watchdog_stale" in payload["blockers"]


def test_prometheus_output_carries_surface_readiness() -> None:
    rendered = monitor.render_prometheus(
        {
            "timestamp_utc": NOW.isoformat(),
            "local_monitor_ready": True,
            "production_monitor_ready": False,
            "blockers": [],
            "surfaces": [{"name": "watchdog", "ready": True, "age_minutes": 1.5}],
        }
    )

    assert "trading_independent_monitor_local_ready 1.0" in rendered
    assert 'trading_independent_surface_ready{surface="watchdog"} 1.0' in rendered
