import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.shadow_watchdog import Target, _heartbeat_health


def _write_hb(path: Path, ts: datetime) -> None:
    path.write_text(json.dumps({"timestamp_utc": ts.isoformat()}), encoding="utf-8")


def test_heartbeat_health_uses_staleness(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    fresh = tmp_path / "fresh.json"
    stale = tmp_path / "stale.json"
    _write_hb(fresh, now - timedelta(seconds=10))
    _write_hb(stale, now - timedelta(seconds=500))

    t = Target(
        name="x",
        match="x",
        start_cmd="echo hi",
        heartbeat_glob=str(tmp_path / "*.json"),
        heartbeat_stale_seconds=120,
        min_healthy_heartbeats=1,
    )

    ok, count, age = _heartbeat_health(t)
    assert ok
    assert count == 1
    assert age is not None
