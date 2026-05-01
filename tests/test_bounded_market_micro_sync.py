from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.ops import bounded_market_micro_sync as sync


def test_timeout_payload_uses_fresh_last_known_payload(monkeypatch, tmp_path: Path) -> None:
    payload_path = tmp_path / "exports" / "external_context" / "market_micro_latest.json"
    payload_path.parent.mkdir(parents=True)
    payload_path.write_text(
        json.dumps({"timestamp_utc": datetime.now(timezone.utc).isoformat(), "derived": {"symbol_features": {"SPY": {}}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(sync, "PAYLOAD_PATH", payload_path)
    monkeypatch.setenv("MARKET_MICRO_TIMEOUT_FALLBACK_MAX_AGE_SECONDS", "86400")

    payload = sync._timeout_payload(
        subprocess.TimeoutExpired(cmd=["collector"], timeout=30),
        outer_timeout_seconds=30,
    )

    assert payload["ok"] is True
    assert payload["fallback_used"] is True
    assert payload["partial_data"] is True
    assert payload["sources"]["last_known_market_micro_payload"]["contract_participates"] is False


def test_timeout_payload_rejects_stale_last_known_payload(monkeypatch, tmp_path: Path) -> None:
    payload_path = tmp_path / "exports" / "external_context" / "market_micro_latest.json"
    payload_path.parent.mkdir(parents=True)
    stale_ts = datetime.now(timezone.utc) - timedelta(days=3)
    payload_path.write_text(json.dumps({"timestamp_utc": stale_ts.isoformat(), "derived": {}}), encoding="utf-8")
    monkeypatch.setattr(sync, "PAYLOAD_PATH", payload_path)
    monkeypatch.setenv("MARKET_MICRO_TIMEOUT_FALLBACK_MAX_AGE_SECONDS", "60")
    monkeypatch.setenv("MARKET_MICRO_TIMEOUT_NEUTRAL_FALLBACK_ENABLED", "0")

    payload = sync._timeout_payload(
        subprocess.TimeoutExpired(cmd=["collector"], timeout=30),
        outer_timeout_seconds=30,
    )

    assert payload["ok"] is False
    assert payload["fallback_used"] is False


def test_timeout_payload_writes_neutral_fallback_when_cache_is_stale(monkeypatch, tmp_path: Path) -> None:
    payload_path = tmp_path / "exports" / "external_context" / "market_micro_latest.json"
    payload_path.parent.mkdir(parents=True)
    stale_ts = datetime.now(timezone.utc) - timedelta(days=3)
    payload_path.write_text(json.dumps({"timestamp_utc": stale_ts.isoformat(), "derived": {}}), encoding="utf-8")
    monkeypatch.setattr(sync, "PAYLOAD_PATH", payload_path)
    monkeypatch.setenv("MARKET_MICRO_TIMEOUT_FALLBACK_MAX_AGE_SECONDS", "60")
    monkeypatch.setenv("MARKET_MICRO_TIMEOUT_NEUTRAL_FALLBACK_ENABLED", "1")

    payload = sync._timeout_payload(
        subprocess.TimeoutExpired(cmd=["collector"], timeout=30),
        outer_timeout_seconds=30,
    )
    fallback_body = json.loads(payload_path.read_text(encoding="utf-8"))

    assert payload["ok"] is True
    assert payload["neutral_fallback_used"] is True
    assert payload["sources"]["local_neutral_fallback"]["contract_participates"] is True
    assert fallback_body["derived"]["global_features"]["market_micro_tradeability_score_norm"] == 0.5
