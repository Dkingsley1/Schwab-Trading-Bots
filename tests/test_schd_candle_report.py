from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageStat
import pytest

from core.decision_price_evidence import build_evidence, timestamp
from scripts.ops.schd_candle_report import (
    READ_ONLY_ENV,
    chart_report,
    fetch_bounded,
    fetch_with_client,
    normalize_schwab_candles,
    render_charts,
)
from scripts.ops.schd_decision_rehearsal import synthetic_packet


def raw_bar(start, **changes):
    return dict(
        datetime=int(timestamp(start).timestamp() * 1000),
        open=32,
        high=33,
        low=31,
        close=32.5,
        volume=1000,
        **changes,
    )


def test_daily_mapping_closed_bars_and_wrong_symbols():
    now = timestamp("2026-09-23T15:00:00+00:00")
    payload = {
        "symbol": "SCHD",
        "candles": [
            raw_bar("2026-09-22T04:00:00+00:00"),
            raw_bar("2026-09-23T04:00:00+00:00"),
        ],
    }
    rows, excluded = normalize_schwab_candles(payload, minutes=None, asof=now)
    assert excluded == 1
    assert rows[0]["start_utc"] == "2026-09-22T13:30:00+00:00"
    assert rows[0]["end_utc"] == "2026-09-22T20:00:00+00:00"
    payload["symbol"] = "O"
    with pytest.raises(ValueError, match="symbol"):
        normalize_schwab_candles(payload, minutes=None, asof=now)


def test_fetch_client_only_three_history_gets_and_http_failure():
    calls = []

    def method(symbol, **kwargs):
        assert symbol == "SCHD"
        assert kwargs["need_extended_hours_data"] is False
        calls.append(kwargs)
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "symbol": "SCHD",
                "candles": [raw_bar("2026-09-22T13:30:00+00:00")],
            },
        )

    client = SimpleNamespace(
        get_price_history_every_day=method,
        get_price_history_every_five_minutes=method,
        get_price_history_every_minute=method,
    )
    result = fetch_with_client(client, now=timestamp("2026-09-23T15:00:00+00:00"))
    assert len(calls) == 3
    assert set(result["candles"]) == {"1d", "5m", "1m"}
    assert result["source"]["price_adjustment_basis"].endswith(
        "not_independently_verified"
    )

    def fail(*args, **kwargs):
        raise ValueError("http_failure")

    client.get_price_history_every_day = fail
    with pytest.raises(ValueError, match="http_failure"):
        fetch_with_client(client, now=timestamp("2026-09-23T15:00:00+00:00"))


def test_subprocess_bound_live_locks_and_no_secret_error(monkeypatch):
    def child(command, **kwargs):
        assert command[-1] == "--fetch-child"
        assert kwargs["timeout"] == 90
        assert all(kwargs["env"][key] == value for key, value in READ_ONLY_ENV.items())
        return SimpleNamespace(
            returncode=2, stdout="SECRET_RESPONSE", stderr="SECRET_AUTH"
        )

    monkeypatch.setattr("scripts.ops.schd_candle_report.subprocess.run", child)
    with pytest.raises(ValueError) as error:
        fetch_bounded(Path("/tmp"))
    assert "SECRET" not in str(error.value)


def test_chart_capture_is_not_a_bot_decision():
    packet = synthetic_packet()
    report = chart_report(
        {
            "candles": packet["candles"],
            "source": {
                "provider": "schwab",
                "fetch_started_at_utc": packet["decision"]["timestamp_utc"],
            },
        },
        now=timestamp(packet["decision"]["timestamp_utc"]),
    )
    assert report["decision_status"] == "WAIT"
    assert report["bot_record"]["reasons"] == []
    assert (
        "chart_capture_is_not_a_bot_decision_or_an_executable_quote"
        in report["blockers"]
    )
    assert not report["live_execution_authority"]


def test_known_cooldown_reason_is_preserved(monkeypatch):
    monkeypatch.setattr(
        "scripts.ops.schd_candle_report.subprocess.run",
        lambda *a, **k: SimpleNamespace(
            returncode=2,
            stdout='{"reason":"schwab_provider_cooldown_active"}',
            stderr="",
        ),
    )
    with pytest.raises(ValueError, match="schwab_provider_cooldown_active"):
        fetch_bounded(Path("/tmp"))


def test_candle_diagrams_nonblank_and_bounded(tmp_path):
    packet = synthetic_packet()
    report = build_evidence(packet, now=timestamp(packet["decision"]["timestamp_utc"]))
    paths = render_charts(
        report, directory=tmp_path, prefix="synthetic", checked_path=lambda p: p
    )
    assert set(paths) == {"5m", "15m", "1h", "1d", "1M", "180d", "1Y"}
    for path in paths.values():
        with Image.open(path) as img:
            assert img.size[0] >= 1000 and img.size[1] >= 500
            assert min(ImageStat.Stat(img.convert("RGB")).stddev) > 15
    assert sum(Path(p).stat().st_size for p in paths.values()) < 2_000_000
    render_charts(
        report, directory=tmp_path, prefix="synthetic", checked_path=lambda p: p
    )
    assert len(list(tmp_path.glob("*.png"))) == 7
