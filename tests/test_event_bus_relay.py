from pathlib import Path

from scripts import event_bus_relay as src


def test_runtime_sources_prefer_channel_runtime_files(tmp_path: Path) -> None:
    legacy = tmp_path / "governance" / "events" / "runtime_events_20260417.jsonl"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text('{"event":"legacy_only"}\n', encoding="utf-8")

    channel = tmp_path / "governance" / "channels" / "runtime" / "default_equities_schwab" / "runtime_20260417.jsonl"
    channel.parent.mkdir(parents=True, exist_ok=True)
    channel.write_text('{"event":"channel_only"}\n', encoding="utf-8")

    paths = src._runtime_sources(tmp_path, "20260417")

    assert paths == [channel]


def test_relay_day_reads_channel_runtime_sources(tmp_path: Path) -> None:
    channel_a = tmp_path / "governance" / "channels" / "runtime" / "default_equities_schwab" / "runtime_20260417.jsonl"
    channel_b = tmp_path / "governance" / "channels" / "runtime" / "aggressive_equities_schwab" / "runtime_20260417.jsonl"
    channel_a.parent.mkdir(parents=True, exist_ok=True)
    channel_b.parent.mkdir(parents=True, exist_ok=True)
    channel_a.write_text('{"event":"decision_made","symbol":"SPY"}\n', encoding="utf-8")
    channel_b.write_text('{"event":"gate_blocked","symbol":"QQQ"}\n', encoding="utf-8")

    state_path = tmp_path / "governance" / "events" / "relay_state.json"
    payload = src.relay_day(tmp_path, day="20260417", state_path=state_path)

    decision_consumer = tmp_path / "governance" / "events" / "consumers" / "decision_made_20260417.jsonl"
    gate_consumer = tmp_path / "governance" / "events" / "consumers" / "gate_blocked_20260417.jsonl"

    assert payload["processed"] == 2
    assert payload["source_count"] == 2
    assert payload["mode"] == "channels"
    assert decision_consumer.exists()
    assert gate_consumer.exists()
