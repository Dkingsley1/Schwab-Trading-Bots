import pytest

from core.event_time_control import EventTimeGuard, EventTimePolicy
from core.collector_transport import qualify_transport_event


def test_watermark_accepts_bounded_out_of_order_and_quarantines_late_data() -> None:
    guard = EventTimeGuard(EventTimePolicy(allowed_lateness_seconds=30.0))
    guard.ingest(
        stream_id="quotes",
        event_id="one",
        event_time_utc="2026-08-21T12:01:00+00:00",
        observed_at_utc="2026-08-21T12:01:01+00:00",
        payload={"price": 100},
    )
    bounded = guard.ingest(
        stream_id="quotes",
        event_id="two",
        event_time_utc="2026-08-21T12:00:45+00:00",
        observed_at_utc="2026-08-21T12:01:02+00:00",
        payload={"price": 99},
    )
    late = guard.ingest(
        stream_id="quotes",
        event_id="three",
        event_time_utc="2026-08-21T11:59:00+00:00",
        observed_at_utc="2026-08-21T12:01:03+00:00",
        payload={"price": 98},
    )

    assert bounded["accepted"] is True
    assert bounded["disposition"] == "out_of_order_within_bound"
    assert late["accepted"] is False
    assert late["reason"] == "event_arrived_after_watermark"


def test_dedupe_conflict_future_skew_and_snapshot_checksum_are_fail_closed() -> None:
    guard = EventTimeGuard(EventTimePolicy(max_future_skew_seconds=2.0))
    kwargs = {
        "stream_id": "trades",
        "event_id": "one",
        "event_time_utc": "2026-08-21T12:00:00+00:00",
        "observed_at_utc": "2026-08-21T12:00:01+00:00",
    }
    first = guard.ingest(**kwargs, payload={"price": 100})
    duplicate = guard.ingest(**kwargs, payload={"price": 100})
    conflict = guard.ingest(**kwargs, payload={"price": 101})
    future = guard.ingest(
        stream_id="trades",
        event_id="future",
        event_time_utc="2026-08-21T12:00:10+00:00",
        observed_at_utc="2026-08-21T12:00:01+00:00",
        payload={"price": 102},
    )
    restored = EventTimeGuard.restore(guard.snapshot())

    assert first["accepted"] is True
    assert duplicate["disposition"] == "duplicate"
    assert conflict["reason"] == "event_identity_payload_conflict"
    assert future["reason"] == "event_time_future_skew"
    assert restored.stream_status("trades") == guard.stream_status("trades")

    tampered = guard.snapshot()
    tampered["streams"]["trades"]["accepted_count"] = 999
    with pytest.raises(ValueError, match="checksum"):
        EventTimeGuard.restore(tampered)


def test_transport_success_and_event_time_usability_remain_separate_facts() -> None:
    guard = EventTimeGuard(EventTimePolicy(allowed_lateness_seconds=10.0))
    transport = {
        "ok": True,
        "request_id": "request-one",
        "fetched_utc": "2026-08-21T12:01:00+00:00",
        "payload_sha256": "payload-one",
    }
    qualified = qualify_transport_event(
        transport,
        guard=guard,
        stream_id="quotes",
        source_event_time_utc="2026-08-21T12:00:59+00:00",
    )
    guard.ingest(
        stream_id="quotes",
        event_id="newer",
        event_time_utc="2026-08-21T12:02:00+00:00",
        observed_at_utc="2026-08-21T12:02:01+00:00",
        payload={"price": 101},
    )
    late = qualify_transport_event(
        {
            **transport,
            "request_id": "request-late",
            "fetched_utc": "2026-08-21T12:02:02+00:00",
        },
        guard=guard,
        stream_id="quotes",
        source_event_time_utc="2026-08-21T12:00:00+00:00",
    )

    assert qualified["ok"] is True
    assert qualified["event_time_usable"] is True
    assert late["ok"] is True
    assert late["event_time_usable"] is False
    assert late["quarantined"] is True
