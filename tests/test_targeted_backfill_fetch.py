import json
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from scripts.ops import collection_gap_census as gaps


def job():
    return gaps.candle_request(gaps.candle_scope({
        "provider": "coinbase_exchange", "symbol": "BTC-USD", "timeframe": "5m",
        "start_utc": "2026-09-24T00:00:00Z", "end_utc": "2026-09-24T00:10:00Z",
    }, datetime.now(timezone.utc)))


def test_fixed_public_transport_and_fresh_receipt(monkeypatch):
    import http.client
    request = job()
    start = int(gaps.timestamp(request["start_utc"]).timestamp())
    response = Mock(status=200)
    response.read.return_value = json.dumps([[start+i*300, 99, 102, 100, 101, 1] for i in range(2)]).encode()
    client = Mock()
    client.getresponse.return_value = response
    connect = Mock(return_value=client)
    monkeypatch.setattr(http.client, "HTTPSConnection", connect)
    receipt = gaps.fetch_public_coinbase_receipt(request)
    connect.assert_called_once_with("api.exchange.coinbase.com", timeout=10)
    assert client.request.call_args.args[0] == "GET"
    assert client.request.call_args.args[1].startswith("/products/BTC-USD/candles?")
    assert receipt["network_requests_performed"] is True
    assert receipt["coverage"]["missing"] == 0
    assert receipt["historical_live_evidence"] is False
    gaps.validate_receipt(receipt, datetime.now(timezone.utc), gaps.Budget())
    client.close.assert_called_once()


@pytest.mark.parametrize("failure", ["redirect", "oversize", "bad_json"])
def test_no_redirect_retry_or_unbounded_payload(monkeypatch, failure):
    import http.client
    response = Mock(status=302 if failure == "redirect" else 200)
    response.read.return_value = b"x"*(512*1024+1) if failure == "oversize" else b"not json"
    client = Mock()
    client.getresponse.return_value = response
    monkeypatch.setattr(http.client, "HTTPSConnection", Mock(return_value=client))
    with pytest.raises(ValueError):
        gaps.fetch_public_coinbase_receipt(job())
    assert client.request.call_count == 1
    client.close.assert_called_once()


def test_bad_binding_never_connects(monkeypatch):
    import http.client
    connect = Mock()
    monkeypatch.setattr(http.client, "HTTPSConnection", connect)
    request = job()
    request["endpoint"] = "https://other.example"
    with pytest.raises(ValueError):
        gaps.fetch_public_coinbase_receipt(request)
    connect.assert_not_called()
