import json
import os

import pytest

from core import schwab_crypto_data as data

NOW = 1789063219.0


def quote(symbol="/BTCU26", root="/BTC", kind="future"):
    return {
        "symbol": symbol,
        "assetMainType": "FUTURE" if kind == "future" else "EQUITY",
        "assetSubType": None if kind == "future" else "ETF",
        "realtime": True,
        "reference": (
            {
                "product": root,
                "futureIsActive": True,
                "futureExpirationDate": (NOW + 86400 * 14) * 1000,
                "futureMultiplier": 5,
            }
            if kind == "future"
            else {}
        ),
        "quote": {
            "mark": 77000,
            "bidPrice": 76995,
            "askPrice": 77005,
            "quoteTime": (NOW - 1) * 1000,
            "bidTime": (NOW - 1) * 1000,
            "askTime": (NOW - 1) * 1000,
            "closePrice": 78000,
            "securityStatus": "Normal",
            "quotedInSession": True,
            "openInterest": 12000,
            "totalVolume": 25000,
        },
    }


def token_file(tmp_path, expires=NOW + 300):
    path = tmp_path / "token.json"
    path.write_text(
        json.dumps({"token": {"access_token": "test-secret", "expires_at": expires}})
    )
    path.chmod(0o600)
    return path


def test_native_future_root_resolves_to_exact_active_contract():
    row = data.normalize_quote({"/BTCU26": quote()}, data.INSTRUMENTS[0], NOW)
    assert row["usable"] is True
    assert row["source_symbol"] == "/BTCU26"
    assert row["requested_symbol"] == "/BTC"
    assert row["instrument_type"] == "future"
    assert row["spot_price_eligible"] is False
    assert row["quote_age_seconds"] == 1
    assert row["contract_multiplier"] == 5
    assert row["return_pct"] < 0


@pytest.mark.parametrize(
    "path,value,reason",
    [
        (("quote", "quoteTime"), (NOW - 121) * 1000, "stale_quote"),
        (("quote", "quoteTime"), (NOW + 1) * 1000, "quote_time_invalid"),
        (("quote", "quoteTime"), None, "quote_time_invalid"),
        (("quote", "quoteTime"), float("nan"), "quote_time_invalid"),
        (("quote", "quoteTime"), NOW, "stale_quote"),
        (("quote", "bidTime"), (NOW - 121) * 1000, "stale_or_invalid_book"),
        (("quote", "askTime"), (NOW + 1) * 1000, "stale_or_invalid_book"),
        (("quote", "mark"), None, "invalid_price_or_spread"),
        (("quote", "mark"), float("inf"), "invalid_price_or_spread"),
        (("quote", "mark"), True, "invalid_price_or_spread"),
        (("quote", "mark"), 50, "implausible_price_or_spread"),
        (("quote", "askPrice"), 70000, "invalid_price_or_spread"),
        (("quote", "askPrice"), 99000, "implausible_price_or_spread"),
        (("quote", "securityStatus"), "Halted", "market_not_normal"),
        (("quote", "quotedInSession"), False, "market_not_normal"),
        (("realtime",), "true", "delayed_or_unverified"),
        (("realtime",), False, "delayed_or_unverified"),
        (("assetMainType",), "EQUITY", "instrument_identity_mismatch"),
        (("reference", "futureIsActive"), False, "instrument_identity_mismatch"),
        (
            ("reference", "futureExpirationDate"),
            NOW * 1000,
            "instrument_identity_mismatch",
        ),
        (("symbol",), "/BTC-BAD", "instrument_identity_mismatch"),
    ],
)
def test_bad_quotes_never_produce_prices(path, value, reason):
    raw = quote()
    target = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    result = data.normalize_quote({"/BTCU26": raw}, data.INSTRUMENTS[0], NOW)
    assert result["quality"] == reason
    assert result["usable"] is False
    assert "mark" not in result


def test_missing_symbols_wrong_roots_duplicates_and_close_only_rejected():
    for payload in (
        {"BTC": {"lastPrice": 70000}},
        {"/BTCU26": quote(root="/MBT")},
        {"a": quote(), "b": quote()},
    ):
        assert not data.normalize_quote(payload, data.INSTRUMENTS[0], NOW)["usable"]
    raw = quote()
    del raw["quote"]["mark"]
    assert not data.normalize_quote({"x": raw}, data.INSTRUMENTS[0], NOW)["usable"]


@pytest.mark.parametrize("subtype", ["ETF", "CEF"])
def test_fund_shares_are_context_never_spot(subtype):
    raw = quote("IBIT", kind="etp")
    raw["assetSubType"] = subtype
    row = data.normalize_quote({"IBIT": raw}, data.INSTRUMENTS[2], NOW)
    assert row["usable"]
    assert row["instrument_type"] == "etp"
    assert row["spot_price_eligible"] is False
    assert row["return_baseline"] == "previous_close"


def test_token_read_is_nonmutating_and_reports_existing_permission_debt(tmp_path):
    path = token_file(tmp_path)
    before = path.read_bytes()
    assert data.read_access_token(path, NOW) == ("test-secret", False)
    path.chmod(0o644)
    assert data.read_access_token(path, NOW) == ("test-secret", True)
    assert path.read_bytes() == before
    assert path.stat().st_mode & 0o777 == 0o644


@pytest.mark.parametrize(
    "fault",
    [
        "expired",
        "symlink",
        "writable",
        "oversize",
        "bad_json",
        "header_injection",
        "fifo",
    ],
)
def test_unsafe_or_expired_tokens_fail_closed(tmp_path, fault):
    path = token_file(tmp_path, NOW if fault == "expired" else NOW + 300)
    if fault == "symlink":
        link = tmp_path / "link.json"
        link.symlink_to(path)
        path = link
    elif fault == "writable":
        path.chmod(0o666)
    elif fault == "oversize":
        path.write_bytes(b"x" * 65537)
    elif fault == "bad_json":
        path.write_bytes(b"\xff")
    elif fault == "header_injection":
        path.write_text(
            json.dumps(
                {"token": {"expires_at": NOW + 300, "access_token": "secret\r\nx:y"}}
            )
        )
    elif fault == "fifo":
        path.unlink()
        os.mkfifo(path, 0o600)
    with pytest.raises(data.SchwabContextError) as error:
        data.read_access_token(path, NOW)
    assert "secret" not in str(error.value)


def fake_transport(monkeypatch, *, status=200, body=b"{}", encoding="identity"):
    calls = []

    class Response:
        def __init__(self):
            self.status = status
            self.body = body

        def getheader(self, name, default):
            return encoding

        def read1(self, size):
            chunk, self.body = self.body[:size], self.body[size:]
            return chunk

    class Connection:
        sock = None

        def __init__(self, host, timeout):
            calls.append((host, timeout))

        def connect(self):
            pass

        def request(self, method, path, headers):
            calls.append((method, path, headers))

        def getresponse(self):
            return Response()

        def close(self):
            calls.append("closed")

    monkeypatch.setattr(data.http.client, "HTTPSConnection", Connection)
    return calls


def test_only_fixed_batch_get_and_no_account_or_order_requests(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "https://untrusted.invalid")
    calls = fake_transport(monkeypatch)
    assert (
        data.fetch_quotes(["/BTC", "IBIT"], access_token="test-secret", timeout=3) == {}
    )
    assert calls[0] == ("api.schwabapi.com", 3)
    assert calls[1][0] == "GET"
    assert calls[1][1].startswith("/marketdata/v1/quotes?symbols=%2FBTC%2CIBIT&")
    assert calls[-1] == "closed"


@pytest.mark.parametrize(
    "status,body,encoding,reason",
    [
        (302, b"secret", "identity", "http_302"),
        (401, b"secret", "identity", "auth_refresh_required"),
        (403, b"secret", "identity", "http_403"),
        (429, b"secret", "identity", "http_429"),
        (200, b"x" * (data.MAX_BYTES + 1), "identity", "response_too_large"),
        (200, b"not-json-secret", "identity", "quote_request_failed"),
        (200, b"[]", "identity", "invalid_quote_payload"),
        (200, b"secret", "gzip", "unsupported_response_encoding"),
    ],
)
def test_transport_fails_without_redirect_retry_or_secret_output(
    monkeypatch, status, body, encoding, reason
):
    calls = fake_transport(monkeypatch, status=status, body=body, encoding=encoding)
    with pytest.raises(data.SchwabContextError, match=f"^{reason}$"):
        data.fetch_quotes(["/BTC"], access_token="test-secret", timeout=2)
    assert len(calls) == 3
    assert calls[-1] == "closed"


def test_transport_deadline_and_allowlist(monkeypatch):
    fake_transport(monkeypatch)
    times = iter([0, 4])
    monkeypatch.setattr(data.time, "monotonic", lambda: next(times))
    with pytest.raises(data.SchwabContextError, match="deadline"):
        data.fetch_quotes(["/BTC"], access_token="secret", timeout=2)
    with pytest.raises(data.SchwabContextError, match="unsupported_instruments"):
        data.fetch_quotes(["BTC-USD"], access_token="secret", timeout=2)


def test_collector_preserves_partial_coverage_and_feature_lineage(
    monkeypatch, tmp_path
):
    path = token_file(tmp_path)
    monkeypatch.setattr(data.time, "time", lambda: NOW)
    calls = []

    def fetch(symbols, **kwargs):
        calls.append(symbols)
        return {"/BTCU26": quote(), "IBIT": quote("IBIT", kind="etp")}

    monkeypatch.setattr(data, "fetch_quotes", fetch)
    features, status = data.collect_context(
        ["BTC", "ETH", "UNKNOWN"], token_path=path, timeout=2
    )
    assert len(calls) == 1 and len(calls[0]) == 8
    assert status["state"] == "partial"
    assert status["usable_instruments"] == 2
    assert status["resolved_assets"] == 1
    assert set(features) == {"BTC"}
    assert features["BTC"]["crypto_schwab_future_return_norm"] < 0.5
    assert set(features["BTC"]) <= set(data.FEATURE_KEYS)
    assert status["spot_feed_verified"] is False
    assert status["live_execution_allowed"] is False
    assert status["transfers_allowed"] is False
    assert "test-secret" not in json.dumps(status)


def test_no_request_when_token_expired_or_assets_unsupported(monkeypatch, tmp_path):
    monkeypatch.setattr(data.time, "time", lambda: NOW)
    monkeypatch.setattr(
        data, "fetch_quotes", lambda *a, **k: pytest.fail("unexpected request")
    )
    path = token_file(tmp_path, NOW)
    assert (
        data.collect_context(["BTC"], token_path=path, timeout=2)[1]["state"]
        == "auth_refresh_required"
    )
    assert (
        data.collect_context(["DOGE"], token_path=path, timeout=2)[1]["state"]
        == "no_supported_assets"
    )


def test_cached_context_is_revalidated_at_consumption_time():
    row = data.normalize_quote({"/BTCU26": quote()}, data.INSTRUMENTS[0], NOW)
    snapshot = {
        "provider": "crypto_market_context",
        "timestamp_utc": data.iso(NOW),
        "sources": {"schwab_data_role": "context_only", "schwab_instruments": [row]},
        "derived": {
            "symbol_features": {"BTC-USD": {"crypto_schwab_future_return_norm": 999}}
        },
    }
    assert (
        data.current_features(snapshot, "BTC-USD", NOW)[
            "crypto_schwab_future_return_norm"
        ]
        < 0.5
    )
    assert data.current_features(snapshot, "BTC-USD", NOW + 121) == {}
    assert data.current_features(snapshot, "BTC-USD", NOW - 1) == {}
    assert data.current_features(snapshot, "ETH-USD", NOW) == {}
    assert data.current_features(snapshot, "SPY", NOW) == {}
    row["quote_timestamp_utc"] = data.iso(NOW + 500)
    assert data.current_features(snapshot, "BTC-USD", NOW) == {}


def test_partial_expiry_recomputes_only_still_fresh_instruments():
    future = data.normalize_quote({"/BTCU26": quote()}, data.INSTRUMENTS[0], NOW)
    fund = data.normalize_quote(
        {"IBIT": quote("IBIT", kind="etp")}, data.INSTRUMENTS[2], NOW
    )
    future["quote_timestamp_utc"] = data.iso(NOW - 119)
    snapshot = {
        "provider": "crypto_market_context",
        "timestamp_utc": data.iso(NOW),
        "sources": {
            "schwab_data_role": "context_only",
            "schwab_instruments": [future, fund],
        },
    }
    features = data.current_features(snapshot, "BTC-USD", NOW + 2)
    assert features["crypto_schwab_etp_available_norm"] == 1.0
    assert "crypto_schwab_future_available_norm" not in features
