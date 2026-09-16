import io
import json
import math
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from scripts.ops import crypto_research as research
from scripts.ops import crypto_workspace as workspace

NOW = datetime(2026, 9, 10, 16, 0, tzinfo=timezone.utc)
REQUEST = {"profile": "day", "capital": 1000, "fee_bps": 60, "spread_bps": 10}


def private_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    path.chmod(0o600)


def saved_account(tmp_path, stamp=NOW):
    private_json(
        tmp_path / "connection.json",
        {
            "credentials": {
                "api_key": "DO_NOT_EXPOSE_ID",
                "private_key": "DO_NOT_EXPOSE_SECRET",
            },
            "verified_at": stamp.isoformat(),
            "snapshot": {
                "complete": True,
                "accounts": [
                    {
                        "account_id": "DO_NOT_EXPOSE_ACCOUNT",
                        "currency": "BTC",
                        "available": "0.0006",
                        "hold": "0.0001",
                    },
                    {
                        "account_id": "ANOTHER_ID",
                        "currency": "BTC",
                        "available": "0.0002",
                        "hold": "0",
                    },
                    {"currency": "USD", "available": "0", "hold": "0"},
                ],
                "permissions": {"can_trade": True},
            },
        },
    )
    private_json(
        tmp_path / "status.json",
        {"overall_status": "ready", "verified_at": stamp.isoformat()},
    )


def test_portfolio_whitelist_and_decimal_aggregation(tmp_path):
    saved_account(tmp_path)
    result = workspace.portfolio(tmp_path, NOW)
    assert result["fresh"] is True
    assert result["balances"] == [
        {"currency": "BTC", "available": "0.0008", "hold": "0.0001", "total": "0.0009"}
    ]
    assert "DO_NOT_EXPOSE" not in json.dumps(result)


@pytest.mark.parametrize(
    "age,source_age,quote_age,usable",
    [
        (0, 0, 1, True),
        (301, 0, 1, False),
        (0, 301, 1, False),
        (-1, 0, 1, False),
        (0, 0, 121, False),
        (0, 0, -1, False),
    ],
)
def test_schwab_ui_preserves_producer_and_quote_freshness(
    tmp_path, age, source_age, quote_age, usable
):
    stamp = lambda seconds: (NOW - timedelta(seconds=seconds)).isoformat()
    private_json(
        tmp_path / "governance/health/crypto_market_context_sync_latest.json",
        {
            "timestamp_utc": stamp(age),
            "sources": {
                "schwab_crypto": {
                    "schema_version": 1,
                    "data_role": "context_only",
                    "state": "ready",
                    "timestamp_utc": stamp(source_age),
                    "secret": "DO_NOT_EXPOSE",
                    "live_execution_allowed": True,
                    "spot_feed_verified": True,
                    "instruments": [
                        {
                            "asset": "BTC",
                            "requested_symbol": "/BTC",
                            "instrument_type": "future",
                            "source_symbol": "/BTCU26",
                            "mark": 77000,
                            "spread_bps": 2,
                            "quote_timestamp_utc": stamp(quote_age),
                            "bid_timestamp_utc": stamp(quote_age),
                            "ask_timestamp_utc": stamp(quote_age),
                            "realtime": True,
                            "usable": True,
                            "quality": "fresh",
                            "secret": "DO_NOT_EXPOSE",
                        }
                    ],
                }
            },
        },
    )
    result = workspace.build_snapshot(tmp_path, tmp_path / "private", NOW)
    context = result["schwab_context"]
    assert context["instruments"][0]["usable"] is usable
    assert context["usable_instruments"] == int(usable)
    assert "DO_NOT_EXPOSE" not in json.dumps(result)
    assert context["live_execution_allowed"] is False
    assert context["spot_feed_verified"] is False
    assert result["capabilities"]["live_execution"] is False
    assert result["capabilities"]["forward_paper_execution"] is False


def test_schwab_ui_missing_or_malformed_sources_do_not_gain_authority(tmp_path):
    path = tmp_path / "governance/health/crypto_market_context_sync_latest.json"
    for source in (None, [], "invalid", {"instruments": [{"asset": ["BTC"]}]}):
        private_json(
            path,
            {"timestamp_utc": NOW.isoformat(), "sources": {"schwab_crypto": source}},
        )
        result = workspace.schwab_context(tmp_path, NOW)
        assert result["instruments"] == []
        assert result["usable_instruments"] == 0
        assert result["live_execution_allowed"] is False
    assert "credentials" not in result
    assert result["live_execution_allowed"] is False
    assert result["transfers_allowed"] is False


def test_stale_preserves_configured_cached_balance(tmp_path):
    saved_account(tmp_path, NOW - timedelta(hours=1))
    result = workspace.portfolio(tmp_path, NOW)
    assert result["configured"] and not result["fresh"]
    assert result["status"] == "stale" and result["balances"]


@pytest.mark.parametrize("offset", [1, 3600])
def test_future_account_timestamp_rejected(tmp_path, offset):
    saved_account(tmp_path, NOW + timedelta(seconds=offset))
    assert workspace.portfolio(tmp_path, NOW)["balances"] == []


def test_receipt_generation_mismatch_is_not_fresh(tmp_path):
    saved_account(tmp_path)
    private_json(
        tmp_path / "status.json",
        {"overall_status": "ready", "verified_at": "another-generation"},
    )
    result = workspace.portfolio(tmp_path, NOW)
    assert result["status"] == "refresh_unverified" and result["fresh"] is False


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-1", "1e40", None])
def test_malformed_balance_fails_closed(tmp_path, bad):
    saved_account(tmp_path)
    path = tmp_path / "connection.json"
    payload = json.loads(path.read_text())
    payload["snapshot"]["accounts"][0]["available"] = bad
    private_json(path, payload)
    assert workspace.portfolio(tmp_path, NOW)["balances"] == []


def test_world_readable_and_symlink_credentials_rejected(tmp_path):
    saved_account(tmp_path)
    path = tmp_path / "connection.json"
    path.chmod(0o644)
    assert workspace.portfolio(tmp_path, NOW)["balances"] == []
    path.rename(tmp_path / "private.json")
    path.symlink_to(tmp_path / "private.json")
    assert workspace.portfolio(tmp_path, NOW)["balances"] == []


def test_missing_gates_fail_closed_and_do_not_start_workers(tmp_path):
    snapshot = workspace.build_snapshot(tmp_path, tmp_path / "private", NOW)
    assert not snapshot["capabilities"]["live_execution"]
    assert not snapshot["capabilities"]["forward_paper_execution"]
    assert snapshot["paper"]["status"] == "held"
    assert (
        "execution_breaker_evidence_stale_or_missing" in snapshot["paper"]["blockers"]
    )
    assert snapshot["training"]["passed"] is None
    assert list(tmp_path.iterdir()) == []


def test_fresh_launcher_does_not_admit_research_profiles(tmp_path):
    health = tmp_path / "governance/health"
    private_json(
        health / "execution_runtime_breaker_latest.json",
        {"timestamp_utc": NOW.isoformat(), "active": False},
    )
    private_json(
        health / "all_sleeves_launcher_latest.json",
        {"timestamp_utc": NOW.isoformat(), "paper_execution_ready": True},
    )
    result = workspace.build_snapshot(tmp_path, tmp_path / "private", NOW)
    assert result["paper"]["blockers"] == [
        "btc_day_and_swing_forward_cohorts_not_admitted"
    ]
    assert result["capabilities"]["forward_paper_execution"] is False


@pytest.mark.parametrize(
    "timestamp", [None, "invalid", "2026-09-10T16:00:01Z", "2026-09-10T16:00:00"]
)
def test_evidence_age_rejects_unknown_future_and_naive_time(timestamp):
    assert workspace.age_seconds(timestamp, NOW) is None


@pytest.mark.parametrize(
    "change",
    [
        {"capital": True},
        {"capital": 0},
        {"capital": "1000"},
        {"capital": math.inf},
        {"fee_bps": -1},
        {"spread_bps": 0},
        {"profile": "live"},
        {"symbol": "ETH-USD"},
    ],
)
def test_replay_request_bounded(change):
    with pytest.raises(ValueError):
        research.validate_request({**REQUEST, **change})


def candles(n=150, step=300):
    prices = [100 + i * 0.25 + 4 * math.sin(i / 8) for i in range(n)]
    return [[i * step, p - 0.8, p + 0.05, p - 0.1, p, 12] for i, p in enumerate(prices)]


def test_closed_candle_validation_filters_extra_buckets():
    rows = candles(121)
    clean = research.validate_candles(rows, start=0, end=120 * 300, step=300)
    assert len(clean) == 120


@pytest.mark.parametrize(
    "mode", ["gap", "duplicate", "nonfinite", "bad_ohlc", "unaligned", "missing_tail"]
)
def test_candle_quality_fails_closed(mode):
    rows = candles(120)
    if mode == "gap":
        rows.pop(50)
    if mode == "duplicate":
        rows.insert(50, rows[50])
    if mode == "nonfinite":
        rows[50][4] = math.nan
    if mode == "bad_ohlc":
        rows[50][4] = 0
    if mode == "unaligned":
        rows[50][0] += 1
    if mode == "missing_tail":
        rows.pop()
    with pytest.raises(ValueError):
        research.validate_candles(rows, start=0, end=120 * 300, step=300)


@pytest.mark.parametrize("profile", ["day", "swing"])
def test_replay_accounting_costs_and_authority(profile, monkeypatch):
    for key in list(research.os.environ):
        if key.startswith("EXEC_SIM_"):
            monkeypatch.delenv(key)
    rows = candles(step=research.PROFILES[profile]["granularity"])
    result = research.replay(rows, {**REQUEST, "profile": profile})
    cash, btc = 1000, 0
    for fill in result["fills"]:
        assert fill["fee"] == pytest.approx(fill["quantity"] * fill["price"] * 0.006)
        if fill["action"] == "BUY":
            cash -= fill["quantity"] * fill["price"] + fill["fee"]
            btc += fill["quantity"]
        else:
            cash += fill["quantity"] * fill["price"] - fill["fee"]
            btc -= fill["quantity"]
        assert cash >= -1e-8 and btc >= -1e-8
    assert btc == pytest.approx(0)
    assert result["ending"] == pytest.approx(cash)
    assert result["net_pnl"] == pytest.approx(cash - 1000)
    assert result["fees"] == pytest.approx(sum(f["fee"] for f in result["fills"]))
    assert result["curve"][-1]["equity"] == pytest.approx(cash)
    assert result["round_trips"] > 0
    assert (
        result["model_trained"] is False and result["forward_paper_authority"] is False
    )
    assert result["start"] == rows[50][0]


def test_future_bars_cannot_change_past_fills():
    rows = candles(150)
    first = research.replay(rows, REQUEST)
    changed = [list(row) for row in rows]
    for row in changed[110:]:
        for i in range(1, 5):
            row[i] *= 2
    second = research.replay(changed, REQUEST)
    cutoff = rows[110][0]
    assert [f for f in first["fills"] if f["time"] < cutoff] == [
        f for f in second["fills"] if f["time"] < cutoff
    ]


def test_no_trades_not_reported_as_zero_win_rate():
    rows = [[i * 300, 99, 101, 100, 100, 12] for i in range(120)]
    result = research.replay(rows, REQUEST)
    assert result["round_trips"] == 0 and result["win_rate_pct"] is None
    assert result["benchmark_return_pct"] < 0


def fake_handler(host="127.0.0.1:8799", client="127.0.0.1", origin=None, content=None):
    headers = {"Host": host}
    if origin is not None:
        headers["Origin"] = origin
    if content is not None:
        headers["Content-Type"] = content
    return SimpleNamespace(
        headers=headers,
        client_address=(client, 123),
        server=SimpleNamespace(server_address=("127.0.0.1", 8799)),
    )


@pytest.mark.parametrize(
    "host,client",
    [
        ("evil.example:8799", "127.0.0.1"),
        ("127.0.0.1:8799", "192.168.1.2"),
        ("127.0.0.1:1", "127.0.0.1"),
    ],
)
def test_private_routes_reject_remote_and_dns_rebinding(host, client):
    assert not workspace.local_request(fake_handler(host, client))


def test_replay_requires_same_origin_json():
    assert workspace.local_request(fake_handler())
    assert not workspace.local_request(fake_handler(), mutation=True)
    assert not workspace.local_request(
        fake_handler(origin="https://evil.example", content="application/json"),
        mutation=True,
    )
    assert workspace.local_request(
        fake_handler(origin="http://127.0.0.1:8799", content="application/json"),
        mutation=True,
    )


def test_replay_singleflight_and_deadline(monkeypatch, tmp_path):
    workspace.REPLAY_LOCK.acquire()
    try:
        assert workspace.run_replay(tmp_path, REQUEST)[0] == 409
    finally:
        workspace.REPLAY_LOCK.release()
    workspace.LAST_ATTEMPT.clear()
    workspace.REPLAYS.clear()

    def timeout(*args, **kwargs):
        assert kwargs["timeout"] == 25
        assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
        assert "COINBASE_API_SECRET" not in kwargs["env"]
        raise workspace.subprocess.TimeoutExpired(args[0], 25)

    monkeypatch.setattr(workspace.subprocess, "run", timeout)
    assert workspace.run_replay(tmp_path, REQUEST)[0] == 504
    assert not workspace.REPLAY_LOCK.locked()
    assert workspace.run_replay(tmp_path, REQUEST)[0] == 429


def test_http_routes_do_not_expose_submit_or_transfer(tmp_path):
    from scripts.ops import live_feed_phone_server as server

    assert (
        "handle_crypto_request" in server._PhoneMirrorHandler.do_POST.__code__.co_names
    )
    handler = fake_handler(origin="http://127.0.0.1:8799", content="application/json")
    handler.path = "/api/crypto/submit"
    handler.wfile = io.BytesIO()
    handler._require_auth = lambda: True
    statuses, headers = [], {}
    handler.send_response = statuses.append
    handler.send_header = lambda k, v: headers.update({k: v})
    handler.end_headers = lambda: None
    assert workspace.handle_request(handler, tmp_path, post=True)
    assert statuses == [405]
    assert headers["Cache-Control"] == "no-store"
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]


@pytest.mark.parametrize("status,oversize", [(302, False), (500, False), (200, True)])
def test_public_transport_rejects_redirect_and_oversized_response(
    monkeypatch, status, oversize
):
    seen = []

    class Client:
        def __init__(self, host, timeout):
            assert host == "api.exchange.coinbase.com" and timeout == 6

        def request(self, method, path, headers):
            assert method == "GET" and path.startswith("/products/BTC-USD/candles?")
            assert "Authorization" not in headers

        def getresponse(self):
            return SimpleNamespace(
                status=status, read=lambda limit: b"x" * limit if oversize else b"[]"
            )

        def close(self):
            seen.append("closed")

    monkeypatch.setattr(research.http.client, "HTTPSConnection", Client)
    with pytest.raises(ValueError):
        research.fetch_and_replay(REQUEST)
    assert seen == ["closed"]


def test_cached_replay_does_not_refetch(monkeypatch, tmp_path):
    workspace.REPLAYS.clear()
    workspace.LAST_ATTEMPT.clear()
    payload = {
        "kind": "historical_research_only",
        "live_execution_allowed": False,
        "parameters": REQUEST,
    }
    calls = []

    def run(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload))

    monkeypatch.setattr(workspace.subprocess, "run", run)
    assert workspace.run_replay(tmp_path, REQUEST)[0] == 200
    assert workspace.run_replay(tmp_path, REQUEST)[0] == 200
    assert len(calls) == 1
    assert workspace.run_replay(tmp_path, {**REQUEST, "capital": 2000})[0] == 429
    workspace.REPLAYS.clear()
    workspace.LAST_ATTEMPT.clear()


def test_crypto_data_requires_existing_feed_auth(tmp_path):
    handler = fake_handler()
    handler.path = "/api/crypto/status"
    called = []
    handler._require_auth = lambda: called.append("auth") or False
    assert workspace.handle_request(handler, tmp_path)
    assert called == ["auth"]


def test_crypto_logs_never_include_tokens(capsys):
    from scripts.ops import live_feed_phone_server as server

    handler = SimpleNamespace(path="/api/crypto/status?token=PRIVATE_TOKEN")
    server._PhoneMirrorHandler.log_message(handler, "%s", "PRIVATE_TOKEN")
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("present", [False, True])
def test_download_is_authenticated_research_attachment(tmp_path, present):
    handler = fake_handler()
    handler.path = "/api/crypto/export/day"
    handler.wfile = io.BytesIO()
    calls, headers = [], {}
    handler._require_auth = lambda: calls.append("authenticated") or True
    handler.send_response = calls.append
    handler.send_header = lambda key, value: headers.update({key: value})
    handler.end_headers = lambda: None
    workspace.REPLAYS.clear()
    if present:
        workspace.REPLAYS["day"] = {
            "kind": "historical_research_only",
            "live_execution_allowed": False,
        }
    try:
        assert workspace.handle_request(handler, tmp_path)
        assert calls == ["authenticated", 200 if present else 404]
        assert headers["Cache-Control"] == "no-store"
        if present:
            assert (
                headers["Content-Disposition"]
                == 'attachment; filename="btc-day-historical-research.json"'
            )
            assert (
                json.loads(handler.wfile.getvalue())["live_execution_allowed"] is False
            )
    finally:
        workspace.REPLAYS.clear()
