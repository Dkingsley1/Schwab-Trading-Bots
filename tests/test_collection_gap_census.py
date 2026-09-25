from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import gzip
import importlib.util
import json
from pathlib import Path
import sys

import pytest

# Loading the sibling source also lets this suite verify a staged handoff against
# the repository's existing core owners without writing into the repository.
_path = Path(__file__).resolve().parents[1] / "scripts/ops/collection_gap_census.py"
_spec = importlib.util.spec_from_file_location(
    "collection_gap_census_under_test", _path
)
src = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(src)

NOW = datetime(2026, 9, 25, 20, tzinfo=timezone.utc)


def scope(
    provider="coinbase_exchange",
    symbol="BTC-USD",
    timeframe="5m",
    start="2026-09-24T14:00:00+00:00",
    end="2026-09-24T14:15:00+00:00",
):
    return src.candle_scope(
        dict(
            provider=provider,
            symbol=symbol,
            timeframe=timeframe,
            start_utc=start,
            end_utc=end,
        ),
        NOW,
    )


def bar(start, end, close=11):
    return dict(
        start_utc=start, end_utc=end, open=10, high=12, low=9, close=close, volume=3
    )


def coinbase_rows(s):
    return [
        [int(src.timestamp(a).timestamp()), 9, 12, 10, 11, 3]
        for a, _ in src.intervals(s, src.Budget())
    ]


def imported(s=None, payload=None):
    s = s or scope()
    job = src.candle_request(s)
    return src.import_response(
        job,
        coinbase_rows(s) if payload is None else payload,
        observed_at_utc=NOW.isoformat(),
        source_endpoint=job["endpoint"],
        now=NOW,
    )


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def root(tmp_path):
    from core.research_data_platform import load_policy

    write(tmp_path / "config/research_data_platform_v1.json", load_policy())
    write(tmp_path / "config/supervised_broker_test_v1.json", {"symbol": "O"})
    write(tmp_path / "config/supervised_schd_broker_test_v1.json", {"symbol": "SCHD"})
    path = tmp_path / "scripts/ops/bitcoin_price_watch.py"
    path.parent.mkdir(parents=True)
    path.write_text("")
    return tmp_path


def manifest(s=None, sources=None):
    return {"candles": [s or scope()], "sources": sources or []}


def test_coinbase_gap_windows_do_not_download_observed_bars(root):
    s = scope()
    rows = coinbase_rows(s)
    receipt = imported(s, [rows[1]])
    path = write(root / "source.json", receipt)
    result = src.census(
        root, manifest(s, [{"kind": "receipt", "path": str(path)}]), now=NOW
    )
    assert result["candles"][0]["distinct_bars"] == 1
    assert result["candles"][0]["missing_bars"] == 2
    assert len(result["requests"]) == 2
    assert result["requests"][0]["end_utc"] == "2026-09-24T14:05:00+00:00"
    assert result["requests"][1]["start_utc"] == "2026-09-24T14:10:00+00:00"
    assert result["dispatch"]["download_authorized"] is False
    assert (
        result["dispatch"]["catalog_authorizations"]["broker_market_observations_v1"][
            "authorized"
        ]
        is False
    )


def test_import_replan_is_idempotent_and_no_historical_live_evidence(root):
    receipt = imported()
    path = write(root / "source.json", receipt)
    m = manifest(sources=[{"kind": "receipt", "path": str(path)}] * 2)
    result = src.census(root, m, now=NOW)
    assert not result["requests"]
    assert result["candles"][0]["distinct_bars"] == 3
    assert result["candles"][0]["counts"]["duplicate_physical_bar"] == 3
    assert all(r["known_at_utc"] == NOW.isoformat() for r in receipt["rows"])
    from core.research_data_platform import select_bitemporal_rows

    rows = [dict(r, effective_at_utc=r["start_utc"]) for r in receipt["rows"]]
    assert (
        select_bitemporal_rows(
            rows,
            as_of_utc="2026-09-24T20:00:00Z",
            valid_at_utc=NOW.isoformat(),
            natural_key_columns=["start_utc"],
            contract={},
        )
        == []
    )


def test_conflicting_duplicate_does_not_earn_coverage(root):
    rows = coinbase_rows(scope())
    other = deepcopy(rows[0])
    other[4] = 10
    receipt = imported(payload=rows + [rows[0], other])
    assert receipt["coverage"]["present"] == 2
    assert receipt["coverage"]["missing"] == 1
    assert receipt["status"] == "incomplete"
    assert receipt["coverage"]["counts"]["identical_duplicates"] == 1


def test_conflicts_across_physical_files_are_missing(root):
    first = imported()
    rows = coinbase_rows(scope())
    rows[0][4] = 10
    second = imported(payload=rows)
    sources = [
        {"kind": "receipt", "path": str(write(root / f"{i}.json", r))}
        for i, r in enumerate((first, second))
    ]
    result = src.census(root, manifest(sources=sources), now=NOW)
    assert result["candles"][0]["conflicting_bars"] == 1
    assert result["candles"][0]["distinct_bars"] == 2


def test_crypto_chunks_match_exchange_transport_limit():
    s = scope(timeframe="1m", start="2026-09-24T00:00:00Z", end="2026-09-25T00:00:00Z")
    expected = src.intervals(s, src.Budget())
    jobs = src.missing_requests(s, expected)
    assert len(jobs) == 5
    assert all(len(src.intervals(j, src.Budget())) <= 299 for j in jobs)
    assert sum(len(src.intervals(j, src.Budget())) for j in jobs) == 1440


def test_calendar_holiday_early_close_and_dst():
    s = scope("schwab", "O", "1m", "2025-11-27T00:00:00Z", "2025-11-29T00:00:00Z")
    expected = src.intervals(s, src.Budget())
    assert len(expected) == 210
    assert expected[0][0] == "2025-11-28T14:30:00+00:00"
    assert expected[-1][1] == "2025-11-28T18:00:00+00:00"
    summer = scope(
        "schwab", "SCHD", "1d", "2026-07-02T00:00:00Z", "2026-07-06T00:00:00Z"
    )
    assert src.intervals(summer, src.Budget()) == [
        ("2026-07-02T13:30:00+00:00", "2026-07-02T20:00:00+00:00")
    ]


def test_schwab_daily_timestamp_mapping_and_symbol_check():
    s = scope("schwab", "O", "1d", "2026-09-24T00:00:00Z", "2026-09-25T00:00:00Z")
    job = src.candle_request(s)
    assert job["request_parameters"]["start_datetime"] == "2026-09-24T04:00:00+00:00"
    # This broad scope ends at UTC midnight, still September 24 in New York.
    assert (
        job["request_parameters"]["end_datetime"] == "2026-09-25T03:59:59.999000+00:00"
    )
    ms = int(datetime(2026, 9, 24, 4, tzinfo=timezone.utc).timestamp() * 1000)
    payload = {
        "symbol": "O",
        "candles": [dict(datetime=ms, open=10, high=12, low=9, close=11, volume=1)],
    }
    result = src.import_response(
        job,
        payload,
        observed_at_utc=NOW.isoformat(),
        source_endpoint=job["endpoint"],
        now=NOW,
    )
    assert result["rows"][0]["start_utc"] == "2026-09-24T13:30:00+00:00"
    payload["symbol"] = "SCHD"
    with pytest.raises(ValueError, match="symbol_mismatch"):
        src.import_response(
            job,
            payload,
            observed_at_utc=NOW.isoformat(),
            source_endpoint=job["endpoint"],
            now=NOW,
        )


def test_native_capture_is_read_without_chart_changes(root):
    s = scope("schwab", "SCHD", "5m", "2026-09-24T13:30:00Z", "2026-09-24T13:45:00Z")
    native = {
        "source": {
            "provider": "schwab",
            "symbol": "SCHD",
            "fetch_started_at_utc": NOW.isoformat(),
        },
        "candles": {"5m": [bar(a, b) for a, b in src.intervals(s, src.Budget())]},
    }
    path = write(root / "capture.json", native)
    result = src.census(
        root,
        manifest(s, [{"kind": "native_schwab_capture", "path": str(path)}]),
        now=NOW,
    )
    assert result["candles"][0]["missing_bars"] == 0
    assert result["corporate_actions"]["status"] == "unavailable_not_certified"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r[0].__setitem__(1, float("nan")),
        lambda r: r[0].__setitem__(5, -1),
        lambda r: r[0].__setitem__(3, True),
        lambda r: r[0].__setitem__(0, r[0][0] + 1),
        lambda r: r.extend(r * 101),
    ],
)
def test_invalid_provider_bars_rejected(mutate):
    rows = coinbase_rows(scope())
    mutate(rows)
    with pytest.raises(ValueError):
        imported(payload=rows)


def test_empty_response_remains_incomplete():
    result = imported(payload=[])
    assert result["status"] == "incomplete"
    assert result["coverage"]["missing"] == 3


def test_outside_bars_do_not_fill_gap():
    rows = coinbase_rows(scope())
    rows[0][0] -= 300
    result = imported(payload=rows)
    assert result["coverage"]["missing"] == 1
    assert result["coverage"]["counts"]["outside_requested_session_or_interval"] == 1


def test_request_tampering_and_source_mismatch():
    job = src.candle_request(scope())
    with pytest.raises(ValueError, match="source_mismatch"):
        src.import_response(
            job, [], observed_at_utc=NOW.isoformat(), source_endpoint="other", now=NOW
        )
    job["request_parameters"]["granularity"] = 60
    with pytest.raises(ValueError, match="binding"):
        src.import_response(
            job,
            [],
            observed_at_utc=NOW.isoformat(),
            source_endpoint=job["endpoint"],
            now=NOW,
        )


def test_receipt_hash_and_original_observation_are_preserved():
    receipt = imported()
    receipt["rows"][0]["known_at_utc"] = "2026-01-01T00:00:00+00:00"
    with pytest.raises(ValueError, match="digest"):
        src.validate_receipt(receipt, NOW, src.Budget())
    receipt["receipt_sha256"] = src.digest(
        {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    )
    with pytest.raises(ValueError, match="knowledge_time"):
        src.validate_receipt(receipt, NOW, src.Budget())


def fred_job():
    return src.fred_request(
        dict(
            series_id="CPIAUCSL",
            observation_start="2026-01-01",
            observation_end="2026-02-01",
            vintage_date="2026-03-15",
        ),
        NOW,
    )


def fred_payload():
    return {
        "realtime_start": "2026-03-15",
        "realtime_end": "2026-03-15",
        "offset": 0,
        "count": 2,
        "observations": [
            {
                "date": f"2026-0{i}-01",
                "value": str(100 + i),
                "realtime_start": "2026-03-15",
                "realtime_end": "2026-03-15",
            }
            for i in (1, 2)
        ],
    }


def fred_import(payload):
    job = fred_job()
    return src.import_response(
        job,
        payload,
        observed_at_utc=NOW.isoformat(),
        source_endpoint=job["endpoint"],
        now=NOW,
    )


def test_fred_vintage_does_not_become_historical_observation(root):
    result = fred_import(fred_payload())
    assert result["status"] == "response_reconciled"
    assert all(
        r["known_at_utc"] == NOW.isoformat() and r["publication_at_utc"] is None
        for r in result["rows"]
    )
    assert result["rows"][0]["vintage_date"] == "2026-03-15"
    path = write(root / "fred.json", result)
    m = manifest(sources=[{"kind": "receipt", "path": str(path)}])
    m["fred_vintages"] = [
        dict(
            series_id="CPIAUCSL",
            observation_start="2026-01-01",
            observation_end="2026-02-01",
            vintage_date="2026-03-15",
        )
    ]
    report = src.census(root, m, now=NOW)
    assert not any(j["product"] == "fred_vintage" for j in report["requests"])


def test_fred_missing_and_paginated_responses_stay_incomplete():
    payload = fred_payload()
    payload["count"] = 3
    payload["observations"][0]["value"] = "."
    result = fred_import(payload)
    assert result["status"] == "incomplete"
    assert set(result["issues"]) == {"fred_missing_value", "fred_pagination_incomplete"}


def test_fred_latest_revised_value_is_not_a_requested_vintage():
    payload = fred_payload()
    payload["realtime_start"] = "2026-09-25"
    with pytest.raises(ValueError, match="vintage_mismatch"):
        fred_import(payload)


def test_fred_conflicting_revisions_not_silently_overwritten():
    payload = fred_payload()
    payload["observations"].append(dict(payload["observations"][0], value="999"))
    payload["count"] = 3
    with pytest.raises(ValueError, match="conflicting_duplicate"):
        fred_import(payload)


def test_deadline_and_global_budgets_are_not_success(monkeypatch):
    budget = src.Budget()
    monkeypatch.setattr(src.time, "monotonic", lambda: budget.deadline + 1)
    with pytest.raises(ValueError, match="deadline"):
        src.intervals(scope(), budget)


def test_unsupported_source_is_explicit_not_empty_history(root):
    path = write(root / "archive.parquet", {"rows": []})
    result = src.census(
        root, manifest(sources=[{"kind": "parquet", "path": str(path)}]), now=NOW
    )
    assert result["inventory_status"] == "incomplete"
    assert result["lifetime_coverage_certified"] is False


def test_protected_and_linked_paths_are_rejected_without_access(tmp_path):
    with pytest.raises(ValueError, match="unsafe"):
        src.read_json("/Volumes/VIDEO/never-inspect", src.Budget())
    link = tmp_path / "link"
    link.symlink_to("/Volumes/VIDEO")
    with pytest.raises(ValueError, match="unsafe"):
        src.read_json(link / "never-inspect", src.Budget())


def test_full_hash_and_compressed_expansion_cap(tmp_path, monkeypatch):
    path = tmp_path / "payload.json.gz"
    path.write_bytes(gzip.compress(json.dumps({"a": "x" * 3000}).encode()))
    monkeypatch.setattr(src, "MAX_BYTES", 1024)
    with pytest.raises(ValueError, match="expanded"):
        src.read_json(path, src.Budget())


def test_future_and_stale_holdings_do_not_assert_current_truth(root):
    snapshot = {
        "broker": "schwab",
        "timestamp_utc": (NOW - timedelta(days=2)).isoformat(),
        "fetched": {
            "ok": True,
            "account_snapshot_partial": False,
            "payload": {
                "positions": [
                    {
                        "instrument": {"symbol": "O", "assetType": "EQUITY"},
                        "longQuantity": 5,
                    },
                    {
                        "instrument": {"symbol": "PG", "assetType": "EQUITY"},
                        "longQuantity": 1,
                    },
                ]
            },
        },
    }
    write(
        root / "governance/health/broker_truth_shared_snapshot_schwab_latest.json",
        snapshot,
    )
    result = src.reconcile_symbols(root, {"O", "BTC-USD"}, NOW, src.Budget())
    assert result["holdings_state"] == "stale_or_future"
    assert result["held_equity_symbols"] == ["O", "PG"]
    assert result["held_but_not_selected"] == ["PG"]
    assert result["automatic_universe_expansion"] is False
    assert "account" not in json.dumps(result).lower()


def test_cli_never_fetches_and_reports_bad_input(tmp_path, capsys, monkeypatch):
    import socket

    monkeypatch.setattr(
        socket, "create_connection", lambda *a, **kw: pytest.fail("network attempted")
    )
    assert src.main([]) == 2
    assert json.loads(capsys.readouterr().out)["network_requests_performed"] is False
    with pytest.raises(SystemExit):
        src.main(["--execute"])


def test_job_count_is_bounded_and_deferred_count_visible(root):
    s = scope("schwab", "O", "1d", "2026-01-01T00:00:00Z", "2026-09-01T00:00:00Z")
    result = src.census(root, manifest(s), now=NOW)
    assert len(result["requests"]) == src.MAX_JOBS
    assert result["deferred_request_count"] > 0
    for job in result["requests"]:
        src._check_request(job, NOW)


def dividend_fixture():
    job = src.dividend_request(
        dict(
            symbol="O",
            start_utc="2026-09-01T00:00:00Z",
            end_utc="2026-09-24T00:00:00Z",
            account_reference_sha256="a" * 64,
        ),
        NOW,
    )
    payload = {
        "rows": [
            {
                "symbol": "O",
                "description": "CASH DIVIDEND",
                "netAmount": 1.23,
                "activityId": "private-transaction-id",
                "transactionDate": "2026-09-15T14:00:00Z",
            }
        ],
        "account_reference_sha256": "a" * 64,
        "source_complete": True,
        "window_start_utc": job["start_utc"],
        "window_end_utc": job["end_utc"],
    }
    return job, payload


def test_dividend_native_classifier_bound_to_account_and_window():
    job, payload = dividend_fixture()
    receipt = src.import_response(
        job,
        payload,
        observed_at_utc=NOW.isoformat(),
        source_endpoint="get_transactions",
        now=NOW,
    )
    assert receipt["status"] == "response_reconciled"
    row = receipt["rows"][0]
    assert row["signed_net_amount"] == 1.23
    assert row["issuer_ex_date"] is None
    assert row["known_at_utc"] == NOW.isoformat()
    assert "private-transaction-id" not in json.dumps(receipt)
    assert receipt["coverage"]["issuer_corporate_action_history_complete"] is False
    payload["account_reference_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="binding_mismatch"):
        src.import_response(
            job,
            payload,
            observed_at_utc=NOW.isoformat(),
            source_endpoint="get_transactions",
            now=NOW,
        )


def test_dividend_unknown_time_and_incomplete_source_are_explicit():
    job, payload = dividend_fixture()
    payload["source_complete"] = False
    payload["rows"][0]["transactionDate"] = "2026-09-15"
    receipt = src.import_response(
        job,
        payload,
        observed_at_utc=NOW.isoformat(),
        source_endpoint="get_transactions",
        now=NOW,
    )
    assert receipt["status"] == "incomplete"
    assert receipt["rows"] == []
    assert set(receipt["issues"]) == {
        "transaction_source_incomplete",
        "unresolved_dividend_transaction",
    }


def test_dividend_census_stops_replanning_only_reconciled_window(root):
    job, payload = dividend_fixture()
    receipt = src.import_response(
        job,
        payload,
        observed_at_utc=NOW.isoformat(),
        source_endpoint="get_transactions",
        now=NOW,
    )
    path = write(root / "dividend.json", receipt)
    s = scope("schwab", "O", "1d", "2026-09-24T00:00:00Z", "2026-09-25T00:00:00Z")
    m = manifest(s, [{"kind": "receipt", "path": str(path)}])
    m["dividends"] = [job]
    result = src.census(root, m, now=NOW)
    assert result["dividends"][0]["status"] == "account_postings_reconciled"
    assert not any(j["product"] == "schwab_dividends" for j in result["requests"])


def test_changed_source_during_read_cannot_earn_full_hash(tmp_path, monkeypatch):
    path = write(tmp_path / "source.json", {"value": 1})
    original = src.safe_path
    calls = []

    def checked(p, **kw):
        calls.append(p)
        if len(calls) == 2:
            path.write_text('{"value": 222}')
        return original(p, **kw)

    monkeypatch.setattr(src, "safe_path", checked)
    with pytest.raises(ValueError, match="source_changed"):
        src.read_json(path, src.Budget())


def test_census_and_import_cli_have_no_network(root, capsys, monkeypatch):
    import socket

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW

    monkeypatch.setattr(src, "datetime", Clock)
    monkeypatch.setattr(
        socket, "create_connection", lambda *a, **kw: pytest.fail("network attempted")
    )
    request_path = write(root / "request.json", src.candle_request(scope()))
    payload_path = write(root / "payload.json", coinbase_rows(scope()))
    assert (
        src.main(
            [
                "--request",
                str(request_path),
                "--response",
                str(payload_path),
                "--observed-at-utc",
                NOW.isoformat(),
                "--source-endpoint",
                "https://api.exchange.coinbase.com/products/BTC-USD/candles",
            ]
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    path = write(root / "receipt.json", receipt)
    mpath = write(
        root / "manifest.json",
        manifest(sources=[{"kind": "receipt", "path": str(path)}]),
    )
    assert src.main(["--root", str(root), "--manifest", str(mpath)]) == 0
    assert json.loads(capsys.readouterr().out)["requests"] == []
