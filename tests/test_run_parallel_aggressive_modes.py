from scripts import run_parallel_aggressive_modes as agg


class _Args:
    broker = "schwab"
    split_ingress_loops = True
    intraday_threshold_shift = -0.08
    swing_threshold_shift = -0.04
    intraday_interval_seconds = 8
    swing_interval_seconds = 75
    intraday_defensive_interval_seconds = 20
    swing_defensive_interval_seconds = 150
    intraday_symbols_core = "SPY,QQQ"
    intraday_symbols_volatile = "TSLA,COIN"
    intraday_symbols_defensive = "TLT,GLD"
    intraday_context_symbols = "$VIX.X,UUP"
    swing_symbols_core = "SPY,QQQ"
    swing_symbols_volatile = "TSLA,COIN"
    swing_symbols_defensive = "TLT,GLD"
    swing_context_symbols = "$VIX.X,UUP"


def test_build_worker_specs_splits_schwab_aggressive_profiles() -> None:
    workers = agg._build_worker_specs(_Args())

    names = [worker.name for worker in workers]
    assert names == [
        "intraday_aggressive_core_volatile",
        "intraday_aggressive_defensive",
        "swing_aggressive_core_volatile",
        "swing_aggressive_defensive",
    ]
    assert workers[0].ingress_instance == "core_volatile"
    assert workers[1].symbols_defensive == "TLT,GLD"
    assert workers[1].symbols_core == ""
    assert workers[3].interval_seconds == 150


def test_aggregate_ingress_payload_sums_split_workers() -> None:
    payload = agg._aggregate_ingress_payload(
        "intraday_aggressive",
        "schwab",
        [
            {
                "timestamp_utc": "2026-03-31T21:10:00+00:00",
                "iter": 10,
                "run_id": "run-1",
                "iter_id": "run-1:10",
                "loop_state": "running",
                "iter_counts": {"api_ok": 5, "api_error": 1},
                "total_counts": {"api_ok": 50, "api_error": 3},
                "iter_total_requests": 6,
                "iter_error_count": 1,
                "symbols_total": 4,
                "context_total": 2,
                "log_schema_version": 2,
                "ingress_instance": "core_volatile",
            },
            {
                "timestamp_utc": "2026-03-31T21:10:01+00:00",
                "iter": 11,
                "run_id": "run-1",
                "iter_id": "run-1:11",
                "loop_state": "running",
                "iter_counts": {"api_ok": 3},
                "total_counts": {"api_ok": 22},
                "iter_total_requests": 3,
                "iter_error_count": 0,
                "symbols_total": 2,
                "context_total": 2,
                "log_schema_version": 2,
                "ingress_instance": "defensive",
            },
        ],
        expected_instances=["core_volatile", "defensive"],
    )

    assert payload["profile"] == "intraday_aggressive"
    assert payload["loop_state"] == "split_running"
    assert payload["iter_total_requests"] == 9
    assert payload["iter_error_count"] == 1
    assert payload["iter_counts"]["api_ok"] == 8
    assert payload["total_counts"]["api_ok"] == 72
    assert payload["symbols_total"] == 6
    assert payload["ingress_instances"] == ["core_volatile", "defensive"]
    assert payload["expected_ingress_instances"] == ["core_volatile", "defensive"]
    assert payload["missing_ingress_instances"] == []


def test_aggregate_ingress_payload_marks_missing_split_worker() -> None:
    payload = agg._aggregate_ingress_payload(
        "intraday_aggressive",
        "schwab",
        [
            {
                "timestamp_utc": "2026-03-31T21:10:01+00:00",
                "iter": 11,
                "run_id": "run-1",
                "iter_id": "run-1:11",
                "loop_state": "running",
                "iter_counts": {"api_ok": 3},
                "total_counts": {"api_ok": 22},
                "iter_total_requests": 3,
                "iter_error_count": 0,
                "symbols_total": 2,
                "context_total": 2,
                "log_schema_version": 2,
                "ingress_instance": "defensive",
            },
        ],
        expected_instances=["core_volatile", "defensive"],
    )

    assert payload["loop_state"] == "split_partial"
    assert payload["missing_ingress_instances"] == ["core_volatile"]
    assert payload["split_profiles"][0]["ingress_instance"] == "core_volatile"
    assert payload["split_profiles"][0]["loop_state"] == "missing"
