from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import collect_free_equity_reference_context as equity_ref


def test_free_equity_reference_context_combines_yahoo_and_nasdaq(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        equity_ref,
        "_fetch_yahoo_chart",
        lambda symbol, **kwargs: (
            {"price": 510.0, "previous_close": 500.0, "volume": 3_000_000, "change": 0.02},
            {"ok": True, "price": 510.0, "volume": 3_000_000, "error": None},
        ),
    )
    monkeypatch.setattr(
        equity_ref,
        "_fetch_nasdaq_quote",
        lambda symbol, **kwargs: (
            {"price": 509.8, "bid": 509.7, "ask": 509.9, "volume": 2_700_000, "change": 0.0196},
            {"ok": True, "price": 509.8, "volume": 2_700_000, "error": None},
        ),
    )

    payload = equity_ref.build_payload(symbols=["SPY"], user_agent="test/1.0", timeout=1.0, enable_nasdaq=True)

    assert payload["ok"] is True
    assert payload["overall_status"] == "ready"
    assert payload["symbols_with_reference"] == 1
    assert payload["ok_source_count"] == 2
    assert payload["sources"]["nasdaq_quote"]["ok"] is True
    assert payload["features"]["free_equity_quote_available_norm"] == 1.0
    features = payload["derived"]["symbol_features"]["SPY"]
    assert features["free_equity_quote_available_norm"] == 1.0
    assert features["free_equity_nasdaq_volume_norm"] > 0.0
    assert features["free_equity_cross_provider_agreement_norm"] > 0.9
    assert features["free_equity_momentum_norm"] > 0.5


def test_free_equity_reference_context_degrades_without_reference_rows(monkeypatch: Any) -> None:
    monkeypatch.setattr(equity_ref, "_fetch_yahoo_chart", lambda symbol, **kwargs: ({}, {"ok": False, "error": "blocked"}))
    monkeypatch.setattr(equity_ref, "_fetch_nasdaq_quote", lambda symbol, **kwargs: ({}, {"ok": False, "error": "blocked"}))

    payload = equity_ref.build_payload(symbols=["SPY"], user_agent="test/1.0", timeout=1.0)

    assert payload["ok"] is False
    assert payload["overall_status"] == "degraded"
    assert payload["symbols_with_reference"] == 0
    assert payload["ok_source_count"] == 0


def test_free_equity_reference_context_verifies_with_partial_public_source(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        equity_ref,
        "_fetch_yahoo_chart",
        lambda symbol, **kwargs: (
            {"price": 510.0, "previous_close": 500.0, "volume": 3_000_000, "change": 0.02},
            {"ok": True, "price": 510.0, "volume": 3_000_000, "error": None},
        ),
    )
    monkeypatch.setattr(equity_ref, "_fetch_nasdaq_quote", lambda symbol, **kwargs: ({}, {"ok": False, "error": "blocked"}))

    payload = equity_ref.build_payload(symbols=["SPY"], user_agent="test/1.0", timeout=1.0, max_runtime_seconds=5.0, enable_nasdaq=True)

    assert payload["ok"] is True
    assert payload["overall_status"] == "ready"
    assert payload["symbols_with_reference"] == 1
    assert payload["ok_source_count"] == 1
    assert payload["sources"]["yahoo_chart"]["ok"] is True
    assert payload["sources"]["nasdaq_quote"]["ok"] is False
    assert payload["attempted_symbol_count"] == 1


def test_free_equity_reference_context_skips_nasdaq_by_default(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        equity_ref,
        "_fetch_yahoo_chart",
        lambda symbol, **kwargs: (
            {"price": 510.0, "previous_close": 500.0, "volume": 3_000_000, "change": 0.02},
            {"ok": True, "price": 510.0, "volume": 3_000_000, "error": None},
        ),
    )
    monkeypatch.setattr(
        equity_ref,
        "_fetch_nasdaq_quote",
        lambda symbol, **kwargs: (_ for _ in ()).throw(AssertionError("nasdaq should be opt-in")),
    )

    payload = equity_ref.build_payload(symbols=["SPY"], user_agent="test/1.0", timeout=1.0)

    assert payload["ok"] is True
    assert payload["ok_source_count"] == 1
    assert "nasdaq_quote" not in payload["sources"]
    assert payload["safety_contract"]["nasdaq_enabled"] is False


def test_free_equity_reference_context_defaults_are_soak_safe(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        equity_ref,
        "_fetch_yahoo_chart",
        lambda symbol, **kwargs: (
            {"price": 510.0, "previous_close": 500.0, "volume": 3_000_000, "change": 0.02},
            {"ok": True, "price": 510.0, "volume": 3_000_000, "error": None},
        ),
    )
    monkeypatch.setattr(equity_ref, "_fetch_nasdaq_quote", lambda symbol, **kwargs: ({}, {"ok": False, "error": "blocked"}))

    payload = equity_ref.build_payload(symbols=[f"SYM{i}" for i in range(30)])

    assert payload["requested_symbol_count"] == equity_ref.DEFAULT_MAX_SYMBOLS
    assert payload["max_runtime_seconds"] == equity_ref.DEFAULT_MAX_RUNTIME_SECONDS
    assert payload["attempted_symbol_count"] == equity_ref.DEFAULT_MAX_SYMBOLS
