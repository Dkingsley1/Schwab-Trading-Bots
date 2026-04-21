import json
from pathlib import Path
from datetime import datetime, timezone
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import scripts.collect_tastytrade_context as tasty


def test_derive_option_chain_metrics_detects_chain_density_and_near_term() -> None:
    payload = {
        "data": {
            "items": [
                {
                    "expirations": [
                        {"days-to-expiration": 0, "strikes": [100, 105, 110]},
                        {"days-to-expiration": 5, "strikes": [95, 100, 105, 110]},
                    ]
                }
            ]
        }
    }

    out = tasty._derive_option_chain_metrics(payload)

    assert out["chain_available"] == 1.0
    assert out["zero_dte_presence_norm"] == 1.0
    assert out["near_term_presence_norm"] == 1.0
    assert out["contract_density_norm"] > 0.0


def test_derive_symbol_features_uses_market_metrics_and_watchlist_presence() -> None:
    option_chain_payload = {
        "data": {
            "items": [
                {
                    "expirations": [
                        {
                            "days-to-expiration": 3,
                            "strikes": [
                                {"strike-price": 500, "call-open-interest": 1200, "put-open-interest": 1800, "call-gamma": 0.28, "put-gamma": 0.44, "call-volume": 210, "put-volume": 330},
                                {"strike-price": 510, "call-open-interest": 1900, "put-open-interest": 2200, "call-gamma": 0.40, "put-gamma": 0.52, "call-volume": 420, "put-volume": 530},
                            ],
                        },
                        {
                            "days-to-expiration": 35,
                            "strikes": [
                                {"strike-price": 500, "call-open-interest": 2600, "put-open-interest": 1300, "call-gamma": 0.22, "put-gamma": 0.14, "call-volume": 180, "put-volume": 120},
                                {"strike-price": 510, "call-open-interest": 2800, "put-open-interest": 1400, "call-gamma": 0.24, "put-gamma": 0.16, "call-volume": 220, "put-volume": 140},
                            ],
                        },
                    ]
                }
            ]
        }
    }
    market_metrics_payload = {
        "data": {
            "items": [
                {
                    "symbol": "SPY",
                    "iv-rank": 42,
                    "implied-volatility-index": 31,
                    "liquidity-rating": 4,
                    "expected-move": 8.5,
                    "beta": 1.2,
                    "underlying-price": 510.0,
                    "utilization": 74,
                    "days-to-cover": 3.5,
                }
            ]
        }
    }
    instrument_payload = {
        "data": {
            "items": [
                {
                    "symbol": "SPY",
                    "lendability": "easy to borrow",
                    "borrow-fee-rate": 4.2,
                }
            ]
        }
    }

    out = tasty._derive_symbol_features(
        symbol="SPY",
        option_chain_payload=option_chain_payload,
        market_metrics_payload=market_metrics_payload,
        instrument_payload=instrument_payload,
        watchlist_symbols={"SPY"},
    )

    assert out["tasty_iv_rank_norm"] > 0.0
    assert out["tasty_implied_volatility_index_norm"] > 0.0
    assert out["tasty_liquidity_rating_norm"] > 0.0
    assert out["tasty_expected_move_norm"] > 0.0
    assert out["tasty_beta_norm"] > 0.0
    assert out["tasty_watchlist_presence_norm"] == 1.0
    assert out["short_borrow_availability_norm"] > 0.0
    assert out["short_borrow_fee_norm"] > 0.0
    assert out["options_iv_skew_norm"] != 0.5
    assert out["options_iv_term_structure_norm"] != 0.5
    assert out["options_gamma_expiry_skew_norm"] != 0.5
    assert out["options_vol_regime_norm"] > 0.0
    assert out["options_surface_change_norm"] > 0.0
    assert out["options_strike_expiry_concentration_change_norm"] > 0.0
    assert 0.0 <= out["options_gamma_flip_distance_norm"] <= 1.0
    assert out["options_earnings_setup_norm"] > 0.0
    assert out["options_iv_crush_risk_norm"] > 0.0
    assert out["options_assignment_risk_norm"] > 0.0
    assert out["options_zero_dte_regime_norm"] > 0.0
    assert out["options_vol_of_vol_change_norm"] > 0.0
    assert out["options_spread_execution_risk_norm"] > 0.0


def test_derive_strike_wall_features_detects_pin_risk() -> None:
    option_chain_payload = {
        "data": {
            "items": [
                {
                    "expirations": [
                        {
                            "days-to-expiration": 2,
                            "strikes": [
                                {"strike-price": 500, "call-open-interest": 1200, "put-open-interest": 200, "call-gamma": 0.42, "put-gamma": 0.08},
                                {"strike-price": 510, "call-open-interest": 1800, "put-open-interest": 1400, "call-gamma": 0.55, "put-gamma": 0.48},
                                {"strike-price": 520, "call-open-interest": 700, "put-open-interest": 1600, "call-gamma": 0.18, "put-gamma": 0.50},
                            ],
                        }
                    ]
                }
            ]
        }
    }

    out = tasty._derive_strike_wall_features(option_chain_payload, last_price=511.0, expected_move=12.0)

    assert out["tasty_call_wall_proximity_norm"] > 0.0
    assert out["tasty_put_wall_proximity_norm"] > 0.0
    assert out["tasty_max_pain_proximity_norm"] > 0.0
    assert out["tasty_pin_risk_norm"] > 0.0


def test_align_symbol_features_with_schwab_zeroes_mismatched_rows() -> None:
    now = datetime(2026, 3, 20, 14, 0, tzinfo=timezone.utc)
    aligned, meta = tasty._align_symbol_features_with_schwab(
        symbol="SPY",
        features={
            "tasty_iv_rank_norm": 0.61,
            "tasty_implied_volatility_index_norm": 0.57,
            "tasty_liquidity_rating_norm": 0.83,
            "tasty_expected_move_norm": 0.29,
            "tasty_beta_norm": 0.54,
            "tasty_watchlist_presence_norm": 1.0,
            "tasty_underlying_price": 510.0,
        },
        schwab_history={"SPY": [(now, 450.0)]},
        now_utc=now,
        sandbox=False,
        max_relative_spread=0.05,
        tolerance_minutes=10,
    )

    assert aligned == tasty._zero_feature_map()
    assert meta["ok"] is False
    assert meta["reason"] == "relative_spread_exceeded"


def test_align_symbol_features_with_schwab_keeps_matched_rows() -> None:
    now = datetime(2026, 3, 20, 14, 0, tzinfo=timezone.utc)
    aligned, meta = tasty._align_symbol_features_with_schwab(
        symbol="SPY",
        features={
            "tasty_iv_rank_norm": 0.61,
            "tasty_implied_volatility_index_norm": 0.57,
            "tasty_liquidity_rating_norm": 0.83,
            "tasty_expected_move_norm": 0.29,
            "tasty_beta_norm": 0.54,
            "tasty_watchlist_presence_norm": 1.0,
            "tasty_underlying_price": 510.0,
        },
        schwab_history={"SPY": [(now, 509.5)]},
        now_utc=now,
        sandbox=False,
        max_relative_spread=0.05,
        tolerance_minutes=10,
    )

    assert aligned["tasty_iv_rank_norm"] == 0.61
    assert "tasty_underlying_price" not in aligned
    assert meta["ok"] is True


def test_align_symbol_features_with_schwab_keeps_reference_only_rows() -> None:
    now = datetime(2026, 3, 20, 14, 0, tzinfo=timezone.utc)
    aligned, meta = tasty._align_symbol_features_with_schwab(
        symbol="SPY",
        features={
            "tasty_iv_rank_norm": 0.61,
            "tasty_implied_volatility_index_norm": 0.57,
            "tasty_liquidity_rating_norm": 0.83,
            "tasty_expected_move_norm": 0.29,
            "tasty_beta_norm": 0.54,
            "tasty_watchlist_presence_norm": 1.0,
            "tasty_underlying_price": 0.0,
        },
        schwab_history={"SPY": [(now, 509.5)]},
        now_utc=now,
        sandbox=True,
        max_relative_spread=0.05,
        tolerance_minutes=25,
    )

    assert aligned["tasty_iv_rank_norm"] == 0.61
    assert meta["ok"] is True
    assert meta["reference_only"] is True
    assert meta["reason"] == "schwab_reference_only"


def test_fetch_market_metrics_marks_404_endpoints_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def _fake_safe_http_json(**kwargs: object) -> tuple[None, str]:
        calls.append(str(kwargs.get("url")))
        return None, "HTTP Error 404: Not Found"

    monkeypatch.setattr(tasty, "_safe_http_json", _fake_safe_http_json)
    capability: dict[str, object] = {}

    payload, err = tasty._fetch_market_metrics(
        tasty.SANDBOX_BASE_URL,
        symbol="SPY",
        user_agent="schwab-trading-bot/1.0",
        session_token="token",
        timeout=8.0,
        capability_state=capability,
    )

    assert payload is None
    assert err == "endpoint_unavailable"
    assert capability["unsupported"] is True
    assert len(calls) == 4

    payload, err = tasty._fetch_market_metrics(
        tasty.SANDBOX_BASE_URL,
        symbol="QQQ",
        user_agent="schwab-trading-bot/1.0",
        session_token="token",
        timeout=8.0,
        capability_state=capability,
    )

    assert payload is None
    assert err == "endpoint_unavailable"
    assert len(calls) == 4


def test_load_recent_schwab_price_history_reads_recent_tail_rows(tmp_path: Path) -> None:
    governance = tmp_path / "governance" / "shadow_conservative_equities"
    governance.mkdir(parents=True)
    path = governance / "master_control_20260320.jsonl"

    older_lines = []
    for idx in range(40):
        older_lines.append(
            {
                "timestamp_utc": f"2026-03-20T08:{idx:02d}:00+00:00",
                "symbol": "SPY",
                "market": {"last_price": 500.0 + idx},
            }
        )
    recent_lines = [
        {
            "timestamp_utc": "2026-03-20T15:01:00+00:00",
            "symbol": "SPY",
            "market": {"last_price": 512.25},
        },
        {
            "timestamp_utc": "2026-03-20T15:02:00+00:00",
            "symbol": "QQQ",
            "market": {"last_price": 440.5},
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in older_lines + recent_lines) + "\n",
        encoding="utf-8",
    )

    history = tasty._load_recent_schwab_price_history(
        tmp_path,
        since=datetime(2026, 3, 20, 15, 0, tzinfo=timezone.utc),
        tail_bytes=512,
    )

    assert history["SPY"][-1][1] == 512.25
    assert history["QQQ"][-1][1] == 440.5


def test_collect_tastytrade_context_falls_back_to_live_when_sandbox_auth_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[tuple[str, bool]] = []

    def _fake_post_session(base_url: str, *, user_agent: str, login: str, password: str, timeout: float) -> tuple[str | None, str | None]:
        attempts.append((base_url, "cert" in base_url))
        if "cert" in base_url:
            return None, "HTTP Error 401: Unauthorized"
        return "live-token", None

    monkeypatch.setattr(tasty, "_post_session", _fake_post_session)
    monkeypatch.setattr(tasty, "_fetch_public_watchlists", lambda *args, **kwargs: ({}, None))
    monkeypatch.setattr(tasty, "_load_recent_schwab_price_history", lambda *args, **kwargs: {"SPY": []})
    monkeypatch.setattr(
        tasty,
        "_fetch_option_chain_nested",
        lambda *args, **kwargs: (
            {"data": {"items": [{"expirations": [{"days-to-expiration": 3, "strikes": [{"strike-price": 500}]}]}]}},
            None,
        ),
    )
    monkeypatch.setattr(
        tasty,
        "_fetch_market_metrics",
        lambda *args, **kwargs: ({"data": {"items": [{"symbol": "SPY", "underlying-price": 510.0, "iv-rank": 25}]}}, None),
    )
    monkeypatch.setattr(
        tasty,
        "_fetch_equity_instrument",
        lambda *args, **kwargs: ({"data": {"items": [{"symbol": "SPY", "lendability": "easy to borrow"}]}}, None),
    )
    monkeypatch.setattr(tasty, "_align_symbol_features_with_schwab", lambda **kwargs: (kwargs["features"], {"ok": True}))

    payload, status = tasty.collect_tastytrade_context(
        login="real-login",
        password="real-password",
        symbols=["SPY"],
        user_agent="schwab-trading-bot/1.0",
        timeout_seconds=8.0,
        sandbox=True,
        schwab_alignment_hours=6.0,
        max_schwab_relative_spread=0.05,
        schwab_tolerance_minutes=25.0,
        schwab_alignment_max_bytes=4096,
    )

    assert status["ok"] is True
    assert status["sandbox"] is False
    assert status["requested_sandbox"] is True
    assert status["base_url"] == tasty.LIVE_BASE_URL
    assert payload["sources"]["session"]["fallback_used"] is True
    assert len(payload["sources"]["session"]["attempts"]) == 2
    assert payload["sources"]["session"]["source_confidence_norm"] > 0.0
    assert payload["sources"]["market_metrics"]["schema_confidence_norm"] > 0.0
    assert payload["derived"]["symbol_features"]["SPY"]["source_confidence_norm"] > 0.0
    assert "source_contracts" in status
    assert payload["collection_contract"]["provider_confidence_norm"] > 0.0
    assert attempts[0][1] is True
    assert attempts[1][0] == tasty.LIVE_BASE_URL


def test_collect_tastytrade_context_marks_live_credential_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        tasty,
        "_post_session",
        lambda *args, **kwargs: (None, "HTTP Error 401: Unauthorized"),
    )

    payload, status = tasty.collect_tastytrade_context(
        login="real-login",
        password="real-password",
        symbols=["SPY"],
        user_agent="schwab-trading-bot/1.0",
        timeout_seconds=8.0,
        sandbox=False,
        schwab_alignment_hours=6.0,
        max_schwab_relative_spread=0.05,
        schwab_tolerance_minutes=25.0,
        schwab_alignment_max_bytes=4096,
    )

    assert status["ok"] is False
    assert status["auth_issue"] == "live_credentials_rejected"
    assert status["operator_action_required"] is True
    assert payload["sources"]["session"]["operator_action_required"] is True
    assert payload["sources"]["session"]["recommended_action"] == "refresh_tastytrade_live_credentials"
    assert payload["sources"]["session"]["source_confidence_norm"] > 0.0
