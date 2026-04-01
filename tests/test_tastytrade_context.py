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
                {"expirations": [{"days-to-expiration": 2, "strikes": [100, 105, 110, 115]}]}
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
                }
            ]
        }
    }

    out = tasty._derive_symbol_features(
        symbol="SPY",
        option_chain_payload=option_chain_payload,
        market_metrics_payload=market_metrics_payload,
        watchlist_symbols={"SPY"},
    )

    assert out["tasty_iv_rank_norm"] > 0.0
    assert out["tasty_implied_volatility_index_norm"] > 0.0
    assert out["tasty_liquidity_rating_norm"] > 0.0
    assert out["tasty_expected_move_norm"] > 0.0
    assert out["tasty_beta_norm"] > 0.0
    assert out["tasty_watchlist_presence_norm"] == 1.0


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
