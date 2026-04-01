from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import collect_crypto_market_context as crypto_ctx


def test_derive_deribit_asset_features() -> None:
    futures_payload = {
        "result": [
            {
                "open_interest": 125_000_000,
                "mark_price": 70500.0,
                "estimated_delivery_price": 70000.0,
                "last": 70520.0,
            }
        ]
    }
    options_payload = {
        "result": [
            {"open_interest": 40_000_000, "mark_iv": 58.0},
            {"open_interest": 20_000_000, "mark_iv": 62.0},
        ]
    }

    features, price = crypto_ctx._derive_deribit_asset_features(futures_payload, options_payload)

    assert price > 0.0
    assert features["crypto_deribit_futures_oi_norm"] > 0.0
    assert features["crypto_deribit_options_oi_norm"] > 0.0
    assert features["crypto_deribit_mark_iv_norm"] > 0.0
    assert 0.0 <= features["crypto_deribit_basis_norm"] <= 1.0
    assert features["crypto_deribit_basis_norm"] != 0.5


def test_derive_hyperliquid_asset_features() -> None:
    features, price = crypto_ctx._derive_hyperliquid_asset_features(
        {
            "funding": "0.00012",
            "openInterest": "25000",
            "oraclePx": "70000",
            "markPx": "70140",
        }
    )

    assert price == 70140.0
    assert features["crypto_hyperliquid_funding_norm"] > 0.5
    assert features["crypto_hyperliquid_open_interest_norm"] > 0.0
    assert features["crypto_hyperliquid_basis_norm"] > 0.5


def test_build_news_row_filters_non_crypto_headlines() -> None:
    row = crypto_ctx._build_news_row(
        headline="How AI Is Being Used to Clear Court Backlogs in LA",
        summary="A local government pilot program for court workflow automation.",
        publisher="Decrypt",
        source_name="decrypt",
        published="2026-03-22T13:01:03+00:00",
        url="https://decrypt.co/example",
        asset_to_symbols={"BTC": ["BTC-USD"]},
    )

    assert row is None


def test_price_agreement_norm_rewards_tight_cross_provider_quotes() -> None:
    tight = crypto_ctx._price_agreement_norm([70000.0, 70020.0, 69990.0], max_relative_spread=0.05)
    wide = crypto_ctx._price_agreement_norm([70000.0, 76000.0], max_relative_spread=0.05)

    assert tight > 0.9
    assert wide == 0.0


def test_collect_crypto_market_context_keeps_partial_source_issues_as_warnings(monkeypatch: Any) -> None:
    def _deribit(*args: Any, **kwargs: Any) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[str, Any]]:
        return {"BTC": {"crypto_deribit_mark_iv_norm": 0.6}}, {"BTC": 70000.0}, {"ok": True, "error": None}

    def _kraken(*args: Any, **kwargs: Any) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[str, Any]]:
        return {"BTC": {"crypto_kraken_volume_norm": 0.5}}, {"BTC": 70010.0}, {"ok": True, "error": None}

    def _hyper(*args: Any, **kwargs: Any) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[str, Any]]:
        return {"BTC": {"crypto_hyperliquid_funding_norm": 0.52}}, {"BTC": 70005.0}, {"ok": True, "error": None}

    def _coinmetrics(*args: Any, **kwargs: Any) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[str, Any]]:
        return (
            {"BTC": {"crypto_coinmetrics_tx_count_norm": 0.4}},
            {"BTC": 69995.0},
            {"ok": True, "resolved_assets": 1, "error": "SOL:HTTP Error 403: Forbidden"},
        )

    def _defillama(*args: Any, **kwargs: Any) -> tuple[dict[str, float], dict[str, Any]]:
        return {"crypto_defillama_stablecoin_growth_norm": 0.55}, {"ok": True, "error": None}

    def _etherscan(*args: Any, **kwargs: Any) -> tuple[dict[str, float], dict[str, Any]]:
        return {"crypto_etherscan_gas_norm": 0.2}, {"ok": True, "error": None}

    def _coingecko(*args: Any, **kwargs: Any) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[str, Any]]:
        return {"BTC": {"crypto_coingecko_momentum_norm": 0.65}}, {"BTC": 70002.0}, {"ok": True, "error": None}

    monkeypatch.setattr(crypto_ctx, "_collect_deribit", _deribit)
    monkeypatch.setattr(crypto_ctx, "_collect_kraken", _kraken)
    monkeypatch.setattr(crypto_ctx, "_collect_hyperliquid", _hyper)
    monkeypatch.setattr(crypto_ctx, "_collect_coinmetrics", _coinmetrics)
    monkeypatch.setattr(crypto_ctx, "_collect_defillama", _defillama)
    monkeypatch.setattr(crypto_ctx, "_collect_etherscan", _etherscan)
    monkeypatch.setattr(crypto_ctx, "_collect_coingecko", _coingecko)
    monkeypatch.setattr(crypto_ctx, "_collect_crypto_news", lambda **kwargs: ([], {}))

    _, status = crypto_ctx.collect_crypto_market_context(
        symbols=["BTC-USD"],
        user_agent="test/1.0",
    )

    assert status["ok"] is True
    assert status["error_count"] == 0
    assert status["warning_count"] == 1
    assert status["warnings"] == ["coinmetrics:SOL:HTTP Error 403: Forbidden"]


def test_collect_crypto_market_context_includes_news_features(monkeypatch: Any) -> None:
    monkeypatch.setattr(crypto_ctx, "_collect_deribit", lambda *args, **kwargs: ({}, {}, {"ok": False, "error": None}))
    monkeypatch.setattr(crypto_ctx, "_collect_kraken", lambda *args, **kwargs: ({}, {}, {"ok": False, "error": None}))
    monkeypatch.setattr(crypto_ctx, "_collect_hyperliquid", lambda *args, **kwargs: ({}, {}, {"ok": False, "error": None}))
    monkeypatch.setattr(crypto_ctx, "_collect_coinmetrics", lambda *args, **kwargs: ({}, {}, {"ok": False, "error": None}))
    monkeypatch.setattr(crypto_ctx, "_collect_defillama", lambda *args, **kwargs: ({}, {"ok": False, "error": None}))
    monkeypatch.setattr(crypto_ctx, "_collect_etherscan", lambda *args, **kwargs: ({}, {"ok": False, "error": None}))
    monkeypatch.setattr(crypto_ctx, "_collect_coingecko", lambda *args, **kwargs: ({}, {}, {"ok": False, "error": None}))
    monkeypatch.setattr(
        crypto_ctx,
        "_collect_crypto_news",
        lambda **kwargs: (
            [
                {
                    "headline": "Coinbase Status: Delayed Transactions on Litecoin Network",
                    "summary": "Investigating delayed LTC transfers.",
                    "publisher": "Coinbase Status",
                    "source": "Coinbase Status",
                    "published": "2026-03-22T13:30:00+00:00",
                    "symbols": ["LTC", "LTC-USD"],
                    "relatedSymbols": ["LTC", "LTC-USD"],
                    "sentiment_hint": -0.9,
                    "shock_hint": 0.8,
                    "broad_market": True,
                },
                {
                    "headline": "CoinDesk: Bitcoin ETF inflows support rebound",
                    "summary": "BTC and ETH gain as inflows accelerate.",
                    "publisher": "CoinDesk",
                    "source": "CoinDesk",
                    "published": "2026-03-22T13:50:00+00:00",
                    "symbols": ["BTC", "BTC-USD", "ETH", "ETH-USD"],
                    "relatedSymbols": ["BTC", "BTC-USD", "ETH", "ETH-USD"],
                    "sentiment_hint": 0.7,
                    "shock_hint": 0.5,
                    "broad_market": True,
                },
            ],
            {
                "coinbase_status": {"ok": True, "rows": 1, "error": None},
                "coindesk": {"ok": True, "rows": 1, "error": None},
            },
        ),
    )

    payload, status = crypto_ctx.collect_crypto_market_context(
        symbols=["BTC-USD", "ETH-USD", "LTC-USD"],
        user_agent="test/1.0",
    )

    assert status["news_row_count"] == 2
    assert payload["derived"]["news_features"]["news_available"] == 1.0
    assert payload["derived"]["news_features"]["news_items_24h"] > 0.0
    assert payload["derived"]["news_features"]["news_shock_rate"] > 0.0
    assert payload["derived"]["news_features"]["news_source_quality_norm"] > 0.0
    assert payload["derived"]["news_symbol_features"]["LTC-USD"]["news_recent_impact"] > 0.0
    assert payload["derived"]["news_symbol_features"]["BTC-USD"]["news_positive_share"] > 0.0
