#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "sleeve_ticker_universe_latest.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.sleeve_ticker_universe_override"
UNIVERSE_VERSION = "sleeve_ticker_universe_v1"

UNIVERSES: dict[str, list[str]] = {
    "SHADOW_SYMBOLS_CORE": [
        "SPY", "QQQ", "DIA", "IWM", "MDY", "VOO", "VTI", "IVV", "SPLG", "RSP",
        "AAPL", "MSFT", "NVDA", "AMD", "AVGO", "TSM", "ASML", "MU", "ARM", "SMH", "SOXX", "QCOM", "TXN", "AMAT", "LRCX", "KLAC", "INTC",
        "AMZN", "GOOG", "GOOGL", "META", "NFLX", "DIS", "WBD", "ORCL", "CRM", "ADBE", "NOW", "PLTR", "SNOW", "SHOP", "UBER", "ABNB",
        "JPM", "BAC", "GS", "MS", "BLK", "SCHW", "AXP", "V", "MA", "C", "WFC", "COF",
        "LLY", "UNH", "JNJ", "ABBV", "MRK", "ABT", "PFE", "ISRG", "TMO", "DHR",
        "COST", "WMT", "HD", "LOW", "MCD", "NKE", "SBUX", "TGT",
        "CAT", "DE", "GE", "BA", "RTX", "LMT", "NOC", "HON", "ETN",
        "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "VLO", "OXY", "LNG",
        "BKNG", "MAR", "HLT", "DAL", "UAL", "LUV",
    ],
    "SHADOW_SYMBOLS_VOLATILE": [
        "SOXL", "SOXS", "TQQQ", "SQQQ", "SPXL", "SPXS", "LABU", "LABD", "UVXY", "VIXY", "SVXY",
        "MSTR", "SMCI", "COIN", "TSLA", "AMD", "NVDA", "PLTR", "ARM", "MARA", "RIOT", "CLSK", "HOOD", "RBLX", "AFRM", "UPST",
        "IBIT", "FBTC", "BITB", "ARKB", "ETHA", "ETHE",
    ],
    "SHADOW_SYMBOLS_DEFENSIVE": [
        "TLT", "GLD", "XLV", "XLU", "XLP", "MO", "HYG", "LQD", "UUP", "XLE", "XLF", "XLI", "XLK", "XLY", "XLC", "XLB", "XLRE",
        "XAR", "KRE", "XOP", "IEF", "SHY", "TIP", "TLH", "JNK", "AGG", "BND", "MUB", "IGIB", "USHY", "FLOT", "VGIT", "VCIT", "EMB", "BIL", "SGOV", "USFR", "TFLO",
        "SCHD", "VIG", "DGRO", "HDV", "NOBL", "VYM", "DIVO", "JEPI", "JEPQ", "SPLV", "VTV",
        "JNJ", "PG", "KO", "PEP", "MCD", "ABBV", "ABT", "MRK", "PFE", "T", "VZ", "O", "VICI", "MAIN",
        "ITA", "LMT", "NOC", "RTX", "GD", "LHX", "LDOS",
    ],
    "SHADOW_SYMBOLS_COMMOD_FX_INTL": [
        "DBC", "USO", "UNG", "CORN", "WEAT", "SOYB", "SLV", "GLD", "CPER", "URA", "XME", "GDX", "GDXJ",
        "UUP", "FXE", "FXY", "FXB", "FXC", "FXA", "CYB", "EUO", "YCS", "UDN", "CEW", "DBV",
        "EFA", "EEM", "IEFA", "VEA", "VWO", "VGK", "EWJ", "FXI", "EWZ", "INDA", "EWU", "EWG", "EWQ", "EWC", "EWA", "EWW", "EWY", "EWT", "IXUS",
    ],
    "DIVIDEND_SYMBOLS": [
        "SCHD", "VIG", "DGRO", "HDV", "NOBL", "VYM", "DIVO", "JEPI", "JEPQ", "SPYD", "DIV", "FDVV", "SCHY", "SDY",
        "JNJ", "PG", "KO", "PEP", "MCD", "MO", "PM", "ABBV", "ABT", "MRK", "PFE", "T", "VZ", "O", "VICI", "MAIN",
        "XOM", "CVX", "COP", "KMI", "MPC", "PSX", "VLO", "EOG", "SLB", "MSFT", "AAPL", "COST", "HD", "LOW", "JPM", "BLK",
    ],
    "DIVIDEND_QUALITY_SYMBOLS": [
        "SCHD", "VIG", "DGRO", "HDV", "NOBL", "VYM", "DIVO", "SCHY", "JNJ", "PG", "KO", "PEP", "MCD", "ABBV", "ABT", "MRK", "XOM", "CVX", "COP", "O", "VICI", "MSFT", "AAPL", "COST", "HD", "LOW",
    ],
    "BOND_SYMBOLS": [
        "TLT", "IEF", "SHY", "TIP", "LQD", "HYG", "JNK", "AGG", "BND", "TLH", "MUB", "IGIB", "USHY", "FLOT", "VGIT", "VCIT", "EMB", "BIL", "SGOV", "USFR", "TFLO", "MINT", "NEAR", "VTIP", "SCHP",
    ],
    "BOND_CONTEXT_SYMBOLS": [
        "UUP", "GLD", "SPY", "QQQ", "TLT", "IEF", "TLH", "VGIT", "SHY", "TIP", "VTIP", "SCHP", "LQD", "IGIB", "HYG", "JNK", "USHY", "AGG", "BND", "MUB", "XLU", "XLF", "XLE", "VIXY", "DBC", "USO",
    ],
    "FX_SYMBOLS": ["UUP", "FXE", "FXY", "FXB", "FXC", "FXA", "CYB", "EUO", "YCS", "UDN", "CEW", "DBV"],
    "FX_CONTEXT_SYMBOLS": ["SPY", "QQQ", "TLT", "GLD", "UUP", "FXE", "FXY", "FXB", "FXC", "FXA", "EFA", "EEM", "USO", "DBC"],
    "COINBASE_WATCH_SYMBOLS": [
        "BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LTC-USD", "LINK-USD", "DOGE-USD", "XRP-USD", "ADA-USD", "DOT-USD", "BCH-USD", "UNI-USD", "AAVE-USD", "ATOM-USD", "NEAR-USD", "OP-USD", "ARB-USD",
    ],
    "COINBASE_FUTURES_WATCH_SYMBOLS": [
        "BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LTC-USD", "LINK-USD", "DOGE-USD", "XRP-USD", "ADA-USD", "DOT-USD",
    ],
    "COINBASE_WEBSOCKET_SYMBOLS": ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LTC-USD", "LINK-USD", "DOGE-USD", "XRP-USD"],
    "CRYPTO_MARKET_CONTEXT_SYMBOLS": [
        "BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LTC-USD", "LINK-USD", "DOGE-USD", "XRP-USD", "ADA-USD", "DOT-USD", "BCH-USD", "UNI-USD", "AAVE-USD", "ATOM-USD",
    ],
    "LONG_TERM_SECTOR_SYMBOLS": [
        "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY", "SMH", "SOXX", "ITB", "KRE", "IBB", "ITA", "JETS", "XOP", "OIH", "XME", "GDX", "URA",
    ],
    "LONG_TERM_SECTOR_CONTEXT_SYMBOLS": ["SPY", "QQQ", "IWM", "TLT", "GLD", "UUP", "VIXY", "HYG", "LQD"],
}


def _csv(values: list[str]) -> str:
    return ",".join(ordered_unique(str(value).strip().upper() for value in values if str(value).strip()))


def _override_lines(payload: dict[str, Any]) -> list[str]:
    env = payload.get("env_overrides") if isinstance(payload.get("env_overrides"), dict) else {}
    lines = [
        "# Auto-managed by scripts/ops/sleeve_ticker_universe_expansion.py",
        f"# Generated at {payload.get('timestamp_utc') or iso_now()}",
    ]
    for key in sorted(env):
        lines.append(f"{key}={shlex.quote(str(env[key]))}")
    return lines


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    del project_root
    env = {key: _csv(values) for key, values in UNIVERSES.items()}
    counts = {key: len(value.split(",")) if value else 0 for key, value in env.items()}
    sleeve_groups = {
        "equity_core": ["SHADOW_SYMBOLS_CORE", "SHADOW_SYMBOLS_VOLATILE", "SHADOW_SYMBOLS_DEFENSIVE"],
        "cross_asset": ["SHADOW_SYMBOLS_COMMOD_FX_INTL", "FX_SYMBOLS", "FX_CONTEXT_SYMBOLS"],
        "income_rates": ["DIVIDEND_SYMBOLS", "DIVIDEND_QUALITY_SYMBOLS", "BOND_SYMBOLS", "BOND_CONTEXT_SYMBOLS"],
        "crypto": ["COINBASE_WATCH_SYMBOLS", "COINBASE_FUTURES_WATCH_SYMBOLS", "COINBASE_WEBSOCKET_SYMBOLS", "CRYPTO_MARKET_CONTEXT_SYMBOLS"],
        "long_term_sector": ["LONG_TERM_SECTOR_SYMBOLS", "LONG_TERM_SECTOR_CONTEXT_SYMBOLS"],
    }
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": True,
        "overall_status": "ready",
        "universe_version": UNIVERSE_VERSION,
        "env_overrides": {
            **env,
            "SLEEVE_TICKER_UNIVERSE_ENABLED": "1",
            "SLEEVE_TICKER_UNIVERSE_VERSION": UNIVERSE_VERSION,
            "SLEEVE_TICKER_UNIVERSE_POLICY": "expanded_applicable_sleeves_with_provider_guarded_crypto_websocket_subset",
        },
        "symbol_counts": counts,
        "sleeve_groups": sleeve_groups,
        "safety_contract": {
            "market_data_only": "1",
            "adds_live_execution": False,
            "coinbase_websocket_subset": "kept_smaller_than_full_crypto_watchlist",
            "applies_through_runtime_env_override": True,
        },
    }


def apply_payload(
    project_root: Path,
    payload: dict[str, Any],
    *,
    out_path: Path = DEFAULT_OUT_PATH,
    override_path: Path = DEFAULT_OVERRIDE_PATH,
) -> dict[str, Any]:
    out = out_path if out_path.is_absolute() else project_root / out_path
    override = override_path if override_path.is_absolute() else project_root / override_path
    override.parent.mkdir(parents=True, exist_ok=True)
    override.write_text("\n".join(_override_lines(payload)) + "\n", encoding="utf-8")
    payload = dict(payload)
    payload["apply_result"] = {
        "applied": True,
        "override_path": str(override),
        "health_path": str(out),
    }
    write_payload(out, payload)
    payload["out_path"] = str(out)
    return payload


def _print_human(payload: dict[str, Any]) -> None:
    counts = payload.get("symbol_counts") if isinstance(payload.get("symbol_counts"), dict) else {}
    print(
        "sleeve_ticker_universe "
        f"status={payload.get('overall_status')} "
        f"core={counts.get('SHADOW_SYMBOLS_CORE')} "
        f"defensive={counts.get('SHADOW_SYMBOLS_DEFENSIVE')} "
        f"crypto={counts.get('COINBASE_WATCH_SYMBOLS')}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Expand ticker universes across applicable live-data sleeves.")
    parser.add_argument("--apply", action="store_true", help="Write health artifact and runtime env override.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH, help="Health artifact path.")
    parser.add_argument("--override", type=Path, default=DEFAULT_OVERRIDE_PATH, help="Runtime env override path.")
    args = parser.parse_args(argv)

    payload = build_payload(PROJECT_ROOT)
    if args.apply:
        payload = apply_payload(PROJECT_ROOT, payload, out_path=args.out, override_path=args.override)
    else:
        payload = {
            **payload,
            "apply_result": {
                "applied": False,
                "override_path": str(args.override if args.override.is_absolute() else PROJECT_ROOT / args.override),
                "health_path": str(args.out if args.out.is_absolute() else PROJECT_ROOT / args.out),
            },
            "out_path": str(args.out if args.out.is_absolute() else PROJECT_ROOT / args.out),
        }

    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        _print_human(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
