#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.advanced_quant_models import quant_model_inventory, real_time_kalman_filter_gpu


def _values_for_symbol(symbol: str, observations: int) -> list[float]:
    seed = sum(ord(ch) for ch in symbol)
    return [math.sin((seed + i) / 13.0) * 0.012 + math.cos((seed + i) / 29.0) * 0.006 for i in range(max(observations, 4))]


def build_payload(args: argparse.Namespace) -> dict[str, object]:
    symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
    if not symbols:
        symbols = ["SPY", "QQQ", "IWM", "TLT", "GLD", "UUP"]
    results = {
        symbol: real_time_kalman_filter_gpu(_values_for_symbol(symbol, args.observations), window=args.window)
        for symbol in symbols
    }
    inventory = quant_model_inventory()
    confidences = [float(row.get("confidence", 0.0) or 0.0) for row in results.values()]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ok": True,
        "script": "kalman_parallel.py",
        "purpose": "GPU-targeted multi-asset regime filter for parallel Kalman-style state confidence",
        "symbols": symbols,
        "symbol_count": len(symbols),
        "observations_per_symbol": int(args.observations),
        "window": int(args.window),
        "mean_confidence": round(sum(confidences) / max(len(confidences), 1), 6),
        "results": results,
        "mlx_hooks": inventory.get("mlx_hooks", {}),
        "execution_policy": {"direct_execution_allowed": False, "paper_trading_allowed": False},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a GPU-targeted parallel Kalman-style filter for multi-asset regimes.")
    parser.add_argument("--symbols", default="SPY,QQQ,IWM,TLT,GLD,UUP")
    parser.add_argument("--observations", type=int, default=256)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = build_payload(args)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "kalman_parallel "
            f"symbols={payload.get('symbol_count')} "
            f"mean_confidence={float(payload.get('mean_confidence', 0.0) or 0.0):.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
