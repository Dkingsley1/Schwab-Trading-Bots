#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.advanced_quant_models import gpu_accelerated_monte_carlo_price, quant_model_inventory


def build_payload(args: argparse.Namespace) -> dict[str, object]:
    start = time.perf_counter()
    result = gpu_accelerated_monte_carlo_price(
        args.spot,
        args.strike,
        args.time_years,
        args.rate,
        args.volatility,
        option_type=args.option_type,
        paths=args.paths,
    )
    elapsed = max(time.perf_counter() - start, 1e-9)
    inventory = quant_model_inventory()
    mlx_hooks = inventory.get("mlx_hooks", {}) if isinstance(inventory.get("mlx_hooks"), dict) else {}
    mlx_available = bool(mlx_hooks.get("mlx_core_random")) and bool(mlx_hooks.get("mx_grad"))
    effective_paths = int(args.paths if mlx_available else min(args.paths, 2048))
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ok": True,
        "script": "gpu_mc_sim.py",
        "purpose": "GPU-targeted Monte Carlo engine for high-path pricing sweeps",
        "target_profile": "1m_paths_under_1s_when_mlx_gpu_runtime_has_sufficient_headroom",
        "requested_paths": int(args.paths),
        "effective_paths": effective_paths,
        "elapsed_seconds": round(elapsed, 6),
        "paths_per_second": round(effective_paths / elapsed, 3),
        "result": result,
        "mlx_hooks": mlx_hooks,
        "execution_policy": {"direct_execution_allowed": False, "paper_trading_allowed": False},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a GPU-targeted Monte Carlo pricing sweep with safe CPU fallback.")
    parser.add_argument("--paths", type=int, default=1_000_000)
    parser.add_argument("--spot", type=float, default=100.0)
    parser.add_argument("--strike", type=float, default=100.0)
    parser.add_argument("--time-years", type=float, default=30.0 / 365.0)
    parser.add_argument("--rate", type=float, default=0.045)
    parser.add_argument("--volatility", type=float, default=0.22)
    parser.add_argument("--option-type", choices=("call", "put"), default="call")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = build_payload(args)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        result = payload["result"] if isinstance(payload.get("result"), dict) else {}
        print(
            "gpu_mc_sim "
            f"price={float(result.get('price', 0.0) or 0.0):.6f} "
            f"paths_per_second={float(payload.get('paths_per_second', 0.0) or 0.0):.0f} "
            f"effective_paths={payload.get('effective_paths')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
