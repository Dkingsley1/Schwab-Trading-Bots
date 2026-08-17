#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.advanced_quant_models import mlx_jump_diffusion_gradient, quant_model_inventory


def build_payload(args: argparse.Namespace) -> dict[str, object]:
    grad = mlx_jump_diffusion_gradient(
        args.spot,
        args.strike,
        args.time_years,
        args.rate,
        args.volatility,
        jump_intensity=args.jump_intensity,
        jump_mean=args.jump_mean,
        jump_volatility=args.jump_volatility,
        option_type=args.option_type,
    )
    inventory = quant_model_inventory()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ok": True,
        "script": "pricing_grad.py",
        "purpose": "mx.grad fair_value_func Greeks for exotic/jump-diffusion proxy options",
        "inputs": {
            "spot": args.spot,
            "strike": args.strike,
            "time_years": args.time_years,
            "rate": args.rate,
            "volatility": args.volatility,
            "jump_intensity": args.jump_intensity,
            "jump_mean": args.jump_mean,
            "jump_volatility": args.jump_volatility,
            "option_type": args.option_type,
        },
        "greeks": grad,
        "mlx_hooks": inventory.get("mlx_hooks", {}),
        "execution_policy": {"direct_execution_allowed": False, "paper_trading_allowed": False},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute MLX mx.grad Greeks for a jump-diffusion fair-value proxy.")
    parser.add_argument("--spot", type=float, default=100.0)
    parser.add_argument("--strike", type=float, default=100.0)
    parser.add_argument("--time-years", type=float, default=30.0 / 365.0)
    parser.add_argument("--rate", type=float, default=0.045)
    parser.add_argument("--volatility", type=float, default=0.22)
    parser.add_argument("--jump-intensity", type=float, default=0.35)
    parser.add_argument("--jump-mean", type=float, default=-0.04)
    parser.add_argument("--jump-volatility", type=float, default=0.18)
    parser.add_argument("--option-type", choices=("call", "put"), default="call")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = build_payload(args)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        greeks = payload["greeks"] if isinstance(payload.get("greeks"), dict) else {}
        print(
            "pricing_grad "
            f"delta={float(greeks.get('delta', 0.0) or 0.0):.6f} "
            f"gamma={float(greeks.get('gamma', 0.0) or 0.0):.6f} "
            f"mlx_grad={bool(greeks.get('grad_available', 0.0))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
