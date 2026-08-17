from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class PortfolioIntent:
    symbol: str
    sleeve: str
    side: str
    raw_qty: float
    score: float
    volatility_1m: float
    price: float = 1.0
    sector: str = "unknown"
    factor_exposure: float = 0.0
    capacity_fraction: float = 1.0
    venue: str = "primary"
    clock_bucket: str = "all_day"
    regime: str = "normal"
    forward_cost_bps: float = 0.0


@dataclass(frozen=True)
class AllocatedIntent:
    symbol: str
    sleeve: str
    side: str
    raw_qty: float
    approved_qty: float
    weight_scale: float
    net_symbol_qty: float
    sector: str
    factor_exposure: float
    score: float
    volatility_1m: float
    price: float = 1.0
    capacity_fraction: float = 1.0
    venue: str = "primary"
    clock_bucket: str = "all_day"
    regime: str = "normal"
    forward_cost_bps: float = 0.0
    reasons: tuple[str, ...] = ()


def risk_budgeted_weight(
    *,
    symbol: str,
    score: float,
    volatility_1m: float,
    base_budget: float,
    symbol_budgets: Dict[str, float],
) -> float:
    s = max(float(score), 0.0)
    vol = max(float(volatility_1m), 1e-6)
    edge_weight = s / max(vol, 1e-6)
    symbol_cap = float(symbol_budgets.get(symbol.upper(), base_budget))
    return min(edge_weight, symbol_cap)


def allocate_quantity(
    *,
    raw_qty: float,
    symbol: str,
    score: float,
    volatility_1m: float,
    base_budget: float,
    symbol_budgets: Dict[str, float],
) -> float:
    rbw = risk_budgeted_weight(
        symbol=symbol,
        score=score,
        volatility_1m=volatility_1m,
        base_budget=base_budget,
        symbol_budgets=symbol_budgets,
    )
    return round(max(float(raw_qty), 0.0) * max(min(rbw, 1.0), 0.0), 6)


def _direction(side: str) -> float:
    text = str(side or "").strip().upper()
    if text.startswith("SELL"):
        return -1.0
    return 1.0


def _normalize_sector(text: str) -> str:
    value = str(text or "").strip().lower()
    return value or "unknown"


def allocate_portfolio_intents(
    intents: Iterable[PortfolioIntent],
    *,
    gross_budget: float,
    base_budget: float,
    symbol_budgets: Dict[str, float],
    sector_budgets: Dict[str, float],
    factor_cap: float,
) -> List[AllocatedIntent]:
    gross_budget = max(min(float(gross_budget), 1.0), 0.0)
    factor_cap = max(float(factor_cap), 0.0)
    sector_remaining = {
        _normalize_sector(key): max(float(value), 0.0) for key, value in (sector_budgets or {}).items()
    }
    if "unknown" not in sector_remaining:
        sector_remaining["unknown"] = max(float(base_budget), 0.0)

    net_symbol_qty: Dict[str, float] = {}
    rows: List[PortfolioIntent] = list(intents)
    for intent in rows:
        symbol = str(intent.symbol or "").strip().upper()
        net_symbol_qty[symbol] = net_symbol_qty.get(symbol, 0.0) + (_direction(intent.side) * max(float(intent.raw_qty), 0.0))

    allocated: List[AllocatedIntent] = []
    factor_used = 0.0
    gross_used = 0.0

    for intent in sorted(rows, key=lambda row: (float(row.score), -float(row.volatility_1m or 0.0)), reverse=True):
        symbol = str(intent.symbol or "").strip().upper()
        sector = _normalize_sector(intent.sector)
        raw_qty = max(float(intent.raw_qty), 0.0)
        reasons: List[str] = []
        if raw_qty <= 0.0:
            allocated.append(
                AllocatedIntent(
                    symbol=symbol,
                    sleeve=str(intent.sleeve or ""),
                    side=str(intent.side or ""),
                    raw_qty=0.0,
                    approved_qty=0.0,
                    weight_scale=0.0,
                    net_symbol_qty=round(net_symbol_qty.get(symbol, 0.0), 6),
                    sector=sector,
                    factor_exposure=float(intent.factor_exposure or 0.0),
                    score=round(float(intent.score or 0.0), 6),
                    volatility_1m=round(float(intent.volatility_1m or 0.0), 6),
                    price=round(max(float(intent.price or 1.0), 1e-6), 6),
                    capacity_fraction=round(float(intent.capacity_fraction or 1.0), 6),
                    venue=str(intent.venue or "primary"),
                    clock_bucket=str(intent.clock_bucket or "all_day"),
                    regime=str(intent.regime or "normal"),
                    forward_cost_bps=round(float(intent.forward_cost_bps or 0.0), 6),
                    reasons=("zero_raw_qty",),
                )
            )
            continue

        weight_scale = min(
            risk_budgeted_weight(
                symbol=symbol,
                score=float(intent.score or 0.0),
                volatility_1m=float(intent.volatility_1m or 0.0),
                base_budget=float(base_budget),
                symbol_budgets=symbol_budgets,
            ),
            max(float(intent.capacity_fraction or 1.0), 0.0),
        )
        reasons.append(f"risk_weight={weight_scale:.4f}")

        symbol_net = abs(float(net_symbol_qty.get(symbol, 0.0)))
        net_scale = 1.0 if symbol_net <= 0.0 else min(1.0, symbol_net / max(raw_qty, 1e-6))
        if net_scale < 1.0:
            reasons.append("cross_sleeve_netting")

        sector_limit = sector_remaining.get(sector, sector_remaining.get("unknown", float(base_budget)))
        sector_scale = min(1.0, sector_limit / max(raw_qty, 1e-6)) if sector_limit > 0.0 else 0.0
        if sector_scale < 1.0:
            reasons.append("sector_budget")

        factor_headroom = max(factor_cap - factor_used, 0.0)
        factor_need = abs(float(intent.factor_exposure or 0.0)) * raw_qty
        factor_scale = 1.0 if factor_need <= 0.0 else min(1.0, factor_headroom / max(factor_need, 1e-6))
        if factor_scale < 1.0:
            reasons.append("factor_cap")

        forward_cost_bps = max(float(intent.forward_cost_bps or 0.0), 0.0)
        cost_scale = max(0.25, 1.0 - min(forward_cost_bps, 75.0) / 100.0)
        if cost_scale < 1.0:
            reasons.append("forward_cost_curve")

        gross_headroom = max(gross_budget - gross_used, 0.0)
        gross_scale = min(1.0, gross_headroom / max(raw_qty, 1e-6)) if gross_headroom > 0.0 else 0.0
        if gross_scale < 1.0:
            reasons.append("gross_budget")

        approved_scale = max(min(weight_scale * net_scale * sector_scale * factor_scale * gross_scale * cost_scale, 1.0), 0.0)
        approved_qty = round(raw_qty * approved_scale, 6)
        gross_used += approved_qty
        factor_used += abs(float(intent.factor_exposure or 0.0)) * approved_qty
        sector_remaining[sector] = max(sector_limit - approved_qty, 0.0)

        allocated.append(
            AllocatedIntent(
                symbol=symbol,
                sleeve=str(intent.sleeve or ""),
                side=str(intent.side or ""),
                raw_qty=round(raw_qty, 6),
                approved_qty=approved_qty,
                weight_scale=round(approved_scale, 6),
                net_symbol_qty=round(net_symbol_qty.get(symbol, 0.0), 6),
                sector=sector,
                factor_exposure=round(float(intent.factor_exposure or 0.0), 6),
                score=round(float(intent.score or 0.0), 6),
                volatility_1m=round(float(intent.volatility_1m or 0.0), 6),
                price=round(max(float(intent.price or 1.0), 1e-6), 6),
                capacity_fraction=round(float(intent.capacity_fraction or 1.0), 6),
                venue=str(intent.venue or "primary"),
                clock_bucket=str(intent.clock_bucket or "all_day"),
                regime=str(intent.regime or "normal"),
                forward_cost_bps=round(forward_cost_bps, 6),
                reasons=tuple(reasons),
            )
        )

    return allocated


def allocated_rows_as_dicts(rows: Iterable[AllocatedIntent]) -> List[dict]:
    return [asdict(row) for row in rows]
