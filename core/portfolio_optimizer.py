from typing import Dict


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
