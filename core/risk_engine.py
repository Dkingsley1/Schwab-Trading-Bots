from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class RiskResult:
    action: str
    risk_limit_ok: bool
    reasons: List[str]
    gates: Dict[str, bool]


class RiskEngine:
    def __init__(
        self,
        *,
        max_symbol_exposure_count: int,
        max_volatility_1m: float,
        max_drawdown_proxy: float,
        max_var_proxy: float,
        max_factor_exposure: float,
        max_daily_loss_proxy: float,
    ) -> None:
        self.max_symbol_exposure_count = max(int(max_symbol_exposure_count), 1)
        self.max_volatility_1m = max(float(max_volatility_1m), 0.0)
        self.max_drawdown_proxy = max(float(max_drawdown_proxy), 0.0)
        self.max_var_proxy = max(float(max_var_proxy), 0.0)
        self.max_factor_exposure = max(float(max_factor_exposure), 0.0)
        self.max_daily_loss_proxy = max(float(max_daily_loss_proxy), 0.0)

    @classmethod
    def from_env(cls) -> "RiskEngine":
        import os

        return cls(
            max_symbol_exposure_count=int(os.getenv("RISK_MAX_SYMBOL_EXPOSURE_COUNT", "3")),
            max_volatility_1m=float(os.getenv("RISK_MAX_VOLATILITY_1M", "0.03")),
            max_drawdown_proxy=float(os.getenv("RISK_MAX_DRAWDOWN_PROXY", "0.08")),
            max_var_proxy=float(os.getenv("RISK_MAX_VAR_PROXY", "0.02")),
            max_factor_exposure=float(os.getenv("RISK_MAX_FACTOR_EXPOSURE", "1.5")),
            max_daily_loss_proxy=float(os.getenv("RISK_MAX_DAILY_LOSS_PROXY", "0.05")),
        )

    def enforce(
        self,
        *,
        action: str,
        symbol: str,
        exposure_state: Dict[str, int],
        features: Dict[str, float],
    ) -> RiskResult:
        action = (action or "HOLD").upper()
        reasons: List[str] = []
        gates = {
            "risk_volatility_ok": True,
            "risk_drawdown_ok": True,
            "risk_symbol_exposure_ok": True,
            "risk_var_ok": True,
            "risk_factor_ok": True,
            "risk_daily_loss_ok": True,
        }

        vol_1m = float(features.get("volatility_1m", 0.0) or 0.0)
        dd_proxy = abs(float(features.get("drawdown_proxy", 0.0) or 0.0))
        symbol_cnt = int(exposure_state.get(symbol, 0) or 0)
        var_proxy = abs(float(features.get("var_proxy", vol_1m * 1.65) or 0.0))
        factor_exp = abs(float(features.get("factor_exposure", 0.0) or 0.0))
        daily_loss_proxy = abs(float(features.get("daily_loss_proxy", 0.0) or 0.0))

        if vol_1m > self.max_volatility_1m and action in {"BUY", "SELL"}:
            action = "HOLD"
            gates["risk_volatility_ok"] = False
            reasons.append(f"risk_volatility_limit vol_1m={vol_1m:.4f}>{self.max_volatility_1m:.4f}")

        if dd_proxy > self.max_drawdown_proxy and action in {"BUY", "SELL"}:
            action = "HOLD"
            gates["risk_drawdown_ok"] = False
            reasons.append(f"risk_drawdown_limit dd_proxy={dd_proxy:.4f}>{self.max_drawdown_proxy:.4f}")

        if symbol_cnt >= self.max_symbol_exposure_count and action in {"BUY", "SELL"}:
            action = "HOLD"
            gates["risk_symbol_exposure_ok"] = False
            reasons.append(f"risk_symbol_exposure_limit symbol={symbol} count={symbol_cnt}")

        if var_proxy > self.max_var_proxy and action in {"BUY", "SELL"}:
            action = "HOLD"
            gates["risk_var_ok"] = False
            reasons.append(f"risk_var_limit var_proxy={var_proxy:.4f}>{self.max_var_proxy:.4f}")

        if factor_exp > self.max_factor_exposure and action in {"BUY", "SELL"}:
            action = "HOLD"
            gates["risk_factor_ok"] = False
            reasons.append(f"risk_factor_limit factor_exposure={factor_exp:.4f}>{self.max_factor_exposure:.4f}")

        if daily_loss_proxy > self.max_daily_loss_proxy and action in {"BUY", "SELL"}:
            action = "HOLD"
            gates["risk_daily_loss_ok"] = False
            reasons.append(f"risk_daily_loss_limit daily_loss_proxy={daily_loss_proxy:.4f}>{self.max_daily_loss_proxy:.4f}")

        return RiskResult(action=action, risk_limit_ok=all(gates.values()), reasons=reasons, gates=gates)


def apply_risk_limits(
    *,
    action: str,
    symbol: str,
    exposure_state: Dict[str, int],
    features: Dict[str, float],
) -> Tuple[str, List[str], Dict[str, bool]]:
    engine = RiskEngine.from_env()
    res = engine.enforce(action=action, symbol=symbol, exposure_state=exposure_state, features=features)
    return res.action, res.reasons, res.gates
