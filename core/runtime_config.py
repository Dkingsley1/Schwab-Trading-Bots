import os
from dataclasses import dataclass
from typing import Dict, List


VALID_MODES = {"shadow", "paper", "live"}


@dataclass
class RuntimeConfig:
    mode: str = "shadow"
    max_position_size: int = 10
    max_daily_loss: float = 500.0
    min_shadow_days: int = 14
    min_paper_trades: int = 100
    min_paper_expectancy: float = 0.0

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        mode = os.getenv("BOT_MODE", "shadow").strip().lower()
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid BOT_MODE '{mode}'. Use one of: {sorted(VALID_MODES)}")

        return cls(
            mode=mode,
            max_position_size=int(os.getenv("MAX_POSITION_SIZE", "10")),
            max_daily_loss=float(os.getenv("MAX_DAILY_LOSS", "500")),
            min_shadow_days=int(os.getenv("MIN_SHADOW_DAYS", "14")),
            min_paper_trades=int(os.getenv("MIN_PAPER_TRADES", "100")),
            min_paper_expectancy=float(os.getenv("MIN_PAPER_EXPECTANCY", "0.0")),
        )


def evaluate_go_live_gates(metrics: Dict[str, float], cfg: RuntimeConfig) -> List[str]:
    failures: List[str] = []

    shadow_days = float(metrics.get("shadow_days", 0.0))
    paper_trades = float(metrics.get("paper_trades", 0.0))
    expectancy = float(metrics.get("paper_expectancy", -999.0))
    max_dd = float(metrics.get("paper_max_drawdown", 1e9))

    if shadow_days < cfg.min_shadow_days:
        failures.append(f"shadow_days {shadow_days:.0f} < {cfg.min_shadow_days}")
    if paper_trades < cfg.min_paper_trades:
        failures.append(f"paper_trades {paper_trades:.0f} < {cfg.min_paper_trades}")
    if expectancy < cfg.min_paper_expectancy:
        failures.append(f"paper_expectancy {expectancy:.4f} < {cfg.min_paper_expectancy:.4f}")
    if max_dd > cfg.max_daily_loss:
        failures.append(f"paper_max_drawdown {max_dd:.2f} > {cfg.max_daily_loss:.2f}")

    return failures
