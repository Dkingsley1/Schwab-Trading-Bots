# Go-Live Gates

Use these minimum checks before enabling `BOT_MODE=live`.

1. Shadow mode runtime: at least 14 days.
2. Paper trade count: at least 100 closed decisions.
3. Paper expectancy after fees/slippage: above 0.
4. Risk discipline: no daily drawdown breach beyond configured limit.
5. Decision logging coverage: every decision must include score, gates, reasons.

Recommended process:

1. Start with `BOT_MODE=shadow`.
2. Move to `BOT_MODE=paper` after stable shadow logs.
3. Run gate evaluation weekly.
4. Enable `BOT_MODE=live` only when all gates pass.
