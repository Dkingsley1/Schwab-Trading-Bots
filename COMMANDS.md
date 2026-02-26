# Commands (Canonical)

Use these exact commands as the source of truth.

## 1) Start All Sleeves (single live terminal feed)
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot && \
export MARKET_DATA_ONLY=1 ALLOW_ORDER_EXECUTION=0 ENABLE_RESOURCE_GUARD=0 && \
export SHADOW_SYMBOLS_CORE="SPY,QQQ,AAPL,MSFT,NVDA,DIA,IWM,MDY" && \
export SHADOW_SYMBOLS_VOLATILE="SOXL,SOXS,MSTR,SMCI,COIN,TSLA,UVXY,VIXY" && \
export SHADOW_SYMBOLS_DEFENSIVE="TLT,GLD,XLV,XLU,XLP,HYG,LQD,UUP,XLE,XLF,XLI,XLK,XLY,IEF,SHY,TIP,TLH,JNK" && \
export SHADOW_SYMBOLS_COMMOD_FX_INTL="DBC,UNG,CORN,SLV,USO,FXE,FXY,EFA,EEM,EWJ,FXI" && \
export DIVIDEND_SYMBOLS="SCHD,VIG,DGRO,JNJ,PG,KO,PEP,XOM,CVX,O,MAIN" && \
export BOND_SYMBOLS="TLT,IEF,SHY,TLH,TIP,LQD,HYG,JNK,BND,AGG,MUB,IGIB,USHY,FLOT,VGIT" && \
./.venv312/bin/python scripts/run_all_sleeves.py --simulate --with-aggressive-modes \
  --symbols-core "$SHADOW_SYMBOLS_CORE" \
  --symbols-volatile "$SHADOW_SYMBOLS_VOLATILE" \
  --symbols-defensive "$SHADOW_SYMBOLS_DEFENSIVE,$SHADOW_SYMBOLS_COMMOD_FX_INTL" \
  --dividend-symbols "$DIVIDEND_SYMBOLS" \
  --bond-symbols "$BOND_SYMBOLS"
```

## 2) Watchdog Health (one-shot)
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./.venv312/bin/python scripts/shadow_watchdog.py --once --watch-coinbase --watch-dividend --watch-bond --simulate-schwab --interval-seconds 30
```

## 3) Retrain (normal, after-hours gate enabled)
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./.venv312/bin/python scripts/weekly_retrain.py --continue-on-error
```

## 4) Retrain Manual Override (run during market hours)
Important: correct override variable is `RETRAIN_AFTER_HOURS_ONLY=0`.
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
RETRAIN_AFTER_HOURS_ONLY=0 ./.venv312/bin/python scripts/weekly_retrain.py --continue-on-error
```

## 5) Walk-Forward + Promotion Gate
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./.venv312/bin/python scripts/walk_forward_validate.py
./.venv312/bin/python scripts/walk_forward_promotion_gate.py
```

## 6) Raise Floor (canary gate enforced)
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
ACTIVE_STREAK_HARD_CAP=12 ./.venv312/bin/python scripts/run_master_bot.py --min-active-bots 35
./.venv312/bin/python scripts/bot_registry_status.py
```

## 7) Raise Floor (manual override if canary is blocked)
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
ACTIVE_STREAK_HARD_CAP=12 ./.venv312/bin/python scripts/run_master_bot.py --no-require-canary-gate --min-active-bots 35
./.venv312/bin/python scripts/bot_registry_status.py
```

## 8) SQL + One Numbers Refresh
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./.venv312/bin/python scripts/link_jsonl_to_sql.py --mode sqlite
./.venv312/bin/python scripts/sqlite_performance_maintenance.py
./.venv312/bin/python scripts/build_one_numbers_report.py --day "$(date -u +%Y%m%d)"
```

## 9) Allocator / Risk / Budget (portfolio control layer)
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./.venv312/bin/python scripts/sleeve_allocator.py
./.venv312/bin/python scripts/portfolio_risk_ledger.py
./.venv312/bin/python scripts/execution_budgeter.py
```

## 10) Executive Dashboard (Numbers-friendly)
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./.venv312/bin/python scripts/build_executive_dashboard.py --day "$(date -u +%Y%m%d)"
open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/executive_dashboard/latest.csv
```

## 11) Daily Full Pipeline
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/daily_log_refresh.sh
```

## 12) Lock Troubleshooting
Check running parallel shadows:
```bash
ps -axo pid,etime,command | grep "run_parallel_shadows.py" | grep -v grep
```

If lock says busy and process is stale:
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
kill <pid>
rm -f governance/parallel_shadow.lock
```

## 13) Common Gotchas
- If you see `No symbols provided`, your symbol env vars are empty. Re-export them before start.
- If retrain says market-open skip, use `RETRAIN_AFTER_HOURS_ONLY=0` for manual override.
- If `run_master_bot.py` says `promotion_gate_blocked`, either pass canary first or use `--no-require-canary-gate`.

## 14) Distillation-Enabled Retrain (teachers -> new bots)
Generate/refresh teacher-student plan:
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./.venv312/bin/python scripts/distill_new_bots.py
```

Run retrain with student priority + optional extra student passes:
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
RETRAIN_AFTER_HOURS_ONLY=0 RETRAIN_DISTILLATION_PRIORITY=1 RETRAIN_DISTILLATION_STUDENT_EXTRA_PASS=2 \
./.venv312/bin/python scripts/weekly_retrain.py --continue-on-error
```

## 15) One-Command Live Feed (seamless)
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh schwab-tail
```

NVDA-only:
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh schwab-tail --symbol NVDA
```

All feeds (Schwab + Coinbase):
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed --source all
```

## 16) Coinbase Paper Trading (new)
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh coinbase-start --paper --force-restart --top-n 2 --min-acc 0.55 --profiles default
```

Watch Coinbase paper/shadow feed:
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh coinbase-tail
```

