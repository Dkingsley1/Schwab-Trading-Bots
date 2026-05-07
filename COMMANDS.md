# Commands (Canonical)

Use these exact commands as the current source of truth.

This file is generated from the curated operator inventory in `scripts/ops/commands_hygiene_bot.py`.
Rebuild it with `./scripts/ops/opsctl.sh commands-hygiene --apply` after changing that inventory.
Command contract hash: `1c623766cbdacab30214e27ef541d399dc4eb7d9d7d2e5ee2198d8c8be5f2220`.
Command contract artifact: `governance/health/commands_contract_latest.json`.

This file is intentionally trimmed down with Most Used pinned first and the remaining sections alphabetized by section and command title:
- paper mode is the operating default
- no simulate variants are listed
- no duplicate restart commands are listed when a broader command already covers them

## Most Used

### Apply pressure relief controls
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh pressure-relief --apply --json
```

This writes the pressure-relief override used by runtime loading, maintenance guards, heavy feed TTL, SQL cadence, foreground-app awareness, macro capture niceness, MLX/quant caps, report caps, and quiet-window behavior.

### Attempt a safe global halt clear
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh global-halt-auto-clear --json
```

This only clears the halt when the runtime, auth, watchdog, and data-plane guardrails are back inside the safe-clear envelope.

### Broker Truth Step 1: refresh Schwab auth
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh token-refresh --always-auth
```

Use this first when broker-truth lanes start showing transient 403s or auth churn.

### Broker Truth Step 2: restart the Schwab loops
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed-refresh --source schwab
```

This forces the Schwab sleeves to pick up the refreshed token and republish their latest broker-truth snapshots.

### Broker Truth Step 3: verify broker readiness and lane statuses
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
/Users/dankingsley/PycharmProjects/schwab_trading_bot/.venv312/bin/python -c "from pathlib import Path; import json; root=Path('/Users/dankingsley/PycharmProjects/schwab_trading_bot/governance/health'); broker=json.loads((root/'broker_readiness_latest.json').read_text()); print(f'ready_for_open={broker.get(\"ready_for_open\")} auth_ok={broker.get(\"auth_ok\")} token_warning_level={broker.get(\"token_warning_level\")}'); print('lane,status,mismatch_count,error'); [print(f'{p.name.replace(\"broker_truth_\", \"\").replace(\"_latest.json\", \"\")},{json.loads(p.read_text()).get(\"status\", \"\")},{int(json.loads(p.read_text()).get(\"mismatch_count\", 0) or 0)},{json.loads(p.read_text()).get(\"error\") or \"\"}') for p in sorted(root.glob('broker_truth_*_latest.json')) if 'shared_snapshot' not in p.name]"
```

Healthy target: `ready_for_open=True`, `auth_ok=True`, and all Schwab broker-truth lanes reporting `status=ok` with `mismatch_count=0`.

### Clear all halt flags now
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh clear-all-halts --json
```

This clears both OPERATOR_STOP and GLOBAL_TRADING_HALT in one command. It is a manual collection-unblock override; it does not mark auth, snapshot recovery, or backpressure gates healthy.

### Emergency stop: engage operator stop and global halt
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh operator-control --engage --set-global-halt --reason operator_emergency_stop --json
```

Use this as the red-button stop when you want both the operator stop flag and the global trading halt set immediately.

### Fast read-only health check
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh health-fast --json
```

This reads the latest health artifacts without starting report refreshes, daily verification, or PDF/report jobs.

### Keep the Mac awake
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
caffeinate -dimsu
```

### Open the framework map PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh framework
```

This refreshes the framework-map source, renders a deterministic PDF, and falls back to HTML if the PDF is unavailable.

### Open the One Numbers CSV in Numbers
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/one_numbers/latest.csv
```

### Open the One Numbers PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/one_numbers/one_numbers_latest.pdf
```

### Open the special features PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh special
```

This refreshes the special-features PDF with the deterministic renderer, then opens it.

### Phone mirror view for the live feed
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh phone-feed --host 0.0.0.0 --source all --include-decisions
```

This starts the phone-friendly live feed mirror and prints the local and Tailscale URLs in the terminal.
When `--host 0.0.0.0` is used without `--token`, the server auto-generates a remote-access token for you.

### Refresh clearable global halt blockers
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh global-halt-refresh --json
```

This refreshes the watchdog, auth, data-plane, and runtime-clearance blocker artifacts, then re-evaluates what still prevents a safe clear. It will not release OPERATOR_STOP for you.

### Refresh the live loops without reinstalling the stack watchdog
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh livefeed-refresh
```

`livefeed-refresh` is the all-feeds shortcut for the `feed-refresh` live-loop restart helper. It kills and restarts the relevant market-data loops. Use `./scripts/ops/opsctl.sh livefeed-refresh --dry-run` to validate the route without touching processes. If you want a full supervised stack refresh instead of a feed-loop refresh, use `./scripts/ops/opsctl.sh start --force-restart`.

### Refresh the special features and framework map reports
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh showcase-refresh
./scripts/ops/opsctl.sh system-explainers
./scripts/ops/opsctl.sh report-pdfs --json
```

Use this when you want the latest special-features packet and framework-map report regenerated together.

### Release operator stop only
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh operator-release --json
```

This releases the manual OPERATOR_STOP flag without bypassing the global halt safe-clear checks.

### Run post-restart settlement
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh post-restart-settle --apply --json
```

This rechecks restart sanity, auth lease, global halt blockers, collector contracts, process watchdog coverage, and runtime throttle after a restart.

### Runtime mode switchboard
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
PY="$(zsh ./scripts/ops/runtime_python.sh)"
SWITCHBOARD_MODES="shadow,paper" "$PY" scripts/run_mode_switchboard.py
```

Valid modes are `shadow`, `paper`, and `live`.
This launches one `main.py` child per mode and sets `BOT_MODE` automatically.

### Show global halt status and blockers
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh global-halt-status --json
```

This prints the current global halt posture, any active halt reasons, and the blockers that still prevent a safe clear.

### Start the full live stack
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh start
```

Use this for the normal supervised start path when the stack is already healthy or only lightly stale.

### Start the full live stack (fresh supervised restart)
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh start --force-restart
```

Use this after stale paper lanes, restart storms, or auth recovery so the running stack is rebuilt cleanly.

### Stop the stack
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh stop
```

This is the normal supervised stop path. It does not automatically engage an emergency operator halt.

### Validate documented commands
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh command-validity --json
```

## Data Context Syncs

### Crypto market context sync
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh crypto-market-sync --json
```

### FX market context sync
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh fx-market-sync --json
```

### Macro context sync
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh macro-context-sync --json
```

### Options flow context sync
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh options-flow-sync --json
```

`options-flow-sync` is the canonical command. `tastytrade-sync` remains a legacy alias for backward compatibility.

### Source verification
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh source-verification --json
```

### Stock / crypto correlation sync
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh market-correlation-sync --json
```

## Live Feed Refreshes

### Refresh all live feeds
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh livefeed-refresh
```

### Refresh Coinbase spot and Coinbase futures
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed-refresh --source coinbase
```

### Refresh FX only
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed-refresh --source fx
```

### Refresh Schwab equities, Schwab futures, and FX
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed-refresh --source schwab
```

## Live Feed Views

### Heavy Coinbase live feed view
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed --source coinbase --heavy
```

### Heavy futures live feed view
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed --source futures --heavy
```

### Heavy FX live feed view
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed --source fx --heavy
```

### Heavy infrastructure live feed view
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed --source infra --heavy --lines 160
```

### Heavy live feed view across all sections
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed --source all --heavy
```

Use this as the primary all-feeds operator view when you want sleeve logs, decision streams, highlighted health states, and infrastructure health artifacts in one window.
Heavy views use a red-only highlight by default while preserving `[ALERT]`, `[WATCH]`, `[OK]`, and `[FLOW]` labels; pass `--no-color` only when redirecting clean text to a file.
If the Mac is running an `air_safe` or `constrained` memory-efficiency profile, the feed automatically trims decision fanout and uses a lower default line budget unless you pass your own `--lines` or `--no-memory-aware`.

### Heavy main live feed view
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed --source main --heavy
```

### Heavy Schwab live feed view
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed --source schwab --heavy
```

### Light live feed tail for all feeds
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed --source all --lines 80
```

### Live feed tail for all futures sleeves
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh futures-tail --lines 80
```

### Live feed tail for Coinbase
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh coinbase-tail --lines 80
```

### Live feed tail for Coinbase futures
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh coinbase-futures-tail --lines 80
```

### Live feed tail for FX
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh fx-tail --lines 80
```

### Live feed tail for Schwab
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh schwab-tail --lines 80
```

### Live feed tail for Schwab futures
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh schwab-futures-tail --lines 80
```

### Live feed tail for Schwab, Coinbase, and futures
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh main-tail --lines 80
```

## Macro And Media

### Show macro auto-watch status
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh macro-auto-status --json
```

### Start the macro auto-watch lane
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh macro-auto-start --force-restart --youtube-channel-url "https://www.youtube.com/@federalreserve" --template fed --speaker "Federal Reserve" --source "Federal Reserve"
```

## Platform Expansion

### Apply Platform Brain v4 Grande
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh platform-brain-v4 --apply --json
```

Adds the decision-brain layer: executive meta-orchestration, causal world modeling, experience memory v2, expansion simulation, priority ranking, self-upgrade planning, critic council, outcome verification, bot economics, data value scoring, training scheduling, and operator intent modeling.
The layer is advisory/read-only, keeps MLX as default, and preserves paper-trade lock and live-execution separation.

### Apply Platform Brain v5 Reflex Cortex
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh platform-brain-v5 --apply --json
```

Adds the reflex layer: temporal self-modeling, safe reflex routing, regret and outcome ledgering, scenario rehearsal, adaptive cadence, safe autonomy boundary, critic fusion, resource budgeting, data contract negotiation, bot curriculum, dependency mapping, and strategic roadmap synthesis.
The layer stays advisory/read-only, keeps MLX as default, and preserves paper-trade lock with live execution disabled.

### Apply the 12-layer platform intelligence control plane
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh platform-intelligence --apply --json
```

Adds the operational layer for bot lifecycle, data quality scoring, provider failover, backpressure prediction, duplicate-alpha detection, paper capacity, self-healing playbooks, sleeve masters, training readiness, regime routing, execution realism, and black-box recording.
This writes the platform-intelligence override while keeping the layer advisory/read-only and MLX-first.

### Apply the 14-muscle trading action systems pack
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh trading-muscles --apply --json
```

Adds 70 collect-only trading muscle bots across intraday momentum, mean reversion, swing trend, options convexity, options income, futures macro, crypto basis, volatility arbitrage, events, relative value, portfolio hedging, execution timing, position sizing, and exits/rebalancing.
The pack generates trade-candidate, sizing, hedge, exit, and execution-rehearsal evidence only; training, paper trading, live trading, allocation, and execution stay blocked until 180 days, 75000 observations, and quality/risk/halts gates clear.

### Apply the 14-organ platform organ systems pack
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh platform-organs --apply --json
```

Adds 70 collect-only platform organ bots across data quality, feature registry, replay lab, execution realism, portfolio brain, alpha decay, regime routing, research assimilation, promotion court, cockpit, resource metabolism, memory lymphatics, backpressure circulation, and audit immunity.
The pack stays zero-weight, training-excluded, paper-disabled, live-disabled, and thin-sampled until 150 days, 60000 observations, and runtime/regression/operator gates clear.

### Apply the 24-sleeve quant strategy gap pack
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh quant-strategy-gap --apply --json
```

Adds 24 practical tradable-alpha strategy sleeves with five collect-only bots each: event arb, relative value, carry, ETF/NAV, auction flow, dealer expiry, rates/credit/crypto basis, and liquidity simulation.
The pack is zero-weight, training-excluded, paper-disabled, live-disabled, and thin-sampled until 120 days, 45000 observations, and strategy evidence gates clear.

### Apply the alpha intelligence evolution pack
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh alpha-intelligence-evolution --apply --json
```

Adds the guarded alpha advancement layer: training readiness, execution reality, portfolio exposure, source confidence, research intake, duplicate-alpha novelty control, regime memory v2, dashboard v2, adapter mesh, and cleanup governor.
The bots are collection-only with paper/live execution blocked until readiness, execution realism, source confidence, and duplicate-alpha gates clear.

### Apply the apex self-awareness intelligence pack
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh apex-self-awareness-intelligence --apply --json
```

Adds the 46 guarded apex bots that bring the platform to 1000 total bots: deep self-modeling, meta-reasoning, experience memory, scenario oracles, upgrade foundry, causal alpha safety, resource autonomy, operator copilot, Grand Master collective intelligence, and research frontier scouting.
The bots are collection-only with live execution, allocation, and training blocked until 120 days, 30000 observations, and safety/resource/memory gates clear.

### Apply the coordination intelligence control-plane pack
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh coordination-intelligence --apply --json
```

Adds the guarded coordination layer: bot genome lineage, strategy conflict resolution, capital simulation, regime memory, research-to-bot intake, feature quality, adversarial paper-trade lab, sleeve master summaries, admission committee, and explainability dashboard.
The bots are collection-only until their evidence, data-quality, and runtime thresholds clear.

### Apply the deep recursive awareness pack
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh deep-recursive-awareness --apply --json
```

Adds the guarded recursive awareness layer: causal self-diagnosis, predictive runtime oracle, experience memory core, self-upgrade critic board, operator context governor, internal critic board, and living platform map.
The bots are collection-only with live execution, allocation, and training blocked until 150 days, 36000 observations, and causal/runtime/memory/critic/operator-context gates clear.

### Apply the intelligence layer advancement pack
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh intelligence-layer-advancement --apply --json
```

Adds the guarded meta-intelligence layer: metacognitive routing, counterfactual world models, alpha benchmarks, memory compression, critic debate, active learning, ensemble uncertainty, library routing, safety invariants, and self-improvement backlog planning.
The bots are collection-only with paper/live execution blocked until benchmark, memory-quality, safety-invariant, and runtime-pressure gates clear.

### Apply the settlement stabilization layer
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh platform-settlement-stabilization --apply --json
```

Adds the post-expansion settlement layer for queue decay, single-writer protection, market-hours cadence, global-halt clear readiness, paper collection floors, off-hours drain planning, and stabilization memory.
This layer keeps MLX as default, leaves live execution disabled, and records whether each stabilization pass actually reduces pressure.

### Apply the seven-part stabilization and quality layer
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh platform-stabilization --apply --json
```

Adds the guarded pre-expansion layer for backlog drainage, bot data quality, duplicate-alpha compression, paper-trade realism, provider cooldown/failover, ready-only microtraining, and expansion rehearsal.
This writes the stabilization override, assigns infrastructure bots to each lane, keeps MLX as default, and keeps live execution disabled.

### Arm the guarded 400 bot paper ramp
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh paper-400-ramp --apply --json
```

Plans the 400-bot paper target now and only writes the high paper caps after Monday 2026-05-11 when global halt, memory, runtime, and ingestion gates are clean.
The controller keeps live execution blocked, paper-trade lock enabled, and explains any blocker in `governance/health/paper_400_ramp_latest.json`.

### Preview Platform Brain v4 Grande
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh platform-brain-v4 --json
```

Use this to inspect the 12 brain sections, next-best command, ranked priorities, expansion simulations, and verification plan without writing overrides.

### Preview Platform Brain v5 Reflex Cortex
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh platform-brain-v5 --json
```

Use this to inspect the reflex queue, safe-vs-operator-reviewed actions, scenario rehearsal, resource budgets, and roadmap without writing overrides.

### Preview the 12-layer platform intelligence control plane
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh platform-intelligence --json
```

Use this to inspect all 12 platform sections and their latest artifacts without changing runtime overrides.

### Preview the 14-muscle trading action systems pack
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh trading-muscles --json
```

Use this dry run to inspect the trading muscles, bot IDs, storage guardrails, and trade-candidate-only floor without changing the registry.

### Preview the 14-organ platform organ systems pack
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh platform-organs --json
```

Use this dry run to inspect the organ systems, bot IDs, storage guardrails, and paper-only floors without changing the registry.

### Preview the 24-sleeve quant strategy gap pack
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh quant-strategy-gap --json
```

Use this dry run to inspect planned strategy sleeves, bot IDs, storage guardrails, and paper-only floors without changing the registry.

### Preview the alpha intelligence evolution pack
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh alpha-intelligence-evolution --json
```

Use the dry run to see planned alpha intelligence bots, data intakes, storage contracts, and self-awareness upgrades without changing the registry.

### Preview the apex self-awareness intelligence pack
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh apex-self-awareness-intelligence --json
```

Use the dry run to see the 46 planned apex bots, 1000-bot target contract, storage limits, and paper-only guardrails without changing the registry.

### Preview the coordination intelligence control-plane pack
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh coordination-intelligence --json
```

Use the dry run to see planned coordination bots, storage contracts, and paper-only guardrails without changing the registry.

### Preview the deep recursive awareness pack
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh deep-recursive-awareness --json
```

Use the dry run to see the 28 planned recursive-awareness bots, data intakes, storage limits, and paper-only guardrails without changing the registry.

### Preview the guarded 400 bot paper ramp
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh paper-400-ramp --json
```

Use this to see whether the ramp is planned, armed, or blocked before changing runtime overrides.

### Preview the intelligence layer advancement pack
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh intelligence-layer-advancement --json
```

Use the dry run to see planned intelligence-layer bots, data intakes, storage contracts, and routing/safety guardrails without changing the registry.

### Preview the settlement stabilization layer
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh platform-settlement-stabilization --json
```

Use this after a large expansion or during market hours to see whether the system is settling cleanly before adding more bots or training load.

### Preview the seven-part stabilization and quality layer
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh platform-stabilization --json
```

Use this before another expansion to see the backlog, quality, duplicate-alpha, paper realism, provider, training, and rehearsal gates in one place.

## Reports And PDFs

This section includes the generate commands plus direct open commands for each report PDF.

### Active bot stack PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh botstack
```

Latest PDF path: `/Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/bot_stack_status/latest.pdf`.
This refreshes the bot-stack source and rebuilds the PDF through the deterministic send-out renderer.

### Incident report
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh incident
```

Latest PDF path: `/Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/reports/incident_report_latest.pdf`.
This refreshes the incident source and rebuilds the PDF through the deterministic send-out renderer.

### Incident review packet PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh incident-packet
```

Latest PDF path: `/Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/reports/incident_review_packet_latest.pdf`.
This writes the immutable incident review packet JSON and rebuilds its PDF companion through the deterministic send-out renderer.

### Install nightly showcase and PDF refresh
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/install_daily_log_refresh_launchd.sh
```

This installs the macOS launchd job that refreshes showcase docs, system explainers, and PDFs automatically each night.

### One Numbers report
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
PY="$(zsh ./scripts/ops/runtime_python.sh)"
"$PY" scripts/build_one_numbers_report.py
```

### Open the active bot stack PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh botstack
```

This refreshes the bot stack report, prefers the PDF artifact, and falls back to HTML or markdown if the PDF renderer is unavailable.

### Open the bot explainability PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh explainability
```

This regenerates bot explainability evidence, renders the report PDF bundle, prefers the PDF artifact, and falls back to JSON evidence if the PDF renderer is unavailable.

### Open the crash digest PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh crash
```

This regenerates the crash digest with a 30-day lookback by default, prefers the PDF artifact, and falls back to printable HTML if the PDF renderer is unavailable.

### Open the daily auto verify PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh daily-auto-verify
```

Latest PDF path: `/Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/sql_reports/daily_auto_verify_latest.pdf`.
This regenerates daily auto verify, renders the report PDF bundle, prefers the PDF artifact, and falls back to JSON evidence if the PDF is unavailable.

### Open the daily ops PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/reports/daily_ops_report_latest.pdf
```

### Open the daily runtime summary PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/sql_reports/daily_runtime_summary_latest.pdf
```

### Open the expansion inventory PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh expansions
```

Latest PDF path: `/Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/reports/expansion_inventory/expansion_inventory_latest.pdf`.
This regenerates the expansion list from registry-backed packs and control-plane config files, then opens the report-ready PDF.

### Open the incident report PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh incident
```

This refreshes the decision-oriented incident report, prefers the PDF artifact, and falls back to HTML or markdown if the PDF renderer is unavailable.

### Open the incident review packet PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh incident-packet
```

This refreshes the immutable incident review packet and opens its compact PDF companion, falling back to the JSON packet if needed.

### Open the macro crosscheck PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh macro
```

This regenerates the macro crosscheck source, renders the report PDF bundle, prefers the PDF artifact, and falls back to markdown if the PDF renderer is unavailable.

### Open the market correlation PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh correlation
```

This renders the report PDF bundle, prefers the market-correlation PDF artifact, and falls back to markdown if the PDF renderer is unavailable.

### Open the model card PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh modelcard
```

Latest PDF path: `/Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/sql_reports/model_card_latest.pdf`.
This renders the report PDF bundle, prefers the model card PDF, and falls back to JSON evidence if the PDF is unavailable.

### Open the paper execution calibration PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh calibration
```

Latest PDF path: `/Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/sql_reports/paper_execution_calibration_latest.pdf`.
This renders the report PDF bundle, prefers the paper execution calibration PDF, and falls back to JSON evidence if the PDF is unavailable.

### Open the paper performance PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh paper
```

This refreshes paper-performance data without the GUI renderer, then opens the report-ready chart PDF with daily, weekly, window-change, and sleeve-scoreboard views.

### Open the post-trade analysis PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh posttrade
```

This refreshes post-trade data with timeout/cached-artifact fallbacks, then opens the report-ready PDF with assessment, calibration, runtime, softguard, and source notes.

### Open the project timeline PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh timeline
```

This regenerates the timeline report, prefers the PDF artifact, and falls back to printable HTML if the PDF renderer is unavailable.

### Open the quant model control PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh quant-model-control --json
open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/reports/quant_model_control/quant_model_control_latest.pdf
```

Latest PDF path: `/Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/reports/quant_model_control/quant_model_control_latest.pdf`.
This refreshes the advanced quant-model feature, MLX, resource-cap, and research-only policy report.

### Open the replay feature ablation PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh replay
```

This regenerates the replay feature ablation evidence, renders the report PDF bundle, prefers the PDF artifact, and falls back to the latest JSON evidence if a PDF cannot be rendered.

### Open the report catalog PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/reports/report_pdf_bundle_latest.pdf
```

### Open the retrain scorecard PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/sql_reports/retrain_scorecard_latest.pdf
```

### Open the sentiment PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh sentiment
```

This regenerates the current sentiment report, prefers the PDF artifact, and falls back to HTML or markdown if the PDF renderer is unavailable.

### Open the source verification PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh source
```

This regenerates source verification, renders the report PDF bundle, prefers the PDF artifact, and falls back to markdown if the PDF renderer is unavailable.

### Open the state snapshot drills PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/state_snapshot_drills/state_snapshot_drills_latest.pdf
```

### Open the strategy attribution PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/reports/strategy_attribution_latest.pdf
```

### Open the strategy inventory PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh strategy-inventory
```

Latest PDF path: `/Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/reports/strategy_inventory/strategy_inventory_latest.pdf`.
This regenerates the complete sleeve/strategy inventory from the system config, renders the PDF bundle, and opens the report-ready PDF.

### Open the system overview PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
open /Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/reports/system_overview/system_overview_weekly_platform_history_latest.pdf
```

This opens the week-by-week platform history and current-position overview PDF.

### Open the training report PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh training
```

This regenerates the training report, prefers the PDF artifact, and falls back to printable HTML or markdown if the PDF renderer is unavailable.

### Open the unified lane scorecard PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh unified
```

This regenerates the unified lane scorecard, renders the report PDF bundle, prefers the PDF artifact, and falls back to markdown if the PDF renderer is unavailable.

### Paper performance report
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh paper
```

This refreshes the paper-performance source and opens the report-ready chart PDF.

### Refresh showcase, framework map, and PDFs now
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh showcase-refresh
./scripts/ops/opsctl.sh system-explainers
./scripts/ops/opsctl.sh report-pdfs --json
```

This is the paste-ready deterministic PDF refresh path when you want the special-features PDF and the framework-map-style reports regenerated together.

### Repair and validate report PDFs
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh report-quality-guard --repair --json
```

This is the report infrabot pass for external sendouts: it rebuilds PDFs, checks header/EOF/size integrity, and verifies report-ready renderers for upgraded reports.

### Report catalog bundle
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh report-pdfs --json
```

## Retrain

Use these commands when you are preparing or launching a manual retrain cycle.

### Force full retrain (bypass prechecks)
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh retrain-force-full
```

Use this only when you intentionally want to bypass the normal data-quality, freshness, snapshot-sync, and sample-quota prechecks.

### Full retrain preflight
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/daily_log_refresh.sh
./scripts/ops/opsctl.sh runtime-training-snapshot --json
./scripts/ops/opsctl.sh coverage-seed --write-queue --json
./scripts/ops/opsctl.sh coverage-gap-closer --apply-stage --launch --json
PY="$(zsh ./scripts/ops/runtime_python.sh)"
"$PY" scripts/retrain_schema_compatibility_guard.py --json
"$PY" scripts/promotion_quality_gate.py --json
```

Run this before a manual full retrain so SQL state, runtime snapshots, coverage, and promotion gates are fresh.

### Guarded retrain orchestrator
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh retrain-orchestrate --json
```

This is the safer manual retrain entrypoint because it refreshes stale artifacts and honors freshness checks before launching weekly retrain.

### Training and labeling intelligence
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh training-labeling-intelligence --apply --json
```

Normalizes label contracts, writes training-process intelligence, and keeps targeted retrain candidates behind schema, feature-store, coverage, runtime, and lineage gates.

## Schwab Auth

Use these exact Schwab authorization commands when tokens expire, browser consent needs renewal, callback ports get stuck, or broker-truth lanes start surfacing 401/403 errors.

### Interactive Schwab authorization re-consent
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh token-refresh-interactive --force --prompt-before-browser --json
```

Run this when you need to update the browser handshake after changing credentials, renewing consent, or clearing stale callback/token state.

### Schwab auth recovery plus lane restart
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh token-refresh --always-auth
./scripts/ops/opsctl.sh feed-refresh --source schwab
```

This is the paste-ready recovery pair when refreshed authorization needs to be picked up by the Schwab loops immediately.

### Schwab auth supervisor
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh schwab-auth-supervisor --json
./scripts/ops/opsctl.sh schwab-auth-supervisor --apply --json
```

Use this first when Schwab auth looks freshly authorized but the system still reports token, callback-port, or browser OAuth drift.
The apply form cleans up stale Schwab auth helper processes and refreshes the token/lease artifacts without opening a browser loop.

### Schwab authorization refresh
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh token-refresh --always-auth
```

Use this when the Schwab browser grant is stale or broker-truth lanes start showing auth churn.

## SQL And Reports

### Data quality refresh bundle
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh livefeed-refresh
./scripts/daily_log_refresh.sh
PY="$(zsh ./scripts/ops/runtime_python.sh)"
"$PY" scripts/build_one_numbers_report.py
```

Use this when One Numbers is stale or you want the latest data-quality averages and report artifacts refreshed together.

### Full SQL refresh pipeline
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/daily_log_refresh.sh
```

Use this when you want the full SQL/log/report refresh instead of the one-pass writer sync.

### Quick SQL sync
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh sql-sync
```

## Status And Health

### Coinbase API health
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh coinbase-api-health --json
```

This checks Coinbase public market-data endpoints and reports only credential presence booleans, never secret values.

### Docs, commands, and reporting intelligence
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh docs-reporting-intelligence --apply --json
```

This refreshes the README, COMMANDS.md, report-quality, and PyCharm visibility intelligence layer, including blue active-bot rows in `docs/pycharm/intelligence_layers_latest.md`.

### Doctor
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh doctor
```

### Golden replay regression guard
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh golden-replay-regression --json
```

This compares deterministic replay against the golden replay pack or the seeded replay hash fallback.

### Health snapshot
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh health
```

### Master infrastructure supervisor
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh master-infra-supervisor --json
```

This parent check watches child infrastructure bots, command routes, storage health, report jobs, and One Numbers original-start coverage as one dependency graph.

### Plan or apply the MLX library upgrade bundle
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh mlx-library-upgrade --json
./scripts/ops/opsctl.sh mlx-library-upgrade --apply --json
```

The dry run prints the pinned MLX package bundle from `config/requirements.lock.txt`; the apply form installs those pins, then you should run `./scripts/ops/opsctl.sh mlx-audit --json`.

### Point-in-time event store
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh point-in-time-event-store --json
```

This rebuilds the normalized event store used to prove source state at replay and report time.

### PyCharm active bot blue highlights
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh pycharm-active-bot-highlights --apply --json
```

This writes the JetBrains `Active Bots` scope and blue file-color mapping so active `core/brain_refinery_*.py` files get a durable Project-pane scope background. PyCharm's bright blue filename text remains reserved for VCS-modified files.

### Refresh runtime dashboard contracts
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh dashboard-refresh
```

This hydrates the runtime gate dashboard prerequisites first so missing sections become explicit health outputs instead of silent omissions.

### Repair safe cross-system drift surfaces
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh system-drift-autopilot --apply --json
```

This runs the safe drift-repair mesh. It refreshes and repairs repairable surfaces without inventing destructive operator actions.

### Replay hash registry guard
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh replay-hash-registry --json
```

This persists expected replay hashes and alerts when deterministic replay output drifts.

### Reporter quality infrabot
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh report-quality-guard --repair --json
```

This repairs the sendout PDF bundle, verifies PDF integrity, and blocks regressions where paper-performance or post-trade lose their report-ready renderers.

### Review Codex project guardrails
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh codex-project-guard --staged --json
```

Run this before Codex-authored commits or GitHub updates to catch source-of-truth drift, mixed-domain staging, and separate-domain README/docs leakage.

### Review the cross-system drift mesh
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh system-drift-guard --json
```

This rolls command drift, summary/report drift, governance drift, workstation drift, and stack-runtime drift into one registry-backed health view.

### Runtime gate dashboard
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh dashboard
```

By default this now runs a runtime-artifact refresh pass first. Use `./scripts/ops/opsctl.sh dashboard --skip-refresh` when you want a pure read of the current artifact set.

### Runtime status
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh status
```

## Storage

### Repair local stateful storage regressions
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh stateful-storage-regression-guard --apply --json
```

This guard keeps SQL shards, execution-lane telemetry, and SQL writer launchd logs routed away from the internal disk.

### Review or prune eligible local standby SQLite copies after BOT_LOGS soak
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh storage-prune-standby --json
```

This is a dry run by default. Add `--apply` after the external route has soaked long enough to prune only the verified standby copies, or add `--include-curated-standby` if you intentionally want to touch curated standby paths too.

### Run the storage disaster recovery bot
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh storage-disaster-recovery --apply --json
```

### Safe force-clear storage pressure supervisor
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh storage-pressure-clearance --apply --force-clear-stale-gate --json
```

This is the parent storage pressure bot. It forces safe refresh/checkpoint/drain actions, but only clears stale storage gates after live WAL and backlog metrics are inside the safe envelope.

### Safe-eject the external BOT_LOGS drive
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh storage-safe-eject
```

### Switch collection back to the external BOT_LOGS drive
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh storage-switch-external
```

### Switch collection to the Mac's internal drive
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh storage-switch-local
```
