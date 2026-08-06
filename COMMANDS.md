# Commands (Canonical)

Use these exact commands as the current source of truth.

This file is generated from the curated operator inventory in `scripts/ops/commands_hygiene_bot.py`.
Rebuild it with `./scripts/ops/opsctl.sh commands-hygiene --apply` after changing that inventory.
Command contract hash: `b344bfd6855f72af80eb92285e6c572e96a9b886ca1f89b7ad79709389c67352`.
Command contract artifact: `governance/health/commands_contract_latest.json`.

This file is intentionally trimmed down with Most Used pinned first and the remaining sections alphabetized by section and command title:
- paper mode is the operating default
- no simulate variants are listed
- no duplicate restart commands are listed when a broader command already covers them
- passive automation installers and expansion-pack reference commands are kept out of the operator-facing list

**Search Bar**

<input type="search" list="command-search-index-options" placeholder="PyCharm: press Command+F or Ctrl+F, then search any command, section, opsctl alias, or script path" style="width: 100%; padding: 8px;" />

PyCharm note: the field above is a visible search landing strip in Markdown preview; the reliable editor search is `Command+F` on Mac or `Ctrl+F` elsewhere.

Fast search tokens: `start` `stop` `paper` `profitability` `soak` `halt` `auth` `schwab` `coinbase` `livefeed` `storage` `dashboard` `runtime` `watchdog` `backlog` `retrain` `reports` `startup` `login` `notification` `most-used` `accounts-and-positions` `data-context-syncs` `event-watches` `live-feed-views` `notifications-and-alerts` `paper-trading` `reports-and-pdfs`.

Useful compound searches: `paper profitability`, `global halt`, `token refresh`, `livefeed heavy`, `storage prune`, `soak readiness`.

Search coverage: `163` generated command entries from the current command contract.

<datalist id="command-search-index-options">
  <option value="Keep the Mac awake (Most Used)"></option>
  <option value="Start the full live stack (Most Used)"></option>
  <option value="Start the full live stack (fresh supervised restart) (Most Used)"></option>
  <option value="Stop the stack (Most Used)"></option>
  <option value="Apply autonomic P-core resource governor (Most Used)"></option>
  <option value="Apply backlog writer catch-up waves (Most Used)"></option>
  <option value="Apply income operating platform controls (Most Used)"></option>
  <option value="Apply memory pressure and multitasking controls (Most Used)"></option>
  <option value="Apply pressure relief controls (Most Used)"></option>
  <option value="Apply raw backlog refinement (Most Used)"></option>
  <option value="Apply runtime throttle and P-core priority controls (Most Used)"></option>
  <option value="Ask what backlog and runtime need next (Most Used)"></option>
  <option value="Attempt a safe global halt clear (Most Used)"></option>
  <option value="Broker Truth Step 1: refresh Schwab auth (Most Used)"></option>
  <option value="Broker Truth Step 2: restart the Schwab loops (Most Used)"></option>
  <option value="Broker Truth Step 3: verify broker readiness and lane statuses (Most Used)"></option>
  <option value="Build the paper evidence packet (Most Used)"></option>
  <option value="Check 12-lane system expansion execution (Most Used)"></option>
  <option value="Check backlog writer and drainer status (Most Used)"></option>
  <option value="Check capital rotation control (Most Used)"></option>
  <option value="Check Schwab indicator intelligence (Most Used)"></option>
  <option value="Check support maintenance yield gate (Most Used)"></option>
  <option value="Clear all halt flags now (Most Used)"></option>
  <option value="Emergency stop: engage operator stop and global halt (Most Used)"></option>
  <option value="Fast read-only health check (Most Used)"></option>
  <option value="Open the framework map PDF (Most Used)"></option>
  <option value="Open the One Numbers CSV in Numbers (Most Used)"></option>
  <option value="Open the One Numbers PDF (Most Used)"></option>
  <option value="Open the special features PDF (Most Used)"></option>
  <option value="Phone mirror view for the live feed (Most Used)"></option>
  <option value="Refresh clearable global halt blockers (Most Used)"></option>
  <option value="Refresh the livefeed mirror without restarting sleeves (Most Used)"></option>
  <option value="Refresh the special features and framework map reports (Most Used)"></option>
  <option value="Release operator stop only (Most Used)"></option>
  <option value="Repair and restart the livefeed mirror (Most Used)"></option>
  <option value="Run adversarial system drills (Most Used)"></option>
  <option value="Run intense system drills (Most Used)"></option>
  <option value="Run post-restart settlement (Most Used)"></option>
  <option value="Run the architecture upgrade scoreboard (Most Used)"></option>
  <option value="Runtime mode switchboard (Most Used)"></option>
  <option value="Show global halt status and blockers (Most Used)"></option>
  <option value="Validate documented commands (Most Used)"></option>
  <option value="Watch P-core/E-core load live/heavy (Most Used)"></option>
  <option value="Watch P-core/E-core load with low overhead (Most Used)"></option>
  <option value="Refresh Schwab account positions (Accounts And Positions)"></option>
  <option value="Review account policy context (Accounts And Positions)"></option>
  <option value="Study all visible account positions (Accounts And Positions)"></option>
  <option value="Watch covered-call roll windows (Accounts And Positions)"></option>
  <option value="Crypto market context sync (Data Context Syncs)"></option>
  <option value="FX market context sync (Data Context Syncs)"></option>
  <option value="Macro context sync (Data Context Syncs)"></option>
  <option value="Options flow context sync (Data Context Syncs)"></option>
  <option value="Source verification (Data Context Syncs)"></option>
  <option value="Stock / crypto correlation sync (Data Context Syncs)"></option>
  <option value="Install the SpaceX/SPCX IPO downside watcher (Event Watches)"></option>
  <option value="Run macro event intelligence (Event Watches)"></option>
  <option value="Run the SpaceX/SPCX downside watch once (Event Watches)"></option>
  <option value="Heavy live feed with file diagnostics (Live Feed Views)"></option>
  <option value="Heavy operator livefeed view (Live Feed Views)"></option>
  <option value="Dry-run the startup Yes/No bot start prompt (Notifications And Alerts)"></option>
  <option value="Install the startup Yes/No bot start prompt (Notifications And Alerts)"></option>
  <option value="Review remote alert control (Notifications And Alerts)"></option>
  <option value="Send a test iMessage notification (Notifications And Alerts)"></option>
  <option value="Start the Mac notification and iMessage watcher (Notifications And Alerts)"></option>
  <option value="Stop the notification watcher (Notifications And Alerts)"></option>
  <option value="Apply paper profitability controls (Paper Trading)"></option>
  <option value="Apply the paper live-data standard (Paper Trading)"></option>
  <option value="Arm or candidate-promote the guarded 400 bot paper ramp (Paper Trading)"></option>
  <option value="Check paper runtime regression guard (Paper Trading)"></option>
  <option value="Review guarded 400 bot paper ramp (Paper Trading)"></option>
  <option value="Active bot stack PDF (Reports And PDFs)"></option>
  <option value="Incident report (Reports And PDFs)"></option>
  <option value="Incident review packet PDF (Reports And PDFs)"></option>
  <option value="One Numbers report (Reports And PDFs)"></option>
  <option value="Open the active bot stack PDF (Reports And PDFs)"></option>
  <option value="Open the bot explainability PDF (Reports And PDFs)"></option>
  <option value="Open the crash digest PDF (Reports And PDFs)"></option>
  <option value="Open the daily auto verify PDF (Reports And PDFs)"></option>
  <option value="Open the daily ops PDF (Reports And PDFs)"></option>
  <option value="Open the daily runtime summary PDF (Reports And PDFs)"></option>
  <option value="Open the expansion inventory PDF (Reports And PDFs)"></option>
  <option value="Open the incident report PDF (Reports And PDFs)"></option>
  <option value="Open the incident review packet PDF (Reports And PDFs)"></option>
  <option value="Open the macro crosscheck PDF (Reports And PDFs)"></option>
  <option value="Open the market correlation PDF (Reports And PDFs)"></option>
  <option value="Open the model card PDF (Reports And PDFs)"></option>
  <option value="Open the paper execution calibration PDF (Reports And PDFs)"></option>
  <option value="Open the paper performance PDF (Reports And PDFs)"></option>
  <option value="Open the post-trade analysis PDF (Reports And PDFs)"></option>
  <option value="Open the project timeline PDF (Reports And PDFs)"></option>
  <option value="Open the quant model control PDF (Reports And PDFs)"></option>
  <option value="Open the replay feature ablation PDF (Reports And PDFs)"></option>
  <option value="Open the report catalog PDF (Reports And PDFs)"></option>
  <option value="Open the retrain scorecard PDF (Reports And PDFs)"></option>
  <option value="Open the sentiment PDF (Reports And PDFs)"></option>
  <option value="Open the source verification PDF (Reports And PDFs)"></option>
  <option value="Open the state snapshot drills PDF (Reports And PDFs)"></option>
  <option value="Open the strategy attribution PDF (Reports And PDFs)"></option>
  <option value="Open the strategy inventory PDF (Reports And PDFs)"></option>
  <option value="Open the system overview PDF (Reports And PDFs)"></option>
  <option value="Open the training report PDF (Reports And PDFs)"></option>
  <option value="Open the unified lane scorecard PDF (Reports And PDFs)"></option>
  <option value="Paper performance report (Reports And PDFs)"></option>
  <option value="Refresh showcase, framework map, and PDFs now (Reports And PDFs)"></option>
  <option value="Repair and validate report PDFs (Reports And PDFs)"></option>
  <option value="Report catalog bundle (Reports And PDFs)"></option>
  <option value="Force full retrain (bypass prechecks) (Retrain)"></option>
  <option value="Full retrain preflight (Retrain)"></option>
  <option value="Guarded retrain orchestrator (Retrain)"></option>
  <option value="Training and labeling intelligence (Retrain)"></option>
  <option value="Interactive Schwab authorization re-consent (Schwab Auth)"></option>
  <option value="Local Schwab credential setup (Schwab Auth)"></option>
  <option value="Schwab auth recovery plus lane restart (Schwab Auth)"></option>
  <option value="Schwab auth supervisor (Schwab Auth)"></option>
  <option value="Schwab authorization refresh (Schwab Auth)"></option>
  <option value="Data quality refresh bundle (SQL And Reports)"></option>
  <option value="Full SQL refresh pipeline (SQL And Reports)"></option>
  <option value="Quick SQL sync (SQL And Reports)"></option>
  <option value="Acquire independent fill evidence (Status And Health)"></option>
  <option value="Adapt infrabots to current system needs (Status And Health)"></option>
  <option value="Advance staged promotion candidates (Status And Health)"></option>
  <option value="Apply system architecture hardening (Status And Health)"></option>
  <option value="Coinbase API health (Status And Health)"></option>
  <option value="Deeper self-awareness intelligence layers (Status And Health)"></option>
  <option value="Docs, commands, and reporting intelligence (Status And Health)"></option>
  <option value="Doctor (Status And Health)"></option>
  <option value="Freeze or accept a production candidate (Status And Health)"></option>
  <option value="Golden replay regression guard (Status And Health)"></option>
  <option value="Health snapshot (Status And Health)"></option>
  <option value="Master infrastructure supervisor (Status And Health)"></option>
  <option value="Plan or apply the MLX library upgrade bundle (Status And Health)"></option>
  <option value="Point-in-time event store (Status And Health)"></option>
  <option value="Publish production-quality repair lanes (Status And Health)"></option>
  <option value="PyCharm active bot blue highlights (Status And Health)"></option>
  <option value="Refresh health gates (Status And Health)"></option>
  <option value="Refresh readiness evidence without the full dashboard (Status And Health)"></option>
  <option value="Refresh runtime dashboard contracts (Status And Health)"></option>
  <option value="Repair safe cross-system drift surfaces (Status And Health)"></option>
  <option value="Replay hash registry guard (Status And Health)"></option>
  <option value="Reporter quality infrabot (Status And Health)"></option>
  <option value="Review causal readiness blockers (Status And Health)"></option>
  <option value="Review Codex project guardrails (Status And Health)"></option>
  <option value="Review system plumbing control (Status And Health)"></option>
  <option value="Review ten-pillar production excellence (Status And Health)"></option>
  <option value="Review the cross-system drift mesh (Status And Health)"></option>
  <option value="Run production hardening watch (Status And Health)"></option>
  <option value="Runtime gate dashboard (Status And Health)"></option>
  <option value="Runtime status (Status And Health)"></option>
  <option value="Track production-quality SLO recurrence (Status And Health)"></option>
  <option value="Track readiness evidence accrual (Status And Health)"></option>
  <option value="Verify the durable live-order ledger (Status And Health)"></option>
  <option value="Repair local stateful storage regressions (Storage)"></option>
  <option value="Review or prune eligible local standby SQLite copies after BOT_LOGS soak (Storage)"></option>
  <option value="Run the storage disaster recovery bot (Storage)"></option>
  <option value="Safe force-clear storage pressure supervisor (Storage)"></option>
  <option value="Safe-eject the external BOT_LOGS drive (Storage)"></option>
  <option value="Switch collection back to the external BOT_LOGS drive (Storage)"></option>
  <option value="Switch collection to the Mac's internal drive (Storage)"></option>
  <option value="Apply the 10-layer dual-mode library efficiency upgrade (Strategy Research)"></option>
  <option value="Push advancement until the safety guard pauses it (Strategy Research)"></option>
  <option value="Push system efficiency until the safety guard pauses it (Strategy Research)"></option>
  <option value="Push the 12-domain whole-system frontier (Strategy Research)"></option>
  <option value="Review the 10-layer deep quant advisory upgrade (Strategy Research)"></option>
</datalist>

<details>
<summary>Generated command search index (163 commands; rebuilt by commands-hygiene)</summary>

Each row is generated from `governance/health/commands_contract_latest.json`, so added, removed, renamed, or cleaned-up commands change this index automatically.

- search-entry:22e73ecb232d03e12ee08ff74049f2413c87ccc69d9d95021f3542bbf6a05ff5 section:`Most Used` section_key:`most-used` title:Keep the Mac awake title_key:`keep-the-mac-awake` opsctl:`none` scripts:`none` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:da6278802587b3b33c87bc3b00c46f3e148080daa6c6f09b608aa8c97201eb70 section:`Most Used` section_key:`most-used` title:Start the full live stack title_key:`start-the-full-live-stack` opsctl:`start` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:f241d97b497d01015c7ce708cefbc8ed8cc6f57c7226474d2648dfbffb030b89 section:`Most Used` section_key:`most-used` title:Start the full live stack (fresh supervised restart) title_key:`start-the-full-live-stack-fresh-supervised-restart` opsctl:`start` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:f53aa42b1a8febd2e6c9269481b3fe1a565d705ab3877a1ddafbb23f309d9d91 section:`Most Used` section_key:`most-used` title:Stop the stack title_key:`stop-the-stack` opsctl:`stop` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:204bb89719335e4d279af4f58aa92f2e5992b348ac8c3fa02532b9189b1fb789 section:`Most Used` section_key:`most-used` title:Apply autonomic P-core resource governor title_key:`apply-autonomic-p-core-resource-governor` opsctl:`autonomic-governor` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:9cb3e9b63db29076ac4df68ccd59a7dcf65dc50c77177f51c75512d481da57e3 section:`Most Used` section_key:`most-used` title:Apply backlog writer catch-up waves title_key:`apply-backlog-writer-catch-up-waves` opsctl:`writer-cycle-coordinator` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:ccdac6998df3ce84520ae16ad9a5a500d2354752760b8973616d502f3f994856 section:`Most Used` section_key:`most-used` title:Apply income operating platform controls title_key:`apply-income-operating-platform-controls` opsctl:`income-operating-platform` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:08915f136d1a0bbd55be57f4a3068973c140fcaab7630cd765370e3b35b3caac section:`Most Used` section_key:`most-used` title:Apply memory pressure and multitasking controls title_key:`apply-memory-pressure-and-multitasking-controls` opsctl:`memory-pressure-intelligence` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:b82238b2506b6263ec666ca2f3ddae9bd2a27c26db00d07224a48144ac81518e section:`Most Used` section_key:`most-used` title:Apply pressure relief controls title_key:`apply-pressure-relief-controls` opsctl:`pressure-relief` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:141e6965176634cfced6e802b91ae6176ddebfdfbab34ab37d27a5ac4086e8a3 section:`Most Used` section_key:`most-used` title:Apply raw backlog refinement title_key:`apply-raw-backlog-refinement` opsctl:`raw-backlog-refiner` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:b4ec197c829c79297d5b206a318875882fcb5d3043aa2e7c50026bf67de5b29b section:`Most Used` section_key:`most-used` title:Apply runtime throttle and P-core priority controls title_key:`apply-runtime-throttle-and-p-core-priority-controls` opsctl:`runtime-throttle` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:57f30273c9e0a76c17e804248b0059ece2a20d4d5b721e1cd31be80084b17a99 section:`Most Used` section_key:`most-used` title:Ask what backlog and runtime need next title_key:`ask-what-backlog-and-runtime-need-next` opsctl:`system-needs` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:2791c2ca3930ab6c83c43a35475613267d9a338634453adf93e5ac24a1f3e67f section:`Most Used` section_key:`most-used` title:Attempt a safe global halt clear title_key:`attempt-a-safe-global-halt-clear` opsctl:`global-halt-auto-clear` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:6da4fa570df2247a7aaf805a0d839d646cad4f02547f71b4a6c8758cdccc0303 section:`Most Used` section_key:`most-used` title:Broker Truth Step 1: refresh Schwab auth title_key:`broker-truth-step-1-refresh-schwab-auth` opsctl:`token-refresh` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:3e494b156f6a54e22c473ee62ccd2b9e3927578cc05bccdd5f79cd2e2090e51e section:`Most Used` section_key:`most-used` title:Broker Truth Step 2: restart the Schwab loops title_key:`broker-truth-step-2-restart-the-schwab-loops` opsctl:`feed-refresh` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:e8ffaee1559eb46b606ea6880bc6a4f0e44d00d5b8097b8f0cae316065dad94d section:`Most Used` section_key:`most-used` title:Broker Truth Step 3: verify broker readiness and lane statuses title_key:`broker-truth-step-3-verify-broker-readiness-and-lane-statuses` opsctl:`none` scripts:`none` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:52f0524f52a0bcb43d3caa9ea06635662f6a6611062a731e2f8b502edb82fa08 section:`Most Used` section_key:`most-used` title:Build the paper evidence packet title_key:`build-the-paper-evidence-packet` opsctl:`evidence-packet` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:6d42226c654c79684b6dbdb8ad35657028fffa9898a54d319b843603ea9fbc7b section:`Most Used` section_key:`most-used` title:Check 12-lane system expansion execution title_key:`check-12-lane-system-expansion-execution` opsctl:`system-expansion-execution` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:ee34ed06b2f445719c88c4cf429d68d1ac6b3b8701e775587b745ae17bf0503a section:`Most Used` section_key:`most-used` title:Check backlog writer and drainer status title_key:`check-backlog-writer-and-drainer-status` opsctl:`writer-cycle-coordinator` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:037fa9514224171e03a04b98912b5de92cdad56c514b04d892db9eee27c192a1 section:`Most Used` section_key:`most-used` title:Check capital rotation control title_key:`check-capital-rotation-control` opsctl:`capital-rotation-control` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:d0c6ae912a39199e49e30bb31fc00e7213ef3f8cf2c507375075583d0ba33509 section:`Most Used` section_key:`most-used` title:Check Schwab indicator intelligence title_key:`check-schwab-indicator-intelligence` opsctl:`schwab-indicator-intelligence` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:5a04b7890366a0378b1a2a8be997ed63c98d929d9f38ed9e50630b54648d0b9d section:`Most Used` section_key:`most-used` title:Check support maintenance yield gate title_key:`check-support-maintenance-yield-gate` opsctl:`support-maintenance-gate` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:2b88c9c6053fdfb06e59133c5e2fd94d8ba95538d438cc0366b357c235e8c153 section:`Most Used` section_key:`most-used` title:Clear all halt flags now title_key:`clear-all-halt-flags-now` opsctl:`clear-all-halts` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:3e9a108107ab5d4ba44045a4fa2a9ef53dfb90055365176b4f8e54c980bba2c3 section:`Most Used` section_key:`most-used` title:Emergency stop: engage operator stop and global halt title_key:`emergency-stop-engage-operator-stop-and-global-halt` opsctl:`operator-control` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:f311e5809cc856703411fdcc5f2aab73a64131c779a23e0341ca7eaabe8d5be0 section:`Most Used` section_key:`most-used` title:Fast read-only health check title_key:`fast-read-only-health-check` opsctl:`health-fast` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:f62ebd1f05ac056fec2dd82c13dc99fbfef89d0ac22557fb8c1f1eec93e5d7c4 section:`Most Used` section_key:`most-used` title:Open the framework map PDF title_key:`open-the-framework-map-pdf` opsctl:`none` scripts:`scripts/ops/open_report_artifact.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:e94229836d34b2313a285fe32a697d3003922232336f707f71b8d60cb78de57e section:`Most Used` section_key:`most-used` title:Open the One Numbers CSV in Numbers title_key:`open-the-one-numbers-csv-in-numbers` opsctl:`none` scripts:`scripts/ops/open_report_artifact.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:89b9dcfdf96dd68b38f71ee72b6d5ec93e501567f27b7cd021040b6b6e050dc4 section:`Most Used` section_key:`most-used` title:Open the One Numbers PDF title_key:`open-the-one-numbers-pdf` opsctl:`none` scripts:`scripts/ops/open_report_artifact.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:2e6a9417d9a96417e253e648c0329a52616b30473b1a80e7d13f49258bb014dc section:`Most Used` section_key:`most-used` title:Open the special features PDF title_key:`open-the-special-features-pdf` opsctl:`none` scripts:`scripts/ops/open_report_artifact.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:bb8455dc31264328e8c9e4325a49d2c469abf3fca9cb02e2c347605c983c4588 section:`Most Used` section_key:`most-used` title:Phone mirror view for the live feed title_key:`phone-mirror-view-for-the-live-feed` opsctl:`phone-feed` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:5f0fd20962b6ac973b85e9c66ec4fe09c031b8510a3876f8d0e841197dca6809 section:`Most Used` section_key:`most-used` title:Refresh clearable global halt blockers title_key:`refresh-clearable-global-halt-blockers` opsctl:`global-halt-refresh` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:cb9ff599eab3d64a969c6a053202f702e6b920a833c86f714e0dc14bd7e4effb section:`Most Used` section_key:`most-used` title:Refresh the livefeed mirror without restarting sleeves title_key:`refresh-the-livefeed-mirror-without-restarting-sleeves` opsctl:`livefeed-refresh` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:e70be1682486604606b5f5613d8153f906b60c2cd1a4ac7b5103a17f537394db section:`Most Used` section_key:`most-used` title:Refresh the special features and framework map reports title_key:`refresh-the-special-features-and-framework-map-reports` opsctl:`showcase-refresh, system-explainers, report-pdfs` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:1fa62afa6d37ea594d0627fce524683b8fc205055a7962c5a77fcd4ab45e8a3d section:`Most Used` section_key:`most-used` title:Release operator stop only title_key:`release-operator-stop-only` opsctl:`operator-release` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:0c9b35c8ca1133d7f8e66e593898531e59f67393645ec063227521fdd1c5ab97 section:`Most Used` section_key:`most-used` title:Repair and restart the livefeed mirror title_key:`repair-and-restart-the-livefeed-mirror` opsctl:`livefeed-refresh-guard` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:58655ff3cbbd18712f0a624beea9df5b4975f1280748ac7aad5468233cb871fd section:`Most Used` section_key:`most-used` title:Run adversarial system drills title_key:`run-adversarial-system-drills` opsctl:`system-adversarial-drills` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:b4f7c4a75fd8d492a3c0c0e462680a7478239dd12e0928812d0a913b2e480740 section:`Most Used` section_key:`most-used` title:Run intense system drills title_key:`run-intense-system-drills` opsctl:`system-intense-drills` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:f92ddcc1552dac49577cf87954d67c293844e50130fb062c554fe8a1ce8ede28 section:`Most Used` section_key:`most-used` title:Run post-restart settlement title_key:`run-post-restart-settlement` opsctl:`post-restart-settle` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:abba61469c13355b518bf65a53577827ee8665a8ad6b31d90f3e39f4caace471 section:`Most Used` section_key:`most-used` title:Run the architecture upgrade scoreboard title_key:`run-the-architecture-upgrade-scoreboard` opsctl:`architecture-upgrade-scoreboard` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:b9ad6a600070399e6a8feae90265df4462762a9944ff403a3dbffca418ec820f section:`Most Used` section_key:`most-used` title:Runtime mode switchboard title_key:`runtime-mode-switchboard` opsctl:`none` scripts:`scripts/run_mode_switchboard.py` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:cc5031b721a273d250a8c8f2ec3209320d1fa49bd7120912ea0119d36e0531ec section:`Most Used` section_key:`most-used` title:Show global halt status and blockers title_key:`show-global-halt-status-and-blockers` opsctl:`global-halt-status` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:7a5f34d4a38a5ce387fc19978a6442baaccee1535dd6a6f6850b03ea08f8349e section:`Most Used` section_key:`most-used` title:Validate documented commands title_key:`validate-documented-commands` opsctl:`command-validity` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:f4a0c304b0b2b6a470031a0f7bd155d8a72564c299e37467783b6d330b842179 section:`Most Used` section_key:`most-used` title:Watch P-core/E-core load live/heavy title_key:`watch-p-core-e-core-load-live-heavy` opsctl:`none` scripts:`none` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:a26b7afb0898ca84d745f44776c0eb60b5f27a7d1573c738e7aae59362bea2ae section:`Most Used` section_key:`most-used` title:Watch P-core/E-core load with low overhead title_key:`watch-p-core-e-core-load-with-low-overhead` opsctl:`none` scripts:`none` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:34fca037f53ae7e652c9e7f2f8ef85c8459c5a2ca3a9170078a75d488d76467c section:`Accounts And Positions` section_key:`accounts-and-positions` title:Refresh Schwab account positions title_key:`refresh-schwab-account-positions` opsctl:`schwab-account-snapshot-refresh` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:3329a9f27ebab407f10027cd5e2319b45f8e164f7514bbeb04c957dfc8adda25 section:`Accounts And Positions` section_key:`accounts-and-positions` title:Review account policy context title_key:`review-account-policy-context` opsctl:`account-policy-context` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:3e92a3af2f20b8e73696039341da09d688222b861e541c2e6c8742742cefc99f section:`Accounts And Positions` section_key:`accounts-and-positions` title:Study all visible account positions title_key:`study-all-visible-account-positions` opsctl:`account-position-study` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:40aa1ccb959b6d48c874093fa56fb9083c5e74f454af950d9954b42e02ad70e3 section:`Accounts And Positions` section_key:`accounts-and-positions` title:Watch covered-call roll windows title_key:`watch-covered-call-roll-windows` opsctl:`covered-call-roll-watch` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:6905edc90e3669f5ba0e5f7cff89ed38d8a11fd7627c7ea1917fdfa9dc0d73cb section:`Data Context Syncs` section_key:`data-context-syncs` title:Crypto market context sync title_key:`crypto-market-context-sync` opsctl:`crypto-market-sync` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:28c8adf1bfbdc184f8a3246d69b035dadc8369dfe2e2719e349b5176b4b4bbe2 section:`Data Context Syncs` section_key:`data-context-syncs` title:FX market context sync title_key:`fx-market-context-sync` opsctl:`fx-market-sync` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:6d9912b0ea8130ddc98949a5b244a8b8e0be487b47c2e883ab942619eb2370ce section:`Data Context Syncs` section_key:`data-context-syncs` title:Macro context sync title_key:`macro-context-sync` opsctl:`macro-context-sync` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:6a1e634c03d65dc842403cab6a10ed7f390b177974b51bde06fa2baa41dd2806 section:`Data Context Syncs` section_key:`data-context-syncs` title:Options flow context sync title_key:`options-flow-context-sync` opsctl:`options-flow-sync` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:e09ffc26df0422653438ce9314334305bc0ce171383e70cebd71948b54ca3ecc section:`Data Context Syncs` section_key:`data-context-syncs` title:Source verification title_key:`source-verification` opsctl:`source-verification` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:45ef884dfcd2650aa72b2c1187d0a69c7ec30b42e1e5567eeeab2b1b454d6dea section:`Data Context Syncs` section_key:`data-context-syncs` title:Stock / crypto correlation sync title_key:`stock-crypto-correlation-sync` opsctl:`market-correlation-sync` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:9021aedf14582e34e52c383973a4edbb1a55ee89133b24627b0c2a6fffe36607 section:`Event Watches` section_key:`event-watches` title:Install the SpaceX/SPCX IPO downside watcher title_key:`install-the-spacex-spcx-ipo-downside-watcher` opsctl:`spacex-ipo-watch-install` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:ea20a1103e57488295c6080f103baf9ab20e0c494aba2bfb06137a0ab73d2003 section:`Event Watches` section_key:`event-watches` title:Run macro event intelligence title_key:`run-macro-event-intelligence` opsctl:`macro-event-intelligence` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:6282757337f51a8df69b4025fdce7df56e8ab54009eb32e623989171da1fae58 section:`Event Watches` section_key:`event-watches` title:Run the SpaceX/SPCX downside watch once title_key:`run-the-spacex-spcx-downside-watch-once` opsctl:`spacex-ipo-watch` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:de884061b10936dd66ff67297ff86cebd94549303ee414376f4579f2a8a1c106 section:`Live Feed Views` section_key:`live-feed-views` title:Heavy live feed with file diagnostics title_key:`heavy-live-feed-with-file-diagnostics` opsctl:`feed` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:73372346d081f5c52569f7302c1b8b637d9ad6d8464bb54b73ac944dd344600b section:`Live Feed Views` section_key:`live-feed-views` title:Heavy operator livefeed view title_key:`heavy-operator-livefeed-view` opsctl:`feed` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:27171a1814ad510bfe0117d4884493d8af5801741b7b28eee955fb0da45d969a section:`Notifications And Alerts` section_key:`notifications-and-alerts` title:Dry-run the startup Yes/No bot start prompt title_key:`dry-run-the-startup-yes-no-bot-start-prompt` opsctl:`startup-start-prompt-test` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:f501fefddd3071705e9741633daa46711a2d003e3de2241ce39e57d2b2bb0095 section:`Notifications And Alerts` section_key:`notifications-and-alerts` title:Install the startup Yes/No bot start prompt title_key:`install-the-startup-yes-no-bot-start-prompt` opsctl:`startup-start-prompt` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:7f6159ac1905567b7b741b898779baa7529339b38b99636e38b069bee05036bd section:`Notifications And Alerts` section_key:`notifications-and-alerts` title:Review remote alert control title_key:`review-remote-alert-control` opsctl:`remote-alert-control` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:e8cbc12e44df7029fed57721d6038cd71d0dfef02cd653d9a3415a6c5f9478b3 section:`Notifications And Alerts` section_key:`notifications-and-alerts` title:Send a test iMessage notification title_key:`send-a-test-imessage-notification` opsctl:`notify-test` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:ef723387f891f9bdbe3a22df35e212547f982ad84bb90eb2947b4e4815bf5545 section:`Notifications And Alerts` section_key:`notifications-and-alerts` title:Start the Mac notification and iMessage watcher title_key:`start-the-mac-notification-and-imessage-watcher` opsctl:`notify-start` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:327e49ed7c69f2ab71407234c9c0bdd3fe281c97edc00fda5d9f5f98de802ff6 section:`Notifications And Alerts` section_key:`notifications-and-alerts` title:Stop the notification watcher title_key:`stop-the-notification-watcher` opsctl:`notify-stop` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:39c33d5022d704eaf771a19ce14a4374f79c0618fd77cad29d8e0f4f69d4bf44 section:`Paper Trading` section_key:`paper-trading` title:Apply paper profitability controls title_key:`apply-paper-profitability-controls` opsctl:`paper-profitability-control` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:98fa28bd8b7a5e6f54321aa6254d8f4aa8b6b9954d061cb947e41937de44c57c section:`Paper Trading` section_key:`paper-trading` title:Apply the paper live-data standard title_key:`apply-the-paper-live-data-standard` opsctl:`paper-live-data-standard` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:befe059d6d49b4e6c3fa6b7c57be46b4c30ff06f7eaf92412dab0c3ac743aeca section:`Paper Trading` section_key:`paper-trading` title:Arm or candidate-promote the guarded 400 bot paper ramp title_key:`arm-or-candidate-promote-the-guarded-400-bot-paper-ramp` opsctl:`paper-400-ramp` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:855de17b5155c31be434930260497ef55cb0c98e91e63ce5db85fb5cd9447a93 section:`Paper Trading` section_key:`paper-trading` title:Check paper runtime regression guard title_key:`check-paper-runtime-regression-guard` opsctl:`runtime-paper-regression-guard` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:227f99d286741989a6dfd2d8961bb05917451c76c731cfae15bb8de007f58291 section:`Paper Trading` section_key:`paper-trading` title:Review guarded 400 bot paper ramp title_key:`review-guarded-400-bot-paper-ramp` opsctl:`paper-400-ramp` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:2b10c3089dd25a74e4e533fd26fec2ddb81e4ce00103187eee31f8cfb4a9ddc2 section:`Reports And PDFs` section_key:`reports-and-pdfs` title:Active bot stack PDF title_key:`active-bot-stack-pdf` opsctl:`none` scripts:`scripts/ops/open_report_artifact.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:8035aac80e3951d9e6f39db691cc8327ed5b733d7bc1033dffb2cf3b74505b54 section:`Reports And PDFs` section_key:`reports-and-pdfs` title:Incident report title_key:`incident-report` opsctl:`none` scripts:`scripts/ops/open_report_artifact.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:2064b1c296e1fcaedee03dced4c4660a6fca0671880e51dbbad7622fcb3887c4 section:`Reports And PDFs` section_key:`reports-and-pdfs` title:Incident review packet PDF title_key:`incident-review-packet-pdf` opsctl:`none` scripts:`scripts/ops/open_report_artifact.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:dcdf7d36ea5c4c290f6550c20ad3ab1154436ac0ed70d2b40dacff5dbf5eead1 section:`Reports And PDFs` section_key:`reports-and-pdfs` title:One Numbers report title_key:`one-numbers-report` opsctl:`none` scripts:`scripts/build_one_numbers_report.py` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:b4089617ef5839fef72f2c2e957b5c7f93b3f69219e87688e884ce1b7432d4cd section:`Reports And PDFs` section_key:`reports-and-pdfs` title:Open the active bot stack PDF title_key:`open-the-active-bot-stack-pdf` opsctl:`none` scripts:`scripts/ops/open_report_artifact.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:3b1fbcab9d5b6a2fec1a385ec5d381f7c8f6a7f3ea9d61d1eccacaa4c4f6dc8c section:`Reports And PDFs` section_key:`reports-and-pdfs` title:Open the bot explainability PDF title_key:`open-the-bot-explainability-pdf` opsctl:`none` scripts:`scripts/ops/open_report_artifact.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:74fd310d9048469aaccb35a8e3df38810aeb69f1f0616d140bc9bcab8fb20145 section:`Reports And PDFs` section_key:`reports-and-pdfs` title:Open the crash digest PDF title_key:`open-the-crash-digest-pdf` opsctl:`none` scripts:`scripts/ops/open_report_artifact.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:3297722128584d31af65e84c5e994a2bd5231c2557fc20387139e9e1eb24b49b section:`Reports And PDFs` section_key:`reports-and-pdfs` title:Open the daily auto verify PDF title_key:`open-the-daily-auto-verify-pdf` opsctl:`none` scripts:`scripts/ops/open_report_artifact.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:56230375fc822e590cd08decc546902bbf23a5426817b2fcf398ec7dcb4a3cc8 section:`Reports And PDFs` section_key:`reports-and-pdfs` title:Open the daily ops PDF title_key:`open-the-daily-ops-pdf` opsctl:`none` scripts:`scripts/ops/open_report_artifact.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:b7ffdb93bbf8f718e6a765e916c12c2714ebfce17fb0ae9e65b713f755c5d6ec section:`Reports And PDFs` section_key:`reports-and-pdfs` title:Open the daily runtime summary PDF title_key:`open-the-daily-runtime-summary-pdf` opsctl:`none` scripts:`scripts/ops/open_report_artifact.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:794baba774439be94f569d2af48f51babb3fd96e236215a46f2718f4e3f694d1 section:`Reports And PDFs` section_key:`reports-and-pdfs` title:Open the expansion inventory PDF title_key:`open-the-expansion-inventory-pdf` opsctl:`none` scripts:`scripts/ops/open_report_artifact.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:22adab77865f8d8028c749c2f706ffca443b823a819c506b2f38864417c266bf section:`Reports And PDFs` section_key:`reports-and-pdfs` title:Open the incident report PDF title_key:`open-the-incident-report-pdf` opsctl:`none` scripts:`scripts/ops/open_report_artifact.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:b58a1d969ef0a0f1148482fa1790797dffd0c62781c92a3732b7eb88f8426670 section:`Reports And PDFs` section_key:`reports-and-pdfs` title:Open the incident review packet PDF title_key:`open-the-incident-review-packet-pdf` opsctl:`none` scripts:`scripts/ops/open_report_artifact.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:20e5fa9b38a6e5f0314acacdfa5166e6cef684a1401cbdee740550cf74b6c7c3 section:`Reports And PDFs` section_key:`reports-and-pdfs` title:Open the macro crosscheck PDF title_key:`open-the-macro-crosscheck-pdf` opsctl:`none` scripts:`scripts/ops/open_report_artifact.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:b32638ffe43bb820255144d4dc15a8d139208ff62ee4a3dd589158fb40bb2546 section:`Reports And PDFs` section_key:`reports-and-pdfs` title:Open the market correlation PDF title_key:`open-the-market-correlation-pdf` opsctl:`none` scripts:`scripts/ops/open_report_artifact.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:b907ae371ca1c69ffe1a96a9ec41ebfbc783bc5a31771835444cf1bbba48b832 section:`Reports And PDFs` section_key:`reports-and-pdfs` title:Open the model card PDF title_key:`open-the-model-card-pdf` opsctl:`none` scripts:`scripts/ops/open_report_artifact.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:c932d3d493838effa78eed6d660ab881100dd00b518c6188e2216f01d8252665 section:`Reports And PDFs` section_key:`reports-and-pdfs` title:Open the paper execution calibration PDF title_key:`open-the-paper-execution-calibration-pdf` opsctl:`none` scripts:`scripts/ops/open_report_artifact.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:f8a8ca79f1f00906bee684395f7f677baa4df80f02282415b27ccc490b5cea18 section:`Reports And PDFs` section_key:`reports-and-pdfs` title:Open the paper performance PDF title_key:`open-the-paper-performance-pdf` opsctl:`none` scripts:`scripts/ops/open_report_artifact.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:fc389b313c97a9df7297f9a4d5fcf8e1fdd1adbbf63ee479740e3d27ee46d66e section:`Reports And PDFs` section_key:`reports-and-pdfs` title:Open the post-trade analysis PDF title_key:`open-the-post-trade-analysis-pdf` opsctl:`none` scripts:`scripts/ops/open_report_artifact.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:5ad0676b3f4693ae2e92e750b8349fab3e87957602e11638ff1373fefeb30f17 section:`Reports And PDFs` section_key:`reports-and-pdfs` title:Open the project timeline PDF title_key:`open-the-project-timeline-pdf` opsctl:`none` scripts:`scripts/ops/open_report_artifact.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:749e096e69b80fa097e7d21499d2284995edd06ded5bb2551d61a2c3d549b4b5 section:`Reports And PDFs` section_key:`reports-and-pdfs` title:Open the quant model control PDF title_key:`open-the-quant-model-control-pdf` opsctl:`none` scripts:`scripts/ops/open_report_artifact.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:d6c39cd830af5f2f5bf3401bb6673cfd41860bd9912b30ae8fabcda61944b595 section:`Reports And PDFs` section_key:`reports-and-pdfs` title:Open the replay feature ablation PDF title_key:`open-the-replay-feature-ablation-pdf` opsctl:`none` scripts:`scripts/ops/open_report_artifact.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:7d8c9ece3d2a0883862d72228d8aa8914f17e86bd206396da226d766f8cc8662 section:`Reports And PDFs` section_key:`reports-and-pdfs` title:Open the report catalog PDF title_key:`open-the-report-catalog-pdf` opsctl:`none` scripts:`scripts/ops/open_report_artifact.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:1643971063061950c3756c75a4747b1defa1e7e49a6eae169bad51f883747715 section:`Reports And PDFs` section_key:`reports-and-pdfs` title:Open the retrain scorecard PDF title_key:`open-the-retrain-scorecard-pdf` opsctl:`none` scripts:`none` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:ad5619a561b0eb045e34fdd9001f8dc2ca9f28d81abbec9b062637aff2c506d0 section:`Reports And PDFs` section_key:`reports-and-pdfs` title:Open the sentiment PDF title_key:`open-the-sentiment-pdf` opsctl:`none` scripts:`scripts/ops/open_report_artifact.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:79e940ef7f38af3293c562fe8d80e8c9884d14b4b3e28df4c2343e20d342475e section:`Reports And PDFs` section_key:`reports-and-pdfs` title:Open the source verification PDF title_key:`open-the-source-verification-pdf` opsctl:`none` scripts:`scripts/ops/open_report_artifact.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:4871f1e5a36c1e84b566213c2383275a174b2e4521c851d9d7a74926a29f4186 section:`Reports And PDFs` section_key:`reports-and-pdfs` title:Open the state snapshot drills PDF title_key:`open-the-state-snapshot-drills-pdf` opsctl:`none` scripts:`scripts/ops/open_report_artifact.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:ddbe2dfa18bb822315bff2bf8d2fcbd5c7a4f58afbac4b1c585e6568dc3cbc32 section:`Reports And PDFs` section_key:`reports-and-pdfs` title:Open the strategy attribution PDF title_key:`open-the-strategy-attribution-pdf` opsctl:`none` scripts:`scripts/ops/open_report_artifact.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:3775f93e4f0d381e3ffb68122b36ccb8dfcf0a52a8d830a3fe43bcadee015c7f section:`Reports And PDFs` section_key:`reports-and-pdfs` title:Open the strategy inventory PDF title_key:`open-the-strategy-inventory-pdf` opsctl:`none` scripts:`scripts/ops/open_report_artifact.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:206a73e62cba9261b09c658bc8abd461b1b207eaf34fe34a61bf787e1c295861 section:`Reports And PDFs` section_key:`reports-and-pdfs` title:Open the system overview PDF title_key:`open-the-system-overview-pdf` opsctl:`none` scripts:`scripts/ops/open_report_artifact.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:64a99b16d49ed5d08618be282a8e7c2ec03a0310d081adc2137edbb8125b509e section:`Reports And PDFs` section_key:`reports-and-pdfs` title:Open the training report PDF title_key:`open-the-training-report-pdf` opsctl:`none` scripts:`scripts/ops/open_report_artifact.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:0100141e7308c3caf4e01ce41094b4378a6b452377bb4754b6c4d1cca24cc516 section:`Reports And PDFs` section_key:`reports-and-pdfs` title:Open the unified lane scorecard PDF title_key:`open-the-unified-lane-scorecard-pdf` opsctl:`none` scripts:`scripts/ops/open_report_artifact.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:6ef7cb3d4f298c32d774b77579e6c348102bbf926044041a112326421e2c9114 section:`Reports And PDFs` section_key:`reports-and-pdfs` title:Paper performance report title_key:`paper-performance-report` opsctl:`none` scripts:`scripts/ops/open_report_artifact.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:6433af154ecc239ea86caac2f3279f5101cad91c7b3452df39b3434dde7fb187 section:`Reports And PDFs` section_key:`reports-and-pdfs` title:Refresh showcase, framework map, and PDFs now title_key:`refresh-showcase-framework-map-and-pdfs-now` opsctl:`showcase-refresh, system-explainers, report-pdfs` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:a95c554ad97c901d7f842e3f9648669d0d572508a55c3a8cfdb00436f3ed46bc section:`Reports And PDFs` section_key:`reports-and-pdfs` title:Repair and validate report PDFs title_key:`repair-and-validate-report-pdfs` opsctl:`report-quality-guard` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:715cb68e02c22a26e0575910c63871955d85364807606fa4fc12972f7caa938d section:`Reports And PDFs` section_key:`reports-and-pdfs` title:Report catalog bundle title_key:`report-catalog-bundle` opsctl:`report-pdfs` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:2d6f3e088d1289a8bdbdec63e7da16b1064b7a1ecfc4b7fb7675e93cf472c021 section:`Retrain` section_key:`retrain` title:Force full retrain (bypass prechecks) title_key:`force-full-retrain-bypass-prechecks` opsctl:`retrain-force-full` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:66f36e9e3bd84732995efe81a154264a21640c64b27037da97e878814c693ae2 section:`Retrain` section_key:`retrain` title:Full retrain preflight title_key:`full-retrain-preflight` opsctl:`runtime-training-snapshot, coverage-seed, coverage-gap-closer` scripts:`scripts/daily_log_refresh.sh, scripts/ops/opsctl.sh, scripts/retrain_schema_compatibility_guard.py, scripts/promotion_quality_gate.py` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:924904d5a4869f0f6b635462ae55c10a8a46d96069d9ad3ce35b3ac2b3a51c39 section:`Retrain` section_key:`retrain` title:Guarded retrain orchestrator title_key:`guarded-retrain-orchestrator` opsctl:`retrain-orchestrate` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:5823b776d532c4dfce081019db8a1139e845594715a1e658fa528cd44344f63e section:`Retrain` section_key:`retrain` title:Training and labeling intelligence title_key:`training-and-labeling-intelligence` opsctl:`training-labeling-intelligence` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:9ab5fdd6ad5064cfef401d806513ff95f45fd4b7069af991a89747c8ef41508f section:`Schwab Auth` section_key:`schwab-auth` title:Interactive Schwab authorization re-consent title_key:`interactive-schwab-authorization-re-consent` opsctl:`token-refresh-interactive` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:23929daa96c42f8741b1e768b7af0231359f3cbea301068c42803824f339c114 section:`Schwab Auth` section_key:`schwab-auth` title:Local Schwab credential setup title_key:`local-schwab-credential-setup` opsctl:`schwab-credentials` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:e1ac0f0b1b5f955d4f613f59a62c79ecf7cf0cac383b7743e562fbb72c754717 section:`Schwab Auth` section_key:`schwab-auth` title:Schwab auth recovery plus lane restart title_key:`schwab-auth-recovery-plus-lane-restart` opsctl:`token-refresh, feed-refresh` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:8c2b9aa5192eb1ae1ec1c2c040374e72647ae04889c5068e9ad5e7154bd379c6 section:`Schwab Auth` section_key:`schwab-auth` title:Schwab auth supervisor title_key:`schwab-auth-supervisor` opsctl:`schwab-auth-supervisor` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:b0b4bec38680f7144f2510a8062b06afeb67ef3918928b99de950d48d9232d8a section:`Schwab Auth` section_key:`schwab-auth` title:Schwab authorization refresh title_key:`schwab-authorization-refresh` opsctl:`token-refresh` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:71f0e1b794e9adbe51a843d7a071f4e39d3ced379530686a58ea04b856eaed80 section:`SQL And Reports` section_key:`sql-and-reports` title:Data quality refresh bundle title_key:`data-quality-refresh-bundle` opsctl:`livefeed-refresh` scripts:`scripts/ops/opsctl.sh, scripts/daily_log_refresh.sh, scripts/build_one_numbers_report.py` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:245edf8a3dca42087e239ac22bff55f4e2cd62db1db342b5c901e6f018a534a3 section:`SQL And Reports` section_key:`sql-and-reports` title:Full SQL refresh pipeline title_key:`full-sql-refresh-pipeline` opsctl:`none` scripts:`scripts/daily_log_refresh.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:d5b760dcc7a20c968d30ed6bde1aaab3f4a22ff4924960f466ecbbfd90c2e24d section:`SQL And Reports` section_key:`sql-and-reports` title:Quick SQL sync title_key:`quick-sql-sync` opsctl:`sql-sync` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:71d3ce37c28cb4cfc0104b436a90dfed970fa67f045953e705c780d999fc5ce7 section:`Status And Health` section_key:`status-and-health` title:Acquire independent fill evidence title_key:`acquire-independent-fill-evidence` opsctl:`independent-fill-acquisition` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:c6a6ebb72bbfc71a788f61cc27c19614ac6d5847c283a9fedf086e0913199a45 section:`Status And Health` section_key:`status-and-health` title:Adapt infrabots to current system needs title_key:`adapt-infrabots-to-current-system-needs` opsctl:`infrabot-adaptive-governor` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:c5e040e4852cd9be32ead9a962b153904c71ec238f2ff4c66b1c8474e87aafd8 section:`Status And Health` section_key:`status-and-health` title:Advance staged promotion candidates title_key:`advance-staged-promotion-candidates` opsctl:`promotion-candidate-advancement` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:67d8b4f85daf3e431e681c653ab5fa72c90517cc1d635b65ccc24fd06f541f71 section:`Status And Health` section_key:`status-and-health` title:Apply system architecture hardening title_key:`apply-system-architecture-hardening` opsctl:`system-architecture-hardening` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:f9437868e421148a7c1e2559aab72a0d323f1f8360c43ebeb1d35b30da449a42 section:`Status And Health` section_key:`status-and-health` title:Coinbase API health title_key:`coinbase-api-health` opsctl:`coinbase-api-health` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:09cf953d33c1c2c51af8ddb9720af132cea3194741a0b6c46c87a25c79f1fab7 section:`Status And Health` section_key:`status-and-health` title:Deeper self-awareness intelligence layers title_key:`deeper-self-awareness-intelligence-layers` opsctl:`deeper-intelligence-layers` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:2f0e9268604a8480a375e0753d1e5a598d4c3d85b4074cab4800bb561be986c3 section:`Status And Health` section_key:`status-and-health` title:Docs, commands, and reporting intelligence title_key:`docs-commands-and-reporting-intelligence` opsctl:`docs-reporting-intelligence` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:bb08a7f936a589c1fdc675f4cc2b1f53276020381016ff2fe5330de5867df9de section:`Status And Health` section_key:`status-and-health` title:Doctor title_key:`doctor` opsctl:`doctor` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:5537215dbb9b08a933a0732ec1c96b7a3ca909e287b18c75ccd20eaff09fb297 section:`Status And Health` section_key:`status-and-health` title:Freeze or accept a production candidate title_key:`freeze-or-accept-a-production-candidate` opsctl:`production-excellence` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:b21f08f7efcb5b9fbff0e11eb5a1438336432259f1d661951d864b1c1f1df7fe section:`Status And Health` section_key:`status-and-health` title:Golden replay regression guard title_key:`golden-replay-regression-guard` opsctl:`golden-replay-regression` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:bb83bdc55a9aed896c6bf2546beaec680166951d130a809fc32298ed8be150e9 section:`Status And Health` section_key:`status-and-health` title:Health snapshot title_key:`health-snapshot` opsctl:`health` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:153348628b2a3d25ae47aeef540a0d30820f1d4e97e21762b45f1a7a50b64f8f section:`Status And Health` section_key:`status-and-health` title:Master infrastructure supervisor title_key:`master-infrastructure-supervisor` opsctl:`master-infra-supervisor` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:9c1f711ea01aaa2f18b476a46dc7dfc7c9a9575883f53ce6430451861d94f242 section:`Status And Health` section_key:`status-and-health` title:Plan or apply the MLX library upgrade bundle title_key:`plan-or-apply-the-mlx-library-upgrade-bundle` opsctl:`mlx-library-upgrade` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:72bf9c0db31d518475afa3d396d19a290051fee283f3f0842da4e40db38663aa section:`Status And Health` section_key:`status-and-health` title:Point-in-time event store title_key:`point-in-time-event-store` opsctl:`point-in-time-event-store` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:9a0b69390a23a39542748fae0108a2bcd9f9a7c935f6effb86ab12d1d7eb91c9 section:`Status And Health` section_key:`status-and-health` title:Publish production-quality repair lanes title_key:`publish-production-quality-repair-lanes` opsctl:`production-quality` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:079d2a1a6cbe25881172205c3997bebea2da6f40f2d81640db1389579346a566 section:`Status And Health` section_key:`status-and-health` title:PyCharm active bot blue highlights title_key:`pycharm-active-bot-blue-highlights` opsctl:`pycharm-active-bot-highlights` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:b78415c6f92b0a099a704e922e977a6c9915872706aad0c3e753e0281d8f75ba section:`Status And Health` section_key:`status-and-health` title:Refresh health gates title_key:`refresh-health-gates` opsctl:`health-gates` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:439df2f7c7cda4e8fe6ea2f258af41da52f8f0703637cdf413114ce17362bac8 section:`Status And Health` section_key:`status-and-health` title:Refresh readiness evidence without the full dashboard title_key:`refresh-readiness-evidence-without-the-full-dashboard` opsctl:`readiness-evidence-refresh` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:5c42cde9787324dfb1206f08bba236c4f22ec3331b467e7b3f8cc25f5445a72d section:`Status And Health` section_key:`status-and-health` title:Refresh runtime dashboard contracts title_key:`refresh-runtime-dashboard-contracts` opsctl:`dashboard-refresh` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:9af5977e0643081899f53d7a99dc149d1e2953fa8400df3d62272e2ad979cb76 section:`Status And Health` section_key:`status-and-health` title:Repair safe cross-system drift surfaces title_key:`repair-safe-cross-system-drift-surfaces` opsctl:`system-drift-autopilot` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:b68225df4373e5ada1af692b9823d7aef7c489c9c6549e42e80dd9eca2c15c4f section:`Status And Health` section_key:`status-and-health` title:Replay hash registry guard title_key:`replay-hash-registry-guard` opsctl:`replay-hash-registry` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:bbdaeb098be7903b2af2e606229145d9c1ce937bbaef0d84edcdb06e6918f3bd section:`Status And Health` section_key:`status-and-health` title:Reporter quality infrabot title_key:`reporter-quality-infrabot` opsctl:`report-quality-guard` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:262ccdcf97cdcf53b207af3aedec35642eddcf3d5431d8c0e1103dea8400836e section:`Status And Health` section_key:`status-and-health` title:Review causal readiness blockers title_key:`review-causal-readiness-blockers` opsctl:`readiness-blocker-rollup` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:0941a33ecf2e9d9f31579a2d05664240e0d893fcecf01959db65122fbe2c1e4e section:`Status And Health` section_key:`status-and-health` title:Review Codex project guardrails title_key:`review-codex-project-guardrails` opsctl:`codex-project-guard` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:d0744ed396c14397231a43061bd3c0f98192ec9c7038b35f4658ee641ba21bde section:`Status And Health` section_key:`status-and-health` title:Review system plumbing control title_key:`review-system-plumbing-control` opsctl:`system-plumbing-control` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:75509af8d32c7f8dff82ce44dfdf3f8f7ce0061733c12acf69dd9a9ca9aead39 section:`Status And Health` section_key:`status-and-health` title:Review ten-pillar production excellence title_key:`review-ten-pillar-production-excellence` opsctl:`production-excellence` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:f048a90ce2aa3a66196801c0d0c06873fb5102bd888d7e89498b2ec0524f0891 section:`Status And Health` section_key:`status-and-health` title:Review the cross-system drift mesh title_key:`review-the-cross-system-drift-mesh` opsctl:`system-drift-guard` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:d67c3b8bfd796b7ba5fa9ed44fcb628b61eb71f3dfa841091a1309c3e6e0bf1b section:`Status And Health` section_key:`status-and-health` title:Run production hardening watch title_key:`run-production-hardening-watch` opsctl:`production-hardening-watch` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:c9994d1b6c30bba687625646026da154e91899e5294beab70202f89283afe2da section:`Status And Health` section_key:`status-and-health` title:Runtime gate dashboard title_key:`runtime-gate-dashboard` opsctl:`dashboard` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:4b60f2351646885db58bcc2366437e4dd736dd6cf55ce08a2fdac505a3c85bbe section:`Status And Health` section_key:`status-and-health` title:Runtime status title_key:`runtime-status` opsctl:`status` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:52642cb66c25bfe102046b575959fdfd7da8862ee6b3d1d79bb05d5fb84d03f5 section:`Status And Health` section_key:`status-and-health` title:Track production-quality SLO recurrence title_key:`track-production-quality-slo-recurrence` opsctl:`production-quality-slo` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:fd98f9e05f04ad9fa5756467259f08f5660d6e646feebd104a41061470d61f5e section:`Status And Health` section_key:`status-and-health` title:Track readiness evidence accrual title_key:`track-readiness-evidence-accrual` opsctl:`readiness-evidence-accrual` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:fd56b63365456606b1ac88f79b682e6d632c16386f7d6b35209bb22c71d6b94e section:`Status And Health` section_key:`status-and-health` title:Verify the durable live-order ledger title_key:`verify-the-durable-live-order-ledger` opsctl:`live-order-ledger` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:7162b1728aa0badca09eeab49bafe5d36d327687a0c9f6a6c059cd95e147c9c3 section:`Storage` section_key:`storage` title:Repair local stateful storage regressions title_key:`repair-local-stateful-storage-regressions` opsctl:`stateful-storage-regression-guard` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:e8dce5198877685a4116a8d493aabd0e79f722defb26ff3e3ba27bf329fee8b0 section:`Storage` section_key:`storage` title:Review or prune eligible local standby SQLite copies after BOT_LOGS soak title_key:`review-or-prune-eligible-local-standby-sqlite-copies-after-bot-logs-soak` opsctl:`storage-prune-standby` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:8abb179beed85141e8937968e8cf4c59f087200b63c0574248d16b70591e8e8d section:`Storage` section_key:`storage` title:Run the storage disaster recovery bot title_key:`run-the-storage-disaster-recovery-bot` opsctl:`storage-disaster-recovery` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:e56085c5fc172d09c7961f2369158a78fb4dd363adad9cc85bea1574485fd5a6 section:`Storage` section_key:`storage` title:Safe force-clear storage pressure supervisor title_key:`safe-force-clear-storage-pressure-supervisor` opsctl:`storage-pressure-clearance` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:d033ff4328b102076033803766c24164b0672eb4c6e5b569cefaa450c0823796 section:`Storage` section_key:`storage` title:Safe-eject the external BOT_LOGS drive title_key:`safe-eject-the-external-bot-logs-drive` opsctl:`storage-safe-eject` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:7006816c228714e0bd4dba9b314968af31c3ffc5dd8faf9330c13014599f9ee9 section:`Storage` section_key:`storage` title:Switch collection back to the external BOT_LOGS drive title_key:`switch-collection-back-to-the-external-bot-logs-drive` opsctl:`storage-switch-external` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:7d6cad11d8246065171ab0eb6205c7661cc14fa900eed6a6e9e2cfeeaf106ad0 section:`Storage` section_key:`storage` title:Switch collection to the Mac's internal drive title_key:`switch-collection-to-the-mac-s-internal-drive` opsctl:`storage-switch-local` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:de4cc1888fe3ac849eedf7a2f386cc3947bb02543adf31f3ccfffb6b9c72fd3f section:`Strategy Research` section_key:`strategy-research` title:Apply the 10-layer dual-mode library efficiency upgrade title_key:`apply-the-10-layer-dual-mode-library-efficiency-upgrade` opsctl:`library-efficiency-deepening` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:1c67cb916e8601e94bee8cb3f5db5cf1a213df1712889965bda85ebd3357dd1d section:`Strategy Research` section_key:`strategy-research` title:Push advancement until the safety guard pauses it title_key:`push-advancement-until-the-safety-guard-pauses-it` opsctl:`safety-bounded-advancement-frontier` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:8293241b91baca4b78f3f152907a1a2b60b5aeca19c189c27df10602d1319095 section:`Strategy Research` section_key:`strategy-research` title:Push system efficiency until the safety guard pauses it title_key:`push-system-efficiency-until-the-safety-guard-pauses-it` opsctl:`system-efficiency-frontier` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:1601a97b8e779208e0e1cedd4ad3cd965e5ab4122b899e31b4aad97a00fadfa2 section:`Strategy Research` section_key:`strategy-research` title:Push the 12-domain whole-system frontier title_key:`push-the-12-domain-whole-system-frontier` opsctl:`whole-system-safety-frontier` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
- search-entry:c977fe164744edfa108b6aef24299e40672375449ad1b68cd15dd40031ae1f84 section:`Strategy Research` section_key:`strategy-research` title:Review the 10-layer deep quant advisory upgrade title_key:`review-the-10-layer-deep-quant-advisory-upgrade` opsctl:`deep-quant-layer-upgrade` scripts:`scripts/ops/opsctl.sh` first_command:`cd /Users/dankingsley/PycharmProjects/schwab_trading_bot`
</details>

## Most Used

### Keep the Mac awake
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
caffeinate -dimsu
```

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

### Apply autonomic P-core resource governor
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh autonomic-governor --apply --json
```

This applies the host-aware budget for live loops, backlog writer, collectors, trainings, MLX/GPU jobs, reports, and foreground apps.
Associated bots/control layers: `autonomic-resource-governor`, `host-capability-contract`, `os-adapter-layer`, `workload-class-registry`, `computer-task-intelligence`.

### Apply backlog writer catch-up waves
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh writer-cycle-coordinator --apply --json
```

This lets the single writer run bounded catch-up waves and then hands off follow-through to the active drainer lane.
Associated bots/control layers: `writer-cycle-coordinator`, `backpressure-drainer-fleet`, `storage-backpressure-autopilot`, `retention-debt-sheriff`.

### Apply income operating platform controls
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh income-operating-platform --apply --json
```

This refreshes the 10-lane income operating platform: promotion gate, realized profit engine, drawdown governor, paper/live fill gap, live-micro lock, withdrawal simulator, account rules, sleeve ranking, failure drills, and human dashboard.
Associated bots/control layers: `income-operating-platform`, `income-readiness-control`, `paper-profitability-control`, `account-policy-context`, `chaos-drill-coordinator`.

### Apply memory pressure and multitasking controls
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh memory-pressure-intelligence --apply --json
```

This refreshes unified-memory, compression, swap, observer-overhead, foreground-app, and P-core widening gates before backlog or training work expands.
Associated bots/control layers: `memory-pressure-intelligence`, `autonomic-resource-governor`, `runtime-throttle`, `creative-cotenant-guard`.

### Apply pressure relief controls
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh pressure-relief --apply --json
```

This writes the pressure-relief override used by runtime loading, maintenance guards, heavy feed TTL, SQL cadence, foreground-app awareness, macro capture niceness, MLX/quant caps, report caps, and quiet-window behavior.
Associated bots/control layers: `pressure-relief-control`, `runtime-throttle`, `ingestion-storage-governor`, `mlx-intelligence-router`.

### Apply raw backlog refinement
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh raw-backlog-refiner --apply --json
```

This expands raw backlog handling into five coordinated sections: measurement, hot-file mapping, focused drain handoff, intake relief, and safe stale/sparse cleanup.
Associated bots/control layers: `raw-backlog-refiner`, `external-backlog-drain`, `ingestion-priority-queue`, `pressure-relief-control`, `stale-artifact-sweeper`.

### Apply runtime throttle and P-core priority controls
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh runtime-throttle --apply --json
```

This refreshes process priority, niceness, fanout limits, P-core feedback, and co-tenant headroom after the host pressure picture changes.
Canonical `master_bot_registry.json` writes are blocked by default; runtime registry adjustments publish `runtime_throttle_registry_candidate_latest.json` unless explicitly source-write authorized.
Associated bots/control layers: `runtime-throttle`, `process-fanout-guard`, `memory-pressure-intelligence`, `autonomic-resource-governor`.

### Ask what backlog and runtime need next
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh system-needs --json
```

Use this when you want the system to name the exact blocker, shard/file, next command, expected impact, risk, and stop condition.
Associated bots/control layers: `system-needs-intelligence`, `autonomic-resource-governor`, `memory-pressure-intelligence`, `writer-process-intelligence`.

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

This ensures the supervised Schwab sleeves are running and lets them pick up the refreshed token without a hard bounce. Add `--force-restart` only when you intentionally want to restart the loops.

### Broker Truth Step 3: verify broker readiness and lane statuses
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
/Users/dankingsley/PycharmProjects/schwab_trading_bot/.venv314/bin/python -c "from pathlib import Path; import json; root=Path('/Users/dankingsley/PycharmProjects/schwab_trading_bot/governance/health'); broker=json.loads((root/'broker_readiness_latest.json').read_text()); print(f'ready_for_open={broker.get(\"ready_for_open\")} auth_ok={broker.get(\"auth_ok\")} token_warning_level={broker.get(\"token_warning_level\")}'); print('lane,status,mismatch_count,error'); [print(f'{p.name.replace(\"broker_truth_\", \"\").replace(\"_latest.json\", \"\")},{json.loads(p.read_text()).get(\"status\", \"\")},{int(json.loads(p.read_text()).get(\"mismatch_count\", 0) or 0)},{json.loads(p.read_text()).get(\"error\") or \"\"}') for p in sorted(root.glob('broker_truth_*_latest.json')) if 'shared_snapshot' not in p.name]"
```

Healthy target: `ready_for_open=True`, `auth_ok=True`, and all Schwab broker-truth lanes reporting `status=ok` with `mismatch_count=0`.

### Build the paper evidence packet
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh evidence-packet --json
```

This builds the repeatable 30/60/90-day paper evidence packet with sleeve attribution, drawdown/income controls, realized-profit conversion, ops stability, and promotion lineage.
Associated bots/control layers: `paper-performance`, `sleeve-profitability-dashboard`, `paper-profitability-control`, `income-operating-platform`, `promotion-quality-gate`.

### Check 12-lane system expansion execution
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh system-expansion-execution --json
```

This builds the 12-lane expansion execution layer: predictive stability, self-healing routes, stale-surface repair, Schwab feature bridge, collector utility, sleeve safe modes, deficiency repair, hot-path storage, capital simulation, promotion ledger, dependency hardening, and operator memory.
Associated bots/control layers: `system-expansion-execution`, `system-architecture-contract-graph`, `schwab-indicator-intelligence`, `capital-rotation-control`, `system-self-model`.

### Check backlog writer and drainer status
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh writer-cycle-coordinator --json
```

This is the read-only writer/drainer check. Use it before launching another catch-up cycle so a running single writer is not duplicated.
Associated bots/control layers: `writer-cycle-coordinator`, `writer-process-intelligence`, `backpressure-drainer-fleet`, `ingestion-storage-governor`.

### Check capital rotation control
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh capital-rotation-control --json
```

This builds the paper-only capital movement map: sleeve inflow/outflow pressure, weak-sleeve outflow, paper tilt recommendations, and live-money promotion blockers.
Associated bots/control layers: `capital-rotation-control`, `capital-growth-intelligence`, `capital-growth-awareness`, `paper-profitability-control`, `whole-system-governor`.

### Check Schwab indicator intelligence
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh schwab-indicator-intelligence --json
```

This builds the Schwab thinkorswim study/strategy catalog, classifies each item by market circumstance, and maps advisory usage to sleeve families.
Associated bots/control layers: `schwab-indicator-intelligence`, `indicator-bot-common`, `sleeve-strategy-coverage`, `system-self-model`.

### Check support maintenance yield gate
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh support-maintenance-gate --json
```

This reports whether support, report, media, and maintenance jobs should yield to memory pressure and Mac fluidity controls.
Associated bots/control layers: `support-maintenance-gate`, `runtime-throttle`, `memory-efficiency-control`, `swap-pressure-governor`.

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
Associated bots/control layers: `runtime-gate-dashboard`, `master-infrastructure-supervisor`, `system-drift-guard`.

### Open the framework map PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh framework
```

This refreshes the framework-map source, renders a deterministic PDF, and falls back to HTML if the PDF is unavailable.

### Open the One Numbers CSV in Numbers
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh one-numbers-csv
```

Latest CSV path: `/Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/one_numbers/latest.csv`.
This refreshes One Numbers first so the CSV alias points at the freshest report day before opening it.

### Open the One Numbers PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh one-numbers
```

Latest PDF path: `/Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/one_numbers/one_numbers_latest.pdf`.
This refreshes One Numbers, rebuilds the PDF bundle, and falls back to markdown or JSON evidence if needed.

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

### Refresh the livefeed mirror without restarting sleeves
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh livefeed-refresh
```

`livefeed-refresh` is the operator-safe livefeed repair path. It refreshes the supervised local livefeed mirror and validates `governance/health/livefeed_local_latest.json` without restarting sleeve loops. Use `feed-refresh --source ... --stack-refresh` only when you intentionally want loop start/recovery work.

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

### Repair and restart the livefeed mirror
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh livefeed-refresh-guard --apply --force-restart --freshness-minutes 10 --json
```

Use this when the terminal livefeed starts showing stale output, escaped JSON fragments, token blobs, or mid-line storage payloads.
This validates every livefeed refresh route, restarts only the supervised local mirror, and checks `governance/health/livefeed_local_latest.json`; it does not restart sleeve loops or change paper/live execution authority.

### Run adversarial system drills
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh system-adversarial-drills --run-probes --json
```

This runs safe read-only probes and ranks cross-layer weak points without enabling live execution or launching duplicate storage drains.
Add `--apply` when you want the drill result artifact written to `governance/drills/system_adversarial_drill_results_latest.json`.
Associated bots/control layers: `system-adversarial-drill-autopilot`, `health-fast`, `system-drift-guard`, `master-infra-supervisor`.

### Run intense system drills
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh system-intense-drills --apply --json
```

This executes the existing intense drill suite and writes the improvement plan, using safe improvements only when explicitly requested.
Associated bots/control layers: `system-intense-drill-autopilot`, `runtime-throttle`, `incident-closeout`, `live-canary-control`.

### Run post-restart settlement
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh post-restart-settle --apply --json
```

This rechecks restart sanity, auth lease, global halt blockers, collector contracts, process watchdog coverage, and runtime throttle after a restart.

### Run the architecture upgrade scoreboard
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh architecture-upgrade-scoreboard --json
```

This scores the current architecture expansion layers against their proof artifacts and separates bounded recovery from true blockers.
Associated bots/control layers: `architecture-upgrade-scoreboard`, `system-architecture-contract-graph`, `system-drift-guard`.

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

### Validate documented commands
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh command-validity --json
```

### Watch P-core/E-core load live/heavy
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
sudo /Library/Frameworks/Python.framework/Versions/3.14/bin/asitop --interval 1 --show_cores 1
```

Use this briefly when you need faster visual feedback. The memory intelligence layer can flag interval-1 asitop as observer overhead if it starts distorting CPU or memory pressure.
Associated bots/control layers: external observer for `memory-pressure-intelligence`, `autonomic-resource-governor`, and `runtime-throttle`.

### Watch P-core/E-core load with low overhead
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
sudo /Library/Frameworks/Python.framework/Versions/3.14/bin/asitop --interval 3 --show_cores 1
```

Use this as the normal Apple Silicon watcher. The 3-second interval reduces observer overhead so the monitor is less likely to create the pressure it is measuring.
Associated bots/control layers: external observer for `memory-pressure-intelligence`, `autonomic-resource-governor`, and `runtime-throttle`.

## Accounts And Positions

### Refresh Schwab account positions
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh schwab-account-snapshot-refresh --json
```

Refreshes the shared Schwab account snapshot so Roth/cash account holdings, equities, and option legs are visible to the position-study and covered-call layers.

### Review account policy context
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh account-policy-context --json
```

Summarizes account-level rules and constraints so Roth/cash position logic stays separated from strategy and roll-watch interpretation.

### Study all visible account positions
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh account-position-study --json
```

Builds `governance/health/account_position_study_latest.json` from all visible Schwab accounts, account aliases, recent sleeve decisions, and covered-call roll context.

### Watch covered-call roll windows
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh covered-call-roll-watch --json
```

Evaluates held covered calls against account aliases, DTE windows, ITM depth, hard roll targets, and per-underlying preferences before publishing roll alerts.

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

## Event Watches

### Install the SpaceX/SPCX IPO downside watcher
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh spacex-ipo-watch-install --poll-seconds 30 --symbol SPCX --until-utc 2026-06-13T01:00:00+00:00
```

Installs the launchd watcher for first-print, high-watermark, IPO-price, spread, and proxy weakness alerts; policy remains monitoring-only with automatic execution disabled.

### Run macro event intelligence
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh macro-event-intelligence --json
```

Checks active macro/event bulletins, calendar verification, market relevance, and event-watch context used by the livefeed status snapshot.

### Run the SpaceX/SPCX downside watch once
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh spacex-ipo-watch --json
```

Reads the current SPCX/SpaceX quote context and writes the monitoring-only downside artifact without creating an order instruction.

## Live Feed Views

### Heavy live feed with file diagnostics
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed --source main --heavy --show-files --no-heavy-ttl --color --red-actions
```

Use this when the feed looks sparse or cut off; it prints followed files plus any skipped unreadable file paths and keeps the operator tab open without the pressure-relief heavy-feed TTL.

### Heavy operator livefeed view
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh feed --source main --heavy --no-heavy-ttl --color --red-actions
```

Use this as the primary operator view when you want decisions plus important storage, backpressure, auth, halt, and alert messages in one window.
The `--red-actions` palette keeps the feed red-dominant while leaving `BUY` green and `SELL` red.
If the Mac is running an `air_safe` or `constrained` memory-efficiency profile, the feed automatically trims decision fanout and uses a lower default line budget unless you pass your own `--lines` or `--no-memory-aware`.
The feed now probes files before following them; unreadable logs are skipped and counted instead of cutting off the stream.
Escaped JSON fragments are hidden by default so byte-tail startup cannot flood the terminal with `stdout_tail`, token, or storage-route payloads; add `--show-json-fragments` only for raw formatter debugging.

## Notifications And Alerts

### Dry-run the startup Yes/No bot start prompt
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh startup-start-prompt-test --dry-run --delay-seconds 0
```

Verifies the startup prompt state artifact without showing the GUI prompt or starting the trading stack.

### Install the startup Yes/No bot start prompt
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh startup-start-prompt --install --no-kickstart --no-browser
```

Arms a login-time macOS banner plus Yes/No prompt for starting `schwab_trading_bot` through the guarded `opsctl start` path.
The startup prompt path suppresses Schwab browser auth, GUI Chrome opens, headless Chrome PDF/render helpers, and timeline auto-PDF work.
The default install waits until the next login so it does not unexpectedly prompt or restart the stack right now.

### Review remote alert control
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh remote-alert-control --json
```

Summarizes critical alert backlog, iMessage bridge state, unacked alerts, and remote-notification readiness.

### Send a test iMessage notification
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh notify-test --enable-imessage --imessage-recipient "you@example.com" --imessage-min-severity critical
```

Use this after changing the recipient or iMessage allowlist; replace the recipient with the phone/email address that receives iMessage.

### Start the Mac notification and iMessage watcher
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh notify-start --enable-imessage --imessage-recipient "you@example.com" --imessage-min-severity critical
```

Installs and starts the macOS notification watcher with iMessage delivery enabled for critical allowed events.

### Stop the notification watcher
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh notify-stop
```

## Paper Trading

### Apply paper profitability controls
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh paper-profitability-control --apply --json
```

Refreshes the profitability, weak-profile containment, and promotion-readiness controls that feed the paper evidence packet.

### Apply the paper live-data standard
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh paper-live-data-standard --apply --json
```

Reapplies the paper-only live-data standard so eligible sleeves can observe real market data while live execution remains blocked.

### Arm or candidate-promote the guarded 400 bot paper ramp
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh paper-400-ramp --apply --promote-roster --json
```

Writes guarded paper caps and publishes a candidate registry promotion when global halt, memory, runtime, and ingestion gates are clean.
Canonical `master_bot_registry.json` writes require `--allow-source-registry-write` or `PAPER_400_RAMP_ALLOW_SOURCE_REGISTRY_WRITE=1`.

### Check paper runtime regression guard
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh runtime-paper-regression-guard --json
```

Verifies runtime throttle, resource guard, paper-ramp, support niceness, and paper execution pause contracts after a ramp or degradation fix.

### Review guarded 400 bot paper ramp
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh paper-400-ramp --json
```

Shows whether the 400-bot paper ramp is planned, armed, promoted, or blocked before writing runtime overrides.

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
./scripts/ops/open_report_artifact.sh daily-ops
```

Latest PDF path: `/Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/reports/daily_ops_report_latest.pdf`.
This refreshes the daily ops source, rebuilds the PDF bundle, then opens the report with markdown/JSON fallback.

### Open the daily runtime summary PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh daily-runtime
```

Latest PDF path: `/Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/sql_reports/daily_runtime_summary_latest.pdf`.
This rebuilds the PDF bundle and falls back to the runtime JSON artifact if the PDF is unavailable.

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
./scripts/ops/open_report_artifact.sh quant
```

Latest PDF path: `/Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/reports/quant_model_control/quant_model_control_latest.pdf`.
This refreshes the advanced quant-model feature, MLX, resource-cap, and research-only policy report, then opens the report-ready PDF.

### Open the replay feature ablation PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh replay
```

This regenerates the replay feature ablation evidence, renders the report PDF bundle, prefers the PDF artifact, and falls back to the latest JSON evidence if a PDF cannot be rendered.

### Open the report catalog PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh report-catalog
```

Latest PDF path: `/Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/reports/report_pdf_bundle_latest.pdf`.
This rebuilds the documented report catalog first, then opens the report-ready bundle PDF with HTML fallback.

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
./scripts/ops/open_report_artifact.sh state-snapshot
```

Latest PDF path: `/Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/state_snapshot_drills/state_snapshot_drills_latest.pdf`.
This rebuilds the state snapshot drill PDF and falls back to the latest drill JSON.

### Open the strategy attribution PDF
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/open_report_artifact.sh strategy-attribution
```

Latest PDF path: `/Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/reports/strategy_attribution_latest.pdf`.
This refreshes strategy attribution, rebuilds the PDF bundle, and falls back to markdown or JSON evidence.

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
./scripts/ops/open_report_artifact.sh system-overview
```

Latest PDF path: `/Users/dankingsley/PycharmProjects/schwab_trading_bot/exports/reports/system_overview/system_overview_weekly_platform_history_latest.pdf`.
This opens the week-by-week platform history and current-position overview PDF with markdown fallback.

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

### Local Schwab credential setup
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh schwab-credentials --check --json
./scripts/ops/opsctl.sh schwab-credentials --interactive --store keychain --json
```

Prompts locally for Schwab API credentials and stores them in macOS Keychain by default; no secret values are printed or written to tracked files.
This command does not open Chrome or a headless browser. Run the interactive token refresh after credentials are stored if OAuth consent needs renewal.

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

### Acquire independent fill evidence
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh independent-fill-acquisition --apply --json
```

Normalizes only provenance-verified broker-paper or replay fills from `exports/independent_fill_inbox`; model-derived fills remain in simulator diagnostics.

### Adapt infrabots to current system needs
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh infrabot-adaptive-governor --apply --json
```

This publishes the shared needs contract, capability registry, adaptive policy router, safety guard, and feedback ledger used to keep infrabots aligned with current degradation.
The apply form writes coordination contracts only; it does not launch repair fanout, retraining, live execution, or competing SQLite writers.

### Advance staged promotion candidates
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh promotion-candidate-advancement --json
./scripts/ops/opsctl.sh promotion-candidate-advancement --execute --json
```

The default form publishes the five-candidate queue. The execute form still requires two consecutive runtime-governor approvals and never updates the master registry directly.

### Apply system architecture hardening
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh system-architecture-hardening --apply --json
```

Writes the cross-layer architecture hardening artifact and read-only guardrails for queue, storage, runtime, paper/live, and reporting contracts.

### Coinbase API health
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh coinbase-api-health --json
```

This checks Coinbase public market-data endpoints and reports only credential presence booleans, never secret values.

### Deeper self-awareness intelligence layers
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh deeper-intelligence-layers --apply --json
```

This installs and scores the 10 deeper self-awareness layers: causal world model, belief ledger, digital twin replay, adversarial simulator, self-scientific method, resource economist, promotion court, living ontology, operator dialogue, and constitutional risk.

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

### Freeze or accept a production candidate
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh production-excellence --apply --initialize-candidate --json
./scripts/ops/opsctl.sh production-excellence --apply --accept-candidate-change --change-reason "Describe the reviewed production change" --json
```

Initialize only after the intended production code is committed. Accepted changes reset only the affected evidence scopes and preserve historical profitability.

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

### Publish production-quality repair lanes
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh production-quality --apply --refresh-contract --json
```

This turns live-canary blockers into ordered safe repair lanes for raw profitability, paper continuity, auth continuity, storage pressure, and promotion/paper freshness.

### PyCharm active bot blue highlights
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh pycharm-active-bot-highlights --apply --json
```

This writes the JetBrains `Active Bots` scope and blue file-color mapping so active `core/brain_refinery_*.py` files get a durable Project-pane scope background. PyCharm's bright blue filename text remains reserved for VCS-modified files.

### Refresh health gates
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh health-gates --json
```

This refreshes the health-gates artifact directly when stale health-gate state is blocking production-quality or live-canary readiness.

### Refresh readiness evidence without the full dashboard
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh readiness-evidence-refresh --apply --json
```

Runs the serialized candidate-bound evidence lane with per-step timeouts and a cooldown. It has no training-launch or live-order authority.

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

### Review causal readiness blockers
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh readiness-blocker-rollup --json
```

Collapses repeated downstream grade failures into unique engineering, elapsed-time, evidence, and outcome roots.

### Review Codex project guardrails
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh codex-project-guard --staged --json
```

Run this before Codex-authored commits or GitHub updates to catch source-of-truth drift, mixed-domain staging, and separate-domain README/docs leakage.

### Review system plumbing control
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh system-plumbing-control --json
```

Publishes the shared queue, storage, writer, data-plane, and paper/live boundary contract used to diagnose present degradation.

### Review ten-pillar production excellence
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh production-excellence --json
```

Reports the frozen candidate, clean soak, recovery drills, live execution, independent fills, promotion candidates, profitability, canary, grading integrity, and institutional evidence as ten fail-closed pillars.
Evidence debt is visible but does not interrupt healthy paper collection; live order submission stays locked until all ten pillars are ready.

### Review the cross-system drift mesh
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh system-drift-guard --json
```

This rolls command drift, summary/report drift, governance drift, workstation drift, and stack-runtime drift into one registry-backed health view.

### Run production hardening watch
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh production-hardening-watch --apply --json
```

The scheduled wrapper runs the bounded readiness-evidence refresh first, then publishes live-canary readiness, production-quality lanes, SLO state, causal blockers, and infrabot routing. Safe repair execution remains opt-in and governor-allowlisted.

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

### Track production-quality SLO recurrence
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh production-quality-slo --apply --refresh-quality --json
```

This keeps state across checks so repeated production-quality lane failures become watch, warning, or breach evidence instead of isolated snapshots.

### Track readiness evidence accrual
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh readiness-evidence-accrual --apply --json
```

Tracks candidate-bound counts, breadth, effective samples, observed rates, honest ETAs, and stalled producers without treating raw rows as independent proof.

### Verify the durable live-order ledger
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh live-order-ledger --json
```

Checks the transactional order-intent ledger, hash-chained lifecycle events, and unresolved submit or cancel outcomes. Unknown broker outcomes require reconciliation and are never auto-retried.
After independently verifying broker truth, use `--resolve-intent ID --resolution STATE --evidence TEXT`; the evidence-backed resolution is appended to the ledger event chain.

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

## Strategy Research

### Apply the 10-layer dual-mode library efficiency upgrade
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh library-efficiency-deepening --apply --json
```

Installs the 10 library-efficiency layers across both MLX and non-MLX libraries: routing, columnar data, MLX inference, incremental feature cache, pricing kernels, econometrics, tabular alpha, graph impact, path signatures, and benchmark-cost governance.
The contracts apply to both paper rehearsal and live advisory parity; paper/live execution authority remains disabled until runtime, promotion, and broker live gates clear.

### Push advancement until the safety guard pauses it
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh safety-bounded-advancement-frontier --apply --json
```

Applies the next 10 safe control-plane frontier stages: route assimilation, freshness DAG, cache ownership, cost ledger, paper/live parity witness, incremental feature reuse, pricing reuse, cross-impact graphing, route retirement, and soak/pause guard.
The command intentionally stops at advisory/control-plane scope when promotion evidence, active training, or live authority gates say the system needs a soak period.

### Push system efficiency until the safety guard pauses it
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh system-efficiency-frontier --apply --json
```

Applies the safe system-efficiency frontier for backend routing, runtime and memory caps, storage/write pressure, feature caching, livefeed trimming, training scheduling, report rendering, alert noise, paper execution truth, model route lifecycle, replay proof, and operator command flow.
This command is low-churn control-plane work only; it stops before execution authority, allocation authority, training intake, new high-volume collectors, heavy replay, destructive cleanup, or automatic model deletion.

### Push the 12-domain whole-system frontier
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh whole-system-safety-frontier --apply --json
```

Applies the safe control-plane frontier for promotion evidence, paper/live fill truth, feature cache, storage/backpressure, livefeed reliability, account positions, risk exposure graph, benchmark cost, model retirement court, A+ cockpit, notifications, and disaster-recovery replay.
This command stops before execution authority, allocation authority, training intake, new high-volume collectors, heavy replay, or automatic model deletion.

### Review the 10-layer deep quant advisory upgrade
```bash
cd /Users/dankingsley/PycharmProjects/schwab_trading_bot
./scripts/ops/opsctl.sh deep-quant-layer-upgrade --json
```

Installs and reports the 10 deeper quant layers: residual alpha, meta-labeling, conformal abstention, execution-cost decay, crowding/cross-impact, changepoints, systematic flow, robust optimization, special situations, and research governance.
The layer pack is collection-only and advisory-only; paper, live, allocation, execution, and training intake stay blocked until promotion gates clear.
