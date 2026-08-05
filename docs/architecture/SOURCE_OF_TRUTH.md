# System Source Of Truth

This file is the top-level source-of-truth map for the Schwab trading bot. Each row names the canonical artifact to trust first, the generated evidence that verifies it, and the owner surface that should be edited when behavior changes.

| Area | Canonical Source | Verification / Runtime Evidence | Edit Here First |
| --- | --- | --- | --- |
| Operator commands | `scripts/ops/commands_hygiene_bot.py` inventory | `COMMANDS.md`, `governance/health/commands_contract_latest.json` | `scripts/ops/commands_hygiene_bot.py` |
| Operator livefeed status | `scripts/ops/live_feed_status_contract.py`, `scripts/ops/live_feed_tail.sh` | freshness, reconciliation, cause, recovery, PAPER impact, and action fields in each livefeed status snapshot | livefeed status contract first, then the owning health producer |
| Report opening and PDF fallbacks | `scripts/ops/open_report_artifact.sh` | `exports/reports/*_latest.{pdf,html,md}` | `scripts/ops/open_report_artifact.sh` |
| Broker truth and account readiness | `scripts/run_shadow_training_loop.py` broker-truth helpers | `governance/health/broker_truth_*_latest.json`, `governance/health/broker_readiness_latest.json` | broker adapter and broker-truth helpers |
| Schwab auth handshake | `scripts/ops/schwab_auth_refresh.py`, `scripts/ops/schwab_auth_supervisor.py` | `governance/health/schwab_auth_refresh_latest.json`, `governance/health/schwab_auth_supervisor_latest.json` | auth helper scripts |
| Sleeve performance metrics | `scripts/paper_performance_report.py` | `governance/health/paper_performance_latest.json`, `exports/reports/paper_performance_latest.*` | paper performance report |
| Decision and signal evidence | `core/accountability.py` | `decision_explanations/**`, `governance/channels/decision/**`, `governance/events/signal_generation_*.jsonl` | shared accountability writer |
| Training labels and outcome authority | `core/runtime_training_common.py`, `scripts/ops/training_labeling_intelligence.py` | `governance/training_labeling_intelligence/all_bot_label_materialization_latest.json`, runtime training diagnostics `label_evidence_audit` | shared runtime label guard and labeling intelligence together |
| Storage routing | `core/storage_router.py`, `core/storage_target_override.py` | `governance/health/storage_*_latest.json`, `.ok` markers | storage router/control scripts |
| Codex project guardrails | `AGENTS.md`, `scripts/ops/codex_project_guard.py` | `governance/health/codex_project_guard_latest.json`, `.githooks/pre-commit` | `AGENTS.md` and guard script together |
| System architecture docs | `docs/architecture/ADR-0001-system-source-of-truth.md` | generated system explainer reports | ADR files in this directory |
Rules:

- Do not hand-edit generated operator docs as the lasting fix. Update the generator, then run `./scripts/ops/opsctl.sh commands-hygiene --apply --json`.
- Decision logs and channel logs are evidence, not control knobs. Change the writer contract in `core/accountability.py` before changing downstream consumers.
- Market, operational, and research labels have different outcome authorities. Never substitute forward price direction for an infrastructure result or an in-sample research score for out-of-sample evidence.
- Broker truth is the account state authority. Sleeve decisions can be confident, but execution should still defer to broker-truth, risk, halt, and storage guard evidence.
- Report commands should open the best available artifact and degrade from PDF to HTML or markdown instead of silently doing nothing.
- Codex guardrails should block mixed-domain commits and separate-domain README/docs drift before GitHub publishing.
