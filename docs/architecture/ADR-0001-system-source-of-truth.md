# ADR-0001: System Source Of Truth And Signal Evidence

## Status

Accepted

## Context

The platform has many generated artifacts: command docs, report PDFs, health snapshots, broker-truth files, decision explanations, and SQL mirrors. Without an explicit ownership map, generated files can be mistaken for hand-maintained control surfaces.

## Decision

The system source of truth is documented in `docs/architecture/SOURCE_OF_TRUTH.md`.

Command documentation is generated from `scripts/ops/commands_hygiene_bot.py`. Report opening is controlled by `scripts/ops/open_report_artifact.sh`. Decision and signal evidence is emitted through `core/accountability.py`, including the `governance/events/signal_generation_*.jsonl` stream for good and bad signal generation.

Aggressive sleeve performance is judged with Sortino ratio, because downside volatility matters more than upside variance for high-conviction sleeves. Conservative sleeve performance is judged with Sharpe ratio, because total volatility should stay low in capital-preservation sleeves.

## Consequences

- Generated docs must be regenerated from their owning source rather than edited as permanent truth.
- Runtime evidence remains auditable even when downstream SQL/report jobs lag.
- Signal generation has a canonical event stream that records both successful trade-intent signals and blocked/weak/no-trade signals.
- Sleeve reports expose the ratio that matches the sleeve mandate instead of applying one metric everywhere.
