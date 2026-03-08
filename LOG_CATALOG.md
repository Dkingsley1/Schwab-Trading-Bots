# Logging and Accountability Catalog

## Purpose
This catalog defines where runtime, training, and audit records are written, how events are correlated, and how to verify ingestion completeness.

## Global Schema Rules
- `log_schema_version`: Required on newly written JSON and JSONL records. Default is `2` (override with `LOG_SCHEMA_VERSION`).
- `timestamp_utc`: UTC ISO-8601 timestamp for event time.
- Correlation fields:
- `run_id`
- `iter_id`
- `decision_id`
- `parent_decision_id`

## Runtime and Trade Logs
- `logs/*.log`: Process logs for loops, watchdogs, and retrain wrappers.
- `exports/trade_logs/**/*.jsonl`: Paper/live order and trade records.
- `decisions/**/*.jsonl`: Raw decision events.
- `decision_explanations/**/*.jsonl`: Decision explanations.

## Governance and Audit Logs
- `governance/events/write_failures_YYYYMMDD.jsonl`: Write-failure events from safe JSON/JSONL writers.
- `governance/audits/registry_mutation_journal_YYYYMMDD.jsonl`: Append-only registry mutation history.
- `governance/audits/registry_mutation_latest.json`: Latest registry mutation snapshot.
- `governance/health/training_success_latest.json`: Confirmed training-success marker used by deletion guards.
- `governance/health/retrain_scorecard_latest.json`: Retrain summary and quality outcomes.
- `governance/health/jsonl_sql_ingestion_health_latest.json`: SQL ingestion lag and invalid-line health summary.

## Training and Lineage Artifacts
- `data/trade_history/trade_learning_dataset.json`:
- Contains `lineage` metadata including source hash, builder script hash, git commit, output payload hash, and feature schema version.
- `models/trade_behavior_policy_*.npz`:
- Behavior model artifacts.
- `logs/trade_behavior_policy_*.json`:
- Training metrics, promotion gates, and lineage (dataset hash, trainer hash, git commit, model hash).
- `exports/sql_reports/retrain_scorecard_*.json`:
- Retrain outcomes plus lineage snapshot.

## SQL Ingestion Accountability
- Script: `scripts/link_jsonl_to_sql.py`.
- SQLite/MySQL schema includes:
- `run_id`, `iter_id`, `decision_id`, `parent_decision_id`, `log_schema_version`.
- Ingestion state file:
- `governance/jsonl_sql_link_state.json`.
- Ingestion health summary:
- `governance/health/jsonl_sql_ingestion_health_latest.json`.

## Deletion Safety Gates
- Deletion logic is gated by confirmed training success and registry row-state checks.
- Guard code: `core/training_guard.py`.
- Guard tests: `tests/test_training_guard.py`.

## Replay and Verification Workflow
1. Verify runtime log freshness in `logs/` and JSONL growth in `decisions/` or `exports/trade_logs/`.
2. Run SQL sync: `./scripts/ops/opsctl.sh sql-sync`.
3. Check `governance/health/jsonl_sql_ingestion_health_latest.json` for `pending_lines` and `oldest_uningested_age_seconds`.
4. Validate retrain markers in `governance/health/training_success_latest.json` and `governance/health/retrain_scorecard_latest.json`.
5. Inspect `governance/audits/registry_mutation_latest.json` after prune, retire, or master updates.

## Retention and Maintenance
- Retention cleanup is managed by ops scripts (for example `scripts/data_retention_policy.py`).
- Keep governance journals and health markers available for audit windows before archival.
- Prefer append-only JSONL trails and immutable model/log artifacts for reproducibility.
