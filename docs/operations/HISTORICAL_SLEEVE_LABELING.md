# Historical Sleeve Labeling

## Ownership

`core/historical_sleeve_labels.py` defines research-only historical annotations subordinate to `config/sleeve_strategy_contracts_v1.json` and `core/sleeve_strategy_specialization.py`. `scripts/ops/historical_sleeve_labeling.py` owns inventory, bounded ingestion, checkpoints, source receipts, supplemental targets, and coverage. Raw source schemas and economic authority owners are unchanged.

`config/historical_sleeve_research_horizons_v1.json` owns explicit primary and secondary research windows for all 111 sleeves. `scripts/ops/historical_sleeve_backfill.py` owns the compressed full-inventory runner and human-readable horizon publication. The windows are predeclared research parameters, not validated optimal settings. They do not change canonical execution holding periods.

## Commands

```bash
./scripts/ops/opsctl.sh historical-sleeve-labeling
./scripts/ops/opsctl.sh historical-sleeve-labeling --run --seconds 120
./scripts/ops/opsctl.sh historical-sleeve-labeling --run --full --seconds 7200
```

The full runner requests background priority once and records denial without retrying
the privileged operation. `core/background_work_budget.py` then paces its single
worker to a 25%-of-one-core CPU-time budget; this is cooperative pacing, not hard
CPU affinity. Its code hash is part of the implementation receipt. All storage,
preparation, memory, time, maintenance, and Mac-fluidity admission checks remain
mandatory. A corrected restore-capacity plan is not a verified restore or launch
permission. See [STATE_SNAPSHOT_RESTORE.md](STATE_SNAPSHOT_RESTORE.md). Use
`--new-inventory` after source changes to retain earlier immutable run evidence.

The first command inventories only. The second processes one batch; repeat it to resume. `--new-inventory` creates a new versioned run without replacing earlier evidence. It is needed after implementation/contract changes or to include newer source files. No recurring job is installed.

The third processes the entire declared inventory until completion or a guard/budget stops it. It resumes verified completed-source partitions rather than reusing the small SQLite label database. An interrupted source restarts from its beginning. The maximum foreground run is two hours; it never schedules itself or overrides admission to finish.

## Evidence

- `inventory.json`: frozen source identities, file fingerprints, exact scope, unavailable routes, all sleeve contracts, code and policy receipts.
- `labels.sqlite3`: compact per-record annotations, source/ordinal/hash receipts, dispositions, and supplemental contexts. This is a research sidecar, not a trainer input.
- `coverage.json` and `governance/health/historical_sleeve_labeling_latest.json`: all-sleeve coverage, processed counts, pending sources, unresolved horizons, and outcome blockers.

The run pointer is `governance/training/historical_sleeve_labels/latest_run.json`. Never treat the count of contracts, annotated rows, or successfully scanned files as a count of verified economic targets. Duplicate raw records retain separate source receipts; reported row totals are not independent samples. Current context counts only include targets recomputed against the latest indexed observation revision.

Full runs use `latest_full_run.json` and `full_runs/<run-id>/`. Each source has a gzip annotation partition and completion receipt with source fingerprint, per-row hashes, compressed-file and decoded-row hashes, count, and scan interval. SQL snapshots are per-source read transactions, not one globally atomic inventory timestamp; committed WAL is visible and future rows are not continuously tailed. Supplemental marks are deduplicated by sleeve, instrument, provider, candidate, timestamp, price, and snapshot. Conflicting endpoint prices remain quarantined. `research_horizons.json` and `research_horizons.md` list every sleeve's windows and required endpoint even when heavy scanning is blocked.

## Research Windows

| Family | Primary | Secondary Range |
| --- | --- | --- |
| Intraday execution | 30 minutes | 1 minute to 1 hour |
| Microstructure | 1 minute | 1 second to 15 minutes |
| Crypto spot | 4 hours | 15 minutes to 7 days |
| Crypto funding | 8 hours | 1 hour to 7 days |
| Directional swing / pairs | 5 days | 1 hour to 42 days, recipe-specific |
| Directional equity / macro | 20 days | 1 to 90 days |
| Event | 1 day; fast reaction 1 hour | 1 minute to 20 days, recipe-specific |
| Dividend capture / income | 7 / 90 days | 1 to 365 days, recipe-specific |
| Derivatives / basis | 1 day | 1 hour to 30 days plus actual expiry/cashflows |
| Hedging / preservation | 5 / 20 days | 1 to 90 days, recipe-specific |
| Structured credit | 30 days | 7 to 365 days plus actual cashflows |
| Pricing validation | 1 day | 7 and 30 days plus completed OOS validation |
| Controls | 1 hour | 5 minutes, 1 day, 7 days; actual incident/result required |

Days mean 86,400 elapsed UTC seconds, not exchange sessions. Missing weekends, event clocks, distributions, maturity dates, synchronized legs, or adjudicated results are not interpolated. Funding windows do not imply a venue's payment schedule. Price context is diagnostic and always decision-anchored; event/control/economic targets still need their own correctly anchored materializers. Purge the feature-through-outcome interval and embargo at least the longest active label horizon. The 365-day auxiliary window is not a blanket one-year waiting requirement for every bot.

## Objective Boundaries

| Sleeve Objective | Required Primary Evidence |
| --- | --- |
| Directional / digital asset | Position outcome after costs and matched benchmark; venue and funding when applicable |
| Execution | Arrival quote, observed fill, costs, adverse selection, session boundary |
| Income | Realized distributions, corporate actions, position path, tax/cost basis |
| Event | Verified event clock/window, matched event cohort, costs |
| Macro / carry | Price, carry, roll, funding, hedge cashflows, matched benchmark |
| Basis / pairs | Synchronized legs, quantities, hedge ratio, convergence and all leg costs |
| Volatility / derivatives | Contracts, expiry, premiums, payoffs, Greeks, hedge cashflows |
| Hedging | Portfolio with and without hedge, tail reduction, carry/false-positive cost |
| Preservation | Portfolio path, cash/defensive benchmark, avoided loss, opportunity/reentry cost |
| Control | Incident/action identity, verified result, recovery/detection clock, adjudicated false positives |

All primary targets currently stay pending in this sidecar. Producer P&L claims are preserved as unverified claims, never promoted automatically. Price context is supplemental only. Explicit research windows do not invent event/expiry endpoints; these require their own materializers. Out-of-sample research validation remains separate from market/control outcomes.

## Safety And Completion

The small runner accepts governed JSONL/gzip and compatible cold `jsonl_records` SQLite sources. The full runner also reads committed WAL snapshots, `json_file_records`, and configured Parquet payload exports. Changed files, oversized/corrupt records, unsupported schemas, excessive WAL growth/Parquet row groups, and unavailable routes remain explicit. Other formats and unconfigured archives are not scanned. No source deletion, migration, live-lane restart, credential access, model launch, or trading enablement occurs.

Limits are 2 MiB per record, 512 MiB of fresh payload per batch (at most one final bounded record beyond that threshold), 256 MiB RSS high-water, a 512 MiB sidecar page cap, fresh preparation/storage admission, and a live free-space reserve. The storage-regression gate must also admit the batch. Gzip seeking can reread earlier compressed input; a parent watchdog bounds that replay and archive I/O. Interrupted database transactions roll back, while prior checkpoints remain. Large gzip files may require a future seek index for efficient deep resumes. No cap is automatically raised.

The full runner replaces the 512 MiB per-run label database with an 8 GiB compressed-output cap and a separate 512 MiB deduplicated mark index. It retains the 2 MiB input record cap, 256 MiB RSS budget, physical reserve, and all admission gates. Full means an uncropped source inventory, not unlimited resources or permission to bypass Mac-fluidity pauses. Receipt hashes are rechecked on resume; a source is never marked complete by a manifest alone.

A denied run republishes status with `previous_evidence_not_recomputed` and the original `evidence_timestamp_utc`; prior row counts do not represent another batch. Code/contract changes require a new versioned inventory when admission reopens. Historical output is not consumed automatically by the training scheduler.

`historical_backfill_complete` remains false until all in-scope source coverage and authority-specific outcome materialization are genuinely complete. Missing outcomes are not negative examples, and historical evidence never earns current-candidate validation or soak credit.
