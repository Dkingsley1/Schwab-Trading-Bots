# Generation Behavior Attribution

## Purpose

The generation attribution control turns the accepted production-candidate event chain into explicit observation windows and compares system behavior between selected generations. It preserves the cumulative segmented soak as developmental context while keeping the current unchanged-candidate promotion clock separate.

Canonical owners:

- `governance/evidence/production_candidate_events.jsonl`: hash-chained accepted candidate events and scope changes.
- `scripts/ops/generation_behavior_attribution.py`: candidate-window reconstruction, hot/archive log discovery, deduplication, behavior aggregation, paper-outcome join, and comparison policy.
- `governance/research/generation_behavior_attribution_latest.json`: machine-readable comparison and evidence limitations.
- `exports/reports/operator/generation_behavior_attribution_latest.md`: compact operator report.

## Attribution Tiers

Candidate-stamped rows are identity-, generation-, and time-bound. They are the strongest behavioral tier.

Older rows that predate candidate stamping may be associated descriptively by the immutable accepted-candidate time window. That legacy tier is labeled separately, never becomes promotion-grade evidence, never grades the current candidate, and can be disabled with `--no-legacy-window-association`.

Unknown candidate IDs, generation mismatches, out-of-window rows, invalid timestamps, and duplicate message IDs are rejected. Missing evidence remains missing rather than becoming zero.

## Measures

The report compares decision count and observation breadth, profiles and symbols, directional intent and final action rates, guard blocks, no-edge holds, ingestion receipts and quality, freshness, latency, utility, source quality, and kill-switch incidence. Candidate-bound paper generation flows add post-cost outcomes when both generations have sufficient samples.

The comparison is associational. Market regime, source availability, and intervening accepted changes remain confounders; more trades alone do not establish alpha. Economic claims require candidate-bound post-cost outcomes.

## Soak Contract

The cumulative main-soak counter includes accepted historical segments and planned-maintenance accounting. It measures accumulated developmental exposure, not uninterrupted runtime and not live-promotion credit. Only the current candidate's clean forward window can satisfy the separate 720-hour gate.

Use:

```bash
./scripts/ops/opsctl.sh generation-behavior-attribution \
  --from-generation 65 \
  --to-generation 99 \
  --last-days 21 \
  --json
```

The command is read-only with respect to signals and trading. It cannot change thresholds, actions, quantities, paper orders, allocation, strategy promotion, candidate acceptance, or live execution.
