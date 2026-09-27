# Generation Fill Learning

## Purpose

The generation-fill learning lane reuses verified paper-fill outcomes without rewriting their provenance. The current accepted candidate owns the derived learning run. Each source fill keeps the candidate ID and generation that produced it.

For the current run, the learning owner is G104. This does not make an older fill a G104 fill and does not add the fill to G104 forward profitability or clean-soak evidence.

## Eight Controls

1. Candidate identity is accepted directly from immutable fill metadata or recovered from a unique exact decision receipt.
2. Rows without exact identity remain in the `legacy_unbound` quarantine ledger. Candidate time windows are never used to relabel them.
3. Compact rows preserve fill, decision, candidate, model, policy, routing, cost, MAE/MFE, and outcome receipts when those fields exist.
4. Verified fills join to the hash-chained accepted-generation change manifest. Multi-scope changes remain associational rather than causal.
5. Negative post-cost outcomes receive bounded hard-negative weight. Positive and neutral outcomes retain separate bounded weights.
6. Recency decay and per-generation weight budgets prevent stale or high-volume generations from dominating the target challenger.
7. Validation is chronological with a purge embargo plus leave-one-generation-out folds. Random shuffle validation is forbidden.
8. The result can enter only the current-generation offline challenger after lineage, training quality, and validation gates pass. It cannot swap runtime models, promote, submit an order, or enable live execution.

## Evidence Classes

| Class | Learning use | Promotion use |
| --- | --- | --- |
| Verified empirical paper fill | Offline challenger after all gates pass | Never from this historical view |
| Verified expected-fill or replay simulation | Low-weight simulation pretraining only | Never |
| Legacy unbound fill | Quarantine and provenance-repair diagnostics | Never |
| Current candidate forward fill | Normal current-candidate evidence path, separately accounted | Only through existing candidate-forward gates |

## G104 Boundary

`learning_target.generation=104` identifies the owner of the derived dataset and challenger experiment. `source_provenance.generation` identifies the generation that produced each fill. These fields must never be collapsed.

The materialized datasets are:

- `governance/training/generation_fill_learning/g104_learning_rows.jsonl`
- `governance/training/generation_fill_learning/g104_quarantine_rows.jsonl`

Build them with:

```bash
./scripts/ops/opsctl.sh generation-fill-learning --target-generation 104 --apply --json
```

The command is fail closed. A noncurrent target generation, invalid candidate chain, incomplete chronological validation, missing lineage, or blocked training-quality control prevents challenger launch.

The same guarded materialization runs before the daily small-batch and weekly full-sweep retrain lanes. A collecting or blocked result is nonfatal to those schedules: it prevents historical rows from entering the challenger while leaving the existing retrain path available. Set `GENERATION_FILL_LEARNING_ENABLED=0` only for an explicit maintenance bypass.
