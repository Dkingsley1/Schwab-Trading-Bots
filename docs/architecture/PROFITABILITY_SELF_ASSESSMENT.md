# Candidate-Bound Profitability Self-Assessment

## Purpose

The profitability self-assessment gives the runtime one canonical answer to two different questions:

1. Are the requested profitability controls implemented correctly?
2. Does the current accepted candidate have enough post-cost evidence to support a profitability claim?

Those grades are deliberately independent. An `A+` implementation grade cannot raise an economic evidence grade, authorize allocation, or unlock live execution.

The assessment also reports an `economic_context_source_grade` from the signed sleeve-routing contract. That grade answers whether every decision family and runtime route has fresh, point-in-time economic context from a sufficiently diverse source pool. It is operational input coverage, not post-cost profitability evidence, and therefore has no authority to raise `economic_evidence_grade`, allocate capital, or enable execution.

## Eight-Lane Contract

The assessment covers:

1. Candidate-bound confidence thresholds and abstention.
2. Sleeve-family and regime-specific thresholds.
3. A `+0.08` paper threshold uplift and `0.88` abstention budget for bond and dividend families until replay supports a safer setting.
4. Tradeability floors, portfolio-conflict ceilings, and fail-closed unknown evidence.
5. MAE, MFE, exit-timing, post-entry regime, continuation, and harvest-regret learning.
6. At least 30 independent candidate-bound fills for each supported Schwab paper market type.
7. A `0.25` paper entry-size cap, with scaling no higher than `1.10` and only after evidence validates it.
8. No automatic portfolio allocation until at least four independently profitable, sufficiently observed, low-correlation sleeves qualify.

Every lane is paper-only. The control rejects direct threshold loosening, loss-recovery sizing increases, automatic allocation, promotion authority, and live-order authority.

## Candidate And Accounting Rules

The current production-candidate identity must match every required source. Source receipts include a freshness result and SHA-256 digest. A missing or conflicting candidate identity blocks the assessment instead of combining records.

Historical paper inventory and losses remain visible for risk management and exit decisions. They do not grade a newly accepted candidate. Current-candidate profitability begins with candidate-bound schema-v2 post-cost outcomes and cannot be inferred from the historical ledger.

Accepted soak generations are not discarded. The paper-performance owner groups identity-stamped schema-v2 outcomes by production candidate. The assessor then joins each group to the tamper-evident candidate event chain and requires the recorded generation and observation timestamps to fit entirely inside that accepted generation's window. Valid associations feed a developmental ledger that records the change reason, affected scopes, samples, observed days, and post-cost delta. Unbound rows, mixed generations, forged chains, generation mismatches, and outcomes outside the accepted window remain visible but cannot be attributed.

Developmental attribution is evidence for what to investigate next, not proof that a code change caused an outcome. It may route bounded paper-only collection, counterfactual replay, independent-fill acquisition, weak-sleeve containment, and loss or missed-opportunity labeling to their existing owners. It cannot force a trade, martingale, average down, raise size to recover a loss, loosen an acceptance threshold without replay, allocate capital, rewrite history, promote a candidate, or enable live execution.

## Status Semantics

- `assessment_status=ready`: the assessor has complete, consistent candidate identity and can publish truth.
- `overall_status=collecting`: the assessor is healthy, but economic evidence is incomplete.
- `overall_status=ready`: the configured economic evidence firewall is ready; this still does not grant live authority.
- `overall_status=blocked`: candidate identity is missing or inconsistent, so affected evidence is rejected.
- `developmental_soak_learning.status=ready`: at least one accepted generation has identity- and time-bound post-cost outcomes available for developmental comparison.
- `developmental_soak_learning.status=collecting`: the generation chain is valid, but bound outcomes are still accruing.
- `developmental_soak_learning.status=blocked`: the event chain is unavailable or invalid, so generation attribution fails closed without blocking ordinary paper collection.

## Commands And Outputs

Run:

```bash
./scripts/ops/opsctl.sh profitability-self-assessment --json
./scripts/ops/opsctl.sh calibration-control --apply --json
./scripts/ops/opsctl.sh counterfactual-replay --json
```

The canonical outputs are:

- `governance/health/profitability_self_assessment_latest.json`
- `governance/health/profitability_self_assessment_latest.md`
- `governance/health/calibration_abstention_overrides_latest.json`

The system-needs intelligence, self-model, runtime artifact refresh graph, and live-feed contract consume the assessment. Each unresolved need names its exact artifact or shard, command, expected impact, risk, stop condition, candidate identity, and soak effect.

## Soak Policy

Applying paper-only tightening preserves cumulative segmented soak history and does not request a full soak-clock reset. A semantic candidate change starts or continues a separately attributable clean candidate window; historical evidence is retained but never relabeled as evidence for the new candidate.

The cumulative soak and clean promotion clock remain separate. Every verified accepted generation can contribute to developmental learning, while only the current unchanged candidate can accrue the clean 720 hours and current-candidate economic evidence required by the live-promotion contracts.
