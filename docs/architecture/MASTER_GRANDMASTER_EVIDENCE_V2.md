# Master And Grand-Master Evidence V2

## Purpose

The v2 evidence plane gives each canonical sleeve a bounded sleeve-master packet and gives the Grand Master one deterministic cross-sleeve rollup. It turns the full bot hierarchy into inspectable evidence without changing the existing decision or execution path.

This plane is advisory and shadow-only. It cannot create an order payload, submit or cancel an order, change execution flags, mutate the bot registry, allocate capital, override broker truth or halt controls, or promote a bot to live money. Profitability is never guaranteed.

## Canonical Sources

- Policy: `config/master_grandmaster_evidence_v2.json`
- Synthesis engine: `core/master_grandmaster_evidence.py`
- Artifact owner: `scripts/ops/master_grandmaster_evidence_control.py`
- Compact health: `governance/health/master_grandmaster_evidence_v2_latest.json`
- Full packet catalog: `governance/master_grandmaster/evidence_packets_v2_latest.json`
- Hierarchy and taxonomy: `config/bot_organization_v1.json` and `governance/bot_organization/bot_hierarchy_latest.json`

Each input records freshness, semantic state, and a content or producer receipt. Required stale inputs fail synthesis closed. Runtime artifact refresh waits for the required upstream producers in the same evidence epoch before publishing this layer.

## Sleeve Masters

One packet is emitted per canonical sleeve, up to the policy cap. Each packet contains:

- active, signal, and shadow-eligible member counts;
- sub-sleeve, cohort, role, and correlation-cluster diversity;
- classification confidence and bounded metadata-review examples;
- multi-axis regime coverage, specificity, compatibility, and hard mismatches;
- correlated-vote concentration;
- available paper-truth and post-cost profitability summaries;
- a weighted evidence score, grade, status, and allowlisted recommendations;
- an explicit authority block with paper and live execution disabled.

The engine evaluates regime compatibility only when the observed context has enough known axes. Thin context is reported as evidence debt; it is not silently treated as a match.

## Grand Master

The Grand Master aggregates sleeve-master evidence by bot count and publishes:

- sleeve-master status and grade distributions;
- the lowest-evidence sleeves that need review first;
- current observed multi-axis regime context;
- paper coordination readiness;
- human live-review evidence readiness;
- exact operational holds and promotion blockers;
- one allowlisted recommended posture.

It does not invent trades or combine packets into an order. A future execution integration would require a separate reviewed contract, locked replay evidence, paper validation, live-money promotion gates, and explicit human authorization.

## Truth Separation

The artifact keeps four truths separate:

1. `structural_grade` reports whether policy, hierarchy, receipts, source freshness, caps, and authority locks are valid.
2. `grade` reports current evidence quality. Weak regime metadata or economic evidence can honestly produce a low grade while structure remains A+.
3. `paper_coordination_ready` requires structurally valid inputs, fresh A+ paper truth, and sufficient runtime capacity.
4. `human_live_review_evidence_ready` additionally requires fresh source, regime, profitability, execution, and account-position evidence plus the configured sleeve-master quality floor.

An operational pressure event creates `operational_hold`; it does not corrupt the structural grade. A stale required source, receipt mismatch, malformed policy, incomplete hierarchy, or unsafe authority setting creates `blocked_integrity`.

## Resource Contract

- Sleeve-master packet count is capped by policy.
- Review and mismatch examples are bounded per sleeve.
- The compact health artifact does not duplicate the complete hierarchy or packet catalog.
- Synthesis is deterministic for the same evidence inputs and evaluation time.
- Atomic replacement and exclusive ownership protect both generated outputs.

## Operations

Run:

```bash
./scripts/ops/opsctl.sh master-grandmaster-evidence --json
```

The control participates in runtime artifact refresh, the required freshness SLO, the runtime dashboard, control-surface ownership, source-mutation protection, command generation, CI compile checks, and adversarial regression tests.
