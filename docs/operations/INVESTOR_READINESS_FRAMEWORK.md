# Investor Readiness Framework

## Purpose

This framework turns investor-readiness work into 20 explicit, machine-evaluated controls. It does not claim the strategy is investable, profitable, legally cleared, or ready for live orders. It distinguishes four outcomes:

- `ready`: the required implementation and evidence are present.
- `implemented_evidence_pending`: the control exists, but organic candidate, runtime, or live evidence has not accrued.
- `external_action_required`: a real independent accountant, verifier, or qualified counsel must complete the work.
- `implementation_gap`: an engineering surface is absent or its fail-closed contract is incomplete.

The system publishes counts, not a blended readiness percentage. A strong operational control cannot rewrite weak economic evidence, and a generated report cannot stand in for live history or an outside attestation.

## Twenty Controls

| ID | Requirement | Completion source |
| --- | --- | --- |
| `i01` | Broker-verified live results | candidate-bound live broker records plus independent verification |
| `i02` | Returns net of all modeled costs | candidate-forward post-cost accounting and independent recomputation |
| `i03` | Controlled drawdowns and risk of ruin | sufficient independent days, stress evidence, and no drawdown breach |
| `i04` | Statistically credible edge | corrected multiple testing, deflated Sharpe, PBO, and sufficient periods |
| `i05` | Capacity evidence | observed liquidity/impact curves and allocator clearance |
| `i06` | Diversification evidence | independently profitable, sufficiently observed, low-correlation sleeves |
| `i07` | Independently checkable records | signed replayable experiments, candidate fills, and external verification |
| `i08` | Bounded automation | declared roles, durable order truth, no automatic scaling, human release |
| `i09` | Operational resilience | implemented controls plus paper-soak and independent-monitor evidence |
| `i10` | Commercial and IP defensibility | commercial boundary controls plus independent IP review |
| `r01` | Shortlist one to three strong sleeves | profitability-firewall qualified sleeves only |
| `r02` | Record every experiment | append-only signed ledger and exact replay |
| `r03` | Require positive post-cost expectancy | sufficient candidate observations and a positive clustered lower bound |
| `r04` | Complete the soak before canarying | headline cumulative soak includes accepted reset segments; canary clearance still requires the separate unchanged-candidate 720-hour window and canary milestones |
| `r05` | Measure paper/live divergence | candidate-bound live samples within declared limits |
| `r06` | Predetermine scaling and rollback | fixed stages, caps, clean windows, and operator release |
| `r07` | Publish a labeled tear sheet | candidate evidence under a paper/hypothetical disclosure |
| `r08` | Obtain independent accounting review | internal validator plus a real outside signed review |
| `r09` | Maintain an investor data room | content-hashed source index with missing evidence visible |
| `r10` | Choose a legal/business structure | qualified external legal review for the actual product model |

The canonical IDs and owners are defined in `config/investor_readiness_v1.json`.

## Canary Capital

The initial `$200` is a founder-funded execution-validation envelope. It is deliberately too small to serve as an income target and does not establish a permanent portfolio size.

Future deposits may increase total account equity. They do not automatically increase a sleeve weight, order size, canary stage, or live authority. Scaling requires all of the following:

1. Positive candidate-bound post-cost expectancy.
2. Controlled drawdown evidence.
3. Acceptable risk-of-ruin evidence.
4. Measured capacity for the proposed capital.
5. Qualified diversification and concentration limits.
6. Paper/live divergence within limits.
7. Required clean evidence windows.
8. Explicit operator release for that stage.

The investor-readiness evaluator is read-only. It cannot create an allowlist, submit an order, promote a strategy, or change a risk limit.

## External Evidence

Independent performance verification, accounting review, IP ownership review, and legal-structure review must come from real outside providers. Each attestation must identify the provider and signer, carry a signing time, point to the reviewed document, and contain a matching SHA-256 digest. Placeholder JSON or a system-generated statement cannot clear these controls.

The applicable registration, disclosure, custody, commodity/futures, and marketing obligations depend on the eventual business model and jurisdiction. The software records review evidence; it does not make the legal determination.

## Generated Outputs

- `governance/health/investor_readiness_control_latest.json`
- `exports/reports/operator/investor_readiness_packet_latest.md`
- `exports/reports/investor/paper_performance_tear_sheet_latest.md`
- `exports/investor_data_room/index_latest.json`
- `exports/investor_data_room/README_latest.md`

Run:

```bash
./scripts/ops/opsctl.sh investor-readiness --json
```

The bounded production evidence-refresh profile also refreshes this control after its profitability, allocation, canary, and production-excellence dependencies.
