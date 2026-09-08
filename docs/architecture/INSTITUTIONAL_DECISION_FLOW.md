# Institutional Decision Flow

## Purpose

The institutional decision flow makes every transition from observation to abstention, bounded paper intent, or protective rejection explicit. It is designed to improve decision quality and capital scalability without claiming guaranteed profit or weakening any existing control.

The canonical evaluator has three consumers. The active paper loop may use it only to veto or reduce an already-approved paper intent. The execution lane revalidates the resulting sleeve-policy receipt before paper execution, promotion, and future live execution. A detached read-only sidecar consumes append-only master-control decisions and produces rankings and funnel evidence. None of these consumers can originate or reverse direction, increase quantity, bypass an existing HOLD, submit an order by itself, mutate registries or candidate state, or grant promotion.

## Sleeve Policy Resolution

One base policy resolves into a deterministic policy family for every runtime sleeve. Resolution uses an exact profile map first, then ordered profile and domain rules, then a conservative fallback. The receipt binds the base policy, family, profile, domain, lifecycle, weights, floors, market-quality limits, paper and live controls, objective, evidence focus, strategy definition, and execution eligibility to a SHA-256 digest.

| Family | Primary emphasis |
| --- | --- |
| `balanced_directional` | Balanced signal, edge, execution, portfolio, and risk evidence. |
| `long_horizon_income` | Income quality, compounding, turnover, overlap, and long-term fit. |
| `intraday_momentum` | Quote freshness, trend persistence, low latency, and short-horizon costs. |
| `swing_directional` | Conviction, regime persistence, multi-day edge, and portfolio fit. |
| `relative_value` | Paired identity, consensus, neutrality, overlap, and spread edge. |
| `volatility_derivatives` | Volatility regime, convexity, liquidity, Greeks, and tail risk. |
| `liquidity_microstructure` | Order-book integrity, toxicity, latency, routes, and inventory risk. |
| `macro_rates_fx` | Central-bank, curve, currency, cross-asset, and regime context. |
| `event_driven` | Event lineage, surprise, repricing, quote freshness, and gap risk. |
| `commodity_inflation` | Curves, inflation, currencies, roll costs, and macro confirmation. |
| `digital_asset_basis` | Cross-venue quality, basis or funding, liquidity, and gap risk. |
| `tail_hedge` | Hedge need, stress regime, convexity cost, and stressed liquidity. |
| `structured_credit` | Dependency, collateral, counterparty, marking, and tail dependency. |
| `research_models` | Point-in-time lineage, out-of-sample tests, uncertainty, and decay. |
| `infrastructure_control` | Source integrity, runtime health, routing, and control freshness. |

All specialized profiles must resolve without the fallback in regression tests. `data_collection_only`, research-only, training-only, structured-credit, and infrastructure-control lifecycles may still be scored and trained, but their receipt is execution-ineligible.

## Strategy Definition Contract

Policy v4 separates a broad family from its concrete profile variant. Every resolved profile has a complete, hash-bound definition for decision horizon, portfolio role, primary edge, entry style, exit style, sizing method, regime dependency, cost model, uncertainty method, capacity method, validation method, shorting policy, and allowed position transitions. Profile overrides distinguish economically different strategies that share a family, including dividend capture versus long-horizon compounding, bond carry versus cash rotation, earnings equity events versus single-name option events, pairs versus cross-sectional stat arb, volatility arbitrage versus gamma scalping, and short-bias versus convex tail protection.

The action contract joins the proposed direction to fresh broker position truth. BUY is classified as entering or adding a long, or covering a short. SELL is classified as reducing a long, entering a short, or adding to a short. The resolved strategy may forbid the transition or require short permission, linked-leg truth, or a defined-risk structure. Unknown account position state is visible during paper collection and fails closed for future live execution.

## Resolved Sleeve Playbook

The resolver now compiles each family definition plus its profile override into one immutable decision playbook. The playbook includes the economic objective, entry and exit contracts, sizing and capacity methods, regime and cost assumptions, validation method, shorting rules, allowed position transitions, family evidence focus, required quantitative axes, ordered stage sequence, mode-specific required stages, and a family-weight-derived stage priority. Its SHA-256 digest is part of the resolved policy receipt and is revalidated in the execution lane. A changed playbook therefore cannot silently reuse an older evaluation or execution intent.

This is specialization without a second decision engine. The profile playbook changes what evidence the canonical evaluator emphasizes and explains; it does not create a direction, reverse an intent, enlarge quantity, or bypass a HOLD.

## Quantitative Evidence Contract

The decision records nine evidence axes: selection-bias control, independent samples, uncertainty calibration, signal-decay fit, payoff asymmetry, capacity headroom, crowding or residual-alpha quality, tail survival, and regime stability. Each family requires the subset relevant to its economic thesis. Every axis records its source, whether it is a direct measurement or proxy, score, floor, and pass state.

Missing evidence is not converted to zero and is not assumed to pass. Paper mode may continue collecting when an axis is missing or proxy-only; an explicit direct failure can only downsize or veto an existing paper quantity. Future live execution requires all family-required axes to be directly measured and passing. A model confidence score, generic risk-headroom score, or structural control grade is never relabeled as independent profitability proof.

### Candidate-Bound Evidence Bridge

The paper runtime refreshes a bounded evidence packet per profile from the current profitability window. The adapter rejects missing candidate IDs, generation mismatches, inactive candidate filters, pre-candidate watermarks, and artifacts generated before the candidate cutoff. It currently promotes five measurements into direct decision evidence when their producers report them as available:

- Cluster-effective sample coverage from the profile's candidate-forward post-cost statistics.
- Payoff asymmetry from realized candidate-forward average wins and average losses.
- Selection-bias resilience from the minimum of false-discovery, probability-of-backtest-overfitting, and deflated-Sharpe evidence.
- Signal-decay fitness from candidate-forward daily post-cost profile history.
- Tail survival from independently recomputed risk-of-ruin and drawdown-breach probabilities.

The multiple-testing guard aligns common candidate-forward trading days before computing PBO. The decay monitor uses the candidate-forward post-cost series whenever candidate binding is required. Lifetime book snapshots remain useful for operations and exit management, but they cannot qualify the current generation. The remaining axes must come from current decision-level calibration, capacity, crowding, and regime evidence; the bridge does not manufacture them from generic confidence scores.

## Ten Stages

1. **Observation** binds the broker, profile, lane, symbol, timestamp, run, and snapshot.
2. **Data qualification** checks freshness, source quality, quote agreement, completeness, and latency.
3. **Signal formation** distinguishes a directional hypothesis from an intentional no-edge HOLD.
4. **Consensus and regime** tests signed independent agreement, disagreement, and regime compatibility.
5. **Post-cost edge** requires an explicit edge lower confidence bound greater than modeled round-trip costs.
6. **Execution feasibility** evaluates spread, latency, slippage, impact, route quality, and the active execution guard.
7. **Portfolio fit** evaluates allocation confidence, overlap, conflict, concentration, lane budget, and turnover.
8. **Non-bypassable risk** preserves broker truth, circuit, account, portfolio, loss, drawdown, and risk vetoes.
9. **Quality priority** ranks opportunities only after the preceding evidence stages pass.
10. **Outcome learning** joins later fills, costs, excursions, exits, and forward outcomes to the original evidence hash.

## Decision Classes

| Class | Meaning |
| --- | --- |
| `no_edge_hold` | The active strategy intentionally found no directional edge. |
| `protected_hold` | A directional intent was stopped by a safety, profitability, portfolio, broker, or execution control. |
| `data_evidence_blocked` | Required fresh or complete evidence is unavailable. |
| `economic_edge_unproven` | Direction exists, but a positive post-cost lower confidence bound does not. |
| `watchlist_near_miss` | Decision quality is close enough to monitor, but qualification is incomplete. |
| `qualified_shadow_candidate` | Every evidence stage passed; this still grants no order or promotion authority. |
| `shadow_candidate_rejected` | The directional candidate is below the bounded shadow policy. |

## Operator Trace And Live Feed

Every evaluation publishes a machine-readable trace with `pass`, `block`, `not_reached`, and pending outcome states; stage evidence and family floors; the first blocker and reason code; paper and live stage progress; regime and post-cost-edge state; position transition; and the next bounded evidence action. After the monotonic paper control runs, a compact operator summary binds the trace to the output action, quantity cap, playbook digest, and summary digest.

The heavy live feed reads that exact summary and displays `flow_state`, `flow_current`, `flow_progress`, `flow_blocker`, `flow_regime`, `flow_edge_state`, `flow_transition`, mode quality-gate states, and short playbook and summary receipts. Older JSONL records remain readable through the prior field fallback. Record timestamp age and file age remain separate, so a recently touched file cannot make an old decision appear fresh.

## Utility Components

Each resolved sleeve policy independently weights data integrity, signal conviction, independent consensus, regime alignment, post-cost edge, execution quality, portfolio fit, risk headroom, evidence maturity, and long-term alignment. Family-specific floors and market-quality ceilings are applied to the same ten stages. The score is a ranking aid, not predicted return, deployable capital, or permission to trade.

Qualification requires all of the following:

- Fresh and sufficiently complete point-in-time evidence.
- A real directional intent with minimum conviction.
- Signed consensus and regime support.
- Explicit lower-confidence-bound edge above realistic costs.
- Execution and portfolio feasibility.
- Non-bypassable controls clear.
- Candidate-bound post-cost evidence maturity.
- Utility above the versioned qualification floor.

## Capital Scalability

A high decision score does not establish that a strategy can absorb one million dollars. Capital-scale evidence additionally requires size-dependent market-impact curves, candidate-bound post-cost out-of-sample outcomes, capacity stress tests, liquidity and drawdown evidence, concentration controls, and staged canary validation. The sidecar never infers a maximum deployable amount from displayed depth or a model score.

## Paper Contract

The active hook runs after the existing broker-truth, profitability, portfolio, execution, and risk controls and before idempotency and queue publication. Its authority is monotonic:

- An existing HOLD remains HOLD.
- BUY can remain BUY, be downsized, or become HOLD; it cannot become SELL.
- SELL can remain SELL, be downsized, or become HOLD; it cannot become BUY.
- Output quantity cannot exceed input quantity.
- An explicit nonpositive post-cost lower confidence bound is vetoed.
- Missing or point-estimate-only edge may receive only the versioned bounded paper evidence-probe cap after every required data, consensus, execution, portfolio, and risk stage passes.
- Missing or proxy-only quantitative evidence remains explicit without automatically suppressing a bounded paper observation; direct adverse evidence applies the lower of the existing cap and the quantitative-evidence cap.
- No minimum quantity is invented, so a zero-sized or blocked intent cannot be resurrected.
- The control has no live-mode mutation, submission, or promotion authority.

## Live Parity Contract

The live path does not run a second or looser strategy thesis. Promotion and live execution consume the same resolved policy receipt and evaluation that produced the bounded paper intent. The execution lane fails closed when the receipt is absent, stale, changed, mismatched, execution-ineligible, or inconsistent with the evaluation digest, paper output action, or paper output quantity.

Live is deliberately stricter than paper:

- The exact decision-family ingestion route must have a valid binding and delivery receipt; paper and live capability coverage are reported separately.
- Paper needs the decision-forming capability tier, while live additionally needs the family enrichment tier, higher route-quality floor, and required independent failover evidence.
- Every stage from data qualification through quality priority must pass.
- The candidate must be `qualified_shadow_candidate` under its sleeve policy.
- A positive post-cost lower-confidence-bound edge is mandatory; point estimates and paper evidence probes are rejected.
- The evaluation must be within the versioned freshness window.
- The exact strategy-definition hash must be current and complete.
- Every family-required quantitative axis must have a direct, passing measurement; proxy-only evidence is rejected.
- The proposed action must be consistent with fresh account positions, the strategy shorting policy, and any linked-leg or defined-risk requirements.
- The live output cannot reverse direction or exceed the paper-authorized quantity.
- Existing promotion, broker-truth, account, risk, canary, production-excellence, and operator-release firewalls still apply after this check.
- This decision-flow layer never submits an order or grants live release. It only supplies an additional fail-closed veto or quantity cap to the execution path.

## Soak Boundary

Canonical behavior lives in `core/institutional_decision_flow.py` and `config/institutional_decision_flow_v1.json`, both covered by production-candidate fingerprints. The compatibility package under `shadow_research/**` remains outside candidate scopes and powers only the detached evidence sidecar.

Activation is recorded as an accepted in-soak hardening event. The headline cumulative soak history and all documented prior hours remain intact. Changed strategy, execution, risk, promotion, and operations scopes begin a new honest clean evidence segment; prior hours are not relabeled as proof of behavior that did not yet exist. The uninterrupted current-candidate window remains the authority for eventual promotion credit.

## Evidence

- Policy: `config/institutional_decision_flow_v1.json`
- Canonical evaluator, sleeve resolver, paper/live controls, and receipt guard: `core/institutional_decision_flow.py`
- Paper integration: `scripts/run_shadow_training_loop.py`
- Decision-aligned ingestion policy: `config/sleeve_ingestion_routing_v2.json`
- Route resolver and signed delivery receipts: `core/collector_capability_routing.py`
- Promotion and execution enforcement: `core/execution_lane_pipeline.py`
- Sidecar compatibility facade: `shadow_research/institutional_decision_flow/evaluator.py`
- Runner: `shadow_research/institutional_decision_flow/runner.py`
- Latest report: `governance/research/institutional_decision_flow/latest.json`
- Compact history: `governance/research/institutional_decision_flow/history_YYYYMMDD.jsonl`

Only independent, positive, candidate-bound, post-cost, capacity-aware results can support later promotion. The controls improve consistency and fail-closed enforcement; they do not guarantee profitability. Live execution remains a separate human-governed promotion decision and is still disabled until its independent gates clear.

## Candidate-Bound Quantitative Challengers

The repository already implements or governs clustered samples, block-bootstrap confidence bounds, deflated Sharpe, false-discovery control, probability of backtest overfitting, risk of ruin, meta-labeling, conformal abstention, residual alpha, changepoints, execution-cost decay, crowding, cross-impact, robust optimization, and tail models. Eight additional concepts now run as bounded, deterministic, candidate-bound research challengers:

- Always-valid sequential inference using e-values or sequential probability-ratio tests, so repeated monitoring does not silently inflate false discoveries.
- Hansen's superior predictive ability test or White's Reality Check for strategy-family selection beyond the existing false-discovery and PBO controls.
- Probabilistic Sharpe and Bayesian posterior utility for uncertainty-aware comparisons between candidates with unequal histories.
- Drawdown-constrained Kelly or risk-sensitive utility sizing, evaluated only as a diagnostic challenger to the existing fractional-Kelly and risk caps.
- Entropy pooling for auditable scenario views and stress-conditioned portfolio weights.
- Explicit optimal-stopping models for entries, exits, and event windows, judged against simpler time-stop and decay baselines.
- Combinatorial purged cross-validation and triple-barrier/meta-label variants where current event labeling does not already provide equivalent leakage controls.
- Online portfolio selection and transaction-cost-aware expert aggregation as a paper counterfactual to the current allocator.

Canonical code lives in `core/quantitative_challengers.py`; its authority contract is pinned in `config/quantitative_challengers_v1.json`; and `scripts/quantitative_challenger_report.py` publishes `governance/research/quantitative_challenger_latest.json`. Run `./scripts/ops/opsctl.sh quantitative-challengers --json` to refresh it. A sleeve receives compact status metadata only when the report candidate and cutoff match the active candidate-forward paper-performance window. The displayed denominator is all eight methods, including the two cross-profile methods.

These remain research hypotheses, not promised upgrades. They cannot change an action, quantity, allocation, label, candidate, promotion decision, or paper/live order. Each must earn any future authority through a separately reviewed policy change backed by point-in-time data, leakage-safe validation, realistic costs, independent outcomes, capacity tests, and champion-challenger evidence. An `8/8` implementation count only means the methods exist; unavailable or unsupported candidate evidence stays explicit, and adding vocabulary or another bot does not increase expected profitability by itself.
