# Sleeve Strategy Specialization

## Purpose

The specialization layer gives each active runtime and collection strategy a stable identity and a complete research contract. It fixes a prior attribution gap where many paper decisions were recorded only under broad master-bot names even though the sleeve catalog contained hundreds of named strategies.

## Canonical Sources

- `config/sleeve_strategy_contracts_v1.json` owns objectives, aliases, conservative defaults, strategy additions, candidate binding, and the zero-authority contract.
- `config/sleeve_strategy_expansion.json` remains the source catalog for existing sleeve and strategy membership.
- `core/sleeve_strategy_specialization.py` materializes deterministic IDs, receipts, runtime identities, and read-only counterfactual rankings.
- `scripts/sleeve_strategy_specialization_report.py` joins current candidate-forward paper evidence and publishes lifecycle states.
- `scripts/strategy_library_query.py` provides bounded family, sleeve, tier, regime, and evidence-verdict searches over the generated artifacts.

## Identity And Attribution

Named strategies use `sleeve::{sleeve_id}::{strategy_name}::v1`. A decision receives a named strategy only when that strategy is the actual runtime source. Broad master or Grand Master decisions use `ensemble_champion`; paper portfolio consensus uses `portfolio_consensus`; unmapped runtime bots use an explicit `runtime_challenger_*` identity. Counterfactual rankings are metadata only and never receive trade credit.

## Objective-Aware Evidence

- Directional, event, macro, digital-asset, basis, volatility, market-neutral, and execution strategies are evaluated on their contract-specific post-cost outcome and benchmark.
- Hedge strategies are evaluated on portfolio drawdown and tail-loss reduction net of carry and false positives, not forced standalone profit.
- Cash and conservative strategies are evaluated on capital preservation net of opportunity cost.
- Infrastructure strategies are control-only and never receive a trading-profit objective.

## Twelve-Thousand-Strategy Library

The policy deterministically materializes exactly `12,000` strategy hypotheses across `111` sleeves. Every sleeve receives `108` or `109` strategies, satisfying the minimum of `100` without giving every process 100 simultaneous jobs. The existing `879` catalog and curated contracts remain the hot runtime catalog. The remaining `11,121` hypotheses are `cold_research`: they are generated only in the research report, consume no runtime strategy slot, and have no training, action, sizing, allocation, promotion, or live-order authority.

Each strategy includes a plain-language summary, edge hypothesis, signal family, confirmation requirement, ideal and hostile regimes, required inputs, expected failure modes, benchmark question, objective, horizon, costs, capacity, and lifecycle rules. Objective-specific archetypes are crossed with bounded confirmation overlays, so an options, macro, crypto, market-neutral, execution, hedge, cash, or control sleeve receives strategies appropriate to that objective rather than generic copies.

## Consolidated Family Catalog

The primary human-facing artifact is `governance/research/sleeve_strategy_families_latest.json`. It presents the same `12,000` identities as `1,989` canonical records: `879` unchanged native hot identities and `11,121` cold variants nested under `1,110` sleeve-and-archetype parent families. No strategy is deleted, renamed, activated, or merged at runtime.

Parents own the shared thesis, signal family, inputs, label, horizon, benchmark, lifecycle, and failure contract. Children retain the exact strategy ID, condition overlay, confirmation, receipt hash, point-in-time regime annotation, and evidence verdict. Evidence remains child-specific and cannot be pooled to make a parent appear validated. Every cold family advertises all `12` supported condition types; a supported-only condition is a research capability, not a fabricated strategy row or evidence result.

Use `./scripts/ops/opsctl.sh strategy-families --sleeve crypto_spot --objective digital_asset_alpha --limit 40` for the consolidated view. Use `strategy-library` when the exact 12,000 child rows are required.

## Regime Adaptation

The current regime comes from `governance/health/regime_control_plane_latest.json`. Strategy relevance is recomputed as `aligned`, `neutral`, `guarded`, or `unknown` whenever the report or runtime metadata is refreshed. A fresh but low-confidence regime may annotate and rank; only a fresh `ready` regime can make an aligned cold hypothesis eligible for reviewed admission. Stale, missing, thin, or unsupported regime evidence blocks cold activation while preserving the existing hot paper runtime.

Regime adaptation changes ranking, research admission, and evidence segmentation only. It cannot rewrite strategy history, mutate an action or quantity, change risk limits, promote a candidate, or authorize a live order. Future live evaluation fails closed without fresh aligned regime evidence.

## Good And Bad Verdicts

The scorecard distinguishes `validated_good`, `promising_unconfirmed`, `mixed_watch`, `weak`, `retirement_candidate`, `insufficient_evidence`, `cold_untested`, `objective_evidence_pending`, and `control_only`. `Good` requires candidate-bound positive clustered confidence after costs and objective-specific risk checks. `Bad` requires mature adverse evidence; an untested strategy is unknown, not bad. Hedge and capital-preservation strategies require portfolio-contribution evidence rather than standalone profit.

Run `./scripts/ops/opsctl.sh strategy-library --sleeve crypto_spot --regime-relevance aligned --limit 40` to inspect exact variants. Use `--good` or `--bad` for evidence verdicts. Until sufficient candidate-forward observations exist, those filters may correctly return no strategies.

## Lifecycle

The hot report uses `parked_candidate`, `probation`, `watch`, `validated_candidate`, `demotion_review`, `retirement_review`, and `control_only`; the generated catalog adds `cold_untested`. A state is descriptive evidence, not promotion authority. Missing or mismatched candidate binding parks the strategy. No historical candidate or lifetime fallback is allowed.

## Candidate-Bound Paper Scaling

`scripts/ops/paper_profitability_control.py` is the sole owner of the executable paper scaling contract. It joins the specialization identity to current-candidate post-cost evidence and publishes separate controls for every profile and every strategy. The specialization layer supplies identity and objective metadata but has no sizing authority of its own.

New `BUY` entries progress through bounded tiers: `paper_probation` at `0.25`, independent-evidence collection at `0.50`, validated-evidence buildup at `0.75`, a validated baseline at `1.00`, and independently supported tiers at `1.05` and `1.10`. Advancement requires increasing candidate-bound sample, independent-day, independent-symbol, effective-sample, profit-factor, drawdown, deflated-Sharpe, and current-regime evidence. Missing source readiness, stale or mismatched candidate binding, incomplete specialization, control-only objectives, or an active quarantine fails closed for new entries.

The contract is entry-only. `SELL` and reduce-only paths always bypass profitability deweighting so a weak or quarantined strategy can reduce risk. The paper-debt cap, portfolio and execution budgets, hierarchy and correlation caps, turnover controls, contract-valuation checks, and order-plan limits remain authoritative and can only reduce the resulting size. The maximum candidate-evidence multiplier is `1.10`; there is no live-order authority and no claim of future profitability.

## Safety And Soak

The specialization layer is candidate-bound and cannot create or reverse intent, increase quantity, allocate capital, change training labels, grant promotion, or submit an order. The separate profitability controller may only multiply an already-authorized paper `BUY` inside the contract above. Paper collection remains available if the metadata layer is unavailable. Future live evaluation fails closed when the required contract receipt or candidate binding is missing. This accepted hardening change preserves cumulative segmented soak history while starting a new clean candidate segment for the changed strategy and execution scopes.
