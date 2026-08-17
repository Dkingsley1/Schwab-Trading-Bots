# Hierarchical Bot Organization

## Purpose

The bot fleet is organized as a deterministic hierarchy:

`sleeve -> sub-sleeve -> horizon/multi-axis regime cohort -> role`

The hierarchy makes ownership, evidence, resource use, correlated influence, and admission decisions inspectable. It does not grant paper or live execution authority, alter current runtime decisions, or claim that organization guarantees profitability.

## Canonical Sources

- Policy: `config/bot_organization_v1.json`
- Classifier: `core/bot_organization.py`
- Regime taxonomy and compatibility scorer: `core/regime_taxonomy.py`
- Shadow ensemble: `core/hierarchical_ensemble.py`
- Control and artifact writer: `scripts/ops/bot_organization_control.py`
- Health evidence: `governance/health/bot_organization_latest.json`
- Complete generated hierarchy: `governance/bot_organization/bot_hierarchy_latest.json`

The master registry remains the authority for bot lifecycle and execution flags. The generated hierarchy is an organizational projection and must never mutate the registry automatically.

## Hierarchy

### Sleeve

A sleeve is the broad trading or operating mandate, such as `equity_core`, `options_flow`, `intraday_aggressive`, or `system_governance`. Explicit registry metadata wins over tags, module literals, catalog categories, and policy fallbacks.

### Sub-Sleeve

A sub-sleeve is an economically or operationally coherent family, such as `trend_and_momentum`, `mean_reversion`, `relative_value`, `volatility_and_convexity`, `execution_and_liquidity`, or `data_and_model_governance`.

Capital, evidence, correlation, and quarantine should be assessed at this level before individual bot votes are combined. A large number of related bots must not be treated as an equal number of independent ideas.

### Cohort

A cohort binds horizon to the primary values from scope-appropriate regime axes. Market-signal cohorts use direction, volatility, liquidity, and event phase; hybrid cohorts add operational posture; operational-control cohorts use operational posture without pretending that a system state is a market state. Raw preferred-regime labels remain visible for review, while stable profile IDs and composite cohorts prevent a single broad label from erasing useful distinctions.

## Multi-Axis Regime Contract

The versioned taxonomy separates nine dimensions:

1. Market direction.
2. Volatility state.
3. Liquidity state.
4. Macro state.
5. Rates and credit state.
6. Correlation state.
7. Event phase.
8. Market session.
9. Operational state.

Each dimension records values, provenance, confidence, and matched evidence. `unknown` means evidence is missing, `any` is a deliberate wildcard, and `not_applicable` means the axis does not belong to that bot's scope. Those states are never collapsed into one another. Explicit `regime_axes` metadata wins over legacy preferred-regime labels, followed by literal module metadata, policy rules, and an explicit unknown fallback.

Axis coverage measures how much required metadata is known. Specificity separately measures how much is concrete rather than wildcarded. The health artifact reports both, along with scope counts, profile counts, per-axis value distributions, and bounded review reasons.

### Scenario Partitions

Routers, modelers, and control bots that legitimately span several contexts must declare `regime_scenarios` instead of flattening every context into one multi-valued profile. Each scenario has a unique ID, one explicit scope, explicit axis values, and its own independently reviewable profile. The v1 partition contract allows 2 through 12 scenarios and at most two values per axis in one scenario.

Compatibility evaluates every declared scenario, chooses the highest compatible score with scenario ID as the deterministic tie-breaker, and reports the selected scenario plus all alternatives. No match excludes the shadow vote and reports the failure. Duplicate IDs, missing axes, mixed scopes, excessive breadth, or malformed declarations fail closed and block organization health until repaired.

Platform-organ bots are operational controls even when their implementation role is named `signal_sub_bot`. Their runtime modes such as normal collection, resource pressure, backlog drain, halt review, and stress replay remain separate operational scenarios; they are not inferred as directional market regimes.

### Role

Roles separate signal generation from evidence, risk, execution simulation, evaluation, coordination, and shared services. Role separation prevents operational observers from accidentally receiving alpha voting authority.

## Classification Contract

Classification precedence is deterministic:

1. Explicit registry fields.
2. Structured registry tags.
3. literal `BOT_SPEC` metadata parsed from source without importing or executing modules.
4. The existing catalog category.
5. A declared policy fallback.

Every assignment records field-level provenance and a confidence score. Low-confidence legacy rows enter a bounded review queue; they are not silently represented as manually verified metadata. Duplicate identities, missing required levels, incomplete coverage, unsafe policy settings, and hard resource-cap breaches fail the control closed.

### Shared regime metadata access

Every organized bot receives a `regime_metadata_access_v1` receipt that points to the
versioned axis catalog and lists the axes readable for its scope. Runtime callers can use
`build_regime_metadata_view()` to provide a validated, provenance-backed context packet for
collection, training, and shadow evaluation. The access layer is read-only, fails closed on
invalid context, and cannot create paper or live execution authority.

Access and preference maturity are deliberately separate. A legacy bot with unknown regime
preferences can observe the current regime metadata, but the system does not invent a
preference or treat access as proof that the bot is compatible with that regime. The
organization control requires 100 percent registry access coverage and reports remaining
preference debt independently.

## Hierarchical Voting

The initial ensemble is shadow-only. It aggregates in this order:

1. Bot votes are confidence weighted and individually capped.
2. Votes in the same correlation cluster are averaged before their weight is counted.
3. Correlation clusters are aggregated into sub-sleeves.
4. Sub-sleeves are capped and aggregated into sleeves.
5. Sleeves are capped and aggregated into a research recommendation.

The ensemble abstains when source assignments are missing, confidence is too low, independent sub-sleeve diversity is insufficient, or cross-cell disagreement is excessive. Adding duplicate bots to a correlation cluster cannot manufacture additional cluster weight.

When an explicit regime context is supplied, the ensemble compares each profile axis independently, records weighted compatibility evidence, excludes malformed or incompatible profiles, and can discount wildcard evidence. An absent context preserves the existing shadow behavior. The compatibility path is fail-closed and remains research-only.

The output is a research-only `BUY`, `SELL`, or `HOLD` recommendation. It creates no order payload and has no paper or live execution authority.

## Master Evidence Plane

`docs/architecture/MASTER_GRANDMASTER_EVIDENCE_V2.md` defines the separate v2 evidence plane built on this hierarchy. It emits one bounded sleeve-master packet per canonical sleeve and a Grand Master rollup across fresh regime, paper-truth, profitability, source, runtime, position, and execution evidence. The layer does not replace the existing decision path and has no order, allocation, registry-mutation, or automatic-promotion authority.

## Capacity And Admission

The policy defines soft and hard shadow-voter limits per cell, a global voter ceiling, and single-flight training limits. Soft breaches create ranking and parking work. Hard breaches block new admission and excess shadow voting.

A new bot must identify its sleeve and sub-sleeve, fill a documented capability gap, improve locked out-of-sample results after stressed costs, satisfy multiple-testing and correlation limits, fit within resource budgets, and receive human registry admission. Bot count alone is never an objective.

## Profitability And Scale Plane

`config/bot_profitability_scalability_v1.json` and
`core/bot_profitability_scalability.py` add an execution-free control plane over the
hierarchy. It attributes paper outcomes to the constituent bot IDs recorded on each
decision, deduplicates by decision identity, and keeps historical diagnostics separate
from evidence collected after the current production-candidate cutoff.

The profitability half learns regime preferences only from sufficiently sampled positive
post-cost outcomes, ranks forward evidence using expectancy, conservative lower bounds,
drawdown, turnover, confidence, and persistence, measures marginal value inside correlation
clusters, consumes independent execution calibration, enforces the existing holdout and
multiple-testing firewall, emits lifecycle advice, and publishes capacity curves. Missing
evidence receives no credit and no proposed capacity.

The scalability half treats the registry as a catalog instead of a process list, publishes
a bounded top-K activation plan, requires one immutable shared-feature snapshot, checks
worker and queue budgets, verifies checkpoints and decision identities, checks hot/cold
storage routing, and supplies a bounded lazy model cache with TTL and memory-pressure
eviction. The activation plan is advisory: it cannot change registry flags, allocate
capital, or grant paper/live execution authority.

Control implementation and economic evidence have separate grades. All 16 controls can be
structurally A+ while the evidence grade remains below A+ during collection. That split is
intentional and prevents configuration quality from being presented as proven profitability.

## Rollout

1. Keep classification and health reporting active during the current soak.
2. Review legacy assignments and improve explicit registry metadata.
3. Replay the hierarchical ensemble beside the existing flat decision path.
4. Require measurable post-cost improvement, lower correlated concentration, and no safety regression.
5. Allow paper routing only through an explicit reviewed change after the evidence window is reset as required.
6. Keep live execution locked until the separate live-money promotion contract passes and the operator releases a microscopic canary.

## Operations

Run:

```bash
./scripts/ops/opsctl.sh bot-organization --json
./scripts/ops/opsctl.sh bot-profitability-scalability --json
./scripts/ops/opsctl.sh master-grandmaster-evidence --json
```

The control is included in bounded runtime artifact refreshes, the freshness SLO, the runtime dashboard, CI, source-mutation protection, and exclusive control-surface ownership.
