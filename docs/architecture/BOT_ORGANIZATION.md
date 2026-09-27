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
- Operating definition contract: `config/bot_operating_definition_contract_v1.json`
- Reviewed operating bindings: `config/bot_operating_definitions_v1.json`
- Operating resolver: `core/bot_operating_definitions.py`

The master registry remains the authority for bot lifecycle and execution flags. The generated hierarchy is an organizational projection and must never mutate the registry automatically.

## Seven-Area Definition Audit

Two complementary specifications now cover the same seven areas and 32 fields.
`core/bot_operating_definitions.py` defines each registered component's actual
operating job. The original `definition_audit_contract` in the organization
policy and `core/bot_definition_contracts.py` retain the stricter standalone
trading mandate requirements; operating completion cannot satisfy that audit.
Every native organization refresh audits every registry row, including inactive,
malformed, duplicate, and source-missing rows. The audit does not import bot
modules, train models, contact brokers, traverse archives, or start workers.
Project-relative source reads reject symlink and external paths before target
probes. File and byte budgets are bounded; unread sources remain incomplete.

Four statuses must remain separate:

1. **Audit coverage:** how many registry records were examined. This can be 100%
   with no complete definitions.
2. **Definition completeness, with explicit scope:** operating definitions bind
   actual component jobs; standalone trading mandates require the fields below.
   Required reviewed source references resolve to their pinned hashes/symbols.
   Inferred labels, generic placeholders and unsupported applicability claims
   cannot satisfy completion. This is specification validation, not proof that
   an arbitrary formula is correct or that implementation follows the prose.
3. **Implementation verification:** actual behavioral conformance still needs
   independent tests and runtime traces. Merely declaring golden test cases does
   not execute or pass them; this static audit leaves conformance unverified.
4. **Economic evidence:** independently produced candidate-bound, objective-aware
   post-cost evidence. Accuracy, organization grade, source existence, complete
   definitions and fixture tests never earn economic or promotion credit.

| Area | Required Standalone Trading Mandate |
| --- | --- |
| Purpose | Existing role, falsifiable hypothesis, one primary family, objective. |
| Scope | Explicit universe, venue, session, decision interval and holding horizon in seconds, required inputs. |
| Decision rules | Exact output formula/schema, parameter values/units, entry/add/trim/exit/time-stop rules or exact owner delegation, normal and abstention golden cases. |
| Abstention | Concrete no-output conditions, maximum input age, supported regimes. |
| Boundaries | Owner, allowed outputs, forbidden actions, risk owner, dependencies. |
| Training | Target, numerical label horizon, point-in-time joins, split policy, candidate/trial/dataset/holdout/embargo contract. |
| Accountability | Objective-appropriate metric, benchmark, invalidation, trace fields, collect/evaluate/invalidate/retire transitions. |

The resolver recovers explicit fields from registry metadata and literal
`BOT_SPEC` declarations without overwriting either. Inferred family and label
values remain useful hints, visibly excluded from authored completeness. A
freshness SLO is an input-age requirement, not a fabricated decision interval;
a label maturity is not a guessed holding horizon. Catalog runner references
are inspected separately and do not prove individual-bot runtime binding.
Collection wrappers remain collection wrappers even when their names describe
strategies. Their helper returns metadata and does not implement those strategies.

The existing owner publishes operating completion under `definition_audit` in
`governance/health/bot_organization_latest.json`, source bindings and area results
under `definition_audit.records` in the native hierarchy, and a readable summary
at `exports/reports/operator/bot_definition_audit_latest.md`. Each record retains
the original `standalone_trading_mandate` with its field values, provenance and
gaps. Its aggregate remains under `standalone_trading_mandate_summary`. The
organization grade describes structure, not either definition scope or economics.

### Area Subsections

Each of the 32 existing required fields is also a named, numbered subsection.
The `subsections` map in `config/bot_operating_definition_contract_v1.json` owns
the titles. Stable field IDs remain unchanged; numbering follows the canonical
area/field order rather than JSON key order. No field is waived or duplicated.

| Area | Subsections |
| --- | --- |
| 1. Purpose | 1.1 Role And Responsibility; 1.2 Hypothesis; 1.3 Primary Family; 1.4 Objective |
| 2. Scope | 2.1 Input Universe; 2.2 Venue And Data Origin; 2.3 Session; 2.4 Decision Cadence; 2.5 Holding And Prediction Horizon; 2.6 Input Contract |
| 3. Decision Rules | 3.1 Output Rule; 3.2 Parameters And Units; 3.3 Position Management; 3.4 Acceptance And Abstention Cases |
| 4. Abstention | 4.1 No-Output Conditions; 4.2 Freshness Limits; 4.3 Regime Support |
| 5. Boundaries | 5.1 Ownership; 5.2 Allowed Outputs; 5.3 Forbidden Actions; 5.4 Risk Ownership; 5.5 Dependencies |
| 6. Training | 6.1 Learning Target; 6.2 Label Horizon; 6.3 Point-In-Time Joins; 6.4 Evaluation Splits; 6.5 Experiment Lineage |
| 7. Accountability | 7.1 Success Metrics; 7.2 Benchmarks; 7.3 Invalidation; 7.4 Traceability; 7.5 Lifecycle |

The per-bot CLI includes `subsection.number` and `subsection.title` beside each
field's unchanged `contract` and `value`. Native hierarchy records expose
`areas[area].subsections[field].complete`; health and Markdown reports show
fleet counts under `area_summary[area].subsections`. An invalid bot binding makes
all of that bot's operating subsections incomplete. A missing, extra, malformed
or changed subsection declaration cannot earn completion or silently rebind.
Titles do not add strategy behavior, prove field correctness or grant economic
evidence; the standalone trading audit retains its original requirements.

### Bot Process Definitions

`config/bot_process_definition_contract_v1.json` organizes each operating profile
into an authored dependency graph. `core/bot_process_definitions.py` expands it
from the already-validated registry/source binding, without importing bots or
running their programs. Each stage names inputs, outputs, owner, dependencies,
source references or an explicit gap, failure/retry semantics and the independent
completion evidence still required. Missing runtime retry evidence is stated as
unknown, not replaced with invented attempt counts.

The existing `--bot-definition BOT_ID --json` output includes the full `processes`
graph. Fleet records retain a compact `process_definition` summary and digest;
`process_summary` reports definition coverage, referenced stages and gaps. The
operator Markdown report lists the stages for each profile. The process policy
digest and resolver source are pinned by explicit operating materialization.
Routine refresh rejects drift, missing policy and symlinked policy routes.

Authored order is not verified execution order. A source reference is not a
successful invocation, a runtime receipt, a test pass or profitability evidence.
Registry-only slots retain no implementation; collection wrappers remain
metadata-only. No new worker, retry dispatcher, trading rule or authority is added.

### Complete Operating Jobs

The four operating profiles preserve the registry role and implementation:

- `collection_wrapper`: literal bot metadata and collection/training prerequisites;
  its current helper returns descriptions, not a fitted model or strategy orders.
- `registry_declared_slot`: a declared research objective and input/label contract;
  no dedicated source implementation or runnable worker is claimed.
- `runtime_model_program`: source-defined runtime-data features, filters, labels,
  training arguments and artifact behavior; input freshness and quality are still
  independently verified by their existing owners.
- `synthetic_research_program`: source-defined simulation/offline training; its
  outputs are not represented as live-data or economic evidence.

All seven areas expand from an authored profile contract plus exact per-bot
registry values, literal source declarations, AST-inspected function signatures,
default values, calls and input keys. Shared trainer defaults and controller
hashes are published once under `shared_source_contracts`. References pin code
without executing it. Inspection identifies a source path, not tested behavior.
Event-driven invocations are explicit; sample windows are never converted into
guessed seconds. Position entry/add/trim/exit/time-stop are not owned by these
components; existing institutional and execution-lane controllers retain them.

Run `./scripts/ops/opsctl.sh bot-organization --bot-definition BOT_ID --json`
to expand a currently valid bot's 32 fields and inspect the exact source facts.
The persisted catalog stores compact binding receipts, avoiding another copy of
every registry field. The hierarchy remains a complete compact JSON snapshot.

Normal native refresh validates every pinned source, stable registry field,
dependency, definition resolver and shared owner. A new/removed identity, changed source, changed
policy, unknown program kind, unreadable source or execution authority keeps
operating completion false. Quality scores, P&L and active flags do not redefine
the job. Refresh never silently repins changed definitions or changes any flags.
This uses the existing platform refresh path, not a new automation or worker.

After reviewing implementation and definition changes, explicitly run
`./scripts/ops/opsctl.sh bot-organization --materialize-operating-definitions --require-definition-complete --json`.
Only this operator-invoked authoring flag replaces the pinned catalog. Scheduled
refresh and ownership-repair commands omit it. The normal
`--require-definition-complete` check returns exit code 2 for incomplete
**operating definitions**, not for absent standalone trading mandates.
This scope is included in JSON, CLI help and the operator report.

### Completing Standalone Trading Mandates

Fix the owning implementation first when a bot has no real rule implementation.
Do not fill a future trading mandate by renaming its collection-only behavior.
Then add a reviewed entry at
`definition_audit_contract.bot_definitions[existing_bot_id]` with exactly
`revision` and `areas`. Each area contains its named fields, each represented by
`{"value": ..., "references": [{"path": ..., "sha256": ..., "symbol": ...}]}`.
References identify existing project-relative Python symbols or JSON pointers
in independent owner files. Pin their actual SHA-256; a stale hash, missing
symbol or external path fails completion. Do not reference the containing policy
file itself with a self-referential hash. Reviewed fields cannot reassign a bot's
registry role or acquire authority.

`parameters` has `values` and `units`; a fixed rule with an empty parameter map
must explicitly set `no_tunable_parameters: true`. Golden cases have unique
`case_id`, `kind`, `inputs`, and `expected`, including both `normal` and
`abstention`. `experiment_contract` declares `candidate_binding`,
`trial_accounting`, `dataset_version`, `untouched_holdout`, and `embargo`.
Exact rule/delegation references still require independent conformance tests.

For actual non-market roles, inapplicable market/training fields may use
`{"not_applicable": true, "reason": "source-backed explanation"}` in a reviewed
record. This cannot waive their operating purpose, inputs, decision cadence,
output rules, abstention, ownership, tests, operational metric, trace or lifecycle.
An operational label never becomes a forward market-return label. Market-role
bots cannot bypass requirements by declaring themselves infrastructure.

Required trace fields are `bot_id`, `definition_sha256`, `candidate_id`,
`experiment_trial_id`, `feature_snapshot_id`, input/decision timestamps,
`rule_results`, `abstention_reason`, `output`, and `outcome_join_id`.
These are required definitions for future conformance checks, not a claim that
every current runtime already emits them. Definition digests exclude registry
accuracy/profitability labels and do not change execution-policy receipts.

Run `./scripts/ops/opsctl.sh bot-organization --json --require-trading-mandate-complete`
to publish the audit and return exit code 2 while any standalone mandate is incomplete.
Routine refresh remains the existing organization command/schedule. The strict
exit check is an audit acceptance check, not a new runtime gate or scheduler.

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
