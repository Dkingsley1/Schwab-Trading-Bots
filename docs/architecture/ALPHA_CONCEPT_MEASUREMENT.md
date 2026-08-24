# Alpha Concept Measurement Laboratory

## Purpose

The laboratory gives the platform a finite, architecture-specific alpha ontology and sixteen deterministic research measurements. It closes named measurement gaps without creating thousands of duplicate bots or implying that every possible market idea can be enumerated.

The canonical owners are:

- `config/alpha_concept_registry_v1.json`: ontology, engine floors, routes, conditional requirements, references, and zero-authority policy.
- `core/alpha_concept_engine.py`: deterministic estimators.
- `scripts/ops/alpha_concept_report.py`: candidate binding, evidence routing, grades, collection priorities, JSON evidence, and the operator report.
- `config/sleeve_alpha_toolbox_v1.json` and `scripts/ops/sleeve_alpha_toolbox_control.py`: explicit evidence-axis routes for every declared sleeve, structural coverage receipts, and candidate evidence gaps.

## Sixteen Engines

1. Information-coefficient term structure: Pearson IC, rank IC, per-period IC stability, ICIR, regime slices, and horizon decay.
2. Effective breadth and transfer coefficient: eigenvalue participation-ratio breadth, lag-adjusted independent periods, and forecast-to-implemented-weight transfer.
3. Hierarchical Bayesian skill: empirical-Bayes shrinkage across sleeves or strategy families with posterior positive-edge probabilities.
4. Stable feature discovery: seeded subsample stability selection with sign consistency and an explicit statement that no Model-X knockoff guarantee is claimed.
5. Economic alpha decomposition: static factor premia, unexplained selection alpha, execution drag, and an optional non-double-counted timing diagnostic.
6. Factor-neutral residualization: explicit multi-factor ridge residualization, factor loadings, residual correlations, intercept uncertainty, and residual return output.
7. Causal transportability: leave-one-environment-out partially linear DML with robust uncertainty, environment effects, heterogeneity, and assumptions; it does not claim causality is proven.
8. Capacity and impact: an explicit square-root impact surface over tested notionals with no unknown-cost defaults or sizing authority.
9. Execution alpha attribution: decision, arrival, fill, midpoint, fee, and markout economics under a documented sign convention.
10. Point-in-time security master: effective-date, stable-identifier, symbol-reuse, corporate-action, delisting, ambiguity, and observation-resolution audits.
11. Split-conformal residual calibration: time-ordered calibration and evaluation windows, empirical coverage, and interval sharpness with no random split.
12. Sequential change-point stability: bounded two-sided Page-CUSUM diagnostics per candidate-bound group with explicit recent-drift flags.
13. Residual redundancy graph: synchronized cross-sleeve return correlation, redundant pairs, and independent connected components.
14. Regime-conditional robustness: per-regime post-cost means and 95% lower confidence bounds with a minimum supported-regime ratio.
15. Cost-stress survival: monotonic fee, spread, slippage, and impact stress scenarios that require conservative net edge to remain positive.
16. Active-learning value of information: cost- and pressure-aware ranking of unresolved measurement gaps; it creates neither labels nor samples.

## Evidence Contract

The report consumes the accepted candidate and candidate-filtered paper-performance watermark. Optional engine inputs belong in `governance/research/alpha_concept_inputs_latest.json` and must carry the same candidate ID plus a timestamp at or after the candidate cutoff. A missing, stale, pre-cutoff, or mismatched packet is ignored.

The report publishes four separate grades:

- **Implementation**: whether all sixteen deterministic engines exist.
- **Catalog routing**: whether every canonical concept has a declared local owner.
- **Candidate evidence**: whether the fifteen economic measurements have sufficient identity-bound inputs. Active-learning readiness is excluded.
- **Economic support**: whether mature candidate evidence passes each estimator's configured diagnostic.

An implementation or routing `A+` is not evidence of alpha. An economic `F` during a fresh candidate simply means the system is collecting; it cannot be upgraded by metadata, synthetic samples, lifetime pooling, or pre-candidate history.

## Runtime Contract

The report runs after paper performance, quantitative challengers, and alpha lifecycle refresh. It is managed paper-soak advisory debt, runs outside the market decision hot path, and exposes only a compact summary to the self-model. It cannot:

- change an action, position transition, quantity, or allocator weight;
- write a training label;
- submit a paper or live order;
- promote a strategy or candidate;
- mutate candidate identity or rewrite soak history;
- guarantee future profitability.

Use:

```bash
./scripts/ops/opsctl.sh alpha-concepts --json
```

Human-readable output is written to `exports/reports/operator/alpha_concept_report_latest.md`.

Every sleeve receives an explicit route for each quantitative evidence axis required by its resolved strategy family. Inspect structural routing and organic evidence gaps with:

```bash
./scripts/ops/opsctl.sh sleeve-alpha-toolbox --json
```

The toolbox route is not evidence by itself. A sleeve remains in collection until current-candidate, point-in-time, post-cost observations satisfy the routed diagnostics.
