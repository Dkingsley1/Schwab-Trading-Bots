# Use Mode Compliance Guardrails

Generated policy: `config/use_mode_compliance_policy_v1.json`

Runtime artifact: `governance/health/use_mode_compliance_guard_latest.json`

Command:

```bash
./scripts/ops/opsctl.sh use-mode-compliance --json
```

## Operating Boundary

The default use mode is `personal`. In this mode the system grades whether guarded paper trading, data collection, auth, storage, process fanout, read-only live-execution boundaries, and labeled profitability evidence are clean enough for unattended personal use.

The next personal-use tier after ordinary production hardening is `operator_grade_personal_autonomy`. It is not a new live-money permission and it is not commercial clearance. It is a stricter private-use bar that requires base personal A+, the A+ operating packet, unattended soak readiness, source mutation cleanliness, production-flow smoke, autonomy/recovery score, disaster recovery and blackstart readiness, managed data-plane recovery, locked live-money boundaries, clean commercial personal boundary, and security/privacy runtime evidence.

Commercial, customer-facing, or money-management use is not inferred from a green personal posture. Any environment flag for paid signals, model portfolios, customer accounts, customer funds, customer order execution, custody, copy trading, performance marketing, testimonials, futures or derivatives advice, or commodity-pool behavior triggers explicit review blockers.

This guard does not provide legal advice, does not approve commercial launch, and never enables live execution.

## Personal Tier After Production

- `A+`: guarded paper trading, data collection, auth, storage, process fanout, read-only execution boundary, and profitability labeling are clean enough for personal unattended use.
- `operator_grade_personal_autonomy`: the system also proves deeper autonomy surfaces are green and can explain source cleanliness, CI/policy smoke, soak safety, DR/blackstart, recovery, security/privacy, and live-money lock boundaries.
- `near_operator_grade_personal_autonomy`: base personal A+ is intact, but one or more deeper autonomy controls still needs evidence before calling the personal posture beyond-production.

## Commercial Expansion Tripwires

- `INVESTMENT_ADVICE_ENABLED`, `PAID_SIGNALS_ENABLED`, or `MODEL_PORTFOLIO_ENABLED` require investment-adviser review evidence.
- `CUSTOMER_ORDER_EXECUTION_ENABLED`, `CUSTOMER_ACCOUNTS_ENABLED`, or `COPY_TRADING_ENABLED` require broker-dealer review evidence.
- `CUSTOMER_FUNDS_ENABLED` or `CUSTODY_ENABLED` are hard-blocked until a registered, reviewed custody/customer-funds program exists.
- `FUTURES_OR_DERIVATIVES_ADVICE_ENABLED` or `COMMODITY_POOL_ENABLED` require commodity-advice review evidence.
- `PERFORMANCE_MARKETING_ENABLED` or `TESTIMONIALS_ENABLED` require marketing-rule review evidence.

## Sources

- SEC Investment Adviser Marketing: https://www.sec.gov/resources-small-businesses/small-business-compliance-guides/investment-adviser-marketing
- SEC Marketing Compliance FAQs: https://www.sec.gov/rules-regulations/staff-guidance/division-investment-management-frequently-asked-questions/marketing-compliance-frequently-asked-questions
- SEC Investment Adviser Definition: https://www.sec.gov/interps/legal/slbim11.htm
- FINRA Broker-Dealer Registration: https://www.finra.org/registration-exams-ce/broker-dealers
- CFTC Intermediaries, CTA, and CPO: https://www.cftc.gov/IndustryOversight/Intermediaries/index.htm
