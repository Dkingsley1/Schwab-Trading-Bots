# Use Mode Compliance Guardrails

Generated policy: `config/use_mode_compliance_policy_v1.json`

Runtime artifact: `governance/health/use_mode_compliance_guard_latest.json`

Command:

```bash
./scripts/ops/opsctl.sh use-mode-compliance --json
```

## Operating Boundary

The default use mode is `personal`. In this mode the system grades whether guarded paper trading, data collection, auth, storage, process fanout, read-only live-execution boundaries, and labeled profitability evidence are clean enough for unattended personal use.

Commercial, customer-facing, or money-management use is not inferred from a green personal posture. Any environment flag for paid signals, model portfolios, customer accounts, customer funds, customer order execution, custody, copy trading, performance marketing, testimonials, futures or derivatives advice, or commodity-pool behavior triggers explicit review blockers.

This guard does not provide legal advice, does not approve commercial launch, and never enables live execution.

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
