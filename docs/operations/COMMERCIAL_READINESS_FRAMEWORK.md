# Commercial Readiness Framework

Runtime command:

```bash
./scripts/ops/opsctl.sh commercial-readiness --json
```

Runtime artifacts:

- `governance/health/commercial_readiness_control_latest.json`
- `exports/reports/operator/commercial_readiness_packet_latest.md`

## Seven Sections

The commercial framework evaluates these sections:

1. `commercial_use_modes`: maps the system to a specific product mode instead of one vague commercial flag.
2. `registration_review_gates`: requires explicit review evidence for legal, compliance, adviser, broker-dealer, commodity, marketing, privacy/security, and terms/disclosure gates.
3. `marketing_claim_control`: blocks public/performance claims unless claims are approved, substantiated, labeled, and reviewed.
4. `customer_funds_hard_blocks`: keeps customer funds, custody, customer order execution, and copy trading behind external reviewed/registered-program evidence.
5. `commercial_evidence_packets`: requires a release packet with business-mode, approvals, methodology, claims, funds/custody attestation, security/privacy, incident response, retention, service-provider, and release-approval artifacts.
6. `self_awareness_expansion`: surfaces commercial mode, claim risk, funds boundary, adviser/broker/CTA-CPO boundaries, release readiness, and security/privacy posture into the system self-model.
7. `security_privacy_layer`: checks security audit, secret scan, redaction, privacy/security review, and customer-data control evidence.

## Boundary

This framework is operational guardrailing, not legal advice. It does not approve commercial release and never grants live execution authority.

## Sources

- SEC Investment Adviser Marketing: https://www.sec.gov/resources-small-businesses/small-business-compliance-guides/investment-adviser-marketing
- SEC Marketing Compliance FAQs: https://www.sec.gov/rules-regulations/staff-guidance/division-investment-management-frequently-asked-questions/marketing-compliance-frequently-asked-questions
- SEC Investment Adviser Definition: https://www.sec.gov/interps/legal/slbim11.htm
- FINRA Broker-Dealer Registration: https://www.finra.org/registration-exams-ce/broker-dealers
- FINRA Best Execution / Order Routing: https://www.finra.org/rules-guidance/guidance/reports/2026-finra-annual-regulatory-oversight-report/best-execution
- CFTC Intermediaries / CTA / CPO: https://www.cftc.gov/IndustryOversight/Intermediaries/index.htm
- FTC Safeguards Rule: https://www.ftc.gov/legal-library/browse/rules/safeguards-rule
- FTC Safeguards Rule Business Guide: https://www.ftc.gov/business-guidance/resources/ftc-safeguards-rule-what-your-business-needs-know
