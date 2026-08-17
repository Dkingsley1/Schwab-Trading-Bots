from __future__ import annotations

from scripts.ops.tax_regulation_update import _rate_shape, parse_revenue_procedure


def _ordinary_table(number: int, thresholds: list[int], rates: list[int]) -> str:
    lines = [f"TABLE {number} - Filing status", "If Taxable Income Is:                                                    The Tax Is:"]
    lines.append(f"Not over ${thresholds[0]:,}".ljust(72) + f"{rates[0]}% of taxable income")
    for index, upper in enumerate(thresholds[1:], start=1):
        lines.append(f"Over ${thresholds[index - 1]:,} but not over".ljust(72) + f"plus {rates[index]}%")
        lines.append(f"${upper:,}".ljust(72) + "of the excess")
    lines.append(f"Over ${thresholds[-1]:,}".ljust(72) + f"plus {rates[-1]}%")
    return "\n".join(lines)


def _revenue_procedure_text() -> str:
    rates = [10, 12, 22, 24, 32, 35, 37]
    return "\n".join(
        [
            ".01 Tax Rate Tables. For taxable years beginning in 2027, the tax rate tables are as follows:",
            _ordinary_table(1, [25000, 102000, 214000, 408000, 518000, 777000], rates),
            _ordinary_table(2, [18000, 68500, 107000, 204000, 260000, 648000], rates),
            _ordinary_table(3, [12600, 51200, 107000, 204000, 260000, 648000], rates),
            _ordinary_table(4, [12600, 51200, 107000, 204000, 260000, 389000], rates),
            ".03 Maximum Capital Gains Rate. For taxable years beginning in 2027, the maximum rates are as follows: "
            "Married Individuals Filing Joint Returns and Surviving Spouse $100,000 $620,000 "
            "Married Individuals Filing Separate Returns $50,000 $310,000 "
            "Heads of Household $67,000 $586,000 All Other Individuals $50,000 $551,000",
            ".04 Adoption Credit.",
            ".14 Standard Deduction. (1) In general. For taxable years beginning in 2027, the standard deduction amounts are as follows: "
            "Married Individuals Filing Joint Returns and Surviving Spouses $33,000 "
            "Heads of Households $24,800 Unmarried Individuals $16,500 "
            "Married Individuals Filing Separate Returns $16,500",
            ".15 Cafeteria Plans.",
        ]
    )


def test_revenue_procedure_parser_extracts_all_individual_tables() -> None:
    parsed = parse_revenue_procedure(_revenue_procedure_text(), tax_year=2027)
    assert parsed["ordinary_income_brackets"]["single"][0]["up_to_usd"] == 12600
    assert parsed["ordinary_income_brackets"]["married_filing_jointly"][-2]["up_to_usd"] == 777000
    assert parsed["preferential_capital_gain_brackets"]["head_of_household"][1]["up_to_taxable_income_usd"] == 586000
    assert parsed["standard_deduction_usd"]["married_filing_separately"] == 16500
    assert parsed["ordinary_income_brackets"]["qualifying_surviving_spouse"] == parsed["ordinary_income_brackets"]["married_filing_jointly"]


def test_rate_shape_ignores_inflation_threshold_changes_but_detects_rate_changes() -> None:
    parsed = parse_revenue_procedure(_revenue_procedure_text(), tax_year=2027)
    policy_a = {
        **parsed,
        "section_1256": {"long_term_fraction": 0.6, "short_term_fraction": 0.4},
        "net_investment_income_tax": {"rate": 0.038},
    }
    policy_b = {
        **parsed,
        "ordinary_income_brackets": {
            key: [{**row, "up_to_usd": (row["up_to_usd"] + 500 if row["up_to_usd"] else None)} for row in rows]
            for key, rows in parsed["ordinary_income_brackets"].items()
        },
        "section_1256": {"long_term_fraction": 0.6, "short_term_fraction": 0.4},
        "net_investment_income_tax": {"rate": 0.038},
    }
    assert _rate_shape(policy_a) == _rate_shape(policy_b)
    policy_b["ordinary_income_brackets"]["single"][0]["rate"] = 0.11
    assert _rate_shape(policy_a) != _rate_shape(policy_b)
