import json
from copy import deepcopy
from datetime import date
from pathlib import Path

from core.collector_capability_routing import _evaluate_capability_proofs
from core.runtime_training_common import _enrich_runtime_observation
from scripts.collector_contracts import _source_status_metrics
from scripts.collect_decision_context_mesh import _build_plane, _symbol_context_features
from scripts.collect_public_financial_context import (
    PUBLIC_FINANCIAL_FEATURE_KEYS,
    PUBLIC_FINANCIAL_SYMBOL_FEATURE_KEYS,
    ROUTING_CONFIG_PATH,
    _apply_routing_taxonomy,
    _build_derived,
    _companyfacts_symbol_row,
    _health_source_rows,
    _isolated_source,
    _parse_ecb_csv,
    _parse_fdic_financials,
    _parse_nyfed_primary_dealer,
    _parse_ofr_csv,
    _select_nyfed_seriesbreak,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _fact(value: float, *, end: str = "2025-12-31", filed: str = "2026-02-10") -> dict:
    return {
        "start": "2025-01-01",
        "end": end,
        "filed": filed,
        "form": "10-K",
        "accn": f"{filed}-{int(value)}",
        "val": value,
    }


def _companyfacts_payload() -> dict:
    values = {
        "Assets": 100.0,
        "AssetsCurrent": 50.0,
        "Liabilities": 40.0,
        "LiabilitiesCurrent": 20.0,
        "StockholdersEquity": 60.0,
        "RevenueFromContractWithCustomerExcludingAssessedTax": 120.0,
        "OperatingIncomeLoss": 24.0,
        "NetIncomeLoss": 12.0,
        "NetCashProvidedByUsedInOperatingActivities": 20.0,
        "PaymentsToAcquirePropertyPlantAndEquipment": 5.0,
        "CashAndCashEquivalentsAtCarryingValue": 15.0,
        "LongTermDebtNoncurrent": 10.0,
    }
    facts = {
        concept: {"units": {"USD": [_fact(value)]}}
        for concept, value in values.items()
    }
    facts["Assets"]["units"]["USD"].append(
        _fact(999.0, end="2027-12-31", filed="2028-02-10")
    )
    return {"entityName": "Example Corp", "facts": {"us-gaap": facts}}


def test_official_csv_parsers_reject_future_rows() -> None:
    ofr = _parse_ofr_csv(
        "Date,OFR FSI,Credit,Equity valuation,Safe assets,Funding,Volatility,United States,Other advanced economies,Emerging markets\n"
        "2026-08-20,1.0,0.2,0.1,0.3,0.4,0.5,0.1,0.2,0.3\n"
        "2026-09-20,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0\n",
        as_of=date(2026, 8, 23),
    )
    assert [row["date"] for row in ofr] == ["2026-08-20"]

    ecb = _parse_ecb_csv(
        "Key,Time period or range,Observation value\n"
        "EST.B.EU000A2X2A25.WT,2026-08-20,2.1\n"
        "EST.B.EU000A2X2A25.WT,2026-09-20,9.9\n",
        as_of=date(2026, 8, 23),
    )
    assert ecb["WT"] == [{"date": "2026-08-20", "value": 2.1}]


def test_nyfed_primary_dealer_parser_selects_active_schema_and_rejects_future_rows() -> None:
    selected = _select_nyfed_seriesbreak(
        {
            "pd": {
                "seriesbreaks": [
                    {"seriesbreak": "SBN2023", "label": "old", "startdate": "2023-01-01", "enddate": "2024-12-31"},
                    {"seriesbreak": "SBN2024", "label": "current", "startdate": "2025-01-01", "enddate": "2026-12-31"},
                    {"seriesbreak": "SBN2027", "label": "future", "startdate": "2027-01-01", "enddate": "2028-12-31"},
                ]
            }
        },
        as_of=date(2026, 8, 23),
    )
    assert selected["seriesbreak"] == "SBN2024"

    payload = {
        "pd": {
            "timeseries": [
                {"keyid": "PDPOSGST-TOT", "asofdate": "2026-08-19", "value": "100"},
                {"keyid": "PDPOSCS-TOT", "asofdate": "2026-08-19", "value": "20"},
                {"keyid": "PDSORA-UTSETTOT", "asofdate": "2026-08-19", "value": "500"},
                {"keyid": "PDSIRRA-UTSETTOT", "asofdate": "2026-08-19", "value": "400"},
                {"keyid": "PDFTD-USTET", "asofdate": "2026-08-19", "value": "5"},
                {"keyid": "PDFTR-USTET", "asofdate": "2026-08-19", "value": "4"},
                {"keyid": "PDPOSGST-TOT", "asofdate": "2026-09-01", "value": "999"},
            ]
        }
    }
    parsed = _parse_nyfed_primary_dealer(payload, as_of=date(2026, 8, 23))
    assert parsed["observation_date"] == "2026-08-19"
    assert parsed["selected_series_count"] == 6
    assert parsed["values"]["treasury_position"] == 100.0
    assert parsed["future_rows_rejected_count"] == 1


def test_fdic_financials_parser_is_point_in_time_and_computes_credit_ratios() -> None:
    rows = [
        ("2026-06-30", 120.0, 90.0, 60.0, 3.0),
        ("2026-03-31", 115.0, 87.0, 58.0, 2.5),
        ("2025-12-31", 110.0, 84.0, 55.0, 2.0),
        ("2025-09-30", 105.0, 80.0, 52.0, 1.8),
        ("2025-06-30", 100.0, 76.0, 50.0, 1.5),
        ("2026-12-31", 999.0, 999.0, 999.0, 999.0),
    ]
    payload = {
        "data": [
            {
                "data": {
                    "REPDTE": report_date,
                    "count": "4500",
                    "sum_ASSET": str(assets),
                    "sum_DEP": str(deposits),
                    "sum_LNLSNET": str(loans),
                    "sum_NCLNLS": str(noncurrent),
                    "sum_NETINC": "2.0",
                }
            }
            for report_date, assets, deposits, loans, noncurrent in rows
        ]
    }
    parsed = _parse_fdic_financials(payload, as_of=date(2026, 8, 23))
    assert parsed["report_date"] == "2026-06-30"
    assert parsed["deposit_to_asset_ratio"] == 0.75
    assert parsed["loan_to_asset_ratio"] == 0.5
    assert parsed["noncurrent_loan_ratio"] == 0.05
    assert round(parsed["asset_growth_yoy_ratio"], 8) == 0.2
    assert parsed["future_rows_rejected_count"] == 1


def test_supplemental_failures_are_isolated_from_baseline_health() -> None:
    baseline_ids = (
        "ofr_financial_stress_index",
        "fdic_bank_failures",
        "federal_register_financial_rules",
        "ecb_estr",
        "sec_companyfacts",
    )
    sources = {source_id: {"ok": True} for source_id in baseline_ids}
    sources.update(
        {
            "nyfed_primary_dealer_statistics": {"ok": False},
            "fdic_bank_financials": {"ok": False},
        }
    )
    derived = _build_derived(sources)
    assert derived["global_features"]["public_financial_source_coverage_norm"] == 1.0
    assert derived["global_features"]["public_financial_supplemental_coverage_norm"] == 0.0

    isolated = _isolated_source(
        "fdic_bank_financials",
        lambda: (_ for _ in ()).throw(RuntimeError("expected adapter failure")),
    )
    assert isolated["ok"] is False
    assert isolated["isolated_failure"] is True
    assert isolated["contract_participates"] is False
    assert isolated["supplemental"] is True

    health_sources = _health_source_rows(sources)
    metrics = _source_status_metrics({"sources": health_sources})
    assert metrics == {"total": 5, "ok": 5, "coverage_ratio": 1.0}
    assert health_sources["nyfed_primary_dealer_statistics"]["supplemental"] is True
    assert health_sources["nyfed_primary_dealer_statistics"]["contract_participates"] is False


def test_companyfacts_is_point_in_time_and_preserves_units() -> None:
    row = _companyfacts_symbol_row(
        "TEST",
        "0000000001",
        _companyfacts_payload(),
        as_of=date(2026, 8, 23),
    )
    assert row["facts"]["assets"]["value"] == 100.0
    assert row["facts"]["assets"]["unit"] == "USD"
    assert row["features"]["fundamental_financial_statement_coverage_norm"] == 1.0
    assert row["features"]["fundamental_free_cash_flow_positive_norm"] == 1.0
    assert 0.0 <= row["features"]["fundamental_quality_norm"] <= 1.0


def test_taxonomy_is_exhaustive_and_quarantines_unknown_features() -> None:
    taxonomy = json.loads(ROUTING_CONFIG_PATH.read_text(encoding="utf-8"))
    expected = set(PUBLIC_FINANCIAL_FEATURE_KEYS) | set(PUBLIC_FINANCIAL_SYMBOL_FEATURE_KEYS)
    assert set(taxonomy["feature_routes"]) == expected
    assert all(value is False for key, value in taxonomy["authority"].items() if key != "classification_and_context_only")
    assert taxonomy["routing_contract"]["cross_family_broadcast_allowed"] is False

    derived = {
        "global_features": {key: 0.5 for key in PUBLIC_FINANCIAL_FEATURE_KEYS},
        "symbol_features": {"AAPL": {key: 0.5 for key in PUBLIC_FINANCIAL_SYMBOL_FEATURE_KEYS}},
        "feature_lineage": {},
    }
    classified, validation = _apply_routing_taxonomy(derived, taxonomy)
    assert validation["ok"] is True
    assert set(classified["global_features"]) == set(PUBLIC_FINANCIAL_FEATURE_KEYS)

    with_unknown = deepcopy(derived)
    with_unknown["global_features"]["unclassified_feature"] = 1.0
    classified, validation = _apply_routing_taxonomy(with_unknown, taxonomy)
    assert validation["ok"] is False
    assert validation["unclassified_global_feature_keys"] == ["unclassified_feature"]
    assert "unclassified_feature" not in classified["global_features"]


def test_capability_proofs_require_true_readiness_values() -> None:
    catalog = json.loads(
        (PROJECT_ROOT / "config" / "collector_capability_catalog_v1.json").read_text(encoding="utf-8")
    )
    producer = next(
        row for row in catalog["producers"] if row["producer_id"] == "public_financial_context"
    )
    false_payload = {
        "capability_readiness": {capability: False for capability in producer["capabilities"]}
    }
    usable, proofs = _evaluate_capability_proofs(producer, false_payload, producer_usable=True)
    assert usable == []
    assert all(row["passed"] is False for row in proofs.values())

    true_payload = {
        "capability_readiness": {capability: True for capability in producer["capabilities"]}
    }
    usable, proofs = _evaluate_capability_proofs(producer, true_payload, producer_usable=True)
    assert set(usable) == set(producer["capabilities"])
    assert all(row["passed"] is True for row in proofs.values())


def test_optional_source_absence_does_not_lower_plane_health() -> None:
    spec = {
        "plane_id": "funding_stress",
        "signal_key": "context_funding_stress_signal_norm",
        "plane_class": "macro",
        "required_feature_count": 3,
        "source_ids": ["required_a", "required_b"],
        "optional_source_ids": ["public_financial_context"],
        "target_domains": ["rates"],
        "target_symbols": ["TLT"],
    }
    candidates = {
        f"feature_{index}": {
            "value": 0.5,
            "lineage": [
                {
                    "source_id": "required_a" if index % 2 == 0 else "required_b",
                    "point_in_time_valid": True,
                }
            ],
        }
        for index in range(3)
    }
    source_states = {
        "required_a": {"ok": True, "freshness_norm": 1.0, "source_family": "family_a"},
        "required_b": {"ok": True, "freshness_norm": 1.0, "source_family": "family_b"},
        "public_financial_context": {"ok": False, "freshness_norm": 0.0, "source_family": "official_public_financial_context"},
    }
    scoring = {
        "source_health_weight": 0.30,
        "feature_completeness_weight": 0.25,
        "freshness_weight": 0.15,
        "point_in_time_lineage_weight": 0.15,
        "routing_weight": 0.10,
        "cross_verification_weight": 0.05,
    }
    row = _build_plane(spec, candidates, source_states, scoring=scoring, minimum_score=70.0)
    assert row["score_pct"] == 100.0
    assert row["status"] == "ready"
    assert row["optional_source_ids"] == ["public_financial_context"]


def test_symbol_fundamentals_route_only_from_healthy_optional_source() -> None:
    payloads = {
        "public_financial_context": {
            "derived": {
                "symbol_features": {
                    "AAPL": {"fundamental_quality_norm": 0.81}
                }
            }
        }
    }
    unavailable = _symbol_context_features(
        payloads,
        {"public_financial_context": {"ok": False}},
    )
    assert unavailable == {}

    available = _symbol_context_features(
        payloads,
        {"public_financial_context": {"ok": True}},
    )
    assert available["AAPL"]["fundamental_quality_norm"] == 0.81


def test_runtime_training_enrichment_enforces_decision_family_routes() -> None:
    context = {
        "external_global_features": {"ofr_funding_stress_norm": 0.9},
        "external_symbol_features": {"AAPL": {"fundamental_quality_norm": 0.8}},
        "external_feature_routes": {
            "ofr_funding_stress_norm": {
                "decision_family_ids": ["macro_rates_fx", "structured_credit"]
            },
            "fundamental_quality_norm": {
                "decision_family_ids": ["long_horizon_income"]
            },
        },
    }
    macro = _enrich_runtime_observation(
        {
            "symbol": "AAPL",
            "decision_policy_family_id": "macro_rates_fx",
            "features": {},
        },
        carry_forward_features={},
        gap_fill_context=context,
    )
    assert macro["features"]["ofr_funding_stress_norm"] == 0.9
    assert "fundamental_quality_norm" not in macro["features"]

    income = _enrich_runtime_observation(
        {
            "symbol": "AAPL",
            "decision_policy_family_id": "long_horizon_income",
            "features": {},
        },
        carry_forward_features={},
        gap_fill_context=context,
    )
    assert income["features"]["fundamental_quality_norm"] == 0.8
    assert "ofr_funding_stress_norm" not in income["features"]

    unclassified = _enrich_runtime_observation(
        {"symbol": "AAPL", "features": {}},
        carry_forward_features={},
        gap_fill_context=context,
    )
    assert "fundamental_quality_norm" not in unclassified["features"]
    assert "ofr_funding_stress_norm" not in unclassified["features"]
