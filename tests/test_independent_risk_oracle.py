from core.independent_risk_oracle import build_risk_request, reconcile_risk_results


def test_risk_oracle_reconciles_but_synthetic_probe_cannot_be_evidence() -> None:
    request = build_risk_request(
        candidate_id="candidate-1",
        product_id="SPY",
        valuation_time_utc="2026-08-21T12:00:00+00:00",
        measures=["pv", "delta"],
    )
    primary = {
        "provider_id": "local",
        "model_id": "local-v1",
        "request_sha256": request["request_sha256"],
        "values": {"pv": 100.0, "delta": 0.5},
    }
    oracle = {
        "provider_id": "external",
        "model_id": "oracle-v1",
        "request_sha256": request["request_sha256"],
        "values": {"pv": 100.001, "delta": 0.5001},
        "signed_attestation_id": "synthetic-receipt",
    }
    report = reconcile_risk_results(
        primary,
        oracle,
        absolute_tolerance={"pv": 0.01, "delta": 0.001},
        synthetic_probe=True,
    )

    assert report["ok"] is True
    assert report["structurally_independent"] is True
    assert report["evidence_eligible"] is False
    assert report["execution_authority"] is False
