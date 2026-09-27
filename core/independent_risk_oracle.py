from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def build_risk_request(
    *, candidate_id: str, product_id: str, valuation_time_utc: str, measures: list[str]
) -> dict[str, Any]:
    body = {
        "candidate_id": candidate_id,
        "product_id": product_id,
        "valuation_time_utc": valuation_time_utc,
        "measures": sorted(set(measures)),
    }
    body["request_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return body


def reconcile_risk_results(
    primary: Mapping[str, Any],
    oracle: Mapping[str, Any],
    *,
    absolute_tolerance: Mapping[str, float],
    relative_tolerance: float = 1e-6,
    synthetic_probe: bool = False,
) -> dict[str, Any]:
    primary_provider = str(primary.get("provider_id") or "")
    oracle_provider = str(oracle.get("provider_id") or "")
    primary_model = str(primary.get("model_id") or "")
    oracle_model = str(oracle.get("model_id") or "")
    request_match = primary.get("request_sha256") == oracle.get("request_sha256")
    values_a = dict(primary.get("values") or {})
    values_b = dict(oracle.get("values") or {})
    measures = sorted(set(values_a) | set(values_b))
    comparisons: dict[str, Any] = {}
    for measure in measures:
        if measure not in values_a or measure not in values_b:
            comparisons[measure] = {"ok": False, "reason": "measure_missing"}
            continue
        left = float(values_a[measure])
        right = float(values_b[measure])
        allowed = float(
            absolute_tolerance.get(measure, 0.0)
        ) + relative_tolerance * max(abs(left), abs(right))
        difference = abs(left - right)
        comparisons[measure] = {
            "ok": difference <= allowed,
            "difference": difference,
            "allowed_difference": allowed,
        }
    signed_attestation = bool(oracle.get("signed_attestation_id"))
    structurally_independent = bool(
        primary_provider
        and oracle_provider
        and primary_provider != oracle_provider
        and primary_model
        and oracle_model
        and primary_model != oracle_model
    )
    reconciled = bool(
        request_match and comparisons and all(row["ok"] for row in comparisons.values())
    )
    evidence_eligible = bool(
        reconciled
        and structurally_independent
        and signed_attestation
        and not synthetic_probe
    )
    return {
        "ok": reconciled,
        "request_match": request_match,
        "structurally_independent": structurally_independent,
        "signed_attestation": signed_attestation,
        "synthetic_probe": synthetic_probe,
        "evidence_eligible": evidence_eligible,
        "comparisons": comparisons,
        "external_evidence_debt": (
            []
            if evidence_eligible
            else ["real_external_provider_observation", "signed_oracle_attestation"]
        ),
        "advisory_only": True,
        "execution_authority": False,
    }
