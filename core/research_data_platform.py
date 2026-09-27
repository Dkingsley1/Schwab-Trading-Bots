from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from core.portfolio_advisory import build_multi_period_advisory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "research_data_platform_v1.json"

REQUIRED_CAPABILITIES = (
    "canonical_data_catalog",
    "license_entitlement_registry",
    "point_in_time_research_api",
    "bitemporal_revision_history",
    "alpha_lifecycle_governance",
    "source_value_accounting",
    "portfolio_alpha_combination",
    "unified_simulation_semantics",
    "feed_service_levels",
    "research_reproducibility",
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _missing(value: Any, *, false_is_missing: bool = False) -> bool:
    if value is None or value == "":
        return True
    if false_is_missing and value is False:
        return True
    return isinstance(value, (list, tuple, dict, set)) and not value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _parse_timestamp(value: Any, *, field: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip()
        if not text:
            raise ValueError(f"{field}_missing")
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{field}_invalid") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso_utc(value: Any, *, field: str) -> str:
    return _parse_timestamp(value, field=field).isoformat()


def _is_active_entitlement(value: Any) -> bool:
    if isinstance(value, Mapping):
        value = value.get("status", value.get("active", False))
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {
        "active",
        "approved",
        "authorized",
        "ready",
        "valid",
    }


def load_policy(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("research data platform policy must be an object")
    return payload


def validate_policy(
    policy: Mapping[str, Any], *, project_root: str | Path = PROJECT_ROOT
) -> dict[str, Any]:
    errors: list[str] = []
    required = [str(value) for value in policy.get("required_capability_ids") or []]
    capabilities = _mapping(policy.get("capabilities"))
    products = [
        dict(row) for row in policy.get("data_products") or [] if isinstance(row, Mapping)
    ]
    families_contract = _mapping(policy.get("decision_family_contract"))
    families = [str(value) for value in families_contract.get("required_family_ids") or []]
    license_profiles = _mapping(policy.get("license_profiles"))
    authority = _mapping(policy.get("authority"))
    root = Path(project_root)

    if int(policy.get("schema_version") or 0) != 1:
        errors.append("schema_version_must_be_1")
    if tuple(required) != REQUIRED_CAPABILITIES:
        errors.append("required_capability_ids_do_not_match_contract")
    if set(capabilities) != set(REQUIRED_CAPABILITIES):
        errors.append("capability_registry_does_not_match_required_ids")
    if authority.get("metadata_and_research_control_only") is not True:
        errors.append("metadata_and_research_control_only_must_be_true")
    for key, value in authority.items():
        if str(key).startswith("can_") and bool(value):
            errors.append(f"forbidden_authority_enabled:{key}")

    for capability_id in REQUIRED_CAPABILITIES:
        row = _mapping(capabilities.get(capability_id))
        owner = str(row.get("owner") or "").strip()
        test = str(row.get("test") or "").strip()
        if not owner or not (root / owner).is_file():
            errors.append(f"capability_owner_missing:{capability_id}:{owner or 'unset'}")
        if not test or not (root / test).is_file():
            errors.append(f"capability_test_missing:{capability_id}:{test or 'unset'}")

    if not products:
        errors.append("data_products_missing")
    seen_ids: set[str] = set()
    required_product_fields = (
        "dataset_id",
        "title",
        "layer",
        "owner",
        "source_ids",
        "license_profile_id",
        "schema_id",
        "natural_key_columns",
        "event_time_column",
        "knowledge_time_column",
        "revision_policy",
        "freshness_slo_seconds",
        "consumer_decision_families",
        "storage_class",
    )
    family_products = {family_id: [] for family_id in families}
    for product in products:
        dataset_id = str(product.get("dataset_id") or "").strip()
        if not dataset_id:
            errors.append("dataset_id_missing")
        elif dataset_id in seen_ids:
            errors.append(f"duplicate_dataset_id:{dataset_id}")
        seen_ids.add(dataset_id)
        for field in required_product_fields:
            if _missing(product.get(field)):
                errors.append(f"dataset_field_missing:{dataset_id or 'unknown'}:{field}")
        profile_id = str(product.get("license_profile_id") or "")
        if profile_id not in license_profiles:
            errors.append(f"dataset_license_profile_unknown:{dataset_id}:{profile_id}")
        try:
            if int(product.get("freshness_slo_seconds") or 0) <= 0:
                errors.append(f"dataset_freshness_slo_invalid:{dataset_id}")
        except (TypeError, ValueError):
            errors.append(f"dataset_freshness_slo_invalid:{dataset_id}")
        consumers = {str(value) for value in product.get("consumer_decision_families") or []}
        unknown_families = sorted(consumers - set(families))
        if unknown_families:
            errors.append(
                f"dataset_unknown_decision_families:{dataset_id}:{','.join(unknown_families)}"
            )
        for family_id in consumers & set(families):
            family_products[family_id].append(product)

    minimum_products = int(families_contract.get("minimum_products_per_family") or 0)
    for family_id, rows in family_products.items():
        if len(rows) < minimum_products:
            errors.append(f"decision_family_product_coverage_low:{family_id}")
        if families_contract.get("every_family_must_have_point_in_time_product") is True:
            if not any(
                row.get("dataset_id") == "point_in_time_feature_store_v1"
                or "point_in_time_features" in (row.get("capability_ids") or [])
                for row in rows
            ):
                errors.append(f"decision_family_point_in_time_product_missing:{family_id}")
        if families_contract.get("every_family_must_have_outcome_evidence_product") is True:
            if not any(
                row.get("dataset_id") == "candidate_outcome_evidence_v2"
                or "candidate_outcomes" in (row.get("capability_ids") or [])
                for row in rows
            ):
                errors.append(f"decision_family_outcome_product_missing:{family_id}")

    for profile_id, raw_profile in license_profiles.items():
        profile = _mapping(raw_profile)
        if not profile.get("allowed_uses"):
            errors.append(f"license_allowed_uses_missing:{profile_id}")
        if profile.get("redistribution_allowed") is not False:
            errors.append(f"license_redistribution_must_default_false:{profile_id}")

    weights = _mapping(_mapping(policy.get("source_value_policy")).get("weights"))
    try:
        weight_total = sum(float(value) for value in weights.values())
    except (TypeError, ValueError):
        weight_total = -1.0
    if abs(weight_total - 1.0) > 1e-9:
        errors.append("source_value_weights_must_sum_to_one")

    lifecycle = _mapping(policy.get("alpha_lifecycle"))
    states = {str(value) for value in lifecycle.get("states") or []}
    transitions = _mapping(lifecycle.get("allowed_transitions"))
    if set(transitions) != states:
        errors.append("alpha_lifecycle_transitions_do_not_cover_states")
    for source, targets in transitions.items():
        unknown = sorted({str(value) for value in targets or []} - states)
        if unknown:
            errors.append(f"alpha_lifecycle_unknown_targets:{source}:{','.join(unknown)}")
    if lifecycle.get("live_execution_authority") is not False:
        errors.append("alpha_lifecycle_live_authority_must_be_false")

    query = _mapping(policy.get("query_contract"))
    if int(query.get("maximum_dataset_count_per_query") or 0) <= 0:
        errors.append("query_dataset_limit_invalid")
    simulation = _mapping(policy.get("simulation_contract"))
    if "live" not in (simulation.get("supported_modes") or []):
        errors.append("simulation_live_representation_missing")
    if simulation.get("live_mode_representation_does_not_grant_submission") is not True:
        errors.append("simulation_submission_guard_missing")
    reproducibility = _mapping(policy.get("reproducibility_contract"))
    if not reproducibility.get("required_materials"):
        errors.append("reproducibility_materials_missing")
    soak = _mapping(policy.get("soak_contract"))
    if soak.get("reset_main_soak_clock") is not False:
        errors.append("soak_reset_must_be_false")
    if soak.get("changes_signal_or_order_semantics") is not False:
        errors.append("signal_or_order_semantics_must_remain_unchanged")

    for influence in policy.get("public_influences") or []:
        row = _mapping(influence)
        parsed = urlparse(str(row.get("official_url") or ""))
        if parsed.scheme != "https" or not parsed.netloc:
            errors.append(f"public_influence_url_invalid:{row.get('id') or 'unknown'}")
        if row.get("influence_only") is not True:
            errors.append(f"public_influence_authority_invalid:{row.get('id') or 'unknown'}")

    return {
        "ok": not errors,
        "errors": errors,
        "capability_count": len(capabilities),
        "data_product_count": len(products),
        "decision_family_count": len(families),
        "family_product_counts": {
            family_id: len(rows) for family_id, rows in sorted(family_products.items())
        },
        "authority_safe": not any(
            error.startswith("forbidden_authority") for error in errors
        ),
    }


class ResearchDataCatalog:
    def __init__(self, policy: Mapping[str, Any]) -> None:
        self.policy = copy.deepcopy(dict(policy))
        self._products = {
            str(row["dataset_id"]): dict(row)
            for row in self.policy.get("data_products") or []
            if isinstance(row, Mapping) and row.get("dataset_id")
        }

    def dataset(self, dataset_id: str) -> dict[str, Any]:
        try:
            return copy.deepcopy(self._products[str(dataset_id)])
        except KeyError as exc:
            raise KeyError(f"unknown_dataset:{dataset_id}") from exc

    def query_catalog(
        self,
        *,
        decision_family_id: str = "",
        capability_id: str = "",
        source_id: str = "",
        layer: str = "",
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for dataset_id in sorted(self._products):
            row = self._products[dataset_id]
            if decision_family_id and decision_family_id not in (
                row.get("consumer_decision_families") or []
            ):
                continue
            if capability_id and capability_id not in (row.get("capability_ids") or []):
                continue
            if source_id and source_id not in (row.get("source_ids") or []):
                continue
            if layer and str(row.get("layer") or "") != layer:
                continue
            rows.append(copy.deepcopy(row))
        return rows

    def data_products_for_sources(
        self, source_ids: Sequence[str], *, decision_family_id: str = ""
    ) -> list[dict[str, Any]]:
        wanted = {str(value) for value in source_ids if str(value).strip()}
        return [
            row
            for row in self.query_catalog(decision_family_id=decision_family_id)
            if wanted.intersection(str(value) for value in row.get("source_ids") or [])
        ]

    def authorize_use(
        self,
        dataset_id: str,
        purpose: str,
        *,
        entitlement_states: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        product = self.dataset(dataset_id)
        profile_id = str(product.get("license_profile_id") or "")
        profile = _mapping(_mapping(self.policy.get("license_profiles")).get(profile_id))
        purpose = str(purpose or "").strip()
        states = _mapping(entitlement_states)
        reasons: list[str] = []
        if purpose not in {str(value) for value in profile.get("allowed_uses") or []}:
            reasons.append("purpose_not_allowed")
        if purpose in {str(value) for value in profile.get("prohibited_uses") or []}:
            reasons.append("purpose_explicitly_prohibited")
        required = [str(value) for value in product.get("required_entitlement_ids") or []]
        missing = [value for value in required if not _is_active_entitlement(states.get(value))]
        if missing:
            reasons.append("required_entitlement_missing")
        review_id = f"license_terms_review:{profile_id}"
        terms_review_required = bool(profile.get("external_terms_review_required", False))
        terms_review_ready = not terms_review_required or _is_active_entitlement(
            states.get(review_id)
        )
        if not terms_review_ready:
            reasons.append("external_terms_review_pending")
        return {
            "dataset_id": dataset_id,
            "purpose": purpose,
            "authorized": not reasons,
            "reasons": reasons,
            "license_profile_id": profile_id,
            "required_entitlement_ids": required,
            "missing_entitlement_ids": missing,
            "terms_review_required": terms_review_required,
            "terms_review_ready": terms_review_ready,
            "credentials_stored": False,
            "execution_authority": False,
        }

    def build_query_plan(
        self,
        *,
        dataset_ids: Sequence[str],
        consumer_id: str,
        purpose: str,
        as_of_utc: Any,
        valid_at_utc: Any,
        entitlement_states: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        query = _mapping(self.policy.get("query_contract"))
        ids = list(dict.fromkeys(str(value) for value in dataset_ids if str(value).strip()))
        if not ids:
            raise ValueError("query_dataset_ids_missing")
        if len(ids) > int(query.get("maximum_dataset_count_per_query") or 0):
            raise ValueError("query_dataset_limit_exceeded")
        if not str(consumer_id or "").strip():
            raise ValueError("query_consumer_id_missing")
        as_of = _iso_utc(as_of_utc, field="as_of_utc")
        valid_at = _iso_utc(valid_at_utc, field="valid_at_utc")
        authorizations = [
            self.authorize_use(
                dataset_id,
                purpose,
                entitlement_states=entitlement_states,
            )
            for dataset_id in ids
        ]
        denied = [row["dataset_id"] for row in authorizations if not row["authorized"]]
        if denied:
            raise PermissionError(f"dataset_use_not_authorized:{','.join(denied)}")
        dataset_receipts = {
            dataset_id: canonical_sha256(self.dataset(dataset_id)) for dataset_id in ids
        }
        body = {
            "contract_id": str(query.get("contract_id") or ""),
            "dataset_ids": ids,
            "consumer_id": str(consumer_id),
            "purpose": str(purpose),
            "as_of_utc": as_of,
            "valid_at_utc": valid_at,
            "dataset_receipts": dataset_receipts,
            "future_knowledge_allowed": False,
            "execution_authority": False,
        }
        return {
            **body,
            "authorization_receipts": authorizations,
            "query_receipt_sha256": canonical_sha256(body),
        }


def select_bitemporal_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    as_of_utc: Any,
    valid_at_utc: Any,
    natural_key_columns: Sequence[str],
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not natural_key_columns:
        raise ValueError("natural_key_columns_missing")
    as_of = _parse_timestamp(as_of_utc, field="as_of_utc")
    valid_at = _parse_timestamp(valid_at_utc, field="valid_at_utc")
    effective_from_field = str(contract.get("effective_from_field") or "effective_at_utc")
    effective_to_field = str(contract.get("effective_to_field") or "effective_until_utc")
    known_at_field = str(contract.get("known_at_field") or "known_at_utc")
    superseded_at_field = str(contract.get("superseded_at_field") or "superseded_at_utc")
    revision_id_field = str(contract.get("revision_id_field") or "revision_id")
    selected: dict[tuple[str, ...], tuple[datetime, str, dict[str, Any]]] = {}
    for raw_row in rows:
        row = dict(raw_row)
        known_at = _parse_timestamp(row.get(known_at_field), field=known_at_field)
        effective_from = _parse_timestamp(
            row.get(effective_from_field), field=effective_from_field
        )
        if known_at > as_of or effective_from > valid_at:
            continue
        effective_to_raw = row.get(effective_to_field)
        if not _missing(effective_to_raw):
            effective_to = _parse_timestamp(effective_to_raw, field=effective_to_field)
            if valid_at >= effective_to:
                continue
        superseded_raw = row.get(superseded_at_field)
        if not _missing(superseded_raw):
            superseded_at = _parse_timestamp(superseded_raw, field=superseded_at_field)
            if superseded_at <= as_of:
                continue
        key = tuple(str(row.get(column, "")) for column in natural_key_columns)
        if any(not part for part in key):
            raise ValueError("natural_key_value_missing")
        revision_id = str(row.get(revision_id_field) or "")
        candidate = (known_at, revision_id, row)
        current = selected.get(key)
        if current is None or candidate[:2] > current[:2]:
            selected[key] = candidate
    return [selected[key][2] for key in sorted(selected)]


def _alpha_event(
    *,
    alpha_id: str,
    source_state: str,
    target_state: str,
    evidence: Mapping[str, Any],
    observed_at_utc: Any,
    previous_event_sha256: str,
) -> dict[str, Any]:
    body = {
        "alpha_id": alpha_id,
        "source_state": source_state,
        "target_state": target_state,
        "observed_at_utc": _iso_utc(observed_at_utc, field="observed_at_utc"),
        "evidence_sha256": canonical_sha256(dict(evidence)),
        "evidence_keys": sorted(str(key) for key in evidence),
        "previous_event_sha256": previous_event_sha256,
    }
    return {**body, "event_sha256": canonical_sha256(body)}


def new_alpha_record(
    alpha_id: str,
    *,
    observed_at_utc: Any,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    alpha_id = str(alpha_id or "").strip()
    if not alpha_id:
        raise ValueError("alpha_id_missing")
    event = _alpha_event(
        alpha_id=alpha_id,
        source_state="none",
        target_state="raw_feature",
        evidence={"origin": "registered"},
        observed_at_utc=observed_at_utc,
        previous_event_sha256="0" * 64,
    )
    return {
        "alpha_id": alpha_id,
        "state": "raw_feature",
        "metadata": dict(metadata or {}),
        "history": [event],
        "live_execution_authority": False,
    }


def transition_alpha(
    record: Mapping[str, Any],
    target_state: str,
    *,
    evidence: Mapping[str, Any],
    observed_at_utc: Any,
    lifecycle: Mapping[str, Any],
) -> dict[str, Any]:
    out = copy.deepcopy(dict(record))
    source_state = str(out.get("state") or "")
    target_state = str(target_state or "").strip()
    allowed = {
        str(value)
        for value in _mapping(lifecycle.get("allowed_transitions")).get(source_state, [])
    }
    if target_state not in allowed:
        raise ValueError(f"alpha_transition_not_allowed:{source_state}:{target_state}")
    evidence_map = dict(evidence)
    required = [
        str(value)
        for value in _mapping(lifecycle.get("required_evidence_by_target_state")).get(
            target_state, []
        )
    ]
    missing = [
        key for key in required if _missing(evidence_map.get(key), false_is_missing=True)
    ]
    if missing:
        raise ValueError(f"alpha_transition_evidence_missing:{','.join(missing)}")
    history = [dict(row) for row in out.get("history") or [] if isinstance(row, Mapping)]
    previous = str(history[-1].get("event_sha256") or "") if history else "0" * 64
    event = _alpha_event(
        alpha_id=str(out.get("alpha_id") or ""),
        source_state=source_state,
        target_state=target_state,
        evidence=evidence_map,
        observed_at_utc=observed_at_utc,
        previous_event_sha256=previous,
    )
    history.append(event)
    out.update(
        {
            "state": target_state,
            "history": history,
            "latest_evidence_sha256": event["evidence_sha256"],
            "live_execution_authority": False,
        }
    )
    return out


def verify_alpha_history(record: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    previous = "0" * 64
    history = [dict(row) for row in record.get("history") or [] if isinstance(row, Mapping)]
    for index, event in enumerate(history):
        if str(event.get("previous_event_sha256") or "") != previous:
            errors.append(f"history_link_invalid:{index}")
        body = {key: value for key, value in event.items() if key != "event_sha256"}
        if canonical_sha256(body) != str(event.get("event_sha256") or ""):
            errors.append(f"history_hash_invalid:{index}")
        previous = str(event.get("event_sha256") or "")
    if history and str(history[-1].get("target_state") or "") != str(
        record.get("state") or ""
    ):
        errors.append("history_terminal_state_mismatch")
    return {"ok": not errors, "errors": errors, "event_count": len(history)}


def _unit_score(value: Any) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def evaluate_source_value(
    source_id: str, metrics: Mapping[str, Any], policy: Mapping[str, Any]
) -> dict[str, Any]:
    source_policy = _mapping(policy.get("source_value_policy"))
    weights = _mapping(source_policy.get("weights"))
    required_outcomes = (
        "candidate_id",
        "candidate_bound_samples",
        "incremental_information",
        "net_post_cost_contribution",
        "nonredundancy",
    )
    missing = [key for key in required_outcomes if _missing(metrics.get(key))]
    samples = int(metrics.get("candidate_bound_samples") or 0)
    minimum_samples = int(source_policy.get("minimum_candidate_bound_samples") or 0)
    reasons = [f"missing:{key}" for key in missing]
    if samples < minimum_samples:
        reasons.append("insufficient_candidate_bound_samples")
    components = {
        "quality": _unit_score(metrics.get("quality")),
        "freshness": _unit_score(metrics.get("freshness")),
        "availability": _unit_score(metrics.get("availability")),
        "incremental_information": _unit_score(metrics.get("incremental_information")),
        "net_post_cost_contribution": _unit_score(
            metrics.get("net_post_cost_contribution")
        ),
        "nonredundancy": _unit_score(metrics.get("nonredundancy")),
    }
    score = 100.0 * sum(
        components[key] * float(weights.get(key) or 0.0) for key in components
    )
    evidence_ready = not reasons
    qualified = bool(
        evidence_ready
        and score >= 60.0
        and components["incremental_information"] > 0.0
        and components["net_post_cost_contribution"] > 0.0
    )
    return {
        "source_id": str(source_id),
        "status": "qualified" if qualified else "evaluated" if evidence_ready else str(
            source_policy.get("missing_outcome_metrics_status") or "collecting"
        ),
        "evidence_ready": evidence_ready,
        "qualified": qualified,
        "score": round(score, 4) if evidence_ready else None,
        "components": components,
        "candidate_id": str(metrics.get("candidate_id") or ""),
        "candidate_bound_samples": samples,
        "minimum_candidate_bound_samples": minimum_samples,
        "reasons": reasons,
        "purchase_authority": False,
        "retirement_authority": False,
        "execution_authority": False,
    }


def portfolio_alpha_advisory(
    *,
    candidate_id: str,
    sleeves: Sequence[Mapping[str, Any]],
    covariance: Mapping[str, Mapping[str, float]],
    current_weights: Mapping[str, float],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    contract = _mapping(policy.get("portfolio_combination"))
    report = build_multi_period_advisory(
        candidate_id=candidate_id,
        sleeves=sleeves,
        covariance=covariance,
        current_weights=current_weights,
        minimum_qualified_sleeves=int(contract.get("minimum_qualified_sleeves") or 4),
        minimum_fills=int(contract.get("minimum_independent_fills") or 30),
        cash_floor=float(contract.get("cash_floor") or 0.10),
        max_sleeve_weight=float(contract.get("maximum_sleeve_weight") or 0.25),
        max_step_turnover=float(contract.get("maximum_step_turnover") or 0.20),
        max_pair_correlation=float(contract.get("maximum_pairwise_correlation") or 0.75),
    )
    report["contract_delegate"] = str(contract.get("delegate") or "")
    report["execution_authority"] = False
    report["advisory_only"] = True
    return report


def normalize_simulation_event(
    event: Mapping[str, Any], *, mode: str, policy: Mapping[str, Any]
) -> dict[str, Any]:
    contract = _mapping(policy.get("simulation_contract"))
    mode = str(mode or "").strip()
    if mode not in {str(value) for value in contract.get("supported_modes") or []}:
        raise ValueError(f"simulation_mode_unsupported:{mode}")
    out = dict(event)
    payload = out.pop("payload", {})
    payload_hash = canonical_sha256(payload)
    out["payload_sha256"] = str(out.get("payload_sha256") or payload_hash)
    missing = [
        str(field)
        for field in contract.get("mode_invariant_fields") or []
        if _missing(out.get(str(field)))
    ]
    if missing:
        raise ValueError(f"simulation_event_fields_missing:{','.join(missing)}")
    out.update(
        {
            "contract_id": str(contract.get("contract_id") or ""),
            "mode": mode,
            "submission_authority": False,
            "representation_only": mode == "live",
        }
    )
    return out


def evaluate_feed_slo(
    product: Mapping[str, Any],
    observation: Mapping[str, Any] | None,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    contract = _mapping(policy.get("feed_slo_contract"))
    dataset_id = str(product.get("dataset_id") or "")
    if not isinstance(observation, Mapping) or not observation:
        return {
            "dataset_id": dataset_id,
            "status": str(contract.get("missing_observation_status") or "unknown"),
            "ok": False,
            "failed_checks": ["observation_missing"],
            "actions": list(contract.get("degradation_actions") or []),
            "execution_authority": False,
        }
    checks = {
        "freshness": float(observation.get("age_seconds") or 0.0)
        <= float(product.get("freshness_slo_seconds") or 0.0),
        "completeness": float(observation.get("completeness_ratio") or 0.0)
        >= float(contract.get("minimum_completeness_ratio") or 0.0),
        "validity": float(observation.get("validity_ratio") or 0.0)
        >= float(contract.get("minimum_validity_ratio") or 0.0),
        "availability": float(observation.get("availability_ratio") or 0.0)
        >= float(contract.get("minimum_availability_ratio") or 0.0),
        "correction": float(observation.get("correction_ratio") or 0.0)
        <= float(contract.get("maximum_correction_ratio") or 0.0),
    }
    failed = [name for name, ready in checks.items() if not ready]
    return {
        "dataset_id": dataset_id,
        "status": "ready" if not failed else "degraded",
        "ok": not failed,
        "checks": checks,
        "failed_checks": failed,
        "actions": [] if not failed else list(contract.get("degradation_actions") or []),
        "execution_authority": False,
    }


def build_reproducibility_receipt(
    materials: Mapping[str, Any], policy: Mapping[str, Any]
) -> dict[str, Any]:
    contract = _mapping(policy.get("reproducibility_contract"))
    required = [str(value) for value in contract.get("required_materials") or []]
    missing = [
        key for key in required if _missing(materials.get(key), false_is_missing=True)
    ]
    if missing:
        raise ValueError(f"reproducibility_materials_missing:{','.join(missing)}")
    hashes = {key: canonical_sha256(materials[key]) for key in required}
    body = {
        "contract_id": str(contract.get("contract_id") or ""),
        "candidate_id": str(materials.get("candidate_id") or ""),
        "material_sha256": hashes,
        "local_only": True,
        "external_attestation": False,
        "historical_rewrite_allowed": False,
        "execution_authority": False,
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def structural_probe(policy: Mapping[str, Any]) -> dict[str, Any]:
    validation = validate_policy(policy)
    catalog = ResearchDataCatalog(policy)
    timestamp = "2026-08-23T12:00:00+00:00"
    official_auth = catalog.authorize_use("official_us_macro_v1", "research")
    restricted_auth = catalog.authorize_use(
        "broker_market_observations_v1", "research"
    )
    query_plan = catalog.build_query_plan(
        dataset_ids=["official_us_macro_v1", "point_in_time_feature_store_v1"],
        consumer_id="structural_probe",
        purpose="research",
        as_of_utc=timestamp,
        valid_at_utc="2026-08-01T12:00:00+00:00",
    )
    bitemporal_rows = select_bitemporal_rows(
        [
            {
                "series_id": "probe",
                "period": "2026-07",
                "effective_at_utc": "2026-08-01T00:00:00+00:00",
                "known_at_utc": "2026-08-02T00:00:00+00:00",
                "superseded_at_utc": "2026-08-10T00:00:00+00:00",
                "revision_id": "r1",
                "value": 1.0,
            },
            {
                "series_id": "probe",
                "period": "2026-07",
                "effective_at_utc": "2026-08-01T00:00:00+00:00",
                "known_at_utc": "2026-08-10T00:00:00+00:00",
                "revision_id": "r2",
                "value": 2.0,
            },
        ],
        as_of_utc="2026-08-05T00:00:00+00:00",
        valid_at_utc="2026-08-01T12:00:00+00:00",
        natural_key_columns=["series_id", "period"],
        contract=_mapping(policy.get("bitemporal_contract")),
    )
    lifecycle = _mapping(policy.get("alpha_lifecycle"))
    alpha = new_alpha_record("alpha-probe", observed_at_utc=timestamp)
    alpha = transition_alpha(
        alpha,
        "candidate_alpha",
        evidence={
            "dataset_receipts": [query_plan["query_receipt_sha256"]],
            "hypothesis_id": "hypothesis-probe",
            "candidate_id": "candidate-probe",
        },
        observed_at_utc="2026-08-23T12:01:00+00:00",
        lifecycle=lifecycle,
    )
    source_value = evaluate_source_value(
        "source-probe",
        {
            "candidate_id": "candidate-probe",
            "candidate_bound_samples": 30,
            "quality": 1.0,
            "freshness": 1.0,
            "availability": 1.0,
            "incremental_information": 0.8,
            "net_post_cost_contribution": 0.7,
            "nonredundancy": 0.9,
        },
        policy,
    )
    sleeve_ids = ("dividend", "bond", "fx", "volatility")
    portfolio = portfolio_alpha_advisory(
        candidate_id="candidate-probe",
        sleeves=[
            {
                "sleeve_id": sleeve_id,
                "candidate_id": "candidate-probe",
                "qualified": True,
                "independent_fills": 30,
                "expected_return_bps": 8.0 - index,
                "cost_bps": 1.0,
            }
            for index, sleeve_id in enumerate(sleeve_ids)
        ],
        covariance={
            left: {right: 1.0 if left == right else 0.1 for right in sleeve_ids}
            for left in sleeve_ids
        },
        current_weights={},
        policy=policy,
    )
    base_event = {
        "event_id": "event-probe",
        "trace_id": "trace-probe",
        "candidate_id": "candidate-probe",
        "symbol": "SPY",
        "event_time_utc": timestamp,
        "known_at_utc": timestamp,
        "event_type": "market_observation",
        "payload": {"price": 100.0},
    }
    paper_event = normalize_simulation_event(base_event, mode="paper", policy=policy)
    shadow_event = normalize_simulation_event(base_event, mode="shadow", policy=policy)
    product = catalog.dataset("official_us_macro_v1")
    feed_slo = evaluate_feed_slo(
        product,
        {
            "age_seconds": 10,
            "completeness_ratio": 1.0,
            "validity_ratio": 1.0,
            "availability_ratio": 1.0,
            "correction_ratio": 0.0,
        },
        policy,
    )
    reproducibility = build_reproducibility_receipt(
        {
            "candidate_id": "candidate-probe",
            "code_revision": "revision-probe",
            "dataset_receipts": query_plan["dataset_receipts"],
            "parameter_receipt": "parameters-probe",
            "label_contract_receipt": "labels-probe",
            "cost_model_receipt": "costs-probe",
            "result_receipt": "results-probe",
        },
        policy,
    )
    family_counts = validation.get("family_product_counts") or {}
    controls = {
        "canonical_data_catalog": bool(
            validation["ok"]
            and len(catalog.query_catalog()) == len(policy.get("data_products") or [])
            and all(int(value) >= 2 for value in family_counts.values())
        ),
        "license_entitlement_registry": bool(
            official_auth["authorized"]
            and not restricted_auth["authorized"]
            and restricted_auth["execution_authority"] is False
        ),
        "point_in_time_research_api": bool(
            query_plan["query_receipt_sha256"]
            and query_plan["future_knowledge_allowed"] is False
        ),
        "bitemporal_revision_history": bool(
            len(bitemporal_rows) == 1 and bitemporal_rows[0]["revision_id"] == "r1"
        ),
        "alpha_lifecycle_governance": bool(
            alpha["state"] == "candidate_alpha" and verify_alpha_history(alpha)["ok"]
        ),
        "source_value_accounting": bool(
            source_value["evidence_ready"]
            and source_value["purchase_authority"] is False
        ),
        "portfolio_alpha_combination": bool(
            portfolio["ok"]
            and portfolio["advisory_only"]
            and portfolio["execution_authority"] is False
        ),
        "unified_simulation_semantics": bool(
            paper_event["payload_sha256"] == shadow_event["payload_sha256"]
            and paper_event["submission_authority"] is False
        ),
        "feed_service_levels": bool(
            feed_slo["ok"] and feed_slo["execution_authority"] is False
        ),
        "research_reproducibility": bool(
            reproducibility["receipt_sha256"]
            and reproducibility["external_attestation"] is False
        ),
    }
    return {
        "ok": bool(validation["ok"] and all(controls.values())),
        "controls": controls,
        "ready_count": sum(1 for ready in controls.values() if ready),
        "control_count": len(controls),
        "validation": validation,
        "synthetic_probe": True,
        "candidate_bound_economic_evidence": False,
        "external_attestation": False,
        "live_execution_authority": False,
    }
