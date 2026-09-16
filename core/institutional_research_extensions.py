"""Advisory-only institutional research and governance extensions.

The module translates public design patterns into local, deterministic controls. It
does not fetch data, launch work, mutate a strategy, or submit an order.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse

from core.independent_risk_oracle import build_risk_request, reconcile_risk_results

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "institutional_research_extensions_v1.json"
REQUIRED_CONTROL_IDS = (
    "independent_factor_benchmarks",
    "pipeline_incident_ownership",
    "material_strategy_change_governance",
    "candidate_risk_schedules",
    "execution_speed_cost_frontier",
    "research_dag_checkpoint_resume",
    "versioned_research_dataset_storage",
    "cross_engine_valuation_reconciliation",
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


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


def _parse_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def load_policy(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("institutional research extension policy must be an object")
    return payload


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [list(matrix[row]) + [float(vector[row])] for row in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-14:
            raise ValueError("factor_design_matrix_singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                current - factor * pivot_value
                for current, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][-1] for row in range(size)]


def evaluate_factor_exposure(
    observations: Sequence[Mapping[str, Any]],
    *,
    candidate_id: str,
    policy: Mapping[str, Any],
    factor_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Fit a deterministic ridge factor diagnostic without declaring alpha."""

    config = _mapping(policy.get("factor_benchmarks"))
    selected = [
        str(value)
        for value in (factor_ids or config.get("benchmark_ids") or [])
        if str(value)
    ]
    minimum = int(config.get("minimum_observations") or 30)
    usable: list[tuple[float, list[float]]] = []
    for raw in observations:
        row = _mapping(raw)
        if "strategy_return" not in row or any(
            factor not in row for factor in selected
        ):
            continue
        target = _number(row.get("strategy_return"), math.nan)
        values = [_number(row.get(factor), math.nan) for factor in selected]
        if math.isfinite(target) and all(math.isfinite(value) for value in values):
            usable.append((target, values))
    base = {
        "candidate_id": str(candidate_id or ""),
        "factor_ids": selected,
        "observation_count": len(usable),
        "minimum_observations": minimum,
        "diagnostic_only": True,
        "strategy_admission_authority": False,
        "execution_authority": False,
        "profitability_guaranteed": False,
    }
    if not candidate_id or not selected or len(usable) < minimum:
        return {
            **base,
            "ok": False,
            "status": "collecting",
            "reason": "candidate_or_observation_floor_missing",
            "evidence_eligible": False,
        }

    design = [[1.0, *values] for _, values in usable]
    targets = [target for target, _ in usable]
    width = len(selected) + 1
    gram = [[0.0 for _ in range(width)] for _ in range(width)]
    rhs = [0.0 for _ in range(width)]
    for row, target in zip(design, targets):
        for left in range(width):
            rhs[left] += row[left] * target
            for right in range(width):
                gram[left][right] += row[left] * row[right]
    ridge = max(_number(config.get("ridge_lambda"), 1e-6), 0.0)
    for index in range(1, width):
        gram[index][index] += ridge
    coefficients = _solve_linear_system(gram, rhs)
    fitted = [
        sum(coef * value for coef, value in zip(coefficients, row)) for row in design
    ]
    mean_target = sum(targets) / len(targets)
    residual_sum = sum((target - fit) ** 2 for target, fit in zip(targets, fitted))
    total_sum = sum((target - mean_target) ** 2 for target in targets)
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else 0.0
    result = {
        **base,
        "ok": True,
        "status": "diagnostic_ready",
        "intercept": coefficients[0],
        "factor_loadings": dict(zip(selected, coefficients[1:])),
        "r_squared": max(min(r_squared, 1.0), 0.0),
        "residual_sum_squares": residual_sum,
        "evidence_eligible": True,
        "intercept_is_proven_alpha": False,
    }
    result["receipt_sha256"] = canonical_sha256(result)
    return result


def evaluate_incident_ownership(
    incidents: Sequence[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any],
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate named ownership, acknowledgement, recovery, and closeout receipts."""

    config = _mapping(policy.get("incident_ownership"))
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    required = [str(value) for value in config.get("required_fields") or []]
    closed_statuses = {
        str(value).lower() for value in config.get("closed_statuses") or []
    }
    ack_slos = _mapping(config.get("acknowledgement_slo_seconds"))
    recovery_slos = _mapping(config.get("recovery_slo_seconds"))
    reports: list[dict[str, Any]] = []
    for raw in incidents:
        row = _mapping(raw)
        missing = [field for field in required if row.get(field) in {None, ""}]
        severity = str(row.get("severity") or "low").lower()
        status = str(row.get("status") or "unknown").lower()
        detected = _parse_timestamp(row.get("detected_at_utc"))
        acknowledged = _parse_timestamp(row.get("acknowledged_at_utc"))
        resolved = _parse_timestamp(row.get("resolved_at_utc"))
        ack_seconds = (
            max((acknowledged - detected).total_seconds(), 0.0)
            if detected and acknowledged
            else max((current - detected).total_seconds(), 0.0) if detected else None
        )
        recovery_seconds = (
            max((resolved - detected).total_seconds(), 0.0)
            if detected and resolved
            else max((current - detected).total_seconds(), 0.0) if detected else None
        )
        closed = status in closed_statuses
        closeout_missing = []
        if (
            closed
            and config.get("root_cause_receipt_required_when_closed")
            and not row.get("root_cause_receipt")
        ):
            closeout_missing.append("root_cause_receipt")
        if (
            closed
            and config.get("remediation_receipt_required_when_closed")
            and not row.get("remediation_receipt")
        ):
            closeout_missing.append("remediation_receipt")
        ack_limit = _number(ack_slos.get(severity), 0.0)
        recovery_limit = _number(recovery_slos.get(severity), 0.0)
        ack_breached = bool(
            ack_seconds is None or not acknowledged or ack_seconds > ack_limit
        )
        recovery_breached = bool(
            recovery_seconds is None
            or (closed and not resolved)
            or (closed and recovery_seconds > recovery_limit)
            or (not closed and recovery_seconds > recovery_limit)
        )
        ready = (
            not missing
            and not closeout_missing
            and not ack_breached
            and not recovery_breached
        )
        fingerprint = canonical_sha256(
            {
                "pipeline_id": row.get("pipeline_id"),
                "root_cause": row.get("root_cause") or row.get("root_cause_receipt"),
            }
        )
        reports.append(
            {
                "incident_id": str(row.get("incident_id") or ""),
                "pipeline_id": str(row.get("pipeline_id") or ""),
                "owner": str(row.get("owner") or ""),
                "severity": severity,
                "status": status,
                "closed": closed,
                "missing_fields": missing,
                "closeout_missing": closeout_missing,
                "acknowledgement_seconds": ack_seconds,
                "recovery_seconds": recovery_seconds,
                "acknowledgement_slo_breached": ack_breached,
                "recovery_slo_breached": recovery_breached,
                "recurrence_fingerprint": fingerprint,
                "ready": ready,
            }
        )
    duplicate_fingerprints = sorted(
        {
            report["recurrence_fingerprint"]
            for report in reports
            if sum(
                1
                for candidate in reports
                if candidate["recurrence_fingerprint"]
                == report["recurrence_fingerprint"]
            )
            > 1
        }
    )
    ready = bool(reports and all(report["ready"] for report in reports))
    return {
        "ok": ready,
        "status": (
            "ready" if ready else "unexercised" if not reports else "needs_attention"
        ),
        "incident_count": len(reports),
        "owned_count": sum(1 for row in reports if row["owner"]),
        "ready_count": sum(1 for row in reports if row["ready"]),
        "recurrence_fingerprints": duplicate_fingerprints,
        "incidents": reports,
        "mutation_authority": False,
        "execution_authority": False,
    }


def classify_material_change(
    paths: Iterable[str], *, candidate_id: str, policy: Mapping[str, Any]
) -> dict[str, Any]:
    """Classify a change and emit review/evidence obligations without accepting it."""

    config = _mapping(policy.get("material_change_governance"))
    classes = [_mapping(row) for row in config.get("classes") or []]
    normalized_paths = sorted(
        {str(path).strip().lstrip("./") for path in paths if str(path).strip()}
    )
    matched: list[dict[str, Any]] = []
    for row in classes:
        if any(
            fnmatch.fnmatch(path, str(pattern))
            for path in normalized_paths
            for pattern in row.get("patterns") or []
        ):
            matched.append(row)
    if matched:
        selected = max(matched, key=lambda row: int(row.get("rank") or 0))
    else:
        default_id = str(config.get("default_class_id") or "")
        selected = next(
            (row for row in classes if row.get("class_id") == default_id), {}
        )
    material = {
        "candidate_id": str(candidate_id or ""),
        "paths": normalized_paths,
        "change_class": str(selected.get("class_id") or "unclassified"),
        "rank": int(selected.get("rank") or 0),
        "required_reviews": sorted(
            {str(value) for value in selected.get("required_reviews") or []}
        ),
        "required_tests": sorted(
            {str(value) for value in selected.get("required_tests") or []}
        ),
        "forward_evidence_hours": int(selected.get("forward_evidence_hours") or 0),
    }
    return {
        **material,
        "ok": bool(candidate_id and selected),
        "human_acceptance_required": bool(
            config.get("human_acceptance_required", True)
        ),
        "cumulative_soak_history_preserved": bool(
            config.get("cumulative_soak_history_preserved", True)
        ),
        "affected_scope_forward_window_required": bool(
            config.get("affected_scope_forward_window_required", True)
        ),
        "automatic_acceptance_authority": False,
        "live_promotion_authority": False,
        "receipt_sha256": canonical_sha256(material),
    }


def evaluate_candidate_risk_schedule(
    snapshot: Mapping[str, Any],
    *,
    candidate_id: str,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate a candidate-bound risk schedule without replacing pre-trade controls."""

    config = _mapping(policy.get("candidate_risk_schedule"))
    limits = _mapping(config.get("limits"))
    row = _mapping(snapshot)
    checks: dict[str, dict[str, Any]] = {}

    def upper(name: str, observed: Any, limit: Any) -> None:
        value = _number(observed, math.inf)
        ceiling = _number(limit, 0.0)
        checks[name] = {"ok": value <= ceiling, "observed": value, "limit": ceiling}

    checks["candidate_binding"] = {
        "ok": bool(candidate_id and str(row.get("candidate_id") or "") == candidate_id),
        "observed": str(row.get("candidate_id") or ""),
        "expected": str(candidate_id or ""),
    }
    upper(
        "broker_truth_freshness",
        row.get("broker_truth_age_seconds"),
        config.get("max_broker_truth_age_seconds"),
    )
    upper(
        "gross_exposure",
        row.get("gross_exposure_ratio"),
        limits.get("gross_exposure_ratio"),
    )
    upper(
        "absolute_net_exposure",
        abs(_number(row.get("net_exposure_ratio"), math.inf)),
        limits.get("absolute_net_exposure_ratio"),
    )
    upper("drawdown", row.get("drawdown_ratio"), limits.get("drawdown_ratio"))
    upper("daily_loss", row.get("daily_loss_ratio"), limits.get("daily_loss_ratio"))
    upper(
        "buying_power_utilization",
        row.get("buying_power_utilization_ratio"),
        limits.get("buying_power_utilization_ratio"),
    )
    symbol_weights = _mapping(row.get("symbol_weights"))
    sleeve_weights = _mapping(row.get("sleeve_weights"))
    upper(
        "symbol_concentration",
        max((_number(value) for value in symbol_weights.values()), default=math.inf),
        limits.get("symbol_weight"),
    )
    upper(
        "sleeve_concentration",
        max((_number(value) for value in sleeve_weights.values()), default=math.inf),
        limits.get("sleeve_weight"),
    )
    asset_types = {str(value).upper() for value in row.get("asset_types") or []}
    instruction_types = {
        str(value).upper() for value in row.get("instruction_types") or []
    }
    allowed_assets = {
        str(value).upper() for value in config.get("allowed_asset_types") or []
    }
    allowed_instructions = {
        str(value).upper() for value in config.get("allowed_instruction_types") or []
    }
    checks["asset_allowlist"] = {
        "ok": bool(asset_types) and asset_types <= allowed_assets,
        "observed": sorted(asset_types),
        "allowed": sorted(allowed_assets),
    }
    checks["instruction_allowlist"] = {
        "ok": bool(instruction_types) and instruction_types <= allowed_instructions,
        "observed": sorted(instruction_types),
        "allowed": sorted(allowed_instructions),
    }
    ready = bool(checks and all(check["ok"] for check in checks.values()))
    return {
        "ok": ready,
        "status": "ready" if ready else "blocked",
        "schedule_id": str(config.get("schedule_id") or ""),
        "candidate_id": str(candidate_id or ""),
        "checks": checks,
        "failed_checks": sorted(
            name for name, check in checks.items() if not check["ok"]
        ),
        "advisory_only": True,
        "changes_risk_limits": False,
        "pretrade_authority": False,
        "execution_authority": False,
    }


def evaluate_execution_frontier(
    alternatives: Sequence[Mapping[str, Any]],
    *,
    candidate_id: str,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare speed and expected post-cost capture, abstaining on thin evidence."""

    config = _mapping(policy.get("execution_frontier"))
    minimum_fills = int(config.get("minimum_independent_fills") or 0)
    minimum_probability = _number(config.get("minimum_fill_probability"), 0.0)
    max_participation = _number(config.get("maximum_participation_rate"), 0.0)
    minimum_capture = _number(config.get("minimum_positive_post_cost_capture_bps"), 0.0)
    cost_terms = [str(value) for value in config.get("cost_terms") or []]
    rows: list[dict[str, Any]] = []
    for raw in alternatives:
        row = _mapping(raw)
        missing = [
            term
            for term in [
                "alternative_id",
                "expected_alpha_bps",
                "fill_probability",
                "participation_rate",
                "independent_fill_count",
                *cost_terms,
            ]
            if row.get(term) is None
        ]
        gross = _number(row.get("expected_alpha_bps"))
        total_cost = sum(_number(row.get(term)) for term in cost_terms)
        post_cost = gross - total_cost
        fill_probability = _number(row.get("fill_probability"))
        participation = _number(row.get("participation_rate"))
        fill_count = int(_number(row.get("independent_fill_count")))
        candidate_match = str(row.get("candidate_id") or candidate_id) == candidate_id
        evidence_ready = bool(
            not missing
            and candidate_id
            and candidate_match
            and fill_count >= minimum_fills
            and fill_probability >= minimum_probability
            and participation <= max_participation
        )
        rows.append(
            {
                "alternative_id": str(row.get("alternative_id") or ""),
                "candidate_id": str(row.get("candidate_id") or candidate_id),
                "missing_fields": missing,
                "gross_alpha_bps": gross,
                "total_cost_bps": total_cost,
                "post_cost_capture_bps": post_cost,
                "fill_probability": fill_probability,
                "expected_realized_capture_bps": post_cost * fill_probability,
                "participation_rate": participation,
                "independent_fill_count": fill_count,
                "evidence_ready": evidence_ready,
                "eligible": bool(evidence_ready and post_cost > minimum_capture),
            }
        )
    eligible = [row for row in rows if row["eligible"]]
    selected = (
        max(eligible, key=lambda row: row["expected_realized_capture_bps"])
        if eligible
        else None
    )
    return {
        "ok": bool(selected),
        "status": "frontier_ready" if selected else "abstain",
        "frontier_id": str(config.get("frontier_id") or ""),
        "candidate_id": str(candidate_id or ""),
        "selected_alternative_id": selected["alternative_id"] if selected else "",
        "alternatives": rows,
        "abstain": selected is None,
        "advisory_only": True,
        "order_authority": False,
        "execution_authority": False,
    }


def _topological_order(stages: Sequence[Mapping[str, Any]]) -> list[str]:
    dependencies = {
        str(_mapping(stage).get("stage_id") or ""): {
            str(value) for value in _mapping(stage).get("depends_on") or []
        }
        for stage in stages
    }
    if not dependencies or "" in dependencies:
        raise ValueError("research_dag_stage_id_missing")
    unknown = sorted(
        {
            dependency
            for values in dependencies.values()
            for dependency in values
            if dependency not in dependencies
        }
    )
    if unknown:
        raise ValueError(f"research_dag_unknown_dependencies:{','.join(unknown)}")
    order: list[str] = []
    remaining = {stage: set(values) for stage, values in dependencies.items()}
    while remaining:
        ready = sorted(stage for stage, values in remaining.items() if not values)
        if not ready:
            raise ValueError("research_dag_cycle_detected")
        for stage in ready:
            order.append(stage)
            remaining.pop(stage)
        for values in remaining.values():
            values.difference_update(ready)
    return order


def plan_research_dag(
    *,
    candidate_id: str,
    code_receipt: str,
    dataset_receipts: Mapping[str, str],
    checkpoint_receipts: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Plan deterministic reuse or rerun; this function never launches a stage."""

    config = _mapping(policy.get("research_dag"))
    stages = [_mapping(row) for row in config.get("stages") or []]
    by_id = {str(row.get("stage_id") or ""): row for row in stages}
    order = _topological_order(stages)
    expected: dict[str, str] = {}
    actions: list[dict[str, Any]] = []
    action_by_stage: dict[str, str] = {}
    for stage_id in order:
        dependencies = [str(value) for value in by_id[stage_id].get("depends_on") or []]
        material = {
            "dag_id": str(config.get("dag_id") or ""),
            "stage_id": stage_id,
            "candidate_id": str(candidate_id or ""),
            "code_receipt": str(code_receipt or ""),
            "dataset_receipts": dict(
                sorted(
                    (str(key), str(value)) for key, value in dataset_receipts.items()
                )
            ),
            "dependency_receipts": {
                dependency: expected[dependency] for dependency in dependencies
            },
        }
        receipt = canonical_sha256(material)
        expected[stage_id] = receipt
        raw_checkpoint = checkpoint_receipts.get(stage_id)
        actual = str(
            _mapping(raw_checkpoint).get("receipt_sha256") or raw_checkpoint or ""
        )
        stale_dependency = any(
            action_by_stage.get(dependency) != "reuse" for dependency in dependencies
        )
        action = "reuse" if actual == receipt and not stale_dependency else "run"
        action_by_stage[stage_id] = action
        actions.append(
            {
                "stage_id": stage_id,
                "depends_on": dependencies,
                "action": action,
                "reason": (
                    "checkpoint_receipt_match"
                    if action == "reuse"
                    else "dependency_or_input_receipt_changed"
                ),
                "expected_receipt_sha256": receipt,
                "checkpoint_receipt_sha256": actual,
            }
        )
    valid_inputs = bool(candidate_id and code_receipt and dataset_receipts)
    return {
        "ok": valid_inputs,
        "status": "ready" if valid_inputs else "blocked",
        "dag_id": str(config.get("dag_id") or ""),
        "candidate_id": str(candidate_id or ""),
        "topological_order": order,
        "stages": actions,
        "reuse_count": sum(1 for row in actions if row["action"] == "reuse"),
        "run_count": sum(1 for row in actions if row["action"] == "run"),
        "launch_authority": False,
        "execution_authority": False,
    }


def build_dataset_version(
    dataset_id: str,
    entries: Sequence[Mapping[str, Any]],
    *,
    committed_at_utc: str,
    parent_version_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = sorted(
        [
            {
                "path": str(_mapping(row).get("path") or ""),
                "sha256": str(_mapping(row).get("sha256") or ""),
                "size_bytes": int(_number(_mapping(row).get("size_bytes"))),
                "row_count": int(_number(_mapping(row).get("row_count"))),
            }
            for row in entries
        ],
        key=lambda row: row["path"],
    )
    if (
        not dataset_id
        or not normalized
        or any(not row["path"] or len(row["sha256"]) != 64 for row in normalized)
    ):
        raise ValueError("dataset_version_manifest_invalid")
    material = {
        "dataset_id": str(dataset_id),
        "parent_version_id": str(parent_version_id or ""),
        "committed_at_utc": str(committed_at_utc),
        "entries": normalized,
        "metadata": dict(metadata or {}),
    }
    return {
        **material,
        "version_id": canonical_sha256(material),
        "immutable": True,
        "source_retirement_authority": False,
    }


def verify_dataset_versions(versions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = sorted(
        [_mapping(row) for row in versions],
        key=lambda row: _parse_timestamp(row.get("committed_at_utc"))
        or datetime.min.replace(tzinfo=timezone.utc),
    )
    errors: list[str] = []
    previous = ""
    for index, row in enumerate(rows):
        material = {
            key: row.get(key)
            for key in (
                "dataset_id",
                "parent_version_id",
                "committed_at_utc",
                "entries",
                "metadata",
            )
        }
        expected = canonical_sha256(material)
        if str(row.get("version_id") or "") != expected:
            errors.append(f"version_hash_mismatch:{index}")
        if str(row.get("parent_version_id") or "") != previous:
            errors.append(f"parent_version_mismatch:{index}")
        previous = str(row.get("version_id") or "")
    return {
        "ok": bool(rows and not errors),
        "version_count": len(rows),
        "head_version_id": previous,
        "errors": errors,
        "source_retirement_authority": False,
    }


def select_dataset_version(
    versions: Sequence[Mapping[str, Any]], *, as_of_utc: str
) -> dict[str, Any] | None:
    cutoff = _parse_timestamp(as_of_utc)
    if cutoff is None:
        raise ValueError("dataset_as_of_timestamp_invalid")
    eligible = [
        _mapping(row)
        for row in versions
        if (timestamp := _parse_timestamp(_mapping(row).get("committed_at_utc")))
        is not None
        and timestamp <= cutoff
    ]
    return (
        max(eligible, key=lambda row: _parse_timestamp(row.get("committed_at_utc")))
        if eligible
        else None
    )


def reconcile_cross_engine_valuation(
    *,
    candidate_id: str,
    product_id: str,
    valuation_time_utc: str,
    measures: Sequence[str],
    primary_engine: Mapping[str, Any],
    oracle_engine: Mapping[str, Any],
    policy: Mapping[str, Any],
    synthetic_probe: bool = False,
) -> dict[str, Any]:
    config = _mapping(policy.get("valuation_reconciliation"))
    request = build_risk_request(
        candidate_id=candidate_id,
        product_id=product_id,
        valuation_time_utc=valuation_time_utc,
        measures=list(measures),
    )
    primary = _mapping(primary_engine)
    oracle = _mapping(oracle_engine)
    primary.setdefault("request_sha256", request["request_sha256"])
    oracle.setdefault("request_sha256", request["request_sha256"])
    reconciliation = reconcile_risk_results(
        primary,
        oracle,
        absolute_tolerance={
            str(key): _number(value)
            for key, value in _mapping(config.get("absolute_tolerance")).items()
        },
        relative_tolerance=_number(config.get("relative_tolerance"), 1e-6),
        synthetic_probe=synthetic_probe,
    )
    return {
        **reconciliation,
        "contract_id": str(config.get("contract_id") or ""),
        "candidate_id": str(candidate_id or ""),
        "product_id": str(product_id or ""),
        "request": request,
        "engine_count": 2,
        "minimum_distinct_engines": int(config.get("minimum_distinct_engines") or 2),
        "connected_external_api_required": False,
        "order_authority": False,
    }


def decision_extension_metadata(
    *, decision_family_id: str, policy: Mapping[str, Any]
) -> dict[str, Any]:
    material = {
        "policy_id": str(policy.get("policy_id") or ""),
        "decision_family_id": str(decision_family_id or ""),
        "control_ids": list(REQUIRED_CONTROL_IDS),
        "factor_benchmark_ids": list(
            _mapping(policy.get("factor_benchmarks")).get("benchmark_ids") or []
        ),
        "risk_schedule_id": str(
            _mapping(policy.get("candidate_risk_schedule")).get("schedule_id") or ""
        ),
        "execution_frontier_id": str(
            _mapping(policy.get("execution_frontier")).get("frontier_id") or ""
        ),
        "research_dag_id": str(
            _mapping(policy.get("research_dag")).get("dag_id") or ""
        ),
    }
    return {
        **material,
        "receipt_sha256": canonical_sha256(material),
        "metadata_only": True,
        "existing_route_authority_unchanged": True,
        "execution_authority": False,
    }


def structural_probe(
    policy: Mapping[str, Any], *, project_root: str | Path = PROJECT_ROOT
) -> dict[str, Any]:
    errors: list[str] = []
    root = Path(project_root)
    controls = _mapping(policy.get("controls"))
    required = [str(value) for value in policy.get("required_control_ids") or []]
    authority = _mapping(policy.get("authority"))
    influences = [_mapping(row) for row in policy.get("firm_influences") or []]
    if int(policy.get("schema_version") or 0) != 1:
        errors.append("schema_version_must_be_1")
    if tuple(required) != REQUIRED_CONTROL_IDS or set(controls) != set(
        REQUIRED_CONTROL_IDS
    ):
        errors.append("eight_control_registry_mismatch")
    if authority.get("advisory_only") is not True:
        errors.append("advisory_only_authority_missing")
    for key, value in authority.items():
        if str(key).startswith("can_") and bool(value):
            errors.append(f"forbidden_authority_enabled:{key}")
    for control_id in REQUIRED_CONTROL_IDS:
        row = _mapping(controls.get(control_id))
        if not (root / str(row.get("owner") or "")).is_file():
            errors.append(f"control_owner_missing:{control_id}")
        if not (root / str(row.get("test") or "")).is_file():
            errors.append(f"control_test_missing:{control_id}")
        if not row.get("evidence_artifact"):
            errors.append(f"control_evidence_artifact_missing:{control_id}")
    adopted = {control_id: 0 for control_id in REQUIRED_CONTROL_IDS}
    organizations: set[str] = set()
    reference_ids: set[str] = set()
    for row in influences:
        reference_id = str(row.get("reference_id") or "")
        parsed = urlparse(str(row.get("official_url") or ""))
        if not reference_id or reference_id in reference_ids:
            errors.append(f"firm_reference_id_invalid:{reference_id or 'missing'}")
        reference_ids.add(reference_id)
        organizations.add(str(row.get("organization") or ""))
        if parsed.scheme != "https" or not parsed.netloc:
            errors.append(f"firm_reference_url_invalid:{reference_id}")
        for control_id in row.get("control_ids") or []:
            if str(control_id) not in adopted:
                errors.append(
                    f"firm_reference_unknown_control:{reference_id}:{control_id}"
                )
            else:
                adopted[str(control_id)] += 1
    if len(influences) < 10 or len(organizations) < 6:
        errors.append("firm_influence_coverage_below_contract")
    for control_id, count in adopted.items():
        if count <= 0:
            errors.append(f"control_without_firm_influence:{control_id}")
    return {
        "ok": not errors,
        "errors": errors,
        "control_count": len(controls),
        "ready_control_count": (
            len(controls) if not errors else max(len(controls) - len(errors), 0)
        ),
        "firm_reference_count": len(influences),
        "firm_organization_count": len(organizations),
        "control_adoption": adopted,
        "authority_safe": not any(
            error.startswith("forbidden_authority") for error in errors
        ),
        "live_execution_authority": False,
    }
