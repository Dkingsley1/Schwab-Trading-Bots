#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import re
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.accountability import safe_write_json_atomic
from scripts.ops.paper_behavior_intervention_drill import (
    build_payload as build_behavior_payload,
    publish_payload as publish_behavior_payload,
)
from scripts.ops.profitability_adversarial_drill import (
    build_payload as build_adversarial_payload,
    publish_payload as publish_adversarial_payload,
)
from scripts.ops.profitability_crisis_drill import (
    build_payload as build_crisis_payload,
    publish_payload as publish_crisis_payload,
)

DEFAULT_POLICY_PATH = Path("config/trading_behavior_drill_program_v1.json")
DEFAULT_CANDIDATE_PATH = Path("governance/runtime/production_candidate_state.json")
DEFAULT_OUT_PATH = Path(
    "governance/research/trading_behavior_drill_program_latest.json"
)
DEFAULT_LOCK_PATH = Path("governance/locks/trading_behavior_drill_program.lock")

SuiteBuilder = Callable[..., dict[str, Any]]
SUITE_BUILDERS: dict[str, SuiteBuilder] = {
    "profitability_crisis": build_crisis_payload,
    "profitability_adversarial": build_adversarial_payload,
    "paper_behavior_intervention": build_behavior_payload,
}
SUITE_PUBLISHERS = {
    "profitability_crisis": publish_crisis_payload,
    "profitability_adversarial": publish_adversarial_payload,
    "paper_behavior_intervention": publish_behavior_payload,
}

FORBIDDEN_AUTHORITY_KEYS = {
    "network_access",
    "broker_access",
    "market_data_access",
    "paper_order_authority",
    "live_order_authority",
    "automatic_promotion_authority",
    "candidate_mutation_authority",
    "historical_outcome_rewrite_authority",
    "training_label_authority",
    "allocation_authority",
    "risk_limit_mutation_authority",
    "automatic_threshold_tuning_authority",
    "can_write_runtime_control",
    "can_submit_paper_orders",
    "can_submit_live_orders",
    "can_access_broker",
    "can_access_network",
    "can_mutate_candidate",
    "can_promote",
    "can_change_allocation",
    "can_relax_risk",
    "can_change_live_execution",
}
RESOURCE_ZERO_KEYS = {
    "persistent_processes_started",
    "network_requests",
    "broker_requests",
    "orders_submitted",
    "paper_orders_submitted",
    "live_orders_submitted",
    "candidate_mutations",
    "runtime_control_writes",
}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[dict[str, Any]]:
    return (
        [dict(row) for row in value if isinstance(row, Mapping)]
        if isinstance(value, list)
        else []
    )


def _float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve(root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else root / path


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _grade(score: float) -> str:
    if score >= 99.5:
        return "A+"
    if score >= 93.0:
        return "A"
    if score >= 87.0:
        return "B+"
    if score >= 80.0:
        return "B"
    if score >= 70.0:
        return "C"
    if score >= 60.0:
        return "D"
    return "F"


def _generated_at(value: str | None) -> str:
    raw = str(value or "").strip().replace("Z", "+00:00")
    if raw:
        try:
            parsed = datetime.fromisoformat(raw)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc).isoformat()
        except ValueError:
            pass
    return datetime.now(timezone.utc).isoformat()


def _candidate_binding(root: Path) -> tuple[dict[str, Any], Path, str]:
    path = root / DEFAULT_CANDIDATE_PATH
    candidate = _load_json(path)
    receipt = _file_sha256(path)
    binding = {
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "candidate_generation": _int(candidate.get("generation"), 0),
        "accepted_at_utc": str(candidate.get("accepted_at_utc") or ""),
        "accepted_git_head": str(candidate.get("accepted_git_head") or ""),
        "candidate_state_sha256": receipt,
        "live_execution_authority": bool(
            candidate.get("live_execution_authority", False)
        ),
    }
    binding["valid"] = bool(
        binding["candidate_id"]
        and binding["candidate_generation"] > 0
        and binding["accepted_at_utc"]
        and receipt
    )
    return binding, path, receipt


def _suite_authority_safe(payload: Mapping[str, Any]) -> tuple[bool, list[str]]:
    authority = _mapping(payload.get("authority_contract"))
    if not authority:
        return False, ["authority_contract_missing"]
    failures = [
        key
        for key in sorted(FORBIDDEN_AUTHORITY_KEYS)
        if key in authority and authority.get(key) is not False
    ]
    return not failures, failures


def _suite_resources_safe(payload: Mapping[str, Any]) -> tuple[bool, list[str]]:
    resources = _mapping(payload.get("resource_contract"))
    failures = [
        key
        for key in sorted(RESOURCE_ZERO_KEYS)
        if key in resources and _float(resources.get(key), 0.0) != 0.0
    ]
    required = {
        "network_requests",
        "broker_requests",
    }
    if "orders_submitted" not in resources:
        required.update({"paper_orders_submitted", "live_orders_submitted"})
    failures.extend(
        f"{key}_missing" for key in sorted(required) if key not in resources
    )
    return not failures, failures


def _scenario_counts(suite_id: str, payload: Mapping[str, Any]) -> tuple[int, int, int]:
    if suite_id == "paper_behavior_intervention":
        summary = _mapping(payload.get("scenario_summary"))
        total = _int(summary.get("executed_scenario_count"), 0)
        passed = _int(summary.get("passed_scenario_count"), 0)
        cases = _int(summary.get("case_count"), 0)
        return total, passed, cases
    if suite_id == "profitability_adversarial":
        summary = _mapping(payload.get("diagnostic_summary"))
        total = _int(summary.get("scenario_count"), 0)
        passed = _int(summary.get("passed_scenario_count"), 0)
        return total, passed, _int(summary.get("scenario_check_count"), 0)
    scenarios = _rows(payload.get("scenario_results"))
    total = len(scenarios)
    passed = sum(bool(row.get("ok", False)) for row in scenarios)
    return (
        total,
        passed,
        _int(_mapping(payload.get("diagnostic_summary")).get("phase_count"), 0),
    )


def _suite_summary(
    *,
    suite_id: str,
    suite_spec: Mapping[str, Any],
    payload: Mapping[str, Any],
    duration_seconds: float,
) -> dict[str, Any]:
    authority_safe, authority_failures = _suite_authority_safe(payload)
    resources_safe, resource_failures = _suite_resources_safe(payload)
    scenario_count, passed_count, case_or_check_count = _scenario_counts(
        suite_id, payload
    )
    candidate = _mapping(payload.get("candidate_binding"))
    resources = _mapping(payload.get("resource_contract"))
    return {
        "suite_id": suite_id,
        "purpose": str(suite_spec.get("purpose") or ""),
        "required_for_admission": bool(suite_spec.get("required_for_admission", False)),
        "status": str(payload.get("status") or "degraded"),
        "ok": bool(payload.get("ok", False)),
        "control_grade": str(payload.get("control_grade") or "F"),
        "control_score": round(_float(payload.get("control_score"), 0.0), 4),
        "scenario_count": scenario_count,
        "passed_scenario_count": passed_count,
        "case_or_check_count": case_or_check_count,
        "failed_check_count": len(payload.get("failed_checks") or []),
        "work_units": max(_int(resources.get("work_units"), case_or_check_count), 0),
        "duration_seconds": round(max(duration_seconds, 0.0), 6),
        "authority_safe": authority_safe,
        "authority_failures": authority_failures,
        "resources_safe": resources_safe,
        "resource_failures": resource_failures,
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "candidate_generation": _int(candidate.get("candidate_generation"), 0),
        "payload_sha256": _canonical_sha256(payload),
        "latest_artifact": str(suite_spec.get("latest_artifact") or ""),
    }


def _comparison(
    *,
    current: Sequence[Mapping[str, Any]],
    previous: Mapping[str, Any],
    input_signature_sha256: str,
    tolerance: float,
) -> dict[str, Any]:
    previous_receipts = _mapping(previous.get("receipts"))
    comparable = bool(
        previous
        and previous_receipts.get("input_signature_sha256") == input_signature_sha256
    )
    previous_rows = {
        str(row.get("suite_id") or ""): row
        for row in _rows(previous.get("suite_results"))
    }
    current_ids = {str(row.get("suite_id") or "") for row in current}
    regressions: list[dict[str, Any]] = []
    deltas: list[dict[str, Any]] = []
    comparable = bool(
        comparable
        and previous.get("ok", False)
        and current_ids
        and current_ids.issubset(previous_rows)
    )
    if comparable:
        for row in current:
            suite_id = str(row.get("suite_id") or "")
            prior = previous_rows.get(suite_id)
            if not prior:
                regressions.append(
                    {"suite_id": suite_id, "reason": "prior_suite_missing"}
                )
                continue
            score_delta = round(
                _float(row.get("control_score")) - _float(prior.get("control_score")),
                4,
            )
            passed_delta = _int(row.get("passed_scenario_count")) - _int(
                prior.get("passed_scenario_count")
            )
            failed_check_delta = _int(row.get("failed_check_count")) - _int(
                prior.get("failed_check_count")
            )
            deltas.append(
                {
                    "suite_id": suite_id,
                    "control_score_delta": score_delta,
                    "passed_scenario_count_delta": passed_delta,
                    "failed_check_count_delta": failed_check_delta,
                }
            )
            if score_delta < -abs(tolerance):
                regressions.append(
                    {"suite_id": suite_id, "reason": "control_score_regressed"}
                )
            if passed_delta < 0:
                regressions.append(
                    {"suite_id": suite_id, "reason": "scenario_pass_count_regressed"}
                )
            if failed_check_delta > 0:
                regressions.append(
                    {"suite_id": suite_id, "reason": "failed_check_count_increased"}
                )
    return {
        "comparable": comparable,
        "status": (
            "regressed"
            if regressions
            else "non_regressed" if comparable else "baseline_established"
        ),
        "non_regressed": not regressions,
        "regressions": regressions,
        "suite_deltas": deltas,
        "prior_run_id": str(previous.get("run_id") or ""),
        "policy": "Only same-candidate, same-policy-input runs are compared; unrelated generations never manufacture a regression.",
    }


def _refresh_program_status(payload: dict[str, Any]) -> None:
    checks = _mapping(payload.get("checks"))
    passed = sum(bool(value) for value in checks.values())
    score = round(100.0 * passed / max(len(checks), 1), 4)
    ready = bool(checks) and all(checks.values())
    payload["control_score"] = score
    payload["control_grade"] = _grade(score)
    payload["ok"] = ready
    payload["status"] = "ready" if ready else "degraded"
    payload["failed_checks"] = [name for name, value in checks.items() if not value]
    behavior = _mapping(payload.get("behavior_change_contract"))
    behavior["eligible_for_single_writer_admission"] = bool(
        ready and behavior.get("intervention_suite_admission_eligible", False)
    )
    behavior["decision"] = (
        "propose_to_single_runtime_writer"
        if behavior["eligible_for_single_writer_admission"]
        else "reject_without_runtime_change"
    )
    payload["behavior_change_contract"] = behavior
    payload["admission_eligible"] = bool(
        behavior.get("eligible_for_single_writer_admission", False)
    )


def _execute_program(
    *,
    project_root: Path = PROJECT_ROOT,
    policy_path: str | Path = DEFAULT_POLICY_PATH,
    requested_suites: Sequence[str] | None = None,
    generated_at_utc: str | None = None,
    previous_payload: Mapping[str, Any] | None = None,
    suite_builders: Mapping[str, SuiteBuilder] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    root = Path(project_root).expanduser().resolve()
    generated_at = _generated_at(generated_at_utc)
    policy_file = _resolve(root, policy_path)
    policy = _load_json(policy_file)
    suite_specs = _rows(policy.get("suite_order"))
    all_suite_ids = [str(row.get("suite_id") or "") for row in suite_specs]
    requested = [
        str(value).strip() for value in requested_suites or [] if str(value).strip()
    ]
    full_mode = not requested or "all" in {value.lower() for value in requested}
    selected_ids = all_suite_ids if full_mode else requested
    unresolved = [value for value in selected_ids if value not in all_suite_ids]
    selected_specs = [
        row for row in suite_specs if str(row.get("suite_id") or "") in selected_ids
    ]
    candidate_binding, candidate_path, candidate_sha_before = _candidate_binding(root)
    policy_receipts = {
        str(row.get("suite_id") or ""): {
            "path": str(_resolve(root, str(row.get("policy_path") or ""))),
            "sha256": _file_sha256(_resolve(root, str(row.get("policy_path") or ""))),
        }
        for row in selected_specs
    }
    program_authority = _mapping(policy.get("authority_contract"))
    program_forbidden = [
        key
        for key in sorted(FORBIDDEN_AUTHORITY_KEYS)
        if key in program_authority and program_authority.get(key) is not False
    ]
    preflight_checks = {
        "program_policy_loaded": bool(policy),
        "program_schema_supported": _int(policy.get("schema_version"), 0) == 1,
        "candidate_binding_valid": bool(candidate_binding.get("valid", False)),
        "candidate_has_no_live_authority": candidate_binding.get(
            "live_execution_authority"
        )
        is False,
        "suite_selection_resolved": not unresolved,
        "suite_selection_nonempty": bool(selected_specs),
        "suite_builders_available": all(
            suite_id in (suite_builders or SUITE_BUILDERS)
            for suite_id in selected_ids
            if suite_id not in unresolved
        ),
        "suite_policy_receipts_present": bool(policy_receipts)
        and all(row.get("sha256") for row in policy_receipts.values()),
        "program_has_zero_forbidden_authority": bool(program_authority)
        and not program_forbidden,
    }
    input_signature = _canonical_sha256(
        {
            "program_policy_sha256": _file_sha256(policy_file),
            "candidate_state_sha256": candidate_sha_before,
            "candidate_id": candidate_binding.get("candidate_id"),
            "candidate_generation": candidate_binding.get("candidate_generation"),
            "selected_suite_ids": selected_ids,
            "suite_policy_receipts": policy_receipts,
        }
    )
    compact_time = re.sub(r"[^0-9]", "", generated_at)[:20]
    run_id = f"tdp-{compact_time}-{input_signature[:12]}"
    suite_payloads: dict[str, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    builders = dict(suite_builders or SUITE_BUILDERS)
    started = time.monotonic()
    candidate_mutated_mid_run = False
    if all(preflight_checks.values()):
        for spec in selected_specs:
            suite_id = str(spec.get("suite_id") or "")
            suite_started = time.monotonic()
            try:
                payload = builders[suite_id](
                    project_root=root,
                    policy_path=_resolve(root, str(spec.get("policy_path") or "")),
                    generated_at_utc=generated_at,
                )
            except (
                Exception
            ) as exc:  # pragma: no cover - exercised through failure contract
                payload = {
                    "timestamp_utc": generated_at,
                    "status": "degraded",
                    "ok": False,
                    "control_grade": "F",
                    "control_score": 0.0,
                    "candidate_binding": candidate_binding,
                    "checks": {"suite_completed_without_exception": False},
                    "failed_checks": ["suite_completed_without_exception"],
                    "authority_contract": {},
                    "resource_contract": {},
                    "error": f"{type(exc).__name__}:{exc}",
                }
            suite_payloads[suite_id] = payload
            summaries.append(
                _suite_summary(
                    suite_id=suite_id,
                    suite_spec=spec,
                    payload=payload,
                    duration_seconds=time.monotonic() - suite_started,
                )
            )
            if _file_sha256(candidate_path) != candidate_sha_before:
                candidate_mutated_mid_run = True
                break
    candidate_sha_after = _file_sha256(candidate_path)
    execution_contract = _mapping(policy.get("execution_contract"))
    admission_contract = _mapping(policy.get("admission_contract"))
    required_ids = {
        str(row.get("suite_id") or "")
        for row in suite_specs
        if row.get("required_for_admission", False)
    }
    executed_ids = {str(row.get("suite_id") or "") for row in summaries}
    required_grade = str(admission_contract.get("required_control_grade") or "A+")
    total_work_units = sum(_int(row.get("work_units"), 0) for row in summaries)
    elapsed_seconds = max(time.monotonic() - started, 0.0)
    previous = dict(previous_payload or {})
    comparison = _comparison(
        current=summaries,
        previous=previous,
        input_signature_sha256=input_signature,
        tolerance=_float(
            execution_contract.get("control_score_regression_tolerance"), 0.0
        ),
    )
    behavior_payload = suite_payloads.get("paper_behavior_intervention", {})
    behavior_proposal = _mapping(behavior_payload.get("runtime_overlay_proposal"))
    checks = dict(preflight_checks)
    checks.update(
        {
            "full_required_suite_set_executed": full_mode
            and required_ids == executed_ids,
            "all_required_suites_ready": required_ids == executed_ids
            and all(
                bool(row.get("ok", False))
                and row.get("control_grade") == required_grade
                for row in summaries
                if row.get("suite_id") in required_ids
            ),
            "all_suite_authority_contracts_safe": bool(summaries)
            and all(row.get("authority_safe", False) for row in summaries),
            "all_suite_resource_contracts_safe": bool(summaries)
            and all(row.get("resources_safe", False) for row in summaries),
            "candidate_state_unchanged": bool(candidate_sha_before)
            and candidate_sha_before == candidate_sha_after
            and not candidate_mutated_mid_run,
            "runtime_budget_met": elapsed_seconds
            <= max(
                _float(execution_contract.get("maximum_runtime_seconds"), 180.0), 0.001
            ),
            "work_unit_budget_met": total_work_units
            <= max(_int(execution_contract.get("maximum_total_work_units"), 15000), 1),
            "comparable_run_non_regressed": bool(
                comparison.get("non_regressed", False)
            ),
            "behavior_proposal_complete": bool(
                behavior_payload.get("admission_eligible", False)
            )
            and bool(behavior_proposal)
            and bool(behavior_payload.get("runtime_overlay_proposal_sha256")),
        }
    )
    payload: dict[str, Any] = {
        "timestamp_utc": generated_at,
        "schema_version": 1,
        "program_id": str(policy.get("program_id") or ""),
        "run_id": run_id,
        "operating_mode": str(policy.get("operating_mode") or ""),
        "lifecycle": {
            "state": "evaluated",
            "stages": [
                "candidate_and_policy_snapshot",
                "preflight_authority_and_resource_validation",
                "deterministic_suite_execution",
                "candidate_mutation_verification",
                "comparable_run_regression_check",
                "evidence_publication",
                "single_writer_admission_or_rejection",
                "candidate_forward_post_cost_observation",
            ],
            "elapsed_seconds": round(elapsed_seconds, 6),
            "full_suite_mode": full_mode,
            "requested_suites": requested or ["all"],
            "unresolved_suites": unresolved,
        },
        "preflight": {
            "ok": all(preflight_checks.values()),
            "checks": preflight_checks,
            "failed_checks": [
                name for name, value in preflight_checks.items() if not value
            ],
            "forbidden_authority_failures": program_forbidden,
        },
        "candidate_binding": candidate_binding,
        "candidate_mutation_guard": {
            "candidate_path": str(candidate_path),
            "sha256_before": candidate_sha_before,
            "sha256_after": candidate_sha_after,
            "unchanged": candidate_sha_before == candidate_sha_after,
            "mutation_detected_mid_run": candidate_mutated_mid_run,
        },
        "suite_results": summaries,
        "suite_artifacts": {
            str(row.get("suite_id") or ""): {
                "path": str(row.get("latest_artifact") or ""),
                "payload_sha256": str(row.get("payload_sha256") or ""),
            }
            for row in summaries
        },
        "comparison": comparison,
        "checks": checks,
        "resource_contract": {
            "persistent_processes_started": 0,
            "network_requests": 0,
            "broker_requests": 0,
            "paper_orders_submitted": 0,
            "live_orders_submitted": 0,
            "candidate_mutations": int(candidate_sha_before != candidate_sha_after),
            "runtime_control_writes": 0,
            "total_work_units": total_work_units,
            "maximum_total_work_units": max(
                _int(execution_contract.get("maximum_total_work_units"), 15000), 1
            ),
            "elapsed_seconds": round(elapsed_seconds, 6),
            "maximum_runtime_seconds": max(
                _float(execution_contract.get("maximum_runtime_seconds"), 180.0),
                0.001,
            ),
        },
        "authority_contract": program_authority,
        "behavior_change_contract": {
            "intervention_suite_admission_eligible": bool(
                behavior_payload.get("admission_eligible", False)
            ),
            "runtime_overlay_proposal_sha256": str(
                behavior_payload.get("runtime_overlay_proposal_sha256") or ""
            ),
            "candidate_id": str(candidate_binding.get("candidate_id") or ""),
            "candidate_generation": _int(
                candidate_binding.get("candidate_generation"), 0
            ),
            "single_runtime_writer": str(
                admission_contract.get("single_runtime_writer") or ""
            ),
            "allowed_effects": list(
                _mapping(policy.get("positive_behavior_contract")).get(
                    "allowed_effects"
                )
                or []
            ),
            "prohibited_effects": list(
                _mapping(policy.get("positive_behavior_contract")).get(
                    "prohibited_effects"
                )
                or []
            ),
            "candidate_forward_post_cost_evidence_required": True,
            "automatic_risk_widening_allowed": False,
            "profitability_claim_allowed": False,
            "live_execution_allowed": False,
        },
        "evidence_classification": {
            "deterministic_diagnostic_evidence": True,
            "candidate_bound_behavior_proposal": True,
            "organic_candidate_profitability_evidence": False,
            "promotion_evidence": False,
            "live_release_evidence": False,
            "profitability_guarantee": False,
        },
        "receipts": {
            "program_policy_path": str(policy_file),
            "program_policy_sha256": _file_sha256(policy_file),
            "candidate_state_sha256": candidate_sha_after,
            "input_signature_sha256": input_signature,
            "suite_policy_receipts": policy_receipts,
        },
        "next_actions": [
            "Admit a complete non-regressed proposal only through paper-profitability-control --apply.",
            "Observe intervention-tagged candidate-forward post-cost paper fills before reviewing thresholds.",
            "Treat diagnostic improvements as mechanics evidence, never as a profitability or live-release claim.",
        ],
    }
    _refresh_program_status(payload)
    return payload, suite_payloads


def build_payload(
    *,
    project_root: Path = PROJECT_ROOT,
    policy_path: str | Path = DEFAULT_POLICY_PATH,
    requested_suites: Sequence[str] | None = None,
    generated_at_utc: str | None = None,
    previous_payload: Mapping[str, Any] | None = None,
    suite_builders: Mapping[str, SuiteBuilder] | None = None,
) -> dict[str, Any]:
    payload, _ = _execute_program(
        project_root=project_root,
        policy_path=policy_path,
        requested_suites=requested_suites,
        generated_at_utc=generated_at_utc,
        previous_payload=previous_payload,
        suite_builders=suite_builders,
    )
    return payload


def _history_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "timestamp_utc": payload.get("timestamp_utc"),
        "schema_version": payload.get("schema_version"),
        "program_id": payload.get("program_id"),
        "run_id": payload.get("run_id"),
        "status": payload.get("status"),
        "ok": payload.get("ok"),
        "control_grade": payload.get("control_grade"),
        "control_score": payload.get("control_score"),
        "admission_eligible": payload.get("admission_eligible"),
        "candidate_binding": payload.get("candidate_binding"),
        "suite_results": payload.get("suite_results"),
        "comparison": payload.get("comparison"),
        "failed_checks": payload.get("failed_checks"),
        "behavior_change_contract": payload.get("behavior_change_contract"),
        "resource_contract": payload.get("resource_contract"),
        "evidence_classification": payload.get("evidence_classification"),
        "receipts": payload.get("receipts"),
        "program_payload_sha256": _canonical_sha256(payload),
    }


def execute_and_publish(
    *,
    project_root: Path = PROJECT_ROOT,
    policy_path: str | Path = DEFAULT_POLICY_PATH,
    requested_suites: Sequence[str] | None = None,
    generated_at_utc: str | None = None,
    out_path: str | Path = DEFAULT_OUT_PATH,
    suite_builders: Mapping[str, SuiteBuilder] | None = None,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    resolved_out = _resolve(root, out_path)
    previous = _load_json(resolved_out)
    payload, suite_payloads = _execute_program(
        project_root=root,
        policy_path=policy_path,
        requested_suites=requested_suites,
        generated_at_utc=generated_at_utc,
        previous_payload=previous,
        suite_builders=suite_builders,
    )
    program_policy = _load_json(_resolve(root, policy_path))
    specs = {
        str(row.get("suite_id") or ""): row
        for row in _rows(program_policy.get("suite_order"))
    }
    publication: dict[str, bool] = {}
    for suite_id, suite_payload in suite_payloads.items():
        artifact_path = _resolve(
            root, str(specs.get(suite_id, {}).get("latest_artifact") or "")
        )
        publisher = SUITE_PUBLISHERS.get(suite_id)
        publication[suite_id] = bool(
            publisher
            and publisher(
                project_root=root,
                payload=suite_payload,
                out_path=artifact_path,
                source=f"{suite_id}_drill_program_delegation",
            )
        )
    history_contract = _mapping(program_policy.get("history_contract"))
    history_dir = _resolve(
        root,
        str(
            history_contract.get("directory")
            or "governance/research/trading_behavior_drill_runs"
        ),
    )
    history_path = history_dir / f"{payload.get('run_id')}.json"
    history = _history_record(payload)
    history_bytes = len(
        json.dumps(history, ensure_ascii=True, sort_keys=True).encode("utf-8")
    )
    maximum_history_bytes = max(
        _int(
            _mapping(program_policy.get("execution_contract")).get(
                "maximum_history_record_bytes"
            ),
            262144,
        ),
        1024,
    )
    payload["checks"]["suite_artifacts_published"] = bool(publication) and all(
        publication.values()
    )
    payload["checks"]["history_record_size_bounded"] = (
        history_bytes <= maximum_history_bytes
    )
    history_written = False
    if history_bytes <= maximum_history_bytes:
        history_written = safe_write_json_atomic(
            str(history_path),
            history,
            project_root=str(root),
            source="trading_behavior_drill_program_history",
        )
    payload["checks"]["history_record_published"] = bool(history_written)
    payload["artifact_publication"] = {
        "suite_artifacts": publication,
        "history_path": str(history_path),
        "history_written": bool(history_written),
        "history_record_bytes": history_bytes,
        "maximum_history_record_bytes": maximum_history_bytes,
        "detailed_suite_artifacts_duplicated_in_history": False,
    }
    _refresh_program_status(payload)
    payload["lifecycle"]["state"] = (
        "admission_eligible" if payload.get("admission_eligible") else "rejected"
    )
    payload["receipts"]["program_payload_sha256"] = _canonical_sha256(payload)
    latest_written = safe_write_json_atomic(
        str(resolved_out),
        payload,
        project_root=str(root),
        source="trading_behavior_drill_program",
    )
    if not latest_written:
        payload["ok"] = False
        payload["status"] = "degraded"
        payload["admission_eligible"] = False
        payload.setdefault("failed_checks", []).append("program_artifact_write_failed")
    return payload


@contextmanager
def _program_lock(path: Path, timeout_seconds: float) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    deadline = time.monotonic() + max(timeout_seconds, 0.0)
    acquired = False
    try:
        while not acquired:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"drill_program_lock_timeout:{path}")
                time.sleep(0.1)
        handle.seek(0)
        handle.truncate(0)
        handle.write(
            json.dumps(
                {
                    "pid": __import__("os").getpid(),
                    "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                ensure_ascii=True,
            )
        )
        handle.flush()
        yield
    finally:
        if acquired:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the bounded candidate-bound trading behavior drill program and "
            "publish a single admission decision."
        )
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--policy", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--suite", action="append", default=[])
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--lock-file", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument("--lock-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    root = Path(args.project_root).expanduser().resolve()
    lock_path = _resolve(root, args.lock_file)
    try:
        with _program_lock(lock_path, float(args.lock_timeout_seconds)):
            payload = execute_and_publish(
                project_root=root,
                policy_path=args.policy,
                requested_suites=args.suite or None,
                out_path=args.out_file,
            )
    except TimeoutError as exc:
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "schema_version": 1,
            "status": "degraded",
            "ok": False,
            "admission_eligible": False,
            "failed_checks": [str(exc)],
        }
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    else:
        suites = _rows(payload.get("suite_results"))
        print(
            "trading_behavior_drill_program "
            f"status={payload.get('status')} grade={payload.get('control_grade', 'F')} "
            f"suites={sum(bool(row.get('ok', False)) for row in suites)}/{len(suites)} "
            f"comparison={_mapping(payload.get('comparison')).get('status', '')} "
            f"admission_eligible={int(bool(payload.get('admission_eligible', False)))}"
        )
    return 0 if payload.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
