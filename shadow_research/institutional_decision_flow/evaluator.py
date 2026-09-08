"""Compatibility facade for the canonical production-candidate decision flow."""

from core.institutional_decision_flow import (  # noqa: F401
    DIRECTIONAL_ACTIONS,
    POLICY_PATH,
    apply_decision_flow_control,
    apply_paper_decision_flow_control,
    build_candidate_bound_quantitative_evidence,
    build_decision_operator_summary,
    build_report,
    evaluate_decision,
    evaluate_execution_policy_guard,
    load_policy,
    resolve_sleeve_policy,
)

__all__ = [
    "DIRECTIONAL_ACTIONS",
    "POLICY_PATH",
    "apply_decision_flow_control",
    "apply_paper_decision_flow_control",
    "build_candidate_bound_quantitative_evidence",
    "build_decision_operator_summary",
    "build_report",
    "evaluate_decision",
    "evaluate_execution_policy_guard",
    "load_policy",
    "resolve_sleeve_policy",
]
