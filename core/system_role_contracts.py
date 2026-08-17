"""Executable responsibility, authority, and ownership contracts for the platform."""

from __future__ import annotations

import fcntl
import fnmatch
import hashlib
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping


DEFAULT_CONFIG_NAME = "system_role_contracts_v1.json"
DEFAULT_OWNERSHIP_NAME = "control_surface_ownership_v1.json"
EXECUTION_AUTHORITY_KEYS = {
    "observe",
    "recommend",
    "paper_trade",
    "risk_veto",
    "live_submit",
    "automatic_promotion",
}
EXECUTION_ACTION_FLAGS = {
    "paper_submit": "paper_trade",
    "live_submit": "live_submit",
    "veto_trade": "risk_veto",
}
REQUIRED_TAXONOMIES = {
    "execution_modes",
    "freshness_classes",
    "failure_classes",
    "resource_profiles",
    "lifecycle_states",
    "configuration_precedence",
}
REQUIRED_ROLE_FIELDS = {
    "role_id",
    "display_name",
    "tier",
    "purpose",
    "allowed_inputs",
    "owned_outputs",
    "write_authority",
    "execution_authority",
    "triggers",
    "freshness_slo",
    "failure_behavior",
    "resource_budget",
    "escalation_owner",
    "evidence_outputs",
    "forbidden_actions",
    "allowed_actions",
}
REQUIRED_HIERARCHY_RULES = {
    "collectors_never_decide_or_trade",
    "strategies_never_submit_orders",
    "masters_never_bypass_risk",
    "risk_may_veto_but_never_originate_signals",
    "execution_is_the_only_order_writer",
    "truth_never_rewrites_outcomes",
    "infrastructure_never_changes_trade_logic",
    "dashboards_never_invent_canonical_truth",
    "grand_master_cannot_grant_itself_authority",
}
REQUIRED_NON_BYPASSABLE_ROLES = {
    "risk_governance",
    "live_execution_gateway",
    "truth_reconciliation",
}
REQUIRED_CONFIGURATION_PRECEDENCE = (
    "safety_flags",
    "candidate_bound_runtime_state",
    "operator_override",
    "runtime_profile",
    "policy_json",
    "environment_default",
    "code_default",
)
SENSITIVE_ACTIONS = {
    "restart_process",
    "paper_submit",
    "live_submit",
    "promote_candidate",
    "write_candidate_state",
}


class RoleAuthorityError(RuntimeError):
    """Raised when a component attempts an action outside its declared role."""


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _strings(value: Any) -> list[str]:
    return [str(item).strip() for item in _list(value) if str(item).strip()]


def _ordered_unique(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw in values:
        value = str(raw or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _role_map(contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("role_id") or "").strip(): row
        for row in _list(contract.get("roles"))
        if isinstance(row, dict) and str(row.get("role_id") or "").strip()
    }


def _component_map(contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("component_id") or "").strip(): row
        for row in _list(contract.get("components"))
        if isinstance(row, dict) and str(row.get("component_id") or "").strip()
    }


def _domain_map(contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("domain_id") or "").strip(): row
        for row in _list(contract.get("state_domains"))
        if isinstance(row, dict) and str(row.get("domain_id") or "").strip()
    }


def _duplicates(values: list[str]) -> list[str]:
    return sorted({value for value in values if value and values.count(value) > 1})


def _taxonomy_ids(rows: Any) -> set[str]:
    return {
        str(row.get("id") or "").strip()
        for row in _list(rows)
        if isinstance(row, dict) and str(row.get("id") or "").strip()
    }


def _unsafe_relative_path(raw: str) -> bool:
    path = Path(str(raw or ""))
    return bool(path.is_absolute() or ".." in path.parts)


def _escalation_routes(roles: Mapping[str, Mapping[str, Any]]) -> dict[str, list[str]]:
    routes: dict[str, list[str]] = {}
    for role_id in sorted(roles):
        route = [role_id]
        visited = {role_id}
        cursor = role_id
        while cursor != "operator":
            owner = str(_dict(roles.get(cursor)).get("escalation_owner") or "").strip()
            if not owner:
                break
            route.append(owner)
            if owner == "operator" or owner in visited:
                break
            visited.add(owner)
            cursor = owner
        routes[role_id] = route
    return routes


def _resource_patterns_overlap(left: str, right: str, project_root: Path | None) -> bool:
    if left == right or fnmatch.fnmatch(left, right) or fnmatch.fnmatch(right, left):
        return True
    if project_root is None:
        return False
    try:
        left_matches = {str(path.relative_to(project_root)) for path in project_root.glob(left) if path.is_file()}
        right_matches = {str(path.relative_to(project_root)) for path in project_root.glob(right) if path.is_file()}
    except (OSError, ValueError, NotImplementedError):
        return False
    return bool(left_matches & right_matches)


def _source_matches_component(owner_source: str, component: Mapping[str, Any]) -> bool:
    source = str(owner_source or "").strip()
    if not source:
        return False
    exact = set(_strings(component.get("source_paths")))
    patterns = _strings(component.get("source_patterns"))
    return source in exact or any(fnmatch.fnmatch(source, pattern) for pattern in patterns)


def validate_contract(
    contract: Mapping[str, Any],
    *,
    project_root: Path | None = None,
    ownership_registry: Mapping[str, Any] | None = None,
    bot_registry: Mapping[str, Any] | None = None,
    check_sources: bool = True,
) -> dict[str, Any]:
    """Validate the whole contract and return machine-readable conflicts."""

    blockers: list[str] = []
    warnings: list[str] = []
    required_fields = _strings(contract.get("required_role_fields"))
    roles_raw = [row for row in _list(contract.get("roles")) if isinstance(row, dict)]
    components_raw = [row for row in _list(contract.get("components")) if isinstance(row, dict)]
    domains_raw = [row for row in _list(contract.get("state_domains")) if isinstance(row, dict)]
    roles = _role_map(contract)
    components = _component_map(contract)
    domains = _domain_map(contract)
    taxonomies = _dict(contract.get("taxonomies"))
    hierarchy = _dict(contract.get("hierarchy"))

    if int(contract.get("schema_version") or 0) != 1:
        blockers.append("schema_version_invalid")
    if str(contract.get("operating_mode") or "") != "enforced_responsibility_contracts":
        blockers.append("operating_mode_not_enforced")
    if not str(contract.get("policy_id") or "").strip():
        blockers.append("policy_id_missing")
    if not required_fields:
        blockers.append("required_role_fields_missing")
    for field in sorted(REQUIRED_ROLE_FIELDS - set(required_fields)):
        blockers.append(f"required_role_field_policy_missing:{field}")
    missing_taxonomies = sorted(REQUIRED_TAXONOMIES - set(taxonomies))
    blockers.extend(f"taxonomy_missing:{name}" for name in missing_taxonomies)
    freshness_ids = _taxonomy_ids(taxonomies.get("freshness_classes"))
    failure_ids = _taxonomy_ids(taxonomies.get("failure_classes"))
    resource_profile_ids = _taxonomy_ids(taxonomies.get("resource_profiles"))
    action_classes = _dict(taxonomies.get("action_classes"))
    classified_actions = [action for rows in action_classes.values() for action in _strings(rows)]
    blockers.extend(f"action_taxonomy_duplicate:{item}" for item in _duplicates(classified_actions))
    if tuple(_strings(taxonomies.get("configuration_precedence"))) != REQUIRED_CONFIGURATION_PRECEDENCE:
        blockers.append("configuration_precedence_invalid")

    role_ids = [str(row.get("role_id") or "").strip() for row in roles_raw]
    component_ids = [str(row.get("component_id") or "").strip() for row in components_raw]
    domain_ids = [str(row.get("domain_id") or "").strip() for row in domains_raw]
    blockers.extend(f"duplicate_role_id:{item}" for item in _duplicates(role_ids))
    blockers.extend(f"duplicate_component_id:{item}" for item in _duplicates(component_ids))
    blockers.extend(f"duplicate_state_domain:{item}" for item in _duplicates(domain_ids))
    if not roles:
        blockers.append("roles_missing")
    if not components:
        blockers.append("components_missing")
    if not domains:
        blockers.append("state_domains_missing")

    list_fields = {
        "allowed_inputs",
        "owned_outputs",
        "write_authority",
        "triggers",
        "evidence_outputs",
        "forbidden_actions",
        "allowed_actions",
    }
    dict_fields = {"execution_authority", "freshness_slo", "failure_behavior", "resource_budget"}
    for role_id, role in roles.items():
        for field in required_fields:
            if field not in role:
                blockers.append(f"role:{role_id}:field_missing:{field}")
                continue
            if field in list_fields and not isinstance(role.get(field), list):
                blockers.append(f"role:{role_id}:field_not_list:{field}")
            elif field in dict_fields and not isinstance(role.get(field), dict):
                blockers.append(f"role:{role_id}:field_not_object:{field}")
            elif field not in list_fields | dict_fields and not str(role.get(field) or "").strip():
                blockers.append(f"role:{role_id}:field_empty:{field}")
        for field in list_fields:
            if field in role and not _strings(role.get(field)):
                blockers.append(f"role:{role_id}:field_empty:{field}")

        allowed = set(_strings(role.get("allowed_actions")))
        forbidden = set(_strings(role.get("forbidden_actions")))
        for overlap in sorted(allowed & forbidden):
            blockers.append(f"role:{role_id}:action_allowed_and_forbidden:{overlap}")
        authority = _dict(role.get("execution_authority"))
        if set(authority) != EXECUTION_AUTHORITY_KEYS or any(not isinstance(value, bool) for value in authority.values()):
            blockers.append(f"role:{role_id}:execution_authority_invalid")
        escalation = str(role.get("escalation_owner") or "").strip()
        if escalation not in roles and escalation != "operator":
            blockers.append(f"role:{role_id}:escalation_owner_unknown:{escalation or 'missing'}")
        freshness = _dict(role.get("freshness_slo"))
        freshness_class = str(freshness.get("class") or "").strip()
        if not freshness_class or float(freshness.get("max_age_seconds") or 0.0) <= 0.0:
            blockers.append(f"role:{role_id}:freshness_slo_invalid")
        elif freshness_class not in freshness_ids:
            blockers.append(f"role:{role_id}:freshness_class_unknown:{freshness_class}")
        failure = _dict(role.get("failure_behavior"))
        failure_class = str(failure.get("class") or "").strip()
        if not failure_class or not str(failure.get("default_action") or "").strip():
            blockers.append(f"role:{role_id}:failure_behavior_invalid")
        elif failure_class not in failure_ids:
            blockers.append(f"role:{role_id}:failure_class_unknown:{failure_class}")
        budget = _dict(role.get("resource_budget"))
        resource_profile = str(budget.get("profile") or "").strip()
        if not resource_profile or int(budget.get("max_parallelism") or 0) < 1:
            blockers.append(f"role:{role_id}:resource_budget_invalid")
        elif resource_profile not in resource_profile_ids:
            blockers.append(f"role:{role_id}:resource_profile_unknown:{resource_profile}")
        for action in sorted(allowed - set(classified_actions)):
            blockers.append(f"role:{role_id}:action_unclassified:{action}")

    for role_id in roles:
        visited: set[str] = set()
        cursor = role_id
        while cursor != "operator":
            if cursor in visited:
                blockers.append(f"escalation_cycle:{role_id}:{cursor}")
                break
            visited.add(cursor)
            cursor = str(_dict(roles.get(cursor)).get("escalation_owner") or "")
            if cursor not in roles and cursor != "operator":
                break

    role_plane_membership: dict[str, list[str]] = {role_id: [] for role_id in roles}
    planes = _dict(hierarchy.get("planes"))
    for plane_id, members_raw in planes.items():
        for role_id in _strings(members_raw):
            if role_id not in roles:
                blockers.append(f"hierarchy:{plane_id}:unknown_role:{role_id}")
            else:
                role_plane_membership[role_id].append(str(plane_id))
    for role_id, memberships in role_plane_membership.items():
        if len(memberships) != 1:
            blockers.append(f"hierarchy:role_plane_membership:{role_id}:{len(memberships)}")
    decision_flow = _strings(hierarchy.get("decision_flow"))
    blockers.extend(f"hierarchy:decision_flow_duplicate:{item}" for item in _duplicates(decision_flow))
    blockers.extend(f"hierarchy:decision_flow_unknown_role:{item}" for item in decision_flow if item not in roles)
    non_bypassable = set(_strings(hierarchy.get("non_bypassable_roles")))
    for role_id in sorted(REQUIRED_NON_BYPASSABLE_ROLES - non_bypassable):
        blockers.append(f"hierarchy:non_bypassable_role_missing:{role_id}")
    for role_id in sorted(non_bypassable - set(roles)):
        blockers.append(f"hierarchy:non_bypassable_role_unknown:{role_id}")
    hierarchy_rules = _dict(hierarchy.get("rules"))
    for rule in sorted(REQUIRED_HIERARCHY_RULES):
        if hierarchy_rules.get(rule) is not True:
            blockers.append(f"hierarchy:rule_not_enforced:{rule}")

    for component_id, component in components.items():
        role_id = str(component.get("role_id") or "").strip()
        role = roles.get(role_id)
        if role is None:
            blockers.append(f"component:{component_id}:unknown_role:{role_id or 'missing'}")
            continue
        sources = _strings(component.get("source_paths"))
        patterns = _strings(component.get("source_patterns"))
        if not isinstance(component.get("allowed_actions"), list):
            blockers.append(f"component:{component_id}:allowed_actions_not_list")
        if not isinstance(component.get("state_domains"), list):
            blockers.append(f"component:{component_id}:state_domains_not_list")
        if not sources and not patterns:
            blockers.append(f"component:{component_id}:source_contract_missing")
        if project_root is not None and check_sources:
            for source in sources:
                if _unsafe_relative_path(source):
                    blockers.append(f"component:{component_id}:unsafe_source_path:{source}")
                    continue
                if not (project_root / source).is_file():
                    blockers.append(f"component:{component_id}:source_missing:{source}")
            for pattern in patterns:
                if _unsafe_relative_path(pattern):
                    blockers.append(f"component:{component_id}:unsafe_source_pattern:{pattern}")
                    continue
                if not any(path.is_file() for path in project_root.glob(pattern)):
                    blockers.append(f"component:{component_id}:source_pattern_empty:{pattern}")
        role_actions = set(_strings(role.get("allowed_actions")))
        component_actions = set(_strings(component.get("allowed_actions")))
        for action in sorted(component_actions - role_actions):
            blockers.append(f"component:{component_id}:action_outside_role:{action}")
        for domain_id in _strings(component.get("state_domains")):
            if domain_id not in domains:
                blockers.append(f"component:{component_id}:unknown_state_domain:{domain_id}")
            elif str(domains[domain_id].get("writer_component_id") or "") != component_id:
                blockers.append(f"component:{component_id}:state_domain_not_owned:{domain_id}")

    resource_writers: dict[str, list[str]] = {}
    resource_claims: list[tuple[str, str, str]] = []
    for domain_id, domain in domains.items():
        writer = str(domain.get("writer_component_id") or "").strip()
        component = components.get(writer)
        resources = _strings(domain.get("resource_patterns"))
        required_action = str(domain.get("required_action") or "write_state").strip()
        if component is None:
            blockers.append(f"state_domain:{domain_id}:writer_component_unknown:{writer or 'missing'}")
        else:
            if domain_id not in _strings(component.get("state_domains")):
                blockers.append(f"state_domain:{domain_id}:writer_missing_domain_binding:{writer}")
            if required_action not in _strings(component.get("allowed_actions")):
                blockers.append(f"state_domain:{domain_id}:writer_action_missing:{required_action}")
        if not resources:
            blockers.append(f"state_domain:{domain_id}:resource_patterns_missing")
        for resource in resources:
            if _unsafe_relative_path(resource):
                blockers.append(f"state_domain:{domain_id}:unsafe_resource_pattern:{resource}")
            resource_writers.setdefault(resource, []).append(writer)
            resource_claims.append((domain_id, writer, resource))
        for reader_role in _strings(domain.get("reader_roles")):
            if reader_role not in roles and reader_role != "operator":
                blockers.append(f"state_domain:{domain_id}:reader_role_unknown:{reader_role}")
        claimants = sorted(
            component_id
            for component_id, candidate in components.items()
            if domain_id in _strings(candidate.get("state_domains"))
        )
        if claimants != ([writer] if writer else []):
            blockers.append(f"state_domain:{domain_id}:writer_claimants_invalid:{','.join(claimants) or 'none'}")
    state_domain_conflicts = {
        resource: sorted(set(writers))
        for resource, writers in resource_writers.items()
        if len(set(writers)) != 1 or len(writers) != 1
    }
    for index, (left_domain, left_writer, left_pattern) in enumerate(resource_claims):
        for right_domain, right_writer, right_pattern in resource_claims[index + 1 :]:
            if left_domain == right_domain:
                continue
            if not _resource_patterns_overlap(left_pattern, right_pattern, project_root):
                continue
            conflict_key = f"{left_pattern}<->{right_pattern}"
            state_domain_conflicts[conflict_key] = sorted({left_writer, right_writer})
    blockers.extend(f"state_resource_writer_conflict:{resource}" for resource in sorted(state_domain_conflicts))

    exclusive_action_owners = _dict(contract.get("exclusive_action_owners"))
    for action, component_id_raw in exclusive_action_owners.items():
        component_id = str(component_id_raw or "").strip()
        component = components.get(component_id)
        if component is None:
            blockers.append(f"exclusive_action:{action}:owner_unknown:{component_id or 'missing'}")
            continue
        role = roles.get(str(component.get("role_id") or ""), {})
        if action not in _strings(component.get("allowed_actions")) or action not in _strings(role.get("allowed_actions")):
            blockers.append(f"exclusive_action:{action}:owner_not_authorized:{component_id}")
        claimants = sorted(
            candidate_id
            for candidate_id, candidate in components.items()
            if action in _strings(candidate.get("allowed_actions"))
        )
        if claimants != [component_id]:
            blockers.append(f"exclusive_action:{action}:component_claimants_invalid:{','.join(claimants) or 'none'}")

    action_leases = _dict(contract.get("action_leases"))
    for action in sorted(SENSITIVE_ACTIONS):
        lease = _dict(action_leases.get(action))
        if lease.get("required") is not True:
            blockers.append(f"action_lease:{action}:not_required")
        lease_path = str(lease.get("path") or "").strip()
        if not lease_path or _unsafe_relative_path(lease_path):
            blockers.append(f"action_lease:{action}:path_invalid")

    for action, authority_flag in EXECUTION_ACTION_FLAGS.items():
        authorized_roles = sorted(
            role_id
            for role_id, role in roles.items()
            if bool(_dict(role.get("execution_authority")).get(authority_flag, False))
        )
        if len(authorized_roles) != 1:
            blockers.append(f"execution_authority:{action}:role_count:{len(authorized_roles)}")
        owner = str(exclusive_action_owners.get(action) or "").strip()
        owner_role = str(_dict(components.get(owner)).get("role_id") or "")
        if authorized_roles and owner_role != authorized_roles[0]:
            blockers.append(f"execution_authority:{action}:exclusive_owner_role_mismatch")

    if ownership_registry is not None:
        controls = [row for row in _list(ownership_registry.get("controls")) if isinstance(row, dict)]
        control_ids = {str(row.get("control_id") or "").strip() for row in controls if str(row.get("control_id") or "").strip()}
        bindings = {str(key): str(value or "").strip() for key, value in _dict(contract.get("control_surface_bindings")).items()}
        missing_bindings = sorted(control_ids - set(bindings))
        extra_bindings = sorted(set(bindings) - control_ids)
        blockers.extend(f"control_surface_binding_missing:{item}" for item in missing_bindings)
        blockers.extend(f"control_surface_binding_unknown:{item}" for item in extra_bindings)
        for row in controls:
            control_id = str(row.get("control_id") or "").strip()
            component_id = bindings.get(control_id, "")
            component = components.get(component_id)
            if component is None:
                if control_id in bindings:
                    blockers.append(f"control_surface:{control_id}:component_unknown:{component_id or 'missing'}")
                continue
            owner_source = str(row.get("owner_source") or "").strip()
            if not _source_matches_component(owner_source, component):
                blockers.append(f"control_surface:{control_id}:source_not_owned_by_component:{component_id}")

    registry_coverage: dict[str, Any] = {
        "registry_role_count": 0,
        "mapped_registry_role_count": 0,
        "unmapped_registry_roles": [],
        "coverage_ratio": 1.0,
    }
    if bot_registry is not None:
        registry_rows = [row for row in _list(bot_registry.get("sub_bots")) if isinstance(row, dict)]
        registry_roles = sorted({str(row.get("bot_role") or "").strip() for row in registry_rows if str(row.get("bot_role") or "").strip()})
        bindings = {str(key): str(value or "").strip() for key, value in _dict(contract.get("registry_role_bindings")).items()}
        unmapped = sorted(role for role in registry_roles if role not in bindings or bindings.get(role) not in roles)
        registry_coverage = {
            "registry_role_count": len(registry_roles),
            "mapped_registry_role_count": len(registry_roles) - len(unmapped),
            "unmapped_registry_roles": unmapped,
            "coverage_ratio": round((len(registry_roles) - len(unmapped)) / max(len(registry_roles), 1), 6),
        }
        blockers.extend(f"registry_role_unmapped:{role}" for role in unmapped)

    safety = _dict(contract.get("safety_contract"))
    for key in (
        "role_self_escalation",
        "automatic_live_authority_grant",
        "automatic_promotion_authority",
        "truth_rewrite_authority",
        "profitability_guaranteed",
    ):
        if safety.get(key) is not False:
            blockers.append(f"safety_contract:{key}:must_be_false")
    for key in (
        "single_writer_state_domains",
        "fail_closed_unknown_actions",
        "explicit_execution_authority",
        "sensitive_action_leases",
    ):
        if safety.get(key) is not True:
            blockers.append(f"safety_contract:{key}:must_be_true")

    return {
        "ok": not blockers,
        "blockers": _ordered_unique(blockers),
        "warnings": _ordered_unique(warnings),
        "state_domain_conflicts": state_domain_conflicts,
        "registry_coverage": registry_coverage,
        "role_plane_membership": role_plane_membership,
    }


def evaluate_component_action(
    project_root: str | Path,
    *,
    component_id: str,
    action: str,
    state_domain: str = "",
    resource_path: str = "",
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Fail closed unless the component, role, action, and state owner all agree."""

    root = Path(project_root).resolve()
    path = Path(config_path) if config_path is not None else root / "config" / DEFAULT_CONFIG_NAME
    if not path.is_absolute():
        path = root / path
    contract = _load_json(path)
    validation = validate_contract(contract, check_sources=False)
    roles = _role_map(contract)
    components = _component_map(contract)
    domains = _domain_map(contract)
    component = components.get(str(component_id or "").strip())
    role = roles.get(str(_dict(component).get("role_id") or ""))
    blockers: list[str] = []
    action_name = str(action or "").strip()
    domain_name = str(state_domain or "").strip()

    if not validation.get("ok", False):
        blockers.append("system_role_contract_invalid")
    if component is None:
        blockers.append("component_unknown")
    if role is None:
        blockers.append("component_role_unknown")
    if not action_name:
        blockers.append("action_missing")
    if component is not None and action_name not in _strings(component.get("allowed_actions")):
        blockers.append("component_action_not_allowed")
    if role is not None and action_name not in _strings(role.get("allowed_actions")):
        blockers.append("role_action_not_allowed")
    if role is not None and action_name in _strings(role.get("forbidden_actions")):
        blockers.append("role_action_forbidden")

    exclusive_owner = str(_dict(contract.get("exclusive_action_owners")).get(action_name) or "").strip()
    if exclusive_owner and exclusive_owner != str(component_id or "").strip():
        blockers.append("exclusive_action_owned_by_other_component")

    authority_flag = EXECUTION_ACTION_FLAGS.get(action_name)
    if authority_flag and not bool(_dict(_dict(role).get("execution_authority")).get(authority_flag, False)):
        blockers.append(f"execution_authority_missing:{authority_flag}")

    domain = domains.get(domain_name) if domain_name else None
    if domain_name and domain is None:
        blockers.append("state_domain_unknown")
    if domain is not None:
        if str(domain.get("writer_component_id") or "") != str(component_id or "").strip():
            blockers.append("state_domain_owned_by_other_component")
        required_action = str(domain.get("required_action") or "write_state")
        if required_action != action_name:
            blockers.append("state_domain_action_mismatch")
        if resource_path:
            normalized = str(resource_path).strip().lstrip("./")
            patterns = _strings(domain.get("resource_patterns"))
            if not any(fnmatch.fnmatch(normalized, pattern) for pattern in patterns):
                blockers.append("resource_outside_state_domain")

    blockers = _ordered_unique(blockers)
    return {
        "ok": not blockers,
        "component_id": str(component_id or "").strip(),
        "role_id": str(_dict(component).get("role_id") or ""),
        "action": action_name,
        "state_domain": domain_name,
        "resource_path": str(resource_path or ""),
        "blockers": blockers,
        "contract_path": str(path),
        "contract_sha256": _canonical_sha256(contract) if contract else "",
        "policy": "deny_unknown_or_ambiguous_actions_and_require_the_single_declared_state_writer",
    }


@contextmanager
def component_action_guard(
    project_root: str | Path,
    *,
    component_id: str,
    action: str,
    state_domain: str = "",
    resource_path: str = "",
    acquire_lease: bool = True,
) -> Iterator[dict[str, Any]]:
    """Authorize an action and serialize sensitive mutations with a declared lease."""

    root = Path(project_root).resolve()
    decision = evaluate_component_action(
        root,
        component_id=component_id,
        action=action,
        state_domain=state_domain,
        resource_path=resource_path,
    )
    if not decision["ok"]:
        raise RoleAuthorityError(",".join(decision["blockers"]) or "role_authority_denied")

    contract = _load_json(root / "config" / DEFAULT_CONFIG_NAME)
    lease_cfg = _dict(_dict(contract.get("action_leases")).get(action))
    lease_required = bool(lease_cfg.get("required", False))
    if not acquire_lease or not lease_required:
        yield decision
        return

    raw_path = str(lease_cfg.get("path") or "").strip()
    if not raw_path:
        raise RoleAuthorityError("required_action_lease_path_missing")
    lease_path = Path(raw_path)
    if not lease_path.is_absolute():
        lease_path = root / lease_path
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lease_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RoleAuthorityError("sensitive_action_lease_busy") from exc
        handle.seek(0)
        handle.truncate(0)
        handle.write(
            json.dumps(
                {
                    "component_id": component_id,
                    "action": action,
                    "pid": os.getpid(),
                    "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
                    "contract_sha256": decision["contract_sha256"],
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )
        handle.flush()
        os.fsync(handle.fileno())
        yield {**decision, "lease_path": str(lease_path), "lease_acquired": True}
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def build_contract_report(
    project_root: str | Path,
    *,
    config_path: str | Path | None = None,
    ownership_path: str | Path | None = None,
    registry_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = Path(config_path) if config_path is not None else root / "config" / DEFAULT_CONFIG_NAME
    ownership_file = (
        Path(ownership_path)
        if ownership_path is not None
        else root / "config" / DEFAULT_OWNERSHIP_NAME
    )
    registry_file = Path(registry_path) if registry_path is not None else root / "master_bot_registry.json"
    if not config_file.is_absolute():
        config_file = root / config_file
    if not ownership_file.is_absolute():
        ownership_file = root / ownership_file
    if not registry_file.is_absolute():
        registry_file = root / registry_file

    contract = _load_json(config_file)
    ownership = _load_json(ownership_file)
    registry = _load_json(registry_file)
    validation = validate_contract(
        contract,
        project_root=root,
        ownership_registry=ownership,
        bot_registry=registry,
        check_sources=True,
    )
    roles = _role_map(contract)
    components = _component_map(contract)
    domains = _domain_map(contract)
    source_paths = sorted(
        {
            source
            for component in components.values()
            for source in _strings(component.get("source_paths"))
        }
    )
    source_receipts = {
        source: _file_sha256(root / source)
        for source in source_paths
        if (root / source).is_file()
    }
    contract_sha = _canonical_sha256(contract) if contract else ""
    receipt = _canonical_sha256(
        {
            "contract_sha256": contract_sha,
            "ownership_sha256": _file_sha256(ownership_file),
            "registry_role_bindings": _dict(contract.get("registry_role_bindings")),
            "source_receipts": source_receipts,
        }
    )
    authority_matrix = {
        role_id: {
            "execution_authority": _dict(role.get("execution_authority")),
            "allowed_actions": _strings(role.get("allowed_actions")),
            "forbidden_actions": _strings(role.get("forbidden_actions")),
            "escalation_owner": str(role.get("escalation_owner") or ""),
        }
        for role_id, role in sorted(roles.items())
    }
    escalation_routes = _escalation_routes(roles)
    action_classes = _dict(_dict(contract.get("taxonomies")).get("action_classes"))
    classified_actions = sorted({action for rows in action_classes.values() for action in _strings(rows)})
    state_rows = [
        {
            "domain_id": domain_id,
            "writer_component_id": str(domain.get("writer_component_id") or ""),
            "required_action": str(domain.get("required_action") or ""),
            "resource_patterns": _strings(domain.get("resource_patterns")),
            "reader_roles": _strings(domain.get("reader_roles")),
        }
        for domain_id, domain in sorted(domains.items())
    ]
    blockers = _strings(validation.get("blockers"))
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": not blockers,
        "overall_status": "ready" if not blockers else "blocked",
        "grade": "A+" if not blockers else "F",
        "policy_id": str(contract.get("policy_id") or ""),
        "operating_mode": str(contract.get("operating_mode") or ""),
        "summary": {
            "role_count": len(roles),
            "component_count": len(components),
            "state_domain_count": len(domains),
            "control_surface_binding_count": len(_dict(contract.get("control_surface_bindings"))),
            "exclusive_action_count": len(_dict(contract.get("exclusive_action_owners"))),
            "registry_role_coverage_ratio": float(_dict(validation.get("registry_coverage")).get("coverage_ratio", 0.0)),
            "authority_conflict_count": len(blockers),
            "operating_plane_count": len(_dict(_dict(contract.get("hierarchy")).get("planes"))),
            "classified_action_count": len(classified_actions),
            "action_lease_count": len(_dict(contract.get("action_leases"))),
            "escalation_route_count": len(escalation_routes),
        },
        "hierarchy": _dict(contract.get("hierarchy")),
        "taxonomies": _dict(contract.get("taxonomies")),
        "authority_matrix": authority_matrix,
        "escalation_routes": escalation_routes,
        "definition_coverage": {
            "required_role_field_count": len(REQUIRED_ROLE_FIELDS),
            "roles_with_complete_required_fields": sum(
                1 for role in roles.values() if REQUIRED_ROLE_FIELDS.issubset(set(role))
            ),
            "classified_actions": classified_actions,
            "classification_complete": not any("action_unclassified" in item for item in blockers),
            "configuration_precedence": _strings(_dict(contract.get("taxonomies")).get("configuration_precedence")),
        },
        "components": [
            {
                "component_id": component_id,
                "role_id": str(component.get("role_id") or ""),
                "allowed_actions": _strings(component.get("allowed_actions")),
                "state_domains": _strings(component.get("state_domains")),
                "source_paths": _strings(component.get("source_paths")),
                "source_patterns": _strings(component.get("source_patterns")),
            }
            for component_id, component in sorted(components.items())
        ],
        "state_domains": state_rows,
        "exclusive_action_owners": _dict(contract.get("exclusive_action_owners")),
        "control_surface_bindings": _dict(contract.get("control_surface_bindings")),
        "registry_coverage": _dict(validation.get("registry_coverage")),
        "state_domain_conflicts": _dict(validation.get("state_domain_conflicts")),
        "blockers": blockers,
        "warnings": _strings(validation.get("warnings")),
        "safety_contract": _dict(contract.get("safety_contract")),
        "runtime_enforcement": {
            "unknown_actions_fail_closed": True,
            "single_writer_checked": True,
            "execution_authority_checked": True,
            "control_surface_sources_cross_checked": True,
            "sensitive_action_leases": _dict(contract.get("action_leases")),
        },
        "source_files": {
            "contract": str(config_file),
            "control_surface_ownership": str(ownership_file),
            "bot_registry": str(registry_file),
        },
        "evidence_epoch": {
            "id": f"system-role-contract:{receipt[:16]}",
            "receipt_sha256": receipt,
            "contract_sha256": contract_sha,
            "ownership_sha256": _file_sha256(ownership_file),
            "source_receipts": source_receipts,
        },
        "recommended_actions": (
            []
            if not blockers
            else [
                "repair duplicate or missing state ownership before allowing the affected mutation",
                "restore complete registry-role and control-surface bindings",
                "keep live execution locked until the role contract returns ready",
            ]
        ),
    }
