from __future__ import annotations

import hashlib
import json
from collections import Counter
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = PROJECT_ROOT / "config" / "sleeve_model_boundary_v1.json"
_PROJECTION_PLAN_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
_MAX_PROJECTION_PLANS = 512


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _normalized(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _ordered_unique(values: Sequence[Any]) -> list[str]:
    return list(
        dict.fromkeys(_normalized(value) for value in values if _normalized(value))
    )


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@lru_cache(maxsize=8)
def _load_policy_cached(path_text: str, mtime_ns: int) -> dict[str, Any]:
    del mtime_ns
    payload = json.loads(Path(path_text).read_text(encoding="utf-8"))
    validate_policy(payload)
    return payload


def load_policy(path: Path | None = None) -> dict[str, Any]:
    target = (path or POLICY_PATH).resolve()
    return deepcopy(_load_policy_cached(str(target), target.stat().st_mtime_ns))


def validate_policy(policy: Mapping[str, Any]) -> None:
    if int(policy.get("schema_version", 0) or 0) != 1:
        raise ValueError("sleeve model boundary schema version must be 1")
    if str(policy.get("policy_id") or "") != "sleeve_model_boundary_v1":
        raise ValueError("sleeve model boundary policy id is invalid")
    projection = _mapping(policy.get("feature_projection"))
    groups = _mapping(projection.get("feature_groups"))
    families = _mapping(projection.get("family_allowed_groups"))
    if not groups or not families:
        raise ValueError(
            "sleeve model boundary requires feature groups and family routes"
        )
    group_ids = set(groups)
    for group_id, tokens in groups.items():
        if not _ordered_unique(tokens if isinstance(tokens, list) else []):
            raise ValueError(f"feature group has no tokens: {group_id}")
    for family_id, allowed in families.items():
        unknown = (
            set(_ordered_unique(allowed if isinstance(allowed, list) else []))
            - group_ids
        )
        if unknown:
            raise ValueError(
                f"sleeve family references unknown feature groups: {family_id}:{','.join(sorted(unknown))}"
            )
    exclusive = set(_ordered_unique(projection.get("exclusive_groups") or []))
    if not exclusive.issubset(group_ids):
        raise ValueError("exclusive feature groups must be declared feature groups")
    additions = _mapping(projection.get("profile_group_additions"))
    for profile, allowed in additions.items():
        unknown = (
            set(_ordered_unique(allowed if isinstance(allowed, list) else []))
            - group_ids
        )
        if unknown:
            raise ValueError(
                f"profile references unknown feature groups: {profile}:{','.join(sorted(unknown))}"
            )
    bot_scope = _mapping(policy.get("bot_scope"))
    if str(bot_scope.get("unscoped_policy") or "") not in {
        "allow_with_audit",
        "block",
    }:
        raise ValueError("invalid unscoped bot policy")
    authority = _mapping(policy.get("authority"))
    forbidden = (
        "can_create_intent",
        "can_reverse_intent",
        "can_increase_quantity",
        "can_change_risk_limits",
        "can_submit_paper_order",
        "can_submit_live_order",
        "can_grant_promotion",
    )
    if any(bool(authority.get(key, False)) for key in forbidden):
        raise ValueError("sleeve model boundary may not grant trading authority")


def _feature_groups_for_name(
    name: str,
    group_tokens: Mapping[str, Sequence[Any]],
) -> tuple[str, ...]:
    normalized = _normalized(name)
    matched = [
        str(group_id)
        for group_id, raw_tokens in group_tokens.items()
        if any(
            _normalized(token) in normalized
            for token in raw_tokens
            if _normalized(token)
        )
    ]
    return tuple(sorted(set(matched)))


def _build_projection_plan(
    feature_names: tuple[str, ...],
    *,
    profile: str,
    family_id: str,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    projection = _mapping(policy.get("feature_projection"))
    groups = _mapping(projection.get("feature_groups"))
    always_tokens = _ordered_unique(projection.get("always_include_tokens") or [])
    allowed_groups = set(
        _ordered_unique(
            _mapping(projection.get("family_allowed_groups")).get(family_id) or []
        )
    )
    allowed_groups.update(
        _ordered_unique(
            _mapping(projection.get("profile_group_additions")).get(profile) or []
        )
    )
    exclusive_groups = set(_ordered_unique(projection.get("exclusive_groups") or []))
    preserve_unclassified = bool(projection.get("preserve_unclassified", True))
    included: list[str] = []
    excluded: list[str] = []
    unclassified: list[str] = []
    group_counts: Counter[str] = Counter()
    exclusion_counts: Counter[str] = Counter()
    for raw_name in feature_names:
        name = str(raw_name)
        normalized = _normalized(name)
        matched = _feature_groups_for_name(normalized, groups)
        group_counts.update(matched)
        always_included = any(token in normalized for token in always_tokens)
        exclusive_matched = set(matched) & exclusive_groups
        if always_included:
            include = True
        elif exclusive_matched:
            include = bool(exclusive_matched & allowed_groups)
        elif not matched:
            unclassified.append(name)
            include = preserve_unclassified
        else:
            # Cross-cutting market, execution, research, and safety context remains
            # available. Only explicitly exclusive domains are sleeve-gated.
            include = True
        if include:
            included.append(name)
        else:
            excluded.append(name)
            exclusion_counts.update(matched or ("unclassified",))

    material = {
        "policy_id": str(policy.get("policy_id") or ""),
        "profile": profile,
        "decision_policy_family_id": family_id,
        "allowed_feature_groups": sorted(allowed_groups),
        "input_feature_schema_sha256": _canonical_hash(list(feature_names)),
        "projected_feature_schema_sha256": _canonical_hash(included),
        "included_feature_names": included,
        "excluded_feature_names": excluded,
    }
    return {
        **material,
        "policy_receipt_sha256": _canonical_hash(policy),
        "projection_receipt_sha256": _canonical_hash(material),
        "included_feature_count": len(included),
        "excluded_feature_count": len(excluded),
        "unclassified_feature_count": len(unclassified),
        "projection_ratio": round(len(included) / max(len(feature_names), 1), 8),
        "matched_group_counts": dict(sorted(group_counts.items())),
        "excluded_group_counts": dict(sorted(exclusion_counts.items())),
        "excluded_feature_examples": excluded[:20],
        "unclassified_feature_examples": unclassified[:20],
    }


def project_model_features(
    features: Mapping[str, Any] | None,
    *,
    profile: Any,
    family_id: Any,
    policy: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    active_policy = dict(policy) if isinstance(policy, Mapping) else load_policy()
    validate_policy(active_policy)
    source = dict(features) if isinstance(features, Mapping) else {}
    profile_name = _normalized(profile) or "default"
    family_name = _normalized(family_id) or "balanced_directional"
    projection = _mapping(active_policy.get("feature_projection"))
    if not bool(projection.get("enabled", True)):
        metadata = {
            "policy_id": str(active_policy.get("policy_id") or ""),
            "status": "disabled",
            "profile": profile_name,
            "decision_policy_family_id": family_name,
            "included_feature_count": len(source),
            "excluded_feature_count": 0,
            "projection_ratio": 1.0,
            "projection_receipt_sha256": "",
        }
        return source, metadata
    feature_names = tuple(str(key) for key in source)
    policy_receipt = _canonical_hash(active_policy)
    cache_key = (policy_receipt, profile_name, family_name, feature_names)
    plan = _PROJECTION_PLAN_CACHE.get(cache_key)
    if plan is None:
        plan = _build_projection_plan(
            feature_names,
            profile=profile_name,
            family_id=family_name,
            policy=active_policy,
        )
        if len(_PROJECTION_PLAN_CACHE) >= _MAX_PROJECTION_PLANS:
            _PROJECTION_PLAN_CACHE.pop(next(iter(_PROJECTION_PLAN_CACHE)), None)
        _PROJECTION_PLAN_CACHE[cache_key] = plan
    projected = {
        key: source[key]
        for key in plan.get("included_feature_names", [])
        if key in source
    }
    metadata = {
        key: deepcopy(value)
        for key, value in plan.items()
        if key not in {"included_feature_names", "excluded_feature_names"}
    }
    metadata.update(
        {
            "status": "ready",
            "operating_mode": str(active_policy.get("operating_mode") or ""),
            "full_observability_preserved": bool(
                _mapping(active_policy.get("authority")).get(
                    "preserves_full_observability_snapshot", False
                )
            ),
            "authority": deepcopy(_mapping(active_policy.get("authority"))),
        }
    )
    return projected, metadata


def _resolve_family(
    profile: str,
    decision_policy: Mapping[str, Any],
) -> tuple[str, str]:
    normalized = _normalized(profile)
    exact = {
        _normalized(key): _normalized(value)
        for key, value in _mapping(decision_policy.get("profile_policy_map")).items()
    }
    if normalized in exact:
        return exact[normalized], "exact_profile"
    for index, raw_rule in enumerate(decision_policy.get("profile_policy_rules") or []):
        rule = _mapping(raw_rule)
        tokens = _ordered_unique(rule.get("profile_tokens_any") or [])
        if any(token in normalized for token in tokens):
            return _normalized(rule.get("policy_family_id")), f"profile_rule:{index}"
    return "balanced_directional", "default_fallback"


def evaluate_bot_scope(
    bot: Mapping[str, Any],
    *,
    runtime_profile: Any,
    runtime_family_id: Any,
    decision_policy: Mapping[str, Any],
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    active_policy = dict(policy) if isinstance(policy, Mapping) else load_policy()
    validate_policy(active_policy)
    scope_policy = _mapping(active_policy.get("bot_scope"))
    profile_name = _normalized(runtime_profile) or "default"
    family_name = _normalized(runtime_family_id) or "balanced_directional"
    role = _normalized(bot.get("bot_role")) or "signal_sub_bot"
    bot_id = str(bot.get("bot_id") or "").strip()
    claims = _ordered_unique(
        [
            bot.get("sleeve_profile"),
            bot.get("paper_sleeve_id"),
        ]
    )
    paper_sub_sleeve_id = _normalized(bot.get("paper_sub_sleeve_id"))
    always_roles = set(_ordered_unique(scope_policy.get("always_allowed_roles") or []))
    role_allowlist = {
        _normalized(key): set(_ordered_unique(value if isinstance(value, list) else []))
        for key, value in _mapping(scope_policy.get("role_family_allowlist")).items()
    }

    reasons: list[str] = []
    claim_families: dict[str, str] = {}
    if not bool(scope_policy.get("enabled", True)):
        allowed = True
        status = "disabled"
        reasons.append("bot_scope_policy_disabled")
    elif role in always_roles:
        allowed = True
        status = "allowed_infrastructure_role"
        reasons.append("always_allowed_role")
    elif role in role_allowlist and family_name not in role_allowlist[role]:
        allowed = False
        status = "blocked_role_family_mismatch"
        reasons.append(f"role_not_allowed_for_family:{role}:{family_name}")
    elif not claims:
        allowed = (
            str(scope_policy.get("unscoped_policy") or "allow_with_audit")
            == "allow_with_audit"
        )
        status = "allowed_legacy_unscoped" if allowed else "blocked_unscoped"
        reasons.append("legacy_bot_scope_missing")
    else:
        for claim in claims:
            claim_family, match_source = _resolve_family(claim, decision_policy)
            claim_families[claim] = claim_family
            reasons.append(f"claim:{claim}:{claim_family}:{match_source}")
        exact_match = profile_name in claims
        family_match = family_name in set(claim_families.values())
        allowed = bool(
            (scope_policy.get("exact_profile_allowed", True) and exact_match)
            or (scope_policy.get("same_family_allowed", True) and family_match)
        )
        status = "allowed_explicit_scope" if allowed else "blocked_cross_sleeve_scope"
        if not allowed:
            reasons.append("explicit_bot_scope_does_not_match_runtime_sleeve")

    material = {
        "policy_id": str(active_policy.get("policy_id") or ""),
        "bot_id": bot_id,
        "bot_role": role,
        "runtime_profile": profile_name,
        "runtime_family_id": family_name,
        "scope_claims": claims,
        "paper_sub_sleeve_id": paper_sub_sleeve_id,
        "claim_families": claim_families,
        "allowed": bool(allowed),
        "status": status,
        "reasons": reasons,
    }
    return {
        **material,
        "scope_receipt_sha256": _canonical_hash(material),
        "authority": "eligibility_filter_only_no_action_sizing_or_execution_authority",
    }
