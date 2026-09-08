"""Shared operating-contract vocabulary for control-plane artifacts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

REQUIRED_OPERATING_CONTRACT_FIELDS = (
    "contract_id",
    "owner",
    "domain",
    "status",
    "why",
    "safe_authority",
    "blocked_authority",
    "evidence_missing",
    "release_conditions",
    "next_commands",
    "definition_gaps",
    "operator_review_required",
    "paper_execution_authority",
    "live_execution_authority",
    "promotion_authority",
    "profitability_guaranteed",
)

ESSENTIAL_OPERATING_CONTRACT_FIELDS = (
    "contract_id",
    "owner",
    "domain",
    "status",
    "why",
    "safe_authority",
    "blocked_authority",
    "release_conditions",
    "next_commands",
)


def _ordered_unique(items: Iterable[Any] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in items or []:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _command_rows(commands: Iterable[Any] | None) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw in commands or []:
        if isinstance(raw, str):
            parts = [raw]
        elif isinstance(raw, Iterable):
            parts = [str(part).strip() for part in raw if str(part).strip()]
        else:
            parts = []
        if parts and parts not in rows:
            rows.append(parts)
    return rows


def _present(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, Iterable):
        return bool(list(value))
    return True


def build_operating_contract(
    *,
    contract_id: str,
    owner: str,
    domain: str,
    status: str,
    why: str,
    safe_authority: Iterable[Any],
    blocked_authority: Iterable[Any],
    evidence_missing: Iterable[Any] | None = None,
    release_conditions: Iterable[Any] | None = None,
    next_commands: Iterable[Any] | None = None,
    definition_gaps: Iterable[Any] | None = None,
    measurement: Mapping[str, Any] | None = None,
    hardening: Mapping[str, Any] | None = None,
    notes: Iterable[Any] | None = None,
    operator_review_required: bool = True,
    paper_execution_authority: bool = False,
    live_execution_authority: bool = False,
    promotion_authority: bool = False,
    allocation_authority: bool = False,
    registry_mutation_authority: bool = False,
    source_code_mutation_authority: bool = False,
    profitability_guaranteed: bool = False,
) -> dict[str, Any]:
    """Build a stable contract block consumed by dashboards and infrabots."""
    contract: dict[str, Any] = {
        "contract_id": str(contract_id or ""),
        "owner": str(owner or ""),
        "domain": str(domain or ""),
        "status": str(status or ""),
        "why": str(why or ""),
        "safe_authority": _ordered_unique(safe_authority),
        "blocked_authority": _ordered_unique(blocked_authority),
        "evidence_missing": _ordered_unique(evidence_missing),
        "release_conditions": _ordered_unique(release_conditions),
        "next_commands": _command_rows(next_commands),
        "definition_gaps": _ordered_unique(definition_gaps),
        "operator_review_required": bool(operator_review_required),
        "paper_execution_authority": bool(paper_execution_authority),
        "live_execution_authority": bool(live_execution_authority),
        "promotion_authority": bool(promotion_authority),
        "allocation_authority": bool(allocation_authority),
        "registry_mutation_authority": bool(registry_mutation_authority),
        "source_code_mutation_authority": bool(source_code_mutation_authority),
        "profitability_guaranteed": bool(profitability_guaranteed),
        "measurement": dict(measurement or {}),
        "hardening": dict(hardening or {}),
        "notes": _ordered_unique(notes),
    }
    missing = [
        field
        for field in ESSENTIAL_OPERATING_CONTRACT_FIELDS
        if not _present(contract.get(field))
    ]
    contract["required_fields"] = list(REQUIRED_OPERATING_CONTRACT_FIELDS)
    contract["missing_required_fields"] = missing
    contract["complete"] = not missing
    return contract
