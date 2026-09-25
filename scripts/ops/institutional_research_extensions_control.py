#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

from core.accountability import safe_write_json_atomic
from core.institutional_research_extensions import (
    REQUIRED_CONTROL_IDS,
    canonical_sha256,
    load_policy,
    structural_probe,
)

DEFAULT_CONFIG = Path("config/institutional_research_extensions_v1.json")
DEFAULT_OUT = Path(
    "governance/health/institutional_research_extensions_control_latest.json"
)
DEFAULT_MARKDOWN = Path(
    "governance/health/institutional_research_extensions_control_latest.md"
)
MAX_EVIDENCE_AGE_SECONDS = 24 * 60 * 60


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


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


def _file_sha256(path: Path) -> str:
    try:
        return canonical_sha256(
            {
                "bytes_sha256": __import__("hashlib")
                .sha256(path.read_bytes())
                .hexdigest()
            }
        )
    except OSError:
        return ""


def _candidate(project_root: Path) -> dict[str, Any]:
    path = project_root / "governance/runtime/production_candidate_state.json"
    payload = _load_json(path)
    scopes = _mapping(payload.get("scope_windows_started_utc"))
    last_change = _mapping(payload.get("last_change"))
    candidate_id = str(payload.get("candidate_id") or "")
    return {
        "candidate_id": candidate_id,
        "generation": int(payload.get("generation") or 0),
        "accepted_at_utc": str(payload.get("accepted_at_utc") or ""),
        "scope_windows_started_utc": scopes,
        "last_change": last_change,
        "bound": bool(candidate_id and scopes),
        "state_path": str(path),
        "state_receipt_sha256": _file_sha256(path),
    }


def _artifact(project_root: Path, relative: str, *, now: datetime) -> dict[str, Any]:
    path = project_root / relative
    payload = _load_json(path)
    observed = _parse_timestamp(payload.get("timestamp_utc"))
    if observed is None and path.is_file():
        observed = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    age = max((now - observed).total_seconds(), 0.0) if observed else None
    return {
        "path": str(path),
        "payload": payload,
        "present": bool(payload),
        "age_seconds": round(age, 3) if age is not None else None,
        "fresh": bool(payload and age is not None and age <= MAX_EVIDENCE_AGE_SECONDS),
    }


def _payload_candidate_id(payload: Mapping[str, Any]) -> str:
    binding = _mapping(payload.get("candidate_binding"))
    return str(payload.get("candidate_id") or binding.get("candidate_id") or "")


def _evidence_row(
    artifact: Mapping[str, Any],
    *,
    candidate_id: str,
    candidate_required: bool = True,
) -> dict[str, Any]:
    payload = _mapping(artifact.get("payload"))
    observed_candidate = _payload_candidate_id(payload)
    candidate_match = bool(
        not candidate_required
        or (candidate_id and observed_candidate and observed_candidate == candidate_id)
    )
    explicit_ready = bool(
        payload.get(
            "evidence_ready",
            payload.get("ok", payload.get("overall_status") == "ready"),
        )
    )
    ready = bool(
        artifact.get("present")
        and artifact.get("fresh")
        and candidate_match
        and explicit_ready
    )
    if not artifact.get("present"):
        status = "missing"
    elif not artifact.get("fresh"):
        status = "stale"
    elif candidate_required and not candidate_match:
        status = "candidate_mismatch"
    elif not explicit_ready:
        status = "collecting"
    else:
        status = "ready"
    return {
        "ready": ready,
        "status": status,
        "path": str(artifact.get("path") or ""),
        "age_seconds": artifact.get("age_seconds"),
        "candidate_required": candidate_required,
        "candidate_id": observed_candidate,
        "candidate_match": candidate_match,
        "synthetic_credit": False,
    }


def _material_change_evidence(candidate: Mapping[str, Any]) -> dict[str, Any]:
    last_change = _mapping(candidate.get("last_change"))
    reason = str(last_change.get("change_reason") or "")
    changed_scopes = [str(value) for value in last_change.get("changed_scopes") or []]
    ready = bool(
        candidate.get("bound")
        and candidate.get("accepted_at_utc")
        and last_change.get("timestamp_utc")
        and reason
        and changed_scopes
        and last_change.get("event_hash")
    )
    return {
        "ready": ready,
        "status": "ready" if ready else "candidate_change_receipt_missing",
        "path": str(candidate.get("state_path") or ""),
        "candidate_required": True,
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "candidate_match": bool(candidate.get("bound")),
        "changed_scopes": changed_scopes,
        "change_reason": reason,
        "event_hash": str(last_change.get("event_hash") or ""),
        "synthetic_credit": False,
    }


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    config_path: Path = DEFAULT_CONFIG,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    config_file = (
        config_path if config_path.is_absolute() else project_root / config_path
    )
    policy = load_policy(config_file)
    probe = structural_probe(policy, project_root=project_root)
    candidate = _candidate(project_root)
    controls = _mapping(policy.get("controls"))
    artifacts = {
        control_id: _artifact(
            project_root,
            str(_mapping(controls.get(control_id)).get("evidence_artifact") or ""),
            now=current,
        )
        for control_id in REQUIRED_CONTROL_IDS
    }
    evidence = {
        "independent_factor_benchmarks": _evidence_row(
            artifacts["independent_factor_benchmarks"],
            candidate_id=str(candidate.get("candidate_id") or ""),
        ),
        "pipeline_incident_ownership": _evidence_row(
            artifacts["pipeline_incident_ownership"],
            candidate_id=str(candidate.get("candidate_id") or ""),
            candidate_required=False,
        ),
        "material_strategy_change_governance": _material_change_evidence(candidate),
        "candidate_risk_schedules": _evidence_row(
            artifacts["candidate_risk_schedules"],
            candidate_id=str(candidate.get("candidate_id") or ""),
        ),
        "execution_speed_cost_frontier": _evidence_row(
            artifacts["execution_speed_cost_frontier"],
            candidate_id=str(candidate.get("candidate_id") or ""),
        ),
        "research_dag_checkpoint_resume": _evidence_row(
            artifacts["research_dag_checkpoint_resume"],
            candidate_id=str(candidate.get("candidate_id") or ""),
        ),
        "versioned_research_dataset_storage": _evidence_row(
            artifacts["versioned_research_dataset_storage"],
            candidate_id=str(candidate.get("candidate_id") or ""),
            candidate_required=False,
        ),
        "cross_engine_valuation_reconciliation": _evidence_row(
            artifacts["cross_engine_valuation_reconciliation"],
            candidate_id=str(candidate.get("candidate_id") or ""),
        ),
    }
    evidence_ready_count = sum(1 for row in evidence.values() if row["ready"])
    implementation = {
        control_id: {
            "ready": bool(probe.get("ok")),
            "owner": str(_mapping(controls.get(control_id)).get("owner") or ""),
            "test": str(_mapping(controls.get(control_id)).get("test") or ""),
        }
        for control_id in REQUIRED_CONTROL_IDS
    }
    implementation_count = len(REQUIRED_CONTROL_IDS)
    implementation_ready_count = sum(
        1 for row in implementation.values() if row["ready"]
    )
    structural_ready = bool(
        probe.get("ok") and implementation_ready_count == implementation_count
    )
    evidence_complete = evidence_ready_count == len(REQUIRED_CONTROL_IDS)
    missing_evidence = sorted(
        control_id for control_id, row in evidence.items() if not row["ready"]
    )
    return {
        "timestamp_utc": current.isoformat(),
        "schema_version": 1,
        "policy_id": str(policy.get("policy_id") or ""),
        "ok": structural_ready,
        "overall_status": (
            "ready"
            if structural_ready and evidence_complete
            else "ready_with_evidence_debt" if structural_ready else "blocked"
        ),
        "implementation_grade": "A+" if structural_ready else "F",
        "implementation_ready_count": implementation_ready_count,
        "implementation_control_count": implementation_count,
        "implementation_controls": implementation,
        "evidence_ready_count": evidence_ready_count,
        "evidence_control_count": len(REQUIRED_CONTROL_IDS),
        "evidence_controls": evidence,
        "missing_evidence_control_ids": missing_evidence,
        "candidate_binding": candidate,
        "firm_influences": {
            "reference_count": int(probe.get("firm_reference_count") or 0),
            "organization_count": int(probe.get("firm_organization_count") or 0),
            "influence_only": True,
            "proprietary_replication_claimed": False,
        },
        "structural_probe": probe,
        "paper_soak_ready": structural_ready,
        "paper_impact": "none",
        "candidate_affected_scope_forward_evidence_required": True,
        "cumulative_soak_history_preserved": True,
        "reset_soak_clock": False,
        "extension_evidence_complete": evidence_complete,
        "live_promotion_ready": False,
        "live_execution_authority": False,
        "paper_execution_authority": False,
        "automatic_promotion_authority": False,
        "profitability_guaranteed": False,
        "actions": [
            f"collect_candidate_bound_evidence:{control_id}"
            for control_id in missing_evidence
        ]
        or ["retain_evidence_and_continue_independent_live_release_review"],
        "evidence_semantics": _mapping(policy.get("evidence_semantics")),
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Institutional Research Extensions",
        "",
        f"- Status: `{payload.get('overall_status')}`",
        f"- Structural implementation: `{payload.get('implementation_ready_count')}/{payload.get('implementation_control_count')}` (`{payload.get('implementation_grade')}`)",
        f"- Earned evidence: `{payload.get('evidence_ready_count')}/{payload.get('evidence_control_count')}`",
        f"- Public firm references: `{(payload.get('firm_influences') or {}).get('reference_count')}` across `{(payload.get('firm_influences') or {}).get('organization_count')}` organizations",
        f"- Paper soak impact: `{payload.get('paper_impact')}`",
        f"- Reset soak clock: `{str(bool(payload.get('reset_soak_clock'))).lower()}`",
        f"- Live execution authority: `{str(bool(payload.get('live_execution_authority'))).lower()}`",
        "",
        "## Evidence",
        "",
    ]
    for control_id, row in (payload.get("evidence_controls") or {}).items():
        marker = "ready" if row.get("ready") else "accruing"
        lines.append(f"- `{control_id}`: **{marker}** (`{row.get('status')}`)")
    lines.extend(
        [
            "",
            "These are public design influences implemented as local controls. Structural readiness does not manufacture factor alpha, incident drills, fills, independent pricing observations, or live-release approval.",
        ]
    )
    return "\n".join(lines) + "\n"


def _atomic_write_text(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(body, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the eight advisory institutional research extensions."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT))
    parser.add_argument("--markdown-file", default=str(DEFAULT_MARKDOWN))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    payload = build_payload(root, config_path=Path(args.config).expanduser())
    out = Path(args.out_file).expanduser()
    markdown = Path(args.markdown_file).expanduser()
    if not out.is_absolute():
        out = root / out
    if not markdown.is_absolute():
        markdown = root / markdown
    if not safe_write_json_atomic(
        str(out),
        payload,
        project_root=str(root),
        source="institutional_research_extensions_control",
    ):
        return 2
    _atomic_write_text(markdown, render_markdown(payload))
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    else:
        print(
            f"institutional_research_extensions status={payload['overall_status']} "
            f"implementation={payload['implementation_ready_count']}/{payload['implementation_control_count']} "
            f"evidence={payload['evidence_ready_count']}/{payload['evidence_control_count']}"
        )
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
