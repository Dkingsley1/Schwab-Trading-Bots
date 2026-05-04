#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OWNERSHIP_PATH = PROJECT_ROOT / "governance" / "ownership" / "bot_support_owners.json"
SCOPE_EXEMPT_TOKENS = (
    "collection_floor",
    "min_active_floor_override",
    "bucket_diversity",
    "manual_collection_restore",
    "manual_canary_restore",
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _parse_bot_id_csv(value: str | None) -> set[str]:
    return {item.strip().lower() for item in str(value or "").split(",") if item.strip()}


def _scope_exempt_reason(row: dict[str, Any]) -> str:
    tokens = " ".join(
        [
            str(row.get("reason", "") or ""),
            str(row.get("promotion_reason", "") or ""),
            str(row.get("promotion_status", "") or ""),
            str(row.get("bot_role", row.get("role", "")) or ""),
        ]
    ).lower()
    if "support_control" in tokens or "infrastructure_sub_bot" in tokens:
        return "support_control"
    if any(token in tokens for token in SCOPE_EXEMPT_TOKENS):
        return "coverage_exempt"
    return ""


def _truthy_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "on"}


def _admission_scope_requested(raw_row: dict[str, Any], wf_row: dict[str, Any]) -> bool:
    if bool(raw_row.get("active", False)):
        return True
    if _truthy_flag(raw_row.get("paper_trading")) or _truthy_flag(raw_row.get("shadow_mode")):
        return True
    promotion_status = str(raw_row.get("promotion_status") or "").strip().lower()
    if promotion_status in {"candidate", "shadow", "paper", "probation", "challenger"}:
        return True
    return False


def _normalized_owner(value: Any) -> str:
    return str(value or "").strip()


def _resolve_support_owner(bot_id: str, ownership_payload: dict[str, Any]) -> tuple[str, str]:
    exact = ownership_payload.get("owners_by_bot_id") if isinstance(ownership_payload.get("owners_by_bot_id"), dict) else {}
    prefixes = ownership_payload.get("owners_by_prefix") if isinstance(ownership_payload.get("owners_by_prefix"), dict) else {}
    normalized_id = str(bot_id or "").strip().lower()

    owner = _normalized_owner(exact.get(normalized_id) or exact.get(bot_id))
    if owner:
        return owner, "owners_by_bot_id"

    best_prefix = ""
    best_owner = ""
    for raw_prefix, raw_owner in prefixes.items():
        prefix = str(raw_prefix or "").strip().lower()
        owner = _normalized_owner(raw_owner)
        if not prefix or not owner:
            continue
        if normalized_id.startswith(prefix) and len(prefix) > len(best_prefix):
            best_prefix = prefix
            best_owner = owner
    if best_owner:
        return best_owner, f"owners_by_prefix:{best_prefix}"

    default_owner = _normalized_owner(ownership_payload.get("default_owner"))
    if default_owner:
        return default_owner, "default_owner"
    return "", ""


def _replay_hashes_present(payload: dict[str, Any]) -> bool:
    details = payload.get("details") if isinstance(payload.get("details"), dict) else {}
    paper = details.get("paper") if isinstance(details.get("paper"), dict) else {}
    e2e = details.get("e2e") if isinstance(details.get("e2e"), dict) else {}
    return bool(
        str(paper.get("current_hash") or paper.get("expected_hash") or "").strip()
        and str(e2e.get("current_hash") or e2e.get("expected_hash") or "").strip()
    )


def build_payload(
    *,
    registry: dict[str, Any],
    walk_forward: dict[str, Any],
    feature_store_manifest: dict[str, Any],
    replay_hash_registry_guard: dict[str, Any],
    ownership_payload: dict[str, Any],
    diagnostics_root: Path,
    min_training_sample_count: int,
    min_eligible_sequences: int,
    min_walk_forward_runs: int,
    include_bot_ids: set[str] | None = None,
    advisory_only: bool = False,
) -> dict[str, Any]:
    sub_bots = registry.get("sub_bots") if isinstance(registry.get("sub_bots"), list) else []
    wf_bots = walk_forward.get("bots") if isinstance(walk_forward.get("bots"), dict) else {}
    include_bot_ids = {item.strip().lower() for item in (include_bot_ids or set()) if item.strip()}

    point_in_time_contract = (
        feature_store_manifest.get("point_in_time_contract")
        if isinstance(feature_store_manifest.get("point_in_time_contract"), dict)
        else {}
    )
    contract_hashes = feature_store_manifest.get("contract_hashes") if isinstance(feature_store_manifest.get("contract_hashes"), dict) else {}
    feature_manifest_ready = bool(
        feature_store_manifest.get("ok", False)
        and point_in_time_contract.get("complete", False)
        and str(contract_hashes.get("dataset_manifest_sha256") or "").strip()
    )
    replay_hash_registry_ready = bool(
        replay_hash_registry_guard.get("ok", False)
        and _replay_hashes_present(replay_hash_registry_guard)
    )

    candidate_rows: list[dict[str, Any]] = []
    passing_rows = 0

    for raw_row in sub_bots:
        if not isinstance(raw_row, dict):
            continue
        bot_id = str(raw_row.get("bot_id") or "").strip()
        if not bot_id or _scope_exempt_reason(raw_row):
            continue
        if include_bot_ids and bot_id.lower() not in include_bot_ids:
            continue
        wf_row = wf_bots.get(bot_id) if isinstance(wf_bots.get(bot_id), dict) else {}
        lifecycle_state = str(raw_row.get("lifecycle_state") or "").strip().lower()
        if not _admission_scope_requested(raw_row, wf_row):
            continue
        runs = _to_int((wf_row or {}).get("runs"), 0)
        is_admission_candidate = lifecycle_state == "probation" or runs < int(min_walk_forward_runs)
        if not is_admission_candidate:
            continue

        diag = _load_json(diagnostics_root / f"{bot_id}_latest.json")
        sample_count = _to_int(diag.get("sample_count"), 0)
        eligible_sequences = _to_int(diag.get("eligible_sequences"), 0)
        sequence_count = _to_int(diag.get("sequence_count"), 0)
        owner, owner_source = _resolve_support_owner(bot_id, ownership_payload)
        status = str(diag.get("status") or "").strip().lower()

        failed_contracts: list[str] = []
        if not owner:
            failed_contracts.append("support_owner_missing")
        if sample_count < int(min_training_sample_count):
            failed_contracts.append(f"sample_count<{int(min_training_sample_count)}")
        if eligible_sequences < int(min_eligible_sequences):
            failed_contracts.append(f"eligible_sequences<{int(min_eligible_sequences)}")
        if runs < int(min_walk_forward_runs):
            failed_contracts.append(f"walk_forward_runs<{int(min_walk_forward_runs)}")
        if status == "deferred_sample_starved":
            failed_contracts.append("training_diagnostics_deferred_sample_starved")

        row = {
            "bot_id": bot_id,
            "lifecycle_state": lifecycle_state or "active",
            "walk_forward_runs": runs,
            "sample_count": sample_count,
            "eligible_sequences": eligible_sequences,
            "sequence_count": sequence_count,
            "support_owner": owner,
            "support_owner_source": owner_source,
            "training_status": status or "unknown",
            "failed_contracts": failed_contracts,
        }
        if not failed_contracts:
            passing_rows += 1
        candidate_rows.append(row)

    global_failed_checks: list[str] = []
    if candidate_rows and not feature_manifest_ready:
        global_failed_checks.append("feature_store_manifest_not_strict_ready")
    if candidate_rows and not replay_hash_registry_ready:
        global_failed_checks.append("replay_hash_registry_not_ready")

    blocking_candidates = [row for row in candidate_rows if row.get("failed_contracts")]
    contract_ok = bool(not global_failed_checks and not blocking_candidates)
    ok = True if advisory_only else contract_ok

    top_actions: list[str] = []
    if any("support_owner_missing" in row.get("failed_contracts", []) for row in candidate_rows):
        top_actions.append("assign_named_support_owner_before_promoting_or_widening_new-bot traffic")
    if any(
        any(item.startswith("sample_count<") or item.startswith("eligible_sequences<") for item in row.get("failed_contracts", []))
        for row in candidate_rows
    ):
        top_actions.append("raise new-bot training sample depth before letting the bot graduate beyond probation")
    if candidate_rows and not feature_manifest_ready:
        top_actions.append("refresh feature_store_manifest until point-in-time completeness and manifest hashes are present")
    if candidate_rows and not replay_hash_registry_ready:
        top_actions.append("refresh replay hash registry so admission candidates inherit a verified replay baseline")

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": ok,
        "contract_ok": contract_ok,
        "advisory_only": bool(advisory_only),
        "scope": {
            "include_bot_ids": sorted(include_bot_ids),
            "target_scoped": bool(include_bot_ids),
        },
        "thresholds": {
            "min_training_sample_count": int(min_training_sample_count),
            "min_eligible_sequences": int(min_eligible_sequences),
            "min_walk_forward_runs": int(min_walk_forward_runs),
        },
        "global_prerequisites": {
            "feature_store_manifest_ready": feature_manifest_ready,
            "replay_hash_registry_ready": replay_hash_registry_ready,
            "global_failed_checks": global_failed_checks,
        },
        "ownership": {
            "configured_exact_owners": len(
                ownership_payload.get("owners_by_bot_id")
                if isinstance(ownership_payload.get("owners_by_bot_id"), dict)
                else {}
            ),
            "configured_prefix_owners": len(
                ownership_payload.get("owners_by_prefix")
                if isinstance(ownership_payload.get("owners_by_prefix"), dict)
                else {}
            ),
            "default_owner_present": bool(_normalized_owner(ownership_payload.get("default_owner"))),
        },
        "candidate_bot_count": len(candidate_rows),
        "admission_scope_active_count": len(candidate_rows),
        "passing_candidate_count": int(passing_rows),
        "blocking_candidate_count": len(blocking_candidates),
        "blocking_candidates": blocking_candidates[:30],
        "candidates": candidate_rows[:50],
        "top_actions": top_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Admission contract guard for new or probationary bots.")
    parser.add_argument("--registry", default=str(PROJECT_ROOT / "master_bot_registry.json"))
    parser.add_argument("--walk-forward-file", default=str(PROJECT_ROOT / "governance" / "walk_forward" / "walk_forward_latest.json"))
    parser.add_argument("--feature-store-manifest", default=str(PROJECT_ROOT / "governance" / "feature_store" / "latest.json"))
    parser.add_argument("--replay-hash-registry-file", default=str(PROJECT_ROOT / "governance" / "health" / "replay_hash_registry_guard_latest.json"))
    parser.add_argument("--ownership-file", default=str(DEFAULT_OWNERSHIP_PATH))
    parser.add_argument("--diagnostics-root", default=str(PROJECT_ROOT / "governance" / "training_diagnostics"))
    parser.add_argument("--min-training-sample-count", type=int, default=40)
    parser.add_argument("--min-eligible-sequences", type=int, default=4)
    parser.add_argument("--min-walk-forward-runs", type=int, default=12)
    parser.add_argument("--include-bot-ids", default="", help="Optional comma-separated bot ids to scope the admission check.")
    parser.add_argument(
        "--advisory-only",
        action="store_true",
        help="Write admission findings but do not block the caller. Used for targeted coverage repair retrains.",
    )
    parser.add_argument("--out-file", default=str(PROJECT_ROOT / "governance" / "health" / "new_bot_admission_guard_latest.json"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        registry=_load_json(Path(args.registry)),
        walk_forward=_load_json(Path(args.walk_forward_file)),
        feature_store_manifest=_load_json(Path(args.feature_store_manifest)),
        replay_hash_registry_guard=_load_json(Path(args.replay_hash_registry_file)),
        ownership_payload=_load_json(Path(args.ownership_file)),
        diagnostics_root=Path(args.diagnostics_root),
        min_training_sample_count=int(args.min_training_sample_count),
        min_eligible_sequences=int(args.min_eligible_sequences),
        min_walk_forward_runs=int(args.min_walk_forward_runs),
        include_bot_ids=_parse_bot_id_csv(args.include_bot_ids),
        advisory_only=bool(args.advisory_only),
    )

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "new_bot_admission_guard "
            f"ok={str(payload['ok']).lower()} "
            f"candidates={int(payload['candidate_bot_count'])} "
            f"blocking={int(payload['blocking_candidate_count'])}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
