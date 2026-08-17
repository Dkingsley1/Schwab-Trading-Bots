#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACK_PATH = PROJECT_ROOT / "governance" / "replay" / "golden_replay_pack.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "golden_replay_regression_latest.json"
DEFAULT_REPLAY_HASH_REGISTRY_PATH = PROJECT_ROOT / "governance" / "health" / "replay_hash_registry_guard_latest.json"

import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import replay_end_to_end_deterministic as replay_src
from scripts.ops.long_runtime_common import write_payload


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _registry_seed_ready(replay_hash_registry: dict[str, Any]) -> bool:
    details = replay_hash_registry.get("details") if isinstance(replay_hash_registry.get("details"), dict) else {}
    paper = details.get("paper") if isinstance(details.get("paper"), dict) else {}
    e2e = details.get("e2e") if isinstance(details.get("e2e"), dict) else {}
    return bool(
        replay_hash_registry.get("ok", False)
        and (
            str(paper.get("current_hash") or "").strip()
            or str(e2e.get("current_hash") or "").strip()
            or str(replay_hash_registry.get("registry_file") or "").strip()
        )
    )


def build_payload(*, golden_pack: dict[str, Any], replay_hash_registry: dict[str, Any] | None = None) -> dict[str, Any]:
    replay_hash_registry = replay_hash_registry or {}
    cases = golden_pack.get("cases") if isinstance(golden_pack.get("cases"), list) else []
    rows: list[dict[str, Any]] = []
    failed_cases: list[str] = []
    case_names: list[str] = []
    covered_contracts: set[str] = set()

    for raw_case in cases:
        if not isinstance(raw_case, dict):
            continue
        name = str(raw_case.get("name") or "").strip() or f"case_{len(rows)}"
        payload = raw_case.get("payload") if isinstance(raw_case.get("payload"), dict) else {}
        expected_hash = str(raw_case.get("expected_hash") or "").strip().lower()
        expected_actions = raw_case.get("expected_actions") if isinstance(raw_case.get("expected_actions"), dict) else {}
        expected_results = raw_case.get("expected_results") if isinstance(raw_case.get("expected_results"), list) else []
        coverage = [str(item) for item in raw_case.get("coverage", []) if str(item or "").strip()]
        replay = replay_src.run_replay(payload)
        actual_actions = {
            str((row or {}).get("symbol") or "").strip(): str((row or {}).get("action_out") or "").strip()
            for row in (replay.get("canonical", {}).get("results") or [])
            if isinstance(row, dict)
        }
        mismatched_actions = [
            symbol
            for symbol, expected_action in expected_actions.items()
            if actual_actions.get(str(symbol).strip()) != str(expected_action).strip()
        ]
        actual_results = [
            {
                "symbol": str((row or {}).get("symbol") or "").strip(),
                "action_out": str((row or {}).get("action_out") or "").strip(),
            }
            for row in (replay.get("canonical", {}).get("results") or [])
            if isinstance(row, dict)
        ]
        normalized_expected_results = [
            {
                "symbol": str((row or {}).get("symbol") or "").strip(),
                "action_out": str((row or {}).get("action_out") or "").strip(),
            }
            for row in expected_results
            if isinstance(row, dict)
        ]
        result_sequence_match = bool(not normalized_expected_results or normalized_expected_results == actual_results)
        hash_match = bool(expected_hash and replay.get("replay_hash") == expected_hash)
        case_ok = bool(hash_match and not mismatched_actions and result_sequence_match)
        if not case_ok:
            failed_cases.append(name)
        else:
            covered_contracts.update(coverage)
        case_names.append(name)
        rows.append(
            {
                "name": name,
                "ok": case_ok,
                "expected_hash": expected_hash,
                "actual_hash": str(replay.get("replay_hash") or ""),
                "hash_match": hash_match,
                "expected_actions": expected_actions,
                "actual_actions": actual_actions,
                "mismatched_actions": mismatched_actions,
                "expected_results": normalized_expected_results,
                "actual_results": actual_results,
                "result_sequence_match": result_sequence_match,
                "coverage": coverage,
            }
        )

    registry_seed_ready = _registry_seed_ready(replay_hash_registry)
    required_coverage = {
        str(item) for item in golden_pack.get("required_coverage", []) if str(item or "").strip()
    }
    duplicate_case_names = sorted({name for name in case_names if case_names.count(name) > 1})
    missing_coverage = sorted(required_coverage - covered_contracts)
    pack_contract_declared = bool(required_coverage)
    pack_contract_valid = bool(not duplicate_case_names and not missing_coverage)
    if rows:
        ok = bool(not failed_cases and pack_contract_valid)
        overall_status = "ready" if ok else "blocked"
        summary = (
            "golden replay scenarios matched the deterministic reference pack"
            if ok
            else "golden replay regression detected a hash or action mismatch"
        )
    else:
        ok = registry_seed_ready
        overall_status = "degraded" if registry_seed_ready else "blocked"
        summary = (
            "golden replay pack is not available yet, but replay hash registry coverage is healthy enough for seeded review"
            if registry_seed_ready
            else "golden replay pack is missing and no replay-hash fallback is available"
        )
    recommended_actions: list[str] = []
    if not rows:
        recommended_actions.append("publish a golden replay pack before treating replay proof as fully strict-ready")
    if failed_cases:
        recommended_actions.append("repair the failing replay scenarios before promoting a new candidate")
    if duplicate_case_names:
        recommended_actions.append("give every golden replay scenario a unique case name")
    if missing_coverage:
        recommended_actions.append("add passing golden replay cases for: " + ", ".join(missing_coverage))
    if not registry_seed_ready:
        recommended_actions.append("refresh the replay hash registry so seeded replay fallback evidence is available")

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 2,
        "ok": ok,
        "overall_status": overall_status,
        "seed_ready": registry_seed_ready,
        "strict_ready": bool(rows and ok and (pack_contract_valid if pack_contract_declared else True)),
        "summary": summary,
        "case_count": len(rows),
        "failed_case_count": len(failed_cases),
        "failed_cases": failed_cases,
        "duplicate_case_names": duplicate_case_names,
        "required_coverage": sorted(required_coverage),
        "covered_contracts": sorted(covered_contracts),
        "missing_coverage": missing_coverage,
        "pack_contract_declared": pack_contract_declared,
        "pack_contract_valid": pack_contract_valid,
        "cases": rows,
        "pack_schema_version": golden_pack.get("schema_version"),
        "recommended_actions": recommended_actions,
        "source_artifacts": {
            "golden_replay_pack": str(DEFAULT_PACK_PATH),
            "replay_hash_registry_guard": str(DEFAULT_REPLAY_HASH_REGISTRY_PATH),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Require deterministic replay to match golden reference scenarios.")
    parser.add_argument("--pack-file", default=str(DEFAULT_PACK_PATH))
    parser.add_argument("--replay-hash-registry-file", default=str(DEFAULT_REPLAY_HASH_REGISTRY_PATH))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(
        golden_pack=_load_json(Path(args.pack_file)),
        replay_hash_registry=_load_json(Path(args.replay_hash_registry_file)),
    )
    out_path = Path(args.out_file)
    write_payload(out_path, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "golden_replay_regression_guard "
            f"ok={str(payload['ok']).lower()} "
            f"cases={int(payload.get('case_count', 0) or 0)} "
            f"failed={int(payload.get('failed_case_count', 0) or 0)}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
