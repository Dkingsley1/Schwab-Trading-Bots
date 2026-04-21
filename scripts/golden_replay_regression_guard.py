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

import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import replay_end_to_end_deterministic as replay_src


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def build_payload(*, golden_pack: dict[str, Any]) -> dict[str, Any]:
    cases = golden_pack.get("cases") if isinstance(golden_pack.get("cases"), list) else []
    rows: list[dict[str, Any]] = []
    failed_cases: list[str] = []

    for raw_case in cases:
        if not isinstance(raw_case, dict):
            continue
        name = str(raw_case.get("name") or "").strip() or f"case_{len(rows)}"
        payload = raw_case.get("payload") if isinstance(raw_case.get("payload"), dict) else {}
        expected_hash = str(raw_case.get("expected_hash") or "").strip().lower()
        expected_actions = raw_case.get("expected_actions") if isinstance(raw_case.get("expected_actions"), dict) else {}
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
        hash_match = bool(expected_hash and replay.get("replay_hash") == expected_hash)
        case_ok = bool(hash_match and not mismatched_actions)
        if not case_ok:
            failed_cases.append(name)
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
            }
        )

    ok = bool(rows) and not failed_cases
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "ok": ok,
        "case_count": len(rows),
        "failed_case_count": len(failed_cases),
        "failed_cases": failed_cases,
        "cases": rows,
        "pack_schema_version": golden_pack.get("schema_version"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Require deterministic replay to match golden reference scenarios.")
    parser.add_argument("--pack-file", default=str(DEFAULT_PACK_PATH))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = build_payload(golden_pack=_load_json(Path(args.pack_file)))
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

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
