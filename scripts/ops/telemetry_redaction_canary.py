#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import iso_now, load_json, ordered_unique, write_payload
    from scripts.ops.production_readiness_control import build_observability_redaction_domain
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
    from .production_readiness_control import build_observability_redaction_domain


DEFAULT_CONFIG = PROJECT_ROOT / "config" / "production_readiness_control_v1.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "telemetry_redaction_canary_latest.json"


def _config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    config_path: Path | None = None,
) -> dict[str, Any]:
    config_path = config_path or project_root / "config" / "production_readiness_control_v1.json"
    config = load_json(config_path)
    redaction_config = (
        config.get("observability_redaction")
        if isinstance(config.get("observability_redaction"), dict)
        else {}
    )
    domain = build_observability_redaction_domain(redaction_config)
    evidence = domain.get("evidence") if isinstance(domain.get("evidence"), dict) else {}
    sample_rows = [row for row in evidence.get("sample_rows") or [] if isinstance(row, dict)]
    pattern_count = int(evidence.get("pattern_count", 0) or 0)
    leak_count = sum(int(row.get("leak_count", 0) or 0) for row in sample_rows)
    blockers = ordered_unique(
        [
            *(domain.get("blockers") or []),
            "redaction_patterns_missing" if pattern_count <= 0 else "",
            "redaction_canary_samples_missing" if not sample_rows else "",
            "redaction_canary_leak_detected" if leak_count > 0 else "",
        ]
    )
    ready = bool(not blockers and str(domain.get("status") or "") == "ready")
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 2,
        "ok": ready,
        "overall_status": "ready" if ready else "blocked",
        "production_grade_ready": ready,
        "sample_count": len(sample_rows),
        "passed_sample_count": sum(1 for row in sample_rows if bool(row.get("ok", False))),
        "leak_count": leak_count,
        "pattern_count": pattern_count,
        "enabled_by_default": bool(evidence.get("enabled_by_default", False)),
        "allowed_export_modes": list(evidence.get("allowed_export_modes") or []),
        "sample_rows": sample_rows,
        "blockers": blockers,
        "recommended_actions": ordered_unique(
            [
                *(domain.get("recommended_actions") or []),
                "restore configured redaction patterns and canary samples before enabling telemetry export"
                if not ready
                else "",
            ]
        ),
        "source_config": {
            "path": str(config_path),
            "sha256": _config_hash(redaction_config),
        },
        "control_contract": {
            "tests_current_configuration": True,
            "raw_canary_inputs_persisted": False,
            "fails_closed_on_missing_patterns_or_samples": True,
            "telemetry_export_activation_separate": True,
            "policy": str(evidence.get("policy") or "telemetry_must_redact_sensitive_identifiers"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute and publish the telemetry redaction canary.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--config", default="")
    parser.add_argument("--out-file", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    config_path = Path(args.config).expanduser() if args.config else project_root / "config" / "production_readiness_control_v1.json"
    out_path = Path(args.out_file).expanduser() if args.out_file else project_root / "governance" / "health" / "telemetry_redaction_canary_latest.json"
    payload = build_payload(project_root, config_path=config_path)
    write_payload(out_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "telemetry_redaction_canary "
            f"overall_status={payload.get('overall_status', '')} "
            f"samples={payload.get('sample_count', 0)} "
            f"leaks={payload.get('leak_count', 0)}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
