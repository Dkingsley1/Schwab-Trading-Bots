#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.collect_schwab_symbol_news import load_ticker_universe, load_ticker_universe_with_policy


REQUIRED_PROFILES = {"local_mac_soak", "ci", "paper_prod", "live_canary"}
TICKER_UNIVERSE_ENV_PREFIXES = ("TICKER_UNIVERSE_",)
POLICY_FILES = {
    "deployment_profiles": "config/deployment_profiles.json",
    "self_healing": "config/self_healing_policy.json",
    "credential_runtime": "config/credential_runtime_policy.json",
    "promotion_gate_snapshots": "config/promotion_gate_snapshot_policy.json",
    "generated_artifacts": "config/generated_artifact_policy.json",
    "use_mode_compliance": "config/use_mode_compliance_policy_v1.json",
    "commercial_readiness": "config/commercial_readiness_framework_v1.json",
    "live_canary_readiness": "config/live_canary_readiness_contract.json",
    "production_excellence": "config/production_excellence_v1.json",
}


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else {}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def check(name: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        detail = fn()
        ok = bool(detail.pop("ok", False))
        return {"name": name, "ok": ok, "status": "pass" if ok else "fail", **detail}
    except Exception as exc:
        return {"name": name, "ok": False, "status": "error", "error": f"{type(exc).__name__}: {exc}"}


@contextmanager
def isolated_ticker_universe_env() -> Any:
    """Keep static contract checks independent from live runtime env overrides."""
    keys = [key for key in os.environ if key.startswith(TICKER_UNIVERSE_ENV_PREFIXES)]
    original = {key: os.environ[key] for key in keys}
    for key in keys:
        os.environ.pop(key, None)
    try:
        yield
    finally:
        for key in [key for key in os.environ if key.startswith(TICKER_UNIVERSE_ENV_PREFIXES)]:
            os.environ.pop(key, None)
        os.environ.update(original)


def check_registry_write_guard(project_root: Path) -> dict[str, Any]:
    master_text = read_text(project_root / "core" / "master_bot.py")
    cli_text = read_text(project_root / "scripts" / "run_master_bot.py")
    paper_standard_text = read_text(project_root / "scripts" / "ops" / "paper_live_data_standard.py")
    paper_ramp_text = read_text(project_root / "scripts" / "ops" / "paper_400_ramp_control.py")
    runtime_throttle_text = read_text(project_root / "scripts" / "ops" / "runtime_throttle_control.py")
    required_tokens = [
        "MASTER_ALLOW_SOURCE_REGISTRY_WRITE",
        "master_bot_registry_candidate_latest.json",
        "canonical_registry_requires_explicit_source_write",
    ]
    missing = [token for token in required_tokens if token not in master_text]
    cli_ok = "--allow-source-registry-write" in cli_text
    paper_tokens = [
        "PAPER_LIVE_DATA_ALLOW_SOURCE_REGISTRY_WRITE",
        "paper_live_data_standard_registry_candidate_latest.json",
        "paper_live_data_standard_source_write_guard_latest.json",
    ]
    paper_missing = [token for token in paper_tokens if token not in paper_standard_text]
    paper_ramp_tokens = [
        "PAPER_400_RAMP_ALLOW_SOURCE_REGISTRY_WRITE",
        "paper_400_ramp_registry_candidate_latest.json",
        "paper_400_ramp_source_write_guard_latest.json",
        "--allow-source-registry-write",
    ]
    paper_ramp_missing = [token for token in paper_ramp_tokens if token not in paper_ramp_text]
    runtime_throttle_tokens = [
        "RUNTIME_THROTTLE_ALLOW_SOURCE_REGISTRY_WRITE",
        "runtime_throttle_registry_candidate_latest.json",
        "runtime_throttle_source_write_guard_latest.json",
        "--allow-source-registry-write",
    ]
    runtime_throttle_missing = [token for token in runtime_throttle_tokens if token not in runtime_throttle_text]
    return {
        "ok": not missing and cli_ok and not paper_missing and not paper_ramp_missing and not runtime_throttle_missing,
        "missing_tokens": missing,
        "cli_flag_present": cli_ok,
        "paper_standard_missing_tokens": paper_missing,
        "paper_ramp_missing_tokens": paper_ramp_missing,
        "runtime_throttle_missing_tokens": runtime_throttle_missing,
        "source_write_default": "candidate_only",
    }


def check_showcase_workflow(project_root: Path) -> dict[str, Any]:
    text = read_text(project_root / ".github" / "workflows" / "refresh-showcase.yml")
    has_push = "git push" in text
    uploads_artifact = "actions/upload-artifact" in text
    contents_read = "contents: read" in text
    return {
        "ok": (not has_push) and uploads_artifact and contents_read,
        "has_git_push": has_push,
        "uploads_artifact": uploads_artifact,
        "contents_read": contents_read,
    }


def check_ticker_universe_contract() -> dict[str, Any]:
    with isolated_ticker_universe_env():
        symbols, groups, source = load_ticker_universe()
    sentinel_missing = [symbol for symbol in ("HUT", "ACWI", "GFF") if symbol not in symbols]

    with tempfile.TemporaryDirectory() as tmp:
        project_root = Path(tmp)
        health_root = project_root / "governance" / "health"
        health_root.mkdir(parents=True)
        (health_root / "ingestion_storage_control_latest.json").write_text(
            json.dumps(
                {
                    "overall_status": "blocked",
                    "severity": "critical",
                    "backpressure": {
                        "effective_raw_live": {
                            "core_pending_lines": 18000,
                            "total_pending_lines": 42000,
                            "oldest_pending_age_seconds": 900,
                        },
                        "pending_lines_threshold": 5000,
                        "total_pending_lines_threshold": 15000,
                        "oldest_age_threshold_seconds": 240,
                    },
                },
                ensure_ascii=True,
                indent=2,
            ),
            encoding="utf-8",
        )
        with isolated_ticker_universe_env():
            pressure_symbols, _pressure_groups, _pressure_source, policy = load_ticker_universe_with_policy(
                project_root=project_root
            )

    return {
        "ok": not sentinel_missing
        and len(symbols) >= 1000
        and bool(source)
        and len(pressure_symbols) == 500
        and bool(policy.get("storage_pressure_active"))
        and policy.get("mode") == "slow_tier_deferred_for_storage_pressure",
        "symbol_count": len(symbols),
        "sentinel_missing": sentinel_missing,
        "hut_groups": groups.get("HUT", []),
        "source": source,
        "pressure_symbol_count": len(pressure_symbols),
        "pressure_mode": policy.get("mode"),
    }


def check_policy_configs(project_root: Path) -> dict[str, Any]:
    loaded = {name: load_json(project_root / rel_path) for name, rel_path in POLICY_FILES.items()}
    profiles = loaded["deployment_profiles"].get("profiles") if isinstance(loaded["deployment_profiles"].get("profiles"), dict) else {}
    profile_missing = sorted(REQUIRED_PROFILES.difference(profiles))
    healing_defaults = loaded["self_healing"].get("defaults") if isinstance(loaded["self_healing"].get("defaults"), dict) else {}
    credential = loaded["credential_runtime"]
    promotion = loaded["promotion_gate_snapshots"]
    generated = loaded["generated_artifacts"]
    use_mode = loaded["use_mode_compliance"]
    commercial = loaded["commercial_readiness"]
    canary = loaded["live_canary_readiness"]
    excellence = loaded["production_excellence"]
    forbidden_paths = loaded["self_healing"].get("forbidden_source_paths") or []

    conditions = {
        "profiles_present": not profile_missing,
        "candidate_only_registry": all(
            str(profiles.get(profile, {}).get("source_registry_write")) in {"candidate_only", "forbidden"}
            for profile in ("local_mac_soak", "ci", "paper_prod")
        ),
        "healer_dry_run_first": healing_defaults.get("dry_run_first") is True,
        "healer_rate_limited": int(healing_defaults.get("max_actions_per_cycle", 0) or 0) <= 3,
        "canonical_files_forbidden": "master_bot_registry.json" in forbidden_paths,
        "interactive_creds_blocked": credential.get("interactive_credential_entry") is False,
        "token_lease_monitor_enabled": bool((credential.get("token_lease_monitor") or {}).get("enabled")),
        "unknown_blocks_promotion": promotion.get("unknown_blocks_promotion") is True,
        "snapshot_versioning_required": bool((promotion.get("snapshot_versioning") or {}).get("required")),
        "generated_autocommit_disabled": generated.get("auto_commit_generated_changes") is False,
        "large_reports_externalized": generated.get("large_report_storage") == "ci_or_release_artifact",
        "use_mode_has_operator_grade_personal_autonomy": {
            str(row.get("criterion_id") or "")
            for row in (use_mode.get("personal_use_operator_grade_criteria") or [])
            if isinstance(row, dict)
        }
        >= {
            "base_personal_a_plus_ready",
            "a_plus_operating_packet_all_lanes",
            "unattended_soak_green",
            "source_mutation_guard_clean",
            "production_flow_smoke_ready",
            "autonomy_recovery_score",
            "disaster_recovery_blackstart_ready",
            "data_plane_recovery_managed",
            "live_money_boundaries_locked",
            "commercial_personal_boundary_clean",
            "security_privacy_runtime_clean",
        },
        "commercial_framework_has_7_sections": set(commercial.get("seven_sections") or []) == {
            "commercial_use_modes",
            "registration_review_gates",
            "marketing_claim_control",
            "customer_funds_hard_blocks",
            "commercial_evidence_packets",
            "self_awareness_expansion",
            "security_privacy_layer",
        },
        "commercial_framework_never_grants_live_authority": "commercial_readiness_does_not_enable_live_execution" in str(commercial.get("control_policy") or "")
        or bool((commercial.get("commercial_evidence_packet") or {}).get("output_json")),
        "live_canary_contract_has_all_hard_gates": set(canary.get("readiness_bar") or []) == set()
        or all(
            phrase in str(canary.get("infrastructure_message") or "")
            for phrase in (
                "no raw D-grade posture",
                "no unexplained sleeve paper-trading dropouts",
                "no auth/token surprises",
                "no source mutation from runtime",
                "clean CI",
                "clean storage pressure",
                "clean promotion/paper gate freshness",
                "sustained window",
            )
        ),
        "production_excellence_has_ten_pillars": set((excellence.get("candidate") or {}).get("soak_scopes") or [])
        >= {"strategy", "execution", "risk", "data", "promotion", "operations", "dependencies"}
        and len((excellence.get("recovery") or {}).get("required_drills") or []) == 10
        and float((excellence.get("soak") or {}).get("required_hours", 0) or 0) >= 720
        and float((excellence.get("canary") or {}).get("max_initial_weight", 1) or 1) <= 0.01,
    }
    return {
        "ok": all(conditions.values()),
        "conditions": conditions,
        "profile_missing": profile_missing,
        "policy_files": POLICY_FILES,
    }


def check_ci_guardrails(project_root: Path) -> dict[str, Any]:
    text = read_text(project_root / ".github" / "workflows" / "ci_guardrails.yml")
    return {
        "ok": "workflow_dispatch:" in text
        and "production_flow_smoke.py --json" in text
        and "command_validity_bot.py --help" in text
        and "commands_hygiene_bot.py --help" in text
        and "use_mode_compliance_guard.py --help" in text
        and "commercial_readiness_control.py --help" in text
        and "source_mutation_guard.py --check-clean --json" in text
        and "production_hardening_watch.py --help" in text
        and "infrabot_library_self_awareness_control.py --help" in text
        and "paper_400_ramp_control.py --help" in text
        and "runtime_throttle_control.py --help" in text
        and "production_level_upgrade_hardener_control.py --help" in text
        and "production_quality_control.py --help" in text
        and "production_quality_slo_guard.py --help" in text
        and "production_excellence_control.py --help" in text
        and "live_order_ledger_control.py --help" in text
        and "canary_rollout_guard.py --help" in text
        and "paper_execution_calibration_report.py --help" in text
        and "promotion_quality_gate.py --help" in text
        and "independent_fill_evidence_acquisition.py --help" in text
        and "promotion_candidate_advancement.py --help" in text
        and "readiness_blocker_rollup.py --help" in text
        and "readiness_evidence_accrual.py --help" in text
        and "readiness_evidence_refresh.py --help" in text
        and "uniform_hardening_contract.py --help" in text
        and "uniform_hardening_contract.py --structural-only --json" in text,
        "manual_dispatch_enabled": "workflow_dispatch:" in text,
        "production_smoke_in_ci": "production_flow_smoke.py --json" in text,
        "command_validity_bot_in_ci": "command_validity_bot.py --help" in text,
        "commands_hygiene_bot_in_ci": "commands_hygiene_bot.py --help" in text,
        "use_mode_compliance_guard_in_ci": "use_mode_compliance_guard.py --help" in text,
        "commercial_readiness_control_in_ci": "commercial_readiness_control.py --help" in text,
        "source_mutation_guard_in_ci": "source_mutation_guard.py --check-clean --json" in text,
        "production_hardening_watch_in_ci": "production_hardening_watch.py --help" in text,
        "infrabot_library_self_awareness_control_in_ci": "infrabot_library_self_awareness_control.py --help" in text,
        "paper_400_ramp_control_in_ci": "paper_400_ramp_control.py --help" in text,
        "runtime_throttle_control_in_ci": "runtime_throttle_control.py --help" in text,
        "production_level_upgrade_hardener_control_in_ci": "production_level_upgrade_hardener_control.py --help" in text,
        "production_quality_control_in_ci": "production_quality_control.py --help" in text,
        "production_quality_slo_guard_in_ci": "production_quality_slo_guard.py --help" in text,
        "production_excellence_control_in_ci": "production_excellence_control.py --help" in text,
        "live_order_ledger_control_in_ci": "live_order_ledger_control.py --help" in text,
        "candidate_bound_canary_in_ci": "canary_rollout_guard.py --help" in text,
        "paper_calibration_in_ci": "paper_execution_calibration_report.py --help" in text,
        "promotion_quality_in_ci": "promotion_quality_gate.py --help" in text,
        "independent_fill_acquisition_in_ci": "independent_fill_evidence_acquisition.py --help" in text,
        "candidate_advancement_in_ci": "promotion_candidate_advancement.py --help" in text,
        "readiness_blocker_rollup_in_ci": "readiness_blocker_rollup.py --help" in text,
        "readiness_evidence_accrual_in_ci": "readiness_evidence_accrual.py --help" in text,
        "readiness_evidence_refresh_in_ci": "readiness_evidence_refresh.py --help" in text,
        "uniform_hardening_contract_in_ci": "uniform_hardening_contract.py --help" in text
        and "uniform_hardening_contract.py --structural-only --json" in text,
    }


def check_gitignore_artifact_policy(project_root: Path) -> dict[str, Any]:
    text = read_text(project_root / ".gitignore")
    required = ["output/pdf/*.pdf", "output/pdf/*.html", "output/pdf/*.png", "output/pdf/*.json", "output/pdf/*.zip"]
    missing = [item for item in required if item not in text]
    return {"ok": not missing, "missing_ignores": missing}


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    checks = [
        check("registry_source_write_guard", lambda: check_registry_write_guard(project_root)),
        check("showcase_generated_artifact_flow", lambda: check_showcase_workflow(project_root)),
        check("stale_latest_ticker_universe_contract", check_ticker_universe_contract),
        check("deployment_healing_credential_promotion_policies", lambda: check_policy_configs(project_root)),
        check("ci_production_smoke_coverage", lambda: check_ci_guardrails(project_root)),
        check("generated_artifact_ignore_policy", lambda: check_gitignore_artifact_policy(project_root)),
    ]
    ok = all(item["ok"] for item in checks)
    return {
        "timestamp_utc": iso_now(),
        "ok": ok,
        "overall_status": "ready" if ok else "blocked",
        "check": "production_flow_smoke",
        "project_root": str(project_root),
        "checks": checks,
        "failed_checks": [item["name"] for item in checks if not item["ok"]],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate production-flow guardrails for unattended soak operation.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    payload = build_payload(Path(args.project_root).resolve())
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        print(
            "production_flow_smoke "
            f"status={payload['overall_status']} "
            f"failed={','.join(payload['failed_checks']) or 'none'}"
        )
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
