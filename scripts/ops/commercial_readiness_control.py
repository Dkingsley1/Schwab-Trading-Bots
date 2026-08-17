#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, load_json, ordered_unique, write_payload


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "commercial_readiness_framework_v1.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "commercial_readiness_control_latest.json"
DEFAULT_MARKDOWN_PATH = PROJECT_ROOT / "exports" / "reports" / "operator" / "commercial_readiness_packet_latest.md"
TRUE_VALUES = {"1", "true", "yes", "on", "enabled", "approved", "ready"}
BAD_STATUSES = {"blocked", "critical", "degraded", "failed", "missing", "needs_work", "stale", "warning"}
READY_STATUSES = {"ready", "ok", "guarded_ready", "stable", "pass", ""}


def _as_dict(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _status(raw: Any) -> str:
    return str(raw or "").strip().lower()


def _env_bool(env: dict[str, str], name: str, default: bool = False) -> bool:
    text = str(env.get(name, "")).strip().lower()
    if not text:
        return default
    return text in TRUE_VALUES


def _project_path(project_root: Path, raw: Any) -> Path:
    path = Path(str(raw or ""))
    return path if path.is_absolute() else project_root / path


def _health(project_root: Path, name: str) -> dict[str, Any]:
    payload = load_json(project_root / "governance" / "health" / name)
    return payload if isinstance(payload, dict) else {}


def _section(section_id: str, title: str, ready: bool, blockers: list[str], evidence: dict[str, Any], *, advisory: bool = False) -> dict[str, Any]:
    clean_blockers = ordered_unique(str(item or "").strip() for item in blockers if str(item or "").strip())
    effective_ready = bool(ready and not clean_blockers)
    return {
        "section_id": section_id,
        "title": title,
        "ready": effective_ready,
        "status": "ready" if effective_ready else ("advisory" if advisory and not clean_blockers else "blocked"),
        "blockers": clean_blockers,
        "evidence": evidence,
    }


def _commercial_flags(config: dict[str, Any], env: dict[str, str]) -> dict[str, bool]:
    return {
        str(name): _env_bool(env, str(name), False)
        for name in _as_list(config.get("commercial_trigger_envs"))
        if str(name).strip()
    }


def _infer_mode(config: dict[str, Any], env: dict[str, str], flags: dict[str, bool], use_mode: dict[str, Any]) -> str:
    mode_env = str(config.get("mode_env") or "COMMERCIAL_PRODUCT_MODE")
    explicit = str(env.get(mode_env) or "").strip().lower()
    if explicit:
        return explicit
    if flags.get("CUSTOMER_FUNDS_ENABLED") or flags.get("CUSTODY_ENABLED") or flags.get("COMMODITY_POOL_ENABLED"):
        return "pooled_customer_funds_model"
    if flags.get("CUSTOMER_ORDER_EXECUTION_ENABLED") or flags.get("CUSTOMER_ACCOUNTS_ENABLED") or flags.get("COPY_TRADING_ENABLED"):
        return "broker_customer_execution"
    if flags.get("FUTURES_OR_DERIVATIVES_ADVICE_ENABLED"):
        return "futures_crypto_commodity_advisory"
    if flags.get("INVESTMENT_ADVICE_ENABLED") or flags.get("MODEL_PORTFOLIO_ENABLED"):
        return "investment_advice_business"
    if flags.get("ADVISER_SUPPORT_SOFTWARE_ENABLED"):
        return "adviser_support_software"
    if flags.get("PAID_SIGNALS_ENABLED"):
        return "paid_signals_newsletter"
    if flags.get("COMMERCIAL_ANALYTICS_REPORTING_ENABLED") or flags.get("PERFORMANCE_MARKETING_ENABLED") or flags.get("PUBLIC_MARKETING_ENABLED"):
        return "paid_analytics_reporting"
    if flags.get("COMMERCIAL_USE_ENABLED") or bool(_as_dict(use_mode.get("commercial_use")).get("commercial_use_intent_detected", False)):
        return "internal_research_tool"
    return str(config.get("default_mode") or "personal_only")


def _review_ready(project_root: Path, env: dict[str, str], row: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    env_name = str(row.get("env") or "")
    evidence_file = str(row.get("evidence_file") or "")
    env_ready = _env_bool(env, env_name, False) if env_name else False
    evidence_path = _project_path(project_root, evidence_file) if evidence_file else None
    file_ready = bool(evidence_path and evidence_path.exists())
    return bool(env_ready or file_ready), {
        "env": env_name,
        "env_ready": env_ready,
        "evidence_file": str(evidence_path) if evidence_path else "",
        "evidence_file_present": file_ready,
    }


def _claims_payload(project_root: Path, config: dict[str, Any], env: dict[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    claim_config = _as_dict(config.get("marketing_claim_control"))
    path = _project_path(project_root, claim_config.get("claims_file") or "governance/commercial/marketing_claims.json")
    payload = load_json(path)
    raw_claims = payload.get("claims") if isinstance(payload.get("claims"), list) else []
    claims = [row for row in raw_claims if isinstance(row, dict)]
    text_claim = str(env.get(str(claim_config.get("claim_text_env") or "COMMERCIAL_MARKETING_CLAIMS_TEXT"), "")).strip()
    if text_claim:
        claims.append({"claim": text_claim, "source": "env", "approved": False})
    return claims, {"claims_file": str(path), "claims_file_present": bool(payload), "claim_count": len(claims)}


def _source_status_ready(payload: dict[str, Any]) -> bool:
    if not payload:
        return False
    status = _status(payload.get("overall_status") or payload.get("status"))
    ok_value = payload.get("ok")
    if ok_value is not None:
        return bool(ok_value) and status not in BAD_STATUSES
    return status in READY_STATUSES


def _mode_section(config: dict[str, Any], mode: str, flags: dict[str, bool], commercial_intent: bool) -> dict[str, Any]:
    modes = _as_dict(config.get("commercial_product_modes"))
    mode_config = _as_dict(modes.get(mode))
    forbidden = [str(item) for item in _as_list(mode_config.get("forbidden_flags")) if str(item).strip()]
    active_forbidden = [name for name in forbidden if flags.get(name, False)]
    blockers = [
        f"unknown_commercial_product_mode={mode}" if not mode_config else "",
        *[f"mode_forbidden_flag_active={name}" for name in active_forbidden],
    ]
    return _section(
        "commercial_use_modes",
        "Commercial Use Modes",
        bool(mode_config and not active_forbidden),
        blockers,
        {
            "mode": mode,
            "commercial_intent": commercial_intent,
            "available_modes": sorted(modes.keys()),
            "active_flags": sorted(name for name, active in flags.items() if active),
            "mode_config": mode_config,
        },
    )


def _review_section(project_root: Path, config: dict[str, Any], env: dict[str, str], mode: str, commercial_intent: bool) -> dict[str, Any]:
    modes = _as_dict(config.get("commercial_product_modes"))
    mode_config = _as_dict(modes.get(mode))
    required = [str(item) for item in _as_list(mode_config.get("required_review_gates")) if str(item).strip()]
    gates = _as_dict(config.get("review_gates"))
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    if commercial_intent:
        for gate in required:
            ready, evidence = _review_ready(project_root, env, _as_dict(gates.get(gate)))
            rows.append({"gate": gate, "ready": ready, **evidence})
            if not ready:
                blockers.append(f"{gate}_not_approved")
    return _section(
        "registration_review_gates",
        "Registration And Review Gates",
        bool((not commercial_intent) or not blockers),
        blockers,
        {"mode": mode, "commercial_intent": commercial_intent, "required_gates": required, "gates": rows},
    )


def _marketing_section(project_root: Path, config: dict[str, Any], env: dict[str, str], flags: dict[str, bool], commercial_intent: bool) -> dict[str, Any]:
    claim_config = _as_dict(config.get("marketing_claim_control"))
    claims, claim_meta = _claims_payload(project_root, config, env)
    marketing_intent = bool(
        flags.get("PERFORMANCE_MARKETING_ENABLED")
        or flags.get("PUBLIC_MARKETING_ENABLED")
        or flags.get("TESTIMONIALS_ENABLED")
        or flags.get("THIRD_PARTY_RATINGS_ENABLED")
        or flags.get("PAID_SIGNALS_ENABLED")
        or claims
    )
    review_ready = _env_bool(env, str(claim_config.get("review_env") or "MARKETING_REVIEW_APPROVED"), False)
    blockers: list[str] = []
    claim_rows: list[dict[str, Any]] = []
    if marketing_intent:
        if not review_ready:
            blockers.append("marketing_review_not_approved")
        if not claims:
            blockers.append("marketing_claim_register_missing")
    for idx, claim in enumerate(claims):
        claim_text = str(claim.get("claim") or claim.get("text") or f"claim_{idx + 1}")
        approved = bool(claim.get("approved", False))
        evidence_artifact = str(claim.get("evidence_artifact") or claim.get("evidence") or "")
        evidence_present = bool(evidence_artifact and _project_path(project_root, evidence_artifact).exists())
        gross_net_ok = bool(claim.get("gross_net_disclosure", False) or not claim.get("uses_gross_performance", False))
        hypothetical_ok = bool(claim.get("hypothetical_or_backtest_label", False) or not claim.get("uses_backtest_or_hypothetical", False))
        raw_label_ok = bool(claim.get("raw_vs_controlled_label", False) or not claim.get("mentions_profitability", False))
        row_blockers = [
            "claim_not_approved" if not approved else "",
            "claim_evidence_missing" if not evidence_present else "",
            "gross_net_disclosure_missing" if not gross_net_ok else "",
            "hypothetical_or_backtest_label_missing" if not hypothetical_ok else "",
            "raw_vs_controlled_profitability_label_missing" if not raw_label_ok else "",
        ]
        row_blockers = [item for item in row_blockers if item]
        blockers.extend(f"claim_{idx + 1}:{item}" for item in row_blockers)
        claim_rows.append(
            {
                "claim": claim_text,
                "approved": approved,
                "evidence_artifact": evidence_artifact,
                "evidence_present": evidence_present,
                "gross_net_disclosure_ok": gross_net_ok,
                "hypothetical_or_backtest_label_ok": hypothetical_ok,
                "raw_vs_controlled_label_ok": raw_label_ok,
                "blockers": row_blockers,
            }
        )
    return _section(
        "marketing_claim_control",
        "Marketing Claim Control",
        bool((not marketing_intent) or not blockers),
        blockers,
        {"marketing_intent": marketing_intent, "marketing_review_ready": review_ready, **claim_meta, "claims": claim_rows},
    )


def _customer_funds_section(config: dict[str, Any], env: dict[str, str], flags: dict[str, bool]) -> dict[str, Any]:
    cfg = _as_dict(config.get("customer_funds_hard_blocks"))
    custody_intent = bool(flags.get("CUSTOMER_FUNDS_ENABLED") or flags.get("CUSTODY_ENABLED") or flags.get("COMMODITY_POOL_ENABLED"))
    order_intent = bool(flags.get("CUSTOMER_ORDER_EXECUTION_ENABLED") or flags.get("CUSTOMER_ACCOUNTS_ENABLED") or flags.get("COPY_TRADING_ENABLED"))
    checks: dict[str, bool] = {
        "customer_funds_program_registered": _env_bool(env, str(cfg.get("customer_funds_program_registered_env") or "CUSTOMER_FUNDS_PROGRAM_REGISTERED"), False),
        "custody_program_review": _env_bool(env, str(cfg.get("custody_program_review_env") or "CUSTODY_PROGRAM_REVIEW_APPROVED"), False),
        "customer_funds_segregation": _env_bool(env, str(cfg.get("customer_funds_segregation_env") or "CUSTOMER_FUNDS_SEGREGATION_EVIDENCE"), False),
        "customer_funds_audit_trail": _env_bool(env, str(cfg.get("customer_funds_audit_trail_env") or "CUSTOMER_FUNDS_AUDIT_TRAIL_READY"), False),
        "broker_dealer_review": _env_bool(env, "BROKER_DEALER_REVIEW_APPROVED", False),
        "best_execution_review": _env_bool(env, str(cfg.get("best_execution_review_env") or "BEST_EXECUTION_REVIEW_READY"), False),
        "customer_order_routing_disclosure": _env_bool(env, str(cfg.get("customer_order_routing_disclosure_env") or "CUSTOMER_ORDER_ROUTING_DISCLOSURE_READY"), False),
        "customer_order_audit_trail": _env_bool(env, str(cfg.get("customer_order_audit_trail_env") or "CUSTOMER_ORDER_AUDIT_TRAIL_READY"), False),
    }
    blockers: list[str] = []
    if custody_intent:
        for key in ("customer_funds_program_registered", "custody_program_review", "customer_funds_segregation", "customer_funds_audit_trail"):
            if not checks[key]:
                blockers.append(f"{key}_missing")
    if order_intent:
        for key in ("broker_dealer_review", "best_execution_review", "customer_order_routing_disclosure", "customer_order_audit_trail"):
            if not checks[key]:
                blockers.append(f"{key}_missing")
    return _section(
        "customer_funds_hard_blocks",
        "Customer/Funds Hard Blocks",
        not blockers,
        blockers,
        {
            "customer_funds_or_custody_intent": custody_intent,
            "customer_order_or_copy_trading_intent": order_intent,
            "checks": checks,
            "authority": {
                "customer_funds_allowed": False,
                "custody_allowed": False,
                "customer_order_execution_allowed": False,
                "copy_trading_allowed": False,
                "live_execution_authority": False,
            },
        },
    )


def _evidence_packet_section(project_root: Path, config: dict[str, Any], commercial_intent: bool) -> dict[str, Any]:
    packet_cfg = _as_dict(config.get("commercial_evidence_packet"))
    required_files = [str(item) for item in _as_list(packet_cfg.get("required_files_for_commercial_intent")) if str(item).strip()]
    rows = []
    missing = []
    for rel in required_files:
        path = _project_path(project_root, rel)
        present = path.exists()
        rows.append({"artifact": rel, "path": str(path), "present": present})
        if commercial_intent and not present:
            missing.append(rel)
    return _section(
        "commercial_evidence_packets",
        "Commercial Evidence Packets",
        bool((not commercial_intent) or not missing),
        [f"commercial_evidence_missing={item}" for item in missing],
        {
            "commercial_intent": commercial_intent,
            "required_file_count": len(required_files),
            "present_file_count": sum(1 for row in rows if row["present"]),
            "files": rows,
            "packet_outputs": {
                "json": str(DEFAULT_OUT_PATH),
                "markdown": str(DEFAULT_MARKDOWN_PATH),
            },
        },
    )


def _self_awareness_section(config: dict[str, Any], use_mode: dict[str, Any]) -> dict[str, Any]:
    surfaces = [str(item) for item in _as_list(config.get("self_awareness_surfaces")) if str(item).strip()]
    use_mode_present = bool(use_mode)
    return _section(
        "self_awareness_expansion",
        "Self-Awareness Expansion",
        bool(surfaces and use_mode_present),
        [
            "self_awareness_surface_catalog_missing" if not surfaces else "",
            "use_mode_compliance_guard_missing" if not use_mode_present else "",
        ],
        {
            "surface_count": len(surfaces),
            "surfaces": surfaces,
            "use_mode_status": use_mode.get("overall_status") if use_mode else "missing",
            "commercial_intent_from_use_mode": bool(_as_dict(use_mode.get("commercial_use")).get("commercial_use_intent_detected", False)),
        },
    )


def _security_privacy_section(project_root: Path, config: dict[str, Any], env: dict[str, str], flags: dict[str, bool], commercial_intent: bool) -> dict[str, Any]:
    cfg = _as_dict(config.get("security_privacy_layer"))
    security = _health(project_root, "security_audit_latest.json")
    secret = _health(project_root, "secret_scan_latest.json")
    redaction = _health(project_root, "telemetry_redaction_canary_latest.json")
    customer_data_intent = any(flags.get(str(name), False) for name in _as_list(cfg.get("customer_data_envs")))
    security_ready = _source_status_ready(security) if security else False
    secret_findings = _safe_int(secret.get("findings_count", _as_dict(secret.get("summary")).get("findings_count", 0)), 0)
    secret_ready = bool(secret and secret_findings == 0 and _status(secret.get("overall_status") or secret.get("status")) not in BAD_STATUSES)
    redaction_ready = _source_status_ready(redaction) if redaction else False
    blockers: list[str] = []
    if commercial_intent:
        if not _env_bool(env, "PRIVACY_SECURITY_REVIEW_APPROVED", False):
            blockers.append("privacy_security_review_not_approved")
        if not security_ready:
            blockers.append("security_audit_not_ready")
        if not secret_ready:
            blockers.append("secret_scan_not_clean")
        if not redaction_ready:
            blockers.append("telemetry_redaction_not_ready")
    customer_controls = {}
    if customer_data_intent:
        for name in [str(item) for item in _as_list(cfg.get("customer_data_required_controls")) if str(item).strip()]:
            ready = _env_bool(env, name, False)
            customer_controls[name] = ready
            if not ready:
                blockers.append(f"customer_data_control_missing={name}")
    return _section(
        "security_privacy_layer",
        "Security And Privacy Layer",
        not blockers,
        blockers,
        {
            "commercial_intent": commercial_intent,
            "customer_data_intent": customer_data_intent,
            "security_audit_status": security.get("overall_status") or security.get("status") or "missing",
            "secret_scan_findings": secret_findings if secret else None,
            "telemetry_redaction_status": redaction.get("overall_status") or redaction.get("status") or "missing",
            "customer_data_controls": customer_controls,
            "policy": "customer_data_requires_written_security_program_access_controls_encryption_mfa_monitoring_training_vendor_oversight_incident_response_and_disposal_controls",
        },
        advisory=not commercial_intent,
    )


def _grade(ready_count: int, total_count: int) -> str:
    ratio = ready_count / max(total_count, 1)
    if ratio >= 1.0:
        return "A+"
    if ratio >= 0.85:
        return "A"
    if ratio >= 0.70:
        return "B"
    if ratio >= 0.55:
        return "C"
    return "D"


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Commercial Readiness Packet",
        "",
        f"Generated UTC: `{payload.get('timestamp_utc', '')}`",
        f"Product mode: `{payload.get('commercial_product_mode', '')}`",
        f"Overall status: `{payload.get('overall_status', '')}`",
        f"Commercial release ready: `{payload.get('commercial_release_ready', False)}`",
        "",
        "## Sections",
        "",
    ]
    for section in _as_list(payload.get("sections")):
        if not isinstance(section, dict):
            continue
        lines.append(
            f"- `{section.get('section_id', '')}`: `{section.get('status', '')}` "
            f"blockers=`{', '.join(_as_list(section.get('blockers'))) or 'none'}`"
        )
    lines.extend(["", "## Recommended Actions", ""])
    for action in _as_list(payload.get("recommended_actions")):
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


def build_payload(
    project_root: Path = PROJECT_ROOT,
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    config = load_json(config_path) or load_json(DEFAULT_CONFIG_PATH)
    runtime_env = dict(os.environ if env is None else env)
    flags = _commercial_flags(config, runtime_env)
    use_mode = _health(project_root, "use_mode_compliance_guard_latest.json")
    mode = _infer_mode(config, runtime_env, flags, use_mode)
    modes = _as_dict(config.get("commercial_product_modes"))
    mode_config = _as_dict(modes.get(mode))
    commercial_intent = bool(mode_config.get("commercial_intent", False) or any(flags.values()))
    mode_section = _mode_section(config, mode, flags, commercial_intent)
    review_section = _review_section(project_root, config, runtime_env, mode, commercial_intent)
    marketing_section = _marketing_section(project_root, config, runtime_env, flags, commercial_intent)
    funds_section = _customer_funds_section(config, runtime_env, flags)
    evidence_section = _evidence_packet_section(project_root, config, commercial_intent)
    awareness_section = _self_awareness_section(config, use_mode)
    security_section = _security_privacy_section(project_root, config, runtime_env, flags, commercial_intent)
    sections = [
        mode_section,
        review_section,
        marketing_section,
        funds_section,
        evidence_section,
        awareness_section,
        security_section,
    ]
    blockers = ordered_unique(
        f"{section['section_id']}:{blocker}"
        for section in sections
        for blocker in _as_list(section.get("blockers"))
    )
    ready_count = sum(1 for section in sections if bool(section.get("ready", False)))
    hard_blocked = bool(mode_config.get("hard_block", False) and commercial_intent)
    commercial_release_ready = bool(commercial_intent and not blockers and not hard_blocked and ready_count == len(sections))
    if commercial_intent and (blockers or hard_blocked):
        overall_status = "blocked"
    elif commercial_intent and commercial_release_ready:
        overall_status = "ready"
    else:
        overall_status = "ready"
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "policy_id": str(config.get("policy_id") or "commercial_readiness_framework_v1"),
        "source": "commercial_readiness_control",
        "ok": overall_status != "blocked",
        "overall_status": overall_status,
        "commercial_product_mode": mode,
        "commercial_intent": commercial_intent,
        "commercial_release_ready": commercial_release_ready,
        "commercial_release_blocked": bool(commercial_intent and not commercial_release_ready),
        "grade": _grade(ready_count, len(sections)),
        "section_count": len(sections),
        "ready_section_count": ready_count,
        "blocked_section_count": sum(1 for section in sections if section.get("status") == "blocked"),
        "sections": sections,
        "blockers": blockers + (["commercial_product_mode_hard_blocked"] if hard_blocked else []),
        "authority_boundaries": {
            "live_execution_authority": False,
            "customer_funds_allowed": False,
            "custody_allowed": False,
            "customer_order_execution_allowed": False,
            "copy_trading_allowed": False,
            "commercial_clearance_is_external_evidence_only": True,
        },
        "seven_section_contract": {
            "commercial_use_modes": True,
            "registration_review_gates": True,
            "marketing_claim_control": True,
            "customer_funds_hard_blocks": True,
            "commercial_evidence_packets": True,
            "self_awareness_expansion": True,
            "security_privacy_layer": True,
        },
        "regulatory_source_references": _as_list(config.get("regulatory_source_references")),
        "recommended_actions": ordered_unique(
            [
                "keep commercial mode personal_only unless a real commercial release packet is being prepared"
                if not commercial_intent
                else "",
                "complete mode-specific legal/compliance/privacy/review evidence before commercial release"
                if commercial_intent and not commercial_release_ready
                else "",
                "do not market performance claims until the claim register is approved and substantiated"
                if marketing_section.get("status") == "blocked"
                else "",
                "do not handle customer funds, custody, customer orders, or copy trading without external registered-program evidence"
                if funds_section.get("status") == "blocked"
                else "",
                "this framework never grants live execution authority",
            ]
        ),
        "artifact_paths": {
            "json": str(DEFAULT_OUT_PATH),
            "markdown": str(DEFAULT_MARKDOWN_PATH),
            "config": str(config_path),
        },
        "control_contract": {
            "covers_all_7_requested_sections": True,
            "commercial_readiness_is_not_legal_approval": True,
            "commercial_readiness_does_not_enable_live_execution": True,
            "personal_soak_is_separate_from_commercial_release": True,
        },
    }


def write_outputs(payload: dict[str, Any], *, out_path: Path = DEFAULT_OUT_PATH, markdown_path: Path = DEFAULT_MARKDOWN_PATH) -> None:
    write_payload(out_path, payload)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate seven-section commercial expansion readiness without granting live execution.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--out-file", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--markdown-file", type=Path, default=DEFAULT_MARKDOWN_PATH)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    payload = build_payload(args.project_root.resolve(), config_path=args.config)
    write_outputs(payload, out_path=args.out_file, markdown_path=args.markdown_file)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "commercial_readiness_control "
            f"status={payload.get('overall_status')} "
            f"mode={payload.get('commercial_product_mode')} "
            f"commercial_release_ready={int(bool(payload.get('commercial_release_ready')))} "
            f"grade={payload.get('grade')} "
            f"blockers={len(_as_list(payload.get('blockers')))}"
        )
    return 0 if payload.get("overall_status") != "blocked" else 2


if __name__ == "__main__":
    raise SystemExit(main())
