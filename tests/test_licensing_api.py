import hashlib
import hmac
import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.brokers import BrokerRuntimeConfig
from core.licensing_api import available_connector_names, build_partner_api
from core.licensing_api.grade_snapshot import build_grade_snapshot
from core.licensing_api.models import LicensingTenantContext


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


def test_grade_snapshot_uses_a_plus_continuous_storage_soak_contract(tmp_path: Path) -> None:
    health = tmp_path / "governance" / "health"
    _write_json(
        health / "platform_control_plane_latest.json",
        {
            "institutional_readiness": {"overall_score": 100.0},
            "institutional_domains_by_slug": {
                "developer_process": {"score": 100.0},
                "formal_model_governance": {"score": 100.0},
                "high_fidelity_simulator": {"score": 100.0},
                "immutable_experiment_tracking": {"score": 100.0},
                "independent_risk_services": {"score": 100.0},
                "observability_and_slo": {"score": 100.0},
                "point_in_time_data_lineage": {"score": 100.0},
                "portfolio_construction": {"score": 100.0},
                "reliability_engineering": {"score": 100.0},
                "security_and_compliance": {"score": 100.0},
                "statistical_research_discipline": {"score": 100.0},
                "transaction_cost_and_capacity": {"score": 100.0},
            },
        },
    )
    _write_json(health / "live_readiness_smoke_latest.json", {"readiness_score": 100.0})
    _write_json(health / "live_canary_control_latest.json", {"preclearance_score": 100.0})
    _write_json(health / "incident_closeout_autopilot_latest.json", {"closeout_score": 100.0, "open_incident_count": 0})
    _write_json(health / "portable_brain_contract_latest.json", {"portability_score": 100.0})
    _write_json(
        health / "cost_telemetry_latest.json",
        {
            "overall_status": "ready",
            "storage_cost_proxy": {"tracked_sqlite_gb": 183.034},
            "portable_backend_cost_proxy": {"proof_present_count": 3},
        },
    )
    _write_json(
        health / "ingestion_storage_control_latest.json",
        {
            "overall_status": "ready",
            "severity": "stable",
            "recovery_state": "steady_state",
            "recovery_quality_score": 75.0,
            "backpressure_quality_score": 99.0,
            "pressure_index": 0.151,
            "continuous_run_soak_contract": {
                "status": "ready",
                "ready": True,
                "soak_ready": True,
                "grade": "A+",
                "blockers": [],
            },
            "backlog_truth": {
                "raw_live": {"grade": "A+", "core_pending_lines": 2261, "total_pending_lines": 4098},
                "sql_overlay": {"grade": "A+", "core_pending_lines": 0, "total_pending_lines": 0},
            },
            "raw_live_expansion_contract": {
                "grade": "A+",
                "expansion_ready": True,
                "hard_block": False,
            },
        },
    )
    _write_json(
        health / "storage_backpressure_autopilot_latest.json",
        {
            "overall_status": "applied_with_followups",
            "metrics": {"backpressure_actionable": True, "attempted_step_count": 1},
        },
    )

    snapshot = build_grade_snapshot(
        project_root=tmp_path,
        runtime_config=BrokerRuntimeConfig.from_env(),
        tenant=LicensingTenantContext(tenant_id="local", company_name="Local", connector_name="default"),
        endpoint_count=18,
    )
    storage = snapshot["section_grades"]["data_ingestion_and_storage"]

    assert storage["letter_grade"] == "A+"
    assert storage["raw_letter_grade"] == "A+"
    assert storage["raw_score"] >= 96.0
    assert storage["floor_state"] == "at_floor"
    assert storage["signals"]["continuous_storage_soak_a_plus_ready"] is True


def test_partner_api_overview_is_tenant_aware(monkeypatch, tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "live_readiness_smoke_latest.json", {"overall_status": "ready", "readiness_score": 91.2, "broker_ready": True, "session_ready": True})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"overall_status": "degraded"})
    _write_json(health_root / "autonomy_control_plane_latest.json", {"overall_status": "ready", "autonomy_score": 82.5})
    _write_json(
        health_root / "portable_brain_contract_latest.json",
        {
            "overall_status": "ready",
            "recommended_runtime_mode": "native",
            "recommended_backend": "native_default",
            "host_contract": {
                "host_profile": "max_throughput",
                "chip": "Apple M5 Max",
                "memory_architecture": "unified",
                "shared_cpu_gpu_memory_pool": True,
                "memory_competitive_advantage": "Shared CPU/GPU memory reduces copies for feature windows and inference.",
            },
        },
    )
    _write_json(
        health_root / "architecture_upgrade_scoreboard_latest.json",
        {
            "upgrade_count": 12,
            "ready_count": 9,
            "special_features_map": {
                "adaptive_apple_silicon_brain": "Adaptive Apple Silicon Brain: unified memory recognized.",
            },
        },
    )

    monkeypatch.setenv("LICENSING_API_KEYS", "partner-a=secret-a")
    monkeypatch.setenv("LICENSING_API_TENANT_COMPANIES", "partner-a=Acme Capital")
    monkeypatch.setenv("LICENSING_API_TENANT_QUOTAS", "partner-a=req_per_minute:120|monthly_requests:100000")
    monkeypatch.setenv("LICENSING_API_TENANT_USAGE", "partner-a=requests_today:45|requests_month:1200")
    monkeypatch.setenv("LICENSING_API_TENANT_KEY_ROTATION", "partner-a=state:scheduled|rotation_due_utc:2026-05-01T00:00:00Z")
    monkeypatch.setenv("DATA_BROKER", "schwab")
    monkeypatch.setenv("MARKET_DATA_PROVIDER", "coinbase")
    monkeypatch.setenv("PAPER_EXECUTION_BROKER", "mock")
    monkeypatch.setenv("LIVE_EXECUTION_BROKER", "schwab")
    monkeypatch.setenv("AUTH_BROKER", "schwab")

    client = TestClient(build_partner_api(project_root=tmp_path))
    response = client.get(
        "/v1/license/overview",
        headers={"X-API-Key": "secret-a", "X-Tenant-Id": "partner-a"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "default" in available_connector_names()
    assert payload["api_version"] == "v1"
    assert payload["endpoint"] == "overview"
    assert payload["tenant"]["tenant_id"] == "partner-a"
    assert payload["tenant"]["company_name"] == "Acme Capital"
    assert payload["quota"]["req_per_minute"] == 120
    assert payload["usage"]["requests_today"] == 45
    assert payload["data"]["brokers"]["configured"]["market_data_provider"] == "coinbase"
    assert payload["data"]["runtime"]["memory_architecture"] == "unified"
    assert payload["data"]["special_features"]["feature_count"] == 12
    assert payload["connector"]["response_envelope_version"] == 1


def test_partner_api_rejects_missing_or_bad_keys(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LICENSING_API_KEYS", "partner-a=secret-a")
    client = TestClient(build_partner_api(project_root=tmp_path))

    missing = client.get("/v1/license/health")
    bad = client.get("/v1/license/health", headers={"X-API-Key": "wrong"})

    assert missing.status_code == 401
    assert bad.status_code == 403


def test_partner_api_supports_hashed_keys_and_signed_responses(monkeypatch, tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "live_readiness_smoke_latest.json", {"overall_status": "ready", "readiness_score": 91.2})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "portable_brain_contract_latest.json", {"overall_status": "ready", "host_contract": {"memory_architecture": "unified"}})

    raw_key = "licensed-secret"
    raw_key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    signing_secret = "sign-me"
    request_id = "req-partner-001"

    monkeypatch.setenv("LICENSING_API_KEYS", f"partner-a=sha256:{raw_key_hash}")
    monkeypatch.setenv("LICENSING_API_TENANT_SIGNING_KEYS", f"partner-a={signing_secret}")
    monkeypatch.setenv("LICENSING_API_TENANT_SIGNING_KEY_IDS", "partner-a=sig-a")

    client = TestClient(build_partner_api(project_root=tmp_path))
    response = client.get(
        "/v1/license/runtime",
        headers={
            "X-API-Key": raw_key,
            "X-Tenant-Id": "partner-a",
            "X-Request-Id": request_id,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["request_id"] == request_id
    assert payload["signature"]["algorithm"] == "hmac-sha256"
    assert payload["signature"]["key_id"] == "sig-a"

    signed_payload = json.dumps(
        {
            "api_version": payload["api_version"],
            "endpoint": payload["endpoint"],
            "request_id": payload["request_id"],
            "tenant_id": payload["tenant"]["tenant_id"],
            "data": payload["data"],
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    expected_digest = hmac.new(signing_secret.encode("utf-8"), signed_payload, hashlib.sha256).hexdigest()
    assert payload["signature"]["digest"] == expected_digest


def test_partner_api_usage_audit_and_contracts_surfaces(monkeypatch, tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "live_readiness_smoke_latest.json", {"overall_status": "ready", "readiness_score": 88.0})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"overall_status": "degraded"})
    _write_json(health_root / "autonomy_control_plane_latest.json", {"overall_status": "blocked", "autonomy_score": 55.0})
    _write_json(
        health_root / "portable_brain_contract_latest.json",
        {
            "overall_status": "ready",
            "recommended_runtime_mode": "native",
            "recommended_backend": "native_default",
            "host_contract": {
                "host_profile": "max_throughput",
                "chip": "Apple M5 Max",
                "memory_architecture": "unified",
                "shared_cpu_gpu_memory_pool": True,
                "memory_competitive_advantage": "Shared CPU/GPU memory reduces copies.",
            },
            "unified_memory_telemetry": {"competitive_advantage": "apple_silicon_unified_memory"},
        },
    )
    _write_json(health_root / "architecture_upgrade_scoreboard_latest.json", {"upgrade_count": 3, "ready_count": 2})
    _write_json(
        health_root / "incident_report_latest.json",
        {
            "review_required": True,
            "review_state": "awaiting_remediation",
            "incident_counts": {"open_incident_count": 2},
            "recent_incidents": [{"category": "auth_lease"}],
            "closeout_contract": {"closeout_ready": False},
        },
    )
    _write_json(
        health_root / "training_quality_control_latest.json",
        {"immutable_lineage": {"lineage_status": "blocked", "exact_replay_ready": False}},
    )

    monkeypatch.setenv("LICENSING_API_KEYS", "partner-a=secret-a")
    monkeypatch.setenv("LICENSING_API_TENANT_QUOTAS", "partner-a=req_per_minute:120|monthly_requests:100000")
    monkeypatch.setenv("LICENSING_API_TENANT_USAGE", "partner-a=requests_today:45|requests_month:1200")
    monkeypatch.setenv("LICENSING_API_TENANT_BILLING", "partner-a=plan:institutional|monthly_mrr_usd:5000")
    monkeypatch.setenv("LICENSING_API_TENANT_METERING", "partner-a=storage_gb:12|parity_runs:4")
    monkeypatch.setenv("LICENSING_API_TENANT_WEBHOOKS", "partner-a=deliveries_today:8|failures_today:0")
    monkeypatch.setenv("LICENSING_API_TENANT_KEY_ROTATION", "partner-a=state:scheduled|rotation_due_utc:2026-05-01T00:00:00Z")

    client = TestClient(build_partner_api(project_root=tmp_path))
    usage = client.get("/v1/license/usage", headers={"X-API-Key": "secret-a", "X-Tenant-Id": "partner-a"})
    audit = client.get("/v1/license/audit", headers={"X-API-Key": "secret-a", "X-Tenant-Id": "partner-a"})
    contracts = client.get("/v1/license/contracts", headers={"X-API-Key": "secret-a", "X-Tenant-Id": "partner-a"})

    assert usage.status_code == 200
    assert usage.json()["data"]["quota"]["monthly_requests"] == 100000
    assert audit.status_code == 200
    assert audit.json()["data"]["key_rotation"]["state"] == "scheduled"
    assert contracts.status_code == 200
    assert contracts.json()["data"]["lineage_contract"]["lineage_status"] == "blocked"
    assert contracts.json()["data"]["incident_contract"]["review_required"] is True


def test_partner_api_exposes_billing_metering_and_webhook_event_surfaces(monkeypatch, tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(
        health_root / "cost_telemetry_latest.json",
        {
            "tenant_metering_contract": {
                "ready": True,
                "billable_dimensions": ["tracked_sqlite_gb", "training_quality_score"],
            },
            "storage_cost_proxy": {"cost_index": 42.0},
            "training_cost_proxy": {"cost_index": 12.0},
        },
    )
    _write_json(health_root / "incident_report_latest.json", {"review_required": True})
    _write_json(health_root / "live_canary_control_latest.json", {"preapproved_supervised_ready": True})
    _write_json(tmp_path / "governance" / "champion_challenger" / "promotion_packet_latest.json", {"packet_complete": True})

    monkeypatch.setenv("LICENSING_API_KEYS", "partner-a=secret-a")
    monkeypatch.setenv("LICENSING_API_TENANT_BILLING", "partner-a=plan:institutional|monthly_mrr_usd:5000")
    monkeypatch.setenv("LICENSING_API_TENANT_METERING", "partner-a=storage_gb:12|parity_runs:4")
    monkeypatch.setenv("LICENSING_API_TENANT_WEBHOOKS", "partner-a=deliveries_today:8|failures_today:0")

    client = TestClient(build_partner_api(project_root=tmp_path))
    billing = client.get("/v1/license/billing", headers={"X-API-Key": "secret-a", "X-Tenant-Id": "partner-a"})
    metering = client.get("/v1/license/metering", headers={"X-API-Key": "secret-a", "X-Tenant-Id": "partner-a"})
    webhook_events = client.get("/v1/license/webhook-events", headers={"X-API-Key": "secret-a", "X-Tenant-Id": "partner-a"})

    assert billing.status_code == 200
    assert billing.json()["data"]["billing"]["monthly_mrr_usd"] == 5000
    assert metering.status_code == 200
    assert metering.json()["data"]["metering"]["parity_runs"] == 4
    assert webhook_events.status_code == 200
    assert webhook_events.json()["data"]["delivery"]["deliveries_today"] == 8


def test_partner_api_exposes_grades_and_readiness_map(monkeypatch, tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "live_readiness_smoke_latest.json", {"overall_status": "ready", "readiness_score": 91.0})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"overall_status": "degraded"})
    _write_json(health_root / "autonomy_control_plane_latest.json", {"overall_status": "degraded", "autonomy_score": 79.0, "component_statuses": {"runtime_throttle_control": "degraded"}})
    _write_json(health_root / "portable_brain_contract_latest.json", {"overall_status": "ready", "portability_score": 100.0})
    _write_json(
        health_root / "training_quality_control_latest.json",
        {
            "overall_status": "needs_attention",
            "training_quality_score": 82.0,
            "failure_buckets": ["coverage_shortfall"],
            "promotion_packet_ready": True,
            "rollout": {"considered_gap": 4},
            "immutable_lineage": {
                "provisional_lineage_ready": True,
                "replay_hash_guard_ok": True,
            },
        },
    )
    _write_json(
        health_root / "training_lineage_manifest_latest.json",
        {
            "overall_status": "degraded",
            "lineage_score": 89.0,
        },
    )
    _write_json(
        tmp_path / "governance" / "champion_challenger" / "promotion_autopilot_packet_latest.json",
        {
            "overall_status": "degraded",
            "committee_packet_seed_ready": True,
        },
    )
    _write_json(
        tmp_path / "governance" / "walk_forward" / "coverage_gap_closer_latest.json",
        {
            "overall_status": "waiting_for_idle",
            "staged_candidate_count": 4,
            "autopilot_contract": {
                "launch_state": "waiting_for_idle",
                "stage_candidate_count": 4,
                "can_apply_stage": True,
            },
        },
    )
    _write_json(health_root / "retrain_launch_latest.json", {"state": "completed", "final_status": "ok"})
    _write_json(
        health_root / "ingestion_storage_control_latest.json",
        {
            "overall_status": "degraded",
            "severity": "critical",
            "backpressure_quality_score": 78.0,
            "recovery_quality_score": 91.0,
            "pressure_index": 4.2,
            "recovery_state": "recovering_under_guard",
            "bounded_recovery_contract": {"active": True},
            "top_actions": ["keep drain active"],
        },
    )
    _write_json(
        health_root / "cost_telemetry_latest.json",
        {
            "overall_status": "ready",
            "storage_cost_proxy": {"tracked_sqlite_gb": 165.5},
            "portable_backend_cost_proxy": {"proof_present_count": 3},
        },
    )
    _write_json(health_root / "security_audit_latest.json", {"overall_status": "needs_work", "summary": {"passed_checks": 15, "failed_checks": 1}})
    _write_json(health_root / "security_evidence_autofix_latest.json", {"overall_status": "ready", "blockers": []})
    _write_json(
        health_root / "incident_closeout_autopilot_latest.json",
        {
            "overall_status": "degraded",
            "closeout_ready": False,
            "closeout_score": 74.0,
            "open_incident_count": 1,
            "bounded_closeout_path_ready": True,
            "blocking_surfaces": [{"surface": "runtime_clearance"}],
        },
    )
    _write_json(health_root / "runtime_artifact_refresh_latest.json", {"overall_status": "ready", "required_missing_after": []})
    _write_json(health_root / "chrome_headless_guard_latest.json", {"overall_status": "degraded"})
    _write_json(health_root / "process_watchdog_latest.json", {"restart_storms": [], "alerts": []})
    _write_json(
        health_root / "platform_control_plane_latest.json",
        {
            "institutional_readiness": {"overall_score": 91.0},
            "institutional_domains_by_slug": {
                "developer_process": {"score": 100.0},
                "formal_model_governance": {"score": 89.0},
                "high_fidelity_simulator": {"score": 100.0},
                "immutable_experiment_tracking": {"score": 68.0},
                "independent_risk_services": {"score": 88.0},
                "observability_and_slo": {"score": 94.0},
                "point_in_time_data_lineage": {"score": 100.0},
                "portfolio_construction": {"score": 84.0},
                "reliability_engineering": {"score": 100.0},
                "security_and_compliance": {"score": 100.0},
                "statistical_research_discipline": {"score": 95.25},
                "transaction_cost_and_capacity": {"score": 84.0},
            },
        },
    )
    _write_json(
        health_root / "live_canary_control_latest.json",
        {
            "overall_status": "degraded",
            "recommended_mode": "preapproved_supervised",
            "supervised_canary_ready": False,
            "staged_preclearance_ready": True,
            "preapproved_supervised_ready": True,
            "preclearance_score": 95.0,
            "blocking_reasons": ["promotion_packet_not_ready"],
        },
    )

    monkeypatch.setenv("LICENSING_API_KEYS", "partner-a=secret-a")

    client = TestClient(build_partner_api(project_root=tmp_path))
    grades = client.get("/v1/license/grades", headers={"X-API-Key": "secret-a", "X-Tenant-Id": "partner-a"})
    readiness = client.get("/v1/license/readiness-map", headers={"X-API-Key": "secret-a", "X-Tenant-Id": "partner-a"})
    capabilities = client.get("/v1/license/capabilities", headers={"X-API-Key": "secret-a", "X-Tenant-Id": "partner-a"})
    schema = client.get("/v1/license/schema", headers={"X-API-Key": "secret-a", "X-Tenant-Id": "partner-a"})
    webhooks = client.get("/v1/license/webhook-contracts", headers={"X-API-Key": "secret-a", "X-Tenant-Id": "partner-a"})

    assert grades.status_code == 200
    assert grades.json()["data"]["section_grades"]["training_and_model_quality"]["letter_grade"].startswith("A")
    assert grades.json()["data"]["section_grades"]["training_and_model_quality"]["raw_score"] >= 84.0
    assert grades.json()["data"]["section_grades"]["training_and_model_quality"]["floor_state"] in {"at_floor", "protected_by_floor"}
    assert grades.json()["data"]["section_grades"]["api_and_partner_readiness"]["letter_grade"] == "A+"
    assert grades.json()["data"]["section_grades"]["architecture_and_modularity"]["letter_grade"].startswith("A")
    assert grades.json()["data"]["section_grades"]["observability_and_reporting"]["letter_grade"] == "A+"
    assert grades.json()["data"]["section_grades"]["research_and_simulation_depth"]["letter_grade"] == "A+"
    assert readiness.status_code == 200
    assert readiness.json()["data"]["live_canary"]["staged_preclearance_ready"] is True
    assert readiness.json()["data"]["live_canary"]["preapproved_supervised_ready"] is True
    assert readiness.json()["data"]["storage"]["recovery_state"] == "recovering_under_guard"
    assert capabilities.status_code == 200
    assert capabilities.json()["data"]["webhook_contracts"] is True
    assert schema.status_code == 200
    assert "schema" in schema.json()["data"]["endpoints"]
    assert webhooks.status_code == 200
    assert len(webhooks.json()["data"]["events"]) >= 4


def test_partner_api_enforces_tenant_endpoint_allowlist(monkeypatch, tmp_path: Path) -> None:
    health_root = tmp_path / "governance" / "health"
    _write_json(health_root / "live_readiness_smoke_latest.json", {"overall_status": "ready", "readiness_score": 91.2})
    _write_json(health_root / "live_runtime_separation_control_latest.json", {"overall_status": "ready"})
    _write_json(health_root / "portable_brain_contract_latest.json", {"overall_status": "ready"})

    monkeypatch.setenv("LICENSING_API_KEYS", "partner-a=secret-a")
    monkeypatch.setenv("LICENSING_API_TENANT_ALLOWED_ENDPOINTS", "partner-a=health|runtime|tenant")

    client = TestClient(build_partner_api(project_root=tmp_path))
    allowed = client.get("/v1/license/health", headers={"X-API-Key": "secret-a"})
    blocked = client.get("/v1/license/overview", headers={"X-API-Key": "secret-a"})
    tenant_profile = client.get("/v1/license/tenant", headers={"X-API-Key": "secret-a"})

    assert allowed.status_code == 200
    assert blocked.status_code == 403
    assert tenant_profile.status_code == 200
    assert tenant_profile.json()["data"]["tenant"]["allowed_endpoints"] == ["health", "runtime", "tenant"]
