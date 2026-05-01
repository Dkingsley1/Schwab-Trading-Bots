from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from core.brokers import BrokerRuntimeConfig, available_broker_names, available_broker_names_for_role
from core.licensing_api.base import LicensingAPIConnector, load_json_file
from core.licensing_api.grade_snapshot import build_grade_snapshot
from core.licensing_api.models import LicensingTenantContext


def _artifact_contract(path: Path) -> Dict[str, Any]:
    payload = load_json_file(path)
    exists = path.exists()
    if not exists:
        return {
            "path": str(path),
            "exists": False,
            "status": "missing",
            "reason": "artifact_missing",
        }
    status = str(payload.get("overall_status") or payload.get("status") or "unknown")
    if not payload:
        return {
            "path": str(path),
            "exists": True,
            "status": "unknown",
            "reason": "artifact_unreadable_or_empty",
        }
    return {
        "path": str(path),
        "exists": True,
        "status": status,
        "reason": ("ok" if status not in {"", "unknown"} else "status_missing"),
    }


class DefaultLicensingAPIConnector(LicensingAPIConnector):
    name = "default"
    display_name = "Default Partner Licensing Connector"
    exposed_endpoints = (
        "tenant",
        "connectors",
        "overview",
        "health",
        "brokers",
        "runtime",
        "special_features",
        "usage",
        "billing",
        "metering",
        "webhook_events",
        "audit",
        "contracts",
        "capabilities",
        "schema",
        "webhook_contracts",
        "grades",
        "readiness_map",
    )

    def describe(self) -> Dict[str, Any]:
        payload = super().describe()
        payload.update(
            {
                "exposes": list(self.exposed_endpoints),
                "control_plane": "enterprise_contract_v4",
            }
        )
        return payload

    def grades_payload(
        self,
        *,
        project_root: Path,
        runtime_config: BrokerRuntimeConfig,
        tenant: LicensingTenantContext,
    ) -> Dict[str, Any]:
        return build_grade_snapshot(
            project_root=project_root,
            runtime_config=runtime_config,
            tenant=tenant,
            endpoint_count=len(self.exposed_endpoints),
        )

    def capabilities_payload(
        self,
        *,
        project_root: Path,
        runtime_config: BrokerRuntimeConfig,
        tenant: LicensingTenantContext,
    ) -> Dict[str, Any]:
        _ = project_root
        return {
            "tenant_aware_auth": True,
            "response_signing_supported": True,
            "quota_and_usage_tracking": True,
            "billing_and_metering": True,
            "audit_surfaces": True,
            "schema_surfaces": True,
            "webhook_contracts": True,
            "webhook_delivery_audit": True,
            "grade_and_readiness_surfaces": True,
            "runtime_roles": {
                "market_data_provider": runtime_config.market_data_provider_name,
                "paper_execution_broker": runtime_config.paper_execution_broker_name,
                "live_execution_broker": runtime_config.execution_broker_name,
                "auth_broker": runtime_config.auth_broker_name,
            },
            "partner_features": [
                "tenant-aware licensing envelopes",
                "signed response envelopes",
                "quota and usage snapshots",
                "billing and metering surfaces",
                "audit and contract surfaces",
                "schema and webhook contracts",
                "grade and readiness maps",
            ],
            "request_context": {
                "tenant_id": tenant.tenant_id,
                "allowed_endpoints": list(tenant.allowed_endpoints),
                "permissions": list(tenant.permissions),
            },
        }

    def schema_payload(
        self,
        *,
        project_root: Path,
        runtime_config: BrokerRuntimeConfig,
        tenant: LicensingTenantContext,
    ) -> Dict[str, Any]:
        _ = (project_root, runtime_config, tenant)
        endpoint_schemas = {
            endpoint: {
                "method": "GET",
                "auth_headers": ["X-API-Key", "X-Tenant-Id", "X-Request-Id"],
                "response_envelope_version": self.response_envelope_version,
            }
            for endpoint in self.exposed_endpoints
        }
        return {
            "api_version": self.api_version,
            "response_envelope_version": self.response_envelope_version,
            "auth_headers": ["X-API-Key", "X-API-Key-Id", "X-Tenant-Id", "X-Request-Id"],
            "signature_contract": {
                "algorithm": "hmac-sha256",
                "supported": True,
                "requires_tenant_signing_secret": True,
            },
            "endpoints": endpoint_schemas,
        }

    def webhook_contracts_payload(
        self,
        *,
        project_root: Path,
        runtime_config: BrokerRuntimeConfig,
        tenant: LicensingTenantContext,
    ) -> Dict[str, Any]:
        _ = (project_root, runtime_config)
        return {
            "supported": True,
            "tenant_opt_in_required": True,
            "delivery_audit_supported": True,
            "events": [
                {"event": "incident.review_required", "delivery_mode": "signed_json"},
                {"event": "promotion.packet_ready", "delivery_mode": "signed_json"},
                {"event": "live.canary.preapproved", "delivery_mode": "signed_json"},
                {"event": "storage.recovery.stabilized", "delivery_mode": "signed_json"},
            ],
            "delivery_snapshot": dict(tenant.webhook_delivery_snapshot),
            "tenant_metadata": dict(tenant.metadata),
        }

    def readiness_map_payload(
        self,
        *,
        project_root: Path,
        runtime_config: BrokerRuntimeConfig,
        tenant: LicensingTenantContext,
    ) -> Dict[str, Any]:
        _ = (runtime_config, tenant)
        health_root = project_root / "governance" / "health"
        training = load_json_file(health_root / "training_quality_control_latest.json")
        storage = load_json_file(health_root / "ingestion_storage_control_latest.json")
        security_audit = load_json_file(health_root / "security_audit_latest.json")
        security_evidence = load_json_file(health_root / "security_evidence_autofix_latest.json")
        incident_closeout = load_json_file(health_root / "incident_closeout_autopilot_latest.json")
        live_canary = load_json_file(health_root / "live_canary_control_latest.json")
        autonomy = load_json_file(health_root / "autonomy_control_plane_latest.json")
        return {
            "training": {
                "overall_status": str(training.get("overall_status") or "unknown"),
                "failure_buckets": list(training.get("failure_buckets") or []),
                "promotion_packet_ready": bool(training.get("promotion_packet_ready", False)),
            },
            "storage": {
                "overall_status": str(storage.get("overall_status") or "unknown"),
                "severity": str(storage.get("severity") or "unknown"),
                "recovery_state": str(storage.get("recovery_state") or ""),
                "top_actions": list(storage.get("top_actions") or []),
            },
            "security": {
                "audit_status": str(security_audit.get("overall_status") or "unknown"),
                "evidence_status": str(security_evidence.get("overall_status") or "unknown"),
                "blockers": list(security_evidence.get("blockers") or []),
            },
            "incident_closeout": {
                "overall_status": str(incident_closeout.get("overall_status") or "unknown"),
                "closeout_ready": bool(incident_closeout.get("closeout_ready", False)),
                "blocking_surfaces": list(incident_closeout.get("blocking_surfaces") or []),
            },
            "live_canary": {
                "overall_status": str(live_canary.get("overall_status") or "unknown"),
                "recommended_mode": str(live_canary.get("recommended_mode") or ""),
                "supervised_canary_ready": bool(live_canary.get("supervised_canary_ready", False)),
                "staged_preclearance_ready": bool(live_canary.get("staged_preclearance_ready", False)),
                "preapproved_supervised_ready": bool(live_canary.get("preapproved_supervised_ready", False)),
                "blocking_reasons": list(live_canary.get("blocking_reasons") or []),
            },
            "autonomy": {
                "overall_status": str(autonomy.get("overall_status") or "unknown"),
                "autonomy_score": float(autonomy.get("autonomy_score", 0.0) or 0.0),
                "component_statuses": dict(autonomy.get("component_statuses") or {}),
            },
        }

    def health_payload(self, *, project_root: Path, tenant: LicensingTenantContext) -> Dict[str, Any]:
        _ = tenant
        health_root = project_root / "governance" / "health"
        live_readiness = load_json_file(health_root / "live_readiness_smoke_latest.json")
        runtime = load_json_file(health_root / "live_runtime_separation_control_latest.json")
        autonomy = load_json_file(health_root / "autonomy_control_plane_latest.json")
        portable = load_json_file(health_root / "portable_brain_contract_latest.json")
        security_evidence = load_json_file(health_root / "security_evidence_autofix_latest.json")
        runtime_throttle = load_json_file(health_root / "runtime_throttle_control_latest.json")
        incident_closeout = load_json_file(health_root / "incident_closeout_autopilot_latest.json")
        live_canary = load_json_file(health_root / "live_canary_control_latest.json")
        return {
            "live_readiness_status": str(live_readiness.get("overall_status") or "unknown"),
            "live_readiness_score": float(live_readiness.get("readiness_score", 0.0) or 0.0),
            "runtime_status": str(runtime.get("overall_status") or "unknown"),
            "autonomy_status": str(autonomy.get("overall_status") or "unknown"),
            "autonomy_score": float(autonomy.get("autonomy_score", 0.0) or 0.0),
            "portability_status": str(portable.get("overall_status") or "unknown"),
            "broker_ready": bool(live_readiness.get("broker_ready", False)),
            "session_ready": bool(live_readiness.get("session_ready", False)),
            "artifact_contracts": {
                "live_readiness": _artifact_contract(health_root / "live_readiness_smoke_latest.json"),
                "runtime_separation": _artifact_contract(health_root / "live_runtime_separation_control_latest.json"),
                "autonomy_control_plane": _artifact_contract(health_root / "autonomy_control_plane_latest.json"),
                "portable_brain_contract": _artifact_contract(health_root / "portable_brain_contract_latest.json"),
                "security_evidence_autofix": _artifact_contract(health_root / "security_evidence_autofix_latest.json"),
                "runtime_throttle_control": _artifact_contract(health_root / "runtime_throttle_control_latest.json"),
                "incident_closeout_autopilot": _artifact_contract(health_root / "incident_closeout_autopilot_latest.json"),
                "live_canary_control": _artifact_contract(health_root / "live_canary_control_latest.json"),
            },
            "security_evidence_status": str(security_evidence.get("overall_status") or "unknown"),
            "runtime_throttle_status": str(runtime_throttle.get("overall_status") or "unknown"),
            "incident_closeout_status": str(incident_closeout.get("overall_status") or "unknown"),
            "staged_canary_preclearance": bool(live_canary.get("staged_preclearance_ready", False)),
            "preapproved_supervised_canary": bool(live_canary.get("preapproved_supervised_ready", False)),
        }

    def brokers_payload(self, *, runtime_config: BrokerRuntimeConfig, tenant: LicensingTenantContext) -> Dict[str, Any]:
        _ = tenant
        return {
            "configured": {
                "market_data_provider": runtime_config.market_data_provider_name,
                "paper_execution_broker": runtime_config.paper_execution_broker_name,
                "live_execution_broker": runtime_config.execution_broker_name,
                "auth_broker": runtime_config.auth_broker_name,
            },
            "available_brokers": list(available_broker_names()),
            "available_by_role": {
                "market_data": list(available_broker_names_for_role("market_data")),
                "paper_execution": list(available_broker_names_for_role("paper")),
                "live_execution": list(available_broker_names_for_role("live_execution")),
                "options": list(available_broker_names_for_role("options")),
                "news_context": list(available_broker_names_for_role("news_context")),
                "calendar_context": list(available_broker_names_for_role("calendar_context")),
            },
        }

    def runtime_payload(self, *, project_root: Path, tenant: LicensingTenantContext) -> Dict[str, Any]:
        _ = tenant
        health_root = project_root / "governance" / "health"
        portable = load_json_file(health_root / "portable_brain_contract_latest.json")
        runtime = load_json_file(health_root / "live_runtime_separation_control_latest.json")
        host_contract = portable.get("host_contract") if isinstance(portable.get("host_contract"), dict) else {}
        unified_memory = portable.get("unified_memory_telemetry") if isinstance(portable.get("unified_memory_telemetry"), dict) else {}
        return {
            "recommended_runtime_mode": str(portable.get("recommended_runtime_mode") or "unknown"),
            "recommended_backend": str(portable.get("recommended_backend") or "unknown"),
            "host_profile": str(host_contract.get("host_profile") or "unknown"),
            "chip": str(host_contract.get("chip") or "unknown"),
            "memory_architecture": str(host_contract.get("memory_architecture") or "unknown"),
            "shared_cpu_gpu_memory_pool": bool(host_contract.get("shared_cpu_gpu_memory_pool", False)),
            "memory_competitive_advantage": str(host_contract.get("memory_competitive_advantage") or ""),
            "runtime_separation_status": str(runtime.get("overall_status") or "unknown"),
            "unified_memory_telemetry": unified_memory,
        }

    def special_features_payload(self, *, project_root: Path, tenant: LicensingTenantContext) -> Dict[str, Any]:
        _ = tenant
        health_root = project_root / "governance" / "health"
        architecture = load_json_file(health_root / "architecture_upgrade_scoreboard_latest.json")
        portable = load_json_file(health_root / "portable_brain_contract_latest.json")
        host_contract = portable.get("host_contract") if isinstance(portable.get("host_contract"), dict) else {}
        unified_memory = portable.get("unified_memory_telemetry") if isinstance(portable.get("unified_memory_telemetry"), dict) else {}
        return {
            "feature_count": int(architecture.get("upgrade_count", 0) or 0),
            "ready_count": int(architecture.get("ready_count", 0) or 0),
            "special_features_map": dict(architecture.get("special_features_map") or {}),
            "host_profile": str(host_contract.get("host_profile") or "unknown"),
            "memory_architecture": str(host_contract.get("memory_architecture") or "unknown"),
            "unified_memory_telemetry": unified_memory,
        }

    def usage_payload(self, *, project_root: Path, tenant: LicensingTenantContext) -> Dict[str, Any]:
        health_root = project_root / "governance" / "health"
        incident = load_json_file(health_root / "incident_report_latest.json")
        cost_telemetry = load_json_file(health_root / "cost_telemetry_latest.json")
        return {
            "quota": dict(tenant.quota_snapshot),
            "usage": dict(tenant.usage_snapshot),
            "metering": dict(tenant.metering_snapshot),
            "tenant_metadata": dict(tenant.metadata),
            "incident_review_required": bool(incident.get("review_required", False)),
            "open_incident_count": int(((incident.get("incident_counts") or {}).get("open_incident_count") or 0)),
            "metering_ready": bool(((cost_telemetry.get("tenant_metering_contract") or {}).get("ready", False))),
        }

    def audit_payload(self, *, project_root: Path, tenant: LicensingTenantContext) -> Dict[str, Any]:
        health_root = project_root / "governance" / "health"
        incident = load_json_file(health_root / "incident_report_latest.json")
        recent_incidents = incident.get("recent_incidents") if isinstance(incident.get("recent_incidents"), list) else []
        return {
            "key_rotation": dict(tenant.key_rotation),
            "tenant_metadata": dict(tenant.metadata),
            "recent_incident_categories": [
                str((row or {}).get("category") or "").strip()
                for row in recent_incidents[:5]
                if isinstance(row, dict) and str((row or {}).get("category") or "").strip()
            ],
            "webhook_delivery": dict(tenant.webhook_delivery_snapshot),
            "response_signing_enabled": bool(tenant.response_signing_secret),
        }

    def billing_payload(self, *, project_root: Path, tenant: LicensingTenantContext) -> Dict[str, Any]:
        health_root = project_root / "governance" / "health"
        cost_telemetry = load_json_file(health_root / "cost_telemetry_latest.json")
        storage_cost = (cost_telemetry.get("storage_cost_proxy") or {}) if isinstance(cost_telemetry.get("storage_cost_proxy"), dict) else {}
        training_cost = (cost_telemetry.get("training_cost_proxy") or {}) if isinstance(cost_telemetry.get("training_cost_proxy"), dict) else {}
        return {
            "billing": dict(tenant.billing_snapshot),
            "estimated_cost_indices": {
                "storage": float(storage_cost.get("cost_index", 0.0) or 0.0),
                "training": float(training_cost.get("cost_index", 0.0) or 0.0),
            },
            "billable_dimensions": list(((cost_telemetry.get("tenant_metering_contract") or {}).get("billable_dimensions") or [])),
            "tenant_metadata": dict(tenant.metadata),
        }

    def metering_payload(self, *, project_root: Path, tenant: LicensingTenantContext) -> Dict[str, Any]:
        health_root = project_root / "governance" / "health"
        cost_telemetry = load_json_file(health_root / "cost_telemetry_latest.json")
        return {
            "metering": dict(tenant.metering_snapshot),
            "usage": dict(tenant.usage_snapshot),
            "quota": dict(tenant.quota_snapshot),
            "metering_contract": dict(cost_telemetry.get("tenant_metering_contract") or {}),
        }

    def webhook_events_payload(self, *, project_root: Path, tenant: LicensingTenantContext) -> Dict[str, Any]:
        health_root = project_root / "governance" / "health"
        incident = load_json_file(health_root / "incident_report_latest.json")
        live_canary = load_json_file(health_root / "live_canary_control_latest.json")
        promotion_packet = load_json_file(project_root / "governance" / "champion_challenger" / "promotion_packet_latest.json")
        return {
            "delivery": dict(tenant.webhook_delivery_snapshot),
            "events": [
                {
                    "event": "incident.review_required",
                    "ready": bool(incident.get("review_required", False)),
                },
                {
                    "event": "live.canary.preapproved",
                    "ready": bool(live_canary.get("preapproved_supervised_ready", False)),
                },
                {
                    "event": "promotion.packet_ready",
                    "ready": bool(promotion_packet.get("packet_complete", False)),
                },
            ],
            "tenant_metadata": dict(tenant.metadata),
        }

    def contracts_payload(
        self,
        *,
        project_root: Path,
        runtime_config: BrokerRuntimeConfig,
        tenant: LicensingTenantContext,
    ) -> Dict[str, Any]:
        health_root = project_root / "governance" / "health"
        incident = load_json_file(health_root / "incident_report_latest.json")
        training = load_json_file(health_root / "training_quality_control_latest.json")
        return {
            "health": self.health_payload(project_root=project_root, tenant=tenant),
            "brokers": self.brokers_payload(runtime_config=runtime_config, tenant=tenant),
            "runtime": self.runtime_payload(project_root=project_root, tenant=tenant),
            "special_features": self.special_features_payload(project_root=project_root, tenant=tenant),
            "usage": self.usage_payload(project_root=project_root, tenant=tenant),
            "billing": self.billing_payload(project_root=project_root, tenant=tenant),
            "metering": self.metering_payload(project_root=project_root, tenant=tenant),
            "webhook_events": self.webhook_events_payload(project_root=project_root, tenant=tenant),
            "audit": self.audit_payload(project_root=project_root, tenant=tenant),
            "capabilities": self.capabilities_payload(project_root=project_root, runtime_config=runtime_config, tenant=tenant),
            "schema": self.schema_payload(project_root=project_root, runtime_config=runtime_config, tenant=tenant),
            "webhook_contracts": self.webhook_contracts_payload(project_root=project_root, runtime_config=runtime_config, tenant=tenant),
            "grade_snapshot": self.grades_payload(project_root=project_root, runtime_config=runtime_config, tenant=tenant),
            "readiness_map": self.readiness_map_payload(project_root=project_root, runtime_config=runtime_config, tenant=tenant),
            "incident_contract": {
                "review_required": bool(incident.get("review_required", False)),
                "review_state": str(incident.get("review_state") or "unknown"),
                "closeout_contract": dict(incident.get("closeout_contract") or {}),
            },
            "lineage_contract": dict(training.get("immutable_lineage") or {}),
        }
