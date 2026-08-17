from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from core.brokers import BrokerRuntimeConfig
from core.licensing_api.models import LicensingTenantContext


class LicensingAPIConnector:
    name = "base"
    display_name = "Base Licensing API Connector"
    api_version = "v1"
    response_envelope_version = 1
    exposed_endpoints = ()

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "read_only": True,
            "tenant_aware": True,
            "api_version": self.api_version,
            "response_envelope_version": self.response_envelope_version,
            "exposed_endpoints": list(self.exposed_endpoints),
        }

    def build_response_envelope(
        self,
        *,
        endpoint: str,
        tenant: LicensingTenantContext,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        out = {
            "api_version": self.api_version,
            "response_envelope_version": self.response_envelope_version,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "endpoint": str(endpoint or "").strip().lower(),
            "request_id": str(tenant.request_id or "").strip(),
            "tenant": tenant.to_dict(),
            "connector": self.describe(),
            "quota": dict(tenant.quota_snapshot),
            "usage": dict(tenant.usage_snapshot),
            "billing": dict(tenant.billing_snapshot),
            "metering": dict(tenant.metering_snapshot),
            "webhook_delivery": dict(tenant.webhook_delivery_snapshot),
            "key_rotation": dict(tenant.key_rotation),
            "data": dict(data or {}),
        }
        secret = str(tenant.response_signing_secret or "").strip()
        if secret:
            signed_payload = json.dumps(
                {
                    "api_version": out["api_version"],
                    "endpoint": out["endpoint"],
                    "request_id": out["request_id"],
                    "tenant_id": tenant.tenant_id,
                    "data": out["data"],
                },
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            digest = hmac.new(secret.encode("utf-8"), signed_payload, hashlib.sha256).hexdigest()
            out["signature"] = {
                "algorithm": "hmac-sha256",
                "key_id": str(tenant.response_signing_key_id or tenant.tenant_id or "").strip(),
                "digest": digest,
            }
        return out

    def overview_payload(
        self,
        *,
        project_root: Path,
        runtime_config: BrokerRuntimeConfig,
        tenant: LicensingTenantContext,
    ) -> Dict[str, Any]:
        return {
            "health": self.health_payload(project_root=project_root, tenant=tenant),
            "brokers": self.brokers_payload(runtime_config=runtime_config, tenant=tenant),
            "runtime": self.runtime_payload(project_root=project_root, tenant=tenant),
            "special_features": self.special_features_payload(project_root=project_root, tenant=tenant),
            "grades": self.grades_payload(project_root=project_root, runtime_config=runtime_config, tenant=tenant),
            "readiness_map": self.readiness_map_payload(project_root=project_root, runtime_config=runtime_config, tenant=tenant),
        }

    def health_payload(self, *, project_root: Path, tenant: LicensingTenantContext) -> Dict[str, Any]:
        raise NotImplementedError

    def brokers_payload(self, *, runtime_config: BrokerRuntimeConfig, tenant: LicensingTenantContext) -> Dict[str, Any]:
        raise NotImplementedError

    def runtime_payload(self, *, project_root: Path, tenant: LicensingTenantContext) -> Dict[str, Any]:
        raise NotImplementedError

    def special_features_payload(self, *, project_root: Path, tenant: LicensingTenantContext) -> Dict[str, Any]:
        raise NotImplementedError

    def usage_payload(self, *, project_root: Path, tenant: LicensingTenantContext) -> Dict[str, Any]:
        return {
            "quota": dict(tenant.quota_snapshot),
            "usage": dict(tenant.usage_snapshot),
        }

    def audit_payload(self, *, project_root: Path, tenant: LicensingTenantContext) -> Dict[str, Any]:
        return {
            "key_rotation": dict(tenant.key_rotation),
            "tenant_metadata": dict(tenant.metadata),
        }

    def billing_payload(self, *, project_root: Path, tenant: LicensingTenantContext) -> Dict[str, Any]:
        _ = project_root
        return {"billing": dict(tenant.billing_snapshot)}

    def metering_payload(self, *, project_root: Path, tenant: LicensingTenantContext) -> Dict[str, Any]:
        _ = project_root
        return {
            "metering": dict(tenant.metering_snapshot),
            "usage": dict(tenant.usage_snapshot),
            "quota": dict(tenant.quota_snapshot),
        }

    def webhook_events_payload(self, *, project_root: Path, tenant: LicensingTenantContext) -> Dict[str, Any]:
        _ = project_root
        return {
            "delivery": dict(tenant.webhook_delivery_snapshot),
            "tenant_metadata": dict(tenant.metadata),
        }

    def contracts_payload(
        self,
        *,
        project_root: Path,
        runtime_config: BrokerRuntimeConfig,
        tenant: LicensingTenantContext,
    ) -> Dict[str, Any]:
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
        }

    def capabilities_payload(
        self,
        *,
        project_root: Path,
        runtime_config: BrokerRuntimeConfig,
        tenant: LicensingTenantContext,
    ) -> Dict[str, Any]:
        _ = (project_root, runtime_config, tenant)
        return {}

    def schema_payload(
        self,
        *,
        project_root: Path,
        runtime_config: BrokerRuntimeConfig,
        tenant: LicensingTenantContext,
    ) -> Dict[str, Any]:
        _ = (project_root, runtime_config, tenant)
        return {}

    def webhook_contracts_payload(
        self,
        *,
        project_root: Path,
        runtime_config: BrokerRuntimeConfig,
        tenant: LicensingTenantContext,
    ) -> Dict[str, Any]:
        _ = (project_root, runtime_config, tenant)
        return {}

    def grades_payload(
        self,
        *,
        project_root: Path,
        runtime_config: BrokerRuntimeConfig,
        tenant: LicensingTenantContext,
    ) -> Dict[str, Any]:
        _ = (project_root, runtime_config, tenant)
        return {}

    def readiness_map_payload(
        self,
        *,
        project_root: Path,
        runtime_config: BrokerRuntimeConfig,
        tenant: LicensingTenantContext,
    ) -> Dict[str, Any]:
        _ = (project_root, runtime_config, tenant)
        return {}


def load_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}
