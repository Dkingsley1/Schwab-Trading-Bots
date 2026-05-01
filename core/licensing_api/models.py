from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class LicensingTenantContext:
    tenant_id: str
    company_name: str = ""
    connector_name: str = "default"
    api_key_id: str = ""
    allowed_endpoints: Tuple[str, ...] = ()
    permissions: Tuple[str, ...] = ()
    request_id: str = ""
    response_signing_key_id: str = ""
    response_signing_secret: str = field(default="", repr=False)
    quota_snapshot: Dict[str, Any] = field(default_factory=dict)
    usage_snapshot: Dict[str, Any] = field(default_factory=dict)
    billing_snapshot: Dict[str, Any] = field(default_factory=dict)
    metering_snapshot: Dict[str, Any] = field(default_factory=dict)
    webhook_delivery_snapshot: Dict[str, Any] = field(default_factory=dict)
    key_rotation: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": str(self.tenant_id or "").strip(),
            "company_name": str(self.company_name or "").strip(),
            "connector_name": str(self.connector_name or "default").strip(),
            "api_key_id": str(self.api_key_id or "").strip(),
            "allowed_endpoints": list(self.allowed_endpoints),
            "permissions": list(self.permissions),
            "request_id": str(self.request_id or "").strip(),
            "response_signing_key_id": str(self.response_signing_key_id or "").strip(),
            "quota_snapshot": dict(self.quota_snapshot),
            "usage_snapshot": dict(self.usage_snapshot),
            "billing_snapshot": dict(self.billing_snapshot),
            "metering_snapshot": dict(self.metering_snapshot),
            "webhook_delivery_snapshot": dict(self.webhook_delivery_snapshot),
            "key_rotation": dict(self.key_rotation),
            "metadata": dict(self.metadata),
        }
