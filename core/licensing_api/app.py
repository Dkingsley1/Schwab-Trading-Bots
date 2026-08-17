from __future__ import annotations

import hashlib
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Tuple

from fastapi import Depends, FastAPI, Header, HTTPException

from core.brokers import BrokerRuntimeConfig
from core.licensing_api.models import LicensingTenantContext
from core.licensing_api.registry import available_connector_names, build_connector, normalize_connector_name


def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default) or default).strip().lower() in {"1", "true", "yes", "on"}


def _parse_assignment_map(raw: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for chunk in str(raw or "").replace(";", ",").split(","):
        text = str(chunk or "").strip()
        if not text or "=" not in text:
            continue
        key, value = text.split("=", 1)
        key_text = str(key or "").strip()
        value_text = str(value or "").strip()
        if key_text and value_text:
            out[key_text] = value_text
    return out


def _normalize_endpoint_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _parse_pipe_list(raw: str) -> Tuple[str, ...]:
    out = []
    seen = set()
    for chunk in str(raw or "").replace(",", "|").split("|"):
        token = _normalize_endpoint_token(chunk)
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return tuple(out)


def _parse_assignment_list_map(raw: str) -> Dict[str, Tuple[str, ...]]:
    return {key: _parse_pipe_list(value) for key, value in _parse_assignment_map(raw).items()}


def _parse_assignment_metadata_map(raw: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for tenant_id, raw_value in _parse_assignment_map(raw).items():
        row: Dict[str, str] = {}
        for chunk in str(raw_value or "").split("|"):
            text = str(chunk or "").strip()
            if not text:
                continue
            if ":" in text:
                key, value = text.split(":", 1)
            elif "=" in text:
                key, value = text.split("=", 1)
            else:
                continue
            key_text = str(key or "").strip()
            value_text = str(value or "").strip()
            if key_text and value_text:
                row[key_text] = value_text
        out[tenant_id] = row
    return out


def _metadata_to_numbers(payload: Dict[str, str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in payload.items():
        text = str(value or "").strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in {"true", "false"}:
            out[key] = lowered == "true"
            continue
        try:
            if "." in text:
                out[key] = float(text)
            else:
                out[key] = int(text)
            continue
        except Exception:
            out[key] = text
    return out


def _api_key_matches(candidate_key: str, configured_value: str) -> bool:
    candidate = str(candidate_key or "")
    configured = str(configured_value or "").strip()
    if not configured:
        return False
    if configured.lower().startswith("sha256:"):
        digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
        return digest == configured.split(":", 1)[1].strip().lower()
    return candidate == configured


def _tenant_context_dependency(project_root: Path, endpoint_slug: str):
    resolved_root = Path(project_root).expanduser().resolve()
    normalized_endpoint = _normalize_endpoint_token(endpoint_slug)

    def _resolve(
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
        x_api_key_id: str | None = Header(default=None, alias="X-API-Key-Id"),
        x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
        x_request_id: str | None = Header(default=None, alias="X-Request-Id"),
    ) -> LicensingTenantContext:
        _ = resolved_root
        tenant_keys = _parse_assignment_map(os.getenv("LICENSING_API_KEYS", ""))
        tenant_connectors = _parse_assignment_map(os.getenv("LICENSING_API_TENANT_CONNECTORS", ""))
        tenant_companies = _parse_assignment_map(os.getenv("LICENSING_API_TENANT_COMPANIES", ""))
        tenant_allowed_endpoints = _parse_assignment_list_map(os.getenv("LICENSING_API_TENANT_ALLOWED_ENDPOINTS", ""))
        tenant_permissions = _parse_assignment_list_map(os.getenv("LICENSING_API_TENANT_PERMISSIONS", ""))
        tenant_key_ids = _parse_assignment_map(os.getenv("LICENSING_API_KEY_IDS", ""))
        tenant_signing_keys = _parse_assignment_map(os.getenv("LICENSING_API_TENANT_SIGNING_KEYS", ""))
        tenant_signing_key_ids = _parse_assignment_map(os.getenv("LICENSING_API_TENANT_SIGNING_KEY_IDS", ""))
        tenant_quotas = _parse_assignment_metadata_map(os.getenv("LICENSING_API_TENANT_QUOTAS", ""))
        tenant_usage = _parse_assignment_metadata_map(os.getenv("LICENSING_API_TENANT_USAGE", ""))
        tenant_billing = _parse_assignment_metadata_map(os.getenv("LICENSING_API_TENANT_BILLING", ""))
        tenant_metering = _parse_assignment_metadata_map(os.getenv("LICENSING_API_TENANT_METERING", ""))
        tenant_webhook_delivery = _parse_assignment_metadata_map(os.getenv("LICENSING_API_TENANT_WEBHOOKS", ""))
        tenant_key_rotation = _parse_assignment_metadata_map(os.getenv("LICENSING_API_TENANT_KEY_ROTATION", ""))
        tenant_metadata = _parse_assignment_metadata_map(os.getenv("LICENSING_API_TENANT_METADATA", ""))
        require_key = bool(tenant_keys) or _env_flag("LICENSING_API_REQUIRE_KEY", "0")

        tenant_id = str(x_tenant_id or "").strip()
        if require_key:
            if not x_api_key:
                raise HTTPException(status_code=401, detail="missing_api_key")
            matched_tenant = ""
            for candidate_tenant, candidate_key in tenant_keys.items():
                if _api_key_matches(str(x_api_key), str(candidate_key)):
                    matched_tenant = candidate_tenant
                    break
            if not matched_tenant:
                raise HTTPException(status_code=403, detail="invalid_api_key")
            if tenant_id and tenant_id != matched_tenant:
                raise HTTPException(status_code=403, detail="tenant_key_mismatch")
            tenant_id = matched_tenant

        if not tenant_id:
            tenant_id = os.getenv("LICENSING_API_DEFAULT_TENANT", "local-dev").strip() or "local-dev"

        connector_name = normalize_connector_name(
            tenant_connectors.get(tenant_id, os.getenv("LICENSING_API_CONNECTOR", "default"))
        )
        company_name = tenant_companies.get(tenant_id, os.getenv("LICENSING_API_COMPANY_NAME", tenant_id))
        allowed_endpoints = tenant_allowed_endpoints.get(tenant_id, tuple())
        if allowed_endpoints and ("all" not in allowed_endpoints) and ("*" not in allowed_endpoints):
            if normalized_endpoint not in allowed_endpoints:
                raise HTTPException(status_code=403, detail="endpoint_not_allowed")
        request_id = str(x_request_id or "").strip() or f"lic-{uuid.uuid4().hex[:12]}"
        return LicensingTenantContext(
            tenant_id=tenant_id,
            company_name=str(company_name or "").strip(),
            connector_name=connector_name,
            api_key_id=str(x_api_key_id or tenant_key_ids.get(tenant_id, tenant_id)).strip(),
            allowed_endpoints=allowed_endpoints,
            permissions=tenant_permissions.get(tenant_id, tuple()),
            request_id=request_id,
            response_signing_key_id=str(tenant_signing_key_ids.get(tenant_id, tenant_id)).strip(),
            response_signing_secret=str(tenant_signing_keys.get(tenant_id, "")).strip(),
            quota_snapshot=_metadata_to_numbers(tenant_quotas.get(tenant_id, {})),
            usage_snapshot=_metadata_to_numbers(tenant_usage.get(tenant_id, {})),
            billing_snapshot=_metadata_to_numbers(tenant_billing.get(tenant_id, {})),
            metering_snapshot=_metadata_to_numbers(tenant_metering.get(tenant_id, {})),
            webhook_delivery_snapshot=_metadata_to_numbers(tenant_webhook_delivery.get(tenant_id, {})),
            key_rotation=_metadata_to_numbers(tenant_key_rotation.get(tenant_id, {})),
            metadata=dict(tenant_metadata.get(tenant_id, {})),
        )

    return _resolve


def build_partner_api(*, project_root: str | Path | None = None) -> FastAPI:
    resolved_root = Path(project_root or Path(__file__).resolve().parents[2]).expanduser().resolve()
    app = FastAPI(
        title="Licensed Partner API",
        version="0.5.0",
        description="Tenant-aware licensing surface with contracts, quotas, billing, metering, webhook delivery, and audit metadata for external partners.",
    )

    @app.get("/healthz")
    def healthz() -> Dict[str, object]:
        return {
            "ok": True,
            "project_root": str(resolved_root),
            "available_connectors": list(available_connector_names()),
        }

    def _wrapped_response(endpoint: str, tenant: LicensingTenantContext, payload: Dict[str, object]) -> Dict[str, object]:
        connector = build_connector(tenant.connector_name)
        return connector.build_response_envelope(
            endpoint=endpoint,
            tenant=tenant,
            data=payload,
        )

    @app.get("/v1/license/tenant")
    def tenant_profile(
        tenant: LicensingTenantContext = Depends(_tenant_context_dependency(resolved_root, "tenant")),
    ) -> Dict[str, object]:
        connector = build_connector(tenant.connector_name)
        return _wrapped_response(
            "tenant",
            tenant,
            {
                "tenant": tenant.to_dict(),
                "active_connector": connector.describe(),
                "available_connectors": list(available_connector_names()),
            },
        )

    @app.get("/v1/license/connectors")
    def connectors(
        tenant: LicensingTenantContext = Depends(_tenant_context_dependency(resolved_root, "connectors")),
    ) -> Dict[str, object]:
        active = build_connector(tenant.connector_name)
        return _wrapped_response(
            "connectors",
            tenant,
            {
                "tenant": tenant.to_dict(),
                "active_connector": active.describe(),
                "available_connectors": list(available_connector_names()),
            },
        )

    @app.get("/v1/license/overview")
    def overview(
        tenant: LicensingTenantContext = Depends(_tenant_context_dependency(resolved_root, "overview")),
    ) -> Dict[str, object]:
        connector = build_connector(tenant.connector_name)
        runtime_config = BrokerRuntimeConfig.from_env()
        return _wrapped_response(
            "overview",
            tenant,
            connector.overview_payload(
                project_root=resolved_root,
                runtime_config=runtime_config,
                tenant=tenant,
            ),
        )

    @app.get("/v1/license/health")
    def health(
        tenant: LicensingTenantContext = Depends(_tenant_context_dependency(resolved_root, "health")),
    ) -> Dict[str, object]:
        connector = build_connector(tenant.connector_name)
        return _wrapped_response(
            "health",
            tenant,
            connector.health_payload(project_root=resolved_root, tenant=tenant),
        )

    @app.get("/v1/license/brokers")
    def brokers(
        tenant: LicensingTenantContext = Depends(_tenant_context_dependency(resolved_root, "brokers")),
    ) -> Dict[str, object]:
        connector = build_connector(tenant.connector_name)
        runtime_config = BrokerRuntimeConfig.from_env()
        return _wrapped_response(
            "brokers",
            tenant,
            connector.brokers_payload(runtime_config=runtime_config, tenant=tenant),
        )

    @app.get("/v1/license/runtime")
    def runtime(
        tenant: LicensingTenantContext = Depends(_tenant_context_dependency(resolved_root, "runtime")),
    ) -> Dict[str, object]:
        connector = build_connector(tenant.connector_name)
        return _wrapped_response(
            "runtime",
            tenant,
            connector.runtime_payload(project_root=resolved_root, tenant=tenant),
        )

    @app.get("/v1/license/special-features")
    def special_features(
        tenant: LicensingTenantContext = Depends(_tenant_context_dependency(resolved_root, "special_features")),
    ) -> Dict[str, object]:
        connector = build_connector(tenant.connector_name)
        return _wrapped_response(
            "special_features",
            tenant,
            connector.special_features_payload(project_root=resolved_root, tenant=tenant),
        )

    @app.get("/v1/license/usage")
    def usage(
        tenant: LicensingTenantContext = Depends(_tenant_context_dependency(resolved_root, "usage")),
    ) -> Dict[str, object]:
        connector = build_connector(tenant.connector_name)
        return _wrapped_response(
            "usage",
            tenant,
            connector.usage_payload(project_root=resolved_root, tenant=tenant),
        )

    @app.get("/v1/license/billing")
    def billing(
        tenant: LicensingTenantContext = Depends(_tenant_context_dependency(resolved_root, "billing")),
    ) -> Dict[str, object]:
        connector = build_connector(tenant.connector_name)
        return _wrapped_response(
            "billing",
            tenant,
            connector.billing_payload(project_root=resolved_root, tenant=tenant),
        )

    @app.get("/v1/license/metering")
    def metering(
        tenant: LicensingTenantContext = Depends(_tenant_context_dependency(resolved_root, "metering")),
    ) -> Dict[str, object]:
        connector = build_connector(tenant.connector_name)
        return _wrapped_response(
            "metering",
            tenant,
            connector.metering_payload(project_root=resolved_root, tenant=tenant),
        )

    @app.get("/v1/license/webhook-events")
    def webhook_events(
        tenant: LicensingTenantContext = Depends(_tenant_context_dependency(resolved_root, "webhook_events")),
    ) -> Dict[str, object]:
        connector = build_connector(tenant.connector_name)
        runtime_config = BrokerRuntimeConfig.from_env()
        _ = runtime_config
        return _wrapped_response(
            "webhook_events",
            tenant,
            connector.webhook_events_payload(project_root=resolved_root, tenant=tenant),
        )

    @app.get("/v1/license/audit")
    def audit(
        tenant: LicensingTenantContext = Depends(_tenant_context_dependency(resolved_root, "audit")),
    ) -> Dict[str, object]:
        connector = build_connector(tenant.connector_name)
        return _wrapped_response(
            "audit",
            tenant,
            connector.audit_payload(project_root=resolved_root, tenant=tenant),
        )

    @app.get("/v1/license/contracts")
    def contracts(
        tenant: LicensingTenantContext = Depends(_tenant_context_dependency(resolved_root, "contracts")),
    ) -> Dict[str, object]:
        connector = build_connector(tenant.connector_name)
        runtime_config = BrokerRuntimeConfig.from_env()
        return _wrapped_response(
            "contracts",
            tenant,
            connector.contracts_payload(project_root=resolved_root, runtime_config=runtime_config, tenant=tenant),
        )

    @app.get("/v1/license/capabilities")
    def capabilities(
        tenant: LicensingTenantContext = Depends(_tenant_context_dependency(resolved_root, "capabilities")),
    ) -> Dict[str, object]:
        connector = build_connector(tenant.connector_name)
        runtime_config = BrokerRuntimeConfig.from_env()
        return _wrapped_response(
            "capabilities",
            tenant,
            connector.capabilities_payload(project_root=resolved_root, runtime_config=runtime_config, tenant=tenant),
        )

    @app.get("/v1/license/schema")
    def schema(
        tenant: LicensingTenantContext = Depends(_tenant_context_dependency(resolved_root, "schema")),
    ) -> Dict[str, object]:
        connector = build_connector(tenant.connector_name)
        runtime_config = BrokerRuntimeConfig.from_env()
        return _wrapped_response(
            "schema",
            tenant,
            connector.schema_payload(project_root=resolved_root, runtime_config=runtime_config, tenant=tenant),
        )

    @app.get("/v1/license/webhook-contracts")
    def webhook_contracts(
        tenant: LicensingTenantContext = Depends(_tenant_context_dependency(resolved_root, "webhook_contracts")),
    ) -> Dict[str, object]:
        connector = build_connector(tenant.connector_name)
        runtime_config = BrokerRuntimeConfig.from_env()
        return _wrapped_response(
            "webhook_contracts",
            tenant,
            connector.webhook_contracts_payload(project_root=resolved_root, runtime_config=runtime_config, tenant=tenant),
        )

    @app.get("/v1/license/grades")
    def grades(
        tenant: LicensingTenantContext = Depends(_tenant_context_dependency(resolved_root, "grades")),
    ) -> Dict[str, object]:
        connector = build_connector(tenant.connector_name)
        runtime_config = BrokerRuntimeConfig.from_env()
        return _wrapped_response(
            "grades",
            tenant,
            connector.grades_payload(project_root=resolved_root, runtime_config=runtime_config, tenant=tenant),
        )

    @app.get("/v1/license/readiness-map")
    def readiness_map(
        tenant: LicensingTenantContext = Depends(_tenant_context_dependency(resolved_root, "readiness_map")),
    ) -> Dict[str, object]:
        connector = build_connector(tenant.connector_name)
        runtime_config = BrokerRuntimeConfig.from_env()
        return _wrapped_response(
            "readiness_map",
            tenant,
            connector.readiness_map_payload(project_root=resolved_root, runtime_config=runtime_config, tenant=tenant),
        )

    return app
