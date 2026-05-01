from __future__ import annotations

from typing import Dict, Tuple, Type

from core.licensing_api.base import LicensingAPIConnector
from core.licensing_api.default_connector import DefaultLicensingAPIConnector


_CONNECTORS: Dict[str, Type[LicensingAPIConnector]] = {
    "default": DefaultLicensingAPIConnector,
}


def normalize_connector_name(name: str) -> str:
    return str(name or "").strip().lower() or "default"


def available_connector_names() -> Tuple[str, ...]:
    return tuple(sorted(_CONNECTORS.keys()))


def build_connector(name: str) -> LicensingAPIConnector:
    connector_name = normalize_connector_name(name)
    connector_cls = _CONNECTORS.get(connector_name)
    if connector_cls is None:
        supported = ",".join(sorted(_CONNECTORS.keys()))
        raise ValueError(f"unsupported_licensing_connector:{connector_name}:supported={supported}")
    return connector_cls()
