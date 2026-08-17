from __future__ import annotations

from collections.abc import Mapping, Sequence


_PRESSURE_AUTHORITATIVE_PREFIXES = (
    "LOG_",
    "DECISION_LOG_",
    "DECISION_EXPLANATION_",
    "SIGNAL_GENERATION_",
)
_PRESSURE_AUTHORITATIVE_KEYS = {
    "CHANNEL_LOG_PRIMARY_MODE",
    "LEGACY_HOT_CHANNEL_MIRROR_ENABLED",
    "MASTER_CONTROL_INFRA_ROWS_MAX",
}


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _pressure_authority_active(pressure: Mapping[str, str]) -> bool:
    profile = str(pressure.get("BOT_INGESTION_STORAGE_PROFILE") or "").strip().lower()
    return bool(
        profile in {"critical_backpressure", "elevated_backpressure"}
        or _truthy(pressure.get("RAW_LIVE_EXPANSION_GUARD_ACTIVE"))
        or _truthy(pressure.get("BACKLOG_RELIEF_CONTRACT_ACTIVE"))
        or _truthy(pressure.get("TRAINING_RUNTIME_PAUSED_FOR_BACKLOG"))
        or _truthy(pressure.get("HEAVY_COLLECTORS_PAUSED_FOR_BACKLOG"))
        or _truthy(pressure.get("SHADOW_LOOP_PAUSED_FOR_BACKLOG"))
    )


def _pressure_authoritative_key(key: str) -> bool:
    normalized = str(key or "").strip().upper()
    return bool(
        normalized in _PRESSURE_AUTHORITATIVE_KEYS
        or normalized.startswith(_PRESSURE_AUTHORITATIVE_PREFIXES)
    )


def merge_runtime_override_layers(
    layers: Sequence[Mapping[str, str]],
    *,
    pressure_layer_index: int = 0,
) -> dict[str, str]:
    """Merge runtime controls while keeping active queue safety authoritative."""
    normalized_layers = [
        {str(key): str(value) for key, value in layer.items() if str(key).strip()}
        for layer in layers
    ]
    merged: dict[str, str] = {}
    for layer in normalized_layers:
        merged.update(layer)
    if not normalized_layers or not (0 <= pressure_layer_index < len(normalized_layers)):
        return merged

    pressure = normalized_layers[pressure_layer_index]
    if not _pressure_authority_active(pressure):
        return merged
    for key, value in pressure.items():
        if _pressure_authoritative_key(key):
            merged[key] = value
    return merged
