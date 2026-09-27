"""Corroborate allocation counters against fresh resident-pressure evidence."""

from datetime import datetime, timezone
import math


def allocation_only_memory_evidence(resource_guard: dict, swap_governor: dict) -> dict:
    result = {"ready": False, "reason": "memory_evidence_missing_or_invalid"}
    now = datetime.now(timezone.utc)

    def number(mapping, key):
        value = mapping[key]
        if isinstance(value, bool):
            raise ValueError("boolean_measurement")
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("invalid_measurement")
        return value

    try:
        ages = []
        for payload in (resource_guard, swap_governor):
            for key in ("timestamp_utc", "source_timestamp_utc"):
                if key == "source_timestamp_utc" and key not in payload:
                    continue
                stamp = datetime.fromisoformat(payload[key].replace("Z", "+00:00"))
                if stamp.tzinfo is None:
                    raise ValueError("timezone_required")
                ages.append((now - stamp).total_seconds())
        if not all(0 <= age <= 120 for age in ages):
            return {"ready": False, "reason": "memory_evidence_stale_or_future"}
        swap = swap_governor["swap_pressure"]
        observed = swap["stale_swap_allocation_relief"]
        raw_swap = number(resource_guard, "swap_used_gb")
        current_swap = number(swap, "swap_used_gb")
        calm_ceiling = number(swap["thresholds"], "calm_swap_gb")
        available = [number(resource_guard, "memory_free_pct"), number(observed, "memory_free_pct")]
        free = min(available)
        compressor = max(number(resource_guard, "compressor_gb"), number(observed, "compressor_gb"))
        throttled = max(number(resource_guard, "pages_throttled"), number(observed, "pages_throttled"))
        disk = number(resource_guard, "local_disk_free_gb")
        ready = bool(
            all(payload.get("memory_pressure_state") == "green" for payload in (resource_guard, swap))
            and all(payload.get("memory_pressure_kind") in {"none", "normal"} for payload in (resource_guard, swap))
            and swap.get("tier") == swap.get("raw_tier") == "normal"
            and abs(raw_swap - current_swap) <= 0.5
            and max(raw_swap, current_swap) < min(calm_ceiling, 20.0)
            and all(85 <= value <= 100 for value in available)
            and compressor <= 1.0 and throttled == 0 and disk >= 32
        )
        result.update(
            ready=ready, reason="fresh_normal_pressure_with_small_resident_compressor" if ready else "resident_pressure_or_swap_policy_not_clear",
            resource_timestamp_utc=resource_guard["timestamp_utc"], swap_timestamp_utc=swap_governor["timestamp_utc"],
            maximum_source_age_seconds=round(max(ages), 3), swap_used_gb=raw_swap,
            normal_swap_ceiling_gb=calm_ceiling, memory_available_pct=free,
            resident_compressor_gb=compressor, pages_throttled=throttled, local_disk_free_gb=disk,
            policy="allocation counters remain visible; this is not training, promotion, or execution authority",
        )
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
        pass
    return result
