"""Helpers for registry-backed bot modules that are still collecting data.

These modules make planned roster-expansion bots visible in PyCharm while the
registry remains the source of truth for activation, training gates, and trading
permissions.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def describe_registry_backed_bot(spec: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the bot spec for dashboards and quick inspection."""
    return deepcopy(spec)


def train_registry_backed_bot(spec: dict[str, Any]) -> dict[str, Any]:
    """Keep collection-first bots out of training until registry gates graduate them."""
    payload = describe_registry_backed_bot(spec)
    payload["trainable_now"] = False
    payload["training_status"] = "blocked_until_minimum_data_collection_threshold_met"
    payload["training_threshold_policy"] = spec.get("training_threshold_policy", "eligible_when_minimum_observations_and_days_met")
    return payload
