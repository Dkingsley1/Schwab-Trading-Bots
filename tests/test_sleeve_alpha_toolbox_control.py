from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.ops.sleeve_alpha_toolbox_control import build_payload, render_markdown

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_toolbox_routes_every_declared_sleeve_without_fallback_authority() -> None:
    sleeve_registry = json.loads(
        (PROJECT_ROOT / "config" / "sleeve_strategy_expansion.json").read_text(
            encoding="utf-8"
        )
    )

    payload = build_payload(
        PROJECT_ROOT,
        generated_at_utc="2026-08-24T15:00:00+00:00",
    )
    markdown = render_markdown(payload)

    declared_count = len(sleeve_registry["sleeves"])
    assert declared_count > 100
    assert payload["ok"] is True
    assert payload["coverage"]["declared_sleeve_count"] == declared_count
    assert payload["coverage"]["routed_sleeve_count"] == declared_count
    assert payload["coverage"]["missing_sleeve_count"] == 0
    assert (
        payload["coverage"]["policy_match_source_counts"].get("default_fallback", 0)
        == 0
    )
    assert all(row["required_axis_count"] > 0 for row in payload["sleeve_routes"])
    assert all(row["route_receipt_sha256"] for row in payload["sleeve_routes"])
    routed_engines = {
        engine
        for row in payload["sleeve_routes"]
        for engine in row["routed_measurement_engines"]
    }
    assert "split_conformal_residual_calibration" in routed_engines
    assert "sequential_change_point_stability" in routed_engines
    assert "residual_redundancy_graph" in routed_engines
    assert "regime_conditional_robustness" in routed_engines
    assert "cost_stress_survival" in routed_engines
    assert not any(payload["authority"].values())
    assert "cannot change signals" in markdown


def test_toolbox_rejects_any_trading_authority(
    tmp_path: Path,
) -> None:
    source = json.loads(
        (PROJECT_ROOT / "config" / "sleeve_alpha_toolbox_v1.json").read_text(
            encoding="utf-8"
        )
    )
    unsafe = deepcopy(source)
    unsafe["authority"]["submits_live_order"] = True
    config_path = tmp_path / "unsafe_toolbox.json"
    config_path.write_text(json.dumps(unsafe), encoding="utf-8")

    with pytest.raises(ValueError, match="forbidden trading or promotion authority"):
        build_payload(PROJECT_ROOT, config_path=config_path)
