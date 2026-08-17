import gzip
import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.decision_context_mesh import (
    DECISION_CONTEXT_MESH_FEATURE_KEYS,
    PLANE_SIGNAL_FEATURE_KEYS,
    assess_decision_context_mesh,
    load_decision_context_mesh_config,
)
from core.runtime_training_common import _load_runtime_gap_fill_context
from scripts.collect_decision_context_mesh import (
    _build_plane,
    _cadence_freshness,
    _routed_source_path,
    parse_bts_freight_tsi,
    parse_eia_weekly_petroleum,
    parse_treasury_tic_history,
)
from scripts.portfolio_capacity_curve_report import _recent_paper_rows


def _valid_snapshot(now: datetime) -> dict:
    config = load_decision_context_mesh_config()
    planes = []
    for spec in config["planes"]:
        score = 94.0 if spec["plane_class"] == "macro" else 88.0
        planes.append(
            {
                "plane_id": spec["plane_id"],
                "plane_class": spec["plane_class"],
                "score_pct": score,
            }
        )
    contract = dict(config["contract"])
    return {
        "timestamp_utc": now.isoformat(),
        "contract": contract,
        "methodology": {
            "point_in_time_only": True,
            "future_observations_rejected": True,
            "missing_values_are_not_zero_filled": True,
        },
        "coverage": {
            "future_observation_selected": False,
            "future_observations_excluded": {},
        },
        "planes": planes,
        "derived": {
            "global_features": {key: 0.5 for key in DECISION_CONTEXT_MESH_FEATURE_KEYS},
            "symbol_features": {
                "AAPL": {"context_capacity_market_impact_signal_norm": 0.91}
            },
        },
    }


def test_config_declares_twelve_balanced_context_planes() -> None:
    config = load_decision_context_mesh_config()
    assert len(config["planes"]) == 12
    assert sum(1 for row in config["planes"] if row["plane_class"] == "macro") == 6
    assert sum(1 for row in config["planes"] if row["plane_class"] == "micro") == 6
    assert {row["signal_key"] for row in config["planes"]} == set(PLANE_SIGNAL_FEATURE_KEYS)
    assert config["contract"]["live_execution_authority"] is False
    assert config["contract"]["automatic_promotion_authority"] is False


def test_assessor_reports_organic_macro_and_micro_percentages() -> None:
    now = datetime.now(timezone.utc)
    assessment = assess_decision_context_mesh(_valid_snapshot(now), now_utc=now)
    assert assessment["ready"] is True
    assert assessment["macro_percentage"] == 94.0
    assert assessment["macro_grade"] == "A"
    assert assessment["micro_percentage"] == 88.0
    assert assessment["micro_grade"] == "B+"


def test_assessor_rejects_future_authority_and_missing_features() -> None:
    now = datetime.now(timezone.utc)
    snapshot = _valid_snapshot(now)
    snapshot["timestamp_utc"] = (now + timedelta(hours=2)).isoformat()
    snapshot["contract"]["live_execution_authority"] = True
    snapshot["derived"]["global_features"].pop(PLANE_SIGNAL_FEATURE_KEYS[0])
    assessment = assess_decision_context_mesh(snapshot, now_utc=now)
    assert assessment["ready"] is False
    assert "artifact_timestamp_in_future" in assessment["reasons"]
    assert "live_execution_authority_not_locked" in assessment["reasons"]
    assert "feature_schema_incomplete" in assessment["reasons"]


def test_official_source_parsers_select_only_point_in_time_rows() -> None:
    as_of = datetime(2026, 8, 15, tzinfo=timezone.utc)
    tic = parse_treasury_tic_history(
        "header\n"
        "2026-Sep\t" + "\t".join(["1"] * 32) + "\n"
        "2026-May\t" + "\t".join(["1"] * 29 + ["132177", "172044", "-39867"]) + "\n",
        as_of=as_of,
    )
    assert tic["observation_period"] == "2026-May"
    assert tic["total_monthly_inflows_usd_millions"] == 132177.0

    eia = parse_eia_weekly_petroleum(
        '"STUB_1","8/7/26","7/31/26","Difference"\n'
        '"Commercial (Excluding SPR)","424.410","406.987","17.422"\n'
        '"Total Motor Gasoline","208.690","209.658","-0.968"\n'
        '"Distillate Fuel Oil","112.300","112.310","-0.010"\n',
        as_of=as_of,
    )
    assert eia["inventories"]["commercial_crude"]["weekly_change_million_barrels"] == 17.422

    bts = parse_bts_freight_tsi(
        json.dumps(
            [
                {"obs_date": "2026-09-01T00:00:00.000", "tsi_freight": "999", "tsi_freight_c": "9"},
                {"obs_date": "2026-06-01T00:00:00.000", "tsi_freight": "134.9", "tsi_freight_c": "-0.3", "truck_d11": "113.3"},
            ]
        ),
        as_of=as_of,
    )
    assert bts["observation_date"] == "2026-06-01"
    assert bts["tsi_freight"] == 134.9


def test_compressed_paper_orders_feed_capacity_fallback(tmp_path: Path) -> None:
    paper_root = tmp_path / "exports" / "paper_broker_bridge" / "paper"
    paper_root.mkdir(parents=True)
    with gzip.open(paper_root / "paper_bridge_orders_20260815.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"symbol": "SPY", "tradeability_score": 0.8}) + "\n")
    rows = _recent_paper_rows(tmp_path)
    assert rows == [{"symbol": "SPY", "tradeability_score": 0.8}]


def test_runtime_gap_fill_routes_valid_global_and_symbol_mesh_features(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    snapshot = _valid_snapshot(now)
    path = tmp_path / "exports" / "external_context" / "decision_context_mesh_latest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(snapshot), encoding="utf-8")

    context = _load_runtime_gap_fill_context(tmp_path)
    assert context["external_global_features"]["context_mesh_macro_grade_norm"] == 0.5
    assert context["external_symbol_features"]["AAPL"]["context_capacity_market_impact_signal_norm"] == 0.91


def test_missing_mesh_signal_is_not_zero_filled() -> None:
    now = datetime.now(timezone.utc)
    snapshot = deepcopy(_valid_snapshot(now))
    snapshot["derived"]["global_features"].pop("context_supply_chain_inventory_signal_norm")
    assert "context_supply_chain_inventory_signal_norm" not in snapshot["derived"]["global_features"]
    assessment = assess_decision_context_mesh(snapshot, now_utc=now)
    assert assessment["ready"] is False
    assert assessment["missing_feature_keys"] == ["context_supply_chain_inventory_signal_norm"]


def test_cadence_freshness_awards_current_release_and_decays_before_hard_slo() -> None:
    assert _cadence_freshness(7.0, target_age=10.0, maximum_age=21.0) == 1.0
    assert 0.0 < _cadence_freshness(15.0, target_age=10.0, maximum_age=21.0) < 1.0
    assert _cadence_freshness(22.0, target_age=10.0, maximum_age=21.0) == 0.0


def test_estimate_plane_cap_lifts_only_with_direct_consensus_evidence() -> None:
    config = load_decision_context_mesh_config()
    spec = next(row for row in config["planes"] if row["plane_id"] == "estimates_dispersion")
    candidates = {
        f"estimate_feature_{index}": {
            "value": 0.5,
            "lineage": [
                {
                    "source_id": "sec_edgar_context" if index % 2 == 0 else "schwab_symbol_news",
                    "point_in_time_valid": True,
                }
            ],
        }
        for index in range(3)
    }
    source_states = {
        "sec_edgar_context": {"ok": True, "freshness_norm": 1.0, "source_family": "official_filings"},
        "schwab_symbol_news": {"ok": True, "freshness_norm": 1.0, "source_family": "broker_news"},
        "analyst_consensus_context": {
            "ok": True,
            "direct_consensus_ready": True,
            "freshness_norm": 1.0,
            "source_family": "configured_analyst_consensus_provider",
        },
    }
    row = _build_plane(
        spec,
        deepcopy(candidates),
        source_states,
        scoring=config["scoring"],
        minimum_score=70.0,
    )
    assert row["score_pct"] == 100.0
    assert row["direct_consensus_ready"] is True
    assert row["caveats"] == []

    source_states.pop("analyst_consensus_context")
    capped = _build_plane(
        spec,
        deepcopy(candidates),
        source_states,
        scoring=config["scoring"],
        minimum_score=70.0,
    )
    assert capped["score_pct"] == 87.0
    assert capped["direct_consensus_ready"] is False


def test_storage_router_prefers_newest_observation_including_conflict_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    external_root = tmp_path / "external"
    monkeypatch.setenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", str(external_root))
    relative = "exports/external_context/source_latest.json"
    local_path = tmp_path / relative
    external_path = external_root / relative
    fallback_path = external_path.with_name(f"{external_path.name}.local_fallback_conflict")
    for path, timestamp in (
        (local_path, "2026-08-15T10:00:00+00:00"),
        (external_path, "2026-08-15T11:00:00+00:00"),
        (fallback_path, "2026-08-15T12:00:00+00:00"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"timestamp_utc": timestamp, "ok": True}), encoding="utf-8")

    selected, route, count = _routed_source_path(tmp_path, relative)
    assert selected == fallback_path
    assert route == "external"
    assert count == 3


def test_storage_router_uses_canonical_as_equal_timestamp_tiebreaker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    external_root = tmp_path / "external"
    monkeypatch.setenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", str(external_root))
    relative = "exports/external_context/source_latest.json"
    canonical_path = external_root / relative
    fallback_path = canonical_path.with_name(f"{canonical_path.name}.local_fallback_conflict")
    for path in (canonical_path, fallback_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"timestamp_utc": "2026-08-15T12:00:00+00:00", "ok": True}),
            encoding="utf-8",
        )

    selected, route, count = _routed_source_path(tmp_path, relative)
    assert selected == canonical_path
    assert route == "external"
    assert count == 2


def test_storage_router_future_fallback_cannot_shadow_current_canonical(
    tmp_path: Path,
    monkeypatch,
) -> None:
    external_root = tmp_path / "external"
    monkeypatch.setenv("BOT_LOGS_EXTERNAL_PROJECT_ROOT", str(external_root))
    relative = "exports/external_context/source_latest.json"
    canonical_path = external_root / relative
    fallback_path = canonical_path.with_name(f"{canonical_path.name}.local_fallback_conflict")
    now = datetime.now(timezone.utc)
    for path, timestamp in (
        (canonical_path, now - timedelta(hours=1)),
        (fallback_path, now + timedelta(days=1)),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"timestamp_utc": timestamp.isoformat(), "ok": True}),
            encoding="utf-8",
        )

    selected, route, count = _routed_source_path(tmp_path, relative)
    assert selected == canonical_path
    assert route == "external"
    assert count == 2
