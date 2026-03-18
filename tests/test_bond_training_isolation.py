import json
from datetime import datetime, timezone
from pathlib import Path

import scripts.data_source_divergence_bot as divergence_bot
import scripts.run_shadow_training_loop as loop
import scripts.weekly_retrain as weekly_retrain


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_divergence_payloads_split_bond_and_non_bond_scopes(tmp_path) -> None:
    ts = datetime.now(timezone.utc).replace(second=0, microsecond=0).isoformat()
    bond_file = tmp_path / "governance" / "shadow_bond_equities" / "master_control_20260313.jsonl"
    aggressive_file = tmp_path / "governance" / "shadow_aggressive_equities" / "master_control_20260313.jsonl"
    conservative_file = tmp_path / "governance" / "shadow_conservative_equities" / "master_control_20260313.jsonl"

    _write_jsonl(
        bond_file,
        [
            {"timestamp_utc": ts, "symbol": "TLT", "market": {"last_price": 111.00}},
            {"timestamp_utc": ts, "symbol": "TLT", "market": {"last_price": 111.08}},
        ],
    )
    _write_jsonl(
        aggressive_file,
        [
            {"timestamp_utc": ts, "symbol": "TLT", "market": {"last_price": 1111.00}},
            {"timestamp_utc": ts, "symbol": "TLT", "market": {"last_price": 1112.00}},
        ],
    )
    _write_jsonl(
        conservative_file,
        [
            {"timestamp_utc": ts, "symbol": "TLT", "market": {"last_price": 111.10}},
            {"timestamp_utc": ts, "symbol": "TLT", "market": {"last_price": 111.20}},
        ],
    )

    payload, scopes = divergence_bot.build_divergence_payloads(tmp_path, hours=2, max_relative_spread=0.03)

    assert payload["ok"] is False
    assert payload["worst_relative_spread"] > 0.03
    assert scopes["bond_profile"]["ok"] is True
    assert scopes["non_bond_profiles"]["ok"] is True


def test_weekly_retrain_include_targets_preserves_requested_order() -> None:
    targets = [
        "/tmp/core/brain_refinery_v96_credit_spread_rotation_bot.py",
        "/tmp/core/brain_refinery_v95_rates_regime_bond_bot.py",
        "/tmp/core/brain_refinery_v92_macro_rates_curve_regime.py",
    ]

    selected = weekly_retrain._apply_included_bot_ids(
        targets,
        "brain_refinery_v92_macro_rates_curve_regime,brain_refinery_v95_rates_regime_bond_bot",
    )

    assert selected == [
        "/tmp/core/brain_refinery_v92_macro_rates_curve_regime.py",
        "/tmp/core/brain_refinery_v95_rates_regime_bond_bot.py",
    ]


def test_weekly_retrain_resolves_bond_divergence_scope() -> None:
    path, scope = weekly_retrain._resolve_data_divergence_file("bond", "/tmp/fallback.json")
    assert scope == "bond_profile"
    assert path.endswith("data_source_divergence_bond_latest.json")

    path2, scope2 = weekly_retrain._resolve_data_divergence_file("non_bond", "/tmp/fallback.json")
    assert scope2 == "non_bond_profiles"
    assert path2.endswith("data_source_divergence_non_bond_latest.json")


def test_bond_quote_quarantine_clamps_implausible_price() -> None:
    last_price, prev_close = loop._apply_bond_quote_quarantine(
        symbol="TLT",
        last_price=1111.0,
        prev_close=111.0,
        closes=[110.8, 111.1],
    )

    assert round(last_price, 4) == 111.1
    assert round(prev_close, 4) == 111.0
