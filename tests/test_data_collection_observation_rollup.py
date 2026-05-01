import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import data_collection_observation_rollup as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _registry(bot_id: str) -> dict:
    return {
        "summary": {},
        "sub_bots": [
            {
                "bot_id": bot_id,
                "bot_role": "signal_sub_bot",
                "active": True,
                "lifecycle_state": "data_collection_only",
                "data_collection_active": True,
                "data_collection_started_utc": "2026-04-20T00:00:00+00:00",
                "data_collection_observations": 0,
                "minimum_training_observations": 2,
                "minimum_data_collection_days": 1,
                "training_excluded": True,
                "exclude_from_training": True,
            }
        ],
    }


def test_observation_rollup_bootstraps_and_updates_registry(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    bot_id = "brain_refinery_v167_intraday_opening_range_momentum_burst"
    registry_path = project_root / "master_bot_registry.json"
    state_path = project_root / "governance" / "health" / "state.json"
    _write_json(registry_path, _registry(bot_id))
    decision_file = project_root / "decision_explanations" / "shadow_intraday_aggressive_equities" / "decision_explanations_20260430.jsonl"
    decision_file.parent.mkdir(parents=True, exist_ok=True)
    decision_file.write_text(
        "\n".join(
            [
                json.dumps({"status": "DATA_ONLY_BLOCKED", "reasons": [f"bot_id={bot_id}"]}),
                json.dumps({"status": "SHADOW_ONLY", "metadata": {"bot_id": bot_id}}),
                json.dumps({"status": "SHADOW_ONLY", "metadata": {"bot_id": "brain_refinery_v1_other"}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = src.build_payload(
        project_root=project_root,
        registry_path=registry_path,
        state_path=state_path,
        days=1,
        bootstrap_tail_lines=20,
        apply=True,
    )

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    row = registry["sub_bots"][0]

    assert payload["mode"] == "bootstrap_tail"
    assert payload["bots_with_observations"] == 1
    assert payload["total_observations"] == 2
    assert row["data_collection_observations"] == 2
    assert row["collected_observation_count"] == 2
    assert row["data_collection_training_ready"] is True
    assert row["training_excluded"] is False
    assert registry["summary"]["data_collection_training_ready_bots"] == 1


def test_observation_rollup_counts_only_new_lines_after_bootstrap(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    bot_id = "brain_refinery_v171_intraday_relative_volume_surge_chaser"
    registry_path = project_root / "master_bot_registry.json"
    state_path = project_root / "governance" / "health" / "state.json"
    _write_json(registry_path, _registry(bot_id))
    decision_file = project_root / "decision_explanations" / "shadow_intraday_aggressive_equities" / "decision_explanations_20260430.jsonl"
    decision_file.parent.mkdir(parents=True, exist_ok=True)
    decision_file.write_text(json.dumps({"status": "DATA_ONLY_BLOCKED", "reasons": [f"bot_id={bot_id}"]}) + "\n", encoding="utf-8")

    src.build_payload(
        project_root=project_root,
        registry_path=registry_path,
        state_path=state_path,
        days=1,
        bootstrap_tail_lines=20,
        apply=True,
    )
    with decision_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"status": "SHADOW_ONLY", "metadata": {"bot_id": bot_id}}) + "\n")

    payload = src.build_payload(
        project_root=project_root,
        registry_path=registry_path,
        state_path=state_path,
        days=1,
        bootstrap_tail_lines=20,
        apply=True,
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))

    assert payload["mode"] == "incremental"
    assert payload["new_rows_counted"] == 1
    assert registry["sub_bots"][0]["data_collection_observations"] == 2
