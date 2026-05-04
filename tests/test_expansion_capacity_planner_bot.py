import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops import expansion_capacity_planner_bot as src


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _seed_registry(project_root: Path) -> None:
    _write_json(
        project_root / "master_bot_registry.json",
        {
            "sub_bots": [
                {
                    "bot_id": "brain_refinery_v672_dcc_garch_correlation_bot",
                    "bot_role": "signal_sub_bot",
                    "active": True,
                    "lifecycle_state": "data_collection_only",
                    "training_excluded": True,
                    "sleeve_profile": "tail_dependency_risk",
                },
                {
                    "bot_id": "brain_refinery_v673_evt_peaks_over_threshold_tail_bot",
                    "bot_role": "signal_sub_bot",
                    "active": True,
                    "lifecycle_state": "data_collection_only",
                    "exclude_from_training": True,
                    "sleeve_profile": "tail_dependency_risk",
                },
            ]
        },
    )


def _seed_health(
    project_root: Path,
    *,
    halt: bool = False,
    swap_tier: str = "normal",
    swap_gb: float = 2.0,
    runtime_status: str = "ready",
    memory_status: str = "ready",
    storage_status: str = "ready",
    admission_blocking: int = 0,
) -> None:
    health = project_root / "governance" / "health"
    _write_json(health / "global_killswitch_latest.json", {"global_halt_active": halt})
    _write_json(health / "swap_pressure_governor_latest.json", {"swap_pressure": {"tier": swap_tier, "swap_used_gb": swap_gb}})
    _write_json(health / "runtime_throttle_control_latest.json", {"overall_status": runtime_status, "host_saturation_score": 22.0})
    _write_json(health / "memory_efficiency_control_latest.json", {"overall_status": memory_status})
    _write_json(health / "ingestion_storage_control_latest.json", {"overall_status": storage_status, "pressure_index": 0.02})
    _write_json(health / "data_collection_storage_guard_latest.json", {"overall_status": storage_status})
    _write_json(health / "new_bot_admission_guard_latest.json", {"candidate_bot_count": 3, "blocking_candidate_count": admission_blocking})
    _write_json(health / "runtime_gate_dashboard_latest.json", {"overall_status": "ready"})


def test_expansion_capacity_allows_collection_only_wave_when_pressure_is_ready(tmp_path: Path) -> None:
    _seed_registry(tmp_path)
    _seed_health(tmp_path)

    payload = src.build_payload(tmp_path, requested_wave_size=20)

    assert payload["overall_status"] == "ready"
    assert payload["capacity_contract"]["recommended_wave_size_now"] == 20
    assert payload["capacity_contract"]["rollout_mode"] == "collection_only_wave_allowed"
    assert payload["capacity_contract"]["next_bot_id_range"]["start"] == "brain_refinery_v674"
    assert "brain_refinery_v674_expansion_capacity_planner_bot" in payload["support_infrabots"]
    assert payload["growth_invariants"][0] == "new bots enter as data_collection_only"


def test_expansion_capacity_blocks_new_runtime_when_halt_or_swap_pressure_active(tmp_path: Path) -> None:
    _seed_registry(tmp_path)
    _seed_health(tmp_path, halt=True, swap_tier="pause_research", swap_gb=21.0, admission_blocking=4)

    payload = src.build_payload(tmp_path, requested_wave_size=20)

    assert payload["overall_status"] == "blocked"
    assert payload["capacity_contract"]["max_new_collectors_now"] == 0
    assert payload["capacity_contract"]["recommended_wave_size_now"] == 0
    assert payload["capacity_contract"]["rollout_mode"] == "protect_live_no_new_runtime_loops"
    assert "global_halt_active" in payload["pressure_snapshot"]["blocking_reasons"]
    assert "clear new-bot admission contracts before allowing any of the expanded roster into training" in payload["recommended_actions"]
