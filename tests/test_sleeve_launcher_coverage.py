import json
from pathlib import Path

import scripts.run_all_sleeves as run_all_sleeves
import scripts.run_specialized_sleeve_shadow as specialized
from scripts.ops import sleeve_strategy_coverage_guard as coverage_guard


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_all_specialized_profiles_are_registered_and_wrapped() -> None:
    specialized_profiles = set(specialized.SLEEVE_DEFAULTS)
    all_sleeves_profiles = set(run_all_sleeves.SPECIALIZED_SLEEVE_PROFILES)

    assert all_sleeves_profiles == specialized_profiles
    assert [
        profile
        for profile in sorted(specialized_profiles)
        if not (PROJECT_ROOT / "scripts" / f"run_{profile}_shadow.py").exists()
    ] == []


def test_active_collection_sleeves_have_launchers() -> None:
    config = json.loads((PROJECT_ROOT / "config" / "sleeve_strategy_expansion.json").read_text(encoding="utf-8"))
    gaps = []

    for row in config.get("sleeves", []):
        name = str(row.get("name") or "").strip()
        status = str(row.get("runtime_status") or "").strip()
        if status not in coverage_guard.ACTIVE_LAUNCHER_STATUSES:
            continue
        if name in coverage_guard.NON_SPECIALIZED_RUNTIME_SLEEVES:
            continue
        if name not in specialized.SLEEVE_DEFAULTS:
            gaps.append((name, "missing_specialized_defaults"))
        if not (PROJECT_ROOT / "scripts" / f"run_{name}_shadow.py").exists():
            gaps.append((name, "missing_wrapper"))

    assert gaps == []
