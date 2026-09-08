import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_deprecated_symbols_are_absent_from_executable_universe_sources() -> None:
    lifecycle = json.loads((PROJECT_ROOT / "config" / "symbol_lifecycle_v1.json").read_text(encoding="utf-8"))
    deprecated = set(lifecycle["renamed_symbols"])
    source_paths = [
        PROJECT_ROOT / "config" / "sleeve_strategy_expansion.json",
        PROJECT_ROOT / "scripts" / "run_specialized_sleeve_shadow.py",
        PROJECT_ROOT / "scripts" / "run_shadow_training_loop.py",
        PROJECT_ROOT / "scripts" / "run_long_term_core_etf_shadow.py",
        PROJECT_ROOT / "scripts" / "ops" / "sleeve_ticker_universe_expansion.py",
    ]

    violations = {
        str(path.relative_to(PROJECT_ROOT)): sorted(symbol for symbol in deprecated if symbol in path.read_text(encoding="utf-8"))
        for path in source_paths
    }

    assert not {path: symbols for path, symbols in violations.items() if symbols}


def test_candidate_fingerprint_covers_symbol_and_canary_policies() -> None:
    policy = json.loads((PROJECT_ROOT / "config" / "production_excellence_v1.json").read_text(encoding="utf-8"))
    scopes = policy["candidate"]["scope_globs"]

    assert "config/sleeve_strategy_expansion.json" in scopes["strategy"]
    assert "config/sleeve_strategy_expansion.json" in scopes["data"]
    assert "config/live_canary_micro_policy_v1.json" in scopes["execution"]
    assert "config/live_canary_micro_policy_v1.json" in scopes["risk"]
    assert "config/symbol_lifecycle_v1.json" in scopes["strategy"]
