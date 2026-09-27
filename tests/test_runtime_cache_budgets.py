import scripts.run_shadow_training_loop as loop
import pytest


def setup_budget(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    monkeypatch.setattr(loop, "PROJECT_ROOT_PATH", tmp_path)
    monkeypatch.setattr(loop, "_RUNTIME_CACHE_BUDGET_STATE", {})
    return tmp_path / "config/.env.memory_efficiency_override"


def test_running_cache_shrinks_without_waiting_for_next_insert(tmp_path, monkeypatch):
    path = setup_budget(tmp_path, monkeypatch)
    key = "RUNTIME_FEATURE_CACHE_MAX_ENTRIES"
    cache = {str(i): object() for i in range(100)}
    path.write_text(f"{key}=32\n")
    limits = loop._refresh_runtime_cache_budgets({key: cache}, {key: 192})
    assert limits[key] == 32
    assert list(cache) == [str(i) for i in range(68, 100)]
    path.write_text(f"{key}=8\n")
    loop._RUNTIME_CACHE_BUDGET_STATE["checked_at"] -= 6
    assert loop._refresh_runtime_cache_budgets({key: cache}, {key: 192})[key] == 8
    assert len(cache) == 8


def test_memory_budget_does_not_expand_bootstrap_cap_or_apply_other_controls(
    tmp_path, monkeypatch
):
    path = setup_budget(tmp_path, monkeypatch)
    key = "RUNTIME_FEATURE_CACHE_MAX_ENTRIES"
    path.write_text(f"{key}=1000\nLIVE_EXECUTION_ALLOWED=1\n")
    assert loop._refresh_runtime_cache_budgets({key: {}}, {key: 16}) == {key: 16}


def test_both_memory_owners_constrain_the_budget(tmp_path, monkeypatch):
    path = setup_budget(tmp_path, monkeypatch)
    key = "RUNTIME_SLOW_BOT_CACHE_MAX_SYMBOLS"
    path.write_text(f"{key}=32\n")
    (path.parent / ".env.swap_pressure_override").write_text(f"{key}=8\n")
    assert loop._refresh_runtime_cache_budgets({key: {}}, {key: 48})[key] == 8


@pytest.mark.parametrize("invalid", ["invalid", "-1"])
def test_invalid_budget_preserves_last_tighter_cap(tmp_path, monkeypatch, invalid):
    path = setup_budget(tmp_path, monkeypatch)
    key = "RUNTIME_FEATURE_CACHE_MAX_ENTRIES"
    path.write_text(f"{key}=8\n")
    loop._refresh_runtime_cache_budgets({key: {}}, {key: 192})
    path.write_text(f"{key}={invalid}\n")
    loop._RUNTIME_CACHE_BUDGET_STATE["checked_at"] -= 6
    assert loop._refresh_runtime_cache_budgets({key: {}}, {key: 192})[key] == 8


def test_pressure_can_bound_legacy_unbounded_cache(tmp_path, monkeypatch):
    path = setup_budget(tmp_path, monkeypatch)
    key = "RUNTIME_FEATURE_CACHE_MAX_ENTRIES"
    path.write_text(f"{key}=4\n")
    cache = dict.fromkeys(range(10))
    assert loop._refresh_runtime_cache_budgets({key: cache}, {key: 0})[key] == 4
    assert len(cache) == 4


def test_changed_bootstrap_budget_is_not_hidden_by_refresh_interval(
    tmp_path, monkeypatch
):
    setup_budget(tmp_path, monkeypatch)
    key = "RUNTIME_FEATURE_CACHE_MAX_ENTRIES"
    loop._refresh_runtime_cache_budgets({key: {}}, {key: 192})
    assert loop._refresh_runtime_cache_budgets({key: {}}, {key: 8}) == {key: 8}
