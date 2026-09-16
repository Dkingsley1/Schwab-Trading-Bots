import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_core_modules_do_not_bypass_process_wide_mlx_guard() -> None:
    offenders = []
    for path in sorted((PROJECT_ROOT / "core").glob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "import mlx.core" in source or "import mlx.nn" in source:
            offenders.append(path.name)

    assert offenders == []


def test_disabled_mlx_is_quarantined_without_native_retry() -> None:
    code = (
        "import json,sys; "
        f"sys.path.insert(0, {str(PROJECT_ROOT)!r}); "
        "import core.indicator_bot_common as indicator; "
        "import core.advanced_quant_models as quant; "
        "from core.mlx_runtime_guard import mlx_available, mlx_modules; "
        "mx,nn,optim,error=mlx_modules(); "
        "print(json.dumps({'available':mlx_available(),"
        "'indicator_available':indicator._MLX_AVAILABLE,"
        "'quant_mx_is_none':quant.mx is None,"
        "'error_type':type(error).__name__}))"
    )
    env = os.environ.copy()
    env["BOT_MLX_DISABLE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload == {
        "available": False,
        "indicator_available": False,
        "quant_mx_is_none": True,
        "error_type": "MLXUnavailableError",
    }


def test_disabled_mlx_does_not_break_cpu_plotting_or_module_introspection():
    code = f"""
import sys
sys.path.insert(0, {str(PROJECT_ROOT)!r})
from core.mlx_runtime_guard import require_mlx, MLXUnavailableError
sys.path.insert(0, {str(PROJECT_ROOT / 'core')!r})
from core import brain_refinery_v10_seasonal, brain_refinery_v12_news_shocks
assert brain_refinery_v10_seasonal.simulate_seasonal(0).shape == (0,)
for bot in (brain_refinery_v10_seasonal, brain_refinery_v12_news_shocks):
    try:
        bot.TradingBrain(5)
    except MLXUnavailableError:
        pass
    else:
        raise AssertionError('GPU model construction was not rejected')
module = sys.modules['mlx.core']
assert getattr(module, '__file__', None) is None
assert not hasattr(module, 'array')
try:
    require_mlx()
except MLXUnavailableError:
    pass
else:
    raise AssertionError('GPU use was not rejected')
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1])
fig.canvas.draw()
plt.close(fig)
import torch
assert torch.tensor([1, 2]).sum().item() == 3
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], cwd=PROJECT_ROOT,
        env={**os.environ, "BOT_MLX_DISABLE": "1", "MPLBACKEND": "Agg"},
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
