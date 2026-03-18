import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "nightly_resilience_check.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("nightly_resilience_check", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load nightly_resilience_check module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NightlyResilienceCheckTests(unittest.TestCase):
    def test_prefers_current_home_log_path(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            log_dir = home / "Library" / "Logs" / "schwab_trading_bot"
            log_dir.mkdir(parents=True, exist_ok=True)
            target = log_dir / "shadow_watchdog.out.log"
            target.write_text("ok\n", encoding="utf-8")

            with mock.patch.object(module.Path, "home", return_value=home):
                resolved = module._resolve_watchdog_log()

            self.assertEqual(resolved, target)

    def test_does_not_require_all_sleeves_log_without_wrapper_process(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            module.PROJECT_ROOT = root
            (root / "governance" / "events").mkdir(parents=True, exist_ok=True)
            out_file = root / "nightly.json"
            event_file = root / "events.jsonl"

            home = root / "fake_home"
            log_dir = home / "Library" / "Logs" / "schwab_trading_bot"
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "shadow_watchdog.out.log").write_text("restart none\n", encoding="utf-8")

            counts = {
                "scripts/shadow_watchdog.py": 1,
                "run_shadow_training_loop.py": 8,
                "scripts/run_all_sleeves.py": 0,
            }

            argv = [
                "nightly_resilience_check.py",
                "--out-file",
                str(out_file),
                "--event-file",
                str(event_file),
                "--json",
            ]

            with mock.patch.object(module.Path, "home", return_value=home):
                with mock.patch.object(module, "_pgrep_count", side_effect=lambda pattern: counts.get(pattern, 0)):
                    with mock.patch("sys.argv", argv):
                        rc = module.main()

            self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
