import importlib.util
from pathlib import Path


MODULE_PATH = Path("/Users/dankingsley/PycharmProjects/schwab_trading_bot/scripts/export_logs_to_csv.py")
spec = importlib.util.spec_from_file_location("export_logs_to_csv_test", MODULE_PATH)
export_logs_to_csv = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(export_logs_to_csv)


def test_publish_latest_alias_prefers_symlink(tmp_path: Path) -> None:
    out_dir = tmp_path / "csv"
    out_dir.mkdir()
    named_file = out_dir / "decision_explanations_20260324.csv"
    named_file.write_text("metric,value\nx,1\n", encoding="utf-8")

    export_logs_to_csv._publish_latest_alias(out_dir, named_file, "latest_decision_explanations.csv")

    alias = out_dir / "latest_decision_explanations.csv"
    assert alias.is_symlink()
    assert alias.resolve() == named_file.resolve()


def test_publish_latest_alias_is_idempotent_when_target_matches(tmp_path: Path) -> None:
    out_dir = tmp_path / "csv"
    out_dir.mkdir()
    named_file = out_dir / "master_control_20260324.csv"
    named_file.write_text("metric,value\nx,1\n", encoding="utf-8")

    export_logs_to_csv._publish_latest_alias(out_dir, named_file, "latest_master_control.csv")
    alias = out_dir / "latest_master_control.csv"
    first_target = Path(alias.readlink())

    export_logs_to_csv._publish_latest_alias(out_dir, named_file, "latest_master_control.csv")

    assert alias.is_symlink()
    assert Path(alias.readlink()) == first_target
