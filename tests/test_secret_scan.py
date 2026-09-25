import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.secret_scan as secret_scan


def test_all_repo_files_skips_venv_variants_and_local_secret_files(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(secret_scan, "PROJECT_ROOT", tmp_path)
    (tmp_path / ".venv314" / "lib").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".venv314" / "lib" / "secret.py").write_text("API_KEY=abcdabcdabcdabcd\n", encoding="utf-8")
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / ".env.live.secrets.local").write_text("API_KEY=abcdabcdabcdabcd\n", encoding="utf-8")
    (tmp_path / "scripts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "scripts" / "safe.py").write_text("print('ok')\n", encoding="utf-8")

    files = secret_scan._all_repo_files()

    assert tmp_path / "scripts" / "safe.py" in files
    assert all(".venv314" not in str(path) for path in files)
    assert all(not str(path).endswith(".secrets.local") for path in files)


def test_scan_allowlists_real_secret_placeholders(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(secret_scan, "PROJECT_ROOT", tmp_path)
    path = tmp_path / "config" / ".env.live"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("SCHWAB_API_KEY=YOUR_REAL_KEY\n", encoding="utf-8")

    findings = secret_scan._scan([path], max_bytes=1024)

    assert findings == []


def test_scan_ignores_code_style_secret_references(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(secret_scan, "PROJECT_ROOT", tmp_path)
    path = tmp_path / "scripts" / "code.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "payload = dict(api_key=coinmetrics_api_key)\n"
        "parser.add_argument('--token', default=os.getenv('LIVE_FEED_PHONE_TOKEN', ''))\n",
        encoding="utf-8",
    )

    findings = secret_scan._scan([path], max_bytes=1024)

    assert findings == []


def test_telemetry_redaction_fixture_is_repo_secret_scan_clean() -> None:
    fixture = PROJECT_ROOT / "tests" / "test_telemetry_redaction_canary.py"

    findings = secret_scan._scan([fixture], max_bytes=1_000_000)

    assert findings == []


def test_renamed_environment_is_pruned_but_other_work_is_scanned(tmp_path, monkeypatch):
    monkeypatch.setattr(secret_scan, "PROJECT_ROOT", tmp_path)
    backup = tmp_path / "work" / "upgrade" / "venv-before"
    backup.mkdir(parents=True)
    (backup / "pyvenv.cfg").write_text("home = /python\n")
    (backup / "dependency.py").write_text("TOKEN=abcdefghijklmnopqrst\n")
    authored = tmp_path / "work" / "source.py"
    authored.write_text("TOKEN=abcdefghijklmnopqrst\n")
    paths = secret_scan._all_repo_files()
    assert authored in paths
    assert not any(p.is_relative_to(backup) for p in paths)
    assert len(secret_scan._scan(paths, 1024)) == 1


def test_full_scan_does_not_follow_directory_or_file_links(tmp_path, monkeypatch):
    monkeypatch.setattr(secret_scan, "PROJECT_ROOT", tmp_path)
    (tmp_path / "linked-file").symlink_to("/unavailable-target")
    (tmp_path / "linked-dir").symlink_to("/unavailable-directory", target_is_directory=True)
    assert secret_scan._all_repo_files() == []
