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
