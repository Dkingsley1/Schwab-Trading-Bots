import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SECRET_PATTERNS = [
    re.compile(r"(?i)api[_-]?key\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}"),
    re.compile(r"(?i)secret\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}"),
    re.compile(r"(?i)token\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}"),
    re.compile(r"(?i)authorization\s*[:=]\s*['\"]?bearer\s+[A-Za-z0-9_\-.]{16,}"),
]
ALLOWLIST_SNIPPETS = (
    "YOUR_KEY_HERE",
    "YOUR_SECRET_HERE",
    "YOUR_REAL_KEY",
    "YOUR_REAL_SECRET",
    "YOUR_REAL_PASSWORD",
    "YOUR_REAL_TOKEN",
    "YOUR_REAL_LOGIN",
    "YOUR_REAL_CLIENT_ID",
    "os.getenv(",
    "${",
    "example",
    "REDACTED",
)
SKIP_DIRS = {
    ".git",
    ".venv",
    ".venv312",
    "__pycache__",
    "models",
    "exports",
    "governance",
    "logs",
    "decisions",
    "decision_explanations",
    "data",
}


def _should_skip_repo_file(path: Path) -> bool:
    if any(part in SKIP_DIRS or part.startswith(".venv") for part in path.parts):
        return True
    name = path.name.lower()
    if name.endswith(".secrets.local"):
        return True
    return False


def _is_probable_code_reference(line: str) -> bool:
    text = str(line or "")
    if "write_text(" in text or "parser.add_argument(" in text or "os.getenv(" in text:
        return True
    rhs = ""
    if "=" in text:
        rhs = text.split("=", 1)[1].strip()
    elif ":" in text:
        rhs = text.split(":", 1)[1].strip()
    if any(token in rhs for token in ("(", ")", ".")):
        return True
    if re.search(r"\b[a-z_][a-z0-9_]*_(?:api_key|secret|token|password|login|client_id)\b", rhs):
        return True
    return False


def _is_text(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            chunk = f.read(2048)
        return b"\x00" not in chunk
    except Exception:
        return False


def _staged_files() -> list[Path]:
    proc = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        cwd=str(PROJECT_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    out = []
    for line in (proc.stdout or "").splitlines():
        p = PROJECT_ROOT / line.strip()
        if p.exists() and p.is_file():
            out.append(p)
    return out


def _all_repo_files() -> list[Path]:
    out = []
    for p in PROJECT_ROOT.rglob("*"):
        if not p.is_file():
            continue
        if _should_skip_repo_file(p):
            continue
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".npz", ".sqlite", ".sqlite3", ".db"}:
            continue
        out.append(p)
    return out


def _scan(paths: list[Path], max_bytes: int) -> list[dict]:
    findings = []
    for path in paths:
        try:
            if path.stat().st_size > max_bytes:
                continue
            if not _is_text(path):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        for i, line in enumerate(text.splitlines(), start=1):
            if any(a in line for a in ALLOWLIST_SNIPPETS):
                continue
            if _is_probable_code_reference(line):
                continue
            for patt in SECRET_PATTERNS:
                if patt.search(line):
                    findings.append(
                        {
                            "file": str(path.relative_to(PROJECT_ROOT)),
                            "line": i,
                            "pattern": patt.pattern,
                            "snippet": line[:180],
                        }
                    )
                    break
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Lightweight secret scanner.")
    parser.add_argument("--staged", action="store_true", help="Scan staged files only")
    parser.add_argument("--max-bytes", type=int, default=1_000_000)
    parser.add_argument("--out", default=str(PROJECT_ROOT / "governance" / "health" / "secret_scan_latest.json"))
    args = parser.parse_args()

    paths = _staged_files() if args.staged else _all_repo_files()
    findings = _scan(paths, args.max_bytes)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scanned_files": len(paths),
        "findings_count": len(findings),
        "findings": findings[:200],
        "mode": "staged" if args.staged else "full",
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    print(json.dumps({"findings_count": len(findings), "mode": payload["mode"]}, ensure_ascii=True))
    return 2 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
