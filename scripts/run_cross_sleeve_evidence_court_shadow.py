import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_specialized_sleeve_shadow import main


if __name__ == "__main__":
    raise SystemExit(main("cross_sleeve_evidence_court"))
