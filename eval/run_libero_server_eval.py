#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.pi0.run_libero_server_eval import main


if __name__ == "__main__":
    raise SystemExit(main())
