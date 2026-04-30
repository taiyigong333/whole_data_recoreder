from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_LEROBOT_SRC = PROJECT_ROOT / "lerobot-0.3.3" / "src"

if LOCAL_LEROBOT_SRC.is_dir() and str(LOCAL_LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_LEROBOT_SRC))
