"""Zero-shot benchmark: pretrained uma-s-1p2p1 (task "omc") vs CP2K labels.

  python models/uma-s-1p2p1/zero_shot.py [--labeled work/labeled.extxyz] [--device cpu|cuda]

Writes models/uma-s-1p2p1/zero_shot/{results.json,predictions.extxyz,scan.png}.
"""

import sys
from pathlib import Path

MODEL_NAME = "uma-s-1p2p1"
DEFAULT_DEVICE = "cpu"  # "cuda" for an NVIDIA GPU; --device overrides
MODEL_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(MODEL_DIR.parent))
import common  # noqa: E402

if __name__ == "__main__":
  common.run_zero_shot(MODEL_NAME, MODEL_DIR, DEFAULT_DEVICE)
