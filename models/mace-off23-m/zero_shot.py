"""Zero-shot benchmark: pretrained mace-off23-m vs CP2K labels. Run from menv-mace, not menv.

  python models/mace-off23-m/zero_shot.py [--labeled work/labeled.extxyz] [--device cpu|cuda]

Writes models/mace-off23-m/zero_shot/{results.json,predictions.extxyz,scan.png}.
"""

import sys
from pathlib import Path

MODEL_NAME = "mace-off23-m"
DEFAULT_DEVICE = "cpu"  # "cuda" for an NVIDIA GPU; --device overrides
MODEL_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(MODEL_DIR.parent))
import mace_off_common  # noqa: E402

if __name__ == "__main__":
  mace_off_common.run_zero_shot(MODEL_NAME, MODEL_DIR, DEFAULT_DEVICE)
