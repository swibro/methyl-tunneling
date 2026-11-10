"""Fine-tune mace-off23-l on CP2K labels, keep the best-validation model, benchmark it.

  python models/mace-off23-l/finetune.py [--epochs 100] [--split by-type|random] [--device cpu|cuda]

Run from menv-mace, not menv. Writes models/mace-off23-l/finetune/: split/*.extxyz, mace_data/,
finetune.yaml, runs/, tuned.pt, benchmark/{train,val,test}/results.json. Pass --prepare-only to stop
before training. mace-off23-l is too large to train comfortably on CPU; the script refuses unless --force.
"""

import sys
from pathlib import Path

MODEL_NAME = "mace-off23-l"
DEFAULT_DEVICE = "cpu"  # "cuda" for an NVIDIA GPU; --device overrides
CPU_OK = False
MODEL_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(MODEL_DIR.parent))
import mace_off_common  # noqa: E402

if __name__ == "__main__":
  mace_off_common.run_finetune(MODEL_NAME, MODEL_DIR, DEFAULT_DEVICE, cpu_ok=CPU_OK)
