"""Fine-tune uma-s-1p2p1 (task "omc") on CP2K labels, keep the final checkpoint, benchmark it.

  python models/uma-s-1p2p1/finetune.py [--epochs 100] [--split by-type|random] [--device cpu|cuda]

Writes models/uma-s-1p2p1/finetune/: split/*.extxyz, lmdb/, finetune.yaml + data/, tuned.pt,
benchmark/{train,val,test}/results.json. Pass --prepare-only to stop before training.
"""

import sys
from pathlib import Path

MODEL_NAME = "uma-s-1p2p1"
DEFAULT_DEVICE = "cpu"  # "cuda" for an NVIDIA GPU; --device overrides
CPU_OK = True
MODEL_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(MODEL_DIR.parent))
import common  # noqa: E402

if __name__ == "__main__":
  common.run_finetune(MODEL_NAME, MODEL_DIR, DEFAULT_DEVICE, cpu_ok=CPU_OK)
