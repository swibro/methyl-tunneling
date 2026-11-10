"""Fine-tune uma-m-1p1 (task "omc") on CP2K labels, keep the final checkpoint, benchmark it.

  python models/uma-m-1p1/finetune.py [--epochs 100] [--split by-type|random] [--device cpu|cuda]

Writes models/uma-m-1p1/finetune/: split/*.extxyz, lmdb/, finetune.yaml + data/, tuned.pt,
benchmark/{train,val,test}/results.json. Pass --prepare-only to stop before training.
uma-m is too large to train on CPU; the script refuses unless --force.
"""

import sys
from pathlib import Path

MODEL_NAME = "uma-m-1p1"
DEFAULT_DEVICE = "cpu"  # "cuda" for an NVIDIA GPU; --device overrides
CPU_OK = False
MODEL_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(MODEL_DIR.parent))
import common  # noqa: E402

if __name__ == "__main__":
  common.run_finetune(MODEL_NAME, MODEL_DIR, DEFAULT_DEVICE, cpu_ok=CPU_OK)
