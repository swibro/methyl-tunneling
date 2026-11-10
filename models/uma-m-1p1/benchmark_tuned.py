"""Benchmark the fine-tuned uma-m-1p1 checkpoint (default models/uma-m-1p1/finetune/tuned.pt) on CP2K labels.

  python models/uma-m-1p1/benchmark_tuned.py [--labeled work/labeled.extxyz] [--checkpoint PATH] [--device cpu|cuda]

Use after labeling more frames; same report format as zero_shot.py.
"""

import sys
from pathlib import Path

MODEL_NAME = "uma-m-1p1"
DEFAULT_DEVICE = "cpu"  # "cuda" for an NVIDIA GPU; --device overrides
MODEL_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(MODEL_DIR.parent))
import common  # noqa: E402

if __name__ == "__main__":
  common.run_benchmark_tuned(MODEL_NAME, MODEL_DIR, DEFAULT_DEVICE)
