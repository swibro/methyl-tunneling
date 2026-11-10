"""Benchmark the fine-tuned mace-off23-m model (default models/mace-off23-m/finetune/tuned.pt) on CP2K labels.

  python models/mace-off23-m/benchmark_tuned.py [--labeled work/labeled.extxyz] [--checkpoint PATH] [--device cpu|cuda]

Run from menv-mace, not menv. Use after labeling more frames; same report format as zero_shot.py.
"""

import sys
from pathlib import Path

MODEL_NAME = "mace-off23-m"
DEFAULT_DEVICE = "cpu"  # "cuda" for an NVIDIA GPU; --device overrides
MODEL_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(MODEL_DIR.parent))
import mace_off_common  # noqa: E402

if __name__ == "__main__":
  mace_off_common.run_benchmark_tuned(MODEL_NAME, MODEL_DIR, DEFAULT_DEVICE)
